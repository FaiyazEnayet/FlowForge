"""Model sweep: train DDPM variants, pick the best by VALIDATION metrics.

The test split is never touched here. For each config we train on TRAIN,
generate conditioned on VAL regimes, and score:
  - mean 1-Wasserstein distance across features (val real vs val generated)
  - discriminator AUC: classifier trained on (real train-subset vs generated
    conditioned on train regimes), evaluated on (real val vs generated val)

The winner is re-trained from scratch with the winning config in demo.py.
Fixed seeds everywhere. CPU only.
"""

import json
import os
import time

import numpy as np
import torch

from flowforge.data import load_dataset
from flowforge.evaluate import (discriminator_auc, marginal_table,
                                wasserstein_1d)
from flowforge.models import (BookDDPM, count_params, sample_with_ema,
                              set_seed, train_ddpm)

SEED = 7
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "book_windows.npz")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")

CONFIGS = [
    # name, ch, n_blocks, epochs, ema_decay
    ("A_baseline_80ep", 96, 3, 80, None),
    ("B_long_300ep", 96, 3, 300, None),
    ("C_long_ema", 96, 3, 300, 0.999),
    ("D_big_long_ema", 128, 4, 300, 0.999),
]


def main():
    t0 = time.time()
    set_seed(SEED)
    ds = load_dataset(os.path.abspath(DATA))
    x_train = torch.from_numpy(ds.train.x)
    r_train = torch.from_numpy(ds.train.regime)
    r_val = torch.from_numpy(ds.val.regime)
    print(f"train {len(x_train)} / val {len(ds.val.x)} / "
          f"test {len(ds.test.x)} windows (test untouched)")

    results = []
    for name, ch, n_blocks, epochs, ema_decay in CONFIGS:
        print(f"\n=== {name}: ch={ch} blocks={n_blocks} "
              f"epochs={epochs} ema={ema_decay} ===")
        model = BookDDPM(ch=ch, n_blocks=n_blocks)
        print(f"  params: {count_params(model)}")
        hist, ema = train_ddpm(model, x_train, r_train, epochs=epochs,
                               seed=SEED, verbose=False,
                               ema_decay=ema_decay)
        print(f"  final train loss: {hist[-1]:.4f}")

        # generate for val regimes
        if ema is not None:
            gen_val = sample_with_ema(model, ema, r_val,
                                      seed=SEED).numpy()
        else:
            gen_val = model.sample(r_val, seed=SEED).numpy()

        mt = marginal_table(ds.val.x, gen_val)
        mean_w1 = mt["mean_w1"]

        # discriminator: train on train-split windows, test on val
        n_sub = min(400, len(x_train))
        r_sub = r_train[:n_sub]
        if ema is not None:
            gen_sub = sample_with_ema(model, ema, r_sub,
                                      seed=SEED + 11).numpy()
        else:
            gen_sub = model.sample(r_sub, seed=SEED + 11).numpy()
        d = discriminator_auc(ds.train.x[:n_sub], gen_sub,
                              ds.val.x, gen_val, seed=SEED, epochs=20)
        print(f"  val mean W1: {mean_w1:.3f} | "
              f"sweep-disc AUC on val: {d['auc_test']:.3f}")
        # per-feature W1 for the gap analysis (imbalance mean shift etc.)
        for row in mt["rows"]:
            print(f"    {row['feature']:<16} w1={row['w1']:.3f} "
                  f"real_mean={row['real_mean']:+.3f} gen_mean={row['gen_mean']:+.3f}")
        results.append({"config": name, "ch": ch, "n_blocks": n_blocks,
                        "epochs": epochs, "ema_decay": ema_decay,
                        "val_mean_w1": mean_w1,
                        "val_disc_auc": d["auc_test"],
                        "final_loss": hist[-1],
                        "params": count_params(model)})

    # pick winner: lowest val mean W1, tie-break by lower disc AUC
    results.sort(key=lambda r: (r["val_mean_w1"], r["val_disc_auc"]))
    winner = results[0]
    print("\n=== sweep results (sorted) ===")
    for r in results:
        print(f"  {r['config']:<16} val W1 {r['val_mean_w1']:.3f} "
              f"disc AUC {r['val_disc_auc']:.3f}")
    print(f"\nWINNER: {winner['config']}")

    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "sweep.json"), "w") as f:
        json.dump({"seed": SEED, "results": results,
                   "winner": winner["config"],
                   "minutes": (time.time() - t0) / 60}, f, indent=1)
    print(f"sweep -> runs/sweep.json ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
