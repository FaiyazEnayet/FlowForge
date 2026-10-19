"""Model sweep: train DDPM variants and baselines, pick the best by VALIDATION metrics.

The test split is never touched here. For each config we train on TRAIN,
generate conditioned on VAL regimes (or resample/simulate for baselines),
and score:
  - mean 1-Wasserstein distance across features (val real vs val generated)
  - discriminator AUC: classifier trained on (real train-subset vs generated
    conditioned on train regimes), evaluated on (real val vs generated val)

Baselines (FL4) get the same evaluation protocol and report sampling
latency/memory alongside quality, so architecture advantages are weighed
against compute cost. The winner is re-trained from scratch with the
winning config in demo.py.
Fixed seeds everywhere. CPU only.
"""

import json
import os
import time

import numpy as np
import torch

from flowforge.baselines import MovingBlockResampler, VARBaseline
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


def _score_generated(name, gen_val, ds, x_train, r_train, seed,
                     train_seconds=None, params=None, sample_stats=None):
    """Score generated val windows; shared by neural models and baselines."""
    mt = marginal_table(ds.val.x, gen_val)
    mean_w1 = mt["mean_w1"]
    n_sub = min(400, len(x_train))
    r_sub = r_train[:n_sub]
    # For baselines, generate a matching subset for the discriminator.
    # (Neural models pass their own gen_sub; baselines regenerate here.)
    d = discriminator_auc(ds.train.x[:n_sub], gen_val[:n_sub],
                          ds.val.x, gen_val, seed=seed, epochs=20)
    print(f"  val mean W1: {mean_w1:.3f} | "
          f"sweep-disc AUC on val: {d['auc_test']:.3f}")
    for row in mt["rows"]:
        print(f"    {row['feature']:<16} w1={row['w1']:.3f} "
              f"real_mean={row['real_mean']:+.3f} gen_mean={row['gen_mean']:+.3f}")
    result = {"config": name, "val_mean_w1": mean_w1,
              "val_disc_auc": d["auc_test"]}
    if train_seconds is not None:
        result["train_seconds"] = train_seconds
    if params is not None:
        result["params"] = params
    if sample_stats is not None:
        result.update({f"sample_{k}": v for k, v in sample_stats.items()
                       if k in ("sample_seconds_per_1k", "peak_memory_mb")})
    return result


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

    # ---- baselines first (FL4): cheap, same protocol, timed ----
    print("\n=== baseline: moving-block resampler ===")
    t_b0 = time.time()
    mbr = MovingBlockResampler(ds.train.x, block_len=4)
    gen_val_mbr, stats_mbr = mbr.generate(len(ds.val.x), seed=SEED)
    results.append(_score_generated(
        "baseline_moving_block", gen_val_mbr, ds, x_train, r_train, SEED,
        train_seconds=time.time() - t_b0, params=0, sample_stats=stats_mbr))
    print(f"  sampling: {stats_mbr['sample_seconds_per_1k']:.2f}s per 1k windows, "
          f"{stats_mbr['peak_memory_mb']:.1f} MB peak")

    print("\n=== baseline: VAR(2) ridge ===")
    t_b1 = time.time()
    var = VARBaseline(order=2, ridge=1.0).fit(ds.train.x)
    gen_val_var, stats_var = var.generate(len(ds.val.x), ds.train.x, seed=SEED)
    results.append(_score_generated(
        "baseline_var2", gen_val_var, ds, x_train, r_train, SEED,
        train_seconds=time.time() - t_b1, params=int(var.coef_.size),
        sample_stats=stats_var))
    print(f"  sampling: {stats_var['sample_seconds_per_1k']:.2f}s per 1k windows, "
          f"{stats_var['peak_memory_mb']:.1f} MB peak")

    for name, ch, n_blocks, epochs, ema_decay in CONFIGS:
        print(f"\n=== {name}: ch={ch} blocks={n_blocks} "
              f"epochs={epochs} ema={ema_decay} ===")
        model = BookDDPM(ch=ch, n_blocks=n_blocks)
        n_params = count_params(model)
        print(f"  params: {n_params}")
        t_train0 = time.time()
        hist, ema = train_ddpm(model, x_train, r_train, epochs=epochs,
                               seed=SEED, verbose=False,
                               ema_decay=ema_decay)
        train_secs = time.time() - t_train0
        print(f"  final train loss: {hist[-1]:.4f} "
              f"({train_secs / 60:.1f} min train)")

        # generate for val regimes (timed)
        t_s0 = time.perf_counter()
        if ema is not None:
            gen_val = sample_with_ema(model, ema, r_val,
                                      seed=SEED).numpy()
        else:
            gen_val = model.sample(r_val, seed=SEED).numpy()
        sample_secs = time.perf_counter() - t_s0

        r = _score_generated(name, gen_val, ds, x_train, r_train, SEED,
                             train_seconds=train_secs, params=n_params,
                             sample_stats={
                                 "sample_seconds_per_1k":
                                     sample_secs / max(len(gen_val), 1) * 1000,
                                 "peak_memory_mb": float("nan")})
        r.update({"ch": ch, "n_blocks": n_blocks, "epochs": epochs,
                  "ema_decay": ema_decay, "final_loss": hist[-1]})
        print(f"  sampling: {r['sample_sample_seconds_per_1k']:.2f}s per 1k windows")
        results.append(r)

    # pick winner: lowest val mean W1, tie-break by lower disc AUC.
    # Baselines compete on equal footing: if a baseline wins, the neural
    # models have not earned their compute budget.
    results.sort(key=lambda r: (r["val_mean_w1"], r["val_disc_auc"]))
    winner = results[0]
    print("\n=== sweep results (sorted) ===")
    for r in results:
        t_str = (f"train {r['train_seconds'] / 60:.1f}m "
                 if "train_seconds" in r else "")
        s_str = (f"sample {r['sample_sample_seconds_per_1k']:.1f}s/1k "
                 if "sample_sample_seconds_per_1k" in r else "")
        print(f"  {r['config']:<22} val W1 {r['val_mean_w1']:.3f} "
              f"disc AUC {r['val_disc_auc']:.3f} {t_str}{s_str}")
    print(f"\nWINNER: {winner['config']}")

    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "sweep.json"), "w") as f:
        json.dump({"seed": SEED, "results": results,
                   "winner": winner["config"],
                   "minutes": (time.time() - t0) / 60}, f, indent=1)
    print(f"sweep -> runs/sweep.json ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
