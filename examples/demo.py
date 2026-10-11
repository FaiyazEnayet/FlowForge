"""End to end: train on real windows, generate, evaluate honestly.

1. Loads the book-state windows (time-split: train/val/test, no overlap).
2. Trains the conditional VAE baseline and the DDPM on TRAIN only.
3. Generates windows conditioned on VAL and TEST regime features.
4. Prints the full real-vs-generated report on TEST (never seen by either
   model) and saves metrics to runs/metrics.json.

Fixed seeds everywhere. CPU only.
"""

import json
import os
import time

import numpy as np
import torch

from flowforge.data import load_dataset
from flowforge.evaluate import full_report, print_report
from flowforge.models import (BookDDPM, CondVAE, count_params,
                              sample_with_ema, set_seed, train_ddpm,
                              train_vae)

SEED = 7
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "book_windows.npz")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")

# DDPM config: winner of examples/sweep.py on the validation split.
# Override with FF_DDPM_CH / FF_DDPM_BLOCKS / FF_DDPM_EPOCHS / FF_DDPM_EMA.
DDPM_CH = int(os.environ.get("FF_DDPM_CH", "96"))
DDPM_BLOCKS = int(os.environ.get("FF_DDPM_BLOCKS", "3"))
DDPM_EPOCHS = int(os.environ.get("FF_DDPM_EPOCHS", "300"))
_ema_raw = os.environ.get("FF_DDPM_EMA", "")
DDPM_EMA = float(_ema_raw) if _ema_raw else None


def main():
    t0 = time.time()
    set_seed(SEED)
    ds = load_dataset(os.path.abspath(DATA))
    print(f"data: {ds.n_chunks} chunks, {ds.n_seconds}s -> "
          f"{len(ds.train.x)}/{len(ds.val.x)}/{len(ds.test.x)} "
          f"train/val/test windows")

    x_train = torch.from_numpy(ds.train.x)
    r_train = torch.from_numpy(ds.train.regime)

    print("\n--- training conditional VAE baseline "
          f"({count_params(CondVAE())} params) ---")
    vae = CondVAE()
    vae_hist = train_vae(vae, x_train, r_train, epochs=25, seed=SEED)

    print(f"\n--- training DDPM ({count_params(BookDDPM(ch=DDPM_CH, n_blocks=DDPM_BLOCKS))} params, "
          f"ch={DDPM_CH} blocks={DDPM_BLOCKS} epochs={DDPM_EPOCHS} ema={DDPM_EMA}) ---")
    ddpm = BookDDPM(ch=DDPM_CH, n_blocks=DDPM_BLOCKS)
    ddpm_hist, ddpm_ema = train_ddpm(ddpm, x_train, r_train,
                                     epochs=DDPM_EPOCHS, seed=SEED,
                                     ema_decay=DDPM_EMA)

    print("\n--- generating ---")
    vae.eval()
    r_val = torch.from_numpy(ds.val.regime)
    r_test = torch.from_numpy(ds.test.regime)
    with torch.no_grad():
        gen_val_vae = vae.generate(r_val).numpy()
        gen_test_vae = vae.generate(r_test).numpy()
    if ddpm_ema is not None:
        gen_val_ddpm = sample_with_ema(ddpm, ddpm_ema, r_val,
                                       seed=SEED).numpy()
        gen_test_ddpm = sample_with_ema(ddpm, ddpm_ema, r_test,
                                        seed=SEED + 1).numpy()
    else:
        gen_val_ddpm = ddpm.sample(r_val, seed=SEED).numpy()
        gen_test_ddpm = ddpm.sample(r_test, seed=SEED + 1).numpy()

    print("\n########## VAE baseline ##########")
    rep_vae = full_report(ds.test.x, gen_test_vae, ds.val.x, gen_val_vae,
                          seed=SEED)
    print_report(rep_vae)

    print("\n########## DDPM ##########")
    rep_ddpm = full_report(ds.test.x, gen_test_ddpm, ds.val.x, gen_val_ddpm,
                           seed=SEED)
    print_report(rep_ddpm)

    os.makedirs(RUNS, exist_ok=True)
    # raw score arrays are for the figures script, not the JSON report.
    # full_report attaches them to both model reports; pop them first.
    d_scores = rep_ddpm["discriminator"]
    scores = np.asarray(d_scores.pop("test_scores"), dtype=np.float64)
    n_real = int(d_scores.pop("test_n_real"))
    rep_vae["discriminator"].pop("test_scores", None)
    rep_vae["discriminator"].pop("test_n_real", None)
    out = {
        "seed": SEED,
        "n_train": len(ds.train.x), "n_val": len(ds.val.x),
        "n_test": len(ds.test.x),
        "seconds": ds.n_seconds, "chunks": ds.n_chunks,
        "ddpm_config": {"ch": DDPM_CH, "n_blocks": DDPM_BLOCKS,
                        "epochs": DDPM_EPOCHS, "ema_decay": DDPM_EMA,
                        "params": count_params(ddpm)},
        "vae": rep_vae, "ddpm": rep_ddpm,
        "vae_final_loss": vae_hist[-1], "ddpm_final_loss": ddpm_hist[-1],
        "train_minutes": (time.time() - t0) / 60,
    }
    with open(os.path.join(RUNS, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nmetrics -> runs/metrics.json "
          f"({(time.time() - t0) / 60:.1f} min total)")

    # arrays for the figures script (gitignored; uploaded manually)
    np.savez_compressed(
        os.path.join(RUNS, "eval_arrays.npz"),
        real_test=ds.test.x.astype(np.float32),
        gen_test_ddpm=gen_test_ddpm.astype(np.float32),
        disc_scores_real=scores[:n_real],
        disc_scores_gen=scores[n_real:],
        feature_mean=ds.feature_mean.astype(np.float64),
        feature_std=ds.feature_std.astype(np.float64),
    )
    print("arrays -> runs/eval_arrays.npz")


if __name__ == "__main__":
    main()
