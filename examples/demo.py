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
from flowforge.models import (BookDDPM, CondVAE, count_params, set_seed,
                              train_ddpm, train_vae)

SEED = 7
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "book_windows.npz")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")


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

    print(f"\n--- training DDPM ({count_params(BookDDPM())} params) ---")
    ddpm = BookDDPM()  # T=100, linear betas, alpha_bar_T ~ 0.02
    ddpm_hist = train_ddpm(ddpm, x_train, r_train, epochs=80, seed=SEED)

    print("\n--- generating ---")
    vae.eval()
    r_val = torch.from_numpy(ds.val.regime)
    r_test = torch.from_numpy(ds.test.regime)
    with torch.no_grad():
        gen_val_vae = vae.generate(r_val).numpy()
        gen_test_vae = vae.generate(r_test).numpy()
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
    out = {
        "seed": SEED,
        "n_train": len(ds.train.x), "n_val": len(ds.val.x),
        "n_test": len(ds.test.x),
        "seconds": ds.n_seconds, "chunks": ds.n_chunks,
        "vae": rep_vae, "ddpm": rep_ddpm,
        "vae_final_loss": vae_hist[-1], "ddpm_final_loss": ddpm_hist[-1],
        "train_minutes": (time.time() - t0) / 60,
    }
    with open(os.path.join(RUNS, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nmetrics -> runs/metrics.json "
          f"({(time.time() - t0) / 60:.1f} min total)")


if __name__ == "__main__":
    main()
