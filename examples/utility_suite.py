"""Utility-task suite: do generated windows preserve task-relevant
conditional structure?

1. Loads the book-state windows (session-disjoint split: train/val/test,
   no shared sessions; provenance gate first).
2. Trains the conditional VAE baseline and the DDPM on TRAIN only, using
   the frozen sweep winner config (B_long_300ep: ch 96, 3 blocks, 300
   epochs, no EMA), and builds the regime-matched chunk-copy baseline
   from TRAIN windows.
3. Generates windows conditioned on VAL regimes for all three methods.
4. Scores the preregistered utility-task suite (flowforge/tasks.py):
   real-to-real, synthetic-to-real, and majority / persistence /
   class-conditional random baselines, on PR-AUC, Brier score, and
   calibration, with blocked bootstrap intervals. The VAL gate is
   decided before the test split is scored; the test split is frozen
   until then.
5. Prints the report and writes examples/utility_suite_results.json.

Fixed seeds everywhere. CPU only. No claim here is about profitable
prediction or execution: these tasks diagnose whether generated data
preserves conditional structure, nothing more.
"""

import json
import os
import time

import numpy as np
import torch

from flowforge.baselines import (RegimeMatchedChunkCopyBaseline,
                                 segment_ids_from_windows)
from flowforge.data import (check_manifest_file, ensure_publishable,
                            load_dataset)
from flowforge.evaluate import print_utility_report, utility_task_report
from flowforge.models import (BookDDPM, CondVAE, count_params, set_seed,
                              train_ddpm, train_vae)

SEED = 7
# Frozen sweep winner (runs/sweep.json in the earlier study): the DDPM
# config with the best val mean W1 among the swept configs.
DDPM_CONFIG = {"ch": 96, "n_blocks": 3, "epochs": 300, "ema_decay": None}
VAE_EPOCHS = 25
BLOCK_LENGTHS = (1, 5, 10, 20)
N_BOOT = 200

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "book_windows.npz")
MANIFEST = os.path.join(os.path.dirname(__file__), "..", "data",
                        "split_manifest.json")
OUT = os.path.join(os.path.dirname(__file__), "utility_suite_results.json")


def _manifest_matches_except_build(manifest_path: str, ds) -> bool:
    """True when the on-disk manifest matches the NPZ manifest except
    for the build-time fields (code_sha, created_utc, manifest_sha256).

    The NPZ in this checkout was rebuilt at the current HEAD, so its
    recorded code_sha differs from the committed manifest's; the
    content (chunk hashes, split, window counts) must otherwise agree
    exactly, or the run aborts.
    """
    with open(manifest_path) as f:
        on_disk = json.load(f)
    embedded = ds.manifest or {}
    skip = ("code_sha", "created_utc", "manifest_sha256")
    a = {k: v for k, v in on_disk.items() if k not in skip}
    b = {k: v for k, v in embedded.items() if k not in skip}
    return a == b


def main() -> None:
    t0 = time.time()
    ds = load_dataset(os.path.abspath(DATA))
    ensure_publishable(ds)
    try:
        check_manifest_file(ds, os.path.abspath(MANIFEST))
    except ValueError:
        if os.path.exists(MANIFEST) and _manifest_matches_except_build(
                os.path.abspath(MANIFEST), ds):
            print("note: on-disk split manifest matches the NPZ manifest "
                  "except build-time fields (code_sha, created_utc); "
                  "proceeding on the NPZ-embedded manifest")
        else:
            raise
    print(f"split-manifest SHA-256: {ds.manifest_sha256}")
    print(f"data: {ds.n_chunks} chunks, {ds.n_seconds}s -> "
          f"{len(ds.train.x)}/{len(ds.val.x)}/{len(ds.test.x)} "
          f"train/val/test windows")

    x_train = torch.from_numpy(ds.train.x)
    r_train = torch.from_numpy(ds.train.regime)
    r_val = torch.from_numpy(ds.val.regime)

    print(f"\n--- training conditional VAE baseline "
          f"({count_params(CondVAE())} params) ---")
    vae = CondVAE()
    vae_hist = train_vae(vae, x_train, r_train, epochs=VAE_EPOCHS,
                         seed=SEED)

    print(f"\n--- training DDPM ({count_params(BookDDPM(ch=DDPM_CONFIG['ch'], n_blocks=DDPM_CONFIG['n_blocks']))} params, "
          f"ch={DDPM_CONFIG['ch']} blocks={DDPM_CONFIG['n_blocks']} "
          f"epochs={DDPM_CONFIG['epochs']} "
          f"ema={DDPM_CONFIG['ema_decay']}) ---")
    ddpm = BookDDPM(ch=DDPM_CONFIG["ch"], n_blocks=DDPM_CONFIG["n_blocks"])
    ddpm_hist, ddpm_ema = train_ddpm(ddpm, x_train, r_train,
                                     epochs=DDPM_CONFIG["epochs"],
                                     seed=SEED,
                                     ema_decay=DDPM_CONFIG["ema_decay"])
    assert ddpm_ema is None, "sweep winner config uses no EMA"

    print("\n--- generating VAL windows (conditioned on VAL regimes) ---")
    vae.eval()
    with torch.inference_mode():
        gen_val_vae = vae.generate(r_val, seed=SEED).numpy()
    gen_val_ddpm = ddpm.sample(r_val, seed=SEED,
                               batch_size=32).numpy()
    seg_train = segment_ids_from_windows(ds.train.session_ids,
                                         ds.train.t_start, ds.train.t_end)
    chunk_copy = RegimeMatchedChunkCopyBaseline(
        ds.train.x, ds.train.regime, block_len=4,
        session_ids=ds.train.session_ids, segment_ids=seg_train)
    gen_val_copy, _ = chunk_copy.generate(len(ds.val.x), ds.val.regime,
                                          seed=SEED)
    print(f"gen val shapes: vae {gen_val_vae.shape}, ddpm "
          f"{gen_val_ddpm.shape}, chunk_copy {gen_val_copy.shape}")

    print("\n--- utility-task suite ---")
    rep = utility_task_report(
        ds.train.x, ds.val.x, ds.test.x,
        {"ddpm": gen_val_ddpm.astype(np.float32),
         "vae": gen_val_vae.astype(np.float32),
         "chunk_copy": gen_val_copy.astype(np.float32)},
        feature_mean=ds.feature_mean, feature_std=ds.feature_std,
        t_start_test=ds.test.t_start,
        block_lengths=BLOCK_LENGTHS, n_boot=N_BOOT, seed=SEED,
        manifest_sha256=ds.manifest_sha256)
    rep["generators"] = {
        "seed": SEED,
        "ddpm": {**DDPM_CONFIG, "params": count_params(ddpm),
                 "final_loss": ddpm_hist[-1],
                 "source": "frozen sweep winner B_long_300ep"},
        "vae": {"epochs": VAE_EPOCHS, "params": count_params(vae),
                "final_loss": vae_hist[-1]},
        "chunk_copy": {"block_len": 4, "kind": "conditional"},
    }
    rep["train_minutes"] = (time.time() - t0) / 60
    print_utility_report(rep)
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"\nresults -> {OUT} ({(time.time() - t0) / 60:.1f} min total)")


if __name__ == "__main__":
    set_seed(SEED)
    main()
