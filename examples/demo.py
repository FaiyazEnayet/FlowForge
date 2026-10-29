"""End to end: train on real windows, generate, evaluate honestly.

1. Loads the book-state windows (session-disjoint split: train/val/test, no shared sessions).
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

from flowforge.checkpoints import (load_checkpoint,
                                   save_inference_checkpoint)
from flowforge.data import (N_FEATURES, WINDOW, check_manifest_file,
                            ensure_publishable, load_dataset)
from flowforge.evaluate import full_report, print_report
from flowforge.models import (LATENT_DIM, REGIME_DIM, BookDDPM, CondVAE,
                              count_params, sample_with_ema, set_seed,
                              train_ddpm, train_vae)

SEED = 7
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "book_windows.npz")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")

ENV_CONFIG_KEYS = ("FF_DDPM_CH", "FF_DDPM_BLOCKS",
                   "FF_DDPM_EPOCHS", "FF_DDPM_EMA")


def resolve_ddpm_config(sweep_path=None, env=None):
    """Resolve the DDPM config from a frozen source, never silent defaults.

    Sources, in order: the sweep winner config in runs/sweep.json when the
    winner is a neural model; explicit FF_DDPM_* environment variables
    (all four required) otherwise. A baseline sweep winner is recorded
    plainly and still requires an explicit DDPM config for the demo run.
    Raises RuntimeError when no frozen config is available.
    Returns (config dict, winner dict with name/kind/source).
    """
    if env is None:
        env = os.environ
    if sweep_path is None:
        sweep_path = os.path.join(RUNS, "sweep.json")
    winner = None
    if os.path.exists(sweep_path):
        try:
            with open(sweep_path) as f:
                artifact = json.load(f)
            w = artifact["winner"]
            name, kind = w["name"], w.get("kind")
            if kind is None:
                kind = ("baseline" if name.startswith("baseline_")
                        else "neural")
            if kind == "neural":
                cfg = w["config"]
                return ({"ch": int(cfg["ch"]),
                         "n_blocks": int(cfg["n_blocks"]),
                         "epochs": int(cfg["epochs"]),
                         "ema_decay": (None if cfg["ema_decay"] is None
                                       else float(cfg["ema_decay"]))},
                        {"name": name, "kind": kind, "source": "sweep"})
            winner = {"name": name, "kind": kind, "source": "sweep"}
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"sweep artifact {sweep_path} is not usable: {exc}")
    if all(k in env for k in ENV_CONFIG_KEYS):
        ema_raw = env["FF_DDPM_EMA"]
        config = {"ch": int(env["FF_DDPM_CH"]),
                  "n_blocks": int(env["FF_DDPM_BLOCKS"]),
                  "epochs": int(env["FF_DDPM_EPOCHS"]),
                  "ema_decay": float(ema_raw) if ema_raw else None}
        if winner is None:
            winner = {"name": None, "kind": None, "source": "env"}
        return config, winner
    raise RuntimeError(
        "no frozen DDPM config: run examples/sweep.py first or set "
        "FF_DDPM_CH, FF_DDPM_BLOCKS, FF_DDPM_EPOCHS and FF_DDPM_EMA")


def audit_saved_checkpoints(ckpt_root):
    """Reload the saved inference checkpoints as a checksum audit.

    load_checkpoint verifies the manifest digest and the weights SHA-256
    before building the model, so a checkpoint altered after saving
    raises ValueError here and aborts the run instead of passing
    silently. Returns {name: manifest} for the audited checkpoints.
    """
    audited = {}
    for name in ("vae", "ddpm"):
        _, manifest = load_checkpoint(os.path.join(ckpt_root, name))
        audited[name] = manifest
    return audited


def main():
    t0 = time.time()
    set_seed(SEED)
    cfg, winner = resolve_ddpm_config()
    if winner["name"] is not None:
        print(f"sweep winner: {winner['name']} ({winner['kind']}, "
              f"{winner['source']})")
    else:
        print("no sweep winner: explicit env config")
    print(f"DDPM config: ch={cfg['ch']} blocks={cfg['n_blocks']} "
          f"epochs={cfg['epochs']} ema={cfg['ema_decay']}")
    ds = load_dataset(os.path.abspath(DATA))
    # Provenance gate (C02), before any training: legacy datasets cannot
    # publish metrics, and a manifest file on disk that disagrees with
    # the NPZ aborts the run. A missing manifest file is not fatal: the
    # NPZ embeds its own manifest, and that is what gets cited.
    ensure_publishable(ds)
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(DATA)),
                                 "split_manifest.json")
    if not check_manifest_file(ds, os.path.abspath(manifest_path)):
        print("note: no split manifest on disk; proceeding on the "
              "NPZ-embedded manifest only")
    print(f"split-manifest SHA-256: {ds.manifest_sha256}")
    print(f"data: {ds.n_chunks} chunks, {ds.n_seconds}s -> "
          f"{len(ds.train.x)}/{len(ds.val.x)}/{len(ds.test.x)} "
          f"train/val/test windows")

    x_train = torch.from_numpy(ds.train.x)
    r_train = torch.from_numpy(ds.train.regime)

    print("\n--- training conditional VAE baseline "
          f"({count_params(CondVAE())} params) ---")
    vae = CondVAE()
    vae_hist = train_vae(vae, x_train, r_train, epochs=25, seed=SEED)

    print(f"\n--- training DDPM ({count_params(BookDDPM(ch=cfg['ch'], n_blocks=cfg['n_blocks']))} params, "
          f"ch={cfg['ch']} blocks={cfg['n_blocks']} epochs={cfg['epochs']} ema={cfg['ema_decay']}) ---")
    ddpm = BookDDPM(ch=cfg["ch"], n_blocks=cfg["n_blocks"])
    ddpm_hist, ddpm_ema = train_ddpm(ddpm, x_train, r_train,
                                     epochs=cfg["epochs"], seed=SEED,
                                     ema_decay=cfg["ema_decay"])

    print("\n--- saving inference checkpoints ---")
    normalizer = {
        "feature_mean": ds.feature_mean.tolist(),
        "feature_std": ds.feature_std.tolist(),
        "regime_mean": ds.regime_mean.tolist(),
        "regime_std": ds.regime_std.tolist(),
    }
    ckpt_root = os.path.join(RUNS, "checkpoints")
    vae_manifest = save_inference_checkpoint(
        os.path.join(ckpt_root, "vae"), vae, "CondVAE",
        {"latent_dim": LATENT_DIM, "n_features": N_FEATURES,
         "window": WINDOW, "regime_dim": REGIME_DIM},
        normalizer, split_manifest_sha256=ds.manifest_sha256,
        overwrite=True)
    ddpm_manifest = save_inference_checkpoint(
        os.path.join(ckpt_root, "ddpm"), ddpm, "BookDDPM",
        {"n_features": N_FEATURES, "window": WINDOW, "t_steps": 100,
         "beta_start": 1e-4, "beta_end": 0.08, "ch": cfg["ch"],
         "n_blocks": cfg["n_blocks"]},
        normalizer, split_manifest_sha256=ds.manifest_sha256,
        ema=ddpm_ema, overwrite=True)
    print(f"checkpoints -> {ckpt_root} (vae {vae_manifest['weights_source']}, "
          f"ddpm {ddpm_manifest['weights_source']})")
    audit_saved_checkpoints(ckpt_root)
    print("checkpoint audit: manifest and weights digests verified "
          "(vae, ddpm)")

    print("\n--- generating ---")
    vae.eval()
    r_val = torch.from_numpy(ds.val.regime)
    r_test = torch.from_numpy(ds.test.regime)
    with torch.inference_mode():
        gen_val_vae = vae.generate(r_val).numpy()
        gen_test_vae = vae.generate(r_test).numpy()
    if ddpm_ema is not None:
        gen_val_ddpm = sample_with_ema(ddpm, ddpm_ema, r_val,
                                       seed=SEED, batch_size=32).numpy()
        gen_test_ddpm = sample_with_ema(ddpm, ddpm_ema, r_test,
                                        seed=SEED + 1,
                                        batch_size=32).numpy()
    else:
        gen_val_ddpm = ddpm.sample(r_val, seed=SEED,
                                   batch_size=32).numpy()
        gen_test_ddpm = ddpm.sample(r_test, seed=SEED + 1,
                                    batch_size=32).numpy()

    print("\n########## VAE baseline ##########")
    rep_vae = full_report(ds.test.x, gen_test_vae, ds.val.x, gen_val_vae,
                          seed=SEED, feature_mean=ds.feature_mean,
                          feature_std=ds.feature_std)
    print_report(rep_vae)

    print("\n########## DDPM ##########")
    rep_ddpm = full_report(ds.test.x, gen_test_ddpm, ds.val.x, gen_val_ddpm,
                           seed=SEED, feature_mean=ds.feature_mean,
                           feature_std=ds.feature_std)
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
        "split_manifest_sha256": ds.manifest_sha256,
        "dataset_legacy": False,
        "ddpm_config": {"ch": cfg["ch"], "n_blocks": cfg["n_blocks"],
                        "epochs": cfg["epochs"], "ema_decay": cfg["ema_decay"],
                        "params": count_params(ddpm),
                        "source": winner["source"]},
        "sweep_winner": winner,
        "checkpoints": {
            "vae": {"path": "runs/checkpoints/vae",
                    "manifest_sha256": vae_manifest["manifest_sha256"],
                    "weights_source": vae_manifest["weights_source"]},
            "ddpm": {"path": "runs/checkpoints/ddpm",
                     "manifest_sha256": ddpm_manifest["manifest_sha256"],
                     "weights_source": ddpm_manifest["weights_source"]},
        },
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
