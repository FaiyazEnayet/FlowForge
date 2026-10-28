"""Freeze a small demo bundle for the Streamlit dashboard.

Reads runs/eval_arrays.npz and runs/metrics.json (written by examples/demo.py,
seed 7) and writes demo/demo.npz + demo/metrics.json: a small precomputed
sample that ships with the repo so `streamlit run app.py` works from a fresh
clone with no data/ or runs/ (both gitignored). Deterministic: seed 7.

Nothing here invents data. The generated windows are outputs of the DDPM
trained in examples/demo.py; the metrics are copied verbatim from
runs/metrics.json.

Provenance (FL7): the bundle records SHA256 hashes of its inputs
(eval_arrays.npz, metrics.json) and the effective config, so the bundle
can be traced back to the exact run that produced it. Since C02 it also
records the split-manifest SHA from runs/metrics.json, and refuses to
build when that SHA is absent: a bundle cannot claim provenance if its
upstream artifacts are missing or changed. See REPRODUCE.md
for the full pipeline from raw recordings to this bundle.
"""

import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
DEMO = os.path.join(HERE, "..", "demo")

SEED = 7
N_WINDOWS = 64  # real + 64 generated windows in the bundle

FEATURES = ["log_ret", "log_spread_bps", "log_bid_sz", "log_ask_sz",
            "imbalance", "signed_flow", "log_trade_count"]
assert len(FEATURES) == 7


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_split_manifest_sha(metrics):
    """Split-manifest SHA a demo bundle's provenance must cite.

    Raises ValueError when runs/metrics.json carries no split-manifest
    SHA: the bundle cannot claim provenance if its upstream artifacts
    are missing. Rebuild metrics with examples/demo.py (C02) first.
    """
    sha = metrics.get("split_manifest_sha256")
    if not isinstance(sha, str) or not sha:
        raise ValueError(
            "runs/metrics.json carries no split-manifest SHA: rebuild "
            "metrics with examples/demo.py first; the demo bundle cannot "
            "claim provenance without the manifest chain")
    return sha


def main():
    rng = np.random.default_rng(SEED)
    z = np.load(os.path.join(RUNS, "eval_arrays.npz"))
    real = z["real_test"]
    gen = z["gen_test_ddpm"]
    assert real.shape[1:] == (60, 7) and gen.shape[1:] == (60, 7)

    i_real = rng.choice(len(real), N_WINDOWS, replace=False)
    i_gen = rng.choice(len(gen), N_WINDOWS, replace=False)
    real_s = np.asarray(real[i_real], dtype=np.float32)
    gen_s = np.asarray(gen[i_gen], dtype=np.float32)
    assert np.isfinite(real_s).all() and np.isfinite(gen_s).all()

    with open(os.path.join(RUNS, "metrics.json")) as f:
        metrics = json.load(f)
    ddpm = metrics["ddpm"]
    vae = metrics["vae"]
    manifest_sha = require_split_manifest_sha(metrics)

    os.makedirs(DEMO, exist_ok=True)
    np.savez_compressed(
        os.path.join(DEMO, "demo.npz"),
        real_windows=real_s,
        gen_windows=gen_s,
        feature_names=np.array(FEATURES),
        feature_mean=z["feature_mean"],
        feature_std=z["feature_std"],
        disc_scores_real=np.asarray(z["disc_scores_real"], dtype=np.float32),
        disc_scores_gen=np.asarray(z["disc_scores_gen"], dtype=np.float32),
        disc_auc=np.float32(ddpm["discriminator"]["auc_test"]),
        real_idx=np.asarray(i_real),
        gen_idx=np.asarray(i_gen),
    )

    summary = {
        "seed": metrics["seed"],
        "n_train": metrics["n_train"],
        "n_val": metrics["n_val"],
        "n_test": metrics["n_test"],
        "seconds": metrics["seconds"],
        "chunks": metrics["chunks"],
        "ddpm_params": metrics["ddpm_config"]["params"],
        "ddpm_epochs": metrics["ddpm_config"]["epochs"],
        "models": {
            name: {
                "mean_w1": rep["marginals"]["mean_w1"],
                "auc_test": rep["discriminator"]["auc_test"],
                "auc_val": rep["discriminator"]["auc_val"],
            }
            for name, rep in (("ddpm", ddpm), ("vae", vae))
        },
        "marginals": ddpm["marginals"]["rows"],
        "flow_acf": ddpm["flow_acf"]["rows"],
        # FL7 provenance: hashes of the exact inputs that produced this
        # bundle, plus the effective config. The bundle is reproducible
        # from these inputs with seed 7; see REPRODUCE.md.
        # C02 adds the split-manifest SHA, chaining the bundle back to
        # the raw recordings through runs/metrics.json.
        "provenance": {
            "split_manifest_sha256": manifest_sha,
            "eval_arrays_sha256": _sha256_file(
                os.path.join(RUNS, "eval_arrays.npz")),
            "metrics_sha256": _sha256_file(
                os.path.join(RUNS, "metrics.json")),
            "ddpm_config": metrics["ddpm_config"],
            "build_seed": SEED,
            "n_windows": N_WINDOWS,
        },
    }
    with open(os.path.join(DEMO, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"bundle -> demo/demo.npz ({N_WINDOWS} real + {N_WINDOWS} generated "
          f"windows, seed {SEED})")
    print("metrics -> demo/metrics.json")
    print(f"provenance: eval_arrays "
          f"{summary['provenance']['eval_arrays_sha256'][:16]}..., "
          f"metrics {summary['provenance']['metrics_sha256'][:16]}...")


if __name__ == "__main__":
    main()
