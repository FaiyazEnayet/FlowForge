"""Demo bundle helpers for the Streamlit dashboard (app.py).

The dashboard must run from a fresh clone with no data/ or runs/ (both are
gitignored). examples/build_demo.py freezes a small, fully precomputed sample
under demo/ from the outputs of examples/demo.py; everything here only reads
that committed bundle. No training, no downloads, no network calls.

Bundle contents (demo/demo.npz):
  real_windows      (n, 60, 7) float32, standardized real test windows
  gen_windows       (n, 60, 7) float32, standardized windows dreamed by the
                    DDPM trained in examples/demo.py (never invented here)
  feature_names     (7,) the book-state feature names, in order
  feature_mean      (7,) train-split standardization mean
  feature_std       (7,) train-split standardization std
  disc_scores_real  (m,) held-out discriminator scores for real test windows
  disc_scores_gen   (m,) held-out discriminator scores for generated windows
  disc_auc          scalar, discriminator AUC on the test split

And demo/metrics.json: the trimmed evaluation report (marginal table rows,
flow autocorrelation rows, model comparison numbers), copied verbatim from
runs/metrics.json by examples/build_demo.py.
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.normpath(os.path.join(HERE, "..", "demo", "demo.npz"))
DEFAULT_METRICS = os.path.normpath(os.path.join(HERE, "..", "demo", "metrics.json"))

REQUIRED_KEYS = (
    "real_windows",
    "gen_windows",
    "feature_names",
    "feature_mean",
    "feature_std",
    "disc_scores_real",
    "disc_scores_gen",
    "disc_auc",
)

N_SECONDS = 60
N_FEATURES = 7


def load_demo_bundle(bundle_path: str = DEFAULT_BUNDLE,
                     metrics_path: str = DEFAULT_METRICS) -> dict:
    """Load and validate the committed demo bundle.

    Raises FileNotFoundError if the bundle is missing and ValueError if it is
    malformed (missing keys, wrong shapes, non-finite values).
    """
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"demo bundle not found: {bundle_path}. It ships with the repo "
            f"under demo/; did you clone the full repository?")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"demo metrics not found: {metrics_path}")
    z = np.load(bundle_path, allow_pickle=True)
    missing = [k for k in REQUIRED_KEYS if k not in z]
    if missing:
        raise ValueError(f"demo bundle {bundle_path} missing keys: {missing}")

    real = np.asarray(z["real_windows"], dtype=np.float64)
    gen = np.asarray(z["gen_windows"], dtype=np.float64)
    names = [str(n) for n in np.asarray(z["feature_names"]).tolist()]
    mean = np.asarray(z["feature_mean"], dtype=np.float64)
    std = np.asarray(z["feature_std"], dtype=np.float64)
    s_real = np.asarray(z["disc_scores_real"], dtype=np.float64).ravel()
    s_gen = np.asarray(z["disc_scores_gen"], dtype=np.float64).ravel()
    auc = float(np.asarray(z["disc_auc"]).ravel()[0])

    for label, arr in (("real_windows", real), ("gen_windows", gen)):
        if arr.ndim != 3 or arr.shape[1] != N_SECONDS or arr.shape[2] != N_FEATURES:
            raise ValueError(
                f"{label} has shape {arr.shape}, expected "
                f"(n_windows, {N_SECONDS}, {N_FEATURES})")
        if not np.isfinite(arr).all():
            raise ValueError(f"{label} contains non-finite values")
    if len(names) != N_FEATURES or mean.shape != (N_FEATURES,) \
            or std.shape != (N_FEATURES,):
        raise ValueError("feature metadata has wrong length")
    if not (np.isfinite(s_real).all() and np.isfinite(s_gen).all()
            and np.isfinite(auc)):
        raise ValueError("discriminator scores/AUC contain non-finite values")
    if not 0.0 <= auc <= 1.0:
        raise ValueError(f"disc_auc={auc} is not a valid AUC")

    with open(metrics_path) as f:
        metrics = json.load(f)

    return {
        "real": real,
        "gen": gen,
        "feature_names": names,
        "feature_mean": mean,
        "feature_std": std,
        "disc_scores_real": s_real,
        "disc_scores_gen": s_gen,
        "disc_auc": auc,
        "metrics": metrics,
    }


def unstandardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Undo train-split standardization."""
    return x * std + mean


def midprice_path(window_unstd: np.ndarray, j_ret: int,
                  start: float = 100.0) -> np.ndarray:
    """Rebuild a readable midprice path from log returns (rebased to start)."""
    return start * np.exp(np.cumsum(window_unstd[:, j_ret]))


def spread_bps(window_unstd: np.ndarray, j_spread: int) -> np.ndarray:
    """Spread is stored as log1p(spread in bps); invert it."""
    return np.expm1(window_unstd[:, j_spread])
