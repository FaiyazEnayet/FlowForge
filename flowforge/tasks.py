"""Utility tasks: downstream prediction problems that ask whether
generated windows preserve task-relevant conditional dependencies.

The existing train-on-synthetic / test-on-real flow-direction task is
uninformative when the DDPM-trained classifier only equals the majority
predictor (accuracy 0.626, equal to majority). This module registers a
small suite of causal prediction tasks and the baselines and metrics
needed to judge them honestly.

Design (registered in advance; VAL-gated before any test evaluation):

- Each task splits one 60-second window into a FEATURE horizon (rows
  0:30) and a TARGET horizon (rows 30:60, or 30:40 for the 10-second
  variants). Features come strictly from the feature horizon; labels
  come strictly from the target horizon. `verify_timing` proves the
  horizons are disjoint in seconds, so no label information can leak
  into the features.
- Labels are computed from PHYSICAL units: callers pass
  inverse-transformed windows (see `flowforge.evaluate.inverse_transform`).
  Thresholds for the median-split tasks are fit on the TRAIN split
  only and frozen before VAL is scored.
- Tasks:

    spread_widen_30s / spread_widen_10s: target-half mean spread
        above the feature-half mean spread.
    activity_high_30s / activity_high_10s: target-half total trade
        count above the train median.
    imbalance_pos_30s: target-half mean imbalance above zero.
    vol_high_30s: target-half realized volatility (std of 1s log
        returns) above the train median.

- Baselines for every task: majority (constant train prevalence),
  persistence (the feature-horizon analogue of the label), and
  class-conditional random scores (fixed seed).
- Metrics are PR-AUC (average precision), Brier score, and a
  10-bin calibration table, not accuracy alone: several tasks are
  imbalanced, and accuracy would hide everything the suite exists to
  measure.

VAL gate outcome (fit on train, scored on val, n=168; recorded in
tests/test_tasks.py and examples/utility_suite_results.json): five
tasks pass (real logistic beats majority on PR-AUC and Brier):
spread_widen_30s, spread_widen_10s, imbalance_pos_30s,
activity_high_10s, and vol_high_30s. activity_high_30s does not
(no predictable signal; published as a null, not dropped). Two
caveats the gate numbers carry with them: the vol task's train median
is exactly zero, because 59% of train target halves have exactly zero
realized volatility (the midprice is unchanged second to second at
the recorded precision), so its median split is really a
zero-vs-nonzero split and its VAL pass is marginal; and the test
split's prevalences shift sharply from val (single val session vs
two test sessions), so test Brier and calibration mostly measure that
session shift, not model quality.

No claim here is about profitable prediction or execution: these are
diagnostics of whether generated data preserves conditional structure,
nothing more.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import FEATURE_NAMES
from .models import set_seed

FEATURE_START = 0
FEATURE_END = 30
TARGET_START = 30
TARGET_10_END = 40
TARGET_30_END = 60

_J = {n: FEATURE_NAMES.index(n) for n in FEATURE_NAMES}

TASK_NAMES = (
    "spread_widen_30s",
    "spread_widen_10s",
    "activity_high_30s",
    "activity_high_10s",
    "imbalance_pos_30s",
    "vol_high_30s",
)

TASK_DESCRIPTIONS = {
    "spread_widen_30s":
        "target-half (30s) mean spread above feature-half mean spread",
    "spread_widen_10s":
        "next-10s mean spread above feature-half mean spread",
    "activity_high_30s":
        "target-half (30s) total trade count above the train median",
    "activity_high_10s":
        "next-10s total trade count above the train median",
    "imbalance_pos_30s":
        "target-half (30s) mean imbalance above zero",
    "vol_high_30s":
        "target-half (30s) realized volatility above the train median",
}

THRESHOLD_KEYS = (
    "activity_30s_median",
    "activity_10s_median",
    "vol_30s_median",
    "activity_feature_median",
    "vol_feature_median",
)


# ------------------------------------------------------------ timing

def verify_timing(t_start: np.ndarray, feature_end_row: int = FEATURE_END,
                  target_start_row: int = TARGET_START) -> dict:
    """Prove the feature horizon ends before the target horizon starts.

    t_start: per-window start times in seconds. Row r of a window is
    second t_start + r. Raises ValueError on any overlap.
    """
    t_start = np.asarray(t_start, dtype=np.float64).ravel()
    if target_start_row < feature_end_row:
        raise ValueError(
            f"target horizon starts at row {target_start_row} but the "
            f"feature horizon ends at row {feature_end_row}: the "
            f"horizons overlap; labels would leak into features")
    feature_end = t_start + (feature_end_row - 1)
    target_start = t_start + target_start_row
    overlap = int(np.sum(target_start <= feature_end))
    if overlap:
        raise ValueError(
            f"target horizon overlaps the feature horizon in {overlap} "
            f"window(s)")
    return {
        "feature_start_row": FEATURE_START,
        "feature_end_row": int(feature_end_row),
        "target_start_row": int(target_start_row),
        "feature_end_second": [float(v) for v in feature_end],
        "target_start_second": [float(v) for v in target_start],
        "overlap_seconds": 0,
    }


# ------------------------------------------------------------ features

def extract_features(windows: np.ndarray) -> np.ndarray:
    """12-dim features from the feature horizon (rows 0:30) only.

    Columns: the 7 per-feature means over rows 0:30, the feature-half
    realized volatility (std of 1s log returns), and 4 trend terms
    (mean over rows 15:30 minus mean over rows 0:15) for spread,
    imbalance, signed flow, and log trade count. Rows 30:60 never
    enter the features.
    """
    w = np.asarray(windows, dtype=np.float64)
    f = w[:, FEATURE_START:FEATURE_END, :]
    means = f.mean(axis=1)
    vol = f[:, :, _J["log_ret"]].std(axis=1, keepdims=True)
    trends = np.stack([
        f[:, 15:, _J["log_spread_bps"]].mean(axis=1)
        - f[:, :15, _J["log_spread_bps"]].mean(axis=1),
        f[:, 15:, _J["imbalance"]].mean(axis=1)
        - f[:, :15, _J["imbalance"]].mean(axis=1),
        f[:, 15:, _J["signed_flow"]].mean(axis=1)
        - f[:, :15, _J["signed_flow"]].mean(axis=1),
        f[:, 15:, _J["log_trade_count"]].mean(axis=1)
        - f[:, :15, _J["log_trade_count"]].mean(axis=1),
    ], axis=1)
    return np.concatenate([means, vol, trends], axis=1)


# ------------------------------------------------------------ labels

def _target_quantities(windows: np.ndarray) -> dict:
    """Target-horizon quantities in physical units, per window."""
    w = np.asarray(windows, dtype=np.float64)
    f = w[:, FEATURE_START:FEATURE_END, :]
    t30 = w[:, TARGET_START:TARGET_30_END, :]
    t10 = w[:, TARGET_START:TARGET_10_END, :]
    return {
        "spread_feature": f[:, :, _J["log_spread_bps"]].mean(axis=1),
        "spread_30s": t30[:, :, _J["log_spread_bps"]].mean(axis=1),
        "spread_10s": t10[:, :, _J["log_spread_bps"]].mean(axis=1),
        "count_feature": np.expm1(
            f[:, :, _J["log_trade_count"]]).sum(axis=1),
        "count_30s": np.expm1(
            t30[:, :, _J["log_trade_count"]]).sum(axis=1),
        "count_10s": np.expm1(
            t10[:, :, _J["log_trade_count"]]).sum(axis=1),
        "imbalance_30s": t30[:, :, _J["imbalance"]].mean(axis=1),
        "imbalance_feature": f[:, :, _J["imbalance"]].mean(axis=1),
        "vol_feature": f[:, :, _J["log_ret"]].std(axis=1),
        "vol_30s": t30[:, :, _J["log_ret"]].std(axis=1),
        "spread_trend": (f[:, 15:, _J["log_spread_bps"]].mean(axis=1)
                         - f[:, :15, _J["log_spread_bps"]].mean(axis=1)),
    }


def fit_thresholds(train_windows: np.ndarray) -> dict:
    """Median-split thresholds, fit on TRAIN windows only.

    The medians are of the TRAIN target-horizon quantities (and of the
    feature-horizon quantities the persistence baselines need). They
    are frozen before VAL is scored and reused unchanged on TEST.
    """
    q = _target_quantities(train_windows)
    return {
        "activity_30s_median": float(np.median(q["count_30s"])),
        "activity_10s_median": float(np.median(q["count_10s"])),
        "vol_30s_median": float(np.median(q["vol_30s"])),
        "activity_feature_median": float(np.median(q["count_feature"])),
        "vol_feature_median": float(np.median(q["vol_feature"])),
    }


def task_labels(windows: np.ndarray, thresholds: dict) -> dict:
    """Binary labels per task, from the target horizon only.

    windows must be in physical units. Every threshold in
    `thresholds` must be present; a missing key raises KeyError
    rather than silently defaulting.
    """
    for key in THRESHOLD_KEYS:
        if key not in thresholds:
            raise KeyError(f"task_labels needs threshold {key!r}")
    q = _target_quantities(windows)
    return {
        "spread_widen_30s":
            (q["spread_30s"] > q["spread_feature"]).astype(float),
        "spread_widen_10s":
            (q["spread_10s"] > q["spread_feature"]).astype(float),
        "activity_high_30s":
            (q["count_30s"] > thresholds["activity_30s_median"]).astype(float),
        "activity_high_10s":
            (q["count_10s"] > thresholds["activity_10s_median"]).astype(float),
        "imbalance_pos_30s":
            (q["imbalance_30s"] > 0.0).astype(float),
        "vol_high_30s":
            (q["vol_30s"] > thresholds["vol_30s_median"]).astype(float),
    }


def persistence_baseline(windows: np.ndarray,
                         thresholds: dict) -> dict:
    """Per-task persistence predictions (0/1 scores).

    The feature-horizon analogue of each label: spread widening
    continues if the spread rose inside the feature half; activity
    stays high if the feature-half count is above its train median;
    the imbalance sign persists; volatility stays high if the
    feature-half volatility is above its train median.
    """
    for key in THRESHOLD_KEYS:
        if key not in thresholds:
            raise KeyError(f"persistence_baseline needs threshold {key!r}")
    q = _target_quantities(windows)
    return {
        "spread_widen_30s":
            (q["spread_trend"] > 0.0).astype(float),
        "spread_widen_10s":
            (q["spread_trend"] > 0.0).astype(float),
        "activity_high_30s":
            (q["count_feature"]
             > thresholds["activity_feature_median"]).astype(float),
        "activity_high_10s":
            (q["count_feature"]
             > thresholds["activity_feature_median"]).astype(float),
        "imbalance_pos_30s":
            (q["imbalance_feature"] > 0.0).astype(float),
        "vol_high_30s":
            (q["vol_feature"]
             > thresholds["vol_feature_median"]).astype(float),
    }


# ------------------------------------------------------------ baselines

def majority_baseline(y_train: np.ndarray,
                      n: int) -> tuple[float, np.ndarray]:
    """Constant train-prevalence scores: (prevalence, scores)."""
    prev = float(np.asarray(y_train, dtype=np.float64).mean())
    return prev, np.full(n, prev, dtype=np.float64)


def random_baseline_scores(n: int, seed: int = 0) -> np.ndarray:
    """Class-conditional random scores: iid uniform draws, fixed seed."""
    return np.random.default_rng(seed).random(n)


# ------------------------------------------------------------ metrics

def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (threshold-based, tie-safe).

    At each distinct score threshold, precision and recall are computed
    over scores >= threshold; AP is the recall-weighted mean precision.
    A constant score therefore yields exactly the prevalence, and a
    perfect ranking yields 1.0. Returns nan when a class is absent.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = float(y.sum())
    n = len(y)
    if n_pos == 0 or n_pos == n or n == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    # one step per distinct score: the step's precision is the
    # precision at the last row carrying that score
    distinct = np.r_[np.where(np.diff(s_sorted) != 0)[0],
                     len(s_sorted) - 1]
    rec_step = np.r_[0.0, rec[distinct]]
    prec_step = prec[distinct]
    return float(np.sum((rec_step[1:] - rec_step[:-1]) * prec_step))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and labels."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray,
                      n_bins: int = 10) -> dict:
    """10-bin calibration table and expected calibration error."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            bins.append({"bin": [float(edges[b]), float(edges[b + 1])],
                         "n": 0, "mean_predicted": float("nan"),
                         "mean_observed": float("nan")})
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (n_b / len(y)) * abs(acc - conf)
        bins.append({"bin": [float(edges[b]), float(edges[b + 1])],
                     "n": n_b, "mean_predicted": conf,
                     "mean_observed": acc})
    return {
        "ece": float(ece),
        "mean_predicted": float(p.mean()) if len(p) else float("nan"),
        "mean_observed": float(y.mean()) if len(y) else float("nan"),
        "n_bins": int(n_bins),
        "bins": bins,
    }


# ------------------------------------------------------------ model

def fit_logistic_probs(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, seed: int = 0,
                       epochs: int = 300) -> np.ndarray:
    """Logistic regression (same protocol as the TSTR model in
    evaluate.py: torch Adam, lr 0.1, train-standardized features,
    fixed seed). Returns test probabilities. A single-class training
    label returns the constant training prevalence.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    x_test = np.asarray(x_test, dtype=np.float64)
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_test), float(y_train.mean()))
    set_seed(seed)
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8
    w = torch.zeros(x_train.shape[1] + 1, dtype=torch.float32,
                    requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.1)
    bce = torch.nn.BCEWithLogitsLoss()
    xb = torch.from_numpy(np.column_stack(
        [(x_train - mu) / sd, np.ones(len(x_train))]).astype(np.float32))
    yb = torch.from_numpy(y_train.astype(np.float32))
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(xb @ w, yb)
        loss.backward()
        opt.step()
    wn = w.detach().numpy()
    z = (x_test - mu) / sd @ wn[:-1] + wn[-1]
    return 1.0 / (1.0 + np.exp(-z))


def score_all(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """PR-AUC, Brier, calibration, and accuracy for one prediction."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    return {
        "pr_auc": pr_auc(y, p),
        "brier": brier_score(y, p),
        "accuracy": float(((p > 0.5) == (y > 0.5)).mean()) if len(y)
        else float("nan"),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "calibration": calibration_table(y, p),
    }
