"""Seam-consistency and error-vs-horizon evaluation for rollouts.

A rollout is a trajectory of consecutive windows: an anchor history
followed by ``n_windows`` generated windows. This module audits it
without any long-path realism claim:

- every seam (the boundary between consecutive windows, including the
  anchor-to-first-window boundary) is an explicit checkpoint with
  physical jump sizes for spread, sizes, signed flow, trade count,
  and log return;
- error-vs-horizon curves compare generated and real continuations of
  the same anchors at several horizon lengths (marginal W1 over the
  horizon, cumulative log-return RMSE, spread-level MAE);
- divergence probability is the fraction of trajectories whose
  cumulative log-return error at a horizon exceeds a threshold, with
  intervals from a session-block bootstrap that resamples whole
  independent recording sessions, never individual windows.

All inputs are standardized feature arrays; physical jumps are
computed after inverse-transforming with the training feature
statistics. Trajectory order inside a session is chronological, and
the bootstrap never mixes sessions.
"""

from __future__ import annotations

import numpy as np

from .data import FEATURE_NAMES, LOOKBACK, N_FEATURES, WINDOW
from .evaluate import inverse_transform, marginal_table

J_RET = FEATURE_NAMES.index("log_ret")
J_SPR = FEATURE_NAMES.index("log_spread_bps")
J_BID = FEATURE_NAMES.index("log_bid_sz")
J_ASK = FEATURE_NAMES.index("log_ask_sz")
J_SF = FEATURE_NAMES.index("signed_flow")
J_CNT = FEATURE_NAMES.index("log_trade_count")


def _checked_paths(values, name, n_windows=None):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"{name} must have shape "
                         f"(n, n_windows, window, features), got {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError(f"{name} must be non-empty in every dimension")
    if arr.shape[3] != N_FEATURES:
        raise ValueError(f"{name} must have {N_FEATURES} features, got "
                         f"{arr.shape[3]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    if n_windows is not None and arr.shape[1] != n_windows:
        raise ValueError(f"{name} has {arr.shape[1]} windows, expected "
                         f"{n_windows}")
    return arr


def _checked_context(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (n, lookback, features), "
                         f"got {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.shape[2] != N_FEATURES:
        raise ValueError(f"{name} must have {N_FEATURES} features, got "
                         f"{arr.shape[2]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _checked_stats(values, name, length):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got "
                         f"{arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def seam_table(anchor_context: np.ndarray, generated: np.ndarray,
               feature_mean: np.ndarray,
               feature_std: np.ndarray) -> list[dict]:
    """Per-seam jump sizes for a batch of generated trajectories.

    ``anchor_context`` is (n, lookback, features) and ``generated`` is
    (n, n_windows, window, features), both standardized. Seam 0 is the
    boundary between the anchor's last second and the first generated
    window's first second; seam k > 0 is the boundary between
    generated windows k-1 and k. Jumps are mean absolute differences
    across the batch, in physical units for spread (bps) and in
    standardized units for the log-sized channels, where a physical
    unit is not defined by the feature set. The log-return jump is the
    continuity proxy at the seam: the last log return before the seam
    against the first log return after it.
    """
    anchor = _checked_context(anchor_context, "anchor_context")
    gen = _checked_paths(generated, "generated")
    if gen.shape[0] != anchor.shape[0]:
        raise ValueError("anchor_context and generated batch sizes differ")
    f_mean = _checked_stats(feature_mean, "feature_mean", N_FEATURES)
    f_std = _checked_stats(feature_std, "feature_std", N_FEATURES)
    n, n_windows = gen.shape[0], gen.shape[1]
    # the full path in physical units: anchor seconds, then each window
    path = np.concatenate(
        [anchor, gen.reshape(n, n_windows * WINDOW, N_FEATURES)], axis=1)
    phys = inverse_transform(path, f_mean, f_std)
    lookback = anchor.shape[1]
    rows = []
    for seam in range(n_windows):
        at = lookback + seam * WINDOW  # first row after the seam
        before = phys[:, at - 1, :]
        after = phys[:, at, :]
        # spread is stored log1p(bps); the jump is in bps.
        rows.append({
            "seam": int(seam),
            "n": int(n),
            "spread_bps_jump_mean": float(np.mean(np.abs(
                np.expm1(after[:, J_SPR]) - np.expm1(before[:, J_SPR])))),
            "log_bid_sz_jump_mean": float(np.mean(np.abs(
                after[:, J_BID] - before[:, J_BID]))),
            "log_ask_sz_jump_mean": float(np.mean(np.abs(
                after[:, J_ASK] - before[:, J_ASK]))),
            "signed_flow_jump_mean": float(np.mean(np.abs(
                after[:, J_SF] - before[:, J_SF]))),
            "log_trade_count_jump_mean": float(np.mean(np.abs(
                after[:, J_CNT] - before[:, J_CNT]))),
            "log_ret_jump_mean": float(np.mean(np.abs(
                after[:, J_RET] - before[:, J_RET]))),
        })
    return rows


def cumulative_log_return(paths: np.ndarray,
                          feature_mean: np.ndarray | None = None,
                          feature_std: np.ndarray | None = None) -> np.ndarray:
    """Cumulative log return at each window end, per trajectory.

    ``paths`` is (n, n_windows, window, features); the return is the
    cumulative sum of the log-return channel from the anchor's last
    second, evaluated at the end of each window. Shape (n, n_windows).
    With ``feature_mean``/``feature_std`` the sum is computed in
    physical log-return units (the stored channel is standardized, so
    a raw sum would be in train-standard-deviation units and a
    physical threshold would be meaningless); without them the sum is
    in the stored standardized units.
    """
    arr = _checked_paths(paths, "paths")
    rets = arr[:, :, :, J_RET]
    if feature_mean is not None or feature_std is not None:
        f_mean = _checked_stats(feature_mean, "feature_mean", N_FEATURES)
        f_std = _checked_stats(feature_std, "feature_std", N_FEATURES)
        rets = rets * f_std[J_RET] + f_mean[J_RET]
    return np.cumsum(rets.sum(axis=2), axis=1)


def error_vs_horizon(real_paths: np.ndarray, gen_paths: np.ndarray,
                     horizons: tuple[int, ...],
                     feature_mean: np.ndarray | None = None,
                     feature_std: np.ndarray | None = None) -> list[dict]:
    """Error of generated continuations against real ones, by horizon.

    For each horizon h (in windows): ``mean_w1`` is the marginal
    1-Wasserstein distance between the real and generated windows
    inside the first h windows (standardized units, per
    ``flowforge.evaluate.marginal_table``); ``cumret_rmse`` is the
    root-mean-square error of the cumulative log return at the end of
    window h, in physical log-return units when ``feature_mean``/
    ``feature_std`` are given (standardized units otherwise);
    ``spread_level_mae`` is the mean absolute error of the
    per-trajectory mean log-spread over the horizon. Horizons must be
    positive and no longer than the trajectories.
    """
    real = _checked_paths(real_paths, "real_paths")
    gen = _checked_paths(gen_paths, "gen_paths", n_windows=real.shape[1])
    if real.shape[0] != gen.shape[0]:
        raise ValueError("real_paths and gen_paths batch sizes differ")
    n_windows = real.shape[1]
    if not horizons:
        raise ValueError("horizons must not be empty")
    rows = []
    for h in horizons:
        if not isinstance(h, int) or isinstance(h, bool) or h <= 0 \
                or h > n_windows:
            raise ValueError(
                f"horizon must be a positive integer no larger than "
                f"{n_windows}, got {h!r}")
        real_h = real[:, :h].reshape(real.shape[0], h * WINDOW, N_FEATURES)
        gen_h = gen[:, :h].reshape(gen.shape[0], h * WINDOW, N_FEATURES)
        real_cum = cumulative_log_return(real, feature_mean, feature_std)[:, h - 1]
        gen_cum = cumulative_log_return(gen, feature_mean, feature_std)[:, h - 1]
        rows.append({
            "horizon_windows": int(h),
            "horizon_seconds": int(h * WINDOW),
            "n": int(real.shape[0]),
            "mean_w1": float(marginal_table(real_h, gen_h)["mean_w1"]),
            "cumret_rmse": float(np.sqrt(np.mean((real_cum - gen_cum) ** 2))),
            "spread_level_mae": float(np.mean(np.abs(
                real_h[:, :, J_SPR].mean(axis=1)
                - gen_h[:, :, J_SPR].mean(axis=1)))),
        })
    return rows


def _session_bootstrap_ci(frac_by_session: np.ndarray,
                          sizes: np.ndarray, n_boot: int,
                          seed: int) -> list[float] | None:
    """Bootstrap CI for a fraction, resampling whole sessions.

    ``frac_by_session`` holds the diverged fraction inside each
    session and ``sizes`` the session trajectory counts. Each
    replicate draws sessions with replacement (same session count)
    and recomputes the size-weighted fraction. Returns the 95%
    percentile interval, or None when there are no sessions.
    """
    if len(frac_by_session) == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(frac_by_session),
                           size=len(frac_by_session))
        w = sizes[idx].astype(np.float64)
        vals.append(float(np.sum(frac_by_session[idx] * w) / w.sum()))
    vals = np.asarray(vals)
    return [float(np.quantile(vals, 0.025)),
            float(np.quantile(vals, 0.975))]


def divergence_probability(real_paths: np.ndarray, gen_paths: np.ndarray,
                           horizons: tuple[int, ...], threshold: float,
                           session_ids: np.ndarray | None = None,
                           n_boot: int = 200, seed: int = 0,
                           feature_mean: np.ndarray | None = None,
                           feature_std: np.ndarray | None = None) -> list[dict]:
    """Fraction of trajectories whose cumret error exceeds threshold.

    A trajectory diverges at horizon h when the absolute error of its
    cumulative log return at the end of window h exceeds ``threshold``
    (in physical log-return units when ``feature_mean``/``feature_std``
    are given, in standardized units otherwise). With ``session_ids``
    the interval is a session-block bootstrap (whole independent
    sessions resampled, 200 replicates by default); without it the
    interval is None and the row carries the point estimate only.
    """
    real = _checked_paths(real_paths, "real_paths")
    gen = _checked_paths(gen_paths, "gen_paths", n_windows=real.shape[1])
    if real.shape[0] != gen.shape[0]:
        raise ValueError("real_paths and gen_paths batch sizes differ")
    n_windows = real.shape[1]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) \
            or threshold <= 0:
        raise ValueError("threshold must be a positive number")
    if not horizons:
        raise ValueError("horizons must not be empty")
    for h in horizons:
        if not isinstance(h, int) or isinstance(h, bool) or h <= 0 \
                or h > n_windows:
            raise ValueError(
                f"horizon must be a positive integer no larger than "
                f"{n_windows}, got {h!r}")
    if session_ids is not None:
        sids = np.asarray(session_ids)
        if sids.shape != (real.shape[0],):
            raise ValueError(
                f"session_ids must have shape ({real.shape[0]},), got "
                f"{sids.shape}")
        sessions, _ = np.unique(sids, return_counts=True)
    else:
        sessions = None
    real_cum = cumulative_log_return(real, feature_mean, feature_std)
    gen_cum = cumulative_log_return(gen, feature_mean, feature_std)
    abs_err = np.abs(real_cum - gen_cum)
    rows = []
    for h in horizons:
        diverged = abs_err[:, h - 1] > threshold
        frac = float(diverged.mean())
        ci = None
        if sessions is not None:
            per_session_frac = np.array([
                float(diverged[sids == s].mean()) for s in sessions])
            per_session_size = np.array([
                int((sids == s).sum()) for s in sessions],
                dtype=np.int64)
            ci = _session_bootstrap_ci(per_session_frac, per_session_size,
                                       n_boot, seed + 1000 * int(h))
        rows.append({
            "horizon_windows": int(h),
            "horizon_seconds": int(h * WINDOW),
            "n": int(real.shape[0]),
            "n_sessions": (int(len(sessions)) if sessions is not None
                           else None),
            "diverged_fraction": frac,
            "diverged_count": int(diverged.sum()),
            "threshold": float(threshold),
            "ci": ci,
        })
    return rows


def rollout_report(real_paths: np.ndarray, gen_by_method: dict,
                   anchor_context: np.ndarray,
                   feature_mean: np.ndarray, feature_std: np.ndarray,
                   horizons: tuple[int, ...], threshold: float,
                   session_ids: np.ndarray | None = None,
                   n_boot: int = 200, seed: int = 0) -> dict:
    """Seam tables, error curves, and divergence for several methods.

    ``gen_by_method`` maps a method name to its generated paths
    (n, n_windows, window, features), aligned trajectory-for-trajectory
    with ``real_paths``. Every method gets the same horizons, the same
    threshold, and the same session-block intervals, so the methods
    are compared on identical anchors. The anchor seam table is
    computed per method from its own generated windows.
    """
    real = _checked_paths(real_paths, "real_paths")
    anchor = _checked_context(anchor_context, "anchor_context")
    if anchor.shape[0] != real.shape[0]:
        raise ValueError("anchor_context and real_paths batch sizes differ")
    methods = {}
    for name, gen in gen_by_method.items():
        gen_arr = _checked_paths(gen, f"gen_paths[{name}]",
                                 n_windows=real.shape[1])
        if gen_arr.shape[0] != real.shape[0]:
            raise ValueError(f"gen_paths[{name}] batch size differs from "
                             f"real_paths")
        methods[name] = {
            "seams": seam_table(anchor, gen_arr, feature_mean, feature_std),
            "error_vs_horizon": error_vs_horizon(
                real, gen_arr, horizons, feature_mean, feature_std),
            "divergence": divergence_probability(
                real, gen_arr, horizons, threshold,
                session_ids=session_ids, n_boot=n_boot, seed=seed,
                feature_mean=feature_mean, feature_std=feature_std),
        }
    return {
        "horizons": [int(h) for h in horizons],
        "threshold": float(threshold),
        "n": int(real.shape[0]),
        "n_windows": int(real.shape[1]),
        "lookback_seconds": int(LOOKBACK),
        "methods": methods,
    }
