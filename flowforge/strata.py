"""Regime stratification for P07: activity imbalance, volatility tails
and regime-balanced training.

Two label kinds, kept apart by type:

- Regime labels come strictly from the lookback BEFORE a window
  (past volatility, spread, trade activity, depth). They are legal
  conditioning: thresholds are fitted on TRAIN only and a fitted
  :class:`StrataSpec` is applied unchanged to val/test. Training
  samplers accept regime labels only.
- Outcome labels are derived from the window itself (its realized
  return tail). They are analysis-only: they describe what happened
  inside the window and must never condition training or sampling.
  They travel as :class:`OutcomeLabels`, which every training entry
  point refuses with a TypeError.

Scarce strata refuse to report: any stratum holding fewer than
``min_independent`` non-overlapping episodes in a split raises
:class:`NotEnoughDataError` instead of publishing a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import FEATURE_NAMES, LOOKBACK, WINDOW

# Causal past features, in the order past_features returns them.
PAST_FEATURE_NAMES = ("past_vol", "past_spread", "past_activity",
                      "past_depth")
_N_PAST = len(PAST_FEATURE_NAMES)

# A window's footprint for independence counting: the lookback that
# conditions it plus the window itself. Two windows whose footprints
# overlap share raw seconds and are not independent episodes.
FOOTPRINT_SECONDS = LOOKBACK + WINDOW

DEFAULT_MIN_INDEPENDENT = 10
DEFAULT_N_BINS = 10
DEFAULT_WEIGHT_CAP = 10.0


class NotEnoughDataError(ValueError):
    """A stratum holds too few independent episodes to report."""


@dataclass
class StrataSpec:
    """Quantile edges fitted on TRAIN for one set of past features.

    ``edges`` maps each feature name to its interior quantile edges
    (``n_bins - 1`` values, increasing). ``assign_strata`` clips
    out-of-range values to the edge bins; it never refits.
    """
    feature_names: tuple[str, ...]
    edges: dict[str, np.ndarray]
    n_bins: int
    kind: str = "regime"
    min_independent: int = DEFAULT_MIN_INDEPENDENT
    n_train: int = 0


@dataclass
class OutcomeLabels:
    """Labels derived from the window itself: analysis only.

    ``labels`` is a boolean tail-membership array (or an int array of
    tail bins). The ``kind`` marker is what training entry points
    check: anything carrying it is refused as a training label.
    """
    labels: np.ndarray
    kind: str = "outcome"
    description: str = ""


def past_features(split, ds: "BookDataset") -> np.ndarray:
    """Raw causal past features per window: (n, 4) float64.

    Columns are past_vol (realized volatility of 1 s log returns over
    the lookback), past_spread (mean spread in bps over the lookback),
    past_activity (total trade count over the lookback) and
    past_depth (mean of the mean log bid/ask sizes over the lookback).
    Volatility and spread come from the standardized regime vector,
    inverted with the TRAIN regime statistics; activity and depth are
    stored raw on the split by build_dataset. Every value is computed
    from seconds strictly before the window start.
    """
    if split.past_activity is None or split.past_depth is None:
        raise ValueError(
            "split carries no past_activity/past_depth: rebuild the "
            "dataset with a build that records lookback aggregates "
            "(datasets built before P07 lack them)")
    n = split.x.shape[0]
    regime_raw = (np.asarray(split.regime, dtype=np.float64)
                  * np.asarray(ds.regime_std, dtype=np.float64)
                  + np.asarray(ds.regime_mean, dtype=np.float64))
    out = np.empty((n, _N_PAST), dtype=np.float64)
    out[:, 0] = regime_raw[:, 0]
    out[:, 1] = regime_raw[:, 1]
    out[:, 2] = np.asarray(split.past_activity, dtype=np.float64)
    out[:, 3] = np.asarray(split.past_depth, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("past features contain non-finite values")
    return out


def window_returns(split, ds: "BookDataset") -> np.ndarray:
    """Raw 60 s log-return sum per window: (n,) float64.

    This is the window's own realized return, so it is an OUTCOME
    quantity: use it for analysis labels only, never for training
    conditioning or sampling.
    """
    j = FEATURE_NAMES.index("log_ret")
    phys = (np.asarray(split.x, dtype=np.float64)
            * np.asarray(ds.feature_std, dtype=np.float64)
            + np.asarray(ds.feature_mean, dtype=np.float64))
    return phys[:, :, j].sum(axis=1)


def fit_strata(train_features: np.ndarray,
               feature_names: tuple[str, ...] | list[str],
               n_bins: int = DEFAULT_N_BINS,
               min_independent: int = DEFAULT_MIN_INDEPENDENT) -> StrataSpec:
    """Fit quantile bin edges on TRAIN features only.

    Edges are the interior quantiles (``n_bins - 1`` per feature).
    A feature with fewer than two distinct edge values (constant on
    train) cannot be stratified and raises.
    """
    names = tuple(feature_names)
    feats = np.asarray(train_features, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[1] != len(names):
        raise ValueError(
            f"train features must have shape (n, {len(names)}), got "
            f"{feats.shape}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names: {names}")
    for name in names:
        if name not in PAST_FEATURE_NAMES:
            raise ValueError(f"unknown past feature: {name!r}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    if not np.isfinite(feats).all():
        raise ValueError("train features contain non-finite values")
    edges: dict[str, np.ndarray] = {}
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for j, name in enumerate(names):
        ed = np.quantile(feats[:, j], qs)
        ed = np.unique(ed)
        if len(ed) < 1:
            raise ValueError(
                f"feature {name!r} is constant on train: cannot stratify")
        edges[name] = ed
    return StrataSpec(feature_names=names, edges=edges, n_bins=n_bins,
                      min_independent=min_independent,
                      n_train=int(feats.shape[0]))


def assign_strata(features: np.ndarray, spec: StrataSpec,
                  feature_names: tuple[str, ...] | list[str] | None = None
                  ) -> np.ndarray:
    """Bin indices for features under a TRAIN-fitted spec: (n, F) ints.

    Values outside the train range clip to the edge bins; the spec is
    never refitted. When ``feature_names`` is given it must match the
    spec's names in order, so a misaligned column order is refused
    rather than silently applied.
    """
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[1] != len(spec.feature_names):
        raise ValueError(
            f"features must have shape (n, {len(spec.feature_names)}), "
            f"got {feats.shape}")
    if feature_names is not None and tuple(feature_names) != spec.feature_names:
        raise ValueError(
            f"feature order {tuple(feature_names)} does not match the "
            f"spec fitted on {spec.feature_names}")
    if not np.isfinite(feats).all():
        raise ValueError("features contain non-finite values")
    out = np.empty(feats.shape, dtype=np.int64)
    for j, name in enumerate(spec.feature_names):
        out[:, j] = np.searchsorted(spec.edges[name], feats[:, j],
                                    side="right")
    return out


def combined_label(assigned: np.ndarray) -> np.ndarray:
    """One stratum id per window from multi-feature bin indices.

    Row-major combination: for a single feature this is the bin index
    itself. Used for training samplers, which take one label per
    window.
    """
    arr = np.asarray(assigned, dtype=np.int64)
    if arr.ndim == 1:
        return arr.copy()
    if arr.ndim != 2:
        raise ValueError(f"assigned must be 1-D or 2-D, got {arr.ndim}-D")
    out = np.zeros(arr.shape[0], dtype=np.int64)
    for j in range(arr.shape[1]):
        out = out * (int(arr[:, j].max()) + 1) + arr[:, j]
    return out


def outcome_tail_labels(returns: np.ndarray, train_returns: np.ndarray,
                        tail: float = 0.05) -> OutcomeLabels:
    """Boolean tail membership from a window's own return (analysis only).

    The threshold is the train ``tail`` quantile of |return|; windows
    at or above it are in the tail. The returned object is marked as
    an outcome label and training entry points refuse it.
    """
    r = np.asarray(returns, dtype=np.float64)
    tr = np.asarray(train_returns, dtype=np.float64)
    if not 0.0 < tail < 0.5:
        raise ValueError(f"tail must be in (0, 0.5), got {tail}")
    thr = float(np.quantile(np.abs(tr), 1.0 - tail))
    labels = np.abs(r) >= thr
    return OutcomeLabels(
        labels=labels, description=f"|window return| top {tail:.0%} "
        f"(train threshold {thr:.6g})")


def independent_episode_mask(t_start: np.ndarray, t_end: np.ndarray,
                             session_ids: np.ndarray,
                             lookback: int = LOOKBACK) -> np.ndarray:
    """Greedy non-overlapping-footprint mask, per session.

    A window's footprint is its lookback plus itself, so consecutive
    kept windows start at least ``lookback + window`` seconds apart.
    Windows are considered in chronological order within each session;
    the first window is always kept.
    """
    ts = np.asarray(t_start, dtype=np.float64)
    te = np.asarray(t_end, dtype=np.float64)
    ss = np.asarray(session_ids)
    if not (ts.shape == te.shape == ss.shape):
        raise ValueError("t_start, t_end and session_ids must agree")
    keep = np.zeros(len(ts), dtype=bool)
    for sid in np.unique(ss):
        idx = np.where(ss == sid)[0]
        idx = idx[np.argsort(ts[idx], kind="stable")]
        last_end = -np.inf
        for i in idx:
            if ts[i] - lookback >= last_end:
                keep[i] = True
                last_end = te[i]
    return keep


def stratum_counts(labels_1d: np.ndarray, n_bins: int,
                   independent_mask: np.ndarray | None = None) -> dict:
    """Per-bin window counts: {"all": n, "independent": n}.

    ``independent_mask`` (from :func:`independent_episode_mask`)
    counts non-overlapping episodes; without it the independent count
    equals the raw count.
    """
    labels = validate_training_labels(labels_1d)
    if labels.min() < 0 or labels.max() >= n_bins:
        raise ValueError(
            f"labels outside [0, {n_bins}): min {labels.min()}, "
            f"max {labels.max()}")
    counts: dict[int, dict[str, int]] = {}
    for b in range(n_bins):
        sel = labels == b
        counts[b] = {"all": int(sel.sum()),
                     "independent": int(sel.sum())}
    if independent_mask is not None:
        mask = np.asarray(independent_mask, dtype=bool)
        if mask.shape != labels.shape:
            raise ValueError("independent_mask shape disagrees with labels")
        for b in range(n_bins):
            counts[b]["independent"] = int((labels[mask] == b).sum())
    return counts


def check_sufficient(counts: dict, min_independent: int,
                     split_name: str) -> None:
    """Raise NotEnoughDataError naming every deficient stratum."""
    bad = [(b, c["independent"]) for b, c in sorted(counts.items())
           if c["independent"] < min_independent]
    if bad:
        detail = ", ".join(f"stratum {b} has {n} independent episodes"
                           for b, n in bad)
        raise NotEnoughDataError(
            f"split {split_name!r}: not enough data: {detail} "
            f"(minimum {min_independent}); refusing to report")


def validate_training_labels(labels) -> np.ndarray:
    """Validate a training label array; outcome labels are refused."""
    if isinstance(labels, OutcomeLabels):
        raise TypeError(
            "outcome labels (derived from the window itself) must "
            "never condition training or sampling; pass regime labels "
            "from assign_strata instead")
    arr = np.asarray(labels)
    if arr.dtype == bool:
        raise TypeError("training labels must be integer bin indices, "
                        "not booleans")
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            f"training labels must be a 1-D integer array, got "
            f"dtype {arr.dtype} shape {arr.shape}")
    return arr.astype(np.int64)


def capped_importance_weights(labels_1d, n_bins: int,
                              cap: float = DEFAULT_WEIGHT_CAP
                              ) -> tuple[np.ndarray, float]:
    """Inverse-frequency weights, capped and renormalized to mean 1.

    Raw weight for a window in bin ``b`` is ``n / (n_bins * count_b)``;
    weights are capped at ``cap`` (after normalizing to mean 1) and
    renormalized to mean 1. Returns (weights, effective sample size).
    """
    labels = validate_training_labels(labels_1d)
    n = len(labels)
    if n == 0:
        raise ValueError("no labels to weight")
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    counts = np.bincount(labels, minlength=n_bins).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(
            "cannot weight an empty stratum: every bin in [0, n_bins) "
            "must hold at least one window")
    w = n / (n_bins * counts[labels])
    w = w / w.mean()
    w = np.minimum(w, cap)
    w = w / w.mean()
    return w.astype(np.float64), effective_sample_size(w)


def effective_sample_size(weights: np.ndarray) -> float:
    """Effective sample size: (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("weights must be a non-empty 1-D array")
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative")
    denom = float((w ** 2).sum())
    if denom <= 0:
        raise ValueError("weights are all zero")
    return float(w.sum() ** 2 / denom)


def stratified_minibatch_indices(labels_1d, n_bins: int, batch_size: int,
                                 n_batches: int,
                                 seed: int) -> list[np.ndarray]:
    """Round-robin stratified minibatches at equal compute.

    Each batch draws as evenly as possible from every stratum (bin
    counts differ by at most one), cycling through each bin's shuffled
    index list. ``n_batches`` is caller-chosen: pass
    ``ceil(n / batch_size)`` and the epoch costs exactly as many
    optimizer steps as uniform sampling. Deterministic for a seed.
    """
    labels = validate_training_labels(labels_1d)
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_batches < 1:
        raise ValueError(f"n_batches must be positive, got {n_batches}")
    rng = np.random.default_rng(seed)
    pools: list[list[int]] = []
    for b in range(n_bins):
        idx = np.where(labels == b)[0]
        if len(idx) == 0:
            raise ValueError(f"stratum {b} is empty: cannot stratify")
        pools.append(list(rng.permutation(idx)))
    pos = [0] * n_bins
    per_bin = [batch_size // n_bins] * n_bins
    for j in range(batch_size % n_bins):
        per_bin[j] += 1
    batches: list[np.ndarray] = []
    for _ in range(n_batches):
        batch: list[int] = []
        for b in range(n_bins):
            for _ in range(per_bin[b]):
                if pos[b] >= len(pools[b]):
                    pools[b] = list(rng.permutation(
                        np.where(labels == b)[0]))
                    pos[b] = 0
                batch.append(pools[b][pos[b]])
                pos[b] += 1
        batches.append(np.array(batch, dtype=np.int64))
    return batches


def _nn_distances(a: np.ndarray, b: np.ndarray, seed: int,
                  max_n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour distances between flattened window sets.

    Subsamples each set to at most ``max_n`` rows with a fixed seed so
    the report stays fast on thousands of windows. Returns
    (a-to-b distances, b-to-b distances excluding self).
    """
    rng = np.random.default_rng(seed)
    af = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    bf = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    if len(af) > max_n:
        af = af[rng.choice(len(af), max_n, replace=False)]
    if len(bf) > max_n:
        bf = bf[rng.choice(len(bf), max_n, replace=False)]
    d2 = ((af[:, None, :] - bf[None, :, :]) ** 2).sum(axis=2)
    a_to_b = np.sqrt(d2.min(axis=1))
    d2_bb = ((bf[:, None, :] - bf[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2_bb, np.inf)
    b_to_b = np.sqrt(d2_bb.min(axis=1))
    return a_to_b, b_to_b


def stratified_report(real: np.ndarray, gen: np.ndarray,
                      real_labels, gen_labels=None,
                      feature_name: str = "past_vol",
                      t_start: np.ndarray | None = None,
                      t_end: np.ndarray | None = None,
                      session_ids: np.ndarray | None = None,
                      min_independent: int = DEFAULT_MIN_INDEPENDENT,
                      seed: int = 0) -> dict:
    """Per-stratum fidelity: mean W1, coverage, and diversity.

    ``gen`` rows must align row-for-row with ``real`` (generated
    windows conditioned on the matching real regimes), so generated
    windows inherit the real window's stratum. Per stratum:

    - ``mean_w1``: marginal Wasserstein distance averaged over the
      seven features (same per-feature W1 as marginal_table).
    - ``coverage``: fraction of real windows whose nearest generated
      neighbour lies within the median real-to-real nearest-neighbour
      distance in that stratum.
    - ``diversity_ratio``: mean generated nearest-neighbour distance
      divided by the mean real one (near 1 means the generator spreads
      like the real data in that stratum).

    ``average_*`` weights strata by their real window counts;
    ``worst_*`` is the worst stratum (highest W1, lowest coverage).
    Any stratum below ``min_independent`` independent episodes in the
    real set raises NotEnoughDataError instead of reporting.
    """
    from .evaluate import marginal_table

    real_labels = validate_training_labels(real_labels)
    gen_labels = real_labels if gen_labels is None \
        else validate_training_labels(gen_labels)
    real = np.asarray(real)
    gen = np.asarray(gen)
    if real.shape != gen.shape:
        raise ValueError(
            f"real and gen must align row-for-row, got {real.shape} "
            f"and {gen.shape}")
    if len(real) != len(real_labels) or len(gen) != len(gen_labels):
        raise ValueError("labels must match window counts")
    n_bins = int(real_labels.max()) + 1
    if (t_start is not None) != (t_end is not None) \
            or (t_start is not None) != (session_ids is not None):
        raise ValueError(
            "t_start, t_end and session_ids must be given together")
    mask = None
    if t_start is not None:
        mask = independent_episode_mask(t_start, t_end, session_ids)
    counts = stratum_counts(real_labels, n_bins, independent_mask=mask)
    check_sufficient(counts, min_independent, "real")
    strata: dict[int, dict] = {}
    for b in range(n_bins):
        sel_r = real_labels == b
        sel_g = gen_labels == b
        if not sel_g.any():
            raise NotEnoughDataError(
                f"stratum {b} has no generated windows; refusing to "
                f"report")
        mean_w1 = float(marginal_table(
            real[sel_r], gen[sel_g])["mean_w1"])
        a_to_b, b_to_b = _nn_distances(real[sel_r], gen[sel_g],
                                      seed=seed + b)
        _, r_to_r = _nn_distances(real[sel_r], real[sel_r],
                                  seed=seed + 1000 + b)
        cover = float((a_to_b <= np.median(r_to_r)).mean())
        # Diversity: how spread out the generated windows are (their
        # own nearest-neighbour distances) against the real spread.
        div = float(b_to_b.mean() / r_to_r.mean()) \
            if r_to_r.mean() > 0 else float("nan")
        strata[b] = {
            "n_real": int(sel_r.sum()),
            "n_gen": int(sel_g.sum()),
            "n_real_independent": counts[b]["independent"],
            "mean_w1": mean_w1,
            "coverage": cover,
            "diversity_ratio": div,
        }
    weights = np.array([strata[b]["n_real"] for b in range(n_bins)],
                       dtype=np.float64)
    avg_w1 = float(np.average([strata[b]["mean_w1"]
                               for b in range(n_bins)], weights=weights))
    worst = max(range(n_bins), key=lambda b: strata[b]["mean_w1"])
    avg_cov = float(np.average([strata[b]["coverage"]
                                for b in range(n_bins)], weights=weights))
    worst_cov = min(strata[b]["coverage"] for b in range(n_bins))
    return {
        "feature": feature_name,
        "n_bins": n_bins,
        "min_independent": int(min_independent),
        "strata": strata,
        "average_mean_w1": avg_w1,
        "worst_mean_w1": float(strata[worst]["mean_w1"]),
        "worst_stratum": int(worst),
        "average_coverage": avg_cov,
        "worst_coverage": float(worst_cov),
        "diversity_ratio": float(np.average(
            [strata[b]["diversity_ratio"] for b in range(n_bins)],
            weights=weights)),
    }
