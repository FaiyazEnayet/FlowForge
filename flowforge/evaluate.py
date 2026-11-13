"""Evaluation: do generated windows look like real ones?

Four families of checks, all computed on HELD-OUT data the generator never
saw during training (the test split):

1. Marginal distributions: for each book-state feature, compare real vs
   generated pooled values by mean, std, and the 1-Wasserstein distance.
2. Temporal structure: autocorrelation of signed order flow at lags 1..5,
   real vs generated. Each window is a genuine continuous 60s series, so the
   autocorrelation is computed per window and averaged, with 95% confidence
   intervals. In this dataset the real flow is near flat at these lags; a
   good generator reproduces that.
3. Real-vs-generated discriminator: train a classifier to tell real from
   generated windows. It is trained on the VAL split (real val windows vs
   generated windows conditioned on val regimes) and its AUC is reported on
   the TEST split. Fixed seeds throughout. AUC near 0.5 means the generator
   fools it.
4. TSTR (train on synthetic, test on real): train a small model to predict
   near-term flow direction on generated windows and test it on real test
   windows, next to the same model trained on real windows. This asks whether
   synthetic data is *useful*, a different question from whether it is
   *indistinguishable* (check 3). A generator can pass one and fail the other.

Every metric in this file has a synthetic sanity check in the test suite
with a known answer (e.g. Wasserstein distance of two identical
distributions is ~0). If a check fails, the metric is wrong, not the test.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
import torch.nn as nn

from .data import FEATURE_NAMES, N_FEATURES, WINDOW
from .models import set_seed
from .strata import stratified_report


# ------------------------------------------------- book consistency audit

def inverse_transform(gen: np.ndarray, feature_mean: np.ndarray,
                      feature_std: np.ndarray) -> np.ndarray:
    """Standardized -> physical units: phys = gen * std + mean."""
    return gen * feature_std + feature_mean


def derive_imbalance(log_bid_sz: np.ndarray,
                     log_ask_sz: np.ndarray) -> np.ndarray:
    """Physical imbalance implied by bid/ask sizes.

    imbalance = (bid_sz - ask_sz) / (bid_sz + ask_sz), elementwise.
    Sizes are reconstructed as exp(log_size). Wherever the plain
    exponentials are finite the direct ratio is used, bit for bit as
    before; cells whose sizes overflow float64 fall back to a
    max-shifted form that is algebraically identical, so the result
    is finite and inside [-1, 1] for any finite inputs instead of
    degenerating to inf/inf = nan.
    """
    lb = np.asarray(log_bid_sz, dtype=np.float64)
    la = np.asarray(log_ask_sz, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        bid = np.exp(lb)
        ask = np.exp(la)
        denom = bid + ask
        out = (bid - ask) / np.where(denom > 0, denom, 1.0)
        bad = ~np.isfinite(out)
        if bad.any():
            m = np.maximum(lb, la)
            b2 = np.exp(lb - m)
            a2 = np.exp(la - m)
            out = np.where(bad, (b2 - a2) / (b2 + a2), out)
    return np.clip(out, -1.0, 1.0)


def book_consistency_table(gen: np.ndarray, feature_mean: np.ndarray,
                           feature_std: np.ndarray,
                           tol: float = 0.01) -> dict:
    """Audit physical constraints on generated book-state windows.

    The generator emits log bid size, log ask size, and imbalance as
    independent features, but physically imbalance must equal
    (bid_sz - ask_sz) / (bid_sz + ask_sz). The audit measures the
    RAW generated imbalance against the size-implied value BEFORE
    any clipping: an earlier version clipped the generated imbalance
    to [-1, 1] first, which hid exactly the out-of-range violations
    an audit exists to catch. Alongside the raw residual (mean/max
    absolute error and violation rate over finite cells) it records:

    - the raw imbalance range and the out-of-range rate (|imb| > 1);
    - sign/finite/overflow checks: the fraction of cells holding a
      non-finite feature value, and the fraction whose size
      reconstruction overflows float64;
    - spread nonnegativity on the implied spread in bps;
    - trade-count validity: negative counts (the strict sign rate
      counts counts below -tol, so float32 standardization dust of
      order 1e-8 on a true zero is not misread as a negative count;
      the legacy -0.5 threshold rate is kept alongside) and the
      rate of counts that are not integers (judged against tol
      plus a 1e-5 relative allowance: at extreme magnitudes the
      float32 storage of standardized windows cannot represent
      integer counts to an absolute 0.01);
    - logical coherence: seconds whose rounded trade count is zero
      but whose signed flow is nonzero.

    The audit only reports. It never mutates or repairs its input;
    repair is `constrain_imbalance`, a separate explicit step.

    gen: (n_windows, WINDOW, N_FEATURES) standardized generated windows.
    Returns violation fractions over all (window, timestep) cells.
    """
    phys = inverse_transform(gen, feature_mean, feature_std)
    j_bid = FEATURE_NAMES.index("log_bid_sz")
    j_ask = FEATURE_NAMES.index("log_ask_sz")
    j_imb = FEATURE_NAMES.index("imbalance")
    j_spr = FEATURE_NAMES.index("log_spread_bps")
    j_sf = FEATURE_NAMES.index("signed_flow")
    j_cnt = FEATURE_NAMES.index("log_trade_count")

    with np.errstate(over="ignore", invalid="ignore"):
        raw_imb = phys[:, :, j_imb]
        implied_imb = derive_imbalance(phys[:, :, j_bid],
                                       phys[:, :, j_ask])
        resid = raw_imb - implied_imb
        abs_resid = np.abs(resid[np.isfinite(resid)])
        spread_bps = np.expm1(phys[:, :, j_spr])
        trade_count = np.expm1(phys[:, :, j_cnt])
        count_round = np.round(trade_count)
        size_overflow = ~np.isfinite(np.exp(phys[:, :, j_bid])) \
            | ~np.isfinite(np.exp(phys[:, :, j_ask]))
        nonfinite_cell = ~np.isfinite(phys).all(axis=2)
        coherence_bad = (count_round == 0) \
            & (np.abs(phys[:, :, j_sf]) > tol)
        finite_raw = raw_imb[np.isfinite(raw_imb)]

    return {
        "n_cells": int(resid.size),
        "imbalance_violation_rate":
            float((abs_resid > tol).mean()) if abs_resid.size
            else float("nan"),
        "imbalance_mae":
            float(abs_resid.mean()) if abs_resid.size else float("nan"),
        "imbalance_max_err":
            float(abs_resid.max()) if abs_resid.size else float("nan"),
        "raw_imbalance_min":
            float(finite_raw.min()) if finite_raw.size else float("nan"),
        "raw_imbalance_max":
            float(finite_raw.max()) if finite_raw.size else float("nan"),
        "imbalance_out_of_range_rate":
            float((np.abs(raw_imb) > 1.0).mean()),
        "nonfinite_rate": float(nonfinite_cell.mean()),
        "overflow_rate": float(size_overflow.mean()),
        "negative_spread_rate": float((spread_bps < 0).mean()),
        "negative_count_rate": float((trade_count < -0.5).mean()),
        "count_negative_rate": float((trade_count < -tol).mean()),
        "count_noninteger_rate":
            float((np.abs(trade_count - count_round)
                   > tol + 1e-5 * np.abs(trade_count)).mean()),
        "coherence_violation_rate": float(coherence_bad.mean()),
        "tol": tol,
    }


def constrain_imbalance(gen: np.ndarray, feature_mean: np.ndarray,
                        feature_std: np.ndarray) -> np.ndarray:
    """Return a copy of gen with imbalance derived from bid/ask sizes.

    The generator's independent imbalance feature is replaced by the
    physical value implied by its own log bid/ask sizes, in standardized
    units. This is constrained decoding: the book is always consistent.
    """
    phys = inverse_transform(gen, feature_mean, feature_std)
    j_bid = FEATURE_NAMES.index("log_bid_sz")
    j_ask = FEATURE_NAMES.index("log_ask_sz")
    j_imb = FEATURE_NAMES.index("imbalance")
    implied = derive_imbalance(phys[:, :, j_bid], phys[:, :, j_ask])
    # Convert back to standardized units for the imbalance feature.
    mean_imb = feature_mean[j_imb]
    std_imb = feature_std[j_imb]
    fixed = gen.copy()
    fixed[:, :, j_imb] = (implied - mean_imb) / std_imb
    return fixed


# ------------------------------------------------------------- basic metrics

def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein distance between two 1D samples (sort-based)."""
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    # compare empirical quantiles on a common grid
    qa = np.quantile(a, np.linspace(0, 1, n))
    qb = np.quantile(b, np.linspace(0, 1, n))
    return float(np.mean(np.abs(qa - qb)))


def acf(x: np.ndarray, lag: int) -> float:
    """Autocorrelation at a lag. Returns nan for constant input."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) <= lag or np.std(x) < 1e-12:
        return float("nan")
    x = x - x.mean()
    denom = np.dot(x, x)
    return float(np.dot(x[: len(x) - lag], x[lag:]) / denom)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic. No sklearn needed."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank all scores, average ranks for ties
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score))
    i = 0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = ranks[y_true == 1].sum()
    u = rank_sum - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


# ------------------------------------------------------- distributional table

def marginal_table(real: np.ndarray, gen: np.ndarray) -> dict:
    """Per-feature real vs generated: mean, std, Wasserstein-1.

    real, gen: (n_windows, WINDOW, N_FEATURES). Features are standardized,
    so distances are in units of train standard deviations.
    """
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        r = real[:, :, j].ravel()
        g = gen[:, :, j].ravel()
        rows.append({
            "feature": name,
            "real_mean": float(r.mean()),
            "gen_mean": float(g.mean()),
            "real_std": float(r.std()),
            "gen_std": float(g.std()),
            "w1": wasserstein_1d(r, g),
        })
    return {"rows": rows,
            "mean_w1": float(np.mean([r["w1"] for r in rows]))}


def _block_bootstrap_ci(x: np.ndarray, block_len: int = 10,
                        n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 0) -> list[float]:
    """Block bootstrap confidence interval for the mean.

    FL6: windows overlap (stride 10 < window 60), so per-window statistics
    are not independent. The normal approximation with n = window count
    understates uncertainty. The moving block bootstrap resamples blocks
    of consecutive windows, preserving short-range dependence.

    x must be in chronological window order. block_len is the tuning
    parameter: too short under-covers, too long adds noise. Report
    sensitivity to it. Returns [lower, upper] for the (1-alpha) interval.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return [float("nan"), float("nan")]
    if block_len < 1:
        block_len = 1
    if block_len > n:
        block_len = n
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n - block_len + 1, size=n_blocks)
        sample = np.concatenate([x[i:i + block_len] for i in idx])[:n]
        boot_means[b] = sample.mean()
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return [lo, hi]


def _mean_ci(x: np.ndarray, z: float = 1.96) -> list[float]:
    """95% confidence interval for the mean (normal approximation).

    DEPRECATED for window-averaged statistics: windows overlap, so the
    effective sample size is smaller than the window count. Use
    _block_bootstrap_ci for those. Kept for backward compatibility and
    for genuinely independent samples.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float("nan"), float("nan")]
    se = x.std(ddof=1) / np.sqrt(len(x))
    m = float(x.mean())
    return [m - z * se, m + z * se]


def flow_acf_table(real: np.ndarray, gen: np.ndarray,
                   lags: tuple[int, ...] = (1, 2, 3, 4, 5),
                   block_len: int = 10) -> dict:
    """Mean autocorrelation of signed order flow, averaged across windows.

    Each window is a genuine continuous series, so the autocorrelation is
    computed per window and averaged. Windows overlap (stride 10 < window
    60), so per-window ACFs are dependent: confidence intervals use the
    moving block bootstrap (FL6), not the normal approximation. The
    bootstrap unit is the window index in chronological order; block_len
    is disclosed and sensitivity should be checked.
    """
    j = FEATURE_NAMES.index("signed_flow")
    rows = []
    for lag in lags:
        r = np.array([acf(w[:, j], lag) for w in real], dtype=np.float64)
        g = np.array([acf(w[:, j], lag) for w in gen], dtype=np.float64)
        r = r[np.isfinite(r)]
        g = g[np.isfinite(g)]
        rows.append({
            "lag": lag,
            "real_acf": float(r.mean()) if len(r) else float("nan"),
            "real_ci": _block_bootstrap_ci(r, block_len=block_len),
            "gen_acf": float(g.mean()) if len(g) else float("nan"),
            "gen_ci": _block_bootstrap_ci(g, block_len=block_len),
            "n_real": int(len(r)),
            "n_gen": int(len(g)),
            "ci_method": f"block_bootstrap_block{block_len}",
        })
    return {"rows": rows}


# ------------------------------------------------------------- discriminator

class DiscMLP(nn.Module):
    def __init__(self, in_dim: int = N_FEATURES * WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1)).squeeze(1)


def discriminator_auc(real_val: np.ndarray, gen_val: np.ndarray,
                      real_test: np.ndarray, gen_test: np.ndarray,
                      seed: int = 0, epochs: int = 30) -> dict:
    """Train on val (real vs generated), report AUC on test.

    The generator never saw val or test, so this measures generalization of
    the generation quality, not memorization. Fixed seed for reproducibility.
    """
    set_seed(seed)

    def prep(r, g):
        x = np.concatenate([r, g]).reshape(len(r) + len(g), -1).astype(np.float32)
        y = np.concatenate([np.ones(len(r)), np.zeros(len(g))]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    xv, yv = prep(real_val, gen_val)
    xt, yt = prep(real_test, gen_test)

    model = DiscMLP(xv.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    n = len(xv)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            loss = bce(model(xv[idx]), yv[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        s_val = torch.sigmoid(model(xv)).numpy()
        s_test = torch.sigmoid(model(xt)).numpy()
    n_pos_test = int((yt.numpy() == 1).sum())
    return {
        "auc_val": auc_score(yv.numpy(), s_val),
        "auc_test": auc_score(yt.numpy(), s_test),
        "n_val": n,
        "n_test": len(xt),
        # raw scores on the test split, for the score-histogram figure.
        # First n_pos_test entries are real windows, the rest generated.
        "test_scores": s_test.astype(np.float64),
        "test_n_real": n_pos_test,
    }


# ------------------------------------------------------------------ TSTR

def _tstr_xy(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-half feature means -> second-half signed-flow direction."""
    j = FEATURE_NAMES.index("signed_flow")
    half = windows.shape[1] // 2
    x = np.asarray(windows[:, :half, :], dtype=np.float64).mean(axis=1)
    y = (np.asarray(windows[:, half:, j], dtype=np.float64).mean(axis=1) > 0)
    return x, y.astype(np.float64)


def tstr_flow_direction(real_val: np.ndarray, gen_val: np.ndarray,
                        real_test: np.ndarray, seed: int = 0,
                        epochs: int = 300) -> dict:
    """Train on synthetic, test on real: is the generated data useful?

    A logistic regression predicts the sign of second-half signed order
    flow from first-half feature means. One copy trains on generated
    windows, the baseline trains on real val windows; both are tested on
    real test windows. Fixed seed. Majority-class accuracy is reported so
    the numbers have a scale.
    """
    xs, ys = _tstr_xy(gen_val)
    xr, yr = _tstr_xy(real_val)
    xt, yt = _tstr_xy(real_test)

    def fit(x, y):
        set_seed(seed)
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-8
        w = torch.zeros(x.shape[1] + 1, dtype=torch.float32,
                        requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.1)
        bce = nn.BCEWithLogitsLoss()
        xb = torch.from_numpy(
            np.column_stack([(x - mu) / sd, np.ones(len(x))]).astype(np.float32))
        yb = torch.from_numpy(y.astype(np.float32))
        for _ in range(epochs):
            opt.zero_grad()
            loss = bce(xb @ w, yb)
            loss.backward()
            opt.step()
        return w.detach().numpy(), mu, sd

    def accuracy(w, mu, sd, x, y):
        s = 1.0 / (1.0 + np.exp(-((x - mu) / sd @ w[:-1] + w[-1])))
        return float(((s > 0.5) == (y > 0.5)).mean())

    out = {
        "n_train_synth": int(len(xs)),
        "n_train_real": int(len(xr)),
        "n_test": int(len(xt)),
        "majority": float(max(yt.mean(), 1.0 - yt.mean())) if len(yt) else float("nan"),
    }
    for name, (x, y) in (("synth", (xs, ys)), ("real", (xr, yr))):
        if len(np.unique(y)) < 2 or len(np.unique(yt)) < 2:
            out[f"acc_{name}"] = float("nan")
        else:
            w, mu, sd = fit(x, y)
            out[f"acc_{name}"] = accuracy(w, mu, sd, xt, yt)
    return out


# --------------------------------------- baseline parity comparison (C03)

def _block_resample_indices(n: int, block_len: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Chronological moving-block resample indices for n positions."""
    if n == 0:
        return np.empty(0, dtype=np.int64)
    block_len = int(min(max(1, block_len), n))
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    return np.concatenate(
        [np.arange(s, s + block_len) for s in starts])[:n]


def _pooled_w1(real: np.ndarray, gen: np.ndarray) -> float:
    """Mean pooled 1-Wasserstein distance across features.

    For equal-size samples this is the exact empirical W1: the mean
    absolute difference of the sorted samples, per feature.
    """
    vals = []
    for j in range(real.shape[2]):
        a = np.sort(np.asarray(real[:, :, j], dtype=np.float64).ravel())
        b = np.sort(np.asarray(gen[:, :, j], dtype=np.float64).ravel())
        vals.append(float(np.mean(np.abs(a - b))))
    return float(np.mean(vals))


def _paired_w1_diffs(real: np.ndarray, gen_by_method: dict,
                     block_len: int, n_boot: int,
                     seed: int) -> dict:
    """Bootstrap W1 differences per method pair.

    Every replicate draws ONE chronological block-resample index vector
    and applies it to the real windows and to every method's generated
    windows, so the pairing is exact: methods are compared on identical
    resampled real/generated blocks.
    """
    methods = sorted(gen_by_method)
    diffs = {(a, b): [] for a, b in combinations(methods, 2)}
    rng = np.random.default_rng(seed)
    n = len(real)
    for _ in range(n_boot):
        idx = _block_resample_indices(n, block_len, rng)
        w1 = {m: _pooled_w1(real[idx], gen_by_method[m][idx])
              for m in methods}
        for pair in diffs:
            diffs[pair].append(w1[pair[0]] - w1[pair[1]])
    return {pair: np.asarray(v) for pair, v in diffs.items()}


def paired_block_bootstrap_w1(real: np.ndarray, gen_by_method: dict,
                              block_lengths=(1, 5, 10, 20),
                              n_boot: int = 200,
                              seeds=(0,)) -> list[dict]:
    """Paired block-bootstrap intervals for W1 differences between
    generators, at several block lengths, pooled over `seeds`."""
    point = {m: marginal_table(real, g)["mean_w1"]
             for m, g in gen_by_method.items()}
    rows = []
    for block_len in block_lengths:
        pooled: dict = {}
        for seed in seeds:
            d = _paired_w1_diffs(real, gen_by_method, block_len,
                                 n_boot, seed + 1000 * block_len)
            for pair, vals in d.items():
                pooled.setdefault(pair, []).extend(vals.tolist())
        for pair, vals in sorted(pooled.items()):
            vals = np.asarray(vals)
            rows.append({
                "block_len": int(block_len),
                "methods": [pair[0], pair[1]],
                "diff": point[pair[0]] - point[pair[1]],
                "ci": [float(np.quantile(vals, 0.025)),
                       float(np.quantile(vals, 0.975))],
                "n_boot": int(len(vals)),
                "seeds": [int(s) for s in seeds],
            })
    return rows


def blocked_auc_interval(scores_real: np.ndarray, scores_gen: np.ndarray,
                         block_lengths=(1, 5, 10, 20), n_boot: int = 200,
                         seed: int = 0) -> dict:
    """Block-bootstrap intervals for a discriminator AUC.

    The real and generated test scores form two chronological groups;
    each replicate resamples moving blocks inside each group (never
    individual windows), preserving the dependence the overlapping
    windows induce.
    """
    sr = np.asarray(scores_real, dtype=np.float64).ravel()
    sg = np.asarray(scores_gen, dtype=np.float64).ravel()
    y = np.concatenate([np.ones(len(sr)), np.zeros(len(sg))])
    auc = auc_score(y, np.concatenate([sr, sg]))
    rows = []
    for block_len in block_lengths:
        rng = np.random.default_rng(seed + 1000 * int(block_len))
        vals = []
        for _ in range(n_boot):
            ir = _block_resample_indices(len(sr), block_len, rng)
            ig = _block_resample_indices(len(sg), block_len, rng)
            yy = np.concatenate([np.ones(len(ir)), np.zeros(len(ig))])
            vals.append(auc_score(yy, np.concatenate([sr[ir], sg[ig]])))
        vals = np.asarray(vals)
        rows.append({"block_len": int(block_len), "auc": float(auc),
                     "ci": [float(np.quantile(vals, 0.025)),
                            float(np.quantile(vals, 0.975))],
                     "n_boot": int(n_boot)})
    return {"rows": rows, "method": "blocked_bootstrap_auc"}


def _tstr_fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                      x_test: np.ndarray, seed: int,
                      epochs: int) -> np.ndarray:
    """Fit the TSTR logistic model and return test class predictions."""
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_test), bool(y_train[0] > 0.5))
    set_seed(seed)
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8
    w = torch.zeros(x_train.shape[1] + 1, dtype=torch.float32,
                    requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.1)
    bce = nn.BCEWithLogitsLoss()
    xb = torch.from_numpy(np.column_stack(
        [(x_train - mu) / sd, np.ones(len(x_train))]).astype(np.float32))
    yb = torch.from_numpy(y_train.astype(np.float32))
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(xb @ w, yb)
        loss.backward()
        opt.step()
    wn = w.detach().numpy()
    s = 1.0 / (1.0 + np.exp(-((x_test - mu) / sd @ wn[:-1] + wn[-1])))
    return s > 0.5


def _paired_tstr_collect(gen_val_by_method: dict, real_val: np.ndarray,
                         real_test: np.ndarray, block_lengths,
                         n_boot: int, seed: int, epochs: int):
    """Per-method TSTR accuracies and paired accuracy-difference draws.

    Every method's classifier is trained and tested under the identical
    protocol (same seed, epochs, real test windows). Paired differences
    resample chronological blocks of the shared test windows, applying
    the same index vector to every method's correctness array.
    """
    xr, yr = _tstr_xy(real_val)
    xt, yt = _tstr_xy(real_test)
    y_test = yt > 0.5
    real_pred = _tstr_fit_predict(xr, yr, xt, seed, epochs)
    real_correct = real_pred == y_test
    majority = (float(max(yt.mean(), 1.0 - yt.mean())) if len(yt)
                else float("nan"))
    stats: dict = {}
    correct: dict = {}
    for name, gv in gen_val_by_method.items():
        xs, ys = _tstr_xy(gv)
        pred = _tstr_fit_predict(xs, ys, xt, seed, epochs)
        correct[name] = (pred == y_test)
        stats[name] = {
            "acc_synth": float(correct[name].mean()) if len(yt)
            else float("nan"),
            "acc_real": float(real_correct.mean()) if len(yt)
            else float("nan"),
            "majority": majority,
            "n_train_synth": int(len(xs)),
            "n_train_real": int(len(xr)),
            "n_test": int(len(xt)),
        }
    diffs: dict = {}
    for block_len in block_lengths:
        rng = np.random.default_rng(seed + 1000 * int(block_len))
        for _ in range(n_boot):
            idx = _block_resample_indices(len(yt), block_len, rng)
            for a, b in combinations(sorted(correct), 2):
                diffs.setdefault((int(block_len), (a, b)), []).append(
                    float(correct[a][idx].mean() - correct[b][idx].mean()))
    return stats, diffs


def _ci_rows(diffs: dict, point_diffs: dict) -> list[dict]:
    rows = []
    for (block_len, pair), vals in sorted(diffs.items()):
        vals = np.asarray(vals, dtype=np.float64)
        rows.append({
            "block_len": int(block_len),
            "methods": [pair[0], pair[1]],
            "diff": point_diffs[pair],
            "ci": [float(np.quantile(vals, 0.025)),
                   float(np.quantile(vals, 0.975))],
            "n_boot": int(len(vals)),
        })
    return rows


COMPARISON_SCHEMA = "flowforge-baseline-parity/1"
COMPARISON_REQUIRED_METHODS = ("ddpm", "vae", "chunk_copy")
COMPARISON_COST_KEYS = ("sample_seconds", "sample_seconds_per_1k",
                        "peak_memory_mb", "nfe")


def comparison_report(real_test: np.ndarray, real_val: np.ndarray,
                      runs: list[dict],
                      block_lengths=(1, 5, 10, 20), n_boot: int = 100,
                      disc_epochs: int = 30, tstr_epochs: int = 300,
                      protocol_seed: int = 0,
                      n_test_sessions: int | None = None) -> dict:
    """Baseline parity report: every method under identical conditions.

    `runs` is a list of {method, kind, seed, gen_val, gen_test, cost}
    records. The published schema requires DDPM, VAE, and the
    regime-matched chunk-copy baseline; further methods (VAR, the
    conditional AR comparator) are optional. Every method must report
    the same seed set and the same val/test window counts, every run
    must carry latency, throughput, peak-memory, and NFE metadata, and
    every method is scored with the identical held-out discriminator
    protocol (same architecture, epochs, and protocol seed, fit on val,
    scored on test) and the identical TSTR protocol.

    Uncertainty is dependence-aware: W1 differences use paired
    chronological block resamples shared across methods, AUC intervals
    resample blocks inside the real and generated score groups, and
    TSTR differences are paired over blocks of the shared test windows.
    With one test session the intervals are within-session rather than
    across-day evidence, and the report says so.
    """
    if not runs:
        raise ValueError("comparison requires at least one run")
    by_method: dict[str, list[dict]] = {}
    for run in runs:
        for key in COMPARISON_COST_KEYS:
            if key not in run.get("cost", {}):
                raise ValueError(
                    f"run for {run.get('method')!r} lacks cost metadata "
                    f"{key!r}: no reported result may lack latency, NFE, "
                    f"or cost metadata")
        if run["gen_val"].shape != real_val.shape \
                or run["gen_test"].shape != real_test.shape:
            raise ValueError(
                f"run for {run['method']!r} does not match the shared "
                f"val/test window counts")
        by_method.setdefault(run["method"], []).append(run)
    missing = [m for m in COMPARISON_REQUIRED_METHODS if m not in by_method]
    if missing:
        raise ValueError(
            f"published metrics schema requires DDPM + VAE + chunk-copy "
            f"results; missing {missing}")
    seed_sets = {m: sorted(r["seed"] for r in rs)
                 for m, rs in by_method.items()}
    if len({tuple(s) for s in seed_sets.values()}) != 1:
        raise ValueError(
            f"all methods must be evaluated on the same seed set, got "
            f"{seed_sets}")
    seeds = next(iter(seed_sets.values()))
    for m, rs in by_method.items():
        kinds = {r["kind"] for r in rs}
        if len(kinds) != 1:
            raise ValueError(f"method {m!r} mixes kinds {kinds}")

    methods: dict = {}
    for name, rs in by_method.items():
        entry = {"kind": rs[0]["kind"], "runs": []}
        for run in sorted(rs, key=lambda r: r["seed"]):
            mt = marginal_table(real_test, run["gen_test"])
            disc = discriminator_auc(real_val, run["gen_val"],
                                     real_test, run["gen_test"],
                                     seed=protocol_seed,
                                     epochs=disc_epochs)
            scores = np.asarray(disc.pop("test_scores"), dtype=np.float64)
            n_real = int(disc.pop("test_n_real"))
            disc["auc_ci"] = blocked_auc_interval(
                scores[:n_real], scores[n_real:],
                block_lengths=block_lengths, n_boot=n_boot,
                seed=protocol_seed)
            entry["runs"].append({
                "seed": int(run["seed"]),
                "mean_w1": mt["mean_w1"],
                "marginals": mt,
                "discriminator": disc,
                "cost": run["cost"],
            })
        entry["mean_w1"] = float(np.mean([r["mean_w1"]
                                          for r in entry["runs"]]))
        entry["auc_test_mean"] = float(np.mean(
            [r["discriminator"]["auc_test"] for r in entry["runs"]]))
        methods[name] = entry

    # paired W1 and TSTR, pooled over the repeated generation seeds
    w1_diffs: dict = {}
    w1_point = {m: methods[m]["mean_w1"] for m in methods}
    tstr_diffs: dict = {}
    tstr_point: dict = {}
    run_by = {(r["method"], r["seed"]): r for r in runs}
    for seed in seeds:
        gens = {m: run_by[(m, seed)]["gen_test"] for m in by_method}
        for block_len in block_lengths:
            d = _paired_w1_diffs(real_test, gens, block_len, n_boot,
                                 protocol_seed + 1000 * int(block_len))
            for pair, vals in d.items():
                w1_diffs.setdefault((int(block_len), pair),
                                    []).extend(vals.tolist())
        gen_vals = {m: run_by[(m, seed)]["gen_val"] for m in by_method}
        stats, d = _paired_tstr_collect(
            gen_vals, real_val, real_test, block_lengths, n_boot,
            protocol_seed, tstr_epochs)
        for key, vals in d.items():
            tstr_diffs.setdefault(key, []).extend(vals)
        for m, s in stats.items():
            entry_run = next(r for r in methods[m]["runs"]
                             if r["seed"] == seed)
            entry_run["tstr"] = s
            tstr_point[m] = s["acc_synth"]
    w1_rows = _ci_rows(w1_diffs,
                       {pair: w1_point[pair[0]] - w1_point[pair[1]]
                        for (_, pair) in w1_diffs})
    tstr_rows = _ci_rows(tstr_diffs,
                         {pair: tstr_point[pair[0]] - tstr_point[pair[1]]
                          for (_, pair) in tstr_diffs})
    for name, entry in methods.items():
        entry["tstr_acc_synth_mean"] = float(np.mean(
            [r["tstr"]["acc_synth"] for r in entry["runs"]]))

    if n_test_sessions is None:
        n_test_sessions = 1
    if n_test_sessions == 1:
        ci_scope = "within-session rather than across-day evidence"
        ci_scope_note = (
            "All confidence intervals in this report are within-session "
            "rather than across-day evidence: the test split contains a "
            "single session, so the intervals quantify resampling "
            "variability inside that session and say nothing about "
            "variation across days.")
    else:
        ci_scope = "across-session evidence"
        ci_scope_note = (
            f"Intervals resample blocks within each of the "
            f"{n_test_sessions} test sessions and pool them; they are "
            f"across-session evidence about the recorded sessions, not "
            f"about unrecorded days.")
    return {
        "schema": COMPARISON_SCHEMA,
        "schema_version": 1,
        "methods": methods,
        "paired_w1": w1_rows,
        "paired_tstr": tstr_rows,
        "block_lengths": [int(b) for b in block_lengths],
        "seeds": [int(s) for s in seeds],
        "discriminator_protocol": {
            "seed": int(protocol_seed), "epochs": int(disc_epochs),
            "fit_split": "val", "score_split": "test",
            "note": "identical protocol for every method and seed",
        },
        "n_val": int(len(real_val)),
        "n_test": int(len(real_test)),
        "n_test_sessions": int(n_test_sessions),
        "ci_scope": ci_scope,
        "ci_scope_note": ci_scope_note,
    }


def print_comparison_report(rep: dict) -> None:
    print("\n=== baseline parity (identical test conditions, "
          "repeated seeds) ===")
    print(f"seeds: {rep['seeds']} | block lengths: {rep['block_lengths']}")
    for name, m in rep["methods"].items():
        print(f"{name:<14} kind={m['kind']:<13} "
              f"mean W1 {m['mean_w1']:.3f} | "
              f"disc AUC {m['auc_test_mean']:.3f} | "
              f"TSTR acc {m['tstr_acc_synth_mean']:.3f}")
    print("paired W1 differences (first minus second):")
    for r in rep["paired_w1"]:
        print(f"  block {r['block_len']:>2}: {r['methods'][0]} - "
              f"{r['methods'][1]}: {r['diff']:+.3f} "
              f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
    print(rep["ci_scope_note"])


def full_report(real_test: np.ndarray, gen_test: np.ndarray,
                real_val: np.ndarray, gen_val: np.ndarray,
                seed: int = 0,
                feature_mean: np.ndarray = None,
                feature_std: np.ndarray = None,
                stratum_labels=None,
                feature_name: str = "past_vol") -> dict:
    rep = {
        "marginals": marginal_table(real_test, gen_test),
        "flow_acf": flow_acf_table(real_test, gen_test),
        "discriminator": discriminator_auc(real_val, gen_val, real_test, gen_test,
                                           seed=seed),
        "tstr": tstr_flow_direction(real_val, gen_val, real_test, seed=seed),
    }
    if stratum_labels is not None:
        # P07: per-stratum fidelity on the same test windows. The
        # generated rows align with the real rows (same regimes), so
        # they inherit the real window's stratum. A scarce stratum
        # raises NotEnoughDataError here instead of reporting.
        rep["strata"] = stratified_report(
            real_test, gen_test, stratum_labels,
            feature_name=feature_name)
    if feature_mean is not None and feature_std is not None:
        pre = book_consistency_table(gen_test, feature_mean, feature_std)
        fixed = constrain_imbalance(gen_test, feature_mean, feature_std)
        fixed_val = constrain_imbalance(gen_val, feature_mean,
                                        feature_std)
        post = book_consistency_table(fixed, feature_mean, feature_std)
        # downstream fidelity: does the constraint fix move the marginals?
        pre_w1 = marginal_table(real_test, gen_test)["mean_w1"]
        con_marginals = marginal_table(real_test, fixed)
        post_w1 = con_marginals["mean_w1"]
        rep["book_consistency"] = {
            "pre": pre,
            "post": post,
            "constrained": True,
            "marginal_w1_pre": pre_w1,
            "marginal_w1_post": post_w1,
        }
        # Both sampling modes, one identical protocol: the constrained
        # windows go through the same marginals, flow ACF, held-out
        # discriminator (same seed and splits), and TSTR as the raw
        # windows above, and the postprocessing quality loss is the
        # signed difference, reported whichever way it lands.
        con_disc = dict(discriminator_auc(real_val, fixed_val,
                                          real_test, fixed, seed=seed))
        con_disc.pop("test_scores", None)  # keep the bundle JSON-safe
        con_disc.pop("test_n_real", None)
        con_tstr = tstr_flow_direction(real_val, fixed_val, real_test,
                                       seed=seed)
        rep["constrained"] = {
            "marginals": con_marginals,
            "flow_acf": flow_acf_table(real_test, fixed),
            "discriminator": con_disc,
            "tstr": con_tstr,
        }
        rep["postprocessing_quality_loss"] = {
            "mean_w1_delta": post_w1 - pre_w1,
            "discriminator_auc_test_delta":
                con_disc["auc_test"] - rep["discriminator"]["auc_test"],
            "tstr_acc_synth_delta":
                con_tstr["acc_synth"] - rep["tstr"]["acc_synth"],
            "note": ("constrained minus raw under the identical "
                     "held-out protocol; a positive W1 or AUC delta "
                     "means the constraint cost fidelity, a negative "
                     "delta means it helped, zero means no measurable "
                     "effect. Reported as measured, including when "
                     "adverse or null."),
        }
    return rep


def print_report(rep: dict) -> None:
    print("\n=== marginal distributions (real vs generated, test split) ===")
    print(f"{'feature':<16}{'real_mean':>10}{'gen_mean':>10}"
          f"{'real_std':>10}{'gen_std':>10}{'w1':>10}")
    for r in rep["marginals"]["rows"]:
        print(f"{r['feature']:<16}{r['real_mean']:>10.3f}{r['gen_mean']:>10.3f}"
              f"{r['real_std']:>10.3f}{r['gen_std']:>10.3f}{r['w1']:>10.3f}")
    print(f"mean W1 across features: {rep['marginals']['mean_w1']:.3f}")
    print("\n=== signed order flow autocorrelation "
          "(mean across windows, 95% CI) ===")
    for r in rep["flow_acf"]["rows"]:
        print(f"lag {r['lag']}: real {r['real_acf']:.3f} "
              f"[{r['real_ci'][0]:.3f}, {r['real_ci'][1]:.3f}] (n={r['n_real']})"
              f" | gen {r['gen_acf']:.3f} "
              f"[{r['gen_ci'][0]:.3f}, {r['gen_ci'][1]:.3f}] (n={r['n_gen']})")
    d = rep["discriminator"]
    print("\n=== real-vs-generated discriminator "
          "(trained on val, tested on test) ===")
    print(f"AUC on val: {d['auc_val']:.3f} | AUC on test: {d['auc_test']:.3f} "
          f"(0.5 = indistinguishable)")
    print(f"(n_val={d['n_val']}, n_test={d['n_test']}, seed fixed)")
    t = rep["tstr"]
    print("\n=== TSTR: train on synthetic, test on real "
          "(flow direction) ===")
    print(f"synthetic-trained accuracy: {t['acc_synth']:.3f} | "
          f"real-trained accuracy: {t['acc_real']:.3f} | "
          f"majority baseline: {t['majority']:.3f}")
    print(f"(n_train_synth={t['n_train_synth']}, "
          f"n_train_real={t['n_train_real']}, n_test={t['n_test']}, "
          f"seed fixed)")
    if "book_consistency" in rep:
        bc = rep["book_consistency"]
        print("\n=== book consistency audit (raw, before any repair) ===")
        for name in ("pre", "post"):
            r = bc[name]
            print(f"{name}: imbalance violation "
                  f"{r['imbalance_violation_rate']:.3f} "
                  f"(raw MAE {r['imbalance_mae']:.4f}, raw range "
                  f"[{r['raw_imbalance_min']:.3f}, "
                  f"{r['raw_imbalance_max']:.3f}]), negative spread "
                  f"{r['negative_spread_rate']:.3f}, non-integer count "
                  f"{r['count_noninteger_rate']:.3f}, coherence "
                  f"{r['coherence_violation_rate']:.3f}")
        print(f"marginal W1 pre {bc['marginal_w1_pre']:.3f} -> "
              f"post {bc['marginal_w1_post']:.3f}")
    if "postprocessing_quality_loss" in rep:
        ql = rep["postprocessing_quality_loss"]
        print("\n=== postprocessing quality loss (constrained - raw) ===")
        print(f"mean W1 {ql['mean_w1_delta']:+.4f} | discriminator AUC "
              f"{ql['discriminator_auc_test_delta']:+.4f} | TSTR acc "
              f"{ql['tstr_acc_synth_delta']:+.4f}")
    if "strata" in rep:
        st = rep["strata"]
        print(f"\n=== per-stratum fidelity ({st['feature']}, test split) ===")
        for b, s in st["strata"].items():
            print(f"stratum {b}: n_real={s['n_real']} "
                  f"(indep {s['n_real_independent']}) mean W1 "
                  f"{s['mean_w1']:.3f} | coverage {s['coverage']:.3f} | "
                  f"diversity {s['diversity_ratio']:.3f}")
        print(f"average mean W1 {st['average_mean_w1']:.3f} | worst "
              f"(stratum {st['worst_stratum']}) "
              f"{st['worst_mean_w1']:.3f} | average coverage "
              f"{st['average_coverage']:.3f} | worst coverage "
              f"{st['worst_coverage']:.3f}")
