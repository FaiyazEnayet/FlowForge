"""Microstructure response diagnostics: real vs generated book dynamics.

The marginal, autocorrelation, discriminator, and TSTR checks in
``evaluate`` ask whether generated windows look alike and whether they
are useful. These diagnostics ask a different question: does the
generated market *respond* like the real one? Each diagnostic is
computed on held-out real windows and generated windows at the same
temporal resolution (1-second bars, 60-second windows), next to a
shuffled null baseline (within-window time shuffle: marginals
preserved, temporal order destroyed) and block-bootstrap intervals
over windows.

Naming discipline, stated once here and enforced by the tests: the
``signed_flow`` feature is observed trade signed volume (buy
positive). It is NOT order-flow imbalance as defined by Cont,
Kukanov, and Stoikov, which also counts quote additions and
cancellations at the touch; FlowForge bars carry no quote-event
stream, so signed flow must never be labeled OFI. The only OFI-like
quantity here is :func:`ofi_proxy`, derived from quote transitions
with its assumptions stated in its docstring and in the report.

Diagnostics:
- signed-flow vs subsequent return response (correlation and slope)
- lagged imbalance vs return response
- joint lead-lag cross-correlations across flow, imbalance, return,
  spread, and trade count
- spread and size transition rates, plus a spread tercile matrix
- return-tail quantiles and exceedance rates
- realized volatility clustering (autocorrelation of |return| and
  squared return)
- trade intensity intermittency (zero-trade fraction, dispersion,
  count autocorrelation)
- regime-conditional response (past-volatility / past-spread regimes)
- OFI-proxy vs subsequent return response (assumption-laden, see
  :func:`ofi_proxy`)

Every function is pure NumPy over (n_windows, WINDOW) physical
arrays. Confidence intervals resample blocks of windows, because
windows overlap (stride 10 < window 60) and per-window statistics
are dependent.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .data import FEATURE_NAMES, WINDOW

RESPONSE_LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
CROSS_LAG_MAX = 5
TAIL_QUANTILES = (0.001, 0.01, 0.05, 0.95, 0.99, 0.999)
EXCEEDANCE_SIGMAS = (2.0, 3.0)
CLUSTERING_LAGS = (1, 2, 3, 4, 5, 10)
COUNT_ACF_LAGS = (1, 2, 3, 4, 5)

# The five series the joint lead-lag table covers, in feature order.
CROSS_SERIES = ("signed_flow", "imbalance", "log_ret", "log_spread_bps",
                "log_trade_count")


# ------------------------------------------------------------ small helpers

def _block_bootstrap_ci_matrix(x: np.ndarray, block_len: int = 10,
                               n_boot: int = 200,
                               seed: int = 0) -> np.ndarray:
    """Block-bootstrap 95% CI for the mean of each column of x.

    x: (n_windows, k) per-window statistics in chronological window
    order. Returns (k, 2) [lower, upper]. Moving blocks of consecutive
    windows preserve the dependence overlapping windows induce.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 2:
        return np.full((k, 2), np.nan)
    block_len = int(min(max(1, block_len), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    means = np.empty((n_boot, k))
    for b in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        idx = np.concatenate(
            [np.arange(s, s + block_len) for s in starts])[:n]
        means[b] = x[idx].mean(axis=0)
    lo = np.quantile(means, 0.025, axis=0)
    hi = np.quantile(means, 0.975, axis=0)
    return np.column_stack([lo, hi])


def _paired_segments(a: np.ndarray, b: np.ndarray, lag: int,
                     a_offset: int = 0):
    """Aligned (x, y) views pairing a[t] with b[t + lag].

    a[:, i] sits at second i + a_offset; b[:, j] at second j. Returns
    None when fewer than two aligned pairs exist.
    """
    ta = a.shape[1]
    tb = b.shape[1]
    i0 = max(0, -a_offset - lag)
    i1 = min(ta, tb - a_offset - lag)
    if i1 - i0 <= 1:
        return None
    j0 = i0 + a_offset + lag
    return a[:, i0:i1], b[:, j0:j0 + (i1 - i0)]


def _per_window_corr(a: np.ndarray, b: np.ndarray, lag: int,
                     a_offset: int = 0) -> np.ndarray:
    """Per-window Pearson correlation of a[t] with b[t + lag].

    lag may be negative (b leads a). Windows with a constant segment
    return nan. Returns (n_windows,).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    seg = _paired_segments(a, b, lag, a_offset)
    if seg is None:
        return np.full(a.shape[0], np.nan)
    x, y = seg
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (x * y).sum(axis=1) / denom
    return np.where(denom < 1e-12, np.nan, out)


def _per_window_slope(a: np.ndarray, b: np.ndarray, lag: int,
                      a_offset: int = 0) -> np.ndarray:
    """Per-window OLS slope of b[t + lag] on a[t]. See _per_window_corr."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    seg = _paired_segments(a, b, lag, a_offset)
    if seg is None:
        return np.full(a.shape[0], np.nan)
    x, y = seg
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = (x * x).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (x * y).sum(axis=1) / denom
    return np.where(denom < 1e-12, np.nan, out)


def _acf_per_window(x: np.ndarray, lag: int) -> np.ndarray:
    """Per-window autocorrelation at a lag. nan for constant windows."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] <= lag:
        return np.full(x.shape[0], np.nan)
    xc = x - x.mean(axis=1, keepdims=True)
    denom = (xc * xc).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (xc[:, :-lag] * xc[:, lag:]).sum(axis=1) / denom
    return np.where(denom < 1e-12, np.nan, out)


def _summarize(per_window: np.ndarray, block_len: int, n_boot: int,
               seed: int) -> tuple[float, list[float], int]:
    """Mean, block-bootstrap CI, and finite count of per-window stats."""
    vals = np.asarray(per_window, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), [float("nan"), float("nan")], 0
    ci = _block_bootstrap_ci_matrix(vals[:, None], block_len, n_boot,
                                    seed)[0]
    return float(vals.mean()), [float(ci[0]), float(ci[1])], int(len(vals))


# ------------------------------------------------------------ physical units

def physical_series(windows: np.ndarray, feature_mean: np.ndarray,
                    feature_std: np.ndarray) -> dict:
    """Standardized windows -> physical per-second series.

    Returns a dict of (n_windows, WINDOW) arrays: log returns as-is,
    spread in bps, bid/ask sizes, imbalance, signed trade volume, and
    trade counts. The signed-flow entry is labeled ``signed_flow``
    and is trade signed volume, not OFI; see the module docstring.
    """
    phys = np.asarray(windows, dtype=np.float64) * feature_std \
        + feature_mean
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    with np.errstate(over="ignore", invalid="ignore"):
        out = {
            "log_ret": phys[:, :, idx["log_ret"]],
            "spread_bps": np.expm1(phys[:, :, idx["log_spread_bps"]]),
            "bid_sz": np.exp(phys[:, :, idx["log_bid_sz"]]),
            "ask_sz": np.exp(phys[:, :, idx["log_ask_sz"]]),
            "imbalance": phys[:, :, idx["imbalance"]],
            "signed_flow": phys[:, :, idx["signed_flow"]],
            "trade_count": np.expm1(phys[:, :, idx["log_trade_count"]]),
        }
    return out


# ------------------------------------------------------- response curves

def response_curve(driver: np.ndarray, target: np.ndarray,
                   lags: tuple[int, ...] = RESPONSE_LAGS,
                   block_len: int = 10, n_boot: int = 200, seed: int = 0,
                   driver_offset: int = 0) -> dict:
    """How a driver series moves the target `lag` seconds later.

    For each lag, the mean across windows of the per-window Pearson
    correlation and OLS slope of target[t + lag] on driver[t]. The
    peak lag is the lag with the largest absolute mean correlation.
    ``driver_offset`` is the second the driver's first column sits
    at (0 for a full window, 1 for a first-difference series such as
    the OFI proxy).
    """
    rows = []
    for lag in lags:
        corr = _per_window_corr(driver, target, lag, driver_offset)
        slope = _per_window_slope(driver, target, lag, driver_offset)
        m_corr, ci_corr, n = _summarize(corr, block_len, n_boot, seed)
        m_slope, ci_slope, _ = _summarize(slope, block_len, n_boot,
                                          seed + 7)
        rows.append({
            "lag": int(lag),
            "corr": m_corr,
            "corr_ci": ci_corr,
            "slope": m_slope,
            "slope_ci": ci_slope,
            "n_windows": n,
        })
    peak = max(rows, key=lambda r: abs(r["corr"]))["lag"] if rows else None
    return {"rows": rows, "peak_lag": peak}


def cross_correlation_table(series: dict, max_lag: int = CROSS_LAG_MAX,
                            block_len: int = 10, n_boot: int = 200,
                            seed: int = 0) -> dict:
    """Joint lead-lag cross-correlations for every unordered pair.

    For pair (a, b) and lag h, the mean per-window correlation of
    a[t] with b[t + h]; negative h means b leads a. Each pair is
    listed once, with lags -max_lag..max_lag.
    """
    names = [name for name in series if name in CROSS_SERIES]
    names = sorted(names, key=CROSS_SERIES.index)
    rows = []
    for a_name, b_name in combinations(names, 2):
        a = np.asarray(series[a_name], dtype=np.float64)
        b = np.asarray(series[b_name], dtype=np.float64)
        for lag in range(-max_lag, max_lag + 1):
            corr = _per_window_corr(a, b, lag)
            m, ci, n = _summarize(corr, block_len, n_boot, seed)
            rows.append({
                "series_a": a_name,
                "series_b": b_name,
                "lag": int(lag),
                "corr": m,
                "ci": ci,
                "n_windows": n,
            })
    return {"rows": rows, "max_lag": int(max_lag)}


# ------------------------------------------------------------- transitions

def directional_transition_rates(series: np.ndarray,
                                 tol: float = 0.0) -> dict:
    """Pooled per-second up / down / flat rates of a series.

    Transitions are consecutive-second differences inside each window
    (never across windows). A difference above +tol is up, below -tol
    is down, otherwise flat. Rates sum to 1.
    """
    s = np.asarray(series, dtype=np.float64)
    d = np.diff(s, axis=1).ravel()
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"up": float("nan"), "down": float("nan"),
                "flat": float("nan"), "n_transitions": 0}
    # each rate is a direct fraction, so they sum to 1 up to rounding
    # and flat is never a negative rounding leftover
    up = float((d > tol).mean())
    down = float((d < -tol).mean())
    flat = float((np.abs(d) <= tol).mean())
    return {"up": up, "down": down, "flat": flat,
            "n_transitions": int(n)}


def tercile_transition_matrix(series: np.ndarray,
                              thresholds: tuple[float, float]) -> dict:
    """3x3 transition matrix between low / mid / high tercile states.

    Thresholds come from the real series (callers pass the real
    pooled terciles so real and generated share the same states).
    Rows are row-normalized; the stationary distribution is the
    pooled state occupancy. Rows with no outgoing transitions are
    zeros.
    """
    s = np.asarray(series, dtype=np.float64)
    lo, hi = thresholds
    finite = np.isfinite(s)
    states = np.full(s.shape, -1, dtype=int)
    states[finite] = np.digitize(s[finite], [lo, hi])  # 0 low, 1 mid, 2 high
    counts = np.zeros((3, 3), dtype=np.float64)
    for w in states:
        for a, b in zip(w[:-1], w[1:]):
            if a >= 0 and b >= 0:
                counts[int(a), int(b)] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(row_sums > 0, counts / np.where(row_sums > 0,
                                                           row_sums, 1.0),
                          0.0)
    total = counts.sum()
    # stationary distribution: fraction of seconds spent in each state.
    occ = np.bincount(states[finite], minlength=3).astype(np.float64)
    stationary = (occ / occ.sum()).tolist() if occ.sum() > 0 \
        else [float("nan")] * 3
    return {"matrix": matrix.tolist(),
            "stationary": [float(v) for v in stationary],
            "thresholds": [float(lo), float(hi)],
            "n_transitions": int(total)}


# ------------------------------------------------------- tails, clustering

def return_tail_table(real_ret: np.ndarray, gen_ret: np.ndarray,
                      quantiles: tuple[float, ...] = TAIL_QUANTILES,
                      thresholds: tuple[float, ...] = EXCEEDANCE_SIGMAS,
                      block_len: int = 10, n_boot: int = 200,
                      seed: int = 0) -> dict:
    """Return-tail quantiles, tail W1, and exceedance rates.

    Quantiles are pooled point values over all seconds; exceedance
    rates (fraction of seconds with |return| beyond k real sigmas)
    carry block-bootstrap intervals from per-window rates.
    """
    r = np.asarray(real_ret, dtype=np.float64).ravel()
    g = np.asarray(gen_ret, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    g = g[np.isfinite(g)]
    rows = []
    for q in quantiles:
        rows.append({
            "q": float(q),
            "real": float(np.quantile(r, q)) if len(r) else float("nan"),
            "gen": float(np.quantile(g, q)) if len(g) else float("nan"),
        })
    tail_w1 = float(np.mean([abs(row["real"] - row["gen"])
                             for row in rows])) if rows else float("nan")
    sigma = float(r.std()) if len(r) else float("nan")
    exc = []
    for k in thresholds:
        thr = k * sigma
        real_rate = float((np.abs(r) > thr).mean()) if len(r) \
            else float("nan")
        gen_rate = float((np.abs(g) > thr).mean()) if len(g) \
            else float("nan")
        _, real_ci, _ = _summarize(
            (np.abs(real_ret) > thr).mean(axis=1), block_len, n_boot,
            seed)
        _, gen_ci, _ = _summarize(
            (np.abs(gen_ret) > thr).mean(axis=1), block_len, n_boot,
            seed + 7)
        exc.append({"threshold": float(k), "sigma": sigma,
                    "real": real_rate, "gen": gen_rate,
                    "real_ci": real_ci, "gen_ci": gen_ci})
    return {"rows": rows, "tail_w1": tail_w1,
            "exceedance": exc, "real_sigma": sigma}


def volatility_clustering(ret: np.ndarray,
                          lags: tuple[int, ...] = CLUSTERING_LAGS,
                          block_len: int = 10, n_boot: int = 200,
                          seed: int = 0) -> dict:
    """Autocorrelation of absolute and squared returns per lag.

    Volatility clustering is persistence in |return| (and return^2):
    a generator that draws returns independently per second scores
    near zero at every lag however good its marginals are.
    """
    ret = np.asarray(ret, dtype=np.float64)
    rows = []
    for lag in lags:
        a = _acf_per_window(np.abs(ret), lag)
        s = _acf_per_window(ret * ret, lag)
        m_a, ci_a, n = _summarize(a, block_len, n_boot, seed)
        m_s, ci_s, _ = _summarize(s, block_len, n_boot, seed + 7)
        rows.append({"lag": int(lag), "abs_acf": m_a, "abs_ci": ci_a,
                     "sq_acf": m_s, "sq_ci": ci_s, "n_windows": n})
    return {"rows": rows}


def trade_intensity_table(counts: np.ndarray,
                          lags: tuple[int, ...] = COUNT_ACF_LAGS,
                          block_len: int = 10, n_boot: int = 200,
                          seed: int = 0) -> dict:
    """Trade intensity intermittency: zeros, dispersion, bursts.

    Zero-trade fraction, mean count, variance-to-mean (Fano) factor,
    coefficient of variation, burstiness (std-mean)/(std+mean) over
    pooled seconds, and the count autocorrelation per lag. A Poisson
    flow has Fano 1 and burstiness near 0; a bursty flow has Fano
    above 1 and positive burstiness.
    """
    c = np.asarray(counts, dtype=np.float64)
    flat = c[np.isfinite(c)].ravel()
    mean = float(flat.mean()) if len(flat) else float("nan")
    var = float(flat.var()) if len(flat) else float("nan")
    sd = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 \
        else float("nan")
    fano = var / mean if mean and mean > 0 else float("nan")
    cv = sd / mean if mean and mean > 0 else float("nan")
    burst = (sd - mean) / (sd + mean) if np.isfinite(sd) and mean \
        and (sd + mean) > 0 else float("nan")
    zero_frac, zero_ci, _ = _summarize((c == 0).mean(axis=1), block_len,
                                       n_boot, seed)
    mean_count, mean_ci, _ = _summarize(c.mean(axis=1), block_len,
                                        n_boot, seed + 7)
    acf_rows = []
    for lag in lags:
        a = _acf_per_window(c, lag)
        m, ci, n = _summarize(a, block_len, n_boot, seed)
        acf_rows.append({"lag": int(lag), "acf": m, "ci": ci,
                         "n_windows": n})
    return {"zero_fraction": zero_frac, "zero_fraction_ci": zero_ci,
            "mean_count": mean_count, "mean_count_ci": mean_ci,
            "pooled_mean_count": mean, "fano_factor": float(fano),
            "cv": float(cv), "burstiness": float(burst),
            "count_acf_rows": acf_rows}


# ------------------------------------------------------------- OFI proxy

def ofi_proxy(bid_sz: np.ndarray, ask_sz: np.ndarray,
              bid_px: np.ndarray | None = None,
              ask_px: np.ndarray | None = None,
              mid_ret: np.ndarray | None = None) -> np.ndarray:
    """Quote-transition flow in the Cont-Kukanov-Stoikov shape.

    With best bid/ask prices this is the exact CKS event flow per
    second: bid-size adds when the bid price rises, bid-size cancels
    when it falls, and symmetrically for the ask, with the size
    carried at the price that moved.

    With only the mid return (the 1-second bar case) the proxy rests
    on two stated assumptions: (1) bid and ask prices move together
    with the mid, so a positive mid return means both prices rose
    and a negative one means both fell; (2) the spread is fixed
    between seconds. Under those assumptions a mid rise contributes
    +bid_sz[t] + ask_sz[t-1], a mid fall contributes -bid_sz[t-1]
    - ask_sz[t], and a flat mid contributes the size change
    (bid_sz[t] - bid_sz[t-1]) - (ask_sz[t] - ask_sz[t-1]).

    This is a proxy, not OFI: the true CKS quantity needs
    quote-level price moves, which 1-second bars do not carry.
    Callers must label it "OFI proxy" with the assumptions, never
    plain OFI. Raises ValueError when neither prices nor a mid
    return are supplied, rather than silently inventing prices.
    """
    bid = np.asarray(bid_sz, dtype=np.float64)
    ask = np.asarray(ask_sz, dtype=np.float64)
    if bid_px is not None and ask_px is not None:
        bp = np.asarray(bid_px, dtype=np.float64)
        ap = np.asarray(ask_px, dtype=np.float64)
        bid_up = bp[:, 1:] >= bp[:, :-1]
        bid_down = bp[:, 1:] <= bp[:, :-1]
        ask_up = ap[:, 1:] >= ap[:, :-1]
        ask_down = ap[:, 1:] <= ap[:, :-1]
        return (np.where(bid_up, bid[:, 1:], 0.0)
                - np.where(bid_down, bid[:, :-1], 0.0)
                - np.where(ask_down, ask[:, 1:], 0.0)
                + np.where(ask_up, ask[:, :-1], 0.0))
    if mid_ret is not None:
        mr = np.asarray(mid_ret, dtype=np.float64)
        up = mr[:, 1:] > 0
        down = mr[:, 1:] < 0
        flat = ~(up | down)
        return (np.where(up, bid[:, 1:] + ask[:, :-1], 0.0)
                + np.where(down, -bid[:, :-1] - ask[:, 1:], 0.0)
                + np.where(flat,
                           (bid[:, 1:] - bid[:, :-1])
                           - (ask[:, 1:] - ask[:, :-1]), 0.0))
    raise ValueError(
        "ofi_proxy needs bid_px and ask_px, or mid_ret: without a "
        "price move to condition on, the CKS quote-transition flow "
        "is undefined")


# ------------------------------------------------------- regimes, baseline

def regime_response(driver: np.ndarray, target: np.ndarray,
                    regime: np.ndarray,
                    lags: tuple[int, ...] = (1, 2, 3, 4, 5),
                    thresholds: tuple[float, float] | None = None,
                    block_len: int = 10, n_boot: int = 200,
                    seed: int = 0) -> dict:
    """Driver response inside past-volatility / past-spread regimes.

    regime: (n_windows, 2) with columns [past_vol, past_spread]
    (standardized or raw; only the medians matter). Windows split at
    the median of each column into low/high vol and low/high spread;
    the response curve is computed inside each cell. `thresholds`
    fixes the split points (pass the real medians when scoring
    generated windows so both sides share the same regimes).
    """
    regime = np.asarray(regime, dtype=np.float64)
    if thresholds is None:
        thresholds = (float(np.median(regime[:, 0])),
                      float(np.median(regime[:, 1])))
    vol_thr, spr_thr = thresholds
    cells = {
        "low_vol": regime[:, 0] <= vol_thr,
        "high_vol": regime[:, 0] > vol_thr,
        "low_spread": regime[:, 1] <= spr_thr,
        "high_spread": regime[:, 1] > spr_thr,
    }
    rows = []
    for name, mask in cells.items():
        if mask.sum() < 2:
            for lag in lags:
                rows.append({"regime": name, "lag": int(lag),
                             "corr": float("nan"),
                             "corr_ci": [float("nan"), float("nan")],
                             "slope": float("nan"),
                             "slope_ci": [float("nan"), float("nan")],
                             "n_windows": int(mask.sum())})
            continue
        sub = response_curve(driver[mask], target[mask], lags=lags,
                             block_len=block_len, n_boot=n_boot,
                             seed=seed)
        for r in sub["rows"]:
            rows.append({"regime": name, **r})
    return {"rows": rows, "thresholds": [float(vol_thr),
                                         float(spr_thr)]}


def shuffle_baseline(*arrays: np.ndarray,
                     seed: int = 0) -> tuple[np.ndarray, ...]:
    """Within-window time shuffle of each array, independently.

    Every window keeps exactly its own values (marginals preserved)
    while temporal order is destroyed, so any response, correlation,
    or clustering measured on the shuffled arrays is the null the
    real and generated values are judged against.
    """
    rng = np.random.default_rng(seed)
    out = []
    for arr in arrays:
        a = np.asarray(arr)
        sh = np.empty_like(a)
        for i in range(a.shape[0]):
            sh[i] = a[i][rng.permutation(a.shape[1])]
        out.append(sh)
    return tuple(out)


def _within_window_regime(phys: dict) -> np.ndarray:
    """Fallback regimes from the windows themselves.

    First-half (30s) realized volatility (std of log returns) and
    first-half mean spread in bps. Used only when no past-context
    regime array is supplied; the report says which regime
    definition was used.
    """
    half = WINDOW // 2
    vol = phys["log_ret"][:, :half].std(axis=1)
    spr = phys["spread_bps"][:, :half].mean(axis=1)
    return np.column_stack([vol, spr])


def _merge_response_rows(real_curve: dict, gen_curve: dict,
                         base_curve: dict) -> list[dict]:
    """One row per lag: real, generated, and shuffled-baseline values."""
    rows = []
    for rr, gr, br in zip(real_curve["rows"], gen_curve["rows"],
                          base_curve["rows"]):
        rows.append({
            "lag": rr["lag"],
            "real": rr["corr"],
            "gen": gr["corr"],
            "baseline": br["corr"],
            "real_ci": rr["corr_ci"],
            "gen_ci": gr["corr_ci"],
            "baseline_ci": br["corr_ci"],
            "real_slope": rr["slope"],
            "gen_slope": gr["slope"],
            "baseline_slope": br["slope"],
            "real_slope_ci": rr["slope_ci"],
            "gen_slope_ci": gr["slope_ci"],
            "n_windows": rr["n_windows"],
        })
    return rows


# ------------------------------------------------------------- full report

def microstructure_report(real: np.ndarray, gen: np.ndarray,
                          feature_mean: np.ndarray,
                          feature_std: np.ndarray,
                          regime_real: np.ndarray | None = None,
                          regime_gen: np.ndarray | None = None,
                          lags: tuple[int, ...] = RESPONSE_LAGS,
                          block_len: int = 10, n_boot: int = 200,
                          seed: int = 0) -> dict:
    """Held-out real vs generated microstructure diagnostics.

    real, gen: (n_windows, WINDOW, N_FEATURES) standardized windows
    at matched temporal resolutions. regime_real / regime_gen are
    optional (n_windows, 2) past-volatility / past-spread arrays;
    without them, regimes come from each window's own first half
    (documented in the report). Every response row carries the real
    value, the generated value, the shuffled null baseline, and
    block-bootstrap intervals for real and generated.
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    pr = physical_series(real, feature_mean, feature_std)
    pg = physical_series(gen, feature_mean, feature_std)

    def _resp(driver_r, target_r, driver_g, target_g,
              driver_offset: int = 0):
        rc = response_curve(driver_r, target_r, lags=lags,
                            block_len=block_len, n_boot=n_boot,
                            seed=seed, driver_offset=driver_offset)
        gc = response_curve(driver_g, target_g, lags=lags,
                            block_len=block_len, n_boot=n_boot,
                            seed=seed, driver_offset=driver_offset)
        sh_r, sh_t = shuffle_baseline(driver_g, target_g,
                                       seed=seed + 1)
        bc = response_curve(sh_r, sh_t, lags=lags,
                            block_len=block_len, n_boot=n_boot,
                            seed=seed, driver_offset=driver_offset)
        return _merge_response_rows(rc, gc, bc), rc, gc, bc

    flow_rows, _, _, _ = _resp(pr["signed_flow"], pr["log_ret"],
                               pg["signed_flow"], pg["log_ret"])
    imb_rows, _, _, _ = _resp(pr["imbalance"], pr["log_ret"],
                              pg["imbalance"], pg["log_ret"])

    # joint lead-lag cross-correlations, all five series
    cross = cross_correlation_table(pr, max_lag=CROSS_LAG_MAX,
                                    block_len=block_len, n_boot=n_boot,
                                    seed=seed)
    cross_g = cross_correlation_table(pg, max_lag=CROSS_LAG_MAX,
                                      block_len=block_len, n_boot=n_boot,
                                      seed=seed)
    # each series gets its own shuffle stream, so the baseline also
    # destroys contemporaneous cross-correlation, not just lags
    sh_pg = {k: shuffle_baseline(v, seed=seed + 1 + i)[0]
             for i, (k, v) in enumerate(pg.items())}
    cross_b = cross_correlation_table(sh_pg, max_lag=CROSS_LAG_MAX,
                                      block_len=block_len, n_boot=n_boot,
                                      seed=seed)
    cross_rows = []
    for rr, gr, br in zip(cross["rows"], cross_g["rows"],
                          cross_b["rows"]):
        cross_rows.append({
            "series_a": rr["series_a"], "series_b": rr["series_b"],
            "lag": rr["lag"], "real": rr["corr"], "gen": gr["corr"],
            "baseline": br["corr"], "real_ci": rr["ci"],
            "gen_ci": gr["ci"], "baseline_ci": br["ci"],
            "n_windows": rr["n_windows"],
        })

    # transitions: directional rates plus the spread tercile matrix
    spread_thr = (float(np.quantile(pr["spread_bps"], 1 / 3)),
                  float(np.quantile(pr["spread_bps"], 2 / 3)))
    trans = {"thresholds_spread_bps": list(spread_thr)}
    # On this corpus the real spread sits at one tick almost every
    # second, so the two tercile cutoffs can nearly coincide and the
    # tercile states split hairs inside a single tick. The report
    # flags that case; the directional rates stay the meaningful
    # spread-stickiness diagnostic there.
    med_spread = float(np.median(pr["spread_bps"]))
    if (spread_thr[1] - spread_thr[0]) < 0.01 * abs(med_spread):
        trans["spread_tercile_note"] = (
            "tercile thresholds are near-degenerate on this corpus "
            "(real spread sits at one tick almost every second), so "
            "the tercile matrix splits hairs inside a single tick; "
            "judge spread stickiness by the directional rates")
    else:
        trans["spread_tercile_note"] = ""
    for name in ("spread_bps", "bid_sz", "ask_sz"):
        r_rates = directional_transition_rates(pr[name])
        g_rates = directional_transition_rates(pg[name])
        b_rates = directional_transition_rates(sh_pg[name])
        trans[name] = {"real": r_rates, "gen": g_rates,
                       "baseline": b_rates}
    trans["spread_tercile"] = {
        "real": tercile_transition_matrix(pr["spread_bps"], spread_thr),
        "gen": tercile_transition_matrix(pg["spread_bps"], spread_thr),
        "baseline": tercile_transition_matrix(sh_pg["spread_bps"],
                                              spread_thr),
    }

    tails = return_tail_table(pr["log_ret"], pg["log_ret"],
                              block_len=block_len, n_boot=n_boot,
                              seed=seed)
    vol = volatility_clustering(pr["log_ret"], block_len=block_len,
                                n_boot=n_boot, seed=seed)
    vol_g = volatility_clustering(pg["log_ret"], block_len=block_len,
                                  n_boot=n_boot, seed=seed)
    sh_ret_g = shuffle_baseline(pg["log_ret"], seed=seed + 1)[0]
    vol_b = volatility_clustering(sh_ret_g, block_len=block_len,
                                  n_boot=n_boot, seed=seed)
    vol_rows = []
    for rr, gr, br in zip(vol["rows"], vol_g["rows"], vol_b["rows"]):
        vol_rows.append({
            "lag": rr["lag"], "real_abs": rr["abs_acf"],
            "gen_abs": gr["abs_acf"], "baseline_abs": br["abs_acf"],
            "real_abs_ci": rr["abs_ci"], "gen_abs_ci": gr["abs_ci"],
            "real_sq": rr["sq_acf"], "gen_sq": gr["sq_acf"],
            "baseline_sq": br["sq_acf"],
            "real_sq_ci": rr["sq_ci"], "gen_sq_ci": gr["sq_ci"],
            "n_windows": rr["n_windows"],
        })

    inten_r = trade_intensity_table(pr["trade_count"],
                                    block_len=block_len, n_boot=n_boot,
                                    seed=seed)
    inten_g = trade_intensity_table(pg["trade_count"],
                                    block_len=block_len, n_boot=n_boot,
                                    seed=seed)
    sh_cnt_g = shuffle_baseline(pg["trade_count"], seed=seed + 1)[0]
    inten_b = trade_intensity_table(sh_cnt_g, block_len=block_len,
                                    n_boot=n_boot, seed=seed)
    inten_rows = []
    for rr, gr, br in zip(inten_r["count_acf_rows"],
                          inten_g["count_acf_rows"],
                          inten_b["count_acf_rows"]):
        inten_rows.append({
            "lag": rr["lag"], "real": rr["acf"], "gen": gr["acf"],
            "baseline": br["acf"], "real_ci": rr["ci"],
            "gen_ci": gr["ci"], "baseline_ci": br["ci"],
            "n_windows": rr["n_windows"],
        })
    intensity = {
        "real": inten_r, "gen": inten_g, "baseline": inten_b,
        "count_acf_rows": inten_rows,
    }

    # regimes: supplied past-context regimes, else first-half fallback
    if regime_real is None or regime_gen is None:
        regime_real = _within_window_regime(pr)
        regime_gen = _within_window_regime(pg)
        regime_note = ("regimes from each window's own first half: "
                       "realized volatility and mean spread over the "
                       "first 30 seconds; thresholds are the real "
                       "medians")
    else:
        regime_note = ("regimes from the supplied past-context "
                       "arrays [past volatility, past spread]; "
                       "thresholds are the real medians")
    thresholds = (float(np.median(regime_real[:, 0])),
                  float(np.median(regime_real[:, 1])))
    reg_r = regime_response(pr["signed_flow"], pr["log_ret"],
                            regime_real, lags=lags[:5],
                            thresholds=thresholds, block_len=block_len,
                            n_boot=n_boot, seed=seed)
    reg_g = regime_response(pg["signed_flow"], pg["log_ret"],
                            regime_gen, lags=lags[:5],
                            thresholds=thresholds, block_len=block_len,
                            n_boot=n_boot, seed=seed)
    sh_flow_g, sh_ret_g2 = shuffle_baseline(pg["signed_flow"],
                                            pg["log_ret"], seed=seed + 1)
    reg_b = regime_response(sh_flow_g, sh_ret_g2, regime_gen,
                            lags=lags[:5], thresholds=thresholds,
                            block_len=block_len, n_boot=n_boot,
                            seed=seed)
    reg_rows = []
    for rr, gr, br in zip(reg_r["rows"], reg_g["rows"], reg_b["rows"]):
        reg_rows.append({
            "regime": rr["regime"], "lag": rr["lag"],
            "real": rr["corr"], "gen": gr["corr"],
            "baseline": br["corr"], "real_ci": rr["corr_ci"],
            "gen_ci": gr["corr_ci"], "baseline_ci": br["corr_ci"],
            "real_slope": rr["slope"], "gen_slope": gr["slope"],
            "baseline_slope": br["slope"],
            "n_windows": rr["n_windows"],
        })

    # OFI proxy response (assumption-laden; never plain OFI)
    proxy_r = ofi_proxy(pr["bid_sz"], pr["ask_sz"],
                        mid_ret=pr["log_ret"])
    proxy_g = ofi_proxy(pg["bid_sz"], pg["ask_sz"],
                        mid_ret=pg["log_ret"])
    proxy_rows, _, _, _ = _resp(proxy_r, pr["log_ret"],
                                proxy_g, pg["log_ret"],
                                driver_offset=1)
    proxy_note = ("OFI proxy from 1-second quote transitions under "
                  "the stated assumptions: bid and ask prices move "
                  "together with the mid and the spread is fixed "
                  "between seconds. This is a proxy, not Cont-"
                  "Kukanov-Stoikov OFI, which needs quote-level "
                  "price moves that 1-second bars do not carry.")

    return {
        "signed_flow_response": {
            "label": "signed trade volume (not OFI)",
            "rows": flow_rows,
        },
        "imbalance_response": {
            "label": "book imbalance",
            "rows": imb_rows,
        },
        "cross_correlations": {"rows": cross_rows},
        "transitions": trans,
        "return_tails": tails,
        "volatility_clustering": {"rows": vol_rows},
        "trade_intensity": intensity,
        "regime_response": {"rows": reg_rows, "note": regime_note,
                            "thresholds": list(thresholds)},
        "ofi_proxy_response": {
            "label": "OFI proxy (assumption-laden, see note)",
            "note": proxy_note,
            "rows": proxy_rows,
        },
        "n_windows_real": int(len(real)),
        "n_windows_gen": int(len(gen)),
        "window_seconds": int(WINDOW),
        "lags": [int(l) for l in lags],
        "baseline": ("within-window time shuffle of the generated "
                     "windows: marginals preserved, temporal order "
                     "destroyed, so any response, correlation, or "
                     "clustering on it is the null"),
        "ci_method": f"block_bootstrap_block{block_len}_n{n_boot}",
        "naming": ("signed_flow is observed trade signed volume "
                   "(buy positive), not order-flow imbalance as "
                   "defined by Cont, Kukanov, and Stoikov: it omits "
                   "quote additions and cancellations."),
    }
