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

import numpy as np
import torch
import torch.nn as nn

from .data import FEATURE_NAMES, N_FEATURES, WINDOW
from .models import set_seed


# ------------------------------------------------- book consistency audit

def inverse_transform(gen: np.ndarray, feature_mean: np.ndarray,
                      feature_std: np.ndarray) -> np.ndarray:
    """Standardized -> physical units: phys = gen * std + mean."""
    return gen * feature_std + feature_mean


def derive_imbalance(log_bid_sz: np.ndarray,
                     log_ask_sz: np.ndarray) -> np.ndarray:
    """Physical imbalance implied by bid/ask sizes.

    imbalance = (bid_sz - ask_sz) / (bid_sz + ask_sz), elementwise.
    Sizes are reconstructed as exp(log_size).
    """
    bid = np.exp(np.asarray(log_bid_sz, dtype=np.float64))
    ask = np.exp(np.asarray(log_ask_sz, dtype=np.float64))
    denom = (bid + ask).clip(min=1e-12)
    return ((bid - ask) / denom).clip(-1.0, 1.0)


def book_consistency_table(gen: np.ndarray, feature_mean: np.ndarray,
                           feature_std: np.ndarray,
                           tol: float = 0.01) -> dict:
    """Audit physical constraints on generated book-state windows.

    The generator emits log bid size, log ask size, and imbalance as
    independent features, but physically imbalance must equal
    (bid_sz - ask_sz) / (bid_sz + ask_sz). This measures the violation
    rate after inverse-transforming to physical units, plus sanity checks
    for negative implied spread and negative implied trade count.

    gen: (n_windows, WINDOW, N_FEATURES) standardized generated windows.
    Returns violation fractions over all (window, timestep) cells.
    """
    phys = inverse_transform(gen, feature_mean, feature_std)
    j_bid = FEATURE_NAMES.index("log_bid_sz")
    j_ask = FEATURE_NAMES.index("log_ask_sz")
    j_imb = FEATURE_NAMES.index("imbalance")
    j_spr = FEATURE_NAMES.index("log_spread_bps")
    j_cnt = FEATURE_NAMES.index("log_trade_count")

    implied_imb = derive_imbalance(phys[:, :, j_bid], phys[:, :, j_ask])
    gen_imb = phys[:, :, j_imb].clip(-1.0, 1.0)
    imb_err = np.abs(gen_imb - implied_imb)
    n_cells = imb_err.size

    spread_bps = np.expm1(phys[:, :, j_spr])
    trade_count = np.expm1(phys[:, :, j_cnt])

    return {
        "n_cells": int(n_cells),
        "imbalance_violation_rate": float((imb_err > tol).mean()),
        "imbalance_mae": float(imb_err.mean()),
        "imbalance_max_err": float(imb_err.max()),
        "negative_spread_rate": float((spread_bps < 0).mean()),
        "negative_count_rate": float((trade_count < -0.5).mean()),
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


def _mean_ci(x: np.ndarray, z: float = 1.96) -> list[float]:
    """95% confidence interval for the mean (normal approximation)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float("nan"), float("nan")]
    se = x.std(ddof=1) / np.sqrt(len(x))
    m = float(x.mean())
    return [m - z * se, m + z * se]


def flow_acf_table(real: np.ndarray, gen: np.ndarray,
                   lags: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    """Mean autocorrelation of signed order flow, averaged across windows.

    Each window is a genuine continuous series, so the autocorrelation is
    computed per window and averaged. Concatenating overlapping windows
    instead would duplicate most of the data and scramble temporal order at
    the junctions. Windows overlap (stride < window length), so the
    confidence intervals are approximate.
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
            "real_ci": _mean_ci(r),
            "gen_acf": float(g.mean()) if len(g) else float("nan"),
            "gen_ci": _mean_ci(g),
            "n_real": int(len(r)),
            "n_gen": int(len(g)),
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


def full_report(real_test: np.ndarray, gen_test: np.ndarray,
                real_val: np.ndarray, gen_val: np.ndarray,
                seed: int = 0,
                feature_mean: np.ndarray = None,
                feature_std: np.ndarray = None) -> dict:
    rep = {
        "marginals": marginal_table(real_test, gen_test),
        "flow_acf": flow_acf_table(real_test, gen_test),
        "discriminator": discriminator_auc(real_val, gen_val, real_test, gen_test,
                                           seed=seed),
        "tstr": tstr_flow_direction(real_val, gen_val, real_test, seed=seed),
    }
    if feature_mean is not None and feature_std is not None:
        pre = book_consistency_table(gen_test, feature_mean, feature_std)
        fixed = constrain_imbalance(gen_test, feature_mean, feature_std)
        post = book_consistency_table(fixed, feature_mean, feature_std)
        # downstream fidelity: does the constraint fix move the marginals?
        pre_w1 = marginal_table(real_test, gen_test)["mean_w1"]
        post_w1 = marginal_table(real_test, fixed)["mean_w1"]
        rep["book_consistency"] = {
            "pre": pre,
            "post": post,
            "constrained": True,
            "marginal_w1_pre": pre_w1,
            "marginal_w1_post": post_w1,
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
