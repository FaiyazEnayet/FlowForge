"""Evaluation: do generated windows look like real ones?

Three families of checks, all computed on HELD-OUT data the generator never
saw during training (the test split):

1. Marginal distributions: for each book-state feature, compare real vs
   generated pooled values by mean, std, and the 1-Wasserstein distance.
2. Temporal structure: autocorrelation of signed order flow at lags 1..5,
   real vs generated. Real flow is positively autocorrelated at short lags
   (parent orders split into children); a good generator reproduces that.
3. TSTR-style discriminator: train a classifier to tell real from generated
   windows. It is trained on the VAL split (real val windows vs generated
   windows conditioned on val regimes) and its AUC is reported on the TEST
   split. Fixed seeds throughout. AUC near 0.5 means the generator fools it.

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


def flow_acf_table(real: np.ndarray, gen: np.ndarray,
                   lags: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    """Autocorrelation of signed order flow, concatenated across windows."""
    j = FEATURE_NAMES.index("signed_flow")
    r = np.concatenate([w[:, j] for w in real])
    g = np.concatenate([w[:, j] for w in gen])
    rows = [{"lag": lag, "real_acf": acf(r, lag), "gen_acf": acf(g, lag)}
            for lag in lags]
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
    return {
        "auc_val": auc_score(yv.numpy(), s_val),
        "auc_test": auc_score(yt.numpy(), s_test),
        "n_val": n,
        "n_test": len(xt),
    }


def full_report(real_test: np.ndarray, gen_test: np.ndarray,
                real_val: np.ndarray, gen_val: np.ndarray,
                seed: int = 0) -> dict:
    return {
        "marginals": marginal_table(real_test, gen_test),
        "flow_acf": flow_acf_table(real_test, gen_test),
        "discriminator": discriminator_auc(real_val, gen_val, real_test, gen_test,
                                           seed=seed),
    }


def print_report(rep: dict) -> None:
    print("\n=== marginal distributions (real vs generated, test split) ===")
    print(f"{'feature':<16}{'real_mean':>10}{'gen_mean':>10}"
          f"{'real_std':>10}{'gen_std':>10}{'w1':>10}")
    for r in rep["marginals"]["rows"]:
        print(f"{r['feature']:<16}{r['real_mean']:>10.3f}{r['gen_mean']:>10.3f}"
              f"{r['real_std']:>10.3f}{r['gen_std']:>10.3f}{r['w1']:>10.3f}")
    print(f"mean W1 across features: {rep['marginals']['mean_w1']:.3f}")
    print("\n=== signed order flow autocorrelation ===")
    print(f"{'lag':>4}{'real_acf':>10}{'gen_acf':>10}")
    for r in rep["flow_acf"]["rows"]:
        print(f"{r['lag']:>4}{r['real_acf']:>10.3f}{r['gen_acf']:>10.3f}")
    d = rep["discriminator"]
    print("\n=== discriminator (trained on val, tested on test) ===")
    print(f"AUC on val: {d['auc_val']:.3f} | AUC on test: {d['auc_test']:.3f} "
          f"(0.5 = indistinguishable)")
    print(f"(n_val={d['n_val']}, n_test={d['n_test']}, seed fixed)")
