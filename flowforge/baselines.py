"""Cheap strong temporal baselines for honest comparison.

FL4: a 25-epoch VAE against a 300-epoch tuned DDPM does not establish an
architecture-only advantage. These baselines are cheap, strong, and make
no neural-network assumptions:

(a) MovingBlockResampler: nonparametric. Samples blocks of consecutive
    training windows (with replacement) to form new sequences. Preserves
    short-range temporal structure by construction. The block length is
    the tuning knob.

(b) VARBaseline: parametric. Fits a vector autoregression of order p on
    flattened training windows with ridge regularization, then generates
    by simulating forward from a random training window start. Captures
    linear temporal dynamics.

Both report sampling latency and peak memory alongside quality metrics,
so quality is always weighed against compute cost.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from .data import N_FEATURES, WINDOW
from .models import set_seed


class MovingBlockResampler:
    """Nonparametric baseline: resample blocks of training windows.

    To generate n windows: repeatedly pick a random training window and
    copy a block of `block_len` consecutive windows starting there
    (wrapping within the training set). This preserves the joint
    distribution of consecutive windows including temporal dependence
    up to the block length.
    """

    def __init__(self, x_train: np.ndarray, block_len: int = 4):
        self.x_train = np.asarray(x_train)
        self.block_len = block_len
        self.n = len(x_train)

    def generate(self, n: int, seed: int = 0) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(seed)
        tracemalloc.start()
        t0 = time.perf_counter()
        out = []
        while len(out) < n:
            start = rng.integers(0, self.n)
            for i in range(self.block_len):
                if len(out) >= n:
                    break
                out.append(self.x_train[(start + i) % self.n])
        gen = np.stack(out).astype(np.float32)
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats = {
            "method": "moving_block",
            "block_len": self.block_len,
            "sample_seconds": dt,
            "sample_seconds_per_1k": dt / max(n, 1) * 1000,
            "peak_memory_mb": peak / 1e6,
            "n_generated": n,
        }
        return gen, stats


class VARBaseline:
    """Regularized VAR(p) baseline on flattened windows.

    Each window is flattened to (WINDOW * N_FEATURES). A VAR(p) model is
    fit with ridge regularization: X[t] = A1 X[t-1] + ... + Ap X[t-p] + e.
    Generation: pick a random training window, then simulate forward p
    steps at a time, adding Gaussian noise with the residual covariance.
    """

    def __init__(self, order: int = 2, ridge: float = 1.0):
        self.order = order
        self.ridge = ridge
        self.coef_ = None
        self.resid_cov_ = None
        self.dim_ = None

    def fit(self, x_train: np.ndarray) -> "VARBaseline":
        x = np.asarray(x_train, dtype=np.float64)
        n, L, F = x.shape
        d = L * F
        self.dim_ = d
        X = x.reshape(n, d)
        # Use consecutive windows as the time series. This is approximate
        # (windows overlap), but it captures the dominant linear dynamics
        # for a baseline. A cleaner fit would use non-overlapping windows.
        p = self.order
        if n <= p:
            raise ValueError(f"need more than {p} windows for VAR({p})")
        Y = X[p:]
        Z = np.column_stack([X[p - k - 1:n - k - 1] for k in range(p)])
        # Ridge: (Z'Z + ridge*I)^{-1} Z'Y
        zt_z = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(zt_z, Z.T @ Y)
        resid = Y - Z @ self.coef_
        self.resid_cov_ = np.cov(resid, rowvar=False) + 1e-8 * np.eye(d)
        return self

    def generate(self, n: int, x_train: np.ndarray,
                 seed: int = 0) -> tuple[np.ndarray, dict]:
        assert self.coef_ is not None, "call fit() first"
        set_seed(seed)
        rng = np.random.default_rng(seed)
        x = np.asarray(x_train, dtype=np.float64)
        d = self.dim_
        p = self.order
        tracemalloc.start()
        t0 = time.perf_counter()
        out = []
        # Start from random training windows, simulate forward.
        L = x.shape[1]
        F = x.shape[2]
        chol = np.linalg.cholesky(self.resid_cov_)
        while len(out) < n:
            idx = rng.integers(0, len(x) - p)
            hist = [x[idx + k].reshape(d) for k in range(p)]
            for _ in range(p * 2):  # generate a few steps per start
                if len(out) >= n:
                    break
                z = np.concatenate(hist[-p:])
                nxt = z @ self.coef_ + chol @ rng.normal(size=d)
                out.append(nxt.reshape(L, F))
                hist.append(nxt)
        gen = np.stack(out[:n]).astype(np.float32)
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats = {
            "method": f"var_p{self.order}",
            "order": self.order,
            "ridge": self.ridge,
            "sample_seconds": dt,
            "sample_seconds_per_1k": dt / max(n, 1) * 1000,
            "peak_memory_mb": peak / 1e6,
            "n_generated": n,
        }
        return gen, stats
