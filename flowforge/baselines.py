"""Cheap strong temporal baselines for honest comparison.

FL4: a 25-epoch VAE against a 300-epoch tuned DDPM does not establish an
architecture-only advantage. These baselines are cheap, strong, and make
no neural-network assumptions:

(a) MovingBlockResampler: nonparametric, unconditional. Samples blocks
    of consecutive training windows (with replacement) to form new
    sequences. Preserves short-range temporal structure by construction.
    The block length is the tuning knob.

(b) VARBaseline: parametric, unconditional. Fits a vector autoregression
    of order p on flattened training windows with ridge regularization,
    then generates by simulating forward from a random training window
    start. Captures linear temporal dynamics.

(c) RegimeMatchedChunkCopyBaseline: nonparametric, conditional. For a
    target regime, copies the training block whose starting regime is
    closest. This is the honest conditional control: a conditional
    generator must beat simply copying nearby real data.

(d) ConditionalARBaseline: parametric, conditional, lightweight. A
    per-timestep vector autoregression inside each window, with the
    regime entering as an input, fit on within-window transitions only.

Session/segment discipline (C03): windows carry session and segment
identities. A copied block never wraps across a session or segment
boundary, VAR lag sequences are fitted only inside one valid session or
segment, and every generation reports the training-window indices it
came from, so the train-only origin of every output is checkable.

All baselines report sampling latency and peak memory alongside quality
metrics, so quality is always weighed against compute cost. None of
them performs a neural function evaluation, so their NFE is 0.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from .data import N_FEATURES, WINDOW
from .models import set_seed


def _segment_keys(n: int, session_ids=None, segment_ids=None) -> np.ndarray:
    """Integer key per window: equal keys <=> same session and segment.

    With neither identity supplied, every window shares one key (the
    whole array is one segment). Session and segment identities are
    combined, because segment labels may repeat across sessions.
    """
    if session_ids is None and segment_ids is None:
        return np.zeros(n, dtype=np.int64)
    ses = (np.zeros(n, dtype=np.int64) if session_ids is None
           else np.asarray(session_ids))
    seg = (np.zeros(n, dtype=np.int64) if segment_ids is None
           else np.asarray(segment_ids))
    if len(ses) != n or len(seg) != n:
        raise ValueError("session_ids/segment_ids must match the window "
                         "count")
    keys = np.empty(n, dtype=np.int64)
    seen: dict = {}
    for i in range(n):
        pair = (ses[i].item(), seg[i].item())
        if pair not in seen:
            seen[pair] = len(seen)
        keys[i] = seen[pair]
    return keys


def segment_ids_from_windows(session_ids, t_start, t_end,
                             stride: int = 10) -> np.ndarray:
    """Segment identity per window from session and window times.

    Windows inside one segment start exactly `stride` seconds apart and
    overlap in time. A new segment starts when the session changes, when
    the start does not continue the stride, or when the window starts
    after the previous window ended (a recording gap separates them).
    """
    session_ids = np.asarray(session_ids)
    t_start = np.asarray(t_start, dtype=np.float64)
    t_end = np.asarray(t_end, dtype=np.float64)
    n = len(t_start)
    seg = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        split = (session_ids[i] != session_ids[i - 1]
                 or t_start[i] != t_start[i - 1] + stride
                 or t_start[i] > t_end[i - 1])
        seg[i] = seg[i - 1] + (1 if split else 0)
    return seg


def _valid_block_starts(keys: np.ndarray, block_len: int) -> list[int]:
    """Start indices whose whole block stays inside one segment."""
    starts = []
    n = len(keys)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and keys[j + 1] == keys[i]:
            j += 1
        # segment run is [i, j]
        for s in range(i, j - block_len + 2):
            starts.append(s)
        i = j + 1
    return starts


def _timed(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, peak / 1e6


def _psd_sqrt(cov: np.ndarray) -> np.ndarray:
    """Square root of a symmetric PSD matrix (Cholesky, eigh fallback)."""
    cov = 0.5 * (cov + cov.T)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 0.0, None)
        return vecs * np.sqrt(vals)


class MovingBlockResampler:
    """Nonparametric unconditional baseline: resample blocks of
    training windows.

    To generate n windows: repeatedly pick a random valid block start
    and copy the `block_len` consecutive windows starting there. A block
    never wraps: starts are restricted to positions whose whole block
    lies inside one session/segment, so no output splices the end of
    one recording onto the start of another. The training indices
    behind every output window are reported in the stats, so the
    train-only origin of the generation is auditable.
    """

    kind = "unconditional"

    def __init__(self, x_train: np.ndarray, block_len: int = 4,
                 session_ids=None, segment_ids=None):
        self.x_train = np.asarray(x_train)
        self.block_len = block_len
        self.n = len(x_train)
        self.session_ids = session_ids
        self.segment_ids = segment_ids
        self.keys_ = _segment_keys(self.n, session_ids, segment_ids)
        self.valid_starts_ = _valid_block_starts(self.keys_, block_len)

    def generate(self, n: int, seed: int = 0) -> tuple[np.ndarray, dict]:
        if not self.valid_starts_:
            raise ValueError(
                f"no segment holds a full block of {self.block_len} "
                f"windows; block resampling cannot proceed without "
                f"wrapping across a session/segment boundary")
        rng = np.random.default_rng(seed)

        def _draw():
            idx: list[int] = []
            while len(idx) < n:
                s = int(self.valid_starts_[
                    rng.integers(0, len(self.valid_starts_))])
                idx.extend(range(s, s + self.block_len))
            return idx[:n]

        source, dt, peak_mb = _timed(_draw)
        gen = np.stack([self.x_train[i] for i in source]).astype(np.float32)
        stats = {
            "method": "moving_block",
            "kind": self.kind,
            "block_len": self.block_len,
            "sample_seconds": dt,
            "sample_seconds_per_1k": dt / max(n, 1) * 1000,
            "peak_memory_mb": peak_mb,
            "n_generated": n,
            "nfe": 0,
            "nfe_per_window": 0,
            "train_only_source": True,
            "source_indices": [int(i) for i in source],
        }
        if self.session_ids is not None:
            ses = np.asarray(self.session_ids)
            stats["source_session_ids"] = [int(ses[i]) for i in source]
        if self.segment_ids is not None:
            seg = np.asarray(self.segment_ids)
            stats["source_segment_ids"] = [int(seg[i]) for i in source]
        return gen, stats


class VARBaseline:
    """Regularized VAR(p) unconditional baseline on flattened windows.

    Each window is flattened to (WINDOW * N_FEATURES). A VAR(p) model is
    fit with ridge regularization: X[t] = A1 X[t-1] + ... + Ap X[t-p] + e.
    Lag sequences are taken only from runs of consecutive windows inside
    one session/segment: the window array is not one continuous series,
    and fitting across a boundary would train on joins that never
    happened. Generation starts from a valid in-segment window run and
    simulates forward, adding Gaussian noise with the residual
    covariance.
    """

    kind = "unconditional"

    def __init__(self, order: int = 2, ridge: float = 1.0):
        self.order = order
        self.ridge = ridge
        self.coef_ = None
        self.resid_cov_ = None
        self.dim_ = None
        self.fit_pair_indices_ = []
        self.n_lag_pairs_ = 0
        self._fit_keys = None

    def fit(self, x_train: np.ndarray, session_ids=None,
            segment_ids=None) -> "VARBaseline":
        x = np.asarray(x_train, dtype=np.float64)
        n, L, F = x.shape
        d = L * F
        self.dim_ = d
        X = x.reshape(n, d)
        p = self.order
        keys = _segment_keys(n, session_ids, segment_ids)
        self._fit_keys = keys
        seqs = []
        for t in range(p, n):
            if np.all(keys[t - p:t + 1] == keys[t]):
                seqs.append(tuple(range(t - p, t + 1)))
        if len(seqs) < 2:
            raise ValueError(
                f"need at least 2 within-session lag sequences for "
                f"VAR({p}), found {len(seqs)}")
        self.fit_pair_indices_ = seqs
        self.n_lag_pairs_ = len(seqs)
        Y = np.stack([X[s[-1]] for s in seqs])
        Z = np.concatenate([np.stack([X[s[-1 - k]] for s in seqs])
                            for k in range(1, p + 1)], axis=1)
        # Ridge: (Z'Z + ridge*I)^{-1} Z'Y
        zt_z = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(zt_z, Z.T @ Y)
        resid = Y - Z @ self.coef_
        self.resid_cov_ = np.cov(resid, rowvar=False) + 1e-8 * np.eye(d)
        return self

    def generate(self, n: int, x_train: np.ndarray,
                 seed: int = 0, session_ids=None,
                 segment_ids=None) -> tuple[np.ndarray, dict]:
        assert self.coef_ is not None, "call fit() first"
        set_seed(seed)
        rng = np.random.default_rng(seed)
        x = np.asarray(x_train, dtype=np.float64)
        d = self.dim_
        p = self.order
        if session_ids is None and segment_ids is None \
                and self._fit_keys is not None and len(x) == len(self._fit_keys):
            keys = self._fit_keys
        else:
            keys = _segment_keys(len(x), session_ids, segment_ids)
        starts = [s for s in range(0, len(x) - p + 1)
                  if np.all(keys[s:s + p] == keys[s])]
        if not starts:
            raise ValueError(
                f"no valid in-segment run of {p} windows to start "
                f"VAR generation from")
        L = x.shape[1]
        F = x.shape[2]
        chol = _psd_sqrt(self.resid_cov_)
        used_starts: list[int] = []

        def _draw():
            out = []
            while len(out) < n:
                idx = int(starts[rng.integers(0, len(starts))])
                used_starts.append(idx)
                hist = [x[idx + k].reshape(d) for k in range(p)]
                for _ in range(p * 2):  # generate a few steps per start
                    if len(out) >= n:
                        break
                    z = np.concatenate(hist[-p:])
                    nxt = z @ self.coef_ + chol @ rng.normal(size=d)
                    out.append(nxt.reshape(L, F))
                    hist.append(nxt)
            return out

        out, dt, peak_mb = _timed(_draw)
        gen = np.stack(out[:n]).astype(np.float32)
        stats = {
            "method": f"var_p{self.order}",
            "kind": self.kind,
            "order": self.order,
            "ridge": self.ridge,
            "sample_seconds": dt,
            "sample_seconds_per_1k": dt / max(n, 1) * 1000,
            "peak_memory_mb": peak_mb,
            "n_generated": n,
            "nfe": 0,
            "nfe_per_window": 0,
            "train_only_source": True,
            "source_start_indices": used_starts,
            "n_lag_pairs": self.n_lag_pairs_,
        }
        return gen, stats


class RegimeMatchedChunkCopyBaseline:
    """Nonparametric conditional baseline: copy the training block
    whose regime best matches the target regime.

    For each output block, the valid block start (inside one
    session/segment, never wrapping) whose starting window's regime is
    closest to the target regime, in train-standardized regime units,
    is copied verbatim. Every output window is a real training window,
    and the copied indices are reported, so the baseline is exactly as
    strong as "just copy nearby real data" and nothing more.
    """

    kind = "conditional"

    def __init__(self, x_train: np.ndarray, regimes_train: np.ndarray,
                 block_len: int = 4, session_ids=None, segment_ids=None):
        self.x_train = np.asarray(x_train)
        self.regimes_train = np.asarray(regimes_train, dtype=np.float64)
        if len(self.regimes_train) != len(self.x_train):
            raise ValueError("regimes_train must match the window count")
        self.block_len = block_len
        self.session_ids = session_ids
        self.segment_ids = segment_ids
        self.keys_ = _segment_keys(len(self.x_train), session_ids,
                                   segment_ids)
        self.valid_starts_ = _valid_block_starts(self.keys_, block_len)
        if not self.valid_starts_:
            raise ValueError(
                f"no segment holds a full block of {block_len} windows "
                f"for chunk copying")
        self.regime_mean_ = self.regimes_train.mean(axis=0)
        self.regime_std_ = self.regimes_train.std(axis=0)
        self.regime_std_[self.regime_std_ < 1e-12] = 1.0
        starts = np.asarray(self.valid_starts_)
        self._start_regimes = (
            (self.regimes_train[starts] - self.regime_mean_)
            / self.regime_std_)

    def generate(self, n: int, regimes_target: np.ndarray,
                 seed: int = 0) -> tuple[np.ndarray, dict]:
        target = np.asarray(regimes_target, dtype=np.float64)
        if len(target) == 0:
            raise ValueError("regimes_target is empty")
        tz = (target - self.regime_mean_) / self.regime_std_

        def _draw():
            idx: list[int] = []
            while len(idx) < n:
                t = tz[len(idx) % len(tz)]
                d2 = np.sum((self._start_regimes - t) ** 2, axis=1)
                s = int(self.valid_starts_[int(np.argmin(d2))])
                idx.extend(range(s, s + self.block_len))
            return idx[:n]

        source, dt, peak_mb = _timed(_draw)
        gen = np.stack([self.x_train[i] for i in source]).astype(np.float32)
        stats = {
            "method": "regime_matched_chunk_copy",
            "kind": self.kind,
            "block_len": self.block_len,
            "sample_seconds": dt,
            "sample_seconds_per_1k": dt / max(n, 1) * 1000,
            "peak_memory_mb": peak_mb,
            "n_generated": n,
            "nfe": 0,
            "nfe_per_window": 0,
            "train_only_source": True,
            "source_indices": [int(i) for i in source],
        }
        if self.session_ids is not None:
            ses = np.asarray(self.session_ids)
            stats["source_session_ids"] = [int(ses[i]) for i in source]
        if self.segment_ids is not None:
            seg = np.asarray(self.segment_ids)
            stats["source_segment_ids"] = [int(seg[i]) for i in source]
        return gen, stats


class ConditionalARBaseline:
    """Lightweight conditional autoregressive comparator.

    A per-timestep VAR(`order`) inside each window: the next book-state
    vector is a ridge fit on the previous `order` vectors plus the
    window's regime. Fitting uses within-window transitions only, so no
    session or segment boundary can enter the fit by construction. The
    first step of a generated window comes from a regime-only ridge fit
    on training first steps; the rest is simulated forward with
    residual noise. Cheap, linear, and conditional: the comparator a
    neural conditional generator should clear without effort.
    """

    kind = "conditional"

    def __init__(self, order: int = 1, ridge: float = 1.0):
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.ridge = ridge
        self.coef_ = None
        self.first_coef_ = None
        self.resid_sqrt_ = None
        self.first_resid_sqrt_ = None
        self.n_transitions_ = 0

    def fit(self, x_train: np.ndarray,
            regimes_train: np.ndarray) -> "ConditionalARBaseline":
        x = np.asarray(x_train, dtype=np.float64)
        regimes = np.asarray(regimes_train, dtype=np.float64)
        n, L, F = x.shape
        if len(regimes) != n:
            raise ValueError("regimes_train must match the window count")
        p = self.order
        if L <= p:
            raise ValueError(f"window length {L} must exceed order {p}")
        R = regimes.shape[1]
        # transitions: for each window, t = p..L-1 (within window only)
        rows = n * (L - p)
        Z = np.empty((rows, p * F + R + 1))
        Y = np.empty((rows, F))
        r = 0
        for i in range(n):
            for t in range(p, L):
                Z[r, :p * F] = np.concatenate(
                    [x[i, t - k] for k in range(1, p + 1)])
                Z[r, p * F:p * F + R] = regimes[i]
                Z[r, -1] = 1.0
                Y[r] = x[i, t]
                r += 1
        self.n_transitions_ = rows
        pen = self.ridge * np.eye(Z.shape[1])
        pen[-1, -1] = 0.0  # do not penalize the intercept
        self.coef_ = np.linalg.solve(Z.T @ Z + pen, Z.T @ Y)
        resid = Y - Z @ self.coef_
        cov = np.cov(resid, rowvar=False) + 1e-8 * np.eye(F)
        self.resid_sqrt_ = _psd_sqrt(cov)
        # first-step model: x[:, 0] from the regime alone
        Z0 = np.column_stack([regimes, np.ones(n)])
        pen0 = self.ridge * np.eye(R + 1)
        pen0[-1, -1] = 0.0
        self.first_coef_ = np.linalg.solve(Z0.T @ Z0 + pen0, Z0.T @ x[:, 0])
        resid0 = x[:, 0] - Z0 @ self.first_coef_
        cov0 = np.cov(resid0, rowvar=False) + 1e-8 * np.eye(F)
        self.first_resid_sqrt_ = _psd_sqrt(cov0)
        self.n_features_ = F
        self.window_ = L
        return self

    def generate(self, n: int, regimes_target: np.ndarray,
                 seed: int = 0) -> tuple[np.ndarray, dict]:
        assert self.coef_ is not None, "call fit() first"
        rng = np.random.default_rng(seed)
        target = np.asarray(regimes_target, dtype=np.float64)
        if len(target) == 0:
            raise ValueError("regimes_target is empty")
        F, L, p = self.n_features_, self.window_, self.order

        def _draw():
            out = []
            for i in range(n):
                regime = target[i % len(target)]
                win = np.empty((L, F))
                hist: list[np.ndarray] = []
                for t in range(L):
                    if t < p:
                        mean = (np.concatenate([regime, [1.0]])
                                @ self.first_coef_)
                        step = mean + self.first_resid_sqrt_ @ rng.normal(size=F)
                    else:
                        z = np.concatenate(
                            [hist[-k] for k in range(1, p + 1)]
                            + [regime, [1.0]])
                        step = z @ self.coef_ + self.resid_sqrt_ @ rng.normal(size=F)
                    win[t] = step
                    hist.append(step)
                out.append(win)
            return out

        out, dt, peak_mb = _timed(_draw)
        gen = np.stack(out).astype(np.float32)
        stats = {
            "method": "conditional_ar",
            "kind": self.kind,
            "order": self.order,
            "ridge": self.ridge,
            "sample_seconds": dt,
            "sample_seconds_per_1k": dt / max(n, 1) * 1000,
            "peak_memory_mb": peak_mb,
            "n_generated": n,
            "nfe": 0,
            "nfe_per_window": 0,
            "train_only_source": True,
            "n_transitions_fit": self.n_transitions_,
        }
        return gen, stats
