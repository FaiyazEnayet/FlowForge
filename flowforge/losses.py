"""Statistical-structure auxiliary losses for the DDPM (P06).

The plain denoising MSE is agnostic to the finance-specific structure
the evaluation checks: lagged dependence inside a window, the
distribution of realized volatility, and the joint dependence between
returns and signed flow. This module adds small, differentiable
penalties that compare the model's denoised estimate of a clean
window (x0_hat, recovered from the noise prediction) against the
clean window itself:

- ``autocov``: within-window lag autocovariance of ``log_ret`` and
  ``signed_flow`` at lags 1..5, the same lags the evaluation's flow
  ACF table reports;
- ``realized_vol``: the distribution of per-window realized
  volatility (standard deviation of ``log_ret`` inside the window),
  matched between estimate and target by sorting, which is the
  differentiable 1-D optimal-transport pairing;
- ``cross``: return/flow dependence: the contemporaneous
  within-window correlation plus the lag-1 cross-correlations in
  both directions (returns leading flow, flow leading returns).

Training-only by construction: every penalty takes exactly two
inputs, the estimate and the target batch, and the target is the
training batch itself (detached: no gradient flows into it). There
is no argument through which a validation or test statistic could
enter, and ``BookDDPM.p_loss`` only ever calls these on training
batches. Weights are predeclared and low (0.01 per component, so
the penalties nudge the denoising objective rather than competing
with it); sensitivity to the weights is pinned in the tests.

Spectral-domain loss: NOT included. ``SPECTRAL_DECISION`` records
the evidence: on 60-step windows a per-batch mean periodogram is
too noisy to be a training target, and the recorded dataset is not
committed, so stability on real training spectra cannot be
verified either. ``assess_spectral_stability`` is the measurement
the decision rests on, kept so the decision can be re-checked.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import FEATURE_NAMES

J_RET = FEATURE_NAMES.index("log_ret")
J_SF = FEATURE_NAMES.index("signed_flow")

DEFAULT_LAGS = (1, 2, 3, 4, 5)

# Predeclared low weights (P06): each component contributes at the
# percent level next to the denoising MSE. Do not tune these on the
# final holdout; sensitivity is checked in tests/test_losses.py.
DEFAULT_AUX_WEIGHTS = {
    "autocov": 0.01,
    "realized_vol": 0.01,
    "cross": 0.01,
}

SPECTRAL_LOSS_INCLUDED = False

SPECTRAL_DECISION = (
    "No spectral-domain loss is shipped. Windows are 60 steps at "
    "1 Hz, so a periodogram has 30 non-DC bins at 1/60 Hz "
    "resolution, and at the training batch size of 32 windows the "
    "mean periodogram is not a stable target: split-half estimates "
    "of the same AR(1) spectrum disagree by ~29% per bin on "
    "average (split-half correlation 0.88 for a strongly "
    "autocorrelated series, ~0.1 for white noise), with a per-bin "
    "coefficient of variation of 0.18-0.22. Those figures use "
    "independent windows; the real training windows overlap "
    "(stride 10 inside 60-step windows), so the effective sample "
    "is smaller and the true instability is worse. A spectral "
    "penalty at batch scale would mostly chase sampling noise. "
    "Measured with assess_spectral_stability; revisit only with "
    "longer windows or a multitaper estimate validated on the "
    "training split."
)


# ------------------------------------------------------------- statistics

def lag_autocovariance(x: torch.Tensor, feature: str,
                       lags=DEFAULT_LAGS) -> torch.Tensor:
    """Mean within-window lag autocovariance of one feature.

    x: (batch, window, features) windows. Each window is demeaned
    over its own time axis; the autocovariance at lag k is the mean
    over windows and over the window of x_t * x_{t+k} (normalized by
    the window length, the biased form, which keeps the statistic
    smooth in the window values). Returns a tensor of len(lags).
    """
    j = FEATURE_NAMES.index(feature)
    s = x[:, :, j]
    s = s - s.mean(dim=1, keepdim=True)
    L = s.shape[1]
    out = []
    for k in lags:
        out.append((s[:, :-k] * s[:, k:]).sum(dim=1).mean() / L)
    return torch.stack(out)


def realized_volatility(x: torch.Tensor) -> torch.Tensor:
    """Per-window realized volatility: std of log_ret over the
    window. x: (batch, window, features). Returns (batch,)."""
    return x[:, :, J_RET].std(dim=1, unbiased=False)


def cross_feature_dependence(x: torch.Tensor) -> torch.Tensor:
    """Return/flow dependence inside each window, averaged.

    Returns a 3-vector: the contemporaneous within-window
    correlation of log_ret and signed_flow, the lag-1
    cross-correlation with returns leading flow, and the lag-1
    cross-correlation with flow leading returns. Correlations are
    computed per window and averaged over the batch.
    """
    r = x[:, :, J_RET]
    f = x[:, :, J_SF]
    r = r - r.mean(dim=1, keepdim=True)
    f = f - f.mean(dim=1, keepdim=True)
    sr = r.std(dim=1, unbiased=False).clamp(min=1e-8)
    sf = f.std(dim=1, unbiased=False).clamp(min=1e-8)
    corr0 = (r * f).mean(dim=1) / (sr * sf)
    lead_ret = (r[:, :-1] * f[:, 1:]).mean(dim=1) / (sr * sf)
    lead_flow = (f[:, :-1] * r[:, 1:]).mean(dim=1) / (sr * sf)
    return torch.stack([corr0.mean(), lead_ret.mean(), lead_flow.mean()])


# ---------------------------------------------------------------- losses

def lag_autocovariance_loss(x_hat: torch.Tensor, x_target: torch.Tensor,
                            lags=DEFAULT_LAGS,
                            features=("log_ret", "signed_flow")) -> torch.Tensor:
    """MSE between estimate and target lag autocovariances, over the
    given features and lags. Zero exactly when the two batches have
    identical autocovariance structure (in particular when equal)."""
    tgt = x_target.detach()
    diffs = []
    for name in features:
        diffs.append(lag_autocovariance(x_hat, name, lags)
                     - lag_autocovariance(tgt, name, lags))
    d = torch.cat(diffs)
    return (d * d).mean()


def realized_volatility_loss(x_hat: torch.Tensor,
                             x_target: torch.Tensor) -> torch.Tensor:
    """Distribution match of per-window realized volatility.

    Sorting both batches' realized volatilities and penalizing the
    squared difference pairs the two empirical distributions by
    rank (the 1-D optimal-transport pairing); it is zero exactly
    when the two distributions coincide as multisets. Requires the
    same batch size on both sides, which training guarantees (the
    estimate and the target are the same batch).
    """
    if x_hat.shape[0] != x_target.shape[0]:
        raise ValueError(
            "realized-volatility distribution matching needs equal "
            f"batch sizes, got {x_hat.shape[0]} and {x_target.shape[0]}")
    a = torch.sort(realized_volatility(x_hat)).values
    b = torch.sort(realized_volatility(x_target.detach())).values
    return ((a - b) ** 2).mean()


def cross_feature_loss(x_hat: torch.Tensor,
                       x_target: torch.Tensor) -> torch.Tensor:
    """MSE between estimate and target return/flow dependence
    vectors (see cross_feature_dependence)."""
    d = cross_feature_dependence(x_hat) \
        - cross_feature_dependence(x_target.detach())
    return (d * d).mean()


def auxiliary_losses(x_hat: torch.Tensor, x_target: torch.Tensor,
                     lags=DEFAULT_LAGS) -> dict[str, torch.Tensor]:
    """All auxiliary components for one batch, unweighted.

    x_hat is the differentiable denoised estimate, x_target the
    clean training windows (detached inside each component). Both
    are (batch, window, features) in standardized units.
    """
    return {
        "autocov": lag_autocovariance_loss(x_hat, x_target, lags=lags),
        "realized_vol": realized_volatility_loss(x_hat, x_target),
        "cross": cross_feature_loss(x_hat, x_target),
    }


def total_auxiliary_loss(x_hat: torch.Tensor, x_target: torch.Tensor,
                         lags=DEFAULT_LAGS,
                         weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    """Weighted total of the auxiliary components.

    ``weights=None`` uses the predeclared DEFAULT_AUX_WEIGHTS.
    Unknown component names raise ValueError instead of being
    silently dropped. Returns (total, components dict).
    """
    w = dict(DEFAULT_AUX_WEIGHTS) if weights is None else dict(weights)
    unknown = sorted(set(w) - set(DEFAULT_AUX_WEIGHTS))
    if unknown:
        raise ValueError(f"unknown auxiliary loss components: {unknown}")
    parts = auxiliary_losses(x_hat, x_target, lags=lags)
    total = sum(w[name] * parts[name] for name in w)
    return total, parts


def predict_x0(xt: torch.Tensor, t: torch.Tensor,
               noise_pred: torch.Tensor, sqrt_acp: torch.Tensor,
               sqrt_1m_acp: torch.Tensor) -> torch.Tensor:
    """Recover the denoised clean-window estimate from a noise
    prediction: x0_hat = (xt - sqrt(1 - alpha_bar_t) * eps) /
    sqrt(alpha_bar_t). Inputs are (batch, features, window)."""
    s = sqrt_acp[t].view(-1, 1, 1)
    s1 = sqrt_1m_acp[t].view(-1, 1, 1)
    return (xt - s1 * noise_pred) / s


# ------------------------------------------------- spectral stability check

def _mean_periodogram(windows: np.ndarray) -> np.ndarray:
    """Mean periodogram over windows (each demeaned), DC excluded."""
    x = np.asarray(windows, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2 / x.shape[1]
    return spec.mean(axis=0)[1:]


def assess_spectral_stability(windows: np.ndarray, seed: int = 0,
                              reps: int = 25) -> dict:
    """How reproducible is a mean periodogram at this sample size?

    Splits the windows into random halves ``reps`` times and reports
    the correlation between the two halves' mean periodograms and
    their mean per-bin relative difference. This is an optimistic
    bound for training: it treats the windows as independent, while
    real training windows overlap. ``windows`` is (n_windows,
    window) for a single series per window.
    """
    x = np.asarray(windows, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    cors, rels = [], []
    for _ in range(reps):
        idx = rng.permutation(n)
        pa = _mean_periodogram(x[idx[: n // 2]])
        pb = _mean_periodogram(x[idx[n // 2:]])
        if np.std(pa) > 0 and np.std(pb) > 0:
            cors.append(float(np.corrcoef(pa, pb)[0, 1]))
        rels.append(float(np.mean(
            np.abs(pa - pb) / np.maximum(pb, 1e-12))))
    return {
        "n_windows": int(n),
        "n_bins": int(x.shape[1] // 2),
        "split_half_corr": float(np.mean(cors)) if cors else float("nan"),
        "mean_rel_diff": float(np.mean(rels)),
        "reps": int(reps),
    }
