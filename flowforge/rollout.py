"""Causal helpers for multiwindow generation.

Part 1 contract: each rollout context contains only the anchor's past and
windows generated earlier in that same trajectory. Inputs and outputs stay
in the training-standardized feature space. Callers must not pass future real
windows into this module, and independent recording sessions remain separate.
This module defines context advancement only; it makes no performance claim.

Part 2 adds the free-running rollout itself plus the teacher-forced
one-step reference. A free-running rollout starts from one real anchor
history and then feeds only its own generated windows back as context:
at every step the regime for the next window is recomputed from the
current (anchor plus generated) context, and the history-conditioned
model samples one window from that context alone. The function takes
no real continuation windows at all, so no future real data can be
read during a rollout; the test suite verifies this by construction,
checking every context the model ever sees against an independent
advance_context computation.

Absolute price is not in the feature set (feature 0 is the 1s log
return), so anchor continuity is preserved on the relative log price:
the cumulative sum of log returns from the anchor's last second. The
seam audit in flowforge.evaluate_rollout measures that continuity at
every window boundary.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import FEATURE_NAMES, LOOKBACK, N_FEATURES, WINDOW
from .models import derive_seed

J_RET = FEATURE_NAMES.index("log_ret")
J_SPR = FEATURE_NAMES.index("log_spread_bps")


def advance_context(past_context: np.ndarray,
                    generated_window: np.ndarray,
                    lookback: int = LOOKBACK) -> np.ndarray:
    """Return the next context from prior context and generated rows only.

    Both arguments have shape ``(batch, time, features)`` and use the same
    standardized feature scale. The result contains the most recent
    ``lookback`` rows, copied so later caller mutations cannot alter history.
    """
    history = np.asarray(past_context)
    generated = np.asarray(generated_window)
    if not isinstance(lookback, int) or isinstance(lookback, bool) \
            or lookback <= 0:
        raise ValueError("lookback must be a positive integer")
    if history.ndim != 3 or generated.ndim != 3:
        raise ValueError("context and generated window must be rank-3 arrays")
    if history.shape[0] == 0 or history.shape[2] == 0:
        raise ValueError(
            "context and generated window must have non-empty batch/features")
    if history.shape[0] != generated.shape[0] \
            or history.shape[2] != generated.shape[2]:
        raise ValueError("context and generated window batch/features must match")
    if history.shape[1] < lookback or generated.shape[1] == 0:
        raise ValueError(
            "context must cover lookback and generated window must be non-empty")
    for name, values in (("context", history), ("generated window", generated)):
        if not np.issubdtype(values.dtype, np.number) \
                or np.iscomplexobj(values):
            raise ValueError(f"{name} must contain real numeric values")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")

    joined = np.concatenate((history, generated), axis=1)
    return joined[:, -lookback:, :].copy()


def _advance_mask(past_mask: np.ndarray, window_len: int,
                  lookback: int = LOOKBACK) -> np.ndarray:
    """Advance an availability mask by one generated window.

    Generated rows are fully available; anchor seconds that were
    unavailable keep their mask value until they slide out of the
    lookback. The mask never borrows seconds from another session.
    """
    mask = np.asarray(past_mask)
    fresh = np.ones((mask.shape[0], window_len), dtype=bool)
    return np.concatenate((mask, fresh), axis=1)[:, -lookback:].copy()


def _checked_standardized_array(values, name, n_features=N_FEATURES):
    """Validate a standardized (batch, time, features) numpy array."""
    arr = np.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a rank-3 array, got shape "
                         f"{arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError(f"{name} must have non-empty batch/time/features")
    if arr.shape[2] != n_features:
        raise ValueError(f"{name} must have {n_features} features, got "
                         f"{arr.shape[2]}")
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr.astype(np.float64)


def _checked_stats(values, name, length):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got "
                         f"{arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def regime_from_context(context: np.ndarray,
                        feature_mean: np.ndarray,
                        feature_std: np.ndarray,
                        regime_mean: np.ndarray,
                        regime_std: np.ndarray) -> np.ndarray:
    """Standardized regime of a standardized context.

    The regime is the same two-number summary the data pipeline
    computes from a lookback (``flowforge/data.py``): realized
    volatility (std of the 1s log returns over the lookback) and mean
    spread in bps over the lookback. The context is inverse-transformed
    to physical units first, so the summary is computed on the same
    quantities as the pipeline's; the result is then standardized with
    the training regime statistics. The context must be (batch,
    lookback, features); the lookback length is whatever the caller
    passes, and every row of it is treated as past.
    """
    ctx = _checked_standardized_array(context, "context")
    f_mean = _checked_stats(feature_mean, "feature_mean", N_FEATURES)
    f_std = _checked_stats(feature_std, "feature_std", N_FEATURES)
    r_mean = _checked_stats(regime_mean, "regime_mean", 2)
    r_std = _checked_stats(regime_std, "regime_std", 2)
    if (f_std <= 0).any() or (r_std <= 0).any():
        raise ValueError("feature_std and regime_std must be positive")
    log_ret = ctx[:, :, J_RET] * f_std[J_RET] + f_mean[J_RET]
    vol = log_ret.std(axis=1)
    spread_bps = np.expm1(ctx[:, :, J_SPR] * f_std[J_SPR] + f_mean[J_SPR])
    mean_spread = spread_bps.mean(axis=1)
    raw = np.stack([vol, mean_spread], axis=1)
    return (raw - r_mean) / r_std


def _require_history(model) -> None:
    """Fail unless the model is built with history conditioning."""
    if hasattr(model, "use_history"):
        if not model.use_history:
            raise ValueError(
                "rollout requires a history-conditioned model "
                "(BookDDPM with use_history=True)")
        return
    if hasattr(model, "context_dim"):
        if not model.context_dim:
            raise ValueError(
                "rollout requires a history-conditioned model "
                "(CondVAE with context_dim > 0)")
        return
    raise TypeError(
        "rollout model must expose use_history (BookDDPM) or "
        "context_dim (CondVAE)")


def _sample_one(model, regime, context, mask, seed, generator,
                batch_size, mode, feature_mean, feature_std):
    """Sample one window from a history-conditioned model.

    BookDDPM exposes ``sample``; CondVAE exposes ``generate``. Both
    return a torch tensor of shape (batch, window, features); a numpy
    return is accepted as well (used by test stubs). ``mode`` applies
    to the DDPM path only.
    """
    regime_t = torch.from_numpy(np.asarray(regime, dtype=np.float32))
    context_t = torch.from_numpy(np.asarray(context, dtype=np.float32))
    mask_t = None if mask is None else torch.from_numpy(
        np.asarray(mask, dtype=bool))
    kw = {"past_context": context_t, "past_mask": mask_t}
    if hasattr(model, "sample"):
        if generator is not None:
            out = model.sample(regime_t, generator=generator,
                               batch_size=batch_size, mode=mode,
                               feature_mean=feature_mean,
                               feature_std=feature_std, **kw)
        else:
            out = model.sample(regime_t, seed=int(seed),
                               batch_size=batch_size, mode=mode,
                               feature_mean=feature_mean,
                               feature_std=feature_std, **kw)
    elif hasattr(model, "generate"):
        if mode != "raw":
            raise ValueError("mode applies to the DDPM sample path only")
        if generator is not None:
            out = model.generate(regime_t, generator=generator, **kw)
        else:
            out = model.generate(regime_t, seed=int(seed), **kw)
    else:
        raise TypeError("rollout model must expose sample or generate")
    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy().astype(np.float64)
    return np.asarray(out, dtype=np.float64)


def teacher_forced_window(model, real_context, real_mask, regime,
                          feature_mean=None, feature_std=None,
                          regime_mean=None, regime_std=None,
                          seed: int = 0, generator=None,
                          batch_size: int | None = None,
                          mode: str = "raw", run_id: str | None = None) -> np.ndarray:
    """Sample one window conditioned on its true past context.

    This is the one-step reference the free-running rollout is measured
    against: the context is real history, so the sampled window is a
    teacher-forced next window, not a rollout step. ``regime`` is the
    standardized (batch, 2) regime of that real context, as the data
    pipeline records it; it is taken as given, never recomputed.
    With ``run_id`` the seed derives from the isolated ``rollout``
    stream, as in :func:`rollout_windows`; otherwise ``seed`` is used
    directly.
    """
    _require_history(model)
    ctx = _checked_standardized_array(real_context, "real_context")
    if real_mask is not None:
        mask = np.asarray(real_mask)
        if mask.shape != ctx.shape[:2]:
            raise ValueError(
                f"real_mask must have shape {ctx.shape[:2]}, got "
                f"{mask.shape}")
        mask = mask.astype(bool)
    else:
        mask = np.ones(ctx.shape[:2], dtype=bool)
    reg = np.asarray(regime, dtype=np.float64)
    if reg.ndim != 2 or reg.shape != (ctx.shape[0], 2):
        raise ValueError(
            f"regime must have shape ({ctx.shape[0]}, 2), got {reg.shape}")
    if not np.isfinite(reg).all():
        raise ValueError("regime must contain only finite values")
    if run_id is not None:
        seed = derive_seed(f"p09-rollout:{run_id}",
                           f"teacher-forced:{seed}")
    return _sample_one(model, reg, ctx, mask, seed, generator,
                       batch_size, mode, feature_mean, feature_std)


def rollout_windows(model, anchor_context, anchor_mask=None,
                    n_windows: int = 1,
                    feature_mean=None, feature_std=None,
                    regime_mean=None, regime_std=None,
                    seed: int = 0, generator=None,
                    batch_size: int | None = None,
                    mode: str = "raw",
                    run_id: str | None = None) -> dict:
    """Free-run a history-conditioned model for ``n_windows`` windows.

    The only real data the rollout ever sees is the anchor history
    (``anchor_context``, the LOOKBACK seconds before the anchor window,
    with ``anchor_mask`` marking unavailable seconds). At every step the
    regime for the next window is recomputed from the current context
    with :func:`regime_from_context`, the model samples one window
    conditioned on that context, and the context advances with
    :func:`advance_context` using the generated window only. There is
    no parameter for real continuation windows: a free-running rollout
    cannot read future real data, by construction.

    Returns a dict with ``windows`` (batch, n_windows, window,
    features), ``regimes`` (batch, n_windows, 2) as sampled, ``seams``
    (one explicit checkpoint per window boundary, seam 0 between the
    anchor and the first generated window), and the ``anchor_context``
    copy the rollout started from. Deterministic for a fixed seed:
    per-step seeds derive from ``seed`` (or ``run_id``) through the
    isolated ``rollout`` stream, so the global RNG state is untouched.
    """
    _require_history(model)
    if not isinstance(n_windows, int) or isinstance(n_windows, bool) \
            or n_windows <= 0:
        raise ValueError("n_windows must be a positive integer")
    ctx = _checked_standardized_array(anchor_context, "anchor_context")
    lookback = ctx.shape[1]
    if anchor_mask is not None:
        mask = np.asarray(anchor_mask)
        if mask.shape != ctx.shape[:2]:
            raise ValueError(
                f"anchor_mask must have shape {ctx.shape[:2]}, got "
                f"{mask.shape}")
        mask = mask.astype(bool)
    else:
        mask = np.ones(ctx.shape[:2], dtype=bool)
    f_mean = _checked_stats(feature_mean, "feature_mean", N_FEATURES)
    f_std = _checked_stats(feature_std, "feature_std", N_FEATURES)
    r_mean = _checked_stats(regime_mean, "regime_mean", 2)
    r_std = _checked_stats(regime_std, "regime_std", 2)
    window_len = int(getattr(model, "window", WINDOW))
    if window_len <= 0:
        raise ValueError("model window must be positive")

    windows = []
    regimes = []
    seams = []
    for k in range(n_windows):
        regime = regime_from_context(ctx, f_mean, f_std, r_mean, r_std)
        if generator is None:
            stream = f"p09-rollout:{run_id}" if run_id is not None \
                else f"p09-rollout:{seed}"
            step_seed = derive_seed(stream, f"step:{k}")
        else:
            step_seed = seed
        window = _sample_one(model, regime, ctx, mask, step_seed,
                             generator, batch_size, mode, f_mean, f_std)
        window = _checked_standardized_array(window, "generated window")
        if window.shape[1] != window_len:
            raise ValueError(
                f"sampled window has length {window.shape[1]}, "
                f"expected {window_len}")
        windows.append(window)
        regimes.append(regime)
        seams.append({"step": int(k),
                      "from": "anchor" if k == 0 else f"window_{k - 1}",
                      "to": f"window_{k}"})
        ctx = advance_context(ctx, window, lookback=lookback)
        mask = _advance_mask(mask, window.shape[1], lookback=lookback)
    return {
        "windows": np.stack(windows, axis=1),
        "regimes": np.stack(regimes, axis=1),
        "seams": seams,
        "anchor_context": np.array(anchor_context, dtype=np.float64),
        "n_windows": int(n_windows),
    }
