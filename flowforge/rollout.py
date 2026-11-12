"""Causal helpers for multiwindow generation.

Part 1 contract: each rollout context contains only the anchor's past and
windows generated earlier in that same trajectory. Inputs and outputs stay
in the training-standardized feature space. Callers must not pass future real
windows into this module, and independent recording sessions remain separate.
This module defines context advancement only; it makes no performance claim.
"""

from __future__ import annotations

import numpy as np

from .data import LOOKBACK


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
