"""Faster samplers for the trained DDPM: DDIM and the NFE frontier.

The shipped ancestral sampler (``BookDDPM.sample``) spends one
denoiser evaluation per reverse timestep, 100 evaluations per window,
and that dominates CPU inference time. The DDIM sampler of Song et
al. (ICLR 2021) reuses the same trained epsilon-prediction weights
and skips timesteps: with ``eta = 0`` the chain is deterministic, with
``eta = 1`` over every timestep it is exactly the ancestral DDPM
sampler (same mean, same posterior variance, same noise-draw order),
and intermediate values interpolate between the two.

The DDIM update at timestep ``t`` with previous timestep ``t_prev``,
writing ``a_t`` for the cumulative noise product and ``eps`` for the
network's noise estimate (converted from whatever objective the model
was trained with, exactly as the ancestral sampler converts it):

    x0_hat = (x_t - sqrt(1 - a_t) eps) / sqrt(a_t)
    sigma  = eta sqrt((1 - a_{t_prev}) / (1 - a_t))
                 sqrt(1 - a_t / a_{t_prev})
    x_{t_prev} = sqrt(a_{t_prev}) x0_hat
                 + sqrt(1 - a_{t_prev} - sigma^2) eps
                 + sigma z,  z ~ N(0, I) when sigma > 0

A final step to ``t_prev = -1`` returns ``x0_hat`` directly: the
chain has no noise left to add. The timestep subsequence is evenly
spaced over the chain and always includes both exact boundaries,
``t_steps - 1`` and ``0``, so the chain starts from pure noise and
ends at a clean window.

Nothing here touches the ancestral sampler: ``BookDDPM.sample`` is
the untouched regression baseline, and this sampler only reads the
model's weights, schedule, and objective. Sampling draws come from
an isolated generator, as in ``BookDDPM.sample``, so the caller's
global RNG state is never touched.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .models import (_checked_context_embedding, derive_seed,
                     make_generator, predict_eps_from)


def ddim_timesteps(t_steps: int, nfe: int) -> np.ndarray:
    """Evenly spaced DDIM timesteps, descending, exact boundaries.

    Returns ``nfe`` distinct timesteps from ``t_steps - 1`` down to
    ``0``. ``nfe == t_steps`` returns every timestep; ``nfe == 1``
    returns only the last timestep, so the chain jumps straight from
    noise to the clean-window estimate.
    """
    if not isinstance(t_steps, int) or isinstance(t_steps, bool) \
            or t_steps < 1:
        raise ValueError("t_steps must be a positive int")
    if not isinstance(nfe, int) or isinstance(nfe, bool) or nfe < 1:
        raise ValueError("nfe must be a positive int")
    if nfe > t_steps:
        raise ValueError(
            f"nfe {nfe} exceeds the chain length {t_steps}: a DDIM "
            f"chain cannot evaluate more steps than the model has")
    if nfe == t_steps:
        return np.arange(t_steps - 1, -1, -1, dtype=np.int64)
    if nfe == 1:
        # a one-step chain jumps straight from the last timestep
        return np.array([t_steps - 1], dtype=np.int64)
    steps = np.unique(np.round(
        np.linspace(0, t_steps - 1, nfe)).astype(np.int64))
    if len(steps) != nfe:
        raise ValueError(
            f"nfe {nfe} over {t_steps} steps collides after rounding; "
            f"use a smaller nfe")
    return steps[::-1]


def ddim_step(x_t: torch.Tensor, eps: torch.Tensor, t: int,
              t_prev: int, alphas_cumprod: torch.Tensor,
              eta: float = 0.0,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """One DDIM reverse step from timestep ``t`` to ``t_prev``.

    ``eps`` is the network's added-noise estimate at ``x_t``.
    ``t_prev == -1`` is the final step and returns the clean-window
    estimate directly. ``eta`` in [0, 1] sets the stochasticity:
    0 is deterministic, 1 uses the DDPM posterior variance. With
    ``eta > 0`` the fresh noise comes from ``generator`` (or the
    global RNG when none is given).
    """
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta!r}")
    acp_t = float(alphas_cumprod[int(t)])
    acp_prev = 1.0 if t_prev < 0 else float(alphas_cumprod[int(t_prev)])
    s = math.sqrt(acp_t)
    s1 = math.sqrt(1.0 - acp_t)
    x0_hat = (x_t - s1 * eps) / s
    if t_prev < 0:
        return x0_hat
    sigma = float(eta) * math.sqrt((1.0 - acp_prev) / (1.0 - acp_t)) \
        * math.sqrt(1.0 - acp_t / acp_prev)
    var = 1.0 - acp_prev - sigma * sigma
    # float32 schedules can push the direction variance a rounding
    # error below zero at the earliest timesteps; it is zero there
    var = max(var, 0.0)
    out = math.sqrt(acp_prev) * x0_hat + math.sqrt(var) * eps
    if sigma > 0.0:
        out = out + sigma * torch.randn_like(
            x_t, generator=generator)
    return out


@torch.no_grad()
def sample_ddim(model, regime, nfe: int = 20, eta: float = 0.0,
                seed: int = 0,
                generator: torch.Generator | None = None,
                run_id: str | None = None,
                batch_size: int | None = None,
                past_context=None, past_mask=None):
    """Sample the trained DDPM with a DDIM chain of ``nfe`` steps.

    The interface mirrors ``BookDDPM.sample`` on purpose: the same
    regime validation, the same isolated draw stream (a fresh
    generator seeded from ``seed``, or from ``run_id`` via the
    "samplers:ddim" stream), the same row-order batching, the same
    history-context threading, and the same non-finite guard after
    every step. The only difference is the reverse update: the
    chain visits the ``nfe`` timesteps of :func:`ddim_timesteps`
    instead of every timestep, and the noise estimate comes from
    the model's own objective through the same conversion the
    ancestral sampler uses, so a model trained with a velocity or
    clean-window objective samples correctly here too.

    ``eta`` selects the stochasticity (0 deterministic, 1 the
    ancestral posterior). ``nfe`` must lie in [1, ``model.t_steps``].
    """
    if not isinstance(nfe, int) or isinstance(nfe, bool) or nfe < 1:
        raise ValueError("nfe must be a positive int")
    if nfe > model.t_steps:
        raise ValueError(
            f"nfe {nfe} exceeds the model chain length "
            f"{model.t_steps}")
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta!r}")
    if not isinstance(regime, torch.Tensor):
        raise ValueError("regime must be a torch.Tensor")
    want = model.net.regime_mlp[0].in_features
    if regime.ndim != 2 or regime.shape[1] != want:
        raise ValueError(
            f"regime must have shape (batch, {want}), "
            f"got {tuple(regime.shape)}")
    if regime.shape[0] == 0:
        raise ValueError("regime batch is empty")
    if not bool(torch.isfinite(regime).all()):
        raise ValueError("regime contains non-finite values")
    if batch_size is not None and (
            not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("batch_size must be a positive int or None")
    emb = _checked_context_embedding(
        model.context_encoder, past_context, past_mask,
        regime.shape[0], model.n_features, "BookDDPM")
    if generator is None:
        if run_id is not None:
            seed = derive_seed(run_id, "samplers:ddim")
        generator = make_generator(seed, device=regime.device)
    step = batch_size if batch_size is not None else regime.shape[0]
    parts = []
    with torch.inference_mode():
        for start in range(0, regime.shape[0], step):
            parts.append(_sample_ddim_rows(
                model, regime[start:start + step], generator,
                None if emb is None else emb[start:start + step],
                nfe, float(eta)))
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)


def _sample_ddim_rows(model, regime, generator, context_emb, nfe, eta):
    """One DDIM chain for a single chunk of regime rows."""
    b = regime.shape[0]
    dev = regime.device
    x = torch.randn(b, model.n_features, model.window, device=dev,
                    generator=generator)
    acp = model.alphas_cumprod.to(dev)
    sacp = model.sqrt_acp.to(dev)
    s1acp = model.sqrt_1m_acp.to(dev)
    steps = ddim_timesteps(model.t_steps, nfe)
    for k, t_int in enumerate(steps):
        t = torch.full((b,), int(t_int), dtype=torch.long, device=dev)
        pred = model.net(x, t, regime, context_emb) \
            if model.use_history else model.net(x, t, regime)
        eps = predict_eps_from(pred, x, t, sacp, s1acp,
                               model.objective)
        t_prev = int(steps[k + 1]) if k + 1 < len(steps) else -1
        x = ddim_step(x, eps, int(t_int), t_prev, acp, eta=eta,
                      generator=generator)
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError(
                f"sample_ddim produced non-finite values at "
                f"reverse step {int(t_int)}")
    return x.permute(0, 2, 1)  # (B, L, F)


def pareto_frontier(points: list[dict]) -> list[str]:
    """Pareto-optimal operating points on (val W1, latency).

    ``points`` carry ``method``, ``val_mean_w1`` (lower is better),
    and ``steady_seconds_per_1k`` (lower is better). A point is
    Pareto-optimal when no other point beats it on both axes with at
    least one strict win. The returned method names are ordered by
    increasing latency, so the frontier reads as the speed/quality
    tradeoff instead of a ranking: the fastest point is never claimed
    best, only cheapest.
    """
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["val_mean_w1"] <= p["val_mean_w1"] \
                    and q["steady_seconds_per_1k"] \
                    <= p["steady_seconds_per_1k"] \
                    and (q["val_mean_w1"] < p["val_mean_w1"]
                         or q["steady_seconds_per_1k"]
                         < p["steady_seconds_per_1k"]):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: p["steady_seconds_per_1k"])
    return [p["method"] for p in front]
