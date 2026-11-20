"""P10 challenger: rectified flow matching on the DDPM backbone.

The flow model reuses the shipped Denoiser as its velocity field,
so it carries the same parameter count as the base DDPM at the same
width and trains under the same optimizer-step budget. Training is
the rectified-flow objective: draw t uniform in [0, 1], draw a noise
point x0, interpolate xt = (1 - t) x0 + t x1 toward a real window
x1, and regress the constant velocity x1 - x0. Sampling integrates
the learned velocity from noise to data with plain Euler steps, so
one generated window costs NFE velocity evaluations (20 here,
against the DDPM's 100).

The Denoiser's time embedding was built for integer diffusion
steps 0..99, so the flow time in [0, 1] is scaled by 100 before it
reaches the embedding: the field sees the same time range the
denoiser was designed for, and the scaling is recorded here rather
than hidden.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import N_FEATURES, WINDOW
from .models import (Denoiser, derive_seed, make_generator, scoped_rng)

TIME_SCALE = 100.0


def rectified_flow_loss(pred: torch.Tensor, x1: torch.Tensor,
                       x0: torch.Tensor) -> torch.Tensor:
    """Mean squared error between a predicted velocity and x1 - x0."""
    return torch.mean((pred - (x1 - x0)) ** 2)


class RectifiedFlow(nn.Module):
    """Velocity field for rectified flow matching, DDPM backbone."""

    def __init__(self, n_features: int = N_FEATURES,
                 window: int = WINDOW, regime_dim: int = 2,
                 ch: int = 96, n_blocks: int = 3, nfe: int = 20,
                 seed: int | None = None,
                 run_id: str | None = None):
        super().__init__()
        self.n_features = n_features
        self.window = window
        self.nfe = nfe
        if seed is None and run_id is not None:
            seed = derive_seed(run_id, "rectifiedflow:init")

        def _build():
            return Denoiser(n_features=n_features, window=window,
                            regime_dim=regime_dim, ch=ch,
                            n_blocks=n_blocks)

        if seed is None:
            self.net = _build()
        else:
            with scoped_rng(seed):
                self.net = _build()

    def forward(self, x: torch.Tensor, t01: torch.Tensor,
                regime: torch.Tensor) -> torch.Tensor:
        """Predicted velocity at flow time t01 in [0, 1].

        ``x`` has shape (B, F, L); ``t01`` has shape (B,).
        """
        return self.net(x, t01 * TIME_SCALE, regime)

    def velocity(self, x1: torch.Tensor, regime: torch.Tensor,
                 t01: torch.Tensor,
                 x0: torch.Tensor) -> torch.Tensor:
        """Predicted velocity at the interpolated point for (x1, x0)."""
        t = t01.reshape(-1, 1, 1)
        xt = (1.0 - t) * x0 + t * x1
        return self.forward(xt, t01, regime)

    @torch.no_grad()
    def sample(self, regime: torch.Tensor, seed: int = 0,
               run_id: str | None = None, nfe: int | None = None,
               batch_size: int | None = None) -> torch.Tensor:
        """Euler integration from noise to data, one fresh generator.

        The sampling stream is derived per run ID when given, so it
        never shares draws with the training stream or with another
        family's sampling stream.
        """
        if run_id is not None:
            seed = derive_seed(run_id, "sampling")
        nfe = self.nfe if nfe is None else int(nfe)
        if nfe < 1:
            raise ValueError("nfe must be at least 1")
        gen = make_generator(seed, device=regime.device)
        was_training = self.training
        self.eval()
        try:
            rows = []
            step = batch_size if batch_size is not None \
                else regime.shape[0]
            for start in range(0, regime.shape[0], step):
                rows.append(self._sample_rows(
                    regime[start:start + step], gen, nfe))
            return torch.cat(rows, dim=0)
        finally:
            self.train(was_training)

    def _sample_rows(self, regime: torch.Tensor,
                     gen: torch.Generator, nfe: int) -> torch.Tensor:
        b = regime.shape[0]
        dev = regime.device
        x = torch.randn(b, self.n_features, self.window, device=dev,
                        generator=gen)
        dt = 1.0 / nfe
        for i in range(nfe):
            t_mid = torch.full((b,), (i + 0.5) / nfe, dtype=torch.float32,
                               device=dev)
            x = x + dt * self.forward(x, t_mid, regime)
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError(
                    f"flow sample produced non-finite values at "
                    f"Euler step {i}")
        return x.permute(0, 2, 1)  # (B, L, F)


def train_flow(model: RectifiedFlow, x: torch.Tensor,
               regime: torch.Tensor, epochs: int = 6, batch: int = 32,
               lr: float = 2e-4, seed: int = 0, verbose: bool = False,
               run_id: str | None = None) -> list[float]:
    """Rectified-flow training on an isolated stream.

    Batch order, flow times, and noise draws come from a scoped
    stream seeded by ``run_id`` (stream "train_flow") or by
    ``seed``; the caller's global RNG state is restored exactly on
    return. One optimizer step per batch, so the step budget is
    epochs * ceil(n / batch), the same count the DDPM gets at the
    same epochs and batch size.
    """
    eff_seed = derive_seed(run_id, "train_flow") \
        if run_id is not None else seed
    n = x.shape[0]
    xc = x.permute(0, 2, 1).contiguous()  # (B, F, L)
    with scoped_rng(eff_seed):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        hist = []
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot, nb = 0.0, 0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                xb = xc[idx]
                t01 = torch.rand(len(idx))
                x0 = torch.randn_like(xb)
                pred = model.velocity(xb, regime[idx], t01, x0)
                loss = rectified_flow_loss(pred, xb, x0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            hist.append(tot / nb)
            if verbose:
                print(f"  flow epoch {ep:3d} loss {hist[-1]:.4f}",
                      flush=True)
    return hist
