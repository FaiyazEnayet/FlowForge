"""Generative models for book-state windows.

Two models, both conditioned on regime features (recent volatility, spread):

(a) CondVAE: a small conditional variational autoencoder baseline. MLP
    encoder/decoder on flattened windows. Small on purpose: it is the
    baseline the diffusion model has to beat.

(b) BookDDPM: a small DDPM diffusion model over sequences. A 1D conv net
    predicts the added noise at each timestep, conditioned on the diffusion
    timestep and the regime vector. T=100 steps, linear beta schedule.

Everything is CPU sized. No trading, no order execution, no exchange
connectivity anywhere in this file: these are pure sequence generators.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_FEATURES, WINDOW

LATENT_DIM = 16
REGIME_DIM = 2


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- VAE baseline

class CondVAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM,
                 n_features: int = N_FEATURES, window: int = WINDOW,
                 regime_dim: int = REGIME_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.window = window
        flat = n_features * window
        self.enc = nn.Sequential(
            nn.Linear(flat + regime_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + regime_dim, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, flat),
        )

    def forward(self, x, regime):
        b = x.shape[0]
        h = self.enc(torch.cat([x.view(b, -1), regime], dim=1))
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.dec(torch.cat([z, regime], dim=1)).view_as(x)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, regime, n_per_regime: int = 1):
        b = regime.shape[0] * n_per_regime
        reg = regime.repeat_interleave(n_per_regime, dim=0)
        z = torch.randn(b, self.latent_dim, device=regime.device)
        return self.dec(torch.cat([z, reg], dim=1)).view(
            b, self.window, self.n_features)


def vae_loss(x, recon, mu, logvar, beta: float = 1e-3):
    mse = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld, mse, kld


def train_vae(model: CondVAE, x: torch.Tensor, regime: torch.Tensor,
              epochs: int = 25, batch: int = 64, lr: float = 1e-3,
              seed: int = 0, verbose: bool = True) -> list[float]:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            recon, mu, logvar = model(x[idx], regime[idx])
            loss, _, _ = vae_loss(x[idx], recon, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        hist.append(tot / nb)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  vae epoch {ep:3d} loss {hist[-1]:.4f}", flush=True)
    return hist


# ------------------------------------------------------------- DDPM diffusion

class TimeEmbed(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=1))


class ResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, ch)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, h, c):
        res = h
        h = self.conv1(F.silu(self.norm1(h)))
        h = h + self.cond(c).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + res


class Denoiser(nn.Module):
    """Predicts noise added to a book-state sequence."""

    def __init__(self, n_features: int = N_FEATURES, window: int = WINDOW,
                 regime_dim: int = REGIME_DIM, ch: int = 96, tdim: int = 64,
                 n_blocks: int = 3):
        super().__init__()
        self.time_embed = TimeEmbed(tdim)
        self.regime_mlp = nn.Sequential(nn.Linear(regime_dim, tdim), nn.SiLU(),
                                        nn.Linear(tdim, tdim))
        cond_dim = tdim * 2
        self.in_conv = nn.Conv1d(n_features, ch, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(ch, cond_dim) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(),
                                 nn.Conv1d(ch, n_features, 3, padding=1))
        self.window = window
        self.n_features = n_features

    def forward(self, x, t, regime):
        # x: (B, F, L)
        c = torch.cat([self.time_embed(t), self.regime_mlp(regime)], dim=1)
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h, c)
        return self.out(h)


class BookDDPM(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, window: int = WINDOW,
                 t_steps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.08, ch: int = 96, n_blocks: int = 3,
                 **kw):
        super().__init__()
        self.net = Denoiser(n_features=n_features, window=window, ch=ch,
                            n_blocks=n_blocks, **kw)
        self.t_steps = t_steps
        # Linear schedule with beta_end chosen so alpha_bar_T ~ 0.02: the
        # reverse chain must start from ~pure noise, and per-step betas must
        # stay small enough that an imperfect denoiser does not blow up the
        # ancestral sampler. (A cosine schedule was tried and diverged.)
        betas = torch.linspace(beta_start, beta_end, t_steps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_1m_acp", torch.sqrt(1.0 - acp))
        # sanity: at realistic step counts the chain must actually reach pure
        # noise, otherwise sampling starts from the wrong distribution and
        # generated variance collapses (this exact bug bit once already)
        if t_steps >= 50:
            assert float(acp[-1]) < 0.05, f"alpha_bar_T={float(acp[-1]):.3f}; schedule broken"
        self.window = window
        self.n_features = n_features

    def q_sample(self, x0, t, noise):
        # x: (B, F, L)
        s = self.sqrt_acp[t].view(-1, 1, 1)
        s1 = self.sqrt_1m_acp[t].view(-1, 1, 1)
        return s * x0 + s1 * noise

    def p_loss(self, x0, regime):
        b = x0.shape[0]
        t = torch.randint(0, self.t_steps, (b,))
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.net(xt, t, regime)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, regime, seed: int = 0):
        set_seed(seed)
        b = regime.shape[0]
        dev = regime.device
        x = torch.randn(b, self.n_features, self.window, device=dev)
        betas = self.betas.to(dev)
        acp = self.alphas_cumprod.to(dev)
        for i in reversed(range(self.t_steps)):
            t = torch.full((b,), i, dtype=torch.long, device=dev)
            eps = self.net(x, t, regime)
            a = 1.0 - betas[i]
            ac = acp[i]
            x = (x - betas[i] / torch.sqrt(1 - ac) * eps) / torch.sqrt(a)
            if i > 0:
                x = x + torch.sqrt(betas[i]) * torch.randn_like(x)
        return x.permute(0, 2, 1)  # (B, L, F)


def train_ddpm(model: BookDDPM, x: torch.Tensor, regime: torch.Tensor,
               epochs: int = 12, batch: int = 32, lr: float = 2e-4,
               seed: int = 0, verbose: bool = True,
               ema_decay: float | None = None) -> tuple[list[float], EMA | None]:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model, ema_decay) if ema_decay is not None else None
    n = x.shape[0]
    xc = x.permute(0, 2, 1).contiguous()  # (B, F, L)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = model.p_loss(xc[idx], regime[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item()
            nb += 1
        hist.append(tot / nb)
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"  ddpm epoch {ep:3d} loss {hist[-1]:.4f}", flush=True)
    return hist, ema


@torch.no_grad()
def sample_with_ema(model: BookDDPM, ema: EMA, regime: torch.Tensor,
                    seed: int = 0) -> torch.Tensor:
    """Sample from EMA weights, then restore the training weights."""
    ema.apply_to(model)
    try:
        return model.sample(regime, seed=seed)
    finally:
        ema.restore(model)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------ EMA averaging

class EMA:
    """Exponential moving average of model weights. Standard DDPM practice:
    sampling from the EMA weights is more stable than from the raw
    training weights, especially late in training."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Swap EMA weights in, keeping a backup of the current weights."""
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n].to(p.device))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        assert self._backup is not None, "apply_to() must be called first"
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._backup[n].to(p.device))
        self._backup = None
