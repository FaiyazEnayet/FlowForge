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

import hashlib
import math
from contextlib import contextmanager

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


def derive_seed(run_id: str, stream: str) -> int:
    """Derive a stable integer seed from a run ID and a stream name.

    Uses SHA-256 over "run_id:stream", never Python's salted hash(), so the
    same run ID yields the same seed in every process. Streams are kept
    apart by name: init, training, and sampling streams never share draws.
    """
    digest = hashlib.sha256(f"{run_id}:{stream}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)


def make_generator(seed: int, device=None) -> torch.Generator:
    """A fresh torch.Generator seeded without touching global RNG state."""
    gen = torch.Generator(device=device) if device is not None \
        else torch.Generator()
    gen.manual_seed(seed)
    return gen


@contextmanager
def scoped_rng(seed: int):
    """Run a block on a seeded RNG stream, then restore global RNG state.

    Inside the block the torch, Python, and NumPy global generators are
    seeded from ``seed``; on exit all three are restored exactly, so the
    caller observes zero shared-global RNG contamination.
    """
    import random
    import numpy as np
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        yield
    finally:
        torch.set_rng_state(torch_state)
        random.setstate(random_state)
        np.random.set_state(numpy_state)


# ---------------------------------------------------------------- VAE baseline

class CondVAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM,
                 n_features: int = N_FEATURES, window: int = WINDOW,
                 regime_dim: int = REGIME_DIM,
                 seed: int | None = None, run_id: str | None = None):
        """Init weights draw from an isolated stream when seeded.

        With ``seed`` (or ``run_id``, resolved via :func:`derive_seed` to
        the "condvae:init" stream) the layer weights are drawn inside a
        scoped RNG block, so build order and unrelated global draws cannot
        change the initialized tensors and the caller's global RNG state
        is untouched. Without either, init uses the global RNG as before.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.window = window
        if seed is None and run_id is not None:
            seed = derive_seed(run_id, "condvae:init")
        flat = n_features * window
        if seed is None:
            self._build_layers(flat, regime_dim)
        else:
            with scoped_rng(seed):
                self._build_layers(flat, regime_dim)

    def _build_layers(self, flat: int, regime_dim: int) -> None:
        self.enc = nn.Sequential(
            nn.Linear(flat + regime_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
        )
        self.mu = nn.Linear(128, self.latent_dim)
        self.logvar = nn.Linear(128, self.latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(self.latent_dim + regime_dim, 128), nn.SiLU(),
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
    def generate(self, regime, n_per_regime: int = 1,
                 generator: torch.Generator | None = None,
                 seed: int | None = None, run_id: str | None = None):
        """Sample latent codes from an isolated stream when requested.

        With ``generator``, ``seed``, or ``run_id`` (resolved to the
        "condvae:generate" stream) the latent draws come from that stream
        and the global RNG state is untouched. Without any of them the
        latent draws use the global RNG, as before.
        """
        b = regime.shape[0] * n_per_regime
        reg = regime.repeat_interleave(n_per_regime, dim=0)
        if generator is None:
            if run_id is not None:
                seed = derive_seed(run_id, "condvae:generate")
            if seed is not None:
                generator = make_generator(seed, device=regime.device)
        if generator is not None:
            z = torch.randn(b, self.latent_dim, device=regime.device,
                            generator=generator)
        else:
            z = torch.randn(b, self.latent_dim, device=regime.device)
        return self.dec(torch.cat([z, reg], dim=1)).view(
            b, self.window, self.n_features)


def vae_loss(x, recon, mu, logvar, beta: float = 1e-3):
    mse = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld, mse, kld


def train_vae(model: CondVAE, x: torch.Tensor, regime: torch.Tensor,
              epochs: int = 25, batch: int = 64, lr: float = 1e-3,
              seed: int = 0, verbose: bool = True,
              run_id: str | None = None) -> list[float]:
    """Train on an isolated stream: batch order and reparameterization
    noise draw from a scoped stream seeded by ``seed`` (or by ``run_id``
    via the "train_vae" stream), and the caller's global RNG state is
    restored exactly when training returns."""
    eff_seed = derive_seed(run_id, "train_vae") if run_id is not None else seed
    with scoped_rng(eff_seed):
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
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32) / half)
        # A buffer, so the frequencies follow .to(), .cuda(), .float(),
        # and .double() moves of the module instead of being rebuilt on
        # CPU in forward (which broke device placement with a device
        # mismatch). Non-persistent: the values are a deterministic
        # function of dim, so they stay out of state dicts and existing
        # checkpoints keep their exact key set.
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
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
                 seed: int | None = None, run_id: str | None = None,
                 **kw):
        """Init weights draw from an isolated stream when seeded, exactly
        as :class:`CondVAE` does (stream "bookddpm:init" for a run ID).
        Without a seed or run ID, init uses the global RNG as before."""
        super().__init__()
        if seed is None and run_id is not None:
            seed = derive_seed(run_id, "bookddpm:init")
        if seed is None:
            self.net = Denoiser(n_features=n_features, window=window, ch=ch,
                                n_blocks=n_blocks, **kw)
        else:
            with scoped_rng(seed):
                self.net = Denoiser(n_features=n_features, window=window,
                                    ch=ch, n_blocks=n_blocks, **kw)
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
    def sample(self, regime, seed: int = 0,
               generator: torch.Generator | None = None,
               run_id: str | None = None,
               batch_size: int | None = None):
        """Sample the reverse chain from an isolated stream.

        With ``generator`` the chain draws from it; otherwise a fresh
        generator is seeded from ``seed`` (or from ``run_id`` via the
        "bookddpm:sample" stream). The global RNG state is never touched,
        unlike the previous implementation which reset it via set_seed.
        A fresh generator seeded identically reproduces the old draw
        sequence, so fixed-seed outputs are unchanged.

        The chain runs under torch.inference_mode. ``batch_size`` chunks
        the regime rows into explicit batches, drawn from the same
        generator in row order; a batched call is reproducible for a
        fixed seed and batch size, but chunking changes the draw
        blocking, so batched values differ from the unbatched stream.
        The default unbatched path keeps the exact pre-existing draw
        sequence and outputs.

        Failed inputs are rejected instead of sampled from: a non-tensor,
        mis-shaped, empty, or non-finite regime raises ValueError, as
        does a non-positive batch_size. After every reverse step the
        chain state is checked and a non-finite value raises
        FloatingPointError naming the step, instead of silently
        returning non-finite windows.
        """
        if not isinstance(regime, torch.Tensor):
            raise ValueError("regime must be a torch.Tensor")
        want = self.net.regime_mlp[0].in_features
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
        if generator is None:
            if run_id is not None:
                seed = derive_seed(run_id, "bookddpm:sample")
            generator = make_generator(seed, device=regime.device)
        step = batch_size if batch_size is not None else regime.shape[0]
        parts = []
        with torch.inference_mode():
            for start in range(0, regime.shape[0], step):
                parts.append(self._sample_rows(
                    regime[start:start + step], generator))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    def _sample_rows(self, regime, generator):
        """One reverse chain for a single chunk of regime rows."""
        b = regime.shape[0]
        dev = regime.device
        x = torch.randn(b, self.n_features, self.window, device=dev,
                        generator=generator)
        betas = self.betas.to(dev)
        acp = self.alphas_cumprod.to(dev)
        for i in reversed(range(self.t_steps)):
            t = torch.full((b,), i, dtype=torch.long, device=dev)
            eps = self.net(x, t, regime)
            a = 1.0 - betas[i]
            ac = acp[i]
            x = (x - betas[i] / torch.sqrt(1 - ac) * eps) / torch.sqrt(a)
            if i > 0:
                x = x + torch.sqrt(betas[i]) * torch.randn_like(
                    x, generator=generator)
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError(
                    f"sample produced non-finite values at "
                    f"reverse step {i}")
        return x.permute(0, 2, 1)  # (B, L, F)


def train_ddpm(model: BookDDPM, x: torch.Tensor, regime: torch.Tensor,
               epochs: int = 12, batch: int = 32, lr: float = 2e-4,
               seed: int = 0, verbose: bool = True,
               ema_decay: float | None = None,
               run_id: str | None = None) -> tuple[list[float], EMA | None]:
    """Train on an isolated stream, as :func:`train_vae` does (stream
    "train_ddpm" for a run ID). Batch order and the per-step noise draws
    in :meth:`BookDDPM.p_loss` come from the scoped stream, and the
    caller's global RNG state is restored exactly on return."""
    eff_seed = derive_seed(run_id, "train_ddpm") if run_id is not None else seed
    with scoped_rng(eff_seed):
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
                    seed: int = 0,
                    generator: torch.Generator | None = None,
                    run_id: str | None = None,
                    batch_size: int | None = None) -> torch.Tensor:
    """Sample from EMA weights, then restore the training weights."""
    ema.apply_to(model)
    try:
        return model.sample(regime, seed=seed, generator=generator,
                            run_id=run_id, batch_size=batch_size)
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
