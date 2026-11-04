"""Generative models for book-state windows.

Two models, both conditioned on regime features (recent volatility, spread):

(a) CondVAE: a small conditional variational autoencoder baseline. MLP
    encoder/decoder on flattened windows. Small on purpose: it is the
    baseline the diffusion model has to beat.

(b) BookDDPM: a small DDPM diffusion model over sequences. A 1D conv net
    predicts the added noise at each timestep, conditioned on the diffusion
    timestep and the regime vector. T=100 steps, linear beta schedule.

(c) StructuralBookDDPM (experimental): a BookDDPM over the six
    structural features (all but imbalance), with a hurdle gate for
    the trade count; imbalance is reconstructed deterministically at
    decode time instead of being generated as a redundant channel.

Everything is CPU sized. No trading, no order execution, no exchange
connectivity anywhere in this file: these are pure sequence generators.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FEATURE_NAMES, N_FEATURES, WINDOW

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
    def __init__(self, ch: int, cond_dim: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=dilation,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=dilation,
                               dilation=dilation)
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


def _conditioning(regime_dim: int, tdim: int):
    """The shared time/regime conditioning trunk every denoiser uses."""
    time_embed = TimeEmbed(tdim)
    regime_mlp = nn.Sequential(nn.Linear(regime_dim, tdim), nn.SiLU(),
                               nn.Linear(tdim, tdim))
    return time_embed, regime_mlp, tdim * 2


class DilatedTCNDenoiser(nn.Module):
    """Dilated residual TCN denoiser (P04 candidate).

    Same conditioning interface and output head as :class:`Denoiser`, but
    the same-resolution residual stack follows the explicit dilation
    cycle 1, 2, 4, 8, 16, 32 (one ResBlock per dilation, both of its
    convolutions at that dilation). The convolutional receptive field is
    257 positions against the 60-position window, so early-window
    offsets reach late predictions through the feature path. Channel
    width 64 keeps the parameter count within the part 1 budget rule
    (5 percent of the baseline Denoiser count).
    """

    DILATION_CYCLE = (1, 2, 4, 8, 16, 32)

    def __init__(self, n_features: int = N_FEATURES, window: int = WINDOW,
                 regime_dim: int = REGIME_DIM, ch: int = 64,
                 tdim: int = 64):
        super().__init__()
        self.time_embed, self.regime_mlp, cond_dim = _conditioning(
            regime_dim, tdim)
        self.in_conv = nn.Conv1d(n_features, ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResBlock(ch, cond_dim, dilation=d)
             for d in self.DILATION_CYCLE])
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


class CompactUNetDenoiser(nn.Module):
    """Compact downsample/upsample 1-D U-Net denoiser (P04 candidate).

    Two stride-2 stages down to quarter resolution, a two-convolution
    bottleneck, and mirrored nearest-neighbour upsampling stages with
    additive skip connections from the encoder tensors at the same
    resolution. Upsampling targets the skip tensor's length, so odd
    input lengths keep their exact shape. The longest encoder-to-decoder
    path has receptive field 37 (half width 18): wider than the baseline
    CNN's 17, but not full-window context, and it is not claimed as such.
    Conditioning is one shared projection of the time/regime vector,
    added at the full-resolution stage, the bottleneck, and the last
    decoder stage. Channel width 96 keeps the parameter count within
    the part 1 budget rule.
    """

    def __init__(self, n_features: int = N_FEATURES, window: int = WINDOW,
                 regime_dim: int = REGIME_DIM, ch: int = 96,
                 tdim: int = 64):
        super().__init__()
        self.time_embed, self.regime_mlp, cond_dim = _conditioning(
            regime_dim, tdim)
        self.cond = nn.Linear(cond_dim, ch)
        self.in_conv = nn.Conv1d(n_features, ch, 3, padding=1)
        self.in_norm = nn.GroupNorm(8, ch)
        self.down1 = nn.Conv1d(ch, ch, 3, stride=2, padding=1)
        self.enc2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.enc2_norm = nn.GroupNorm(8, ch)
        self.down2 = nn.Conv1d(ch, ch, 3, stride=2, padding=1)
        self.mid1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.mid2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.mid_norm = nn.GroupNorm(8, ch)
        self.dec1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.dec1_norm = nn.GroupNorm(8, ch)
        self.dec2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(),
                                 nn.Conv1d(ch, n_features, 3, padding=1))
        self.window = window
        self.n_features = n_features

    def forward(self, x, t, regime):
        # x: (B, F, L)
        c = torch.cat([self.time_embed(t), self.regime_mlp(regime)], dim=1)
        cond = self.cond(c).unsqueeze(-1)
        s1 = F.silu(self.in_norm(self.in_conv(x))) + cond
        h = F.silu(self.enc2_norm(self.enc2(F.silu(self.down1(s1)))))
        s2 = h
        h = F.silu(self.down2(h))
        h = F.silu(self.mid_norm(self.mid2(F.silu(self.mid1(h))))) + cond
        h = F.interpolate(h, size=s2.shape[-1], mode="nearest") + s2
        h = F.silu(self.dec1_norm(self.dec1(h)))
        h = F.interpolate(h, size=s1.shape[-1], mode="nearest") + s1
        h = F.silu(self.dec2(h)) + cond
        return self.out(h)


class AttentionDenoiser(nn.Module):
    """Attention denoiser (P04 candidate): one global mixing layer.

    An input convolution, a single multi-head self-attention block over
    positions (no mask, so every position attends to every other
    position: global context by construction) with a position-wise
    feed-forward network, and an output convolution. This is the
    minimal global-mixing variant declared in the part 1 spec, not a
    transformer stack. Channel width 124 with a 4x feed-forward keeps
    the parameter count within the part 1 budget rule.
    """

    def __init__(self, n_features: int = N_FEATURES, window: int = WINDOW,
                 regime_dim: int = REGIME_DIM, ch: int = 124,
                 tdim: int = 64, n_heads: int = 4):
        super().__init__()
        self.time_embed, self.regime_mlp, cond_dim = _conditioning(
            regime_dim, tdim)
        self.cond = nn.Linear(cond_dim, ch)
        self.in_conv = nn.Conv1d(n_features, ch, 3, padding=1)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(nn.Linear(ch, 4 * ch), nn.SiLU(),
                                 nn.Linear(4 * ch, ch))
        self.out = nn.Sequential(nn.GroupNorm(1, ch), nn.SiLU(),
                                 nn.Conv1d(ch, n_features, 3, padding=1))
        self.window = window
        self.n_features = n_features

    def forward(self, x, t, regime):
        # x: (B, F, L)
        c = torch.cat([self.time_embed(t), self.regime_mlp(regime)], dim=1)
        h = self.in_conv(x) + self.cond(c).unsqueeze(-1)
        z = h.transpose(1, 2)  # (B, L, ch)
        a, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm1(z + a)
        z = self.norm2(z + self.ffn(z))
        return self.out(z.transpose(1, 2))


# Denoiser architectures selectable in BookDDPM. "cnn" is the shipped
# default and its construction is unchanged; the others are the P04
# candidates declared as layer specs in examples/architecture_ablate.py.
DENOISER_ARCHITECTURES = {
    "cnn": Denoiser,
    "dilated_tcn": DilatedTCNDenoiser,
    "compact_unet": CompactUNetDenoiser,
    "attention": AttentionDenoiser,
}


def _imbalance_from_log_sizes(log_bid: np.ndarray,
                              log_ask: np.ndarray) -> np.ndarray:
    """Imbalance implied by log sizes, sampler-side (NumPy mirror of
    evaluate.derive_imbalance; evaluate imports this module, so the
    formula lives here in private form and the two are pinned to
    identical outputs by the C04 tests). Overflow-safe: cells whose
    exponentials overflow float64 use the algebraically identical
    max-shifted form."""
    lb = np.asarray(log_bid, dtype=np.float64)
    la = np.asarray(log_ask, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        bid = np.exp(lb)
        ask = np.exp(la)
        denom = bid + ask
        out = (bid - ask) / np.where(denom > 0, denom, 1.0)
        bad = ~np.isfinite(out)
        if bad.any():
            m = np.maximum(lb, la)
            b2 = np.exp(lb - m)
            a2 = np.exp(la - m)
            out = np.where(bad, (b2 - a2) / (b2 + a2), out)
    return np.clip(out, -1.0, 1.0)


def _constrain_imbalance_np(gen: np.ndarray, feature_mean: np.ndarray,
                            feature_std: np.ndarray) -> np.ndarray:
    """NumPy mirror of evaluate.constrain_imbalance for the sampler:
    replace the imbalance channel with the value implied by the
    window's own log bid/ask sizes."""
    phys = gen * feature_std + feature_mean
    j_bid = FEATURE_NAMES.index("log_bid_sz")
    j_ask = FEATURE_NAMES.index("log_ask_sz")
    j_imb = FEATURE_NAMES.index("imbalance")
    implied = _imbalance_from_log_sizes(phys[:, :, j_bid],
                                        phys[:, :, j_ask])
    fixed = np.array(gen, dtype=np.float64, copy=True)
    fixed[:, :, j_imb] = (implied - feature_mean[j_imb]) \
        / feature_std[j_imb]
    return fixed


class BookDDPM(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, window: int = WINDOW,
                 t_steps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.08, ch: int = 96, n_blocks: int = 3,
                 seed: int | None = None, run_id: str | None = None,
                 arch: str = "cnn", **kw):
        """Init weights draw from an isolated stream when seeded, exactly
        as :class:`CondVAE` does (stream "bookddpm:init" for a run ID).
        Without a seed or run ID, init uses the global RNG as before.

        ``arch`` selects the denoiser from DENOISER_ARCHITECTURES. The
        default "cnn" is the shipped Denoiser, constructed exactly as
        before (``ch`` and ``n_blocks`` apply to it); the P04 candidate
        architectures take their budget-matched widths by default and
        accept overrides such as ``ch`` through ``**kw``.
        """
        super().__init__()
        if arch not in DENOISER_ARCHITECTURES:
            raise ValueError(f"unknown denoiser architecture: {arch!r}")
        denoiser_cls = DENOISER_ARCHITECTURES[arch]
        if seed is None and run_id is not None:
            seed = derive_seed(run_id, "bookddpm:init")

        def _build():
            if arch == "cnn":
                return denoiser_cls(n_features=n_features, window=window,
                                    ch=ch, n_blocks=n_blocks, **kw)
            return denoiser_cls(n_features=n_features, window=window, **kw)

        if seed is None:
            self.net = _build()
        else:
            with scoped_rng(seed):
                self.net = _build()
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
               batch_size: int | None = None,
               mode: str = "raw",
               feature_mean=None, feature_std=None):
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

        ``mode`` selects the sampling mode. ``"raw"`` (the default)
        returns the chain's windows untouched. ``"constrained"``
        returns windows whose imbalance channel has been replaced by
        the value derived solely from each window's own generated log
        bid/ask sizes (the same constrained decoding the evaluation
        applies, mirrored sampler-side); it requires the training
        ``feature_mean``/``feature_std`` so the derivation happens in
        physical units. Both modes share the identical draw stream:
        for a fixed seed, the constrained output is exactly the
        constrained transform of the raw output.

        Failed inputs are rejected instead of sampled from: a non-tensor,
        mis-shaped, empty, or non-finite regime raises ValueError, as
        does a non-positive batch_size. After every reverse step the
        chain state is checked and a non-finite value raises
        FloatingPointError naming the step, instead of silently
        returning non-finite windows.
        """
        if mode not in ("raw", "constrained"):
            raise ValueError(f"unknown sampling mode: {mode!r}")
        if mode == "constrained" and (feature_mean is None
                                      or feature_std is None):
            raise ValueError(
                "constrained sampling requires feature_mean and "
                "feature_std")
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
            out = parts[0]
        else:
            out = torch.cat(parts, dim=0)
        if mode == "constrained":
            arr = out.detach().cpu().numpy().astype(np.float64)
            fixed = _constrain_imbalance_np(
                arr, np.asarray(feature_mean, dtype=np.float64),
                np.asarray(feature_std, dtype=np.float64))
            return torch.from_numpy(fixed).to(device=out.device,
                                              dtype=out.dtype)
        return out

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


# ------------------------------------- structural variant (C04)

STRUCTURAL_FEATURE_NAMES = [name for name in FEATURE_NAMES
                            if name != "imbalance"]
_N_STRUCT = len(STRUCTURAL_FEATURE_NAMES)
_STRUCT_J = {name: i for i, name in enumerate(STRUCTURAL_FEATURE_NAMES)}


def build_structural_dataset(windows: np.ndarray,
                             feature_mean: np.ndarray,
                             feature_std: np.ndarray):
    """Map full standardized windows into the structural feature space.

    Returns (structural_windows, struct_mean, struct_std). Every
    channel keeps the full feature's standardization except the
    spread channel: there the structural value is log(spread_bps)
    rather than log1p(spread_bps), and its mean/std are the supplied
    windows' own statistics for that re-expressed channel, so the
    structural space is self-contained and the mapping is exactly
    invertible by :func:`decode_structural_windows`.
    """
    w = np.asarray(windows, dtype=np.float64)
    fm = np.asarray(feature_mean, dtype=np.float64)
    fs = np.asarray(feature_std, dtype=np.float64)
    phys = w * fs + fm
    j_spr = FEATURE_NAMES.index("log_spread_bps")
    spread_bps = np.expm1(phys[:, :, j_spr])
    log_spread = np.log(np.clip(spread_bps, 1e-9, None))
    s_mean = np.array([fm[FEATURE_NAMES.index(n)]
                       for n in STRUCTURAL_FEATURE_NAMES])
    s_std = np.array([fs[FEATURE_NAMES.index(n)]
                      for n in STRUCTURAL_FEATURE_NAMES])
    j = _STRUCT_J["log_spread_bps"]
    s_mean[j] = float(log_spread.mean())
    s_std[j] = float(log_spread.std()) or 1.0
    cols = []
    for name in STRUCTURAL_FEATURE_NAMES:
        if name == "log_spread_bps":
            cols.append((log_spread - s_mean[j]) / s_std[j])
        else:
            cols.append(w[:, :, FEATURE_NAMES.index(name)])
    struct = np.stack(cols, axis=2)
    return struct.astype(np.float32), s_mean, s_std


def decode_structural_windows(struct: np.ndarray,
                              struct_mean: np.ndarray,
                              struct_std: np.ndarray,
                              feature_mean: np.ndarray,
                              feature_std: np.ndarray,
                              gate_prob: np.ndarray) -> np.ndarray:
    """Structural windows -> full standardized windows, deterministically.

    Reconstruction, per (window, second):

    - log return and the two log sizes pass through unchanged;
    - the spread is exp of the structural log-spread channel (an
      invertible positive transform), so the implied spread in bps
      is strictly positive by construction, then re-expressed as the
      standard log1p feature;
    - the trade count goes through the hurdle decode: where the
      gate probability says "no trades" (gate >= 0.5) the count is
      exactly 0; otherwise it is the rounded model count, at least
      1. Counts are therefore nonnegative integers by construction;
    - signed flow is set to exactly 0 wherever the decoded count is
      0 (the audit's coherence rule);
    - imbalance is not generated at all: it is derived from the
      decoded log sizes, exactly as the audit derives it.

    Decoded windows therefore satisfy every audited constraint by
    construction. The structural features are per-second
    top-of-book aggregates: they summarize the book but do not
    encode the order-message stream, queue positions, or individual
    orders, and nothing here claims that full message-level
    execution could be reconstructed from top-of-book features
    alone.
    """
    s = np.asarray(struct, dtype=np.float64)
    sm = np.asarray(struct_mean, dtype=np.float64)
    ss = np.asarray(struct_std, dtype=np.float64)
    fm = np.asarray(feature_mean, dtype=np.float64)
    fs = np.asarray(feature_std, dtype=np.float64)
    phys = s * ss + sm  # structural feature values
    fp = np.zeros(s.shape[:2] + (N_FEATURES,), dtype=np.float64)
    for name in ("log_ret", "log_bid_sz", "log_ask_sz"):
        fp[:, :, FEATURE_NAMES.index(name)] = \
            phys[:, :, _STRUCT_J[name]]
    # log-spread is clipped before exp only as a decode safety bound
    # (e^30 bps is beyond any plausible book); the transform pair
    # itself is exp/log, exactly invertible inside the bound.
    log_spread = np.clip(phys[:, :, _STRUCT_J["log_spread_bps"]],
                         -30.0, 30.0)
    fp[:, :, FEATURE_NAMES.index("log_spread_bps")] = \
        np.log1p(np.exp(log_spread))
    with np.errstate(over="ignore", invalid="ignore"):
        count_raw = np.expm1(phys[:, :, _STRUCT_J["log_trade_count"]])
    count_raw = np.clip(np.nan_to_num(count_raw, nan=0.0), 0.0, 1e12)
    gate = np.asarray(gate_prob, dtype=np.float64)
    count = np.where(gate >= 0.5, 0.0,
                     np.maximum(1.0, np.round(count_raw)))
    fp[:, :, FEATURE_NAMES.index("log_trade_count")] = np.log1p(count)
    flow = phys[:, :, _STRUCT_J["signed_flow"]]
    fp[:, :, FEATURE_NAMES.index("signed_flow")] = \
        np.where(count > 0, flow, 0.0)
    fp[:, :, FEATURE_NAMES.index("imbalance")] = \
        _imbalance_from_log_sizes(
            fp[:, :, FEATURE_NAMES.index("log_bid_sz")],
            fp[:, :, FEATURE_NAMES.index("log_ask_sz")])
    return ((fp - fm) / fs).astype(np.float32)


class StructuralDenoiser(nn.Module):
    """Denoiser over the structural feature space, plus a hurdle gate.

    Same conditioning trunk and residual stack as :class:`Denoiser`,
    with one extra 1x1 convolution head: the gate logit per
    (window, second), trained to predict whether the trade count is
    zero, which the hurdle decode consumes. The noise-prediction
    output has one channel per structural feature; there is no
    imbalance channel anywhere in this network.
    """

    def __init__(self, n_features: int = _N_STRUCT,
                 window: int = WINDOW, regime_dim: int = REGIME_DIM,
                 ch: int = 96, tdim: int = 64, n_blocks: int = 3):
        super().__init__()
        self.time_embed, self.regime_mlp, cond_dim = _conditioning(
            regime_dim, tdim)
        self.in_conv = nn.Conv1d(n_features, ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResBlock(ch, cond_dim) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(),
                                 nn.Conv1d(ch, n_features, 3,
                                           padding=1))
        self.gate = nn.Conv1d(ch, 1, 1)
        self.window = window
        self.n_features = n_features

    def _trunk(self, x, t, regime):
        c = torch.cat([self.time_embed(t), self.regime_mlp(regime)],
                      dim=1)
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h, c)
        return h

    def forward(self, x, t, regime):
        return self.out(self._trunk(x, t, regime))

    def forward_with_gate(self, x, t, regime):
        h = self._trunk(x, t, regime)
        return self.out(h), self.gate(h)

    def gate_logit(self, x, t, regime):
        return self.gate(self._trunk(x, t, regime))


class StructuralBookDDPM(BookDDPM):
    """Experimental structural DDPM (C04): no redundant imbalance output.

    A BookDDPM over the six structural features (FEATURE_NAMES minus
    imbalance). Training is the usual diffusion loss on those six
    channels plus a binary cross-entropy on the hurdle gate against
    the observed zero-count indicator, so the gate learns the
    zero/nonzero split of the trade count while the count channel
    itself models the positive part. Sampling runs the ordinary
    reverse chain in structural space and then decodes
    deterministically with :func:`decode_structural_windows`:
    imbalance is reconstructed from the generated sizes, spread and
    sizes pass through invertible positive transforms, and the
    trade count is a hurdle decode, so decoded windows satisfy the
    physical audit by construction rather than by post-hoc repair
    of a model that emitted the redundant channel.

    Scope limit, stated plainly: the structural features are
    per-second top-of-book aggregates. They do not encode the
    order-message stream, queue positions, or individual orders,
    and this variant does not claim that full message-level
    execution could be reconstructed from top-of-book features
    alone.
    """

    def __init__(self, struct_mean, struct_std, feature_mean,
                 feature_std, gate_weight: float = 1.0,
                 ch: int = 96, n_blocks: int = 3, t_steps: int = 100,
                 seed: int | None = None, run_id: str | None = None):
        super().__init__(n_features=_N_STRUCT, window=WINDOW,
                         t_steps=t_steps, ch=ch, n_blocks=n_blocks,
                         seed=seed, run_id=run_id, arch="cnn")
        self.net = StructuralDenoiser(n_features=_N_STRUCT,
                                      window=WINDOW, ch=ch,
                                      n_blocks=n_blocks)
        self.gate_weight = float(gate_weight)
        self._struct_mean_np = np.asarray(struct_mean,
                                          dtype=np.float64)
        self._struct_std_np = np.asarray(struct_std, dtype=np.float64)
        self._feature_mean_np = np.asarray(feature_mean,
                                           dtype=np.float64)
        self._feature_std_np = np.asarray(feature_std,
                                          dtype=np.float64)
        self.register_buffer(
            "_struct_mean",
            torch.tensor(self._struct_mean_np, dtype=torch.float32))
        self.register_buffer(
            "_struct_std",
            torch.tensor(self._struct_std_np, dtype=torch.float32))

    def p_loss(self, x0, regime):
        b = x0.shape[0]
        t = torch.randint(0, self.t_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred, gate = self.net.forward_with_gate(xt, t, regime)
        mse = F.mse_loss(pred, noise)
        j_cnt = _STRUCT_J["log_trade_count"]
        phys_cnt = x0[:, j_cnt, :] * self._struct_std[j_cnt] \
            + self._struct_mean[j_cnt]
        zero = (torch.round(torch.expm1(phys_cnt)) == 0).float()
        gate_loss = F.binary_cross_entropy_with_logits(
            gate.squeeze(1), zero)
        return mse + self.gate_weight * gate_loss

    @torch.no_grad()
    def decode(self, struct: torch.Tensor,
               regime: torch.Tensor) -> torch.Tensor:
        """Decode structural samples to full standardized windows.

        Deterministic: the gate is evaluated once, at t=0, on the
        finished structural sample, and the reconstruction itself is
        a fixed function (see decode_structural_windows).
        """
        x = struct.detach().cpu().permute(0, 2, 1)  # (B, F, L)
        t = torch.zeros(x.shape[0], dtype=torch.long)
        with torch.inference_mode():
            gate = torch.sigmoid(
                self.net.gate_logit(
                    x, t, regime.detach().cpu())).squeeze(1).numpy()
        full = decode_structural_windows(
            struct.detach().cpu().numpy(), self._struct_mean_np,
            self._struct_std_np, self._feature_mean_np,
            self._feature_std_np, gate)
        return torch.from_numpy(full).to(device=struct.device)

    @torch.no_grad()
    def sample_structural(self, regime, seed: int = 0,
                          generator: torch.Generator | None = None,
                          run_id: str | None = None,
                          batch_size: int | None = None):
        """Sample the reverse chain in structural space and decode."""
        struct = self.sample(regime, seed=seed, generator=generator,
                             run_id=run_id, batch_size=batch_size)
        return self.decode(struct, regime)


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
