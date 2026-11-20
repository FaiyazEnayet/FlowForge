"""P10 challenger: a compact causal probabilistic autoregressor.

The smallest fair challenger first, per the source: a GRU reads the
causal within-window history plus the per-window regime conditioning
and predicts the next second as a Gaussian mixture, one step at a
time. The conditioning information is exactly the regime the DDPM
baseline sees; the only history used is the window itself, strictly
causal, so a prediction at step t never sees x_t or anything after
it. Training is teacher-forced negative log-likelihood under the
same optimizer-step budget as the DDPM; sampling draws one step at
a time from the predicted mixture, so one generated window costs 60
sequential GRU steps (NFE 60, one small forward per step).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import N_FEATURES, WINDOW
from .models import derive_seed, make_generator, scoped_rng

REGIME_DIM = 2
LOG_SCALE_MIN = -7.0
LOG_SCALE_MAX = 2.0


def mixture_nll(x: torch.Tensor, logits: torch.Tensor,
                means: torch.Tensor,
                log_scales: torch.Tensor) -> torch.Tensor:
    """Mean mixture negative log-likelihood over all positions.

    ``x`` has shape (..., F), ``logits`` (..., K), ``means`` and
    ``log_scales`` (..., K, F). The per-component log density is
    summed over features, mixed with the softmax weights by
    logsumexp, and averaged over every position.
    """
    scales = torch.exp(log_scales)
    z = (x.unsqueeze(-2) - means) / scales
    log_comp = (-0.5 * z ** 2 - log_scales
                - 0.5 * torch.log(torch.tensor(2.0 * torch.pi,
                                              dtype=x.dtype,
                                              device=x.device))).sum(-1)
    log_pi = torch.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_pi + log_comp, dim=-1).mean()


class CausalGRUMixture(nn.Module):
    """GRU autoregressor with a Gaussian-mixture next-step head.

    Input at step t is the previous second's features plus the
    window regime, so the first step conditions on a zero start and
    the regime only. The head emits, per step, K mixture weights,
    K mean vectors, and K log-scale vectors over the seven features.
    """

    def __init__(self, n_features: int = N_FEATURES,
                 window: int = WINDOW, regime_dim: int = REGIME_DIM,
                 hidden: int = 256, n_components: int = 4,
                 seed: int | None = None,
                 run_id: str | None = None):
        super().__init__()
        self.n_features = n_features
        self.window = window
        self.regime_dim = regime_dim
        self.hidden = hidden
        self.n_components = n_components
        if seed is None and run_id is not None:
            seed = derive_seed(run_id, "causalgru:init")

        def _build():
            gru = nn.GRU(n_features + regime_dim, hidden,
                         batch_first=True)
            head = nn.Linear(
                hidden, n_components * (2 * n_features + 1))
            return gru, head

        if seed is None:
            self.gru, self.head = _build()
        else:
            with scoped_rng(seed):
                self.gru, self.head = _build()

    def forward(self, x: torch.Tensor,
                regime: torch.Tensor) -> tuple[torch.Tensor,
                                               torch.Tensor,
                                               torch.Tensor]:
        """Mixture parameters predicting each step from earlier ones.

        Returns (logits, means, log_scales) with shapes (B, L, K),
        (B, L, K, F), (B, L, K, F). The prediction at step t depends
        only on x_0..x_{t-1} and the regime.
        """
        b, length, _ = x.shape
        start = torch.zeros(b, 1, self.n_features, dtype=x.dtype,
                            device=x.device)
        shifted = torch.cat([start, x[:, :-1, :]], dim=1)
        cond = regime.unsqueeze(1).expand(-1, length, -1)
        inp = torch.cat([shifted, cond], dim=-1)
        h, _ = self.gru(inp)
        out = self.head(h)
        k = self.n_components
        f = self.n_features
        logits = out[:, :, :k]
        means = out[:, :, k:k + k * f].reshape(b, length, k, f)
        log_scales = out[:, :, k + k * f:].reshape(b, length, k, f)
        log_scales = torch.clamp(log_scales, LOG_SCALE_MIN,
                                 LOG_SCALE_MAX)
        return logits, means, log_scales

    @torch.no_grad()
    def generate(self, regime: torch.Tensor, seed: int = 0,
                 run_id: str | None = None,
                 batch_size: int | None = None) -> torch.Tensor:
        """Sample windows one step at a time from a fresh generator.

        The sampling stream is derived per run ID when given, so it
        never shares draws with the training stream or with another
        family's sampling stream. A fixed seed reproduces the same
        windows exactly.
        """
        if run_id is not None:
            seed = derive_seed(run_id, "sampling")
        gen = make_generator(seed, device=regime.device)
        was_training = self.training
        self.eval()
        try:
            rows = []
            step = batch_size if batch_size is not None \
                else regime.shape[0]
            for start in range(0, regime.shape[0], step):
                rows.append(self._generate_rows(
                    regime[start:start + step], gen))
            return torch.cat(rows, dim=0)
        finally:
            self.train(was_training)

    def _generate_rows(self, regime: torch.Tensor,
                       gen: torch.Generator) -> torch.Tensor:
        b = regime.shape[0]
        dev = regime.device
        hidden = None
        x_prev = torch.zeros(b, self.n_features, dtype=torch.float32,
                             device=dev)
        out = []
        for _ in range(self.window):
            inp = torch.cat([x_prev, regime], dim=-1).unsqueeze(1)
            h, hidden = self.gru(inp, hidden)
            raw = self.head(h[:, 0, :])
            k = self.n_components
            f = self.n_features
            logits = raw[:, :k]
            means = raw[:, k:k + k * f].reshape(b, k, f)
            log_scales = torch.clamp(
                raw[:, k + k * f:].reshape(b, k, f),
                LOG_SCALE_MIN, LOG_SCALE_MAX)
            probs = torch.softmax(logits, dim=-1)
            u = torch.rand(b, 1, device=dev, generator=gen)
            comp = (torch.cumsum(probs, dim=-1) < u).sum(dim=-1)
            mu = means[torch.arange(b, device=dev), comp]
            sd = torch.exp(log_scales[torch.arange(b, device=dev),
                                      comp])
            x_prev = mu + sd * torch.randn(b, f, device=dev,
                                           generator=gen)
            out.append(x_prev)
        return torch.stack(out, dim=1)


def train_autoregressive(model: CausalGRUMixture, x: torch.Tensor,
                         regime: torch.Tensor, epochs: int = 6,
                         batch: int = 32, lr: float = 1e-3,
                         seed: int = 0, verbose: bool = False,
                         run_id: str | None = None) -> list[float]:
    """Teacher-forced mixture NLL training on an isolated stream.

    Batch order draws from a scoped stream seeded by ``run_id``
    (stream "train_ar") or by ``seed``; the caller's global RNG
    state is restored exactly on return. One optimizer step per
    batch, so the step budget is epochs * ceil(n / batch), the same
    count the DDPM gets at the same epochs and batch size.
    """
    eff_seed = derive_seed(run_id, "train_ar") \
        if run_id is not None else seed
    n = x.shape[0]
    with scoped_rng(eff_seed):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        hist = []
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot, nb = 0.0, 0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                logits, means, log_scales = model(x[idx], regime[idx])
                loss = mixture_nll(x[idx], logits, means, log_scales)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            hist.append(tot / nb)
            if verbose:
                print(f"  ar epoch {ep:3d} loss {hist[-1]:.4f}",
                      flush=True)
    return hist
