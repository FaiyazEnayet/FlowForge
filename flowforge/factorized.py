"""Factorized feasible top-of-book representation (P02).

The raw representation emits log bid size and log ask size as two
independent channels, so a generated window's size pair can imply
any imbalance at all, including one that contradicts the window's
own imbalance channel; the audit in flowforge.evaluate measures
exactly that disagreement. The structural variant (C04) removes
the redundancy from the other side: it drops the imbalance channel
and derives it from the two generated sizes.

This module implements the complementary factorization named in
the P02 contract. The generator works on six channels:

  0. log midprice return (unchanged)
  1. log spread in bps (positive by construction at decode: the
     channel models log(spread_bps), and the spread is exp of it)
  2. log total depth, log(bid_sz + ask_sz)
  3. imbalance in [-1, 1]
  4. signed trade flow (unchanged)
  5. log1p(trade count)

and the size pair is DERIVED, never generated independently:

  bid_sz = depth * (1 + imbalance) / 2
  ask_sz = depth * (1 - imbalance) / 2

so bid + ask = depth and (bid - ask) / (bid + ask) = imbalance
hold as exact algebra on every decoded window. The encode/decode
pair is exactly invertible on physically valid windows: encoding
then decoding a real window reproduces it. Normalizers for the two
re-expressed channels (log spread, log total depth) are fit on the
windows handed to :func:`factorize_windows`, which the caller
restricts to TRAIN windows, matching the train-only standardization
discipline of the rest of the pipeline; every other channel keeps
the full feature's train statistics.

Two decodes exist. The continuous decode rounds the generated
count to a nonnegative integer and leaves signed flow untouched.
The hurdle decode (the discrete count head) takes a gate
probability per second: where the gate says "no trades" the count
is exactly 0 and the signed flow is exactly 0 with it; where it
says "trades" the count is a positive integer. The gate is a
per-second binary head trained against the observed zero-count
indicator, the same hurdle design as the structural variant.

Nothing is bounded silently: :func:`audit_factorized_windows`
reports the raw generated factorized windows BEFORE any decode
bound is applied (out-of-range imbalance rate, channels beyond
the decode safety bounds, raw negative counts, raw zero-count
seconds with nonzero flow). The decode itself applies only the
documented safety bounds: the log-spread and log-depth channels
are clipped to [-30, 30] before exponentiation (e^30 is beyond
any plausible book), the imbalance used for size derivation is
clipped to [-1, 1] (outside that range one derived size would be
negative), and derived sizes are floored at 1e-9 before the log,
the same floor the data pipeline uses. The scope limit of the
structural variant applies here unchanged: these are per-second
top-of-book aggregates; they do not encode the order-message
stream, queue positions, or individual orders.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import FEATURE_NAMES, N_FEATURES, WINDOW
from .models import REGIME_DIM, BookDDPM, ResBlock, _conditioning

FACTORIZED_FEATURE_NAMES = [
    "log_ret",
    "log_spread_bps",
    "log_total_depth",
    "imbalance",
    "signed_flow",
    "log_trade_count",
]
N_FACTORIZED = len(FACTORIZED_FEATURE_NAMES)
_FJ = {name: i for i, name in enumerate(FACTORIZED_FEATURE_NAMES)}

# Decode safety bounds for the two exponentiated channels. Values
# outside them are counted by audit_factorized_windows, never
# hidden: the audit runs on raw windows before decode.
_LOG_BOUND = 30.0
_SIZE_FLOOR = 1e-9


def _imbalance_from_sizes(bid: np.ndarray,
                          ask: np.ndarray) -> np.ndarray:
    """(bid - ask) / (bid + ask), overflow-safe (sizes are plain
    exponentials here, so no max-shift is needed; the denominator
    floor keeps a both-zero cell at 0 instead of nan)."""
    denom = bid + ask
    return (bid - ask) / np.where(denom > 0, denom, 1.0)


def factorize_windows(windows: np.ndarray, feature_mean: np.ndarray,
                      feature_std: np.ndarray):
    """Full standardized windows -> factorized standardized windows.

    Returns (factorized_windows, factorized_mean, factorized_std).
    Channels carried over unchanged keep the full feature's mean
    and std. The log-spread channel is re-expressed as
    log(spread_bps) and the log-total-depth channel is new; both
    take their mean/std from the supplied windows (which the
    caller restricts to the training split), so the factorized
    space is self-contained and the mapping is exactly invertible
    by :func:`decode_factorized_windows`.
    """
    w = np.asarray(windows, dtype=np.float64)
    fm = np.asarray(feature_mean, dtype=np.float64)
    fs = np.asarray(feature_std, dtype=np.float64)
    phys = w * fs + fm
    j_bid = FEATURE_NAMES.index("log_bid_sz")
    j_ask = FEATURE_NAMES.index("log_ask_sz")
    j_spr = FEATURE_NAMES.index("log_spread_bps")
    depth = np.exp(phys[:, :, j_bid]) + np.exp(phys[:, :, j_ask])
    log_depth = np.log(np.clip(depth, _SIZE_FLOOR, None))
    log_spread = np.log(np.clip(np.expm1(phys[:, :, j_spr]),
                                _SIZE_FLOOR, None))
    f_mean = np.array([fm[FEATURE_NAMES.index(n)]
                       if n in FEATURE_NAMES else 0.0
                       for n in FACTORIZED_FEATURE_NAMES])
    f_std = np.array([fs[FEATURE_NAMES.index(n)]
                      if n in FEATURE_NAMES else 1.0
                      for n in FACTORIZED_FEATURE_NAMES])
    j = _FJ["log_total_depth"]
    f_mean[j] = float(log_depth.mean())
    f_std[j] = float(log_depth.std()) or 1.0
    j = _FJ["log_spread_bps"]
    f_mean[j] = float(log_spread.mean())
    f_std[j] = float(log_spread.std()) or 1.0
    cols = []
    for name in FACTORIZED_FEATURE_NAMES:
        if name == "log_total_depth":
            cols.append((log_depth - f_mean[_FJ[name]])
                        / f_std[_FJ[name]])
        elif name == "log_spread_bps":
            cols.append((log_spread - f_mean[_FJ[name]])
                        / f_std[_FJ[name]])
        else:
            cols.append(w[:, :, FEATURE_NAMES.index(name)])
    fact = np.stack(cols, axis=2)
    return fact.astype(np.float32), f_mean, f_std


def decode_factorized_windows(fact: np.ndarray,
                              factorized_mean: np.ndarray,
                              factorized_std: np.ndarray,
                              feature_mean: np.ndarray,
                              feature_std: np.ndarray,
                              gate_prob: np.ndarray = None
                              ) -> np.ndarray:
    """Factorized windows -> full standardized windows.

    Deterministic. Per (window, second): the spread is exp of the
    log-spread channel (strictly positive) re-expressed as the
    standard log1p feature; the size pair is derived from total
    depth and imbalance by the exact identities in the module
    docstring; the output imbalance channel is then recomputed
    from the derived sizes, so the decoded window's channels agree
    by construction. With ``gate_prob`` the count goes through the
    hurdle decode (gate >= 0.5: count exactly 0 and flow exactly
    0; otherwise a positive integer count). Without it the count
    is the rounded generated count, floored at 0, and signed flow
    passes through unchanged.
    """
    f = np.asarray(fact, dtype=np.float64)
    fmean = np.asarray(factorized_mean, dtype=np.float64)
    fstd = np.asarray(factorized_std, dtype=np.float64)
    fm = np.asarray(feature_mean, dtype=np.float64)
    fs = np.asarray(feature_std, dtype=np.float64)
    phys = f * fstd + fmean
    fp = np.zeros(f.shape[:2] + (N_FEATURES,), dtype=np.float64)
    fp[:, :, FEATURE_NAMES.index("log_ret")] = \
        phys[:, :, _FJ["log_ret"]]
    log_spread = np.clip(phys[:, :, _FJ["log_spread_bps"]],
                         -_LOG_BOUND, _LOG_BOUND)
    fp[:, :, FEATURE_NAMES.index("log_spread_bps")] = \
        np.log1p(np.exp(log_spread))
    log_depth = np.clip(phys[:, :, _FJ["log_total_depth"]],
                        -_LOG_BOUND, _LOG_BOUND)
    depth = np.exp(log_depth)
    imb = np.clip(phys[:, :, _FJ["imbalance"]], -1.0, 1.0)
    bid = np.clip(depth * (1.0 + imb) / 2.0, _SIZE_FLOOR, None)
    ask = np.clip(depth * (1.0 - imb) / 2.0, _SIZE_FLOOR, None)
    log_bid = np.log(bid)
    log_ask = np.log(ask)
    fp[:, :, FEATURE_NAMES.index("log_bid_sz")] = log_bid
    fp[:, :, FEATURE_NAMES.index("log_ask_sz")] = log_ask
    fp[:, :, FEATURE_NAMES.index("imbalance")] = \
        _imbalance_from_sizes(bid, ask)
    with np.errstate(over="ignore", invalid="ignore"):
        count_raw = np.expm1(phys[:, :, _FJ["log_trade_count"]])
    count_raw = np.clip(np.nan_to_num(count_raw, nan=0.0), 0.0, 1e12)
    flow = phys[:, :, _FJ["signed_flow"]]
    if gate_prob is None:
        count = np.maximum(0.0, np.round(count_raw))
    else:
        gate = np.asarray(gate_prob, dtype=np.float64)
        count = np.where(gate >= 0.5, 0.0,
                         np.maximum(1.0, np.round(count_raw)))
        flow = np.where(count > 0, flow, 0.0)
    fp[:, :, FEATURE_NAMES.index("log_trade_count")] = np.log1p(count)
    fp[:, :, FEATURE_NAMES.index("signed_flow")] = flow
    return ((fp - fm) / fs).astype(np.float32)


def audit_factorized_windows(fact: np.ndarray,
                             factorized_mean: np.ndarray,
                             factorized_std: np.ndarray,
                             tol: float = 0.01) -> dict:
    """Raw audit of generated factorized windows, BEFORE decode.

    Reports the violations the decode bounds exist for, on the
    raw model output exactly as generated: imbalance outside
    [-1, 1], the exponentiated channels outside their decode
    safety bounds, non-finite cells, negative raw counts, and
    raw zero-count seconds carrying nonzero signed flow. The
    audit only reports; it never mutates its input.
    """
    f = np.asarray(fact, dtype=np.float64)
    fmean = np.asarray(factorized_mean, dtype=np.float64)
    fstd = np.asarray(factorized_std, dtype=np.float64)
    phys = f * fstd + fmean
    imb = phys[:, :, _FJ["imbalance"]]
    log_depth = phys[:, :, _FJ["log_total_depth"]]
    log_spread = phys[:, :, _FJ["log_spread_bps"]]
    with np.errstate(over="ignore", invalid="ignore"):
        count_raw = np.expm1(phys[:, :, _FJ["log_trade_count"]])
    flow = phys[:, :, _FJ["signed_flow"]]
    finite_imb = imb[np.isfinite(imb)]
    return {
        "n_cells": int(imb.size),
        "imbalance_out_of_range_rate":
            float((np.abs(imb) > 1.0).mean()),
        "raw_imbalance_min":
            float(finite_imb.min()) if finite_imb.size
            else float("nan"),
        "raw_imbalance_max":
            float(finite_imb.max()) if finite_imb.size else float("nan"),
        "log_depth_out_of_bound_rate":
            float((np.abs(log_depth) > _LOG_BOUND).mean()),
        "log_spread_out_of_bound_rate":
            float((np.abs(log_spread) > _LOG_BOUND).mean()),
        "nonfinite_rate":
            float((~np.isfinite(phys)).any(axis=2).mean()),
        "count_negative_rate": float((count_raw < -tol).mean()),
        "raw_coherence_violation_rate":
            float(((np.round(count_raw) == 0)
                   & (np.abs(flow) > tol)).mean()),
        "tol": tol,
    }


# ------------------------------------------------------------- model

class FactorizedDenoiser(torch.nn.Module):
    """Denoiser over the factorized feature space, plus a hurdle gate.

    Same conditioning trunk and residual stack as the shipped
    Denoiser, with one extra 1x1 convolution head: the gate logit
    per (window, second), trained to predict whether the trade
    count is zero, which the hurdle decode consumes. There is no
    log bid size or log ask size channel anywhere in this network;
    the size pair exists only through total depth and imbalance.
    """

    def __init__(self, n_features: int = N_FACTORIZED,
                 window: int = WINDOW, regime_dim: int = REGIME_DIM,
                 ch: int = 96, tdim: int = 64, n_blocks: int = 3):
        super().__init__()
        self.time_embed, self.regime_mlp, cond_dim = _conditioning(
            regime_dim, tdim)
        self.in_conv = torch.nn.Conv1d(n_features, ch, 3, padding=1)
        self.blocks = torch.nn.ModuleList(
            [ResBlock(ch, cond_dim) for _ in range(n_blocks)])
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(8, ch), torch.nn.SiLU(),
            torch.nn.Conv1d(ch, n_features, 3, padding=1))
        self.gate = torch.nn.Conv1d(ch, 1, 1)
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


class FactorizedBookDDPM(BookDDPM):
    """DDPM over the factorized representation (P02).

    Training is the usual diffusion loss on the six factorized
    channels plus, when ``gate_weight`` is positive, a binary
    cross-entropy on the hurdle gate against the observed
    zero-count indicator. Sampling runs the ordinary reverse
    chain in factorized space and decodes deterministically with
    :func:`decode_factorized_windows`: the size pair is derived
    from depth and imbalance, the spread is positive by
    construction, and with the hurdle decode the count is a
    nonnegative integer with flow exactly zero wherever the
    count is zero. With ``gate_weight=0`` the gate head is
    untrained and sampling uses the continuous decode; that is
    the plain factorized arm of the P02 ablation, and the gated
    configuration is the discrete-head arm.
    """

    def __init__(self, factorized_mean, factorized_std, feature_mean,
                 feature_std, gate_weight: float = 1.0,
                 ch: int = 96, n_blocks: int = 3, t_steps: int = 100,
                 seed: int | None = None, run_id: str | None = None):
        super().__init__(n_features=N_FACTORIZED, window=WINDOW,
                         t_steps=t_steps, ch=ch, n_blocks=n_blocks,
                         seed=seed, run_id=run_id, arch="cnn")
        self.net = FactorizedDenoiser(n_features=N_FACTORIZED,
                                      window=WINDOW, ch=ch,
                                      n_blocks=n_blocks)
        self.gate_weight = float(gate_weight)
        self._fact_mean_np = np.asarray(factorized_mean,
                                        dtype=np.float64)
        self._fact_std_np = np.asarray(factorized_std,
                                       dtype=np.float64)
        self._feature_mean_np = np.asarray(feature_mean,
                                           dtype=np.float64)
        self._feature_std_np = np.asarray(feature_std,
                                          dtype=np.float64)
        self.register_buffer(
            "_fact_mean",
            torch.tensor(self._fact_mean_np, dtype=torch.float32))
        self.register_buffer(
            "_fact_std",
            torch.tensor(self._fact_std_np, dtype=torch.float32))

    def p_loss(self, x0, regime, aux_weights=None):
        if aux_weights is not None:
            raise ValueError(
                "the statistical-structure auxiliary losses are "
                "defined on the full seven-feature windows and are "
                "not applied to the factorized variant")
        b = x0.shape[0]
        t = torch.randint(0, self.t_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred, gate = self.net.forward_with_gate(xt, t, regime)
        mse = F.mse_loss(pred, noise)
        if self.gate_weight == 0.0:
            return mse
        j_cnt = _FJ["log_trade_count"]
        phys_cnt = x0[:, j_cnt, :] * self._fact_std[j_cnt] \
            + self._fact_mean[j_cnt]
        zero = (torch.round(torch.expm1(phys_cnt)) == 0).float()
        gate_loss = F.binary_cross_entropy_with_logits(
            gate.squeeze(1), zero)
        return mse + self.gate_weight * gate_loss

    @torch.no_grad()
    def decode(self, fact: torch.Tensor, regime: torch.Tensor,
               hurdle: bool = False) -> torch.Tensor:
        """Decode factorized samples to full standardized windows.

        Deterministic: with ``hurdle`` the gate is evaluated once,
        at t=0, on the finished factorized sample, and the
        reconstruction itself is a fixed function.
        """
        gate = None
        if hurdle:
            x = fact.detach().cpu().permute(0, 2, 1)  # (B, F, L)
            t = torch.zeros(x.shape[0], dtype=torch.long)
            with torch.inference_mode():
                gate = torch.sigmoid(
                    self.net.gate_logit(
                        x, t, regime.detach().cpu())).squeeze(1).numpy()
        full = decode_factorized_windows(
            fact.detach().cpu().numpy(), self._fact_mean_np,
            self._fact_std_np, self._feature_mean_np,
            self._feature_std_np, gate_prob=gate)
        return torch.from_numpy(full).to(device=fact.device)

    @torch.no_grad()
    def sample_factorized(self, regime, seed: int = 0,
                          generator: torch.Generator | None = None,
                          run_id: str | None = None,
                          batch_size: int | None = None,
                          hurdle: bool = False):
        """Sample the reverse chain in factorized space and decode."""
        fact = self.sample(regime, seed=seed, generator=generator,
                           run_id=run_id, batch_size=batch_size)
        return self.decode(fact, regime, hurdle=hurdle)
