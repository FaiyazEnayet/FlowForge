"""Stage 3 P02 tests: factorized feasible top-of-book representation.

Fixture-first: these tests define the P02 contract before the
implementation in flowforge/factorized.py. All tests in this file are
prefixed test_stage3_flowforge_02_.

The factorized representation generates six channels (log return,
log spread, log total depth, imbalance, signed flow, log trade
count) and derives the bid/ask size pair exactly from total depth
and imbalance, instead of emitting the two sizes independently and
repairing the imbalance afterwards.
"""

import numpy as np
import pytest
import torch

from flowforge.data import FEATURE_NAMES, WINDOW
from flowforge.evaluate import book_consistency_table, derive_imbalance
from flowforge.factorized import (FACTORIZED_FEATURE_NAMES,
                                  FactorizedBookDDPM,
                                  audit_factorized_windows,
                                  decode_factorized_windows,
                                  factorize_windows)

J = {n: i for i, n in enumerate(FEATURE_NAMES)}
FJ = {n: i for i, n in enumerate(FACTORIZED_FEATURE_NAMES)}


def _physical_windows(n=24, seed=5):
    """Physically consistent windows in physical units.

    Imbalance is derived from the sizes and signed flow is zero
    wherever the trade count is zero, exactly as in recorded data.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros((n, WINDOW, len(FEATURE_NAMES)))
    w[:, :, J["log_ret"]] = rng.normal(0, 0.002, size=(n, WINDOW))
    w[:, :, J["log_spread_bps"]] = np.log1p(rng.uniform(0.5, 3.0,
                                                       size=(n, WINDOW)))
    w[:, :, J["log_bid_sz"]] = rng.normal(4.0, 0.3, size=(n, WINDOW))
    w[:, :, J["log_ask_sz"]] = rng.normal(4.0, 0.3, size=(n, WINDOW))
    bid = np.exp(w[:, :, J["log_bid_sz"]])
    ask = np.exp(w[:, :, J["log_ask_sz"]])
    w[:, :, J["imbalance"]] = (bid - ask) / (bid + ask)
    counts = rng.integers(0, 6, size=(n, WINDOW)).astype(np.float64)
    w[:, :, J["log_trade_count"]] = np.log1p(counts)
    flow = rng.normal(0, 1.0, size=(n, WINDOW))
    w[:, :, J["signed_flow"]] = np.where(counts > 0, flow, 0.0)
    return w


def _stats(phys):
    flat = phys.reshape(-1, len(FEATURE_NAMES))
    return flat.mean(axis=0), flat.std(axis=0)


def test_stage3_flowforge_02_factorized_names_and_shape():
    phys = _physical_windows()
    fm, fs = _stats(phys)
    std = ((phys - fm) / fs).astype(np.float32)
    fact, fmean, fstd = factorize_windows(std, fm, fs)
    assert len(FACTORIZED_FEATURE_NAMES) == 6
    assert "log_bid_sz" not in FACTORIZED_FEATURE_NAMES
    assert "log_ask_sz" not in FACTORIZED_FEATURE_NAMES
    assert "log_total_depth" in FACTORIZED_FEATURE_NAMES
    assert fact.shape == (len(phys), WINDOW, 6)
    assert fmean.shape == (6,) and fstd.shape == (6,)
    assert np.isfinite(fact).all()


def test_stage3_flowforge_02_roundtrip_exact():
    phys = _physical_windows()
    fm, fs = _stats(phys)
    std = ((phys - fm) / fs).astype(np.float32)
    fact, fmean, fstd = factorize_windows(std, fm, fs)
    back = decode_factorized_windows(fact, fmean, fstd, fm, fs)
    assert back.shape == std.shape
    np.testing.assert_allclose(back, std, atol=2e-5, rtol=1e-4)
    # The derived size pair reproduces the physical sizes, so the
    # decoded imbalance equals the original imbalance.
    back_phys = back.astype(np.float64) * fs + fm
    # Exact algebra in float64; the 1e-6 bound is the float32
    # storage quantization of the windows themselves.
    np.testing.assert_allclose(
        derive_imbalance(back_phys[:, :, J["log_bid_sz"]],
                         back_phys[:, :, J["log_ask_sz"]]),
        phys[:, :, J["imbalance"]], atol=1e-6)


def test_stage3_flowforge_02_derived_sizes_identity():
    # Handcrafted factorized physical cells across the imbalance
    # range: derived sizes are nonnegative, sum to the depth, and
    # imply exactly the imbalance they were derived from.
    depth = np.array([[100.0, 3.5, 57.25, 1000.0]])
    imb = np.array([[-0.999, 0.0, 0.25, 0.999]])
    fact_phys = np.zeros((1, 4, 6))
    fact_phys[:, :, FJ["log_total_depth"]] = np.log(depth)
    fact_phys[:, :, FJ["imbalance"]] = imb
    fact_phys[:, :, FJ["log_spread_bps"]] = np.log(2.0)
    fact = fact_phys.copy()
    fmean = np.zeros(6)
    fstd = np.ones(6)
    fm = np.zeros(len(FEATURE_NAMES))
    fs = np.ones(len(FEATURE_NAMES))
    out = decode_factorized_windows(fact, fmean, fstd, fm, fs)
    bid = np.exp(out[0, :, J["log_bid_sz"]])
    ask = np.exp(out[0, :, J["log_ask_sz"]])
    assert (bid >= 0).all() and (ask >= 0).all()
    # Identities are exact in the float64 decode; the tolerance is
    # the float32 storage quantization of the returned windows.
    np.testing.assert_allclose(bid + ask, depth[0], rtol=1e-6)
    np.testing.assert_allclose((bid - ask) / (bid + ask), imb[0],
                               atol=1e-6)


def test_stage3_flowforge_02_stats_from_supplied_windows_only():
    # The depth and spread normalizers are fit on the windows handed
    # to factorize_windows and nothing else: adding more windows
    # must move them (the caller passes TRAIN windows only).
    phys = _physical_windows(n=24)
    fm, fs = _stats(phys)
    std = ((phys - fm) / fs).astype(np.float32)
    _, fmean_a, _ = factorize_windows(std[:12], fm, fs)
    _, fmean_b, _ = factorize_windows(std, fm, fs)
    log_depth = np.log(np.exp(phys[:, :, J["log_bid_sz"]])
                       + np.exp(phys[:, :, J["log_ask_sz"]]))
    expect_a = log_depth[:12].mean()
    expect_b = log_depth.mean()
    assert fmean_a[FJ["log_total_depth"]] == pytest.approx(expect_a)
    assert fmean_b[FJ["log_total_depth"]] == pytest.approx(expect_b)
    assert expect_a != pytest.approx(expect_b)


def test_stage3_flowforge_02_hurdle_decode_constraints():
    phys = _physical_windows()
    fm, fs = _stats(phys)
    std = ((phys - fm) / fs).astype(np.float32)
    fact, fmean, fstd = factorize_windows(std, fm, fs)
    n = len(phys)
    gate_zero = np.ones((n, WINDOW))  # gate says "no trades" everywhere
    out = decode_factorized_windows(fact, fmean, fstd, fm, fs,
                                    gate_prob=gate_zero)
    out_phys = out.astype(np.float64) * fs + fm
    counts = np.expm1(out_phys[:, :, J["log_trade_count"]])
    # The decode sets the count to exactly 0 in physical units;
    # what survives the float32 standardized storage is dust of
    # order 1e-8, the same tolerance the repo audit documents.
    assert (np.abs(counts) < 1e-6).all()
    assert (np.abs(out_phys[:, :, J["signed_flow"]]) < 1e-6).all()
    gate_pos = np.zeros((n, WINDOW))  # gate says "trades" everywhere
    out2 = decode_factorized_windows(fact, fmean, fstd, fm, fs,
                                     gate_prob=gate_pos)
    out2_phys = out2.astype(np.float64) * fs + fm
    counts2 = np.expm1(out2_phys[:, :, J["log_trade_count"]])
    assert (counts2 >= 1).all()
    audit = book_consistency_table(out2, fm, fs)
    assert audit["imbalance_violation_rate"] == 0.0
    assert audit["coherence_violation_rate"] == 0.0
    assert audit["count_noninteger_rate"] == 0.0


def test_stage3_flowforge_02_raw_audit_counts_without_mutating():
    # Base of zeros (every channel in range), then plant exactly
    # 12 out-of-range imbalance cells and 6 log-depth cells beyond
    # the decode bound. Stats are 0/1, so standardized == physical.
    fact = np.zeros((2, WINDOW, 6))
    fact[0, :12, FJ["imbalance"]] = 1.5
    fact[1, :6, FJ["log_total_depth"]] = 40.0
    before = fact.copy()
    audit = audit_factorized_windows(fact, np.zeros(6), np.ones(6))
    total = 2 * WINDOW
    assert audit["imbalance_out_of_range_rate"] == \
        pytest.approx(12 / total)
    assert audit["log_depth_out_of_bound_rate"] == \
        pytest.approx(6 / total)
    assert audit["raw_imbalance_max"] >= 1.5
    np.testing.assert_array_equal(fact, before)


def test_stage3_flowforge_02_model_loss_and_sample():
    torch.manual_seed(0)
    phys = _physical_windows(n=16)
    fm, fs = _stats(phys)
    std = ((phys - fm) / fs).astype(np.float32)
    fact, fmean, fstd = factorize_windows(std, fm, fs)
    model = FactorizedBookDDPM(fmean, fstd, fm, fs, gate_weight=1.0,
                               ch=16, n_blocks=1, t_steps=10, seed=3)
    x = torch.from_numpy(fact[:8])
    regime = torch.zeros(8, 2)
    loss = model.p_loss(x.permute(0, 2, 1).contiguous(), regime)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    gen = model.sample_factorized(regime[:4], seed=1, hurdle=True)
    assert gen.shape == (4, WINDOW, len(FEATURE_NAMES))
    audit = book_consistency_table(gen.numpy(), fm, fs)
    assert audit["imbalance_violation_rate"] == 0.0
    assert audit["coherence_violation_rate"] == 0.0
    assert audit["count_noninteger_rate"] == 0.0
    assert audit["negative_spread_rate"] == 0.0


def test_stage3_flowforge_02_continuous_decode_keeps_flow():
    # Without the hurdle gate (the plain factorized arm), the decode
    # derives sizes and rounds the count but does not touch signed
    # flow: the coherence constraint belongs to the discrete head.
    phys = _physical_windows(n=8)
    fm, fs = _stats(phys)
    std = ((phys - fm) / fs).astype(np.float32)
    fact, fmean, fstd = factorize_windows(std, fm, fs)
    out = decode_factorized_windows(fact, fmean, fstd, fm, fs)
    out_phys = out.astype(np.float64) * fs + fm
    counts = np.expm1(out_phys[:, :, J["log_trade_count"]])
    assert (np.abs(counts - np.round(counts)) < 1e-6).all()
    np.testing.assert_allclose(out_phys[:, :, J["signed_flow"]],
                               phys[:, :, J["signed_flow"]], atol=1e-4)
