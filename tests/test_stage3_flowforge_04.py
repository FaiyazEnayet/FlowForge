"""Stage 3 flowforge C04: physical audit and constrained generation.

C04 scope (issue: Physical audit and constrained generative
representation):

- The book-consistency audit measures the RAW imbalance residual
  before any clipping and records raw range, sign/finite/overflow
  checks, spread nonnegativity, trade-count nonnegativity and integer
  validity, and count/flow coherence. It reports violations; it never
  repairs them.
- ``BookDDPM.sample`` gains explicit raw and constrained modes, and
  the full report covers BOTH modes with the postprocessing quality
  loss stated explicitly.
- An experimental structural DDPM drops the redundant imbalance
  output channel, reconstructs imbalance deterministically at decode
  time, represents spread/sizes with invertible positive transforms,
  and decodes the trade count through a discrete hurdle head, so its
  windows satisfy the audited constraints by construction.
"""

import json

import numpy as np
import pytest
import torch

from flowforge.data import FEATURE_NAMES
from flowforge.evaluate import (book_consistency_table, comparison_report,
                                constrain_imbalance, derive_imbalance,
                                full_report)
from flowforge.models import (BookDDPM, train_ddpm)

J_BID = FEATURE_NAMES.index("log_bid_sz")
J_ASK = FEATURE_NAMES.index("log_ask_sz")
J_IMB = FEATURE_NAMES.index("imbalance")
J_SPR = FEATURE_NAMES.index("log_spread_bps")
J_SF = FEATURE_NAMES.index("signed_flow")
J_CNT = FEATURE_NAMES.index("log_trade_count")
J_RET = FEATURE_NAMES.index("log_ret")

F_MEAN = np.array([0.0, 1.5, 4.0, 4.0, 0.0, 0.0, 1.2])
F_STD = np.array([0.02, 0.6, 1.1, 1.1, 0.35, 1.5, 0.9])


def _standardize(phys):
    return (phys - F_MEAN) / F_STD


def _valid_phys(n, seed=0):
    """Physically valid feature-space windows (auditable, no violations)."""
    rng = np.random.default_rng(seed)
    phys = np.zeros((n, 60, 7))
    phys[:, :, J_RET] = rng.normal(0, 0.01, size=(n, 60))
    phys[:, :, J_SPR] = np.log1p(rng.uniform(0.5, 3.0, size=(n, 60)))
    phys[:, :, J_BID] = rng.normal(4.0, 0.3, size=(n, 60))
    phys[:, :, J_ASK] = rng.normal(4.0, 0.3, size=(n, 60))
    phys[:, :, J_IMB] = derive_imbalance(phys[:, :, J_BID],
                                         phys[:, :, J_ASK])
    counts = rng.integers(0, 6, size=(n, 60)).astype(np.float64)
    phys[:, :, J_CNT] = np.log1p(counts)
    flow = rng.normal(0, 0.5, size=(n, 60))
    phys[:, :, J_SF] = np.where(counts > 0, flow, 0.0)
    return phys


# ------------------------------------------------------------- the audit

def test_stage3_flowforge_04_audit_raw_imbalance_before_clipping():
    # Imbalance a generator emitted at 2.5, far outside [-1, 1], while
    # its sizes imply 0. An audit that clips first would report a
    # residual of 1.0; the raw audit must report 2.5.
    phys = _valid_phys(4, seed=1)
    phys[:, :, J_BID] = 4.0
    phys[:, :, J_ASK] = 4.0
    phys[:, :, J_IMB] = 2.5
    gen = _standardize(phys)
    rep = book_consistency_table(gen, F_MEAN, F_STD)
    assert rep["raw_imbalance_max"] == pytest.approx(2.5)
    assert rep["raw_imbalance_min"] == pytest.approx(2.5)
    assert rep["imbalance_out_of_range_rate"] == 1.0
    assert rep["imbalance_violation_rate"] == 1.0
    assert rep["imbalance_mae"] == pytest.approx(2.5)


def test_stage3_flowforge_04_audit_adversarial_violations_before_repair():
    phys = _valid_phys(8, seed=2)
    phys[0, 0, J_IMB] = np.nan            # non-finite generated value
    phys[1, 0, J_RET] = np.inf            # non-finite generated value
    phys[2, 0, J_SPR] = -0.25             # spread_bps = expm1(-0.25) < 0
    phys[3, 0, J_CNT] = np.log1p(-0.4)    # negative trade count
    phys[4, 0, J_CNT] = np.log1p(2.5)     # non-integer trade count
    phys[5, 0, J_CNT] = 0.0               # zero trades ...
    phys[5, 0, J_SF] = 3.0                # ... but nonzero signed flow
    phys[6, 0, J_BID] = 800.0             # exp(800) overflows float64
    gen = _standardize(phys)
    before = gen.copy()
    rep = book_consistency_table(gen, F_MEAN, F_STD)
    # The audit reports; it does not repair. Input is bit-identical.
    assert np.array_equal(gen, before, equal_nan=True)
    assert rep["nonfinite_rate"] > 0
    assert rep["overflow_rate"] > 0
    assert rep["negative_spread_rate"] > 0
    assert rep["count_negative_rate"] > 0
    assert rep["count_noninteger_rate"] > 0
    assert rep["coherence_violation_rate"] > 0
    # Repair touches imbalance only: after constraining, the imbalance
    # violations are gone while the count violations remain reported.
    fixed = constrain_imbalance(gen, F_MEAN, F_STD)
    rep_fixed = book_consistency_table(fixed, F_MEAN, F_STD)
    assert rep_fixed["imbalance_violation_rate"] == 0.0
    assert rep_fixed["count_noninteger_rate"] > 0


def test_stage3_flowforge_04_derive_imbalance_extreme_inputs():
    out = derive_imbalance(np.array([700.0, -700.0, 1e6, 0.0]),
                           np.array([-700.0, 700.0, 1e6, 0.0]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(-1.0)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(0.0)


def test_stage3_flowforge_04_constrained_mode_algebraic_invariants():
    rng = np.random.default_rng(4)
    fixtures = []
    for scale in (0.1, 1.0, 10.0, 100.0):
        fixtures.append(rng.normal(size=(512, 60, 7)) * scale)
    # Extreme numerical inputs: log sizes whose exponentials overflow
    # or underflow float64, in standardized units.
    phys = _valid_phys(256, seed=5)
    phys[:, :, J_BID] = 700.0
    phys[:, :, J_ASK] = -700.0
    phys[::2, :, J_BID] = 1e6
    fixtures.append(_standardize(phys))
    n_windows = sum(f.shape[0] for f in fixtures)
    assert n_windows >= 2000  # thousands of synthetic fixtures
    for gen in fixtures:
        fixed = constrain_imbalance(gen, F_MEAN, F_STD)
        assert np.isfinite(fixed).all()
        rep = book_consistency_table(fixed, F_MEAN, F_STD)
        assert rep["imbalance_violation_rate"] == 0.0
        assert rep["imbalance_mae"] <= 1e-9
        assert rep["raw_imbalance_max"] <= 1.0 + 1e-12
        assert rep["raw_imbalance_min"] >= -1.0 - 1e-12


# --------------------------------------------------------- sampler modes

def _tiny_ddpm(seed=3):
    return BookDDPM(n_features=7, window=60, t_steps=10, ch=16,
                    n_blocks=1, seed=seed)


def _regime(n, seed=6):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        rng.normal(size=(n, 2)).astype(np.float32))


def test_stage3_flowforge_04_sample_raw_and_constrained_modes():
    model = _tiny_ddpm()
    regime = _regime(6)
    raw_default = model.sample(regime, seed=11)
    raw = model.sample(regime, seed=11, mode="raw")
    assert torch.equal(raw_default, raw)  # raw is the legacy behaviour
    con = model.sample(regime, seed=11, mode="constrained",
                       feature_mean=F_MEAN, feature_std=F_STD)
    expected = constrain_imbalance(raw.numpy(), F_MEAN, F_STD)
    assert np.allclose(con.numpy(), expected, atol=1e-5)
    rep = book_consistency_table(con.numpy(), F_MEAN, F_STD)
    assert rep["imbalance_violation_rate"] == 0.0
    with pytest.raises(ValueError):
        model.sample(regime, mode="sideways")
    with pytest.raises(ValueError):
        model.sample(regime, mode="constrained")  # stats are required


# ---------------------------------------------------- structural model

def test_stage3_flowforge_04_structural_model_by_construction():
    from flowforge.models import (STRUCTURAL_FEATURE_NAMES,
                                  StructuralBookDDPM,
                                  build_structural_dataset)
    assert STRUCTURAL_FEATURE_NAMES == [
        n for n in FEATURE_NAMES if n != "imbalance"]
    phys = _valid_phys(48, seed=7)
    x = _standardize(phys).astype(np.float32)
    struct, s_mean, s_std = build_structural_dataset(x, F_MEAN, F_STD)
    assert struct.shape == (48, 60, 6)
    model = StructuralBookDDPM(struct_mean=s_mean, struct_std=s_std,
                               feature_mean=F_MEAN, feature_std=F_STD,
                               ch=16, n_blocks=1, t_steps=10, seed=5)
    assert model.n_features == 6
    assert model.net.n_features == 6  # no imbalance output channel
    regime = _regime(48, seed=8)
    hist, _ = train_ddpm(model, torch.from_numpy(struct), regime,
                         epochs=2, seed=5, verbose=False)
    assert np.isfinite(hist).all()
    out = model.sample_structural(regime[:8], seed=7)
    assert out.shape == (8, 60, 7)
    rep = book_consistency_table(out.numpy(), F_MEAN, F_STD)
    assert rep["imbalance_violation_rate"] == 0.0
    # Exactness is bounded by the float32 storage of standardized
    # windows (~1e-8); the float64 postprocessing path above is exact
    # to 1e-9, this decoded path is pinned at 1e-6 with margin.
    assert rep["imbalance_mae"] <= 1e-6
    assert rep["negative_spread_rate"] == 0.0
    assert rep["count_negative_rate"] == 0.0
    assert rep["count_noninteger_rate"] == 0.0
    assert rep["coherence_violation_rate"] == 0.0
    assert rep["nonfinite_rate"] == 0.0
    # Decode is deterministic: the same structural sample decodes to
    # the same full windows, and sample_structural is that decode.
    raw_struct = model.sample(regime[:8], seed=7)
    d1 = model.decode(raw_struct, regime[:8])
    d2 = model.decode(raw_struct, regime[:8])
    assert torch.equal(d1, d2)
    assert torch.equal(out, d1)
    # The reconstructed imbalance is exactly the sizes' implied value.
    phys_out = out.numpy() * F_STD + F_MEAN
    implied = derive_imbalance(phys_out[:, :, J_BID],
                               phys_out[:, :, J_ASK])
    assert np.max(np.abs(phys_out[:, :, J_IMB] - implied)) <= 1e-6


# ------------------------------------------------------------- reporting

def test_stage3_flowforge_04_report_covers_both_modes_quality_loss():
    rng = np.random.default_rng(9)
    real_val = (rng.normal(size=(24, 60, 7)) * 0.3).astype(np.float32)
    real_test = (rng.normal(size=(24, 60, 7)) * 0.3).astype(np.float32)
    gen_val = (real_val + rng.normal(0, 0.05, real_val.shape)
               ).astype(np.float32)
    gen_test = (real_test + rng.normal(0, 0.05, real_test.shape)
                ).astype(np.float32)
    rep = full_report(real_test, gen_test, real_val, gen_val, seed=0,
                      feature_mean=F_MEAN, feature_std=F_STD)
    assert "constrained" in rep
    assert "postprocessing_quality_loss" in rep
    ql = rep["postprocessing_quality_loss"]
    assert ql["mean_w1_delta"] == pytest.approx(
        rep["constrained"]["marginals"]["mean_w1"]
        - rep["marginals"]["mean_w1"], abs=1e-9)
    assert ql["discriminator_auc_test_delta"] == pytest.approx(
        rep["constrained"]["discriminator"]["auc_test"]
        - rep["discriminator"]["auc_test"], abs=1e-9)
    assert ql["tstr_acc_synth_delta"] == pytest.approx(
        rep["constrained"]["tstr"]["acc_synth"]
        - rep["tstr"]["acc_synth"], abs=1e-9)
    # The constrained bundle must stay JSON-safe for runs/metrics.json:
    # no raw score arrays ride along.
    assert "test_scores" not in rep["constrained"]["discriminator"]
    json.dumps(ql)
    json.dumps(rep["book_consistency"])


def _cost(nfe):
    return {"sample_seconds": 0.01, "sample_seconds_per_1k": 0.1,
            "peak_memory_mb": 1.0, "nfe": nfe, "nfe_per_window": nfe}


def test_stage3_flowforge_04_comparison_parity_includes_constrained():
    # Parity guard: a constrained method runs through the C03 parity
    # report next to the raw model and every baseline, under the one
    # shared seed set and the identical discriminator protocol.
    rng = np.random.default_rng(10)
    real_val = (rng.normal(size=(24, 60, 7)) * 0.3).astype(np.float32)
    real_test = (rng.normal(size=(24, 60, 7)) * 0.3).astype(np.float32)
    runs = []
    specs = [("ddpm", "conditional", 100), ("vae", "conditional", 1),
             ("chunk_copy", "conditional", 0),
             ("conditional_ar", "conditional", 0),
             ("var", "unconditional", 0)]
    for seed in (7, 107, 207):
        base_val = (real_val + rng.normal(0, 0.05, real_val.shape)
                    ).astype(np.float32)
        base_test = (real_test + rng.normal(0, 0.05, real_test.shape)
                     ).astype(np.float32)
        for name, kind, nfe in specs:
            runs.append({"method": name, "kind": kind, "seed": seed,
                         "gen_val": base_val, "gen_test": base_test,
                         "cost": _cost(nfe)})
        runs.append({
            "method": "ddpm_constrained", "kind": "conditional",
            "seed": seed,
            "gen_val": constrain_imbalance(base_val, F_MEAN, F_STD),
            "gen_test": constrain_imbalance(base_test, F_MEAN, F_STD),
            "cost": _cost(100)})
    rep = comparison_report(real_test, real_val, runs,
                            block_lengths=(1, 5), n_boot=20,
                            disc_epochs=5, tstr_epochs=40,
                            protocol_seed=0)
    assert set(rep["methods"]) == {"ddpm", "ddpm_constrained", "vae",
                                   "chunk_copy", "conditional_ar", "var"}
    assert rep["seeds"] == [7, 107, 207]
    pairs = {tuple(r["methods"]) for r in rep["paired_w1"]}
    assert ("ddpm", "ddpm_constrained") in pairs
    assert rep["discriminator_protocol"]["seed"] == 0
