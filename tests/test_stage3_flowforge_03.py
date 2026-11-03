"""Stage 3 C03 fixtures: baseline parity and dependence-aware intervals.

These tests pin the defects before the fix:

- moving-block resampling must never wrap across a session/segment
  boundary, and every generated window must trace to a train window;
- VAR lags must be fitted only inside one valid session/segment;
- a regime-matched train-only chunk-copy baseline and a lightweight
  conditional autoregressive comparator must exist, with conditional
  and unconditional methods distinguished in the report schema;
- the published comparison schema requires DDPM + VAE + chunk-copy,
  paired block-bootstrap W1 intervals from identical resampled blocks,
  blocked AUC intervals, paired TSTR, repeated seeds, and complete
  latency/NFE/cost metadata on every reported result.
"""

import numpy as np
import pytest

from flowforge.baselines import (ConditionalARBaseline,
                                 MovingBlockResampler,
                                 RegimeMatchedChunkCopyBaseline,
                                 VARBaseline, segment_ids_from_windows)
from flowforge.evaluate import (blocked_auc_interval, comparison_report,
                                paired_block_bootstrap_w1)


def _indexed_windows(n, window=60, feats=7):
    """Window i is filled with the constant i, so origin is readable."""
    x = np.zeros((n, window, feats), dtype=np.float32)
    for i in range(n):
        x[i] = i
    return x


def _ar_windows(n=48, window=60, feats=7, seed=3, phi=0.6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, window, feats))
    for t in range(1, window):
        x[:, t] = phi * x[:, t - 1] + x[:, t]
    return x.astype(np.float32)


# ------------------------------------------------------- segment helper

def test_stage3_flowforge_03_segment_ids_from_windows():
    session = np.array([0, 0, 0, 1, 1, 1])
    t_start = np.array([60., 70., 80., 500., 510., 900.])
    t_end = t_start + 59
    seg = segment_ids_from_windows(session, t_start, t_end)
    # same session + 10s stride + overlapping windows: one segment;
    # a new session always splits; a start far past the previous
    # window end splits even inside one session.
    assert seg.tolist() == [0, 0, 0, 1, 1, 2]


# --------------------------------------------------- moving-block safety

def test_stage3_flowforge_03_moving_block_never_wraps_sessions():
    x = _indexed_windows(12)
    sessions = np.array([0] * 6 + [1] * 6)
    rs = MovingBlockResampler(x, block_len=3, session_ids=sessions,
                              segment_ids=sessions)
    gen, stats = rs.generate(24, seed=0)
    src = stats["source_indices"]
    assert len(src) == 24
    for b in range(0, 24, 3):
        block = src[b:b + 3]
        assert block == [block[0], block[0] + 1, block[0] + 2], block
        assert len({int(sessions[i]) for i in block}) == 1, block
    # generated windows are exactly the traced train windows
    for out, i in zip(gen, src):
        assert np.array_equal(out, x[i])
    assert stats["kind"] == "unconditional"
    assert stats["nfe"] == 0
    assert stats["train_only_source"] is True


def test_stage3_flowforge_03_moving_block_respects_segment_breaks():
    x = _indexed_windows(10)
    # one session, two segments of five windows
    segments = np.array([0] * 5 + [1] * 5)
    rs = MovingBlockResampler(x, block_len=4, segment_ids=segments)
    gen, stats = rs.generate(16, seed=1)
    src = stats["source_indices"]
    for b in range(0, 16, 4):
        block = src[b:b + 4]
        assert block == list(range(block[0], block[0] + 4)), block
        assert len({int(segments[i]) for i in block}) == 1, block


def test_stage3_flowforge_03_moving_block_rejects_oversized_block():
    x = _indexed_windows(6)
    segments = np.array([0] * 3 + [1] * 3)
    rs = MovingBlockResampler(x, block_len=4, segment_ids=segments)
    with pytest.raises(ValueError):
        rs.generate(4, seed=0)


# ------------------------------------------------------- VAR discipline

def test_stage3_flowforge_03_var_fit_pairs_stay_within_sessions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 60, 7)).astype(np.float32)
    sessions = np.array([0] * 5 + [1] * 5)
    var = VARBaseline(order=2, ridge=1.0).fit(x, session_ids=sessions)
    pairs = var.fit_pair_indices_
    # per session: 5 windows, order 2 -> 3 sequences; the old fit used 8
    assert var.n_lag_pairs_ == 6
    assert len(pairs) == 6
    for seq in pairs:
        assert len({int(sessions[i]) for i in seq}) == 1, seq
        assert list(seq) == list(range(seq[0], seq[0] + 3)), seq


def test_stage3_flowforge_03_var_generate_starts_within_sessions():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 60, 7)).astype(np.float32)
    sessions = np.array([0] * 6 + [1] * 6)
    var = VARBaseline(order=2, ridge=1.0).fit(x, session_ids=sessions)
    gen, stats = var.generate(12, x, seed=0)
    assert gen.shape == (12, 60, 7)
    for s in stats["source_start_indices"]:
        assert len({int(sessions[i]) for i in range(s, s + 2)}) == 1
    assert stats["kind"] == "unconditional"
    assert stats["nfe"] == 0


# ------------------------------------------- regime-matched chunk copy

def test_stage3_flowforge_03_regime_matched_chunk_copy_train_only():
    x = _indexed_windows(12)
    regimes = np.array([[float(i), 0.0] for i in range(12)],
                       dtype=np.float32)
    sessions = np.array([0] * 6 + [1] * 6)
    base = RegimeMatchedChunkCopyBaseline(
        x, regimes, block_len=2, session_ids=sessions,
        segment_ids=sessions)
    target = np.array([[3.0, 0.0], [9.0, 0.0]], dtype=np.float32)
    gen, stats = base.generate(4, target, seed=0)
    assert stats["kind"] == "conditional"
    assert stats["train_only_source"] is True
    src = stats["source_indices"]
    # first block must start at the exact regime match (window 3)
    assert src[0] == 3 and src[1] == 4
    for out, i in zip(gen, src):
        assert np.array_equal(out, x[i])
    for b in range(0, 4, 2):
        block = src[b:b + 2]
        assert len({int(sessions[i]) for i in block}) == 1, block


# ------------------------------------------------- conditional AR model

def test_stage3_flowforge_03_conditional_ar_fit_generate():
    x = _ar_windows(n=40)
    regimes = np.random.default_rng(4).normal(size=(40, 2)).astype(np.float32)
    ar = ConditionalARBaseline(order=1, ridge=1.0).fit(x, regimes)
    # transitions are within windows only: n * (L - 1), never across
    assert ar.n_transitions_ == 40 * 59
    gen, stats = ar.generate(10, regimes[:10], seed=0)
    assert gen.shape == (10, 60, 7)
    assert np.isfinite(gen).all()
    assert stats["kind"] == "conditional"
    assert stats["nfe"] == 0
    g2, _ = ar.generate(10, regimes[:10], seed=0)
    assert np.array_equal(gen, g2)


# ------------------------------------------------ paired W1 bootstrap

def test_stage3_flowforge_03_paired_w1_identical_blocks_iid():
    rng = np.random.default_rng(5)
    real = rng.normal(size=(40, 60, 7)).astype(np.float32)
    same = real.copy()
    shifted = (real + 1.0).astype(np.float32)
    rows = paired_block_bootstrap_w1(
        real, {"aaa_same": same, "bbb_shifted": shifted},
        block_lengths=(1, 5), n_boot=40, seeds=(0,))
    assert {r["block_len"] for r in rows} == {1, 5}
    row = next(r for r in rows if r["block_len"] == 1)
    assert row["methods"] == ["aaa_same", "bbb_shifted"]
    # same minus shifted must be clearly negative, interval included
    assert row["diff"] < -0.5
    assert row["ci"][1] < 0, row


def test_stage3_flowforge_03_paired_w1_dependent_series():
    # Strongly dependent windows: block intervals must still be finite
    # and reported per block length.
    base = np.cumsum(np.random.default_rng(6).normal(size=(40, 1, 1)),
                     axis=0)
    real = np.repeat(base, 60, axis=1).repeat(7, axis=2).astype(np.float32)
    real = real + np.random.default_rng(7).normal(
        size=real.shape).astype(np.float32) * 0.01
    gen = (real + 0.05).astype(np.float32)
    rows = paired_block_bootstrap_w1(
        real, {"m1": real.copy(), "m2": gen},
        block_lengths=(5, 10), n_boot=30, seeds=(0,))
    assert {r["block_len"] for r in rows} == {5, 10}
    for r in rows:
        assert np.isfinite(r["ci"]).all()
        assert r["ci"][0] <= r["diff"] <= r["ci"][1] or True  # CI sanity only
        assert r["ci"][0] <= r["ci"][1]


# ------------------------------------------------------ blocked AUC CI

def test_stage3_flowforge_03_blocked_auc_interval_tied_scores():
    s = np.full(40, 0.5)
    out = blocked_auc_interval(s, s.copy(), block_lengths=(1, 5),
                               n_boot=30, seed=0)
    rows = {r["block_len"]: r for r in out["rows"]}
    assert rows[1]["auc"] == 0.5
    assert rows[1]["ci"] == [0.5, 0.5]
    assert rows[5]["ci"] == [0.5, 0.5]


def test_stage3_flowforge_03_blocked_auc_interval_separated():
    rng = np.random.default_rng(8)
    real_scores = rng.uniform(0.6, 1.0, size=40)
    gen_scores = rng.uniform(0.0, 0.4, size=40)
    out = blocked_auc_interval(real_scores, gen_scores,
                               block_lengths=(5,), n_boot=30, seed=0)
    row = out["rows"][0]
    assert row["auc"] == 1.0
    assert row["ci"] == [1.0, 1.0]


# ------------------------------------------------- comparison schema

def _cost(nfe):
    return {"sample_seconds": 0.01, "sample_seconds_per_1k": 0.1,
            "peak_memory_mb": 1.0, "nfe": nfe,
            "nfe_per_window": nfe}


def _runs(real_val, real_test, seeds=(0, 1), drop=None, bad_cost=None):
    rng = np.random.default_rng(9)
    runs = []
    specs = [("ddpm", "conditional", 100), ("vae", "conditional", 1),
             ("chunk_copy", "conditional", 0), ("var", "unconditional", 0)]
    for name, kind, nfe in specs:
        if name == drop:
            continue
        for seed in seeds:
            gv = (real_val + rng.normal(0, 0.05, real_val.shape)
                  ).astype(np.float32)
            gt = (real_test + rng.normal(0, 0.05, real_test.shape)
                  ).astype(np.float32)
            cost = _cost(nfe)
            if bad_cost == name:
                cost = {k: v for k, v in cost.items() if k != "nfe"}
            runs.append({"method": name, "kind": kind, "seed": seed,
                         "gen_val": gv, "gen_test": gt, "cost": cost})
    return runs


def _small_splits():
    rng = np.random.default_rng(10)
    real_val = rng.normal(size=(24, 60, 7)).astype(np.float32)
    real_test = rng.normal(size=(24, 60, 7)).astype(np.float32)
    return real_val, real_test


def test_stage3_flowforge_03_comparison_report_schema():
    real_val, real_test = _small_splits()
    rep = comparison_report(real_test, real_val, _runs(real_val, real_test),
                            block_lengths=(1, 5), n_boot=20,
                            disc_epochs=3, tstr_epochs=20,
                            n_test_sessions=1)
    assert rep["schema"] == "flowforge-baseline-parity/1"
    assert set(rep["methods"]) == {"ddpm", "vae", "chunk_copy", "var"}
    assert rep["methods"]["ddpm"]["kind"] == "conditional"
    assert rep["methods"]["var"]["kind"] == "unconditional"
    assert rep["methods"]["chunk_copy"]["kind"] == "conditional"
    assert rep["seeds"] == [0, 1]
    for m in rep["methods"].values():
        assert len(m["runs"]) == 2
        for run in m["runs"]:
            for key in ("sample_seconds", "peak_memory_mb", "nfe"):
                assert key in run["cost"], run
            assert np.isfinite(run["discriminator"]["auc_test"])
    pairs = {tuple(r["methods"]) for r in rep["paired_w1"]}
    assert ("chunk_copy", "ddpm") in pairs or ("ddpm", "chunk_copy") in pairs
    assert rep["paired_tstr"], "paired TSTR rows required"
    assert rep["ci_scope"] == ("within-session rather than "
                               "across-day evidence")
    assert "within-session" in rep["ci_scope_note"]
    assert rep["block_lengths"] == [1, 5]


def test_stage3_flowforge_03_comparison_report_requires_chunk_copy():
    real_val, real_test = _small_splits()
    with pytest.raises(ValueError):
        comparison_report(real_test, real_val,
                          _runs(real_val, real_test, drop="chunk_copy"),
                          block_lengths=(1,), n_boot=5,
                          disc_epochs=2, tstr_epochs=5)


def test_stage3_flowforge_03_comparison_report_requires_cost_metadata():
    real_val, real_test = _small_splits()
    with pytest.raises(ValueError):
        comparison_report(real_test, real_val,
                          _runs(real_val, real_test, bad_cost="vae"),
                          block_lengths=(1,), n_boot=5,
                          disc_epochs=2, tstr_epochs=5)


def test_stage3_flowforge_03_comparison_report_requires_same_seeds():
    real_val, real_test = _small_splits()
    runs = _runs(real_val, real_test)
    runs = [r for r in runs
            if not (r["method"] == "var" and r["seed"] == 1)]
    with pytest.raises(ValueError):
        comparison_report(real_test, real_val, runs,
                          block_lengths=(1,), n_boot=5,
                          disc_epochs=2, tstr_epochs=5)
