"""Stage 3 P09 part 2: free-running multiwindow rollout and seam audit.

Part 2 builds on the part-1 context contract (`flowforge/rollout.py::
advance_context`): a free-running rollout starts from one real anchor
history and then feeds only its own generated windows back as context.
No future real window is ever read during a rollout, every seam is an
explicit checkpoint, and teacher-forced one-step generation (real
context) is kept distinct from free-running generation (generated
context). All numbers here are synthetic fixtures, not real-data
results.
"""

import numpy as np
import pytest
import torch

from flowforge.data import FEATURE_NAMES, LOOKBACK, N_FEATURES, WINDOW

J_RET = FEATURE_NAMES.index("log_ret")
J_SPR = FEATURE_NAMES.index("log_spread_bps")
J_BID = FEATURE_NAMES.index("log_bid_sz")
J_ASK = FEATURE_NAMES.index("log_ask_sz")
J_SF = FEATURE_NAMES.index("signed_flow")
J_CNT = FEATURE_NAMES.index("log_trade_count")


class _EchoModel:
    """Stub history-conditioned model for causality tests.

    The generated window is a deterministic function of the context it
    was handed (the context tail plus a fixed bias), so any context the
    rollout feeds it is visible in the output. It records every call.
    It ignores the seed by design: seed sensitivity is covered by the
    real-model test below, not by this echo.
    """

    use_history = True
    window = WINDOW

    def __init__(self, bias=0.01):
        self.bias = float(bias)
        self.calls = []

    def sample(self, regime, past_context=None, past_mask=None, seed=0,
               generator=None, batch_size=None, mode="raw",
               feature_mean=None, feature_std=None):
        assert past_context is not None
        ctx = past_context.detach().cpu().numpy() \
            if isinstance(past_context, torch.Tensor) else past_context
        self.calls.append({
            "regime": regime.detach().cpu().numpy()
            if isinstance(regime, torch.Tensor) else np.asarray(regime),
            "context": np.array(ctx, dtype=np.float64),
            "mask": None if past_mask is None else np.array(
                past_mask.detach().cpu().numpy()
                if isinstance(past_mask, torch.Tensor) else past_mask),
        })
        tail = ctx[:, -WINDOW:, :]
        return torch.from_numpy(
            (tail + self.bias).astype(np.float32))


class _PlainModel:
    """Stub model built without history conditioning."""

    use_history = False
    window = WINDOW

    def sample(self, regime, **kw):
        raise AssertionError("must not be called")


def _anchor(batch=2, lookback=LOOKBACK, seed=0):
    rng = np.random.default_rng(seed)
    ctx = rng.normal(size=(batch, lookback, N_FEATURES)).astype(np.float64)
    return ctx


def _stats():
    fm = np.zeros(N_FEATURES)
    fs = np.ones(N_FEATURES)
    rm = np.array([0.0, 0.0])
    rs = np.array([1.0, 1.0])
    return fm, fs, rm, rs


# ------------------------------------------------------------ regime from context


def test_stage3_flowforge_09_part_2_regime_from_context_matches_oracle():
    from flowforge.rollout import regime_from_context

    fm, fs, rm, rs = _stats()
    # log returns with a known std over the lookback, spread fixed.
    ctx = np.zeros((2, LOOKBACK, N_FEATURES))
    ret = np.linspace(-1.0, 1.0, LOOKBACK)
    ctx[:, :, J_RET] = ret[None, :]
    ctx[:, :, J_SPR] = np.log1p(2.0)  # spread 2 bps in physical units
    out = regime_from_context(ctx, fm, fs, rm, rs)
    assert out.shape == (2, 2)
    assert np.isfinite(out).all()
    # independent oracle: vol is the std of the physical log returns,
    # mean spread is the mean of the physical spreads in bps.
    phys_ret = ret * fs[J_RET] + fm[J_RET]
    phys_spr = np.expm1(np.log1p(2.0) * fs[J_SPR] + fm[J_SPR])
    expected = np.array([
        [(phys_ret.std() - rm[0]) / rs[0],
         (phys_spr - rm[1]) / rs[1]]] * 2)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="rank-3"):
        regime_from_context(ctx[0], fm, fs, rm, rs)
    bad = ctx.copy()
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        regime_from_context(bad, fm, fs, rm, rs)
    with pytest.raises(ValueError, match="feature"):
        regime_from_context(ctx[:, :, :3], fm, fs, rm, rs)


# ------------------------------------------------------------ free-running rollout


def test_stage3_flowforge_09_part_2_free_running_reads_no_future_real_data():
    """Every context the model ever sees derives from anchor + generated.

    The stub echoes its context, so the recorded call contexts are
    checked against an independent advance_context computation. The
    real continuation windows are never passed anywhere: changing
    them cannot change the rollout output.
    """
    from flowforge.rollout import advance_context, rollout_windows

    anchor = _anchor()
    fm, fs, rm, rs = _stats()
    model = _EchoModel(bias=0.01)
    out = rollout_windows(model, anchor, n_windows=3,
                          feature_mean=fm, feature_std=fs,
                          regime_mean=rm, regime_std=rs, seed=5)
    assert out["windows"].shape == (2, 3, WINDOW, N_FEATURES)
    assert np.isfinite(out["windows"]).all()
    assert out["regimes"].shape == (2, 3, 2)
    assert len(model.calls) == 3
    # call 0 sees the anchor; each later call sees the context advanced
    # from the previous generated window only.
    np.testing.assert_allclose(model.calls[0]["context"], anchor,
                                      rtol=1e-6, atol=1e-6)
    expected_ctx = anchor
    for k in range(1, 3):
        expected_ctx = advance_context(
            expected_ctx, out["windows"][:, k - 1], lookback=LOOKBACK)
        np.testing.assert_allclose(model.calls[k]["context"],
                                       expected_ctx, rtol=1e-6, atol=1e-6)
    # the generated windows are exactly the stub's echo of those
    # contexts, so no other data source contributed anything.
    for k in range(3):
        np.testing.assert_allclose(
            out["windows"][:, k],
            model.calls[k]["context"][:, -WINDOW:, :] + 0.01,
            rtol=0, atol=1e-6)
    # determinism: the same seed reproduces the rollout exactly.
    model2 = _EchoModel(bias=0.01)
    out2 = rollout_windows(model2, anchor, n_windows=3,
                           feature_mean=fm, feature_std=fs,
                           regime_mean=rm, regime_std=rs, seed=5)
    np.testing.assert_array_equal(out["windows"], out2["windows"])


def test_stage3_flowforge_09_part_2_rollout_seed_changes_real_model_output():
    """The per-step seed stream actually changes a real model's draws."""
    from flowforge.models import BookDDPM
    from flowforge.rollout import rollout_windows

    anchor = _anchor()
    fm, fs, rm, rs = _stats()
    model = BookDDPM(t_steps=5, ch=8, n_blocks=1, seed=0, use_history=True)
    out_a = rollout_windows(model, anchor, n_windows=2,
                            feature_mean=fm, feature_std=fs,
                            regime_mean=rm, regime_std=rs, seed=5)
    out_b = rollout_windows(model, anchor, n_windows=2,
                            feature_mean=fm, feature_std=fs,
                            regime_mean=rm, regime_std=rs, seed=5)
    np.testing.assert_array_equal(out_a["windows"], out_b["windows"])
    out_c = rollout_windows(model, anchor, n_windows=2,
                            feature_mean=fm, feature_std=fs,
                            regime_mean=rm, regime_std=rs, seed=6)
    assert not np.array_equal(out_a["windows"], out_c["windows"])
    assert np.isfinite(out_a["windows"]).all()


def test_stage3_flowforge_09_part_2_rollout_rejects_non_history_model():
    from flowforge.rollout import rollout_windows, teacher_forced_window

    anchor = _anchor()
    fm, fs, rm, rs = _stats()
    with pytest.raises(ValueError, match="history-condition"):
        rollout_windows(_PlainModel(), anchor, n_windows=2,
                        feature_mean=fm, feature_std=fs,
                        regime_mean=rm, regime_std=rs)
    with pytest.raises(ValueError, match="history-condition"):
        teacher_forced_window(_PlainModel(), anchor, None,
                              np.zeros((2, 2)), fm, fs, rm, rs)
    model = _EchoModel()
    with pytest.raises(ValueError, match="positive integer"):
        rollout_windows(model, anchor, n_windows=0,
                        feature_mean=fm, feature_std=fs,
                        regime_mean=rm, regime_std=rs)
    bad = anchor.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rollout_windows(model, bad, n_windows=2,
                        feature_mean=fm, feature_std=fs,
                        regime_mean=rm, regime_std=rs)


def test_stage3_flowforge_09_part_2_teacher_forced_uses_real_context():
    """Teacher-forced generation is conditioned on the real past."""
    from flowforge.rollout import teacher_forced_window

    anchor = _anchor()
    fm, fs, rm, rs = _stats()
    model = _EchoModel(bias=0.02)
    regime = np.zeros((2, 2))
    window = teacher_forced_window(model, anchor, None, regime,
                                   fm, fs, rm, rs, seed=3)
    assert window.shape == (2, WINDOW, N_FEATURES)
    assert len(model.calls) == 1
    np.testing.assert_allclose(model.calls[0]["context"], anchor,
                                      rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(window, anchor[:, -WINDOW:, :] + 0.02,
                               rtol=0, atol=1e-6)


def test_stage3_flowforge_09_part_2_mask_advances_with_generated_rows():
    """Unavailable anchor seconds stay masked; generated rows are available."""
    from flowforge.rollout import rollout_windows

    anchor = _anchor()
    mask = np.ones((2, LOOKBACK), dtype=bool)
    mask[:, :10] = False  # the first ten anchor seconds are missing
    fm, fs, rm, rs = _stats()
    model = _EchoModel()
    rollout_windows(model, anchor, anchor_mask=mask, n_windows=2,
                    feature_mean=fm, feature_std=fs,
                    regime_mean=rm, regime_std=rs, seed=1)
    assert len(model.calls) == 2
    # step 1 sees the anchor mask (early seconds unavailable);
    # step 2 sees the mask advanced by one generated window, so the
    # unavailable seconds have slid out of the lookback entirely.
    np.testing.assert_array_equal(model.calls[0]["mask"], mask)
    expected = np.concatenate(
        [mask, np.ones((2, WINDOW), dtype=bool)], axis=1)[:, -LOOKBACK:]
    np.testing.assert_array_equal(model.calls[1]["mask"], expected)
    # after one generated window the ten unavailable anchor seconds
    # have slid out of the lookback entirely: the mask is all True.
    assert expected.all()


# ------------------------------------------------------------ seam audit


def test_stage3_flowforge_09_part_2_seam_table_has_every_seam():
    from flowforge.evaluate_rollout import seam_table

    rng = np.random.default_rng(0)
    anchor = rng.normal(size=(3, LOOKBACK, N_FEATURES))
    gen = rng.normal(size=(3, 4, WINDOW, N_FEATURES))
    fm = np.zeros(N_FEATURES)
    fs = np.ones(N_FEATURES)
    rows = seam_table(anchor, gen, fm, fs)
    # one checkpoint per seam: anchor -> window 0, then 3 internal
    assert len(rows) == 4
    assert [r["seam"] for r in rows] == [0, 1, 2, 3]
    for r in rows:
        assert r["n"] == 3
        for k in ("spread_bps_jump_mean", "signed_flow_jump_mean",
                  "log_trade_count_jump_mean", "log_ret_jump_mean",
                  "log_bid_sz_jump_mean", "log_ask_sz_jump_mean"):
            assert np.isfinite(r[k]) and r[k] >= 0
    # a perfectly continuous path has zero jumps at every seam.
    gen_flat = np.zeros((2, 2, WINDOW, N_FEATURES))
    gen_flat[:, :, :, J_SPR] = np.log1p(1.5)
    anchor_c = np.zeros((2, LOOKBACK, N_FEATURES))
    anchor_c[:, :, J_SPR] = np.log1p(1.5)
    rows = seam_table(anchor_c, gen_flat, fm, fs)
    assert len(rows) == 2
    assert all(r["spread_bps_jump_mean"] == 0.0 for r in rows)
    assert all(r["log_ret_jump_mean"] == 0.0 for r in rows)


def test_stage3_flowforge_09_part_2_seam_jump_matches_oracle():
    from flowforge.evaluate_rollout import seam_table

    fm = np.zeros(N_FEATURES)
    fs = np.ones(N_FEATURES)
    anchor = np.zeros((1, LOOKBACK, N_FEATURES))
    gen = np.zeros((1, 1, WINDOW, N_FEATURES))
    anchor[:, :, J_SPR] = np.log1p(1.0)
    gen[:, :, :, J_SPR] = np.log1p(4.0)
    rows = seam_table(anchor, gen, fm, fs)
    assert len(rows) == 1
    # physical spread jump at the seam: |4 - 1| bps.
    assert rows[0]["spread_bps_jump_mean"] == pytest.approx(3.0)


# ------------------------------------------------------------ error vs horizon


def _paths(n=6, n_windows=4, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    real = rng.normal(size=(n, n_windows, WINDOW, N_FEATURES))
    gen = real.copy()
    if drift:
        for k in range(n_windows):
            gen[:, k, :, J_RET] += drift * (k + 1)
    return real, gen


def test_stage3_flowforge_09_part_2_cumret_uses_physical_units():
    """Cumulative log return is in physical units when stats are given.

    The stored log-return channel is standardized, so a raw sum is in
    train-standard-deviation units: a physical divergence threshold
    applied to it would be meaningless (a 480 s random walk sums to
    tens of standardized units). With feature stats the sum converts
    back to log-return units first.
    """
    from flowforge.evaluate_rollout import (
        cumulative_log_return, divergence_probability, error_vs_horizon)

    paths = np.zeros((2, 2, WINDOW, N_FEATURES))
    paths[:, :, :, J_RET] = 1.0  # one standardized unit per second
    fm = np.zeros(N_FEATURES)
    fs = np.ones(N_FEATURES)
    fs[J_RET] = 2.0
    fm[J_RET] = 0.5
    # physical per-second return is 1*2 + 0.5 = 2.5, so the cumulative
    # return at window 1 is 60 * 2.5 and at window 2 is 120 * 2.5.
    cum = cumulative_log_return(paths, fm, fs)
    assert cum.shape == (2, 2)
    np.testing.assert_allclose(cum[:, 0], 60 * 2.5, rtol=1e-12)
    np.testing.assert_allclose(cum[:, 1], 120 * 2.5, rtol=1e-12)
    # without stats the same paths sum in standardized units.
    cum_std = cumulative_log_return(paths)
    np.testing.assert_allclose(cum_std[:, 0], 60.0, rtol=1e-12)
    # the error and divergence metrics inherit the conversion: a
    # generated path off by 0.0001 physical log-return units per
    # second accumulates 0.006 over one 60 s window, under the 0.01
    # threshold, so nothing diverges at horizon 1.
    gen = paths.copy()
    gen[:, :, :, J_RET] += 0.0001 / fs[J_RET]
    rows = error_vs_horizon(paths, gen, horizons=(1,), feature_mean=fm,
                            feature_std=fs)
    assert rows[0]["cumret_rmse"] == pytest.approx(60 * 0.0001, rel=1e-9)
    rows = divergence_probability(paths, gen, horizons=(1,),
                                  threshold=0.01, feature_mean=fm,
                                  feature_std=fs)
    assert rows[0]["diverged_fraction"] == 0.0
    # the same per-second error without the conversion would read as
    # 60 x 0.00005 = 0.003 in standardized units, still under 0.01,
    # so check the other direction: an error of 0.01 per second in
    # standardized units is 0.02 physical per second, 1.2 over a
    # window, and must diverge under the physical threshold.
    gen2 = paths.copy()
    gen2[:, :, :, J_RET] += 0.01
    rows = divergence_probability(paths, gen2, horizons=(1,),
                                  threshold=0.01, feature_mean=fm,
                                  feature_std=fs)
    assert rows[0]["diverged_fraction"] == 1.0


def test_stage3_flowforge_09_part_2_error_vs_horizon_oracle():
    from flowforge.evaluate_rollout import error_vs_horizon

    real, gen = _paths()
    rows = error_vs_horizon(real, gen, horizons=(1, 2, 4))
    assert [r["horizon_windows"] for r in rows] == [1, 2, 4]
    # a perfect generator has zero error at every horizon.
    for r in rows:
        assert r["mean_w1"] == pytest.approx(0.0, abs=1e-12)
        assert r["cumret_rmse"] == pytest.approx(0.0, abs=1e-12)
    # a generator whose log returns drift by a fixed amount per
    # window accumulates error with horizon: the cumulative-return
    # error grows, and the marginal error grows with it.
    _, gen_drift = _paths(drift=0.05)
    rows = error_vs_horizon(real, gen_drift, horizons=(1, 2, 4))
    errs = [r["cumret_rmse"] for r in rows]
    assert errs[0] < errs[1] < errs[2]
    assert errs[0] == pytest.approx(0.05 * WINDOW, rel=1e-9)
    # exact oracle for the drifted cumulative return at horizon h:
    # each window k contributes drift*(k+1) per second for WINDOW
    # seconds, so the horizon-h error is drift*WINDOW*h*(h+1)/2.
    for r, h in zip(rows, (1, 2, 4)):
        assert r["cumret_rmse"] == pytest.approx(
            0.05 * WINDOW * h * (h + 1) / 2, rel=1e-9)
    with pytest.raises(ValueError, match="horizon"):
        error_vs_horizon(real, gen, horizons=(5,))
    with pytest.raises(ValueError, match="horizon"):
        error_vs_horizon(real, gen, horizons=(0,))


def test_stage3_flowforge_09_part_2_divergence_probability_oracle():
    from flowforge.evaluate_rollout import divergence_probability

    real, gen = _paths()
    rows = divergence_probability(real, gen, horizons=(1, 4),
                                  threshold=0.5)
    # identical paths never diverge at any horizon.
    assert all(r["diverged_fraction"] == 0.0 for r in rows)
    assert all(r["n"] == 6 for r in rows)
    _, gen_drift = _paths(drift=0.05)
    rows = divergence_probability(real, gen_drift, horizons=(1, 2, 4),
                                  threshold=2.0)
    # horizon-1 cumret error is 0.05*60 = 3 > 2: every path diverges.
    assert rows[0]["diverged_fraction"] == 1.0
    # a huge threshold diverges nothing.
    rows = divergence_probability(real, gen_drift, horizons=(4,),
                                  threshold=1e9)
    assert rows[0]["diverged_fraction"] == 0.0
    with pytest.raises(ValueError, match="threshold"):
        divergence_probability(real, gen, horizons=(1,), threshold=0.0)


def test_stage3_flowforge_09_part_2_divergence_session_intervals():
    """Session-block intervals resample whole sessions, not windows."""
    from flowforge.evaluate_rollout import divergence_probability

    real, gen = _paths(n=8, drift=0.0)
    # half the trajectories diverge at horizon 1 (drifted copies).
    gen2 = gen.copy()
    gen2[4:, :, :, J_RET] += 1.0
    session_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    rows = divergence_probability(real, gen2, horizons=(1,),
                                  threshold=0.5,
                                  session_ids=session_ids,
                                  n_boot=200, seed=0)
    assert rows[0]["diverged_fraction"] == pytest.approx(0.5)
    assert rows[0]["n_sessions"] == 4
    lo, hi = rows[0]["ci"]
    assert lo <= 0.5 <= hi
    assert 0.0 <= lo and hi <= 1.0
    # with a single session the bootstrap still returns a valid row.
    rows = divergence_probability(real, gen2, horizons=(1,),
                                  threshold=0.5,
                                  session_ids=np.zeros(8, dtype=int),
                                  n_boot=50, seed=0)
    assert rows[0]["n_sessions"] == 1
    assert rows[0]["ci"] is not None


def test_stage3_flowforge_09_part_2_rollout_report_structure():
    from flowforge.evaluate_rollout import rollout_report

    real, _ = _paths(n=6, n_windows=2)
    _, gen_a = _paths(n=6, n_windows=2, drift=0.01, seed=1)
    _, gen_b = _paths(n=6, n_windows=2, drift=0.05, seed=2)
    anchor = _anchor(batch=6)
    rep = rollout_report(
        real_paths=real,
        gen_by_method={"a": gen_a, "b": gen_b},
        anchor_context=anchor,
        feature_mean=np.zeros(N_FEATURES),
        feature_std=np.ones(N_FEATURES),
        horizons=(1, 2),
        threshold=1.0,
        session_ids=np.array([0, 0, 0, 1, 1, 1]))
    assert set(rep["methods"]) == {"a", "b"}
    for name in ("a", "b"):
        m = rep["methods"][name]
        assert len(m["error_vs_horizon"]) == 2
        assert len(m["divergence"]) == 2
        assert len(m["seams"]) == 2
    assert rep["horizons"] == [1, 2]
    assert rep["threshold"] == 1.0
    # the more drifted method has the larger horizon-2 error.
    assert (rep["methods"]["b"]["error_vs_horizon"][1]["cumret_rmse"]
            > rep["methods"]["a"]["error_vs_horizon"][1]["cumret_rmse"])


# ------------------------------------------------------------ experiment smoke


def test_stage3_flowforge_09_part_2_experiment_fixture_smoke():
    from examples.rollout_evaluate import run_rollout_evaluation

    out = run_rollout_evaluation(fixture=True, n_anchors=4,
                                 horizons=(1, 2), epochs=2, t_steps=20,
                                 ch=16, n_blocks=1, vae_epochs=2,
                                 verbose=False)
    assert out["fixture_only"] is True
    assert out["comparable_to_published_real_data"] is False
    assert "val" in out["reports"]
    methods = out["reports"]["val"]["methods"]
    for method in ("ddpm", "vae", "chunk_copy", "teacher_forced"):
        assert method in methods
        m = methods[method]
        assert len(m["error_vs_horizon"]) == 2
        assert len(m["divergence"]) == 2
        assert len(m["seams"]) == 2
        for row in m["error_vs_horizon"]:
            assert np.isfinite(row["mean_w1"])
            assert np.isfinite(row["cumret_rmse"])
    assert out["verdict"] in ("measured",)
    assert out["horizons"] == [1, 2]
