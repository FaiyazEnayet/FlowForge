"""Stage 3 flowforge P07: activity imbalance, volatility tails and
regime-balanced training.

The stratification must be causal (labels from the lookback BEFORE a
window), thresholds must be fixed on TRAIN, scarce strata must refuse
to report rather than claim, and the training arms (uniform, capped
importance weights, stratified minibatches) must run at equal compute.
Outcome labels (derived from the future window) are analysis-only and
must never reach a training sampler.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from flowforge.data import (FEATURE_NAMES, LOOKBACK, WINDOW, BookDataset,
                            SplitWindows, build_dataset, load_dataset,
                            save_dataset)
from flowforge.evaluate import full_report
from flowforge.models import train_ddpm, train_vae, BookDDPM, CondVAE
from flowforge.strata import (
    NotEnoughDataError,
    OutcomeLabels,
    StrataSpec,
    assign_strata,
    capped_importance_weights,
    check_sufficient,
    effective_sample_size,
    fit_strata,
    independent_episode_mask,
    outcome_tail_labels,
    past_features,
    stratified_minibatch_indices,
    stratified_report,
    stratum_counts,
    window_returns,
)


def _synth_chunk(tmp_path, name, n_sec=900, seed=0,
                 t0_ms=1_700_000_000_000):
    rng = np.random.default_rng(seed)
    n_q = n_sec * 10
    times = t0_ms + np.arange(n_q) * 100
    mid = 80000 * np.exp(np.cumsum(rng.normal(0, 2e-5, n_q)))
    spread = np.abs(rng.normal(1.0, 0.3, n_q)).clip(0.1)
    q = pd.DataFrame({"time": times, "bid_px": mid - spread / 2,
                      "bid_sz": rng.uniform(0.01, 2.0, n_q),
                      "ask_px": mid + spread / 2,
                      "ask_sz": rng.uniform(0.01, 2.0, n_q)})
    n_t = n_sec * 4
    t = pd.DataFrame(
        {"time": np.sort(rng.integers(t0_ms, t0_ms + n_sec * 1000, n_t)),
         "price": 80000 + rng.normal(0, 5, n_t),
         "qty": rng.uniform(0.001, 0.5, n_t),
         "is_buyer_maker": rng.random(n_t) < 0.5})
    d = tmp_path / name
    d.mkdir()
    q.to_parquet(d / "quotes.parquet")
    t.to_parquet(d / "trades.parquet")
    return str(d)


def _three_chunks(tmp_path, n_sec=900, gap_ms=2_000_000, seed0=0):
    return [_synth_chunk(tmp_path, f"c{i}", n_sec=n_sec,
                         t0_ms=1_700_000_000_000 + i * gap_ms,
                         seed=seed0 + i)
            for i in range(3)]


def _tiny_split(n=24, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, WINDOW, len(FEATURE_NAMES))).astype(np.float32)
    regime = rng.normal(size=(n, 2)).astype(np.float32)
    t_start = np.arange(n, dtype=np.float64) * 10.0
    return SplitWindows(x=x, regime=regime, t_start=t_start,
                        t_end=t_start + 59.0,
                        session_ids=np.zeros(n, dtype=np.int64))


# ---------------------------------------------------------------- causality

def test_stage3_flowforge_07_past_features_are_causal(tmp_path):
    # Past features for a window must not move when data at or after the
    # window start changes, and must move when the lookback changes.
    # The test session is chunk 2 (default 3-chunk split: train/val/test).
    t0 = 1_700_000_000_000
    c1 = _synth_chunk(tmp_path, "c1", n_sec=1200, t0_ms=t0, seed=0)
    c2 = _synth_chunk(tmp_path, "c2", n_sec=1200, t0_ms=t0 + 2_000_000,
                      seed=1)
    c3 = _synth_chunk(tmp_path, "c3", n_sec=1200, t0_ms=t0 + 4_000_000,
                      seed=2)
    ds = build_dataset([c1, c2, c3])
    base = past_features(ds.test, ds)
    assert base.shape == (ds.test.x.shape[0], 4)
    assert np.isfinite(base).all()

    # Perturb quotes and trades at/after the first test window start:
    # every past feature of that window must be unchanged.
    q = pd.read_parquet(f"{c3}/quotes.parquet")
    t = pd.read_parquet(f"{c3}/trades.parquet")
    w0 = float(ds.test.t_start[0]) * 1000
    q.loc[q["time"] >= w0, ["bid_px", "ask_px"]] *= 1.5
    t.loc[t["time"] >= w0, "qty"] *= 7.0
    q.to_parquet(f"{c3}/quotes.parquet")
    t.to_parquet(f"{c3}/trades.parquet")
    ds_after = build_dataset([c1, c2, c3])
    assert ds_after.test.x.shape == ds.test.x.shape
    np.testing.assert_array_equal(past_features(ds_after.test, ds_after)[0],
                                  base[0])
    # The window's own returns DID change: the perturbation is real.
    assert not np.allclose(window_returns(ds_after.test, ds_after)[0],
                           window_returns(ds.test, ds)[0])

    # Perturb inside the lookback (the first half of the 60 s before
    # the window start, so internal lookback returns move rather than
    # a uniform rescaling that cancels): the past features of that
    # window must move.
    q = pd.read_parquet(f"{c3}/quotes.parquet")
    q.loc[(q["time"] >= w0 - 60_000) & (q["time"] < w0 - 30_000),
          ["bid_px", "ask_px"]] *= 1.5
    q.to_parquet(f"{c3}/quotes.parquet")
    ds_lb = build_dataset([c1, c2, c3])
    assert not np.allclose(past_features(ds_lb.test, ds_lb)[0], base[0])


def test_stage3_flowforge_07_outcome_label_is_window_derived_not_regime(tmp_path):
    # The outcome label (window return tail) is derived from the future
    # window; the regime label comes only from the past. Editing the
    # window content must change the outcome label and leave the regime
    # label untouched, and an outcome label must never reach training.
    t0 = 1_700_000_000_000
    c1 = _synth_chunk(tmp_path, "c1", n_sec=1200, t0_ms=t0, seed=0)
    c2 = _synth_chunk(tmp_path, "c2", n_sec=1200, t0_ms=t0 + 2_000_000,
                      seed=1)
    c3 = _synth_chunk(tmp_path, "c3", n_sec=1200, t0_ms=t0 + 4_000_000,
                      seed=2)
    ds = build_dataset([c1, c2, c3])
    feats = past_features(ds.test, ds)
    spec = fit_strata(past_features(ds.train, ds),
                      ("past_vol", "past_spread",
                       "past_activity", "past_depth"), n_bins=4)
    regime_before = assign_strata(feats, spec)
    out_before = outcome_tail_labels(window_returns(ds.test, ds),
                                     window_returns(ds.train, ds))
    # Pick the last window that is NOT in the return tail, then move
    # only the last 5 seconds of that window's own midprice: its
    # outcome label must flip into the tail (a log(1.5) jump dwarfs
    # the ~0.001 tail threshold). Regime labels of that window and
    # every earlier one must stay put: their lookbacks end before the
    # edit, so the edit cannot reach them. (Later windows may move:
    # the edited seconds fall inside their lookbacks, which is exactly
    # what causal features are supposed to see.)
    k = int(np.where(~out_before.labels)[0][-1])
    q = pd.read_parquet(f"{c3}/quotes.parquet")
    wk = (float(ds.test.t_start[k]) + 55.0) * 1000
    q.loc[q["time"] >= wk, ["bid_px", "ask_px"]] *= 1.5
    q.to_parquet(f"{c3}/quotes.parquet")
    ds2 = build_dataset([c1, c2, c3])
    np.testing.assert_array_equal(
        assign_strata(past_features(ds2.test, ds2), spec)[:k + 1],
        regime_before[:k + 1])
    out_after = outcome_tail_labels(window_returns(ds2.test, ds2),
                                    window_returns(ds.train, ds))
    assert out_after.labels[k] and not out_before.labels[k]
    # Training samplers refuse outcome labels outright.
    x = torch.from_numpy(ds.train.x[:16])
    reg = torch.from_numpy(ds.train.regime[:16])
    model = BookDDPM(seed=0)
    with pytest.raises(TypeError, match="outcome"):
        train_ddpm(model, x, reg, epochs=1, batch=8, seed=0,
                   stratum_labels=out_before)


# ------------------------------------------------------- thresholds on train

def test_stage3_flowforge_07_thresholds_fixed_on_train():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 4))
    train[:, 0] = np.abs(train[:, 0])  # past vol is non-negative
    spec = fit_strata(train, ("past_vol", "past_spread",
                              "past_activity", "past_depth"), n_bins=5)
    assert isinstance(spec, StrataSpec)
    assert spec.n_train == 500
    # Values far outside the train range clip to the edge bins, never
    # refit and never raise.
    extreme = np.array([[1e6, -1e6, 1e6, -1e6], [-1e6, 1e6, -1e6, 1e6]])
    got = assign_strata(extreme, spec)
    assert got[0, 0] == 4 and got[1, 0] == 0
    # Fitting the same bins on shifted data gives different edges: the
    # spec really is the train fit, and assign never refits.
    shifted = fit_strata(train + 10.0, spec.feature_names, n_bins=5)
    assert not np.allclose(shifted.edges["past_vol"],
                           spec.edges["past_vol"])
    # Wrong feature names are refused, not silently realigned, and a
    # wrong column count is refused too.
    with pytest.raises(ValueError):
        fit_strata(train, ("past_vol", "no_such_feature",
                           "past_activity", "past_depth"), n_bins=5)
    with pytest.raises(ValueError):
        assign_strata(train, spec,
                      feature_names=("past_spread", "past_vol",
                                     "past_activity", "past_depth"))
    with pytest.raises(ValueError):
        assign_strata(train[:, :3], spec)


def test_stage3_flowforge_07_independent_episode_mask_exact():
    # Windows every 10 s, 60 s long, one session: footprints are 120 s,
    # so the greedy mask keeps starts 0, 120, 240, ... exactly.
    n = 101
    t_start = np.arange(n, dtype=np.float64) * 10.0
    mask = independent_episode_mask(t_start, t_start + 59.0,
                                    np.zeros(n, dtype=np.int64))
    kept = t_start[mask]
    np.testing.assert_array_equal(kept, np.arange(0, 1001, 120.0))
    # Two sessions interleave independently: each keeps its own grid.
    t2 = np.concatenate([t_start, t_start])
    s2 = np.concatenate([np.zeros(n, dtype=np.int64),
                         np.ones(n, dtype=np.int64)])
    mask2 = independent_episode_mask(t2, t2 + 59.0, s2)
    assert int(mask2.sum()) == 2 * int(mask.sum())


# ---------------------------------------------------------- scarce strata

def test_stage3_flowforge_07_scarce_stratum_raises_not_enough_data():
    # A stratum with fewer independent episodes than the minimum must
    # raise, naming the stratum, instead of reporting a number.
    labels = np.array([0] * 90 + [1] * 10)
    t_start = np.arange(100, dtype=np.float64) * 10.0
    mask = independent_episode_mask(t_start, t_start + 59.0,
                                    np.zeros(100, dtype=np.int64))
    counts = stratum_counts(labels, n_bins=2, independent_mask=mask)
    assert counts[1]["independent"] == 1  # 10 overlapping, 1 footprint
    with pytest.raises(NotEnoughDataError, match="stratum 1"):
        check_sufficient(counts, min_independent=10, split_name="test")
    # The report refuses the same scarcity instead of publishing it.
    rng = np.random.default_rng(0)
    real = rng.normal(size=(100, WINDOW, len(FEATURE_NAMES)))
    gen = rng.normal(size=(100, WINDOW, len(FEATURE_NAMES)))
    with pytest.raises(NotEnoughDataError):
        stratified_report(real, gen, labels, labels,
                          t_start=t_start, t_end=t_start + 59.0,
                          session_ids=np.zeros(100, dtype=np.int64),
                          min_independent=10)


# ------------------------------------------------- weights, ESS, batching

def test_stage3_flowforge_07_capped_weights_and_ess():
    labels = np.array([0] * 90 + [1] * 10)
    w, ess = capped_importance_weights(labels, n_bins=2, cap=2.0)
    assert w.shape == (100,)
    assert np.isclose(w.mean(), 1.0)
    # Rare stratum capped at 2 after renormalization: 20/7; common is
    # (5/9) * (10/7) = 50/63.
    assert np.isclose(w[labels == 1][0], 20 / 7, atol=1e-9)
    assert np.isclose(w[labels == 0][0], 50 / 63, atol=1e-9)
    assert np.isclose(ess, 10000 / (90 * (50 / 63) ** 2
                                    + 10 * (20 / 7) ** 2), rtol=1e-9)
    # Without the cap the ESS is lower (more extreme weights).
    _, ess_raw = capped_importance_weights(labels, n_bins=2, cap=1e9)
    assert ess_raw < ess
    # ESS identities on known weights.
    assert effective_sample_size(np.ones(4)) == 4.0
    assert effective_sample_size(np.array([4.0, 0.0, 0.0, 0.0])) == 1.0
    # Outcome labels are refused here too.
    with pytest.raises(TypeError, match="outcome"):
        capped_importance_weights(
            OutcomeLabels(labels=np.array([True] * 10)), n_bins=2)


def test_stage3_flowforge_07_balanced_decile_weights_are_uniform():
    # Quantile deciles hold equal mass by construction, so
    # inverse-frequency weights over them are all 1 (ESS = n). The
    # P07 weights arm therefore uses the binary tail partition
    # (top decile vs rest) instead; this pins the fact so a future
    # reader does not mistake it for a bug in the weighting.
    labels = np.repeat(np.arange(10), 100)
    w, ess = capped_importance_weights(labels, n_bins=10, cap=10.0)
    assert np.allclose(w, 1.0)
    assert ess == pytest.approx(1000.0)


def test_stage3_flowforge_07_stratified_minibatches_balanced_equal_compute():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=400)
    batches = stratified_minibatch_indices(labels, n_bins=4,
                                           batch_size=32, n_batches=13,
                                           seed=7)
    assert len(batches) == 13  # ceil(400 / 32), same step count as uniform
    for b in batches:
        assert len(b) == 32
        counts = np.bincount(labels[b], minlength=4)
        assert counts.max() - counts.min() <= 1  # balanced within one
    assert stratified_minibatch_indices(
        labels, n_bins=4, batch_size=32, n_batches=13,
        seed=7)[0].tolist() == batches[0].tolist()  # deterministic
    assert not np.array_equal(
        stratified_minibatch_indices(labels, n_bins=4, batch_size=32,
                                     n_batches=13, seed=8)[0], batches[0])


# ------------------------------------------------------------- trainers

def test_stage3_flowforge_07_trainers_accept_weights_and_stratified():
    torch.manual_seed(0)
    n = 64
    x = torch.randn(n, WINDOW, len(FEATURE_NAMES))
    regime = torch.randn(n, 2)
    labels = np.array([0] * 32 + [1] * 32)
    w, _ = capped_importance_weights(labels, n_bins=2, cap=2.0)
    ddpm = BookDDPM(seed=0)
    hist_w, _ = train_ddpm(ddpm, x, regime, epochs=1, batch=16, seed=0,
                           verbose=False, weights=w)
    assert len(hist_w) == 1 and np.isfinite(hist_w[0])
    ddpm2 = BookDDPM(seed=0)
    hist_s, _ = train_ddpm(ddpm2, x, regime, epochs=1, batch=16, seed=0,
                           verbose=False, stratum_labels=labels)
    assert len(hist_s) == 1 and np.isfinite(hist_s[0])
    # Uniform weights reproduce the unweighted run exactly: the weights
    # path must not change the default behaviour.
    ddpm3 = BookDDPM(seed=0)
    hist_u, _ = train_ddpm(ddpm3, x, regime, epochs=1, batch=16, seed=0,
                           verbose=False)
    ddpm4 = BookDDPM(seed=0)
    hist_uw, _ = train_ddpm(ddpm4, x, regime, epochs=1, batch=16, seed=0,
                            verbose=False, weights=np.ones(n))
    assert hist_uw == pytest.approx(hist_u, rel=1e-5)
    vae = CondVAE(seed=0)
    hist_v = train_vae(vae, x, regime, epochs=1, batch=16, seed=0,
                       verbose=False, weights=w)
    assert len(hist_v) == 1 and np.isfinite(hist_v[0])
    vae2 = CondVAE(seed=0)
    hist_vs = train_vae(vae2, x, regime, epochs=1, batch=16, seed=0,
                        verbose=False, stratum_labels=labels)
    assert len(hist_vs) == 1 and np.isfinite(hist_vs[0])
    with pytest.raises(ValueError):
        train_ddpm(BookDDPM(seed=0), x, regime, epochs=1, batch=16,
                   seed=0, weights=np.ones(n - 1))
    with pytest.raises(ValueError):
        train_vae(CondVAE(seed=0), x, regime, epochs=1, batch=16,
                  seed=0, stratum_labels=labels[:10])


# ------------------------------------------------------- dataset fields

def test_stage3_flowforge_07_dataset_carries_past_activity_depth(tmp_path):
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    for split in (ds.train, ds.val, ds.test):
        assert split.past_activity is not None
        assert split.past_depth is not None
        assert split.past_activity.shape == (split.x.shape[0],)
        assert split.past_depth.shape == (split.x.shape[0],)
        assert np.isfinite(split.past_activity).all()
        assert np.isfinite(split.past_depth).all()
        assert (split.past_activity >= 0).all()
    # Hand-check the first train window against the bars: activity is
    # the trade count over the 60 s before the window, depth the mean
    # of the mean log sizes over the same lookback.
    from flowforge.data import _bars_from_chunk, _load_chunk, _segments
    q, t = _load_chunk(dirs[0])
    b = _bars_from_chunk(q, t)
    seg = _segments(b)[0]
    cnt = np.expm1(seg["log_trade_count"].to_numpy(dtype=np.float64))
    depth = (seg["log_bid_sz"].to_numpy(dtype=np.float64)
             + seg["log_ask_sz"].to_numpy(dtype=np.float64)) / 2
    start = LOOKBACK  # first window start in the segment
    assert ds.train.past_activity[0] == pytest.approx(
        cnt[start - LOOKBACK:start].sum())
    assert ds.train.past_depth[0] == pytest.approx(
        depth[start - LOOKBACK:start].mean())
    # Save/load round-trip preserves the new fields exactly.
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    ds2 = load_dataset(p)
    np.testing.assert_array_equal(ds.train.past_activity,
                                  ds2.train.past_activity)
    np.testing.assert_array_equal(ds.test.past_depth,
                                  ds2.test.past_depth)
    # A dataset built without the fields loads them as None, never
    # invented: past_features must refuse it loudly.
    bare = BookDataset(
        train=_tiny_split(), val=_tiny_split(), test=_tiny_split(),
        feature_mean=np.zeros(len(FEATURE_NAMES)),
        feature_std=np.ones(len(FEATURE_NAMES)),
        regime_mean=np.zeros(2), regime_std=np.ones(2))
    assert bare.train.past_activity is None
    with pytest.raises(ValueError, match="past_activity"):
        past_features(bare.train, bare)


# ------------------------------------------------------------- reporting

def test_stage3_flowforge_07_stratified_report_average_and_worst():
    rng = np.random.default_rng(0)
    n = 400
    real = rng.normal(size=(n, WINDOW, len(FEATURE_NAMES)))
    gen = rng.normal(size=(n, WINDOW, len(FEATURE_NAMES)))
    # Stratum 1 is deliberately worse: generated windows are shifted.
    labels = np.array([0] * 200 + [1] * 200)
    gen[labels == 1] += 1.0
    rep = stratified_report(real, gen, labels, labels)
    assert set(rep["strata"]) == {0, 1}
    assert rep["strata"][0]["n_real"] == 200
    assert rep["strata"][1]["mean_w1"] > rep["strata"][0]["mean_w1"]
    assert rep["worst_stratum"] == 1
    assert rep["average_mean_w1"] < rep["worst_mean_w1"]
    assert 0.0 <= rep["average_coverage"] <= 1.0
    assert 0.0 <= rep["worst_coverage"] <= 1.0
    assert rep["diversity_ratio"] > 0


def test_stage3_flowforge_07_full_report_strata_section_opt_in():
    rng = np.random.default_rng(0)
    real_test = rng.normal(size=(60, WINDOW, len(FEATURE_NAMES)))
    gen_test = rng.normal(size=(60, WINDOW, len(FEATURE_NAMES)))
    real_val = rng.normal(size=(60, WINDOW, len(FEATURE_NAMES)))
    gen_val = rng.normal(size=(60, WINDOW, len(FEATURE_NAMES)))
    rep = full_report(real_test, gen_test, real_val, gen_val, seed=0)
    assert "strata" not in rep
    labels = np.array([0] * 30 + [1] * 30)
    rep2 = full_report(real_test, gen_test, real_val, gen_val, seed=0,
                       stratum_labels=labels)
    assert rep2["strata"]["average_mean_w1"] > 0
    assert rep2["strata"]["worst_stratum"] in (0, 1)
