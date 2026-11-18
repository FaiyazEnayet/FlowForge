"""Stage 3 P11: utility tasks better than majority-class signed-flow sign.

Fixture-first tests for the preregistered utility-task suite
(flowforge/tasks.py, wired through flowforge/evaluate.py). Every numeric
expectation below is a hand-computed known answer or an independent
NumPy oracle; none of it shares code with the module under test.

Registered suite (VAL-gated before any test evaluation, test frozen):

  spread_widen_30s / spread_widen_10s: target-half mean spread above
      feature-half mean spread. VAL gate PASSED (real logistic beats
      majority on PR-AUC and Brier).
  imbalance_pos_30s: target-half mean imbalance above zero. VAL gate
      PASSED.
  activity_high_10s: next-10s trade count above the train median.
      VAL gate PASSED.
  vol_high_30s: target realized volatility above the train median.
      VAL gate PASSED, marginally; the train median is exactly zero
      (59% of train target halves have exactly zero realized vol),
      so the split is zero-vs-nonzero, not a true median split.
  activity_high_30s: target-half trade count above the train median.
      VAL gate FAILED (no signal; honest null).

Features use rows 0:30 only; labels use rows 30:60 (or 30:40) only, so
the feature horizon ends strictly before the target horizon starts.
"""

import json

import numpy as np

from flowforge.data import FEATURE_NAMES, WINDOW
from flowforge.evaluate import inverse_transform


def _phys(n=8, seed=0):
    """Physical-unit windows: (n, 60, 7), feature order of FEATURE_NAMES.

    Baseline values are neutral (log_ret 0, spread log1p(5bps),
    sizes log(1), imbalance 0, signed_flow 0, log_trade_count
    log1p(2)); tests overwrite the cells a task reads.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n, WINDOW, len(FEATURE_NAMES)))
    x[:, :, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(5.0)
    x[:, :, FEATURE_NAMES.index("log_bid_sz")] = 0.0
    x[:, :, FEATURE_NAMES.index("log_ask_sz")] = 0.0
    x[:, :, FEATURE_NAMES.index("log_trade_count")] = np.log1p(2.0)
    return x


def _spread(windows, value, rows):
    windows[:, rows, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(value)
    return windows


def _imbalance(windows, value, rows):
    windows[:, rows, FEATURE_NAMES.index("imbalance")] = value
    return windows


def _counts(windows, count, rows):
    windows[:, rows, FEATURE_NAMES.index("log_trade_count")] = np.log1p(count)
    return windows


def _rets(windows, values, rows):
    windows[:, rows, FEATURE_NAMES.index("log_ret")] = values
    return windows


# ------------------------------------------------------------ timing

def test_stage3_flowforge_11_feature_target_horizons_disjoint():
    from flowforge.tasks import (FEATURE_END, FEATURE_START, TARGET_10_END,
                                TARGET_30_END, TARGET_START,
                                verify_timing)
    assert FEATURE_START == 0
    assert FEATURE_END == 30
    assert TARGET_START == 30
    assert TARGET_10_END == 40
    assert TARGET_30_END == 60
    assert FEATURE_END <= TARGET_START
    # hand-built window times: 60 consecutive seconds
    t_start = np.array([1000.0, 2000.0])
    timing = verify_timing(t_start)
    assert timing["feature_end_second"] == [1029.0, 2029.0]
    assert timing["target_start_second"] == [1030.0, 2030.0]
    assert timing["overlap_seconds"] == 0
    # a target that starts inside the feature horizon is refused
    try:
        verify_timing(t_start, target_start_row=29)
    except ValueError as e:
        assert "overlap" in str(e)
    else:
        raise AssertionError("overlapping horizons were not refused")


# ------------------------------------------------------------ labels

def test_stage3_flowforge_11_spread_widen_label_known_answer():
    from flowforge.tasks import task_labels
    w = _phys(4)
    _spread(w, 5.0, slice(0, 30))
    _spread(w, 9.0, slice(30, 60))   # window 0: widening
    w[1, 30:, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(5.0)
    w[2, 30:, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(2.0)
    w[3, 30:, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(9.0)
    # window 3 target rows 30:40 widen, rows 40:60 narrow: 10s label 1,
    # 30s label depends on the whole target half mean
    w[3, 40:, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(1.0)
    labels = task_labels(w, {"activity_30s_median": 0.0,
                             "activity_10s_median": 0.0,
                             "vol_30s_median": 0.0,
                             "activity_feature_median": 0.0,
                             "vol_feature_median": 0.0})
    assert labels["spread_widen_30s"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert labels["spread_widen_10s"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_stage3_flowforge_11_imbalance_label_known_answer():
    from flowforge.tasks import task_labels
    w = _phys(3)
    _imbalance(w, 0.4, slice(30, 60))
    w[1, 30:, FEATURE_NAMES.index("imbalance")] = -0.2
    w[2, 30:, FEATURE_NAMES.index("imbalance")] = 0.0
    labels = task_labels(w, {"activity_30s_median": 0.0,
                             "activity_10s_median": 0.0,
                             "vol_30s_median": 0.0,
                             "activity_feature_median": 0.0,
                             "vol_feature_median": 0.0})
    assert labels["imbalance_pos_30s"].tolist() == [1.0, 0.0, 0.0]


def test_stage3_flowforge_11_activity_vol_thresholds_from_train_only():
    from flowforge.tasks import fit_thresholds, task_labels
    train = _phys(6)
    _counts(train, 1.0, slice(30, 60))
    _counts(train, 10.0, slice(30, 60))
    train[0, 30:, FEATURE_NAMES.index("log_trade_count")] = np.log1p(1.0)
    train[1, 30:, FEATURE_NAMES.index("log_trade_count")] = np.log1p(3.0)
    train[2, 30:, FEATURE_NAMES.index("log_trade_count")] = np.log1p(5.0)
    train[3, 30:, FEATURE_NAMES.index("log_trade_count")] = np.log1p(7.0)
    train[4, 30:, FEATURE_NAMES.index("log_trade_count")] = np.log1p(9.0)
    train[5, 30:, FEATURE_NAMES.index("log_trade_count")] = np.log1p(11.0)
    # target-half totals per window: 30 * count; median of
    # [30, 90, 150, 210, 270, 330] is (150 + 210) / 2 = 180
    th = fit_thresholds(train)
    assert abs(th["activity_30s_median"] - 180.0) < 1e-9
    # independent oracle: median of the per-window target totals
    totals = np.expm1(
        train[:, 30:, FEATURE_NAMES.index("log_trade_count")]).sum(axis=1)
    assert th["activity_30s_median"] == float(np.median(totals))
    # a test window with total 200 is high, 100 is low, under the TRAIN
    # median even though a pooled median would differ
    test = _phys(2)
    _counts(test, 200.0 / 30, slice(30, 60))
    test[1, 30:, FEATURE_NAMES.index("log_trade_count")] = np.log1p(100.0 / 30)
    labels = task_labels(test, th)
    assert labels["activity_high_30s"].tolist() == [1.0, 0.0]


def test_stage3_flowforge_11_vol_label_known_answer():
    from flowforge.tasks import fit_thresholds, task_labels
    train = _phys(4)
    # target-half log returns with known stds: 0, 1, 2, 3 (symmetric
    # two-point patterns give exactly those stds)
    for i, s in enumerate((0.0, 1.0, 2.0, 3.0)):
        vals = np.tile([s, -s], 15)
        train[i, 30:, FEATURE_NAMES.index("log_ret")] = vals
    th = fit_thresholds(train)
    assert th["vol_30s_median"] == 1.5  # median of [0, 1, 2, 3]
    test = _phys(2)
    test[0, 30:, FEATURE_NAMES.index("log_ret")] = np.tile([2.0, -2.0], 15)
    test[1, 30:, FEATURE_NAMES.index("log_ret")] = np.tile([0.5, -0.5], 15)
    labels = task_labels(test, th)
    assert labels["vol_high_30s"].tolist() == [1.0, 0.0]


# ------------------------------------------------------------ features

def test_stage3_flowforge_11_feature_extraction_known_answer():
    from flowforge.tasks import extract_features
    w = _phys(2)
    j = {n: FEATURE_NAMES.index(n) for n in FEATURE_NAMES}
    w[0, :30, j["log_spread_bps"]] = np.log1p(5.0)
    w[0, :15, j["log_spread_bps"]] = np.log1p(3.0)
    w[0, 15:30, j["log_spread_bps"]] = np.log1p(7.0)
    w[0, :30, j["imbalance"]] = 0.2
    w[0, :30, j["log_ret"]] = np.tile([0.1, -0.1], 15)
    x = extract_features(w)
    assert x.shape == (2, 12)
    # independent NumPy oracle for window 0
    f = w[0, :30, :]
    expected_means = f.mean(axis=0)
    expected_vol = f[:, j["log_ret"]].std()
    expected_trend_spread = (
        f[15:, j["log_spread_bps"]].mean()
        - f[:15, j["log_spread_bps"]].mean())
    assert np.allclose(x[0, :7], expected_means)
    assert abs(x[0, 7] - expected_vol) < 1e-12
    assert abs(x[0, 8] - expected_trend_spread) < 1e-12
    assert abs(expected_trend_spread - (np.log1p(7.0) - np.log1p(3.0))) < 1e-12
    assert abs(x[0, 7] - 0.1) < 1e-12  # std of alternating +-0.1
    # rows after the feature horizon never enter the features
    w2 = w.copy()
    w2[0, 30:, :] = 99.0
    assert np.allclose(extract_features(w2), x)


# ------------------------------------------------------------ metrics

def test_stage3_flowforge_11_pr_auc_known_answers():
    from flowforge.tasks import pr_auc
    y = np.array([1.0, 0.0, 1.0, 0.0])
    # perfect ranking: both positives on top
    assert abs(pr_auc(y, np.array([0.9, 0.1, 0.8, 0.2])) - 1.0) < 1e-12
    # worst ranking: both positives at the bottom
    # AP = (1/3 + 2/4) / 2 = 0.416666...
    assert abs(pr_auc(y, np.array([0.1, 0.9, 0.2, 0.8]))
               - (1 / 3 + 2 / 4) / 2) < 1e-12
    # constant scores: AP equals the prevalence
    assert abs(pr_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-12
    # single positive: AP is the precision at its rank
    assert abs(pr_auc(np.array([0.0, 1.0, 0.0]),
                      np.array([0.9, 0.5, 0.1])) - 0.5) < 1e-12


def test_stage3_flowforge_11_brier_known_answers():
    from flowforge.tasks import brier_score
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y, np.array([1.0, 0.0, 1.0, 0.0])) == 0.0
    assert abs(brier_score(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.25) < 1e-12
    # worst possible: 1.0
    assert brier_score(y, np.array([0.0, 1.0, 0.0, 1.0])) == 1.0


def test_stage3_flowforge_11_calibration_known_answers():
    from flowforge.tasks import calibration_table
    y = np.array([1.0, 1.0, 0.0, 0.0])
    perfect = calibration_table(y, np.array([1.0, 1.0, 0.0, 0.0]))
    assert perfect["ece"] == 0.0
    assert perfect["mean_predicted"] == 0.5
    assert perfect["mean_observed"] == 0.5
    # all 0.5, half positive: every bin is off by 0.0 in the single
    # occupied bin? No: bin [0.5, 0.6) holds all 4, acc 0.5, conf 0.5.
    flat = calibration_table(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert flat["ece"] == 0.0
    # confident and wrong: probs 0.9 on negatives, 0.1 on positives.
    # Bin [0.1, 0.2) holds the positives (acc 1.0, conf 0.1) and bin
    # [0.9, 1.0] the negatives (acc 0.0, conf 0.9): ECE = 0.45 + 0.45.
    bad = calibration_table(y, np.array([0.1, 0.1, 0.9, 0.9]))
    assert abs(bad["ece"] - 0.9) < 1e-12


def test_stage3_flowforge_11_majority_baseline_known_answers():
    from flowforge.tasks import brier_score, majority_baseline, pr_auc
    y_train = np.array([1.0, 1.0, 1.0, 0.0])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    prev, scores = majority_baseline(y_train, len(y))
    assert prev == 0.75
    assert (scores == 0.75).all()
    assert abs(pr_auc(y, scores) - 0.6) < 1e-12  # prevalence of y
    # Brier of a constant 0.75 against y with prevalence 0.6:
    # 0.6 * (1 - 0.75)^2 + 0.4 * 0.75^2 = 0.2625
    assert abs(brier_score(y, scores) - 0.2625) < 1e-12


def test_stage3_flowforge_11_persistence_baseline_known_answer():
    from flowforge.tasks import persistence_baseline
    w = _phys(3)
    _imbalance(w, 0.3, slice(0, 30))
    w[1, :30, FEATURE_NAMES.index("imbalance")] = -0.3
    w[2, :30, FEATURE_NAMES.index("imbalance")] = 0.0
    th = {"activity_30s_median": 0.0, "activity_10s_median": 0.0,
          "vol_30s_median": 0.0, "activity_feature_median": 0.0,
          "vol_feature_median": 0.0}
    base = persistence_baseline(w, th)
    assert base["imbalance_pos_30s"].tolist() == [1.0, 0.0, 0.0]
    # spread persistence: rising within the feature half predicts widening
    _spread(w, 3.0, slice(0, 15))
    _spread(w, 7.0, slice(15, 30))
    w[1, :15, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(7.0)
    w[1, 15:30, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(3.0)
    base = persistence_baseline(w, th)
    assert base["spread_widen_30s"].tolist() == [1.0, 0.0, 1.0]


def test_stage3_flowforge_11_random_baseline_reproducible():
    from flowforge.tasks import random_baseline_scores
    a = random_baseline_scores(50, seed=0)
    b = random_baseline_scores(50, seed=0)
    c = random_baseline_scores(50, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert ((a >= 0.0) & (a <= 1.0)).all()


# ------------------------------------------------------------ model

def test_stage3_flowforge_11_logistic_recovers_learnable_structure():
    from flowforge.tasks import (extract_features, fit_logistic_probs,
                                pr_auc, task_labels)
    rng = np.random.default_rng(0)
    # feature-half imbalance sign determines the target-half sign:
    # a learnable conditional dependency the majority predictor misses.
    n_tr, n_va = 120, 60
    def make(n):
        w = np.zeros((n, WINDOW, len(FEATURE_NAMES)))
        w[:, :, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(5.0)
        w[:, :, FEATURE_NAMES.index("log_trade_count")] = np.log1p(2.0)
        sign = rng.choice([-1.0, 1.0], size=n)
        w[:, :30, FEATURE_NAMES.index("imbalance")] = (
            sign * 0.5)[:, None] + rng.normal(0, 0.05, (n, 30))
        w[:, 30:, FEATURE_NAMES.index("imbalance")] = (
            sign * 0.5)[:, None] + rng.normal(0, 0.05, (n, 30))
        return w
    th = {"activity_30s_median": 0.0, "activity_10s_median": 0.0,
          "vol_30s_median": 0.0, "activity_feature_median": 0.0,
          "vol_feature_median": 0.0}
    w_tr, w_va = make(n_tr), make(n_va)
    y_tr = task_labels(w_tr, th)["imbalance_pos_30s"]
    y_va = task_labels(w_va, th)["imbalance_pos_30s"]
    p = fit_logistic_probs(extract_features(w_tr), y_tr,
                           extract_features(w_va), seed=0)
    assert pr_auc(y_va, p) > float(y_va.mean()) + 0.2
    assert pr_auc(y_va, p) > 0.9


def test_stage3_flowforge_11_inverse_transform_roundtrip_for_tasks():
    # tasks read physical windows; standardized windows must be
    # inverse-transformed first, and the round trip is exact.
    from flowforge.tasks import extract_features, task_labels
    w = _phys(4)
    _spread(w, 9.0, slice(30, 60))
    mean = np.array([0.5, -0.2, 1.0, 1.0, 0.0, 0.0, 0.3])
    std = np.array([1.0, 0.5, 2.0, 2.0, 0.5, 3.0, 0.8])
    z = (w - mean) / std
    back = inverse_transform(z.astype(np.float32), mean, std)
    assert np.allclose(back, w, atol=1e-5)
    th = {"activity_30s_median": 0.0, "activity_10s_median": 0.0,
          "vol_30s_median": 0.0, "activity_feature_median": 0.0,
          "vol_feature_median": 0.0}
    assert task_labels(back, th)["spread_widen_30s"].tolist() == \
        task_labels(w, th)["spread_widen_30s"].tolist()
    assert np.allclose(extract_features(back), extract_features(w),
                       atol=1e-5)


# ------------------------------------------------------------ suite runner

def test_stage3_flowforge_11_suite_runner_smoke():
    from flowforge.evaluate import utility_task_report
    rng = np.random.default_rng(3)
    # small synthetic splits where imbalance sign persists (learnable)
    # and spread widening is random (null), so the runner must report
    # both a pass and a null without crashing.
    def make(n):
        w = np.zeros((n, WINDOW, len(FEATURE_NAMES)))
        w[:, :, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(5.0)
        w[:, :, FEATURE_NAMES.index("log_trade_count")] = np.log1p(2.0)
        sign = rng.choice([-1.0, 1.0], size=n)
        w[:, :30, FEATURE_NAMES.index("imbalance")] = (
            sign * 0.5)[:, None] + rng.normal(0, 0.05, (n, 30))
        w[:, 30:, FEATURE_NAMES.index("imbalance")] = (
            sign * 0.5)[:, None] + rng.normal(0, 0.05, (n, 30))
        w[:, 30:, FEATURE_NAMES.index("log_spread_bps")] = np.log1p(
            rng.choice([2.0, 9.0], size=n))[:, None]
        return w.astype(np.float32)
    mean = np.zeros(len(FEATURE_NAMES))
    std = np.ones(len(FEATURE_NAMES))
    train, val, test = make(120), make(60), make(60)
    gen_val = make(60)
    rep = utility_task_report(train, val, test,
                              {"chunk_copy": gen_val},
                              feature_mean=mean, feature_std=std,
                              n_boot=20, block_lengths=(1, 5), seed=0)
    assert rep["schema"] == "flowforge-utility-tasks/1"
    assert set(rep["tasks"]) == {
        "spread_widen_30s", "spread_widen_10s", "activity_high_30s",
        "activity_high_10s", "imbalance_pos_30s", "vol_high_30s"}
    for name, t in rep["tasks"].items():
        for section in ("val_gate", "test"):
            assert "real" in t[section] and "majority" in t[section]
            assert "pr_auc" in t[section]["real"]
            assert "brier" in t[section]["real"]
            assert "calibration" in t[section]["real"]
        assert "chunk_copy" in t["test"]
        assert "paired_diffs" in t
        assert t["paired_diffs"], f"no blocked intervals for {name}"
        assert t["timing"]["overlap_seconds"] == 0
    # the learnable task passes its VAL gate; the suite is JSON-safe
    assert rep["tasks"]["imbalance_pos_30s"]["val_gate"]["passed"] is True
    json.dumps(rep)
    # per-task test counts match the split sizes
    assert rep["n_train"] == 120 and rep["n_val"] == 60
    assert rep["n_test"] == 60
