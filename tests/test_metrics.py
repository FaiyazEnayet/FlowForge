"""Sanity checks for every metric on synthetic inputs with known answers.

Rule: if any of these fail, the metric is wrong, not the test.
"""

import numpy as np

from flowforge.data import FEATURE_NAMES
from flowforge.evaluate import (acf, auc_score, discriminator_auc,
                                flow_acf_table, tstr_flow_direction,
                                wasserstein_1d)


def test_wasserstein_identical_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert wasserstein_1d(x, x.copy()) < 1e-9


def test_wasserstein_shifted_normals():
    # W1 between N(0,1) and N(2,1) is exactly |mean diff| = 2
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=20000)
    b = rng.normal(2, 1, size=20000)
    assert abs(wasserstein_1d(a, b) - 2.0) < 0.05


def test_acf_ar1():
    # AR(1) with phi=0.8 has lag-1 autocorrelation 0.8
    rng = np.random.default_rng(2)
    e = rng.normal(size=50000)
    x = np.zeros_like(e)
    for i in range(1, len(e)):
        x[i] = 0.8 * x[i - 1] + e[i]
    assert abs(acf(x, 1) - 0.8) < 0.02


def test_acf_iid_is_zero():
    rng = np.random.default_rng(3)
    x = rng.normal(size=50000)
    assert abs(acf(x, 1)) < 0.03
    assert abs(acf(x, 5)) < 0.03


def test_auc_perfect():
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert auc_score(y, s) == 1.0


def test_auc_random_is_half():
    rng = np.random.default_rng(4)
    y = np.repeat([1, 0], 500)
    s = rng.random(1000)
    assert abs(auc_score(y, s) - 0.5) < 0.05


def test_discriminator_cannot_separate_identical():
    # Same distribution for "real" and "generated": test AUC must be ~0.5.
    # If this fails, the discriminator or AUC code is broken.
    # Note: val AUC is NOT asserted here. The val split is the
    # discriminator's own training data, and a 60k-parameter MLP memorizes
    # 120 training windows (val AUC = 1.0 by overfitting). The honest number
    # is the held-out test AUC, which must be ~0.5.
    rng = np.random.default_rng(5)
    real_val = rng.normal(size=(60, 60, 7))
    gen_val = rng.normal(size=(60, 60, 7))
    real_test = rng.normal(size=(60, 60, 7))
    gen_test = rng.normal(size=(60, 60, 7))
    out = discriminator_auc(real_val, gen_val, real_test, gen_test,
                            seed=0, epochs=10)
    assert 0.40 < out["auc_test"] < 0.60, out
    assert np.isfinite(out["auc_val"])


def test_discriminator_detects_obvious_shift():
    # Distributions far apart: AUC must be near 1, or the test is blind.
    rng = np.random.default_rng(6)
    real_val = rng.normal(0, 1, size=(60, 60, 7))
    gen_val = rng.normal(5, 1, size=(60, 60, 7))
    real_test = rng.normal(0, 1, size=(60, 60, 7))
    gen_test = rng.normal(5, 1, size=(60, 60, 7))
    out = discriminator_auc(real_val, gen_val, real_test, gen_test,
                            seed=0, epochs=10)
    assert out["auc_test"] > 0.95, out


def test_discriminator_returns_test_scores():
    # The figure script needs per-window test scores, real first.
    rng = np.random.default_rng(7)
    n_r, n_g = 25, 30
    real_val = rng.normal(size=(40, 60, 7))
    gen_val = rng.normal(size=(40, 60, 7))
    real_test = rng.normal(size=(n_r, 60, 7))
    gen_test = rng.normal(size=(n_g, 60, 7))
    out = discriminator_auc(real_val, gen_val, real_test, gen_test,
                            seed=0, epochs=5)
    scores = np.asarray(out["test_scores"], dtype=np.float64)
    assert scores.shape == (n_r + n_g,)
    assert int(out["test_n_real"]) == n_r
    assert np.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_flow_acf_averages_per_window():
    # Level-shifted regimes: concatenating overlapping windows (the old bug)
    # centers on the grand mean and reports near-1.0 autocorrelation, while
    # per-window averaging recovers the true AR(1) coefficient.
    rng = np.random.default_rng(11)
    phi = 0.3
    j = FEATURE_NAMES.index("signed_flow")
    wins = rng.normal(size=(40, 60, 7)) * 0.1
    for rows, mean in ((slice(0, 20), 5.0), (slice(20, 40), -5.0)):
        e = rng.normal(size=(20, 60))
        x = np.zeros_like(e)
        for i in range(1, 60):
            x[:, i] = phi * x[:, i - 1] + e[:, i]
        wins[rows, :, j] = mean + x
    row = flow_acf_table(wins, wins)["rows"][0]
    assert abs(row["real_acf"] - phi) < 0.15, row
    assert row["real_ci"][0] <= row["real_acf"] <= row["real_ci"][1]
    assert row["n_real"] == 40


def test_flow_acf_ci_covers_truth():
    # On plain AR(1) windows the 95% CI should contain the true coefficient.
    # Long windows: the per-window autocorrelation estimator has a
    # finite-sample downward bias that vanishes as the window grows.
    rng = np.random.default_rng(12)
    phi = 0.5
    j = FEATURE_NAMES.index("signed_flow")
    e = rng.normal(size=(120, 600))
    x = np.zeros_like(e)
    for i in range(1, 600):
        x[:, i] = phi * x[:, i - 1] + e[:, i]
    wins = np.zeros((120, 600, 7))
    wins[:, :, j] = x
    row = flow_acf_table(wins, wins)["rows"][0]
    assert row["real_ci"][0] <= phi <= row["real_ci"][1], row


def test_tstr_recovers_learnable_structure():
    # Synthetic windows where first-half imbalance determines second-half
    # flow direction: both the synthetic-trained and real-trained models
    # must ace it, or the TSTR machinery is broken.
    rng = np.random.default_rng(13)
    j_imb = FEATURE_NAMES.index("imbalance")
    j_sf = FEATURE_NAMES.index("signed_flow")

    def make(n):
        w = rng.normal(size=(n, 60, 7)) * 0.1
        s = rng.choice([-1.0, 1.0], size=n)
        w[:, :30, j_imb] += s[:, None] * 2.0
        w[:, 30:, j_sf] += s[:, None] * 2.0
        return w

    out = tstr_flow_direction(make(60), make(60), make(60), seed=0)
    assert out["acc_synth"] > 0.9, out
    assert out["acc_real"] > 0.9, out
    assert 0.0 <= out["majority"] <= 1.0
    assert out["n_test"] == 60


def test_book_consistency_detects_violation():
    """FL2: independent imbalance feature violates the physical constraint."""
    from flowforge.evaluate import book_consistency_table
    rng = np.random.default_rng(0)
    n, w = 10, 60
    # Standardized gen windows; physical units via mean/std.
    f_mean = np.zeros(7)
    f_std = np.ones(7)
    gen = rng.normal(size=(n, w, 7)) * 0.1
    j_bid = FEATURE_NAMES.index("log_bid_sz")
    j_ask = FEATURE_NAMES.index("log_ask_sz")
    j_imb = FEATURE_NAMES.index("imbalance")
    # Set bid/ask sizes to imply imbalance ~0.5, but set the imbalance
    # feature to -0.5 (violation).
    gen[:, :, j_bid] = np.log(3.0)  # bid=3
    gen[:, :, j_ask] = np.log(1.0)  # ask=1 -> implied imb = 0.5
    gen[:, :, j_imb] = -0.5  # violates the constraint
    rep = book_consistency_table(gen, f_mean, f_std)
    assert rep["imbalance_violation_rate"] == 1.0
    assert rep["imbalance_mae"] == 1.0  # |-0.5 - 0.5| = 1.0


def test_constrain_imbalance_fixes_violation():
    """FL2: deriving imbalance from sizes yields zero violation rate."""
    from flowforge.evaluate import (book_consistency_table,
                                    constrain_imbalance)
    rng = np.random.default_rng(1)
    n, w = 10, 60
    f_mean = np.zeros(7)
    f_std = np.ones(7)
    gen = rng.normal(size=(n, w, 7)) * 0.1
    j_imb = FEATURE_NAMES.index("imbalance")
    gen[:, :, j_imb] = rng.uniform(-1, 1, size=(n, w))  # arbitrary
    fixed = constrain_imbalance(gen, f_mean, f_std)
    rep = book_consistency_table(fixed, f_mean, f_std)
    assert rep["imbalance_violation_rate"] == 0.0
    assert rep["imbalance_mae"] == 0.0


def test_block_bootstrap_ci_covers_true_mean_iid():
    from flowforge.evaluate import _block_bootstrap_ci
    rng = np.random.default_rng(100)
    covers = 0
    trials = 40
    for t in range(trials):
        x = rng.normal(5.0, 2.0, size=500)
        lo, hi = _block_bootstrap_ci(x, block_len=1, n_boot=500, seed=t)
        if lo <= 5.0 <= hi:
            covers += 1
    assert covers >= 30, f"coverage {covers}/{trials}"


def test_block_bootstrap_wider_than_naive_under_dependence():
    from flowforge.evaluate import _block_bootstrap_ci, _mean_ci
    rng = np.random.default_rng(101)
    e = rng.normal(size=2000)
    x = np.zeros_like(e)
    for i in range(1, len(e)):
        x[i] = 0.8 * x[i - 1] + e[i]
    naive = _mean_ci(x)
    boot = _block_bootstrap_ci(x, block_len=20, n_boot=1000, seed=0)
    naive_w = naive[1] - naive[0]
    boot_w = boot[1] - boot[0]
    assert boot_w > naive_w, f"boot {boot_w:.4f} vs naive {naive_w:.4f}"


def test_block_bootstrap_ci_sensitivity_to_block_len():
    from flowforge.evaluate import _block_bootstrap_ci
    rng = np.random.default_rng(102)
    e = rng.normal(size=1500)
    x = np.zeros_like(e)
    for i in range(1, len(e)):
        x[i] = 0.6 * x[i - 1] + e[i]
    widths = []
    for bl in (1, 5, 20):
        lo, hi = _block_bootstrap_ci(x, block_len=bl, n_boot=500, seed=0)
        widths.append(hi - lo)
    assert widths[2] > widths[0], widths


def test_flow_acf_table_uses_block_bootstrap():
    from flowforge.evaluate import flow_acf_table
    rng = np.random.default_rng(103)
    real = rng.normal(size=(40, 60, 7))
    gen = rng.normal(size=(40, 60, 7))
    tab = flow_acf_table(real, gen, lags=(1, 2))
    for row in tab["rows"]:
        assert row["ci_method"] == "block_bootstrap_block10", row
        assert len(row["real_ci"]) == 2
        assert row["real_ci"][0] <= row["real_ci"][1]
