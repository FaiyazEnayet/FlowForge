"""Tests for cheap temporal baselines (FL4)."""

import numpy as np

from flowforge.baselines import MovingBlockResampler, VARBaseline


def _synth_windows(n=200, seed=0):
    rng = np.random.default_rng(seed)
    # AR(1)-like temporal structure across windows
    x = np.zeros((n, 60, 7))
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.normal(0, 0.5, size=(60, 7))
    return x.astype(np.float32)


def test_block_resampler_shapes_and_stats():
    x = _synth_windows()
    rs = MovingBlockResampler(x, block_len=4)
    gen, stats = rs.generate(50, seed=0)
    assert gen.shape == (50, 60, 7)
    assert np.isfinite(gen).all()
    assert stats["method"] == "moving_block"
    assert stats["n_generated"] == 50
    assert stats["sample_seconds"] >= 0
    assert stats["peak_memory_mb"] >= 0


def test_block_resampler_reproducible():
    x = _synth_windows()
    rs = MovingBlockResampler(x, block_len=4)
    g1, _ = rs.generate(20, seed=42)
    g2, _ = rs.generate(20, seed=42)
    assert np.array_equal(g1, g2)


def test_block_resampler_copies_training_windows():
    # Every generated window must exactly equal some training window.
    x = _synth_windows(n=50)
    rs = MovingBlockResampler(x, block_len=3)
    gen, _ = rs.generate(30, seed=1)
    train_set = {x[i].tobytes() for i in range(len(x))}
    for i in range(len(gen)):
        assert gen[i].tobytes() in train_set


def test_var_fit_and_generate():
    x = _synth_windows(n=200)
    var = VARBaseline(order=2, ridge=1.0).fit(x)
    assert var.coef_ is not None
    assert var.coef_.shape[1] == 60 * 7
    gen, stats = var.generate(30, x, seed=0)
    assert gen.shape == (30, 60, 7)
    assert np.isfinite(gen).all()
    assert stats["method"] == "var_p2"
    assert stats["sample_seconds_per_1k"] >= 0


def test_var_reproducible():
    x = _synth_windows(n=200)
    var = VARBaseline(order=2, ridge=1.0).fit(x)
    g1, _ = var.generate(10, x, seed=5)
    g2, _ = var.generate(10, x, seed=5)
    assert np.array_equal(g1, g2)
