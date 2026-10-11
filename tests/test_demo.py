"""Tests for flowforge.demo: the dashboard's data helpers."""

import json
import os

import numpy as np
import pytest

from flowforge.demo import (N_FEATURES, N_SECONDS, load_demo_bundle,
                            midprice_path, spread_bps, unstandardize)

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "demo")


def _write_bundle(tmp_path, real, gen, names=None, mean=None, std=None,
                  s_real=None, s_gen=None, auc=0.6):
    names = names if names is not None else [f"f{i}" for i in range(7)]
    mean = mean if mean is not None else np.zeros(7)
    std = std if std is not None else np.ones(7)
    s_real = s_real if s_real is not None else np.array([0.4, 0.5])
    s_gen = s_gen if s_gen is not None else np.array([0.5, 0.6])
    bp = tmp_path / "demo.npz"
    mp = tmp_path / "metrics.json"
    np.savez(bp, real_windows=real, gen_windows=gen,
             feature_names=np.array(names), feature_mean=mean,
             feature_std=std, disc_scores_real=s_real,
             disc_scores_gen=s_gen, disc_auc=np.float32(auc))
    mp.write_text(json.dumps({"ok": True}))
    return str(bp), str(mp)


def _good_windows(n=4):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, N_SECONDS, N_FEATURES)).astype(np.float32)


def test_loader_accepts_well_formed_bundle(tmp_path):
    bp, mp = _write_bundle(tmp_path, _good_windows(), _good_windows())
    b = load_demo_bundle(bp, mp)
    assert b["real"].shape == (4, N_SECONDS, N_FEATURES)
    assert b["gen"].shape == (4, N_SECONDS, N_FEATURES)
    assert b["disc_auc"] == pytest.approx(0.6)
    assert b["metrics"] == {"ok": True}


def test_loader_rejects_missing_keys(tmp_path):
    bp = tmp_path / "demo.npz"
    mp = tmp_path / "metrics.json"
    np.savez(bp, real_windows=_good_windows())
    mp.write_text("{}")
    with pytest.raises(ValueError, match="missing keys"):
        load_demo_bundle(str(bp), str(mp))


def test_loader_rejects_bad_shape(tmp_path):
    bad = np.zeros((4, 30, N_FEATURES), dtype=np.float32)
    bp, mp = _write_bundle(tmp_path, bad, _good_windows())
    with pytest.raises(ValueError, match="shape"):
        load_demo_bundle(bp, mp)


def test_loader_rejects_non_finite(tmp_path):
    bad = _good_windows()
    bad[0, 0, 0] = np.nan
    bp, mp = _write_bundle(tmp_path, bad, _good_windows())
    with pytest.raises(ValueError, match="non-finite"):
        load_demo_bundle(bp, mp)


def test_loader_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_demo_bundle(str(tmp_path / "nope.npz"),
                         str(tmp_path / "nope.json"))


def test_loader_rejects_bad_auc(tmp_path):
    bp, mp = _write_bundle(tmp_path, _good_windows(), _good_windows(),
                           auc=1.5)
    with pytest.raises(ValueError, match="AUC"):
        load_demo_bundle(bp, mp)


def test_unstandardize_roundtrip():
    x = np.array([[[1.0, -2.0, 0.5, 0.0, 0.1, -0.3, 2.0]]])
    mean = np.arange(7.0)
    std = np.arange(1.0, 8.0)
    u = unstandardize(x, mean, std)
    np.testing.assert_allclose(u, x * std + mean)
    np.testing.assert_allclose((u - mean) / std, x)


def test_midprice_path_flat_on_zero_returns():
    w = np.zeros((60, 7))
    w[:, 1] = 0.5  # other features irrelevant
    px = midprice_path(w, j_ret=0, start=100.0)
    np.testing.assert_allclose(px, 100.0)


def test_midprice_path_compounds_returns():
    w = np.zeros((5, 7))
    w[:, 0] = 0.01
    px = midprice_path(w, j_ret=0, start=100.0)
    np.testing.assert_allclose(px, 100.0 * np.exp(0.01 * np.arange(1, 6)))


def test_spread_bps_inverts_log1p():
    w = np.zeros((10, 7))
    bps = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 1.5, 3.0, 0.1, 4.0, 8.0])
    w[:, 3] = np.log1p(bps)
    np.testing.assert_allclose(spread_bps(w, j_spread=3), bps, rtol=1e-6)


def test_committed_demo_bundle_loads():
    """Smoke test on the real committed demo/ bundle (shipped with the repo)."""
    bp = os.path.join(DEMO_DIR, "demo.npz")
    mp = os.path.join(DEMO_DIR, "metrics.json")
    if not (os.path.exists(bp) and os.path.exists(mp)):
        pytest.skip("demo bundle not built yet")
    b = load_demo_bundle(bp, mp)
    assert b["real"].shape[1:] == (N_SECONDS, N_FEATURES)
    assert b["gen"].shape[1:] == (N_SECONDS, N_FEATURES)
    assert 0.5 < b["disc_auc"] < 1.0
    assert set(b["metrics"]["models"]) == {"ddpm", "vae"}
    assert len(b["metrics"]["marginals"]) == N_FEATURES
