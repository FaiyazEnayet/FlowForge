"""Stage 3 flowforge C01: sweep discriminator and selection contract.

The sweep discriminator must fit and score on disjoint validation blocks,
and the demo must use the frozen sweep winner config, never silent defaults.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "stage3_sweep", os.path.join(REPO_ROOT, "examples", "sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "stage3_demo", os.path.join(REPO_ROOT, "examples", "demo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _val_arrays(n=200, seed=0):
    """Synthetic val windows: chronological, one session, 10s stride."""
    rng = np.random.default_rng(seed)
    t_start = np.arange(n, dtype=np.float64) * 10.0
    t_end = t_start + 60.0
    session = np.zeros(n, dtype=np.int64)
    x = rng.normal(size=(n, 60, 7)).astype(np.float32)
    regime = rng.normal(size=(n, 2)).astype(np.float32)
    return x, regime, t_start, t_end, session


def _intervals_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def test_split_blocks_are_disjoint_including_lookbacks():
    sweep = _load_sweep()
    _, _, t_start, t_end, session = _val_arrays()
    fit_idx, score_idx, gap = sweep.split_val_blocks(
        t_start, t_end, session)
    assert len(fit_idx) > 0 and len(score_idx) > 0
    assert gap >= 120.0
    assert not set(fit_idx) & set(score_idx)
    # No shared raw second between any fit and score window, lookbacks
    # included: score lookback starts at t_start - 60.
    for i in fit_idx:
        for j in score_idx:
            assert not _intervals_overlap(
                t_start[i] - 60.0, t_end[i],
                t_start[j] - 60.0, t_end[j]), (
                f"fit window {i} overlaps score window {j} including lookback")


def test_split_blocks_drop_gap_windows():
    sweep = _load_sweep()
    _, _, t_start, t_end, session = _val_arrays(n=40)
    fit_idx, score_idx, _ = sweep.split_val_blocks(
        t_start, t_end, session)
    # Windows between the fit block end and the score block start (the
    # 120s boundary gap) belong to neither block.
    assert max(fit_idx) < min(score_idx)
    assert min(score_idx) - max(fit_idx) > 1


def test_score_generated_reports_score_block_auc():
    sweep = _load_sweep()
    rng = np.random.default_rng(1)
    n = 64
    real_fit = rng.normal(size=(n, 60, 7)).astype(np.float32)
    real_score = rng.normal(size=(n, 60, 7)).astype(np.float32)
    # Both fit and score generations come from the same shifted generator:
    # the discriminator learns the direction on fit and must detect it on
    # the disjoint score block.
    gen_fit = (rng.normal(size=(n, 60, 7)) + 3.0).astype(np.float32)
    gen_score = (rng.normal(size=(n, 60, 7)) + 3.0).astype(np.float32)
    res = sweep._score_generated(
        "test", gen_fit, gen_score, real_fit, real_score,
        seed=7, disc_epochs=5)
    assert res["insufficient_samples"] is False
    assert res["val_disc_auc"] > 0.7
    assert res["fit_n"] == n and res["score_n"] == n


def test_score_generated_identical_distribution_near_chance():
    sweep = _load_sweep()
    rng = np.random.default_rng(2)
    n = 64
    real_fit = rng.normal(size=(n, 60, 7)).astype(np.float32)
    gen_fit = rng.normal(size=(n, 60, 7)).astype(np.float32)
    real_score = rng.normal(size=(n, 60, 7)).astype(np.float32)
    gen_score = rng.normal(size=(n, 60, 7)).astype(np.float32)
    res = sweep._score_generated(
        "test", gen_fit, gen_score, real_fit, real_score,
        seed=7, disc_epochs=5)
    assert 0.3 < res["val_disc_auc"] < 0.7


def test_score_generated_insufficient_samples_admitted():
    sweep = _load_sweep()
    rng = np.random.default_rng(3)
    n = 8
    real_fit = rng.normal(size=(n, 60, 7)).astype(np.float32)
    gen_fit = rng.normal(size=(n, 60, 7)).astype(np.float32)
    real_score = rng.normal(size=(n, 60, 7)).astype(np.float32)
    gen_score = rng.normal(size=(n, 60, 7)).astype(np.float32)
    res = sweep._score_generated(
        "test", gen_fit, gen_score, real_fit, real_score,
        seed=7, disc_epochs=1)
    assert res["insufficient_samples"] is True
    assert res["val_disc_auc"] is None
    assert np.isfinite(res["val_mean_w1"])


def _write_sweep_artifact(tmp_path, winner_name, kind, config):
    art = {
        "protocol": "disjoint-discriminator",
        "protocol_version": 1,
        "seed": 7,
        "data_sha256": "0" * 64,
        "selection_rule": "lowest val mean W1, tie-break lower disc AUC",
        "results": [],
        "winner": {"name": winner_name, "kind": kind, "config": config},
    }
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps(art))
    return str(p)


def test_demo_config_follows_neural_sweep_winner(tmp_path):
    demo = _load_demo()
    art = _write_sweep_artifact(
        tmp_path, "C_long_ema", "neural",
        {"ch": 96, "n_blocks": 3, "epochs": 300, "ema_decay": 0.999})
    config, winner = demo.resolve_ddpm_config(sweep_path=art, env={})
    assert config == {"ch": 96, "n_blocks": 3, "epochs": 300,
                      "ema_decay": 0.999}
    assert winner["name"] == "C_long_ema" and winner["kind"] == "neural"
    assert winner["source"] == "sweep"


def test_demo_config_records_baseline_winner_plainly(tmp_path):
    demo = _load_demo()
    art = _write_sweep_artifact(
        tmp_path, "baseline_var2", "baseline", {"params": 123})
    env = {"FF_DDPM_CH": "96", "FF_DDPM_BLOCKS": "3",
           "FF_DDPM_EPOCHS": "300", "FF_DDPM_EMA": ""}
    config, winner = demo.resolve_ddpm_config(sweep_path=art, env=env)
    assert winner["name"] == "baseline_var2"
    assert winner["kind"] == "baseline"
    assert winner["source"] == "sweep"
    assert config["ch"] == 96


def test_demo_config_requires_explicit_config(tmp_path):
    demo = _load_demo()
    with pytest.raises(RuntimeError, match="no frozen DDPM config"):
        demo.resolve_ddpm_config(
            sweep_path=str(tmp_path / "missing.json"), env={})


def test_demo_config_explicit_env_without_artifact(tmp_path):
    demo = _load_demo()
    env = {"FF_DDPM_CH": "128", "FF_DDPM_BLOCKS": "4",
           "FF_DDPM_EPOCHS": "300", "FF_DDPM_EMA": "0.999"}
    config, winner = demo.resolve_ddpm_config(
        sweep_path=str(tmp_path / "missing.json"), env=env)
    assert config == {"ch": 128, "n_blocks": 4, "epochs": 300,
                      "ema_decay": 0.999}
    assert winner["name"] is None and winner["source"] == "env"
