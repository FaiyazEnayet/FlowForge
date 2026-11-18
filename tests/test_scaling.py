"""Stage 3 flowforge P12: sensible scaling and regularization.

A compact model-size sweep of the BookDDPM denoiser: a preregistered
small / base / large capacity envelope compared under equal training
compute and under convergence-normalized training. Metrics: train and
validation denoising loss, nearest-neighbor distances, train-versus-
validation reconstruction probes, generated diversity, and inference
costs. The frozen test split is never touched: the study entry point
takes explicit train and validation arrays only, so no test array can
enter by construction.

Marker: stage3_flowforge_12.
"""

import importlib.util
import inspect
import os
import sys

import numpy as np
import pytest
import torch

from flowforge.models import BookDDPM, count_params, eval_denoising_loss, scoped_rng

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))


def _load_scaling_study():
    spec = importlib.util.spec_from_file_location(
        "stage3_scaling_study",
        os.path.join(REPO_ROOT, "examples", "scaling_study.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(n_train=64, n_val=32, seed=0, window=60, n_features=7):
    g = torch.Generator().manual_seed(seed)
    x_train = torch.randn(n_train, window, n_features, generator=g)
    r_train = torch.randn(n_train, 2, generator=g)
    x_val = torch.randn(n_val, window, n_features, generator=g)
    r_val = torch.randn(n_val, 2, generator=g)
    return x_train, r_train, x_val, r_val


# ------------------------------------------------------- capacity envelope

def test_stage3_flowforge_12_envelope_predeclared_and_ordered():
    ss = _load_scaling_study()
    env = ss.CAPACITY_ENVELOPE
    assert tuple(env) == ("small", "base", "large")
    assert env["base"] == {"ch": 96, "n_blocks": 3}
    params = {}
    for name, kw in env.items():
        torch.manual_seed(0)
        params[name] = count_params(BookDDPM(**kw))
    assert params["small"] == 55447
    assert params["base"] == 221767
    assert params["large"] == 480775
    assert params["small"] < params["base"] < params["large"]


def test_stage3_flowforge_12_total_nfe_accounting():
    ss = _load_scaling_study()
    n_train, batch, epochs, t_steps, n_gen = 64, 32, 2, 100, 16
    steps = epochs * ((n_train + batch - 1) // batch)
    total = ss.total_network_evals(steps, n_gen, t_steps)
    assert total == steps + n_gen * t_steps
    assert total == 4 + 1600


# ------------------------------------------------------- deterministic eval loss

def test_stage3_flowforge_12_eval_denoising_loss_deterministic():
    x_train, r_train, x_val, r_val = _synthetic()
    model = BookDDPM(seed=0)
    a = eval_denoising_loss(model, x_val, r_val, seed=11)
    b = eval_denoising_loss(model, x_val, r_val, seed=11)
    assert a == b
    assert np.isfinite(a) and a > 0


def test_stage3_flowforge_12_eval_denoising_loss_matches_p_loss_oracle():
    x_train, r_train, x_val, r_val = _synthetic(n_val=16)
    model = BookDDPM(seed=0)
    got = eval_denoising_loss(model, x_val, r_val, seed=11, batch=16)
    # Oracle: the same scoped draws, one p_loss call per batch.
    xc = x_val.permute(0, 2, 1).contiguous()
    with scoped_rng(11):
        with torch.no_grad():
            vals = [float(model.p_loss(xc[i:i + 16], r_val[i:i + 16]))
                    for i in range(0, 16, 16)]
    assert got == pytest.approx(float(np.mean(vals)), rel=1e-6)


def test_stage3_flowforge_12_eval_denoising_loss_rejects_empty():
    model = BookDDPM(seed=0)
    with pytest.raises(ValueError):
        eval_denoising_loss(model, torch.empty(0, 60, 7),
                            torch.empty(0, 2), seed=0)


def test_stage3_flowforge_12_eval_does_not_disturb_global_rng():
    x_train, r_train, x_val, r_val = _synthetic()
    model = BookDDPM(seed=0)
    torch.manual_seed(123)
    before = torch.randn(3)
    torch.manual_seed(123)
    eval_denoising_loss(model, x_val, r_val, seed=11)
    after = torch.randn(3)
    assert torch.equal(before, after)


# ------------------------------------------------------- nearest neighbors

def test_stage3_flowforge_12_nn_distance_zero_for_copies():
    ss = _load_scaling_study()
    x_train, _, x_val, _ = _synthetic()
    d_self = ss.nearest_neighbor_distances(x_val, x_val)
    assert np.allclose(d_self, 0.0, atol=1e-5)
    d_copy = ss.nearest_neighbor_distances(x_train[:8], x_train)
    assert np.allclose(d_copy, 0.0, atol=1e-5)
    d_rand = ss.nearest_neighbor_distances(
        torch.randn(32, 60, 7, generator=torch.Generator().manual_seed(9)),
        x_train)
    assert float(np.mean(d_copy)) < float(np.mean(d_rand))


def test_stage3_flowforge_12_nn_distance_rejects_mismatched_features():
    ss = _load_scaling_study()
    x_train, _, x_val, _ = _synthetic()
    with pytest.raises(ValueError):
        ss.nearest_neighbor_distances(x_val, torch.randn(8, 60, 5))


# ------------------------------------------------------- reconstruction probe

def test_stage3_flowforge_12_reconstruction_probe_deterministic():
    ss = _load_scaling_study()
    x_train, r_train, x_val, r_val = _synthetic()
    model = BookDDPM(seed=0)
    a = ss.reconstruction_probe(model, x_val, r_val, t_step=50, seed=5)
    b = ss.reconstruction_probe(model, x_val, r_val, t_step=50, seed=5)
    assert a == b
    assert np.isfinite(a) and a > 0


def test_stage3_flowforge_12_reconstruction_probe_null_on_random_data():
    ss = _load_scaling_study()
    x_train, r_train, x_val, r_val = _synthetic(n_train=64, n_val=64,
                                                seed=3)
    model = BookDDPM(seed=0)
    train_pw = ss.reconstruction_probe_per_window(
        model, x_train, r_train, t_step=50, seed=5)
    val_pw = ss.reconstruction_probe_per_window(
        model, x_val, r_val, t_step=50, seed=5)
    assert np.isfinite(train_pw).all() and np.isfinite(val_pw).all()
    assert (train_pw > 0).all() and (val_pw > 0).all()
    lo, hi = ss.block_bootstrap_diff_ci(
        train_pw, val_pw, block=8, n_boot=500, seed=0, paired=False)
    assert lo <= 0.0 <= hi


# ------------------------------------------------------- diversity

def test_stage3_flowforge_12_pairwise_diversity():
    ss = _load_scaling_study()
    same = torch.zeros(8, 60, 7)
    assert ss.pairwise_diversity(same) == pytest.approx(0.0)
    x_train, _, _, _ = _synthetic()
    assert ss.pairwise_diversity(x_train) > 0
    assert ss.pairwise_diversity(x_train) == \
        pytest.approx(ss.pairwise_diversity(x_train))


# ------------------------------------------------------- blocked intervals

def test_stage3_flowforge_12_block_bootstrap_ci():
    ss = _load_scaling_study()
    lo, hi = ss.block_bootstrap_diff_ci(np.full(40, 1.0), np.full(40, 1.0),
                                        block=10, n_boot=200, seed=0)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0)
    a = np.arange(40, dtype=float)
    b = a + 5.0
    lo, hi = ss.block_bootstrap_diff_ci(a, b, block=10, n_boot=500, seed=0)
    assert hi < 0.0
    lo, hi = ss.block_bootstrap_diff_ci(a, a.copy(), block=10, n_boot=500,
                                        seed=0)
    assert lo <= 0.0 <= hi


def test_stage3_flowforge_12_fast_w1_matches_repo_w1():
    ss = _load_scaling_study()
    rng = np.random.default_rng(0)
    real = rng.normal(size=(40, 60, 7))
    gen = rng.normal(size=(40, 60, 7)) + 0.1
    assert ss._mean_w1_fast(real, gen) == pytest.approx(
        ss._mean_w1(real, gen), abs=1e-9)
    lo, hi = ss.block_bootstrap_w1_diff_ci(
        real, [gen], [gen + 0.05], block=10, n_boot=50, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


def test_stage3_flowforge_12_block_bootstrap_rejects_bad_block():
    ss = _load_scaling_study()
    with pytest.raises(ValueError):
        ss.block_bootstrap_diff_ci(np.zeros(10), np.zeros(10), block=0,
                                   n_boot=100, seed=0)
    with pytest.raises(ValueError):
        ss.block_bootstrap_diff_ci(np.zeros(0), np.zeros(10), block=5,
                                   n_boot=100, seed=0)
    with pytest.raises(ValueError):
        ss.block_bootstrap_diff_ci(np.zeros(10), np.zeros(8), block=5,
                                   n_boot=100, seed=0, paired=True)


# ------------------------------------------------------- promotion rule

def test_stage3_flowforge_12_promotion_rule():
    ss = _load_scaling_study()
    gains = {"val_loss_ci": (-0.05, -0.01), "val_w1_ci": (-0.04, -0.005)}
    feasible = {"params_ratio": 2.17, "sample_ratio": 1.8}
    v = ss.promotion_verdict("large", gains, feasible)
    assert v["promoted"] is True
    v = ss.promotion_verdict("large", {"val_loss_ci": (-0.05, 0.01),
                                       "val_w1_ci": (-0.04, -0.005)},
                             feasible)
    assert v["promoted"] is False
    assert "val_loss" in v["reasons"][0]
    v = ss.promotion_verdict("large", gains,
                             {"params_ratio": 2.17, "sample_ratio": 2.9})
    assert v["promoted"] is False
    assert any("sample" in r for r in v["reasons"])
    v = ss.promotion_verdict("large", gains,
                             {"params_ratio": 3.5, "sample_ratio": 1.8})
    assert v["promoted"] is False


# ------------------------------------------------------- convergence epoch

def test_stage3_flowforge_12_convergence_epoch():
    ss = _load_scaling_study()
    assert ss.convergence_epoch([1.0, 0.5, 0.21, 0.205, 0.2]) == 2
    assert ss.convergence_epoch([1.0, 0.9, 0.8, 0.7]) == 3
    assert ss.convergence_epoch([0.2]) == 0
    with pytest.raises(ValueError):
        ss.convergence_epoch([])


# ------------------------------------------------------- end-to-end cell run

def test_stage3_flowforge_12_run_cell_equal_compute():
    ss = _load_scaling_study()
    x_train, r_train, x_val, r_val = _synthetic()
    cells = {}
    for name in ("small", "base", "large"):
        cells[name] = ss.run_cell(
            name, x_train, r_train, x_val, r_val, seed=0, epochs=2,
            batch=32, t_steps=20, n_gen=8, n_val_eval=16, n_nn_ref=16)
    steps = set(c["optimizer_steps"] for c in cells.values())
    assert steps == {4}
    for name, c in cells.items():
        assert c["params"] == {"small": 55447, "base": 221767,
                               "large": 480775}[name]
        assert c["total_nfe"] == c["optimizer_steps"] \
            + c["n_generated"] * c["t_steps"]
        assert np.isfinite(c["final_train_loss"]) \
            and np.isfinite(c["val_loss"])
        assert np.isfinite(c["val_mean_w1"])
        assert np.isfinite(c["nn_to_train"]) and np.isfinite(c["nn_to_val"])
        assert np.isfinite(c["recon_train"]) and np.isfinite(c["recon_val"])
        assert np.isfinite(c["diversity"])
        assert c["sample_seconds_per_1k"] > 0
    assert "test" not in inspect.signature(ss.run_cell).parameters
    assert "test" not in inspect.signature(ss.run_study).parameters


def test_stage3_flowforge_12_checkpoint_roundtrip(tmp_path):
    ss = _load_scaling_study()
    x_train, r_train, x_val, r_val = _synthetic()
    cell = ss.run_cell("base", x_train, r_train, x_val, r_val, seed=0,
                       epochs=1, batch=32, t_steps=20, n_gen=8,
                       n_val_eval=16, n_nn_ref=16)
    assert ss._load_checkpoint_cell(str(tmp_path), "equal_compute",
                                    "base", 0) is None
    ss._save_checkpoint_cell(str(tmp_path), "equal_compute", "base", 0,
                             cell)
    hit = ss._load_checkpoint_cell(str(tmp_path), "equal_compute",
                                   "base", 0)
    assert hit is not None
    assert hit["val_loss"] == pytest.approx(cell["val_loss"])
    assert hit["val_mean_w1"] == pytest.approx(cell["val_mean_w1"])
    assert np.allclose(hit["gen_windows"], cell["gen_windows"])
    assert np.allclose(hit["val_losses"], cell["val_losses"])
    assert hit["total_nfe"] == cell["total_nfe"]


def test_stage3_flowforge_12_run_study_schema_on_synthetic():
    ss = _load_scaling_study()
    x_train, r_train, x_val, r_val = _synthetic()
    out = ss.run_study(x_train, r_train, x_val, r_val, seeds=(0,),
                       epochs=1, batch=32, t_steps=20, n_gen=8,
                       n_val_eval=16, n_nn_ref=16, n_boot=100,
                       data_sha256="synthetic-fixture",
                       split_manifest_sha256="synthetic-fixture")
    assert out["protocol"] == "scaling-study"
    assert out["protocol_version"] == 1
    assert out["preregistration"]["capacity_envelope"] \
        == ss.CAPACITY_ENVELOPE
    assert set(out["rounds"]) == {"equal_compute", "convergence_normalized"}
    for rnd in out["rounds"].values():
        assert set(rnd["cells"]) == {"small", "base", "large"}
        for arm_cells in rnd["cells"].values():
            for cell in arm_cells.values():
                assert cell["optimizer_steps"] == 2
    assert out["verdict"]["arm"] == "large"
    assert out["verdict"]["promoted"] in (True, False)
    assert "reasons" in out["verdict"]
    assert out["optimizer"] == {"name": "Adam", "lr": 2e-4, "batch": 32}
    assert out["test_split_touched"] is False
