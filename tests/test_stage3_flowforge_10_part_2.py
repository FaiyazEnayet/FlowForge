"""Stage 3 P10 part 2: challenger model families and the comparison.

Part 2 builds the challengers under the part-1 contract and runs the
model-versus-model comparison on the expanded corpus. These tests
use small synthetic fixtures only, so they run without the recorded
corpus: the comparison itself lives in examples/compare_families.py.

Independent oracles: the mixture NLL and the rectified-flow loss are
recomputed here from their definitions, the causality check perturbs
one step of the input and requires earlier predictions to be
bit-identical, and the sampling checks require exact determinism
under a fixed seed and different draws under a different seed.
"""

import numpy as np
import pytest
import torch

from flowforge.data import N_FEATURES, WINDOW, BookDataset, SplitWindows
from flowforge.family_contract import (
    GENERATION_SEEDS, default_contract)

torch.set_num_threads(2)


# ------------------------------------------------------- fixtures

def _synthetic_windows(n=48, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, WINDOW, N_FEATURES)).astype(np.float32)
    # mild autocorrelation so the causal model has something to learn
    x[:, 1:, :] += 0.3 * x[:, :-1, :]
    regime = rng.standard_normal((n, 2)).astype(np.float32)
    past = rng.standard_normal((n, WINDOW, N_FEATURES)).astype(np.float32)
    mask = np.ones((n, WINDOW), dtype=bool)
    return x, regime, past, mask


def _tiny_dataset():
    x_tr, r_tr, p_tr, m_tr = _synthetic_windows(48, seed=7)
    x_va, r_va, p_va, m_va = _synthetic_windows(16, seed=8)
    x_te, r_te, p_te, m_te = _synthetic_windows(16, seed=9)
    sess = np.zeros(48, dtype=int)
    sess_va = np.zeros(16, dtype=int)
    sess_te = np.ones(16, dtype=int)
    # window starts stride 10 seconds apart, as in the recorded
    # corpus, so the whole train split is one segment for the
    # chunk-copy baseline
    t = 10 * np.arange(48, dtype=np.float64)
    t_va = 10 * np.arange(16, dtype=np.float64)
    t_te = 10 * np.arange(16, dtype=np.float64)
    return BookDataset(
        train=SplitWindows(x=x_tr, regime=r_tr, t_start=t, t_end=t + 59,
                           session_ids=sess, past_context=p_tr,
                           past_mask=m_tr),
        val=SplitWindows(x=x_va, regime=r_va, t_start=t_va,
                         t_end=t_va + 59, session_ids=sess_va,
                         past_context=p_va, past_mask=m_va),
        test=SplitWindows(x=x_te, regime=r_te, t_start=t_te,
                          t_end=t_te + 59, session_ids=sess_te,
                          past_context=p_te, past_mask=m_te),
        feature_mean=np.zeros(N_FEATURES),
        feature_std=np.ones(N_FEATURES),
        regime_mean=np.zeros(2), regime_std=np.ones(2),
        manifest_sha256="synthetic-fixture",
        n_seconds=96, n_chunks=2)


def _tiny_config(**over):
    cfg = {
        "run_id": "p10-test",
        "train_epochs": 1,
        "batch": 16,
        "gen_seeds": (0,),
        "disc_epochs": 1,
        "tstr_epochs": 5,
        "n_boot": 5,
        "block_lengths": (1,),
        "ar_hidden": 8,
        "ar_components": 2,
        "ar_lr": 1e-3,
        "flow_nfe": 4,
        "flow_ch": 8,
        "ddpm_ch": 8,
        "ddpm_t_steps": 5,
        "vae_epochs": 1,
        "vae_lr": 1e-3,
        "methods": ("ddpm", "vae", "chunk_copy", "autoregressive",
                    "conditional_diffusion_forecast", "flow_matching"),
    }
    cfg.update(over)
    return cfg


# ------------------------------------------- autoregressive: known answers

def _oracle_mixture_nll(x, logits, means, log_scales):
    """Mixture NLL straight from the definition, no module code."""
    x = np.asarray(x, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    scales = np.exp(np.asarray(log_scales, dtype=np.float64))
    # log N(x; mu, sigma) per component, summed over features
    z = (x[:, None, :] - means) / scales
    log_comp = (-0.5 * (z ** 2) - np.log(scales)
                - 0.5 * np.log(2 * np.pi)).sum(axis=-1)
    # logsumexp over components of (log pi + log N)
    mx = logits.max(axis=-1, keepdims=True)
    log_pi = logits - mx - np.log(np.exp(logits - mx).sum(
        axis=-1, keepdims=True))
    a = log_pi + log_comp
    m = a.max(axis=-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(a - m).sum(axis=-1))
    return float(-lse.mean())


def test_stage3_flowforge_10_part_2_ar_mixture_nll_matches_oracle():
    from flowforge.autoregressive import mixture_nll
    rng = np.random.default_rng(2)
    b, k = 3, 2
    x = rng.standard_normal((b, N_FEATURES)).astype(np.float32)
    logits = rng.standard_normal((b, k)).astype(np.float32)
    means = rng.standard_normal((b, k, N_FEATURES)).astype(np.float32)
    log_scales = rng.standard_normal((b, k, N_FEATURES)).astype(
        np.float32) * 0.2
    got = mixture_nll(torch.from_numpy(x), torch.from_numpy(logits),
                      torch.from_numpy(means),
                      torch.from_numpy(log_scales))
    assert abs(float(got) - _oracle_mixture_nll(
        x, logits, means, log_scales)) < 1e-5
    # known answer: one component, unit scale, mean zero. The NLL
    # is the joint density over the seven features, so the
    # 0.5 * log(2 pi) term is charged once per feature:
    # 0.5 * sum over features of mean(x^2) + 7 * 0.5 * log(2 pi)
    logits1 = np.zeros((b, 1), dtype=np.float32)
    means1 = np.zeros((b, 1, N_FEATURES), dtype=np.float32)
    scales1 = np.zeros((b, 1, N_FEATURES), dtype=np.float32)
    got1 = mixture_nll(torch.from_numpy(x), torch.from_numpy(logits1),
                       torch.from_numpy(means1),
                       torch.from_numpy(scales1))
    expected = float(0.5 * np.sum(np.mean(x ** 2, axis=0))
                     + 0.5 * N_FEATURES * np.log(2 * np.pi))
    assert abs(float(got1) - expected) < 1e-5


def test_stage3_flowforge_10_part_2_ar_is_causal():
    from flowforge.autoregressive import CausalGRUMixture
    torch.manual_seed(0)
    model = CausalGRUMixture(hidden=8, n_components=2)
    model.eval()
    x, regime, _, _ = _synthetic_windows(n=4, seed=3)
    with torch.no_grad():
        out = model(torch.from_numpy(x), torch.from_numpy(regime))
        x_pert = x.copy()
        x_pert[0, 30, :] += 5.0
        out_pert = model(torch.from_numpy(x_pert),
                         torch.from_numpy(regime))
    # step t predicts x_t from x_{<t}: perturbing x_30 leaves every
    # prediction at steps 0..30 bit-identical and moves step 31
    for name, a, b in zip(("logits", "means", "log_scales"),
                          out, out_pert):
        assert torch.equal(a[:, :31], b[:, :31]), name
        assert not torch.equal(a[:, 31:], b[:, 31:]), name


def test_stage3_flowforge_10_part_2_ar_generate_deterministic():
    from flowforge.autoregressive import CausalGRUMixture
    torch.manual_seed(0)
    model = CausalGRUMixture(hidden=8, n_components=2)
    model.eval()
    _, regime, _, _ = _synthetic_windows(n=6, seed=3)
    r = torch.from_numpy(regime)
    first = model.generate(r, seed=100)
    second = model.generate(r, seed=100)
    assert torch.equal(first, second)
    other = model.generate(r, seed=200)
    assert not torch.equal(first, other)
    assert first.shape == (6, WINDOW, N_FEATURES)
    assert bool(torch.isfinite(first).all())
    # a different regime row must change the draw stream only through
    # conditioning: same seed, different regime gives different windows
    r2 = r.clone()
    r2[0] += 1.0
    third = model.generate(r2, seed=100)
    assert not torch.equal(first, third)


def test_stage3_flowforge_10_part_2_ar_trains_and_counts_params():
    from flowforge.autoregressive import (
        CausalGRUMixture, train_autoregressive)
    from flowforge.models import count_params
    x, regime, _, _ = _synthetic_windows(n=48, seed=7)
    model = CausalGRUMixture(hidden=8, n_components=2,
                             run_id="p10-test:ar")
    hist = train_autoregressive(model, torch.from_numpy(x),
                                torch.from_numpy(regime), epochs=2,
                                batch=16, run_id="p10-test:ar")
    assert len(hist) == 2
    assert all(np.isfinite(hist))
    # hidden 8, 2 components: GRU 3*(8*9 + 8*8 + 16) = 456, head
    # 8*30 + 30 = 270, total 726
    assert count_params(model) == 726
    # the shipped size stays near the DDPM envelope: hidden 256 with
    # 4 components is within 5 percent of the 221,767-parameter base
    big = CausalGRUMixture(hidden=256, n_components=4)
    n_big = count_params(big)
    assert abs(n_big - 221767) / 221767 < 0.05, n_big


# ------------------------------------------- flow matching: known answers

def test_stage3_flowforge_10_part_2_flow_loss_matches_oracle():
    from flowforge.flow_matching import rectified_flow_loss
    rng = np.random.default_rng(4)
    pred = rng.standard_normal((2, N_FEATURES, WINDOW)).astype(np.float32)
    x1 = rng.standard_normal((2, N_FEATURES, WINDOW)).astype(np.float32)
    x0 = rng.standard_normal((2, N_FEATURES, WINDOW)).astype(np.float32)
    got = rectified_flow_loss(torch.from_numpy(pred),
                              torch.from_numpy(x1),
                              torch.from_numpy(x0))
    expected = float(np.mean((pred - (x1 - x0)) ** 2))
    assert abs(float(got) - expected) < 1e-6
    # a perfect velocity field scores exactly zero
    perfect = rectified_flow_loss(torch.from_numpy(x1 - x0),
                                  torch.from_numpy(x1),
                                  torch.from_numpy(x0))
    assert abs(float(perfect)) < 1e-12


def test_stage3_flowforge_10_part_2_flow_sample_deterministic():
    from flowforge.flow_matching import RectifiedFlow
    torch.manual_seed(0)
    model = RectifiedFlow(ch=8, nfe=4, run_id="p10-test:flow")
    model.eval()
    _, regime, _, _ = _synthetic_windows(n=6, seed=3)
    r = torch.from_numpy(regime)
    first = model.sample(r, seed=100)
    second = model.sample(r, seed=100)
    assert torch.equal(first, second)
    other = model.sample(r, seed=200)
    assert not torch.equal(first, other)
    assert first.shape == (6, WINDOW, N_FEATURES)
    assert bool(torch.isfinite(first).all())
    # more Euler steps is a different draw path, same shape
    finer = model.sample(r, seed=100, nfe=8)
    assert finer.shape == first.shape
    assert not torch.equal(first, finer)


def test_stage3_flowforge_10_part_2_flow_trains():
    from flowforge.flow_matching import RectifiedFlow, train_flow
    from flowforge.models import count_params
    x, regime, _, _ = _synthetic_windows(n=48, seed=7)
    model = RectifiedFlow(ch=8, run_id="p10-test:flow")
    hist = train_flow(model, torch.from_numpy(x),
                      torch.from_numpy(regime), epochs=2, batch=16,
                      run_id="p10-test:flow")
    assert len(hist) == 2
    assert all(np.isfinite(hist))
    # the flow field reuses the DDPM denoiser backbone, so at ch=96
    # it carries the same parameter count as the base DDPM denoiser
    base = RectifiedFlow(ch=96)
    from flowforge.models import BookDDPM
    ddpm = BookDDPM(ch=96, n_blocks=3)
    assert count_params(base) == count_params(ddpm)


# ------------------------------------------- conditional diffusion variant

def test_stage3_flowforge_10_part_2_history_ddpm_needs_past_context():
    from flowforge.models import BookDDPM, train_ddpm
    x, regime, past, mask = _synthetic_windows(n=48, seed=7)
    xt = torch.from_numpy(x)
    rt = torch.from_numpy(regime)
    pt = torch.from_numpy(past)
    mt = torch.from_numpy(mask)
    model = BookDDPM(t_steps=5, ch=8, use_history=True,
                     run_id="p10-test:hist")
    hist, _ = train_ddpm(model, xt, rt, epochs=1, batch=16,
                         past_context=pt, past_mask=mt,
                         run_id="p10-test:hist")
    assert len(hist) == 1 and np.isfinite(hist[0])
    model.eval()
    gen = model.sample(rt[:4], seed=0, past_context=pt[:4],
                       past_mask=mt[:4])
    assert gen.shape == (4, WINDOW, N_FEATURES)
    assert bool(torch.isfinite(gen).all())
    # sampling a history-conditioned model without its past context
    # is refused instead of silently falling back to regime only
    with pytest.raises(ValueError, match="past_context"):
        model.sample(rt[:4], seed=0)


# ------------------------------------------- contract: filled Pareto rows

def test_stage3_flowforge_10_part_2_contract_accepts_filled_rows():
    c = default_contract()
    run = {"method": "autoregressive", "kind": "conditional",
           "seed": 100,
           "gen_val": np.zeros((16, WINDOW, N_FEATURES)),
           "gen_test": np.zeros((16, WINDOW, N_FEATURES)),
           "cost": {"sample_seconds": 1.0,
                    "sample_seconds_per_1k": 10.0,
                    "peak_memory_mb": 100.0, "nfe": 960}}
    c.validate_run(run, n_val=16, n_test=16)
    c.check_seed_independence({
        "ddpm": [0, 100, 200],
        "autoregressive": [0, 100, 200],
        "conditional_diffusion_forecast": [0, 100, 200],
        "flow_matching": [0, 100, 200]})
    rows = c.pareto_table([
        {"method": "ddpm", "status": "baseline", "mean_w1": 0.14,
         "joint_error": 0.04, "lag_acf_error": 0.03,
         "corr_error": 0.05, "tail_outside_rate": 0.02,
         "auc_test": 0.71, "tstr_acc_synth": 0.52,
         "nfe_per_window": 100, "sample_seconds_per_1k": 30.0,
         "peak_memory_mb": 500.0, "params": 221767},
        {"method": "autoregressive", "status": "measured",
         "mean_w1": 0.20, "joint_error": 0.06, "lag_acf_error": 0.05,
         "corr_error": 0.07, "tail_outside_rate": 0.03,
         "auc_test": 0.65, "tstr_acc_synth": 0.50,
         "nfe_per_window": 60, "sample_seconds_per_1k": 5.0,
         "peak_memory_mb": 400.0, "params": 216476},
    ])
    assert rows["rows"][0]["method"] == "ddpm"
    assert rows["rows"][1]["method"] == "autoregressive"
    assert "best" not in str(rows).lower()
    with pytest.raises(ValueError, match="verdict"):
        c.pareto_table([{"method": "autoregressive",
                         "status": "measured", "verdict": "best"}])


# ------------------------------------------- end-to-end mini comparison

def test_stage3_flowforge_10_part_2_mini_comparison_runs():
    from flowforge.compare_families import run_family_comparison
    ds = _tiny_dataset()
    record = run_family_comparison(ds, _tiny_config())
    table = record["pareto_table"]
    methods = [r["method"] for r in table["rows"]]
    for name in ("ddpm", "vae", "chunk_copy", "autoregressive",
                 "conditional_diffusion_forecast", "flow_matching"):
        assert name in methods, name
    by_method = {r["method"]: r for r in table["rows"]}
    for name, row in by_method.items():
        assert row["status"] == "measured", name
        assert row["mean_w1"] is not None and np.isfinite(
            row["mean_w1"]), name
        assert row["auc_test"] is not None, name
        assert row["nfe_per_window"] is not None, name
        # every trained family carries parameters; the chunk-copy
        # reference is nonparametric, so its count is exactly zero
        assert row["params"] is not None and row["params"] >= 0, name
        if name != "chunk_copy":
            assert row["params"] > 0, name
        else:
            assert row["params"] == 0, name
    # the table reports metrics only: no ranking language anywhere
    blob = str(record).lower()
    for banned in ("verdict", "winner", "superior"):
        assert banned not in blob, banned
    # every family generated under the same single seed in this run
    assert record["comparison"]["seeds"] == [0]
    # the autoregressive row records its one-step-per-second cost
    assert by_method["autoregressive"]["nfe_per_window"] == WINDOW
    assert by_method["flow_matching"]["nfe_per_window"] == 4
    assert by_method["ddpm"]["nfe_per_window"] == 5
