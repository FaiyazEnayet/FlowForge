"""Stage 3 P05: diffusion numerical/training optimization.

Fixture-first tests for the validated schedule family, prediction
objectives, EMA warmup, learning-rate schedule, gradient clipping,
mixed precision gating, and the device/dtype-safe time embedding.
Every numeric expectation below is a hand-computed known answer or an
independent NumPy oracle; none of it shares code with the module
under test.
"""

import importlib.util
import math
import os

import numpy as np
import torch

from flowforge.models import (BookDDPM, EMA, TimeEmbed, count_params,
                              set_seed, train_ddpm)
from flowforge.data import N_FEATURES, WINDOW


def _batch(n=8, seed=0):
    set_seed(seed)
    x = torch.randn(n, WINDOW, N_FEATURES)
    r = torch.randn(n, 2)
    return x, r


# ------------------------------------------- schedule family

def test_stage3_flowforge_05_linear_schedule_matches_shipped():
    from flowforge.models import build_schedule
    betas = build_schedule("linear", 100)
    expected = torch.linspace(1e-4, 0.08, 100)
    assert torch.equal(betas, expected)
    # independent NumPy oracle for the terminal cumulative noise
    b = np.linspace(1e-4, 0.08, 100)
    alpha_bar_t = float(np.cumprod(1.0 - b)[-1])
    acp = torch.cumprod(1.0 - betas, dim=0)
    assert abs(float(acp[-1]) - alpha_bar_t) < 1e-6
    assert alpha_bar_t < 0.05
    # the shipped default model still builds on exactly this schedule
    m = BookDDPM(t_steps=100)
    assert torch.equal(m.betas, expected)
    assert m.schedule == "linear"
    assert m.objective == "epsilon"


def test_stage3_flowforge_05_schedule_family_validated():
    from flowforge.models import build_schedule
    for kind, kw in (("linear", {}), ("sqrt", {}),
                     ("quadratic", {"beta_end": 0.10}),
                     ("sigmoid", {})):
        betas = build_schedule(kind, 100, **kw)
        b = betas.numpy().astype(np.float64)
        assert np.isfinite(b).all()
        assert ((b > 0) & (b < 1)).all()
        acp = np.cumprod(1.0 - b)
        # cumulative noise strictly increases: log-SNR strictly falls
        assert (np.diff(acp) < 0).all()
        assert acp[-1] < 0.05
        log_snr = np.log(acp / (1.0 - acp))
        assert (np.diff(log_snr) < 0).all()
        # every schedule reaches the same terminal-noise gate the
        # shipped model asserts: the reverse chain starts from noise
        m = BookDDPM(t_steps=100, schedule=kind, run_id="p05-sched",
                     **kw)
        assert float(m.alphas_cumprod[-1]) < 0.05
        assert abs(m.terminal_snr
                   - float(acp[-1] / (1.0 - acp[-1]))) < 1e-6
    # the front-loaded schedule really is front-loaded, same endpoint
    assert abs(float(build_schedule("sqrt", 100)[-1]) - 0.08) < 1e-6
    assert float(build_schedule("sqrt", 100)[1]) > \
        float(build_schedule("linear", 100)[1])
    # the gate refuses schedules that do not reach noise: quadratic
    # at the shipped endpoint (terminal cumulative noise 0.064) and
    # a linear schedule ending at 0.02 (terminal 0.36)
    for kind, kw in (("quadratic", {}), ("linear", {"beta_end": 0.02})):
        try:
            build_schedule(kind, 100, **kw)
        except ValueError as exc:
            assert "0.05" in str(exc)
        else:
            raise AssertionError(
                f"{kind} {kw} must fail the terminal-noise gate")


def test_stage3_flowforge_05_schedule_rejects_cosine_and_unknown():
    from flowforge.models import build_schedule
    for kind in ("cosine", "exp", "constant"):
        try:
            build_schedule(kind, 100)
        except ValueError as exc:
            assert "cosine" in str(exc).lower() or kind != "cosine"
        else:
            raise AssertionError(f"schedule {kind!r} must be rejected")
    try:
        BookDDPM(schedule="cosine")
    except ValueError:
        pass
    else:
        raise AssertionError("cosine schedule must be rejected by the model")
    try:
        BookDDPM(objective="score")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown objective must be rejected")


# ------------------------------------------- prediction objectives

def _oracle_conversions():
    """Hand-built fixture: true x0 and noise, one timestep per row."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, N_FEATURES, WINDOW)).astype(np.float64)
    noise = rng.standard_normal((4, N_FEATURES, WINDOW)).astype(np.float64)
    t = np.array([0, 10, 50, 99])
    b = np.linspace(1e-4, 0.08, 100)
    acp = np.cumprod(1.0 - b)
    s = np.sqrt(acp[t])[:, None, None]
    s1 = np.sqrt(1.0 - acp[t])[:, None, None]
    xt = s * x0 + s1 * noise
    # the network prediction each objective would make if perfect
    pred_eps = noise
    pred_v = s * noise - s1 * x0
    pred_x0 = x0
    return xt, t, acp, (pred_eps, pred_v, pred_x0), (x0, noise)


def test_stage3_flowforge_05_objective_conversions_known_answers():
    from flowforge.models import predict_eps_from, predict_x0_from
    xt, t, acp, preds, (x0, noise) = _oracle_conversions()
    xt_t = torch.from_numpy(xt).float()
    t_t = torch.from_numpy(t)
    sacp = torch.from_numpy(np.sqrt(acp)).float()
    s1acp = torch.from_numpy(np.sqrt(1.0 - acp)).float()
    for obj, pred in zip(("epsilon", "v", "x0"), preds):
        p = torch.from_numpy(pred).float()
        x0_hat = predict_x0_from(p, xt_t, t_t, sacp, s1acp, obj).numpy()
        eps_hat = predict_eps_from(p, xt_t, t_t, sacp, s1acp, obj).numpy()
        assert np.allclose(x0_hat, x0, atol=1e-5), obj
        # the noise recovery divides by sqrt(1 - alpha_bar_t), which
        # is ~0.01 at t = 0, so float32 storage noise in xt is
        # amplified there; 1e-4 still pins the formula tightly
        assert np.allclose(eps_hat, noise, atol=1e-4), obj


def test_stage3_flowforge_05_p_loss_objectives_finite():
    x, r = _batch(n=8)
    xc = x.permute(0, 2, 1).contiguous()
    for obj in ("epsilon", "v", "x0"):
        m = BookDDPM(t_steps=10, objective=obj, run_id="p05-obj")
        loss = m.p_loss(xc, r)
        assert torch.isfinite(loss) and float(loss) > 0
        per = m.p_loss(xc, r, per_example=True)
        assert per.shape == (8,) and bool(torch.isfinite(per).all())


def test_stage3_flowforge_05_sampling_deterministic_per_objective():
    _, r = _batch(n=4)
    for obj in ("epsilon", "v", "x0"):
        m = BookDDPM(t_steps=10, objective=obj, run_id="p05-sample")
        g1 = m.sample(r, seed=3)
        g2 = m.sample(r, seed=3)
        assert torch.equal(g1, g2)
        assert g1.shape == (4, WINDOW, N_FEATURES)
        assert torch.isfinite(g1).all()
    # the objective changes the reverse chain: same weights and seed
    # must not produce identical samples across objectives
    outs = {}
    for obj in ("epsilon", "v", "x0"):
        m = BookDDPM(t_steps=10, objective=obj, run_id="p05-sample")
        outs[obj] = m.sample(r, seed=3)
    assert not torch.equal(outs["epsilon"], outs["v"])
    assert not torch.equal(outs["epsilon"], outs["x0"])


# ------------------------------------------- EMA warmup

def test_stage3_flowforge_05_ema_warmup():
    m = BookDDPM(t_steps=10, run_id="p05-ema")
    plain = EMA(m, decay=0.999)
    warm = EMA(m, decay=0.999, warmup_steps=100)
    assert plain.warmup_steps == 0
    assert warm.warmup_steps == 100
    # one update from identical shadows: warmup uses a smaller
    # effective decay early, so its shadow moves further toward the
    # new weights than the plain shadow does
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1)
    plain.update(m)
    warm.update(m)
    d_plain = sum(float((plain.shadow[n] - p.detach()).abs().sum())
                  for n, p in m.named_parameters() if p.requires_grad)
    d_warm = sum(float((warm.shadow[n] - p.detach()).abs().sum())
                 for n, p in m.named_parameters() if p.requires_grad)
    assert d_warm < d_plain
    # after the warmup window the effective decay is the plain decay
    assert warm.effective_decay(101) == 0.999
    assert warm.effective_decay(1) < 0.999
    # warmup 0 reproduces the old EMA exactly
    m2 = BookDDPM(t_steps=10, run_id="p05-ema")
    e0 = EMA(m2, decay=0.9)
    e1 = EMA(m2, decay=0.9, warmup_steps=0)
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(0.05)
    e0.update(m2)
    e1.update(m2)
    for n in e0.shadow:
        assert torch.equal(e0.shadow[n], e1.shadow[n])


# ------------------------------------------- training regularity

def test_stage3_flowforge_05_cosine_lr_factor_known_answers():
    from flowforge.models import cosine_lr_factor
    assert cosine_lr_factor(0, 100) == 1.0
    assert cosine_lr_factor(100, 100) == 0.0
    assert abs(cosine_lr_factor(50, 100) - 0.5) < 1e-9
    assert abs(cosine_lr_factor(25, 100)
               - 0.5 * (1 + math.cos(math.pi * 0.25))) < 1e-9


def test_stage3_flowforge_05_train_ddpm_regularity_options():
    x, r = _batch(n=32)
    h, ema = train_ddpm(BookDDPM(t_steps=10, run_id="p05-train"),
                         x, r, epochs=2, batch=16, seed=0, verbose=False,
                         lr_schedule="cosine", clip_grad=1.0,
                         ema_decay=0.999, ema_warmup_steps=5)
    assert len(h) == 2 and all(np.isfinite(h))
    assert ema is not None and ema.warmup_steps == 5
    for bad in ("exp", "linear-decay"):
        try:
            train_ddpm(BookDDPM(t_steps=10), x, r, epochs=1,
                       lr_schedule=bad, verbose=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f"lr_schedule {bad!r} must be rejected")


def test_stage3_flowforge_05_clip_grad_enforces_norm():
    from flowforge.models import apply_grad_clip
    m = BookDDPM(t_steps=10, run_id="p05-clip")
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, 1000.0)
    before = apply_grad_clip(m, 1.0)
    assert before > 1.0
    total = math.sqrt(sum(float((p.grad.double() ** 2).sum())
                          for p in m.parameters() if p.grad is not None))
    # clip_grad_norm_ works in float32 over 221k parameters, so the
    # enforced norm is exact only to float32 accumulation (~2e-5)
    assert abs(total - 1.0) < 1e-4
    # and the training loop accepts the option and stays finite
    x, r = _batch(n=32)
    h, _ = train_ddpm(BookDDPM(t_steps=10, run_id="p05-clip"),
                      x, r, epochs=1, batch=16, seed=0, verbose=False,
                      clip_grad=1.0)
    assert len(h) == 1 and np.isfinite(h[0])
    try:
        train_ddpm(BookDDPM(t_steps=10), x, r, epochs=1,
                   clip_grad=0.0, verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive clip_grad must be rejected")


def test_stage3_flowforge_05_amp_requires_cuda():
    x, r = _batch(n=16)
    try:
        train_ddpm(BookDDPM(t_steps=10), x, r, epochs=1, seed=0,
                   verbose=False, amp=True)
    except ValueError as exc:
        assert "cuda" in str(exc).lower()
    else:
        raise AssertionError("AMP on a CPU-only run must raise")


# ------------------------------------------- TimeEmbed device/dtype

def test_stage3_flowforge_05_timeembed_device_dtype():
    emb = TimeEmbed(dim=4)
    # hand-computed frequencies for dim 4: exp(-log(10000) * [0, 1] / 2)
    assert torch.allclose(emb.freqs,
                          torch.tensor([1.0, 0.01], dtype=torch.float32))
    t = torch.zeros(2, dtype=torch.long)
    out = emb(t)
    # t = 0: sin(0) = 0, cos(0) = 1, so the MLP input is [0, 0, 1, 1]
    expected = emb.mlp(torch.tensor([[0.0, 0.0, 1.0, 1.0],
                                     [0.0, 0.0, 1.0, 1.0]]))
    assert torch.allclose(out, expected)
    assert torch.isfinite(out).all()
    # the frequencies follow dtype moves instead of staying float32
    emb64 = TimeEmbed(dim=4).double()
    assert emb64.freqs.dtype == torch.float64
    assert torch.isfinite(emb64(t)).all()
    # the frequencies are a non-persistent buffer: they follow
    # device/dtype moves but stay out of state dicts, so existing
    # checkpoints keep their exact key set
    m = BookDDPM(t_steps=10, run_id="p05-te")
    assert "net.time_embed.freqs" not in m.state_dict()
    assert "freqs" in dict(m.net.time_embed.named_buffers())


# ------------------------------------------- ablation harness smoke

def _load_ablate():
    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "diffusion_ablate.py")
    spec = importlib.util.spec_from_file_location("diffusion_ablate",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage3_flowforge_05_ablation_smoke():
    from flowforge.models import derive_seed, scoped_rng
    ab = _load_ablate()
    with scoped_rng(derive_seed("p05-smoke", "p05:data")):
        splits = {
            name: (torch.randn(n, WINDOW, N_FEATURES),
                   torch.randn(n, 2))
            for name, n in (("train", 48), ("val", 16), ("test", 16))
        }
    record = ab.compare_options(
        splits, run_id="p05-smoke", epochs=1, t_steps=20, verbose=False)
    assert record["run_id"] == "p05-smoke"
    assert set(record["arms"]) == set(ab.ARM_NAMES)
    steps = {a["optimizer_steps"] for a in record["arms"].values()}
    assert len(steps) == 1, "every arm costs the same step count"
    for name, entry in record["arms"].items():
        assert np.isfinite(entry["val_p_loss"])
        assert np.isfinite(entry["val_mean_w1"])
        assert entry["sample_seconds_per_1k"] > 0
        assert entry["nfe_per_window"] == 20
        assert entry["verdict"] in ("improve", "null", "baseline")
    assert record["arms"]["baseline"]["verdict"] == "baseline"
    assert record["baseline"] == "baseline"
    assert record["verdict_rule"].startswith("improve only if")
