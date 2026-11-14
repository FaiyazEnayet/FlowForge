"""Stage 3 P08: faster score samplers, the DDIM NFE frontier.

Fixture-first tests for the deterministic/stochastic DDIM sampler in
flowforge/samplers.py. Every numeric expectation below is a
hand-computed known answer or an independent NumPy oracle; none of it
shares code with the module under test. The shipped ancestral
``BookDDPM.sample`` is the regression baseline and is not modified.
"""

import numpy as np
import torch
import torch.nn as nn

from flowforge.data import N_FEATURES, WINDOW
from flowforge.models import BookDDPM, set_seed


def _regime(n=4, seed=0):
    set_seed(seed)
    return torch.randn(n, 2)


def _toy_schedule(t_steps=100):
    """The shipped linear schedule, as an independent NumPy oracle."""
    b = np.linspace(1e-4, 0.08, t_steps, dtype=np.float64)
    return np.cumprod(1.0 - b)


# ------------------------------------------- timestep subsequence

def test_stage3_flowforge_08_ddim_timesteps_boundaries():
    from flowforge.samplers import ddim_timesteps
    steps = ddim_timesteps(100, 10)
    assert len(steps) == 10
    assert int(steps[0]) == 99, "the chain starts at the last timestep"
    assert int(steps[-1]) == 0, "the chain ends at the first timestep"
    assert (np.diff(steps) < 0).all(), "strictly decreasing"
    # evenly spaced: the gaps differ by at most one
    gaps = -np.diff(steps)
    assert gaps.max() - gaps.min() <= 1
    assert list(ddim_timesteps(100, 100)) == list(range(99, -1, -1))
    assert list(ddim_timesteps(100, 1)) == [99]
    assert list(ddim_timesteps(10, 2)) == [9, 0]
    for t_steps, nfe in ((100, 0), (100, -3), (100, 101), (0, 1)):
        try:
            ddim_timesteps(t_steps, nfe)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"ddim_timesteps({t_steps}, {nfe}) must be rejected")


# ------------------------------------------- one DDIM step, toy denoiser

def _oracle_step(x0, noise, t, t_prev, acp, eta, rng_noise=None):
    """Independent NumPy oracle for one DDIM step with a perfect
    denoiser (the network predicts exactly the noise that built x_t).

    x_t = sqrt(acp_t) x0 + sqrt(1 - acp_t) noise, so the clean estimate
    recovers x0 exactly and the deterministic update lands exactly
    on q_sample(x0, t_prev, noise).
    """
    s = np.sqrt(acp[t])
    s1 = np.sqrt(1.0 - acp[t])
    xt = s * x0 + s1 * noise
    x0_hat = (xt - s1 * noise) / s
    assert np.allclose(x0_hat, x0, atol=1e-12)
    if t_prev < 0:
        return x0_hat
    ap = acp[t_prev]
    sigma = eta * np.sqrt((1.0 - ap) / (1.0 - acp[t])) \
        * np.sqrt(1.0 - acp[t] / ap)
    var = 1.0 - ap - sigma ** 2
    assert var >= 0, "DDIM direction coefficient must stay real"
    out = np.sqrt(ap) * x0_hat + np.sqrt(var) * noise
    if sigma > 0:
        assert rng_noise is not None
        out = out + sigma * rng_noise
    return out


def test_stage3_flowforge_08_ddim_step_toy_denoiser_known_answer():
    from flowforge.samplers import ddim_step
    rng = np.random.default_rng(0)
    acp = _toy_schedule()
    acp_t = torch.from_numpy(acp)
    x0 = rng.standard_normal((3, N_FEATURES, WINDOW)).astype(np.float64)
    noise = rng.standard_normal((3, N_FEATURES, WINDOW)).astype(np.float64)
    for t, t_prev in ((99, 89), (50, 40), (10, 0), (1, 0)):
        s = np.sqrt(acp[t])
        s1 = np.sqrt(1.0 - acp[t])
        xt = s * x0 + s1 * noise
        got = ddim_step(torch.from_numpy(xt).double(),
                        torch.from_numpy(noise).double(),
                        t, t_prev, acp_t, eta=0.0).numpy()
        want = _oracle_step(x0, noise, t, t_prev, acp, 0.0)
        assert np.allclose(got, want, atol=1e-10), (t, t_prev)
        # deterministic: the step lands exactly on the forward
        # diffusion of the clean window at the earlier timestep
        q_prev = np.sqrt(acp[t_prev]) * x0 \
            + np.sqrt(1.0 - acp[t_prev]) * noise
        assert np.allclose(got, q_prev, atol=1e-10), (t, t_prev)


def test_stage3_flowforge_08_ddim_step_final_step_recovers_x0():
    from flowforge.samplers import ddim_step
    rng = np.random.default_rng(1)
    acp = _toy_schedule()
    acp_t = torch.from_numpy(acp)
    x0 = rng.standard_normal((2, N_FEATURES, WINDOW)).astype(np.float64)
    noise = rng.standard_normal((2, N_FEATURES, WINDOW)).astype(np.float64)
    t = 50
    xt = np.sqrt(acp[t]) * x0 + np.sqrt(1.0 - acp[t]) * noise
    got = ddim_step(torch.from_numpy(xt).double(),
                    torch.from_numpy(noise).double(),
                    t, -1, acp_t, eta=0.0).numpy()
    assert np.allclose(got, x0, atol=1e-10)


def test_stage3_flowforge_08_ddim_step_stochastic_sigma_known_answer():
    from flowforge.samplers import ddim_step
    from flowforge.models import make_generator
    rng = np.random.default_rng(2)
    acp = _toy_schedule()
    acp_t = torch.from_numpy(acp)
    x0 = rng.standard_normal((2, N_FEATURES, WINDOW)).astype(np.float64)
    noise = rng.standard_normal((2, N_FEATURES, WINDOW)).astype(np.float64)
    t, t_prev, eta = 50, 40, 1.0
    xt = np.sqrt(acp[t]) * x0 + np.sqrt(1.0 - acp[t]) * noise
    # the exact noise draw the step will consume, from an identical
    # generator: the call's first RNG use is this draw
    g_draw = make_generator(7)
    draw = torch.randn(xt.shape, dtype=torch.float64, generator=g_draw)
    g_call = make_generator(7)
    got = ddim_step(torch.from_numpy(xt).double(),
                    torch.from_numpy(noise).double(),
                    t, t_prev, acp_t, eta=eta,
                    generator=g_call).numpy()
    want = _oracle_step(x0, noise, t, t_prev, acp, eta,
                        rng_noise=draw.numpy())
    assert np.allclose(got, want, atol=1e-10)
    # eta 1 is the noisiest stochastic DDIM: it must differ from the
    # deterministic step, and eta 0 must reproduce it exactly
    det = ddim_step(torch.from_numpy(xt).double(),
                    torch.from_numpy(noise).double(),
                    t, t_prev, acp_t, eta=0.0).numpy()
    assert not np.allclose(got, det, atol=1e-6)
    # sigma at eta 1: the jump from t to t_prev spans several schedule
    # steps, so the second factor is 1 - acp[t] / acp[t_prev], not the
    # single-step beta 1 - acp[t] / acp[t - 1]
    sigma = np.sqrt((1.0 - acp[t_prev]) / (1.0 - acp[t])) \
        * np.sqrt(1.0 - acp[t] / acp[t_prev])
    assert sigma > 0
    # exact decomposition of the stochastic step: the difference from
    # the deterministic step is the direction-coefficient shift on
    # the noise estimate plus sigma times the drawn noise
    coef = np.sqrt(1.0 - acp[t_prev] - sigma ** 2) \
        - np.sqrt(1.0 - acp[t_prev])
    draw = torch.randn(xt.shape, dtype=torch.float64,
                       generator=make_generator(7)).numpy()
    assert np.allclose(got - det, coef * noise + sigma * draw,
                       atol=1e-10)
    for bad in (-0.1, 1.5):
        try:
            ddim_step(torch.from_numpy(xt).double(),
                      torch.from_numpy(noise).double(),
                      t, t_prev, acp_t, eta=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"eta {bad} must be rejected")


# ------------------------------------------- full sampler vs ancestral DDPM

def _ancestral_posterior_sampler(model, regime, seed):
    """Independent ancestral DDPM chain with posterior variance.

    Written from the DDPM paper formulas, not from the module under
    test: the reverse mean is (x - beta / sqrt(1 - acp_t) eps) /
    sqrt(1 - beta), and the noise scale is the posterior standard
    deviation sqrt((1 - acp_{t-1}) / (1 - acp_t) * beta_t). The
    shipped ``BookDDPM.sample`` instead adds noise with std
    sqrt(beta_t), so this oracle is deliberately not that sampler:
    DDIM at eta 1 matches the posterior-variance chain, and the two
    ancestral variants differ by that noise scale.
    """
    import math
    from flowforge.models import make_generator, predict_eps_from
    g = make_generator(seed, device=regime.device)
    b = regime.shape[0]
    x = torch.randn(b, model.n_features, model.window,
                    device=regime.device, generator=g)
    for i in reversed(range(model.t_steps)):
        t = torch.full((b,), i, dtype=torch.long)
        pred = model.net(x, t, regime)
        eps = predict_eps_from(pred, x, t, model.sqrt_acp,
                               model.sqrt_1m_acp, model.objective)
        beta = float(model.betas[i])
        ac = float(model.alphas_cumprod[i])
        x = (x - beta / math.sqrt(1.0 - ac) * eps) / math.sqrt(1.0 - beta)
        if i > 0:
            ac_prev = float(model.alphas_cumprod[i - 1])
            beta_tilde = (1.0 - ac_prev) / (1.0 - ac) * beta
            x = x + math.sqrt(beta_tilde) * torch.randn_like(x, generator=g)
    return x.permute(0, 2, 1)


def test_stage3_flowforge_08_ddim_full_step_eta1_matches_ddpm():
    """DDIM with eta 1 over every timestep is the ancestral DDPM
    sampler with posterior variance: same mean, same variance, same
    draw order. The two implementations must agree to float32
    rounding."""
    from flowforge.samplers import sample_ddim
    m = BookDDPM(t_steps=10, run_id="p08-equiv")
    r = _regime(n=4, seed=1)
    ref = _ancestral_posterior_sampler(m, r, seed=3)
    got = sample_ddim(m, r, nfe=10, eta=1.0, seed=3)
    assert got.shape == ref.shape
    assert torch.isfinite(got).all()
    max_diff = float((got - ref).abs().max())
    assert max_diff < 1e-4, f"max diff {max_diff}"
    # and it is not a vacuous match: a different seed differs
    other = sample_ddim(m, r, nfe=10, eta=1.0, seed=4)
    assert not torch.equal(got, other)
    # the shipped ancestral sampler is the other variance variant
    # (noise std sqrt(beta_t) instead of the posterior std), so it
    # genuinely differs from eta-1 DDIM rather than matching it
    shipped = m.sample(r, seed=3)
    assert float((shipped - got).abs().max()) > 0.01


def test_stage3_flowforge_08_ddim_shapes_finite_deterministic():
    from flowforge.samplers import sample_ddim
    m = BookDDPM(t_steps=20, run_id="p08-shape")
    r = _regime(n=6, seed=2)
    for nfe in (1, 5, 10, 20):
        a = sample_ddim(m, r, nfe=nfe, eta=0.0, seed=5)
        b = sample_ddim(m, r, nfe=nfe, eta=0.0, seed=5)
        assert a.shape == (6, WINDOW, N_FEATURES)
        assert torch.isfinite(a).all()
        assert torch.equal(a, b), f"nfe {nfe} must be deterministic"
    # fewer steps change the trajectory, they do not just resample it
    outs = {nfe: sample_ddim(m, r, nfe=nfe, eta=0.0, seed=5)
            for nfe in (5, 10, 20)}
    assert not torch.equal(outs[5], outs[10])
    assert not torch.equal(outs[10], outs[20])
    # batched generation keeps shapes and finiteness
    batched = sample_ddim(m, r, nfe=10, eta=0.0, seed=5, batch_size=2)
    assert batched.shape == (6, WINDOW, N_FEATURES)
    assert torch.isfinite(batched).all()


def test_stage3_flowforge_08_ddim_uses_same_trained_weights():
    """The sampler reads the model's own denoiser: identical weights
    give identical outputs, and perturbing the weights moves them."""
    from flowforge.samplers import sample_ddim
    r = _regime(n=4, seed=3)
    m1 = BookDDPM(t_steps=10, run_id="p08-weights")
    m2 = BookDDPM(t_steps=10, run_id="p08-weights")
    a = sample_ddim(m1, r, nfe=5, eta=0.0, seed=9)
    b = sample_ddim(m2, r, nfe=5, eta=0.0, seed=9)
    assert torch.equal(a, b)
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(0.01)
    c = sample_ddim(m2, r, nfe=5, eta=0.0, seed=9)
    assert not torch.equal(a, c)


class _CountingNet(nn.Module):
    """Wrap a denoiser, counting forward evaluations."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.regime_mlp = net.regime_mlp
        self.calls = 0

    def forward(self, x, t, regime, emb=None):
        self.calls += 1
        if emb is None:
            return self.net(x, t, regime)
        return self.net(x, t, regime, emb)


def test_stage3_flowforge_08_ddim_nfe_is_exact():
    """NFE is the number of denoiser evaluations, counted directly:
    DDIM spends exactly nfe, the ancestral sampler spends t_steps."""
    from flowforge.samplers import sample_ddim
    m = BookDDPM(t_steps=20, run_id="p08-nfe")
    r = _regime(n=4, seed=4)
    counter = _CountingNet(m.net)
    m.net = counter
    for nfe in (5, 10, 20):
        counter.calls = 0
        out = sample_ddim(m, r, nfe=nfe, eta=0.0, seed=6)
        assert torch.isfinite(out).all()
        assert counter.calls == nfe, f"nfe {nfe} made {counter.calls}"
    counter.calls = 0
    out = m.sample(r, seed=6)
    assert torch.isfinite(out).all()
    assert counter.calls == 20, \
        f"ancestral baseline must spend t_steps, spent {counter.calls}"


def test_stage3_flowforge_08_ddim_rejects_bad_inputs():
    from flowforge.samplers import sample_ddim
    m = BookDDPM(t_steps=10, run_id="p08-bad")
    r = _regime(n=4, seed=5)
    bad = [
        lambda: sample_ddim(m, r, nfe=0),
        lambda: sample_ddim(m, r, nfe=11),
        lambda: sample_ddim(m, r, nfe=5, eta=-0.1),
        lambda: sample_ddim(m, r, nfe=5, eta=1.1),
        lambda: sample_ddim(m, torch.randn(4, 3)),
        lambda: sample_ddim(m, torch.full((4, 2), float("nan"))),
        lambda: sample_ddim(m, torch.randn(0, 2)),
        lambda: sample_ddim(m, r, nfe=5, batch_size=0),
        lambda: sample_ddim(m, [[0.0, 0.0]]),
    ]
    for fn in bad:
        try:
            fn()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("bad input must be rejected")


def test_stage3_flowforge_08_ddim_objectives_finite():
    from flowforge.samplers import sample_ddim
    r = _regime(n=4, seed=6)
    for obj in ("epsilon", "v", "x0"):
        m = BookDDPM(t_steps=10, objective=obj, run_id="p08-obj")
        a = sample_ddim(m, r, nfe=5, eta=0.0, seed=3)
        b = sample_ddim(m, r, nfe=5, eta=0.0, seed=3)
        assert a.shape == (4, WINDOW, N_FEATURES)
        assert torch.isfinite(a).all()
        assert torch.equal(a, b), f"objective {obj} must be deterministic"
    # the objective changes the reverse chain, as it does for DDPM
    outs = {}
    for obj in ("epsilon", "v", "x0"):
        m = BookDDPM(t_steps=10, objective=obj, run_id="p08-obj")
        outs[obj] = sample_ddim(m, r, nfe=5, eta=0.0, seed=3)
    assert not torch.equal(outs["epsilon"], outs["v"])
    assert not torch.equal(outs["epsilon"], outs["x0"])


# ------------------------------------------- Pareto selection

def test_stage3_flowforge_08_pareto_frontier_known_answers():
    from flowforge.samplers import pareto_frontier
    pts = [
        {"method": "ddim_10", "val_mean_w1": 0.30,
         "steady_seconds_per_1k": 5.0},
        {"method": "ddim_20", "val_mean_w1": 0.25,
         "steady_seconds_per_1k": 9.0},
        {"method": "ddim_50", "val_mean_w1": 0.24,
         "steady_seconds_per_1k": 20.0},
        {"method": "ddpm", "val_mean_w1": 0.23,
         "steady_seconds_per_1k": 40.0},
        # dominated on both axes by ddim_20: worse W1, slower
        {"method": "slow_bad", "val_mean_w1": 0.26,
         "steady_seconds_per_1k": 12.0},
        # dominated by ddim_10: same speed, worse W1
        {"method": "slow_same", "val_mean_w1": 0.31,
         "steady_seconds_per_1k": 5.0},
    ]
    front = pareto_frontier(pts)
    assert front == ["ddim_10", "ddim_20", "ddim_50", "ddpm"]


# ------------------------------------------- frontier harness smoke

def _load_frontier():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "latency_frontier.py")
    spec = importlib.util.spec_from_file_location("latency_frontier",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage3_flowforge_08_frontier_smoke():
    from flowforge.models import derive_seed, scoped_rng
    fr = _load_frontier()
    with scoped_rng(derive_seed("p08-smoke", "p08:data")):
        splits = {
            name: (torch.randn(n, WINDOW, N_FEATURES),
                   torch.randn(n, 2))
            for name, n in (("train", 48), ("val", 16), ("test", 16))
        }
    meta = {
        "x": splits["train"][0].numpy(),
        "regime": splits["train"][1].numpy(),
        "session_ids": np.zeros(len(splits["train"][0]), dtype=int),
        "segment_ids": np.zeros(len(splits["train"][0]), dtype=int),
    }
    record = fr.compare_samplers(
        splits, train_meta=meta, run_id="p08-smoke", epochs=1,
        t_steps=10, nfe_list=(5, 10), eta=0.0, seeds=(0,),
        verbose=False)
    assert record["run_id"] == "p08-smoke"
    assert set(record["operating_points"]) == {
        "ddpm", "vae", "chunk_copy", "ddim_5", "ddim_10"}
    for name, point in record["operating_points"].items():
        assert np.isfinite(point["val_mean_w1"])
        assert np.isfinite(point["test_mean_w1"])
        assert point["steady_seconds_per_1k"] > 0
        assert point["cold_start_seconds"] > 0
        assert point["batch_size"] == 32
        # chunk-copy has no trainable parameters; the neural points do
        if name == "chunk_copy":
            assert point["params"] == 0
        else:
            assert point["params"] > 0
        assert point["peak_rss_mb"] > 0
        assert point["pareto"] in (True, False)
    ddpm = record["operating_points"]["ddpm"]
    assert ddpm["nfe_per_window"] == 10
    assert record["operating_points"]["ddim_5"]["nfe_per_window"] == 5
    assert record["operating_points"]["ddim_10"]["nfe_per_window"] == 10
    assert record["operating_points"]["vae"]["nfe_per_window"] == 1
    assert record["operating_points"]["chunk_copy"]["nfe_per_window"] == 0
    # the Pareto set is non-empty and every member is genuinely
    # non-dominated under the rule (on this random fixture the
    # chunk-copy point dominates: it returns real windows, so it is
    # fastest and best, and the neural points are all dominated)
    assert len(record["pareto"]) >= 1
    pts = record["operating_points"]
    for name in record["pareto"]:
        assert pts[name]["pareto"] is True
        for other, q in pts.items():
            if other == name:
                continue
            assert not (q["val_mean_w1"] <= pts[name]["val_mean_w1"]
                        and q["steady_seconds_per_1k"]
                        <= pts[name]["steady_seconds_per_1k"]
                        and (q["val_mean_w1"] < pts[name]["val_mean_w1"]
                             or q["steady_seconds_per_1k"]
                             < pts[name]["steady_seconds_per_1k"])), \
                f"{name} is dominated by {other}"
    for name, point in pts.items():
        assert point["pareto"] == (name in record["pareto"])
    assert record["comparison"]["schema"] == "flowforge-baseline-parity/1"
    assert record["pareto_rule"].startswith("Pareto-optimal")
