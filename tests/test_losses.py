"""Stage 3 flowforge P06: statistical-structure auxiliary losses.

P06 scope (issue #40):

- ``flowforge/losses.py`` adds training-only auxiliary penalties for the
  DDPM: within-window lag autocovariance of returns and signed flow, a
  realized-volatility distribution penalty, and a cross-feature
  return/flow dependence penalty. Every penalty compares the
  differentiable denoised estimate of a clean training window against
  that same training window; no validation or test statistic is an
  input anywhere.
- Weights are low and predeclared; the penalties are off by default in
  ``BookDDPM.p_loss`` / ``train_ddpm``.
- A spectral-domain penalty is NOT shipped: the stability assessment
  pinned here shows per-batch frequency estimates on 60-step windows
  are too noisy to be a training target (see losses.SPECTRAL_DECISION).
"""

import importlib.util
import inspect
import os

import numpy as np
import pytest
import torch

from flowforge.data import FEATURE_NAMES
from flowforge.losses import (
    DEFAULT_AUX_WEIGHTS,
    SPECTRAL_DECISION,
    SPECTRAL_LOSS_INCLUDED,
    assess_spectral_stability,
    auxiliary_losses,
    cross_feature_dependence,
    lag_autocovariance,
    lag_autocovariance_loss,
    realized_volatility,
    realized_volatility_loss,
    cross_feature_loss,
    total_auxiliary_loss,
)
from flowforge.models import BookDDPM, train_ddpm

J_RET = FEATURE_NAMES.index("log_ret")
J_SF = FEATURE_NAMES.index("signed_flow")
LAGS = (1, 2, 3, 4, 5)


def _ar1_windows(n, rho, seed, window=60):
    """Standardized windows whose returns/flow are AR(1) with lag-1
    autocorrelation ``rho`` and contemporaneous corr(ret, flow) = 0.5."""
    rng = np.random.default_rng(seed)
    e1 = rng.normal(size=(n, window))
    e2 = rng.normal(size=(n, window))
    ret = np.zeros((n, window))
    flow = np.zeros((n, window))
    ret[:, 0] = e1[:, 0] / np.sqrt(1 - rho ** 2)
    flow[:, 0] = (0.5 * e1[:, 0] + np.sqrt(0.75) * e2[:, 0]) / np.sqrt(1 - rho ** 2)
    for t in range(1, window):
        ret[:, t] = rho * ret[:, t - 1] + e1[:, t]
        flow[:, t] = rho * flow[:, t - 1] + 0.5 * e1[:, t] + np.sqrt(0.75) * e2[:, t]
    x = rng.normal(size=(n, window, len(FEATURE_NAMES))).astype(np.float32)
    x[:, :, J_RET] = ret
    x[:, :, J_SF] = flow
    return torch.from_numpy(x)


def _white_windows(n, seed, window=60):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        rng.normal(size=(n, window, len(FEATURE_NAMES))).astype(np.float32))


# ------------------------------------------------------- zero on identical

def test_stage3_flowforge_06_aux_losses_zero_on_identical_fixtures():
    x = _ar1_windows(24, 0.6, seed=1)
    est = x.clone()
    assert lag_autocovariance_loss(est, x, lags=LAGS).item() == pytest.approx(0.0, abs=1e-12)
    assert realized_volatility_loss(est, x).item() == pytest.approx(0.0, abs=1e-12)
    assert cross_feature_loss(est, x).item() == pytest.approx(0.0, abs=1e-12)
    parts = auxiliary_losses(est, x, lags=LAGS)
    for name, val in parts.items():
        assert val.item() == pytest.approx(0.0, abs=1e-12), name
    total, _ = total_auxiliary_loss(est, x, lags=LAGS)
    assert total.item() == pytest.approx(0.0, abs=1e-12)


def test_stage3_flowforge_06_statistic_known_answers():
    # lag_autocovariance of a constant series is 0; of a perfectly
    # alternating series at lag 1 it is negative.
    x = torch.zeros(4, 60, len(FEATURE_NAMES))
    ac = lag_autocovariance(x, "signed_flow", lags=(1, 2))
    assert torch.allclose(ac, torch.zeros(2), atol=1e-7)
    alt = torch.zeros(4, 60, len(FEATURE_NAMES))
    alt[:, :, J_SF] = torch.tensor([1.0, -1.0] * 30)
    ac1 = lag_autocovariance(alt, "signed_flow", lags=(1,))
    assert ac1.item() < -0.9
    # realized volatility of a constant-return window is 0.
    rv = realized_volatility(x)
    assert torch.allclose(rv, torch.zeros(4), atol=1e-7)
    # cross dependence: flow == returns exactly -> correlation 1.
    y = _white_windows(16, seed=2)
    y[:, :, J_SF] = y[:, :, J_RET]
    dep = cross_feature_dependence(y)
    assert dep[0].item() == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------- known shifts

def test_stage3_flowforge_06_aux_losses_grow_on_known_shifts():
    target = _ar1_windows(32, 0.6, seed=3)
    base = auxiliary_losses(target.clone(), target, lags=LAGS)

    # Destroy the temporal structure only: shuffle time inside each
    # window (marginals are preserved exactly, autocorrelation is not).
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(60, generator=g)
    shuffled = target[:, perm, :]
    assert lag_autocovariance_loss(shuffled, target, lags=LAGS).item() \
        > base["autocov"].item() + 0.05

    # Flatten the realized-vol distribution: rescale every window's
    # returns to the same realized volatility.
    flat = target.clone()
    rv = realized_volatility(flat)
    flat[:, :, J_RET] = flat[:, :, J_RET] / rv.view(-1, 1) \
        * torch.median(rv)
    assert realized_volatility_loss(flat, target).item() \
        > base["realized_vol"].item() + 0.01

    # Break the return/flow link: signed flow replaced by independent
    # noise with matching marginal scale.
    broken = target.clone()
    broken[:, :, J_SF] = _white_windows(32, seed=4)[:, :, J_SF]
    assert cross_feature_loss(broken, target).item() \
        > base["cross"].item() + 0.01

    # White-noise estimate against an autocorrelated target: every
    # component must move off zero.
    white = _white_windows(32, seed=5)
    parts = auxiliary_losses(white, target, lags=LAGS)
    for name, val in parts.items():
        assert val.item() > 1e-3, name


# -------------------------------------------------------------- gradients

def test_stage3_flowforge_06_aux_losses_finite_gradients():
    target = _ar1_windows(16, 0.5, seed=6)
    est = (target * 0.5).clone().requires_grad_(True)
    total, parts = total_auxiliary_loss(est, target, lags=LAGS)
    total.backward()
    assert torch.isfinite(est.grad).all()
    assert est.grad.abs().sum() > 0
    for name, val in parts.items():
        assert torch.isfinite(val), name


def test_stage3_flowforge_06_aux_target_receives_no_gradient():
    # The target side is the training window itself, detached inside the
    # loss: no gradient may flow into it, and the loss API takes no
    # input beyond the estimate, the target batch, lags and weights.
    target = _ar1_windows(8, 0.5, seed=7).requires_grad_(True)
    est = (target.detach() * 0.5).requires_grad_(True)
    total, _ = total_auxiliary_loss(est, target, lags=LAGS)
    total.backward()
    assert target.grad is None
    params = list(inspect.signature(total_auxiliary_loss).parameters)
    assert params == ["x_hat", "x_target", "lags", "weights"]
    params = list(inspect.signature(auxiliary_losses).parameters)
    assert params == ["x_hat", "x_target", "lags"]


# ------------------------------------------------------- predeclared weights

def test_stage3_flowforge_06_predeclared_weights_and_sensitivity():
    assert DEFAULT_AUX_WEIGHTS == {
        "autocov": 0.01, "realized_vol": 0.01, "cross": 0.01}
    target = _ar1_windows(16, 0.6, seed=8)
    est = _white_windows(16, seed=9)
    total1, parts = total_auxiliary_loss(est, target, lags=LAGS)
    doubled = {k: 2 * v for k, v in DEFAULT_AUX_WEIGHTS.items()}
    total2, _ = total_auxiliary_loss(est, target, lags=LAGS,
                                     weights=doubled)
    assert total2.item() == pytest.approx(2 * total1.item(), rel=1e-5)
    zeroed = {k: 0.0 for k in DEFAULT_AUX_WEIGHTS}
    total0, _ = total_auxiliary_loss(est, target, lags=LAGS,
                                     weights=zeroed)
    assert total0.item() == pytest.approx(0.0, abs=1e-12)
    manual = sum(DEFAULT_AUX_WEIGHTS[k] * parts[k]
                 for k in DEFAULT_AUX_WEIGHTS)
    assert total1.item() == pytest.approx(manual.item(), rel=1e-6)
    with pytest.raises(ValueError):
        total_auxiliary_loss(est, target, weights={"bogus": 1.0})


# --------------------------------------------------------- spectral decision

def test_stage3_flowforge_06_spectral_loss_not_shipped_unstable_estimates():
    assert SPECTRAL_LOSS_INCLUDED is False
    assert "60" in SPECTRAL_DECISION
    rng = np.random.default_rng(10)
    # Independent windows are the optimistic case (training windows
    # overlap). Even then, a 32-window batch's mean periodogram is not
    # a stable target: split halves disagree by well over 15 percent.
    e = rng.normal(size=(32, 60))
    ar = np.zeros_like(e)
    ar[:, 0] = e[:, 0]
    for t in range(1, 60):
        ar[:, t] = 0.5 * ar[:, t - 1] + e[:, t]
    rep = assess_spectral_stability(ar, seed=0)
    assert rep["n_windows"] == 32
    assert rep["n_bins"] == 30
    assert rep["mean_rel_diff"] > 0.15
    assert np.isfinite(rep["split_half_corr"])


# ------------------------------------------------------------ DDPM wiring

def _tiny_ddpm(seed=3):
    return BookDDPM(n_features=7, window=60, t_steps=10, ch=16,
                    n_blocks=1, seed=seed)


def _regime(n, seed=6):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.normal(size=(n, 2)).astype(np.float32))


def test_stage3_flowforge_06_p_loss_default_unchanged_and_aux_finite():
    model = _tiny_ddpm()
    x = _ar1_windows(8, 0.4, seed=11).permute(0, 2, 1).contiguous()
    regime = _regime(8)
    torch.manual_seed(123)
    plain = model.p_loss(x, regime)
    torch.manual_seed(123)
    zeroed = model.p_loss(x, regime, aux_weights={
        "autocov": 0.0, "realized_vol": 0.0, "cross": 0.0})
    assert zeroed.item() == pytest.approx(plain.item(), rel=1e-6)
    torch.manual_seed(123)
    with_aux = model.p_loss(x, regime, aux_weights=dict(DEFAULT_AUX_WEIGHTS))
    assert torch.isfinite(with_aux)
    assert with_aux.item() >= plain.item() - 1e-6
    comps = model.last_aux_components
    assert set(comps) == {"autocov", "realized_vol", "cross"}
    assert all(np.isfinite(v) for v in comps.values())
    with_aux.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_stage3_flowforge_06_train_ddpm_with_aux_toy_run():
    model = _tiny_ddpm(seed=4)
    x = _ar1_windows(32, 0.5, seed=12)
    regime = _regime(32, seed=13)
    hist, _ = train_ddpm(model, x, regime, epochs=2, seed=4,
                         verbose=False,
                         aux_weights=dict(DEFAULT_AUX_WEIGHTS))
    assert len(hist) == 2 and np.isfinite(hist).all()
    hist_plain, _ = train_ddpm(_tiny_ddpm(seed=4), x, regime, epochs=2,
                               seed=4, verbose=False)
    assert len(hist_plain) == 2 and np.isfinite(hist_plain).all()


# ------------------------------------------------------- ablation harness

def _load_ablate():
    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "aux_loss_ablate.py")
    spec = importlib.util.spec_from_file_location("aux_loss_ablate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage3_flowforge_06_ablation_harness_smoke():
    mod = _load_ablate()
    # The preregistered metric set is declared in the harness itself,
    # before any result exists.
    assert mod.PRIMARY_METRICS == (
        "flow_acf_mae", "ret_acf_mae", "rv_w1", "cross_corr_err")
    assert mod.GUARDRAIL_METRICS == (
        "mean_w1", "discriminator_auc", "imbalance_violation_rate")
    out = mod.run_ablation(seeds=(7,), epochs=2, n_train=48, n_eval=24,
                           ch=16, n_blocks=1, t_steps=10)
    assert set(out["arms"]) == {"plain", "aux"}
    for arm in out["arms"].values():
        for name in mod.PRIMARY_METRICS + mod.GUARDRAIL_METRICS:
            assert np.isfinite(arm[name]), (arm, name)
    assert out["fixture_only"] is True
