"""Preregistered ablation for the P06 auxiliary losses.

Question: do the statistical-structure auxiliary losses
(flowforge/losses.py) improve the DDPM's joint and temporal
structure, without degenerating its unconditional distributions,
physical validity, or held-out discriminator score?

Preregistered before any run of this harness:

- Primary metrics (lower is better; the aux arm is compared on
  these): ``flow_acf_mae`` and ``ret_acf_mae`` (mean absolute error
  of the per-window-averaged autocorrelation at lags 1..5, signed
  flow and log returns), ``rv_w1`` (1-Wasserstein distance between
  the real and generated per-window realized-volatility
  distributions), ``cross_corr_err`` (absolute error of the
  within-window return/flow correlation).
- Guardrail metrics (must not degenerate): ``mean_w1`` (marginal
  fidelity), ``discriminator_auc`` (held-out, fit on val / scored
  on test; closer to 0.5 is better, so an increase is
  degeneration), ``imbalance_violation_rate`` (physical validity).
- Verdict rule: the aux arm "improves" only if the mean of the four
  primary metrics is lower than the plain arm's AND no guardrail
  degrades by more than 0.05 (mean W1 and violation rate up, AUC
  up). Otherwise the verdict is "null" and that is the reported
  result. The final holdout is not used for tuning: weights are the
  predeclared defaults from flowforge.losses, and the fixture's
  test split is only scored, never trained on or selected against.

Data caveat: the recorded dataset (data/book_windows.npz) is not
committed, so this harness runs on a synthetic fixture with known
temporal structure (AR(1) signed flow, volatility clustering,
return/flow cross-correlation, physically valid book features).
Fixture numbers describe the protocol on the fixture; they are NOT
comparable to the published real-data numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flowforge.data import FEATURE_NAMES, WINDOW  # noqa: E402
from flowforge.evaluate import (acf, book_consistency_table,  # noqa: E402
                                discriminator_auc, marginal_table,
                                wasserstein_1d)
from flowforge.losses import DEFAULT_AUX_WEIGHTS  # noqa: E402
from flowforge.models import BookDDPM, train_ddpm  # noqa: E402

PRIMARY_METRICS = ("flow_acf_mae", "ret_acf_mae", "rv_w1", "cross_corr_err")
GUARDRAIL_METRICS = ("mean_w1", "discriminator_auc",
                     "imbalance_violation_rate")
GUARDRAIL_TOLERANCE = 0.05

J_RET = FEATURE_NAMES.index("log_ret")
J_SPR = FEATURE_NAMES.index("log_spread_bps")
J_BID = FEATURE_NAMES.index("log_bid_sz")
J_ASK = FEATURE_NAMES.index("log_ask_sz")
J_IMB = FEATURE_NAMES.index("imbalance")
J_SF = FEATURE_NAMES.index("signed_flow")
J_CNT = FEATURE_NAMES.index("log_trade_count")


def make_fixture(n, seed):
    """Physically valid windows with known temporal structure.

    Returns (physical windows, regimes). Returns carry smooth
    within-window volatility clustering and a dispersed per-window
    volatility level; signed flow is AR(1) with rho 0.6; flow
    innovations correlate 0.5 with return innovations; imbalance is
    derived from the sizes; signed flow is zero wherever the trade
    count is zero.
    """
    rng = np.random.default_rng(seed)
    L = WINDOW
    phys = np.zeros((n, L, len(FEATURE_NAMES)))
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
    t_axis = np.arange(L)[None, :]
    vol_level = np.exp(rng.normal(0, 0.4, size=(n, 1)))
    sigma = 0.01 * vol_level * (1.0 + 0.5 * np.sin(2 * np.pi * t_axis / 20.0 + phase))
    e_ret = rng.normal(size=(n, L))
    e_flow = rng.normal(size=(n, L))
    ret = np.zeros((n, L))
    flow = np.zeros((n, L))
    innov_ret = sigma * e_ret
    innov_flow = 0.5 * e_ret + np.sqrt(0.75) * e_flow
    ret[:, 0] = innov_ret[:, 0]
    flow[:, 0] = innov_flow[:, 0]
    for t in range(1, L):
        ret[:, t] = 0.3 * ret[:, t - 1] + innov_ret[:, t]
        flow[:, t] = 0.6 * flow[:, t - 1] + innov_flow[:, t]
    phys[:, :, J_RET] = ret
    phys[:, :, J_SF] = flow
    phys[:, :, J_SPR] = np.log1p(rng.uniform(0.5, 3.0, size=(n, L)))
    phys[:, :, J_BID] = rng.normal(4.0, 0.3, size=(n, L))
    phys[:, :, J_ASK] = rng.normal(4.0, 0.3, size=(n, L))
    bid = np.exp(phys[:, :, J_BID])
    ask = np.exp(phys[:, :, J_ASK])
    phys[:, :, J_IMB] = (bid - ask) / (bid + ask)
    counts = rng.integers(0, 6, size=(n, L)).astype(np.float64)
    phys[:, :, J_CNT] = np.log1p(counts)
    phys[:, :, J_SF] = np.where(counts > 0, phys[:, :, J_SF], 0.0)
    regime = rng.normal(size=(n, 2)).astype(np.float32)
    return phys, regime


def _acf_mae(real, gen, j, lags=(1, 2, 3, 4, 5)):
    errs = []
    for lag in lags:
        r = np.nanmean([acf(w[:, j], lag) for w in real])
        g = np.nanmean([acf(w[:, j], lag) for w in gen])
        errs.append(abs(float(r) - float(g)))
    return float(np.mean(errs))


def _window_corr(x, j_a, j_b):
    vals = []
    for w in x:
        a, b = w[:, j_a], w[:, j_b]
        if a.std() > 1e-12 and b.std() > 1e-12:
            vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(vals))


def _score_arm(real_test, gen_test, real_val, gen_val, f_mean, f_std):
    rv_real = real_test[:, :, J_RET].std(axis=1)
    rv_gen = gen_test[:, :, J_RET].std(axis=1)
    disc = discriminator_auc(real_val, gen_val, real_test, gen_test,
                             seed=0, epochs=20)
    audit = book_consistency_table(gen_test, f_mean, f_std)
    return {
        "flow_acf_mae": _acf_mae(real_test, gen_test, J_SF),
        "ret_acf_mae": _acf_mae(real_test, gen_test, J_RET),
        "rv_w1": wasserstein_1d(rv_real, rv_gen),
        "cross_corr_err": abs(_window_corr(real_test, J_RET, J_SF)
                              - _window_corr(gen_test, J_RET, J_SF)),
        "mean_w1": marginal_table(real_test, gen_test)["mean_w1"],
        "discriminator_auc": float(disc["auc_test"]),
        "imbalance_violation_rate": float(audit["imbalance_violation_rate"]),
    }


def run_ablation(seeds=(7, 107), epochs=30, n_train=240, n_eval=120,
                 ch=48, n_blocks=2, t_steps=100, weights=None):
    """Train plain vs aux DDPMs on the fixture and score both arms.

    Identical data, model size, training seeds, epochs and sampling
    seeds for both arms; the only difference is the objective.
    Metrics are averaged over the training seeds.
    """
    if weights is None:
        weights = dict(DEFAULT_AUX_WEIGHTS)
    train_phys, train_regime = make_fixture(n_train, seed=101)
    val_phys, val_regime = make_fixture(n_eval, seed=202)
    test_phys, test_regime = make_fixture(n_eval, seed=303)
    f_mean = train_phys.reshape(-1, len(FEATURE_NAMES)).mean(axis=0)
    f_std = train_phys.reshape(-1, len(FEATURE_NAMES)).std(axis=0)

    def std(a):
        return ((a - f_mean) / f_std).astype(np.float32)

    x_train = torch.from_numpy(std(train_phys))
    r_train = torch.from_numpy(train_regime)
    r_val = torch.from_numpy(val_regime)
    r_test = torch.from_numpy(test_regime)
    real_val, real_test = std(val_phys), std(test_phys)

    arms = {}
    for arm, aux in (("plain", None), ("aux", weights)):
        per_seed = []
        for seed in seeds:
            model = BookDDPM(n_features=len(FEATURE_NAMES), window=WINDOW,
                             t_steps=t_steps, ch=ch, n_blocks=n_blocks,
                             seed=int(seed))
            train_ddpm(model, x_train, r_train, epochs=epochs,
                       seed=int(seed), verbose=False, aux_weights=aux)
            gen_val = model.sample(r_val, seed=999).numpy()
            gen_test = model.sample(r_test, seed=999).numpy()
            per_seed.append(_score_arm(real_test, gen_test, real_val,
                                       gen_val, f_mean, f_std))
        arms[arm] = {k: float(np.mean([s[k] for s in per_seed]))
                     for k in per_seed[0]}
        arms[arm]["per_seed"] = per_seed

    primary_plain = float(np.mean([arms["plain"][k] for k in PRIMARY_METRICS]))
    primary_aux = float(np.mean([arms["aux"][k] for k in PRIMARY_METRICS]))
    guardrails_ok = (
        arms["aux"]["mean_w1"] <= arms["plain"]["mean_w1"] + GUARDRAIL_TOLERANCE
        and arms["aux"]["discriminator_auc"]
        <= arms["plain"]["discriminator_auc"] + GUARDRAIL_TOLERANCE
        and arms["aux"]["imbalance_violation_rate"]
        <= arms["plain"]["imbalance_violation_rate"] + GUARDRAIL_TOLERANCE)
    verdict = "improvement" if (primary_aux < primary_plain
                                and guardrails_ok) else "null"
    return {
        "fixture_only": True,
        "comparable_to_published_real_data": False,
        "seeds": [int(s) for s in seeds],
        "epochs": int(epochs),
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "weights": weights,
        "arms": arms,
        "primary_composite": {"plain": primary_plain, "aux": primary_aux},
        "guardrails_within_tolerance": bool(guardrails_ok),
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 107])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = run_ablation(seeds=tuple(args.seeds), epochs=args.epochs)
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
