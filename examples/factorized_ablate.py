"""Preregistered ablation for the P02 factorized representation.

Question: does generating total depth and imbalance (and deriving
the bid/ask size pair exactly) produce more physically valid book
windows than generating the two sizes independently, without
losing distributional fidelity?

Arms (identical fixture, training seeds, epochs, and sampling
seeds; the DDPM arms share channel width and block count, and the
exact parameter count of every arm is reported):

- A ``raw``: the shipped seven-feature BookDDPM, sampled raw.
- B ``factorized``: FactorizedBookDDPM with the gate untrained
  (gate_weight=0) and the continuous decode. Representation only.
- C ``projection``: arm A's own samples with the post-hoc
  imbalance projection (evaluate.constrain_imbalance) applied.
- D ``factorized_hurdle``: FactorizedBookDDPM with the hurdle
  gate trained (gate_weight=1) and the hurdle decode. This is
  the full P02 proposal and the primary comparison arm.
- E ``vae``: the conditional VAE baseline.
- E ``chunk_copy``: the regime-matched chunk-copy baseline.

Preregistered before any run of this harness:

- Validity metrics (lower is better), from
  evaluate.book_consistency_table on the test split:
  ``imbalance_violation_rate``, ``negative_spread_rate``,
  ``count_noninteger_rate``, ``coherence_violation_rate``; the
  validity composite is their mean.
- Fidelity metrics: ``mean_w1`` (marginal fidelity),
  ``flow_acf_mae`` (temporal structure), ``discriminator_auc``
  (held-out, fit on val / scored on test; closer to 0.5 is
  better), plus ``ret_acf_mae``, ``rv_w1``, ``cross_corr_err``
  and TSTR accuracy reported without a verdict role.
- Verdict rule: the factorized representation "improves" only
  if arm D's validity composite is strictly lower than arm A's
  AND no fidelity guardrail (mean W1, flow ACF MAE, AUC)
  degrades by more than 0.05 against arm A. Otherwise the
  verdict is "null" and that is the reported result. The test
  split is only scored, never trained on or selected against.

Data caveat: the recorded dataset (data/book_windows.npz) is
not committed, so this harness runs on the same synthetic
fixture as the P06 ablation (AR(1) signed flow, volatility
clustering, return/flow cross-correlation, physically valid
book features). Fixture numbers describe the protocol on the
fixture; they are NOT comparable to the published real-data
numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flowforge.baselines import RegimeMatchedChunkCopyBaseline  # noqa: E402
from flowforge.data import FEATURE_NAMES, WINDOW  # noqa: E402
from flowforge.evaluate import (acf, book_consistency_table,  # noqa: E402
                                constrain_imbalance, discriminator_auc,
                                marginal_table, tstr_flow_direction,
                                wasserstein_1d)
from flowforge.factorized import (FactorizedBookDDPM,  # noqa: E402
                                  audit_factorized_windows,
                                  factorize_windows)
from flowforge.models import (BookDDPM, CondVAE, count_params,  # noqa: E402
                              train_ddpm, train_vae)

GUARDRAIL_TOLERANCE = 0.05
VALIDITY_METRICS = ("imbalance_violation_rate", "negative_spread_rate",
                    "count_noninteger_rate", "coherence_violation_rate")
GUARDRAILS = ("mean_w1", "flow_acf_mae", "discriminator_auc")

J_RET = FEATURE_NAMES.index("log_ret")
J_SPR = FEATURE_NAMES.index("log_spread_bps")
J_BID = FEATURE_NAMES.index("log_bid_sz")
J_ASK = FEATURE_NAMES.index("log_ask_sz")
J_IMB = FEATURE_NAMES.index("imbalance")
J_SF = FEATURE_NAMES.index("signed_flow")
J_CNT = FEATURE_NAMES.index("log_trade_count")


def make_fixture(n, seed):
    """Physically valid windows with known temporal structure.

    Identical generator to the P06 ablation fixture: returns
    carry smooth within-window volatility clustering and a
    dispersed per-window volatility level; signed flow is AR(1)
    with rho 0.6; flow innovations correlate 0.5 with return
    innovations; imbalance is derived from the sizes; signed
    flow is zero wherever the trade count is zero.
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
    tstr = tstr_flow_direction(real_val, gen_val, real_test, seed=0)
    out = {
        "flow_acf_mae": _acf_mae(real_test, gen_test, J_SF),
        "ret_acf_mae": _acf_mae(real_test, gen_test, J_RET),
        "rv_w1": wasserstein_1d(rv_real, rv_gen),
        "cross_corr_err": abs(_window_corr(real_test, J_RET, J_SF)
                              - _window_corr(gen_test, J_RET, J_SF)),
        "mean_w1": marginal_table(real_test, gen_test)["mean_w1"],
        "discriminator_auc": float(disc["auc_test"]),
        "tstr_acc_synth": float(tstr["acc_synth"]),
    }
    for k in VALIDITY_METRICS:
        out[k] = float(audit[k])
    out["validity_composite"] = float(np.mean(
        [out[k] for k in VALIDITY_METRICS]))
    return out


def run_ablation(seeds=(7, 107), epochs=30, n_train=240, n_eval=120,
                 ch=48, n_blocks=2, t_steps=100):
    """Train and score all arms on the fixture.

    Identical data, training seeds, epochs and sampling seeds
    across the learned arms; factorized normalizers are fit on
    the training windows only.
    """
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

    fact_train, fact_mean, fact_std = factorize_windows(
        std(train_phys), f_mean, f_std)
    x_fact = torch.from_numpy(fact_train)

    arms: dict[str, list] = {}
    params: dict[str, int] = {}
    latency: dict[str, list] = {}
    raw_audits: dict[str, list] = {}

    def record(arm, scores, n_params, seconds, n_windows, raw=None):
        arms.setdefault(arm, []).append(scores)
        params[arm] = int(n_params)
        latency.setdefault(arm, []).append(seconds / n_windows)
        if raw is not None:
            raw_audits.setdefault(arm, []).append(raw)

    for seed in seeds:
        seed = int(seed)
        # A: raw seven-feature DDPM.
        raw_model = BookDDPM(n_features=len(FEATURE_NAMES),
                             window=WINDOW, t_steps=t_steps, ch=ch,
                             n_blocks=n_blocks, seed=seed)
        train_ddpm(raw_model, x_train, r_train, epochs=epochs,
                   seed=seed, verbose=False)
        t0 = time.perf_counter()
        gen_val_a = raw_model.sample(r_val, seed=999).numpy()
        gen_test_a = raw_model.sample(r_test, seed=999).numpy()
        dt = time.perf_counter() - t0
        record("raw", _score_arm(real_test, gen_test_a, real_val,
                                 gen_val_a, f_mean, f_std),
               count_params(raw_model), dt, 2 * n_eval)
        # C: post-hoc projection of arm A's own samples.
        t0 = time.perf_counter()
        gen_val_c = constrain_imbalance(gen_val_a, f_mean, f_std)
        gen_test_c = constrain_imbalance(gen_test_a, f_mean, f_std)
        dt = time.perf_counter() - t0
        record("projection", _score_arm(real_test, gen_test_c,
                                        real_val, gen_val_c,
                                        f_mean, f_std),
               count_params(raw_model), dt, 2 * n_eval)
        # B and D: factorized DDPMs, continuous vs hurdle decode.
        for arm, gw, hurdle in (("factorized", 0.0, False),
                                ("factorized_hurdle", 1.0, True)):
            model = FactorizedBookDDPM(
                fact_mean, fact_std, f_mean, f_std, gate_weight=gw,
                ch=ch, n_blocks=n_blocks, t_steps=t_steps, seed=seed)
            train_ddpm(model, x_fact, r_train, epochs=epochs,
                       seed=seed, verbose=False)
            t0 = time.perf_counter()
            fact_val = model.sample(r_val, seed=999)
            fact_test = model.sample(r_test, seed=999)
            gen_val = model.decode(fact_val, r_val,
                                   hurdle=hurdle).numpy()
            gen_test = model.decode(fact_test, r_test,
                                    hurdle=hurdle).numpy()
            dt = time.perf_counter() - t0
            raw_audit = audit_factorized_windows(
                fact_test.numpy(), fact_mean, fact_std)
            record(arm, _score_arm(real_test, gen_test, real_val,
                                   gen_val, f_mean, f_std),
                   count_params(model), dt, 2 * n_eval, raw=raw_audit)
        # E: conditional VAE.
        vae = CondVAE(seed=seed)
        train_vae(vae, x_train, r_train, epochs=epochs, seed=seed,
                  verbose=False)
        t0 = time.perf_counter()
        gen_val_e = vae.generate(r_val, seed=999).numpy()
        gen_test_e = vae.generate(r_test, seed=999).numpy()
        dt = time.perf_counter() - t0
        record("vae", _score_arm(real_test, gen_test_e, real_val,
                                 gen_val_e, f_mean, f_std),
               count_params(vae), dt, 2 * n_eval)

    # E: chunk-copy (seed-independent apart from its draw seed).
    copy_base = RegimeMatchedChunkCopyBaseline(
        std(train_phys), train_regime)
    gen_val_cc, _ = copy_base.generate(n_eval, val_regime, seed=999)
    gen_test_cc, _ = copy_base.generate(n_eval, test_regime, seed=999)
    record("chunk_copy", _score_arm(real_test, gen_test_cc, real_val,
                                    gen_val_cc, f_mean, f_std),
           0, 0.0, 2 * n_eval)

    summary = {}
    for arm, per_seed in arms.items():
        entry = {k: float(np.mean([s[k] for s in per_seed]))
                 for k in per_seed[0]}
        entry["per_seed"] = per_seed
        entry["params"] = params[arm]
        entry["sample_seconds_per_window"] = float(
            np.mean(latency[arm]))
        if arm in raw_audits:
            entry["raw_factorized_audit"] = {
                k: float(np.mean([a[k] for a in raw_audits[arm]]))
                for k in raw_audits[arm][0]
                if isinstance(raw_audits[arm][0][k], float)}
        summary[arm] = entry

    a, d = summary["raw"], summary["factorized_hurdle"]
    guardrails_ok = all(
        d[k] <= a[k] + GUARDRAIL_TOLERANCE for k in GUARDRAILS)
    verdict = "improvement" if (d["validity_composite"]
                                < a["validity_composite"]
                                and guardrails_ok) else "null"
    return {
        "fixture_only": True,
        "comparable_to_published_real_data": False,
        "seeds": [int(s) for s in seeds],
        "epochs": int(epochs),
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "arms": summary,
        "primary_comparison": {
            "raw_validity_composite": a["validity_composite"],
            "factorized_hurdle_validity_composite":
                d["validity_composite"],
        },
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
