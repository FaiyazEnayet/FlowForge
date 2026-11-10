"""Preregistered ablation for the P03 history conditioning.

Question: does conditioning the DDPM on the full 60 seconds of book
features before each window (encoded by a small causal TCN) beat
conditioning on the two-number regime summary of that same past
(realized volatility and mean spread), without losing unconditional
fidelity?

Arms (identical fixture, training seeds, epochs, and sampling
seeds; the DDPM arms share channel width and block count, and the
exact parameter count of every arm is reported):

- A ``regime_only``: the shipped BookDDPM, conditioned on the
  two-number regime summary. The comparator.
- B ``history``: BookDDPM with history conditioning, trained and
  sampled with the true past context of every window
  (teacher-forced context: at generation time the past is known,
  so conditioning on it is legitimate).
- C ``no_history``: the same history-conditioned architecture,
  trained and sampled with the context withheld (zeros with an
  all-unavailable mask). Isolates the value of the context content
  from the value of the extra parameters.
- D ``chunk_copy``: the regime-matched chunk-copy baseline.

The fixture is built so the past genuinely carries information the
two numbers discard: each window's volatility level follows the
context's LATE volatility (last 20 seconds) rather than its
whole-minute average, the signed flow and returns continue across
the context/window boundary as one AR sequence (state
persistence), and the spread and sizes drift smoothly through the
boundary. The regime vector is computed from the context exactly
as the data pipeline computes it (volatility and mean spread over
the lookback), so arm A sees the honest summary of the same past
arm B sees in full.

Preregistered before any run of this harness:

- Primary metrics (lower is better), computed per regime bin
  (bins are the train split's past-volatility terciles, applied to
  the test regimes) and averaged: ``cond_w1_mean`` (conditional
  marginal W1) and ``cond_mmd_mean`` (conditional RBF-kernel MMD
  on flattened windows, one bandwidth from the real test sample).
- Fidelity guardrails: ``mean_w1``, ``flow_acf_mae``,
  ``discriminator_auc`` (held-out, fit on val / scored on test;
  closer to 0.5 is better), tolerance 0.05 against arm A.
  ``ret_acf_mae``, ``rv_w1``, ``cross_corr_err``, TSTR accuracy,
  and the nearest-neighbour ratio ``nn_ratio`` (mean distance of
  generated windows to their nearest train window, divided by the
  same quantity for real test windows; far above 1 suggests the
  generator is not covering the train distribution, near or below
  1 with poor fidelity would suggest copying) are reported without
  a verdict role.
- Verdict rule: history conditioning "improves" only if arm B's
  ``cond_w1_mean`` is strictly lower than arm A's AND arm B's
  ``cond_mmd_mean`` is no higher than arm A's AND no fidelity
  guardrail degrades by more than 0.05 against arm A. Otherwise
  the verdict is "null" and that is the reported result. The test
  split is only scored, never trained on or selected against.

Data caveat: the recorded dataset (data/book_windows.npz) is
not committed, so this harness runs on a synthetic fixture in the
same family as the P06/P02 ablations. Fixture numbers describe
the protocol on the fixture; they are NOT comparable to the
published real-data numbers.
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
from flowforge.data import FEATURE_NAMES, LOOKBACK, WINDOW  # noqa: E402
from flowforge.evaluate import (acf, discriminator_auc,  # noqa: E402
                                marginal_table, tstr_flow_direction,
                                wasserstein_1d)
from flowforge.models import (BookDDPM, count_params, train_ddpm)  # noqa: E402

GUARDRAIL_TOLERANCE = 0.05
GUARDRAILS = ("mean_w1", "flow_acf_mae", "discriminator_auc")

J_RET = FEATURE_NAMES.index("log_ret")
J_SPR = FEATURE_NAMES.index("log_spread_bps")
J_BID = FEATURE_NAMES.index("log_bid_sz")
J_ASK = FEATURE_NAMES.index("log_ask_sz")
J_IMB = FEATURE_NAMES.index("imbalance")
J_SF = FEATURE_NAMES.index("signed_flow")
J_CNT = FEATURE_NAMES.index("log_trade_count")


def make_fixture_with_context(n, seed):
    """Physically valid context+window pairs with persistent state.

    Returns (windows, contexts, regime) in physical units: windows
    and contexts are (n, 60, 7) feature arrays, and regime is the
    (n, 2) [past volatility, past mean spread in bps] summary of
    the context, computed the way the data pipeline computes it.
    """
    rng = np.random.default_rng(seed)
    L = WINDOW
    T = LOOKBACK + L
    phys = np.zeros((n, T, len(FEATURE_NAMES)))
    # volatility levels: early context, late context (last 20s of
    # the lookback), and the window, which follows the LATE level
    lev_e = np.exp(rng.normal(0, 0.4, size=(n, 1)))
    lev_l = np.exp(rng.normal(0, 0.4, size=(n, 1)))
    lev_w = lev_l * np.exp(rng.normal(0, 0.1, size=(n, 1)))
    level = np.concatenate([
        np.repeat(lev_e, LOOKBACK - 20, axis=1),
        np.repeat(lev_l, 20, axis=1),
        np.repeat(lev_w, L, axis=1)], axis=1)
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
    t_axis = np.arange(T)[None, :]
    sigma = 0.01 * level * (1.0 + 0.5 * np.sin(2 * np.pi * t_axis / 20.0 + phase))
    e_ret = rng.normal(size=(n, T))
    e_flow = rng.normal(size=(n, T))
    ret = np.zeros((n, T))
    flow = np.zeros((n, T))
    innov_ret = sigma * e_ret
    innov_flow = 0.5 * e_ret + np.sqrt(0.75) * e_flow
    ret[:, 0] = innov_ret[:, 0]
    flow[:, 0] = innov_flow[:, 0]
    for t in range(1, T):
        ret[:, t] = 0.3 * ret[:, t - 1] + innov_ret[:, t]
        flow[:, t] = 0.6 * flow[:, t - 1] + innov_flow[:, t]
    phys[:, :, J_RET] = ret
    phys[:, :, J_SF] = flow
    spr0 = rng.uniform(0.5, 3.0, size=(n, 1))
    spread_bps = spr0 * (1.0 + 0.3 * (t_axis / T - 0.5))
    phys[:, :, J_SPR] = np.log1p(spread_bps)
    drift_b = rng.normal(0, 0.3, size=(n, 1))
    drift_a = rng.normal(0, 0.3, size=(n, 1))
    phys[:, :, J_BID] = 4.0 + drift_b * (t_axis / T) \
        + rng.normal(0, 0.05, size=(n, T))
    phys[:, :, J_ASK] = 4.0 + drift_a * (t_axis / T) \
        + rng.normal(0, 0.05, size=(n, T))
    bid = np.exp(phys[:, :, J_BID])
    ask = np.exp(phys[:, :, J_ASK])
    phys[:, :, J_IMB] = (bid - ask) / (bid + ask)
    counts = rng.integers(0, 6, size=(n, T)).astype(np.float64)
    phys[:, :, J_CNT] = np.log1p(counts)
    phys[:, :, J_SF] = np.where(counts > 0, phys[:, :, J_SF], 0.0)
    contexts = phys[:, :LOOKBACK]
    windows = phys[:, LOOKBACK:]
    regime = np.stack([
        contexts[:, :, J_RET].std(axis=1),
        np.expm1(contexts[:, :, J_SPR]).mean(axis=1)], axis=1)
    return windows, contexts, regime.astype(np.float64)


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


def _mmd2(a, b, gamma):
    """Biased RBF-kernel MMD^2 between two window sets (flattened)."""
    af = a.reshape(len(a), -1).astype(np.float64)
    bf = b.reshape(len(b), -1).astype(np.float64)

    def k(x, y):
        d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-gamma * d2)

    return float(k(af, af).mean() + k(bf, bf).mean()
                 - 2 * k(af, bf).mean())


def _nn_mean_dist(query, ref):
    qf = query.reshape(len(query), -1).astype(np.float64)
    rf = ref.reshape(len(ref), -1).astype(np.float64)
    d2 = ((qf[:, None, :] - rf[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min(axis=1)).mean())


def _score_arm(real_test, gen_test, real_val, gen_val, train_x,
               test_regime_raw, bin_edges, mmd_gamma):
    rv_real = real_test[:, :, J_RET].std(axis=1)
    rv_gen = gen_test[:, :, J_RET].std(axis=1)
    disc = discriminator_auc(real_val, gen_val, real_test, gen_test,
                             seed=0, epochs=20)
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
    bins = np.digitize(test_regime_raw[:, 0], bin_edges)
    cond_w1, cond_mmd = {}, {}
    for b in sorted(set(bins.tolist())):
        m = bins == b
        if m.sum() < 5:
            continue
        cond_w1[str(int(b))] = float(np.mean([
            wasserstein_1d(real_test[m][:, :, j], gen_test[m][:, :, j])
            for j in range(len(FEATURE_NAMES))]))
        cond_mmd[str(int(b))] = _mmd2(real_test[m], gen_test[m],
                                      mmd_gamma)
    out["cond_w1_by_bin"] = cond_w1
    out["cond_mmd_by_bin"] = cond_mmd
    out["cond_w1_mean"] = float(np.mean(list(cond_w1.values())))
    out["cond_mmd_mean"] = float(np.mean(list(cond_mmd.values())))
    nn_gen = _nn_mean_dist(gen_test, train_x)
    nn_real = _nn_mean_dist(real_test, train_x)
    out["nn_dist_gen"] = nn_gen
    out["nn_dist_real"] = nn_real
    out["nn_ratio"] = float(nn_gen / nn_real) if nn_real > 0 else float("inf")
    return out


def run_ablation(seeds=(7, 107), epochs=30, n_train=240, n_eval=120,
                 ch=48, n_blocks=2, t_steps=100):
    """Train and score all arms on the fixture.

    Identical data, training seeds, epochs and sampling seeds
    across the learned arms; feature and regime normalizers are
    fit on the training split only, and the context is standardized
    with the same train feature statistics as the windows.
    """
    train_phys, train_ctx, train_regime_raw = make_fixture_with_context(
        n_train, seed=101)
    val_phys, val_ctx, val_regime_raw = make_fixture_with_context(
        n_eval, seed=202)
    test_phys, test_ctx, test_regime_raw = make_fixture_with_context(
        n_eval, seed=303)
    f_mean = train_phys.reshape(-1, len(FEATURE_NAMES)).mean(axis=0)
    f_std = train_phys.reshape(-1, len(FEATURE_NAMES)).std(axis=0)
    r_mean = train_regime_raw.mean(axis=0)
    r_std = train_regime_raw.std(axis=0)

    def std(a):
        return ((a - f_mean) / f_std).astype(np.float32)

    def std_r(r):
        return ((r - r_mean) / r_std).astype(np.float32)

    x_train = torch.from_numpy(std(train_phys))
    ctx_train = torch.from_numpy(std(train_ctx))
    ctx_val = torch.from_numpy(std(val_ctx))
    ctx_test = torch.from_numpy(std(test_ctx))
    mask_train = torch.ones(n_train, LOOKBACK, dtype=torch.bool)
    mask_val = torch.ones(n_eval, LOOKBACK, dtype=torch.bool)
    mask_test = torch.ones(n_eval, LOOKBACK, dtype=torch.bool)
    zero_ctx_train = torch.zeros_like(ctx_train)
    zero_mask_train = torch.zeros_like(mask_train)
    zero_ctx_val = torch.zeros_like(ctx_val)
    zero_mask_val = torch.zeros_like(mask_val)
    zero_ctx_test = torch.zeros_like(ctx_test)
    zero_mask_test = torch.zeros_like(mask_test)
    r_train = torch.from_numpy(std_r(train_regime_raw))
    r_val = torch.from_numpy(std_r(val_regime_raw))
    r_test = torch.from_numpy(std_r(test_regime_raw))
    real_val, real_test = std(val_phys), std(test_phys)
    train_np = std(train_phys)

    bin_edges = np.quantile(train_regime_raw[:, 0], [1 / 3, 2 / 3])
    flat = real_test.reshape(len(real_test), -1).astype(np.float64)
    rng = np.random.default_rng(0)
    sub = flat[rng.choice(len(flat), size=min(60, len(flat)),
                          replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2)
    med = float(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
    mmd_gamma = 1.0 / med if med > 0 else 1.0

    arms: dict[str, list] = {}
    params: dict[str, int] = {}
    latency: dict[str, list] = {}

    def record(arm, scores, n_params, seconds, n_windows):
        arms.setdefault(arm, []).append(scores)
        params[arm] = int(n_params)
        latency.setdefault(arm, []).append(seconds / n_windows)

    def score(gen_val, gen_test):
        return _score_arm(real_test, gen_test, real_val, gen_val,
                          train_np, test_regime_raw, bin_edges,
                          mmd_gamma)

    for seed in seeds:
        seed = int(seed)
        # A: regime-only comparator.
        plain = BookDDPM(n_features=len(FEATURE_NAMES), window=WINDOW,
                         t_steps=t_steps, ch=ch, n_blocks=n_blocks,
                         seed=seed)
        train_ddpm(plain, x_train, r_train, epochs=epochs, seed=seed,
                   verbose=False)
        t0 = time.perf_counter()
        gv = plain.sample(r_val, seed=999).numpy()
        gt = plain.sample(r_test, seed=999).numpy()
        record("regime_only", score(gv, gt), count_params(plain),
               time.perf_counter() - t0, 2 * n_eval)
        # B: history conditioning on the true past (teacher-forced).
        hist = BookDDPM(n_features=len(FEATURE_NAMES), window=WINDOW,
                        t_steps=t_steps, ch=ch, n_blocks=n_blocks,
                        seed=seed, use_history=True)
        train_ddpm(hist, x_train, r_train, epochs=epochs, seed=seed,
                   verbose=False, past_context=ctx_train,
                   past_mask=mask_train)
        t0 = time.perf_counter()
        gv = hist.sample(r_val, seed=999, past_context=ctx_val,
                         past_mask=mask_val).numpy()
        gt = hist.sample(r_test, seed=999, past_context=ctx_test,
                         past_mask=mask_test).numpy()
        record("history", score(gv, gt), count_params(hist),
               time.perf_counter() - t0, 2 * n_eval)
        # C: same architecture, context withheld.
        nohist = BookDDPM(n_features=len(FEATURE_NAMES), window=WINDOW,
                          t_steps=t_steps, ch=ch, n_blocks=n_blocks,
                          seed=seed, use_history=True)
        train_ddpm(nohist, x_train, r_train, epochs=epochs, seed=seed,
                   verbose=False, past_context=zero_ctx_train,
                   past_mask=zero_mask_train)
        t0 = time.perf_counter()
        gv = nohist.sample(r_val, seed=999,
                           past_context=zero_ctx_val,
                           past_mask=zero_mask_val).numpy()
        gt = nohist.sample(r_test, seed=999,
                           past_context=zero_ctx_test,
                           past_mask=zero_mask_test).numpy()
        record("no_history", score(gv, gt), count_params(nohist),
               time.perf_counter() - t0, 2 * n_eval)

    # D: chunk-copy (seed-independent apart from its draw seed).
    copy_base = RegimeMatchedChunkCopyBaseline(train_np,
                                               std_r(train_regime_raw))
    gen_val_cc, _ = copy_base.generate(n_eval, std_r(val_regime_raw),
                                       seed=999)
    gen_test_cc, _ = copy_base.generate(n_eval, std_r(test_regime_raw),
                                        seed=999)
    record("chunk_copy", score(gen_val_cc, gen_test_cc), 0, 0.0,
           2 * n_eval)

    summary = {}
    for arm, per_seed in arms.items():
        entry = {k: float(np.mean([s[k] for s in per_seed]))
                 for k in per_seed[0]
                 if isinstance(per_seed[0][k], float)}
        entry["per_seed"] = per_seed
        entry["params"] = params[arm]
        entry["sample_seconds_per_window"] = float(
            np.mean(latency[arm]))
        summary[arm] = entry

    a, b = summary["regime_only"], summary["history"]
    guardrails_ok = all(
        b[k] <= a[k] + GUARDRAIL_TOLERANCE for k in GUARDRAILS)
    verdict = "improvement" if (
        b["cond_w1_mean"] < a["cond_w1_mean"]
        and b["cond_mmd_mean"] <= a["cond_mmd_mean"]
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
            "regime_only_cond_w1_mean": a["cond_w1_mean"],
            "history_cond_w1_mean": b["cond_w1_mean"],
            "regime_only_cond_mmd_mean": a["cond_mmd_mean"],
            "history_cond_mmd_mean": b["cond_mmd_mean"],
            "no_history_cond_w1_mean": summary["no_history"]["cond_w1_mean"],
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
