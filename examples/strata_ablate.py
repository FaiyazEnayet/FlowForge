"""Preregistered ablation for the P07 regime-balanced training.

Question: does training the DDPM with regime-balanced sampling
(capped importance weights, or stratified minibatches) beat uniform
sampling on rare but important market states, without losing
unconditional fidelity?

Arms (identical data, architecture, epochs, batch size, and seeds, so
every arm costs exactly the same number of optimizer steps):

- A ``uniform``: the shipped training loop, uniform minibatches.
  The comparator.
- B ``capped_weights``: the same loop with per-window importance
  weights over the binary tail partition (top past-volatility decile
  vs the rest, inverse frequency, capped at 10, renormalized to mean
  1). Same batches as A, weighted loss. (Inverse-frequency weights
  over the deciles themselves are degenerate: quantile deciles hold
  equal mass by construction, so their weights are all 1 and the ESS
  equals n. The tail partition is where the reweighting bites.)
- C ``stratified``: minibatches built round-robin across the ten
  past-volatility deciles. Same step count per epoch as A.
- D ``chunk_copy``: the regime-matched chunk-copy baseline (no
  training). Reference only, never a contender.

Strata are the past-volatility deciles fitted on TRAIN (thresholds
recorded in the output); val/test never affect them. The evaluation
uses a fixed stratified subsample of the test split (2,000 windows,
shares preserved per decile, seed 1234) and a matching val subsample
for the held-out discriminator, so every arm is scored under
identical conditions.

Preregistered before any run of this harness:

- Primary metrics (lower is better): ``worst_mean_w1`` (worst
  decile's marginal W1) and ``average_coverage`` (mean over deciles
  of the nearest-neighbour coverage ratio) from the stratified
  report on the test subsample.
- Tail metrics: ``tail_w1`` (1-D Wasserstein between real and
  generated 60 s window returns on the real top-5% |return| test
  windows) and ``tail_rate_ratio`` (generated fraction above the
  train top-5% |return| threshold divided by the real fraction).
- Fidelity guardrails: ``mean_w1``, ``flow_acf_mae`` (mean absolute
  ACF error over lags 1..5), and ``disc_auc_dist`` (|AUC - 0.5| on
  test), tolerance 0.05 against arm A.
- Verdict rule: an arm "improves" only if its ``worst_mean_w1`` is
  strictly lower than arm A's AND its ``average_coverage`` is no
  lower than arm A's AND no guardrail degrades by more than 0.05
  against arm A. Otherwise the verdict is "null" and that is the
  reported result. The test split is only scored, never trained on
  or selected against.

Data caveat: the recorded dataset (data/book_windows.npz) is
gitignored, so these numbers describe the protocol on the dataset
whose manifest SHA is recorded in the output, not a committed
artifact. The default run uses the expanded 28-session corpus
(8 original chunks + 20 B1-QC 4 h sessions, Nov 9-12 2026);
pass --data to point at another build. One seed only (0): a
single-seed result is a limitation, stated in the output.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))

from flowforge.baselines import RegimeMatchedChunkCopyBaseline  # noqa: E402
from flowforge.data import load_dataset  # noqa: E402
from flowforge.evaluate import full_report  # noqa: E402
from flowforge.models import BookDDPM, train_ddpm  # noqa: E402
from flowforge.strata import (  # noqa: E402
    DEFAULT_MIN_INDEPENDENT,
    assign_strata,
    capped_importance_weights,
    check_sufficient,
    fit_strata,
    independent_episode_mask,
    outcome_tail_labels,
    past_features,
    stratum_counts,
    window_returns,
)

SEED = 0
EVAL_SEED = 1234
N_EVAL = 2000
N_BINS = 10
WEIGHT_CAP = 10.0
EPOCHS = 12
BATCH = 32


def _w1_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein distance by sorted pairing."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def _stratified_subsample(labels: np.ndarray, n: int, seed: int
                          ) -> np.ndarray:
    """Fixed stratified subsample preserving per-bin shares."""
    rng = np.random.default_rng(seed)
    out = []
    for b in np.unique(labels):
        idx = np.where(labels == b)[0]
        k = max(1, int(round(n * len(idx) / len(labels))))
        out.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


def _arm_metrics(rep: dict, tail_w1: float, tail_rate_ratio: float,
                 params: int, train_seconds: float) -> dict:
    st = rep["strata"]
    acf_rows = rep["flow_acf"]["rows"]
    acf_mae = float(np.mean([abs(r["real_acf"] - r["gen_acf"])
                             for r in acf_rows]))
    return {
        "mean_w1": float(rep["marginals"]["mean_w1"]),
        "flow_acf_mae": acf_mae,
        "disc_auc_test": float(rep["discriminator"]["auc_test"]),
        "disc_auc_dist": float(
            abs(rep["discriminator"]["auc_test"] - 0.5)),
        "tstr_acc_synth": float(rep["tstr"]["acc_synth"]),
        "tstr_acc_real": float(rep["tstr"]["acc_real"]),
        "worst_mean_w1": float(st["worst_mean_w1"]),
        "worst_stratum": int(st["worst_stratum"]),
        "average_mean_w1": float(st["average_mean_w1"]),
        "average_coverage": float(st["average_coverage"]),
        "worst_coverage": float(st["worst_coverage"]),
        "diversity_ratio": float(st["diversity_ratio"]),
        "tail_w1": tail_w1,
        "tail_rate_ratio": tail_rate_ratio,
        "params": int(params),
        "train_seconds": float(train_seconds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        REPO_ROOT, "data", "book_windows.npz"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "strata_ablation_results.json"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    args = ap.parse_args()

    ds = load_dataset(args.data)
    print(f"dataset: {ds.n_chunks} sessions, {ds.n_seconds} s, "
          f"train {ds.train.x.shape[0]} / val {ds.val.x.shape[0]} / "
          f"test {ds.test.x.shape[0]} windows", flush=True)
    print(f"manifest sha: {ds.manifest_sha256}", flush=True)

    # Strata: past-volatility deciles fitted on TRAIN only.
    feats_train = past_features(ds.train, ds)
    spec = fit_strata(feats_train, ("past_vol", "past_spread",
                                    "past_activity", "past_depth"),
                      n_bins=N_BINS,
                      min_independent=DEFAULT_MIN_INDEPENDENT)
    vol_j = 0
    labels_train = assign_strata(feats_train, spec)[:, vol_j]
    labels_val = assign_strata(past_features(ds.val, ds), spec)[:, vol_j]
    labels_test = assign_strata(past_features(ds.test, ds), spec)[:, vol_j]
    # Gate re-verification at runtime: every decile must hold enough
    # independent episodes in every split, or the run stops here.
    for name, labels, split in (
            ("train", labels_train, ds.train),
            ("val", labels_val, ds.val),
            ("test", labels_test, ds.test)):
        mask = independent_episode_mask(split.t_start, split.t_end,
                                        split.session_ids)
        counts = stratum_counts(labels, N_BINS, independent_mask=mask)
        check_sufficient(counts, DEFAULT_MIN_INDEPENDENT, name)
        print(f"gate {name}: min independent per decile "
              f"{min(c['independent'] for c in counts.values())}",
              flush=True)
    edges = [float(v) for v in spec.edges["past_vol"]]
    print(f"train vol decile edges: {[f'{e:.2e}' for e in edges]}",
          flush=True)

    # Weights arm: inverse-frequency weights over the binary tail
    # partition (top past-volatility decile vs the rest), capped at
    # 10. Decile inverse-frequency weights would be degenerate (each
    # decile holds 10% of train by construction, ESS = n), so the
    # tail partition is the reweighting that actually bites.
    tail_labels = (labels_train == N_BINS - 1).astype(np.int64)
    weights, ess = capped_importance_weights(
        tail_labels, 2, cap=WEIGHT_CAP)
    print(f"capped weights (tail vs rest): ESS {ess:.1f} of "
          f"{len(weights)} (cap {WEIGHT_CAP})", flush=True)

    # Fixed evaluation subsamples, stratified by test/val decile.
    idx_test = _stratified_subsample(labels_test, args.n_eval, EVAL_SEED)
    idx_val = _stratified_subsample(labels_val, args.n_eval, EVAL_SEED)
    print(f"eval subsamples: test {len(idx_test)}, val {len(idx_val)}",
          flush=True)
    x_test = torch.from_numpy(ds.test.x[idx_test])
    r_test = torch.from_numpy(ds.test.regime[idx_test])
    x_val = torch.from_numpy(ds.val.x[idx_val])
    r_val = torch.from_numpy(ds.val.regime[idx_val])
    real_test = ds.test.x[idx_test]
    real_val = ds.val.x[idx_val]
    lab_test = labels_test[idx_test]

    # Tail reference: train top-5% |window return| threshold.
    ret_train = window_returns(ds.train, ds)
    ret_test = window_returns(ds.test, ds)[idx_test]
    out_lab = outcome_tail_labels(ret_test, ret_train, tail=0.05)
    tail_sel = out_lab.labels
    thr = float(np.quantile(np.abs(ret_train), 0.95))
    print(f"tail threshold |60s return| >= {thr:.6g}; "
          f"{int(tail_sel.sum())} of {len(ret_test)} eval test windows "
          f"in the tail", flush=True)

    def score_arm(gen_test: np.ndarray, gen_val: np.ndarray,
                  params: int, train_seconds: float) -> dict:
        rep = full_report(real_test, gen_test, real_val, gen_val,
                          seed=args.seed, stratum_labels=lab_test,
                          feature_name="past_vol")
        gen_ret = window_returns_from_gen(gen_test, ds)
        tail_w1 = _w1_1d(ret_test[tail_sel], gen_ret[tail_sel])
        tail_rate = float((np.abs(gen_ret) >= thr).mean())
        real_rate = float((np.abs(ret_test) >= thr).mean())
        return _arm_metrics(
            rep, tail_w1, tail_rate / real_rate if real_rate > 0 else
            float("nan"), params, train_seconds)

    x_train = torch.from_numpy(ds.train.x)
    r_train = torch.from_numpy(ds.train.regime)
    results: dict = {
        "protocol": "P07 strata ablation (see script docstring)",
        "dataset": {
            "path": os.path.abspath(args.data),
            "manifest_sha256": ds.manifest_sha256,
            "n_chunks": int(ds.n_chunks),
            "n_seconds": int(ds.n_seconds),
            "windows": {s: int(getattr(ds, s).x.shape[0])
                        for s in ("train", "val", "test")},
            "session_split": {k: [int(v) for v in vs]
                              for k, vs in ds.session_split.items()},
        },
        "strata": {
            "feature": "past_vol",
            "n_bins": N_BINS,
            "edges": edges,
            "min_independent": DEFAULT_MIN_INDEPENDENT,
            "weight_scheme": "inverse-frequency over binary tail "
                             "partition (top past-vol decile vs rest)",
            "weight_cap": WEIGHT_CAP,
            "ess": float(ess),
        },
        "train": {"epochs": args.epochs, "batch": args.batch,
                  "seed": args.seed, "lr": 2e-4},
        "eval": {"n_eval": int(args.n_eval), "eval_seed": EVAL_SEED,
                 "tail_threshold": thr},
        "arms": {},
    }

    arms = {}
    # Arm A: uniform.
    for name, kw in (("uniform", {}),
                     ("capped_weights", {"weights": weights}),
                     ("stratified", {"stratum_labels": labels_train})):
        model = BookDDPM(seed=args.seed)
        t0 = time.time()
        hist, _ = train_ddpm(model, x_train, r_train,
                             epochs=args.epochs, batch=args.batch,
                             seed=args.seed, verbose=False, **kw)
        train_seconds = time.time() - t0
        params = sum(p.numel() for p in model.parameters())
        print(f"arm {name}: trained in {train_seconds:.0f}s, "
              f"final loss {hist[-1]:.4f}", flush=True)
        gen_test = model.sample(r_test, seed=args.seed,
                                batch_size=512).numpy()
        gen_val = model.sample(r_val, seed=args.seed,
                               batch_size=512).numpy()
        arms[name] = score_arm(gen_test, gen_val, params, train_seconds)
        arms[name]["train_loss_final"] = float(hist[-1])
        print(f"arm {name}: worst W1 {arms[name]['worst_mean_w1']:.3f}, "
              f"avg coverage {arms[name]['average_coverage']:.3f}, "
              f"mean W1 {arms[name]['mean_w1']:.3f}", flush=True)

    # Arm D: regime-matched chunk-copy reference (no training).
    cc = RegimeMatchedChunkCopyBaseline(ds.train.x, ds.train.regime)
    gen_test_cc, _ = cc.generate(len(idx_test), ds.test.regime[idx_test],
                                 seed=args.seed)
    gen_val_cc, _ = cc.generate(len(idx_val), ds.val.regime[idx_val],
                                seed=args.seed)
    arms["chunk_copy"] = score_arm(gen_test_cc, gen_val_cc, 0, 0.0)
    arms["chunk_copy"]["train_loss_final"] = None
    print(f"arm chunk_copy: worst W1 "
          f"{arms['chunk_copy']['worst_mean_w1']:.3f}, avg coverage "
          f"{arms['chunk_copy']['average_coverage']:.3f}", flush=True)

    results["arms"] = arms
    a = arms["uniform"]
    verdicts = {}
    for name in ("capped_weights", "stratified"):
        m = arms[name]
        improves = (
            m["worst_mean_w1"] < a["worst_mean_w1"]
            and m["average_coverage"] >= a["average_coverage"]
            and m["mean_w1"] - a["mean_w1"] <= 0.05
            and m["flow_acf_mae"] - a["flow_acf_mae"] <= 0.05
            and m["disc_auc_dist"] - a["disc_auc_dist"] <= 0.05)
        verdicts[name] = "improves" if improves else "null"
    results["verdicts"] = verdicts
    results["limitations"] = [
        "Single training/sampling seed (0); no across-seed spread.",
        "Evaluation on a fixed 2,000-window stratified subsample of "
        "test (and val), not the full splits.",
        "Stratification is on the past-volatility decile only; spread, "
        "activity and depth strata are reported as fitted features, "
        "not as training arms.",
        "Test sessions (Nov 11-12) are more volatile than train: "
        "18.6% of test windows sit above the train top-decile "
        "volatility threshold.",
    ]
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {os.path.abspath(args.out)}", flush=True)
    print(f"verdicts: {verdicts}", flush=True)


def window_returns_from_gen(gen: np.ndarray, ds) -> np.ndarray:
    """60 s log-return sums of generated windows (physical units)."""
    from flowforge.data import FEATURE_NAMES
    j = FEATURE_NAMES.index("log_ret")
    phys = (np.asarray(gen, dtype=np.float64)
            * np.asarray(ds.feature_std, dtype=np.float64)
            + np.asarray(ds.feature_mean, dtype=np.float64))
    return phys[:, :, j].sum(axis=1)


if __name__ == "__main__":
    main()
