"""P10 part 1: capture the DDPM baseline for the model-family challenge.

Reads only committed or already-recorded artifacts and recomputes
what can be recomputed from them:

- the committed demo bundle (demo/demo.npz, 64 real + 64 generated
  windows, build seed 7): marginal W1, flow ACF, joint/conditional
  scores, tail coverage, and the discriminator AUC from the stored
  scores, all recomputed here with flowforge.evaluate;
- the full-split demo numbers (demo/metrics.json, n_test 198);
- the expanded-corpus DDPM row measured by FF-R08
  (examples/ranking_stability_results.json): val/test mean W1,
  joint error, discriminator AUC, TSTR accuracy, NFE, sampling
  latency, parameter count.

Nothing is trained and nothing is sampled: this script captures the
baseline, it does not produce a new one. TSTR and sampling latency
cannot be measured from the demo bundle (it carries no val split and
no timing), so those cells are null with the reason recorded, and the
expanded-corpus row supplies them instead. Writes
examples/family_baseline_results.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import numpy as np

from flowforge.data import FEATURE_NAMES, N_FEATURES
from flowforge.evaluate import auc_score, flow_acf_table, marginal_table
from flowforge.family_contract import (
    CHALLENGER_METHODS, CONTRACT_SCHEMA, conditional_scores,
    default_contract, pending_challenger_row, tail_coverage)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _mean(xs):
    return float(np.mean([x for x in xs if x is not None]))


def capture() -> dict:
    contract = default_contract()

    demo_npz_path = os.path.join(REPO, "demo", "demo.npz")
    demo_metrics_path = os.path.join(REPO, "demo", "metrics.json")
    study_path = os.path.join(REPO, "examples",
                              "ranking_stability_results.json")
    manifest_path = os.path.join(REPO, "data", "split_manifest.json")

    bundle = np.load(demo_npz_path)
    real = bundle["real_windows"]
    gen = bundle["gen_windows"]
    assert list(bundle["feature_names"]) == FEATURE_NAMES
    assert real.shape == gen.shape == (64, 60, N_FEATURES)

    marginals = marginal_table(real, gen)
    acf_table = flow_acf_table(real, gen)
    joint = conditional_scores(real, gen)
    tails = tail_coverage(real, gen)
    y = np.concatenate([np.ones(len(bundle["disc_scores_real"])),
                        np.zeros(len(bundle["disc_scores_gen"]))])
    s = np.concatenate([bundle["disc_scores_real"],
                        bundle["disc_scores_gen"]])
    disc_auc = auc_score(y, s)
    # the stored scores are float32, so the recomputation agrees
    # with the stored AUC only to float32 storage noise (~1e-8)
    assert abs(disc_auc - float(bundle["disc_auc"])) < 1e-7

    with open(demo_metrics_path) as f:
        demo_metrics = json.load(f)
    with open(study_path) as f:
        study = json.load(f)
    with open(manifest_path) as f:
        committed_manifest = json.load(f)

    ddpm = study["table"]["ddpm"]
    costs = [r["cost"]
             for r in study["comparison"]["methods"]["ddpm"]["runs"]]
    ddpm_row = {
        "method": "ddpm",
        "status": "baseline",
        "mean_w1": ddpm["test_mean_w1"],
        "joint_error": ddpm["val_joint_error"],
        "lag_acf_error": ddpm["val_lag_acf_error"],
        "corr_error": ddpm["val_corr_error"],
        "tail_outside_rate": None,
        "auc_test": ddpm["auc_test_mean"],
        "tstr_acc_synth": ddpm["tstr_acc_synth_mean"],
        "nfe_per_window": ddpm["nfe_per_window"],
        "sample_seconds_per_1k": _mean(
            [c["sample_seconds_per_1k"] for c in costs]),
        "peak_memory_mb": None,
        "params": ddpm["params"],
    }
    pareto_rows = [ddpm_row] + [pending_challenger_row(m)
                                for m in CHALLENGER_METHODS]

    return {
        "schema": CONTRACT_SCHEMA,
        "contract": {
            "splits": contract.splits,
            "representation": contract.representation,
            "discriminator": contract.discriminator,
            "tstr": contract.tstr,
            "tuning": contract.tuning,
            "train_seed": contract.train_seed,
            "generation_seeds": list(contract.generation_seeds),
            "block_lengths": list(contract.block_lengths),
            "n_boot": contract.n_boot,
            "challenger_order": list(CHALLENGER_METHODS),
            "challenger_order_note": (
                "Smallest fair challenger first: a compact causal "
                "probabilistic autoregressor, then a conditional "
                "diffusion-forecasting variant, then flow matching "
                "only if justified under the same train data, "
                "parameter and compute envelopes."),
            "lob_bench_note": (
                "LOB-Bench (ICML 2025) reports strong autoregressive "
                "results on a different message-based LOB task; those "
                "scores cannot be compared numerically with "
                "FlowForge's 1-second feature-window metrics."),
        },
        "provenance": {
            "demo_npz_sha256": _sha256(demo_npz_path),
            "demo_metrics_sha256": _sha256(demo_metrics_path),
            "ranking_stability_results_sha256": _sha256(study_path),
            "split_manifest_sha256": study["split_manifest_sha256"],
            "committed_manifest_sha256": committed_manifest[
                "manifest_sha256"],
            "challenge_corpus_note": (
                "The expanded 28-session corpus NPZ is gitignored "
                "and not present in this checkout; its numbers below "
                "are the FF-R08 record, cited with the manifest SHA, "
                "not a recomputation. Part 2 must locate or re-record "
                "the corpus before running the comparison."),
        },
        "demo_bundle": {
            "n_windows": int(len(real)),
            "build_seed": int(demo_metrics["seed"]),
            "mean_w1": marginals["mean_w1"],
            "marginals": marginals["rows"],
            "flow_acf": acf_table["rows"],
            "joint_error": joint["joint_error"],
            "lag_acf_error": joint["lag_acf_error"],
            "corr_error": joint["corr_error"],
            "tail_outside_rate": tails["mean_outside_rate"],
            "tail_real_outside_rate": tails["real_outside_rate"],
            "tail_rows": tails["rows"],
            "auc_test": disc_auc,
            "tstr_acc_synth": None,
            "tstr_note": "Not measurable from the demo bundle: it "
                         "carries no val split to train the TSTR "
                         "model on.",
            "sample_seconds_per_1k": None,
            "sample_note": "Not measurable from the demo bundle: it "
                           "carries no sampling timing.",
        },
        "demo_full_split": {
            "n_test": int(demo_metrics["n_test"]),
            "mean_w1": demo_metrics["models"]["ddpm"]["mean_w1"],
            "auc_test": demo_metrics["models"]["ddpm"]["auc_test"],
            "flow_acf": demo_metrics["flow_acf"],
            "note": "Full-split demo numbers from demo/metrics.json; "
                    "the bundle above is a 64-window subsample of "
                    "the same evaluation, so its mean W1 differs by "
                    "subsample noise.",
        },
        "expanded_corpus": {
            "run_id": study["run_id"],
            "n_train": study["n_train"],
            "n_val": study["n_val"],
            "n_test": study["n_test"],
            "n_test_sessions": study["n_test_sessions"],
            "test_days": study["test_days"],
            "train_seed": study["preregistration"]["train_seed"],
            "generation_seeds": study["preregistration"]["gen_seeds"],
            "disc_epochs": study["preregistration"]["disc_epochs"],
            "tstr_epochs": study["preregistration"]["tstr_epochs"],
            "ddpm": {
                "val_mean_w1": ddpm["val_mean_w1"],
                "val_mean_w1_seeds": ddpm["val_mean_w1_seeds"],
                "test_mean_w1": ddpm["test_mean_w1"],
                "test_mean_w1_seeds": ddpm["test_mean_w1_seeds"],
                "val_joint_error": ddpm["val_joint_error"],
                "val_lag_acf_error": ddpm["val_lag_acf_error"],
                "val_corr_error": ddpm["val_corr_error"],
                "auc_test_mean": ddpm["auc_test_mean"],
                "auc_test_seeds": ddpm["auc_test_seeds"],
                "tstr_acc_synth_mean": ddpm["tstr_acc_synth_mean"],
                "tstr_acc_synth_seeds": ddpm["tstr_acc_synth_seeds"],
                "params": ddpm["params"],
                "nfe_per_window": ddpm["nfe_per_window"],
                "sample_seconds_per_1k_mean": _mean(
                    [c["sample_seconds_per_1k"] for c in costs]),
                "sample_seconds_per_1k_seeds": [
                    c["sample_seconds_per_1k"] for c in costs],
                "peak_memory_mb": None,
                "peak_memory_note": (
                    "FF-R08 recorded a per-run allocation delta "
                    "(0.006 MB), not peak RSS; the run peak RSS of "
                    "545.9 MB covers all seven policies, not the "
                    "DDPM alone, so no per-policy peak is claimed."),
                "hardware": study["hardware"],
                "peak_rss_mb": study["peak_rss_mb"],
                "minutes": study["minutes"],
                "train_loss": study["train_losses"]["ddpm"],
            },
        },
        "pareto_rows": pareto_rows,
        "parent_status": "open",
        "parent_note": (
            "Part 1 of 2: source contract, baseline capture, fixture "
            "and independent oracle. No family is implemented and "
            "none is labelled superior; the parent stays open until "
            "part 2 runs the family comparison."),
    }


def main() -> None:
    record = capture()
    out_path = os.path.join(HERE, "family_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")
    ddpm = record["expanded_corpus"]["ddpm"]
    print(f"DDPM baseline (expanded corpus, test): "
          f"mean W1 {ddpm['test_mean_w1']:.4f}, "
          f"AUC {ddpm['auc_test_mean']:.4f}, "
          f"TSTR {ddpm['tstr_acc_synth_mean']:.4f}, "
          f"NFE {ddpm['nfe_per_window']}, "
          f"{ddpm['sample_seconds_per_1k_mean']:.2f} s/1k windows")
    print(f"DDPM baseline (demo bundle, 64 windows): "
          f"mean W1 {record['demo_bundle']['mean_w1']:.4f}, "
          f"AUC {record['demo_bundle']['auc_test']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
