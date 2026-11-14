"""Preregistered latency frontier for the P08 DDIM sampler.

Question: how much wall time does DDIM save at 10/20/50/100
function evaluations per window, and what fidelity does each budget
cost against the ancestral 100-step reference, under identical
conditions?

Operating points (one trained DDPM, one trained VAE, one chunk-copy
baseline; every point generates for the same validation and test
regimes under the same generation seeds):

- ``ddpm``: the shipped ancestral sampler, 100 NFE. The reference.
- ``ddim_10`` / ``ddim_20`` / ``ddim_50`` / ``ddim_100``: the DDIM
  sampler of flowforge.samplers at that NFE, deterministic (eta 0),
  reusing the same trained DDPM weights.
- ``vae``: the conditional VAE baseline, 1 forward pass per window.
- ``chunk_copy``: the regime-matched chunk-copy baseline, 0 NFE.

Every point reports the same cost fields: NFE per window, cold-start
seconds (first 32-window call), steady-state seconds per 1,000
windows, batch size, trainable parameter count, peak traced memory,
and peak process RSS. Hardware (CPU model, core count, torch
threads, torch version) is recorded with the run.

Comparison: all points go through the identical baseline-parity
protocol of flowforge.evaluate.comparison_report (the same held-out
discriminator, the same TSTR protocol, paired W1 differences with
chronological block-bootstrap intervals at block lengths 1/5/10/20,
pooled over the generation seeds), so every operating point carries
full DDPM/VAE/chunk-copy comparisons and blocked intervals.

Preregistered before any run of this harness:

- Primary quality metric (lower is better): ``val_mean_w1``
  (marginal 1-Wasserstein distance of samples generated for a fixed
  1,000-window strided subsample of the validation regimes).
  Pareto selection uses the validation metric; the test metric is
  reported as confirmation and is never used to pick a point.
- Primary cost metric (lower is better): steady-state seconds per
  1,000 windows over the same val/test generation.
- Pareto rule: an operating point is Pareto-optimal when no other
  point has both a lower-or-equal val mean W1 and a lower-or-equal
  steady-state seconds per 1k with at least one strict win. The
  reported result is the Pareto set, never "fastest is best": a
  point is selected only if it trades quality for speed on the
  frontier, and every point's quality loss against the ancestral
  reference is reported with its blocked interval, including when
  the loss is large.
- No promotion rule: no operating point is promoted to a default.
  The fastest point is not the recommended point unless it is
  Pareto-optimal with a quality loss inside its interval.

Data caveat: the recorded dataset (data/book_windows.npz) is
gitignored, so these numbers describe the protocol on the dataset
whose manifest SHA is recorded in the output, not a committed
artifact. The default run uses the expanded 28-session corpus
(8 original chunks + 20 B1-QC 4 h sessions, Nov 9-12 2026);
pass --data to point at another build. Three generation seeds
(0, 100, 200), one training seed (0): a single training run is a
limitation, stated in the output.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
import tracemalloc

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))

from flowforge.baselines import (  # noqa: E402
    RegimeMatchedChunkCopyBaseline, segment_ids_from_windows)
from flowforge.data import load_dataset  # noqa: E402
from flowforge.evaluate import (  # noqa: E402
    comparison_report, marginal_table)
from flowforge.models import (  # noqa: E402
    BookDDPM, CondVAE, count_params, train_ddpm, train_vae)
from flowforge.samplers import pareto_frontier, sample_ddim  # noqa: E402

SEED = 0
GEN_SEEDS = (0, 100, 200)
N_EVAL = 1000
EPOCHS = 6
VAE_EPOCHS = 25
# The VAE comparator trains at lr 2e-4, not the shipped 1e-3: on the
# expanded corpus the shipped rate diverges to nan in 2 of 3 tried
# initializations (the run-id stream and the set_seed(0) demo
# protocol; the third completed only after loss spikes to 3e12), and
# lr 5e-4 still diverges on one seed. At 2e-4 all four tried inits
# complete (final loss 0.56-0.62, val W1 0.389-0.396), though spikes
# to 1e10 still occur and recover. 7.7% of the training windows
# carry |x| > 10 (max 153), and the KLD exp(logvar) term is the
# fragile part. This is a disclosed stabilization of the comparator
# only; the DDPM arm is untouched and the numbers are reported as
# measured.
VAE_LR = 2e-4
BATCH = 32
T_STEPS = 100
NFE_LIST = (10, 20, 50, 100)
ETA = 0.0

DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "..", "data", "book_windows.npz")


def hardware_record():
    """Hardware and thread settings that decide the latency numbers."""
    return {
        "device": "cpu",
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def _timed_cost(generate, nfe_per_window, n_windows):
    """Run a (gen_val, gen_test) generator and record its cost."""
    tracemalloc.start()
    t0 = time.perf_counter()
    gen_val, gen_test = generate()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return gen_val, gen_test, {
        "sample_seconds": dt,
        "sample_seconds_per_1k": dt / max(n_windows, 1) * 1000,
        "peak_memory_mb": peak / 1e6,
        "nfe": int(nfe_per_window * n_windows),
        "nfe_per_window": int(nfe_per_window),
    }


def _cold_start_seconds(sampler, r_probe):
    """Wall time of the first 32-window call for one operating point.

    The probe uses its own seed and its windows are discarded; it
    measures first-call cost (thread warmup, first kernels) for that
    sampler. Only the first probe in the process also pays process
    warmup, so cold-start numbers are comparable as first-call costs
    and the steady-state rate is the headline latency.
    """
    t0 = time.perf_counter()
    sampler(r_probe)
    return time.perf_counter() - t0


def compare_samplers(splits, train_meta=None, run_id="p08-frontier",
                     epochs=EPOCHS, vae_epochs=VAE_EPOCHS,
                     vae_lr=VAE_LR,
                     t_steps=T_STEPS, nfe_list=NFE_LIST, eta=ETA,
                     seeds=GEN_SEEDS, batch_size=BATCH, verbose=True):
    """Run the paired latency frontier on in-memory splits.

    ``splits`` maps "train"/"val"/"test" to (x, regime) torch tensors
    in (windows, window, features) / (windows, 2) layout. The val and
    test splits given here are the scored subsamples; the test split
    is scored only and never used to pick an operating point.
    ``train_meta`` carries the numpy training windows, regimes, and
    session/segment ids the chunk-copy baseline needs; without it
    the chunk-copy point is built from the train split with one
    session and one segment. Returns the record dict.
    """
    x_train, r_train = splits["train"]
    x_val, r_val = splits["val"]
    x_test, r_test = splits["test"]
    real_val = x_val.numpy()
    real_test = x_test.numpy()
    n_val, n_test = len(x_val), len(x_test)
    n_windows = n_val + n_test
    r_val_np = r_val.numpy()
    r_test_np = r_test.numpy()

    model = BookDDPM(t_steps=t_steps, run_id=run_id)
    n_params_ddpm = count_params(model)
    t0 = time.time()
    hist, _ = train_ddpm(model, x_train, r_train, epochs=epochs,
                         batch=batch_size, seed=SEED, verbose=False,
                         run_id=run_id)
    train_seconds = time.time() - t0
    if verbose:
        print(f"  ddpm trained {epochs} epochs in "
              f"{train_seconds / 60:.1f}m "
              f"(final loss {hist[-1]:.4f})", flush=True)
    vae = CondVAE(run_id=run_id)
    n_params_vae = count_params(vae)
    t0 = time.time()
    vae_hist = train_vae(vae, x_train, r_train, epochs=vae_epochs,
                         seed=SEED, verbose=False, run_id=run_id,
                         lr=vae_lr)
    vae_seconds = time.time() - t0
    if verbose:
        print(f"  vae trained {vae_epochs} epochs in "
              f"{vae_seconds / 60:.1f}m "
              f"(final loss {vae_hist[-1]:.4f})", flush=True)
    if train_meta is None:
        train_meta = {
            "x": x_train.numpy(), "regime": r_train.numpy(),
            "session_ids": np.zeros(len(x_train), dtype=int),
            "segment_ids": np.zeros(len(x_train), dtype=int),
        }
    chunk_copy = RegimeMatchedChunkCopyBaseline(
        train_meta["x"], train_meta["regime"], block_len=4,
        session_ids=train_meta["session_ids"],
        segment_ids=train_meta["segment_ids"])

    model.eval()
    vae.eval()
    r_probe = r_val[:min(32, len(r_val))]

    # one generator per (method, seed) pair: val then test
    def _gen_ddpm(seed):
        return (model.sample(r_val, seed=seed, batch_size=batch_size)
                .numpy(),
                model.sample(r_test, seed=seed + 1,
                             batch_size=batch_size).numpy())

    def _gen_ddim(nfe, seed):
        return (sample_ddim(model, r_val, nfe=nfe, eta=eta, seed=seed,
                            batch_size=batch_size).numpy(),
                sample_ddim(model, r_test, nfe=nfe, eta=eta, seed=seed + 1,
                            batch_size=batch_size).numpy())

    def _gen_vae(seed):
        with torch.inference_mode():
            return (vae.generate(r_val, seed=seed).numpy(),
                    vae.generate(r_test, seed=seed + 1).numpy())

    def _gen_chunk_copy(seed):
        gv, s1 = chunk_copy.generate(n_val, r_val_np, seed=seed)
        gt, s2 = chunk_copy.generate(n_test, r_test_np, seed=seed)
        dt = s1["sample_seconds"] + s2["sample_seconds"]
        cost = {
            "sample_seconds": dt,
            "sample_seconds_per_1k": dt / max(n_windows, 1) * 1000,
            "peak_memory_mb": max(s1["peak_memory_mb"],
                                  s2["peak_memory_mb"]),
            "nfe": 0, "nfe_per_window": 0,
        }
        return gv, gt, cost

    runs = []
    val_w1 = {}
    steady = {}
    cold = {}
    params = {}
    # generation order is fixed so cold-start probes are comparable:
    # the ancestral reference first, the DDIM points at their own
    # budgets, then the non-diffusion baselines
    plan = [("ddpm", t_steps, _gen_ddpm, n_params_ddpm)]
    for nfe in nfe_list:
        plan.append((f"ddim_{nfe}", nfe,
                     (lambda nfe: (lambda seed: _gen_ddim(nfe, seed)))(nfe),
                     n_params_ddpm))
    plan.append(("vae", 1, _gen_vae, n_params_vae))
    plan.append(("chunk_copy", 0, None, 0))
    for name, nfe, gen, n_params in plan:
        if name == "chunk_copy":
            cold[name] = _cold_start_seconds(
                lambda r: chunk_copy.generate(
                    len(r), r.numpy(), seed=999), r_probe)
        elif name == "vae":
            cold[name] = _cold_start_seconds(
                lambda r: vae.generate(r, seed=999), r_probe)
        elif name == "ddpm":
            cold[name] = _cold_start_seconds(
                lambda r: model.sample(r, seed=999, batch_size=batch_size),
                r_probe)
        else:
            cold[name] = _cold_start_seconds(
                lambda r, nfe=nfe: sample_ddim(
                    model, r, nfe=nfe, eta=eta, seed=999,
                    batch_size=batch_size), r_probe)
        params[name] = int(n_params)
        val_w1[name] = []
        steady[name] = []
        for seed in seeds:
            if name == "chunk_copy":
                gv, gt, cost = _gen_chunk_copy(seed)
            else:
                gv, gt, cost = _timed_cost(
                    lambda seed=seed: gen(seed), nfe, n_windows)
            runs.append({"method": name, "kind": "conditional",
                         "seed": seed, "gen_val": gv, "gen_test": gt,
                         "cost": cost})
            val_w1[name].append(
                marginal_table(real_val, gv)["mean_w1"])
            steady[name].append(cost["sample_seconds_per_1k"])
        if verbose:
            print(f"  {name:<10} val W1 {np.mean(val_w1[name]):.4f} "
                  f"steady {np.mean(steady[name]):.2f}s/1k "
                  f"cold {cold[name]:.3f}s", flush=True)

    n_test_sessions = None
    if train_meta is not None and "test_session_ids" in train_meta:
        n_test_sessions = len(np.unique(train_meta["test_session_ids"]))
    comparison = comparison_report(
        real_test, real_val, runs,
        block_lengths=(1, 5, 10, 20), n_boot=100,
        n_test_sessions=n_test_sessions)
    peak_rss_mb = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024.0

    points = {}
    for name, _, _, _ in plan:
        points[name] = {
            "method": name,
            "kind": "conditional",
            "nfe_per_window": int(
                t_steps if name == "ddpm" else
                int(name.split("_")[1]) if name.startswith("ddim_")
                else 1 if name == "vae" else 0),
            "eta": float(eta) if name.startswith("ddim_") else None,
            "val_mean_w1": float(np.mean(val_w1[name])),
            "val_mean_w1_seeds": [float(v) for v in val_w1[name]],
            "test_mean_w1": float(
                comparison["methods"][name]["mean_w1"]),
            "discriminator_auc_test_mean": float(
                comparison["methods"][name]["auc_test_mean"]),
            "tstr_acc_synth_mean": float(
                comparison["methods"][name]["tstr_acc_synth_mean"]),
            "steady_seconds_per_1k": float(np.mean(steady[name])),
            "steady_seconds_per_1k_seeds": [float(v)
                                            for v in steady[name]],
            "cold_start_seconds": float(cold[name]),
            "batch_size": int(batch_size),
            "params": int(params[name]),
            # process peak RSS, shared by every point: the harness
            # runs in one process, so per-point RSS is not separable
            "peak_rss_mb": float(peak_rss_mb),
        }
    front = pareto_frontier([
        {"method": name,
         "val_mean_w1": points[name]["val_mean_w1"],
         "steady_seconds_per_1k": points[name]["steady_seconds_per_1k"]}
        for name in points])
    for name in points:
        points[name]["pareto"] = name in front
    return {
        "run_id": run_id,
        "reference": "ddpm",
        "epochs": int(epochs),
        "vae_epochs": int(vae_epochs),
        "vae_lr": float(vae_lr),
        "batch": int(batch_size),
        "t_steps": int(t_steps),
        "nfe_list": [int(n) for n in nfe_list],
        "eta": float(eta),
        "seed": int(SEED),
        "gen_seeds": [int(s) for s in seeds],
        "n_train": int(len(x_train)),
        "n_val_eval": int(n_val),
        "n_test_eval": int(n_test),
        "ddpm_final_train_loss": float(hist[-1]),
        "vae_final_train_loss": float(vae_hist[-1]),
        "ddpm_train_seconds": float(train_seconds),
        "vae_train_seconds": float(vae_seconds),
        "hardware": hardware_record(),
        "peak_rss_mb": float(peak_rss_mb),
        "operating_points": points,
        "pareto": front,
        "pareto_rule": (
            "Pareto-optimal when no other operating point has both a "
            "lower-or-equal val mean W1 and a lower-or-equal "
            "steady-state seconds per 1k with at least one strict "
            "win. The Pareto set is the result; the fastest point is "
            "never claimed best on speed alone."),
        "comparison": comparison,
        "limitations": [
            "single training seed (0); one DDPM and one VAE trained",
            "validation/test subsamples of 1000 windows, not the full splits",
            "test metrics are confirmation only; Pareto selection uses val W1",
            "cold-start probe pays process warmup only for the first method",
            "CPU-only run; GPU latency is not measured here",
            ("VAE comparator trained at lr 2e-4, not the shipped 1e-3: "
             "the shipped rate diverges to nan in 2 of 3 tried "
             "initializations on the expanded corpus; see VAE_LR note"),
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="P08 DDIM latency frontier")
    parser.add_argument("--data", default=os.path.abspath(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--vae-epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--t-steps", type=int, default=T_STEPS)
    parser.add_argument("--nfe", default=",".join(map(str, NFE_LIST)),
                        help="comma-separated NFE budgets for DDIM")
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "latency_frontier_results.json"))
    parser.add_argument("--run-id", default="p08-frontier")
    args = parser.parse_args()
    t0 = time.time()
    ds = load_dataset(os.path.abspath(args.data))
    print(f"split-manifest SHA-256: {ds.manifest_sha256}")
    print(f"train {len(ds.train.x)} / val {len(ds.val.x)} "
          f"/ test {len(ds.test.x)} windows")
    x_train = torch.from_numpy(ds.train.x)
    r_train = torch.from_numpy(ds.train.regime)
    idx_val = np.linspace(0, len(ds.val.x) - 1, N_EVAL).astype(int)
    idx_test = np.linspace(0, len(ds.test.x) - 1, N_EVAL).astype(int)
    splits = {
        "train": (x_train, r_train),
        "val": (torch.from_numpy(ds.val.x[idx_val]),
                torch.from_numpy(ds.val.regime[idx_val])),
        "test": (torch.from_numpy(ds.test.x[idx_test]),
                 torch.from_numpy(ds.test.regime[idx_test])),
    }
    seg_train = segment_ids_from_windows(ds.train.session_ids,
                                         ds.train.t_start, ds.train.t_end)
    train_meta = {
        "x": ds.train.x, "regime": ds.train.regime,
        "session_ids": ds.train.session_ids,
        "segment_ids": seg_train,
        "test_session_ids": ds.test.session_ids[idx_test],
    }
    nfe_list = tuple(int(a) for a in args.nfe.split(",") if a.strip())
    record = compare_samplers(
        splits, train_meta=train_meta, run_id=args.run_id,
        epochs=args.epochs, vae_epochs=args.vae_epochs,
        t_steps=args.t_steps, nfe_list=nfe_list, eta=args.eta)
    record.update({
        "data_path": os.path.basename(args.data),
        "split_manifest_sha256": ds.manifest_sha256,
        "val_subsample": {"n": int(N_EVAL), "method": "strided",
                          "first_index": int(idx_val[0]),
                          "last_index": int(idx_val[-1])},
        "test_subsample": {"n": int(N_EVAL), "method": "strided",
                           "first_index": int(idx_test[0]),
                           "last_index": int(idx_test[-1])},
        "minutes": (time.time() - t0) / 60,
    })
    print("\n=== P08 latency frontier (val W1 vs steady s/1k) ===")
    for name, p in record["operating_points"].items():
        print(f"  {name:<10} NFE {p['nfe_per_window']:>3} "
              f"val W1 {p['val_mean_w1']:.4f} "
              f"test W1 {p['test_mean_w1']:.4f} "
              f"AUC {p['discriminator_auc_test_mean']:.3f} "
              f"{p['steady_seconds_per_1k']:.2f}s/1k "
              f"{'PARETO' if p['pareto'] else ''}")
    print(f"Pareto set: {record['pareto']}")
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print(f"results -> {args.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
