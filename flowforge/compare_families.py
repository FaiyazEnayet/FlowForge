"""P10 part 2: run the model-family comparison under the part-1 contract.

Every family trains on the same splits, in the same physical
representation, under the same optimizer-step budget, and generates
under the same generation-seed set (0, 100, 200), paired seed by
seed. The sampling stream for each family is derived per family and
seed, so draws stay independent across families even though the seed
integers coincide. Every family is scored with exactly the same
final discriminators (DiscMLP 128/64, 30 epochs, protocol seed 0,
fit on val, scored on test) and the same TSTR protocol, through the
shared ``flowforge.evaluate.comparison_report``. The single Pareto
table reports W1, conditional scores, temporal dependence, tail
coverage, AUC, utility, and NFE/latency/memory per family, and
reports metrics only: no row claims a ranking.

Families, in source order:

- ``ddpm``: the shipped base DDPM (ch 96, 100 NFE), retrained here
  under the FF-R08 protocol so its row is paired with the
  challengers. The recorded FF-R08 row is kept alongside as a
  cross-check, not as the paired row.
- ``autoregressive``: the compact causal GRU mixture
  (``flowforge.autoregressive``), the smallest fair challenger.
- ``conditional_diffusion_forecast``: the history-conditioned DDPM
  (``BookDDPM`` with ``use_history=True``) on the past-context
  lookback. It sees the full 60-second past-context tensor in
  addition to the two regime aggregates, so its row carries that
  conditioning note: it is not conditioning-matched with the other
  families, which see the regime aggregates only.
- ``flow_matching``: rectified flow on the DDPM denoiser backbone
  (``flowforge.flow_matching``), 20 Euler steps per window.
- ``vae`` and ``chunk_copy``: the reference comparators the parity
  schema requires, trained and generated exactly as in FF-R08.

Equal-budget view: DDPM, autoregressive, history-conditioned DDPM,
and flow matching each train 6 epochs at batch 32 (3,462 optimizer
steps on the expanded corpus); the VAE keeps the FF-R08 reference
protocol (25 epochs at batch 64, lr 2e-4). Equal-parameter view:
parameter counts are recorded per family next to the DDPM base
count of 221,767.
"""

from __future__ import annotations

import json
import math
import os
import time
import tracemalloc

import numpy as np
import torch

from .autoregressive import CausalGRUMixture, train_autoregressive
from .baselines import (RegimeMatchedChunkCopyBaseline,
                        segment_ids_from_windows)
from .data import N_FEATURES, WINDOW
from .evaluate import comparison_report, marginal_table
from .family_contract import (GENERATION_SEEDS, default_contract,
                              conditional_scores, tail_coverage)
from .flow_matching import RectifiedFlow, train_flow
from .models import (BookDDPM, CondVAE, count_params, derive_seed,
                     train_ddpm, train_vae)

METHODS = ("ddpm", "vae", "chunk_copy", "autoregressive",
           "conditional_diffusion_forecast", "flow_matching")
CHALLENGERS = ("autoregressive", "conditional_diffusion_forecast",
               "flow_matching")

DEFAULT_CONFIG = {
    "run_id": "p10-families",
    "train_epochs": 6,
    "batch": 32,
    "gen_seeds": GENERATION_SEEDS,
    "disc_epochs": 30,
    "tstr_epochs": 300,
    "n_boot": 100,
    "block_lengths": (1, 5, 10, 20),
    "ar_hidden": 256,
    "ar_components": 4,
    "ar_lr": 1e-3,
    "flow_nfe": 20,
    "flow_ch": 96,
    "ddpm_ch": 96,
    "ddpm_t_steps": 100,
    "vae_epochs": 25,
    "vae_lr": 2e-4,
    "methods": METHODS,
}

HISTORY_CONDITIONING_NOTE = (
    "The history-conditioned DDPM sees the full 60-second "
    "past-context tensor in addition to the two regime aggregates, "
    "so its row is not conditioning-matched with the other "
    "families, which see the regime aggregates only.")


def optimizer_steps(n_train: int, epochs: int, batch: int) -> int:
    """Optimizer steps for a run: epochs times batches per epoch."""
    return int(epochs * math.ceil(n_train / batch))


def sampling_seed(run_id: str, method: str, gen_seed: int) -> int:
    """The independent sampling-stream seed for one family and seed.

    The generation-seed integers are the same set for every family
    (the paired comparison), while the actual draw stream is
    derived per family and seed, so no two families share draws.
    """
    return derive_seed(f"{run_id}:{method}", f"sampling:{gen_seed}")


def _timed_generate(generate, nfe_per_window, n_windows):
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


def _save_state(checkpoint_dir, method, model):
    path = os.path.join(checkpoint_dir, f"train_{method}.pt")
    torch.save({"state_dict": model.state_dict()}, path)
    return path


def _load_state(checkpoint_dir, method, model):
    path = os.path.join(checkpoint_dir, f"train_{method}.pt")
    if not os.path.exists(path):
        return False
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
        ["state_dict"])
    return True


def train_families(ds, config, checkpoint_dir=None,
                   verbose=True):
    """Train every family on the shared train split.

    A checkpoint directory lets an interrupted run resume: a method
    whose state dict is already saved is loaded, not retrained.
    Returns (models, info) where info records train loss, parameter
    count, and optimizer steps per method.
    """
    run_id = config["run_id"]
    epochs = config["train_epochs"]
    batch = config["batch"]
    x_train = torch.from_numpy(ds.train.x)
    r_train = torch.from_numpy(ds.train.regime)
    n_train = len(x_train)
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    models = {}
    info = {}
    for method in config["methods"]:
        t0 = time.time()
        resumed = False
        if method == "ddpm":
            model = BookDDPM(t_steps=config["ddpm_t_steps"],
                             ch=config["ddpm_ch"], n_blocks=3,
                             run_id=f"{run_id}:ddpm")
            resumed = checkpoint_dir is not None and _load_state(
                checkpoint_dir, method, model)
            if resumed:
                hist = [float("nan")]
            else:
                hist, _ = train_ddpm(
                    model, x_train, r_train, epochs=epochs,
                    batch=batch, seed=0, verbose=False,
                    run_id=f"{run_id}:ddpm")
                if checkpoint_dir is not None:
                    _save_state(checkpoint_dir, method, model)
            steps = optimizer_steps(n_train, epochs, batch)
            train_loss = hist[-1]
        elif method == "vae":
            model = CondVAE(run_id=f"{run_id}:vae")
            resumed = checkpoint_dir is not None and _load_state(
                checkpoint_dir, method, model)
            if resumed:
                hist = [float("nan")]
            else:
                hist = train_vae(model, x_train, r_train,
                                 epochs=config["vae_epochs"], batch=64,
                                 seed=0, verbose=False,
                                 lr=config["vae_lr"],
                                 run_id=f"{run_id}:vae")
                if checkpoint_dir is not None:
                    _save_state(checkpoint_dir, method, model)
            steps = optimizer_steps(n_train, config["vae_epochs"], 64)
            train_loss = hist[-1]
        elif method == "chunk_copy":
            seg_train = segment_ids_from_windows(
                ds.train.session_ids, ds.train.t_start, ds.train.t_end)
            model = RegimeMatchedChunkCopyBaseline(
                ds.train.x, ds.train.regime, block_len=4,
                session_ids=ds.train.session_ids,
                segment_ids=seg_train)
            hist = [float("nan")]
            steps = 0
            train_loss = float("nan")
        elif method == "autoregressive":
            model = CausalGRUMixture(hidden=config["ar_hidden"],
                                     n_components=config["ar_components"],
                                     run_id=f"{run_id}:autoregressive")
            resumed = checkpoint_dir is not None and _load_state(
                checkpoint_dir, method, model)
            if resumed:
                hist = [float("nan")]
            else:
                hist = train_autoregressive(
                    model, x_train, r_train, epochs=epochs,
                    batch=batch, lr=config["ar_lr"], seed=0,
                    run_id=f"{run_id}:autoregressive")
                if checkpoint_dir is not None:
                    _save_state(checkpoint_dir, method, model)
            steps = optimizer_steps(n_train, epochs, batch)
            train_loss = hist[-1]
        elif method == "conditional_diffusion_forecast":
            model = BookDDPM(t_steps=config["ddpm_t_steps"],
                             ch=config["ddpm_ch"], n_blocks=3,
                             use_history=True,
                             run_id=f"{run_id}:cond-diffusion")
            resumed = checkpoint_dir is not None and _load_state(
                checkpoint_dir, method, model)
            if resumed:
                hist = [float("nan")]
            else:
                hist, _ = train_ddpm(
                    model, x_train, r_train, epochs=epochs,
                    batch=batch, seed=0, verbose=False,
                    past_context=torch.from_numpy(
                        ds.train.past_context),
                    past_mask=torch.from_numpy(ds.train.past_mask),
                    run_id=f"{run_id}:cond-diffusion")
                if checkpoint_dir is not None:
                    _save_state(checkpoint_dir, method, model)
            steps = optimizer_steps(n_train, epochs, batch)
            train_loss = hist[-1]
        elif method == "flow_matching":
            model = RectifiedFlow(ch=config["flow_ch"],
                                  nfe=config["flow_nfe"],
                                  run_id=f"{run_id}:flow-matching")
            resumed = checkpoint_dir is not None and _load_state(
                checkpoint_dir, method, model)
            if resumed:
                hist = [float("nan")]
            else:
                hist = train_flow(model, x_train, r_train,
                                  epochs=epochs, batch=batch,
                                  seed=0,
                                  run_id=f"{run_id}:flow-matching")
                if checkpoint_dir is not None:
                    _save_state(checkpoint_dir, method, model)
            steps = optimizer_steps(n_train, epochs, batch)
            train_loss = hist[-1]
        else:
            raise ValueError(f"unknown method {method!r}")
        model.eval() if hasattr(model, "eval") else None
        models[method] = model
        params = int(count_params(model)) \
            if method != "chunk_copy" else 0
        info[method] = {
            "params": params,
            "optimizer_steps": int(steps),
            "train_loss": None if not np.isfinite(train_loss)
            else float(train_loss),
            "train_minutes": (time.time() - t0) / 60,
            "resumed_from_checkpoint": bool(resumed),
        }
        if verbose:
            print(f"  {method:<32} params {params:>7} "
                  f"steps {steps:>6} "
                  f"train loss {train_loss:.4f} "
                  f"({info[method]['train_minutes']:.1f}m)",
                  flush=True)
    return models, info


def _nfe_per_window(method, config):
    if method == "ddpm" or method == "conditional_diffusion_forecast":
        return int(config["ddpm_t_steps"])
    if method == "vae":
        return 1
    if method == "chunk_copy":
        return 0
    if method == "autoregressive":
        return int(WINDOW)
    if method == "flow_matching":
        return int(config["flow_nfe"])
    raise ValueError(f"unknown method {method!r}")


def generate_families(models, ds_val, ds_test, train_meta, config,
                      checkpoint_dir=None, verbose=True):
    """Generate val and test windows for every family and seed.

    Returns the run records ``comparison_report`` expects: one
    {method, kind, seed, gen_val, gen_test, cost} record per family
    and generation seed, with the cost measured here. Generated
    arrays are saved under the checkpoint directory when given, so
    an interrupted run resumes without regenerating.
    """
    run_id = config["run_id"]
    r_val = torch.from_numpy(ds_val.regime)
    r_test = torch.from_numpy(ds_test.regime)
    r_val_np = ds_val.regime
    r_test_np = ds_test.regime
    n_windows = len(ds_val.x) + len(ds_test.x)
    past_val = torch.from_numpy(ds_val.past_context) \
        if ds_val.past_context is not None else None
    past_test = torch.from_numpy(ds_test.past_context) \
        if ds_test.past_context is not None else None
    mask_val = torch.from_numpy(ds_val.past_mask) \
        if ds_val.past_mask is not None else None
    mask_test = torch.from_numpy(ds_test.past_mask) \
        if ds_test.past_mask is not None else None
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    runs = []
    for method in config["methods"]:
        model = models[method]
        nfe = _nfe_per_window(method, config)
        for gseed in config["gen_seeds"]:
            gen_path = None
            if checkpoint_dir is not None:
                gen_path = os.path.join(
                    checkpoint_dir, f"gen_{method}_{gseed}.npz")
            if gen_path is not None and os.path.exists(gen_path):
                saved = np.load(gen_path)
                gv, gt = saved["gen_val"], saved["gen_test"]
                cost = json.loads(str(saved["cost_json"]))
            else:
                seed = sampling_seed(run_id, method, gseed)
                if method == "chunk_copy":
                    gv, s1 = model.generate(len(ds_val.x), r_val_np,
                                            seed=gseed)
                    gt, s2 = model.generate(len(ds_test.x), r_test_np,
                                            seed=gseed)
                    dt = s1["sample_seconds"] + s2["sample_seconds"]
                    cost = {
                        "sample_seconds": dt,
                        "sample_seconds_per_1k": dt / max(n_windows, 1)
                        * 1000,
                        "peak_memory_mb": max(s1["peak_memory_mb"],
                                              s2["peak_memory_mb"]),
                        "nfe": 0, "nfe_per_window": 0,
                    }
                elif method == "ddpm":
                    gv, gt, cost = _timed_generate(
                        lambda: (model.sample(
                            r_val, seed=seed, batch_size=32).numpy(),
                                 model.sample(
                            r_test, seed=seed + 1,
                            batch_size=32).numpy()),
                        nfe, n_windows)
                elif method == "conditional_diffusion_forecast":
                    gv, gt, cost = _timed_generate(
                        lambda: (model.sample(
                            r_val, seed=seed, batch_size=32,
                            past_context=past_val,
                            past_mask=mask_val).numpy(),
                                 model.sample(
                            r_test, seed=seed + 1, batch_size=32,
                            past_context=past_test,
                            past_mask=mask_test).numpy()),
                        nfe, n_windows)
                elif method == "vae":
                    with torch.inference_mode():
                        gv, gt, cost = _timed_generate(
                            lambda: (model.generate(
                                r_val, seed=seed).numpy(),
                                     model.generate(
                                r_test, seed=seed + 1).numpy()),
                            nfe, n_windows)
                elif method == "autoregressive":
                    gv, gt, cost = _timed_generate(
                        lambda: (model.generate(
                            r_val, seed=seed, batch_size=32).numpy(),
                                 model.generate(
                            r_test, seed=seed + 1,
                            batch_size=32).numpy()),
                        nfe, n_windows)
                elif method == "flow_matching":
                    gv, gt, cost = _timed_generate(
                        lambda: (model.sample(
                            r_val, seed=seed, batch_size=32).numpy(),
                                 model.sample(
                            r_test, seed=seed + 1,
                            batch_size=32).numpy()),
                        nfe, n_windows)
                else:
                    raise ValueError(f"unknown method {method!r}")
                if gen_path is not None:
                    np.savez_compressed(
                        gen_path, gen_val=gv, gen_test=gt,
                        cost_json=np.array(json.dumps(cost)))
            runs.append({"method": method, "kind": "conditional",
                         "seed": int(gseed), "gen_val": gv,
                         "gen_test": gt, "cost": cost})
        if verbose:
            print(f"  {method:<32} generated "
                  f"{len(config['gen_seeds'])} seeds", flush=True)
    return runs


def score_families(real_val, real_test, runs, config, train_info,
                   test_session_ids=None, recorded_ddpm=None):
    """Score every family under the identical final protocols.

    The discriminator and TSTR numbers come from the shared
    ``comparison_report``; the conditional scores and tail coverage
    come from the shared ``family_contract`` definitions. The one
    Pareto table is built through the contract, which refuses any
    row that claims a ranking.
    """
    contract = default_contract()
    seed_by_method = {m: sorted(r["seed"] for r in runs
                                if r["method"] == m)
                      for m in {r["method"] for r in runs}}
    if sorted(config["gen_seeds"]) == sorted(GENERATION_SEEDS):
        contract.check_seed_independence(seed_by_method)
    else:
        # a reduced seed set (a fast run or a test fixture) still
        # has to be paired: every family on exactly the same seeds
        sets = {tuple(s) for s in seed_by_method.values()}
        if len(sets) != 1:
            raise ValueError(
                f"all families must generate under the same seed "
                f"set, got {seed_by_method}")
    for run in runs:
        contract.validate_run(run, n_val=len(real_val),
                              n_test=len(real_test))
    n_test_sessions = len(set(np.asarray(test_session_ids).tolist())) \
        if test_session_ids is not None else 1
    comparison = comparison_report(
        real_test, real_val, runs,
        block_lengths=tuple(config["block_lengths"]),
        n_boot=int(config["n_boot"]),
        disc_epochs=int(config["disc_epochs"]),
        tstr_epochs=int(config["tstr_epochs"]),
        protocol_seed=0, n_test_sessions=n_test_sessions)
    by_method = {}
    for run in runs:
        by_method.setdefault(run["method"], []).append(run)
    rows = []
    details = {}
    for method in config["methods"]:
        method_runs = sorted(by_method[method], key=lambda r: r["seed"])
        entry = comparison["methods"][method]
        val_w1 = [marginal_table(real_val, r["gen_val"])["mean_w1"]
                  for r in method_runs]
        test_w1 = [marginal_table(real_test, r["gen_test"])["mean_w1"]
                   for r in method_runs]
        val_joint = [conditional_scores(real_val, r["gen_val"])
                     for r in method_runs]
        test_joint = [conditional_scores(real_test, r["gen_test"])
                      for r in method_runs]
        test_tails = [tail_coverage(real_test, r["gen_test"])
                      for r in method_runs]
        costs = [r["cost"] for r in method_runs]
        row = {
            "method": method,
            "status": "measured",
            "mean_w1": float(np.mean(test_w1)),
            "joint_error": float(np.mean(
                [j["joint_error"] for j in val_joint])),
            "lag_acf_error": float(np.mean(
                [j["lag_acf_error"] for j in val_joint])),
            "corr_error": float(np.mean(
                [j["corr_error"] for j in val_joint])),
            "tail_outside_rate": float(np.mean(
                [t["mean_outside_rate"] for t in test_tails])),
            "auc_test": float(entry["auc_test_mean"]),
            "tstr_acc_synth": float(entry["tstr_acc_synth_mean"]),
            "nfe_per_window": int(method_runs[0]["cost"]
                                  ["nfe_per_window"]),
            "sample_seconds_per_1k": float(np.mean(
                [c["sample_seconds_per_1k"] for c in costs])),
            "peak_memory_mb": float(np.mean(
                [c["peak_memory_mb"] for c in costs])),
            "params": int(train_info[method]["params"]),
        }
        rows.append(row)
        details[method] = {
            "val_mean_w1": float(np.mean(val_w1)),
            "val_mean_w1_seeds": [float(v) for v in val_w1],
            "test_mean_w1": float(np.mean(test_w1)),
            "test_mean_w1_seeds": [float(v) for v in test_w1],
            "val_joint_error": row["joint_error"],
            "val_joint_error_seeds": [float(j["joint_error"])
                                      for j in val_joint],
            "val_lag_acf_error": row["lag_acf_error"],
            "val_corr_error": row["corr_error"],
            "test_joint_error": float(np.mean(
                [j["joint_error"] for j in test_joint])),
            "test_lag_acf_error": float(np.mean(
                [j["lag_acf_error"] for j in test_joint])),
            "test_corr_error": float(np.mean(
                [j["corr_error"] for j in test_joint])),
            "tail_outside_rate": row["tail_outside_rate"],
            "tail_real_outside_rate": float(np.mean(
                [t["real_outside_rate"] for t in test_tails])),
            "auc_test_mean": row["auc_test"],
            "auc_test_seeds": [float(r["discriminator"]["auc_test"])
                               for r in entry["runs"]],
            "tstr_acc_synth_mean": row["tstr_acc_synth"],
            "tstr_acc_synth_seeds": [float(r["tstr"]["acc_synth"])
                                     for r in entry["runs"]],
            "params": row["params"],
            "optimizer_steps": int(
                train_info[method]["optimizer_steps"]),
            "train_loss": train_info[method]["train_loss"],
            "sample_seconds_per_1k_seeds": [
                float(c["sample_seconds_per_1k"]) for c in costs],
            "peak_memory_mb_seeds": [
                float(c["peak_memory_mb"]) for c in costs],
            "nfe_per_window": row["nfe_per_window"],
        }
    pareto = contract.pareto_table(rows)
    ddpm_params = train_info["ddpm"]["params"]
    equal_parameter_view = {
        method: {
            "params": int(train_info[method]["params"]),
            "params_ratio_to_ddpm": (
                None if ddpm_params == 0 else float(
                    train_info[method]["params"] / ddpm_params)),
        }
        for method in config["methods"]
    }
    record = {
        "schema": contract.schema,
        "run_id": config["run_id"],
        "methods": list(config["methods"]),
        "generation_seeds": [int(s) for s in config["gen_seeds"]],
        "discriminator": contract.discriminator,
        "tstr": contract.tstr,
        "tuning": {
            "basis": "equal_optimizer_steps",
            "optimizer_steps": {
                m: int(train_info[m]["optimizer_steps"])
                for m in config["methods"]},
            "train_epochs": int(config["train_epochs"]),
            "batch": int(config["batch"]),
            "vae_epochs": int(config["vae_epochs"]),
            "vae_lr": float(config["vae_lr"]),
            "ar_lr": float(config["ar_lr"]),
            "note": ("DDPM, autoregressive, history-conditioned "
                     "DDPM, and flow matching share the same "
                     "optimizer-step count; the VAE keeps the "
                     "FF-R08 reference protocol and its step count "
                     "is recorded here, not assumed equal."),
        },
        "equal_parameter_view": equal_parameter_view,
        "conditioning_note": HISTORY_CONDITIONING_NOTE,
        "pareto_table": pareto,
        "per_method": details,
        "comparison": comparison,
        "recorded_ddpm_cross_check": recorded_ddpm,
    }
    return record


def run_family_comparison(ds, config, checkpoint_dir=None,
                          verbose=True, recorded_ddpm=None):
    """Train, generate, and score every family on the given splits.

    ``ds`` is a BookDataset whose val/test splits are already the
    scored subsamples; the caller owns subsampling. Returns the
    record dict, including the one Pareto table.
    """
    config = {**DEFAULT_CONFIG, **config}
    t0 = time.time()
    models, train_info = train_families(
        ds, config, checkpoint_dir=checkpoint_dir, verbose=verbose)
    seg_train = segment_ids_from_windows(
        ds.train.session_ids, ds.train.t_start, ds.train.t_end)
    train_meta = {
        "x": ds.train.x, "regime": ds.train.regime,
        "session_ids": ds.train.session_ids,
        "segment_ids": seg_train,
        "test_session_ids": ds.test.session_ids,
        "fit_session_ids": sorted(set(
            ds.val.session_ids.tolist())),
    }
    runs = generate_families(
        models, ds.val, ds.test, train_meta, config,
        checkpoint_dir=checkpoint_dir, verbose=verbose)
    record = score_families(
        ds.val.x, ds.test.x, runs, config, train_info,
        test_session_ids=ds.test.session_ids,
        recorded_ddpm=recorded_ddpm)
    record["minutes"] = (time.time() - t0) / 60
    record["splits"] = {
        "n_train": int(len(ds.train.x)),
        "n_val": int(len(ds.val.x)),
        "n_test": int(len(ds.test.x)),
        "manifest_sha256": ds.manifest_sha256,
    }
    return record
