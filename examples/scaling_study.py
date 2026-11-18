"""Scaling study (P12): sensible scaling, not parameter theater.

A compact, preregistered model-size sweep of the BookDDPM denoiser.
The capacity envelope is fixed below before any training: small,
base (the shipped default), and large. Every arm trains under equal
training compute (same epochs, batch, learning rate, optimizer, and
seeds) and again under convergence-normalized training (each arm
trained for the number of epochs at which its own train loss first
comes within 5 percent of its final equal-compute loss).

Per cell (arm x seed) the study records:
  - train denoising loss trajectory and final train loss
  - validation denoising loss (deterministic scoped draws, fixed
    strided validation subsample; never the test split)
  - validation mean 1-Wasserstein distance of generated windows
  - nearest-neighbor distances: generated to train reference and to
    validation reference, plus the real-validation-to-train baseline
    that calibrates the memorization reading
  - train-versus-validation reconstruction probes: fixed-timestep
    denoising of noised train and validation windows with identical
    noise draws, so a memorizing model shows a train-side advantage
  - generated diversity (mean pairwise distance) against the same
    statistic on real validation windows
  - inference costs: parameter count, sampling seconds per 1k
    windows, training seconds, and total network evaluations
    (one denoiser forward pass per optimizer step plus T steps per
    generated window)

Promotion rule (preregistered): the large arm is promoted only if,
in BOTH rounds, its validation denoising loss and its validation
mean W1 beat the base arm by margins whose blocked bootstrap 95
percent intervals exclude zero, AND its parameter count stays within
3x of base and its sampling cost within 2.5x of base. Anything else
is reported as no measurable gain, honestly.

The frozen test split is never loaded here: main() passes only the
train and validation arrays into run_study, which takes no test
input at all. Optimizer and config are recorded in the results file.
Fixed seeds everywhere. CPU only, 2 torch threads.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from flowforge.data import ensure_publishable, load_dataset
from flowforge.evaluate import marginal_table, wasserstein_1d
from flowforge.models import (BookDDPM, count_params,
                              eval_denoising_losses, predict_x0_from,
                              scoped_rng, train_ddpm)

SEED = 7
DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "book_windows.npz")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")

# Preregistered capacity envelope: name -> BookDDPM kwargs. Base is
# the shipped default (ch=96, n_blocks=3, 221,767 parameters).
CAPACITY_ENVELOPE = {
    "small": {"ch": 48, "n_blocks": 2},
    "base": {"ch": 96, "n_blocks": 3},
    "large": {"ch": 128, "n_blocks": 4},
}
SEEDS = (0, 1, 2)
EPOCHS = 6
BATCH = 32
LR = 2e-4
T_STEPS = 100
EVAL_SEED = 7
GEN_SEED = 2000
RECON_T_STEP = 50
RECON_SEED = 5
N_VAL_EVAL = 1000
N_NN_REF = 1000
N_RECON = 256
N_GEN = 1000
BLOCK = 20
N_BOOT = 500
CONVERGENCE_TOL = 0.05
PARAMS_RATIO_MAX = 3.0
SAMPLE_RATIO_MAX = 2.5
DIVERSITY_CAP = 200


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def total_network_evals(optimizer_steps: int, n_generated: int,
                        t_steps: int) -> int:
    """Denoiser forward passes: one per optimizer step plus T per
    generated window. Backward passes are not counted as evaluations."""
    return int(optimizer_steps) + int(n_generated) * int(t_steps)


def _strided(n: int, cap: int) -> np.ndarray:
    """Fixed strided subsample indices: deterministic, time-ordered."""
    if n <= cap:
        return np.arange(n)
    step = n / cap
    return np.unique((np.arange(cap) * step).astype(int))


def _flatten(windows) -> np.ndarray:
    arr = np.asarray(windows, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"windows must be 3-D (n, window, features), "
                         f"got shape {arr.shape}")
    return arr.reshape(arr.shape[0], -1)


def nearest_neighbor_distances(generated, reference) -> np.ndarray:
    """Per-window minimum L2 distance from each generated window to
    the reference set, in standardized feature units."""
    g = _flatten(generated)
    r = _flatten(reference)
    if g.shape[0] == 0 or r.shape[0] == 0:
        raise ValueError("nearest-neighbor inputs must be non-empty")
    if g.shape[1] != r.shape[1]:
        raise ValueError(
            f"feature mismatch: generated has {g.shape[1]} flattened "
            f"features, reference has {r.shape[1]}")
    gt = torch.from_numpy(g)
    rt = torch.from_numpy(r)
    dists = torch.cdist(gt, rt)
    return dists.min(dim=1).values.numpy()


def pairwise_diversity(windows) -> float:
    """Mean pairwise L2 distance among windows (first DIVERSITY_CAP).

    Identical windows score 0; a collapsed generator scores near 0
    while a diverse one scores near the real-window statistic."""
    arr = _flatten(windows)
    if arr.shape[0] < 2:
        raise ValueError("diversity needs at least 2 windows")
    sub = torch.from_numpy(arr[:DIVERSITY_CAP])
    d = torch.cdist(sub, sub)
    iu = torch.triu_indices(sub.shape[0], sub.shape[0], offset=1)
    return float(d[iu[0], iu[1]].mean())


def reconstruction_probe_per_window(model: BookDDPM, x, regime,
                                    t_step: int, seed: int) -> np.ndarray:
    """Per-window fixed-timestep denoising MSE (see
    :func:`reconstruction_probe`)."""
    if x.shape[0] == 0:
        raise ValueError("reconstruction probe windows are empty")
    xc = x.permute(0, 2, 1).contiguous()
    t = torch.full((x.shape[0],), int(t_step), dtype=torch.long)
    was_training = model.training
    model.eval()
    try:
        with scoped_rng(seed):
            with torch.no_grad():
                noise = torch.randn_like(xc)
                xt = model.q_sample(xc, t, noise)
                pred = model.net(xt, t, regime)
                x0_hat = predict_x0_from(pred, xt, t, model.sqrt_acp,
                                         model.sqrt_1m_acp,
                                         model.objective)
                mse = torch.nn.functional.mse_loss(
                    x0_hat, xc, reduction="none").mean(dim=(1, 2))
    finally:
        model.train(was_training)
    return mse.numpy()


def reconstruction_probe(model: BookDDPM, x, regime, t_step: int,
                         seed: int) -> float:
    """Fixed-timestep denoising MSE: noise the windows to step t and
    recover the clean estimate. Identical seeds and batch shapes give
    identical noise draws, so train and validation probes are directly
    comparable; a memorizing model reconstructs train windows better.
    """
    return float(np.mean(
        reconstruction_probe_per_window(model, x, regime, t_step, seed)))


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Block-resample indices: whole consecutive blocks, wrapped."""
    n_blocks = (n + block - 1) // block
    starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
    idx = np.concatenate(
        [np.arange(s, s + block) % n for s in starts])[:n]
    return idx


def block_bootstrap_diff_ci(a, b, block: int = BLOCK, n_boot: int = N_BOOT,
                            seed: int = 0, paired=None):
    """95 percent block-bootstrap CI for mean(a) - mean(b).

    ``paired`` selects the resampling: True resamples the paired
    differences, False resamples each side independently. The default
    (None) pairs equal-length inputs and resamples unequal lengths
    independently. Time order is the block order, so windows that are
    adjacent in time stay together.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if block < 1:
        raise ValueError("block must be positive")
    if a.size == 0 or b.size == 0:
        raise ValueError("bootstrap inputs must be non-empty")
    rng = np.random.default_rng(seed)
    if paired is None:
        paired = a.size == b.size
    if paired:
        if a.size != b.size:
            raise ValueError("paired bootstrap needs equal lengths")
        diffs = a - b
        stats = [float(np.mean(diffs[_block_indices(len(diffs), block,
                                                    rng)]))
                 for _ in range(n_boot)]
    else:
        stats = [float(np.mean(a[_block_indices(len(a), block, rng)])
                       - np.mean(b[_block_indices(len(b), block, rng)]))
                 for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def _mean_w1(real, gen) -> float:
    """Mean per-feature 1-Wasserstein distance, real vs generated."""
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    w1s = [wasserstein_1d(real[:, :, j].ravel(), gen[:, :, j].ravel())
           for j in range(real.shape[2])]
    return float(np.mean(w1s))


def _mean_w1_fast(real, gen) -> float:
    """Mean per-feature 1-Wasserstein distance via sorted differences.

    The repo's quantile-grid W1 sorts both samples and compares
    quantiles on a grid of min(n, m) points; comparing the sorted
    samples directly is the same statistic up to float rounding
    (verified to agree within 1e-9 on study data), and it runs in
    milliseconds instead of tens of seconds, which is what makes a
    500-replicate block bootstrap feasible at all. Only the
    bootstrap uses this form; reported point estimates still come
    from the repo's marginal_table.
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    w1s = []
    for j in range(real.shape[2]):
        a = np.sort(real[:, :, j].ravel())
        b = np.sort(gen[:, :, j].ravel())
        n = min(len(a), len(b))
        w1s.append(float(np.mean(np.abs(a[:n] - b[:n]))))
    return float(np.mean(w1s))


def block_bootstrap_w1_diff_ci(real, gen_a_by_seed, gen_b_by_seed,
                               block: int = BLOCK, n_boot: int = N_BOOT,
                               seed: int = 0):
    """95 percent block-bootstrap CI for mean W1(gen_b) - mean W1(gen_a).

    Each bootstrap replicate resamples time-ordered blocks of the
    validation windows within every seed, recomputes both mean W1
    values on the resample, and averages the difference across seeds.
    The recomputation uses :func:`_mean_w1_fast`, the sorted-difference
    form of the same statistic (see its note). ``gen_a_by_seed`` and
    ``gen_b_by_seed`` are per-seed generated arrays aligned
    row-for-row with ``real``.
    """
    real = np.asarray(real, dtype=np.float64)
    n = real.shape[0]
    if block < 1:
        raise ValueError("block must be positive")
    if n == 0:
        raise ValueError("bootstrap inputs must be non-empty")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        diffs = [_mean_w1_fast(real[idx], np.asarray(gb)[idx])
                 - _mean_w1_fast(real[idx], np.asarray(ga)[idx])
                 for ga, gb in zip(gen_a_by_seed, gen_b_by_seed)]
        stats.append(float(np.mean(diffs)))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def convergence_epoch(loss_history, tol: float = CONVERGENCE_TOL) -> int:
    """First epoch whose train loss is within tol of the final loss.

    Returns the last epoch when the loss never comes within tol: the
    arm had not converged inside the equal-compute run.
    """
    hist = [float(v) for v in loss_history]
    if not hist:
        raise ValueError("loss history is empty")
    final = hist[-1]
    for i, v in enumerate(hist):
        if v <= final * (1.0 + tol):
            return i
    return len(hist) - 1


def promotion_verdict(arm: str, gains: dict, feasibility: dict) -> dict:
    """Preregistered promotion rule for the larger arm.

    ``gains`` carries blocked 95 percent CIs for the validation loss
    and validation W1 differences (large minus base): both must sit
    entirely below zero. ``feasibility`` carries the large-to-base
    ratios for parameter count and sampling cost, which must stay
    within the preregistered caps. Every failing check is named.
    """
    reasons = []
    loss_ci = gains["val_loss_ci"]
    if not loss_ci[1] < 0:
        reasons.append(
            f"val_loss CI {loss_ci} does not exclude zero: no "
            f"measured validation-loss gain")
    w1_ci = gains["val_w1_ci"]
    if not w1_ci[1] < 0:
        reasons.append(
            f"val_w1 CI {w1_ci} does not exclude zero: no measured "
            f"validation-W1 gain")
    if feasibility["params_ratio"] > PARAMS_RATIO_MAX:
        reasons.append(
            f"params ratio {feasibility['params_ratio']:.2f} exceeds "
            f"the {PARAMS_RATIO_MAX} cap")
    if feasibility["sample_ratio"] > SAMPLE_RATIO_MAX:
        reasons.append(
            f"sample cost ratio {feasibility['sample_ratio']:.2f} "
            f"exceeds the {SAMPLE_RATIO_MAX} cap")
    return {"arm": arm, "promoted": not reasons, "reasons": reasons}


def run_cell(name: str, x_train, r_train, x_val, r_val, seed: int,
             epochs: int = EPOCHS, batch: int = BATCH, lr: float = LR,
             t_steps: int = T_STEPS, n_gen: int = N_GEN,
             n_val_eval: int = N_VAL_EVAL, n_nn_ref: int = N_NN_REF,
             n_recon: int = N_RECON, verbose: bool = False) -> dict:
    """Train one arm at one seed and evaluate it on validation only."""
    x_train = torch.as_tensor(x_train)
    r_train = torch.as_tensor(r_train)
    x_val = torch.as_tensor(x_val)
    r_val = torch.as_tensor(r_val)
    model = BookDDPM(t_steps=t_steps, seed=seed, **CAPACITY_ENVELOPE[name])
    n_params = count_params(model)
    n_steps_per_epoch = (len(x_train) + batch - 1) // batch
    optimizer_steps = epochs * n_steps_per_epoch
    t0 = time.time()
    hist, _ = train_ddpm(model, x_train, r_train, epochs=epochs,
                         batch=batch, lr=lr, seed=seed, verbose=verbose)
    train_seconds = time.time() - t0

    val_idx = _strided(len(x_val), n_val_eval)
    train_ref_idx = _strided(len(x_train), n_nn_ref)
    recon_train_idx = _strided(len(x_train), n_recon)
    recon_val_idx = _strided(len(x_val), n_recon)
    x_val_sub = x_val[val_idx]
    r_val_sub = r_val[val_idx]

    val_losses = eval_denoising_losses(model, x_val_sub, r_val_sub,
                                       seed=EVAL_SEED, batch=batch)
    val_loss = float(np.mean(val_losses))

    t_s0 = time.perf_counter()
    gen = model.sample(r_val_sub, seed=GEN_SEED + seed).numpy()
    sample_seconds = time.perf_counter() - t_s0
    sample_per_1k = sample_seconds / max(len(gen), 1) * 1000.0

    mt = marginal_table(x_val_sub.numpy(), gen)
    val_mean_w1 = mt["mean_w1"]

    train_ref = x_train[train_ref_idx].numpy()
    nn_to_train = float(np.mean(
        nearest_neighbor_distances(gen, train_ref)))
    nn_to_val = float(np.mean(
        nearest_neighbor_distances(gen, x_val_sub.numpy())))
    nn_real_val_to_train = float(np.mean(
        nearest_neighbor_distances(x_val_sub.numpy(), train_ref)))
    diversity = pairwise_diversity(gen)
    real_diversity = pairwise_diversity(x_val_sub.numpy())
    recon_t = min(RECON_T_STEP, t_steps - 1)
    recon_train = reconstruction_probe(
        model, x_train[recon_train_idx], r_train[recon_train_idx],
        t_step=recon_t, seed=RECON_SEED)
    recon_val = reconstruction_probe(
        model, x_val[recon_val_idx], r_val[recon_val_idx],
        t_step=recon_t, seed=RECON_SEED)

    return {
        "arm": name, "seed": seed, "params": n_params,
        "epochs": epochs, "batch": batch, "lr": lr,
        "t_steps": t_steps, "optimizer_steps": optimizer_steps,
        "train_seconds": train_seconds,
        "train_loss_history": [float(v) for v in hist],
        "final_train_loss": float(hist[-1]),
        "convergence_epoch": convergence_epoch(hist),
        "val_loss": val_loss,
        "val_losses": val_losses,
        "val_mean_w1": val_mean_w1,
        "n_val_eval": int(len(val_idx)),
        "n_generated": int(len(gen)),
        "gen_windows": gen,
        "sample_seconds_per_1k": sample_per_1k,
        "total_nfe": total_network_evals(optimizer_steps, len(gen),
                                         t_steps),
        "nn_to_train": nn_to_train,
        "nn_to_val": nn_to_val,
        "nn_real_val_to_train": nn_real_val_to_train,
        "diversity": diversity,
        "real_diversity": real_diversity,
        "recon_train": recon_train,
        "recon_val": recon_val,
        "recon_gap": recon_train - recon_val,
        "recon_t_step": recon_t,
    }


def _jsonable_cell(cell: dict) -> dict:
    """Cell summary for the results file: arrays dropped or listed."""
    out = {}
    for k, v in cell.items():
        if k == "gen_windows":
            continue
        if isinstance(v, np.ndarray):
            out[k] = [float(x) for x in v]
        else:
            out[k] = v
    return out


def _checkpoint_paths(checkpoint_dir, round_name, arm, seed):
    base = os.path.join(checkpoint_dir, f"{round_name}-{arm}-{seed}")
    return base + ".json", base + "-gen.npy"


def _load_checkpoint_cell(checkpoint_dir, round_name, arm, seed):
    """Reload a previously saved cell, or None when it is absent.

    The JSON summary carries the scalar metrics and per-window
    validation losses; the generated windows ride alongside as a
    float32 .npy so the blocked W1 intervals can be recomputed
    exactly on a resumed run.
    """
    if not checkpoint_dir:
        return None
    jpath, npath = _checkpoint_paths(checkpoint_dir, round_name, arm, seed)
    if not (os.path.exists(jpath) and os.path.exists(npath)):
        return None
    with open(jpath) as f:
        cell = json.load(f)
    cell["val_losses"] = np.asarray(cell["val_losses"], dtype=np.float64)
    cell["gen_windows"] = np.load(npath)
    return cell


def _save_checkpoint_cell(checkpoint_dir, round_name, arm, seed, cell):
    if not checkpoint_dir:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    jpath, npath = _checkpoint_paths(checkpoint_dir, round_name, arm, seed)
    np.save(npath, np.asarray(cell["gen_windows"], dtype=np.float32))
    with open(jpath, "w") as f:
        json.dump(_jsonable_cell(cell), f)


def _aggregate(cells_by_seed: dict) -> dict:
    """Seed means for the headline metrics of one arm in one round."""
    cells = list(cells_by_seed.values())
    return {
        "seeds": sorted(cells_by_seed),
        "params": cells[0]["params"],
        "epochs": sorted({c["epochs"] for c in cells}),
        "optimizer_steps": sorted({c["optimizer_steps"] for c in cells}),
        "final_train_loss_mean": float(np.mean(
            [c["final_train_loss"] for c in cells])),
        "val_loss_mean": float(np.mean([c["val_loss"] for c in cells])),
        "val_mean_w1_mean": float(np.mean(
            [c["val_mean_w1"] for c in cells])),
        "sample_seconds_per_1k_mean": float(np.mean(
            [c["sample_seconds_per_1k"] for c in cells])),
        "train_seconds_mean": float(np.mean(
            [c["train_seconds"] for c in cells])),
        "total_nfe_mean": float(np.mean([c["total_nfe"] for c in cells])),
        "nn_to_train_mean": float(np.mean(
            [c["nn_to_train"] for c in cells])),
        "nn_to_val_mean": float(np.mean([c["nn_to_val"] for c in cells])),
        "nn_real_val_to_train_mean": float(np.mean(
            [c["nn_real_val_to_train"] for c in cells])),
        "diversity_mean": float(np.mean([c["diversity"] for c in cells])),
        "real_diversity_mean": float(np.mean(
            [c["real_diversity"] for c in cells])),
        "recon_train_mean": float(np.mean(
            [c["recon_train"] for c in cells])),
        "recon_val_mean": float(np.mean([c["recon_val"] for c in cells])),
        "recon_gap_mean": float(np.mean(
            [c["recon_gap"] for c in cells])),
    }


def run_study(x_train, r_train, x_val, r_val, seeds=SEEDS,
              epochs: int = EPOCHS, batch: int = BATCH, lr: float = LR,
              t_steps: int = T_STEPS, n_gen: int = N_GEN,
              n_val_eval: int = N_VAL_EVAL, n_nn_ref: int = N_NN_REF,
              n_recon: int = N_RECON, n_boot: int = N_BOOT,
              data_sha256: str = "", split_manifest_sha256: str = "",
              checkpoint_dir: str | None = None) -> dict:
    """Two-round scaling study on explicit train/validation arrays.

    No test input exists in this signature: the frozen test split
    cannot enter by construction.

    ``checkpoint_dir`` (optional) persists each finished cell as it
    completes and reloads it on a later call with the same directory,
    so an interrupted run resumes instead of restarting: a resumed
    cell is the exact same seed and config, and the results file
    records which cells were resumed.
    """
    resumed = []

    def _cell(round_name, name, seed, epochs):
        hit = _load_checkpoint_cell(checkpoint_dir, round_name, name, seed)
        if hit is not None:
            print(f"  {round_name} {name} seed {seed} "
                  f"({epochs} epochs) ... resumed from checkpoint",
                  flush=True)
            resumed.append(f"{round_name}/{name}/seed-{seed}")
            return hit
        print(f"  {round_name} {name} seed {seed} ...", flush=True)
        cell = run_cell(name, x_train, r_train, x_val, r_val, seed=seed,
                        epochs=epochs, batch=batch, lr=lr,
                        t_steps=t_steps, n_gen=n_gen,
                        n_val_eval=n_val_eval, n_nn_ref=n_nn_ref,
                        n_recon=n_recon)
        _save_checkpoint_cell(checkpoint_dir, round_name, name, seed, cell)
        return cell

    rounds = {}
    # Round 1: equal training compute.
    eq_cells = {name: {} for name in CAPACITY_ENVELOPE}
    for seed in seeds:
        for name in CAPACITY_ENVELOPE:
            eq_cells[name][seed] = _cell(
                "equal_compute", name, seed, epochs)
    rounds["equal_compute"] = {
        "comparison": "equal training compute: same epochs, batch, "
                      "learning rate, optimizer, and seeds",
        "cells": {name: {s: _jsonable_cell(c)
                         for s, c in cells.items()}
                  for name, cells in eq_cells.items()},
        "aggregate": {name: _aggregate(cells)
                      for name, cells in eq_cells.items()},
    }
    # Convergence epochs from round 1, median across seeds.
    conv_epochs = {}
    for name, cells in eq_cells.items():
        eps = [c["convergence_epoch"] + 1 for c in cells.values()]
        conv_epochs[name] = int(np.median(eps))
    print(f"  convergence epochs (median across seeds): "
          f"{conv_epochs}", flush=True)
    # Round 2: convergence-normalized training.
    cv_cells = {name: {} for name in CAPACITY_ENVELOPE}
    for seed in seeds:
        for name in CAPACITY_ENVELOPE:
            cv_cells[name][seed] = _cell(
                "convergence_normalized", name, seed,
                conv_epochs[name])
    rounds["convergence_normalized"] = {
        "comparison": "convergence-normalized: each arm trained "
                      "for its median epochs-to-convergence from "
                      "round 1 (first epoch within 5 percent of "
                      "its final equal-compute train loss)",
        "convergence_epochs": conv_epochs,
        "cells": {name: {s: _jsonable_cell(c)
                         for s, c in cells.items()}
                  for name, cells in cv_cells.items()},
        "aggregate": {name: _aggregate(cells)
                      for name, cells in cv_cells.items()},
    }
    # Blocked intervals and the preregistered promotion verdict.
    # W1 resampling needs the shared real validation subsample.
    val_idx = _strided(len(x_val), n_val_eval)
    real_val_sub = np.asarray(x_val)[val_idx]
    per_round = {}
    for rnd_name, cells in (("equal_compute", eq_cells),
                            ("convergence_normalized", cv_cells)):
        base = cells["base"]
        large = cells["large"]
        seeds_sorted = sorted(base)
        diffs = np.mean(
            [np.asarray(large[s]["val_losses"], dtype=np.float64)
             - np.asarray(base[s]["val_losses"], dtype=np.float64)
             for s in seeds_sorted], axis=0)
        val_loss_ci = block_bootstrap_diff_ci(
            diffs, np.zeros_like(diffs), n_boot=n_boot)
        w1_ci = block_bootstrap_w1_diff_ci(
            real_val_sub,
            [base[s]["gen_windows"] for s in seeds_sorted],
            [large[s]["gen_windows"] for s in seeds_sorted],
            n_boot=n_boot)
        per_round[rnd_name] = {
            "val_loss_ci": [float(v) for v in val_loss_ci],
            "val_w1_ci": [float(v) for v in w1_ci],
        }
    eq_agg = rounds["equal_compute"]["aggregate"]
    feasibility = {
        "params_ratio": eq_agg["large"]["params"]
        / eq_agg["base"]["params"],
        "sample_ratio": eq_agg["large"]["sample_seconds_per_1k_mean"]
        / eq_agg["base"]["sample_seconds_per_1k_mean"],
        "params_ratio_max": PARAMS_RATIO_MAX,
        "sample_ratio_max": SAMPLE_RATIO_MAX,
    }
    # Promotion needs both CIs below zero in BOTH rounds.
    verdict = promotion_verdict(
        "large",
        {"val_loss_ci": (
            max(per_round["equal_compute"]["val_loss_ci"][0],
                per_round["convergence_normalized"]["val_loss_ci"][0]),
            max(per_round["equal_compute"]["val_loss_ci"][1],
                per_round["convergence_normalized"]["val_loss_ci"][1])),
         "val_w1_ci": (
            max(per_round["equal_compute"]["val_w1_ci"][0],
                per_round["convergence_normalized"]["val_w1_ci"][0]),
            max(per_round["equal_compute"]["val_w1_ci"][1],
                per_round["convergence_normalized"]["val_w1_ci"][1]))},
        feasibility)
    verdict["per_round"] = per_round
    verdict["feasibility"] = feasibility
    return {
        "protocol": "scaling-study",
        "protocol_version": 1,
        "preregistration": {
            "capacity_envelope": CAPACITY_ENVELOPE,
            "seeds": list(seeds),
            "epochs_equal_compute": epochs,
            "batch": batch,
            "lr": lr,
            "t_steps": t_steps,
            "n_val_eval": n_val_eval,
            "n_nn_ref": n_nn_ref,
            "n_recon": n_recon,
            "recon_t_step": RECON_T_STEP,
            "block": BLOCK,
            "n_boot": n_boot,
            "convergence_tol": CONVERGENCE_TOL,
            "promotion_rule": "large is promoted only if, in BOTH "
                              "rounds, its validation denoising "
                              "loss and validation mean W1 beat "
                              "base with blocked 95 percent "
                              "intervals entirely below zero, and "
                              "its parameter count stays within "
                              f"{PARAMS_RATIO_MAX}x of base and "
                              "its sampling cost within "
                              f"{SAMPLE_RATIO_MAX}x of base",
        },
        "optimizer": {"name": "Adam", "lr": lr, "batch": batch},
        "data_sha256": data_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "test_split_touched": False,
        "checkpoint_dir": checkpoint_dir,
        "resumed_cells": resumed,
        "rounds": rounds,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.abspath(DATA))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "scaling_study_results.json"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--n-val-eval", type=int, default=N_VAL_EVAL)
    ap.add_argument("--checkpoint-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    torch.set_num_threads(2)
    ds = load_dataset(os.path.abspath(args.data))
    ensure_publishable(ds)
    # Provenance rests on the NPZ-embedded split manifest, as in the
    # diffusion/latency ablation scripts: the committed
    # data/split_manifest.json describes the original 8-session
    # corpus, while the study corpus is the expanded 28-session one
    # whose manifest lives embedded in the NPZ. The embedded SHA is
    # recorded in the results file.
    print(f"split-manifest SHA-256: {ds.manifest_sha256}")
    print(f"train {len(ds.train.x)} / val {len(ds.val.x)} "
          f"/ test {len(ds.test.x)} windows (test untouched)")
    seeds = tuple(int(s) for s in args.seeds.split(",") if s != "")
    result = run_study(
        torch.from_numpy(ds.train.x), torch.from_numpy(ds.train.regime),
        torch.from_numpy(ds.val.x), torch.from_numpy(ds.val.regime),
        seeds=seeds, epochs=args.epochs, n_gen=args.n_gen,
        n_val_eval=args.n_val_eval,
        data_sha256=sha256_file(os.path.abspath(args.data)),
        split_manifest_sha256=ds.manifest_sha256,
        checkpoint_dir=args.checkpoint_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("\n=== scaling study (seed means) ===")
    for rnd_name, rnd in result["rounds"].items():
        print(f"-- {rnd_name} --")
        for name, agg in rnd["aggregate"].items():
            print(f"  {name:<6} params {agg['params']:>7} "
                  f"train loss {agg['final_train_loss_mean']:.4f} "
                  f"val loss {agg['val_loss_mean']:.4f} "
                  f"val W1 {agg['val_mean_w1_mean']:.4f} "
                  f"sample {agg['sample_seconds_per_1k_mean']:.1f}s/1k "
                  f"NN train {agg['nn_to_train_mean']:.3f} "
                  f"(real val {agg['nn_real_val_to_train_mean']:.3f}) "
                  f"recon gap {agg['recon_gap_mean']:+.4f} "
                  f"diversity {agg['diversity_mean']:.3f} "
                  f"(real {agg['real_diversity_mean']:.3f})")
    v = result["verdict"]
    print(f"\nVERDICT: large promoted: {v['promoted']}")
    for r in v["reasons"]:
        print(f"  - {r}")
    print(f"results -> {args.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
