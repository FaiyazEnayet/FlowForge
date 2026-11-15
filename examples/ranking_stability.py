"""Preregistered ranking-stability study for FF-R08.

Question: does the ranking of generative methods by marginal
1-Wasserstein distance agree with the ranking by joint and
conditional lag quality and by a powered held-out discriminator? The
study answers it by scoring every policy under one frozen protocol
and comparing the three rankings directly, with Kendall agreement
and Pareto domination, instead of reporting any single ranking as
the verdict.

Preregistered before any run of this harness (frozen design):

- Corpus: the expanded 28-session corpus (8 original chunks plus 20
  B1-QC 4 h sessions, Nov 9-12 2026). Train sessions 0-19 (18,468
  windows), val sessions 20-23 (5,712), test sessions 24-27 (5,712).
  Split-manifest SHA
  b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd74fb861e1446ae581e.
  The main entry point refuses any other manifest: the design is
  locked to this corpus. Fixed 1,000-window strided val/test
  subsamples, identical across every policy.
- Model factorial: DDPM size {base ch=96 (221,767 parameters),
  small ch=48 (75,751 parameters)} crossed with EMA {off, on
  (decay 0.999)}. Each cell trains 6 epochs, batch 32, training
  seed 0. One ancestral (100 NFE) sample per cell, plus a DDIM 10
  NFE budget point on the base raw weights. VAE comparator at
  lr 2e-4, 25 epochs, seed 0. Regime-matched chunk-copy as reference.
  Generation seeds (0, 100, 200) for every policy. Zero missing
  cells are allowed.
- Metrics per policy and seed: marginal mean W1 (val primary, test
  confirmation), joint fidelity (signed-flow ACF error at lags 1-5
  plus cross-feature correlation error), held-out discriminator AUC
  (fit on val only, scored on test only, 30 epochs, identical
  protocol for every policy and seed), TSTR flow-direction accuracy.
- Rankings: policies ranked by val mean W1, by val joint error, by
  test AUC, and by TSTR accuracy. Kendall tau-b between the W1
  ranking and each other ranking is the primary stability result,
  computed over the six trained-model policies; the same taus with
  the chunk-copy reference included are reported as a secondary
  view. Pareto domination is over (val W1, val joint error, test
  AUC). Paired blocked CIs use block lengths 1/5/10/20.
- The test split is confirmation only and never picks a ranking.
  All nulls are preserved: a ranking that does not move, a tau
  near zero, and a CI that includes zero are reported as measured.
- No size effect is claimed from the unmatched historical
  221767/480775 variants: those studies used different split
  populations and capacities, and historical weight tensors were
  never saved, so old figures are not a controlled architecture
  experiment. Only the within-study factorial is a controlled size
  contrast.

Data caveat: the recorded dataset (data/book_windows.npz) is
gitignored, so these numbers describe the protocol on the dataset
whose manifest SHA is recorded in the output, not a committed
artifact. A single training seed (0) is a stated limitation.
"""

import argparse
import datetime
import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from itertools import combinations

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))

from flowforge.baselines import (  # noqa: E402
    RegimeMatchedChunkCopyBaseline, segment_ids_from_windows)
from flowforge.data import FEATURE_NAMES, load_dataset  # noqa: E402
from flowforge.evaluate import (  # noqa: E402
    acf, comparison_report, marginal_table)
from flowforge.models import (  # noqa: E402
    BookDDPM, CondVAE, count_params, sample_with_ema, train_ddpm,
    train_vae)
from flowforge.samplers import sample_ddim  # noqa: E402

SEED = 0
GEN_SEEDS = (0, 100, 200)
N_EVAL = 1000
EPOCHS = 6
VAE_EPOCHS = 25
VAE_LR = 2e-4
BATCH = 32
T_STEPS = 100
CH_BASE = 96
CH_SMALL = 48
EMA_DECAY = 0.999
DDIM_NFE = 10
DISC_EPOCHS = 30
TSTR_EPOCHS = 300
N_BOOT = 100
BLOCK_LENGTHS = (1, 5, 10, 20)
LAGS = (1, 2, 3, 4, 5)
EXPECTED_MANIFEST_SHA256 = (
    "b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd74fb861e1446ae581e")

DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "..", "data", "book_windows.npz")

# The preregistered policy grid. "ddpm" is the base raw ancestral
# reference the published parity schema requires; the other DDPM
# cells are the size x EMA factorial; "ddim_10" is the sample-budget
# point on the base raw weights; "vae" and "chunk_copy" are the
# comparators the parity schema requires.
POLICIES = ("ddpm", "ddpm_ema", "ddpm_small", "ddpm_small_ema",
            "ddim_10", "vae", "chunk_copy")
TRAINED_POLICIES = ("ddpm", "ddpm_ema", "ddpm_small",
                    "ddpm_small_ema", "ddim_10", "vae")
FLOW_J = FEATURE_NAMES.index("signed_flow")


def hardware_record():
    """Hardware and thread settings that decide the run numbers."""
    return {
        "device": "cpu",
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


# ------------------------------------------------------------ ranking math

def kendall_tau_b(x, y) -> float:
    """Kendall tau-b between two orderings, with tie correction.

    Returns nan when either ordering is fully tied (no comparable
    pair) or when fewer than two items are given. Raises ValueError
    on mismatched lengths.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError(
            f"kendall_tau_b needs equal lengths, got {len(x)} and "
            f"{len(y)}")
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[j] - x[i])
            dy = np.sign(y[j] - y[i])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt((concordant + discordant + ties_x)
                    * (concordant + discordant + ties_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def rank_scores(values, lower_is_better=True) -> dict:
    """Average ranks (1 is best) for a {method: score} mapping.

    Ties share the average of the positions they span.
    """
    items = sorted(values.items(),
                   key=lambda kv: kv[1] if lower_is_better else -kv[1])
    ranks: dict = {}
    i = 0
    while i < len(items):
        j = i
        while (j + 1 < len(items)
               and items[j + 1][1] == items[i][1]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[items[k][0]] = float(avg)
        i = j + 1
    return ranks


def pareto_frontier_rows(rows) -> set:
    """Non-dominated method names over (W1, joint error, AUC).

    Every metric is lower-is-better. A row is dominated when another
    row is no worse on all three metrics and strictly better on at
    least one. Rows are dicts with keys "method", "val_mean_w1",
    "val_joint_error", "auc_test_mean".
    """
    keys = ("val_mean_w1", "val_joint_error", "auc_test_mean")
    names = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if all(other[k] <= row[k] for k in keys) and any(
                    other[k] < row[k] for k in keys):
                dominated = True
                break
        if not dominated:
            names.add(row["method"])
    return names


# ------------------------------------------------- preregistration guards

def check_cells_present(cells, policies, seeds) -> int:
    """Refuse a run whose preregistered model x seed grid is incomplete.

    ``cells`` is a set of (policy, seed) pairs. Missing cells and
    unexpected cells are both errors: a silent substitution would
    break the preregistration. Returns 0 (the missing count) when
    the grid is exact.
    """
    expected = {(p, s) for p in policies for s in seeds}
    missing = sorted(expected - set(cells))
    extra = sorted(set(cells) - expected)
    if missing:
        raise ValueError(
            f"preregistered cells missing: {missing}: 0 missing "
            f"cells are allowed")
    if extra:
        raise ValueError(
            f"unexpected cells outside the preregistered grid: {extra}")
    return 0


def check_no_fit_test_overlap(fit_session_ids,
                              test_session_ids) -> bool:
    """Refuse any overlap between discriminator fit and test sessions.

    Returns True when the two session sets are disjoint.
    """
    overlap = sorted(set(fit_session_ids)
                     & set(test_session_ids))
    if overlap:
        raise ValueError(
            f"classifier fit/test session overlap: {overlap}: the "
            f"discriminator must fit on val sessions only and score "
            f"on test sessions only")
    return True


# --------------------------------------------------------- joint fidelity

def _window_acf(windows, lags=LAGS) -> np.ndarray:
    """Per-window signed-flow ACF at each lag: (n_windows, len(lags))."""
    return np.array(
        [[acf(w[:, FLOW_J], lag) for lag in lags] for w in windows],
        dtype=np.float64)


def _corr_matrix(windows) -> np.ndarray:
    """Cross-feature correlation matrix over flattened windows.

    A constant feature (zero variance) yields a nan row in corrcoef;
    it is recorded as zero correlation, since a constant carries no
    linear association with anything.
    """
    flat = np.asarray(windows, dtype=np.float64).reshape(
        -1, windows.shape[2])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(flat, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def joint_fidelity(real, gen, lags=LAGS) -> dict:
    """Joint and conditional lag quality of generated windows.

    - ``lag_acf_error``: mean over lags of the absolute difference
      between the generated and real mean signed-flow ACF. This is
      the conditional lag quality: it sees whether the generated
      series carries the real temporal dependence.
    - ``corr_error``: mean absolute off-diagonal difference between
      the generated and real cross-feature correlation matrices.
      This is the joint quality: it sees whether the generated
      features move together the way the real ones do.
    - ``joint_error``: the mean of the two, the single joint score
      the rankings use. Both parts are absolute errors in
      correlation units, so the mean keeps them on one scale.
    """
    real_acf = _window_acf(real, lags).mean(axis=0)
    gen_acf = _window_acf(gen, lags).mean(axis=0)
    lag_err = float(np.mean(np.abs(gen_acf - real_acf)))
    corr_r = _corr_matrix(real)
    corr_g = _corr_matrix(gen)
    mask = ~np.eye(corr_r.shape[0], dtype=bool)
    corr_err = float(np.mean(np.abs(corr_r[mask] - corr_g[mask])))
    return {
        "lag_acf_error": lag_err,
        "corr_error": corr_err,
        "joint_error": float((lag_err + corr_err) / 2),
        "real_acf": [float(v) for v in real_acf],
        "gen_acf": [float(v) for v in gen_acf],
        "lags": [int(l) for l in lags],
    }


def _block_resample_indices(n, block_len, rng) -> np.ndarray:
    """Chronological moving-block resample indices for n positions."""
    if n == 0:
        return np.empty(0, dtype=np.int64)
    block_len = int(min(max(1, block_len), n))
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    return np.concatenate(
        [np.arange(s, s + block_len) for s in starts])[:n]


def _joint_error_on(real_acf, gen_acf, real_flat, gen_flat,
                    idx, lags_n) -> float:
    """Joint error on a block-resampled window index vector."""
    real_acf_mean = real_acf[idx].mean(axis=0)
    gen_acf_mean = gen_acf[idx].mean(axis=0)
    lag_err = float(np.mean(np.abs(gen_acf_mean - real_acf_mean)))
    steps = real_flat.shape[0] // len(real_acf)
    flat_idx = (idx[:, None] * steps
                + np.arange(steps)[None, :]).ravel()
    corr_r = np.nan_to_num(
        np.corrcoef(real_flat[flat_idx], rowvar=False),
        nan=0.0, posinf=0.0, neginf=0.0)
    corr_g = np.nan_to_num(
        np.corrcoef(gen_flat[flat_idx], rowvar=False),
        nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.eye(corr_r.shape[0], dtype=bool)
    corr_err = float(np.mean(np.abs(corr_r[mask] - corr_g[mask])))
    return float((lag_err + corr_err) / 2)


def paired_joint_diff_ci(real, gen_by_method_by_seed,
                         block_lengths=BLOCK_LENGTHS, n_boot=N_BOOT,
                         seeds=(0,)) -> list:
    """Paired block-bootstrap intervals for joint-error differences.

    ``gen_by_method_by_seed`` maps each method to a list of generated
    arrays, one per generation seed, each aligned with ``real`` in
    window order. Every replicate draws ONE chronological block
    resample and applies it to the real windows and to every
    method's generated windows, so the pairing is exact; draws are
    pooled over the generation seeds, matching the W1 treatment in
    flowforge.evaluate. Returns rows with block_len, methods, point
    diff, CI, n_boot, and seeds.
    """
    methods = sorted(gen_by_method_by_seed)
    n_seeds = len(next(iter(gen_by_method_by_seed.values())))
    real_acf = _window_acf(real, LAGS)
    real_flat = np.asarray(real, dtype=np.float64).reshape(
        -1, real.shape[2])
    gen_acf = {m: [_window_acf(g, LAGS)
                   for g in gen_by_method_by_seed[m]]
               for m in methods}
    gen_flat = {m: [np.asarray(g, dtype=np.float64).reshape(
        -1, g.shape[2]) for g in gen_by_method_by_seed[m]]
        for m in methods}
    point = {m: float(np.mean([
        joint_fidelity(real, g)["joint_error"]
        for g in gen_by_method_by_seed[m]])) for m in methods}
    rows = []
    for block_len in block_lengths:
        pooled: dict = {}
        for s in range(n_seeds):
            for seed in seeds:
                rng = np.random.default_rng(
                    seed + 1000 * int(block_len) + 7919 * s)
                for _ in range(n_boot):
                    idx = _block_resample_indices(
                        len(real), int(block_len), rng)
                    joint = {m: _joint_error_on(
                        real_acf, gen_acf[m][s], real_flat,
                        gen_flat[m][s], idx, len(LAGS))
                        for m in methods}
                    for a, b in combinations(methods, 2):
                        pooled.setdefault((a, b), []).append(
                            joint[a] - joint[b])
        for pair, vals in sorted(pooled.items()):
            vals = np.asarray(vals, dtype=np.float64)
            rows.append({
                "block_len": int(block_len),
                "methods": [pair[0], pair[1]],
                "diff": point[pair[0]] - point[pair[1]],
                "ci": [float(np.quantile(vals, 0.025)),
                       float(np.quantile(vals, 0.975))],
                "n_boot": int(len(vals)),
                "seeds": [int(s) for s in seeds],
            })
    return rows


# ------------------------------------------------------------- the study

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


def _clean_float(v):
    """JSON-safe float: nan becomes None so the record stays valid."""
    v = float(v)
    return None if not np.isfinite(v) else v


def run_ranking_stability(splits, train_meta, policies=None,
                          gen_seeds=GEN_SEEDS, train_epochs=EPOCHS,
                          vae_epochs=VAE_EPOCHS, vae_lr=VAE_LR,
                          t_steps=T_STEPS, ch=CH_BASE,
                          small_ch=CH_SMALL, ema_decay=EMA_DECAY,
                          ddim_nfe=DDIM_NFE, batch_size=BATCH,
                          seed=SEED, n_eval=N_EVAL,
                          disc_epochs=DISC_EPOCHS,
                          tstr_epochs=TSTR_EPOCHS, n_boot=N_BOOT,
                          block_lengths=BLOCK_LENGTHS,
                          run_id="ff-r08", verbose=True):
    """Run the preregistered ranking-stability study on in-memory splits.

    ``splits`` maps "train"/"val"/"test" to (x, regime) torch tensors
    in (windows, window, features) / (windows, 2) layout. The val and
    test splits given here are the scored subsamples; the test split
    is scored only and never used to pick a ranking. ``train_meta``
    carries the numpy training windows, regimes, and session/segment
    ids the chunk-copy baseline needs, plus "fit_session_ids" (the
    val sessions the discriminator fits on) and "test_session_ids"
    (the per-window test session ids). Returns the record dict.
    """
    if policies is None:
        policies = list(POLICIES)
    policies = list(policies)
    x_train, r_train = splits["train"]
    x_val, r_val = splits["val"]
    x_test, r_test = splits["test"]
    real_val = x_val.numpy()
    real_test = x_test.numpy()
    n_val, n_test = len(x_val), len(x_test)
    n_windows = n_val + n_test
    r_val_np = r_val.numpy()
    r_test_np = r_test.numpy()

    # ---- train the factorial cells (one training seed, stated limit)
    models = {}
    train_losses = {}
    params = {}
    if any(p in policies for p in
           ("ddpm", "ddpm_ema", "ddpm_small", "ddpm_small_ema",
            "ddim_10")):
        base = BookDDPM(t_steps=t_steps, ch=ch, n_blocks=3,
                        run_id=f"{run_id}-ddpm")
        params["ddpm"] = int(count_params(base))
        t0 = time.time()
        hist, _ = train_ddpm(base, x_train, r_train,
                             epochs=train_epochs, batch=batch_size,
                             seed=seed, verbose=False,
                             run_id=f"{run_id}-ddpm")
        train_losses["ddpm"] = float(hist[-1])
        if verbose:
            print(f"  ddpm (ch={ch}) trained {train_epochs} epochs in "
                  f"{(time.time() - t0) / 60:.1f}m "
                  f"(final loss {hist[-1]:.4f})", flush=True)
        models["ddpm"] = (base, None)
    ema_models = {}
    for name, n_ch in (("ddpm_ema", ch), ("ddpm_small_ema", small_ch)):
        if name not in policies:
            continue
        m = BookDDPM(t_steps=t_steps, ch=n_ch, n_blocks=3,
                     run_id=f"{run_id}-{name}")
        params[name] = int(count_params(m))
        t0 = time.time()
        hist, ema = train_ddpm(m, x_train, r_train,
                               epochs=train_epochs, batch=batch_size,
                               seed=seed, verbose=False,
                               ema_decay=ema_decay,
                               run_id=f"{run_id}-{name}")
        train_losses[name] = float(hist[-1])
        if verbose:
            print(f"  {name} (ch={n_ch}, EMA {ema_decay}) trained "
                  f"{train_epochs} epochs in "
                  f"{(time.time() - t0) / 60:.1f}m "
                  f"(final loss {hist[-1]:.4f})", flush=True)
        ema_models[name] = (m, ema)
    for name, n_ch in (("ddpm_small", small_ch),):
        if name not in policies:
            continue
        m = BookDDPM(t_steps=t_steps, ch=n_ch, n_blocks=3,
                     run_id=f"{run_id}-{name}")
        params[name] = int(count_params(m))
        t0 = time.time()
        hist, _ = train_ddpm(m, x_train, r_train,
                             epochs=train_epochs, batch=batch_size,
                             seed=seed, verbose=False,
                             run_id=f"{run_id}-{name}")
        train_losses[name] = float(hist[-1])
        if verbose:
            print(f"  {name} (ch={n_ch}) trained {train_epochs} "
                  f"epochs in {(time.time() - t0) / 60:.1f}m "
                  f"(final loss {hist[-1]:.4f})", flush=True)
        models[name] = (m, None)
    vae = None
    if "vae" in policies:
        vae = CondVAE(run_id=f"{run_id}-vae")
        params["vae"] = int(count_params(vae))
        t0 = time.time()
        vae_hist = train_vae(vae, x_train, r_train,
                             epochs=vae_epochs, seed=seed,
                             verbose=False, run_id=f"{run_id}-vae",
                             lr=vae_lr)
        train_losses["vae"] = float(vae_hist[-1])
        if verbose:
            print(f"  vae trained {vae_epochs} epochs in "
                  f"{(time.time() - t0) / 60:.1f}m "
                  f"(final loss {vae_hist[-1]:.4f})", flush=True)
    chunk_copy = None
    if "chunk_copy" in policies:
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

    for m, _ in list(models.values()) + list(ema_models.values()):
        m.eval()
    if vae is not None:
        vae.eval()

    def _gen_ddpm(seed):
        m, _ = models["ddpm"]
        return (m.sample(r_val, seed=seed, batch_size=batch_size)
                .numpy(),
                m.sample(r_test, seed=seed + 1,
                         batch_size=batch_size).numpy())

    def _gen_ddim(seed):
        m, _ = models["ddpm"]
        return (sample_ddim(m, r_val, nfe=ddim_nfe, eta=0.0,
                            seed=seed, batch_size=batch_size).numpy(),
                sample_ddim(m, r_test, nfe=ddim_nfe, eta=0.0,
                            seed=seed + 1,
                            batch_size=batch_size).numpy())

    def _gen_ema(name):
        def gen(seed):
            m, ema = ema_models[name]
            return (sample_with_ema(m, ema, r_val, seed=seed,
                                    batch_size=batch_size).numpy(),
                    sample_with_ema(m, ema, r_test, seed=seed + 1,
                                    batch_size=batch_size).numpy())
        return gen

    def _gen_small(name):
        def gen(seed):
            m, _ = models[name]
            return (m.sample(r_val, seed=seed,
                             batch_size=batch_size).numpy(),
                    m.sample(r_test, seed=seed + 1,
                             batch_size=batch_size).numpy())
        return gen

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

    generators = {}
    nfe = {}
    if "ddpm" in policies:
        generators["ddpm"] = _gen_ddpm
        nfe["ddpm"] = t_steps
    if "ddim_10" in policies:
        generators["ddim_10"] = _gen_ddim
        nfe["ddim_10"] = ddim_nfe
    for name in ("ddpm_ema", "ddpm_small_ema"):
        if name in policies:
            generators[name] = _gen_ema(name)
            nfe[name] = t_steps
    if "ddpm_small" in policies:
        generators["ddpm_small"] = _gen_small("ddpm_small")
        nfe["ddpm_small"] = t_steps
    if "vae" in policies:
        generators["vae"] = _gen_vae
        nfe["vae"] = 1
    if "chunk_copy" in policies:
        nfe["chunk_copy"] = 0

    runs = []
    gen_val_by_method = {p: [] for p in policies}
    gen_test_by_method = {p: [] for p in policies}
    for name in policies:
        for gseed in gen_seeds:
            if name == "chunk_copy":
                gv, gt, cost = _gen_chunk_copy(gseed)
            else:
                gv, gt, cost = _timed_generate(
                    lambda gseed=gseed: generators[name](gseed),
                    nfe[name], n_windows)
            runs.append({"method": name, "kind": "conditional",
                         "seed": gseed, "gen_val": gv, "gen_test": gt,
                         "cost": cost})
            gen_val_by_method[name].append(gv)
            gen_test_by_method[name].append(gt)
        if verbose:
            print(f"  {name:<16} generated {len(gen_seeds)} seeds",
                  flush=True)

    # ---- preregistration guards: exact grid, zero fit/test overlap
    cells = {(r["method"], r["seed"]) for r in runs}
    n_missing = check_cells_present(cells, policies, gen_seeds)
    fit_sessions = list(train_meta.get("fit_session_ids", []))
    test_sessions = sorted(set(
        np.asarray(train_meta["test_session_ids"]).tolist()))
    check_no_fit_test_overlap(fit_sessions, test_sessions)
    overlap = sorted(set(fit_sessions) & set(test_sessions))
    n_test_sessions = len(test_sessions)

    # ---- the parity comparison: identical discriminator and TSTR
    comparison = comparison_report(
        real_test, real_val, runs,
        block_lengths=block_lengths, n_boot=n_boot,
        disc_epochs=disc_epochs, tstr_epochs=tstr_epochs,
        protocol_seed=seed, n_test_sessions=n_test_sessions)

    # ---- per-policy val metrics (primary) and test metrics
    table = {}
    for name in policies:
        val_w1 = [marginal_table(real_val, gv)["mean_w1"]
                  for gv in gen_val_by_method[name]]
        val_joint = [joint_fidelity(real_val, gv)
                     for gv in gen_val_by_method[name]]
        test_w1 = [marginal_table(real_test, gt)["mean_w1"]
                   for gt in gen_test_by_method[name]]
        entry = comparison["methods"][name]
        table[name] = {
            "method": name,
            "kind": "conditional",
            "nfe_per_window": int(nfe[name]),
            "params": (int(params[name]) if name in params else 0),
            "val_mean_w1": float(np.mean(val_w1)),
            "val_mean_w1_seeds": [float(v) for v in val_w1],
            "val_joint_error": float(np.mean(
                [j["joint_error"] for j in val_joint])),
            "val_joint_error_seeds": [float(j["joint_error"])
                                      for j in val_joint],
            "val_lag_acf_error": float(np.mean(
                [j["lag_acf_error"] for j in val_joint])),
            "val_corr_error": float(np.mean(
                [j["corr_error"] for j in val_joint])),
            "test_mean_w1": float(np.mean(test_w1)),
            "test_mean_w1_seeds": [float(v) for v in test_w1],
            "auc_test_mean": float(entry["auc_test_mean"]),
            "auc_test_seeds": [float(r["discriminator"]["auc_test"])
                               for r in entry["runs"]],
            "tstr_acc_synth_mean": float(entry["tstr_acc_synth_mean"]),
            "tstr_acc_synth_seeds": [float(r["tstr"]["acc_synth"])
                                     for r in entry["runs"]],
        }
    if verbose:
        print("\n  policy           val W1   val joint  test W1  "
              "AUC    TSTR", flush=True)
        for name in policies:
            t = table[name]
            print(f"  {name:<16} {t['val_mean_w1']:.4f}   "
                  f"{t['val_joint_error']:.4f}    "
                  f"{t['test_mean_w1']:.4f}  "
                  f"{t['auc_test_mean']:.3f}  "
                  f"{t['tstr_acc_synth_mean']:.3f}", flush=True)

    # ---- rankings and their agreement
    trained = [p for p in TRAINED_POLICIES if p in policies]
    rankings = {
        "w1": rank_scores({p: table[p]["val_mean_w1"]
                           for p in policies}),
        "joint": rank_scores({p: table[p]["val_joint_error"]
                              for p in policies}),
        "auc": rank_scores({p: table[p]["auc_test_mean"]
                            for p in policies}),
        "tstr": rank_scores({p: table[p]["tstr_acc_synth_mean"]
                             for p in policies},
                            lower_is_better=False),
    }
    rankings_trained = {
        key: {p: ranks[p] for p in trained}
        for key, ranks in rankings.items()
    }

    def _tau(a, b, keys):
        return _clean_float(kendall_tau_b([a[k] for k in keys],
                                          [b[k] for k in keys]))

    kendall = {
        "w1_vs_joint": _tau(rankings_trained["w1"],
                            rankings_trained["joint"], trained),
        "w1_vs_auc": _tau(rankings_trained["w1"],
                          rankings_trained["auc"], trained),
        "w1_vs_tstr": _tau(rankings_trained["w1"],
                           rankings_trained["tstr"], trained),
        "joint_vs_auc": _tau(rankings_trained["joint"],
                             rankings_trained["auc"], trained),
    }
    kendall_with_reference = {
        "w1_vs_joint": _tau(rankings["w1"], rankings["joint"],
                            policies),
        "w1_vs_auc": _tau(rankings["w1"], rankings["auc"], policies),
        "w1_vs_tstr": _tau(rankings["w1"], rankings["tstr"], policies),
        "joint_vs_auc": _tau(rankings["joint"], rankings["auc"],
                             policies),
    }
    pareto = sorted(pareto_frontier_rows([
        {"method": p, "val_mean_w1": table[p]["val_mean_w1"],
         "val_joint_error": table[p]["val_joint_error"],
         "auc_test_mean": table[p]["auc_test_mean"]}
        for p in policies]))

    # ---- paired blocked CIs for the joint-error differences
    joint_diffs = paired_joint_diff_ci(
        real_val, gen_val_by_method, block_lengths=block_lengths,
        n_boot=n_boot, seeds=(seed,))

    peak_rss_mb = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "run_id": run_id,
        "preregistration": {
            "policies": list(policies),
            "gen_seeds": [int(s) for s in gen_seeds],
            "train_seed": int(seed),
            "train_epochs": int(train_epochs),
            "vae_epochs": int(vae_epochs),
            "vae_lr": float(vae_lr),
            "batch": int(batch_size),
            "t_steps": int(t_steps),
            "ch_base": int(ch),
            "ch_small": int(small_ch),
            "ema_decay": float(ema_decay),
            "ddim_nfe": int(ddim_nfe),
            "n_eval": int(n_eval),
            "disc_epochs": int(disc_epochs),
            "tstr_epochs": int(tstr_epochs),
            "n_boot": int(n_boot),
            "block_lengths": [int(b) for b in block_lengths],
            "lags": [int(l) for l in LAGS],
            "note": ("Frozen before any run of this harness; the "
                     "test split is confirmation only and never "
                     "picks a ranking."),
        },
        "n_missing_cells": int(n_missing),
        "fit_session_ids": [int(s) for s in fit_sessions],
        "test_session_ids": [int(s) for s in test_sessions],
        "fit_test_overlap": [int(s) for s in overlap],
        "n_test_sessions": int(n_test_sessions),
        "policies": list(policies),
        "train_losses": {k: float(v)
                         for k, v in train_losses.items()},
        "table": table,
        "rankings": rankings,
        "rankings_trained_only": rankings_trained,
        "kendall_tau": kendall,
        "kendall_tau_with_reference": kendall_with_reference,
        "pareto": pareto,
        "pareto_rule": (
            "Pareto-optimal when no other policy has a lower-or-equal "
            "val mean W1, val joint error, and test AUC with at least "
            "one strict win. The Pareto set is the result; no policy "
            "is promoted to a default."),
        "paired_joint_diffs": joint_diffs,
        "comparison": comparison,
        "hardware": hardware_record(),
        "peak_rss_mb": float(peak_rss_mb),
        "limitations": [
            "single training seed (0); one trained model per cell",
            "validation/test subsamples of 1000 windows, not the "
            "full splits",
            "test metrics are confirmation only; rankings are picked "
            "on val",
            "the joint score averages the lag ACF error and the "
            "correlation error on one scale; both parts are reported "
            "separately alongside it",
            ("no size effect is claimed from the unmatched "
             "historical 221767/480775 variants: different split "
             "populations and capacities, historical weights never "
             "saved"),
        ],
    }


def _days(t_start) -> list:
    """Sorted unique UTC calendar days for an array of unix seconds."""
    days = sorted({datetime.datetime.fromtimestamp(
        float(t), datetime.timezone.utc).date().isoformat()
        for t in np.asarray(t_start).ravel()})
    return days


def main():
    parser = argparse.ArgumentParser(
        description="FF-R08 ranking-stability study")
    parser.add_argument("--data", default=os.path.abspath(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--vae-epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--t-steps", type=int, default=T_STEPS)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "ranking_stability_results.json"))
    parser.add_argument("--run-id", default="ff-r08")
    args = parser.parse_args()
    t0 = time.time()
    ds = load_dataset(os.path.abspath(args.data))
    if ds.manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "dataset manifest SHA "
            f"{ds.manifest_sha256} does not match the preregistered "
            f"{EXPECTED_MANIFEST_SHA256}: the FF-R08 design is "
            f"locked to the expanded 28-session corpus")
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
        "fit_session_ids": sorted(set(
            ds.val.session_ids.tolist())),
        "train_days": _days(ds.train.t_start),
        "val_days": _days(ds.val.t_start),
        "test_days": _days(ds.test.t_start),
    }
    record = run_ranking_stability(
        splits, train_meta=train_meta, run_id=args.run_id,
        train_epochs=args.epochs, vae_epochs=args.vae_epochs,
        t_steps=args.t_steps)
    record.update({
        "data_path": os.path.basename(args.data),
        "split_manifest_sha256": ds.manifest_sha256,
        "n_train": int(len(ds.train.x)),
        "n_val": int(len(ds.val.x)),
        "n_test": int(len(ds.test.x)),
        "train_days": train_meta["train_days"],
        "val_days": train_meta["val_days"],
        "test_days": train_meta["test_days"],
        "val_subsample": {"n": int(N_EVAL), "method": "strided",
                          "first_index": int(idx_val[0]),
                          "last_index": int(idx_val[-1])},
        "test_subsample": {"n": int(N_EVAL), "method": "strided",
                           "first_index": int(idx_test[0]),
                           "last_index": int(idx_test[-1])},
        "minutes": (time.time() - t0) / 60,
    })
    print("\n=== FF-R08 ranking stability ===")
    print("rankings (1 is best):")
    for metric, ranks in record["rankings"].items():
        order = sorted(ranks, key=lambda p: ranks[p])
        print(f"  {metric:<6} " + ", ".join(
            f"{p}={ranks[p]:.1f}" for p in order))
    print("Kendall tau-b vs the W1 ranking (trained policies only):")
    for key, tau in record["kendall_tau"].items():
        print(f"  {key:<14} {tau}")
    print(f"Pareto set (val W1, val joint, test AUC): "
          f"{record['pareto']}")
    print(f"missing cells: {record['n_missing_cells']}; "
          f"fit/test overlap: {record['fit_test_overlap']}")
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print(f"results -> {args.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
