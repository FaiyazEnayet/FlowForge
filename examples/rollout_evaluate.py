"""Preregistered rollout evaluation for the P09 multiwindow rollout.

Question: when a history-conditioned generator is free-run for several
consecutive 60-second windows, feeding only its own generated windows
back as context, how do seam consistency and error accumulate with
horizon, and how does that compare with teacher-forced one-step
generation and with the VAE and chunk-copy baselines?

Methods (one trained DDPM, one trained VAE, one chunk-copy baseline;
every method rolls out from the same anchor histories under the same
generation seed):

- ``ddpm``: the history-conditioned DDPM (``BookDDPM`` with
  ``use_history=True``), free-run with ``flowforge.rollout.
  rollout_windows``. The regime at each step is recomputed from the
  current context; no future real data is read.
- ``vae``: the history-conditioned VAE (``CondVAE`` with
  ``context_dim=32``), free-run the same way.
- ``chunk_copy``: the regime-matched chunk-copy baseline, chained one
  window at a time with the regime recomputed from the generated
  context at each step. Its seams join real windows drawn from
  different train sessions, which is a known limitation of chaining
  copies, recorded not hidden.
- ``teacher_forced``: the same history-conditioned DDPM, but at each
  step conditioned on the REAL preceding window (the true past
  context of the window being generated). This is the one-step-ahead
  reference: it shows what the model achieves without feedback, so
  the gap to the free-running arm is the compounding cost. It is a
  reference, not a rollout, and it is labelled as such everywhere.

Anchors are real validation (and test) windows with a full real
continuation of ``max(horizons)`` consecutive windows in the same
session, so every generated trajectory has a real continuation to be
scored against. Windows never cross a recording gap or a session
boundary: a continuation window must start exactly 60 seconds after
the previous one in the same session.

Preregistered before any run of this harness:

- Horizons (windows): 1, 2, 4, 8 (60/120/240/480 seconds).
- Primary metric (lower is better): ``cumret_rmse`` at horizon 8
  (root-mean-square error of the cumulative log return at the end of
  the eighth window). Secondary: ``mean_w1`` at each horizon and the
  seam jump table.
- Divergence threshold: 0.01 in cumulative log-return units. The
  random-walk scale of the cumulative 1s log return over 480 s is
  about 0.002 (1s log-return std 1e-4, sqrt(480) x 1e-4), so 0.01 is
  roughly a 4-5x scale bar, not a hair trigger.
- Intervals: divergence fractions carry session-block bootstrap
  intervals (whole independent sessions resampled, 200 replicates).
  Error-vs-horizon rows are point estimates on the anchor set; no
  method is declared better on a point estimate alone.
- Verdict rule: the verdict is always "measured". No method is
  promoted to a default and no long-path realism claim is made: the
  output reports whether free-running error grows with horizon, the
  horizon-8 gap between free-running and teacher-forced DDPM, and
  the seam tables, with the null (error flat or teacher-forced no
  better than free-running) reported as measured, not hidden.
- The test split is scored only and never used to pick anything.

Data caveat: the recorded dataset (data/book_windows.npz) is
gitignored, so these numbers describe the protocol on the dataset
whose manifest SHA is recorded in the output, not a committed
artifact. The default run uses the expanded 28-session corpus
(8 original chunks + 20 B1-QC 4 h sessions, Nov 9-12 2026) rebuilt
with past context (``with_past_context=True``); the rebuild carries
the same manifest SHA b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd
74fb861e1446ae581e as the P05/P07/P08 corpus, so the windows,
splits, and standardizers are identical. Pass --data to point at
another build. One training seed (0): a single training run is a
limitation, stated in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flowforge.baselines import (  # noqa: E402
    RegimeMatchedChunkCopyBaseline, segment_ids_from_windows)
from flowforge.data import FEATURE_NAMES, LOOKBACK, WINDOW, load_dataset  # noqa: E402
from flowforge.evaluate_rollout import rollout_report  # noqa: E402
from flowforge.models import (  # noqa: E402
    BookDDPM, CondVAE, count_params, train_ddpm, train_vae)
from flowforge.rollout import (  # noqa: E402
    advance_context, regime_from_context, rollout_windows,
    teacher_forced_window)

SEED = 0
GEN_SEED = 0
N_ANCHORS = 48
HORIZONS = (1, 2, 4, 8)
DIVERGENCE_THRESHOLD = 0.01
EPOCHS = 6
VAE_EPOCHS = 25
# The VAE comparator trains at lr 2e-4, not the shipped 1e-3: on the
# expanded corpus the shipped rate diverges to nan in 2 of 3 tried
# initializations (see examples/latency_frontier.py VAE_LR note).
VAE_LR = 2e-4
VAE_CONTEXT_DIM = 32
BATCH = 32
T_STEPS = 100
RUN_ID = "p09-rollout"

DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "..", "data",
    "book_windows_expanded_context.npz")

J_RET = FEATURE_NAMES.index("log_ret")
J_SPR = FEATURE_NAMES.index("log_spread_bps")
J_BID = FEATURE_NAMES.index("log_bid_sz")
J_ASK = FEATURE_NAMES.index("log_ask_sz")
J_IMB = FEATURE_NAMES.index("imbalance")
J_SF = FEATURE_NAMES.index("signed_flow")
J_CNT = FEATURE_NAMES.index("log_trade_count")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_trajectories(split, n_windows: int, max_anchors: int,
                      seed: int = 0):
    """Real anchor trajectories with full real continuations.

    ``split`` is a SplitWindows with past_context. An anchor is a
    window whose continuation chain (windows starting exactly 60 s
    apart, same session) runs the full ``n_windows``. Returns
    (real_paths, anchor_context, anchor_regime, anchor_session_ids):
    real_paths is (n, n_windows, WINDOW, FEATURES), the anchor window
    followed by its real continuations; anchor_context is the anchor's
    real past context. Anchors are a strided subsample of all eligible
    anchors, so overlapping trajectories are possible; the session-block
    bootstrap in the report treats sessions as the independent unit.
    """
    n = len(split.x)
    index = {(int(s), int(t)): i for i, (s, t) in
             enumerate(zip(split.session_ids, split.t_start))}
    eligible = []
    for i in range(n):
        s, t = int(split.session_ids[i]), int(split.t_start[i])
        chain = [index.get((s, t + WINDOW * k)) for k in range(n_windows)]
        if all(c is not None for c in chain):
            eligible.append((i, chain))
    if not eligible:
        raise ValueError("no anchor has a full real continuation")
    take = np.linspace(0, len(eligible) - 1,
                       min(max_anchors, len(eligible))).astype(int)
    chosen = [eligible[j] for j in take]
    real_paths = np.stack(
        [np.stack([split.x[c] for c in chain]) for _, chain in chosen])
    anchor_context = np.stack(
        [split.past_context[i] for i, _ in chosen]).astype(np.float64)
    anchor_regime = np.stack(
        [split.regime[i] for i, _ in chosen]).astype(np.float64)
    anchor_session_ids = np.array(
        [int(split.session_ids[i]) for i, _ in chosen], dtype=int)
    return {
        "real_paths": real_paths,
        "anchor_context": anchor_context,
        "anchor_regime": anchor_regime,
        "anchor_session_ids": anchor_session_ids,
        "n_eligible": int(len(eligible)),
        "n_anchors": int(len(chosen)),
    }


def make_fixture_trajectories(n, seed, n_windows):
    """Synthetic continuous trajectories in the P03 fixture family.

    One AR path per trajectory, sliced into a LOOKBACK-second anchor
    context and ``n_windows`` consecutive windows, so the fixture
    carries real state persistence across every seam. Returns
    (train_paths, eval_paths) in physical units plus the regime stats
    used to standardize them; standardizers are fit on the train
    paths only. Fixture numbers are not comparable to real-data
    numbers.
    """
    rng = np.random.default_rng(seed)

    def _paths(count):
        total = LOOKBACK + n_windows * WINDOW
        phys = np.zeros((count, total, len(FEATURE_NAMES)))
        lev = np.exp(rng.normal(0, 0.4, size=(count, 1)))
        sigma = 0.01 * lev
        e_ret = rng.normal(size=(count, total))
        e_flow = rng.normal(size=(count, total))
        ret = np.zeros((count, total))
        flow = np.zeros((count, total))
        ret[:, 0] = sigma[:, 0] * e_ret[:, 0]
        flow[:, 0] = 0.5 * e_ret[:, 0] + np.sqrt(0.75) * e_flow[:, 0]
        for t in range(1, total):
            ret[:, t] = 0.3 * ret[:, t - 1] + sigma[:, 0] * e_ret[:, t]
            flow[:, t] = 0.6 * flow[:, t - 1] \
                + 0.5 * e_ret[:, t] + np.sqrt(0.75) * e_flow[:, t]
        phys[:, :, J_RET] = ret
        phys[:, :, J_SF] = flow
        spr0 = rng.uniform(0.5, 3.0, size=(count, 1))
        t_axis = np.arange(total)[None, :]
        phys[:, :, J_SPR] = np.log1p(
            spr0 * (1.0 + 0.3 * (t_axis / total - 0.5)))
        phys[:, :, J_BID] = 4.0 + rng.normal(0, 0.05, size=(count, total))
        phys[:, :, J_ASK] = 4.0 + rng.normal(0, 0.05, size=(count, total))
        bid = np.exp(phys[:, :, J_BID])
        ask = np.exp(phys[:, :, J_ASK])
        phys[:, :, J_IMB] = (bid - ask) / (bid + ask)
        counts = rng.integers(0, 6, size=(count, total)).astype(np.float64)
        phys[:, :, J_CNT] = np.log1p(counts)
        phys[:, :, J_SF] = np.where(counts > 0, phys[:, :, J_SF], 0.0)
        return phys

    train_phys = _paths(n * 4)
    eval_phys = _paths(n)
    f_mean = train_phys.reshape(-1, len(FEATURE_NAMES)).mean(axis=0)
    f_std = train_phys.reshape(-1, len(FEATURE_NAMES)).std(axis=0)
    r_mean = np.array([train_phys[:, :LOOKBACK, J_RET].std(axis=1).mean(),
                       np.expm1(train_phys[:, :LOOKBACK, J_SPR]).mean(axis=1).mean()])
    r_std = np.array([train_phys[:, :LOOKBACK, J_RET].std(axis=1).std(),
                      np.expm1(train_phys[:, :LOOKBACK, J_SPR]).mean(axis=1).std()])

    def std(a):
        return ((a - f_mean) / f_std).astype(np.float32)

    def paths_of(phys):
        ctx = std(phys[:, :LOOKBACK])
        wins = np.stack([std(phys[:, LOOKBACK + k * WINDOW:
                                  LOOKBACK + (k + 1) * WINDOW])
                         for k in range(n_windows)], axis=1)
        regime = np.stack([
            phys[:, :LOOKBACK, J_RET].std(axis=1),
            np.expm1(phys[:, :LOOKBACK, J_SPR]).mean(axis=1)], axis=1)
        regime = ((regime - r_mean) / r_std).astype(np.float32)
        return ctx, wins, regime

    train_ctx, train_wins, train_regime = paths_of(train_phys)
    eval_ctx, eval_wins, eval_regime = paths_of(eval_phys)
    return {
        "train_ctx": train_ctx, "train_wins": train_wins,
        "train_regime": train_regime,
        "eval_ctx": eval_ctx, "eval_wins": eval_wins,
        "eval_regime": eval_regime,
        "feature_mean": f_mean.astype(np.float64),
        "feature_std": f_std.astype(np.float64),
        "regime_mean": r_mean.astype(np.float64),
        "regime_std": r_std.astype(np.float64),
    }


def _rollout_method(method, model, anchor_context, anchor_mask,
                    n_windows, stats, seed, chunk_copy=None):
    """Roll one method out from the anchor histories."""
    if method == "chunk_copy":
        ctx = anchor_context.copy()
        mask = anchor_mask.copy()
        windows = []
        for _ in range(n_windows):
            regime = regime_from_context(ctx, stats["feature_mean"],
                                         stats["feature_std"],
                                         stats["regime_mean"],
                                         stats["regime_std"])
            drawn, _ = chunk_copy.generate(
                len(ctx), regime.astype(np.float32), seed=seed)
            drawn = np.asarray(drawn, dtype=np.float64)
            windows.append(drawn)
            ctx = advance_context(ctx, drawn, lookback=ctx.shape[1])
            mask = np.concatenate(
                [mask, np.ones((len(ctx), drawn.shape[1]), dtype=bool)],
                axis=1)[:, -ctx.shape[1]:].copy()
        return np.stack(windows, axis=1)
    return rollout_windows(
        model, anchor_context, anchor_mask=anchor_mask,
        n_windows=n_windows, feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"], regime_mean=stats["regime_mean"],
        regime_std=stats["regime_std"], seed=seed,
        run_id=RUN_ID)["windows"]


def _teacher_forced(model, traj, stats, seed):
    """One-step-ahead reference: real context at every step.

    Step k is conditioned on the real window k-1 (the true past
    context of real window k, which is that window's recorded
    past_context). This uses real history at every step, so it is a
    reference, not a rollout.
    """
    real_paths = traj["real_paths"]
    n, n_windows = real_paths.shape[0], real_paths.shape[1]
    # past contexts of the real windows: window 0 uses the anchor
    # context; window k > 0 uses real window k-1 itself.
    contexts = np.concatenate(
        [traj["anchor_context"][:, None, :, :],
         real_paths[:, :-1, :, :]], axis=1)
    regimes = traj["anchor_regime"]  # regime of the anchor window
    # regimes for later windows are recomputed from their real past
    # context, exactly as the pipeline would record them.
    reg_all = [regimes]
    for k in range(1, n_windows):
        reg_all.append(regime_from_context(
            real_paths[:, k - 1], stats["feature_mean"],
            stats["feature_std"], stats["regime_mean"],
            stats["regime_std"]))
    reg_all = np.stack(reg_all, axis=1)
    windows = []
    for k in range(n_windows):
        mask = np.ones((n, LOOKBACK), dtype=bool)
        windows.append(teacher_forced_window(
            model, contexts[:, k], mask, reg_all[:, k].astype(np.float64),
            seed=seed + k, run_id=RUN_ID))
    return np.stack(windows, axis=1)


def run_rollout_evaluation(data_path=None, horizons=HORIZONS,
                           n_anchors=N_ANCHORS, epochs=EPOCHS,
                           vae_epochs=VAE_EPOCHS, vae_lr=VAE_LR,
                           t_steps=T_STEPS, ch=96, n_blocks=3,
                           gen_seed=GEN_SEED, train_seed=SEED,
                           batch=BATCH, fixture=False, verbose=True):
    """Train the history-conditioned models and roll all methods out.

    Returns the JSON-able record. In fixture mode everything runs on
    the synthetic fixture (small models, few anchors) and the record
    is labelled fixture-only.
    """
    n_windows = max(horizons)
    stats = {}
    if fixture:
        fix = make_fixture_trajectories(n_anchors, seed=101,
                                        n_windows=n_windows)
        stats = {"feature_mean": fix["feature_mean"],
                 "feature_std": fix["feature_std"],
                 "regime_mean": fix["regime_mean"],
                 "regime_std": fix["regime_std"]}
        train_x = torch.from_numpy(
            fix["train_wins"].reshape(-1, WINDOW, len(FEATURE_NAMES)))
        # past context per train window: the window's own preceding
        # 60 s inside its trajectory; the first window of each
        # trajectory uses the anchor context.
        n_train_traj = fix["train_wins"].shape[0]
        train_ctx_arr = np.concatenate(
            [fix["train_ctx"][:, None, :, :],
             fix["train_wins"][:, :-1, :, :]], axis=1)
        # each window's regime is the summary of its own preceding
        # context, as the data pipeline records it.
        train_regime = torch.from_numpy(
            regime_from_context(
                train_ctx_arr.reshape(-1, LOOKBACK, len(FEATURE_NAMES)),
                stats["feature_mean"], stats["feature_std"],
                stats["regime_mean"], stats["regime_std"]).astype(np.float32))
        train_ctx = torch.from_numpy(
            train_ctx_arr.reshape(-1, LOOKBACK, len(FEATURE_NAMES)))
        train_mask = torch.ones(train_ctx.shape[0], LOOKBACK,
                                dtype=torch.bool)
        eval_traj = {"real_paths": fix["eval_wins"].astype(np.float64),
                     "anchor_context": fix["eval_ctx"].astype(np.float64),
                     "anchor_regime": fix["eval_regime"].astype(np.float64),
                     "anchor_session_ids": np.zeros(n_anchors, dtype=int),
                     "n_eligible": int(n_anchors),
                     "n_anchors": int(n_anchors)}
        eval_trajs = {"val": eval_traj}
        chunk_train_x = fix["train_wins"].reshape(
            -1, WINDOW, len(FEATURE_NAMES)).astype(np.float64)
        chunk_train_regime = train_regime.numpy().astype(np.float64)
        chunk_session_ids = np.repeat(np.arange(n_train_traj), n_windows)
        chunk_segment_ids = chunk_session_ids.copy()
        data_info = {"fixture_only": True, "n_train": int(len(train_x))}
    else:
        ds = load_dataset(os.path.abspath(data_path))
        stats = {"feature_mean": ds.feature_mean,
                 "feature_std": ds.feature_std,
                 "regime_mean": ds.regime_mean,
                 "regime_std": ds.regime_std}
        if ds.train.past_context is None or ds.val.past_context is None \
                or ds.test.past_context is None:
            raise ValueError(
                "dataset was built without past context; rebuild with "
                "with_past_context=True")
        train_x = torch.from_numpy(ds.train.x)
        train_regime = torch.from_numpy(ds.train.regime)
        train_ctx = torch.from_numpy(ds.train.past_context)
        train_mask = torch.from_numpy(ds.train.past_mask)
        eval_trajs = {
            "val": build_trajectories(ds.val, n_windows, n_anchors),
            "test": build_trajectories(ds.test, n_windows, n_anchors),
        }
        seg_train = segment_ids_from_windows(ds.train.session_ids,
                                             ds.train.t_start,
                                             ds.train.t_end)
        chunk_train_x = ds.train.x
        chunk_train_regime = ds.train.regime
        chunk_session_ids = ds.train.session_ids
        chunk_segment_ids = seg_train
        data_info = {
            "fixture_only": False,
            "data_path": os.path.basename(data_path),
            "data_sha256": _sha256_file(os.path.abspath(data_path)),
            "split_manifest_sha256": ds.manifest_sha256,
            "n_train": int(len(ds.train.x)),
            "n_val": int(len(ds.val.x)),
            "n_test": int(len(ds.test.x)),
            "val_eligible_anchors": eval_trajs["val"]["n_eligible"],
            "test_eligible_anchors": eval_trajs["test"]["n_eligible"],
        }
        if verbose:
            print(f"  train {len(ds.train.x)} / val {len(ds.val.x)} / "
                  f"test {len(ds.test.x)} windows; manifest "
                  f"{ds.manifest_sha256[:12]}...", flush=True)

    # --- train the history-conditioned models (identical data/seed) ---
    model = BookDDPM(t_steps=t_steps, ch=ch, n_blocks=n_blocks,
                     run_id=RUN_ID, use_history=True)
    n_params_ddpm = count_params(model)
    t0 = time.time()
    hist, _ = train_ddpm(model, train_x, train_regime, epochs=epochs,
                         batch=batch, seed=train_seed, verbose=False,
                         run_id=RUN_ID, past_context=train_ctx,
                         past_mask=train_mask)
    train_seconds = time.time() - t0
    if verbose:
        print(f"  ddpm (history) trained {epochs} epochs in "
              f"{train_seconds / 60:.1f}m (final loss {hist[-1]:.4f})",
              flush=True)
    vae = CondVAE(context_dim=VAE_CONTEXT_DIM, run_id=RUN_ID)
    n_params_vae = count_params(vae)
    t0 = time.time()
    vae_hist = train_vae(vae, train_x, train_regime, epochs=vae_epochs,
                         seed=train_seed, verbose=False, run_id=RUN_ID,
                         lr=vae_lr, past_context=train_ctx,
                         past_mask=train_mask)
    vae_seconds = time.time() - t0
    if verbose:
        print(f"  vae (history) trained {vae_epochs} epochs in "
              f"{vae_seconds / 60:.1f}m (final loss {vae_hist[-1]:.4f})",
              flush=True)
    chunk_copy = RegimeMatchedChunkCopyBaseline(
        chunk_train_x, chunk_train_regime,
        block_len=1 if fixture else 4,
        session_ids=chunk_session_ids, segment_ids=chunk_segment_ids)
    model.eval()
    vae.eval()

    # --- roll out every method on each anchor set ---
    methods = {}
    reports = {}
    for split_name, traj in eval_trajs.items():
        real_paths = traj["real_paths"]
        anchor_context = traj["anchor_context"]
        anchor_mask = np.ones(anchor_context.shape[:2], dtype=bool)
        gen = {}
        t0 = time.time()
        gen["ddpm"] = _rollout_method(
            "ddpm", model, anchor_context, anchor_mask, n_windows,
            stats, gen_seed)
        gen["vae"] = _rollout_method(
            "vae", vae, anchor_context, anchor_mask, n_windows,
            stats, gen_seed)
        gen["chunk_copy"] = _rollout_method(
            "chunk_copy", None, anchor_context, anchor_mask, n_windows,
            stats, gen_seed, chunk_copy=chunk_copy)
        gen["teacher_forced"] = _teacher_forced(
            model, traj, stats, gen_seed)
        if verbose:
            print(f"  {split_name}: rolled {len(real_paths)} anchors x "
                  f"{n_windows} windows in {time.time() - t0:.1f}s",
                  flush=True)
        reports[split_name] = rollout_report(
            real_paths, gen, anchor_context,
            stats["feature_mean"], stats["feature_std"],
            horizons, DIVERGENCE_THRESHOLD,
            session_ids=traj["anchor_session_ids"])
        if verbose:
            for name, rep in reports[split_name]["methods"].items():
                h8 = rep["error_vs_horizon"][-1]
                print(f"    {split_name} {name:<15} h8 cumret_rmse "
                      f"{h8['cumret_rmse']:.4f} mean_w1 "
                      f"{h8['mean_w1']:.4f}", flush=True)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "run_id": RUN_ID,
        "fixture_only": bool(fixture),
        "comparable_to_published_real_data": False,
        "seed": int(train_seed),
        "gen_seed": int(gen_seed),
        "horizons": [int(h) for h in horizons],
        "n_anchors": int(n_anchors),
        "n_windows": int(n_windows),
        "divergence_threshold": float(DIVERGENCE_THRESHOLD),
        "epochs": int(epochs),
        "vae_epochs": int(vae_epochs),
        "vae_lr": float(vae_lr),
        "vae_context_dim": int(VAE_CONTEXT_DIM),
        "batch": int(batch),
        "t_steps": int(t_steps),
        "ddpm_params": int(n_params_ddpm),
        "vae_params": int(n_params_vae),
        "ddpm_final_train_loss": float(hist[-1]),
        "vae_final_train_loss": float(vae_hist[-1]),
        "ddpm_train_seconds": float(train_seconds),
        "vae_train_seconds": float(vae_seconds),
        "hardware": {
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "peak_rss_mb": float(peak_rss_mb),
        "data": data_info,
        "reports": reports,
        "verdict": "measured",
        "verdict_rule": (
            "The verdict is always 'measured': no method is promoted "
            "and no long-path realism claim is made. The report states "
            "whether free-running error grows with horizon, the "
            "horizon-8 gap between free-running and teacher-forced "
            "DDPM, and the seam tables."),
        "limitations": [
            "single training seed (0); one DDPM and one VAE trained",
            "anchor sets of 48 trajectories per split, not the full splits",
            "test metrics are confirmation only; nothing is selected on test",
            "chunk-copy seams join real windows from different train sessions",
            "teacher-forced reference uses real history at every step",
        ],
    }


def main():
    global RUN_ID
    parser = argparse.ArgumentParser(
        description="P09 multiwindow rollout evaluation")
    parser.add_argument("--data", default=os.path.abspath(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--vae-epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--t-steps", type=int, default=T_STEPS)
    parser.add_argument("--anchors", type=int, default=N_ANCHORS)
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)),
                        help="comma-separated horizon lengths in windows")
    parser.add_argument("--fixture", action="store_true",
                        help="run on the synthetic fixture, not real data")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "rollout_evaluation_results.json"))
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    RUN_ID = args.run_id
    t0 = time.time()
    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    record = run_rollout_evaluation(
        data_path=None if args.fixture else args.data,
        horizons=horizons, n_anchors=args.anchors,
        epochs=args.epochs, vae_epochs=args.vae_epochs,
        t_steps=args.t_steps, fixture=args.fixture)
    record["minutes"] = (time.time() - t0) / 60
    print("\n=== P09 rollout evaluation (val, horizon 8) ===")
    for name, rep in record["reports"]["val"]["methods"].items():
        h8 = rep["error_vs_horizon"][-1]
        d8 = rep["divergence"][-1]
        print(f"  {name:<15} cumret_rmse {h8['cumret_rmse']:.4f} "
              f"mean_w1 {h8['mean_w1']:.4f} "
              f"diverged {d8['diverged_fraction']:.2f} "
              f"CI {d8['ci']}")
    print("verdict: measured (no promotion, no long-path realism claim)")
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print(f"results -> {args.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    raise SystemExit(main())
