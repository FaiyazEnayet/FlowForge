"""Model sweep: train DDPM variants and baselines, pick the best by VALIDATION metrics.

The test split is never touched here. For each config we train on TRAIN,
generate conditioned on VAL regimes (or resample/simulate for baselines),
and score:
  - mean 1-Wasserstein distance across features (val score-block real vs
    generated)
  - discriminator AUC: classifier trained on the val fit block (real fit
    vs generated fit, matched conditioning), evaluated on the val score
    block (real score vs generated score)

Fit and score blocks are temporally disjoint validation segments separated
by at least window-length + lookback seconds, so no raw second (window or
regime lookback) appears on both sides. Generations for the two blocks are
separate draws conditioned on each block's regimes. If either block is too
small, the discriminator score is admitted as insufficient instead of
falling back to overlapping windows.

Baselines (FL4) get the same evaluation protocol and report sampling
latency/memory alongside quality, so architecture advantages are weighed
against compute cost. The winner is recorded with its config in
runs/sweep.json; examples/demo.py reads that frozen config.
Fixed seeds everywhere. CPU only.
"""

import hashlib
import json
import os
import time

import numpy as np
import torch

from flowforge.baselines import MovingBlockResampler, VARBaseline
from flowforge.data import WINDOW, LOOKBACK, load_dataset
from flowforge.evaluate import (discriminator_auc, marginal_table,
                                wasserstein_1d)
from flowforge.models import (BookDDPM, count_params, sample_with_ema,
                              set_seed, train_ddpm)

SEED = 7
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "book_windows.npz")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")

BOUNDARY_GAP = WINDOW + LOOKBACK  # seconds between fit and score blocks
MIN_BLOCK = 30  # windows; below this the discriminator score is insufficient
DISC_FIT_CAP = 400

CONFIGS = [
    # name, ch, n_blocks, epochs, ema_decay
    ("A_baseline_80ep", 96, 3, 80, None),
    ("B_long_300ep", 96, 3, 300, None),
    ("C_long_ema", 96, 3, 300, 0.999),
    ("D_big_long_ema", 128, 4, 300, 0.999),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_val_blocks(t_start, t_end, session_ids, gap=BOUNDARY_GAP):
    """Split validation windows into temporally disjoint fit/score blocks.

    Windows are ordered by (session, start time). The fit block is the
    first half; the score block is the second half with every window
    dropped whose regime lookback reaches back into the fit block: a
    score window is kept only when its start is at least `gap` seconds
    after the last fit window end in the same session. Windows from
    sessions with no fit windows are kept whole, since windows never
    cross chunk boundaries. Returns (fit_idx, score_idx, gap).
    """
    t_start = np.asarray(t_start, dtype=np.float64)
    t_end = np.asarray(t_end, dtype=np.float64)
    session_ids = np.asarray(session_ids)
    order = np.lexsort((t_start, session_ids))
    n = len(order)
    fit = list(order[: n // 2])
    candidates = list(order[n // 2:])
    fit_end_by_session = {}
    for i in fit:
        s = session_ids[i]
        fit_end_by_session[s] = max(
            fit_end_by_session.get(s, -np.inf), t_end[i])
    score = [j for j in candidates
             if session_ids[j] not in fit_end_by_session
             or t_start[j] >= fit_end_by_session[session_ids[j]] + gap]
    return fit, score, float(gap)


def _score_generated(name, gen_fit, gen_score, real_fit, real_score, seed,
                     kind=None, train_seconds=None, params=None,
                     sample_stats=None, disc_epochs=20,
                     min_block=MIN_BLOCK):
    """Score generated windows on the disjoint score block.

    The discriminator fits on (real_fit, gen_fit) and is scored on
    (real_score, gen_score). Mean W1 is computed on the score block only.
    When either block is smaller than min_block, the discriminator score
    is reported as insufficient (None) instead of weakening the split.
    """
    mt = marginal_table(real_score, gen_score)
    mean_w1 = mt["mean_w1"]
    fit_n = min(DISC_FIT_CAP, len(real_fit), len(gen_fit))
    score_n = len(real_score)
    insufficient = min(fit_n, score_n) < min_block
    if insufficient:
        disc_auc = None
        disc_auc_fit = None
        print(f"  val mean W1: {mean_w1:.3f} | sweep-disc: insufficient "
              f"samples (fit {fit_n}, score {score_n})")
    else:
        d = discriminator_auc(real_fit[:fit_n], gen_fit[:fit_n],
                              real_score, gen_score, seed=seed,
                              epochs=disc_epochs)
        disc_auc = d["auc_test"]
        disc_auc_fit = d["auc_val"]
        print(f"  val mean W1: {mean_w1:.3f} | "
              f"sweep-disc AUC on score block: {disc_auc:.3f}")
    for row in mt["rows"]:
        print(f"    {row['feature']:<16} w1={row['w1']:.3f} "
              f"real_mean={row['real_mean']:+.3f} gen_mean={row['gen_mean']:+.3f}")
    result = {"config": name, "val_mean_w1": mean_w1,
              "val_disc_auc": disc_auc, "disc_auc_fit": disc_auc_fit,
              "fit_n": fit_n, "score_n": score_n,
              "insufficient_samples": insufficient}
    if kind is not None:
        result["kind"] = kind
    if train_seconds is not None:
        result["train_seconds"] = train_seconds
    if params is not None:
        result["params"] = params
    if sample_stats is not None:
        result.update({f"sample_{k}": v for k, v in sample_stats.items()
                       if k in ("sample_seconds_per_1k", "peak_memory_mb")})
    return result


def main():
    t0 = time.time()
    set_seed(SEED)
    ds = load_dataset(os.path.abspath(DATA))
    x_train = torch.from_numpy(ds.train.x)
    r_train = torch.from_numpy(ds.train.regime)
    val = ds.val
    fit_idx, score_idx, gap = split_val_blocks(
        val.t_start, val.t_end, val.session_ids)
    r_fit = torch.from_numpy(val.regime[fit_idx])
    r_score = torch.from_numpy(val.regime[score_idx])
    real_fit = val.x[fit_idx]
    real_score = val.x[score_idx]
    print(f"train {len(x_train)} / val {len(val.x)} "
          f"(fit {len(fit_idx)}, score {len(score_idx)}, "
          f"gap {gap:.0f}s) / test {len(ds.test.x)} windows "
          f"(test untouched)")

    results = []
    configs_by_name = {}

    # ---- baselines first (FL4): cheap, same protocol, timed ----
    print("\n=== baseline: moving-block resampler ===")
    t_b0 = time.time()
    mbr = MovingBlockResampler(ds.train.x, block_len=4)
    gen_fit_mbr, _ = mbr.generate(len(fit_idx), seed=SEED)
    gen_score_mbr, stats_mbr = mbr.generate(len(score_idx), seed=SEED + 1)
    configs_by_name["baseline_moving_block"] = {"params": 0}
    results.append(_score_generated(
        "baseline_moving_block", gen_fit_mbr, gen_score_mbr,
        real_fit, real_score, SEED, kind="baseline",
        train_seconds=time.time() - t_b0, params=0,
        sample_stats=stats_mbr))
    print(f"  sampling: {stats_mbr['sample_seconds_per_1k']:.2f}s per 1k windows, "
          f"{stats_mbr['peak_memory_mb']:.1f} MB peak")

    print("\n=== baseline: VAR(2) ridge ===")
    t_b1 = time.time()
    var = VARBaseline(order=2, ridge=1.0).fit(ds.train.x)
    gen_fit_var, _ = var.generate(len(fit_idx), ds.train.x, seed=SEED)
    gen_score_var, stats_var = var.generate(
        len(score_idx), ds.train.x, seed=SEED + 1)
    configs_by_name["baseline_var2"] = {"params": int(var.coef_.size)}
    results.append(_score_generated(
        "baseline_var2", gen_fit_var, gen_score_var,
        real_fit, real_score, SEED, kind="baseline",
        train_seconds=time.time() - t_b1, params=int(var.coef_.size),
        sample_stats=stats_var))
    print(f"  sampling: {stats_var['sample_seconds_per_1k']:.2f}s per 1k windows, "
          f"{stats_var['peak_memory_mb']:.1f} MB peak")

    for name, ch, n_blocks, epochs, ema_decay in CONFIGS:
        print(f"\n=== {name}: ch={ch} blocks={n_blocks} "
              f"epochs={epochs} ema={ema_decay} ===")
        model = BookDDPM(ch=ch, n_blocks=n_blocks)
        n_params = count_params(model)
        print(f"  params: {n_params}")
        t_train0 = time.time()
        hist, ema = train_ddpm(model, x_train, r_train, epochs=epochs,
                               seed=SEED, verbose=False,
                               ema_decay=ema_decay)
        train_secs = time.time() - t_train0
        print(f"  final train loss: {hist[-1]:.4f} "
              f"({train_secs / 60:.1f} min train)")

        # generate for fit and score regimes separately (disjoint draws)
        t_s0 = time.perf_counter()
        if ema is not None:
            gen_fit = sample_with_ema(model, ema, r_fit,
                                      seed=SEED).numpy()
            gen_score = sample_with_ema(model, ema, r_score,
                                        seed=SEED + 1).numpy()
        else:
            gen_fit = model.sample(r_fit, seed=SEED).numpy()
            gen_score = model.sample(r_score, seed=SEED + 1).numpy()
        sample_secs = time.perf_counter() - t_s0

        configs_by_name[name] = {"ch": ch, "n_blocks": n_blocks,
                                 "epochs": epochs, "ema_decay": ema_decay}
        r = _score_generated(name, gen_fit, gen_score,
                             real_fit, real_score, SEED, kind="neural",
                             train_seconds=train_secs, params=n_params,
                             sample_stats={
                                 "sample_seconds_per_1k":
                                     sample_secs / max(
                                         len(gen_fit) + len(gen_score), 1)
                                     * 1000,
                                 "peak_memory_mb": float("nan")})
        r.update({"ch": ch, "n_blocks": n_blocks, "epochs": epochs,
                  "ema_decay": ema_decay, "final_loss": hist[-1]})
        print(f"  sampling: {r['sample_sample_seconds_per_1k']:.2f}s per 1k windows")
        results.append(r)

    # pick winner: lowest val mean W1, tie-break by lower disc AUC.
    # A skipped disc score (insufficient samples) sorts last on the tie-break.
    # Baselines compete on equal footing: if a baseline wins, the neural
    # models have not earned their compute budget.
    def _sort_key(r):
        auc = r["val_disc_auc"]
        return (r["val_mean_w1"],
                auc if auc is not None else float("inf"))

    results.sort(key=_sort_key)
    winner = results[0]
    print("\n=== sweep results (sorted) ===")
    for r in results:
        t_str = (f"train {r['train_seconds'] / 60:.1f}m "
                 if "train_seconds" in r else "")
        s_str = (f"sample {r['sample_sample_seconds_per_1k']:.1f}s/1k "
                 if "sample_sample_seconds_per_1k" in r else "")
        print(f"  {r['config']:<22} val W1 {r['val_mean_w1']:.3f} "
              f"disc AUC {r['val_disc_auc']} {t_str}{s_str}")
    print(f"\nWINNER: {winner['config']} ({winner.get('kind')})")

    os.makedirs(RUNS, exist_ok=True)
    artifact = {"protocol": "disjoint-discriminator", "protocol_version": 1,
                "seed": SEED, "data_sha256": sha256_file(os.path.abspath(DATA)),
                "selection_rule": "lowest val mean W1, tie-break lower "
                                  "score-block disc AUC (disc skipped when "
                                  "a block is below MIN_BLOCK windows)",
                "val_blocks": {"fit_n": len(fit_idx),
                               "score_n": len(score_idx),
                               "gap_seconds": gap},
                "results": results,
                "winner": {"name": winner["config"],
                           "kind": winner.get("kind"),
                           "config": configs_by_name[winner["config"]]},
                "minutes": (time.time() - t0) / 60}
    with open(os.path.join(RUNS, "sweep.json"), "w") as f:
        json.dump(artifact, f, indent=1)
    print(f"sweep -> runs/sweep.json ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
