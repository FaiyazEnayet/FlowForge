"""Preregistered ablation for the P05 diffusion numerical/training
optimization.

Question: does tuning the noise schedule, prediction objective, EMA,
or training regularity improve diffusion fidelity or compute
efficiency at the same budget, without a larger network?

Arms (identical data, architecture, epochs, batch size, seeds, and
optimizer step counts, so every arm costs exactly the same compute):

- ``baseline``: the shipped model, linear schedule ending at 0.08,
  epsilon objective, no EMA, constant learning rate, no clipping.
  The comparator.
- ``sched_sqrt``: square-root schedule, same 0.08 endpoint, noise
  front-loaded (large early betas, terminal cumulative noise 0.004).
- ``sched_quadratic``: beta grows quadratically, endpoint 0.10
  (the shipped 0.08 endpoint leaves terminal cumulative noise at
  0.064, above the gate; 0.10 reaches 0.031).
- ``sched_sigmoid``: log-SNR schedule moving from +6 to -6 along a
  smoothstep. This is a validated log-SNR family, not a transplanted
  cosine schedule: the cosine schedule is refused by the model
  because one was tried before and diverged.
- ``obj_v``: velocity prediction objective.
- ``obj_x0``: clean-window prediction objective.
- ``ema_0999``: EMA decay 0.999, sampling from the EMA weights.
- ``ema_0999_warmup100``: EMA decay 0.999 with a 100-step warmup.
- ``lr_cosine``: cosine learning-rate annealing to zero.
- ``clip_1``: gradient clipping at global norm 1.0.

Every arm starts from identical initialized weights (same run ID
init stream) and the same training seed. Sampling uses the EMA
weights when an arm trains one, else the raw weights, under a fixed
seed. The validation split is scored only; the test split is never
touched here.

Preregistered before any run of this harness:

- Primary metric (lower is better): ``val_mean_w1`` (marginal
  1-Wasserstein distance of samples generated for a fixed
  1,000-window strided subsample of the validation regimes).
- Secondary metric: ``val_p_loss`` (mean denoising loss on the same
  validation subsample under one fixed evaluation stream).
- Throughput: ``sample_seconds_per_1k`` and ``nfe_per_window``
  (reverse steps per window, equal to t_steps for every arm).
- Verdict rule: an arm "improves" only if its ``val_mean_w1`` is
  at least 0.02 below the baseline's AND its ``val_p_loss`` is no
  more than 0.01 above the baseline's. Otherwise the verdict is
  "null" and that is the reported result. A null is a result, not a
  failure: the options ship as tested opt-in settings either way.
- Factorial phase: options that improve in the screens are combined
  (``+``-joined arm names, e.g. ``sched_sigmoid+obj_v``) and rerun
  at the same budget; a combination improves only under the same
  rule against the same-budget baseline.

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

from flowforge.data import load_dataset  # noqa: E402
from flowforge.evaluate import marginal_table  # noqa: E402
from flowforge.models import (  # noqa: E402
    BookDDPM, count_params, derive_seed, sample_with_ema, scoped_rng,
    train_ddpm)

SEED = 0
EVAL_SEED = 7
N_VAL_EVAL = 1000
EPOCHS = 6
BATCH = 32
T_STEPS = 100

DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "..", "data", "book_windows.npz")

# option name -> (model kwargs, train kwargs)
ARM_SPECS = {
    "baseline": ({}, {}),
    "sched_sqrt": ({"schedule": "sqrt"}, {}),
    "sched_quadratic": ({"schedule": "quadratic", "beta_end": 0.10}, {}),
    "sched_sigmoid": ({"schedule": "sigmoid"}, {}),
    "obj_v": ({"objective": "v"}, {}),
    "obj_x0": ({"objective": "x0"}, {}),
    "ema_0999": ({}, {"ema_decay": 0.999}),
    "ema_0999_warmup100": ({}, {"ema_decay": 0.999,
                                "ema_warmup_steps": 100}),
    "lr_cosine": ({}, {"lr_schedule": "cosine"}),
    "clip_1": ({}, {"clip_grad": 1.0}),
}

ARM_NAMES = list(ARM_SPECS)

W1_IMPROVE = 0.02
PLOSS_TOLERANCE = 0.01


def arm_spec(name: str):
    """Resolve an arm name (or +-joined combination) to kwargs.

    A combination merges the model and train kwargs of its parts;
    a repeated key with different values raises instead of silently
    picking one.
    """
    model_kw: dict = {}
    train_kw: dict = {}
    for part in name.split("+"):
        if part not in ARM_SPECS:
            raise ValueError(f"unknown ablation arm: {part!r}")
        m_kw, t_kw = ARM_SPECS[part]
        for target, kw in ((model_kw, m_kw), (train_kw, t_kw)):
            for k, v in kw.items():
                if k in target and target[k] != v:
                    raise ValueError(
                        f"conflicting {k} in combination {name!r}")
                target[k] = v
    return model_kw, train_kw


def _val_p_loss(model, x_val, r_val, run_id, batch=64):
    """Mean denoising loss on the validation subsample under one
    fixed evaluation stream, so every arm sees identical t draws
    and noise draws."""
    model.eval()
    losses = []
    with torch.no_grad(), scoped_rng(derive_seed(run_id, "p05:eval")):
        xc = x_val.permute(0, 2, 1).contiguous()
        for start in range(0, len(x_val), batch):
            losses.append(float(model.p_loss(
                xc[start:start + batch], r_val[start:start + batch])))
    return float(np.mean(losses))


def _run_arm(name, x_train, r_train, x_val, r_val, run_id, epochs,
             t_steps, verbose=True):
    model_kw, train_kw = arm_spec(name)
    model = BookDDPM(t_steps=t_steps, run_id=run_id, **model_kw)
    n_params = count_params(model)
    t0 = time.time()
    hist, ema = train_ddpm(model, x_train, r_train, epochs=epochs,
                           batch=BATCH, seed=SEED, verbose=False,
                           run_id=run_id, **train_kw)
    train_seconds = time.time() - t0
    steps = epochs * ((len(x_train) + BATCH - 1) // BATCH)
    val_p_loss = _val_p_loss(model, x_val, r_val, run_id)
    t_s0 = time.perf_counter()
    with torch.no_grad():
        if ema is not None:
            gen = sample_with_ema(model, ema, r_val,
                                  seed=EVAL_SEED).numpy()
        else:
            gen = model.sample(r_val, seed=EVAL_SEED).numpy()
    sample_seconds = time.perf_counter() - t_s0
    mean_w1 = marginal_table(x_val.numpy(), gen)["mean_w1"]
    entry = {
        "arm": name,
        "model_kwargs": model_kw,
        "train_kwargs": train_kw,
        "params": int(n_params),
        "optimizer_steps": int(steps),
        "final_train_loss": float(hist[-1]),
        "val_p_loss": float(val_p_loss),
        "val_mean_w1": float(mean_w1),
        "sample_seconds_per_1k": float(sample_seconds / max(len(gen), 1)
                                      * 1000),
        "nfe_per_window": int(t_steps),
        "train_seconds": float(train_seconds),
    }
    if verbose:
        print(f"  {name:<22} val W1 {mean_w1:.4f} "
              f"val p-loss {val_p_loss:.4f} "
              f"train {train_seconds / 60:.1f}m", flush=True)
    return entry


def compare_options(splits, run_id="p05-ablation", epochs=EPOCHS,
                    t_steps=T_STEPS, arm_names=None, verbose=True):
    """Run the paired ablation on in-memory splits.

    ``splits`` maps "train"/"val"/"test" to (x, regime) torch
    tensors in (windows, window, features) / (windows, 2) layout.
    The test split is accepted for shape parity and never used.
    Returns the record dict.
    """
    x_train, r_train = splits["train"]
    x_val, r_val = splits["val"]
    if arm_names is None:
        arm_names = list(ARM_NAMES)
    arms = {}
    for name in arm_names:
        arms[name] = _run_arm(name, x_train, r_train, x_val, r_val,
                              run_id, epochs, t_steps, verbose=verbose)
    base = arms["baseline"] if "baseline" in arms else None
    for name, entry in arms.items():
        if base is None or name == "baseline":
            entry["verdict"] = "null" if name != "baseline" else "baseline"
            continue
        improves = (
            entry["val_mean_w1"] < base["val_mean_w1"] - W1_IMPROVE
            and entry["val_p_loss"] <= base["val_p_loss"]
            + PLOSS_TOLERANCE
            and np.isfinite(entry["val_mean_w1"])
            and np.isfinite(entry["val_p_loss"]))
        entry["verdict"] = "improve" if improves else "null"
    return {
        "run_id": run_id,
        "baseline": "baseline",
        "epochs": int(epochs),
        "batch": int(BATCH),
        "t_steps": int(t_steps),
        "seed": int(SEED),
        "eval_seed": int(EVAL_SEED),
        "n_train": int(len(x_train)),
        "n_val_eval": int(len(x_val)),
        "verdict_rule": ("improve only if val_mean_w1 is at least "
                         f"{W1_IMPROVE} below baseline and val_p_loss "
                         "is no more than "
                         f"{PLOSS_TOLERANCE} above baseline"),
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="P05 diffusion optimization ablation")
    parser.add_argument("--data", default=os.path.abspath(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--t-steps", type=int, default=T_STEPS)
    parser.add_argument("--arms", default=None,
                        help="comma-separated arm names (default: all)")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "diffusion_ablation_results.json"))
    parser.add_argument("--run-id", default="p05-ablation")
    args = parser.parse_args()
    t0 = time.time()
    ds = load_dataset(os.path.abspath(args.data))
    print(f"split-manifest SHA-256: {ds.manifest_sha256}")
    print(f"train {len(ds.train.x)} / val {len(ds.val.x)} "
          f"/ test {len(ds.test.x)} windows (test untouched)")
    x_train = torch.from_numpy(ds.train.x)
    r_train = torch.from_numpy(ds.train.regime)
    # fixed strided validation subsample: covers all val sessions
    idx = np.linspace(0, len(ds.val.x) - 1, N_VAL_EVAL).astype(int)
    x_val = torch.from_numpy(ds.val.x[idx])
    r_val = torch.from_numpy(ds.val.regime[idx])
    splits = {
        "train": (x_train, r_train),
        "val": (x_val, r_val),
        "test": (torch.from_numpy(ds.test.x[:1]),
                 torch.from_numpy(ds.test.regime[:1])),
    }
    arm_names = None
    if args.arms:
        arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
        if "baseline" not in arm_names:
            arm_names = ["baseline"] + arm_names
    record = compare_options(splits, run_id=args.run_id,
                             epochs=args.epochs, t_steps=args.t_steps,
                             arm_names=arm_names)
    record.update({
        "data_path": os.path.basename(args.data),
        "split_manifest_sha256": ds.manifest_sha256,
        "val_subsample": {"n": int(N_VAL_EVAL), "method": "strided",
                          "first_index": int(idx[0]),
                          "last_index": int(idx[-1])},
        "minutes": (time.time() - t0) / 60,
        "limitations": [
            "single seed (0); one run only",
            "validation subsample of 1000 windows, not the full split",
            "test split scored by no arm and never used for tuning",
        ],
    })
    print("\n=== P05 ablation verdicts ===")
    for name, entry in record["arms"].items():
        print(f"  {name:<22} val W1 {entry['val_mean_w1']:.4f} "
              f"val p-loss {entry['val_p_loss']:.4f} "
              f"-> {entry['verdict']}")
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print(f"results -> {args.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
