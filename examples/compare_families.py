"""P10 part 2: run the model-family comparison on the expanded corpus.

Trains the DDPM baseline and the three challengers (compact causal
autoregressor first, history-conditioned diffusion variant next,
rectified flow matching last) plus the VAE and chunk-copy references,
generates under the paired generation-seed set (0, 100, 200), and
scores every family with the identical final discriminators and
TSTR protocol. Writes examples/compare_families_results.json with
the one Pareto table.

The run is resumable: trained weights and generated windows are
checkpointed under runs/p10-families, so an interrupted run picks up
where it stopped instead of retraining. The expanded-corpus NPZ is
gitignored; the run refuses any other manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from flowforge.compare_families import (  # noqa: E402
    DEFAULT_CONFIG, run_family_comparison)
from flowforge.data import load_dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXPECTED_MANIFEST_SHA256 = (
    "b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd74fb861e1446ae581e")
N_EVAL = 1000


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _subsample(split, n: int):
    idx = np.linspace(0, len(split.x) - 1, n).astype(int)
    return dataclasses.replace(
        split,
        x=split.x[idx], regime=split.regime[idx],
        t_start=split.t_start[idx], t_end=split.t_end[idx],
        session_ids=split.session_ids[idx],
        past_context=split.past_context[idx]
        if split.past_context is not None else None,
        past_mask=split.past_mask[idx]
        if split.past_mask is not None else None,
        past_activity=split.past_activity[idx]
        if split.past_activity is not None else None,
        past_depth=split.past_depth[idx]
        if split.past_depth is not None else None)


def _recorded_ddpm():
    path = os.path.join(HERE, "family_baseline_results.json")
    with open(path) as f:
        baseline = json.load(f)
    row = next(r for r in baseline["pareto_rows"]
               if r["method"] == "ddpm")
    return {
        "source": "FF-R08 recorded row, via "
                  "examples/family_baseline_results.json",
        "row": row,
        "expanded_corpus": baseline["expanded_corpus"]["ddpm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P10 model-family comparison")
    parser.add_argument("--data", default=os.path.abspath(
        os.path.join(REPO, "data", "book_windows.npz")))
    parser.add_argument("--out", default=os.path.join(
        HERE, "compare_families_results.json"))
    parser.add_argument("--run-id", default="p10-families")
    parser.add_argument("--checkpoint-dir", default=os.path.join(
        REPO, "runs", "p10-families"))
    parser.add_argument("--epochs", type=int,
                        default=DEFAULT_CONFIG["train_epochs"])
    args = parser.parse_args()
    t0 = time.time()
    torch.set_num_threads(2)
    ds = load_dataset(os.path.abspath(args.data))
    if ds.manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "dataset manifest SHA "
            f"{ds.manifest_sha256} does not match the preregistered "
            f"{EXPECTED_MANIFEST_SHA256}: the P10 design is locked "
            "to the expanded 28-session corpus")
    print(f"split-manifest SHA-256: {ds.manifest_sha256}")
    print(f"train {len(ds.train.x)} / val {len(ds.val.x)} "
          f"/ test {len(ds.test.x)} windows")
    scored = dataclasses.replace(
        ds, val=_subsample(ds.val, N_EVAL),
        test=_subsample(ds.test, N_EVAL))
    config = {**DEFAULT_CONFIG, "run_id": args.run_id,
              "train_epochs": args.epochs}
    record = run_family_comparison(
        scored, config, checkpoint_dir=args.checkpoint_dir,
        recorded_ddpm=_recorded_ddpm())
    record.update({
        "data_path": os.path.basename(args.data),
        "data_file_sha256": _sha256(os.path.abspath(args.data)),
        "split_manifest_sha256": ds.manifest_sha256,
        "val_subsample": {"n": N_EVAL, "method": "strided"},
        "test_subsample": {"n": N_EVAL, "method": "strided"},
        "hardware": {
            "device": "cpu", "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "total_minutes": (time.time() - t0) / 60,
    })
    print("\n=== P10 model-family comparison: Pareto table ===")
    cols = record["pareto_table"]["columns"]
    print(" | ".join(cols))
    for row in record["pareto_table"]["rows"]:
        print(" | ".join(str(row[c]) for c in cols))
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"results -> {args.out} "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
