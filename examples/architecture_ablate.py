"""Architecture ablation for the BookDDPM denoiser: contract, oracle, capture.

Source contract (P04, part 1 of 2)
----------------------------------
The current Denoiser stacks same-resolution 3-wide 1-D convolutions: an
input convolution, three residual blocks of two convolutions each, and an
output convolution, all stride 1 with unit dilation. Its convolutional
receptive field is 17 positions against a 60-position window, so a signal
injected at an early window offset cannot reach late denoiser predictions
through the convolutional path. This file pins down, before any candidate
is built:

- the architectures under comparison, declared as layer specs below
  (current CNN baseline; dilated residual TCN with an explicit dilation
  cycle; compact downsample/upsample 1-D U-Net; optional attention). A
  transformer is not added merely because it sounds advanced.
- the parameter budget rule: a candidate's trainable parameter count must
  be within PARAM_BUDGET_TOLERANCE (5 percent) of the baseline count
  captured here; channel width is the knob used to meet the budget.
- the paired wall-time comparator protocol: latency is only ever compared
  inside one run of this script, same process, same input shape, warmup
  passes discarded, median of timed passes reported next to the baseline
  measured in the same run. Numbers from different runs are not compared.
- the measurement protocol: theoretical receptive field from the
  independent oracle below; empirical influence from autograd gradients
  of a late-position readout with respect to the input window and the
  regime vector (conditioning gradient sensitivity at distant offsets).

Independent oracle
------------------
The oracle computes receptive fields and influence spans from the
declarative layer specs with plain integer arithmetic. It imports nothing
from flowforge.models and shares no code with the architectures, so its
answers are known independently of the implementation under test. Known
answers for small hand-computed stacks are pinned in tests/test_models.py.

One documented subtlety: GroupNorm computes its statistics across the
whole window, so a weak global coupling exists outside the convolutional
span (measured on the baseline at a few percent of the within-span
gradient magnitude). The oracle describes the convolutional path, which
is the path an architecture change in part 2 acts on; the fixture bounds
the out-of-span gradient relative to the in-span gradient instead of
pretending it is exactly zero, and the capture records both numbers.

Part 1 captures the baseline only and promotes nothing. Part 2 implements
the candidate specs and runs the paired fidelity, separation, latency and
overfit comparison against this capture. Fixed seeds everywhere. CPU only.

Part 2 (paired comparison)
--------------------------
compare_architectures trains every architecture in DENOISER_ARCHITECTURES
on the same windows with the same seed streams, steps, and budget rule,
and re-measures the baseline in the same run: fidelity is the marginal
Wasserstein distance of samples generated for the validation regimes
against the real validation windows, separation is the repository
discriminator protocol (fit on validation, scored on test), latency is
the paired protocol from part 1, and the overfit record is the per-epoch
training loss next to a validation p-loss evaluated under one fixed
evaluation stream shared by every architecture and epoch. Empirical
influence is recorded per architecture at the last offset under the
part 1 fixture protocol. Nothing is promoted by the comparison itself;
the record reports what was measured, including null outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flowforge.data import N_FEATURES, WINDOW, load_dataset
from flowforge.evaluate import discriminator_auc, marginal_table
from flowforge.models import BookDDPM, count_params, derive_seed, scoped_rng

BASELINE_RUN_ID = "p04-part1-baseline"
PART2_RUN_ID = "p04-part2-comparison"
# BookDDPM architecture name -> spec name in ARCHITECTURE_SPECS.
ARCH_TO_SPEC = {
    "cnn": "baseline_cnn",
    "dilated_tcn": "dilated_tcn",
    "compact_unet": "compact_unet",
    "attention": "attention",
}
INFLUENCE_OFFSETS = (0, 9, 19, 29, 41, 59)
PARAM_BUDGET_TOLERANCE = 0.05
# An architecture is called local to its oracle span when the largest
# out-of-span input gradient is below this fraction of the largest
# in-span gradient. The baseline measures 0.04-0.07 (the GroupNorm
# statistics path); the bound leaves roughly 2x headroom.
INFLUENCE_RELATIVE_BOUND = 0.15

# ------------------------------------------------------------- layer specs
# A spec is a list of layers in signal order. A conv layer is
# {"kind": "conv", "kernel": k, "stride": s, "dilation": d}. An upsample
# layer is {"kind": "upsample", "factor": f} (nearest-neighbour, halves
# the jump per factor of 2). A global mixing layer is
# {"kind": "global_mix"} and makes the receptive field unbounded.

def _conv(kernel=3, stride=1, dilation=1):
    return {"kind": "conv", "kernel": kernel, "stride": stride,
            "dilation": dilation}


# The current Denoiser: in_conv, 3 ResBlocks x 2 convolutions, out conv.
BASELINE_CNN_SPEC = [_conv()] + [_conv(), _conv()] * 3 + [_conv()]

# Dilated residual TCN: in/out convolutions plus six residual blocks of
# two convolutions each, block dilations following the explicit cycle
# 1, 2, 4, 8, 16, 32. Channel width is set in part 2 to meet the budget.
_DILATION_CYCLE = (1, 2, 4, 8, 16, 32)
DILATED_TCN_SPEC = (
    [_conv()]
    + [layer for d in _DILATION_CYCLE
       for layer in (_conv(dilation=d), _conv(dilation=d))]
    + [_conv()]
)

# Compact 1-D U-Net, longest encoder-to-decoder path: two stride-2
# downsampling stages, a two-convolution bottleneck, and mirrored
# upsampling stages with a convolution after each upsample.
COMPACT_UNET_SPEC = [
    _conv(),                    # full resolution
    _conv(stride=2),            # down to L/2
    _conv(),
    _conv(stride=2),            # down to L/4
    _conv(),
    _conv(),                    # bottleneck
    {"kind": "upsample", "factor": 2},
    _conv(),
    {"kind": "upsample", "factor": 2},
    _conv(),
    _conv(),                    # output
]

# Optional attention variant: one global mixing layer over positions.
ATTENTION_SPEC = [_conv(), {"kind": "global_mix"}, _conv()]

ARCHITECTURE_SPECS = {
    "baseline_cnn": BASELINE_CNN_SPEC,
    "dilated_tcn": DILATED_TCN_SPEC,
    "compact_unet": COMPACT_UNET_SPEC,
    "attention": ATTENTION_SPEC,
}


# --------------------------------------------------------- independent oracle

def oracle_receptive_field(spec) -> int | None:
    """Receptive field size of a layer spec, in input positions.

    Standard recurrence over the spec with plain integer arithmetic:
    each convolution adds (kernel - 1) * dilation * jump positions and
    multiplies the jump by its stride; an upsample divides the jump.
    Returns None when the spec contains a global mixing layer (unbounded
    receptive field). Raises ValueError on a malformed spec.
    """
    rf = 1
    jump = 1
    for layer in spec:
        kind = layer["kind"]
        if kind == "conv":
            rf += (layer["kernel"] - 1) * layer["dilation"] * jump
            jump *= layer["stride"]
        elif kind == "upsample":
            if jump % layer["factor"] != 0:
                raise ValueError("upsample factor does not divide jump")
            jump //= layer["factor"]
        elif kind == "global_mix":
            return None
        else:
            raise ValueError(f"unknown layer kind: {kind!r}")
    return rf


def oracle_half_width(spec) -> int | None:
    """Largest input offset distance a spec can connect, or None.

    Only defined for specs whose receptive field is odd and whose total
    jump returns to 1 (true for every spec in ARCHITECTURE_SPECS), where
    a centred kernel stack connects offsets up to (rf - 1) // 2.
    """
    rf = oracle_receptive_field(spec)
    if rf is None:
        return None
    if rf % 2 == 0:
        raise ValueError("half width requires an odd receptive field")
    return (rf - 1) // 2


def oracle_influence_span(spec, source: int, length: int) -> tuple[int, int]:
    """Inclusive [lo, hi] output positions an input at `source` can reach.

    Exact for centred stride-1 stacks and for stacks whose total jump
    returns to 1: the reachable outputs are the window positions within
    the spec's half width of the source, clipped to the window.
    """
    hw = oracle_half_width(spec)
    if hw is None:
        return (0, length - 1)
    return (max(0, source - hw), min(length - 1, source + hw))


def within_param_budget(candidate_params: int, baseline_params: int) -> bool:
    """The part 2 budget rule: within PARAM_BUDGET_TOLERANCE of baseline."""
    return candidate_params <= round(baseline_params
                                     * (1.0 + PARAM_BUDGET_TOLERANCE))


# ------------------------------------------------------- empirical measurement

def gradient_influence(model, x, t, regime, target: int) -> torch.Tensor:
    """Per-offset input influence on one output position.

    Returns a length-L tensor: for each input offset, the largest
    absolute gradient of the summed output at `target` over batch and
    features with respect to the input at that offset. Computed with the
    model's own autograd; the oracle never uses this.
    """
    x = x.detach().clone().requires_grad_(True)
    out = model(x, t, regime)
    grad, = torch.autograd.grad(out[:, :, target].sum(), x)
    return grad.abs().amax(dim=(0, 1))


def conditioning_sensitivity(model, x, t, regime, target: int) -> float:
    """Largest absolute gradient of the output at `target` wrt the regime.

    The regime vector is added at every position by construction, so this
    measures how strongly conditioning reaches a distant offset.
    """
    regime = regime.detach().clone().requires_grad_(True)
    out = model(x, t, regime)
    grad, = torch.autograd.grad(out[:, :, target].sum(), regime)
    return float(grad.abs().max())


def influence_summary(profile: torch.Tensor, target: int,
                      half_width: int) -> dict:
    """Split an influence profile into in-span and out-of-span parts."""
    lo = max(0, target - half_width)
    hi = min(profile.shape[0] - 1, target + half_width)
    mask = torch.ones(profile.shape[0], dtype=torch.bool)
    mask[lo:hi + 1] = False
    within = float(profile[lo:hi + 1].max())
    beyond = float(profile[mask].max()) if bool(mask.any()) else 0.0
    total = float(profile.sum())
    beyond_mass = float(profile[mask].sum()) if bool(mask.any()) else 0.0
    return {
        "target": target,
        "span": [lo, hi],
        "within_span_max_grad": within,
        "beyond_span_max_grad": beyond,
        "beyond_to_within_ratio": beyond / within if within > 0 else 0.0,
        "beyond_span_mass_fraction": beyond_mass / total if total > 0 else 0.0,
    }


def _state_dict_sha256(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _canonical_sha256(record: dict) -> str:
    blob = json.dumps(record, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def measure_forward_latency_ms(model, batch: int = 32, warmup: int = 3,
                               reps: int = 15) -> float:
    """Median forward latency under the paired comparator protocol."""
    x = torch.zeros(batch, N_FEATURES, WINDOW)
    t = torch.zeros(batch, dtype=torch.long)
    regime = torch.zeros(batch, 2)
    times = []
    with torch.no_grad():
        for i in range(warmup + reps):
            start = time.perf_counter()
            model(x, t, regime)
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    return times[len(times) // 2]


# ------------------------------------------------------------- baseline capture

def capture_baseline(run_id: str = BASELINE_RUN_ID) -> dict:
    """Capture the current Denoiser baseline under a fixed run ID.

    Everything except the latency fields is a deterministic function of
    the run ID on CPU: seeded init, a seeded fixture input, and autograd
    gradients. The record carries a hash of the initialised weights and
    of the capture configuration so part 2 compares against an immutable
    baseline, and so a capture taken after any model change is visibly a
    different capture.
    """
    ddpm = BookDDPM(run_id=run_id)
    net = ddpm.net
    params = count_params(net)
    half_width = oracle_half_width(BASELINE_CNN_SPEC)
    with scoped_rng(derive_seed(run_id, "p04:fixture")):
        x = torch.randn(2, N_FEATURES, WINDOW)
        regime = torch.randn(2, 2)
    t = torch.full((2,), 3, dtype=torch.long)
    targets = {}
    for target in (0, WINDOW // 2, WINDOW - 1):
        profile = gradient_influence(net, x, t, regime, target)
        targets[str(target)] = influence_summary(profile, target, half_width)
    record = {
        "run_id": run_id,
        "architecture": "baseline_cnn",
        "model": {
            "params": params,
            "param_budget_max": round(params * (1.0 + PARAM_BUDGET_TOLERANCE)),
            "init_state_sha256": _state_dict_sha256(net),
        },
        "spec": BASELINE_CNN_SPEC,
        "theoretical": {
            "receptive_field": oracle_receptive_field(BASELINE_CNN_SPEC),
            "half_width": half_width,
            "window": WINDOW,
            "full_window_context": half_width >= WINDOW - 1,
        },
        "empirical": {
            "targets": targets,
            "conditioning_sensitivity_at_last_offset":
                conditioning_sensitivity(net, x, t, regime, WINDOW - 1),
            "influence_relative_bound": INFLUENCE_RELATIVE_BOUND,
        },
        "config": {
            "n_features": N_FEATURES,
            "window": WINDOW,
            "fixture_batch": 2,
            "fixture_timestep": 3,
            "torch_version": torch.__version__,
        },
    }
    record["latency"] = {
        "forward_median_ms_batch32": measure_forward_latency_ms(net),
        "protocol": "same process, batch 32, 3 warmup + 15 timed passes, "
                    "median; comparable only within one run of this script",
    }
    record["capture_sha256"] = _canonical_sha256(
        {k: v for k, v in record.items() if k != "latency"})
    return record


def spec_report() -> dict:
    """Oracle receptive fields for every declared architecture spec."""
    report = {}
    for name, spec in ARCHITECTURE_SPECS.items():
        rf = oracle_receptive_field(spec)
        hw = oracle_half_width(spec) if rf is not None else None
        report[name] = {
            "receptive_field": rf,
            "half_width": hw,
            "full_window_context": hw is None or hw >= WINDOW - 1,
        }
    return report


def _eval_p_loss(model, x, regime, run_id: str, batch: int = 64) -> float:
    """Validation p-loss under one fixed evaluation stream.

    The stream is re-seeded identically on every call, so every
    architecture and every epoch is scored on the same timesteps and
    noise draws: the overfit curves are paired, not just comparable.
    """
    xc = x.permute(0, 2, 1).contiguous()
    losses = []
    with scoped_rng(derive_seed(run_id, "p04:valeval")):
        with torch.no_grad():
            for i in range(0, xc.shape[0], batch):
                losses.append(float(model.p_loss(xc[i:i + batch],
                                                 regime[i:i + batch])))
    return sum(losses) / len(losses)


def _train_with_curves(model, x, regime, val_x, val_regime, run_id: str,
                       epochs: int, batch: int = 32, lr: float = 2e-4):
    """train_ddpm's protocol (scoped stream, Adam, same batching) with a
    per-epoch validation p-loss appended; no EMA, as in the default
    comparator configuration."""
    eff_seed = derive_seed(run_id, "train_ddpm")
    with scoped_rng(eff_seed):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n = x.shape[0]
        xc = x.permute(0, 2, 1).contiguous()
        curve = []
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot, nb = 0.0, 0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                loss = model.p_loss(xc[idx], regime[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            curve.append({
                "epoch": ep,
                "train_loss": tot / nb,
                "val_loss": _eval_p_loss(model, val_x, val_regime, run_id),
            })
    return curve


def compare_architectures(splits, run_id: str = PART2_RUN_ID,
                          epochs: int = 12, t_steps: int = 100,
                          batch: int = 32, lr: float = 2e-4,
                          disc_epochs: int = 30,
                          dataset_record: dict | None = None,
                          verbose: bool = True) -> dict:
    """Paired comparison of every denoiser architecture in one run.

    ``splits`` maps "train"/"val"/"test" to (x, regime) float tensors in
    the repository layout (n, WINDOW, N_FEATURES) / (n, 2). Every
    architecture is initialised and trained from the same run ID
    streams, for the same number of steps, and measured in this process;
    the baseline is re-measured here rather than copied from the part 1
    capture, so every number in the record is paired within the run.
    """
    train_x, train_regime = splits["train"]
    val_x, val_regime = splits["val"]
    test_x, test_regime = splits["test"]
    baseline_params = count_params(BookDDPM(run_id=run_id).net)
    with scoped_rng(derive_seed(run_id, "p04:fixture")):
        fix_x = torch.randn(2, N_FEATURES, WINDOW)
        fix_regime = torch.randn(2, 2)
    fix_t = torch.full((2,), 3, dtype=torch.long)
    record = {
        "run_id": run_id,
        "protocol": {
            "epochs": epochs,
            "batch": batch,
            "lr": lr,
            "t_steps": t_steps,
            "ema": None,
            "training": "train_ddpm protocol (scoped stream derived "
                        "from the run ID, Adam, batch order and noise "
                        "draws identical across architectures)",
            "fidelity": "marginal_table mean W1, generated for the "
                        "validation regimes vs real validation windows",
            "separation": "discriminator_auc: fit on validation "
                          "(real vs generated), scored on test",
            "latency": "paired protocol: same process, batch 32, "
                       "3 warmup + 15 timed passes, median; comparable "
                       "only within this run",
            "overfit": "per-epoch train p-loss and validation p-loss "
                       "under one fixed evaluation stream",
            "influence": "gradient influence at the last offset under "
                         "the part 1 fixture streams, measured on the "
                         "trained model at the end of the run",
            "comparability": "all architectures measured in this single "
                             "run on identical data and seed streams",
        },
        "dataset": dataset_record,
        "baseline_params": baseline_params,
        "param_budget_max": round(baseline_params
                                  * (1.0 + PARAM_BUDGET_TOLERANCE)),
        "architectures": {},
    }
    for arch in ARCH_TO_SPEC:
        if verbose:
            print(f"  comparing {arch} ...", flush=True)
        model = BookDDPM(arch=arch, run_id=run_id, t_steps=t_steps)
        net = model.net
        params = count_params(net)
        curve = _train_with_curves(model, train_x, train_regime,
                                   val_x, val_regime, run_id, epochs,
                                   batch=batch, lr=lr)
        gen_val = model.sample(
            val_regime, seed=derive_seed(run_id, "p04:gen:val")).numpy()
        gen_test = model.sample(
            test_regime, seed=derive_seed(run_id, "p04:gen:test")).numpy()
        fidelity = marginal_table(val_x.numpy(), gen_val)
        separation = discriminator_auc(
            val_x.numpy(), gen_val, test_x.numpy(), gen_test,
            seed=derive_seed(run_id, "p04:disc"), epochs=disc_epochs)
        profile = gradient_influence(net, fix_x, fix_t, fix_regime,
                                     WINDOW - 1)
        spec = ARCHITECTURE_SPECS[ARCH_TO_SPEC[arch]]
        record["architectures"][arch] = {
            "spec": ARCH_TO_SPEC[arch],
            "params": params,
            "within_param_budget": within_param_budget(params,
                                                       baseline_params),
            "init_state_sha256": _state_dict_sha256(net),
            "theoretical": {
                "receptive_field": oracle_receptive_field(spec),
                "half_width": (oracle_half_width(spec)
                               if oracle_receptive_field(spec) is not None
                               else None),
            },
            "overfit_curve": curve,
            "final_train_loss": curve[-1]["train_loss"],
            "final_val_loss": curve[-1]["val_loss"],
            "fidelity_mean_w1": fidelity["mean_w1"],
            "separation_auc_test": separation["auc_test"],
            "separation_auc_val": separation["auc_val"],
            "latency_forward_median_ms_batch32":
                measure_forward_latency_ms(net),
            "empirical_influence_at_last_offset": {
                "offsets": list(INFLUENCE_OFFSETS),
                "gradient": [float(profile[o]) for o in INFLUENCE_OFFSETS],
                "max_gradient": float(profile.max()),
            },
        }
    return record


def load_comparison_splits(data_path: str):
    """Load the historical dataset for a comparison run.

    The canonical archive predates the typed NPZ format, so it is read
    through the explicit legacy path; the record carries the file hash
    and split counts so the population is identified exactly.
    """
    ds = load_dataset(data_path, legacy=True)
    with open(data_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    splits = {}
    for name in ("train", "val", "test"):
        part = getattr(ds, name)
        splits[name] = (torch.from_numpy(np.asarray(part.x,
                                                    dtype=np.float32)),
                        torch.from_numpy(np.asarray(part.regime,
                                                    dtype=np.float32)))
    dataset_record = {
        "path_basename": os.path.basename(data_path),
        "sha256": digest,
        "legacy_format": True,
        "splits": {name: int(splits[name][0].shape[0])
                   for name in splits},
    }
    return splits, dataset_record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=BASELINE_RUN_ID)
    ap.add_argument("--out", default=None,
                    help="also write the capture JSON to this path")
    ap.add_argument("--compare", action="store_true",
                    help="run the part 2 paired architecture comparison")
    ap.add_argument("--data", default=None,
                    help="dataset NPZ for --compare (loaded legacy=True)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--t-steps", type=int, default=100)
    args = ap.parse_args()
    if args.compare:
        if not args.data:
            ap.error("--compare requires --data")
        run_id = (args.run_id if args.run_id != BASELINE_RUN_ID
                  else PART2_RUN_ID)
        splits, dataset_record = load_comparison_splits(args.data)
        record = compare_architectures(
            splits, run_id=run_id, epochs=args.epochs,
            t_steps=args.t_steps, dataset_record=dataset_record)
    else:
        record = capture_baseline(args.run_id)
        record["spec_report"] = spec_report()
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
