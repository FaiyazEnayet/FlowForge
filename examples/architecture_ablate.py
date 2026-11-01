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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flowforge.data import N_FEATURES, WINDOW
from flowforge.models import BookDDPM, count_params, derive_seed, scoped_rng

BASELINE_RUN_ID = "p04-part1-baseline"
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=BASELINE_RUN_ID)
    ap.add_argument("--out", default=None,
                    help="also write the capture JSON to this path")
    args = ap.parse_args()
    record = capture_baseline(args.run_id)
    record["spec_report"] = spec_report()
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
