"""P13: source-level software performance and reliability.

Fixture-first tests for the device and serialization hardening in
flowforge/models.py (TimeEmbed.forward, BookDDPM.sample) and the
checkpoint audit in examples/demo.py.

Independent defect fixture: TimeEmbed used to build its frequency tensor
with torch.arange on CPU, so a model moved to any other device failed in
forward with a device mismatch. The meta device reproduces that mismatch
on a CPU-only machine. BookDDPM.sample used to return non-finite output
silently for a NaN regime; the failed-input tests pin the rejection.

The CPU baseline numbers below were recorded from the pre-change code on
the deterministic CPU backend (same model config, seeds, and regime) and
must not move: mean -0.0873168111, std 1.8965748549, min -5.0644407272,
max 4.9570026398 for the pinned sample fixture.
"""

import importlib.util
import math
import os

import pytest
import torch

from flowforge import checkpoints
from flowforge.models import BookDDPM, TimeEmbed

REGIME2 = torch.tensor([[0.5, -0.25], [1.0, 0.75]], dtype=torch.float32)
REGIME3 = torch.tensor([[0.5, -0.25], [1.0, 0.75], [0.0, 0.0]],
                       dtype=torch.float32)

BASE_CFG = {"window": 12, "t_steps": 20, "ch": 16, "n_blocks": 1}
TINY_CFG = {"n_features": 5, "window": 8, "t_steps": 10, "ch": 8,
            "n_blocks": 1}
NORMALIZER5 = {
    "feature_mean": [0.1, -0.2, 0.0, 0.3, -0.1],
    "feature_std": [1.0, 2.0, 0.5, 1.5, 1.0],
    "regime_mean": [0.0, 0.0],
    "regime_std": [1.0, 1.0],
}
COMMIT = "367a20408d12df0de644cffacedc2693f130bc4e"


def _base_model():
    model = BookDDPM(seed=5, **BASE_CFG)
    model.eval()
    return model


def _load_demo_module():
    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "demo.py")
    spec = importlib.util.spec_from_file_location("examples_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- TimeEmbed

def test_stage3_flowforge_13_timeembed_runs_on_meta_device():
    # The defect fixture: before the fix this raised RuntimeError
    # (frequency tensor on cpu, timestep tensor on meta).
    te = TimeEmbed(16).to("meta")
    t = torch.tensor([0, 1, 2], dtype=torch.long, device="meta")
    out = te(t)
    assert out.device.type == "meta"
    assert out.shape == (3, 16)


def test_stage3_flowforge_13_timeembed_freqs_follow_dtype():
    te = TimeEmbed(16)
    assert isinstance(te.freqs, torch.Tensor)
    assert te.freqs.dtype == torch.float32
    te = te.double()
    assert te.freqs.dtype == torch.float64
    out = te(torch.tensor([0, 3], dtype=torch.long))
    assert out.dtype == torch.float64


def test_stage3_flowforge_13_timeembed_known_answer_cpu():
    torch.manual_seed(0)
    te = TimeEmbed(64)
    te.eval()
    t = torch.tensor([0, 1, 5, 99], dtype=torch.long)
    out = te(t)
    # Independent known answer: sinusoid built by hand, then the mlp.
    half = 32
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    expected = te.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=1))
    assert torch.allclose(out, expected, atol=1e-6)
    # Values recorded from the pre-change implementation.
    assert [float(v) for v in out.detach().ravel()[:4]] == pytest.approx(
        [-0.15006751, 0.18397820, 0.09235460, 0.07538748], abs=1e-6)


# ------------------------------------------------------- sampler guards

def test_stage3_flowforge_13_sample_rejects_nonfinite_regime():
    model = _base_model()
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        regime = REGIME2.clone()
        regime[0, 0] = bad_val
        with pytest.raises(ValueError):
            model.sample(regime, seed=1)


def test_stage3_flowforge_13_sample_rejects_malformed_inputs():
    model = _base_model()
    with pytest.raises(ValueError):
        model.sample(torch.tensor([0.5, -0.25]), seed=1)  # 1-D regime
    with pytest.raises(ValueError):
        model.sample(torch.zeros(0, 2), seed=1)  # empty batch
    with pytest.raises(ValueError):
        model.sample(torch.zeros(2, 3), seed=1)  # wrong regime width
    with pytest.raises(ValueError):
        model.sample(REGIME2, seed=1, batch_size=0)
    with pytest.raises(ValueError):
        model.sample("not a tensor", seed=1)


def test_stage3_flowforge_13_finite_guard_trips_on_bad_denoiser():
    model = _base_model()
    orig_forward = model.net.forward

    def bad_forward(x, t, regime):
        return torch.full_like(x, float("nan"))

    model.net.forward = bad_forward
    try:
        with pytest.raises(FloatingPointError):
            model.sample(REGIME2, seed=1)
    finally:
        model.net.forward = orig_forward


# ------------------------------------------- reproducibility and batching

def test_stage3_flowforge_13_fixed_seed_reproducible_and_batched():
    model = _base_model()
    full = model.sample(REGIME3, seed=123)
    again = model.sample(REGIME3, seed=123)
    assert torch.equal(full, again)
    # Batched calls are reproducible for a fixed seed and batch size,
    # including a final partial chunk (3 rows, batch_size 2). Chunking
    # changes the draw blocking, so batched values are not the unbatched
    # stream; the unbatched default above is the pinned baseline.
    by_two = model.sample(REGIME3, seed=123, batch_size=2)
    by_two_again = model.sample(REGIME3, seed=123, batch_size=2)
    assert by_two.shape == full.shape
    assert bool(torch.isfinite(by_two).all())
    assert torch.equal(by_two, by_two_again)
    by_one = model.sample(REGIME3, seed=123, batch_size=1)
    assert by_one.shape == full.shape
    assert bool(torch.isfinite(by_one).all())
    other = model.sample(REGIME3, seed=124)
    assert not torch.equal(full, other)


def test_stage3_flowforge_13_sample_leaves_global_rng_unchanged():
    model = _base_model()
    torch.manual_seed(2024)
    before = torch.get_rng_state()
    model.sample(REGIME2, seed=9)
    assert torch.equal(torch.get_rng_state(), before)


def test_stage3_flowforge_13_cpu_baseline_unchanged():
    model = _base_model()
    out = model.sample(REGIME2, seed=123)
    arr = out.numpy()
    assert arr.shape == (2, 12, 7)
    assert float(arr.mean()) == pytest.approx(-0.0873168111, abs=1e-5)
    assert float(arr.std()) == pytest.approx(1.8965748549, abs=1e-5)
    assert float(arr.min()) == pytest.approx(-5.0644407272, abs=1e-5)
    assert float(arr.max()) == pytest.approx(4.9570026398, abs=1e-5)
    assert [float(v) for v in arr.ravel()[:4]] == pytest.approx(
        [1.1223431826, -0.4336493313, 1.9317213297, -0.8071666956],
        abs=1e-5)


# ------------------------------------------------------ serialization

def _save_tiny(tmp_path):
    torch.manual_seed(0)
    model = BookDDPM(**TINY_CFG)
    dest = str(tmp_path / "ckpt")
    manifest = checkpoints.save_inference_checkpoint(
        dest, model, "BookDDPM", dict(TINY_CFG), NORMALIZER5,
        commit_sha=COMMIT)
    return model, manifest


def test_stage3_flowforge_13_state_save_load_bit_compatible(tmp_path):
    model, _ = _save_tiny(tmp_path)
    loaded, _manifest = checkpoints.load_checkpoint(str(tmp_path / "ckpt"))
    mine = model.state_dict()
    theirs = loaded.state_dict()
    assert set(mine) == set(theirs)
    assert all(torch.equal(mine[k], theirs[k]) for k in mine)
    regime = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    assert torch.equal(model.sample(regime, seed=7),
                       loaded.sample(regime, seed=7))


def test_stage3_flowforge_13_checkpoint_checksum_audit_rejects_tamper(
        tmp_path):
    _save_tiny(tmp_path)
    weights = tmp_path / "ckpt" / "model.pt"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(str(tmp_path / "ckpt"))


def test_stage3_flowforge_13_demo_checkpoint_audit(tmp_path):
    demo = _load_demo_module()
    root = tmp_path / "ckpts"
    root.mkdir(parents=True)
    from flowforge.models import CondVAE
    torch.manual_seed(0)
    vae = CondVAE(latent_dim=4, n_features=5, window=8, regime_dim=2)
    checkpoints.save_inference_checkpoint(
        str(root / "vae"), vae, "CondVAE",
        {"latent_dim": 4, "n_features": 5, "window": 8, "regime_dim": 2},
        NORMALIZER5, commit_sha=COMMIT)
    torch.manual_seed(0)
    ddpm = BookDDPM(**TINY_CFG)
    checkpoints.save_inference_checkpoint(
        str(root / "ddpm"), ddpm, "BookDDPM", dict(TINY_CFG),
        NORMALIZER5, commit_sha=COMMIT)
    audited = demo.audit_saved_checkpoints(str(root))
    assert set(audited) == {"vae", "ddpm"}
    weights = root / "ddpm" / "model.pt"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        demo.audit_saved_checkpoints(str(root))


# ------------------------------------------------------------- hardware

def test_stage3_flowforge_13_accelerator_smoke_if_supported():
    if not torch.cuda.is_available():
        pytest.skip("no supported accelerator on this machine")
    model = BookDDPM(seed=5, **BASE_CFG).to("cuda")
    model.eval()
    out = model.sample(REGIME2.to("cuda"), seed=123)
    assert out.device.type == "cuda"
    assert bool(torch.isfinite(out).all())
