"""FF-N01: exact DDPM/VAE inference checkpoints and faithful reload.

Fixture-first tests for flowforge/checkpoints.py. Tiny models only: no
training runs, no real dataset, no retrain. The fresh-process test reloads
a checkpoint in a subprocess through the module's own CLI entry point
(`python -m flowforge.checkpoints generate`) so the reload path is
exercised outside the saving process.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from flowforge import checkpoints
from flowforge.models import EMA, BookDDPM, CondVAE

VENV_PY = "/home/hatch/workspace/flowforge/.venv/bin/python"

TINY_DDPM = {"n_features": 5, "window": 8, "t_steps": 10, "ch": 8,
              "n_blocks": 1}
TINY_VAE = {"latent_dim": 4, "n_features": 5, "window": 8, "regime_dim": 2}

NORMALIZER = {
    "feature_mean": [0.1, -0.2, 0.0, 0.3, -0.1],
    "feature_std": [1.0, 2.0, 0.5, 1.5, 1.0],
    "regime_mean": [0.0, 0.0],
    "regime_std": [1.0, 1.0],
}


def _head_sha():
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True, cwd=os.getcwd())
    return out.stdout.strip()


def _tiny_ddpm(seed=0):
    torch.manual_seed(seed)
    return BookDDPM(**TINY_DDPM)


def _tiny_vae(seed=0):
    torch.manual_seed(seed)
    return CondVAE(**TINY_VAE)


def _ema_after_update(model, decay=0.9):
    ema = EMA(model, decay=decay)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    return ema


def test_stage3_flowforge_n01_inference_roundtrip_ddpm(tmp_path):
    model = _tiny_ddpm()
    path = tmp_path / "ddpm"
    checkpoints.save_inference_checkpoint(
        str(path), model, "BookDDPM", TINY_DDPM, NORMALIZER,
        split_manifest_sha256="deadbeef" * 8)
    loaded, manifest = checkpoints.load_checkpoint(str(path))
    assert manifest["format"] == "ff-checkpoint/1"
    assert manifest["format_version"] == 1
    assert manifest["kind"] == "inference"
    assert manifest["model_class"] == "BookDDPM"
    assert manifest["config"] == TINY_DDPM
    assert manifest["weights_source"] == "trained"
    assert manifest["split_manifest_sha256"] == "deadbeef" * 8
    assert manifest["normalizer"] == NORMALIZER
    assert manifest["commit_sha"] == _head_sha()
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_stage3_flowforge_n01_inference_roundtrip_vae(tmp_path):
    model = _tiny_vae()
    path = tmp_path / "vae"
    checkpoints.save_inference_checkpoint(
        str(path), model, "CondVAE", TINY_VAE, NORMALIZER)
    loaded, manifest = checkpoints.load_checkpoint(str(path))
    assert manifest["model_class"] == "CondVAE"
    assert manifest["split_manifest_sha256"] is None
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_stage3_flowforge_n01_ema_effective_weights_equal_shadow(tmp_path):
    model = _tiny_ddpm()
    ema = _ema_after_update(model)
    raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert any(not torch.equal(raw[k], ema.shadow[k])
               for k in ema.shadow)
    path = tmp_path / "ema"
    checkpoints.save_inference_checkpoint(
        str(path), model, "BookDDPM", TINY_DDPM, NORMALIZER, ema=ema)
    loaded, manifest = checkpoints.load_checkpoint(str(path))
    assert manifest["weights_source"] == "ema"
    assert manifest["ema_decay"] == 0.9
    for k, v in ema.shadow.items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_stage3_flowforge_n01_fresh_process_generation_matches(tmp_path):
    model = _tiny_ddpm(seed=3)
    path = tmp_path / "gen"
    checkpoints.save_inference_checkpoint(
        str(path), model, "BookDDPM", TINY_DDPM, NORMALIZER)
    torch.manual_seed(11)
    regime = torch.randn(2, 2)
    regime_path = tmp_path / "regime.npz"
    np.savez(regime_path, regime=regime.numpy())
    with torch.no_grad():
        in_process = model.sample(regime, seed=7).numpy()
    in_process_inv = checkpoints.inverse_transform(in_process, NORMALIZER)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd()
    out_path = tmp_path / "samples.npz"
    proc = subprocess.run(
        [VENV_PY, "-m", "flowforge.checkpoints", "generate",
         "--checkpoint", str(path), "--regime", str(regime_path),
         "--seed", "7", "--out", str(out_path)],
        capture_output=True, text=True, env=env, cwd=os.getcwd())
    assert proc.returncode == 0, proc.stderr
    printed = json.loads(proc.stdout.strip().splitlines()[-1])
    saved = np.load(out_path)
    assert np.allclose(saved["samples"], in_process, atol=1e-7)
    assert np.allclose(saved["samples_inverse"], in_process_inv, atol=1e-7)
    assert printed["samples_sha256"] == hashlib.sha256(
        np.ascontiguousarray(in_process, dtype=np.float64).tobytes()
    ).hexdigest()
    assert printed["samples_inverse_sha256"] == hashlib.sha256(
        np.ascontiguousarray(in_process_inv, dtype=np.float64).tobytes()
    ).hexdigest()


def test_stage3_flowforge_n01_altered_manifest_fails(tmp_path):
    model = _tiny_ddpm()
    path = tmp_path / "ckpt"
    checkpoints.save_inference_checkpoint(
        str(path), model, "BookDDPM", TINY_DDPM, NORMALIZER)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["ch"] = 16
    manifest_path.write_text(json.dumps(manifest, indent=1))
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(str(path))


def test_stage3_flowforge_n01_altered_weights_fails(tmp_path):
    model = _tiny_ddpm()
    path = tmp_path / "ckpt"
    checkpoints.save_inference_checkpoint(
        str(path), model, "BookDDPM", TINY_DDPM, NORMALIZER)
    weights_path = path / "model.pt"
    blob = bytearray(weights_path.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    weights_path.write_bytes(bytes(blob))
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(str(path))


def test_stage3_flowforge_n01_wrong_version_fails(tmp_path):
    model = _tiny_ddpm()
    path = tmp_path / "ckpt"
    checkpoints.save_inference_checkpoint(
        str(path), model, "BookDDPM", TINY_DDPM, NORMALIZER)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 999
    manifest_path.write_text(json.dumps(manifest, indent=1))
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(str(path))


def test_stage3_flowforge_n01_config_weights_mismatch_refused_at_save(tmp_path):
    model = _tiny_ddpm()
    wrong = dict(TINY_DDPM, ch=16)
    with pytest.raises(ValueError):
        checkpoints.save_inference_checkpoint(
            str(tmp_path / "ckpt"), model, "BookDDPM", wrong, NORMALIZER)


def test_stage3_flowforge_n01_resumable_optimizer_step_deterministic(tmp_path):
    torch.manual_seed(0)
    x = torch.randn(8, 8, 5)
    regime = torch.randn(8, 2)

    def run(resume_from=None):
        torch.manual_seed(0)
        model = _tiny_ddpm(seed=0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        if resume_from is not None:
            checkpoints.load_resumable_checkpoint(
                resume_from, model, opt)
        for _ in range(3 if resume_from is None else 1):
            loss = model.p_loss(x.permute(0, 2, 1).contiguous(), regime)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return {k: v.detach().clone()
                for k, v in model.state_dict().items()}

    torch.manual_seed(0)
    model = _tiny_ddpm(seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        loss = model.p_loss(x.permute(0, 2, 1).contiguous(), regime)
        opt.zero_grad()
        loss.backward()
        opt.step()
    path = tmp_path / "resumable"
    checkpoints.save_resumable_checkpoint(
        str(path), model, opt, "BookDDPM", TINY_DDPM, NORMALIZER)
    uninterrupted = run()
    resumed = run(resume_from=str(path))
    assert set(uninterrupted) == set(resumed)
    for k in uninterrupted:
        assert torch.equal(uninterrupted[k], resumed[k])


def test_stage3_flowforge_n01_resumable_rng_state_roundtrip(tmp_path):
    import random
    model = _tiny_ddpm()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    path = tmp_path / "rng"
    checkpoints.save_resumable_checkpoint(
        str(path), model, opt, "BookDDPM", TINY_DDPM, NORMALIZER)
    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)
    checkpoints.load_resumable_checkpoint(str(path), model, opt)
    assert torch.equal(
        torch.randn(4),
        torch.randn(4, generator=torch.Generator().manual_seed(123)))
    assert random.random() == random.Random(123).random()
    assert float(np.random.rand()) == float(np.random.RandomState(123).rand())


def test_stage3_flowforge_n01_historic_weights_unavailable():
    assert checkpoints.HISTORIC_WEIGHTS_STATUS.startswith("unavailable")


def test_stage3_flowforge_n01_inverse_transform_matches_manual():
    samples = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    out = checkpoints.inverse_transform(samples, NORMALIZER)
    expected = samples * np.array(NORMALIZER["feature_std"]) + np.array(
        NORMALIZER["feature_mean"])
    assert np.allclose(out, expected)
