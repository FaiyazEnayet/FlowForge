"""Stage 3 flowforge FF-N02: RNG-isolated model init, training, inference.

Each model init, training run, and sampling call must use its own named
stream, derived from a stable run ID, never the shared global RNG. So:

- building models in any order, with arbitrary global draws in between,
  yields per-model initialized tensors that are exactly equal;
- a fixed checkpoint/sampling stream reproduces to 1e-7 on CPU;
- no isolated call changes the caller's global RNG state.

Fixture-first: these tests define the FF-N02 contract before implementation.
All tests in this file are prefixed test_ffn02_.
"""

import hashlib
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from flowforge.data import N_FEATURES, WINDOW
from flowforge.models import (
    BookDDPM,
    CondVAE,
    derive_seed,
    train_ddpm,
    train_vae,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

VAE_SEED = 123
DDPM_SEED = 456


def _tiny_regime(n=4):
    return torch.zeros(n, 2)


def _tiny_windows(n=16):
    g = torch.Generator().manual_seed(7)
    return torch.randn(n, WINDOW, N_FEATURES, generator=g)


def _state_hash(model):
    h = hashlib.sha256()
    for _, p in model.named_parameters():
        h.update(np.ascontiguousarray(p.detach().numpy()).tobytes())
    for _, b in model.named_buffers():
        h.update(np.ascontiguousarray(b.detach().numpy()).tobytes())
    return h.hexdigest()


def _global_states():
    return (
        torch.get_rng_state().clone(),
        random.getstate(),
        np.random.get_state(),
    )


def _assert_states_equal(before, after):
    assert torch.equal(before[0], after[0]), "torch global RNG state moved"
    assert before[1] == after[1], "python random state moved"
    assert (
        before[2][0] == after[2][0]
        and np.array_equal(before[2][1], after[2][1])
        and before[2][2] == after[2][2]
    ), "numpy RNG state moved"


def _small_ddpm():
    return BookDDPM(ch=16, n_blocks=1, t_steps=10, seed=DDPM_SEED)


def test_ffn02_permuted_init_order_exact():
    """10 permuted build orders give exactly equal per-model init tensors."""
    orders = []
    for i in range(10):
        orders.append(i)
    hashes_vae, hashes_ddpm = [], []
    for i in orders:
        torch.manual_seed(1000 + i)
        # Arbitrary global draws between builds: must not leak into init.
        torch.randn(i + 1, 8)
        if i % 2 == 0:
            vae = CondVAE(seed=VAE_SEED)
            ddpm = _small_ddpm()
        else:
            ddpm = _small_ddpm()
            vae = CondVAE(seed=VAE_SEED)
        if i % 3 == 0:
            torch.randn(3, 3)
            vae2 = CondVAE(seed=VAE_SEED)
            assert _state_hash(vae2) == _state_hash(vae)
        hashes_vae.append(_state_hash(vae))
        hashes_ddpm.append(_state_hash(ddpm))
    assert len(set(hashes_vae)) == 1, "VAE init changed with build order"
    assert len(set(hashes_ddpm)) == 1, "DDPM init changed with build order"


def test_ffn02_run_id_init_matches_explicit_seed():
    """A run ID derives the same init stream as its explicit seed."""
    run_id = "ff-n02-fixture-run"
    vae_a = CondVAE(run_id=run_id)
    vae_b = CondVAE(seed=derive_seed(run_id, "condvae:init"))
    assert _state_hash(vae_a) == _state_hash(vae_b)
    ddpm_a = BookDDPM(ch=16, n_blocks=1, t_steps=10, run_id=run_id)
    ddpm_b = BookDDPM(
        ch=16, n_blocks=1, t_steps=10,
        seed=derive_seed(run_id, "bookddpm:init"))
    assert _state_hash(ddpm_a) == _state_hash(ddpm_b)


def test_ffn02_generate_no_global_contamination():
    vae = CondVAE(seed=VAE_SEED)
    regime = _tiny_regime()
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    before = _global_states()
    g1 = torch.Generator().manual_seed(5)
    out1 = vae.generate(regime, generator=g1)
    _assert_states_equal(before, _global_states())
    g2 = torch.Generator().manual_seed(5)
    out2 = vae.generate(regime, generator=g2)
    assert torch.equal(out1, out2), "same generator seed gave different output"
    out3 = vae.generate(regime, seed=5)
    assert torch.equal(out1, out3), "seed and generator paths disagree"


def test_ffn02_sample_no_global_contamination():
    ddpm = _small_ddpm()
    regime = _tiny_regime()
    torch.manual_seed(77)
    random.seed(77)
    np.random.seed(77)
    before = _global_states()
    g1 = torch.Generator().manual_seed(5)
    out1 = ddpm.sample(regime, generator=g1)
    _assert_states_equal(before, _global_states())
    out2 = ddpm.sample(regime, seed=5)
    assert torch.equal(out1, out2), "seed and generator sample paths disagree"
    # Fixed stream reproduces to 1e-7 under a different global state.
    torch.manual_seed(12345)
    out3 = ddpm.sample(regime, seed=5)
    assert float((out1 - out3).abs().max()) <= 1e-7


def test_ffn02_train_no_global_contamination():
    x = _tiny_windows()
    regime = torch.zeros(16, 2)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    before = _global_states()
    vae = CondVAE(seed=VAE_SEED)
    train_vae(vae, x, regime, epochs=1, batch=8, seed=11, verbose=False)
    _assert_states_equal(before, _global_states())
    hash_a = _state_hash(vae)
    # Same training seed after different global draws: identical weights.
    torch.manual_seed(2024)
    torch.randn(50, 50)
    vae2 = CondVAE(seed=VAE_SEED)
    train_vae(vae2, x, regime, epochs=1, batch=8, seed=11, verbose=False)
    assert _state_hash(vae2) == hash_a


def test_ffn02_train_ddpm_scoped_and_repeatable():
    x = _tiny_windows()
    regime = torch.zeros(16, 2)
    torch.manual_seed(31)
    before = _global_states()
    ddpm = _small_ddpm()
    train_ddpm(ddpm, x, regime, epochs=1, batch=8, seed=13, verbose=False)
    _assert_states_equal(before, _global_states())
    hash_a = _state_hash(ddpm)
    torch.manual_seed(808)
    torch.randn(20, 20)
    ddpm2 = _small_ddpm()
    train_ddpm(ddpm2, x, regime, epochs=1, batch=8, seed=13, verbose=False)
    assert _state_hash(ddpm2) == hash_a


def _run_subprocess(code, hashseed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    env["PYTHONPATH"] = os.path.abspath(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip()


def test_ffn02_derive_seed_stable_across_processes():
    code = (
        "from flowforge.models import derive_seed;"
        "print(derive_seed('run-abc', 'condvae:init'));"
        "print(derive_seed('run-abc', 'bookddpm:init'));"
        "print(derive_seed('run-xyz', 'condvae:init'))"
    )
    out_a = _run_subprocess(code, 0)
    out_b = _run_subprocess(code, 12345)
    assert out_a == out_b, "derived seeds changed across processes"
    lines = out_a.splitlines()
    assert len(set(lines)) == 3, "distinct streams must derive distinct seeds"


def test_ffn02_fresh_process_permuted_sample():
    """Two fresh processes, different build orders and global draws,
    produce identical fixed-stream samples."""
    script = "\n".join([
        "import torch, hashlib, os",
        "from flowforge.models import BookDDPM, CondVAE",
        "torch.manual_seed(int(os.environ['GLOBAL_SEED']))",
        "torch.randn(7, 7)",
        "regime = torch.zeros(2, 2)",
        "if os.environ['ORDER'] == 'vae_first':",
        "    _v = CondVAE(seed=123)",
        "    m = BookDDPM(ch=16, n_blocks=1, t_steps=10, seed=456)",
        "else:",
        "    m = BookDDPM(ch=16, n_blocks=1, t_steps=10, seed=456)",
        "    _v = CondVAE(seed=123)",
        "torch.randn(3, 3)",
        "out = m.sample(regime, seed=5)",
        "print(hashlib.sha256(out.numpy().tobytes()).hexdigest())",
    ])
    env_a = dict(os.environ)
    env_a["PYTHONPATH"] = os.path.abspath(REPO_ROOT)
    env_a["ORDER"] = "vae_first"
    env_a["GLOBAL_SEED"] = "1"
    env_b = dict(os.environ)
    env_b["PYTHONPATH"] = os.path.abspath(REPO_ROOT)
    env_b["ORDER"] = "ddpm_first"
    env_b["GLOBAL_SEED"] = "999"
    proc_a = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, env=env_a, timeout=300)
    proc_b = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, env=env_b, timeout=300)
    assert proc_a.returncode == 0, proc_a.stderr[-2000:]
    assert proc_b.returncode == 0, proc_b.stderr[-2000:]
    assert proc_a.stdout.strip() == proc_b.stdout.strip()
