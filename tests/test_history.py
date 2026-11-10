"""Stage 3 P03 tests: causal history conditioning.

Fixture-first: these tests define the P03 contract before the
implementation. All tests in this file are prefixed
test_stage3_flowforge_03_ (the shared unit prefix, as in the P02
tests in tests/test_factorized.py).

The contract under test:

- every window can carry a past-context tensor holding the full
  LOOKBACK seconds of book features immediately before the window
  start, drawn strictly from earlier seconds inside the same
  recording segment, plus an availability mask;
- the context is standardized with the same train-split feature
  statistics as the windows, roundtrips through save/load
  bit-identically, and is absent (None) for datasets built or
  saved without it;
- the leakage gate: perturbing a target window and every later bar
  leaves that window's past context bit-identical;
- a context whose lookback would cross a segment boundary or is
  too short is rejected, never borrowed from a neighbouring
  segment;
- the context encoder is causal (a context position never sees
  later positions) and honours the availability mask (masked-out
  positions cannot influence the embedding);
- BookDDPM and CondVAE accept history conditioning as an opt-in:
  with it off, construction, training, and sampling are exactly
  as before; with it on, the context is required and actually
  flows into the output.
"""

import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from flowforge.data import (FEATURE_NAMES, LOOKBACK, N_FEATURES, WINDOW,
                            build_dataset, extract_past_context,
                            load_dataset, save_dataset)
from flowforge.models import (BookDDPM, CausalContextEncoder, CondVAE,
                              Denoiser, scoped_rng, train_ddpm, train_vae)


def _synth_chunk(tmp_path, name, n_sec=900, seed=0, t0_ms=1_700_000_000_000):
    rng = np.random.default_rng(seed)
    n_q = n_sec * 10
    times = t0_ms + np.arange(n_q) * 100
    mid = 80000 * np.exp(np.cumsum(rng.normal(0, 2e-5, n_q)))
    spread = np.abs(rng.normal(1.0, 0.3, n_q)).clip(0.1)
    q = pd.DataFrame({"time": times, "bid_px": mid - spread / 2,
                      "bid_sz": rng.uniform(0.01, 2.0, n_q),
                      "ask_px": mid + spread / 2,
                      "ask_sz": rng.uniform(0.01, 2.0, n_q)})
    n_t = n_sec * 4
    t = pd.DataFrame({"time": np.sort(rng.integers(t0_ms, t0_ms + n_sec * 1000, n_t)),
                      "price": 80000 + rng.normal(0, 5, n_t),
                      "qty": rng.uniform(0.001, 0.5, n_t),
                      "is_buyer_maker": rng.random(n_t) < 0.5})
    d = tmp_path / name
    d.mkdir()
    q.to_parquet(d / "quotes.parquet")
    t.to_parquet(d / "trades.parquet")
    return str(d)


def _three_chunks(tmp_path, n_sec=900, seed0=0):
    return [_synth_chunk(tmp_path, f"c{i}", n_sec=n_sec,
                         t0_ms=1_700_000_000_000 + i * 2_000_000,
                         seed=seed0 + i)
            for i in range(3)]


# ------------------------------------------------------------- data side

def test_stage3_flowforge_03_context_is_the_strictly_past_segment_rows(tmp_path):
    from flowforge.data import _bars_from_chunk, _load_chunk, _segments
    dirs = _three_chunks(tmp_path, n_sec=1200)
    ds = build_dataset(dirs, with_past_context=True)
    # default split with 3 sessions: train is session 0
    sw = ds.train
    assert sw.past_context is not None and sw.past_mask is not None
    assert sw.past_context.shape == (sw.x.shape[0], LOOKBACK, N_FEATURES)
    assert sw.past_context.dtype == np.float32
    assert sw.past_mask.shape == (sw.x.shape[0], LOOKBACK)
    assert sw.past_mask.dtype == bool
    assert sw.past_mask.all()
    # Recompute the expected context independently: the raw feature
    # rows of the same segment in the 60 seconds ending one second
    # before the window start, standardized with the train stats.
    q, t = _load_chunk(dirs[0])
    bars = _bars_from_chunk(q, t)
    segs = _segments(bars)
    checked = 0
    for i in range(sw.x.shape[0]):
        if int(sw.session_ids[i]) != 0:
            continue
        ctx_raw = sw.past_context[i] * ds.feature_std + ds.feature_mean
        found = False
        for seg in segs:
            secs = seg.index.to_numpy()
            pos = np.where(secs == int(sw.t_start[i]))[0]
            if len(pos) == 0:
                continue
            p = int(pos[0])
            if p < LOOKBACK:
                continue
            expect = seg[FEATURE_NAMES].to_numpy()[p - LOOKBACK:p]
            assert np.allclose(ctx_raw, expect, atol=1e-5)
            # strictly past: the context's last second is t_start - 1
            assert secs[p - 1] == int(sw.t_start[i]) - 1
            found = True
            checked += 1
        assert found
    assert checked > 0


def test_stage3_flowforge_03_leakage_gate_perturbation(tmp_path):
    # Perturbing every bar at or after a window's start must leave
    # that window's past context (and regime) bit-identical, while
    # the window itself moves.
    t0 = 1_700_000_000_000
    c1 = _synth_chunk(tmp_path, "c1", n_sec=1200, t0_ms=t0, seed=0)
    c2 = _synth_chunk(tmp_path, "c2", n_sec=1200, t0_ms=t0 + 2_000_000, seed=1)
    c3a = _synth_chunk(tmp_path, "c3a", n_sec=1200, t0_ms=t0 + 4_000_000, seed=2)
    ds = build_dataset([c1, c2, c3a], with_past_context=True)
    assert set(np.unique(ds.test.session_ids).tolist()) == {2}

    t3 = t0 + 4_000_000
    cut_sec = t3 // 1000 + 1 + 1090
    cut_ms = cut_sec * 1000
    c3b = tmp_path / "c3b"
    c3b.mkdir()
    q = pd.read_parquet(f"{c3a}/quotes.parquet")
    after = q["time"] >= cut_ms
    q.loc[after, "bid_px"] *= 1.5
    q.loc[after, "ask_px"] *= 1.5
    q.to_parquet(c3b / "quotes.parquet")
    tr = pd.read_parquet(f"{c3a}/trades.parquet")
    tr.loc[tr["time"] >= cut_ms, "price"] *= 1.5
    tr.to_parquet(c3b / "trades.parquet")
    ds2 = build_dataset([c1, c2, str(c3b)], with_past_context=True)

    assert np.array_equal(ds.test.t_start, ds2.test.t_start)
    idx = int(np.where(ds.test.t_start == cut_sec)[0][0])
    # the gate: context and regime of the cut window are untouched
    assert np.array_equal(ds.test.past_context[idx],
                          ds2.test.past_context[idx])
    assert np.array_equal(ds.test.regime[idx], ds2.test.regime[idx])
    # the perturbation was real: the window itself moved
    assert not np.array_equal(ds.test.x[idx], ds2.test.x[idx])
    # and a later window's context legitimately absorbs the cut
    assert not np.array_equal(ds.test.past_context[-1],
                              ds2.test.past_context[-1])


def test_stage3_flowforge_03_context_never_crosses_a_segment_gap(tmp_path):
    # A chunk with a 60-second hole in the middle: windows after the
    # hole must draw context only from their own side of it.
    from flowforge.data import _bars_from_chunk, _load_chunk, _segments
    rng = np.random.default_rng(5)
    n_sec, hole = 1200, (600, 660)
    n_q = n_sec * 10
    t0_ms = 1_700_000_000_000
    times = t0_ms + np.arange(n_q) * 100
    keep = ~(((times - t0_ms) // 1000 >= hole[0])
             & ((times - t0_ms) // 1000 < hole[1]))
    times = times[keep]
    mid = 80000 * np.exp(np.cumsum(rng.normal(0, 2e-5, len(times))))
    spread = np.abs(rng.normal(1.0, 0.3, len(times))).clip(0.1)
    q = pd.DataFrame({"time": times, "bid_px": mid - spread / 2,
                      "bid_sz": rng.uniform(0.01, 2.0, len(times)),
                      "ask_px": mid + spread / 2,
                      "ask_sz": rng.uniform(0.01, 2.0, len(times))})
    n_t = n_sec * 4
    tt = np.sort(rng.integers(t0_ms, t0_ms + n_sec * 1000, n_t))
    tt = tt[~((tt // 1000 - t0_ms // 1000 >= hole[0])
              & (tt // 1000 - t0_ms // 1000 < hole[1]))]
    tr = pd.DataFrame({"time": tt, "price": 80000 + rng.normal(0, 5, len(tt)),
                       "qty": rng.uniform(0.001, 0.5, len(tt)),
                       "is_buyer_maker": rng.random(len(tt)) < 0.5})
    d = tmp_path / "gapped"
    d.mkdir()
    q.to_parquet(d / "quotes.parquet")
    tr.to_parquet(d / "trades.parquet")
    dirs = [str(d),
            _synth_chunk(tmp_path, "c2", n_sec=1200,
                         t0_ms=t0_ms + 2_000_000, seed=1),
            _synth_chunk(tmp_path, "c3", n_sec=1200,
                         t0_ms=t0_ms + 4_000_000, seed=2)]
    ds = build_dataset(dirs, with_past_context=True)
    sw = ds.train
    bars = _bars_from_chunk(*_load_chunk(dirs[0]))
    segs = [seg[FEATURE_NAMES].to_numpy() for seg in _segments(bars)]
    assert len(segs) >= 2  # the hole really split the chunk
    for i in range(sw.x.shape[0]):
        if int(sw.session_ids[i]) != 0:
            continue
        joint = np.concatenate([
            sw.past_context[i] * ds.feature_std + ds.feature_mean,
            sw.x[i] * ds.feature_std + ds.feature_mean])
        # the 120-row context+window run must appear contiguously
        # inside exactly one segment
        hits = 0
        for arr in segs:
            for p in range(LOOKBACK, len(arr) - WINDOW + 1):
                if np.allclose(arr[p - LOOKBACK:p + WINDOW], joint,
                               atol=1e-5):
                    hits += 1
        assert hits == 1


def test_stage3_flowforge_03_extract_past_context_rejects_short_lookback():
    arr = np.arange(100 * N_FEATURES, dtype=np.float64).reshape(
        100, N_FEATURES)
    assert extract_past_context(arr, start=LOOKBACK - 1) is None
    assert extract_past_context(arr, start=0) is None
    out = extract_past_context(arr, start=LOOKBACK)
    assert out is not None
    ctx, mask = out
    assert np.array_equal(ctx, arr[:LOOKBACK])
    assert mask.shape == (LOOKBACK,) and mask.all()
    ctx2, _ = extract_past_context(arr, start=100)
    assert np.array_equal(ctx2, arr[100 - LOOKBACK:100])


def test_stage3_flowforge_03_context_serialization_roundtrip(tmp_path):
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs, with_past_context=True)
    p = str(tmp_path / "ds_ctx.npz")
    save_dataset(ds, p)
    ds2 = load_dataset(p)
    for a, b in ((ds.train, ds2.train), (ds.val, ds2.val),
                 (ds.test, ds2.test)):
        assert np.array_equal(a.past_context, b.past_context)
        assert np.array_equal(a.past_mask, b.past_mask)
        assert b.past_context.dtype == np.float32
        assert b.past_mask.dtype == bool
    # a dataset built without the flag carries no context, before
    # or after a roundtrip (the pre-P03 file layout loads as None)
    ds3 = build_dataset(dirs)
    assert ds3.train.past_context is None
    p3 = str(tmp_path / "ds_plain.npz")
    save_dataset(ds3, p3)
    ds4 = load_dataset(p3)
    assert ds4.train.past_context is None
    assert ds4.train.past_mask is None


def test_stage3_flowforge_03_context_shape_mismatch_fails_closed(tmp_path):
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs, with_past_context=True)
    p = str(tmp_path / "ds_ctx.npz")
    save_dataset(ds, p)
    with np.load(p, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["train_past_context"] = payload["train_past_context"][:, :30]
    bad = str(tmp_path / "ds_bad.npz")
    np.savez_compressed(bad, **payload)
    with pytest.raises(ValueError):
        load_dataset(bad)


# ------------------------------------------------------------- encoder

def _rand_context(b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, LOOKBACK, N_FEATURES, generator=g)


def test_stage3_flowforge_03_encoder_is_causal():
    enc = CausalContextEncoder(seed=0)
    enc.eval()
    ctx = _rand_context()
    with torch.no_grad():
        states = enc.encode_sequence(ctx)
        cut = 25
        perturbed = ctx.clone()
        perturbed[:, cut + 1:] = torch.randn(
            perturbed[:, cut + 1:].shape,
            generator=torch.Generator().manual_seed(99)) * 5
        states2 = enc.encode_sequence(perturbed)
    # positions up to the cut cannot see the perturbed future
    torch.testing.assert_close(states2[:, :cut + 1], states[:, :cut + 1],
                               rtol=0, atol=1e-6)
    # and the perturbation really landed downstream
    assert not torch.allclose(states2[:, -1], states[:, -1], atol=1e-5)


def test_stage3_flowforge_03_encoder_honours_mask():
    enc = CausalContextEncoder(seed=0)
    enc.eval()
    ctx = _rand_context()
    mask = torch.ones(3, LOOKBACK, dtype=torch.bool)
    mask[:, 40:] = False
    garbage = ctx.clone()
    garbage[:, 40:] = 1e4
    with torch.no_grad():
        a = enc(ctx, mask)
        b = enc(garbage, mask)
        c = enc(ctx)  # no mask: the tail counts
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    assert not torch.allclose(a, c, atol=1e-5)
    # no available positions at all: a defined zero embedding
    with torch.no_grad():
        z = enc(ctx, torch.zeros(3, LOOKBACK, dtype=torch.bool))
    assert torch.isfinite(z).all()
    assert (z == 0).all()


# ------------------------------------------------------------- models

def _rand_regime(b):
    return torch.randn(b, 2)


def test_stage3_flowforge_03_default_models_unchanged():
    a = BookDDPM(seed=7)
    b = BookDDPM(seed=7, use_history=False)
    assert list(a.state_dict()) == list(b.state_dict())
    for k in a.state_dict():
        assert torch.equal(a.state_dict()[k], b.state_dict()[k])
    x = torch.randn(4, N_FEATURES, WINDOW)
    r = _rand_regime(4)
    with scoped_rng(3):
        la = a.p_loss(x, r)
    with scoped_rng(3):
        lb = b.p_loss(x, r)
    assert float(la) == float(lb)
    out_a = a.sample(r, seed=11)
    out_b = b.sample(r, seed=11)
    assert torch.equal(out_a, out_b)
    # the plain denoiser rejects a context embedding outright
    d = Denoiser()
    with pytest.raises(ValueError):
        d(torch.randn(2, N_FEATURES, WINDOW), torch.zeros(2, dtype=torch.long),
          _rand_regime(2), torch.randn(2, 64))
    # a non-history model rejects a handed context instead of
    # silently ignoring it
    with pytest.raises(ValueError):
        a.p_loss(x, r, past_context=_rand_context(4))
    with pytest.raises(ValueError):
        a.sample(r, seed=1, past_context=_rand_context(4))
    va = CondVAE(seed=5)
    vb = CondVAE(seed=5, context_dim=0)
    assert torch.equal(va.generate(r, seed=2), vb.generate(r, seed=2))
    with pytest.raises(ValueError):
        va.generate(r, seed=2, past_context=_rand_context(4))


def test_stage3_flowforge_03_history_ddpm_trains_and_samples():
    torch.manual_seed(0)
    model = BookDDPM(seed=3, use_history=True, ch=32, n_blocks=2)
    x = torch.randn(6, WINDOW, N_FEATURES)
    r = _rand_regime(6)
    ctx = _rand_context(6, seed=4)
    mask = torch.ones(6, LOOKBACK, dtype=torch.bool)
    # the context is required, not optional, on a history model
    with pytest.raises(ValueError):
        model.p_loss(x.permute(0, 2, 1), r)
    with pytest.raises(ValueError):
        model.sample(r, seed=1)
    hist, _ = train_ddpm(model, x, r, epochs=2, batch=3, seed=1,
                         verbose=False, past_context=ctx,
                         past_mask=mask)
    assert len(hist) == 2 and np.isfinite(hist).all()
    gen = model.sample(r, seed=5, past_context=ctx, past_mask=mask)
    assert gen.shape == (6, WINDOW, N_FEATURES)
    assert torch.isfinite(gen).all()
    # the context actually conditions: a different past gives a
    # different sample under the same sampling seed
    gen2 = model.sample(r, seed=5, past_context=_rand_context(6, seed=8),
                        past_mask=mask)
    assert not torch.allclose(gen, gen2, atol=1e-5)


def test_stage3_flowforge_03_history_vae_generates_conditionally():
    vae = CondVAE(seed=5, context_dim=32)
    r = _rand_regime(4)
    ctx = _rand_context(4, seed=6)
    mask = torch.ones(4, LOOKBACK, dtype=torch.bool)
    with pytest.raises(ValueError):
        vae.generate(r, seed=2)
    gen = vae.generate(r, seed=2, past_context=ctx, past_mask=mask)
    assert gen.shape == (4, WINDOW, N_FEATURES)
    assert torch.isfinite(gen).all()
    gen2 = vae.generate(r, seed=2, past_context=_rand_context(4, seed=9),
                        past_mask=mask)
    assert not torch.allclose(gen, gen2, atol=1e-5)
    # n_per_regime repeats the context alongside the regime
    gen3 = vae.generate(r, n_per_regime=2, seed=2, past_context=ctx,
                        past_mask=mask)
    assert gen3.shape == (8, WINDOW, N_FEATURES)
    # a short training run stays finite
    x = torch.randn(6, WINDOW, N_FEATURES)
    hist = train_vae(vae, x, _rand_regime(6), epochs=2, batch=3, seed=1,
                     verbose=False, past_context=_rand_context(6, seed=7),
                     past_mask=torch.ones(6, LOOKBACK, dtype=torch.bool))
    assert len(hist) == 2 and np.isfinite(hist).all()


def test_stage3_flowforge_03_ablation_smoke():
    from examples.history_ablate import run_ablation
    out = run_ablation(seeds=(7,), epochs=2, n_train=48, n_eval=24,
                       ch=16, n_blocks=1)
    assert out["fixture_only"] is True
    assert out["comparable_to_published_real_data"] is False
    assert out["verdict"] in ("improvement", "null")
    for arm in ("regime_only", "history", "no_history", "chunk_copy"):
        assert arm in out["arms"]
        for k in ("mean_w1", "flow_acf_mae", "discriminator_auc",
                  "cond_w1_mean", "cond_mmd_mean", "nn_ratio"):
            assert np.isfinite(out["arms"][arm][k]), (arm, k)
