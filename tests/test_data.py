"""Data pipeline tests: shapes, NaN handling, and the no-leakage guarantee."""

import shutil

import numpy as np
import pandas as pd

from flowforge.data import (N_FEATURES, WINDOW, build_dataset, load_dataset,
                            save_dataset)


def _synth_chunk(tmp_path, name, n_sec=900, seed=0, t0_ms=1_700_000_000_000):
    rng = np.random.default_rng(seed)
    # 10 Hz quotes
    n_q = n_sec * 10
    times = t0_ms + np.arange(n_q) * 100
    mid = 80000 * np.exp(np.cumsum(rng.normal(0, 2e-5, n_q)))
    spread = np.abs(rng.normal(1.0, 0.3, n_q)).clip(0.1)
    bid_px = mid - spread / 2
    ask_px = mid + spread / 2
    q = pd.DataFrame({"time": times, "bid_px": bid_px,
                      "bid_sz": rng.uniform(0.01, 2.0, n_q),
                      "ask_px": ask_px,
                      "ask_sz": rng.uniform(0.01, 2.0, n_q)})
    # trades
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


def _three_chunks(tmp_path, n_sec=900, gap_ms=2_000_000, seed0=0):
    # Three sessions is the minimum for session-disjoint train/val/test.
    return [_synth_chunk(tmp_path, f"c{i}", n_sec=n_sec,
                         t0_ms=1_700_000_000_000 + i * gap_ms, seed=seed0 + i)
            for i in range(3)]


def test_build_shapes_and_no_nan(tmp_path):
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    for split in (ds.train, ds.val, ds.test):
        assert split.x.ndim == 3
        assert split.x.shape[1] == WINDOW
        assert split.x.shape[2] == N_FEATURES
        assert split.regime.shape == (split.x.shape[0], 2)
        assert np.isfinite(split.x).all()
        assert np.isfinite(split.regime).all()
    assert ds.n_chunks == 3
    assert ds.n_seconds > 0


def test_no_time_overlap_between_splits(tmp_path):
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    # every train window ends before every val window starts, etc.
    assert ds.train.t_end.max() <= ds.val.t_start.min()
    assert ds.val.t_end.max() <= ds.test.t_start.min()
    # and no window appears in two splits (check start times are unique)
    all_starts = np.concatenate([ds.train.t_start, ds.val.t_start, ds.test.t_start])
    assert len(np.unique(all_starts)) == len(all_starts)


def test_standardization_uses_train_only(tmp_path):
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    pooled = ds.train.x.reshape(-1, N_FEATURES)
    # train split standardized to ~zero mean unit variance by construction
    assert np.abs(pooled.mean(axis=0)).max() < 0.05
    assert np.abs(pooled.std(axis=0) - 1).max() < 0.05


def test_save_load_roundtrip(tmp_path):
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    ds2 = load_dataset(p)
    assert np.array_equal(ds.train.x, ds2.train.x)
    assert np.array_equal(ds.test.regime, ds2.test.regime)
    assert ds2.n_chunks == ds.n_chunks
    assert np.array_equal(ds.train.session_ids, ds2.train.session_ids)
    assert ds2.chunk_dirs == ds.chunk_dirs
    assert ds2.chunk_hashes == ds.chunk_hashes
    assert ds2.chunk_symbols == ds.chunk_symbols
    assert ds2.chunk_stats == ds.chunk_stats
    assert ds2.session_split == ds.session_split


def test_regime_uses_strictly_past_data(tmp_path):
    # Regression test: regime features for a window must not move when data
    # at or after the window start changes. The lookback ends at the window
    # start, so perturbing quotes from one window start onward must leave
    # that window's regime untouched. The perturbed chunk is the test
    # session (default split with 3 sessions: train/val/test = c1/c2/c3).
    t0 = 1_700_000_000_000
    c1 = _synth_chunk(tmp_path, "c1", n_sec=1200, t0_ms=t0, seed=0)
    c2 = _synth_chunk(tmp_path, "c2", n_sec=1200, t0_ms=t0 + 2_000_000, seed=1)
    c3a = _synth_chunk(tmp_path, "c3a", n_sec=1200, t0_ms=t0 + 4_000_000, seed=2)
    ds = build_dataset([c1, c2, c3a])
    assert set(np.unique(ds.test.session_ids).tolist()) == {2}

    # cut lands exactly on a window start: bars drop the first second, so
    # window starts sit at t3 + 1 + {60, 70, ...}; 1090 is one of them.
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
    shutil.copy(tmp_path / "c3a" / "trades.parquet", c3b / "trades.parquet")
    ds2 = build_dataset([c1, c2, str(c3b)])

    assert np.array_equal(ds.test.t_start, ds2.test.t_start)
    idx = int(np.where(ds.test.t_start == cut_sec)[0][0])
    assert np.array_equal(ds.test.regime[idx], ds2.test.regime[idx])
    # the perturbation was real: the last test window's lookback covers
    # the cut, so its volatility moves
    assert not np.array_equal(ds.test.regime[-1], ds2.test.regime[-1])


def test_large_gap_between_chunks_keeps_splits_populated(tmp_path):
    # Regression test: chunks recorded hours apart must not leave a split
    # empty. Splits are session-disjoint, so with three sessions the late
    # one becomes the test split by construction.
    t0 = 1_700_000_000_000
    d1 = _synth_chunk(tmp_path, "c1", n_sec=1200, t0_ms=t0, seed=0)
    d2 = _synth_chunk(tmp_path, "c2", n_sec=1200, t0_ms=t0 + 1_300_000, seed=1)
    # four hours later, like a second recording session
    d3 = _synth_chunk(tmp_path, "c3", n_sec=1200,
                      t0_ms=t0 + 4 * 3600_000 + 1_300_000, seed=2)
    ds = build_dataset([d1, d2, d3])
    assert len(ds.train.x) > 0 and len(ds.val.x) > 0 and len(ds.test.x) > 0
    # the late session (4h later) is the whole test split: every test
    # window starts after the 4h gap
    assert set(np.unique(ds.test.session_ids).tolist()) == {2}
    assert ds.test.t_start.min() >= t0 / 1000 + 4 * 3600
    # sessions are chronological, so splits stay time-ordered
    assert ds.train.t_end.max() <= ds.val.t_start.min()
    assert ds.val.t_end.max() <= ds.test.t_start.min()
    all_starts = np.concatenate([ds.train.t_start, ds.val.t_start, ds.test.t_start])
    assert len(np.unique(all_starts)) == len(all_starts)


def test_overlapping_chunks_raise(tmp_path):
    # Overlapping sessions would let the same market seconds land in two
    # splits. That must fail loudly, never silently.
    # Contract change (C02): overlapping raw sessions are now detected
    # on raw quote time ranges before any chunk is dropped, raising
    # ValueError instead of the old bar-range assert.
    import pytest
    t0 = 1_700_000_000_000
    d1 = _synth_chunk(tmp_path, "c1", n_sec=1200, t0_ms=t0, seed=0)
    d2 = _synth_chunk(tmp_path, "c2", n_sec=1200, t0_ms=t0 + 600_000, seed=1)
    with pytest.raises(ValueError, match="overlap"):
        build_dataset([d1, d2])


def _synth_chunk_with_gap(tmp_path, name, gap_sec, n_sec=900, seed=0,
                          t0_ms=1_700_000_000_000):
    """Like _synth_chunk but with no quotes at all during second gap_sec."""
    d = _synth_chunk(tmp_path, name, n_sec=n_sec, seed=seed, t0_ms=t0_ms)
    q = pd.read_parquet(f"{d}/quotes.parquet")
    t0_s = t0_ms // 1000
    q = q[((q["time"] // 1000) != t0_s + gap_sec)].reset_index(drop=True)
    q.to_parquet(f"{d}/quotes.parquet")
    return d


def test_missing_second_drops_cross_gap_return(tmp_path):
    # The bar after a missing second must not carry a multi-second log
    # return, and segments must split at every gap.
    from flowforge.data import _bars_from_chunk, _load_chunk, _segments
    t0 = 1_700_000_000_000
    d = _synth_chunk_with_gap(tmp_path, "c1", gap_sec=450, t0_ms=t0)
    q, t = _load_chunk(d)
    b = _bars_from_chunk(q, t)
    secs = b.index.to_numpy()
    # the missing second and the bar after it are both gone
    assert (t0 // 1000 + 450) not in secs
    assert (t0 // 1000 + 451) not in secs
    # every segment is a run of consecutive seconds
    for seg in _segments(b):
        s = seg.index.to_numpy()
        assert np.all(np.diff(s) == 1), "segment must be consecutive seconds"


def test_windows_span_exactly_59_seconds(tmp_path):
    # End to end: with gaps in the data, every accepted 60-row window
    # still spans exactly 59 seconds between endpoints.
    from flowforge.data import WINDOW
    t0 = 1_700_000_000_000
    d1 = _synth_chunk_with_gap(tmp_path, "c1", gap_sec=200, t0_ms=t0, seed=0)
    d2 = _synth_chunk_with_gap(tmp_path, "c2", gap_sec=500,
                               t0_ms=t0 + 2_000_000, seed=1)
    d3 = _synth_chunk_with_gap(tmp_path, "c3", gap_sec=300,
                               t0_ms=t0 + 4_000_000, seed=2)
    ds = build_dataset([d1, d2, d3])
    for split in (ds.train, ds.val, ds.test):
        assert split.x.shape[1] == WINDOW
        elapsed = split.t_end - split.t_start
        assert np.all(elapsed == WINDOW - 1), \
            f"window spans {elapsed.max()}s, expected {WINDOW - 1}s"


def test_chunk_boundary_return_not_shared(tmp_path):
    # The first bar of a chunk cannot borrow the previous chunk's last
    # mid for its return: each chunk's bars are built independently.
    from flowforge.data import _bars_from_chunk, _load_chunk
    t0 = 1_700_000_000_000
    d1 = _synth_chunk(tmp_path, "c1", n_sec=300, t0_ms=t0, seed=0)
    d2 = _synth_chunk(tmp_path, "c2", n_sec=300, t0_ms=t0 + 400_000, seed=1)
    q1, t1 = _load_chunk(d1)
    q2, t2 = _load_chunk(d2)
    b1 = _bars_from_chunk(q1, t1)
    b2 = _bars_from_chunk(q2, t2)
    # chunk 2's first kept bar is its second recorded second (first dropped:
    # no previous second inside the chunk)
    assert b2.index.min() == t0 // 1000 + 400 + 1
    # and its return uses only chunk-2 mids
    assert np.isfinite(b2["log_ret"]).all()


def test_session_disjoint_no_shared_sessions(tmp_path):
    # FL3: whole sessions go to one split; no session id appears in two
    # splits. Four synthetic chunks -> default 70/15/15 by chunk count.
    dirs = [_synth_chunk(tmp_path, f"c{i}", n_sec=900,
                         t0_ms=1_700_000_000_000 + i * 2_000_000, seed=i)
            for i in range(4)]
    ds = build_dataset(dirs)
    seen = {}
    for name in ("train", "val", "test"):
        for sid in np.unique(getattr(ds, name).session_ids).tolist():
            assert sid not in seen, f"session {sid} in two splits"
            seen[sid] = name
    assert len(seen) == 4
    # session_ids are valid chunk indices
    for name in ("train", "val", "test"):
        sids = getattr(ds, name).session_ids
        assert ((sids >= 0) & (sids < 4)).all()


def test_explicit_session_manifest_respected(tmp_path):
    # An explicit manifest locks test session identities before sweeps.
    dirs = [_synth_chunk(tmp_path, f"c{i}", n_sec=900,
                         t0_ms=1_700_000_000_000 + i * 2_000_000, seed=i)
            for i in range(4)]
    ds = build_dataset(dirs, session_split={"train": [0, 1],
                                            "val": [2], "test": [3]})
    assert set(np.unique(ds.train.session_ids).tolist()) == {0, 1}
    assert set(np.unique(ds.val.session_ids).tolist()) == {2}
    assert set(np.unique(ds.test.session_ids).tolist()) == {3}
    assert ds.session_split == {"train": [0, 1], "val": [2], "test": [3]}


def test_session_manifest_must_cover_each_chunk_once(tmp_path):
    import pytest
    dirs = [_synth_chunk(tmp_path, f"c{i}", n_sec=900,
                         t0_ms=1_700_000_000_000 + i * 2_000_000, seed=i)
            for i in range(3)]
    with pytest.raises(ValueError):
        build_dataset(dirs, session_split={"train": [0, 1],
                                           "val": [2], "test": [2]})
    with pytest.raises(ValueError):
        build_dataset(dirs, session_split={"train": [0], "val": [1],
                                           "test": []})


def test_write_manifest_and_dataset_card(tmp_path):
    import json
    from flowforge.data import dataset_card, write_manifest
    dirs = [_synth_chunk(tmp_path, f"c{i}", n_sec=900,
                         t0_ms=1_700_000_000_000 + i * 2_000_000, seed=i)
            for i in range(4)]
    for d in dirs:
        with open(f"{d}/meta.json", "w") as f:
            json.dump({"symbol": "BTCUSDT"}, f)
    ds = build_dataset(dirs)
    assert ds.chunk_symbols == ["BTCUSDT"] * 4
    mp = tmp_path / "manifest.json"
    manifest = write_manifest(ds, str(mp))
    assert mp.exists()
    assert manifest["n_chunks"] == 4
    assert len(manifest["chunks"]) == 4
    assert all(len(c["sha256"]) == 64 for c in manifest["chunks"])
    assert all(c["quotes_sha256"] and c["trades_sha256"]
               for c in manifest["chunks"])
    assert all(c["symbol"] == "BTCUSDT" for c in manifest["chunks"])
    assert all("seconds_missing" in c for c in manifest["chunks"])
    card = dataset_card(ds)
    assert "session-disjoint" in card.lower() or "Session-disjoint" in card
    assert "BTCUSDT" in card
    assert "Missingness" in card
    assert "SHA256" in card
    for s in ("train", "val", "test"):
        assert s in card


def test_fewer_than_three_sessions_raises(tmp_path):
    # Session-disjoint train/val/test needs at least three sessions; the
    # error must say so plainly instead of emitting an empty split.
    import pytest
    d1 = _synth_chunk(tmp_path, "c1", t0_ms=1_700_000_000_000)
    d2 = _synth_chunk(tmp_path, "c2", t0_ms=1_700_000_000_000 + 2_000_000,
                      seed=1)
    with pytest.raises(ValueError, match="at least 3 sessions"):
        build_dataset([d1])
    with pytest.raises(ValueError, match="at least 3 sessions"):
        build_dataset([d1, d2])


def test_skipped_empty_chunk_keeps_hashes_aligned(tmp_path):
    # A chunk with no usable bars is skipped; the surviving chunk dirs,
    # hashes, symbols and stats must stay aligned with each other.
    # Contract change (C02): the skipped chunk must NOT overlap the
    # others. An overlapping chunk is a hard error even when its bars
    # are dropped (see test_stage3_flowforge_02_overlapping_but_dropped_
    # chunk_detected), so this fixture places the empty chunk in its
    # own time range, well after the three kept sessions.
    empty = _synth_chunk(tmp_path, "empty", n_sec=1, seed=9,
                         t0_ms=1_700_000_000_000 + 20_000_000)
    dirs = _three_chunks(tmp_path)
    ds = build_dataset([empty] + dirs)
    assert ds.n_chunks == 3
    assert ds.chunk_dirs == dirs
    assert len(set(ds.chunk_hashes)) == 3
    assert ds.chunk_symbols == ["unknown"] * 3
    assert all(s["usable"] > 0 for s in ds.chunk_stats)


def test_save_load_roundtrip_preserves_sessions(tmp_path):
    dirs = [_synth_chunk(tmp_path, f"c{i}", n_sec=900,
                         t0_ms=1_700_000_000_000 + i * 2_000_000, seed=i)
            for i in range(4)]
    ds = build_dataset(dirs)
    p = tmp_path / "ds.npz"
    save_dataset(ds, str(p))
    ds2 = load_dataset(str(p))
    for name in ("train", "val", "test"):
        a = getattr(ds, name).session_ids
        b = getattr(ds2, name).session_ids
        assert np.array_equal(a, b), name
    assert ds2.session_split == ds.session_split
    assert ds2.chunk_hashes == ds.chunk_hashes
