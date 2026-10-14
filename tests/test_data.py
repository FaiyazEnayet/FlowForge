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


def test_build_shapes_and_no_nan(tmp_path):
    d1 = _synth_chunk(tmp_path, "c1", t0_ms=1_700_000_000_000)
    d2 = _synth_chunk(tmp_path, "c2", t0_ms=1_700_000_000_000 + 1_000_000, seed=1)
    ds = build_dataset([d1, d2])
    for split in (ds.train, ds.val, ds.test):
        assert split.x.ndim == 3
        assert split.x.shape[1] == WINDOW
        assert split.x.shape[2] == N_FEATURES
        assert split.regime.shape == (split.x.shape[0], 2)
        assert np.isfinite(split.x).all()
        assert np.isfinite(split.regime).all()
    assert ds.n_chunks == 2
    assert ds.n_seconds > 0


def test_no_time_overlap_between_splits(tmp_path):
    d1 = _synth_chunk(tmp_path, "c1", n_sec=900, t0_ms=1_700_000_000_000)
    d2 = _synth_chunk(tmp_path, "c2", n_sec=900,
                      t0_ms=1_700_000_000_000 + 2_000_000, seed=1)
    ds = build_dataset([d1, d2])
    # every train window ends before every val window starts, etc.
    assert ds.train.t_end.max() <= ds.val.t_start.min()
    assert ds.val.t_end.max() <= ds.test.t_start.min()
    # and no window appears in two splits (check start times are unique)
    all_starts = np.concatenate([ds.train.t_start, ds.val.t_start, ds.test.t_start])
    assert len(np.unique(all_starts)) == len(all_starts)


def test_standardization_uses_train_only(tmp_path):
    d1 = _synth_chunk(tmp_path, "c1", n_sec=900, t0_ms=1_700_000_000_000)
    ds = build_dataset([d1])
    pooled = ds.train.x.reshape(-1, N_FEATURES)
    # train split standardized to ~zero mean unit variance by construction
    assert np.abs(pooled.mean(axis=0)).max() < 0.05
    assert np.abs(pooled.std(axis=0) - 1).max() < 0.05


def test_save_load_roundtrip(tmp_path):
    d1 = _synth_chunk(tmp_path, "c1", t0_ms=1_700_000_000_000)
    ds = build_dataset([d1])
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    ds2 = load_dataset(p)
    assert np.array_equal(ds.train.x, ds2.train.x)
    assert np.array_equal(ds.test.regime, ds2.test.regime)
    assert ds2.n_chunks == ds.n_chunks


def test_regime_uses_strictly_past_data(tmp_path):
    # Regression test: regime features for a window must not move when data
    # at or after the window start changes. The lookback ends at the window
    # start, so perturbing quotes from one window start onward must leave
    # that window's regime untouched.
    t0 = 1_700_000_000_000
    d1 = _synth_chunk(tmp_path, "c1", n_sec=1200, t0_ms=t0, seed=0)
    ds = build_dataset([d1])

    # cut lands exactly on a window start: bars drop the first second, so
    # window starts sit at t0 + 1 + {60, 70, ...}; 1090 is one of them.
    cut_sec = t0 // 1000 + 1 + 1090
    cut_ms = cut_sec * 1000
    d2 = tmp_path / "c2"
    d2.mkdir()
    q = pd.read_parquet(f"{d1}/quotes.parquet")
    after = q["time"] >= cut_ms
    q.loc[after, "bid_px"] *= 1.5
    q.loc[after, "ask_px"] *= 1.5
    q.to_parquet(d2 / "quotes.parquet")
    shutil.copy(tmp_path / "c1" / "trades.parquet", d2 / "trades.parquet")
    ds2 = build_dataset([str(d2)])

    assert np.array_equal(ds.test.t_start, ds2.test.t_start)
    idx = int(np.where(ds.test.t_start == cut_sec)[0][0])
    assert np.array_equal(ds.test.regime[idx], ds2.test.regime[idx])
    # the perturbation was real: the last test window's lookback covers
    # the cut, so its volatility moves
    assert not np.array_equal(ds.test.regime[-1], ds2.test.regime[-1])


def test_large_gap_between_chunks_keeps_splits_populated(tmp_path):
    # Regression test: chunks recorded hours apart must not leave a split
    # empty. The split runs on the concatenated recording timeline, so the
    # later session lands in test (different time of day by design).
    t0 = 1_700_000_000_000
    d1 = _synth_chunk(tmp_path, "c1", n_sec=1200, t0_ms=t0, seed=0)
    d2 = _synth_chunk(tmp_path, "c2", n_sec=1200, t0_ms=t0 + 1_300_000, seed=1)
    # four hours later, like a second recording session
    d3 = _synth_chunk(tmp_path, "c3", n_sec=1200,
                      t0_ms=t0 + 4 * 3600_000 + 1_300_000, seed=2)
    ds = build_dataset([d1, d2, d3])
    assert len(ds.train.x) > 0 and len(ds.val.x) > 0 and len(ds.test.x) > 0
    # the late session (4h later) lands in val/test, not train: with 3600s
    # total, the 85% cut sits 660s into chunk 3, so all test windows start
    # after the 4h gap
    assert ds.test.t_start.min() >= t0 / 1000 + 4 * 3600
    # concatenated order is chronological, so splits stay time-ordered
    assert ds.train.t_end.max() <= ds.val.t_start.min()
    assert ds.val.t_end.max() <= ds.test.t_start.min()
    all_starts = np.concatenate([ds.train.t_start, ds.val.t_start, ds.test.t_start])
    assert len(np.unique(all_starts)) == len(all_starts)


def test_overlapping_chunks_raise(tmp_path):
    # Overlapping sessions would let the same market seconds land in two
    # splits. That must fail loudly, never silently.
    import pytest
    t0 = 1_700_000_000_000
    d1 = _synth_chunk(tmp_path, "c1", n_sec=1200, t0_ms=t0, seed=0)
    d2 = _synth_chunk(tmp_path, "c2", n_sec=1200, t0_ms=t0 + 600_000, seed=1)
    with pytest.raises(AssertionError):
        build_dataset([d1, d2])
