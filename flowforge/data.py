"""Data pipeline: recorded top-of-book sessions -> standardized book-state windows.

Source data (from toxicflow recordings):
  quotes.parquet: time (ms), bid_px, bid_sz, ask_px, ask_sz  (top of book, ~10 Hz)
  trades.parquet: time (ms), price, qty, is_buyer_maker

Representation choice: the recorder saves only the touch, not full depth, so a
top-K depth representation is not possible. Instead each second becomes one
book-state vector built from 1s bars of quotes plus signed trade flow:

  0. log midprice return vs previous second
  1. log1p(spread in bps)
  2. log(best bid size)
  3. log(best ask size)
  4. book imbalance (bid_sz - ask_sz) / (bid_sz + ask_sz), in [-1, 1]
  5. signed trade volume that second (buy positive), standardized later
  6. log1p(trade count that second)

Leakage discipline (non-negotiable):
  - Bars are standardized with mean/std fit on the TRAIN split only.
  - Splits are session-disjoint: whole recording sessions (chunks) are
    assigned to train, val, or test, never split across them. By default
    the first 70% of sessions (chronological) go to train, the next 15%
    to val, and the remainder to test; an explicit session_split manifest
    locks the assignment before any sweep. No raw second ever appears in
    two splits by construction, so no window content is ever shared
    between splits.
  - Windows never cross chunk boundaries (recording gaps/resyncs): seconds
    are segmented per chunk and a new segment starts at every missing
    second, so each segment is a run of consecutive seconds. Windows are
    built inside segments only, which guarantees every 60-row window
    spans exactly 59 seconds of elapsed time. The 1s log return is only
    defined where the previous second is present; the bar after a gap is
    dropped rather than carrying a multi-second return.
  - Regime conditioning features come strictly from a lookback that ends at
    the window start (past only, never the window itself).

Regime features per window (2 dims): realized volatility (std of 1s log
returns over the 60s lookback) and mean spread in bps over the lookback.
Standardized with train statistics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "log_ret",
    "log_spread_bps",
    "log_bid_sz",
    "log_ask_sz",
    "imbalance",
    "signed_flow",
    "log_trade_count",
]
N_FEATURES = len(FEATURE_NAMES)

WINDOW = 60          # seconds per window
STRIDE = 10          # seconds between window starts
LOOKBACK = 60        # seconds of past context for regime features

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder (0.15)


@dataclass
class SplitWindows:
    x: np.ndarray          # (n_windows, WINDOW, N_FEATURES) standardized
    regime: np.ndarray     # (n_windows, 2) standardized [vol, spread]
    t_start: np.ndarray    # window start time in seconds (absolute)
    t_end: np.ndarray      # window end time in seconds (absolute)
    session_ids: np.ndarray  # (n_windows,) int: source chunk index per window


@dataclass
class BookDataset:
    train: SplitWindows
    val: SplitWindows
    test: SplitWindows
    feature_mean: np.ndarray = field(repr=False)
    feature_std: np.ndarray = field(repr=False)
    regime_mean: np.ndarray = field(repr=False)
    regime_std: np.ndarray = field(repr=False)
    n_seconds: int = 0
    n_chunks: int = 0
    chunk_dirs: list = field(default_factory=list, repr=False)
    chunk_hashes: list = field(default_factory=list, repr=False)
    chunk_symbols: list = field(default_factory=list, repr=False)
    chunk_stats: list = field(default_factory=list, repr=False)
    session_split: dict = field(default_factory=dict, repr=False)


def _load_chunk(chunk_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_parquet(f"{chunk_dir}/quotes.parquet")
    t = pd.read_parquet(f"{chunk_dir}/trades.parquet")
    return q, t


def _chunk_symbol(chunk_dir: str) -> str:
    """Recording symbol from the chunk's meta.json, if present."""
    try:
        with open(os.path.join(chunk_dir, "meta.json")) as f:
            return str(json.load(f).get("symbol", "unknown"))
    except (OSError, ValueError):
        return "unknown"


def _bars_from_chunk(q: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """One row per second, on a regular 1-second grid.

    Seconds with no quote data are missing from the grid (never
    forward-filled). The 1s log return is only defined where the
    previous second is present: the bar after a gap is dropped, so no
    return ever spans more than one second.
    """
    q = q.copy()
    q["sec"] = (q["time"] // 1000).astype(np.int64)
    # last touch of each second
    b = q.sort_values("time").groupby("sec").last()
    b["mid"] = (b["bid_px"] + b["ask_px"]) / 2.0
    b["spread_bps"] = (b["ask_px"] - b["bid_px"]) / b["mid"] * 1e4

    t = t.copy()
    t["sec"] = (t["time"] // 1000).astype(np.int64)
    # is_buyer_maker True -> buyer was maker -> taker sold -> sell flow
    t["signed_qty"] = np.where(t["is_buyer_maker"].astype(bool), -t["qty"], t["qty"])
    agg = t.groupby("sec").agg(signed_flow=("signed_qty", "sum"),
                               trade_count=("qty", "size"))
    b = b.join(agg, how="left")
    b["signed_flow"] = b["signed_flow"].fillna(0.0)
    b["trade_count"] = b["trade_count"].fillna(0).astype(np.int64)

    b["log_ret"] = np.log(b["mid"] / b["mid"].shift(1))
    # A return is only a 1-second return when the previous second exists.
    # The bar after a gap would silently span multiple seconds, so it is
    # dropped (dropna below) rather than mislabeled.
    prev_gap = b.index.to_series().diff().ne(1)
    b.loc[prev_gap, "log_ret"] = np.nan
    b["log_spread_bps"] = np.log1p(b["spread_bps"].clip(lower=0))
    b["log_bid_sz"] = np.log(b["bid_sz"].clip(lower=1e-9))
    b["log_ask_sz"] = np.log(b["ask_sz"].clip(lower=1e-9))
    b["imbalance"] = (b["bid_sz"] - b["ask_sz"]) / (b["bid_sz"] + b["ask_sz"]).clip(lower=1e-9)
    b["imbalance"] = b["imbalance"].clip(-1.0, 1.0)
    b["log_trade_count"] = np.log1p(b["trade_count"].astype(float))

    cols = ["log_ret", "log_spread_bps", "log_bid_sz", "log_ask_sz",
            "imbalance", "signed_flow", "log_trade_count",
            "spread_bps", "mid"]
    b = b[cols].dropna()
    # sanity: drop non-finite rows rather than letting NaNs poison training
    b = b[np.isfinite(b[FEATURE_NAMES]).all(axis=1)]
    return b


def _segments(b: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a chunk's bars at every missing second.

    Regular-grid policy: a segment is a run of consecutive seconds.
    Windows are built inside segments only, so every accepted 60-row
    window spans exactly 59 seconds between its endpoints.
    """
    secs = b.index.to_numpy()
    cuts = np.where(np.diff(secs) > 1)[0] + 1
    return [b.iloc[s:e] for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(b)])]


def _windows_in_segment(seg: pd.DataFrame) -> list[tuple[int, int]]:
    """(start_pos, end_pos) index pairs fully inside the segment."""
    n = len(seg)
    out = []
    # need LOOKBACK seconds of context before the window start
    for start in range(LOOKBACK, n - WINDOW + 1, STRIDE):
        out.append((start, start + WINDOW))
    return out


def build_dataset(chunk_dirs: list[str],
                  session_split: dict[str, list[int]] | None = None) -> BookDataset:
    """Load chunks, build 1s bars, split by session, emit standardized windows.

    Session-disjoint splits (FL3): whole recording sessions (chunks) are
    assigned to splits, never split across them. By default, chunks are
    assigned chronologically: first 70% train, next 15% val, remainder test
    (by chunk count, rounding). Pass an explicit `session_split` manifest
    like {"train": [0,1,2,3,4], "val": [5], "test": [6,7]} to lock test
    session identities before sweeps. Chunk indices refer to chronological
    order. No raw second ever appears in two splits by construction.
    """
    import hashlib
    # Parallel kept-lists: a chunk with no usable bars is skipped, and the
    # surviving entries stay aligned (dir <-> bars <-> hash <-> symbol).
    kept_dirs: list[str] = []
    kept_bars = []
    kept_hashes: list[str] = []
    kept_symbols: list[str] = []
    kept_stats: list[dict] = []
    for d in chunk_dirs:
        q, t = _load_chunk(d)
        # hash the raw inputs for the manifest
        h = hashlib.sha256()
        h.update(q.to_numpy().tobytes())
        h.update(t.to_numpy().tobytes())
        b = _bars_from_chunk(q, t)
        if len(b) == 0:
            continue
        span = int(b.index.max() - b.index.min()) + 1
        kept_dirs.append(d)
        kept_bars.append(b)
        kept_hashes.append(h.hexdigest()[:16])
        kept_symbols.append(_chunk_symbol(d))
        kept_stats.append({"span": span, "usable": int(len(b)),
                           "missing": span - int(len(b))})
    if not kept_bars:
        raise ValueError("no usable bars in any chunk")
    n_seconds = int(sum(len(b) for b in kept_bars))

    # Sort chunks chronologically; guard against overlapping sessions.
    order = sorted(range(len(kept_bars)),
                   key=lambda i: kept_bars[i].index.min())
    ordered = [kept_bars[i] for i in order]
    ordered_dirs = [kept_dirs[i] for i in order]
    ordered_hashes = [kept_hashes[i] for i in order]
    ordered_symbols = [kept_symbols[i] for i in order]
    ordered_stats = [kept_stats[i] for i in order]
    for prev, cur in zip(ordered[:-1], ordered[1:]):
        assert cur.index.min() > prev.index.max(), \
            "chunk time ranges overlap; sort/deduplicate recording sessions"

    n = len(ordered)
    if session_split is None:
        # Default: chronological 70/15/15 by chunk count, whole sessions.
        # Fewer than three sessions cannot populate three disjoint splits.
        if n < 3:
            raise ValueError(
                f"need at least 3 sessions for session-disjoint "
                f"train/val/test splits, got {n}; pass an explicit "
                f"session_split or add sessions")
        n_test = max(1, int(round(n * (1.0 - TRAIN_FRAC - VAL_FRAC))))
        n_val = max(1, int(round(n * VAL_FRAC)))
        n_train = n - n_test - n_val
        session_split = {
            "train": list(range(n_train)),
            "val": list(range(n_train, n_train + n_val)),
            "test": list(range(n_train + n_val, n)),
        }
    # Validate the manifest covers every chunk exactly once.
    seen = []
    for name in ("train", "val", "test"):
        if name not in session_split:
            raise ValueError(f"session_split missing '{name}'")
        seen.extend(session_split[name])
    if sorted(seen) != list(range(n)):
        raise ValueError("session_split must assign each chunk exactly once")
    chunk_to_split = {}
    for name, idxs in session_split.items():
        for i in idxs:
            chunk_to_split[i] = name

    raw: dict[str, list] = {"train": [], "val": [], "test": []}
    raw_regime: dict[str, list] = {"train": [], "val": [], "test": []}
    raw_t: dict[str, list] = {"train": [], "val": [], "test": []}
    raw_session: dict[str, list] = {"train": [], "val": [], "test": []}

    for ci, b in enumerate(ordered):
        split = chunk_to_split[ci]
        for seg in _segments(b):
            if len(seg) < LOOKBACK + WINDOW:
                continue
            arr = seg[FEATURE_NAMES].to_numpy(dtype=np.float64)
            secs = seg.index.to_numpy()
            mid = seg["mid"].to_numpy(dtype=np.float64)
            spread = seg["spread_bps"].to_numpy(dtype=np.float64)
            for start, end in _windows_in_segment(seg):
                w_t0, w_t1 = secs[start], secs[end - 1]
                win = arr[start:end]
                # regime from strictly-past lookback: ends at the window
                # start, never includes the window itself
                lb_ret = np.diff(np.log(mid[start - LOOKBACK:start]))
                vol = float(np.std(lb_ret)) if len(lb_ret) else 0.0
                sp = float(np.mean(spread[start - LOOKBACK:start]))
                raw[split].append(win)
                raw_regime[split].append([vol, sp])
                raw_t[split].append((float(w_t0), float(w_t1)))
                raw_session[split].append(ci)

    for name in raw:
        if not raw[name]:
            raise ValueError(f"no windows in split '{name}'; check session_split")

    # standardize with TRAIN statistics only (no leakage from val/test)
    train_x = np.stack(raw["train"])
    f_mean = train_x.reshape(-1, N_FEATURES).mean(axis=0)
    f_std = train_x.reshape(-1, N_FEATURES).std(axis=0)
    f_std[f_std < 1e-12] = 1.0
    train_r = np.asarray(raw_regime["train"], dtype=np.float64)
    r_mean = train_r.mean(axis=0)
    r_std = train_r.std(axis=0)
    r_std[r_std < 1e-12] = 1.0

    def pack(name: str) -> SplitWindows:
        x = (np.stack(raw[name]) - f_mean) / f_std
        r = (np.asarray(raw_regime[name], dtype=np.float64) - r_mean) / r_std
        tt = np.asarray(raw_t[name], dtype=np.float64)
        ss = np.asarray(raw_session[name], dtype=np.int64)
        assert np.isfinite(x).all() and np.isfinite(r).all()
        return SplitWindows(x=x.astype(np.float32), regime=r.astype(np.float32),
                            t_start=tt[:, 0], t_end=tt[:, 1],
                            session_ids=ss)

    return BookDataset(
        train=pack("train"), val=pack("val"), test=pack("test"),
        feature_mean=f_mean, feature_std=f_std,
        regime_mean=r_mean, regime_std=r_std,
        n_seconds=n_seconds, n_chunks=len(ordered),
        chunk_dirs=ordered_dirs, chunk_hashes=ordered_hashes,
        chunk_symbols=ordered_symbols, chunk_stats=ordered_stats,
        session_split={k: list(v) for k, v in session_split.items()},
    )


def write_manifest(ds: BookDataset, path: str) -> None:
    """Write the split manifest: chunk identities, hashes, and split assignment.

    This is the provenance record. Test session identities are fixed here
    before any sweep; reruns with the same manifest reproduce the split.
    """
    import json
    from datetime import datetime, timezone
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_chunks": ds.n_chunks,
        "n_seconds": ds.n_seconds,
        "chunks": [
            {"session_id": i, "dir": d, "symbol": sym, "sha16": h,
             "split": next(s for s, idxs in ds.session_split.items() if i in idxs),
             "seconds_span": st["span"], "seconds_usable": st["usable"],
             "seconds_missing": st["missing"]}
            for i, (d, h, sym, st) in enumerate(zip(
                ds.chunk_dirs, ds.chunk_hashes, ds.chunk_symbols, ds.chunk_stats))
        ],
        "session_split": ds.session_split,
        "windows": {s: int(getattr(ds, s).x.shape[0])
                    for s in ("train", "val", "test")},
    }
    # Integrity check: no session appears in two splits.
    seen_sessions = {}
    for s in ("train", "val", "test"):
        for sid in np.unique(getattr(ds, s).session_ids).tolist():
            assert sid not in seen_sessions, f"session {sid} in two splits"
            seen_sessions[sid] = s
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def dataset_card(ds: BookDataset) -> str:
    """Markdown dataset card: sessions, hours, symbols, missingness."""
    lines = ["# Dataset card: book-state windows", ""]
    lines.append(f"- Sessions (chunks): {ds.n_chunks}")
    lines.append(f"- Total seconds of book data: {ds.n_seconds} "
                 f"({ds.n_seconds / 3600:.2f} hours)")
    symbols = sorted(set(ds.chunk_symbols)) if ds.chunk_symbols else ["unknown"]
    lines.append(f"- Symbols: {', '.join(symbols)}")
    if ds.chunk_stats:
        total_span = sum(s["span"] for s in ds.chunk_stats)
        total_missing = sum(s["missing"] for s in ds.chunk_stats)
        lines.append(f"- Missingness: {total_missing} of {total_span} "
                     f"session-seconds have no usable bar (recording gaps; "
                     f"the bar after a gap is dropped)")
    lines.append("")
    lines.append("## Split manifest (session-disjoint)")
    lines.append("")
    lines.append("| Split | Sessions | Windows |")
    lines.append("|-------|----------|---------|")
    for s in ("train", "val", "test"):
        sw = getattr(ds, s)
        n_sess = len(np.unique(sw.session_ids))
        lines.append(f"| {s} | {n_sess} | {sw.x.shape[0]} |")
    lines.append("")
    lines.append("## Sessions")
    lines.append("")
    lines.append("| ID | Directory | Symbol | SHA16 | Split | Span (s) | "
                 "Usable (s) | Missing (s) |")
    lines.append("|----|-----------|--------|-------|-------|----------|"
                 "------------|-------------|")
    for i, (d, sym, h, st) in enumerate(zip(
            ds.chunk_dirs, ds.chunk_symbols, ds.chunk_hashes, ds.chunk_stats)):
        split = next(s for s, idxs in ds.session_split.items() if i in idxs)
        lines.append(f"| {i} | {d} | {sym} | {h} | {split} | {st['span']} | "
                     f"{st['usable']} | {st['missing']} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Windows never cross chunk boundaries; every 60-row window "
                 "spans exactly 59 seconds.")
    lines.append("- No raw second appears in more than one split by construction "
                 "(whole sessions assigned).")
    lines.append("- Features standardized with train-split statistics only.")
    return "\n".join(lines)


def save_dataset(ds: BookDataset, path: str) -> None:
    import json
    np.savez_compressed(
        path,
        **{f"{s}_{k}": getattr(getattr(ds, s), k)
           for s in ("train", "val", "test")
           for k in ("x", "regime", "t_start", "t_end", "session_ids")},
        feature_mean=ds.feature_mean, feature_std=ds.feature_std,
        regime_mean=ds.regime_mean, regime_std=ds.regime_std,
        n_seconds=np.array(ds.n_seconds), n_chunks=np.array(ds.n_chunks),
        chunk_dirs=np.array(ds.chunk_dirs, dtype=object),
        chunk_hashes=np.array(ds.chunk_hashes, dtype=object),
        chunk_symbols=np.array(ds.chunk_symbols, dtype=object),
        chunk_stats_json=np.array(json.dumps(ds.chunk_stats)),
        session_split_json=np.array(json.dumps(ds.session_split)),
    )


def load_dataset(path: str) -> BookDataset:
    import json
    z = np.load(path, allow_pickle=True)

    def unpack(s: str) -> SplitWindows:
        # session_ids may be absent in datasets built before FL3.
        if f"{s}_session_ids" in z:
            ss = z[f"{s}_session_ids"]
        else:
            ss = np.full(z[f"{s}_x"].shape[0], -1, dtype=np.int64)
        return SplitWindows(x=z[f"{s}_x"], regime=z[f"{s}_regime"],
                            t_start=z[f"{s}_t_start"], t_end=z[f"{s}_t_end"],
                            session_ids=ss)

    def _str_list(key):
        return [str(x) for x in z[key].tolist()] if key in z else []

    split = json.loads(str(z["session_split_json"])) if "session_split_json" in z else {}
    stats = json.loads(str(z["chunk_stats_json"])) if "chunk_stats_json" in z else []
    return BookDataset(
        train=unpack("train"), val=unpack("val"), test=unpack("test"),
        feature_mean=z["feature_mean"], feature_std=z["feature_std"],
        regime_mean=z["regime_mean"], regime_std=z["regime_std"],
        n_seconds=int(z["n_seconds"]), n_chunks=int(z["n_chunks"]),
        chunk_dirs=_str_list("chunk_dirs"),
        chunk_hashes=_str_list("chunk_hashes"),
        chunk_symbols=_str_list("chunk_symbols"),
        chunk_stats=stats,
        session_split=split,
    )
