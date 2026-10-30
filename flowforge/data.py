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

import hashlib
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

# Provenance versions (C02). Manifest schema v1 was the pre-C02 manifest:
# truncated frame-buffer hashes, no canonical raw hashes, no validation.
# Schema v2 records full SHA-256 of explicitly ordered raw file contents
# plus the full record set below. DATA_TRANSFORM_VERSION is the first
# versioned transform (introduced by C02); bump it if the bar/window
# construction changes.
MANIFEST_SCHEMA_VERSION = 2
DATA_TRANSFORM_VERSION = 1


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
    # C02 provenance: per-chunk canonical raw-file hashes, raw time
    # ranges, and recorder metadata, aligned with chunk_dirs.
    chunk_file_hashes: list = field(default_factory=list, repr=False)
    chunk_time_ranges: list = field(default_factory=list, repr=False)
    chunk_recorder_meta: list = field(default_factory=list, repr=False)
    code_sha: str | None = None
    manifest: dict | None = field(default=None, repr=False)
    manifest_sha256: str | None = None
    # True only for datasets built before the C02 manifest: they load
    # only under explicit legacy=True and cannot publish new evidence.
    legacy: bool = False


def _load_chunk(chunk_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_parquet(f"{chunk_dir}/quotes.parquet")
    t = pd.read_parquet(f"{chunk_dir}/trades.parquet")
    return q, t


def _chunk_meta(chunk_dir: str) -> dict | None:
    """Recording metadata from the chunk's meta.json, if present."""
    try:
        with open(os.path.join(chunk_dir, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _chunk_symbol(chunk_dir: str) -> str:
    """Recording symbol from the chunk's meta.json, if present."""
    meta = _chunk_meta(chunk_dir)
    if meta is None:
        return "unknown"
    return str(meta.get("symbol", "unknown"))


def _sha256_file(path: str) -> str:
    """Full SHA-256 of a file's raw bytes, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_json(obj) -> str:
    """Canonical JSON for hashing: sorted keys, tight separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _day_from_ms(ms: int | None) -> str | None:
    """UTC calendar day (YYYY-MM-DD) of a millisecond timestamp."""
    if ms is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


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
                  session_split: dict[str, list[int]] | None = None,
                  code_sha: str | None = None) -> BookDataset:
    """Load chunks, build 1s bars, split by session, emit standardized windows.

    Session-disjoint splits (FL3): whole recording sessions (chunks) are
    assigned to splits, never split across them. By default, chunks are
    assigned chronologically: first 70% train, next 15% val, remainder test
    (by chunk count, rounding). Pass an explicit `session_split` manifest
    like {"train": [0,1,2,3,4], "val": [5], "test": [6,7]} to lock test
    session identities before sweeps. Chunk indices refer to chronological
    order. No raw second ever appears in two splits by construction.

    Provenance (C02): chunk hashes are full SHA-256 of explicitly ordered
    raw file bytes (quotes.parquet bytes, then trades.parquet bytes),
    never parsed-frame bytes. Duplicate or overlapping raw sessions are
    detected before any chunk is dropped, so a session whose bars are
    later dropped cannot hide a duplicate. The returned dataset carries
    its split manifest and manifest SHA; see build_manifest.
    """
    # Raw-record checks run BEFORE any chunk is dropped, so a duplicated
    # or overlapping session is detected even if its bars are later
    # dropped. Hashes are full SHA-256 of raw file bytes in fixed order
    # (quotes then trades), never parsed-frame bytes.
    if len(chunk_dirs) != len(set(chunk_dirs)):
        raise ValueError("duplicate chunk directory in chunk_dirs")

    def _minmax(frame):
        if len(frame) == 0:
            return None, None
        return int(frame["time"].min()), int(frame["time"].max())

    raw_info = []
    for d in chunk_dirs:
        quotes_sha = _sha256_file(f"{d}/quotes.parquet")
        trades_sha = _sha256_file(f"{d}/trades.parquet")
        combined = hashlib.sha256()
        for fname in ("quotes.parquet", "trades.parquet"):
            with open(f"{d}/{fname}", "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    combined.update(block)
        q, t = _load_chunk(d)
        qmin, qmax = _minmax(q)
        tmin, tmax = _minmax(t)
        raw_info.append({
            "dir": d, "q": q, "t": t,
            "quotes_sha256": quotes_sha, "trades_sha256": trades_sha,
            "sha256": combined.hexdigest(),
            "quotes_t_min_ms": qmin, "quotes_t_max_ms": qmax,
            "trades_t_min_ms": tmin, "trades_t_max_ms": tmax,
            "day": _day_from_ms(qmin),
            "meta": _chunk_meta(d), "symbol": _chunk_symbol(d),
        })
    seen_hash = {}
    for info in raw_info:
        if info["sha256"] in seen_hash:
            raise ValueError(
                f"duplicated raw session: {info['dir']} has identical raw "
                f"bytes to {seen_hash[info['sha256']]}; deduplicate "
                f"recording sessions")
        seen_hash[info["sha256"]] = info["dir"]
    for i, a in enumerate(raw_info):
        if a["quotes_t_min_ms"] is None:
            continue
        for b in raw_info[i + 1:]:
            if b["quotes_t_min_ms"] is None:
                continue
            if (a["quotes_t_min_ms"] <= b["quotes_t_max_ms"]
                    and b["quotes_t_min_ms"] <= a["quotes_t_max_ms"]):
                raise ValueError(
                    f"overlapping raw sessions: {a['dir']} and {b['dir']} "
                    f"share raw quote seconds; sort/deduplicate recording "
                    f"sessions")

    # Parallel kept-lists: a chunk with no usable bars is skipped, and the
    # surviving entries stay aligned (dir <-> bars <-> hash <-> symbol).
    kept_dirs: list[str] = []
    kept_bars = []
    kept_hashes: list[str] = []
    kept_symbols: list[str] = []
    kept_stats: list[dict] = []
    kept_file_hashes: list[dict] = []
    kept_time_ranges: list[dict] = []
    kept_recorder_meta: list = []
    for info in raw_info:
        d = info["dir"]
        b = _bars_from_chunk(info["q"], info["t"])
        if len(b) == 0:
            continue
        span = int(b.index.max() - b.index.min()) + 1
        kept_dirs.append(d)
        kept_bars.append(b)
        kept_hashes.append(info["sha256"])
        kept_symbols.append(info["symbol"])
        kept_stats.append({"span": span, "usable": int(len(b)),
                           "missing": span - int(len(b)),
                           "segments": len(_segments(b))})
        kept_file_hashes.append({
            "quotes_sha256": info["quotes_sha256"],
            "trades_sha256": info["trades_sha256"],
            "sha256": info["sha256"]})
        kept_time_ranges.append({
            "quotes_t_min_ms": info["quotes_t_min_ms"],
            "quotes_t_max_ms": info["quotes_t_max_ms"],
            "trades_t_min_ms": info["trades_t_min_ms"],
            "trades_t_max_ms": info["trades_t_max_ms"],
            "day": info["day"]})
        kept_recorder_meta.append(info["meta"])
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
    ordered_file_hashes = [kept_file_hashes[i] for i in order]
    ordered_time_ranges = [kept_time_ranges[i] for i in order]
    ordered_recorder_meta = [kept_recorder_meta[i] for i in order]
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

    ds = BookDataset(
        train=pack("train"), val=pack("val"), test=pack("test"),
        feature_mean=f_mean, feature_std=f_std,
        regime_mean=r_mean, regime_std=r_std,
        n_seconds=n_seconds, n_chunks=len(ordered),
        chunk_dirs=ordered_dirs, chunk_hashes=ordered_hashes,
        chunk_symbols=ordered_symbols, chunk_stats=ordered_stats,
        chunk_file_hashes=ordered_file_hashes,
        chunk_time_ranges=ordered_time_ranges,
        chunk_recorder_meta=ordered_recorder_meta,
        code_sha=code_sha,
        session_split={k: list(v) for k, v in session_split.items()},
    )
    ds.manifest = build_manifest(ds)
    ds.manifest_sha256 = ds.manifest["manifest_sha256"]
    return ds


def manifest_sha256(manifest: dict) -> str:
    """Canonical SHA-256 of a manifest's content.

    The hash covers everything except the hash itself and the wall-clock
    `created_utc`, so the same dataset built in separate processes (or
    rewritten later) yields the identical hash.
    """
    content = {k: v for k, v in manifest.items()
               if k not in ("manifest_sha256", "created_utc")}
    return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def build_manifest(ds: BookDataset) -> dict:
    """Build the split manifest: chunk identities, canonical raw hashes,
    and the full provenance record.

    The manifest is sanitized by construction: chunk directories are
    recorded as basenames only, never absolute local paths, so the file
    is safe to commit. Test session identities are fixed here before any
    sweep; reruns with the same manifest reproduce the split.
    """
    from datetime import datetime, timezone

    def _at(lst, i, key, default=None):
        if i < len(lst) and isinstance(lst[i], dict):
            return lst[i].get(key, default)
        return default

    chunks = []
    for i, (d, h, sym, st) in enumerate(zip(
            ds.chunk_dirs, ds.chunk_hashes, ds.chunk_symbols, ds.chunk_stats)):
        meta = ds.chunk_recorder_meta[i] if i < len(ds.chunk_recorder_meta) else None
        tr = ds.chunk_time_ranges[i] if i < len(ds.chunk_time_ranges) else {}
        chunks.append({
            "session_id": i,
            "dir": os.path.basename(os.path.normpath(d)),
            "symbol": sym,
            "sha256": h,
            "quotes_sha256": _at(ds.chunk_file_hashes, i, "quotes_sha256"),
            "trades_sha256": _at(ds.chunk_file_hashes, i, "trades_sha256"),
            "split": next(s for s, idxs in ds.session_split.items() if i in idxs),
            "seconds_span": st["span"], "seconds_usable": st["usable"],
            "seconds_missing": st["missing"],
            "segments": st.get("segments"),
            "quotes_t_min_ms": tr.get("quotes_t_min_ms"),
            "quotes_t_max_ms": tr.get("quotes_t_max_ms"),
            "trades_t_min_ms": tr.get("trades_t_min_ms"),
            "trades_t_max_ms": tr.get("trades_t_max_ms"),
            "day": tr.get("day"),
            "recorder_meta": meta,
            "recorder_schema_version": (meta.get("schema_version")
                                        if isinstance(meta, dict) else None),
        })
    content = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "data_transform_version": DATA_TRANSFORM_VERSION,
        # The dataset build is deterministic (no sampling): no random
        # seed exists to record, so the field is present and null rather
        # than omitted or invented.
        "random_seed": None,
        "deterministic": True,
        "code_sha": ds.code_sha,
        "n_chunks": ds.n_chunks,
        "n_seconds": ds.n_seconds,
        "chunks": chunks,
        "session_split": {k: list(v) for k, v in ds.session_split.items()},
        "windows": {s: int(getattr(ds, s).x.shape[0])
                    for s in ("train", "val", "test")},
    }
    sha = hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()
    return {"manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "manifest_sha256": sha,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **content}


def write_manifest(ds: BookDataset, path: str) -> dict:
    """Write the split manifest: chunk identities, hashes, and split assignment.

    This is the provenance record. Test session identities are fixed here
    before any sweep; reruns with the same manifest reproduce the split.
    Writes the dataset's own manifest when it has one (so the file is
    byte-identical to the manifest embedded in the NPZ); otherwise builds
    one from the dataset fields.
    """
    manifest = ds.manifest if ds.manifest is not None else build_manifest(ds)
    # Integrity check: no session appears in two splits.
    seen_sessions = {}
    for s in ("train", "val", "test"):
        for sid in np.unique(getattr(ds, s).session_ids).tolist():
            assert sid not in seen_sessions, f"session {sid} in two splits"
            seen_sessions[sid] = s
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


_MANIFEST_REQUIRED_TOP = ("manifest_schema_version", "manifest_sha256",
                          "data_transform_version", "random_seed", "code_sha",
                          "n_chunks", "n_seconds", "chunks", "session_split",
                          "windows")
_MANIFEST_REQUIRED_CHUNK = ("session_id", "dir", "symbol", "sha256",
                            "quotes_sha256", "trades_sha256", "split",
                            "seconds_span", "seconds_usable",
                            "seconds_missing", "segments",
                            "quotes_t_min_ms", "quotes_t_max_ms",
                            "trades_t_min_ms", "trades_t_max_ms", "day",
                            "recorder_meta", "recorder_schema_version")


def validate_manifest(manifest: dict, chunk_dirs: list[str]) -> dict:
    """Validate a split manifest against raw recording directories.

    Recomputes every chunk's raw file hashes and raw timestamp
    minima/maxima from the parquet files and compares them to the
    manifest, and checks the manifest's own canonical hash, required
    fields, and session coverage. Raises ValueError naming the chunk
    and field on any missing or mismatching value. Returns the manifest
    unchanged on success. This is the check to run before trusting a
    manifest: it catches a manifest edited by hand, a manifest paired
    with different recordings, and a manifest missing required fields.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    for field in _MANIFEST_REQUIRED_TOP:
        if field not in manifest:
            raise ValueError(f"manifest missing required field {field!r}")
    if manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema version {manifest['manifest_schema_version']} "
            f"is not current schema {MANIFEST_SCHEMA_VERSION}: manifests "
            f"from before C02 (truncated frame-buffer hashes) cannot be "
            f"validated; rebuild with examples/build_data.py")
    if manifest_sha256(manifest) != manifest["manifest_sha256"]:
        raise ValueError("manifest SHA mismatch: the manifest content does "
                         "not hash to its recorded manifest_sha256; the "
                         "manifest was edited or corrupted")
    chunks = manifest["chunks"]
    if len(chunks) != manifest["n_chunks"]:
        raise ValueError(
            f"manifest lists {len(chunks)} chunks but n_chunks is "
            f"{manifest['n_chunks']}")
    split = manifest["session_split"]
    if set(split.keys()) != {"train", "val", "test"}:
        raise ValueError("session_split must have exactly train/val/test")
    seen = []
    for name in ("train", "val", "test"):
        seen.extend(split[name])
    if sorted(seen) != list(range(manifest["n_chunks"])):
        raise ValueError("session_split must assign each chunk exactly once")
    by_base: dict[str, list[str]] = {}
    for d in chunk_dirs:
        by_base.setdefault(os.path.basename(os.path.normpath(d)), []).append(d)
    sids = set()
    for c in chunks:
        for field in _MANIFEST_REQUIRED_CHUNK:
            if field not in c:
                raise ValueError(
                    f"chunk {c.get('session_id')}: manifest missing required "
                    f"field {field!r}")
        sid = c["session_id"]
        if sid in sids:
            raise ValueError(f"duplicate session_id {sid} in manifest")
        sids.add(sid)
        base = c["dir"]
        cands = by_base.get(base, [])
        if not cands:
            raise ValueError(
                f"chunk {sid} ({base}): recording directory not found "
                f"among the supplied chunk directories")
        if len(cands) > 1:
            raise ValueError(
                f"chunk {sid} ({base}): ambiguous recording directory "
                f"(multiple supplied dirs share the basename)")
        d = cands[0]
        for fname, key in (("quotes.parquet", "quotes_sha256"),
                           ("trades.parquet", "trades_sha256")):
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                raise ValueError(
                    f"chunk {sid} ({base}): missing file {fname}")
            h = _sha256_file(p)
            if h != c[key]:
                raise ValueError(
                    f"chunk {sid} ({base}): {key} mismatch: manifest "
                    f"records {c[key]}, recomputed {h}")
        q, t = _load_chunk(d)

        def _minmax(frame):
            if len(frame) == 0:
                return None, None
            return int(frame["time"].min()), int(frame["time"].max())

        qmin, qmax = _minmax(q)
        tmin, tmax = _minmax(t)
        for label, got, want in (
                ("quotes_t_min_ms", qmin, c["quotes_t_min_ms"]),
                ("quotes_t_max_ms", qmax, c["quotes_t_max_ms"]),
                ("trades_t_min_ms", tmin, c["trades_t_min_ms"]),
                ("trades_t_max_ms", tmax, c["trades_t_max_ms"])):
            if got != want:
                raise ValueError(
                    f"chunk {sid} ({base}): {label} mismatch: manifest "
                    f"records {want}, recomputed {got}")
    if sids != set(range(manifest["n_chunks"])):
        raise ValueError("manifest chunk session_ids do not cover "
                         "0..n_chunks-1 exactly once")
    return manifest


def check_manifest_file(ds: BookDataset, path: str) -> bool:
    """Cross-check a dataset against a split-manifest file on disk.

    Returns False when the file is absent (the caller may proceed on the
    NPZ-embedded manifest alone). Raises ValueError when the file exists
    but its recorded SHA or its content hash does not match the
    dataset's manifest SHA: a failed match must abort before training,
    never warn-and-continue.
    """
    if not os.path.exists(path):
        return False
    with open(path) as f:
        manifest = json.load(f)
    if ds.manifest_sha256 is None:
        raise ValueError("dataset carries no split-manifest SHA; it was "
                         "built before C02 or loaded in legacy mode")
    if manifest.get("manifest_sha256") != ds.manifest_sha256:
        raise ValueError(
            f"split-manifest SHA mismatch: manifest file records "
            f"{manifest.get('manifest_sha256')}, dataset records "
            f"{ds.manifest_sha256}; the NPZ and the manifest do not belong "
            f"together")
    if manifest_sha256(manifest) != ds.manifest_sha256:
        raise ValueError("split-manifest content hash mismatch: the "
                         "manifest file content does not hash to the "
                         "dataset's manifest SHA")
    return True


def ensure_publishable(ds: BookDataset) -> None:
    """Refuse benchmark evidence from a legacy dataset.

    A dataset loaded in legacy mode (no C02 split manifest) may be
    inspected, but it cannot publish new benchmark evidence: metrics,
    sweep, or bundle artifacts built from it would claim provenance the
    dataset cannot back. Raises RuntimeError for legacy datasets.
    """
    if ds.legacy:
        raise RuntimeError(
            "legacy dataset (no C02 split manifest): cannot publish new "
            "benchmark evidence; rebuild the dataset with "
            "examples/build_data.py first")


def dataset_card(ds: BookDataset) -> str:
    """Markdown dataset card: sessions, hours, symbols, missingness."""
    lines = ["# Dataset card: book-state windows", ""]
    lines.append(f"- Sessions (chunks): {ds.n_chunks}")
    lines.append(f"- Total seconds of book data: {ds.n_seconds} "
                 f"({ds.n_seconds / 3600:.2f} hours)")
    symbols = sorted(set(ds.chunk_symbols)) if ds.chunk_symbols else ["unknown"]
    lines.append(f"- Symbols: {', '.join(symbols)}")
    lines.append(f"- Manifest schema: {MANIFEST_SCHEMA_VERSION}")
    lines.append(f"- Manifest SHA-256: "
                 f"{ds.manifest_sha256 or 'none (built before C02)'}")
    lines.append(f"- Data transform version: {DATA_TRANSFORM_VERSION}")
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
    lines.append("| ID | Directory | Symbol | SHA256 | Split | Span (s) | "
                 "Usable (s) | Missing (s) |")
    lines.append("|----|-----------|--------|-------|-------|----------|"
                 "------------|-------------|")
    for i, (d, sym, h, st) in enumerate(zip(
            ds.chunk_dirs, ds.chunk_symbols, ds.chunk_hashes, ds.chunk_stats)):
        split = next(s for s, idxs in ds.session_split.items() if i in idxs)
        lines.append(f"| {i} | {os.path.basename(os.path.normpath(d))} | "
                     f"{sym} | {h} | {split} | {st['span']} | "
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
    # The split manifest travels inside the NPZ so the artifact is
    # self-describing: the manifest SHA recorded here is what metrics,
    # sweep, and bundle artifacts cite. A dataset without a manifest
    # (built by hand, not by build_dataset) saves without provenance and
    # loads only in explicit legacy mode.
    #
    # FF-N03: chunk metadata is stored as typed UTF-8 JSON, never as
    # NumPy object arrays, so the artifact contains nothing that needs
    # pickle to read back and the default loader can stay pickle-free.
    extra = {}
    if ds.manifest is not None:
        extra["manifest_json"] = np.array(json.dumps(ds.manifest))
    if ds.manifest_sha256 is not None:
        extra["manifest_sha256"] = np.array(ds.manifest_sha256)
    extra["chunk_file_hashes_json"] = np.array(json.dumps(ds.chunk_file_hashes))
    extra["chunk_time_ranges_json"] = np.array(json.dumps(ds.chunk_time_ranges))
    extra["chunk_recorder_meta_json"] = np.array(json.dumps(ds.chunk_recorder_meta))
    np.savez_compressed(
        path,
        **{f"{s}_{k}": getattr(getattr(ds, s), k)
           for s in ("train", "val", "test")
           for k in ("x", "regime", "t_start", "t_end", "session_ids")},
        feature_mean=ds.feature_mean, feature_std=ds.feature_std,
        regime_mean=ds.regime_mean, regime_std=ds.regime_std,
        n_seconds=np.array(ds.n_seconds), n_chunks=np.array(ds.n_chunks),
        chunk_dirs_json=np.array(json.dumps(ds.chunk_dirs)),
        chunk_hashes_json=np.array(json.dumps(ds.chunk_hashes)),
        chunk_symbols_json=np.array(json.dumps(ds.chunk_symbols)),
        chunk_stats_json=np.array(json.dumps(ds.chunk_stats)),
        session_split_json=np.array(json.dumps(ds.session_split)),
        **extra,
    )


_TYPED_META_KEYS = ("chunk_dirs_json", "chunk_hashes_json",
                    "chunk_symbols_json")
_OBJECT_META_KEYS = ("chunk_dirs", "chunk_hashes", "chunk_symbols")


def load_dataset(path: str, legacy: bool = False) -> BookDataset:
    """Load a dataset NPZ.

    The default loader is pickle-free (FF-N03): it opens the archive
    with allow_pickle=False and validates dtypes, finiteness, and split
    identity against the dataset's own embedded manifest (window
    counts, per-split session membership, session_split agreement).
    Archives in the pre-FF-N03 layout (chunk metadata stored as object
    arrays) and datasets built before the C02 split manifest load only
    with explicit legacy=True, which routes to the trusted reader; the
    loaded dataset is flagged legacy and cannot publish new benchmark
    evidence (see ensure_publishable). Session IDs are never silently
    filled with -1: a manifest-carrying file with session_ids stripped
    is corrupt and fails even in legacy mode, because the manifest
    claims session coverage the arrays cannot back. Use
    convert_legacy_dataset to migrate a trusted legacy archive to the
    typed format.
    """
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
    has_manifest = "manifest_json" in keys and "manifest_sha256" in keys
    if has_manifest:
        missing = [s for s in ("train", "val", "test")
                   if f"{s}_session_ids" not in keys]
        if missing:
            raise ValueError(
                f"dataset manifest present but session_ids missing for "
                f"{missing}: corrupt artifact, refusing to invent -1 "
                f"session identities")
    object_meta = [k for k in _OBJECT_META_KEYS if k in keys]
    if object_meta and not legacy:
        raise ValueError(
            f"dataset NPZ stores {object_meta} as pickled object arrays "
            f"(pre-FF-N03 legacy format): the default loader never "
            f"evaluates pickle; convert the archive with "
            f"flowforge.data.convert_legacy_dataset, or load with "
            f"legacy=True for inspection only (legacy datasets cannot "
            f"publish new benchmark evidence)")
    if not has_manifest and not legacy:
        raise ValueError(
            "dataset was built before the C02 split manifest (no "
            "manifest SHA in the NPZ): rerun examples/build_data.py to "
            "rebuild it with provenance, or load with legacy=True for "
            "inspection only (legacy datasets cannot publish new "
            "benchmark evidence)")
    typed_meta = all(k in keys for k in _TYPED_META_KEYS)
    if has_manifest and typed_meta and not object_meta:
        return _load_dataset_validated(path)
    # Explicit legacy mode: the trusted reader is the only path that
    # evaluates pickle, and reaching it requires legacy=True.
    return _load_dataset_trusted(path)


def _load_dataset_validated(path: str) -> BookDataset:
    """Default-loader path: pickle-free, fully validated.

    Split identity is self-consistency against the dataset's own
    embedded manifest (FF-N03 ruling): the manifest is the tamper
    reference, so window counts, per-split session membership, and the
    session_split record must all agree with the arrays.
    """
    import json
    z = np.load(path, allow_pickle=False)
    manifest = json.loads(str(z["manifest_json"]))
    msha = str(z["manifest_sha256"])
    if (manifest_sha256(manifest) != msha
            or manifest.get("manifest_sha256") != msha):
        raise ValueError(
            "embedded split-manifest SHA mismatch: the NPZ manifest "
            "content does not hash to its recorded manifest SHA; "
            "the artifact was edited or corrupted")
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"embedded split manifest schema version "
            f"{manifest.get('manifest_schema_version')} is not current "
            f"schema {MANIFEST_SCHEMA_VERSION}")
    split = (json.loads(str(z["session_split_json"]))
             if "session_split_json" in z else {})
    if split != manifest.get("session_split"):
        raise ValueError(
            "session_split_json disagrees with the embedded split "
            "manifest session_split: the archive metadata was edited "
            "or the records do not belong together")
    n_chunks = int(z["n_chunks"])
    chunks = manifest.get("chunks") or []
    if manifest.get("n_chunks") != n_chunks or len(chunks) != n_chunks:
        raise ValueError(
            f"manifest n_chunks {manifest.get('n_chunks')} / chunk list "
            f"{len(chunks)} does not match the NPZ n_chunks {n_chunks}")
    meta: dict[str, list] = {}
    for typed, field in (("chunk_dirs_json", "chunk_dirs"),
                         ("chunk_hashes_json", "chunk_hashes"),
                         ("chunk_symbols_json", "chunk_symbols")):
        vals = json.loads(str(z[typed]))
        if (not isinstance(vals, list) or len(vals) != n_chunks
                or not all(isinstance(v, str) for v in vals)):
            raise ValueError(
                f"{typed} must be a list of {n_chunks} strings")
        meta[field] = vals
    windows = manifest.get("windows") or {}
    manifest_split = manifest.get("session_split") or {}

    def unpack(s: str) -> SplitWindows:
        x, regime = z[f"{s}_x"], z[f"{s}_regime"]
        t_start, t_end = z[f"{s}_t_start"], z[f"{s}_t_end"]
        session_ids = z[f"{s}_session_ids"]
        for name, arr, dt in (("x", x, np.float32),
                              ("regime", regime, np.float32),
                              ("t_start", t_start, np.float64),
                              ("t_end", t_end, np.float64),
                              ("session_ids", session_ids, np.int64)):
            if arr.dtype != dt:
                raise ValueError(
                    f"{s}_{name} dtype {arr.dtype} != expected "
                    f"{np.dtype(dt).name}: archive does not match the "
                    f"typed dataset format")
        if x.ndim != 3 or x.shape[1:] != (WINDOW, N_FEATURES):
            raise ValueError(
                f"{s}_x shape {x.shape} != (n, {WINDOW}, {N_FEATURES})")
        n = int(x.shape[0])
        if (regime.shape != (n, 2) or t_start.shape != (n,)
                or t_end.shape != (n,) or session_ids.shape != (n,)):
            raise ValueError(f"{s} split arrays disagree on window count")
        if windows.get(s) != n:
            raise ValueError(
                f"manifest windows[{s!r}] records {windows.get(s)} "
                f"windows but {s}_x holds {n}: split identity mismatch "
                f"against the embedded manifest")
        allowed = set(manifest_split.get(s, []))
        bad = sorted(set(session_ids.tolist()) - allowed)
        if bad:
            raise ValueError(
                f"{s} session_ids {bad} are not assigned to split "
                f"{s!r} by the embedded manifest session_split: split "
                f"identity mismatch")
        for name, arr in (("x", x), ("regime", regime),
                          ("t_start", t_start), ("t_end", t_end)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{s}_{name} contains non-finite values")
        return SplitWindows(x=x, regime=regime, t_start=t_start,
                            t_end=t_end, session_ids=session_ids)

    def _stats(key: str, width: int) -> np.ndarray:
        arr = z[key]
        if arr.dtype != np.float64 or arr.shape != (width,):
            raise ValueError(
                f"{key} dtype/shape {arr.dtype}/{arr.shape} != float64/"
                f"({width},)")
        if not np.isfinite(arr).all():
            raise ValueError(f"{key} contains non-finite values")
        return arr

    def _json_list(key):
        return json.loads(str(z[key])) if key in z else []

    return BookDataset(
        train=unpack("train"), val=unpack("val"), test=unpack("test"),
        feature_mean=_stats("feature_mean", N_FEATURES),
        feature_std=_stats("feature_std", N_FEATURES),
        regime_mean=_stats("regime_mean", 2),
        regime_std=_stats("regime_std", 2),
        n_seconds=int(z["n_seconds"]), n_chunks=n_chunks,
        chunk_dirs=meta["chunk_dirs"],
        chunk_hashes=meta["chunk_hashes"],
        chunk_symbols=meta["chunk_symbols"],
        chunk_stats=_json_list("chunk_stats_json"),
        chunk_file_hashes=_json_list("chunk_file_hashes_json"),
        chunk_time_ranges=_json_list("chunk_time_ranges_json"),
        chunk_recorder_meta=_json_list("chunk_recorder_meta_json"),
        code_sha=manifest.get("code_sha"),
        manifest=manifest,
        manifest_sha256=msha,
        session_split=split,
        legacy=False,
    )


def _load_dataset_trusted(path: str) -> BookDataset:
    """Trusted reader for legacy archives (explicit legacy=True only).

    This is the only loader path that opens an NPZ with
    allow_pickle=True: the pre-FF-N03 layout stores chunk metadata as
    object arrays, and the pre-C02 layout has no manifest at all.
    Reaching this function requires the caller's explicit opt-in.
    """
    import json
    z = np.load(path, allow_pickle=True)
    has_manifest = "manifest_json" in z and "manifest_sha256" in z
    manifest = None
    msha = None
    if has_manifest:
        manifest = json.loads(str(z["manifest_json"]))
        msha = str(z["manifest_sha256"])
        if (manifest_sha256(manifest) != msha
                or manifest.get("manifest_sha256") != msha):
            raise ValueError(
                "embedded split-manifest SHA mismatch: the NPZ manifest "
                "content does not hash to its recorded manifest SHA; "
                "the artifact was edited or corrupted")

    def unpack(s: str) -> SplitWindows:
        # session_ids may be absent in datasets built before FL3, but
        # only in legacy mode: a manifest-carrying file without them is
        # corrupt, and -1 identities are never constructed silently.
        if f"{s}_session_ids" in z:
            ss = z[f"{s}_session_ids"]
        elif has_manifest:
            raise ValueError(
                f"dataset manifest present but {s} session_ids missing: "
                f"corrupt artifact, refusing to invent -1 session identities")
        else:
            ss = np.full(z[f"{s}_x"].shape[0], -1, dtype=np.int64)
        return SplitWindows(x=z[f"{s}_x"], regime=z[f"{s}_regime"],
                            t_start=z[f"{s}_t_start"], t_end=z[f"{s}_t_end"],
                            session_ids=ss)

    def _meta_list(typed_key: str, object_key: str) -> list:
        if typed_key in z:
            return [str(v) for v in json.loads(str(z[typed_key]))]
        if object_key in z:
            return [str(v) for v in z[object_key].tolist()]
        return []

    def _json_list(key):
        return json.loads(str(z[key])) if key in z else []

    split = json.loads(str(z["session_split_json"])) if "session_split_json" in z else {}
    stats = json.loads(str(z["chunk_stats_json"])) if "chunk_stats_json" in z else []
    return BookDataset(
        train=unpack("train"), val=unpack("val"), test=unpack("test"),
        feature_mean=z["feature_mean"], feature_std=z["feature_std"],
        regime_mean=z["regime_mean"], regime_std=z["regime_std"],
        n_seconds=int(z["n_seconds"]), n_chunks=int(z["n_chunks"]),
        chunk_dirs=_meta_list("chunk_dirs_json", "chunk_dirs"),
        chunk_hashes=_meta_list("chunk_hashes_json", "chunk_hashes"),
        chunk_symbols=_meta_list("chunk_symbols_json", "chunk_symbols"),
        chunk_stats=stats,
        chunk_file_hashes=_json_list("chunk_file_hashes_json"),
        chunk_time_ranges=_json_list("chunk_time_ranges_json"),
        chunk_recorder_meta=_json_list("chunk_recorder_meta_json"),
        code_sha=manifest.get("code_sha") if manifest else None,
        manifest=manifest,
        manifest_sha256=msha,
        session_split=split,
        legacy=not has_manifest,
    )


def convert_legacy_dataset(src_path: str, dst_path: str) -> BookDataset:
    """Convert a trusted legacy dataset NPZ to the safe typed format.

    Opt-in and trusted: reading the source requires allow_pickle=True
    (the legacy layout stores chunk metadata as object arrays), so call
    this only on archives you trust. The source is opened read-only and
    never rewritten; dst_path must name a different file, so the legacy
    archive is preserved as historical.

    A source that predates the C02 manifest gains one built from its
    own recorded metadata (build_manifest), so the converted artifact
    is self-describing and loads through the default safe loader. The
    conversion fabricates no provenance the source did not record:
    per-file raw hashes stay null where the source lacks them, and the
    manifest records the source's own split assignment unchanged.
    Returns the converted dataset as reloaded through the default
    loader, so a successful return proves the roundtrip.
    """
    src = os.path.abspath(src_path)
    dst = os.path.abspath(dst_path)
    if src == dst:
        raise ValueError(
            "refusing to convert in place: source and destination are "
            "the same file; the legacy archive is preserved as historical")
    ds = _load_dataset_trusted(src_path)
    if ds.manifest is None:
        ds.manifest = build_manifest(ds)
        ds.manifest_sha256 = ds.manifest["manifest_sha256"]
    save_dataset(ds, dst_path)
    return load_dataset(dst_path)
