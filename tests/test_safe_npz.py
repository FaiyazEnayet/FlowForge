"""FF-N03: safe NPZ dataset loading and explicit legacy migration.

The pre-FF-N03 format stored chunk_dirs / chunk_hashes / chunk_symbols as
NumPy object arrays and load_dataset opened every archive with
allow_pickle=True, so loading a dataset evaluated pickle and a tampered
NPZ could execute arbitrary code. These tests pin the replacement:

- save_dataset writes metadata as typed UTF-8 JSON (no object arrays);
- the default loader opens with allow_pickle=False and validates
  dtypes, finiteness, and split identity against the dataset's own
  embedded manifest (window counts, per-split session membership,
  session_split agreement);
- pre-C02 / object-array archives load only under explicit legacy=True
  or through the opt-in trusted converter, which writes a NEW file and
  never rewrites the source archive.

Fixtures are synthetic; the shape-faithful converter test mirrors the
historical artifact's population (996/168/168 windows, sessions
train [0-5] / val [6] / test [7]).
"""

import hashlib
import json
import os

import numpy as np
import pytest

from flowforge.data import (BookDataset, SplitWindows, build_manifest,
                            load_dataset, manifest_sha256, save_dataset)

SPLITS = ("train", "val", "test")
TENSOR_KEYS = ("x", "regime", "t_start", "t_end", "session_ids")


def _split_windows(n, session_ids, seed):
    rng = np.random.default_rng(seed)
    return SplitWindows(
        x=rng.standard_normal((n, 60, 7)).astype(np.float32),
        regime=rng.standard_normal((n, 2)).astype(np.float32),
        t_start=(np.arange(n, dtype=np.float64) * 600.0),
        t_end=(np.arange(n, dtype=np.float64) * 600.0 + 59.0),
        session_ids=np.asarray(session_ids, dtype=np.int64),
    )


def _make_dataset(n_train=6, n_val=4, n_test=3, session_split=None,
                  seed=0, with_manifest=True):
    if session_split is None:
        session_split = {"train": [0], "val": [1], "test": [2]}
    n_chunks = 1 + max(max(v) for v in session_split.values())
    counts = {"train": n_train, "val": n_val, "test": n_test}

    def ids_for(split):
        sessions = session_split[split]
        n = counts[split]
        return [sessions[i % len(sessions)] for i in range(n)]

    ds = BookDataset(
        train=_split_windows(n_train, ids_for("train"), seed + 1),
        val=_split_windows(n_val, ids_for("val"), seed + 2),
        test=_split_windows(n_test, ids_for("test"), seed + 3),
        feature_mean=np.zeros(7), feature_std=np.ones(7),
        regime_mean=np.zeros(2), regime_std=np.ones(2),
        n_seconds=10_000, n_chunks=n_chunks,
        chunk_dirs=[f"/recordings/chunk{i}" for i in range(n_chunks)],
        chunk_hashes=[f"{i:064x}" for i in range(n_chunks)],
        chunk_symbols=["BTCUSDT"] * n_chunks,
        chunk_stats=[{"span": 3600, "usable": 3600, "missing": 0,
                      "segments": 1} for _ in range(n_chunks)],
        session_split={k: list(v) for k, v in session_split.items()},
    )
    if with_manifest:
        ds.manifest = build_manifest(ds)
        ds.manifest_sha256 = ds.manifest["manifest_sha256"]
    return ds


def _write_old_format(ds, path, with_manifest=True):
    """Replicate the pre-FF-N03 save_dataset layout byte-for-byte in
    structure: chunk metadata as object arrays, manifest embedded only
    when the dataset carries one."""
    extra = {}
    if with_manifest and ds.manifest is not None:
        extra["manifest_json"] = np.array(json.dumps(ds.manifest))
        extra["manifest_sha256"] = np.array(ds.manifest_sha256)
    extra["chunk_file_hashes_json"] = np.array(
        json.dumps(ds.chunk_file_hashes))
    extra["chunk_time_ranges_json"] = np.array(
        json.dumps(ds.chunk_time_ranges))
    extra["chunk_recorder_meta_json"] = np.array(
        json.dumps(ds.chunk_recorder_meta))
    np.savez_compressed(
        path,
        **{f"{s}_{k}": getattr(getattr(ds, s), k)
           for s in SPLITS for k in TENSOR_KEYS},
        feature_mean=ds.feature_mean, feature_std=ds.feature_std,
        regime_mean=ds.regime_mean, regime_std=ds.regime_std,
        n_seconds=np.array(ds.n_seconds), n_chunks=np.array(ds.n_chunks),
        chunk_dirs=np.array(ds.chunk_dirs, dtype=object),
        chunk_hashes=np.array(ds.chunk_hashes, dtype=object),
        chunk_symbols=np.array(ds.chunk_symbols, dtype=object),
        chunk_stats_json=np.array(json.dumps(ds.chunk_stats)),
        session_split_json=np.array(json.dumps(ds.session_split)),
        **extra,
    )


def _rewrite(src, dst, **overrides):
    """Copy an NPZ applying key overrides; returns dst."""
    z = dict(np.load(src, allow_pickle=True))
    for k, v in overrides.items():
        if v is _DROP:
            z.pop(k, None)
        else:
            z[k] = v
    np.savez_compressed(dst, **z)
    return dst


class _Drop:
    pass


_DROP = _Drop()


def _assert_same_tensors(a, b):
    for s in SPLITS:
        for k in TENSOR_KEYS:
            x, y = getattr(getattr(a, s), k), getattr(getattr(b, s), k)
            assert x.dtype == y.dtype, f"{s}_{k} dtype"
            assert x.shape == y.shape, f"{s}_{k} shape"
            assert x.tobytes() == y.tobytes(), f"{s}_{k} bytes differ"
    for k in ("feature_mean", "feature_std", "regime_mean", "regime_std"):
        x, y = getattr(a, k), getattr(b, k)
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), k


# ---------------------------------------------------------------- format

def test_saved_format_has_no_object_arrays(tmp_path):
    ds = _make_dataset()
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    z = np.load(p, allow_pickle=False)
    assert all(z[k].dtype != object for k in z.files)
    for key in ("chunk_dirs", "chunk_hashes", "chunk_symbols"):
        assert key not in z.files
        assert f"{key}_json" in z.files


def test_default_roundtrip_typed_format(tmp_path):
    ds = _make_dataset()
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    ds2 = load_dataset(p)
    assert ds2.legacy is False
    _assert_same_tensors(ds, ds2)
    assert ds2.chunk_dirs == ds.chunk_dirs
    assert ds2.chunk_hashes == ds.chunk_hashes
    assert ds2.chunk_symbols == ds.chunk_symbols
    assert ds2.chunk_stats == ds.chunk_stats
    assert ds2.session_split == ds.session_split
    assert ds2.manifest_sha256 == ds.manifest_sha256


# ------------------------------------------------------- malicious cases

def test_default_loader_rejects_object_array_format(tmp_path):
    # The pre-FF-N03 layout (manifest embedded, metadata pickled) must
    # not load through the default loader at all.
    ds = _make_dataset()
    p = str(tmp_path / "old.npz")
    _write_old_format(ds, p, with_manifest=True)
    with pytest.raises(ValueError):
        load_dataset(p)


class _Marker:
    """Pickle payload: unpickling creates the marker file."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (os.system, (f"touch {self.path}",))


def test_poisoned_object_array_is_never_unpickled(tmp_path):
    ds = _make_dataset()
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    marker = tmp_path / "pwned"
    evil = _rewrite(
        p, str(tmp_path / "evil.npz"),
        chunk_dirs=np.array([_Marker(str(marker))], dtype=object),
        chunk_dirs_json=_DROP)
    with pytest.raises((ValueError, Exception)):
        load_dataset(evil)
    assert not marker.exists(), "default loader evaluated pickle"


def test_legacy_artifact_rejected_by_default_loads_with_flag(tmp_path):
    ds = _make_dataset()
    p = str(tmp_path / "legacy.npz")
    _write_old_format(ds, p, with_manifest=False)
    with pytest.raises(ValueError, match="legacy"):
        load_dataset(p)
    ds2 = load_dataset(p, legacy=True)
    assert ds2.legacy is True
    _assert_same_tensors(ds, ds2)
    assert ds2.chunk_dirs == ds.chunk_dirs


def test_tampered_manifest_rejected(tmp_path):
    ds = _make_dataset()
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    m = json.loads(str(np.load(p)["manifest_json"]))
    m["windows"]["train"] += 1  # content no longer hashes to recorded SHA
    bad = _rewrite(p, str(tmp_path / "bad.npz"),
                   manifest_json=np.array(json.dumps(m)))
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_dataset(bad)


def test_session_split_key_disagreement_rejected(tmp_path):
    # Top-level session_split_json edited; manifest left intact and its
    # hash still valid. The loader must cross-check the two.
    ds = _make_dataset()
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    other = {"train": [1], "val": [0], "test": [2]}
    bad = _rewrite(p, str(tmp_path / "bad.npz"),
                   session_split_json=np.array(json.dumps(other)))
    with pytest.raises(ValueError, match="session_split"):
        load_dataset(bad)


def test_wrong_dtype_rejected(tmp_path):
    ds = _make_dataset()
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    z = np.load(p)
    bad = _rewrite(p, str(tmp_path / "bad.npz"),
                   train_x=z["train_x"].astype(np.float64))
    with pytest.raises(ValueError, match="dtype"):
        load_dataset(bad)


def test_nonfinite_tensor_rejected(tmp_path):
    ds = _make_dataset()
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    x = np.array(np.load(p)["val_x"])
    x[0, 0, 0] = np.nan
    bad = _rewrite(p, str(tmp_path / "bad.npz"), val_x=x)
    with pytest.raises(ValueError, match="non-finite"):
        load_dataset(bad)


def test_manifest_window_count_mismatch_rejected(tmp_path):
    # Manifest edited AND its hash honestly recomputed, so the SHA gate
    # passes; split-identity validation against the arrays must catch
    # the lie.
    ds = _make_dataset()
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    m = json.loads(str(np.load(p)["manifest_json"]))
    m["windows"]["train"] += 1
    m["manifest_sha256"] = manifest_sha256(m)
    bad = _rewrite(p, str(tmp_path / "bad.npz"),
                   manifest_json=np.array(json.dumps(m)),
                   manifest_sha256=np.array(m["manifest_sha256"]))
    with pytest.raises(ValueError, match="windows"):
        load_dataset(bad)


def test_session_membership_mismatch_rejected(tmp_path):
    # A val window relabeled with a train session id: arrays are not
    # covered by the manifest hash, so only split-identity validation
    # catches this.
    ds = _make_dataset()
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    ids = np.zeros_like(np.load(p)["val_session_ids"])  # session 0 = train
    bad = _rewrite(p, str(tmp_path / "bad.npz"), val_session_ids=ids)
    with pytest.raises(ValueError, match="session"):
        load_dataset(bad)


# ------------------------------------------------------------- converter

def test_converter_roundtrip_bit_identical_and_source_preserved(tmp_path):
    from flowforge.data import convert_legacy_dataset
    ds = _make_dataset()
    src = str(tmp_path / "legacy.npz")
    _write_old_format(ds, src, with_manifest=False)
    before = hashlib.sha256(open(src, "rb").read()).hexdigest()
    dst = str(tmp_path / "converted.npz")
    out = convert_legacy_dataset(src, dst)
    after = hashlib.sha256(open(src, "rb").read()).hexdigest()
    assert before == after, "converter rewrote the source archive"
    ds2 = load_dataset(dst)  # default loader, no legacy flag
    assert ds2.legacy is False
    _assert_same_tensors(ds, ds2)
    assert ds2.session_split == ds.session_split
    assert ds2.chunk_dirs == ds.chunk_dirs
    assert out.manifest_sha256 == ds2.manifest_sha256


def test_converter_refuses_in_place(tmp_path):
    from flowforge.data import convert_legacy_dataset
    ds = _make_dataset()
    src = str(tmp_path / "legacy.npz")
    _write_old_format(ds, src, with_manifest=False)
    with pytest.raises(ValueError, match="in place|same"):
        convert_legacy_dataset(src, src)


def test_converter_acceptance_population_shapes(tmp_path):
    # Shape-faithful mirror of the historical artifact: 996/168/168
    # windows over sessions train [0-5], val [6], test [7].
    from flowforge.data import convert_legacy_dataset
    split = {"train": [0, 1, 2, 3, 4, 5], "val": [6], "test": [7]}
    ds = _make_dataset(n_train=996, n_val=168, n_test=168,
                       session_split=split, seed=7)
    src = str(tmp_path / "book_windows_legacy.npz")
    _write_old_format(ds, src, with_manifest=False)
    dst = str(tmp_path / "book_windows.npz")
    convert_legacy_dataset(src, dst)
    ds2 = load_dataset(dst)
    assert ds2.train.x.shape == (996, 60, 7)
    assert ds2.val.x.shape == (168, 60, 7)
    assert ds2.test.x.shape == (168, 60, 7)
    _assert_same_tensors(ds, ds2)
    assert ds2.session_split == split


def test_demo_bundle_loads(tmp_path):
    # The committed demo bundle carries no object arrays; its loader
    # must work with pickle disabled.
    from flowforge.demo import load_demo_bundle
    bundle = load_demo_bundle()
    assert bundle["feature_names"]
    assert 0.0 <= bundle["disc_auc"] <= 1.0
