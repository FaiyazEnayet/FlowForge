"""Stage 3 C02 tests: canonical dataset hashes, locked manifests, legacy mode.

Fixture-first: these tests define the C02 contract before implementation.
All tests in this file are prefixed test_stage3_flowforge_02_.
"""

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from flowforge.data import build_dataset, load_dataset, save_dataset

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _synth_chunk(tmp_path, name, n_sec=900, seed=0, t0_ms=1_700_000_000_000):
    rng = np.random.default_rng(seed)
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
    n_t = n_sec * 4
    t = pd.DataFrame({"time": np.sort(rng.integers(t0_ms, t0_ms + n_sec * 1000, n_t)),
                      "price": 80000 + rng.normal(0, 5, n_t),
                      "qty": rng.uniform(0.001, 0.5, n_t),
                      "is_buyer_maker": rng.random(n_t) < 0.5})
    d = tmp_path / name
    d.mkdir()
    q.to_parquet(d / "quotes.parquet")
    t.to_parquet(d / "trades.parquet")
    with open(d / "meta.json", "w") as f:
        json.dump({"symbol": "BTCUSDT"}, f)
    return str(d)


def _three_chunks(tmp_path, n_sec=900, gap_ms=2_000_000, seed0=0):
    return [_synth_chunk(tmp_path, f"c{i}", n_sec=n_sec,
                         t0_ms=1_700_000_000_000 + i * gap_ms, seed=seed0 + i)
            for i in range(3)]


def test_stage3_flowforge_02_canonical_hash_stable_across_processes(tmp_path):
    # Identical raw bytes in separate processes must yield identical hashes.
    # The canonical hash excludes wall-clock fields, so it is stable.
    from flowforge.data import build_manifest
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    assert ds.manifest_sha256 is not None
    assert ds.manifest_sha256 == build_manifest(ds)["manifest_sha256"]
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); "
        "from flowforge.data import build_dataset; "
        "ds = build_dataset(sys.argv[2:]); print(ds.manifest_sha256)"
    )
    import os
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    out = subprocess.run([sys.executable, "-c", code, str(REPO_ROOT), *dirs],
                         capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == ds.manifest_sha256


def test_stage3_flowforge_02_one_altered_record_changes_hash(tmp_path):
    import shutil
    from flowforge.data import build_manifest
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    base_sha = ds.manifest_sha256
    base_chunk = ds.chunk_file_hashes[2]["trades_sha256"]
    # Alter exactly one trade record in the third chunk and rewrite.
    alt = tmp_path / "c2alt"
    shutil.copytree(dirs[2], alt)
    t = pd.read_parquet(alt / "trades.parquet")
    t.loc[t.index[0], "qty"] = float(t.loc[t.index[0], "qty"]) + 1.0
    t.to_parquet(alt / "trades.parquet")
    ds2 = build_dataset([dirs[0], dirs[1], str(alt)])
    assert ds2.manifest_sha256 != base_sha
    assert ds2.chunk_file_hashes[2]["trades_sha256"] != base_chunk
    # Quotes untouched: quotes hash identical.
    assert ds2.chunk_file_hashes[2]["quotes_sha256"] == \
        ds.chunk_file_hashes[2]["quotes_sha256"]
    assert build_manifest(ds2)["manifest_sha256"] == ds2.manifest_sha256


def test_stage3_flowforge_02_duplicate_raw_bytes_detected(tmp_path):
    import shutil
    dirs = _three_chunks(tmp_path)
    dup = tmp_path / "cdup"
    shutil.copytree(dirs[0], dup)
    with pytest.raises(ValueError, match="duplicat"):
        build_dataset([dirs[0], str(dup), dirs[1], dirs[2]])


def test_stage3_flowforge_02_overlapping_but_dropped_chunk_detected(tmp_path):
    # A chunk whose bars are all dropped (one second: first bar has no
    # previous second) must still be caught when its raw time range
    # overlaps another session. The old code skipped empty chunks before
    # the overlap check, so this slipped through silently.
    dirs = _three_chunks(tmp_path)
    q = pd.read_parquet(f"{dirs[0]}/quotes.parquet")
    t0 = 1_700_000_000_000
    one_sec = q[q["time"] < t0 + 1000]
    t = pd.read_parquet(f"{dirs[0]}/trades.parquet")
    t_one = t[t["time"] < t0 + 1000]
    if len(t_one) == 0:  # keep the fixture deterministic
        t_one = pd.DataFrame({"time": [t0 + 500], "price": [80000.0],
                              "qty": [0.1], "is_buyer_maker": [True]})
    d = tmp_path / "overlap1s"
    d.mkdir()
    one_sec.to_parquet(d / "quotes.parquet")
    t_one.to_parquet(d / "trades.parquet")
    with pytest.raises(ValueError, match="overlap"):
        build_dataset([dirs[0], dirs[1], dirs[2], str(d)])


def test_stage3_flowforge_02_manifest_required_fields(tmp_path):
    from flowforge.data import MANIFEST_SCHEMA_VERSION, build_manifest, \
        manifest_sha256
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    m = build_manifest(ds)
    assert m["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION == 2
    assert len(m["manifest_sha256"]) == 64
    assert manifest_sha256(m) == m["manifest_sha256"]
    assert "data_transform_version" in m
    assert "random_seed" in m  # present even when null (deterministic build)
    assert "code_sha" in m
    assert m["n_chunks"] == 3
    assert len(m["chunks"]) == 3
    chunk_keys = {"session_id", "dir", "symbol", "sha256", "quotes_sha256",
                  "trades_sha256", "split", "seconds_span", "seconds_usable",
                  "seconds_missing", "segments", "quotes_t_min_ms",
                  "quotes_t_max_ms", "trades_t_min_ms", "trades_t_max_ms",
                  "day", "recorder_meta", "recorder_schema_version"}
    for c in m["chunks"]:
        assert chunk_keys <= set(c.keys()), f"missing keys: {chunk_keys - set(c.keys())}"
        assert len(c["sha256"]) == 64
        assert len(c["quotes_sha256"]) == 64
        assert len(c["trades_sha256"]) == 64
        assert c["segments"] >= 1
        assert c["day"] == "2023-11-14"  # UTC date of t0 1700000000000 ms
        assert c["symbol"] == "BTCUSDT"
        assert c["dir"] in ("c0", "c1", "c2")  # sanitized: basename only
    assert m["session_split"] == ds.session_split
    assert set(m["windows"].keys()) == {"train", "val", "test"}


def test_stage3_flowforge_02_manifest_file_match(tmp_path):
    from flowforge.data import check_manifest_file, write_manifest
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    mp = tmp_path / "manifest.json"
    write_manifest(ds, str(mp))
    assert check_manifest_file(ds, str(mp)) is True
    assert check_manifest_file(ds, str(tmp_path / "nope.json")) is False
    # Tamper with one chunk hash in the file: must abort, not warn.
    with open(mp) as f:
        m = json.load(f)
    m["chunks"][0]["sha256"] = "0" * 64
    with open(mp, "w") as f:
        json.dump(m, f)
    with pytest.raises(ValueError, match="mismatch"):
        check_manifest_file(ds, str(mp))


def test_stage3_flowforge_02_old_npz_requires_legacy_mode(tmp_path):
    from flowforge.data import ensure_publishable
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    # Strip all C02 provenance keys to simulate a pre-C02 NPZ.
    z = dict(np.load(p, allow_pickle=True))
    for k in ("manifest_json", "manifest_sha256", "chunk_file_hashes_json",
              "chunk_time_ranges_json", "chunk_recorder_meta_json"):
        z.pop(k, None)
    legacy_p = str(tmp_path / "ds_old.npz")
    np.savez(legacy_p, **z)
    with pytest.raises(ValueError, match="legacy"):
        load_dataset(legacy_p)
    ds_old = load_dataset(legacy_p, legacy=True)
    assert ds_old.legacy is True
    assert np.array_equal(ds_old.train.x, ds.train.x)
    with pytest.raises(RuntimeError, match="legacy"):
        ensure_publishable(ds_old)
    ensure_publishable(ds)  # fresh dataset publishes fine


def test_stage3_flowforge_02_no_silent_minus_one_identities(tmp_path):
    # A true pre-FL3 NPZ (no manifest, no session_ids): without legacy mode
    # it must fail; with explicit legacy mode the -1 fill is explicit
    # (legacy flag set), never silent.
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    z = dict(np.load(p, allow_pickle=True))
    for k in ("manifest_json", "manifest_sha256", "chunk_file_hashes_json",
              "chunk_time_ranges_json", "chunk_recorder_meta_json"):
        z.pop(k, None)
    for s in ("train", "val", "test"):
        z.pop(f"{s}_session_ids", None)
    p2 = str(tmp_path / "ds_old_nosess.npz")
    np.savez(p2, **z)
    with pytest.raises(ValueError):
        load_dataset(p2)
    ds_old = load_dataset(p2, legacy=True)
    assert ds_old.legacy is True
    assert (ds_old.train.session_ids == -1).all()
    # A manifest-carrying file with session_ids stripped is corrupt: the
    # manifest claims session coverage the arrays cannot back, so -1 fill
    # is refused even under explicit legacy mode.
    z2 = dict(np.load(p, allow_pickle=True))
    for s in ("train", "val", "test"):
        z2.pop(f"{s}_session_ids", None)
    p3 = str(tmp_path / "ds_corrupt.npz")
    np.savez(p3, **z2)
    with pytest.raises(ValueError):
        load_dataset(p3, legacy=True)


def test_stage3_flowforge_02_save_load_roundtrip_preserves_manifest(tmp_path):
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    p = str(tmp_path / "ds.npz")
    save_dataset(ds, p)
    ds2 = load_dataset(p)
    assert ds2.legacy is False
    assert ds2.manifest_sha256 == ds.manifest_sha256
    assert ds2.manifest["manifest_sha256"] == ds.manifest["manifest_sha256"]
    assert ds2.chunk_file_hashes == ds.chunk_file_hashes
    assert ds2.chunk_time_ranges == ds.chunk_time_ranges
    assert ds2.chunk_recorder_meta == ds.chunk_recorder_meta
    assert ds2.chunk_hashes == ds.chunk_hashes
    assert ds2.code_sha == ds.code_sha


def test_stage3_flowforge_02_validate_manifest_against_recordings(tmp_path):
    from flowforge.data import validate_manifest, write_manifest
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    mp = tmp_path / "manifest.json"
    manifest = write_manifest(ds, str(mp))
    assert validate_manifest(manifest, dirs) == manifest
    # Alter one raw quote value: the file hash must catch it.
    q = pd.read_parquet(f"{dirs[1]}/quotes.parquet")
    q.loc[q.index[0], "bid_px"] = float(q.loc[q.index[0], "bid_px"]) + 1.0
    q.to_parquet(f"{dirs[1]}/quotes.parquet")
    with pytest.raises(ValueError, match="mismatch"):
        validate_manifest(manifest, dirs)
    # Missing field must also fail, naming the field.
    bad = json.loads(json.dumps(manifest))
    del bad["chunks"][0]["sha256"]
    with pytest.raises(ValueError, match="sha256"):
        validate_manifest(bad, dirs)


def _load_build_demo():
    path = REPO_ROOT / "examples" / "build_demo.py"
    spec = importlib.util.spec_from_file_location("ff_build_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_runs(tmp_path, with_manifest_sha, name="runs"):
    runs = tmp_path / name
    runs.mkdir()
    rng = np.random.default_rng(0)
    np.savez(
        runs / "eval_arrays.npz",
        real_test=rng.normal(size=(70, 60, 7)).astype(np.float32),
        gen_test_ddpm=rng.normal(size=(70, 60, 7)).astype(np.float32),
        disc_scores_real=np.array([0.4, 0.5], dtype=np.float32),
        disc_scores_gen=np.array([0.5, 0.6], dtype=np.float32),
        feature_mean=np.zeros(7), feature_std=np.ones(7),
    )
    metrics = {
        "seed": 7, "n_train": 10, "n_val": 5, "n_test": 8,
        "seconds": 100, "chunks": 3,
        "ddpm_config": {"params": 123, "epochs": 2},
        "ddpm": {"discriminator": {"auc_test": 0.6, "auc_val": 0.55},
                 "marginals": {"mean_w1": 0.3,
                               "rows": [{"feature": "log_ret"}]},
                 "flow_acf": {"rows": []}},
        "vae": {"discriminator": {"auc_test": 0.5, "auc_val": 0.5},
                "marginals": {"mean_w1": 0.4},
                "flow_acf": {"rows": []}},
    }
    if with_manifest_sha:
        metrics["split_manifest_sha256"] = "a" * 64
    with open(runs / "metrics.json", "w") as f:
        json.dump(metrics, f)
    return runs


def test_stage3_flowforge_02_demo_bundle_requires_manifest_provenance(tmp_path):
    mod = _load_build_demo()
    demo = tmp_path / "demo"
    mod.RUNS = str(_fake_runs(tmp_path, with_manifest_sha=False, name="runs1"))
    mod.DEMO = str(demo)
    with pytest.raises(ValueError, match="split-manifest"):
        mod.main()
    mod.RUNS = str(_fake_runs(tmp_path, with_manifest_sha=True, name="runs2"))
    mod.main()
    with open(demo / "metrics.json") as f:
        summary = json.load(f)
    assert summary["provenance"]["split_manifest_sha256"] == "a" * 64
    assert summary["provenance"]["build_seed"] == 7


def test_stage3_flowforge_02_committed_manifest_is_sanitized(tmp_path):
    from flowforge.data import write_manifest
    dirs = _three_chunks(tmp_path)
    ds = build_dataset(dirs)
    mp = tmp_path / "manifest.json"
    write_manifest(ds, str(mp))
    text = mp.read_text()
    assert str(tmp_path) not in text  # no absolute local paths
    assert "faiyaz" not in text.lower() and "hatch" not in text.lower()
    for c in json.loads(text)["chunks"]:
        assert "/" not in c["dir"]  # basename only
