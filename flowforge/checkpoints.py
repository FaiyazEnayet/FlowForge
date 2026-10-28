"""Exact inference and resumable checkpoints for the generative models.

Two checkpoint kinds, both stored as a directory:

  <dir>/manifest.json   provenance and digests (canonical JSON)
  <dir>/model.pt        model state dict (torch.save)
  <dir>/ema.pt          EMA shadow weights, resumable checkpoints only
  <dir>/optimizer.pt    optimizer state dict, resumable checkpoints only
  <dir>/rng.json        torch, Python, and NumPy RNG states, resumable only

An inference checkpoint holds ready-to-sample weights: for a DDPM trained
with EMA, the saved weights are the EMA shadow (the effective sampling
weights), recorded as weights_source "ema"; otherwise the trained weights
(weights_source "trained"). A resumable checkpoint holds the raw training
weights plus EMA shadow, optimizer state, and RNG states, so the next
optimizer step after a reload matches the uninterrupted run exactly.

Every load is fail-closed: a wrong format version, a manifest digest
mismatch, a weights digest mismatch, or a state dict key mismatch raises
ValueError and loads nothing. Nothing here reconstructs history:
historical weight tensors were never saved, so HISTORIC_WEIGHTS_STATUS
says unavailable, and no function in this module invents them.

No trading, no order execution, no exchange connectivity: these are pure
sequence generators, and checkpoints are just their weights plus
provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess

import numpy as np
import torch

from .models import EMA, BookDDPM, CondVAE

FORMAT = "ff-checkpoint/1"
FORMAT_VERSION = 1

HISTORIC_WEIGHTS_STATUS = (
    "unavailable: historical weight tensors were never saved, so no "
    "checkpoint can reproduce the historical models; only checkpoints "
    "saved by this module, with their recorded provenance, are loadable"
)

_MODEL_CLASSES = {"BookDDPM": BookDDPM, "CondVAE": CondVAE}

_NORMALIZER_KEYS = ("feature_mean", "feature_std", "regime_mean",
                    "regime_std")


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_commit_sha(explicit: str | None = None) -> str:
    """Commit SHA a checkpoint cites. Explicit value wins; otherwise the
    git HEAD of this repository. Raises ValueError when neither is
    available, instead of writing a manifest with a missing SHA."""
    if explicit:
        if not isinstance(explicit, str):
            raise ValueError("commit_sha must be a string")
        return explicit
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True,
                             cwd=repo_root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "cannot resolve commit SHA: not a git checkout and no "
            "explicit commit_sha given") from exc
    sha = out.stdout.strip()
    if not sha:
        raise ValueError("cannot resolve commit SHA: empty git output")
    return sha


def build_model(model_class: str, config: dict):
    """Rebuild a model from its recorded class and config."""
    cls = _MODEL_CLASSES.get(model_class)
    if cls is None:
        raise ValueError(f"unknown model_class {model_class!r}")
    try:
        return cls(**config)
    except TypeError as exc:
        raise ValueError(
            f"config does not construct {model_class}: {exc}") from exc


def _check_normalizer(normalizer: dict, n_features: int) -> dict:
    if set(normalizer) != set(_NORMALIZER_KEYS):
        raise ValueError(
            f"normalizer must carry exactly {_NORMALIZER_KEYS}")
    out = {}
    for key in _NORMALIZER_KEYS:
        vals = [float(v) for v in normalizer[key]]
        want = n_features if key.startswith("feature") else 2
        if len(vals) != want:
            raise ValueError(
                f"normalizer {key} has {len(vals)} values, expected {want}")
        out[key] = vals
    return out


def _check_config_against_model(model, model_class: str, config: dict):
    cls = _MODEL_CLASSES.get(model_class)
    if cls is None:
        raise ValueError(f"unknown model_class {model_class!r}")
    if not isinstance(model, cls):
        raise ValueError(
            f"model is {type(model).__name__}, not {model_class}")
    # Rebuilding consumes the global torch RNG stream; restore it so
    # saving a checkpoint never perturbs the caller's training run.
    rng_state = torch.get_rng_state()
    try:
        rebuilt = build_model(model_class, config)
    finally:
        torch.set_rng_state(rng_state)
    saved_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    rebuilt_shapes = {k: tuple(v.shape)
                      for k, v in rebuilt.state_dict().items()}
    if saved_shapes != rebuilt_shapes:
        raise ValueError(
            "config does not reproduce the model architecture: "
            "state dict keys or shapes differ")


def inverse_transform(samples: np.ndarray, normalizer: dict) -> np.ndarray:
    """Standardized windows back to physical units using the train-only
    normalizer recorded in the checkpoint manifest."""
    mean = np.asarray(normalizer["feature_mean"], dtype=np.float64)
    std = np.asarray(normalizer["feature_std"], dtype=np.float64)
    return np.asarray(samples, dtype=np.float64) * std + mean


def _serialize_rng() -> dict:
    torch_state = torch.get_rng_state().tolist()
    py_version, py_internal, py_gauss = random.getstate()
    np_kind, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    return {
        "torch": torch_state,
        "python": {"version": py_version, "internal": list(py_internal),
                   "gauss_next": py_gauss},
        "numpy": {"kind": str(np_kind), "keys": [int(k) for k in np_keys],
                  "pos": int(np_pos), "has_gauss": int(np_has_gauss),
                  "cached_gaussian": float(np_cached)},
    }


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(torch.ByteTensor(state["torch"]))
    py = state["python"]
    random.setstate((py["version"], tuple(py["internal"]),
                     py["gauss_next"]))
    npy = state["numpy"]
    np.random.set_state((npy["kind"],
                         np.array(npy["keys"], dtype=np.uint32),
                         npy["pos"], npy["has_gauss"],
                         npy["cached_gaussian"]))


def _write_manifest(path: str, manifest: dict, weights_sha: str) -> dict:
    manifest = dict(manifest)
    manifest["weights_sha256"] = weights_sha
    digest = _sha256_text(_canonical_json(manifest))
    manifest["manifest_sha256"] = digest
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


def _prepare_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite or not os.path.isdir(path):
            raise ValueError(
                f"checkpoint directory {path!r} already exists")
        for name in os.listdir(path):
            os.remove(os.path.join(path, name))
    else:
        os.makedirs(path)


def _base_manifest(kind: str, model, model_class: str, config: dict,
                   normalizer: dict, commit_sha: str | None,
                   split_manifest_sha256: str | None) -> dict:
    _check_config_against_model(model, model_class, config)
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "model_class": model_class,
        "config": config,
        "normalizer": _check_normalizer(normalizer,
                                        model.n_features),
        "commit_sha": resolve_commit_sha(commit_sha),
        "split_manifest_sha256": split_manifest_sha256,
    }


def save_inference_checkpoint(path: str, model, model_class: str,
                              config: dict, normalizer: dict,
                              commit_sha: str | None = None,
                              split_manifest_sha256: str | None = None,
                              ema: EMA | None = None,
                              overwrite: bool = False) -> dict:
    """Save ready-to-sample weights. With ema given, the EMA shadow is the
    saved weight set (weights_source "ema"); otherwise the model weights
    (weights_source "trained"). Returns the written manifest."""
    manifest = _base_manifest("inference", model, model_class, config,
                              normalizer, commit_sha, split_manifest_sha256)
    if ema is not None:
        # Effective weights: EMA shadow for parameters, schedule buffers
        # from the model (the shadow tracks parameters only, and buffers
        # are identical either way). Saving the shadow alone would drop
        # the buffers and fail a strict reload.
        weights = {k: v.detach().clone()
                   for k, v in model.state_dict().items()}
        for k, v in ema.shadow.items():
            weights[k] = v.detach().clone()
        manifest["weights_source"] = "ema"
        manifest["ema_decay"] = ema.decay
    else:
        weights = model.state_dict()
        manifest["weights_source"] = "trained"
    _prepare_dir(path, overwrite)
    weights_path = os.path.join(path, "model.pt")
    torch.save(weights, weights_path)
    with open(weights_path, "rb") as f:
        weights_sha = _sha256_bytes(f.read())
    return _write_manifest(path, manifest, weights_sha)


def save_resumable_checkpoint(path: str, model, optimizer, model_class: str,
                              config: dict, normalizer: dict,
                              commit_sha: str | None = None,
                              split_manifest_sha256: str | None = None,
                              ema: EMA | None = None,
                              overwrite: bool = False) -> dict:
    """Save raw training weights plus EMA shadow (when given), optimizer
    state, and RNG states. Returns the written manifest."""
    manifest = _base_manifest("resumable", model, model_class, config,
                              normalizer, commit_sha, split_manifest_sha256)
    manifest["has_ema"] = ema is not None
    _prepare_dir(path, overwrite)
    weights_path = os.path.join(path, "model.pt")
    torch.save(model.state_dict(), weights_path)
    with open(weights_path, "rb") as f:
        weights_sha = _sha256_bytes(f.read())
    if ema is not None:
        torch.save({k: v.detach().clone()
                    for k, v in ema.shadow.items()},
                   os.path.join(path, "ema.pt"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    with open(os.path.join(path, "rng.json"), "w") as f:
        json.dump(_serialize_rng(), f)
    return _write_manifest(path, manifest, weights_sha)


def _read_manifest(path: str) -> dict:
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise ValueError(f"no manifest.json in checkpoint {path!r}")
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"checkpoint manifest is not valid JSON: {exc}"
                         ) from exc
    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"checkpoint format {manifest.get('format')!r} is not {FORMAT}")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format_version {manifest.get('format_version')!r} "
            f"is not {FORMAT_VERSION}")
    return manifest


def _verify_digests(path: str, manifest: dict) -> None:
    weights_path = os.path.join(path, "model.pt")
    if not os.path.isfile(weights_path):
        raise ValueError(f"no model.pt in checkpoint {path!r}")
    with open(weights_path, "rb") as f:
        weights_sha = _sha256_bytes(f.read())
    if weights_sha != manifest.get("weights_sha256"):
        raise ValueError("checkpoint weights digest mismatch: model.pt "
                         "does not match the manifest")
    check = {k: v for k, v in manifest.items()
             if k != "manifest_sha256"}
    if _sha256_text(_canonical_json(check)) != manifest.get(
            "manifest_sha256"):
        raise ValueError("checkpoint manifest digest mismatch: manifest "
                         "was altered after saving")


def load_checkpoint(path: str, model_class: str | None = None,
                    config: dict | None = None):
    """Load an inference checkpoint. Returns (model, manifest). Raises
    ValueError on any version, digest, class, config, or key mismatch,
    and loads nothing in that case."""
    manifest = _read_manifest(path)
    if manifest["kind"] != "inference":
        raise ValueError(
            f"checkpoint kind {manifest['kind']!r} is not 'inference'; "
            "use load_resumable_checkpoint")
    if model_class is not None and model_class != manifest["model_class"]:
        raise ValueError(
            f"checkpoint model_class {manifest['model_class']!r} does not "
            f"match expected {model_class!r}")
    if config is not None and config != manifest["config"]:
        raise ValueError("checkpoint config does not match expected config")
    _verify_digests(path, manifest)
    model = build_model(manifest["model_class"], manifest["config"])
    try:
        model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"),
                       weights_only=True))
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint weights do not fit the recorded architecture: "
            f"{exc}") from exc
    model.eval()
    return model, manifest


def load_resumable_checkpoint(path: str, model, optimizer, ema=None):
    """Restore a resumable checkpoint into a live model, optimizer, and
    (when the checkpoint carries one) EMA object, plus the RNG states.
    The model and optimizer must already match the recorded architecture;
    returns the manifest. Raises ValueError on any mismatch."""
    manifest = _read_manifest(path)
    if manifest["kind"] != "resumable":
        raise ValueError(
            f"checkpoint kind {manifest['kind']!r} is not 'resumable'")
    _verify_digests(path, manifest)
    if model.__class__.__name__ != manifest["model_class"]:
        raise ValueError(
            f"model is {model.__class__.__name__}, checkpoint records "
            f"{manifest['model_class']}")
    rebuilt = build_model(manifest["model_class"], manifest["config"])
    live_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    rebuilt_shapes = {k: tuple(v.shape)
                      for k, v in rebuilt.state_dict().items()}
    if live_shapes != rebuilt_shapes:
        raise ValueError("model architecture does not match the "
                         "checkpoint config")
    try:
        model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"),
                       weights_only=True))
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint weights do not fit the model: {exc}") from exc
    if manifest.get("has_ema"):
        if ema is None:
            raise ValueError("checkpoint carries EMA weights but no EMA "
                             "object was given to restore them into")
        ema_path = os.path.join(path, "ema.pt")
        if not os.path.isfile(ema_path):
            raise ValueError("checkpoint claims EMA weights but ema.pt "
                             "is missing")
        shadow = torch.load(ema_path, weights_only=True)
        if set(shadow) != set(ema.shadow):
            raise ValueError("EMA shadow keys do not match the model")
        for k, v in shadow.items():
            ema.shadow[k].copy_(v)
    elif ema is not None:
        raise ValueError("checkpoint carries no EMA weights but an EMA "
                         "object was given")
    opt_path = os.path.join(path, "optimizer.pt")
    if not os.path.isfile(opt_path):
        raise ValueError("resumable checkpoint is missing optimizer.pt")
    saved_opt = torch.load(opt_path, weights_only=True)
    if len(saved_opt["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("optimizer param group count does not match "
                         "the checkpoint")
    optimizer.load_state_dict(saved_opt)
    rng_path = os.path.join(path, "rng.json")
    if not os.path.isfile(rng_path):
        raise ValueError("resumable checkpoint is missing rng.json")
    with open(rng_path) as f:
        _restore_rng(json.load(f))
    return manifest


def _cli_generate(args) -> int:
    model, manifest = load_checkpoint(args.checkpoint)
    data = np.load(args.regime)
    regime = torch.from_numpy(np.asarray(data["regime"], dtype=np.float32))
    if manifest["model_class"] == "BookDDPM":
        with torch.no_grad():
            samples = model.sample(regime, seed=args.seed).numpy()
    else:
        from .models import set_seed
        set_seed(args.seed)
        with torch.no_grad():
            samples = model.generate(regime).numpy()
    samples = np.ascontiguousarray(samples, dtype=np.float64)
    inv = np.ascontiguousarray(
        inverse_transform(samples, manifest["normalizer"]))
    np.savez(args.out, samples=samples, samples_inverse=inv)
    print(json.dumps({
        "samples_sha256": _sha256_bytes(samples.tobytes()),
        "samples_inverse_sha256": _sha256_bytes(inv.tobytes()),
    }))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="flowforge.checkpoints",
        description="Load a checkpoint and generate from it.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="generate from a checkpoint")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--regime", required=True,
                     help=".npz file with a 'regime' array")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", required=True, help="output .npz path")
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _cli_generate(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
