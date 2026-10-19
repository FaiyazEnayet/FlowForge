# Reproducing the FlowForge demo bundle

This documents the full pipeline from raw recordings to the shipped
`demo/` bundle. The bundle itself carries SHA256 hashes of its inputs
(see `demo/metrics.json` under `provenance`), so you can verify exactly
which run produced it.

## Raw data

The models train on limit-order-book recordings made with the toxicflow
recorder (sibling project). Each chunk is ~30 minutes of BTCUSDT top-of-book
data:

- `quotes.parquet`: time (ms), bid_px, bid_sz, ask_px, ask_sz at ~10 Hz
- `trades.parquet`: time (ms), price, qty, is_buyer_maker

The 8 chunks used for the published results are listed in
`data/split_manifest.json` with per-chunk SHA16 hashes. Recordings are
sequential in time with no overlap. Data rights: recorded from public
Binance market data (data.binance.vision); no exchange account needed.

## Pipeline

All commands assume the repo root and `PYTHONPATH=.`.

### 1. Build the windowed dataset

```bash
python examples/build_data.py \
    ~/workspace/toxicflow/data/chunk1 \
    ... ~/workspace/toxicflow/data/chunk8 \
    --split "train:0,1,2,3,4 val:5 test:6,7"
```

Writes:
- `data/book_windows.npz` — standardized windows (gitignored)
- `data/split_manifest.json` — chunk hashes + split assignment
- `data/dataset_card.md` — sessions, hours, split tables

Splits are session-disjoint (FL3): whole chunks go to one split.

### 2. Sweep model configs (validation only)

```bash
python examples/sweep.py
```

Trains DDPM variants on train, scores on val (mean W1, discriminator
AUC), and compares against the moving-block and VAR(2) baselines (FL4)
with sampling latency. Writes `runs/sweep.json`. Test split untouched.

Measured compute (CPU): the 300-epoch DDPM configs take several hours
each. The baselines take seconds. See `runs/sweep.json` for per-config
train/sample times from the published run.

### 3. Train the winner and evaluate on test

```bash
python examples/demo.py
```

Trains the winning DDPM config (see `DDPM_*` at the top of
`examples/demo.py`) and the VAE baseline on train, generates for val/test
regimes, and prints the full report on test. Writes:
- `runs/metrics.json` — all metrics
- `runs/eval_arrays.npz` — arrays for figures (gitignored)

### 4. Freeze the demo bundle

```bash
python examples/build_demo.py
```

Samples 64 real + 64 generated windows (seed 7) and writes:
- `demo/demo.npz` — ships with the repo
- `demo/metrics.json` — includes `provenance` with SHA256 of
  `runs/eval_arrays.npz` and `runs/metrics.json` plus the DDPM config

### 5. Build the dashboard (optional)

```bash
python examples/build_dashboard.py
```

Writes `docs/dashboard.html` from the demo bundle.

## Offline smoke test

`tests/test_demo.py::test_demo_bundle_provenance` verifies the shipped
bundle: required keys exist, arrays are finite, and `provenance` has the
expected hashes and config. It does not need the raw data or a GPU.

```bash
python -m pytest tests/test_demo.py -q
```

## Environment

- Python 3.10+
- `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `pip install -r requirements.txt`
- CPU only; no GPU required. The published DDPM was trained on CPU.

## What is NOT reproduced in CI

CI runs the unit tests only. Full training (sweep + demo) is documented
above with measured compute but not executed in CI. The shipped `demo/`
bundle is the precomputed output; its provenance hashes let you verify
it matches a documented run.
