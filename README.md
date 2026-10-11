# FlowForge

**Teach an AI what a real market looks like, then ask it to dream up new ones.**

A limit order book is the waiting list of a market: who wants to buy, who wants to sell, at what price, in what size, updating thousands of times a second. I recorded about three hours of the real one for BTCUSDT, cut it into one minute windows, and trained generative models to produce new windows from scratch. Then I measured, honestly, how close the dreams come to reality.

This is simulation research. Nothing here trades, connects to a trading endpoint, or touches real money. The point is the generator and the measuring stick, not a strategy.

## How it works

**Data.** My toxicflow recorder saves the top of the book (best bid and ask with sizes) ten times a second, plus every trade with its aggressor side. Six thirty minute sessions gave 10,781 seconds. I turn each second into one state vector: log return of the midprice, spread, log sizes, book imbalance, signed trade volume, trade count. Sixty seconds make one window. Windows are split by time, first 70 percent train, next 15 percent validation, last 15 percent test, and a window that straddles a boundary is dropped. The generator never sees validation or test. Standardization uses train statistics only.

**Models.** Two, both conditioned on regime features (recent volatility and spread, computed strictly from the past before each window):

- A conditional VAE baseline, 289k params. Small on purpose. It is the thing to beat.
- A DDPM diffusion model, 222k params. A small 1D conv net learns to remove noise from a window over 100 steps, conditioned on the diffusion timestep and the regime. One real bug caught along the way: my first noise schedule never reached pure noise at the last step, so sampling started from the wrong distribution and every generated window came out under dispersed. The code now asserts the schedule actually reaches noise.

**Evaluation.** This is the deliverable. On the held out test split:

1. Marginal distributions. For each of the seven features: real vs generated mean, std, and 1-Wasserstein distance.
2. Temporal structure. Autocorrelation of signed order flow at lags 1 to 5. In this sample the real flow is near zero at one second bars, so the check is whether the generator reproduces that flatness instead of inventing structure.
3. A discriminator test. Train a classifier on validation windows (real vs generated) and report its AUC on test windows the generator never saw. Fixed seeds. AUC near 0.5 means the dreams are indistinguishable.

Every metric was sanity checked on synthetic inputs with known answers first: Wasserstein of identical distributions is ~0 and of N(0,1) vs N(2,1) is ~2, autocorrelation recovers 0.8 on AR(1) data, the discriminator scores ~0.5 on identical distributions and ~1.0 on separated ones. If a check had failed I would have fixed the metric, not the test.

## What I found

Real numbers from `examples/demo.py`, seed 7, about 1.2 minutes of CPU training. 672 train windows, 166 validation, 168 test.

The diffusion model beats the VAE clearly. Mean Wasserstein distance across features: 0.267 vs 0.356. Discriminator AUC on the held out test split: 0.734 vs 0.915. The VAE collapses to conditional means, generating about a fifth of the real variance on most features, and even invents fake autocorrelation the real data does not have (0.28 at lag 4 vs 0.00 real). The diffusion model matches variances closely (returns 0.605 vs 0.568 real, imbalance 0.840 vs 0.879, signed flow 0.321 vs 0.334) and reproduces the flat autocorrelation.

Honest gaps. The discriminator still separates diffusion output from real windows at 0.734, so the dreams are good but not perfect. There are mean shifts on some features (imbalance -0.31 generated vs 0.21 real). And one data truth worth knowing: the BTC spread sits pinned at one tick most of the time, so the spread feature is nearly a point mass and mostly tests whether the generator reproduces that.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy pandas pyarrow pytest

# build windows from toxicflow recordings (data/ is gitignored, rebuild it)
PYTHONPATH=. .venv/bin/python examples/build_data.py ~/workspace/toxicflow/data/chunk1 .../chunk6

# train both models and print the full evaluation
PYTHONPATH=. .venv/bin/python examples/demo.py

# tests, including the metric sanity checks
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

## What is in here

- `flowforge/data.py`: parquet to standardized windows, time based splits, leakage discipline documented in the docstring.
- `flowforge/models.py`: conditional VAE baseline and the DDPM (1D conv denoiser, regime conditioning, validated noise schedule).
- `flowforge/evaluate.py`: Wasserstein distances, autocorrelation, the held out discriminator with fixed seeds, plus a printable report.
- `examples/demo.py`: end to end training and evaluation. `examples/build_data.py`: rebuild the window dataset.
- `tests/`: 18 tests. Shapes, NaN handling, no time overlap between splits, and every metric checked against synthetic inputs with known answers.

## Honest limitations

- Small models on CPU, three hours of one asset. This is a proof of concept, not a production simulator.
- BTCUSDT spot only. Crypto microstructure is not equities: no open or close, different fee structure, 24/7 retail flow.
- The recorder saves only the touch, not full depth, so the model learns top of book dynamics only.
- One calm three hour session. Different regimes would give different numbers.
- Research code, not trading advice. Nothing here should touch real money.
