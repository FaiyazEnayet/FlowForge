# Experiments

This log records the preregistered studies run on the recorded market
data: the frozen design, the exact corpus and protocol, the measured
numbers, and the nulls. Nothing here is a claim about a committed
artifact: the recorded dataset NPZ is gitignored, so every number
describes the protocol on the dataset whose manifest SHA is recorded
with the run. File note: this log did not exist at the base commit of
FF-R08; it was created by that unit as the discovered home for
experiment records (the contract named `research/EXPERIMENTS.md` as a
proposed path and no research directory existed at HEAD).

## FF-R08: Preregistered marginal-W1 versus joint fidelity ranking stability

Question: does the ranking of generative methods by marginal
1-Wasserstein distance agree with the ranking by joint and
conditional lag quality and by a powered held-out discriminator?
Model selection on marginal W1 alone is only trustworthy when the
ranking is stable across those views.

### Frozen design

- Corpus: expanded 28-session corpus (8 original chunks plus 20
  B1-QC 4 h sessions, Nov 9-12 2026). Train sessions 0-19 (18,468
  windows), val sessions 20-23 (5,712), test sessions 24-27 (5,712).
  Split-manifest SHA
  b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd74fb861e1446ae581e.
  Fixed 1,000-window strided val/test subsamples, identical across
  every policy. Source days: train Oct 10-11 and Nov 9-11, val
  Nov 11-12, test Nov 12.
- Model factorial: DDPM size {base ch=96 (221,767 parameters),
  small ch=48 (75,751 parameters)} crossed with EMA {off, on
  (decay 0.999)}. Each cell trained 6 epochs, batch 32, training
  seed 0. One ancestral (100 NFE) sample per cell, plus a DDIM 10
  NFE budget point reusing the base raw weights. VAE comparator at
  lr 2e-4, 25 epochs, seed 0. Regime-matched chunk-copy as
  reference. Generation seeds (0, 100, 200) for every policy.
- Metrics per policy and seed: marginal mean W1 (val primary, test
  confirmation), joint fidelity (signed-flow ACF error at lags 1-5
  plus cross-feature correlation error, averaged on one scale, both
  parts reported separately), held-out discriminator AUC (fit on val
  only, scored on test only, 30 epochs, identical protocol for every
  policy and seed), TSTR flow-direction accuracy.
- Rankings compared with Kendall tau-b; Pareto domination over
  (val W1, val joint error, test AUC); paired blocked CIs at block
  lengths 1/5/10/20. The test split is confirmation only and never
  picks a ranking. All nulls preserved.
- No size effect is claimed from the unmatched historical
  221767/480775 variants: those studies used different split
  populations and capacities, and historical weight tensors were
  never saved, so old figures are not a controlled architecture
  experiment. Only the within-study factorial is a controlled size
  contrast.

### Measured results

Harness: `examples/ranking_stability.py`, record
`examples/ranking_stability_results.json`. Run time 50.9 min, CPU
with 2 torch threads, peak RSS 545.9 MB. Final train losses: DDPM
0.1801, DDPM with EMA 0.1782, small DDPM 0.2002, small DDPM with
EMA 0.1981, VAE 0.5850. All preregistered model x seed cells
present (0 missing); discriminator fit/test session overlap empty;
4 test sessions, so intervals are across-session evidence.

| Policy | Val mean W1 | Val joint error | Test mean W1 | Test AUC | TSTR |
| --- | --- | --- | --- | --- | --- |
| DDPM | 0.1501 | 0.0362 | 0.1376 | 0.709 | 0.515 |
| DDPM with EMA | 0.1064 | 0.0215 | 0.0832 | 0.563 | 0.502 |
| Small DDPM | 0.1420 | 0.0342 | 0.1289 | 0.620 | 0.498 |
| Small DDPM with EMA | 0.1331 | 0.0311 | 0.1118 | 0.609 | 0.523 |
| DDIM 10 NFE | 0.1516 | 0.0269 | 0.1408 | 0.741 | 0.519 |
| VAE | 0.4354 | 0.1791 | 0.4366 | 0.977 | 0.520 |
| Chunk-copy (reference) | 0.0529 | 0.0159 | 0.0519 | 0.508 | 0.533 |

Joint error parts (val, lag ACF error / correlation error): DDPM
0.0064 / 0.0660; DDPM with EMA 0.0052 / 0.0378; small DDPM 0.0085 /
0.0599; small DDPM with EMA 0.0122 / 0.0499; DDIM 10 NFE 0.0091 /
0.0447; VAE 0.1344 / 0.2239; chunk-copy 0.0082 / 0.0235.

Rankings (1 is best):

| Metric | Order |
| --- | --- |
| Val W1 | chunk-copy, DDPM with EMA, small DDPM with EMA, small DDPM, DDPM, DDIM 10, VAE |
| Val joint error | chunk-copy, DDPM with EMA, DDIM 10, small DDPM with EMA, small DDPM, DDPM, VAE |
| Test AUC | chunk-copy, DDPM with EMA, small DDPM with EMA, small DDPM, DDPM, DDIM 10, VAE |
| TSTR accuracy | chunk-copy, small DDPM with EMA, VAE, DDIM 10, DDPM, DDPM with EMA, small DDPM |

Kendall tau-b against the W1 ranking (trained policies only):
W1 vs joint 0.6, W1 vs AUC 1.0, W1 vs TSTR -0.33, joint vs AUC
0.6. With the chunk-copy reference included: 0.714, 1.0, 0.048,
0.714.

Paired blocked CIs (block length 20): EMA against raw DDPM is
better on marginal W1 (test paired difference +0.0544, CI [0.0392,
0.0649]) and on joint error (+0.0148, CI [0.0074, 0.0195]), with
non-overlapping discriminator intervals (EMA 0.560, CI [0.517,
0.597] against raw 0.712, CI [0.675, 0.744]). DDIM 10 against the
ancestral sampler is a tie on marginal W1 (+0.0032, CI [-0.0079,
0.0161]), better on joint error (-0.0093, CI [-0.0148, -0.0033]),
and not established on AUC (0.747, CI [0.719, 0.779] against
0.712, CI [0.675, 0.744]). Small against base is a tie on marginal
W1 (+0.0088, CI [-0.018, 0.0441]) and on joint error (+0.002, CI
[-0.0025, 0.0063]). TSTR paired differences include zero for
every pair. The Pareto set over val W1, val joint error, and test
AUC is the chunk-copy reference alone; among trained models the
base EMA cell dominates the other five on all three axes.

### Interpretation

Marginal ranking is a reliable guide to discriminator and joint
quality on this corpus at this training length: the W1 and AUC
rankings agree exactly, and the W1 and joint rankings agree
moderately. The instability is in downstream utility: the TSTR
ranking disagrees with the marginal ranking, and every TSTR
accuracy sits at the majority level, so flow-direction utility
measures something the marginal ranking does not see. The DDIM 10
against ancestral pair shows the same point at small scale: the
pair is tied on W1, DDIM 10 wins on joint error, and the
discriminator leans the other way. EMA is the clearest win among
the trained models, leading on all three primary axes. The size
contrast is mixed and mostly null within this study. No policy is
promoted to a default and no ranking is presented as a verdict on
its own.

### Protocol checks and limitations

- Chunk-copy reproduces the P08 numbers to four decimals on every
  metric (val W1 0.0529, test W1 0.0519, AUC 0.508), confirming
  the scoring protocol.
- This run's ancestral DDPM scores val W1 0.1501 against P08's
  0.2630 at near-identical final train loss (0.1801 against
  0.1774). The trained weights differ only by RNG stream (run ID),
  so marginal W1 varies substantially across training streams at
  6 epochs. This is recorded as a single-training-seed limitation,
  not a correction of P08, and it is the reason no narrow claim
  rests on one trained model here.
- Limitations: single training seed (0); 1,000-window subsamples,
  not the full splits; test metrics are confirmation only; CPU
  only.

## P10 part 1: Model-family challenge contract and DDPM baseline capture

Question: is the DDPM the best model for seven small tabular
time-series dimensions, or does a cheaper family match it? Part 1
captures the contract the comparison must run under and the DDPM
baseline it must beat; no challenger is implemented here and none is
labelled superior.

### Frozen contract

- Challenge corpus: expanded 28-session corpus, train 18,468 /
  val 5,712 / test 5,712 windows, session-disjoint, split-manifest
  SHA b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd74fb861e1446ae581e,
  fixed 1,000-window strided val/test subsamples.
- Same physical representation (seven features, 60-second windows,
  train-standardized) and same conditioning information (2-dim
  regime lookback, past only) for every family.
- Exactly the same final discriminators: DiscMLP (128, 64), fit on
  val, scored on test, 30 epochs, protocol seed 0; same TSTR
  protocol, 300 epochs, protocol seed 0.
- Equivalent tuning entitlement: same optimizer-step budget and the
  same parameter-count view, recorded per family.
- Independent generation seeds: the same seed set (0, 100, 200) for
  every family so the comparison is paired; draws stay independent
  across families through per-family sampling streams derived via
  `derive_seed(run_id, "sampling")`.
- Metric set, one Pareto table: marginal mean W1, joint error (lag
  ACF error at lags 1-5 plus cross-feature correlation error, both
  parts reported), tail coverage (generated values outside the real
  1st/99th percentile band, next to the real-side reference rate),
  held-out discriminator AUC, TSTR accuracy, NFE, sampling latency,
  peak memory, parameter count. Dependence-aware intervals at block
  lengths 1/5/10/20, as in C03.
- Challenger order: compact causal probabilistic autoregressor
  first, conditional diffusion-forecasting variant next, flow
  matching only if justified under the same train data, parameter
  and compute envelopes. LOB-Bench scores do not compare: different
  message-based task, not 1-second feature windows.
- Full contract: `docs/model-family-challenge-contract.md`;
  contract as code: `flowforge/family_contract.py`.

### Captured baseline

Harness: `examples/family_baseline.py`, record
`examples/family_baseline_results.json`. Nothing trained, nothing
sampled; the script recomputes from committed and already-recorded
artifacts only.

- Expanded corpus (FF-R08 record): DDPM test mean W1 0.1376, val
  joint error 0.0362 (lag ACF 0.0064, correlation 0.0660),
  discriminator AUC 0.7091, TSTR 0.5153, 221,767 parameters, 100
  NFE per window, 29.90 seconds per 1,000 windows. No per-policy
  peak memory is claimed: FF-R08 recorded a per-run allocation
  delta, not peak RSS, and the run peak RSS of 545.9 MB covers all
  seven policies.
- Committed demo bundle (64 windows, build seed 7, recomputed
  here): mean W1 0.2973, discriminator AUC 0.6690 (recomputed from
  the stored float32 scores to 1.2e-8), joint error 0.0534, tail
  outside rate 0.1592 against a real-side reference of 0.02. The
  full-split demo numbers are mean W1 0.2763 and AUC 0.6690 on 198
  test windows; the bundle is a 64-window subsample, so its mean W1
  differs by subsample noise. TSTR and latency are null on the
  bundle (no val split, no timing).
- The Pareto table carries the DDPM row measured and the three
  challenger rows pending. No ranking is computed and no family is
  promoted.

### Open items for part 2

- The expanded-corpus NPZ is gitignored and not present in this
  checkout; part 2 must locate or re-record it before running the
  comparison rather than substituting a different corpus.
- Tail coverage has no FF-R08 value (the study did not measure it),
  so the DDPM row carries the demo-bundle rate only until part 2
  measures it on the challenge corpus.
