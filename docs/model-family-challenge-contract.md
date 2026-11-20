# Model-family challenge contract (P10, part 1 of 2)

Source: FlowForge research roadmap (P10). The DDPM may not be the best
model for seven small tabular time-series dimensions. A strong
lightweight causal probabilistic autoregressor, a conditional
diffusion-forecasting variant, or flow matching might beat it at a
fraction of sampling cost.

This document is the contract part 2 must run under. It is written
before any challenger exists, so no family is labelled superior in
advance and none can be: the Pareto table in
`examples/family_baseline_results.json` carries the DDPM baseline row
and challenger rows marked pending.

## The question

Seven standardized book-state features over 60-second windows, seven
small tabular time-series dimensions. The DDPM is the incumbent. The
challenge asks whether a cheaper family matches or beats it under
identical conditions, and at what sampling cost. The answer is a
Pareto table, not a winner announcement: fidelity against NFE,
latency, and memory, one row per family.

## Fairness rules

Every family runs under the same rules, enforced in code by
`flowforge/family_contract.py` (`FamilyChallengeContract`):

- Same splits. The challenge corpus is the expanded 28-session
  corpus measured by FF-R08: train 18,468 / val 5,712 / test 5,712
  windows, session-disjoint, split-manifest SHA
  `b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd74fb861e1446ae581e`,
  fixed 1,000-window strided val/test subsamples, identical across
  every family.
- Same physical representation. Seven features
  (`log_ret`, `log_spread_bps`, `log_bid_sz`, `log_ask_sz`,
  `imbalance`, `signed_flow`, `log_trade_count`), 60-second windows,
  standardized with train-split statistics only.
- Same conditioning information. The 2-dim regime lookback (realized
  volatility and mean spread over the 60 seconds before the window
  start, past only, never the window itself) and, where a family
  uses it, the full 60-second past context with its availability
  mask. One exception is recorded in part 2: the
  history-conditioned DDPM sees the full 60-second past-context
  tensor in addition to the regime aggregates, so its row is not
  conditioning-matched with the other families, which see the
  regime aggregates only.
- Exactly the same final discriminators. One held-out discriminator
  protocol for every family and seed: `DiscMLP` (128, 64 hidden
  units), fit on val, scored on test, 30 epochs, protocol seed 0.
  The same TSTR protocol: logistic flow-direction model, 300 epochs,
  protocol seed 0, trained on generated val windows and tested on
  real test windows.
- Equivalent tuning entitlement. Every family gets the same
  optimizer-step budget, and the same parameter-count view is
  recorded alongside it. The entitlement is recorded per family, not
  assumed equal after the fact.
- Independent generation seeds. Every family generates under the
  same seed set (0, 100, 200), so the comparison is paired seed by
  seed. The draws stay independent across families because each
  family's sampling stream is derived per family via
  `derive_seed(run_id, "sampling")`, which never shares draws with
  another family's sampling stream or with the training and
  discriminator streams of the same run, even though the integer
  seed set is shared with them by design.

## Metric set (one Pareto table)

Per family and generation seed: marginal mean W1 (val primary, test
confirmation), joint error with its two parts reported separately
(lag ACF error at lags 1-5, cross-feature correlation error), tail
coverage (fraction of generated values outside the real 1st/99th
percentile band per feature, next to the real-side reference rate),
held-out discriminator AUC, TSTR flow-direction accuracy, NFE per
window, sampling latency (seconds per 1,000 windows), peak memory,
parameter count. Uncertainty is dependence-aware, as in C03: paired
chronological block resamples shared across methods for W1 and joint
differences, blocked intervals for AUC, paired blocks of the shared
test windows for TSTR differences, at block lengths 1/5/10/20.

## Challenger order

Smallest fair challenger first, from the source: a compact
Gaussian-mixture / normalizing-flow / GRU autoregressive conditional
model with causal history, under the same train data, parameter and
compute envelopes. Only then a conditional diffusion-forecasting
variant. Flow matching only if justified under the same envelopes.
The existing `ConditionalARBaseline` in `flowforge/baselines.py` is
a candidate starting point for the autoregressive challenger, not a
shipped family: part 2 decides whether it meets the contract or a
fresh implementation is needed.

## What does not compare

LOB-Bench (ICML 2025) reports strong autoregressive results on a
different message-based LOB task. Those scores cannot be compared
numerically with FlowForge's 1-second feature-window metrics. The
challenge is judged on FlowForge's own corpus and protocol only.

## Baseline (captured, part 1)

Recorded in `examples/family_baseline_results.json`, captured by
`examples/family_baseline.py` (no training, no sampling; it
recomputes from committed and already-recorded artifacts only):

- Expanded corpus (FF-R08 record, test split, generation seeds
  0/100/200): DDPM mean W1 0.1376, joint error 0.0362 (lag ACF
  0.0064, correlation 0.0660), discriminator AUC 0.7091, TSTR
  accuracy 0.5153, 221,767 parameters, 100 NFE per window, 29.90
  seconds per 1,000 windows. Peak memory is not claimed per policy:
  FF-R08 recorded a per-run allocation delta, not peak RSS, and the
  run peak RSS of 545.9 MB covers all seven policies.
- Committed demo bundle (64 windows, build seed 7, recomputed here):
  mean W1 0.2973, discriminator AUC 0.6690, joint error 0.0534,
  tail outside rate 0.1592 against a real-side reference of 0.02.
  The full-split demo numbers in `demo/metrics.json` are mean W1
  0.2763 and AUC 0.6690 on 198 test windows; the bundle is a
  64-window subsample, so its mean W1 differs by subsample noise.
  TSTR and sampling latency cannot be measured from the bundle (no
  val split, no timing) and are recorded null with the reason.

## Corpus availability (open item for part 2)

The expanded 28-session corpus NPZ is gitignored and not present in
this checkout; its numbers above are the FF-R08 record, cited with
the manifest SHA, not a recomputation. Part 2 must locate or
re-record the corpus before running the comparison rather than
substituting a different corpus. The committed cross-check corpus is
the 8-chunk dataset in `data/` (manifest SHA
`eeca9db2c52c0a9d4ba6c05d79db85c226ec04a62e9522357433a26aa7e2c1d4`).

## Part 2 outcome

Part 2 built the challengers fresh rather than reusing the existing
`ConditionalARBaseline`, which is a linear ridge VAR comparator,
not a probabilistic neural autoregressor: a compact GRU with a
Gaussian-mixture next-step head (`flowforge/autoregressive.py`,
220,476 parameters, 60 sequential steps per window), a
history-conditioned DDPM (`BookDDPM` with `use_history=True`,
255,239 parameters, 100 NFE), and rectified flow matching on the
DDPM denoiser backbone (`flowforge/flow_matching.py`, 221,767
parameters, 20 Euler steps). The comparison ran in
`examples/compare_families.py` and is recorded in
`examples/compare_families_results.json`. Seed means over
generation seeds 0/100/200 on the 1,000-window strided test
subsample:

| Method | Test W1 | Val joint | AUC | TSTR | NFE | s / 1k | Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DDPM (retrained) | 0.138 | 0.024 | 0.734 | 0.476 | 100 | 29.8 | 221,767 |
| Autoregressive | 0.192 | 0.078 | 0.871 | 0.482 | 60 | 1.5 | 220,476 |
| Conditional diffusion forecast | 0.134 | 0.027 | 0.637 | 0.524 | 100 | 29.8 | 255,239 |
| Flow matching | 0.126 | 0.025 | 0.702 | 0.470 | 20 | 5.9 | 221,767 |
| VAE (reference) | 0.407 | 0.228 | 0.980 | 0.505 | 1 | 0.01 | 288,708 |
| Chunk copy (reference) | 0.052 | 0.016 | 0.508 | 0.533 | 0 | 0.16 | 0 |

The paired block-20 W1 intervals carry the comparison: flow
matching minus DDPM is +0.012 with an interval of [-0.016, 0.035],
history-conditioned DDPM minus DDPM is -0.004 with [-0.020,
0.011], so neither shows a measured W1 difference from the DDPM;
the autoregressor minus DDPM is +0.054 with [0.041, 0.073], a
measured higher W1, and chunk copy minus DDPM is -0.086 with
[-0.121, -0.036]. No TSTR difference survives the block-20
intervals. The retrained DDPM cross-checks against the recorded
FF-R08 row (test W1 0.138 against 0.138, AUC 0.734 against 0.709,
TSTR 0.476 against 0.515). Every trained family overshoots the
tails (outside rate 0.14 to 0.19 against the real-side reference
of 0.02); chunk copy sits at 0.026. The run was interrupted once
by a runtime restart after the DDPM, VAE, and autoregressive
checkpoints were saved, and resumed from those checkpoints with
no method retrained.

## Status

Part 2 of 2 complete: the families are implemented, the comparison
ran under this contract, and the Pareto table is filled. The
challenge question stays open as a table, not a verdict: the
cheaper families trade sampling cost against fidelity and
discrimination on the measures above, and no family is labelled
superior.
