# Changelog

## [v1.2.0] - 2026-10-18
### Fixed
- Elapsed-time honesty in market windows (`flowforge/data.py`). Seconds without quotes were dropped, gaps up to five seconds were tolerated inside a segment, and windows counted rows: a "60-second" window could span more than 60 seconds, and the first log return after a gap silently covered multiple seconds. The pipeline now follows a documented regular-grid policy: a new segment starts at every missing second, and the bar after a gap is dropped so the 1s log return is only ever defined where the previous second is present. Every accepted 60-row window now spans exactly 59 seconds between endpoints. Added regression tests: a missing-second fixture (cross-gap return dropped, segments split at the gap) and a chunk-boundary return test. Note: the window counts and evaluation metrics recorded in the README were produced under the old policy and should be refreshed on a full rerun.

## [v1.1.0] - 2026-10-16

### Fixed
- Autocorrelation evaluation (`flowforge/evaluate.py`). `flow_acf_table` concatenated signed order flow across overlapping windows (60s windows, 10s stride) before computing autocorrelation. That duplicated most of the data, scrambled temporal order at window junctions, and centered everything on the grand mean. On level-shifted synthetic data the old method reported lag-1 autocorrelation of 0.97 where the true value was 0.3. It now computes autocorrelation per window and averages across windows, with 95% confidence intervals. Added regression tests pinning the per-window behavior and CI coverage.

### Added
- Train-on-synthetic-test-on-real (TSTR) check (`flowforge/evaluate.py`). A logistic regression predicts near-term flow direction from generated windows and is tested on real test windows, next to the same model trained on real windows, with a majority-class baseline for scale. It is reported separately from the real-vs-generated discriminator, which asks whether synthetic windows are indistinguishable rather than whether they are useful. First run: diffusion-trained scores 0.626 (the majority baseline), VAE-trained scores 0.409, so the VAE's invented temporal structure is actively misleading.

### Changed
- Reran the VAE baseline and the DDPM (seed 7, same config) and updated the README with the new numbers: DDPM mean W1 0.280 (was 0.277), discriminator AUC 0.741 (was 0.669). The AUC moved between identical reruns because torch CPU training is not bit-reproducible across runs; I verified the discriminator itself is deterministic on fixed inputs, so the movement is training noise. The W1 moved only in the third decimal.

### Fixed
- Conditioning leakage in the regime features (`flowforge/data.py`). The volatility lookback was built as `mid[start - LOOKBACK:start + 1]`, which included `mid[start]`, the first second of the target window the model generates, contradicting the documented strictly-past conditioning. It now uses `mid[start - LOOKBACK:start]`. Added a regression test pinning temporal invariance: regime features for a window are identical when data at or after the window start is perturbed.

### Changed
- Reran the VAE baseline and the DDPM on the leak-free dataset (seed 7, same config as before) and updated the README with the new numbers. They came out effectively unchanged: DDPM mean W1 0.277 with discriminator AUC 0.669, VAE 0.527 with 0.956.
