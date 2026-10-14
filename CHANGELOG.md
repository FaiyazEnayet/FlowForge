# Changelog

## [Unreleased]

### Fixed
- Conditioning leakage in the regime features (`flowforge/data.py`). The volatility lookback was built as `mid[start - LOOKBACK:start + 1]`, which included `mid[start]`, the first second of the target window the model generates, contradicting the documented strictly-past conditioning. It now uses `mid[start - LOOKBACK:start]`. Added a regression test pinning temporal invariance: regime features for a window are identical when data at or after the window start is perturbed.
- Reran both models on the corrected features and updated the README with the rerun numbers. The headline metrics barely moved: DDPM mean W1 0.276 to 0.277, discriminator AUC 0.669 to 0.669; VAE mean W1 0.536 to 0.527, discriminator AUC 0.937 to 0.956. The conclusions are unchanged.
