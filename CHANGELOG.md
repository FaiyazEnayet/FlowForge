# Changelog

## [Unreleased]

### Fixed
- Conditioning leakage in the regime features (`flowforge/data.py`). The volatility lookback was built as `mid[start - LOOKBACK:start + 1]`, which included `mid[start]`, the first second of the target window the model generates, contradicting the documented strictly-past conditioning. It now uses `mid[start - LOOKBACK:start]`. Added a regression test pinning temporal invariance: regime features for a window are identical when data at or after the window start is perturbed.
