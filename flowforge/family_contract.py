"""P10 model-family challenge: the fairness contract, as code.

Source: FlowForge research roadmap (P10): the DDPM may not be the best
model for seven small tabular time-series dimensions. A strong
lightweight causal probabilistic autoregressor, a conditional
diffusion-forecasting variant, or flow matching might beat it at a
fraction of sampling cost. LOB-Bench (ICML 2025) reports strong
autoregressive results on a DIFFERENT message-based LOB task; those
scores cannot be compared numerically with FlowForge's 1-second
feature-window metrics.

Part 1 (stage3 P10#1/2) captures this contract, the DDPM baseline,
and the fixture and independent oracle. No model family is implemented
here and none is labelled superior: the Pareto table carries the DDPM
baseline row and challenger rows marked pending until part 2 fills
them. The challenge corpus is the expanded 28-session corpus measured
by FF-R08; its recorded NPZ is not committed, so part 2 must locate
or re-record it before running the comparison rather than
substituting a different corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import FEATURE_NAMES, LOOKBACK, N_FEATURES, WINDOW
from .evaluate import acf
from .models import derive_seed

CONTRACT_SCHEMA = "flowforge-family-challenge/1"

# The fairness rules, pinned to the values the repo already measures
# under. The discriminator protocol and generation seeds are the FF-R08
# values, so the challenge reuses the exact protocol that produced the
# baseline row instead of inventing a new one.
TRAIN_SEED = 0
GENERATION_SEEDS = (0, 100, 200)
DISC_PROTOCOL_SEED = 0
DISC_EPOCHS = 30
TSTR_EPOCHS = 300
BLOCK_LENGTHS = (1, 5, 10, 20)
N_BOOT = 100
LAGS = (1, 2, 3, 4, 5)

# Expanded 28-session challenge corpus, as measured by FF-R08
# (examples/ranking_stability_results.json). The recorded NPZ is
# gitignored and not present in this checkout.
CHALLENGE_SPLITS = {
    "n_train": 18468,
    "n_val": 5712,
    "n_test": 5712,
    "n_sessions": 28,
    "manifest_sha256": (
        "b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd74fb861e1446ae581e"),
    "session_disjoint": True,
    "val_subsample": 1000,
    "test_subsample": 1000,
}

# Committed cross-check corpus: the 8-chunk dataset card and manifest
# shipped in data/ (the demo bundle is built from it).
COMMITTED_SPLITS = {
    "manifest_sha256": (
        "eeca9db2c52c0a9d4ba6c05d79db85c226ec04a62e9522357433a26aa7e2c1d4"),
    "n_chunks": 8,
    "n_seconds": 14377,
    "session_disjoint": True,
}

REQUIRED_COST_KEYS = ("sample_seconds", "sample_seconds_per_1k",
                      "peak_memory_mb", "nfe")

# One Pareto table, one row per family, no ranking column: the table
# reports the metrics and lets the reader judge the trade-off. A row
# that claims a verdict is refused, so no family can be labelled
# superior in advance.
PARETO_COLUMNS = (
    "method", "status", "mean_w1", "joint_error", "lag_acf_error",
    "corr_error", "tail_outside_rate", "auc_test", "tstr_acc_synth",
    "nfe_per_window", "sample_seconds_per_1k", "peak_memory_mb",
    "params",
)

CHALLENGER_METHODS = ("autoregressive",
                      "conditional_diffusion_forecast",
                      "flow_matching")

# Order of first attempt, from the source: the smallest fair
# challenger first (a compact Gaussian-mixture / normalizing-flow /
# GRU autoregressive conditional model with causal history), the
# conditional diffusion-forecasting variant next, and flow matching
# only if justified under the same train data, parameter and compute
# envelopes.
CHALLENGER_ORDER = ("autoregressive",
                    "conditional_diffusion_forecast",
                    "flow_matching")


# ------------------------------------------------------- metric definitions

def tail_coverage(real: np.ndarray, gen: np.ndarray,
                  low: float = 0.01, high: float = 0.99) -> dict:
    """Tail coverage: generated values outside the real 1/99 band.

    For each feature, the band is the real pooled 1st and 99th
    percentile. The reported rate is the fraction of generated values
    outside it, next to the real-side reference rate (the fraction of
    real values outside their own band, about 2% by construction). A
    generator that never reaches the tails scores near 0; one that
    overshoots them scores above the reference. Both directions are
    reported, so neither is silently rewarded.
    """
    rows = []
    real_rates = []
    gen_rates = []
    for j, name in enumerate(FEATURE_NAMES):
        r = np.asarray(real[:, :, j], dtype=np.float64).ravel()
        g = np.asarray(gen[:, :, j], dtype=np.float64).ravel()
        lo, hi = np.quantile(r, [low, high])
        real_out = float(((r < lo) | (r > hi)).mean())
        gen_out = float(((g < lo) | (g > hi)).mean())
        real_rates.append(real_out)
        gen_rates.append(gen_out)
        rows.append({"feature": name,
                     "low": float(lo), "high": float(hi),
                     "real_outside_rate": real_out,
                     "outside_rate": gen_out})
    return {"low_quantile": low, "high_quantile": high, "rows": rows,
            "real_outside_rate": float(np.mean(real_rates)),
            "mean_outside_rate": float(np.mean(gen_rates))}


def _window_acf(windows: np.ndarray, lags) -> np.ndarray:
    """Mean signed-flow ACF per window, one row per window."""
    j = FEATURE_NAMES.index("signed_flow")
    return np.array([[acf(w[:, j], lag) for lag in lags]
                     for w in windows], dtype=np.float64)


def _corr_matrix(windows: np.ndarray) -> np.ndarray:
    flat = np.asarray(windows, dtype=np.float64).reshape(
        -1, windows.shape[2])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(flat, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def conditional_scores(real: np.ndarray, gen: np.ndarray,
                       lags=LAGS) -> dict:
    """Joint and conditional lag quality, the FF-R08 definition.

    lag_acf_error is the mean over lags of the absolute difference
    between the generated and real mean signed-flow ACF; corr_error is
    the mean absolute off-diagonal difference between the generated
    and real cross-feature correlation matrices; joint_error is the
    mean of the two, the single joint score. Both parts are absolute
    errors in correlation units, so the mean keeps them on one scale.
    """
    real_acf = _window_acf(real, lags).mean(axis=0)
    gen_acf = _window_acf(gen, lags).mean(axis=0)
    lag_err = float(np.mean(np.abs(gen_acf - real_acf)))
    corr_r = _corr_matrix(real)
    corr_g = _corr_matrix(gen)
    mask = ~np.eye(corr_r.shape[0], dtype=bool)
    corr_err = float(np.mean(np.abs(corr_r[mask] - corr_g[mask])))
    return {
        "lag_acf_error": lag_err,
        "corr_error": corr_err,
        "joint_error": float((lag_err + corr_err) / 2),
        "real_acf": [float(v) for v in real_acf],
        "gen_acf": [float(v) for v in gen_acf],
        "lags": [int(l) for l in lags],
    }


# ---------------------------------------------------------------- contract

@dataclass(frozen=True)
class FamilyChallengeContract:
    """The fairness rules every family in the challenge must meet.

    Same splits, same physical representation, same conditioning
    information, exactly the same final discriminators, equivalent
    tuning entitlement, and independent generation seeds. A run that
    breaks any rule is refused here, before its numbers can enter
    the Pareto table.
    """
    schema: str = CONTRACT_SCHEMA
    splits: dict = field(default_factory=lambda: dict(CHALLENGE_SPLITS))
    representation: dict = field(default_factory=lambda: {
        "feature_names": list(FEATURE_NAMES),
        "n_features": N_FEATURES,
        "window_seconds": WINDOW,
        "regime_dims": 2,
        "past_context_lookback_seconds": LOOKBACK,
        "standardized_with": "train",
        "conditioning": "regime lookback ending at the window start "
                        "(past only, never the window itself)",
    })
    discriminator: dict = field(default_factory=lambda: {
        "architecture": "DiscMLP (128, 64 hidden units)",
        "epochs": DISC_EPOCHS,
        "seed": DISC_PROTOCOL_SEED,
        "fit_split": "val",
        "score_split": "test",
        "note": "identical protocol for every method and seed",
    })
    tstr: dict = field(default_factory=lambda: {
        "epochs": TSTR_EPOCHS,
        "seed": DISC_PROTOCOL_SEED,
        "task": "second-half signed-flow direction from first-half "
                "feature means",
    })
    tuning: dict = field(default_factory=lambda: {
        "basis": "equal_optimizer_steps",
        "equal_parameter_view": True,
        "note": "every family gets the same optimizer-step budget "
                "and the same parameter-count view; the entitlement "
                "is recorded per family, not assumed equal after "
                "the fact",
    })
    train_seed: int = TRAIN_SEED
    generation_seeds: tuple = GENERATION_SEEDS
    block_lengths: tuple = BLOCK_LENGTHS
    n_boot: int = N_BOOT

    def validate_run(self, run: dict, n_val: int, n_test: int) -> None:
        """Refuse a run that breaks the fairness rules."""
        for key in REQUIRED_COST_KEYS:
            if key not in run.get("cost", {}):
                raise ValueError(
                    f"run for {run.get('method')!r} lacks cost metadata "
                    f"{key!r}: no reported result may lack latency, "
                    f"NFE, or cost metadata")
        if run["gen_val"].shape[0] != n_val \
                or run["gen_test"].shape[0] != n_test:
            raise ValueError(
                f"run for {run['method']!r} does not match the shared "
                f"val/test window counts")
        seed = run["seed"]
        if seed not in self.generation_seeds:
            raise ValueError(
                f"generation seed {seed} is outside the contract seed "
                f"set {list(self.generation_seeds)}")
        # Note: the integer seed set is shared with the training and
        # discriminator seeds by design (FF-R08 used 0 for all three).
        # Independence comes from the streams, not the integers:
        # sampling draws use derive_seed(run_id, "sampling"), which
        # never shares draws with the training or discriminator
        # streams even when the integer coincides.

    def check_seed_independence(self, seed_by_method: dict) -> None:
        """Require the same seed set for every family, paired.

        Every family generates under exactly the contract seed set,
        so the comparison is paired seed by seed (the C03
        comparison_report rule). A family on a different set, or a
        subset, is refused: unpaired seeds would let a lucky draw
        pose as a better family. The draws stay independent across
        families because each family's sampling stream is derived
        per family via derive_seed(run_id, "sampling"); see
        sampling_stream_seed.
        """
        expected = sorted(self.generation_seeds)
        for method, seeds in seed_by_method.items():
            if sorted(seeds) != expected:
                raise ValueError(
                    f"method {method!r} uses generation seeds "
                    f"{sorted(seeds)}, not the contract set "
                    f"{expected}: all families must generate under "
                    f"the same seed set so the comparison is paired")

    def sampling_stream_seed(self, run_id: str) -> int:
        """The independent sampling-stream seed for one family run.

        derive_seed namespaces streams by name, so the sampling
        stream of one family never shares draws with another
        family's sampling stream or with the training and
        discriminator streams of the same run.
        """
        return derive_seed(run_id, "sampling")

    def pareto_table(self, rows: list[dict]) -> dict:
        """Build the one Pareto table from per-family metric rows.

        The baseline row carries measured numbers; challenger rows
        carry None until part 2 fills them. Any row that claims a
        ranking, verdict, or winner is refused: the table reports
        metrics and never labels a family superior.
        """
        out = []
        for row in rows:
            for banned in ("rank", "verdict", "winner", "best",
                           "superior"):
                if banned in {k.lower() for k in row}:
                    raise ValueError(
                        f"row for {row.get('method')!r} claims a "
                        f"{banned!r}: the Pareto table reports metrics "
                        f"and never labels a family superior")
            out.append({col: row.get(col) for col in PARETO_COLUMNS})
        return {"schema": self.schema, "columns": list(PARETO_COLUMNS),
                "rows": out,
                "note": "One row per family under the identical "
                        "protocol. The baseline row is measured; "
                        "challenger rows are pending until part 2 "
                        "fills them. The table reports metrics only."}


def default_contract() -> FamilyChallengeContract:
    """The challenge contract with the pinned FF-R08 values."""
    return FamilyChallengeContract()


def pending_challenger_row(method: str) -> dict:
    """A challenger row with every metric unfilled, status pending."""
    if method not in CHALLENGER_METHODS:
        raise ValueError(f"unknown challenger method {method!r}")
    row = {col: None for col in PARETO_COLUMNS}
    row["method"] = method
    row["status"] = "pending"
    return row
