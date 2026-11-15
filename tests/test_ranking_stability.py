"""Stage 3 FF-R08: marginal-W1 versus joint fidelity ranking stability.

Fixture-first tests for the ranking-stability analysis in
``examples/ranking_stability.py``. Every numeric expectation below is
a hand-computed known answer or an independent NumPy oracle written
here in the test; none of it shares code with the module under test.

The question under test: does the ranking of generative methods by
marginal W1 agree with the ranking by joint/conditional lag quality
and by a powered held-out discriminator? The fixtures pin the three
moving parts of that question separately:

- Kendall tau-b between two rankings (hand-computed values, ties
  included), so a ranking-agreement claim cannot come from a broken
  tau.
- the joint-fidelity score (lag ACF error plus cross-feature
  correlation error), with sharp invariances: an exact copy scores
  zero, a pure mean shift is invisible to it, a time shuffle destroys
  the lag part while leaving the correlation part at zero.
- Pareto domination over (W1, joint error, AUC), with hand-computed
  dominated and trade-off sets.
- the preregistration guards: every model x seed cell present, and
  zero overlap between the discriminator fit sessions and the test
  sessions.
"""

import numpy as np
import pytest

from examples.ranking_stability import (
    check_cells_present,
    check_no_fit_test_overlap,
    joint_fidelity,
    kendall_tau_b,
    paired_joint_diff_ci,
    pareto_frontier_rows,
    rank_scores,
)

FLOW_J = 5  # signed_flow index in FEATURE_NAMES
LAGS = (1, 2, 3, 4, 5)


def _ar1_windows(n=48, window=60, n_features=7, phi=0.8, seed=0):
    """Synthetic windows with strong lag-1 structure in signed flow."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, window, n_features)).astype(np.float64)
    for i in range(n):
        s = rng.normal(size=window)
        for t in range(1, window):
            s[t] += phi * s[t - 1]
        x[i, :, FLOW_J] = s / np.std(s)
    return x


def _oracle_acf(series, lag):
    """Independent ACF: demeaned dot product over the lag, by hand."""
    s = np.asarray(series, dtype=np.float64)
    s = s - s.mean()
    denom = float(np.dot(s, s))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(s[: len(s) - lag], s[lag:]) / denom)


def _oracle_lag_acf_error(real, gen, lags=LAGS):
    """Mean |gen ACF - real ACF| over lags, from the oracle above."""
    errs = []
    for lag in lags:
        r = np.mean([_oracle_acf(w[:, FLOW_J], lag) for w in real])
        g = np.mean([_oracle_acf(w[:, FLOW_J], lag) for w in gen])
        errs.append(abs(g - r))
    return float(np.mean(errs))


def _oracle_corr_error(real, gen):
    """Mean absolute off-diagonal correlation difference, by hand."""
    rf = np.asarray(real, dtype=np.float64).reshape(-1, real.shape[2])
    gf = np.asarray(gen, dtype=np.float64).reshape(-1, gen.shape[2])
    cr = np.corrcoef(rf, rowvar=False)
    cg = np.corrcoef(gf, rowvar=False)
    mask = ~np.eye(cr.shape[0], dtype=bool)
    return float(np.mean(np.abs(cr[mask] - cg[mask])))


# ------------------------------------------------------------ Kendall tau-b

def test_stage3_ff_r08_kendall_identical_is_one():
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_stage3_ff_r08_kendall_reversed_is_minus_one():
    assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_stage3_ff_r08_kendall_single_swap_is_one_third():
    # pairs: (1,2) concordant, (1,3) concordant, (2,3) discordant,
    # so tau = (2 - 1) / 3 = 1/3, computed by hand.
    assert kendall_tau_b([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)


def test_stage3_ff_r08_kendall_tie_case_hand_computed():
    # pairs: (0,1) tied in x only, (0,2) concordant, (1,2) concordant.
    # C=2, D=0, Tx=1, Ty=0, so tau = 2 / sqrt(3 * 2) = 2 / sqrt(6).
    assert kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(
        2 / np.sqrt(6))


def test_stage3_ff_r08_kendall_all_tied_is_nan():
    assert np.isnan(kendall_tau_b([1, 1, 1], [1, 2, 3]))
    assert np.isnan(kendall_tau_b([], []))


def test_stage3_ff_r08_kendall_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        kendall_tau_b([1, 2, 3], [1, 2])


def test_stage3_ff_r08_kendall_rank_dicts_agree():
    # the same ordering expressed as rank dicts: perfect agreement.
    a = {"ddpm": 1.0, "vae": 2.0, "chunk_copy": 3.0}
    b = {"ddpm": 1.0, "vae": 2.0, "chunk_copy": 3.0}
    keys = sorted(a)
    assert kendall_tau_b([a[k] for k in keys],
                         [b[k] for k in keys]) == pytest.approx(1.0)


# ------------------------------------------------------------ average ranks

def test_stage3_ff_r08_rank_scores_lower_is_better():
    ranks = rank_scores({"a": 0.5, "b": 0.2, "c": 0.9})
    assert ranks == {"b": 1.0, "a": 2.0, "c": 3.0}


def test_stage3_ff_r08_rank_scores_ties_share_average_rank():
    ranks = rank_scores({"a": 1.0, "b": 1.0, "c": 3.0})
    assert ranks["a"] == pytest.approx(1.5)
    assert ranks["b"] == pytest.approx(1.5)
    assert ranks["c"] == pytest.approx(3.0)


def test_stage3_ff_r08_rank_scores_higher_is_better():
    ranks = rank_scores({"a": 0.5, "b": 0.2}, lower_is_better=False)
    assert ranks == {"a": 1.0, "b": 2.0}


# ------------------------------------------------------- joint fidelity

def test_stage3_ff_r08_joint_exact_copy_scores_zero():
    real = _ar1_windows()
    out = joint_fidelity(real, real.copy())
    assert out["lag_acf_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["corr_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["joint_error"] == pytest.approx(0.0, abs=1e-12)


def test_stage3_ff_r08_joint_mean_shift_is_invisible():
    # a pure mean shift changes the marginals but not the lag
    # structure or the correlations: the joint score must stay zero.
    real = _ar1_windows()
    out = joint_fidelity(real, real + 0.5)
    assert out["lag_acf_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["corr_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["joint_error"] == pytest.approx(0.0, abs=1e-12)


def test_stage3_ff_r08_joint_time_shuffle_matches_oracle():
    real = _ar1_windows()
    rng = np.random.default_rng(7)
    perm = rng.permutation(real.shape[1])
    gen = real[:, perm, :]
    out = joint_fidelity(real, gen)
    # shuffling time destroys the lag structure but preserves the
    # flattened per-feature values, so the correlation part stays 0
    assert out["corr_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["lag_acf_error"] == pytest.approx(
        _oracle_lag_acf_error(real, gen), abs=1e-9)
    assert out["lag_acf_error"] > 0.2
    assert out["joint_error"] == pytest.approx(
        (out["lag_acf_error"] + out["corr_error"]) / 2)


def test_stage3_ff_r08_joint_feature_swap_matches_oracle():
    real = _ar1_windows()
    gen = real.copy()
    gen[:, :, [0, 1]] = gen[:, :, [1, 0]]
    out = joint_fidelity(real, gen)
    # swapping two non-flow features leaves the flow lags untouched
    assert out["lag_acf_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["corr_error"] == pytest.approx(
        _oracle_corr_error(real, gen), abs=1e-9)
    assert out["corr_error"] > 0.0
    assert out["joint_error"] == pytest.approx(
        (out["lag_acf_error"] + out["corr_error"]) / 2)


# ------------------------------------------------------------ Pareto

def _row(method, w1, joint, auc):
    return {"method": method, "val_mean_w1": w1,
            "val_joint_error": joint, "auc_test_mean": auc}


def test_stage3_ff_r08_pareto_single_dominator():
    rows = [_row("a", 1, 1, 1), _row("b", 2, 2, 2),
            _row("c", 1, 3, 3), _row("d", 3, 1, 3)]
    assert pareto_frontier_rows(rows) == {"a"}


def test_stage3_ff_r08_pareto_tradeoff_keeps_all():
    rows = [_row("a", 1, 3, 2), _row("b", 3, 1, 2),
            _row("c", 2, 2, 2)]
    assert pareto_frontier_rows(rows) == {"a", "b", "c"}


def test_stage3_ff_r08_pareto_single_and_empty():
    assert pareto_frontier_rows([_row("a", 1, 1, 1)]) == {"a"}
    assert pareto_frontier_rows([]) == set()


# ------------------------------------------------- preregistration guards

def test_stage3_ff_r08_cells_complete_passes():
    cells = {("ddpm", 0), ("ddpm", 100), ("vae", 0), ("vae", 100)}
    assert check_cells_present(cells, ["ddpm", "vae"], [0, 100]) == 0


def test_stage3_ff_r08_cells_missing_names_the_gap():
    cells = {("ddpm", 0), ("vae", 0), ("vae", 100)}
    with pytest.raises(ValueError, match=r"ddpm.*100|100.*ddpm"):
        check_cells_present(cells, ["ddpm", "vae"], [0, 100])


def test_stage3_ff_r08_cells_unexpected_is_refused():
    cells = {("ddpm", 0), ("vae", 0), ("vae", 100), ("ddpm", 100),
             ("surprise", 0)}
    with pytest.raises(ValueError, match="surprise"):
        check_cells_present(cells, ["ddpm", "vae"], [0, 100])


def test_stage3_ff_r08_fit_test_overlap_refused():
    with pytest.raises(ValueError, match="21"):
        check_no_fit_test_overlap([20, 21], [21, 24])
    assert check_no_fit_test_overlap([20, 21], [24, 25]) is True


# ------------------------------------------------- paired joint CIs

def test_stage3_ff_r08_joint_ci_identical_gens_are_zero():
    real = _ar1_windows(n=32)
    rows = paired_joint_diff_ci(
        real, {"a": [real.copy()], "b": [real.copy()]},
        block_lengths=(1,), n_boot=20, seeds=(0,))
    assert len(rows) == 1
    assert rows[0]["diff"] == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["ci"] == [pytest.approx(0.0, abs=1e-12),
                             pytest.approx(0.0, abs=1e-12)]


def test_stage3_ff_r08_joint_ci_shuffled_gen_is_worse():
    real = _ar1_windows(n=120, phi=0.8, seed=3)
    rng = np.random.default_rng(7)
    perm = rng.permutation(real.shape[1])
    shuffled = real[:, perm, :]
    rows = paired_joint_diff_ci(
        real, {"copy": [real.copy()], "shuffled": [shuffled]},
        block_lengths=(1,), n_boot=100, seeds=(0,))
    assert len(rows) == 1
    assert rows[0]["methods"] == ["copy", "shuffled"]
    assert rows[0]["diff"] < -0.2
    assert rows[0]["ci"][1] < 0.0


# ------------------------------------------------------- end-to-end smoke

def test_stage3_ff_r08_harness_smoke_on_tiny_splits():
    """The full harness runs end to end on tiny synthetic splits.

    This is a wiring check, not evidence: three policies, two
    generation seeds, one epoch, ten diffusion steps. It proves the
    preregistered cell grid, the ranking tables, and the Kendall
    agreement are all produced, with zero missing cells.
    """
    import torch

    from examples.ranking_stability import run_ranking_stability

    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    n_train, n_eval = 64, 16
    x_train = torch.from_numpy(
        rng.normal(size=(n_train, 60, 7)).astype(np.float32))
    r_train = torch.from_numpy(
        rng.normal(size=(n_train, 2)).astype(np.float32))
    splits = {
        "train": (x_train, r_train),
        "val": (torch.from_numpy(
            rng.normal(size=(n_eval, 60, 7)).astype(np.float32)),
            torch.from_numpy(
                rng.normal(size=(n_eval, 2)).astype(np.float32))),
        "test": (torch.from_numpy(
            rng.normal(size=(n_eval, 60, 7)).astype(np.float32)),
            torch.from_numpy(
                rng.normal(size=(n_eval, 2)).astype(np.float32))),
    }
    train_meta = {
        "x": x_train.numpy(), "regime": r_train.numpy(),
        "session_ids": np.zeros(n_train, dtype=int),
        "segment_ids": np.zeros(n_train, dtype=int),
        "test_session_ids": np.array([24] * n_eval),
        "fit_session_ids": [20, 21],
    }
    record = run_ranking_stability(
        splits, train_meta, policies=["ddpm", "vae", "chunk_copy"],
        gen_seeds=(0, 1), train_epochs=1, vae_epochs=1, t_steps=10,
        ch=16, n_eval=n_eval, disc_epochs=2, tstr_epochs=10,
        n_boot=5, block_lengths=(1, 5), run_id="ff-r08-smoke")
    assert record["n_missing_cells"] == 0
    assert sorted(record["policies"]) == [
        "chunk_copy", "ddpm", "vae"]
    assert set(record["rankings"]["w1"]) == {
        "chunk_copy", "ddpm", "vae"}
    assert set(record["rankings"]["joint"]) == {
        "chunk_copy", "ddpm", "vae"}
    assert set(record["rankings"]["auc"]) == {
        "chunk_copy", "ddpm", "vae"}
    for tau in record["kendall_tau"].values():
        # a tau is None only when one ranking is fully tied
        assert tau is None or -1.0 <= tau <= 1.0
    assert len(record["pareto"]) >= 1
    assert record["fit_test_overlap"] == []
