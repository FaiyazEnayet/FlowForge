"""Stage 3 P10 part 1: model-family challenge contract, baseline capture,
fixture and independent oracle.

Part 1 only: the source contract is captured as code, the DDPM baseline
is captured from the committed demo bundle and the recorded ranking
study, and every number is cross-checked by an independent oracle that
shares no code with the module under test. No model family is
implemented here and none is labelled superior: the Pareto table
carries the DDPM baseline row and challenger rows marked pending until
part 2 fills them.
"""

import importlib.util
import json
import os

import numpy as np
import pytest
from scipy.stats import mannwhitneyu, rankdata, wasserstein_distance

from flowforge.data import FEATURE_NAMES, N_FEATURES, WINDOW
from flowforge.evaluate import (auc_score, discriminator_auc,
                                marginal_table, wasserstein_1d)
from flowforge.family_contract import (
    CONTRACT_SCHEMA, DISC_EPOCHS, DISC_PROTOCOL_SEED, GENERATION_SEEDS,
    PARETO_COLUMNS, TRAIN_SEED, FamilyChallengeContract,
    conditional_scores, default_contract, tail_coverage)

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)


# ------------------------------------------------------- fixed fixture

def _fixture_windows(n=32, seed=11):
    """Deterministic synthetic windows: real and a shifted copy."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n, WINDOW, N_FEATURES)).astype(np.float64)
    gen = real + 0.25
    return real, gen


def _load_ranking_stability():
    path = os.path.join(REPO, "examples", "ranking_stability.py")
    spec = importlib.util.spec_from_file_location("ranking_stability",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_baseline_results():
    path = os.path.join(REPO, "examples",
                        "family_baseline_results.json")
    with open(path) as f:
        return json.load(f)


# ------------------------------------------- independent W1 oracle

def test_stage3_flowforge_10_part_1_w1_oracle_matches_repo():
    real, gen = _fixture_windows()
    for j, name in enumerate(FEATURE_NAMES):
        repo = wasserstein_1d(real[:, :, j].ravel(),
                              gen[:, :, j].ravel())
        oracle = wasserstein_distance(real[:, :, j].ravel(),
                                      gen[:, :, j].ravel())
        assert abs(repo - oracle) < 1e-9, name
    # identical distributions: both oracles read zero
    assert wasserstein_1d(real.ravel(), real.ravel()) < 1e-12
    assert wasserstein_distance(real.ravel(), real.ravel()) < 1e-12
    # a constant shift of 0.25 moves every pooled value by 0.25
    mt = marginal_table(real, gen)
    assert abs(mt["mean_w1"] - 0.25) < 1e-9
    # the shift is identical for every feature
    assert all(abs(r["w1"] - 0.25) < 1e-9 for r in mt["rows"])


def test_stage3_flowforge_10_part_1_w1_oracle_known_answers():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(wasserstein_distance(a, b) - 1.0) < 1e-12
    assert abs(wasserstein_1d(a, b) - 1.0) < 1e-12
    # disjoint point masses: the distance is the gap
    assert abs(wasserstein_distance([0.0, 0.0], [5.0, 5.0]) - 5.0) < 1e-12


# ------------------------------------------- independent AUC oracle

def _oracle_auc(y_true, y_score):
    """AUC from the Mann-Whitney U statistic, via scipy ranks."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    u = mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (len(pos) * len(neg)))


def test_stage3_flowforge_10_part_1_auc_oracle_matches_repo():
    rng = np.random.default_rng(3)
    y = np.concatenate([np.ones(40), np.zeros(40)])
    s = rng.standard_normal(80) + np.where(y == 1, 0.8, 0.0)
    assert abs(auc_score(y, s) - _oracle_auc(y, s)) < 1e-9
    # ties: scipy rankdata averages tied ranks, as the repo does
    s_tied = np.array([0.5, 0.5, 0.5, 0.9, 0.1, 0.1])
    y_tied = np.array([1, 1, 0, 1, 0, 0])
    assert abs(auc_score(y_tied, s_tied)
               - _oracle_auc(y_tied, s_tied)) < 1e-9
    # perfect separation and perfect inversion are exact
    assert abs(auc_score([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) - 1.0) < 1e-12
    assert abs(auc_score([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) - 0.0) < 1e-12
    # rankdata cross-check on the tied case pins the tie handling:
    # scores 0.1, 0.1 share rank 1.5; 0.5, 0.5, 0.5 share rank 4;
    # 0.9 is rank 6. Positives sit at indices 0, 1, 3: ranks 4+4+6
    ranks = rankdata(s_tied)
    assert abs(float(ranks[y_tied == 1].sum()) - 14.0) < 1e-12


def test_stage3_flowforge_10_part_1_stored_disc_auc_recomputes():
    d = np.load(os.path.join(REPO, "demo", "demo.npz"))
    y = np.concatenate([np.ones(len(d["disc_scores_real"])),
                        np.zeros(len(d["disc_scores_gen"]))])
    s = np.concatenate([d["disc_scores_real"], d["disc_scores_gen"]])
    assert abs(auc_score(y, s) - float(d["disc_auc"])) < 1e-7
    assert abs(_oracle_auc(y, s) - float(d["disc_auc"])) < 1e-7
    assert abs(float(d["disc_auc"]) - 0.6689878702163696) < 1e-9


# ------------------------------------------- tail coverage oracle

def test_stage3_flowforge_10_part_1_tail_coverage_known_answers():
    real = np.arange(100, dtype=np.float64).reshape(1, 100, 1)
    real = np.repeat(real, N_FEATURES, axis=2)
    # generated identical to real: outside rate equals the real
    # outside rate exactly, feature by feature
    tc = tail_coverage(real, real)
    assert abs(tc["mean_outside_rate"] - tc["real_outside_rate"]) < 1e-12
    assert abs(tc["real_outside_rate"] - 0.02) < 1e-9
    # generated shifted far right: everything lands outside the band
    gen = real + 1000.0
    tc = tail_coverage(real, gen)
    assert abs(tc["mean_outside_rate"] - 1.0) < 1e-12
    # generated at the real median: nothing lands outside the band
    gen = np.full_like(real, 49.5)
    tc = tail_coverage(real, gen)
    assert abs(tc["mean_outside_rate"] - 0.0) < 1e-12
    assert tc["low_quantile"] == 0.01 and tc["high_quantile"] == 0.99
    assert len(tc["rows"]) == N_FEATURES


def test_stage3_flowforge_10_part_1_tail_coverage_oracle_on_fixture():
    real, gen = _fixture_windows()
    tc = tail_coverage(real, gen)
    # hand oracle: count generated values outside the real 1/99 band
    for j, row in enumerate(tc["rows"]):
        lo, hi = np.quantile(real[:, :, j].ravel(), [0.01, 0.99])
        g = gen[:, :, j].ravel()
        expected = float(((g < lo) | (g > hi)).mean())
        assert abs(row["outside_rate"] - expected) < 1e-12
        assert row["feature"] == FEATURE_NAMES[j]


# ------------------------------------------- conditional scores oracle

def test_stage3_flowforge_10_part_1_conditional_scores_match_study():
    real, gen = _fixture_windows()
    rs = _load_ranking_stability()
    mine = conditional_scores(real, gen)
    theirs = rs.joint_fidelity(real, gen)
    for key in ("lag_acf_error", "corr_error", "joint_error"):
        assert abs(mine[key] - theirs[key]) < 1e-9, key
    assert mine["joint_error"] == pytest.approx(
        (mine["lag_acf_error"] + mine["corr_error"]) / 2)
    # identical windows: zero error on every part
    same = conditional_scores(real, real)
    assert same["lag_acf_error"] < 1e-12
    assert same["corr_error"] < 1e-12
    assert same["joint_error"] < 1e-12


# ------------------------------------------- the contract itself

def test_stage3_flowforge_10_part_1_contract_pins_fairness_rules():
    c = default_contract()
    assert c.schema == CONTRACT_SCHEMA
    # same splits for every family: session-disjoint, manifest-locked
    assert c.splits["n_train"] == 18468
    assert c.splits["n_val"] == 5712
    assert c.splits["n_test"] == 5712
    assert c.splits["manifest_sha256"] == (
        "b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd74fb861e1446ae581e")
    assert c.splits["session_disjoint"] is True
    # same physical representation and conditioning information
    assert c.representation["feature_names"] == FEATURE_NAMES
    assert c.representation["n_features"] == 7
    assert c.representation["window_seconds"] == WINDOW
    assert c.representation["regime_dims"] == 2
    assert c.representation["standardized_with"] == "train"
    assert c.representation["past_context_lookback_seconds"] == 60
    # exactly the same final discriminators
    assert c.discriminator["epochs"] == DISC_EPOCHS == 30
    assert c.discriminator["seed"] == DISC_PROTOCOL_SEED == 0
    assert c.discriminator["fit_split"] == "val"
    assert c.discriminator["score_split"] == "test"
    # equivalent tuning entitlement, recorded not assumed
    assert c.tuning["basis"] == "equal_optimizer_steps"
    assert c.tuning["equal_parameter_view"] is True
    # one generation seed set for every family, so the comparison is
    # paired seed by seed (the FF-R08 and C03 rule); independence
    # comes from per-family sampling streams, not from distinct
    # integers: seed 0 is shared with the training and discriminator
    # seeds by design, and derive_seed keeps the streams apart
    assert tuple(c.generation_seeds) == GENERATION_SEEDS == (0, 100, 200)
    assert c.train_seed == TRAIN_SEED == 0
    assert c.sampling_stream_seed("p10:ddpm") != \
        c.sampling_stream_seed("p10:autoregressive")
    from flowforge.models import derive_seed
    assert c.sampling_stream_seed("p10:ddpm") != \
        derive_seed("p10:ddpm", "training")
    assert c.sampling_stream_seed("p10:ddpm") != \
        derive_seed("p10:ddpm", "init")


def test_stage3_flowforge_10_part_1_contract_rejects_unfair_runs():
    c = default_contract()
    real, gen = _fixture_windows()
    good = {"method": "ddpm", "kind": "conditional", "seed": 100,
            "gen_val": gen, "gen_test": gen,
            "cost": {"sample_seconds": 1.0, "sample_seconds_per_1k": 10.0,
                     "peak_memory_mb": 100.0, "nfe": 100}}
    c.validate_run(good, n_val=32, n_test=32)
    # wrong window counts: a family scored on different splits
    bad = dict(good, gen_test=gen[:16])
    with pytest.raises(ValueError, match="window counts"):
        c.validate_run(bad, n_val=32, n_test=32)
    # missing cost metadata: no result may lack latency, NFE, memory
    bad = dict(good, cost={"nfe": 100})
    with pytest.raises(ValueError, match="cost metadata"):
        c.validate_run(bad, n_val=32, n_test=32)
    # generation seed outside the contract seed set
    bad = dict(good, seed=7)
    with pytest.raises(ValueError, match="seed"):
        c.validate_run(bad, n_val=32, n_test=32)


def test_stage3_flowforge_10_part_1_generation_seeds_independent():
    c = default_contract()
    # distinct seeds draw distinct windows from the same generator
    draws = {}
    for seed in GENERATION_SEEDS:
        rng = np.random.default_rng(seed)
        draws[seed] = rng.standard_normal((8, WINDOW, N_FEATURES))
    assert not np.array_equal(draws[0], draws[100])
    assert not np.array_equal(draws[0], draws[200])
    assert not np.array_equal(draws[100], draws[200])
    # same seed redraws the same windows: seeds are the identity
    again = np.random.default_rng(100).standard_normal(
        (8, WINDOW, N_FEATURES))
    assert np.array_equal(draws[100], again)
    # every family generates under exactly the contract seed set,
    # so the comparison is paired seed by seed
    c.check_seed_independence({"ddpm": [0, 100, 200],
                               "autoregressive": [0, 100, 200],
                               "flow_matching": [0, 100, 200]})
    # a family on a different set, or a subset, is refused: unpaired
    # seeds would let a lucky draw pose as a better family
    with pytest.raises(ValueError, match="seed set"):
        c.check_seed_independence({"ddpm": [0, 100, 200],
                                   "autoregressive": [300, 400, 500]})
    with pytest.raises(ValueError, match="seed set"):
        c.check_seed_independence({"ddpm": [0, 100, 200],
                                   "autoregressive": [0, 100]})
    # independence across families and from training lives in the
    # streams: derive_seed namespaces by run and stream name, so the
    # sampling stream of one family never shares draws with another
    # family or with the training stream of the same run
    from flowforge.models import derive_seed
    assert derive_seed("p10:ddpm", "sampling") != \
        derive_seed("p10:autoregressive", "sampling")
    assert derive_seed("p10:ddpm", "sampling") != \
        derive_seed("p10:ddpm", "training")
    assert derive_seed("p10:ddpm", "sampling") != \
        derive_seed("p10:ddpm", "init")


def test_stage3_flowforge_10_part_1_same_discriminator_is_deterministic():
    rng = np.random.default_rng(5)
    real_val = rng.standard_normal((8, WINDOW, N_FEATURES))
    gen_val = rng.standard_normal((8, WINDOW, N_FEATURES))
    real_test = rng.standard_normal((8, WINDOW, N_FEATURES))
    gen_test = rng.standard_normal((8, WINDOW, N_FEATURES))
    first = discriminator_auc(real_val, gen_val, real_test, gen_test,
                              seed=DISC_PROTOCOL_SEED, epochs=2)
    second = discriminator_auc(real_val, gen_val, real_test, gen_test,
                               seed=DISC_PROTOCOL_SEED, epochs=2)
    assert first["auc_test"] == second["auc_test"]
    assert first["auc_val"] == second["auc_val"]
    # a different protocol seed is a different discriminator: the
    # contract forbids scoring families under different seeds
    other = discriminator_auc(real_val, gen_val, real_test, gen_test,
                              seed=DISC_PROTOCOL_SEED + 1, epochs=2)
    assert other["auc_test"] != first["auc_test"]


# ------------------------------------------- Pareto table: no verdicts

def test_stage3_flowforge_10_part_1_pareto_table_no_superiority_labels():
    c = default_contract()
    baseline = _load_baseline_results()
    rows = c.pareto_table(baseline["pareto_rows"])
    assert rows["schema"] == CONTRACT_SCHEMA
    assert rows["columns"] == list(PARETO_COLUMNS)
    by_method = {r["method"]: r for r in rows["rows"]}
    assert by_method["ddpm"]["status"] == "baseline"
    for name in ("autoregressive", "conditional_diffusion_forecast",
                 "flow_matching"):
        assert by_method[name]["status"] == "pending"
        assert by_method[name]["mean_w1"] is None
    # no ranking, verdict, or winner anywhere in the table
    blob = json.dumps(rows)
    for banned in ("rank", "verdict", "winner", "best", "superior"):
        assert banned not in blob.lower()
    # the contract refuses a row that tries to smuggle one in
    with pytest.raises(ValueError, match="verdict"):
        c.pareto_table([{"method": "ddpm", "status": "baseline",
                         "mean_w1": 0.1, "verdict": "best"}])


def test_stage3_flowforge_10_part_1_baseline_row_matches_capture():
    baseline = _load_baseline_results()
    ddpm = next(r for r in baseline["pareto_rows"]
                if r["method"] == "ddpm")
    # expanded-corpus DDPM row, as measured by FF-R08
    assert abs(ddpm["mean_w1"] - 0.13764220411102132) < 1e-9
    assert abs(ddpm["auc_test"] - 0.709098) < 1e-9
    assert abs(ddpm["tstr_acc_synth"] - 0.5153333333333333) < 1e-9
    assert abs(ddpm["joint_error"] - 0.036234349360438144) < 1e-9
    assert ddpm["nfe_per_window"] == 100
    assert ddpm["params"] == 221767
    # demo-bundle cross-check, recomputed here by the oracle
    demo = baseline["demo_bundle"]
    d = np.load(os.path.join(REPO, "demo", "demo.npz"))
    oracle_w1 = float(np.mean([
        wasserstein_distance(d["real_windows"][:, :, j].ravel(),
                             d["gen_windows"][:, :, j].ravel())
        for j in range(N_FEATURES)]))
    assert abs(demo["mean_w1"] - oracle_w1) < 1e-9
    assert abs(demo["mean_w1"] - 0.2973) < 1e-3
    # the stored scores are float32, so the recomputation agrees
    # with the stored AUC only to float32 storage noise (~1e-8)
    assert abs(demo["auc_test"] - float(d["disc_auc"])) < 1e-7
    assert demo["n_windows"] == 64
    # the recorded demo-bundle hash is the hash of the committed file
    import hashlib
    with open(os.path.join(REPO, "demo", "demo.npz"), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert baseline["provenance"]["demo_npz_sha256"] == digest
    assert baseline["provenance"]["split_manifest_sha256"] == (
        "b7c40026cca52f826a4f0cb5f7025faab907627a8c9cbd74fb861e1446ae581e")
