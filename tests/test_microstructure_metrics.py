"""Stage 3 P15: stronger microstructure response diagnostics.

Fixture-first tests for flowforge/microstructure_metrics.py. Every numeric
expectation below is a hand-computed known answer or an independent NumPy
oracle; none of it shares code with the module under test.

Naming discipline under test: FlowForge signed_flow is observed trade
signed volume (buy positive), NOT order-flow imbalance as defined by
Cont, Kukanov, and Stoikov (it omits quote additions and cancellations).
No function, label, or report section may call it OFI. An OFI proxy is
derived from quote transitions only, with its assumptions explicit.
"""

import json
import math
import os

import numpy as np
import pytest

from flowforge.data import FEATURE_NAMES, N_FEATURES, WINDOW

J_RET = FEATURE_NAMES.index("log_ret")
J_SPR = FEATURE_NAMES.index("log_spread_bps")
J_BID = FEATURE_NAMES.index("log_bid_sz")
J_ASK = FEATURE_NAMES.index("log_ask_sz")
J_IMB = FEATURE_NAMES.index("imbalance")
J_FLOW = FEATURE_NAMES.index("signed_flow")
J_CNT = FEATURE_NAMES.index("log_trade_count")

# Identity standardization: physical values equal stored values, so toy
# windows can be built directly in physical units.
MEAN0 = np.zeros(N_FEATURES)
STD1 = np.ones(N_FEATURES)


def _toy_windows(n=200, seed=0):
    """(n, 60, 7) windows of small independent noise, physical units."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.1, size=(n, WINDOW, N_FEATURES)).astype(np.float64)
    # keep sizes positive in physical space: log sizes near 0 are fine,
    # spread via log1p stays small and positive after expm1.
    return w, rng


def _delayed_response_windows(n=200, lag=3, beta=0.6, seed=0, noise=0.5):
    """Signed flow drives the return exactly `lag` seconds later.

    ret[t + lag] = beta * flow[t] + noise, for every t the lag allows.
    All other features are independent noise.
    """
    w, rng = _toy_windows(n, seed)
    flow = rng.normal(size=(n, WINDOW))
    ret = rng.normal(0.0, noise, size=(n, WINDOW))
    ret[:, lag:] += beta * flow[:, :-lag]
    w[:, :, J_FLOW] = flow
    w[:, :, J_RET] = ret
    return w


# ------------------------------------------- signed-flow response

def test_stage3_flowforge_15_delayed_response_peaks_at_true_lag():
    from flowforge.microstructure_metrics import response_curve
    w = _delayed_response_windows(n=200, lag=3, beta=0.6, seed=0)
    phys = {"signed_flow": w[:, :, J_FLOW], "log_ret": w[:, :, J_RET]}
    out = response_curve(phys["signed_flow"], phys["log_ret"],
                         lags=(1, 2, 3, 4, 5), n_boot=50, seed=0)
    rows = {r["lag"]: r for r in out["rows"]}
    # known-answer structure: the peak is at the true lag, and only there
    assert out["peak_lag"] == 3
    assert rows[3]["corr"] > 0.3
    assert abs(rows[1]["corr"]) < 0.12
    assert abs(rows[5]["corr"]) < 0.12
    # slope at the true lag recovers beta up to sampling noise
    assert abs(rows[3]["slope"] - 0.6) < 0.15
    assert abs(rows[1]["slope"]) < 0.12
    # intervals are ordered and finite
    for r in out["rows"]:
        assert r["corr_ci"][0] <= r["corr"] <= r["corr_ci"][1]
        assert np.isfinite(r["corr_ci"]).all()


def test_stage3_flowforge_15_negative_delayed_response_sign():
    from flowforge.microstructure_metrics import response_curve
    w = _delayed_response_windows(n=200, lag=2, beta=-0.5, seed=1)
    out = response_curve(w[:, :, J_FLOW], w[:, :, J_RET],
                         lags=(1, 2, 3), n_boot=50, seed=0)
    rows = {r["lag"]: r for r in out["rows"]}
    assert out["peak_lag"] == 2
    assert rows[2]["corr"] < -0.25
    assert rows[2]["slope"] < -0.3


def test_stage3_flowforge_15_independent_noise_response_near_zero():
    from flowforge.microstructure_metrics import response_curve
    w, _ = _toy_windows(n=200, seed=2)
    out = response_curve(w[:, :, J_FLOW], w[:, :, J_RET],
                         lags=(1, 2, 3, 4, 5), n_boot=50, seed=0)
    for r in out["rows"]:
        assert abs(r["corr"]) < 0.12
        assert abs(r["slope"]) < 0.12
        # the null interval covers zero
        assert r["corr_ci"][0] <= 0.0 <= r["corr_ci"][1]


def test_stage3_flowforge_15_exact_unit_slope_known_answer():
    from flowforge.microstructure_metrics import response_curve
    rng = np.random.default_rng(3)
    n = 50
    flow = rng.normal(size=(n, WINDOW))
    ret = np.zeros((n, WINDOW))
    ret[:, 1:] = 2.0 * flow[:, :-1]
    out = response_curve(flow, ret, lags=(1, 2), n_boot=20, seed=0)
    rows = {r["lag"]: r for r in out["rows"]}
    assert abs(rows[1]["slope"] - 2.0) < 1e-9
    assert abs(rows[1]["corr"] - 1.0) < 1e-9
    # lag 2 has no structural link: independent draws, so only near zero
    assert abs(rows[2]["slope"]) < 0.05
    assert abs(rows[2]["corr"]) < 0.05


def test_stage3_flowforge_15_imbalance_response_known_lag():
    from flowforge.microstructure_metrics import response_curve
    n = 200
    w, rng = _toy_windows(n=n, seed=4)
    imb = rng.normal(size=(n, WINDOW))
    ret = rng.normal(0.0, 0.5, size=(n, WINDOW))
    ret[:, 1:] += 0.5 * imb[:, :-1]
    w[:, :, J_IMB] = imb
    w[:, :, J_RET] = ret
    out = response_curve(w[:, :, J_IMB], w[:, :, J_RET],
                         lags=(1, 2, 3), n_boot=50, seed=0)
    rows = {r["lag"]: r for r in out["rows"]}
    assert out["peak_lag"] == 1
    assert rows[1]["corr"] > 0.25
    assert abs(rows[3]["corr"]) < 0.12


# ------------------------------------------- cross-correlations

def test_stage3_flowforge_15_shifted_sine_cross_corr_peaks_at_shift():
    from flowforge.microstructure_metrics import cross_correlation_table
    n, t = 40, WINDOW
    tt = np.arange(t)
    a = np.sin(2 * np.pi * tt / 12.0)
    b = np.sin(2 * np.pi * (tt - 2) / 12.0)
    series = {
        "signed_flow": np.tile(a, (n, 1)),
        "log_ret": np.tile(b, (n, 1)),
    }
    out = cross_correlation_table(series, max_lag=5, n_boot=20, seed=0)
    rows = [r for r in out["rows"]
            if {r["series_a"], r["series_b"]} == {"signed_flow", "log_ret"}]
    by_lag = {r["lag"]: r for r in rows}
    # b lags a by 2, so corr(a[t], b[t+2]) is 1
    assert abs(by_lag[2]["corr"] - 1.0) < 1e-9
    assert by_lag[2]["corr"] > by_lag[0]["corr"]
    assert by_lag[2]["corr"] > by_lag[-2]["corr"]


def test_stage3_flowforge_15_cross_corr_symmetric_pair_listed_once():
    from flowforge.microstructure_metrics import cross_correlation_table
    w, _ = _toy_windows(n=10, seed=5)
    series = {name: w[:, :, FEATURE_NAMES.index(name)]
              for name in ("log_ret", "signed_flow", "imbalance")}
    out = cross_correlation_table(series, max_lag=1, n_boot=10, seed=0)
    pairs = {(r["series_a"], r["series_b"]) for r in out["rows"]}
    # unordered pairs in canonical feature order, each listed once,
    # both lag signs present
    assert pairs == {("signed_flow", "imbalance"),
                     ("signed_flow", "log_ret"),
                     ("imbalance", "log_ret")}
    lags = {r["lag"] for r in out["rows"]}
    assert lags == {-1, 0, 1}


# ------------------------------------------- transitions

def test_stage3_flowforge_15_spread_transition_rates_known_answer():
    from flowforge.microstructure_metrics import directional_transition_rates
    # hand-built spread path: up, up, down, flat, up, down, down, flat
    # diffs: +2, +1, -3, 0, +1, -1, -2, 0 -> up 3, down 3, flat 2 of 8
    spread = np.array([[10.0, 12.0, 13.0, 10.0, 10.0, 11.0, 10.0, 8.0,
                        8.0]])
    out = directional_transition_rates(spread)
    assert out["n_transitions"] == 8
    assert abs(out["up"] - 3 / 8) < 1e-12
    assert abs(out["down"] - 3 / 8) < 1e-12
    assert abs(out["flat"] - 2 / 8) < 1e-12
    assert abs(out["up"] + out["down"] + out["flat"] - 1.0) < 1e-12


def test_stage3_flowforge_15_constant_series_is_all_flat():
    from flowforge.microstructure_metrics import directional_transition_rates
    out = directional_transition_rates(np.full((5, WINDOW), 7.0))
    assert out["flat"] == 1.0
    assert out["up"] == 0.0
    assert out["down"] == 0.0


def test_stage3_flowforge_15_spread_tercile_matrix_known_answer():
    from flowforge.microstructure_metrics import tercile_transition_matrix
    # thresholds at 10 and 20; path cycles low -> mid -> high -> low ...
    spread = np.array([[5.0, 15.0, 25.0, 5.0, 15.0, 25.0, 5.0, 15.0]])
    out = tercile_transition_matrix(spread, thresholds=(10.0, 20.0))
    # transitions: L->M, M->H, H->L, L->M, M->H, H->L, L->M
    assert out["matrix"][0][1] == 1.0  # low always goes to mid
    assert out["matrix"][1][2] == 1.0  # mid always goes to high
    assert out["matrix"][2][0] == 1.0  # high always goes to low
    assert abs(sum(out["stationary"]) - 1.0) < 1e-12


# ------------------------------------------- tails and clustering

def test_stage3_flowforge_15_return_tail_quantiles_match_oracle():
    from flowforge.microstructure_metrics import return_tail_table
    rng = np.random.default_rng(6)
    ret = rng.normal(0.0, 1.0, size=(120, WINDOW))
    out = return_tail_table(ret, ret.copy(), n_boot=20, seed=0)
    qs = (0.01, 0.05, 0.95, 0.99)
    flat = ret.ravel()
    for row in out["rows"]:
        assert abs(row["real"] - float(np.quantile(flat, row["q"]))) < 1e-12
        assert abs(row["gen"] - row["real"]) < 1e-12
    # exceedance beyond 2 sigma for a standard normal is about 4.55%
    exc = {r["threshold"]: r for r in out["exceedance"]}
    assert abs(exc[2.0]["real"] - 0.0455) < 0.015
    assert abs(exc[3.0]["real"] - 0.0027) < 0.004


def test_stage3_flowforge_15_tail_w1_zero_for_identical_samples():
    from flowforge.microstructure_metrics import return_tail_table
    rng = np.random.default_rng(7)
    ret = rng.normal(size=(50, WINDOW))
    out = return_tail_table(ret, ret.copy(), n_boot=10, seed=0)
    assert out["tail_w1"] == 0.0


def test_stage3_flowforge_15_vol_clustering_ar1_known_answer():
    from flowforge.microstructure_metrics import volatility_clustering
    rng = np.random.default_rng(8)
    n = 200
    # |ret| follows an AR(1) with phi 0.7: build the absolute series,
    # then sign it at random so the signed returns stay mean-zero.
    phi = 0.7
    absret = np.empty((n, WINDOW))
    absret[:, 0] = np.abs(rng.normal(size=n))
    for t in range(1, WINDOW):
        absret[:, t] = (phi * absret[:, t - 1]
                        + (1 - phi) * np.abs(rng.normal(size=n)))
    signs = rng.choice([-1.0, 1.0], size=(n, WINDOW))
    ret = absret * signs
    out = volatility_clustering(ret, lags=(1, 2, 3), n_boot=50, seed=0)
    rows = {r["lag"]: r for r in out["rows"]}
    # ACF of |ret| at lag 1 recovers phi; it decays with lag
    assert abs(rows[1]["abs_acf"] - phi) < 0.15
    assert rows[1]["abs_acf"] > rows[3]["abs_acf"]
    assert rows[1]["abs_ci"][0] <= rows[1]["abs_acf"] <= rows[1]["abs_ci"][1]


def test_stage3_flowforge_15_vol_clustering_independent_noise_near_zero():
    from flowforge.microstructure_metrics import volatility_clustering
    w, _ = _toy_windows(n=200, seed=9)
    out = volatility_clustering(w[:, :, J_RET], lags=(1, 2, 3),
                                n_boot=50, seed=0)
    for r in out["rows"]:
        assert abs(r["abs_acf"]) < 0.12
        assert abs(r["sq_acf"]) < 0.12


# ------------------------------------------- trade intensity

def test_stage3_flowforge_15_trade_intensity_known_answers():
    from flowforge.microstructure_metrics import trade_intensity_table
    # constant counts: no zeros, zero dispersion, burstiness -1
    const = np.full((20, WINDOW), 5.0)
    out = trade_intensity_table(const, lags=(1, 2), n_boot=10, seed=0)
    assert out["zero_fraction"] == 0.0
    assert abs(out["fano_factor"]) < 1e-12
    assert abs(out["mean_count"] - 5.0) < 1e-12
    assert abs(out["burstiness"] - (-1.0)) < 1e-12
    # half the seconds empty: zero fraction 0.5, count ACF 0 for iid draws
    rng = np.random.default_rng(10)
    sparse = rng.choice([0.0, 4.0], size=(100, WINDOW))
    out = trade_intensity_table(sparse, lags=(1,), n_boot=20, seed=0)
    assert abs(out["zero_fraction"] - 0.5) < 0.03
    assert abs(out["mean_count"] - 2.0) < 0.1
    assert abs(out["count_acf_rows"][0]["acf"]) < 0.15


# ------------------------------------------- OFI proxy

def test_stage3_flowforge_15_ofi_proxy_exact_cks_with_prices():
    from flowforge.microstructure_metrics import ofi_proxy
    # hand-computed Cont-Kukanov-Stoikov flow, one window, 4 seconds.
    # bid px: 100, 101, 101, 100 ; ask px: 102, 102, 103, 103
    # bid sz: 10, 12, 9, 9 ; ask sz: 8, 8, 11, 7
    # t1: bid up -> +12 ; ask same -> -(8-8)=0 ; total +12
    # t2: bid same -> 9-12=-3 ; ask up -> +8 ; total +5
    # t3: bid down -> -9 ; ask same -> -(7-11)=+4 ; total -5
    bid_sz = np.array([[10.0, 12.0, 9.0, 9.0]])
    ask_sz = np.array([[8.0, 8.0, 11.0, 7.0]])
    bid_px = np.array([[100.0, 101.0, 101.0, 100.0]])
    ask_px = np.array([[102.0, 102.0, 103.0, 103.0]])
    out = ofi_proxy(bid_sz, ask_sz, bid_px=bid_px, ask_px=ask_px)
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], [12.0, 5.0, -5.0])


def test_stage3_flowforge_15_ofi_proxy_midret_assumption_known_answer():
    from flowforge.microstructure_metrics import ofi_proxy
    # mid rises then falls then flat; spread held fixed, so bid and ask
    # prices move together with the mid by assumption.
    # t1 mid up: +bid_sz[1] + ask_sz[0] = 12 + 8 = 20
    # t2 mid down: -bid_sz[1] - ask_sz[2] = -12 - 11 = -23
    # t3 mid flat: (9-9) - (7-11) = +4
    bid_sz = np.array([[10.0, 12.0, 9.0, 9.0]])
    ask_sz = np.array([[8.0, 8.0, 11.0, 7.0]])
    mid_ret = np.array([[0.0, 0.01, -0.01, 0.0]])
    out = ofi_proxy(bid_sz, ask_sz, mid_ret=mid_ret)
    np.testing.assert_allclose(out[0], [20.0, -23.0, 4.0])


def test_stage3_flowforge_15_ofi_proxy_needs_prices_or_mid():
    from flowforge.microstructure_metrics import ofi_proxy
    with pytest.raises(ValueError, match="bid_px|mid_ret"):
        ofi_proxy(np.ones((2, 5)), np.ones((2, 5)))


# ------------------------------------------- regimes and baseline

def test_stage3_flowforge_15_regime_response_only_in_high_vol():
    from flowforge.microstructure_metrics import regime_response
    rng = np.random.default_rng(11)
    n = 200
    flow = rng.normal(size=(n, WINDOW))
    ret = rng.normal(0.0, 0.5, size=(n, WINDOW))
    # response only in the high-vol half of windows
    ret[n // 2:, 2:] += 0.6 * flow[n // 2:, :-2]
    regime = np.zeros((n, 2))
    regime[:n // 2, 0] = -1.0  # low past vol
    regime[n // 2:, 0] = 1.0   # high past vol
    out = regime_response(flow, ret, regime, lags=(1, 2, 3),
                          n_boot=50, seed=0)
    cells = {(r["regime"], r["lag"]): r for r in out["rows"]}
    assert cells[("high_vol", 2)]["corr"] > 0.3
    assert abs(cells[("low_vol", 2)]["corr"]) < 0.12
    assert abs(cells[("high_vol", 1)]["corr"]) < 0.12


def test_stage3_flowforge_15_shuffle_baseline_destroys_response():
    from flowforge.microstructure_metrics import (response_curve,
                                                  shuffle_baseline)
    w = _delayed_response_windows(n=200, lag=3, beta=0.6, seed=12)
    flow = w[:, :, J_FLOW]
    ret = w[:, :, J_RET]
    # marginals preserved by the shuffle, temporal order destroyed
    sh_flow, sh_ret = shuffle_baseline(flow, ret, seed=0)
    assert abs(float(np.sort(sh_flow.ravel())[0])
               - float(np.sort(flow.ravel())[0])) < 1e-12
    out = response_curve(sh_flow, sh_ret, lags=(1, 2, 3, 4, 5),
                         n_boot=50, seed=0)
    for r in out["rows"]:
        assert abs(r["corr"]) < 0.12


# ------------------------------------------- report assembly

def test_stage3_flowforge_15_report_sections_baseline_and_labels():
    from flowforge.microstructure_metrics import microstructure_report
    w_real = _delayed_response_windows(n=64, lag=3, beta=0.6, seed=13)
    w_gen = _delayed_response_windows(n=64, lag=3, beta=0.6, seed=14)
    rep = microstructure_report(w_real, w_gen, MEAN0, STD1, n_boot=20,
                                seed=0)
    for section in ("signed_flow_response", "imbalance_response",
                    "cross_correlations", "transitions", "return_tails",
                    "volatility_clustering", "trade_intensity",
                    "regime_response", "ofi_proxy_response"):
        assert section in rep, section
    # the signed-flow label names it trade signed volume, never OFI:
    # the only OFI mention is the explicit "(not OFI)" disclaimer
    label = rep["signed_flow_response"]["label"]
    assert label == "signed trade volume (not OFI)"
    assert label.replace("(not OFI)", "").find("OFI") == -1
    # every response row carries real, generated, baseline, and intervals
    for row in rep["signed_flow_response"]["rows"]:
        assert {"real", "gen", "baseline", "real_ci", "gen_ci"} <= set(row)
    # the report is JSON-safe and records its population
    json.dumps(rep)
    assert rep["n_windows_real"] == 64
    assert rep["n_windows_gen"] == 64
    assert rep["baseline"].startswith("within-window time shuffle")
    # the delayed response survives in the report at the true lag
    rows = {r["lag"]: r for r in rep["signed_flow_response"]["rows"]}
    assert rows[3]["real"] > 0.3
    assert abs(rows[3]["baseline"]) < 0.15


def test_stage3_flowforge_15_report_on_demo_bundle_sections_present():
    from flowforge.microstructure_metrics import microstructure_report
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundle = np.load(os.path.join(root, "demo", "demo.npz"),
                     allow_pickle=False)
    real = bundle["real_windows"].astype(np.float64)
    gen = bundle["gen_windows"].astype(np.float64)
    mean = bundle["feature_mean"]
    std = bundle["feature_std"]
    rep = microstructure_report(real, gen, mean, std, n_boot=20, seed=0)
    json.dumps(rep)
    # point values on the held-out demo windows are finite
    for row in rep["signed_flow_response"]["rows"]:
        assert np.isfinite(row["real"]) and np.isfinite(row["gen"])
    for row in rep["return_tails"]["rows"]:
        assert np.isfinite(row["real"]) and np.isfinite(row["gen"])
    assert rep["n_windows_real"] == 64 and rep["n_windows_gen"] == 64


def test_stage3_flowforge_15_perfect_invariants_zero_violations():
    from flowforge.microstructure_metrics import physical_series
    rng = np.random.default_rng(15)
    n = 30
    # build physical windows where imbalance is exactly the size-implied
    # value, spreads are positive, and counts are integers
    log_bid = rng.normal(3.0, 0.5, size=(n, WINDOW))
    log_ask = rng.normal(3.0, 0.5, size=(n, WINDOW))
    bid, ask = np.exp(log_bid), np.exp(log_ask)
    imb = (bid - ask) / (bid + ask)
    counts = rng.integers(0, 10, size=(n, WINDOW)).astype(np.float64)
    w = np.zeros((n, WINDOW, N_FEATURES))
    w[:, :, J_RET] = rng.normal(0, 0.001, size=(n, WINDOW))
    w[:, :, J_SPR] = np.log1p(rng.uniform(0.5, 5.0, size=(n, WINDOW)))
    w[:, :, J_BID] = log_bid
    w[:, :, J_ASK] = log_ask
    w[:, :, J_IMB] = imb
    w[:, :, J_FLOW] = rng.normal(size=(n, WINDOW))
    w[:, :, J_CNT] = np.log1p(counts)
    phys = physical_series(w, MEAN0, STD1)
    # imbalance reconstructed from sizes matches the stored feature
    derived = (phys["bid_sz"] - phys["ask_sz"]) / \
        (phys["bid_sz"] + phys["ask_sz"])
    np.testing.assert_allclose(derived, phys["imbalance"], atol=1e-12)
    # trade counts round-trip to integers; spreads stay positive
    np.testing.assert_allclose(phys["trade_count"], counts, atol=1e-9)
    assert (phys["spread_bps"] > 0).all()


def test_stage3_flowforge_15_full_report_wires_microstructure():
    from flowforge.evaluate import full_report, print_report
    rng = np.random.default_rng(16)
    real = rng.normal(size=(16, WINDOW, N_FEATURES)).astype(np.float64)
    gen = rng.normal(size=(16, WINDOW, N_FEATURES)).astype(np.float64)
    rep = full_report(real, gen, real, gen, seed=0,
                      feature_mean=MEAN0, feature_std=STD1)
    assert "microstructure" in rep
    assert "signed_flow_response" in rep["microstructure"]
    assert rep["microstructure"]["n_windows_real"] == 16
    print_report(rep)  # must not raise
