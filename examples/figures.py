"""Standout visuals: real vs dreamed market windows, side by side.

Reads runs/eval_arrays.npz (written by examples/demo.py) and saves:
  runs/figures/windows.png       -- midprice path, spread, imbalance for one
                                    real and one generated test window
  runs/figures/discriminator.png -- score histograms: real vs generated test
                                    windows, annotated with the test AUC

PNGs are gitignored; Faiyaz uploads them manually and the README embeds them.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RUNS = os.path.join(HERE, "..", "runs")
FIGS = os.path.join(RUNS, "figures")

FEATURES = ["log_ret", "log_spread_bps", "log_bid_sz", "log_ask_sz",
            "imbalance", "signed_flow", "log_trade_count"]
J_RET, J_SPREAD, J_IMB = (FEATURES.index(n) for n in
                          ("log_ret", "log_spread_bps", "imbalance"))


def _unstandardize(x, mean, std):
    return x * std + mean


def main():
    z = np.load(os.path.join(RUNS, "eval_arrays.npz"))
    real = z["real_test"]          # (n, 60, 7) standardized
    gen = z["gen_test_ddpm"]       # (n, 60, 7) standardized
    s_real = z["disc_scores_real"]  # discriminator scores, test split
    s_gen = z["disc_scores_gen"]
    f_mean, f_std = z["feature_mean"], z["feature_std"]
    with open(os.path.join(RUNS, "metrics.json")) as f:
        metrics = json.load(f)
    auc = metrics["ddpm"]["discriminator"]["auc_test"]

    os.makedirs(FIGS, exist_ok=True)

    # pick representative windows: median realized volatility among test
    ru = _unstandardize(real, f_mean, f_std)
    vol = np.abs(ru[:, :, J_RET]).mean(axis=1)
    i_real = int(np.argsort(vol)[len(vol) // 2])
    gu = _unstandardize(gen, f_mean, f_std)
    gvol = np.abs(gu[:, :, J_RET]).mean(axis=1)
    i_gen = int(np.argsort(gvol)[len(gvol) // 2])

    rw, gw = ru[i_real], gu[i_gen]
    # midprice path from log returns (start at 100 for readability)
    r_px = 100.0 * np.exp(np.cumsum(rw[:, J_RET]))
    g_px = 100.0 * np.exp(np.cumsum(gw[:, J_RET]))
    # spread back to bps: stored as log1p(spread_bps)
    r_sp = np.expm1(rw[:, J_SPREAD])
    g_sp = np.expm1(gw[:, J_SPREAD])

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    secs = np.arange(60)
    panels = [
        ("Midprice path (rebased to 100)", r_px, g_px),
        ("Spread (bps)", r_sp, g_sp),
        ("Book imbalance", rw[:, J_IMB], gw[:, J_IMB]),
    ]
    for row, (title, r, g) in enumerate(panels):
        axes[row, 0].plot(secs, r, color="#1f77b4", lw=1.5)
        axes[row, 0].set_title(f"Real — {title}", fontsize=11)
        axes[row, 1].plot(secs, g, color="#d62728", lw=1.5)
        axes[row, 1].set_title(f"Dreamed (diffusion) — {title}", fontsize=11)
        for col in range(2):
            axes[row, col].grid(alpha=0.3)
    axes[2, 0].set_xlabel("seconds into window")
    axes[2, 1].set_xlabel("seconds into window")
    fig.suptitle("Real market vs dreamed market: one test window each",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "windows.png"), dpi=150)
    plt.close(fig)
    print("saved runs/figures/windows.png")

    # discriminator score histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)
    ax.hist(s_real, bins=bins, alpha=0.6, label="real test windows",
            color="#1f77b4", density=True)
    ax.hist(s_gen, bins=bins, alpha=0.6, label="generated test windows",
            color="#d62728", density=True)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("discriminator score (1 = looks real)")
    ax.set_ylabel("density")
    ax.set_title(f"Can a classifier tell dreams from reality? "
                 f"Test AUC = {auc:.3f}  (0.5 = indistinguishable)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "discriminator.png"), dpi=150)
    plt.close(fig)
    print("saved runs/figures/discriminator.png")


if __name__ == "__main__":
    main()
