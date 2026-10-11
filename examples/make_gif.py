"""Make an animated GIF preview of the dashboard: docs/demo.gif.

Programmatically rendered with matplotlib from the committed demo bundle
(examples/build_demo.py). Six real-vs-dreamed window pairs; each pair draws
the midprice path progressively over the 60 second window, side by side.
Nothing is faked: every frame is real test data or DDPM output from
examples/demo.py, seed 7.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "demo")
DOCS = os.path.join(HERE, "..", "docs")

N_PAIRS = 6
FRAMES_PER_PAIR = 15
FPS = 10


def main():
    z = np.load(os.path.join(DEMO, "demo.npz"), allow_pickle=True)
    names = [str(n) for n in z["feature_names"].tolist()]
    j_ret = names.index("log_ret")
    mean = np.asarray(z["feature_mean"], dtype=np.float64)
    std = np.asarray(z["feature_std"], dtype=np.float64)
    real_idx = np.asarray(z["real_idx"]).tolist()
    gen_idx = np.asarray(z["gen_idx"]).tolist()

    def path(w):
        w = w.astype(np.float64) * std + mean
        return 100.0 * np.exp(np.cumsum(w[:, j_ret]))

    pairs = [(path(z["real_windows"][i]), path(z["gen_windows"][i]),
              real_idx[i], gen_idx[i]) for i in range(N_PAIRS)]
    t = np.arange(60)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), sharey=False)
    lines = []
    for ax, title, color in zip(axes, ["REAL test window", "DDPM dream"],
                                ["steelblue", "darkorange"]):
        (ln,) = ax.plot([], [], lw=2, color=color)
        ax.set_xlim(0, 59)
        ax.set_xlabel("seconds")
        ax.grid(alpha=0.3)
        lines.append((ax, ln, title))

    def init():
        for ax, ln, _ in lines:
            ln.set_data([], [])
        return [ln for _, ln, _ in lines]

    def frame(f):
        pair = f // FRAMES_PER_PAIR
        step = f % FRAMES_PER_PAIR
        n = int(60 * (step + 1) / FRAMES_PER_PAIR)
        artists = []
        for (ax, ln, title), px, idx in zip(
                lines,
                [pairs[pair][0], pairs[pair][1]],
                [pairs[pair][2], pairs[pair][3]]):
            ln.set_data(t[:n], px[:n])
            ax.set_title(f"{title} #{idx}", fontsize=11)
            lo, hi = px.min(), px.max()
            pad = max((hi - lo) * 0.15, 0.05)
            ax.set_ylim(lo - pad, hi + pad)
            artists.append(ln)
        fig.suptitle("FlowForge: real market windows vs DDPM dreams",
                     fontsize=12, y=0.98)
        return artists

    anim = FuncAnimation(fig, frame, frames=N_PAIRS * FRAMES_PER_PAIR,
                         init_func=init, blit=False)
    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "demo.gif")
    anim.save(out, writer="pillow", fps=FPS)
    plt.close(fig)
    print(f"gif -> {out} ({os.path.getsize(out) / 1e6:.2f} MB, "
          f"{N_PAIRS * FRAMES_PER_PAIR} frames)")


if __name__ == "__main__":
    main()
