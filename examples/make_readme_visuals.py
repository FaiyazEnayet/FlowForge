"""Build FlowForge's README figures from its recorded demo bundle."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from build_dashboard import DOCS, _load

BG, SURFACE, INK, MUTED = "#0c1727", "#111f32", "#e6edf1", "#a8bac8"
BLUE, COPPER = "#a9d8eb", "#d3a484"


def canvas(title, subtitle, rows=1, cols=2):
    fig, axes = plt.subplots(rows, cols, figsize=(11, 5.4), dpi=100,
                             facecolor=BG, squeeze=False)
    fig.subplots_adjust(left=.105, right=.94, top=.66, bottom=.17, wspace=.34, hspace=.85)
    fig.text(.065, .93, "FLOWFORGE  /  MARKET MICROSTRUCTURE", color=BLUE, size=10, weight="bold")
    fig.text(.065, .83, title, color=INK, size=27, weight="bold")
    fig.text(.065, .75, subtitle, color=MUTED, size=11)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, length=0, pad=6, labelsize=9)
        ax.spines[:].set_visible(False)
        ax.grid(axis="x" if cols == 2 else "y", color="#30475d", alpha=.7)
        ax.set_axisbelow(True)
    return fig, axes.flatten()


def compare(metrics):
    fig, axes = canvas("How close are generated markets?",
                       f"DDPM and VAE on the same {metrics['n_test']} held-out test windows")
    for ax, key, label, ceiling in zip(axes, ("mean_w1", "auc_test"),
                                      ("Mean Wasserstein-1 distance", "Discriminator ROC AUC"), (.65, 1.18)):
        values = [metrics["models"][model][key] for model in ("ddpm", "vae")]
        ax.barh([1, 0], values, color=[BLUE, COPPER], height=.45)
        ax.set_yticks([1, 0], ["DDPM", "VAE"])
        ax.set_xlim(0, ceiling)
        if key == "auc_test":
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.axvline(.5, color=INK, linestyle=":", linewidth=1.2)
        ax.set_title(label, loc="left", color=INK, size=12, pad=12)
        for y, value in zip([1, 0], values):
            ax.text(value+.016, y, f"{value:.3f}", va="center", color=INK, size=11)
        ax.set_xlabel("Lower means closer marginals" if key == "mean_w1" else "0.50 = chance-level separation",
                      color=MUTED, labelpad=9)
    fig.text(.065, .045, "Source: demo/metrics.json · Recorded evaluation bundle", color=MUTED, size=10)
    return fig


def frame(fig):
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format="png", facecolor=BG)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").quantize(colors=192)


def sequence(d):
    fig, axes = canvas("Replay a market window",
                       "Observed test sample and DDPM sample · independent windows aligned by elapsed time",
                       rows=3, cols=1)
    fig.subplots_adjust(top=.68, bottom=.14, hspace=.75)
    j = d["j"]
    real, gen = d["real"][0], d["gen"][0]
    xs = np.arange(len(real))
    lines = []
    for ax, label, observed, generated in zip(
        axes, ("Midprice index", "Spread · bps", "Book imbalance"),
        (d["midpath"](real), d["spread"](real), real[:, j["imbalance"]]),
        (d["midpath"](gen), d["spread"](gen), gen[:, j["imbalance"]])
    ):
        ax.set_ylabel(label, color=MUTED, size=9)
        ax.set_xlim(0, xs[-1])
        lower, upper = min(observed.min(), generated.min()), max(observed.max(), generated.max())
        pad = max((upper-lower)*.14, 1e-9)
        ax.set_ylim(lower-pad, upper+pad)
        for data, color, style in ((observed, BLUE, "-"), (generated, COPPER, "--")):
            line, = ax.plot([], [], color=color, linewidth=1.8, linestyle=style)
            lines.append((line, data))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Seconds into sample", color=MUTED, size=9)
    fig.text(.105, .035, "━━ Observed", color=BLUE, size=10)
    fig.text(.24, .035, "┅┅ Generated DDPM", color=COPPER, size=10)
    frames = []
    for stop in range(4, len(real)+1, 4):
        for line, data in lines:
            line.set_data(xs[:stop], data[:stop])
        frames.append(frame(fig))
    plt.close(fig)
    return frames


def main():
    d = _load()
    frames = sequence(d)
    fig = compare(d["metrics"])
    fig.savefig(f"{DOCS}/results.png", facecolor=BG)
    frames.append(frame(fig))
    plt.close(fig)
    frames[0].save(f"{DOCS}/demo.gif", save_all=True, append_images=frames[1:],
                   duration=[130]*(len(frames)-2)+[2000, 2800], loop=0, optimize=True)


if __name__ == "__main__":
    main()
