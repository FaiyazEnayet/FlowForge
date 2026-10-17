"""Build the Pages-ready interactive dashboard: docs/dashboard.html.

A single self-contained HTML file (plotly.js inlined) so it can be served by
GitHub Pages with no build step. All data comes from the committed demo/
bundle (examples/build_demo.py), which itself comes from examples/demo.py
(seed 7). Nothing is invented: the "generated" windows are outputs of the
trained DDPM.

Sections:
  1. Window explorer: dropdown over 8 real + 8 generated test windows,
     midprice path / spread / imbalance per window.
  2. Discriminator: overlaid score histograms, real vs generated.
  3. Evaluation: marginal table, order-flow autocorrelation table,
     model comparison numbers.
"""

import json
import os

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "demo")
DOCS = os.path.join(HERE, "..", "docs")

N_SHOW = 8  # windows per source in the explorer


def _load():
    z = np.load(os.path.join(DEMO, "demo.npz"), allow_pickle=True)
    with open(os.path.join(DEMO, "metrics.json")) as f:
        metrics = json.load(f)
    names = [str(n) for n in z["feature_names"].tolist()]
    j = {n: i for i, n in enumerate(names)}
    mean = np.asarray(z["feature_mean"], dtype=np.float64)
    std = np.asarray(z["feature_std"], dtype=np.float64)

    def unstd(x):
        return x * std + mean

    def midpath(w):
        return 100.0 * np.exp(np.cumsum(w[:, j["log_ret"]]))

    def spread(w):
        return np.expm1(w[:, j["log_spread_bps"]])

    real = [unstd(np.asarray(z["real_windows"][i], dtype=np.float64))
            for i in range(N_SHOW)]
    gen = [unstd(np.asarray(z["gen_windows"][i], dtype=np.float64))
           for i in range(N_SHOW)]
    real_idx = np.asarray(z["real_idx"][:N_SHOW]).tolist()
    gen_idx = np.asarray(z["gen_idx"][:N_SHOW]).tolist()
    return {
        "names": names, "j": j, "metrics": metrics,
        "real": real, "gen": gen, "real_idx": real_idx, "gen_idx": gen_idx,
        "s_real": np.asarray(z["disc_scores_real"], dtype=np.float64),
        "s_gen": np.asarray(z["disc_scores_gen"], dtype=np.float64),
        "disc_auc": float(z["disc_auc"]),
        "midpath": midpath, "spread": spread,
    }


def _explorer(d):
    """Three subplots with a dropdown to flip through real/dreamed windows."""
    t = np.arange(60)
    sources = ([("real", d["real"], d["real_idx"], "steelblue")] +
               [("generated", d["gen"], d["gen_idx"], "darkorange")])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("midprice path (rebased to 100)",
                                        "spread (bps)", "book imbalance"),
                        vertical_spacing=0.08)
    buttons = []
    trace_no = 0
    n_traces = 2 * N_SHOW * 3
    for src, windows, idxs, color in sources:
        for k, w in enumerate(windows):
            px = d["midpath"](w)
            sp = d["spread"](w)
            imb = w[:, d["j"]["imbalance"]]
            vis = [False] * n_traces
            for r in range(3):
                vis[trace_no + r] = True
            fig.add_trace(go.Scatter(x=t, y=px, mode="lines", name="midprice",
                                     line=dict(color=color, width=2),
                                     visible=(trace_no == 0)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=sp, mode="lines", name="spread",
                                     line=dict(color=color, width=2),
                                     visible=(trace_no == 0)), row=2, col=1)
            fig.add_trace(go.Scatter(x=t, y=imb, mode="lines", name="imbalance",
                                     line=dict(color=color, width=2),
                                     visible=(trace_no == 0)), row=3, col=1)
            label = (f"real test window #{idxs[k]}" if src == "real"
                     else f"DDPM dream #{idxs[k]}")
            buttons.append(dict(label=label, method="update",
                                args=[{"visible": vis},
                                      {"title": label}]))
            trace_no += 3
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down",
                          showactive=True, x=0.0, y=1.12,
                          xanchor="left", yanchor="top")],
        title="real test window #0",
        height=640, margin=dict(t=110))
    fig.update_xaxes(title_text="seconds into window", row=3, col=1)
    return fig


def _discriminator(d):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=d["s_real"], nbinsx=40, opacity=0.6,
                               name="real test windows",
                               marker_color="steelblue"))
    fig.add_trace(go.Histogram(x=d["s_gen"], nbinsx=40, opacity=0.6,
                               name="generated windows",
                               marker_color="darkorange"))
    fig.update_layout(
        barmode="overlay",
        title=(f"Discriminator scores on the held-out test split "
               f"(AUC {d['disc_auc']:.3f}; 0.5 = indistinguishable)"),
        xaxis_title="discriminator score (1 = looks generated)",
        yaxis_title="windows", height=420)
    return fig


def _tables(d):
    m = d["metrics"]
    rows = m["marginals"]
    marg = go.Figure(data=[go.Table(
        header=dict(values=["feature", "real mean", "gen mean", "real std",
                            "gen std", "W1"],
                    fill_color="lightsteelblue", align="left"),
        cells=dict(values=[[r["feature"] for r in rows],
                           [round(r["real_mean"], 3) for r in rows],
                           [round(r["gen_mean"], 3) for r in rows],
                           [round(r["real_std"], 3) for r in rows],
                           [round(r["gen_std"], 3) for r in rows],
                           [round(r["w1"], 3) for r in rows]],
                   align="left"))])
    marg.update_layout(title="Marginal distributions, test split (DDPM)",
                       height=340, margin=dict(t=60, b=10))

    acf = m["flow_acf"]
    acf_fig = go.Figure(data=[go.Table(
        header=dict(values=["lag", "real", "generated"],
                    fill_color="lightsteelblue", align="left"),
        cells=dict(values=[[r["lag"] for r in acf],
                           [round(r["real_acf"], 3) for r in acf],
                           [round(r["gen_acf"], 3) for r in acf]],
                   align="left"))])
    acf_fig.update_layout(
        title=("Signed order flow autocorrelation: real flow is near zero "
               "here, so the check is whether the generator reproduces that "
               "flatness instead of inventing structure."),
        height=300, margin=dict(t=80, b=10))
    return marg, acf_fig


def main():
    d = _load()
    m = d["metrics"]
    md = m["models"]

    explorer = _explorer(d)
    disc = _discriminator(d)
    marg, acf_fig = _tables(d)

    div_explorer = pio.to_html(explorer, include_plotlyjs=True,
                               full_html=False)
    div_disc = pio.to_html(disc, include_plotlyjs=False, full_html=False)
    div_marg = pio.to_html(marg, include_plotlyjs=False, full_html=False)
    div_acf = pio.to_html(acf_fig, include_plotlyjs=False, full_html=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlowForge: limit order book dreams</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         max-width: 1000px; margin: 0 auto; padding: 24px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #555; margin-top: 0; }}
  .kpis {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
  .kpi {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px;
          min-width: 200px; }}
  .kpi .v {{ font-size: 28px; font-weight: 700; }}
  .kpi .l {{ font-size: 13px; color: #555; }}
  h2 {{ margin-top: 40px; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
  .note {{ font-size: 13px; color: #555; }}
</style>
</head>
<body>
<h1>FlowForge</h1>
<p class="sub">Teach an AI what a real market looks like, then ask it to dream
up new ones. A DDPM diffusion model trained on real BTCUSDT top-of-book
windows, evaluated honestly against reality.</p>
<p class="note">Dataset: {m['n_train']}/{m['n_val']}/{m['n_test']}
train/val/test windows from {m['seconds']:,} seconds of book data
({m['chunks']} sessions). DDPM: {m['ddpm_params']:,} params,
{m['ddpm_epochs']} epochs, seed {m['seed']}. Everything on this page comes from
a precomputed demo bundle; the generated windows are outputs of the trained
model, not hand drawn.</p>

<h2>Model comparison</h2>
<div class="kpis">
  <div class="kpi"><div class="v">{md['ddpm']['mean_w1']:.3f}</div>
    <div class="l">DDPM mean Wasserstein distance (lower is better)</div></div>
  <div class="kpi"><div class="v">{md['vae']['mean_w1']:.3f}</div>
    <div class="l">VAE baseline mean Wasserstein distance</div></div>
  <div class="kpi"><div class="v">{md['ddpm']['auc_test']:.3f}</div>
    <div class="l">DDPM discriminator AUC, test (0.5 = indistinguishable)</div></div>
  <div class="kpi"><div class="v">{md['vae']['auc_test']:.3f}</div>
    <div class="l">VAE baseline discriminator AUC, test</div></div>
</div>

<h2>Window explorer</h2>
<p class="note">Pick a real test window or a DDPM dream from the dropdown.
Midprice is rebased to 100 for readability.</p>
{div_explorer}

<h2>Discriminator</h2>
<p class="note">A classifier trained on validation windows (real vs generated),
scored on test windows the generator never saw.</p>
{div_disc}

<h2>Evaluation</h2>
{div_marg}
{div_acf}

<p class="note">Built by <code>examples/build_dashboard.py</code> from the
demo bundle (<code>demo/</code>), which comes from
<code>examples/demo.py</code>, seed {m['seed']}. Source:
<a href="https://github.com/FaiyazEnayet/flowforge">github.com/FaiyazEnayet/flowforge</a></p>
</body>
</html>
"""
    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(_portfolio_html(html))
    print(f"dashboard -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")



def _portfolio_html(raw):
    """Presentation-only transform of this run’s freshly calculated figures."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "portfolio_ui"))
    from renderer import render_html
    return render_html(root, 'FlowForge', raw)[0]

if __name__ == "__main__":
    main()
