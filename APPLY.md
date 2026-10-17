# FlowForge generator update

Copy the contents of this folder into your own checkout when ready. This package did not modify any existing checkout or GitHub.

Install your repository's original generator dependencies, plus `python -m pip install -r portfolio_ui/requirements.txt`.
Run the original command for `examples/build_dashboard.py`. Its calculations are unchanged; the final write applies the new local presentation.

To rebuild this exact recorded snapshot without retraining:

`python portfolio_ui/renderer.py --repo FlowForge --root . --input portfolio_ui/original-dashboard-snapshot.html --output docs/dashboard.html`

For SukukLab, `python examples/make_dashboard.py` is the Julia-free alternative. The supplied Python alternative retains the original structure-schedule snapshot and recomputes valuation/curve data from the documented source formulas.

The HTML itself needs no Python, Julia, Go, network or server. Keep DeepArcade's game.html beside dashboard.html for local gameplay.
