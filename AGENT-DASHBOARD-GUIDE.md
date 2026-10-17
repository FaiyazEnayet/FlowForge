# Dashboard and README Update Guide for Coding Agents

## Purpose

Use this guide when changing any of the eleven FaiyazEnayet portfolio repositories. It explains how to make a code or data change appear consistently in the interactive dashboard, the published GitHub Pages page, README images and figures, and any generated local files. Follow the repository's existing calculations and data contracts. A polished presentation must never change or overstate the underlying result.

## The update path

For a normal dashboard change, follow this path:

1. Identify the repository, its primary language, the calculation and data source, and the dashboard generator listed below.
2. Change the source data, model, analysis, or generator only when the task calls for it. Preserve model assumptions and output schemas unless a deliberate migration is part of the request.
3. Update the dashboard renderer or its project-specific design settings. Keep the visual identity described below. A shared technical renderer may provide accessibility and responsive behavior, but do not make different projects look alike.
4. Run the project's supported generator from its documented working directory. Confirm it writes the Pages artifact at `docs/dashboard.html`. For FormulaForge, compile and run the Go command. For SukukLab, use the Julia source when the required package is installed; the Python alternative is available for a reproducible dashboard export.
5. Update `docs/demo.gif` only when a chart, simulation, or interaction changes enough that its README animation would be stale. Animate source-backed quantities or a real model rollout. Never record a scrolling tour of the webpage or make an animation imply evidence the project does not have.
6. Update `docs/results.png` only when the underlying documented result changes. Recompute the figure from the repository's actual evaluation outputs and describe its sample, units, and conditions.
7. Update the README's explanation, methodology, result figure, demo image, commands, runtime badge, and limitations when they are affected. Keep the direct “Open interactive demo” link pointed at `https://faiyazenayet.github.io/REPOSITORY/docs/dashboard.html`. Retain the limitations section. Do not add a separate “HTML file” link, a limitations navigation link, an unexplained technology tagline, em dashes, or Codex branding.
8. Check all touched files together. A changed generator with an old generated HTML, or a new metric with an old README claim, is an incomplete update.

## Repository map

| Repository | Main language | Dashboard generator | Data and design notes |
| --- | --- | --- | --- |
| `neural-cppi` | Python | `scripts/make_dashboard.py` | Preserve the distinction between synthetic experiments, historical returns, and bootstrap scenarios. Check floors, breach rates, horizons, allocation outputs, and the README's risk caveats. |
| `FlowForge` | Python | `examples/build_dashboard.py` | Keep real and generated samples clearly labeled and independently sampled. Maintain the sample selector, time offset, spread, imbalance, distribution metrics, and dependence diagnostics. |
| `toxicflow` | Rust and Python | `scripts/dashboard.py` | Treat event timestamps as the shared chart clock. Preserve actual classifier and market-making results, even when a baseline performs better. |
| `efficient-frontier-fc` | Python | `scripts/make_dashboard.py` | Keep squad feasibility, the wage budget, position constraints, selected XI, benchmark outputs, and wage-proxy caveats consistent. Player names must remain readable. |
| `sukuklab` | Julia | `examples/make_dashboard.jl` | Keep face-value units, compounding convention, date basis, coupon or profit cash flows, and the January 2026 IILM observation context explicit. Financial modeling does not certify legal or Sharia compliance. |
| `formulaforge` | Go | `internal/dashboard/dashboard.go`, invoked through the Go CLI | Preserve the eight-sheet dashboard scope and AND-within-path, OR-across-path derivation semantics. Structural connectivity is not automatically derivability. |
| `grind-bandit` | Python | `scripts/make_dashboard.py` | Label latent mastery as a model estimate. Keep observed attempts, uncertainty, forgetting, and recommendation evidence distinguishable. |
| `studyforge` | Python | `tools/make_dashboard.py` | Display the scheduler's actual plan and constraints. Readiness at a future exam is an estimate, not measured performance. |
| `deeparcade` | Python plus browser JavaScript game | `scripts/make_dashboard.py`; `scripts/make_game.py` generates `web/game.html` | Preserve actual policy outputs, evaluation counts and game controls. A single replay is illustrative and must not be presented as the agent's overall performance. |
| `wattwise` | Python | `scripts/make_dashboard.py` | Label seeded household and weather data as simulated. Savings must come from the existing optimizer and source tariff assumptions. |
| `freefood-radar` | Python | `scripts/make_dashboard.py`; `scripts/refresh_sbengaged.py` refreshes the public feed | Distinguish public SB Engaged records from classifier predictions and seeded evaluation charts. Do not claim food availability is verified. Retain feed freshness, source link, and withheld private-location behavior. |

## Keep design consistent

Each project has a different palette, typography, layout, and chart language. Read its existing `README.md`, dashboard, generator, design configuration, and `portfolio_ui/` package before editing. `portfolio_ui/` supplies the shared offline renderer and foundational behavior. Its shared code is infrastructure, not permission to replace a project's identity with a common template.

Keep these project identities:

- **Neural-CPPI:** Cormorant Garamond and IBM Plex Mono; obsidian, oxblood, antique brass, and warm ivory; institutional risk terminal.
- **FlowForge:** Newsreader and IBM Plex Mono; midnight navy, oxidized copper, and icy blue; scientific market-data observatory.
- **Toxicflow:** Bricolage Grotesque and JetBrains Mono; graphite, ember red, and industrial bronze; dense order-flow workstation.
- **Efficient Frontier FC:** Cinzel and IBM Plex Sans; forest green, ivory, and trophy gold; tactical squad room and football pitch.
- **SukukLab:** Fraunces and IBM Plex Mono; lapis, ivory, and antique gold; treasury and cash-flow atlas.
- **FormulaForge:** Crimson Pro and Fira Code; parchment, Oxford blue, and ink; mathematical codex and dependency map.
- **Grind-Bandit:** Unbounded and JetBrains Mono; dark iron, crimson, and gold; mastery arena and skill progression.
- **StudyForge:** Literata and IBM Plex Sans; vellum, sage, and walnut; academic planner and exam calendar.
- **DeepArcade:** Pirata One and Space Mono; dungeon green, amber, and restrained pixel details; playable agent bestiary.
- **WattWise:** Sora and IBM Plex Mono; deep pine, copper, and amber; household energy instruments.
- **Freefood-Radar:** Young Serif and DM Sans; terracotta, buttercream, and forest green; campus exploration map.

Use the bundled font files and licensing notes where present. Retain offline fallbacks. Avoid default system fonts as the primary face, generic repeated KPI grids, fake glass effects, decorative clutter, and motion that obscures chart reading. Preserve responsive layouts, accessible contrast, visible focus, keyboard use, reduced-motion support, clear labels and units, and readable charts at desktop and mobile widths.

## Evidence and data rules

- Trace every number to a committed data file, model output, tested calculation, or explicitly labeled public-feed record. Never invent values to fill a visual gap.
- Keep provenance close to the chart: identify the data period, experiment or simulation, units, sample size, and important assumptions.
- Preserve train, validation, and test boundaries. Do not recompute a held-out metric on training data to make a dashboard look better.
- Distinguish observation from estimate: forecasts, mastery, probabilities, bootstrap paths, synthetic samples, and seeded demos are not realized outcomes.
- Keep baselines and uncertainty visible. A zero observed failure count is not a proof of a universal guarantee.
- Prefer a faithful original visualization when a proposed new chart cannot be computed from available data.
- Make interactions change the displayed data or state. Remove controls that have no meaningful behavior; do not leave placeholder buttons.

## README contract

Keep the README detailed enough for an unfamiliar engineer to understand the project. Include a project-specific title banner, accurate CI and license badges, actual runtime and main technologies, a direct interactive demo link, a relevant source-backed animated GIF, a concise project introduction, a useful explanatory paragraph, exploration guidance, methodology, results and their interpretation, technical design or architecture, setup and test commands, data provenance, known limitations, and license attribution. Match the new README to the current dashboard and generator; never claim an unverified test pass or a live data source that is not running.

The banner should show the owner's name and use the project's design fonts. Use embedded font files or vector outlines so GitHub can render the design predictably. The demo animation should show changing graph data, a financial calculation, mastery trajectory, plan allocation, or game simulation relevant to that repository. Preserve several meaningful frames and ensure labels remain legible at the README's displayed width.

## Checks before calling an update complete

1. Rebuild the HTML using the repository's normal generator, where its dependencies and source inputs are available.
2. Open the generated HTML as a local file. Confirm its charts render from embedded or bundled assets and its primary interaction works.
3. Inspect a desktop viewport and a narrow mobile viewport. Look for clipping, overlap, axes that are difficult to read, missing legends, horizontal overflow, and low-contrast labels.
4. Compare displayed values to the source tables, arrays, or evaluation outputs. Check units, dates, denominators, sample counts, and labels.
5. Run the repository's relevant tests and record the exact command and outcome. If dependencies prevent a full run, report which check was run and what remains unverified.
6. Inspect the README locally and on GitHub. Verify the banner, animated image, result graphic, and demo link resolve. Confirm GitHub Pages displays the newly generated `docs/dashboard.html` after the repository's configured Pages workflow completes.
7. When a code or data change affects any README claim or graph, update those README assets in the same change.

## Updating the source and published page

`docs/dashboard.html` is the GitHub Pages artifact. It updates when the changed file is committed to the branch configured for Pages and the Pages build finishes. Some repositories generate this file in CI; others keep the generated output in the repository. Inspect `.github/workflows/`, repository Pages settings, and the project's generator before deciding the delivery path. A local file edit does not update GitHub or the live website by itself.

Use a focused change set containing the source or generator, generated dashboard if tracked or published by CI, relevant data animation or result figure, README text, and any tests or dependency changes. Review the final diff to make sure generated output corresponds to the same source revision. Preserve existing repository history and user changes. Do not expose tokens, browser cookies, campus login data, private event locations, or other credentials in source, logs, screenshots, or generated assets.

## Completion note template

When reporting an update, state which repository changed, which generator and source data produced the dashboard, which interactions were checked, the test command and result, whether the README animation and figure were refreshed, and any dependency, data, rendering, or Pages-build limitation. Separate measured results from model estimates and synthetic or demonstration data. Keep the note concise and factual.

## Portfolio changes already made

The following changes are already present in the redesigned portfolio. Use this list as context when extending the work. Preserve these improvements unless the new task explicitly changes them.

- **neural-cppi:** Rebuilt the dashboard as a flagship risk explorer with a large wealth-versus-floor view, classical versus learned policies, allocation and risk comparisons, and explicit scenario provenance. Improved chart text contrast, responsive chart labels, offline Plotly support, and the CI dashboard test. Added a source-data wealth animation and expanded the methodology and risk discussion in the README.
- **FlowForge:** Rebuilt the dashboard as an observatory centered on synchronized real and generated market sequences, with clear independent-sample labeling, distribution and temporal diagnostics, and a retained sample selector. Added an animated market-sequence figure and improved generator support, README results, and source descriptions.
- **toxicflow:** Reworked the dashboard around synchronized chronological price and VPIN views, clearer axes and legends, and useful linked chart inspection. Retained actual experimental performance and limitations. Added a data-backed trading animation and expanded the README and generator support.
- **efficient-frontier-fc:** Redesigned the squad frontier, budget interaction, tactical pitch, positional views, player register, best XI, and historical comparisons. Fixed player-name clipping and made constraints and wage-proxy limitations explicit. Replaced the GIF with a budget sweep showing the selected eleven and updated the README and generator.
- **sukuklab:** Added a treasury-style presentation for yield, price sensitivity, duration, convexity, cash flows, and structure comparisons using supported source observations. Clarified units, conventions, date context, and compliance limits. Kept Julia generation and added a reproducible Python presentation alternative, a changing yield-to-price animation, and expanded README material.
- **formulaforge:** Removed CSE 214 from the dashboard and preserved the eight intended sheets. Presented search, topic filtering, prerequisite exploration, and structural graph views with the distinction between graph reach and valid AND/OR derivation semantics. Updated the Go dashboard generator, README and graph animation.
- **grind-bandit:** Rebuilt the mastery view as a distinct progression interface showing recommendations, topic comparisons, uncertainty, forgetting and learning outcomes. Distinguished modeled mastery from observed attempts. Added a source-data animation comparing forgetting and solve/fail updates, then expanded the README and generator.
- **studyforge:** Redesigned the planner around the actual scheduled calendar, study-time allocation, exam priorities, estimated recall and forgetting curves. Kept engine constraints and marked future readiness as an estimate. Added an animated schedule/recall figure and updated README and generator outputs.
- **deeparcade:** Refined the dashboard and playable Snake game, embedded the game in the dashboard, and simplified game-over actions so restart appears over the play area. Removed the replay and direction-pad controls at the user's request. The README now includes a 175-frame recording from the existing trained DQN at seed 2036; the animation is an illustrative rollout, not a new evaluation. Updated dashboard and game generators.
- **wattwise:** Reorganized the energy dashboard around seeded household load, tariff timing, optimization, bill estimates, appliance contribution, weather context and scenario controls. Kept calculated savings and identified simulated data. Added an energy-profile animation and updated README and generator support.
- **freefood-radar:** Integrated the public SB Engaged iCalendar feed with a daily refresh workflow, feed status and source links. Separated public event records, model-ranked food predictions and historical classifier metrics. Withheld private locations and labeled seeded activity charts as demonstrations. Added tests, refresh tooling, a README animation of public model-ranked events, and an expanded README.

All projects received repository-specific typography, palettes, README banners, badges, source-backed result figures, and accessibility and responsive fixes. READMEs retain limitations, link the interactive demo directly, and explain methodology and provenance. The generated HTML and bundled renderer assets are marked as generated or vendored in `.gitattributes` so GitHub reports the actual research code language.
