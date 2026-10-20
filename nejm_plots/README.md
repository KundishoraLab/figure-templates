# nejm_plots

NEJM-style figure blocks for Matplotlib. Every block draws onto an Axes you give it, so you can combine them in any layout.

| Module | What it draws |
|---|---|
| `km.py` | Kaplan-Meier estimation (Greenwood, log-log CI), log-rank test, KM curves with CI bands and censor ticks, numbers-at-risk table (optionally with cumulative events), P value and HR text, direct curve labels |
| `forest.py` | Subgroup forest plot with striped rows, N and HR text columns, log axis, "X Better" arrows |
| `grotta.py` | mRS (or any ordinal) Grotta bars with dashed connectors and the stepped legend |
| `estimates.py` | Point estimates with CIs (single or grouped), trajectories with CI bands, "Difference at" brackets |
| `style.py` | rcParams, palette, inch-based layout and panel frames, broken y axis (`0 // 80 ... 100`) |

`recreate_figures.py` rebuilds the six reference figures into `out/`. Printed values (HRs, mRS distributions, MS estimates) are used directly. KM and trajectory panels use synthetic data shaped like the originals.

## KM with your own data

```python
import matplotlib.pyplot as plt
from nejm_plots import use_nejm_style, KMArm, plot_km, add_at_risk_table, add_km_pvalue, add_y_break, COLORS

use_nejm_style(10)
fig, ax = plt.subplots(figsize=(4.5, 3))
fig.subplots_adjust(left=0.2, bottom=0.42)

arms = [KMArm("Treated", df_t["time"], df_t["event"], COLORS["orange"]),
        KMArm("Control", df_c["time"], df_c["event"], COLORS["blue"])]
fits = plot_km(ax, arms, xlim=(0, 36), xticks=range(0, 37, 6), xlabel="Months")
add_at_risk_table(ax, fits, at=range(0, 37, 6))   # cumulative_events=True for "n (events)"
add_km_pvalue(ax, fits)                            # log-rank, formatted P<0.001 / P=0.04
# optional: add_y_break(ax, 0, 30) after setting ylim for a 0 // 30-100 axis
fig.savefig("km.svg")
```

`KMArm` options: `ci`, `ci_alpha`, `lw`, `censor_marker`, `censor_size`. `plot_km(..., fits=[...])` accepts precomputed `KMFit` objects if the curves come from elsewhere.

Run: `python recreate_figures.py --formats png svg`

## Added blocks

| Module | What it draws |
|---|---|
| `trajectory.py` | `fit_trajectory` (restricted cubic spline, cluster-robust CI when `ids` are given), `plot_trajectory_scatter` (raw points and/or per-participant lines behind the fit), `divergence_onset` (earliest time after which the two curves' difference CI excludes 0 for good) |
| `groups.py` | `plot_groups` (box, violin, bar or points only, sina jitter, paired lines per unit), `compare_groups` (Mann-Whitney, t, Welch, Wilcoxon, paired t, with Holm/BH/Bonferroni), `add_brackets` (stars only, non-significant pairs hidden) |
| `scatter.py` | `plot_correlation` (OLS line and CI, Spearman or Pearson returned, optional legend text), `plot_volcano` (labels stacked with leader lines), `plot_roc` (bootstrap AUC CI in the legend) |
| `km.py` | `plot_km(..., cumulative_incidence=True)` for 1 - KM curves |
| `style.py` | `auto_y_break` picks nice ticks and adds a 0 // break when the data sit far above zero |

`gallery.py` renders `fig6b_ad_trajectories_scatter` and `fig7_gallery` from synthetic data.

Trajectory with your data:

```python
fc = fit_trajectory(df_c.years, df_c.value, ids=df_c.pid)
fa = fit_trajectory(df_a.years, df_a.value, ids=df_a.pid)
plot_trajectory_scatter(ax, df_c.years, df_c.value, fc, ids=df_c.pid, color=COLORS["teal"], raw="points")
plot_trajectory_scatter(ax, df_a.years, df_a.value, fa, ids=df_a.pid, color=COLORS["red"], raw="both")
onset = divergence_onset(fa, fc)
add_difference_bracket(ax, onset, 0, f"Difference at {onset:.0f} yr", y=1.04)
```

## CAR T kinetics blocks (`cart_figure.py` -> fig8)

| Module | What it draws |
|---|---|
| `timecourse.py` | `segment_positions` + `segmented_time_axis` (Day / Week / Month brackets under a categorical time axis), `plot_time_strip` (points per timepoint with mean/median bar and SEM), `plot_time_line` (summary line, error bars both ways or up only), `summarize` / `geo_summary`, `plot_violin_outline` (unfilled, cut at data range, coloured median, dotted quartiles), `plot_bar_points` (open bars with SEM and points, dodge by passing offset positions), `bracket` (comparison bracket with any text, safe on log and broken axes) |
| `flow.py` | `plot_flow` (density-shaded events on a biexponential-like axis), `flow_axes`, `add_gate` (rectangle gate, `"auto"` label computes the % inside) |
| `style.py` | `add_y_break(..., upper_scale=0.25)` compresses the segment above the break, `log10_axis` (10^n labels), `sci_times_labels` (2.5x10^4 style) |
| `groups.py` | `plot_groups` NEJM solid boxes: `box_face="full", box_edge="black", median_color="black", whis=(0, 100), point_face="white", point_edge="black"` |

## Assembling figures: Theme, panel makers, Composite

Three layers, each usable on its own:

1. **Theme** (`assembly.py`) fixes every presentation number once: tick, axis-label, legend and panel-letter sizes, line and axis widths, tick length, marker size, the default margins around every axes box, gutters and row gaps. Presets: `Theme.paper()`, `Theme.nature()` (6.5 pt, lowercase letters), `Theme.nejm()` (8 pt, uppercase letters, grey frames). Override any field: `Theme.nature(font_size=6, line_lw=1.1)`.
2. **Panel makers** (`panels.py`) take data and return a `Panel`. Nothing is drawn until layout, and all sizes are read from the theme at that moment, so the same panel objects can be assembled under any theme.
3. **Composite** places panels row by row on a 12-column grid at a fixed figure width (`"single"` 89 mm, `"onehalf"` 120 mm, `"double"` 183 mm, `"nejm"` 7.1 in, or inches). In each row all axes boxes share the same height and the same top and bottom edges, panels of equal span get equal axes widths, and letters are assigned in reading order.

```python
from nejm_plots import Composite, Theme
from nejm_plots.panels import km_panel, roc_panel, groups_panel, paired_panel

P = {
    "km": km_panel(arms, xlim=(0, 14), xticks=range(0, 15, 2), cumulative_incidence=True),
    "roc": roc_panel(models, y_true, n_boot=500),
    "box": groups_panel({"Control": a, "Case": b}, ylabel="NfL (pg/ml)", brackets="all"),
    "pair": paired_panel({"Pre": pre, "Post": post}, ids=lesion_ids, ylabel="Score"),
}

theme = Theme.nature()
fig = Composite(width="double", theme=theme)
fig.row([(P["km"], 8), (P["roc"], 4)], height=1.6)      # (panel, span of 12)
fig.row([P["box"], P["pair"], None], height=1.5)        # even split, None = empty slot
fig.save("out/figure2", formats=("png", "svg", "pdf"))

P["km"].save("out/panels/km", width=4.2, height=1.6, theme=theme)   # any panel alone
```

| Maker | Draws |
|---|---|
| `km_panel` | KM or cumulative incidence, at-risk table (room reserved automatically), log-rank P, legend or direct labels, optional 0 // break |
| `roc_panel` | ROC curves, square axes, bootstrap AUC CI in legend |
| `groups_panel` | box / violin / bar / points, brackets (`"all"`, `"adjacent"` or pairs) with multiplicity correction, log axis, median/mean bars |
| `paired_panel` | one line per unit across conditions, median bars, Wilcoxon bracket |
| `volcano_panel` | volcano with stacked labels, optional Bonferroni guide |
| `correlation_panel` | scatter, fit, CI band, coefficient in legend |
| `trajectory_panel` | raw points behind spline fits with cluster-robust CI, computed divergence bracket |
| `forest_panel` | subgroup or multi-outcome forest; text column widths in inches |
| `grotta_panel` | ordinal distribution bars with legend |
| `point_range_panel` | estimates with CIs, grouped, optional break |
| `timecourse_panel` | Day / Week / Month axis with strips or summary lines |
| `violin_panel`, `bar_points_panel`, `flow_panel`, `legend_panel` | as named |
| `custom_panel(fn, ...)` | wraps any `fn(ax, ...)` you already have |

Writing a new maker: return `Panel(draw, margins, aspect, title, label)` where `draw(region)` calls `region.axes()` (or `region.grid(r, c)` / `region.inset(x, y, w, h)` for multi-axes panels) and reads sizes from `region.theme`. `margins` may be a dict or `fn(theme) -> dict` in inches; use `"left!"` etc. to force a side smaller than the theme default.

Every maker also accepts `title=` (shown after the letter), `label=` (`False` to suppress, or a fixed letter) and `margins=`. Values computed while drawing (P values, AUCs, fits, divergence onset) are kept on the region: `fig.regions[i].data`.

`compose_demo.py` builds 18 panels once, saves each alone to `out/panels/`, and assembles them into `composite_1`, `composite_2` and `composite_2_nejm`.
