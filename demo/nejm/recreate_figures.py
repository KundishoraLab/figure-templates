"""Recreate six NEJM figures with the reusable blocks in nejm_plots.

Values read off the published figures are used where they are printed (forest plot HRs,
mRS distributions, MS point estimates). Kaplan-Meier panels and the AD trajectories use
synthetic data shaped to resemble the originals, since patient-level data are not public.
Swap in your own arrays to reuse any panel.

Run:  python recreate_figures.py  [--out out] [--formats png svg]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nejm_plots import (COLORS, MRS_COLORS, ForestRow, InchLayout, KMArm, add_at_risk_table,
                        add_difference_bracket, add_km_annotation, add_km_pvalue,
                        add_y_break, band_legend, dot_legend, label_curves, mrs_legend,
                        plot_forest, plot_grotta, plot_km, plot_point_range,
                        plot_trajectory, use_nejm_style)

MINUS = "−"


# ======================================================================================
# Synthetic data helpers (only used for the demo)
# ======================================================================================
def sim_arm(rng, n, event_sampler, p_event, cens_lo, cens_hi, fixed_events=()):
    """Event times from `event_sampler` for a fraction p_event, uniform censoring.
    fixed_events: exact event times to include (for small arms with 1-3 events)."""
    n_free = n - len(fixed_events)
    ev = event_sampler(rng, n_free)
    cens = rng.uniform(cens_lo, cens_hi, n_free)
    has_ev = rng.random(n_free) < p_event
    t = np.where(has_ev, np.minimum(ev, cens), cens)
    e = (has_ev & (ev <= cens)).astype(int)
    t = np.concatenate([t, np.asarray(fixed_events, float)])
    e = np.concatenate([e, np.ones(len(fixed_events), int)])
    return t, e


def normal(mu, sd, lo=0.5):
    return lambda rng, n: np.clip(rng.normal(mu, sd, n), lo, None)


def mixture(*parts):
    """parts: (weight, sampler)"""
    def f(rng, n):
        w = np.array([p[0] for p in parts], float)
        idx = rng.choice(len(parts), n, p=w / w.sum())
        out = np.empty(n)
        for i, (_, s) in enumerate(parts):
            m = idx == i
            out[m] = s(rng, m.sum())
        return out
    return f


def uniform(a, b):
    return lambda rng, n: rng.uniform(a, b, n)


def piecewise_exp(rng, n, cuts, rates):
    """Event times from a piecewise-constant hazard."""
    u = -np.log(rng.random(n))           # target cumulative hazard
    t = np.full(n, np.inf)
    H0, edges = 0.0, list(cuts) + [np.inf]
    for (a, b), r in zip(zip([0] + list(cuts), edges), rates):
        m = np.isinf(t) & (u <= H0 + r * (b - a))
        t[m] = a + (u[m] - H0) / r
        H0 += r * (b - a) if np.isfinite(b) else 0
    return t


def save(fig, out, name, formats):
    for fmt in formats:
        fig.savefig(out / f"{name}.{fmt}", dpi=200)
    plt.close(fig)


# ======================================================================================
# Figure 1: arsa-cel MLD, 3 x 3 KM grid with at-risk tables
# ======================================================================================
def fig_mld_km(out, formats):
    use_nejm_style(12)
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(14.6, 11.5))
    L = InchLayout(fig)
    O, B = COLORS["orange"], COLORS["blue"]

    # panel frames
    L.frame(0.03, 0.03, 14.54, 11.44)
    L.hline(0.03, 14.57, 0.55)
    col_w = 14.54 / 3
    for c in (1, 2):
        L.vline(0.03 + c * col_w, 0.55, 11.47)

    fig.legend([Line2D([], [], color=O, lw=2), Line2D([], [], color=B, lw=2)],
               ["Arsa-cel", "Untreated"], loc="center", ncol=2,
               bbox_to_anchor=(0.5, 1 - 0.29 / 11.5), handlelength=1.6, columnspacing=2)

    titles = ["Presymptomatic Late-Infantile MLD", "Presymptomatic Early-Juvenile MLD",
              "Early-Symptomatic Early-Juvenile MLD"]
    col_titles = [("A", "Severe Motor Impairment–free Survival"),
                  ("B", "Motor Impairment–free Survival"), ("C", "Overall Survival")]
    rows_x = [(14, 1), (12, 1), (20, 2)]
    pvals = [["P<0.001", None, "P<0.001"], ["P=0.04", None, None], ["P<0.001", None, None]]

    # (arsa-cel, untreated) data for [row][col]
    data = [
        [  # late-infantile
            (sim_arm(rng, 18, normal(99, 1), 0, 1.9, 13.4, [7.2]),
             sim_arm(rng, 28, normal(2.75, 0.42, 1.8), 0.97, 2.5, 6)),
            (sim_arm(rng, 18, normal(99, 1), 0, 2.6, 12.9, [4.2]),
             sim_arm(rng, 26, normal(2.45, 0.35, 1.6), 0.97, 2.5, 6)),
            (sim_arm(rng, 18, normal(99, 1), 0, 2.7, 13.4),
             sim_arm(rng, 28, mixture((0.55, normal(5.4, 0.9, 3.5)), (0.45, uniform(10, 14))),
                     0.8, 5, 13.5)),
        ],
        [  # presymptomatic early-juvenile
            (sim_arm(rng, 8, normal(99, 1), 0, 1.5, 11.9, [2.0]),
             sim_arm(rng, 21, normal(6.9, 1.6, 4.8), 0.92, 2.5, 11)),
            (sim_arm(rng, 8, normal(99, 1), 0, 1.5, 11.9, [2.0]),
             sim_arm(rng, 17, normal(6.6, 1.4, 2.6), 0.95, 3, 11)),
            (sim_arm(rng, 8, normal(99, 1), 0, 2.5, 11.9, [2.0]),
             sim_arm(rng, 21, normal(99, 1), 0, 3, 12, [9.4, 10.3])),
        ],
        [  # early-symptomatic early-juvenile
            (sim_arm(rng, 11, normal(99, 1), 0, 1, 19, [6.6, 12.5, 15.3]),
             sim_arm(rng, 21, normal(6.8, 1.4, 4.8), 0.95, 1.5, 11)),
            (sim_arm(rng, 11, normal(99, 1), 0, 1, 16.3, [6.5, 9.9, 10.4, 10.8, 15.8]),
             sim_arm(rng, 17, normal(6.2, 1.3, 2.5), 0.95, 1.5, 11)),
            (sim_arm(rng, 11, normal(99, 1), 0, 1, 19, [6.6]),
             sim_arm(rng, 21, normal(99, 1), 0, 2, 19, [9.3, 10.3, 17.0])),
        ],
    ]

    row_h = 3.48
    for c in range(3):
        cl = 0.03 + c * col_w
        L.panel_heading(cl + 0.15, 0.85, *col_titles[c], size=14)
        for r in range(3):
            top = 1.05 + r * row_h
            ax_left, ax_w = cl + 1.18, 3.42
            ax = L.axes(ax_left, top + 0.38, ax_w, 1.62)
            L.text(ax_left + ax_w / 2, top + 0.12, titles[r], ha="center", fontsize=13,
                   fontweight="semibold")
            xmax, step = rows_x[r]
            ticks = np.arange(0, xmax + 1, step)
            (ta, ea), (tu, eu) = data[r][c]
            fits = plot_km(ax, [KMArm("Arsa-cel", ta, ea, O), KMArm("Untreated", tu, eu, B)],
                           xlim=(0, xmax), xticks=ticks, xlabel="Age (yr)")
            add_at_risk_table(ax, fits, at=ticks, label_x=(cl + 0.2 - ax_left) / ax_w,
                              title_offset=50, row_height=20)
            if pvals[r][c]:
                add_km_pvalue(ax, text=pvals[r][c], xy=(0.025, 0.03))
    save(fig, out, "fig1_mld_km_grid", formats)


# ======================================================================================
# Figure 2: rituximab vs ocrelizumab point-range panels
# ======================================================================================
def fig_ms_pointrange(out, formats):
    use_nejm_style(11)
    fig = plt.figure(figsize=(7.0, 9.15))
    L = InchLayout(fig)
    L.frame(0.06, 0.03, 6.9, 9.08)
    L.vline(3.39, 0.03, 9.11)
    L.hline(0.06, 6.96, 4.53)
    dk, lt = COLORS["dark_blue"], COLORS["light_blue"]

    # A: risk differences
    L.panel_heading(0.13, 0.14, "A", "Risk Differences for No New or\nEnlarging Lesions on "
                    "T2-Weighted\nMRI", size=12.5, gap=0.3)
    ax = L.axes(1.05, 0.94, 2.12, 2.9)
    plot_point_range(ax, ["Baseline to\nMonth 6", "Baseline to\nMonth 24"],
                     {"RD": ([-2, 1], [-13, -11], [10, 13])}, colors=["black"], zero_line=0,
                     ms=6, capsize=4.5, lw=1.1)
    ax.set_xlim(-0.48, 1.52)
    ax.set_ylim(-20, 20)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_ylabel("Risk Difference (percentage points)")

    # B: annualized relapse rate
    L.panel_heading(3.47, 0.25, "B", "Annualized Relapse Rate", size=12.5, gap=0.3)
    ax = L.axes(4.38, 0.93, 2.12, 2.85)
    plot_point_range(ax, ["Rituximab", "Ocrelizumab"], {"ARR": ([0.09, 0.04], [0, 0], [0.20, 0.10])},
                     point_colors=[dk, lt], capsize=5)
    ax.set_xlim(-0.43, 1.53)
    ax.set_ylim(0, 0.25)
    ax.set_yticks(np.arange(0, 0.26, 0.05), [f"{v:.2f}" for v in np.arange(0, 0.26, 0.05)])
    ax.set_ylabel("No. of Relapses per Year")

    # C: freedom from relapse (broken axis 0 // 80-100)
    L.panel_heading(0.13, 4.75, "C", "Freedom from Relapse", size=12.5, gap=0.3)
    ax = L.axes(1.02, 5.49, 2.13, 2.85)
    plot_point_range(ax, ["Rituximab", "Ocrelizumab"], {"Free": ([91.7, 94], [85, 89], [99, 100])},
                     point_colors=[dk, lt], capsize=5)
    ax.set_xlim(-0.47, 1.49)
    ax.set_ylim(0, 100)
    add_y_break(ax, 0, 80, gap_frac=0.16)
    ax.set_yticks([0, 80, 85, 90, 95, 100])
    ax.set_ylabel("Percentage of Participants")

    # D: disability and cognitive end points (grouped, 0-40 // 100)
    L.panel_heading(3.47, 4.75, "D", "Disability and Cognitive End Points", size=12.5, gap=0.3)
    ax = L.axes(4.36, 5.49, 2.32, 2.85)
    plot_point_range(ax, ["6M-CDP", "6M-CDI", "SDMT\nWorsening"],
                     {"Rituximab": ([3, 29, 2.3], [0, 16, 0], [7, 40, 6]),
                      "Ocrelizumab": ([7, 24, 3.6], [2, 11, 0], [14, 33, 9])},
                     colors=[dk, lt], offset=0.19, ms=6.5, capsize=4.5)
    ax.set_xlim(-0.43, 2.43)
    ax.set_ylim(0, 100)
    add_y_break(ax, 40, 100, gap_frac=0.16)
    ax.set_yticks([0, 10, 20, 30, 40, 100])
    ax.set_ylabel("Percentage of Participants")
    dot_legend(ax, ["Rituximab", "Ocrelizumab"], [dk, lt], bbox=(0.5, 1.2))
    save(fig, out, "fig2_ms_point_range", formats)


# ======================================================================================
# Figure 3: single Grotta bar (EVT + usual care vs usual care)
# ======================================================================================
def fig_mrs_single(out, formats):
    use_nejm_style(17)
    fig = plt.figure(figsize=(10.0, 4.38))
    L = InchLayout(fig)
    L.frame(0.02, 0.0, 9.96, 4.33)
    ax = L.axes(2.39, 1.15, 7.13, 1.95)
    plot_grotta(ax, [[17.6, 23.9, 12.5, 18.0, 11.0, 3.5, 13.3],
                     [13.9, 29.2, 15.7, 19.0, 9.5, 4.4, 8.4]],
                ["EVT+Usual Care\n(N=255)", "Usual Care\n(N=274)"],
                bar_height=0.71, label_fontsize=21, connector_kw=dict(lw=1.8, ls=(0, (2.5, 2))))
    ax.set_ylim(-0.523, 1.533)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.tick_params(axis="x", width=1.6, length=7)
    ax.xaxis.label.set_size(20)
    mrs_legend(ax, bbox=(0.5, 1.06), title_fontsize=21, fontsize=19, columnspacing=1.3)
    save(fig, out, "fig3_mrs_grotta_single", formats)


# ======================================================================================
# Figure 4: vutrisiran KM (with cumulative events) + subgroup forest plots
# ======================================================================================
def fig_vutrisiran(out, formats):
    use_nejm_style(11)
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(13.5, 9.1))
    L = InchLayout(fig)
    L.frame(0.02, 0.0, 13.46, 9.07)
    L.vline(6.7, 0.0, 9.07)
    L.hline(0.02, 13.48, 4.15)
    O, B = COLORS["orange"], COLORS["blue"]

    def trial_arm(n, late_rate):
        ev = piecewise_exp(rng, n, [6], [0.0157, late_rate])
        admin = rng.uniform(33, 38.3, n)
        drop = rng.exponential(400, n)
        cens = np.minimum(admin, drop)
        return np.minimum(ev, cens), (ev <= cens).astype(int)

    km_specs = [
        ("A", "Time to First Event in the Overall Population", 0.02,
         trial_arm(326, 0.0139), trial_arm(328, 0.0200),
         ["Hazard ratio for primary end point, 0.72 (95% CI, 0.56–0.93); P=0.01",
          "Hazard ratio for time to first event, 0.72 (95% CI, 0.57–0.91); P=0.006"],
         {"Vutrisiran": (33.0, 71.0), "Placebo": (33.3, 42.0)}),
        ("B", "Time to First Event in the Monotherapy Population", 6.7,
         trial_arm(196, 0.0141), trial_arm(199, 0.0235),
         ["Hazard ratio for primary end point, 0.67 (95% CI, 0.49–0.93); P=0.02",
          "Hazard ratio for time to first event, 0.64 (95% CI, 0.48–0.87); P=0.004"],
         {"Vutrisiran": (33.2, 71.0), "Placebo": (33.5, 39.5)}),
    ]
    ticks = [0, 6, 12, 18, 24, 30, 36, 39]
    for letter, title, x0, (tv, ev), (tp, ep), hr, labs in km_specs:
        L.panel_heading(x0 + 0.1, 0.16, letter, title, size=11.5, gap=0.25)
        ax = L.axes(x0 + 1.22, 0.45, 5.06, 2.38)
        fits = plot_km(ax, [KMArm("Vutrisiran", tv, ev, O, ci=False, lw=1.4, censor_size=5,
                                  censor_mew=0.8),
                            KMArm("Placebo", tp, ep, B, ci=False, lw=1.4, censor_size=5,
                                  censor_mew=0.8)],
                       xlim=(0, 39), xticks=ticks, xlabel="Months since First Dose",
                       ylabel="Percentage of Patients Free\nfrom Event")
        add_y_break(ax, 0, 30, gap_frac=0.11)
        ax.set_yticks([0, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.yaxis.label.set_linespacing(1.1)
        label_curves(ax, labs, fontsize=12)
        add_km_annotation(ax, hr, xy=(0.025, 0.035), fontsize=11, linespacing=1.3)
        add_at_risk_table(ax, fits, at=[0, 6, 12, 18, 24, 30, 36, 39], cumulative_events=True,
                          title="No. at Risk", subtitle="(cumulative no. of events)",
                          label_x=(x0 + 0.18 - (x0 + 1.22)) / 5.06, title_offset=46,
                          row_height=17, fontsize=10.5,
                          x_positions=[0, 6, 12, 18, 24, 30, 35.4, 39.6])

    # ---- forest plots -------------------------------------------------------------
    def R(label, n=None, est=None, lo=None, hi=None, indent=0):
        return ForestRow(label, n, est, lo, hi, indent)

    rows_c = [R("Overall", 654, .72, .56, .93), R("Age"), R("<75 yr", 257, .55, .35, .85, 1),
              R("≥75 yr", 397, .81, .58, 1.11, 1), R("Tafamidis use at baseline"),
              R("No", 395, .67, .49, .93, 1), R("Yes", 259, .79, .51, 1.21, 1),
              R("ATTR disease type"), R("Variant", 76, .92, .49, 1.72, 1),
              R("Wild type", 578, .67, .51, .90, 1), R("NYHA class"),
              R("I or II", 592, .73, .55, .96, 1), R("III", 62, .68, .33, 1.41, 1),
              R("Baseline NT-proBNP level"), R("≤2000 pg/ml", 342, .53, .35, .79, 1),
              R(">2000 pg/ml", 312, .80, .56, 1.13, 1)]
    rows_d = [R("Overall", 395, .67, .49, .93), R("Age"), R("<75 yr", 153, .53, .32, .88, 1),
              R("≥75 yr", 242, .72, .47, 1.10, 1), R("ATTR disease type"),
              R("Variant", 48, .67, .31, 1.44, 1), R("Wild type", 347, .66, .45, .95, 1),
              R("NYHA class"), R("I or II", 368, .73, .53, 1.02, 1),
              R("III", 27, .31, .09, 1.02, 1), R("Baseline NT-proBNP level"),
              R("≤2000 pg/ml", 188, .50, .28, .92, 1), R(">2000 pg/ml", 207, .71, .47, 1.07, 1)]

    forest_specs = [
        ("C", "Subgroup Analyses of the Primary End Point (overall population)", 0.02, rows_c,
         2.47, 2.87, (0.235, 3.3), (0.25, 0.5, 1.0, 2.0), "{:.2f}",
         dict(label=-0.77, n=-0.06, est=1.38, h_label=-0.77, h_n=-0.275, h_est=0.588,
              h_est_ha="left")),
        ("D", "Subgroup Analyses of the Primary End Point (monotherapy population)", 6.7, rows_d,
         2.47, 3.33, (0.089, 2.75), (0.125, 0.25, 0.5, 1.0, 2.0), "{:.3f}",
         dict(label=-0.68, n=-0.04, est=1.20, h_label=-0.68, h_n=-0.23, h_est=0.54,
              h_est_ha="left")),
    ]
    for letter, title, x0, rows, ax_l, ax_w, xlim, xt, tfmt, cols in forest_specs:
        L.panel_heading(x0 + 0.1, 4.35, letter, title, size=11.5, gap=0.25)
        h = (len(rows) - 1 + 1.2) * 0.205
        ax = L.axes(x0 + ax_l, 4.78, ax_w, h)
        plot_forest(ax, rows, xlim=xlim, xticks=xt, xticklabels=[tfmt.format(t) for t in xt],
                    cols=cols, better=("Vutrisiran Better", "Placebo Better"),
                    shade_extent=(cols["label"] - 0.02, cols["est"] + 0.02), fontsize=11,
                    arrow_offset=30, better_offset=46, ms=5, header_gap=1.25)
    save(fig, out, "fig4_vutrisiran_km_forest", formats)


# ======================================================================================
# Figure 5: three stacked Grotta panels
# ======================================================================================
def fig_mrs_three(out, formats):
    use_nejm_style(10.5)
    fig = plt.figure(figsize=(6.5, 9.25))
    L = InchLayout(fig)
    L.frame(0.25, 0.03, 6.14, 9.12)
    for y in (0.67, 3.47, 6.29):
        L.hline(0.25, 6.39, y)
    leg_ax = L.axes(0.25, 0.05, 6.14, 0.6)
    leg_ax.axis("off")
    mrs_legend(leg_ax, title="Score on the Modified Rankin Scale at 90 Days", loc="center",
               bbox=(0.5, 0.5), columnspacing=1.0, handlelength=1.1, fontsize=11,
               title_fontsize=11.5)

    panels = [
        ("A", "Undifferentiated Stroke",
         [[13.8, 19.0, 7.9, 14.0, 11.1, 11.6, 22.5], [16.1, 16.3, 6.3, 14.8, 10.9, 13.0, 22.6]],
         ["Intervention\n(N=1205)", "Usual Care\n(N=1199)"]),
        ("B", "Ischemic Stroke",
         [[18.6, 19.6, 7.9, 12.2, 8.8, 10.5, 22.5], [21.8, 19.9, 8.4, 13.0, 7.9, 11.8, 17.1]],
         ["Intervention\n(N=599)", "Usual Care\n(N=600)"]),
        ("C", "Hemorrhagic Stroke",
         [[3.9, 18.1, 8.3, 16.9, 14.2, 14.4, 24.3], [5.5, 11.4, 3.9, 17.8, 15.0, 15.6, 30.8]],
         ["Intervention\n(N=522)", "Usual Care\n(N=519)"]),
    ]
    for i, (letter, title, dat, labs) in enumerate(panels):
        off = i * 2.82
        L.panel_heading(0.35, 0.87 + off, letter, title, size=12, gap=0.25)
        ax = L.axes(1.38, 1.07 + off, 4.69, 1.63)
        plot_grotta(ax, dat, labs, bar_height=0.45, label_fontsize=11,
                    connector_kw=dict(lw=0.8, ls=(0, (3, 2.5))))
        ax.set_ylim(-0.56, 1.54)
        ax.xaxis.label.set_size(11)
    save(fig, out, "fig5_mrs_grotta_three", formats)


# ======================================================================================
# Figure 6: AD biomarker trajectories (7 panels)
# ======================================================================================
def fig_ad_trajectories(out, formats):
    use_nejm_style(9)
    fig = plt.figure(figsize=(6.2, 9.24))
    L = InchLayout(fig)
    L.frame(0.05, 0.03, 6.12, 9.1)
    L.vline(3.11, 0.03, 9.13)
    for y in (2.29, 4.56, 6.83):
        L.hline(0.05, 6.17, y)

    x = np.linspace(-21, 0, 200)

    def quad(p_start, p_mid, p_end, mid=-10):
        c = np.polyfit([-21, mid, 0], [p_start, p_mid, p_end], 2)
        return np.polyval(c, x)

    cdr_ad = 0.12 + 4.78 * np.exp(x / 2.85) + 0.33 * np.exp(-(x + 21) / 1.8)
    panels = [
        # letter, row, col, ctrl, ad, half-band, ylim, break(lo,hi) or None, yticks, label, bracket
        ("A", 0, 0, quad(718, 712, 645), quad(712, 560, 412), 14, (0, 800), (0, 300),
         [0, 300, 400, 500, 600, 700, 800], r"CSF A$\beta_{42}$ (pg/ml)", (-18, 0.08)),
        ("B", 1, 0, quad(.111, .110, .101), quad(.110, .088, .066), .0025, (0, .15), None,
         [0, .05, .10, .15], r"Ratio of CSF A$\beta_{42}$ to A$\beta_{40}$", (-14, 0.1)),
        ("C", 1, 1, quad(51, 52, 73), quad(44, 75, 134), 2.2, (0, 140), (0, 40),
         [0, 40, 60, 80, 100, 120, 140], "CSF p-tau 181 (pg/ml)", (-11, 0.08)),
        ("D", 2, 0, quad(214, 240, 295), quad(188, 320, 540), 11, (0, 600), (0, 100),
         [0, 100, 200, 300, 400, 500, 600], "CSF t-tau (pg/ml)", (-10, 0.08)),
        ("E", 2, 1, quad(620, 720, 850), quad(585, 850, 1560), 30, (0, 2000), (0, 500),
         [0, 500, 1000, 1500, 2000], "CSF NfL (pg/ml)", (-9, 0.08)),
        ("F", 3, 0, quad(7720, 7400, 7300), quad(7650, 7230, 6300), 75, (0, 8000), (0, 5000),
         [0, 5000, 6000, 7000, 8000], r"Hippocampal Volume (mm$^3$)", (-8, 0.08)),
        ("G", 3, 1, np.full_like(x, 0.16) + 0.0015 * (x + 21), cdr_ad, .06, (0, 6), None,
         list(range(7)), "CDR-SB Score", (-6, 0.92)),
    ]
    red, red_b, teal, teal_b = COLORS["red"], COLORS["red_band"], COLORS["teal"], COLORS["teal_band"]
    for letter, r, c, ctrl, ad, hb, ylim, brk, yt, ylab, (bx, by) in panels:
        cell_x, cell_y = 0.05 + c * 3.06, 0.03 + r * 2.27
        L.text(cell_x + 0.1, cell_y + 0.13, letter, fontsize=10.5, fontweight="bold")
        ax = L.axes(cell_x + 0.67 + (0.0 if c == 0 else 0.0), cell_y + 0.39, 2.21, 1.4)
        plot_trajectory(ax, x, ctrl, ctrl - hb, ctrl + hb, color=teal, band_color=teal_b)
        plot_trajectory(ax, x, ad, ad - hb, ad + hb, color=red, band_color=red_b)
        ax.set_xlim(-21.6, 0.3)
        ax.set_xticks([-20, -15, -10, -5, 0])
        ax.set_ylim(*ylim)
        if brk:
            add_y_break(ax, *brk, gap_frac=0.16, mark_size=8)
        ax.set_yticks(yt)
        if letter == "B":
            ax.set_yticklabels([f"{v:.2f}" for v in yt])
        ax.set_ylabel(ylab)
        ax.set_xlabel("Years before Diagnosis")
        ax.tick_params(labelsize=8.5)
        add_difference_bracket(ax, bx, 0, f"Difference at {MINUS}{abs(bx)} yr", y=by,
                               fontsize=8.5)

    leg_ax = L.axes(3.16, 0.6, 2.9, 0.6)
    leg_ax.axis("off")
    band_legend(leg_ax, ["Cognitively normal matched controls",
                         "Participants with diagnosis of\nAlzheimer’s disease"],
                [teal, red], [teal_b, red_b], loc="center left", bbox_to_anchor=(0.0, 0.5),
                fontsize=9)
    save(fig, out, "fig6_ad_trajectories", formats)


# ======================================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--formats", nargs="+", default=["png", "svg"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_mld_km, fig_ms_pointrange, fig_mrs_single, fig_vutrisiran, fig_mrs_three,
               fig_ad_trajectories):
        fn(out, a.formats)
        print("done:", fn.__name__)
