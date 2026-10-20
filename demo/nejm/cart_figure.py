"""Recreate the NEJM CAR T cell kinetics figure (panels A-I) with synthetic data.

New blocks used: plot_violin_outline, segmented_time_axis, plot_time_strip,
plot_time_line, plot_bar_points, bracket, log10_axis, sci_times_labels, plot_flow,
add_gate, and plot_groups with NEJM solid boxes.

Run:  python cart_figure.py [--out out] [--formats png svg]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nejm_plots import (InchLayout, add_gate, add_y_break, bracket, log10_axis, plot_bar_points,
                        plot_flow, plot_groups, plot_time_line, plot_time_strip,
                        plot_violin_outline, sci_times_labels, segment_positions,
                        segmented_time_axis, summarize, use_nejm_style)

BLUE = "#3A7DC0"
RED = "#C8442F"
GREEN = "#3C9A5F"
ORANGE_BAR = "#D4613E"
BOX_BLUE, BOX_YEL, BOX_RED = "#5B7FC7", "#E3B740", "#D4553A"
EN = "–"
TIMES = r"$\times$"


def lognorm_mean(rng, mean, n, sigma=1.0, zero_frac=0.0):
    """Skewed non-negative values with roughly the requested mean, some exact zeros."""
    mu = np.log(max(mean, 1e-3)) - sigma**2 / 2
    v = rng.lognormal(mu, sigma, n)
    v[rng.random(n) < zero_frac] = 0
    return v


def build(out, formats):
    use_nejm_style(8)
    rng = np.random.default_rng(21)
    fig = plt.figure(figsize=(7.1, 9.13))
    L = InchLayout(fig)
    L.frame(0.03, 0.03, 7.05, 9.07)
    L.vline(3.58, 0.03, 7.37)
    for y in (1.41, 3.13, 4.88, 7.37):
        L.hline(0.03, 7.08, y)
    hs = 8.6

    # ---- A: violins by dose ----------------------------------------------------------
    L.panel_heading(0.15, 0.13, "A", "Peak Levels on Flow Cytometry, According to Dose", size=hs, gap=0.2)
    ax = L.axes(0.8, 0.28, 2.33, 0.77)
    dose_labels = [f"3{TIMES}10$^6$", f"6{TIMES}10$^6$", f"10{TIMES}10$^6$"]
    vals = [np.r_[rng.gamma(2.5, 8, 6), 68], np.r_[rng.normal(140, 50, 11), 245, 30],
            np.r_[rng.normal(175, 70, 9), 300, 2]]
    vals = [np.clip(v, 0, None) for v in vals]
    plot_violin_outline(ax, vals, width=0.75, median_color=GREEN, labels=dose_labels)
    ax.set_xlim(-0.67, 2.25)
    ax.set_ylim(0, 300)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel("CAR T Cells/mm$^3$")
    ax.set_xlabel("CAR T Cells/kg")
    ax.tick_params(axis="x", length=0)

    # ---- B: violins of transgene copies ----------------------------------------------
    L.panel_heading(3.65, 0.13, "B", "Vector Transgene Copies on ddPCR, According to Dose", size=hs, gap=0.2)
    ax = L.axes(4.53, 0.27, 2.27, 0.78)
    vals = [[400, 10000, 16200], [8500, 14500, 23000], [16200, 16700, 17000]]
    plot_violin_outline(ax, vals, width=0.6, median_color="black",
                        quartile_ls=(0, (3, 2)), labels=dose_labels, bw=0.9)
    for i, v in enumerate(vals):
        ax.scatter(np.full(3, i), v, s=10, color="black", zorder=4)
    ax.set_xlim(-0.59, 2.5)
    ax.set_ylim(0, 25000)
    yt = np.arange(0, 25001, 5000)
    ax.set_yticks(yt, sci_times_labels(yt, exp=4))
    ax.set_ylabel("No. of\nCopies/100\nng of DNA", linespacing=1.0)
    ax.set_xlabel("CAR T cells/kg")
    ax.tick_params(axis="x", length=0)

    # ---- C: circulating levels, strip + mean +/- SEM, broken axis --------------------
    L.panel_heading(0.15, 1.55, "C", "Circulating Levels, According to Time since Infusion", size=hs, gap=0.2)
    ax = L.axes(0.8, 1.68, 2.37, 1.07)
    pos, spans = segment_positions([2, 6, 6], gap=0.25)
    labs = ["0", f"3{EN}5", "1", "2", "4", "6", "8", "12", "6", "9", "12", "18", "24", "30"]
    means = [0.5, 3, 50, 100, 40, 15, 20, 12, 25, 15, 18, 15, 18, 8]
    ns = [24, 24, 26, 26, 25, 22, 20, 18, 10, 8, 7, 6, 5, 4]
    data_c = [lognorm_mean(rng, m, n, 1.2, 0.15) for m, n in zip(means, ns)]
    data_c[1][:3] = [580, 300, 280]
    data_c[0][:] = rng.uniform(0, 2, ns[0])
    data_c[3][:2] = [920, 300]
    plot_time_strip(ax, pos, data_c, color=BLUE, s=9, jitter=0.2, center_width=0.55)
    ax.set_ylim(0, 1000)
    add_y_break(ax, 300, 600, gap_frac=0.11, mark_size=8, upper_scale=0.24)
    ax.set_yticks([0, 100, 200, 300, 600, 800, 1000])
    ax.set_ylabel("CAR T Cells/mm$^3$")
    segmented_time_axis(ax, pos, labs, [("Day", *spans[0]), ("Week", *spans[1]),
                                        ("Month", *spans[2])], pad_x=0.5)

    # ---- D: percentages, thick median bar --------------------------------------------
    L.panel_heading(3.65, 1.55, "D", "Levels in Peripheral Blood, According to Time since Infusion", size=hs, gap=0.2)
    ax = L.axes(4.53, 1.68, 2.37, 1.07)
    pos, spans = segment_positions([2, 7, 7], gap=0.25)
    labs = ["0", f"3{EN}5", "1", "2", "4", "6", "8", "10", "12", "6", "9", "12", "15", "18", "24", "30"]
    meds = [0, 3, 25, 35, 10, 5, 5, 3, 4, 5, 2, 3, 4, 3, 1, 1]
    ns = [24, 24, 26, 26, 24, 22, 20, 18, 16, 10, 8, 7, 6, 5, 4, 3]
    data_d = [np.clip(lognorm_mean(rng, m + 0.5, n, 1.0, 0.1), 0, 95) for m, n in zip(meds, ns)]
    plot_time_strip(ax, pos, data_d, color="black", s=8, jitter=0.18, center="median", err=None,
                    center_color=ORANGE_BAR, center_lw=4.5, center_width=0.75, zorder=2)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Percentage of\nCD3-Positive Cells", linespacing=1.05)
    segmented_time_axis(ax, pos, labs, [("Day", *spans[0]), ("Week", *spans[1]),
                                        ("Month", *spans[2])], pad_x=0.5)

    # ---- E: transgene copies over time (summary-level input, log axis) ----------------
    L.panel_heading(0.15, 3.27, "E", "Vector Transgene Copies, According to Time since Infusion", size=hs, gap=0.2)
    ax = L.axes(0.83, 3.41, 2.33, 1.04)
    pos, spans = segment_positions([1, 8, 7], gap=0.25)
    labs = ["Day 0", "1", "2", "3", "4", "6", "8", "10", "12", "6", "9", "12", "15", "18", "24", "30"]
    gm = np.array([60, 12000, 9500, 6000, 2200, 1800, 1200, 750, 800, 500, 550, 330, 600, 900, 150, 170])
    up = gm * np.array([.25, .15, .15, .15, .2, .2, .2, .2, .2, .35, .3, .35, .5, .65, .1, .2])
    plot_time_line(ax, pos, gm, up, direction="up", color="black", lw=1.1, capsize=2.5)
    log10_axis(ax, 1, 5)
    ax.set_ylabel("No. of Copies/\n100 ng of DNA", linespacing=1.05)
    segmented_time_axis(ax, pos, labs, [("Week", *spans[1]), ("Month", *spans[2])],
                        rotate_first=45, pad_x=0.5)

    # ---- F: two response groups over time --------------------------------------------
    L.panel_heading(3.65, 3.27, "F", "Circulating Levels, According to Response and Time since Infusion", size=hs, gap=0.2)
    ax = L.axes(4.53, 3.41, 2.37, 1.04)
    pos, spans = segment_positions([2, 6, 6], gap=0.25)
    labs = ["0", f"3{EN}5", "1", "2", "4", "6", "8", "12", "6", "9", "12", "18", "24", "30"]
    cr = np.array([0, 5, 70, 158, 17, 17, 25, 18, 38, 27, 25, 22, 18, 10])
    cr_e = np.array([0, 3, 38, 98, 3, 4, 12, 4, 17, 8, 8, 9, 8, 3])
    ncr = np.array([0, 3, 42, 70, 57, 16, 17, 15, 3, 2, 3])
    ncr_e = np.array([0, 2, 12, 15, 15, 4, 5, 4, 1, 1, 1])
    plot_time_line(ax, pos[:len(ncr)], ncr, ncr_e, color=RED, marker="o", ms=3.5, lw=1.0,
                   capsize=2, label="No complete response")
    plot_time_line(ax, pos, cr, cr_e, color=BLUE, marker="o", ms=3.5, lw=1.0, capsize=2,
                   label="Complete response")
    ax.set_ylim(0, 300)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel("CAR T Cells/mm$^3$")
    segmented_time_axis(ax, pos, labs, [("Day", *spans[0]), ("Week", *spans[1]),
                                        ("Month", *spans[2])], pad_x=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.06), handlelength=1.4,
              labelspacing=0.2, fontsize=8)

    # ---- G: cytokines by CRS grade, log boxes with P brackets ------------------------
    L.panel_heading(0.15, 5.0, "G", "Peak Levels of Cytokines, According to CRS Grade", size=hs, gap=0.2)
    grades = ["G0", "G1", "G2–3"]
    cyt = [
        ("Interleukin-6\n(pg/ml)", (0.78, 5.45, 0.65), (0, 4), [5, 80, 900], [7, 13, 5],
         [(0, 1, 10**2.4, "P<0.001"), (1, 2, 10**3.4, "P<0.01"), (0, 2, 10**4.4, "P<0.01")]),
        ("Interferon-γ\n(pg/ml)", (2.38, 5.45, 0.65), (0, 4), [15, 600, 1500], [7, 13, 5],
         [(0, 1, 10**3.3, "P<0.001"), (1, 2, 10**4.2, "ns"), (0, 2, 10**5.1, "P<0.01")]),
        ("TNF-α\n(pg/ml)", (0.78, 6.47, 0.63), (0, 3), [10, 25, 40], [7, 13, 5],
         [(0, 1, 10**1.9, "P<0.001"), (1, 2, 10**2.6, "ns"), (0, 2, 10**3.4, "P<0.01")]),
        ("Interleukin-10\n(pg/ml)", (2.38, 6.43, 0.69), (0, 4), [8, 70, 250], [7, 13, 5],
         [(0, 1, 10**2.6, "P<0.001"), (1, 2, 10**3.4, "P<0.05"), (0, 2, 10**4.3, "P<0.01")]),
    ]
    for k, (ylab, (lx, ty, h), (e0, e1), meds, ns, brs) in enumerate(cyt):
        ax = L.axes(lx, ty, 0.85, h)
        data = {g: 10 ** (np.log10(m) + rng.normal(0, 0.35 if i < 2 else 0.45, n))
                for i, (g, m, n) in enumerate(zip(grades, meds, ns))}
        log10_axis(ax, e0, e1)
        plot_groups(ax, data, colors=[BOX_BLUE, BOX_YEL, BOX_RED], kind="box",
                    box_face="full", box_edge="black", median_color="black", whis=(0, 100),
                    point_face="white", point_edge="black", point_size=9, point_alpha=1,
                    jitter=0.45, box_width=0.6, label_weight="bold")
        log10_axis(ax, e0, e1)
        if k < 2:
            ax.set_xticklabels([])
        ax.set_ylabel(ylab, linespacing=1.05)
        for i, j, y, t in brs:
            bracket(ax, i, j, y, t, fontsize=7, tick_pt=3)

    # ---- H: biodistribution, dodged open bars + points -------------------------------
    L.panel_heading(3.65, 5.0, "H", "Biodistribution in Bone Marrow and CSF, According to Time\n    since Infusion",
                    size=hs, gap=0.2)
    ax = L.axes(4.17, 5.82, 1.27, 1.12)
    pos, spans = segment_positions([2, 4], gap=0.2)
    labs = ["6", "12", "6", "12", "18", "24"]
    ns = [24, 18, 8, 6, 5, 4]
    mns = [25, 20, 25, 12, 25, 12]
    bars, cols, xs = [], [], []
    for p, m, n in zip(pos, mns, ns):
        for d, c in ((-0.18, BLUE), (0.18, RED)):
            bars.append(lognorm_mean(rng, m, n, 1.1, 0.2))
            cols.append(c)
            xs.append(p + d)
    plot_bar_points(ax, xs, bars, colors=[c for c in cols], bar_width=0.3, s=11, jitter=0.1,
                    point_edge="none")
    ax.set_ylim(0, 200)
    ax.set_yticks(range(0, 201, 50))
    ax.set_ylabel("CAR T Cells/mm$^3$")
    segmented_time_axis(ax, pos, labs, [("Week", *spans[0]), ("Month", *spans[1])], pad_x=0.5)
    ax.legend([Line2D([], [], ls="none", marker="o", ms=4.5, color=BLUE),
               Line2D([], [], ls="none", marker="o", ms=4.5, color=RED)],
              ["Bone marrow, right side", "Bone marrow, left side"], loc="lower left",
              bbox_to_anchor=(-0.05, 1.03), handletextpad=0.2, labelspacing=0.2, fontsize=8)

    ax = L.axes(5.83, 5.82, 1.1, 1.12)
    plot_bar_points(ax, [0, 1], [[5, 20, 50, 95, 165], [5, 10, 60]], colors=GREEN,
                    bar_width=0.6, s=13, jitter=0.15, point_edge="none")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 200)
    ax.set_yticks(range(0, 201, 50))
    ax.set_xticks([0, 1], ["Week\n6", "Week\n12"], fontweight="bold")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("CAR T Cells/ml")
    ax.legend([Line2D([], [], ls="none", marker="o", ms=4.5, color=GREEN)], ["CSF"],
              loc="lower left", bbox_to_anchor=(0.05, 1.03), handletextpad=0.2, fontsize=8)

    # ---- I: single-patient course with annotations + flow plots -----------------------
    L.panel_heading(0.15, 7.5, "I", "Circulating Levels, According to Time Relative to First or Second Infusion",
                    size=hs, gap=0.2)
    ax = L.axes(0.79, 7.65, 2.98, 0.73)
    pos, spans = segment_positions([12, 3, 6], gap=0.15)
    labs = ["0", f"3{EN}5", "7", "14", "20", "20", "21", "21", "22", "23", "24", "25",
            "4", "6", "8", "6", "9", "12", "18", "24", "30"]
    y = [0, 20, 265, 148, 78, 5, 10, 12, 8, 10, 7, 5, 5, 5, 120, 110, 45, 48, 45, 38, 18]
    ax.plot(pos, y, color="#2B4C9B", lw=1.1, zorder=2)
    ax.scatter(pos, y, s=18, color="#3C6FC0", edgecolor="#1F3B7A", lw=0.5, zorder=3, clip_on=False)
    ax.set_ylim(0, 300)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel("CAR T Cells/mm$^3$")
    segmented_time_axis(ax, pos, labs, [("Day", *spans[0]), ("Week", *spans[1]),
                                        ("Month", *spans[2])], line_offset=34, text_offset=35.5,
                        pad_x=0.4)
    ax.tick_params(axis="x", labelsize=7.5)
    for i in (4, 6):     # rimiducid doses
        ax.annotate("", xy=(pos[i], y[i] + 18), xytext=(pos[i], y[i] + 85),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0, mutation_scale=8))
    # labels under specific ticks
    from matplotlib.transforms import blended_transform_factory as btf, ScaledTranslation as ST
    tr = btf(ax.transData, ax.transAxes)
    dn = lambda p: tr + ST(0, -p / 72, fig.dpi_scale_trans)
    for i in (5, 6, 7):          # short arrows from the notes up to the tick labels
        ax.annotate("", xy=(pos[i], 0), xycoords=dn(11), xytext=(pos[i], 0), textcoords=dn(19),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))
    for x, txt, row, ha in ((pos[5] - 0.3, "Before 1st infusion", 0, "right"),
                            (pos[5] + 0.3, "After 1st infusion", 1, "right"),
                            (pos[7] + 0.3, "After 2nd infusion", 0, "left"),
                            (pos[6] - 0.3, "Before 2nd infusion", 1, "left")):
        ax.text(x, 0, txt, transform=dn(21 + row * 8.5), ha=ha, va="center", fontsize=7.5)
    # arrow from the second expansion toward the flow plots
    ax.annotate("", xy=(4.42 / 7.1, 1 - 7.98 / 9.13), xycoords="figure fraction",
                xytext=(pos[14] + 0.4, y[14]), textcoords="data",
                arrowprops=dict(arrowstyle="-|>", lw=0.8, mutation_scale=8))

    L.text(6.08, 7.63, "Ex Vivo Rimiducid Treatment", ha="center", fontsize=9, fontweight="bold")
    for lx, sub, frac in ((4.53, "Before treatment", 0.092), (6.03, "96 Hr after treatment", 0.005)):
        L.text(lx + 0.4, 7.82, sub, ha="center", fontsize=8.5)
        n = 20000
        npos = int(n * frac)
        car = np.r_[rng.normal(0, 260, n - npos), 10 ** rng.normal(3.55, 0.3, npos)]
        cd3 = np.r_[10 ** rng.normal(4.25, 0.18, n - npos), 10 ** rng.normal(4.3, 0.18, npos)]
        if frac > 0.05:   # smear of the positive population out of the negative cloud
            k = int(n * 0.03)
            car[:k] = 10 ** rng.uniform(2.8, 3.3, k)
        fax = L.axes(lx, 7.95, 0.77, 0.65)
        plot_flow(fax, car, cd3, bins=110)
        add_gate(fax, 1800, 6e4, 6e3, 1.3e5, "auto", events=(car, cd3), label_xy=(4000, 2600),
                 fontsize=8.5)
        fax.set_xlabel("CAR")
        fax.set_ylabel("CD3")
        fax.tick_params(labelsize=7.5)

    for fmt in formats:
        fig.savefig(out / f"fig8_cart_kinetics.{fmt}", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--formats", nargs="+", default=["png", "svg"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    build(out, a.formats)
    print("done")
