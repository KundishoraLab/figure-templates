"""Demo of the newer blocks, on synthetic data.

fig6b  AD trajectories rebuilt from simulated longitudinal visits: raw points behind
       spline fits with cluster-robust CIs, 'difference at' onsets computed from the fits.
fig7   gallery: cumulative incidence + at-risk, ROC with bootstrap AUC, box/sina,
       violin, paired per-unit lines, volcano with stacked labels, correlation.

Run:  python gallery.py [--out out] [--formats png svg]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from nejm_plots import (COLORS, InchLayout, KMArm, add_at_risk_table, add_brackets,
                        add_difference_bracket, auto_y_break, band_legend, compare_groups,
                        divergence_onset, fit_trajectory, plot_correlation, plot_groups,
                        plot_km, plot_roc, plot_trajectory_scatter, plot_volcano, tint,
                        use_nejm_style)

MINUS = "−"
TEAL, RED = COLORS["teal"], COLORS["red"]


# ======================================================================================
# Synthetic longitudinal cohort
# ======================================================================================
def simulate_visits(rng, n, curve, between_sd, resid_sd, t0=-21, t1=0, max_visits=5,
                    floor=None):
    """Each participant enters at a random time and is seen 1..max_visits times."""
    xs, ys, ids = [], [], []
    for i in range(n):
        start = rng.uniform(t0, t1 - 0.5)
        k = rng.integers(1, max_visits + 1)
        t = start + np.cumsum(np.r_[0, rng.uniform(1.0, 3.0, k - 1)])
        t = t[t <= t1]
        u = rng.normal(0, between_sd)
        y = curve(t) + u + rng.normal(0, resid_sd, len(t))
        if floor is not None:
            y = np.clip(y, floor, None)
        xs.append(t); ys.append(y); ids.append(np.full(len(t), i))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def quad(p0, pm, p1, mid=-10):
    c = np.polyfit([-21, mid, 0], [p0, pm, p1], 2)
    return lambda x: np.polyval(c, x)


# ======================================================================================
# fig6b: trajectories with raw data
# ======================================================================================
def fig_ad_trajectories_scatter(out, formats, raw="points"):
    use_nejm_style(9)
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(6.2, 9.4))
    L = InchLayout(fig)
    L.frame(0.05, 0.03, 6.12, 9.32)
    L.vline(3.11, 0.03, 9.35)
    for y in (2.36, 4.69, 7.02):
        L.hline(0.05, 6.17, y)

    cdr_ad = lambda x: 0.12 + 4.78 * np.exp(x / 2.85) + 0.33 * np.exp(-(x + 21) / 1.8)
    panels = [
        # letter, row, col, ctrl curve, AD curve, between sd, resid sd, floor, y label
        ("A", 0, 0, quad(718, 712, 645), quad(712, 560, 412), 55, 30, None,
         r"CSF A$\beta_{42}$ (pg/ml)"),
        ("B", 1, 0, quad(.111, .110, .101), quad(.110, .088, .066), .008, .004, None,
         r"Ratio of CSF A$\beta_{42}$ to A$\beta_{40}$"),
        ("C", 1, 1, quad(51, 52, 73), quad(44, 75, 134), 9, 5, None, "CSF p-tau 181 (pg/ml)"),
        ("D", 2, 0, quad(214, 240, 295), quad(188, 320, 540), 45, 20, None, "CSF t-tau (pg/ml)"),
        ("E", 2, 1, quad(620, 720, 850), quad(585, 850, 1560), 140, 60, None, "CSF NfL (pg/ml)"),
        ("F", 3, 0, quad(7720, 7400, 7300), quad(7650, 7230, 6300), 380, 110, None,
         r"Hippocampal Volume (mm$^3$)"),
        ("G", 3, 1, lambda x: 0.16 + 0 * x, cdr_ad, 0.25, 0.25, 0, "CDR-SB Score"),
    ]
    grid = np.linspace(-21, 0, 200)
    onsets = {}
    for letter, r, c, f_ctrl, f_ad, bsd, rsd, floor, ylab in panels:
        cx, cy = 0.05 + c * 3.06, 0.03 + r * 2.33
        L.text(cx + 0.1, cy + 0.13, letter, fontsize=10.5, fontweight="bold")
        ax = L.axes(cx + 0.67, cy + 0.5, 2.21, 1.36)
        xc, yc, ic = simulate_visits(rng, 220, f_ctrl, bsd, rsd, floor=floor)
        xa, ya, ia = simulate_visits(rng, 220, f_ad, bsd, rsd, floor=floor)
        knots = [-19, -12, -5, -1] if letter == "G" else None
        fc = fit_trajectory(xc, yc, ids=ic, grid=grid, knots=knots)
        fa = fit_trajectory(xa, ya, ids=ia, grid=grid, knots=knots)
        plot_trajectory_scatter(ax, xc, yc, fc, ids=ic, color=TEAL, raw=raw, point_size=4,
                                point_alpha=0.45)
        plot_trajectory_scatter(ax, xa, ya, fa, ids=ia, color=RED, raw=raw, point_size=4,
                                point_alpha=0.45)
        ax.set_xlim(-21.6, 0.3)
        ax.set_xticks([-20, -15, -10, -5, 0])
        ticks = auto_y_break(ax, np.r_[yc, ya], nbins=5, mark_size=8)
        if letter == "B":
            ax.set_yticklabels([f"{v:.2f}" for v in ticks])
        ax.set_ylabel(ylab)
        ax.set_xlabel("Years before Diagnosis")
        ax.tick_params(labelsize=8.5)
        on = divergence_onset(fa, fc)
        onsets[letter] = on
        if on is not None:
            add_difference_bracket(ax, on, 0, f"Difference at {MINUS}{abs(round(on))} yr",
                                   y=1.04, fontsize=8.5)

    leg_ax = L.axes(3.16, 0.6, 2.9, 0.8)
    leg_ax.axis("off")
    band_legend(leg_ax, ["Cognitively normal matched controls",
                         "Participants with diagnosis of\nAlzheimer’s disease"],
                [TEAL, RED], [tint(TEAL, 0.45), tint(RED, 0.45)], loc="center left",
                bbox_to_anchor=(0.0, 0.5), fontsize=9)
    leg_ax.text(0.02, 0.0, "Points, individual visits. Bands, 95% CI\n(cluster-robust by participant).",
                transform=leg_ax.transAxes, fontsize=8, color="#555555", va="top")
    for fmt in formats:
        fig.savefig(out / f"fig6b_ad_trajectories_scatter.{fmt}", dpi=250)
    plt.close(fig)
    return onsets


# ======================================================================================
# fig7: gallery of other common panels
# ======================================================================================
def fig_gallery(out, formats):
    use_nejm_style(8)
    rng = np.random.default_rng(5)
    fig = plt.figure(figsize=(7.2, 8.6))       # 183 mm wide
    L = InchLayout(fig)
    lab = dict(fontsize=11, fontweight="bold", ha="left")
    c3 = [TEAL, COLORS["orange"], RED]

    # a. cumulative incidence by tertile (UK Biobank style) ----------------------------
    L.text(0.05, 0.12, "a", **lab)
    ax = L.axes(0.75, 0.3, 3.55, 1.75)
    arms = []
    for name, rate, col in (("Tertile 1", 0.004, TEAL), ("Tertile 2", 0.008, COLORS["orange"]),
                            ("Tertile 3", 0.018, RED)):
        n = 4000
        ev = rng.exponential(1 / rate, n)
        cens = np.minimum(rng.uniform(10, 15, n), rng.exponential(60, n))
        arms.append(KMArm(name, np.minimum(ev, cens), (ev <= cens).astype(int), col,
                          lw=1.4, censor_marker="", ci_alpha=0.2))
    fits = plot_km(ax, arms, xlim=(0, 14), xticks=range(0, 15, 2), ylim=(0, 25),
                   yticks=range(0, 26, 5), xlabel="Years since Baseline",
                   ylabel="Cumulative Incidence (%)", cumulative_incidence=True)
    ax.legend(loc="upper left", handlelength=1.2)
    add_at_risk_table(ax, fits, at=range(0, 15, 2), label_x=-0.19, title_offset=30,
                      row_height=10.5, fontsize=7.5, title_fontsize=7.5)

    # b. ROC with bootstrap AUC ---------------------------------------------------------
    L.text(4.75, 0.12, "b", **lab)
    ax = L.axes(5.2, 0.3, 1.85, 1.85)
    n = 600
    y = rng.random(n) < 0.3
    base = rng.normal(0, 1, n)
    models = {"Clinical": base * 0.6 + y * 0.8 + rng.normal(0, 1, n),
              "Proteomic": base * 0.3 + y * 1.4 + rng.normal(0, 1, n),
              "Combined": base * 0.3 + y * 1.8 + rng.normal(0, 1, n)}
    plot_roc(ax, models, y, colors=["#9A9A9A", TEAL, RED], n_boot=300, lw=1.4,
             legend_kw=dict(fontsize=6.5, bbox_to_anchor=(1.05, 0.0)))

    # c. box + sina, three groups -------------------------------------------------------
    L.text(0.05, 3.2, "c", **lab)
    ax = L.axes(0.6, 3.4, 1.6, 1.85)
    data = {"Control": rng.lognormal(2.4, 0.35, 60), "MCI": rng.lognormal(2.6, 0.38, 55),
            "AD": rng.lognormal(2.9, 0.4, 50)}
    plot_groups(ax, data, colors=c3, kind="box", point_size=7)
    ax.set_ylabel("Plasma p-tau217 (pg/ml)")
    p = compare_groups(data, [("Control", "MCI"), ("MCI", "AD"), ("Control", "AD")])
    add_brackets(ax, data, p)

    # d. violin ------------------------------------------------------------------------
    L.text(2.45, 3.2, "d", **lab)
    ax = L.axes(3.0, 3.4, 1.6, 1.85)
    data = {"WT": rng.normal(0, 1, 300), "Het": rng.normal(0.5, 1.1, 300),
            "Hom": rng.normal(1.3, 1.2, 200)}
    plot_groups(ax, data, colors=c3, kind="violin", point_size=2, jitter=0.28,
                point_alpha=0.35, rasterize_points=True)
    ax.set_ylabel("Protein Level (z)")

    # e. paired per-unit lines (e.g. one point per lesion) -------------------------------
    L.text(4.85, 3.2, "e", **lab)
    ax = L.axes(5.4, 3.4, 1.6, 1.85)
    ids = [f"L{i}" for i in range(12)]
    base = rng.normal(0.30, 0.08, 12)
    data = {"Bystander": base, "Mutant": base + rng.normal(0.12, 0.05, 12)}
    plot_groups(ax, data, colors=[TEAL, RED], kind="none", point_size=16,
                paired_ids={"Bystander": ids, "Mutant": ids}, pair_lw=0.8)
    for i, (k, c) in enumerate(zip(data, [TEAL, RED])):
        ax.plot([i - 0.18, i + 0.18], [np.median(data[k])] * 2, color=c, lw=2.2)
    ax.set_ylabel("Program Score")
    p = compare_groups(data, [("Bystander", "Mutant")], test="wilcoxon",
                       paired_ids={"Bystander": ids, "Mutant": ids})
    add_brackets(ax, data, p)

    # f. volcano ------------------------------------------------------------------------
    L.text(0.05, 5.95, "f", **lab)
    ax = L.axes(0.6, 6.1, 3.7, 2.0)
    m = 2900
    fc = rng.normal(0, 0.45, m)
    hits = rng.choice(m, 70, replace=False)
    fc[hits] += rng.choice([-1, 1], 70) * rng.uniform(0.9, 2.2, 70)
    se = rng.uniform(0.25, 0.45, m)
    pv = 2 * __import__("scipy").stats.norm.sf(np.abs(fc) / se)
    names = np.array([f"PROT{i}" for i in range(m)])
    real = ["NEFL", "GFAP", "CHI3L1", "SPP1", "TREM2", "YWHAG", "SMOC1", "NPTX2", "VGF",
            "SCG2", "CD44", "MMP10"]
    top = np.argsort(-np.abs(fc) / se)[:len(real)]
    names[top] = real
    plot_volcano(ax, fc, pv, names, fc_thresh=1.0, p_thresh=0.05 / m, label_top=6,
                 up_color=RED, down_color=TEAL, s=4, s_sig=8, label_fontsize=7,
                 label_italic=True)
    ax.axhline(-np.log10(0.05 / m), color="#888888", lw=0.7, ls=(0, (3, 3)), zorder=0)

    # g. correlation --------------------------------------------------------------------
    L.text(4.6, 5.95, "g", **lab)
    ax = L.axes(5.15, 6.25, 1.9, 1.85)
    x = rng.normal(0, 1, 160)
    yv = 0.55 * x + rng.normal(0, 0.8, 160)
    grp = np.where(rng.random(160) < 0.5, "Control", "Case")
    st = plot_correlation(ax, x, yv, color="#333333", groups=grp,
                          group_colors={"Control": TEAL, "Case": RED}, s=9,
                          label="Pooled, {r}")
    ax.set_xlabel("CSF NfL (z)")
    ax.set_ylabel("Plasma NfL (z)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.5, handlelength=1.0,
              handletextpad=0.4, ncol=2, columnspacing=0.8, borderaxespad=0.2)

    for fmt in formats:
        fig.savefig(out / f"fig7_gallery.{fmt}", dpi=250)
    plt.close(fig)
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--formats", nargs="+", default=["png", "svg"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("onsets:", fig_ad_trajectories_scatter(out, a.formats))
    print("corr:", fig_gallery(out, a.formats))
