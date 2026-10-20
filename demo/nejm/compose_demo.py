"""Panel makers + Composite assembler, end to end.

1. Build each panel once from data (synthetic here).
2. Save every panel on its own (out/panels/).
3. Assemble the same panel objects into composite figures:
     composite_1  - survival, discrimination, group comparisons, omics (Nature theme)
     composite_2  - trajectories, forest, point estimates, ordinal outcome, time course,
                    flow (Nature theme)
     composite_2_nejm - the same panels re-assembled under the NEJM theme

Run:  python compose_demo.py [--out out] [--formats png svg]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from nejm_plots import COLORS, Composite, ForestRow, KMArm, Theme
from nejm_plots.panels import (bar_points_panel, correlation_panel, flow_panel, forest_panel,
                               grotta_panel, groups_panel, km_panel, legend_panel,
                               paired_panel, point_range_panel, roc_panel, timecourse_panel,
                               trajectory_panel, violin_panel, volcano_panel)

TEAL, ORANGE, RED = COLORS["teal"], COLORS["orange"], COLORS["red"]
EN = "–"


# ======================================================================================
# Synthetic data (replace with your own)
# ======================================================================================
def data_km(rng):
    arms = []
    for name, rate, col in (("Tertile 1", 0.004, TEAL), ("Tertile 2", 0.008, ORANGE),
                            ("Tertile 3", 0.018, RED)):
        ev = rng.exponential(1 / rate, 4000)
        cens = np.minimum(rng.uniform(10, 15, 4000), rng.exponential(60, 4000))
        arms.append(KMArm(name, np.minimum(ev, cens), (ev <= cens).astype(int), col,
                          censor_marker="", ci_alpha=0.2))
    return arms


def data_roc(rng, n=600):
    y = rng.random(n) < 0.3
    base = rng.normal(0, 1, n)
    return y, {"Clinical": base * 0.6 + y * 0.8 + rng.normal(0, 1, n),
               "Proteomic": base * 0.3 + y * 1.4 + rng.normal(0, 1, n),
               "Combined": base * 0.3 + y * 1.8 + rng.normal(0, 1, n)}


def data_volcano(rng, m=2900):
    fc = rng.normal(0, 0.45, m)
    hits = rng.choice(m, 70, replace=False)
    fc[hits] += rng.choice([-1, 1], 70) * rng.uniform(0.9, 2.2, 70)
    se = rng.uniform(0.25, 0.45, m)
    p = 2 * stats.norm.sf(np.abs(fc) / se)
    names = np.array([f"PROT{i}" for i in range(m)])
    real = ["NEFL", "GFAP", "CHI3L1", "SPP1", "TREM2", "YWHAG", "SMOC1", "NPTX2", "VGF",
            "SCG2", "CD44", "MMP10"]
    names[np.argsort(-np.abs(fc) / se)[:len(real)]] = real
    return fc, p, names


def data_visits(rng, curve, n=200, bsd=1.0, rsd=0.5):
    xs, ys, ids = [], [], []
    for i in range(n):
        t = rng.uniform(-21, -0.5) + np.cumsum(np.r_[0, rng.uniform(1, 3, rng.integers(0, 4))])
        t = t[t <= 0]
        y = curve(t) + rng.normal(0, bsd) + rng.normal(0, rsd, len(t))
        xs.append(t); ys.append(y); ids.append(np.full(len(t), i))
    return dict(x=np.concatenate(xs), y=np.concatenate(ys), ids=np.concatenate(ids))


def quad(p0, pm, p1):
    c = np.polyfit([-21, -10, 0], [p0, pm, p1], 2)
    return lambda x: np.polyval(c, x)


# ======================================================================================
# Panels (each built once, reused everywhere)
# ======================================================================================
def build_panels(seed=5):
    rng = np.random.default_rng(seed)
    P = {}

    P["km"] = km_panel(data_km(rng), xlim=(0, 14), xticks=range(0, 15, 2), ylim=(0, 25),
                       yticks=range(0, 26, 5), xlabel="Years since Baseline",
                       cumulative_incidence=True, legend="upper left")
    y, models = data_roc(rng)
    P["roc"] = roc_panel(models, y, colors=["#9A9A9A", TEAL, RED], n_boot=300)

    P["box"] = groups_panel({"Control": rng.lognormal(2.4, 0.35, 60),
                             "MCI": rng.lognormal(2.6, 0.38, 55),
                             "AD": rng.lognormal(2.9, 0.4, 50)},
                            colors=[TEAL, ORANGE, RED], ylabel="Plasma p-tau217 (pg/ml)",
                            brackets="all")
    P["violin"] = groups_panel({"WT": rng.normal(0, 1, 300), "Het": rng.normal(0.5, 1.1, 300),
                                "Hom": rng.normal(1.3, 1.2, 200)}, colors=[TEAL, ORANGE, RED],
                               kind="violin", ylabel="Protein Level (z)", point_alpha=0.35)
    base = rng.normal(0.30, 0.08, 12)
    P["paired"] = paired_panel({"Bystander": base, "Mutant": base + rng.normal(0.12, 0.05, 12)},
                               ids=[f"L{i}" for i in range(12)], colors=[TEAL, RED],
                               ylabel="Program Score")
    fc, p, names = data_volcano(rng)
    P["volcano"] = volcano_panel(fc, p, names, p_thresh=0.05 / len(p), p_line=0.05 / len(p),
                                 label_top=6)
    x = rng.normal(0, 1, 160)
    P["corr"] = correlation_panel(x, 0.55 * x + rng.normal(0, 0.8, 160),
                                  groups=np.where(rng.random(160) < 0.5, "Control", "Case"),
                                  group_colors={"Control": TEAL, "Case": RED},
                                  xlabel="CSF NfL (z)", ylabel="Plasma NfL (z)")

    # trajectories
    ab = (quad(718, 712, 645), quad(712, 560, 412), 55, 30)
    nfl = (quad(620, 720, 850), quad(585, 850, 1560), 140, 60)
    for key, (fc_, fa_, b, r_), ylab in (("traj_ab", ab, r"CSF A$\beta_{42}$ (pg/ml)"),
                                         ("traj_nfl", nfl, "CSF NfL (pg/ml)")):
        c = data_visits(rng, fc_, bsd=b, rsd=r_)
        a = data_visits(rng, fa_, bsd=b, rsd=r_)
        P[key] = trajectory_panel([dict(**c, color=TEAL, label="Controls"),
                                   dict(**a, color=RED, label="Alzheimer's disease")],
                                  xlabel="Years before Diagnosis", ylabel=ylab,
                                  xticks=[-20, -15, -10, -5, 0], xlim=(-21.6, 0.3))
    P["traj_legend"] = legend_panel([("Cognitively normal controls", TEAL, "band"),
                                     ("Alzheimer’s disease", RED, "band")])

    # forest across outcomes
    R = ForestRow
    P["forest"] = forest_panel(
        [R("All participants", 52418, 1.42, 1.31, 1.54), R("Retinal disease"),
         R("Glaucoma", 1210, 1.18, 1.02, 1.37, 1), R("AMD", 980, 1.35, 1.14, 1.60, 1),
         R("Retinal vein occlusion", 310, 1.62, 1.21, 2.17, 1), R("Neuro-ophthalmic"),
         R("Optic neuritis", 95, 2.10, 1.30, 3.39, 1),
         R("Ischemic optic neuropathy", 140, 1.71, 1.18, 2.48, 1),
         R("Diplopia", 420, 1.08, 0.88, 1.33, 1)],
        xlim=(0.5, 4), xticks=(0.5, 1, 2, 4), headers=("Outcome", "Events", "HR per SD (95% CI)"),
        better=("Lower risk", "Higher risk"), label_w=1.25, n_w=0.42, est_w=0.92)
    P["pointrange"] = point_range_panel(
        ["6M-CDP", "6M-CDI", "SDMT"],
        {"Rituximab": ([3, 29, 2.3], [0, 16, 0], [7, 40, 6]),
         "Ocrelizumab": ([7, 24, 3.6], [2, 11, 0], [14, 33, 9])},
        ylabel="Participants (%)", ylim=(0, 45), yticks=range(0, 46, 10), legend="upper right")
    P["grotta"] = grotta_panel([[17.6, 23.9, 12.5, 18.0, 11.0, 3.5, 13.3],
                                [13.9, 29.2, 15.7, 19.0, 9.5, 4.4, 8.4]],
                               ["EVT\n(N=255)", "Usual care\n(N=274)"],
                               legend_title="Modified Rankin Scale score at 90 days")

    # time course
    means = [0.5, 3, 50, 100, 40, 15, 20, 12, 25, 15, 18, 15]
    vals = [np.clip(rng.lognormal(np.log(m + .5) - .7, 1.2, 18), 0, None) for m in means]
    P["timecourse"] = timecourse_panel(
        [("Day", ["0", f"3{EN}5"]), ("Week", ["1", "2", "4", "6", "8", "12"]),
         ("Month", ["6", "9", "12", "18"])],
        [dict(kind="strip", values=vals, color=COLORS["blue"])],
        ylabel="CAR T Cells/mm$^3$", ylim=(0, 400), yticks=range(0, 401, 100))

    # flow before / after
    for key, frac, sub in (("flow_pre", 0.092, "Before"), ("flow_post", 0.005, "96 hr after")):
        n = 15000
        k = int(n * frac)
        car = np.r_[rng.normal(0, 260, n - k), 10 ** rng.normal(3.55, 0.3, k)]
        cd3 = 10 ** rng.normal(4.25, 0.18, n)
        P[key] = flow_panel(car, cd3, gate=(1800, 6e4, 6e3, 1.3e5), xlabel="CAR",
                            ylabel="CD3", subtitle=sub)
    P["violin_dose"] = violin_panel([np.clip(rng.gamma(2.5, 8, 7), 0, None),
                                     np.clip(rng.normal(140, 50, 12), 0, None),
                                     np.clip(rng.normal(175, 70, 10), 0, None)],
                                    ["Low", "Mid", "High"], ylabel="Peak (cells/mm$^3$)",
                                    xlabel="Dose", median_color=COLORS["green"] if "green" in COLORS else "#3C9A5F",
                                    ylim=(0, 320), yticks=range(0, 301, 100))
    P["csf_bars"] = bar_points_panel([[5, 20, 50, 95, 165], [5, 10, 60]], ["Week 6", "Week 12"],
                                     colors="#3C9A5F", ylabel="CAR T Cells/ml", ylim=(0, 200))
    return P


# ======================================================================================
# Composites
# ======================================================================================
def composite_1(P, theme):
    c = Composite(width="double", theme=theme)
    c.row([(P["km"], 8), (P["roc"], 4)], height=1.55)
    c.row([P["box"], P["violin"], P["paired"]], height=1.5)
    c.row([(P["volcano"], 7), (P["corr"], 5)], height=1.6)
    return c


def composite_2(P, theme):
    c = Composite(width="double", theme=theme)
    c.row([(P["traj_ab"], 4.5), (P["traj_nfl"], 4.5), (P["traj_legend"], 3)], height=1.25)
    c.row([(P["forest"], 8), (P["pointrange"], 4)], height=1.55)
    c.row([(P["grotta"], 12)], height=0.62)
    c.row([(P["timecourse"], 5), (P["violin_dose"], 3), (P["flow_pre"], 2), (P["flow_post"], 2)],
          height=1.2)
    return c


def save_panels(P, theme, out, formats):
    """Each panel on its own, at a width that suits its type."""
    widths = {"km": 4.2, "forest": 4.6, "grotta": 6.0, "roc": 2.2, "flow_pre": 1.7,
              "flow_post": 1.7, "traj_legend": 2.0, "volcano": 3.8, "timecourse": 3.4}
    heights = {"grotta": 0.62, "traj_legend": 0.6, "forest": 1.55, "km": 1.55}
    for k, p in P.items():
        p.save(out / "panels" / k, width=widths.get(k, 2.6), height=heights.get(k, 1.5),
               theme=theme, formats=formats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--formats", nargs="+", default=["png", "svg"])
    a = ap.parse_args()
    out = Path(a.out)
    P = build_panels()
    nature = Theme.nature()
    composite_1(P, nature).save(out / "composite_1", formats=a.formats)
    composite_2(P, nature).save(out / "composite_2", formats=a.formats)
    composite_2(P, Theme.nejm()).save(out / "composite_2_nejm", formats=a.formats)
    save_panels(P, nature, out, a.formats)
    print("done")
