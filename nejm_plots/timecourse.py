"""Time-course and distribution blocks (CAR T style figures).

segment_positions / segmented_time_axis
    categorical time axis with grouped brackets underneath (Day | Week | Month)
plot_time_strip
    per-timepoint points with a mean (or median) bar and SEM whiskers
plot_time_line
    mean +/- SEM line across timepoints (error bars can point up only, as on log axes)
plot_violin_outline
    unfilled violins cut at the data range, coloured median, dotted quartiles
plot_bar_points
    open bars (mean) with SEM and the individual points, optionally dodged
bracket
    comparison bracket with any text, works on linear or log axes
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from matplotlib.transforms import blended_transform_factory, ScaledTranslation


# --------------------------------------------------------------------------------------
# Segmented categorical time axis
# --------------------------------------------------------------------------------------
def segment_positions(sizes: Sequence[int], gap=0.35, start=0.0):
    """x positions for consecutive segments of sizes n1, n2, ... with an extra gap
    between segments. Returns (positions, [(first, last) per segment])."""
    pos, spans, x = [], [], start
    for n in sizes:
        seg = x + np.arange(n)
        pos.extend(seg)
        spans.append((seg[0], seg[-1]))
        x = seg[-1] + 1 + gap
    return np.array(pos), spans


def segmented_time_axis(ax, positions, labels, segments, *, line_offset=14, text_offset=15.5,
                        overhang=0.3, tick=3.5, lw=0.9, fontsize=None, weight="bold",
                        rotate_first=None, pad_x=0.6):
    """Tick labels at `positions`; under them, one bracket per segment with its name.

    segments: [("Day", first_pos, last_pos), ...]  (positions in data units)
    rotate_first: angle for the first tick label (e.g. 'Day 0' set at 45 degrees).
    Offsets are points below the x axis.
    """
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", pad=1.5, length=3)
    if rotate_first is not None:
        t = ax.get_xticklabels()[0]
        t.set_rotation(rotate_first); t.set_ha("right"); t.set_va("top")
    ax.set_xlim(min(positions) - pad_x, max(positions) + pad_x)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    fig = ax.figure

    def down(pts):
        return tr + ScaledTranslation(0, -pts / 72, fig.dpi_scale_trans)

    for name, a, b in segments:
        x0, x1 = a - overhang, b + overhang
        ax.plot([x0, x1], [0, 0], transform=down(line_offset), color="black", lw=lw,
                clip_on=False)
        for xe in (x0, x1):          # end ticks pointing up
            ax.annotate("", xy=(xe, 0), xycoords=down(line_offset - tick), xytext=(xe, 0),
                        textcoords=down(line_offset),
                        arrowprops=dict(arrowstyle="-", lw=lw, shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2, 0, name, transform=down(text_offset), ha="center", va="top",
                fontsize=fontsize, fontweight=weight, clip_on=False)


# --------------------------------------------------------------------------------------
# Per-timepoint strip with mean / median bar
# --------------------------------------------------------------------------------------
def _center_err(v, center, err):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    c = np.mean(v) if center == "mean" else np.median(v)
    if err == "sem":
        e = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    elif err == "sd":
        e = v.std(ddof=1) if len(v) > 1 else 0.0
    elif err == "ci95":
        e = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
    else:
        e = 0.0
    return c, e


def plot_time_strip(ax, positions, values: Sequence[Sequence[float]], *, color="#3A7DC0",
                    s=14, jitter=0.18, center="mean", err="sem", center_color="black",
                    center_width=0.5, center_lw=1.2, err_lw=1.0, cap=0.15, edge=None,
                    rng=None, zorder=3, clip=False):
    """One column of jittered points per timepoint, with a horizontal bar at the mean
    (or median) and SEM whiskers. Use center_lw ~5 for a thick coloured median bar."""
    rng = rng or np.random.default_rng(0)
    for x, v in zip(positions, values):
        v = np.asarray(v, float)
        if len(v) == 0:
            continue
        xo = x + rng.uniform(-jitter, jitter, len(v))
        ax.scatter(xo, v, s=s, color=color, lw=0.4 if edge else 0, edgecolor=edge,
                   zorder=zorder, clip_on=clip)
        c, e = _center_err(v, center, err)
        ax.plot([x - center_width / 2, x + center_width / 2], [c, c], color=center_color,
                lw=center_lw, zorder=zorder + 1, solid_capstyle="butt")
        if e > 0:
            ax.plot([x, x], [c - e, c + e], color=center_color, lw=err_lw, zorder=zorder + 1)
            for yy in (c - e, c + e):
                ax.plot([x - cap, x + cap], [yy, yy], color=center_color, lw=err_lw,
                        zorder=zorder + 1)


def plot_time_line(ax, positions, mean, err=None, *, color="black", marker=None, ms=5,
                   lw=1.2, capsize=3, direction="both", label=None, mfc=None, zorder=3):
    """Line through summary values with error bars. direction: 'both', 'up', 'down'."""
    positions = np.asarray(positions, float)
    mean = np.asarray(mean, float)
    ax.plot(positions, mean, color=color, lw=lw, marker=marker, ms=ms,
            mfc=mfc or color, mec=color, label=label, zorder=zorder, clip_on=False)
    if err is not None:
        err = np.asarray(err, float)
        lo = np.zeros_like(err) if direction == "up" else err
        hi = np.zeros_like(err) if direction == "down" else err
        ax.errorbar(positions, mean, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=lw * 0.8,
                    capsize=capsize, capthick=lw * 0.8, zorder=zorder - 0.5)


def summarize(values, center="mean", err="sem"):
    """(centers, errors) arrays for a list of per-timepoint value arrays."""
    ce = [_center_err(v, center, err) for v in values]
    return np.array([c for c, _ in ce]), np.array([e for _, e in ce])


def geo_summary(values):
    """Geometric mean and multiplicative SEM (upper error, for log axes)."""
    out_m, out_e = [], []
    for v in values:
        lv = np.log10(np.asarray(v, float)[np.asarray(v) > 0])
        m = lv.mean()
        se = lv.std(ddof=1) / np.sqrt(len(lv)) if len(lv) > 1 else 0
        out_m.append(10 ** m)
        out_e.append(10 ** (m + se) - 10 ** m)
    return np.array(out_m), np.array(out_e)


# --------------------------------------------------------------------------------------
# Outline violins
# --------------------------------------------------------------------------------------
def plot_violin_outline(ax, values: Sequence[Sequence[float]], positions=None, *, width=0.8,
                        edge="black", lw=0.9, median_color="#3C9A5F", median_lw=1.6,
                        quartile_ls=(0, (1.5, 1.5)), quartile_lw=0.9, fill="none",
                        bw=None, n=200, cut=0.0, labels=None):
    """NEJM-style violins: outline only, cut at the data range (cut=0), a coloured median
    line and dotted quartile lines drawn to the violin edge."""
    positions = np.arange(len(values)) if positions is None else np.asarray(positions, float)
    for x, v in zip(positions, values):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        lo, hi = v.min(), v.max()
        pad = cut * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, n)
        if len(v) >= 2 and np.ptp(v) > 0:
            d = stats.gaussian_kde(v, bw_method=bw)(grid)
        else:
            d = np.ones_like(grid)
        w = d / d.max() * width / 2
        ax.fill_betweenx(grid, x - w, x + w, facecolor=fill, edgecolor=edge, lw=lw, zorder=2)
        # close the flat ends
        ax.plot([x - w[0], x + w[0]], [grid[0]] * 2, color=edge, lw=lw, zorder=2)
        ax.plot([x - w[-1], x + w[-1]], [grid[-1]] * 2, color=edge, lw=lw, zorder=2)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        for q, ls, c, l in ((q1, quartile_ls, edge, quartile_lw), (q3, quartile_ls, edge, quartile_lw),
                            (med, "-", median_color, median_lw)):
            hw = np.interp(q, grid, w)
            ax.plot([x - hw, x + hw], [q, q], color=c, lw=l, ls=ls, zorder=3)
    ax.set_xticks(positions)
    if labels is not None:
        ax.set_xticklabels(labels)
    return positions


# --------------------------------------------------------------------------------------
# Open bars with points
# --------------------------------------------------------------------------------------
def plot_bar_points(ax, positions, values: Sequence[Sequence[float]], *, colors="#3A7DC0",
                    bar_width=0.6, bar_face="white", bar_edge="#555555", bar_lw=0.9,
                    err="sem", err_color="black", err_lw=0.9, capsize=2.5, s=16,
                    point_edge="#333333", jitter=None, rng=None, mean_line=True):
    """Bars at the mean with SEM, the individual values on top.
    colors: one colour or one per bar (for dodged groups pass per-bar colours)."""
    rng = rng or np.random.default_rng(0)
    if isinstance(colors, str):
        colors = [colors] * len(values)
    jitter = bar_width * 0.3 if jitter is None else jitter
    for x, v, c in zip(positions, values, colors):
        v = np.asarray(v, float)
        m, e = _center_err(v, "mean", err)
        ax.bar(x, m, width=bar_width, color=bar_face, edgecolor=bar_edge, lw=bar_lw, zorder=1)
        ax.scatter(x + rng.uniform(-jitter, jitter, len(v)), v, s=s, color=c,
                   edgecolor=point_edge, lw=0.4, zorder=3, clip_on=False)
        if e > 0:
            ax.errorbar(x, m, yerr=e, fmt="none", ecolor=err_color, elinewidth=err_lw,
                        capsize=capsize, capthick=err_lw, zorder=4)


# --------------------------------------------------------------------------------------
# Comparison bracket with text
# --------------------------------------------------------------------------------------
def bracket(ax, x1, x2, y, text="", *, tick_pt=4, lw=0.8, pad_pt=1.5, fontsize=None,
            color="black"):
    """Bracket from x1 to x2 at height y (data units), legs drop tick_pt points.
    Works on log or broken axes because the legs are offsets in points."""
    fig = ax.figure
    down = ax.transData + ScaledTranslation(0, -tick_pt / 72, fig.dpi_scale_trans)
    ax.plot([x1, x2], [y, y], color=color, lw=lw, clip_on=False)
    for x in (x1, x2):
        ax.annotate("", xy=(x, y), xytext=(x, y), textcoords=down, xycoords="data",
                    arrowprops=dict(arrowstyle="-", lw=lw, color=color, shrinkA=0, shrinkB=0))
    if text:
        up = ax.transData + ScaledTranslation(0, pad_pt / 72, fig.dpi_scale_trans)
        ax.text((x1 + x2) / 2, y, text, transform=up, ha="center", va="bottom",
                fontsize=fontsize, color=color)
