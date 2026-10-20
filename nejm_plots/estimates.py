"""Point estimates with CIs (point-range) and smooth trajectories with CI bands."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory, ScaledTranslation


# --------------------------------------------------------------------------------------
# Point-range
# --------------------------------------------------------------------------------------
def plot_point_range(ax, categories: Sequence[str], series: dict, *, colors=None,
                     offset=0.18, ms=6.5, capsize=5, lw=1.4, xpad=0.6, zero_line=None,
                     zero_kw=None, label_weight="bold", point_colors=None):
    """Grouped point estimates with vertical CIs.

    series: {"Rituximab": (est, lo, hi), ...} where each of est/lo/hi has one value per
            category. One series -> centred; several -> spread by `offset`.
    point_colors: with a single series, colour each category separately
                  (e.g. one estimate per drug on the x axis).
    """
    names = list(series)
    S = len(names)
    x = np.arange(len(categories), dtype=float)
    shifts = (np.arange(S) - (S - 1) / 2) * offset
    colors = colors or ["black"] * S
    for name, sh, c in zip(names, shifts, colors):
        est, lo, hi = (np.asarray(v, float) for v in series[name])
        pcs = point_colors if (point_colors is not None and S == 1) else [c] * len(x)
        for j in range(len(x)):
            ax.errorbar([x[j] + sh], [est[j]], yerr=[[est[j] - lo[j]], [hi[j] - est[j]]],
                        fmt="o", ms=ms, color=pcs[j], ecolor=pcs[j], elinewidth=lw,
                        capsize=capsize, capthick=lw, label=name if j == 0 else None,
                        zorder=3, clip_on=False)
    if zero_line is not None:
        zk = dict(color="black", lw=1.0, ls=(0, (4, 3)), zorder=1)
        zk.update(zero_kw or {})
        ax.axhline(zero_line, **zk)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight=label_weight, linespacing=1.15)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlim(x[0] - xpad, x[-1] + xpad)
    return x


def dot_legend(ax, names, colors, loc="upper center", bbox=(0.5, 1.02), ms=6, **kw):
    handles = [Line2D([], [], ls="none", marker="o", ms=ms, color=c) for c in colors]
    return ax.legend(handles, names, loc=loc, bbox_to_anchor=bbox, ncol=len(names),
                     handletextpad=0.3, columnspacing=1.5, frameon=False, **kw)


# --------------------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------------------
def plot_trajectory(ax, x, y, lo, hi, *, color, band_color=None, band_alpha=0.9, lw=1.7,
                    label=None, band_edge=True):
    """Line plus confidence band (e.g. modelled biomarker trajectory)."""
    band_color = band_color or color
    ax.fill_between(x, lo, hi, color=band_color, alpha=band_alpha, lw=0, zorder=1)
    if band_edge:
        ax.plot(x, lo, color=band_color, lw=0.6, zorder=1)
        ax.plot(x, hi, color=band_color, lw=0.6, zorder=1)
    ax.plot(x, y, color=color, lw=lw, zorder=2, label=label, solid_capstyle="round")


def add_difference_bracket(ax, x0, x1, text, *, y=0.08, color="#8A8A8A", lw=1.0,
                           tick=3.5, pad=3, fontsize=None, text_color="black"):
    """Horizontal bracket spanning [x0, x1] (data) at height y (axes fraction), label above."""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([x0, x1], [y, y], transform=tr, color=color, lw=lw, clip_on=False, zorder=4)
    ax.plot([x0, x1], [y, y], transform=tr, ls="none", marker="|", ms=tick * 2, mew=lw,
            color=color, clip_on=False, zorder=4)
    up = tr + ScaledTranslation(0, (tick + pad) / 72, ax.figure.dpi_scale_trans)
    ax.text((x0 + x1) / 2, y, text, transform=up, ha="center", va="bottom",
            fontsize=fontsize, color=text_color)


def band_legend(ax, names, line_colors, band_colors, loc="center left", **kw):
    """Legend entries drawn as a short band with a line through it."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch
    handles = [(Patch(color=b, lw=0), Line2D([], [], color=c, lw=1.7))
               for c, b in zip(line_colors, band_colors)]
    return ax.legend(handles, names, loc=loc, handler_map={tuple: HandlerTuple(ndivide=1)},
                     handlelength=1.4, handleheight=0.6, frameon=False, **kw)
