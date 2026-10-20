"""NEJM-style subgroup forest plot (text columns, striped rows, log axis, 'X Better' arrows)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory, ScaledTranslation

from .style import COLORS


@dataclass
class ForestRow:
    label: str
    n: int | str | None = None
    est: float | None = None
    lo: float | None = None
    hi: float | None = None
    indent: int = 0           # 0 = header/overall, 1 = subgroup level


def plot_forest(ax, rows: Sequence[ForestRow], *, xlim=(0.2, 3.0), xticks=(0.25, 0.5, 1.0, 2.0),
                xticklabels=None, ref=1.0, log=True,
                cols=None, headers=("Subgroup", "No. of Patients", "Hazard Ratio (95% CI)"),
                est_fmt="{:.2f} ({:.2f}–{:.2f})", shade=True, shade_color=COLORS["shade"],
                shade_extent=None, marker="s", ms=5.5, capsize=2.5, lw=1.0, color="black",
                better=None, arrow_offset=26, better_offset=46, indent_width=0.035,
                fontsize=None, header_weight="semibold", row_height=1.0, header_gap=1.0):
    """Draw a forest plot on `ax`, with text columns placed in axes-fraction x coordinates.

    cols: dict with keys label, n, est (x positions, axes fraction) and n_ha / est_ha.
          Text columns live outside the axes, so leave room in your layout.
    better: ("Drug Better", "Placebo Better") labels under the arrows (None to skip).
    """
    cols = {"label": -1.08, "n": -0.55, "n_ha": "right", "est": 1.85, "est_ha": "right",
            "h_label": -1.08, "h_n": -0.72, "h_est": 1.85, "h_est_ha": "right", **(cols or {})}
    shade_extent = shade_extent or (cols["label"] - 0.02, cols["est"] + 0.02)

    N = len(rows)
    ys = -np.arange(N) * row_height
    ax.set_ylim(ys[-1] - 0.6 * row_height, 0.6 * row_height)
    ax.patch.set_visible(False)
    tr_rows = blended_transform_factory(ax.transAxes, ax.transData)

    for i, (r, y) in enumerate(zip(rows, ys)):
        if shade and i % 2 == 0:
            ax.add_patch(Rectangle((shade_extent[0], y - 0.5 * row_height),
                                   shade_extent[1] - shade_extent[0], row_height,
                                   transform=tr_rows, color=shade_color, lw=0, zorder=0,
                                   clip_on=False))
        ax.text(cols["label"] + indent_width * r.indent, y, r.label, transform=tr_rows,
                ha="left", va="center", fontsize=fontsize, clip_on=False)
        if r.n is not None:
            ax.text(cols["n"], y, f"{r.n}", transform=tr_rows, ha=cols["n_ha"], va="center",
                    fontsize=fontsize, clip_on=False)
        if r.est is not None:
            ax.errorbar([r.est], [y], xerr=[[r.est - r.lo], [r.hi - r.est]], fmt=marker,
                        ms=ms, color=color, ecolor=color, elinewidth=lw, capsize=capsize,
                        capthick=lw, zorder=3, clip_on=False)
            ax.text(cols["est"], y, est_fmt.format(r.est, r.lo, r.hi), transform=tr_rows,
                    ha=cols["est_ha"], va="center", fontsize=fontsize, clip_on=False)

    hy = ys[0] + header_gap * row_height
    if headers:
        for key, h in zip(("h_label", "h_n", "h_est"), headers):
            ax.text(cols[key], hy, h, transform=tr_rows, va="center", fontsize=fontsize,
                    fontweight=header_weight, clip_on=False,
                    ha=cols.get(key + "_ha", "left"))

    if log:
        ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(list(xticks)))
    ax.xaxis.set_major_formatter(FixedFormatter(
        list(xticklabels) if xticklabels else [f"{t:.2f}" for t in xticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(ref, color="black", lw=1.0, zorder=2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["bottom"].set_bounds(*xlim)

    if better:
        fig = ax.figure
        tr = blended_transform_factory(ax.transData, ax.transAxes)

        def down(pts):
            return tr + ScaledTranslation(0, -pts / 72, fig.dpi_scale_trans)

        f = 0.93 if log else 0.0
        l_start, l_end = ref * f, xlim[0] * 1.05
        r_start, r_end = ref / f, xlim[1] * 0.96
        for s, e in ((l_start, l_end), (r_start, r_end)):
            ax.annotate("", xy=(e, 0), xycoords=down(arrow_offset), xytext=(s, 0),
                        textcoords=down(arrow_offset),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black",
                                        mutation_scale=10, shrinkA=0, shrinkB=0))
        for lab, (s, e) in zip(better, ((l_start, l_end), (r_start, r_end))):
            mid = np.sqrt(s * e) if log else (s + e) / 2
            ax.text(mid, 0, lab, transform=down(better_offset), ha="center", va="center",
                    fontweight="bold", fontsize=fontsize, clip_on=False)
    return ys
