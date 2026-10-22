"""NEJM-style subgroup forest plot (text columns, striped rows, log axis, 'X Better' arrows)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.patches import Polygon, Rectangle
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
    pooled: bool = False      # True draws a diamond spanning the interval (a pooled / overall estimate)
    p: float | str | None = None  # a P value for the optional P column (a float is formatted with p_fmt)


def format_p_cell(p):
    """A P value as an NEJM table cell: "<0.001", three places below 0.01, otherwise two (half-up)."""
    import decimal
    if p < 0.001:
        return "<0.001"
    q = decimal.Decimal("0.001" if p < 0.01 else "0.01")
    return str(decimal.Decimal(repr(float(p))).quantize(q, rounding=decimal.ROUND_HALF_UP))


def plot_forest(ax, rows: Sequence[ForestRow], *, xlim=(0.2, 3.0), xticks=(0.25, 0.5, 1.0, 2.0),
                xticklabels=None, ref=1.0, log=True,
                cols=None, headers=("Subgroup", "No. of Patients", "Hazard Ratio (95% CI)"),
                est_fmt="{:.2f} ({:.2f}–{:.2f})", shade=True, shade_color=COLORS["shade"],
                shade_extent=None, marker="s", ms=5.5, capsize=2.5, lw=1.0, color="black",
                better=None, arrow_offset=26, better_offset=46, indent_width=0.035,
                fontsize=None, header_weight="semibold", row_height=1.0, header_gap=1.0,
                xbreak=None, break_gap=0.035, xlabel=None, xlabel_offset=None, p_fmt=format_p_cell):
    """Draw a forest plot on `ax`, with text columns placed in axes-fraction x coordinates.

    cols: dict with keys label, n, est (x positions, axes fraction) and n_ha / est_ha; an optional
          "p" key (with p_ha / h_p) draws each row's ForestRow.p in a P column, and a fourth header names it.
          Text columns live outside the axes, so leave room in your layout.
    better: ("Drug Better", "Placebo Better") labels under the arrows (None to skip).
    xbreak: (hi_left, lo_right) splits the axis into two segments, xlim[0]..hi_left and
            lo_right..xlim[1], with a gap of `break_gap` (axes fraction) and break marks; each
            segment keeps the scale, gets the ticks that fall inside it, and its share of the
            width follows its span. Intervals are clipped to the segments (a wide interval
            runs into the gap and continues on the other side).
    xlabel: a title under the axis (centred on it; with `better` it sits under the arrow labels).
    """
    segs = [(xlim[0], xbreak[0]), (xbreak[1], xlim[1])] if xbreak else [tuple(xlim)]

    def span(a, b):
        return np.log10(b / a) if log else b - a

    total = sum(span(*sg) for sg in segs)
    gap = break_gap if xbreak else 0.0
    starts, widths = [], []
    x0 = 0.0
    for sg in segs:
        w = span(*sg) / total * (1 - gap * (len(segs) - 1))
        starts.append(x0); widths.append(w); x0 += w + gap

    def xfrac(v):
        """A data value's position in parent-axes fraction, through the segments."""
        for sg, s0, w in zip(segs, starts, widths):
            if sg[0] <= v <= sg[1] or sg is segs[-1] and v > sg[1] or sg is segs[0] and v < sg[0]:
                return s0 + span(sg[0], v) / span(*sg) * w
        # in the gap: the right edge of the left segment
        return starts[0] + widths[0]

    cols = {"label": -1.08, "n": -0.55, "n_ha": "right", "est": 1.85, "est_ha": "right",
            "h_label": -1.08, "h_n": -0.72, "h_est": 1.85, "h_est_ha": "right", **(cols or {})}
    shade_extent = shade_extent or (cols["label"] - 0.02, cols["est"] + 0.02)

    N = len(rows)
    ys = -np.arange(N) * row_height
    ax.set_ylim(ys[-1] - 0.6 * row_height, 0.6 * row_height)
    ax.patch.set_visible(False)
    tr_rows = blended_transform_factory(ax.transAxes, ax.transData)
    if xbreak:                                  # the data live on one inset axes per segment
        data_axes = []
        for sg, s0, w in zip(segs, starts, widths):
            a_ = ax.inset_axes([s0, 0, w, 1])
            a_.patch.set_visible(False)
            a_.set_ylim(*ax.get_ylim())
            for side in ("left", "right", "top"):
                a_.spines[side].set_visible(False)
            a_.tick_params(axis="y", left=False, labelleft=False)
            data_axes.append((a_, sg))
    else:
        data_axes = [(ax, tuple(xlim))]

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
            for a_, _ in data_axes:
                if r.pooled:                    # the diamond: its width is the interval
                    hh = 0.32 * row_height
                    a_.add_patch(Polygon([(r.lo, y), (r.est, y + hh), (r.hi, y), (r.est, y - hh)], closed=True,
                                         facecolor=color, edgecolor=color, lw=lw, zorder=3, clip_on=bool(xbreak)))
                else:
                    a_.errorbar([r.est], [y], xerr=[[r.est - r.lo], [r.hi - r.est]], fmt=marker,
                                ms=ms, color=color, ecolor=color, elinewidth=lw, capsize=capsize,
                                capthick=lw, zorder=3, clip_on=bool(xbreak))
            ax.text(cols["est"], y, est_fmt.format(r.est, r.lo, r.hi), transform=tr_rows,
                    ha=cols["est_ha"], va="center", fontsize=fontsize, clip_on=False)
        if "p" in cols and r.p is not None:
            ax.text(cols["p"], y, r.p if isinstance(r.p, str) else p_fmt(r.p), transform=tr_rows,
                    ha=cols.get("p_ha", "right"), va="center", fontsize=fontsize, clip_on=False)

    hy = ys[0] + header_gap * row_height
    if headers:
        for key, h in zip(("h_label", "h_n", "h_est", "h_p"), headers):
            ax.text(cols[key], hy, h, transform=tr_rows, va="center", fontsize=fontsize,
                    fontweight=header_weight, clip_on=False,
                    ha=cols.get(key + "_ha", "left"))

    labels = list(xticklabels) if xticklabels else [f"{t:.2f}" for t in xticks]
    for a_, sg in data_axes:
        if log:
            a_.set_xscale("log")
        a_.set_xlim(*sg)
        keep = [(t, l) for t, l in zip(xticks, labels) if sg[0] <= t <= sg[1]]
        a_.xaxis.set_major_locator(FixedLocator([t for t, _ in keep]))
        a_.xaxis.set_major_formatter(FixedFormatter([l for _, l in keep]))
        a_.xaxis.set_minor_locator(NullLocator())
        if sg[0] <= ref <= sg[1]:
            a_.axvline(ref, color="black", lw=1.0, zorder=2)
        a_.spines["bottom"].set_bounds(*sg)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    if xbreak:
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(0, 1)
        dx, dy = 0.006, 0.025                   # the break marks: a pair of slanted strokes at each cut
        for xc in (starts[0] + widths[0], starts[1]):
            ax.plot([xc - dx, xc + dx], [-dy, dy], transform=ax.transAxes, color="black",
                    lw=1.0, clip_on=False, zorder=4)
    else:
        ax.spines["bottom"].set_bounds(*xlim)

    fig = ax.figure

    def down(pts):
        return ax.transAxes + ScaledTranslation(0, -pts / 72, fig.dpi_scale_trans)

    if xlabel:
        off = xlabel_offset if xlabel_offset is not None else (better_offset + 16 if better else arrow_offset)
        ax.text(0.5, 0, xlabel, transform=down(off), ha="center", va="center", fontsize=fontsize, clip_on=False)
    if better:

        # the arrows run in parent-axes fraction (through the segments when the axis is split):
        # from a little either side of the reference to a little inside each end
        xr, d = xfrac(ref), 0.035
        l_start, r_start, l_end, r_end = xr - d, xr + d, 0.04, 0.96
        for s, e in ((l_start, l_end), (r_start, r_end)):
            ax.annotate("", xy=(e, 0), xycoords=down(arrow_offset), xytext=(s, 0),
                        textcoords=down(arrow_offset),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black",
                                        mutation_scale=10, shrinkA=0, shrinkB=0))
        for lab, (s, e) in zip(better, ((l_start, l_end), (r_start, r_end))):
            ax.text((s + e) / 2, 0, lab, transform=down(better_offset), ha="center", va="center",
                    fontweight="bold", fontsize=fontsize, clip_on=False)
    return ys
