"""Bipartite flow diagram.

A sankey answers one question well — *where did each of these go?* — and most
other questions badly. It is the right panel for a label transfer, a cluster
relabelling, or a cohort passing through a filter, and the wrong panel for
anything where the reader needs to compare two ribbons that are not adjacent.
Ribbon width is judged by eye to about 20%, so a sankey is a claim about
"almost all of A became B", never about 31% versus 27%. Put those in a heatmap.

Only two columns of nodes are drawn. Multi-stage sankeys exist, and every one
of them is harder to read than the two-stage version plus a second panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from ..palettes import categorical_palette
from ..theme import Theme, resolve


def _stack(sizes: np.ndarray, pad: float) -> np.ndarray:
    """Node spans down a unit column, `pad` of the height spent on gaps."""
    n = len(sizes)
    total = sizes.sum()
    if total <= 0:
        return np.zeros((n, 2))
    gaps = pad if n > 1 else 0.0
    heights = sizes / total * (1.0 - gaps)
    gap = gaps / (n - 1) if n > 1 else 0.0
    tops = np.concatenate([[0.0], np.cumsum(heights + gap)[:-1]])
    return np.column_stack([tops, tops + heights])


def _ribbon(x0, x1, y0a, y1a, y0b, y1b, curvature: float):
    """Two cubic Beziers and two verticals, closed."""
    xm0 = x0 + (x1 - x0) * curvature
    xm1 = x1 - (x1 - x0) * curvature
    verts = [(x0, y0a), (xm0, y0a), (xm1, y0b), (x1, y0b),
             (x1, y1b), (xm1, y1b), (xm0, y1a), (x0, y1a), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    return Path(verts, codes)


def _barycentre(order_a, order_b, weights, n_sweeps: int = 4):
    """Order both margins by the mean position of what they connect to.

    The standard crossing-reduction heuristic. It is a heuristic: it does not
    find the minimum, and on a dense matrix it will not find anything much,
    because a dense matrix has no readable ordering to find. Pass an explicit
    order whenever the labels have a meaning — a canonical cell-type order the
    reader already knows beats a marginally tidier picture.
    """
    a, b = np.asarray(order_a), np.asarray(order_b)
    for _ in range(n_sweeps):
        w = weights[a][:, b]
        pos = np.arange(len(a), dtype=float)
        col = w.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = b[np.argsort(np.where(col > 0, (pos @ w) / col, np.inf),
                             kind="stable")]
        w = weights[a][:, b]
        pos = np.arange(len(b), dtype=float)
        row = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            a = a[np.argsort(np.where(row > 0, (w @ pos) / row, np.inf),
                             kind="stable")]
    return a, b


def sankey(flows, ax, palette=None, target_palette=None,
           source_order=None, target_order=None, normalize: str = "none",
           min_flow: float = 0.0, node_width: float = 0.035,
           node_pad: float = 0.18, curvature: float = 0.4,
           alpha: float = 0.55, label_fontsize: float = 10,
           source_label: str = "", target_label: str = "",
           reduce_crossings: bool = True, theme: Theme | None = None):
    """Draw a two-column sankey from a source x target matrix.

    flows : DataFrame, rows are source nodes and columns target nodes. A
        `pd.crosstab` goes straight in.

    normalize : "none" draws raw magnitudes, so a node's height is its n and
        big sources dominate — correct when the reader should see that most of
        the data went one way. "source" rescales every row to the same height,
        which turns each ribbon into "% of this source" and makes a 400-cell
        cluster as legible as a 40,000-cell one. That is usually what a label
        transfer wants, and it *destroys the n*: say so in the caption, or the
        panel silently promises a rare cluster is as well-supported as a
        common one.

    min_flow : drop ribbons below this, in the same units as the (possibly
        normalized) values. Sub-percent ribbons render as hairlines that read
        as noise but cost the reader the same attention as real ones. Dropping
        them is a claim in itself, so it belongs in the caption too; the
        returned dict reports how much was dropped.

    Colour follows the source node, because the question a sankey answers is
    about where things went, not where they came from. Pass `target_palette`
    to give the right-hand nodes their own colours; they default to the
    theme's neutral so they read as destinations rather than categories.
    """
    t = resolve(theme)
    df = pd.DataFrame(flows).fillna(0.0)
    if (df.values < 0).any():
        raise ValueError("sankey flows must be non-negative")

    if normalize == "source":
        totals = df.sum(axis=1).replace(0, np.nan)
        df = df.div(totals, axis=0).fillna(0.0)
    elif normalize != "none":
        raise ValueError("normalize must be 'none' or 'source'")

    if source_order is not None:
        df = df.loc[[s for s in source_order if s in df.index]]
    if target_order is not None:
        df = df[[c for c in target_order if c in df.columns]]

    dropped = float(df.values[df.values < min_flow].sum()) if min_flow else 0.0
    w = np.where(df.values >= min_flow, df.values, 0.0)

    ia, ib = np.arange(w.shape[0]), np.arange(w.shape[1])
    if reduce_crossings and source_order is None and target_order is None:
        ia, ib = _barycentre(ia, ib, w)
    elif reduce_crossings and target_order is None:
        _, ib = _barycentre(ia, ib, w, n_sweeps=1)
    w = w[ia][:, ib]
    sources = df.index[ia]
    targets = df.columns[ib]

    if palette is None:
        palette = categorical_palette(sources, t.categorical)

    left = _stack(w.sum(axis=1), node_pad)
    right = _stack(w.sum(axis=0), node_pad)
    lh = np.diff(left, axis=1).ravel()
    rh = np.diff(right, axis=1).ravel()
    lsum = w.sum(axis=1)
    rsum = w.sum(axis=0)

    x0, x1 = node_width, 1.0 - node_width
    cursor_l = left[:, 0].copy()
    cursor_r = right[:, 0].copy()
    n_links = 0
    for i in range(w.shape[0]):
        color = palette.get(sources[i], t.na)
        for j in range(w.shape[1]):
            v = w[i, j]
            if v <= 0:
                continue
            ha = v / lsum[i] * lh[i]
            hb = v / rsum[j] * rh[j]
            ax.add_patch(PathPatch(
                _ribbon(x0, x1, cursor_l[i], cursor_l[i] + ha,
                        cursor_r[j], cursor_r[j] + hb, curvature),
                facecolor=color, edgecolor="none", alpha=alpha, zorder=1))
            cursor_l[i] += ha
            cursor_r[j] += hb
            n_links += 1

    for i, name in enumerate(sources):
        ax.add_patch(Rectangle((0.0, left[i, 0]), node_width, lh[i], zorder=3,
                               facecolor=palette.get(name, t.na),
                               edgecolor="none"))
        ax.text(-0.015, left[i].mean(), str(name), ha="right", va="center",
                fontsize=label_fontsize, zorder=3)
    for j, name in enumerate(targets):
        c = (target_palette or {}).get(name, t.na)
        ax.add_patch(Rectangle((1.0 - node_width, right[j, 0]), node_width,
                               rh[j], facecolor=c, edgecolor="none", zorder=3))
        ax.text(1.015, right[j].mean(), str(name), ha="left", va="center",
                fontsize=label_fontsize, zorder=3)

    if source_label:
        ax.text(0.0, -0.03, source_label, ha="left", va="top",
                fontsize=label_fontsize, fontweight="bold")
    if target_label:
        ax.text(1.0, -0.03, target_label, ha="right", va="top",
                fontsize=label_fontsize, fontweight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(1.02, -0.02)
    ax.set_axis_off()
    return {"n_sources": len(sources), "n_targets": len(targets),
            "n_links": n_links, "flow_dropped": dropped,
            "source_order": list(sources), "target_order": list(targets)}
