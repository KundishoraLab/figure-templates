"""Lollipop plot for signed per-feature statistics (TF activity, enrichment)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..style import despine, size_legend
from ..theme import Theme, resolve


def lollipop(df, ax, label_col: str = "source", value_col: str = "delta",
             q_col: str | None = "q", top_n: int = 30, q_sig: float = 0.05,
             theme: Theme | None = None,
             up_color: str | None = None, down_color: str | None = None,
             size_range=(25, 115), rotation: int = 60,
             y_label: str = "Δ activity", color_labels: bool = True,
             show_size_legend: bool = True, orient: str = "v"):
    """Signed lollipop: stem length = effect, color = direction, size = -log10(q).

    Selection is by |value| (top_n), then display order is by signed value —
    so the panel shows the strongest movers in both directions and reads
    left-to-right as up -> down. Ranking by raw value instead would silently
    return only the up side whenever any exist, which is how a "no negative
    regulators" artifact gets manufactured.

    Significance is a star strip above the points, not a size threshold, so
    effect size and significance stay independently readable. Pass q_col=None
    when you have no q-values — points get uniform size and no strip.
    """
    t = resolve(theme)
    up_c, dn_c = up_color or t.up, down_color or t.down

    d = pd.DataFrame(df).copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d[d[value_col].notna()].copy()
    if d.empty:
        raise ValueError("lollipop: no finite rows to plot")

    d = d.assign(_abs=d[value_col].abs()).nlargest(top_n, "_abs")
    d = d.sort_values(value_col, ascending=False).reset_index(drop=True)

    x = np.arange(len(d))
    colors = np.where(d[value_col] > 0, up_c, dn_c)

    if q_col and q_col in d.columns:
        nlq = -np.log10(pd.to_numeric(d[q_col], errors="coerce")
                        .fillna(1.0).clip(lower=1e-300))
        lo, hi = float(nlq.min()), float(nlq.max())
        s0, s1 = size_range

        def _size(v):
            return s0 + (s1 - s0) * (v - lo) / max(hi - lo, 1e-9)
        sizes = [_size(v) for v in nlq]
    else:
        sizes = [size_range[1] * 0.5] * len(d)
        _size = None

    vals = d[value_col].values
    stem_lo = np.zeros(len(d))
    if orient == "v":
        for i, v in zip(x, vals):
            ax.plot([i, i], [0, v], color="#999", lw=0.6, zorder=1)
        ax.scatter(x, vals, c=colors, s=sizes, edgecolors="black",
                   linewidths=0.4, zorder=3)
        ax.axhline(0, color="#444", lw=0.6, zorder=2)
    else:
        for i, v in zip(x, vals):
            ax.plot([0, v], [i, i], color="#999", lw=0.6, zorder=1)
        ax.scatter(vals, x, c=colors, s=sizes, edgecolors="black",
                   linewidths=0.4, zorder=3)
        ax.axvline(0, color="#444", lw=0.6, zorder=2)

    vmax, vmin = float(vals.max()), float(vals.min())
    span = (vmax - vmin) or 1.0
    top = vmax + 0.10 * span

    if q_col and q_col in d.columns:
        sig = pd.to_numeric(d[q_col], errors="coerce").fillna(1.0).values < q_sig
        if sig.any():
            if orient == "v":
                ax.scatter(x[sig], np.full(sig.sum(), top), marker="*", s=30,
                           c="black", zorder=4)
            else:
                ax.scatter(np.full(sig.sum(), top), x[sig], marker="*", s=30,
                           c="black", zorder=4)

    ticks = d[label_col].astype(str).values
    if orient == "v":
        ax.set_xticks(x); ax.set_xticklabels(ticks, rotation=rotation,
                                             ha="right", fontsize=8)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_xlim(-0.6, len(d) - 0.4)
        ax.set_ylim(vmin - 0.05 * span, top + 0.12 * span)
        tick_labels = ax.get_xticklabels()
    else:
        ax.set_yticks(x); ax.set_yticklabels(ticks, fontsize=8)
        ax.set_xlabel(y_label, fontsize=10)
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.set_xlim(vmin - 0.05 * span, top + 0.12 * span)
        ax.invert_yaxis()
        tick_labels = ax.get_yticklabels()

    if color_labels:
        for lbl, v in zip(tick_labels, vals):
            lbl.set_color(up_c if v > 0 else dn_c)

    if show_size_legend and _size is not None and len(d) > 1:
        qs = pd.to_numeric(d[q_col], errors="coerce").fillna(1.0)
        refs = sorted({q_sig, float(qs.median()), float(qs.min())})
        refs = [q for q in refs if q > 0]
        if refs:
            size_legend(ax, refs,
                        lambda q: _size(-np.log10(max(q, 1e-300))),
                        label_fn=lambda q: (f"q={q:.0e}" if q < 0.01 else f"q={q:g}"),
                        title="", loc="upper right")
    despine(ax)
    return {"n_shown": len(d), "n_up": int((vals > 0).sum()),
            "n_down": int((vals < 0).sum())}


# Back-compat alias for the original name.
lollipop_tf = lollipop


def dumbbell(ax, labels, a_values, b_values, a_label: str = "A",
             b_label: str = "B", a_color=None, b_color=None,
             y_label: str = "", italic_labels: bool = False,
             legend: bool = True, rotation: int = 0, point_size: float = 16.0,
             connector_color: str = "#C4C4C4", connector_width: float = 1.0,
             legend_loc: str = "upper right", theme: Theme | None = None):
    """Two measurements of the same feature, joined so the gap is the subject.

    `labels`, `a_values` and `b_values` are parallel sequences over one x axis.
    Both values are drawn as points on a shared scale with a connector between
    them, so what the reader measures is the *distance* — which is the quantity
    a two-series bar chart makes hardest to see, because each bar is read from
    the axis and the difference has to be done in the head.

    Nothing is sorted and nothing is dropped. Order is the caller's, because on
    these panels the x axis is usually grouped into blocks that `group_strip`
    then annotates, and re-ordering here would silently break the alignment
    between the strip and the ticks.

    NaN in either series is left as a gap rather than filled or skipped: the
    feature keeps its tick, so the strip below still lines up, and a missing
    measurement does not masquerade as a small one.

    Draws its own key unless `legend=False`, because the two colours are the
    whole encoding. It is a normal axes legend, so a caller that also wants a
    `group_strip` must keep the handle and re-add it:

        dumbbell(ax, ...); leg = ax.get_legend()
        group_strip(ax, counts); ax.add_artist(leg)

    Returns the legend, or None.
    """
    t = resolve(theme)
    a_c = a_color or t.categorical[0]
    b_c = b_color or t.categorical[1]
    labels = list(labels)
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    if not (len(labels) == a.size == b.size):
        raise ValueError(f"dumbbell: {len(labels)} labels but {a.size} a-values "
                         f"and {b.size} b-values")

    x = np.arange(len(labels))
    both = np.isfinite(a) & np.isfinite(b)
    ax.vlines(x[both], a[both], b[both], color=connector_color,
              linewidth=connector_width, zorder=1)
    ax.scatter(x, a, s=point_size, color=a_c, linewidths=0, zorder=3,
               label=a_label)
    ax.scatter(x, b, s=point_size, color=b_c, linewidths=0, zorder=3,
               label=b_label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation,
                       ha="right" if rotation else "center",
                       style="italic" if italic_labels else "normal")
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylabel(y_label)
    # Horizontal only. A vertical gridline through a categorical axis separates
    # nothing that position has not already separated, and here it would run
    # straight down the connector it is meant to sit behind.
    ax.grid(visible=True, axis="y")
    ax.grid(visible=False, axis="x")

    leg = None
    if legend:
        leg = ax.legend(loc=legend_loc, ncol=2, frameon=False, fontsize=7,
                        handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1)
    despine(ax)
    return leg
