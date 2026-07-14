"""Composition bars."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..palettes import categorical_palette
from ..style import despine
from ..theme import Theme, resolve


def stacked_bar(df, ax, palette=None, order=None, normalize: bool = True,
                orient: str = "h", x_label: str | None = None,
                y_label: str = "", legend: bool = True,
                legend_title: str = "", min_label_pct: float | None = None,
                label_fmt: str = "{:.0f}%", theme: Theme | None = None,
                edge_color: str = "white", bar_width: float = 0.8):
    """Stacked composition bar. Rows = groups, columns = categories.

    normalize=True converts each row to fractions of its own total, which is
    what you want when group sizes differ — raw stacked counts conflate "this
    group is bigger" with "this group has a different composition".

    When you normalize, the group's n stops being visible. Put it in the row
    label (e.g. "Sample A (n=812)") or the reader cannot tell a 50% built
    from 4 cells from one built from 4000.

    min_label_pct : write the percentage inside segments at least this large.
        Leave None for a clean fill; the numbers belong in a table.
    """
    t = resolve(theme)
    df = pd.DataFrame(df)
    if order is not None:
        df = df[[c for c in order if c in df.columns]]
    if palette is None:
        palette = categorical_palette(df.columns, t.categorical)

    d = df.div(df.sum(axis=1), axis=0).fillna(0) if normalize else df.fillna(0)
    pos = np.arange(len(d))
    offset = np.zeros(len(d))

    for col in d.columns:
        vals = d[col].values
        color = palette.get(col, t.na)
        if orient == "h":
            ax.barh(pos, vals, left=offset, color=color, label=str(col),
                    edgecolor=edge_color, linewidth=0.3, height=bar_width)
        else:
            ax.bar(pos, vals, bottom=offset, color=color, label=str(col),
                   edgecolor=edge_color, linewidth=0.3, width=bar_width)
        if min_label_pct is not None:
            for i, v in enumerate(vals):
                pct = v * 100 if normalize else v
                if pct >= min_label_pct:
                    cx, cy = offset[i] + v / 2, pos[i]
                    if orient != "h":
                        cx, cy = cy, cx
                    ax.text(cx, cy, label_fmt.format(pct), ha="center",
                            va="center", fontsize=7, color="white")
        offset = offset + vals

    default_x = "fraction" if normalize else "count"
    if orient == "h":
        ax.set_yticks(pos); ax.set_yticklabels(d.index, fontsize=11)
        ax.set_xlabel(x_label if x_label is not None else default_x)
        ax.set_ylabel(y_label)
        if normalize:
            ax.set_xlim(0, 1)
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels(d.index, fontsize=11, rotation=45, ha="right")
        ax.set_ylabel(x_label if x_label is not None else default_x)
        ax.set_xlabel(y_label)
        if normalize:
            ax.set_ylim(0, 1)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                  fontsize=9, title=legend_title,
                  ncol=1 if d.shape[1] <= 12 else 2)
    despine(ax)
    return {"n_groups": len(d), "n_categories": d.shape[1]}


# Back-compat alias for the original name.
stacked_hbar = stacked_bar


def ordered_bar(df, ax, value_col: str, label_col: str = None, order=None,
                palette=None, sig_col: str | None = None,
                x_label: str = "", theme: Theme | None = None,
                orient: str = "h", bar_width: float = 0.72):
    """Bar of one statistic per category, held in a caller-supplied `order`.

    `order` exists so a categorical axis with real structure (a developmental
    series, an anatomical axis, a dose ladder) keeps that structure instead
    of being re-sorted by value. Categories not in `order` are dropped, and
    it says how many.

    sig_col : column of pre-rendered annotations (e.g. "*", "n.s.") drawn at
        the bar tip.
    """
    t = resolve(theme)
    d = pd.DataFrame(df).copy()
    labels = d[label_col] if label_col else d.index
    d = d.assign(_label=pd.Series(labels, index=d.index).astype(str))

    if order is not None:
        order = [str(o) for o in order]
        keep = d["_label"].isin(order)
        if (~keep).any():
            print(f"  [ordered_bar] dropped {int((~keep).sum())} rows not in `order`: "
                  f"{sorted(d.loc[~keep, '_label'].unique())[:6]}")
        d = d[keep].copy()
        d["_label"] = pd.Categorical(d["_label"], categories=order, ordered=True)
        d = d.sort_values("_label")
    if d.empty:
        raise ValueError("ordered_bar: no rows left to plot")

    if palette is None:
        palette = categorical_palette(d["_label"].astype(str), t.categorical)
    colors = [palette.get(str(l), t.na) for l in d["_label"]]
    pos = np.arange(len(d))
    vals = pd.to_numeric(d[value_col], errors="coerce").fillna(0).values

    if orient == "h":
        ax.barh(pos, vals, color=colors, edgecolor="white", linewidth=0.3,
                height=bar_width)
        ax.set_yticks(pos); ax.set_yticklabels(d["_label"].astype(str), fontsize=9)
        ax.set_xlabel(x_label)
        ax.invert_yaxis()  # first entry of `order` on top
    else:
        ax.bar(pos, vals, color=colors, edgecolor="white", linewidth=0.3,
               width=bar_width)
        ax.set_xticks(pos)
        ax.set_xticklabels(d["_label"].astype(str), rotation=45, ha="right",
                           fontsize=9)
        ax.set_ylabel(x_label)

    if sig_col and sig_col in d.columns:
        span = (np.nanmax(vals) - min(0, np.nanmin(vals))) or 1.0
        for p, v, s in zip(pos, vals, d[sig_col].astype(str)):
            if not s or s.lower() in ("nan", "none"):
                continue
            if orient == "h":
                ax.text(v + 0.02 * span, p, s, va="center", ha="left",
                        fontsize=8, color="#404040")
            else:
                ax.text(p, v + 0.02 * span, s, ha="center", va="bottom",
                        fontsize=8, color="#404040")
    despine(ax)
    return {"n_bars": len(d)}
