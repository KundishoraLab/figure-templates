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

        # Clamped, and the degenerate range answered explicitly rather than
        # by a floored denominator. `max(hi - lo, 1e-9)` looks like a
        # divide-by-zero guard and is not one: when every q is equal it turns
        # a range of zero into a slope of 1e9, so any value off the observed
        # range extrapolates to an absurd area. `size_legend` calls this
        # function at `q_sig`, which is off the range exactly when nothing is
        # significant -- a panel where every q was 1.0 asked for the area at
        # q = 0.05 and got 1.2e11 pt^2, a marker 342,000 pt across. The PDF
        # backend writes that as coordinates and the page looks normal; Agg
        # tries to rasterise it and the process is killed with no traceback,
        # so the failure reads as a crash rather than as a plotting bug.
        # Clamping also fixes the non-degenerate case, where the key could
        # otherwise show an area larger than any mark on the panel.
        flat = (hi - lo) < 1e-9

        def _size(v):
            if flat:
                return s0
            return s0 + (s1 - s0) * (min(max(v, lo), hi) - lo) / (hi - lo)
        sizes = [_size(v) for v in nlq]
    else:
        sizes = [size_range[1] * 0.5] * len(d)
        _size, flat = None, True

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

    # No key when every mark is the same size: it would key an area to a
    # value the panel does not distinguish.
    if show_size_legend and _size is not None and len(d) > 1 and not flat:
        qs = pd.to_numeric(d[q_col], errors="coerce").fillna(1.0)
        refs = sorted({q_sig, float(qs.median()), float(qs.min())})
        # Only reference values the panel actually spans. `q_sig` is a useful
        # anchor when some mark is near it and a lie when none is: with the
        # sizes clamped, a `q_sig` below every observed q keys at the same area
        # as the smallest observed q, so the key shows two different numbers
        # drawn at one size. Dropping it says "nothing here is near 0.05" by
        # omission, which is the truth the panel is carrying anyway.
        refs = [q for q in refs
                if q > 0 and float(qs.min()) <= q <= float(qs.max())]
        if refs:
            size_legend(ax, refs,
                        lambda q: _size(-np.log10(max(q, 1e-300))),
                        label_fn=lambda q: (f"q={q:.0e}" if q < 0.01 else f"q={q:g}"),
                        # "best", not "upper right". The star strip sits at the
                        # top of the data range, so a significant mark near the
                        # right edge lands under a corner-anchored key and the
                        # star that carries the panel's verdict is drawn through
                        # its own label. Let matplotlib place the key where the
                        # artists are not.
                        title="", loc="best")
    despine(ax)
    return {"n_shown": len(d), "n_up": int((vals > 0).sum()),
            "n_down": int((vals < 0).sum())}


# Back-compat alias for the original name.
lollipop_tf = lollipop
