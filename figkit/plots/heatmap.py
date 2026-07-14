"""Heatmaps."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..palettes import quantile_norm
from ..theme import Theme, resolve


def heatmap(df, ax, cmap=None, vmin=None, vmax=None, symmetric: bool = False,
            cbar_label: str = "", colorbar: bool = True,
            row_label_size: float = 9, col_label_size: float = 9,
            col_label_rotation: float = 90, na_color: str | None = None,
            annotate: bool = False, annot_fmt: str = "{:.2f}",
            annot_size: float = 7, theme: Theme | None = None):
    """ComplexHeatmap-style heatmap from a rows x columns DataFrame.

    NaN cells render in `na_color` (grey by default) rather than as a
    colormap value — "missing" must not be mistakable for "zero" or "low".

    symmetric=True centers the scale on zero, which is required whenever
    `cmap` is diverging. Limits come from the 5th/95th percentile unless you
    pass vmin/vmax, so a couple of extreme cells don't flatten the rest.

    col_label_rotation=0 is the right call for a single- or few-column
    heatmap, where vertical labels are just harder to read.
    """
    t = resolve(theme)
    df = pd.DataFrame(df)
    cmap = cmap if cmap is not None else (t.diverging if symmetric else t.heatmap_cmap)
    arr = df.values.astype(float)

    if vmin is None or vmax is None:
        n = quantile_norm(arr, 0.05, 0.95, symmetric=symmetric)
        vmin = n.vmin if vmin is None else vmin
        vmax = n.vmax if vmax is None else vmax

    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    cmap.set_bad(na_color or t.na)

    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="none",
                   rasterized=arr.size > 5000)

    ax.set_xticks(np.arange(df.shape[1]))
    ha = "right" if col_label_rotation not in (0, 360) else "center"
    ax.set_xticklabels(df.columns, rotation=col_label_rotation,
                       fontsize=col_label_size, ha=ha)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index, rotation=0, fontsize=row_label_size)

    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.6)

    if annotate:
        # Flip text to white on dark cells so every value stays legible.
        rng = (vmax - vmin) or 1.0
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = arr[i, j]
                if not np.isfinite(v):
                    continue
                freq = abs(v - (vmin + vmax) / 2) / (rng / 2)
                ax.text(j, i, annot_fmt.format(v), ha="center", va="center",
                        fontsize=annot_size,
                        color="white" if freq > 0.6 else "#222")

    if colorbar:
        import matplotlib.pyplot as plt
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=9)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=10)
    return im


# Back-compat alias for the original name.
heatmap_complex = heatmap
