"""Figure-wide rcParams and panel output.

Two jobs, both boring and both load-bearing:
  apply_rcparams()  — one call at script start, so every panel matches
  save_panel()      — write a panel to disk in publication-ready formats
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .palettes import YLGNBU3, NAVY_YELLOW, PUOR_DIV
from .theme import Theme, resolve


def apply_rcparams(theme: Theme | None = None, grid: bool = False) -> None:
    """Set publication defaults. Call once at script start, before plotting.

    The font-type settings are the important part: `pdf.fonttype = 42` and
    `svg.fonttype = "none"` keep text as real text rather than outlines, so
    the panel stays editable in Illustrator/Inkscape and a typo is a text
    edit instead of a re-run. Most journals require this for vector figures.
    """
    t = resolve(theme)
    for cm in (YLGNBU3, NAVY_YELLOW, PUOR_DIV):
        try:
            mpl.colormaps.register(cm)
        except (ValueError, AttributeError):
            pass  # already registered
    s = t.base_size
    mpl.rcParams.update({
        "font.family": list(t.font_family),
        "font.size": s,
        "axes.titlesize": s, "axes.labelsize": s,
        "xtick.labelsize": s, "ytick.labelsize": s,
        "legend.fontsize": s, "legend.title_fontsize": s,
        "axes.linewidth": 0.6, "axes.edgecolor": "black",
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": grid,
        "savefig.dpi": t.dpi, "savefig.bbox": "tight",
        # Keep text editable in vector output — see docstring.
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def save_panel(fig, name: str, outdir, formats=("png", "svg"),
               dpi: int | None = None, close: bool = True) -> list[Path]:
    """Write `fig` to outdir/<name>.<ext> for each format. Returns the paths.

    PNG for looking at, SVG (or PDF) for the composite — ship both so the
    figure assembler never has to re-run the producer to get a vector copy.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        dpi = mpl.rcParams.get("savefig.dpi", 600)
        if not isinstance(dpi, (int, float)):
            dpi = 600
    written = []
    for ext in formats:
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    if close:
        plt.close(fig)
    return written


def despine(ax, sides=("top", "right")) -> None:
    """Hide the named spines. despine(ax, 'all') hides every one."""
    if sides == "all":
        sides = ("top", "right", "bottom", "left")
    for s in sides:
        ax.spines[s].set_visible(False)


def size_legend(ax, values, size_fn, label_fn=str, title: str = "",
                loc: str = "upper right", color: str = "#666"):
    """Legend keying marker area back to the value it encodes.

    A size-encoded plot without this is unreadable — the viewer can see that
    one dot is bigger but cannot say how much bigger. `size_fn` must be the
    same value -> area function the plot used.
    """
    from matplotlib.lines import Line2D
    import numpy as np
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markersize=float(np.sqrt(max(size_fn(v), 1.0))),
               markeredgecolor="black", markeredgewidth=0.4,
               label=label_fn(v))
        for v in values
    ]
    return ax.legend(handles=handles, loc=loc, frameon=False, fontsize=8,
                     labelspacing=0.7, handletextpad=0.5, title=title,
                     title_fontsize=9)


def group_strip(ax, counts, palette=None, legend_cols: int = 3,
                extra_handles=None, y: float = -0.02, height: float = 0.022,
                gap: float = 0.004, legend_y: float = -0.34,
                fontsize: float = 8, theme: Theme | None = None):
    """Annotate contiguous runs of the x axis with a colour bar and a key.

    `counts` is an ordered {group: n_categories} mapping over the x axis, read
    left to right: the first group takes the first n ticks, the next the next,
    and so on. Nothing is inferred from the tick labels, because a gene can
    belong to two modules and the caller is the only one who knows which run it
    was drawn in.

    This exists instead of text over the plot. Labelling groups in the data area
    puts ink where the reader is measuring, and it collides as soon as a
    category is tall; a strip under the axis is out of the way and stays
    readable at any data range. Group order must match the plotting order or
    the strip is silently wrong — the strip cannot detect that, so keep the same
    ordered mapping that built the x axis.

    `extra_handles` are appended to the key, for the case where the panel also
    size- or shade-encodes something: one legend below a panel reads better than
    two competing for its corners. Everything is in axes coordinates so the
    strip tracks the axes through `tight_layout`, and the caller reserves room
    with `fig.tight_layout(rect=...)`.

    Replaces any existing legend on `ax`. If the panel already had one, keep the
    handle and `ax.add_artist(it)` afterwards.
    """
    import matplotlib.patches as mpatches

    t = resolve(theme)
    pal = palette if palette is not None else {
        g: t.categorical[i % len(t.categorical)] for i, g in enumerate(counts)
    }

    start = 0
    n_total = sum(counts.values())
    for g, n in counts.items():
        if n <= 0:
            continue
        x0, x1 = start / n_total, (start + n) / n_total
        ax.add_patch(mpatches.Rectangle(
            (x0, y - height), max(x1 - x0 - gap, 0.0), height,
            transform=ax.transAxes, clip_on=False, linewidth=0,
            facecolor=pal[g], zorder=5))
        start += n

    handles = [mpatches.Patch(facecolor=pal[g], edgecolor="none", label=g)
               for g, n in counts.items() if n > 0]
    handles += list(extra_handles or [])
    return ax.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, legend_y), ncol=legend_cols,
                     frameon=False, fontsize=fontsize, handlelength=1.1,
                     handletextpad=0.5, columnspacing=1.4, borderaxespad=0)
