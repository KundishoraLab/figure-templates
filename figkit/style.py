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
        # Grey-93 major grid, matching the R track's theme_avm(). A grid is a
        # reading aid and must sit behind the data — axisbelow=True is what
        # stops gridlines being drawn over a violin body and looking like
        # structure in the density.
        "grid.color": "#EDEDED", "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True,
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
