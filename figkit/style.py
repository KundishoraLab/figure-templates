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
    from matplotlib.legend import Legend

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        dpi = mpl.rcParams.get("savefig.dpi", 600)
        if not isinstance(dpi, (int, float)):
            dpi = 600
    # A legend anchored below the axes has to be excluded from layout, or
    # tight_layout reserves room proportional to how far below it sits and
    # crushes the panel. The tight bbox honours the same flag, so without this
    # the key would be laid out correctly and then cropped off the page.
    extra = fig.get_default_bbox_extra_artists()
    extra += [a for ax in fig.axes for a in ax.get_children()
              if isinstance(a, Legend) and a not in extra]
    written = []
    for ext in formats:
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", bbox_extra_artists=extra)
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
                extra_handles=None, pad_pt: float = 3.6, height: float = 0.036,
                legend_y: float = -0.60, fontsize: float = 7.5,
                theme: Theme | None = None):
    """Annotate contiguous runs of the x axis with a colour bar and a key.

    `counts` is an ordered {group: n_categories} mapping over the x axis, read
    left to right: the first group takes the first n ticks, the next the next,
    and so on. Nothing is inferred from the tick labels, because a gene can
    belong to two modules and the caller is the only one who knows which run it
    was drawn in.

    A run spans its ticks in *data* x, from `start - 0.5` to `start + n - 0.5`,
    so a boundary falls halfway between the two ticks it separates. Spreading
    the runs across the full axes width instead would put the boundary wherever
    the x margins happen to leave it, off by a fraction of a category.

    `pad_pt` is in points, not axes fraction, because it exists to clear the
    tick marks — which are a fixed physical length whatever the panel is. At
    the default the strip starts just past a 3.5 pt tick; shrink the panel with
    a fractional offset instead and the ticks end up drawn through the strip.

    This exists instead of text over the plot. Labelling groups in the data area
    puts ink where the reader is measuring, and it collides as soon as a
    category is tall; a strip under the axis is out of the way and stays
    readable at any data range. Group order must match the plotting order or
    the strip is silently wrong — the strip cannot detect that, so keep the same
    ordered mapping that built the x axis.

    `extra_handles` are appended to the key, for the case where the panel also
    size- or shade-encodes something: one legend below a panel reads better than
    two competing for its corners. The caller reserves room for the key with
    `fig.tight_layout(rect=...)`; the key itself is excluded from layout so that
    a key anchored well below the axes cannot squeeze the panel it belongs to.

    Replaces any existing legend on `ax`. If the panel already had one, keep the
    handle and `ax.add_artist(it)` afterwards.
    """
    import matplotlib.patches as mpatches
    from matplotlib.transforms import (
        ScaledTranslation, blended_transform_factory,
    )

    t = resolve(theme)
    pal = palette if palette is not None else {
        g: t.categorical[i % len(t.categorical)] for i, g in enumerate(counts)
    }

    fig = ax.figure
    trans = (blended_transform_factory(ax.transData, ax.transAxes)
             + ScaledTranslation(0, -pad_pt / 72.0, fig.dpi_scale_trans))

    start = 0
    for g, n in counts.items():
        if n <= 0:
            continue
        ax.add_patch(mpatches.Rectangle(
            (start - 0.5, -height), n, height, transform=trans, clip_on=False,
            facecolor=pal[g], edgecolor="white", linewidth=0.6, zorder=5))
        start += n

    # Push the tick labels clear of the strip. Without this they are drawn at
    # the default pad and the strip lands on top of them — which looks like a
    # styling slip but is worse than one, because the band that says which
    # group a tick belongs to is covering the tick. The axes height is read
    # before the caller's tight_layout, so this over-reserves rather than
    # under-reserves; erring the other way would put the strip back on the ticks.
    ax_h_pt = ax.get_position().height * fig.get_figheight() * 72.0
    ax.tick_params(axis="x", pad=pad_pt + height * ax_h_pt)

    handles = [mpatches.Patch(facecolor=pal[g], edgecolor="none", label=g)
               for g, n in counts.items() if n > 0]
    handles += list(extra_handles or [])
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, legend_y), ncol=legend_cols,
                    frameon=False, fontsize=fontsize, handlelength=1.0,
                    handleheight=0.9, handletextpad=0.5, columnspacing=1.4,
                    borderaxespad=0)
    leg.set_in_layout(False)
    return leg
