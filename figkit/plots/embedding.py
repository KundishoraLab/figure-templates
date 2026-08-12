"""UMAP / embedding scatters, and spatial maps.

A UMAP axis has no units and no meaning — only the *relative* arrangement of
points does. So these helpers strip ticks and spines, and force equal aspect:
stretching a UMAP to fill a non-square panel silently rescales the distances
the plot exists to show.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..palettes import NA_COLOR, categorical_palette
from ..style import despine
from ..theme import Theme, resolve

UMAP_OBSM_KEYS = ("X_umap", "umap", "X_UMAP", "X_umap_integrated")


def get_embedding(adata, keys=UMAP_OBSM_KEYS) -> np.ndarray:
    """First matching obsm key -> (n, 2) coords. Raises if none found."""
    for k in keys:
        if k in adata.obsm:
            return np.asarray(adata.obsm[k])[:, :2]
    raise KeyError(f"no embedding in obsm; tried {list(keys)}, "
                   f"available: {list(adata.obsm)}")


def embedding_axes(ax, xlim=None, ylim=None, x_label: str = "UMAP 1",
                   y_label: str = "UMAP 2", equal: bool = True) -> None:
    """Standard embedding axes: no ticks, no spines, equal aspect.

    Pass the same xlim/ylim to every panel that shares an embedding — paired
    UMAPs at different limits look like different embeddings.
    """
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if equal:
        ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    despine(ax, "all")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)


def embedding_categorical(ax, xy, labels, palette=None, order=None,
                          point_size: float = 2.0, alpha: float = 0.8,
                          na_color: str = NA_COLOR, legend: bool = True,
                          legend_title: str = "", legend_loc: str = "side",
                          theme: Theme | None = None, axes: bool = True,
                          **axes_kw):
    """Embedding colored by a categorical variable.

    Draws one scatter per category rather than one big call, so the legend
    entries are real artists and z-order follows `order` — put rare, small
    categories last so common ones don't bury them.

    palette : {label: hex}. Defaults to cycling the theme's categorical list.
    order   : draw + legend order. Defaults to palette order, else sorted.
    """
    t = resolve(theme)
    xy = np.asarray(xy)
    labels = np.asarray(pd.Series(labels).astype(str))

    if palette is None:
        palette = categorical_palette(sorted(pd.unique(labels)), t.categorical)
    if order is None:
        order = [c for c in palette if c in set(labels)] or sorted(pd.unique(labels))

    seen = set()
    for cat in order:
        m = labels == str(cat)
        if not m.any():
            continue
        seen.add(str(cat))
        ax.scatter(xy[m, 0], xy[m, 1], c=[palette.get(cat, na_color)],
                   s=point_size, alpha=alpha, edgecolors="none",
                   rasterized=True, label=str(cat))
    other = ~np.isin(labels, list(seen))
    if other.any():
        ax.scatter(xy[other, 0], xy[other, 1], c=na_color, s=point_size,
                   alpha=alpha * 0.6, edgecolors="none", rasterized=True,
                   label="other")

    if axes:
        embedding_axes(ax, **axes_kw)
    if legend:
        items = [(str(c), palette.get(c, na_color)) for c in order if str(c) in seen]
        side_legend(ax, items, title=legend_title,
                    loc=legend_loc, ncol=1 if len(items) <= 12 else 2)
    return {"n_categories": len(seen), "n_points": len(xy)}


def embedding_continuous(ax, xy, values, cmap=None, norm=None,
                         point_size: float = 2.0, alpha: float = 0.9,
                         cbar_label: str = "", colorbar: bool = True,
                         sort_by_value: bool = True,
                         theme: Theme | None = None, axes: bool = True,
                         **axes_kw):
    """Embedding colored by a continuous value (score, expression, pseudotime).

    sort_by_value draws high values last so they land on top — otherwise a
    handful of high-scoring cells can be hidden under a mass of low ones and
    the signal looks absent.
    """
    t = resolve(theme)
    xy = np.asarray(xy)
    v = np.asarray(values, dtype=float)
    cmap = cmap if cmap is not None else t.sequential

    if sort_by_value:
        o = np.argsort(np.nan_to_num(v, nan=-np.inf))
        xy, v = xy[o], v[o]

    sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=cmap, norm=norm,
                    s=point_size, alpha=alpha, edgecolors="none",
                    rasterized=True)
    if axes:
        embedding_axes(ax, **axes_kw)
    if colorbar:
        import matplotlib.pyplot as plt
        cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=9)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=10)
    return sc


def two_layer_scatter(ax, bg_xy, fg_xy=None, fg_color=None, fg_cmap=None,
                      bg_color=None, bg_alpha: float = 0.25,
                      fg_alpha: float = 0.9, bg_size: float = 1.0,
                      fg_size: float = 4.0, colorbar: bool = True,
                      cbar_label: str = "", label: str | None = None,
                      theme: Theme | None = None, axes: bool = True,
                      **axes_kw):
    """Grey backdrop of all points + highlighted foreground subset.

    The backdrop is what makes a highlight interpretable: without it, the
    reader cannot tell whether a sparse foreground means "rare" or "we only
    measured a few". Always pass the *full* set as bg_xy, including the
    foreground points.
    """
    t = resolve(theme)
    bg_xy = np.asarray(bg_xy)
    ax.scatter(bg_xy[:, 0], bg_xy[:, 1], c=bg_color or t.background,
               s=bg_size, alpha=bg_alpha, edgecolors="none", rasterized=True)
    sc = None
    if fg_xy is not None and len(fg_xy):
        fg_xy = np.asarray(fg_xy)
        c = fg_color if fg_color is not None else t.up
        if isinstance(c, str):
            c = [c]
        sc = ax.scatter(fg_xy[:, 0], fg_xy[:, 1], c=c, cmap=fg_cmap,
                        s=fg_size, alpha=fg_alpha, edgecolors="none",
                        rasterized=True, label=label)
        if colorbar and fg_cmap is not None:
            import matplotlib.pyplot as plt
            cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=9)
            if cbar_label:
                cb.set_label(cbar_label, fontsize=10)
    if axes:
        embedding_axes(ax, **axes_kw)
    return sc


def highlight_mask(ax, xy, mask, fg_color=None, label: str | None = None,
                   fg_size: float = 2.0, bg_size: float = 0.25,
                   theme: Theme | None = None, axes: bool = True, **axes_kw):
    """Two-layer scatter driven by a boolean mask over `xy`."""
    xy = np.asarray(xy)
    mask = np.asarray(mask, dtype=bool)
    return two_layer_scatter(ax, xy, xy[mask], fg_color=fg_color,
                             fg_size=fg_size, bg_size=bg_size,
                             bg_alpha=0.30, label=label, colorbar=False,
                             theme=theme, axes=axes, **axes_kw)


def side_legend(ax, items, title: str = "", loc: str = "side",
                markersize: float = 7, fontsize: float = 9, ncol: int = 1):
    """Legend built from (label, color) pairs.

    loc='side' mounts it outside the axes, which is almost always right for
    an embedding — an inset legend covers data points, and on a UMAP you
    cannot know in advance which corner is empty.
    """
    import matplotlib.pyplot as plt
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markersize=markersize, markerfacecolor=c,
                          markeredgecolor="none", label=str(l))
               for l, c in items]
    kw = dict(handles=handles, fontsize=fontsize, frameon=False, title=title,
              title_fontsize=fontsize + 1, ncol=ncol)
    if loc == "side":
        return ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), **kw)
    return ax.legend(loc=loc, **kw)


def add_scale_bar(ax, length: float = 1.0, label: str | None = None,
                  position: str = "lower right", pad_frac: float = 0.04,
                  color: str = "#111", linewidth: float = 2.0,
                  fontsize: float = 8, unit: str = "mm"):
    """Scale bar for a spatial plot whose axes are in real units.

    Only meaningful on real coordinates — never put one on a UMAP, whose
    axes have no unit. `length` is in the axes' own units; `unit` only
    labels it.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xspan, yspan = abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])
    pad_x, pad_y = pad_frac * xspan, pad_frac * yspan
    if "right" in position:
        x1 = max(xlim) - pad_x
        x0 = x1 - length
    else:
        x0 = min(xlim) + pad_x
        x1 = x0 + length
    y0 = (min(ylim) + pad_y) if "lower" in position else (max(ylim) - pad_y)
    ax.plot([x0, x1], [y0, y0], color=color, lw=linewidth,
            solid_capstyle="butt", zorder=10)
    if label is None:
        label = (f"{length * 1000:g} µm" if unit == "mm" and length < 1
                 else f"{length:g} {unit}")
    ax.text(0.5 * (x0 + x1), y0 + 0.35 * pad_y, label, ha="center",
            va="bottom", fontsize=fontsize, color=color, zorder=10)


def embedding_outline(ax, xy, labels, palette=None, order=None,
                      coverage: float = 0.90, smooth: float = 1.0,
                      min_area_frac: float = 0.0, grid: int = 256,
                      linewidth: float = 1.1, legend: bool = False,
                      legend_title: str = "", legend_y: float = -0.12,
                      theme: Theme | None = None):
    """Where each group sits, as a contour instead of a colour.

    Draws, per group, the smallest region containing `coverage` of that group's
    points. Use it when the scatter's colour is already spent on something else
    — a score, a pseudotime — and re-colouring by group would cost the panel
    the encoding it exists for. It also answers a question a coloured scatter
    answers badly at scale: with tens of thousands of overplotted points,
    whichever group is drawn last looks like it occupies the whole embedding.

    **The bandwidth is shared across groups**, derived once from the pooled
    points. This is the argument for the function existing rather than each
    caller contouring its own KDE. Per-group bandwidths scale with group size,
    so a smaller group is smoothed with a fatter kernel and its territory comes
    out looking larger — an artefact that runs in the same direction as the
    biology in every depleted-population comparison, and is invisible in the
    output.

    coverage      : fraction of the group's points inside the contour. This is
      a mass threshold on the density, not a quantile of any one axis, so the
      region can be several pieces — and is left as several, because merging
      them would draw a boundary through empty space.
    smooth        : multiplies the shared bandwidth. Above 1 for a summary; at
      the automatic bandwidth a contour tracks every local fluctuation and
      throws off specks in sparse regions, which read as satellite populations.
    min_area_frac : drop contour pieces smaller than this fraction of the
      group's total enclosed area. The specks again — a piece holding a handful
      of points is noise in the density, not a subpopulation. Reports what it
      dropped, because a silently simplified contour is a claim about where a
      group is not.
    grid          : mesh resolution. Below about 150 the contour polygonises.

    Returns {group: fraction of the panel its contour encloses}.
    """
    from matplotlib.patheffects import withStroke
    from scipy.ndimage import gaussian_filter, label as cc_label

    t = resolve(theme)
    xy = np.asarray(xy, dtype=float)
    labels = np.asarray(pd.Series(labels).astype(str))
    if xy.shape[0] != labels.size:
        raise ValueError(f"embedding_outline: {xy.shape[0]} points but "
                         f"{labels.size} labels")
    if not 0.0 < coverage < 1.0:
        raise ValueError("embedding_outline: coverage must be in (0, 1), "
                         f"got {coverage}")

    if order is None:
        order = list(pd.unique(labels))
    else:
        order = [str(o) for o in order]
    if palette is None:
        palette = categorical_palette(order, t.categorical)

    # One grid and one bandwidth for every group — see the docstring.
    (lo_x, lo_y), (hi_x, hi_y) = xy.min(0), xy.max(0)
    mx, my = 0.06 * (hi_x - lo_x), 0.06 * (hi_y - lo_y)
    lo_x, hi_x, lo_y, hi_y = lo_x - mx, hi_x + mx, lo_y - my, hi_y + my
    xe = np.linspace(lo_x, hi_x, grid + 1)
    ye = np.linspace(lo_y, hi_y, grid + 1)
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    # Scott's rule on the pooled points, expressed in grid cells.
    scott = xy.shape[0] ** (-1.0 / 6.0)
    sx = smooth * scott * xy[:, 0].std() / ((hi_x - lo_x) / grid)
    sy = smooth * scott * xy[:, 1].std() / ((hi_y - lo_y) / grid)

    areas = {}
    for g in order:
        pts = xy[labels == g]
        if pts.shape[0] < 3:
            print(f"  [embedding_outline] {g}: {pts.shape[0]} points — skipped")
            continue
        counts, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[xe, ye])
        dens = gaussian_filter(counts, sigma=(sx, sy), mode="constant")
        total = dens.sum()
        if total <= 0:
            continue
        dens /= total
        # The level enclosing `coverage` of the mass: sort cells by density and
        # walk down until the cumulative mass reaches it.
        flat = np.sort(dens.ravel())[::-1]
        cum = np.cumsum(flat)
        level = float(flat[min(int(np.searchsorted(cum, coverage)), flat.size - 1)])

        keep = dens >= level
        if min_area_frac > 0:
            comps, n_comp = cc_label(keep)
            if n_comp > 1:
                sizes = np.bincount(comps.ravel())[1:]
                drop = [i + 1 for i, s in enumerate(sizes)
                        if s < min_area_frac * sizes.sum()]
                if drop:
                    print(f"  [embedding_outline] {g}: dropped {len(drop)} of "
                          f"{n_comp} pieces below {min_area_frac:.0%} of its area")
                    keep &= ~np.isin(comps, drop)
        # Contour the density, not the mask: a binary mask contours as a
        # staircase along the cell edges.
        cs = ax.contour(xc, yc, np.where(keep, dens, 0.0).T, levels=[level],
                        colors=[palette.get(g, t.na)], linewidths=linewidth,
                        zorder=6)
        # White halo. The contour runs over the scatter it describes, and a
        # bare 1.1 pt line crossing a dense region is the same width as the
        # gaps between points — it disappears exactly where the group is
        # densest, which is where the reader is looking.
        cs.set(path_effects=[withStroke(linewidth=3 * linewidth,
                                        foreground="white")])
        areas[g] = float(keep.sum()) / keep.size

    if legend:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], color=palette.get(g, t.na),
                                  linewidth=linewidth, label=g)
                           for g in order if g in areas],
                  loc="upper center", bbox_to_anchor=(0.5, legend_y),
                  ncol=max(len(areas), 1), frameon=False, fontsize=8,
                  title=legend_title, title_fontsize=8,
                  handlelength=1.6, handletextpad=0.5, columnspacing=1.6,
                  borderaxespad=0)
    return areas
