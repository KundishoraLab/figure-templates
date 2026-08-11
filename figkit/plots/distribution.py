"""Distributions across categories — the QC panel that every single-cell paper needs.

`violin` is the one you want for "nUMI / nGenes / percent-mito across N
libraries". `box` is the same grammar when the sample is small enough that a
density is a lie.

Design decisions baked in, each one a failure mode that produced a wrong panel:

- **The KDE is clipped to the observed range.** A violin of UMI counts routinely
  renders a tail below zero, because a Gaussian kernel does not know the
  quantity is non-negative. Readers correctly read that as "some cells have
  negative counts" and stop trusting the figure.
- **A violin without its n is a shape.** The group label carries `n = …` unless
  you turn it off. Three cells and three thousand cells make the same silhouette.
- **Median and IQR are drawn on top of the density**, not offered as an
  alternative to it. The density shows structure — bimodality, a dying-cell
  shoulder — and the median gives the reader something to actually compare.
- **`thresholds` is an argument, not something you draw afterwards.** A QC violin
  exists to justify a cutoff. Passing the cutoff in means the line and the filter
  cannot drift apart; drawing it separately means they will.
- **`log` is explicit and defaults to False.** An earlier version inferred it
  from the spread and got it wrong in both directions on the first real panel:
  UMI counts (which need a log axis) came out linear and squashed against the
  floor, while a mitochondrial percentage (which must not be logged) came out
  spanning 10^-3 to 10^3. The caller knows what the quantity is; a heuristic
  does not, and it fails silently and plausibly.
- **The y-axis hugs the data.** matplotlib autoscales `fill_betweenx` with
  generous margins, which leaves a violin sitting in the bottom third of its
  panel with most of the vertical space empty. Across a strip of QC panels that
  wastes the axis every reader is trying to compare along. Limits are set from
  the plotted range with a small margin instead.
- **Points are off by default.** At 20,000 cells a strip plot is a filled
  rectangle. When on, it subsamples and says so, because a subsampled overlay
  that silently claims to be all the data is worse than none.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..palettes import categorical_palette
from ..style import despine
from ..theme import Theme, resolve

__all__ = ["violin", "box", "ranked_strip"]

# Below this many observations a kernel density is decoration, not an estimate.
MIN_FOR_DENSITY = 20


def _density(values: np.ndarray, n: int = 256):
    """Gaussian KDE clipped to the observed range. Returns (grid, density)."""
    from scipy.stats import gaussian_kde

    v = values[np.isfinite(values)]
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return None, None
    grid = np.linspace(lo, hi, n)          # clipped: never extends past the data
    try:
        d = gaussian_kde(v)(grid)
    except Exception:
        return None, None
    peak = d.max()
    return grid, d / peak if peak > 0 else d


def violin(df, ax, value_col: str, group_col: str, order=None, palette=None,
           log: bool = False, thresholds=None, show_n: bool = True,
           points: bool = False, max_points: int = 500, width: float = 0.8,
           y_label: str = "", theme: Theme | None = None, seed: int = 0):
    """Distribution of `value_col` per level of `group_col`.

    order      : left-to-right group order. Groups outside it are dropped loudly.
    thresholds : scalar, or {group: value}, drawn as the cutoff being justified.
    log        : log10 the y axis. Explicit on purpose — see module docstring.
    points     : overlay a subsample of the raw values (see module docstring).

    Returns a per-group summary dict — median, IQR, n — so the caller can put
    the same numbers in a table without recomputing them differently.
    """
    t = resolve(theme)
    d = pd.DataFrame(df)[[value_col, group_col]].copy()
    d[group_col] = d[group_col].astype(str)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    dropped_na = int(d[value_col].isna().sum())
    d = d.dropna(subset=[value_col])
    if dropped_na:
        print(f"  [violin] dropped {dropped_na} rows with no {value_col!r}")

    if order is None:
        order = list(pd.unique(d[group_col]))
    else:
        order = [str(o) for o in order]
        keep = d[group_col].isin(order)
        if (~keep).any():
            print(f"  [violin] dropped {int((~keep).sum())} rows not in `order`: "
                  f"{sorted(d.loc[~keep, group_col].unique())[:6]}")
        d = d[keep]
    if d.empty:
        raise ValueError("violin: no rows left to plot")

    if palette is None:
        palette = categorical_palette(order, t.categorical)
    rng = np.random.default_rng(seed)
    summary = {}
    lo_all, hi_all = np.inf, -np.inf

    for i, grp in enumerate(order):
        v = d.loc[d[group_col] == grp, value_col].to_numpy()
        if v.size == 0:
            continue
        color = palette.get(grp, t.na)
        plot_v = np.log10(v[v > 0]) if log else v
        if plot_v.size:
            lo_all = min(lo_all, float(plot_v.min()))
            hi_all = max(hi_all, float(plot_v.max()))

        if plot_v.size >= MIN_FOR_DENSITY:
            grid, dens = _density(plot_v)
            if grid is not None:
                half = dens * (width / 2)
                ax.fill_betweenx(grid, i - half, i + half, facecolor=color,
                                 edgecolor="white", linewidth=0.3, zorder=2)

        q1, med, q3 = np.percentile(plot_v, [25, 50, 75])
        ax.vlines(i, q1, q3, color="black", linewidth=1.1, zorder=3)
        ax.plot(i, med, marker="o", markersize=2.6, color="white",
                markeredgecolor="black", markeredgewidth=0.5, zorder=4)

        if points:
            take = plot_v if plot_v.size <= max_points else rng.choice(plot_v, max_points, replace=False)
            if plot_v.size > max_points:
                print(f"  [violin] {grp}: showing {max_points} of {plot_v.size} points")
            ax.scatter(i + rng.uniform(-width / 5, width / 5, take.size), take,
                       s=1.0, color="black", alpha=0.25, linewidths=0, zorder=1)

        raw_q1, raw_med, raw_q3 = np.percentile(v, [25, 50, 75])
        summary[grp] = {"n": int(v.size), "median": float(raw_med),
                        "q1": float(raw_q1), "q3": float(raw_q3)}

    # Thresholds: the reason the panel exists.
    if thresholds is not None:
        if np.isscalar(thresholds):
            y = np.log10(thresholds) if log and thresholds > 0 else thresholds
            ax.axhline(y, color=t.up, linestyle="--", linewidth=0.7, zorder=5)
        else:
            for i, grp in enumerate(order):
                if grp in thresholds:
                    val = thresholds[grp]
                    y = np.log10(val) if log and val > 0 else val
                    ax.plot([i - width / 2, i + width / 2], [y, y],
                            color=t.up, linestyle="--", linewidth=0.7, zorder=5)

    # Hug the data — see module docstring.
    if np.isfinite(lo_all) and hi_all > lo_all:
        pad = 0.04 * (hi_all - lo_all)
        ax.set_ylim(lo_all - pad, hi_all + pad)

    labels = [f"{g}\nn = {summary[g]['n']:,}" if show_n and g in summary else g
              for g in order]
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(y_label or value_col)

    if log:
        # Tick in real units on a log-transformed axis, so the reader never has
        # to exponentiate a label in their head.
        lo, hi = ax.get_ylim()
        decades = np.arange(np.ceil(lo), np.floor(hi) + 1)
        if decades.size:
            ax.set_yticks(decades)
            ax.set_yticklabels([f"$10^{{{int(e)}}}$" for e in decades], fontsize=9)
        ax.set_ylim(lo, hi)   # set_yticks can re-expand; put the limits back

    despine(ax)
    return {"groups": summary, "log": bool(log)}


def box(df, ax, value_col: str, group_col: str, order=None, palette=None,
        log: bool = False, thresholds=None, show_n: bool = True,
        width: float = 0.6, y_label: str = "", theme: Theme | None = None):
    """Box-and-whisker version of `violin`, for when n is too small for a density.

    Whiskers are 1.5 x IQR and outliers are drawn. Hiding them would make a
    dying-cell tail — the thing a QC panel is looking for — invisible.
    """
    t = resolve(theme)
    d = pd.DataFrame(df)[[value_col, group_col]].copy()
    d[group_col] = d[group_col].astype(str)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])

    if order is None:
        order = list(pd.unique(d[group_col]))
    else:
        order = [str(o) for o in order]
        d = d[d[group_col].isin(order)]
    if palette is None:
        palette = categorical_palette(order, t.categorical)
    data, summary = [], {}
    for grp in order:
        v = d.loc[d[group_col] == grp, value_col].to_numpy()
        data.append(np.log10(v[v > 0]) if log else v)
        if v.size:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            summary[grp] = {"n": int(v.size), "median": float(med),
                            "q1": float(q1), "q3": float(q3)}

    bp = ax.boxplot(data, positions=np.arange(len(order)), widths=width,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", linewidth=1.0),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6),
                    flierprops=dict(marker="o", markersize=1.2, alpha=0.3,
                                    markerfacecolor="black", markeredgewidth=0))
    for patch, grp in zip(bp["boxes"], order):
        patch.set_facecolor(palette.get(grp, t.na))
        patch.set_edgecolor("white")
        patch.set_linewidth(0.3)

    if thresholds is not None and np.isscalar(thresholds):
        y = np.log10(thresholds) if log and thresholds > 0 else thresholds
        ax.axhline(y, color=t.up, linestyle="--", linewidth=0.7)

    labels = [f"{g}\nn = {summary[g]['n']:,}" if show_n and g in summary else g
              for g in order]
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(y_label or value_col)
    if log:
        lo, hi = ax.get_ylim()
        decades = np.arange(np.floor(lo), np.ceil(hi) + 1)
        ax.set_yticks(decades)
        ax.set_yticklabels([f"$10^{{{int(e)}}}$" for e in decades], fontsize=9)
    despine(ax)
    return {"groups": summary, "log": bool(log)}


def ranked_strip(ax, values, reference=None, reference_label: str = "observed",
                 order: str = "ascending", point_size: float = 14,
                 color=None, reference_color=None, log: bool = False,
                 y_label: str = "", x_label: str = "rank",
                 annotate: bool = True, fontsize: float = 8,
                 theme: Theme | None = None):
    """A calibration bank: every value in a null distribution, ranked, with the
    observed one marked against them.

    This is the panel for "my test returned p = 0.22 — is that a null about the
    biology, or a design that could never have returned anything?". A histogram
    of the null answers it badly at the sizes this is for: 100 permutations or
    100 random gene lists give a histogram whose shape is mostly binning, and
    the tail — the only part that matters — is where the counts are smallest.
    Ranked points show every draw, so the reader can see the floor, the density
    near it, and where the observation sits, without a smoothing choice
    intervening.

    Returns the count and fraction of null values at least as extreme as
    `reference`, which is the empirical p-value's numerator. `annotate` prints
    it on the panel, because a calibration panel whose whole content is that
    fraction should not make the reader compute it off the y-axis.

    `order="ascending"` puts the most extreme draws at the left, which is where
    a reader looks first and where the reference usually needs to be legible.
    """
    t = resolve(theme)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        raise ValueError("ranked_strip got no finite values")
    v = np.sort(v) if order == "ascending" else np.sort(v)[::-1]
    x = np.arange(1, len(v) + 1)
    y = np.log10(v) if log else v

    ax.scatter(x, y, s=point_size, color=color or t.na, edgecolors="none",
               zorder=2, clip_on=False)

    out = {"n_null": int(len(v))}
    if reference is not None:
        r = np.log10(reference) if log else reference
        rc = reference_color or t.up
        ax.axhline(r, color=rc, linewidth=1.0, zorder=3)
        n_le = int((v <= reference).sum())
        # (1 + k) / (1 + n): the observation is one draw from the same
        # procedure, so excluding it from its own reference set biases the
        # p-value down and can return an impossible zero.
        emp = (1 + n_le) / (1 + len(v))
        out.update(reference=float(reference), n_at_least_as_extreme=n_le,
                   empirical_p=emp)
        if annotate:
            ax.annotate(
                f"{reference_label} = {reference:.4g}\n"
                f"{n_le} of {len(v)} null ≤ it · empirical p = {emp:.4f}",
                xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=fontsize, color=rc)

    ax.set_xlim(0, len(v) + 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log:
        lo, hi = ax.get_ylim()
        decades = np.arange(np.floor(lo), np.ceil(hi) + 1)
        ax.set_yticks(decades)
        ax.set_yticklabels([f"$10^{{{int(e)}}}$" for e in decades], fontsize=9)
    despine(ax)
    return out
