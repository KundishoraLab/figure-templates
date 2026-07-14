"""Colormaps and categorical palettes.

Nothing here is domain-specific. Domain palettes (cell types, sample IDs,
mutation classes) belong in a theme module — see `figkit/themes/avm.py`.
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np

# ── Sequential colormaps ────────────────────────────────────────────────────
YLGNBU3 = LinearSegmentedColormap.from_list(
    "ylgnbu3", ["#ffffd9", "#edf8b1", "#225ea8"], N=256)
NAVY_YELLOW = LinearSegmentedColormap.from_list(
    "navy_yellow", ["#00204C", "#FFE945"], N=256)

# ── Diverging colormaps ─────────────────────────────────────────────────────
PUOR_DIV = LinearSegmentedColormap.from_list(
    "puor_div", ["#5e3c99", "#d9d9d9", "#e66101"], N=256)

# ── Categorical palettes ────────────────────────────────────────────────────
# Kelly's 22 maximally-contrasting colors (order preserved: adjacent entries
# are maximally distinguishable, so truncating to the first N is still good).
KELLY = [
    "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032", "#C2B280", "#848482",
    "#008856", "#E68FAC", "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C",
    "#DCD300", "#882D17", "#8DB600", "#654522", "#E25822", "#2B3D26",
]

# Wong's colorblind-safe qualitative palette. Prefer this over KELLY when the
# number of categories is <= 8 — it survives deuteranopia/protanopia, KELLY
# does not.
WONG = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
    "#DDCC77", "#CC6677", "#882255", "#AA4499",
]

NA_COLOR = "#cccccc"
BG_COLOR = "#dcdcdc"  # grey backdrop for two-layer scatters


def diverging_cmap(low: str, high: str, mid: str = "#f5f5f5", name: str = "div"):
    """Build a low -> mid -> high diverging colormap.

    Pair with `quantile_norm(..., symmetric=True)` so `mid` lands on zero.
    A diverging ramp whose midpoint is NOT the data's neutral value is a
    lie — it reads a meaningless value as "no change".
    """
    return LinearSegmentedColormap.from_list(name, [low, mid, high], N=256)


def sequential_cmap(high: str, low: str = "#ffffff", name: str = "seq"):
    """Build a low -> high sequential colormap for magnitude-only data."""
    return LinearSegmentedColormap.from_list(name, [low, high], N=256)


def gradient_cmap(colors, name: str = "gradient"):
    """Build a colormap that interpolates through an ordered list of colors.

    Use to derive a continuous ramp from an ordered categorical palette, so a
    continuous score and its discrete categories read as the same scale.
    """
    return LinearSegmentedColormap.from_list(name, list(colors), N=256)


def quantile_norm(values, q_low: float = 0.1, q_high: float = 0.9,
                  symmetric: bool = False) -> Normalize:
    """Robust vmin/vmax from quantiles, so outliers don't flatten the ramp.

    symmetric=True returns a TwoSlopeNorm centered on 0 — required for
    diverging colormaps, where an off-center midpoint misreads the data.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if not len(v):
        return Normalize(0, 1)
    lo, hi = float(np.quantile(v, q_low)), float(np.quantile(v, q_high))
    if symmetric:
        m = max(abs(lo), abs(hi))
        if m == 0:
            m = 1e-9
        return TwoSlopeNorm(vcenter=0, vmin=-m, vmax=m)
    if lo == hi:
        lo, hi = lo - 1e-9, hi + 1e-9
    return Normalize(vmin=lo, vmax=hi)


def categorical_palette(categories, colors=None, na_color: str = NA_COLOR) -> dict:
    """Map an iterable of category names -> hex colors, cycling `colors`."""
    colors = list(colors) if colors is not None else KELLY
    cats = list(dict.fromkeys(categories))  # de-dup, preserve order
    return {c: colors[i % len(colors)] for i, c in enumerate(cats)}
