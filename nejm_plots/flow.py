"""Flow cytometry dot plots: density-shaded events on biexponential-like axes, gates."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter

GREY_TO_BLACK = LinearSegmentedColormap.from_list("g2k", ["#B8B8B8", "#5A5A5A", "#000000"])


def _sym(v, t):
    """symlog-like transform used for density estimation (matches the axis scale)."""
    v = np.asarray(v, float)
    return np.sign(v) * np.log10(1 + np.abs(v) / t)


def flow_axes(ax, *, linthresh=1000, linscale=0.55, lim=(-400, 2e5), decades=(3, 4, 5),
              axis="both", arrows=True):
    """Biexponential-style axis: linear near 0, log above. Ticks 0, 10^3, 10^4, 10^5
    with log minor ticks, and arrowheads at the spine ends."""
    major = [0] + [10.0 ** d for d in decades]
    labels = ["0"] + [f"$10^{d}$" for d in decades]
    minor = [m * 10.0 ** d for d in range(decades[0], decades[-1]) for m in range(2, 10)]
    for a in (["x", "y"] if axis == "both" else [axis]):
        getattr(ax, f"set_{a}scale")("symlog", linthresh=linthresh, linscale=linscale)
        getattr(ax, f"set_{a}lim")(*lim)
        ax_ = getattr(ax, f"{a}axis")
        ax_.set_major_locator(FixedLocator(major))
        ax_.set_major_formatter(FixedFormatter(labels))
        ax_.set_minor_locator(FixedLocator(minor))
    ax.tick_params(which="minor", length=2)
    if arrows:
        ax.plot(1, 0, ">k", ms=4, transform=ax.transAxes, clip_on=False)
        ax.plot(0, 1, "^k", ms=4, transform=ax.transAxes, clip_on=False)


def plot_flow(ax, x, y, *, linthresh=1000, bins=120, s=0.6, cmap=GREY_TO_BLACK,
              rasterize=True, vmax_pct=97, **axis_kw):
    """Events shaded by local density (dense core black, sparse events grey)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    tx, ty = _sym(x, linthresh), _sym(y, linthresh)
    H, xe, ye = np.histogram2d(tx, ty, bins=bins)
    ix = np.clip(np.searchsorted(xe, tx) - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(ye, ty) - 1, 0, bins - 1)
    d = np.log1p(H[ix, iy])
    o = np.argsort(d)
    ax.scatter(x[o], y[o], c=d[o], s=s, cmap=cmap, lw=0, vmin=0,
               vmax=np.percentile(d, vmax_pct), rasterized=rasterize, zorder=1)
    flow_axes(ax, linthresh=linthresh, **axis_kw)


def add_gate(ax, x0, x1, y0, y1, label=None, *, label_xy=None, lw=0.9, color="black",
             fontsize=None, events=None):
    """Rectangle gate. label may be a string or 'auto' to print the % of `events`
    (tuple of x, y arrays) that fall inside."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw, zorder=3))
    if label == "auto" and events is not None:
        ex, ey = (np.asarray(a) for a in events)
        inside = (ex >= x0) & (ex <= x1) & (ey >= y0) & (ey <= y1)
        label = f"{100 * inside.mean():.1f}%"
    if label:
        lx, ly = label_xy if label_xy else (np.sqrt(max(x0, 1) * x1), y0 / 6)
        ax.text(lx, ly, label, ha="center", va="top", fontsize=fontsize)
    return label
