"""Global NEJM-like style, palette, figure layout helpers, and broken axes."""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.transforms import blended_transform_factory

# NEJM figures are set in Myriad Pro. The stack falls back to similar humanist sans faces.
FONT_STACK = ["Myriad Pro", "Source Sans 3", "Source Sans Pro", "Carlito", "Arial", "DejaVu Sans"]

COLORS = {
    # two-arm KM palette (orange treatment, blue control)
    "orange": "#EE9F2F",
    "blue": "#5D84C6",
    # point-range palette (dark and light steel blue)
    "dark_blue": "#2D6C9C",
    "light_blue": "#6FA8D3",
    # trajectory palette
    "red": "#D9654B",
    "red_band": "#F1B39F",
    "teal": "#2F6A89",
    "teal_band": "#9EBFD3",
    # misc
    "frame": "#8C8C8C",
    "shade": "#EFEFEF",
    "bracket": "#8A8A8A",
}

# Modified Rankin Scale 0..6, light to dark slate blue
MRS_COLORS = ["#E3EBEF", "#C4D3DC", "#B5C8D4", "#A4BAC8", "#7C9BAF", "#446B83", "#234A5E"]


def _first_available(stack):
    names = {f.name for f in font_manager.fontManager.ttflist}
    for s in stack:
        if s in names:
            return s
    return "DejaVu Sans"


def use_nejm_style(base_size: float = 10):
    """Apply rcParams. Call once before building a figure (sizes scale from base_size)."""
    font = _first_available(FONT_STACK)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font] + FONT_STACK,
        "svg.fonttype": "none",       # text stays text: editable vectors (NEJM), measurable type
        "pdf.fonttype": 42,           # TrueType embedded, editable in Illustrator
        "font.size": base_size,
        "mathtext.fontset": "custom",
        "mathtext.rm": font,
        "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold",
        "axes.titlesize": base_size * 1.1,
        "axes.labelsize": base_size,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.facecolor": "none",
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.frameon": False,
        "legend.fontsize": base_size,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "lines.solid_capstyle": "butt",
    })
    return font


# --------------------------------------------------------------------------------------
# Layout in inches, measured from the TOP-LEFT of the figure (easier to reason about)
# --------------------------------------------------------------------------------------
class InchLayout:
    """Place axes, frames and text in inches from the top-left corner of a figure."""

    def __init__(self, fig):
        self.fig = fig
        self.W, self.H = fig.get_size_inches()

    def _frac(self, left, top, width, height):
        return [left / self.W, 1 - (top + height) / self.H, width / self.W, height / self.H]

    def axes(self, left, top, width, height, **kw):
        return self.fig.add_axes(self._frac(left, top, width, height), **kw)

    def frame(self, left, top, width, height, color=COLORS["frame"], lw=1.4):
        r = Rectangle(self._frac(left, top, width, height)[:2], width / self.W, height / self.H,
                      transform=self.fig.transFigure, fill=False, ec=color, lw=lw, zorder=10)
        self.fig.add_artist(r)
        return r

    def hline(self, left, right, y, color=COLORS["frame"], lw=1.4):
        ln = Line2D([left / self.W, right / self.W], [1 - y / self.H] * 2,
                    transform=self.fig.transFigure, color=color, lw=lw, zorder=10)
        self.fig.add_artist(ln)

    def vline(self, x, top, bottom, color=COLORS["frame"], lw=1.4):
        ln = Line2D([x / self.W] * 2, [1 - top / self.H, 1 - bottom / self.H],
                    transform=self.fig.transFigure, color=color, lw=lw, zorder=10)
        self.fig.add_artist(ln)

    def text(self, x, y, s, **kw):
        kw.setdefault("va", "center")
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def panel_heading(self, x, y, letter, title="", size=None, gap=0.28):
        """NEJM panel heading: bold letter followed by a semibold title."""
        size = size or mpl.rcParams["font.size"] * 1.25
        va = "top" if "\n" in title else "center"
        self.text(x, y, letter, fontsize=size * 1.05, fontweight="bold", ha="left", va=va)
        if title:
            self.text(x + gap, y, title, fontsize=size, fontweight="semibold", ha="left",
                      linespacing=1.15, va="top" if "\n" in title else "center")


# --------------------------------------------------------------------------------------
# Broken y axis ( 0 // 80 ... 100 ) implemented as a piecewise-linear scale
# --------------------------------------------------------------------------------------
_SLASH = Path([(-1, -0.75), (1, 0.05), (-1, -0.05), (1, 0.75)],
              [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])
_GAP = Path([(0, -1), (0, 1)], [Path.MOVETO, Path.LINETO])


def add_y_break(ax, lo, hi, gap_frac=0.10, mark=True, bg="white", mark_size=10,
                upper_scale=1.0):
    """Collapse the interval (lo, hi) on the y axis and draw a '//' mark on the spine.

    Set the final ylim BEFORE calling this. Values in (lo, hi) are squeezed into the gap.
    upper_scale < 1 compresses the segment above the break (e.g. a few outliers up to
    1000 above a 0-300 bulk).
    Example: ax.set_ylim(0, 100); add_y_break(ax, 0, 80) -> 0 // 80 85 90 95 100
             ax.set_ylim(0, 100); add_y_break(ax, 40, 100) -> 0 10 20 30 40 // 100
    """
    y0, y1 = ax.get_ylim()
    k = upper_scale
    span = (y1 - hi) * k + (lo - y0)
    gap = gap_frac * span / (1 - gap_frac)

    def fwd(y):
        y = np.asarray(y, float)
        return np.where(y <= lo, y, np.where(y >= hi, lo + gap + (y - hi) * k,
                                             lo + (y - lo) / (hi - lo) * gap))

    def inv(z):
        z = np.asarray(z, float)
        return np.where(z <= lo, z, np.where(z >= lo + gap, hi + (z - lo - gap) / k,
                                             lo + (z - lo) / gap * (hi - lo)))

    ax.set_yscale("function", functions=(fwd, inv))
    ax.set_ylim(y0, y1)
    if mark:
        mid = lo + 0.5 * (hi - lo)
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        ax.plot([0], [mid], transform=tr, marker=_GAP, ms=mark_size * 0.55, mew=4, color=bg,
                clip_on=False, zorder=6)
        ax.plot([0], [mid], transform=tr, marker=_SLASH, ms=mark_size, mew=0.9, color="black",
                clip_on=False, zorder=7)
    return fwd, inv


def auto_y_break(ax, values, *, nbins=5, pad_top=0.03, low_pct=0.5, high_pct=99.5,
                 gap_frac=0.14, mark_size=8, force_zero=True):
    """Pick 'nice' y ticks covering the data and, if the data sit far above 0, show a
    0 // break so the axis still starts at zero (NEJM convention). Returns the ticks."""
    from matplotlib.ticker import MaxNLocator
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [low_pct, high_pct])
    ticks = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    ticks = ticks[(ticks <= hi + (ticks[1] - ticks[0]))]
    step = ticks[1] - ticks[0]
    first = ticks[ticks <= lo].max() if (ticks <= lo).any() else ticks[0] - step
    ticks = ticks[ticks >= first]
    top = max(ticks[-1], hi * (1 + pad_top))
    if force_zero and first > 0 and first > 1.5 * step:
        ax.set_ylim(0, top)
        add_y_break(ax, 0, first, gap_frac=gap_frac, mark_size=mark_size)
        ticks = np.r_[0, ticks]
    else:
        bottom = min(0, first) if force_zero else first
        if force_zero and first > 0:      # small gap to zero: just extend the ticks down
            ticks = np.r_[np.arange(0, first, step), ticks]
        ax.set_ylim(bottom, top)
    ax.set_yticks(ticks)
    return ticks


def clean_axes(ax, left=True, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if not bottom:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)


# --------------------------------------------------------------------------------------
# Tick formatters
# --------------------------------------------------------------------------------------
def log10_axis(ax, lo_exp, hi_exp, axis="y", minor=False):
    """Log axis with 10^n labels from 10^lo_exp to 10^hi_exp."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, LogLocator
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"set_{axis}lim")(10.0 ** lo_exp, 10.0 ** hi_exp)
    a = getattr(ax, f"{axis}axis")
    exps = list(range(lo_exp, hi_exp + 1))
    a.set_major_locator(FixedLocator([10.0 ** e for e in exps]))
    a.set_major_formatter(FixedFormatter([f"$10^{{{e}}}$" for e in exps]))
    a.set_minor_locator(LogLocator(subs=np.arange(2, 10)) if minor else NullLocator())


def sci_times_labels(values, exp=None, decimals=1, zero="0"):
    """Labels like 2.5x10^4 (NEJM style) for linear ticks."""
    out = []
    for v in values:
        if v == 0:
            out.append(zero)
            continue
        e = exp if exp is not None else int(np.floor(np.log10(abs(v))))
        out.append(f"{v / 10 ** e:.{decimals}f}$\\times$10$^{{{e}}}$")
    return out
