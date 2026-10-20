"""Theme + composite assembler.

A Theme fixes every presentation number once (font sizes, line widths, tick lengths,
default margins, gutters, panel-letter style). A Composite lays panels on a 12-column
grid, row by row. Within a row every axes box has the same height and the same top and
bottom edges, so panels line up; panels of equal span get equal axes widths.

    theme = Theme.paper()                       # or Theme.nejm(), Theme.nature()
    fig = Composite(width="double", theme=theme)
    fig.row([(km_panel(...), 8), (roc_panel(...), 4)], height=1.7)
    fig.row([groups_panel(...), groups_panel(...), paired_panel(...)], height=1.6)
    fig.save("out/figure2", formats=("png", "svg"))

Every panel is also a figure on its own:

    km_panel(...).save("out/panels/km", width=3.5, height=1.7, theme=theme)

Writing your own panel: a Panel wraps a draw(region) function. region.axes() gives the
main axes in its standard box; region.grid(r, c) splits that box; region.inset(...)
places axes by inches. Ask for extra room with margins={"left": 0.9, "bottom": 0.6}.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import use_nejm_style

MM = 1 / 25.4
WIDTHS = {"single": 89 * MM, "onehalf": 120 * MM, "double": 183 * MM, "nejm": 7.1}

_CURRENT = None


# --------------------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------------------
@dataclass
class Theme:
    font_size: float = 7.0        # tick labels
    label_size: float = 7.5       # axis labels
    small_size: float = 6.5       # legends, at-risk tables, annotations
    title_size: float = 8.0       # panel titles (when used)
    panel_label_size: float = 10.0
    panel_label_case: str = "lower"     # "lower" a b c  |  "upper" A B C
    label_weight: str = "bold"          # axis label weight
    axes_lw: float = 0.7
    line_lw: float = 1.3
    tick_len: float = 3.0
    marker_s: float = 9.0               # scatter size (pt^2)
    # default margins around each axes box, inches
    left: float = 0.50
    right: float = 0.10
    top: float = 0.24             # holds the panel letter
    bottom: float = 0.36
    gutter: float = 0.10          # space between cells
    row_gap: float = 0.06
    outer: float = 0.04           # figure padding
    title_height: float = 0.14
    frames: bool = False          # NEJM grey rules between panels
    frame_color: str = "#8C8C8C"

    @classmethod
    def paper(cls, **kw):
        return cls(**kw)

    @classmethod
    def nature(cls, **kw):
        """Nature: 5-7 pt type, lowercase bold letters."""
        base = dict(font_size=6.5, label_size=7, small_size=6, title_size=7,
                    panel_label_size=9, left=0.46, bottom=0.33)
        base.update(kw)
        return cls(**base)

    @classmethod
    def nejm(cls, **kw):
        """NEJM: uppercase letters followed by titles, grey frames."""
        base = dict(font_size=8, label_size=8, small_size=7.5, title_size=8.6,
                    panel_label_size=9.2, panel_label_case="upper", frames=True,
                    left=0.62, top=0.30, bottom=0.42, gutter=0.14, outer=0.06,
                    axes_lw=0.9, line_lw=1.4)
        base.update(kw)
        return cls(**base)

    def apply(self):
        global _CURRENT
        use_nejm_style(self.font_size)
        mpl.rcParams.update({
            "axes.labelsize": self.label_size,
            "axes.labelweight": self.label_weight,
            "legend.fontsize": self.small_size,
            "legend.title_fontsize": self.small_size,
            "axes.linewidth": self.axes_lw,
            "xtick.major.width": self.axes_lw,
            "ytick.major.width": self.axes_lw,
            "xtick.major.size": self.tick_len,
            "ytick.major.size": self.tick_len,
            "xtick.minor.size": self.tick_len * 0.6,
            "ytick.minor.size": self.tick_len * 0.6,
            "lines.linewidth": self.line_lw,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "axes.labelpad": 3.0,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.2,
            "legend.labelspacing": 0.3,
        })
        _CURRENT = self
        return self

    def letter(self, i):
        s = string.ascii_lowercase[i]
        return s.upper() if self.panel_label_case == "upper" else s

    @staticmethod
    def pt(points):
        return points / 72.0


def get_theme() -> Theme:
    return _CURRENT or Theme().apply()


# --------------------------------------------------------------------------------------
# Region handed to each panel's draw function
# --------------------------------------------------------------------------------------
class Region:
    """Where a panel draws. All boxes are (x, y, w, h) in inches from the top-left."""

    def __init__(self, fig, theme, cell, box):
        self.fig, self.theme, self.cell, self.box = fig, theme, cell, box
        self.W, self.H = fig.get_size_inches()
        self._ax = None
        self.data = {}

    def _frac(self, x, y, w, h):
        return [x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H]

    def axes(self, **kw):
        """The panel's main axes (created once)."""
        if self._ax is None:
            self._ax = self.fig.add_axes(self._frac(*self.box), **kw)
        return self._ax

    def inset(self, x, y, w, h, **kw):
        """Axes at (x, y, w, h) inches relative to the main box's top-left corner."""
        bx, by, _, _ = self.box
        return self.fig.add_axes(self._frac(bx + x, by + y, w, h), **kw)

    def grid(self, nrows, ncols, wspace=0.35, hspace=0.35, width_ratios=None,
             height_ratios=None, **kw):
        """Split the main box into sub-axes. wspace/hspace in inches. Returns 2-D list."""
        bx, by, bw, bh = self.box
        wr = width_ratios or [1] * ncols
        hr = height_ratios or [1] * nrows
        cw = [(bw - wspace * (ncols - 1)) * r / sum(wr) for r in wr]
        rh = [(bh - hspace * (nrows - 1)) * r / sum(hr) for r in hr]
        out, y = [], by
        for i in range(nrows):
            row, x = [], bx
            for j in range(ncols):
                row.append(self.fig.add_axes(self._frac(x, y, cw[j], rh[i]), **kw))
                x += cw[j] + wspace
            out.append(row)
            y += rh[i] + hspace
        return out

    @property
    def left_margin(self):
        """Distance in inches from the cell's left edge to the axes box."""
        return self.box[0] - self.cell[0]

    def axes_fraction_x(self, inches_from_cell_left):
        """Convert a cell-relative x (inches) into main-axes fraction units."""
        return (self.cell[0] + inches_from_cell_left - self.box[0]) / self.box[2]


# --------------------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------------------
@dataclass
class Panel:
    draw: Callable[[Region], object]
    margins: dict | Callable = field(default_factory=dict)  # inches, or fn(theme) -> dict;
    # each side is max(theme default, this). Use "left!" etc. to force a smaller value.
    aspect: float | None = None                   # axes height / width (1.0 for ROC)
    title: str | None = None
    label: str | bool | None = None               # None = auto letter, False = none
    no_axes_frame: bool = False

    def save(self, path, *, width=3.4, height=1.6, theme: Theme | None = None,
             formats=("png", "svg"), dpi=300, label=False):
        """Render this panel alone at `width` inches, axes `height` inches tall."""
        c = Composite(width=width, theme=theme or get_theme(), outer=0.02)
        c.row([replace(self, label=self.label if label else False)], height=height)
        return c.save(path, formats=formats, dpi=dpi)


# --------------------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------------------
@dataclass
class _Row:
    cells: list
    height: float
    gap_after: float | None


class Composite:
    def __init__(self, width="double", theme: Theme | None = None, columns=12,
                 outer=None, start_letter=0):
        self.theme = theme or get_theme()
        self.W = WIDTHS.get(width, width) if isinstance(width, str) else float(width)
        self.columns = columns
        self.outer = self.theme.outer if outer is None else outer
        self.rows: list[_Row] = []
        self.start_letter = start_letter
        self.fig = None
        self.regions = []

    def row(self, cells: Sequence, height=1.6, gap_after=None):
        """cells: Panel or (Panel, span) items; spans default to an even split of the
        12 columns. A None entry (or (None, span)) leaves an empty slot.
        height: axes height in inches (identical for every panel in the row)."""
        norm = []
        bare = [c for c in cells if not isinstance(c, tuple)]
        used = sum(c[1] for c in cells if isinstance(c, tuple))
        each = (self.columns - used) / max(len(bare), 1)
        for c in cells:
            norm.append(c if isinstance(c, tuple) else (c, each))
        self.rows.append(_Row(norm, height, gap_after))
        return self

    # ---- layout ----------------------------------------------------------------------
    def _margins(self, p: Panel):
        t = self.theme
        pm = p.margins(t) if callable(p.margins) else (p.margins or {})
        m = {k: max(getattr(t, k), pm.get(k, 0.0)) for k in ("left", "right", "top", "bottom")}
        if p.title:
            m["top"] = max(m["top"], t.top + t.title_height)
        for k in ("left", "right", "top", "bottom"):   # explicit small margins allowed
            if pm.get(k + "!") is not None:
                m[k] = pm[k + "!"]
        return m

    def _plan(self):
        t = self.theme
        unit = (self.W - 2 * self.outer) / self.columns
        y = self.outer
        plan = []
        for r in self.rows:
            ms = [self._margins(p) if p is not None else None for p, _ in r.cells]
            top = max(m["top"] for m in ms if m) if any(ms) else t.top
            bot = max(m["bottom"] for m in ms if m) if any(ms) else t.bottom
            col = 0.0
            cells = []
            for (p, span), m in zip(r.cells, ms):
                cx = self.outer + col * unit
                cw = span * unit
                col += span
                if p is None:
                    continue
                last = col >= self.columns - 1e-9
                ax_x = cx + m["left"]
                ax_w = cw - m["left"] - m["right"] - (0 if last else t.gutter)
                if p.aspect:
                    tw = r.height / p.aspect
                    if tw < ax_w:
                        ax_x += (ax_w - tw) / 2
                        ax_w = tw
                cells.append((p, (cx, y, cw, top + r.height + bot), (ax_x, y + top, ax_w, r.height)))
            total = top + r.height + bot
            plan.append((cells, y, total))
            y += total + (t.row_gap if r.gap_after is None else r.gap_after)
        height = y - (t.row_gap if self.rows and self.rows[-1].gap_after is None else 0) + self.outer
        return plan, height

    def render(self):
        t = self.theme.apply()
        plan, H = self._plan()
        fig = plt.figure(figsize=(self.W, H))
        self.fig = fig
        k = self.start_letter
        for cells, y0, rh in plan:
            for p, cell, box in cells:
                reg = Region(fig, t, cell, box)
                p.draw(reg)
                self.regions.append(reg)
                lab = p.label
                if lab is None:
                    lab = t.letter(k)
                    k += 1
                if lab:
                    fig.text((cell[0] + 0.02) / self.W, 1 - (cell[1] + 0.03) / H, lab,
                             fontsize=t.panel_label_size, fontweight="bold", ha="left", va="top")
                if p.title:
                    dx = 0.02 + (t.panel_label_size * 1.15 / 72 if lab else 0)
                    fig.text((cell[0] + dx) / self.W, 1 - (cell[1] + 0.035) / H, p.title,
                             fontsize=t.title_size, fontweight="semibold", ha="left", va="top")
        if t.frames:
            self._draw_frames(plan, H)
        return fig

    def _draw_frames(self, plan, H):
        t = self.theme
        W = self.W
        def line(x0, y0, x1, y1):
            self.fig.add_artist(Line2D([x0 / W, x1 / W], [1 - y0 / H, 1 - y1 / H],
                                       transform=self.fig.transFigure, color=t.frame_color,
                                       lw=1.1, zorder=10))
        o = self.outer / 2
        line(o, o, W - o, o); line(o, H - o, W - o, H - o)
        line(o, o, o, H - o); line(W - o, o, W - o, H - o)
        for i, (cells, y0, rh) in enumerate(plan):
            if i:
                yy = y0 - t.row_gap / 2
                line(o, yy, W - o, yy)
            for p, cell, box in cells[1:]:
                xx = cell[0] - 0.01
                line(xx, y0 - (t.row_gap / 2 if i else o), xx, y0 + rh + t.row_gap / 2)

    def save(self, path, formats=("png", "svg"), dpi=300):
        if self.fig is None:
            self.render()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        outs = []
        for fmt in formats:
            out = path.with_suffix(f".{fmt}")
            self.fig.savefig(out, dpi=dpi)
            outs.append(out)
        plt.close(self.fig)
        self.fig = None
        return outs
