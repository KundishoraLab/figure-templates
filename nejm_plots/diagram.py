"""Diagram primitives: measured text blocks, boxes, routed arrows, on an inch canvas.

Everything here works in inches with the origin at the top-left of a Canvas, so layouts
can be computed (and measured for auto height) before anything is drawn.

    cv = Canvas(ax, width, height)                       # ax spans the panel
    blk = TextBlock([Para("Screening", weight="bold"),
                     Para("People living with HIV with a history of ...", hang=1.5),
                     Item("1.", "Documented detectable HIV-1 viral load ...")],
                    width=1.7, size=7.5, color="#3A9142")
    h = blk.height                                        # measured, wrapped
    blk.draw(cv, x=0.1, y=0.1)
    node = cv.box(cx=3.0, top=0.2, w=2.0, text="845 Participants were assessed")
    cv.arrow([(node.cx, node.bottom), (node.cx, 1.2)])

Text blocks
    Para(text, weight, color, align, indent, hang, size, space_before)
        indent / hang in em: first-line indent and extra indent of wrapped lines.
    Item(label, text, indent)
        a label column (counts in CONSORT boxes, "1." in numbered lists) with the text
        hanging after it. Consecutive Items at the same indent share one label column;
        numeric labels are right-aligned in it, others left-aligned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

_R = RendererAgg(100, 100, 72)          # 72 dpi: pixel widths are points


@lru_cache(maxsize=20000)
def _width_pt(s, size, weight, family):
    if not s:
        return 0.0
    fp = FontProperties(family=family, size=size, weight=weight)
    w, _, _ = _R.get_text_width_height_descent(s, fp, ismath=False)
    return w


def text_width(s, size, weight="normal"):
    """Rendered width of s in inches with the current font."""
    fam = tuple(mpl.rcParams["font.sans-serif"])[0]
    return _width_pt(s, size, weight, fam) / 72


def wrap(text, width, size, weight="normal", first_offset=0.0, rest_offset=0.0):
    """Greedy word wrap to `width` inches. Explicit \\n are kept. Returns lines."""
    out = []
    for hard in text.split("\n"):
        words = hard.split(" ")
        line, avail = "", width - (first_offset if not out else rest_offset)
        for w in words:
            cand = w if not line else line + " " + w
            if line and text_width(cand, size, weight) > avail:
                out.append(line)
                line, avail = w, width - rest_offset
            else:
                line = cand
        out.append(line)
    return out


@dataclass
class Para:
    text: str
    weight: str = "normal"
    color: str | None = None
    align: str = "left"            # left | center | right
    indent: float = 0.0            # em, first line
    hang: float = 0.0              # em, added to wrapped lines
    size: float | None = None
    space_before: float = 0.0      # lines


@dataclass
class Item:
    label: str
    text: str
    indent: float = 0.0            # em, where the label column starts
    weight: str = "normal"
    color: str | None = None
    size: float | None = None
    space_before: float = 0.0
    gap: float = 0.3               # em between label column and text
    hang: float = 0.0              # em, extra indent of the text's wrapped lines


@dataclass
class _Frag:
    x: float
    y: float                       # baseline, inches from block top
    s: str
    weight: str
    color: str
    ha: str
    size: float


class TextBlock:
    """Wrapped, measured block of paragraphs and list items."""

    def __init__(self, parts: Sequence, width: float, size: float = 7.5, color="black",
                 line_spacing: float = 1.22):
        self.width, self.size, self.color, self.ls = width, size, color, line_spacing
        self.parts = [Para(p) if isinstance(p, str) else p for p in parts]
        self.frags: list[_Frag] = []
        self.height = self._layout()

    def _layout(self):
        em = self.size / 72
        lh = self.size * self.ls / 72
        y = 0.0
        # label column widths for runs of Items with the same indent
        colw = {}
        for p in self.parts:
            if isinstance(p, Item):
                sz = p.size or self.size
                colw[p.indent] = max(colw.get(p.indent, 0), text_width(p.label, sz, p.weight))
        for p in self.parts:
            sz = p.size or self.size
            col = p.color or self.color
            y += p.space_before * lh
            if isinstance(p, Item):
                x0 = p.indent * em
                cw = colw[p.indent]
                tx = x0 + cw + p.gap * em
                numeric = p.label.replace(".", "").replace(",", "").strip().isdigit() and not p.label.endswith(".")
                hx = p.hang * em
                lines = wrap(p.text, self.width - tx, sz, p.weight, 0.0, hx)
                self.frags.append(_Frag(x0 + cw if numeric else x0, y + lh * 0.8, p.label, p.weight, col,
                                        "right" if numeric else "left", sz))
                for i, ln in enumerate(lines):
                    self.frags.append(_Frag(tx + (hx if i else 0), y + lh * 0.8, ln, p.weight, col, "left", sz))
                    y += lh
            else:
                fo, ro = p.indent * em, (p.indent + p.hang) * em
                lines = wrap(p.text, self.width, sz, p.weight, fo, ro)
                for i, ln in enumerate(lines):
                    off = fo if i == 0 else ro
                    if p.align == "center":
                        x, ha = self.width / 2, "center"
                    elif p.align == "right":
                        x, ha = self.width, "right"
                    else:
                        x, ha = off, "left"
                    self.frags.append(_Frag(x, y + lh * 0.8, ln, p.weight, col, ha, sz))
                    y += lh
        return y

    @property
    def natural_width(self):
        """Width of the longest line (inches); useful to shrink-wrap a box."""
        w = 0
        for f in self.frags:
            fw = text_width(f.s, f.size, f.weight)
            w = max(w, f.x + fw if f.ha == "left" else (f.x if f.ha == "right" else self.width / 2 + fw / 2))
        return w

    def draw(self, cv: "Canvas", x, y, zorder=5, **kw):
        for f in self.frags:
            cv.ax.text(x + f.x, y + f.y, f.s, fontsize=f.size, fontweight=f.weight, color=f.color,
                       ha=f.ha, va="baseline", zorder=zorder, **kw)


# --------------------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------------------
@dataclass
class Node:
    left: float
    top: float
    w: float
    h: float

    @property
    def right(self): return self.left + self.w
    @property
    def bottom(self): return self.top + self.h
    @property
    def cx(self): return self.left + self.w / 2
    @property
    def cy(self): return self.top + self.h / 2


class Canvas:
    """An axes in inch units, origin top-left, y downward."""

    def __init__(self, ax, width, height):
        self.ax, self.W, self.H = ax, width, height
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        ax.patch.set_visible(False)

    # -- shapes -------------------------------------------------------------------------
    def rect(self, x, y, w, h, fc="white", ec="none", lw=0.8, zorder=2, **kw):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=zorder, **kw))
        return Node(x, y, w, h)

    def chevron(self, x0, y, x1, h, tip=0.3, fc="#DDDDDD", ec="none", zorder=1):
        """Band from x0 to x1 that ends in an arrow point reaching x1 + tip."""
        pts = [(x0, y), (x1, y), (x1 + tip, y + h / 2), (x1, y + h), (x0, y + h)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, zorder=zorder, lw=0))

    def line(self, pts, color="black", lw=0.8, ls="-", zorder=3):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=zorder, solid_capstyle="butt",
                     solid_joinstyle="miter")

    def arrow(self, pts, color="black", lw=0.8, head=3.0, zorder=3):
        """Polyline through pts with a filled head at the last point (head in points)."""
        pts = list(pts)
        if len(pts) > 2:
            self.line(pts[:-1], color=color, lw=lw, zorder=zorder)
        a, b = pts[-2], pts[-1]
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle=f"-|>,head_length={head * 0.11},head_width={head * 0.06}",
                                          mutation_scale=10, color=color, lw=lw, zorder=zorder,
                                          shrinkA=0, shrinkB=0, joinstyle="miter"))

    def box(self, text=None, *, cx=None, left=None, top, w, block: TextBlock | None = None,
            size=7.5, align="center", pad_x=0.07, pad_y=0.06, min_h=0.0, fc="white",
            ec="#555555", lw=0.8, color="black", weight="normal", draw=True, zorder=4):
        """Box with wrapped text. Give cx or left. Returns a Node (height from the text)."""
        tw = w - 2 * pad_x
        if block is None:
            parts = [Para(text, align=align, weight=weight)] if isinstance(text, str) else text
            block = TextBlock(parts, tw, size=size, color=color)
        h = max(block.height + 2 * pad_y, min_h)
        left = cx - w / 2 if left is None else left
        node = Node(left, top, w, h)
        if draw:
            self.rect(left, top, w, h, fc=fc, ec=ec, lw=lw, zorder=zorder)
            block.draw(self, left + pad_x, top + (h - block.height) / 2, zorder=zorder + 1)
        return node


def box_height(text_or_parts, w, size=7.5, pad_x=0.07, pad_y=0.06, min_h=0.0, align="center",
               weight="normal"):
    """Height a Canvas.box would get, without drawing it."""
    parts = [Para(text_or_parts, align=align, weight=weight)] if isinstance(text_or_parts, str) else text_or_parts
    return max(TextBlock(parts, w - 2 * pad_x, size=size).height + 2 * pad_y, min_h)
