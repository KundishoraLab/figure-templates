"""Flow diagrams: CONSORT-style cohort flow, boxes with text rows and orthogonal arrows.

A diagram is a list of FlowNode (position and size as FRACTIONS of the drawing box, so the
same spec renders at any panel size) and a list of edges (from, to, style). Boxes are white
with a dark rule, text is set from the active Theme (title_size for the box title,
font_size for the rows); an "excluded" side box is dashed. Arrows run straight down, or as
an elbow (down, across, down) when the target is not under the source; a side arm runs
horizontally from the source's edge.

    nodes = [FlowNode("inst", 0.25, 0.42, "3 institutional cohorts", ["CHOP, n = 110", "n = 344"], tier=0),
             FlowNode("sporadic", 0.5, 0.5, "Sporadic bAVMs", ["n = 530"], tier=1), ...]
    plot_flowchart(ax, nodes, [("inst", "sporadic"), ("inst", "excl", "dashed")], w, h)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .assembly import get_theme

RULE = "#333333"


@dataclass
class FlowNode:
    name: str
    fx: float            # centre x, fraction of the box width
    fw: float            # width, fraction
    title: str = ""      # bold first line(s); "\n" allowed
    lines: list = field(default_factory=list)   # regular rows under the title
    tier: int | None = None   # row of the diagram, 0 at the top: height and y are then
                              # computed from the text (see layout_tiers)
    fy: float | None = None   # centre y, fraction from the top (given, or set by the layout)
    fh: float | None = None   # height, fraction (given, or set by the layout)
    style: str = "solid"                        # "solid" | "dashed"
    align: str = "center"                       # text alignment of the rows
    bold_last: bool = False                     # last row bold (a total)

    def rows(self, t):
        """(text, weight, size) per line, from the active theme: titles at font_size bold,
        rows at small_size."""
        out = [(s, "bold", t.font_size) for s in self.title.split("\n")] if self.title else []
        for i, s in enumerate(self.lines):
            bold = self.bold_last and i == len(self.lines) - 1
            out.append((s, "bold" if bold else "normal", t.small_size))
        return out


LEAD = 1.18          # line spacing, in em
PAD = 0.08           # box padding above and below the text, inches


def text_height(node, t):
    return sum(sz * LEAD for _, _, sz in node.rows(t)) / 72


def layout_tiers(nodes, height_in, t, *, gap=0.16, top=0.01):
    """Set fy and fh of every node that has a tier: each tier is as tall as its tallest
    text plus padding; tiers stack from the top with `gap` inches between them, and the
    gap shrinks (never below 0.06 in) when the stack would not fit."""
    tiers = sorted({n.tier for n in nodes if n.tier is not None})
    h = {k: max(text_height(n, t) for n in nodes if n.tier == k) + 2 * PAD for k in tiers}
    need = sum(h.values()) + gap * (len(tiers) - 1) + 2 * top
    if need > height_in and len(tiers) > 1:
        gap = max(0.06, (height_in - sum(h.values()) - 2 * top) / (len(tiers) - 1))
    y = top
    for k in tiers:
        for n in nodes:
            if n.tier == k:
                n.fh, n.fy = h[k] / height_in, (y + h[k] / 2) / height_in
        y += h[k] + gap


def _edge_points(a: FlowNode, b: FlowNode, W, H):
    """Start and end in inches (y down) for the arrow from a to b."""
    ax, ay, aw, ah = a.fx * W, a.fy * H, a.fw * W, a.fh * H
    bx, by, bw, bh = b.fx * W, b.fy * H, b.fw * W, b.fh * H
    if abs(by - ay) < (ah + bh) / 2:                   # same tier: a side arm
        if bx > ax:
            return [(ax + aw / 2, ay), (bx - bw / 2, ay)]
        return [(ax - aw / 2, ay), (bx + bw / 2, ay)]
    top, bot = (ay + ah / 2), (by - bh / 2)
    if abs(bx - ax) < 0.02 * W:                        # straight down
        return [(ax, top), (bx, bot)]
    mid = (top + bot) / 2                              # elbow: down, across, down
    return [(ax, top), (ax, mid), (bx, mid), (bx, bot)]


def plot_flowchart(ax, nodes, edges, width_in, height_in, *, lw=0.8, arrow_scale=6):
    """Draw nodes and edges on `ax`, which is sized `width_in` x `height_in` inches."""
    t = get_theme()
    if any(n.tier is not None for n in nodes):
        layout_tiers(nodes, height_in, t)
    ax.set_xlim(0, width_in)
    ax.set_ylim(height_in, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    by_name = {n.name: n for n in nodes}
    for e in edges:
        a, b = by_name[e[0]], by_name[e[1]]
        style = e[2] if len(e) > 2 else "solid"
        pts = _edge_points(a, b, width_in, height_in)
        ls = (0, (3, 2)) if style == "dashed" else "-"
        for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x0, x1], [y0, y1], color=RULE, lw=lw, ls=ls, solid_capstyle="butt",
                    zorder=1)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=arrow_scale, color=RULE, lw=lw,
                                     linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))
    for n in nodes:
        x, y, w, h = n.fx * width_in, n.fy * height_in, n.fw * width_in, n.fh * height_in
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor="white", edgecolor=RULE, lw=lw,
                                    linestyle=(0, (3, 2)) if n.style == "dashed" else "-",
                                    zorder=2))
        rows = n.rows(t)
        if not rows:
            continue
        total = text_height(n, t)
        yy = y - total / 2
        tx = x if n.align == "center" else x - w / 2 + 0.06
        for s, wt, sz in rows:
            step = sz * LEAD / 72
            ax.text(tx, yy + step / 2, s, ha=n.align, va="center", fontsize=sz,
                    fontweight=wt, zorder=3)
            yy += step
    return ax
