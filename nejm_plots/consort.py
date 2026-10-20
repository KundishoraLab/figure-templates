"""CONSORT participant-flow diagrams.

    flow = Consort(
        pre=["845 Participants were assessed for eligibility",
             Exit("305 Were excluded (may have >1 reason for exclusion)",
                  [(87, "Did not meet required viral suppression criteria"), ...]),
             "540 Underwent randomization"],
        arms=[["271 Were assigned to receive long-acting therapy",
               Exit("3 Switched to standard oral treatment during oral lead-in phase"),
               "268 Began long-acting therapy and were included in the ITT-exposed analysis",
               Exit("24 Were excluded (may have >1 reason for exclusion)", [(4, "..."), ...]),
               "244 Were included in the per-protocol analysis"],
              ["269 Were assigned to continue oral therapy",
               None,
               "269 Were included in the ITT-exposed analysis",
               Exit("21 Were excluded ...", [...]),
               "248 Were included in the per-protocol analysis"]])

`pre` runs down the central trunk (stages alternating with optional Exit boxes, which sit
to the right); its last stage splits into the arms. Each arm alternates stage, exit,
stage, ...; arms are aligned row by row, and exits sit on the outer side of their arm.
One arm (no split) and three arms also work. All widths are fractions of the panel width.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .diagram import Canvas, Item, Node, Para, TextBlock


@dataclass
class Exit:
    header: str
    items: Sequence[tuple] = ()
    side: str | None = None          # force "left" / "right"

    def parts(self):
        head = self.header.split(" ", 1)
        out = []
        if len(head) == 2 and head[0].replace(",", "").isdigit():
            out.append(Item(head[0], head[1], indent=0, hang=0.6))
        else:
            out.append(Para(self.header, hang=0.6))
        for c, t in self.items:
            out.append(Item(str(c), t, indent=1.3))
        return out


@dataclass
class SourceGroup:
    """Source cohorts that share one intermediate stage (e.g. three sites genotyped together):
    boxes in the top row, then a bar merging them into `stage`, which then merges with the
    other groups into the trunk. stage=None: the boxes merge straight into the trunk."""
    boxes: Sequence[str]
    stage: str | None = None


@dataclass
class ConsortStyle:
    size: float = 7.2
    edge: str = "#4D4D4D"
    lw: float = 0.8
    arrow_lw: float = 0.8
    pad_x: float = 0.08
    pad_y: float = 0.06
    min_h: float = 0.34              # main boxes
    gap: float = 0.13                # between a box and the next row
    split_drop: float = 0.15         # randomization box -> horizontal bar
    split_rise: float = 0.27         # bar -> arm boxes
    pre_w: float = 0.30              # fractions of the panel width
    pre_exit_w: float = 0.36
    pre_exit_dx: float = 0.107       # trunk -> left edge of the pre exits
    arm_w: float = 0.325
    arm_exit_w: float = 0.27
    arm_exit_dx: float = 0.048       # arm trunk -> near edge of its exits
    trunk: float = 0.508
    arm_centers: Sequence[float] | None = None
    source_gap: float = 0.03         # fraction of the panel width between source boxes
    source_drop: float = 0.15        # source boxes -> merge bar
    source_rise: float = 0.27        # merge bar -> first trunk box


class Consort:
    def __init__(self, pre: Sequence, arms: Sequence[Sequence] = (), style: ConsortStyle | None = None,
                 sources: Sequence = ()):
        """sources: boxes in a top row (one per contributing cohort) that merge into the first
        trunk stage; each is a string or a TextBlock parts list."""
        self.pre, self.arms = list(pre), [list(a) for a in arms]
        self.sources = list(sources)
        self.st = style or ConsortStyle()

    # ---- layout --------------------------------------------------------------------------
    def _block(self, item, w, center):
        st = self.st
        parts = item.parts() if isinstance(item, Exit) else [Para(item, align="center" if center else "left")]
        return TextBlock(parts, w - 2 * st.pad_x, size=st.size)

    def layout(self, W):
        st = self.st
        ops = []                                     # (kind, payload)
        trunk = st.trunk * W
        y = 0.0
        prev = None
        merge_y = None
        if self.sources:
            groups = [g if isinstance(g, SourceGroup) else SourceGroup([g]) for g in self.sources]
            flat = [b for g in groups for b in g.boxes]
            k = len(flat)
            gap = st.source_gap * W
            w = (W - gap * (k - 1)) / k
            blocks = [self._block(sx, w, True) for sx in flat]
            h = max(max(b.height + 2 * st.pad_y for b in blocks), st.min_h)
            nodes = [Node(i * (w + gap), 0.0, w, h) for i in range(k)]
            for n, b in zip(nodes, blocks):
                ops.append(("box", (n, b)))
            bar1 = h + st.source_drop
            feeds = []                                   # (cx, y) of what merges into the trunk
            i = 0
            stage_nodes = []
            for g in groups:
                gn = nodes[i:i + len(g.boxes)]
                i += len(g.boxes)
                gx = (gn[0].cx + gn[-1].cx) / 2
                if g.stage is None:
                    feeds += [(n.cx, n.bottom) for n in gn]
                    continue
                for n in gn:
                    ops.append(("line", [(n.cx, n.bottom), (n.cx, bar1)]))
                if len(gn) > 1:
                    ops.append(("line", [(gn[0].cx, bar1), (gn[-1].cx, bar1)]))
                span = gn[-1].right - gn[0].left
                sw = min(max(st.arm_w * W, 0.55 * span), 0.45 * W)
                b = self._block(g.stage, sw, True)
                left = min(max(gx - sw / 2, 0.0), W - sw)          # inside the panel
                sn = Node(left, bar1 + st.source_rise, sw, max(b.height + 2 * st.pad_y, st.min_h))
                stage_nodes.append((sn, b))
            if stage_nodes:
                hs = max(sn.h for sn, _ in stage_nodes)
                for sn, b in stage_nodes:
                    sn = Node(sn.left, sn.top + (hs - sn.h) / 2, sn.w, sn.h)
                    ops.append(("box", (sn, b)))
                    ops.append(("arrow", [(sn.cx, bar1), (sn.cx, sn.top)]))
                    if abs(sn.cx - gx) > 1e-6:                      # the box was shifted inside
                        ops.append(("line", [(min(gx, sn.cx), bar1), (max(gx, sn.cx), bar1)]))
                    feeds.append((sn.cx, sn.bottom))
                bottom = max(sn.top + sn.h for sn, _ in stage_nodes)
            else:
                bottom = h
            merge_y = bottom + st.source_drop
            for cx, yb in feeds:
                ops.append(("line", [(cx, yb), (cx, merge_y)]))
            xs = sorted(cx for cx, _ in feeds)
            if len(xs) > 1:
                ops.append(("line", [(xs[0], merge_y), (xs[-1], merge_y)]))
            y = merge_y + st.source_rise
        for it in self.pre:
            if it is None:
                continue
            if isinstance(it, Exit):
                w = st.pre_exit_w * W
                b = self._block(it, w, False)
                n = Node(min(trunk + st.pre_exit_dx * W, W - w), y, w, b.height + 2 * st.pad_y)
                ops.append(("box", (n, b)))
                ops.append(("arrow", [(trunk, n.cy), (n.left, n.cy)]))
                y = n.bottom + st.gap
            else:
                w = st.pre_w * W
                b = self._block(it, w, True)
                n = Node(trunk - w / 2, y, w, max(b.height + 2 * st.pad_y, st.min_h))
                ops.append(("box", (n, b)))
                if prev is not None:
                    ops.append(("arrow", [(trunk, prev.bottom), (trunk, n.top)]))
                elif merge_y is not None:
                    ops.append(("arrow", [(trunk, merge_y), (trunk, n.top)]))
                prev = n
                y = n.bottom + st.gap
        if not self.arms:
            self.ops, self.height = ops, y - st.gap
            return self
        k = len(self.arms)
        centers = st.arm_centers or ([st.trunk] if k == 1 else
                                     [st.trunk + (i - (k - 1) / 2) * (0.345 if k == 2 else 0.31) for i in range(k)])
        cxs = [c * W for c in centers]
        # split
        y_bar = prev.bottom + st.split_drop
        if k > 1:
            ops.append(("line", [(trunk, prev.bottom), (trunk, y_bar)]))
            ops.append(("line", [(cxs[0], y_bar), (cxs[-1], y_bar)]))
            y = y_bar + st.split_rise
            tops = [(cx, y_bar) for cx in cxs]
        else:
            y = prev.bottom + st.gap * 2
            tops = [(trunk, prev.bottom)]
        n_rows = max(len(a) for a in self.arms)
        last = [None] * k
        for r in range(n_rows):
            row_nodes = []
            for i, arm in enumerate(self.arms):
                it = arm[r] if r < len(arm) else None
                if it is None:
                    row_nodes.append(None)
                    continue
                cx = cxs[i]
                if isinstance(it, Exit):
                    side = it.side or ("left" if (k > 1 and i < (k - 1) / 2) else "right")
                    w = st.arm_exit_w * W
                    b = self._block(it, w, False)
                    left = cx - st.arm_exit_dx * W - w if side == "left" else cx + st.arm_exit_dx * W
                    left = min(max(left, 0), W - w)
                    n = Node(left, y, w, b.height + 2 * st.pad_y)
                    row_nodes.append(("exit", n, b, side))
                else:
                    w = st.arm_w * W
                    b = self._block(it, w, True)
                    n = Node(cx - w / 2, y, w, max(b.height + 2 * st.pad_y, st.min_h))
                    row_nodes.append(("stage", n, b, None))
            if all(rn is None for rn in row_nodes):
                continue
            h = max(rn[1].h for rn in row_nodes if rn)
            for i, rn in enumerate(row_nodes):
                if rn is None:
                    continue
                kind, n, b, side = rn
                cx = cxs[i]
                if kind == "stage":
                    n = Node(n.left, y + (h - n.h) / 2, n.w, n.h)
                    ops.append(("box", (n, b)))
                    src = tops[i] if last[i] is None else (cx, last[i].bottom)
                    ops.append(("arrow", [src, (cx, n.top)]))
                    last[i] = n
                else:
                    ops.append(("box", (n, b)))
                    ops.append(("arrow", [(cx, n.cy), (n.right if side == "left" else n.left, n.cy)]))
            y += h + st.gap
        self.ops, self.height = ops, y - st.gap
        return self

    def draw(self, cv: Canvas, x0=0.0, y0=0.0):
        st = self.st
        for kind, pay in self.ops:
            if kind == "box":
                n, b = pay
                cv.rect(x0 + n.left, y0 + n.top, n.w, n.h, fc="white", ec=st.edge, lw=st.lw, zorder=4)
                b.draw(cv, x0 + n.left + st.pad_x, y0 + n.top + (n.h - b.height) / 2, zorder=5)
            elif kind == "line":
                cv.line([(x0 + a, y0 + b) for a, b in pay], lw=st.arrow_lw)
            else:
                cv.arrow([(x0 + a, y0 + b) for a, b in pay], lw=st.arrow_lw)
