"""Trial-design schematics: arm bands on a broken time axis, run-in box, timeline events.

    schema = TrialSchema(
        anchors=[(-4, 0.0, "−4\\nmo"), (-0.5, 0.134, "−2\\nwk"), (0, 0.241, "0"),
                 (1, 0.26, "1\\nmo"), (12, 0.495, "12\\nmo"), (24, 0.749, "24\\nmo"),
                 (72, 1.0, "72\\nmo")],
        breaks=[(-4, -0.5), (-0.5, 0, 0.3), (24, 72)],
        bands=[Band("CAB LA ... (N=270)", 1, 24, fc=..., color=..., post=Band("4-yr Post-trial ...", 24, 72)),
               Band("2 NRTIs+DTG\\n(N=270)", 0, 24, post=Band("Transition to standard care", 24, 72))],
        events=[Event(-0.5, "Confirm viral load <200 copies/ml"), Event(0, [...], lift=0.36), ...],
        run_in=RunIn(-4, -0.1, [Para("Screening Phase", weight="bold"), ...]),
        randomization=0, bracket=Bracket(0, 1, "Optional 1-mo\\noral lead-in"),
        criteria=[Para("Screening", weight="bold"), Para("People living with ...", hang=1.5), Item("1.", ...)],
        axis_title="Time Relative to Randomization")

Times are in any unit you like; anchors map them onto [0, 1] of the timeline width
piecewise-linearly, so compressed segments sit between // breaks. All positions on the
page are computed from the panel width, and the height is measured from the text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .diagram import Canvas, Item, Para, TextBlock, text_width


@dataclass
class Band:
    text: str | Sequence
    t0: float
    t1: float
    fc: str = "#E3E3E3"
    color: str = "#4D4D4D"
    weight: str = "bold"
    post: "Band | None" = None       # lighter continuation, drawn as a chevron
    tip: bool = False                 # chevron on this band itself (when there is no post)


@dataclass
class Event:
    t: float
    text: str | Sequence             # str or list of Para
    ha: str = "center"               # text block alignment relative to t: center | right | left
    dx: float = 0.0                  # inches
    width: float = 1.35              # text block width (inches)
    lift: float = 0.0                # raise the text block (inches) above the common baseline
    guide: bool = True               # light vertical line from the band zone to the axis
    arrow: bool = True


@dataclass
class RunIn:
    t0: float
    t1: float
    parts: Sequence
    fc: str = "#D5E8B4"
    color: str = "#3E8E41"


@dataclass
class Bracket:
    t0: float
    t1: float
    text: str
    color: str = "#2C78B9"


@dataclass
class SchemaStyle:
    size: float = 7.2
    band_h: float = 0.62
    band_gap: float = 0.095
    tip: float = 0.33
    arrow_len: float = 0.2
    criteria_width: float = 1.72
    criteria_color: str = "#3A9142"
    guide_color: str = "#C9C9C9"
    dash_color: str = "#8C8C8C"
    run_in_inset: float = 0.11


class TrialSchema:
    def __init__(self, anchors, bands: Sequence[Band], events: Sequence[Event] = (), breaks=(),
                 run_in: RunIn | None = None, randomization: float | None = None,
                 bracket: Bracket | None = None, criteria: Sequence | None = None,
                 axis_title: str = "", style: SchemaStyle | None = None):
        self.anchors = sorted(anchors, key=lambda a: a[0])
        self.bands, self.events, self.breaks = list(bands), list(events), list(breaks)
        self.run_in, self.rand, self.bracket = run_in, randomization, bracket
        self.criteria, self.axis_title = criteria, axis_title
        self.st = style or SchemaStyle()

    def tx(self, t):
        ts = [a[0] for a in self.anchors]
        fs = [a[1] for a in self.anchors]
        return self.x_lo + float(np.interp(t, ts, fs)) * (self.x_hi - self.x_lo)

    # ---- layout ------------------------------------------------------------------------
    def layout(self, W):
        st = self.st
        lh = st.size * 1.22 / 72
        self.W = W
        self.crit = None
        x_left = 0.0
        if self.criteria:
            self.crit = TextBlock(self.criteria, st.criteria_width, size=st.size, color=st.criteria_color)
            self.rule_x = st.criteria_width + 0.07
            x_left = self.rule_x + 0.05
        self.x_lo = x_left
        self.x_hi = W - st.tip - 0.04
        # events: text blocks bottom-aligned on one baseline, arrows down to y_arrow
        self.ev_blocks = []
        for e in self.events:
            parts = [Para(e.text, align="center")] if isinstance(e.text, str) else e.text
            self.ev_blocks.append(TextBlock(parts, e.width, size=st.size))
        top_need = max([b.height + e.lift for b, e in zip(self.ev_blocks, self.events)] + [0])
        self.y_text_bottom = top_need + 0.02
        self.y_arrow = self.y_text_bottom + 0.06 + st.arrow_len
        self.br_block = None
        zone = 0.09
        if self.bracket:
            self.br_block = TextBlock([Para(self.bracket.text, align="center")], 1.2, size=st.size,
                                      color=self.bracket.color)
            zone = self.br_block.height + 0.14
        self.y_b0 = self.y_arrow + zone
        y = self.y_b0
        self.band_y = []
        for _ in self.bands:
            self.band_y.append(y)
            y += st.band_h + st.band_gap
        self.y_b1 = y - st.band_gap
        self.y_axis = self.y_b1 + 0.07
        self.y_ticklab = self.y_axis + 0.07
        self.y_title = self.y_ticklab + 2 * lh + 0.06
        h = self.y_title + lh * 1.1
        if self.crit:
            h = max(h, self.crit.height + 0.05)
        self.height = h
        return self

    # ---- drawing -----------------------------------------------------------------------
    def draw(self, cv: Canvas):
        st, tx = self.st, self.tx
        lh = st.size * 1.22 / 72
        # criteria column, rule, and elbow into the run-in box
        if self.crit:
            self.crit.draw(cv, 0.0, 0.02)
            y_rule1 = self.y_b1 - st.run_in_inset
            cv.line([(self.rule_x, 0.04), (self.rule_x, y_rule1)], color=st.criteria_color, lw=1.0)
            if self.run_in:
                cxr = (tx(self.run_in.t0) + tx(self.run_in.t1)) / 2
                ye = self.y_b0 - 0.1
                cv.arrow([(self.rule_x, ye), (cxr, ye), (cxr, self.y_b0 + st.run_in_inset)],
                         color=st.criteria_color, lw=1.0)
        # guides and randomization line, behind everything
        g_top = self.y_b0 - 0.04
        for e in self.events:
            if e.guide and e.t != self.rand:
                cv.line([(tx(e.t), g_top), (tx(e.t), self.y_axis)], color=st.guide_color, lw=0.8, zorder=0.5)
        if self.rand is not None:
            cv.line([(tx(self.rand), g_top), (tx(self.rand), self.y_axis)], color=st.dash_color, lw=0.8,
                    ls=(0, (3, 2.5)), zorder=0.6)
        # events
        for e, b in zip(self.events, self.ev_blocks):
            x = tx(e.t) + e.dx
            left = x - b.width / 2 if e.ha == "center" else (x - b.width if e.ha == "right" else x)
            b.draw(cv, left, self.y_text_bottom - e.lift - b.height)
            if e.arrow:
                cv.arrow([(tx(e.t), self.y_text_bottom - e.lift + 0.05), (tx(e.t), self.y_arrow)], lw=0.9)
        # lead-in bracket
        if self.bracket:
            b = self.bracket
            x0, x1 = tx(b.t0), tx(b.t1)
            yb = self.y_b0 - 0.045
            cv.line([(x0, yb + 0.04), (x0, yb), (x1, yb), (x1, yb + 0.04)], color=b.color, lw=0.9)
            self.br_block.draw(cv, (x0 + x1) / 2 - self.br_block.width / 2, yb - 0.04 - self.br_block.height)
        # bands
        for bd, y in zip(self.bands, self.band_y):
            self._band(cv, bd, y)
        # run-in box and the fork into the arms
        if self.run_in:
            r = self.run_in
            x0, x1 = tx(r.t0), tx(r.t1)
            y0, y1 = self.y_b0 + st.run_in_inset, self.y_b1 - st.run_in_inset
            cv.rect(x0, y0, x1 - x0, y1 - y0, fc=r.fc, zorder=2)
            blk = TextBlock(r.parts, x1 - x0 - 0.08, size=st.size, color=r.color)
            blk.draw(cv, x0 + 0.04, (y0 + y1) / 2 - blk.height / 2)
            if self.rand is not None and len(self.bands) > 1:
                xr = tx(self.rand)
                xf = xr - 0.14
                ys = []
                mid = (y0 + y1) / 2
                for y in self.band_y:
                    cy = y + st.band_h / 2
                    ys.append(cy + 0.2 * st.band_h * np.sign(mid - cy))
                cv.line([(x1, mid), (xf, mid)], lw=0.8)
                cv.line([(xf, min(ys)), (xf, max(ys))], lw=0.8)
                for yy in ys:
                    cv.arrow([(xf, yy), (xr - 0.015, yy)], lw=0.8)
        self._axis(cv, lh)

    def _band(self, cv, bd: Band, y):
        st, tx = self.st, self.tx
        h = st.band_h
        x0, x1 = tx(bd.t0), tx(bd.t1)
        if bd.tip and bd.post is None:
            cv.chevron(x0, y, x1, h, tip=st.tip, fc=bd.fc)
        else:
            cv.rect(x0, y, x1 - x0, h, fc=bd.fc, zorder=1)
        self._band_text(cv, bd, x0, x1, y)
        if bd.post:
            p = bd.post
            px1 = tx(p.t1)
            cv.chevron(x1, y, px1, h, tip=st.tip, fc=p.fc)
            self._band_text(cv, p, x1, px1 + st.tip * 0.6, y)     # text may run into the point

    def _band_text(self, cv, bd, x0, x1, y):
        st = self.st
        parts = [Para(bd.text, align="center", weight=bd.weight)] if isinstance(bd.text, str) else bd.text
        blk = TextBlock(parts, x1 - x0 - 0.1, size=st.size * 1.04, color=bd.color)
        blk.draw(cv, x0 + 0.05, y + st.band_h / 2 - blk.height / 2)

    def _axis(self, cv, lh):
        tx, st = self.tx, self.st
        ya = self.y_axis
        xa, xb = tx(self.anchors[0][0]), tx(self.anchors[-1][0])
        cv.line([(xa, ya), (xb, ya)], lw=0.9)
        for t, _, lab in self.anchors:
            x = tx(t)
            cv.line([(x, ya), (x, ya + 0.05)], lw=0.9)
            for i, ln in enumerate(lab.split("\n")):
                cv.ax.text(x, self.y_ticklab + (i + 0.8) * lh, ln, ha="center", va="baseline", fontsize=st.size)
        for br in self.breaks:
            ta, tb = br[0], br[1]
            pos = br[2] if len(br) > 2 else 0.5
            x = tx(ta) + pos * (tx(tb) - tx(ta))
            cv.rect(x - 0.025, ya - 0.04, 0.05, 0.08, fc="white", zorder=3.5)
            for dx in (-0.022, 0.022):
                cv.line([(x + dx - 0.03, ya + 0.045), (x + dx + 0.03, ya - 0.045)], lw=0.8, zorder=3.6)
        if self.axis_title:
            cv.ax.text((xa + xb) / 2, self.y_title + 0.8 * lh, self.axis_title, ha="center", va="baseline",
                       fontsize=st.size * 1.08, fontweight="bold")
