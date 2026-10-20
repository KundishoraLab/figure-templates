"""Grotta bars: stacked horizontal distribution of an ordinal outcome (e.g. mRS 0-6)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.patches import Patch

from .style import MRS_COLORS


def plot_grotta(ax, data: Sequence[Sequence[float]], group_labels: Sequence[str], *,
                colors=MRS_COLORS, bar_height=0.6, value_fmt="{:.1f}", dark_from=4,
                light_text="white", dark_text="#222222", connectors=True,
                connector_kw=None, connect_right_edge=True, xlabel="Percentage of Patients",
                xticks=range(0, 101, 10), fontsize=None, label_fontsize=None,
                label_weight="bold", min_label_width=0.0):
    """Stacked bars, top group first. `data` rows are percentages per category (sum ~100).

    Dashed connectors join matching category boundaries of adjacent bars.
    Segments with index >= dark_from get white value labels.
    """
    data = np.asarray(data, float)
    G, K = data.shape
    ys = np.arange(G)[::-1].astype(float)          # first group on top
    cum = np.cumsum(data, axis=1)
    left = cum - data

    for g in range(G):
        for k in range(K):
            ax.barh(ys[g], data[g, k], left=left[g, k], height=bar_height, color=colors[k],
                    lw=0, zorder=2)
            if value_fmt and data[g, k] >= min_label_width:
                ax.text(left[g, k] + data[g, k] / 2, ys[g], value_fmt.format(data[g, k]),
                        ha="center", va="center", fontsize=fontsize, zorder=3,
                        color=light_text if k >= dark_from else dark_text)

    if connectors:
        ckw = dict(color="black", lw=1.0, ls=(0, (3, 2.5)), zorder=1, clip_on=False)
        ckw.update(connector_kw or {})
        for g in range(G - 1):
            y_top, y_bot = ys[g] - bar_height / 2, ys[g + 1] + bar_height / 2
            for k in range(K - 1):
                ax.plot([cum[g, k], cum[g + 1, k]], [y_top, y_bot], **ckw)
            if connect_right_edge:
                ax.plot([cum[g, -1], cum[g + 1, -1]], [y_top, y_bot], **ckw)

    ax.set_yticks(ys)
    ax.set_yticklabels(group_labels, fontweight=label_weight, fontsize=label_fontsize,
                       linespacing=1.15)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlim(0, 100)
    ax.set_ylim(ys[-1] - 0.5 - bar_height / 2 * 0.6, ys[0] + 0.5 + bar_height / 2 * 0.6)
    ax.set_xticks(list(xticks))
    ax.set_xlabel(xlabel)
    return ys


def mrs_legend(ax_or_fig, title="Score on the Modified Rankin Scale at Day 90",
               colors=MRS_COLORS, labels=None, loc="lower center", bbox=(0.5, 1.0),
               fontsize=None, title_fontsize=None, handlelength=1.6, columnspacing=1.6,
               **kw):
    labels = labels or [str(i) for i in range(len(colors))]
    handles = [Patch(color=c, lw=0) for c in colors]
    leg = ax_or_fig.legend(handles, labels, title=title, ncol=len(colors), loc=loc,
                           bbox_to_anchor=bbox, fontsize=fontsize,
                           title_fontsize=title_fontsize, handlelength=handlelength,
                           handleheight=1.1, columnspacing=columnspacing, handletextpad=0.5,
                           frameon=False, **kw)
    leg.get_title().set_fontweight("bold")
    return leg
