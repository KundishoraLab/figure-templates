"""Figure makers: each function takes data and returns a Panel.

A Panel can be saved on its own (panel.save(...)) or placed in a Composite row. All sizes
(fonts, line widths, margins) come from the active Theme, so panels made here look the
same wherever they land.

    from nejm_plots.panels import *
    p = km_panel(arms, xlim=(0, 14), xticks=range(0, 15, 2), xlabel="Years")
    p.save("out/panels/km", width=3.6, height=1.7)

Every maker accepts title=, label= (None auto letter, False none, or a string) and
margins= (extra inches) to override the layout for that one panel.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter

from .assembly import Panel, Region, get_theme
from .style import COLORS, MRS_COLORS, add_y_break, auto_y_break, log10_axis
from .km import KMArm, add_at_risk_table, add_km_pvalue, label_curves, plot_km
from .forest import ForestRow, plot_forest
from .grotta import mrs_legend, plot_grotta
from .estimates import plot_point_range, add_difference_bracket
from .trajectory import divergence_onset, fit_trajectory, plot_trajectory_scatter, tint
from .groups import add_brackets, compare_groups, plot_groups
from .scatter import plot_correlation, plot_roc, plot_volcano
from .timecourse import (plot_bar_points, plot_time_line, plot_time_strip, plot_violin_outline,
                         segment_positions, segmented_time_axis)
from .flow import add_gate, plot_flow
from .flowchart import FlowNode, plot_flowchart

__all__ = ["custom_panel", "km_panel", "roc_panel", "groups_panel", "paired_panel",
           "volcano_panel", "correlation_panel", "trajectory_panel", "forest_panel",
           "grotta_panel", "point_range_panel", "timecourse_panel", "violin_panel",
           "bar_points_panel", "flow_panel", "legend_panel", "flowchart_panel"]


def _common(kw):
    return {k: kw.pop(k) for k in ("title", "label") if k in kw}


def _merge(base, extra):
    """Margins resolved at layout time: base is fn(theme) -> dict, extra is the user's dict."""
    return lambda t: {**base(t), **(extra or {})}


# --------------------------------------------------------------------------------------
def custom_panel(fn, *args, margins=None, aspect=None, title=None, label=None, **kwargs):
    """Wrap any fn(ax, *args, **kwargs) as a panel."""
    return Panel(lambda r: fn(r.axes(), *args, **kwargs), margins or {}, aspect, title, label)


# --------------------------------------------------------------------------------------
# Kaplan-Meier / cumulative incidence
# --------------------------------------------------------------------------------------
def km_panel(arms: Sequence[KMArm], *, xlim, xticks=None, ylim=(0, 100), yticks=None,
             xlabel="Time", ylabel=None, cumulative_incidence=False, at_risk=True,
             at_risk_at=None, cumulative_events=False, pvalue=None, legend="best",
             curve_labels=None, ybreak=None, margins=None, title=None, label=None):
    """KM (or 1 - KM) curves with an optional numbers-at-risk table.
    pvalue: None, "logrank", or literal text. legend: matplotlib loc, or None.
    curve_labels: {"Arm": (x, y)} to label curves directly instead of a legend.
    ybreak: (lo, hi) for a 0 // hi axis."""
    n = len(arms)
    ylabel = ylabel or ("Cumulative Incidence (%)" if cumulative_incidence else "Percentage of Patients")

    def draw(r: Region):
        t = r.theme
        ax = r.axes()
        row_pt = t.small_size * 1.5
        for a in arms:
            a.lw = t.line_lw
        fits = plot_km(ax, arms, xlim=xlim, xticks=xticks, ylim=ylim, yticks=yticks,
                       xlabel=xlabel, ylabel=ylabel, cumulative_incidence=cumulative_incidence)
        if ybreak:
            add_y_break(ax, *ybreak, gap_frac=0.1, mark_size=t.font_size)
            if yticks is not None:
                ax.set_yticks(list(yticks))
        if at_risk:
            at = list(at_risk_at if at_risk_at is not None else (xticks if xticks is not None else ax.get_xticks()))
            add_at_risk_table(ax, fits, at=at, cumulative_events=cumulative_events,
                              label_x=r.axes_fraction_x(0.03),
                              title_offset=t.bottom * 72 + row_pt * 0.3, row_height=row_pt,
                              fontsize=t.small_size, title_fontsize=t.small_size)
        if pvalue:
            add_km_pvalue(ax, fits if pvalue == "logrank" else None,
                          text=None if pvalue == "logrank" else pvalue, fontsize=t.small_size)
        if curve_labels:
            label_curves(ax, curve_labels, fontsize=t.small_size)
        elif legend:
            ax.legend(loc=legend)
        r.data["fits"] = fits

    lab_w = max(len(a.label) for a in arms)

    def extra(t):
        if not at_risk:
            return {}
        # room for the at-risk row labels left of the first column of numbers
        return {"bottom": t.bottom + (t.small_size * 1.5 * (n + 1) + 4) / 72,
                "left": lab_w * t.small_size * 0.52 / 72 + 0.26, "right": 0.2}
    return Panel(draw, _merge(extra, margins), None, title, label)


# --------------------------------------------------------------------------------------
def roc_panel(models: dict, y_true=None, *, colors=None, n_boot=0, ci_band=False,
              margins=None, title=None, label=None):
    """ROC curves; AUC (with bootstrap CI if n_boot) goes in the legend."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        r.data["auc"] = plot_roc(ax, models, y_true, colors=colors, n_boot=n_boot,
                                 ci_band=ci_band, lw=t.line_lw,
                                 legend_kw=dict(fontsize=t.small_size))
    return Panel(draw, margins or {}, 1.0, title, label)


# --------------------------------------------------------------------------------------
def groups_panel(data, *, colors=None, kind="box", ylabel=None, brackets=None,
                 test="mannwhitney", adjust="holm", paired_ids=None, center=None,
                 log=None, ylim=None, yticks=None, point_size=None, jitter=0.32,
                 hide_ns=True, rotate=0, point_scale=1.0, margins=None, title=None, label=None,
                 **group_kw):
    """Box / violin / bar / points-only comparison of groups.
    data: {group: values}. brackets: None, "all", "adjacent", or [(a, b), ...].
    center: "median" or "mean" bar (useful with kind="none").
    log: (lo_exp, hi_exp) for a log10 axis.
    The P values are stored on the region (region.data["p"]) for the legend/caption."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        names = list(data)
        cols = colors or [COLORS["teal"], COLORS["orange"], COLORS["red"], COLORS["blue"]][:len(names)]
        if log:
            log10_axis(ax, *log)
        plot_groups(ax, data, colors=cols, kind=kind, paired_ids=paired_ids, rotate=rotate,
                    point_size=point_size or t.marker_s * point_scale * (0.25 if kind == "violin" else 0.8),
                    jitter=jitter, **group_kw)
        if center:
            f = np.median if center == "median" else np.mean
            for i, (k, c) in enumerate(zip(names, cols)):
                ax.plot([i - 0.2, i + 0.2], [f(data[k])] * 2, color=c, lw=t.line_lw * 1.6,
                        zorder=5, solid_capstyle="butt")
        if log:
            log10_axis(ax, *log)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(list(yticks))
        if ylabel:
            ax.set_ylabel(ylabel)
        if brackets:
            pairs = (list(combinations(names, 2)) if brackets == "all" else
                     list(zip(names[:-1], names[1:])) if brackets == "adjacent" else brackets)
            p = compare_groups(data, pairs, test=test, adjust=adjust, paired_ids=paired_ids)
            add_brackets(ax, data, p, hide_ns=hide_ns, fontsize=t.font_size)
            r.data["p"] = p
    return Panel(draw, margins or {}, None, title, label)


def paired_panel(data: dict, ids: Sequence, *, colors=None, ylabel=None, test="wilcoxon",
                 brackets=True, center="median", margins=None, title=None, label=None, **kw):
    """One line per unit (lesion, patient) across conditions, median bar per condition."""
    pid = {k: list(ids) for k in data}
    return groups_panel(data, colors=colors, kind="none", ylabel=ylabel, paired_ids=pid,
                        center=center, brackets="adjacent" if brackets else None, test=test,
                        point_scale=2.0, margins=margins, title=title,
                        label=label, **kw)


# --------------------------------------------------------------------------------------
def volcano_panel(logfc, pvals, names=None, *, fc_thresh=1.0, p_thresh=0.05, padj=None,
                  label_top=6, label=None, label_names=None, up_color=None, down_color=None,
                  p_line=None, italic=True, margins=None, title=None):
    """Volcano plot. p_line: draw a horizontal guide at this P (e.g. Bonferroni)."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        r.data["sig"] = plot_volcano(
            ax, logfc, pvals, names, fc_thresh=fc_thresh, p_thresh=p_thresh, padj=padj,
            label_top=label_top, label=label_names, up_color=up_color or COLORS["red"],
            down_color=down_color or COLORS["teal"], s=t.marker_s * 0.45, s_sig=t.marker_s * 0.9,
            label_fontsize=t.small_size, label_italic=italic, guides=True)
        if p_line:
            ax.axhline(-np.log10(p_line), color="#888888", lw=0.7, ls=(0, (3, 3)), zorder=0)
    return Panel(draw, margins or {}, None, title, label)


# --------------------------------------------------------------------------------------
def correlation_panel(x, y, *, groups=None, group_colors=None, color="#333333",
                      method="spearman", xlabel=None, ylabel=None, legend=True,
                      stats_label="{r}", margins=None, title=None, label=None):
    """Scatter + least-squares line and CI; the coefficient goes in the legend."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        r.data["stats"] = plot_correlation(ax, x, y, color=color, groups=groups,
                                           group_colors=group_colors, method=method,
                                           s=t.marker_s, lw=t.line_lw,
                                           label=stats_label if legend else None)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        if legend:
            ax.legend(loc="upper left", handlelength=1.0, markerscale=1.2)
    return Panel(draw, margins or {}, None, title, label)


# --------------------------------------------------------------------------------------
def trajectory_panel(series: Sequence[dict], *, xlabel="", ylabel="", raw="points",
                     onset=True, onset_fmt="Difference at {x} yr", xlim=None, xticks=None,
                     break_zero=True, knots=None, grid=None, legend=False, point_size=None,
                     margins=None, title=None, label=None):
    """Fitted curves (spline, cluster-robust CI) over raw observations.
    series: [dict(x=, y=, ids=, color=, label=), ...]. With onset=True, the bracket marks
    where the first two series separate for good (computed from the fits)."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        fits = []
        for s in series:
            f = fit_trajectory(s["x"], s["y"], ids=s.get("ids"), grid=grid, knots=knots)
            plot_trajectory_scatter(ax, s["x"], s["y"], f, ids=s.get("ids"), color=s["color"],
                                    raw=raw, point_size=point_size or t.marker_s * 0.45,
                                    point_alpha=0.45, lw=t.line_lw * 1.15, label=s.get("label"))
            fits.append(f)
        if xlim:
            ax.set_xlim(*xlim)
        if xticks is not None:
            ax.set_xticks(list(xticks))
        auto_y_break(ax, np.concatenate([np.asarray(s["y"], float) for s in series]),
                     mark_size=t.font_size, force_zero=break_zero)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        r.data["fits"] = fits
        if onset and len(fits) >= 2:
            x0 = divergence_onset(fits[1], fits[0])
            r.data["onset"] = x0
            if x0 is not None:
                xe = min(fits[0].grid.max(), fits[1].grid.max())
                lab = onset_fmt.format(x=f"−{abs(round(x0))}" if x0 < 0 else round(x0))
                add_difference_bracket(ax, x0, xe, lab, y=1.03, fontsize=t.small_size)
        if legend:
            ax.legend(loc=legend if isinstance(legend, str) else "best")
    return Panel(draw, _merge(lambda t: {"top": t.top + (0.16 if onset else 0)}, margins),
                 None, title, label)


# --------------------------------------------------------------------------------------
def forest_panel(rows: Sequence[ForestRow], *, xlim=(0.25, 4), xticks=(0.25, 0.5, 1, 2, 4),
                 xticklabels=None, log=True, ref=1.0, label_w=1.25, n_w=0.45, est_w=0.95,
                 headers=("Subgroup", "No.", "Hazard Ratio (95% CI)"), better=None,
                 shade=True, est_fmt="{:.2f} ({:.2f}–{:.2f})", margins=None, title=None,
                 label=None):
    """Subgroup / multi-outcome forest plot. Suggested axes height: ~0.16 in per row.
    label_w, n_w, est_w: widths (in) of the label, N and estimate text columns."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        bw = r.box[2]
        nx = -0.06 / bw                     # N column: right edge just left of the axes
        cols = dict(label=r.axes_fraction_x(0.03), n=nx, n_ha="right",
                    est=1 + (est_w - 0.04) / bw, est_ha="right",
                    h_label=r.axes_fraction_x(0.03), h_n=nx, h_n_ha="right",
                    h_est=1 + (est_w - 0.04) / bw, h_est_ha="right")
        ya = t.font_size * 1.4 + 8
        plot_forest(ax, rows, xlim=xlim, xticks=xticks,
                    xticklabels=xticklabels or [f"{v:g}" for v in xticks], ref=ref, log=log,
                    cols=cols, headers=headers, better=better, shade=shade, est_fmt=est_fmt,
                    shade_extent=(r.axes_fraction_x(0.0), 1 + est_w / bw),
                    fontsize=t.small_size, ms=t.font_size * 0.6, capsize=2, lw=t.axes_lw,
                    arrow_offset=ya, better_offset=ya + t.small_size * 1.5,
                    indent_width=0.1 / bw)
    return Panel(draw, _merge(lambda t: {"left": label_w + n_w, "right": est_w,
                                         "bottom": t.bottom + (0.28 if better else 0),
                                         "top": t.top + 0.12}, margins), None, title, label)


# --------------------------------------------------------------------------------------
def grotta_panel(data, group_labels, *, legend=True,
                 legend_title="Score on the Modified Rankin Scale", label_w=0.8,
                 bar_height=0.55, colors=MRS_COLORS, margins=None, title=None, label=None):
    """Ordinal distribution bars (mRS etc.) with connectors and the stepped legend."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        plot_grotta(ax, data, group_labels, colors=colors, bar_height=bar_height,
                    fontsize=t.small_size, label_fontsize=t.label_size,
                    connector_kw=dict(lw=t.axes_lw))
        if legend:
            mrs_legend(ax, title=legend_title, colors=colors, bbox=(0.5, 1.0),
                       fontsize=t.small_size, title_fontsize=t.small_size,
                       columnspacing=0.9, handlelength=1.0)
    return Panel(draw, _merge(lambda t: {"left": label_w, "top": t.top + (0.34 if legend else 0)},
                              margins), None, title, label)


# --------------------------------------------------------------------------------------
def point_range_panel(categories, series: dict, *, colors=None, ylabel=None, zero_line=None,
                      ylim=None, yticks=None, ybreak=None, legend=None, offset=0.2,
                      margins=None, title=None, label=None):
    """Point estimates with CIs, one or several series per category."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        names = list(series)
        cols = colors or [COLORS["dark_blue"], COLORS["light_blue"]][:len(names)]
        plot_point_range(ax, categories, series, colors=cols, offset=offset,
                         ms=t.font_size * 0.75, capsize=t.font_size * 0.5, lw=t.line_lw * 0.9,
                         zero_line=zero_line, xpad=0.55)
        if ylim:
            ax.set_ylim(*ylim)
        if ybreak:
            add_y_break(ax, *ybreak, gap_frac=0.12, mark_size=t.font_size)
        if yticks is not None:
            ax.set_yticks(list(yticks))
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend(
                [Line2D([], [], ls="none", marker="o", ms=t.font_size * 0.7, color=c) for c in cols],
                names, loc=legend, handletextpad=0.2)
    return Panel(draw, margins or {}, None, title, label)


# --------------------------------------------------------------------------------------
def timecourse_panel(segments: Sequence[tuple], series: Sequence[dict], *, ylabel="",
                     ylim=None, yticks=None, log=None, ybreak=None, legend=None, gap=0.3,
                     margins=None, title=None, label=None):
    """Categorical time axis with Day / Week / Month brackets.

    segments: [("Day", ["0", "3-5"]), ("Week", ["1", "2", ...]), ...]
              Use ("", ["Day 0"]) for an unbracketed lead-in tick.
    series: dicts with kind="strip" (values=[arrays per timepoint]) or kind="line"
            (mean=[...], err=[...] optional, direction="both"|"up"), plus color, label,
            and for strips center="mean"|"median", err="sem"|None.
    """

    def draw(r):
        t = r.theme
        ax = r.axes()
        pos, spans = segment_positions([len(s[1]) for s in segments], gap=gap)
        labels = [l for _, ls in segments for l in ls]
        n = len(pos)
        for s in series:
            if s.get("kind", "strip") == "strip":
                plot_time_strip(ax, pos[:len(s["values"])], s["values"], color=s.get("color", COLORS["blue"]),
                                s=t.marker_s * 0.9, center=s.get("center", "mean"),
                                err=s.get("err", "sem"),
                                center_color=s.get("center_color", "black"),
                                center_lw=s.get("center_lw", t.line_lw))
            else:
                m = np.asarray(s["mean"], float)
                plot_time_line(ax, pos[:len(m)], m, s.get("err"), color=s.get("color", "black"),
                               marker=s.get("marker", "o"), ms=t.font_size * 0.5,
                               lw=t.line_lw * 0.85, capsize=2, direction=s.get("direction", "both"),
                               label=s.get("label"))
        if log:
            log10_axis(ax, *log)
        if ylim:
            ax.set_ylim(*ylim)
        if ybreak:
            add_y_break(ax, *ybreak[:2], gap_frac=0.1, mark_size=t.font_size,
                        upper_scale=ybreak[2] if len(ybreak) > 2 else 1.0)
        if yticks is not None:
            ax.set_yticks(list(yticks))
        segs = [(name, *sp) for (name, _), sp in zip(segments, spans) if name]
        segmented_time_axis(ax, pos, labels, segs, pad_x=0.5, fontsize=t.label_size,
                            line_offset=t.font_size * 1.55 + 2, text_offset=t.font_size * 1.55 + 3.5)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(loc=legend)
    return Panel(draw, margins or {}, None, title, label)


# --------------------------------------------------------------------------------------
def violin_panel(values, labels, *, median_color=None, ylabel="", xlabel="", points=False,
                 ylim=None, yticks=None, width=0.75, margins=None, title=None, label=None):
    """Unfilled violins with median and dotted quartiles."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        plot_violin_outline(ax, values, width=width, labels=labels, lw=t.axes_lw,
                            median_color=median_color or COLORS["red"], median_lw=t.line_lw * 1.2)
        if points:
            rng = np.random.default_rng(0)
            for i, v in enumerate(values):
                ax.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=t.marker_s * 0.6,
                           color="black", zorder=4)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(values) - 0.4)
        if ylim:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(list(yticks))
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    return Panel(draw, margins or {}, None, title, label)


def bar_points_panel(values, labels, *, colors=None, ylabel="", ylim=None, yticks=None,
                     margins=None, title=None, label=None):
    """Open bars at the mean with SEM, points on top."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        plot_bar_points(ax, np.arange(len(values)), values, colors=colors or COLORS["teal"],
                        s=t.marker_s * 1.2, bar_lw=t.axes_lw, err_lw=t.axes_lw)
        ax.set_xticks(range(len(values)), labels, fontweight="bold")
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(values) - 0.4)
        if ylim:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(list(yticks))
        ax.set_ylabel(ylabel)
    return Panel(draw, margins or {}, None, title, label)


# --------------------------------------------------------------------------------------
def flow_panel(x, y, *, gate=None, gate_label="auto", xlabel="", ylabel="", subtitle=None,
               margins=None, title=None, label=None):
    """Flow cytometry dot plot shaded by density, optional rectangle gate (x0, x1, y0, y1)."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        plot_flow(ax, x, y)
        if gate:
            x0, x1, y0, y1 = gate
            r.data["gate"] = add_gate(ax, x0, x1, y0, y1, gate_label, events=(x, y),
                                      label_xy=(np.sqrt(max(x0, 1) * x1), y0 / 2.5),
                                      fontsize=t.small_size)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=t.small_size)
        if r.box[2] < 1.1:          # narrow: leave 10^4 unlabelled so labels don't collide
            ax.xaxis.set_major_formatter(FixedFormatter(["0", "$10^3$", "", "$10^5$"]))
        if subtitle:
            ax.set_title(subtitle, fontsize=t.small_size, pad=4)
    return Panel(draw, margins or {}, 0.85, title, label)


# --------------------------------------------------------------------------------------
def legend_panel(items: Sequence[tuple], *, title=None, ncol=1, loc="center left",
                 margins=None, label=False):
    """A cell holding only a legend. items: (text, color, kind) with kind in
    'line', 'dot', 'patch', 'band'."""

    def draw(r):
        t = r.theme
        ax = r.axes()
        ax.axis("off")
        handles = []
        for text, color, kind in items:
            if kind == "line":
                handles.append(Line2D([], [], color=color, lw=t.line_lw * 1.3))
            elif kind == "dot":
                handles.append(Line2D([], [], ls="none", marker="o", ms=t.font_size * 0.7, color=color))
            elif kind == "band":
                handles.append((Patch(color=tint(color, 0.45), lw=0), Line2D([], [], color=color, lw=t.line_lw)))
            else:
                handles.append(Patch(color=color, lw=0))
        from matplotlib.legend_handler import HandlerTuple
        ax.legend(handles, [i[0] for i in items], loc=loc, ncol=ncol, title=title,
                  handler_map={tuple: HandlerTuple(ndivide=1)}, fontsize=t.label_size,
                  title_fontproperties=dict(weight="bold", size=t.label_size),
                  alignment="left", frameon=False)
    base = {"left!": 0.05, "right!": 0.05}
    return Panel(draw, {**base, **(margins or {})}, None, None, label)


def flowchart_panel(nodes, edges, *, margins=None, aspect=None, **kw):
    """CONSORT-style flow: FlowNode positions are fractions of the axes box, so the panel
    fills whatever cell it lands in; set aspect (height / width) to fix its shape."""
    common = _common(kw)
    m = _merge(lambda t: {"left!": 0.02, "right!": 0.02, "bottom!": 0.02}, margins)

    def draw(region):
        ax = region.axes()
        _, _, w, h = region.box
        plot_flowchart(ax, nodes, edges, w, h, **kw)
        return ax

    return Panel(draw, margins=m, aspect=aspect, no_axes_frame=True, **common)
