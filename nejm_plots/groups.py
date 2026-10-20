"""Group comparisons: box / violin with sina-jittered points, paired lines, brackets.

    data = {"Control": v0, "Case": v1}                      # or long DataFrame + by=
    plot_groups(ax, data, colors=[COLORS["teal"], COLORS["red"]])
    p = compare_groups(data, [("Control", "Case")])          # {("Control","Case"): p}
    add_brackets(ax, data, p)                                # stars only, no 'ns'
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from .trajectory import tint


def _as_dict(data, value=None, by=None, order=None):
    if isinstance(data, dict):
        d = {k: np.asarray(v, float) for k, v in data.items()}
    else:  # long DataFrame
        d = {k: g[value].to_numpy(float) for k, g in data.groupby(by, sort=False)}
    if order:
        d = {k: d[k] for k in order}
    return d


def sina_offsets(values, width=0.35, rng=None, bw=None):
    """Horizontal jitter scaled by the local density (a 'sina' plot)."""
    rng = rng or np.random.default_rng(0)
    v = np.asarray(values, float)
    if len(v) < 3 or np.ptp(v) == 0:
        return rng.uniform(-width, width, len(v)) * 0.5
    dens = stats.gaussian_kde(v, bw_method=bw)(v)
    return rng.uniform(-1, 1, len(v)) * width * dens / dens.max()


def plot_groups(ax, data, *, value=None, by=None, order=None, colors=None, kind="box",
                points=True, jitter=0.32, point_size=12, point_alpha=0.75,
                box_width=0.55, paired_ids=None, pair_color="#B5B5B5", pair_lw=0.6,
                fill=True, rng=None, label_weight="bold", rotate=0, rasterize_points=False,
                box_face=None, box_edge=None, median_color=None, whis=1.5,
                point_face=None, point_edge=None, positions=None):
    """kind: "box", "violin", "bar" (mean +/- 95% CI), or "none" (points only).

    paired_ids: {group: ids} aligned with values; draws a grey line per id across groups
                and places points without jitter so the lines meet them.
    NEJM-style solid boxes: box_face=colors (or "full"), box_edge="black",
    median_color="black", whis=(0, 100) for min-max whiskers, point_face="white",
    point_edge="black".
    """
    d = _as_dict(data, value, by, order)
    names = list(d)
    xs = np.arange(len(names), dtype=float) if positions is None else np.asarray(positions, float)
    colors = colors or ["#555555"] * len(names)
    rng = rng or np.random.default_rng(0)
    offsets = {}

    for i, (k, c) in enumerate(zip(names, colors)):
        v = d[k][np.isfinite(d[k])]
        if kind == "box":
            ec = box_edge or c
            fc = (c if box_face == "full" else box_face[i] if isinstance(box_face, (list, tuple))
                  else box_face if box_face else (tint(c, 0.82) if fill else "none"))
            ax.boxplot([v], positions=[xs[i]], widths=box_width, showfliers=False,
                       patch_artist=True, zorder=2, whis=whis,
                       medianprops=dict(color=median_color or c, lw=1.4 if median_color else 1.8),
                       whiskerprops=dict(color=ec, lw=0.9),
                       capprops=dict(color=ec, lw=0.9),
                       boxprops=dict(edgecolor=ec, lw=1.0, facecolor=fc))
        elif kind == "violin":
            vp = ax.violinplot([v], positions=[xs[i]], widths=box_width * 1.4,
                               showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(tint(c, 0.75)); b.set_edgecolor(c); b.set_alpha(1); b.set_lw(0.8)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([xs[i]] * 2, [q1, q3], color=c, lw=2.2, zorder=3)
            ax.plot([xs[i] - 0.12, xs[i] + 0.12], [med] * 2, color=c, lw=2, zorder=3)
        elif kind == "bar":
            m = v.mean()
            ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(xs[i], m, width=box_width, color=tint(c, 0.7), edgecolor=c, lw=1, zorder=1)
            ax.errorbar(xs[i], m, yerr=ci, color=c, capsize=4, lw=1.2, zorder=3)
        if points:
            off = np.zeros(len(d[k])) if paired_ids is not None else \
                sina_offsets(d[k], jitter * (0.6 if kind == "box" else 1), rng)
            offsets[k] = off
            ax.scatter(xs[i] + off, d[k], s=point_size, color=point_face or c,
                       alpha=point_alpha, lw=0.6 if point_edge else 0, edgecolor=point_edge,
                       zorder=4, rasterized=rasterize_points)

    if paired_ids is not None:
        lookup = [dict(zip(paired_ids[k], d[k])) for k in names]
        for pid in set().union(*[set(paired_ids[k]) for k in names]):
            yy = [lk.get(pid, np.nan) for lk in lookup]
            ax.plot(xs, yy, color=pair_color, lw=pair_lw, zorder=1.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontweight=label_weight, rotation=rotate,
                       ha="right" if rotate else "center")
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    return xs


def compare_groups(data, pairs: Sequence[tuple], *, test="mannwhitney", value=None, by=None,
                   paired_ids=None, adjust="holm"):
    """P values for each pair. test: mannwhitney | ttest | welch | wilcoxon | paired_t.
    adjust: 'holm', 'bonferroni', 'bh', or None."""
    d = _as_dict(data, value, by)
    ps = []
    for a, b in pairs:
        x, y = d[a], d[b]
        if test in ("wilcoxon", "paired_t"):
            ia, ib = dict(zip(paired_ids[a], x)), dict(zip(paired_ids[b], y))
            common = [k for k in ia if k in ib]
            x, y = np.array([ia[k] for k in common]), np.array([ib[k] for k in common])
            p = (stats.wilcoxon(x, y) if test == "wilcoxon" else stats.ttest_rel(x, y)).pvalue
        elif test == "mannwhitney":
            p = stats.mannwhitneyu(x, y).pvalue
        else:
            p = stats.ttest_ind(x, y, equal_var=(test == "ttest")).pvalue
        ps.append(p)
    ps = adjust_p(np.array(ps), adjust)
    return dict(zip([tuple(p) for p in pairs], ps))


def adjust_p(p, method="holm"):
    p = np.asarray(p, float)
    m = len(p)
    if method is None or m == 1:
        return p
    o = np.argsort(p)
    if method == "bonferroni":
        return np.minimum(p * m, 1)
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    elif method == "bh":
        adj = np.minimum.accumulate((m / np.arange(1, m + 1) * p[o])[::-1])[::-1]
    else:
        raise ValueError(method)
    out = np.empty(m)
    out[o] = np.minimum(adj, 1)
    return out


def stars(p, thresholds=((0.001, "***"), (0.01, "**"), (0.05, "*"))):
    for t, s in thresholds:
        if p < t:
            return s
    return ""


def add_brackets(ax, data, pvals: dict, *, order=None, hide_ns=True, fmt="stars",
                 step=0.08, start=0.04, tick=0.015, lw=0.9, fontsize=None):
    """Brackets above groups. Heights stack in the order given. fmt: 'stars' or 'p'.
    Non-significant pairs are skipped when hide_ns=True (report them in the legend)."""
    d = _as_dict(data, order=order)
    names = list(d)
    top = max(np.nanmax(v) for v in d.values())
    lo, hi = ax.get_ylim()
    span = hi - lo
    level = 0
    for (a, b), p in pvals.items():
        txt = stars(p) if fmt == "stars" else (f"P={p:.2g}" if p >= 0.001 else "P<0.001")
        if hide_ns and p >= 0.05:
            continue
        i, j = names.index(a), names.index(b)
        y = top + span * (start + level * step)
        ax.plot([i, i, j, j], [y - tick * span, y, y, y - tick * span], color="black",
                lw=lw, clip_on=False)
        ax.text((i + j) / 2, y + (0 if fmt == "stars" else 0.01 * span), txt, ha="center",
                va="center_baseline" if fmt == "stars" else "bottom", fontsize=fontsize)
        level += 1
    if level:
        ax.set_ylim(lo, max(hi, top + span * (start + level * step + 0.02)))
