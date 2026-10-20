"""Correlation scatter, volcano plot, ROC curves."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

from .trajectory import tint


# --------------------------------------------------------------------------------------
# Correlation scatter with OLS line and CI band
# --------------------------------------------------------------------------------------
def plot_correlation(ax, x, y, *, color="#333333", groups=None, group_colors=None,
                     method="spearman", line=True, conf=0.95, s=14, alpha=0.7,
                     band_alpha=0.35, lw=1.6, label=None, rasterize=False):
    """Scatter + least-squares line with CI band. Returns dict(r, p, n, slope, intercept).

    groups/group_colors colour the points only; the fit is pooled. Fit groups separately
    by calling this once per group. The statistics are returned, not printed, so you can
    put them in the legend (label="{r}") or the figure caption.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if groups is not None:
        groups = np.asarray(groups)[ok]
        for g, c in group_colors.items():
            m = groups == g
            ax.scatter(x[m], y[m], s=s, color=c, alpha=alpha, lw=0, zorder=2, label=g,
                       rasterized=rasterize)
    else:
        ax.scatter(x, y, s=s, color=tint(color, 0.25), alpha=alpha, lw=0, zorder=2,
                   rasterized=rasterize)

    r, p = (stats.spearmanr if method == "spearman" else stats.pearsonr)(x, y)
    res = stats.linregress(x, y)
    out = dict(r=float(r), p=float(p), n=len(x), slope=res.slope, intercept=res.intercept)
    if line:
        n = len(x)
        g = np.linspace(x.min(), x.max(), 100)
        yhat = res.intercept + res.slope * g
        resid = y - (res.intercept + res.slope * x)
        s_err = np.sqrt(resid @ resid / (n - 2))
        se = s_err * np.sqrt(1 / n + (g - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
        tq = stats.t.ppf(0.5 + conf / 2, n - 2)
        ax.fill_between(g, yhat - tq * se, yhat + tq * se, color=tint(color, 0.5),
                        alpha=band_alpha, lw=0, zorder=1)
        lab = None
        if label:
            sym = "ρ" if method == "spearman" else "r"
            ptxt = "P<0.001" if p < 0.001 else f"P={p:.2g}"
            lab = label.format(r=f"{sym}={r:.2f}, {ptxt}", n=n)
        ax.plot(g, yhat, color=color, lw=lw, zorder=3, label=lab)
    return out


# --------------------------------------------------------------------------------------
# Volcano
# --------------------------------------------------------------------------------------
def plot_volcano(ax, logfc, pvals, names=None, *, fc_thresh=1.0, p_thresh=0.05,
                 padj=None, up_color="#C0392B", down_color="#2E6DA4", ns_color="#C9C9C9",
                 s=8, s_sig=12, alpha=0.8, label_top=10, label=None, label_fontsize=None,
                 label_italic=False, guides=True, rasterize=True, min_gap_pt=11):
    """Volcano plot. Significance uses padj if given, else pvals.

    label_top: number of top hits per side to label (ranked by -log10 p * |logFC|),
    or pass `label` = list of names to label explicitly. Labels are stacked on the
    outside of each wing with leader lines so they do not overlap.
    Returns a boolean mask of significant features.
    """
    logfc = np.asarray(logfc, float)
    pvals = np.asarray(pvals, float)
    sig_p = np.asarray(padj, float) if padj is not None else pvals
    y = -np.log10(np.clip(pvals, 1e-300, 1))
    up = (sig_p < p_thresh) & (logfc >= fc_thresh)
    dn = (sig_p < p_thresh) & (logfc <= -fc_thresh)
    ns = ~(up | dn)
    ax.scatter(logfc[ns], y[ns], s=s, color=ns_color, lw=0, zorder=1, rasterized=rasterize)
    ax.scatter(logfc[up], y[up], s=s_sig, color=up_color, alpha=alpha, lw=0, zorder=2,
               rasterized=rasterize)
    ax.scatter(logfc[dn], y[dn], s=s_sig, color=down_color, alpha=alpha, lw=0, zorder=2,
               rasterized=rasterize)
    if guides:
        kw = dict(color="#888888", lw=0.7, ls=(0, (3, 3)), zorder=0)
        ax.axvline(fc_thresh, **kw); ax.axvline(-fc_thresh, **kw)
        if padj is None:
            ax.axhline(-np.log10(p_thresh), **kw)
    ax.set_xlabel(r"log$_2$ Fold Change")
    ax.set_ylabel(r"$-$log$_{10}$ P Value")
    lim = np.nanmax(np.abs(logfc)) * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, np.nanmax(y) * 1.08)

    if names is not None:
        names = np.asarray(names)
        score = y * np.abs(logfc)
        if label is not None:
            pick_up = np.where(np.isin(names, label) & (logfc > 0))[0]
            pick_dn = np.where(np.isin(names, label) & (logfc <= 0))[0]
        else:
            pick_up = np.where(up)[0][np.argsort(-score[up])][:label_top]
            pick_dn = np.where(dn)[0][np.argsort(-score[dn])][:label_top]
        style = "italic" if label_italic else "normal"
        for idx, side in ((pick_up, 1), (pick_dn, -1)):
            _stack_labels(ax, logfc[idx], y[idx], names[idx], side, min_gap_pt,
                          label_fontsize, style)
    return up | dn


def _stack_labels(ax, xs, ys, labels, side, min_gap_pt, fontsize, style):
    """Place labels beside points, pushed apart vertically, with leader lines."""
    if len(xs) == 0:
        return
    fig = ax.figure
    fig.canvas.draw()  # finalise limits so data->display is right
    T = ax.transData
    disp = T.transform(np.column_stack([xs, ys]))
    o = np.argsort(-disp[:, 1])
    gap = min_gap_pt * fig.dpi / 72
    ly = np.minimum(disp[o, 1].copy(), ax.bbox.y1 - gap * 0.6)
    for i in range(1, len(ly)):
        ly[i] = min(ly[i], ly[i - 1] - gap)
    # shift the column up if it ran below the axis
    ax_bottom = ax.bbox.y0 + gap
    if ly[-1] < ax_bottom:
        ly += ax_bottom - ly[-1]
        for i in range(len(ly) - 2, -1, -1):
            ly[i] = max(ly[i], ly[i + 1] + gap)
    pt = fig.dpi / 72
    # If the wing reaches the axis edge, stack the labels in a column on the inner side
    edge = ax.bbox.x1 if side > 0 else ax.bbox.x0
    if np.any(np.abs(disp[:, 0] - edge) < 70 * pt):
        side = -side
        col = (disp[:, 0].min() if side < 0 else disp[:, 0].max()) + 14 * pt * side
        lx = np.full(len(o), col)
    else:
        lx = disp[o, 0] + 14 * pt * side
    inv = T.inverted()
    for k, i in enumerate(o):
        tx, ty = inv.transform((lx[k], ly[k]))
        ax.annotate(labels[i], xy=(xs[i], ys[i]), xytext=(tx, ty), textcoords="data",
                    ha="left" if side > 0 else "right", va="center", fontsize=fontsize,
                    fontstyle=style, zorder=5,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666",
                                    shrinkA=1, shrinkB=2))


# --------------------------------------------------------------------------------------
# ROC
# --------------------------------------------------------------------------------------
def roc_curve(y_true, score):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, float)
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    tpr = np.r_[0, tps / max(y.sum(), 1)]
    fpr = np.r_[0, fps / max((1 - y).sum(), 1)]
    return fpr, tpr


def auc_score(y_true, score):
    f, t = roc_curve(y_true, score)
    return float(np.trapezoid(t, f))


def plot_roc(ax, models: dict, y_true=None, *, colors=None, n_boot=0, conf=0.95, lw=1.8,
             ci_band=False, rng=None, legend=True, legend_kw=None):
    """models: {"Name": scores} with a shared y_true, or {"Name": (y_true, scores)}.
    AUC (and bootstrap CI if n_boot>0) goes in the legend. Returns {name: (auc, lo, hi)}.
    ci_band=True shades the bootstrap pointwise TPR band."""
    rng = rng or np.random.default_rng(0)
    colors = colors or [None] * len(models)
    out = {}
    grid = np.linspace(0, 1, 101)
    for (name, val), c in zip(models.items(), colors):
        yt, sc = (val if isinstance(val, tuple) else (y_true, val))
        yt, sc = np.asarray(yt).astype(int), np.asarray(sc, float)
        f, t = roc_curve(yt, sc)
        a = float(np.trapezoid(t, f))
        lo = hi = np.nan
        if n_boot:
            aucs, tprs = [], []
            n = len(yt)
            for _ in range(n_boot):
                i = rng.integers(0, n, n)
                if yt[i].min() == yt[i].max():
                    continue
                fb, tb = roc_curve(yt[i], sc[i])
                aucs.append(np.trapezoid(tb, fb))
                tprs.append(np.interp(grid, fb, tb))
            lo, hi = np.percentile(aucs, [50 * (1 - conf), 100 - 50 * (1 - conf)])
            if ci_band:
                tl, th = np.percentile(tprs, [2.5, 97.5], axis=0)
                ax.fill_between(grid, tl, th, color=c, alpha=0.15, lw=0)
        lab = f"{name} (AUC {a:.2f}" + (f", {lo:.2f}–{hi:.2f})" if n_boot else ")")
        ax.plot(f, t, color=c, lw=lw, label=lab, drawstyle="steps-post")
        out[name] = (a, lo, hi)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    if legend:
        kw = dict(loc="lower right", handlelength=1.2, fontsize=None)
        kw.update(legend_kw or {})
        ax.legend(**kw)
    return out
