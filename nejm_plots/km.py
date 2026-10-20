"""Kaplan-Meier estimation and NEJM-style KM panels.

Typical use with your own data:

    from nejm_plots import use_nejm_style, KMArm, plot_km, add_at_risk_table, COLORS
    use_nejm_style(10)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    arms = [KMArm("Treated", df_t.time, df_t.event, COLORS["orange"]),
            KMArm("Control", df_c.time, df_c.event, COLORS["blue"])]
    fits = plot_km(ax, arms, xlim=(0, 14), xticks=range(15), xlabel="Age (yr)")
    add_at_risk_table(ax, fits, at=range(15))
    add_km_pvalue(ax, fits)            # log-rank, NEJM formatting
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats
from matplotlib.transforms import blended_transform_factory, ScaledTranslation


# --------------------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------------------
@dataclass
class KMFit:
    t: np.ndarray          # step times, starting at 0, ending at last observation
    s: np.ndarray          # survival (0..1) on [t_i, t_{i+1})
    lo: np.ndarray         # lower CI
    hi: np.ndarray         # upper CI
    censor_t: np.ndarray   # times of censored observations
    censor_s: np.ndarray   # survival at those times (for tick marks)
    time: np.ndarray       # raw data
    event: np.ndarray
    label: str = ""

    def at_risk(self, at):
        return np.array([(self.time >= a).sum() for a in at])

    def cum_events(self, at):
        return np.array([((self.time <= a) & (self.event == 1)).sum() for a in at])

    def survival_at(self, at):
        idx = np.searchsorted(self.t, np.asarray(at, float), side="right") - 1
        return self.s[np.clip(idx, 0, None)]


def kaplan_meier(time, event, conf=0.95, label="", conf_type="loglog") -> KMFit:
    """Product-limit estimate with Greenwood variance; confidence limits on the log(-log)
    scale (default) or the log scale (conf_type="log", survfit's default)."""
    time = np.asarray(time, float)
    event = np.asarray(event).astype(int)
    z = stats.norm.ppf(0.5 + conf / 2)

    ts, ss, los, his = [0.0], [1.0], [1.0], [1.0]
    s, gw = 1.0, 0.0
    for t in np.unique(time[event == 1]):
        n = (time >= t).sum()
        d = ((time == t) & (event == 1)).sum()
        s *= 1 - d / n
        gw = gw + d / (n * (n - d)) if n > d else np.inf
        if 0 < s < 1 and np.isfinite(gw):
            if conf_type == "log":
                lo, hi = s * np.exp(-z * np.sqrt(gw)), min(1.0, s * np.exp(z * np.sqrt(gw)))
            else:
                se = np.sqrt(gw) / abs(np.log(s))
                lo, hi = s ** np.exp(z * se), s ** np.exp(-z * se)
        elif s <= 0:
            lo = hi = 0.0
        else:
            lo = hi = s
        ts.append(t); ss.append(s); los.append(lo); his.append(hi)

    tmax = time.max()
    if tmax > ts[-1]:
        ts.append(tmax); ss.append(ss[-1]); los.append(los[-1]); his.append(his[-1])

    fit = KMFit(np.array(ts), np.array(ss), np.array(los), np.array(his),
                np.array([]), np.array([]), time, event, label)
    ct = np.sort(time[event == 0])
    fit.censor_t, fit.censor_s = ct, fit.survival_at(ct)
    return fit


def logrank_test(fits: Sequence[KMFit]):
    """k-group log-rank test. Returns (chi2, p)."""
    times = np.concatenate([f.time for f in fits])
    events = np.concatenate([f.event for f in fits])
    k = len(fits)
    U = np.zeros(k)
    V = np.zeros((k, k))
    for t in np.unique(times[events == 1]):
        n_g = np.array([(f.time >= t).sum() for f in fits], float)
        d_g = np.array([((f.time == t) & (f.event == 1)).sum() for f in fits], float)
        n, d = n_g.sum(), d_g.sum()
        if n < 2:
            continue
        U += d_g - d * n_g / n
        c = d * (n - d) / (n - 1) / n**2
        V += c * (np.diag(n_g) * n - np.outer(n_g, n_g))
    chi2 = float(U[:-1] @ np.linalg.pinv(V[:-1, :-1]) @ U[:-1])
    return chi2, float(stats.chi2.sf(chi2, k - 1))


def median_survival(fit: KMFit):
    """Median and its 95% CI as survfit reports them: the first time the curve reaches 0.5
    (if it sits exactly at 0.5, the midpoint of that flat step); the bounds are where the
    lower and upper confidence curves reach 0.5; nan when a curve never does."""
    def first_at_or_below(curve):
        idx = np.where(curve <= 0.5 + 1e-12)[0]
        if not len(idx):
            return float("nan")
        i = idx[0]
        if abs(curve[i] - 0.5) < 1e-12:                   # flat at 0.5: midpoint to the next drop
            j = np.where(curve[i + 1:] < 0.5 - 1e-12)[0]
            if len(j):
                return float((fit.t[i] + fit.t[i + 1 + j[0]]) / 2)
        return float(fit.t[i])
    return first_at_or_below(fit.s), first_at_or_below(fit.lo), first_at_or_below(fit.hi)


def format_p(p):
    """NEJM convention: P<0.001, three decimals below 0.01, otherwise two."""
    if p < 0.001:
        return "P<0.001"
    if p < 0.01:
        return f"P={p:.3f}"
    return f"P={p:.2f}"


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
@dataclass
class KMArm:
    label: str
    time: Sequence[float]
    event: Sequence[int]
    color: str
    ci: bool = True
    ci_alpha: float = 0.22
    lw: float = 1.7
    censor_marker: str = "|"
    censor_size: float = 7
    censor_mew: float = 1.0
    zorder: float = 3
    extra: dict = field(default_factory=dict)


def _truncate(f: KMFit, xmax):
    """Copy of the fit with the curve cut at xmax (raw data kept for at-risk counts)."""
    if f.t[-1] <= xmax:
        return f
    k = np.searchsorted(f.t, xmax, side="right")
    import copy
    g = copy.copy(f)
    g.t = np.r_[f.t[:k], xmax]
    g.s, g.lo, g.hi = (np.r_[a[:k], a[k - 1]] for a in (f.s, f.lo, f.hi))
    m = f.censor_t <= xmax
    g.censor_t, g.censor_s = f.censor_t[m], f.censor_s[m]
    return g


def plot_km(ax, arms: Sequence[KMArm], *, xlim, xticks=None, ylim=(0, 100), yticks=None,
            percent=True, xlabel="Time", ylabel="Percentage of Patients", conf=0.95,
            fits: Sequence[KMFit] | None = None, cumulative_incidence=False):
    """Draw one KM curve per arm (step line, CI band, censor ticks). Returns the KMFit list.

    Pass precomputed `fits` to skip estimation (e.g. curves from another package).
    cumulative_incidence=True plots 1 - S(t) rising from 0 (incident-disease style).
    This is 1 - KM, which overstates incidence when competing risks (e.g. death) matter.
    """
    scale = 100 if percent else 1
    ci_mode = cumulative_incidence

    def tf(v):
        return (1 - v) * scale if ci_mode else v * scale
    out = []
    for i, arm in enumerate(arms):
        f = fits[i] if fits is not None else kaplan_meier(arm.time, arm.event, conf, arm.label)
        f.label = arm.label
        f = _truncate(f, xlim[1])
        if arm.ci:
            ax.fill_between(f.t, tf(f.lo), tf(f.hi), step="post", color=arm.color,
                            alpha=arm.ci_alpha, lw=0, zorder=arm.zorder - 2)
        ax.step(f.t, tf(f.s), where="post", color=arm.color, lw=arm.lw, zorder=arm.zorder,
                clip_on=False, label=arm.label, **arm.extra)
        if len(f.censor_t):
            ax.plot(f.censor_t, tf(f.censor_s), ls="none", marker=arm.censor_marker,
                    ms=arm.censor_size, mew=arm.censor_mew, color=arm.color,
                    zorder=arm.zorder + 1, clip_on=False)
        out.append(f)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    ax.set_yticks(list(yticks) if yticks is not None else np.arange(ylim[0], ylim[1] + 1, 20))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return out


def add_at_risk_table(ax, fits: Sequence[KMFit], at, *, labels=None, cumulative_events=False,
                      title="No. at Risk", subtitle="", title_offset=34, row_height=15,
                      label_x=-0.28, fontsize=None, title_fontsize=None, x_positions=None):
    """Numbers at risk under the x axis, aligned to the `at` positions (data units).

    Offsets are in points below the x axis so the table follows the axes wherever it is.
    label_x is the left edge of the row labels in axes-fraction units.
    cumulative_events=True prints 'n (events)' as in the NEJM vutrisiran figure.
    x_positions: optional display positions (data units) if columns need nudging apart.
    """
    fig = ax.figure
    labels = labels or [f.label for f in fits]
    fs = fontsize
    tr = blended_transform_factory(ax.transData, ax.transAxes)

    def off(pts):
        return ScaledTranslation(0, -pts / 72, fig.dpi_scale_trans)

    t = ax.text(label_x, 0, title, transform=ax.transAxes + off(title_offset), ha="left",
                va="center", fontweight="bold", fontsize=title_fontsize or fs, clip_on=False)
    if subtitle:
        ax.annotate(" " + subtitle, xy=(1, 0.5), xycoords=t, va="center", ha="left",
                    fontsize=title_fontsize or fs, fontweight="normal")
    for i, (f, lab) in enumerate(zip(fits, labels)):
        y = title_offset + (i + 1) * row_height
        ax.text(label_x, 0, lab, transform=ax.transAxes + off(y), ha="left", va="center",
                fontsize=fs, clip_on=False)
        nr, ce = f.at_risk(at), f.cum_events(at)
        for a, n_, e_ in zip(x_positions if x_positions is not None else at, nr, ce):
            s = f"{n_} ({e_})" if cumulative_events else f"{n_}"
            ax.text(a, 0, s, transform=tr + off(y), ha="center", va="center", fontsize=fs,
                    clip_on=False)


def add_km_pvalue(ax, fits=None, text=None, xy=(0.03, 0.05), **kw):
    """Write a log-rank P value (or any literal `text`) inside the panel."""
    if text is None:
        text = format_p(logrank_test(fits)[1])
    kw.setdefault("fontsize", None)
    ax.text(*xy, text, transform=ax.transAxes, ha="left", va="bottom", **kw)
    return text


def add_km_annotation(ax, lines, xy=(0.04, 0.06), linespacing=1.45, **kw):
    """Multi-line text block inside the panel (e.g. hazard ratios)."""
    ax.text(*xy, "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom",
            linespacing=linespacing, **kw)


def label_curves(ax, positions: dict, **kw):
    """Direct curve labels instead of a legend: {"Vutrisiran": (x, y), ...} in data units."""
    for lab, (x, y) in positions.items():
        ax.text(x, y, lab, ha="left", va="center", **kw)
