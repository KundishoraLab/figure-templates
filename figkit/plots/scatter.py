"""Scatter with a per-group least-squares fit — the "does the slope differ" panel.

`scatter_fit` is the panel for a per-sample quantity regressed on a per-sample
covariate: composition against age, signature score against dose, a rate against
depth. One line per group, drawn over that group's own data, with every point
visible underneath it.

Design decisions baked in, each one a failure mode that produced a wrong panel:

- **This function does not test whether two groups' slopes differ, and says so.**
  Two fits with overlapping confidence bands are routinely captioned "no
  difference" and two with a visible gap "a difference"; neither follows. The
  comparison is an interaction term the caller has to fit, and the returned frame
  deliberately has no p-value for it — only a per-group p against zero slope.
  Print the interaction p in the title if you have one.
- **Each line stops at its own group's x range.** Extending every fit across the
  shared axis makes groups appear to converge or diverge in a region where one of
  them has no data, which is the most common way a two-group regression panel
  overclaims.
- **The band is the confidence interval of the fitted mean, not a prediction
  interval.** They differ by a factor of several and are drawn identically. The
  return names it `ci` so a caption cannot quietly promote one to the other.
- **A group with fewer than three points gets its points and no line**, and NaN
  for every statistic. Two points fit perfectly, and `scipy.stats.linregress`
  reports that as p = 0 with zero standard error rather than declining, so the
  line would be the most confident-looking thing on the panel and the returned
  row would agree with it.
- **n goes in the legend label**, as in `violin`. A slope from eleven donors and
  a slope from forty-five look the same once drawn.
- **Points are on top of the line, not under it.** The fit is the summary; the
  points are the evidence, and the panel exists so a reader can see three
  subjects carrying an arm of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..palettes import categorical_palette
from ..style import despine
from ..theme import Theme, resolve

__all__ = ["scatter_fit"]

# Below this a line is an interpolation, not an estimate.
MIN_FOR_FIT = 3


def _fit(x: np.ndarray, y: np.ndarray, ci: float):
    """OLS of y on x with the confidence band of the fitted mean.

    Returns (stats, grid, lo, hi). scipy's `linregress` and statsmodels'
    `ols("y ~ x")` agree to machine precision for one predictor, so a caller can
    assert a drawn slope against a table written by either.
    """
    from scipy import stats

    n = len(x)
    grid = np.linspace(x.min(), x.max(), 100)
    if n < MIN_FOR_FIT:
        # `linregress` on two points returns p = 0.0 and stderr = 0.0 rather
        # than refusing, so a caller that printed the returned row would print
        # a perfect fit for the group this function declined to draw a line for.
        return ({"n": n, "slope": np.nan, "intercept": np.nan, "stderr": np.nan,
                 "p": np.nan, "r2": np.nan}, grid, None, None)
    r = stats.linregress(x, y)
    s = {"n": n, "slope": float(r.slope), "intercept": float(r.intercept),
         "stderr": float(r.stderr), "p": float(r.pvalue), "r2": float(r.rvalue) ** 2}
    resid = y - (r.intercept + r.slope * x)
    mse = float(resid @ resid) / (n - 2)
    sxx = float(((x - x.mean()) ** 2).sum())
    se = np.sqrt(mse * (1.0 / n + (grid - x.mean()) ** 2 / sxx))
    t = stats.t.ppf(0.5 + ci / 2, n - 2)
    yhat = r.intercept + r.slope * grid
    return s, grid, yhat - t * se, yhat + t * se


def scatter_fit(df, ax, x_col: str, y_col: str, group_col: str | None = None,
                order=None, palette=None, ci: float = 0.95,
                point_size: float = 24, alpha: float = 0.9,
                show_band: bool = True, show_n: bool = True,
                line_width: float = 1.4, x_label: str | None = None,
                y_label: str | None = None, legend: bool = True,
                legend_loc: str = "best", fontsize: float = 8,
                theme: Theme | None = None) -> pd.DataFrame:
    """Scatter of `y_col` on `x_col`, with one OLS fit per level of `group_col`.

    Returns one row per group: n, slope, intercept, stderr, p, r2. The p is the
    two-sided test of that group's slope against zero and nothing else — see the
    module docstring on why no between-group p is returned.

    `group_col=None` fits the whole frame as one group.
    """
    t = resolve(theme)
    d = df[[c for c in {x_col, y_col, group_col} if c]].dropna()
    if not len(d):
        raise ValueError(f"scatter_fit: no rows with both {x_col} and {y_col}")

    if group_col is None:
        groups = [("", d)]
        colors = {"": t.up}
    else:
        levels = list(order) if order is not None else list(
            pd.unique(d[group_col].astype(str)))
        d = d[d[group_col].astype(str).isin(levels)]
        groups = [(g, d[d[group_col].astype(str) == g]) for g in levels]
        colors = palette or categorical_palette(levels, colors=t.categorical)

    rows = []
    for g, sub in groups:
        xv = sub[x_col].to_numpy(dtype=float)
        yv = sub[y_col].to_numpy(dtype=float)
        c = colors[g] if group_col is not None else colors[""]
        s, grid, lo, hi = _fit(xv, yv, ci)
        if len(xv) >= MIN_FOR_FIT:
            ax.plot(grid, s["intercept"] + s["slope"] * grid, color=c,
                    linewidth=line_width, zorder=2)
            if show_band and lo is not None:
                ax.fill_between(grid, lo, hi, color=c, alpha=0.15, linewidth=0,
                                zorder=1)
        label = f"{g} (n = {len(xv)})" if show_n and g else (g or None)
        ax.scatter(xv, yv, s=point_size, color=c, edgecolors="white",
                   linewidths=0.4, alpha=alpha, zorder=3, label=label)
        rows.append({"group": g, **s})

    ax.set_xlabel(x_label if x_label is not None else x_col)
    ax.set_ylabel(y_label if y_label is not None else y_col)
    if legend and group_col is not None:
        ax.legend(loc=legend_loc, frameon=False, fontsize=fontsize)
    despine(ax)
    return pd.DataFrame(rows).set_index("group")
