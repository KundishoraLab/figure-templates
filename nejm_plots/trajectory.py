"""Trajectories fitted from raw (often repeated) measurements, drawn over the data.

The NEJM AD figure shows modelled curves only. Here the curve is fitted from the points
with a restricted cubic spline, the band is a cluster-robust 95% CI (participants with
repeated visits count once), and the 'difference at' onset is computed, not typed in.

    fit_c = fit_trajectory(x_c, y_c, ids=id_c)
    fit_a = fit_trajectory(x_a, y_a, ids=id_a)
    plot_trajectory_scatter(ax, x_c, y_c, fit_c, ids=id_c, color=COLORS["teal"])
    plot_trajectory_scatter(ax, x_a, y_a, fit_a, ids=id_a, color=COLORS["red"])
    onset = divergence_onset(fit_a, fit_c)        # e.g. -11.4 (years before diagnosis)
    add_difference_bracket(ax, onset, 0, f"Difference at {onset:.0f} yr")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from matplotlib.colors import to_rgb


# --------------------------------------------------------------------------------------
# Restricted cubic spline (Harrell) and OLS with cluster-robust variance
# --------------------------------------------------------------------------------------
def rcs_basis(x, knots):
    """Restricted cubic spline basis: linear beyond the outer knots. Returns n x (k-1)."""
    x = np.asarray(x, float)
    t = np.asarray(knots, float)
    k = len(t)
    norm = (t[-1] - t[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        p = lambda a: np.clip(x - a, 0, None) ** 3
        cols.append((p(t[j]) - p(t[-2]) * (t[-1] - t[j]) / (t[-1] - t[-2])
                     + p(t[-1]) * (t[-2] - t[j]) / (t[-1] - t[-2])) / norm)
    return np.column_stack(cols)


def default_knots(x, k=4):
    q = {3: [10, 50, 90], 4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95]}[k]
    return np.percentile(x, q)


@dataclass
class TrajectoryFit:
    grid: np.ndarray
    y: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    se: np.ndarray
    beta: np.ndarray
    vcov: np.ndarray
    knots: np.ndarray
    df: int
    n_obs: int
    n_clusters: int

    def predict(self, x):
        X = np.column_stack([np.ones(len(np.atleast_1d(x))), rcs_basis(np.atleast_1d(x), self.knots)])
        y = X @ self.beta
        se = np.sqrt(np.einsum("ij,jk,ik->i", X, self.vcov, X))
        return y, se


def fit_trajectory(x, y, *, ids=None, grid=None, knots=None, n_knots=4, conf=0.95,
                   n_grid=200, extrapolate=False):
    """Spline fit of y on x with a pointwise CI.

    ids: participant IDs for repeated measures. With ids, the variance is
         cluster-robust (CR1 sandwich) and df = n_clusters - 1.
    grid: x values to evaluate. Defaults to the observed range.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ids = None if ids is None else np.asarray(ids)[ok]
    knots = default_knots(x, n_knots) if knots is None else np.asarray(knots, float)
    X = np.column_stack([np.ones(len(x)), rcs_basis(x, knots)])
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)

    if ids is None:
        V = bread * (e @ e) / (n - p)
        df, G = n - p, n
    else:
        groups = np.unique(ids)
        G = len(groups)
        meat = np.zeros((p, p))
        for g in groups:
            m = ids == g
            s = X[m].T @ e[m]
            meat += np.outer(s, s)
        c = G / (G - 1) * (n - 1) / (n - p)
        V = c * bread @ meat @ bread
        df = G - 1

    if grid is None:
        grid = np.linspace(x.min(), x.max(), n_grid)
    grid = np.asarray(grid, float)
    if not extrapolate:
        grid = grid[(grid >= x.min()) & (grid <= x.max())]
    Xg = np.column_stack([np.ones(len(grid)), rcs_basis(grid, knots)])
    yg = Xg @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, V, Xg))
    tq = stats.t.ppf(0.5 + conf / 2, df)
    return TrajectoryFit(grid, yg, yg - tq * se, yg + tq * se, se, beta, V, knots, df, n, G)


def divergence_onset(fit_a: TrajectoryFit, fit_b: TrajectoryFit, conf=0.95, grid=None):
    """Earliest x from which the two curves differ at every later x (pointwise CI of the
    difference excludes 0 through the end of the grid). Returns None if never."""
    lo = max(fit_a.grid.min(), fit_b.grid.min())
    hi = min(fit_a.grid.max(), fit_b.grid.max())
    grid = np.linspace(lo, hi, 400) if grid is None else np.asarray(grid, float)
    ya, sa = fit_a.predict(grid)
    yb, sb = fit_b.predict(grid)
    z = stats.norm.ppf(0.5 + conf / 2)
    sig = np.abs(ya - yb) > z * np.sqrt(sa**2 + sb**2)
    if not sig[-1]:
        return None
    not_sig = np.where(~sig)[0]
    return float(grid[0] if len(not_sig) == 0 else grid[not_sig[-1] + 1])


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def tint(c, f=0.55):
    """Blend a colour toward white by fraction f."""
    r, g, b = to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def plot_trajectory_scatter(ax, x, y, fit: TrajectoryFit | None = None, *, ids=None,
                            color, band_color=None, raw="points", point_size=7,
                            point_alpha=0.55, point_color=None, line_alpha=0.35,
                            line_width=0.5, band_alpha=0.55, lw=1.8, label=None,
                            rasterize_raw=True):
    """Raw observations behind a fitted curve and its CI band.

    raw: "points", "lines" (one thin line per participant across visits), "both", or None.
    fit: a TrajectoryFit; if None it is fitted here from x, y, ids.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if fit is None:
        fit = fit_trajectory(x, y, ids=ids)
    pc = point_color or tint(color, 0.35)
    if raw in ("lines", "both") and ids is not None:
        ids = np.asarray(ids)
        for g in np.unique(ids):
            m = ids == g
            if m.sum() > 1:
                o = np.argsort(x[m])
                ax.plot(x[m][o], y[m][o], color=pc, lw=line_width, alpha=line_alpha,
                        zorder=0.5, rasterized=rasterize_raw)
    if raw in ("points", "both"):
        ax.scatter(x, y, s=point_size, color=pc, alpha=point_alpha, lw=0, zorder=0.6,
                   rasterized=rasterize_raw)
    bc = band_color or tint(color, 0.45)
    ax.fill_between(fit.grid, fit.lo, fit.hi, color=bc, alpha=band_alpha, lw=0, zorder=1)
    ax.plot(fit.grid, fit.y, color=color, lw=lw, zorder=2, label=label,
            solid_capstyle="round")
    return fit
