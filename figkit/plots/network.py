"""Chord diagrams and set-overlap plots.

Both depend on optional third-party packages (pycirclize, upsetplot). They
raise a clear install hint rather than failing deep inside the library.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..theme import Theme, resolve


def _require(mod: str, pkg: str):
    try:
        return __import__(mod)
    except ImportError as e:
        raise ImportError(
            f"{pkg} is required for this plot: pip install {pkg}") from e


def chord(matrix, fig_path, cmap=None, title: str = "", figsize=(5, 5),
          keep_top_frac: float = 0.0, dpi: int = 600, formats=("png", "svg"),
          theme: Theme | None = None):
    """Chord diagram from a square sender x receiver matrix (unsigned).

    Writes to `fig_path` (extension-less) because pycirclize builds its own
    figure. For signed deltas use `chord_signed` — taking abs() of a signed
    matrix here would silently merge increases and decreases into one band.

    keep_top_frac : keep only the strongest fraction of links (0.1 = top 10%).
        A dense chord is unreadable, but thresholding hides data — say the
        threshold in the caption.
    """
    _require("pycirclize", "pycirclize")
    from pycirclize import Circos
    import matplotlib.pyplot as plt

    t = resolve(theme)
    cmap = cmap if cmap is not None else t.heatmap_cmap
    m = pd.DataFrame(matrix)
    arr = m.fillna(0).abs().to_numpy()
    if keep_top_frac > 0:
        nz = arr[arr > 0]
        if len(nz):
            thresh = float(np.quantile(nz, 1 - keep_top_frac))
            n_before = int((arr > 0).sum())
            arr = np.where(arr >= thresh, arr, 0)
            print(f"  [chord] kept {int((arr > 0).sum())}/{n_before} links "
                  f"(top {keep_top_frac:.0%})")

    sectors = list(m.index)
    circos = Circos.chord_diagram(
        pd.DataFrame(arr, index=sectors, columns=sectors),
        space=4, r_lim=(95, 100),
        cmap=cmap.name if hasattr(cmap, "name") else "tab20")
    fig = circos.plotfig(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=12)
    return _save_raw(fig, fig_path, dpi, formats)


def chord_signed(matrix, fig_path, up_color=None, down_color=None,
                 title: str = "", figsize=(5, 5), top_n_per_sign: int = 25,
                 dpi: int = 600, formats=("png", "svg"),
                 theme: Theme | None = None):
    """Chord diagram with separate bands for positive and negative deltas.

    Takes top_n from *each* sign, so a panel where one direction dominates
    still shows the other. Link opacity scales with |delta|.
    """
    _require("pycirclize", "pycirclize")
    from pycirclize import Circos
    import matplotlib.patches as mpatches

    t = resolve(theme)
    up_c, dn_c = up_color or t.up, down_color or t.down
    m = pd.DataFrame(matrix)
    sectors = list(m.index)

    pairs = [(s, r, float(m.loc[s, r])) for s in sectors for r in sectors
             if np.isfinite(m.loc[s, r]) and m.loc[s, r] != 0]
    if not pairs:
        raise ValueError("chord_signed: matrix has no nonzero links")
    d = pd.DataFrame(pairs, columns=["sender", "receiver", "delta"])
    pos = d[d["delta"] > 0].nlargest(top_n_per_sign, "delta")
    neg = d[d["delta"] < 0].nsmallest(top_n_per_sign, "delta")

    circos = Circos({s: 1 for s in sectors}, space=3)
    for sector in circos.sectors:
        sector.add_track((95, 100)).axis(fc="#dddddd", ec="black", lw=0.5)
        sector.text(sector.name, r=108, size=9, orientation="vertical")

    max_abs = max(float(pos["delta"].max()) if len(pos) else 1.0,
                  abs(float(neg["delta"].min())) if len(neg) else 1.0)

    def _alpha(v):
        return float(np.clip(0.35 + 0.55 * abs(v) / max(max_abs, 1e-9), 0.35, 0.9))

    for frame, color in ((pos, up_c), (neg, dn_c)):
        for _, r in frame.iterrows():
            circos.link((r["sender"], 0.1, 0.9), (r["receiver"], 0.1, 0.9),
                        color=color, alpha=_alpha(r["delta"]))

    fig = circos.plotfig(figsize=figsize)
    fig.legend(handles=[mpatches.Patch(color=up_c, label=f"increased (top {len(pos)})"),
                        mpatches.Patch(color=dn_c, label=f"decreased (top {len(neg)})")],
               loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False,
               fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    return _save_raw(fig, fig_path, dpi, formats)


def upset(set_dict, fig_path, min_subset_size: int = 1, figsize=(6, 4),
          dpi: int = 600, formats=("png", "svg"), show_counts: bool = True):
    """UpSet plot of intersections among named sets.

    Use instead of a Venn beyond ~3 sets — a 4+ way Venn is not readable and
    cannot show every intersection at true relative size.
    """
    _require("upsetplot", "upsetplot")
    from upsetplot import UpSet, from_contents
    import matplotlib.pyplot as plt

    contents = from_contents({k: set(v) for k, v in set_dict.items()})
    fig = plt.figure(figsize=figsize)
    UpSet(contents, subset_size="count", min_subset_size=min_subset_size,
          show_counts=show_counts, sort_by="cardinality").plot(fig=fig)
    return _save_raw(fig, fig_path, dpi, formats)


def _save_raw(fig, fig_path, dpi: int, formats=("png", "svg")) -> list[Path]:
    """These plots own their figure, so they write it themselves.

    Mirrors save_panel's (dpi, formats) contract — `fig_path` is
    extension-less and each format is appended.
    """
    import matplotlib.pyplot as plt
    p = Path(fig_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        q = p.with_suffix(f".{ext}")
        fig.savefig(q, dpi=dpi, bbox_inches="tight")
        out.append(q)
    plt.close(fig)
    return out
