"""Volcano plot for differential-expression results."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..theme import Theme, resolve
from ..style import despine

# Genes that crowd the top of a volcano without carrying interpretable
# biology. Opt-in via `exclude_patterns=NOISE_GENE_PATTERNS`; the default is
# to plot everything, because "which genes are noise" is a per-assay call and
# silently dropping rows from someone else's DE table is the wrong default.
NOISE_GENE_PATTERNS = (
    r"^MT-",                # mitochondrial
    r"^MTRNR",              # mitochondrial rRNA-like
    r"^A[CLPF]\d+\.\d+$",   # unannotated Ensembl clone IDs (AL591895.1)
    r"^LINC\d+$",           # lncRNA without a symbol
    r"^MIR\d+",             # microRNA
    r"^RNU\d+",             # snRNA
    r"^RP[LS]\d+[A-Z]?$",   # ribosomal proteins
    r"^\d+$",               # bare numeric IDs
    r"-ENSG\d+$",           # symbol-Ensembl fusions (MALAT1-ENSG...)
    r"-AS\d+$",             # antisense transcripts
)

# Probe classes engineered into targeted panels (CosMx, Xenium, ...) as
# controls. These are not genes and must never surface as hits.
CONTROL_PROBE_PATTERNS = (
    r"SystemControl", r"Negative", r"FalseCode",
)


def _compile(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def filter_labels(names, patterns=NOISE_GENE_PATTERNS) -> np.ndarray:
    """Boolean keep-mask over `names` for rows matching none of `patterns`."""
    rx = _compile(patterns)
    return np.array([not any(r.search(str(n)) for r in rx) for n in names])


def volcano(df, ax, x_col: str = "log2fc", y_col: str = "pvalue",
            label_col: str = "gene",
            lfc_thresh: float = 0.5, p_thresh: float = 0.05,
            top_n_labels: int = 14, label_genes=None,
            label_all_top: bool = False,
            point_size: float = 2.0, point_alpha: float = 0.85,
            lfc_cap: float | None = None,
            exclude_patterns=None,
            star_set=None,
            jitter: bool = True,
            theme: Theme | None = None,
            up_color: str | None = None, down_color: str | None = None,
            x_label: str | None = None, y_label: str | None = None):
    """Volcano: x = effect size, y = -log10(p), colored by direction.

    Design decisions worth keeping (each fixes a real failure mode):

    * **p-value floor** is the smallest observed nonzero p x 0.5, not a fixed
      1e-300. A fixed floor stacks every p=0 gene into one vertical spike at
      an arbitrary height; deriving it from the data keeps the spike just
      above the real points.
    * **Tie jitter** (`jitter=True`) breaks exact duplicate stats. Tools like
      DESeq2 return identical statistics for groups of low-count genes, so
      without jitter dozens of genes hide under a single dot and the plot
      understates its own density. Deterministic (fixed seed) so reruns are
      byte-identical.
    * **Symmetric x-axis**, so "further right" and "further left" are the
      same visual distance and the eye can compare directions honestly.
    * **Labels placed last**, after limits are final, so adjustText routes
      leader lines in the coordinate space the panel actually renders in.

    Parameters
    ----------
    y_col : column of p-values. Pass *nominal* p when adjusted p saturates
        (small-n pseudobulk often returns padj ~ 1 for everything, which
        flattens the volcano into a line) — but then say so in the axis
        label and caption. `y_label` exists for exactly that.
    label_genes : explicit iterable of names to label, overriding top-N.
        Prefer this for a final figure: hand-picked labels beat whatever
        |lfc| x -log10(p) happens to rank first.
    label_all_top : label top-N by score even if not significant. Use when
        the contrast is near-null and you want to show the best candidates —
        but the caption must then say they are not significant.
    star_set : names to mark with a star, e.g. genes replicated in an
        orthogonal test.
    exclude_patterns : regex list to drop before plotting, e.g.
        `NOISE_GENE_PATTERNS + CONTROL_PROBE_PATTERNS`. Prints what it drops.
    """
    t = resolve(theme)
    up_c = up_color or t.up
    dn_c = down_color or t.down

    d = pd.DataFrame(df).copy()
    for c in (x_col, y_col):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if exclude_patterns and label_col in d.columns:
        keep = filter_labels(d[label_col], exclude_patterns)
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"  [volcano] excluded {n_drop} rows matching exclude_patterns")
        d = d[keep].copy()

    # Rows with no p-value carry no y-position. DESeq2 emits NaN padj for
    # independent-filtered (low-count) genes; they are not "p = 1".
    n0 = len(d)
    d = d[d[y_col].notna() & d[x_col].notna()].copy()
    if n0 - len(d):
        print(f"  [volcano] dropped {n0 - len(d)} rows with NA {x_col}/{y_col}")
    if d.empty:
        raise ValueError("volcano: no finite rows left to plot")

    if lfc_cap is not None:
        d[x_col] = d[x_col].clip(-lfc_cap, lfc_cap)

    nonzero = d.loc[d[y_col] > 0, y_col]
    p_floor = float(nonzero.min()) * 0.5 if len(nonzero) else 1e-300
    d["_nlp"] = -np.log10(d[y_col].clip(lower=p_floor))

    if jitter:
        rng = np.random.RandomState(0)  # deterministic: reruns are identical
        d["_nlp"] += rng.uniform(-0.03, 0.03, size=len(d))
        d[x_col] += rng.uniform(-0.01, 0.01, size=len(d))

    sig = (d[y_col] < p_thresh) & (d[x_col].abs() >= lfc_thresh)
    up, dn = sig & (d[x_col] > 0), sig & (d[x_col] < 0)

    ax.scatter(d.loc[~sig, x_col], d.loc[~sig, "_nlp"], c=t.nonsig,
               s=point_size * 0.7, alpha=min(point_alpha, 0.5),
               edgecolors="none", rasterized=True)
    ax.scatter(d.loc[up, x_col], d.loc[up, "_nlp"], c=up_c, s=point_size,
               alpha=point_alpha, edgecolors="none", rasterized=True)
    ax.scatter(d.loc[dn, x_col], d.loc[dn, "_nlp"], c=dn_c, s=point_size,
               alpha=point_alpha, edgecolors="none", rasterized=True)

    # ── choose labels ───────────────────────────────────────────────────────
    d["_score"] = d[x_col].abs() * d["_nlp"]
    if label_genes is not None:
        want = set(map(str, label_genes))
        chosen = d[d[label_col].astype(str).isin(want)]
        missing = want - set(chosen[label_col].astype(str))
        if missing:
            print(f"  [volcano] label_genes not found: {sorted(missing)}")
        to_label = [(chosen[chosen[x_col] > 0], up_c), (chosen[chosen[x_col] < 0], dn_c)]
    elif top_n_labels > 0:
        n_each = max(1, top_n_labels // 2)
        pool_up = d[d[x_col] > 0] if label_all_top else d[up]
        pool_dn = d[d[x_col] < 0] if label_all_top else d[dn]
        to_label = [(pool_up.nlargest(n_each, "_score"), up_c),
                    (pool_dn.nlargest(n_each, "_score"), dn_c)]
    else:
        to_label = []

    stars = set(map(str, star_set)) if star_set else set()
    texts = []
    for frame, color in to_label:
        for _, r in frame.iterrows():
            name = str(r[label_col])
            texts.append(ax.text(r[x_col], r["_nlp"],
                                 f"{name}*" if name in stars else name,
                                 fontsize=8, color=color, fontweight="medium"))

    ax.axhline(-np.log10(p_thresh), color="#888", lw=0.4, ls=":")
    ax.axvline(lfc_thresh, color="#888", lw=0.4, ls=":")
    ax.axvline(-lfc_thresh, color="#888", lw=0.4, ls=":")

    sym = max(abs(v) for v in ax.get_xlim())
    ax.set_xlim(-sym, sym)
    ymax = float(d["_nlp"].max())
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1)  # headroom for labels

    xl = x_label if x_label is not None else "log2 fold change"
    if lfc_cap is not None and x_label is None:
        xl += f" (capped at ±{lfc_cap:g})"
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(y_label if y_label is not None else "−log10 p", fontsize=11)

    # Run last: adjustText needs the final coordinate space.
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="#888", lw=0.5,
                                        alpha=0.75, shrinkA=2, shrinkB=2),
                        expand_text=(2.0, 2.5), expand_points=(2.0, 2.5),
                        force_text=(1.0, 1.5), force_points=(0.5, 0.8),
                        only_move={"text": "xy", "points": "xy"},
                        max_move=(50, 50), iter_lim=500)
        except ImportError:
            print("  [volcano] adjustText not installed; labels may overlap")

    despine(ax)
    return {"n_plotted": len(d), "n_up": int(up.sum()), "n_down": int(dn.sum()),
            "p_floor": p_floor}
