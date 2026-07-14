"""Dot plots: pathway enrichment, and gene-by-group expression."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..palettes import diverging_cmap, quantile_norm, sequential_cmap
from ..style import despine, size_legend
from ..theme import Theme, resolve

# Prefixes stripped from MSigDB-style pathway names for display.
PATHWAY_PREFIXES = ("HALLMARK_", "REACTOME_", "KEGG_", "BIOCARTA_",
                    "GO_", "GOBP_", "GOCC_", "GOMF_", "WP_", "PID_")


def prettify_pathway(name, max_len: int = 50, prefixes=PATHWAY_PREFIXES) -> str:
    """HALLMARK_TNFA_SIGNALING_VIA_NFKB -> 'Tnfa Signaling Via Nfkb'."""
    s = str(name)
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):]
            break
    s = s.replace("_", " ").title()
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def dotplot_pathway(df, ax, pathway_col: str = "pathway", nes_col: str = "NES",
                    padj_col: str = "padj", top_n: int = 20,
                    max_label_len: int = 50, sort_by: str = "significance",
                    theme: Theme | None = None,
                    up_color: str | None = None, down_color: str | None = None,
                    size_range=(30, 110), show_size_legend: bool = True,
                    x_label: str = "NES", cbar_label: str | None = None,
                    prettify: bool = True):
    """Pathway enrichment dotplot: y = pathway, x = NES, size = -log10(padj),
    color = NES.

    Color and x redundantly encode NES on purpose — the color makes direction
    readable at a glance, x makes magnitude comparable. Size carries
    significance, which is a *different* axis of information from effect
    size; conflating the two (the classic "color by p-value" mistake) hides
    large-but-uncertain effects.

    sort_by : 'significance' (default) ranks by -log10(padj); 'nes' ranks by
        |NES|; 'signed' keeps the most positive and most negative (top_n/2
        each), which is the honest choice when you want to show both
        directions rather than whichever one happens to dominate.
    """
    t = resolve(theme)
    up_c, dn_c = up_color or t.up, down_color or t.down
    cmap = diverging_cmap(dn_c, up_c, t.neutral, name="pathway_div")

    d = pd.DataFrame(df).copy()
    d[padj_col] = pd.to_numeric(d[padj_col], errors="coerce").clip(lower=1e-300)
    d[nes_col] = pd.to_numeric(d[nes_col], errors="coerce")
    d = d[d[padj_col].notna() & d[nes_col].notna()].copy()
    if d.empty:
        raise ValueError("dotplot_pathway: no finite rows to plot")
    d["_nlp"] = -np.log10(d[padj_col])

    if sort_by == "significance":
        d = d.nlargest(top_n, "_nlp")
    elif sort_by == "nes":
        d = d.assign(_a=d[nes_col].abs()).nlargest(top_n, "_a")
    elif sort_by == "signed":
        half = max(1, top_n // 2)
        d = pd.concat([d.nlargest(half, nes_col), d.nsmallest(half, nes_col)])
        d = d.drop_duplicates(subset=[pathway_col])
    else:
        raise ValueError(f"sort_by must be significance|nes|signed, got {sort_by!r}")

    d = d.sort_values(nes_col)  # ascending: largest ends up at the top
    y = np.arange(len(d))

    lo, hi = float(d["_nlp"].min()), float(d["_nlp"].max())
    span = max(hi - lo, 1e-9)
    s0, s1 = size_range

    def _size(nlp):
        return s0 + (s1 - s0) * (nlp - lo) / span

    norm = quantile_norm(d[nes_col].values, 0.05, 0.95, symmetric=True)
    sc = ax.scatter(d[nes_col], y, s=[_size(v) for v in d["_nlp"]],
                    c=d[nes_col], cmap=cmap, norm=norm,
                    edgecolors="black", linewidths=0.4, zorder=3)

    ax.axvline(0, color="#888", lw=0.4, ls=":", zorder=1)
    ax.set_yticks(y)
    labels = [prettify_pathway(p, max_label_len) if prettify else str(p)
              for p in d[pathway_col]]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(x_label)

    import matplotlib.pyplot as plt
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=10)
    cb.set_label(cbar_label if cbar_label is not None else x_label, fontsize=10)

    if show_size_legend and len(d) > 1:
        refs = [v for v in (-np.log10(0.05), (lo + hi) / 2, hi) if lo <= v <= hi]
        if refs:
            size_legend(ax, sorted(set(refs)), _size,
                        label_fn=lambda v: f"{10 ** (-v):.1e}",
                        title="FDR", loc="lower right")
    despine(ax)
    return {"n_shown": len(d)}


# Back-compat alias for the original name.
dotplot_gsea = dotplot_pathway


def expression_matrix(adata, genes, group_col: str, groups=None,
                      layer: str | None = None, use_raw: bool = False):
    """Mean expression + percent-expressing per (group, gene) from an AnnData.

    Returns (mean_df, pct_df), both groups x genes. Genes absent from
    var_names are dropped with a warning rather than silently zero-filled —
    a zero row is indistinguishable from "not detected" and would be read
    as biology.

    Expects `adata.X` to already be normalized and log1p'd; this function
    does not transform for you, because silently re-normalizing someone's
    already-normalized matrix is a classic double-transform bug.
    """
    X = adata.raw.X if use_raw else (adata.layers[layer] if layer else adata.X)
    var_names = list(adata.raw.var_names if use_raw else adata.var_names)
    idx = {g: i for i, g in enumerate(var_names)}

    wanted = list(dict.fromkeys(map(str, genes)))
    valid = [g for g in wanted if g in idx]
    if len(valid) < len(wanted):
        missing = [g for g in wanted if g not in idx]
        print(f"  [expression_matrix] {len(missing)} genes not in var_names: "
              f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
    if not valid:
        raise ValueError("expression_matrix: none of `genes` are in var_names")

    labels = np.asarray(adata.obs[group_col].astype(str))
    groups = [str(g) for g in groups] if groups is not None \
        else sorted(pd.unique(labels))

    mean_M = np.zeros((len(groups), len(valid)))
    pct_M = np.zeros((len(groups), len(valid)))
    cols = [idx[g] for g in valid]
    for i, grp in enumerate(groups):
        m = labels == grp
        if not m.any():
            print(f"  [expression_matrix] group {grp!r} has 0 cells")
            continue
        sub = X[m][:, cols]
        if hasattr(sub, "toarray"):
            sub = sub.toarray()
        sub = np.asarray(sub, dtype=float)
        mean_M[i] = sub.mean(axis=0)
        pct_M[i] = (sub > 0).mean(axis=0) * 100

    return (pd.DataFrame(mean_M, index=groups, columns=valid),
            pd.DataFrame(pct_M, index=groups, columns=valid))


def dotplot_expression(mean_df, pct_df, ax, cmap=None,
                       theme: Theme | None = None,
                       vmax_pct: float = 99.0, max_dot: float = 130.0,
                       group_colors=None, gene_colors=None,
                       dividers=None, block_labels=None,
                       cbar_label: str = "mean expression",
                       show_size_legend: bool = True,
                       swap_axes: bool = False):
    """Gene-by-group dotplot: color = mean expression, size = % expressing.

    Rows = groups, columns = genes (set `swap_axes=True` to transpose). Both
    frames must share index and columns — `expression_matrix()` guarantees it.

    Color is scaled to the `vmax_pct` percentile rather than the max, so a
    single hot gene doesn't wash out every other column.

    group_colors / gene_colors : {label: hex} maps that tint the tick labels.
        Cheap and effective — it ties each row/column back to the contrast
        color used in the volcano without adding a legend.
    dividers : x positions (after the Nth column) to draw a dashed separator,
        for visually blocking e.g. up-genes from down-genes.
    block_labels : [(x_center, text, color)] headers drawn above the blocks.
    """
    t = resolve(theme)
    mean_df, pct_df = pd.DataFrame(mean_df), pd.DataFrame(pct_df)
    if list(mean_df.index) != list(pct_df.index) or \
       list(mean_df.columns) != list(pct_df.columns):
        raise ValueError("dotplot_expression: mean_df and pct_df must share "
                         "index and columns")
    if swap_axes:
        mean_df, pct_df = mean_df.T, pct_df.T
        group_colors, gene_colors = gene_colors, group_colors

    cmap = cmap if cmap is not None else sequential_cmap(t.up, name="expr_seq")
    from matplotlib.colors import Normalize
    vals = mean_df.values.ravel()
    vmax = float(np.percentile(vals, vmax_pct)) if np.isfinite(vals).any() else 1.0
    norm = Normalize(vmin=0, vmax=vmax if vmax > 0 else 1.0)

    rows, cols = list(mean_df.index), list(mean_df.columns)

    def _size(pct):
        return max(2.0, float(pct) / 100.0 * max_dot)

    xs, ys, ss, cs = [], [], [], []
    for i, _ in enumerate(rows):
        for j, _ in enumerate(cols):
            xs.append(j); ys.append(i)
            ss.append(_size(pct_df.values[i, j]))
            cs.append(mean_df.values[i, j])
    sc = ax.scatter(xs, ys, s=ss, c=cs, cmap=cmap, norm=norm,
                    edgecolors="#333", linewidths=0.25, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=70, fontsize=8, ha="right")
    if group_colors:
        for lab, tick in zip(rows, ax.get_yticklabels()):
            if lab in group_colors:
                tick.set_color(group_colors[lab])
    if gene_colors:
        for lab, tick in zip(cols, ax.get_xticklabels()):
            if lab in gene_colors:
                tick.set_color(gene_colors[lab])
    ax.tick_params(length=0)

    for x in (dividers or []):
        ax.axvline(x, color="#bbb", lw=0.6, ls="--", alpha=0.7, zorder=1)
    # Block headers sit above the first row. The y-axis is inverted below, so
    # "above the first row" is a *smaller* y than row 0.
    header_y = -0.38
    for xc, text, color in (block_labels or []):
        ax.text(xc, header_y, text, ha="center", va="bottom", color=color,
                fontsize=9, fontweight="medium")

    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(header_y - 0.35 if block_labels else -0.6, len(rows) - 0.5)
    ax.invert_yaxis()

    import matplotlib.pyplot as plt
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=9)
    cb.set_label(cbar_label, fontsize=10)

    if show_size_legend:
        size_legend(ax, [25, 50, 100], _size, label_fn=lambda v: f"{v:g}%",
                    title="% expressing", loc="upper left", color="#888")
        ax.get_legend().set_bbox_to_anchor((1.06, 1.0))

    despine(ax, "all")
    return {"n_groups": len(rows), "n_genes": len(cols), "vmax": vmax}
