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
                    size_label: str = "FDR", prettify: bool = True):
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
    size_label : what the size key calls the quantity in `padj_col`. Defaults
        to "FDR", which is what the parameter name promises and what every
        caller has passed; it is a parameter rather than a constant because
        the key used to say FDR over whatever arrived, and a panel drawn from
        an unadjusted p had no way to say so. See `dotplot_matrix` for the
        case that made this necessary.
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
                        title=size_label, loc="lower right")
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
                       cbar_label: str | None = None,
                       show_size_legend: bool = True,
                       swap_axes: bool = False,
                       scale: bool = False, scale_clip: float = 2.5,
                       center: bool = False,
                       vmin: float | None = None, vmax: float | None = None,
                       col_rotation: int = 70, label_size: float | None = None,
                       colorbar: bool = True):
    """Gene-by-group dotplot: color = mean expression, size = % expressing.

    Rows = groups, columns = genes (set `swap_axes=True` to transpose). Both
    frames must share index and columns — `expression_matrix()` guarantees it.

    Color is scaled to the `vmax_pct` percentile rather than the max, so a
    single hot gene doesn't wash out every other column.

    scale : z-score each gene across groups and clip to +/- `scale_clip`, which
        is Seurat's `DotPlot(scale = TRUE)`. Use it when the panel's question is
        "which group is this gene highest in"; leave it off when it is "how much
        of this gene is there", because a z-score discards the level. A gene
        with no variance across groups maps to 0. Note that z-scores are
        estimated from as many observations as there are groups, so a group
        built from a handful of cells moves every gene's mean — drop tiny
        groups before scaling rather than after.
    center : pin zero to the middle of the ramp (TwoSlopeNorm). Required with a
        diverging cmap, whose midpoint colour claims to mean zero; wrong with a
        two-colour gradient like Seurat's blue->red, which has no privileged
        middle and should span the observed range instead.

    group_colors / gene_colors : {label: hex} maps that tint the tick labels.
        Cheap and effective — it ties each row/column back to the contrast
        color used in the volcano without adding a legend.
    dividers : x positions (after the Nth column) to draw a dashed separator,
        for visually blocking e.g. up-genes from down-genes.
    block_labels : [(x_center, text, color)] headers drawn above the blocks.

    vmin / vmax : pin the colour limits instead of deriving them per call. Pass
        both whenever a set of panels is meant to be compared to each other:
        the `vmax_pct` default is computed from the data in front of it, so
        three organoid panels drawn separately each get their own ramp and the
        weakest is rendered as strongly as the strongest. Nothing warns about
        that — the panels simply lie side by side.
    col_rotation : column-label angle. 70 is right when the labels are long
        enough to overlap otherwise; 45 reads better when they fit, because a
        near-vertical name is read one character at a time.
    label_size : one font size for both tick axes. Defaults to the built-in
        10 / 8 split.
    colorbar : draw the colourbar. Turn it off when the panel also carries
        furniture this function cannot see — a `group_strip`, a shared bar
        across a row of panels — and place it from the caller, which is the
        only place that knows the final layout.
    """
    t = resolve(theme)
    mean_df, pct_df = pd.DataFrame(mean_df), pd.DataFrame(pct_df)
    if list(mean_df.index) != list(pct_df.index) or \
       list(mean_df.columns) != list(pct_df.columns):
        raise ValueError("dotplot_expression: mean_df and pct_df must share "
                         "index and columns")
    if cbar_label is None:
        cbar_label = "scaled expression" if scale else "mean expression"
    # Per gene, over groups — so it has to happen while genes are still the
    # columns, i.e. before swap_axes. ddof=1 to match R's scale().
    if scale:
        sd = mean_df.std(axis=0, ddof=1).replace(0.0, np.nan)
        mean_df = (((mean_df - mean_df.mean(axis=0)) / sd)
                   .fillna(0.0).clip(-scale_clip, scale_clip))
    if swap_axes:
        mean_df, pct_df = mean_df.T, pct_df.T
        group_colors, gene_colors = gene_colors, group_colors

    cmap = cmap if cmap is not None else sequential_cmap(t.up, name="expr_seq")
    from matplotlib.colors import Normalize
    vals = mean_df.values.ravel()
    finite = np.isfinite(vals).any()
    if scale:
        # Over the observed range, not symmetric about zero and not
        # percentile-derived, which is what Seurat's scale_color_gradient()
        # does. It matters: with k groups a gene that is specific to one of
        # them puts k-1 values at about -1/sqrt(k-1) and one at the clip, so a
        # symmetric ramp would spend its whole cold half on values that never
        # occur and render every negative cell the same lukewarm mid-tone.
        lo = float(np.nanmin(vals)) if finite else -1.0
        hi = float(np.nanmax(vals)) if finite else 1.0
        if hi <= lo:
            lo, hi = lo - 0.5, lo + 0.5
    else:
        lo = 0.0
        hi = float(np.percentile(vals, vmax_pct)) if finite else 1.0
        if hi <= lo:
            hi = 1.0
    # An explicit limit is a claim about a set of panels, so it outranks
    # anything derived from the one in front of us.
    if vmin is not None:
        lo = float(vmin)
    if vmax is not None:
        hi = float(vmax)
    if hi <= lo:
        raise ValueError(f"dotplot_expression: vmax ({hi}) must exceed vmin ({lo})")

    if scale and center:
        # For a diverging cmap the midpoint colour means zero, so zero has to
        # land on it. TwoSlopeNorm rather than a symmetric Normalize because
        # the two halves are not the same length here and forcing them to be
        # wastes most of the ramp: scaled expression is skewed by construction,
        # k-1 groups just below zero and one at the clip.
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vcenter=0.0, vmin=min(lo, -1e-9), vmax=max(hi, 1e-9))
    else:
        norm = Normalize(vmin=lo, vmax=hi)

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
    ax.set_yticklabels(rows, fontsize=10 if label_size is None else label_size)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=col_rotation,
                       fontsize=8 if label_size is None else label_size,
                       ha="right" if col_rotation else "center")
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

    cb = None
    if colorbar:
        import matplotlib.pyplot as plt
        cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.tick_params(labelsize=9)
        cb.set_label(cbar_label, fontsize=10)

    if show_size_legend:
        size_legend(ax, [25, 50, 100], _size, label_fn=lambda v: f"{v:g}%",
                    title="% expressing", loc="upper left", color="#888")
        ax.get_legend().set_bbox_to_anchor((1.06, 1.0))

    despine(ax, "all")
    # `scatter` is the mappable. A caller that suppressed the colourbar needs it
    # to draw its own, and there is no other way to get back a norm+cmap pair
    # that is guaranteed to match the dots.
    return {"n_groups": len(rows), "n_genes": len(cols),
            "vmin": lo, "vmax": hi, "scatter": sc, "colorbar": cb,
            "size_fn": _size}


# Below this, a q-value stops buying dot area. Without a floor one term at
# q=1e-40 sets the scale and every other dot collapses to the same speck —
# the panel then encodes "is anything significant", which it already knows.
FDR_FLOOR = 1e-4


def dotplot_matrix(value_df, ax, size_df=None, cmap=None,
                   size_range=(18.0, 98.0), col_rotation: int = 45,
                   rule_after: int | None = None, rule_label: str | None = None,
                   cbar_label: str | None = None, label_size: float | None = None,
                   size_label: str = "FDR", theme: Theme | None = None):
    """Signed statistic by category: colour = effect, size = significance.

    Rows are `value_df.index`, columns are its columns, and `size_df` — same
    shape, same labels — holds the q-values. Both encodings are needed and
    neither substitutes for the other: colour without size shows a strong
    effect that might be noise, size without colour shows a confident result
    without saying which way it went.

    The colour scale is symmetric about zero and the cmap is expected to be
    diverging, because the quantity is signed and the midpoint has to mean "no
    effect". An asymmetric ramp on signed data puts the neutral colour at some
    arbitrary non-zero value, and every reader takes it for zero anyway.

    **`size_label` is a claim, and this function cannot check it.** The size
    key read "FDR" whatever arrived in `size_df`, which is a statement about
    someone else's arithmetic printed on their behalf. It happened: a caller
    wrote an *unadjusted* p to a file named `progeny_padj` and passed it here,
    and the panel said FDR three times over — in the filename, in the column
    name, and in this key — with no correction anywhere behind it. This
    parameter does not catch that and nothing here can; what it does is make
    the true label sayable, so a panel drawn from a nominal p can say so
    instead of being forced to lie. The default stays "FDR" because every
    existing caller passes one and a changed default would silently relabel
    their panels. If your `size_df` is not adjusted, pass what it is.

    Significance is `-log10(q)` clipped at `FDR_FLOOR` — see the constant.
    Returns `(colorbar, handles)`, where `handles` is the size key. It is
    returned rather than drawn because these panels usually already carry a
    `group_strip`, and one legend below a panel reads better than a second
    competing for a corner; pass it straight through as `extra_handles`.

    rule_after : draw a separator after this many rows, for a block of rows
      that met some criterion the rest did not.
    rule_label : what the rule means. Returned as a legend handle, not drawn
      above the axes — a sentence over the plot is a title, and this paper's
      panels do not carry one. Ignored unless `rule_after` is set.
    """
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    t = resolve(theme)
    value_df = pd.DataFrame(value_df)
    rows, cols = list(value_df.index), list(value_df.columns)
    if size_df is not None:
        size_df = pd.DataFrame(size_df)
        if list(size_df.index) != rows or list(size_df.columns) != cols:
            raise ValueError("dotplot_matrix: size_df must share index and "
                             "columns with value_df")

    cmap = cmap if cmap is not None else diverging_cmap(t.down, t.up,
                                                        name="matrix_div")
    vals = value_df.values.astype(float)
    lim = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    norm = Normalize(vmin=-lim, vmax=lim if lim > 0 else 1.0)

    s0, s1 = float(size_range[0]), float(size_range[1])
    span = -np.log10(FDR_FLOOR)

    def _size(q):
        if not np.isfinite(q):
            return s0
        nlq = -np.log10(min(max(float(q), FDR_FLOOR), 1.0))
        return s0 + (s1 - s0) * (nlq / span)

    xs, ys, ss, cs = [], [], [], []
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            xs.append(j); ys.append(i); cs.append(v)
            ss.append(_size(size_df.values[i, j]) if size_df is not None else s1)
    sc = ax.scatter(xs, ys, s=ss, c=cs, cmap=cmap, norm=norm,
                    edgecolors="#333", linewidths=0.3, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9 if label_size is None else label_size)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=col_rotation,
                       fontsize=8 if label_size is None else label_size,
                       ha="right" if col_rotation else "center")
    ax.tick_params(length=0)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    # Horizontal only: the rows are what a reader tracks across, and a vertical
    # line through a categorical axis separates nothing.
    ax.grid(visible=True, axis="y")
    ax.grid(visible=False, axis="x")

    rule_handle = None
    if rule_after is not None:
        ax.axhline(rule_after - 0.5, color="#333", linewidth=0.8, zorder=4)
        if rule_label:
            rule_handle = Line2D([0], [0], color="#333", linewidth=0.8,
                                 label=rule_label)

    import matplotlib.pyplot as plt
    cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9)

    refs = [1.0, 0.05, FDR_FLOOR]
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCCCCC",
                      markeredgecolor="black", markeredgewidth=0.3,
                      markersize=float(np.sqrt(_size(q))),
                      label=(f"{size_label} ≤{q:g}" if q <= FDR_FLOOR
                             else f"{size_label} {q:g}"))
               for q in refs] if size_df is not None else []
    if rule_handle is not None:
        handles.append(rule_handle)

    despine(ax, "all")
    return cb, handles
