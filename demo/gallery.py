"""Render one example panel per plot type, from synthetic data.

    python demo/gallery.py                # base theme -> gallery/
    python demo/gallery.py --theme avm    # AVM domain theme
    python demo/gallery.py --out /tmp/g   # elsewhere

Every panel here is a working template: find the one that looks like what you
need, and copy its function body. No private data, no network, no HPC paths.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figkit import apply_rcparams, quantile_norm, save_panel, set_theme  # noqa: E402
from figkit.plots import (  # noqa: E402
    CONTROL_PROBE_PATTERNS, NOISE_GENE_PATTERNS, add_scale_bar, chord_signed,
    dotplot_expression, dotplot_pathway, embedding_categorical,
    embedding_continuous, heatmap, highlight_mask, lollipop, ordered_bar,
    sankey, stacked_bar, upset, volcano,
)
import synth  # noqa: E402

PANELS = []
FORMATS = ("png",)  # set from --formats
DPI = 200           # set from --dpi


def save(fig, name, out):
    """save_panel with the gallery's format/dpi choice (see --dpi/--formats)."""
    return save_panel(fig, name, out, formats=FORMATS)


def panel(name, description):
    """Register a gallery panel. `description` shows up in the HTML index."""
    def deco(fn):
        PANELS.append((name, description, fn))
        return fn
    return deco


# ── volcano ─────────────────────────────────────────────────────────────────
@panel("volcano", "Differential expression: effect size vs significance, "
                  "colored by direction, with noise-probe filtering.")
def _volcano(out, theme):
    de = synth.de_table()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    stats = volcano(
        de, ax, x_col="log2FoldChange", y_col="pvalue", label_col="gene",
        lfc_thresh=0.8, p_thresh=0.01, top_n_labels=12,
        exclude_patterns=NOISE_GENE_PATTERNS + CONTROL_PROBE_PATTERNS,
        x_label="log2FC (proximal − distal)", y_label="−log10 nominal p",
        theme=theme,
    )
    ax.set_title("Proximal vs distal", fontsize=12)
    save(fig, "01_volcano", out)
    return stats


# ── pathway dotplot ─────────────────────────────────────────────────────────
@panel("dotplot_pathway", "Pathway enrichment: x/color = NES, size = FDR. "
                          "sort_by='signed' keeps both directions.")
def _pathway(out, theme):
    gs = synth.pathway_table()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    stats = dotplot_pathway(gs, ax, top_n=18, sort_by="signed", theme=theme,
                            x_label="NES (proximal − distal)")
    ax.set_title("Hallmark enrichment", fontsize=12)
    save(fig, "02_dotplot_pathway", out)
    return stats


# ── expression dotplot ──────────────────────────────────────────────────────
@panel("dotplot_expression", "Gene x group expression: color = mean, "
                             "size = % expressing, blocked by direction.")
def _expr(out, theme):
    de = synth.de_table()
    d = de[de["pvalue"].notna()].copy()
    d["score"] = d["log2FoldChange"].abs() * -np.log10(d["pvalue"].clip(lower=1e-300))
    keep = ~d["gene"].str.contains("MT-|RPL|RPS|LINC|MIR|Negative|SystemControl|FalseCode|^AL",
                                   regex=True)
    d = d[keep]
    up = d[d["log2FoldChange"] > 0].nlargest(10, "score")["gene"].tolist()
    dn = d[d["log2FoldChange"] < 0].nlargest(10, "score")["gene"].tolist()
    genes = up + dn

    mean_df, pct_df = synth.expression_matrices(genes, ("proximal", "distal"))
    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    stats = dotplot_expression(
        mean_df, pct_df, ax, theme=theme,
        group_colors={"proximal": theme.up if theme else None,
                      "distal": theme.down if theme else None},
        gene_colors={**{g: (theme.up if theme else "#c51b8a") for g in up},
                     **{g: (theme.down if theme else "#1f3a6e") for g in dn}},
        dividers=[len(up) - 0.5],
        block_labels=[(len(up) / 2 - 0.5, "proximal-up", theme.up if theme else "#c51b8a"),
                      (len(up) + len(dn) / 2 - 0.5, "distal-up", theme.down if theme else "#1f3a6e")],
        cbar_label="mean log1p expr",
    )
    save(fig, "03_dotplot_expression", out)
    return stats


# ── UMAP: categorical ───────────────────────────────────────────────────────
@panel("umap_categorical", "Embedding colored by category, one scatter per "
                           "level, legend mounted outside the axes.")
def _umap_cat(out, theme):
    xy, obs = synth.embedding()
    palette = None
    if theme is not None and "ec_subtype" in theme.palettes:
        from figkit.themes.avm import CELL_TYPE_COLORS, EC_SUBTYPE_COLORS
        palette = {**EC_SUBTYPE_COLORS, **CELL_TYPE_COLORS}
    fig, ax = plt.subplots(figsize=(7, 5))
    stats = embedding_categorical(ax, xy, obs["label"], palette=palette,
                                  point_size=1.6, legend_title="Cell type",
                                  theme=theme)
    ax.set_title("Atlas by cell type", fontsize=12)
    save(fig, "04_umap_categorical", out)
    return stats


# ── UMAP: continuous ────────────────────────────────────────────────────────
@panel("umap_continuous", "Embedding colored by a continuous score; high "
                          "values drawn last so signal isn't buried.")
def _umap_cont(out, theme):
    xy, obs = synth.embedding()
    fig, ax = plt.subplots(figsize=(6, 5))
    embedding_continuous(ax, xy, obs["score"], point_size=1.8,
                         norm=quantile_norm(obs["score"], 0.02, 0.98),
                         cbar_label="signature score", theme=theme)
    ax.set_title("Signature score", fontsize=12)
    save(fig, "05_umap_continuous", out)
    return {"n_points": len(xy)}


# ── UMAP: two-layer highlight ───────────────────────────────────────────────
@panel("umap_highlight", "Two-layer scatter: grey backdrop of all cells + "
                         "highlighted subset. The backdrop is the control.")
def _umap_hl(out, theme):
    xy, obs = synth.embedding()
    mask = obs["is_mutant"].values
    fig, ax = plt.subplots(figsize=(6, 5))
    highlight_mask(ax, xy, mask, label=f"variant+ (n={int(mask.sum())})",
                   fg_size=6, bg_size=1.2, theme=theme)
    ax.legend(loc="upper right", frameon=False, fontsize=9, markerscale=2)
    ax.set_title("Variant-positive cells on atlas", fontsize=12)
    save(fig, "06_umap_highlight", out)
    return {"n_highlighted": int(mask.sum())}


# ── spatial map ─────────────────────────────────────────────────────────────
@panel("spatial_map", "Tissue map in real units: continuous score + scale bar.")
def _spatial(out, theme):
    xy, obs = synth.spatial()
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    embedding_continuous(ax, xy, obs["score"], point_size=3.5,
                         norm=quantile_norm(obs["score"], 0.02, 0.98),
                         cbar_label="niche score", theme=theme,
                         x_label="", y_label="")
    add_scale_bar(ax, length=0.5, unit="mm", position="lower right")
    ax.set_title("Spatial niche score", fontsize=12)
    save(fig, "07_spatial_map", out)
    return {"n_points": len(xy)}


# ── heatmap ─────────────────────────────────────────────────────────────────
@panel("heatmap", "Signed score matrix; diverging scale centered on zero, "
                  "NaN cells grey (missing ≠ zero).")
def _heat(out, theme):
    m = synth.score_matrix()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    heatmap(m, ax, symmetric=True, cbar_label="signed score",
            col_label_rotation=45, theme=theme)
    ax.set_title("Program activity by subtype", fontsize=12)
    save(fig, "08_heatmap", out)
    return {"shape": list(m.shape)}


# ── lollipop ────────────────────────────────────────────────────────────────
@panel("lollipop", "Signed per-feature effect (TF activity): length = effect, "
                   "size = −log10 q, star strip = significant.")
def _lolli(out, theme):
    tf = synth.tf_table()
    fig, ax = plt.subplots(figsize=(8, 4))
    stats = lollipop(tf, ax, label_col="source", value_col="delta", q_col="q",
                     top_n=24, y_label="TF activity Δ (proximal − distal)",
                     theme=theme)
    ax.set_title("Transcription-factor activity", fontsize=12)
    save(fig, "09_lollipop_tf", out)
    return stats


# ── stacked composition bar ─────────────────────────────────────────────────
@panel("stacked_bar", "Per-sample composition, normalized to fractions so "
                      "group sizes don't confound composition.")
def _stack(out, theme):
    comp = synth.composition()
    palette = None
    if theme is not None and "ec_subtype" in theme.palettes:
        from figkit.themes.avm import CELL_TYPE_COLORS, EC_SUBTYPE_COLORS
        palette = {**EC_SUBTYPE_COLORS, **CELL_TYPE_COLORS}
    labelled = comp.copy()
    labelled.index = [f"{s} (n={int(n):,})" for s, n in zip(comp.index, comp.sum(axis=1))]
    fig, ax = plt.subplots(figsize=(7.5, 3))
    stats = stacked_bar(labelled, ax, palette=palette, legend_title="Cell type",
                        theme=theme)
    ax.set_title("Composition by sample", fontsize=12)
    save(fig, "10_stacked_bar", out)
    return stats


# ── ordered bar ─────────────────────────────────────────────────────────────
@panel("ordered_bar", "One statistic per category, held in a meaningful order "
                      "(anatomical axis) rather than sorted by value.")
def _obar(out, theme):
    import pandas as pd
    rng = np.random.RandomState(11)
    order = synth.EC_SUBTYPES
    df = pd.DataFrame({
        "cell_type": order,
        "pct_variant": np.clip(rng.uniform(0, 3, len(order)) +
                               np.linspace(0, 4, len(order)), 0, None),
    })
    palette = None
    if theme is not None and "ec_subtype" in theme.palettes:
        palette = theme.palette("ec_subtype")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    stats = ordered_bar(df, ax, value_col="pct_variant", label_col="cell_type",
                        order=order, palette=palette,
                        x_label="Variant+ cells (% of subtype)", theme=theme)
    ax.set_title("Variant burden along the arteriovenous axis", fontsize=12)
    save(fig, "11_ordered_bar", out)
    return stats


# ── chord (optional dep) ────────────────────────────────────────────────────
@panel("chord_signed", "Signed interaction chord: separate bands for increased "
                       "and decreased links. Needs `pycirclize`.")
def _chord(out, theme):
    m = synth.interaction_matrix()
    try:
        chord_signed(m, out / "12_chord_signed", top_n_per_sign=14,
                     title="Ligand–receptor Δ", theme=theme, dpi=DPI, formats=FORMATS)
    except ImportError as e:
        print(f"  [skip] {e}")
        return {"skipped": "pycirclize not installed"}
    return {"n_sectors": m.shape[0]}


# ── upset (optional dep) ────────────────────────────────────────────────────
@panel("upset", "Set intersections across modalities. Needs `upsetplot`.")
def _upset(out, theme):
    sets = synth.gene_sets()
    try:
        upset(sets, out / "13_upset", min_subset_size=1, dpi=DPI, formats=FORMATS)
    except ImportError as e:
        print(f"  [skip] {e}")
        return {"skipped": "upsetplot not installed"}
    return {"n_sets": len(sets)}


# ── sankey ──────────────────────────────────────────────────────────────────
@panel("sankey", "Label transfer: where each query label landed in the "
                 "reference's vocabulary.")
def _sankey(out, theme):
    m = synth.label_transfer()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    stats = sankey(m, ax, normalize="source", min_flow=0.02,
                   source_label="Query", target_label="Reference", theme=theme)
    save(fig, "19_sankey", out)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theme", default="base", choices=["base", "avm"])
    ap.add_argument("--out", default=None, help="output dir (default gallery/)")
    ap.add_argument("--only", default=None, help="render only this panel name")
    ap.add_argument("--dpi", type=int, default=200,
                    help="gallery is a reference, not a print artifact — 200 "
                         "keeps it light. Use 600 for real panels (default: 200)")
    ap.add_argument("--formats", default="png",
                    help="comma-separated, e.g. png,svg (default: png)")
    args = ap.parse_args()
    global FORMATS, DPI
    FORMATS = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    DPI = args.dpi

    theme = None
    if args.theme == "avm":
        from figkit.themes.avm import AVM_THEME
        theme = set_theme(AVM_THEME)
    else:
        from figkit import BASE_THEME
        theme = set_theme(BASE_THEME)

    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    apply_rcparams(theme)
    # apply_rcparams installs the theme's publication dpi (600). The gallery is
    # a reference, so override it — 600dpi PNGs of 6000-point scatters make the
    # committed gallery an order of magnitude bigger than the code.
    matplotlib.rcParams["savefig.dpi"] = args.dpi

    print(f"[gallery] theme={args.theme} dpi={args.dpi} "
          f"formats={','.join(FORMATS)} -> {out}")
    failures = []
    for name, _desc, fn in PANELS:
        if args.only and args.only != name:
            continue
        print(f"[render] {name}")
        try:
            stats = fn(out, theme)
            if stats:
                print(f"         {stats}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append((name, repr(e)))

    n = len(PANELS) if not args.only else 1
    print(f"\n[done] {n - len(failures)}/{n} panels -> {out}")
    if failures:
        for name, err in failures:
            print(f"  [FAIL] {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
