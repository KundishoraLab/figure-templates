"""Build a single self-contained gallery/index.html with panels embedded.

    python demo/gallery.py --theme avm     # render panels first
    python demo/build_gallery_html.py      # then bundle them

Images are base64 data-URIs, so the file opens over file:// with no server
and survives being emailed or dropped into another repo.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

# (file stem, title, what it is, the call that made it)
PANELS = [
    ("01_volcano", "Volcano", "Differential expression: effect size vs significance.",
     'volcano(de, ax, x_col="log2FoldChange", y_col="pvalue",\n'
     '        lfc_thresh=0.8, p_thresh=0.01, top_n_labels=12,\n'
     '        exclude_patterns=NOISE_GENE_PATTERNS + CONTROL_PROBE_PATTERNS)'),
    ("02_dotplot_pathway", "Pathway dotplot", "Enrichment: x/colour = NES, size = FDR.",
     'dotplot_pathway(gsea, ax, top_n=18, sort_by="signed")'),
    ("03_dotplot_expression", "Expression dotplot",
     "Gene × group: colour = mean expression, size = % expressing.",
     'mean_df, pct_df = expression_matrix(adata, genes, "pole")\n'
     'dotplot_expression(mean_df, pct_df, ax, dividers=[9.5],\n'
     '                   block_labels=[(4.5, "up", UP), (14.5, "down", DN)])'),
    ("04_umap_categorical", "UMAP — categorical", "Embedding coloured by cell type.",
     'embedding_categorical(ax, xy, obs["label"], palette=pal,\n'
     '                      point_size=1.6, legend_title="Cell type")'),
    ("05_umap_continuous", "UMAP — continuous", "Embedding coloured by a score.",
     'embedding_continuous(ax, xy, obs["score"],\n'
     '                     norm=quantile_norm(obs["score"], .02, .98),\n'
     '                     cbar_label="signature score")'),
    ("06_umap_highlight", "UMAP — highlight", "Grey backdrop + highlighted subset.",
     'highlight_mask(ax, xy, mask, fg_size=6, bg_size=1.2,\n'
     '               label=f"variant+ (n={mask.sum()})")'),
    ("07_spatial_map", "Spatial map", "Tissue coordinates in mm, with scale bar.",
     'embedding_continuous(ax, xy, obs["score"], point_size=3.5)\n'
     'add_scale_bar(ax, length=0.5, unit="mm")'),
    ("08_heatmap", "Heatmap", "Signed matrix; diverging scale centred on zero.",
     'heatmap(m, ax, symmetric=True, cbar_label="signed score",\n'
     '        col_label_rotation=45)'),
    ("09_lollipop_tf", "Lollipop", "Signed per-feature effect; size = −log10 q.",
     'lollipop(tf, ax, label_col="source", value_col="delta",\n'
     '         q_col="q", top_n=24)'),
    ("10_stacked_bar", "Stacked bar", "Composition per sample, normalised.",
     'stacked_bar(comp, ax, palette=pal, legend_title="Cell type")'),
    ("11_ordered_bar", "Ordered bar", "One statistic per category, in a fixed order.",
     'ordered_bar(df, ax, value_col="pct_variant", order=AV_ORDER,\n'
     '            palette=theme.palette("ec_subtype"))'),
    ("12_chord_signed", "Chord (signed)", "Interaction Δ; needs pycirclize.",
     'chord_signed(m, "panels/chord", top_n_per_sign=14)'),
    ("13_upset", "UpSet", "Set intersections across modalities.",
     'upset(sets, "panels/upset", min_subset_size=1)'),
    ("19_sankey", "Sankey", "Label transfer, as % of each query label.",
     'sankey(crosstab, ax, normalize="source", min_flow=0.02)'),
]

# R-only panels (demo/gallery_clinical.R). These have no Python twin — the
# forest is a hand-rolled ggplot layout engine and KM leans on survminer.
R_PANELS = [
    ("14_forest_general", "Forest — predictors",
     "JAMA table-forest: text table | forest | estimate column. Arrowheads "
     "mark CIs clipped by the axis; box area ∝ N.",
     'fk_forest_general(df,\n'
     '  cols = list(list(col = "n_label", header = "N"),\n'
     '              list(col = "p_label", header = "FDR P")),\n'
     '  x_lab = "Odds ratio (95% CI)", est_col_header = "OR (95% CI)")'),
    ("15_forest_meta", "Forest — meta-analysis",
     "Per-study rows + pooled diamond. `label_header` retitles the implicit "
     "first column; directional labels sit under the axis.",
     'fk_forest_meta(df,\n'
     '  cols = list(list(col = "n_label", header = "N")),\n'
     '  label_header = "Study",\n'
     '  left_label = "← Negative", right_label = "Variant+ →")'),
    ("16_km_curve", "Kaplan-Meier",
     "Survival curves + risk table. CI ribbons at alpha 0.2, censor ticks "
     "dimmed to 0.4 — they annotate, they aren't the estimate.",
     'km <- fk_km_panel(df, "age", "event", "group",\n'
     '                  xlab = "Age (years)", xlim_max = 60)\n'
     'fk_save_km_panel("panels/", "fig1d_km", km, w = 5.5, h = 5)'),
    ("17_regression_scatter", "Regression scatter",
     "Points + fitted line with CI ribbon, by group. Faint points so density "
     "reads; `fixed_axis` pins one scale across panels.",
     'fk_regression_scatter(df, x_var = "age", y_var = "vaf",\n'
     '  color_var = "genotype", fixed_axis = "y",\n'
     '  fixed_limits = c(0, 8), fixed_breaks = c(0, 2, 4, 6, 8))'),
    ("18_dumbbell", "Dumbbell prevalence",
     "Paired prevalence per feature, dot size ∝ N, optional 95% CI bars. "
     "No connecting line — the dodge does the pairing.",
     'fk_dumbbell(df, feature_col = "feature", group_col = "group",\n'
     '            value_col = "prevalence", n_col = "n",\n'
     '            lo_col = "lo", hi_col = "hi")'),
]

CSS = """
:root { --bg:#ffffff; --fg:#16181d; --muted:#5c6370; --card:#f7f8fa;
        --border:#e3e6ea; --code:#f0f2f5; --accent:#c51b8a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --fg:#e6e8ec; --muted:#9aa1ad; --card:#171a21;
          --border:#272b34; --code:#1d212a; --accent:#f06fbe; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2.5rem 1.5rem 4rem; background:var(--bg); color:var(--fg);
       font:16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size:1.9rem; margin:0 0 .3rem; letter-spacing:-.02em; }
.sub { color:var(--muted); margin:0 0 2rem; }
.meta { display:inline-block; background:var(--card); border:1px solid var(--border);
        border-radius:6px; padding:.15rem .5rem; font-size:.8rem; color:var(--muted);
        margin-right:.4rem; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px;
        padding:1.25rem; margin:0 0 1.75rem; }
.card h2 { font-size:1.15rem; margin:0 0 .2rem; }
.card h2 .n { color:var(--accent); font-variant-numeric:tabular-nums; margin-right:.5rem; }
.card p { margin:0 0 .9rem; color:var(--muted); font-size:.92rem; }
.sect { font-size:1.05rem; margin:2.2rem 0 1rem; padding-bottom:.4rem;
        border-bottom:1px solid var(--border); letter-spacing:.02em;
        text-transform:uppercase; color:var(--muted); font-weight:600; }
.imgwrap { overflow-x:auto; background:#fff; border:1px solid var(--border);
           border-radius:6px; padding:.5rem; }
img { display:block; max-width:100%; height:auto; margin:0 auto; }
pre { background:var(--code); border:1px solid var(--border); border-radius:6px;
      padding:.75rem .9rem; overflow-x:auto; font-size:.82rem; line-height:1.5;
      margin:.9rem 0 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.missing { color:var(--muted); font-style:italic; padding:2rem; text-align:center;
           border:1px dashed var(--border); border-radius:6px; }
footer { color:var(--muted); font-size:.85rem; margin-top:2.5rem;
         border-top:1px solid var(--border); padding-top:1rem; }
code { background:var(--code); padding:.1rem .3rem; border-radius:4px; font-size:.9em; }
"""


def embed(p: Path) -> str:
    return ("data:image/png;base64,"
            + base64.b64encode(p.read_bytes()).decode("ascii"))


def main() -> int:
    gal = ROOT / "gallery"
    if not gal.exists():
        print(f"[error] {gal} not found — run demo/gallery.py first")
        return 1

    parts = [
        '<div class="wrap">',
        "<h1>figure-templates gallery</h1>",
        '<p class="sub">Every panel below is rendered from synthetic data by '
        '<code>demo/gallery.py</code> — no private inputs. Find the one that '
        "looks like what you need and copy its call.</p>",
        f"<p><span class=\"meta\">theme: avm</span>"
        f"<span class=\"meta\">{len(PANELS) + len(R_PANELS)} panel types</span>"
        f"<span class=\"meta\">python + R</span></p>",
    ]

    n_found = n_total = 0

    def emit(panels, src_dir, start):
        nonlocal n_found, n_total
        for i, (stem, title, desc, code) in enumerate(panels, start):
            n_total += 1
            png = src_dir / f"{stem}.png"
            parts.append('<div class="card">')
            parts.append(f'<h2><span class="n">{i:02d}</span>{title}</h2>')
            parts.append(f"<p>{desc}</p>")
            if png.exists():
                parts.append(f'<div class="imgwrap"><img alt="{title}" '
                             f'src="{embed(png)}"></div>')
                n_found += 1
            else:
                parts.append('<div class="missing">not rendered — optional '
                             "dependency missing (see README)</div>")
            parts.append(f"<pre>{code}</pre>")
            parts.append("</div>")

    parts.append('<h2 class="sect">Python — matplotlib</h2>')
    emit(PANELS, gal, 1)

    gal_r = ROOT / "gallery_R"
    parts.append('<h2 class="sect">R — ggplot2 (no Python twin)</h2>')
    parts.append('<p class="sub">Clinical panels ported from the '
                 "genotype-phenotype manuscript pipeline. Render with "
                 "<code>Rscript demo/gallery_clinical.R --theme avm</code>.</p>")
    emit(R_PANELS, gal_r, len(PANELS) + 1)

    parts.append(
        f"<footer>{n_found}/{n_total} panels embedded. "
        "Rebuild: <code>python demo/gallery.py --theme avm && "
        "Rscript demo/gallery.R --theme avm && "
        "Rscript demo/gallery_clinical.R --theme avm && "
        "python demo/build_gallery_html.py</code></footer></div>")

    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>figure-templates gallery</title><style>{CSS}</style></head>"
            f"<body>{''.join(parts)}</body></html>")

    out = gal / "index.html"
    out.write_text(html)
    size_mb = out.stat().st_size / 1e6
    print(f"[ok] {out}  ({n_found}/{n_total} panels, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
