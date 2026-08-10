# figure-templates

A modular publication-figure toolkit for single-cell, spatial and clinical
omics, in **Python (matplotlib)** and **R (ggplot2)**.

Volcano, UMAP/embedding, expression dotplot, pathway dotplot, heatmap,
lollipop, composition bars, chord, UpSet, sankey — plus a JAMA-style table-forest,
Kaplan-Meier curves, regression scatter and dumbbell. Each is a small function
that takes an `ax` (or returns a `ggplot`), reads its colours from a swappable
**theme**, and carries the design decisions that make the plot honest.

Pulled out of the Kundishora Lab's brain-AVM manuscript pipelines and
generalised, so the same visual language can be reused on an unrelated dataset.
**No data ships with it** — every panel in the gallery runs on synthetic
inputs, so you can clone it and see all 18 immediately.

```bash
python demo/gallery.py --theme avm        # render every panel type -> gallery/
Rscript demo/gallery.R --theme avm        # the R track -> gallery_R/
Rscript demo/gallery_clinical.R --theme avm   # forest / KM / regression / dumbbell
python demo/build_gallery_html.py         # bundle them into gallery/index.html
```

Open `gallery/index.html` to see all 18 panels with the call that made each.

The gallery renders at 200dpi/PNG because it's a *reference*. For real panels
use the library defaults (600dpi, PNG+SVG) — or
`--dpi 600 --formats png,svg` to see what they look like.

---

## Quick start

```python
import matplotlib.pyplot as plt
from figkit import apply_rcparams, save_panel
from figkit.plots import volcano, NOISE_GENE_PATTERNS

apply_rcparams()                              # once, at script start
fig, ax = plt.subplots(figsize=(6.5, 5.5))
volcano(de_table, ax,
        x_col="log2FoldChange", y_col="pvalue", label_col="gene",
        lfc_thresh=0.8, p_thresh=0.01,
        exclude_patterns=NOISE_GENE_PATTERNS)
save_panel(fig, "fig1a_volcano", "panels/")   # -> panels/fig1a_volcano.{png,svg}
```

```r
source("R/figkit.R")
p <- fk_volcano(de_table, x_col = "log2FoldChange", y_col = "pvalue")
fk_save_panel(p, "fig1a_volcano", "panels/", width = 6.5, height = 5.5)
```

## Retheming — the point of the whole thing

Every plot reads its colours from one `Theme` object. Swap it and the whole
figure set follows; you never edit hex codes in individual renderers.

```python
from figkit import set_theme, BASE_THEME
from figkit.themes.avm import AVM_THEME

set_theme(AVM_THEME)                       # install globally
set_theme(BASE_THEME.derive(up="#d62728")) # or tweak one field
volcano(df, ax, theme=AVM_THEME)           # or override per call
```

`figkit/themes/avm.py` is a **worked example, not a dependency** — nothing in
`figkit/` imports it. Copy it to `themes/<yours>.py`, replace the vocabularies,
and delete the AVM one. It's annotated with why each choice was made.

A theme carries the directional contrast (`up`/`down`), a categorical list, and
named domain palettes:

```python
theme.palette("cell_type")   # {"Astrocytes": "#117733", ...}
theme.diverging              # down -> neutral -> up colormap
theme.sequential             # white -> up colormap
```

## What's here

| Module | Functions |
|---|---|
| `figkit.plots.volcano` | `volcano`, `NOISE_GENE_PATTERNS`, `CONTROL_PROBE_PATTERNS` |
| `figkit.plots.dotplot` | `dotplot_pathway`, `dotplot_expression`, `expression_matrix` |
| `figkit.plots.embedding` | `embedding_categorical`, `embedding_continuous`, `two_layer_scatter`, `highlight_mask`, `add_scale_bar`, `get_embedding` |
| `figkit.plots.heatmap` | `heatmap` |
| `figkit.plots.lollipop` | `lollipop` |
| `figkit.plots.bars` | `stacked_bar`, `ordered_bar` |
| `figkit.plots.network` | `chord`, `chord_signed`, `upset` |
| `figkit.plots.sankey` | `sankey` |
| `figkit` | `apply_rcparams`, `save_panel`, `despine`, `size_legend`, `quantile_norm`, palettes |

R mirrors these as `fk_*` (`fk_volcano`, `fk_dotplot_pathway`, `fk_umap_categorical`,
`fk_lollipop`, `fk_stacked_bar`, `fk_ordered_bar`, `fk_heatmap`, `fk_save_panel`)
plus `fk_theme_pub` / `fk_theme_umap` / `fk_umap_arrows`.

**R-only** — clinical panels with no Python twin (the forest is a hand-rolled
ggplot layout engine; KM leans on survminer):

| Module | Functions |
|---|---|
| `R/forest.R` | `fk_table_forest`, `fk_forest_meta`, `fk_forest_general`, `fk_forest_compact_*` |
| `R/clinical.R` | `fk_km_panel`, `fk_save_km_panel`, `fk_regression_scatter`, `fk_dumbbell` |

```r
source("R/figkit.R"); source("R/forest.R"); source("R/clinical.R")

# JAMA table-forest. NOTE: data$label renders automatically as the left-most
# column — `cols` lists only the columns AFTER it.
fk_forest_general(df, cols = list(list(col = "n_label", header = "N")),
                  x_lab = "Odds ratio (95% CI)", label_header = "Predictor")

km <- fk_km_panel(df, "age", "event", "group", xlab = "Age (years)")
fk_save_km_panel("panels/", "fig1d_km", km, w = 5.5, h = 5)
```

Every Python plot takes an `ax` and returns a summary dict, so panels compose
into a figure with plain `plt.subplots` / `GridSpec`. (`network.py` is the
exception — pycirclize and upsetplot build their own figures.)

## Design decisions baked in

These are the reason this is a library and not a pile of snippets. Each one
fixes a failure mode that produced a wrong-looking figure at least once:

- **Editable vector text.** `pdf.fonttype=42`, `svg.fonttype="none"` (and
  `cairo_pdf` in R) keep text as text, so a typo is an Illustrator edit rather
  than a pipeline re-run. Most journals require it.
- **Volcano p-floor from the data**, not a constant — a fixed floor stacks
  every `p≈0` gene into a fake spike at an arbitrary height.
- **Volcano tie-jitter** (deterministic) — DESeq2 returns identical stats for
  groups of low-count genes; without jitter dozens hide under one dot.
- **NA ≠ zero.** Rows with no p-value are dropped, not treated as `p=1`; NaN
  heatmap cells render grey, never as a colormap value.
- **Diverging scales centre on zero.** An off-centre midpoint marks a
  meaningless value as "no change". `quantile_norm(symmetric=True)` enforces it.
- **Size and colour encode different things.** Effect size and significance
  stay on separate channels; every size encoding gets a `size_legend`.
- **Two-layer scatters keep the full backdrop** — without it the reader can't
  tell "rare" from "we only measured a few".
- **`ordered_bar` won't re-sort by value**, so an anatomical or dose axis keeps
  its meaning; categories outside `order` are dropped *loudly*.
- **No significance stars on bars.** Stars, `n=`, and stat text drawn on bars
  belong in the table or caption: the bar carries the estimate, the text
  carries the inference. `ordered_bar` has no option for them.
- **`lollipop` selects by |value|, displays by signed value** — ranking by raw
  value silently returns only the up side.
- **Equal aspect on embeddings.** A stretched UMAP rescales the distances the
  plot exists to show.
- **Filtering is opt-in and prints what it dropped.** "Which genes are noise"
  is a per-assay call; silently dropping rows from someone else's DE table is
  the wrong default.
- **`sankey` colours by source, not target.** A sankey is read left to right
  and answers "where did each of these go"; colouring by destination turns it
  into a different, worse plot that answers nothing the target axis doesn't
  already say.

## Caveats worth knowing

- **`normalize=True` on `stacked_bar` hides each group's n.** Put it in the
  label (`"Sample A (n=812)"`) — a 50% from 4 cells must not look like a 50%
  from 4000. The gallery does this.
- **`dotplot_expression` does not normalise for you.** Pass an already
  log1p'd matrix; re-transforming an already-normalised matrix is a silent
  double-transform bug.
- **`dotplot_expression(scale=True)` shows rank, not level.** It is Seurat's
  `DotPlot(scale = TRUE)` — z-score per gene across groups, clipped to ±2.5 —
  so the colour answers "which group is this gene highest in" and no longer
  answers "how much of it is there". Each z-score is estimated from as many
  observations as there are groups, so drop tiny groups *before* scaling; a
  12-cell group otherwise moves every gene's mean. Two groups is degenerate
  (every gene reads ±0.707).
- **`scale=True` with a diverging cmap wants `center=True`.** Scaled expression
  is skewed by construction: with k groups, a gene specific to one puts k-1
  values just below zero and one at the clip. Left uncentred, the ramp's
  midpoint colour lands wherever the data put it, so a diverging cmap's neutral
  stops meaning "average". `center=True` pins zero to the midpoint with a
  `TwoSlopeNorm`. Leave it off for a sequential cmap, which has no midpoint to
  honour.
- **`top_n` selection is selection bias.** A top-DEG dotplot looks separated
  even under a null. If the contrast is near-null, say so in the caption.
- **Nominal vs adjusted p.** `volcano` doesn't care which you pass — small-n
  pseudobulk often saturates padj≈1 and flattens the plot. If you pass nominal
  p, label the axis and caption accordingly (`y_label` exists for this).
- **Ribbon width is only good to about 20%.** `sankey` is the right panel for
  "almost all of A became B" and the wrong one for 31% versus 27%. If the
  reader needs to compare two ribbons that are not adjacent, they need a
  heatmap; several published sankeys are heatmaps that lost their numbers.
- **`sankey(normalize="source")` hides each source's n**, the same trade
  `stacked_bar` makes — it is usually the right one for a label transfer,
  because otherwise the largest cluster is the only legible one. Put the n in
  the node label or the caption.
- Fonts fall back to Helvetica/DejaVu if Arial is missing; panels stay
  reproducible but metrics shift slightly.

## Install

```bash
pip install -r requirements.txt        # matplotlib, pandas, numpy, adjustText
pip install pycirclize upsetplot       # optional: chord + upset panels
```

`anndata`/`scanpy` are only needed for `expression_matrix()`; every other
function takes plain DataFrames and arrays.

R needs `ggplot2`; `ggrepel` (volcano labels) and `patchwork` (composites) are
optional and degrade with a message rather than failing.

## Layout

```
figkit/            the library
  theme.py         Theme object, set_theme/get_theme
  palettes.py      colormaps, Kelly/Wong, quantile_norm
  style.py         apply_rcparams, save_panel, despine, size_legend
  plots/           one module per plot family
  themes/avm.py    worked example domain theme — copy, don't import
R/
  figkit.R         ggplot2 mirror
  themes_avm.R     worked example domain theme
demo/
  synth.py         seeded synthetic generators (DE, pathway, TF, UMAP, spatial)
  gallery.py       renders every panel type
  gallery.R        the R track
  build_gallery_html.py
gallery/           rendered panels + index.html
```

## Provenance

This is the figure layer of the Kundishora Lab's brain-AVM work — spatial
transcriptomics and the genotype–phenotype cohort study — pulled out of those
pipelines and generalised. The design decisions listed above are the accumulated
result of taking those figures through review; each one is here because getting
it wrong produced a misleading panel at least once.

Companion repo: [bAVM-genotype-phenotype](https://github.com/KundishoraLab/bAVM-genotype-phenotype).

Names from the original code survive as aliases, so existing scripts keep
working: `dotplot_gsea`, `heatmap_complex`, `lollipop_tf`, `stacked_hbar`.

The table-forest is ported faithfully from the manuscript pipeline, with four
deliberate changes:

- `point_col` follows the theme rather than a hardcoded blue.
- `label_header` is new. The first column's header used to be hardcoded to
  `"Predictor"`, which is wrong for a meta-analysis forest ("Study").
- `size_col` strips thousands separators before coercing to numeric. A display
  N of `"1,204"` previously became `NA` and silently dropped that row's dot
  size — the row still plotted, just with no size mapping.
- **Bug fix:** passing `left_label`/`right_label` clipped the x-axis title off
  the panel. The lower y-limit was derived from the directional labels alone,
  leaving it above the title; it is now the minimum of both. ggplot surfaced
  this only as a generic "Removed 1 row containing missing values", which is
  the kind of thing this repo exists to stop repeating.

## License

MIT — see [LICENSE](LICENSE).
