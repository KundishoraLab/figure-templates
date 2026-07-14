# figure-design-kit

A modular publication-figure toolkit for single-cell / spatial omics, in
**Python (matplotlib)** and **R (ggplot2)**.

Volcano, UMAP/embedding, expression dotplot, pathway dotplot, heatmap,
lollipop, composition bars, chord, UpSet — each a small function that takes an
`ax` (or returns a `ggplot`), reads its colours from a swappable **theme**, and
carries the design decisions that make the plot honest.

Extracted from a working brain-AVM spatial-transcriptomics manuscript pipeline
and generalised. **No data ships with it** — the gallery runs entirely on
synthetic inputs.

```bash
python demo/gallery.py --theme avm       # render every panel type -> gallery/
python demo/build_gallery_html.py        # bundle them into gallery/index.html
Rscript demo/gallery.R --theme avm       # the R track -> gallery_R/
```

Open `gallery/index.html` to see all 13 panels with the call that made each.

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
| `figkit` | `apply_rcparams`, `save_panel`, `despine`, `size_legend`, `quantile_norm`, palettes |

R mirrors these as `fk_*` (`fk_volcano`, `fk_dotplot_pathway`, `fk_umap_categorical`,
`fk_lollipop`, `fk_stacked_bar`, `fk_ordered_bar`, `fk_heatmap`, `fk_save_panel`)
plus `fk_theme_pub` / `fk_theme_umap` / `fk_umap_arrows`.

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
- **`lollipop` selects by |value|, displays by signed value** — ranking by raw
  value silently returns only the up side.
- **Equal aspect on embeddings.** A stretched UMAP rescales the distances the
  plot exists to show.
- **Filtering is opt-in and prints what it dropped.** "Which genes are noise"
  is a per-assay call; silently dropping rows from someone else's DE table is
  the wrong default.

## Caveats worth knowing

- **`normalize=True` on `stacked_bar` hides each group's n.** Put it in the
  label (`"Sample A (n=812)"`) — a 50% from 4 cells must not look like a 50%
  from 4000. The gallery does this.
- **`dotplot_expression` does not normalise for you.** Pass an already
  log1p'd matrix; re-transforming an already-normalised matrix is a silent
  double-transform bug.
- **`top_n` selection is selection bias.** A top-DEG dotplot looks separated
  even under a null. If the contrast is near-null, say so in the caption.
- **Nominal vs adjusted p.** `volcano` doesn't care which you pass — small-n
  pseudobulk often saturates padj≈1 and flattens the plot. If you pass nominal
  p, label the axis and caption accordingly (`y_label` exists for this).
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

Extracted from `avm-spatial-tx` (`scripts/05_scrna_niche_analysis/_style.py`,
`scripts/07_manuscript_figures/_helpers/_style_fig456.py` and `utils.R`), then
decoupled from that project's paths, HPC mounts and cohort vocabulary. The
expression dotplot was promoted out of a single figure renderer into a real
function. Original behaviour is preserved; original names survive as aliases
(`dotplot_gsea`, `heatmap_complex`, `lollipop_tf`, `stacked_hbar`).
