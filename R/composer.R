# =============================================================================
# composer.R — figkit composite-figure layer.
# Turn per-panel .rds objects (saved by fk_save_panel) into a publication
# multi-panel figure: one unified theme, dedicated legend cells, a..z tags,
# vector PDF/PNG/SVG. Ported from the brain-avm manuscript composers.
#
# ── THE WORKFLOW ─────────────────────────────────────────────────────────────
#   1. Producers build panels with their review theme and call fk_save_panel(),
#      which also writes <name>.rds (the raw ggplot object).
#   2. A per-figure composer script:
#        source figkit; set fk_panel_dirs()
#        p <- fk_panel("token")                     # loads .rds, re-themes
#        L <- fk_legend_cell(fk_panel_legend("token"))
#        design <- "AAB\nCCB"                        # patchwork ASCII grid
#        fig <- p1 + p2 + L + plot_layout(design=design) +
#               plot_annotation(tag_levels=list(c("a","b",""))) & fk_tag_theme()
#        fk_save_composite(fig, "results/Fig1")
#
# ── HARD-WON LESSONS (read before composing) ─────────────────────────────────
#   * fk_theme_nature_panel() is theme_classic-based and REPLACES the panel
#     theme, so it RESETS axis-text angle, strip.text size, etc. Any angled
#     axis labels or custom strip sizes MUST be re-applied in the composer
#     AFTER fk_panel(), e.g.  fk_panel("x") + theme(axis.text.x=element_text(angle=45)).
#   * Standalone panels use point/text sizes tuned for solo viewing; they render
#     too big in a small grid cell. fk_panel() runs fk_scale_marks() to shrink
#     fixed point/text sizes. Mapped sizes (dotplot %expr) need the producer to
#     pick a small scale_size() range instead.
#   * Put legends in dedicated cells ADJACENT to what they explain (a legend
#     column, or a cell beside the panel) — never patchwork guides="collect" at
#     the bottom. Extract them from FRESH themed copies via fk_panel_legend().
#   * patchwork matches plots to `design` areas ALPHABETICALLY by area letter
#     (uppercase A-Z sort before lowercase a-z), so add plots in that order.
#   * Prefer coloured axis-end direction labels (annotate) over a legend for a
#     simple up/down split — it frees the panel width.
#   * Title Case category/legend/axis labels; wrap long x labels onto two lines.
#   * Wrap a panel whose wide categorical axis shoves its neighbours in
#     patchwork::free().
#   * Captions are review-only; strip them at compose time (plot.caption=blank).
# =============================================================================

# Nature sizing spec (single source of truth)
NM <- list(
  width_in = 7.20, height_in = 6.69,   # 183 mm double-column x 170 mm max
  body_pt = 7, tick_pt = 6, label_pt = 8,
  font_family = if (requireNamespace("systemfonts", quietly = TRUE) &&
                    "Arial" %in% systemfonts::system_fonts()$family) "Arial" else ""
)

# composer-side theme: replaces each producer's review theme so fonts are identical
fk_theme_nature_panel <- function() {
  ggplot2::theme_classic(base_size = NM$body_pt, base_family = NM$font_family) +
    ggplot2::theme(
      axis.text        = ggplot2::element_text(size = NM$tick_pt, colour = "black"),
      axis.title       = ggplot2::element_text(size = NM$body_pt, colour = "black"),
      legend.text      = ggplot2::element_text(size = NM$tick_pt),
      legend.title     = ggplot2::element_text(size = NM$body_pt, face = "bold"),
      legend.key.size  = ggplot2::unit(0.28, "cm"),
      strip.text       = ggplot2::element_text(size = NM$body_pt, face = "bold"),
      plot.margin      = ggplot2::margin(2, 4, 2, 2, "pt"),
      plot.tag         = ggplot2::element_text(size = NM$label_pt, face = "bold",
                                               family = NM$font_family))
}
# theme applied across the whole composite via `& fk_tag_theme()`
fk_tag_theme <- function() ggplot2::theme(
  plot.tag     = ggplot2::element_text(size = NM$label_pt, face = "bold", family = NM$font_family),
  plot.caption = ggplot2::element_blank(),
  plot.margin  = ggplot2::margin(2, 3, 2, 3, "pt"))

# where fk_load_panel searches for <name>.rds (set once per project/composer)
.fk_panel_dirs <- new.env(parent = emptyenv()); .fk_panel_dirs$dirs <- "figures"
fk_panel_dirs <- function(dirs) { .fk_panel_dirs$dirs <- dirs; invisible(dirs) }

fk_load_panel <- function(name) {
  cands <- c(file.path(.fk_panel_dirs$dirs, paste0(name, ".rds")),
             file.path(.fk_panel_dirs$dirs, paste0(basename(name), ".rds")),
             paste0(name, ".rds"))
  hit <- cands[file.exists(cands)][1]
  if (is.na(hit)) stop("fk_load_panel: no .rds for '", name, "' (re-render its producer)")
  readRDS(hit)
}

# shrink FIXED (aes_param) point / text sizes for composite scale
fk_scale_marks <- function(p, pt = 0.55, txt = 0.7) {
  if (!inherits(p, "ggplot")) return(p)
  for (i in seq_along(p$layers)) {
    g <- class(p$layers[[i]]$geom)[1]; ap <- p$layers[[i]]$aes_params
    if (!is.null(ap$size)) {
      if      (grepl("Point|Jitter", g)) p$layers[[i]]$aes_params$size <- ap$size * pt
      else if (grepl("Text|Label",   g)) p$layers[[i]]$aes_params$size <- ap$size * txt
    }
  }
  p
}

.fk_strip   <- ggplot2::theme(legend.position = "none")
.fk_no_grid <- ggplot2::theme(panel.grid.major = ggplot2::element_blank(),
                              panel.grid.minor = ggplot2::element_blank())

# themed, legend-stripped, gridless, mark-scaled panel ready for the grid
fk_panel <- function(name, extra = NULL, pt = 0.55, txt = 0.7, grid = FALSE) {
  p <- fk_scale_marks(fk_load_panel(name), pt = pt, txt = txt) + fk_theme_nature_panel() + .fk_strip
  if (!grid) p <- p + .fk_no_grid
  if (!is.null(extra)) p <- p + extra
  p
}
# a legend grob from a FRESH themed copy (right/vertical), for a dedicated cell
fk_panel_legend <- function(name, keep = NULL) {
  p <- fk_load_panel(name) + fk_theme_nature_panel() +
    ggplot2::theme(legend.position = "right", legend.direction = "vertical",
                   legend.margin = ggplot2::margin(0, 0, 0, 0))
  if (!is.null(keep)) p <- p + keep
  cowplot::get_legend(p)
}
# stack 1..n legend grobs vertically into one patchwork element (dedicated cell)
fk_legend_cell <- function(...) {
  legs <- Filter(Negate(is.null), list(...))
  if (!length(legs)) return(patchwork::plot_spacer())
  patchwork::wrap_elements(full = cowplot::plot_grid(plotlist = legs, ncol = 1, align = "v"))
}
# an external raster (schematic / spatial PNG) as a patchwork element
fk_panel_png <- function(path)
  patchwork::wrap_elements(full = grid::rasterGrob(png::readPNG(path), interpolate = TRUE))

# a watermarked placeholder box for a not-yet-drawn schematic / graphical abstract cell
fk_placeholder <- function(label = "Graphical Abstract", watermark = "TEMPLATE") {
  ggplot2::ggplot() +
    ggplot2::annotate("rect", xmin = 0, xmax = 1, ymin = 0, ymax = 1,
                      fill = "grey97", colour = "grey65", linewidth = 0.5) +
    ggplot2::annotate("text", x = 0.5, y = 0.5, label = watermark, angle = 16,
                      size = 11, colour = "grey88", fontface = "bold", family = NM$font_family) +
    ggplot2::annotate("text", x = 0.5, y = 0.5, label = label,
                      size = 5, colour = "grey40", fontface = "bold", family = NM$font_family) +
    ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    ggplot2::theme_void()
}

# save composite as PDF (cairo) + PNG + SVG (svglite, native text for Illustrator)
fk_save_composite <- function(composite, stem, width = NM$width_in, height = NM$height_in) {
  dir.create(dirname(stem), recursive = TRUE, showWarnings = FALSE)
  fam <- if (nzchar(NM$font_family)) NM$font_family else NULL
  ggplot2::ggsave(paste0(stem, ".pdf"), composite, width = width, height = height,
                  device = grDevices::cairo_pdf, family = fam)
  ggplot2::ggsave(paste0(stem, ".png"), composite, width = width, height = height, dpi = 300)
  if (requireNamespace("svglite", quietly = TRUE))
    ggplot2::ggsave(paste0(stem, ".svg"), composite, width = width, height = height,
                    device = svglite::svglite, fix_text_size = FALSE)
  message(sprintf("  composite -> %s.{pdf,png,svg}  (%.2f x %.2f in)", stem, width, height))
}
