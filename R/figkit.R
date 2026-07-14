## figkit.R — ggplot2 mirror of the Python figkit API.
##
##   source("R/figkit.R")
##   fk_set_theme(fk_theme_avm())
##   p <- fk_volcano(de, x_col = "log2FoldChange", y_col = "pvalue")
##   fk_save_panel(p, "panel_a_volcano", "panels/", width = 6, height = 5)
##
## Function names and arguments deliberately track the Python side
## (fk_volcano / volcano, fk_dotplot_pathway / dotplot_pathway, ...) so a
## figure set split across both languages stays one visual system.
##
## Deps: ggplot2. Optional: ggrepel (volcano labels), patchwork (composites).

suppressPackageStartupMessages({
  library(ggplot2)
})

# ── Theme object ────────────────────────────────────────────────────────────
# Mirrors figkit.theme.Theme. A plain list, so a project can override one
# field without subclassing anything.
fk_theme <- function(name = "base",
                     up = "#c51b8a",
                     down = "#1f3a6e",
                     neutral = "#f5f5f5",
                     nonsig = "#cccccc",
                     na = "#cccccc",
                     background = "#dcdcdc",
                     categorical = NULL,
                     font_family = "Arial",
                     base_size = 12,
                     palettes = list()) {
  if (is.null(categorical)) categorical <- FK_KELLY
  structure(list(
    name = name, up = up, down = down, neutral = neutral, nonsig = nonsig,
    na = na, background = background, categorical = categorical,
    font_family = font_family, base_size = base_size, palettes = palettes
  ), class = "fk_theme")
}

FK_KELLY <- c(
  "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032", "#C2B280", "#848482",
  "#008856", "#E68FAC", "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C",
  "#DCD300", "#882D17", "#8DB600", "#654522", "#E25822", "#2B3D26"
)

# Wong colorblind-safe. Prefer over Kelly at <= 8 categories.
FK_WONG <- c("#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
             "#DDCC77", "#CC6677", "#882255", "#AA4499")

fk_set_theme <- function(theme) {
  stopifnot(inherits(theme, "fk_theme"))
  options(figkit.theme = theme)
  invisible(theme)
}

fk_get_theme <- function() {
  t <- getOption("figkit.theme")
  if (is.null(t)) t <- fk_theme()
  t
}

fk_palette <- function(key, theme = fk_get_theme()) {
  if (!key %in% names(theme$palettes)) {
    stop(sprintf("theme '%s' has no palette '%s'; available: %s",
                 theme$name, key, paste(names(theme$palettes), collapse = ", ")))
  }
  theme$palettes[[key]]
}

# ── ggplot themes ───────────────────────────────────────────────────────────
# Journal display-item spec: body text 5-7pt at final print size. `base_size`
# is the axis-title size; ticks render one step down. Set it to the size the
# panel will be *printed* at, not the size it looks good on screen — text
# scaled down inside a composite is the most common way figures end up
# below a journal's text-size floor.
fk_theme_pub <- function(base_size = 12, family = NULL, grid = TRUE) {
  if (is.null(family)) family <- fk_get_theme()$font_family
  th <- theme_classic(base_size = base_size, base_family = family) +
    theme(
      plot.title      = element_text(face = "bold", size = base_size + 1),
      plot.subtitle   = element_text(colour = "grey40", size = base_size - 1),
      axis.title      = element_text(size = base_size, colour = "black"),
      axis.text       = element_text(size = base_size - 1, colour = "black"),
      strip.text      = element_text(face = "bold", size = base_size),
      legend.title    = element_text(size = base_size - 1, face = "bold"),
      legend.text     = element_text(size = base_size - 1),
      legend.key.size = unit(0.9, "lines"),
      plot.margin     = margin(4, 6, 4, 6),
      # theme_classic uses %+replace%, which nulls plot.tag — restate it or
      # patchwork's tag_levels panel letters silently vanish in composites.
      plot.tag        = element_text(face = "bold", size = base_size + 3)
    )
  if (grid) th <- th + theme(panel.grid.major = element_line(colour = "grey93"))
  th
}

# Embedding theme: blank axes, square aspect. A UMAP axis carries no units,
# and a non-square panel silently rescales the distances the plot exists to
# show — so aspect.ratio = 1 is not cosmetic.
fk_theme_umap <- function(base_size = 12, family = NULL) {
  fk_theme_pub(base_size = base_size, family = family, grid = FALSE) +
    theme(
      axis.text         = element_blank(),
      axis.ticks        = element_blank(),
      axis.ticks.length = unit(0, "pt"),
      axis.title        = element_blank(),
      axis.line         = element_blank(),
      panel.grid        = element_blank(),
      aspect.ratio      = 1
    )
}

# ── UMAP origin arrows ──────────────────────────────────────────────────────
# Draws the axis arrows + labels as data-coordinate annotations anchored at
# the bottom-left. Why not the theme's axis.line arrows: those span the whole
# panel axis, so their tips land at different physical positions whenever
# patchwork hands a panel a different-shaped cell — paired UMAPs end up with
# visibly mismatched arrows. Data-coord arrows share the coordinate space, so
# two panels of the same embedding always match.
fk_umap_arrows <- function(p, x_lab = "UMAP 1", y_lab = "UMAP 2",
                           base_size = 12, colour = "grey40",
                           arrow_frac = 0.18, arrow_head = 0.18) {
  bld <- ggplot_build(p)
  xr <- bld$layout$panel_params[[1]]$x.range
  yr <- bld$layout$panel_params[[1]]$y.range
  xs <- diff(xr); ys <- diff(yr)
  ox <- xr[1]; oy <- yr[1]
  tsize <- (base_size - 2) / ggplot2::.pt

  p +
    annotate("segment", x = ox, xend = ox + xs * arrow_frac, y = oy, yend = oy,
             arrow = grid::arrow(length = unit(arrow_head, "cm"),
                                 ends = "last", type = "closed"),
             colour = colour, linewidth = 0.5) +
    annotate("segment", x = ox, xend = ox, y = oy, yend = oy + ys * arrow_frac,
             arrow = grid::arrow(length = unit(arrow_head, "cm"),
                                 ends = "last", type = "closed"),
             colour = colour, linewidth = 0.5) +
    annotate("text", x = ox, y = oy - ys * 0.04, label = x_lab,
             hjust = 0, vjust = 1, size = tsize, colour = colour) +
    annotate("text", x = ox - xs * 0.04, y = oy, label = y_lab,
             hjust = 0, vjust = 0, angle = 90, size = tsize, colour = colour) +
    # Re-read the built ranges so any upstream coord_cartesian (e.g. shared
    # limits across a facet grid) survives this call instead of being reset.
    coord_cartesian(xlim = xr, ylim = yr, clip = "off") +
    theme(plot.margin = margin(6, 8, 12, 14))
}

# ── UMAP: categorical ───────────────────────────────────────────────────────
fk_umap_categorical <- function(df, color_col, x = "umap_1", y = "umap_2",
                                palette = NULL, point_size = 0.35,
                                alpha = 0.75, legend_title = NULL,
                                legend_ncol = 1, base_size = 12,
                                theme = fk_get_theme()) {
  if (is.null(palette)) {
    lv <- unique(as.character(df[[color_col]]))
    palette <- setNames(rep(theme$categorical, length.out = length(lv)), lv)
  }
  ggplot(df, aes(x = .data[[x]], y = .data[[y]],
                 colour = .data[[color_col]])) +
    geom_point(size = point_size, alpha = alpha, stroke = 0) +
    scale_colour_manual(values = palette, name = legend_title,
                        na.value = theme$na) +
    guides(colour = guide_legend(ncol = legend_ncol,
                                 override.aes = list(size = 2.8, alpha = 1))) +
    labs(x = "UMAP 1", y = "UMAP 2") +
    fk_theme_umap(base_size = base_size)
}

# ── UMAP: two-layer highlight ───────────────────────────────────────────────
# `background_df` must be the FULL set of cells, including the highlighted
# ones. Without the backdrop the reader cannot tell "rare" from "we only
# measured a few".
fk_umap_highlight <- function(background_df, highlight_df, color_col = NULL,
                              x = "umap_1", y = "umap_2", palette = NULL,
                              fg_colour = NULL, bg_colour = NULL,
                              bg_alpha = 0.35, bg_size = 0.25,
                              fg_size = 1.4, fg_alpha = 0.9,
                              facet_col = NULL, legend_title = NULL,
                              base_size = 12, theme = fk_get_theme()) {
  if (is.null(bg_colour)) bg_colour <- theme$background
  if (is.null(fg_colour)) fg_colour <- theme$up

  p <- ggplot() +
    geom_point(data = background_df, aes(x = .data[[x]], y = .data[[y]]),
               colour = bg_colour, alpha = bg_alpha, size = bg_size, stroke = 0)

  if (is.null(color_col)) {
    p <- p + geom_point(data = highlight_df, aes(x = .data[[x]], y = .data[[y]]),
                        colour = fg_colour, size = fg_size, alpha = fg_alpha,
                        stroke = 0)
  } else {
    p <- p + geom_point(data = highlight_df,
                        aes(x = .data[[x]], y = .data[[y]],
                            colour = .data[[color_col]]),
                        size = fg_size, alpha = fg_alpha, stroke = 0) +
      scale_colour_manual(values = palette, name = legend_title,
                          na.value = theme$na)
  }
  p <- p + labs(x = "UMAP 1", y = "UMAP 2") +
    fk_theme_umap(base_size = base_size)
  if (!is.null(facet_col)) p <- p + facet_wrap(as.formula(paste("~", facet_col)))
  p
}

# ── Volcano ─────────────────────────────────────────────────────────────────
FK_NOISE_PATTERNS <- c("^MT-", "^MTRNR", "^A[CLPF]\\d+\\.\\d+$", "^LINC\\d+$",
                       "^MIR\\d+", "^RNU\\d+", "^RP[LS]\\d+[A-Z]?$", "^\\d+$",
                       "-ENSG\\d+$", "-AS\\d+$")
FK_CONTROL_PATTERNS <- c("SystemControl", "Negative", "FalseCode")

fk_volcano <- function(df, x_col = "log2fc", y_col = "pvalue",
                       label_col = "gene", lfc_thresh = 0.5, p_thresh = 0.05,
                       top_n_labels = 14, label_genes = NULL,
                       exclude_patterns = NULL, point_size = 0.7,
                       point_alpha = 0.85, base_size = 12,
                       x_label = NULL, y_label = NULL,
                       theme = fk_get_theme()) {
  d <- as.data.frame(df)
  d[[x_col]] <- suppressWarnings(as.numeric(d[[x_col]]))
  d[[y_col]] <- suppressWarnings(as.numeric(d[[y_col]]))

  if (!is.null(exclude_patterns)) {
    drop <- Reduce(`|`, lapply(exclude_patterns, function(p)
      grepl(p, d[[label_col]], ignore.case = TRUE)))
    if (any(drop)) message(sprintf("  [volcano] excluded %d rows", sum(drop)))
    d <- d[!drop, , drop = FALSE]
  }
  n0 <- nrow(d)
  d <- d[!is.na(d[[x_col]]) & !is.na(d[[y_col]]), , drop = FALSE]
  if (n0 - nrow(d) > 0)
    message(sprintf("  [volcano] dropped %d rows with NA stats", n0 - nrow(d)))
  if (!nrow(d)) stop("fk_volcano: no finite rows to plot")

  # Floor from the data, not a constant — see the Python docstring.
  nz <- d[[y_col]][d[[y_col]] > 0]
  p_floor <- if (length(nz)) min(nz) * 0.5 else 1e-300
  d$.nlp <- -log10(pmax(d[[y_col]], p_floor))
  d$.dir <- ifelse(d[[y_col]] < p_thresh & abs(d[[x_col]]) >= lfc_thresh,
                   ifelse(d[[x_col]] > 0, "up", "down"), "ns")

  lab <- if (!is.null(label_genes)) {
    d[as.character(d[[label_col]]) %in% as.character(label_genes), ]
  } else if (top_n_labels > 0) {
    sig <- d[d$.dir != "ns", ]
    if (nrow(sig)) {
      sig$.score <- abs(sig[[x_col]]) * sig$.nlp
      n_each <- max(1, top_n_labels %/% 2)
      rbind(utils::head(sig[sig$.dir == "up", ][order(-sig[sig$.dir == "up", ]$.score), ], n_each),
            utils::head(sig[sig$.dir == "down", ][order(-sig[sig$.dir == "down", ]$.score), ], n_each))
    } else sig
  } else d[0, ]

  sym <- max(abs(d[[x_col]]), na.rm = TRUE) * 1.05
  cols <- c(up = theme$up, down = theme$down, ns = theme$nonsig)

  p <- ggplot(d, aes(x = .data[[x_col]], y = .data$.nlp)) +
    geom_point(aes(colour = .data$.dir), size = point_size, alpha = point_alpha,
               stroke = 0) +
    scale_colour_manual(values = cols, guide = "none") +
    geom_hline(yintercept = -log10(p_thresh), linetype = "dotted",
               colour = "#888", linewidth = 0.3) +
    geom_vline(xintercept = c(-lfc_thresh, lfc_thresh), linetype = "dotted",
               colour = "#888", linewidth = 0.3) +
    scale_x_continuous(limits = c(-sym, sym)) +
    labs(x = if (is.null(x_label)) "log2 fold change" else x_label,
         y = if (is.null(y_label)) "-log10 p" else y_label) +
    fk_theme_pub(base_size = base_size, grid = FALSE)

  if (nrow(lab)) {
    if (requireNamespace("ggrepel", quietly = TRUE)) {
      p <- p + ggrepel::geom_text_repel(
        data = lab, aes(label = .data[[label_col]], colour = .data$.dir),
        size = base_size / 3.5, fontface = "plain", max.overlaps = Inf,
        segment.colour = "#888", segment.size = 0.3, show.legend = FALSE)
    } else {
      message("  [volcano] ggrepel not installed; labels may overlap")
      p <- p + geom_text(data = lab, aes(label = .data[[label_col]],
                                         colour = .data$.dir),
                         size = base_size / 3.5, vjust = -0.6,
                         show.legend = FALSE)
    }
  }
  p
}

# ── Pathway dotplot ─────────────────────────────────────────────────────────
fk_prettify_pathway <- function(x, max_len = 50) {
  s <- as.character(x)
  s <- sub("^(HALLMARK|REACTOME|KEGG|BIOCARTA|GOBP|GOCC|GOMF|GO|WP|PID)_", "", s)
  s <- gsub("_", " ", s)
  s <- paste0(toupper(substring(s, 1, 1)), tolower(substring(s, 2)))
  ifelse(nchar(s) > max_len, paste0(substr(s, 1, max_len - 1), "…"), s)
}

fk_dotplot_pathway <- function(df, pathway_col = "pathway", nes_col = "NES",
                               padj_col = "padj", top_n = 20,
                               sort_by = c("significance", "nes", "signed"),
                               base_size = 12, x_label = "NES",
                               prettify = TRUE, theme = fk_get_theme()) {
  sort_by <- match.arg(sort_by)
  d <- as.data.frame(df)
  d[[padj_col]] <- pmax(suppressWarnings(as.numeric(d[[padj_col]])), 1e-300)
  d[[nes_col]] <- suppressWarnings(as.numeric(d[[nes_col]]))
  d <- d[!is.na(d[[padj_col]]) & !is.na(d[[nes_col]]), , drop = FALSE]
  if (!nrow(d)) stop("fk_dotplot_pathway: no finite rows to plot")
  d$.nlp <- -log10(d[[padj_col]])

  d <- switch(sort_by,
    significance = utils::head(d[order(-d$.nlp), ], top_n),
    nes          = utils::head(d[order(-abs(d[[nes_col]])), ], top_n),
    signed       = {
      h <- max(1, top_n %/% 2)
      unique(rbind(utils::head(d[order(-d[[nes_col]]), ], h),
                   utils::head(d[order(d[[nes_col]]), ], h)))
    })

  d$.label <- if (prettify) fk_prettify_pathway(d[[pathway_col]]) else
    as.character(d[[pathway_col]])
  d$.label <- factor(d$.label, levels = d$.label[order(d[[nes_col]])])
  lim <- max(abs(d[[nes_col]]), na.rm = TRUE)

  ggplot(d, aes(x = .data[[nes_col]], y = .data$.label)) +
    geom_vline(xintercept = 0, linetype = "dotted", colour = "#888",
               linewidth = 0.3) +
    geom_point(aes(size = .data$.nlp, fill = .data[[nes_col]]), shape = 21,
               colour = "black", stroke = 0.3) +
    # Diverging fill MUST be centred on zero, else the midpoint colour marks
    # a meaningless value and reads as "no change".
    scale_fill_gradient2(low = theme$down, mid = theme$neutral,
                         high = theme$up, midpoint = 0,
                         limits = c(-lim, lim), name = x_label) +
    scale_size_continuous(range = c(1.5, 6), name = "-log10 FDR") +
    labs(x = x_label, y = NULL) +
    fk_theme_pub(base_size = base_size, grid = FALSE)
}

# ── Lollipop ────────────────────────────────────────────────────────────────
fk_lollipop <- function(df, label_col = "source", value_col = "delta",
                        q_col = "q", top_n = 30, q_sig = 0.05,
                        y_label = "Δ activity", base_size = 12,
                        theme = fk_get_theme()) {
  d <- as.data.frame(df)
  d[[value_col]] <- suppressWarnings(as.numeric(d[[value_col]]))
  d <- d[!is.na(d[[value_col]]), , drop = FALSE]
  if (!nrow(d)) stop("fk_lollipop: no finite rows to plot")
  # Select by |value| then display by signed value — ranking by raw value
  # would return only the up side whenever any exist.
  d <- utils::head(d[order(-abs(d[[value_col]])), ], top_n)
  d[[label_col]] <- factor(as.character(d[[label_col]]),
                           levels = as.character(d[[label_col]])[order(-d[[value_col]])])
  d$.dir <- ifelse(d[[value_col]] > 0, "up", "down")
  has_q <- !is.null(q_col) && q_col %in% names(d)
  if (has_q) d$.nlq <- -log10(pmax(suppressWarnings(as.numeric(d[[q_col]])), 1e-300))

  p <- ggplot(d, aes(x = .data[[label_col]], y = .data[[value_col]])) +
    geom_segment(aes(xend = .data[[label_col]], y = 0, yend = .data[[value_col]]),
                 colour = "#999", linewidth = 0.3) +
    geom_hline(yintercept = 0, colour = "#444", linewidth = 0.3)
  p <- p + (if (has_q)
    geom_point(aes(fill = .data$.dir, size = .data$.nlq), shape = 21,
               colour = "black", stroke = 0.3)
  else
    geom_point(aes(fill = .data$.dir), shape = 21, size = 3, colour = "black",
               stroke = 0.3))
  p <- p +
    scale_fill_manual(values = c(up = theme$up, down = theme$down),
                      guide = "none") +
    labs(x = NULL, y = y_label) +
    fk_theme_pub(base_size = base_size, grid = FALSE) +
    theme(axis.text.x = element_text(angle = 60, hjust = 1, size = base_size - 3))
  if (has_q) {
    p <- p + scale_size_continuous(range = c(1.5, 6), name = "-log10 q")
    sig <- d[d[[q_col]] < q_sig, , drop = FALSE]
    if (nrow(sig)) {
      top <- max(d[[value_col]]) + 0.10 * diff(range(d[[value_col]]))
      p <- p + geom_point(data = sig, aes(x = .data[[label_col]], y = top),
                          shape = 8, size = 1.2, colour = "black",
                          inherit.aes = FALSE)
    }
  }
  p
}

# ── Stacked composition bar ─────────────────────────────────────────────────
# Normalizing hides each group's n. Put it in the group label
# ("Sample A (n=812)") — a 50% built from 4 cells must not look like a 50%
# built from 4000.
fk_stacked_bar <- function(df, group_col = "group", cat_col = "category",
                           value_col = "value", palette = NULL,
                           normalize = TRUE, legend_title = NULL,
                           x_label = NULL, base_size = 12,
                           theme = fk_get_theme()) {
  d <- as.data.frame(df)
  if (is.null(palette)) {
    lv <- unique(as.character(d[[cat_col]]))
    palette <- setNames(rep(theme$categorical, length.out = length(lv)), lv)
  }
  if (is.null(x_label)) x_label <- if (normalize) "fraction" else "count"
  pos <- if (normalize) "fill" else "stack"

  ggplot(d, aes(y = .data[[group_col]], x = .data[[value_col]],
                fill = .data[[cat_col]])) +
    geom_col(position = pos, width = 0.72, colour = "white", linewidth = 0.2) +
    scale_fill_manual(values = palette, name = legend_title,
                      na.value = theme$na) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.02))) +
    labs(x = x_label, y = NULL) +
    fk_theme_pub(base_size = base_size, grid = FALSE)
}

# ── Ordered bar ─────────────────────────────────────────────────────────────
# `order` exists so a categorical axis with real structure (anatomical,
# developmental, a dose ladder) keeps it instead of being re-sorted by value.
#
# Deliberately has no significance-annotation option: stars, n=, and stat text
# drawn on bars belong in the table or caption instead.
fk_ordered_bar <- function(df, value_col, label_col = "cell_type",
                           order = NULL, palette = NULL,
                           x_label = "", base_size = 12,
                           theme = fk_get_theme()) {
  d <- as.data.frame(df)
  d[[label_col]] <- as.character(d[[label_col]])
  if (!is.null(order)) {
    keep <- d[[label_col]] %in% order
    if (any(!keep))
      message(sprintf("  [ordered_bar] dropped %d rows not in `order`", sum(!keep)))
    d <- d[keep, , drop = FALSE]
    d[[label_col]] <- factor(d[[label_col]],
                             levels = rev(order[order %in% d[[label_col]]]))
  }
  if (!nrow(d)) stop("fk_ordered_bar: no rows left to plot")
  if (is.null(palette)) {
    lv <- levels(d[[label_col]]) %||% unique(d[[label_col]])
    palette <- setNames(rep(theme$categorical, length.out = length(lv)), lv)
  }

  p <- ggplot(d, aes(x = .data[[value_col]], y = .data[[label_col]],
                     fill = .data[[label_col]])) +
    geom_col(width = 0.72, colour = "white", linewidth = 0.3) +
    scale_fill_manual(values = palette, guide = "none", na.value = theme$na) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.08))) +
    labs(x = x_label, y = NULL) +
    fk_theme_pub(base_size = base_size, grid = FALSE)
  p
}

# ── Heatmap ─────────────────────────────────────────────────────────────────
# `df` is a matrix or data.frame of rows x cols. NA cells render grey:
# "missing" must never be mistakable for "zero".
fk_heatmap <- function(mat, symmetric = TRUE, cbar_label = "",
                       base_size = 12, col_label_angle = 45,
                       theme = fk_get_theme()) {
  m <- as.matrix(mat)
  long <- data.frame(
    row = rep(rownames(m), times = ncol(m)),
    col = rep(colnames(m), each = nrow(m)),
    value = as.vector(m), stringsAsFactors = FALSE)
  long$row <- factor(long$row, levels = rev(rownames(m)))
  long$col <- factor(long$col, levels = colnames(m))

  p <- ggplot(long, aes(x = .data$col, y = .data$row, fill = .data$value)) +
    geom_tile(colour = "white", linewidth = 0.2) +
    labs(x = NULL, y = NULL) +
    fk_theme_pub(base_size = base_size, grid = FALSE) +
    theme(axis.text.x = element_text(angle = col_label_angle, hjust = 1),
          axis.line = element_blank(), axis.ticks = element_blank())
  if (symmetric) {
    lim <- max(abs(long$value), na.rm = TRUE)
    p + scale_fill_gradient2(low = theme$down, mid = theme$neutral,
                             high = theme$up, midpoint = 0,
                             limits = c(-lim, lim), na.value = theme$na,
                             name = cbar_label)
  } else {
    p + scale_fill_gradient(low = "#ffffff", high = theme$up,
                            na.value = theme$na, name = cbar_label)
  }
}

# ── Save ────────────────────────────────────────────────────────────────────
# Writes PNG (to look at) + PDF (vector, for the composite). cairo_pdf keeps
# text as text so the panel stays editable in Illustrator — the R-side
# equivalent of matplotlib's pdf.fonttype = 42.
fk_save_panel <- function(p, name, dir, width = 6, height = 5, dpi = 600,
                          formats = c("png", "pdf")) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  out <- character(0)
  for (ext in formats) {
    f <- file.path(dir, paste0(name, ".", ext))
    if (ext == "pdf") {
      ggsave(f, p, width = width, height = height, device = grDevices::cairo_pdf)
    } else {
      ggsave(f, p, width = width, height = height, dpi = dpi)
    }
    out <- c(out, f)
  }
  invisible(out)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
