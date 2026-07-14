## forest.R — JAMA-style table-forest.
##
##   source("R/figkit.R"); source("R/forest.R")
##   p <- fk_table_forest(df,
##                        cols = list(list(col = "n_label", header = "N")),
##                        x_lab = "Odds Ratio (95% CI)")
##
## NOTE: `data$label` is rendered automatically as the left-most column —
## `cols` lists only the columns AFTER it. Passing label in `cols` renders it
## twice. Use `label_header` to retitle it ("Study", "Outcome", ...).
##
## Layout: left-side text table | centre forest | estimate column. Ported
## verbatim from the bAVM genotype-phenotype manuscript pipeline
## (genotype-phenotype/analysis/helper_scripts/utils.R `table_forest`, the
## canonical copy — an older fork in avm-spatial-tx lacked the 2026-05-28
## one-row-mode fixes). Originally adapted from the UKB NfL cardiovascular
## project (Keller et al.).
##
## `data` needs columns: label, est, lo, hi. Everything else is optional.
##
## The whole thing is hand-rolled on a theme_void() canvas with annotate():
## that is why it can align a text table against a forest at any panel width,
## and why it does not compose with fk_theme_pub() — it IS the layout.
##
## Deps: ggplot2, dplyr, rlang. Optional: grid (arrowheads on clipped CIs).

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(rlang)
})

fk_table_forest <- function(data,
                         # `data$label` ALWAYS renders as the left-most column;
                         # `cols` lists only the columns that come after it, so
                         # do not pass label in `cols` or it renders twice.
                         cols = list(),
                         # Header for that implicit label column. The original
                         # hardcoded "Predictor"; parameterised because a
                         # meta-analysis forest's first column is "Study", and
                         # an outcome forest's is "Outcome".
                         label_header = "Predictor",
                         null_value = 1,
                         log_scale = TRUE,
                         axis_ticks = NULL,
                         # NULL -> theme$down. A forest point encodes an
                         # estimate, not a direction, so every row is one
                         # colour; following the theme keeps it consistent with
                         # the rest of the figure instead of hardcoding the
                         # original JAMA blue (#2166AC, ~= the base theme's navy).
                         point_col = NULL,
                         pooled_col = NULL,
                         pooled_flag = "pooled",
                         size_col = NULL,
                         size_range = c(2, 6),
                         title = NULL,
                         subtitle = NULL,
                         x_lab = "Odds Ratio (95% CI)",
                         est_col_header = NULL,
                         est_fmt = "%.2f (%.2f\u2013%.2f)",
                         # est_col_side: "left" (default; OR becomes the last
                         # column of the left-side text table, aligned with
                         # N / FDR P etc.) or "right" (OR text renders to the
                         # right of the forest bars). The repo convention is
                         # "left" — used by Fig 2C/E and every ED forest. The
                         # default was flipped from "right" → "left" after a
                         # round of ED panels rendered with the OR floating in
                         # otherwise-empty white space to the right of the
                         # forest bars; making "left" the default means new
                         # producers don't have to opt in to the convention.
                         est_col_side = c("left", "right"),
                         # est_col_two_rows: when TRUE, the estimate string
                         #   (e.g. "28.6 (11.3-52.2)") is rendered on a sub-row
                         #   directly beneath the predictor/N row instead of
                         #   competing for horizontal space alongside them.
                         #   Header gets the same treatment ("Rate (95% CI)"
                         #   sits below "Predictor / N"). Used in composites
                         #   where forest cells are squeezed below ~7 in wide
                         #   and the est-column text would otherwise collide.
                         est_col_two_rows = FALSE,
                         base_size = 14,
                         # ── ELab visual options (Keller/avm-survey style) ──
                         # group_col: column whose values define row groupings;
                         #   rows whose group_col value matches group_bg_flag
                         #   are rendered against a light-grey band (used for
                         #   the "Primary / Secondary" split in the avm-survey
                         #   forest plot). Default NULL = no band.
                         group_col = NULL,
                         group_bg_flag = NULL,
                         group_bg_fill = "#f5f5f5",
                         group_bg_alpha = 0.5,
                         # left_label / right_label: directional text placed
                         #   under the x-axis (e.g. "<- Favours control",
                         #   "Favours treatment ->"). Default NULL = omitted.
                         left_label = NULL,
                         right_label = NULL,
                         # show_bottom_rule: when FALSE, suppresses the
                         #   horizontal rule between the last data row and
                         #   the x-axis. Default TRUE preserves the JAMA-
                         #   table appearance. Fig 2's high-risk OR forest
                         #   uses FALSE so the panel reads as a clean forest
                         #   without an extra hairline above the OR axis.
                         show_bottom_rule = TRUE) {

  if (is.null(point_col)) point_col <- fk_get_theme()$down

  # ── Transform to log scale if requested ─────────────────────────────────────
  if (log_scale) {
    data <- data %>% mutate(
      x_est = log(.data$est),
      x_lo  = log(.data$lo),
      x_hi  = log(.data$hi)
    )
    null_x <- log(null_value)
    if (is.null(axis_ticks)) axis_ticks <- c(0.25, 0.5, 1, 2, 4)
    tick_x <- log(axis_ticks)
    tick_labels <- as.character(axis_ticks)
  } else {
    data <- data %>% mutate(x_est = .data$est, x_lo = .data$lo, x_hi = .data$hi)
    null_x <- null_value
    if (is.null(axis_ticks)) axis_ticks <- pretty(c(data$lo, data$hi), n = 5)
    tick_x <- axis_ticks
    tick_labels <- as.character(axis_ticks)
  }

  # ── Auto-extend axis so every point estimate lies within the forest range ──
  # Rationale: CI endpoints can be arbitrarily wide under sparse data and are
  # still best handled by clipping + arrowheads, but a *point estimate* that
  # falls outside the axis produces a hollow marker at the axis edge that
  # visually collides with the CI's arrowhead. Extending the axis to cover all
  # point estimates (with a small pad) keeps every box cleanly inside the
  # range. Ticks are auto-added in the user's tick cadence (doubling for log
  # OR axes, pretty breaks otherwise).
  est_finite <- data$x_est[is.finite(data$x_est)]
  if (length(est_finite) > 0) {
    est_lo <- min(est_finite)
    est_hi <- max(est_finite)
    if (log_scale) {
      # Extend by doubling/halving the outermost tick (i) until every
      # median fits inside the range AND (ii) one extra step beyond the
      # most extreme median, so a clipped CI arrowhead always has
      # horizontal headroom to render. Without the extra step, a median
      # that sits within half a log-step of the edge tick would clip its
      # arrow into the axis line. tick-step on a log-2 axis = log(2),
      # so we require >= half a tick (i.e. est_hi within log(sqrt(2)) of
      # the edge → extend).
      half_log_step <- log(2) / 2  # log(sqrt(2)) ≈ 0.347
      max_iter <- 20L
      i <- 0L
      while ((est_hi + half_log_step) > max(tick_x) + 1e-9 && i < max_iter) {
        next_val <- max(axis_ticks) * 2
        axis_ticks <- c(axis_ticks, next_val)
        tick_x <- c(tick_x, log(next_val))
        tick_labels <- c(tick_labels, formatC(next_val, format = "fg"))
        i <- i + 1L
      }
      i <- 0L
      while ((est_lo - half_log_step) < min(tick_x) - 1e-9 && i < max_iter) {
        next_val <- min(axis_ticks) / 2
        axis_ticks <- c(next_val, axis_ticks)
        tick_x <- c(log(next_val), tick_x)
        tick_labels <- c(formatC(next_val, format = "fg"), tick_labels)
        i <- i + 1L
      }
    } else {
      # Linear scale: widen pretty ticks to cover all point estimates,
      # PLUS one extra step of headroom beyond the extreme median so
      # clipped CI arrowheads always have horizontal room to render.
      step <- if (length(tick_x) >= 2) diff(range(tick_x)) / (length(tick_x) - 1) else 1
      pad  <- max(step, (max(tick_x) - min(tick_x)) * 0.05)
      need_extend_hi <- est_hi >= max(tick_x) - 1e-9
      need_extend_lo <- est_lo <= min(tick_x) + 1e-9
      if (need_extend_hi || need_extend_lo) {
        lo_lim <- if (need_extend_lo) est_lo - pad else min(tick_x)
        hi_lim <- if (need_extend_hi) est_hi + pad else max(tick_x)
        new_range <- range(c(tick_x, lo_lim, hi_lim))
        axis_ticks <- pretty(new_range, n = length(axis_ticks))
        tick_x <- axis_ticks
        tick_labels <- as.character(axis_ticks)
      }
    }
  }

  # ── Default estimate column header ──────────────────────────────────────────
  if (is.null(est_col_header)) {
    est_col_header <- if (log_scale) "OR (95% CI)" else "Estimate (95% CI)"
  }

  # ── Identify pooled row ─────────────────────────────────────────────────────
  if (!is.null(pooled_col) && pooled_col %in% names(data)) {
    data$is_pooled <- data[[pooled_col]] == pooled_flag
  } else {
    data$is_pooled <- FALSE
  }

  # ── Convert size_col to numeric ─────────────────────────────────────────────
  # size_col is usually the *display* N string, so strip thousands separators
  # before coercing: a formatted "1,204" would otherwise land as NA and
  # silently drop that row's dot size — the row still plots, just with no size
  # mapping, which is near-impossible to spot on a finished panel. Empty
  # strings are a legitimate "no size" placeholder and stay NA quietly;
  # anything else that won't parse warns.
  if (!is.null(size_col) && size_col %in% names(data)) {
    raw_size <- as.character(data[[size_col]])
    data$size_val <- suppressWarnings(as.numeric(gsub("[,  ]", "", raw_size)))
    unparsed <- is.na(data$size_val) & !is.na(raw_size) & nzchar(raw_size)
    if (any(unparsed)) {
      warning(sprintf("table_forest: %d value(s) in size_col '%s' are not numeric (%s); those points get no size mapping",
                      sum(unparsed), size_col,
                      paste(unique(raw_size[unparsed]), collapse = ", ")),
              call. = FALSE)
    }
  }

  # ── Y positions (bottom-up: first row at top) ──────────────────────────────
  # In two-row mode each study occupies a vertically taller block (top
  # sub-row + bottom sub-row + breathing room), so we multiply y_pos by
  # row_spacing > 1 to give every study its own visual band and stop the
  # bottom sub-row from colliding with the next study's top sub-row.
  n_rows      <- nrow(data)
  row_spacing <- if (isTRUE(est_col_two_rows)) 2.2 else 1.0
  data$y_pos  <- (rev(seq_len(n_rows)) - 1) * row_spacing
  if (any(data$is_pooled)) {
    pooled_y <- data$y_pos[data$is_pooled]
    # Lift non-pooled rows above the pooled row by 0.7 * row_spacing in
    # two-row mode (was 0.5) so the gap between BCH's CI sub-row and the
    # pooled "Pooled" label is visibly larger than the inter-study gap.
    pooled_lift <- if (isTRUE(est_col_two_rows)) 0.7 else 0.5
    data$y_pos[!data$is_pooled & data$y_pos > pooled_y] <-
      data$y_pos[!data$is_pooled & data$y_pos > pooled_y] + pooled_lift * row_spacing
  }

  # ── X layout: LEFT table | CENTER forest | (optional) RIGHT estimate text ─
  est_col_side <- match.arg(est_col_side)
  forest_min   <- min(tick_x)
  forest_max   <- max(tick_x)
  forest_range <- forest_max - forest_min

  # Left table region. When the OR column is on the left, one extra slot is
  # packed into the same strip, so widen the strip proportionally to keep
  # long Predictor labels clear of the right-aligned text columns (we saw
  # "Venous Outflow Stenosis" / N = "184" collide at the 2.0x width).
  #
  # Wide-axis guard: for log-scale forests that span >2 log10 decades (e.g.
  # Fig 4A, axis 0.25–1024), a forest_range-proportional table balloons to
  # ~11 log-units wide and visually pushes the forest into the right ~25% of
  # the cell. When log_scale && forest_range > 2, switch to a fixed per-slot
  # width so the table packs tight regardless of axis span. Fig 2 forests
  # stay on the original formula (their axes span ~1.2 log-units).
  #
  # Linear-scale forests (log_scale = FALSE) live in raw data units (e.g.
  # 0–100 for rate-%, 0–60 for years), so forest_range > 2 is essentially
  # always true and the fixed per-slot constant (3.5) is far too tight in
  # raw units — it crams every column into the same x position. For those
  # we use a fraction-of-forest_range formula tuned to leave the forest
  # ~60% of the panel and the table ~30% (linear_left/linear_right below).
  n_extra      <- length(cols)
  n_col_slots  <- if (est_col_side == "left") n_extra + 1 else n_extra
  # Per-column slot width (log10 units). Constants below were hand-tuned at
  # base_size 11; physical text width scales linearly with base_size, so we
  # rescale the multipliers by (base_size / 11) so the table packs the same
  # PROPORTION of the panel cell at any base_size.
  bs_scale     <- base_size / 11
  # In single-row mode the est column renders the full "%.2f (%.2f–%.2f)"
  # string (~17 chars) on one line vs. just "(%.2f–%.2f)" (~10 chars) in
  # two-row mode, so the est slot needs more horizontal room when est sits
  # LEFT of the forest. We don't widen table_width itself — total axis span
  # grows with it, so text width in axis units scales the same way and the
  # gap stays proportionally cramped. Instead, shift only the est slot
  # rightward by one full col_spacing below, after table_width is set.
  one_row_left <- !isTRUE(est_col_two_rows) && est_col_side == "left"
  table_width  <- if (log_scale && forest_range > 2) {
    (n_col_slots + 1) * 3.5 * bs_scale
  } else if (!log_scale) {
    # Linear-scale axes (raw rate-%, years, slope-units): take the larger of
    # (a) the log-scale wide-axis fixed-per-slot width, which dominates when
    #     forest_range is small (few raw units) — without it, a slope axis
    #     spanning −6…2 leaves only ~6 raw-units of table strip and the
    #     Predictor/N columns collide.
    # (b) a forest_range-proportional width, which dominates when forest_range
    #     is large (rate-% 0–100, age 0–60), so the forest doesn't get
    #     squeezed into a small fraction of the panel.
    max(
      (n_col_slots + 1) * 3.5,
      forest_range * (if (est_col_side == "left") 0.70 else 0.30)
    ) * bs_scale
  } else {
    forest_range * (if (est_col_side == "left") 3.0 else 2.0) * bs_scale
  }
  # Cap the table-to-forest gap so it doesn't balloon with wide axes.
  table_gap    <- min(forest_range * 0.25, 0.4)
  x_table_left <- forest_min - table_gap - table_width
  TX_LABEL     <- x_table_left

  if (n_col_slots > 0) {
    avail        <- table_width * 0.92
    # Predictor column gets 2.5 slot widths because its text is typically
    # the widest in the table (e.g. "Pooled (REML)", "KRAS G12V (|z| ≤ 2)"
    # or "Venous Outflow Stenosis"); the numeric / FDR / OR columns are
    # narrow. The extra padding keeps long predictor labels clear of the
    # right-aligned N column even at half-width composite cells.
    pred_slots   <- 2.5
    col_spacing  <- avail / (n_col_slots + pred_slots)
    slot_x       <- TX_LABEL + col_spacing * (pred_slots - 1 + seq_len(n_col_slots))
    # Single-row left-side est: shift the est slot one col_spacing right so
    # the "%.1f (%.1f–%.1f)" string clears the adjacent N column. The slot
    # ends ~1 col_spacing shy of forest_min (table_gap reserves ≤0.4 raw
    # units), so the shift stays inside the table strip.
    if (one_row_left && n_col_slots >= 2) {
      slot_x[n_col_slots] <- slot_x[n_col_slots] + col_spacing
    }
  } else {
    slot_x <- numeric(0)
  }

  if (est_col_side == "left") {
    tx_positions <- if (n_extra > 0) slot_x[seq_len(n_extra)] else numeric(0)
    TX_EST       <- slot_x[n_col_slots]
    x_right      <- forest_max + forest_range * 0.12
  } else {
    tx_positions <- slot_x
    TX_EST       <- forest_max + forest_range * 0.12
    x_right      <- TX_EST + forest_range * 0.85
  }

  # ── Key Y positions ────────────────────────────────────────────────────────
  # Two-row mode: each data row + the header gets a sub-row 0.5 axis-units
  # below its primary y-position, on which the est-column string is rendered.
  # We lift the header-band and drop the axis-band by sub_gap so the sub-rows
  # don't crowd their neighbours.
  sub_gap  <- if (isTRUE(est_col_two_rows)) 1.0 else 0
  # In two-row mode at composite scale, annotate text spans ~0.7 axis units
  # vertically. Push header_y up by hdr_pad and the rule down by rule_offset
  # so neither the "(95% CI)" sub-row nor the first data row's top text
  # crosses the rule line.
  # One-row mode pads bumped 2026-05-28 (ED Fig 4 panel c): each text row is
  # ~0.5 axis-units tall at base_size 8 in a 2.85-in panel, so the previous
  # 0.6 / 0.35 hdr/rule pads put the top rule right on the first row's text
  # top edge. Lift hdr_pad to 1.0 → rule_y sits ~0.65 au above the top row,
  # giving a clean separation that matches the two-row case proportionally.
  hdr_pad     <- if (isTRUE(est_col_two_rows)) 1.5 else 1.0
  rule_offset <- if (isTRUE(est_col_two_rows)) 0.6 else 0.35
  header_y <- max(data$y_pos) + hdr_pad + sub_gap
  axis_y   <- min(data$y_pos) - 1.0 - sub_gap
  rule_y   <- header_y - rule_offset - sub_gap

  # ── Build plot ──────────────────────────────────────────────────────────────
  dr <- data %>% filter(!is_pooled)
  pl <- data %>% filter(is_pooled)

  p <- ggplot()

  # ── ELab-style group shading band (opt-in via group_col + group_bg_flag) ─
  # Rendered first so all other geoms sit on top. Typical use: flag the
  # secondary-outcome block in a multi-row forest so readers see the
  # primary/secondary boundary at a glance (see avm-survey Figure 3).
  if (!is.null(group_col) && !is.null(group_bg_flag) &&
      group_col %in% names(data)) {
    band_rows <- data[data[[group_col]] == group_bg_flag, , drop = FALSE]
    if (nrow(band_rows) > 0) {
      p <- p + annotate(
        "rect",
        xmin = TX_LABEL - 0.3, xmax = x_right,
        ymin = min(band_rows$y_pos) - 0.4,
        ymax = max(band_rows$y_pos) + 0.4,
        fill = group_bg_fill, alpha = group_bg_alpha
      )
    }
  }

  # 2026-05-19 NM pass: dashed (not dotted) reference line at the null
  # value, slightly thicker and darker so it's actually visible at
  # native composite scale. Spans from the axis line up to the top of
  # the data rows (just below the header rule).
  p <- p +
    annotate("segment", x = null_x, xend = null_x,
             y = axis_y + 0.1, yend = rule_y,
             linetype = "dashed", colour = "grey45", linewidth = 0.35)

  # Axis ticks + labels. tick_text_y_pad is the gap between the axis line
  # and the tick label centre. Text bounding boxes span ~0.7 axis units in
  # two-row mode at composite scale, so a 0.08-unit pad isn't enough — the
  # tick labels touch the bottom rule. Use 0.55 in two-row mode.
  # 2026-05-28 (ED Fig 4 panel c): tick text top edge sits ~0.25 au above
  # its centre at this scale, so a 0.08 pad left the labels straddling the
  # axis line (axis_y + 0.10). Drop the labels by 0.40 au so the gap to the
  # axis line is ~0.25 au — visible but compact.
  tick_text_y_pad <- if (isTRUE(est_col_two_rows)) 0.55 else 0.40
  # 2026-05-19 NM pass: pad each tick label with a thin space on either
  # side so adjacent log-spaced labels (e.g. "0.5", "1", "2") read as
  # distinct words instead of running together. U+2009 THIN SPACE is
  # narrower than a normal space; one on each side adds ~0.15 axis
  # units of visual gap at base_size=14 without offsetting the centre.
  tick_labels_padded <- paste0("  ", tick_labels, "  ")
  for (i in seq_along(tick_x)) {
    p <- p +
      annotate("segment", x = tick_x[i], xend = tick_x[i],
               y = axis_y + 0.05, yend = axis_y + 0.18,
               colour = "grey60", linewidth = 0.3) +
      annotate("text", x = tick_x[i], y = axis_y - tick_text_y_pad,
               label = tick_labels_padded[i], size = base_size / 4,
               colour = "black", hjust = 0.5)
  }
  p <- p + annotate("segment", x = forest_min, xend = forest_max,
                     y = axis_y + 0.1, yend = axis_y + 0.1,
                     colour = "grey60", linewidth = 0.4)
  # x-axis title sits below the tick labels. 1.4 units below axis_y in
  # two-row mode (was 0.85) to leave a clear gap from the tick labels.
  # 2026-05-28: paired with the tick_text_y_pad bump above; the title needs
  # to drop with the ticks or it lands on top of them.
  axis_title_y_pad <- if (isTRUE(est_col_two_rows)) 1.4 else 1.30
  p <- p + annotate("text", x = mean(c(forest_min, forest_max)),
                     y = axis_y - axis_title_y_pad, label = x_lab,
                     size = base_size / 3.8, colour = "black", hjust = 0.5)

  # ── ELab-style directional labels (opt-in via left_label / right_label) ──
  # Placed beneath the x-axis on the null-line sides, e.g.
  # "<- Favours no therapy" / "Favours therapy ->". Y-position is tracked in
  # `dir_label_y` so the final scale_y_continuous() call (below) can extend
  # the lower bound to keep the labels inside the plot region.
  dir_label_y <- NA_real_
  if (!is.null(left_label) || !is.null(right_label)) {
    dir_label_y <- axis_y - 0.85
    if (!is.null(left_label)) {
      p <- p + annotate(
        "text",
        x = mean(c(forest_min, null_x)), y = dir_label_y,
        label = left_label, size = base_size / 4.5,
        colour = "black", hjust = 0.5
      )
    }
    if (!is.null(right_label)) {
      p <- p + annotate(
        "text",
        x = mean(c(null_x, forest_max)), y = dir_label_y,
        label = right_label, size = base_size / 4.5,
        colour = "black", hjust = 0.5
      )
    }
  }

  # ── Data row CIs + sized points ───────────────────────────────────────────
  # Clipping behavior (standard forest-plot convention, e.g. Cochrane/metafor):
  #   • CI endpoints outside [forest_min, forest_max] are clipped to the axis
  #     and marked with an arrowhead on the clipped side.
  #   • Point estimates outside the axis are clamped to the axis edge and
  #     rendered as an open (hollow) marker to indicate the estimate itself
  #     lies beyond the displayed range. The numeric OR (right-side column)
  #     still shows the true value.
  # This preserves faithful representation without silently dropping markers
  # or misaligning the point with its error bar.
  arrow_style <- grid::arrow(length = unit(0.10, "inches"),
                             ends = "last", type = "closed")

  draw_forest_row <- function(p, rows, err_height, err_lw, pt_size_default,
                              pt_shape = 15, pt_col, pt_fill = NA,
                              pt_border = NULL, use_size_col = FALSE) {
    if (nrow(rows) == 0) return(p)
    rows <- rows %>% mutate(
      x_lo_clip = pmax(x_lo, forest_min),
      x_hi_clip = pmin(x_hi, forest_max),
      x_est_clip = pmax(pmin(x_est, forest_max), forest_min),
      lo_clipped = x_lo < forest_min,
      hi_clipped = x_hi > forest_max,
      est_clipped = x_est < forest_min | x_est > forest_max
    )

    # Base horizontal bar (between the two clipped endpoints)
    p <- p + geom_segment(
      data = rows,
      aes(y = y_pos, yend = y_pos, x = x_lo_clip, xend = x_hi_clip),
      linewidth = err_lw, colour = pt_col,
      inherit.aes = FALSE
    )
    # Whisker caps on unclipped sides
    cap_lo <- rows %>% filter(!lo_clipped)
    cap_hi <- rows %>% filter(!hi_clipped)
    if (nrow(cap_lo) > 0) {
      p <- p + geom_segment(
        data = cap_lo,
        aes(x = x_lo_clip, xend = x_lo_clip,
            y = y_pos - err_height / 2, yend = y_pos + err_height / 2),
        linewidth = err_lw, colour = pt_col,
        inherit.aes = FALSE
      )
    }
    if (nrow(cap_hi) > 0) {
      p <- p + geom_segment(
        data = cap_hi,
        aes(x = x_hi_clip, xend = x_hi_clip,
            y = y_pos - err_height / 2, yend = y_pos + err_height / 2),
        linewidth = err_lw, colour = pt_col,
        inherit.aes = FALSE
      )
    }
    # Arrowheads on clipped sides (point inward → outward along axis)
    arr_lo <- rows %>% filter(lo_clipped)
    arr_hi <- rows %>% filter(hi_clipped)
    if (nrow(arr_lo) > 0) {
      p <- p + geom_segment(
        data = arr_lo,
        aes(y = y_pos, yend = y_pos,
            x = forest_min + forest_range * 0.04, xend = forest_min),
        linewidth = err_lw, colour = pt_col, arrow = arrow_style,
        inherit.aes = FALSE
      )
    }
    if (nrow(arr_hi) > 0) {
      p <- p + geom_segment(
        data = arr_hi,
        aes(y = y_pos, yend = y_pos,
            x = forest_max - forest_range * 0.04, xend = forest_max),
        linewidth = err_lw, colour = pt_col, arrow = arrow_style,
        inherit.aes = FALSE
      )
    }

    # Points (clamped). Clamped points use hollow marker (shape + 0 = filled,
    # so we override with open variants when clipped).
    pts_inside  <- rows %>% filter(!est_clipped)
    pts_clipped <- rows %>% filter(est_clipped)

    if (use_size_col && "size_val" %in% names(rows)) {
      if (nrow(pts_inside) > 0) {
        p <- p + geom_point(
          data = pts_inside,
          aes(y = y_pos, x = x_est_clip, size = size_val),
          shape = pt_shape, colour = pt_col, fill = pt_col,
          show.legend = FALSE, inherit.aes = FALSE
        )
      }
      if (nrow(pts_clipped) > 0) {
        # Hollow marker to flag: point estimate is beyond axis range
        p <- p + geom_point(
          data = pts_clipped,
          aes(y = y_pos, x = x_est_clip, size = size_val),
          shape = 0, colour = pt_col, stroke = 1.1,
          show.legend = FALSE, inherit.aes = FALSE
        )
      }
      p <- p + scale_size_continuous(range = size_range)
    } else {
      if (nrow(pts_inside) > 0) {
        p <- p + geom_point(
          data = pts_inside,
          aes(y = y_pos, x = x_est_clip),
          size = pt_size_default, shape = pt_shape,
          colour = ifelse(is.null(pt_border), pt_col, pt_border),
          fill = if (!is.na(pt_fill)) pt_fill else pt_col,
          inherit.aes = FALSE
        )
      }
      if (nrow(pts_clipped) > 0) {
        p <- p + geom_point(
          data = pts_clipped,
          aes(y = y_pos, x = x_est_clip),
          size = pt_size_default, shape = 0,
          colour = pt_col, stroke = 1.1,
          inherit.aes = FALSE
        )
      }
    }
    p
  }

  # In two-row mode, the dot/CI bar should sit vertically centered between
  # the predictor+N row (top sub-row) and the rate-value row (bottom sub-row).
  # Shift the y_pos in the data frame passed to dot rendering by -sub_gap/2;
  # text rendering further down still uses the original data$y_pos.
  dot_y_offset <- if (isTRUE(est_col_two_rows)) -sub_gap / 2 else 0
  dr_dot <- if (dot_y_offset != 0) dr %>% mutate(y_pos = y_pos + dot_y_offset) else dr

  # Per-study rows
  p <- draw_forest_row(
    p, dr_dot,
    err_height = 0.18, err_lw = 0.8,
    pt_size_default = 3.5, pt_shape = 15, pt_col = point_col,
    use_size_col = !is.null(size_col)
  )

  # Pooled row (diamond) — keep diamond; if clipped (unlikely for pooled),
  # fall back to hollow diamond.
  if (nrow(pl) > 0) {
    pl <- pl %>% mutate(
      x_lo_clip = pmax(x_lo, forest_min),
      x_hi_clip = pmin(x_hi, forest_max),
      x_est_clip = pmax(pmin(x_est, forest_max), forest_min),
      lo_clipped = x_lo < forest_min,
      hi_clipped = x_hi > forest_max,
      est_clipped = x_est < forest_min | x_est > forest_max,
      y_pos = y_pos + dot_y_offset
    )
    p <- p +
      geom_segment(
        data = pl,
        aes(y = y_pos, yend = y_pos, x = x_lo_clip, xend = x_hi_clip),
        linewidth = 1.0, colour = "black", inherit.aes = FALSE
      )
    # Pooled arrows if clipped
    arr_lo_p <- pl %>% filter(lo_clipped)
    arr_hi_p <- pl %>% filter(hi_clipped)
    if (nrow(arr_lo_p) > 0) {
      p <- p + geom_segment(
        data = arr_lo_p,
        aes(y = y_pos, yend = y_pos,
            x = forest_min + forest_range * 0.04, xend = forest_min),
        linewidth = 1.0, colour = "black", arrow = arrow_style,
        inherit.aes = FALSE
      )
    }
    if (nrow(arr_hi_p) > 0) {
      p <- p + geom_segment(
        data = arr_hi_p,
        aes(y = y_pos, yend = y_pos,
            x = forest_max - forest_range * 0.04, xend = forest_max),
        linewidth = 1.0, colour = "black", arrow = arrow_style,
        inherit.aes = FALSE
      )
    }
    pooled_inside  <- pl %>% filter(!est_clipped)
    pooled_clipped <- pl %>% filter(est_clipped)
    if (nrow(pooled_inside) > 0) {
      p <- p + geom_point(
        data = pooled_inside,
        aes(y = y_pos, x = x_est_clip),
        size = 5, shape = 23, fill = point_col, colour = "black",
        inherit.aes = FALSE
      )
    }
    if (nrow(pooled_clipped) > 0) {
      p <- p + geom_point(
        data = pooled_clipped,
        aes(y = y_pos, x = x_est_clip),
        size = 5, shape = 5, colour = "black", stroke = 1.1,
        inherit.aes = FALSE
      )
    }
  }

  # ── Left table: label column ───────────────────────────────────────────────
  # In two-row mode, parenthetical qualifiers in the label (typically the
  # pooled row's "Pooled (REML)") would push the bold label into the N
  # column. Split the label at the first " (" boundary and render the head
  # ("Pooled") on the top sub-row at full weight, with the parenthetical
  # ("(REML)") on the bottom sub-row at lighter weight, mirroring the est
  # column's point/CI split.
  label_face <- ifelse(data$is_pooled, "bold", "plain")
  display_label <- data$label
  sub_label     <- rep("", nrow(data))
  if (isTRUE(est_col_two_rows)) {
    has_parens   <- grepl(" \\(.*\\)$", data$label, perl = TRUE)
    display_label[has_parens] <- sub(" \\(.*\\)$", "", data$label[has_parens],
                                     perl = TRUE)
    sub_label[has_parens]     <- regmatches(data$label[has_parens],
                                            regexpr(" \\(.*\\)$",
                                                    data$label[has_parens],
                                                    perl = TRUE))
    sub_label[has_parens]     <- sub("^ ", "", sub_label[has_parens])
  }
  p <- p +
    annotate("text", x = TX_LABEL, y = data$y_pos,
             label = display_label, hjust = 0,
             size = base_size / 3.5, fontface = label_face, colour = "black")
  if (any(nzchar(sub_label))) {
    sub_rows <- which(nzchar(sub_label))
    p <- p +
      annotate("text", x = TX_LABEL, y = data$y_pos[sub_rows] - sub_gap,
               label = sub_label[sub_rows], hjust = 0,
               size = base_size / 3.8, fontface = "plain", colour = "black")
  }

  # ── Left table: additional columns ─────────────────────────────────────────
  for (j in seq_along(cols)) {
    col_data <- as.character(data[[cols[[j]]$col]])
    col_data[is.na(col_data) | col_data == "NA"] <- "\u2014"
    p <- p +
      annotate("text", x = tx_positions[j], y = data$y_pos,
               label = col_data, hjust = 1,
               size = base_size / 4, colour = "black")
  }

  # ── Estimate (95% CI) text ─────────────────────────────────────────────────
  # hjust: 0 (left-aligned) when OR sits to the right of the forest; 1
  # (right-aligned) when OR is the last column of the left table so its
  # text ends flush with neighbouring table columns.
  #
  # In two-row mode the est-column is split: the point estimate ("28.6")
  # stays on the predictor/N row and only the CI portion ("(11.3-52.2)")
  # drops to the sub-row. We split est_fmt at the first " (" or " [" so the
  # producer can keep its existing one-string format.
  est_h    <- if (est_col_side == "left") 1 else 0
  .split_fmt <- function(s) {
    m <- regmatches(s, regexpr(" \\(.*\\)$| \\[.*\\]$", s, perl = TRUE))
    if (length(m) == 1L && nzchar(m))
      list(point = sub(" \\(.*\\)$| \\[.*\\]$", "", s, perl = TRUE),
           ci    = sub("^ ", "", m))
    else list(point = s, ci = NULL)
  }
  if (isTRUE(est_col_two_rows)) {
    est_split   <- .split_fmt(est_fmt)
    point_text  <- sprintf(est_split$point, data$est)
    ci_text     <- if (!is.null(est_split$ci))
      sprintf(est_split$ci, data$lo, data$hi) else rep("", nrow(data))
    p <- p +
      annotate("text", x = TX_EST, y = data$y_pos,
               label = point_text, hjust = est_h,
               size = base_size / 3.8, colour = "black",
               fontface = ifelse(data$is_pooled, "bold", "plain"))
    if (any(nzchar(ci_text))) {
      # Center the CI text under the rate value's visual center, rather
      # than right-aligning it at TX_EST (which makes the wider CI string
      # extend further left than the rate it belongs to). Estimate the
      # rate's half-width using the longest formatted point and shift x.
      #
      # The shift is expressed as a fraction of the inter-column spacing
      # rather than in raw axis units. The previous formula
      # (max_pt_chars * 0.45 * base_size/11) worked when the axis was
      # linear (rate %, age years, raw counts) — there 0.45 axis units
      # ≈ one character of rendered text. Under a log10 axis (OR / HR
      # forests with span ≤ 2 log10-units) the axis units are an order
      # of magnitude smaller per char, so the same formula shifted the
      # CI text out of the OR column entirely (visually landing under
      # the N column instead). Tying the shift to col_spacing keeps the
      # CI under the point estimate regardless of axis scale.
      max_pt_chars <- max(nchar(point_text), na.rm = TRUE)
      half_pt_w    <- if (n_col_slots > 0) {
        # ~7% of inter-column gap per character, capped at half a column
        # so the CI never crosses into the previous text column.
        min(max_pt_chars * col_spacing * 0.07, col_spacing * 0.45)
      } else {
        max_pt_chars * 0.45 * (base_size / 11)
      }
      ci_x         <- if (est_col_side == "left") TX_EST - half_pt_w
                      else TX_EST + half_pt_w
      p <- p +
        annotate("text", x = ci_x, y = data$y_pos - sub_gap,
                 label = ci_text, hjust = 0.5,
                 size = base_size / 3.8, colour = "black",
                 fontface = "plain")
    }
  } else {
    or_text <- sprintf(est_fmt, data$est, data$lo, data$hi)
    p <- p +
      annotate("text", x = TX_EST, y = data$y_pos,
               label = or_text, hjust = est_h,
               size = base_size / 3.8, colour = "black",
               fontface = ifelse(data$is_pooled, "bold", "plain"))
  }

  # ── Column headers ──────────────────────────────────────────────────────────
  p <- p +
    annotate("text", x = TX_LABEL, y = header_y, label = label_header,
             hjust = 0, size = base_size / 3.8, fontface = "bold")
  for (j in seq_along(cols)) {
    p <- p +
      annotate("text", x = tx_positions[j], y = header_y,
               label = cols[[j]]$header, hjust = 1,
               size = base_size / 3.8, fontface = "bold")
  }
  if (isTRUE(est_col_two_rows)) {
    # Drop the "(95% CI)" sub-line on the header — the parenthetical CI
    # format is self-evident from the data rows below, and removing the
    # sub-header simplifies the top of the plot.
    hdr_split <- .split_fmt(est_col_header)
    p <- p +
      annotate("text", x = TX_EST, y = header_y,
               label = hdr_split$point, hjust = est_h,
               size = base_size / 3.8, fontface = "bold")
  } else {
    p <- p +
      annotate("text", x = TX_EST, y = header_y,
               label = est_col_header, hjust = est_h,
               size = base_size / 3.8, fontface = "bold")
  }

  # ── Horizontal rules ───────────────────────────────────────────────────────
  x_left <- TX_LABEL - 0.05
  p <- p +
    annotate("segment", x = x_left, xend = x_right,
             y = rule_y, yend = rule_y,
             colour = "grey50", linewidth = 0.4)
  # Bottom rule (between last data row and x-axis) is opt-out via the
  # show_bottom_rule arg so panels that want a cleaner forest without
  # the extra hairline above the axis can suppress it.
  if (isTRUE(show_bottom_rule)) {
    p <- p +
      annotate("segment", x = x_left, xend = x_right,
               y = axis_y + 0.35, yend = axis_y + 0.35,
               colour = "grey50", linewidth = 0.4)
  }

  # ── Title / subtitle ───────────────────────────────────────────────────────
  # In two-row mode, annotate text spans ~0.7 axis-units, so the previous
  # 0.6-unit title-to-header gap had the title baseline overlapping the
  # Predictor header. Push title up by 1.4 units in two-row mode.
  if (!is.null(title)) {
    title_pad <- if (isTRUE(est_col_two_rows)) 1.4 else 0.6
    title_y <- header_y + ifelse(!is.null(subtitle), 1.0, title_pad)
    p <- p + annotate("text", x = TX_LABEL, y = title_y,
                       label = title, hjust = 0,
                       size = base_size / 2.8, fontface = "bold")
  }
  if (!is.null(subtitle)) {
    p <- p + annotate("text", x = TX_LABEL, y = header_y + 0.55,
                       label = subtitle, hjust = 0,
                       size = base_size / 3.4, colour = "black")
  }

  # ── Final theme ─────────────────────────────────────────────────────────────
  # The lower y-limit must clear EVERYTHING drawn below the axis: the axis
  # title (axis_y - axis_title_y_pad) and, when present, the directional
  # labels (dir_label_y). Take the minimum of the two rather than letting
  # either one alone define the floor.
  #
  # This is a fix, not a port. The upstream version read:
  #     y_lo <- if (!is.na(dir_label_y)) dir_label_y - 0.25 else axis_y - y_lo_pad
  # so passing left_label/right_label swapped the floor to dir_label_y - 0.25
  # (= axis_y - 1.10) and silently clipped the axis title at axis_y - 1.30 —
  # ggplot drops it with only a generic "Removed 1 row containing missing
  # values" warning, which is easy to lose in render noise. The 2026-05-28
  # polish that dropped the title from -0.85 to -1.30 (and raised this pad
  # 1.05 -> 1.70) updated only the no-directional-label branch.
  y_lo_pad <- if (isTRUE(est_col_two_rows)) 1.8 else 1.70
  y_lo <- min(axis_y - y_lo_pad,
              if (!is.na(dir_label_y)) dir_label_y - 0.25 else Inf)
  # Top buffer above the header row. 1.6 units reserves room for the
  # optional title/subtitle; when neither is passed we collapse the
  # buffer so short (2-3 row) forests fill their allocated vertical
  # space instead of floating in the middle of the panel.
  top_buffer <- if (!is.null(title) || !is.null(subtitle)) {
    if (isTRUE(est_col_two_rows)) 2.2 else 1.6
  } else 0.25
  p <- p +
    scale_x_continuous(limits = c(x_left, x_right), expand = c(0, 0)) +
    scale_y_continuous(limits = c(y_lo, header_y + top_buffer)) +
    labs(x = NULL, y = NULL) +
    theme_void(base_size = base_size) +
    theme(plot.margin = margin(8, 14, 8, 14))

  p
}


# ── Meta-analysis forest ─────────────────────────────────────────────────────
# Per-study rows + a pooled diamond; dot size ∝ N.
# `data` needs: label, est, lo, hi, type ("study"/"pooled"), n_label.
fk_forest_meta <- function(data, ..., size_range = c(2, 7)) {
  fk_table_forest(data, pooled_col = "type", pooled_flag = "pooled",
                  size_col = "n_label", size_range = size_range, ...)
}

# ── General predictor forest ─────────────────────────────────────────────────
# No pooled row; dot size ∝ N. `data` needs: label, est, lo, hi, n_label.
fk_forest_general <- function(data, ..., size_range = c(2, 7)) {
  fk_table_forest(data, pooled_col = NULL, pooled_flag = NULL,
                  size_col = "n_label", size_range = size_range, ...)
}

# ── Compact-cell wrappers ────────────────────────────────────────────────────
# est_col_two_rows = TRUE flips on the narrow-cell layout: each row splits into
# a top sub-row (predictor + N + estimate) and a bottom sub-row (the CI in
# parentheses), with row spacing and header pads scaled up to match. Use when
# the forest will be composed into a cell below ~7in wide, where the estimate
# column would otherwise collide with the predictor text.
fk_forest_compact_meta <- function(...) fk_forest_meta(..., est_col_two_rows = TRUE)
fk_forest_compact_general <- function(...) fk_forest_general(..., est_col_two_rows = TRUE)
