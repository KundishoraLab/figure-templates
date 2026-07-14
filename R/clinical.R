## clinical.R — Kaplan-Meier curves, regression scatter, dumbbell prevalence.
##
##   source("R/figkit.R"); source("R/clinical.R")
##   km <- fk_km_panel(df, "age", "event", "group", xlab = "Age (years)")
##   fk_save_km_panel("panels/", "fig1d_km", km, w = 5, h = 5)
##
## Ported from the bAVM genotype-phenotype manuscript pipeline
## (analysis/helper_scripts/utils.R + analysis/manuscript/genotype_phenotype/
## 04b_km_age.R / 04c_km_sex_stratified.R), generalised off the cohort.
##
## Deps: ggplot2. fk_km_panel/fk_save_km_panel additionally need survival,
## survminer, patchwork; fk_regression_scatter needs scales.

suppressPackageStartupMessages(library(ggplot2))


# ── Kaplan-Meier panel ───────────────────────────────────────────────────────
# Returns a ggsurvplot object (NOT a plain ggplot) — pass it straight to
# fk_save_km_panel(), which knows how to persist the curve and the risk table
# separately.
#
# Defaults encode the manuscript's KM grammar so every curve in a figure set
# matches without each producer re-specifying it: linewidth 0.5, CI ribbon at
# alpha 0.2, censor ticks dimmed to alpha 0.4 (they are annotation, not data —
# undimmed they compete with the curve), and a risk table underneath.
#
# `xlim_max` truncates the *rendered window* only; the KM fit always uses the
# full range. Truncating the fit would change the estimate, not just the view.
fk_km_panel <- function(data, time_col, event_col, group_col,
                        palette = NULL, xlab = "Time", ylab = "Survival probability",
                        legend_title = "Group", legend_labs = NULL,
                        legend_position = "none",
                        xlim_max = NULL, break_time_by = 10,
                        risk_table = TRUE, risk_table_height = 0.28,
                        conf_int = TRUE, conf_int_alpha = 0.2,
                        linewidth = 0.5, censor_size = 1.5,
                        censor_alpha = 0.4, base_size = 12,
                        theme = fk_get_theme()) {
  for (p in c("survival", "survminer")) {
    if (!requireNamespace(p, quietly = TRUE))
      stop(sprintf("fk_km_panel needs the '%s' package: install.packages('%s')", p, p))
  }

  # Base-R subset, not dplyr: NSE can mask a function argument with a
  # same-named column and silently return an empty frame.
  keep <- !is.na(data[[time_col]]) & !is.na(data[[event_col]]) &
          !is.na(data[[group_col]])
  sub <- data[which(keep), , drop = FALSE]
  n_drop <- sum(!keep)
  if (n_drop > 0)
    message(sprintf("  [km] dropped %d rows with NA time/event/group", n_drop))
  if (!nrow(sub)) stop("fk_km_panel: no complete rows")
  sub$.gg <- droplevels(factor(sub[[group_col]]))

  if (is.null(palette)) {
    lv <- levels(sub$.gg)
    palette <- rep(theme$categorical, length.out = length(lv))
  } else if (!is.null(names(palette))) {
    palette <- as.character(palette[levels(sub$.gg)])
  }
  if (is.null(legend_labs)) legend_labs <- levels(sub$.gg)
  if (is.null(xlim_max)) xlim_max <- max(sub[[time_col]], na.rm = TRUE)

  # survminer's internals call formula(fit) and expect a fit built by
  # surv_fit(), which preserves the original formula for surv_summary().
  # Building the formula with sprintf() loses the call attachment, so use
  # do.call with a literal formula via bquote.
  form <- bquote(survival::Surv(.(as.name(time_col)), .(as.name(event_col))) ~ .gg)
  fit <- do.call(survminer::surv_fit, list(formula = form, data = sub))

  p <- survminer::ggsurvplot(
    fit, data = sub,
    legend = legend_position,
    conf.int = conf_int, conf.int.alpha = conf_int_alpha,
    size = linewidth,
    palette = palette,
    risk.table = risk_table, risk.table.height = risk_table_height,
    risk.table.y.text.col = TRUE, risk.table.y.text = FALSE,
    xlab = xlab, ylab = ylab,
    legend.title = legend_title, legend.labs = legend_labs,
    break.time.by = break_time_by, xlim = c(0, xlim_max),
    censor.size = censor_size,
    ggtheme = fk_theme_pub(base_size = base_size),
    tables.theme = survminer::theme_cleantable()
  )

  # Dim the censor ticks: they annotate, they aren't the estimate.
  p$plot$layers <- lapply(p$plot$layers, function(l) {
    if (inherits(l$geom, "GeomPoint")) l$aes_params$alpha <- censor_alpha
    l
  })
  p
}


# ── KM panel saver ───────────────────────────────────────────────────────────
# survminer::ggsurvplot(risk.table = TRUE) returns $plot (the curve) and
# $table (the at-risk table). Producers used to patchwork them together and
# save that as one object — but then a composer can't re-theme the pieces at
# its own size: the combined patchwork rasterises as a single grob with the
# curve/table text baked in at the producer's dimensions.
#
# So write THREE RDS files plus the standalone PNG/PDF:
#   <name>__curve.rds   the curve ggplot, untouched
#   <name>__table.rds   the risk table ggplot, untouched
#   <name>.rds          a list(kind = "km_panel", ...) pointer
#
# A composer reads the pointer, re-themes curve and table independently (table
# usually a size down so it doesn't shout), and reassembles at `ratio`.
fk_save_km_panel <- function(dir, name, ggsurv, w = 5, h = 5, ratio = c(3, 1),
                             dpi = 600) {
  if (!requireNamespace("patchwork", quietly = TRUE))
    stop("fk_save_km_panel needs 'patchwork': install.packages('patchwork')")
  stopifnot(is.list(ggsurv), inherits(ggsurv$plot, "ggplot"))
  if (!inherits(ggsurv$table, "ggplot"))
    stop("fk_save_km_panel: no risk table — build with fk_km_panel(risk_table = TRUE)")

  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  composite <- patchwork::wrap_plots(ggsurv$plot, ggsurv$table, ncol = 1,
                                     heights = ratio)
  ggsave(file.path(dir, paste0(name, ".pdf")), composite, width = w, height = h,
         device = grDevices::cairo_pdf)
  ggsave(file.path(dir, paste0(name, ".png")), composite, width = w, height = h,
         dpi = dpi)
  saveRDS(ggsurv$plot,  file.path(dir, paste0(name, "__curve.rds")))
  saveRDS(ggsurv$table, file.path(dir, paste0(name, "__table.rds")))
  saveRDS(list(kind = "km_panel", curve = paste0(name, "__curve.rds"),
               table = paste0(name, "__table.rds"), ratio = ratio),
          file.path(dir, paste0(name, ".rds")))
  invisible(file.path(dir, paste0(name, c(".png", ".pdf", ".rds"))))
}


# ── Regression scatter ───────────────────────────────────────────────────────
# Points + a fitted line with CI ribbon, grouped by a categorical variable.
#
# Points are deliberately faint (alpha 0.3, size 0.6) and the fit is thin
# (linewidth 0.5): the reader should see the trend and the spread, not a mass
# of ink. With many points, an opaque scatter hides its own density.
#
# `fixed_axis` / `fixed_breaks` / `fixed_limits` pin ONE axis to a fixed scale
# (e.g. a 0-8% VAF axis at 0/2/4/6/8) so panels sharing that measure stay
# comparable across a figure. Leave NULL to let both axes float.
fk_regression_scatter <- function(data, x_var, y_var, color_var = NULL,
                                  palette = NULL, x_lab = NULL, y_lab = NULL,
                                  fixed_axis = c("none", "x", "y"),
                                  fixed_limits = NULL, fixed_breaks = NULL,
                                  fixed_accuracy = 0.01,
                                  method = "lm", se = TRUE,
                                  point_alpha = 0.3, point_size = 0.6,
                                  smooth_lw = 0.5, smooth_alpha = 0.2,
                                  legend_title = NULL, base_size = 12,
                                  theme = fk_get_theme()) {
  fixed_axis <- match.arg(fixed_axis)

  if (is.null(color_var)) {
    # Ungrouped: one colour, still a real fit + ribbon.
    p <- ggplot(data, aes(x = .data[[x_var]], y = .data[[y_var]])) +
      geom_point(alpha = point_alpha, size = point_size, colour = theme$down) +
      geom_smooth(method = method, se = se, linewidth = smooth_lw,
                  alpha = smooth_alpha, colour = theme$down, fill = theme$down)
  } else {
    if (is.null(palette)) {
      lv <- unique(as.character(data[[color_var]]))
      palette <- setNames(rep(theme$categorical, length.out = length(lv)), lv)
    }
    p <- ggplot(data, aes(x = .data[[x_var]], y = .data[[y_var]],
                          colour = .data[[color_var]])) +
      geom_point(alpha = point_alpha, size = point_size) +
      geom_smooth(aes(fill = .data[[color_var]]), method = method, se = se,
                  linewidth = smooth_lw, alpha = smooth_alpha) +
      scale_colour_manual(values = palette, name = legend_title,
                          na.value = theme$na) +
      scale_fill_manual(values = palette, guide = "none", na.value = theme$na)
  }

  p <- p + labs(x = x_lab, y = y_lab) +
    fk_theme_pub(base_size = base_size, grid = FALSE)

  if (fixed_axis != "none") {
    lab_fn <- if (requireNamespace("scales", quietly = TRUE))
      scales::label_number(accuracy = fixed_accuracy) else waiver()
    sc <- if (fixed_axis == "y")
      scale_y_continuous(breaks = fixed_breaks, minor_breaks = NULL,
                         labels = lab_fn, limits = fixed_limits)
    else
      scale_x_continuous(breaks = fixed_breaks, minor_breaks = NULL,
                         labels = lab_fn, limits = fixed_limits)
    p <- p + sc
  }
  p
}


# ── Dumbbell prevalence ──────────────────────────────────────────────────────
# One row per feature, one dot per group, dot size ∝ N, optional 95% CI bar.
#
# No connecting line between the paired dots: with the groups dodged into
# separate y bands, a line drawn at the un-dodged centreline floats *between*
# the dots rather than through them and reads as an orphan. The vertical dodge
# alone does the pairing.
#
# `size_breaks` / `size_limits` let two dumbbells in one composite share an
# identical N legend — patchwork's `guides = "collect"` only merges legends
# whose scales match exactly, so pass the union of N across both frames.
fk_dumbbell <- function(data, feature_col = "feature", group_col = "group",
                        value_col = "prevalence", n_col = NULL,
                        lo_col = NULL, hi_col = NULL, palette = NULL,
                        size_range = c(2.5, 5), size_breaks = NULL,
                        size_limits = NULL, dodge_width = 0.6,
                        x_lab = "Prevalence (%)", x_limits = c(0, 100),
                        legend_title = NULL, base_size = 12,
                        theme = fk_get_theme()) {
  if (is.null(palette)) {
    lv <- unique(as.character(data[[group_col]]))
    palette <- setNames(rep(theme$categorical, length.out = length(lv)), lv)
  }
  dodge <- position_dodge(width = dodge_width)

  p <- ggplot(data, aes(x = .data[[value_col]], y = .data[[feature_col]]))

  if (!is.null(lo_col) && !is.null(hi_col)) {
    p <- p + geom_errorbarh(
      aes(xmin = .data[[lo_col]], xmax = .data[[hi_col]],
          colour = .data[[group_col]]),
      position = dodge, width = 0.12, linewidth = 0.6, alpha = 0.85,
      show.legend = FALSE)
  }

  p <- p + (if (is.null(n_col))
    geom_point(aes(colour = .data[[group_col]]), position = dodge,
               size = mean(size_range), alpha = 0.85)
  else
    geom_point(aes(colour = .data[[group_col]], size = .data[[n_col]]),
               position = dodge, alpha = 0.85))

  p <- p + scale_colour_manual(values = palette, name = legend_title,
                               na.value = theme$na) +
    scale_x_continuous(limits = x_limits,
                       expand = expansion(mult = c(0.02, 0.05))) +
    labs(x = x_lab, y = NULL) +
    guides(colour = guide_legend(order = 1, override.aes = list(size = 4))) +
    fk_theme_pub(base_size = base_size) +
    theme(panel.grid.major.y = element_blank())

  if (!is.null(n_col)) {
    p <- p + scale_size_continuous(
      range = size_range, name = "N", limits = size_limits,
      breaks = if (is.null(size_breaks) && requireNamespace("scales", quietly = TRUE))
        scales::breaks_pretty(3) else size_breaks) +
      guides(size = guide_legend(order = 2))
  }
  p
}
