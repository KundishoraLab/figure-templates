## gallery_clinical.R — forest / KM / regression / dumbbell panels.
##
##   Rscript demo/gallery_clinical.R                # base theme -> gallery_R/
##   Rscript demo/gallery_clinical.R --theme avm
##
## Synthetic cohort only — no patient data. Shapes mirror what the real
## producers pass in (broom/logistic output for forests, a survival frame for
## KM), so swapping in real data is a column-name change.

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
theme_name <- if ("--theme" %in% args) args[which(args == "--theme") + 1] else "base"
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=",
  commandArgs(FALSE), value = TRUE)[1])), ".."), mustWork = FALSE)
if (!dir.exists(file.path(root, "R"))) root <- getwd()

source(file.path(root, "R", "figkit.R"))
source(file.path(root, "R", "forest.R"))
source(file.path(root, "R", "clinical.R"))
source(file.path(root, "R", "themes_avm.R"))

th <- if (theme_name == "avm") fk_theme_avm() else fk_theme()
fk_set_theme(th)

out <- file.path(root, "gallery_R")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("[gallery] clinical, theme=%s -> %s\n", theme_name, out))

set.seed(0)

ok <- 0; total <- 0
render <- function(name, expr, w, h) {
  total <<- total + 1
  cat(sprintf("[render] %s\n", name))
  tryCatch({
    p <- force(expr)
    fk_save_panel(p, name, out, width = w, height = h, dpi = 200,
                  formats = "png")
    ok <<- ok + 1
  }, error = function(e) cat(sprintf("  [FAIL] %s: %s\n", name, conditionMessage(e))))
}

## ── 14: general OR forest ──────────────────────────────────────────────────
predictors <- c("Age > 40 y", "Male sex", "Deep location", "Eloquent cortex",
                "Associated aneurysm", "Deep venous drainage",
                "Nidus > 3 cm", "Prior haemorrhage")
n_pred <- length(predictors)
est <- exp(rnorm(n_pred, 0.15, 0.55))
selo <- runif(n_pred, 0.18, 0.42)
forest_gen <- data.frame(
  label   = predictors,
  est     = est,
  lo      = exp(log(est) - 1.96 * selo),
  hi      = exp(log(est) + 1.96 * selo),
  n_label = as.character(sample(40:220, n_pred)),
  p_label = sprintf("%.3f", runif(n_pred, 0.001, 0.4)),
  stringsAsFactors = FALSE
)

render("14_forest_general", {
  fk_forest_general(
    forest_gen,
    cols = list(list(col = "n_label", header = "N"),
                list(col = "p_label", header = "FDR P")),
    x_lab = "Odds ratio (95% CI)",
    est_col_header = "OR (95% CI)",
    title = "Predictors of rupture",
    base_size = 11
  )
}, 8.5, 4.2)

## ── 15: meta-analysis forest with pooled diamond ───────────────────────────
studies <- c("Cohort A (2019)", "Cohort B (2020)", "Cohort C (2021)",
             "Cohort D (2022)", "Cohort E (2024)")
s_est <- exp(rnorm(length(studies), 0.55, 0.35))
s_se <- runif(length(studies), 0.15, 0.45)
pooled_est <- exp(weighted.mean(log(s_est), 1 / s_se^2))
pooled_se <- sqrt(1 / sum(1 / s_se^2))
forest_meta <- data.frame(
  label   = c(studies, "Pooled (REML)"),
  est     = c(s_est, pooled_est),
  lo      = c(exp(log(s_est) - 1.96 * s_se), exp(log(pooled_est) - 1.96 * pooled_se)),
  hi      = c(exp(log(s_est) + 1.96 * s_se), exp(log(pooled_est) + 1.96 * pooled_se)),
  type    = c(rep("study", length(studies)), "pooled"),
  n_label = c(as.character(sample(60:400, length(studies))), "1,204"),
  stringsAsFactors = FALSE
)

render("15_forest_meta", {
  fk_forest_meta(
    forest_meta,
    cols = list(list(col = "n_label", header = "N")),
    label_header = "Study",
    x_lab = "Odds ratio (95% CI)",
    est_col_header = "OR (95% CI)",
    title = "Variant positivity across cohorts",
    # Keep directional labels short: they sit inside the forest region, which
    # is only the slice of panel width the text table doesn't use.
    left_label = "← Negative", right_label = "Variant+ →",
    base_size = 11
  )
}, 8, 3.6)

## ── 16: Kaplan-Meier with risk table ───────────────────────────────────────
n <- 400
km_df <- data.frame(
  group = factor(sample(c("KRAS G12D", "KRAS G12V", "Negative"), n, TRUE,
                        prob = c(0.35, 0.25, 0.40))),
  stringsAsFactors = FALSE
)
# Hazard differs by group so the curves actually separate.
rate <- c("KRAS G12D" = 0.045, "KRAS G12V" = 0.033, "Negative" = 0.020)
km_df$age <- pmin(rexp(n, rate[as.character(km_df$group)]), 60)
km_df$event <- as.integer(km_df$age < 55 & runif(n) < 0.75)

km_pal <- if (theme_name == "avm") {
  c("KRAS G12D" = FK_AVM_UP, "KRAS G12V" = "#e66101", "Negative" = "#5F5F5F")
} else {
  NULL
}

km <- tryCatch(
  fk_km_panel(km_df, "age", "event", "group", palette = km_pal,
              xlab = "Age (years)", ylab = "Cumulative presentation-free",
              legend_title = "Genotype", legend_position = "top",
              xlim_max = 60, break_time_by = 10, base_size = 11),
  error = function(e) { cat(sprintf("  [FAIL] 16_km: %s\n", conditionMessage(e))); NULL })

total <- total + 1
if (!is.null(km)) {
  cat("[render] 16_km_curve\n")
  fk_save_km_panel(out, "16_km_curve", km, w = 5.5, h = 5)
  # fk_save_km_panel writes its own png/pdf/rds — drop the pdf + rds so the
  # committed gallery stays PNG-only like the rest.
  unlink(file.path(out, c("16_km_curve.pdf", "16_km_curve.rds",
                          "16_km_curve__curve.rds", "16_km_curve__table.rds")))
  ok <- ok + 1
}

## ── 17: regression scatter ─────────────────────────────────────────────────
n_r <- 260
reg <- data.frame(
  genotype = factor(sample(c("KRAS G12D", "KRAS G12V"), n_r, TRUE)),
  age = runif(n_r, 2, 60)
)
# VAF declines with age, with a genotype offset — a real trend to fit.
reg$vaf <- pmax(0, 6.5 - 0.055 * reg$age +
                  ifelse(reg$genotype == "KRAS G12D", 0.8, 0) +
                  rnorm(n_r, 0, 1.1))
reg$vaf <- pmin(reg$vaf, 8)

render("17_regression_scatter", {
  fk_regression_scatter(
    reg, x_var = "age", y_var = "vaf", color_var = "genotype",
    palette = if (theme_name == "avm")
      c("KRAS G12D" = FK_AVM_UP, "KRAS G12V" = "#e66101") else NULL,
    x_lab = "Age at surgery (years)", y_lab = "VAF (%)",
    fixed_axis = "y", fixed_limits = c(0, 8), fixed_breaks = c(0, 2, 4, 6, 8),
    legend_title = "Genotype", base_size = 11
  ) + ggtitle("VAF vs age")
}, 5.5, 4)

## ── 18: dumbbell prevalence ────────────────────────────────────────────────
feats <- c("Headache", "Seizure", "Focal deficit", "Haemorrhage",
           "Incidental finding", "Deep drainage", "Aneurysm")
db <- do.call(rbind, lapply(c("Variant-positive", "Panel-negative"), function(g) {
  pr <- runif(length(feats), 8, 72) + ifelse(g == "Variant-positive", 9, 0)
  nn <- sample(30:180, length(feats))
  se <- sqrt(pr * (100 - pr) / nn)
  data.frame(feature = feats, group = g, prevalence = pr, n = nn,
             lo = pmax(0, pr - 1.96 * se), hi = pmin(100, pr + 1.96 * se),
             stringsAsFactors = FALSE)
}))

render("18_dumbbell", {
  fk_dumbbell(db, feature_col = "feature", group_col = "group",
              value_col = "prevalence", n_col = "n",
              lo_col = "lo", hi_col = "hi",
              palette = if (theme_name == "avm")
                c("Variant-positive" = FK_AVM_UP, "Panel-negative" = "#5F5F5F")
              else NULL,
              legend_title = "Genotype", base_size = 11) +
    ggtitle("Clinical features by genotype")
}, 6.5, 4)

cat(sprintf("\n[done] %d/%d panels -> %s\n", ok, total, out))
if (ok < total) quit(status = 1)
