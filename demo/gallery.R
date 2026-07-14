## gallery.R — render one example panel per R plot type, from synthetic data.
##
##   Rscript demo/gallery.R                 # base theme -> gallery_R/
##   Rscript demo/gallery.R --theme avm
##
## Mirrors demo/gallery.py. Same synthetic shapes, same design decisions —
## the two tracks should be visually interchangeable.

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
theme_name <- if ("--theme" %in% args) args[which(args == "--theme") + 1] else "base"
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=",
  commandArgs(FALSE), value = TRUE)[1])), ".."), mustWork = FALSE)
if (!dir.exists(file.path(root, "R"))) root <- getwd()

source(file.path(root, "R", "figkit.R"))
source(file.path(root, "R", "themes_avm.R"))

th <- if (theme_name == "avm") fk_theme_avm() else fk_theme()
fk_set_theme(th)

out <- file.path(root, "gallery_R")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("[gallery] theme=%s -> %s\n", theme_name, out))

set.seed(0)

## ── synthetic data (mirrors demo/synth.py) ─────────────────────────────────
n_genes <- 2000
genes <- c(sprintf("GENE%04d", seq_len(n_genes - 41)),
           paste0("MT-ND", 1:6), paste0("RPL", 1:10), paste0("RPS", 1:7),
           paste0("LINC", 1000:1007), paste0("MIR", 100:103),
           "AL100000.1", "AL100001.1",
           "Negative01", "Negative02", "SystemControl1", "FalseCode3")
lfc <- rnorm(n_genes, 0, 0.35)
base_mean <- 10^runif(n_genes, 0.5, 3.5)
real <- sample(n_genes, 60)
lfc[real] <- lfc[real] + sample(c(-1, 1), 60, TRUE) * runif(60, 0.8, 3.2)
se <- 0.25 + 1.5 / sqrt(base_mean)
# 2*pnorm(-|z|) is the accurate upper tail; 1 - pnorm(|z|) underflows to 0.
pval <- pmax(2 * pnorm(-abs(lfc / se)), 1e-300)
de <- data.frame(gene = genes, log2FoldChange = lfc, pvalue = pval,
                 baseMean = base_mean, stringsAsFactors = FALSE)

hallmarks <- c("HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_HYPOXIA",
               "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
               "HALLMARK_ANGIOGENESIS", "HALLMARK_INFLAMMATORY_RESPONSE",
               "HALLMARK_KRAS_SIGNALING_UP", "HALLMARK_MYC_TARGETS_V1",
               "HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT",
               "HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_APOPTOSIS",
               "HALLMARK_P53_PATHWAY", "HALLMARK_GLYCOLYSIS",
               "HALLMARK_MTORC1_SIGNALING", "HALLMARK_NOTCH_SIGNALING",
               "HALLMARK_TGF_BETA_SIGNALING", "HALLMARK_COMPLEMENT",
               "HALLMARK_COAGULATION")
nes <- rnorm(length(hallmarks), 0, 1.4)
nes[1:4] <- nes[1:4] + runif(4, 0.8, 1.8)
nes[15:18] <- nes[15:18] - runif(4, 0.8, 1.6)
gsea <- data.frame(pathway = hallmarks, NES = nes,
                   padj = pmax(10^(-abs(nes) * runif(length(nes), 1, 2.6)), 1e-30),
                   stringsAsFactors = FALSE)

tfs <- c("RELA", "NFKB1", "JUN", "FOS", "STAT1", "STAT3", "MYC", "E2F1",
         "SP1", "KLF2", "KLF4", "HIF1A", "EPAS1", "SOX17", "ERG", "FLI1",
         "GATA2", "TCF4", "SMAD3", "TEAD1", "FOXO1", "ETS1", "SRF", "HES1")
delta <- rnorm(length(tfs), 0, 1)
delta[1:5] <- delta[1:5] + runif(5, 1, 2.5)
delta[20:24] <- delta[20:24] - runif(5, 1, 2.2)
tf <- data.frame(source = tfs, delta = delta,
                 q = pmax(10^(-abs(delta) * runif(length(delta), 0.8, 3)), 1e-25),
                 stringsAsFactors = FALSE)

ec <- FK_AVM_AV_ORDER
ct <- names(FK_CELL_TYPE_COLORS)[1:8]
labels <- c(ec, ct)
ang <- seq(0, 2 * pi, length.out = length(labels) + 1)[seq_along(labels)]
centers <- cbind(cos(ang), sin(ang)) * 7
n_per <- sample(200:600, length(labels), TRUE)
umap <- do.call(rbind, lapply(seq_along(labels), function(i) {
  data.frame(umap_1 = rnorm(n_per[i], centers[i, 1], 1.1),
             umap_2 = rnorm(n_per[i], centers[i, 2], 1.1),
             label = labels[i], stringsAsFactors = FALSE)
}))
umap$score <- rnorm(nrow(umap)) + (umap$label %in% FK_EC_VENOUS) * 2.2
umap$is_mutant <- runif(nrow(umap)) < ifelse(umap$score > 1.5, 0.25, 0.02)

pal <- c(FK_EC_SUBTYPE_COLORS, FK_CELL_TYPE_COLORS)
if (theme_name != "avm") pal <- NULL

## ── panels ─────────────────────────────────────────────────────────────────
ok <- 0; total <- 0
# The gallery is a reference, not a print artifact: PNG only at 200dpi keeps it
# a few MB instead of tens. Real panels want fk_save_panel's defaults
# (600dpi + cairo_pdf).
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

render("01_volcano", {
  fk_volcano(de, x_col = "log2FoldChange", y_col = "pvalue", label_col = "gene",
             lfc_thresh = 0.8, p_thresh = 0.01, top_n_labels = 12,
             exclude_patterns = c(FK_NOISE_PATTERNS, FK_CONTROL_PATTERNS),
             x_label = "log2FC (proximal - distal)",
             y_label = "-log10 nominal p") +
    ggtitle("Proximal vs distal")
}, 6.5, 5.5)

render("02_dotplot_pathway", {
  fk_dotplot_pathway(gsea, top_n = 16, sort_by = "signed",
                     x_label = "NES (proximal - distal)") +
    ggtitle("Hallmark enrichment")
}, 7, 5.5)

render("04_umap_categorical", {
  fk_umap_categorical(umap, "label", palette = pal, point_size = 0.5,
                      legend_title = "Cell type", legend_ncol = 2) +
    ggtitle("Atlas by cell type")
}, 8, 5)

render("06_umap_highlight", {
  fk_umap_highlight(umap, umap[umap$is_mutant, ], fg_size = 1.1) +
    ggtitle(sprintf("Variant-positive cells (n=%d)", sum(umap$is_mutant)))
}, 6, 5)

render("09_lollipop_tf", {
  fk_lollipop(tf, top_n = 22, y_label = "TF activity Δ (proximal - distal)") +
    ggtitle("Transcription-factor activity")
}, 8, 4)

render("11_ordered_bar", {
  bar <- data.frame(cell_type = ec,
                    pct_variant = pmax(runif(length(ec), 0, 3) +
                                       seq(0, 4, length.out = length(ec)), 0),
                    sig = c("", "", "", "*", "*", "**", "***", "**"),
                    stringsAsFactors = FALSE)
  fk_ordered_bar(bar, value_col = "pct_variant", label_col = "cell_type",
                 order = ec,
                 palette = if (theme_name == "avm") FK_EC_SUBTYPE_COLORS else NULL,
                 sig_col = "sig", x_label = "Variant+ cells (% of subtype)") +
    ggtitle("Variant burden along the arteriovenous axis")
}, 6.5, 3.8)

render("08_heatmap", {
  m <- matrix(rnorm(12 * 8), nrow = 12,
              dimnames = list(fk_prettify_pathway(hallmarks[1:12]), ec))
  m[1:4, 6:8] <- m[1:4, 6:8] + 1.8
  m[9:12, 1:3] <- m[9:12, 1:3] - 1.5
  m[3, 7] <- NA  # exercise the NA-grey path
  fk_heatmap(m, symmetric = TRUE, cbar_label = "signed score") +
    ggtitle("Program activity by subtype")
}, 7, 5)

cat(sprintf("\n[done] %d/%d panels -> %s\n", ok, total, out))
if (ok < total) quit(status = 1)
