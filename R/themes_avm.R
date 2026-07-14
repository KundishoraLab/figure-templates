## themes_avm.R — R mirror of figkit/themes/avm.py.
##
## A worked example, not a dependency: figkit.R never sources this. Copy it to
## themes_<yours>.R and replace the vocabularies. The annotated rationale for
## each choice lives in the Python twin — read that first.
##
##   source("R/figkit.R"); source("R/themes_avm.R")
##   fk_set_theme(fk_theme_avm())

FK_AVM_UP      <- "#c51b8a"  # magenta — niche / AVM / mutant / state-high pole
FK_AVM_DN_PROX <- "#0a9396"  # teal    — distal pole
FK_AVM_DN_MR   <- "#1f3a6e"  # navy    — reference pole
FK_AVM_NEUTRAL <- "#f5f5f5"

# Ordered categories get an ordered ramp: the arteriovenous axis is real
# anatomy, so colour runs warm-red (arterial) -> purple -> cold-blue (venous).
FK_AVM_AV_ORDER <- c("Large artery", "Artery", "Arteriole", "Capillary",
                     "Angiogenic capillary", "Venule", "Vein", "Large vein")

FK_EC_SUBTYPE_COLORS <- c(
  "Large artery"         = "#8B1E1E",
  "Artery"               = "#C0392B",
  "Arteriole"            = "#E74C3C",
  "Capillary"            = "#9B59B6",
  "Angiogenic capillary" = "#6C3483",
  "Venule"               = "#2471A3",
  "Vein"                 = "#1A5276",
  "Large vein"           = "#0E2F43",
  # Off-axis EC states — greys, so they recede. The panel is about the axis.
  "EndoMT"               = "#7F8C8D",
  "Stem-to-EC"           = "#95A5A6",
  "Proliferating cell"   = "#BDC3C7",
  "Mitochondrial"        = "#D5DBDB",
  "Lymphatic"            = "#1ABC9C"
)

FK_EC_NONAV_ORDER   <- c("EndoMT", "Stem-to-EC", "Proliferating cell",
                         "Mitochondrial")
FK_EC_SUBTYPE_ORDER <- c(FK_AVM_AV_ORDER, FK_EC_NONAV_ORDER)
FK_EC_VENOUS        <- c("Venule", "Vein", "Large vein")

# Parenchyma — Wong colorblind-safe, deliberately off the EC ramp.
FK_CELL_TYPE_COLORS <- c(
  "Astrocytes"                = "#117733",
  "Neurons"                   = "#332288",
  "Neuron progenitor"         = "#88CCEE",
  "Oligodendrocytes"          = "#999933",
  "Microglia and Macrophages" = "#CC6677",
  "T cells"                   = "#AA4499",
  "Fibroblasts"               = "#DDCC77",
  "Pericytes"                 = "#882255",
  "Smooth muscle cells"       = "#44AA99",
  "Stem cells"                = "#888888"
)

FK_SAMPLE_COLORS <- c("AVM_01" = "#5e3c99", "AVM_04" = "#e66101",
                      "AVM_05" = "#1b9e77", "AVM_06" = "#a6611a")

FK_MUTATION_COLORS <- c("KRAS_G12D" = FK_AVM_UP, "KRAS_G12V" = "#e66101",
                        "BRAF_V600E" = "#5e3c99",
                        "?" = "#bdbdbd")  # unknown is grey, never a real hue

# Collapse label variants that leak in from different tools. An unmatched
# palette key silently becomes grey NA, which looks like data, not a bug.
fk_normalize_ec_subtype <- function(x) {
  s <- as.character(x)
  s <- sub("^EC_", "", s)
  gsub("Angiogenic_capillary", "Angiogenic capillary", s)
}

fk_is_venous <- function(x) fk_normalize_ec_subtype(x) %in% FK_EC_VENOUS

fk_theme_avm <- function(contrast = c("proximal_distal", "mutant_reference")) {
  contrast <- match.arg(contrast)
  down <- if (contrast == "mutant_reference") FK_AVM_DN_MR else FK_AVM_DN_PROX
  fk_theme(
    name = paste0("avm_", contrast),
    up = FK_AVM_UP, down = down, neutral = FK_AVM_NEUTRAL,
    categorical = FK_WONG,
    palettes = list(
      ec_subtype = FK_EC_SUBTYPE_COLORS,
      cell_type  = FK_CELL_TYPE_COLORS,
      sample     = FK_SAMPLE_COLORS,
      mutation   = FK_MUTATION_COLORS
    )
  )
}
