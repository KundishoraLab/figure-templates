"""AVM_THEME — a worked example of a domain theme.

This is the theme the kit was extracted from (brain arteriovenous malformation
spatial transcriptomics). It is here as a **template**, not a dependency:
nothing in `figkit/` imports it. Read it to see how a domain vocabulary plugs
into `Theme`, then copy it to `themes/<yours>.py` and replace the contents.

    from figkit import set_theme
    from figkit.themes.avm import AVM_THEME
    set_theme(AVM_THEME)

The three ideas worth stealing, whatever your domain is:

1. **One contrast pair, used everywhere.** `up` (magenta) is the perturbed /
   mutant / treated pole and `down` is the reference pole, in every panel of
   every figure. The reader learns the color once.

2. **Ordered categories get an ordered ramp.** EC subtypes run artery -> vein,
   a real anatomical axis, so their colors run red -> purple -> blue along it.
   `AV_CMAP` is that same ramp made continuous, so a continuous score and the
   discrete subtypes read as one scale. If your categories have no order,
   don't fake one — use WONG or KELLY.

3. **Off-axis states are deliberately desaturated greys.** They exist, they
   are labelled, and they visually recede — the panel is about the axis.
"""
from __future__ import annotations

from ..palettes import WONG, gradient_cmap
from ..theme import Theme

# ── Directional contrast ────────────────────────────────────────────────────
UP_COLOR = "#c51b8a"   # magenta — niche / AVM / mutant / state-high pole
DN_PROX = "#0a9396"    # teal    — distal pole (proximal-vs-distal contrast)
DN_MR = "#1f3a6e"      # navy    — reference pole (mutant-vs-reference contrast)
NEUTRAL = "#f5f5f5"

# Which "down" color belongs to which contrast. Two contrasts sharing one
# down-color would be indistinguishable across panels.
DN_BY_CONTRAST = {
    "proximal_distal": DN_PROX,
    "mutant_reference": DN_MR,
}

# ── Endothelial subtypes along the arteriovenous axis ───────────────────────
AV_ORDER = ["Large artery", "Artery", "Arteriole", "Capillary",
            "Angiogenic capillary", "Venule", "Vein", "Large vein"]

EC_SUBTYPE_COLORS = {
    # Arterial -> capillary -> venous: warm red through purple to cold blue.
    "Large artery":         "#8B1E1E",
    "Artery":               "#C0392B",
    "Arteriole":            "#E74C3C",
    "Capillary":            "#9B59B6",
    "Angiogenic capillary": "#6C3483",
    "Venule":               "#2471A3",
    "Vein":                 "#1A5276",
    "Large vein":           "#0E2F43",
    # Off-axis EC states — greys, so they recede (see docstring note 3).
    "EndoMT":               "#7F8C8D",
    "Stem-to-EC":           "#95A5A6",
    "Proliferating cell":   "#BDC3C7",
    "Mitochondrial":        "#D5DBDB",
    "Lymphatic":            "#1ABC9C",
}

EC_NONAV_ORDER = ["EndoMT", "Stem-to-EC", "Proliferating cell", "Mitochondrial"]
EC_SUBTYPE_ORDER = AV_ORDER + EC_NONAV_ORDER
EC_VENOUS = ("Venule", "Vein", "Large vein")

# Continuous ramp derived from the ordered categories — see docstring note 2.
AV_CMAP = gradient_cmap([EC_SUBTYPE_COLORS[s] for s in AV_ORDER],
                        name="arteriovenous")

# ── Parenchymal cell types — Wong colorblind-safe, distinct from the EC ramp ─
CELL_TYPE_COLORS = {
    "Astrocytes":                "#117733",
    "Neurons":                   "#332288",
    "Neuron progenitor":         "#88CCEE",
    "Oligodendrocytes":          "#999933",
    "Microglia and Macrophages": "#CC6677",
    "T cells":                   "#AA4499",
    "Fibroblasts":               "#DDCC77",
    "Pericytes":                 "#882255",
    "Smooth muscle cells":       "#44AA99",
    "Stem cells":                "#888888",
}

# ── Sample and variant vocabularies ─────────────────────────────────────────
SAMPLE_COLORS = {
    "AVM_01": "#5e3c99",
    "AVM_04": "#e66101",
    "AVM_05": "#1b9e77",
    "AVM_06": "#a6611a",
}

MUTATION_COLORS = {
    "KRAS_G12D":  UP_COLOR,
    "KRAS_G12V":  "#e66101",
    "BRAF_V600E": "#5e3c99",
    "?":          "#bdbdbd",   # unknown is grey, never a real hue
}

PATHWAY_CLASS_COLORS = {
    "MAPK":     "#882255",
    "RTK":      "#DDCC77",
    "PI3K-AKT": "#332288",
    "SFK":      "#44AA99",
    "BCR":      "#88CCEE",
    "DDR":      "#999933",
    "IAP":      "#117733",
    "Other":    "#888888",
}


def normalize_ec_subtype(label) -> str:
    """Collapse the label variants that leak in from different tools.

    Real vocabularies drift: one tool prefixes `EC_`, another underscores the
    spaces. Normalizing at read time beats maintaining N spellings in the
    palette — an unmatched key silently becomes grey "NA", which looks like
    data rather than a bug.
    """
    s = str(label)
    if s.startswith("EC_"):
        s = s[3:]
    return s.replace("Angiogenic_capillary", "Angiogenic capillary")


def ec_order_key(label) -> int:
    """Sort key: AV axis first, then off-axis states, then unknown."""
    s = normalize_ec_subtype(label)
    return EC_SUBTYPE_ORDER.index(s) if s in EC_SUBTYPE_ORDER else len(EC_SUBTYPE_ORDER)


def is_venous(label) -> bool:
    return normalize_ec_subtype(label) in EC_VENOUS


def dn_for_contrast(contrast: str) -> str:
    return DN_BY_CONTRAST.get(contrast, DN_PROX)


AVM_THEME = Theme(
    name="avm",
    up=UP_COLOR,
    down=DN_PROX,
    neutral=NEUTRAL,
    categorical=tuple(WONG),
    palettes={
        "ec_subtype":    EC_SUBTYPE_COLORS,
        "cell_type":     CELL_TYPE_COLORS,
        "sample":        SAMPLE_COLORS,
        "mutation":      MUTATION_COLORS,
        "pathway_class": PATHWAY_CLASS_COLORS,
    },
)

# The mutant-vs-reference contrast uses navy as its down-pole (see
# DN_BY_CONTRAST). Same theme, one field swapped.
AVM_THEME_MUTREF = AVM_THEME.derive(name="avm_mutref", down=DN_MR)
