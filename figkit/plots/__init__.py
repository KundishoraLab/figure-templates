"""Plot functions. Every one takes an `ax` and returns a summary dict or artist.

The `ax`-in contract is deliberate: it means any plot composes into a
multi-panel figure with plt.subplots / GridSpec without modification. Only
`network.py` breaks it, because pycirclize and upsetplot build their own
figures.
"""
from .bars import ordered_bar, stacked_bar, stacked_hbar
from .dotplot import (
    dotplot_expression, dotplot_gsea, dotplot_pathway, expression_matrix,
    prettify_pathway,
)
from .embedding import (
    add_scale_bar, embedding_axes, embedding_categorical, embedding_continuous,
    get_embedding, highlight_mask, side_legend, two_layer_scatter,
)
from .heatmap import heatmap, heatmap_complex
from .lollipop import lollipop, lollipop_tf
from .network import chord, chord_signed, upset
from .sankey import sankey
from .volcano import (
    CONTROL_PROBE_PATTERNS, NOISE_GENE_PATTERNS, filter_labels, volcano,
)

__all__ = [
    "violin", "box",
    "volcano", "NOISE_GENE_PATTERNS", "CONTROL_PROBE_PATTERNS", "filter_labels",
    "dotplot_pathway", "dotplot_gsea", "dotplot_expression",
    "expression_matrix", "prettify_pathway",
    "get_embedding", "embedding_axes", "embedding_categorical",
    "embedding_continuous", "two_layer_scatter", "highlight_mask",
    "side_legend", "add_scale_bar",
    "heatmap", "heatmap_complex",
    "lollipop", "lollipop_tf",
    "stacked_bar", "stacked_hbar", "ordered_bar",
    "chord", "chord_signed", "upset",
    "sankey",
]

from .distribution import violin, box  # noqa: E402,F401
