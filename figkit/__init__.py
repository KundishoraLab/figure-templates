"""figkit — a modular publication-figure toolkit for single-cell / spatial omics.

    import matplotlib.pyplot as plt
    from figkit import apply_rcparams, save_panel
    from figkit.plots import volcano

    apply_rcparams()
    fig, ax = plt.subplots(figsize=(6, 5))
    volcano(de_table, ax, x_col="log2FoldChange", y_col="pvalue")
    save_panel(fig, "panel_a_volcano", "panels/")

Retheme everything at once by installing a theme:

    from figkit import set_theme
    from figkit.themes.avm import AVM_THEME
    set_theme(AVM_THEME)

See `figkit/themes/avm.py` for a worked domain theme, and `demo/gallery.py`
for a runnable example of every plot type.
"""
from .palettes import (
    KELLY, NAVY_YELLOW, NA_COLOR, PUOR_DIV, WONG, YLGNBU3,
    categorical_palette, diverging_cmap, gradient_cmap, quantile_norm,
    sequential_cmap,
)
from .style import apply_rcparams, despine, group_strip, save_panel, size_legend
from .theme import BASE_THEME, Theme, get_theme, set_theme

__version__ = "0.1.0"

__all__ = [
    "Theme", "BASE_THEME", "set_theme", "get_theme",
    "apply_rcparams", "save_panel", "despine", "size_legend", "group_strip",
    "KELLY", "WONG", "NA_COLOR", "YLGNBU3", "NAVY_YELLOW", "PUOR_DIV",
    "diverging_cmap", "sequential_cmap", "gradient_cmap", "quantile_norm",
    "categorical_palette",
    "__version__",
]
