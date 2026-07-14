"""Theme: the single object every plot function pulls its colors from.

The point of this indirection is that you retheme a whole figure set by
swapping one object, not by editing hex codes in thirty renderers:

    from figkit import set_theme
    from figkit.themes.avm import AVM_THEME
    set_theme(AVM_THEME)

`up`/`down` are a *directional contrast* pair (treated-vs-control,
mutant-vs-wildtype, ...). Keep the same pair across every panel in a figure —
if magenta means "up" in the volcano it must mean "up" in the pathway dotplot,
or the reader has to relearn the color at every panel.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .palettes import (
    KELLY, NA_COLOR, BG_COLOR, PUOR_DIV,
    diverging_cmap, sequential_cmap,
)


@dataclass(frozen=True)
class Theme:
    """Colors + typography for a figure set.

    up / down    : the directional contrast pair (see module docstring)
    neutral      : midpoint of diverging ramps; also non-significant points
    categorical  : ordered list cycled for unordered categories
    palettes     : named {category: hex} maps for domain vocabularies, e.g.
                   {"cell_type": {"Astrocyte": "#117733", ...}}. Plot
                   functions never read this directly — you pass
                   `theme.palette("cell_type")` in as an argument.
    """

    name: str = "base"

    # Directional contrast
    up: str = "#c51b8a"        # magenta
    down: str = "#1f3a6e"      # navy
    neutral: str = "#f5f5f5"   # diverging midpoint
    nonsig: str = "#cccccc"    # non-significant points
    na: str = NA_COLOR
    background: str = BG_COLOR

    # Categorical
    categorical: tuple = tuple(KELLY)

    # Typography
    font_family: tuple = ("Arial", "Helvetica", "DejaVu Sans")
    base_size: float = 12.0
    dpi: int = 600

    # Domain vocabularies
    palettes: dict = field(default_factory=dict)

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def diverging(self):
        """up <-> down diverging cmap. Center it with quantile_norm(symmetric=True)."""
        return diverging_cmap(self.down, self.up, self.neutral,
                              name=f"{self.name}_div")

    @property
    def sequential(self):
        """white -> up sequential cmap for magnitude-only data."""
        return sequential_cmap(self.up, name=f"{self.name}_seq")

    @property
    def heatmap_cmap(self):
        return PUOR_DIV

    def palette(self, key: str) -> dict:
        """Return a named domain palette, e.g. theme.palette('cell_type')."""
        if key not in self.palettes:
            raise KeyError(
                f"theme {self.name!r} has no palette {key!r}; "
                f"available: {sorted(self.palettes)}")
        return dict(self.palettes[key])

    def derive(self, **kw) -> "Theme":
        """Return a copy with fields overridden: BASE.derive(up='#d62728')."""
        return replace(self, **kw)


BASE_THEME = Theme()

_active: Theme = BASE_THEME


def set_theme(theme: Theme) -> Theme:
    """Install `theme` as the default for all plot functions. Returns it."""
    global _active
    if not isinstance(theme, Theme):
        raise TypeError(f"expected Theme, got {type(theme).__name__}")
    _active = theme
    return theme


def get_theme() -> Theme:
    """The active theme. Plot functions call this when `theme=None`."""
    return _active


def resolve(theme: Theme | None) -> Theme:
    """Internal: per-call theme override, else the active theme."""
    return theme if theme is not None else _active
