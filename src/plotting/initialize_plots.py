"""
Centralised plotting configuration.

This module defines global style parameters for all plots in the project.  To
change the appearance of figures (fonts, colour maps, file formats, etc.),
modify the variables below rather than editing individual plotting functions.

The default colour schemes follow conventions used throughout the thesis:

* Distance matrices are visualised with the ``Blues`` colormap.
* Energy landscapes are visualised with the ``hot`` colormap (dark=low energy,
  light=high energy).

Users can override these settings to suit their own preferences.  See
``apply_plot_style`` for how these values are applied to matplotlib.
"""

from __future__ import annotations

# Colour maps for common plot types
PLOT_COLORMAP_DISTANCE: str = "Blues"
"""Colormap used for visualising distance matrices."""

PLOT_COLORMAP_ENERGY: str = "hot"
"""Colormap used for energy heatmaps (low values appear dark)."""

# Default figure size (in inches) for heatmaps and similar plots
PLOT_FIGSIZE: tuple[float, float] = (6.2, 5.2)
"""Default size of figures created by plotting functions (width, height)."""

# Default file format for saving figures (extensions such as 'png', 'pdf', 'svg')
PLOT_SAVE_FORMAT: str = "png"
"""File format used when saving plots if none is specified by the caller."""

# Font size used for axis labels and titles
PLOT_FONT_SIZE: int = 12
"""Base font size for all text elements in plots."""

def apply_plot_style() -> None:
    """Apply the global plotting style to matplotlib.

    Call this function once before creating any figures to ensure that
    matplotlib uses the global settings defined in this module.  It updates
    ``matplotlib.rcParams`` with the font size defined in ``PLOT_FONT_SIZE``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from initialize_plots import apply_plot_style
    >>> apply_plot_style()
    >>> plt.figure()
    """
    # Delayed import to avoid importing matplotlib in modules that do not need it
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": PLOT_FONT_SIZE,
    })
