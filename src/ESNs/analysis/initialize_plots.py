# initialize_plots.py
"""
Central plotting config (fonts, dpi, colormaps, save formats).
Call `apply_plot_style()` once before creating figures.
"""

from matplotlib import pyplot as plt

# Global style knobs (edit once, all plots follow)
FIG_DPI = 180
FONT_SIZE = 12
TITLE_SIZE = 13
LABEL_SIZE = 12
TICK_SIZE = 10
CMAP_DIST = "Blues"    # distance matrices
CMAP_ENERGY = "hot"    # energy landscapes
CMAP_GENERIC = "viridis"
SAVE_FORMATS = ("png",)  # add "pdf", "svg" if you like
PAD_INCHES = 0.02

def apply_plot_style() -> None:
    """Apply global matplotlib rcParams based on constants above."""
    plt.rcParams.update({
        "figure.dpi": FIG_DPI,
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": PAD_INCHES,
    })

def save_figure(fig, outpath_no_ext: str) -> None:
    """Save fig to all formats in SAVE_FORMATS."""
    for ext in SAVE_FORMATS:
        fig.savefig(f"{outpath_no_ext}.{ext}")
