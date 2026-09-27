"""Shared measure legend strip for the manuscript figures (legend_8_measures.pdf).

The figures themselves carry no measure legend; this strip is placed underneath
them in the manuscript. Colours come from the same METRIC_COLORS the figures use,
so the strip cannot drift from them. The previous hand-made version still listed
Resistance, which left the plausibility filter, instead of communicability
correlation - regenerating from the config is what keeps the two in step.

    conda activate ma_thesis
    python make_legend_8_measures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from vizman import viz

from experiments_config import METRIC_COLORS, SELECTED_MEASURES

ROOT_DIR = Path(__file__).resolve().parent
OUT_PDF = ROOT_DIR / "figures" / "legend_8_measures.pdf"

# Column-major over 4 columns x 2 rows, keeping the layout of the version this
# replaces. Labels are the short forms used in the figure panels, not the longer
# prose names.
LABELS = {
    "frobenius": "Frobenius",
    "delta_con": "DeltaCon",
    "communicability_corr": "Communicability Correlation",
    "netrd_non_backtracking_spectral": "Spectral (Non-Backtracking)",
    "net_simile": "NetSimile",
    "spectral_distance_adjacency": "Spectral (Adjacency)",
    "portrait": "Portrait",
    "energy": "Energy",
}
ORDER = list(LABELS)


def main() -> None:
    assert set(ORDER) == set(SELECTED_MEASURES), \
        f"legend and SELECTED_MEASURES disagree: {set(ORDER) ^ set(SELECTED_MEASURES)}"

    fig = plt.figure(figsize=viz.cm_to_inch((18, 1.2)))
    handles = [Patch(facecolor=METRIC_COLORS[m], edgecolor="none", label=LABELS[m])
               for m in ORDER]
    fig.legend(handles=handles, 
               loc="center", ncol=4, frameon=False, # fontsize=8,
            #    handlelength=1.1, 
            #    handleheight=0.85, 
            #    columnspacing=1.4,
            #    labelspacing=0.5, handletextpad=0.5, borderpad=0
               )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
