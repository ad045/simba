"""
Paper figure for the wide-target parameter recovery (appendix: grid_and_variance)
=================================================================================

Renders ``figures/appendix/grid_and_variance.pdf`` of the benchmarking manuscript
from the cached per-measure distance CSVs produced by
``run_synthetic_gnm_comparison_grid.py``. Nothing is recomputed here.

  A  4 x 4 grid of recovered locations, one subpanel per measure: crosses mark
     the five true parameter combinations, dots the predicted cells of the 10
     test networks each. Subpanel title carries the mean grid-step error.
  B  Per-combination variability: mean std of predicted eta / gamma (normalised
     to [0, 1]) across the 10 test networks, rows ordered by mean recovery error.

Panel drawing and all error / variability maths are reused from
``run_synthetic_gnm_plots_grid.py`` - this script only composes the two panels
into one figure at the manuscript's column width.

Usage
-----
  conda activate ma_thesis
  python run_grid_and_variance_paper_plot.py
  python run_grid_and_variance_paper_plot.py --out /path/to/fig.pdf
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
import textwrap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_THIS_DIR = Path(__file__).resolve().parent
for _p in (str(_THIS_DIR.parent), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vizman import viz  # noqa: E402

import run_synthetic_gnm_plots_grid as G  # noqa: E402

OUT_PDF = Path(
    os.environ.get("MANUSCRIPT_FIGURE_DIR", str(Path(__file__).resolve().parents[1] / "figures" / "appendix"))
    + "/grid_and_variance.pdf"
)

# Manuscript width; the published figure is 15.65 x 8.0 cm.
FIG_SIZE_CM = (15.65, 8.0)
NCOLS = 4


def _panel_letter(ax, letter, x=-0.02, y=1.04, size=11):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="bottom", ha="left",
            color=G._colors["HALF_BLACK"])


def plot_grid_and_variance(results, out_pdf: Path):
    measures = list(results)
    n_meas = len(measures)
    n_rows = int(np.ceil(n_meas / NCOLS))
    n_combos = len(G.TRUE_PARAM_COMBOS)

    fig = plt.figure(figsize=viz.cm_to_inch(FIG_SIZE_CM))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.55)
    gs_a = outer[0].subgridspec(n_rows, NCOLS, hspace=1.15, wspace=0.3)
    ax_heat = fig.add_subplot(outer[1])

    # ---- Panel A: recovered locations, one subpanel per measure -----------
    axes = np.empty((n_rows, NCOLS), dtype=object)
    for idx, measure in enumerate(measures):
        row, col = divmod(idx, NCOLS)
        share = dict(sharex=axes[0, 0], sharey=axes[0, 0]) if idx else {}
        ax = fig.add_subplot(gs_a[row, col], **share)
        axes[row, col] = ax
        G._draw_scatter_ax(
            ax, results[measure], measure,
            show_xlabel=(row == n_rows - 1),
            show_ylabel=(col == 0),
            panel_letter=chr(ord("A") + idx),
        )
        # _draw_scatter_ax sizes its markers for a full-page grid; these
        # subpanels are ~5x smaller, so scale every marker down with them.
        for coll in ax.collections:
            coll.set_sizes(coll.get_sizes() / 5.0)
            if coll.get_linewidths()[0] > 0:
                coll.set_linewidths(coll.get_linewidths()[0] / 2.5)
        # Long measure names run into the neighbouring subpanel at this width.
        name, _, err_line = ax.get_title().partition("\n")
        ax.set_title("\n".join(textwrap.wrap(name, 20) + [err_line]),
                     fontsize=4.0, pad=1.5)
        ax.xaxis.label.set_size(5)
        ax.yaxis.label.set_size(5)
        ax.tick_params(labelsize=4, length=2, pad=1)
        # Subpanel letters sit inside panel A, so keep them small.
        for txt in ax.texts:
            txt.set_fontsize(5)
        if col:
            ax.tick_params(labelleft=False)
        if row != n_rows - 1:
            ax.tick_params(labelbottom=False)

    axes[0, 0].set_xlim(G.GRID_ETA[0], G.GRID_ETA[-1])
    axes[0, 0].set_ylim(G.GRID_GAMMA[0], G.GRID_GAMMA[-1])
    axes[0, 0].set_xticks([-8, 0, 3])
    axes[0, 0].set_yticks([-0.1, 0, 1])

    handles = [
        Line2D([], [], linestyle="none", marker="x", markersize=4,
               markeredgewidth=0.9, color=G.COMBO_COLORS[i],
               label=G.COMBO_LABELS[i])
        for i in range(n_combos)
    ]
    fig.legend(handles=handles, title="True parameter combination",
               loc="lower left", ncol=n_combos, frameon=False,
               fontsize=4.0, title_fontsize=4.5, handletextpad=0.2,
               columnspacing=0.8, borderaxespad=0.0,
               bbox_to_anchor=(0.0, -0.03, 0.60, 0.1), mode="expand")
    _panel_letter(axes[0, 0], "A", x=-0.45, y=1.35)

    # ---- Panel B: variability heatmap, rows ordered by recovery error -----
    err = np.array([G.compute_accuracy_error(results[m]) for m in measures])
    var = np.array([G.compute_variability(results[m])[0] for m in measures])

    order = np.argsort(err)
    var = var[order]
    labels = [G.METRIC_NAMES.get(measures[i], measures[i]) for i in order]

    im = ax_heat.imshow(var, aspect="auto", cmap="RdYlGn_r",
                        vmin=0, vmax=np.nanmax(var) or 1)
    ax_heat.set_xticks(range(n_combos))
    ax_heat.set_xticklabels(
        [rf"$\eta$={e:.1f}" "\n" rf"$\gamma$={g:.2f}"
         for e, g in G.TRUE_PARAM_COMBOS], fontsize=4.5)
    ax_heat.set_yticks(range(len(labels)))
    ax_heat.set_yticklabels(labels, fontsize=4.5)
    ax_heat.tick_params(length=2, pad=1)
    ax_heat.set_title("Variability (smaller is better)", fontsize=6, pad=4)

    threshold = (np.nanmax(var) or 1) * 0.55
    for r in range(var.shape[0]):
        for c in range(var.shape[1]):
            val = var[r, c]
            if not np.isfinite(val):
                continue
            ax_heat.text(c, r, f"{val:.2f}", ha="center", va="center",
                         fontsize=4,
                         color=G._colors["BONE_WHITE"] if val > threshold
                         else G._colors["HALF_BLACK"])

    cb = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.03)
    cb.set_label("Mean std (norm. space)", fontsize=5)
    cb.ax.tick_params(labelsize=4, length=2, pad=1)
    _panel_letter(ax_heat, "B", x=-0.42, y=1.06)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=G.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_pdf}")

    print("\nMeasure ranking by mean recovery error (grid steps):")
    for rank, i in enumerate(order, 1):
        print(f"  {rank:2d}. {labels[rank - 1]:30s} {err[i]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PDF)
    args = ap.parse_args()

    results = G.load_all_results()
    if not results:
        sys.exit(f"No CSVs in {G.COMPARISON_DIR} - "
                 "run run_synthetic_gnm_comparison_grid.py first.")
    plot_grid_and_variance(results, args.out)


if __name__ == "__main__":
    main()
