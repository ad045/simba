# """
# Paper figure for the wide-target parameter recovery (appendix: grid_and_variance)
# =================================================================================

# Renders ``figures/appendix/grid_and_variance.pdf`` of the benchmarking manuscript
# from the cached per-measure distance CSVs produced by
# ``run_synthetic_gnm_comparison_grid.py``. Nothing is recomputed here.

#   A  4 x 4 grid of recovered locations, one subpanel per measure: crosses mark
#      the five true parameter combinations, dots the predicted cells of the 10
#      test networks each. Thin lines connect each true combination to every one
#      of its predicted points, coloured by that point's recovery error. Subpanel
#      title carries the mean grid-step error.
#   B  Per-combination variability: mean std of predicted eta / gamma (normalised
#      to [0, 1]) across the 10 test networks, rows ordered by mean recovery error.

# Panel drawing and all error / variability maths are reused from
# ``run_synthetic_gnm_plots_grid.py`` - this script only composes the two panels
# into one figure at the manuscript's column width.

# Colour scheme
# -------------
# Fixed four-colour palette, used instead of G's defaults for this figure only:

#   DARK_GRAY  #232324   low end of the combo gradient (panel A)
#   LIGHT_BLUE #3EA5C4   high end of the combo gradient / low end of the error ramp
#   ORANGE     #FE961F   high end of the error ramp / high end of panel B
#   BONE_WHITE #FFFCF2   low end of panel B (variability heatmap)

# Panel A's five combination colours are sampled evenly along DARK_GRAY -> LIGHT_BLUE,
# re-colouring whatever G._draw_scatter_ax already drew (matched to the original
# G.COMBO_COLORS[i] it used). The true -> predicted connecting lines use a separate
# LIGHT_BLUE -> ORANGE ramp, normalised globally across every subpanel so line colour
# is comparable panel-to-panel. Panel B swaps RdYlGn_r for a BONE_WHITE -> ORANGE ramp
# (still "smaller is better").

# Usage
# -----
#   conda activate ma_thesis
#   python run_grid_and_variance_paper_plot.py
#   python run_grid_and_variance_paper_plot.py --out /path/to/fig.pdf
# """

# import os
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# import sys
# import argparse
# import textwrap
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.markers as mmarkers
# from matplotlib.lines import Line2D
# from matplotlib.collections import LineCollection

# _THIS_DIR = Path(__file__).resolve().parent
# for _p in (str(_THIS_DIR.parent), str(_THIS_DIR)):
#     if _p not in sys.path:
#         sys.path.insert(0, _p)

# from vizman import viz  # noqa: E402

# import run_synthetic_gnm_plots_grid as G  # noqa: E402

# OUT_PDF = Path(
#     os.environ.get("MANUSCRIPT_FIGURE_DIR", str(Path(__file__).resolve().parents[1] / "figures" / "appendix"))
#     + "/grid_and_variance.pdf"
# )

# # Manuscript width; the published figure is 15.65 x 8.0 cm.
# FIG_SIZE_CM = (16, 8) # 15.65, 8.0)
# NCOLS = 4

# # ---------------------------------------------------------------------------
# # Fixed four-colour palette (overrides G's defaults for this figure only)
# # ---------------------------------------------------------------------------
# DARK_GRAY = "#232324"
# ORANGE = "#FE961F"
# LIGHT_BLUE = "#3EA5C4"
# BONE_WHITE = "#FFFCF2"

# # Panel A: one colour per true combination, sampled along this ramp.
# COMBO_CMAP = mcolors.LinearSegmentedColormap.from_list("combo_grad", [DARK_GRAY, LIGHT_BLUE])
# # Panel A: connecting lines, coloured by recovery error (short -> long).
# ERROR_CMAP = mcolors.LinearSegmentedColormap.from_list("error_grad", [LIGHT_BLUE, ORANGE])
# # Panel B: variability heatmap (small -> large, "smaller is better").
# VAR_CMAP = mcolors.LinearSegmentedColormap.from_list("var_grad", [BONE_WHITE, ORANGE])

# DOT_ALPHA = 0.55    # prediction dots: translucent so overlapping points stay visible
# LINE_ALPHA = 0.55   # connecting lines: same idea
# LINE_WIDTH = 0.35

# _X_PATH = mmarkers.MarkerStyle("x").get_path()


# def _combo_colors(n_combos):
#     if n_combos == 1:
#         return [COMBO_CMAP(0.0)]
#     return [COMBO_CMAP(i / (n_combos - 1)) for i in range(n_combos)]


# def _is_cross(coll):
#     """Crosses are drawn with matplotlib's 'x' marker path; dots aren't."""
#     paths = coll.get_paths()
#     if not paths:
#         return False
#     verts = paths[0].vertices
#     return verts.shape == _X_PATH.vertices.shape and np.allclose(verts, _X_PATH.vertices)


# def _restyle_and_collect_lines(ax, n_combos):
#     """Re-colour an already-drawn Panel-A subpanel onto COMBO_CMAP and return the
#     (true_xy, pred_xy, error_length) segments to connect. Lines aren't drawn here:
#     their colour needs a global error norm computed across every subpanel first.

#     Matching assumption: G._draw_scatter_ax paints a combination's cross and its
#     10 dots with the same original colour, G.COMBO_COLORS[i]. If a future version
#     of G changes that, this matching needs to change with it.
#     """
#     new_colors = _combo_colors(n_combos)
#     orig_hex = [mcolors.to_hex(c) for c in G.COMBO_COLORS[:n_combos]]

#     crosses, dots = {}, {}
#     for coll in ax.collections:
#         fc, ec = coll.get_facecolor(), coll.get_edgecolor()
#         ref = fc if len(fc) else ec
#         if not len(ref):
#             continue
#         try:
#             idx = orig_hex.index(mcolors.to_hex(ref[0]))
#         except ValueError:
#             continue
#         new_c = new_colors[idx]
#         if _is_cross(coll):
#             coll.set_edgecolor(new_c)
#             if len(fc):
#                 coll.set_facecolor(new_c)
#             crosses[idx] = np.asarray(coll.get_offsets())[0]
#         else:
#             coll.set_facecolor(mcolors.to_rgba(new_c, DOT_ALPHA))
#             coll.set_edgecolor(mcolors.to_rgba(new_c, DOT_ALPHA))
#             dots.setdefault(idx, []).append(np.asarray(coll.get_offsets()))

#     segments = []
#     for idx, true_xy in crosses.items():
#         for offs in dots.get(idx, []):
#             for xy in offs:
#                 length = float(np.hypot(xy[0] - true_xy[0], xy[1] - true_xy[1]))
#                 segments.append((true_xy, xy, length))
#     return segments


# def _panel_letter(ax, letter, x=-0.02, y=1.04, size=11):
#     ax.text(x, y, letter, transform=ax.transAxes,
#             fontsize=size, fontweight="bold", va="bottom", ha="left",
#             color=DARK_GRAY)


# def plot_grid_and_variance(results, out_pdf: Path):
#     measures = list(results)
#     n_meas = len(measures)
#     n_rows = int(np.ceil(n_meas / NCOLS))
#     n_combos = len(G.TRUE_PARAM_COMBOS)
#     combo_colors = _combo_colors(n_combos)

#     fig = plt.figure(figsize=viz.cm_to_inch(FIG_SIZE_CM))
#     outer = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.55)
#     gs_a = outer[0].subgridspec(n_rows, NCOLS, wspace=0.3, hspace=0.4) # , hspace=1.15, wspace=0.3)
#     ax_heat = fig.add_subplot(outer[1])

#     # ---- Panel A: recovered locations, one subpanel per measure -----------
#     axes = np.empty((n_rows, NCOLS), dtype=object)
#     axis_segments = []  # (ax, [(true_xy, pred_xy, length), ...]) per subpanel
#     for idx, measure in enumerate(measures):
#         row, col = divmod(idx, NCOLS)
#         share = dict(sharex=axes[0, 0], sharey=axes[0, 0]) if idx else {}
#         ax = fig.add_subplot(gs_a[row, col], **share)
#         axes[row, col] = ax
#         G._draw_scatter_ax(
#             ax, results[measure], measure,
#             show_xlabel=(row == n_rows - 1),
#             show_ylabel=(col == 0),
#             panel_letter=chr(ord("A") + idx),
#         )
#         # _draw_scatter_ax sizes its markers for a full-page grid; these
#         # subpanels are ~5x smaller, so scale every marker down with them.
#         for coll in ax.collections:
#             coll.set_sizes(coll.get_sizes() / 5.0)
#             if coll.get_linewidths()[0] > 0:
#                 coll.set_linewidths(coll.get_linewidths()[0] / 2.5)
#         # Recolour onto the fixed palette and stash the true -> predicted
#         # segments; their line colour needs a global error norm, added below.
#         axis_segments.append((ax, _restyle_and_collect_lines(ax, n_combos)))
#         # Long measure names run into the neighbouring subpanel at this width.
#         name, _, err_line = ax.get_title().partition("\n")
#         ax.set_title("\n".join(textwrap.wrap(name, 20) + [err_line]),
#                      fontsize=4.0, pad=1.5)
#         ax.xaxis.label.set_size(5)
#         ax.yaxis.label.set_size(5)
#         ax.tick_params(labelsize=4, length=2, pad=1)
#         # Subpanel letters sit inside panel A, so keep them small.
#         for txt in ax.texts:
#             txt.set_fontsize(5)
#         if col:
#             ax.tick_params(labelleft=False)
#         if row != n_rows - 1:
#             ax.tick_params(labelbottom=False)

#     # Thin lines from each true combination to every one of its predicted
#     # points, coloured by recovery error on one scale shared across all
#     # subpanels - drawn now, after the loop, so that scale can be built from
#     # every subpanel's segments at once.
#     all_lengths = [seg[2] for _, segs in axis_segments for seg in segs]
#     if all_lengths:
#         norm = mcolors.Normalize(vmin=min(all_lengths), vmax=max(all_lengths))
#         for ax, segs in axis_segments:
#             if not segs:
#                 continue
#             lines = [(t, p) for t, p, _ in segs]
#             colors = [ERROR_CMAP(norm(length)) for _, _, length in segs]
#             lc = LineCollection(lines, colors=colors, linewidths=LINE_WIDTH,
#                                  alpha=LINE_ALPHA, zorder=0.5)
#             ax.add_collection(lc)

#     axes[0, 0].set_xlim(G.GRID_ETA[0], G.GRID_ETA[-1])
#     axes[0, 0].set_ylim(G.GRID_GAMMA[0], G.GRID_GAMMA[-1])
#     axes[0, 0].set_xticks([-8, 0, 3])
#     axes[0, 0].set_yticks([-0.1, 0, 1])

#     handles = [
#         Line2D([], [], linestyle="none", marker="x", markersize=4,
#                markeredgewidth=0.9, color=combo_colors[i],
#                label=G.COMBO_LABELS[i])
#         for i in range(n_combos)
#     ]
#     fig.legend(handles=handles, title="True parameter combination",
#                loc="lower left", ncol=2, # n_combos, 
#                frameon=False,
#                fontsize=4.0, title_fontsize=4.5, # handletextpad=0.2,
#                columnspacing=0.8, 
#                borderaxespad=0.0,
#                bbox_to_anchor=(0.0, -0.03, 0.60, 0.1), mode="expand")
#     _panel_letter(axes[0, 0], "A", x=-0.45, y=1.35)

#     # ---- Panel B: variability heatmap, rows ordered by recovery error -----
#     err = np.array([G.compute_accuracy_error(results[m]) for m in measures])
#     var = np.array([G.compute_variability(results[m])[0] for m in measures])

#     order = np.argsort(err)
#     var = var[order]
#     labels = [G.METRIC_NAMES.get(measures[i], measures[i]) for i in order]

#     im = ax_heat.imshow(var, aspect="auto", cmap=VAR_CMAP,
#                         vmin=0, vmax=np.nanmax(var) or 1)
#     ax_heat.set_xticks(range(n_combos))
#     ax_heat.set_xticklabels(
#         [rf"$\eta$={e:.1f}" "\n" rf"$\gamma$={g:.2f}"
#          for e, g in G.TRUE_PARAM_COMBOS], fontsize=4.5)
#     ax_heat.set_yticks(range(len(labels)))
#     ax_heat.set_yticklabels(labels, fontsize=4.5)
#     ax_heat.tick_params(length=2, pad=1)
#     ax_heat.set_title("Variability (smaller is better)", fontsize=6, pad=4)

#     # BONE_WHITE -> ORANGE stays light-to-midtone throughout, so dark text
#     # reads cleanly across the whole range (unlike the old RdYlGn_r, which
#     # needed a light/dark switch to stay legible on its darker red end).
#     for r in range(var.shape[0]):
#         for c in range(var.shape[1]):
#             val = var[r, c]
#             if not np.isfinite(val):
#                 continue
#             ax_heat.text(c, r, f"{val:.2f}", ha="center", va="center",
#                          fontsize=4, color=DARK_GRAY)

#     cb = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.03)
#     cb.set_label("Mean std (norm. space)", fontsize=5)
#     cb.ax.tick_params(labelsize=4, length=2, pad=1)
#     _panel_letter(ax_heat, "B", x=-0.42, y=1.06)

#     out_pdf.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_pdf, dpi=G.DPI, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved -> {out_pdf}")

#     print("\nMeasure ranking by mean recovery error (grid steps):")
#     for rank, i in enumerate(order, 1):
#         print(f"  {rank:2d}. {labels[rank - 1]:30s} {err[i]:.3f}")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", type=Path, default=OUT_PDF)
#     args = ap.parse_args()

#     results = G.load_all_results()
#     if not results:
#         sys.exit(f"No CSVs in {G.COMPARISON_DIR} - "
#                  "run run_synthetic_gnm_comparison_grid.py first.")
#     plot_grid_and_variance(results, args.out)


# if __name__ == "__main__":
#     main()



"""
Paper figure for the wide-target parameter recovery (appendix: grid_and_variance)
=================================================================================

Renders ``figures/appendix/grid_and_variance.pdf`` of the benchmarking manuscript
from the cached per-measure distance CSVs produced by
``run_synthetic_gnm_comparison_grid.py``. Nothing is recomputed here.

  A  4 x 4 grid of recovered locations, one subpanel per measure: crosses mark
     the five true parameter combinations, dots the predicted cells of the 10
     test networks each. Thin lines connect each true combination to every one
     of its predicted points, coloured by that point's recovery error. Subpanel
     title carries the mean grid-step error.
  B  Per-combination variability: mean std of predicted eta / gamma (normalised
     to [0, 1]) across the 10 test networks, rows ordered by mean recovery error.

Panel drawing and all error / variability maths are reused from
``run_synthetic_gnm_plots_grid.py`` - this script only composes the two panels
into one figure at the manuscript's column width.

Colour scheme
-------------
Fixed four-colour palette, used instead of G's defaults for this figure only:

  DARK_GRAY  #232324   low end of the combo gradient (panel A)
  LIGHT_BLUE #3EA5C4   high end of the combo gradient / low end of the error ramp
  ORANGE     #FE961F   high end of the error ramp / high end of panel B
  BONE_WHITE #FFFCF2   low end of panel B (variability heatmap)

Panel A's five combination colours are sampled evenly along DARK_GRAY -> LIGHT_BLUE,
re-colouring whatever G._draw_scatter_ax already drew (matched to the original
G.COMBO_COLORS[i] it used). The true -> predicted connecting lines use a separate
LIGHT_BLUE -> ORANGE ramp, normalised globally across every subpanel so line colour
is comparable panel-to-panel. Panel B swaps RdYlGn_r for a BONE_WHITE -> ORANGE ramp
(still "smaller is better").

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
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

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
FIG_SIZE_CM = (16, 8) # 15.65, 8.0)
NCOLS = 4

# ---------------------------------------------------------------------------
# Fixed four-colour palette (overrides G's defaults for this figure only)
# ---------------------------------------------------------------------------
DARK_GRAY = "#232324"
ORANGE = "#FE961F"
LIGHT_BLUE = "#3EA5C4"
BONE_WHITE = "#FFFCF2"

# Panel A: one colour per true combination, sampled along this ramp.
COMBO_CMAP = mcolors.LinearSegmentedColormap.from_list("combo_grad", [DARK_GRAY, LIGHT_BLUE])
# Panel A: connecting lines, coloured by recovery error (short -> long).
ERROR_CMAP = mcolors.LinearSegmentedColormap.from_list("error_grad", [LIGHT_BLUE, ORANGE])
# Panel B: variability heatmap (small -> large, "smaller is better").
VAR_CMAP = mcolors.LinearSegmentedColormap.from_list("var_grad", [BONE_WHITE, ORANGE])

DOT_ALPHA = 0.55    # prediction dots: translucent so overlapping points stay visible
LINE_ALPHA = 0.55   # connecting lines: same idea
LINE_WIDTH = 0.35

_X_PATH = mmarkers.MarkerStyle("x").get_path()


def _combo_colors(n_combos):
    if n_combos == 1:
        return [COMBO_CMAP(0.0)]
    return [COMBO_CMAP(i / (n_combos - 1)) for i in range(n_combos)]


def _is_cross(coll):
    """Crosses are drawn with matplotlib's 'x' marker path; dots aren't."""
    paths = coll.get_paths()
    if not paths:
        return False
    verts = paths[0].vertices
    return verts.shape == _X_PATH.vertices.shape and np.allclose(verts, _X_PATH.vertices)


def _restyle_and_collect_lines(ax, n_combos):
    """Re-colour an already-drawn Panel-A subpanel onto COMBO_CMAP and return the
    (true_xy, pred_xy, error_length) segments to connect. Lines aren't drawn here:
    their colour needs a global error norm computed across every subpanel first.

    Matching assumption: G._draw_scatter_ax paints a combination's cross and its
    10 dots with the same original colour, G.COMBO_COLORS[i]. If a future version
    of G changes that, this matching needs to change with it.
    """
    new_colors = _combo_colors(n_combos)
    orig_hex = [mcolors.to_hex(c) for c in G.COMBO_COLORS[:n_combos]]

    crosses, dots = {}, {}
    for coll in ax.collections:
        fc, ec = coll.get_facecolor(), coll.get_edgecolor()
        ref = fc if len(fc) else ec
        if not len(ref):
            continue
        try:
            idx = orig_hex.index(mcolors.to_hex(ref[0]))
        except ValueError:
            continue
        new_c = new_colors[idx]
        if _is_cross(coll):
            coll.set_edgecolor(new_c)
            if len(fc):
                coll.set_facecolor(new_c)
            crosses[idx] = np.asarray(coll.get_offsets())[0]
        else:
            coll.set_facecolor(mcolors.to_rgba(new_c, DOT_ALPHA))
            coll.set_edgecolor(mcolors.to_rgba(new_c, DOT_ALPHA))
            dots.setdefault(idx, []).append(np.asarray(coll.get_offsets()))

    segments = []
    for idx, true_xy in crosses.items():
        for offs in dots.get(idx, []):
            for xy in offs:
                length = float(np.hypot(xy[0] - true_xy[0], xy[1] - true_xy[1]))
                segments.append((true_xy, xy, length))
    return segments


def _panel_letter(ax, letter, x=-0.02, y=1.04, size=11):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="bottom", ha="left",
            color=DARK_GRAY)


def plot_grid_and_variance(results, out_pdf: Path):
    measures = list(results)
    n_meas = len(measures)
    n_rows = int(np.ceil(n_meas / NCOLS))
    n_combos = len(G.TRUE_PARAM_COMBOS)
    combo_colors = _combo_colors(n_combos)

    fig = plt.figure(figsize=viz.cm_to_inch(FIG_SIZE_CM))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.55)
    gs_a = outer[0].subgridspec(n_rows, NCOLS) # , hspace=1.15, wspace=0.3)
    ax_heat = fig.add_subplot(outer[1])

    # ---- Panel A: recovered locations, one subpanel per measure -----------
    axes = np.empty((n_rows, NCOLS), dtype=object)
    axis_segments = []  # (ax, [(true_xy, pred_xy, length), ...]) per subpanel
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
        # Recolour onto the fixed palette and stash the true -> predicted
        # segments; their line colour needs a global error norm, added below.
        axis_segments.append((ax, _restyle_and_collect_lines(ax, n_combos)))
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

    # Thin lines from each true combination to every one of its predicted
    # points, coloured by recovery error on one scale shared across all
    # subpanels - drawn now, after the loop, so that scale can be built from
    # every subpanel's segments at once.
    all_lengths = [seg[2] for _, segs in axis_segments for seg in segs]
    if all_lengths:
        norm = mcolors.Normalize(vmin=min(all_lengths), vmax=max(all_lengths))
        for ax, segs in axis_segments:
            if not segs:
                continue
            lines = [(t, p) for t, p, _ in segs]
            colors = [ERROR_CMAP(norm(length)) for _, _, length in segs]
            lc = LineCollection(lines, colors=colors, linewidths=LINE_WIDTH,
                                 alpha=LINE_ALPHA, zorder=0.5)
            ax.add_collection(lc)

    axes[0, 0].set_xlim(G.GRID_ETA[0], G.GRID_ETA[-1])
    axes[0, 0].set_ylim(G.GRID_GAMMA[0], G.GRID_GAMMA[-1])
    axes[0, 0].set_xticks([-8, 0, 3])
    axes[0, 0].set_yticks([-0.1, 0, 1])

    handles = [
        Line2D([], [], linestyle="none", marker="x", markersize=4,
               markeredgewidth=0.9, color=combo_colors[i],
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

    im = ax_heat.imshow(var, aspect="auto", cmap=VAR_CMAP,
                        vmin=0, vmax=np.nanmax(var) or 1)
    ax_heat.set_xticks(range(n_combos))
    ax_heat.set_xticklabels(
        [rf"$\eta$={e:.1f}" "\n" rf"$\gamma$={g:.2f}"
         for e, g in G.TRUE_PARAM_COMBOS], fontsize=4.5)
    ax_heat.set_yticks(range(len(labels)))
    ax_heat.set_yticklabels(labels, fontsize=4.5)
    ax_heat.tick_params(length=2, pad=1)
    ax_heat.set_title("Variability (smaller is better)", fontsize=6, pad=4)

    # BONE_WHITE -> ORANGE stays light-to-midtone throughout, so dark text
    # reads cleanly across the whole range (unlike the old RdYlGn_r, which
    # needed a light/dark switch to stay legible on its darker red end).
    for r in range(var.shape[0]):
        for c in range(var.shape[1]):
            val = var[r, c]
            if not np.isfinite(val):
                continue
            ax_heat.text(c, r, f"{val:.2f}", ha="center", va="center",
                         fontsize=4, color=DARK_GRAY)

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