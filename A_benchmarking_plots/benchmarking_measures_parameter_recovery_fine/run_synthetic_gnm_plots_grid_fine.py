#######
# CURRENTLY NEED TO BE IN RIGHT FOLDER!!
# (ma_thesis) adrian@Mac benchmarking_measures_parameter_recovery_fine % python run_synthetic_gnm_plots_grid_fine.py
#######
"""
Grid-step style plots for the FINE-GRAINED parameter-recovery experiment.

This is the fine-window analogue of ../benchmarking_measures_parameter_recovery/
run_synthetic_gnm_plots_grid.py. The wide experiment places 5 discrete, widely
spread true (eta, gamma) combos; the fine experiment instead samples
N_GROUND_TRUTH = 100 *continuous* true points from a tight, human-plausible
window and recovers each by argmin against a 10x10 fine consensus grid.

Because there are no discrete combos here (and only 1 net per ground truth), the
per-combo variability heatmap from the wide script does not apply and is dropped.
Everything below is computed straight from the existing comparison CSVs - no new
(expensive) generation is required.

Reads comparison CSVs from:
  output/gnm/synthetic_parameter_recovery_fine/comparison_results/

Produces (into output/gnm/synthetic_parameter_recovery_fine/plots/):
  landscape_fine_<measure>_grid_steps.pdf   - distance landscape for a handful of
                                              representative ground-truth points
  parameter_recovery_scatter_fine_grid_steps.pdf
                                            - true -> recovered "recovery field"
                                              in eta-gamma space (all measures)
  grid_steps_bars_fine.pdf                  - mean grid-step error per measure

Usage
-----
  python run_synthetic_gnm_plots_grid_fine.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from vizman import viz

# Fine-experiment config (single source of truth for the grid).
from gnm_fine_config import (
    GRID_ETA, GRID_GAMMA, GRID_COMBOS, GRID_N_ETA, GRID_N_GAMMA,
    ETA_RANGE, GAMMA_RANGE,
    SAMPLE_ETA_RANGE, SAMPLE_GAMMA_RANGE, CENTER_ETA, CENTER_GAMMA,
)


# ---------------------------------------------------------------------------
# vizman setup
# ---------------------------------------------------------------------------

viz.set_visual_style(font_family="Arial")

_pkg    = Path(viz.__file__).parent
_colors = {
    name: val
    for cat in json.loads((_pkg / "colors.json").read_text()).values()
    for name, val in cat.items()
}
_cmaps = viz.give_colormaps()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR       = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
COMPARISON_DIR = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_fine" / "comparison_results"
PLOT_DIR       = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_fine" / "plots"

# Mean Euclidean distance in grid-index space (0 = correct cell). The same
# normalisation-free, interpretable error used by the wide experiment.
ERROR_METRIC = "grid_steps"

SCATTER_NCOLS = 4
DPI = 300

# How many representative ground-truth points to show in the landscape figure.
N_LANDSCAPE_REP = 5

# Display names for each distance measure (manuscript naming convention).
METRIC_NAMES: Dict[str, str] = {
    "delta_con":                        "DeltaCon",
    "frobenius":                        "Frobenius",
    "portrait":                         "Portrait",
    "hamming":                          "Hamming",
    "f1":                               "F1",
    "jaccard":                          "Jaccard",
    "spectral_distance_norm_laplacian": "Spectral (Norm. Laplacian)",
    "spectral_distance_adjacency":      "Spectral (Adjacency)",
    "communicability_corr":             "Communicability Corr.",
    "communicability_jsd":              "Communicability JSD",
    "network_mutual_information":       "Network MI",
    "dc_network_mutual_information":    "DC Network MI",
    "net_simile":                       "NetSimile",
    "resistance":                       "Resistance",
    "netrd_non_backtracking_spectral":  "NBS Spectral",
    "energy":                           "Energy",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_results() -> Dict[str, pd.DataFrame]:
    results = {}
    for csv_path in sorted(COMPARISON_DIR.glob("distances_*.csv")):
        measure = csv_path.stem.replace("distances_", "")
        results[measure] = pd.read_csv(csv_path)
        print(f"  Loaded {measure}: {len(results[measure])} rows")
    return results


def _dist_grid_cols() -> List[str]:
    return [f"dist_to_grid_{i}" for i in range(len(GRID_COMBOS))]


def _true_grid_indices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest grid (eta_idx, gamma_idx) for each true (eta, gamma)."""
    te = df["true_eta"].apply(lambda e: int(np.argmin(np.abs(GRID_ETA   - e)))).values
    tg = df["true_gamma"].apply(lambda g: int(np.argmin(np.abs(GRID_GAMMA - g)))).values
    return te, tg


def grid_step_errors(df: pd.DataFrame) -> np.ndarray:
    """Per-row Euclidean distance (grid cells) between recovered and true cell.

    Returns NaN for rows whose recovery failed (recovered_grid_idx < 0).
    Flat index layout is gamma-outer / eta-inner (matches GRID_COMBOS).
    """
    idx = df["recovered_grid_idx"].values
    pred_gamma_idx = np.where(idx >= 0, idx // GRID_N_ETA, np.nan)
    pred_eta_idx   = np.where(idx >= 0, idx %  GRID_N_ETA, np.nan)
    true_eta_idx, true_gamma_idx = _true_grid_indices(df)
    return np.sqrt(
        (pred_eta_idx   - true_eta_idx)   ** 2 +
        (pred_gamma_idx - true_gamma_idx) ** 2
    )


def compute_grid_step_error(df: pd.DataFrame) -> float:
    """Mean grid-step error across all ground-truth points (lower is better)."""
    return float(np.nanmean(grid_step_errors(df)))


def _error_label() -> str:
    return "Grid steps"


def _panel_label(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter,
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
            color=_colors["HALF_BLACK"])


def representative_gt(df: pd.DataFrame) -> List[Tuple[int, str]]:
    """Pick N_LANDSCAPE_REP ground-truth rows nearest to anchor locations.

    Anchors span the *sampling* window (4 corners + centre) so the landscape
    figure illustrates recovery across the plausible region. Selection is based
    on true params only, so it is identical for every measure. Returns a list of
    (row position in df, short label).
    """
    e_lo, e_hi = SAMPLE_ETA_RANGE
    g_lo, g_hi = SAMPLE_GAMMA_RANGE
    anchors = [
        ((e_lo, g_lo), r"low $\eta$, low $\gamma$"),
        ((e_lo, g_hi), r"low $\eta$, high $\gamma$"),
        ((e_hi, g_lo), r"high $\eta$, low $\gamma$"),
        ((e_hi, g_hi), r"high $\eta$, high $\gamma$"),
        ((CENTER_ETA, CENTER_GAMMA), "centre"),
    ][:N_LANDSCAPE_REP]

    e_span = ETA_RANGE[1]   - ETA_RANGE[0]
    g_span = GAMMA_RANGE[1] - GAMMA_RANGE[0]
    te = (df["true_eta"].values   - ETA_RANGE[0]) / e_span
    tg = (df["true_gamma"].values - GAMMA_RANGE[0]) / g_span

    picks, used = [], set()
    for (ae, ag), label in anchors:
        aen = (ae - ETA_RANGE[0]) / e_span
        agn = (ag - GAMMA_RANGE[0]) / g_span
        d = (te - aen) ** 2 + (tg - agn) ** 2
        order = np.argsort(d)
        for cand in order:                 # avoid picking the same row twice
            if int(cand) not in used:
                used.add(int(cand))
                picks.append((int(cand), label))
                break
    return picks


# ---------------------------------------------------------------------------
# Figure 1 - Landscape heatmaps for representative ground-truth points
# ---------------------------------------------------------------------------

def _row_landscape(df: pd.DataFrame, row_pos: int) -> np.ndarray:
    """(GRID_N_GAMMA, GRID_N_ETA) distance matrix for a single ground-truth row."""
    vals = df.iloc[row_pos][_dist_grid_cols()].values.astype(float)
    return vals.reshape(GRID_N_GAMMA, GRID_N_ETA)   # rows=gamma, cols=eta


def plot_landscapes(results: Dict[str, pd.DataFrame]) -> None:
    """One PDF per measure: distance landscape for N_LANDSCAPE_REP true points."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for measure, df in results.items():
        reps = representative_gt(df)
        n = len(reps)
        fig, axes = plt.subplots(1, n, figsize=viz.cm_to_inch((18, 4.5)),
                                 squeeze=False)

        landscapes = [_row_landscape(df, pos) for pos, _ in reps]
        finite = [m[np.isfinite(m)] for m in landscapes]
        vmin = min(f.min() for f in finite if f.size)
        vmax = max(f.max() for f in finite if f.size)

        for ci, (ax, (pos, label)) in enumerate(zip(axes[0], reps)):
            row = df.iloc[pos]
            land = landscapes[ci]
            true_eta, true_gamma = float(row["true_eta"]), float(row["true_gamma"])

            im = ax.imshow(
                land, origin="lower", aspect="auto", cmap=_cmaps["bw_lr"],
                vmin=vmin, vmax=vmax,
                extent=[GRID_ETA[0], GRID_ETA[-1], GRID_GAMMA[0], GRID_GAMMA[-1]],
            )

            # True location (X), behind the predicted marker.
            ax.scatter(true_eta, true_gamma, marker="x", s=80,
                       color=_colors["LECKER_RED"], linewidths=2.0, zorder=3)

            # Recovered = argmin of the landscape.
            min_row, min_col = np.unravel_index(np.nanargmin(land), land.shape)
            ax.scatter(GRID_ETA[min_col], GRID_GAMMA[min_row], marker="o", s=60,
                       color=_colors["DEEP_BLUE"],
                       edgecolors=_colors["HALF_BLACK"], linewidths=0.5, zorder=5)

            ax.set_xlabel(r"$\eta$")
            if ci == 0:
                ax.set_ylabel(r"$\gamma$")
            ax.set_title(f"{label}\n" rf"($\eta$={true_eta:.2f}, $\gamma$={true_gamma:.2f})",
                         fontsize=7)
            sns.despine(ax=ax)
            _panel_label(ax, chr(ord("A") + ci))

        cb = fig.colorbar(im, ax=axes[0, -1], shrink=0.85, pad=0.04)
        cb.set_label("Distance")

        handles = [
            mpatches.Patch(facecolor=_colors["DEEP_BLUE"], label="recovered (min)"),
            plt.scatter([], [], marker="x", s=60,
                        color=_colors["LECKER_RED"], linewidths=2.0, label="true"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, -0.04))

        name = METRIC_NAMES.get(measure, measure.replace("_", " ").title())
        fig.suptitle(f"{name}  -  distance landscape (representative true points)")
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        out_path = PLOT_DIR / f"landscape_fine_{measure}_grid_steps.pdf"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved landscape -> {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2 - Recovery field: true -> recovered in eta-gamma space
# ---------------------------------------------------------------------------

def _draw_recovery_ax(ax, df: pd.DataFrame, measure: str,
                      err_norm, err_cmap,
                      show_xlabel: bool, show_ylabel: bool,
                      panel_letter: str):
    rng = np.random.default_rng(42)

    # Faint grid (the candidate recovery locations).
    ax.scatter([c[0] for c in GRID_COMBOS], [c[1] for c in GRID_COMBOS],
               color=_colors["GRAY"], s=4, alpha=0.3, zorder=1, linewidths=0)

    steps = grid_step_errors(df)
    valid = df["recovered_grid_idx"].values >= 0

    # Small jitter on the (discrete) recovered points so coincident cells show.
    jit_e = rng.normal(0, (GRID_ETA[1]   - GRID_ETA[0])   * 0.12, size=len(df))
    jit_g = rng.normal(0, (GRID_GAMMA[1] - GRID_GAMMA[0]) * 0.12, size=len(df))

    true_e = df["true_eta"].values[valid]
    true_g = df["true_gamma"].values[valid]
    rec_e  = df["recovered_eta"].values[valid]   + jit_e[valid]
    rec_g  = df["recovered_gamma"].values[valid] + jit_g[valid]
    col    = err_cmap(err_norm(steps[valid]))

    # true -> recovered segments, coloured by grid-step error.
    segs = np.stack([np.column_stack([true_e, true_g]),
                     np.column_stack([rec_e,  rec_g])], axis=1)
    ax.add_collection(LineCollection(segs, colors=col, linewidths=0.6,
                                     alpha=0.7, zorder=2))
    # true points (small) and recovered points (coloured).
    ax.scatter(true_e, true_g, s=5, color=_colors["HALF_BLACK"],
               alpha=0.5, zorder=3, linewidths=0)
    ax.scatter(rec_e, rec_g, s=16, c=col, zorder=4, linewidths=0)

    err  = compute_grid_step_error(df)
    name = METRIC_NAMES.get(measure, measure.replace("_", " ").title())
    ax.set_title(f"{name}\n{_error_label()}={err:.3f}", fontsize=7)

    if show_xlabel:
        ax.set_xlabel(r"$\eta$")
    if show_ylabel:
        ax.set_ylabel(r"$\gamma$")
    _panel_label(ax, panel_letter)


def plot_scatter(results: Dict[str, pd.DataFrame]) -> None:
    n_measures = len(results)
    n_cols     = min(SCATTER_NCOLS, n_measures)
    n_rows     = math.ceil(n_measures / n_cols)

    # Shared error colour scale (0 = exact cell -> green; far -> red).
    all_steps = np.concatenate([grid_step_errors(df) for df in results.values()])
    vmax      = float(np.nanmax(all_steps)) or 1.0
    err_norm  = plt.Normalize(vmin=0, vmax=vmax)
    err_cmap  = plt.cm.get_cmap("RdYlGn_r")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=viz.cm_to_inch((18, 5 * n_rows)),
        sharex=True, sharey=True, squeeze=False,
    )
    axes[0, 0].set_xlim(GRID_ETA[0],   GRID_ETA[-1])
    axes[0, 0].set_ylim(GRID_GAMMA[0], GRID_GAMMA[-1])

    letters = [chr(ord("A") + i) for i in range(n_measures)]
    for ax_idx, (measure, df) in enumerate(results.items()):
        row, col = divmod(ax_idx, n_cols)
        _draw_recovery_ax(
            axes[row, col], df, measure, err_norm, err_cmap,
            show_xlabel=(row == n_rows - 1), show_ylabel=(col == 0),
            panel_letter=letters[ax_idx],
        )

    for empty in range(n_measures, n_rows * n_cols):
        row, col = divmod(empty, n_cols)
        axes[row, col].set_visible(False)

    sm = plt.cm.ScalarMappable(norm=err_norm, cmap=err_cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Grid-step error", fontsize=8)

    fig.suptitle(
        r"Fine-window recovery field - true points $\to$ recovered grid cell "
        r"(coloured by grid-step error)")
    out_path = PLOT_DIR / "parameter_recovery_scatter_fine_grid_steps.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter -> {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3 - Grid-step error per measure (bar chart)
# ---------------------------------------------------------------------------

def plot_grid_steps_bar(results: Dict[str, pd.DataFrame]) -> None:
    measures   = list(results.keys())
    n_measures = len(measures)

    err_values = np.array([compute_grid_step_error(results[m]) for m in measures])
    order         = np.argsort(err_values)
    sorted_err    = err_values[order]
    sorted_labels = [METRIC_NAMES.get(measures[i], measures[i]) for i in order]

    fig, ax = plt.subplots(
        figsize=viz.cm_to_inch((12, max(6, n_measures * 0.55 + 2))))

    bar_cmap   = plt.cm.get_cmap("RdYlGn_r", n_measures)
    bar_colors = [bar_cmap(i / max(n_measures - 1, 1)) for i in range(n_measures)]

    bars = ax.barh(range(n_measures), sorted_err, color=bar_colors,
                   edgecolor=_colors["HALF_BLACK"], linewidth=0.3, height=0.6)
    ax.set_yticks(range(n_measures))
    ax.set_yticklabels(sorted_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"{_error_label()}  (mean over {len(next(iter(results.values())))} "
                  "ground-truth points)", fontsize=8)
    ax.set_xlim(0, sorted_err.max() * 1.22 + 1e-8)
    ax.axvline(0, color=_colors["HALF_BLACK"], linewidth=0.6)
    for bar, val in zip(bars, sorted_err):
        ax.text(val + sorted_err.max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=6,
                color=_colors["HALF_BLACK"])
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)

    fig.suptitle(
        rf"Fine-window recovery accuracy - "
        rf"$\eta\in[{SAMPLE_ETA_RANGE[0]},{SAMPLE_ETA_RANGE[1]}]$, "
        rf"$\gamma\in[{SAMPLE_GAMMA_RANGE[0]},{SAMPLE_GAMMA_RANGE[1]}]$ "
        "(lower is better)", fontsize=8, y=1.01)
    fig.tight_layout()

    out_path = PLOT_DIR / "grid_steps_bars_fine.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"{out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results ...")
    results = load_all_results()
    if not results:
        print(f"No CSVs found in {COMPARISON_DIR}\n"
              "Run run_synthetic_gnm_fine.py --stage compare first.")
        return

    print(f"\nFound {len(results)} measure(s): {list(results.keys())}")
    print(f"\n--- {_error_label()} summary ---")
    for measure, df in sorted(results.items(),
                              key=lambda kv: compute_grid_step_error(kv[1])):
        print(f"  {measure:35s}  grid_steps={compute_grid_step_error(df):.4f}")

    print("\nPlotting landscapes ...")
    plot_landscapes(results)

    print("\nPlotting recovery field ...")
    plot_scatter(results)

    print("\nPlotting grid-step bars ...")
    plot_grid_steps_bar(results)

    print(f"\nAll plots saved -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
