#######
# CURRENTLY NEED TO BE IN RIGHT FOLDER!! 
# (ma_thesis) adrian@Mac benchmarking_measures_parameter_recovery % python run_synthetic_gnm_plots_grid.py
#######
"""
Plots for the continuous-landscape parameter-recovery experiment.

Reads comparison CSVs from:
  output/gnm/synthetic_parameter_recovery_grid/comparison_results/

Produces three figures per distance measure + two summary figures:
  plots/
    landscape_<measure>.pdf      - 10×10 distance heatmap, one panel per true combo
    parameter_recovery_scatter.pdf  - predicted vs true in η-γ space (all measures)
    precision_accuracy.pdf          - MAE bar chart + variability heatmap

Usage
-----
  python run_synthetic_gnm_plots_grid.py

Adding a new distance measure: just add its CSV via run_synthetic_gnm_comparison_grid.py.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from vizman import viz


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

# Coarse parameter-recovery config (single source of truth, at the repo root).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_config import coarse as _cfg
from experiments_config import to_distance

# Repo root, so this file works from any clone. The `data/` and `output/`
# symlinks at the root point at the run tree that used to be hardcoded here.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
TRUE_PARAM_COMBOS, COMBO_LABELS = _cfg.TRUE_PARAM_COMBOS, _cfg.COMBO_LABELS
GRID_ETA, GRID_GAMMA, GRID_COMBOS = _cfg.GRID_ETA, _cfg.GRID_GAMMA, _cfg.GRID_COMBOS
GRID_N_ETA, GRID_N_GAMMA = _cfg.GRID_N_ETA, _cfg.GRID_N_GAMMA
ETA_RANGE, GAMMA_RANGE = _cfg.ETA_RANGE, _cfg.GAMMA_RANGE

ROOT_DIR       = Path(f"{_REPO_ROOT}") # Path(__file__).parent
COMPARISON_DIR = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_grid" / "comparison_results"
PLOT_DIR       = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_grid" / "plots"

# One colour per true combo - extend this list if you add more combos
_PALETTE = [
    _colors["LECKER_RED"],
    _colors["LAKE_BLUE"],
    _colors["JUST_GREEN"],
    _colors["ORANGE"],
    _colors["PURPLE"],
    _colors["NIGHT_BLUE"],   # 6th combo: dark navy, so it stays apart from LAKE_BLUE
    _colors["TEAL"],
    _colors["YELLOW"],
]
COMBO_COLORS = _PALETTE[: len(TRUE_PARAM_COMBOS)]

# Error metric - three options:
#   "mae"        - mean absolute error (Euclidean distance between true and predicted)
#   "mse"        - mean squared error
#   "grid_steps" - mean Euclidean distance in grid-index space (0 = correct cell,
#                  max ≈ 12.7 for a 10×10 grid). Normalisation-free and most interpretable
#                  given that predictions are discrete grid points.
ERROR_METRIC = "grid_steps" # mae"

# Space for mae/mse (ignored when ERROR_METRIC == "grid_steps"):
#   "normalized" - η and γ each mapped to [0, 1] before computing error
#   "raw"        - original parameter units (η range 11, γ range 1.1 - η dominates)
ERROR_SPACE = "normalized"

SCATTER_NCOLS = 4
DPI = 300

# Display names for each distance measure (from manuscript naming convention)
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


def normalise(eta: float, gamma: float) -> Tuple[float, float]:
    return (
        (eta   - ETA_RANGE[0])   / (ETA_RANGE[1]   - ETA_RANGE[0]),
        (gamma - GAMMA_RANGE[0]) / (GAMMA_RANGE[1] - GAMMA_RANGE[0]),
    )


def compute_grid_step_error(df: pd.DataFrame) -> float:
    """
    Mean Euclidean distance in grid-index space.
    E.g. 2.0 means 'on average 2 grid cells away from the true location'.
    Max possible: sqrt((GRID_N_ETA-1)^2 + (GRID_N_GAMMA-1)^2) ≈ 12.7 for 10×10.
    """
    valid = df[df["predicted_grid_idx"] >= 0].copy()

    # Predicted indices (flat: gamma-outer, eta-inner)
    pred_gamma_idx = valid["predicted_grid_idx"] // GRID_N_ETA
    pred_eta_idx   = valid["predicted_grid_idx"] %  GRID_N_ETA

    # True nearest grid indices
    true_eta_idx   = valid["true_eta"].apply(
        lambda e: int(np.argmin(np.abs(GRID_ETA - e)))
    )
    true_gamma_idx = valid["true_gamma"].apply(
        lambda g: int(np.argmin(np.abs(GRID_GAMMA - g)))
    )

    step_dist = np.sqrt(
        (pred_eta_idx.values   - true_eta_idx.values)   ** 2 +
        (pred_gamma_idx.values - true_gamma_idx.values) ** 2
    )
    return float(np.nanmean(step_dist))


def compute_accuracy_error(df: pd.DataFrame) -> float:
    """MAE, MSE, or grid-step distance depending on ERROR_METRIC / ERROR_SPACE."""
    if ERROR_METRIC == "grid_steps":
        return compute_grid_step_error(df)

    if ERROR_SPACE == "normalized":
        # Pre-computed normalised columns in the CSV
        col = "abs_error" if ERROR_METRIC == "mae" else "sq_error"
        return float(df[col].mean())

    # Raw parameter units
    d_eta   = df["predicted_eta"]   - df["true_eta"]
    d_gamma = df["predicted_gamma"] - df["true_gamma"]
    if ERROR_METRIC == "mae":
        return float(np.sqrt(d_eta**2 + d_gamma**2).mean())
    else:  # mse
        return float((d_eta**2 + d_gamma**2).mean())


def _error_label() -> str:
    """Short display label for the current error metric."""
    if ERROR_METRIC == "grid_steps":
        return "Grid steps"
    space = "norm." if ERROR_SPACE == "normalized" else "raw"
    return f"{'MAE' if ERROR_METRIC == 'mae' else 'MSE'} ({space})"


def compute_variability(df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Per-true-combo std of predicted locations in normalised space."""
    eta_span   = ETA_RANGE[1]   - ETA_RANGE[0]
    gamma_span = GAMMA_RANGE[1] - GAMMA_RANGE[0]
    df = df.copy()
    df["pred_eta_n"]   = (df["predicted_eta"]   - ETA_RANGE[0]) / eta_span
    df["pred_gamma_n"] = (df["predicted_gamma"] - GAMMA_RANGE[0]) / gamma_span

    per_combo = []
    for combo_idx in range(len(TRUE_PARAM_COMBOS)):
        sub = df[df["true_combo_idx"] == combo_idx].dropna(subset=["pred_eta_n", "pred_gamma_n"])
        if len(sub) < 2:
            per_combo.append(float("nan"))
        else:
            per_combo.append(
                (sub["pred_eta_n"].std(ddof=0) + sub["pred_gamma_n"].std(ddof=0)) / 2.0
            )
    return np.array(per_combo), float(np.nanmean(per_combo))


def _dist_grid_cols(df: pd.DataFrame) -> List[str]:
    return [f"dist_to_grid_{i}" for i in range(len(GRID_COMBOS))]


def _panel_label(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter,
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
            color=_colors["HALF_BLACK"])


# ---------------------------------------------------------------------------
# Figure 1 - Landscape heatmaps (one PDF per measure)
# ---------------------------------------------------------------------------

def _mean_landscape(df: pd.DataFrame, true_combo_idx: int,
                    measure: str) -> np.ndarray:
    """(GRID_N_GAMMA, GRID_N_ETA) matrix of mean ORIENTED distances for one combo.

    Oriented: the five similarity measures (Jaccard, F1, NMI, DC-NMI,
    communicability correlation) score higher = closer, so they are flipped and
    "lower = closer" holds for all 16 - which is what the argmin marker assumes.
    """
    sub  = df[df["true_combo_idx"] == true_combo_idx]
    cols = _dist_grid_cols(sub)
    mean = to_distance(sub[cols].mean(axis=0).values, measure)   # (n_grid,)
    return mean.reshape(GRID_N_GAMMA, GRID_N_ETA)  # rows=γ, cols=η


def plot_landscapes(results: Dict[str, pd.DataFrame]) -> None:
    """One PDF per measure: 5 subplots showing the mean distance landscape."""
    n_combos = len(TRUE_PARAM_COMBOS)

    for measure, df in results.items():
        fig, axes = plt.subplots(
            1, n_combos,
            figsize=viz.cm_to_inch((18, 4.5)),
            squeeze=False,
        )

        # Determine shared colour scale across all combos for this measure
        all_vals = [_mean_landscape(df, ci, measure) for ci in range(n_combos)]
        vmin = min(m.min() for m in all_vals)
        vmax = max(m.max() for m in all_vals)

        for ci, (ax, landscape) in enumerate(zip(axes[0], all_vals)):
            true_eta, true_gamma = TRUE_PARAM_COMBOS[ci]

            im = ax.imshow(
                landscape,
                origin="lower",           # γ increases upward
                aspect="auto",
                cmap=_cmaps["bw_lr"],
                vmin=vmin, vmax=vmax,
                extent=[GRID_ETA[0], GRID_ETA[-1], GRID_GAMMA[0], GRID_GAMMA[-1]],
            )

            # True location - bold X, drawn first (behind everything)
            ax.scatter(true_eta, true_gamma,
                       marker="x", s=80, color=COMBO_COLORS[ci],
                       linewidths=2.0, zorder=3)

            # Minimum (predicted) - drawn on top
            min_flat  = np.argmin(landscape)
            min_row, min_col = np.unravel_index(min_flat, landscape.shape)
            ax.scatter(GRID_ETA[min_col], GRID_GAMMA[min_row],
                       marker="o", s=60, color=_colors["DEEP_BLUE"],
                       edgecolors=_colors["HALF_BLACK"], linewidths=0.5, zorder=5)

            ax.set_xlabel(r"$\eta$") # , fontsize=8)
            if ci == 0:
                ax.set_ylabel(r"$\gamma$") # , fontsize=8)
            ax.set_title(COMBO_LABELS[ci]) # , fontsize=7, pad=3)
            # ax.tick_params(labelsize=6)
            sns.despine(ax=ax)
            _panel_label(ax, chr(ord("A") + ci))

        # Shared colourbar
        cb = fig.colorbar(im, ax=axes[0, -1], shrink=0.85, pad=0.04)
        cb.set_label("Mean distance") # , fontsize=7)
        # cb.ax.tick_params(labelsize=6)

        # Shared legend
        handles = [
            mpatches.Patch(facecolor=_colors["DEEP_BLUE"], label="predicted (min)"),
            plt.scatter([], [], marker="x", s=60,
                        color=_colors["GRAY"], linewidths=2.0, label="true"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                #    fontsize=7, 
                   frameon=False, bbox_to_anchor=(0.5, -0.04))

        name = METRIC_NAMES.get(measure, measure.replace("_", " ").title())
        fig.suptitle(f"{name}  -  distance landscape") # , fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        out_path = PLOT_DIR / f"landscape_{measure}_{ERROR_METRIC}_{ERROR_SPACE if ERROR_METRIC != "grid_steps" else ""}.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved landscape → {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2 - Scatter: true vs predicted in η-γ space
# ---------------------------------------------------------------------------

def _draw_scatter_ax(ax, df: pd.DataFrame, measure: str,
                     show_xlabel: bool, show_ylabel: bool,
                     panel_letter: str) -> None:
    rng = np.random.default_rng(42)

    # Grid points (faint background)
    ax.scatter([c[0] for c in GRID_COMBOS], [c[1] for c in GRID_COMBOS],
               color=_colors["GRAY"], s=4, alpha=0.3, zorder=1, linewidths=0)

    # True locations - bold X, drawn before predicted dots
    for ci, (te, tg) in enumerate(TRUE_PARAM_COMBOS):
        ax.scatter(te, tg, marker="x", s=80,
                   color=COMBO_COLORS[ci], linewidths=2.0, zorder=2)

    # Predicted locations (jittered dots) - drawn on top of X
    for true_idx, (true_eta, true_gamma) in enumerate(TRUE_PARAM_COMBOS):
        color = COMBO_COLORS[true_idx]
        sub   = df[df["true_combo_idx"] == true_idx]
        # jitter      = rng.normal(0, 0.04, size=(len(sub), 2))
        pred_etas   = sub["predicted_eta"].values #   + jitter[:, 0]
        pred_gammas = sub["predicted_gamma"].values # + jitter[:, 1]
        ax.scatter(pred_etas, pred_gammas,
                   color=color, alpha=0.5, s=18, zorder=3, linewidths=0)

    err  = compute_accuracy_error(df)
    name = METRIC_NAMES.get(measure, measure.replace("_", " ").title())
    ax.set_title(f"{name}\n{_error_label()}={err:.3f}") # , fontsize=7, pad=3)

    if show_xlabel:
        ax.set_xlabel(r"$\eta$") # , fontsize=8)
    if show_ylabel:
        ax.set_ylabel(r"$\gamma$") # , fontsize=8)

    ax.set_xticks([-8, 0, 3])
    ax.set_yticks([-0.1, 0, 1])
    # ax.tick_params() # labelsize=6)
    # sns.despine(ax=ax)
    _panel_label(ax, panel_letter)


def plot_scatter(results: Dict[str, pd.DataFrame]) -> None:
    n_measures = len(results)
    n_cols     = min(SCATTER_NCOLS, n_measures)
    n_rows     = math.ceil(n_measures / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=viz.cm_to_inch((18, 5 * n_rows)),
        sharex=True, sharey=True,
        squeeze=False,
    )

    # Shared limits and ticks (set once; propagated via sharex/sharey)
    # axes[0, 0].set_xlim(GRID_ETA[0] - 0.3,   GRID_ETA[-1]   + 0.3)
    # axes[0, 0].set_ylim(GRID_GAMMA[0] - 0.05, GRID_GAMMA[-1] + 0.05)
    axes[0, 0].set_xlim(GRID_ETA[0],   GRID_ETA[-1])
    axes[0, 0].set_ylim(GRID_GAMMA[0], GRID_GAMMA[-1])
    axes[0, 0].set_xticks([-8, 0, 3])
    axes[0, 0].set_yticks([-0.1, 0, 1])

    letters = [chr(ord("A") + i) for i in range(n_measures)]

    for ax_idx, (measure, df) in enumerate(results.items()):
        row, col = divmod(ax_idx, n_cols)
        _draw_scatter_ax(
            axes[row, col], df, measure,
            show_xlabel=(row == n_rows - 1),
            show_ylabel=(col == 0),
            panel_letter=letters[ax_idx],
        )

    for empty in range(n_measures, n_rows * n_cols):
        row, col = divmod(empty, n_cols)
        axes[row, col].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=COMBO_COLORS[i], label=COMBO_LABELS[i])
        for i in range(len(TRUE_PARAM_COMBOS))
    ]
    fig.legend(handles=legend_patches, title="True parameter combination",
               loc="lower center", ncol=len(TRUE_PARAM_COMBOS),
            #    fontsize=7, title_fontsize=7, 
               frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        r"Parameter recovery on continuous grid - true ($\times$) vs predicted ($\bullet$)",
        # fontsize=9
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = PLOT_DIR / f"parameter_recovery_scatter_{ERROR_METRIC}_{ERROR_SPACE if ERROR_METRIC != "grid_steps" else ""}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 - Accuracy (MAE) + Precision (variability)
# ---------------------------------------------------------------------------

def plot_precision_accuracy(results: Dict[str, pd.DataFrame]) -> None:
    measures   = list(results.keys())
    n_measures = len(measures)
    n_combos   = len(TRUE_PARAM_COMBOS)

    err_values = np.array([compute_accuracy_error(results[m]) for m in measures])
    var_matrix = np.zeros((n_measures, n_combos))
    for m_idx, measure in enumerate(measures):
        per_combo, _ = compute_variability(results[measure])
        var_matrix[m_idx] = per_combo

    order         = np.argsort(err_values)
    sorted_err    = err_values[order]
    sorted_labels = [METRIC_NAMES[measures[i]] for i in order]
    sorted_var    = var_matrix[order]
    err_label     = _error_label()

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2,
        figsize=viz.cm_to_inch((18, max(6, n_measures * 0.55 + 2))),
        # gridspec_kw={"width_ratios": [1.5, 1]},
    )

    # ---- Panel A: horizontal bar chart ------------------------------------
    bar_cmap   = plt.cm.get_cmap("RdYlGn_r", n_measures)
    bar_colors = [bar_cmap(i / max(n_measures - 1, 1)) for i in range(n_measures)]

    bars = ax_bar.barh(range(n_measures), sorted_err,
                       color=bar_colors,
                       edgecolor=_colors["HALF_BLACK"], linewidth=0.3,
                       height=0.6)
    ax_bar.set_yticks(range(n_measures))
    ax_bar.set_yticklabels(sorted_labels, fontsize=7)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(f"{err_label}  (normalised space)", fontsize=8)
    ax_bar.set_xlim(0, sorted_err.max() * 1.22 + 1e-8)
    ax_bar.axvline(0, color=_colors["HALF_BLACK"], linewidth=0.6)
    for bar, val in zip(bars, sorted_err):
        ax_bar.text(val + sorted_err.max() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=6,
                    color=_colors["HALF_BLACK"])
    sns.despine(ax=ax_bar, left=True)
    ax_bar.tick_params(left=False)
    _panel_label(ax_bar, "A")

    # ---- Panel B: heatmap -------------------------------------------------
    combo_short = [f"η={e:.1f}\nγ={g:.2f}" for e, g in TRUE_PARAM_COMBOS]
    im = ax_heat.imshow(sorted_var, aspect="auto",
                        cmap="RdYlGn_r", # _cmaps["bw_lr"],
                        vmin=0, vmax=sorted_var.max() or 1)
    ax_heat.set_xticks(range(n_combos))
    ax_heat.set_xticklabels(combo_short, fontsize=6)
    ax_heat.set_yticks(range(n_measures))
    ax_heat.set_yticklabels(sorted_labels, fontsize=7)
    ax_heat.set_title("Variability (smaller is better)", fontsize=8, pad=4)
    threshold = sorted_var.max() * 0.55 if sorted_var.max() > 0 else 1
    for row in range(n_measures):
        for col in range(n_combos):
            val = sorted_var[row, col]
            txt = _colors["BONE_WHITE"] if val > threshold else _colors["HALF_BLACK"]
            ax_heat.text(col, row, f"{val:.2f}",
                         ha="center", va="center", fontsize=5.5, color=txt)
    cb = fig.colorbar(im, ax=ax_heat, shrink=0.7, pad=0.03)
    cb.set_label("Mean std (norm. space)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    _panel_label(ax_heat, "B")

    fig.suptitle(
        rf"Recovery quality on continuous grid - "
        rf"$\eta\in[{ETA_RANGE[0]},{ETA_RANGE[1]}]$, "
        rf"$\gamma\in[{GAMMA_RANGE[0]},{GAMMA_RANGE[1]}]$",
        fontsize=8, y=1.01)
    fig.tight_layout()

    out_path = PLOT_DIR / f"precision_accuracy_{ERROR_SPACE if ERROR_METRIC != "grid_steps" else ""}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy/precision → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results …")
    results = load_all_results()
    if not results:
        print(f"No CSVs found in {COMPARISON_DIR}\n"
              "Run run_synthetic_gnm_comparison_grid.py first.")
        return

    print(f"\nFound {len(results)} measure(s): {list(results.keys())}")
    lbl = _error_label()
    print(f"\n--- {lbl} / variability summary ---")
    for measure, df in results.items():
        err          = compute_accuracy_error(df)
        _, mean_var  = compute_variability(df)
        print(f"  {measure:35s}  {lbl}={err:.4f}  variability={mean_var:.4f}")

    # print("\nPlotting landscapes …")
    # plot_landscapes(results)

    print("\nPlotting scatter …")
    plot_scatter(results)

    print("\nPlotting accuracy/precision …")
    plot_precision_accuracy(results)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
