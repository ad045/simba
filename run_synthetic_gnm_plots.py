"""
Plots for the synthetic GNM parameter-recovery experiment.

Reads comparison CSVs from:
  output/gnm/synthetic_parameter_recovery/comparison _results/distances_<measure>.csv

Produces two figures saved to:
  output/gnm/synthetic_parameter_recovery/plots/
    parameter_recovery_scatter.pdf  — eta-gamma space: true vs predicted locations
    precision_accuracy.pdf          — per-measure accuracy + per-class precision

Usage
-----
  python run_synthetic_gnm_plots.py

Adding a new distance measure
------------------------------
  Add its CSV to the comparison results directory (via run_synthetic_gnm_comparison.py)
  and it will be picked up automatically.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
COMPARISON_DIR = (
    ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery" / "comparison_results"
)
PLOT_DIR = (
    ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery" / "plots"
)

PARAM_COMBOS: List[Tuple[float, float]] = [
    (-6.9,    1.11),
    ( 4.1,    1.11),
    (-6.9,    0.01),
    ( 4.1,    0.01),
    (-3.734694004058838, 0.595918357372283),
]

# Display labels for each parameter combination
COMBO_LABELS = [
    r"$\eta{=}{-}6.9,\ \gamma{=}1.11$",
    r"$\eta{=}4.1,\ \gamma{=}1.11$",
    r"$\eta{=}{-}6.9,\ \gamma{=}0.01$",
    r"$\eta{=}4.1,\ \gamma{=}0.01$",
    r"$\eta{=}{-}3.73,\ \gamma{=}0.60$",
]

# One colour per true parameter combination
COMBO_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]

SCATTER_NCOLS = 4   # columns in the scatter grid; rows are computed dynamically
FIGSIZE_CELL  = (5, 4.5)   # (w, h) per scatter subplot cell
FIGSIZE_PREC  = (16, 9)
DPI = 150

# Parameter space bounds used when generating the GNM sweep.
# Used to normalise (eta, gamma) → [0,1] × [0,1] for distance computations.
ETA_RANGE   = (-8.0,  3.0)
GAMMA_RANGE = (-0.1,  1.0)

# Error metric for the accuracy panel: "mse" or "mae"
ERROR_METRIC = "mae" 


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_results() -> Dict[str, pd.DataFrame]:
    """Return {measure_name: DataFrame} for every CSV in COMPARISON_DIR."""
    results = {}
    for csv_path in sorted(COMPARISON_DIR.glob("distances_*.csv")):
        measure = csv_path.stem.replace("distances_", "")
        results[measure] = pd.read_csv(csv_path)
        print(f"  Loaded {measure}: {len(results[measure])} rows")
    return results


def _add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append normalised parameter columns to df (does not modify in place).
    Normalisation maps ETA_RANGE → [0,1] and GAMMA_RANGE → [0,1].
    """
    df = df.copy()
    eta_span   = ETA_RANGE[1]   - ETA_RANGE[0]
    gamma_span = GAMMA_RANGE[1] - GAMMA_RANGE[0]

    df["true_eta_n"]  = (df["true_eta"]       - ETA_RANGE[0])   / eta_span
    df["true_gamma_n"] = (df["true_gamma"]    - GAMMA_RANGE[0]) / gamma_span
    df["pred_eta_n"]  = (df["predicted_eta"]  - ETA_RANGE[0])   / eta_span
    df["pred_gamma_n"] = (df["predicted_gamma"] - GAMMA_RANGE[0]) / gamma_span

    d_eta   = df["pred_eta_n"]   - df["true_eta_n"]
    d_gamma = df["pred_gamma_n"] - df["true_gamma_n"]
    df["sq_error"]  = d_eta ** 2 + d_gamma ** 2
    df["abs_error"] = np.sqrt(df["sq_error"])   # Euclidean distance (= sqrt of sq_error)
    return df


def compute_accuracy_error(df: pd.DataFrame) -> float:
    """
    Return MSE or MAE in normalised parameter space depending on ERROR_METRIC.
    Lower = better accuracy.
    """
    df = _add_normalized_columns(df)
    if ERROR_METRIC == "mae":
        return df["abs_error"].mean()
    return df["sq_error"].mean()


# Keep compute_mse as a direct alias for backward compatibility
def compute_mse(df: pd.DataFrame) -> float:
    return _add_normalized_columns(df)["sq_error"].mean()


def compute_variability(df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Per-combo variability of predicted locations in normalised parameter space.

    For each true parameter combination compute the mean of
    std(pred_eta_n) and std(pred_gamma_n) across its N_TEST networks.
    Returns (per_combo array of length n_combos, overall mean variability).
    Lower = more precise (consistent) predictions.
    """
    df = _add_normalized_columns(df)
    per_combo = []
    for combo_idx in range(len(PARAM_COMBOS)):
        sub = df[df["true_combo_idx"] == combo_idx]
        std_eta   = sub["pred_eta_n"].std(ddof=0)
        std_gamma = sub["pred_gamma_n"].std(ddof=0)
        per_combo.append((std_eta + std_gamma) / 2.0)
    return np.array(per_combo), float(np.mean(per_combo))


# ---------------------------------------------------------------------------
# Figure 1 — Scatter grid: true vs predicted in eta-gamma space
# ---------------------------------------------------------------------------

def _draw_scatter_ax(ax, df: pd.DataFrame, measure: str, show_ylabel: bool) -> None:
    """Populate a single scatter subplot for one distance measure."""
    eta_vals   = [c[0] for c in PARAM_COMBOS]
    gamma_vals = [c[1] for c in PARAM_COMBOS]
    rng = np.random.default_rng(42)

    for true_idx, (true_eta, true_gamma) in enumerate(PARAM_COMBOS):
        color = COMBO_COLORS[true_idx]
        sub   = df[df["true_combo_idx"] == true_idx]

        jitter      = rng.normal(0, 0.06, size=(len(sub), 2))
        pred_etas   = sub["predicted_eta"].values   + jitter[:, 0]
        pred_gammas = sub["predicted_gamma"].values + jitter[:, 1]

        ax.scatter(pred_etas, pred_gammas,
                   c=color, alpha=0.55, s=35, zorder=3)

        mean_pred_eta   = sub["predicted_eta"].mean()
        mean_pred_gamma = sub["predicted_gamma"].mean()
        ax.annotate("",
                    xy=(mean_pred_eta, mean_pred_gamma),
                    xytext=(true_eta, true_gamma),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=1.1, alpha=0.75),
                    zorder=4)

    # True locations (stars)
    for true_idx, (te, tg) in enumerate(PARAM_COMBOS):
        ax.scatter(te, tg, marker="*", s=220,
                   c=COMBO_COLORS[true_idx],
                   edgecolors="black", linewidths=0.5, zorder=5)

    err = compute_accuracy_error(df)
    error_label = "MAE" if ERROR_METRIC == "mae" else "MSE"
    ax.set_title(f"{measure.replace('_', ' ').title()}\n{error_label}={err:.3f}", fontsize=9)
    ax.set_xlabel(r"$\eta$", fontsize=9)
    if show_ylabel:
        ax.set_ylabel(r"$\gamma$", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.35)

    eta_span   = max(eta_vals)   - min(eta_vals)
    gamma_span = max(gamma_vals) - min(gamma_vals)
    ax.set_xlim(min(eta_vals)   - 0.15 * eta_span,
                max(eta_vals)   + 0.15 * eta_span)
    ax.set_ylim(min(gamma_vals) - 0.25 * gamma_span,
                max(gamma_vals) + 0.25 * gamma_span)


def plot_scatter(results: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Grid of scatter subplots (SCATTER_NCOLS columns), one per distance measure.
    True locations: ★ coloured by combo.
    Predicted locations: dots with jitter, same colour, arrow from true → mean predicted.
    """
    n_measures = len(results)
    n_cols     = min(SCATTER_NCOLS, n_measures)
    n_rows     = math.ceil(n_measures / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIGSIZE_CELL[0] * n_cols, FIGSIZE_CELL[1] * n_rows),
        squeeze=False,
    )

    for ax_idx, (measure, df) in enumerate(results.items()):
        row, col = divmod(ax_idx, n_cols)
        _draw_scatter_ax(axes[row, col], df, measure, show_ylabel=(col == 0))

    # Hide unused cells in the last row
    for empty_idx in range(n_measures, n_rows * n_cols):
        row, col = divmod(empty_idx, n_cols)
        axes[row, col].set_visible(False)

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=COMBO_COLORS[i], label=COMBO_LABELS[i])
        for i in range(len(PARAM_COMBOS))
    ]
    fig.legend(
        handles=legend_patches,
        title="True parameter combo",
        loc="lower center",
        ncol=min(len(PARAM_COMBOS), 3),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        r"Parameter recovery: true ($\bigstar$) vs predicted ($\bullet$) — "
        "all distance measures",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Accuracy (MSE) and precision (variability) across all measures
# ---------------------------------------------------------------------------

def plot_precision_accuracy(results: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Two panels, both in normalised parameter space ([0,1]×[0,1]).

    Left  — Accuracy: horizontal bar chart of MSE per measure, sorted best→worst.
            Lower = better.

    Right — Precision: heatmap (measures × parameter combos) of per-combo
            variability (mean std of predicted location).
            Lower = more consistent predictions.
    """
    measures   = list(results.keys())
    n_measures = len(measures)
    n_combos   = len(PARAM_COMBOS)

    # Pre-compute metrics
    mse_values  = np.array([compute_accuracy_error(results[m]) for m in measures])
    error_label = "MAE" if ERROR_METRIC == "mae" else "MSE"
    var_matrix = np.zeros((n_measures, n_combos))   # rows=measures, cols=combos
    for m_idx, measure in enumerate(measures):
        per_combo, _ = compute_variability(results[measure])
        var_matrix[m_idx] = per_combo

    # Sort measures by MSE (best first) for the bar chart
    sort_order   = np.argsort(mse_values)
    sorted_mse   = mse_values[sort_order]
    sorted_labels = [measures[i].replace("_", " ") for i in sort_order]
    sorted_var    = var_matrix[sort_order]        # keep same order for heatmap

    fig = plt.figure(figsize=FIGSIZE_PREC)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.35)
    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # ---- Panel 1: horizontal bar chart (MSE) ------------------------------
    cmap_bar  = plt.cm.get_cmap("RdYlGn_r", n_measures)
    bar_colors = [cmap_bar(i / n_measures) for i in range(n_measures)]

    bars = ax_bar.barh(
        range(n_measures), sorted_mse,
        color=bar_colors,
        edgecolor="black", linewidth=0.4, height=0.6,
    )
    ax_bar.set_yticks(range(n_measures))
    ax_bar.set_yticklabels(sorted_labels, fontsize=9)
    ax_bar.invert_yaxis()          # best measure at top
    ax_bar.set_xlabel(f"{error_label}  (normalised parameter space)", fontsize=10)
    ax_bar.set_title(f"Accuracy — {error_label}  (↓ better)", fontsize=11)
    ax_bar.axvline(0, color="black", linewidth=0.5)
    for bar, val in zip(bars, sorted_mse):
        ax_bar.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
    ax_bar.set_xlim(0, sorted_mse.max() * 1.2 + 1e-6)

    # ---- Panel 2: heatmap (variability) -----------------------------------
    combo_short = [f"η={e:.1f}\nγ={g:.2f}" for e, g in PARAM_COMBOS]

    im = ax_heat.imshow(sorted_var, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=sorted_var.max())
    ax_heat.set_xticks(range(n_combos))
    ax_heat.set_xticklabels(combo_short, fontsize=8)
    ax_heat.set_yticks(range(n_measures))
    ax_heat.set_yticklabels(sorted_labels, fontsize=9)
    ax_heat.set_title("Precision per combo  (↓ better)", fontsize=11)

    # Annotate each cell
    for row in range(n_measures):
        for col in range(n_combos):
            val = sorted_var[row, col]
            txt_color = "white" if val > sorted_var.max() * 0.6 else "black"
            ax_heat.text(col, row, f"{val:.2f}",
                         ha="center", va="center",
                         fontsize=7, color=txt_color)

    cb = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cb.set_label("Mean std (norm. space)", fontsize=8)

    fig.suptitle(
        r"Parameter recovery quality — normalised $\eta{\in}[-8,3]$,"
        r" $\gamma{\in}[-0.1,1]$",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy/precision plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading comparison results …")
    results = load_all_results()

    if not results:
        print(
            "No comparison CSVs found in:\n  "
            f"{COMPARISON_DIR}\n"
            "Run run_synthetic_gnm_comparison.py first."
        )
        return

    print(f"\nFound {len(results)} measure(s): {list(results.keys())}")

    # Print quick summary table
    print("\n--- Accuracy / Precision summary (normalised parameter space) ---")
    for measure, df in results.items():
        mse              = compute_mse(df)
        _, mean_variab   = compute_variability(df)
        print(f"  {measure:25s}  MSE={mse:.4f}   variability={mean_variab:.4f}")

    plot_scatter(
        results,
        PLOT_DIR / "parameter_recovery_scatter.pdf",
    )
    plot_precision_accuracy(
        results,
        PLOT_DIR / "precision_accuracy.pdf",
    )

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
