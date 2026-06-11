"""
Plots for the synthetic GNM parameter-recovery experiment.

Reads comparison CSVs from:
  output/gnm/synthetic_parameter_recovery/comparison_results/distances_<measure>.csv

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
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from vizman import viz


# ---------------------------------------------------------------------------
# vizman setup — must come before any plt calls
# ---------------------------------------------------------------------------

viz.set_visual_style(font_family="Arial")

_pkg = Path(viz.__file__).parent
_colors = {
    name: val
    for cat in json.loads((_pkg / "colors.json").read_text()).values()
    for name, val in cat.items()
}
_cmaps = viz.give_colormaps()


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
#     (-6.9,    1.11),
#     ( 4.1,    1.11),
#     (-6.9,    0.01),
#     ( 4.1,    0.01),
#     (-3.734694004058838, 0.595918357372283),
# ]
    (-6.9,    0.89),
    ( 1.9,    0.89),
    (-6.9,    0.01),
    ( 1.9,    0.01),
    (-3.734694004058838, 0.595918357372283),
]

COMBO_LABELS = [
    r"$\eta{=}{-}6.9,\ \gamma{=}0.89$",
    r"$\eta{=}1.9,\ \gamma{=}0.89$",
    r"$\eta{=}{-}6.9,\ \gamma{=}0.01$",
    r"$\eta{=}1.9,\ \gamma{=}0.01$",
    r"$\eta{=}{-}3.73,\ \gamma{=}0.60$",
]

# Five visually distinct vizman colours, one per parameter combination
COMBO_COLORS = [
    _colors["LECKER_RED"],
    _colors["LAKE_BLUE"],
    _colors["JUST_GREEN"],
    _colors["ORANGE"],
    _colors["PURPLE"],
]

# Parameter space bounds for normalisation (η and γ sweep ranges)
ETA_RANGE   = (-8.0,  3.0)
GAMMA_RANGE = (-0.1,  1.0)

# Error metric for the accuracy panel: "mse" or "mae"
ERROR_METRIC = "mae"

# Scatter grid columns (rows computed automatically)
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
    """Return {measure_name: DataFrame} for every CSV in COMPARISON_DIR."""
    results = {}
    for csv_path in sorted(COMPARISON_DIR.glob("distances_*.csv")):
        measure = csv_path.stem.replace("distances_", "")
        results[measure] = pd.read_csv(csv_path)
        print(f"  Loaded {measure}: {len(results[measure])} rows")
    return results


def _add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append normalised parameter columns. Does not modify in place."""
    df = df.copy()
    eta_span   = ETA_RANGE[1]   - ETA_RANGE[0]
    gamma_span = GAMMA_RANGE[1] - GAMMA_RANGE[0]

    df["true_eta_n"]   = (df["true_eta"]       - ETA_RANGE[0])   / eta_span
    df["true_gamma_n"] = (df["true_gamma"]      - GAMMA_RANGE[0]) / gamma_span
    df["pred_eta_n"]   = (df["predicted_eta"]   - ETA_RANGE[0])   / eta_span
    df["pred_gamma_n"] = (df["predicted_gamma"] - GAMMA_RANGE[0]) / gamma_span

    d_eta   = df["pred_eta_n"]   - df["true_eta_n"]
    d_gamma = df["pred_gamma_n"] - df["true_gamma_n"]
    df["sq_error"]  = d_eta ** 2 + d_gamma ** 2
    df["abs_error"] = np.sqrt(df["sq_error"])
    return df


def compute_accuracy_error(df: pd.DataFrame) -> float:
    """MSE or MAE in normalised parameter space. Lower = better accuracy."""
    df = _add_normalized_columns(df)
    return df["abs_error" if ERROR_METRIC == "mae" else "sq_error"].mean()


def compute_mse(df: pd.DataFrame) -> float:
    return _add_normalized_columns(df)["sq_error"].mean()


def compute_variability(df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Per-combo variability: mean of std(pred_η_norm) and std(pred_γ_norm)
    across N_TEST networks.  Lower = more consistent predictions.
    """
    df = _add_normalized_columns(df)
    per_combo = []
    for combo_idx in range(len(PARAM_COMBOS)):
        sub = df[df["true_combo_idx"] == combo_idx]
        per_combo.append(
            (sub["pred_eta_n"].std(ddof=0) + sub["pred_gamma_n"].std(ddof=0)) / 2.0
        )
    return np.array(per_combo), float(np.mean(per_combo))


def _panel_label(ax, letter: str) -> None:
    """Add a bold uppercase panel label (A, B, …) to the top-left."""
    ax.text(
        -0.12, 1.06, letter,
        transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", ha="left",
        color=_colors["HALF_BLACK"],
    )


# ---------------------------------------------------------------------------
# Figure 1 — Scatter grid: true vs predicted in η-γ space
# ---------------------------------------------------------------------------

def _draw_scatter_ax(ax, df: pd.DataFrame, measure: str,
                     show_xlabel: bool, show_ylabel: bool,
                     panel_letter: str) -> None:
    """Populate a single scatter subplot."""
    rng = np.random.default_rng(42)

    # True locations — bold X, drawn before predicted dots
    for true_idx, (te, tg) in enumerate(PARAM_COMBOS):
        ax.scatter(te, tg, marker="x", s=80,
                   color=COMBO_COLORS[true_idx],
                   linewidths=2.0, zorder=2)

    # Predicted locations (jittered dots) — drawn on top of X
    for true_idx, (true_eta, true_gamma) in enumerate(PARAM_COMBOS):
        color = COMBO_COLORS[true_idx]
        sub   = df[df["true_combo_idx"] == true_idx]
        jitter      = rng.normal(0, 0.06, size=(len(sub), 2))
        pred_etas   = sub["predicted_eta"].values   + jitter[:, 0]
        pred_gammas = sub["predicted_gamma"].values + jitter[:, 1]
        ax.scatter(pred_etas, pred_gammas,
                   color=color, alpha=0.5, s=20, zorder=3, linewidths=0)

    err   = compute_accuracy_error(df)
    label = "MAE" if ERROR_METRIC == "mae" else "MSE"
    name  = METRIC_NAMES.get(measure, measure.replace("_", " ").title())
    ax.set_title(f"{name}\n{label}={err:.3f}", fontsize=7, pad=3)

    if show_xlabel:
        ax.set_xlabel(r"$\eta$", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(r"$\gamma$", fontsize=8)

    ax.set_xticks([-8, 0, 3])
    ax.set_yticks([-0.1, 0, 1])
    ax.tick_params(labelsize=6)
    sns.despine(ax=ax)
    _panel_label(ax, panel_letter)


def plot_scatter(results: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """4×N grid of scatter subplots, one per distance measure."""
    n_measures = len(results)
    n_cols     = min(SCATTER_NCOLS, n_measures)
    n_rows     = math.ceil(n_measures / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=viz.cm_to_inch((18, 4.8 * n_rows)),
        sharex=True, sharey=True,
        squeeze=False,
    )

    # Shared limits and ticks — set once, propagated via sharex/sharey
    # Span all true combo values plus padding; gamma goes to 1.11 so extend ylim
    axes[0, 0].set_xlim(-8.8, 5.2)
    axes[0, 0].set_ylim(-0.2,  1.3)
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

    for empty_idx in range(n_measures, n_rows * n_cols):
        row, col = divmod(empty_idx, n_cols)
        axes[row, col].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=COMBO_COLORS[i], label=COMBO_LABELS[i])
        for i in range(len(PARAM_COMBOS))
    ]
    fig.legend(
        handles=legend_patches,
        title="True parameter combination",
        loc="lower center",
        ncol=len(PARAM_COMBOS),
        fontsize=7,
        title_fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        r"Parameter recovery — true ($\times$) vs predicted ($\bullet$)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Accuracy (MAE/MSE) + Precision (variability)
# ---------------------------------------------------------------------------

def plot_precision_accuracy(results: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Left  — Horizontal bar chart of MAE/MSE per measure, sorted best→worst.
    Right — Heatmap of per-combo variability (measures × param combos).
    """
    measures   = list(results.keys())
    n_measures = len(measures)
    n_combos   = len(PARAM_COMBOS)

    error_values = np.array([compute_accuracy_error(results[m]) for m in measures])
    var_matrix   = np.zeros((n_measures, n_combos))
    for m_idx, measure in enumerate(measures):
        per_combo, _ = compute_variability(results[measure])
        var_matrix[m_idx] = per_combo

    # Sort best (lowest error) first
    order        = np.argsort(error_values)
    sorted_err   = error_values[order]
    sorted_labels = [measures[i].replace("_", " ") for i in order]
    sorted_var    = var_matrix[order]

    error_label = "MAE" if ERROR_METRIC == "mae" else "MSE"

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2,
        figsize=viz.cm_to_inch((18, max(6, n_measures * 0.55 + 2))),
        gridspec_kw={"width_ratios": [1.5, 1]},
    )

    # ---- Panel A: horizontal bar chart ------------------------------------
    # Colour encodes rank: best → JUST_GREEN, worst → LECKER_RED
    bar_cmap   = plt.cm.get_cmap("RdYlGn_r", n_measures)
    bar_colors = [bar_cmap(i / max(n_measures - 1, 1)) for i in range(n_measures)]

    bars = ax_bar.barh(
        range(n_measures), sorted_err,
        color=bar_colors,
        edgecolor=_colors["HALF_BLACK"], linewidth=0.3,
        height=0.6,
    )
    ax_bar.set_yticks(range(n_measures))
    ax_bar.set_yticklabels(sorted_labels, fontsize=7)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(f"{error_label}  (normalised space)", fontsize=8)
    ax_bar.set_xlim(0, sorted_err.max() * 1.22 + 1e-8)
    ax_bar.axvline(0, color=_colors["HALF_BLACK"], linewidth=0.6)

    for bar, val in zip(bars, sorted_err):
        ax_bar.text(
            val + sorted_err.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=6,
            color=_colors["HALF_BLACK"],
        )

    sns.despine(ax=ax_bar, left=True)
    ax_bar.tick_params(left=False)
    _panel_label(ax_bar, "A")

    # ---- Panel B: heatmap -------------------------------------------------
    combo_short = [f"η={e:.1f}\nγ={g:.2f}" for e, g in PARAM_COMBOS]

    im = ax_heat.imshow(
        sorted_var,
        aspect="auto",
        cmap=_cmaps["bw_lr"],
        vmin=0,
        vmax=sorted_var.max() or 1,
    )
    ax_heat.set_xticks(range(n_combos))
    ax_heat.set_xticklabels(combo_short, fontsize=6)
    ax_heat.set_yticks(range(n_measures))
    ax_heat.set_yticklabels(sorted_labels, fontsize=7)
    ax_heat.set_title("Precision  (variability, ↓ better)", fontsize=8, pad=4)

    # Cell annotations
    threshold = sorted_var.max() * 0.55 if sorted_var.max() > 0 else 1
    for row in range(n_measures):
        for col in range(n_combos):
            val = sorted_var[row, col]
            txt_col = _colors["BONE_WHITE"] if val > threshold else _colors["HALF_BLACK"]
            ax_heat.text(col, row, f"{val:.2f}",
                         ha="center", va="center",
                         fontsize=5.5, color=txt_col)

    cb = fig.colorbar(im, ax=ax_heat, shrink=0.7, pad=0.03)
    cb.set_label("Mean std (norm. space)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    _panel_label(ax_heat, "B")

    fig.suptitle(
        rf"Parameter recovery quality — norm. $\eta\in[{ETA_RANGE[0]},{ETA_RANGE[1]}]$,"
        rf" $\gamma\in[{GAMMA_RANGE[0]},{GAMMA_RANGE[1]}]$",
        fontsize=8, y=1.01,
    )
    fig.tight_layout()
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
            f"No comparison CSVs found in:\n  {COMPARISON_DIR}\n"
            "Run run_synthetic_gnm_comparison.py first."
        )
        return

    print(f"\nFound {len(results)} measure(s): {list(results.keys())}")

    print(f"\n--- {ERROR_METRIC.upper()} / variability summary (normalised space) ---")
    for measure, df in results.items():
        err            = compute_accuracy_error(df)
        _, mean_variab = compute_variability(df)
        print(f"  {measure:35s}  {ERROR_METRIC.upper()}={err:.4f}   variability={mean_variab:.4f}")

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
