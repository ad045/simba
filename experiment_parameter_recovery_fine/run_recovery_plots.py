"""
Grid-step style plots for the parameter-recovery experiments (wide AND fine).
=============================================================================

One code path, two experiments - pick with ``--experiment``:

  wide   5 widely-spread true (eta, gamma) combos, 10 test networks each,
         recovered against a coarse 10x10 grid spanning the whole morphospace
         (output/gnm/synthetic_parameter_recovery_grid/)

  fine   100 continuous true points sampled from the human-plausible window,
         recovered against a 10x10 grid covering that window
         (output/gnm/synthetic_parameter_recovery_fine/)

Both experiments produce exactly the same figure set, so the two are directly
comparable and cannot drift apart. Everything is computed from the cached
comparison CSVs - no (expensive) regeneration is required.

Orientation
-----------
Five of the 16 measures (jaccard, f1, communicability_corr, network_mutual_
information, dc_network_mutual_information) return a SIMILARITY, where a higher
value means MORE similar. Every landscape read here is passed through
``experiments_config.to_distance`` first, so "lower = closer" holds for all 16
and the argmin marker on the landscape is the actually-recovered cell.

Produces (into <experiment output dir>/plots/):
  landscape_<exp>_<measure>_grid_steps.pdf   - distance landscape for a handful
                                               of representative true points
  parameter_recovery_scatter_<exp>_grid_steps.pdf
                                             - true -> recovered "recovery
                                               field" in eta-gamma space
  grid_steps_bars_<exp>.pdf                  - mean grid-step error per measure
  <exp>_recovery_r_bars.pdf                  - Pearson r (true vs recovered)
  <exp>_recovery_scatter_top.pdf             - true vs recovered, top measures

Usage
-----
  python run_recovery_plots.py --experiment fine
  python run_recovery_plots.py --experiment wide
  python run_recovery_plots.py --experiment both
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

from vizman import viz

# Experiment config (single source of truth, at the benchmarking repo root).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_config import coarse as _coarse, fine as _fine, to_distance


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

ROOT_DIR = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")

EXPERIMENTS = {
    "fine": SimpleNamespace(
        cfg=_fine,
        out_dir=ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_fine",
        label="Fine-window",
    ),
    "wide": SimpleNamespace(
        cfg=_coarse,
        out_dir=ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_grid",
        label="Wide-target",
    ),
}

# Mean Euclidean distance in grid-index space (0 = correct cell).
ERROR_METRIC = "grid_steps"

SCATTER_NCOLS = 4
DPI = 300

# How many representative true points to show in the landscape figure.
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

# Set by configure(); every helper below reads these.
EXP = CFG = COMPARISON_DIR = PLOT_DIR = None
GRID_ETA = GRID_GAMMA = GRID_COMBOS = None
GRID_N_ETA = GRID_N_GAMMA = None
ETA_RANGE = GAMMA_RANGE = None


def configure(experiment: str) -> None:
    """Point the module at one of the two recovery experiments."""
    global EXP, CFG, COMPARISON_DIR, PLOT_DIR
    global GRID_ETA, GRID_GAMMA, GRID_COMBOS, GRID_N_ETA, GRID_N_GAMMA
    global ETA_RANGE, GAMMA_RANGE

    spec = EXPERIMENTS[experiment]
    EXP, CFG = experiment, spec.cfg
    COMPARISON_DIR = spec.out_dir / "comparison_results"
    PLOT_DIR = spec.out_dir / "plots"
    GRID_ETA, GRID_GAMMA, GRID_COMBOS = CFG.GRID_ETA, CFG.GRID_GAMMA, CFG.GRID_COMBOS
    GRID_N_ETA, GRID_N_GAMMA = CFG.GRID_N_ETA, CFG.GRID_N_GAMMA
    ETA_RANGE, GAMMA_RANGE = CFG.ETA_RANGE, CFG.GAMMA_RANGE


def _exp_label() -> str:
    return EXPERIMENTS[EXP].label


def _target_description() -> str:
    """One line describing what the ground truth of this experiment is."""
    if EXP == "fine":
        return (rf"$\eta\in[{CFG.SAMPLE_ETA_RANGE[0]},{CFG.SAMPLE_ETA_RANGE[1]}]$, "
                rf"$\gamma\in[{CFG.SAMPLE_GAMMA_RANGE[0]},{CFG.SAMPLE_GAMMA_RANGE[1]}]$")
    return f"{len(CFG.TRUE_PARAM_COMBOS)} widely-spread true combinations"


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


def _pearson(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _error_label() -> str:
    return "Grid steps"


def _panel_label(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter,
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
            color=_colors["HALF_BLACK"])


def representative_gt(df: pd.DataFrame) -> List[Tuple[int, str]]:
    """Pick N_LANDSCAPE_REP true-point rows to show as landscapes.

    Wide: one row per discrete true combination (that IS the design).
    Fine: rows nearest to the 4 corners + centre of the sampling window, so the
    figure illustrates recovery across the plausible region.

    Selection is based on true params only, so it is identical for every measure.
    Returns a list of (row position in df, short label).
    """
    if EXP == "wide":
        picks = []
        for combo_idx, label in enumerate(CFG.COMBO_LABELS[:N_LANDSCAPE_REP]):
            match = np.flatnonzero(df["true_combo_idx"].values == combo_idx)
            if match.size:
                picks.append((int(match[0]), label))
        return picks

    e_lo, e_hi = CFG.SAMPLE_ETA_RANGE
    g_lo, g_hi = CFG.SAMPLE_GAMMA_RANGE
    anchors = [
        ((e_lo, g_lo), r"low $\eta$, low $\gamma$"),
        ((e_lo, g_hi), r"low $\eta$, high $\gamma$"),
        ((e_hi, g_lo), r"high $\eta$, low $\gamma$"),
        ((e_hi, g_hi), r"high $\eta$, high $\gamma$"),
        ((CFG.CENTER_ETA, CFG.CENTER_GAMMA), "centre"),
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
# Figure 1 - Landscape heatmaps for representative true points
# ---------------------------------------------------------------------------

def _row_landscape(df: pd.DataFrame, row_pos: int, measure: str) -> np.ndarray:
    """(GRID_N_GAMMA, GRID_N_ETA) ORIENTED distance matrix for one true point.

    Oriented means: similarity measures are flipped, so low = close for all 16.
    """
    vals = df.iloc[row_pos][_dist_grid_cols()].values.astype(float)
    return to_distance(vals, measure).reshape(GRID_N_GAMMA, GRID_N_ETA)


def plot_landscapes(results: Dict[str, pd.DataFrame]) -> None:
    """One PDF per measure: distance landscape for N_LANDSCAPE_REP true points."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for measure, df in results.items():
        reps = representative_gt(df)
        n = len(reps)
        fig, axes = plt.subplots(1, n, figsize=viz.cm_to_inch((18, 4.5)),
                                 squeeze=False)

        landscapes = [_row_landscape(df, pos, measure) for pos, _ in reps]
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

            # Recovered = argmin of the ORIENTED landscape.
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
        cb.set_label("Distance (oriented)")

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

        out_path = PLOT_DIR / f"landscape_{EXP}_{measure}_grid_steps.pdf"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved landscape -> {out_path}")


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
        figsize=viz.cm_to_inch((18, 4.5 * n_rows)),
        sharex=True, sharey=True) #  squeeze=False,
    
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
        rf"{_exp_label()} recovery field - true points $\to$ recovered grid cell "
        r"(coloured by grid-step error)")
    out_path = PLOT_DIR / f"parameter_recovery_scatter_{EXP}_grid_steps.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter -> {out_path}")


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
                  "ground-truth networks)", fontsize=8)
    ax.set_xlim(0, sorted_err.max() * 1.22 + 1e-8)
    ax.axvline(0, color=_colors["HALF_BLACK"], linewidth=0.6)
    for bar, val in zip(bars, sorted_err):
        ax.text(val + sorted_err.max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=6,
                color=_colors["HALF_BLACK"])
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)

    fig.suptitle(f"{_exp_label()} recovery accuracy - {_target_description()} "
                 "(lower is better)", fontsize=8, y=1.01)
    fig.tight_layout()

    out_path = PLOT_DIR / f"grid_steps_bars_{EXP}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"{out_path}")


# ---------------------------------------------------------------------------
# Figure 4 - Pearson r (true vs recovered) per measure
# ---------------------------------------------------------------------------

def r_scores(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-measure Pearson r for eta and gamma, best mean r first."""
    rows = []
    for measure, df in results.items():
        r_eta   = _pearson(df["true_eta"],   df["recovered_eta"])
        r_gamma = _pearson(df["true_gamma"], df["recovered_gamma"])
        rows.append({
            "measure": measure,
            "name": METRIC_NAMES.get(measure, measure),
            "r_eta": r_eta, "r_gamma": r_gamma,
            "r_mean": np.nanmean([r_eta, r_gamma]),
            "grid_steps": compute_grid_step_error(df),
        })
    return (pd.DataFrame(rows).sort_values("r_mean", ascending=False)
                              .reset_index(drop=True))


def plot_r_bars(scores: pd.DataFrame, n_truth: int) -> None:
    BLUE, GREEN = _colors["LAKE_BLUE"], _colors["JUST_GREEN"]
    DARK = _colors["HALF_BLACK"]

    n = len(scores)
    fig, ax = plt.subplots(figsize=viz.cm_to_inch((18, max(6, n * 0.55 + 2))))
    y = np.arange(n)[::-1]  # best at top
    h = 0.38
    ax.barh(y + h / 2, scores["r_eta"],   height=h, color=BLUE,  label=r"$\eta$",
            edgecolor=DARK, linewidth=0.3)
    ax.barh(y - h / 2, scores["r_gamma"], height=h, color=GREEN, label=r"$\gamma$",
            edgecolor=DARK, linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(scores["name"], fontsize=7)
    ax.set_xlabel("Pearson r  (true vs recovered)")
    ax.set_xlim(min(0, np.nanmin(scores[["r_eta", "r_gamma"]].values)) - 0.05, 1.0)
    ax.axvline(0, color=DARK, linewidth=0.6)
    for yi, (re, rg) in zip(y, zip(scores["r_eta"], scores["r_gamma"])):
        if not np.isnan(re):
            ax.text(re + 0.01, yi + h / 2, f"{re:.2f}", va="center",
                    fontsize=5.5, color=DARK)
        if not np.isnan(rg):
            ax.text(rg + 0.01, yi - h / 2, f"{rg:.2f}", va="center",
                    fontsize=5.5, color=DARK)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(f"{_exp_label()} parameter recovery\n"
                 f"({_target_description()}, {n_truth} ground-truth networks)",
                 fontsize=9)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()

    out_path = PLOT_DIR / f"{EXP}_recovery_r_bars.pdf"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 - True vs recovered scatter for the top measures
# ---------------------------------------------------------------------------

def plot_scatter_top(results: Dict[str, pd.DataFrame], scores: pd.DataFrame) -> None:
    BLUE, GREEN = _colors["LAKE_BLUE"], _colors["JUST_GREEN"]
    GRAY, DARK  = _colors["GRAY"], _colors["HALF_BLACK"]

    top = scores.head(min(4, len(scores)))
    fig, axes = plt.subplots(2, len(top),
                             figsize=viz.cm_to_inch((4.5 * len(top), 9)),
                             squeeze=False)
    for j, (_, srow) in enumerate(top.iterrows()):
        df = results[srow["measure"]]
        for i, (param, lo_hi, color) in enumerate([
            ("eta",   ETA_RANGE,   BLUE),
            ("gamma", GAMMA_RANGE, GREEN),
        ]):
            ax = axes[i][j]
            rng = np.random.default_rng(0)
            jit = rng.normal(0, (lo_hi[1] - lo_hi[0]) * 0.004, size=len(df))
            ax.scatter(df[f"true_{param}"], df[f"recovered_{param}"] + jit,
                       s=14, alpha=0.55, color=color, linewidths=0)
            ax.plot(lo_hi, lo_hi, color=GRAY, lw=0.8, ls="--", zorder=0)
            rr = _pearson(df[f"true_{param}"], df[f"recovered_{param}"])
            if i == 0:
                ax.set_title(f"{srow['name']}", fontsize=8)
            ax.text(0.04, 0.92, f"r={rr:.2f}", transform=ax.transAxes,
                    fontsize=7, va="top", color=DARK)
            if j == 0:
                ax.set_ylabel(rf"recovered $\{param}$")
            if i == 1:
                ax.set_xlabel(rf"true $\{param}$")
    fig.suptitle(f"{_exp_label()}: true vs recovered (top measures by mean r)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = PLOT_DIR / f"{EXP}_recovery_scatter_top.pdf"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(experiment: str) -> None:
    configure(experiment)
    print(f"\n{'='*70}\n{experiment.upper()} experiment: {COMPARISON_DIR}\n{'='*70}")

    print("Loading results ...")
    results = load_all_results()
    if not results:
        print(f"No CSVs found in {COMPARISON_DIR}\n"
              "Run the corresponding --stage compare first.")
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

    scores = r_scores(results)
    scores.to_csv(PLOT_DIR / f"{EXP}_recovery_scores.csv", index=False)

    print("\nPlotting Pearson r bars ...")
    plot_r_bars(scores, n_truth=len(next(iter(results.values()))))

    print("\nPlotting true-vs-recovered scatter ...")
    plot_scatter_top(results, scores)

    print(f"\nAll plots saved -> {PLOT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["wide", "fine", "both"], default="both")
    args = ap.parse_args()

    targets = ["wide", "fine"] if args.experiment == "both" else [args.experiment]
    for experiment in targets:
        run(experiment)


if __name__ == "__main__":
    main()
