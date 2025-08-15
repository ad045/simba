# esn_analysis.py
"""
Analyze ESN hyperparameter sweeps and produce publication-quality plots.
Assumes per-combination results or per-subject results saved as CSVs.
"""

from pathlib import Path
from typing import Iterable, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


from initialize_plots import apply_plot_style, save_figure, CMAP_GENERIC

# ------------------------
# IO helpers
# ------------------------

def _find_csvs(root: Path, pattern: str) -> list[Path]:
    """Recursively collect CSVs under `root` matching `pattern`."""
    return sorted(root.rglob(pattern))

def load_esn_results(root: Path) -> pd.DataFrame:
    """
    Load and concatenate ESN result CSVs (final or partial).
    Returns a single DataFrame with harmonized column names.
    """
    # Try common patterns; tweak here if your filenames differ
    candidates = (
        _find_csvs(root, "esn_*results_*.csv") +
        _find_csvs(root, "*esn_mc_results*.csv") +
        _find_csvs(root, "esn_grid_*/*.csv")
    )
    if not candidates:
        raise FileNotFoundError(f"No ESN result CSVs found under: {root}")

    dfs = []
    for p in candidates:
        try:
            df = pd.read_csv(p)
            df["__source"] = str(p)
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        raise RuntimeError("Found files but none could be loaded.")

    df_all = pd.concat(dfs, ignore_index=True)

    # Harmonize column names (robust to slight naming variations)
    rename_map = {
        "density": "density_percent",
        "density_pct": "density_percent",
        "method": "regression_method",
        "reg_method": "regression_method",
        "alpha": "ridge_alpha",
        "act": "activation",
        "mc": "mc_mean",        # if file only has aggregate, treat as mean
        "MC": "mc_mean",
    }
    df_all = df_all.rename(columns=rename_map)

    # Ensure presence of expected columns (create if missing)
    for c, default in [
        ("activation", "tanh"),
        ("ridge_alpha", np.nan),
        ("leak_rate", 1.0),
        ("input_scaling", 1.0),
        ("spectral_radius", 1.0),
        ("regression_method", "pinv"),
        ("density_percent", np.nan),
    ]:
        if c not in df_all.columns:
            df_all[c] = default

    # prefer explicit mc_mean / mc_std; if only per-subject is present, aggregate later
    return df_all


# ------------------------
# Aggregations
# ------------------------

HYPERPARAMS = [
    "density_percent", "regression_method", "activation",
    "spectral_radius", "input_scaling", "leak_rate", "ridge_alpha"
]

def _has_column(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def aggregate_mc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate memory capacity across subjects for each hyperparam combination.
    Returns a tidy dataframe with mc_mean, mc_std, n_subjects.
    """
    # Detect if we have per-subject MC (column 'mc' or 'mc_subject') or already aggregated
    if _has_column(df, "mc") and not _has_column(df, "mc_mean"):
        agg = (
            df.groupby(HYPERPARAMS, dropna=False)["mc"]
              .agg(mc_mean="mean", mc_std="std", n_subjects="count")
              .reset_index()
        )
    elif _has_column(df, "mc_mean"):
        # Some files already provide mean/std; if multiple rows per combo, average them
        to_use = ["mc_mean"]
        if _has_column(df, "mc_std"):
            to_use.append("mc_std")
        agg = (
            df.groupby(HYPERPARAMS, dropna=False)[to_use]
              .mean()
              .reset_index()
        )
        if "mc_std" not in agg.columns:
            agg["mc_std"] = np.nan
        # n_subjects may be absent; best effort from original df
        if _has_column(df, "n_subjects"):
            nmap = df.groupby(HYPERPARAMS, dropna=False)["n_subjects"].max()
            agg = agg.merge(nmap, on=HYPERPARAMS, how="left")
        else:
            agg["n_subjects"] = np.nan
    else:
        raise ValueError("Could not find 'mc' or 'mc_mean' columns in ESN results.")
    return agg.sort_values("mc_mean", ascending=False)


# ------------------------
# Plots
# ------------------------

def _facet_values(df: pd.DataFrame, col: str) -> list:
    vals = sorted([v for v in df[col].dropna().unique()])
    return vals if vals else ["(all)"]

def plot_mc_heatmaps_by_density_and_method(df: pd.DataFrame, outdir: Path) -> list[Path]:
    """
    Heatmaps of MC vs (spectral_radius × input_scaling),
    faceted by (density_percent × regression_method).
    """
    out_paths = []
    dens_vals = _facet_values(df, "density_percent")
    meth_vals = _facet_values(df, "regression_method")

    for d in dens_vals:
        df_d = df if pd.isna(d) else df[df["density_percent"] == d]
        for m in meth_vals:
            df_dm = df_d if m == "(all)" else df_d[df_d["regression_method"] == m]
            if df_dm.empty:
                continue
            # pick best across other dims
            cols = ["spectral_radius", "input_scaling"]
            pivot = (df_dm
                     .groupby(cols, as_index=False)["mc_mean"]
                     .mean()
                     .pivot(index="input_scaling", columns="spectral_radius", values="mc_mean"))

            if pivot.isna().all().all():
                continue

            fig, ax = plt.subplots(figsize=(6, 4.8))
            im = ax.imshow(pivot.values, origin="lower", aspect="auto",
                           extent=[pivot.columns.min(), pivot.columns.max(),
                                   pivot.index.min(), pivot.index.max()],
                           cmap=CMAP_GENERIC)
            c = fig.colorbar(im, ax=ax)
            c.set_label("MC (mean)")
            ax.set_xlabel("spectral radius")
            ax.set_ylabel("input scaling")
            title = f"MC heatmap | density={d} | method={m}"
            ax.set_title(title)
            out = outdir / f"esn_mc_heatmap_density{d}_method{m}.mc"
            save_figure(fig, str(out))
            plt.close(fig)
            out_paths.append(Path(str(out) + ".png"))
    return out_paths

def plot_mc_vs_spectral_radius(df: pd.DataFrame, outdir: Path) -> Path:
    """Line plot MC vs spectral radius, colored by leak rate, faceted by activation."""
    activations = _facet_values(df, "activation")
    fig, axes = plt.subplots(1, len(activations), figsize=(6*len(activations), 4.2), squeeze=False)
    axes = axes[0]
    for ax, act in zip(axes, activations):
        dd = df if act == "(all)" else df[df["activation"] == act]
        if dd.empty:
            continue
        # average across other dims except spectral_radius & leak_rate
        keep = ["spectral_radius", "leak_rate"]
        mean_df = (dd.groupby(keep, as_index=False)["mc_mean"].mean()
                     .sort_values(["leak_rate", "spectral_radius"]))
        for lr, grp in mean_df.groupby("leak_rate"):
            ax.plot(grp["spectral_radius"], grp["mc_mean"], marker="o", label=f"leak={lr:.2f}")
        ax.set_title(f"activation={act}")
        ax.set_xlabel("spectral radius")
        ax.set_ylabel("MC (mean)")
        ax.legend(frameon=False)
    out = outdir / "esn_mc_vs_spectral_radius"
    save_figure(fig, str(out))
    plt.close(fig)
    return Path(str(out) + ".png")

def plot_method_comparison(df: pd.DataFrame, outdir: Path) -> Path:
    """Bar chart comparing avg MC by regression method (pinv vs ridge)."""
    fig, ax = plt.subplots(figsize=(5, 3.6))
    m = df.groupby("regression_method", as_index=False)["mc_mean"].mean()
    ax.bar(m["regression_method"], m["mc_mean"])
    ax.set_ylabel("MC (mean)")
    ax.set_title("Regression method comparison")
    out = outdir / "esn_method_comparison"
    save_figure(fig, str(out))
    plt.close(fig)
    return Path(str(out) + ".png")

def _draw_heatmap_on_ax(df: pd.DataFrame, density, method: str, ax: plt.Axes) -> bool:
    """Draw a MC heatmap for a (density, method) combo on a provided axis. Returns True if plotted."""
    df_d = df if (pd.isna(density) or density == "(all)") else df[df["density_percent"] == density]
    df_dm = df_d if (method == "(all)") else df_d[df_d["regression_method"] == method]
    if df_dm.empty:
        ax.set_axis_off()
        return False

    cols = ["spectral_radius", "input_scaling"]
    pivot = (df_dm.groupby(cols, as_index=False)["mc_mean"]
                  .mean()
                  .pivot(index="input_scaling", columns="spectral_radius", values="mc_mean"))
    if pivot.isna().all().all():
        ax.set_axis_off()
        return False

    im = ax.imshow(pivot.values, origin="lower", aspect="auto",
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(), pivot.index.max()],
                   cmap=CMAP_GENERIC)
    ax.figure.colorbar(im, ax=ax).set_label("MC (mean)")
    ax.set_xlabel("spectral radius")
    ax.set_ylabel("input scaling")
    ax.set_title(f"density={density}, method={method}")
    return True


def _draw_mc_vs_sr_on_ax(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw MC vs spectral radius (colored by leak rate) on the given axis."""
    keep = ["spectral_radius", "leak_rate"]
    mean_df = (df.groupby(keep, as_index=False)["mc_mean"]
                 .mean()
                 .sort_values(["leak_rate", "spectral_radius"]))
    if mean_df.empty:
        ax.set_axis_off()
        return
    for lr, grp in mean_df.groupby("leak_rate"):
        ax.plot(grp["spectral_radius"], grp["mc_mean"], marker="o", label=f"leak={lr:.2f}")
    ax.set_xlabel("spectral radius")
    ax.set_ylabel("MC (mean)")
    ax.set_title("MC vs spectral radius (avg over other params)")
    ax.legend(frameon=False)


def _draw_method_comparison_on_ax(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw bar chart comparing mean MC by regression method."""
    m = df.groupby("regression_method", as_index=False)["mc_mean"].mean()
    if m.empty:
        ax.set_axis_off()
        return
    ax.bar(m["regression_method"], m["mc_mean"])
    ax.set_ylabel("MC (mean)")
    ax.set_title("Regression method comparison")

def plot_all_in_one(df: pd.DataFrame, outdir: Path, max_heatmaps: int = 3) -> Path:
    """
    Composite figure: a row of MC heatmaps (up to `max_heatmaps`) and a row with
    (MC vs spectral radius) and (method comparison) side by side.
    """
    dens_vals = _facet_values(df, "density_percent")
    meth_vals = _facet_values(df, "regression_method")

    # Pick up to `max_heatmaps` (density, method) combos that actually produce a heatmap
    selected: list[tuple] = []
    for d in dens_vals:
        for m in meth_vals:
            selected.append((d, m))
            if len(selected) >= max_heatmaps:
                break
        if len(selected) >= max_heatmaps:
            break

    n_heat = len(selected)
    ncols = max(n_heat, 2)  # ensure room for bottom row (2 panels)

    # Layout
    fig = plt.figure(figsize=(4.8 * ncols, 9.0))
    gs = gridspec.GridSpec(2, ncols, height_ratios=[1.0, 1.0], figure=fig)

    # Top row: heatmaps
    for i, (d, m) in enumerate(selected):
        ax = fig.add_subplot(gs[0, i])
        plotted = _draw_heatmap_on_ax(df, d, m, ax)
        if not plotted:
            ax.set_axis_off()

    # Bottom row: left = MC vs SR, right = method comparison
    ax_left  = fig.add_subplot(gs[1, : ncols // 2 or 1])
    ax_right = fig.add_subplot(gs[1, ncols // 2 or 1 :])

    _draw_mc_vs_sr_on_ax(df, ax_left)
    _draw_method_comparison_on_ax(df, ax_right)

    out = outdir / "esn_all_in_one"
    save_figure(fig, str(out))
    plt.close(fig)
    return Path(str(out) + ".png")

# ------------------------
# CLI
# ------------------------

def main_esn(root_dir: str, out_dir: str, topk: int = 20) -> None:
    """End-to-end ESN analysis from CSVs in `root_dir` → plots in `out_dir`."""
    apply_plot_style()
    root = Path(root_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_all = load_esn_results(root)
    df_agg = aggregate_mc(df_all)

    # Save top-k table
    top_path = out / "esn_topK.csv"
    df_agg.head(topk).to_csv(top_path, index=False)

    # Plots
    plot_mc_heatmaps_by_density_and_method(df_agg, out)
    plot_mc_vs_spectral_radius(df_agg, out)
    plot_method_comparison(df_agg, out)
    plot_all_in_one(df_agg, out, max_heatmaps=3)


if __name__ == "__main__":
    # import argparse
    # p = argparse.ArgumentParser()
    # p.add_argument("--root", required=True, help="Directory containing ESN result CSVs (partial or final).")
    # p.add_argument("--out", required=True, help="Directory to save plots/tables.")
    # p.add_argument("--topk", type=int, default=20)
    # args = p.parse_args()
    # main_esn(args.root, args.out, args.topk)
    
    # either a folder or a path to one sweep
    one_path = "/Users/adrian/Documents/01_projects/14_4D_lab/output/02_gnm_estimation"
    # one_path = "/Users/adrian/Documents/01_projects/14_4D_lab/output/02_gnm_estimation/esn_grid_resolution68_2025-08-15_01-22-16"
    # if the last part of the path is not an int, then get the latest file name of data
    if not one_path or not one_path[-1].isdigit():
        from src.utils.saving_and_finding_files import get_latest_file_name_of_data
        file_name = get_latest_file_name_of_data(one_path, "esn_grid_resolution*.csv").split("/")[:-1]
        file_name = "/".join(file_name)
    else: # if it is already a path to one sweep 
        file_name = one_path
        
        
    main_esn(
         root_dir =  one_path,
         out_dir = one_path + "/plots",
         topk = 20)