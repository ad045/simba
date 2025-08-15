#!/usr/bin/env python3
# visualize_gnm_results.py
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_csv(results_dir: Path, pattern: str) -> Path | None:
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_results(results_dir: Path,
                 mc_path: Path | None = None,
                 dur_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if mc_path is None:
        mc_path = find_latest_csv(results_dir, "gnm_mc_results_*.csv")
    if dur_path is None:
        dur_path = find_latest_csv(results_dir, "gnm_durations_*.csv")

    if mc_path is None:
        raise FileNotFoundError("Could not find gnm_mc_results_*.csv in the given directory.")
    df_mc = pd.read_csv(mc_path)

    # Drop footer or malformed lines (e.g., "COMPLETED")
    df_mc = df_mc[pd.to_numeric(df_mc.get("subject", pd.Series(dtype=object)), errors="coerce").notna()].copy()
    df_mc["subject"] = df_mc["subject"].astype(int)

    df_dur = None
    if dur_path and dur_path.exists():
        df_dur = pd.read_csv(dur_path)
        df_dur = df_dur[pd.to_numeric(df_dur.get("subject", pd.Series(dtype=object)), errors="coerce").notna()].copy()
        df_dur["subject"] = df_dur["subject"].astype(int)

    return df_mc, df_dur


def expand_hparams(df_mc: pd.DataFrame) -> pd.DataFrame:
    # Parse dict-like strings in the 'hparams' column
    def parse_cell(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                try:
                    return json.loads(x)
                except Exception:
                    return {}
        return {}

    hp = df_mc["hyper_params"].apply(parse_cell)
    hp_norm = pd.json_normalize(hp)
    # Avoid column name clashes; fill missing if absent
    for col in hp_norm.columns:
        if col in df_mc.columns:
            df_mc[f"hp_param_{col}"] = hp_norm[col]
        else:
            df_mc[col] = hp_norm[col]

    # Common expected hparams (create if missing)
    for col in ["spectral_radius", "input_length", "train_len", "input_scaling", "regularization_method", "n_runs"]:
        if col not in df_mc.columns:
            df_mc[col] = np.nan

    return df_mc


def attach_timings_by_order(df_mc: pd.DataFrame, df_dur: pd.DataFrame | None) -> pd.DataFrame:
    if df_dur is None or df_dur.empty:
        return df_mc

    # Row-wise order mapping (durations are written right after MC per-task)
    df_mc = df_mc.reset_index(drop=False).rename(columns={"index": "row_ix"})
    df_dur = df_dur.reset_index(drop=False).rename(columns={"index": "row_ix"})

    # Align by position; if shapes mismatch, use the min length
    n = min(len(df_mc), len(df_dur))
    merged = df_mc.iloc[:n].merge(df_dur.iloc[:n][["row_ix", "time_metrics_sec", "time_esn_sec", "time_total_sec"]],
                                  on="row_ix", how="left")
    # Append any tail rows without timings (unlikely)
    if len(df_mc) > n:
        merged = pd.concat([merged, df_mc.iloc[n:]], ignore_index=True)
    return merged


def aggregate_per_hparams(df: pd.DataFrame) -> pd.DataFrame:
    # Detect which hparam columns are present
    candidate_cols = ["spectral_radius", "input_length", "train_len", "input_scaling", "regularization_method", "n_runs"]
    hp_cols = [c for c in candidate_cols if c in df.columns and df[c].notna().any()]
    if "train_len" in hp_cols and "input_length" in hp_cols:
        # Prefer the newer name
        hp_cols.remove("train_len")

    grouped = df.groupby(hp_cols, dropna=False, as_index=False).agg(
        mean_mc=("mc_mean", "mean"),
        std_mc=("mc_mean", "std"),
        median_mc=("mc_mean", "median"),
        n_subjects=("subject", "nunique"),
        mean_time_sec=("time_total_sec", "mean")
    ).sort_values(["mean_mc", "std_mc"], ascending=[False, True])

    return grouped, hp_cols


def pick_best_config(agg: pd.DataFrame) -> pd.Series:
    # Highest mean_mc, tie-breaker: lower std, then lower mean_time
    return agg.sort_values(["mean_mc", "std_mc", "mean_time_sec"], ascending=[False, True, True]).iloc[0]


def plot_topk_bar(agg: pd.DataFrame, hp_cols: list[str], outdir: Path, k: int = 15):
    topk = agg.head(k).copy()
    labels = [
        ", ".join(f"{c}={topk.iloc[i][c]}" for c in hp_cols if pd.notna(topk.iloc[i][c]))
        for i in range(len(topk))
    ]
    x = np.arange(len(topk))
    y = topk["mean_mc"].values
    yerr = topk["std_mc"].values

    plt.figure(figsize=(12, 6))
    plt.bar(x, y, yerr=yerr, alpha=0.9, capsize=4)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Mean MC across subjects")
    plt.title(f"Top {k} ESN configs (error = std across subjects)")
    plt.tight_layout()
    plt.savefig(outdir / "topk_mean_mc_bar.png", dpi=180)
    plt.close()


def plot_heatmaps(agg: pd.DataFrame, outdir: Path,
                  spectral_col: str = "spectral_radius",
                  x_col: str = "input_length",
                  stratify_cols: list[str] = None):
    """
    Heatmap of mean_mc vs spectral radius (y) and input length (x),
    stratified by input_scaling and regularization_method when present.
    """
    if stratify_cols is None:
        stratify_cols = []
    present_strata = [c for c in stratify_cols if c in agg.columns and agg[c].notna().any()]
    if x_col not in agg.columns or spectral_col not in agg.columns:
        return

    groups = agg.groupby(present_strata) if present_strata else [((), agg)]
    for key, sub in groups:
        pivot = sub.pivot_table(index=spectral_col, columns=x_col, values="mean_mc", aggfunc="mean")
        plt.figure(figsize=(8, 6))
        plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.colorbar(label="Mean MC")
        plt.yticks(ticks=np.arange(len(pivot.index)), labels=[f"{v:g}" for v in pivot.index])
        plt.xticks(ticks=np.arange(len(pivot.columns)), labels=[str(int(v)) if pd.notna(v) else "" for v in pivot.columns], rotation=45, ha="right")
        title_bits = []
        if present_strata:
            if isinstance(key, tuple):
                for c, v in zip(present_strata, key):
                    title_bits.append(f"{c}={v}")
            else:
                title_bits.append(f"{present_strata[0]}={key}")
        plt.title("Mean MC heatmap" + (": " + ", ".join(title_bits) if title_bits else ""))
        plt.xlabel(x_col)
        plt.ylabel(spectral_col)
        plt.tight_layout()

        fname = "heatmap_mean_mc"
        if title_bits:
            safe = "_".join(str(t).replace(" ", "").replace("/", "-") for t in title_bits)
            fname += f"__{safe}"
        plt.savefig(outdir / f"{fname}.png", dpi=180)
        plt.close()


def plot_lines_spectral_sweeps(agg: pd.DataFrame, outdir: Path,
                               spectral_col: str = "spectral_radius",
                               group_col: str = "input_scaling",
                               fix_col: str = "regularization_method",
                               x_col: str = "spectral_radius"):
    # If required columns are missing, skip
    need_cols = {spectral_col, group_col}
    if not need_cols.issubset(set(agg.columns)) or agg[spectral_col].isna().all():
        return

    # Optional: one panel per fix_col if available
    fix_vals = sorted(agg[fix_col].dropna().unique()) if fix_col in agg.columns and agg[fix_col].notna().any() else [None]
    for fx in fix_vals:
        sub = agg if fx is None else agg[agg[fix_col] == fx]
        if sub.empty:
            continue

        plt.figure(figsize=(9, 5))
        for gval, gdf in sub.groupby(group_col):
            # For each spectral radius, average over other dims (e.g., input_length)
            sweep = gdf.groupby(spectral_col, as_index=False)["mean_mc"].mean().sort_values(spectral_col)
            plt.plot(sweep[spectral_col].values, sweep["mean_mc"].values, marker="o", label=f"{group_col}={gval}")

        title = f"Mean MC vs {spectral_col}" + (f" ({fix_col}={fx})" if fx is not None else "")
        plt.title(title)
        plt.xlabel(spectral_col)
        plt.ylabel("Mean MC")
        plt.legend()
        plt.tight_layout()
        safe_fx = "" if fx is None else f"__{fix_col}={fx}".replace("/", "-").replace(" ", "")
        plt.savefig(outdir / f"line_spectral_sweeps{safe_fx}.png", dpi=180)
        plt.close()


def plot_time_tradeoff(agg: pd.DataFrame, outdir: Path):
    if "mean_time_sec" not in agg.columns or agg["mean_time_sec"].isna().all():
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(agg["mean_time_sec"], agg["mean_mc"])
    plt.xlabel("Mean time per task [s]")
    plt.ylabel("Mean MC")
    plt.title("Performance vs. Time")
    plt.tight_layout()
    plt.savefig(outdir / "time_vs_performance.png", dpi=180)
    plt.close()


def save_best_summary(best: pd.Series, outdir: Path):
    # Save text + JSON for easy consumption
    with open(outdir / "best_config.txt", "w") as f:
        f.write("Best configuration by mean MC (tie-break: std, time)\n")
        f.write("-" * 60 + "\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")
    # Also as JSON
    best.to_json(outdir / "best_config.json", indent=2)


def main():
    parser = argparse.ArgumentParser(description="Visualize ESN grid-search results and pick the best model.")
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory containing gnm_mc_results_*.csv (and optionally gnm_durations_*.csv).")
    parser.add_argument("--mc", type=str, default=None, help="Optional explicit path to MC CSV.")
    parser.add_argument("--dur", type=str, default=None, help="Optional explicit path to durations CSV.")
    parser.add_argument("--topk", type=int, default=15, help="Number of top configs to show in the bar chart.")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_mc, df_dur = load_results(results_dir, Path(args.mc) if args.mc else None, Path(args.dur) if args.dur else None)
    df_mc = expand_hparams(df_mc)
    df_mc = attach_timings_by_order(df_mc, df_dur)

    agg, hp_cols = aggregate_per_hparams(df_mc)

    print("\n=== Aggregate overview (top 10) ===")
    print(agg.head(10).to_string(index=False))

    best = pick_best_config(agg)
    print("\n=== Best configuration ===")
    print(best.to_string())
    save_best_summary(best, figures_dir)

    # Plots
    plot_topk_bar(agg, hp_cols, figures_dir, k=args.topk)

    # Heatmaps: spectral_radius vs input_length (or train_len fallback), stratified by scaling & reg if present
    x_col = "input_length" if "input_length" in agg.columns and agg["input_length"].notna().any() else "train_len"
    stratify = [c for c in ["input_scaling", "regularization_method"] if c in agg.columns and agg[c].notna().any()]
    plot_heatmaps(agg, figures_dir, spectral_col="spectral_radius", x_col=x_col, stratify_cols=stratify)

    # Lines: mean MC vs spectral radius for different input_scalings (and reg facets)
    if "input_scaling" in agg.columns and agg["input_scaling"].notna().any():
        plot_lines_spectral_sweeps(agg, figures_dir,
                                   spectral_col="spectral_radius",
                                   group_col="input_scaling",
                                   fix_col="regularization_method")

    # Optional: time-performance tradeoff scatter
    plot_time_tradeoff(agg, figures_dir)

    # Export aggregates for convenience
    agg.to_csv(figures_dir / "aggregated_performance.csv", index=False)
    print(f"\nSaved figures and summaries to: {figures_dir.resolve()}")

def plot_grid_heatmaps_shared_colorbar(
    agg: pd.DataFrame,
    outdir: Path,
    reg_col: str = "regularization_method",
    x_col: str = "train_len",         # change to "input_length" anytime
    y_col: str = "input_scaling",     # swap to anything present in agg
    value_col: str = "mean_mc",
    reduce_over: list[str] | None = None,  # average over other dims (e.g., spectral_radius)
    fig_name: str = "heatmaps_by_reg_shared_cb.png",
):
    # sanity checks
    if reg_col not in agg.columns or agg[reg_col].isna().all():
        print(f"[heatmaps] Column '{reg_col}' not found or empty — skipping.")
        return
    for c in (x_col, y_col, value_col):
        if c not in agg.columns:
            print(f"[heatmaps] Column '{c}' not found — skipping.")
            return

    # if x_col missing but 'input_length' exists, fallback automatically
    if x_col not in agg.columns and "input_length" in agg.columns:
        x_col = "input_length"

    # build a reduced table: mean over unspecified columns
    if reduce_over is None:
        # everything except these four will be reduced away
        fixed = {reg_col, x_col, y_col}
        reduce_over = [c for c in agg.columns if c not in fixed and c != value_col]

    grp_cols = [reg_col, y_col, x_col]
    df_red = agg.groupby(grp_cols, dropna=False, as_index=False)[value_col].mean()

    # collect pivots per reg
    reg_values = [r for r in df_red[reg_col].dropna().unique()]
    if not reg_values:
        print("[heatmaps] No regularization values found — skipping.")
        return

    # consistent x/y ordering
    def _sorted_unique(series):
        try:
            vals = pd.to_numeric(series, errors="coerce")
            if vals.notna().all():
                return sorted(series.unique(), key=lambda v: float(v))
        except Exception:
            pass
        return sorted(series.unique(), key=lambda v: str(v))

    x_all = _sorted_unique(df_red[x_col])
    y_all = _sorted_unique(df_red[y_col])

    # compute global vmin/vmax for shared colorbar
    vmin = df_red[value_col].min()
    vmax = df_red[value_col].max()

    # layout
    n = len(reg_values)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    last_im = None

    for i, reg in enumerate(reg_values):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub = df_red[df_red[reg_col] == reg]

        # make complete grid with NaNs for missing combos
        pivot = sub.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
        # reindex to ensure consistent axes across subplots
        pivot = pivot.reindex(index=y_all, columns=x_all)

        im = ax.imshow(pivot.values, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        last_im = im  # keep reference for colorbar

        # ticks/labels
        ax.set_xticks(np.arange(len(x_all)))
        ax.set_xticklabels([str(int(x)) if pd.api.types.is_numeric_dtype(pd.Series(x_all)) else str(x) for x in x_all],
                           rotation=45, ha="right")
        ax.set_yticks(np.arange(len(y_all)))
        ax.set_yticklabels([f"{y:g}" if isinstance(y, (int, float, np.floating)) else str(y) for y in y_all])

        ax.set_title(f"{reg_col} = {reg}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    # hide any empty axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    # shared colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.98)
        cbar.set_label(value_col)

    fig.tight_layout()
    fig.savefig(outdir / fig_name, dpi=180)
    print("HERE")
    plt.close(fig)


# --- ADD: programmatic entrypoint (no argparse needed) ---
def run_visualization(dir_path, mc: str | None = None, dur: str | None = None, topk: int = 15):
    results_dir = Path(dir_path)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_mc, df_dur = load_results(results_dir, Path(mc) if mc else None, Path(dur) if dur else None)
    df_mc = expand_hparams(df_mc)
    df_mc = attach_timings_by_order(df_mc, df_dur)

    agg, hp_cols = aggregate_per_hparams(df_mc)
    best = pick_best_config(agg)
    save_best_summary(best, figures_dir)

    # Plots
    plot_topk_bar(agg, hp_cols, figures_dir, k=topk)
    x_col = "input_length" if ("input_length" in agg.columns and agg["input_length"].notna().any()) else "train_len"
    stratify = [c for c in ["input_scaling", "regularization_method"] if c in agg.columns and agg[c].notna().any()]
    plot_heatmaps(agg, figures_dir, spectral_col="spectral_radius", x_col=x_col, stratify_cols=stratify)
    if "input_scaling" in agg.columns and agg["input_scaling"].notna().any():
        plot_lines_spectral_sweeps(agg, figures_dir,
                                   spectral_col="spectral_radius",
                                   group_col="input_scaling",
                                   fix_col="regularization_method")
        
    plot_time_tradeoff(agg, figures_dir)

    # One subplot per regularization method, shared colorbar.
    # Choose axes here; switch to x_col="input_length" anytime.
    print("[visualize] Plotting heatmaps by regularization method...")
    x_axis = "train_len" if ("train_len" in agg.columns and agg["train_len"].notna().any()) else "input_length"
    plot_grid_heatmaps_shared_colorbar(
        agg,
        outdir=figures_dir,
        reg_col="regularization_method",
        x_col=x_axis,
        y_col="input_scaling",
        value_col="mean_mc",
        fig_name="heatmaps_by_reg_shared_cb.png",
    )

    # Export aggregates
    agg.to_csv(figures_dir / "aggregated_performance.csv", index=False)
    print(f"[visualize] Best config:\n{best.to_string()}")
    print(f"[visualize] Saved figures to: {figures_dir.resolve()}")
    return {"agg": agg, "best": best, "figures_dir": figures_dir}


if __name__ == "__main__":
    # Set args here. 
    # dir_path = "/Users/adrian/Documents/01_projects/14_4D_lab/output/02_gnm_estimation/esn_grid_resolution68_2025-08-14_17-32-20" # /gnm_mc_results_2025-08-14_17-32-20.csv
    dir_path = "/Users/adrian/Documents/01_projects/14_4D_lab/output/02_gnm_estimation/esn_grid_resolution68_2025-08-15_01-22-16"
    # main()
    dict_results = run_visualization(dir_path)
    print(f"Results: {dict_results}")
