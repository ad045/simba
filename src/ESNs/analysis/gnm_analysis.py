# gnm_analysis.py
"""
Analyze GNM grid/best CSVs and produce energy landscape & timing plots.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from initialize_plots import apply_plot_style, save_figure, CMAP_ENERGY

# ------------------------
# Energy landscape (heatmap)
# ------------------------

def _to_energy_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (etas, gammas, energy_grid) sorted and gridded from long-form df."""
    if not {"eta", "gamma", "energy"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'eta', 'gamma', 'energy'.")
    etas = np.sort(df["eta"].unique())
    gammas = np.sort(df["gamma"].unique())
    grid_df = df.groupby(["gamma", "eta"], as_index=False)["energy"].mean()
    mat = (grid_df.pivot(index="gamma", columns="eta", values="energy")
                  .reindex(index=gammas, columns=etas)
                  .to_numpy())
    return etas, gammas, mat

def plot_energy_landscape(etas, gammas, energy_grid, df_best: Optional[pd.DataFrame], outpath_no_ext: str,
                          title: str = "Energy landscape") -> None:
    """Heatmap of energy with optional overlay of best (η, γ)."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(energy_grid, origin="lower", aspect="auto",
                   extent=[etas.min(), etas.max(), gammas.min(), gammas.max()],
                   cmap=CMAP_ENERGY)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Energy", rotation=270, labelpad=12)
    ax.set_xlabel("η")
    ax.set_ylabel("γ")
    ax.set_title(title)
    if df_best is not None and {"eta","gamma"}.issubset(df_best.columns):
        ax.scatter(df_best["eta"], df_best["gamma"], s=16, c="white", edgecolors="none", alpha=0.95, zorder=3)
    save_figure(fig, outpath_no_ext)
    plt.close(fig)

# ------------------------
# Distributions & scatter
# ------------------------

def plot_best_scatter(df_best: pd.DataFrame, outpath_no_ext: str) -> None:
    """Scatter of best (η, γ) per subject."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(df_best["eta"], df_best["gamma"], s=18)
    ax.set_xlabel("η")
    ax.set_ylabel("γ")
    ax.set_title("Best (η, γ) per subject")
    save_figure(fig, outpath_no_ext)
    plt.close(fig)

def plot_param_hist(df_best: pd.DataFrame, outdir: Path) -> None:
    """Histograms of best η and γ."""
    for col in ["eta", "gamma"]:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.hist(df_best[col], bins=30)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(f"Distribution of best {col}")
        save_figure(fig, str(outdir / f"best_{col}_hist"))
        plt.close(fig)

# ------------------------
# Timing plots
# ------------------------

def plot_timings(timing_partial_csv: Optional[Path], outdir: Path) -> None:
    """Plot per-subject stage timings if a partial timing CSV exists."""
    if not timing_partial_csv or not timing_partial_csv.exists():
        return
    tdf = pd.read_csv(timing_partial_csv)
    # Expect columns: subject, metrics_s, grow_s, esn_s, total_s
    cols = [c for c in ["subject","metrics_s","grow_s","esn_s","total_s"] if c in tdf.columns]
    if "subject" not in cols:
        return

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    tdf2 = tdf.sort_values("subject")
    for c in ["metrics_s","grow_s","esn_s","total_s"]:
        if c in tdf2.columns:
            ax.plot(tdf2["subject"], tdf2[c], marker=".", linestyle="-", label=c)
    ax.set_xlabel("subject")
    ax.set_ylabel("seconds")
    ax.set_title("Per-subject timings")
    ax.legend(frameon=False)
    save_figure(fig, str(outdir / "timings_per_subject"))
    plt.close(fig)

# ------------------------
# CLI
# ------------------------

def main_gnm(grid_csv: str, best_csv: str,
             timing_partial: str | None,
             out_dir: str, title: str = "") -> None:
    """Create energy heatmap + best-pairs plots + timing plots."""
    apply_plot_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_grid = pd.read_csv(grid_csv)
    df_best = pd.read_csv(best_csv)

    etas, gammas, M = _to_energy_grid(df_grid)
    plot_energy_landscape(etas, gammas, M, df_best, str(out / "energy_landscape"), title=title)
    plot_best_scatter(df_best, str(out / "best_scatter"))
    plot_param_hist(df_best, out)

    if timing_partial:
        plot_timings(Path(timing_partial), out)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True, help="Path to gnm_grid_results_*.csv")
    p.add_argument("--best", required=True, help="Path to gnm_best_results_*.csv")
    p.add_argument("--timings", default=None, help="(Optional) path to gnm_timing_partial_*.csv")
    p.add_argument("--out", required=True, help="Output directory for figures")
    p.add_argument("--title", default="")
    args = p.parse_args()
    main_gnm(args.grid, args.best, args.timings, args.out, args.title)
