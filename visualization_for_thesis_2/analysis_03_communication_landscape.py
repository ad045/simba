"""
analysis_03_communication_landscape.py
========================================
Focuses on the five core communication / efficiency metrics and maps their
joint behaviour across the GNM η–γ parameter space.

Analyses
--------
A. Individual η–γ heatmaps for each communication metric (mean over 10 seeds)
B. Pairwise correlation of communication metrics inside GNM space vs inside
   empirical datasets (MaMI, Lexi developing) — reveals whether GNM reproduces
   the same co-variation structure as real brains
C. Scatter: Global Efficiency vs Wiring Cost, coloured by η (columns) and γ (rows)
D. Efficiency–communication triangle: three efficiency metrics in 2-D UMAP /
   scatter with η–γ gradient colouring
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analysis_config import (
    DATA_PKL, OUTPUT_DIR, FEATURE_CATEGORY, CATEGORY_COLORS,
    label, load_all, gnm_pivot, cm_to_inch,
)

# ── Communication feature set ─────────────────────────────────────────────────
COMM_FEATURES = [
    "global_efficiency",
    "diffusion_efficiency",
    "propagation_efficiency",
    "char_path_length",
    "algebraic_connectivity_fiedler_value",
    "avg_communicability",
    "wiring_cost",
    "proportion_long_range_connections_0.3956",
]

# colour per metric
COMM_COLORS = {
    "global_efficiency":                            "#3FA5C4",
    "diffusion_efficiency":                         "#394D73",
    "propagation_efficiency":                       "#5DC400",
    "char_path_length":                             "#E84653",
    "algebraic_connectivity_fiedler_value":         "#A6587C",
    "avg_communicability":                          "#44cfcf",
    "wiring_cost":                                  "#BF003F",
    "proportion_long_range_connections_0.3956":     "#E6B213",
}


def make_heatmap(pivot, ax, title, cmap="viridis", fig=None):
    """Draw a (η × γ) image. η on x-axis (left→right), γ on y-axis (bottom→top)."""
    arr = pivot.values
    eta_vals = pivot.index.values
    gam_vals = pivot.columns.values
    img = ax.imshow(
        arr.T,
        origin="lower", aspect="auto", cmap=cmap,
        extent=[eta_vals[0], eta_vals[-1], gam_vals[0], gam_vals[-1]],
        interpolation="nearest",
    )
    ax.set_xlabel(r"$\eta$", fontsize=8)
    ax.set_ylabel(r"$\gamma$", fontsize=8)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    if fig is not None:
        cb = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
    return img


def corr_triangle(ax, corr_mat, col_names, title, cmap="RdBu_r"):
    """Plot a lower-triangle correlation heatmap on *ax*."""
    n = corr_mat.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    sns.heatmap(corr_mat, ax=ax, mask=mask,
                cmap=cmap, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, linecolor="white",
                xticklabels=[label(c) for c in col_names],
                yticklabels=[label(c) for c in col_names],
                cbar=False)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6.5)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    all_data = load_all(DATA_PKL)
    gnm = all_data["hcp_schaefer_100_dataset_gnm"]

    # Available features
    avail = [f for f in COMM_FEATURES if f in gnm.columns]
    print(f"  Communication features available: {avail}")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel A — Individual heatmaps
    # ──────────────────────────────────────────────────────────────────────────
    print("\nPanel A: individual heatmaps …")

    ncols = 4
    nrows = (len(avail) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=cm_to_inch((7 * ncols, 6 * nrows)),
                              dpi=130)
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(avail):
        ax = axes[idx]
        pivot = gnm_pivot(gnm[["eta", "gamma", feat]].dropna(), feat, agg="mean")
        cmap = "viridis_r" if "cost" in feat or "path" in feat else "viridis"
        img = make_heatmap(pivot, ax, label(feat), cmap=cmap, fig=None)
        color = COMM_COLORS.get(feat, "black")
        ax.set_title(label(feat), fontsize=8, color=color, pad=3)
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

    for ax in axes[len(avail):]:
        ax.set_visible(False)

    fig.suptitle("Communication & Efficiency — mean over seeds per (η, γ) cell",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_03a_comm_heatmaps.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel B — Cross-dataset correlation comparison
    # ──────────────────────────────────────────────────────────────────────────
    print("\nPanel B: cross-dataset correlation comparison …")

    # GNM: average over seeds first to reduce size
    gnm_mean = gnm.groupby(["eta", "gamma"])[avail].mean().reset_index()

    datasets = {
        "GNM (all 25k)": gnm[avail].dropna(),
        "GNM (cell means)": gnm_mean[avail].dropna(),
    }
    for ds_name, ds_df in all_data.items():
        if "gnm" in ds_name:
            continue
        cols = [f for f in avail if f in ds_df.columns]
        if len(cols) >= 3:
            datasets[ds_name.replace("_", " ")] = ds_df[cols].dropna()

    n_ds = len(datasets)
    ncols_b = min(3, n_ds)
    nrows_b = (n_ds + ncols_b - 1) // ncols_b

    fig, axes = plt.subplots(nrows_b, ncols_b,
                              figsize=cm_to_inch((9 * ncols_b, 8 * nrows_b)),
                              dpi=120)
    axes = np.array(axes).flatten()

    for idx, (ds_name, df_sub) in enumerate(datasets.items()):
        ax = axes[idx]
        cols_here = [f for f in avail if f in df_sub.columns]
        corr = df_sub[cols_here].corr()
        corr_triangle(ax, corr, cols_here,
                      f"{ds_name}\n(n={len(df_sub)})")

    for ax in axes[n_ds:]:
        ax.set_visible(False)

    fig.suptitle("Pairwise correlation of communication metrics across datasets",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_03b_comm_cross_dataset_corr.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel C — Global Efficiency vs Wiring Cost, coloured by η and γ
    # ──────────────────────────────────────────────────────────────────────────
    print("\nPanel C: efficiency vs wiring cost scatter …")

    if "global_efficiency" in gnm.columns and "wiring_cost" in gnm.columns:
        sub = gnm[["eta", "gamma", "global_efficiency", "wiring_cost"]].dropna()
        eta_vals_sorted = np.sort(sub["eta"].unique())
        gam_vals_sorted = np.sort(sub["gamma"].unique())

        fig, axes = plt.subplots(1, 2, figsize=cm_to_inch((18, 8)), dpi=150)

        for ax, color_by, cmap, cbar_label in [
            (axes[0], "eta",   "coolwarm", r"$\eta$"),
            (axes[1], "gamma", "plasma",   r"$\gamma$"),
        ]:
            norm = Normalize(vmin=sub[color_by].min(), vmax=sub[color_by].max())
            sc = ax.scatter(
                sub["wiring_cost"], sub["global_efficiency"],
                c=sub[color_by], cmap=cmap, norm=norm,
                s=2, alpha=0.35, rasterized=True,
            )
            cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label(cbar_label, fontsize=9)
            cb.ax.tick_params(labelsize=7)
            ax.set_xlabel(label("wiring_cost"), fontsize=9)
            ax.set_ylabel(label("global_efficiency"), fontsize=9)
            ax.set_title(f"Coloured by {cbar_label}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(r"Global Efficiency vs Wiring Cost across $\eta$–$\gamma$ space",
                     fontsize=10, y=1.02)
        plt.tight_layout()
        out = OUTPUT_DIR / "gnm_03c_efficiency_vs_cost.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel D — Trade-off heatmap: efficiency ÷ cost
    # ──────────────────────────────────────────────────────────────────────────
    print("\nPanel D: efficiency/cost ratio heatmap …")

    if "global_efficiency" in gnm.columns and "wiring_cost" in gnm.columns:
        gnm_copy = gnm.copy()
        # Normalise cost to [0,1] to avoid division near zero artefacts
        wc_min, wc_max = gnm_copy["wiring_cost"].min(), gnm_copy["wiring_cost"].max()
        gnm_copy["wc_norm"] = (gnm_copy["wiring_cost"] - wc_min) / (wc_max - wc_min + 1e-12)
        gnm_copy["eff_cost_ratio"] = gnm_copy["global_efficiency"] / (gnm_copy["wc_norm"] + 0.01)

        mean_df = gnm_copy.groupby(["eta", "gamma"])[["global_efficiency",
                                                       "wiring_cost",
                                                       "eff_cost_ratio"]].mean()

        fig, axes = plt.subplots(1, 3, figsize=cm_to_inch((21, 7)), dpi=130)

        for ax, col, cmap, ttl in [
            (axes[0], "global_efficiency", "viridis",   label("global_efficiency")),
            (axes[1], "wiring_cost",       "viridis_r", label("wiring_cost")),
            (axes[2], "eff_cost_ratio",    "RdYlGn",    "Efficiency / Cost ratio"),
        ]:
            p = mean_df[col].reset_index().pivot(index="eta", columns="gamma", values=col)
            arr = p.values.T
            eta_v = p.index.values
            gam_v = p.columns.values
            img = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap,
                            extent=[eta_v[0], eta_v[-1], gam_v[0], gam_v[-1]],
                            interpolation="nearest")
            ax.set_xlabel(r"$\eta$", fontsize=8)
            ax.set_ylabel(r"$\gamma$", fontsize=8)
            ax.set_title(ttl, fontsize=9)
            ax.tick_params(labelsize=6)
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

        fig.suptitle("Global Efficiency, Wiring Cost, and their ratio across (η, γ)",
                     fontsize=10, y=1.02)
        plt.tight_layout()
        out = OUTPUT_DIR / "gnm_03d_efficiency_cost_ratio.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    print("\nDone — all outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()