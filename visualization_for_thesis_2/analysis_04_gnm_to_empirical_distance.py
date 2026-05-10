"""
analysis_04_gnm_to_empirical_distance.py
==========================================
Measures how far each (η, γ) region of GNM network space is from the
empirical (biological) networks in a shared feature space.

Method
------
1. Pool all non-GNM datasets as "empirical" reference.
2. Identify features present in both GNM and every reference set.
3. Z-score all features jointly across the full pooled dataset so that
   the scale is shared.
4. For each (η, γ) cell (averaged over seeds) compute:
   a) Euclidean distance to the empirical centroid
   b) Mahalanobis distance (if covariance is well-conditioned)
5. Plot 50×50 distance maps; mark the minimum (best-fit parameter region).
6. Additional: show which individual features drive the distance.

Outputs
-------
gnm_04_distance_euclidean.pdf     — Euclidean distance heat-map + histogram
gnm_04_distance_mahalanobis.pdf   — Mahalanobis distance heat-map
gnm_04_distance_per_dataset.pdf   — Separate heat-map for MaMI and Lexi
gnm_04_feature_contribution.pdf   — Squared z-difference per feature at best/worst (η, γ)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import mahalanobis
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analysis_config import (
    DATA_PKL, OUTPUT_DIR, GOOD_FEATURES, FEATURE_CATEGORY,
    CATEGORY_COLORS, label, load_all, gnm_pivot, cm_to_inch,
)


def standardize_jointly(dfs: list[pd.DataFrame], cols: list[str]):
    """
    Z-score *cols* jointly (fit on all pooled rows, transform each df).
    Returns list of transformed DataFrames and (mean, std) used.
    """
    pooled = pd.concat([df[cols] for df in dfs], ignore_index=True)
    means = pooled.mean()
    stds  = pooled.std().replace(0, 1)
    return [(df[cols] - means) / stds for df in dfs], means, stds


def plot_distance_heatmap(pivot, ax, title, fig, best_eta=None, best_gamma=None,
                          cmap="magma_r"):
    arr = pivot.values
    eta_vals = pivot.index.values
    gam_vals = pivot.columns.values
    img = ax.imshow(
        arr.T, origin="lower", aspect="auto", cmap=cmap,
        extent=[eta_vals[0], eta_vals[-1], gam_vals[0], gam_vals[-1]],
        interpolation="nearest",
    )
    if best_eta is not None:
        ax.scatter([best_eta], [best_gamma],
                   marker="*", s=150, color="#FFD700", zorder=10,
                   label=f"Min: η={best_eta:.2f}, γ={best_gamma:.2f}")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel(r"$\eta$", fontsize=8)
    ax.set_ylabel(r"$\gamma$", fontsize=8)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Distance", fontsize=7)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    all_data = load_all(DATA_PKL)
    gnm = all_data["hcp_schaefer_100_dataset_gnm"]

    # ── collect empirical datasets ────────────────────────────────────────────
    empirical = {k: v for k, v in all_data.items() if "gnm" not in k}
    print(f"Empirical datasets: {list(empirical.keys())}")
    for k, v in empirical.items():
        print(f"  {k}: {v.shape}")

    # ── find shared feature columns ───────────────────────────────────────────
    all_dfs = [gnm] + list(empirical.values())
    # Intersect GOOD_FEATURES with what exists in ALL datasets
    shared_cols = [
        f for f in GOOD_FEATURES
        if all(f in df.columns for df in all_dfs)
        and gnm[f].isna().mean() < 0.1
    ]
    print(f"\nShared features (present in all datasets, <10% NaN in GNM): {len(shared_cols)}")

    if len(shared_cols) < 3:
        print("WARNING: too few shared features, falling back to GNM+MaMI only")
        mami = empirical.get("suarez_MaMI_dataset")
        if mami is not None:
            shared_cols = [
                f for f in GOOD_FEATURES
                if f in gnm.columns and f in mami.columns
                and gnm[f].isna().mean() < 0.1
            ]
        print(f"  Fallback shared features: {len(shared_cols)}")

    # ── cell-level GNM means ──────────────────────────────────────────────────
    gnm_cell = gnm.groupby(["eta", "gamma"])[shared_cols].mean().reset_index()
    print(f"\nGNM cell means: {gnm_cell.shape}")

    # ── joint z-scoring ───────────────────────────────────────────────────────
    emp_combined = pd.concat(
        [df[shared_cols].dropna() for df in empirical.values()],
        ignore_index=True,
    )
    [gnm_scaled, emp_scaled], means, stds = standardize_jointly(
        [gnm_cell[shared_cols].fillna(0), emp_combined],
        shared_cols,
    )
    gnm_arr = gnm_scaled.values          # shape (2500, n_features)
    emp_arr = emp_scaled.values           # shape (n_emp, n_features)

    emp_centroid = emp_arr.mean(axis=0)   # shape (n_features,)

    # ─────────────────────────────────────────────────────────────────────────
    # Euclidean distance to empirical centroid
    # ─────────────────────────────────────────────────────────────────────────
    print("Computing Euclidean distances …")
    euc_dists = np.linalg.norm(gnm_arr - emp_centroid, axis=1)

    gnm_cell["dist_euclidean"] = euc_dists
    pivot_euc = gnm_cell.pivot(index="eta", columns="gamma", values="dist_euclidean")

    best_idx = np.argmin(euc_dists)
    best_eta   = gnm_cell.iloc[best_idx]["eta"]
    best_gamma = gnm_cell.iloc[best_idx]["gamma"]
    worst_idx  = np.argmax(euc_dists)
    worst_eta  = gnm_cell.iloc[worst_idx]["eta"]
    worst_gamma = gnm_cell.iloc[worst_idx]["gamma"]
    print(f"  Best  fit: η={best_eta:.2f}, γ={best_gamma:.2f}  (dist={euc_dists[best_idx]:.3f})")
    print(f"  Worst fit: η={worst_eta:.2f}, γ={worst_gamma:.2f}  (dist={euc_dists[worst_idx]:.3f})")

    # ─────────────────────────────────────────────────────────────────────────
    # Mahalanobis distance
    # ─────────────────────────────────────────────────────────────────────────
    print("Computing Mahalanobis distances …")
    cov_emp = np.cov(emp_arr.T)
    try:
        VI = np.linalg.inv(cov_emp + np.eye(cov_emp.shape[0]) * 1e-6)
        maha_dists = np.array([
            mahalanobis(gnm_arr[i], emp_centroid, VI)
            for i in range(len(gnm_arr))
        ])
        maha_ok = True
    except np.linalg.LinAlgError:
        print("  WARNING: covariance matrix singular; skipping Mahalanobis")
        maha_ok = False

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1 — Euclidean distance map
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=cm_to_inch((18, 8)), dpi=150)

    # Heatmap
    plot_distance_heatmap(pivot_euc, axes[0],
                          "Euclidean distance to empirical centroid",
                          fig, best_eta=best_eta, best_gamma=best_gamma)

    # Histogram
    axes[1].hist(euc_dists, bins=40, color="#3FA5C4", edgecolor="white", linewidth=0.4)
    axes[1].axvline(euc_dists[best_idx], color="#FFD700", lw=1.5,
                    label=f"Best (η={best_eta:.1f}, γ={best_gamma:.1f})")
    axes[1].axvline(euc_dists[worst_idx], color="#E84653", lw=1.5,
                    label=f"Worst (η={worst_eta:.1f}, γ={worst_gamma:.1f})")
    axes[1].set_xlabel("Euclidean distance", fontsize=9)
    axes[1].set_ylabel("Count (cells)", fontsize=9)
    axes[1].set_title("Distribution of distances to empirical centroid", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("GNM → Empirical distance in z-scored feature space",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_04_distance_euclidean.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2 — Mahalanobis distance map
    # ─────────────────────────────────────────────────────────────────────────
    if maha_ok:
        gnm_cell["dist_mahalanobis"] = maha_dists
        pivot_maha = gnm_cell.pivot(index="eta", columns="gamma",
                                    values="dist_mahalanobis")
        best_maha_idx = np.argmin(maha_dists)
        fig, ax = plt.subplots(figsize=cm_to_inch((10, 8)), dpi=150)
        plot_distance_heatmap(
            pivot_maha, ax,
            "Mahalanobis distance to empirical centroid",
            fig,
            best_eta=gnm_cell.iloc[best_maha_idx]["eta"],
            best_gamma=gnm_cell.iloc[best_maha_idx]["gamma"],
            cmap="magma_r",
        )
        fig.suptitle("Mahalanobis distance (accounts for feature covariance)",
                     fontsize=10, y=1.02)
        plt.tight_layout()
        out = OUTPUT_DIR / "gnm_04_distance_mahalanobis.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3 — Per-dataset distance maps
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot 3: per-dataset distance maps …")

    ds_results = {}
    for ds_name, ds_df in empirical.items():
        ds_cols = [f for f in shared_cols if f in ds_df.columns]
        if len(ds_cols) < 3:
            continue
        ds_z, _, _ = standardize_jointly(
            [gnm_cell[ds_cols].fillna(0), ds_df[ds_cols].dropna()],
            ds_cols,
        )
        gnm_ds_arr = ds_z[0].values
        emp_ds_arr = ds_z[1].values
        centroid   = emp_ds_arr.mean(axis=0)
        dists      = np.linalg.norm(gnm_ds_arr[:, :len(ds_cols)] - centroid, axis=1)
        gnm_cell[f"dist_{ds_name}"] = dists
        ds_results[ds_name] = dists

    n_ds = len(ds_results)
    if n_ds > 0:
        fig, axes = plt.subplots(1, n_ds, figsize=cm_to_inch((10 * n_ds, 8)), dpi=130)
        if n_ds == 1:
            axes = [axes]
        for ax, (ds_name, dists) in zip(axes, ds_results.items()):
            gnm_cell["_tmp"] = dists
            pivot_ds = gnm_cell.pivot(index="eta", columns="gamma", values="_tmp")
            best_ds_idx = np.argmin(dists)
            plot_distance_heatmap(
                pivot_ds, ax,
                f"Distance to {ds_name.replace('_', ' ')}",
                fig,
                best_eta=gnm_cell.iloc[best_ds_idx]["eta"],
                best_gamma=gnm_cell.iloc[best_ds_idx]["gamma"],
            )
        gnm_cell.drop(columns=["_tmp"], inplace=True, errors="ignore")
        fig.suptitle("Euclidean distance to individual reference datasets",
                     fontsize=10, y=1.02)
        plt.tight_layout()
        out = OUTPUT_DIR / "gnm_04_distance_per_dataset.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 4 — Feature contribution at best vs worst (η, γ)
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot 4: feature contributions at best/worst cells …")

    def feature_sq_diff(cell_idx):
        return (gnm_arr[cell_idx] - emp_centroid) ** 2

    sq_best  = feature_sq_diff(best_idx)
    sq_worst = feature_sq_diff(worst_idx)

    fig, axes = plt.subplots(1, 2, figsize=cm_to_inch((18, 10)), dpi=130,
                              sharey=True)
    y_pos = np.arange(len(shared_cols))

    for ax, sq_vals, title, color in [
        (axes[0], sq_best,  f"Best fit (η={best_eta:.2f}, γ={best_gamma:.2f})",  "#5DC400"),
        (axes[1], sq_worst, f"Worst fit (η={worst_eta:.2f}, γ={worst_gamma:.2f})", "#E84653"),
    ]:
        colors_bars = [CATEGORY_COLORS.get(FEATURE_CATEGORY.get(f, ""), "#aaa")
                       for f in shared_cols]
        ax.barh(y_pos, sq_vals, color=colors_bars, height=0.8, edgecolor="none")
        ax.set_title(title, fontsize=9, color=color)
        ax.set_xlabel(r"$(z_{\rm GNM} - z_{\rm emp})^2$", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(
        [label(f) for f in shared_cols], fontsize=6.5
    )
    for tick, feat in zip(axes[0].get_yticklabels(), shared_cols):
        tick.set_color(CATEGORY_COLORS.get(FEATURE_CATEGORY.get(feat, ""), "#232324"))

    # Legend for categories
    cat_handles = [
        mpatches.Patch(color=col, label=cat)
        for cat, col in CATEGORY_COLORS.items()
        if any(FEATURE_CATEGORY.get(f) == cat for f in shared_cols)
    ]
    axes[1].legend(handles=cat_handles, fontsize=6.5, title="Category",
                   title_fontsize=7, framealpha=0.9, loc="lower right")

    fig.suptitle("Per-feature squared z-distance to empirical centroid",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_04_feature_contribution.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    print("\nDone — all outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()