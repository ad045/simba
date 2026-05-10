"""
analysis_01_seed_variability.py
================================
For each (η, γ) cell of the GNM parameter grid (50 × 50 = 2 500 cells,
each with 10 independent seeds), compute the coefficient of variation
(CV = std / |mean|) of every network property across seeds.

Outputs
-------
1. gnm_01_mean_cv_heatmap.pdf        — single heatmap: mean CV across all features
2. gnm_01_cv_per_category.pdf        — one heatmap per category (representative feature)
3. gnm_01_cv_all_features.pdf        — full grid: one panel per feature
4. gnm_01_cv_most_variable_eta.pdf   — 2 most variable features from eta + gamma direction
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from analysis_config import (
    DATA_PKL, OUTPUT_DIR, GOOD_FEATURES, HIGH_NAN_FEATURES,
    FEATURE_CATEGORY, CATEGORY_COLORS, label, load_gnm, gnm_pivot, cm_to_inch,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cv(series):
    """Coefficient of variation; returns NaN if |mean| ≈ 0."""
    m = series.mean()
    s = series.std(ddof=1)
    if np.isnan(m) or abs(m) < 1e-10:
        return np.nan
    return s / abs(m)


def make_heatmap(pivot: pd.DataFrame, ax, title: str, cmap="magma",
                 vmin=None, vmax=None, add_colorbar=True, fig=None):
    """Draw a (η × γ) heatmap on *ax*. η on x-axis, γ on y-axis."""
    # pivot: rows = eta (sorted), cols = gamma (sorted)
    arr = pivot.values          # shape (n_eta, n_gamma)
    eta_vals = pivot.index.values
    gam_vals = pivot.columns.values

    img = ax.imshow(
        arr.T,                  # transpose so gamma is on y-axis
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[eta_vals[0], eta_vals[-1], gam_vals[0], gam_vals[-1]],
        interpolation="nearest",
    )
    ax.set_xlabel(r"$\eta$", fontsize=8)
    ax.set_ylabel(r"$\gamma$", fontsize=8)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=6)

    if add_colorbar and fig is not None:
        cbar = fig.colorbar(img, ax=ax, pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=6)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading GNM data …")
    gnm = load_gnm(DATA_PKL)
    all_features = [f for f in GOOD_FEATURES + HIGH_NAN_FEATURES
                    if f in gnm.columns]
    print(f"  {len(gnm)} rows, {len(all_features)} features")

    # ── compute CV per (eta, gamma) cell ──────────────────────────────────────
    print("Computing CV per (η, γ) cell …")
    cv_records = []
    for (eta, gamma), grp in gnm.groupby(["eta", "gamma"]):
        rec = {"eta": eta, "gamma": gamma}
        for feat in all_features:
            rec[feat] = cv(grp[feat].dropna())
        cv_records.append(rec)

    cv_df = pd.DataFrame(cv_records)
    print(f"  CV dataframe: {cv_df.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1 — Mean CV heatmap (single panel)
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot 1: mean CV heatmap …")

    cv_df["mean_cv"] = cv_df[all_features].mean(axis=1)
    pivot_mean = cv_df.pivot(index="eta", columns="gamma", values="mean_cv")

    fig, ax = plt.subplots(figsize=cm_to_inch((10, 8)), dpi=150)
    img = make_heatmap(pivot_mean, ax, "Mean CV across all features", fig=fig)
    fig.colorbar(img, ax=ax, label="Mean CV", fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_01_mean_cv_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2 — Representative feature per category
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot 2: per-category representative CV heatmaps …")

    # Pick one representative from each category (first available)
    categories = list(dict.fromkeys(FEATURE_CATEGORY[f]
                                     for f in all_features
                                     if f in FEATURE_CATEGORY))
    representatives = {}
    for cat in categories:
        for feat in all_features:
            if FEATURE_CATEGORY.get(feat) == cat and not cv_df[feat].isna().all():
                representatives[cat] = feat
                break

    n_cats = len(representatives)
    ncols = 4
    nrows = (n_cats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=cm_to_inch((7 * ncols, 6 * nrows)),
                              dpi=120)
    axes = np.array(axes).flatten()

    for idx, (cat, feat) in enumerate(representatives.items()):
        ax = axes[idx]
        pivot = cv_df.pivot(index="eta", columns="gamma", values=feat)
        color = CATEGORY_COLORS.get(cat, "viridis")
        make_heatmap(pivot, ax,
                     f"{cat}\n({label(feat)})",
                     cmap="magma", fig=fig)
        ax.set_title(f"{cat}\n({label(feat)})", fontsize=7,
                     color=CATEGORY_COLORS.get(cat, "black"), pad=3)

    for ax in axes[n_cats:]:
        ax.set_visible(False)

    fig.suptitle("Coefficient of Variation (seed variability) — representative per category",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_01_cv_per_category.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3 — Full grid: CV for every feature
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot 3: full feature-by-feature CV grid …")

    ncols = 5
    nrows = (len(all_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=cm_to_inch((6 * ncols, 5 * nrows)),
                              dpi=100)
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(all_features):
        ax = axes[idx]
        pivot = cv_df.pivot(index="eta", columns="gamma", values=feat)
        make_heatmap(pivot, ax, label(feat), cmap="magma", fig=None,
                     add_colorbar=False)
        cat = FEATURE_CATEGORY.get(feat, "")
        ax.set_title(label(feat), fontsize=6,
                     color=CATEGORY_COLORS.get(cat, "black"), pad=2)

    for ax in axes[len(all_features):]:
        ax.set_visible(False)

    fig.suptitle("CV across 10 seeds per (η, γ) cell — all features",
                 fontsize=11, y=1.002)
    plt.tight_layout(h_pad=1.5, w_pad=0.8)
    out = OUTPUT_DIR / "gnm_01_cv_all_features.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 4 — Most variable features + their mean profiles
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot 4: top-N most variable features …")

    # Sort features by their mean CV across all cells
    mean_cv_per_feat = {
        f: cv_df[f].mean() for f in all_features if not cv_df[f].isna().all()
    }
    sorted_feats = sorted(mean_cv_per_feat, key=lambda x: -mean_cv_per_feat[x])
    top_n = min(8, len(sorted_feats))
    top_feats = sorted_feats[:top_n]

    fig, axes = plt.subplots(2, top_n,
                              figsize=cm_to_inch((6 * top_n, 10)),
                              dpi=120)

    for idx, feat in enumerate(top_feats):
        cat = FEATURE_CATEGORY.get(feat, "")

        # Top row: mean value heatmap
        pivot_val = gnm_pivot(gnm[["eta", "gamma", feat]].dropna(), feat, agg="mean")
        make_heatmap(pivot_val, axes[0, idx],
                     f"Mean: {label(feat)}", cmap="viridis", fig=None,
                     add_colorbar=False)
        axes[0, idx].set_title(f"Mean\n{label(feat)}", fontsize=7,
                               color=CATEGORY_COLORS.get(cat, "black"), pad=2)

        # Bottom row: CV heatmap
        pivot_cv = cv_df.pivot(index="eta", columns="gamma", values=feat)
        make_heatmap(pivot_cv, axes[1, idx],
                     f"CV: {label(feat)}", cmap="magma", fig=None,
                     add_colorbar=False)
        axes[1, idx].set_title(
            f"CV (mean={mean_cv_per_feat[feat]:.2f})", fontsize=7, pad=2
        )

    fig.suptitle("Top most variable features: mean value vs. seed variability (CV)",
                 fontsize=10, y=1.01)
    plt.tight_layout(h_pad=1.5, w_pad=0.8)
    out = OUTPUT_DIR / "gnm_01_most_variable_features.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\nSummary — mean CV per feature (top 10):")
    for feat in sorted_feats[:10]:
        print(f"  {label(feat):35s}  CV={mean_cv_per_feat[feat]:.4f}")

    # Where is max overall variability in parameter space?
    max_row = cv_df.loc[cv_df["mean_cv"].idxmax()]
    print(f"\nHighest mean CV at η={max_row.eta:.2f}, γ={max_row.gamma:.2f}  "
          f"(mean CV={max_row.mean_cv:.4f})")

    print("\nDone — all outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
