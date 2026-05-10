"""
analysis_02_eta_gamma_sensitivity.py
=====================================
Two-way ANOVA variance decomposition for every network property across the
50 × 50 × 10 GNM parameter grid.

For each feature y (n = 25 000 observations):
  SS_total = Var(y) × (n-1)
  SS_η     = γ-levels × replicates × Σ_i (ȳ_η=i − ȳ)²
  SS_γ     = η-levels × replicates × Σ_j (ȳ_γ=j − ȳ)²
  SS_ηγ    = replicates × Σ_ij (ȳ_ij − ȳ_η=i − ȳ_γ=j + ȳ)²
  SS_err   = Σ_ijk (y_ijk − ȳ_ij)²

R²_x = SS_x / SS_total  (sums to 1)

Outputs
-------
1. gnm_02_anova_stacked_bar.pdf   — features sorted by η share, stacked R² bars
2. gnm_02_anova_scatter.pdf       — R²_η vs R²_γ scatter (coloured by category)
3. gnm_02_interaction_heatmap.pdf — R²_ηγ (interaction) per feature as bar
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analysis_config import (
    DATA_PKL, OUTPUT_DIR, GOOD_FEATURES, FEATURE_CATEGORY,
    CATEGORY_COLORS, label, load_gnm, cm_to_inch,
)


# ─────────────────────────────────────────────────────────────────────────────
# ANOVA decomposition
# ─────────────────────────────────────────────────────────────────────────────

def two_way_anova_r2(series_y, eta_labels, gamma_labels):
    """
    Compute R² components for a balanced two-way ANOVA.

    Parameters
    ----------
    series_y     : 1-D array of observations
    eta_labels   : 1-D array of η level for each observation
    gamma_labels : 1-D array of γ level for each observation

    Returns
    -------
    dict with keys: r2_eta, r2_gamma, r2_interaction, r2_residual
    """
    y = np.asarray(series_y, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 10:
        return dict(r2_eta=np.nan, r2_gamma=np.nan,
                    r2_interaction=np.nan, r2_residual=np.nan)

    y = y[mask]
    eta = np.asarray(eta_labels)[mask]
    gamma = np.asarray(gamma_labels)[mask]

    grand_mean = y.mean()
    n = len(y)
    SS_total = np.sum((y - grand_mean) ** 2)
    if SS_total < 1e-12:
        return dict(r2_eta=0.0, r2_gamma=0.0,
                    r2_interaction=0.0, r2_residual=1.0)

    # Unique levels
    eta_u = np.unique(eta)
    gam_u = np.unique(gamma)
    n_eta = len(eta_u)
    n_gam = len(gam_u)

    # Cell means (dictionary for quick lookup)
    cell_mean = {}
    for ei in eta_u:
        for gj in gam_u:
            mask_ij = (eta == ei) & (gamma == gj)
            cell_mean[(ei, gj)] = y[mask_ij].mean() if mask_ij.any() else grand_mean

    eta_mean = {ei: y[eta == ei].mean() for ei in eta_u}
    gam_mean = {gj: y[gamma == gj].mean() for gj in gam_u}

    # Build arrays aligned to y
    eta_mean_arr   = np.array([eta_mean[ei] for ei in eta])
    gam_mean_arr   = np.array([gam_mean[gj] for gj in gamma])
    cell_mean_arr  = np.array([cell_mean[(ei, gj)] for ei, gj in zip(eta, gamma)])

    SS_eta   = np.sum((eta_mean_arr  - grand_mean) ** 2)
    SS_gamma = np.sum((gam_mean_arr  - grand_mean) ** 2)
    SS_inter = np.sum((cell_mean_arr - eta_mean_arr - gam_mean_arr + grand_mean) ** 2)
    SS_resid = np.sum((y             - cell_mean_arr) ** 2)

    return dict(
        r2_eta         = SS_eta   / SS_total,
        r2_gamma       = SS_gamma / SS_total,
        r2_interaction = SS_inter / SS_total,
        r2_residual    = SS_resid / SS_total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading GNM data …")
    gnm = load_gnm(DATA_PKL)
    features = [f for f in GOOD_FEATURES if f in gnm.columns]
    eta_arr   = gnm["eta"].values
    gamma_arr = gnm["gamma"].values
    print(f"  {len(gnm)} rows, {len(features)} features")

    # ── decompose variance for every feature ──────────────────────────────────
    print("Running two-way ANOVA for each feature …")
    rows = []
    for feat in features:
        r2 = two_way_anova_r2(gnm[feat].values, eta_arr, gamma_arr)
        rows.append({"feature": feat, **r2})
        pct_explained = (r2["r2_eta"] + r2["r2_gamma"] + r2["r2_interaction"]) * 100
        print(f"  {label(feat):35s}  η={r2['r2_eta']:.3f}  γ={r2['r2_gamma']:.3f}"
              f"  ηγ={r2['r2_interaction']:.3f}  resid={r2['r2_residual']:.3f}"
              f"  (explained {pct_explained:.1f}%)")

    anova_df = pd.DataFrame(rows).set_index("feature")

    # Assign categories and colours
    anova_df["category"] = [FEATURE_CATEGORY.get(f, "Other") for f in anova_df.index]
    anova_df["cat_color"] = [CATEGORY_COLORS.get(FEATURE_CATEGORY.get(f, ""), "#555")
                              for f in anova_df.index]

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1 — Stacked bar chart (R² components, sorted by R²_η)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nPlot 1: stacked bar chart …")

    df_sorted = anova_df.dropna().sort_values("r2_eta", ascending=True)
    n = len(df_sorted)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=cm_to_inch((16, max(8, n * 0.45))), dpi=150)

    # Stacked bars: η | γ | ηγ | residual
    c_eta   = "#3FA5C4"   # LAKE_BLUE
    c_gam   = "#E84653"   # LECKER_RED
    c_inter = "#E6B213"   # YELLOW
    c_resid = "#CCCCCC"   # light gray

    left = np.zeros(n)
    for val_col, color, leg_label in [
        ("r2_eta",         c_eta,   r"$\eta$"),
        ("r2_gamma",       c_gam,   r"$\gamma$"),
        ("r2_interaction", c_inter, r"$\eta \times \gamma$"),
        ("r2_residual",    c_resid, "Residual"),
    ]:
        vals = df_sorted[val_col].values
        ax.barh(y_pos, vals, left=left, color=color, label=leg_label,
                height=0.75, edgecolor="none")
        left += vals

    # Feature labels coloured by category
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [label(f) for f in df_sorted.index],
        fontsize=6.5,
    )
    for tick, feat in zip(ax.get_yticklabels(), df_sorted.index):
        tick.set_color(CATEGORY_COLORS.get(FEATURE_CATEGORY.get(feat, ""), "#232324"))

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Fraction of total variance ($R^2$)", fontsize=9)
    ax.set_title(r"Two-way ANOVA: variance explained by $\eta$, $\gamma$, "
                 r"$\eta \times \gamma$, and residual seed noise",
                 fontsize=9)

    ax.axvline(0.5, color="black", lw=0.5, ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Category legend (right margin)
    cat_handles = [
        mpatches.Patch(color=col, label=cat)
        for cat, col in CATEGORY_COLORS.items()
        if any(FEATURE_CATEGORY.get(f) == cat for f in df_sorted.index)
    ]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=cat_handles, loc="upper right",
               fontsize=6.5, title="Category", title_fontsize=7,
               framealpha=0.85, borderpad=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_02_anova_stacked_bar.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2 — Scatter R²_η vs R²_γ
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot 2: R²_η vs R²_γ scatter …")

    df_sc = anova_df.dropna()
    fig, ax = plt.subplots(figsize=cm_to_inch((11, 10)), dpi=150)

    for cat, grp in df_sc.groupby("category"):
        color = CATEGORY_COLORS.get(cat, "#888")
        ax.scatter(grp["r2_eta"], grp["r2_gamma"],
                   color=color, s=40, alpha=0.85, label=cat, zorder=3)
        for feat, row in grp.iterrows():
            ax.annotate(label(feat), (row["r2_eta"], row["r2_gamma"]),
                        fontsize=5.5, ha="left", va="bottom",
                        color=color, alpha=0.9,
                        xytext=(2, 2), textcoords="offset points")

    # Diagonal: equal sensitivity
    lim = max(df_sc["r2_eta"].max(), df_sc["r2_gamma"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, label="Equal sensitivity")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"$R^2_\eta$  (variance explained by $\eta$)", fontsize=9)
    ax.set_ylabel(r"$R^2_\gamma$  (variance explained by $\gamma$)", fontsize=9)
    ax.set_title(r"Feature sensitivity to $\eta$ vs. $\gamma$", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_02_anova_scatter.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3 — Interaction and residual bar chart
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot 3: interaction / residual bar chart …")

    df_inter = anova_df.dropna().sort_values("r2_interaction", ascending=False)
    n_i = len(df_inter)

    fig, axes = plt.subplots(1, 2, figsize=cm_to_inch((16, max(8, n_i * 0.4))),
                              dpi=150, sharey=True)

    axes[0].barh(np.arange(n_i), df_inter["r2_interaction"].values,
                 color="#E6B213", height=0.75, edgecolor="none")
    axes[0].set_title(r"$R^2_{\eta \times \gamma}$ (interaction)", fontsize=9)
    axes[0].set_xlabel(r"$R^2$", fontsize=8)
    axes[0].set_yticks(np.arange(n_i))
    axes[0].set_yticklabels([label(f) for f in df_inter.index], fontsize=6.5)
    for tick, feat in zip(axes[0].get_yticklabels(), df_inter.index):
        tick.set_color(CATEGORY_COLORS.get(FEATURE_CATEGORY.get(feat, ""), "#232324"))

    df_resid_sorted = df_inter.sort_values("r2_residual", ascending=False)
    axes[1].barh(np.arange(n_i),
                 df_resid_sorted["r2_residual"].reindex(df_inter.index).values,
                 color="#CCCCCC", height=0.75, edgecolor="none")
    axes[1].set_title(r"$R^2_{\rm residual}$ (seed noise)", fontsize=9)
    axes[1].set_xlabel(r"$R^2$", fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(r"Interaction and residual (seed-noise) variance components",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_02_interaction_residual.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── Top η-driven features ──")
    for feat in anova_df.dropna().sort_values("r2_eta", ascending=False).head(5).index:
        r = anova_df.loc[feat]
        print(f"  {label(feat):35s}  R²_η={r.r2_eta:.3f}")

    print("── Top γ-driven features ──")
    for feat in anova_df.dropna().sort_values("r2_gamma", ascending=False).head(5).index:
        r = anova_df.loc[feat]
        print(f"  {label(feat):35s}  R²_γ={r.r2_gamma:.3f}")

    print("\nDone — all outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
