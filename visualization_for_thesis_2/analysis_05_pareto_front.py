"""
analysis_05_pareto_front.py
=============================
Identifies Pareto-optimal (η, γ) configurations for key trade-off pairs and
maps the resulting Pareto front relative to empirical brain networks.

Trade-off pairs examined (maximise efficiency, minimise cost / seed noise)
---------------------------------------------------------------------------
1. Global Efficiency ↑  vs  Wiring Cost ↓
2. Memory Capacity ↑    vs  Wiring Cost ↓
3. Robustness AUC ↑     vs  Wiring Cost ↓
4. Global Efficiency ↑  vs  Seed CV ↓      (reliability)

For each pair:
  a) Scatter of all 2 500 GNM cells coloured by Pareto rank
  b) 50 × 50 heatmap highlighting Pareto-optimal cells
  c) Empirical networks (MaMI, Lexi) overlaid in same objective space

Outputs
-------
gnm_05_pareto_<pair>.pdf      — one figure per pair
gnm_05_pareto_summary.pdf     — 2×2 summary grid
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analysis_config import (
    DATA_PKL, OUTPUT_DIR, GOOD_FEATURES, FEATURE_CATEGORY,
    CATEGORY_COLORS, label, load_all, gnm_pivot, cm_to_inch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pareto helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_pareto_efficient(costs):
    """
    Find Pareto-efficient rows (minimisation problem).
    *costs* : 2-D array (n_points, n_objectives), all objectives to MINIMISE.
    Returns boolean mask of length n_points.
    """
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if is_efficient[i]:
            # A point dominates i if it is ≤ in all objectives and < in at least one
            dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
            dominated[i] = False
            is_efficient[i] = not dominated.any()
    return is_efficient


def pareto_rank(costs):
    """
    Assign integer Pareto rank: rank 1 = non-dominated, rank 2 = non-dominated
    after removing rank-1 points, etc. (minimisation).
    """
    ranks = np.zeros(len(costs), dtype=int)
    remaining = np.ones(len(costs), dtype=bool)
    rank = 1
    while remaining.any():
        idx = np.where(remaining)[0]
        front = is_pareto_efficient(costs[idx])
        ranks[idx[front]] = rank
        remaining[idx[front]] = False
        rank += 1
    return ranks


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_pair(ax_scatter, ax_heatmap, fig,
                     cell_df, obj1_col, obj2_col,
                     obj1_label, obj2_label,
                     maximize_obj1, maximize_obj2,
                     emp_dfs=None, emp_names=None,
                     emp_colors=None):
    """
    Draw scatter (Pareto-rank coloured) and heatmap for one objective pair.

    Parameters
    ----------
    maximize_obj1/2 : bool — if True, negate before computing Pareto (convert to min)
    """
    sub = cell_df[[obj1_col, obj2_col, "eta", "gamma"]].dropna()
    if len(sub) < 10:
        ax_scatter.set_title("Insufficient data", fontsize=8)
        return

    c1 = sub[obj1_col].values * (-1 if maximize_obj1 else 1)
    c2 = sub[obj2_col].values * (-1 if maximize_obj2 else 1)
    costs = np.column_stack([c1, c2])
    ranks = pareto_rank(costs)

    # ── scatter ──
    cmap_rank = matplotlib.colormaps["YlOrRd_r"].resampled(min(int(ranks.max()), 6))
    sc = ax_scatter.scatter(
        sub[obj1_col], sub[obj2_col],
        c=np.clip(ranks, 1, 6), cmap=cmap_rank, vmin=1, vmax=6,
        s=12, alpha=0.7, zorder=3, rasterized=True,
    )
    # Pareto front (rank 1)
    front_mask = ranks == 1
    sorted_front = sub[front_mask].sort_values(obj1_col)
    ax_scatter.plot(sorted_front[obj1_col], sorted_front[obj2_col],
                    "k-", lw=1.5, zorder=5, alpha=0.8)
    ax_scatter.scatter(sub.loc[front_mask, obj1_col],
                       sub.loc[front_mask, obj2_col],
                       color="black", s=25, zorder=6, label="Pareto front")

    # Empirical networks overlay
    if emp_dfs is not None:
        for df_emp, ds_name, ds_col in zip(emp_dfs, emp_names, emp_colors):
            cols_ok = [c for c in [obj1_col, obj2_col] if c in df_emp.columns]
            if len(cols_ok) < 2:
                continue
            ax_scatter.scatter(
                df_emp[obj1_col].dropna(), df_emp[obj2_col].dropna(),
                marker="D", s=30, color=ds_col, zorder=7,
                alpha=0.8, label=ds_name, edgecolors="white", linewidths=0.4,
            )

    ax_scatter.set_xlabel(obj1_label, fontsize=8)
    ax_scatter.set_ylabel(obj2_label, fontsize=8)
    ax_scatter.tick_params(labelsize=6)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)
    cb = fig.colorbar(sc, ax=ax_scatter, pad=0.02, fraction=0.046)
    cb.set_label("Pareto rank", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax_scatter.legend(fontsize=6.5, loc="best", framealpha=0.9)

    # ── heatmap ──
    sub2 = sub.copy()
    sub2["rank"] = ranks
    sub2["is_front"] = (ranks == 1).astype(float)

    # Background: objective 1 value
    pivot_obj1 = sub2.pivot(index="eta", columns="gamma", values=obj1_col)
    eta_v = pivot_obj1.index.values
    gam_v = pivot_obj1.columns.values
    img = ax_heatmap.imshow(
        pivot_obj1.values.T,
        origin="lower", aspect="auto", cmap="viridis",
        extent=[eta_v[0], eta_v[-1], gam_v[0], gam_v[-1]],
        interpolation="nearest",
    )
    # Pareto-front cells as contour overlay
    pivot_front = sub2.pivot(index="eta", columns="gamma", values="is_front")
    ax_heatmap.contour(
        np.linspace(eta_v[0], eta_v[-1], len(eta_v)),
        np.linspace(gam_v[0], gam_v[-1], len(gam_v)),
        pivot_front.values.T,
        levels=[0.5], colors="gold", linewidths=2.5,
    )

    # Mark empirical eta/gamma if available
    # (not applicable to MaMI which has no eta/gamma, but kept for generality)

    ax_heatmap.set_xlabel(r"$\eta$", fontsize=8)
    ax_heatmap.set_ylabel(r"$\gamma$", fontsize=8)
    ax_heatmap.set_title(f"Pareto front (gold) over {obj1_label}", fontsize=8)
    ax_heatmap.tick_params(labelsize=6)
    fig.colorbar(img, ax=ax_heatmap, fraction=0.046, pad=0.03,
                 label=obj1_label).ax.tick_params(labelsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    all_data = load_all(DATA_PKL)
    gnm = all_data["hcp_schaefer_100_dataset_gnm"]

    empirical = {k: v for k, v in all_data.items() if "gnm" not in k}
    emp_color_map = {
        "suarez_MaMI_dataset": "#799372",
        "lexis_data_developing": "#C74800",
        "kaysons_generated_networks_diffusion": "#F7BE18",
        "kaysons_generated_networks_propagation": "#262F3F",
        "kaysons_generated_networks_routing": "#6E3AA3",
    }

    # ── cell-level means ──────────────────────────────────────────────────────
    all_feats = list({f for f in GOOD_FEATURES if f in gnm.columns})
    gnm_cell = gnm.groupby(["eta", "gamma"])[all_feats].mean().reset_index()
    print(f"GNM cell means: {gnm_cell.shape}")

    # ── compute per-cell CV for 'reliability' objective ───────────────────────
    def cell_cv(col):
        def cv(s):
            m = s.mean(); return s.std(ddof=1) / abs(m) if abs(m) > 1e-10 else np.nan
        return gnm.groupby(["eta", "gamma"])[col].apply(cv).reset_index()

    if "global_efficiency" in gnm.columns:
        cv_eff = cell_cv("global_efficiency")
        cv_eff.columns = ["eta", "gamma", "cv_global_efficiency"]
        gnm_cell = gnm_cell.merge(cv_eff, on=["eta", "gamma"], how="left")

    # ── define trade-off pairs ────────────────────────────────────────────────
    # (obj1_col, obj2_col, label1, label2, max1, max2, title)
    pairs = []

    if "global_efficiency" in gnm_cell.columns and "wiring_cost" in gnm_cell.columns:
        pairs.append((
            "global_efficiency", "wiring_cost",
            label("global_efficiency"), label("wiring_cost"),
            True, False,   # maximise efficiency, minimise cost
            "Global Efficiency ↑ vs Wiring Cost ↓",
        ))

    if "mc_input_scaling_0_1_mc_mean" in gnm_cell.columns and "wiring_cost" in gnm_cell.columns:
        pairs.append((
            "mc_input_scaling_0_1_mc_mean", "wiring_cost",
            label("mc_input_scaling_0_1_mc_mean"), label("wiring_cost"),
            True, False,
            "Memory Capacity ↑ vs Wiring Cost ↓",
        ))

    if "targeted_attack_robustness_rob_targeted_auc" in gnm_cell.columns \
            and "wiring_cost" in gnm_cell.columns:
        pairs.append((
            "targeted_attack_robustness_rob_targeted_auc", "wiring_cost",
            label("targeted_attack_robustness_rob_targeted_auc"), label("wiring_cost"),
            True, False,
            "Robustness ↑ vs Wiring Cost ↓",
        ))

    if "global_efficiency" in gnm_cell.columns and "cv_global_efficiency" in gnm_cell.columns:
        pairs.append((
            "global_efficiency", "cv_global_efficiency",
            label("global_efficiency"), "Seed CV (reliability)",
            True, False,   # maximise efficiency, minimise variability
            "Global Efficiency ↑ vs Seed Variability ↓",
        ))

    print(f"\nTrade-off pairs to analyse: {len(pairs)}")

    # Empirical lists
    emp_dfs    = list(empirical.values())
    emp_names  = [k.replace("_", " ") for k in empirical.keys()]
    emp_colors = [emp_color_map.get(k, "#888888") for k in empirical.keys()]

    # ─────────────────────────────────────────────────────────────────────────
    # Individual figures per pair
    # ─────────────────────────────────────────────────────────────────────────
    for obj1_col, obj2_col, lab1, lab2, max1, max2, ttl in pairs:
        print(f"\nProcessing: {ttl} …")

        fig, axes = plt.subplots(1, 2, figsize=cm_to_inch((20, 9)), dpi=150)

        plot_pareto_pair(
            axes[0], axes[1], fig,
            gnm_cell, obj1_col, obj2_col,
            lab1, lab2, max1, max2,
            emp_dfs=emp_dfs, emp_names=emp_names, emp_colors=emp_colors,
        )
        axes[0].set_title(f"Pareto scatter\n{ttl}", fontsize=8)

        fig.suptitle(ttl, fontsize=11, y=1.02)
        plt.tight_layout()
        safe_name = ttl.lower().replace(" ", "_").replace("↑", "up").replace("↓", "dn")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        out = OUTPUT_DIR / f"gnm_05_pareto_{safe_name}.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary grid (2×2)
    # ─────────────────────────────────────────────────────────────────────────
    if len(pairs) >= 2:
        print("\nPlotting summary grid …")
        nrows, ncols = 2, len(pairs)
        fig = plt.figure(figsize=cm_to_inch((10 * ncols, 9 * nrows)), dpi=120)
        gs = plt.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.35)

        for col_i, (obj1_col, obj2_col, lab1, lab2, max1, max2, ttl) in enumerate(pairs):
            ax_sc   = fig.add_subplot(gs[0, col_i])
            ax_heat = fig.add_subplot(gs[1, col_i])
            plot_pareto_pair(
                ax_sc, ax_heat, fig,
                gnm_cell, obj1_col, obj2_col,
                lab1, lab2, max1, max2,
                emp_dfs=emp_dfs, emp_names=emp_names, emp_colors=emp_colors,
            )
            ax_sc.set_title(ttl, fontsize=7)

        fig.suptitle("Pareto trade-off analysis — GNM parameter space", fontsize=12)
        out = OUTPUT_DIR / "gnm_05_pareto_summary.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Print summary: how many (η, γ) cells are Pareto-optimal in each pair
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Pareto front sizes ──")
    for obj1_col, obj2_col, lab1, lab2, max1, max2, ttl in pairs:
        sub = gnm_cell[[obj1_col, obj2_col]].dropna()
        c1 = sub[obj1_col].values * (-1 if max1 else 1)
        c2 = sub[obj2_col].values * (-1 if max2 else 1)
        front = is_pareto_efficient(np.column_stack([c1, c2]))
        pct = front.sum() / len(front) * 100
        print(f"  {ttl}: {front.sum()} / {len(front)} cells on front ({pct:.1f}%)")

    print("\nDone — all outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()