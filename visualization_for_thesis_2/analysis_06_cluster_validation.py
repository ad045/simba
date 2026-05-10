"""
analysis_06_cluster_validation.py
===================================
Answers two questions:

  Q1. Are the manual category labels in `remaining_categories` justified by
      the data's correlation structure?

  Q2. Which features need their sign flipped so that every axis of a spider
      / radar plot points in a consistent conceptual direction?

Outputs (all to OUTPUT_DIR)
---------------------------
gnm_06a_corr_heatmap_manual.pdf
    Correlation matrix of all 30 features, rows/cols grouped by your manual
    categories (coloured sidebar). Shows which features are tightly coupled
    within categories vs across them.

gnm_06b_dendrogram_vs_manual.pdf
    Hierarchical clustering dendrogram (Ward linkage, distance = 1 - |r|)
    with manual-category colour bars on the leaves.  Visual answer to
    "does the tree agree with my groupings?"

gnm_06c_silhouette_sweep.pdf
    Silhouette score for k = 2 … 15 data-driven clusters.
    The dashed vertical line marks your current manual k = 11.

gnm_06d_cluster_comparison.pdf
    Side-by-side: your manual grouping vs the data-driven grouping at the
    optimal k (and at k = 11 for a fair comparison).  Columns = features,
    rows = cluster labels.  Colour = category assignment.

gnm_06e_within_between_ratio.pdf
    Per-category bar chart: mean |r| within category vs mean |r| to all
    other features.  A ratio > 1 means the grouping is coherent.

gnm_06f_sign_flip_audit.pdf
    For each manual category: shows every feature's loading on the category's
    PC1.  Features with negative loading are highlighted — those are the ones
    that need their sign flipped before computing a category score.
    Also prints the recommended flip dictionary to stdout.

Console also prints:
  - overall cluster quality numbers
  - the recommended `FLIP_SIGNS` dict to paste into your code
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

sys.path.insert(0, str(Path(__file__).parent))
from analysis_config import DATA_PKL, OUTPUT_DIR, load_all, cm_to_inch

# ── pull the canonical feature set from config ────────────────────────────────
from config import remaining_categories

MANUAL_CATS = remaining_categories      # OrderedDict str → list[str]
ALL_FEATS   = [f for feats in MANUAL_CATS.values() for f in feats]

# Short display names (strip long prefixes)
def short(col):
    for prefix in [
        "repertoire_sweep_weighted_by_distances_",
        "algebraic_connectivity_",
        "synchronizability_eigenratio_",
        "targeted_attack_robustness_",
        "community_synchronization_vulnerability_",
        "participation_coefficient_",
        "ollivier_ricci_curvature_",
        "rich_club_coefficient_rc_",
        "persistent_homology_ph_",
        "mc_input_scaling_0_1_",
        "mc_nonlinear_input_scaling_0_1_",
        "nct_control_",
        "nct_energies_",
    ]:
        if col.startswith(prefix):
            return col[len(prefix):]
    return col

# Short category labels
CAT_SHORT = {
    "Fundamental Topology":                                          "Topology",
    "Paths, Efficiency & Communication":                             "Efficiency",
    "Spatial Embedding & Wiring Cost":                               "Wiring",
    "Rich-Club Organization":                                        "Rich-Club",
    "Spectral Properties, Algebraic Connectivity & Fiedler Analysis":"Spectral",
    "Metastability & Kuramoto Synchronization":                      "Metastab.",
    "Participation Coefficient (Louvain)":                           "Participation",
    "Ollivier-Ricci Curvature":                                      "Curvature",
    "Targeted Attack Robustness & Community Structure & Vulnerability":"Robustness",
    "Network Control Theory - Average Controllability":              "NCT",
    "Memory Capacity":                                               "Memory",
}

# Palette for manual categories
_palette = [
    "#394D73","#3FA5C4","#E84653","#E6B213","#A6587C",
    "#5DC400","#F99465","#44cfcf","#BF003F","#6A7870","#FFD700",
]
CAT_COLORS = {cat: _palette[i % len(_palette)]
              for i, cat in enumerate(MANUAL_CATS.keys())}

FEAT_TO_CAT  = {f: cat for cat, feats in MANUAL_CATS.items() for f in feats}
FEAT_TO_CATSHORT = {f: CAT_SHORT.get(cat, cat) for f, cat in FEAT_TO_CAT.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Load data — pool all datasets for robust correlation estimates
# ─────────────────────────────────────────────────────────────────────────────

def load_pooled():
    all_data = load_all(DATA_PKL)

    # GNM: sample 2500 cell-level means (avoid over-weighting the 25k grid)
    gnm = all_data["hcp_schaefer_100_dataset_gnm"]
    gnm_mean = gnm.groupby(["eta", "gamma"])[ALL_FEATS].mean().reset_index()[ALL_FEATS]

    # Other datasets
    others = [
        df[ALL_FEATS].dropna()
        for k, df in all_data.items() if "gnm" not in k and len(df) > 1
    ]

    pooled = pd.concat([gnm_mean] + others, ignore_index=True).dropna()
    print(f"Pooled data: {pooled.shape}  "
          f"(gnm cell means={len(gnm_mean)}, "
          f"others={sum(len(d) for d in others)})")
    return pooled


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def correlation_matrix(df):
    scaler = StandardScaler()
    Z = scaler.fit_transform(df)
    return pd.DataFrame(np.corrcoef(Z.T), index=df.columns, columns=df.columns)


def abs_dist(corr):
    """Distance matrix from absolute correlations: d = 1 - |r|."""
    return 1 - np.abs(corr.values)


def ward_linkage(corr):
    dist = abs_dist(corr)
    np.fill_diagonal(dist, 0)
    return linkage(squareform(dist, checks=False), method="ward")


def within_between(corr_abs, feature_list, feat_to_cat):
    """Return (within, between) mean |r| for the given labelling."""
    n = len(feature_list)
    cats = np.array([feat_to_cat.get(f, "?") for f in feature_list])
    within_vals, between_vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            v = corr_abs[i, j]
            if cats[i] == cats[j]:
                within_vals.append(v)
            else:
                between_vals.append(v)
    return np.mean(within_vals), np.mean(between_vals)


def silhouette_for_labels(corr, feature_list, labels):
    """Silhouette score using (1-|r|) as distance."""
    n = len(feature_list)
    if len(set(labels)) < 2:
        return np.nan
    dist = abs_dist(corr)
    np.fill_diagonal(dist, 0)
    return silhouette_score(dist, labels, metric="precomputed")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pooled = load_pooled()
    corr   = correlation_matrix(pooled)
    corr_abs = np.abs(corr.values)
    feats  = list(corr.columns)   # = ALL_FEATS in same order

    Z = ward_linkage(corr)

    # Manual category integer labels (for silhouette)
    cat_names_sorted = list(MANUAL_CATS.keys())
    manual_int = np.array([cat_names_sorted.index(FEAT_TO_CAT[f]) for f in feats])

    manual_sil = silhouette_for_labels(corr, feats, manual_int)
    manual_w, manual_b = within_between(corr_abs, feats, FEAT_TO_CAT)
    print(f"\nManual categories (k={len(cat_names_sorted)}):")
    print(f"  Silhouette (|r| distance): {manual_sil:.3f}")
    print(f"  Within-category mean |r|:  {manual_w:.3f}")
    print(f"  Between-category mean |r|: {manual_b:.3f}")
    print(f"  Separation ratio:          {manual_w/manual_b:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # A: Correlation heatmap ordered by manual categories
    # ─────────────────────────────────────────────────────────────────────────
    print("\nPlot A: correlation heatmap (manual order) …")

    # Sort features by category
    feats_manual_order = [f for feats_cat in MANUAL_CATS.values() for f in feats_cat
                           if f in feats]
    corr_manual = corr.loc[feats_manual_order, feats_manual_order]
    n = len(feats_manual_order)

    fig, ax = plt.subplots(figsize=cm_to_inch((16, 14)), dpi=150)
    im = ax.imshow(corr_manual.values, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="equal", interpolation="none")

    # Category dividers + colour sidebar
    boundaries = []
    idx = 0
    for cat, cat_feats in MANUAL_CATS.items():
        k = len([f for f in cat_feats if f in feats])
        if k == 0:
            continue
        boundaries.append((idx, idx + k, CAT_SHORT.get(cat, cat), CAT_COLORS[cat]))
        if idx > 0:
            ax.axhline(idx - 0.5, color="white", lw=1.5)
            ax.axvline(idx - 0.5, color="white", lw=1.5)
        idx += k

    ax.set_xticks([])
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([short(f) for f in feats_manual_order], fontsize=6)
    ax.tick_params(length=0)

    # Left colour bar
    for (r0, r1, cat_label, color) in boundaries:
        y_top = 1 - r0 / n
        y_bot = 1 - r1 / n
        ax.annotate("", xy=(-0.06, y_bot), xytext=(-0.06, y_top),
                    xycoords="axes fraction", textcoords="axes fraction",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=3))
        ax.text(-0.075, (y_top + y_bot) / 2, cat_label,
                transform=ax.transAxes, ha="right", va="center",
                fontsize=6.5, color=color, fontweight="bold", clip_on=False)

    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01, label="Pearson r")
    ax.set_title(
        f"Feature correlation — manual categories "
        f"(within |r|={manual_w:.2f}, between |r|={manual_b:.2f}, "
        f"ratio={manual_w/manual_b:.2f})",
        fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_06a_corr_heatmap_manual.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # B: Dendrogram coloured by manual categories
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot B: dendrogram vs manual categories …")

    leaf_order  = leaves_list(Z)   # feature indices in dendrogram order
    leaf_feats  = [feats[i] for i in leaf_order]
    leaf_colors = [CAT_COLORS[FEAT_TO_CAT[f]] for f in leaf_feats]

    fig, axes = plt.subplots(2, 1, figsize=cm_to_inch((22, 14)), dpi=150,
                              gridspec_kw={"height_ratios": [10, 1], "hspace": 0.04})

    ddata = dendrogram(Z, ax=axes[0], labels=[short(f) for f in feats],
                       leaf_rotation=90, leaf_font_size=6.5,
                       color_threshold=0, above_threshold_color="#888888",
                       link_color_func=lambda _: "#555555")
    axes[0].set_ylabel("Ward distance (1 − |r|)", fontsize=8)
    axes[0].tick_params(axis="x", labelsize=6.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Colour strip below leaves
    dend_leaf_order = ddata["leaves"]   # order as plotted by dendrogram
    for xi, leaf_idx in enumerate(dend_leaf_order):
        feat = feats[leaf_idx]
        cat  = FEAT_TO_CAT[feat]
        color = CAT_COLORS[cat]
        axes[1].barh(0, 1, left=xi, height=1, color=color, edgecolor="none")
    axes[1].set_xlim(0, len(feats))
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_frame_on(False)

    # Legend
    legend_handles = [
        mpatches.Patch(color=CAT_COLORS[cat], label=CAT_SHORT.get(cat, cat))
        for cat in MANUAL_CATS if any(f in feats for f in MANUAL_CATS[cat])
    ]
    axes[0].legend(handles=legend_handles, fontsize=6.5, loc="upper right",
                   title="Manual category", title_fontsize=7,
                   framealpha=0.9, ncol=2)
    axes[0].set_title(
        "Ward dendrogram (distance = 1 − |r|)  ·  "
        "leaf colour = manual category\n"
        "Features that cluster together in the tree but have different colours "
        "are cross-category neighbours",
        fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_06b_dendrogram_vs_manual.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # C: Silhouette sweep k = 2 … 15
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot C: silhouette sweep …")

    k_range   = range(2, 16)
    sil_scores = []
    wb_ratios  = []

    for k in k_range:
        clust = fcluster(Z, t=k, criterion="maxclust")
        sil = silhouette_for_labels(corr, feats, clust)
        feat_to_clust = {f: str(clust[i]) for i, f in enumerate(feats)}
        w, b = within_between(corr_abs, feats, feat_to_clust)
        sil_scores.append(sil)
        wb_ratios.append(w / b if b > 0 else np.nan)

    best_k = k_range.start + int(np.argmax(sil_scores))
    print(f"  Best k by silhouette: {best_k}  "
          f"(sil={max(sil_scores):.3f})")

    fig, ax1 = plt.subplots(figsize=cm_to_inch((13, 7)), dpi=150)
    ax2 = ax1.twinx()

    ax1.plot(list(k_range), sil_scores, "o-", color="#3FA5C4", lw=2,
             label="Silhouette score")
    ax2.plot(list(k_range), wb_ratios, "s--", color="#E84653", lw=1.5,
             alpha=0.7, label="Within/between ratio")

    ax1.axvline(len(cat_names_sorted), color="#555555", lw=1.2, ls=":",
                label=f"Your k = {len(cat_names_sorted)}")
    ax1.axvline(best_k, color="#FFD700", lw=1.5, ls="--",
                label=f"Best k = {best_k}")

    ax1.set_xlabel("Number of clusters k", fontsize=9)
    ax1.set_ylabel("Silhouette score  (higher = better)", fontsize=9,
                   color="#3FA5C4")
    ax2.set_ylabel("Within / between |r| ratio", fontsize=9, color="#E84653")
    ax1.tick_params(axis="y", labelcolor="#3FA5C4")
    ax2.tick_params(axis="y", labelcolor="#E84653")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.set_title("Cluster quality vs number of categories", fontsize=10)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_06c_silhouette_sweep.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # D: Side-by-side cluster comparison (manual vs data-driven)
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot D: cluster comparison …")

    def cluster_assignment_matrix(labels_int, label_names, leaf_idx_order):
        """Binary matrix: rows = clusters, cols = features (in leaf_idx_order)."""
        n_clust = max(labels_int) + 1
        mat = np.zeros((n_clust, len(feats)))
        for fi, li in enumerate(labels_int):
            mat[li, fi] = 1
        return mat[:, leaf_idx_order]

    # Use dendrogram leaf order so both panels have identical column order
    dend_order = ddata["leaves"]
    feats_dend = [feats[i] for i in dend_order]

    # Manual labels re-ordered to dendrogram order
    manual_labels_dend  = [manual_int[i] for i in dend_order]
    # Data-driven at best_k
    dd_clust = fcluster(Z, t=best_k, criterion="maxclust") - 1   # 0-indexed
    dd_labels_dend = [dd_clust[i] for i in dend_order]
    # Data-driven at manual k (fair comparison)
    mk_clust = fcluster(Z, t=len(cat_names_sorted), criterion="maxclust") - 1
    mk_labels_dend = [mk_clust[i] for i in dend_order]

    # Build colour matrices: each row is a cluster, each column is a feature
    # Colour = category colour for manual; gradient for data-driven
    def _rgb(hex_color):
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

    n_cols = len(feats)

    fig, axes = plt.subplots(3, 1, figsize=cm_to_inch((22, 12)), dpi=130,
                              gridspec_kw={"hspace": 0.6})

    titles = [
        f"Manual categories (k={len(cat_names_sorted)}, "
        f"sil={manual_sil:.2f}, ratio={manual_w/manual_b:.2f})",
        f"Data-driven at best k={best_k}  "
        f"(sil={sil_scores[best_k - k_range.start]:.2f}, "
        f"ratio={wb_ratios[best_k - k_range.start]:.2f})",
        f"Data-driven at k={len(cat_names_sorted)} (same as manual)  "
        f"(sil={sil_scores[len(cat_names_sorted) - k_range.start]:.2f}, "
        f"ratio={wb_ratios[len(cat_names_sorted) - k_range.start]:.2f})",
    ]

    for ax, labels_dend, title, is_manual in zip(
        axes,
        [manual_labels_dend, dd_labels_dend, mk_labels_dend],
        titles,
        [True, False, False],
    ):
        # Build strip: one row, each cell coloured by cluster
        n_clust_here = max(labels_dend) + 1
        row_colors = np.zeros((1, n_cols, 3))
        for xi, label in enumerate(labels_dend):
            if is_manual:
                feat = feats_dend[xi]
                cat  = FEAT_TO_CAT[feat]
                row_colors[0, xi] = _rgb(CAT_COLORS[cat])
            else:
                hue = label / max(1, n_clust_here - 1)
                row_colors[0, xi] = plt.cm.tab20(hue)[:3]

        ax.imshow(row_colors, aspect="auto", interpolation="none")
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([short(f) for f in feats_dend],
                            rotation=90, fontsize=5.5)

        # Colour each label by category
        for tick, f in zip(ax.get_xticklabels(), feats_dend):
            if is_manual:
                tick.set_color(CAT_COLORS[FEAT_TO_CAT[f]])
            else:
                tick.set_color("#232324")

        ax.set_yticks([])
        ax.set_title(title, fontsize=8, pad=4)

        # Draw dividers between consecutive different labels
        for xi in range(1, n_cols):
            if labels_dend[xi] != labels_dend[xi - 1]:
                ax.axvline(xi - 0.5, color="white", lw=1.5)

    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_06d_cluster_comparison.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # E: Within / between ratio per manual category
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot E: per-category within/between ratio …")

    cat_stats = []
    for cat, cat_feats in MANUAL_CATS.items():
        inside = [f for f in cat_feats if f in feats]
        if len(inside) < 2:
            cat_stats.append({"cat": CAT_SHORT.get(cat, cat),
                               "within": np.nan, "between": np.nan,
                               "n": len(inside), "color": CAT_COLORS[cat]})
            continue
        inside_idx  = [feats.index(f) for f in inside]
        outside_idx = [i for i in range(len(feats)) if i not in inside_idx]
        w_vals = [corr_abs[i, j] for i in inside_idx for j in inside_idx if i < j]
        b_vals = [corr_abs[i, j] for i in inside_idx for j in outside_idx]
        cat_stats.append({
            "cat":     CAT_SHORT.get(cat, cat),
            "within":  np.mean(w_vals) if w_vals else np.nan,
            "between": np.mean(b_vals) if b_vals else np.nan,
            "n":       len(inside),
            "color":   CAT_COLORS[cat],
        })

    cs = pd.DataFrame(cat_stats).dropna(subset=["within"])
    cs["ratio"] = cs["within"] / cs["between"]
    cs = cs.sort_values("ratio", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=cm_to_inch((18, 8)), dpi=150,
                              sharey=True)
    y_pos = np.arange(len(cs))

    # Left: within vs between
    axes[0].barh(y_pos - 0.2, cs["within"],  height=0.35,
                 color=[c for c in cs["color"]], alpha=0.9, label="Within")
    axes[0].barh(y_pos + 0.2, cs["between"], height=0.35,
                 color=[c for c in cs["color"]], alpha=0.4, label="Between")
    axes[0].axvline(manual_b, color="black", lw=0.8, ls="--", alpha=0.5,
                    label="Global between mean")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(
        [f"{r['cat']}  (n={r['n']})" for _, r in cs.iterrows()],
        fontsize=7.5
    )
    for tick, (_, row) in zip(axes[0].get_yticklabels(), cs.iterrows()):
        tick.set_color(row["color"])
    axes[0].set_xlabel("Mean |r|", fontsize=8)
    axes[0].set_title("Within (dark) vs between (light)", fontsize=8)
    axes[0].legend(fontsize=7)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: ratio
    bar_colors = [("#5DC400" if r > 1 else "#E84653") for r in cs["ratio"]]
    axes[1].barh(y_pos, cs["ratio"], color=bar_colors, height=0.6, edgecolor="none")
    axes[1].axvline(1.0, color="black", lw=1, ls="--")
    axes[1].set_xlabel("Within / between ratio  (> 1 = coherent)", fontsize=8)
    axes[1].set_title("Coherence ratio per category", fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Per-category correlation coherence — manual groupings",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_06e_within_between_ratio.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # F: Sign / flip audit
    # ─────────────────────────────────────────────────────────────────────────
    print("Plot F: sign/flip audit …")

    flip_recommendations = {}   # feat → True means "flip this feature"
    all_loadings = {}

    for cat, cat_feats in MANUAL_CATS.items():
        inside = [f for f in cat_feats if f in feats]
        if not inside:
            continue

        # z-scored features
        Z_cat = StandardScaler().fit_transform(pooled[inside].values)

        if len(inside) == 1:
            loadings = np.array([1.0])
        else:
            pca_cat = PCA(n_components=1)
            pca_cat.fit(Z_cat)
            loadings = pca_cat.components_[0]  # shape (n_features_in_cat,)

            # Fix sign: majority of loadings positive
            if (loadings < 0).sum() > (loadings > 0).sum():
                loadings = -loadings

        all_loadings[cat] = dict(zip(inside, loadings))
        for feat, loading in zip(inside, loadings):
            flip_recommendations[feat] = bool(loading < 0)

    # Plot
    n_cats_with_feats = sum(
        1 for cat, feats_c in MANUAL_CATS.items()
        if any(f in feats for f in feats_c)
    )
    ncols_f = 3
    nrows_f = (n_cats_with_feats + ncols_f - 1) // ncols_f

    fig, axes = plt.subplots(nrows_f, ncols_f,
                              figsize=cm_to_inch((7 * ncols_f, 4.5 * nrows_f)),
                              dpi=130)
    axes_flat = np.array(axes).flatten()
    ax_idx = 0

    for cat, cat_feats in MANUAL_CATS.items():
        inside = [f for f in cat_feats if f in feats]
        if not inside or cat not in all_loadings:
            continue

        ax = axes_flat[ax_idx]
        ax_idx += 1

        loadings_here = [all_loadings[cat][f] for f in inside]
        colors_bars   = ["#E84653" if l < 0 else "#5DC400" for l in loadings_here]
        y_pos_f = np.arange(len(inside))

        ax.barh(y_pos_f, loadings_here, color=colors_bars, height=0.7, edgecolor="none")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y_pos_f)
        ax.set_yticklabels([short(f) for f in inside], fontsize=7)
        for tick, l in zip(ax.get_yticklabels(), loadings_here):
            tick.set_color("#E84653" if l < 0 else "#232324")
        ax.set_title(CAT_SHORT.get(cat, cat), fontsize=8,
                     color=CAT_COLORS.get(cat, "black"), pad=3)
        ax.set_xlabel("PC1 loading", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotation
        n_flip = sum(1 for l in loadings_here if l < 0)
        if n_flip > 0:
            ax.text(0.98, 0.02, f"⚠ {n_flip} need flip",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.5, color="#E84653")

    for ax in axes_flat[ax_idx:]:
        ax.set_visible(False)

    fig.suptitle(
        "PC1 loading audit per category\n"
        "Red bars = negative loading → feature should be sign-flipped\n"
        "for consistent 'more of this category' direction",
        fontsize=9, y=1.02
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "gnm_06f_sign_flip_audit.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Console summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("CLUSTER QUALITY SUMMARY")
    print("═" * 65)
    print(f"{'Method':<35}  {'sil':>6}  {'ratio':>6}")
    print("-" * 65)
    print(f"{'Manual (k=' + str(len(cat_names_sorted)) + ')':<35}  "
          f"{manual_sil:>6.3f}  {manual_w/manual_b:>6.2f}")
    for k, sil, ratio in zip(k_range, sil_scores, wb_ratios):
        marker = " ← best" if k == best_k else (
                 " ← manual k" if k == len(cat_names_sorted) else "")
        print(f"{'Data-driven k=' + str(k):<35}  {sil:>6.3f}  {ratio:>6.2f}{marker}")

    print("\n" + "═" * 65)
    print("SIGN FLIP RECOMMENDATIONS")
    print("Paste this dict into your code and negate features where True:")
    print("═" * 65)
    flips_needed = {f: v for f, v in flip_recommendations.items() if v}
    no_flip      = {f: v for f, v in flip_recommendations.items() if not v}
    print(f"\n# {len(flips_needed)} feature(s) need sign flip:")
    print("FLIP_SIGNS = {")
    for f, need_flip in sorted(flip_recommendations.items(),
                               key=lambda x: (not x[1], x[0])):
        cat = FEAT_TO_CAT.get(f, "?")
        comment = f"  # [{CAT_SHORT.get(cat, cat)}]"
        flag = "True " if need_flip else "False"
        print(f'    "{f}": {flag},{comment}')
    print("}")

    if not flips_needed:
        print("\n→ All features already point in the positive direction "
              "within their categories. No flipping needed!")
    else:
        print(f"\n→ {len(flips_needed)} features need flipping (shown in red in plot F).")
        print("  These are anti-correlated with their category's dominant direction.")

    print("\n" + "═" * 65)
    print("INTERPRETATION GUIDE")
    print("─" * 65)
    print("• Silhouette > 0.3 = reasonable clusters; > 0.5 = strong clusters")
    print("• Within/between ratio > 1 = features within a category are more")
    print("  similar to each other than to features outside it  (= coherent)")
    print("• Categories with ratio < 1 = weakly justified; consider merging")
    print("  them with their nearest dendrogram neighbour (plot B)")
    print("• If best_k < your manual k, you may have split some natural")
    print("  clusters into two; look at plot D for which ones")
    print(f"\nYour manual k={len(cat_names_sorted)}, data-driven best k={best_k}")
    if best_k == len(cat_names_sorted):
        print("→ The number of categories is well supported by the data.")
    elif best_k < len(cat_names_sorted):
        print(f"→ The data supports {best_k} clusters — consider merging "
              f"{len(cat_names_sorted) - best_k} category pair(s).")
        print("  Check plot B: find leaves of the same colour that are "
              "dendrogram-adjacent.")
    else:
        print(f"→ The data could support {best_k} clusters — some categories "
              "might be splittable.")

    print("\nDone — all outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
