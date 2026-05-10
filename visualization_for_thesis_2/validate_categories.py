"""
validate_categories.py
──────────────────────
Two things in one script:

  1. VALIDATE manual categories
     Compute Spearman feature-feature correlations, then check whether
     features within the same category are more correlated than features
     across categories.  Produces:
       • heatmap with manual category boundaries (saves PDF)
       • within- vs between-category correlation box plots
       • silhouette score for the manual partition
       • permutation p-value: is the manual partition significantly better
         than random?

  2. DISCOVER data-driven categories
     Ward hierarchical clustering on the correlation-distance matrix,
     cut to the same number of categories as the manual partition.
     Produces:
       • dendrogram (saves PDF)
       • Adjusted Rand Index between manual and data-driven partitions
       • lists which features switched categories

  3. FIX PC1 sign with anchor features
     For each category, one "anchor" feature is designated whose positive
     direction is semantically clear (e.g. global_efficiency: higher = better
     integration).  PC1 is flipped if its correlation with the anchor is
     negative.  Produces a corrected feat_to_cat + anchor dict ready to
     drop into build_category_scores().

Usage
─────
Run from the B_pca_and_pareto_trade-off/ folder after loading your data:

    from validate_categories import run_validation
    run_validation(huge_df_props, PRECISE_FEATURES)

Or run standalone with the path to the data pickle and the config.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA


OUTPUT_DIR = Path("/Users/adrian/Documents/01_projects/14_4D_lab/connectome_distances/B_pca_and_pareto_trade-off")

# ── Anchor features: one per category, positive direction = "more brain-like" ──
# Higher value = more of the category concept after any pre-flip already applied.
ANCHOR_FEATURES = {
    "Fundamental Topology":
        "modularity",                                           # higher Q = more modular
    "Paths, Efficiency & Communication":
        "global_efficiency",                                    # higher = better integration
    "Spatial Embedding & Wiring Cost":
        "wiring_cost",                                          # higher = more expensive wiring
    "Rich-Club Organization":
        "rich_club_coefficient_rc_k_at_max",                   # higher k = richer rich club
    "Spectral Properties, Algebraic Connectivity & Fiedler Analysis":
        "spectral_radius",                                      # higher = more complex dynamics
    "Metastability & Kuramoto Synchronization":
        "repertoire_sweep_weighted_by_distances_T_critical",   # higher T_crit = richer metastability
    "Participation Coefficient (Louvain)":
        "participation_coefficient_pc_std",                    # higher std = more heterogeneous roles
    "Ollivier-Ricci Curvature":
        "ollivier_ricci_curvature_orc_skewness",               # higher skewness = more negatively curved
    "Targeted Attack Robustness":
        "targeted_attack_robustness_rob_targeted_auc",         # higher AUC = more robust
    "Network Control Theory - Average Controllability":
        "nct_control_std",                                      # higher std = more heterogeneous control
    "Memory Capacity":
        "mc_input_scaling_0_1_mc_mean",                        # higher = more memory
}


# ──────────────────────────────────────────────────────────────────────────────
# CORE
# ──────────────────────────────────────────────────────────────────────────────

def feature_correlation_matrix(df_props: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of all features (columns) across all networks (rows)."""
    corr, _ = spearmanr(df_props.values)
    if df_props.shape[1] == 2:
        corr = np.array([[1.0, corr], [corr, 1.0]])
    return pd.DataFrame(corr, index=df_props.columns, columns=df_props.columns)


def category_label_vector(feature_list: list, feat_to_cat: dict) -> np.ndarray:
    """Integer label per feature, in the order of feature_list."""
    cats = sorted(set(feat_to_cat[f] for f in feature_list if f in feat_to_cat))
    cat_to_int = {c: i for i, c in enumerate(cats)}
    return np.array([cat_to_int[feat_to_cat[f]] for f in feature_list
                     if f in feat_to_cat]), cats


def within_between_correlations(corr_df: pd.DataFrame,
                                 feat_to_cat: dict) -> tuple[list, list]:
    """
    Returns two flat lists: within-category |r| values and between-category |r|.
    Only the upper triangle is used to avoid double-counting.
    """
    features = [f for f in corr_df.columns if f in feat_to_cat]
    within, between = [], []
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if j <= i:
                continue
            r = abs(corr_df.loc[f1, f2])
            if feat_to_cat[f1] == feat_to_cat[f2]:
                within.append(r)
            else:
                between.append(r)
    return within, between


def permutation_test_silhouette(corr_df: pd.DataFrame,
                                 feat_to_cat: dict,
                                 n_permutations: int = 1000,
                                 seed: int = 42) -> tuple[float, float]:
    """
    Test whether the silhouette score of the manual partition is significantly
    higher than random.  Returns (observed_silhouette, p_value).
    """
    rng = np.random.default_rng(seed)
    features = [f for f in corr_df.columns if f in feat_to_cat]
    dist_matrix = 1 - corr_df.loc[features, features].values
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)  # NaN → max distance
    dist_matrix = np.clip(dist_matrix, 0, None)        # numerical safety

    labels, _ = category_label_vector(features, feat_to_cat)
    observed   = silhouette_score(dist_matrix, labels, metric="precomputed")

    null_scores = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        null_scores.append(
            silhouette_score(dist_matrix, shuffled, metric="precomputed")
        )

    p_value = np.mean(np.array(null_scores) >= observed)
    return observed, p_value, np.array(null_scores)


def discover_clusters(corr_df: pd.DataFrame,
                       feat_to_cat: dict,
                       n_clusters: int | None = None) -> dict:
    """
    Ward hierarchical clustering on 1 − |r| distance.
    n_clusters defaults to the number of manual categories.
    Returns a dict: feature -> discovered_cluster_label.
    """
    features = [f for f in corr_df.columns if f in feat_to_cat]
    dist     = 1 - np.abs(corr_df.loc[features, features].values)
    np.fill_diagonal(dist, 0)

    if n_clusters is None:
        n_clusters = len(set(feat_to_cat.values()))

    Z      = linkage(squareform(dist), method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return {f: int(labels[i]) for i, f in enumerate(features)}, Z, features


def anchor_based_sign_fix(cat_scores: np.ndarray,
                           cat_labels: list,
                           per_cat_pca: dict,
                           df_props_scaled: np.ndarray,
                           feature_cols: list,
                           anchor_features: dict) -> np.ndarray:
    """
    For each category, flip the PC1 score if it is negatively correlated
    with the designated anchor feature (after z-scoring).

    Parameters
    ----------
    cat_scores      : (n_networks, n_cats) from build_category_scores()
    cat_labels      : list of category names, same order as cat_scores columns
    per_cat_pca     : dict category -> fitted PCA(1) or None
    df_props_scaled : (n_networks, n_features) already z-scored
    feature_cols    : list of feature names, same order as df_props_scaled cols
    anchor_features : dict  category -> anchor_feature_name

    Returns
    -------
    cat_scores_fixed : same shape, signs corrected
    flip_report      : dict  category -> bool (True = was flipped)
    """
    cat_scores_fixed = cat_scores.copy()
    flip_report      = {}
    feat_idx         = {f: i for i, f in enumerate(feature_cols)}

    for col_i, cat in enumerate(cat_labels):
        anchor = anchor_features.get(cat)
        if anchor is None or anchor not in feat_idx:
            flip_report[cat] = False
            continue

        anchor_vals = df_props_scaled[:, feat_idx[anchor]]
        pc1_vals    = cat_scores[:, col_i]

        r = np.corrcoef(pc1_vals, anchor_vals)[0, 1]
        if r < 0:
            cat_scores_fixed[:, col_i] *= -1
            flip_report[cat] = True
        else:
            flip_report[cat] = False

    return cat_scores_fixed, flip_report


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_df: pd.DataFrame,
                              feat_to_cat: dict,
                              cat_order: list,
                              output_path: Path = None,
                              title: str = "Feature–feature Spearman correlation\n(sorted by manual category)",
                              label_fontsize: int = 5,
                              show_feature_names: bool = False):
    """
    Sorted heatmap with manual category boundaries overlaid as coloured rectangles.
    Features are sorted by category (in cat_order).  Features not present in
    feat_to_cat are appended at the end under an implicit 'Other' group.

    Parameters
    ----------
    show_feature_names : if True, tick labels show shortened feature names
                         (useful when n_features is small).
    """
    sorted_feats = []
    for cat in cat_order:
        sorted_feats.extend([f for f in corr_df.columns
                              if feat_to_cat.get(f) == cat])
    sorted_feats.extend([f for f in corr_df.columns if f not in sorted_feats])

    mat = corr_df.loc[sorted_feats, sorted_feats].values
    n   = len(sorted_feats)

    cmap = LinearSegmentedColormap.from_list(
        "div", ["#2166ac", "#f7f7f7", "#d6604d"], N=256
    )

    fig_size = max(8, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9), dpi=150)
    im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman ρ")

    cat_palette = plt.cm.tab10.colors
    cat_colours = {c: cat_palette[i % 10] for i, c in enumerate(cat_order)}
    cursor = 0
    for cat in cat_order:
        n_cat = sum(1 for f in sorted_feats if feat_to_cat.get(f) == cat)
        if n_cat == 0:
            continue
        rect = plt.Rectangle((cursor - 0.5, cursor - 0.5), n_cat, n_cat,
                               edgecolor=cat_colours[cat], facecolor="none",
                               linewidth=2.0)
        ax.add_patch(rect)
        # Category label above the block
        short_label = cat.split("&")[0].split(",")[0].strip()  # first part only
        ax.text(cursor + n_cat / 2, -1.5, short_label,
                ha="center", va="bottom", fontsize=label_fontsize,
                color=cat_colours[cat], fontweight="bold")
        cursor += n_cat

    if show_feature_names:
        short_names = [f.split("_")[-1][:12] for f in sorted_feats]
        ax.set_xticks(range(len(sorted_feats)))
        ax.set_yticks(range(len(sorted_feats)))
        ax.set_xticklabels(short_names, rotation=90, fontsize=4)
        ax.set_yticklabels(short_names, fontsize=4)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {output_path}")
    plt.show()
    return fig


def plot_within_between(within: list, between: list, output_path: Path = None):
    """Box plot of within- vs between-category |r|."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    bp = ax.boxplot(
        [within, between],
        labels=["Within\ncategory", "Between\ncategories"],
        patch_artist=True,
        showfliers=False,
    )
    colours = ["#5B8DB8", "#C75D5D"]
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_ylabel("|Spearman ρ|")
    ax.set_title("Within- vs between-category\nfeature correlation")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {output_path}")
    plt.show()
    return fig


def plot_dendrogram(Z: np.ndarray,
                    features: list,
                    feat_to_cat: dict,
                    cat_order: list,
                    n_clusters: int,
                    output_path: Path = None):
    """Dendrogram coloured by manual category membership."""
    cat_palette   = plt.cm.tab10.colors
    cat_to_colour = {c: cat_palette[i % 10] for i, c in enumerate(cat_order)}

    fig, ax = plt.subplots(figsize=(max(8, len(features) * 0.4), 5), dpi=150)

    dendrogram(Z, labels=features, ax=ax,
               color_threshold=0,
               above_threshold_color="#aaaaaa",
               leaf_rotation=90, leaf_font_size=6)

    for lbl in ax.get_xticklabels():
        f   = lbl.get_text()
        col = cat_to_colour.get(feat_to_cat.get(f, ""), "#999999")
        lbl.set_color(col)

    handles = [mpatches.Patch(color=cat_to_colour[c], label=c)
               for c in cat_order if c in cat_to_colour]
    ax.legend(handles=handles, fontsize=6, loc="upper right",
              frameon=True, title="Manual category")

    ax.set_title(f"Ward hierarchical clustering of features\n"
                 f"(leaf colour = manual category, cut at k={n_clusters})",
                 fontsize=9)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {output_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def detrend_by_dataset(df_props: pd.DataFrame,
                        dataset_labels: pd.Series) -> pd.DataFrame:
    """
    Remove per-dataset mean from every feature so that between-dataset
    differences do not dominate the correlation structure.
    Each feature is z-scored within its dataset (subtract mean, divide by std).
    Datasets with only one row keep their values unchanged.
    """
    df_out = df_props.copy()
    for ds in dataset_labels.unique():
        mask = dataset_labels == ds
        subset = df_out.loc[mask]
        mu  = subset.mean()
        sig = subset.std().replace(0, np.nan)   # avoid div-by-zero
        df_out.loc[mask] = (subset - mu) / sig
    return df_out


def run_validation(df_props: pd.DataFrame,
                   precise_features: dict,
                   all_features: dict = None,
                   dataset_labels: pd.Series = None,
                   output_dir: Path = OUTPUT_DIR,
                   n_permutations: int = 1000):
    """
    Parameters
    ----------
    df_props         : pd.DataFrame  rows = networks, cols = feature names
    precise_features : dict  category -> list of feature names  (the selected subset)
    all_features     : optional dict  category -> list of feature names for a
                       second heatmap showing all available features, e.g.
                       remaining_categories from config.py.  If None, only the
                       precise_features heatmap is produced.
    dataset_labels   : optional pd.Series (same index as df_props) with dataset
                       names.  If provided, each feature is z-scored within
                       dataset before computing correlations, so that
                       between-dataset mean differences do not inflate the
                       correlation structure.  Pass huge_df["dataset"].
    output_dir       : where to save PDFs
    n_permutations   : for the permutation test
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_to_cat = {f: cat for cat, feats in precise_features.items() for f in feats}
    cat_order   = list(precise_features.keys())

    common_feats = [f for f in df_props.columns if f in feat_to_cat]
    df_sub       = df_props[common_feats].copy()
    df_sub.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sub.dropna(axis=1, inplace=True)
    common_feats = list(df_sub.columns)
    feat_to_cat  = {f: feat_to_cat[f] for f in common_feats}

    # ── Remove dataset-level mean differences if labels are provided ─────────
    if dataset_labels is not None:
        labels = dataset_labels.loc[df_sub.index]
        df_sub = detrend_by_dataset(df_sub, labels)
        # Zero-variance features within a dataset produce NaN after z-scoring;
        # replace with 0 (= "at the group mean" in z-score space).
        n_nan_cols = df_sub.isna().any(axis=0).sum()
        if n_nan_cols:
            print(f"  {n_nan_cols} features had zero variance in at least one "
                  f"dataset — NaN filled with 0.")
        df_sub = df_sub.fillna(0)
        print(f"Detrended by dataset ({labels.nunique()} datasets).")

    print(f"Features used: {len(common_feats)}")
    print(f"Categories   : {len(set(feat_to_cat.values()))}")

    # ── 1. Spearman correlation matrix (selected features) ──────────────────
    print("\nComputing Spearman correlations …")
    corr_df = feature_correlation_matrix(df_sub)

    plot_correlation_heatmap(
        corr_df, feat_to_cat, cat_order,
        output_path   = output_dir / "feat_corr_heatmap_selected.pdf",
        title         = f"Feature–feature Spearman ρ — {len(common_feats)} selected features\n(sorted by manual category, dataset-detrended)",
        label_fontsize= 6,
    )

    # ── 1b. Second heatmap: all features ────────────────────────────────────
    if all_features is not None:
        all_feat_to_cat = {f: cat for cat, feats in all_features.items() for f in feats}
        all_cat_order   = list(all_features.keys())

        # Keep only features present in df_props (before the precise_features filter)
        all_common = [f for f in df_props.columns if f in all_feat_to_cat]
        df_all = df_props[all_common].copy()
        df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_all.dropna(axis=1, inplace=True)
        all_common = list(df_all.columns)
        all_feat_to_cat = {f: all_feat_to_cat[f] for f in all_common}

        if dataset_labels is not None:
            df_all = detrend_by_dataset(df_all, dataset_labels.loc[df_all.index])
            df_all = df_all.fillna(0)

        print(f"\nComputing Spearman correlations for all {len(all_common)} features …")
        corr_df_all = feature_correlation_matrix(df_all)

        plot_correlation_heatmap(
            corr_df_all, all_feat_to_cat, all_cat_order,
            output_path    = output_dir / "feat_corr_heatmap_all.pdf",
            title          = f"Feature–feature Spearman ρ — all {len(all_common)} features\n(sorted by manual category, dataset-detrended)",
            label_fontsize = 5,
        )

    # ── 2. Within vs between category correlations ──────────────────────────
    within, between = within_between_correlations(corr_df, feat_to_cat)
    print(f"\nWithin-category  |ρ| : mean={np.mean(within):.3f}  "
          f"median={np.median(within):.3f}")
    print(f"Between-category |ρ| : mean={np.mean(between):.3f}  "
          f"median={np.median(between):.3f}")

    plot_within_between(within, between,
                        output_dir / "within_vs_between_corr.pdf")

    # ── 3. Silhouette + permutation test ────────────────────────────────────
    print(f"\nRunning permutation test ({n_permutations} permutations) …")
    sil, p_val, null_dist = permutation_test_silhouette(
        corr_df, feat_to_cat, n_permutations=n_permutations
    )
    print(f"Observed silhouette score : {sil:.4f}")
    print(f"Permutation p-value        : {p_val:.4f}")
    print(f"Null mean ± SD             : "
          f"{null_dist.mean():.4f} ± {null_dist.std():.4f}")

    fig_sil, ax_sil = plt.subplots(figsize=(5, 3), dpi=150)
    ax_sil.hist(null_dist, bins=40, color="#aaaaaa", alpha=0.7,
                label="Random partitions")
    ax_sil.axvline(sil, color="#C75D5D", linewidth=2,
                   label=f"Manual categories\n(s={sil:.3f}, p={p_val:.3f})")
    ax_sil.set_xlabel("Silhouette score")
    ax_sil.set_ylabel("Count")
    ax_sil.legend(fontsize=8)
    ax_sil.set_title("Permutation test: manual category partition")
    fig_sil.tight_layout()
    fig_sil.savefig(output_dir / "silhouette_permutation_test.pdf",
                    bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {output_dir / 'silhouette_permutation_test.pdf'}")

    # ── 4. Data-driven clustering ────────────────────────────────────────────
    n_cats = len(set(feat_to_cat.values()))
    print(f"\nDiscovering {n_cats} data-driven clusters (Ward) …")
    discovered, Z, feat_order = discover_clusters(corr_df, feat_to_cat, n_cats)

    ari = adjusted_rand_score(
        [feat_to_cat[f] for f in feat_order],
        [discovered[f] for f in feat_order]
    )
    print(f"Adjusted Rand Index (manual vs discovered) : {ari:.4f}")
    print("  (1.0 = perfect agreement, 0.0 = random)")

    plot_dendrogram(Z, feat_order, feat_to_cat, cat_order, n_cats,
                    output_dir / "feature_dendrogram.pdf")

    # Print which features changed cluster
    print("\nFeatures whose data-driven cluster differs from manual:")
    manual_int, cats_list = category_label_vector(feat_order, feat_to_cat)
    discovered_int = np.array([discovered[f] for f in feat_order])

    mismatched = [(f, feat_to_cat[f], discovered[f])
                  for f in feat_order if feat_to_cat[f] != cats_list[
                      discovered[f] - 1 if discovered[f] - 1 < len(cats_list) else 0
                  ]]
    if mismatched:
        print(f"  {len(mismatched)} features sit in a different data-driven cluster.")
        for f, manual_cat, disc_int in mismatched[:10]:
            print(f"    {f[:50]:<50}  manual={manual_cat}  → cluster={disc_int}")
    else:
        print("  All features match their manual category in the data-driven clustering.")

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Within-cat  mean |ρ| : {np.mean(within):.3f}")
    print(f"  Between-cat mean |ρ| : {np.mean(between):.3f}")
    print(f"  Ratio (within/between): {np.mean(within)/np.mean(between):.2f}x")
    print(f"  Silhouette score      : {sil:.4f}  (p={p_val:.4f})")
    print(f"  Adjusted Rand Index   : {ari:.4f}")
    if p_val < 0.05 and np.mean(within) > np.mean(between):
        print("\n  → Your manual categories are statistically justified:")
        print("    within-category correlations are significantly higher than")
        print("    between-category correlations.")
    else:
        print("\n  → The manual partition is NOT clearly better than random.")
        print("    Consider using the data-driven clusters instead,")
        print("    or merging / splitting categories with low internal correlation.")
    print("="*60)

    return corr_df, sil, p_val, ari


# ──────────────────────────────────────────────────────────────────────────────
# HOW TO USE ANCHOR-BASED SIGN FIX (call after build_category_scores)
# ──────────────────────────────────────────────────────────────────────────────
# from validate_categories import anchor_based_sign_fix, ANCHOR_FEATURES
#
# cat_scores_fixed, flip_report = anchor_based_sign_fix(
#     cat_scores      = cat_scores,
#     cat_labels      = cat_labels,
#     per_cat_pca     = per_cat_pca,
#     df_props_scaled = scaled,          # StandardScaler output
#     feature_cols    = feature_cols,
#     anchor_features = ANCHOR_FEATURES,
# )
#
# print("Flipped categories:", [c for c, flipped in flip_report.items() if flipped])
# ──────────────────────────────────────────────────────────────────────────────


# if __name__ == "__main__":
#     # ── Minimal smoke-test with synthetic data ────────────────────────────────
#     rng = np.random.default_rng(0)

#     PRECISE_FEATURES_DEMO = {
#         "Topology":   ["modularity", "omega", "simplices"],
#         "Efficiency": ["glob_eff", "diff_eff", "prop_eff"],
#         "Spectral":   ["spec_radius", "spec_gap", "sync_ratio", "alg_conn"],
#         "Dynamics":   ["T_crit", "size_crit", "div_crit"],
#         "Robustness": ["rob_auc", "rob_ratio", "vuln"],
#     }

#     feats = [f for fs in PRECISE_FEATURES_DEMO.values() for f in fs]
#     n     = 300

#     data = {}
#     for cat, fs in PRECISE_FEATURES_DEMO.items():
#         latent = rng.standard_normal(n)
#         for f in fs:
#             data[f] = latent + rng.standard_normal(n) * 0.5

#     df_demo = pd.DataFrame(data)
#     run_validation(df_demo, PRECISE_FEATURES_DEMO, n_permutations=200)

#     # from validate_categories import run_validation
#     from spider_per_order_categories import PRECISE_FEATURES
#     from config import remaining_categories   # the full 30-feature dict

#     run_validation(
#         df_props       = huge_df_props,
#         precise_features = PRECISE_FEATURES,      # the 11-category subset
#         all_features   = remaining_categories,    # full 30-feature set
#         dataset_labels = huge_df["dataset"],
#     )
