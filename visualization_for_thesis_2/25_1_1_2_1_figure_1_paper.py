#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# External configs (assumes these are in your config.py and vizman)
from vizman import viz
from config import COLOR_SCHEME, LABEL_MAP # PROPERTY_NAMES, CATEGORY_COLOURS, CATEGORY_ORDER, remaining_categories, SELECTED_PROPERTIES_NAMES 
from analysis_08_cluster_contributions import CLUSTERS 
from analysis_08_cluster_contributions import CLUSTER_COLORS as CATEGORY_COLOURS
from config import MERGED_PROPERTIES_NAMES as SELECTED_PROPERTIES_NAMES
from utils import get_combined_colors
from config_pca_parameter_selection import remaining_categories

CATEGORY_ORDER = [k for k in CLUSTERS.keys()]
print(CATEGORY_ORDER)

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[10]:


# ==========================================
# 1. Configuration & Setup
# ==========================================

# # %%
# output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis")
# output_folder.mkdir(exist_ok=True)


# with open(output_folder / "all_datasets_precise_categories.pkl", "rb") as f:
#     dict_with_all_datasets = pickle.load(f)


OUTPUT_FOLDER = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis") # Path("output/00_trade_off_analysis")
DATA_PATH = OUTPUT_FOLDER / "all_datasets_precise_categories.pkl" # Path("data/all_datasets_precise_categories.pkl")
EXCEL_META_PATH = Path("/Users/adrian/Desktop/network_properties_full_excel.xlsx") # network_properties_full_excel.xlsx")   # ← adjust path if needed""data/network_properties_full_excel.xlsx")

ORDERED_DATASETS = [
    'hcp_schaefer_100_dataset_gnm', 
    'suarez_MaMI_dataset', 
    'lexis_data_developing', 
    'kaysons_generated_networks_diffusion', 
    'kaysons_generated_networks_propagation', 
    'kaysons_generated_networks_routing', 
    "ring_lattice_networks",
    "erdos_renyi_networks",
]

# For plots including everything (like Kayson and I had originally talked about)
INCLUDE_MC_IN_PCA, USE_REMAINING_CATEGORIES_ONLY = True, True #  False, True 
appendix = "main_selected_props_with_mc"

# # For plots on impact of structure on MC / IPC / spectral stuff
# INCLUDE_MC_IN_PCA, USE_REMAINING_CATEGORIES_ONLY = False, True 
# appendix = "main_selected_props_without_mc"

# # # For plot in appendix that uses ALL PROPERTIES
# INCLUDE_MC_IN_PCA, USE_REMAINING_CATEGORIES_ONLY = True, False #  False, True 
# appendix = "apdx_all_properties"

# Toggle MC integration in PCA here
# INCLUDE_MC_IN_PCA = False

# # Toggle feature set: False = all active structural features; True = only remaining_categories from config.py
# USE_REMAINING_CATEGORIES_ONLY = True # False


# In[11]:


def load_and_merge_datasets(dataset_names, data_path):
    """Loads datasets from a pickle file and merges them into a single DataFrame."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    with open(data_path, "rb") as f:
        datasets_dict = pickle.load(f)

    huge_df = pd.DataFrame()
    for dataset_name in dataset_names: 
        if dataset_name not in datasets_dict:
            continue
            
        df = datasets_dict[dataset_name].copy()
        df["dataset"] = dataset_name
        
        if dataset_name in ["kaysons_generated_networks_topology", "hcp_schaefer_100_dataset"]: 
            continue

        if dataset_name == "hcp_schaefer_100_dataset_gnm":
            # eta/gamma are excluded from precise_categories pkl — load them separately
            eta_gamma_path = OUTPUT_FOLDER / "hcp_schaefer_100_dataset_gnm_eta_and_gamma.pkl"
            with open(eta_gamma_path, "rb") as f:
                df_eta_gamma = pickle.load(f)
            df = df.join(df_eta_gamma[["eta", "gamma"]])
            df["color_dataset"] = get_combined_colors(df)
        else: 
            df["color_dataset"] = COLOR_SCHEME.get(dataset_name, "#000000")

        huge_df = pd.concat([huge_df, df], ignore_index=True)
        
    return huge_df


def get_feature_lists(df_columns, include_mc=False):
    """Separates topological features from functional/MC features."""
    mc_keywords = ['computational_capacity', 'mc_', 'ipc_', "repertoire", "spectral"]
    mc_features = [c for c in df_columns if any(kw in c for kw in mc_keywords)]
    
    if include_mc:
        return df_columns, []
    
    active_features = [c for c in df_columns if c not in mc_features]
    return active_features, mc_features

# ==========================================
# 3. Visualization (Deduplicated)
# ==========================================

def plot_clustered_correlation_matrix(df_features, meta, output_filename, title_suffix=""):
    """Deduplicated function to plot hierarchically clustered correlation matrices."""
    cols_in_data = [c for c in df_features.columns if c in meta.index]
    print("cols_in_data:", cols_in_data)
    def sort_key(col):
        row = meta.loc[col]
        print(col, row)
        cat = row["Category"] if pd.notna(row["Category"]) else "ZZZ"
        cat_idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
        return (cat_idx,) #  str(row["section"]))

    cols_sorted_by_cat = sorted(cols_in_data, key=sort_key)
    corr_full = df_features[cols_in_data].corr()

    final_order = []
    for cat in CATEGORY_ORDER:
        cat_cols = [c for c in cols_sorted_by_cat if meta.loc[c, "Category"] == cat]
        print(len(cat_cols), "columns in category", cat)
        if not cat_cols: continue
        if len(cat_cols) == 1:
            final_order.extend(cat_cols)
            continue
            
        sub_corr = corr_full.loc[cat_cols, cat_cols].fillna(0)
        dist = np.clip(1 - sub_corr.values, 0, 2)
        np.fill_diagonal(dist, 0)
        Z = linkage(squareform(dist, checks=False), method="average")
        final_order.extend([cat_cols[i] for i in leaves_list(Z)])

    final_order.extend([c for c in cols_in_data if c not in final_order])
    reduced_corr = corr_full.loc[final_order, final_order]
    new_labels = [SELECTED_PROPERTIES_NAMES[c] for c in final_order]

    n = len(final_order)
    fig = plt.figure(figsize=viz.cm_to_inch((18,12)))
    gs = gridspec.GridSpec(1, 1, left=0.22, right=0.97, top=0.97, bottom=0.12)
    ax = fig.add_subplot(gs[0])

    im = ax.imshow(reduced_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", interpolation="none")

    # Category boundaries
    boundaries, prev_cat = [0], meta.loc[final_order[0], "Category"]
    for i, c in enumerate(final_order[1:], start=1):
        curr_cat = meta.loc[c, "Category"] if c in meta.index else None
        if curr_cat != prev_cat:
            boundaries.append(i)
            prev_cat = curr_cat
    boundaries.append(n)

    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=1.2, alpha=0.85)
        ax.axvline(b - 0.5, color="white", linewidth=1.2, alpha=0.85)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(new_labels, fontsize=8)
    ax.tick_params(axis="both", which="both", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("Pearson r", fontsize=9)
    ax.set_title(f"{n} properties {title_suffix}")

    output_path = OUTPUT_FOLDER / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # plt.close()
    plt.show()
    print(f"Saved correlation matrix to {output_path}")

# ==========================================
# 4. PCA & Functional Correlation Pipeline
# ==========================================

def run_pca_and_correlate(huge_df, active_features, excluded_features):
    """Refits PCA on structural features and correlates morphospace position with function."""
    df_active = huge_df[active_features].copy()

    # Handle infinities and NaNs
    df_active.replace([np.inf, -np.inf], np.nan, inplace=True)
    dropped_cols = df_active.columns[df_active.isnull().any()].tolist()
    if dropped_cols:
        print(f"Dropping columns with NaNs: {dropped_cols}")
    df_active.dropna(axis=1, inplace=True)
    used_features = df_active.columns.tolist()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_active)

    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(scaled_data)

    print("\n--- PCA Explained Variance ---")
    for i, var in enumerate(pca.explained_variance_ratio_[:3]):
        print(f"PC{i+1}: {var*100:.2f}%")

    # Correlate morphospace with functional (MC) properties
    if excluded_features:
        print("\n--- Correlating PC Axes with Excluded Function (MC) ---")
        for mc_col in excluded_features:
            if mc_col in huge_df.columns:
                valid_idx = huge_df[mc_col].notna()
                if not valid_idx.any():
                    continue

                pc1_scores = pca_result[valid_idx, 0]
                pc2_scores = pca_result[valid_idx, 1]
                target_mc = huge_df.loc[valid_idx, mc_col]

                r_pc1, p_pc1 = pearsonr(pc1_scores, target_mc)
                r_pc2, p_pc2 = pearsonr(pc2_scores, target_mc)

                print(f"Feature: {mc_col}")

                pc1_flag = " ***" if abs(r_pc1) > 0.3 else ""
                pc2_flag = " ***" if abs(r_pc2) > 0.3 else ""

                print(f"  -> PC1: r = {r_pc1:+.3f} (p = {p_pc1:.3e}){pc1_flag}")
                print(f"  -> PC2: r = {r_pc2:+.3f} (p = {p_pc2:.3e}){pc2_flag}")

    return pca, pca_result, scaled_data, used_features


def _find_corners(pca_result_2d):
    """Return dict of corner name → row index within pca_result_2d."""
    return {
        "top_left":     int(np.argmax( pca_result_2d[:, 1])),
        "bottom_left":  int(np.argmax(-pca_result_2d[:, 0] - pca_result_2d[:, 1])),
        "bottom_right": int(np.argmax( pca_result_2d[:, 0] - pca_result_2d[:, 1])),
    }



def plot_functional_coloring(pca_result, huge_df, features_to_color_list, group_name, output_filename):
    """For each feature: PC1 vs PC2 morphospace re-colored by that feature's value."""
    if not features_to_color_list:
        return

    valid_features = [f for f in features_to_color_list
                      if f in huge_df.columns and huge_df[f].notna().any()]
    if not valid_features:
        return

    ncols = min(len(valid_features), 3)
    nrows = (len(valid_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=viz.cm_to_inch((ncols * 8, nrows * 7)),
                             squeeze=False)

    for idx, feat in enumerate(valid_features):
        ax = axes[idx // ncols][idx % ncols]
        valid_mask = huge_df[feat].notna().values
        feat_vals = huge_df.loc[valid_mask, feat].values

        # All points in gray, then overlay with functional value coloring
        ax.scatter(pca_result[:, 0], pca_result[:, 1],
                   c="#cccccc", s=4, alpha=0.2, linewidths=0, rasterized=True)
        sc = ax.scatter(pca_result[valid_mask, 0], pca_result[valid_mask, 1],
                        c=feat_vals, cmap="RdBu_r", s=6, alpha=0.75, linewidths=0,
                        rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title(SELECTED_PROPERTIES_NAMES.get(feat, feat), fontsize=8)
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(len(valid_features), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Morphospace colored by {group_name}, {len(valid_features)} parameters", fontsize=10, y=1.01)
    output_path = OUTPUT_FOLDER / output_filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    # plt.close(fig)
    plt.show()
    print(f"Saved functional coloring ({group_name}) to {output_path}")


def compute_biplot_loadings(pca, used_features):
    """
    Returns a DataFrame of biplot-scaled loadings (n_features × 2).

    Each loading is scaled by sqrt(eigenvalue) so that arrow length in a biplot
    reflects the variance explained for that variable on that PC.
    Columns: PC1, PC2.  Index: used_features.
    """
    biplot = pca.components_.T * np.sqrt(pca.explained_variance_)   # (n_features, n_components)
    return pd.DataFrame(
        {"PC1": biplot[:, 0], "PC2": biplot[:, 1]},
        index=used_features,
    )


def correlate_gnm_morphospace_with_ipc(huge_df, active_features, ipc_features):
    """
    Refit PCA on GNM networks only, then correlate each PC axis with every IPC/MC feature.
    Tests whether the r≈0.79–0.81 IPC correlation survives in the GNM-only morphospace.
    """
    gnm_mask = huge_df["dataset"] == "hcp_schaefer_100_dataset_gnm"
    gnm_df   = huge_df[gnm_mask].copy()
    print(f"\n=== GNM-only morphospace: {gnm_mask.sum()} networks ===")

    # PCA on structural features, GNM rows only
    df_active = gnm_df[active_features].copy()
    df_active.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_active.dropna(axis=1, inplace=True)
    usable_features = df_active.columns.tolist()

    scaler = StandardScaler()
    pca    = PCA(n_components=10)
    pca_result = pca.fit_transform(scaler.fit_transform(df_active))

    print("--- GNM PCA Explained Variance ---")
    for i, var in enumerate(pca.explained_variance_ratio_[:3]):
        print(f"  PC{i+1}: {var*100:.2f}%")

    # Correlate every PC axis with every IPC/MC feature
    print("\n--- IPC/MC correlations within GNM morphospace ---")
    rows = []
    for feat in ipc_features:
        if feat not in gnm_df.columns:
            continue
        valid = gnm_df[feat].notna().values
        if valid.sum() < 10:
            continue
        target = gnm_df.loc[gnm_df.index[valid], feat].values
        for pc_idx in range(3):
            scores = pca_result[valid, pc_idx]
            r, p   = pearsonr(scores, target)
            rows.append({"feature": feat, "pc": f"PC{pc_idx+1}", "r": r, "p": p})
            flag = " ***" if abs(r) > 0.3 else ""
            print(f"  {feat:55s}  PC{pc_idx+1}: r={r:+.3f}  p={p:.2e}{flag}")

    return pd.DataFrame(rows)


def plot_axis_correlations(pca_result, huge_df, excluded_properties, output_filename):
    """Grouped bar chart: Pearson r of PC1 and PC2 vs each excluded functional feature."""
    all_features = [("Memory\nCapacity", f) for f in excluded_properties]

    results = []
    for group, feat in all_features:
        if feat not in huge_df.columns:
            continue
        valid_idx = huge_df[feat].notna()
        if not valid_idx.any():
            continue
        pc1 = pca_result[valid_idx, 0]
        pc2 = pca_result[valid_idx, 1]
        target = huge_df.loc[valid_idx, feat]
        r1, p1 = pearsonr(pc1, target)
        r2, p2 = pearsonr(pc2, target)
        results.append({
            "feature": SELECTED_PROPERTIES_NAMES.get(feat, feat),
            "group": group,
            "PC1_r": r1, "PC1_p": p1,
            "PC2_r": r2, "PC2_p": p2,
        })

    if not results:
        return

    df_res = pd.DataFrame(results)
    n = len(df_res)
    x = np.arange(n)
    width = 0.35

    def sig_star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((max(14, n * 2.8), 8)))

    bars1 = ax.bar(x - width / 2, df_res["PC1_r"], width,
                   label="PC1", color="#2C7D8F", alpha=0.85)
    bars2 = ax.bar(x + width / 2, df_res["PC2_r"], width,
                   label="PC2", color="#C74800", alpha=0.85)

    for bar, p in zip(bars1, df_res["PC1_p"]):
        star = sig_star(p)
        if star:
            y = bar.get_height() + (0.03 if bar.get_height() >= 0 else -0.06)
            ax.text(bar.get_x() + bar.get_width() / 2, y, star,
                    ha="center", va="bottom", fontsize=7)
    for bar, p in zip(bars2, df_res["PC2_p"]):
        star = sig_star(p)
        if star:
            y = bar.get_height() + (0.03 if bar.get_height() >= 0 else -0.06)
            ax.text(bar.get_x() + bar.get_width() / 2, y, star,
                    ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(0.3,  color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axhline(-0.3, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)

    # Shade alternating groups
    group_changes = ([0]
                     + [i for i in range(1, n) if df_res["group"].iloc[i] != df_res["group"].iloc[i - 1]]
                     + [n])
    for gi in range(len(group_changes) - 1):
        start, end = group_changes[gi], group_changes[gi + 1]
        if gi % 2 == 1:
            ax.axvspan(start - 0.5, end - 0.5, color="gray", alpha=0.07, zorder=0)
        ax.text((start + end - 1) / 2, 1.02, df_res["group"].iloc[start],
                ha="center", va="bottom", fontsize=7, color="gray", style="italic",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(df_res["feature"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation of PCA Axes with Functional Features")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8)

    output_path = OUTPUT_FOLDER / output_filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    # plt.close(fig)
    plt.show()
    print(f"Saved axis correlations to {output_path}")



# In[12]:


# 1. Load Data
print("Loading datasets...")
huge_df = load_and_merge_datasets(ORDERED_DATASETS, DATA_PATH)

# Base property columns (dropping metadata)
raw_properties = [c for c in huge_df.columns if c not in ["dataset", "color_dataset"]]

# Load metadata
# meta_raw = pd.read_excel(EXCEL_META_PATH, sheet_name=0)
# meta_raw["section"] = meta_raw.apply(
#     lambda r: r["#"] if pd.isna(r["Variable Name"]) and pd.notna(r["#"]) else None, axis=1
# ).ffill()
# meta = meta_raw[meta_raw["Variable Name"].notna()][["Variable Name", "Category", "section"]].rename(columns={"Variable Name": "var"}).set_index("var")

# UPDATE META 

# Build a reverse mapping: var -> category
var_to_category = {
    var: category
    for category, vars in CLUSTERS.items()
    for var in vars
}

# meta["Category"] =  meta.index.map(var_to_category)
meta = pd.DataFrame()

meta["var"] = var_to_category.keys()
meta["Category"] = var_to_category.values()
meta.set_index("var", inplace=True)

# 2. Filter Properties based on precise categories
if USE_REMAINING_CATEGORIES_ONLY:
    rc_flat = [col for cat_cols in remaining_categories.values() for col in cat_cols]
    precise_categories = [c for c in raw_properties if c in rc_flat]
    print(f"Using remaining_categories only: {len(precise_categories)} features")
else:
    precise_categories = [c for c in raw_properties if c in meta.index]

active_features, excluded_features = get_feature_lists(precise_categories, include_mc=INCLUDE_MC_IN_PCA)

# drop effective_dimensionality from active_features: 
active_features.remove("effective_dimensionality")

print(f"Active Structural Features: {len(active_features)}")
print(f"Excluded Functional Features (MC): {len(excluded_features)}")



# In[13]:


meta


# In[14]:


set(meta) - set(active_features)


# In[15]:


# 3. Correlation Matrix Plotting
print("Generating correlation matrices...")
plot_clustered_correlation_matrix(huge_df[active_features], meta, f"corr_matrix_categorised_clean_{appendix}.pdf")

# # Flipping specific properties for better clustering logic
# properties_to_flip = [
#     "rich_club_coefficient_rc_k_at_max", 
#     "directed_simplices_count",
#     "participation_coefficient_pc_frac_connector",
#     "community_synchronization_vulnerability_n_communities",
#     "participation_coefficient_n_communities",
#     "repertoire_sweep_weighted_by_distances_diversity_critical"
# ]
# for prop in properties_to_flip:
#     if prop in huge_df.columns:
#         huge_df[prop] = -huge_df[prop]
        
plot_clustered_correlation_matrix(huge_df[active_features], meta, f"corr_matrix_categorised_flipped_clean_{appendix}.pdf", title_suffix="(Flipped)")


# In[16]:


# ── 2. Sort variables by category, then section ───────────────────────────────
# cols_in_data = [c for c in huge_df.columns if c in meta.index]

def sort_key(col):
    row = meta.loc[col]
    cat = row["Category"] if pd.notna(row["Category"]) else "ZZZ"
    cat_idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (cat_idx,) #  str(row["section"]))

cols_sorted_by_cat = sorted(active_features, key=sort_key)

# ── 3. Within-category hierarchical clustering ────────────────────────────────
corr_full = huge_df[active_features].corr()

final_order = []
for cat in CATEGORY_ORDER:
    cat_cols = [c for c in cols_sorted_by_cat if meta.loc[c, "Category"] == cat]
    if len(cat_cols) == 0:
        continue
    if len(cat_cols) == 1:
        final_order.extend(cat_cols)
        continue
    sub_corr = corr_full.loc[cat_cols, cat_cols].fillna(0)
    dist = np.clip(1 - sub_corr.values, 0, 2)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    final_order.extend([cat_cols[i] for i in leaves_list(Z)])

# Append any uncategorised variables at the end
final_order.extend([c for c in active_features if c not in final_order])

# ── 4. Reorder & label ────────────────────────────────────────────────────────
reduced_corr = corr_full.loc[final_order, final_order]
new_labels = [SELECTED_PROPERTIES_NAMES.get(c, c) for c in final_order]

# ── 5. Plot ───────────────────────────────────────────────────────────────────
n = len(final_order)
fig = plt.figure(figsize=viz.cm_to_inch((18,12)))

gs = gridspec.GridSpec(1, 1, left=0.22, right=0.97, top=0.97, bottom=0.12)
ax = fig.add_subplot(gs[0])

im = ax.imshow(
    reduced_corr.values,
    cmap="RdBu_r", vmin=-1, vmax=1,
    aspect="equal", 
    interpolation="none",
)

# White dividers between categories
boundaries, prev_cat = [0], meta.loc[final_order[0], "Category"]
for i, c in enumerate(final_order[1:], start=1):
    curr_cat = meta.loc[c, "Category"] if c in meta.index else None
    if curr_cat != prev_cat:
        boundaries.append(i)
        prev_cat = curr_cat
boundaries.append(n)

for b in boundaries[1:-1]:
    ax.axhline(b - 0.5, color="white", linewidth=1.2, alpha=0.85)
    ax.axvline(b - 0.5, color="white", linewidth=1.2, alpha=0.85)

# Tick labels
# ax.set_xticks(np.arange(n))
# ax.set_xticklabels(new_labels, rotation=90, fontsize=8, ha="right")
ax.set_yticks(np.arange(n))
ax.set_yticklabels(new_labels, fontsize=8)
ax.tick_params(axis="both", which="both", length=0)

# ── 6. Category brackets on the left ─────────────────────────────────────────
moving_stuff = -0.5
BRACKET_X = -0.20 + moving_stuff
SERIF_W   = 0.005 
LABEL_X   = -0.21 + moving_stuff
trans     = ax.transAxes

cat_row_spans = {}
for row_i, c in enumerate(final_order):
    cat = meta.loc[c, "Category"] if c in meta.index else "Other"
    cat_row_spans.setdefault(cat, [row_i, row_i])[1] = row_i

def row_to_y(row_i, n):
    return 1.0 - (row_i + 0.5) / n

for cat, (r0, r1) in cat_row_spans.items():
    colour  = CATEGORY_COLOURS.get(cat, "#444444")
    weird_adapter = 0.00085
    y_top   = row_to_y(r0, n) + 0.5 / n + weird_adapter
    y_bot   = row_to_y(r1, n) - 0.5 / n - weird_adapter
    y_mid   = (y_top + y_bot) / 2
    kw      = dict(xycoords=trans, textcoords=trans, annotation_clip=False,
                   arrowprops=dict(arrowstyle="-", color=colour, lw=1.5))

    ax.annotate("", xy=(BRACKET_X, y_bot),       xytext=(BRACKET_X, y_top),   **kw)  # vertical bar
    ax.annotate("", xy=(BRACKET_X, y_top),        xytext=(BRACKET_X+SERIF_W, y_top), **kw)  # top serif
    ax.annotate("", xy=(BRACKET_X, y_bot),        xytext=(BRACKET_X+SERIF_W, y_bot), **kw)  # bottom serif
    ax.text(LABEL_X, y_mid, cat, transform=trans,
            ha="right", va="center", fontsize=8, fontweight="bold",
            color=colour, clip_on=False)

cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label("Pearson r", fontsize=9)
ax.set_title(f"{n} properties", # Correlation Matrix - {n} metrics (ordered by category, clustered within)",
            #  fontsize=11, 
            #  pad=8
             )

# plt.savefig(output_folder / "corr_matrix_categorised_flipped.pdf", dpi=150, bbox_inches="tight")
# print(output_folder / "corr_matrix_categorised_flipped.pdf")
plt.show()


# In[ ]:


# 4. Run PCA & Test Morphospace against Function (all datasets)
print("Running PCA (all datasets)...")
pca, pca_result, scaled_data, used_features = run_pca_and_correlate(huge_df, active_features, excluded_features)

suffix = "_remaining" if USE_REMAINING_CATEGORIES_ONLY else "_all"


def plot_pca_morphospace(pca_result, huge_df, pca, output_filename):
    """PC1 vs PC2 scatter with highlighted corners and special-dataset points."""
    GNM_DATASET = "hcp_schaefer_100_dataset_gnm"
    SPECIAL_DATASETS = [
        "kaysons_generated_networks_routing",
        "kaysons_generated_networks_diffusion",
        "kaysons_generated_networks_propagation",
        "ring_lattice_networks",
        "erdos_renyi_networks",
    ]

    # ── Corner detection ──────────────────────────────────────────────────────
    corners_full = _find_corners(pca_result)

    # corners_no_gnm: searched only within lexis_developing + MaMI
    subset_mask = (
        (huge_df["dataset"] == "lexis_data_developing") |
        (huge_df["dataset"] == "suarez_MaMI_dataset")
    ).values
    orig_idx_subset = np.where(subset_mask)[0]
    corners_no_gnm_filtered = _find_corners(pca_result[subset_mask])
    corners_no_gnm = {
        name: int(orig_idx_subset[filt_idx])
        for name, filt_idx in corners_no_gnm_filtered.items()
    }

    # special: first row of each special dataset
    special_indices = {
        ds: int(huge_df.index[huge_df["dataset"] == ds][0])
        for ds in SPECIAL_DATASETS
        if (huge_df["dataset"] == ds).any()
    }

    # ── Build highlighted lookup ──────────────────────────────────────────────
    highlighted = {}
    for corner_name, idx in corners_full.items():
        ds = huge_df["dataset"].iloc[idx]
        highlighted[f"Overall · {corner_name.replace('_', ' ')}"] = {
            "idx": idx, "color": COLOR_SCHEME[ds], "source": "full"
        }
    for corner_name, idx in corners_no_gnm.items():
        ds = huge_df["dataset"].iloc[idx]
        highlighted[f"No-GNM · {corner_name.replace('_', ' ')}"] = {
            "idx": idx, "color": COLOR_SCHEME[ds], "source": "no_gnm"
        }
    for ds_key, idx in special_indices.items():
        highlighted[f"Special · {LABEL_MAP[ds_key]}"] = {
            "idx": idx, "color": COLOR_SCHEME[ds_key], "source": "special"
        }

    # ── Background scatter ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=viz.cm_to_inch((12, 9)), dpi=150) # 18, 12)), dpi=150)

    for ds in huge_df["dataset"].unique():
        mask = (huge_df["dataset"] == ds).values
        ax.scatter(
            pca_result[mask, 0], pca_result[mask, 1],
            c=huge_df[mask]["color_dataset"],
            label=LABEL_MAP.get(ds, ds),
            alpha=0.3 if ds == GNM_DATASET else 0.8,
            s=2 if ds == GNM_DATASET else 9,
            edgecolors="none" if ds == GNM_DATASET else "black",
            linewidths=0 if ds == GNM_DATASET else 0.5,
            zorder=1,
            rasterized=True if ds == GNM_DATASET else False,
        )

    # ── Highlighted points ────────────────────────────────────────────────────
    for label, info in highlighted.items():
        idx = info["idx"]
        linewidth = 0.25 if info["source"] == "full" else 1 # 1.4
        ax.scatter(
            pca_result[idx, 0], pca_result[idx, 1],
            color=huge_df["color_dataset"].iloc[idx],
            s=30, # 50,
            zorder=6,
            edgecolors="black",
            linewidths=linewidth,
        )

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    buffer_x = 1
    buffer_y = 1
    ax.set_xlim(ax.get_xlim()[0] - buffer_x, ax.get_xlim()[1] + buffer_x)
    ax.set_ylim(ax.get_ylim()[0] - 2 * buffer_y, ax.get_ylim()[1] + buffer_y)

    plt.tight_layout()
    output_path = OUTPUT_FOLDER / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # plt.close()
    plt.show()
    print(f"Saved PCA morphospace to {output_path}")



plot_pca_morphospace(pca_result, huge_df, pca, f"pca_morphospace{suffix}_{appendix}.pdf")


# In[ ]:


def plot_pca_loadings(pca, used_features, meta, output_filename):
    """Horizontal bar chart of PCA loadings for PC1–PC3, coloured by category."""
    loadings     = pca.components_.T          # (n_features, n_components)
    metric_names = np.array(used_features)
    num_metrics  = len(metric_names)

    fig, axs = plt.subplots(nrows=1, ncols=3,
                            figsize=viz.cm_to_inch((18,9)), # 40, 12)),
                            sharex=False)

    for i in range(3):
        ordered_pc_i    = np.argsort(loadings[:, i])
        loadings_sorted = loadings[ordered_pc_i]
        names_sorted    = metric_names[ordered_pc_i]
        names_long      = [SELECTED_PROPERTIES_NAMES[m] for m in names_sorted] # PROPERTY_NAMES.get(m, m) for m in names_sorted]

        bar_colours = []
        for m in names_sorted:
            cat = meta.loc[m, "Category"] if m in meta.index else None
            bar_colours.append(CATEGORY_COLOURS.get(cat, "#AAAAAA"))

        axs[i].barh(range(num_metrics), loadings_sorted[:, i], color=bar_colours)
        axs[i].axvline(0, color="black", linewidth=0.6)
        axs[i].set_title(f"Loadings for PC{i+1}")
        axs[i].set_yticks([])

        PAD = 0.02
        for j, (val, name, colour) in enumerate(zip(loadings_sorted[:, i], names_long, bar_colours)):
            if val >= 0:
                axs[i].text(-PAD, j, name, ha="right", va="center", fontsize=5, # 6.5, 
                            color=colour)
            else:
                axs[i].text( PAD, j, name, ha="left",  va="center", fontsize=5, # 6.5, 
                            color=colour)

        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["left"].set_visible(False)

    cats_present = [c for c in meta["Category"].value_counts().index if c in CATEGORY_COLOURS]
    handles = [mpatches.Patch(color=CATEGORY_COLOURS[cat], label=cat) for cat in cats_present]
    # fig.legend(handles=handles, title="Category",
    #            bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=8)

    plt.tight_layout()
    output_path = OUTPUT_FOLDER / output_filename
    plt.savefig(output_path, bbox_inches="tight")
    # plt.close()
    plt.show()
    print(f"Saved PCA loadings to {output_path}")





plot_pca_loadings(pca, used_features, meta, f"pca_loadings{suffix}_{appendix}.pdf")


# In[ ]:


# from config import CATEGORY_ORDER, CATEGORY_COLOURS

fig = plt.figure(figsize=viz.cm_to_inch((12, 2)))

handles = [mpatches.Patch(color=CATEGORY_COLOURS[cat], label=cat) for cat in CATEGORY_ORDER]
fig.legend(
    handles=handles,
    title="Category",
    ncol=6,
    loc="center",
    frameon=False
)

plt.axis("off")
output_path = OUTPUT_FOLDER / "category_legend.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()

print(output_path)


# In[ ]:


def plot_scree(pca, output_filename):
    """Bar + cumulative line scree plot."""
    evr = pca.explained_variance_ratio_ * 100
    n   = len(evr)

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,4)), dpi=150)

    # Horizontal line at 90%
    ax.axhline(90, 0.05, 10, color="grey", linestyle="--", lw=0.5) # lw=0.5, alpha=0.4)
    # Ticks on y for 0, 45, 90 
    ax.set_yticks([0, 45, 90])


    ax.bar(range(1, n + 1), evr, color="#394D73", label="Individual")
    ax.plot(range(1, n + 1), np.cumsum(evr), color="#E84653",
            marker="o", markersize=4, linewidth=1.2, label="Cumulative")
    # ax.set_xlabel("Principal Component")
    ax.set_xlabel("PC")
    # ax.set_ylabel("Explained Variance (%)")
    ax.set_ylabel("Explained Var.")
    ax.set_xticks(range(1, n + 1))
    # ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)



    plt.tight_layout()
    output_path = OUTPUT_FOLDER / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # plt.close()
    plt.show()
    print(f"Saved scree plot to {output_path}")



plot_scree(pca, f"pca_scree{suffix}_{appendix}.pdf")


# In[ ]:


def plot_category_quivers(loading_and_cat_df, meta, output_filename, with_labels=False, equal_axes=True):

    """
    Quiver plot of category-aggregate loading vectors in PC1–PC2 space.

    Each category arrow = sum of biplot-scaled loadings for all variables in that category.
    """
    cat_vectors = {}
    for cat in CATEGORY_COLOURS:
        cat_vars = [
            c for c in loading_and_cat_df.index
            if c in meta.index and meta.loc[c, "Category"] == cat
        ]
        if not cat_vars:
            continue
        cat_loads = loading_and_cat_df.loc[cat_vars]
        vx = cat_loads["PC1"].sum()
        vy = cat_loads["PC2"].sum()
        cat_vectors[cat] = (vx, vy)
        
    print(cat_vectors)

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,6)), 
                           dpi=150)

    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
    ax.scatter([0], [0], color="black", s=20, zorder=6)

    for cat, (vx, vy) in cat_vectors.items():
        colour = CATEGORY_COLOURS[cat]
        ax.quiver(
            0, 0, vx, vy,
            angles="xy", scale_units="xy", scale=1,
            color=colour, 
            width=0.018, # 012,
            headwidth=4,
            headlength=5, headaxislength=5,
            zorder=5,
        )
        if with_labels:
            nudge = 1.3 # 1.15
            ax.text(vx * nudge, vy * nudge, cat,
                    ha="center", va="center", color=colour)

    # max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) # * 1.4
    # maximum value in any direction (PC1 or PC2)
    max_val = max([max(abs(vx), abs(vy)) for vx, vy in cat_vectors.values()]) # * 1.4
    # # ax.set_xlim(-max_val, max_val)
    # # ax.set_ylim(-max_val, max_val)
    # ax.set_xlim(-max_val, max_val)
    # ax.set_ylim(-max_val, max_val)
    # if equal_axes:
    #     ax.set_aspect("equal")
    # else:
    #     plt.tight_layout()  # only apply when axes are free to scale

    # if equal_axes:
    #     ax.set_xlim(-max_val, max_val)
    #     ax.set_ylim(-max_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    # else:
    #     ax.set_xlim(-max_val, max_val)
    #     ax.set_ylim(-max_val, max_val)

    max_val = 5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ticks = [-4, -2, 0, 2, 4]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # ax.set_aspect("equal")
    # ax.set_xlabel("PC1", fontsize=9)
    # ax.set_ylabel("PC2", fontsize=9)
    # ax.set_title("Category vectors in PCA space\n(weighted by loading magnitude)", fontsize=9)
    print("Category vectors in PCA space\n(weighted by loading magnitude)")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # plt.tight_layout()
    output_path = OUTPUT_FOLDER / output_filename
    plt.savefig(output_path, bbox_inches="tight")
    # plt.close()
    plt.show()
    print(f"Saved category quivers to {output_path}")


loading_and_cat_df = compute_biplot_loadings(pca, used_features)
with_labels = False
plot_category_quivers(loading_and_cat_df, meta, f"pca_quivers{suffix}_{appendix}_with_labels.pdf", with_labels=True)
with_labels = False
plot_category_quivers(loading_and_cat_df, meta, f"pca_quivers{suffix}_{appendix}_no_labels.pdf", with_labels=False)

plot_category_quivers(loading_and_cat_df, meta, 
                      f"pca_quivers{suffix}_{appendix}_no_labels.pdf", 
                      with_labels=False, 
                      equal_axes=True)


# # GNM without MC & Later correlation (skipped currently, as this affects later code) 

# In[ ]:


# # 5. GNM-only morphospace: refit PCA on GNM rows only, correlate with IPC/MC
# #    ipc_ipc_* columns are GNM-only and not in Excel metadata / remaining_categories,
# #    so we detect them directly from huge_df rather than routing through excluded_features.
# ipc_cols = [c for c in huge_df.columns if "ipc_ipc_" in c]
# print(f"\nFound {len(ipc_cols)} ipc_ipc_* columns for GNM correlation.")
# gnm_ipc_corr_df = correlate_gnm_morphospace_with_ipc(huge_df, active_features, ipc_cols)


# In[ ]:


# 6. Morphospace re-colored by each excluded feature
print("Plotting morphospace colored by functional features...")
plot_functional_coloring(
    pca_result, huge_df, excluded_features, # excluded_comp,
    "Computational / Memory Capacity", f"pca_coloring_comp_capacity_{appendix}.pdf",
)


# In[ ]:


# # 7. Summary bar chart: PC axis correlations with functional features
# print("Plotting axis–feature correlations...")
# plot_axis_correlations(
#     pca_result, huge_df, excluded_features,  # excluded_memory, excluded_comp,
#     f"pca_axis_correlations_{appendix}.pdf",
# )
# # Summary: features with |r| > 0.5 on any PC
# strong = gnm_ipc_corr_df[gnm_ipc_corr_df["r"].abs() > 0.5].sort_values("r", key=abs, ascending=False)
# if not strong.empty:
#     print("\n--- Strong correlations (|r| > 0.5) in GNM morphospace ---")
#     print(strong.to_string(index=False))


# In[ ]:


def find_corners(pca_result_2d):
    """Return a dict of corner name → row index within pca_result_2d."""
    return {
        "top_left":     int(np.argmax( pca_result_2d[:, 1])),
        "bottom_left":  int(np.argmax(-pca_result_2d[:, 0] - pca_result_2d[:, 1])),
        "bottom_right": int(np.argmax( pca_result_2d[:, 0] - pca_result_2d[:, 1])),
    }


# In[ ]:


# ── Corners in the full selected-properties PCA space ────────────────────────
corners_full = find_corners(pca_result)

print("Corner points (full dataset):")
for name, idx in corners_full.items():
    ds = huge_df["dataset"].iloc[idx]
    print(f"  {name}: idx={idx}, PC1={pca_result[idx,0]:.2f}, PC2={pca_result[idx,1]:.2f}, dataset={ds}")


# In[ ]:


print(pca_result.shape[0], len(active_features))  # these must match


# In[ ]:


pca_result


# In[ ]:


scaled_data


# In[ ]:


# sel_scaled = scaled_data


# In[ ]:


cat_order_present = [c for c in CATEGORY_ORDER if c in meta["Category"].values]

def get_cat_values(row_vals: np.ndarray, feature_cols: list, cat_order: list) -> np.ndarray:
    """Return mean of z-scored feature values within each category (one value per cat)."""
    result = []
    for cat in cat_order:
        cols = [c for c in feature_cols if c in meta.index and meta.loc[c, "Category"] == cat]
        if not cols:
            result.append(np.nan)
            continue
        idxs = [feature_cols.index(c) for c in cols]
        result.append(np.nanmean(row_vals[idxs]))
    return np.array(result)


# In[ ]:


huge_df


# In[ ]:


print(scaled_data.shape[1], len(active_features))  # these must match


# In[ ]:


# feature_cols = active_features
meta = meta.loc[active_features]
meta


# In[ ]:


cat_scaled_all = np.array([
    get_cat_values(scaled_data[i], active_features, cat_order_present)
        for i in range(len(scaled_data))
])

# Re-z-score each category column across all networks
cat_scaled_all = np.apply_along_axis(zscore, 0, cat_scaled_all)

cat_ylim = (np.nanmin(cat_scaled_all), np.nanmax(cat_scaled_all))



# In[ ]:


# Only get datasets that are mami or hcp 
mask = (huge_df["dataset"] == "lexis_data_developing").values | (huge_df["dataset"] == "suarez_MaMI_dataset").values
orig_idx = np.where(mask)[0]        # positions in sel_scaled / pca_result

# PCA results restricted to non-GNM networks
pca_result_no_gnm = pca_result[mask]

# Find corners inside the filtered view (0-indexed within the subset)
corners_no_gnm_filtered = find_corners(pca_result_no_gnm)

# ── Remap to original row indices for sel_scaled ──────────────────────────────
# Without this step, e.g. filtered_idx=5 would incorrectly retrieve row 5 of
# sel_scaled (a GNM network), not the 5th non-GNM network.
corners_no_gnm = {
    name: int(orig_idx[filt_idx])
    for name, filt_idx in corners_no_gnm_filtered.items()
}

SPECIAL_DATASETS = [
    "kaysons_generated_networks_routing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "ring_lattice_networks",
    "erdos_renyi_networks",
]

# # Original row index for single-network datasets
special_indices = {
    ds: int(huge_df.index[huge_df["dataset"] == ds][0])
    for ds in SPECIAL_DATASETS
}


# In[ ]:





# In[ ]:


# # ── 2. Updated get_neighbours (unchanged logic, works with cat_scaled_all too) ─
# from sklearn.metrics import pairwise_distances

# def get_neighbours(orig_idx, embed, n=5, exclude_gnm=False, df_label=None, gnm_name=None):
#     dists = pairwise_distances(embed[orig_idx:orig_idx + 1], embed)[0]
#     dists[orig_idx] = np.inf
#     if exclude_gnm and df_label is not None:
#         dists[(df_label["dataset"] == gnm_name).values] = np.inf
#     return np.argsort(dists)[:n].tolist()


# # ── 3. Rebuild panels using category values ───────────────────────────────────

# panels = []

# for corner_name, orig_idx in corners_full.items():
#     ds    = huge_df["dataset"].iloc[orig_idx]
#     color = huge_df["color_dataset"].iloc[orig_idx] #### OR "black" OR COLOR_SCHEME[ds]
#     nb_idx  = get_neighbours(orig_idx, pca_result, n=5)
#     nb_vals = [cat_scaled_all[i] for i in nb_idx]      # ← cat-level rows
#     panels.append((
#         f"Overall · {corner_name.replace('_', ' ')}",
#         cat_scaled_all[orig_idx], 
#         color,                # ← cat-level row
#         f"spider_overall_{corner_name}.pdf",
#         nb_vals,
#     ))

# for corner_name, orig_idx in corners_no_gnm.items():
#     ds    = huge_df["dataset"].iloc[orig_idx]
#     color = COLOR_SCHEME[ds]
#     nb_idx  = get_neighbours(orig_idx, pca_result, n=5,
#                              exclude_gnm=True, df_label=huge_df, gnm_name="hcp_schaefer_100_dataset_gnm")
#     nb_vals = [cat_scaled_all[i] for i in nb_idx]
#     panels.append((
#         f"No-GNM · {corner_name.replace('_', ' ')}",
#         cat_scaled_all[orig_idx], color,
#         f"spider_no_gnm_{corner_name}.pdf",
#         nb_vals,
#     ))

# for ds_key, orig_idx in special_indices.items():
#     label_text = LABEL_MAP[ds_key]
#     color = COLOR_SCHEME[ds_key]
#     nb_idx  = get_neighbours(orig_idx, pca_result, n=5)
#     nb_vals = [cat_scaled_all[i] for i in nb_idx]
#     panels.append((
#         f"Special · {label_text}",
#         cat_scaled_all[orig_idx], color,
#         f"spider_special_{label_text.lower()}.pdf",
#         nb_vals,
#     ))
    
    
# # One empty one - no neighbors either, just to show the axes with category labels and no data. But make it have all rings visible to show the full scale of category values (unlike the others which may have some rings cut off if their max value is low)
# COLOR = "gray" # lightgray 
# panels.append((
#     "Categories only",
#     np.full(len(cat_order_present), np.nan),
#     COLOR,
#     "spider_categories_only.pdf",
#     None,
# ))

# # ── 4. Render ─────────────────────────────────────────────────────────────────

# all_corner_indices = (
#     list(corners_full.values()) +
#     list(corners_no_gnm.values()) +
#     list(special_indices.values())
# )
# # all_vals = np.concatenate([scaled_data[idx, rep_idx] for idx in all_corner_indices])

# # YLIM = (np.floor(all_vals.min()), np.ceil(all_vals.max()))
# YLIM = [-2, 2] # 4] # 10]
# RING_VALS = [-2, 0, 2] # -6, 2] # 

# # ── Spider drawing function ────────────────────────────────────────────────────

# # def make_spider_standalone(values, label, color, feature_labels, title, ylim, filepath):
# #     n = len(values)
# #     angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
# #     angles_closed = angles + angles[:1]
# #     vals_closed   = list(values) + [values[0]]

# #     fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), dpi=50)

# #     # ── Limits & ticks ──
# #     ax.set_ylim(*ylim)
# #     ax.set_yticks(RING_VALS + [0]) # [-2, 0, 2])
# #     ax.set_yticklabels(RING_VALS + [0], fontsize=8, color="gray")

# #     # ── Remove outermost border ring ──
# #     ax.spines["polar"].set_visible(False)

# #     # ── Custom gridlines: draw manually so we can style zero differently ──
# #     ax.yaxis.grid(False)   # turn off auto y-gridlines
# #     ax.xaxis.grid(False)   # turn off auto x-gridlines (spokes we'll redraw)
# #     theta_ring = np.linspace(0, 2 * np.pi, 300)
# #     for r_val in RING_VALS: # [-2, 2]:
# #         ax.plot(theta_ring, np.full(300, r_val),
# #                 color="lightgray", linewidth=0.6, linestyle="--", zorder=0)
# #     # Zero ring — heavier
# #     ax.plot(theta_ring, np.zeros(300),
# #             color="gray", linewidth=1.8, linestyle="-", zorder=1)
# #     # Other rings: 
# #     for r_val in RING_VALS:  # YLIM: # [-2, 2]:
# #         theta_ring = np.linspace(0, 2 * np.pi, 300)
# #         ax.plot(theta_ring, np.full(300, r_val), color="lightgray", linewidth=0.6, linestyle="--", zorder=0)

# #     # Spoke lines
# #     for angle in angles:
# #         ax.plot([angle, angle], YLIM, # [ylim[0], ylim[1]],
# #                 color="lightgray", linewidth=0.5, linestyle="--", zorder=0)


# #     # ── Data ──
# #     ax.plot(angles_closed, vals_closed, color=color, linewidth=2.2, zorder=5)
# #     ax.fill(angles_closed, vals_closed, color=color, alpha=0.20, zorder=4)

# #     # ── Feature labels (spokes) — no angle labels by default ──
# #     ax.set_thetagrids([])          # hide spoke degree labels; add manually below
# #     ax.set_xticklabels([])

# #     ax.set_title(title, pad=16) # , fontsize=10, pad=16, fontweight="bold")

# #     plt.tight_layout()
# #     plt.savefig(filepath, dpi=150, bbox_inches="tight")
# #     plt.show()
# #     plt.close(fig)
# #     print(f"  Saved: {filepath.name}")

# def make_spider_standalone(values, label, color, feature_labels, title, ylim,
#                            filepath, neighbour_vals_list=None,
#                            axis_colors=None, ring_vals=None):
#     n = len(values)
#     angles        = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
#     angles_closed = angles + angles[:1]
#     vals_closed   = list(values) + [values[0]]
    
#     print(max(vals_closed))
#     ring_color = "gray"
#     if max(vals_closed) > 2: # 4:
#         ring_colors = [ring_color] * 5
#         tick_labels = ["-2", "0", "2", "4", "6"]
#         y_max_change = 0 
#     else: 
#     # elif max(vals_closed) > 0: # 2:
#         ring_colors = [ring_color] * 3 + ["white"]
#         tick_labels = ["-2", "0", "2", "", ""]
#         y_max_change = 2
#     # else:
#     #     ring_colors = [ring_color] * 2 + ["white"] * 2
#     #     tick_labels = ["-2", "0", "", "", ""]
#     #     y_max_change = 4

#     fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,6)), subplot_kw=dict(polar=True), dpi=100)
#     ax.set_ylim(*ylim) # y_max)
#     # ax.set_ylim(ylim[0], ylim[1]-y_max_change) # y_max)
#     print(*ylim)
#     ax.spines["polar"].set_visible(False)
#     ax.yaxis.grid(False) 
#     ax.xaxis.grid(False)

#     theta_ring = np.linspace(0, 2 * np.pi, 300)

#     # Gridlines
#     ring_vals = ring_vals if ring_vals is not None else RING_VALS
#     print(ring_vals)
#     for i, r_val in enumerate(ring_vals):   
#         ax.plot(theta_ring, np.full(300, r_val),
#                 color=ring_colors[i], linewidth=0.5, # 0.6, 
#                 linestyle="--", zorder=0)
#         ax.yaxis.set_ticklabels(labels=tick_labels, 
#                                 fontdict={'verticalalignment': 'baseline',
#                                 'horizontalalignment': 'left'}) # center'}) # label_position("right")

#     ax.plot(theta_ring, np.zeros(300),
#             color=axis_colors, linewidth=1, # 1.8, 
#             linestyle="-", zorder=1)
    
#     for angle in angles:
#         ax.plot([angle, angle], [ylim[0], ylim[1]-y_max_change], # ylim,
#                 color=axis_colors, linewidth=0.5, # 0.5, 
#                 linestyle="--", zorder=0)

#     # Neighbour lines
#     alpha_fill = 0.05
#     linewidth  = 1
#     zorder     = 3
#     if neighbour_vals_list is not None:
#         for nb_vals in neighbour_vals_list:
#             nb_closed = list(nb_vals) + [nb_vals[0]]
#             ax.plot(angles_closed, nb_closed,
#                     color=color, #  if not "Overall" in label else "black", 
#                     linewidth=linewidth, 
#                     alpha=1, # 0.25, 
#                     zorder=zorder + 1)
#             ax.fill(angles_closed, nb_closed,
#                     color=color, alpha=alpha_fill, zorder=zorder)

#     # Main polygon
#     ax.fill(angles_closed, vals_closed, color=color, alpha=alpha_fill, zorder=zorder)
#     ax.plot(angles_closed, vals_closed,
#             color="black", # color if not "Overall" in label else "black", 
#             linewidth=linewidth, zorder=zorder + 1)

#     # # ── Category axis labels, colored per category ────────────────────────────
#     # ax.set_thetagrids(np.degrees(angles), labels=feature_labels, fontsize=7)
#     # for i, (label_text, angle) in enumerate(zip(feature_labels, angles)):
#     #     lbl_color = axis_colors[i] if axis_colors is not None else "black"
#     #     # Find the Text object matplotlib just created and recolor it
#     #     for txt in ax.get_xticklabels():
#     #         if txt.get_text() == label_text:
#     #             txt.set_color(lbl_color)
#     #             txt.set_fontweight("bold")
#     #             break
#     # Remove labels 
#     ax.set_thetagrids([])   

#     # x ticks should be gray
#     ax.tick_params(axis='y', labelcolor=axis_colors) # , labelsize=7)

#     ax.set_title(title) # , pad=16, fontsize=9)
#     plt.tight_layout()
#     plt.savefig(filepath, dpi=150, bbox_inches="tight")
#     plt.show()
#     plt.close(fig)
#     print(filepath)
    
    

# all_vals = np.nanmax([v for _, vals, *_ in panels for v in vals])
# dyn_ylim     = (cat_ylim[0], 2 if all_vals <= 2 else 4 if all_vals <= 4 else cat_ylim[1])
# # dyn_ring_vals = [r for r in RING_VALS if r <= dyn_ylim[1]]
# dyn_ring_vals = RING_VALS
# print(dyn_ring_vals)

# output_folder = OUTPUT_FOLDER / "spiders"
# output_folder.mkdir(exist_ok=True)

# for title, vals, color, fname, nb_vals in panels:
#     make_spider_standalone(
#         values          = vals,
#         label           = title,
#         color           = color,
#         feature_labels  = CATEGORY_ORDER, # cat_labels,        # ← category names
#         title           = title,
#         # ylim            = cat_ylim,
#         ylim            = dyn_ylim,
#         ring_vals       = dyn_ring_vals,
#         filepath        = output_folder / fname,
#         neighbour_vals_list = nb_vals,
#         axis_colors     = COLOR, # cat_axis_colors,   # ← per-category label colors
#     )



# In[ ]:


# ── 2. Updated get_neighbours (unchanged logic, works with cat_scaled_all too) ─
from sklearn.metrics import pairwise_distances

def get_neighbours(orig_idx, embed, n=5, exclude_gnm=False, df_label=None, gnm_name=None):
    dists = pairwise_distances(embed[orig_idx:orig_idx + 1], embed)[0]
    dists[orig_idx] = np.inf
    if exclude_gnm and df_label is not None:
        dists[(df_label["dataset"] == gnm_name).values] = np.inf
    return np.argsort(dists)[:n].tolist()


# ── 3. Rebuild panels using category values ───────────────────────────────────

panels = []

for corner_name, orig_idx in corners_full.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = huge_df["color_dataset"].iloc[orig_idx] #### OR "black" OR COLOR_SCHEME[ds]
    nb_idx  = get_neighbours(orig_idx, pca_result, n=5)
    nb_vals = [cat_scaled_all[i] for i in nb_idx]      # ← cat-level rows
    panels.append((
        f"Overall - {corner_name.replace('_', ' ')}",
        cat_scaled_all[orig_idx], 
        color,                # ← cat-level row
        f"spider_overall_{corner_name}.pdf",
        nb_vals,
    ))

for corner_name, orig_idx in corners_no_gnm.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    nb_idx  = get_neighbours(orig_idx, pca_result, n=5,
                             exclude_gnm=True, df_label=huge_df, gnm_name="hcp_schaefer_100_dataset_gnm")
    nb_vals = [cat_scaled_all[i] for i in nb_idx]
    panels.append((
        f"No-GNM - {corner_name.replace('_', ' ')}",
        cat_scaled_all[orig_idx], color,
        f"spider_no_gnm_{corner_name}.pdf",
        nb_vals,
    ))

for ds_key, orig_idx in special_indices.items():
    label_text = LABEL_MAP[ds_key]
    color = COLOR_SCHEME[ds_key]
    nb_idx  = get_neighbours(orig_idx, pca_result, n=5)
    nb_vals = [cat_scaled_all[i] for i in nb_idx]
    panels.append((
        f"Special - {label_text}",
        cat_scaled_all[orig_idx], color,
        f"spider_special_{label_text.lower()}.pdf",
        nb_vals,
    ))
    
    
# One empty one - no neighbors either, just to show the axes with category labels and no data. But make it have all rings visible to show the full scale of category values (unlike the others which may have some rings cut off if their max value is low)
COLOR = "gray" # lightgray 
panels.append((
    "Categories only",
    np.full(len(cat_order_present), np.nan),
    COLOR,
    "spider_categories_only.pdf",
    None,
))

# ── 4. Render ─────────────────────────────────────────────────────────────────

all_corner_indices = (
    list(corners_full.values()) +
    list(corners_no_gnm.values()) +
    list(special_indices.values())
)
# all_vals = np.concatenate([scaled_data[idx, rep_idx] for idx in all_corner_indices])

# YLIM = (np.floor(all_vals.min()), np.ceil(all_vals.max()))
YLIM = [-1, 2] # 4] # 10]
RING_VALS = [-1, 0, 1, 2, 3, 4, 5, 6] # -6, 2] # 

# ── Spider drawing function ────────────────────────────────────────────────────

# def make_spider_standalone(values, label, color, feature_labels, title, ylim, filepath):
#     n = len(values)
#     angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
#     angles_closed = angles + angles[:1]
#     vals_closed   = list(values) + [values[0]]

#     fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), dpi=50)

#     # ── Limits & ticks ──
#     ax.set_ylim(*ylim)
#     ax.set_yticks(RING_VALS + [0]) # [-2, 0, 2])
#     ax.set_yticklabels(RING_VALS + [0], fontsize=8, color="gray")

#     # ── Remove outermost border ring ──
#     ax.spines["polar"].set_visible(False)

#     # ── Custom gridlines: draw manually so we can style zero differently ──
#     ax.yaxis.grid(False)   # turn off auto y-gridlines
#     ax.xaxis.grid(False)   # turn off auto x-gridlines (spokes we'll redraw)
#     theta_ring = np.linspace(0, 2 * np.pi, 300)
#     for r_val in RING_VALS: # [-2, 2]:
#         ax.plot(theta_ring, np.full(300, r_val),
#                 color="lightgray", linewidth=0.6, linestyle="--", zorder=0)
#     # Zero ring — heavier
#     ax.plot(theta_ring, np.zeros(300),
#             color="gray", linewidth=1.8, linestyle="-", zorder=1)
#     # Other rings: 
#     for r_val in RING_VALS:  # YLIM: # [-2, 2]:
#         theta_ring = np.linspace(0, 2 * np.pi, 300)
#         ax.plot(theta_ring, np.full(300, r_val), color="lightgray", linewidth=0.6, linestyle="--", zorder=0)

#     # Spoke lines
#     for angle in angles:
#         ax.plot([angle, angle], YLIM, # [ylim[0], ylim[1]],
#                 color="lightgray", linewidth=0.5, linestyle="--", zorder=0)


#     # ── Data ──
#     ax.plot(angles_closed, vals_closed, color=color, linewidth=2.2, zorder=5)
#     ax.fill(angles_closed, vals_closed, color=color, alpha=0.20, zorder=4)

#     # ── Feature labels (spokes) — no angle labels by default ──
#     ax.set_thetagrids([])          # hide spoke degree labels; add manually below
#     ax.set_xticklabels([])

#     ax.set_title(title, pad=16) # , fontsize=10, pad=16, fontweight="bold")

#     plt.tight_layout()
#     plt.savefig(filepath, dpi=150, bbox_inches="tight")
#     plt.show()
#     plt.close(fig)
#     print(f"  Saved: {filepath.name}")

def make_spider_standalone(values, label, color, feature_labels, title, ylim,
                           filepath, neighbour_vals_list=None,
                           axis_colors=None, ring_vals=None):
    n = len(values)
    angles        = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    vals_closed   = list(values) + [values[0]]
    
    print(max(vals_closed))
    ring_color = "gray"
    if max(vals_closed) > 2: # 4:
        ring_colors = [ring_color] * len(RING_VALS) # 5
        tick_labels = RING_VALS # ["-1", "0", "1", "2", "4", "6"]
        y_max_change = 0 
    else: 
    # elif max(vals_closed) > 0: # 2:
        ring_colors = [ring_color] * 3 + ["white"] * (len(RING_VALS) - 3)
        tick_labels = ["-1", "0", "1", "2"] + [""] * (len(RING_VALS) - 4) # , "", ""]
        y_max_change = 2
    # else:
    #     ring_colors = [ring_color] * 2 + ["white"] * 2
    #     tick_labels = ["-2", "0", "", "", ""]
    #     y_max_change = 4

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,6)), subplot_kw=dict(polar=True), dpi=100)
    ax.set_ylim(*ylim) # y_max)
    # ax.set_ylim(ylim[0], ylim[1]-y_max_change) # y_max)
    print(*ylim)
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False) 
    ax.xaxis.grid(False)

    theta_ring = np.linspace(0, 2 * np.pi, 300)

    # Gridlines
    ring_vals = ring_vals if ring_vals is not None else RING_VALS
    print(ring_vals)
    for i, r_val in enumerate(ring_vals):   
        ax.plot(theta_ring, np.full(300, r_val),
                color=ring_colors[i], linewidth=0.5, # 0.6, 
                linestyle="--", zorder=0)
        ax.yaxis.set_ticklabels(labels=tick_labels, 
                                fontdict={'verticalalignment': 'baseline',
                                'horizontalalignment': 'left'}) # center'}) # label_position("right")

    ax.plot(theta_ring, np.zeros(300),
            color=axis_colors, linewidth=1, # 1.8, 
            linestyle="-", zorder=1)
    
    for angle in angles:
        ax.plot([angle, angle], [ylim[0], ylim[1]-y_max_change], # ylim,
                color=axis_colors, linewidth=0.5, # 0.5, 
                linestyle="--", zorder=0)

    # Neighbour lines
    alpha_fill = 0.05
    linewidth  = 0.5
    zorder     = 3
    if neighbour_vals_list is not None:
        for nb_vals in neighbour_vals_list:
            nb_closed = list(nb_vals) + [nb_vals[0]]
            ax.plot(angles_closed, nb_closed,
                    color=color, #  if not "Overall" in label else "black", 
                    linewidth=linewidth, 
                    alpha=1, # 0.25, 
                    zorder=zorder + 1)
            ax.fill(angles_closed, nb_closed,
                    color=color, alpha=alpha_fill, zorder=zorder)

    # Main polygon
    ax.fill(angles_closed, vals_closed, color=color, alpha=alpha_fill, zorder=zorder)
    ax.plot(angles_closed, vals_closed,
            color="black", # color if not "Overall" in label else "black", 
            linewidth=linewidth, 
            zorder=zorder + 1)

    # ── Category axis labels, colored per category ────────────────────────────
    ax.set_thetagrids(np.degrees(angles), labels=feature_labels, fontsize=7)
    for i, (label_text, angle) in enumerate(zip(feature_labels, angles)):
        lbl_color = axis_colors[i] if axis_colors is not None else "black"
        # Find the Text object matplotlib just created and recolor it
        for txt in ax.get_xticklabels():
            if txt.get_text() == label_text:
                txt.set_color(lbl_color)
                txt.set_fontweight("bold")
                break
    # Remove labels 
    ax.set_thetagrids([])   

    # x ticks should be gray
    ax.tick_params(axis='y', labelcolor=axis_colors) # , labelsize=7)

    ax.set_title(title) # , pad=16, fontsize=9)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(filepath)
    
    

all_vals = np.nanmax([v for _, vals, *_ in panels for v in vals])
dyn_ylim     = (cat_ylim[0], 2 if all_vals <= 2 else 4 if all_vals <= 4 else cat_ylim[1])
# dyn_ring_vals = [r for r in RING_VALS if r <= dyn_ylim[1]]
dyn_ring_vals = RING_VALS
print(dyn_ring_vals)

output_folder = OUTPUT_FOLDER / "spiders"
output_folder.mkdir(exist_ok=True)

for title, vals, color, fname, nb_vals in panels:
    make_spider_standalone(
        values          = vals,
        label           = title,
        color           = color,
        feature_labels  = CATEGORY_ORDER, # cat_labels,        # ← category names
        title           = title,
        # ylim            = cat_ylim,
        ylim            = (-4, dyn_ylim[1]),
        ring_vals       = dyn_ring_vals,
        filepath        = output_folder / fname,
        neighbour_vals_list = nb_vals,
        axis_colors     = COLOR, # cat_axis_colors,   # ← per-category label colors
    )



# In[ ]:


# FILLED 

# ── 2. Updated get_neighbours (unchanged logic, works with cat_scaled_all too) ─
from sklearn.metrics import pairwise_distances

def get_neighbours(orig_idx, embed, n=5, exclude_gnm=False, df_label=None, gnm_name=None):
    dists = pairwise_distances(embed[orig_idx:orig_idx + 1], embed)[0]
    dists[orig_idx] = np.inf
    if exclude_gnm and df_label is not None:
        dists[(df_label["dataset"] == gnm_name).values] = np.inf
    return np.argsort(dists)[:n].tolist()


# ── 3. Rebuild panels using category values ───────────────────────────────────

panels = []

for corner_name, orig_idx in corners_full.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = huge_df["color_dataset"].iloc[orig_idx] #### OR "black" OR COLOR_SCHEME[ds]
    nb_idx  = get_neighbours(orig_idx, pca_result, n=5)
    nb_vals = [cat_scaled_all[i] for i in nb_idx]      # ← cat-level rows
    panels.append((
        f"Overall - {corner_name.replace('_', ' ')}",
        cat_scaled_all[orig_idx], 
        color,                # ← cat-level row
        f"spider_overall_{corner_name}.pdf",
        nb_vals,
    ))

for corner_name, orig_idx in corners_no_gnm.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    nb_idx  = get_neighbours(orig_idx, pca_result, n=5,
                             exclude_gnm=True, df_label=huge_df, gnm_name="hcp_schaefer_100_dataset_gnm")
    nb_vals = [cat_scaled_all[i] for i in nb_idx]
    panels.append((
        f"No-GNM - {corner_name.replace('_', ' ')}",
        cat_scaled_all[orig_idx], color,
        f"spider_no_gnm_{corner_name}.pdf",
        nb_vals,
    ))

for ds_key, orig_idx in special_indices.items():
    label_text = LABEL_MAP[ds_key]
    color = COLOR_SCHEME[ds_key]
    nb_idx  = get_neighbours(orig_idx, pca_result, n=5)
    nb_vals = [cat_scaled_all[i] for i in nb_idx]
    panels.append((
        f"Special - {label_text}",
        cat_scaled_all[orig_idx], color,
        f"spider_special_{label_text.lower()}.pdf",
        nb_vals,
    ))
    
    
# One empty one - no neighbors either, just to show the axes with category labels and no data. But make it have all rings visible to show the full scale of category values (unlike the others which may have some rings cut off if their max value is low)
COLOR = "gray" # lightgray 
panels.append((
    "Categories only",
    np.full(len(cat_order_present), np.nan),
    COLOR,
    "spider_categories_only.pdf",
    None,
))

# ── 4. Render ─────────────────────────────────────────────────────────────────

all_corner_indices = (
    list(corners_full.values()) +
    list(corners_no_gnm.values()) +
    list(special_indices.values())
)
# all_vals = np.concatenate([scaled_data[idx, rep_idx] for idx in all_corner_indices])

# YLIM = (np.floor(all_vals.min()), np.ceil(all_vals.max()))
YLIM = [-1, 2] # 4] # 10]
RING_VALS = [-1, 0, 1, 2, 3, 4, 5, 6] # -6, 2] # 


def make_spider_standalone(values, label, color, feature_labels, title, ylim,
                           filepath, neighbour_vals_list=None,
                           axis_colors=None, ring_vals=None):
    n = len(values)
    angles        = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    vals_closed   = list(values) + [values[0]]
    
    print(max(vals_closed))
    ring_color = "gray"
    if max(vals_closed) > 2: # 4:
        ring_colors = [ring_color] * len(RING_VALS) # 5
        tick_labels = RING_VALS # ["-1", "0", "1", "2", "4", "6"]
        y_max_change = 0 
    else: 
    # elif max(vals_closed) > 0: # 2:
        ring_colors = [ring_color] * 3 + ["white"] * (len(RING_VALS) - 3)
        tick_labels = ["-1", "0", "1", "2"] + [""] * (len(RING_VALS) - 4) # , "", ""]
        y_max_change = 2
    # else:
    #     ring_colors = [ring_color] * 2 + ["white"] * 2
    #     tick_labels = ["-2", "0", "", "", ""]
    #     y_max_change = 4

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,6)), subplot_kw=dict(polar=True), dpi=100)
    ax.set_ylim(*ylim) # y_max)
    # ax.set_ylim(ylim[0], ylim[1]-y_max_change) # y_max)
    print(*ylim)
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False) 
    ax.xaxis.grid(False)

    theta_ring = np.linspace(0, 2 * np.pi, 300)

    # Neighbour lines
    alpha_fill = 0.05
    linewidth  = 0.5
    zorder     = 3

    ########################################################################################
    # ONCE FOR THE WHITE FILLING 
    ########################################################################################

    # Neighbors 
    if neighbour_vals_list is not None:
        for nb_vals in neighbour_vals_list:
            nb_closed = list(nb_vals) + [nb_vals[0]]
            ax.plot(angles_closed, nb_closed,
                    color=color, #  if not "Overall" in label else "black", 
                    linewidth=linewidth, 
                    alpha=1, # 0.25, 
                    zorder=zorder + 1)
            ax.fill(angles_closed, nb_closed,
                    color="white", alpha=1, 
                    zorder=zorder)


    # Main polygon
    ax.fill(angles_closed, vals_closed, color="white", alpha=1, zorder=zorder)
    ax.plot(angles_closed, vals_closed,
            color="black", # color if not "Overall" in label else "black", 
            linewidth=linewidth, 
            zorder=zorder + 1)


    ########################################################################################
    # ONCE FOR THE RINGS 
    ########################################################################################

    # Neighbour lines
    alpha_fill = 0.05
    linewidth  = 0.5
    zorder     = 3
    if neighbour_vals_list is not None:
        for nb_vals in neighbour_vals_list:
            nb_closed = list(nb_vals) + [nb_vals[0]]
            ax.plot(angles_closed, nb_closed,
                    color=color, #  if not "Overall" in label else "black", 
                    linewidth=linewidth, 
                    alpha=1, # 0.25, 
                    zorder=zorder + 1)
            ax.fill(angles_closed, nb_closed,
                    color=color, alpha=alpha_fill, zorder=zorder)


    # Main polygon
    ax.fill(angles_closed, vals_closed, color=color, alpha=alpha_fill, zorder=zorder)
    ax.plot(angles_closed, vals_closed,
            color="black", # color if not "Overall" in label else "black", 
            linewidth=linewidth, 
            zorder=zorder + 1)


    ########################################################################################


    # Remove labels 
    ax.set_thetagrids([])   

    # x ticks should be gray
    ax.tick_params(axis='y', labelcolor=axis_colors) # , labelsize=7)


    # Gridlines
    zorder_grid = 10 
    ring_vals = ring_vals if ring_vals is not None else RING_VALS
    print(ring_vals)
    for i, r_val in enumerate(ring_vals):   
        ax.plot(theta_ring, np.full(300, r_val),
                color=ring_colors[i], linewidth=0.5, # 0.6, 
                linestyle="--", zorder=zorder_grid)
        ax.yaxis.set_ticklabels(labels=tick_labels, 
                                fontdict={'verticalalignment': 'baseline',
                                'horizontalalignment': 'left'}) # center'}) # label_position("right")

    ax.plot(theta_ring, np.zeros(300),
            color=axis_colors, linewidth=1, # 1.8, 
            linestyle="-", zorder=zorder_grid)
    
    for angle in angles:
        ax.plot([angle, angle], [ylim[0], ylim[1]-y_max_change], # ylim,
                color=axis_colors, linewidth=0.5, # 0.5, 
                linestyle="--", zorder=zorder_grid)



    ax.set_title(title) # , pad=16, fontsize=9)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(filepath)
    
    

all_vals = np.nanmax([v for _, vals, *_ in panels for v in vals])
dyn_ylim     = (cat_ylim[0], 2 if all_vals <= 2 else 4 if all_vals <= 4 else cat_ylim[1])
# dyn_ring_vals = [r for r in RING_VALS if r <= dyn_ylim[1]]
dyn_ring_vals = RING_VALS
print(dyn_ring_vals)

output_folder = OUTPUT_FOLDER / "spiders_new"
output_folder.mkdir(exist_ok=True)

for title, vals, color, fname, nb_vals in panels:
    make_spider_standalone(
        values          = vals,
        label           = title,
        color           = color,
        feature_labels  = CATEGORY_ORDER, # cat_labels,        # ← category names
        title           = title,
        # ylim            = cat_ylim,
        ylim            = (-4, dyn_ylim[1]),
        ring_vals       = dyn_ring_vals,
        filepath        = output_folder / fname,
        neighbour_vals_list = nb_vals,
        axis_colors     = COLOR, # cat_axis_colors,   # ← per-category label colors
    )



# In[ ]:


def make_legend_plot(feature_labels, angles, filepath, label_color=None):
    """Empty radar with only the spoke labels visible."""
    
    n = len(feature_labels)
    angles_plot = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((9,6)), subplot_kw=dict(polar=True), dpi=150)
    YLIM = (-4, 4) 
    print(YLIM)
    # fig, ax = plt.subplots(figsize=viz.cm_to_inch((12,12)), subplot_kw=dict(polar=True), dpi=150)
    ax.set_ylim(YLIM[0], YLIM[1]) # -3.5, 3.5
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    tick_labels = [-2,0,2]
    
    # Draw spoke lines
    for angle in angles_plot:
        ax.plot([angle, angle], YLIM, # [-3.5, 3.5],
                color=label_color, linewidth=0.5, # 1.8, # 2, # 0.5, 
                linestyle="--", zorder=0)
        
    # ── Group arcs just outside the outermost ring ──
    local_min, local_max = YLIM[0], YLIM[1]
    arc_r = local_max + (local_max - local_min) * 0.12
    # draw_axis_group_arcs(ax, list(feature_labels), angles, arc_r, AXIS_GROUPS)
    # Expand ylim slightly so the arc isn't clipped
    ax.set_ylim(local_min, arc_r + (local_max - local_min) * 0.05)

    # Zero ring
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta_ring, np.zeros(300), color=label_color, # "gray", 
            # linewidth=1.8,
            zorder=1)
    # Other rings: 
    print(RING_VALS)
    for r_val in RING_VALS:  # YLIM: # [-2, 2]:
        theta_ring = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_ring, np.full(300, r_val), color=label_color, linewidth=0.5, 
                linestyle="--", zorder=0)
        
        
        # ax.yaxis.set_ticklabels(labels=tick_labels, 
        #                         fontdict={'verticalalignment': 'baseline',
        #                         'horizontalalignment': 'left'}) # center'}) # label_position("right")


    # ── Category axis labels, colored per category ────────────────────────────

    # Replace " " in labels with newlines for better spacing
    feature_labels_to_print = [lbl.replace(" ", "\n") for lbl in feature_labels]
    font_colors = [CATEGORY_COLOURS[cat] for cat in feature_labels]
    
    # Add spoke labels manually
    for i, (angle, lbl) in enumerate(zip(angles_plot, feature_labels_to_print)):
        x = np.degrees(angle)
        ax.set_thetagrids(np.degrees(angles_plot), 
                          feature_labels_to_print,
                        #   fontweight="bold",
                        #   fontsize=8
                          )
        for idx, txt in enumerate(ax.get_xticklabels()):
            if txt.get_text() == lbl:
                txt.set_color(font_colors[i])
                # shift it a bit to the right (away from the spoke line)
                txt.set_horizontalalignment("left")
                txt.set_verticalalignment("center")
                
                if idx in [2,3,4]: 
                    txt.set_horizontalalignment("right")
                # txt.set_fontweight("bold")
                break



        # for i, (label_text, angle) in enumerate(zip(feature_labels_to_print, angles)):
        #     lbl_color = font_colors[i] if font_colors is not None else "black"
        #     # Find the Text object matplotlib just created and recolor it
            

        # ax.set_title("Legend / Labels", pad=16) # , fontsize=10, pad=16, fontweight="bold")
        
    # Remove labels 
    ax.set_thetagrids([])   

    # x ticks should be gray
    ax.tick_params(axis='y', labelcolor=COLOR) # , labelsize=7)

    plt.title("placeholder") 
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(filepath)

angles_for_legend = np.linspace(0, 2 * np.pi, 6, # len(rep_labels), 
                                endpoint=False).tolist()

make_legend_plot(CATEGORY_ORDER, angles_for_legend, filepath=output_folder / "spider_labels_only.pdf", label_color=COLOR)


# 

# In[ ]:


measures_of_interest = [ # "mc_input_scaling_0_1_mc_mean",
                        # "mc_nonlinear_input_scaling_0_1_mc_mean", # "mc_mean", 
                        # "avg_clustering", "spectral_radius"] + 
                        # [
                        "ipc_ipc_deg1_mean", "ipc_ipc_deg2_mean", 
                        "mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean", "spectral_radius", 
                        "avg_clustering", "spectral_radius", "targeted_attack_robustness_rob_targeted_auc",                 
                        "global_efficiency", "modularity", 
                        "proportion_long_range_connections_0.3956", "targeted_attack_robustness_rob_targeted_auc", 
                        "algebraic_connectivity_fiedler_value", "spectral_radius", 
                        # "computational_capacity_memory_capacity_total", "computational_capacity_nonlinear_capacity_total", 
                        "repertoire_sweep_weighted_by_distances_diversity_critical", 
                        "local_efficiency_stats_mean", "betweenness_centrality_stats_mean", "gromov_hyperbolicity", 
                        "synchronizability_eigenratio_eigenratio", "targeted_attack_robustness_rob_random_auc", 
                        
]
# ] + precise_categories 
cmap = plt.get_cmap("managua_r") # cividis") # berlin")

for thing_to_be_colored in measures_of_interest:

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,6)), dpi=150)

    vmin, vmax = huge_df[thing_to_be_colored].min(), huge_df[thing_to_be_colored].max()
    norm = plt.Normalize(vmin, vmax)
    
    for ds in huge_df["dataset"].unique():
        mask = (huge_df["dataset"] == ds).values
        ax.scatter(
            pca_result[mask, 0], pca_result[mask, 1],
            # c=COLOR_SCHEME[ds], 
            # c=huge_df[mask]["color_dataset"], # COLOR_SCHEME[ds], 
            c=huge_df[mask][thing_to_be_colored], # color by the chosen metric instead of dataset
            cmap=cmap,
            label=LABEL_MAP.get(ds, ds),
            alpha=0.3 if "gnm" in ds else 0.8, 
            s=2 if "gnm" in ds else 3, 
            edgecolors="none" if "gnm" in ds else "black", 
            linewidths=0 if "gnm" in ds else 0.2,
            zorder=1,
            norm=norm
        )
        
    ax.spines[["top", "right"]].set_visible(False)
    # ax.set_xlabel("PC1") # , fontsize=10)
    # ax.set_ylabel("PC2") # , fontsize=10)
    ax.set_xticks([]) # Hide x-axis ticks for cleaner look
    ax.set_yticks([]) # Hide y-axis ticks for cleaner look
    # plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), 
    #              ax=ax, 
    #              label=SELECTED_PROPERTIES_NAMES[thing_to_be_colored] if thing_to_be_colored in SELECTED_PROPERTIES_NAMES else thing_to_be_colored,
    #             #  label=PROPERTY_NAMES[thing_to_be_colored], 
    #              orientation='horizontal')
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, 
                 label=SELECTED_PROPERTIES_NAMES.get(thing_to_be_colored, thing_to_be_colored),
                 orientation='horizontal')
    
    # ax.set_title("PCA — All 9 Highlighted Points") # , fontsize=11, fontweight="bold")

    plt.tight_layout(pad=0.01)
    plt.savefig(output_folder / f"pca_colored_by_{thing_to_be_colored}.pdf", dpi=150) # , bbox_inches="tight")
    print(output_folder / f"pca_colored_by_{thing_to_be_colored}.pdf")
    plt.show()

