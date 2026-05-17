#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from vizman import viz
import os
import pickle

import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


print(os.getcwd())  # Should show the project root 

from config import COLORS, COLOR_SCHEME, LABEL_MAP, gray_cmap, bone_white, half_black, emp_dataset_and_experiment_pairs, PROPERTY_NAMES, REPRESENTATIVES_FOR_GOALS


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis")
output_folder.mkdir(exist_ok=True)


with open(output_folder / "all_datasets_precise_categories.pkl", "rb") as f:
    dict_with_all_datasets = pickle.load(f)


# In[3]:


# ordered_dataset_names = ['hcp_schaefer_100_dataset_gnm', 
#                         # 'hcp_schaefer_100_dataset', 
#                         'suarez_MaMI_dataset', 
#                         'lexis_data_young', 
#                         'lexis_data_aging',
#                         'lexis_data_developing', 
#                         'kaysons_generated_networks_diffusion', 
#                         'kaysons_generated_networks_propagation', 
#                         'kaysons_generated_networks_routing', 
#                         # 'kaysons_generated_networks_topology', 
#                         ]

ordered_dataset_names = [
                        'hcp_schaefer_100_dataset_gnm', 
                        # 'hcp_schaefer_100_dataset', 
                        'suarez_MaMI_dataset', 
                        'lexis_data_young', 
                        'lexis_data_aging',
                        'lexis_data_developing', 
                        'kaysons_generated_networks_diffusion', 
                        'kaysons_generated_networks_propagation', 
                        'kaysons_generated_networks_routing', 
                        'kaysons_generated_networks_topology', 
                        ]


# In[4]:


huge_df = pd.DataFrame()

# combine all the metrics into one dataframe for this experiment
for dataset_name in ordered_dataset_names: 
    df = dict_with_all_datasets[dataset_name]
    # Add a column to identify the dataset
    df["dataset"] = dataset_name
    
    print(dataset_name, len(df))
    # Remove this, as eta, gamma, etc are not defined here? 
    if dataset_name == "kaysons_generated_networks_topology" or dataset_name == "hcp_schaefer_100_dataset": 
        print(" ---> skipped")
        continue
    huge_df = pd.concat([huge_df, df], ignore_index=True)


# In[5]:


# For pca always drop the dataset column and only use the metric columns
huge_df_properties = huge_df.drop(columns=["dataset"]) # , "color_dataset"])

# Make a correlation axis between the columns of the huge_df. Don't use the dataset column for this, only the metric columns.
correlation_matrix = huge_df_properties.corr()

# ── 1. Load category / section metadata from Excel ────────────────────────────
excel_path = "/Users/adrian/Desktop/network_properties_full_excel.xlsx" # network_properties_full_excel.xlsx"   # ← adjust path if needed
meta_raw = pd.read_excel(excel_path, sheet_name=0)

meta_raw["section"] = meta_raw.apply(
    lambda r: r["#"] if pd.isna(r["Variable Name"]) and pd.notna(r["#"]) else None,
    axis=1,
).ffill()

meta = (
    meta_raw[meta_raw["Variable Name"].notna()]
    [["Variable Name", "Category", "section"]]
    .rename(columns={"Variable Name": "var"})
    .set_index("var")
)

CATEGORY_ORDER = [
    "Structure", "Integration", "Segregation",
    "Dynamics", "Computation", "Robustness",
]

CATEGORY_COLOURS = {
    "Integration": "#4C72B0",
    "Segregation": "#DD8452",
    "Robustness":  "#55A868",
    "Topology":    "#C44E52",
    "Wiring":      "#8172B2",
    "Geometry":    "#937860",
    "Dynamics":    "#DA8BC3",
    "Computation": "#8C8C8C",
    "Structure":   "#CCB974",
}

# ── 2. Sort variables by category, then section ───────────────────────────────
cols_in_data = [c for c in huge_df_properties.columns if c in meta.index]

def sort_key(col):
    row = meta.loc[col]
    cat = row["Category"] if pd.notna(row["Category"]) else "ZZZ"
    cat_idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (cat_idx, str(row["section"]))

cols_sorted_by_cat = sorted(cols_in_data, key=sort_key)

# ── 3. Within-category hierarchical clustering ────────────────────────────────
corr_full = huge_df_properties[cols_in_data].corr()

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
final_order.extend([c for c in cols_in_data if c not in final_order])

# ── 4. Reorder & label ────────────────────────────────────────────────────────
reduced_corr = corr_full.loc[final_order, final_order]
new_labels = [PROPERTY_NAMES.get(c, c) for c in final_order]

# ── 5. Plot ───────────────────────────────────────────────────────────────────
n = len(final_order)
fig = plt.figure(figsize=viz.cm_to_inch((18,12)))

gs = gridspec.GridSpec(1, 1, left=0.22, right=0.97, top=0.97, bottom=0.12)
ax = fig.add_subplot(gs[0])

im = ax.imshow(
    reduced_corr.values,
    cmap="RdBu", vmin=-1, vmax=1,
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
ax.set_yticklabels(new_labels, fontsize=4)
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

plt.savefig(output_folder / "corr_matrix_categorised.pdf", dpi=150, bbox_inches="tight")
print(output_folder / "corr_matrix_categorised.pdf")
plt.show()


# In[6]:


from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

# ── 1. Full-matrix hierarchical clustering ────────────────────────────────────
cols_in_data = [c for c in huge_df_properties.columns if c in meta.index]
corr_full    = huge_df_properties[cols_in_data].corr()

# dist = np.clip(1 - corr_full.values, 0, 2)
dist = np.clip(1 - np.abs(corr_full.values), 0, 1)  # was: 1 - corr_full.values

np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), method="average")

final_order  = [cols_in_data[i] for i in leaves_list(Z)]
reduced_corr = corr_full.loc[final_order, final_order]
new_labels   = [PROPERTY_NAMES.get(c, c) for c in final_order]

# ── 2. Cut the dendrogram into N clusters & assign colours ───────────────────
N_CLUSTERS = 6  # ← tune this
cluster_ids = fcluster(Z, N_CLUSTERS, criterion="maxclust")
# fcluster returns ids in original column order, remap to final_order
orig_to_cluster = dict(zip(cols_in_data, cluster_ids))

cluster_palette = plt.cm.tab10.colors
def cluster_colour(col):
    cid = orig_to_cluster.get(col, 0)
    return cluster_palette[(cid - 1) % len(cluster_palette)]

# ── 3. Plot ───────────────────────────────────────────────────────────────────
n   = len(final_order)
fig = plt.figure(figsize=viz.cm_to_inch((18, 12)))
gs  = gridspec.GridSpec(1, 1, left=0.22, right=0.97, top=0.97, bottom=0.12)
ax  = fig.add_subplot(gs[0])

im = ax.imshow(
    reduced_corr.values,
    cmap="RdBu", vmin=-1, vmax=1,
    aspect="equal", interpolation="nearest",
)

# ── 4. White dividers between clusters ───────────────────────────────────────
prev_c = orig_to_cluster[final_order[0]]
for i, col in enumerate(final_order[1:], start=1):
    if orig_to_cluster[col] != prev_c:
        ax.axhline(i - 0.5, color="white", linewidth=1.2, alpha=0.85)
        ax.axvline(i - 0.5, color="white", linewidth=1.2, alpha=0.85)
    prev_c = orig_to_cluster[col]

# ── 5. Tick labels ────────────────────────────────────────────────────────────
ax.set_yticks(np.arange(n))
ax.set_yticklabels(new_labels, fontsize=8)
ax.set_xticks([])
ax.tick_params(axis="both", which="both", length=0)

# ── 6. Cluster brackets on the left ──────────────────────────────────────────
moving_stuff = -0.5
BRACKET_X = -0.20 + moving_stuff
SERIF_W   =  0.005
LABEL_X   = -0.21 + moving_stuff
trans     = ax.transAxes

# Build row spans per cluster (in display order)
cluster_row_spans = {}
for row_i, col in enumerate(final_order):
    cid = orig_to_cluster[col]
    cluster_row_spans.setdefault(cid, [row_i, row_i])[1] = row_i

def row_to_y(row_i, n):
    return 1.0 - (row_i + 0.5) / n

for cid, (r0, r1) in cluster_row_spans.items():
    colour = cluster_palette[(cid - 1) % len(cluster_palette)]
    y_top  = row_to_y(r0, n) + 0.5 / n
    y_bot  = row_to_y(r1, n) - 0.5 / n
    y_mid  = (y_top + y_bot) / 2
    kw     = dict(xycoords=trans, textcoords=trans, annotation_clip=False,
                  arrowprops=dict(arrowstyle="-", color=colour, lw=1.5))

    ax.annotate("", xy=(BRACKET_X, y_bot),            xytext=(BRACKET_X, y_top),            **kw)
    ax.annotate("", xy=(BRACKET_X, y_top),             xytext=(BRACKET_X + SERIF_W, y_top),  **kw)
    ax.annotate("", xy=(BRACKET_X, y_bot),             xytext=(BRACKET_X + SERIF_W, y_bot),  **kw)
    ax.text(LABEL_X, y_mid, f"C{cid}", transform=trans,
            ha="right", va="center", fontsize=8, fontweight="bold",
            color=colour, clip_on=False)

cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label("Pearson r", fontsize=9)
ax.set_title(f"{n} properties — {N_CLUSTERS} clusters (hierarchical)")

# plt.savefig(output_folder / f"{dataset_name}_corr_matrix_clustered_automatically.pdf", bbox_inches="tight")
# print(output_folder / f"{dataset_name}_corr_matrix_clustered_automatically.pdf")
plt.show()

# ── 7. Print cluster compositions ─────────────────────────────────────────────
for cid in sorted(cluster_row_spans):
    members = [PROPERTY_NAMES.get(c, c) for c in final_order
               if orig_to_cluster[c] == cid]
    print(f"\nCluster {cid} ({len(members)} vars):")
    print("  " + ", ".join(members))


# In[7]:


import re
from config import PROPERTY_NAMES

# ── Category definitions ──────────────────────────────────────────────────────
remaining_categories = {
    "Fundamental Topology": [ 
        # "transitivity", # corr too highly with avg_clustering
        # "avg_clustering", # omega
        "modularity", 
        # "degree_gini", # too high corr with spectral_radius
        # "degree_assortativity", 
        "omega", 
        # "structural_complexity", 
        
        "directed_simplices_count", # Higher order
        "directed_simplices_max_size",
        
        # "energy" # i.e.: Fit to networks - maybe remove again? 
    ],
    "Paths, Efficiency & Communication": [
        # "char_path_length", 
        "global_efficiency", # corr too highly with omega
        "diffusion_efficiency",
        "propagation_efficiency", # corr too highly with diffusion_efficiency
        # "avg_communicability", # corr too highly with spectral_radius # !!!!!!!
        # "topological_distance_mean", 
        # "topological_distance_std",
    ],
    "Spatial Embedding & Wiring Cost": [
        # "avg_edge_distance", 
        "wiring_cost",
        # "proportion_long_range_connections_0.1",
        # "proportion_long_range_connections_0.3",
        "proportion_long_range_connections_0.3956",
        # "proportion_long_range_connections_0.5",
    ],
    "Rich-Club Organization": [
        # "richclub_n_edges", 
        # "richclub_avg_length",
        # "rich_club_coefficient_rc_max_norm",
        # "rich_club_coefficient_rc_mean_norm",
        "rich_club_coefficient_rc_k_at_max",
        # "rich_club_coefficient_rc_regime_frac",
        # "rich_club_coefficient_rc_weighted_auc",
    ],
    "Spectral Properties, Algebraic Connectivity & Fiedler Analysis": [
        "spectral_radius", 
        "spectral_gap", # Has exactly the same output as Fatemehs version, bc it is identical (lambda_n - lambda_(n-1))
        "synchronizability_eigenratio_eigenratio", # lambda_n / lambda_2

        # "kernel_rank_thresholded_and_summed_0.01",          # np.sum(np.abs(eigs) > threshold * np.abs(eigs).max())
        # "kernel_rank_phase_diff_of_lambda_max_and_2nd",     # np.angle(eigs[np.argsort(np.abs(eigs))[-2]]) - np.angle(eigs.max())
        
        "algebraic_connectivity_fiedler_value", # Relies on a different library, I hope? 
        # "algebraic_connectivity_fiedler_value_norm",
        "algebraic_connectivity_laplacian_spectral_gap",
        # "algebraic_connectivity_fiedler_bipartition_balance",
        
        "departure_from_normality_schur",

        # "effective_dimensionality", # corr too highly with omega
    ],
    "Metastability & Kuramoto Synchronization": [ 
        "repertoire_sweep_weighted_by_distances_T_critical",
        "repertoire_sweep_weighted_by_distances_size_critical",
        "repertoire_sweep_weighted_by_distances_diversity_critical", 
        
        # "kuramoto_averaged_synchronization_r_final",
        # "kuramoto_averaged_synchronization_r_mean",
        # "kuramoto_averaged_synchronization_r_mean_se",
        # "kuramoto_averaged_synchronization_r_std",
        # "kuramoto_averaged_synchronization_r_std_se",
    ],

    "Participation Coefficient (Louvain)": [
        "participation_coefficient_n_communities",
        # "participation_coefficient_pc_mean", # corr too highly with omega
        # "participation_coefficient_pc_median",
        "participation_coefficient_pc_std",
        "participation_coefficient_pc_frac_connector", # needs to be put into relation with Q
        # "participation_coefficient_wmd_std",
    ],
    "Ollivier-Ricci Curvature": [ # If too annoying to describe, keep only one. 
        # "ollivier_ricci_curvature_orc_mean",
        # "ollivier_ricci_curvature_orc_median",
        # "ollivier_ricci_curvature_orc_std", 
        "ollivier_ricci_curvature_orc_min",
        # "ollivier_ricci_curvature_orc_max",
        "ollivier_ricci_curvature_orc_skewness",
        # "ollivier_ricci_curvature_orc_frac_neg",
    ],
    # "Persistent Homology (TDA)": [
    #     # "persistent_homology_ph_h1_n_features",
    #     # "persistent_homology_ph_h1_persistence_mean",
    #     # "persistent_homology_ph_h1_entropy",
    #     # "persistent_homology_ph_total_persistence",
    # ],

    "Targeted Attack Robustness & Community Structure & Vulnerability": [
        "targeted_attack_robustness_rob_targeted_auc",
        # "targeted_attack_robustness_rob_targeted_half",
        # "targeted_attack_robustness_rob_random_auc", # corr too highly with 'char_path_length'
        # "targeted_attack_robustness_rob_random_half",
        "targeted_attack_robustness_rob_ratio",
        
        # "community_synchronization_vulnerability_value", # too high corr with participation_coefficient_pc_frac_connector
        "community_synchronization_vulnerability_n_communities",
        ],
    "Network Control Theory - Average Controllability": [
        # "nct_control_avg",
        "nct_control_std",  
        # "nct_control_max",
        # "nct_control_n_nodes_90_percent",
        # "nct_control_n_nodes_50_percent",
        # "nct_control_n_nodes_10_percent",
    ],
    # "Network Control Theory - Control Energy": [
    #     "nct_energies_total",
    #     "nct_energies_std", 
    #     "nct_energies_max",
    #     "nct_energies_n_nodes_90_percent",
    #     "nct_energies_n_nodes_50_percent",
    #     "nct_energies_n_nodes_10_percent",
    # ],
    "Computational Capacity": [
        "computational_capacity_memory_capacity_total",
        # "computational_capacity_memory_timescale",
        "computational_capacity_nonlinear_capacity_total",
        # "computational_capacity_cubic_capacity_total",
        # "computational_capacity_cross_capacity_total",
        # "computational_capacity_memory_nonlinear_ratio",
        # "computational_capacity_total_capacity",
        # "computational_capacity_state_dimensionality", 
        # "computational_capacity_state_entropy",
        # "computational_capacity_state_rank",
        # "computational_capacity_separation_ratio",
        # "computational_capacity_lyapunov_exponent",
    ],
    # "Reservoir Computing - Basic Measures": 
    "Memory Capacity - Full Lag Profile": [
        # [f"mc_{i}" for i in range(1, 50)]
        # "mc_mean" # , "mc_std"]
    ],
}


precise_categories = []
for cat_name, cat_cols in remaining_categories.items(): 
    
    for col in cat_cols:
        precise_categories.append(col)


# precise_categories
len(precise_categories)


# In[8]:


from config import PROPERTY_NAMES

# --- Config ---
THRESHOLD = 0.95  # only flag pairs above this

# Start fresh each run from the full metric set
remaining_cols = huge_df_properties.columns.tolist()
print(len(remaining_cols), "columns before removals")
# remove all columns that are not in PROPERTY_NAMES.keys(): 
remaining_cols = [c for c in remaining_cols if c in precise_categories] # PROPERTY_NAMES.keys()]
print(len(remaining_cols), "columns after removing those not in 'precise_categories'")

to_remove = [
            # "char_path_length",
            ] 

if to_remove is not None:
    remaining_cols = [c for c in remaining_cols if c not in to_remove]


# ── Run this cell repeatedly ──────────────────────────────────────────────────
working_df = huge_df_properties[remaining_cols].copy()
corr = working_df.corr().abs()

# Zero out diagonal and lower triangle to avoid duplicates
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
corr_upper = corr.where(mask)

# Find the single highest correlation
max_corr = corr_upper.stack().max()
max_pair = corr_upper.stack().idxmax()

if max_corr < THRESHOLD:
    print(f"✅ No correlations above {THRESHOLD}. Done! {len(remaining_cols)} variables remain.")
    print(f"Removed so far: {to_remove}")
else:
    var_a, var_b = max_pair
    print(f"⚠️  Highest correlation: {max_corr:.4f}")
    print(f"   → '{var_a}'")
    print(f"   → '{var_b}'")
    print()
    
    # Show each variable's mean absolute correlation with everything else
    # (helps you decide which is more redundant)
    mean_corr_a = corr_upper[var_a].fillna(corr_upper.T[var_a]).mean()
    mean_corr_b = corr_upper[var_b].fillna(corr_upper.T[var_b]).mean()
    print(f"   Mean |corr| with others:  '{var_a}' = {mean_corr_a:.3f}  |  '{var_b}' = {mean_corr_b:.3f}")
    print(f"   (Higher mean → more redundant overall → better candidate to drop)")
    print()
    # print("👉 Add one to `to_remove`, then re-run this cell:")
    # print(f"   to_remove.append('{var_a}')   # or '{var_b}'")

# # Apply removals
# remaining_cols = [c for c in huge_df_metrics.columns if c not in to_remove]
print(f"\n🗑️  Removed so far ({len(to_remove)}): {to_remove}")
print(f"📊 Remaining variables: {len(remaining_cols)}")


# In[9]:


# Apply the selection
huge_df_properties = huge_df_properties[remaining_cols]


# In[10]:


# ── 1. Load category / section metadata from Excel ────────────────────────────
excel_path = "/Users/adrian/Desktop/network_properties_full_excel.xlsx" # network_properties_full_excel.xlsx"   # ← adjust path if needed
meta_raw = pd.read_excel(excel_path, sheet_name=0)

meta_raw["section"] = meta_raw.apply(
    lambda r: r["#"] if pd.isna(r["Variable Name"]) and pd.notna(r["#"]) else None,
    axis=1,
).ffill()

meta = (
    meta_raw[meta_raw["Variable Name"].notna()]
    [["Variable Name", "Category", "section"]]
    .rename(columns={"Variable Name": "var"})
    .set_index("var")
)

CATEGORY_ORDER = [
    "Structure", "Integration", "Segregation",
    "Dynamics", "Computation", "Robustness",
]

CATEGORY_COLOURS = {
    "Integration": "#4C72B0",
    "Segregation": "#DD8452",
    "Robustness":  "#55A868",
    "Topology":    "#C44E52",
    "Wiring":      "#8172B2",
    "Geometry":    "#937860",
    "Dynamics":    "#DA8BC3",
    "Computation": "#8C8C8C",
    "Structure":   "#CCB974",
}

# ── 2. Sort variables by category, then section ───────────────────────────────
cols_in_data = [c for c in huge_df_properties.columns if c in meta.index]

def sort_key(col):
    row = meta.loc[col]
    cat = row["Category"] if pd.notna(row["Category"]) else "ZZZ"
    cat_idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (cat_idx, str(row["section"]))

cols_sorted_by_cat = sorted(cols_in_data, key=sort_key)

# ── 3. Within-category hierarchical clustering ────────────────────────────────
corr_full = huge_df_properties[cols_in_data].corr()

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
final_order.extend([c for c in cols_in_data if c not in final_order])

# ── 4. Reorder & label ────────────────────────────────────────────────────────
reduced_corr = corr_full.loc[final_order, final_order]
new_labels = [PROPERTY_NAMES.get(c, c) for c in final_order]

# ── 5. Plot ───────────────────────────────────────────────────────────────────
n = len(final_order)
fig = plt.figure(figsize=viz.cm_to_inch((18,12)))

gs = gridspec.GridSpec(1, 1, left=0.22, right=0.97, top=0.97, bottom=0.12)
ax = fig.add_subplot(gs[0])

im = ax.imshow(
    reduced_corr.values,
    cmap="RdBu", vmin=-1, vmax=1,
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

plt.savefig(output_folder / "corr_matrix_categorised.pdf", dpi=150, bbox_inches="tight")
print(output_folder / "corr_matrix_categorised.pdf")
plt.show()


# In[11]:


from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

# ── 1. Full-matrix hierarchical clustering ────────────────────────────────────
cols_in_data = [c for c in huge_df_properties.columns if c in meta.index]
corr_full    = huge_df_properties[cols_in_data].corr()

# dist = np.clip(1 - corr_full.values, 0, 2)
dist = np.clip(1 - np.abs(corr_full.values), 0, 1)  # was: 1 - corr_full.values

np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), method="average")

final_order  = [cols_in_data[i] for i in leaves_list(Z)]
reduced_corr = corr_full.loc[final_order, final_order]
new_labels   = [PROPERTY_NAMES.get(c, c) for c in final_order]

# ── 2. Cut the dendrogram into N clusters & assign colours ───────────────────
N_CLUSTERS = 6  # ← tune this
cluster_ids = fcluster(Z, N_CLUSTERS, criterion="maxclust")
# fcluster returns ids in original column order, remap to final_order
orig_to_cluster = dict(zip(cols_in_data, cluster_ids))

cluster_palette = plt.cm.tab10.colors
def cluster_colour(col):
    cid = orig_to_cluster.get(col, 0)
    return cluster_palette[(cid - 1) % len(cluster_palette)]

# ── 3. Plot ───────────────────────────────────────────────────────────────────
n   = len(final_order)
fig = plt.figure(figsize=viz.cm_to_inch((18, 12)))
gs  = gridspec.GridSpec(1, 1, left=0.22, right=0.97, top=0.97, bottom=0.12)
ax  = fig.add_subplot(gs[0])

im = ax.imshow(
    reduced_corr.values,
    cmap="RdBu", vmin=-1, vmax=1,
    aspect="equal", interpolation="none",
)

# ── 4. White dividers between clusters ───────────────────────────────────────
prev_c = orig_to_cluster[final_order[0]]
for i, col in enumerate(final_order[1:], start=1):
    if orig_to_cluster[col] != prev_c:
        ax.axhline(i - 0.5, color="white", linewidth=1.2, alpha=0.85)
        ax.axvline(i - 0.5, color="white", linewidth=1.2, alpha=0.85)
    prev_c = orig_to_cluster[col]

# ── 5. Tick labels ────────────────────────────────────────────────────────────
ax.set_yticks(np.arange(n))
ax.set_yticklabels(new_labels, fontsize=8)
ax.set_xticks([])
ax.tick_params(axis="both", which="both", length=0)

# ── 6. Cluster brackets on the left ──────────────────────────────────────────
moving_stuff = -0.5
BRACKET_X = -0.20 + moving_stuff
SERIF_W   =  0.005
LABEL_X   = -0.21 + moving_stuff
trans     = ax.transAxes

# Build row spans per cluster (in display order)
cluster_row_spans = {}
for row_i, col in enumerate(final_order):
    cid = orig_to_cluster[col]
    cluster_row_spans.setdefault(cid, [row_i, row_i])[1] = row_i

def row_to_y(row_i, n):
    return 1.0 - (row_i + 0.5) / n

for cid, (r0, r1) in cluster_row_spans.items():
    colour = cluster_palette[(cid - 1) % len(cluster_palette)]
    y_top  = row_to_y(r0, n) + 0.5 / n
    y_bot  = row_to_y(r1, n) - 0.5 / n
    y_mid  = (y_top + y_bot) / 2
    kw     = dict(xycoords=trans, textcoords=trans, annotation_clip=False,
                  arrowprops=dict(arrowstyle="-", color=colour, lw=1.5))

    ax.annotate("", xy=(BRACKET_X, y_bot),            xytext=(BRACKET_X, y_top),            **kw)
    ax.annotate("", xy=(BRACKET_X, y_top),             xytext=(BRACKET_X + SERIF_W, y_top),  **kw)
    ax.annotate("", xy=(BRACKET_X, y_bot),             xytext=(BRACKET_X + SERIF_W, y_bot),  **kw)
    ax.text(LABEL_X, y_mid, f"C{cid}", transform=trans,
            ha="right", va="center", fontsize=8, fontweight="bold",
            color=colour, clip_on=False)

cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label("Pearson r", fontsize=9)
ax.set_title(f"{n} properties — {N_CLUSTERS} clusters (hierarchical)")

plt.savefig(output_folder / "corr_matrix_clustered.pdf", dpi=150, bbox_inches="tight")
plt.show()

# ── 7. Print cluster compositions ─────────────────────────────────────────────
for cid in sorted(cluster_row_spans):
    members = [PROPERTY_NAMES.get(c, c) for c in final_order
               if orig_to_cluster[c] == cid]
    print(f"\nCluster {cid} ({len(members)} vars):")
    print("  " + ", ".join(members))


# In[12]:


huge_df["color_dataset"] = [COLOR_SCHEME[d] for d in huge_df["dataset"]]


# In[13]:


# Create a PCA between the columns of the huge_df
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Drop the columns that have inf or -inf values (if any). Print the dropped columns
dropped_cols = huge_df_properties.columns[huge_df_properties.isnull().any()].tolist()
huge_df_properties.replace([np.inf, -np.inf], np.nan, inplace=True)
huge_df_properties.dropna(axis=1, inplace=True)
print(f"Dropped columns: {dropped_cols}")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(huge_df_properties)

pca = PCA(n_components=10) # 10)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
plt.scatter(x=pca_result[:,0], 
            y=pca_result[:,1], 
            s=2, 
            alpha=0.4, 
            c=huge_df['color_dataset'], 
            # label=huge_df['dataset']
        )
plt.title("PCA of Metrics")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.show()

# Scree plot to show explained variance
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(6,4))
sns.barplot(x=[f"PC{i+1}" for i in range(len(explained_variance))], y=explained_variance)
plt.plot(range(0, len(explained_variance)), np.cumsum(explained_variance), marker="o", color="red",
        label="Cumulative")
plt.title("Scree Plot")
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.tight_layout()
plt.show()


# In[14]:


# Subplots of loadings for the first 3 principal components, ONLY top 10 contributors
loadings = pca.components_.T
num_metrics = loadings.shape[0]
metric_names = huge_df_properties.columns

# order them all by the absolute value of their loading on the first principal component
sorted_indices = np.argsort((loadings[:, 0]))[::-1]
# sorted_indices = np.argsort(np.abs(loadings[:, 0]) + np.abs(loadings[:, 1]))[::-1]
abs_ordered_loadings = np.abs(loadings[sorted_indices, 0])
loadings = loadings[sorted_indices]
metric_names = metric_names[sorted_indices]

print("Top 30 contributors to PC1:")
for i in range(30):
    print(f"     {metric_names[i]}          {abs_ordered_loadings[i]}") # : {loadings[i, 0]:.4f}")


# In[ ]:


# # SPIDERS 

# "global_efficiency"       
# "algebraic_connectivity_fiedler_value"  
# "targeted_attack_robustness_rob_targeted_auc" 
# "proportion_long_range_connections_0.3956" 
# "mc_mean"
# "mc_nonlin_mean"
# "repertoire_sweep_weighted_by_distances_diversity_critical"
# "spectral_radius"        
# "modularity"         


# In[ ]:


# # participation_coefficient_pc_frac_connector          0.24862004599666435
# "global_efficiency"       
# "algebraic_connectivity_fiedler_value"         
# # departure_from_normality_schur          0.2388229393629557
# # "computational_capacity_nonlinear_capacity_total"     
# # "computational_capacity_memory_capacity_total" 
# "targeted_attack_robustness_rob_targeted_auc" 

# # diffusion_efficiency          0.21098271161958115
# # ollivier_ricci_curvature_orc_skewness          0.2109514193117663
# # directed_simplices_count          0.19870729434396828
# # ollivier_ricci_curvature_orc_min          0.19364961825949942
# # algebraic_connectivity_laplacian_spectral_gap          0.18533221777214667
# "proportion_long_range_connections_0.3956" #           0.16468292052408973
# # participation_coefficient_n_communities          0.15748229441310163
# # community_synchronization_vulnerability_n_communities          0.15745520625071896
# # wiring_cost          0.14857909559505741
# # "repertoire_sweep_weighted_by_distances_T_critical"          
# # targeted_attack_robustness_rob_ratio          0.11264555775221687
# # ollivier_ricci_curvature_orc_mean          0.10072076340611079
# "mc_mean"
# "mc_nonlin_mean"
# # spectral_gap          0.04719648284302054
# # ollivier_ricci_curvature_orc_frac_neg          0.03047916149412621
# # repertoire_sweep_weighted_by_distances_size_critical          0.02496388972575507
# "repertoire_sweep_weighted_by_distances_diversity_critical"
# # synchronizability_eigenratio_eigenratio          0.12568922326379192
# # ollivier_ricci_curvature_orc_std          0.14893043845258178
# # rich_club_coefficient_rc_k_at_max          0.15704640658263644
# # participation_coefficient_pc_std          0.1874144132782656
# "spectral_radius"        
# "modularity"         


# In[ ]:


# Subplots of loadings for the first 3 principal components, ONLY top 10 contributors
loadings = pca.components_.T[:, :]
num_metrics = loadings.shape[0]
metric_names = huge_df_properties.columns

# order them all by the absolute value of their loading on the first principal component
sorted_indices = np.argsort((loadings[:, 0])) # [::-1]
loadings = loadings[sorted_indices]
metric_names = metric_names[sorted_indices]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6), dpi=100)
for i in range(3):
    axs[i].barh(range(num_metrics), loadings[:, i])
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(metric_names, rotation=0)
    axs[i].set_title(f"Loadings for PC{i+1}")
plt.tight_layout()
plt.show()


# In[ ]:


for m in metric_names[::-1]: 
    print(m)


# In[ ]:


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6), dpi=100)
for i in range(2):
    axs[i].barh(range(num_metrics), loadings[:, i])
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(metric_names, rotation=0)
    axs[i].set_title(f"Loadings for PC{i+1}")

# Plot 3: Addition of PC1 and PC2, ordered by magnitude of PC1+PC2
ordered_pc1_plus_pc2_indices = np.argsort(np.abs(loadings[:, 0] + loadings[:, 1])) #[::-1]
loadings = loadings[ordered_pc1_plus_pc2_indices]
metric_names = metric_names[ordered_pc1_plus_pc2_indices]
axs[2].barh(range(num_metrics), loadings[:, 0] + loadings[:, 1], color="orange")
axs[2].set_yticks(range(num_metrics))
axs[2].set_yticklabels(metric_names, rotation=0)
axs[2].set_title("Loadings for PC1 + PC2")

plt.tight_layout()
plt.show()


# In[ ]:


# Subplots of loadings for the first 3 principal components
loadings = pca.components_.T
num_metrics = loadings.shape[0]
metric_names = huge_df_properties.columns

# order them all by the absolute value of their loading on the first principal component
sorted_indices = np.argsort(np.abs(loadings[:, 0])) #[::-1]
loadings = loadings[sorted_indices]
metric_names = metric_names[sorted_indices]


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6), dpi=100)
for i in range(3):
    axs[i].barh(range(num_metrics), loadings[:, i])
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(metric_names, rotation=0)
    axs[i].set_title(f"Loadings for PC{i+1}")
plt.tight_layout()
plt.show()



# In[ ]:


# REPRESENTATIVES_FOR_GOALS = {
#     'mc_mean':                                          'Capacity (Memory)',
#     'modularity':                                       'Segregation',
#     'proportion_long_range_connections_0.3956':         'Wiring Economy',
#     'global_efficiency':                                'Integration',
#     'synchronizability_eigenratio_eigenratio':          'Synchronizability',
#     'kuramoto_synchronization_r_std':                   'Metastability',
#     'algebraic_connectivity_nx':                        'Robustness',
#     'computational_capacity_nonlinear_capacity_total':  'Capacity (Nonlinear)',
#     'repertoire_sweep_weighted_by_distances_diversity_critical': 'Repertoire Diversity',
#     'targeted_attack_robustness_rob_targeted_auc':      'Robustness (Targeted)',
# }

# rep_keys = list(REPRESENTATIVES_FOR_GOALS.keys())
# rep_labels = list(REPRESENTATIVES_FOR_GOALS.values())


# In[ ]:


# ── Helper: spider plot ───────────────────────────────────────────────────────
def make_spider(ax, values, label, color, axis_labels, title=None):
    """Draw a single radar/spider plot on a polar axis."""
    n = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals = list(values) + [values[0]]
    ax.plot(angles, vals, color=color, linewidth=2, label=label)
    ax.fill(angles, vals, color=color, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), axis_labels, fontsize=7)
    ax.tick_params(pad=6)
    if title:
        ax.set_title(title, fontsize=9, pad=14)



# GOAL_REPRESENTATIVES = {
#     "proportion_long_range_connections_0.3956":                 "Wiring\nEconomy",
    
#     "modularity":                                               "Segregation",
    
#     "computational_capacity_nonlinear_capacity_total":          "Capacity\n(Nonlinear)",
#     "mc_mean":                                                  "Capacity\n(Memory)",
    
#     # NCT control? 
#     "nct_control_avg":                                          "Control\n(NCT)",
    
#     # "global_efficiency":                                        "Integration",
#     "char_path_length":                                         "Integration", # ???? \n(Path Length)",
    
#     "targeted_attack_robustness_rob_targeted_auc":              "Robustness\n(Targeted)",
#     "algebraic_connectivity_fiedler_value":                     "Robustness\n(lambda_2)", # lambda_2. # algebraic_connectivity_nx": 

#     # "synchronizability_eigenratio_eigenratio":                  "Synchroniz-\nability",
    
#     # "kuramoto_synchronization_r_std":                            "Metastability",
#     "kuramoto_averaged_synchronization_r_std":                      "Metastability", # or "se"? 
#     "repertoire_sweep_weighted_by_distances_diversity_critical": "Repertoire\nDiversity",
    
# }

selected_properties = {
    "Integration": "global_efficiency",
    "Segregation": "modularity", 
    "Wiring\neconomy": "proportion_long_range_connections_0.3956", 
    "Robustness": "targeted_attack_robustness_rob_targeted_auc", 
    "Robustness\n(lambda_2)": "algebraic_connectivity_fiedler_value",
    # "Synchronizability": ["synchronizability_eigenratio_eigenratio"], # "synchronisability", "synchronisability_normalised"],
    "Dynamics": "spectral_radius", 
    # "Computational\ncapacity": ["computational_capacity_total_capacity"], # "computational_capacity", "computational_capacity_normalised"],
    "Memory": "computational_capacity_memory_capacity_total", # mc_mean",
    "Computational\ncapacity": "computational_capacity_nonlinear_capacity_total", # CHANGE THIS!! "mc_nonlin_mean",
    # "Metastability": ["kuramoto_averaged_synchronization_r_std"], # "metastability", "metastability_normalised"],
    # "Metastability Reservoir Size": ["repertoire_sweep_weighted_by_distances_size_critical"], 
    "Metastability Reservoir Diversity": "repertoire_sweep_weighted_by_distances_diversity_critical", 

}
rep_keys   = list(selected_properties.values())
rep_labels = list(selected_properties.keys()) 
rep_idx    = [huge_df_properties.columns.get_loc(k) for k in rep_keys]

# ── Angles for spider (shared across all spider plots) ───────────────────────
n_axes   = len(rep_labels)
angles   = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles  += angles[:1]


# ── Corner point definitions ──────────────────────────────────────────────────
CORNER_COLORS = {
    "top_left":     "steelblue",
    "bottom_left":  "orange",
    "bottom_right": "forestgreen",
}

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

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for ds in huge_df["dataset"].unique():
    mask = (huge_df["dataset"] == ds).values
    ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
               c=COLOR_SCHEME[ds], label=LABEL_MAP.get(ds, ds),
               alpha=0.4, s=8, edgecolors="none")
for name, idx in corners_full.items():
    ax.scatter(pca_result[idx, 0], pca_result[idx, 1],
               s=300, zorder=5, edgecolors="black", linewidths=2,
               color=CORNER_COLORS[name])
    ax.annotate(name, (pca_result[idx, 0], pca_result[idx, 1]),
                textcoords="offset points", xytext=(8, 8), fontsize=9)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA — Corner Points (Full Dataset)")
ax.legend(markerscale=2, fontsize=7)
plt.tight_layout()
plt.show()


# In[ ]:


# ── Spider plot for the top-6 features by PC1+PC2 loading magnitude ──────────
top6_idx = np.argsort(np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2))[::-1][:6]
top6_names  = metric_names[top6_idx]
top6_labels = [n.replace("_", "\n") for n in top6_names]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for name, idx in corners_full.items():
    vals = list(scaled_data[idx, top6_idx]) + [scaled_data[idx, top6_idx[0]]]
    a    = np.linspace(0, 2*np.pi, 6, endpoint=False).tolist() + [0]
    ax.plot(a, vals, color=CORNER_COLORS[name], linewidth=2, label=name)
    ax.fill(a, vals, color=CORNER_COLORS[name], alpha=0.15)
ax.set_thetagrids(np.degrees(np.linspace(0, 2*np.pi, 6, endpoint=False)), top6_labels, fontsize=8)
ax.set_title("Top-6 Features at PCA Extremes (z-scored)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
plt.tight_layout()
plt.show()

# ── Spider plot using predefined goal representatives ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for name, idx in corners_full.items():
    vals = list(scaled_data[idx, rep_idx]) + [scaled_data[idx, rep_idx[0]]]
    ax.plot(angles, vals, color=CORNER_COLORS[name], linewidth=2, label=name)
    ax.fill(angles, vals, color=CORNER_COLORS[name], alpha=0.15)
ax.set_thetagrids(np.degrees(angles[:-1]), rep_labels, fontsize=8)
ax.set_title("Network Properties at PCA Extremes (z-scored)\n— Full Dataset", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
plt.tight_layout()
plt.show()


# In[ ]:


GNM_DATASET = "hcp_schaefer_100_dataset_gnm"

# Boolean mask & original indices of non-GNM rows
# no_gnm_mask          = (huge_df["dataset"] != GNM_DATASET).values
# no_gnm_original_idx  = np.where(no_gnm_mask)[0]        # positions in sel_scaled / pca_result
# mask = no_gnm_mask
# orig_idx = no_gnm_original_idx

# Only get datasets that are mami or hcp 
mask = (huge_df["dataset"] == "hcp_schaefer_100_dataset").values | (huge_df["dataset"] == "suarez_MaMI_dataset").values
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

print("Corner points (no-GNM):")
for name, orig_idx in corners_no_gnm.items():
    ds = huge_df["dataset"].iloc[orig_idx]
    filt_idx = corners_no_gnm_filtered[name]
    print(f"  {name}: filtered_idx={filt_idx}, original_idx={orig_idx}, "
          f"PC1={pca_result[orig_idx,0]:.2f}, PC2={pca_result[orig_idx,1]:.2f}, dataset={ds}")

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Background: all datasets (GNM faded)
for ds in huge_df["dataset"].unique():
    mask_ds = (huge_df["dataset"] == ds).values
    alpha = 0.15 if ds == GNM_DATASET else 0.5
    ax.scatter(pca_result[mask_ds, 0], pca_result[mask_ds, 1],
               c=COLOR_SCHEME[ds], label=LABEL_MAP.get(ds, ds),
               alpha=alpha, s=8, edgecolors="none")

# Corner markers (using original indices so they are in the correct location)
for name, orig_idx in corners_no_gnm.items():
    ax.scatter(pca_result[orig_idx, 0], pca_result[orig_idx, 1],
               s=300, zorder=5, edgecolors="black", linewidths=2,
               color=CORNER_COLORS[name])
    ax.annotate(name, (pca_result[orig_idx, 0], pca_result[orig_idx, 1]),
                textcoords="offset points", xytext=(8, 8), fontsize=9)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA — Corner Points (GNM excluded from search)")
ax.legend(markerscale=2, fontsize=7)
plt.tight_layout()
plt.show()


# In[ ]:


SPECIAL_DATASETS = {
    "kaysons_generated_networks_routing":     ("Routing",     "purple"),
    "kaysons_generated_networks_diffusion":   ("Diffusion",   "teal"),
    "kaysons_generated_networks_propagation": ("Propagation", "crimson"),
}

# Original row index for single-network datasets
special_indices = {
    ds: int(huge_df.index[huge_df["dataset"] == ds][0])
    for ds in SPECIAL_DATASETS
}

# ── Build 3 × 3 figure ────────────────────────────────────────────────────────
fig, axs = plt.subplots(3, 3, figsize=(16, 16), subplot_kw=dict(polar=True), dpi=120)

row_configs = [
    # (title prefix, corners_dict or special list, use_corners_colors)
    ("Overall",   list(corners_full.items()),    CORNER_COLORS),
    ("No-GNM",    list(corners_no_gnm.items()),  CORNER_COLORS),
    ("Special",   [(ds, idx) for ds, idx in special_indices.items()], None),
]

for row, (prefix, items, color_map) in enumerate(row_configs):
    for col, (key, orig_idx) in enumerate(items):
        ax = axs[row, col]

        if color_map is not None:
            color = color_map[key]
            label = key
        else:
            label, color = SPECIAL_DATASETS[key]

        vals = list(scaled_data[orig_idx, rep_idx]) + [scaled_data[orig_idx, rep_idx[0]]]
        make_spider(ax, vals[:-1], label, color, rep_labels,
                    title=f"{prefix}: {label}")
        # ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=7))
        ax.set_xticklabels([])  # Hide x-axis labels for cleaner look
        # ax.

# Row annotations
for row, label in enumerate(["Overall PCA", "PCA excl. GNM", "Special Networks"]):
    axs[row, 0].annotate(
        label, xy=(-0.3, 0.5), xycoords="axes fraction",
        fontsize=11, fontweight="bold", ha="center", va="center", rotation=90,
    )

fig.suptitle("Network Properties at PCA Extremes (z-scored)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(output_folder / "spider_multipanel.pdf", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: spider_multipanel.pdf")


# In[ ]:


all_corner_indices = (
    list(corners_full.values()) +
    list(corners_no_gnm.values()) +
    list(special_indices.values())
)
all_vals = np.concatenate([scaled_data[idx, rep_idx] for idx in all_corner_indices])
# YLIM = (np.floor(all_vals.min()) - 0.5, np.ceil(all_vals.max()) + 0.5)
YLIM = (np.floor(all_vals.min()), np.ceil(all_vals.max()))
# YLIM = [-2, 2] # 4] # 10]
# YLIM = (-3.5, 3.5)  # override to a clean symmetric range if preferred
RING_VALS = [-2, 0, 2] # -6, 2] # 
# RING_VALS = [-1 * YLIM[0], -0.5 * YLIM[0]] + [0] + [0.5 * YLIM[1], 1 * YLIM[1]] # , -2, -1, 1, 2, YLIM[1]] # 2, 2] # [-6, 4] # , 8, 12] # [-8, -6, -4, -2, 2, 4, 6, 8, 10] # [-4, -2, 2, 4]


# In[ ]:


# ── Spider drawing function ────────────────────────────────────────────────────

def make_spider_standalone(values, label, color, feature_labels, title, ylim, filepath):
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    vals_closed   = list(values) + [values[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), dpi=50)

    # ── Limits & ticks ──
    ax.set_ylim(*ylim)
    ax.set_yticks(RING_VALS + [0]) # [-2, 0, 2])
    ax.set_yticklabels(RING_VALS + [0], fontsize=8, color="gray")

    # ── Remove outermost border ring ──
    ax.spines["polar"].set_visible(False)

    # ── Custom gridlines: draw manually so we can style zero differently ──
    ax.yaxis.grid(False)   # turn off auto y-gridlines
    ax.xaxis.grid(False)   # turn off auto x-gridlines (spokes we'll redraw)
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    for r_val in RING_VALS: # [-2, 2]:
        ax.plot(theta_ring, np.full(300, r_val),
                color="lightgray", linewidth=0.6, linestyle="--", zorder=0)
    # Zero ring — heavier
    ax.plot(theta_ring, np.zeros(300),
            color="gray", linewidth=1.8, linestyle="-", zorder=1)
    # Other rings: 
    for r_val in RING_VALS:  # YLIM: # [-2, 2]:
        theta_ring = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_ring, np.full(300, r_val), color="lightgray", linewidth=0.6, linestyle="--", zorder=0)

    # Spoke lines
    for angle in angles:
        ax.plot([angle, angle], YLIM, # [ylim[0], ylim[1]],
                color="lightgray", linewidth=0.5, linestyle="--", zorder=0)


    # ── Data ──
    ax.plot(angles_closed, vals_closed, color=color, linewidth=2.2, zorder=5)
    ax.fill(angles_closed, vals_closed, color=color, alpha=0.20, zorder=4)

    # ── Feature labels (spokes) — no angle labels by default ──
    ax.set_thetagrids([])          # hide spoke degree labels; add manually below
    ax.set_xticklabels([])

    ax.set_title(title, pad=16) # , fontsize=10, pad=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


# ── Build the list of 9 panels ────────────────────────────────────────────────

angles_for_legend = np.linspace(0, 2 * np.pi, len(rep_labels), endpoint=False).tolist()

panels = []

# First 6: Overall and No-GNM corners — color by *dataset* of that point
for corner_name, orig_idx in corners_full.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    vals  = scaled_data[orig_idx, rep_idx]
    panels.append((f"Overall · {corner_name.replace('_',' ')}", vals, color, f"spider_overall_{corner_name}.pdf"))

for corner_name, orig_idx in corners_no_gnm.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    vals  = scaled_data[orig_idx, rep_idx]
    panels.append((f"No-GNM · {corner_name.replace('_',' ')}", vals, color, f"spider_no_gnm_{corner_name}.pdf"))

# Last 3: Special single-network datasets
for ds_key, orig_idx in special_indices.items():
    label, color = SPECIAL_DATASETS[ds_key]
    vals = scaled_data[orig_idx, rep_idx]
    panels.append((f"Special · {label}", vals, color, f"spider_special_{label.lower()}.pdf"))

# ── Save all 9 plots ──────────────────────────────────────────────────────────

for title, vals, color, fname in panels:
    make_spider_standalone(
        values=vals,
        label=title,
        color=color,
        feature_labels=rep_labels,
        title=title,
        ylim=YLIM,
        filepath=output_folder / fname,
    )

print(output_folder / fname) 


# In[ ]:


def make_legend_plot(feature_labels, angles, filepath):
    """Empty radar with only the spoke labels visible."""
    n = len(feature_labels)
    angles_plot = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), dpi=50)
    ax.set_ylim(YLIM[0], YLIM[1]) # -3.5, 3.5)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # ax.get_yticklabels(kwargs={"fontsize": 10})  # force generation of ytick labels for correct zorder

    # Draw spoke lines
    for angle in angles_plot:
        ax.plot([angle, angle], YLIM, # [-3.5, 3.5],
                color="lightgray", linewidth=1.8, # 2, # 0.5, 
                linestyle="--", zorder=0)
    # Zero ring
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta_ring, np.zeros(300), color="gray", linewidth=1.8, zorder=1)
    # Other rings: 
    for r_val in RING_VALS:  # YLIM: # [-2, 2]:
        theta_ring = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_ring, np.full(300, r_val), color="lightgray", linewidth=0.6, linestyle="--", zorder=0)

    # Add spoke labels manually
    for angle, lbl in zip(angles_plot, feature_labels):
        x = np.degrees(angle)
        ax.set_thetagrids(np.degrees(angles_plot), feature_labels, fontsize=12)

    # ax.set_title("Legend / Labels", pad=16) # , fontsize=10, pad=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(filepath)


# ── Save the label-only plot ──────────────────────────────────────────────────

make_legend_plot(rep_labels, angles_for_legend, filepath=output_folder / "spider_labels_only.pdf")


# In[ ]:


def draw_axis_group_arcs(ax, feature_labels, angles, arc_radius, groups):
    """
    Draw a bold arc just outside the plot connecting paired spokes.
    arc_radius: draw the arc at this r value (slightly beyond local_max).
    """
    theta_dense = np.linspace(0, 2 * np.pi, 1000)

    for label_a, label_b, color in groups:
        if label_a not in feature_labels or label_b not in feature_labels:
            continue
        i_a = feature_labels.index(label_a)
        i_b = feature_labels.index(label_b)

        angle_a = angles[i_a]
        angle_b = angles[i_b]

        # Always sweep the short way between the two spokes
        if angle_b < angle_a:
            angle_a, angle_b = angle_b, angle_a

        # Choose the shorter arc
        if angle_b - angle_a > np.pi:
            arc_angles = np.linspace(angle_b, angle_a + 2 * np.pi, 80)
        else:
            arc_angles = np.linspace(angle_a, angle_b, 80)

        arc_r = np.full_like(arc_angles, arc_radius)

        ax.plot(arc_angles, arc_r,
                color=color, linewidth=3.5, solid_capstyle="round",
                zorder=0, alpha=0.85)

        # Small dots at each endpoint to cap the arc neatly
        ax.scatter([arc_angles[0], arc_angles[-1]],
                   [arc_radius, arc_radius],
                   color=color, s=18, zorder=7, alpha=0.85)


# In[ ]:


# ── Global y-range (shared across all plots) ──────────────────────────────────

all_corner_indices = (
    list(corners_full.values()) +
    list(corners_no_gnm.values()) +
    list(special_indices.values())
)
all_vals = np.concatenate([scaled_data[idx, rep_idx] for idx in all_corner_indices])
# YLIM = (np.floor(all_vals.min()) - 0.5, np.ceil(all_vals.max()) + 0.5)
YLIM = (np.floor(all_vals.min()), np.ceil(all_vals.max()))
# YLIM = [-2, 2] # 4] # 10]
# YLIM = (-3.5, 3.5)  # override to a clean symmetric range if preferred
RING_VALS = [-2, 0, 2] # -6, 2] # 
# RING_VALS = [-1 * YLIM[0], -0.5 * YLIM[0]] + [0] + [0.5 * YLIM[1], 1 * YLIM[1]] # , -2, -1, 1, 2, YLIM[1]] # 2, 2] # [-6, 4] # , 8, 12] # [-8, -6, -4, -2, 2, 4, 6, 8, 10] # [-4, -2, 2, 4]

AXIS_GROUPS = [
    ("Capacity\n(Nonlinear)", "Capacity\n(Memory)",     "lightgray"), # "#E07B39"),  # orange
    ("Robustness\n(Targeted)", "Robustness",              "lightgray"),  # "#5B8DB8"),  # blue
    ("Metastability",          "Repertoire\nDiversity",   "lightgray"),  # "#6AAB6A"),  # green
]

# ── Spider drawing function ────────────────────────────────────────────────────

def make_spider_standalone(values, label, color, feature_labels, title, ylim, filepath):
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    vals_closed   = list(values) + [values[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), dpi=50)

    # ── Limits & ticks ──
    # ax.set_ylim(*ylim)
    # ax.set_yticks(RING_VALS + [0]) # [-2, 0, 2])
    # ax.set_yticklabels(RING_VALS + [0], fontsize=8, color="gray")
    # ax.set_yticklabels(fontsize=8, color="gray")

    # ── Remove outermost border ring ──
    ax.spines["polar"].set_visible(False)

    # ── Custom gridlines: draw manually so we can style zero differently ──
    ax.yaxis.grid(False)   # turn off auto y-gridlines
    ax.xaxis.grid(False)   # turn off auto x-gridlines (spokes we'll redraw)
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    for r_val in RING_VALS: # [-2, 2]:
        ax.plot(theta_ring, np.full(300, r_val),
                color="lightgray", linewidth=0.6, linestyle="--", zorder=0)
    # Zero ring — heavier
    ax.plot(theta_ring, np.zeros(300),
            color="gray", linewidth=1.8, linestyle="-", zorder=1)
    # Other rings: 
    for r_val in RING_VALS:  # YLIM: # [-2, 2]:
        theta_ring = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_ring, np.full(300, r_val), color="lightgray", linewidth=0.6, linestyle="--", zorder=0)

    # Spoke lines
    for angle in angles:
        ax.plot([angle, angle], YLIM, # [ylim[0], ylim[1]],
                color="lightgray", linewidth=0.5, linestyle="--", zorder=0)


    # ── Data ──
    ax.plot(angles_closed, vals_closed, color=color, linewidth=2.2, zorder=5)
    ax.fill(angles_closed, vals_closed, color=color, alpha=0.20, zorder=4)

    # ── Feature labels (spokes) — no angle labels by default ──
    ax.set_thetagrids([])          # hide spoke degree labels; add manually below
    ax.set_xticklabels([])

    ax.set_title(title, pad=16) # , fontsize=10, pad=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def make_legend_plot(feature_labels, angles, filepath):
    """Empty radar with only the spoke labels visible."""
    n = len(feature_labels)
    angles_plot = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), dpi=50)
    ax.set_ylim(YLIM[0], YLIM[1]) # -3.5, 3.5)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # Draw spoke lines
    for angle in angles_plot:
        ax.plot([angle, angle], YLIM, # [-3.5, 3.5],
                color="lightgray", linewidth=1.8, # 2, # 0.5, 
                linestyle="--", zorder=0)
        
    # ── Group arcs just outside the outermost ring ──
    local_min, local_max = YLIM[0], YLIM[1]
    arc_r = local_max + (local_max - local_min) * 0.12
    draw_axis_group_arcs(ax, list(feature_labels), angles, arc_r, AXIS_GROUPS)
    # Expand ylim slightly so the arc isn't clipped
    ax.set_ylim(local_min, arc_r + (local_max - local_min) * 0.05)

    # Zero ring
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta_ring, np.zeros(300), color="gray", linewidth=1.8, zorder=1)
    # Other rings: 
    for r_val in RING_VALS:  # YLIM: # [-2, 2]:
        theta_ring = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_ring, np.full(300, r_val), color="lightgray", linewidth=0.6, linestyle="--", zorder=0)
        

    # Add spoke labels manually
    for angle, lbl in zip(angles_plot, feature_labels):
        x = np.degrees(angle)
        ax.set_thetagrids(np.degrees(angles_plot), feature_labels, fontsize=8)
        

    # ax.set_title("Legend / Labels", pad=16) # , fontsize=10, pad=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


# ── Build the list of 9 panels ────────────────────────────────────────────────

angles_for_legend = np.linspace(0, 2 * np.pi, len(rep_labels), endpoint=False).tolist()

panels = []

# First 6: Overall and No-GNM corners — color by *dataset* of that point
for corner_name, orig_idx in corners_full.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    vals  = scaled_data[orig_idx, rep_idx]
    panels.append((f"Overall · {corner_name.replace('_',' ')}", vals, color, f"spider_overall_{corner_name}.pdf"))

for corner_name, orig_idx in corners_no_gnm.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    vals  = scaled_data[orig_idx, rep_idx]
    panels.append((f"No-GNM · {corner_name.replace('_',' ')}", vals, color, f"spider_no_gnm_{corner_name}.pdf"))

# Last 3: Special single-network datasets
for ds_key, orig_idx in special_indices.items():
    label, color = SPECIAL_DATASETS[ds_key]
    vals = scaled_data[orig_idx, rep_idx]
    panels.append((f"Special · {label}", vals, color, f"spider_special_{label.lower()}.pdf"))

# ── Save all 9 plots ──────────────────────────────────────────────────────────

for title, vals, color, fname in panels:
    make_spider_standalone(
        values=vals,
        label=title,
        color=color,
        feature_labels=rep_labels,
        title=title,
        ylim=YLIM,
        filepath=output_folder / fname,
    )

# ── Save the label-only plot ──────────────────────────────────────────────────

make_legend_plot(rep_labels, angles_for_legend, filepath=output_folder / "spider_labels_only.pdf")

print("\nDone — 10 files saved.")
print(f"Files are in: {output_folder.resolve()}")


# In[ ]:





# In[ ]:


# ── Build color + label lookup for all 9 highlighted points ──────────────────
highlighted = {}

# First 6: corners_full and corners_no_gnm — same color logic as spider plots
for corner_name, orig_idx in corners_full.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    label = f"Overall · {corner_name.replace('_', ' ')}"
    highlighted[label] = {"idx": orig_idx, "color": color, # "marker": "o", 
                          "source": "full"}

for corner_name, orig_idx in corners_no_gnm.items():
    ds    = huge_df["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    label = f"No-GNM · {corner_name.replace('_', ' ')}"
    highlighted[label] = {"idx": orig_idx, "color": color, # "marker": "s", 
                          "source": "no_gnm"}

# Last 3: special datasets
for ds_key, orig_idx in special_indices.items():
    sp_label, color = SPECIAL_DATASETS[ds_key]
    label = f"Special · {sp_label}"
    highlighted[label] = {"idx": orig_idx, "color": color, # "marker": "^", 
                          "source": "special"}

# ── PCA scatter — background points ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

for ds in huge_df["dataset"].unique():
    mask = (huge_df["dataset"] == ds).values
    ax.scatter(
        pca_result[mask, 0], pca_result[mask, 1],
        c=COLOR_SCHEME[ds], label=LABEL_MAP.get(ds, ds),
        alpha=0.3, s=8, edgecolors="none", zorder=1,
    )

# ── Highlighted points ────────────────────────────────────────────────────────

marker_legend = {"full": ("o", "Overall corner"), "no_gnm": ("s", "No-GNM corner"), "special": ("^", "Special network")}

for label, info in highlighted.items():
    idx    = info["idx"]
    color  = info["color"]
    # marker = info["marker"]
    ax.scatter(
        pca_result[idx, 0], pca_result[idx, 1],
        color=color, # marker=marker,
        s=50, # 220, 
        zorder=6,
        edgecolors="black", linewidths=1.4,
    )
    
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("PC1") # , fontsize=10)
ax.set_ylabel("PC2") # , fontsize=10)
ax.set_title("PCA — All 9 Highlighted Points") # , fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(output_folder / "pca_highlighted_9points.pdf", dpi=150, bbox_inches="tight")



for label, info in highlighted.items():
    idx    = info["idx"]
    color  = info["color"]
    ax.annotate(
        label,
        (pca_result[idx, 0], pca_result[idx, 1]),
        textcoords="offset points", xytext=(8, 6),
        fontsize=7.5, zorder=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )


plt.show()
print(output_folder / "pca_highlighted_9points.pdf")


# In[ ]:


sel_df_label = huge_df
sel_pca_result = pca_result
sel_scaled = scaled_data


# In[ ]:


# ── Build color + label lookup for all 9 highlighted points ──────────────────
highlighted = {}

# First 6: corners_full and corners_no_gnm — same color logic as spider plots
for corner_name, orig_idx in corners_full.items():
    ds    = sel_df_label["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    label = f"Overall · {corner_name.replace('_', ' ')}"
    highlighted[label] = {"idx": orig_idx, "color": color, # "marker": "o", 
                          "source": "full"}

for corner_name, orig_idx in corners_no_gnm.items():
    ds    = sel_df_label["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    label = f"No-GNM · {corner_name.replace('_', ' ')}"
    highlighted[label] = {"idx": orig_idx, "color": color, # "marker": "s", 
                          "source": "no_gnm"}

# Last 3: special datasets
for ds_key, orig_idx in special_indices.items():
    sp_label, color = SPECIAL_DATASETS[ds_key]
    label = f"Special · {sp_label}"
    highlighted[label] = {"idx": orig_idx, "color": color, # "marker": "^", 
                          "source": "special"}

# ── PCA scatter — background points ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

for ds in sel_df_label["dataset"].unique():
    mask = (sel_df_label["dataset"] == ds).values
    ax.scatter(
        sel_pca_result[mask, 0], sel_pca_result[mask, 1],
        c=COLOR_SCHEME[ds], 
        label=LABEL_MAP.get(ds, ds),
        alpha=0.3 if ds == GNM_DATASET else 0.8, 
        s=8 if ds == GNM_DATASET else 12, 
        edgecolors="none" if ds == GNM_DATASET else "black", 
        linewidths=0 if ds == GNM_DATASET else 0.5,
        zorder=1,
    )

# ── Highlighted points ────────────────────────────────────────────────────────

marker_legend = {"full": ("o", "Overall corner"), 
                 "no_gnm": ("s", "No-GNM corner"), 
                 "special": ("^", "Special network")}

for label, info in highlighted.items():
    idx    = info["idx"]
    color  = info["color"]
    # marker = info["marker"]
    ax.scatter(
        sel_pca_result[idx, 0], sel_pca_result[idx, 1],
        color=color, # marker=marker,
        s=50, # 220, 
        zorder=6,
        edgecolors="black", 
        linewidths=1.4,
    )
    
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("PC1") # , fontsize=10)
ax.set_ylabel("PC2") # , fontsize=10)
# ax.set_title("PCA — All 9 Highlighted Points") # , fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(output_folder / "pca_highlighted_9points.pdf", dpi=150, bbox_inches="tight")



for label, info in highlighted.items():
    idx    = info["idx"]
    color  = info["color"]
    ax.annotate(
        label,
        (sel_pca_result[idx, 0], sel_pca_result[idx, 1]),
        textcoords="offset points", xytext=(8, 6),
        fontsize=7.5, zorder=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )


plt.show()
print(output_folder / "pca_highlighted_9points.pdf")


# In[ ]:


from sklearn.metrics import pairwise_distances

# ── Compute 5 nearest neighbours for each panel point ────────────────────────
# Use the full PCA embedding (all 10 components) for distance

def get_neighbours(orig_idx, pca_result, n=5, exclude_gnm=False, df_label=None, gnm_name=None):
    """Return indices of the n closest rows in pca_result to orig_idx."""
    dists = pairwise_distances(pca_result[orig_idx:orig_idx+1], pca_result)[0]
    dists[orig_idx] = np.inf  # exclude self
    if exclude_gnm and df_label is not None:
        gnm_mask = (df_label["dataset"] == gnm_name).values
        dists[gnm_mask] = np.inf
    return np.argsort(dists)[:n].tolist()


# ── Modified spider drawing function with neighbours ─────────────────────────

def make_spider_standalone(values, label, color, feature_labels, title, ylim,
                           filepath, neighbour_vals_list=None):
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    vals_closed   = list(values) + [values[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), dpi=50)

    ax.set_ylim(*ylim)
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    theta_ring = np.linspace(0, 2 * np.pi, 300)

    # ── Gridlines ──
    for r_val in RING_VALS:
        ax.plot(theta_ring, np.full(300, r_val),
                color="lightgray", linewidth=0.6, linestyle="--", zorder=0)
    ax.plot(theta_ring, np.zeros(300),
            color="gray", linewidth=1.8, linestyle="-", zorder=1)
    for angle in angles:
        ax.plot([angle, angle], ylim,
                color="lightgray", linewidth=0.5, linestyle="--", zorder=0)

    # # ── Neighbour lines (faded, thin, same color) ──
    # if neighbour_vals_list is not None:
    #     for nb_vals in neighbour_vals_list:
    #         nb_closed = list(nb_vals) + [nb_vals[0]]
    #         ax.plot(angles_closed, nb_closed,
    #                 color=color, linewidth=0.6, alpha=0.25, zorder=3)
    #         ax.fill(angles_closed, nb_closed,
    #                 color=color, alpha=0.04, zorder=2)

    # # ── Main data (on top) ──
    # ax.plot(angles_closed, vals_closed, color=color, linewidth=2.2, zorder=5)
    # ax.fill(angles_closed, vals_closed, color=color, alpha=0.20, zorder=4)


    # ── Neighbour lines and main line all same style --- # (faded, thin, same color) ──
    alpha_fill = 0.05
    linewidth = 1
    zorder = 3
    
    if neighbour_vals_list is not None:
        for nb_vals in neighbour_vals_list:
            nb_closed = list(nb_vals) + [nb_vals[0]]
            ax.plot(angles_closed, nb_closed,
                    color=color, linewidth=linewidth, alpha=alpha, zorder=zorder+1)
            ax.fill(angles_closed, nb_closed,
                    color=color, alpha=alpha_fill, zorder=zorder)

    # ── Main data (on top) ──
    ax.plot(angles_closed, vals_closed, color=color, linewidth=linewidth, zorder=zorder+1)
    ax.fill(angles_closed, vals_closed, color=color, alpha=alpha_fill, zorder=zorder)



    ax.set_thetagrids([])
    ax.set_xticklabels([])
    ax.set_title(title, pad=16)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


# ── Rebuild panels list with neighbour data ───────────────────────────────────

panels = []

# First 6: Overall and No-GNM corners
for corner_name, orig_idx in corners_full.items():
    ds    = sel_df_label["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    nb_idx = get_neighbours(orig_idx, sel_pca_result, n=5)
    nb_vals = [sel_scaled[i, rep_idx] for i in nb_idx]
    panels.append((
        f"Overall · {corner_name.replace('_', ' ')}",
        sel_scaled[orig_idx, rep_idx], color,
        f"spider_overall_{corner_name}.pdf",
        nb_vals,
    ))

for corner_name, orig_idx in corners_no_gnm.items():
    ds    = sel_df_label["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    # Exclude GNM from neighbour search too, since we're in the no-GNM regime
    nb_idx = get_neighbours(orig_idx, sel_pca_result, n=5,
                            exclude_gnm=True, df_label=sel_df_label, gnm_name=GNM_DATASET)
    nb_vals = [sel_scaled[i, rep_idx] for i in nb_idx]
    panels.append((
        f"No-GNM · {corner_name.replace('_', ' ')}",
        sel_scaled[orig_idx, rep_idx], color,
        f"spider_no_gnm_{corner_name}.pdf",
        nb_vals,
    ))

# Last 3: Special datasets
for ds_key, orig_idx in special_indices.items():
    label, color = SPECIAL_DATASETS[ds_key]
    nb_idx = get_neighbours(orig_idx, sel_pca_result, n=5)
    nb_vals = [sel_scaled[i, rep_idx] for i in nb_idx]
    panels.append((
        f"Special · {label}",
        sel_scaled[orig_idx, rep_idx], color,
        f"spider_special_{label.lower()}.pdf",
        nb_vals,
    ))

# ── Save all 9 plots ──────────────────────────────────────────────────────────

for title, vals, color, fname, nb_vals in panels:
    make_spider_standalone(
        values=vals,
        label=title,
        color=color,
        feature_labels=rep_labels,
        title=title,
        ylim=YLIM,
        filepath=output_folder / fname,
        neighbour_vals_list=nb_vals,
    )

# ── Label-only plot (unchanged) ───────────────────────────────────────────────

make_legend_plot(rep_labels, angles_for_legend, filepath=output_folder / "spider_labels_only.pdf")

print("\nDone — 10 files saved.")


# In[ ]:


# ── Rebuild panels list with neighbour data ───────────────────────────────────

panels = []

# First 6: Overall and No-GNM corners
for corner_name, orig_idx in corners_full.items():
    ds    = sel_df_label["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    nb_idx = get_neighbours(orig_idx, sel_pca_result, n=5)
    nb_vals = [sel_scaled[i, rep_idx] for i in nb_idx]
    panels.append((
        f"Overall · {corner_name.replace('_', ' ')}",
        sel_scaled[orig_idx, rep_idx], color,
        f"spider_overall_{corner_name}.pdf",
        nb_vals,
    ))

for corner_name, orig_idx in corners_no_gnm.items():
    ds    = sel_df_label["dataset"].iloc[orig_idx]
    color = COLOR_SCHEME[ds]
    # Exclude GNM from neighbour search too, since we're in the no-GNM regime
    nb_idx = get_neighbours(orig_idx, sel_pca_result, n=5,
                            exclude_gnm=True, df_label=sel_df_label, gnm_name=GNM_DATASET)
    nb_vals = [sel_scaled[i, rep_idx] for i in nb_idx]
    panels.append((
        f"No-GNM · {corner_name.replace('_', ' ')}",
        sel_scaled[orig_idx, rep_idx], color,
        f"spider_no_gnm_{corner_name}.pdf",
        nb_vals,
    ))

# Last 3: Special datasets
for ds_key, orig_idx in special_indices.items():
    label, color = SPECIAL_DATASETS[ds_key]
    nb_idx = get_neighbours(orig_idx, sel_pca_result, n=5)
    nb_vals = [sel_scaled[i, rep_idx] for i in nb_idx]
    panels.append((
        f"Special · {label} - plus neighbours",
        sel_scaled[orig_idx, rep_idx], color,
        f"spider_special_{label.lower()}.pdf",
        nb_vals,
    ))

# ── Save all 9 plots ──────────────────────────────────────────────────────────

# for title, vals, color, fname, nb_vals in panels:
#     make_spider_standalone(
#         values=vals,
#         label=title,
#         color=color,
#         feature_labels=rep_labels,
#         title=title,
#         ylim=YLIM,
#         filepath=output_folder / fname,
#         neighbour_vals_list=nb_vals,
#     )
for title, vals, color, fname, nb_vals in panels:
    make_spider_standalone(
        values=vals,
        label=title,
        color=color,
        ylim=YLIM,
        feature_labels=rep_labels,
        title=title,
        filepath=output_folder / fname,
        neighbour_vals_list=nb_vals,
    )
# ── Label-only plot (unchanged) ───────────────────────────────────────────────

make_legend_plot(rep_labels, angles_for_legend, filepath=output_folder / "spider_labels_only.pdf")

print(output_folder / fname)


# In[37]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].keys()


# In[38]:


from utils import get_colorchannels_from_eta_gamma

combined_colors = get_colorchannels_from_eta_gamma(huge_df) 

plt.figure(figsize=viz.cm_to_inch((3,3)))
plt.scatter(huge_df["eta"], huge_df["gamma"], c=combined_colors, s=2, edgecolors="none")
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\gamma$")
plt.xticks([])
plt.yticks([])
plt.xlim(-8,3)
plt.ylim(-0.1,1)
plt.tight_layout()
plt.savefig(output_folder / "eta_gamma_color_mapping.pdf", dpi=150, bbox_inches="tight")
print(output_folder / "eta_gamma_color_mapping.pdf")


# # Taxonomies 

# In[39]:


# Taxonomies 
info_mami = pd.read_csv("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv")
taxonomy = "order" # phylogenetic_group" # "order" 

GROUPS_TO_INCLUDE = [
                    'Primates',
                    'Rodentia',
                    # 'Hyracoidea',
                    'Carnivora',
                    # 'Perissodactyla',
                    # 'Chiroptera',
                    'Cetartiodactyla',
                    # 'Eulipotyphla',
                    # 'Scandentia',
                    # 'Xenarthra',
                    # 'Lagomorpha',
                    # 'Marsupialia'
                ]
# info_mami = info_mami[info_mami[taxonomy].isin(GROUPS_TO_INCLUDE)]

# info_mami["tax_color"] = info_mami[taxonomy].astype("category").cat.codes

df_mami = pd.concat([sel_df_label[sel_df_label["dataset"] == "suarez_MaMI_dataset"].reset_index(), info_mami], axis=1) # , left_index=True, right_on=True) # "animal_name")

names = df_mami[taxonomy].unique() # ["tax_color"]
color_map = plt.cm.get_cmap("tab20", len(names)) # "RdYlGn_r", len(names))
color_dict = {name: color_map(i) for i, name in enumerate(names)}

df_mami["tax_color"] = df_mami[taxonomy].map(color_dict)


# In[40]:


plt.scatter(sel_pca_result[sel_df_label["dataset"] == "suarez_MaMI_dataset", 0], 
            sel_pca_result[sel_df_label["dataset"] == "suarez_MaMI_dataset", 1], 
            c=df_mami["tax_color"], 
            s=50)

# Faking the legend
for name in df_mami[taxonomy].unique(): # GROUPS_TO_INCLUDE: # color_dict:
    plt.scatter([], [], # c=color, # alpha=0.3 # , s=unique_scatters[0],
                    c=color_dict[name],
                    label=str(name))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # , fontsize=3, frameon=False, labelspacing=1, title='City Area') phylogenetic_group

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()


# In[41]:


# Taxonomies 
info_mami = pd.read_csv("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv")
taxonomy = "order" 

GROUPS_TO_INCLUDE = [
                    'Primates',
                    'Rodentia',
                    # 'Hyracoidea',
                    'Carnivora',
                    # 'Perissodactyla',
                    # 'Chiroptera',
                    'Cetartiodactyla',
                    # 'Eulipotyphla',
                    # 'Scandentia',
                    # 'Xenarthra',
                    # 'Lagomorpha',
                    # 'Marsupialia'
                ]
# info_mami = info_mami[info_mami[taxonomy].isin(GROUPS_TO_INCLUDE)]

# info_mami["tax_color"] = info_mami[taxonomy].astype("category").cat.codes

df_mami = pd.concat([sel_df_label[sel_df_label["dataset"] == "suarez_MaMI_dataset"].reset_index(), info_mami], axis=1) # , left_index=True, right_on=True) # "animal_name")

names = df_mami[taxonomy].unique() # ["tax_color"]
color_map = plt.cm.get_cmap("tab20", len(names)) # "RdYlGn_r", len(names))
color_dict = {name: color_map(i) for i, name in enumerate(names)}

df_mami["tax_color"] = df_mami[taxonomy].map(color_dict)



# In[42]:


from sklearn.metrics import pairwise_distances
from scipy.stats import f_oneway
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ── Extract the relevant PCA coordinates + labels ─────────────────────────────

mami_mask  = (sel_df_label["dataset"] == "suarez_MaMI_dataset").values
tax_mask   = df_mami[taxonomy].isin(GROUPS_TO_INCLUDE).values
combined   = mami_mask.copy()
combined[mami_mask] = tax_mask          # both conditions

X      = sel_pca_result[combined, :2]   # PC1 + PC2 only
labels = df_mami[tax_mask][taxonomy].values

# ── Test 1: PERMANOVA (permutation-based MANOVA on distances) ─────────────────
# "Are the groups more separated than random chance?"

from sklearn.utils import shuffle

def permanova(X, labels, n_permutations=999):
    def f_stat(X, labels):
        groups  = [X[labels == g] for g in np.unique(labels)]
        grand_m = X.mean(axis=0)
        ss_between = sum(len(g) * ((g.mean(0) - grand_m)**2).sum() for g in groups)
        ss_within  = sum(((g - g.mean(0))**2).sum() for g in groups)
        k, n = len(groups), len(X)
        return (ss_between / (k-1)) / (ss_within / (n-k))

    observed = f_stat(X, labels)
    null_dist = [f_stat(X, shuffle(labels)) for _ in range(n_permutations)]
    p = (np.sum(np.array(null_dist) >= observed) + 1) / (n_permutations + 1)
    return observed, p

f_obs, p_perm = permanova(X, labels)
print(f"PERMANOVA  →  F = {f_obs:.3f},  p = {p_perm:.4f}")

# ── Test 2: Per-axis ANOVA ─────────────────────────────────────────────────────
# "Does any single PC axis separate the groups?"

groups_pc1 = [X[labels == g, 0] for g in np.unique(labels)]
groups_pc2 = [X[labels == g, 1] for g in np.unique(labels)]
f1, p1 = f_oneway(*groups_pc1)
f2, p2 = f_oneway(*groups_pc2)
print(f"ANOVA PC1  →  F = {f1:.3f},  p = {p1:.4f}")
print(f"ANOVA PC2  →  F = {f2:.3f},  p = {p2:.4f}")

# ── Test 3: LDA cross-validated accuracy ──────────────────────────────────────
# "Can we predict group membership better than chance?"

from sklearn.model_selection import cross_val_score
lda      = LinearDiscriminantAnalysis()
cv_score = cross_val_score(lda, X, labels, cv=5, scoring="accuracy").mean()
chance   = 1 / len(np.unique(labels))
print(f"LDA 5-fold CV accuracy = {cv_score:.3f}  (chance = {chance:.3f})")


# # Ages 

# In[43]:


for dataset_name, d in dict_with_all_datasets.items():
    

    if "lexis_data_young" in dataset_name:
    # if "lexis_data_" in dataset_name:
    
            print(dataset_name)
            ages = np.load(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/{dataset_name}/04_further_info/00_ages.npy")
            # Turn dict_with_all_datasets["lexis_data_aging"].index into a mask
            
            mask = np.zeros_like(ages)
            mask[dict_with_all_datasets[dataset_name].index] = 1
            mask = mask.astype(bool)
            
            # add an age column by index to the main dataframe
            dict_with_all_datasets[dataset_name]["age"] = ages[mask]
        


# In[44]:


huge_df = pd.DataFrame()

# combine all the metrics into one dataframe for this experiment
for dataset_name in ordered_dataset_names: 
    df = dict_with_all_datasets[dataset_name]
    # Add a column to identify the dataset
    df["dataset"] = dataset_name
    
    print(dataset_name, len(df))
    # Remove this, as eta, gamma, etc are not defined here? 
    if dataset_name == "kaysons_generated_networks_topology" or dataset_name == "hcp_schaefer_100_dataset": 
        print(" ---> skipped")
        continue
    huge_df = pd.concat([huge_df, df], ignore_index=True)
    


# In[45]:


huge_df


# In[46]:


# Create a PCA between the columns of the huge_df
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Drop the columns that have inf or -inf values (if any). Print the dropped columns
dropped_cols = huge_df_properties.columns[huge_df_properties.isnull().any()].tolist()
huge_df_properties.replace([np.inf, -np.inf], np.nan, inplace=True)
huge_df_properties.dropna(axis=1, inplace=True)
print(f"Dropped columns: {dropped_cols}")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(huge_df_properties)

pca = PCA(n_components=10) # 10)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
plt.scatter(x=pca_result[:,0], 
            y=pca_result[:,1], 
            s=4,
            alpha=0.4, 
            c=huge_df["age"],
            # c=huge_df['color_dataset'], 
            # label=huge_df['dataset']
        )
plt.title("PCA of Metrics")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.colorbar(label="Age") # if using age as color
plt.show()

# Scree plot to show explained variance
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(6,4))
sns.barplot(x=[f"PC{i+1}" for i in range(len(explained_variance))], y=explained_variance)
plt.plot(range(0, len(explained_variance)), np.cumsum(explained_variance), marker="o", color="red",
        label="Cumulative")
plt.title("Scree Plot")
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.tight_layout()
plt.show()


# In[47]:


len(huge_df_properties)


# In[48]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── 1. Clean ──────────────────────────────────────────────────────────────────
dropped_cols = huge_df_properties.columns[huge_df_properties.isnull().any()].tolist()
huge_df_properties.replace([np.inf, -np.inf], np.nan, inplace=True)
huge_df_properties.dropna(axis=1, inplace=True)
print(f"Dropped columns: {dropped_cols}")

# ── 2. Scale & PCA on the full data ───────────────────────────────────────────
scaler = StandardScaler()
scaled_data = scaler.fit_transform(huge_df_properties)

pca = PCA(n_components=10)
pca_result = pca.fit_transform(scaled_data)

# ── 3. Average PC scores per age ──────────────────────────────────────────────
pca_df = pd.DataFrame(pca_result[:, :2], columns=["PC1", "PC2"])
pca_df["age"] = huge_df["age"].values

age_means = pca_df.groupby("age")[["PC1", "PC2"]].mean().reset_index().sort_values("age")

# ── 4. PCA scatter: averaged dots coloured by age, connected by line ──────────
fig, ax = plt.subplots(figsize=(8, 6))

norm = plt.Normalize(age_means["age"].min(), age_means["age"].max())
cmap = plt.cm.viridis

# Line through the age-averaged points (sorted by age)
ax.plot(age_means["PC1"], age_means["PC2"],
        color="grey", linewidth=0.8, alpha=0.6, zorder=1)

# One dot per age, coloured by age
sc = ax.scatter(age_means["PC1"], age_means["PC2"],
                c=age_means["age"], cmap=cmap, norm=norm,
                s=40, zorder=2, edgecolors="white", linewidths=0.4)

plt.colorbar(sc, ax=ax, label="Age")
ax.set_title("PCA of Metrics — averaged per age")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# ── 5. Scree plot ─────────────────────────────────────────────────────────────
explained_variance = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=[f"PC{i+1}" for i in range(len(explained_variance))],
            y=explained_variance, ax=ax)
ax.plot(range(len(explained_variance)), np.cumsum(explained_variance),
        marker="o", color="red", label="Cumulative")
ax.set_title("Scree Plot")
ax.set_ylabel("Explained Variance Ratio")
ax.set_xlabel("Principal Components")
ax.legend()
plt.tight_layout()
plt.show()


# In[49]:


from scipy.interpolate import make_smoothing_spline  # or UnivariateSpline

# ── After computing age_means (sorted by age) ────────────────────────────────

# Smooth PC1 and PC2 as a function of age using a smoothing spline
age_vals = age_means["age"].values
t = np.linspace(age_vals.min(), age_vals.max(), 300)  # fine grid for smooth curve

spl_pc1 = make_smoothing_spline(age_vals, age_means["PC1"].values, lam=5)
spl_pc2 = make_smoothing_spline(age_vals, age_means["PC2"].values, lam=5)

smooth_pc1 = spl_pc1(t)
smooth_pc2 = spl_pc2(t)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

norm = plt.Normalize(age_vals.min(), age_vals.max())
cmap = plt.cm.terrain # gist_rainbow # viridis

# Smoothed line coloured by age using a LineCollection
from matplotlib.collections import LineCollection

points = np.array([smooth_pc1, smooth_pc2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, alpha=0.8, zorder=1)
lc.set_array(t)  # colour the line by age along its length
ax.add_collection(lc)

# Raw averaged dots on top
sc = ax.scatter(age_means["PC1"], age_means["PC2"],
                c=age_means["age"],
                cmap=cmap, norm=norm,
                s=40, zorder=2, edgecolors="white", linewidths=0.4)

ax.autoscale()
plt.colorbar(sc, ax=ax, label="Age")
ax.set_title("Averaged per age (after PCA)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
# plt.tight_layout()



############# THE COSY

# ── After the main plot, before plt.show() ────────────────────────────────────

# Get PC loadings (components_ shape: n_components × n_features)
loadings = pd.DataFrame(
    pca.components_[:2].T,
    index=huge_df_properties.columns,
    columns=["PC1", "PC2"]
)

# Top 3 by absolute loading on PC1 and PC2 (union, deduplicated)
top_pc1 = loadings["PC1"].abs().nlargest(3).index.tolist()
top_pc2 = loadings["PC2"].abs().nlargest(3).index.tolist()
top_vars = list(dict.fromkeys(top_pc1 + top_pc2))  # preserve order, no duplicates

# ── Inset axes in bottom-right corner ────────────────────────────────────────
ax_inset = ax.inset_axes([0.72, 1-0.26-0.02, 0.26, 0.26])  # [x, y, w, h] in axes coords
# ax_inset.set_facecolor("#f7f7f7")
ax_inset.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
ax_inset.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
# remove frame
for spine in ax_inset.spines.values():
    spine.set_visible(False)

for var in top_vars:
    vx = loadings.loc[var, "PC1"]
    vy = loadings.loc[var, "PC2"]
    label = PROPERTY_NAMES.get(var, var)

    # Colour by which PC it ranks highest on
    colour = "#E63946" if var in top_pc1 else "#457B9D"
    if var in top_pc1 and var in top_pc2:
        colour = "#9B2226"  # appears in both — use distinct colour

    ax_inset.annotate(
        "", xy=(vx, vy), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.2,
                        mutation_scale=6)
    )
    # Nudge label slightly beyond arrow tip
    ax_inset.text(vx * 1.15, vy * 1.15, label,
                  fontsize=5.5, ha="center", va="center",
                  color=colour, fontweight="bold")

# Square, symmetric axes limits with a little padding
max_val = loadings.loc[top_vars, ["PC1", "PC2"]].abs().values.max() * 1.4
ax_inset.set_xlim(-max_val, max_val)
ax_inset.set_ylim(-max_val, max_val)
ax_inset.set_aspect("equal")
ax_inset.set_xlabel("PC1", fontsize=6, labelpad=1)
ax_inset.set_ylabel("PC2", fontsize=6, labelpad=1)
ax_inset.tick_params(labelsize=5)
ax_inset.set_title("Top loadings", fontsize=6, pad=2)

plt.tight_layout()
plt.show()






# In[50]:


# Subplots of loadings for the first 3 principal components, ONLY top 10 contributors
loadings = pca.components_.T[:, :]
num_metrics = loadings.shape[0]
metric_names = huge_df_properties.columns


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6), dpi=100)
for i in range(3):
    
    # order them all by the absolute value of their loading on the first principal component
    sorted_indices = np.argsort((loadings[:, i])) # [::-1]
    loadings = loadings[sorted_indices]
    metric_names = metric_names[sorted_indices]


    axs[i].barh(range(num_metrics), loadings[:, i])
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(metric_names, rotation=0)
    axs[i].set_title(f"Loadings for PC{i+1}")
plt.tight_layout()
plt.show()


# In[51]:


# Subplots of loadings for the first 3 principal components, ONLY top 10 contributors
loadings = pca.components_.T[:, :]
num_metrics = loadings.shape[0]
metric_names = huge_df_properties.columns

# order them all by the absolute value of their loading on the first principal component
sorted_indices = np.argsort((loadings[:, 0])) # [::-1]
loadings = loadings[sorted_indices]
metric_names = metric_names[sorted_indices]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6), dpi=100)
for i in range(3):
    axs[i].barh(range(num_metrics), loadings[:, i])
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(metric_names, rotation=0)
    axs[i].set_title(f"Loadings for PC{i+1}")
plt.tight_layout()
plt.show()


# In[52]:


# Subplots of loadings for the first 3 principal components, ONLY top 10 contributors
loadings = pca.components_.T[:, :]
num_metrics = loadings.shape[0]
metric_names = huge_df_properties.columns

# order them all by the absolute value of their loading on the first principal component
sorted_indices = np.argsort((loadings[:, 1])) # [::-1]
loadings = loadings[sorted_indices]
metric_names = metric_names[sorted_indices]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6), dpi=100)
for i in range(3):
    axs[i].barh(range(num_metrics), loadings[:, i])
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(metric_names, rotation=0)
    axs[i].set_title(f"Loadings for PC{i+1}")
plt.tight_layout()
plt.show()


# In[53]:


# ── 3. Average PC scores per dataset × age ───────────────────────────────────
pca_df = pd.DataFrame(pca_result[:, :2], columns=["PC1", "PC2"])
pca_df["age"]     = huge_df["age"].values
pca_df["dataset"] = huge_df["dataset"].values

dataset_age_means = (
    pca_df.groupby(["dataset", "age"])[["PC1", "PC2"]]
    .mean()
    .reset_index()
    .sort_values(["dataset", "age"])
)

# ── 4. Plot ───────────────────────────────────────────────────────────────────
datasets  = dataset_age_means["dataset"].unique()
palette   = dict(zip(datasets, plt.cm.tab10.colors))

fig, ax = plt.subplots(figsize=(8, 6))

for ds, grp in dataset_age_means.groupby("dataset"):
    colour = palette[ds]
    age_vals = grp["age"].values
    t = np.linspace(age_vals.min(), age_vals.max(), 300)

    spl_pc1 = make_smoothing_spline(age_vals, grp["PC1"].values, lam=5)
    spl_pc2 = make_smoothing_spline(age_vals, grp["PC2"].values, lam=5)

    ax.plot(grp["PC1"].values, 
            grp["PC2"].values, # spl_pc1(t), spl_pc2(t), 
            color="gray", linewidth=1.5, alpha=0.7, zorder=1)
    # Line through the age-averaged points (sorted by age)
    # ax.plot(age_means["PC1"], age_means["PC2"],
    #         color="grey", linewidth=0.8, alpha=0.6, zorder=1)

    ax.scatter(grp["PC1"], grp["PC2"],
               color=colour, s=40, zorder=2,
               edgecolors="white", linewidths=0.4, label=ds)

ax.legend(title="Dataset", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
# ax.set_title("PCA of Metrics — averaged per dataset × age")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
plt.tight_layout()
plt.show()


# In[54]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_smoothing_spline
from matplotlib.collections import LineCollection

# ── 1. Clean ──────────────────────────────────────────────────────────────────
dropped_cols = huge_df_properties.columns[huge_df_properties.isnull().any()].tolist()
huge_df_properties.replace([np.inf, -np.inf], np.nan, inplace=True)
huge_df_properties.dropna(axis=1, inplace=True)
print(f"Dropped columns: {dropped_cols}")

# ── 2. Average over all networks of the same age FIRST ───────────────────────
props_with_age = huge_df_properties.copy()
props_with_age["age"] = huge_df["age"].values
age_means_raw = props_with_age.groupby("age").mean().reset_index().sort_values("age")

age_vals    = age_means_raw["age"].values
feature_mat = age_means_raw.drop(columns="age").values

# ── 3. Scale & PCA on the age-averaged data ───────────────────────────────────
scaler = StandardScaler()
scaled = scaler.fit_transform(feature_mat)

pca = PCA(n_components=min(10, scaled.shape[0]))
pca_result = pca.fit_transform(scaled)

age_means = pd.DataFrame({
    "age": age_vals,
    "PC1": pca_result[:, 0],
    "PC2": pca_result[:, 1],
})

# ── 4. Smooth spline through age-ordered PC scores ───────────────────────────
t = np.linspace(age_vals.min(), age_vals.max(), 300)
smooth_pc1 = make_smoothing_spline(age_vals, age_means["PC1"].values, lam=5)(t)
smooth_pc2 = make_smoothing_spline(age_vals, age_means["PC2"].values, lam=5)(t)

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

norm = plt.Normalize(age_vals.min(), age_vals.max())
cmap = plt.cm.viridis

# Smoothed trajectory coloured by age
points   = np.array([smooth_pc1, smooth_pc2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, alpha=0.8, zorder=1)
lc.set_array(t)
ax.add_collection(lc)

# One dot per age
sc = ax.scatter(age_means["PC1"], age_means["PC2"],
                c=age_means["age"], 
                cmap=cmap, norm=norm,
                s=40, zorder=2, edgecolors="white", linewidths=0.4)

ax.autoscale()
fig.colorbar(sc, ax=ax, label="Age")
ax.set_title("Averaged per age (before PCA)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# ── 6. Scree plot ─────────────────────────────────────────────────────────────
explained_variance = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=[f"PC{i+1}" for i in range(len(explained_variance))],
            y=explained_variance, ax=ax)
ax.plot(range(len(explained_variance)), np.cumsum(explained_variance),
        marker="o", color="red", label="Cumulative")
ax.set_title("Scree Plot")
ax.set_ylabel("Explained Variance Ratio")
ax.set_xlabel("Principal Components")
ax.legend()
plt.tight_layout()
plt.show()


# In[55]:


# Look at huge_df where age is defined. Then plot the pca space colored by age.
huge_df_age = huge_df[huge_df["age"].notna()]
plt.figure(figsize=(6,5))
plt.scatter(sel_pca_result[huge_df["age"].notna(), 0], sel_pca_result[huge_df["age"].notna(), 1],
            c=huge_df_age["age"], 
            cmap="hot", # viridis", 
            s=5, 
            alpha=0.4) # s=20, edgecolors="none")
plt.colorbar(label="Age")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA colored by age (only datasets with age info)")
plt.tight_layout()


# In[56]:


# Look at huge_df where age is defined. Then plot the pca space colored by age.
huge_df_age = huge_df[huge_df["age"].notna()]
plt.figure(figsize=(6,5))
plt.scatter(sel_pca_result[huge_df["age"].notna(), 0], sel_pca_result[huge_df["age"].notna(), 1],
            c=huge_df["age"], # [huge_df_age["age"].loc[i] for i in huge_df["age"].notna() if huge_df_age["age"].loc[i] != -1 else 0], 
            cmap="hot", # viridis", 
            s=5, 
            alpha=0.4) # s=20, edgecolors="none")
plt.colorbar(label="Age")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA colored by age (only datasets with age info)")
plt.tight_layout()


# In[ ]:


# Now in 3D: Look at huge_df where age is defined. Then plot the pca space colored by age. Make it interactive. 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(sel_pca_result[huge_df["age"].notna(), 0], 
                     sel_pca_result[huge_df["age"].notna(), 1], 
                     sel_pca_result[huge_df["age"].notna(), 2],
                     c=huge_df_age["age"], 
                     cmap="hot",
                     s=20,
                     alpha=0.6)
fig.colorbar(scatter, label="Age")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA colored by age (only datasets with age info)")
plt.tight_layout()
plt.show()


# In[ ]:


# Now in 3D: Look at huge_df where age is defined. Then plot the pca space colored by age. Make it interactive. 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(sel_pca_result[:, 0], 
                     sel_pca_result[:, 1], 
                     sel_pca_result[:, 2],
                     c=huge_df["age"], 
                     cmap="hot",
                     s=20,
                     alpha=0.6)
fig.colorbar(scatter, label="Age")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA colored by age (only datasets with age info)")
plt.tight_layout()
plt.show()


# In[ ]:


import plotly.express as px
import pandas as pd

plot_df = pd.DataFrame({
    'PC1': sel_pca_result[:, 0],
    'PC2': sel_pca_result[:, 1],
    'PC3': sel_pca_result[:, 2],
    'age': huge_df["age"].values
})

fig = px.scatter_3d(
    plot_df, x='PC1', y='PC2', z='PC3',
    color='age',
    color_continuous_scale='hot',
    opacity=0.6,
    size_max=10,
    title='3D PCA colored by age (only datasets with age info)'
)
fig.update_traces(marker=dict(size=3))
fig.show()


# In[ ]:


import umap
import matplotlib.pyplot as plt
import numpy as np

# Fit UMAP on all data, then visualize age-labeled subset
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

# Fit on the full dataset (or just age subset — your choice, see note below)
age_mask = huge_df["age"].notna()
embedding = reducer.fit_transform(sel_pca_result)  # use PCA result as input

# Plot only age-labeled points
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    # embedding[:, 0],
    # embedding[:, 1],
    embedding[age_mask, 0],
    embedding[age_mask, 1],
    c=huge_df_age["age"],
    cmap="hot",
    s=10,
    alpha=0.6
)
plt.colorbar(sc, label="Age")
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("UMAP colored by age")
plt.tight_layout()
plt.show()


# In[ ]:


import umap
import matplotlib.pyplot as plt
import numpy as np

# Fit UMAP on all data, then visualize age-labeled subset
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

# Fit on the full dataset (or just age subset — your choice, see note below)
embedding = reducer.fit_transform(sel_pca_result)  # use PCA result as input

# Plot only age-labeled points
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=huge_df_age["age"],
    cmap="hot",
    s=10,
    alpha=0.6
)
plt.colorbar(sc, label="Age")
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("UMAP colored by age")
plt.tight_layout()
plt.show()


# In[ ]:


# PHATE is actually purpose-built for continuous gradients
import phate
phate_op = phate.PHATE(n_components=2, random_state=42)
embedding_phate = phate_op.fit_transform(sel_pca_result[age_mask])


# In[ ]:


plt.scatter(embedding_phate[:, 0], embedding_phate[:, 1], c=huge_df_age["age"], cmap="hot", s=10, alpha=0.6)


# In[ ]:


age_mask


# In[ ]:


# Which datasets are in that isolated cluster?
from sklearn.cluster import DBSCAN

clusters = DBSCAN(eps=0.5, min_samples=5).fit_predict(embedding)
# print(huge_df_age[age_mask].groupby(clusters)["dataset"].value_counts())  # or species/source column


# In[ ]:


clusters


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: full view
# Right: zoomed to main cluster
main = (embedding[:, 0] > -2) & (embedding[:, 1] > 10)  # adjust to your blob

for ax, mask in zip(axes, [age_mask, age_mask & main]):
    sc = ax.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=huge_df_age["age"][mask],  # adjust indexing
                    cmap="plasma",   # better perceptual uniformity than "hot"
                    s=15, alpha=0.7,
                    vmin=np.percentile(huge_df_age["age"], 5),   # clip outliers
                    vmax=np.percentile(huge_df_age["age"], 95))
    plt.colorbar(sc, ax=ax, label="Age")

axes[0].set_title("Full UMAP")
axes[1].set_title("Main cluster zoomed")
plt.tight_layout()
plt.show()


# In[ ]:


import seaborn as sns

age_bins = pd.cut(huge_df_age["age"], bins=[0, 30, 50, 70, 100], 
                  labels=["<30", "30-50", "50-70", ">70"])

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
for ax, (label, grp) in zip(axes, huge_df_age.assign(bin=age_bins).groupby("bin")):
    idx = grp.index  # make sure this aligns with embedding rows
    sns.kdeplot(x=embedding[idx, 0], y=embedding[idx, 1], ax=ax, fill=True)
    ax.set_title(f"Age {label} (n={len(grp)})")
plt.tight_layout()


# In[ ]:


from sklearn.metrics import pairwise_distances
from scipy.stats import f_oneway
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle


X = pca_result[huge_df["age"].notna()][:, :2]  # PC1 + PC2 only
labels = sel_df_label[huge_df["age"].notna()]["dataset"].values

# # ── Extract the relevant PCA coordinates + labels ─────────────────────────────

# mami_mask  = (sel_df_label["dataset"] == "suarez_MaMI_dataset").values
# tax_mask   = df_mami[taxonomy].isin(GROUPS_TO_INCLUDE).values
# combined   = mami_mask.copy()
# combined[mami_mask] = tax_mask          # both conditions

# X      = sel_pca_result[combined, :2]   # PC1 + PC2 only
# labels = df_mami[tax_mask][taxonomy].values

# # ── Test 1: PERMANOVA (permutation-based MANOVA on distances) ─────────────────
# # "Are the groups more separated than random chance?"

# from sklearn.utils import shuffle

def permanova(X, labels, n_permutations=999):
    def f_stat(X, labels):
        groups  = [X[labels == g] for g in np.unique(labels)]
        grand_m = X.mean(axis=0)
        ss_between = sum(len(g) * ((g.mean(0) - grand_m)**2).sum() for g in groups)
        ss_within  = sum(((g - g.mean(0))**2).sum() for g in groups)
        k, n = len(groups), len(X)
        return (ss_between / (k-1)) / (ss_within / (n-k))

    observed = f_stat(X, labels)
    null_dist = [f_stat(X, shuffle(labels)) for _ in range(n_permutations)]
    p = (np.sum(np.array(null_dist) >= observed) + 1) / (n_permutations + 1)
    return observed, p

f_obs, p_perm = permanova(X, labels)
print(f"PERMANOVA  →  F = {f_obs:.3f},  p = {p_perm:.4f}")

# ── Test 2: Per-axis ANOVA ─────────────────────────────────────────────────────
# "Does any single PC axis separate the groups?"

groups_pc1 = [X[labels == g, 0] for g in np.unique(labels)]
groups_pc2 = [X[labels == g, 1] for g in np.unique(labels)]
f1, p1 = f_oneway(*groups_pc1)
f2, p2 = f_oneway(*groups_pc2)
print(f"ANOVA PC1  →  F = {f1:.3f},  p = {p1:.4f}")
print(f"ANOVA PC2  →  F = {f2:.3f},  p = {p2:.4f}")

# ── Test 3: LDA cross-validated accuracy ──────────────────────────────────────
# "Can we predict group membership better than chance?"

from sklearn.model_selection import cross_val_score
lda      = LinearDiscriminantAnalysis()
cv_score = cross_val_score(lda, X, labels, cv=5, scoring="accuracy").mean()
chance   = 1 / len(np.unique(labels))
print(f"LDA 5-fold CV accuracy = {cv_score:.3f}  (chance = {chance:.3f})")


# In[ ]:


from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np

age_mask = huge_df["age"].notna()
X = huge_df[age_mask].drop(columns=["dataset", "age"]) #  sel_pca_result[age_mask]  # using PCA result as input (50 components recommended)
ages = huge_df_age["age"].values

# ── 1. t-SNE ──────────────────────────────────────────────────────────────────
# Try multiple perplexity values — this is the key hyperparameter
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, perp in zip(axes, [15, 30, 50]):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42) # , n_iter=1000)
    emb = tsne.fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=ages, # ages,
                    cmap="plasma", s=10, alpha=0.6,
                    vmin=np.percentile(ages, 5),
                    vmax=np.percentile(ages, 95))
    plt.colorbar(sc, ax=ax, label="Age")
    ax.set_title(f"t-SNE (perplexity={perp})")
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")

plt.suptitle("t-SNE colored by age", y=1.02)
plt.tight_layout()
plt.show()

# ── 2. Kernel PCA ─────────────────────────────────────────────────────────────
# Try rbf (non-linear) vs linear (should match regular PCA) vs poly
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, kernel in zip(axes, ["rbf", "poly", "cosine"]):
    kpca = KernelPCA(n_components=2, kernel=kernel, random_state=42)
    emb = kpca.fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=ages, cmap="plasma", s=10, alpha=0.6,
                    vmin=np.percentile(ages, 5),
                    vmax=np.percentile(ages, 95))
    plt.colorbar(sc, ax=ax, label="Age")
    ax.set_title(f"Kernel PCA ({kernel})")
    ax.set_xlabel("KPC1")
    ax.set_ylabel("KPC2")

plt.suptitle("Kernel PCA colored by age", y=1.02)
plt.tight_layout()
plt.show()

# ── 3. Quantify: can Kernel PCA components predict age? ───────────────────────
# This is the key advantage of KernelPCA over UMAP/tSNE — you CAN regress on it
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

results = {}
for kernel in ["linear", "rbf", "poly", "cosine"]:
    kpca = KernelPCA(n_components=10, kernel=kernel, random_state=42)
    X_kpca = kpca.fit_transform(X)
    cv_r2 = cross_val_score(LinearRegression(), X_kpca, ages, cv=5, scoring="r2")
    results[kernel] = (cv_r2.mean(), cv_r2.std())
    print(f"Kernel={kernel:8s} | CV R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")


# In[ ]:


age_mask = huge_df["age"] != -1
age_mask


# In[ ]:


# 3D plot of KPCA (cosine) colored by age
from mpl_toolkits.mplot3d import Axes3D
kpca = KernelPCA(n_components=3, kernel="cosine", random_state=42)
X_kpca = kpca.fit_transform(X)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

age_mask = huge_df["age"] != -1

scatter = ax.scatter(X_kpca[age_mask][:, 0], 
                     X_kpca[age_mask][:, 1], 
                     X_kpca[age_mask][:, 2],
                     c=ages, 
                     cmap="plasma",
                     s=20, alpha=0.6)
fig.colorbar(scatter, label="Age")
ax.set_xlabel("KPC1")
ax.set_ylabel("KPC2")
ax.set_zlabel("KPC3")
ax.set_title("Kernel PCA (cosine) colored by age")
plt.tight_layout()
plt.show()


# In[ ]:


# 3D plot of KPCA (poly) colored by age
from mpl_toolkits.mplot3d import Axes3D
kpca = KernelPCA(n_components=3, kernel="poly", random_state=42)
X_kpca = kpca.fit_transform(X)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

age_mask = huge_df["age"] != -1

scatter = ax.scatter(X_kpca[age_mask][:, 0], 
                     X_kpca[age_mask][:, 1], 
                     X_kpca[age_mask][:, 2],
                     c=ages, 
                     cmap="plasma",
                     s=20, alpha=0.6)
fig.colorbar(scatter, label="Age")
ax.set_xlabel("KPC1")
ax.set_ylabel("KPC2")
ax.set_zlabel("KPC3")
ax.set_title("Kernel PCA (cosine) colored by age")
plt.tight_layout()
plt.show()


# In[ ]:


# Color your existing UMAP/t-SNE by dataset instead of age
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: colored by dataset
sc1 = axes[0].scatter(embedding[age_mask, 0], embedding[age_mask, 1],
                       c=pd.factorize(huge_df_age[age_mask]["dataset"])[0],
                       cmap="tab20", s=10, alpha=0.6)
axes[0].set_title("Colored by dataset")

# Right: colored by age
sc2 = axes[1].scatter(embedding[age_mask, 0], 
                      embedding[age_mask, 1],
                       c=ages, # [age_mask], 
                       cmap="plasma", s=10, alpha=0.6)
axes[1].set_title("Colored by age")

plt.tight_layout()
plt.show()


# In[ ]:


huge_df.keys()


# In[ ]:


plt.scatter(embedding[:,0], 
            embedding[:,1], 
            c=huge_df["age"].values,
            s=1, 
            alpha=0.5,
            )


# In[ ]:


plt.scatter(embedding[:,0], 
            embedding[:,1], 
            c=huge_df["age"].values,
            s=3, 
            alpha=0.5,
            cmap="hot",
            )

plt.xlim(-2, 2)
plt.ylim(11, 14)


# In[ ]:


colors_dataset = [COLOR_SCHEME[d] for d in huge_df["dataset"]]
colors_dataset = []

plt.scatter(embedding[:,0], 
            embedding[:,1], 
            c=colors_dataset,
            s=1, 
            alpha=0.5,
            )

# Add legend for datasets
unique_datasets = huge_df["dataset"].unique()
color_map = plt.cm.get_cmap("tab20", len(unique_datasets))
dataset_colors = {ds: color_map(i) for i, ds in enumerate(unique_datasets)}
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=ds,
                              markerfacecolor=dataset_colors[ds], markersize=10)
                    for ds in unique_datasets],
           title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP colored by age and dataset")


# In[ ]:


# plot each dataset separately in a grid of subplots
unique_datasets = huge_df["dataset"].unique()
n = len(unique_datasets)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows), sharex=True, sharey=True)

for ax, dataset in zip(axes.flatten(), unique_datasets):
    mask = huge_df["dataset"] == dataset
    # convert to boolean array
    mask = mask.values.astype(bool)

    ax.scatter(embedding[mask, 0], 
               embedding[mask, 1], 
               c=COLOR_SCHEME[dataset], 
               s=4, #1
               alpha=0.5)
    
    ax.set_title(dataset)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")


# In[ ]:


fig, axs = plt.subplots(1, 4, figsize=(14, 5))

for i, dataset in enumerate(huge_df_age[age_mask]["dataset"].unique()): # Left: colored by dataset
    that_dataset_mask = (huge_df_age[age_mask]["dataset"] == dataset)
    sc1 = axs[i].scatter(embedding[that_dataset_mask, 0], 
                         embedding[that_dataset_mask, 1],
                        c=i, # pd.factorize(huge_df_age[that_dataset_mask][age_mask]["dataset"])[0],
                        cmap="tab20", s=10, alpha=0.6)
    axs[i].set_title("Colored by dataset")


# In[ ]:


# Create a PCA between the columns of the huge_df
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(huge_df.drop("dataset", "age")) # _properties)

pca = PCA(n_components=10) # 10)
pca_result = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(x=pca_result[:,0], 
            y=pca_result[:,1], 
            s=2, 
            alpha=0.4, 
            c=huge_df['color_dataset'], 
            # label=huge_df['dataset']
        )
plt.title("PCA of Metrics")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.show()

# Scree plot to show explained variance
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(6,4))
sns.barplot(x=[f"PC{i+1}" for i in range(len(explained_variance))], y=explained_variance)
plt.plot(range(0, len(explained_variance)), np.cumsum(explained_variance), marker="o", color="red",
        label="Cumulative")
plt.title("Scree Plot")
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np

# age_mask = huge_df["age"].notna()
X = huge_df.drop(columns=["dataset", "age"]) #  sel_pca_result[age_mask]  # using PCA result as input (50 components recommended)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ages = huge_df_age["age"].values
ages_or_gray = huge_df["age"].values  # use age where available, else gray
ages_or_gray[huge_df["age"].isna()] = -1  # set missing ages to -1 for gray color
# ── 1. t-SNE ──────────────────────────────────────────────────────────────────
# Try multiple perplexity values — this is the key hyperparameter
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, perp in zip(axes, [15, 30, 50]):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42) # , n_iter=1000)
    emb = tsne.fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=ages_or_gray, cmap="plasma", s=10, alpha=0.6,
                    vmin=np.percentile(ages_or_gray, 5),
                    vmax=np.percentile(ages_or_gray, 95))
    plt.colorbar(sc, ax=ax, label="Age")
    ax.set_title(f"t-SNE (perplexity={perp})")
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")

plt.suptitle("t-SNE colored by age", y=1.02)
plt.tight_layout()
plt.show()


# In[ ]:


# Try multiple perplexity values — this is the key hyperparameter
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, perp in zip(axes, [15, 30, 50]):
    # tsne = TSNE(n_components=2, perplexity=perp, random_state=42) # , n_iter=1000)
    # emb = tsne.fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=ages_or_gray, cmap="plasma", s=10, alpha=0.6,
                    vmin=np.percentile(ages_or_gray, 5),
                    vmax=np.percentile(ages_or_gray, 95))
    plt.colorbar(sc, ax=ax, label="Age")
    ax.set_xlim(-30, 80)
    ax.set_ylim(-90, -50)
    ax.set_title(f"t-SNE (perplexity={perp})")
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")


# In[ ]:


# ── 2. Kernel PCA ─────────────────────────────────────────────────────────────
# Try rbf (non-linear) vs linear (should match regular PCA) vs poly
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, kernel in zip(axes, ["rbf", "poly", "cosine"]):
    kpca = KernelPCA(n_components=2, kernel=kernel, random_state=42)
    emb = kpca.fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=ages, cmap="plasma", s=10, alpha=0.6,
                    vmin=np.percentile(ages, 5),
                    vmax=np.percentile(ages, 95))
    plt.colorbar(sc, ax=ax, label="Age")
    ax.set_title(f"Kernel PCA ({kernel})")
    ax.set_xlabel("KPC1")
    ax.set_ylabel("KPC2")

plt.suptitle("Kernel PCA colored by age", y=1.02)
plt.tight_layout()
plt.show()

# ── 3. Quantify: can Kernel PCA components predict age? ───────────────────────
# This is the key advantage of KernelPCA over UMAP/tSNE — you CAN regress on it
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

results = {}
for kernel in ["linear", "rbf", "poly", "cosine"]:
    kpca = KernelPCA(n_components=10, kernel=kernel, random_state=42)
    X_kpca = kpca.fit_transform(X)
    cv_r2 = cross_val_score(LinearRegression(), X_kpca, ages, cv=5, scoring="r2")
    results[kernel] = (cv_r2.mean(), cv_r2.std())
    print(f"Kernel={kernel:8s} | CV R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")


# In[ ]:


from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np

# age_mask = huge_df["age"].notna()
X = sel_pca_result # [age_mask]  # using PCA result as input (50 components recommended)
ages = huge_df_age["age"].values

# ── 1. t-SNE ──────────────────────────────────────────────────────────────────
# Try multiple perplexity values — this is the key hyperparameter
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, perp in zip(axes, [15, 30, 50]):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42) # , n_iter=1000)
    emb = tsne.fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=ages, cmap="plasma", s=10, alpha=0.6,
                    vmin=np.percentile(ages, 5),
                    vmax=np.percentile(ages, 95))
    plt.colorbar(sc, ax=ax, label="Age")
    ax.set_title(f"t-SNE (perplexity={perp})")
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")

plt.suptitle("t-SNE colored by age", y=1.02)
plt.tight_layout()
plt.show()

# ── 2. Kernel PCA ─────────────────────────────────────────────────────────────
# Try rbf (non-linear) vs linear (should match regular PCA) vs poly
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, kernel in zip(axes, ["rbf", "poly", "cosine"]):
    kpca = KernelPCA(n_components=2, kernel=kernel, random_state=42)
    emb = kpca.fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=ages, cmap="plasma", s=10, alpha=0.6,
                    vmin=np.percentile(ages, 5),
                    vmax=np.percentile(ages, 95))
    plt.colorbar(sc, ax=ax, label="Age")
    ax.set_title(f"Kernel PCA ({kernel})")
    ax.set_xlabel("KPC1")
    ax.set_ylabel("KPC2")

plt.suptitle("Kernel PCA colored by age", y=1.02)
plt.tight_layout()
plt.show()

# ── 3. Quantify: can Kernel PCA components predict age? ───────────────────────
# This is the key advantage of KernelPCA over UMAP/tSNE — you CAN regress on it
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

results = {}
for kernel in ["linear", "rbf", "poly", "cosine"]:
    kpca = KernelPCA(n_components=10, kernel=kernel, random_state=42)
    X_kpca = kpca.fit_transform(X)
    cv_r2 = cross_val_score(LinearRegression(), X_kpca, ages, cv=5, scoring="r2")
    results[kernel] = (cv_r2.mean(), cv_r2.std())
    print(f"Kernel={kernel:8s} | CV R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")


# In[ ]:


import cebra
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
import pandas as pd

# ── Setup ──────────────────────────────────────────────────────────────────────
age_mask = huge_df["age"] != -1  #  huge_df["age"].notna() &only rows with age and dataset info
X = sel_pca_result[age_mask].astype(np.float32)        # input: PCA features
ages = huge_df_age[age_mask]["age"].values.astype(np.float32)    # continuous label
groups = huge_df_age[age_mask]["dataset"].values              # for dataset-aware CV

# ── 1. CEBRA with age as the conditioning variable (supervised) ────────────────
cebra_model = cebra.CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=1e-3,
    temperature=1.0,
    output_dimension=3,          # 3D embedding
    max_iterations=5000,
    verbose=True,
    device="cpu"                 # change to "cuda" if available
)

# Fit conditioned on age — this is the key difference from PCA/UMAP/t-SNE
cebra_model.fit(X, ages)
embedding_age = cebra_model.transform(X)

# ── 2. CEBRA without label (unsupervised) for comparison ──────────────────────
cebra_model_unsup = cebra.CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=1e-3,
    output_dimension=3,
    max_iterations=5000,
    verbose=True,
    device="cpu"
)
cebra_model_unsup.fit(X)
embedding_unsup = cebra_model_unsup.transform(X)

# ── 3. Visualize: supervised vs unsupervised, colored by age AND dataset ───────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

dataset_ids = pd.factorize(groups)[0]

plots = [
    (embedding_age,  ages,        "plasma", "CEBRA (age-supervised) — colored by age"),
    (embedding_age,  dataset_ids, "tab20",  "CEBRA (age-supervised) — colored by dataset"),
    (embedding_unsup, ages,       "plasma", "CEBRA (unsupervised) — colored by age"),
    (embedding_unsup, dataset_ids,"tab20",  "CEBRA (unsupervised) — colored by dataset"),
]

for ax, (emb, color, cmap, title) in zip(axes.flat, plots):
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=color, cmap=cmap, s=10, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Age" if cmap == "plasma" else "Dataset")
    ax.set_title(title)
    ax.set_xlabel("CEBRA1")
    ax.set_ylabel("CEBRA2")

plt.suptitle("CEBRA embeddings: supervised vs unsupervised", y=1.01, fontsize=14)
plt.tight_layout()
plt.show


# In[ ]:


# VON CLAUDE; NOCH NICHT AUSPROBIERT 
# 
# # ── Per-category loading vector composition ───────────────────────────────────
#   loadings = loading_and_cat_df  # already computed above

#   cat_vector_folder = OUTPUT_FOLDER / "category_vectors"
#   cat_vector_folder.mkdir(exist_ok=True)

#   for target_cat in CATEGORY_COLOURS.keys():
#       colour = CATEGORY_COLOURS[target_cat]

#       cat_vars = [c for c in loadings.index
#                   if c in meta.index and meta.loc[c, "Category"] == target_cat]
#       if not cat_vars:
#           continue

#       cat_loads = loadings.loc[cat_vars]

#       # ── Pre-compute cumulative path for axis limits ───────────────────────
#       cum_x, cum_y = [0], [0]
#       cx, cy = 0, 0
#       for _, row in cat_loads.iterrows():
#           cx += row["PC1"]
#           cy += row["PC2"]
#           cum_x.append(cx)
#           cum_y.append(cy)

#       max_lim = max(max(map(abs, cum_x)), max(map(abs, cum_y)), 0.1) * 1.3

#       fig, ax = plt.subplots(figsize=viz.cm_to_inch((8, 8)), dpi=150)
#       ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
#       ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
#       ax.scatter([0], [0], color="black", s=20, zorder=6)

#       current_x, current_y = 0, 0
#       for var_name, row in cat_loads.iterrows():
#           vx, vy = row["PC1"], row["PC2"]

#           ax.quiver(
#               current_x, current_y, vx, vy,
#               angles="xy", scale_units="xy", scale=1,
#               color=colour, width=0.008, alpha=0.6,
#               headwidth=4, headlength=5, headaxislength=4, zorder=4,
#           )

#           nice_name = PROPERTY_NAMES.get(var_name, var_name)
#           mid_x = current_x + vx / 2
#           mid_y = current_y + vy / 2
#           offset_x = 0.02 * np.sign(vx) if vx != 0 else 0.02
#           offset_y = 0.02 * np.sign(vy) if vy != 0 else 0.02
#           ax.text(mid_x + offset_x, mid_y + offset_y, nice_name,
#                   fontsize=7, ha="center", va="center", color=colour, alpha=0.9,
#                   bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
#   pad=0.5))

#           current_x += vx
#           current_y += vy

#       # Final sum vector
#       ax.quiver(
#           0, 0, current_x, current_y,
#           angles="xy", scale_units="xy", scale=1,
#           color="black", width=0.012,
#           headwidth=4, headlength=5, headaxislength=4, zorder=5,
#       )
#       nudge_x = 0.05 * np.sign(current_x) if current_x != 0 else 0.05
#       nudge_y = 0.05 * np.sign(current_y) if current_y != 0 else 0.05
#       ax.text(current_x + nudge_x, current_y + nudge_y, f"Total {target_cat}",
#               fontsize=9, ha="center", va="center", color="black",
#   fontweight="bold")

#       ax.set_xlim(-max_lim, max_lim)
#       ax.set_ylim(-max_lim, max_lim)
#       ax.set_aspect("equal")
#       ax.set_xlabel("PC1", fontsize=9)
#       ax.set_ylabel("PC2", fontsize=9)
#       ax.set_title(f"'{target_cat}' vectors in PCA space", fontsize=10)
#       for spine in ax.spines.values():
#           spine.set_visible(False)
#       ax.tick_params(labelsize=8)

#       plt.tight_layout()
#       out_path = cat_vector_folder / f"cat_vectors_{target_cat.lower().replace('
#   ', '_')}.pdf"
#       plt.savefig(out_path, dpi=150, bbox_inches="tight")
#       plt.show()
#       print(f"Saved: {out_path}")

