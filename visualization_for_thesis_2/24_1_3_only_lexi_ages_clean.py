#!/usr/bin/env python
# coding: utf-8

# In[21]:


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

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

# Create a PCA between the columns of the huge_df
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.interpolate import make_smoothing_spline  # or UnivariateSpline
from matplotlib.collections import LineCollection

print(os.getcwd())  # Should show the project root 

from config import COLORS, COLOR_SCHEME, LABEL_MAP, gray_cmap, bone_white, half_black, emp_dataset_and_experiment_pairs
from config import CATEGORY_ORDER, CATEGORY_COLOURS # Integration etc.
import re
from config import PROPERTY_NAMES, remaining_categories, REPRESENTATIVES_FOR_GOALS, selected_properties

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[22]:


output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis")
output_folder.mkdir(exist_ok=True)


with open(output_folder / "all_datasets_precise_categories.pkl", "rb") as f:
    dict_with_all_datasets = pickle.load(f)
    

output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/ages")
output_folder.mkdir(exist_ok=True)


# In[23]:


ordered_dataset_names = [# 'hcp_schaefer_100_dataset_gnm', 
                        # 'hcp_schaefer_100_dataset', 
                        # 'suarez_MaMI_dataset', 
                        'lexis_data_young', 
                        # 'lexis_data_aging',
                        # 'lexis_data_developing', 
                        # 'kaysons_generated_networks_diffusion', 
                        # 'kaysons_generated_networks_propagation', 
                        # 'kaysons_generated_networks_routing', 
                        # 'kaysons_generated_networks_topology', 
                        ]


# In[24]:


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


# In[25]:


precise_categories = []
for cat_name, cat_cols in remaining_categories.items(): 
    
    for col in cat_cols:
        precise_categories.append(col)


# precise_categories
len(precise_categories)

# For pca always drop the dataset column and only use the metric columns
huge_df_properties = huge_df.drop(columns=["dataset"]) # , "color_dataset"])


# In[26]:


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


# In[27]:


# Apply the selection
huge_df_properties = huge_df_properties[remaining_cols]


# In[19]:


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

# Print the names of the variables that are in our data but not in the metadata, to check for mismatches
missing_vars = [c for c in huge_df_properties.columns if c not in meta.index]
if missing_vars:
    print("Missing variables in metadata:")
    print("\n".join(missing_vars))

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
    interpolation="nearest",
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

plt.savefig(output_folder / f"{dataset_name}_corr_matrix_categorised_manually.pdf", bbox_inches="tight", dpi=300)
print(output_folder / f"{dataset_name}_corr_matrix_categorised_manually.pdf")
plt.show()


# In[20]:


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

plt.savefig(output_folder / f"{dataset_name}_corr_matrix_clustered_automatically.pdf", bbox_inches="tight")
print(output_folder / f"{dataset_name}_corr_matrix_clustered_automatically.pdf")
plt.show()

# ── 7. Print cluster compositions ─────────────────────────────────────────────
for cid in sorted(cluster_row_spans):
    members = [PROPERTY_NAMES.get(c, c) for c in final_order
               if orig_to_cluster[c] == cid]
    print(f"\nCluster {cid} ({len(members)} vars):")
    print("  " + ", ".join(members))


# In[28]:


# ── 1. Full-matrix hierarchical clustering ────────────────────────────────────
cols_in_data = [c for c in huge_df_properties.columns if c in meta.index]
corr_full    = huge_df_properties[cols_in_data].corr()

dist = np.clip(1 - corr_full.values, 0, 2)
# dist = np.clip(1 - np.abs(corr_full.values), 0, 1)  # was: 1 - corr_full.values

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

plt.savefig(output_folder / f"{dataset_name}_corr_matrix_clustered_automatically_not_abs.pdf", bbox_inches="tight")
print(output_folder / f"{dataset_name}_corr_matrix_clustered_automatically_not_abs.pdf")
plt.show()

# ── 7. Print cluster compositions ─────────────────────────────────────────────
for cid in sorted(cluster_row_spans):
    members = [PROPERTY_NAMES.get(c, c) for c in final_order
               if orig_to_cluster[c] == cid]
    print(f"\nCluster {cid} ({len(members)} vars):")
    print("  " + ", ".join(members))


# In[10]:


huge_df_properties['proportion_long_range_connections_0.3956'].isna().sum()


# In[11]:


huge_df_properties.columns 


# In[29]:


# ── 1. Full-matrix hierarchical clustering ────────────────────────────────────
cols_in_data = [c for c in huge_df_properties.columns if c in meta.index]
corr_full    = huge_df_properties[cols_in_data].corr()

# Fill NaNs before computing distance (NaN arises from constant or collinear columns)
corr_clean = corr_full.fillna(0)  # treat unknown correlation as uncorrelated

dist = np.clip(1 - corr_clean.values, 0, 2)
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

plt.savefig(output_folder / f"{dataset_name}_corr_matrix_clustered.pdf", bbox_inches="tight")
plt.show()

# ── 7. Print cluster compositions ─────────────────────────────────────────────
for cid in sorted(cluster_row_spans):
    members = [PROPERTY_NAMES.get(c, c) for c in final_order
               if orig_to_cluster[c] == cid]
    print(f"\nCluster {cid} ({len(members)} vars):")
    print("  " + ", ".join(members))


# In[13]:


huge_df["color_dataset"] = [COLOR_SCHEME[d] for d in huge_df["dataset"]]


# In[ ]:


MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)
  • mc_0
  • avg_clustering_static
  • avg_degree_static
  • modularity_static
  • transitivity_static
  • wiring_cost_static
  • global_efficiency_dynamic
  • synchronizability_eigenratio_lambda_2
  • synchronizability_eigenratio_lambda_N
  • spectral_gap_fatemeh
  • kuramoto_synchronization_r_final
  • kuramoto_synchronization_r_mean
  • kuramoto_synchronization_r_std
  • participation_coefficient_wmd_mean
  • algebraic_connectivity_nx
  • kernel_rank_max
  • repertoire_size
  • repertoire_diversity
  • mc_nonlinear_original_mc_mean
  • mc_nonlinear_original_mc_std
  • participation_coefficient_pc_mean_further
  • participation_coefficient_pc_std_further
  • participation_coefficient_pc_median_further
  • participation_coefficient_pc_frac_connector_further
  • participation_coefficient_wmd_mean_further
  • participation_coefficient_wmd_std_further
  • participation_coefficient_n_communities_further
  • persistent_homology_ph_h0_n_features
  • persistent_homology_ph_h0_persistence_mean
  • persistent_homology_ph_h0_persistence_std
  • persistent_homology_ph_h0_entropy
  • persistent_homology_ph_h1_persistence_std


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


# Subplots of loadings for the first 3 principal components, ONLY top 10 contributors
loadings = pca.components_.T[:, :]
num_metrics = loadings.shape[0]
metric_names = huge_df_properties.columns

# order them all by the absolute value of their loading on the first principal component
sorted_indices = np.argsort((loadings[:, 0])) # [::-1]
loadings = loadings[sorted_indices]
metric_names = metric_names[sorted_indices]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6)) #  dpi=100)
for i in range(3):
    axs[i].barh(range(num_metrics), loadings[:, i])
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(metric_names, rotation=0)
    axs[i].set_title(f"Loadings for PC{i+1}")
plt.tight_layout()
plt.show()


# In[18]:


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6)) # , dpi=100)
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


# In[19]:


fig, axs = plt.subplots(nrows=1, ncols=3, 
                        figsize=viz.cm_to_inch((40,12)), 
                        # sharey=True, 
                        sharex=True) # ,
                        # dpi=100)
for i in range(3):
    
    ordered_pc_i = np.argsort((loadings[:, i])) #  + loadings[:, 1])) #[::-1]
    loadings = loadings[ordered_pc_i]
    metric_names = metric_names[ordered_pc_i]
    metric_names_long = [PROPERTY_NAMES[m] for m in metric_names]

    axs[i].barh(range(num_metrics), loadings[:, i])
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(metric_names_long, rotation=0)
    axs[i].set_title(f"Loadings for PC{i+1}")

plt.tight_layout()
plt.show()


# In[20]:


import matplotlib.patches as mpatches

fig, axs = plt.subplots(nrows=1, ncols=3,
                        figsize=viz.cm_to_inch((40, 12)),
                        sharex=True) # , dpi=100)

for i in range(3):
    ordered_pc_i     = np.argsort(loadings[:, i])
    loadings_sorted  = loadings[ordered_pc_i]
    names_sorted     = metric_names[ordered_pc_i]
    names_long       = [PROPERTY_NAMES[m] for m in names_sorted]

    # Bar colour from category
    bar_colours = []
    for m in names_sorted:
        cat    = meta.loc[m, "Category"] if m in meta.index else None
        colour = CATEGORY_COLOURS.get(cat, "#AAAAAA")
        bar_colours.append(colour)

    axs[i].barh(range(num_metrics), loadings_sorted[:, i], color=bar_colours)
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(names_long, fontsize=7)
    axs[i].set_title(f"Loadings for PC{i+1}")
    axs[i].axvline(0, color="black", linewidth=0.6)

# Legend
handles = [mpatches.Patch(color=c, label=cat)
           for cat, c in CATEGORY_COLOURS.items()
           if cat in meta["Category"].values]
fig.legend(handles=handles, title="Category",
           bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=8)

plt.tight_layout()

plt.savefig(output_folder / f"{dataset_name}_pca_loadings_coloured_by_category.pdf", bbox_inches="tight")
print(output_folder / f"{dataset_name}_pca_loadings_coloured_by_category.pdf")
plt.show()


# In[21]:


fig, axs = plt.subplots(nrows=1, ncols=3,
                        figsize=viz.cm_to_inch((40, 12)),
                        sharex=True)

for i in range(3):
    ordered_pc_i    = np.argsort(loadings[:, i])
    loadings_sorted = loadings[ordered_pc_i]
    names_sorted    = metric_names[ordered_pc_i]
    names_long      = [PROPERTY_NAMES[m] for m in names_sorted]

    bar_colours = []
    for m in names_sorted:
        cat = meta.loc[m, "Category"] if m in meta.index else None
        bar_colours.append(CATEGORY_COLOURS.get(cat, "#AAAAAA"))

    axs[i].barh(range(num_metrics), loadings_sorted[:, i], color=bar_colours)
    axs[i].axvline(0, color="black", linewidth=0.6)
    axs[i].set_title(f"Loadings for PC{i+1}")

    # Remove default y tick labels
    axs[i].set_yticks([])

    PAD = 0.02  # small gap from the zero line in data units
    for j, (val, name, colour) in enumerate(zip(loadings_sorted[:, i], names_long, bar_colours)):
        if val >= 0:
            # Bar goes right → label on the left of zero
            axs[i].text(-PAD, j, name,
                        ha="right", va="center", fontsize=6.5,
                        color=colour, # fontweight="bold"
                        )
        else:
            # Bar goes left → label on the right of zero
            axs[i].text(PAD, j, name,
                        ha="left", va="center", fontsize=6.5,
                        color=colour,
                        # fontweight="bold"
                        )
            
    # Remove spines and ticks for a cleaner look
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['left'].set_visible(False)

# Legend
handles = [mpatches.Patch(color=c, label=cat)
           for cat, c in CATEGORY_COLOURS.items()
           if cat in meta["Category"].values]
fig.legend(handles=handles, title="Category",
           bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=8)

plt.tight_layout()
plt.savefig(output_folder / f"{dataset_name}_pca_loadings_coloured_by_category.pdf", bbox_inches="tight")
print(output_folder / f"{dataset_name}_pca_loadings_coloured_by_category.pdf")
plt.show()


# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


# from sklearn.metrics import pairwise_distances
# from scipy.stats import f_oneway
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

# def permanova(X, labels, n_permutations=999):
#     def f_stat(X, labels):
#         groups  = [X[labels == g] for g in np.unique(labels)]
#         grand_m = X.mean(axis=0)
#         ss_between = sum(len(g) * ((g.mean(0) - grand_m)**2).sum() for g in groups)
#         ss_within  = sum(((g - g.mean(0))**2).sum() for g in groups)
#         k, n = len(groups), len(X)
#         return (ss_between / (k-1)) / (ss_within / (n-k))

#     observed = f_stat(X, labels)
#     null_dist = [f_stat(X, shuffle(labels)) for _ in range(n_permutations)]
#     p = (np.sum(np.array(null_dist) >= observed) + 1) / (n_permutations + 1)
#     return observed, p

# f_obs, p_perm = permanova(X, labels)
# print(f"PERMANOVA  →  F = {f_obs:.3f},  p = {p_perm:.4f}")

# # ── Test 2: Per-axis ANOVA ─────────────────────────────────────────────────────
# # "Does any single PC axis separate the groups?"

# groups_pc1 = [X[labels == g, 0] for g in np.unique(labels)]
# groups_pc2 = [X[labels == g, 1] for g in np.unique(labels)]
# f1, p1 = f_oneway(*groups_pc1)
# f2, p2 = f_oneway(*groups_pc2)
# print(f"ANOVA PC1  →  F = {f1:.3f},  p = {p1:.4f}")
# print(f"ANOVA PC2  →  F = {f2:.3f},  p = {p2:.4f}")

# # ── Test 3: LDA cross-validated accuracy ──────────────────────────────────────
# # "Can we predict group membership better than chance?"

# from sklearn.model_selection import cross_val_score
# lda      = LinearDiscriminantAnalysis()
# cv_score = cross_val_score(lda, X, labels, cv=5, scoring="accuracy").mean()
# chance   = 1 / len(np.unique(labels))
# print(f"LDA 5-fold CV accuracy = {cv_score:.3f}  (chance = {chance:.3f})")


# # Ages 

# In[27]:


for dataset_name, d in dict_with_all_datasets.items():
    

    # if "lexis_data_young" in dataset_name:
    if "lexis_data_" in dataset_name:
    
            print(dataset_name)
            ages = np.load(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/{dataset_name}/04_further_info/00_ages.npy")
            # Turn dict_with_all_datasets["lexis_data_aging"].index into a mask
            
            mask = np.zeros_like(ages)
            mask[dict_with_all_datasets[dataset_name].index] = 1
            mask = mask.astype(bool)
            
            # add an age column by index to the main dataframe
            dict_with_all_datasets[dataset_name]["age"] = ages[mask]
        


# In[28]:


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
    


# In[29]:


huge_df


# In[30]:


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


# In[31]:


# # Create a PCA between the columns of the huge_df
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # Drop the columns that have inf or -inf values (if any). Print the dropped columns
# dropped_cols = huge_df_properties.columns[huge_df_properties.isnull().any()].tolist()
# huge_df_properties.replace([np.inf, -np.inf], np.nan, inplace=True)
# huge_df_properties.dropna(axis=1, inplace=True)
# print(f"Dropped columns: {dropped_cols}")

# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(huge_df_properties)

# pca = PCA(n_components=10) # 10)
# pca_result = pca.fit_transform(scaled_data)

# plt.figure(figsize=(8,6))
# plt.scatter(x=pca_result[:,0], 
#             y=pca_result[:,1], 
#             s=4,
#             alpha=0.4, 
#             c=huge_df["age"],
#             # c=huge_df['color_dataset'], 
#             # label=huge_df['dataset']
#         )
# plt.title("PCA of Metrics")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend()
# plt.tight_layout()
# plt.colorbar(label="Age") # if using age as color
# plt.show()


# In[32]:


len(huge_df_properties)


# In[33]:


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
cmap = plt.cm.terrain # viridis

# Line through the age-averaged points (sorted by age)
ax.plot(age_means["PC1"], age_means["PC2"],
        color="grey", linewidth=0.8, alpha=0.6, zorder=1)

# All individual ones
sc = ax.scatter(pca_df["PC1"], 
                pca_df["PC2"],
                c=pca_df["age"], cmap=cmap, norm=norm,
                s=5, 
                zorder=2, edgecolors="white", 
                linewidths=0.4)

# One dot per age, coloured by age
sc = ax.scatter(age_means["PC1"], age_means["PC2"],
                c=age_means["age"], cmap=cmap, norm=norm,
                s=40, zorder=2, edgecolors="white", linewidths=0.4)

plt.colorbar(sc, ax=ax, label="Age")
ax.set_title("PCA of Metrics — averaged per age")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
plt.tight_layout()

plt.savefig(output_folder / f"{dataset_name}_pca_age_scatter_individuals.pdf", bbox_inches="tight")
print(output_folder / f"{dataset_name}_pca_age_scatter_individuals.pdf")
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

plt.savefig(output_folder / f"{dataset_name}_pca_scree_plot.pdf", dpi=150, bbox_inches="tight")
print(output_folder / f"{dataset_name}_pca_scree_plot.pdf")
plt.show()


# In[ ]:





# In[34]:


age_vals = age_means["age"].values
t = np.linspace(age_vals.min(), age_vals.max(), 300)

smooth_pc1 = make_smoothing_spline(age_vals, age_means["PC1"].values, lam=5)(t)
smooth_pc2 = make_smoothing_spline(age_vals, age_means["PC2"].values, lam=5)(t)

norm = plt.Normalize(age_vals.min(), age_vals.max())
cmap = plt.cm.terrain

# ── Loadings ──────────────────────────────────────────────────────────────────
scale    = np.sqrt(pca.explained_variance_[:2])
loadings = pd.DataFrame(
    (pca.components_[:2] * scale[:, None]).T,
    index=huge_df_properties.columns,
    columns=["PC1", "PC2"]
)
n_largest = 2
top_pc1   = loadings["PC1"].abs().nlargest(n_largest).index.tolist()
top_pc2   = loadings["PC2"].abs().nlargest(n_largest).index.tolist()
top_vars  = list(dict.fromkeys(top_pc1 + top_pc2))

# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — age trajectory
# ════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=viz.cm_to_inch((12, 9)))

points   = np.array([smooth_pc1, smooth_pc2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, 
                    norm=norm, 
                    linewidth=4, # 2, # 1.5, 
                    alpha=1, # 0.8, 
                    zorder=2
                    )
lc.set_array(t)
ax.add_collection(lc)

sc = ax.scatter(age_means["PC1"], age_means["PC2"],
                c=age_means["age"], cmap=cmap, norm=norm,
                s=40, 
                zorder=3, 
                edgecolors="white", linewidths=0.4)

lines_ax = ax.plot(age_means["PC1"], 
             age_means["PC2"], # age_means is already ordered. 
                # c=age_means["age"], 
                # cmap=cmap, 
                # norm=norm,
                zorder=1, 
                linewidth=0.2,
                alpha=0.6,
                # edgecolors="white", 
                # linewidths=0.2
                )
# print(age_means["age"])
ax.autoscale()
fig1.colorbar(sc, ax=ax, label="Age")
ax.set_title(f"{dataset_name.replace('_', ' ').capitalize()}: Averaged per age (after PCA)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

fig1.tight_layout()
fig1.savefig(output_folder / f"{dataset_name}_pca_age.pdf") # , dpi=300)
print(output_folder / f"{dataset_name}_pca_age.pdf")
plt.show()


# In[35]:


from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import make_smoothing_spline

# ── Extend to PC3 ─────────────────────────────────────────────────────────────
pca_df["PC3"] = pca_result[:, 2]
age_means = pca_df.groupby("age")[["PC1", "PC2", "PC3"]].mean().reset_index().sort_values("age")

age_vals   = age_means["age"].values
t          = np.linspace(age_vals.min(), age_vals.max(), 300)
norm       = plt.Normalize(age_vals.min(), age_vals.max())
cmap       = plt.cm.terrain

smooth_pc1 = make_smoothing_spline(age_vals, age_means["PC1"].values, lam=5)(t)
smooth_pc2 = make_smoothing_spline(age_vals, age_means["PC2"].values, lam=5)(t)
smooth_pc3 = make_smoothing_spline(age_vals, age_means["PC3"].values, lam=5)(t)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=viz.cm_to_inch((14, 11)))
ax   = fig1.add_subplot(111, projection="3d")

# Smooth coloured tube via Line3DCollection
points   = np.array([smooth_pc1, smooth_pc2, smooth_pc3]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=3, alpha=0.9, zorder=2)
lc.set_array(t)
ax.add_collection3d(lc)

# Raw averaged dots
sc = ax.scatter(age_means["PC1"], age_means["PC2"], age_means["PC3"],
                c=age_means["age"], cmap=cmap, norm=norm,
                s=40, zorder=3, edgecolors="white", linewidths=0.4,
                depthshade=True)

# Faint raw polyline
ax.plot(age_means["PC1"], age_means["PC2"], age_means["PC3"],
        color="grey", linewidth=0.4, alpha=0.5, zorder=1)

# Shadow projections on each wall (optional but nice in 3D)
z_floor = age_means["PC3"].min() - 0.3
x_wall  = ax.get_xlim()[0] if ax.get_xlim()[0] != ax.get_xlim()[1] else age_means["PC1"].min() - 0.5
ax.plot(age_means["PC1"], age_means["PC2"],
        zs=z_floor, zdir="z",
        color="grey", linewidth=0.5, alpha=0.25)
ax.plot(age_means["PC1"], age_means["PC3"],
        zs=age_means["PC2"].max() + 0.3, zdir="y",
        color="grey", linewidth=0.5, alpha=0.25)

fig1.colorbar(sc, ax=ax, label="Age", shrink=0.5, pad=0.1)
ax.set_xlabel("PC1", labelpad=6)
ax.set_ylabel("PC2", labelpad=6)
ax.set_zlabel("PC3", labelpad=6)
ax.set_title(f"{dataset_name.replace('_', ' ').capitalize()}: Age trajectory (PC1–3)")

# Nice default viewing angle — adjust to taste
ax.view_init(elev=25, azim=-60)

fig1.tight_layout()
fig1.savefig(output_folder / f"{dataset_name}_pca_age_3d.pdf") # , dpi=300)
print(output_folder / f"{dataset_name}_pca_age_3d.pdf")
plt.show()


# In[36]:


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — loading vectors
# ════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=viz.cm_to_inch((6,6)))

ax2.axhline(0, color="grey", lw=0.5, alpha=0.5)
ax2.axvline(0, color="grey", lw=0.5, alpha=0.5)
ax2.scatter([0], [0], color="black", s=15, zorder=6)

label_positions = []  # track placed labels to nudge overlaps

from adjustText import adjust_text

texts = []
for var in top_vars:
    vx     = loadings.loc[var, "PC1"]
    vy     = loadings.loc[var, "PC2"]
    colour = "#E63946" if var in top_pc1 else "#457B9D"
    if var in top_pc1 and var in top_pc2:
        colour = "#9B2226"

    ax2.quiver(
        0, 0, vx, vy,
        angles="xy", scale_units="xy", scale=1,
        color=colour, width=0.012,
        headwidth=4, headlength=5, headaxislength=4,
        zorder=5,
    )
    texts.append((var, vx, vy, colour))

# ── Fan apart labels that are too close in angle ──────────────────────────────
MIN_LABEL_DIST = 0.25  # minimum distance between label positions
LABEL_RADIUS   = 1.35  # how far beyond arrow tip

label_positions = []
for var, vx, vy, colour in texts:
    norm_r = np.sqrt(vx**2 + vy**2)
    lx = vx / norm_r * norm_r * LABEL_RADIUS
    ly = vy / norm_r * norm_r * LABEL_RADIUS

    # Fan away from already-placed labels along the perpendicular
    for attempt in range(20):
        too_close = [(px, py) for px, py in label_positions
                     if np.sqrt((lx-px)**2 + (ly-py)**2) < MIN_LABEL_DIST]
        if not too_close:
            break
        # Rotate label position slightly around origin
        angle = np.arctan2(ly, lx) + 0.15
        r     = np.sqrt(lx**2 + ly**2)
        lx, ly = r * np.cos(angle), r * np.sin(angle)

    label_positions.append((lx, ly))
    label = PROPERTY_NAMES.get(var, var).replace(" ", "\n")
    ax2.text(lx, ly, label,
             fontsize=7, ha="center", va="center",
             color=colour, #  fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


# Axis limits: symmetric, tight around actual vectors + label room
all_vals = loadings.loc[top_vars, ["PC1", "PC2"]].abs().values.max() * 1.5
ax2.set_xlim(-all_vals, all_vals)
ax2.set_ylim(-all_vals, all_vals)
ax2.set_aspect("equal")
ax2.set_xlabel("PC1") # , fontsize=9)
ax2.set_ylabel("PC2") # , fontsize=9)
ax2.set_title(f"{dataset_name.replace('_', ' ').capitalize()}: Top PCA loadings") # , fontsize=9)
ax2.tick_params() # labelsize=7)


fig2.tight_layout()
# fig2.savefig(output_folder / f"{dataset_name}_pca_loadings.pdf", dpi=300)
# print(output_folder / f"{dataset_name}_pca_loadings.pdf")
plt.show()


# In[37]:


# ── Fan labels by enforcing minimum angular separation ────────────────────────
LABEL_RADIUS  = 1.2   # in data units, beyond arrow tip
MIN_ANG_GAP   = 0.8 # radians (~20°) minimum between labels

# Collect arrows + initial angles
arrow_info = []
for var in top_vars:
    vx = loadings.loc[var, "PC1"]
    vy = loadings.loc[var, "PC2"]
    colour = "#E63946" if var in top_pc1 else "#457B9D"
    if var in top_pc1 and var in top_pc2:
        colour = "#9B2226"
    angle = np.arctan2(vy, vx)
    r     = np.sqrt(vx**2 + vy**2)
    arrow_info.append(dict(var=var, vx=vx, vy=vy, colour=colour, angle=angle, r=r))

# Sort by angle so we can spread neighbours apart
arrow_info.sort(key=lambda d: d["angle"])

# Push angles apart until all gaps >= MIN_ANG_GAP (a few passes suffice)
label_angles = [d["angle"] for d in arrow_info]
for _ in range(50):
    changed = False
    for i in range(len(label_angles)):
        j = (i + 1) % len(label_angles)
        gap = label_angles[j] - label_angles[i]
        if i == len(label_angles) - 1:        # wrap-around pair
            gap += 2 * np.pi
        if gap < MIN_ANG_GAP:
            push = (MIN_ANG_GAP - gap) / 2
            label_angles[i] -= push
            label_angles[j] += push
            changed = True
    if not changed:
        break

# ── Draw ──────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=viz.cm_to_inch((7, 7)))
ax2.axhline(0, color="grey", lw=0.5, alpha=0.5)
ax2.axvline(0, color="grey", lw=0.5, alpha=0.5)
ax2.scatter([0], [0], color="black", s=15, zorder=6)

label_coords = []
for d, langle in zip(arrow_info, label_angles):
    ax2.quiver(0, 0, d["vx"], d["vy"],
               angles="xy", scale_units="xy", scale=1,
               color=d["colour"], width=0.012,
               headwidth=4, headlength=5, headaxislength=4, zorder=5)

    # Label position: at LABEL_RADIUS along the (possibly fanned) angle
    lx = LABEL_RADIUS * np.cos(langle)
    ly = LABEL_RADIUS * np.sin(langle)
    label_coords.append((lx, ly))

    # Thin connector from arrow tip to label if angle was moved
    # tip_x, tip_y = d["vx"] * 1.05, d["vy"] * 1.05
    # if abs(langle - d["angle"]) > 0.05:
    #     ax2.plot([tip_x, lx], [tip_y, ly],
    #              color=d["colour"], lw=0.5, alpha=0.6, zorder=4)

    label = PROPERTY_NAMES.get(d["var"], d["var"]).replace(" ", "\n")
    ax2.text(lx, ly, label,
             fontsize=7, ha="center", va="center",
             color=d["colour"],
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

# Axis limits: fit both arrows and labels
all_lx = [lx for lx, ly in label_coords]
all_ly = [ly for lx, ly in label_coords]
pad = 0.3
ax2.set_xlim(min(min(all_lx), -0.5) - pad, max(max(all_lx), 0.5) + pad)
ax2.set_ylim(min(min(all_ly), -0.5) - pad, max(max(all_ly), 0.5) + pad)
ax2.set_aspect("equal")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_title(f"{dataset_name.replace('_', ' ').capitalize()}: Top PCA loadings")

fig2.tight_layout()
fig2.savefig(output_folder / f"{dataset_name}_pca_loadings.pdf") # , dpi=300)
print(output_folder / f"{dataset_name}_pca_loadings.pdf")
plt.show()


# In[38]:


# ── Category-level aggregate vectors ─────────────────────────────────────────
# loadings: DataFrame (n_vars × 2), already scaled by √eigenvalue

cat_vectors = {}
for cat, colour in CATEGORY_COLOURS.items():
    # Variables in this category that also have loadings
    cat_vars = [c for c in loadings.index
                if c in meta.index and meta.loc[c, "Category"] == cat]
    if not cat_vars:
        continue

    cat_loads = loadings.loc[cat_vars]  # shape: (n_cat_vars, 2)

    # Weight each variable by its total loading magnitude (L2 norm across PC1+PC2)
    weights = np.sqrt(cat_loads["PC1"]**2 + cat_loads["PC2"]**2)
    weights = weights / weights.sum()  # normalise to sum=1

    # Weighted sum → one (PC1, PC2) vector per category
    vx = (cat_loads["PC1"] * weights).sum()
    vy = (cat_loads["PC2"] * weights).sum()
    cat_vectors[cat] = (vx, vy)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))

ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
ax.scatter([0], [0], color="black", s=20, zorder=6)

for cat, (vx, vy) in cat_vectors.items():
    colour = CATEGORY_COLOURS[cat]
    ax.quiver(
        0, 0, vx, vy,
        angles="xy", scale_units="xy", scale=1,
        color=colour, width=0.012,
        headwidth=4, headlength=5, headaxislength=4,
        zorder=5,
    )
    nudge = 1.15
    ax.text(vx * nudge, vy * nudge, cat,
            fontsize=8, ha="center", va="center",
            color=colour, fontweight="bold")

# Symmetric axis limits
max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) * 1.4
ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, max_val)
ax.set_aspect("equal")
ax.set_xlabel("PC1", fontsize=9)
ax.set_ylabel("PC2", fontsize=9)
ax.set_title("Category vectors in PCA space\n(weighted by loading magnitude)", fontsize=9)

# Remove frame, keep ticks
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(labelsize=7)

plt.tight_layout()

plt.savefig(output_folder / f"{dataset_name}_pca_optimizers.pdf") # , dpi=300)
print(output_folder / f"{dataset_name}_pca_optimizers.pdf")
plt.show()


# In[39]:


# Subplots of loadings for the first 3 principal components, ONLY top 10 contributors
loadings = pca.components_.T[:, :]
num_metrics = loadings.shape[0]
metric_names = huge_df_properties.columns


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6)) # , dpi=100)
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


# In[40]:


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


# In[ ]:





# In[45]:


"""
Three-panel PCA figure: young / developing / aging
Each panel:
  • scatter dots coloured by age
  • smoothing-spline trajectory through age-averaged PC scores (LineCollection)
  • inset "compass" in lower-right corner: category-aggregate loading vectors
Shared colorbar.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import make_smoothing_spline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
AGE_DATASETS  = ["lexis_data_young", "lexis_data_developing", "lexis_data_aging"]
PANEL_TITLES  = ["Young", "Developing", "Aging"]
N_PCA_COMPS   = 10
SPLINE_LAM    = 5        # smoothing-spline lambda (larger = smoother)
SPLINE_N_PTS  = 300      # interpolation resolution
CMAP          = plt.cm.terrain # plasma
DOT_SIZE      = 4
DOT_ALPHA     = 0.35
SPLINE_LW     = 3.5
AVG_DOT_SIZE  = 40
INSET_SIZE    = "32%"    # inset width/height as % of parent axes
output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/ages")
output_folder.mkdir(exist_ok=True)

# ── 1. Build combined dataframe ───────────────────────────────────────────────
huge_df = pd.DataFrame()
for dataset_name in AGE_DATASETS:
    df = dict_with_all_datasets[dataset_name].copy()
    df["dataset"] = dataset_name

    ages = np.load(
        f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
        f"data/preprocessed/{dataset_name}/04_further_info/00_ages.npy"
    )
    idx_mask = np.zeros(len(ages), dtype=bool)
    idx_mask[dict_with_all_datasets[dataset_name].index] = True
    df["age"] = ages[idx_mask]

    huge_df = pd.concat([huge_df, df], ignore_index=True)

# ── 2. Property columns only ──────────────────────────────────────────────────
prop_cols = [c for c in huge_df.columns
             if c not in ("dataset", "age") and c in precise_categories]

huge_df_props = huge_df[prop_cols].copy()
huge_df_props.replace([np.inf, -np.inf], np.nan, inplace=True)
huge_df_props.dropna(axis=1, inplace=True)

# ── 3. Single PCA fit on all data ─────────────────────────────────────────────
scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(huge_df_props)
pca        = PCA(n_components=N_PCA_COMPS)
pca_result = pca.fit_transform(X_scaled)

exp_var = pca.explained_variance_ratio_

# ── 4. Scaled loadings for category vectors ───────────────────────────────────
sqrt_eig = np.sqrt(pca.explained_variance_[:2])
loadings = pd.DataFrame(
    (pca.components_[:2] * sqrt_eig[:, None]).T,
    index=huge_df_props.columns,
    columns=["PC1", "PC2"],
)

# Category-aggregate vectors (weighted mean by loading magnitude)
cat_vectors = {}
for cat, colour in CATEGORY_COLOURS.items():
    cat_vars = [c for c in loadings.index
                if c in meta.index and meta.loc[c, "Category"] == cat]
    if not cat_vars:
        continue
    cat_loads = loadings.loc[cat_vars]
    weights   = np.sqrt(cat_loads["PC1"]**2 + cat_loads["PC2"]**2)
    if weights.sum() == 0:
        continue
    weights  /= weights.sum()
    vx = (cat_loads["PC1"] * weights).sum()
    vy = (cat_loads["PC2"] * weights).sum()
    cat_vectors[cat] = (vx, vy)

# ── 5. Global colour scale ────────────────────────────────────────────────────
age_all  = huge_df["age"].values
vmin     = np.nanpercentile(age_all, 2)
vmax     = np.nanpercentile(age_all, 98)
norm_global = Normalize(vmin=vmin, vmax=vmax)

# ── 6. Draw figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 5.6))

outer_gs = gridspec.GridSpec(
    1, 4,
    width_ratios=[1, 1, 1, 0.04],
    wspace=0.07,
    left=0.05, right=0.94,
    top=0.87, bottom=0.10,
)

axes    = [fig.add_subplot(outer_gs[0, i]) for i in range(3)]
cbar_ax = fig.add_subplot(outer_gs[0, 3])

# ── Per-panel ─────────────────────────────────────────────────────────────────
for ax, dataset_name, title in zip(axes, AGE_DATASETS, PANEL_TITLES):

    ds_mask     = (huge_df["dataset"] == dataset_name).values
    pc1_all     = pca_result[ds_mask, 0]
    pc2_all     = pca_result[ds_mask, 1]
    ages_panel  = huge_df["age"].values[ds_mask]

    # Per-age PCA norm (keeps spline colours local to each dataset's age range)
    norm_local = Normalize(ages_panel.min(), ages_panel.max())

    # # ── 6a. Scatter of individuals ────────────────────────────────────────────
    # ax.scatter(
    #     pc1_all, pc2_all,
    #     c=ages_panel, cmap=CMAP, norm=norm_global,
    #     s=DOT_SIZE, alpha=DOT_ALPHA,
    #     edgecolors="none", zorder=2,
    #     rasterized=True,
    # )

    # ── 6b. Age-averaged PC scores ────────────────────────────────────────────
    pca_panel_df = pd.DataFrame({
        "PC1": pc1_all,
        "PC2": pc2_all,
        "age": ages_panel,
    })
    age_means = (
        pca_panel_df.groupby("age")[["PC1", "PC2"]]
        .mean()
        .reset_index()
        .sort_values("age")
    )

    age_vals   = age_means["age"].values
    t_dense    = np.linspace(age_vals.min(), age_vals.max(), SPLINE_N_PTS)

    smooth_pc1 = make_smoothing_spline(age_vals, age_means["PC1"].values, lam=SPLINE_LAM)(t_dense)
    smooth_pc2 = make_smoothing_spline(age_vals, age_means["PC2"].values, lam=SPLINE_LAM)(t_dense)

    # Coloured LineCollection for the spline
    points   = np.array([smooth_pc1, smooth_pc2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=CMAP, 
        norm=norm_global, # norm_local,
        linewidth=SPLINE_LW, alpha=0.95, zorder=4,
    )
    lc.set_array(t_dense)
    ax.add_collection(lc)

    # Faint raw polyline underneath spline
    ax.plot(
        age_means["PC1"], age_means["PC2"],
        color="white", linewidth=0.5, alpha=0.4, zorder=3,
    )

    # Averaged dots on top
    ax.scatter(
        age_means["PC1"], age_means["PC2"],
        c=age_means["age"], cmap=CMAP, norm=norm_global,
        s=AVG_DOT_SIZE, zorder=5,
        edgecolors="white", linewidths=0.5,
    )

    ax.autoscale()

    # ── 6c. Axes decoration ───────────────────────────────────────────────────
    ax.axhline(0, color="grey", lw=0.4, alpha=0.35, zorder=1)
    ax.axvline(0, color="grey", lw=0.4, alpha=0.35, zorder=1)

    ax.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)", fontsize=9)
    else:
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ── 6d. Category-vector inset (lower-right corner) ────────────────────────
    ax_ins = inset_axes(
        ax,
        width=INSET_SIZE, height=INSET_SIZE,
        loc="lower right",
        borderpad=0.8,
    )

    ax_ins.axhline(0, color="grey", lw=0.4, alpha=0.5)
    ax_ins.axvline(0, color="grey", lw=0.4, alpha=0.5)
    ax_ins.scatter([0], [0], color="black", s=8, zorder=6)

    for cat, (vx, vy) in cat_vectors.items():
        colour = CATEGORY_COLOURS[cat]
        ax_ins.quiver(
            0, 0, vx, vy,
            angles="xy", scale_units="xy", scale=1,
            color=colour, width=0.018,
            headwidth=4, headlength=5, headaxislength=4,
            zorder=5,
        )
        nudge = 1.22
        ax_ins.text(
            vx * nudge, vy * nudge, cat,
            fontsize=5.5, ha="center", va="center",
            color=colour, fontweight="bold",
        )

    max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) * 1.5
    ax_ins.set_xlim(-max_val, max_val)
    ax_ins.set_ylim(-max_val, max_val)
    ax_ins.set_aspect("equal")
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    for spine in ax_ins.spines.values():
        spine.set_visible(False)

    # Light background so inset is legible over dots
    ax_ins.set_facecolor((1, 1, 1, 0.7))

# ── 7. Shared colorbar ────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=CMAP, norm=norm_global)
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Age (years)", fontsize=9)
cb.ax.tick_params(labelsize=7)

# ── 8. Legend for inset categories ───────────────────────────────────────────
cat_handles = [
    mpatches.Patch(color=CATEGORY_COLOURS[cat], label=cat)
    for cat in cat_vectors
    if cat in CATEGORY_COLOURS
]
fig.legend(
    handles=cat_handles,
    loc="upper center",
    ncol=len(cat_handles),
    fontsize=7.5,
    frameon=False,
    bbox_to_anchor=(0.47, 1.0),
    title="Category vectors (inset)",
    title_fontsize=8,
)

fig.suptitle(
    "PCA of brain network properties — age groups  "
    f"(spline through age-averaged scores,  λ={SPLINE_LAM})",
    fontsize=11, y=1.055,
)

out_path = output_folder / "pca_age_groups_spline_inset.pdf"
plt.savefig(out_path, bbox_inches="tight", dpi=300)
print(out_path)
plt.show()


# In[51]:


"""
Three-panel PCA figure: young / developing / aging
• One independent PCA per dataset
• Global colorbar 0–100 years
• Smoothing-spline trajectory through age-averaged PC scores
• Category-vector inset (lower-right corner of each panel)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import make_smoothing_spline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
AGE_DATASETS  = ["lexis_data_young", "lexis_data_developing", "lexis_data_aging"]
PANEL_TITLES  = ["Young", "Developing", "Aging"]
N_PCA_COMPS   = 10
SPLINE_LAM    = 5
SPLINE_N_PTS  = 300
CMAP          = plt.cm.plasma
DOT_SIZE      = 4
DOT_ALPHA     = 0.35
SPLINE_LW     = 3.5
AVG_DOT_SIZE  = 40
INSET_SIZE    = "32%"
norm_global   = Normalize(vmin=0, vmax=100)   # ← fixed 0–100 scale

output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/ages")
output_folder.mkdir(exist_ok=True)

# ── 1. Load ages into each dataset ────────────────────────────────────────────
for dataset_name in AGE_DATASETS:
    ages = np.load(
        f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
        f"data/preprocessed/{dataset_name}/04_further_info/00_ages.npy"
    )
    idx_mask = np.zeros(len(ages), dtype=bool)
    idx_mask[dict_with_all_datasets[dataset_name].index] = True
    dict_with_all_datasets[dataset_name]["age"] = ages[idx_mask]

# ── 2. Property columns (shared across datasets) ──────────────────────────────
# Use the intersection of precise_categories and columns present in all three
shared_prop_cols = None
for dataset_name in AGE_DATASETS:
    df = dict_with_all_datasets[dataset_name]
    cols = set(c for c in df.columns if c in precise_categories and c != "age")
    shared_prop_cols = cols if shared_prop_cols is None else shared_prop_cols & cols

shared_prop_cols = list(shared_prop_cols)

# ── 3. Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 5.6))

outer_gs = gridspec.GridSpec(
    1, 4,
    width_ratios=[1, 1, 1, 0.04],
    wspace=0.07,
    left=0.05, right=0.94,
    top=0.87, bottom=0.10,
)

axes    = [fig.add_subplot(outer_gs[0, i]) for i in range(3)]
cbar_ax = fig.add_subplot(outer_gs[0, 3])

# ── 4. Per-dataset PCA + plot ─────────────────────────────────────────────────
for ax, dataset_name, title in zip(axes, AGE_DATASETS, PANEL_TITLES):

    df = dict_with_all_datasets[dataset_name].copy()

    # --- clean & scale ---
    X = df[shared_prop_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(axis=1, inplace=True)
    usable_cols = X.columns.tolist()

    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X)
    pca        = PCA(n_components=N_PCA_COMPS)
    pca_result = pca.fit_transform(X_scaled)
    exp_var    = pca.explained_variance_ratio_

    ages_panel = df["age"].values
    pc1_all    = pca_result[:, 0]
    pc2_all    = pca_result[:, 1]

    # --- scatter of individuals ---
    # ax.scatter(
    #     pc1_all, pc2_all,
    #     c=ages_panel, cmap=CMAP, norm=norm_global,
    #     s=DOT_SIZE, alpha=DOT_ALPHA,
    #     edgecolors="none", zorder=2,
    #     rasterized=True,
    # )

    # --- age-averaged trajectory ---
    pca_df = pd.DataFrame({"PC1": pc1_all, "PC2": pc2_all, "age": ages_panel})
    age_means = (
        pca_df.groupby("age")[["PC1", "PC2"]]
        .mean()
        .reset_index()
        .sort_values("age")
    )

    age_vals   = age_means["age"].values
    t_dense    = np.linspace(age_vals.min(), age_vals.max(), SPLINE_N_PTS)
    smooth_pc1 = make_smoothing_spline(age_vals, age_means["PC1"].values, lam=SPLINE_LAM)(t_dense)
    smooth_pc2 = make_smoothing_spline(age_vals, age_means["PC2"].values, lam=SPLINE_LAM)(t_dense)

    points   = np.array([smooth_pc1, smooth_pc2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=CMAP, norm=norm_global,
        linewidth=SPLINE_LW, alpha=0.95, zorder=4,
    )
    lc.set_array(t_dense)
    ax.add_collection(lc)

    # faint raw polyline underneath spline
    ax.plot(
        age_means["PC1"], age_means["PC2"],
        color="white", linewidth=0.5, alpha=0.4, zorder=3,
    )

    # averaged dots on top
    ax.scatter(
        age_means["PC1"], age_means["PC2"],
        c=age_means["age"], cmap=CMAP, norm=norm_global,
        s=AVG_DOT_SIZE, zorder=5,
        edgecolors="white", linewidths=0.5,
    )

    ax.autoscale()

    # --- axes decoration ---
    ax.axhline(0, color="grey", lw=0.4, alpha=0.35, zorder=1)
    ax.axvline(0, color="grey", lw=0.4, alpha=0.35, zorder=1)
    ax.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)", fontsize=9)
    # if ax is axes[0]:
    ax.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)", fontsize=9)
    # else:
    #     ax.set_yticklabels([])
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ── Category-vector inset ─────────────────────────────────────────────────
    # Recompute loadings for this dataset's PCA
    sqrt_eig = np.sqrt(pca.explained_variance_[:2])
    loadings = pd.DataFrame(
        (pca.components_[:2] * sqrt_eig[:, None]).T,
        index=usable_cols,
        columns=["PC1", "PC2"],
    )

    cat_vectors = {}
    for cat, colour in CATEGORY_COLOURS.items():
        cat_vars = [c for c in loadings.index
                    if c in meta.index and meta.loc[c, "Category"] == cat]
        if not cat_vars:
            continue
        cat_loads = loadings.loc[cat_vars]
        weights   = np.sqrt(cat_loads["PC1"]**2 + cat_loads["PC2"]**2)
        if weights.sum() == 0:
            continue
        weights  /= weights.sum()
        vx = (cat_loads["PC1"] * weights).sum()
        vy = (cat_loads["PC2"] * weights).sum()
        cat_vectors[cat] = (vx, vy)

    ax_ins = inset_axes(
        ax,
        width=INSET_SIZE, height=INSET_SIZE,
        loc="lower right",
        borderpad=0.8,
    )

    ax_ins.axhline(0, color="grey", lw=0.4, alpha=0.5)
    ax_ins.axvline(0, color="grey", lw=0.4, alpha=0.5)
    ax_ins.scatter([0], [0], color="black", s=8, zorder=6)

    for cat, (vx, vy) in cat_vectors.items():
        colour = CATEGORY_COLOURS[cat]
        ax_ins.quiver(
            0, 0, vx, vy,
            angles="xy", scale_units="xy", scale=1,
            color=colour, width=0.018,
            headwidth=4, headlength=5, headaxislength=4,
            zorder=5,
        )
        ax_ins.text(
            vx * 1.22, vy * 1.22, cat,
            fontsize=5.5, ha="center", va="center",
            color=colour, fontweight="bold",
        )

    max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) * 1.5
    ax_ins.set_xlim(-max_val, max_val)
    ax_ins.set_ylim(-max_val, max_val)
    ax_ins.set_aspect("equal")
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    for spine in ax_ins.spines.values():
        spine.set_visible(False)
    ax_ins.set_facecolor((1, 1, 1, 0.7))

# ── 5. Shared colorbar ────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=CMAP, norm=norm_global)
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Age (years)", fontsize=9)
cb.set_ticks([0, 20, 40, 60, 80, 100])
cb.ax.tick_params(labelsize=7)

# ── 6. Category legend ────────────────────────────────────────────────────────
# cat_handles = [
#     mpatches.Patch(color=CATEGORY_COLOURS[cat], label=cat)
#     for cat in CATEGORY_COLOURS
#     if any(
#         c in meta.index and meta.loc[c, "Category"] == cat
#         for c in shared_prop_cols
#     )
# ]
# fig.legend(
#     handles=cat_handles,
#     loc="upper center",
#     ncol=len(cat_handles),
#     fontsize=7.5,
#     frameon=False,
#     bbox_to_anchor=(0.47, 1.0),
#     title="Category vectors (inset)",
#     title_fontsize=8,
# )

fig.suptitle(
    "PCA of brain network properties — independent PCA per age group  "
    f"(spline λ={SPLINE_LAM})", # ,  colour scale 0–100 years)",
    fontsize=11, # y=1.055,
)

out_path = output_folder / "pca_age_groups_per_dataset.pdf"
plt.savefig(out_path, bbox_inches="tight", dpi=300)
print(out_path)
plt.show()


# In[60]:


"""
Three-panel PCA figure: young / developing / aging
• One independent PCA per dataset
• Global colorbar 0–100 years
• Smoothing-spline trajectory through age-averaged PC scores
• Category-vector inset (lower-right corner of each panel)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import make_smoothing_spline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
AGE_DATASETS  = ["lexis_data_developing", "lexis_data_young", "lexis_data_aging"]
PANEL_TITLES  = ["Developing", "Young", "Aging"]
N_PCA_COMPS   = 10
SPLINE_LAM    = 5
SPLINE_N_PTS  = 300
CMAP          = plt.cm.nipy_spectral # plasma
DOT_SIZE      = 4
DOT_ALPHA     = 0.35
SPLINE_LW     = 3.5
AVG_DOT_SIZE  = 40
INSET_SIZE    = "32%"
norm_global   = Normalize(vmin=0, vmax=100)   # ← fixed 0–100 scale

output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/ages")
output_folder.mkdir(exist_ok=True)

# ── 1. Load ages into each dataset ────────────────────────────────────────────
for dataset_name in AGE_DATASETS:
    ages = np.load(
        f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
        f"data/preprocessed/{dataset_name}/04_further_info/00_ages.npy"
    )
    idx_mask = np.zeros(len(ages), dtype=bool)
    idx_mask[dict_with_all_datasets[dataset_name].index] = True
    dict_with_all_datasets[dataset_name]["age"] = ages[idx_mask]

# ── 2. Property columns (shared across datasets) ──────────────────────────────
# Use the intersection of precise_categories and columns present in all three
shared_prop_cols = None
for dataset_name in AGE_DATASETS:
    df = dict_with_all_datasets[dataset_name]
    cols = set(c for c in df.columns if c in precise_categories and c != "age")
    shared_prop_cols = cols if shared_prop_cols is None else shared_prop_cols & cols

shared_prop_cols = list(shared_prop_cols)

# ── 3. Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 5.6))

outer_gs = gridspec.GridSpec(
    1, 4,
    width_ratios=[1, 1, 1, 0.04],
    wspace=0.07,
    left=0.05, right=0.94,
    top=0.87, bottom=0.10,
)

axes    = [fig.add_subplot(outer_gs[0, i]) for i in range(3)]
cbar_ax = fig.add_subplot(outer_gs[0, 3])

# ── 4. Per-dataset PCA + plot ─────────────────────────────────────────────────
for ax, dataset_name, title in zip(axes, AGE_DATASETS, PANEL_TITLES):

    df = dict_with_all_datasets[dataset_name].copy()

    # --- clean & scale ---
    X = df[shared_prop_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(axis=1, inplace=True)
    usable_cols = X.columns.tolist()

    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X)
    pca        = PCA(n_components=N_PCA_COMPS)
    pca_result = pca.fit_transform(X_scaled)
    exp_var    = pca.explained_variance_ratio_

    ages_panel = df["age"].values
    pc1_all    = pca_result[:, 0]
    pc2_all    = pca_result[:, 1]

    # --- scatter of individuals ---
    # ax.scatter(
    #     pc1_all, pc2_all,
    #     c=ages_panel, cmap=CMAP, norm=norm_global,
    #     s=DOT_SIZE, alpha=DOT_ALPHA,
    #     edgecolors="none", zorder=2,
    #     rasterized=True,
    # )

    # --- age-averaged trajectory ---
    pca_df = pd.DataFrame({"PC1": pc1_all, "PC2": pc2_all, "age": ages_panel})
    age_means = (
        pca_df.groupby("age")[["PC1", "PC2"]]
        .mean()
        .reset_index()
        .sort_values("age")
    )

    age_vals   = age_means["age"].values
    t_dense    = np.linspace(age_vals.min(), age_vals.max(), SPLINE_N_PTS)
    smooth_pc1 = make_smoothing_spline(age_vals, age_means["PC1"].values, lam=SPLINE_LAM)(t_dense)
    smooth_pc2 = make_smoothing_spline(age_vals, age_means["PC2"].values, lam=SPLINE_LAM)(t_dense)

    points   = np.array([smooth_pc1, smooth_pc2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=CMAP, norm=norm_global,
        linewidth=SPLINE_LW, alpha=0.95, zorder=4,
    )
    lc.set_array(t_dense)
    ax.add_collection(lc)

    # faint raw polyline underneath spline
    ax.plot(
        age_means["PC1"], age_means["PC2"],
        color="white", linewidth=0.5, alpha=0.4, zorder=3,
    )

    # averaged dots on top
    ax.scatter(
        age_means["PC1"], age_means["PC2"],
        c=age_means["age"], cmap=CMAP, norm=norm_global,
        s=AVG_DOT_SIZE, zorder=5,
        edgecolors="white", linewidths=0.5,
    )
    
    # Add an age label for the oldest and youngest point each
    ax.text(
        age_means["PC1"].iloc[0], age_means["PC2"].iloc[0],
        f"{age_means['age'].iloc[0]:.0f} yrs",
        fontsize=6, ha="right", va="bottom", color="black",
        zorder=6,
    )
    ax.text(
        age_means["PC1"].iloc[-1], age_means["PC2"].iloc[-1],
        f"{age_means['age'].iloc[-1]:.0f} yrs",
        fontsize=6, ha="left", va="top", color="black",
        zorder=6,
    )

    ax.autoscale()

    # --- axes decoration ---
    ax.axhline(0, color="grey", lw=0.4, alpha=0.35, zorder=1)
    ax.axvline(0, color="grey", lw=0.4, alpha=0.35, zorder=1)
    ax.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)", fontsize=9)
    # if ax is axes[0]:
    ax.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)", fontsize=9)
    # else:
    #     ax.set_yticklabels([])
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ── Category-vector inset ─────────────────────────────────────────────────
    # Recompute loadings for this dataset's PCA
    sqrt_eig = np.sqrt(pca.explained_variance_[:2])
    loadings = pd.DataFrame(
        (pca.components_[:2] * sqrt_eig[:, None]).T,
        index=usable_cols,
        columns=["PC1", "PC2"],
    )

    cat_vectors = {}
    for cat, colour in CATEGORY_COLOURS.items():
        cat_vars = [c for c in loadings.index
                    if c in meta.index and meta.loc[c, "Category"] == cat]
        if not cat_vars:
            continue
        cat_loads = loadings.loc[cat_vars]
        weights   = np.sqrt(cat_loads["PC1"]**2 + cat_loads["PC2"]**2)
        if weights.sum() == 0:
            continue
        weights  /= weights.sum()
        vx = (cat_loads["PC1"] * weights).sum()
        vy = (cat_loads["PC2"] * weights).sum()
        cat_vectors[cat] = (vx, vy)

    ax_ins = inset_axes(
        ax,
        width=INSET_SIZE, height=INSET_SIZE,
        loc="lower right",
        borderpad=0.8,
    )

    ax_ins.axhline(0, color="grey", lw=0.4, alpha=0.5)
    ax_ins.axvline(0, color="grey", lw=0.4, alpha=0.5)
    ax_ins.scatter([0], [0], color="black", s=8, zorder=6)

    for cat, (vx, vy) in cat_vectors.items():
        colour = CATEGORY_COLOURS[cat]
        ax_ins.quiver(
            0, 0, vx, vy,
            angles="xy", scale_units="xy", scale=1,
            color=colour, width=0.018,
            headwidth=4, headlength=5, headaxislength=4,
            zorder=5,
        )
        ax_ins.text(
            vx * 1.22, vy * 1.22, cat,
            fontsize=5.5, ha="center", va="center",
            color=colour, fontweight="bold",
        )

    max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) * 1.5
    ax_ins.set_xlim(-max_val, max_val)
    ax_ins.set_ylim(-max_val, max_val)
    ax_ins.set_aspect("equal")
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    for spine in ax_ins.spines.values():
        spine.set_visible(False)
    ax_ins.set_facecolor((1, 1, 1, 0.7))
    
    

# ── 5. Shared colorbar ────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=CMAP, norm=norm_global)
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Age (years)") # , fontsize=9)
cb.set_ticks([0, 20, 40, 60, 80, 100])
cb.ax.tick_params(labelsize=7)



fig.suptitle(
    "PCA of brain network properties — independent PCA per age group  "
    f"(spline λ={SPLINE_LAM})", # ,  colour scale 0–100 years)",
    fontsize=11, # y=1.055,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = output_folder / "pca_age_groups_per_dataset.pdf"
plt.savefig(out_path) # , bbox_inches="tight", dpi=300)
print(out_path)
plt.show()


# # OLD

# In[41]:


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

