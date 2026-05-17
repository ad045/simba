#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from vizman import viz
import pickle


from config import COLORS, COLOR_SCHEME, LABEL_MAP, gray_cmap, bone_white, half_black, emp_dataset_and_experiment_pairs, PROPERTY_NAMES


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Load data 
output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/trade_off_analysis")
with open(output_folder / "all_datasets_filtered.pkl", "rb") as f:
    dict_with_all_datasets = pickle.load(f)
    
# Output of this script: pickle.dump(pca_df, open(output_folder / "pca_input_df.pkl", "wb"))


# In[3]:


dict_with_all_datasets.keys()


# In[4]:


dict_with_all_datasets["suarez_MaMI_dataset"]


# In[ ]:





# In[8]:


huge_df = pd.DataFrame()

for dataset_name, df in dict_with_all_datasets.items():
    print(len(df.keys()), df.keys().to_list()[:5])
    # if dataset_name == "kaysons_generated_networks_topology":
    #     print(" ---> skipped")
    #     continue
    
    # df = df.copy()  # avoid modifying the original
    # df["dataset"] = dataset_name
    
    # print(dataset_name, len(df["eta"]))
    # huge_df = pd.concat([huge_df, df], ignore_index=True)


# In[ ]:


dict_with_all_datasets["hcp_schaefer_100_dataset"] 


# In[11]:


for dataset_name, df in dict_with_all_datasets.items():
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        print(f"{dataset_name}: duplicate columns -> {dupes}")


# In[15]:


huge_df = pd.DataFrame()

for dataset_name, df in dict_with_all_datasets.items():
    if dataset_name == "kaysons_generated_networks_topology":
        print(" ---> skipped")
        continue
    
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]  # drop duplicate columns
    df["dataset"] = dataset_name
    print(dataset_name, len(df), "| columns:", df.shape[1])
    huge_df = pd.concat([huge_df, df], ignore_index=True)

print("Final shape:", huge_df.shape)


# In[16]:


# my_favs = ["global_efficiency", "modularity", "proportion_long_range_connections_0.5", "synchronizability_eigenratio_lambda_2", "synchronizability_eigenratio_eigenratio", "computational_capacity_total_capacity", "kuramoto_averaged_synchronization_r_std"]
# huge_df = huge_df[my_favs+["dataset"]]


# In[17]:


# for i in huge_df.keys(): 
#     print(i)


# In[18]:


huge_df.keys()


# In[19]:


correlation_matrix = huge_df.corr() if "dataset" not in huge_df.keys() else huge_df.drop(columns=["dataset"]).corr()
correlation_matrix


# In[20]:


# Make a correlation axis between the columns of the huge_df. Don't use the dataset column for this, only the metric columns.
correlation_matrix = huge_df.drop(columns=["dataset"]).corr()
plt.figure(figsize=(12,12))
sns.heatmap(correlation_matrix, annot=False, # fmt=".2f", 
            cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix of Metrics (Not everything is labelled... - {correlation_matrix.shape[0]} metrics)")
plt.tight_layout()
plt.show()
# Not everything is labelled... 


# In[21]:


# Get correlations with wiring_cost, sorted by absolute value
wiring_correlations = correlation_matrix["wiring_cost"].abs().sort_values(ascending=False)

# Display top correlations (excluding wiring_cost itself)
print("Top correlations with wiring_cost:")
print(wiring_correlations[1:11])  # Skip first (itself), show top 10

# Or if you want to see positive and negative correlations separately
wiring_correlations_signed = correlation_matrix["wiring_cost"].sort_values(ascending=False)
print("\nAll correlations with wiring_cost (sorted):")
print(wiring_correlations_signed[1:])  # Skip wiring_cost itself


# In[22]:


# Bar plot of top correlations
top_n = 15
top_corrs = wiring_correlations[1:top_n+1]  # Skip wiring_cost itself

plt.figure(figsize=(10, 6))
top_corrs.plot(kind='barh')
plt.xlabel('Absolute Correlation with wiring_cost')
plt.title(f'Top {top_n} Features Correlated with wiring_cost')
plt.tight_layout()
plt.show()


# In[23]:


huge_df


# In[24]:


huge_df_metrics = huge_df.drop(columns=["dataset", "id"])
huge_df_metrics


# In[25]:


huge_df_metrics.isnull().any()


# In[26]:


huge_df_metrics.columns[huge_df_metrics.isnull().any()].tolist()


# In[27]:


huge_df_metrics = huge_df.drop(columns=["dataset", "id"])
huge_df_metrics


# In[28]:


dropped_cols = huge_df_metrics.columns[huge_df_metrics.isnull().any()].tolist()
dropped_cols


# In[29]:


# Create a PCA between the columns of the huge_df
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For pca always drop the dataset column and only use the metric columns
huge_df_metrics = huge_df.drop(columns=["dataset", "id"]) if "dataset" in huge_df.keys() else huge_df.drop(columns=["id"])
print(huge_df_metrics.keys())

# print all max values of all columns in huge_df_metrics
print("\nMax values of metrics:")
for col in huge_df_metrics.columns:
    print(f"{col}: {huge_df_metrics[col].max():.4f}")
print("\nMin values of metrics:")
for col in huge_df_metrics.columns:
    print(f"{col}: {huge_df_metrics[col].min():.4f}")
    
# Drop the columns that have inf or -inf values (if any). Print the dropped columns
dropped_cols = huge_df_metrics.columns[huge_df_metrics.isnull().any()].tolist()
huge_df_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)
huge_df_metrics.dropna(axis=1, inplace=True)
print(f"Dropped columns: {dropped_cols}")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(huge_df_metrics)

pca = PCA(n_components=3) # 10)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=huge_df['dataset'])
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
plt.title("Scree Plot")
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.tight_layout()
plt.show()


# In[30]:


# Subplots of loadings for the first 3 principal components
loadings = pca.components_.T
num_metrics = loadings.shape[0]
metric_names = huge_df_metrics.columns

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



# In[31]:


from config import PROPERTY_NAMES

# --- Config ---
THRESHOLD = 0.95  # only flag pairs above this

# Start fresh each run from the full metric set
remaining_cols = huge_df_metrics.columns.tolist()
print(len(remaining_cols), "columns before removals")
# remove all columns that are not in PROPERTY_NAMES.keys(): 
remaining_cols = [c for c in remaining_cols if c in PROPERTY_NAMES.keys()]
print(len(remaining_cols), "columns after removing those not in PROPERTY_NAMES")

to_remove = [
            "participation_coefficient_pc_median_further", "participation_coefficient_pc_std_further", 
             "kernel_rank_max", 
             "spectral_gap", # 100 % identical to fatemeh
             "participation_coefficient_wmd_mean_further", "participation_coefficient_wmd_std_further",
             "participation_coefficient_n_communities_further", 
             "persistent_homology_ph_total_persistence", # identical to "h1_n_features"
             'participation_coefficient_pc_mean_further', 
             "synchronizability_eigenratio_lambda_2", # same as fiedler value
             "participation_coefficient_pc_frac_connector_further", 
             "algebraic_connectivity_fiedler_value", 
             "algebraic_connectivity_fiedler_value_norm", 
             "computational_capacity_cubic_capacity_total", 
             "avg_edge_distance", 
             "structural_complexity", # too high correlation with "effective_dimensionality"
             "topological_distance_std",
             "transitivity", # too high correlation with "avg_clustering"
             "computational_capacity_cross_capacity_total", 
             'omega', # too high correlation with "omega"
             'computational_capacity_state_entropy', 
             "participation_coefficient_pc_median", 
             "proportion_long_range_connections_0.3", 'proportion_long_range_connections_0.5', 'proportion_long_range_connections_0.1', 
             "targeted_attack_robustness_rob_random_half", 
             'participation_coefficient_pc_mean', 
             'char_path_length', 
             'nct_control_std', 
            'computational_capacity_total_capacity', 
            'participation_coefficient_pc_frac_connector', 
            'propagation_efficiency', 
            'spectral_radius', # avg_commmunicability 
            'persistent_homology_ph_h1_entropy', 
            'computational_capacity_memory_timescale', 
            'richclub_n_edges', # too high correlation with avg_communicability
            'community_synchronization_vulnerability_value', 
             
             "effective_dimensionality", # !!! (propagation_efficiency)
             
             "ollivier_ricci_curvature_orc_mean", 
             "targeted_attack_robustness_rob_targeted_half", 
             ] + [f"mc_{i}" for i in range(1,15)] + [f"mc_{i}" for i in range(16,50)]

if to_remove is not None:
    remaining_cols = [c for c in remaining_cols if c not in to_remove]


# ── Run this cell repeatedly ──────────────────────────────────────────────────
working_df = huge_df_metrics[remaining_cols].copy()
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


# In[32]:


# # ── STEP 1: Find the 3 extreme corner points ──────────────────────────────────

# corners = {
#     "top_left":    np.argmax(pca_result[:, 1]),                              # max PC2
#     "bottom_left": np.argmax(-pca_result[:, 0] - pca_result[:, 1]),         # min PC1 + min PC2
#     "right":       np.argmax(pca_result[:, 0]),                              # max PC1
# }

# print("Corner points:")
# for name, idx in corners.items():
#     print(f"  {name}: index={idx}, PC1={pca_result[idx,0]:.2f}, PC2={pca_result[idx,1]:.2f}, "
#           f"dataset={pca_df_with_dataset_label['dataset'].iloc[idx]}")

# # Verify visually
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=pca_df_with_dataset_label['dataset'], alpha=0.5)
# for name, idx in corners.items():
#     plt.scatter(pca_result[idx, 0], pca_result[idx, 1], s=300, zorder=5, edgecolors='black', linewidths=2)
#     plt.annotate(name, (pca_result[idx, 0], pca_result[idx, 1]), textcoords="offset points", xytext=(8, 8), fontsize=10)
# plt.title("PCA of Metrics — Selected Corner Points")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.tight_layout()
# plt.show()

# # ── STEP 2: Get the top 6 features by loading magnitude (PC1 + PC2) ───────────

# feature_names = pca_df.columns  # from your pipeline: pca_df = pca_df_with_dataset_label.drop(columns="dataset")
# loading_magnitude = np.sqrt(pca.components_[0]**2 + pca.components_[1]**2)
# top6_idx = np.argsort(loading_magnitude)[::-1][:6]
# top6_features = feature_names[top6_idx]

# print("\nTop 6 features by PC1+PC2 loading magnitude:")
# for feat, mag in zip(top6_features, loading_magnitude[top6_idx]):
#     print(f"  {feat}: {mag:.4f}")

# # ── STEP 3: Extract standardized values for each corner ───────────────────────

# corner_data = {
#     name: scaled_data[idx, top6_idx]
#     for name, idx in corners.items()
# }

# # ── STEP 4: Spider plot ────────────────────────────────────────────────────────

# labels = [f.replace("_", "\n") for f in top6_features]  # wrap long names
# n = len(labels)
# angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
# angles += angles[:1]

# colors = {"top_left": "steelblue", "bottom_left": "orange", "right": "brown"}

# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# for name, values in corner_data.items():
#     vals = values.tolist() + [values[0]]
#     ax.plot(angles, vals, label=name, color=colors[name], linewidth=2)
#     ax.fill(angles, vals, alpha=0.15, color=colors[name])

# ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
# ax.set_title("Top 6 Features at PCA Extremes\n(z-scored values)", pad=20)
# ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
# plt.tight_layout()
# plt.show()


# In[33]:


reduced_corr = huge_df[remaining_cols].corr()

plt.figure(figsize=(18,18))
sns.heatmap(reduced_corr, annot=False, # fmt=".2f",
            cmap="coolwarm", cbar=True, 
            xticklabels=True, yticklabels=True)

# show all x and y ticks. Name them PROPERTY_NAMES if they are in PROPERTY_NAMES, otherwise use the original column name. Rotate x labels by 90 degrees and y labels by 0 degrees.
new_labels = []
for col in reduced_corr.columns:
    if col in PROPERTY_NAMES.keys():
        new_labels.append(PROPERTY_NAMES[col])
    else:
        new_labels.append(col)
plt.xticks(ticks=np.arange(len(new_labels))+0.5, labels=new_labels, rotation=90)
plt.yticks(ticks=np.arange(len(new_labels))+0.5, labels=new_labels, rotation=0)

plt.title(f"Correlation Matrix - {reduced_corr.shape[0]} metrics)")
plt.tight_layout()
plt.show()
# Not everything is labelled... 


# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ── 1. Load category / section metadata from Excel ────────────────────────────
excel_path = "/Users/adrian/Desktop/network_properties_full_excel.xlsx" # network_properties_full_excel.xlsx"   # ← adjust path if needed
meta_raw = pd.read_excel(excel_path, sheet_name=0)

# Forward-fill the section headers (rows where 'Variable Name' is NaN)
meta_raw["section"] = meta_raw.apply(
    lambda r: r["#"] if pd.isna(r["Variable Name"]) and pd.notna(r["#"]) else None,
    axis=1,
).ffill()

# Keep only actual variable rows
meta = (
    meta_raw[meta_raw["Variable Name"].notna()]
    [["Variable Name", "Category", "section"]]
    .rename(columns={"Variable Name": "var"})
    .set_index("var")
)

# Preferred display order for broad categories
CATEGORY_ORDER = [
    "Integration", "Segregation", "Robustness",
    "Topology", "Wiring", "Geometry",
    "Dynamics", "Computation",
]

# Colour palette — one colour per broad category
CATEGORY_COLOURS = {
    "Integration":  "#4C72B0",
    "Segregation":  "#DD8452",
    "Robustness":   "#55A868",
    "Topology":     "#C44E52",
    "Wiring":       "#8172B2",
    "Geometry":     "#937860",
    "Dynamics":     "#DA8BC3",
    "Computation":  "#8C8C8C",
}

# ── 2. Filter to only the variables present in your reduced_corr ──────────────
# (replace `remaining_cols` and `huge_df` with your actual variable names / df)
cols_in_data = [c for c in remaining_cols if c in meta.index]

# Assign a sort key: primary = category order index, secondary = section string
def sort_key(col):
    row = meta.loc[col]
    cat = row["Category"] if pd.notna(row["Category"]) else "ZZZ"
    cat_idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (cat_idx, str(row["section"]))

cols_sorted_by_cat = sorted(cols_in_data, key=sort_key)

# ── 3. Within-category hierarchical clustering ────────────────────────────────
corr_full = huge_df[cols_in_data].corr()

final_order = []
for cat in CATEGORY_ORDER:
    cat_cols = [c for c in cols_sorted_by_cat if meta.loc[c, "Category"] == cat]
    if len(cat_cols) == 0:
        continue
    if len(cat_cols) == 1:
        final_order.extend(cat_cols)
        continue

    sub_corr = corr_full.loc[cat_cols, cat_cols].fillna(0)
    # Convert correlation to distance; clip to avoid numerical issues
    dist = np.clip(1 - sub_corr.values, 0, 2)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    final_order.extend([cat_cols[i] for i in order])

# Variables not assigned a known category — append at end
uncategorised = [c for c in cols_in_data if c not in final_order]
final_order.extend(uncategorised)

# ── 4. Reorder correlation matrix ─────────────────────────────────────────────
reduced_corr = corr_full.loc[final_order, final_order]

# ── 5. Build display labels (use PROPERTY_NAMES where available) ──────────────
new_labels = [PROPERTY_NAMES.get(c, c) for c in final_order]

# ── 6. Build category colour arrays for the side-bar ─────────────────────────
bar_colours = []
for c in final_order:
    cat = meta.loc[c, "Category"] if c in meta.index else None
    bar_colours.append(CATEGORY_COLOURS.get(cat, "#CCCCCC"))

# ── 7. Plot ───────────────────────────────────────────────────────────────────
n = len(final_order)
fig_size = max(16, n * 0.22)
fig = plt.figure(figsize=(fig_size + 1.5, fig_size))

# GridSpec: [colour-bar column | heatmap column]
gs = gridspec.GridSpec(
    1, 2,
    width_ratios=[0.018, 1],
    wspace=0.01,
    left=0.12, right=0.98,
    top=0.97, bottom=0.12,
)

ax_bar  = fig.add_subplot(gs[0])
ax_heat = fig.add_subplot(gs[1])

# — Colour bar (left side) —
for i, colour in enumerate(bar_colours):
    ax_bar.add_patch(mpatches.Rectangle((0, n - i - 1), 1, 1, color=colour))
ax_bar.set_xlim(0, 1)
ax_bar.set_ylim(0, n)
ax_bar.axis("off")

# Annotate category bracket labels on the colour bar
cat_positions = {}
for i, c in enumerate(final_order):
    cat = meta.loc[c, "Category"] if c in meta.index else "Other"
    cat_positions.setdefault(cat, []).append(n - i - 1)

for cat, positions in cat_positions.items():
    mid = (min(positions) + max(positions)) / 2 + 0.5
    ax_bar.text(
        -0.3, mid, cat,
        ha="right", va="center",
        fontsize=7.5, fontweight="bold",
        color=CATEGORY_COLOURS.get(cat, "#444444"),
        rotation=0,
    )

# — Heatmap —
im = ax_heat.imshow(
    reduced_corr.values,
    cmap="coolwarm", vmin=-1, vmax=1,
    aspect="auto", interpolation="none",
)

# Draw thin white lines between categories to visually separate them
boundaries = [0]
prev_cat = meta.loc[final_order[0], "Category"] if final_order[0] in meta.index else None
for i, c in enumerate(final_order[1:], start=1):
    curr_cat = meta.loc[c, "Category"] if c in meta.index else None
    if curr_cat != prev_cat:
        boundaries.append(i)
        prev_cat = curr_cat
boundaries.append(n)

for b in boundaries[1:-1]:
    ax_heat.axhline(b - 0.5, color="white", linewidth=1.2, alpha=0.85)
    ax_heat.axvline(b - 0.5, color="white", linewidth=1.2, alpha=0.85)

# Tick labels
ax_heat.set_xticks(np.arange(n))
ax_heat.set_xticklabels(new_labels, rotation=90, # fontsize=6.5, 
                        ha="right")
ax_heat.set_yticks(np.arange(n))
ax_heat.set_yticklabels(new_labels) # , fontsize=6.5)
ax_heat.tick_params(axis="both", which="both", length=0)

# Colour-bar legend
cbar = fig.colorbar(im, ax=ax_heat, shrink=0.5, pad=0.01)
cbar.set_label("Pearson r", fontsize=9)

# Category legend
legend_handles = [
    mpatches.Patch(color=CATEGORY_COLOURS[cat], label=cat)
    for cat in CATEGORY_ORDER
    if cat in set(meta.loc[final_order, "Category"].dropna())
]
ax_heat.legend(
    handles=legend_handles,
    loc="upper right", bbox_to_anchor=(1.18, 1.0),
    fontsize=7.5, title="Category", title_fontsize=8,
    framealpha=0.9,
)

ax_heat.set_title(
    f"Correlation Matrix — {n} metrics  (ordered by category, clustered within)",
    fontsize=11, pad=8,
)

plt.savefig("corr_matrix_categorised.pdf", dpi=150, bbox_inches="tight")
plt.savefig("corr_matrix_categorised.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: corr_matrix_categorised.pdf / .png")


# In[35]:


pca_df_with_dataset_label = huge_df[remaining_cols+["dataset"]]
pickle.dump(pca_df_with_dataset_label, open(output_folder / "pca_input_df.pkl", "wb"))


# In[36]:


# # For pca always drop the dataset column and only use the metric columns
# pca_df_with_dataset_label = huge_df[remaining_cols+["dataset"]]
# pca_df_with_dataset_label.drop(columns="dataset") 
# # Create a PCA between the columns of the huge_df
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # For pca always drop the dataset column and only use the metric columns
# pca_df_with_dataset_label = huge_df[kept+["dataset"]]
# # print(pca_df.keys())
# # pca_df.head()
# pickle.dump(pca_df_with_dataset_label, open(output_folder / "pca_input_df.pkl", "wb"))
# pca_df = pca_df_with_dataset_label.drop(columns="dataset")
# # print all max values of all columns in pca_df
# print("\nMax values of metrics:")
# for col in pca_df.columns:
#     print(f"{col}: {pca_df[col].max():.4f}")
# print("\nMin values of metrics:")
# for col in pca_df.columns:
#     print(f"{col}: {pca_df[col].min():.4f}")

# # Drop the columns that have inf or -inf values (if any). Print the dropped columns
# dropped_cols = pca_df.columns[pca_df.isnull().any()].tolist()
# pca_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# pca_df.dropna(axis=1, inplace=True)
# print(f"Dropped columns: {dropped_cols}")

# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(pca_df)

# pca = PCA(n_components=10, random_state=8)
# pca_result = pca.fit_transform(scaled_data)

# plt.figure(figsize=(8,6))
# sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=pca_df_with_dataset_label['dataset'])
# plt.title("PCA of Metrics")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Scree plot to show explained variance
# explained_variance = pca.explained_variance_ratio_
# plt.figure(figsize=(6,4))
# sns.barplot(x=[f"PC{i+1}" for i in range(len(explained_variance))], y=explained_variance)
# plt.title("Scree Plot")
# plt.ylabel("Explained Variance Ratio")
# plt.xlabel("Principal Components")
# plt.tight_layout()
# plt.show()
# # Subplots of loadings for the first 3 principal components
# loadings = pca.components_.T
# num_metrics = loadings.shape[0]
# metric_names = huge_df_metrics.columns

# # order them all by the absolute value of their loading on the first principal component
# sorted_indices = np.argsort(np.abs(loadings[:, 0])) #[::-1]
# loadings = loadings[sorted_indices]
# metric_names = metric_names[sorted_indices]


# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,20), dpi=100)
# for i in range(3):
#     axs[i].barh(range(num_metrics), loadings[:, i])
#     axs[i].set_yticks(range(num_metrics))
#     axs[i].set_yticklabels(metric_names, rotation=0)
#     axs[i].set_title(f"Loadings for PC{i+1}")
# plt.tight_layout()
# plt.show()


# pca_result.shape
# # After fitting PCA and getting pca_scores (shape: n_samples x n_components)

# corners = {
#     "top_left":    pca_result[:, 1].argmax(),                        # max PC2
#     "bottom_left": (-pca_result[:, 0] - pca_result[:, 1]).argmax(), # min PC1 + min PC2
#     "right":       pca_result[:, 0].argmax(),                        # max PC1
# }
# huge_df_metrics
# # pca.components_ shape: (n_components, n_features)
# # Use PC1 and PC2 loadings combined
# loading_magnitude = np.sqrt(pca.components_[0]**2 + pca.components_[1]**2)
# top6_idx = np.argsort(loading_magnitude)[::-1][:6]
# top6_features = [feature_names[i] for i in top6_idx]


# # OLD
# 

# In[37]:


class MetricSelector:
    """Interactive tool for selecting metrics while avoiding high correlations."""
    
    def __init__(self, correlation_matrix, threshold=0.95):
        """
        Initialize the metric selector.
        
        Parameters:
        -----------
        correlation_matrix : pd.DataFrame
            Correlation matrix of all metrics
        threshold : float
            Correlation threshold above which metrics are considered too similar
        """
        self.corr_matrix = correlation_matrix
        self.threshold = threshold
        self.selected_metrics = []
        self.excluded_metrics = []
        
    def find_high_correlations(self):
        """Find all pairs of metrics with correlation above threshold."""
        high_corr_pairs = []
        
        # Get upper triangle of correlation matrix (avoid duplicates)
        for i in range(len(self.corr_matrix.columns)):
            for j in range(i+1, len(self.corr_matrix.columns)):
                metric1 = self.corr_matrix.columns[i]
                metric2 = self.corr_matrix.columns[j]
                corr_value = abs(self.corr_matrix.iloc[i, j])
                
                if corr_value > self.threshold:
                    high_corr_pairs.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': corr_value
                    })
        
        return pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
    
    def get_conflict_groups(self):
        """Group metrics that are highly correlated with each other."""
        high_corr_df = self.find_high_correlations()
        
        if len(high_corr_df) == 0:
            return []
        
        # Build groups of correlated metrics
        groups = []
        processed = set()
        
        for _, row in high_corr_df.iterrows():
            m1, m2 = row['metric1'], row['metric2']
            
            # Find if either metric is already in a group
            found_group = None
            for group in groups:
                if m1 in group or m2 in group:
                    found_group = group
                    break
            
            if found_group:
                found_group.add(m1)
                found_group.add(m2)
            else:
                groups.append({m1, m2})
        
        return [sorted(list(g)) for g in groups]
    
    def analyze_metric_conflicts(self, metric):
        """Show which metrics conflict with a given metric."""
        conflicts = []
        
        for other_metric in self.corr_matrix.columns:
            if metric != other_metric:
                corr_value = abs(self.corr_matrix.loc[metric, other_metric])
                if corr_value > self.threshold:
                    conflicts.append({
                        'conflicting_metric': other_metric,
                        'correlation': corr_value
                    })
        
        return pd.DataFrame(conflicts).sort_values('correlation', ascending=False) if conflicts else pd.DataFrame()
    
    def suggest_selection_order(self):
        """Suggest an order to review metrics based on number of conflicts."""
        conflict_counts = {}
        
        for metric in self.corr_matrix.columns:
            conflicts = self.analyze_metric_conflicts(metric)
            conflict_counts[metric] = len(conflicts)
        
        # Sort by number of conflicts (descending) - tackle problematic metrics first
        sorted_metrics = sorted(conflict_counts.items(), key=lambda x: x[1], reverse=True)
        
        return pd.DataFrame(sorted_metrics, columns=['metric', 'num_conflicts'])
    
    def visualize_conflict_network(self, figsize=(15, 10)):
        """Visualize metrics and their correlations as a network."""
        import networkx as nx
        
        high_corr_df = self.find_high_correlations()
        
        if len(high_corr_df) == 0:
            print("No high correlations found above threshold!")
            return
        
        G = nx.Graph()
        
        # Add edges for high correlations
        for _, row in high_corr_df.iterrows():
            G.add_edge(row['metric1'], row['metric2'], weight=row['correlation'])
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.7)
        
        # Draw edges with width based on correlation
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title(f"Network of Highly Correlated Metrics (|r| > {self.threshold})", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return G
    
    def create_decision_summary(self):
        """Create a summary table to help with manual selection."""
        groups = self.get_conflict_groups()
        
        if not groups:
            print(f"✓ No metrics correlate above {self.threshold}! All metrics can be included.")
            return None
        
        print(f"\n{'='*80}")
        print(f"CONFLICT GROUPS (correlation > {self.threshold})")
        print(f"{'='*80}\n")
        
        summaries = []
        
        for i, group in enumerate(groups, 1):
            print(f"\nGroup {i}: {len(group)} conflicting metrics")
            print("-" * 80)
            
            for metric in group:
                conflicts = self.analyze_metric_conflicts(metric)
                # Only show conflicts within this group
                group_conflicts = conflicts[conflicts['conflicting_metric'].isin(group)]
                
                summaries.append({
                    'group': i,
                    'metric': metric,
                    'conflicts_in_group': len(group_conflicts),
                    'avg_correlation': group_conflicts['correlation'].mean() if len(group_conflicts) > 0 else 0
                })
                
                print(f"  {metric}")
                if len(group_conflicts) > 0:
                    for _, conflict in group_conflicts.iterrows():
                        print(f"    ├─ {conflict['conflicting_metric']}: r={conflict['correlation']:.3f}")
            
            print(f"\n  → DECISION: Select 1 metric from this group")
        
        return pd.DataFrame(summaries)
    
    def interactive_selection(self):
        """Guide through interactive selection process."""
        print("\n" + "="*80)
        print("INTERACTIVE METRIC SELECTION")
        print("="*80)
        
        # Show summary
        summary_df = self.create_decision_summary()
        
        if summary_df is None:
            return list(self.corr_matrix.columns)
        
        groups = self.get_conflict_groups()
        selected = []
        
        print("\n\nFor each group, select the most interpretable metric:\n")
        
        for i, group in enumerate(groups, 1):
            print(f"\nGroup {i}: Choose 1 from {group}")
            print("-" * 40)
            
            # This would be interactive in a Jupyter notebook
            # For now, just return the groups for manual selection
            
        return groups


# Main analysis function
def analyze_and_select_metrics(correlation_matrix, threshold=0.95):
    """
    Main function to analyze correlations and guide metric selection.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix of metrics
    threshold : float
        Correlation threshold (default 0.95)
    
    Returns:
    --------
    MetricSelector object with analysis results
    """
    selector = MetricSelector(correlation_matrix, threshold)
    
    # 1. Show high correlations
    print("="*80)
    print("HIGH CORRELATIONS FOUND")
    print("="*80)
    high_corr = selector.find_high_correlations()
    if len(high_corr) > 0:
        print(f"\nFound {len(high_corr)} pairs with |correlation| > {threshold}:\n")
        print(high_corr.to_string(index=False))
    else:
        print(f"\nNo correlations above {threshold} found!")
    
    # 2. Show conflict groups
    print("\n")
    summary = selector.create_decision_summary()
    
    # 3. Show suggestion order
    print("\n" + "="*80)
    print("SUGGESTED REVIEW ORDER (by number of conflicts)")
    print("="*80)
    order = selector.suggest_selection_order()
    print(order.to_string(index=False))
    
    # 4. Visualize network
    print("\n\nGenerating conflict network visualization...")
    selector.visualize_conflict_network()
    
    return selector


# In[38]:


"""
Usage script for metric selection with your correlation matrix.

This script will:
1. Identify all groups of metrics that correlate > 0.95
2. Present them in an organized way for you to make decisions
3. Generate visualizations to help you understand the relationships
4. Create a final selected metric list for PCA
"""



# Calculate correlation matrix
correlation_matrix = huge_df.drop(columns=["dataset"]).corr()

# === METRIC SELECTION WORKFLOW ===

print("\n" + "="*80)
print("STEP 1: ANALYZE CORRELATIONS")
print("="*80)

# Run the analysis
selector = analyze_and_select_metrics(correlation_matrix, threshold=0.95)


# In[39]:


# === STEP 2: MANUAL SELECTION ===

print("\n\n" + "="*80)
print("STEP 2: MAKE YOUR SELECTIONS")
print("="*80)

# Get conflict groups
groups = selector.get_conflict_groups()

# Create a manual selection dictionary
# YOU FILL THIS IN based on interpretability
manual_selections = {}

print("\nBased on the groups above, fill in your selections:")
print("\nmanual_selections = {")
for i, group in enumerate(groups, 1):
    print(f"    {i}: 'metric_name_you_choose',  # From: {group}")
print("}\n")


# In[ ]:





# In[40]:


"""
Greedy minimum removal with priority list.

The priority list specifies metrics you want to KEEP.
The algorithm will remove non-priority metrics first before touching priority ones.
"""

import numpy as np
import pandas as pd
import networkx as nx


def greedy_minimum_removal(correlation_matrix, priority_list=[], threshold=0.95):
    """
    Find the minimum set of metrics to REMOVE so that no remaining 
    pair has |correlation| > threshold.
    
    Priority list: Metrics you want to KEEP. The algorithm will remove 
    non-priority metrics first, only removing priority metrics if necessary.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix of metrics
    priority_list : list
        List of metric names to prioritize (keep if possible)
    threshold : float
        Correlation threshold (default 0.95)
    
    Returns:
    --------
    to_remove : list of metrics to drop
    to_keep : list of metrics to keep
    """
    
    # Build conflict graph
    G = nx.Graph()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                G.add_edge(correlation_matrix.columns[i], 
                          correlation_matrix.columns[j],
                          weight=abs(correlation_matrix.iloc[i, j]))
    
    # Convert priority_list to set for faster lookup
    priority_set = set(priority_list)
    
    removed = []
    
    while G.number_of_edges() > 0:
        # Separate nodes into priority and non-priority
        priority_nodes = [n for n in G.nodes() if n in priority_set]
        non_priority_nodes = [n for n in G.nodes() if n not in priority_set]
        
        # Try to remove non-priority nodes first
        if non_priority_nodes:
            # Among non-priority nodes, find the worst offender
            # (most conflicts, highest average correlation)
            worst = max(non_priority_nodes, key=lambda n: (
                G.degree(n), 
                np.mean([G[n][nb]['weight'] for nb in G.neighbors(n)]) if G.degree(n) > 0 else 0
            ))
        else:
            # All remaining nodes are priority - must remove one anyway
            worst = max(priority_nodes, key=lambda n: (
                G.degree(n), 
                np.mean([G[n][nb]['weight'] for nb in G.neighbors(n)]) if G.degree(n) > 0 else 0
            ))
            print(f"  ⚠ Forced to remove priority metric: {worst} (conflicts: {G.degree(worst)})")
        
        removed.append(worst)
        G.remove_node(worst)
    
    all_metrics = set(correlation_matrix.columns)
    kept = sorted(all_metrics - set(removed))
    
    # Statistics
    priority_removed = [m for m in removed if m in priority_set]
    non_priority_removed = [m for m in removed if m not in priority_set]
    priority_kept = [m for m in kept if m in priority_set]
    
    print(f"\n{'='*80}")
    print("GREEDY REMOVAL RESULTS")
    print(f"{'='*80}")
    print(f"Total metrics: {len(all_metrics)}")
    print(f"Priority metrics specified: {len(priority_list)}")
    print(f"Threshold: |r| > {threshold}")
    print(f"\n{'='*80}")
    print(f"REMOVED: {len(removed)} metrics")
    print(f"{'='*80}")
    print(f"  Non-priority removed: {len(non_priority_removed)}")
    print(f"  Priority removed: {len(priority_removed)}")
    
    if non_priority_removed:
        print(f"\n  Non-priority metrics removed ({len(non_priority_removed)}):")
        for m in sorted(non_priority_removed):
            print(f"    ✗ {m}")
    
    if priority_removed:
        print(f"\n  ⚠ Priority metrics removed ({len(priority_removed)}):")
        for m in sorted(priority_removed):
            print(f"    ✗ {m}")
    
    print(f"\n{'='*80}")
    print(f"KEPT: {len(kept)} metrics")
    print(f"{'='*80}")
    print(f"  Priority metrics kept: {len(priority_kept)}/{len(priority_list)}")
    
    if priority_kept:
        print(f"\n  ✓ Priority metrics kept ({len(priority_kept)}):")
        for m in sorted(priority_kept):
            print(f"    ✓ {m}")
    
    non_priority_kept = [m for m in kept if m not in priority_set]
    if non_priority_kept:
        print(f"\n  Other metrics kept ({len(non_priority_kept)}):")
        for m in sorted(non_priority_kept):
            print(f"    ✓ {m}")
    
    return removed, kept


def greedy_minimum_removal_with_explanation(correlation_matrix, priority_list=[], threshold=0.95):
    """
    Same as greedy_minimum_removal but with step-by-step explanation.
    
    Shows which metric was removed at each step and why.
    """
    import networkx as nx
    
    # Build conflict graph
    G = nx.Graph()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                G.add_edge(correlation_matrix.columns[i], 
                          correlation_matrix.columns[j],
                          weight=abs(correlation_matrix.iloc[i, j]))
    
    priority_set = set(priority_list)
    
    removed = []
    step = 0
    
    print(f"\n{'='*80}")
    print("GREEDY REMOVAL - STEP BY STEP")
    print(f"{'='*80}")
    print(f"Initial conflicts: {G.number_of_edges()}")
    print(f"Initial nodes: {G.number_of_nodes()}")
    print(f"Priority metrics: {len(priority_list)}")
    
    while G.number_of_edges() > 0:
        step += 1
        
        # Separate nodes
        priority_nodes = [n for n in G.nodes() if n in priority_set]
        non_priority_nodes = [n for n in G.nodes() if n not in priority_set]
        
        # Find worst offender
        if non_priority_nodes:
            worst = max(non_priority_nodes, key=lambda n: (
                G.degree(n), 
                np.mean([G[n][nb]['weight'] for nb in G.neighbors(n)]) if G.degree(n) > 0 else 0
            ))
            is_priority = False
        else:
            worst = max(priority_nodes, key=lambda n: (
                G.degree(n), 
                np.mean([G[n][nb]['weight'] for nb in G.neighbors(n)]) if G.degree(n) > 0 else 0
            ))
            is_priority = True
        
        conflicts = G.degree(worst)
        avg_corr = np.mean([G[worst][nb]['weight'] for nb in G.neighbors(worst)])
        conflicting_with = list(G.neighbors(worst))
        
        print(f"\nStep {step}:")
        print(f"  Removed: {worst} {'[PRIORITY]' if is_priority else ''}")
        print(f"  Conflicts: {conflicts}")
        print(f"  Avg correlation: {avg_corr:.3f}")
        print(f"  Conflicted with: {', '.join(conflicting_with[:5])}{', ...' if len(conflicting_with) > 5 else ''}")
        print(f"  Remaining conflicts: {G.number_of_edges() - conflicts}")
        
        removed.append(worst)
        G.remove_node(worst)
    
    all_metrics = set(correlation_matrix.columns)
    kept = sorted(all_metrics - set(removed))
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT")
    print(f"{'='*80}")
    print(f"Removed: {len(removed)} metrics in {step} steps")
    print(f"Kept: {len(kept)} metrics")
    
    return removed, kept


# === USAGE EXAMPLES ===
if __name__ == "__main__":
    print("""
Usage:
------

# Example 1: No priority list (basic greedy)
removed, kept = greedy_minimum_removal(correlation_matrix, threshold=0.95)

# Example 2: With priority list (keep these if possible)
priority_metrics = [
    'metric_A',  # I really want to keep this
    'metric_B',  # This is important too
    'metric_C',  # Also interpretable
]
removed, kept = greedy_minimum_removal(
    correlation_matrix, 
    priority_list=priority_metrics,
    threshold=0.95
)

# Example 3: With step-by-step explanation
removed, kept = greedy_minimum_removal_with_explanation(
    correlation_matrix,
    priority_list=priority_metrics,
    threshold=0.95
)

# Use the kept metrics for PCA
pca_data = huge_df[kept]
    """)


# In[41]:


metrics_to_certainly_remove = [] + [f"mc_{i}" for i in range(1, 50)]
#                                 'proportion_long_range_connections_0.1', 
#                                 'proportion_long_range_connections_0.3',
#                                 # "mc_24", "mc_28", "mc_20", "mc_23", "mc_27", "mc_19", "mc_16", "mc_30", "mc_3", "mc_"
#                                 "computational_capacity_cubic_capacity_total", 
#                                 "computational_capacity_memory_timescale",
#                                 "computational_capacity_cross_capacity_total", 
#                                 "avg_edge_distance", 
#                                 "id", 
# ] + [f"mc_{i}" for i in range(1, 10)] + [f"mc_{i}" for i in range(11, 31)]


selected_properties = {
    "Integration": ["global_efficiency"], # , "char_path_length"],
    "Segregation": ["modularity"], # , "avg_communicability"],
    "Wiring\neconomy": ["proportion_long_range_connections_0.5"], # "wiring_cost", "algebraic_connectivity_fiedler_value"],
    "Robustness": ["synchronizability_eigenratio_lambda_2"], # "targeted_attack_robustness_rob_ratio", "persistent_homology_ph_total_persistence"],
    "Synchronisability": ["synchronizability_eigenratio_eigenratio"], # "synchronisability", "synchronisability_normalised"],
    "Computational\ncapacity": ["computational_capacity_total_capacity"], # "computational_capacity", "computational_capacity_normalised"],
    "Metastability": ["kuramoto_averaged_synchronization_r_std"], # "metastability", "metastability_normalised"],
}

my_favs = ["global_efficiency", "modularity", "proportion_long_range_connections_0.5", "synchronizability_eigenratio_lambda_2", "synchronizability_eigenratio_eigenratio", "computational_capacity_total_capacity", "kuramoto_averaged_synchronization_r_std"]


filtered_corr = correlation_matrix.drop(index=metrics_to_certainly_remove, columns=metrics_to_certainly_remove)



priority_metrics = [
    "proportion_long_range_connections_0.5", 
    # "omega"
    "algebraic_connectivity_nx", 
]


removed, kept = greedy_minimum_removal_with_explanation(
    filtered_corr, 
    # correlation_matrix, # filtered_corr,
    priority_list=priority_metrics,
    threshold=1.01 # 0.95
)


# In[42]:


reduced_corr = huge_df[kept].corr()

plt.figure(figsize=(12,12))
sns.heatmap(reduced_corr, annot=False, # fmt=".2f",
            cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix of Metrics (Not everything is labelled... - {reduced_corr.shape[0]} metrics)")
plt.tight_layout()
plt.show()
# Not everything is labelled... 


# In[43]:


# For pca always drop the dataset column and only use the metric columns
pca_df_with_dataset_label = huge_df[kept+["dataset"]]
pca_df_with_dataset_label.drop(columns="dataset") 


# In[44]:


# Create a PCA between the columns of the huge_df
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For pca always drop the dataset column and only use the metric columns
pca_df_with_dataset_label = huge_df[kept+["dataset"]]
# print(pca_df.keys())
# pca_df.head()
pickle.dump(pca_df_with_dataset_label, open(output_folder / "pca_input_df.pkl", "wb"))
pca_df = pca_df_with_dataset_label.drop(columns="dataset")
# print all max values of all columns in pca_df
print("\nMax values of metrics:")
for col in pca_df.columns:
    print(f"{col}: {pca_df[col].max():.4f}")
print("\nMin values of metrics:")
for col in pca_df.columns:
    print(f"{col}: {pca_df[col].min():.4f}")

# Drop the columns that have inf or -inf values (if any). Print the dropped columns
dropped_cols = pca_df.columns[pca_df.isnull().any()].tolist()
pca_df.replace([np.inf, -np.inf], np.nan, inplace=True)
pca_df.dropna(axis=1, inplace=True)
print(f"Dropped columns: {dropped_cols}")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(pca_df)

pca = PCA(n_components=10, random_state=8)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=pca_df_with_dataset_label['dataset'])
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
plt.title("Scree Plot")
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.tight_layout()
plt.show()


# In[45]:


# Subplots of loadings for the first 3 principal components
loadings = pca.components_.T
num_metrics = loadings.shape[0]
metric_names = huge_df_metrics.columns

# order them all by the absolute value of their loading on the first principal component
sorted_indices = np.argsort(np.abs(loadings[:, 0])) #[::-1]
loadings = loadings[sorted_indices]
metric_names = metric_names[sorted_indices]


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,20), dpi=100)
for i in range(3):
    axs[i].barh(range(num_metrics), loadings[:, i])
    axs[i].set_yticks(range(num_metrics))
    axs[i].set_yticklabels(metric_names, rotation=0)
    axs[i].set_title(f"Loadings for PC{i+1}")
plt.tight_layout()
plt.show()



# In[46]:


pca_result.shape


# In[47]:


# After fitting PCA and getting pca_scores (shape: n_samples x n_components)

corners = {
    "top_left":    pca_result[:, 1].argmax(),                        # max PC2
    "bottom_left": (-pca_result[:, 0] - pca_result[:, 1]).argmax(), # min PC1 + min PC2
    "right":       pca_result[:, 0].argmax(),                        # max PC1
}


# In[48]:


huge_df_metrics


# In[49]:


# pca.components_ shape: (n_components, n_features)
# Use PC1 and PC2 loadings combined
loading_magnitude = np.sqrt(pca.components_[0]**2 + pca.components_[1]**2)
top6_idx = np.argsort(loading_magnitude)[::-1][:6]
top6_features = [feature_names[i] for i in top6_idx]


# # OLDER 

# In[ ]:


# # manual_selections = {
# #     1: 'metric_name_you_choose',  # From: ['avg_edge_distance', 'modularity_dup', 'ollivier_ricci_curvature_orc_frac_neg', 'proportion_long_range_connections_0.1', 'proportion_long_range_connections_0.3', 'proportion_long_range_connections_0.5', 'richclub_avg_length', 'wiring_cost', 'wiring_cost_dup', 'wiring_cost_static']
# #     2: 'metric_name_you_choose',  # From: ['network_idx_computational', 'network_idx_dynamic', 'network_idx_further']
# #     3: 'metric_name_you_choose',  # From: ['avg_clustering', 'avg_clustering_static', 'avg_communicability', 'char_path_length', 'effective_dimensionality', 'global_efficiency', 'global_efficiency_dynamic', 'modularity_dup', 'nct_control_n_nodes_50_percent', 'nct_control_std', 'omega', 'participation_coefficient_pc_frac_connector', 'participation_coefficient_pc_mean', 'participation_coefficient_pc_median', 'persistent_homology_ph_h1_entropy', 'propagation_efficiency', 'richclub_n_edges', 'structural_complexity', 'topological_distance_std', 'transitivity', 'transitivity_static']
# #     4: 'metric_name_you_choose',  # From: ['avg_clustering', 'avg_clustering_static', 'effective_dimensionality', 'omega', 'participation_coefficient_pc_mean', 'participation_coefficient_pc_median', 'topological_distance_std', 'transitivity', 'transitivity_static']
# #     5: 'metric_name_you_choose',  # From: ['avg_clustering', 'avg_clustering_static']
# #     6: 'metric_name_you_choose',  # From: ['avg_communicability', 'degree_gini', 'kernel_rank_max', 'richclub_n_edges', 'spectral_radius']
# #     7: 'metric_name_you_choose',  # From: ['algebraic_connectivity_fiedler_value_norm', 'persistent_homology_ph_h1_n_features', 'persistent_homology_ph_total_persistence']
# #     8: 'wiring_cost_static',  # !!!!!! From: ['wiring_cost', 'wiring_cost_static']
# #     9: 'metric_name_you_choose',  # From: ['algebraic_connectivity_fiedler_value', 'algebraic_connectivity_fiedler_value_norm', 'synchronizability_eigenratio_lambda_2']
# #     10: 'metric_name_you_choose',  # From: ['computational_capacity_cross_capacity_total', 'computational_capacity_cubic_capacity_total', 'computational_capacity_memory_capacity_total', 'computational_capacity_memory_timescale', 'computational_capacity_nonlinear_capacity_total', 'computational_capacity_total_capacity']
# #     11: 'mc_15',  # From: ['mc_14', 'mc_15', 'mc_16', 'mc_17', 'mc_18', 'mc_19', 'mc_20', 'mc_21', 'mc_22', 'mc_23', 'mc_24', 'mc_25', 'mc_26', 'mc_27', 'mc_28', 'mc_29', 'mc_30', 'mc_31', 'mc_32', 'mc_33']
# #     12: 'modularity', # modularity_static',  # From: ['diffusion_efficiency', 'modularity', 'modularity_dup', 'modularity_static']
# #     13: 'omega',  # From: ['omega', 'topological_distance_std']
# #     14: 'avg_communicability',  # From: ['avg_communicability', 'effective_dimensionality', 'nct_control_std', 'propagation_efficiency', 'structural_complexity']
# #     15: 'computational_capacity_total_capacity',  # From: ['computational_capacity_cross_capacity_total', 'computational_capacity_total_capacity']
# #     16: 'computational_capacity_state_entropy',  # !! From: ['computational_capacity_state_dimensionality', 'computational_capacity_state_entropy']
# #     17: 'metric_name_you_choose',  # From: ['participation_coefficient_pc_mean', 'participation_coefficient_pc_median']
# #     18: 'char_path_length',  # From: ['char_path_length', 'targeted_attack_robustness_rob_random_auc', 'targeted_attack_robustness_rob_random_half']
# #     19: 'mc_3',  # From: ['mc_1', 'mc_2', 'mc_3', 'mc_4']
# #     20: 'ollivier_ricci_curvature_orc_mean',  # From: ['ollivier_ricci_curvature_orc_mean', 'ollivier_ricci_curvature_orc_median']
# # }

# manual_selections = {
#     1: 'metric_name_you_choose',  # From: ['avg_clustering', 'avg_communicability', 'char_path_length', 'community_synchronization_vulnerability_value', 'degree_gini', 'effective_dimensionality', 'global_efficiency', 'kernel_rank_max', 'nct_control_n_nodes_50_percent', 'nct_control_std', 'omega', 'participation_coefficient_pc_frac_connector', 'participation_coefficient_pc_frac_connector_further', 'participation_coefficient_pc_mean', 'participation_coefficient_pc_mean_further', 'participation_coefficient_pc_median', 'participation_coefficient_pc_median_further', 'persistent_homology_ph_h1_entropy', 'propagation_efficiency', 'richclub_n_edges', 'spectral_radius', 'structural_complexity', 'topological_distance_std', 'transitivity']
#     2: 'metric_name_you_choose',  # From: ['participation_coefficient_pc_std', 'participation_coefficient_pc_std_further']
#     3: 'metric_name_you_choose',  # From: ['participation_coefficient_n_communities', 'participation_coefficient_n_communities_further']
#     4: 'spectral_radius',  # From: ['kernel_rank_max', 'spectral_radius']
#     5: 'spectral_gap_fatemeh',  # From: ['spectral_gap', 'spectral_gap_fatemeh']
#     6: 'metric_name_you_choose',  # From: ['participation_coefficient_wmd_std', 'participation_coefficient_wmd_std_further']
#     7: 'metric_name_you_choose',  # From: ['participation_coefficient_wmd_mean', 'participation_coefficient_wmd_mean_further']
#     8: 'metric_name_you_choose',  # From: ['algebraic_connectivity_fiedler_value_norm', 'persistent_homology_ph_h1_n_features', 'persistent_homology_ph_total_persistence']
#     9: 'metric_name_you_choose',  # From: ['participation_coefficient_pc_mean', 'participation_coefficient_pc_mean_further']
#     10: 'metric_name_you_choose',  # From: ['algebraic_connectivity_fiedler_value', 'algebraic_connectivity_fiedler_value_norm', 'synchronizability_eigenratio_lambda_2']
#     11: 'metric_name_you_choose',  # From: ['community_synchronization_vulnerability_value', 'participation_coefficient_pc_frac_connector', 'participation_coefficient_pc_frac_connector_further']
#     12: 'metric_name_you_choose',  # From: ['computational_capacity_cross_capacity_total', 'computational_capacity_cubic_capacity_total', 'computational_capacity_memory_capacity_total', 'computational_capacity_memory_timescale', 'computational_capacity_nonlinear_capacity_total', 'computational_capacity_total_capacity']
#     13: 'metric_name_you_choose',  # From: ['mc_14', 'mc_15', 'mc_16', 'mc_17', 'mc_18', 'mc_19', 'mc_20', 'mc_21', 'mc_22', 'mc_23', 'mc_24', 'mc_25', 'mc_26', 'mc_27', 'mc_28', 'mc_29', 'mc_30', 'mc_31', 'mc_32']
#     14: 'proportion_long_range_connections_0.5',  # From: ['avg_edge_distance', 'proportion_long_range_connections_0.1', 'proportion_long_range_connections_0.3', 'proportion_long_range_connections_0.5']
#     15: 'metric_name_you_choose',  # From: ['avg_communicability', 'effective_dimensionality', 'nct_control_std', 'propagation_efficiency', 'structural_complexity']
#     16: 'metric_name_you_choose',  # From: ['avg_clustering', 'omega', 'topological_distance_std', 'transitivity']
#     17: 'metric_name_you_choose',  # From: ['avg_clustering', 'transitivity']
#     18: 'metric_name_you_choose',  # From: ['computational_capacity_cross_capacity_total', 'computational_capacity_total_capacity']
#     19: 'metric_name_you_choose',  # From: ['computational_capacity_state_dimensionality', 'computational_capacity_state_entropy']
#     20: 'metric_name_you_choose',  # From: ['char_path_length', 'targeted_attack_robustness_rob_random_auc', 'targeted_attack_robustness_rob_random_half']
#     21: 'metric_name_you_choose',  # From: ['mc_1', 'mc_2', 'mc_3']
#     22: 'metric_name_you_choose',  # From: ['ollivier_ricci_curvature_orc_mean', 'ollivier_ricci_curvature_orc_median']
#     23: 'metric_name_you_choose',  # From: ['targeted_attack_robustness_rob_targeted_auc', 'targeted_attack_robustness_rob_targeted_half']
# }


# In[ ]:


# def greedy_minimum_removal(correlation_matrix, priority_list=[], threshold=0.95):
#     """
#     Find the minimum set of metrics to REMOVE so that no remaining 
#     pair has |correlation| > threshold.
#     Priority list: Remove the other ones first, before removing these. 
    
#     Returns:
#     --------
#     to_remove : list of metrics to drop
#     to_keep : list of metrics to keep
#     """
#     import networkx as nx
    
#     # Build conflict graph
#     G = nx.Graph()
#     for i in range(len(correlation_matrix.columns)):
#         for j in range(i+1, len(correlation_matrix.columns)):
#             if abs(correlation_matrix.iloc[i, j]) > threshold:
#                 G.add_edge(correlation_matrix.columns[i], 
#                           correlation_matrix.columns[j],
#                           weight=abs(correlation_matrix.iloc[i, j]))
    
#     removed = []
#     while G.number_of_edges() > 0:
#         # Remove node with most conflicts (ties broken by avg correlation)
#         worst = max(G.nodes(), key=lambda n: (
#             G.degree(n), 
#             np.mean([G[n][nb]['weight'] for nb in G.neighbors(n)]) if G.degree(n) > 0 else 0
#         ))
#         removed.append(worst)
#         G.remove_node(worst)
    
#     all_metrics = set(correlation_matrix.columns)
#     kept = sorted(all_metrics - set(removed))
    
#     print(f"Total metrics: {len(all_metrics)}")
#     print(f"Removed: {len(removed)}")
#     print(f"Kept: {len(kept)}")
#     print(f"\nMetrics to REMOVE ({len(removed)}):")
#     for m in sorted(removed):
#         print(f"  ✗ {m}")
    
#     return removed, kept


# In[ ]:


# def interactive_removal(correlation_matrix, priority_list=[], threshold=0.95):
#     """
#     Propose removals, let user override to keep specific metrics.
#     Properties in priority_list will be considered more interpretable and thus less likely to be removed.
#     """
#     removed, kept = greedy_minimum_removal(correlation_matrix, threshold, priority_list)
    
#     print("\n" + "="*80)
#     print("PROPOSED REMOVALS — override any you disagree with")
#     print("="*80)
    
#     for i, metric in enumerate(sorted(removed), 1):
#         # Show what it conflicts with in the kept set
#         conflicts_with_kept = []
#         for k in kept:
#             if abs(correlation_matrix.loc[metric, k]) > threshold:
#                 conflicts_with_kept.append(
#                     f"{k} (r={correlation_matrix.loc[metric, k]:.3f})"
#                 )
#         print(f"\n  {i}. REMOVE: {metric}")
#         if conflicts_with_kept:
#             print(f"     Because it conflicts with kept metrics:")
#             for c in conflicts_with_kept:
#                 print(f"       → {c}")
#         else:
#             print(f"     (Transitively required — conflicts were with other removed metrics)")
    
#     return removed, kept



# # If you want to force-keep a metric, swap it with its conflict:
# def force_keep(metric, removed, kept, correlation_matrix, threshold=0.95):
#     """Force-keep a metric, removing its conflicts instead."""
#     if metric not in removed:
#         print(f"{metric} is already kept!")
#         return removed, kept
    
#     removed = list(removed)
#     kept = list(kept)
#     removed.remove(metric)
#     kept.append(metric)
    
#     # Now find new conflicts and resolve them
#     for other in list(kept):
#         if other != metric and abs(correlation_matrix.loc[metric, other]) > threshold:
#             print(f"  Conflict: {metric} ↔ {other} (r={correlation_matrix.loc[metric, other]:.3f})")
#             print(f"  → Removing {other} instead")
#             kept.remove(other)
#             removed.append(other)
    
#     return removed, kept


# In[ ]:


removed, kept = interactive_removal(correlation_matrix, threshold=0.95)


# In[ ]:


# # remove one property from the correlation_matrix 
# changed_correlation_matrix = correlation_matrix.drop(columns=["algebraic_connectivity_nx"])
# changed_correlation_matrix


# In[ ]:


# for property_to_keep in ["algebraic_connectivity_nx"]: 
#     removed, kept = force_keep(property_to_keep, removed, kept, correlation_matrix, threshold=0.95)
    
# # removed, kept = force_keep("avg_communicability", removed, kept, correlation_matrix, threshold=0.95)
# # removed, kept = force_keep("computational_capacity_memory_capacity_total", removed, kept, correlation_matrix, threshold=0.95)


# In[ ]:


# """
# Progress visualization for metric selection.

# This script shows your progress through the selection process:
# - Green nodes: Metrics you've selected
# - Red nodes: Metrics you've excluded (from same group as selected)
# - Gray nodes: Metrics still needing decisions
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx


# class ProgressVisualizer:
#     """Visualize selection progress on the conflict network."""
    
#     def __init__(self, correlation_matrix, threshold=0.95):
#         self.selector = MetricSelector(correlation_matrix, threshold)
#         self.corr_matrix = correlation_matrix
#         self.threshold = threshold
        
#     def plot_progress(self, manual_selections, figsize=(16, 12)):
#         """
#         Plot conflict network with progress visualization.
        
#         Parameters:
#         -----------
#         manual_selections : dict
#             Dictionary mapping group numbers to selected metric names
#             {1: 'metric_name', 2: 'another_metric', ...}
#             Can be incomplete - groups without selections will show as undecided
#         figsize : tuple
#             Figure size
#         """
#         # Get conflict groups
#         groups = self.selector.get_conflict_groups()
        
#         # Categorize metrics
#         selected_metrics = set()
#         excluded_metrics = set()
#         undecided_metrics = set()
        
#         for group_num, group in enumerate(groups, 1):
#             if group_num in manual_selections:
#                 chosen = manual_selections[group_num]
#                 selected_metrics.add(chosen)
#                 # All others in this group are excluded
#                 for metric in group:
#                     if metric != chosen:
#                         excluded_metrics.add(metric)
#             else:
#                 # Entire group is still undecided
#                 undecided_metrics.update(group)
        
#         # Build network
#         high_corr_df = self.selector.find_high_correlations()
        
#         if len(high_corr_df) == 0:
#             print("No high correlations found above threshold!")
#             return
        
#         G = nx.Graph()
        
#         # Add edges for high correlations
#         for _, row in high_corr_df.iterrows():
#             G.add_edge(row['metric1'], row['metric2'], weight=row['correlation'])
        
#         # Create figure
#         fig, ax = plt.subplots(figsize=figsize)
        
#         # Layout
#         pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
#         # Separate nodes by status
#         selected_nodes = [n for n in G.nodes() if n in selected_metrics]
#         excluded_nodes = [n for n in G.nodes() if n in excluded_metrics]
#         undecided_nodes = [n for n in G.nodes() if n in undecided_metrics]
        
#         # Draw nodes with different colors
#         if selected_nodes:
#             nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, 
#                                   node_size=3500, node_color='lightgreen', 
#                                   edgecolors='darkgreen', linewidths=3,
#                                   alpha=0.9, ax=ax, label='Selected')
        
#         if excluded_nodes:
#             nx.draw_networkx_nodes(G, pos, nodelist=excluded_nodes, 
#                                   node_size=3000, node_color='lightcoral', 
#                                   edgecolors='darkred', linewidths=2,
#                                   alpha=0.5, ax=ax, label='Excluded')
        
#         if undecided_nodes:
#             nx.draw_networkx_nodes(G, pos, nodelist=undecided_nodes, 
#                                   node_size=3000, node_color='lightgray', 
#                                   edgecolors='black', linewidths=2,
#                                   alpha=0.7, ax=ax, label='Needs Decision')
        
#         # Draw edges
#         edges = G.edges()
#         weights = [G[u][v]['weight'] for u, v in edges]
        
#         # Color edges based on whether they're resolved or not
#         edge_colors = []
#         edge_widths = []
#         for u, v in edges:
#             u_selected = u in selected_metrics
#             v_selected = v in selected_metrics
#             u_excluded = u in excluded_metrics
#             v_excluded = v in excluded_metrics
            
#             if (u_selected or u_excluded) and (v_selected or v_excluded):
#                 # Both nodes decided - edge is resolved
#                 edge_colors.append('green')
#                 edge_widths.append(1)
#             else:
#                 # At least one node undecided - conflict remains
#                 edge_colors.append('red')
#                 edge_widths.append(3)
        
#         nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, 
#                               edge_color=edge_colors, ax=ax)
        
#         # Draw labels
#         nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        
#         # Create progress summary
#         total_groups = len(groups)
#         decided_groups = len(manual_selections)
#         total_conflicts = len(high_corr_df)
        
#         # Count resolved conflicts (both nodes decided)
#         resolved_conflicts = sum(1 for color in edge_colors if color == 'green')
        
#         # Title with progress
#         progress_pct = (decided_groups / total_groups * 100) if total_groups > 0 else 100
#         title = (f"Metric Selection Progress: {decided_groups}/{total_groups} groups decided "
#                 f"({progress_pct:.0f}%)\n"
#                 f"Conflicts: {resolved_conflicts}/{total_conflicts} resolved")
        
#         ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
#         # Legend
#         ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        
#         # Stats box
#         stats_text = (
#             f"Selected: {len(selected_metrics)}\n"
#             f"Excluded: {len(excluded_metrics)}\n"
#             f"Undecided: {len(undecided_metrics)}"
#         )
#         ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
#                fontsize=11, verticalalignment='bottom',
#                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
#         ax.axis('off')
#         plt.tight_layout()
#         plt.show()
        
#         # Print remaining decisions
#         print("\n" + "="*80)
#         print("REMAINING DECISIONS")
#         print("="*80)
        
#         undecided_groups = []
#         for group_num, group in enumerate(groups, 1):
#             if group_num not in manual_selections:
#                 undecided_groups.append((group_num, group))
        
#         if undecided_groups:
#             print(f"\nYou still need to decide on {len(undecided_groups)} groups:\n")
#             for group_num, group in undecided_groups:
#                 print(f"Group {group_num}: Choose 1 from {group}")
#         else:
#             print("\n✓ All groups decided! Ready for validation.")
        
#         return {
#             'selected': list(selected_metrics),
#             'excluded': list(excluded_metrics),
#             'undecided': list(undecided_metrics),
#             'progress_pct': progress_pct
#         }


# def show_progress(correlation_matrix, manual_selections, threshold=0.95):
#     """
#     Quick function to show progress visualization.
    
#     Parameters:
#     -----------
#     correlation_matrix : pd.DataFrame
#         Your correlation matrix
#     manual_selections : dict
#         Your selections so far {group_num: 'metric_name', ...}
#     threshold : float
#         Correlation threshold
    
#     Returns:
#     --------
#     dict with progress statistics
#     """
#     visualizer = ProgressVisualizer(correlation_matrix, threshold)
#     return visualizer.plot_progress(manual_selections)


# # === USAGE EXAMPLE ===
# if __name__ == "__main__":
#     print("""
# Usage:
# ------

# # After you've made some selections:
# manual_selections = {
#     1: 'chosen_metric_A',
#     2: 'chosen_metric_B',
#     # Group 3 not decided yet
#     # Group 4 not decided yet
# }

# # Show progress:
# from progress_visualizer import show_progress
# stats = show_progress(correlation_matrix, manual_selections, threshold=0.95)

# # Green nodes = your selections
# # Red nodes = excluded (in same group as your selection)  
# # Gray nodes = still need decisions
# # Green edges = conflict resolved
# # Red edges = conflict still active
#     """)


# In[ ]:


# # === Show progress visualization ===
# # Assumes you already have 'correlation_matrix' from your previous code
# stats = show_progress(correlation_matrix, manual_selections, threshold=0.95)

# print("\n" + "="*80)
# print("SUMMARY")
# print("="*80)
# print(f"Progress: {stats['progress_pct']:.1f}% complete")
# print(f"Selected: {len(stats['selected'])} metrics")
# print(f"Excluded: {len(stats['excluded'])} metrics")
# print(f"Undecided: {len(stats['undecided'])} metrics")

# if stats['undecided']:
#     print(f"\nMetrics still needing decisions:")
#     for metric in sorted(stats['undecided']):
#         print(f"  - {metric}")


# In[ ]:


# """
# Show conflict network for only the metrics still under consideration.

# After you've made some selections, this filters out the decided metrics
# and shows you only the remaining conflicts.
# """


# def get_remaining_metrics(correlation_matrix, groups, manual_selections):
#     """
#     Get list of metrics still under consideration.
    
#     Parameters:
#     -----------
#     correlation_matrix : pd.DataFrame
#         Full correlation matrix
#     groups : list of lists
#         Conflict groups from selector
#     manual_selections : dict
#         Your selections so far {group_num: 'metric_name', ...}
    
#     Returns:
#     --------
#     set of metric names still under consideration
#     """
#     all_metrics = set(correlation_matrix.columns)
#     excluded_metrics = set()
    
#     # For each group where you've made a selection
#     for group_num, group in enumerate(groups, 1):
#         if group_num in manual_selections:
#             chosen = manual_selections[group_num]
#             # Exclude all metrics in this group except the chosen one
#             for metric in group:
#                 if metric != chosen:
#                     excluded_metrics.add(metric)
    
#     # Remaining metrics = all metrics minus excluded ones
#     remaining = all_metrics - excluded_metrics
    
#     return remaining


# def plot_remaining_conflicts(correlation_matrix, manual_selections, threshold=0.95, figsize=(15, 10)):
#     """
#     Plot conflict network showing ONLY metrics still under consideration.
    
#     Parameters:
#     -----------
#     correlation_matrix : pd.DataFrame
#         Full correlation matrix
#     manual_selections : dict
#         Your selections so far {group_num: 'metric_name', ...}
#     threshold : float
#         Correlation threshold
#     figsize : tuple
#         Figure size
#     """
#     # Get conflict groups
#     selector = MetricSelector(correlation_matrix, threshold)
#     groups = selector.get_conflict_groups()
    
#     # Get metrics still under consideration
#     remaining_metrics = get_remaining_metrics(correlation_matrix, groups, manual_selections)
    
#     print(f"\nTotal metrics: {len(correlation_matrix.columns)}")
#     print(f"Remaining under consideration: {len(remaining_metrics)}")
#     print(f"Excluded: {len(correlation_matrix.columns) - len(remaining_metrics)}")
    
#     # Filter correlation matrix to only remaining metrics
#     remaining_corr = correlation_matrix.loc[list(remaining_metrics), list(remaining_metrics)]
    
#     # Find high correlations among remaining metrics
#     high_corr_pairs = []
#     for i in range(len(remaining_corr.columns)):
#         for j in range(i+1, len(remaining_corr.columns)):
#             metric1 = remaining_corr.columns[i]
#             metric2 = remaining_corr.columns[j]
#             corr_value = abs(remaining_corr.iloc[i, j])
            
#             if corr_value > threshold:
#                 high_corr_pairs.append({
#                     'metric1': metric1,
#                     'metric2': metric2,
#                     'correlation': corr_value
#                 })
    
#     if len(high_corr_pairs) == 0:
#         print(f"\n✓ No remaining conflicts! All metrics under consideration have |r| ≤ {threshold}")
#         return None
    
#     print(f"\nRemaining conflicts: {len(high_corr_pairs)} pairs")
    
#     # Build network graph
#     G = nx.Graph()
    
#     for pair in high_corr_pairs:
#         G.add_edge(pair['metric1'], pair['metric2'], weight=pair['correlation'])
    
#     # Plot
#     plt.figure(figsize=figsize)
#     pos = nx.spring_layout(G, k=2, iterations=50)
    
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.7)
    
#     # Draw edges with width based on correlation
#     edges = G.edges()
#     weights = [G[u][v]['weight'] for u, v in edges]
#     nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.5)
    
#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=10)
    
#     # Title
#     decided_groups = len(manual_selections)
#     total_groups = len(groups)
#     plt.title(f"Remaining Conflicts (|r| > {threshold})\n"
#              f"Decisions made: {decided_groups}/{total_groups} groups | "
#              f"Metrics remaining: {len(remaining_metrics)}/{len(correlation_matrix.columns)}",
#              fontsize=14, fontweight='bold')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
#     # Print remaining conflict groups
#     print("\n" + "="*80)
#     print("REMAINING CONFLICT GROUPS")
#     print("="*80)
    
#     # Rebuild groups for remaining metrics
#     processed = set()
#     remaining_groups = []
    
#     for pair in high_corr_pairs:
#         m1, m2 = pair['metric1'], pair['metric2']
        
#         # Find if either metric is already in a group
#         found_group = None
#         for group in remaining_groups:
#             if m1 in group or m2 in group:
#                 found_group = group
#                 break
        
#         if found_group:
#             found_group.add(m1)
#             found_group.add(m2)
#         else:
#             remaining_groups.append({m1, m2})
    
#     for i, group in enumerate(remaining_groups, 1):
#         print(f"\nGroup {i}: Choose 1 from {sorted(list(group))}")
    
#     return {
#         'remaining_metrics': list(remaining_metrics),
#         'remaining_conflicts': high_corr_pairs,
#         'remaining_groups': [sorted(list(g)) for g in remaining_groups]
#     }


# def plot_remaining_correlation_heatmap(correlation_matrix, manual_selections, threshold=0.95, figsize=(12, 12)):
#     """
#     Plot correlation heatmap showing ONLY metrics still under consideration.
    
#     Parameters:
#     -----------
#     correlation_matrix : pd.DataFrame
#         Full correlation matrix
#     manual_selections : dict
#         Your selections so far {group_num: 'metric_name', ...}
#     threshold : float
#         Correlation threshold
#     figsize : tuple
#         Figure size
#     """
#     # Get conflict groups
#     selector = MetricSelector(correlation_matrix, threshold)
#     groups = selector.get_conflict_groups()
    
#     # Get metrics still under consideration
#     remaining_metrics = get_remaining_metrics(correlation_matrix, groups, manual_selections)
    
#     # Filter correlation matrix
#     remaining_corr = correlation_matrix.loc[list(remaining_metrics), list(remaining_metrics)]
    
#     # Plot
#     plt.figure(figsize=figsize)
#     sns.heatmap(remaining_corr, annot=False, fmt=".2f", cmap="coolwarm", cbar=True,
#                 vmin=-1, vmax=1, center=0)
    
#     decided_groups = len(manual_selections)
#     total_groups = len(groups)
#     plt.title(f"Correlation Matrix - Remaining Metrics\n"
#              f"Decisions: {decided_groups}/{total_groups} | "
#              f"Metrics: {len(remaining_metrics)}/{len(correlation_matrix.columns)}",
#              fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.show()






# In[ ]:


# # Show only the remaining conflicts after some selections.

# print("\n" + "="*80)
# print("REMAINING CONFLICTS NETWORK")
# print("="*80)

# # This will show the conflict network with ONLY undecided metrics
# result = plot_remaining_conflicts(correlation_matrix, manual_selections, threshold=0.95)

# # === OPTIONAL: Show remaining correlation heatmap ===

# print("\n" + "="*80)
# print("REMAINING CORRELATION HEATMAP")
# print("="*80)

# # This shows the correlation heatmap with ONLY undecided metrics
# plot_remaining_correlation_heatmap(correlation_matrix, manual_selections, threshold=0.95)

# # === SUMMARY ===

# if result:
#     print("\n" + "="*80)
#     print("NEXT STEPS")
#     print("="*80)
#     print(f"\nYou have {len(result['remaining_groups'])} groups left to decide:")
#     for i, group in enumerate(result['remaining_groups'], 1):
#         print(f"\nGroup {i}: Choose 1 from:")
#         for metric in group:
#             print(f"  - {metric}")


# In[ ]:


# # Calculate correlation matrix
# correlation_matrix = huge_df.drop(columns=["dataset"]).corr()

# # === METRIC SELECTION WORKFLOW ===

# print("\n" + "="*80)
# print("STEP 1: ANALYZE CORRELATIONS")
# print("="*80)

# # Run the analysis
# selector = analyze_and_select_metrics(correlation_matrix, threshold=0.95)


# In[ ]:


huge_df


# In[ ]:


# remaining_metrics = get_remaining_metrics(correlation_matrix, groups, manual_selections)
# len(remaining_metrics)


# In[ ]:


# # Make a correlation axis between the columns of the huge_df. Don't use the dataset column for this, only the metric columns.
# correlation_matrix = huge_df[list(remaining_metrics)].corr()
# plt.figure(figsize=(12,12))
# sns.heatmap(correlation_matrix, annot=False, # fmt=".2f", 
#             cmap="coolwarm", cbar=True)
# plt.title(f"Correlation Matrix of Metrics (Not everything is labelled... - {correlation_matrix.shape[0]} metrics)")
# plt.tight_layout()
# plt.show()
# # Not everything is labelled... 


# In[ ]:





# In[ ]:


# Get correlations with wiring_cost, sorted by absolute value
# wiring_correlations = correlation_matrix["wiring_cost"].abs().sort_values(ascending=False)

# # Display top correlations (excluding wiring_cost itself)
# print("Top correlations with wiring_cost:")
# print(wiring_correlations[1:11])  # Skip first (itself), show top 10

# # Or if you want to see positive and negative correlations separately
# wiring_correlations_signed = correlation_matrix["wiring_cost"].sort_values(ascending=False)
# print("\nAll correlations with wiring_cost (sorted):")
# print(wiring_correlations_signed[1:])  # Skip wiring_cost itself
# # Bar plot of top correlations
# top_n = 15
# top_corrs = wiring_correlations[1:top_n+1]  # Skip wiring_cost itself

plt.figure(figsize=(10, 6))
top_corrs.plot(kind='barh')
plt.xlabel('Absolute Correlation with wiring_cost')
plt.title(f'Top {top_n} Features Correlated with wiring_cost')
plt.tight_layout()
plt.show()
# Create a PCA between the columns of the huge_df
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For pca always drop the dataset column and only use the metric columns
huge_df_metrics = huge_df[list(remaining_metrics)] # huge_df.drop(columns=["dataset"])
print(huge_df_metrics.keys())

# print all max values of all columns in huge_df_metrics
print("\nMax values of metrics:")
for col in huge_df_metrics.columns:
    print(f"{col}: {huge_df_metrics[col].max():.4f}")
print("\nMin values of metrics:")
for col in huge_df_metrics.columns:
    print(f"{col}: {huge_df_metrics[col].min():.4f}")
    
# Drop the columns that have inf or -inf values (if any). Print the dropped columns
dropped_cols = huge_df_metrics.columns[huge_df_metrics.isnull().any()].tolist()
huge_df_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)
huge_df_metrics.dropna(axis=1, inplace=True)
print(f"Dropped columns: {dropped_cols}")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(huge_df_metrics)

pca = PCA(n_components=10)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=huge_df['dataset'])
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
plt.title("Scree Plot")
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.tight_layout()
plt.show()
# Subplots of loadings for the first 3 principal components
loadings = pca.components_.T
num_metrics = loadings.shape[0]
metric_names = huge_df_metrics.columns

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


# Calculate correlation matrix
correlation_matrix = huge_df.drop(columns=["dataset"]).corr()

# === METRIC SELECTION WORKFLOW ===

print("\n" + "="*80)
print("STEP 1: ANALYZE CORRELATIONS")
print("="*80)

# Run the analysis
selector = analyze_and_select_metrics(correlation_matrix, threshold=0.95)


# # OLDER 
# 

# In[ ]:


plt.scatter(huge_df["wiring_cost_dup"], huge_df["wiring_cost"], c=huge_df["dataset"].astype('category').cat.codes, cmap='tab10', alpha=0.3, s=4)


# In[ ]:


plt.scatter(huge_df["wiring_cost_dup"], huge_df["mc_19"], c=huge_df["dataset"].astype('category').cat.codes, cmap='tab10', alpha=0.3, s=4)


# In[ ]:


plt.hist(np.array(total_possible_wiring_len_mami)/100/99*2)
total_possible_wiring_len_schaeffer/100/99*2


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

wiring_cost_versions = ["wiring_cost", "wiring_cost_dup", "wiring_cost_div_by_matrix", "wiring_cost_dup_div_by_matrix"] 

datasets = huge_df["dataset"].unique()
n_datasets = len(datasets)

fig, axs = plt.subplots(nrows=len(wiring_cost_versions), ncols=n_datasets, 
                        figsize=(n_datasets*4, len(wiring_cost_versions)*3), dpi=100)

for i, version in enumerate(wiring_cost_versions):
    for j, dataset in enumerate(datasets):
        ax = axs[i, j] if n_datasets > 1 else axs[i]
        
        subset = huge_df[huge_df["dataset"] == dataset]
        valid_data = subset[version].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(valid_data) > 0:
            ax.hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f"{dataset}")
            ax.set_xlabel(version)
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
        
        # Add statistics as text
        if len(valid_data) > 0:
            stats_text = f"μ={valid_data.mean():.2e}\nσ={valid_data.std():.2e}\nn={len(valid_data)}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # print those statistics to the console as well
            print(f"{dataset} - {version}: mean={valid_data.mean():.2e}, std={valid_data.std():.2e}, n={len(valid_data)}")


plt.tight_layout()
plt.show()


# In[ ]:


# create subplots with the different wiring_cost versions: 
wiring_cost_versions = ["wiring_cost", "wiring_cost_dup", "wiring_cost_div_by_matrix", "wiring_cost_dup_div_by_matrix"] 
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=viz.cm_to_inch((18,12)), dpi=100) 

for i, version in enumerate(wiring_cost_versions): 
    row = i // 2 
    col = i % 2 
    
    for dataset in huge_df["dataset"].unique():
        subset = huge_df[huge_df["dataset"] == dataset]
        print(f"Dataset: {dataset}, Version: {version}.") 
        print(f"    Mean: {subset[version].mean():.4f}, Std: {subset[version].std():.4f}, Number of samples: {len(subset[version])}")
        
        # Filter out NaN and inf values
        valid_data = subset[version].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(valid_data) > 0:  # Only plot if there's valid data
            axs[row, col].hist(valid_data, bins=50, alpha=0.7, density=True, label=dataset)
        else:
            print(f"    Warning: No valid data for {dataset} - {version}")
    
    axs[row, col].set_xlabel(version)
    axs[row, col].set_ylabel("Density")
# axs[row, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left')


plt.tight_layout() 
plt.show()


# In[ ]:


# create subplots with the different wiring_cost versions: 
wiring_cost_versions = ["wiring_cost", "wiring_cost_dup", "wiring_cost_div_by_matrix", "wiring_cost_dup_div_by_matrix"] 
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=viz.cm_to_inch((18,12)), dpi=100) 
for i, version in enumerate(wiring_cost_versions): 
    row = i // 2 
    col = i % 2 
    if "dup" in version: 
        axs[row, col].scatter(huge_df[version]/ total_possible_wiring_len_schaeffer *100, huge_df["transitivity"], c=huge_df["dataset"].astype('category').cat.codes, cmap='tab10', alpha=0.3, s=4) 
    else: 
        axs[row, col].scatter(huge_df[version], huge_df["transitivity"], c=huge_df["dataset"].astype('category').cat.codes, cmap='tab10', alpha=0.3, s=4) 
    axs[row, col].set_xlabel(version) 
    axs[row, col].set_ylabel("transitivity") 
    # axs[row, col].set_title(f"{version} vs transitivity") 
plt.tight_layout() 
plt.show()

# plt.scatter(huge_df["wiring_cost"], huge_df["transitivity"], c=huge_df["dataset"].astype('category').cat.codes, cmap='tab10', alpha=0.3, s=4)


# In[ ]:


# create subplots with the different wiring_cost versions: 
wiring_cost_versions = ["wiring_cost", "wiring_cost_dup", "wiring_cost_div_by_matrix", "wiring_cost_dup_div_by_matrix"] 
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=viz.cm_to_inch((18,12)), dpi=100) 
for i, version in enumerate(wiring_cost_versions): 
    row = i // 2 
    col = i % 2 
    for dataset in huge_df["dataset"].unique():
        subset = huge_df[huge_df["dataset"] == dataset]
        print(f"Dataset: {dataset}, Version: {version}.") 
        print(f"    Mean: {subset[version].mean():.4f}, Std: {subset[version].std():.4f}, Number of samples: {len(subset[version])}")
        # set all entries that are nan or inf to -1
        # subset[version] = subset[version].replace([np.inf, -np.inf], np.nan).fillna(-1)
        axs[row, col].hist(subset[version], bins=50, alpha=0.7, density=True, label=dataset)
    # break

    # if "dup" in version: 
    #     axs[row, col].scatter(huge_df[version]/ total_possible_wiring_len_schaeffer *100, huge_df["transitivity"], c=huge_df["dataset"].astype('category').cat.codes, cmap='tab10', alpha=0.3, s=4) 
    # else: 
    #     axs[row, col].scatter(huge_df[version], huge_df["transitivity"], c=huge_df["dataset"].astype('category').cat.codes, cmap='tab10', alpha=0.3, s=4) 
    # axs[row, col].set_xlabel(version) 
    # axs[row, col].set_ylabel("transitivity") 
    # axs[row, col].set_title(f"{version} vs transitivity") 
plt.tight_layout() 
plt.show()

# plt.scatter(huge_df["wiring_cost"], huge_df["transitivity"], c=huge_df["dataset"].astype('category').cat.codes, cmap='tab10', alpha=0.3, s=4)


# In[ ]:


# import seaborn as sns

# # Create pairplot with dataset coloring
# sns.pairplot(huge_df, hue="dataset", plot_kws={'alpha': 0.3, 's': 4}, 
#              diag_kind='kde', corner=True)
# plt.show()


# In[ ]:


metric_cols = [col for col in huge_df.columns if col != "dataset"]
n_cols = len(metric_cols)


# Only plot lower triangle to avoid redundancy
fig, axs = plt.subplots(n_cols, n_cols, figsize=(20, 20))
dataset_codes = huge_df["dataset"].astype('category').cat.codes

for i, col_y in enumerate(metric_cols):
    for j, col_x in enumerate(metric_cols):
        ax = axs[i, j]
        
        if i < j:
            # Upper triangle: hide
            ax.axis('off')
        elif i == j:
            # Diagonal: distribution
            ax.hist(huge_df[col_x], bins=30, alpha=0.7)
            ax.set_ylabel(col_y, fontsize=8)
        else:
            # Lower triangle: scatter
            ax.scatter(huge_df[col_x], huge_df[col_y], 
                      c=dataset_codes, cmap='tab10', alpha=0.3, s=4)
            if j == 0:
                ax.set_ylabel(col_y, fontsize=8)
            if i == n_cols - 1:
                ax.set_xlabel(col_x, fontsize=8)

plt.tight_layout()
plt.show()


# In[ ]:


# create subplots with the different wiring_cost versions: 
wiring_cost_versions = ["wiring_cost_dup","wiring_cost_dup_div_by_matrix"] 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=viz.cm_to_inch((18,12)), dpi=100) 
for i, version in enumerate(wiring_cost_versions): 
    col = i % 2 
    axs[col].scatter(huge_df[version], huge_df["transitivity"], c=huge_df["dataset"].astype('category').cat.codes, 
                     cmap='tab10', alpha=0.3, s=4) 
    # version_2 is equal version, but with "_dup" removed
    version_2 = version.replace("_dup", "")
    axs[col].scatter(huge_df[version_2], huge_df["transitivity"], c=huge_df["dataset"].astype('category').cat.codes, 
                     cmap='Reds', alpha=0.3, s=4) 
    axs[col].set_xlabel(version) 
    axs[col].set_ylabel("transitivity") 
    # axs[col].set_title(f"{version} vs transitivity") 
plt.tight_layout() 
plt.show()

# plt.scatter(huge_df["wiring_cost"], huge_df["transitivity"], c=huge_df["dataset"].astype('category').cat.codes, cmap='tab10', alpha=0.3, s=4)


# In[ ]:


manual_selections = {
    1: 'metric_name_you_choose',  # From: ['avg_edge_distance', 'modularity_dup', 'ollivier_ricci_curvature_orc_frac_neg', 'propagation_efficiency', 'richclub_avg_length', 'transitivity', 'wiring_cost', 'wiring_cost_dup']
    2: 'metric_name_you_choose',  # From: ['avg_clustering', 'effective_dimensionality', 'global_efficiency', 'modularity', 'persistent_homology_ph_h1_entropy', 'persistent_homology_ph_h1_n_features', 'persistent_homology_ph_total_persistence']
    3: 'metric_name_you_choose',  # From: ['algebraic_connectivity_fiedler_value', 'synchronizability_eigenratio_lambda_2']
    4: 'metric_name_you_choose',  # From: ['degree_gini', 'kernel_rank_max', 'spectral_radius']
    5: 'metric_name_you_choose',  # From: ['diffusion_efficiency', 'directed_simplices_count', 'ollivier_ricci_curvature_orc_frac_neg', 'ollivier_ricci_curvature_orc_mean', 'ollivier_ricci_curvature_orc_median', 'omega', 'persistent_homology_ph_h1_entropy', 'wiring_cost']
    6: 'metric_name_you_choose',  # From: ['targeted_attack_robustness_rob_ratio', 'targeted_attack_robustness_rob_targeted_auc']
    7: 'metric_name_you_choose',  # From: ['computational_capacity_cross_capacity_total', 'computational_capacity_cubic_capacity_total', 'computational_capacity_memory_capacity_total', 'computational_capacity_memory_timescale', 'computational_capacity_nonlinear_capacity_total', 'computational_capacity_total_capacity']
    8: 'metric_name_you_choose',  # From: ['mc_14', 'mc_15', 'mc_16', 'mc_17', 'mc_18', 'mc_19', 'mc_20', 'mc_21', 'mc_22', 'mc_23', 'mc_24', 'mc_25', 'mc_26', 'mc_27', 'mc_28', 'mc_29', 'mc_30', 'mc_31', 'mc_32', 'mc_33']
    9: 'metric_name_you_choose',  # From: ['avg_clustering', 'effective_dimensionality', 'global_efficiency', 'modularity_dup', 'nct_control_n_nodes_50_percent', 'nct_control_std', 'omega', 'persistent_homology_ph_h1_entropy', 'propagation_efficiency', 'structural_complexity', 'topological_distance_std', 'transitivity']
    10: 'metric_name_you_choose',  # From: ['effective_dimensionality', 'nct_control_std', 'propagation_efficiency', 'structural_complexity']
    11: 'metric_name_you_choose',  # From: ['avg_clustering', 'transitivity']
    12: 'metric_name_you_choose',  # From: ['computational_capacity_cross_capacity_total', 'computational_capacity_total_capacity']
    13: 'metric_name_you_choose',  # From: ['computational_capacity_state_dimensionality', 'computational_capacity_state_entropy']
    14: 'metric_name_you_choose',  # From: ['diffusion_efficiency', 'modularity', 'modularity_dup']
    15: 'metric_name_you_choose',  # From: ['participation_coefficient_pc_mean', 'participation_coefficient_pc_median']
    16: 'metric_name_you_choose',  # From: ['mc_1', 'mc_2', 'mc_3', 'mc_4']
    17: 'metric_name_you_choose',  # From: ['avg_communicability', 'richclub_n_edges']
}


# In[ ]:


# === STEP 3: CREATE FINAL METRIC LIST ===

def create_final_metric_list(correlation_matrix, groups, manual_selections):
    """
    Create final list of metrics for PCA.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Full correlation matrix
    groups : list of lists
        Conflict groups from selector
    manual_selections : dict
        Manual selections for each group {group_num: metric_name}
    
    Returns:
    --------
    list of selected metric names
    """
    all_metrics = set(correlation_matrix.columns)
    conflicting_metrics = set()
    
    # Collect all metrics in conflict groups
    for group in groups:
        conflicting_metrics.update(group)
    
    # Start with non-conflicting metrics
    selected_metrics = list(all_metrics - conflicting_metrics)
    
    # Add manual selections from each group
    for group_num, metric in manual_selections.items():
        if metric in conflicting_metrics:
            selected_metrics.append(metric)
    
    return sorted(selected_metrics)


# === STEP 4: VALIDATION ===

def validate_selections(correlation_matrix, selected_metrics, threshold=0.95):
    """Verify that selected metrics don't correlate > threshold."""
    selected_corr = correlation_matrix.loc[selected_metrics, selected_metrics]
    
    # Check for high correlations
    high_corrs = []
    for i in range(len(selected_metrics)):
        for j in range(i+1, len(selected_metrics)):
            corr_val = abs(selected_corr.iloc[i, j])
            if corr_val > threshold:
                high_corrs.append({
                    'metric1': selected_metrics[i],
                    'metric2': selected_metrics[j],
                    'correlation': corr_val
                })
    
    if len(high_corrs) == 0:
        print(f"\n✓ SUCCESS! All selected metrics have |correlation| ≤ {threshold}")
        print(f"  Total metrics selected: {len(selected_metrics)}")
    else:
        print(f"\n✗ WARNING! Found {len(high_corrs)} pairs with |correlation| > {threshold}:")
        for pair in high_corrs:
            print(f"  {pair['metric1']} <-> {pair['metric2']}: r={pair['correlation']:.3f}")
    
    return len(high_corrs) == 0


# === HELPER: Save selection results ===

def save_selection_results(selected_metrics, output_path="selected_metrics.txt"):
    """Save the final metric list to a file."""
    with open(output_path, 'w') as f:
        f.write("Selected Metrics for PCA\n")
        f.write("="*50 + "\n\n")
        for i, metric in enumerate(selected_metrics, 1):
            f.write(f"{i}. {metric}\n")
    
    print(f"\nSaved selected metrics to: {output_path}")


# === USAGE EXAMPLE ===

print("\n\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("""
1. Review the conflict groups and network visualization above
2. For each group, choose the most interpretable metric
3. Fill in the manual_selections dictionary
4. Run the validation:


   # After filling in manual_selections:
   final_metrics = create_final_metric_list(correlation_matrix, groups, manual_selections)
   is_valid = validate_selections(correlation_matrix, final_metrics, threshold=0.95)
   
   if is_valid:
       save_selection_results(final_metrics)
       # Use final_metrics for your PCA
       pca_data = huge_df[final_metrics]
""")

# Show the correlation heatmap as before
plt.figure(figsize=(12,12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Metrics")
plt.tight_layout()
plt.show()
# === STEP 3: CREATE FINAL METRIC LIST ===

def create_final_metric_list(correlation_matrix, groups, manual_selections):
    """
    Create final list of metrics for PCA.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Full correlation matrix
    groups : list of lists
        Conflict groups from selector
    manual_selections : dict
        Manual selections for each group {group_num: metric_name}
    
    Returns:
    --------
    list of selected metric names
    """
    all_metrics = set(correlation_matrix.columns)
    conflicting_metrics = set()
    
    # Collect all metrics in conflict groups
    for group in groups:
        conflicting_metrics.update(group)
    
    # Start with non-conflicting metrics
    selected_metrics = list(all_metrics - conflicting_metrics)
    
    # Add manual selections from each group
    for group_num, metric in manual_selections.items():
        if metric in conflicting_metrics:
            selected_metrics.append(metric)
    
    return sorted(selected_metrics)


# === STEP 4: VALIDATION ===

def validate_selections(correlation_matrix, selected_metrics, threshold=0.95):
    """Verify that selected metrics don't correlate > threshold."""
    selected_corr = correlation_matrix.loc[selected_metrics, selected_metrics]
    
    # Check for high correlations
    high_corrs = []
    for i in range(len(selected_metrics)):
        for j in range(i+1, len(selected_metrics)):
            corr_val = abs(selected_corr.iloc[i, j])
            if corr_val > threshold:
                high_corrs.append({
                    'metric1': selected_metrics[i],
                    'metric2': selected_metrics[j],
                    'correlation': corr_val
                })
    
    if len(high_corrs) == 0:
        print(f"\n✓ SUCCESS! All selected metrics have |correlation| ≤ {threshold}")
        print(f"  Total metrics selected: {len(selected_metrics)}")
    else:
        print(f"\n✗ WARNING! Found {len(high_corrs)} pairs with |correlation| > {threshold}:")
        for pair in high_corrs:
            print(f"  {pair['metric1']} <-> {pair['metric2']}: r={pair['correlation']:.3f}")
    
    return len(high_corrs) == 0


# === HELPER: Save selection results ===

def save_selection_results(selected_metrics, output_path="selected_metrics.txt"):
    """Save the final metric list to a file."""
    with open(output_path, 'w') as f:
        f.write("Selected Metrics for PCA\n")
        f.write("="*50 + "\n\n")
        for i, metric in enumerate(selected_metrics, 1):
            f.write(f"{i}. {metric}\n")
    
    print(f"\nSaved selected metrics to: {output_path}")


# === USAGE EXAMPLE ===

print("\n\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("""
1. Review the conflict groups and network visualization above
2. For each group, choose the most interpretable metric
3. Fill in the manual_selections dictionary
4. Run the validation:


   # After filling in manual_selections:
   final_metrics = create_final_metric_list(correlation_matrix, groups, manual_selections)
   is_valid = validate_selections(correlation_matrix, final_metrics, threshold=0.95)
   
   if is_valid:
       save_selection_results(final_metrics)
       # Use final_metrics for your PCA
       pca_data = huge_df[final_metrics]
""")

# Show the correlation heatmap as before
plt.figure(figsize=(12,12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Metrics")
plt.tight_layout()
plt.show()


# In[ ]:


plt.scatter(huge_df["dataset"], huge_df["density"])
plt.xticks(rotation=45)


# In[ ]:


# conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/00_connectomes_50.npy")
conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_50.npy") 
# conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/hcp_schaefer_100_dataset/01_connectomes/00_connectomes_density10.npy")
# conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/kaysons_generated_networks_diffusion/05_mst_animal_0_compared_with_diffusion/generated_networks/net_eta-0.36734688282013_gamma-0.010204084217549_ruleMatchingIndex_id001.npy")
# conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/kaysons_generated_networks_routing/01_connectomes/routing_10.npy")
# conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/lexis_data/10_mst_9_animal_0/generated_networks/net_eta-2.5_gamma-0.100000001490117_ruleMatchingIndex_id001.npy")
# conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/lexis_data/10_mst_9_animal_0/generated_networks/net_eta3.0_gamma1.0_ruleMatchingIndex_id016.npy")
# conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/lexis_data/10_mst_9_animal_0/generated_networks/net_eta3.0_gamma-0.100000001490117_ruleMatchingIndex_id017.npy")
# conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/lexis_data/10_mst_9_animal_0/generated_networks/net_eta-2.5_gamma-0.100000001490117_ruleMatchingIndex_id000.npy") 
conns.shape


# In[ ]:


for c in conns: 
    print(np.mean(c))

