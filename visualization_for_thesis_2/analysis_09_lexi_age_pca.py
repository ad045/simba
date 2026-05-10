# ── Imports ───────────────────────────────────────────────────────────────────
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.interpolate import make_smoothing_spline
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from vizman import viz
from config import COLOR_SCHEME, LABEL_MAP
from analysis_08_cluster_contributions import selected_properties
from analysis_08_cluster_contributions import CLUSTERS_SUBSELECTION, CLUSTERS_OLD # CLUSTERS 

# Choose here if only selected properties (CLUSTERS_SUBSELECTION) or the ones for the apdx (CLUSTERS) 
CHOSEN_CLUSTERS = CLUSTERS_OLD # CLUSTERS_SUBSELECTION # CLUSTERS or CLUSTERS_SUBSELECTION
CATEGORY_ORDER = [k for k in CHOSEN_CLUSTERS.keys()]
print(CATEGORY_ORDER)

from analysis_08_cluster_contributions import CLUSTER_COLORS as CATEGORY_COLOURS
from config import MERGED_PROPERTIES_NAMES as SELECTED_PROPERTIES_NAMES
from config_pca_parameter_selection import remaining_categories

CATEGORY_ORDER = list(CHOSEN_CLUSTERS.keys())
cmap = LinearSegmentedColormap.from_list("custom", ["#E6B212", "#C74800", "#BF013F"])

# ── Paths ─────────────────────────────────────────────────────────────────────
data_folder   = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis_only_consensus_age")
# /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis_only_consensus_age/all_datasets_precise_age_consensus.pkl
output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/01_lexi_ages_consensus")
output_folder.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(data_folder / "all_datasets_precise_age_consensus.pkl", "rb") as f:
    dict_with_all_datasets = pickle.load(f)

ordered_dataset_names = [
    "lexis_data_developing_consensus_per_age_1_year",
    # "lexis_data_young_consensus_per_age_1_year",
    # "lexis_data_aging_consensus_per_age_1_year",
    # "lexis_data_all_consensus_per_age_1_year",
    # "lexis_data_all_consensus_per_age_2_year",
    # "lexis_data_developing_consensus_per_age_2_year",
]

# ── Add ages to datasets ───────────────────────────────────────────────────────
for dataset_name, d in dict_with_all_datasets.items():
    if "lexis_data_" in dataset_name:
        ages_path = (
            f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code"
            f"/data/preprocessed/{dataset_name}/04_further_info/00_ages.npy"
        )
        ages = np.load(ages_path)
        mask = np.zeros(len(ages), dtype=bool)
        mask[d.index] = True
        dict_with_all_datasets[dataset_name]["age"] = ages[mask]

# ── Build huge_df (once, with ages already present) ───────────────────────────
huge_df = pd.DataFrame()
for dataset_name in ordered_dataset_names:
    df = dict_with_all_datasets[dataset_name]
    df["dataset"] = dataset_name
    huge_df = pd.concat([huge_df, df], ignore_index=True)

# ── Select and clean property columns ─────────────────────────────────────────
precise_categories = [col for cat_cols in CHOSEN_CLUSTERS.values() for col in cat_cols]

huge_df_properties = huge_df.drop(columns=["dataset", "age"])
remaining_cols = [c for c in huge_df_properties.columns if c in precise_categories]

to_remove = []  # add column names here to exclude them
remaining_cols = [c for c in remaining_cols if c not in to_remove]
huge_df_properties = huge_df_properties[remaining_cols]

# Replace inf with NaN then drop any column that has NaN
huge_df_properties.replace([np.inf, -np.inf], np.nan, inplace=True)
dropped_cols = huge_df_properties.columns[huge_df_properties.isnull().any()].tolist()
huge_df_properties.dropna(axis=1, inplace=True)
if dropped_cols:
    print(f"Dropped columns with NaN/inf: {dropped_cols}")

# ── Build metadata from CHOSEN_CLUSTERS ──────────────────────────────────────────────
var_to_category = {var: cat for cat, vars_ in CHOSEN_CLUSTERS.items() for var in vars_}
meta = pd.DataFrame({"Category": var_to_category}).rename_axis("var")

missing_vars = [c for c in huge_df_properties.columns if c not in meta.index]
if missing_vars:
    print("Variables in data but missing from metadata:")
    print("\n".join(missing_vars))


def sort_key(col):
    cat = meta.loc[col, "Category"] if col in meta.index else "ZZZ"
    cat_idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (cat_idx, col)


def row_to_y(row_i, n):
    return 1.0 - (row_i + 0.5) / n


def draw_category_brackets(ax, final_order, meta, n, bracket_x=-0.70, serif_w=0.005, label_x=-0.71):
    """Draw coloured bracket + label for each category on the left of an imshow axis."""
    cat_row_spans = {}
    for row_i, c in enumerate(final_order):
        cat = meta.loc[c, "Category"] if c in meta.index else "Other"
        cat_row_spans.setdefault(cat, [row_i, row_i])[1] = row_i

    trans = ax.transAxes
    weird_adapter = 0.00085
    for cat, (r0, r1) in cat_row_spans.items():
        colour = CATEGORY_COLOURS.get(cat, "#444444")
        y_top  = row_to_y(r0, n) + 0.5 / n + weird_adapter
        y_bot  = row_to_y(r1, n) - 0.5 / n - weird_adapter
        y_mid  = (y_top + y_bot) / 2
        kw     = dict(xycoords=trans, textcoords=trans, annotation_clip=False,
                      arrowprops=dict(arrowstyle="-", color=colour, lw=1.5))
        ax.annotate("", xy=(bracket_x, y_bot),             xytext=(bracket_x, y_top),             **kw)
        ax.annotate("", xy=(bracket_x, y_top),             xytext=(bracket_x + serif_w, y_top),   **kw)
        ax.annotate("", xy=(bracket_x, y_bot),             xytext=(bracket_x + serif_w, y_bot),   **kw)
        ax.text(label_x, y_mid, cat, transform=trans,
                ha="right", va="center", fontsize=8, fontweight="bold",
                color=colour, clip_on=False)


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1: Correlation matrix — ordered by category, clustered within
# ═════════════════════════════════════════════════════════════════════════════
cols_in_data       = [c for c in huge_df_properties.columns if c in meta.index]
cols_sorted_by_cat = sorted(cols_in_data, key=sort_key)
corr_full          = huge_df_properties[cols_in_data].corr()

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
final_order.extend([c for c in cols_in_data if c not in final_order])

reduced_corr = corr_full.loc[final_order, final_order]
new_labels   = [SELECTED_PROPERTIES_NAMES.get(c, c) for c in final_order]
n = len(final_order)

fig = plt.figure(figsize=viz.cm_to_inch((18, 12)))
gs  = gridspec.GridSpec(1, 1, left=0.22, right=0.97, top=0.97, bottom=0.12)
ax  = fig.add_subplot(gs[0])

im = ax.imshow(reduced_corr.values, cmap="RdBu_r", vmin=-1, vmax=1,
               aspect="equal", interpolation="nearest")

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

ax.set_yticks(np.arange(n))
ax.set_yticklabels(new_labels, fontsize=8)
ax.set_xticks([])
ax.tick_params(axis="both", which="both", length=0)

draw_category_brackets(ax, final_order, meta, n)

cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label("Pearson r", fontsize=9)
ax.set_title(f"{n} properties")

plt.savefig(output_folder / f"{dataset_name}_corr_matrix_categorised.pdf", bbox_inches="tight", dpi=300)
print(output_folder / f"{dataset_name}_corr_matrix_categorised.pdf")
plt.show()

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2: Correlation matrix — full-matrix hierarchical clustering (automatic)
# ═════════════════════════════════════════════════════════════════════════════
N_CLUSTERS = 6  # tune as needed

corr_clean = corr_full.fillna(0)
dist = np.clip(1 - corr_clean.values, 0, 2)
np.fill_diagonal(dist, 0)
Z_full = linkage(squareform(dist, checks=False), method="average")

auto_order  = [cols_in_data[i] for i in leaves_list(Z_full)]
cluster_ids = fcluster(Z_full, N_CLUSTERS, criterion="maxclust")
orig_to_cluster = dict(zip(cols_in_data, cluster_ids))
cluster_palette = plt.cm.tab10.colors

reduced_corr_auto = corr_full.loc[auto_order, auto_order]
new_labels_auto   = [SELECTED_PROPERTIES_NAMES.get(c, c) for c in auto_order]
n_auto = len(auto_order)

fig = plt.figure(figsize=viz.cm_to_inch((18, 12)))
gs  = gridspec.GridSpec(1, 1, left=0.22, right=0.97, top=0.97, bottom=0.12)
ax  = fig.add_subplot(gs[0])

im = ax.imshow(reduced_corr_auto.values, cmap="RdBu_r", vmin=-1, vmax=1,
               aspect="equal", interpolation="nearest")

# White dividers between auto-clusters
prev_c = orig_to_cluster[auto_order[0]]
for i, col in enumerate(auto_order[1:], start=1):
    if orig_to_cluster[col] != prev_c:
        ax.axhline(i - 0.5, color="white", linewidth=1.2, alpha=0.85)
        ax.axvline(i - 0.5, color="white", linewidth=1.2, alpha=0.85)
    prev_c = orig_to_cluster[col]

ax.set_yticks(np.arange(n_auto))
ax.set_yticklabels(new_labels_auto, fontsize=8)
ax.set_xticks([])
ax.tick_params(axis="both", which="both", length=0)

# Cluster brackets
cluster_row_spans = {}
for row_i, col in enumerate(auto_order):
    cid = orig_to_cluster[col]
    cluster_row_spans.setdefault(cid, [row_i, row_i])[1] = row_i

BRACKET_X = -0.70
trans = ax.transAxes
for cid, (r0, r1) in cluster_row_spans.items():
    colour = cluster_palette[(cid - 1) % len(cluster_palette)]
    y_top  = row_to_y(r0, n_auto) + 0.5 / n_auto
    y_bot  = row_to_y(r1, n_auto) - 0.5 / n_auto
    y_mid  = (y_top + y_bot) / 2
    kw     = dict(xycoords=trans, textcoords=trans, annotation_clip=False,
                  arrowprops=dict(arrowstyle="-", color=colour, lw=1.5))
    ax.annotate("", xy=(BRACKET_X, y_bot),             xytext=(BRACKET_X, y_top),             **kw)
    ax.annotate("", xy=(BRACKET_X, y_top),             xytext=(BRACKET_X + 0.005, y_top),     **kw)
    ax.annotate("", xy=(BRACKET_X, y_bot),             xytext=(BRACKET_X + 0.005, y_bot),     **kw)
    ax.text(BRACKET_X - 0.01, y_mid, f"C{cid}", transform=trans,
            ha="right", va="center", fontsize=8, fontweight="bold",
            color=colour, clip_on=False)

cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label("Pearson r", fontsize=9)
ax.set_title(f"{n_auto} properties — {N_CLUSTERS} clusters (hierarchical)")

plt.savefig(output_folder / f"{dataset_name}_corr_matrix_clustered.pdf", bbox_inches="tight")
print(output_folder / f"{dataset_name}_corr_matrix_clustered.pdf")
plt.show()

for cid in sorted(cluster_row_spans):
    members = [SELECTED_PROPERTIES_NAMES.get(c, c) for c in auto_order if orig_to_cluster[c] == cid]
    print(f"Cluster {cid} ({len(members)} vars): {', '.join(members)}")

# ═════════════════════════════════════════════════════════════════════════════
# PCA
# ═════════════════════════════════════════════════════════════════════════════
scaler     = StandardScaler()
scaled_data = scaler.fit_transform(huge_df_properties)
pca        = PCA(n_components=10)
pca_result = pca.fit_transform(scaled_data)

# ── Loadings bar chart (first 3 PCs, coloured by category) ───────────────────
raw_loadings = pca.components_.T           # shape: (n_vars, n_components)
num_metrics  = raw_loadings.shape[0]
metric_names = huge_df_properties.columns

# Order by PC1 loading value (for consistent display)
sorted_indices  = np.argsort(raw_loadings[:, 0])[::-1]
raw_loadings    = raw_loadings[sorted_indices]
metric_names    = metric_names[sorted_indices]

fig, axs = plt.subplots(nrows=1, ncols=3,
                        figsize=viz.cm_to_inch((40, 12)),
                        sharex=True)

for i in range(3):
    ordered      = np.argsort(raw_loadings[:, i])
    loads_sorted = raw_loadings[ordered]
    names_sorted = metric_names[ordered]
    names_long   = [SELECTED_PROPERTIES_NAMES.get(m, m) for m in names_sorted]
    bar_colours  = [CATEGORY_COLOURS.get(meta.loc[m, "Category"] if m in meta.index else None, "#AAAAAA")
                    for m in names_sorted]

    axs[i].barh(range(num_metrics), loads_sorted[:, i], color=bar_colours)
    axs[i].axvline(0, color="black", linewidth=0.6)
    axs[i].set_title(f"Loadings for PC{i+1}")
    axs[i].set_yticks([])

    PAD = 0.02
    for j, (val, name, colour) in enumerate(zip(loads_sorted[:, i], names_long, bar_colours)):
        if val >= 0:
            axs[i].text(-PAD, j, name, ha="right", va="center", fontsize=6.5, color=colour)
        else:
            axs[i].text(PAD,  j, name, ha="left",  va="center", fontsize=6.5, color=colour)

    axs[i].spines["top"].set_visible(False)
    axs[i].spines["right"].set_visible(False)
    axs[i].spines["left"].set_visible(False)

handles = [mpatches.Patch(color=c, label=cat)
           for cat, c in CATEGORY_COLOURS.items() if cat in meta["Category"].values]
fig.legend(handles=handles, title="Category",
           bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=8)
plt.tight_layout()
plt.savefig(output_folder / f"{dataset_name}_pca_loadings_by_category.pdf", bbox_inches="tight")
print(output_folder / f"{dataset_name}_pca_loadings_by_category.pdf")
plt.show()

# ═════════════════════════════════════════════════════════════════════════════
# Age trajectory setup (shared by 2D and 3D plots)
# ═════════════════════════════════════════════════════════════════════════════
# Scaled loadings (√eigenvalue-weighted) for top-variable identification
scale    = np.sqrt(pca.explained_variance_[:2])
loadings = pd.DataFrame(
    (pca.components_[:2] * scale[:, None]).T,
    index=huge_df_properties.columns,
    columns=["PC1", "PC2"],
)

n_largest = 2
top_pc1   = loadings["PC1"].abs().nlargest(n_largest).index.tolist()
top_pc2   = loadings["PC2"].abs().nlargest(n_largest).index.tolist()
top_vars  = list(dict.fromkeys(top_pc1 + top_pc2))

pca_df        = pd.DataFrame(pca_result[:, :3], columns=["PC1", "PC2", "PC3"])
pca_df["age"] = huge_df["age"].values
age_means     = pca_df.groupby("age")[["PC1", "PC2", "PC3"]].mean().reset_index().sort_values("age")

age_vals  = age_means["age"].values
t         = np.linspace(age_vals.min(), age_vals.max(), 300)
norm      = plt.Normalize(age_vals.min(), age_vals.max())

smooth_pc1 = make_smoothing_spline(age_vals, age_means["PC1"].values, lam=5)(t)
smooth_pc2 = make_smoothing_spline(age_vals, age_means["PC2"].values, lam=5)(t)
smooth_pc3 = make_smoothing_spline(age_vals, age_means["PC3"].values, lam=5)(t)

# ── Scree plot ────────────────────────────────────────────────────────────────
# explained_variance = pca.explained_variance_ratio_
# fig, ax = plt.subplots(figsize=(6, 4))
# sns.barplot(x=[f"PC{i+1}" for i in range(len(explained_variance))],
#             y=explained_variance, ax=ax)
# ax.plot(range(len(explained_variance)), np.cumsum(explained_variance),
#         marker="o", color="red", label="Cumulative")
# ax.set_title("Scree Plot")
# ax.set_ylabel("Explained Variance Ratio")
# ax.set_xlabel("Principal Components")
# ax.legend()
# plt.tight_layout()
# plt.savefig(output_folder / f"{dataset_name}_pca_scree_plot.pdf", dpi=150, bbox_inches="tight")
# print(output_folder / f"{dataset_name}_pca_scree_plot.pdf")
# plt.show()
from utils_viz_for_paper import plot_scree
plot_scree(pca, output_folder / f"{dataset_name}_pca_scree_plot.pdf")


# ── PCA scatter coloured by age (individuals + age-averaged) ─────────────────
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(age_means["PC1"], age_means["PC2"], color="grey", linewidth=0.8, alpha=0.6, zorder=1)
# sc = ax.scatter(pca_df["PC1"], pca_df["PC2"],
#                 c=pca_df["age"], cmap=cmap, norm=norm,
#                 s=5, zorder=2, edgecolors="white", linewidths=0.4)
# ax.scatter(age_means["PC1"], age_means["PC2"],
#            c=age_means["age"], cmap=cmap, norm=norm,
#            s=40, zorder=3, edgecolors="white", linewidths=0.4)
# plt.colorbar(sc, ax=ax, label="Age")
# ax.set_title("PCA — individuals and age averages")
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# plt.tight_layout()
# plt.savefig(output_folder / f"{dataset_name}_pca_age_scatter_individuals.pdf", bbox_inches="tight")
# print(output_folder / f"{dataset_name}_pca_age_scatter_individuals.pdf")
# plt.show()

# ═════════════════════════════════════════════════════════════════════════════
# PLOT: 2D age trajectory (smoothed spline, coloured by age)
# ═════════════════════════════════════════════════════════════════════════════
points   = np.array([smooth_pc1, smooth_pc2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4, alpha=1, zorder=2)
lc.set_array(t)

fig, ax = plt.subplots(figsize=viz.cm_to_inch((12, 9)))
ax.add_collection(lc)
sc = ax.scatter(age_means["PC1"], age_means["PC2"],
                c=age_means["age"], cmap=cmap, norm=norm,
                s=40, zorder=3, edgecolors="white", linewidths=0.4)
ax.autoscale()
fig.colorbar(sc, ax=ax, label="Age")
ax.set_title(dataset_name.replace("_", " ").capitalize())
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
fig.tight_layout()
fig.savefig(output_folder / f"{dataset_name}_pca_age.pdf")
print(output_folder / f"{dataset_name}_pca_age.pdf")
plt.show()

# ═════════════════════════════════════════════════════════════════════════════
# PLOT: 3D age trajectory (PC1–3)
# ═════════════════════════════════════════════════════════════════════════════
points_3d   = np.array([smooth_pc1, smooth_pc2, smooth_pc3]).T.reshape(-1, 1, 3)
segments_3d = np.concatenate([points_3d[:-1], points_3d[1:]], axis=1)
lc3 = Line3DCollection(segments_3d, cmap=cmap, norm=norm, linewidth=3, alpha=0.9, zorder=2)
lc3.set_array(t)

fig = plt.figure(figsize=viz.cm_to_inch((14, 11)))
ax  = fig.add_subplot(111, projection="3d")
ax.add_collection3d(lc3)
sc = ax.scatter(age_means["PC1"], age_means["PC2"], age_means["PC3"],
                c=age_means["age"], cmap=cmap, norm=norm,
                s=40, zorder=3, edgecolors="white", linewidths=0.4, depthshade=True)
fig.colorbar(sc, ax=ax, label="Age", shrink=0.5, pad=0.1)
ax.set_xlabel("PC1", labelpad=6)
ax.set_ylabel("PC2", labelpad=6)
ax.set_zlabel("PC3", labelpad=6)
ax.set_title(f"{dataset_name.replace('_', ' ').capitalize()}: Age trajectory (PC1–3)")
ax.view_init(elev=25, azim=-60)
fig.tight_layout()
fig.savefig(output_folder / f"{dataset_name}_pca_age_3d.pdf")
print(output_folder / f"{dataset_name}_pca_age_3d.pdf")
plt.show()

# ═════════════════════════════════════════════════════════════════════════════
# PLOT: Top loading vectors (fan-label layout)
# ═════════════════════════════════════════════════════════════════════════════
LABEL_RADIUS = 1.2
MIN_ANG_GAP  = 0.8  # radians (~46°)

arrow_info = []
for var in top_vars:
    vx = loadings.loc[var, "PC1"]
    vy = loadings.loc[var, "PC2"]
    colour = "#9B2226" if (var in top_pc1 and var in top_pc2) else \
             "#E63946" if var in top_pc1 else "#457B9D"
    arrow_info.append(dict(var=var, vx=vx, vy=vy, colour=colour,
                           angle=np.arctan2(vy, vx)))

arrow_info.sort(key=lambda d: d["angle"])
label_angles = [d["angle"] for d in arrow_info]

# Push label angles apart until all gaps >= MIN_ANG_GAP
for _ in range(50):
    changed = False
    for i in range(len(label_angles)):
        j   = (i + 1) % len(label_angles)
        gap = label_angles[j] - label_angles[i]
        if i == len(label_angles) - 1:
            gap += 2 * np.pi
        if gap < MIN_ANG_GAP:
            push            = (MIN_ANG_GAP - gap) / 2
            label_angles[i] -= push
            label_angles[j] += push
            changed          = True
    if not changed:
        break

fig, ax = plt.subplots(figsize=viz.cm_to_inch((7, 7)))
ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
ax.axvline(0, color="grey", lw=0.5, alpha=0.5)
ax.scatter([0], [0], color="black", s=15, zorder=6)

label_coords = []
for d, langle in zip(arrow_info, label_angles):
    ax.quiver(0, 0, d["vx"], d["vy"],
              angles="xy", scale_units="xy", scale=1,
              color=d["colour"], width=0.012,
              headwidth=4, headlength=5, headaxislength=4, zorder=5)
    lx = LABEL_RADIUS * np.cos(langle)
    ly = LABEL_RADIUS * np.sin(langle)
    label_coords.append((lx, ly))
    label = SELECTED_PROPERTIES_NAMES.get(d["var"], d["var"]).replace(" ", "\n")
    ax.text(lx, ly, label, fontsize=7, ha="center", va="center", color=d["colour"],
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

all_lx = [lx for lx, _ in label_coords]
all_ly = [ly for _, ly in label_coords]
pad = 0.3
ax.set_xlim(min(min(all_lx), -0.5) - pad, max(max(all_lx), 0.5) + pad)
ax.set_ylim(min(min(all_ly), -0.5) - pad, max(max(all_ly), 0.5) + pad)
ax.set_aspect("equal")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"{dataset_name.replace('_', ' ').capitalize()}: Top PCA loadings")
fig.tight_layout()
fig.savefig(output_folder / f"{dataset_name}_pca_loadings.pdf")
print(output_folder / f"{dataset_name}_pca_loadings.pdf")
plt.show()

# # ═════════════════════════════════════════════════════════════════════════════
# # PLOT: Category aggregate vectors in PCA space
# # ═════════════════════════════════════════════════════════════════════════════
# cat_vectors = {}
# for cat, rep_var in selected_properties.items():
#     if cat not in CATEGORY_COLOURS:
#         continue
#     if rep_var not in loadings.index:
#         print(f"  Warning: representative '{rep_var}' for '{cat}' not in loadings, skipping.")
#         continue
#     cat_vectors[cat] = (loadings.loc[rep_var, "PC1"], loadings.loc[rep_var, "PC2"])

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
# ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
# ax.scatter([0], [0], color="black", s=20, zorder=6)

# for cat, (vx, vy) in cat_vectors.items():
#     colour = CATEGORY_COLOURS[cat]
#     ax.quiver(0, 0, vx, vy, angles="xy", scale_units="xy", scale=1,
#               color=colour, width=0.012, headwidth=4, headlength=5, headaxislength=4, zorder=5)
#     ax.text(vx * 1.15, vy * 1.15, cat, fontsize=8, ha="center", va="center",
#             color=colour, fontweight="bold")

# max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) * 1.4
# ax.set_xlim(-max_val, max_val)
# ax.set_ylim(-max_val, max_val)
# ax.set_aspect("equal")
# ax.set_xlabel("PC1", fontsize=9)
# ax.set_ylabel("PC2", fontsize=9)
# ax.set_title("Category vectors in PCA space", fontsize=9)
# for spine in ax.spines.values():
#     spine.set_visible(False)
# ax.tick_params(labelsize=7)
# plt.tight_layout()
# plt.savefig(output_folder / f"{dataset_name}_pca_category_vectors.pdf")
# print(output_folder / f"{dataset_name}_pca_category_vectors.pdf")
# plt.show()



# ═════════════════════════════════════════════════════════════════════════════
# PLOT: Category aggregate vectors in PCA space
# ═════════════════════════════════════════════════════════════════════════════
cat_vectors = {}
for cat, rep_var in selected_properties.items():
    if cat not in CATEGORY_COLOURS:
        continue
    if rep_var not in loadings.index:
        print(f"  Warning: representative '{rep_var}' for '{cat}' not in loadings, skipping.")
        continue
    cat_vectors[cat] = (loadings.loc[rep_var, "PC1"], loadings.loc[rep_var, "PC2"])

fig, ax = plt.subplots(figsize=(5, 5))
ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
ax.scatter([0], [0], color="black", s=20, zorder=6)

for cat, (vx, vy) in cat_vectors.items():
    colour = CATEGORY_COLOURS[cat]
    ax.quiver(0, 0, vx, vy, angles="xy", scale_units="xy", scale=1,
              color=colour, width=0.012, headwidth=4, headlength=5, headaxislength=4, zorder=5)
    ax.text(vx * 1.15, vy * 1.15, cat,
            # fontsize=8, 
            ha="center", va="center",
            color=colour, fontweight="bold")

max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) * 1.4
ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, max_val)
ax.set_aspect("equal")
ax.set_xlabel("PC1") #, fontsize=9)
ax.set_ylabel("PC2") # , fontsize=9)
ax.set_title("Category vectors in PCA space", fontsize=9)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params() # labelsize=7)
plt.tight_layout()
plt.savefig(output_folder / f"{dataset_name}_pca_category_vectors.pdf")
print(output_folder / f"{dataset_name}_pca_category_vectors.pdf")
plt.show()





###### PLOT: ALTERNATIVE CATEGORY ARROWS #########

# from utils_viz_for_paper import compute_biplot_loadings, plot_category_quivers

# loading_and_cat_df = compute_biplot_loadings(pca, used_features)
# plot_category_quivers(loading_and_cat_df, meta, output_folder / f"pca_quivers{suffix}_{appendix}_with_labels.pdf", with_labels=True)

# fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,6)), 
#                         dpi=150)

# ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
# ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
# ax.scatter([0], [0], color="black", s=20, zorder=6)

# for cat, (vx, vy) in cat_vectors.items():
#     colour = CATEGORY_COLOURS[cat]
#     ax.quiver(
#         0, 0, vx * 1.4, vy * 1.4,   
#         angles="xy", scale_units="xy", scale=1,
#         color=colour, 
#         width=0.012, # 8, # 012,
#         headwidth=4,
#         headlength=5, headaxislength=5,
#         zorder=5,
#     )
#     # if with_labels:
#     nudge = 1.3 # 1.15
#     ax.text(vx * nudge, vy * nudge, cat,
#             ha="center", va="center", color=colour)

# # max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) # * 1.4
# # maximum value in any direction (PC1 or PC2)
# max_val = max([max(abs(vx), abs(vy)) for vx, vy in cat_vectors.values()]) # * 1.4
# # # ax.set_xlim(-max_val, max_val)
# # # ax.set_ylim(-max_val, max_val)
# # ax.set_xlim(-max_val, max_val)
# # ax.set_ylim(-max_val, max_val)
# # if equal_axes:
# #     ax.set_aspect("equal")
# # else:
# #     plt.tight_layout()  # only apply when axes are free to scale

# # if equal_axes:
# #     ax.set_xlim(-max_val, max_val)
# #     ax.set_ylim(-max_val, max_val)
# ax.set_aspect("equal", adjustable="box")
# # else:
# #     ax.set_xlim(-max_val, max_val)
# #     ax.set_ylim(-max_val, max_val)

# max_val = 5
# ax.set_xlim(-max_val, max_val)
# ax.set_ylim(-max_val, max_val)
# ticks = [-4, -2, 0, 2, 4]
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)

# # ax.set_aspect("equal")
# # ax.set_xlabel("PC1", fontsize=9)
# # ax.set_ylabel("PC2", fontsize=9)
# # ax.set_title("Category vectors in PCA space\n(weighted by loading magnitude)", fontsize=9)
# print("Category vectors in PCA space\n(weighted by loading magnitude)")

# for spine in ax.spines.values():
#     spine.set_visible(False)

# # plt.tight_layout()
# # output_path = OUTPUT_FOLDER / output_filename
# plt.savefig(output_folder / f"{dataset_name}_pca_category_vectors_alterative_layout.pdf", bbox_inches="tight")
# # plt.close()
# plt.show()
# print(f"Saved category quivers to {output_folder / f"{dataset_name}_pca_category_vectors_alterative_layout.pdf"}")





# ═════════════════════════════════════════════════════════════════════════════
# PLOT: Per-category vector composition (chain addition of individual variables)
# ═════════════════════════════════════════════════════════════════════════════
for target_cat in CATEGORY_COLOURS:
    cat_vars = [c for c in loadings.index
                if c in meta.index and meta.loc[c, "Category"] == target_cat]
    if not cat_vars:
        continue

    colour    = CATEGORY_COLOURS[target_cat]
    cat_loads = loadings.loc[cat_vars]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
    ax.scatter([0], [0], color="black", s=20, zorder=6)

    cx, cy = 0.0, 0.0
    all_x, all_y = [0.0], [0.0]
    for var_name, row in cat_loads.iterrows():
        vx, vy = row["PC1"], row["PC2"]
        ax.quiver(cx, cy, vx, vy,
                  angles="xy", scale_units="xy", scale=1,
                  color=colour, width=0.008, # 6, #8, 
                  alpha=0.6,
                  headwidth=4, headlength=5, headaxislength=4, zorder=4)
        mid_x = cx + vx / 2 + (0.02 * np.sign(vx) if vx != 0 else 0.02)
        mid_y = cy + vy / 2 + (0.02 * np.sign(vy) if vy != 0 else 0.02)
        nice_name = SELECTED_PROPERTIES_NAMES.get(var_name, var_name)
        ax.text(mid_x, mid_y, nice_name, 
                fontsize=6, 
                ha="center", va="center",
                color=colour, alpha=0.9,
                # bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5)
                )
        cx += vx
        cy += vy
        all_x.append(cx)
        all_y.append(cy)

    # Final sum vector from origin
    ax.quiver(0, 0, cx, cy, angles="xy", scale_units="xy", scale=1,
              color="black", width=0.012, headwidth=4, headlength=5, headaxislength=4, zorder=5)
    nudge_x = 0.05 * np.sign(cx) if cx != 0 else 0.05
    nudge_y = 0.05 * np.sign(cy) if cy != 0 else 0.05
    ax.text(cx + nudge_x, cy + nudge_y, f"Total {target_cat}",
            # fontsize=10, 
            ha="center", va="center", color="black", fontweight="bold")

    # max_lim = max(max(all_x), abs(min(all_x)), max(all_y), abs(min(all_y)), 0.1) * 1.3
    # ax.set_xlim(-max_lim, max_lim)
    # ax.set_ylim(-max_lim, max_lim)
    buffer = 0.1
    ax.set_xlim(min(all_x)-buffer, max(all_x)+buffer)
    ax.set_ylim(min(all_y)-buffer, max(all_y)+buffer)
    ax.set_aspect("equal")
    ax.set_xlabel("PC1") # , fontsize=9)
    ax.set_ylabel("PC2") # , fontsize=9)
    # ax.set_title(f"Composition of '{target_cat}' in PCA space", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(output_folder / f"{dataset_name}_pca_category_composition_{target_cat}.pdf", dpi=150, bbox_inches="tight")
    print(output_folder / f"{dataset_name}_pca_category_composition_{target_cat}.pdf")
    plt.show()



