# %% [markdown]
# # 01 · Trade-off Analysis
# Correlation matrices → PCA → spider profiles → taxonomy analysis.
# Run notebook_00 first to generate `all_datasets_precise_categories.pkl`.

# %% ── Imports ──────────────────────────────────────────────────────────────
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from scipy.stats import f_oneway
from vizman import viz

from config import (
    COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES,
    CATEGORY_COLOURS, CATEGORY_ORDER,
)
from utils import get_combined_colors
from utils_permanova import run_permanova, sig_stars, print_latex_permanova_table
from viz_utils import (
    sort_cols_by_category, plot_corr_matrix,
    make_spider, make_spider_legend,
    find_corners, get_neighbours,
    build_cat_scaled, build_cat_quiver_vectors, plot_quiver,
    permanova,
)

# %load_ext autoreload
# %autoreload 2

OUTPUT = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis")
OUTPUT.mkdir(exist_ok=True)

# %% ── Config ────────────────────────────────────────────────────────────────
GNM_DATASET = "hcp_schaefer_100_dataset_gnm"

ORDERED_DATASETS = [
    "hcp_schaefer_100_dataset_gnm",
    "suarez_MaMI_dataset",
    "lexis_data_developing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
]

SPECIAL_DATASETS = [
    "kaysons_generated_networks_routing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
]

# Spider plot y-range and ring positions (shared across all spider plots)
SPIDER_YLIM   = (-4, 7)
SPIDER_RINGS  = [-2, 2, 4, 6]

# Axis pair arcs on the legend spider
AXIS_GROUPS = [
    ("Memory",            "Computational\ncapacity", "lightgray"),
    ("Robustness",        "Robustness\n(lambda_2)",  "lightgray"),
    ("Metastability\nReservoir Diversity", "Dynamics", "lightgray"),
]

# Features shown on each spider axis
SPIDER_PROPERTIES = {
    "Integration":                   "global_efficiency",
    "Segregation":                   "modularity",
    "Wiring\neconomy":               "proportion_long_range_connections_0.3956",
    "Robustness":                    "targeted_attack_robustness_rob_targeted_auc",
    "Robustness\n(lambda_2)":        "algebraic_connectivity_fiedler_value",
    "Dynamics":                      "spectral_radius",
    "Memory":                        "mc_input_scaling_0_1_mc_mean",
    "Computational\ncapacity":       "mc_nonlinear_input_scaling_0_1_mc_mean",
    "Metastability\nReservoir Diversity": "repertoire_sweep_weighted_by_distances_diversity_critical",
}

# Properties to flip so all intra-cluster correlations are positive (cosmetic)
PROPERTIES_TO_FLIP = [
    "rich_club_coefficient_rc_k_at_max",
    "directed_simplices_count",
    "participation_coefficient_pc_frac_connector",
    "community_synchronization_vulnerability_n_communities",
    "participation_coefficient_n_communities",
    "repertoire_sweep_weighted_by_distances_diversity_critical",
]

# Reduced feature set used in the main analysis (after collinearity pruning)
PRECISE_FEATURES = {
    "Fundamental Topology": [
        "modularity", "omega",
        "directed_simplices_count", "directed_simplices_max_size",
    ],
    "Paths, Efficiency & Communication": [
        "global_efficiency", "diffusion_efficiency", "propagation_efficiency",
    ],
    "Spatial Embedding & Wiring Cost": [
        "wiring_cost", "proportion_long_range_connections_0.3956",
    ],
    "Rich-Club Organization": [
        "rich_club_coefficient_rc_k_at_max",
    ],
    "Spectral Properties, Algebraic Connectivity & Fiedler Analysis": [
        "spectral_radius", "spectral_gap",
        "synchronizability_eigenratio_eigenratio",
        "algebraic_connectivity_fiedler_value",
        "algebraic_connectivity_laplacian_spectral_gap",
        "departure_from_normality_schur",
    ],
    "Metastability & Kuramoto Synchronization": [
        "repertoire_sweep_weighted_by_distances_T_critical",
        "repertoire_sweep_weighted_by_distances_size_critical",
        "repertoire_sweep_weighted_by_distances_diversity_critical",
    ],
    "Participation Coefficient (Louvain)": [
        "participation_coefficient_n_communities",
        "participation_coefficient_pc_std",
        "participation_coefficient_pc_frac_connector",
    ],
    "Ollivier-Ricci Curvature": [
        "ollivier_ricci_curvature_orc_min",
        "ollivier_ricci_curvature_orc_skewness",
    ],
    "Targeted Attack Robustness": [
        "targeted_attack_robustness_rob_targeted_auc",
        "targeted_attack_robustness_rob_ratio",
        "community_synchronization_vulnerability_n_communities",
    ],
    "Network Control Theory - Average Controllability": [
        "nct_control_std",
    ],
    "Memory Capacity": [
        "mc_input_scaling_0_1_mc_mean",
        "mc_nonlinear_input_scaling_0_1_mc_mean",
    ],
}

PRECISE_FEATURE_LIST: list[str] = [
    col for cols in PRECISE_FEATURES.values() for col in cols
]

# %% ── Load data ─────────────────────────────────────────────────────────────
with open(OUTPUT / "all_datasets_precise_categories.pkl", "rb") as f:
    dict_with_all_datasets = pickle.load(f)

# %% ── Build combined dataframe ──────────────────────────────────────────────
frames = []
for ds in ORDERED_DATASETS:
    df = dict_with_all_datasets[ds].copy()
    df["dataset"] = ds
    df["color_dataset"] = (
        get_combined_colors(df) if ds == GNM_DATASET else COLOR_SCHEME[ds]
    )
    frames.append(df)

huge_df = pd.concat(frames, ignore_index=True)
huge_df_props = huge_df.drop(columns=["dataset", "color_dataset"])

# Restrict to the agreed reduced feature set
huge_df_props = huge_df_props[[c for c in PRECISE_FEATURE_LIST if c in huge_df_props.columns]]

print(f"Combined dataframe: {huge_df_props.shape}")

# %% ── Load metadata (Excel) ─────────────────────────────────────────────────
EXCEL_PATH = "/Users/adrian/Desktop/network_properties_full_excel.xlsx"

meta_raw = pd.read_excel(EXCEL_PATH, sheet_name=0)
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
# Restrict meta to features actually present
meta = meta.loc[[c for c in huge_df_props.columns if c in meta.index]]

# %% ── Section 1: Correlation matrix ─────────────────────────────────────────
cols_in_data = [c for c in huge_df_props.columns if c in meta.index]
corr_full    = huge_df_props[cols_in_data].corr()

final_order = sort_cols_by_category(cols_in_data, corr_full, meta, CATEGORY_ORDER)
new_labels  = [PROPERTY_NAMES.get(c, c) for c in final_order]

plot_corr_matrix(
    huge_df_props, final_order, meta, CATEGORY_COLOURS, PROPERTY_NAMES,
    title=f"{len(final_order)} properties",
    savepath=OUTPUT / "corr_matrix_categorised.pdf",
    viz=viz,
)
plt.show()

# %% ── Sign-flip selected properties (cosmetic) ──────────────────────────────
for prop in PROPERTIES_TO_FLIP:
    if prop in huge_df_props.columns:
        huge_df_props[prop] = -huge_df_props[prop]
        print(f"Flipped: {prop}")

# Recompute order and correlation after flip
cols_in_data = [c for c in huge_df_props.columns if c in meta.index]
corr_full    = huge_df_props[cols_in_data].corr()
final_order  = sort_cols_by_category(cols_in_data, corr_full, meta, CATEGORY_ORDER)

plot_corr_matrix(
    huge_df_props, final_order, meta, CATEGORY_COLOURS, PROPERTY_NAMES,
    title=f"{len(final_order)} properties (sign-flipped)",
    savepath=OUTPUT / "corr_matrix_categorised_flipped.pdf",
    viz=viz,
)
plt.show()

# %% ── Section 2: PCA ─────────────────────────────────────────────────────────
# Drop any remaining NaN / Inf columns before fitting
huge_df_props.replace([np.inf, -np.inf], np.nan, inplace=True)
huge_df_props.dropna(axis=1, inplace=True)

scaler     = StandardScaler()
scaled     = scaler.fit_transform(huge_df_props)
pca        = PCA(n_components=10)
pca_result = pca.fit_transform(scaled)

feature_cols = list(huge_df_props.columns)
print("Explained variance:", np.round(pca.explained_variance_ratio_, 3))

# %% Scree plot
fig, ax = plt.subplots(figsize=viz.cm_to_inch((6, 6)), dpi=100)
ax.bar(range(1, 11), pca.explained_variance_ratio_, color="steelblue", alpha=0.6)
ax.plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_),
        marker="o", color="red", label="Cumulative")
ax.set_xlabel("PC")
ax.set_ylabel("Explained variance ratio")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "scree_plot.pdf", dpi=150, bbox_inches="tight")
plt.show()

# %% ── PCA loadings per category (quiver plot) ───────────────────────────────
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=feature_cols,
)
loadings_df["category"] = [
    meta.loc[c, "Category"] if c in meta.index else "Unknown"
    for c in feature_cols
]

cat_vecs = build_cat_quiver_vectors(loadings_df, feature_cols, meta, CATEGORY_COLOURS)

plot_quiver(cat_vecs, CATEGORY_COLOURS, with_labels=False,
            savepath=OUTPUT / "pca_quiver_no_labels.pdf", viz=viz)
plt.show()

plot_quiver(cat_vecs, CATEGORY_COLOURS, with_labels=True,
            figsize_cm=(12, 12),
            savepath=OUTPUT / "pca_quiver_labels.pdf", viz=viz)
plt.show()

# %% ── Section 3: Corner points + special networks ───────────────────────────

# --- 3a. Corners in the full dataset
corners_full = find_corners(pca_result)

# --- 3b. Corners restricted to biological networks (MaMI + developing cohort)
bio_mask = (
    (huge_df["dataset"] == "lexis_data_developing") |
    (huge_df["dataset"] == "suarez_MaMI_dataset")
).values
bio_orig_idx  = np.where(bio_mask)[0]
corners_bio_filtered = find_corners(pca_result[bio_mask])
corners_bio = {name: int(bio_orig_idx[fi]) for name, fi in corners_bio_filtered.items()}

# --- 3c. Single-network special datasets
special_idx = {
    ds: int(huge_df.index[huge_df["dataset"] == ds][0])
    for ds in SPECIAL_DATASETS
}

print("Corners (full):", {k: huge_df["dataset"].iloc[v] for k, v in corners_full.items()})
print("Corners (bio):", {k: huge_df["dataset"].iloc[v] for k, v in corners_bio.items()})

# %% ── PCA scatter with highlighted points ───────────────────────────────────

highlighted = {}
for corner_name, idx in corners_full.items():
    highlighted[f"Full · {corner_name}"] = {"idx": idx, "source": "full"}
for corner_name, idx in corners_bio.items():
    highlighted[f"Bio · {corner_name}"] = {"idx": idx, "source": "bio"}
for ds, idx in special_idx.items():
    highlighted[f"Special · {LABEL_MAP[ds]}"] = {"idx": idx, "source": "special"}

fig, ax = plt.subplots(figsize=viz.cm_to_inch((18, 12)), dpi=150)
for ds in huge_df["dataset"].unique():
    m = (huge_df["dataset"] == ds).values
    ax.scatter(
        pca_result[m, 0], pca_result[m, 1],
        c=huge_df[m]["color_dataset"],
        label=LABEL_MAP.get(ds, ds),
        alpha=0.3 if ds == GNM_DATASET else 0.8,
        s=8   if ds == GNM_DATASET else 12,
        edgecolors="none"  if ds == GNM_DATASET else "black",
        linewidths=0       if ds == GNM_DATASET else 0.5,
    )

for label, info in highlighted.items():
    idx = info["idx"]
    lw  = 0.5 if info["source"] == "full" else 1.4
    ax.scatter(pca_result[idx, 0], pca_result[idx, 1],
               color=huge_df["color_dataset"].iloc[idx],
               s=50, zorder=6, edgecolors="black", linewidths=lw)

ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(OUTPUT / "pca_highlighted.pdf", dpi=150, bbox_inches="tight")
plt.show()

# %% ── Section 4: Category-level spider plots ─────────────────────────────────
cat_order_present = [c for c in CATEGORY_ORDER if c in meta["Category"].values]

cat_scaled = build_cat_scaled(scaled, feature_cols, cat_order_present, meta)

# Determine shared y-range from all networks we'll plot
all_plot_idx = (
    list(corners_full.values()) +
    list(corners_bio.values()) +
    list(special_idx.values())
)
all_vals = np.concatenate([cat_scaled[i] for i in all_plot_idx])
dyn_ylim   = (float(np.floor(np.nanmin(all_vals))), min(float(np.ceil(np.nanmax(all_vals))), SPIDER_YLIM[1]))
dyn_rings  = [r for r in SPIDER_RINGS if r <= dyn_ylim[1]]

angles_spider = np.linspace(0, 2 * np.pi, len(cat_order_present), endpoint=False).tolist()

# --- Build panels
panels = []

for corner_name, idx in corners_full.items():
    nb = get_neighbours(idx, pca_result, n=5,
                        exclude_mask=(huge_df["dataset"] == GNM_DATASET).values)
    panels.append({
        "title": f"Full · {corner_name}",
        "vals":  cat_scaled[idx],
        "color": huge_df["color_dataset"].iloc[idx],
        "nb":    [cat_scaled[i] for i in nb],
        "fname": f"spider_full_{corner_name}.pdf",
    })

for corner_name, idx in corners_bio.items():
    nb = get_neighbours(idx, pca_result, n=5,
                        exclude_mask=(huge_df["dataset"] == GNM_DATASET).values)
    panels.append({
        "title": f"Bio · {corner_name}",
        "vals":  cat_scaled[idx],
        "color": COLOR_SCHEME[huge_df["dataset"].iloc[idx]],
        "nb":    [cat_scaled[i] for i in nb],
        "fname": f"spider_bio_{corner_name}.pdf",
    })

for ds, idx in special_idx.items():
    nb = get_neighbours(idx, pca_result, n=5)
    panels.append({
        "title": f"Special · {LABEL_MAP[ds]}",
        "vals":  cat_scaled[idx],
        "color": COLOR_SCHEME[ds],
        "nb":    [cat_scaled[i] for i in nb],
        "fname": f"spider_special_{LABEL_MAP[ds].lower()}.pdf",
    })

# --- Render
for p in panels:
    make_spider(
        values=p["vals"], label=p["title"], color=p["color"],
        feature_labels=cat_order_present, title=p["title"],
        ylim=dyn_ylim, filepath=OUTPUT / p["fname"],
        neighbour_vals_list=p["nb"], ring_vals=dyn_rings,
        viz=viz,
    )

# --- Legend (axis labels only)
make_spider_legend(
    feature_labels=cat_order_present, angles=angles_spider,
    ylim=dyn_ylim, ring_vals=dyn_rings,
    filepath=OUTPUT / "spider_legend.pdf",
    label_color="gray",
    category_colours=CATEGORY_COLOURS,
    axis_groups=AXIS_GROUPS,
    viz=viz,
)

# %% ── Section 5: PCA coloured by individual metrics ─────────────────────────
METRICS_TO_COLOUR = [
    "mc_input_scaling_0_1_mc_mean",
    "mc_nonlinear_input_scaling_0_1_mc_mean",
    "spectral_radius",
    "global_efficiency", "modularity",
    "proportion_long_range_connections_0.3956",
    "targeted_attack_robustness_rob_targeted_auc",
    "algebraic_connectivity_fiedler_value",
    "repertoire_sweep_weighted_by_distances_diversity_critical",
]

CMAP = plt.get_cmap("managua")

for metric in METRICS_TO_COLOUR:
    if metric not in huge_df.columns:
        continue
    vmin, vmax = huge_df[metric].min(), huge_df[metric].max()
    norm = plt.Normalize(vmin, vmax)

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6, 6)), dpi=150)
    for ds in huge_df["dataset"].unique():
        m = (huge_df["dataset"] == ds).values
        ax.scatter(
            pca_result[m, 0], pca_result[m, 1],
            c=huge_df[m][metric], cmap=CMAP, norm=norm,
            alpha=0.3 if ds == GNM_DATASET else 0.8,
            s=2   if ds == GNM_DATASET else 3,
            edgecolors="none"  if ds == GNM_DATASET else "black",
            linewidths=0       if ds == GNM_DATASET else 0.2,
        )

    plt.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm), ax=ax,
                 label=PROPERTY_NAMES.get(metric, metric), orientation="horizontal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=0.01)
    plt.savefig(OUTPUT / f"pca_coloured_{metric}.pdf", dpi=150)
    plt.show()

# %% ── Section 6: Taxonomy analysis (MaMI dataset) ───────────────────────────
TAXONOMY      = "order"
GROUPS        = ["Primates", "Rodentia", "Carnivora", "Chiroptera", "Cetartiodactyla"]
COLOR_MAP_TAX = {
    "Primates":       "#AD5C4D",
    "Rodentia":       "#AD8B4E",
    "Carnivora":      "#4D9EAD",
    "Cetartiodactyla":"#4C6FAD",
    "Chiroptera":     "#6B4DAD",
}

info_mami = pd.read_csv(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "data/preprocessed/suarez_MaMI_dataset/04_further_info/"
    "names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv"
)

mami_rows = huge_df[huge_df["dataset"] == "suarez_MaMI_dataset"].reset_index(drop=True)
df_mami   = pd.concat([mami_rows, info_mami], axis=1)
df_mami["tax_color"] = df_mami[TAXONOMY].map(COLOR_MAP_TAX)

tax_mask = df_mami[TAXONOMY].isin(GROUPS).values

# Aligned PCA coordinates and labels for the four groups
X_full = pca_result[(huge_df["dataset"] == "suarez_MaMI_dataset").values][tax_mask, :2]
labels = df_mami[tax_mask][TAXONOMY].values

# --- Pairwise PERMANOVA
results = {}
for a, b in combinations(GROUPS, 2):
    m = np.isin(labels, [a, b])
    X_, y_ = X_full[m], labels[m]
    if len(np.unique(y_)) < 2:
        continue
    F, p = run_permanova(X_, y_)
    xa, xb = X_[y_ == a, 0], X_[y_ == b, 0]
    d = abs(xa.mean() - xb.mean()) / np.sqrt((xa.var() + xb.var()) / 2 + 1e-9)
    results[(a, b)] = dict(F=F, p=p, d=d)

print_latex_permanova_table(results, GROUPS, sig_stars)

# --- Per-group KDE panels
fig, axes = plt.subplots(1, len(GROUPS), figsize=viz.cm_to_inch((22, 6)),
                          gridspec_kw={"width_ratios": [1] * len(GROUPS)},
                          dpi=120, sharex=True, sharey=True)

for ax, focal in zip(axes, GROUPS):
    m_focal = labels == focal
    color   = COLOR_MAP_TAX.get(focal, "gray")

    sns.kdeplot(x=X_full[:, 0], y=X_full[:, 1], ax=ax,
                fill=True, color="lightgray", alpha=0.3)
    if m_focal.sum() > 3:
        sns.kdeplot(x=X_full[m_focal, 0], y=X_full[m_focal, 1],
                    ax=ax, fill=False, color=color, alpha=1)
    ax.scatter(X_full[m_focal, 0], X_full[m_focal, 1],
               color=color, s=15, edgecolors="black", linewidths=0.4, zorder=5)

    ax.set_title(f"{focal}\n(n={m_focal.sum()})", fontsize=8, color=color)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2" if ax is axes[0] else "")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT / "pca_kde_orders.pdf", bbox_inches="tight")
plt.show()

# --- Within-MaMI PCA (optional cross-check)
scaled_mami = StandardScaler().fit_transform(
    huge_df_props[(huge_df["dataset"] == "suarez_MaMI_dataset").values]
)
pca_mami        = PCA(n_components=10).fit_transform(scaled_mami)
X_mami_groups   = pca_mami[tax_mask, :2]

fig, ax = plt.subplots(figsize=viz.cm_to_inch((9, 9)), dpi=150)
ax.scatter(pca_mami[:, 0], pca_mami[:, 1], c="lightgray", s=5, zorder=1)
ax.scatter(X_mami_groups[:, 0], X_mami_groups[:, 1],
           c=df_mami[tax_mask]["tax_color"].values,
           s=30, edgecolors="black", linewidths=0.5, zorder=2)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(OUTPUT / "pca_mami_only.pdf", dpi=150, bbox_inches="tight")
plt.show()
