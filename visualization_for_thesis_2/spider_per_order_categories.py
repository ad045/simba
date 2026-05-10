"""
spider_per_order_categories.py
------------------------------
Same goal as spider_per_order.py, but spider axes = broad functional
categories (Integration, Segregation, …) assembled from the Excel metadata,
each axis being the mean z-score of all features assigned to that category.

Pipeline mirrors notebook_01_analysis.py §§ 1-4:
  Excel meta → huge_df (all datasets) → sign-flip → StandardScaler →
  build_cat_scaled → MaMI rows → group by order → spider per order.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from vizman import viz

from config import COLOR_SCHEME, CATEGORY_COLOURS, CATEGORY_ORDER
from utils import get_combined_colors
from viz_utils import build_cat_scaled, make_spider, make_spider_legend

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "output/00_trade_off_analysis"
)
OUTPUT.mkdir(exist_ok=True)

DATA_PKL   = OUTPUT / "all_datasets_precise_categories.pkl"
EXCEL_PATH = "/Users/adrian/Desktop/network_properties_full_excel.xlsx"

INFO_CSV = (
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "data/preprocessed/suarez_MaMI_dataset/04_further_info/"
    "names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv"
)

MIN_ANIMALS = 10
TAXONOMY    = "order"
GNM_DATASET = "hcp_schaefer_100_dataset_gnm"

ORDERED_DATASETS = [
    "hcp_schaefer_100_dataset_gnm",
    "suarez_MaMI_dataset",
    "lexis_data_developing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
]

# Features that are sign-flipped before scaling (same as notebook_01)
PROPERTIES_TO_FLIP = [
    "rich_club_coefficient_rc_k_at_max",
    "directed_simplices_count",
    "participation_coefficient_pc_frac_connector",
    "community_synchronization_vulnerability_n_communities",
    "participation_coefficient_n_communities",
    "repertoire_sweep_weighted_by_distances_diversity_critical",
]

# Reduced feature set (same as notebook_01)
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
PRECISE_FEATURE_LIST = [col for cols in PRECISE_FEATURES.values() for col in cols]

SPIDER_RINGS = [-2, 0, 2, 4]

COLOR_MAP_ORDERS = {
    "Primates":        "#AD5C4D",
    "Rodentia":        "#AD8B4E",
    "Carnivora":       "#4D9EAD",
    "Cetartiodactyla": "#4C6FAD",
    "Chiroptera":      "#6B4DAD",
}

# ── 1. Load dataset pickle ────────────────────────────────────────────────────

with open(DATA_PKL, "rb") as f:
    dict_with_all_datasets = pickle.load(f)

frames = []
for ds in ORDERED_DATASETS:
    df = dict_with_all_datasets[ds].copy()
    df["dataset"] = ds
    df["color_dataset"] = (
        get_combined_colors(df) if ds == GNM_DATASET else COLOR_SCHEME[ds]
    )
    frames.append(df)

huge_df       = pd.concat(frames, ignore_index=True)
huge_df_props = huge_df.drop(columns=["dataset", "color_dataset"])
huge_df_props = huge_df_props[[c for c in PRECISE_FEATURE_LIST if c in huge_df_props.columns]]

print(f"Combined dataframe: {huge_df_props.shape}")

# ── 2. Load Excel metadata → category assignments ─────────────────────────────

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
meta = meta.loc[[c for c in huge_df_props.columns if c in meta.index]]

print(f"Categories in metadata: {sorted(meta['Category'].dropna().unique())}")

# ── 3. Sign-flip + scale ──────────────────────────────────────────────────────

for prop in PROPERTIES_TO_FLIP:
    if prop in huge_df_props.columns:
        huge_df_props[prop] = -huge_df_props[prop]

huge_df_props.replace([np.inf, -np.inf], np.nan, inplace=True)
huge_df_props.dropna(axis=1, inplace=True)

feature_cols = list(huge_df_props.columns)
scaled       = StandardScaler().fit_transform(huge_df_props)

# ── 4. Build category-level z-score matrix (n_networks × n_categories) ────────

cat_order_present = [c for c in CATEGORY_ORDER if c in meta["Category"].values]
cat_scaled        = build_cat_scaled(scaled, feature_cols, cat_order_present, meta)

print(f"Spider axes (categories): {cat_order_present}")

# ── 5. Join MaMI rows with taxonomy ──────────────────────────────────────────

mami_mask = (huge_df["dataset"] == "suarez_MaMI_dataset").values
info_mami = pd.read_csv(INFO_CSV)

mami_indices    = np.where(mami_mask)[0]
cat_scaled_mami = cat_scaled[mami_mask]                           # (n_mami × n_cats)
df_mami_tax     = info_mami.reset_index(drop=True)

assert len(cat_scaled_mami) == len(df_mami_tax), (
    f"Shape mismatch: {len(cat_scaled_mami)} networks vs {len(df_mami_tax)} taxonomy rows"
)

print(f"\nAnimals per order:\n{df_mami_tax[TAXONOMY].value_counts().to_string()}")

# ── 6. Select orders ──────────────────────────────────────────────────────────

order_counts   = df_mami_tax[TAXONOMY].value_counts()
orders_to_plot = order_counts[order_counts > MIN_ANIMALS].index.tolist()

print(f"\nOrders with > {MIN_ANIMALS} animals ({len(orders_to_plot)} total):")
for o in orders_to_plot:
    print(f"  {o}: {order_counts[o]}")

# ── 7. Assign colours ─────────────────────────────────────────────────────────

cmap_fallback = plt.get_cmap("tab10")
extra = [o for o in orders_to_plot if o not in COLOR_MAP_ORDERS]
for i, o in enumerate(extra):
    COLOR_MAP_ORDERS[o] = cmap_fallback(i % 10)

# ── 8. Compute per-order mean + std (in category z-score space) ───────────────

order_means = {}
order_stds  = {}
for order in orders_to_plot:
    mask = (df_mami_tax[TAXONOMY] == order).values
    order_means[order] = np.nanmean(cat_scaled_mami[mask], axis=0)
    order_stds[order]  = np.nanstd(cat_scaled_mami[mask], axis=0)

# ── 9. Shared y-range (ring at 2 always shown when there are positive values) ──

all_vals = np.concatenate(list(order_means.values()))
ylim_min = float(np.floor(np.nanmin(all_vals)))
ylim_max = float(np.ceil(np.nanmax(all_vals)))
if ylim_max > 0:
    ylim_max = max(ylim_max, 2)
ylim  = (ylim_min, ylim_max)
rings = [r for r in SPIDER_RINGS if ylim[0] <= r <= ylim[1]]

print(f"\nShared y-range: {ylim},  rings: {rings}")

# ── 10. Render one spider per order ──────────────────────────────────────────

spider_out = OUTPUT / "spider_per_order_categories"
spider_out.mkdir(exist_ok=True)

for order in orders_to_plot:
    n     = int((df_mami_tax[TAXONOMY] == order).sum())
    color = COLOR_MAP_ORDERS[order]

    make_spider(
        values=order_means[order],
        label=order,
        color=color,
        feature_labels=cat_order_present,
        title=f"{order}  (n={n})",
        ylim=ylim,
        filepath=spider_out / f"spider_{order.lower().replace(' ', '_')}.pdf",
        std_vals=order_stds[order],
        ring_vals=rings,
        viz=viz,
    )

# ── 11. Legend ────────────────────────────────────────────────────────────────

angles = np.linspace(0, 2 * np.pi, len(cat_order_present), endpoint=False).tolist()
make_spider_legend(
    feature_labels=cat_order_present,
    angles=angles,
    ylim=ylim,
    ring_vals=rings,
    filepath=spider_out / "spider_legend.pdf",
    label_color="gray",
    category_colours=CATEGORY_COLOURS,
    viz=viz,
)

print(f"\nAll plots saved to: {spider_out}")
