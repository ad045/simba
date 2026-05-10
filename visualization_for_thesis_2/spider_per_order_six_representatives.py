"""
spider_per_order_six_representatives.py
----------------------------------------
Spider plots with exactly six axes, one representative metric per
category (Wiring Economy, Global Integration, Local Segregation,
Communication Efficiency, Computational Capacity, Robustness).

Each axis is z-scored across all datasets before grouping by
mammalian order.

Pipeline mirrors spider_per_order_categories.py §§ 1–10.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from vizman import viz

from utils import get_combined_colors
from viz_utils import make_spider, make_spider_legend

# ── Colors ───────────────────────────────────────────────────────────────────

okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']

CLUSTER_COLORS = {
    "Wiring Economy":         okabe_ito[0],
    "Global Integration":     okabe_ito[1],
    "Local Segregation":      okabe_ito[2],
    "Communication Efficiency": okabe_ito[3],
    "Computational Capacity": okabe_ito[4],
    "Robustness":             okabe_ito[5],
}

COLOR_SCHEME = {
    "hcp_schaefer_100_dataset_gnm": "#232324",
    "suarez_MaMI_dataset":          "#799372",
    "lexis_data_developing":        "#C74800",
    "kaysons_generated_networks_diffusion":   "#F7BE18",
    "kaysons_generated_networks_propagation": "#262F3F",
    "kaysons_generated_networks_routing":     "#6E3AA3",
}

# ── Representatives: category → raw feature key ──────────────────────────────

REPRESENTATIVES = {
    "Wiring Economy":           "wiring_cost",
    "Global Integration":       "global_efficiency",
    "Local Segregation":        "modularity",
    "Communication Efficiency": "diffusion_efficiency",
    "Computational Capacity":   "computational_capacity_total_capacity",
    "Robustness":               "targeted_attack_robustness_rob_targeted_auc",
}

AXIS_ORDER   = list(REPRESENTATIVES.keys())
FEATURE_KEYS = [REPRESENTATIVES[cat] for cat in AXIS_ORDER]

DISPLAY_NAMES = {
    "wiring_cost":                                  "Wiring Cost",
    "global_efficiency":                            "Global Efficiency",
    "modularity":                                   "Modularity",
    "diffusion_efficiency":                         "Diffusion Eff.",
    "computational_capacity_total_capacity":        "Total Comp. Capacity",
    "targeted_attack_robustness_rob_targeted_auc":  "Robustness (Targeted)",
}

# ── Paths ─────────────────────────────────────────────────────────────────────

OUTPUT = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "output/00_trade_off_analysis"
)
DATA_PKL = OUTPUT / "all_datasets_precise_categories.pkl"

INFO_CSV = (
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "data/preprocessed/suarez_MaMI_dataset/04_further_info/"
    "names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv"
)

GNM_DATASET  = "hcp_schaefer_100_dataset_gnm"
MIN_ANIMALS  = 10
TAXONOMY     = "order"
SPIDER_RINGS = [-2, 0, 2, 4]

ORDERED_DATASETS = [
    "hcp_schaefer_100_dataset_gnm",
    "suarez_MaMI_dataset",
    "lexis_data_developing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
]

COLOR_MAP_ORDERS = {
    "Primates":        "#AD5C4D",
    "Rodentia":        "#AD8B4E",
    "Carnivora":       "#4D9EAD",
    "Cetartiodactyla": "#4C6FAD",
    "Chiroptera":      "#6B4DAD",
}

# ── 1. Load data ──────────────────────────────────────────────────────────────

with open(DATA_PKL, "rb") as f:
    dict_with_all_datasets = pickle.load(f)

frames = []
for ds in ORDERED_DATASETS:
    if ds not in dict_with_all_datasets:
        print(f"  skipping {ds} (not in pickle)")
        continue
    df = dict_with_all_datasets[ds].copy()
    df["dataset"] = ds
    df["color_dataset"] = (
        get_combined_colors(df) if ds == GNM_DATASET else COLOR_SCHEME.get(ds, "#999999")
    )
    frames.append(df)

huge_df = pd.concat(frames, ignore_index=True)

# ── 2. Extract & check representative columns ─────────────────────────────────

missing = [k for k in FEATURE_KEYS if k not in huge_df.columns]
if missing:
    raise KeyError(f"Missing representative features in data: {missing}")

rep_df = huge_df[FEATURE_KEYS].copy()
rep_df.replace([np.inf, -np.inf], np.nan, inplace=True)
rep_df.dropna(inplace=True)
valid_idx = rep_df.index

print(f"Networks with all 6 features present: {len(valid_idx)} / {len(huge_df)}")

# ── 3. Z-score each feature across all networks ───────────────────────────────

scaled = StandardScaler().fit_transform(rep_df.values)   # (n_networks, 6)

# ── 4. Filter MaMI rows ───────────────────────────────────────────────────────

mami_mask_full = (huge_df["dataset"] == "suarez_MaMI_dataset").values
mami_in_valid  = np.isin(valid_idx, np.where(mami_mask_full)[0])

scaled_mami  = scaled[mami_in_valid]
info_mami    = pd.read_csv(INFO_CSV).reset_index(drop=True)
huge_valid   = huge_df.loc[valid_idx].reset_index(drop=True)
df_mami_rows = huge_valid[huge_valid["dataset"] == "suarez_MaMI_dataset"].reset_index(drop=True)

assert len(scaled_mami) == len(info_mami), (
    f"Shape mismatch: {len(scaled_mami)} networks vs {len(info_mami)} taxonomy rows"
)

print(f"\nAnimals per order:\n{info_mami[TAXONOMY].value_counts().to_string()}")

# ── 5. Select orders ──────────────────────────────────────────────────────────

order_counts   = info_mami[TAXONOMY].value_counts()
orders_to_plot = order_counts[order_counts > MIN_ANIMALS].index.tolist()

print(f"\nOrders with > {MIN_ANIMALS} animals ({len(orders_to_plot)} total):")
for o in orders_to_plot:
    print(f"  {o}: {order_counts[o]}")

# ── 6. Fill missing order colours ────────────────────────────────────────────

cmap_fallback = plt.get_cmap("tab10")
for i, o in enumerate([o for o in orders_to_plot if o not in COLOR_MAP_ORDERS]):
    COLOR_MAP_ORDERS[o] = cmap_fallback(i % 10)

# ── 7. Per-order mean + std in representative-feature space ──────────────────

order_means = {}
order_stds  = {}
for order in orders_to_plot:
    mask = (info_mami[TAXONOMY] == order).values
    order_means[order] = np.nanmean(scaled_mami[mask], axis=0)
    order_stds[order]  = np.nanstd(scaled_mami[mask],  axis=0)

# ── 8. Shared y-range ─────────────────────────────────────────────────────────

all_vals = np.concatenate(list(order_means.values()))
ylim_min = float(np.floor(np.nanmin(all_vals)))
ylim_max = float(np.ceil(np.nanmax(all_vals)))
if ylim_max > 0:
    ylim_max = max(ylim_max, 2)
ylim  = (ylim_min, ylim_max)
rings = [r for r in SPIDER_RINGS if ylim[0] <= r <= ylim[1]]

print(f"\nShared y-range: {ylim},  rings: {rings}")

# ── 9. Render one spider per order ───────────────────────────────────────────

spider_out = OUTPUT / "spider_per_order_six_representatives"
spider_out.mkdir(exist_ok=True)

axis_labels = [DISPLAY_NAMES[k] for k in FEATURE_KEYS]

for order in orders_to_plot:
    n     = int((info_mami[TAXONOMY] == order).sum())
    color = COLOR_MAP_ORDERS[order]

    make_spider(
        values=order_means[order],
        label=order,
        color=color,
        feature_labels=axis_labels,
        title=f"{order}  (n={n})",
        ylim=ylim,
        filepath=spider_out / f"spider_{order.lower().replace(' ', '_')}.pdf",
        std_vals=order_stds[order],
        ring_vals=rings,
        viz=viz,
    )

# ── 10. Legend ────────────────────────────────────────────────────────────────

n_axes      = len(AXIS_ORDER)
angles      = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
axis_colors = {lbl: CLUSTER_COLORS[cat] for lbl, cat in zip(axis_labels, AXIS_ORDER)}

make_spider_legend(
    feature_labels=axis_labels,
    angles=angles,
    ylim=ylim,
    ring_vals=rings,
    filepath=spider_out / "spider_legend.pdf",
    label_color="gray",
    category_colours=axis_colors,
    viz=viz,
)

print(f"\nAll plots saved to: {spider_out}")
