"""
spider_per_order.py
-------------------
For each mammalian taxonomic order with > MIN_ANIMALS in the MaMI dataset,
build one spider / radar plot showing the mean z-scored profile across the
nine functional properties used in the main analysis.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import zscore
from vizman import viz

from viz_utils import make_spider, make_spider_legend

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "output/00_trade_off_analysis"
)
OUTPUT.mkdir(exist_ok=True)

DATA_PKL = OUTPUT / "all_datasets_precise_categories.pkl"

INFO_CSV = (
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "data/preprocessed/suarez_MaMI_dataset/04_further_info/"
    "names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv"
)

MIN_ANIMALS = 10   # orders with ≤ this many animals are skipped
TAXONOMY    = "order"

# Nine spider axes → column names in the MaMI dataframe
SPIDER_PROPERTIES = {
    "Integration":                        "global_efficiency",
    "Segregation":                        "modularity",
    "Wiring\neconomy":                    "proportion_long_range_connections_0.3956",
    "Robustness":                         "targeted_attack_robustness_rob_targeted_auc",
    "Robustness\n(lambda_2)":             "algebraic_connectivity_fiedler_value",
    "Dynamics":                           "spectral_radius",
    "Memory":                             "mc_input_scaling_0_1_mc_mean",
    "Computational\ncapacity":            "mc_nonlinear_input_scaling_0_1_mc_mean",
    "Metastability\nReservoir Diversity": "repertoire_sweep_weighted_by_distances_diversity_critical",
}

# Fixed colours for the five main orders used in the rest of the analysis
COLOR_MAP_ORDERS = {
    "Primates":        "#AD5C4D",
    "Rodentia":        "#AD8B4E",
    "Carnivora":       "#4D9EAD",
    "Cetartiodactyla": "#4C6FAD",
    "Chiroptera":      "#6B4DAD",
}

SPIDER_RINGS = [-2, 0, 2, 4]

# ── Load data ─────────────────────────────────────────────────────────────────

with open(DATA_PKL, "rb") as f:
    dict_with_all_datasets = pickle.load(f)

df_mami = dict_with_all_datasets["suarez_MaMI_dataset"].copy().reset_index(drop=True)
info_mami = pd.read_csv(INFO_CSV)
df_mami = pd.concat([df_mami, info_mami], axis=1)

print(f"MaMI dataset: {len(df_mami)} animals")
print(f"\nAnimals per order:\n{df_mami[TAXONOMY].value_counts().to_string()}")

# ── Verify columns are present ────────────────────────────────────────────────

missing = [col for col in SPIDER_PROPERTIES.values() if col not in df_mami.columns]
if missing:
    print(f"\nWARNING — missing columns (will be skipped): {missing}")

spider_cols = {k: v for k, v in SPIDER_PROPERTIES.items() if v in df_mami.columns}
labels = list(spider_cols.keys())   # axis labels
cols   = list(spider_cols.values()) # dataframe columns

# ── Z-score each spider column across ALL MaMI animals ───────────────────────
# This puts every property on a common scale relative to the MaMI population.

df_z = df_mami[cols].copy().apply(zscore, nan_policy="omit")
df_z.columns = labels  # rename for clarity

# ── Select orders to plot ─────────────────────────────────────────────────────

order_counts = df_mami[TAXONOMY].value_counts()
orders_to_plot = order_counts[order_counts > MIN_ANIMALS].index.tolist()

print(f"\nOrders with > {MIN_ANIMALS} animals ({len(orders_to_plot)} total):")
for o in orders_to_plot:
    print(f"  {o}: {order_counts[o]}")

# ── Assign colours (fallback: tab10) ─────────────────────────────────────────

cmap_fallback = plt.get_cmap("tab10")
extra = [o for o in orders_to_plot if o not in COLOR_MAP_ORDERS]
for i, o in enumerate(extra):
    COLOR_MAP_ORDERS[o] = cmap_fallback(i % 10)

# ── Compute shared y-range from all order means ───────────────────────────────

order_means = {}
order_stds  = {}
for order in orders_to_plot:
    mask = df_mami[TAXONOMY] == order
    order_means[order] = df_z[mask].mean(axis=0).values
    order_stds[order]  = df_z[mask].std(axis=0).values / 10

all_vals  = np.concatenate(list(order_means.values()))
ylim_min  = float(np.floor(np.nanmin(all_vals)))
ylim_max  = float(np.ceil(np.nanmax(all_vals)))
# Always show ring at 2 when there are positive values
if ylim_max > 0:
    ylim_max = max(ylim_max, 2)
ylim  = (ylim_min, ylim_max)
rings = [r for r in SPIDER_RINGS if ylim[0] <= r <= ylim[1]]

print(f"\nShared y-range: {ylim},  rings: {rings}")

# ── Render one spider per order ───────────────────────────────────────────────

spider_out = OUTPUT / "spider_per_order"
spider_out.mkdir(exist_ok=True)

for order in orders_to_plot:
    n = int((df_mami[TAXONOMY] == order).sum())
    color = COLOR_MAP_ORDERS[order]

    make_spider(
        values=order_means[order],
        label=order,
        color=color,
        feature_labels=labels,
        title=f"{order}  (n={n})",
        ylim=ylim,
        filepath=spider_out / f"spider_{order.lower().replace(' ', '_')}.pdf",
        std_vals=order_stds[order],
        ring_vals=rings,
        viz=viz,
    )

# ── Legend (axis labels only, no data) ───────────────────────────────────────

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
make_spider_legend(
    feature_labels=labels,
    angles=angles,
    ylim=ylim,
    ring_vals=rings,
    filepath=spider_out / "spider_legend.pdf",
    label_color="gray",
    viz=viz,
)

print(f"\nAll plots saved to: {spider_out}")
