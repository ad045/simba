"""
run_validate_categories.py
──────────────────────────
Loads the real data and runs the category validation.
Mirrors the data loading from spider_per_order_categories.py exactly.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from config import remaining_categories
from validate_categories import run_validation

# ── Constants (copied from spider_per_order_categories to avoid its module-level
#    data loading, which requires an eta column not present here) ───────────────

DATA_PKL = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "output/00_trade_off_analysis/all_datasets_precise_categories.pkl"
)

ORDERED_DATASETS = [
    "hcp_schaefer_100_dataset_gnm",
    "suarez_MaMI_dataset",
    "lexis_data_developing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
]

PROPERTIES_TO_FLIP = [
    "rich_club_coefficient_rc_k_at_max",
    "directed_simplices_count",
    "participation_coefficient_pc_frac_connector",
    "community_synchronization_vulnerability_n_communities",
    "participation_coefficient_n_communities",
    "repertoire_sweep_weighted_by_distances_diversity_critical",
]

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

# ── Load data ─────────────────────────────────────────────────────────────────

with open(DATA_PKL, "rb") as f:
    dict_with_all_datasets = pickle.load(f)

frames = []
for ds in ORDERED_DATASETS:
    df = dict_with_all_datasets[ds].copy()
    df["dataset"] = ds
    df["color_dataset"] = ds
    frames.append(df)

huge_df       = pd.concat(frames, ignore_index=True)
huge_df_props = huge_df.drop(columns=["dataset", "color_dataset"])

# ── Sign-flip ─────────────────────────────────────────────────────────────────

for prop in PROPERTIES_TO_FLIP:
    if prop in huge_df_props.columns:
        huge_df_props[prop] = -huge_df_props[prop]

huge_df_props.replace([np.inf, -np.inf], np.nan, inplace=True)

print(f"Loaded {len(huge_df)} networks across {huge_df['dataset'].nunique()} datasets.")
print(f"Features available: {huge_df_props.shape[1]}")

# ── Run validation ────────────────────────────────────────────────────────────

run_validation(
    df_props         = huge_df_props,
    precise_features = PRECISE_FEATURES,       # 11 categories, ~30 selected features
    all_features     = remaining_categories,   # all features with category assignments
    dataset_labels   = huge_df["dataset"],     # detrend by dataset
)
