
# %% [markdown]
# # 00 · Data Preparation
# Loads raw metric CSVs for each dataset, merges them, cleans columns,
# filters to the agreed feature set, and saves `all_datasets_precise_categories.pkl`.

# %% ── Imports ──────────────────────────────────────────────────────────────
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from vizman import viz

from config import (
    COLORS, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES,
    emp_dataset_and_experiment_pairs,
)

# %load_ext autoreload
# %autoreload 2

OUTPUT = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis")
OUTPUT.mkdir(exist_ok=True)

# %% ── Dataset selection ────────────────────────────────────────────────────
# Core datasets — always included.
DATASETS = [
    "hcp_schaefer_100_dataset_gnm",
    "suarez_MaMI_dataset",
    "lexis_data_developing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
    "lexis_data_developing_consensus_per_age_1_year",
    # "lexis_data_young_consensus_per_age_1_year",
    # "lexis_data_aging_consensus_per_age_1_year",
    # "lexis_data_all_consensus_per_age_1_year",
    # "lexis_data_developing_consensus_per_age_2_year",
]

# Reference networks — added automatically if their metrics CSV already exists.
OPTIONAL_DATASETS = [
    "ring_lattice_networks",
    "erdos_renyi_networks",
]

METRIC_FILES = {
    "static":        "metrics_2_static.csv",
    "dynamic":       "metrics_2_dynamic.csv",
    "computational": "metrics_2_computational.csv",
    "further":       "metrics_2_further.csv",
}

# Columns that are never needed for analysis
COLS_TO_DROP_ALWAYS = [
    "distance_relationship_type", "generative_rule", "num_iterations",
    "preferential_relationship_type", "id", "network_idx", "network_index",
    "mc_values_for_indiv_lags",
]
COLS_WITH_SWEEP_VECTORS = [
    "repertoire_sweep_T_vec", "repertoire_sweep_sizes", "repertoire_sweep_diversities",
    "repertoire_sweep_weighted_by_distances_T_vec",
    "repertoire_sweep_weighted_by_distances_sizes",
    "repertoire_sweep_weighted_by_distances_diversities",
    "repertoire_sweep_T_critical", "repertoire_sweep_size_critical",
    "repertoire_sweep_diversity_critical",
    "proportion_long_range_connections_0.356",
    "repertoire_diversity", "repertoire_size",
    "departure_from_normality",
]

# %% ── Feature set ──────────────────────────────────────────────────────────
# Each list is one logical category; order here determines category ordering
# in downstream plots.  Comment-out to exclude individual features.

FEATURE_CATEGORIES: dict[str, list[str]] = {
    "Fundamental Topology": [
        "transitivity", "avg_clustering", "modularity", "degree_gini",
        "degree_assortativity", "omega", "structural_complexity",
        "directed_simplices_count", "directed_simplices_max_size",
        # "energy",   # fit-to-data metric — omit by default
        "eta", "gamma",
    ],
    "Paths, Efficiency & Communication": [
        "char_path_length", "global_efficiency", "diffusion_efficiency",
        "propagation_efficiency", "avg_communicability",
        "topological_distance_mean", "topological_distance_std",
    ],
    "Spatial Embedding & Wiring Cost": [
        "avg_edge_distance", "wiring_cost",
        "proportion_long_range_connections_0.1",
        "proportion_long_range_connections_0.3",
        "proportion_long_range_connections_0.3956",
        "proportion_long_range_connections_0.5",
    ],
    "Rich-Club Organization": [
        "richclub_n_edges", "richclub_avg_length",
        "rich_club_coefficient_rc_max_norm", "rich_club_coefficient_rc_mean_norm",
        "rich_club_coefficient_rc_k_at_max", "rich_club_coefficient_rc_regime_frac",
        "rich_club_coefficient_rc_weighted_auc",
    ],
    "Spectral Properties, Algebraic Connectivity & Fiedler Analysis": [
        "spectral_radius", "spectral_gap",
        "synchronizability_eigenratio_eigenratio",
        "kernel_rank_thresholded_and_summed_0.01",
        "kernel_rank_phase_diff_of_lambda_max_and_2nd",
        "algebraic_connectivity_fiedler_value",
        "algebraic_connectivity_fiedler_value_norm",
        "algebraic_connectivity_laplacian_spectral_gap",
        "algebraic_connectivity_fiedler_bipartition_balance",
        "departure_from_normality_schur",
        "effective_dimensionality",
    ],
    "Metastability & Kuramoto Synchronization": [
        "repertoire_sweep_weighted_by_distances_T_critical",
        "repertoire_sweep_weighted_by_distances_size_critical",
        "repertoire_sweep_weighted_by_distances_diversity_critical",
        "kuramoto_averaged_synchronization_r_final",
        "kuramoto_averaged_synchronization_r_mean",
        "kuramoto_averaged_synchronization_r_mean_se",
        "kuramoto_averaged_synchronization_r_std",
        "kuramoto_averaged_synchronization_r_std_se",
    ],
    "Participation Coefficient (Louvain)": [
        "participation_coefficient_n_communities",
        "participation_coefficient_pc_mean", "participation_coefficient_pc_median",
        "participation_coefficient_pc_std", "participation_coefficient_pc_frac_connector",
        "participation_coefficient_wmd_std",
    ],
    "Ollivier-Ricci Curvature": [
        "ollivier_ricci_curvature_orc_mean", "ollivier_ricci_curvature_orc_median",
        "ollivier_ricci_curvature_orc_std", "ollivier_ricci_curvature_orc_min",
        "ollivier_ricci_curvature_orc_max", "ollivier_ricci_curvature_orc_skewness",
        "ollivier_ricci_curvature_orc_frac_neg",
    ],
    "Persistent Homology (TDA)": [
        "persistent_homology_ph_h1_n_features",
        "persistent_homology_ph_h1_persistence_mean",
        "persistent_homology_ph_h1_entropy",
        "persistent_homology_ph_total_persistence",
    ],
    "Targeted Attack Robustness": [
        "targeted_attack_robustness_rob_targeted_auc",
        "targeted_attack_robustness_rob_targeted_half",
        "targeted_attack_robustness_rob_random_auc",
        "targeted_attack_robustness_rob_random_half",
        "targeted_attack_robustness_rob_ratio",
        "community_synchronization_vulnerability_value",
        "community_synchronization_vulnerability_n_communities",
    ],
    "Network Control Theory - Average Controllability": [
        "nct_control_avg", "nct_control_std", "nct_control_max",
        "nct_control_n_nodes_90_percent",
        "nct_control_n_nodes_50_percent",
        "nct_control_n_nodes_10_percent",
    ],
    "Network Control Theory - Control Energy": [
        "nct_energies_total", "nct_energies_std", "nct_energies_max",
        "nct_energies_n_nodes_90_percent",
        "nct_energies_n_nodes_50_percent",
        "nct_energies_n_nodes_10_percent",
    ],
    "Memory Capacity": [
        "mc_input_scaling_0_1_mc_mean",
        "mc_nonlinear_input_scaling_0_1_mc_mean",
    ],
}

PRECISE_CATEGORIES: list[str] = [
    col for cat_cols in FEATURE_CATEGORIES.values() for col in cat_cols
    if col not in ("energy",)           # global exclusions
]

# %% ── Loading + cleaning helpers ───────────────────────────────────────────

def _fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip known noisy prefixes from column names."""
    df.columns = [
        re.sub(r"^basic_measures_|^mc_original_", "", c)
        for c in df.columns
    ]
    return df


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop content-identical duplicate columns; rename non-identical ones."""
    cols_to_keep, seen = [], {}
    for i, col in enumerate(df.columns):
        if col not in seen:
            seen[col] = i
            cols_to_keep.append(i)
        else:
            if df.iloc[:, seen[col]].equals(df.iloc[:, i]):
                print(f"  Dropping identical duplicate: '{col}'")
            else:
                new_name = f"{col}_dup"
                df.columns.values[i] = new_name
                cols_to_keep.append(i)
                print(f"  Renamed non-identical duplicate '{col}' → '{new_name}'")
    return df.iloc[:, cols_to_keep]


def _filter_connected(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the graph is not a single connected component."""
    if "n_connected_components" not in df.columns:
        return df
    bad = df[df["n_connected_components"] != 1].index
    if len(bad):
        print(f"  Removing {len(bad)} disconnected networks.")
        df = df.drop(index=bad)
    return df


# The GNM sweep data lives under hcp_schaefer_100_dataset/, not the _gnm suffix.
_DIR_OVERRIDE = {
    "hcp_schaefer_100_dataset_gnm": "hcp_schaefer_100_dataset",
}


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load, merge, and clean all metric CSVs for one dataset."""
    exp     = emp_dataset_and_experiment_pairs[dataset_name]
    dir_name = _DIR_OVERRIDE.get(dataset_name, dataset_name)
    base = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dir_name}/{exp}")

    dfs = {k: pd.read_csv(base / v) for k, v in METRIC_FILES.items()}
    df = dfs["static"]
    for key in ("dynamic", "computational", "further"):
        df = df.merge(dfs[key], left_index=True, right_index=True,
                      how="left", suffixes=("", f"_{key}"))

    df = _fix_column_names(df)
    df = _drop_duplicate_columns(df)

    # Drop vector / sweep columns and always-remove columns
    drop = COLS_TO_DROP_ALWAYS + COLS_WITH_SWEEP_VECTORS
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.drop(columns=[c for c in df.columns if c.startswith("h_params")])
    df = df.dropna(axis=1, how="all")

    df = _filter_connected(df)

    # Drop non-numeric columns and profile columns
    profile_cols = [
        c for c in df.columns if "profile" in c
        and df[c].dtype == object
    ]
    nonnumeric = [
        c for c in df.columns
        if df[c].dtype not in (np.float64, np.float32, np.int64, np.int32)
    ]
    df = df.drop(columns=profile_cols + nonnumeric)

    print(f"  {dataset_name}: {df.shape}")
    return df


# %% ── Load all datasets ────────────────────────────────────────────────────
raw_datasets: dict[str, pd.DataFrame] = {}
for name in DATASETS:
    raw_datasets[name] = load_dataset(name)

# %% ── Restrict to agreed feature set ──────────────────────────────────────
datasets: dict[str, pd.DataFrame] = {}
for name, df in raw_datasets.items():
    cols = [c for c in PRECISE_CATEGORIES if c in df.columns]
    datasets[name] = df[cols]
    print(f"{name}: {len(cols)} features retained")

# %% ── Add optional reference networks (if metrics already computed) ─────────
for name in OPTIONAL_DATASETS:
    try:
        df = load_dataset(name)
        cols = [c for c in PRECISE_CATEGORIES if c in df.columns]
        datasets[name] = df[cols]
        print(f"  {name}: added ({len(cols)} features)")
    except FileNotFoundError:
        print(f"  {name}: skipped (run run_reference_networks_metrics.py first)")

# %% ── Save ─────────────────────────────────────────────────────────────────
print(f"\nDatasets in pkl: {list(datasets.keys())}")
with open(OUTPUT / "all_datasets_precise_categories.pkl", "wb") as f:
    pickle.dump(datasets, f)
print("Saved:", OUTPUT / "all_datasets_precise_categories.pkl")

# Also save the feature list for reference
with open(OUTPUT / "selected_categories.pkl", "wb") as f:
    pickle.dump(PRECISE_CATEGORIES, f)

# %% ── Memory-capacity lag profile preview (optional) ───────────────────────
# Uncomment to inspect the per-lag MC profiles before summarising.

# LIN_COL    = "mc_input_scaling_0_1_mc_values_for_indiv_lags"
# NONLIN_COL = "mc_nonlinear_input_scaling_0_1_mc_values_for_indiv_lags"

# for ds_name in DATASETS:
#     df = raw_datasets[ds_name]
#     fig, ax = plt.subplots(figsize=viz.cm_to_inch((10, 6)))
#     for idx, row in df.head(10).iterrows():
#         lin = np.array(row[LIN_COL].strip("[]").split(","), dtype=np.float32)
#         nonlin = np.array(row[NONLIN_COL].strip("[]").split(","), dtype=np.float32)
#         ax.plot(lin, color="steelblue", alpha=0.6)
#         ax.plot(nonlin, color="darkorange", alpha=0.6)
#     ax.set_title(ds_name)
#     plt.tight_layout()
#     plt.show()
