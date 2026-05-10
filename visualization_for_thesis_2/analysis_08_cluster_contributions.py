"""
Cluster contribution decomposition of global PCA axes (Option 2).

For each network i and PC k:
    PC_k_score[i] = sum over all features j of (z[i,j] * loading[k,j])

This script decomposes that sum into 6 cluster contributions:
    PC_k_score[i] = contrib_wiring[i] + contrib_integration[i] + ... (6 terms, exact)

The contributions sum to the total PC score for every network.
Output: CSV + diagnostic plots.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup — import shared config from the same directory
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from config import COLOR_SCHEME, LABEL_MAP
from vizman import viz

OUTPUT_FOLDER = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis")
DATA_PATH     = OUTPUT_FOLDER / "all_datasets_precise_categories.pkl"
ETA_GAMMA_PATH = OUTPUT_FOLDER / "hcp_schaefer_100_dataset_gnm_eta_and_gamma.pkl"
OUT_DIR       = OUTPUT_FOLDER / "cluster_contributions"
OUT_DIR.mkdir(exist_ok=True)

INCLUDE_MC_IN_PCA = True   # match the main notebook setting

ORDERED_DATASETS = [
    "hcp_schaefer_100_dataset_gnm",
    "lexis_data_developing",
    "suarez_MaMI_dataset",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
    "ring_lattice_networks",
    "erdos_renyi_networks",
]

# ---------------------------------------------------------------------------
# 6 clusters — exact column names as they appear in the feature DataFrames
# ---------------------------------------------------------------------------
# CLUSTERS = {
#     "Wiring Economy": [
#         "wiring_cost",
#         "proportion_long_range_connections_0.3956",
#     ],
#     "Global Integration": [
#         "global_efficiency",
#         "diffusion_efficiency",
#         "propagation_efficiency",
#     ],
#     "Local Segregation": [
#         "modularity",
#         "local_efficiency_stats_mean",
#         "local_efficiency_stats_std",
#         "omega",
#         "directed_simplices_count",
#         "directed_simplices_max_size",
#         "ollivier_ricci_curvature_orc_mean",
#         "ollivier_ricci_curvature_orc_min",
#         "ollivier_ricci_curvature_orc_skewness",
#     ],
#     "Hub Architecture & Resilience": [
#         "degree_gini",
#         "betweenness_centrality_stats_gini",
#         "betweenness_centrality_stats_mean",
#         "degree_assortativity",
#         "rich_club_coefficient_rc_k_at_max",
#         "participation_coefficient_pc_std",
#         "nct_control_std",
#         "targeted_attack_robustness_rob_targeted_auc",
#         "targeted_attack_robustness_rob_random_auc",
#     ],
#     "Dynamical Regime": [
#         "spectral_radius",
#         "spectral_gap",
#         "synchronizability_eigenratio_eigenratio",
#         "algebraic_connectivity_fiedler_value",
#         "algebraic_connectivity_laplacian_spectral_gap",
#         "community_synchronization_vulnerability_value",
#         "structural_complexity",
#         "repertoire_sweep_weighted_by_distances_T_critical",
#         "repertoire_sweep_weighted_by_distances_size_critical",
#         "repertoire_sweep_weighted_by_distances_diversity_critical",
#     ],
#     "Computational Capacity": [
#         "ipc_ipc_deg1_mean",
#         "ipc_ipc_deg2_mean",
#         "kernel_rank_max",
#         "effective_dimensionality",
#         "departure_from_normality_schur",
#     ],
# }


# My novely defined cluster subselection
CLUSTERS_SUBSELECTION = {
    "Wiring Economy": [
        "wiring_cost", # y
        "proportion_long_range_connections_0.3956", # y
    ],
    "Global Integration": [
        "global_efficiency", # y
        "diffusion_efficiency", # y
        "propagation_efficiency", # y
    ],
    "Local Segregation": [
        "modularity", # y
        # "local_efficiency_stats_mean",
        # "local_efficiency_stats_std",
        "omega", # y
        "directed_simplices_count", # y
        "directed_simplices_max_size", # y
        # "ollivier_ricci_curvature_orc_mean",
        "ollivier_ricci_curvature_orc_min", # y
        "ollivier_ricci_curvature_orc_skewness", # y
    ],
    # How fast and metabolically cheap the network coordinates dynamically.
    # Two sub-groups:
    #   (a) Hub routing: centralized architecture reduces average hop count
    #   (b) Synchronization speed: spectral properties governing the rate of
    #       oscillatory coordination and random-walk mixing
    "Communication Efficiency": [
        # Hub routing
        # "degree_gini",
        # "betweenness_centrality_stats_gini",
        # "betweenness_centrality_stats_mean",
        # "degree_assortativity",
        "rich_club_coefficient_rc_k_at_max", # y
        "participation_coefficient_pc_std", # y
        # With Kayson: WE ACTUALLY ALSO HAD FRAC CONNECTIONS
        "nct_control_std", # y
        # Synchronization speed
        "spectral_gap", # y
        "synchronizability_eigenratio_eigenratio", # y
        # "synchronizability_eigenratio_eigenratio_lambda_2", # y >> has nan values...
        "algebraic_connectivity_fiedler_value",
        "algebraic_connectivity_laplacian_spectral_gap", # y
        # "community_synchronization_vulnerability_value",
    ],
    # Reservoir-like computation: temporal memory, nonlinear transformation,
    # and diversity of accessible dynamical states.
    # Three sub-groups:
    #   (a) Reservoir properties: IPC and state-space geometry
    #   (b) Spectral substrate: eigenspectrum structure enabling rich dynamics
    #   (c) Critical state repertoire: pattern diversity at the excitability threshold
    "Computational Capacity": [
        # Reservoir properties
        "ipc_ipc_deg1_mean", # y
        "ipc_ipc_deg2_mean", # y
        # "kernel_rank_max",
        # "effective_dimensionality",
        "departure_from_normality_schur", # y
        # Spectral substrate
        "spectral_radius", # y
        # "structural_complexity",
        # Critical state repertoire
        "repertoire_sweep_weighted_by_distances_T_critical", # y
        "repertoire_sweep_weighted_by_distances_size_critical", # y
        "repertoire_sweep_weighted_by_distances_diversity_critical", # y
    ],
    "Robustness": [
        "targeted_attack_robustness_rob_targeted_auc", # y
        "targeted_attack_robustness_rob_random_auc", # we had the ratio instead... Added that again. (ADD ALSO IN APDX VERSION?)
        # "targeted_attack_robustness_rob_ratio", # y
    ],
}


CLUSTERS_FULL = {
    "Wiring Economy": [
        "wiring_cost",
        "proportion_long_range_connections_0.3956",
    ],
    "Global Integration": [
        "global_efficiency",
        "diffusion_efficiency",
        "propagation_efficiency",
    ],
    "Local Segregation": [
        "modularity",
        "local_efficiency_stats_mean",
        "local_efficiency_stats_std",
        "omega",
        "directed_simplices_count",
        "directed_simplices_max_size",
        "ollivier_ricci_curvature_orc_mean",
        "ollivier_ricci_curvature_orc_min",
        "ollivier_ricci_curvature_orc_skewness",
    ],
    # How fast and metabolically cheap the network coordinates dynamically.
    # Two sub-groups:
    #   (a) Hub routing: centralized architecture reduces average hop count
    #   (b) Synchronization speed: spectral properties governing the rate of
    #       oscillatory coordination and random-walk mixing
    "Communication Efficiency": [
        # Hub routing
        "degree_gini",
        "betweenness_centrality_stats_gini",
        "betweenness_centrality_stats_mean",
        "degree_assortativity",
        "rich_club_coefficient_rc_k_at_max",
        "participation_coefficient_pc_std",
        "nct_control_std",
        # Synchronization speed
        "spectral_gap",
        "synchronizability_eigenratio_eigenratio",
        "synchronizability_eigenratio_eigenratio_lambda_2",
        # "algebraic_connectivity_fiedler_value",
        "algebraic_connectivity_laplacian_spectral_gap",
    ],
    # Reservoir-like computation: temporal memory, nonlinear transformation,
    # and diversity of accessible dynamical states.
    # Three sub-groups:
    #   (a) Reservoir properties: IPC and state-space geometry
    #   (b) Spectral substrate: eigenspectrum structure enabling rich dynamics
    #   (c) Critical state repertoire: pattern diversity at the excitability threshold
    "Computational Capacity": [
        # Reservoir properties
        "ipc_ipc_deg1_mean",
        "ipc_ipc_deg2_mean",
        "kernel_rank_max",
        "effective_dimensionality",
        "departure_from_normality_schur",
        # Spectral substrate
        "spectral_radius",
        "structural_complexity",
        # Critical state repertoire
        "repertoire_sweep_weighted_by_distances_T_critical",
        "repertoire_sweep_weighted_by_distances_size_critical",
        "repertoire_sweep_weighted_by_distances_diversity_critical",
    ],
    "Robustness": [
        "targeted_attack_robustness_rob_targeted_auc",
        "targeted_attack_robustness_rob_random_auc",
        "targeted_attack_robustness_rob_ratio", # y
        "community_synchronization_vulnerability_value",
    ],
}




CLUSTERS_OLD = {
    "Wiring Economy": [
        "wiring_cost",
        "proportion_long_range_connections_0.3956",
    ],
    "Global Integration": [
        "ollivier_ricci_curvature_orc_min",
        "ollivier_ricci_curvature_orc_skewness",
        "rich_club_coefficient_rc_k_at_max",
        "global_efficiency",
        "diffusion_efficiency",
        "propagation_efficiency",
        "omega",
    ],
    "Local Segregation": [
        "modularity",
        # "local_efficiency_stats_mean",
        # "local_efficiency_stats_std",
        "directed_simplices_count",
        "directed_simplices_max_size",
        "participation_coefficient_pc_frac_connector",
        "participation_coefficient_n_communities",
        "participation_coefficient_pc_std",
    ],
    # How fast and metabolically cheap the network coordinates dynamically.
    # Two sub-groups:
    #   (a) Hub routing: centralized architecture reduces average hop count
    #   (b) Synchronization speed: spectral properties governing the rate of
    #       oscillatory coordination and random-walk mixing
    "Communication Efficiency": [
        # Hub routing
        # "degree_gini",
        # "betweenness_centrality_stats_gini",
        # "betweenness_centrality_stats_mean",
        # "degree_assortativity",
        
        # "participation_coefficient_pc_std",
        "nct_control_std",
        # Synchronization speed
        "spectral_gap",
        "synchronizability_eigenratio_eigenratio",
        "spectral_radius",
        # "synchronizability_eigenratio_eigenratio_lambda_2",
        # "algebraic_connectivity_laplacian_spectral_gap",
    ],
    # Reservoir-like computation: temporal memory, nonlinear transformation,
    # and diversity of accessible dynamical states.
    # Three sub-groups:
    #   (a) Reservoir properties: IPC and state-space geometry
    #   (b) Spectral substrate: eigenspectrum structure enabling rich dynamics
    #   (c) Critical state repertoire: pattern diversity at the excitability threshold
    "Computational Capacity": [
        # Reservoir properties
        "ipc_ipc_deg1_mean",
        "ipc_ipc_deg2_mean",
        # "kernel_rank_max",
        # "effective_dimensionality",
        "departure_from_normality_schur",
        
        # Spectral substrate
        # "structural_complexity",
        
        # Critical state repertoire
        "repertoire_sweep_weighted_by_distances_T_critical",
        "repertoire_sweep_weighted_by_distances_size_critical",
        "repertoire_sweep_weighted_by_distances_diversity_critical",
    ],
    "Robustness": [
        "targeted_attack_robustness_rob_targeted_auc",
        # "targeted_attack_robustness_rob_random_auc",
        "targeted_attack_robustness_rob_ratio", # y
        # "community_synchronization_vulnerability_value",
        "algebraic_connectivity_fiedler_value",
        "algebraic_connectivity_laplacian_spectral_gap",
    ],
}


# # DETERMINED WITH KAYSON (until Mar 16th) 
# # ── Category definitions ──────────────────────────────────────────────────────
# CLUSTERS_SUBSELECTION = {
#     "Fundamental Topology": [ 
#         # "transitivity", # corr too highly with avg_clustering
#         # "avg_clustering", # omega
#         "modularity", 
#         # "degree_gini", # too high corr with spectral_radius
#         # "degree_assortativity", 
#         "omega", 
#         # "structural_complexity", 
        
#         "directed_simplices_count", # Higher order
#         "directed_simplices_max_size",
        
#         # "energy" # i.e.: Fit to networks - maybe remove again? 
#     ],
#     "Paths, Efficiency & Communication": [
#         # "char_path_length", 
#         "global_efficiency", # corr too highly with omega
#         "diffusion_efficiency",
#         "propagation_efficiency", # corr too highly with diffusion_efficiency
#         # "avg_communicability", # corr too highly with spectral_radius # !!!!!!!
#         # "topological_distance_mean", 
#         # "topological_distance_std",
#     ],
#     "Spatial Embedding & Wiring Cost": [
#         # "avg_edge_distance", 
#         "wiring_cost",
#         # "proportion_long_range_connections_0.1",
#         # "proportion_long_range_connections_0.3",
#         "proportion_long_range_connections_0.3956",
#         # "proportion_long_range_connections_0.5",
#     ],
#     "Rich-Club Organization": [
#         # "richclub_n_edges", 
#         # "richclub_avg_length",
#         # "rich_club_coefficient_rc_max_norm",
#         # "rich_club_coefficient_rc_mean_norm",
#         "rich_club_coefficient_rc_k_at_max",
#         # "rich_club_coefficient_rc_regime_frac",
#         # "rich_club_coefficient_rc_weighted_auc",
#     ],
#     "Spectral Properties, Algebraic Connectivity & Fiedler Analysis": [
#         "spectral_radius", 
#         "spectral_gap", # Has exactly the same output as Fatemehs version, bc it is identical (lambda_n - lambda_(n-1))
#         "synchronizability_eigenratio_eigenratio", # lambda_n / lambda_2

#         # "kernel_rank_thresholded_and_summed_0.01",          # np.sum(np.abs(eigs) > threshold * np.abs(eigs).max())
#         # "kernel_rank_phase_diff_of_lambda_max_and_2nd",     # np.angle(eigs[np.argsort(np.abs(eigs))[-2]]) - np.angle(eigs.max())
        
#         "algebraic_connectivity_fiedler_value", # Relies on a different library, I hope? 
#         # "algebraic_connectivity_fiedler_value_norm",
#         "algebraic_connectivity_laplacian_spectral_gap",
#         # "algebraic_connectivity_fiedler_bipartition_balance",
        
#         "departure_from_normality_schur",

#         # "effective_dimensionality", # corr too highly with omega
#     ],
#     "Metastability & Kuramoto Synchronization": [ 
#         "repertoire_sweep_weighted_by_distances_T_critical",
#         "repertoire_sweep_weighted_by_distances_size_critical",
#         "repertoire_sweep_weighted_by_distances_diversity_critical", 
        
#         # "kuramoto_averaged_synchronization_r_final",
#         # "kuramoto_averaged_synchronization_r_mean",
#         # "kuramoto_averaged_synchronization_r_mean_se",
#         # "kuramoto_averaged_synchronization_r_std",
#         # "kuramoto_averaged_synchronization_r_std_se",
#     ],

#     "Participation Coefficient (Louvain)": [
#         "participation_coefficient_n_communities",
#         # "participation_coefficient_pc_mean", # corr too highly with omega
#         # "participation_coefficient_pc_median",
#         "participation_coefficient_pc_std",
#         "participation_coefficient_pc_frac_connector", # needs to be put into relation with Q
#         # "participation_coefficient_wmd_std",
#     ],
#     "Ollivier-Ricci Curvature": [ # If too annoying to describe, keep only one. 
#         # "ollivier_ricci_curvature_orc_mean",
#         # "ollivier_ricci_curvature_orc_median",
#         # "ollivier_ricci_curvature_orc_std", 
#         "ollivier_ricci_curvature_orc_min",
#         # "ollivier_ricci_curvature_orc_max",
#         "ollivier_ricci_curvature_orc_skewness",
#         # "ollivier_ricci_curvature_orc_frac_neg",
#     ],
#     # "Persistent Homology (TDA)": [
#     #     # "persistent_homology_ph_h1_n_features",
#     #     # "persistent_homology_ph_h1_persistence_mean",
#     #     # "persistent_homology_ph_h1_entropy",
#     #     # "persistent_homology_ph_total_persistence",
#     # ],

#     "Targeted Attack Robustness & Community Structure & Vulnerability": [
#         "targeted_attack_robustness_rob_targeted_auc",
#         # "targeted_attack_robustness_rob_targeted_half",
#         # "targeted_attack_robustness_rob_random_auc", # corr too highly with 'char_path_length'
#         # "targeted_attack_robustness_rob_random_half",
#         "targeted_attack_robustness_rob_ratio",
        
#         # "community_synchronization_vulnerability_value", # too high corr with participation_coefficient_pc_frac_connector
#         "community_synchronization_vulnerability_n_communities",
#         ],
#     "Network Control Theory - Average Controllability": [
#         # "nct_control_avg",
#         "nct_control_std",  
#         # "nct_control_max",
#         # "nct_control_n_nodes_90_percent",
#         # "nct_control_n_nodes_50_percent",
#         # "nct_control_n_nodes_10_percent",
#     ],
#     # "Network Control Theory - Control Energy": [
#     #     "nct_energies_total",
#     #     "nct_energies_std", 
#     #     "nct_energies_max",
#     #     "nct_energies_n_nodes_90_percent",
#     #     "nct_energies_n_nodes_50_percent",
#     #     "nct_energies_n_nodes_10_percent",
#     # ],
#     "Computational Capacity": [
#         "ipc_ipc_deg1_mean",
#         "ipc_ipc_deg2_mean",
#         "computational_capacity_memory_capacity_total",
#         # "computational_capacity_memory_timescale",
#         "computational_capacity_nonlinear_capacity_total",
#         # "computational_capacity_cubic_capacity_total",
#         # "computational_capacity_cross_capacity_total",
#         # "computational_capacity_memory_nonlinear_ratio",
#         # "computational_capacity_total_capacity",
#         # "computational_capacity_state_dimensionality", 
#         # "computational_capacity_state_entropy",
#         # "computational_capacity_state_rank",
#         # "computational_capacity_separation_ratio",
#         # "computational_capacity_lyapunov_exponent",
#     ],
#     # "Reservoir Computing - Basic Measures": 
#     # "Memory Capacity - Full Lag Profile": [
#     #     # [f"mc_{i}" for i in range(1, 50)]
#     #     # "mc_mean" # , "mc_std"]
#     # ],
# }




CLUSTERS = CLUSTERS_SUBSELECTION # choose which one to use for the analysis

selected_properties = {
    "Wiring Economy": 
        "proportion_long_range_connections_0.3956",
    "Global Integration": 
        "global_efficiency",
    "Local Segregation": 
        "modularity",
    "Communication Efficiency": 
        "spectral_gap",
    "Computational Capacity": 
        "ipc_ipc_deg1_mean",
    "Robustness": 
        "targeted_attack_robustness_rob_targeted_auc",
}



# color_arr = [# "#c8c8c8", 
            #  "#59a89c", 
            #  "#0b81a2",
            #  "#e25759", "#9d2c00", "#7E4794",
            #  "#f0c571", 
            #  "#36b700"]
# okabe_ito = color_arr # not really, but who cares. 
okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']

CLUSTER_COLORS = {
    "Wiring Economy": okabe_ito[0],
    "Global Integration": okabe_ito[1],
    "Local Segregation": okabe_ito[2],
    "Communication Efficiency": okabe_ito[3],
    "Computational Capacity": okabe_ito[4],
    "Robustness": okabe_ito[5],
    # "Wiring Economy":             "#E07B39",
    # "Global Integration":         "#2C7D8F",
    # "Local Segregation":          "#5B9E6A",
    # "Hub Architecture & Resilience": "#8B5EA4",
    # "Dynamical Regime":           "#C74800",
    # "Computational Capacity":     "#3A6DB5",
}


# ---------------------------------------------------------------------------
# Data loading — mirrors the notebook's load_and_merge_datasets
# ---------------------------------------------------------------------------

def load_huge_df():
    with open(DATA_PATH, "rb") as f:
        datasets_dict = pickle.load(f)
    with open(ETA_GAMMA_PATH, "rb") as f:
        df_eta_gamma = pickle.load(f)

    frames = []
    for dataset_name in ORDERED_DATASETS:
        if dataset_name not in datasets_dict:
            print(f"  Warning: {dataset_name} not in pickle, skipping.")
            continue
        df = datasets_dict[dataset_name].copy()
        df["dataset"] = dataset_name
        if dataset_name == "hcp_schaefer_100_dataset_gnm":
            df = df.join(df_eta_gamma[["eta", "gamma"]])
        frames.append(df)

    huge_df = pd.concat(frames, ignore_index=True)
    return huge_df


def get_active_features(columns, include_mc=True):
    """
    Return only the 38 cluster-assigned features that are present in the DataFrame.
    Restricting to these ensures the 6-cluster decomposition is exact (residual = 0)
    and avoids overflow from unscreened features with extreme values.
    """
    selected = [f for cluster_cols in CLUSTERS.values() for f in cluster_cols]
    return [c for c in selected if c in columns]


# ---------------------------------------------------------------------------
# PCA — same procedure as run_pca_and_correlate in the notebook
# ---------------------------------------------------------------------------

def fit_pca(huge_df, active_features, n_components=10):
    df_active = huge_df[active_features].copy()
    df_active.replace([np.inf, -np.inf], np.nan, inplace=True)
    dropped = df_active.columns[df_active.isnull().any()].tolist()
    if dropped:
        print(f"  Dropped {len(dropped)} columns with NaN/inf: {dropped[:5]}{'...' if len(dropped)>5 else ''}")
    df_active.dropna(axis=1, inplace=True)
    used_features = df_active.columns.tolist()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_active)

    # Drop features where scaling produced non-finite values (e.g. zero-variance columns)
    finite_cols = np.all(np.isfinite(scaled_data), axis=0)
    if not finite_cols.all():
        bad = [used_features[i] for i, ok in enumerate(finite_cols) if not ok]
        print(f"  Dropping {len(bad)} features with non-finite scaled values: {bad}")
        scaled_data = scaled_data[:, finite_cols]
        used_features = [f for f, ok in zip(used_features, finite_cols) if ok]

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    print("\n--- PCA Explained Variance ---")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var*100:.1f}%")

    return pca, pca_result, scaled_data, used_features


# ---------------------------------------------------------------------------
# Cluster contribution decomposition
# ---------------------------------------------------------------------------

def compute_cluster_contributions(scaled_data, pca, used_features, n_pcs=3):
    """
    Decompose each network's PC score into 6 cluster contributions.

    Returns a DataFrame with columns:
        pc{k}_total, pc{k}_{cluster_name}, ... for k in 1..n_pcs
    Contributions sum exactly to pc{k}_total for every row.
    """
    feature_index = {f: i for i, f in enumerate(used_features)}

    rows = {}

    for pc_idx in range(n_pcs):
        loadings = pca.components_[pc_idx]          # shape (n_used_features,)
        total    = scaled_data @ loadings            # shape (n_networks,) — exact PC scores

        col_prefix = f"pc{pc_idx+1}"
        rows[f"{col_prefix}_total"] = total

        residual = total.copy()
        for cluster_name, cluster_cols in CLUSTERS.items():
            in_pca = [c for c in cluster_cols if c in feature_index]
            missing = [c for c in cluster_cols if c not in feature_index]
            if missing:
                print(f"  PC{pc_idx+1} | {cluster_name}: {len(missing)} properties not in PCA "
                      f"(dropped due to NaN or not in active_features): {missing}")

            col_key = f"{col_prefix}_{cluster_name.lower().replace(' ', '_').replace('&', 'and')}"
            if in_pca:
                idx_in_pca = [feature_index[c] for c in in_pca]
                contrib = (scaled_data[:, idx_in_pca] * loadings[idx_in_pca]).sum(axis=1)
            else:
                contrib = np.zeros(scaled_data.shape[0])
            rows[col_key] = contrib
            residual -= contrib

        # Sanity check: residual should be ~0 for all properties assigned
        max_residual = np.abs(residual).max()
        if max_residual > 1e-8:
            print(f"  Warning: PC{pc_idx+1} unassigned residual max={max_residual:.2e} "
                  f"(some used_features not in any cluster)")
        rows[f"{col_prefix}_unassigned"] = residual

    df_contrib = pd.DataFrame(rows)
    return df_contrib


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

DATASET_LABELS = {k: v for k, v in LABEL_MAP.items() if k in ORDERED_DATASETS}

def _shorten(name):
    return DATASET_LABELS.get(name, name)


def plot_mean_contributions(df_contrib, huge_df, output_path):
    """
    For PC1 and PC2: grouped stacked bar chart of mean cluster contributions per dataset.
    Positive and negative parts drawn separately so bars sum to the total PC mean.
    """
    cluster_cols_base = [
        k.lower().replace(" ", "_").replace("&", "and") for k in CLUSTERS
    ]

    datasets = [d for d in ORDERED_DATASETS if d in huge_df["dataset"].values]
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, 2, figsize=viz.cm_to_inch((18, 8)), sharey=False)

    for ax, pc_idx in zip(axes, [1, 2]):
        prefix   = f"pc{pc_idx}"
        col_keys = [f"{prefix}_{b}" for b in cluster_cols_base]
        labels   = list(CLUSTERS.keys())
        colors   = [CLUSTER_COLORS[k] for k in CLUSTERS]

        x = np.arange(n_ds)
        width = 0.6

        for ds_i, dataset in enumerate(datasets):
            mask  = (huge_df["dataset"] == dataset).values
            means = df_contrib.loc[mask, col_keys].mean().values  # (6,)

            pos = np.clip(means, 0, None)
            neg = np.clip(means, None, 0)

            bottom_pos = np.zeros(1)
            bottom_neg = np.zeros(1)
            for ci, (val_p, val_n, color) in enumerate(zip(pos, neg, colors)):
                ax.bar(ds_i, val_p, width, bottom=bottom_pos, color=color,
                       label=labels[ci] if ds_i == 0 else None, alpha=0.85, linewidth=0)
                ax.bar(ds_i, val_n, width, bottom=bottom_neg, color=color, alpha=0.85, linewidth=0)
                bottom_pos += val_p
                bottom_neg += val_n

            # Total mean as black dot
            total_mean = df_contrib.loc[mask, f"{prefix}_total"].mean()
            ax.scatter(ds_i, total_mean, color="black", s=20, zorder=5)

        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([_shorten(d) for d in datasets], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(f"Mean PC{pc_idx} contribution")
        ax.set_title(f"PC{pc_idx} cluster contributions")
        ax.spines[["top", "right"]].set_visible(False)

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLUSTER_COLORS[k], alpha=0.85) for k in CLUSTERS]
    fig.legend(handles, list(CLUSTERS.keys()), loc="lower center",
               ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_contribution_fractions(df_contrib, huge_df, output_path):
    """
    Heatmap: for each dataset × PC, what fraction of |PC score| variance
    comes from each cluster (based on mean squared contribution).
    """
    cluster_cols_base = [
        k.lower().replace(" ", "_").replace("&", "and") for k in CLUSTERS
    ]
    datasets = [d for d in ORDERED_DATASETS if d in huge_df["dataset"].values]

    n_pcs = 3
    fig, axes = plt.subplots(1, n_pcs, figsize=viz.cm_to_inch((18, 6)))

    for pc_idx, ax in enumerate(axes):
        prefix   = f"pc{pc_idx+1}"
        col_keys = [f"{prefix}_{b}" for b in cluster_cols_base]
        labels   = list(CLUSTERS.keys())

        # Fraction of total variance (sum of squared contributions, normalised)
        matrix = np.zeros((len(datasets), len(CLUSTERS)))
        for ds_i, dataset in enumerate(datasets):
            mask   = (huge_df["dataset"] == dataset).values
            contribs = df_contrib.loc[mask, col_keys].values   # (n_networks, 6)
            sq_means = (contribs ** 2).mean(axis=0)            # mean squared contribution
            denom    = sq_means.sum()
            matrix[ds_i] = sq_means / denom if denom > 0 else sq_means

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(len(CLUSTERS)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels([_shorten(d) for d in datasets], fontsize=7)
        ax.set_title(f"PC{pc_idx+1}", fontsize=9)

        for i in range(len(datasets)):
            for j in range(len(CLUSTERS)):
                ax.text(j, i, f"{matrix[i,j]:.0%}", ha="center", va="center",
                        fontsize=5, color="white" if matrix[i,j] > 0.6 else "black")

        fig.colorbar(im, ax=ax, shrink=0.6, label="Fraction of variance")

    plt.suptitle("Cluster contribution fractions to each PC axis", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    huge_df = load_huge_df()
    print(f"  huge_df: {huge_df.shape}")

    raw_cols       = [c for c in huge_df.columns if c not in ("dataset", "color_dataset", "eta", "gamma")]
    active_features = get_active_features(huge_df.columns, include_mc=INCLUDE_MC_IN_PCA)
    print(f"  Active features: {len(active_features)}")

    print("\nFitting PCA...")
    pca, pca_result, scaled_data, used_features = fit_pca(huge_df, active_features)

    print("\nComputing cluster contributions...")
    df_contrib = compute_cluster_contributions(scaled_data, pca, used_features, n_pcs=3)
    df_contrib["dataset"] = huge_df["dataset"].values

    # Sanity check: max deviation of sum-of-contributions from total PC score
    for pc_idx in range(1, 4):
        prefix = f"pc{pc_idx}"
        cluster_cols_base = [k.lower().replace(" ", "_").replace("&", "and") for k in CLUSTERS]
        contrib_sum = df_contrib[[f"{prefix}_{b}" for b in cluster_cols_base]].sum(axis=1)
        total       = df_contrib[f"{prefix}_total"]
        max_err     = (contrib_sum - total).abs().max()
        print(f"  PC{pc_idx} max decomposition error: {max_err:.2e} (should be ~0 if all features assigned)")

    # Save results
    csv_path = OUT_DIR / "cluster_contributions.csv"
    df_contrib.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    pkl_path = OUT_DIR / "cluster_contributions.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "df_contrib":    df_contrib,
            "pca":           pca,
            "used_features": used_features,
            "clusters":      CLUSTERS,
        }, f)
    print(f"  Saved: {pkl_path}")

    print("\nPlotting...")
    plot_mean_contributions(df_contrib, huge_df, OUT_DIR / "cluster_contributions_bars.pdf")
    plot_contribution_fractions(df_contrib, huge_df, OUT_DIR / "cluster_contributions_heatmap.pdf")

    # Print summary table
    cluster_cols_base = [k.lower().replace(" ", "_").replace("&", "and") for k in CLUSTERS]
    print("\n=== Mean PC1 contributions per dataset ===")
    datasets = [d for d in ORDERED_DATASETS if d in huge_df["dataset"].values]
    header = f"{'Dataset':40s}" + "".join(f"{k[:8]:>10s}" for k in CLUSTERS) + f"{'Total':>8s}"
    print(header)
    for dataset in datasets:
        mask   = (huge_df["dataset"] == dataset).values
        means  = df_contrib.loc[mask, [f"pc1_{b}" for b in cluster_cols_base]].mean().values
        total  = df_contrib.loc[mask, "pc1_total"].mean()
        row    = f"{_shorten(dataset):40s}" + "".join(f"{m:+10.3f}" for m in means) + f"{total:+8.3f}"
        print(row)

    print("\nDone.")
