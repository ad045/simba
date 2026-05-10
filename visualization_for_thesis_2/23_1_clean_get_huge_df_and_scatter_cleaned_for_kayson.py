import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from vizman import viz

# ==========================================
# 1. Configuration
# ==========================================

OUTPUT_BASE  = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output")
OUTPUT_FOLDER = OUTPUT_BASE / "00_trade_off_analysis"
OUTPUT_FOLDER.mkdir(exist_ok=True)

GNM_DATASET    = "hcp_schaefer_100_dataset"
GNM_EXPERIMENT = "11_mst_2500_animal_0"

# Datasets 
DATASETS_TO_PLOT = [
    "hcp_schaefer_100_dataset_gnm",
    "suarez_MaMI_dataset",
    "lexis_data_developing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
    "ring_lattice_networks",
    "erdos_renyi_networks"
]

# Columns that are never useful
_META_COLUMNS = [
    "distance_relationship_type", "generative_rule", "num_iterations",
    "preferential_relationship_type", "id", "network_idx", "network_index",
    "mc_values_for_indiv_lags",
]
# Profile/array columns and columns with systematic NaN issues
_ALWAYS_DROP = [
    "repertoire_sweep_T_vec", "repertoire_sweep_sizes", "repertoire_sweep_diversities",
    "repertoire_sweep_weighted_by_distances_T_vec",
    "repertoire_sweep_weighted_by_distances_sizes",
    "repertoire_sweep_weighted_by_distances_diversities",
    "repertoire_sweep_T_critical", "repertoire_sweep_size_critical",
    "repertoire_sweep_diversity_critical",
    "departure_from_normality",
    "proportion_long_range_connections_0.356",
    "repertoire_diversity", "repertoire_size",
    "computational_capacity_notebook_memory_capacity_profile_notebook",
    "computational_capacity_notebook_nonlinear_capacity_profile_notebook",
    "computational_capacity_notebook_damicelli_memory_capacity_profile_notebook",
    "computational_capacity_notebook_damicelli_nonlinear_capacity_profile_notebook",
]

# GNM columns that are constant across all networks (uninformative in the η-γ space)
_GNM_CONSTANT_COLUMNS = [
    "avg_degree", "n_connected_components", "density", "density_bct",
    "kernel_rank_phase_of_lambda_max",
]

# Categories. One figure per category will be produced.  Columns absent from the data are silently skipped.
CATEGORIES = {
    "Fundamental Topology": [
        "transitivity", "avg_clustering", "modularity", "degree_gini",
        "degree_assortativity", "omega", "structural_complexity",
        "directed_simplices_count", "directed_simplices_max_size",
        "gromov_wasserstein",
        "eta", "gamma",
    ],
    "Paths, Efficiency & Communication": [
        "char_path_length", "global_efficiency", "local_efficiency",
        "diffusion_efficiency", "propagation_efficiency", "avg_communicability",
        "betweenness_centrality_mean",
        "topological_distance_mean", "topological_distance_std",
        "resistance_distance_mean", # ??
        "resistance_distance_std", 
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
        "rich_club_coefficient_rc_max_norm",
        "rich_club_coefficient_rc_mean_norm",
        "rich_club_coefficient_rc_k_at_max",
        "rich_club_coefficient_rc_regime_frac",
        "rich_club_coefficient_rc_weighted_auc",
    ],
    "Spectral Properties & Algebraic Connectivity": [
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
        "participation_coefficient_pc_mean",
        "participation_coefficient_pc_median",
        "participation_coefficient_pc_std",
        "participation_coefficient_pc_frac_connector",
        "participation_coefficient_wmd_std",
    ],
    "Ollivier-Ricci Curvature": [
        "ollivier_ricci_curvature_orc_mean",
        "ollivier_ricci_curvature_orc_median",
        "ollivier_ricci_curvature_orc_std",
        "ollivier_ricci_curvature_orc_min",
        "ollivier_ricci_curvature_orc_max",
        "ollivier_ricci_curvature_orc_skewness",
        "ollivier_ricci_curvature_orc_frac_neg",
    ],
    "Persistent Homology (TDA)": [
        "persistent_homology_ph_h1_n_features",
        "persistent_homology_ph_h1_persistence_mean",
        "persistent_homology_ph_h1_entropy",
        "persistent_homology_ph_total_persistence",
    ],
    "Targeted Attack Robustness & Vulnerability": [
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
        "ipc_ipc_deg1_mean", "ipc_ipc_deg1_std",
        "ipc_ipc_deg2_mean", "ipc_ipc_deg2_std",
    ],
    "Betweenness Centrality": [
        "betweenness_centrality_stats_mean",
        "betweenness_centrality_stats_std",
        "betweenness_centrality_stats_max",
        "betweenness_centrality_stats_gini",
    ], 
    "Local Efficiency": [
        "local_efficiency_stats_mean",
        "local_efficiency_stats_std",
        "local_efficiency_stats_min",
        "local_efficiency_stats_max",
    ], 
    "Hyperbolicity": [
        "gromov_hyperbolicity",
    ]
}

# Colors general (stolen from Kayson) 
COLORS = {
    "neutrals": {
        "BONE_WHITE": "#FFFCF2",
        "ALL_WHITE": "#FFFFFF",
        "GRAY": "#b8b8b8",
        "OLIVE_GRAY": "#6A7870",
        "HALF_BLACK": "#232324",
    },
    "colds": {
        "INK_BLUE": "#0F14F7",
        "NIGHT_BLUE": "#394D73",
        "DEEP_BLUE": "#006685",
        "LAKE_BLUE": "#3FA5C4",
        "TEAL": "#44cfcf",
    },
    "warms": {
        "SAND": "#E1C5A2",
        "YELLOW": "#E6B213",
        "ORANGE": "#F99465",
        "LECKER_RED": "#E84653",
        "DEEP_RED": "#BF003F",
    },
    "purples": {
        "PINK": "#EFBBD3",
        "PURPLE": "#A6587C",
        "PURPLER": "#591154",
        "PURPLEST": "#260126",
    },
    "greens": {
        "GRASS_GREEN": "#5DC400",
        "SLOW_GREEN": "#a1d4ca",
        "JUST_GREEN": "#4BAE6A",
        "LUXUARY_GREEN": "#275036",
        "ULTRA_GREEN": "#52FF94",
    },
}

# Colors datasets 
COLOR_SCHEME = {
    "hcp_schaefer_100_dataset_gnm": "#232324",       # HALF_BLACK
    "hcp_schaefer_100_dataset": "#4BAE6A",            # DEEP_BLUE
    "kaysons_generated_networks_diffusion":  "#F7BE18", # #FFFF2E", # #46BD16", # 44cfcf", # "#E6B213", # YELLOW
    "kaysons_generated_networks_propagation":"#262F3F", # #127F55", # #8E18BC", # BF003F", # Deep red "#E84653", # LECKER_RED
    "kaysons_generated_networks_routing": "#6E3AA3", #8F443D", #1E73B8", #BD901A", # 2B4AC3",  # JUST_GREEN
    # "kaysons_generated_networks_resistance": "#44cfcf", # TEAL
    "kaysons_generated_networks_topology": "#394D73", # NIGHT_BLUE
    "ring_lattice_networks": "#1a1f1f",   # TEAL — economy anchor (short-range only)
    "erdos_renyi_networks":  "#B8B8B8",   # GRAY — random null baseline
    "suarez_MaMI_dataset": "#799372", # "#4396B0",  #1847BD", # 006685" , # ORANGE "#A6587C",                 # PURPLE or: "#591154", # PURPLER # 
    "lexis_data_young": "#BD3B1A", ##5DC400",
    "lexis_data_aging": "#5E1E0C", # E8D43C",
    "lexis_data_developing": "#C74800", # the youngest yellow: E6B212", # #000100", # #A6587C",
    "lexis_data_developing_consensus_per_age_1_year": "#A6587C", # PURPLE
    "lexis_data_young_consensus_per_age_1_year": "#5DC400", # GREEN
    "lexis_data_aging_consensus_per_age_1_year": "#E8D43C", # YELLOW
    "lexis_data_all_consensus_per_age_1_year": "#4BAE6A", # DEEP_BLUE
    "lexis_data_all_consensus_per_age_2_year": "#4BAE6A", # DEEP_BLUE
}

# Labels datasets
LABEL_MAP = {
    "hcp_schaefer_100_dataset_gnm": "HCP (GNM)",
    "hcp_schaefer_100_dataset": "HCP (Empirical)",
    "kaysons_generated_networks_diffusion": "Diffusion",
    "kaysons_generated_networks_propagation": "Propagation",
    "kaysons_generated_networks_routing": "Routing",
    "kaysons_generated_networks_resistance": "Resistance",
    "kaysons_generated_networks_topology": "Topology",
    "ring_lattice_networks": "Ring Lattice",
    "erdos_renyi_networks":  "Erdős-Rényi",
    "suarez_MaMI_dataset": "MaMI",
    "lexis_data_young": "Young", 
    "lexis_data_aging": "Aging",
    "lexis_data_developing": "Developing",
    "lexis_data_developing_consensus_per_age_1_year": "Developing (Consensus, 1 Year)",
    "lexis_data_young_consensus_per_age_1_year": "Young (Consensus, 1 Year)",
    "lexis_data_aging_consensus_per_age_1_year": "Aging (Consensus, 1 Year)",
    "lexis_data_all_consensus_per_age_1_year": "All (Consensus, 1 Year)",
    "lexis_data_all_consensus_per_age_2_year": "All (Consensus, 2 Year)",
}

# Property names 
PROPERTY_NAMES = {
    "algebraic_connectivity_fiedler_bipartition_balance": "Fiedler Bipartition Balance",
    "algebraic_connectivity_fiedler_value": "Fiedler Value",
    "algebraic_connectivity_fiedler_value_norm": "Fiedler Value (Normalized)",
    "algebraic_connectivity_laplacian_spectral_gap": "Laplacian Spectral Gap",
    "algebraic_connectivity_nx": "Algebraic Connectivity",
    "avg_clustering": "Clustering",
    "avg_communicability": "Communicability",
    "avg_edge_distance": "Edge Distance (Mean)",
    "char_path_length": "Char. Path Length",
    "community_synchronization_vulnerability_n_communities": "Number Communities",
    "community_synchronization_vulnerability_value": "Synchronization Vulnerability",
    "computational_capacity_cross_capacity_total": "Cross Capacity",
    "computational_capacity_cubic_capacity_total": "Cubic Capacity",
    "computational_capacity_lyapunov_exponent": "Lyapunov Exponent",
    "computational_capacity_memory_capacity_total": "Memory Capacity",
    "computational_capacity_memory_nonlinear_ratio": "Memory to Nonlinear Ratio",
    "computational_capacity_memory_timescale": "Memory Timescale",
    "computational_capacity_nonlinear_capacity_total": "Nonlinear Capacity",
    "computational_capacity_separation_ratio": "Separation Ratio",
    "computational_capacity_state_dimensionality": "State Dimensionality",
    "computational_capacity_state_entropy": "State Entropy",
    "computational_capacity_state_rank": "State Rank",
    "computational_capacity_total_capacity": "Total Computational Capacity",
    "degree_assortativity": "Degree Assortativity",
    "degree_gini": "Degree Gini Coefficient",
    "departure_from_normality_schur": "Departure from Normality",
    "diffusion_efficiency": "Diffusion Eff.",
    "directed_simplices_count": "Directed Simplices Count",
    "directed_simplices_max_size": "Directed Simplices Max Size",
    "effective_dimensionality": "Effective Dimensionality",
    "energy": "Energy",
    "eta": "$\eta$",
    "gamma": "$\gamma$",
    "global_efficiency": "Global Efficiency",
    "kernel_rank_max": "Kernel Rank (Max)",
    "kernel_rank_phase_diff_of_lambda_max_and_2nd": "Kernel Rank: Phase Diff (λ_max vs 2nd)",
    "kernel_rank_thresholded_and_summed_0.01": "Kernel Rank (Thresholded, 0.01)",
    "kuramoto_averaged_synchronization_r_final": "Kuramoto Avg. Sync. r (Final)",
    "kuramoto_averaged_synchronization_r_mean": "Kuramoto Avg. Sync. r (Mean)",
    "kuramoto_averaged_synchronization_r_mean_se": "Kuramoto Avg. Sync. r (Mean SE)",
    "kuramoto_averaged_synchronization_r_std": "Kuramoto Avg. Sync. r (Std)",
    "kuramoto_averaged_synchronization_r_std_se": "Kuramoto Avg. Sync. r (Std SE)",
    "kuramoto_synchronization_r_final": "Kuramoto Sync. r (Final)",
    "kuramoto_synchronization_r_mean": "Kuramoto Sync. r (Mean)",
    "kuramoto_synchronization_r_std": "Kuramoto Sync. r (Std)",
    "mc_1": "MC (Lag 1)",
    "mc_10": "MC (Lag 10)",
    "mc_11": "MC (Lag 11)",
    "mc_12": "MC (Lag 12)",
    "mc_13": "MC (Lag 13)",
    "mc_14": "MC (Lag 14)",
    "mc_15": "MC (Lag 15)",
    "mc_16": "MC (Lag 16)",
    "mc_17": "MC (Lag 17)",
    "mc_18": "MC (Lag 18)",
    "mc_19": "MC (Lag 19)",
    "mc_2": "MC (Lag 2)",
    "mc_20": "MC (Lag 20)",
    "mc_21": "MC (Lag 21)",
    "mc_22": "MC (Lag 22)",
    "mc_23": "MC (Lag 23)",
    "mc_24": "MC (Lag 24)",
    "mc_25": "MC (Lag 25)",
    "mc_26": "MC (Lag 26)",
    "mc_27": "MC (Lag 27)",
    "mc_28": "MC (Lag 28)",
    "mc_29": "MC (Lag 29)",
    "mc_3": "MC (Lag 3)",
    "mc_30": "MC (Lag 30)",
    "mc_31": "MC (Lag 31)",
    "mc_32": "MC (Lag 32)",
    "mc_33": "MC (Lag 33)",
    "mc_34": "MC (Lag 34)",
    "mc_35": "MC (Lag 35)",
    "mc_36": "MC (Lag 36)",
    "mc_37": "MC (Lag 37)",
    "mc_38": "MC (Lag 38)",
    "mc_39": "MC (Lag 39)",
    "mc_4": "MC (Lag 4)",
    "mc_40": "MC (Lag 40)",
    "mc_41": "MC (Lag 41)",
    "mc_42": "MC (Lag 42)",
    "mc_43": "MC (Lag 43)",
    "mc_44": "MC (Lag 44)",
    "mc_45": "MC (Lag 45)",
    "mc_46": "MC (Lag 46)",
    "mc_47": "MC (Lag 47)",
    "mc_48": "MC (Lag 48)",
    "mc_49": "MC (Lag 49)",
    "mc_5": "MC (Lag 5)",
    "mc_6": "MC (Lag 6)",
    "mc_7": "MC (Lag 7)",
    "mc_8": "MC (Lag 8)",
    "mc_9": "MC (Lag 9)",
    "mc_mean": "MC (Mean)",
    "mc_std": "MC (Std)",
    "mc_nonlinear_original_mc_mean": "Non-Linear (Mean)",
    "mc_nonlinear_original_mc_std": "Non-Linear (Std)",
    "mc_input_scaling_0_1_mc_mean": "MC Linear",
    "mc_input_scaling_0_1_mc_std": "MC Input Scaling 0.1 (Std)",
    "mc_nonlinear_input_scaling_0_1_mc_mean": "MC Non-Linear",
    "mc_nonlinear_input_scaling_0_1_mc_std": "MC Non-Linear Input Scaling 0.1 (Std)",
    "mc_lin_cut40": "MC Linear Cut 40",
    "mc_nonlin_cut40": "Computational: MC Non-Linear Cut 40",
    "modularity": "Modularity",
    "nct_control_avg": "NCT: Average Controllability",
    "nct_control_max": "NCT: Max Controllability",
    "nct_control_n_nodes_10_percent": "NCT: N Nodes (Top 10%)",
    "nct_control_n_nodes_50_percent": "NCT: N Nodes (Top 50%)",
    "nct_control_n_nodes_90_percent": "NCT: N Nodes (Top 90%)",
    "nct_control_std": "Controllability (std)",
    "nct_energies_max": "NCT: Max Control Energy",
    "nct_energies_n_nodes_10_percent": "NCT: Energy N Nodes (Top 10%)",
    "nct_energies_n_nodes_50_percent": "NCT: Energy N Nodes (Top 50%)",
    "nct_energies_n_nodes_90_percent": "NCT: Energy N Nodes (Top 90%)",
    "nct_energies_std": "NCT: Control Energy Std",
    "nct_energies_total": "NCT: Total Control Energy",
    "ollivier_ricci_curvature_orc_frac_neg": "ORC (fraction negative)",
    "ollivier_ricci_curvature_orc_max": "ORC (max)",
    "ollivier_ricci_curvature_orc_mean": "ORC (mean)",
    "ollivier_ricci_curvature_orc_median": "ORC (median)",
    "ollivier_ricci_curvature_orc_min": "ORC (min)",
    "ollivier_ricci_curvature_orc_skewness": "ORC (skewness)",
    "ollivier_ricci_curvature_orc_std": "ORC (std)",
    "omega": "Omega",
    "participation_coefficient_n_communities": "N Communities",
    "participation_coefficient_n_communities_further": "N Communities",
    "participation_coefficient_pc_frac_connector": "Fraction Connector Hubs",
    "participation_coefficient_pc_frac_connector_further": "Fraction Connector Hubs",
    "participation_coefficient_pc_mean": "Participation Coeff. (mean)",
    "participation_coefficient_pc_mean_further": "Participation Coeff. (mean)",
    "participation_coefficient_pc_median": "Participation Coeff. (median)",
    "participation_coefficient_pc_median_further": "Participation Coeff. (median)",
    "participation_coefficient_pc_std": "Participation Coeff. (std)",
    "participation_coefficient_pc_std_further": "Participation Coeff. (std)",
    "participation_coefficient_wmd_mean": "Within-Module Degree (mean)",
    "participation_coefficient_wmd_mean_further": "Within-Module Degree (mean, further)",
    "participation_coefficient_wmd_std": "Within-Module Degree (std)",
    "participation_coefficient_wmd_std_further": "Within-Module Degree (std, further)",
    "persistent_homology_ph_h1_entropy": "Persistent Homology H1: Entropy",
    "persistent_homology_ph_h1_n_features": "Persistent Homology H1: N Features",
    "persistent_homology_ph_h1_persistence_mean": "Persistent Homology H1: Mean Persistence",
    "persistent_homology_ph_total_persistence": "Persistent Homology: Total Persistence",
    "propagation_efficiency": "Propagation Eff.",
    "proportion_long_range_connections_0.1": r"$f_{\rm LR\,(10\%)}$",
    "proportion_long_range_connections_0.3": r"$f_{\rm LR\,(30\%)}$",
    "proportion_long_range_connections_0.356": r"$f_{\rm LR\,(35.6\%)}$",
    "proportion_long_range_connections_0.365": r"$f_{\rm LR\,(36.5\%)}$",
    "proportion_long_range_connections_0.3956": "Fraction LR connections", # $f_{\rm LR\,(39.56\%)}$",
    "proportion_long_range_connections_0.5": r"$f_{\rm LR\,(50\%)}$",
    "repertoire_sweep_weighted_by_distances_T_critical": "Metastability (critical T)",
    "repertoire_sweep_weighted_by_distances_diversity_critical": "Metastability (critical diversity)",
    "repertoire_sweep_weighted_by_distances_size_critical": "Metastability (critical size)",
    "rich_club_coefficient_rc_k_at_max": "Rich-Club Coeff.: k at Max",
    "rich_club_coefficient_rc_max_norm": "Rich-Club Coeff.: Max (Normalized)",
    "rich_club_coefficient_rc_mean_norm": "Rich-Club Coeff.: Mean (Normalized)",
    "rich_club_coefficient_rc_regime_frac": "Rich-Club Coeff.: Regime Fraction",
    "rich_club_coefficient_rc_weighted_auc": "Rich-Club Coeff.: Weighted AUC",
    "richclub_avg_length": "Rich-Club: Average Path Length",
    "richclub_n_edges": "Rich-Club: N Edges",
    "spectral_gap": "Spectral Gap",
    "spectral_gap_fatemeh": "Spectral Gap",
    "spectral_radius": "Spectral Radius",
    "structural_complexity": "Structural Complexity",
    "synchronizability_eigenratio_eigenratio": "Synchronizability: Eigenratio",
    "synchronizability_eigenratio_lambda_2": "Synchronizability: λ₂ (Fiedler)",
    "synchronizability_eigenratio_lambda_N": "Synchronizability: λ_N (Largest)",
    "targeted_attack_robustness_rob_random_auc": "Robustness (Random)",
    "targeted_attack_robustness_rob_random_half": "Robustness: Random Attack Half",
    "targeted_attack_robustness_rob_ratio": "Robustness: Targeted/Random Ratio",
    "targeted_attack_robustness_rob_targeted_auc": "Robustness (Targeted)",
    "targeted_attack_robustness_rob_targeted_half": "Robustness: Targeted Attack Half",
    "topological_distance_mean": "Topological Distance (Mean)",
    "topological_distance_std": "Topological Distance (Std)",
    "transitivity": "Transitivity",
    "wiring_cost": "Wiring Cost",
    "ipc_ipc_deg1_mean": "MC Linear",
    "ipc_ipc_deg1_std": "IPC: Deg 1 (Std)",
    "ipc_ipc_deg2_mean": "MC Non-Linear",
    "ipc_ipc_deg2_std": "IPC: Deg 2 (Std)",
    "gromov_hyperbolicity": "Gromov Hyperbolicity",
    "betweenness_centrality_stats_mean": "Betweenness Centrality (Mean)",
    "betweenness_centrality_stats_std": "Betweenness Centrality (Std)",
    "betweenness_centrality_stats_max": "Betweenness Centrality (Max)",
    "betweenness_centrality_stats_gini": "Betweenness Centrality (Gini)",
    "local_efficiency_stats_mean": "Local Eff. (Mean)",
    "local_efficiency_stats_std": "Local Eff. (Std)",
    "local_efficiency_stats_min": "Local Eff. (Min)",
    "local_efficiency_stats_max": "Local Eff. (Max)"
}

# Experimental folders where all the empirical datasets are stored: I know I know, this will prob suck to adapt. Sorry! 
emp_dataset_and_experiment_pairs = {
                    "hcp_schaefer_100_dataset_gnm": "11_mst_2500_animal_0", 
                    "hcp_schaefer_100_dataset": "05_mst_animal_0_compared_with_hcp_schaefer_100", 
                    "kaysons_generated_networks_diffusion": "05_mst_animal_0_compared_with_diffusion", 
                    "kaysons_generated_networks_propagation": "05_mst_animal_0_compared_with_propagation", 
                    #  "kaysons_generated_networks_resistance": "05_mst_animal_0_compared_with_resistance", 
                    "kaysons_generated_networks_routing": "05_mst_animal_0_compared_with_routing", 
                    "kaysons_generated_networks_topology": "05_mst_animal_0_compared_with_topology", 
                    "ring_lattice_networks": "05_mst_animal_0_compared_with_ring_lattice_networks",
                    "erdos_renyi_networks":  "05_mst_animal_0_compared_with_erdos_renyi_networks",
                    "suarez_MaMI_dataset": "05_mst_animal_0_compared_with_mami",
                    "lexis_data_developing": "05_mst_animal_0_compared_with_lexis_data_developing", 
                    "lexis_data_young": "05_mst_animal_0_compared_with_lexis_data_young",
                    "lexis_data_aging": "05_mst_animal_0_compared_with_lexis_data_aging",
                    "lexis_data_developing_consensus_per_age_1_year": "00_pca",
                    "lexis_data_young_consensus_per_age_1_year": "00_pca",
                    "lexis_data_aging_consensus_per_age_1_year": "00_pca",
                    "lexis_data_all_consensus_per_age_1_year": "00_pca",
                    "lexis_data_all_consensus_per_age_2_year": "00_pca",
                }

# Properties: Flat list for downstream PCA (excludes eta/gamma and energy)
PRECISE_CATEGORIES = [
    col for cat_cols in CATEGORIES.values()
    for col in cat_cols
    if col not in ("eta", "gamma", "energy")
]


# ==========================================
# 2. Data Loading
# ==========================================

def _drop_standard_columns(df, extra_to_remove=None):
    """Drop meta columns, always-drop columns, h_params columns, and all-NaN columns."""
    to_drop = set(_META_COLUMNS + _ALWAYS_DROP)
    if extra_to_remove:
        to_drop.update(extra_to_remove)
    to_drop.update(c for c in df.columns if c.startswith("h_params"))
    df = df.drop(columns=[c for c in to_drop if c in df.columns])
    df = df.dropna(axis=1, how="all")
    return df


def load_gnm_data():
    """Load and merge the five GNM metric CSVs into one DataFrame."""
    base = OUTPUT_BASE / f"gnm/{GNM_DATASET}/{GNM_EXPERIMENT}"
    files = {
        "fundamental":   f"all_metrics_for_{GNM_EXPERIMENT}.csv",
        "static":        f"all_static_metrics_for_{GNM_EXPERIMENT}_updated.csv",
        "dynamic":       f"all_dynamic_metrics_for_{GNM_EXPERIMENT}_updated.csv",
        "computational": f"all_computational_metrics_for_{GNM_EXPERIMENT}_updated.csv",
        "further":       f"all_further_metrics_for_{GNM_EXPERIMENT}_updated.csv",
    }
    dfs = {k: pd.read_csv(base / f) for k, f in files.items()}

    df = dfs["fundamental"]
    for key in ("static", "dynamic", "computational", "further"):
        df = df.merge(dfs[key], on=["eta", "gamma", "id"], how="left",
                      suffixes=("", f"_{key}"))

    df = df.loc[:, ~df.columns.duplicated()]
    df = _drop_standard_columns(df)
    df = df.drop(columns=[c for c in ("wiring_cost_static",) if c in df.columns])
    return df


def load_empirical_data():
    """Load and merge the four empirical metric CSVs for each non-GNM dataset."""
    dict_emp = {}
    for dataset_name in DATASETS_TO_PLOT:
        if dataset_name == "hcp_schaefer_100_dataset_gnm":
            continue

        experiment_name = emp_dataset_and_experiment_pairs[dataset_name]
        # if "ring" in dataset_name or "erdos" in dataset_name:
        #     base = OUTPUT_BASE / f"gnm/{dataset_name}/{experiment_name}"
        # else: 
        base = OUTPUT_BASE / f"gnm/{dataset_name}/{experiment_name}"
        files = {
            "static":        "metrics_2_static.csv",
            "dynamic":       "metrics_2_dynamic.csv",
            "computational": "metrics_2_computational.csv",
            "further":       "metrics_2_further.csv",
        }
        dfs = {k: pd.read_csv(base / f) for k, f in files.items()}

        df = dfs["static"]
        for key in ("dynamic", "computational", "further"):
            df = df.merge(dfs[key], left_index=True, right_index=True,
                          how="left", suffixes=("", f"_{key}"))

        # Strip legacy prefixes
        df.columns = [c.replace("basic_measures_", "").replace("mc_original_", "")
                      for c in df.columns]

        # Drop exact-duplicate columns, keep first
        seen, keep_idx = {}, []
        for i, col in enumerate(df.columns):
            if col not in seen:
                seen[col] = i
                keep_idx.append(i)
            elif not df.iloc[:, seen[col]].equals(df.iloc[:, i]):
                df.columns.values[i] = f"{col}_dup"
                keep_idx.append(i)
        df = df.iloc[:, keep_idx]

        df = _drop_standard_columns(df)

        # Remove rows with non-singleton connected components
        if "n_connected_components" in df.columns:
            bad = df.index[df["n_connected_components"] != 1]
            if len(bad):
                print(f"  {dataset_name}: removing {len(bad)} non-singleton rows.")
                df = df.drop(index=bad)

        # Fix duplicated wiring_cost (dup has the real euclidean values)
        if "wiring_cost" in df.columns and "wiring_cost_dup" in df.columns:
            df = df.drop(columns=["wiring_cost"])
            df = df.rename(columns={"wiring_cost_dup": "wiring_cost"})

        dict_emp[dataset_name] = df
        
        # Print all property names
        # for c in df.columns: 
        #     print(c) 
            
    return dict_emp


# ==========================================
# 3. Plotting
# ==========================================

def plot_gnm_heatmaps(df_gnm_g, output_folder):
    """One figure per category: η-γ heatmap for each property in that category."""
    n_cols = 4
    for cat_name, cat_cols in CATEGORIES.items():
        present = [c for c in cat_cols if c in df_gnm_g.columns]
        if not present:
            print(f"  Skipping '{cat_name}' – no columns in data.")
            continue

        n_rows = (len(present) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=viz.cm_to_inch((n_cols * 3.5, n_rows * 3.5)),
            sharex=True, sharey=True, dpi=200,
        )
        axes = np.array(axes).reshape(n_rows, n_cols)
        fig.suptitle(cat_name, fontsize=8, fontweight="bold", y=1.01)

        for idx, col in enumerate(present):
            ax = axes[idx // n_cols, idx % n_cols]
            pivoted = df_gnm_g.pivot(index="gamma", columns="eta", values=col)
            im = ax.imshow(pivoted, aspect="equal", origin="lower", cmap="RdBu_r") # managua_r") # RdBu_r")
            ax.set_title(PROPERTY_NAMES[col], fontsize=5)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=4)
            cbar.ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.1e}")
            )

        for idx in range(len(present), n_rows * n_cols):
            fig.delaxes(axes[idx // n_cols, idx % n_cols])

        plt.tight_layout(pad=0.5)
        safe_name = re.sub(r"[^\w]+", "_", cat_name).strip("_").lower()
        out_path = output_folder / f"gnm_heatmaps_{safe_name}.pdf"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_lr_fraction_scatters(dict_with_all_datasets, output_folder):
    """For each dataset: scatter of wiring_cost vs long-range fraction at 4 thresholds."""
    import seaborn as sns

    x_value   = "wiring_cost"
    thresholds = ["0.1", "0.3", "0.3956", "0.5"]
    labels     = ["10%", "30%", "39.65%", "50%"]

    for dataset_name, df in dict_with_all_datasets.items():
        fig, axs = plt.subplots(1, 4, figsize=viz.cm_to_inch((18, 5)),
                                sharex=True, dpi=200)
        for i, (thresh, label) in enumerate(zip(thresholds, labels)):
            col = f"proportion_long_range_connections_{thresh}"
            if col not in df.columns or x_value not in df.columns:
                axs[i].set_visible(False)
                continue

            axs[i].scatter(
                df[x_value], df[col],
                alpha=0.03 if "gnm" in dataset_name else 0.4,
                color=COLOR_SCHEME[dataset_name],
                edgecolor="none", s=2, rasterized=True,
            )
            sns.regplot(x=x_value, y=col, data=df, scatter=False, ax=axs[i],
                        line_kws={"color": COLORS["colds"]["LAKE_BLUE"], "linewidth": 1.2})

            r = df[[x_value, col]].corr().iloc[0, 1]
            axs[i].text(0.95, 0.05, f"$r = {r:.3f}$", transform=axs[i].transAxes,
                        fontsize=7, color=COLORS["colds"]["LAKE_BLUE"],
                        va="bottom", ha="right")
            axs[i].set_xlabel(PROPERTY_NAMES.get(x_value, x_value))
            axs[i].set_ylabel(r"$f_{\\rm LR\\,({label.replace('%', r'\%')})}$")
            axs[i].set_yticks([])

        axs[0].set_ylabel("Long-range fraction\n$f_{\\mathregular{LR}(10\\%)}$")
        plt.suptitle(dataset_name)
        plt.tight_layout()
        out_path = output_folder / f"scatter_{dataset_name}_wiring_cost_vs_lr_fraction.pdf"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def _find_lag_cols(df, prefix):
    """Return columns matching '{prefix}_lag_N' sorted by lag index N."""
    pat = re.compile(rf"^{re.escape(prefix)}_lag_(\d+)$")
    hits = [(int(m.group(1)), c) for c in df.columns if (m := pat.match(c))]
    return [c for _, c in sorted(hits)]


def plot_mc_analysis(mc_per_dataset, dict_with_all_datasets, datasets_with_mc,
                     lin_lag_cols, nonlin_lag_cols, output_folder):
    """Boxplot of linear vs nonlinear MC totals + cumulative R² lag profiles per dataset."""
    n = len(datasets_with_mc)
    # Up to 5 profile panels in a 2-row layout (cols 3+4+5 of a 6-col mosaic)
    panel_keys = [chr(65 + i) for i in range(min(n, 5))]   # "A"…"E"
    n_panels = len(panel_keys)
    # Static mosaic: 3 box columns, 2 panel columns, 1 legend column
    panel_top    = panel_keys[:2] if n_panels >= 2 else panel_keys + ["_"]
    panel_bottom = panel_keys[2:5] if n_panels > 2 else ["_", "_", "_"]
    while len(panel_top)    < 2: panel_top.append("_")
    while len(panel_bottom) < 3: panel_bottom.append("_")
    mosaic = [
        ["box", "box", "box"] + panel_top    + ["leg"],
        ["box", "box", "box"] + panel_bottom,
    ]

    fig, axd = plt.subplot_mosaic(
        mosaic, figsize=viz.cm_to_inch((18, 9)),
        gridspec_kw={"wspace": 0.6, "hspace": 0.4},
    )

    # ── Boxplot ───────────────────────────────────────────────────────────────────
    ax_box = axd["box"]
    for i, dataset in enumerate(datasets_with_mc):
        color = COLOR_SCHEME[dataset]
        for values, hatch, alpha, zorder in [
            (mc_per_dataset[dataset]["lin"],    "",    1.0, 3),
            (mc_per_dataset[dataset]["nonlin"], "///", 0.5, 2),
        ]:
            ax_box.boxplot(
                values, positions=[i], widths=0.5, patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor="black", hatch=hatch, alpha=alpha),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black", alpha=alpha),
                capprops=dict(color="black", alpha=alpha),
                flierprops=dict(marker="o", markeredgecolor="black",
                                markerfacecolor=color, markersize=3, alpha=alpha),
                zorder=zorder,
            )
    ax_box.set_xticks(range(n))
    ax_box.set_xticklabels([LABEL_MAP[d] for d in datasets_with_mc], rotation=30, ha="right")
    ax_box.set_ylabel("Memory capacity")
    ax_box.set_xlim(-0.6, n - 0.4)
    ax_box.spines[["top", "right"]].set_visible(False)

    # ── Cumulative R² profiles ────────────────────────────────────────────────────
    valid_panel_keys = [k for k in panel_keys if k in axd]
    ref_ax = axd[valid_panel_keys[0]]
    for key in valid_panel_keys[1:]:
        axd[key].sharex(ref_ax)
        axd[key].sharey(ref_ax)

    top_keys    = [k for k in panel_top    if k in axd]
    bottom_keys = [k for k in panel_bottom if k in axd]

    for key, dataset in zip(valid_panel_keys, datasets_with_mc):
        ax_p = axd[key]
        color = COLOR_SCHEME[dataset]
        for idx, row in dict_with_all_datasets[dataset].iterrows():
            lin_p    = np.array([row[c] for c in lin_lag_cols],    dtype=np.float32)
            nonlin_p = np.array([row[c] for c in nonlin_lag_cols], dtype=np.float32)
            lags = np.arange(len(lin_p))
            ax_p.fill_between(lags, np.cumsum(lin_p),    color=color, alpha=0.2, linewidth=0)
            ax_p.fill_between(lags, np.cumsum(nonlin_p), facecolor="none",
                              hatch="///", edgecolor=color, linewidth=0.0, alpha=0.3)
            ax_p.plot(lags, np.cumsum(nonlin_p), color=color, linewidth=0.7, alpha=0.7)
            if idx % 10 == 5:
                break
        ax_p.spines[["top", "right"]].set_visible(False)
        # y-label only on left column
        if key == top_keys[0] or key == (bottom_keys[0] if bottom_keys else None):
            ax_p.set_ylabel("Cumulative R²")
        else:
            ax_p.set_yticklabels([])
        # x-label only on bottom row
        if key in bottom_keys:
            ax_p.set_xlabel("Lag")
        else:
            ax_p.set_xticklabels([])

    for key in ["_"] + [chr(65 + i) for i in range(n_panels, 5)]:
        if key in axd:
            axd[key].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────────
    if "leg" in axd:
        axd["leg"].axis("off")
        axd["leg"].legend(
            handles=[
                Patch(facecolor="gray", edgecolor="none", alpha=0.5, label="linear"),
                Patch(facecolor="none", edgecolor="gray", hatch="///", label="nonlinear"),
            ],
            loc="center", framealpha=0.9, title="MC type",
        )

    plt.tight_layout()
    out_path = output_folder / "mc_analysis.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ==========================================
# 4. Main
# ==========================================

if __name__ == "__main__":
    # ── Load data ────────────────────────────────────────────────────────────────
    print("Loading GNM data...")
    df_gnm = load_gnm_data()
    print(f"  GNM: {df_gnm.shape}")

    print("Loading empirical datasets...")
    dict_with_all_datasets = load_empirical_data()
    dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"] = df_gnm

    with open(OUTPUT_FOLDER / "all_datasets.pkl", "wb") as f:
        pickle.dump(dict_with_all_datasets, f)
    print("  Saved all_datasets.pkl")

    # ── Memory capacity analysis (uses individual lag columns; run before numeric drop) ─
    print("Plotting MC analysis...")
    datasets_no_gnm = [d for d in DATASETS_TO_PLOT if "gnm" not in d]

    # Auto-detect lag column prefixes from the first available dataset
    _sample = next((dict_with_all_datasets[d] for d in datasets_no_gnm
                    if d in dict_with_all_datasets), pd.DataFrame())
    lin_lag_cols    = _find_lag_cols(_sample, "ipc_ipc_linear")
    nonlin_lag_cols = _find_lag_cols(_sample, "ipc_ipc_nonlinear")

    if not lin_lag_cols or not nonlin_lag_cols:
        print(f"  Skipping MC analysis (no lag columns found; "
              f"searched for 'ipc_ipc_linear_lag_N' and 'ipc_ipc_nonlinear_lag_N').")
    else:
        print(f"  Found {len(lin_lag_cols)} linear lags, {len(nonlin_lag_cols)} nonlinear lags.")
        mc_per_dataset, datasets_with_mc = {}, []
        for dataset in datasets_no_gnm:
            df = dict_with_all_datasets.get(dataset, pd.DataFrame())
            missing = [c for c in lin_lag_cols + nonlin_lag_cols if c not in df.columns]
            if missing:
                print(f"  Skipping MC for {dataset} ({len(missing)} lag columns absent).")
                continue
            lin_arr, nonlin_arr = [], []
            for idx, row in df.iterrows():
                lin_arr.append(   np.array([row[c] for c in lin_lag_cols],    dtype=np.float32).sum())
                nonlin_arr.append(np.array([row[c] for c in nonlin_lag_cols], dtype=np.float32).sum())
                if idx % 10 == 5:
                    break
            mc_per_dataset[dataset] = {"lin": lin_arr, "nonlin": nonlin_arr}
            datasets_with_mc.append(dataset)

        if mc_per_dataset:
            plot_mc_analysis(mc_per_dataset, dict_with_all_datasets, datasets_with_mc,
                             lin_lag_cols, nonlin_lag_cols, OUTPUT_FOLDER)

    # ── Drop non-numeric columns ─────────────────────────────────────────────────
    for name in list(dict_with_all_datasets):
        df = dict_with_all_datasets[name]
        bad = [c for c in df.columns
               if df[c].dtype not in (np.float64, np.float32, np.int64, np.int32)]
        dict_with_all_datasets[name] = df.drop(columns=bad)

    # ── Property presence table ───────────────────────────────────────────────────
    all_cols = sorted(set(c for df in dict_with_all_datasets.values() for c in df.columns))
    presence = pd.DataFrame(
        {name: ["y" if c in df.columns else "missing" for c in all_cols]
         for name, df in dict_with_all_datasets.items()},
        index=all_cols,
    )
    presence.rename(columns=LABEL_MAP, inplace=True)
    presence.to_csv(OUTPUT_FOLDER / "property_presence_table.csv")
    print(f"  Saved {OUTPUT_FOLDER / "property_presence_table.csv"}")

    # ── Precise-categories pkl (for downstream PCA) ───────────────────────────────
    with open(OUTPUT_FOLDER / "selected_categories.pkl", "wb") as f:
        pickle.dump(PRECISE_CATEGORIES, f)

    dict_precise = {
        name: df[[c for c in PRECISE_CATEGORIES if c in df.columns]]
        for name, df in dict_with_all_datasets.items()
    }
    with open(OUTPUT_FOLDER / "all_datasets_precise_categories.pkl", "wb") as f:
        pickle.dump(dict_precise, f)
    print(f"  Saved all_datasets_precise_categories.pkl ({len(PRECISE_CATEGORIES)} categories)")

    # ── Save η/γ index ────────────────────────────────────────────────────────────
    df_gnm_eta_gamma = df_gnm[["eta", "gamma", "id"]].copy() if "id" in df_gnm.columns \
        else df_gnm[["eta", "gamma"]].copy()
    with open(OUTPUT_FOLDER / f"{GNM_DATASET}_gnm_eta_and_gamma.pkl", "wb") as f:
        pickle.dump(df_gnm_eta_gamma, f)

    # ── GNM η-γ heatmaps, one figure per category ────────────────────────────────
    print("Plotting GNM heatmaps...")
    df_gnm_g = df_gnm.groupby(["eta", "gamma"]).mean(numeric_only=True).reset_index()
    df_gnm_g = df_gnm_g.drop(columns=[c for c in _GNM_CONSTANT_COLUMNS if c in df_gnm_g.columns])
    plot_gnm_heatmaps(df_gnm_g, OUTPUT_FOLDER)

    # ── Wiring cost vs LR fraction scatters ──────────────────────────────────────
    # print("Plotting LR-fraction scatters...")
    # plot_lr_fraction_scatters(dict_with_all_datasets, OUTPUT_FOLDER)

    print("\nDone.")
