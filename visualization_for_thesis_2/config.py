from vizman import viz

# Scatter plot: wiring_cost vs computational_capacity (all datasets)
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

# CORNERS_OF_2D_SPACE = { # Pink-ish
#     "HEX_BOTTOM_LEFT": "#E3E3E3", #  # "AD5C4D",
#     "HEX_BOTTOM_RIGHT": "#A6587C",
#     "HEX_TOP_LEFT": "#9C9C9C", # "#A6587C", # "4D9EAD",
#     "HEX_TOP_RIGHT": "#430019", # #4C6FAD",
# }

# CORNERS_OF_2D_SPACE = { # Green-ish
#     "HEX_BOTTOM_LEFT": "#E3E3E3", #  # "AD5C4D",
#     "HEX_BOTTOM_RIGHT": "#A2DB8F",  #A6587C",
#     "HEX_TOP_LEFT": "#9C9C9C", # "#A6587C", # "4D9EAD",
#     "HEX_TOP_RIGHT": "#638657", # 430019", # #4C6FAD",
# }



# CORNERS_OF_2D_SPACE = {
#     "HEX_BOTTOM_LEFT": "#E3E3E3", #  # "AD5C4D",
#     "HEX_BOTTOM_RIGHT": "#CFA382", #A2DB8F",  #A6587C",
#     "HEX_TOP_LEFT": "#737373", #9C9C9C", # "#A6587C", # "4D9EAD",
#     "HEX_TOP_RIGHT": "#554B43", # 7F6450", # 638657", # 430019", # #4C6FAD",
# }



CORNERS_OF_2D_SPACE = {
    "HEX_BOTTOM_LEFT": "#5D5C55", #8E8E8E", # "#B5B3A5", # E3E3E3", #  # "AD5C4D",
    "HEX_BOTTOM_RIGHT": "#2C7D8F", #  "#3FAFC9", # CFA382", #A2DB8F",  #A6587C",
    "HEX_TOP_LEFT": "#E5E4E4", # 737373", #9C9C9C", # "#A6587C", # "4D9EAD",
    "HEX_TOP_RIGHT": "#A3FFFF", # "#554B43", # 7F6450", # 638657", # 430019", # #4C6FAD",
}





# Fixed color scheme
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

gray_cmap = viz.give_colormaps()["bw_hb"]
gray_cmap = viz.give_colormaps()["bw_hb"]

bone_white = COLORS["neutrals"]["BONE_WHITE"]
half_black = COLORS["neutrals"]["HALF_BLACK"]


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



# ["colds"]["LAKE_BLUE"]


PROPERTY_NAMES = {
    "algebraic_connectivity_fiedler_bipartition_balance": "Fiedler Bipartition Balance",
    "algebraic_connectivity_fiedler_value": "Fiedler Value",
    "algebraic_connectivity_fiedler_value_norm": "Fiedler Value (Normalized)",
    "algebraic_connectivity_laplacian_spectral_gap": "Laplacian Spectral Gap",
    "algebraic_connectivity_nx": "Algebraic Connectivity",
    "avg_clustering": "Clustering", # "Average Clustering Coefficient",
    "avg_communicability": "Communicability", # "Average Communicability",
    "avg_edge_distance": "Average Edge Distance", # "Average Edge Distance",
    "char_path_length": "Characteristic Path Length",
    
    "community_synchronization_vulnerability_n_communities": "Sync. Vulnerability: N Communities",
    "community_synchronization_vulnerability_value": "Synchronization Vulnerability",
    
    "computational_capacity_cross_capacity_total": "Cross Capacity", #  (Total)",
    "computational_capacity_cubic_capacity_total": "Cubic Capacity", #  (Total)",
    "computational_capacity_lyapunov_exponent": "Lyapunov Exponent",
    "computational_capacity_memory_capacity_total": "Memory Capacity", # MC (Total)",
    "computational_capacity_memory_nonlinear_ratio": "Memory to Nonlinear Ratio",
    "computational_capacity_memory_timescale": "Memory Timescale",
    "computational_capacity_nonlinear_capacity_total": "Nonlinear Capacity", #  (Total)",
    "computational_capacity_separation_ratio": "Separation Ratio",
    "computational_capacity_state_dimensionality": "State Dimensionality",
    "computational_capacity_state_entropy": "State Entropy",
    "computational_capacity_state_rank": "State Rank",
    "computational_capacity_total_capacity": "Total Computational Capacity",
    
    "degree_assortativity": "Degree Assortativity",
    "degree_gini": "Degree Gini Coefficient",
    "departure_from_normality_schur": "Departure from Normality", #  (Schur)",
    "diffusion_efficiency": "Diffusion Efficiency",
    
    "directed_simplices_count": "Directed Simplices Count",
    "directed_simplices_max_size": "Directed Simplices Max Size",
    "effective_dimensionality": "Effective Dimensionality",
    
    "energy": "Energy",
    "eta": r"$\eta$", # Eta (η)",
    "gamma": r"$\gamma$", # Gamma (γ)",
    "global_efficiency": "Global Efficiency",
    # "id": "ID",
    
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
    
    "mc_input_scaling_0_1_mc_mean": "MC Input Scaling 0.1 (Mean)",
    "mc_input_scaling_0_1_mc_std": "MC Input Scaling 0.1 (Std)",
    "mc_nonlinear_input_scaling_0_1_mc_mean": "MC Non-Linear Input Scaling 0.1 (Mean)",
    "mc_nonlinear_input_scaling_0_1_mc_std": "MC Non-Linear Input Scaling 0.1 (Std)",
    "mc_lin_cut40": "MC Linear Cut 40",
    "mc_nonlin_cut40": "Computational: MC Non-Linear Cut 40",

    # "mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"
    "modularity": "Modularity",
    
    "nct_control_avg": "NCT: Average Controllability",
    "nct_control_max": "NCT: Max Controllability",
    "nct_control_n_nodes_10_percent": "NCT: N Nodes (Top 10%)",
    "nct_control_n_nodes_50_percent": "NCT: N Nodes (Top 50%)",
    "nct_control_n_nodes_90_percent": "NCT: N Nodes (Top 90%)",
    "nct_control_std": "NCT: Controllability Std",
    "nct_energies_max": "NCT: Max Control Energy",
    
    "nct_energies_n_nodes_10_percent": "NCT: Energy N Nodes (Top 10%)",
    "nct_energies_n_nodes_50_percent": "NCT: Energy N Nodes (Top 50%)",
    "nct_energies_n_nodes_90_percent": "NCT: Energy N Nodes (Top 90%)",
    
    "nct_energies_std": "NCT: Control Energy Std",
    "nct_energies_total": "NCT: Total Control Energy",
    
    "ollivier_ricci_curvature_orc_frac_neg": "Ollivier-Ricci Curvature: Fraction Negative",
    "ollivier_ricci_curvature_orc_max": "Ollivier-Ricci Curvature: Max",
    "ollivier_ricci_curvature_orc_mean": "Ollivier-Ricci Curvature: Mean",
    "ollivier_ricci_curvature_orc_median": "Ollivier-Ricci Curvature: Median",
    "ollivier_ricci_curvature_orc_min": "Ollivier-Ricci Curvature: Min",
    "ollivier_ricci_curvature_orc_skewness": "Ollivier-Ricci Curvature: Skewness",
    "ollivier_ricci_curvature_orc_std": "Ollivier-Ricci Curvature: Std",
    
    "omega": "Omega (Small-World Index)",
    
    "participation_coefficient_n_communities": "Participation Coeff.: N Communities",
    "participation_coefficient_n_communities_further": "Participation Coeff.: N Communities ",
    "participation_coefficient_pc_frac_connector": "Participation Coeff.: Fraction Connector Hubs",
    "participation_coefficient_pc_frac_connector_further": "Participation Coeff.: Fraction Connector Hubs ",
    "participation_coefficient_pc_mean": "Participation Coeff.: Mean",
    "participation_coefficient_pc_mean_further": "Participation Coeff.: Mean ",
    "participation_coefficient_pc_median": "Participation Coeff.: Median",
    "participation_coefficient_pc_median_further": "Participation Coeff.: Median ",
    "participation_coefficient_pc_std": "Participation Coeff.: Std",
    "participation_coefficient_pc_std_further": "Participation Coeff.: Std ",
    "participation_coefficient_wmd_mean": "Within-Module Degree (Mean)",
    "participation_coefficient_wmd_mean_further": "Within-Module Degree (Mean, Further)",
    "participation_coefficient_wmd_std": "Within-Module Degree (Std)",
    "participation_coefficient_wmd_std_further": "Within-Module Degree (Std, Further)",
    # "persistent_homology_ph_h0_entropy": "Persistent Homology H0: Entropy",
    # "persistent_homology_ph_h0_n_features": "Persistent Homology H0: N Features",
    # "persistent_homology_ph_h0_persistence_mean": "Persistent Homology H0: Mean Persistence",
    # "persistent_homology_ph_h0_persistence_std": "Persistent Homology H0: Std Persistence",
    "persistent_homology_ph_h1_entropy": "Persistent Homology H1: Entropy",
    "persistent_homology_ph_h1_n_features": "Persistent Homology H1: N Features",
    "persistent_homology_ph_h1_persistence_mean": "Persistent Homology H1: Mean Persistence",
    # "persistent_homology_ph_h1_persistence_std": "Persistent Homology H1: Std Persistence",
    "persistent_homology_ph_total_persistence": "Persistent Homology: Total Persistence",
    
    "propagation_efficiency": "Propagation Efficiency",

    "proportion_long_range_connections_0.1": r"$f_{{\rm LR\,(10\%)}}$", # "Long-range fraction (> 0.1)",
    "proportion_long_range_connections_0.3": r"$f_{{\rm LR\,(30\%)}}$", # "Long-range fraction (> 0.3)",
    "proportion_long_range_connections_0.356": r"$f_{{\rm LR\,(35.6\%)}}$", # "Long-range fraction (> 0.356)",
    "proportion_long_range_connections_0.365": r"$f_{{\rm LR\,(36.5\%)}}$", # "Long-range fraction (> 0.365)",
    "proportion_long_range_connections_0.3956": r"$f_{{\rm LR\,(39.56\%)}}$", # "Long-range fraction (> 0.3956)",
    "proportion_long_range_connections_0.5": r"$f_{{\rm LR\,(50\%)}}$", # "Long-range fraction (> 0.5)",
    "repertoire_sweep_weighted_by_distances_T_critical": "Metastability: Crit. T", # Repertoire Sweep: Critical Temperature",
    "repertoire_sweep_weighted_by_distances_diversity_critical": "Metastability: Crit. Diversity", # Repertoire Sweep: Critical Diversity",
    "repertoire_sweep_weighted_by_distances_size_critical": "Metastability: Crit. Size", # Repertoire Sweep: Critical Size",
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
    "targeted_attack_robustness_rob_random_auc": "Robustness (Random, AUC)", # : Random Attack AUC",
    "targeted_attack_robustness_rob_random_half": "Robustness: Random Attack Half",
    "targeted_attack_robustness_rob_ratio": "Robustness: Targeted/Random Ratio",
    "targeted_attack_robustness_rob_targeted_auc": "Robustness (Targeted, AUC)", # : Targeted Attack AUC",
    "targeted_attack_robustness_rob_targeted_half": "Robustness: Targeted Attack Half",
    "topological_distance_mean": "Topological Distance (Mean)",
    "topological_distance_std": "Topological Distance (Std)",
    "transitivity": "Transitivity",
    "wiring_cost": "Wiring Cost",
    
    "ipc_ipc_deg1_mean": "IPC: Deg 1 (Mean)",
    "ipc_ipc_deg1_std": "IPC: Deg 1 (Std)",
    "ipc_ipc_deg2_mean": "IPC: Deg 2 (Mean)",
    "ipc_ipc_deg2_std": "IPC: Deg 2 (Std)", 
    
    "gromov_hyperbolicity": "Gromov Hyperbolicity",
    "betweenness_centrality_stats_mean": "Betweenness Centrality (Mean)",
    "betweenness_centrality_stats_std": "Betweenness Centrality (Std)",
    "betweenness_centrality_stats_max": "Betweenness Centrality (Max)",
    "betweenness_centrality_stats_gini": "Betweenness Centrality (Gini)",
    "local_efficiency_stats_mean": "Local Eff. (Mean)",
    "local_efficiency_stats_std": "Local Eff. (Std)",
    "local_efficiency_stats_min": "Local Eff. (Min)",
    "local_efficiency_stats_max": "Local Eff. (Max)",

}



REPRESENTATIVES_FOR_GOALS = { # Those are all the properties that were still left after pruning to r <= 0.95. 
#     'energy', 
#     'avg_communicability', 'richclub_avg_length', 
    'mc_mean': "Capacity (Memory)",
#        'avg_clustering', 'degree_assortativity', 
    'modularity': "Segregation",
#        'topological_distance_mean', 'degree_gini', 'directed_simplices_count',
#        'directed_simplices_max_size',
    'proportion_long_range_connections_0.3956': "Wiring Economy", 
    'global_efficiency': "Integration", 
#        'diffusion_efficiency', 'nct_control_avg', 'nct_control_max',
#        'nct_control_n_nodes_90_percent', 'nct_control_n_nodes_50_percent',
#        'nct_control_n_nodes_10_percent', 'nct_energies_total',
#        'nct_energies_std', 'nct_energies_max',
#        'nct_energies_n_nodes_90_percent', 'nct_energies_n_nodes_50_percent',
#        'nct_energies_n_nodes_10_percent',
       'synchronizability_eigenratio_eigenratio': "Synchronizability: Eigenratio",
#        'synchronizability_eigenratio_lambda_N', 
#         'spectral_gap_fatemeh',
#        'kuramoto_synchronization_r_final', 'kuramoto_synchronization_r_mean',
       'kuramoto_synchronization_r_std': "Kuramoto: Std",
#        'departure_from_normality_schur',
#        'kuramoto_averaged_synchronization_r_final',
#        'kuramoto_averaged_synchronization_r_mean',
#        'kuramoto_averaged_synchronization_r_std',
#        'kuramoto_averaged_synchronization_r_mean_se',
#        'kuramoto_averaged_synchronization_r_std_se',
#        'participation_coefficient_pc_std',
#        'participation_coefficient_wmd_mean',
#        'participation_coefficient_wmd_std',
#        'participation_coefficient_n_communities',
#        'community_synchronization_vulnerability_n_communities',
       'algebraic_connectivity_nx': "Robustness", 
#        'computational_capacity_memory_capacity_total',
       'computational_capacity_nonlinear_capacity_total': "Capacity (Nonlinear Computation)",
#        'computational_capacity_memory_nonlinear_ratio',
#        'computational_capacity_state_dimensionality',
#        'computational_capacity_state_rank',
#        'computational_capacity_separation_ratio',
#        'computational_capacity_lyapunov_exponent',
#        'kernel_rank_thresholded_and_summed_0.01',
#        'kernel_rank_phase_diff_of_lambda_max_and_2nd',
#        'repertoire_sweep_weighted_by_distances_T_critical',
#        'repertoire_sweep_weighted_by_distances_size_critical',
       'repertoire_sweep_weighted_by_distances_diversity_critical': "Metastability (Repertoire Diversity)",
#        'ollivier_ricci_curvature_orc_std', 
#        'ollivier_ricci_curvature_orc_min',
#        'ollivier_ricci_curvature_orc_max',
#        'ollivier_ricci_curvature_orc_median',
#        'ollivier_ricci_curvature_orc_skewness',
#        'ollivier_ricci_curvature_orc_frac_neg',
#        'rich_club_coefficient_rc_max_norm',
#        'rich_club_coefficient_rc_k_at_max',
#        'rich_club_coefficient_rc_mean_norm',
#        'rich_club_coefficient_rc_weighted_auc',
#        'rich_club_coefficient_rc_regime_frac',
#        'persistent_homology_ph_h1_n_features',
#        'persistent_homology_ph_h1_persistence_mean',
       'targeted_attack_robustness_rob_targeted_auc': "Robustness (Targeted Attack)",
#        'targeted_attack_robustness_rob_random_auc',
#        'targeted_attack_robustness_rob_ratio',
#        'algebraic_connectivity_laplacian_spectral_gap',
#        'algebraic_connectivity_fiedler_bipartition_balance'
}



# DETERMINED WITH KAYSON (until Mar 16th) 
# ── Category definitions ──────────────────────────────────────────────────────
# remaining_categories = {
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



# # DETERMINED WITH KAYSON + small change (beginning March 16th)
# # ── Category definitions ──────────────────────────────────────────────────────
# remaining_categories = {
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
#         # "participation_coefficient_pc_frac_connector", # needs to be put into relation with Q
#         # "participation_coefficient_wmd_std",
#     ],
#     "Ollivier-Ricci Curvature": [ # If too annoying to describe, keep only one. 
#         "ollivier_ricci_curvature_orc_mean",
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
#         "targeted_attack_robustness_rob_random_auc", # CHANGED: WAS NOT ACTIVATED AFTER TALKING TO KAYSONcorr too highly with 'char_path_length'
#         # "targeted_attack_robustness_rob_random_half",
#         # "targeted_attack_robustness_rob_ratio", # CHANGED: WAS ACTIVATED AFTER TALKING TO KAYSON
        
#         "community_synchronization_vulnerability_value", # too high corr with participation_coefficient_pc_frac_connector
#         # "community_synchronization_vulnerability_n_communities",
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
#     # "Computational Capacity": [
#     #     "computational_capacity_memory_capacity_total",
#     #     # "computational_capacity_memory_timescale",
#     #     "computational_capacity_nonlinear_capacity_total",
#     #     # "computational_capacity_cubic_capacity_total",
#         # "computational_capacity_cross_capacity_total",
#         # "computational_capacity_memory_nonlinear_ratio",
#         # "computational_capacity_total_capacity",
#         # "computational_capacity_state_dimensionality", 
#         # "computational_capacity_state_entropy",
#         # "computational_capacity_state_rank",
#         # "computational_capacity_separation_ratio",
#         # "computational_capacity_lyapunov_exponent",
#     # ],
#     "Memory Capacity": [
#         # "mc_input_scaling_0_1_mc_mean",
#         # "mc_nonlinear_input_scaling_0_1_mc_mean", 
#         "ipc_ipc_deg1_mean", 
#         "ipc_ipc_deg2_mean"
#     ],
#     # "Reservoir Computing - Basic Measures": 
#     # "Memory Capacity - Full Lag Profile": [
#     #     # [f"mc_{i}" for i in range(1, 50)]
#     #     # "mc_mean" # , "mc_std"]
#     # ],
# }







CATEGORY_COLOURS = { # Integration etc. 
    "Integration": "#4C72B0",
    "Segregation": "#DD8452",
    "Robustness":  "#55A868",
    # "Topology":    "#C44E52",
    "Wiring":      "#8172B2",
    # "Geometry":    "#937860",
    "Dynamics":    "#DA8BC3",
    "Computation": "#8C8C8C",
    "Structure":   "#CCB974",
    "Communication": "#64B5CD",
    "Economic efficiency": "#E24A33",
}


CATEGORY_ORDER = [
    "Economic efficiency", 
    # "Structure", 
    "Integration", 
    "Segregation",
    # "Dynamics", 
    "Computation", 
    "Robustness",
    "Communication"
]



selected_properties = {
    "Integration": "global_efficiency",
    "Segregation": "modularity", 
    "Wiring\neconomy": "proportion_long_range_connections_0.3956", 
    "Robustness": "targeted_attack_robustness_rob_targeted_auc", 
    "Robustness\n(lambda_2)": "algebraic_connectivity_fiedler_value",
    "Dynamics": "spectral_radius", 
    "Memory": "mc_input_scaling_0_1_mc_mean", # "computational_capacity_memory_capacity_total",
    "Computational\ncapacity": "mc_nonlinear_input_scaling_0_1_mc_mean", # "computational_capacity_nonlinear_capacity_total", 
    "Metastability Reservoir Diversity": "repertoire_sweep_weighted_by_distances_diversity_critical", 
}

# "global_efficiency", "modularity", 
# "proportion_long_range_connections_0.3956", "targeted_attack_robustness_rob_targeted_auc", 
# "algebraic_connectivity_fiedler_value", "spectral_radius", 
# "computational_capacity_memory_capacity_total", "computational_capacity_nonlinear_capacity_total", 
# "repertoire_sweep_weighted_by_distances_diversity_critical", 


SELECTED_PROPERTIES_NAMES = {
    "algebraic_connectivity_fiedler_bipartition_balance": "Fiedler Bipartition Balance",

    # "algebraic_connectivity_fiedler_value": r"$\lambda_2$", # "Fiedler Value",
    "algebraic_connectivity_fiedler_value": "Fiedler Value",
    
    "algebraic_connectivity_fiedler_value_norm": "Fiedler Value (Normalized)",
    "algebraic_connectivity_laplacian_spectral_gap": "Laplacian Spectral Gap",
    "algebraic_connectivity_nx": "Algebraic Connectivity",
    "avg_clustering": "Clustering", # "Average Clustering Coefficient",
    "avg_communicability": "Communicability", # "Average Communicability",
    "avg_edge_distance": "Edge Distance (Mean)", # "Average Edge Distance",
    "char_path_length": "Char. Path Length",

    "community_synchronization_vulnerability_n_communities": "Number Communities",
    # "community_synchronization_vulnerability_value": "Synchronization Vulnerability",
    
    # "computational_capacity_cross_capacity_total": "Cross Capacity", #  (Total)",
    # "computational_capacity_cubic_capacity_total": "Cubic Capacity", #  (Total)",
    # "computational_capacity_lyapunov_exponent": "Lyapunov Exponent",
    # "computational_capacity_memory_capacity_total": "Memory Capacity", # MC (Total)",
    # "computational_capacity_memory_nonlinear_ratio": "Memory to Nonlinear Ratio",
    # "computational_capacity_memory_timescale": "Memory Timescale",
    # "computational_capacity_nonlinear_capacity_total": "Nonlinear Capacity", #  (Total)",
    # "computational_capacity_separation_ratio": "Separation Ratio",
    # "computational_capacity_state_dimensionality": "State Dimensionality",
    # "computational_capacity_state_entropy": "State Entropy",
    # "computational_capacity_state_rank": "State Rank",
    # "computational_capacity_total_capacity": "Total Computational Capacity",
    
    # "degree_assortativity": "Degree Assortativity",
    # "degree_gini": "Degree Gini Coefficient",
    "departure_from_normality_schur": "Departure from Normality", #  (Schur)",
    
    # "diffusion_efficiency": r"$E_{{\rm diff}}$", # "Diffusion Eff.",
    "diffusion_efficiency": "Diffusion Eff.",
    
    
    "directed_simplices_count": "Directed Simplices Count",
    "directed_simplices_max_size": "Directed Simplices Max Size",
    "effective_dimensionality": "Effective Dimensionality",
    
    # "energy": "Energy",
    # "eta": r"$\eta$", # Eta (η)",
    # "gamma": r"$\gamma$", # Gamma (γ)",
    
    # "global_efficiency": r"$E_{{\rm glob}}$", # "Global Efficiency",
    "global_efficiency": "Global Efficiency",
    # # "id": "ID",
    
    "kernel_rank_max": "Kernel Rank (Max)",
    "kernel_rank_phase_diff_of_lambda_max_and_2nd": "Kernel Rank: Phase Diff (λ_max vs 2nd)",
    "kernel_rank_thresholded_and_summed_0.01": "Kernel Rank (Thresholded, 0.01)",
    
    "kuramoto_averaged_synchronization_r_final": "Kuramoto Avg. Sync. r (Final)",
    "kuramoto_averaged_synchronization_r_mean": "Kuramoto Avg. Sync. r (Mean)",
    "kuramoto_averaged_synchronization_r_mean_se": "Kuramoto Avg. Sync. r (Mean SE)",
    "kuramoto_averaged_synchronization_r_std": "Kuramoto Avg. Sync. r (Std)",
    "kuramoto_averaged_synchronization_r_std_se": "Kuramoto Avg. Sync. r (Std SE)",
    
    # "kuramoto_synchronization_r_final": "Kuramoto Sync. r (Final)",
    # "kuramoto_synchronization_r_mean": "Kuramoto Sync. r (Mean)",
    "kuramoto_synchronization_r_std": "Kuramoto Sync. r (Std)",
    # "mc_1": "MC (Lag 1)",
    # "mc_10": "MC (Lag 10)",
    # "mc_11": "MC (Lag 11)",
    # "mc_12": "MC (Lag 12)",
    # "mc_13": "MC (Lag 13)",
    # "mc_14": "MC (Lag 14)",
    # "mc_15": "MC (Lag 15)",
    # "mc_16": "MC (Lag 16)",
    # "mc_17": "MC (Lag 17)",
    # "mc_18": "MC (Lag 18)",
    # "mc_19": "MC (Lag 19)",
    # "mc_2": "MC (Lag 2)",
    # "mc_20": "MC (Lag 20)",
    # "mc_21": "MC (Lag 21)",
    # "mc_22": "MC (Lag 22)",
    # "mc_23": "MC (Lag 23)",
    # "mc_24": "MC (Lag 24)",
    # "mc_25": "MC (Lag 25)",
    # "mc_26": "MC (Lag 26)",
    # "mc_27": "MC (Lag 27)",
    # "mc_28": "MC (Lag 28)",
    # "mc_29": "MC (Lag 29)",
    # "mc_3": "MC (Lag 3)",
    # "mc_30": "MC (Lag 30)",
    # "mc_31": "MC (Lag 31)",
    # "mc_32": "MC (Lag 32)",
    # "mc_33": "MC (Lag 33)",
    # "mc_34": "MC (Lag 34)",
    # "mc_35": "MC (Lag 35)",
    # "mc_36": "MC (Lag 36)",
    # "mc_37": "MC (Lag 37)",
    # "mc_38": "MC (Lag 38)",
    # "mc_39": "MC (Lag 39)",
    # "mc_4": "MC (Lag 4)",
    # "mc_40": "MC (Lag 40)",
    # "mc_41": "MC (Lag 41)",
    # "mc_42": "MC (Lag 42)",
    # "mc_43": "MC (Lag 43)",
    # "mc_44": "MC (Lag 44)",
    # "mc_45": "MC (Lag 45)",
    # "mc_46": "MC (Lag 46)",
    # "mc_47": "MC (Lag 47)",
    # "mc_48": "MC (Lag 48)",
    # "mc_49": "MC (Lag 49)",
    # "mc_5": "MC (Lag 5)",
    # "mc_6": "MC (Lag 6)",
    # "mc_7": "MC (Lag 7)",
    # "mc_8": "MC (Lag 8)",
    # "mc_9": "MC (Lag 9)",
    # "mc_mean": "MC (Mean)",
    # "mc_std": "MC (Std)",
    # "mc_nonlinear_original_mc_mean": "Non-Linear (Mean)",
    # "mc_nonlinear_original_mc_std": "Non-Linear (Std)",
    "ipc_ipc_deg1_mean": "MC Linear", # r"$MC_{{\rm lin}}$", # "IPC Deg. 1",
    "ipc_ipc_deg2_mean": "MC Non-Linear", # r"$MC_{{\rm quad}}$", # "IPC Deg. 2",
    
    "mc_input_scaling_0_1_mc_mean": "MC Linear",
    # "mc_input_scaling_0_1_mc_std": "MC Input Scaling 0.1 (Std)",
    "mc_nonlinear_input_scaling_0_1_mc_mean": "MC Non-Linear",
    # "mc_nonlinear_input_scaling_0_1_mc_std": "MC Non-Linear Input Scaling 0.1 (Std)",
    # "mc_lin_cut40": "MC Linear Cut 40",
    # "mc_nonlin_cut40": "Computational: MC Non-Linear Cut 40",

    # # "mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"
    "modularity": "Modularity",
    
    # "nct_control_avg": "NCT: Average Controllability",
    # "nct_control_max": "NCT: Max Controllability",
    # "nct_control_n_nodes_10_percent": "NCT: N Nodes (Top 10%)",
    # "nct_control_n_nodes_50_percent": "NCT: N Nodes (Top 50%)",
    # "nct_control_n_nodes_90_percent": "NCT: N Nodes (Top 90%)",
    "nct_control_std": "Controllability (std)", # NCT: Controllability Std",
    # "nct_energies_max": "NCT: Max Control Energy",
    
    # "nct_energies_n_nodes_10_percent": "NCT: Energy N Nodes (Top 10%)",
    # "nct_energies_n_nodes_50_percent": "NCT: Energy N Nodes (Top 50%)",
    # "nct_energies_n_nodes_90_percent": "NCT: Energy N Nodes (Top 90%)",
    
    # "nct_energies_std": "NCT: Control Energy Std",
    # "nct_energies_total": "NCT: Total Control Energy",
    
    "ollivier_ricci_curvature_orc_frac_neg": "ORC (fraction negative)", # "Ollivier-Ricci Curvature: Fraction Negative",
    "ollivier_ricci_curvature_orc_max": "ORC (max)", # Ollivier-Ricci Curvature: Max",
    "ollivier_ricci_curvature_orc_mean": "ORC (mean)", # Ollivier-Ricci Curvature: Mean",
    "ollivier_ricci_curvature_orc_median": "ORC (median)", # Ollivier-Ricci Curvature: Median",
    "ollivier_ricci_curvature_orc_min": "ORC (min)", # : Min",
    "ollivier_ricci_curvature_orc_skewness": "ORC (skewness)", # "Ollivier-Ricci Curvature: Skewness",
    "ollivier_ricci_curvature_orc_std": "ORC (std)", # Ollivier-Ricci Curvature: Std",
    
    "omega": "Omega",
    
    "participation_coefficient_n_communities": "N Communities",
    "participation_coefficient_n_communities_further": "N Communities",
    "participation_coefficient_pc_frac_connector": "Fraction Connector Hubs",
    "participation_coefficient_pc_frac_connector_further": "Fraction Connector Hubs",
    "participation_coefficient_pc_mean": "Participation Coeff. (mean)",
    "participation_coefficient_pc_mean_further": "Participation Coeff. (mean)",
    "participation_coefficient_pc_median": "Participation Coeff. (median)",
    "participation_coefficient_pc_median_further": "Participation Coeff. (median)", # : Median ",
    "participation_coefficient_pc_std": "Participation Coeff. (std)",
    "participation_coefficient_pc_std_further": "Participation Coeff. (std)",
    "participation_coefficient_wmd_mean": "Within-Module Degree (mean)",
    "participation_coefficient_wmd_mean_further": "Within-Module Degree (mean, further)",
    "participation_coefficient_wmd_std": "Within-Module Degree (std)",
    "participation_coefficient_wmd_std_further": "Within-Module Degree (std, further)",
    # "persistent_homology_ph_h0_entropy": "Persistent Homology H0: Entropy",
    # "persistent_homology_ph_h0_n_features": "Persistent Homology H0: N Features",
    # "persistent_homology_ph_h0_persistence_mean": "Persistent Homology H0: Mean Persistence",
    # "persistent_homology_ph_h0_persistence_std": "Persistent Homology H0: Std Persistence",
    "persistent_homology_ph_h1_entropy": "Persistent Homology H1: Entropy",
    "persistent_homology_ph_h1_n_features": "Persistent Homology H1: N Features",
    "persistent_homology_ph_h1_persistence_mean": "Persistent Homology H1: Mean Persistence",
    # "persistent_homology_ph_h1_persistence_std": "Persistent Homology H1: Std Persistence",
    "persistent_homology_ph_total_persistence": "Persistent Homology: Total Persistence",
    
    # "propagation_efficiency": r"$E_{{\rm prop}}$", # "Propagation Eff.",
    "propagation_efficiency": "Propagation Eff.",

#         r"$E_{{\rm Prop}}$"
#         r"$\lambda_2$"
    # "proportion_long_range_connections_0.1": r"$f_{{\rm LR\,(10\%)}}$", # "Long-range fraction (> 0.1)",
    # "proportion_long_range_connections_0.3": r"$f_{{\rm LR\,(30\%)}}$", # "Long-range fraction (> 0.3)",
    # "proportion_long_range_connections_0.356": r"$f_{{\rm LR\,(35.6\%)}}$", # "Long-range fraction (> 0.356)",
    # "proportion_long_range_connections_0.365": r"$f_{{\rm LR\,(36.5\%)}}$", # "Long-range fraction (> 0.365)",
    "proportion_long_range_connections_0.3956": r"$f_{{\rm LR\,(39.56\%)}}$", # "Long-range fraction (> 0.3956)",
    # "proportion_long_range_connections_0.5": r"$f_{{\rm LR\,(50\%)}}$", # "Long-range fraction (> 0.5)",
    
    # "repertoire_sweep_weighted_by_distances_T_critical": "Metastability:" + r"$Threshold_{{\rm critical}}$", #  Critical T", # Repertoire Sweep: Critical Temperature",
    # "repertoire_sweep_weighted_by_distances_diversity_critical": "Metastability: " + r"$D_{{\rm critical}}$", # Repertoire Sweep: Critical Diversity",
    # "repertoire_sweep_weighted_by_distances_size_critical": "Metastability: " + r"$Size_{{\rm critical}}$", # Repertoire Sweep: Critical Size",
    "repertoire_sweep_weighted_by_distances_T_critical": "Metastability (critical T)", # Repertoire Sweep: Critical Temperature",
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
    "targeted_attack_robustness_rob_random_auc": "Robustness (Random)", # , AUC)", # : Random Attack AUC",
    "targeted_attack_robustness_rob_random_half": "Robustness: Random Attack Half",
    "targeted_attack_robustness_rob_ratio": "Robustness: Targeted/Random Ratio",
    "targeted_attack_robustness_rob_targeted_auc": "Robustness (Targeted)", # , AUC)", # : Targeted Attack AUC",
    "targeted_attack_robustness_rob_targeted_half": "Robustness: Targeted Attack Half",
    "topological_distance_mean": "Topological Distance (Mean)",
    "topological_distance_std": "Topological Distance (Std)",
    "transitivity": "Transitivity",
    "wiring_cost": "Wiring Cost",
}


MERGED_PROPERTIES_NAMES = {**PROPERTY_NAMES, **SELECTED_PROPERTIES_NAMES}