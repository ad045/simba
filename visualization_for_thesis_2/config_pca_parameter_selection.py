# DETERMINED WITH KAYSON + huge change (Apr 10th) 
# ── Category definitions ──────────────────────────────────────────────────────
# remaining_categories = {
#         "degree_gini",
#         "betweenness_centrality_stats_gini", 
#         "betweenness_centrality_stats_mean",
#         "degree_assortativity", 
#         "omega", 
#         "structural_complexity", 
#         "directed_simplices_count", 
#         "directed_simplices_max_size",
#         "modularity", 
#         "local_efficiency_stats_mean", 
#         "local_efficiency_stats_std", 
#         "global_efficiency", 
#         "diffusion_efficiency",
#         "propagation_efficiency", 
#         "wiring_cost",
#         "proportion_long_range_connections_0.3956",
#         "rich_club_coefficient_rc_k_at_max",
#         "spectral_radius", 
#         "spectral_gap", 
#         "synchronizability_eigenratio_eigenratio",
#         "kernel_rank_max", 
#         "algebraic_connectivity_fiedler_value",
#         "algebraic_connectivity_laplacian_spectral_gap",
#         "departure_from_normality_schur",
#         "effective_dimensionality", 
#         "repertoire_sweep_weighted_by_distances_T_critical",
#         "repertoire_sweep_weighted_by_distances_size_critical",
#         "repertoire_sweep_weighted_by_distances_diversity_critical", 
        
#         "participation_coefficient_pc_std",
#         "ollivier_ricci_curvature_orc_mean",
#         "ollivier_ricci_curvature_orc_min",
#         "ollivier_ricci_curvature_orc_skewness",
#         "targeted_attack_robustness_rob_targeted_auc",
#         "targeted_attack_robustness_rob_random_auc", 
#         "community_synchronization_vulnerability_value", 
#         "nct_control_std",  
#         "ipc_ipc_deg1_mean", 
#         "ipc_ipc_deg2_mean"
# }

# DETERMINED WITH KAYSON + huge change (Apr 10th) 
# ── Category definitions ──────────────────────────────────────────────────────
remaining_categories = {
    "Fundamental Topology": [ 
        # "transitivity", # corr too highly with avg_clustering
        # "avg_clustering", 
        
        
        "degree_gini", # was previously removed, as it correlates too strongly with spectral_radius
        
        "betweenness_centrality_stats_gini", 
        "betweenness_centrality_stats_mean", 

        "degree_assortativity", 
        
        "omega", 
        "structural_complexity", # Added it back due to sheet
        
        "directed_simplices_count", # Higher order
        "directed_simplices_max_size",
        
        # "energy" # i.e.: Fit to networks - maybe remove again? 
    ],
    
    "Integration": [
        "modularity", 
    ], 
    
    "Segregation": [
        "local_efficiency_stats_mean", 
        "local_efficiency_stats_std", 
    ],
    
    "Paths, Efficiency & Communication": [
        # "char_path_length", 
        "global_efficiency", # corr too highly with omega
        "diffusion_efficiency",
        "propagation_efficiency", # corr too highly with diffusion_efficiency
        # "avg_communicability", # corr too highly with spectral_radius # !!!!!!!
        # "topological_distance_mean", 
        # "topological_distance_std",
    ],
    "Spatial Embedding & Wiring wCost": [
        # "avg_edge_distance", 
        "wiring_cost",
        # "proportion_long_range_connections_0.1",
        # "proportion_long_range_connections_0.3",
        "proportion_long_range_connections_0.3956",
        # "proportion_long_range_connections_0.5",
    ],
    "Rich-Club Organization": [
        # "richclub_n_edges", 
        # "richclub_avg_length",
        # "rich_club_coefficient_rc_max_norm",
        # "rich_club_coefficient_rc_mean_norm",
        "rich_club_coefficient_rc_k_at_max",
        # "rich_club_coefficient_rc_regime_frac",
        # "rich_club_coefficient_rc_weighted_auc",
    ],
    "Spectral Properties, Algebraic Connectivity & Fiedler Analysis": [
        "spectral_radius", 
        "spectral_gap", # Has exactly the same output as Fatemehs version, bc it is identical (lambda_n - lambda_(n-1))
        "synchronizability_eigenratio_eigenratio", # lambda_n / lambda_2 # would also have lambda_2 and lambda_n

        "kernel_rank_max", # ??? Too high correlation with other features? 
        # "kernel_rank_thresholded_and_summed_0.01",          # np.sum(np.abs(eigs) > threshold * np.abs(eigs).max())
        # "kernel_rank_phase_diff_of_lambda_max_and_2nd",     # np.angle(eigs[np.argsort(np.abs(eigs))[-2]]) - np.angle(eigs.max())
        
        # "algebraic_connectivity_fiedler_value", # Relies on a different library, I hope? 
        "synchronizability_eigenratio_eigenratio_lambda_2", 
        # "algebraic_connectivity_fiedler_value_norm",
        "algebraic_connectivity_laplacian_spectral_gap",
        # "algebraic_connectivity_fiedler_bipartition_balance",

        "departure_from_normality_schur",

        "effective_dimensionality", # smooth, but corr too highly with omega
    ],
    "Metastability & Kuramoto Synchronization": [ 
        "repertoire_sweep_weighted_by_distances_T_critical",
        "repertoire_sweep_weighted_by_distances_size_critical",
        "repertoire_sweep_weighted_by_distances_diversity_critical", 
        
        # "kuramoto_averaged_synchronization_r_final",
        # "kuramoto_averaged_synchronization_r_mean",
        # "kuramoto_averaged_synchronization_r_mean_se",
        # "kuramoto_averaged_synchronization_r_std",
        # "kuramoto_averaged_synchronization_r_std_se",
    ],


    "Participation Coefficient (Louvain)": [
        # "participation_coefficient_n_communities", # too noisy 
        # "participation_coefficient_pc_mean", # corr too highly with omega
        # "participation_coefficient_pc_median",
        "participation_coefficient_pc_std",
        # "participation_coefficient_pc_frac_connector", # needs to be put into relation with Q
        # "participation_coefficient_wmd_std",
    ],
    "Ollivier-Ricci Curvature": [ # If too annoying to describe, keep only one. 
        "ollivier_ricci_curvature_orc_mean",
        # "ollivier_ricci_curvature_orc_median",
        # "ollivier_ricci_curvature_orc_std", 
        "ollivier_ricci_curvature_orc_min",
        # "ollivier_ricci_curvature_orc_max",
        "ollivier_ricci_curvature_orc_skewness",
        # "ollivier_ricci_curvature_orc_frac_neg",
    ],
    # "Persistent Homology (TDA)": [
    #     # "persistent_homology_ph_h1_n_features",
    #     # "persistent_homology_ph_h1_persistence_mean",
    #     # "persistent_homology_ph_h1_entropy",
    #     # "persistent_homology_ph_total_persistence",
    # ],

    "Targeted Attack Robustness & Community Structure & Vulnerability": [
        "targeted_attack_robustness_rob_targeted_auc",
        # "targeted_attack_robustness_rob_targeted_half",
        "targeted_attack_robustness_rob_random_auc", # CHANGED: WAS NOT ACTIVATED AFTER TALKING TO KAYSONcorr too highly with 'char_path_length'
        # "targeted_attack_robustness_rob_random_half",
        # "targeted_attack_robustness_rob_ratio", # CHANGED: WAS ACTIVATED AFTER TALKING TO KAYSON
        
        "community_synchronization_vulnerability_value", # too high corr with participation_coefficient_pc_frac_connector
        # "community_synchronization_vulnerability_n_communities",
        ],
    "Network Control Theory - Average Controllability": [
        # "nct_control_avg",
        "nct_control_std",  
        # "nct_control_max",
        # "nct_control_n_nodes_90_percent",
        # "nct_control_n_nodes_50_percent",
        # "nct_control_n_nodes_10_percent",
    ],
    # "Network Control Theory - Control Energy": [
    #     "nct_energies_total",
    #     "nct_energies_std", 
    #     "nct_energies_max",
    #     "nct_energies_n_nodes_90_percent",
    #     "nct_energies_n_nodes_50_percent",
    #     "nct_energies_n_nodes_10_percent",
    # ],
    # "Computational Capacity": [
    #     "computational_capacity_memory_capacity_total",
    #     # "computational_capacity_memory_timescale",
    #     "computational_capacity_nonlinear_capacity_total",
    #     # "computational_capacity_cubic_capacity_total",
        # "computational_capacity_cross_capacity_total",
        # "computational_capacity_memory_nonlinear_ratio",
        # "computational_capacity_total_capacity",
        # "computational_capacity_state_dimensionality", 
        # "computational_capacity_state_entropy",
        # "computational_capacity_state_rank",
        # "computational_capacity_separation_ratio",
        # "computational_capacity_lyapunov_exponent",
    # ],
    "Memory Capacity": [
        # "mc_input_scaling_0_1_mc_mean",
        # "mc_nonlinear_input_scaling_0_1_mc_mean", 
        "ipc_ipc_deg1_mean", 
        "ipc_ipc_deg2_mean"
    ],
    # "Reservoir Computing - Basic Measures": 
    # "Memory Capacity - Full Lag Profile": [
    #     # [f"mc_{i}" for i in range(1, 50)]
    #     # "mc_mean" # , "mc_std"]
    # ],
}



# PROPERTIES TO BE CONSIDERED: 
""" 
resistance_distance_mean
"""

# ALL PROPERTIES: 
"""
,MaMI,Developing,Diffusion,Propagation,Routing,Ring Lattice,Erdős-Rényi,HCP (GNM)
"MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)",missing,missing,missing,missing,missing,missing,missing,y
algebraic_connectivity_fiedler_bipartition_balance,y,y,y,y,y,y,y,y
algebraic_connectivity_fiedler_value,y,y,y,y,y,y,y,y
algebraic_connectivity_fiedler_value_norm,y,y,y,y,y,y,y,y
algebraic_connectivity_laplacian_spectral_gap,y,y,y,y,y,y,y,y
algebraic_connectivity_nx,y,y,y,y,y,y,y,y
avg_clustering,y,y,y,y,y,y,y,y
avg_clustering_static,missing,missing,missing,missing,missing,missing,missing,y
avg_communicability,y,y,y,y,y,y,y,y
avg_degree,y,y,y,y,y,y,y,y
avg_degree_static,missing,missing,missing,missing,missing,missing,missing,y
avg_edge_distance,y,y,y,y,y,y,y,y
betweenness_centrality_stats_gini,y,y,y,y,y,y,y,y
betweenness_centrality_stats_max,y,y,y,y,y,y,y,y
betweenness_centrality_stats_mean,y,y,y,y,y,y,y,y
betweenness_centrality_stats_std,y,y,y,y,y,y,y,y
char_path_length,y,y,y,y,y,y,y,y
community_synchronization_vulnerability_n_communities,y,y,y,y,y,y,y,y
community_synchronization_vulnerability_value,y,y,y,y,y,y,y,y
computational_capacity_cross_capacity_total,y,y,y,y,y,y,y,y
computational_capacity_cubic_capacity_total,y,y,y,y,y,y,y,y
computational_capacity_lyapunov_exponent,y,y,y,y,y,y,y,y
computational_capacity_memory_capacity_total,y,y,y,y,y,y,y,y
computational_capacity_memory_nonlinear_ratio,y,y,y,y,y,y,y,y
computational_capacity_memory_timescale,y,y,y,y,y,y,y,y
computational_capacity_nonlinear_capacity_total,y,y,y,y,y,y,y,y
computational_capacity_notebook_damicelli_memory_capacity_total_notebook,y,y,y,y,y,y,y,y
computational_capacity_notebook_damicelli_nonlinear_capacity_total_notebook,y,y,y,y,y,y,y,y
computational_capacity_notebook_memory_capacity_total_notebook,y,y,y,y,y,y,y,y
computational_capacity_notebook_nonlinear_capacity_total_notebook,y,y,y,y,y,y,y,y
computational_capacity_separation_ratio,y,y,y,y,y,y,y,y
computational_capacity_state_dimensionality,y,y,y,y,y,y,y,y
computational_capacity_state_entropy,y,y,y,y,y,y,y,y
computational_capacity_state_rank,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_0,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_1,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_10,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_11,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_12,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_13,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_14,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_15,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_16,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_17,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_18,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_19,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_2,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_20,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_21,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_22,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_23,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_24,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_25,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_26,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_27,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_28,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_29,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_3,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_30,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_31,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_32,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_33,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_34,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_35,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_36,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_37,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_38,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_39,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_4,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_5,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_6,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_7,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_8,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_9,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_mean,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_0,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_1,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_10,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_11,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_12,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_13,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_14,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_15,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_16,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_17,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_18,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_19,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_2,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_20,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_21,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_22,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_23,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_24,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_25,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_26,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_27,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_28,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_29,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_3,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_30,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_31,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_32,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_33,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_34,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_35,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_36,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_37,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_38,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_39,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_4,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_5,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_6,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_7,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_8,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlin_9,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlinear_mean,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_nonlinear_std,y,y,y,y,y,y,y,y
computational_capacity_supplementary_kayson_mc_std,y,y,y,y,y,y,y,y
computational_capacity_total_capacity,y,y,y,y,y,y,y,y
degree_assortativity,y,y,y,y,y,y,y,y
degree_gini,y,y,y,y,y,y,y,y
density,y,y,y,y,y,y,y,y
density_bct,y,y,y,y,y,y,y,y
departure_from_normality_schur,y,y,y,y,y,y,y,y
diffusion_efficiency,y,y,y,y,y,y,y,y
directed_simplices_count,y,y,y,y,y,y,y,y
directed_simplices_max_size,y,y,y,y,y,y,y,y
effective_dimensionality,y,y,y,y,y,y,y,y
eta,missing,missing,missing,missing,missing,missing,missing,y
gamma,missing,missing,missing,missing,missing,missing,missing,y
global_efficiency,y,y,y,y,y,y,y,y
global_efficiency_dynamic,missing,missing,missing,missing,missing,missing,missing,y
gromov_hyperbolicity,missing,missing,y,missing,y,y,missing,y
ipc_ipc_deg1_mean,y,y,y,y,y,y,y,y
ipc_ipc_deg1_std,y,y,y,y,y,y,y,y
ipc_ipc_deg2_mean,y,y,y,y,y,y,y,y
ipc_ipc_deg2_std,y,y,y,y,y,y,y,y
ipc_ipc_deg3_mean,y,y,y,y,y,y,y,y
ipc_ipc_deg3_std,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_0,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_1,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_10,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_11,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_12,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_13,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_14,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_15,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_16,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_17,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_18,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_19,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_2,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_20,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_21,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_22,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_23,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_24,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_25,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_26,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_27,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_28,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_29,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_3,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_30,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_31,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_32,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_33,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_34,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_35,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_36,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_37,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_38,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_39,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_4,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_40,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_5,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_6,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_7,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_8,y,y,y,y,y,y,y,y
ipc_ipc_linear_lag_9,y,y,y,y,y,y,y,y
ipc_ipc_linear_mean,y,y,y,y,y,y,y,y
ipc_ipc_nonlinear_mean,y,y,y,y,y,y,y,y
ipc_ipc_total_mean,y,y,y,y,y,y,y,y
ipc_ipc_total_std,y,y,y,y,y,y,y,y
kernel_rank_max,y,y,y,y,y,y,y,y
kernel_rank_phase_diff_of_lambda_max_and_2nd,y,y,y,y,y,y,y,y
kernel_rank_phase_of_lambda_max,y,y,y,y,y,y,y,y
kernel_rank_thresholded_and_summed_0.01,y,y,y,y,y,y,y,y
kuramoto_averaged_synchronization_r_final,y,y,y,y,y,y,y,y
kuramoto_averaged_synchronization_r_mean,y,y,y,y,y,y,y,y
kuramoto_averaged_synchronization_r_mean_se,y,y,y,y,y,y,y,y
kuramoto_averaged_synchronization_r_std,y,y,y,y,y,y,y,y
kuramoto_averaged_synchronization_r_std_se,y,y,y,y,y,y,y,y
kuramoto_synchronization_r_final,y,y,y,y,y,y,y,y
kuramoto_synchronization_r_mean,y,y,y,y,y,y,y,y
kuramoto_synchronization_r_std,y,y,y,y,y,y,y,y
local_efficiency_stats_max,y,y,y,y,y,y,y,y
local_efficiency_stats_mean,y,y,y,y,y,y,y,y
local_efficiency_stats_min,y,y,y,y,y,y,y,y
local_efficiency_stats_std,y,y,y,y,y,y,y,y
mc_0,y,y,y,y,y,y,y,y
mc_1,y,y,y,y,y,y,y,y
mc_10,y,y,y,y,y,y,y,y
mc_11,y,y,y,y,y,y,y,y
mc_12,y,y,y,y,y,y,y,y
mc_13,y,y,y,y,y,y,y,y
mc_14,y,y,y,y,y,y,y,y
mc_15,y,y,y,y,y,y,y,y
mc_16,y,y,y,y,y,y,y,y
mc_17,y,y,y,y,y,y,y,y
mc_18,y,y,y,y,y,y,y,y
mc_19,y,y,y,y,y,y,y,y
mc_2,y,y,y,y,y,y,y,y
mc_20,y,y,y,y,y,y,y,y
mc_21,y,y,y,y,y,y,y,y
mc_22,y,y,y,y,y,y,y,y
mc_23,y,y,y,y,y,y,y,y
mc_24,y,y,y,y,y,y,y,y
mc_25,y,y,y,y,y,y,y,y
mc_26,y,y,y,y,y,y,y,y
mc_27,y,y,y,y,y,y,y,y
mc_28,y,y,y,y,y,y,y,y
mc_29,y,y,y,y,y,y,y,y
mc_3,y,y,y,y,y,y,y,y
mc_30,y,y,y,y,y,y,y,y
mc_31,y,y,y,y,y,y,y,y
mc_32,y,y,y,y,y,y,y,y
mc_33,y,y,y,y,y,y,y,y
mc_34,y,y,y,y,y,y,y,y
mc_35,y,y,y,y,y,y,y,y
mc_36,y,y,y,y,y,y,y,y
mc_37,y,y,y,y,y,y,y,y
mc_38,y,y,y,y,y,y,y,y
mc_39,y,y,y,y,y,y,y,y
mc_4,y,y,y,y,y,y,y,y
mc_40,y,y,y,y,y,missing,missing,y
mc_41,y,y,y,y,y,missing,missing,y
mc_42,y,y,y,y,y,missing,missing,y
mc_43,y,y,y,y,y,missing,missing,y
mc_44,y,y,y,y,y,missing,missing,y
mc_45,y,y,y,y,y,missing,missing,y
mc_46,y,y,y,y,y,missing,missing,y
mc_47,y,y,y,y,y,missing,missing,y
mc_48,y,y,y,y,y,missing,missing,y
mc_49,y,y,y,y,y,missing,missing,y
mc_5,y,y,y,y,y,y,y,y
mc_6,y,y,y,y,y,y,y,y
mc_7,y,y,y,y,y,y,y,y
mc_8,y,y,y,y,y,y,y,y
mc_9,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_0,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_1,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_10,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_11,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_12,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_13,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_14,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_15,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_16,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_17,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_18,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_19,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_2,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_20,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_21,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_22,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_23,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_24,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_25,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_26,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_27,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_28,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_29,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_3,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_30,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_31,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_32,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_33,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_34,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_35,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_36,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_37,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_38,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_39,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_4,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_5,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_6,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_7,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_8,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_9,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_mean,y,y,y,y,y,y,y,y
mc_input_scaling_0_1_mc_std,y,y,y,y,y,y,y,y
mc_lin_cut40_mc_0,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_1,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_10,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_11,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_12,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_13,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_14,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_15,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_16,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_17,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_18,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_19,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_2,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_20,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_21,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_22,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_23,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_24,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_25,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_26,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_27,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_28,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_29,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_3,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_30,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_31,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_32,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_33,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_34,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_35,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_36,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_37,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_38,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_39,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_4,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_5,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_6,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_7,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_8,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_9,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_mean,y,y,y,y,y,missing,missing,y
mc_lin_cut40_mc_std,y,y,y,y,y,missing,missing,y
mc_mean,y,y,y,y,y,y,y,y
mc_nonlin_cut40_mc_mean,y,y,y,y,y,missing,missing,y
mc_nonlin_cut40_mc_std,y,y,y,y,y,missing,missing,y
mc_nonlinear_input_scaling_0_1_mc_mean,y,y,y,y,y,y,y,y
mc_nonlinear_input_scaling_0_1_mc_std,y,y,y,y,y,y,y,y
mc_nonlinear_original_mc_mean,y,y,y,y,y,y,y,y
mc_nonlinear_original_mc_std,y,y,y,y,y,y,y,y
mc_std,y,y,y,y,y,y,y,y
modularity,y,y,y,y,y,y,y,y
modularity_dup,y,y,y,y,y,y,y,missing
modularity_static,missing,missing,missing,missing,missing,missing,missing,y
n_connected_components,y,y,y,y,y,y,y,y
nct_control_avg,y,y,y,y,y,y,y,y
nct_control_max,y,y,y,y,y,y,y,y
nct_control_n_nodes_10_percent,y,y,y,y,y,y,y,y
nct_control_n_nodes_50_percent,y,y,y,y,y,y,y,y
nct_control_n_nodes_90_percent,y,y,y,y,y,y,y,y
nct_control_std,y,y,y,y,y,y,y,y
nct_energies_max,y,y,y,y,y,y,y,y
nct_energies_n_nodes_10_percent,y,y,y,y,y,y,y,y
nct_energies_n_nodes_50_percent,y,y,y,y,y,y,y,y
nct_energies_n_nodes_90_percent,y,y,y,y,y,y,y,y
nct_energies_std,y,y,y,y,y,y,y,y
nct_energies_total,y,y,y,y,y,y,y,y
network_idx_computational,y,y,y,y,y,y,y,missing
network_idx_dynamic,y,y,y,y,y,y,y,missing
network_idx_further,y,y,y,y,y,y,y,missing
ollivier_ricci_curvature_orc_frac_neg,y,y,y,y,y,y,y,y
ollivier_ricci_curvature_orc_max,y,y,y,y,y,y,y,y
ollivier_ricci_curvature_orc_mean,y,y,y,y,y,y,y,y
ollivier_ricci_curvature_orc_median,y,y,y,y,y,y,y,y
ollivier_ricci_curvature_orc_min,y,y,y,y,y,y,y,y
ollivier_ricci_curvature_orc_skewness,y,y,y,y,y,y,y,y
ollivier_ricci_curvature_orc_std,y,y,y,y,y,y,y,y
omega,y,y,y,y,y,y,y,y
participation_coefficient_n_communities,y,y,y,y,y,y,y,y
participation_coefficient_n_communities_further,y,y,y,y,y,y,y,y
participation_coefficient_pc_frac_connector,y,y,y,y,y,y,y,y
participation_coefficient_pc_frac_connector_further,y,y,y,y,y,y,y,y
participation_coefficient_pc_mean,y,y,y,y,y,y,y,y
participation_coefficient_pc_mean_further,y,y,y,y,y,y,y,y
participation_coefficient_pc_median,y,y,y,y,y,y,y,y
participation_coefficient_pc_median_further,y,y,y,y,y,y,y,y
participation_coefficient_pc_std,y,y,y,y,y,y,y,y
participation_coefficient_pc_std_further,y,y,y,y,y,y,y,y
participation_coefficient_wmd_mean,y,y,y,y,y,y,y,y
participation_coefficient_wmd_mean_further,y,y,y,y,y,y,y,y
participation_coefficient_wmd_std,y,y,y,y,y,y,y,y
participation_coefficient_wmd_std_further,y,y,y,y,y,y,y,y
persistent_homology_ph_h0_entropy,y,y,y,y,y,y,y,y
persistent_homology_ph_h0_n_features,y,y,y,y,y,y,y,y
persistent_homology_ph_h0_persistence_mean,y,y,y,y,y,y,y,y
persistent_homology_ph_h0_persistence_std,y,y,y,y,y,y,y,y
persistent_homology_ph_h1_entropy,y,y,y,y,y,y,y,y
persistent_homology_ph_h1_n_features,y,y,y,y,y,y,y,y
persistent_homology_ph_h1_persistence_mean,y,y,y,y,y,y,y,y
persistent_homology_ph_h1_persistence_std,y,y,y,y,y,y,y,y
persistent_homology_ph_total_persistence,y,y,y,y,y,y,y,y
propagation_efficiency,y,y,y,y,y,y,y,y
proportion_long_range_connections_0.1,y,y,y,y,y,y,y,y
proportion_long_range_connections_0.3,y,y,y,y,y,y,y,y
proportion_long_range_connections_0.3956,y,y,y,y,y,y,y,y
proportion_long_range_connections_0.5,y,y,y,y,y,y,y,y
repertoire_sweep_weighted_by_distances_T_critical,y,y,y,y,y,y,y,y
repertoire_sweep_weighted_by_distances_diversity_critical,y,y,y,y,y,y,y,y
repertoire_sweep_weighted_by_distances_size_critical,y,y,y,y,y,y,y,y
resistance_distance_mean,y,y,y,y,y,missing,missing,missing
resistance_distance_std,y,y,y,y,y,missing,missing,missing
rich_club_coefficient_rc_k_at_max,y,y,y,y,y,y,y,y
rich_club_coefficient_rc_max_norm,y,y,y,y,y,y,y,y
rich_club_coefficient_rc_mean_norm,y,y,y,y,y,y,y,y
rich_club_coefficient_rc_regime_frac,y,y,y,y,y,y,y,y
rich_club_coefficient_rc_weighted_auc,y,y,y,y,y,y,y,y
richclub_avg_length,y,y,y,y,y,y,y,y
richclub_n_edges,y,y,y,y,y,y,y,y
spectral_gap,y,y,y,y,y,y,y,y
spectral_gap_fatemeh,y,y,y,y,y,y,y,y
spectral_radius,y,y,y,y,y,y,y,y
structural_complexity,y,y,y,y,y,y,y,y
synchronizability_eigenratio_eigenratio,y,y,y,y,y,y,y,y
synchronizability_eigenratio_lambda_2,y,y,y,y,y,y,y,y
synchronizability_eigenratio_lambda_N,y,y,y,y,y,y,y,y
targeted_attack_robustness_rob_random_auc,y,y,y,y,y,y,y,y
targeted_attack_robustness_rob_random_half,y,y,y,y,y,y,y,y
targeted_attack_robustness_rob_ratio,y,y,y,y,y,y,y,y
targeted_attack_robustness_rob_targeted_auc,y,y,y,y,y,y,y,y
targeted_attack_robustness_rob_targeted_half,y,y,y,y,y,y,y,y
topological_distance_mean,y,y,y,y,y,y,y,y
topological_distance_std,y,y,y,y,y,y,y,y
transitivity,y,y,y,y,y,y,y,y
transitivity_static,missing,missing,missing,missing,missing,missing,missing,y
wiring_cost,y,y,y,y,y,y,y,y
"""