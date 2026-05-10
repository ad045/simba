
# NEW, built together with claude
remaining_properties = [
    
    # # "transitivity", # corr too highly with avg_clustering
    "avg_clustering", 
    #     # # omega
        "modularity", 
        "degree_gini", # too high corr with spectral_radius
        "degree_assortativity", 
    #     
        "structural_complexity", 
        
        # REMOVE AGAIN? 
        "resistance_distance", 
        
        "directed_simplices_count", # Higher order
    #     "directed_simplices_max_size",
        
    #     # "char_path_length", 
        "global_efficiency", # corr too highly with omega
        # "diffusion_efficiency",
    #     "propagation_efficiency", # corr too highly with diffusion_efficiency
    #     # "avg_communicability", # corr too highly with spectral_radius # !!!!!!!
    #     # "topological_distance_mean", 
    #     # "topological_distance_std",

    #     # "avg_edge_distance", 
        "wiring_cost",
    # "proportion_long_range_connections_0.3956", 
    # "omega", 
    # "resistance_distance", # ??? 
        
    #     # "richclub_n_edges", 
    #     # "richclub_avg_length",
    #     # "rich_club_coefficient_rc_max_norm",
    #     # "rich_club_coefficient_rc_mean_norm",
    #     "rich_club_coefficient_rc_k_at_max",
    #     # "rich_club_coefficient_rc_regime_frac",
    #     # "rich_club_coefficient_rc_weighted_auc",

        "spectral_radius", 
        "spectral_gap", # Has exactly the same output as Fatemehs version, bc it is identical (lambda_n - lambda_(n-1))
    #     "synchronizability_eigenratio_eigenratio", # lambda_n / lambda_2

    #     # "kernel_rank_thresholded_and_summed_0.01",          # np.sum(np.abs(eigs) > threshold * np.abs(eigs).max())
    #     # "kernel_rank_phase_diff_of_lambda_max_and_2nd",     # np.angle(eigs[np.argsort(np.abs(eigs))[-2]]) - np.angle(eigs.max())
        
    #     "algebraic_connectivity_fiedler_value", # Relies on a different library, I hope? 
    #     # "algebraic_connectivity_fiedler_value_norm",
    #     "algebraic_connectivity_laplacian_spectral_gap",
    #     # "algebraic_connectivity_fiedler_bipartition_balance",
        
    #     "departure_from_normality_schur",

    #     # "effective_dimensionality", # corr too highly with omega

    #     "repertoire_sweep_weighted_by_distances_T_critical",
    #     "repertoire_sweep_weighted_by_distances_size_critical",
    #     "repertoire_sweep_weighted_by_distances_diversity_critical", 
        
    #     # "kuramoto_averaged_synchronization_r_final",
    #     # "kuramoto_averaged_synchronization_r_mean",
    #     # "kuramoto_averaged_synchronization_r_mean_se",
    #     # "kuramoto_averaged_synchronization_r_std",
    #     # "kuramoto_averaged_synchronization_r_std_se",

    #     "participation_coefficient_n_communities",
    #     # "participation_coefficient_pc_mean", # corr too highly with omega
    #     # "participation_coefficient_pc_median",
    #     "participation_coefficient_pc_std",
    #     "participation_coefficient_pc_frac_connector", # needs to be put into relation with Q
    #     # "participation_coefficient_wmd_std",

    # "Ollivier-Ricci Curvature": [ # If too annoying to describe, keep only one. 
    #     # "ollivier_ricci_curvature_orc_mean",
    #     # "ollivier_ricci_curvature_orc_median",
    #     # "ollivier_ricci_curvature_orc_std", 
    #     "ollivier_ricci_curvature_orc_min",
    #     # "ollivier_ricci_curvature_orc_max",
    #     "ollivier_ricci_curvature_orc_skewness",
    #     # "ollivier_ricci_curvature_orc_frac_neg",
    # ],
    # # "Persistent Homology (TDA)": [
    # #     # "persistent_homology_ph_h1_n_features",
    # #     # "persistent_homology_ph_h1_persistence_mean",
    # #     # "persistent_homology_ph_h1_entropy",
    # #     # "persistent_homology_ph_total_persistence",
    # # ],
    #     "targeted_attack_robustness_rob_targeted_auc",
    #     # "targeted_attack_robustness_rob_targeted_half",
    #     "targeted_attack_robustness_rob_random_auc", # CHANGED: WAS NOT ACTIVATED AFTER TALKING TO KAYSONcorr too highly with 'char_path_length'
    #     # "targeted_attack_robustness_rob_random_half",
    #     # "targeted_attack_robustness_rob_ratio", # CHANGED: WAS ACTIVATED AFTER TALKING TO KAYSON
        
    #     # "community_synchronization_vulnerability_value", # too high corr with participation_coefficient_pc_frac_connector
    #     "community_synchronization_vulnerability_n_communities",
        "nct_control_avg",
        "nct_control_std",  
    #     # "nct_control_max",
    #     # "nct_control_n_nodes_90_percent",
    #     # "nct_control_n_nodes_50_percent",
    #     # "nct_control_n_nodes_10_percent",
    # # "Network Control Theory - Control Energy": [
    # #     "nct_energies_total",
    # #     "nct_energies_std", 
    # #     "nct_energies_max",
    # #     "nct_energies_n_nodes_90_percent",
    # #     "nct_energies_n_nodes_50_percent",
    # #     "nct_energies_n_nodes_10_percent",
    # # ],
    # # "Computational Capacity": [
    # #     "computational_capacity_memory_capacity_total",
    # #     # "computational_capacity_memory_timescale",
    # #     "computational_capacity_nonlinear_capacity_total",
    # #     # "computational_capacity_cubic_capacity_total",
    #     # "computational_capacity_cross_capacity_total",
    #     # "computational_capacity_memory_nonlinear_ratio",
    #     # "computational_capacity_total_capacity",
    #     # "computational_capacity_state_dimensionality", 
    #     # "computational_capacity_state_entropy",
    #     # "computational_capacity_state_rank",
    #     # "computational_capacity_separation_ratio",
    #     # "computational_capacity_lyapunov_exponent",
    # # ],

    #     # "mc_input_scaling_0_1_mc_mean",
    #     # "mc_nonlinear_input_scaling_0_1_mc_mean", 
    #     "ipc_ipc_deg1_mean", 
    #     "ipc_ipc_deg2_mean"
        
    # # "Reservoir Computing - Basic Measures": 
    # # "Memory Capacity - Full Lag Profile": [
    # #     # [f"mc_{i}" for i in range(1, 50)]
    # #     # "mc_mean" # , "mc_std"]
    # # ],
    
]