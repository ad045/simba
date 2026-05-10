# Recommended features for each of the 60 network metrics.
#
# Naming convention (mirrors evaluate_further_metrics_utils.py):
#   scalar result  →  column name == metric_name
#   dict result    →  column name == f"{metric_name}_{key}"
#
# Organised by category. Each entry is the exact DataFrame column name.

# ── Static measures ───────────────────────────────────────────────────────────
STATIC_FEATURES = [
    "density",
    "avg_clustering",
    "avg_degree",
    "degree_assortativity",
    "modularity",
    "characteristic_path_length",
    "transitivity",
    "wiring_cost",
    # shortest_path_distance  →  currently returns raw N×N matrix, not a scalar
    "structural_complexity",
    "n_connected_components",
    "omega",
    "topological_distance_mean",
    "resistance_distance_mean",
    "degree_gini",
    "proportion_long_range_connections_0.5",   # highest-quantile threshold
    "directed_simplices_count",
]

# ── Dynamic measures ──────────────────────────────────────────────────────────
DYNAMIC_FEATURES = [
    "spectral_radius",
    "spectral_gap",
    "spectral_gap_fatemeh",          # duplicate of spectral_gap
    "global_efficiency",
    "diffusion_efficiency",
    "propagation_efficiency",
    "nct_control_avg",
    "nct_control_std",
    "nct_energies_total",
    "nct_energies_std",
    "novel_metastability_optimal_value",
    "synchronizability_eigenratio_eigenratio",
    "algebraic_connectivity_nx",     # duplicate of synchronizability_eigenratio lambda_2
    "kuramoto_synchronization_r_mean",
    "kuramoto_synchronization_r_std",
    "kuramoto_averaged_synchronization_r_mean",
    "kuramoto_averaged_synchronization_r_std",
    "community_synchronization_vulnerability_value",
    "participation_coefficient_dynamic_mean",
    "departure_from_normality",
    "departure_from_normality_schur",
]

# ── Computational measures ────────────────────────────────────────────────────
COMPUTATIONAL_FEATURES = [
    "kernel_rank_thresholded_and_summed_0.01",
    "kernel_rank_fatemeh",
    "effective_dimensionality",
    "multifunctionality",
    "computational_capacity_memory_capacity_total",
    "computational_capacity_nonlinear_capacity_total",
    "computational_capacity_total_capacity",
    "computational_capacity_notebook_memory_capacity_total_notebook",
    "computational_capacity_notebook_nonlinear_capacity_total_notebook",
    "computational_capacity_notebook_damicelli_memory_capacity_total_notebook",
    "computational_capacity_notebook_damicelli_nonlinear_capacity_total_notebook",
    "computational_capacity_supplementary_kayson_mc_mean",
    "computational_capacity_supplementary_kayson_mc_nonlinear_mean",
    "mc_original_mc_mean",
    "mc_nonlinear_original_mc_mean",
    "mc_lin_cut40_mc_mean",
    "mc_nonlin_cut40_mc_mean",
    "mc_input_scaling_0_1_mc_mean",
    "mc_nonlinear_input_scaling_0_1_mc_mean",
    "repertoire_sweep_weighted_by_distances_size_critical",
    "repertoire_sweep_weighted_by_distances_diversity_critical",
    "ipc_ipc_total_mean",
    "ipc_ipc_nonlinear_mean",
]

# ── Further measures ──────────────────────────────────────────────────────────
FURTHER_FEATURES = [
    "ollivier_ricci_curvature_orc_mean",
    "ollivier_ricci_curvature_orc_std",
    "ollivier_ricci_curvature_orc_frac_neg",
    "rich_club_coefficient_rc_weighted_auc",
    "rich_club_coefficient_rc_max_norm",
    "participation_coefficient_further_pc_mean",
    "participation_coefficient_further_pc_frac_connector",
    "persistent_homology_ph_h1_persistence_mean",
    "persistent_homology_ph_h1_entropy",
    "targeted_attack_robustness_rob_ratio",
    "algebraic_connectivity_further_fiedler_value_norm",
    "algebraic_connectivity_further_laplacian_spectral_gap",
    "basic_measures_avg_communicability",
    "gromov_hyperbolicity",              # scalar return
    "betweenness_centrality_stats_mean",
    "betweenness_centrality_stats_gini",
    "local_efficiency_stats_mean",
]

# ── All recommended features (flat list) ─────────────────────────────────────
ALL_RECOMMENDED_FEATURES = (
    STATIC_FEATURES
    + DYNAMIC_FEATURES
    + COMPUTATIONAL_FEATURES
    + FURTHER_FEATURES
)

# ── PCA subset: non-redundant, informative features ──────────────────────────
# Selection criteria:
#   - No constant features (density, avg_degree at fixed density)
#   - No near-exact duplicates (spectral_gap_fatemeh ≈ spectral_gap;
#     algebraic_connectivity_nx ≈ synchronizability_eigenratio lambda_2)
#   - One representative per family of MC variants
#   - Prefer the averaged/more-reliable version when two are similar
PCA_FEATURES = [
    # Structure / topology
    "avg_clustering",
    "degree_assortativity",
    "modularity",
    "wiring_cost",
    "structural_complexity",
    "resistance_distance_mean",
    "degree_gini",
    "directed_simplices_count",
    # Spectral / propagation
    "spectral_radius",
    "spectral_gap",
    "global_efficiency",
    # Dynamics
    "nct_control_avg",
    "nct_control_std",
    "novel_metastability_optimal_value",
    "synchronizability_eigenratio_eigenratio",
    "community_synchronization_vulnerability_value",
    "departure_from_normality_schur",
    # Computation
    "effective_dimensionality",
    "computational_capacity_supplementary_kayson_mc_mean",
    "computational_capacity_supplementary_kayson_mc_nonlinear_mean",
    "repertoire_sweep_weighted_by_distances_size_critical",
    "repertoire_sweep_weighted_by_distances_diversity_critical",
    "ipc_ipc_total_mean",
    "ipc_ipc_nonlinear_mean",
    # Geometry / topology
    "ollivier_ricci_curvature_orc_mean",
    "ollivier_ricci_curvature_orc_frac_neg",
    "rich_club_coefficient_rc_weighted_auc",
    "participation_coefficient_further_pc_mean",
    "persistent_homology_ph_h1_persistence_mean",
    "targeted_attack_robustness_rob_ratio",
    "betweenness_centrality_stats_mean",
    "betweenness_centrality_stats_gini",
    "local_efficiency_stats_mean",
]

# ── Brain trade-off axes ──────────────────────────────────────────────────────
# The brain negotiates several competing pressures simultaneously.
# GOALS partitions ALL_RECOMMENDED_FEATURES (77 features, each exactly once)
# into nine biologically motivated axes.
#
# Key tensions encoded here:
#   wiring_economy      ↔  global_integration   (cheap wire vs. short global paths)
#   local_segregation   ↔  global_integration   (modular specialisation vs. communication)
#   dynamical_stability ↔  dynamical_flexibility (order vs. metastability / criticality)
#   wiring_economy      ↔  structural_robustness (centralised = efficient but fragile)
#
# state_controllability (NCT), computational_capacity, and topological_complexity
# are each treated as independent axes rather than poles of a single tension.

GOALS = {

    # Spatial and metabolic cost of maintaining the connectivity pattern.
    # Directly reflects the "cost" side of the economy–efficiency trade-off.
    "wiring_economy": [
        "wiring_cost",                          # mean Euclidean edge length
        "proportion_long_range_connections_0.5",# fraction of costly long-range edges
        # "density",                              # total edge count (constant at fixed density)
        # "avg_degree",                           # mean connections per node (idem)
        "degree_gini",                          # degree heterogeneity: hubs concentrate cost
    ],

    # How efficiently information can flow between any two nodes globally.
    # The "benefit" side of the economy–efficiency trade-off; also opposes segregation.
    "global_integration": [
        "global_efficiency",                    # 1 / harmonic mean path length
        "characteristic_path_length",           # mean hop-count path (inverse of above)
        "resistance_distance_mean",             # effective resistance: integrates all paths
        "diffusion_efficiency",                 # random-walk traversal efficiency
        "propagation_efficiency",               # signal propagation speed
        "basic_measures_avg_communicability",   # Estrada communicability (all-path)
        "synchronizability_eigenratio_eigenratio", # λ_N/λ_2: width of sync window
    ],

    # Degree to which the network organises into internally cohesive,
    # externally sparse modules — the specialisation pole of the seg–int trade-off.
    "local_segregation": [
        "modularity",                           # Newman–Girvan Q
        "avg_clustering",                       # mean local triangle density
        "transitivity",                         # global clustering (network-level)
        "local_efficiency_stats_mean",          # efficiency within node neighbourhoods
        "participation_coefficient_further_pc_mean",           # mean cross-module bridging
        "participation_coefficient_further_pc_frac_connector", # fraction of connector hubs
        "participation_coefficient_dynamic_mean",              # same via dynamic calculator
        "community_synchronization_vulnerability_value",       # between/within coupling ratio
    ],

    # Tendency toward ordered, predictable, easily controllable dynamics.
    # High spectral radius or low Fiedler value push the network toward instability.
    "dynamical_stability": [
        "spectral_radius",                      # ρ(A): echo-state stability threshold at 1
        "spectral_gap",                         # leading eigenvalue gap: dominance of top mode
        # "spectral_gap_fatemeh",                 # duplicate of spectral_gap (alt. implementation)
        "algebraic_connectivity_nx",            # λ_2(L): diffusion / consensus speed
        "algebraic_connectivity_further_fiedler_value_norm",     # λ_2 / λ_max (normalised)
        "algebraic_connectivity_further_laplacian_spectral_gap", # λ_2 / λ_3: Laplacian gap
    ],

    # Capacity for metastability, criticality, and transitions between states.
    # The flexibility pole of the stability–flexibility trade-off.
    # Non-normality enables transient amplification even with ρ(A) < 1.
    "dynamical_flexibility": [
        "novel_metastability_optimal_value",            # peak metastability across K values
        "kuramoto_synchronization_r_mean",              # mean synchrony (intermediate = flexible)
        "kuramoto_synchronization_r_std",               # synchrony variance = metastability
        "kuramoto_averaged_synchronization_r_mean",     # averaged version (more reliable)
        "kuramoto_averaged_synchronization_r_std",      # averaged metastability
        # "departure_from_normality",                     # ||A − normal|| (transient amplification)
        "departure_from_normality_schur",               # same via Schur decomposition
    ],

    # Memory and nonlinear computation in the reservoir / echo-state framework.
    # Closely tied to dynamical flexibility: edge-of-chaos maximises memory capacity.
    "computational_capacity": [
        "effective_dimensionality",                     # participation ratio of eigenspectrum
        "kernel_rank_thresholded_and_summed_0.01",      # effective kernel rank (thresholded)
        "kernel_rank_fatemeh",                          # integer kernel rank (Fatemeh impl.)
        "multifunctionality",                           # fraction of linearly separable dichotomies
        # "computational_capacity_memory_capacity_total",
        # "computational_capacity_nonlinear_capacity_total",
        # "computational_capacity_total_capacity",
        # "computational_capacity_notebook_memory_capacity_total_notebook",
        # "computational_capacity_notebook_nonlinear_capacity_total_notebook",
        # "computational_capacity_notebook_damicelli_memory_capacity_total_notebook",
        # "computational_capacity_notebook_damicelli_nonlinear_capacity_total_notebook",
        # "computational_capacity_supplementary_kayson_mc_mean",
        # "computational_capacity_supplementary_kayson_mc_nonlinear_mean",
        # "mc_original_mc_mean",
        # "mc_nonlinear_original_mc_mean",
        # "mc_lin_cut40_mc_mean",
        # "mc_nonlin_cut40_mc_mean",
        # "mc_input_scaling_0_1_mc_mean",
        # "mc_nonlinear_input_scaling_0_1_mc_mean",
        "ipc_ipc_total_mean",                           # total IPC (all degrees)
        "ipc_ipc_nonlinear_mean",                       # nonlinear IPC (deg 2 + 3)
        "repertoire_sweep_weighted_by_distances_size_critical",      # state repertoire size
        "repertoire_sweep_weighted_by_distances_diversity_critical", # repertoire diversity
    ],

    # Energy cost and spatial heterogeneity of driving the network between
    # target states (network control theory / optimal control framework).
    # Orthogonal to reservoir-computing capacity: asks "can you steer it?" not
    # "can it compute?"
    "state_controllability": [
        "nct_control_avg",      # mean average controllability: Tr(W_K) / N
        "nct_control_std",      # node heterogeneity of controllability
        "nct_energies_total",   # total optimal control energy for a random state transition
        "nct_energies_std",     # node heterogeneity of control energy
    ],

    # Resilience to targeted hub removal and random failure.
    # Opposes wiring economy: distributed / redundant topologies are robust but costly.
    "structural_robustness": [
        "targeted_attack_robustness_rob_ratio", # R_targeted / R_random < 1 = hub-vulnerable
        "n_connected_components",               # fragmentation under damage (1 = connected)
        "degree_assortativity",                 # disassortative (r < 0) = hub-dependent = fragile
        "betweenness_centrality_stats_mean",    # average traffic load per node
        "betweenness_centrality_stats_gini",    # load inequality: high = bottleneck-prone
        "rich_club_coefficient_rc_weighted_auc",# AUC of normalised rich-club curve
        "rich_club_coefficient_rc_max_norm",    # peak rich-club coefficient
    ],

    # Higher-order geometric and topological structure that cannot be reduced
    # to pairwise measures. Captures mesoscale cycle structure, curvature,
    # and feedforward motifs — emergent from all trade-offs combined.
    "topological_complexity": [
        "structural_complexity",                # topological entropy-like scalar
        "omega",                                # small-world index (−1=lattice, 0=SW, +1=random)
        "topological_distance_mean",            # mean neighbourhood-similarity distance
        "directed_simplices_count",             # number of directed feedforward simplices
        "ollivier_ricci_curvature_orc_mean",    # mean Wasserstein curvature
        "ollivier_ricci_curvature_orc_std",     # curvature heterogeneity
        "ollivier_ricci_curvature_orc_frac_neg",# fraction of bottleneck (neg-curvature) edges
        "persistent_homology_ph_h1_persistence_mean", # mean H1 loop persistence
        "persistent_homology_ph_h1_entropy",    # topological diversity of cycles
        "gromov_hyperbolicity",                 # δ*: tree-likeness of global metric geometry
    ],
}

# ── 6-category version ────────────────────────────────────────────────────────
# Merges relative to GOALS:
#   wiring_economy + structural_robustness   → economy_and_robustness
#   dynamical_stability + dynamical_flexibility + state_controllability → network_dynamics
# Integration and segregation are kept separate (the most important named tension).
# Computational capacity and topological complexity each stay as independent lenses.

GOALS_6 = {

    # Physical cost of wiring AND resilience of that wiring to damage — two sides
    # of the same structural trade-off: cheap/centralised ↔ expensive/distributed.
    "economy_and_robustness": [
        # cost
        *GOALS["wiring_economy"],
        # resilience
        *GOALS["structural_robustness"],
    ],

    # How efficiently signals travel across the whole network.
    "global_integration": GOALS["global_integration"],

    # How strongly the network partitions into internally dense, externally sparse modules.
    "local_segregation": GOALS["local_segregation"],

    # Everything the connectivity pattern implies for the network's temporal behaviour:
    # eigenspectrum stability, metastability / criticality, and NCT controllability
    # all answer "what can this structure do dynamically?" from different angles.
    "network_dynamics": [
        *GOALS["dynamical_stability"],
        *GOALS["dynamical_flexibility"],
        *GOALS["state_controllability"],
    ],

    # Reservoir-computing / information-processing capacity.
    "computational_capacity": GOALS["computational_capacity"],

    # Higher-order geometric structure that is not reducible to pairwise metrics.
    "topological_complexity": GOALS["topological_complexity"],
}

# ── Causal / hierarchical version ────────────────────────────────────────────
# The flat GOALS dict mixes ontological levels. This version makes the causal
# structure explicit across four tiers:
#
#   Tier 1 — CONSTRAINT   : costs the brain cannot escape
#   Tier 2 — ENABLER (structural): how wiring is organised to allow function
#   Tier 3 — ENABLER (dynamical) : what the structure implies for the dynamics
#   Tier 4 — OBJECTIVE    : fitness-relevant outcomes selection operates on
#
# Causal chain:
#   wiring_economy (budget)
#     → communication_architecture (structural solution within budget)
#       → dynamical_regime (dynamical operating point implied by structure)
#         → computational_capacity  (what the network can do)
#         → structural_robustness   (whether it keeps doing it under damage)
#
# Note: integration and segregation are proximate objectives AND structural
# enablers; placed here as enablers and left to PCA to separate empirically.

GOALS_CAUSAL = {

    # ── Tier 1: constraint ────────────────────────────────────────────────────
    # Metabolic and spatial budget. Every other property is a solution within
    # this constraint; wiring_economy sets the feasible region.
    "wiring_economy": GOALS["wiring_economy"],

    # ── Tier 2: structural enablers ───────────────────────────────────────────
    # How the wiring budget is spent: the integration–segregation balance,
    # the geometric embedding, and higher-order topological organisation.
    # These are instruments that shape dynamics and thereby enable computation.
    "communication_architecture": [
        # integration side
        *GOALS["global_integration"],
        # segregation side
        *GOALS["local_segregation"],
        # higher-order geometry (shapes both sides)
        *GOALS["topological_complexity"],
    ],

    # ── Tier 3: dynamical enablers ────────────────────────────────────────────
    # The dynamical operating regime implied by the structure: how stable,
    # how metastable, and how steerable the network's activity is.
    # Proximate cause of computational capacity.
    "dynamical_regime": [
        *GOALS["dynamical_stability"],
        *GOALS["dynamical_flexibility"],
        *GOALS["state_controllability"],
    ],

    # ── Tier 4: objectives ────────────────────────────────────────────────────
    # What natural selection directly operates on.

    # Information processing: memory, nonlinear transformation, state diversity.
    "computational_capacity": GOALS["computational_capacity"],

    # Maintaining function under targeted attack or random failure.
    "structural_robustness": GOALS["structural_robustness"],
}
