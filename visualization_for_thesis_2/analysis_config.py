"""
analysis_config.py — Shared configuration for the 5 GNM analysis scripts.

Defines paths, feature lists, category mappings, and small helpers.
"""

import sys
from pathlib import Path

# ── vizman path ───────────────────────────────────────────────────────────────
VIZMAN_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/Pyvizman")
if str(VIZMAN_PATH) not in sys.path:
    sys.path.insert(0, str(VIZMAN_PATH))

# ── data / output paths ───────────────────────────────────────────────────────
DATA_PKL = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code"
    "/output/00_trade_off_analysis/all_datasets_precise_categories.pkl"
)
OUTPUT_DIR = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code"
    "/output/00_gnm_new_analyses" # trade_off_analysis/gnm_new_analyses"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── feature selection ─────────────────────────────────────────────────────────
# Columns that exist in the GNM dataset with <10 % NaN and meaningful signal.
# Columns with ~38 % NaN (kernel_rank_phase_diff, repertoire_sweep_*, effective_dim)
# are kept as optional extras but excluded from multivariate analyses.

GOOD_FEATURES = [
    # ── Topology / Static ────────────────────────────────────────────────────
    "transitivity",
    "avg_clustering",
    "modularity",
    "degree_gini",
    "degree_assortativity",
    "omega",
    "structural_complexity",
    "directed_simplices_count",
    "directed_simplices_max_size",
    "topological_distance_mean",
    "wiring_cost",
    "proportion_long_range_connections_0.3956",
    # ── Communication / Dynamic ───────────────────────────────────────────────
    "char_path_length",
    "global_efficiency",
    "diffusion_efficiency",
    "propagation_efficiency",
    "avg_communicability",
    "spectral_radius",
    "spectral_gap",
    "synchronizability_eigenratio_eigenratio",
    "algebraic_connectivity_fiedler_value",
    "algebraic_connectivity_laplacian_spectral_gap",
    "departure_from_normality_schur",
    "kuramoto_averaged_synchronization_r_mean",
    "kuramoto_averaged_synchronization_r_std",
    "community_synchronization_vulnerability_value",
    "participation_coefficient_pc_mean",
    "participation_coefficient_pc_frac_connector",
    # ── Computational / Memory ────────────────────────────────────────────────
    "mc_input_scaling_0_1_mc_mean",
    "mc_nonlinear_input_scaling_0_1_mc_mean",
    # ── Rich club / Spatial ───────────────────────────────────────────────────
    "rich_club_coefficient_rc_weighted_auc",
    "rich_club_coefficient_rc_regime_frac",
    "richclub_avg_length",
    # ── Curvature / Topology ──────────────────────────────────────────────────
    "ollivier_ricci_curvature_orc_mean",
    "ollivier_ricci_curvature_orc_std",
    # ── Persistent homology ───────────────────────────────────────────────────
    "persistent_homology_ph_h1_n_features",
    "persistent_homology_ph_h1_entropy",
    "persistent_homology_ph_total_persistence",
    # ── Robustness / Control ──────────────────────────────────────────────────
    "targeted_attack_robustness_rob_targeted_auc",
    "targeted_attack_robustness_rob_ratio",
    "nct_control_avg",
    "nct_energies_total",
]

# High-NaN features kept for individual heatmaps but excluded from multivariate
HIGH_NAN_FEATURES = [
    "kernel_rank_thresholded_and_summed_0.01",
    "effective_dimensionality",
    "repertoire_sweep_weighted_by_distances_T_critical",
    "repertoire_sweep_weighted_by_distances_diversity_critical",
]

# ── category mapping ──────────────────────────────────────────────────────────
FEATURE_CATEGORY = {
    "transitivity":                                     "Topology",
    "avg_clustering":                                   "Topology",
    "modularity":                                       "Topology",
    "degree_gini":                                      "Topology",
    "degree_assortativity":                             "Topology",
    "omega":                                            "Topology",
    "structural_complexity":                            "Topology",
    "directed_simplices_count":                         "Topology",
    "directed_simplices_max_size":                      "Topology",
    "topological_distance_mean":                        "Topology",
    "wiring_cost":                                      "Spatial",
    "proportion_long_range_connections_0.3956":         "Spatial",
    "char_path_length":                                 "Communication",
    "global_efficiency":                                "Communication",
    "diffusion_efficiency":                             "Communication",
    "propagation_efficiency":                           "Communication",
    "avg_communicability":                              "Communication",
    "spectral_radius":                                  "Spectral",
    "spectral_gap":                                     "Spectral",
    "synchronizability_eigenratio_eigenratio":          "Spectral",
    "algebraic_connectivity_fiedler_value":             "Spectral",
    "algebraic_connectivity_laplacian_spectral_gap":    "Spectral",
    "departure_from_normality_schur":                   "Spectral",
    "kuramoto_averaged_synchronization_r_mean":         "Dynamics",
    "kuramoto_averaged_synchronization_r_std":          "Dynamics",
    "community_synchronization_vulnerability_value":    "Dynamics",
    "participation_coefficient_pc_mean":                "Dynamics",
    "participation_coefficient_pc_frac_connector":      "Dynamics",
    "mc_input_scaling_0_1_mc_mean":                     "Memory",
    "mc_nonlinear_input_scaling_0_1_mc_mean":           "Memory",
    "rich_club_coefficient_rc_weighted_auc":            "Rich Club",
    "rich_club_coefficient_rc_regime_frac":             "Rich Club",
    "richclub_avg_length":                              "Rich Club",
    "ollivier_ricci_curvature_orc_mean":                "Curvature",
    "ollivier_ricci_curvature_orc_std":                 "Curvature",
    "persistent_homology_ph_h1_n_features":             "Topology",
    "persistent_homology_ph_h1_entropy":                "Topology",
    "persistent_homology_ph_total_persistence":         "Topology",
    "targeted_attack_robustness_rob_targeted_auc":      "Robustness",
    "targeted_attack_robustness_rob_ratio":             "Robustness",
    "nct_control_avg":                                  "Control",
    "nct_energies_total":                               "Control",
    # high-nan extras
    "kernel_rank_thresholded_and_summed_0.01":          "Memory",
    "effective_dimensionality":                         "Memory",
    "repertoire_sweep_weighted_by_distances_T_critical":   "Dynamics",
    "repertoire_sweep_weighted_by_distances_diversity_critical": "Dynamics",
}

CATEGORY_COLORS = {
    "Topology":      "#394D73",   # NIGHT_BLUE
    "Spatial":       "#E84653",   # LECKER_RED
    "Communication": "#3FA5C4",   # LAKE_BLUE
    "Spectral":      "#A6587C",   # PURPLE
    "Dynamics":      "#F99465",   # ORANGE
    "Memory":        "#5DC400",   # GRASS_GREEN
    "Rich Club":     "#E6B213",   # YELLOW
    "Curvature":     "#44cfcf",   # TEAL
    "Robustness":    "#BF003F",   # DEEP_RED
    "Control":       "#6A7870",   # OLIVE_GRAY
}

# Short human-readable labels
PROPERTY_NAMES = {
    "transitivity":                                     "Transitivity",
    "avg_clustering":                                   "Clustering",
    "modularity":                                       "Modularity",
    "degree_gini":                                      "Degree Gini",
    "degree_assortativity":                             "Assortativity",
    "omega":                                            "Omega",
    "structural_complexity":                            "Struct. Complexity",
    "directed_simplices_count":                         "Simplices Count",
    "directed_simplices_max_size":                      "Simplices Max Size",
    "topological_distance_mean":                        "Topol. Distance",
    "wiring_cost":                                      "Wiring Cost",
    "proportion_long_range_connections_0.3956":         "Long-range frac.",
    "char_path_length":                                 "Path Length",
    "global_efficiency":                                "Global Efficiency",
    "diffusion_efficiency":                             "Diffusion Effic.",
    "propagation_efficiency":                           "Propagation Effic.",
    "avg_communicability":                              "Communicability",
    "spectral_radius":                                  "Spectral Radius",
    "spectral_gap":                                     "Spectral Gap",
    "synchronizability_eigenratio_eigenratio":          "Sync. Eigenratio",
    "algebraic_connectivity_fiedler_value":             "Fiedler Value",
    "algebraic_connectivity_laplacian_spectral_gap":    "Laplacian Spec. Gap",
    "departure_from_normality_schur":                   "Depart. Normality",
    "kuramoto_averaged_synchronization_r_mean":         "Kuramoto r (mean)",
    "kuramoto_averaged_synchronization_r_std":          "Kuramoto r (std)",
    "community_synchronization_vulnerability_value":    "Sync. Vulnerability",
    "participation_coefficient_pc_mean":                "Part. Coeff. Mean",
    "participation_coefficient_pc_frac_connector":      "Connector Hub Frac.",
    "mc_input_scaling_0_1_mc_mean":                     "Memory Capacity",
    "mc_nonlinear_input_scaling_0_1_mc_mean":           "Nonlinear MC",
    "rich_club_coefficient_rc_weighted_auc":            "Rich-Club AUC",
    "rich_club_coefficient_rc_regime_frac":             "Rich-Club Regime",
    "richclub_avg_length":                              "Rich-Club Avg Length",
    "ollivier_ricci_curvature_orc_mean":                "ORC Mean",
    "ollivier_ricci_curvature_orc_std":                 "ORC Std",
    "persistent_homology_ph_h1_n_features":             "PH H1 Features",
    "persistent_homology_ph_h1_entropy":                "PH H1 Entropy",
    "persistent_homology_ph_total_persistence":         "PH Total Persist.",
    "targeted_attack_robustness_rob_targeted_auc":      "Robustness AUC",
    "targeted_attack_robustness_rob_ratio":             "Robustness Ratio",
    "nct_control_avg":                                  "NCT Avg Control",
    "nct_energies_total":                               "NCT Total Energy",
    "kernel_rank_thresholded_and_summed_0.01":          "Kernel Rank",
    "effective_dimensionality":                         "Eff. Dimensionality",
    "repertoire_sweep_weighted_by_distances_T_critical":    "Metastab. T_crit",
    "repertoire_sweep_weighted_by_distances_diversity_critical": "Metastab. Diversity",
    "eta":   r"$\eta$",
    "gamma": r"$\gamma$",
}


# ── small helpers ─────────────────────────────────────────────────────────────

def label(col: str) -> str:
    """Human-readable label for a column name."""
    return PROPERTY_NAMES.get(col, col)


def load_gnm(data_pkl: Path = DATA_PKL):
    """Load GNM dataset from pickle."""
    import pickle, pandas as pd
    with open(data_pkl, "rb") as f:
        d = pickle.load(f)
    return d["hcp_schaefer_100_dataset_gnm"]


def load_all(data_pkl: Path = DATA_PKL):
    """Load all datasets from pickle."""
    import pickle
    with open(data_pkl, "rb") as f:
        return pickle.load(f)


def gnm_pivot(df, value_col, agg="mean"):
    """
    Pivot a GNM dataframe to a 2D (eta × gamma) array.
    Returns (pivot_df, eta_vals_sorted, gamma_vals_sorted).
    """
    import pandas as pd
    p = df.groupby(["eta", "gamma"])[value_col].agg(agg).reset_index()
    pivot = p.pivot(index="eta", columns="gamma", values=value_col)
    return pivot


def cm_to_inch(cm_tuple):
    return (cm_tuple[0] / 2.54, cm_tuple[1] / 2.54)
