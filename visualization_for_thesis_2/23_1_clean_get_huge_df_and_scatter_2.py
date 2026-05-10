import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from vizman import viz

from config import COLORS, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES, emp_dataset_and_experiment_pairs

# ==========================================
# 1. Configuration
# ==========================================

OUTPUT_BASE  = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output")
OUTPUT_FOLDER = OUTPUT_BASE / "00_trade_off_analysis"
OUTPUT_FOLDER.mkdir(exist_ok=True)

GNM_DATASET    = "hcp_schaefer_100_dataset"
GNM_EXPERIMENT = "11_mst_2500_animal_0"

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

# ==========================================
# 2. Category Definitions
# ==========================================

# One figure per category will be produced.  Columns absent from the data are silently skipped.
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

# Flat list for downstream PCA (excludes eta/gamma and energy)
PRECISE_CATEGORIES = [
    col for cat_cols in CATEGORIES.values()
    for col in cat_cols
    if col not in ("eta", "gamma", "energy")
]


# ==========================================
# 3. Data Loading
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
# 4. Plotting
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
            axs[i].set_ylabel(f"$f_{{\\rm LR\\,({label.replace('%', r'\%')})}}$")
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
# 5. Main
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
    print("Plotting LR-fraction scatters...")
    plot_lr_fraction_scatters(dict_with_all_datasets, OUTPUT_FOLDER)

    print("\nDone.")
