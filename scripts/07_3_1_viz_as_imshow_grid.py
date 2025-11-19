"""
Using the visualization module for connectome analysis pipeline.
This script generates imshow-based landscape plots for gridded data.
"""

import os
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer


class GridVisualizer(PipelineVisualizer):
    """Extension of PipelineVisualizer for gridded data visualization using imshow."""
    
    def __init__(self, 
                 mode: str="draft", # "presentation"
                 **kwargs):
        """Initialize GridVisualizer with optional mode.
        Args:
            mode: If "draft", use parent's colorbar formatting. In "presentation" mode, use aesthetic formating. 
        """
        super().__init__(**kwargs)
        self.mode = mode 
        
        
    def _format_colorbar(self, metric_name: str, cbar) -> None:
        """Format colorbar with appropriate label."""
        
        # In draft mode, use parent's formatting
        if self.mode == "draft":
            super()._format_colorbar(metric_name, cbar)

        elif self.mode == "presentation":
            # Aesthetic formatting ("presentation" mode)
            from matplotlib.ticker import ScalarFormatter
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 3))  # Use scientific notation outside this range
            cbar.ax.yaxis.set_major_formatter(formatter)
            
            # print(cbar.ax.yaxis)
            # # Get the highest and lowest selected ticks
            # ticks = cbar.ax.get_yticks()
            # vmin = ticks[0]
            # vmax = ticks[-1]
            
            ticks = cbar.ax.get_yticks()
            vmin, vmax = cbar.ax.get_ylim()
            visible_ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
            lowest_tick = visible_ticks[0]
            highest_tick = visible_ticks[-1]
            
            # Show only min and max ticks
            # vmin, vmax = cbar.mappable.get_clim() # cbar.get_clim()
            cbar.set_ticks([lowest_tick, highest_tick]) # [vmin, vmax])

        else: 
            raise ValueError(f"Unknown mode (see colorbar code): {self.mode}")
        
        

    def plot_metric_landscape_grid(self, df: pd.DataFrame,
                                   dot_color: str = "steelblue",
                                   title: str = "",
                                   cmap: str = "hot",
                                   savepath: Path = None,
                                   metric_name: str = None,
                                   show: bool = True,
                                   show_dots: bool = False,
                                   point_size: int = 8,
                                   vmin: float = None,
                                   eta_span: list[float] = None,
                                   gamma_span: list[float] = None,
                                   vmax: float = None,
                                   ax: plt.Axes = None,
                                   annotate_extremes: bool = False,
                                   show_colorbar: bool = True,
                                   estimated_indiv_connectomes: pd.DataFrame = None,
                                   duplicate_handling: str = "mean",
                                   show_number_samples: bool = False,
                                   interpolation: str = 'nearest') -> tuple:
        """
        Plot a metric landscape using imshow for gridded data.
        
        Args:
            df: DataFrame with columns ['eta', 'gamma', metric_name]
            interpolation: Interpolation method for imshow ('nearest', 'bilinear', 'bicubic', etc.)
            All other args match the Voronoi version
        """
        
        # Filter data by spans if provided
        if eta_span:
            df = df[(df["eta"] >= eta_span[0]) & (df["eta"] <= eta_span[1])]
        if gamma_span:
            df = df[(df["gamma"] >= gamma_span[0]) & (df["gamma"] <= gamma_span[1])]
        
        # Prepare data using parent class method
        points, metric_values, _ = self._prepare_landscape_data(
            df, metric_name, normalize_aspect=False, duplicate_handling=duplicate_handling
        )
        
        # Create grid
        grid_data, eta_edges, gamma_edges = self._create_grid_from_points(
            points, metric_values
        )
        
        # Create or use provided axis
        if ax is None and self.mode == "presentation":
            fig, ax = plt.subplots(figsize=(viz.cm_to_inch((18, 18*0.52/0.62))), dpi=150) # (6.2, 5.2))
        elif ax is None and self.mode == "draft":
            fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=100) # (6.2, 5.2))
        else:
            fig = ax.get_figure()
        
        # Setup colormap
        _, colormap, vmin, vmax = self._setup_colormap(metric_values, vmin, vmax, cmap)
        
        # Plot using imshow
        extent = [eta_edges[0], eta_edges[-1], gamma_edges[0], gamma_edges[-1]]
        im = ax.imshow(grid_data, origin='lower', extent=extent,
                      aspect='equal', cmap=colormap, vmin=vmin, vmax=vmax,
                      interpolation=interpolation)
        
        # Add colorbar
        if show_colorbar:
            sm = cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            self._format_colorbar(metric_name, cbar)
        
        # Add sample points and annotations
        self._add_sample_points(ax, df, points, metric_name, show_dots, dot_color, point_size)
        self._add_estimated_connectomes(ax, estimated_indiv_connectomes, dot_color)
        
        if annotate_extremes:
            self._annotate_extreme_points(ax, df, metric_name)
        
        # Format plot
        if show_number_samples:
            title = f"{title} ({len(points)} shown connectomes)"
        
        xlim = [eta_edges[0], eta_edges[-1]]
        ylim = [gamma_edges[0], gamma_edges[-1]]
        self._format_landscape_plot(ax, xlim, ylim, title)
        
        # Save/show
        self._handle_plot_output(fig, ax, savepath, show)
        
        return fig, ax
    
    def _create_grid_from_points(self, points: np.ndarray, values: np.ndarray,
                                 grid_resolution: int = None) -> tuple:
        """
        Create a regular grid from scattered points.
        
        Args:
            points: Nx2 array of (eta, gamma) coordinates
            values: N array of metric values
            grid_resolution: Number of grid points (if None, use unique values)
        
        Returns:
            grid_data: 2D array of gridded values
            eta_edges: 1D array of eta bin edges
            gamma_edges: 1D array of gamma bin edges
        """
        # Get unique sorted values for each dimension
        unique_eta = np.unique(points[:, 0])
        unique_gamma = np.unique(points[:, 1])
        
        if grid_resolution is None:
            # Use actual grid points
            eta_edges = unique_eta
            gamma_edges = unique_gamma
        else:
            # Create regular grid with specified resolution
            eta_edges = np.linspace(points[:, 0].min(), points[:, 0].max(), grid_resolution)
            gamma_edges = np.linspace(points[:, 1].min(), points[:, 1].max(), grid_resolution)
        
        # Initialize grid with NaN
        grid_data = np.full((len(gamma_edges), len(eta_edges)), np.nan)
        
        # Fill grid with values
        for point, value in zip(points, values):
            eta_idx = np.argmin(np.abs(eta_edges - point[0]))
            gamma_idx = np.argmin(np.abs(gamma_edges - point[1]))
            grid_data[gamma_idx, eta_idx] = value
        
        return grid_data, eta_edges, gamma_edges


if __name__ == "__main__":
    
    ##### CONFIG STUFF #######################

    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/80_more_animals_animal_169/all_metrics_for_80_more_animals_animal_169.csv
# /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/hcp_schaefer_100_dataset/04_big_overnight_run/summary_indiv_portrait_for_exp_04_big_overnight_run_2.csv
# /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/hcp_schaefer_100_dataset/76_90000_samples_animal_206/summary_indiv_portrait_for_exp_76_90000_samples_animal_206.csv
    # dataset_name = "kaysons_generated_networks_propagation" # diffusion" # 
    dataset_name = "hcp_schaefer_100_dataset" 
    experiment_name = "07_high_res_90_000_plot" # 05_second_big_overnight_run_10201" # 07_high_res_90_000_plot" # "05_second_big_overnight_run_10201"
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/80_more_animals_animal_0/all_metrics_for_80_more_animals_animal_0.csv
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/hcp_schaefer_100_dataset/12_seed_chimp_206_idx_0/all_metrics_for_12_seed_chimp_206_idx_0.csv
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/hcp_schaefer_100_dataset/11_consensus_seed_idx_0/all_metrics_for_11_consensus_seed_idx_0.csv
    # dataset_name = "suarez_MaMI_dataset" # hcp_schaefer_100_dataset" # "suarez_MaMI_dataset"
    # experiment_name = "80_more_animals_animal_124" # 80_more_animals_animal_22" # 13_seeds_of_all_animals_density1_animal_0" # 13_seeds_of_all_animals_density1_animal_103" # 12_seed_chimp_206_idx_0" # 11_consensus_seed_idx_0" # 80_more_animals_animal_0" # 12_seed_chimp_206_idx_0" # 11_consensus_seed_idx_0" # 75_10000_samples_hopefully_no_lost_entries_gamma_minus0p1_to_1_animal_206_identical_version_just_without_minus_etc" # 80_more_animals_animal_103" # 10_serious_sweep_copy_idx_96" # 09_finally_working_idx_0" # 75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206" # 76_90000_samples_animal_206" # 04_big_overnight_run" # 01_first_bigger_run_animal_0" # 75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206" # 0" 
    #  "76_90000_samples_animal_206" 
        # "75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_0" 
        # File i am typically doing everything with, but it has this line?: 75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206" 
        # all metrics: "75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206" (with appendix ==  "_updated")
        # high resolution: 76_90000_samples_animal_206"
    appendix = "" # _updated" # or ""^
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/13_seeds_of_all_animals_density1_animal_169/all_metrics_for_13_seeds_of_all_animals_density1_animal_169.csv
    base_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/{experiment_name}")
    # save_path = base_path / f"all_computational_metrics_for_{experiment_name}_updated_combined.csv"
    save_path = base_path / f"all_metrics_for_{experiment_name}.csv"
    
    # Turn on/off regarding if one wants to use the original or the new metrics 
    # ORIGINAL METRICS OR NEW ONES
    only_all_metrics_file = False # True  # True #  False #  True # False # True 
    bin_to_100 = True # False # TODO: DOES NOT WORK YET. 
    mode = "draft" # presentation" # draft" # presentation"  # "draft" or "presentation"
    # "terrain" # cubehelix" 
    ##########################################
    if only_all_metrics_file: 
        df_paths = [base_path / f"all_metrics_for_{experiment_name}{appendix}.csv"]
    else: 
        df_static_path = base_path / f"all_static_metrics_for_{experiment_name}_updated.csv"
        df_static = pd.read_csv(df_static_path) if df_static_path.exists() else pd.DataFrame()
        df_dynamic_path = base_path / f"all_dynamic_metrics_for_{experiment_name}_updated.csv"
        df_dynamic = pd.read_csv(df_dynamic_path) if df_dynamic_path.exists() else pd.DataFrame()
        df_computational_path = base_path / f"all_computational_metrics_for_{experiment_name}_updated.csv"
        df_computational = pd.read_csv(df_computational_path) if df_computational_path.exists() else pd.DataFrame()
        df_combined = pd.concat([df_static, df_dynamic, df_computational], axis=1)
        df_combined.to_csv(save_path)
        df_paths = [save_path]

    # Change this here if individual points (minimum estimates, best fits) should be shown 
    plot_indiv_connectomes = "portrait" #  None # "portrait" # or "energy" or None
    duplicate_handling = "mean" # first" # "mean"
    show_number_samples = True
    eta_span = [-8, 3]
    gamma_span = [-0.1, 1]
    
    # Output directory
    parent_folder = Path(df_paths[0]).parent
    if duplicate_handling == "first" or duplicate_handling == "last": 
        save_path = parent_folder / f"figures_grid_only_{mode}_{duplicate_handling}"
    elif duplicate_handling == "mean": 
        save_path = parent_folder / f"figures_grid_only_{mode}_mean"
    else: 
        print("Attention: Duplicate handling is not really set.")
        exit()
    
    # --- 2. Define Metrics to Plot ---
    lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 12, 15, 18, 20, 49]
    # metrics_to_plot = "all" 
    metrics_to_plot = [  # or "all" to display all of the metrics in the csv
        ############### ORIG ####################################
        "MaxCriteria",
        "avg_communicability", 
        "global_efficiency", 
        "modularity", 
        "avg_clustering", 
        # "avg_degree", 
        "transitivity", 
        "avg_edge_distance", 
        "char_path_length", 
        "richclub_n_edges", 
        "richclub_avg_length", 
        "mc_mean", 
        "mc_std", 
        "wiring_cost", 
        "mean_mc_divided_by_wiring_cost",
        "mc_5_divided_by_wiring_cost", 
    ] + [f"mc_{lag}" for lag in lags_to_plot] + [
    ############### NEW #####################################
        # "density",
        "avg_clustering",
        # "avg_degree",
        
        "directed_simplices_count",
        "directed_simplices_max_size",
        # ---
        "directed_simplices_combined_with_n_components",
        
        "degree_assortativity",
        "modularity",
        "transitivity",
        "topological_distance_mean",
        "topological_distance_std",
        "degree_gini",
        "wiring_cost",
        "structural_complexity",
        "n_connected_components",
        "omega","spectral_radius",
        "spectral_gap","spectral_gap_fatemeh",
        "global_efficiency",
        "diffusion_efficiency",
        "propagation_efficiency",
        "nct_control_avg","nct_control_std",
        "nct_control_max","nct_control_n_nodes_90_percent",
        "nct_control_n_nodes_50_percent",
        "nct_control_n_nodes_10_percent",
        "nct_energies_total","nct_energies_std",
        "nct_energies_max","nct_energies_n_nodes_90_percent",
        "nct_energies_n_nodes_50_percent",
        "nct_energies_n_nodes_10_percent",
        
        "metastability_global",
        "metastability_local_mean",
        "metastability_local_std",
        "metastability_local_skewness",
        "metastability_local_kurtosis",
        
        "synchronizability_eigenratio_eigenratio",
        "synchronizability_eigenratio_lambda_2",
        "synchronizability_eigenratio_lambda_N",
        "algebraic_connectivity_nx",
        "kuramoto_synchronization",
        "community_synchronization_vulnerability_vulnerability",
        "community_synchronization_vulnerability_n_communities",
        
        "kernel_rank_thresholded_and_summed_0.01",
        "kernel_rank_max",
        "kernel_rank_phase_of_lambda_max",
        "kernel_rank_phase_diff_of_lambda_max_and_2nd",
        "kernel_rank_fatemeh",
        
        "effective_dimensionality",
        "multifunctionality"

    ] 
    
    plot_combined_lag_plot = False
    
    print(f"Starting Grid-based visualization process...")
    print(f"Output will be saved to: {save_path}\n")

    # --- 3. Load Data ---
    print("Loading and combining data...")
    try:
        if not df_paths:
            raise ValueError("Input 'df_paths' is an empty list.")
        
        if len(df_paths) > 1:
            gnm_results_df = generate_entire_df(df_paths)
        else:
            gnm_results_df = pd.read_csv(df_paths[0], index_col=False)
            if metrics_to_plot == "all": 
                metrics_to_plot = list(set(gnm_results_df.columns.tolist()) - set(['eta', 'gamma', 'animal_id'])) # TODO: exclude here anything else that is not a metric.
            
        if gnm_results_df.empty:
            raise ValueError("Dataframe is empty after loading.")
        # if os.path.exists(path_to_best_gamma_and_eta_estimations) and plot_indiv_connectomes is not None:
        if plot_indiv_connectomes is not None:
            path_to_best_gamma_and_eta_estimations = parent_folder / f"min_{plot_indiv_connectomes}_results.csv"
            df_best_gamma_and_eta_estimates = pd.read_csv(path_to_best_gamma_and_eta_estimations)
        else:
            df_best_gamma_and_eta_estimates = None
                
    except Exception as e:
        print(f"ERROR: Failed to load data. {e}")
        exit()

    visualizer = GridVisualizer(mode=mode)
    
    # --- 4. Generate Individual Plots in a Loop ---
    for metric in metrics_to_plot:
        print(f"--- Generating Grid plot for metric: {metric} ---")
        try:
            # --- Prepare data for the specific metric ---
            df = gnm_results_df.copy()

            # Add computed metrics if needed
            if metric == "mean_mc_divided_by_wiring_cost": 
                df["mean_mc_divided_by_wiring_cost"] = (
                    pd.to_numeric(df["mc_mean"], errors='coerce') / 
                    pd.to_numeric(df["wiring_cost"], errors='coerce')
                )
            
            if metric == "mc_5_divided_by_wiring_cost": 
                df["mc_5_divided_by_wiring_cost"] = (
                    pd.to_numeric(df["mc_5"], errors='coerce') / 
                    pd.to_numeric(df["wiring_cost"], errors='coerce')
                )
            
            if metric == "directed_simplices_combined_with_n_components": 
                # Take directed_simplices_count, if n_components is == 1. Otherwise, set it to -1. 
                df["directed_simplices_combined_with_n_components"] = np.where(
                    pd.to_numeric(df["n_connected_components"], errors='coerce') == 1,
                    pd.to_numeric(df["directed_simplices_count"], errors='coerce'),
                    np.nan
                )
                df["directed_simplices_combined_with_n_components"] = np.log(df["directed_simplices_combined_with_n_components"])
                
            df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
            df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
            
            try:
                # Exact match for mc_lag columns
                if metric.startswith("mc_") and metric.split("_")[1].isdigit():
                    metric_col_name = metric
                    if metric not in df.columns:
                        raise StopIteration
                else:
                    metric_col_name = next(col for col in df.columns if metric in col)
            except StopIteration:
                print(f"SKIPPING: Metric '{metric}' not found in DataFrame columns.\n")
                continue

            df[metric_col_name] = pd.to_numeric(df[metric_col_name], errors='coerce')
            df = df.dropna(subset=['eta', 'gamma', metric_col_name])
            
            if df.empty:
                print(f"SKIPPING: No valid data for '{metric}' after cleaning.\n")
                continue

            # --- Create and save the plot ---
            plot_title = visualizer._format_plot_title(metric_col_name)
            
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            figure_save_name = f"grid_landscape_{metric}.pdf"
            full_save_path = save_dir / figure_save_name

            # Get good cmap 
            # from pypalettes import load_cmap
            # cmap = load_cmap("Antique")
            if mode == "presentation":
                from vizman import viz
                viz.set_visual_style()
                default_cmaps = viz.give_colormaps()
                cmap = default_cmaps["hb_bw"] # topological_map"] # hb_bw"] 
            elif mode == "draft":
                cmap = "hot" # "terrain" # cubehelix" # hot"
                
            if mode == "draft": 
                annotate_extremes = True
            elif mode == "presentation":
                annotate_extremes = False
                
            print("Using e.g.:", df_paths[0])
            fig, ax = visualizer.plot_metric_landscape_grid(
                df, 
                title=plot_title,
                metric_name=metric_col_name,
                savepath=full_save_path,
                cmap=cmap, 
                dot_color="steelblue", 
                eta_span=eta_span,
                gamma_span=gamma_span,
                show=False,
                show_dots=False,
                annotate_extremes=annotate_extremes, 
                estimated_indiv_connectomes=df_best_gamma_and_eta_estimates if plot_indiv_connectomes else None, 
                duplicate_handling=duplicate_handling, 
                show_number_samples=show_number_samples,
                interpolation='nearest',  # Can be 'nearest', 'bilinear', 'bicubic'
            )
            
            matplotlib.pyplot.close()
            
            print(f"Successfully generated plot for {metric} at {full_save_path}\n")
            
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
            traceback.print_exc()
            print("\n")

    print("--- Visualization process completed. ---")