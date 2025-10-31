"""
Enhanced visualization module for connectome analysis with taxonomic coloring.
Generates grid-based (imshow) landscape plots with color-coded points by taxonomy.
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
from matplotlib.patches import Patch
import seaborn as sns

from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer


class TaxonomicColorMapper:
    """Helper class to manage color mapping for taxonomic categories."""
    
    def __init__(self, category_column, df):
        """
        Initialize color mapper for a taxonomic category.
        
        Args:
            category_column: Name of the column to color by (e.g., 'family', 'order')
            df: DataFrame containing the category column
        """
        self.category_column = category_column
        self.unique_categories = sorted(df[category_column].dropna().unique())
        self.n_categories = len(self.unique_categories)
        
        # Generate distinct colors
        if self.n_categories <= 10:
            self.colors = sns.color_palette("tab10", self.n_categories)
        elif self.n_categories <= 20:
            self.colors = sns.color_palette("tab20", self.n_categories)
        else:
            self.colors = sns.color_palette("husl", self.n_categories)
        
        # Create mapping
        self.color_map = dict(zip(self.unique_categories, self.colors))
    
    def get_color(self, category_value):
        """Get color for a specific category value."""
        return self.color_map.get(category_value, 'gray')
    
    def get_colors_for_df(self, df):
        """Get list of colors for all rows in dataframe."""
        return [self.get_color(val) for val in df[self.category_column]]
    
    def create_legend_elements(self):
        """Create legend elements for matplotlib."""
        return [Patch(facecolor=color, label=cat) 
                for cat, color in self.color_map.items()]


class GridVisualizer(PipelineVisualizer):
    """Extension of PipelineVisualizer for gridded data visualization using imshow."""
    
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
                                   eta_span: list = None,
                                   gamma_span: list = None,
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
        else:
            fig = ax.get_figure()
        
        # Setup colormap
        _, colormap, vmin, vmax = self._setup_colormap(metric_values, vmin, vmax, cmap)
        
        # Plot using imshow
        extent = [eta_edges[0], eta_edges[-1], gamma_edges[0], gamma_edges[-1]]
        im = ax.imshow(grid_data, origin='lower', extent=extent,
                      aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax,
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
            title = f"{title} ({len(df)} samples)"
        
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


def load_and_merge_taxonomic_data(gnm_results_df, animal_metadata_path):
    """
    Merge GNM results with taxonomic metadata.
    
    Args:
        gnm_results_df: DataFrame with gamma, eta, and metrics
        animal_metadata_path: Path to CSV with taxonomic information
    
    Returns:
        Merged DataFrame with taxonomic columns
    """
    # Load animal metadata
    metadata_df = pd.read_csv(animal_metadata_path)
    
    # Merge on animal name
    if 'animal' in gnm_results_df.columns:
        merged_df = gnm_results_df.merge(
            metadata_df,
            on='animal',
            how='left'
        )
        print(f"Merged {len(gnm_results_df)} rows with metadata. "
              f"Successful matches: {merged_df['family'].notna().sum()}")
    else:
        print("Warning: 'animal' column not found in results. Merging by index.")
        merged_df = pd.concat([gnm_results_df, metadata_df], axis=1)
    
    return merged_df


def plot_grid_with_taxonomic_colors(visualizer, df, metric_name, title, 
                                    color_mapper, label_by, 
                                    df_best_estimates,
                                    savepath, 
                                    eta_span, gamma_span,
                                    duplicate_handling,
                                    interpolation='nearest'):
    """
    Create grid-based imshow plot with taxonomic coloring overlaid.
    
    This function creates the base grid landscape and then adds colored points on top.
    """
    # First, create the base grid landscape (without colored dots)
    fig, ax = visualizer.plot_metric_landscape_grid(
        df, 
        title=title,
        metric_name=metric_name,
        savepath=None,  # Don't save yet
        dot_color="steelblue", 
        eta_span=eta_span,
        gamma_span=gamma_span,  
        show=False,
        show_dots=False,  # Don't show dots yet
        annotate_extremes=True, 
        estimated_indiv_connectomes=None,  # Add separately
        show_colorbar=True,
        duplicate_handling=duplicate_handling,
        interpolation=interpolation,
    )
    
    # Add estimated individual connectomes if provided
    if df_best_estimates is not None and not df_best_estimates.empty:
        if color_mapper.category_column in df_best_estimates.columns:
            est_colors = color_mapper.get_colors_for_df(df_best_estimates)
        else:
            est_colors = 'black'
            
        ax.scatter(
            df_best_estimates["eta"],
            df_best_estimates["gamma"], 
            facecolors=est_colors if isinstance(est_colors, list) else [est_colors]*len(df_best_estimates),
            s=10, # 20, # POINT SIZE
            edgecolors="black", 
            linewidths=0.25,
            marker='o',
            label='Estimated Optima',
            zorder=15
        )
    
    # Add labels if requested
    if label_by and label_by in df_best_estimates.columns:
        for index, row in df_best_estimates.iterrows():
            label = str(row[label_by])
            ax.annotate(
                label, 
                (row['eta'], row['gamma']), 
                fontsize=3, 
                alpha=0.8,
                xytext=(2, 2), # 5, 5), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         alpha=0.7, edgecolor='none')
            )
    
    # Add legend for taxonomic categories
    legend_elements = color_mapper.create_legend_elements()
    legend = ax.legend(
        handles=legend_elements, 
        title=color_mapper.category_column.replace('_', ' ').title(),
        loc='lower left',
        fontsize=5,
    )
    
    plt.tight_layout()
    
    # Save
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print(f"Successfully saved: {savepath}")
    
    return fig, ax


def main(COLOR_BY="order", LABEL_BY="name"): 
    
    # --- 1. Define Input and Output ---
    
    # df_paths = [
    #     "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206/all_metrics_for_75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206_updated copy.csv"
    # ]
    
    # # Path to the enriched animal metadata
    animal_metadata_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50.csv"
    
    # parent_folder = Path(df_paths[0]).parent
    # save_path = parent_folder / "figures_grid_taxonomic"
    
        
    ##### CONFIG STUFF #######################
    experiment_name = "76_90000_samples_animal_206"
    base_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/{experiment_name}")
    save_path_df = base_path / f"all_computational_metrics_for_{experiment_name}_updated_combined.csv"
    save_path_plots = base_path / f"figures_grid_taxonomic_{COLOR_BY}"
    save_path_plots.mkdir(parents=True, exist_ok=True)
    all_metrics_file = True 

    ##########################################
    if all_metrics_file: 
        df_paths = [base_path / f"all_metrics_for_{experiment_name}.csv"]
    else: 
        df_static_path = base_path / f"all_static_metrics_for_{experiment_name}.csv"
        df_static = pd.read_csv(df_static_path)
        df_dynamic_path = base_path / f"all_dynamic_metrics_for_{experiment_name}_updated.csv"
        df_dynamic = pd.read_csv(df_dynamic_path)
        df_computational_path = base_path / f"all_computational_metrics_for_{experiment_name}_updated.csv"
        df_computational = pd.read_csv(df_computational_path)
        df_combined = pd.concat([df_static, df_dynamic, df_computational], axis=1)
        df_combined.to_csv(save_path_df)
        df_paths = [save_path_df]


    # --- 2. Taxonomic Visualization Settings ---
    
    # Whether to plot individual connectomes
    plot_indiv_connectomes = True
    duplicate_handling = "mean"
    
    eta_span = [-8, 3]
    gamma_span = [-0.1, 1]
    
    # Interpolation method for imshow
    INTERPOLATION = 'nearest'  # Options: 'nearest', 'bilinear', 'bicubic'
    
    # Path to best gamma and eta estimations
    path_to_best_gamma_and_eta_estimations = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/70_mix_and_match_animal_0/min_energy_results.csv"
    
    # --- 3. Define Metrics to Plot ---
    
    lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 20, 49]
    metrics_to_plot = [
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
    ] + [f"mc_{lag}" for lag in lags_to_plot

        # "density", "avg_clustering", "avg_degree", "degree_assortativity", "modularity", 
        # "characteristic_path_length", "transitivity", "wiring_cost", "shortest_path_distance", 
        # "structural_complexity", "n_connected_components", "omega",
        # "topological_distance", 
        # "degree_gini", 
        # "spectral_radius",
        # "spectral_gap",
        # "spectral_gap_fatemeh",
        # "global_efficiency",
        # "diffusion_efficiency",
        # "propagation_efficiency",
        # "nct_control",
        # "nct_energies",
        # "metastability",
        # "synchronizability_eigenratio",
        # "algebraic_connectivity_nx",
        # "kuramoto_synchronization",
        # "community_synchronization_vulnerability",
        # "kernel_rank",
        # "kernel_rank_fatemeh",
        # "effective_dimensionality",
        # "multifunctionality",
    ]
    
    print(f"Starting Taxonomic Grid visualization process...")
    print(f"Coloring by: {COLOR_BY}")
    print(f"Labeling by: {LABEL_BY}")
    print(f"Interpolation: {INTERPOLATION}")
    print(f"Output will be saved to: {save_path_plots}\n")

    # --- 4. Load Data ---
    print("Loading and combining data...")
    try:
        if not df_paths:
            raise ValueError("Input 'df_paths' is an empty list.")
        
        if len(df_paths) > 1:
            gnm_results_df = generate_entire_df(df_paths)
        else:
            gnm_results_df = pd.read_csv(df_paths[0], index_col=False)

        if gnm_results_df.empty:
            raise ValueError("Dataframe is empty after loading.")

        # Merge with taxonomic data
        print(f"Original data shape: {gnm_results_df.shape}")
        gnm_results_df = load_and_merge_taxonomic_data(gnm_results_df, animal_metadata_path)
        print(f"After merging with taxonomy: {gnm_results_df.shape}")
        
        # Check if the COLOR_BY column exists
        if COLOR_BY not in gnm_results_df.columns:
            print(f"Warning: '{COLOR_BY}' column not found.")
            print(f"Available columns: {gnm_results_df.columns.tolist()}")
            print("Proceeding without taxonomic coloring.")
            COLOR_BY = None

        # Load best estimates
        df_best_gamma_and_eta_estimates = None
        if os.path.exists(path_to_best_gamma_and_eta_estimations):
            df_best_gamma_and_eta_estimates = pd.read_csv(path_to_best_gamma_and_eta_estimations)
            # Also merge taxonomic data for the best estimates
            df_best_gamma_and_eta_estimates = load_and_merge_taxonomic_data(
                df_best_gamma_and_eta_estimates, 
                animal_metadata_path
            )
            print(f"Loaded {len(df_best_gamma_and_eta_estimates)} best estimates")
                
    except Exception as e:
        print(f"ERROR: Failed to load data. {e}")
        traceback.print_exc()
        exit()

    visualizer = GridVisualizer()
    
    # --- 5. Generate Individual Plots in a Loop ---
    for metric in metrics_to_plot:
        print(f"\n--- Generating Grid plot for metric: {metric} ---")
        try:
            df = gnm_results_df.copy()

            # Custom metric calculations
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
            
            if metric == "avg_clustering_divided_by_global_efficiency": 
                upper_factor = pd.to_numeric(df["avg_clustering"], errors='coerce')
                lower_factor = pd.to_numeric(df["global_efficiency"], errors='coerce')
                total = upper_factor * lower_factor * (-1)
                total = (total - total.min()) / (total.max() - total.min())
                df["avg_clustering_divided_by_global_efficiency"] = total
            
            df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
            df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
            
            # Find metric column
            try:
                if metric.startswith("mc_") and metric.split("_")[1].isdigit():
                    metric_col_name = metric
                    if metric not in df.columns:
                        raise StopIteration
                else:
                    metric_col_name = next(col for col in df.columns if metric in col)
            except StopIteration:
                print(f"SKIPPING: Metric '{metric}' not found in DataFrame columns.")
                continue

            df[metric_col_name] = pd.to_numeric(df[metric_col_name], errors='coerce')
            df = df.dropna(subset=['eta', 'gamma', metric_col_name])
            
            if df.empty:
                print(f"SKIPPING: No valid data for '{metric}' after cleaning.")
                continue

            # --- Setup taxonomic coloring ---
            if COLOR_BY and COLOR_BY in df_best_gamma_and_eta_estimates.columns:
                df_best_gamma_and_eta_estimates_clean = df_best_gamma_and_eta_estimates.dropna(subset=[COLOR_BY])
                if df_best_gamma_and_eta_estimates_clean.empty:
                    print(f"SKIPPING: No data after removing missing {COLOR_BY} values.")
                    continue
                    
                color_mapper = TaxonomicColorMapper(COLOR_BY, df_best_gamma_and_eta_estimates_clean)
                print(f"Found {color_mapper.n_categories} unique {COLOR_BY} categories")
            else:
                print("SKIPPING: Cannot color by taxonomy - column not found")
                continue

            # --- Create and save the plot ---
            plot_title = visualizer._format_plot_title(metric_col_name)
            
            figure_save_name = f"grid_landscape_{metric}_by_{COLOR_BY}.pdf"
            full_save_path = save_path_plots / figure_save_name

            fig, ax = plot_grid_with_taxonomic_colors(
                visualizer=visualizer,
                df=df,
                metric_name=metric_col_name,
                title=plot_title,
                color_mapper=color_mapper,
                label_by=LABEL_BY,
                df_best_estimates=df_best_gamma_and_eta_estimates_clean if plot_indiv_connectomes else None,
                savepath=full_save_path,
                eta_span=eta_span,
                gamma_span=gamma_span,
                duplicate_handling=duplicate_handling,
                interpolation=INTERPOLATION,
            )
            
            matplotlib.pyplot.close(fig)
            
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
            traceback.print_exc()

    print("\n--- Taxonomic Grid visualization process completed. ---")
    
    
if __name__ == "__main__": 
    
    # ,animal,common_name,name,species,genus,sub_family,
    # family,sub_order,order,super_order,phylogenetic_group
    # Choose which taxonomic level to color by

    main(COLOR_BY="phylogenetic_group", LABEL_BY="name")
    main(COLOR_BY="super_order", LABEL_BY="name")
    main(COLOR_BY="order", LABEL_BY="name")
    main(COLOR_BY="sub_order", LABEL_BY="name")
    main(COLOR_BY="family", LABEL_BY="name")
    
    