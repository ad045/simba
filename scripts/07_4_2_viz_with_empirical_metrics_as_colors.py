"""
Enhanced visualization for connectome analysis with metric-based coloring.
Generates grid-based (imshow) landscape plots with points colored by their metric values.
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


def load_empirical_metrics(empirical_csv_path):
    """
    Load empirical metrics from CSV file.
    
    Args:
        empirical_csv_path: Path to CSV with empirical metric values
    
    Returns:
        DataFrame with empirical metrics
    """
    df = pd.read_csv(empirical_csv_path)
    print(f"Loaded {len(df)} empirical samples with {len(df.columns)} metrics")
    return df


def merge_with_best_estimates(df_empirical, df_best_estimates):
    """
    Merge empirical metrics with best gamma/eta estimates.
    
    Args:
        df_empirical: DataFrame with empirical metrics (indexed by animal ID)
        df_best_estimates: DataFrame with eta, gamma, and animal identifiers
    
    Returns:
        Merged DataFrame with eta, gamma, and metric values
    """
    # Merge the dataframes
    if 'animal' in df_best_estimates.columns and 'animal' in df_empirical.columns:
        merged = df_best_estimates.merge(df_empirical, on='animal', how='left')
    elif 'id' in df_empirical.columns:
        # Assume best_estimates rows correspond to empirical data rows
        merged = pd.concat([df_best_estimates.reset_index(drop=True), 
                           df_empirical.reset_index(drop=True)], axis=1)
    else:
        # Merge by index
        merged = pd.concat([df_best_estimates, df_empirical], axis=1)
    
    print(f"Merged data shape: {merged.shape}")
    return merged


def plot_grid_with_metric_colored_points(visualizer, df_landscape, metric_name, 
                                        title, df_points, cmap,
                                        savepath, eta_span, gamma_span,
                                        duplicate_handling, interpolation='nearest',
                                        point_size=50, label_by=None,
                                        point_edgecolor='black', point_linewidth=0.5):
    """
    Create grid-based imshow plot with points colored by their metric values.
    
    Args:
        visualizer: GridVisualizer instance
        df_landscape: DataFrame for the landscape (GNM results)
        metric_name: Name of the metric to plot
        title: Plot title
        df_points: DataFrame with eta, gamma, and metric values for points
        cmap: Colormap name
        savepath: Path to save the figure
        eta_span: [min, max] for eta axis
        gamma_span: [min, max] for gamma axis
        duplicate_handling: How to handle duplicate points
        interpolation: Interpolation method for imshow
        point_size: Size of scatter points
        label_by: Column name to use for labeling points (optional)
        point_edgecolor: Edge color for scatter points
        point_linewidth: Edge line width for scatter points
    """
    # First, create the base grid landscape
    fig, ax = visualizer.plot_metric_landscape_grid(
        df_landscape, 
        title=title,
        metric_name=metric_name,
        savepath=None,  # Don't save yet
        cmap=cmap,
        eta_span=eta_span,
        gamma_span=gamma_span,  
        show=False,
        show_dots=False,  # Don't show default dots
        annotate_extremes=True, 
        estimated_indiv_connectomes=None,  # Add separately with colors
        show_colorbar=True,
        duplicate_handling=duplicate_handling,
        interpolation=interpolation,
    )
    
    # Get the colormap and normalization from the landscape
    if metric_name in df_points.columns:
        metric_values = df_points[metric_name].values
        
        # Use the same vmin/vmax as the landscape for consistency
        vmin = df_landscape[metric_name].min()
        vmax = df_landscape[metric_name].max()
        
        # Create normalization
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap(cmap)
        
        # Map metric values to colors
        point_colors = colormap(norm(metric_values))
        
        # Add scatter points with metric-based colors
        scatter = ax.scatter(
            df_points["eta"],
            df_points["gamma"], 
            c=metric_values,
            cmap=cmap,
            norm=norm,
            s=point_size,
            edgecolors=point_edgecolor, 
            linewidths=point_linewidth,
            marker='o',
            label='Empirical Connectomes',
            zorder=15
        )
        
        # Add labels if requested
        if label_by and label_by in df_points.columns:
            for idx, row in df_points.iterrows():
                if pd.notna(row['eta']) and pd.notna(row['gamma']):
                    label = str(row[label_by])
                    ax.annotate(
                        label, 
                        (row['eta'], row['gamma']), 
                        fontsize=3, 
                        alpha=0.8,
                        xytext=(2, 2), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 alpha=0.7, edgecolor='none')
                    )
    else:
        print(f"Warning: Metric '{metric_name}' not found in points DataFrame")
    
    plt.tight_layout()
    
    # Save
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)
        print(f"Successfully saved: {savepath}")
    
    return fig, ax


def main(LABEL_BY="name"):
    
    # --- 1. Define Input and Output ---
    ####################################################################
    # Path to empirical metrics. THIS TIME THE ORIIGNAL MATRIX!
    empirical_metrics_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/emprirical_analysis/empirical_analysis_binarized.csv"
    # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/empirical_all_metrics_analysis/all_metrics_empirical_all_metrics_analysis.csv" 
    # Path to best gamma and eta estimations
    path_to_best_gamma_and_eta_estimations = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/70_mix_and_match_animal_0/min_energy_results.csv"
    
    ##### CONFIG STUFF #######################
    experiment_name = "76_90000_samples_animal_206"
    base_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/{experiment_name}")
    save_path_df = base_path / f"all_computational_metrics_for_{experiment_name}_updated_combined.csv"
    save_path_plots = base_path / "figures_grid_metric_colored"
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

    # --- 2. Visualization Settings ---
    
    duplicate_handling = "mean"
    
    eta_span = [-8, 3]
    gamma_span = [-0.1, 1]
    
    # Interpolation method for imshow
    INTERPOLATION = 'nearest'  # Options: 'nearest', 'bilinear', 'bicubic'
    
    # Point styling
    POINT_SIZE = 50
    POINT_EDGE_COLOR = 'black'
    POINT_LINEWIDTH = 0.5
    
    # Colormap - use the same as the landscape
    CMAP = "hot"
    
    # --- 3. Define Metrics to Plot ---
    
    lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 20, 49]
    metrics_to_plot = [
        "MaxCriteria",
        "avg_communicability", 
        "global_efficiency", 
        "modularity", 
        "avg_clustering", 
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
    ] + [f"mc_{lag}" for lag in lags_to_plot]
    
    print(f"Starting Metric-Colored Grid visualization process...")
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

        print(f"Loaded GNM results shape: {gnm_results_df.shape}")
        
        # Load empirical metrics
        df_empirical = load_empirical_metrics(empirical_metrics_path)
        
        # Load best estimates
        df_best_estimates = pd.read_csv(path_to_best_gamma_and_eta_estimations)
        print(f"Loaded {len(df_best_estimates)} best estimates")
        
        # Merge empirical metrics with best estimates
        df_points = merge_with_best_estimates(df_empirical, df_best_estimates)
                
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

            # Custom metric calculations for landscape
            if metric == "mean_mc_divided_by_wiring_cost": 
                df["mean_mc_divided_by_wiring_cost"] = (
                    pd.to_numeric(df["mc_mean"], errors='coerce') / 
                    pd.to_numeric(df["wiring_cost"], errors='coerce')
                )
                # Also calculate for points
                if "mc_mean" in df_points.columns and "wiring_cost" in df_points.columns:
                    df_points["mean_mc_divided_by_wiring_cost"] = (
                        pd.to_numeric(df_points["mc_mean"], errors='coerce') / 
                        pd.to_numeric(df_points["wiring_cost"], errors='coerce')
                    )
            
            if metric == "mc_5_divided_by_wiring_cost": 
                df["mc_5_divided_by_wiring_cost"] = (
                    pd.to_numeric(df["mc_5"], errors='coerce') / 
                    pd.to_numeric(df["wiring_cost"], errors='coerce')
                )
                # Also calculate for points
                if "mc_5" in df_points.columns and "wiring_cost" in df_points.columns:
                    df_points["mc_5_divided_by_wiring_cost"] = (
                        pd.to_numeric(df_points["mc_5"], errors='coerce') / 
                        pd.to_numeric(df_points["wiring_cost"], errors='coerce')
                    )
            
            df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
            df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
            
            # Find metric column in landscape data
            try:
                if metric.startswith("mc_") and metric.split("_")[1].isdigit():
                    metric_col_name = metric
                    if metric not in df.columns:
                        raise StopIteration
                else:
                    metric_col_name = next(col for col in df.columns if metric in col)
            except StopIteration:
                print(f"SKIPPING: Metric '{metric}' not found in landscape DataFrame columns.")
                continue

            # Check if metric exists in points data
            if metric_col_name not in df_points.columns:
                print(f"SKIPPING: Metric '{metric_col_name}' not found in empirical data.")
                continue

            df[metric_col_name] = pd.to_numeric(df[metric_col_name], errors='coerce')
            df = df.dropna(subset=['eta', 'gamma', metric_col_name])
            
            if df.empty:
                print(f"SKIPPING: No valid landscape data for '{metric}' after cleaning.")
                continue
            
            # Clean points data
            df_points_clean = df_points.copy()
            df_points_clean["eta"] = pd.to_numeric(df_points_clean["eta"], errors='coerce')
            df_points_clean["gamma"] = pd.to_numeric(df_points_clean["gamma"], errors='coerce')
            df_points_clean[metric_col_name] = pd.to_numeric(df_points_clean[metric_col_name], errors='coerce')
            df_points_clean = df_points_clean.dropna(subset=['eta', 'gamma', metric_col_name])
            
            if df_points_clean.empty:
                print(f"SKIPPING: No valid point data for '{metric}' after cleaning.")
                continue

            # --- Create and save the plot ---
            plot_title = visualizer._format_plot_title(metric_col_name)
            
            figure_save_name = f"grid_landscape_{metric}_metric_colored.pdf"
            full_save_path = save_path_plots / figure_save_name

            fig, ax = plot_grid_with_metric_colored_points(
                visualizer=visualizer,
                df_landscape=df,
                metric_name=metric_col_name,
                title=plot_title,
                df_points=df_points_clean,
                cmap=CMAP,
                savepath=full_save_path,
                eta_span=eta_span,
                gamma_span=gamma_span,
                duplicate_handling=duplicate_handling,
                interpolation=INTERPOLATION,
                point_size=POINT_SIZE,
                label_by=LABEL_BY,
                point_edgecolor=POINT_EDGE_COLOR,
                point_linewidth=POINT_LINEWIDTH,
            )
            
            matplotlib.pyplot.close(fig)
            
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
            traceback.print_exc()

    print("\n--- Metric-Colored Grid visualization process completed. ---")
    
    
if __name__ == "__main__": 
    main(LABEL_BY="name")