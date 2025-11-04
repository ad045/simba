"""
Enhanced visualization for connectome analysis with averaged energy from closest neighbors.
"""

import os
import re
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
        """Plot a metric landscape using imshow for gridded data."""
        
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
        """Create a regular grid from scattered points."""
        # Get unique sorted values for each dimension
        unique_eta = np.unique(points[:, 0])
        unique_gamma = np.unique(points[:, 1])
        
        if grid_resolution is None:
            eta_edges = unique_eta
            gamma_edges = unique_gamma
        else:
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


def find_k_closest_connectomes(target_eta: float, 
                                target_gamma: float, 
                                df_landscape: pd.DataFrame,
                                k: int = 20) -> pd.DataFrame:
    """
    Find the k closest connectomes in the landscape to the target (eta, gamma).
    
    Args:
        target_eta: Target eta value
        target_gamma: Target gamma value
        df_landscape: DataFrame with 'eta', 'gamma', and energy/metric columns
        k: Number of closest neighbors to find
    
    Returns:
        DataFrame containing the k closest rows, sorted by distance
    """
    # Calculate Euclidean distance to target
    df_landscape = df_landscape.copy()
    df_landscape['distance'] = np.sqrt(
        (df_landscape['eta'] - target_eta)**2 + 
        (df_landscape['gamma'] - target_gamma)**2
    )
    
    # Sort by distance and take top k
    closest = df_landscape.nsmallest(k, 'distance')
    
    return closest


def compute_averaged_energies(df_best_estimates: pd.DataFrame,
                               df_individual_energies: pd.DataFrame,
                               k: int = 20) -> pd.DataFrame:
    """
    For each empirical connectome's best (eta, gamma), find the k closest 
    generated connectomes and average their energies.
    
    Args:
        df_best_estimates: DataFrame with columns ['eta', 'gamma', 'animal'/ID]
        df_individual_energies: DataFrame with ['eta', 'gamma', 'energy'] for all generated connectomes
        k: Number of neighbors to average over
    
    Returns:
        DataFrame with ['eta', 'gamma', 'avg_energy_k_neighbors', 'animal'/ID]
    """
    results = []
    
    for idx, row in df_best_estimates.iterrows():
        target_eta = row['eta']
        target_gamma = row['gamma']
        
        # Find k closest connectomes
        closest = find_k_closest_connectomes(
            target_eta, 
            target_gamma, 
            df_individual_energies,
            k=k
        )
        
        # Average their energies. ˝ TODO 
        avg_energy = closest['energy'].mean()
        
        result_row = {
            'eta': target_eta,
            'gamma': target_gamma,
            'avg_energy_k_neighbors': avg_energy,
        }
        
        # Preserve any ID columns
        for col in ['animal', 'id', 'subj_index', 'name']:
            if col in row:
                result_row[col] = row[col]
        
        results.append(result_row)
    
    return pd.DataFrame(results)


def plot_grid_with_averaged_energy_points(visualizer, df_landscape, metric_name, 
                                          title, df_points, cmap,
                                          savepath, eta_span, gamma_span,
                                          duplicate_handling, interpolation='nearest',
                                          point_size=50, label_by=None,
                                          point_edgecolor='black', point_linewidth=0.5):
    """
    Create grid-based imshow plot with points colored by averaged energy from k neighbors.
    """
    # Create the base grid landscape
    fig, ax = visualizer.plot_metric_landscape_grid(
        df_landscape, 
        title=title,
        metric_name=metric_name,
        savepath=None,
        cmap=cmap,
        eta_span=eta_span,
        gamma_span=gamma_span,  
        show=False,
        show_dots=False,
        annotate_extremes=True, 
        estimated_indiv_connectomes=None,
        show_colorbar=True,
        duplicate_handling=duplicate_handling,
        interpolation=interpolation,
    )
    
    # Get the colormap and normalization from the landscape
    if 'avg_energy_k_neighbors' in df_points.columns:
        metric_values = df_points['avg_energy_k_neighbors'].values
        
        # Use the same vmin/vmax as the landscape for consistency
        vmin = df_landscape[metric_name].min()
        vmax = df_landscape[metric_name].max()
        
        # Create normalization
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap(cmap)
        
        # Add scatter points with averaged energy colors
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
            label='Empirical Connectomes (20-neighbor avg)',
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
        print("Warning: 'avg_energy_k_neighbors' not found in points DataFrame")
    
    plt.tight_layout()
    
    # Save
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)
        print(f"Successfully saved: {savepath}")
    
    return fig, ax


def main(LABEL_BY="name", K_NEIGHBORS=20):
    
    # --- 1. Define Input and Output ---
    experiment_name = "75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206"
    base_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/{experiment_name}")
    
    # Path to individual energies for all generated connectomes
    individual_energies_path = base_path / f"summary_indiv_energies_for_exp_{experiment_name}.csv"
    
    # Path to best gamma and eta estimations
    path_to_best_gamma_and_eta_estimations = base_path.parent / "70_mix_and_match_animal_0" / "min_energy_results.csv"
    
    # Output paths
    save_path_plots = base_path / f"figures_grid_averaged_energy_k{K_NEIGHBORS}"
    save_path_plots.mkdir(parents=True, exist_ok=True)
    
    # Visualization settings
    duplicate_handling = "mean"
    eta_span = [-8, 3]
    gamma_span = [-0.1, 1]
    INTERPOLATION = 'nearest'
    POINT_SIZE = 50
    POINT_EDGE_COLOR = 'black'
    POINT_LINEWIDTH = 0.5
    CMAP = "hot"
    
    print(f"Starting Averaged Energy (k={K_NEIGHBORS}) Grid visualization...")
    print(f"Labeling by: {LABEL_BY}")
    print(f"Output: {save_path_plots}\n")

    # --- 2. Load Data ---
    print("Loading data...")
    try:
        # Load individual energies (all generated connectomes)
        df_individual_energies = pd.read_csv(individual_energies_path)
        print(f"Loaded {len(df_individual_energies)} individual energy measurements")
        
        # Load best estimates
        df_best_estimates = pd.read_csv(path_to_best_gamma_and_eta_estimations)
        print(f"Loaded {len(df_best_estimates)} best estimates")
        
        # Compute averaged energies for each empirical connectome
        print(f"\nComputing {K_NEIGHBORS}-neighbor averaged energies...")
        df_averaged_energies = compute_averaged_energies(
            df_best_estimates,
            df_individual_energies,
            k=K_NEIGHBORS
        )
        print(f"Computed averaged energies for {len(df_averaged_energies)} points")
        
        # Save the averaged energies
        output_csv = save_path_plots / f"averaged_energies_k{K_NEIGHBORS}.csv"
        df_averaged_energies.to_csv(output_csv, index=False)
        print(f"Saved averaged energies to: {output_csv}")
        
    except Exception as e:
        print(f"ERROR: Failed to load data. {e}")
        traceback.print_exc()
        exit()

    # --- 3. Create Visualization ---
    visualizer = GridVisualizer()
    
    print("\n--- Generating Grid plot with averaged energies ---")
    try:
        # Clean the data
        df_landscape = df_individual_energies.copy()
        df_landscape["eta"] = pd.to_numeric(df_landscape["eta"], errors='coerce')
        df_landscape["gamma"] = pd.to_numeric(df_landscape["gamma"], errors='coerce')
        df_landscape["energy"] = pd.to_numeric(df_landscape["energy"], errors='coerce')
        df_landscape = df_landscape.dropna(subset=['eta', 'gamma', 'energy'])
        
        df_points = df_averaged_energies.copy()
        df_points["eta"] = pd.to_numeric(df_points["eta"], errors='coerce')
        df_points["gamma"] = pd.to_numeric(df_points["gamma"], errors='coerce')
        df_points["avg_energy_k_neighbors"] = pd.to_numeric(
            df_points["avg_energy_k_neighbors"], errors='coerce'
        )
        df_points = df_points.dropna(subset=['eta', 'gamma', 'avg_energy_k_neighbors'])
        
        if df_landscape.empty or df_points.empty:
            print("ERROR: No valid data after cleaning.")
            exit()
        
        # Create plot
        plot_title = f"Energy Landscape with {K_NEIGHBORS}-Neighbor Averaged Energy"
        figure_save_name = f"grid_landscape_energy_avg_k{K_NEIGHBORS}.pdf"
        full_save_path = save_path_plots / figure_save_name

        fig, ax = plot_grid_with_averaged_energy_points(
            visualizer=visualizer,
            df_landscape=df_landscape,
            metric_name="energy",
            title=plot_title,
            df_points=df_points,
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
        print(f"ERROR: Failed to create visualization. {e}")
        traceback.print_exc()

    print("\n--- Averaged Energy Grid visualization completed. ---")
    
    
if __name__ == "__main__": 
    main(LABEL_BY="name", K_NEIGHBORS=20)