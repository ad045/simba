"""
Visualization module for connectome analysis pipeline.
Integrates energy landscape plotting for both GNM and ESN results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json


class PipelineVisualizer:
    """Visualization tools for connectome analysis pipeline results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_energy_landscape_voronoi(self, df, dot_color="white", title="", 
                                      cmap="hot", savepath=None, show=True,
                                      show_points=True, point_size=8,
                                      vmin=None, vmax=None):
        """
        Plot energy landscape using Voronoi diagram for randomly sampled points.
        
        Args:
            - df (pd.DataFrame): DataFrame containing 'eta', 'gamma', and 'energy' columns.
            - dot_color (str): Color for the sample points.
            - title (str): Title for the plot.
            - cmap (str): Colormap for the energy values.
            - savepath (str): Path to save the figure.
            - show (bool): Whether to display the plot.
            - show_points (bool): Whether to show the actual sample points.
            - point_size (int): Size of the sample points if shown.
            - vmin (float): Minimum value for color scale.
            - vmax (float): Maximum value for color scale.
            
        Returns:
            - fig, ax: Matplotlib figure and axis objects.
        """
        
        # Ensure df has the required columns
        if not all(col in df.columns for col in ["eta", "gamma", "energy"]):
            raise ValueError("DataFrame must contain 'eta', 'gamma', and 'energy' columns.")
        
        # For each unique (eta, gamma) pair, average the energy
        grid_df = df.groupby(["eta", "gamma"], as_index=False)["energy"].mean()
        
        # Extract points and values
        points = grid_df[["eta", "gamma"]].values
        energies = grid_df["energy"].values
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        
        # Determine plot bounds with some padding
        eta_min, eta_max = points[:, 0].min(), points[:, 0].max()
        gamma_min, gamma_max = points[:, 1].min(), points[:, 1].max()
        eta_range = eta_max - eta_min
        gamma_range = gamma_max - gamma_min
        padding = 0.1
        
        xlim = [eta_min - padding * eta_range, eta_max + padding * eta_range]
        ylim = [gamma_min - padding * gamma_range, gamma_max + padding * gamma_range]
        
        # Add boundary points to ensure complete Voronoi cells
        boundary_points = []
        n_boundary = 20
        
        # Add points along the boundaries
        for i in range(n_boundary):
            # Bottom and top
            boundary_points.append([xlim[0] + i * (xlim[1] - xlim[0]) / (n_boundary - 1), ylim[0] - 1])
            boundary_points.append([xlim[0] + i * (xlim[1] - xlim[0]) / (n_boundary - 1), ylim[1] + 1])
            # Left and right
            boundary_points.append([xlim[0] - 1, ylim[0] + i * (ylim[1] - ylim[0]) / (n_boundary - 1)])
            boundary_points.append([xlim[1] + 1, ylim[0] + i * (ylim[1] - ylim[0]) / (n_boundary - 1)])
        
        # Combine actual points with boundary points
        all_points = np.vstack([points, boundary_points])
        
        # Create Voronoi diagram
        vor = Voronoi(all_points)
        
        # Set up color normalization
        if vmin is None:
            vmin = energies.min()
        if vmax is None:
            vmax = energies.max()
        
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap(cmap)
        
        # Color each Voronoi region
        for i, point_idx in enumerate(range(len(points))):  # Only color regions for actual data points
            region_idx = vor.point_region[point_idx]
            region = vor.regions[region_idx]
            
            if -1 not in region and len(region) > 0:
                polygon_vertices = [vor.vertices[v] for v in region]
                polygon = patches.Polygon(polygon_vertices, 
                                         facecolor=colormap(norm(energies[i])),
                                         edgecolor='none')
                ax.add_patch(polygon)
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Energy", rotation=270, labelpad=12)
        
        # Optionally show the actual sample points
        if show_points:
            ax.scatter(points[:, 0], points[:, 1], 
                      s=point_size, c=dot_color, 
                      edgecolors='black', linewidths=0.5,
                      alpha=0.8, zorder=5)
        
        # If best points per subject are needed
        if "subject" in df.columns:
            df_best = df.loc[df.groupby("subject")["energy"].idxmin()].reset_index(drop=True)
            ax.scatter(df_best["eta"].values, df_best["gamma"].values,
                      s=16, c="cyan", edgecolors="black", linewidths=0.5,
                      alpha=0.95, zorder=10, marker='*')
        
        # Set limits and labels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("η")
        ax.set_ylabel("γ")
        ax.set_title(title)
        ax.set_aspect('auto')
        
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
        
        if show:
            plt.show()
        
        return fig, ax
    
    def plot_energy_landscape_interpolated(self, df, method='cubic', resolution=100,
                                           dot_color="white", title="", 
                                           cmap="hot", savepath=None, show=True,
                                           vmin=None, vmax=None):
        """
        Plot energy landscape using interpolation for randomly sampled points.
        
        Args:
            - df (pd.DataFrame): DataFrame containing 'eta', 'gamma', and 'energy' columns.
            - method (str): Interpolation method ('linear', 'cubic', 'nearest').
            - resolution (int): Grid resolution for interpolation.
            - dot_color (str): Color for the best-fit points.
            - title (str): Title for the plot.
            - cmap (str): Colormap for the energy values.
            - savepath (str): Path to save the figure.
            - show (bool): Whether to display the plot.
            - vmin (float): Minimum value for color scale.
            - vmax (float): Maximum value for color scale.
            
        Returns:
            - fig, ax: Matplotlib figure and axis objects.
        """
        from scipy.interpolate import griddata
        
        # Ensure df has the required columns
        if not all(col in df.columns for col in ["eta", "gamma", "energy"]):
            raise ValueError("DataFrame must contain 'eta', 'gamma', and 'energy' columns.")
        
        # For each unique (eta, gamma) pair, average the energy
        grid_df = df.groupby(["eta", "gamma"], as_index=False)["energy"].mean()
        
        # Create regular grid for interpolation
        eta_min, eta_max = grid_df["eta"].min(), grid_df["eta"].max()
        gamma_min, gamma_max = grid_df["gamma"].min(), grid_df["gamma"].max()
        
        eta_grid = np.linspace(eta_min, eta_max, resolution)
        gamma_grid = np.linspace(gamma_min, gamma_max, resolution)
        eta_mesh, gamma_mesh = np.meshgrid(eta_grid, gamma_grid)
        
        # Interpolate
        points = grid_df[["eta", "gamma"]].values
        values = grid_df["energy"].values
        
        energy_interp = griddata(points, values, (eta_mesh, gamma_mesh), method=method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        
        # Plot interpolated surface
        im = ax.imshow(energy_interp,
                       origin="lower",
                       extent=[eta_min, eta_max, gamma_min, gamma_max],
                       aspect="auto",
                       cmap=cmap,
                       vmin=vmin if vmin is not None else values.min(),
                       vmax=vmax if vmax is not None else values.max())
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Energy", rotation=270, labelpad=12)
        
        # Show sample points
        ax.scatter(points[:, 0], points[:, 1], 
                  s=8, c=dot_color, edgecolors='black', 
                  linewidths=0.5, alpha=0.7, zorder=5)
        
        # If best points per subject are needed
        if "subject" in df.columns:
            df_best = df.loc[df.groupby("subject")["energy"].idxmin()].reset_index(drop=True)
            ax.scatter(df_best["eta"].values, df_best["gamma"].values,
                      s=16, c="cyan", edgecolors="black", linewidths=0.5,
                      alpha=0.95, zorder=10, marker='*')
        
        ax.set_xlabel("η")
        ax.set_ylabel("γ")
        ax.set_title(title)
        
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
        
        if show:
            plt.show()
        
        return fig, ax
    
    def plot_gnm_energy_landscape(self, gnm_results_file: Path, 
                                  visualization_type: str = "voronoi",
                                  density: Optional[int] = None,
                                  savepath: Optional[Path] = None,
                                  show: bool = True) -> tuple:
        """
        Plot GNM energy landscape from pipeline results.
        
        Args:
            gnm_results_file: Path to GNM results JSON file
            visualization_type: "voronoi" or "interpolated"
            density: Specific density to plot (if multiple)
            savepath: Path to save figure
            show: Whether to display plot
            
        Returns:
            fig, ax objects
        """
        # Load GNM results
        with open(gnm_results_file, 'r') as f:
            results = json.load(f)
        
        # Extract energy data
        energy_data = []
        
        for network_result in results.get("network_results", []):
            if "parameter_fitting" in network_result:
                params = network_result["parameter_fitting"]
                energy_data.append({
                    "eta": params.get("best_eta", 0),
                    "gamma": params.get("best_gamma", 0),
                    "energy": params.get("best_energy", 0),
                    "subject": network_result.get("network_index", 0)
                })
        
        if not energy_data:
            raise ValueError("No energy data found in results")
        
        # Create DataFrame
        df = pd.DataFrame(energy_data)
        
        # Add density information if available
        if density is not None:
            title = f"GNM Energy Landscape (Density {density}%)"
        else:
            title = "GNM Energy Landscape"
        
        # Plot based on visualization type
        if visualization_type == "voronoi":
            fig, ax = self.plot_energy_landscape_voronoi(
                df, title=title, savepath=savepath, show=show,
                cmap="viridis", show_points=True
            )
        else:
            fig, ax = self.plot_energy_landscape_interpolated(
                df, method='cubic', title=title, 
                savepath=savepath, show=show, cmap="viridis"
            )
        
        return fig, ax
    
    def plot_esn_memory_capacity_landscape(self, esn_results_file: Path,
                                           param1: str = "spectral_radius",
                                           param2: str = "input_length",
                                           visualization_type: str = "voronoi",
                                           density: Optional[int] = None,
                                           savepath: Optional[Path] = None,
                                           show: bool = True) -> tuple:
        """
        Plot ESN memory capacity landscape from pipeline results.
        
        Args:
            esn_results_file: Path to ESN results CSV file
            param1: First parameter for x-axis
            param2: Second parameter for y-axis
            visualization_type: "voronoi" or "interpolated"
            density: Specific density to plot
            savepath: Path to save figure
            show: Whether to display plot
            
        Returns:
            fig, ax objects
        """
        # Load ESN results
        df_results = pd.read_csv(esn_results_file)
        
        # Filter out non-data rows
        df_results = df_results[df_results['subject'] != 'COMPLETED']
        
        # Convert to numeric
        numeric_cols = ['subject', 'density_percent', 'spectral_radius', 
                       'input_length', 'input_scaling', 'mc_mean']
        for col in numeric_cols:
            if col in df_results.columns:
                df_results[col] = pd.to_numeric(df_results[col], errors='coerce')
        
        # Filter by density if specified
        if density is not None:
            df_results = df_results[df_results['density_percent'] == density]
        
        # Create energy landscape data
        # Note: We use negative memory capacity as "energy" (lower is better)
        df_landscape = pd.DataFrame({
            "eta": df_results[param1],
            "gamma": df_results[param2],
            "energy": -df_results['mc_mean'],  # Negative so higher MC appears as lower energy
            "subject": df_results['subject']
        })
        
        # Drop NaN values
        df_landscape = df_landscape.dropna()
        
        if density is not None:
            title = f"ESN Memory Capacity Landscape (Density {density}%)\n{param1} vs {param2}"
        else:
            title = f"ESN Memory Capacity Landscape\n{param1} vs {param2}"
        
        # Plot based on visualization type
        if visualization_type == "voronoi":
            fig, ax = self.plot_energy_landscape_voronoi(
                df_landscape, title=title, savepath=savepath, show=show,
                cmap="coolwarm", show_points=True
            )
        else:
            fig, ax = self.plot_energy_landscape_interpolated(
                df_landscape, method='cubic', title=title,
                savepath=savepath, show=show, cmap="coolwarm"
            )
        
        # Adjust colorbar label for memory capacity
        if fig.axes:
            for ax_item in fig.axes:
                if hasattr(ax_item, 'get_ylabel'):
                    if ax_item.get_ylabel() == "Energy":
                        ax_item.set_ylabel("Memory Capacity", rotation=270, labelpad=12)
        
        return fig, ax
    
    def compare_visualizations(self, df: pd.DataFrame, 
                              title_prefix: str = "Energy Landscape",
                              savepath: Optional[Path] = None) -> plt.Figure:
        """
        Create a comparison of different visualization methods.
        
        Args:
            df: DataFrame with 'eta', 'gamma', 'energy' columns
            title_prefix: Prefix for titles
            savepath: Path to save figure
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Get data range for consistent coloring
        grid_df = df.groupby(["eta", "gamma"], as_index=False)["energy"].mean()
        vmin, vmax = grid_df["energy"].min(), grid_df["energy"].max()
        
        # 1. Voronoi diagram
        plt.sca(axes[0])
        self.plot_energy_landscape_voronoi(
            df, title=f"{title_prefix} - Voronoi", 
            show=False, vmin=vmin, vmax=vmax
        )
        
        # 2. Interpolated (cubic)
        plt.sca(axes[1])
        self.plot_energy_landscape_interpolated(
            df, method='cubic', 
            title=f"{title_prefix} - Cubic Interpolation",
            show=False, vmin=vmin, vmax=vmax
        )
        
        # 3. Interpolated (linear)
        plt.sca(axes[2])
        self.plot_energy_landscape_interpolated(
            df, method='linear',
            title=f"{title_prefix} - Linear Interpolation",
            show=False, vmin=vmin, vmax=vmax
        )
        
        plt.tight_layout()
        
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
        
        plt.show()
        
        return fig
    
    def create_summary_plot(self, esn_results_file: Optional[Path] = None,
                           gnm_results_file: Optional[Path] = None,
                           savepath: Optional[Path] = None) -> plt.Figure:
        """
        Create a summary plot combining ESN and GNM results.
        
        Args:
            esn_results_file: Path to ESN results
            gnm_results_file: Path to GNM results
            savepath: Path to save figure
            
        Returns:
            Figure object
        """
        n_plots = sum([esn_results_file is not None, gnm_results_file is not None])
        
        if n_plots == 0:
            raise ValueError("At least one results file must be provided")
        
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot ESN results if available
        if esn_results_file and esn_results_file.exists():
            plt.sca(axes[plot_idx])
            self.plot_esn_memory_capacity_landscape(
                esn_results_file,
                visualization_type="voronoi",
                show=False
            )
            axes[plot_idx].set_title("ESN Memory Capacity")
            plot_idx += 1
        
        # Plot GNM results if available
        if gnm_results_file and gnm_results_file.exists():
            plt.sca(axes[plot_idx])
            self.plot_gnm_energy_landscape(
                gnm_results_file,
                visualization_type="voronoi",
                show=False
            )
            axes[plot_idx].set_title("GNM Energy Landscape")
        
        plt.suptitle("Pipeline Results Summary", fontsize=14, y=1.02)
        plt.tight_layout()
        
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
        
        plt.show()
        
        return fig


# Integration functions for the main pipeline
def visualize_pipeline_results(experiment_dir: Path, 
                              visualization_types: List[str] = ["voronoi"],
                              save_plots: bool = True) -> Dict[str, Any]:
    """
    Visualize all results from a pipeline experiment.
    
    Args:
        experiment_dir: Directory containing experiment results
        visualization_types: Types of visualizations to create
        save_plots: Whether to save plots to disk
        
    Returns:
        Dictionary with paths to created visualizations
    """
    visualizer = PipelineVisualizer(output_dir=experiment_dir / "visualizations")
    created_plots = {}
    
    # Find ESN results
    esn_files = list(experiment_dir.glob("esn_mc_results_*.csv"))
    if esn_files:
        latest_esn = max(esn_files, key=lambda x: x.stat().st_mtime)
        
        for viz_type in visualization_types:
            savepath = visualizer.output_dir / f"esn_landscape_{viz_type}.png" if save_plots else None
            
            try:
                fig, ax = visualizer.plot_esn_memory_capacity_landscape(
                    latest_esn,
                    visualization_type=viz_type,
                    savepath=savepath,
                    show=not save_plots
                )
                created_plots[f"esn_{viz_type}"] = savepath
                plt.close(fig)
            except Exception as e:
                print(f"Failed to create ESN {viz_type} plot: {e}")
    
    # Find GNM results
    gnm_files = list(experiment_dir.glob("gnm_comprehensive_results.json"))
    if gnm_files:
        for gnm_file in gnm_files:
            for viz_type in visualization_types:
                savepath = visualizer.output_dir / f"gnm_landscape_{viz_type}.png" if save_plots else None
                
                try:
                    fig, ax = visualizer.plot_gnm_energy_landscape(
                        gnm_file,
                        visualization_type=viz_type,
                        savepath=savepath,
                        show=not save_plots
                    )
                    created_plots[f"gnm_{viz_type}"] = savepath
                    plt.close(fig)
                except Exception as e:
                    print(f"Failed to create GNM {viz_type} plot: {e}")
    
    # Create summary plot if both results exist
    if esn_files and gnm_files:
        savepath = visualizer.output_dir / "summary_plot.png" if save_plots else None
        try:
            fig = visualizer.create_summary_plot(
                esn_results_file=latest_esn if esn_files else None,
                gnm_results_file=gnm_files[0] if gnm_files else None,
                savepath=savepath
            )
            created_plots["summary"] = savepath
            plt.close(fig)
        except Exception as e:
            print(f"Failed to create summary plot: {e}")
    
    return created_plots


# Example usage functions
def example_with_real_data():
    """Example with real data: Visualize GNM parameter sweep results."""

    # esn_results_df = pd.read_csv("/Users/adrian/Documents/01_projects/14_4D_lab/output/02_esns_on_observed_weighted_connectomes/comprehensive_esn_sweep/esn_mc_results_2025-08-15_16-18-45.csv")
    
    # FOR GNM
    gnm_results = 
    
    # FOR REAL CONNECTOMES: 
        # subject,density_percent,spectral_radius,input_length,input_scaling,
        # regularization_method,n_runs,
        # mc_mean,mc_std,mean_mc_of_individual_runs,hyper_params

    df = gnm_results_df[["eta", "energy", "subject"]].copy()

    visualizer = PipelineVisualizer()
    
    # Compare visualization methods
    visualizer.compare_visualizations(df, title_prefix="GNM Parameter Sweep")


def example_visualize_esn_results():
    """Example: Visualize ESN hyperparameter optimization results."""
    # Create sample data for ESN results
    np.random.seed(42)
    n_samples = 150
    
    spectral_radius = np.random.uniform(0.5, 2.0, n_samples)
    input_length = np.random.choice([1000, 2000, 4000], n_samples)
    # Simulate memory capacity with peak around sr=0.99
    mc_values = np.exp(-2 * (spectral_radius - 0.99)**2) + np.random.normal(0, 0.05, n_samples)
    
    df = pd.DataFrame({
        'eta': spectral_radius,  # Using eta for x-axis
        'gamma': input_length,    # Using gamma for y-axis
        'energy': -mc_values,     # Negative MC as energy
        'subject': np.random.randint(0, 20, n_samples)
    })
    
    visualizer = PipelineVisualizer()
    
    # Create Voronoi plot
    fig, ax = visualizer.plot_energy_landscape_voronoi(
        df, 
        title="ESN Memory Capacity Landscape\n(Spectral Radius vs Input Length)",
        cmap="coolwarm",
        show_points=True
    )
    
    # Adjust labels
    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("Input Length")


if __name__ == "__main__":
    print("Testing visualization module...")
    
    # Run examples
    print("\n1. Testing GNM visualization:")
    example_visualize_gnm_sweep()
    
    print("\n2. Testing ESN visualization:")
    example_visualize_esn_results()
    
    print("\nVisualization tests completed!")