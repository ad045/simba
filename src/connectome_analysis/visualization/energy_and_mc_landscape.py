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
from typing import Optional, Dict, Any, List, Union, Tuple
import json
from datetime import datetime

from src.connectome_analysis.visualization import PlotManager


class PipelineVisualizer:
    """Visualization tools for connectome analysis pipeline results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        
        # Setup matplotlib with default config
        self.plot_manager = PlotManager() 
        self.output_dir = Path(output_dir) if output_dir else self.plot_manager.path_config.figures_dir # default path: output/figures
        # self.plot_manager.plot_config.dpi = 400 
        
    
    def plot_energy_landscape_voronoi(self, df: pd.DataFrame, 
                                      dot_color: str = "white", 
                                      title: str = "", 
                                      cmap: str = "hot", 
                                      savepath: Optional[Path] = None, 
                                      show: bool = True,
                                      show_points: bool = False, 
                                      point_size: int = 8,
                                      vmin: Optional[float] = None, 
                                      vmax: Optional[float] = None,
                                      ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot energy landscape using Voronoi diagram for randomly sampled points.
        
        Args:
            df: DataFrame containing 'eta', 'gamma', and 'energy' columns.
            dot_color: Color for the sample points.
            title: Title for the plot.
            cmap: Colormap for the energy values.
            savepath: Path to save the figure.
            show: Whether to display the plot.
            show_points: Whether to show the actual sample points.
            point_size: Size of the sample points if shown.
            vmin: Minimum value for color scale.
            vmax: Maximum value for color scale.
            ax: Existing axis to plot on (if None, creates new figure).
            
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        
        # Validate input
        required_cols = ["eta", "gamma", "energy"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Average energy for each unique (eta, gamma) pair
        grid_df = df.groupby(["eta", "gamma"], as_index=False)["energy"].mean()
        
        # Extract points and values
        points = grid_df[["eta", "gamma"]].values
        energies = grid_df["energy"].values
        
        # Create or use provided axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
        else:
            fig = ax.get_figure()
        
        # Determine plot bounds with padding
        eta_min, eta_max = points[:, 0].min(), points[:, 0].max()
        gamma_min, gamma_max = points[:, 1].min(), points[:, 1].max()
        eta_range = eta_max - eta_min
        gamma_range = gamma_max - gamma_min
        padding = 0.1
        
        xlim = [eta_min - padding * eta_range, eta_max + padding * eta_range]
        ylim = [gamma_min - padding * gamma_range, gamma_max + padding * gamma_range]
        
        # Create boundary points for complete Voronoi cells
        boundary_points = self._create_boundary_points(xlim, ylim, n_boundary=20)
        all_points = np.vstack([points, boundary_points])
        
        # Create Voronoi diagram
        vor = Voronoi(all_points)
        
        # Setup color normalization
        vmin = vmin if vmin is not None else energies.min()
        vmax = vmax if vmax is not None else energies.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap(cmap)
        
        # Color each Voronoi region (only for actual data points)
        for i in range(len(points)):
            region_idx = vor.point_region[i]
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
        
        # Show sample points if requested
        if show_points:
            ax.scatter(points[:, 0], points[:, 1], 
                      s=point_size, c=dot_color, 
                      edgecolors='black', linewidths=0.5,
                      alpha=0.8, zorder=5)
        
        # Show best points per subject if available
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
        
        # Save if path provided
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
        elif show and ax is None:  # Only show if not part of subplot
            plt.show()
        
        return fig, ax
    
    def plot_energy_landscape_interpolated(self, df: pd.DataFrame, 
                                          method: str = 'cubic', 
                                          resolution: int = 100,
                                          dot_color: str = "white", 
                                          title: str = "", 
                                          cmap: str = "hot", 
                                          savepath: Optional[Path] = None, 
                                          show: bool = True,
                                          vmin: Optional[float] = None, 
                                          vmax: Optional[float] = None,
                                          ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot energy landscape using interpolation for randomly sampled points.
        
        Args:
            df: DataFrame containing 'eta', 'gamma', and 'energy' columns.
            method: Interpolation method ('linear', 'cubic', 'nearest').
            resolution: Grid resolution for interpolation.
            dot_color: Color for the best-fit points.
            title: Title for the plot.
            cmap: Colormap for the energy values.
            savepath: Path to save the figure.
            show: Whether to display the plot.
            vmin: Minimum value for color scale.
            vmax: Maximum value for color scale.
            ax: Existing axis to plot on (if None, creates new figure).
            
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        from scipy.interpolate import griddata
        
        # Validate input
        required_cols = ["eta", "gamma", "energy"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Average energy for each unique (eta, gamma) pair
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
        
        # Create or use provided axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
        else:
            fig = ax.get_figure()
        
        # Plot interpolated surface
        vmin = vmin if vmin is not None else values.min()
        vmax = vmax if vmax is not None else values.max()
        
        im = ax.imshow(energy_interp,
                      origin="lower",
                      extent=[eta_min, eta_max, gamma_min, gamma_max],
                      aspect="auto",
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Energy", rotation=270, labelpad=12)
        
        # Show sample points
        ax.scatter(points[:, 0], points[:, 1], 
                  s=8, c=dot_color, edgecolors='black', 
                  linewidths=0.5, alpha=0.7, zorder=5)
        
        # Show best points per subject if available
        if "subject" in df.columns:
            df_best = df.loc[df.groupby("subject")["energy"].idxmin()].reset_index(drop=True)
            ax.scatter(df_best["eta"].values, df_best["gamma"].values,
                      s=16, c="cyan", edgecolors="black", linewidths=0.5,
                      alpha=0.95, zorder=10, marker='*')
        
        ax.set_xlabel("η")
        ax.set_ylabel("γ")
        ax.set_title(title)
        
        # Save if path provided
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
        elif show and ax is None:  # Only show if not part of subplot
            plt.show()
        
        return fig, ax
    
    
    def compare_visualizations(self, df: pd.DataFrame, 
                          title_prefix: str = "Energy Landscape",
                          savepath: Optional[Path] = None,
                          save_individual: bool = True,
                          save_format: str = "png") -> plt.Figure:
        """
        Create a comparison of different visualization methods.
        
        Args:
            df: DataFrame with 'eta', 'gamma', 'energy' columns
            title_prefix: Prefix for titles
            savepath: Base path for saving (without extension)
            save_individual: Whether to save individual plots
        
        Returns:
            Figure object with comparison plots
        """
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Get data range for consistent coloring
        grid_df = df.groupby(["eta", "gamma"], as_index=False)["energy"].mean()
        vmin, vmax = grid_df["energy"].min(), grid_df["energy"].max()
        
        # Setup save paths
        individual_paths = {}
        if savepath:
            savepath = Path(savepath)
            save_dir = savepath.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            base_name = savepath.stem
            
            if save_individual:
                individual_paths = {
                    'voronoi': save_dir / f"{base_name}_voronoi.{save_format}",
                    'cubic': save_dir / f"{base_name}_cubic.{save_format}",
                    'linear': save_dir / f"{base_name}_linear.{save_format}"
                }
        
        # 1. Voronoi diagram - DO NOT save individual here
        self.plot_energy_landscape_voronoi(
            df, 
            title=f"{title_prefix} - Voronoi", 
            show=False, 
            vmin=vmin, 
            vmax=vmax,
            ax=axes[0],
            savepath=None  # Don't save individual plot here
        )
        
        # 2. Interpolated (cubic) - DO NOT save individual here
        self.plot_energy_landscape_interpolated(
            df, 
            method='cubic', 
            title=f"{title_prefix} - Cubic Interpolation",
            show=False, 
            vmin=vmin, 
            vmax=vmax,
            ax=axes[1],
            savepath=None  # Don't save individual plot here
        )
        
        # 3. Interpolated (linear) - DO NOT save individual here
        self.plot_energy_landscape_interpolated(
            df, 
            method='linear',
            title=f"{title_prefix} - Linear Interpolation",
            show=False, 
            vmin=vmin, 
            vmax=vmax,
            ax=axes[2],
            savepath=None  # Don't save individual plot here
        )
        
        plt.tight_layout()
        
        # Save individual plots if requested (create separate figures for this)
        if save_individual and savepath:
            for method, path in individual_paths.items():
                if method == 'voronoi':
                    fig_ind, _ = self.plot_energy_landscape_voronoi(
                        df, title=f"{title_prefix} - Voronoi", 
                        show=False, vmin=vmin, vmax=vmax,
                        savepath=path
                    )
                elif method == 'cubic':
                    fig_ind, _ = self.plot_energy_landscape_interpolated(
                        df, method='cubic',
                        title=f"{title_prefix} - Cubic Interpolation",
                        show=False, vmin=vmin, vmax=vmax,
                        savepath=path
                    )
                elif method == 'linear':
                    fig_ind, _ = self.plot_energy_landscape_interpolated(
                        df, method='linear',
                        title=f"{title_prefix} - Linear Interpolation",
                        show=False, vmin=vmin, vmax=vmax,
                        savepath=path
                    )
                plt.close(fig_ind)
            print(f"Saved individual plots to: {save_dir}")
        
        # Save comparison figure
        if savepath:
            comparison_path = savepath.parent / f"{base_name}_comparison.{save_format}"
            fig.savefig(comparison_path, bbox_inches="tight", dpi=150)
            print(f"Saved comparison plot to: {comparison_path}")
        
        plt.show()
        
        return fig



    def _create_boundary_points(self, xlim: List[float], ylim: List[float],
                                n_boundary: int = 20) -> np.ndarray:
        """
        Create boundary points for Voronoi diagram.
        
        Args:
            xlim: X-axis limits [min, max]
            ylim: Y-axis limits [min, max]
            n_boundary: Number of boundary points per edge
            
        Returns:
            Array of boundary points
        """
        boundary_points = []
        
        for i in range(n_boundary):
            # Bottom and top edges
            x_pos = xlim[0] + i * (xlim[1] - xlim[0]) / (n_boundary - 1)
            boundary_points.append([x_pos, ylim[0] - 1])
            boundary_points.append([x_pos, ylim[1] + 1])
            
            # Left and right edges
            y_pos = ylim[0] + i * (ylim[1] - ylim[0]) / (n_boundary - 1)
            boundary_points.append([xlim[0] - 1, y_pos])
            boundary_points.append([xlim[1] + 1, y_pos])
        
        return np.array(boundary_points)
    
    def plot_gnm_energy_landscape(self, gnm_results_file: Path, 
                                  visualization_type: str = "voronoi",
                                  density: Optional[int] = None,
                                  savepath: Optional[Path] = None,
                                  show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
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
        
        # Build title
        title = f"GNM Energy Landscape (Density {density}%)" if density else "GNM Energy Landscape"
        
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
                                          show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
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
        # Load and process ESN results
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
        
        # Create energy landscape data (negative MC as "energy")
        df_landscape = pd.DataFrame({
            "eta": df_results[param1],
            "gamma": df_results[param2],
            "energy": -df_results['mc_mean'],  # Negative so higher MC appears as lower energy
            "subject": df_results['subject']
        })
        
        # Drop NaN values
        df_landscape = df_landscape.dropna()
        
        # Build title
        title_parts = ["ESN Memory Capacity Landscape"]
        if density is not None:
            title_parts[0] += f" (Density {density}%)"
        title_parts.append(f"{param1} vs {param2}")
        title = "\n".join(title_parts)
        
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
        
        # Update colorbar label for memory capacity
        if fig.axes:
            for ax_item in fig.axes:
                if hasattr(ax_item, 'get_ylabel') and ax_item.get_ylabel() == "Energy":
                    ax_item.set_ylabel("Memory Capacity", rotation=270, labelpad=12)
        
        return fig, ax
    
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
            print("Processing ESN results...")
            self.plot_esn_memory_capacity_landscape(
                esn_results_file,
                visualization_type="voronoi",
                show=False,
                ax=axes[plot_idx]
            )
            axes[plot_idx].set_title("ESN Memory Capacity")
            plot_idx += 1
        
        # Plot GNM results if available
        if gnm_results_file and gnm_results_file.exists():
            print("Processing GNM results...")
            self.plot_gnm_energy_landscape(
                gnm_results_file,
                visualization_type="voronoi",
                show=False,
                ax=axes[plot_idx]
            )
            axes[plot_idx].set_title("GNM Energy Landscape")
        
        plt.suptitle("Pipeline Results Summary", fontsize=14, y=1.02)
        plt.tight_layout()
        
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
            print(f"Saved summary plot to: {savepath}")
        
        plt.show()
        
        return fig


# Integration functions
def visualize_pipeline_results(experiment_dir: Path, 
                              visualization_types: List[str] = ["voronoi"],
                              save_plots: bool = True, 
                              save_format: Optional[str] = "png") -> Dict[str, Any]:
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
    
    # Find and process ESN results
    esn_files = list(experiment_dir.glob("esn_mc_results_*.csv"))
    if esn_files:
        latest_esn = max(esn_files, key=lambda x: x.stat().st_mtime)
        
        for viz_type in visualization_types:
            plot_key = f"esn_{viz_type}"
            savepath = visualizer.output_dir / f"esn_landscape_{viz_type}.{save_format}" if save_plots else None
            
            try:
                fig, ax = visualizer.plot_esn_memory_capacity_landscape(
                    latest_esn,
                    visualization_type=viz_type,
                    savepath=savepath,
                    show=not save_plots
                )
                created_plots[plot_key] = savepath
                plt.close(fig)
                print(f"Created {plot_key} visualization")
            except Exception as e:
                print(f"Failed to create ESN {viz_type} plot: {e}")
    
    # Find and process GNM results
    gnm_files = list(experiment_dir.glob("gnm_comprehensive_results.json"))
    if gnm_files:
        for gnm_file in gnm_files:
            for viz_type in visualization_types:
                plot_key = f"gnm_{viz_type}"
                savepath = visualizer.output_dir / f"gnm_landscape_{viz_type}.{save_format}" if save_plots else None
                
                try:
                    fig, ax = visualizer.plot_gnm_energy_landscape(
                        gnm_file,
                        visualization_type=viz_type,
                        savepath=savepath,
                        show=not save_plots
                    )
                    created_plots[plot_key] = savepath
                    plt.close(fig)
                    print(f"Created {plot_key} visualization")
                except Exception as e:
                    print(f"Failed to create GNM {viz_type} plot: {e}")
    
    # Create summary plot if both results exist
    if esn_files and gnm_files:
        savepath = visualizer.output_dir / f"summary_plot.{save_format}" if save_plots else None
        try:
            fig = visualizer.create_summary_plot(
                esn_results_file=latest_esn,
                gnm_results_file=gnm_files[0],
                savepath=savepath
            )
            created_plots["summary"] = savepath
            plt.close(fig)
            print("Created summary plot")
        except Exception as e:
            print(f"Failed to create summary plot: {e}")
    
    return created_plots


def visualize_gnm_results(df_path: Union[str, Path], 
                         name_of_energy_metric: str = "MaxCriteria(DegreeKS_ClusteringKS)",
                         save_dir: Optional[Path] = None,
                         save_name: Optional[str] = None,
                         save_format: str = "png",
                         save_individual: bool = True) -> None:
    """
    Visualize GNM parameter sweep results.
    
    Args:
        df_path: Path to the results CSV file
        name_of_energy_metric: Name of the energy metric column
        save_dir: Directory to save visualizations
        save_name: Base name for saved files
        save_individual: Whether to save individual plots in addition to comparison
    """
    # Load and process data
    gnm_results_df = pd.read_csv(df_path, index_col=False, sep=", ")
    
    # Prepare DataFrame
    df = gnm_results_df[["eta", "gamma", name_of_energy_metric]].copy()
    df.columns = ["eta", "gamma", "energy"]
    
    # Convert to numeric, handling string format
    df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
    df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
    
    # Handle energy values (remove trailing comma if present)
    energy_values = df["energy"].astype(str).str.rstrip(',')
    df["energy"] = pd.to_numeric(energy_values, errors='coerce')
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    if df.empty:
        raise ValueError("No valid data after processing")
    
    # Initialize visualizer
    visualizer = PipelineVisualizer()
    
    # Setup save path
    save_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"gnm_landscape_{timestamp}"
        
        save_path = save_dir / save_name
    
    # Create visualizations
    print(f"Creating visualizations for {len(df)} data points...")
    print(f"Energy range: {df['energy'].min():.4f} to {df['energy'].max():.4f}")
    
    visualizer.compare_visualizations(
        df, 
        title_prefix="GNM Parameter Sweep", 
        savepath=save_path,
        save_individual=save_individual, 
        save_format=save_format
    )


# Example usage -> see other files for this... 
if __name__ == "__main__":
    
    print("Testing improved visualization module...")
    
    # Test with GNM results
    print("\n1. Testing GNM visualization:")
    
    name_of_energy_metric = "MaxCriteria(DegreeKS_ClusteringKS)"
    print(f"Using energy metric: {name_of_energy_metric}")
    
    # Update with your actual file path
    # One old file (not the hyper large one)
    df_name = "/Users/adrian/Documents/01_projects/14_4D_lab/OLD_output_3/default_folder/binary_evaluations_resultsdistance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_400.csv"

    # Extract folder name for organization
    folder_name = Path(df_name).stem
    save_path = Path("output/visualizations") / folder_name
    
    try:
        visualize_gnm_results(
            df_path=df_name,
            name_of_energy_metric=name_of_energy_metric,
            save_dir=save_path,
            save_format="pdf", 
            save_individual=True  # Save both individual and comparison plots
        )
        print(f"\nVisualization completed successfully!")
        print(f"Results saved to: {save_path}")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nVisualization tests completed!")