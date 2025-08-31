"""
Enhanced visualization for GNM-ESN energy landscapes.
Creates publication-quality heatmaps similar to the reference image.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata, interp2d
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json


class EnhancedLandscapeVisualizer:
    """Advanced visualization for GNM-ESN landscapes."""
    
    def __init__(self):
        # Create custom colormaps similar to the reference image
        self.create_custom_colormaps()
    
    def create_custom_colormaps(self):
        """Create custom colormaps matching the reference image."""
        # Yellow-red colormap for energy/MC landscapes
        colors_energy = [
            (0.0, (0.2, 0.0, 0.0)),      # Dark red/black
            (0.3, (0.6, 0.0, 0.0)),      # Red
            (0.5, (0.9, 0.3, 0.0)),      # Orange-red
            (0.7, (1.0, 0.6, 0.0)),      # Orange
            (0.85, (1.0, 0.9, 0.3)),     # Yellow-orange
            (1.0, (1.0, 1.0, 0.6))       # Light yellow
        ]
        
        n_bins = 256
        cmap_name = 'energy_landscape'
        self.energy_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_energy, N=n_bins)
        
        # Alternative: Use reversed hot colormap for memory capacity
        self.mc_cmap = plt.cm.hot_r
    
    def plot_gnm_esn_landscape(self,
                               results_file: Path,
                               metric: str = "mc_mean",
                               show_empirical: bool = True,
                               show_best_region: bool = True,
                               interpolation_method: str = "cubic",
                               n_interp: int = 100,
                               save_path: Optional[Path] = None,
                               figsize: Tuple[float, float] = (10, 8),
                               title: Optional[str] = None) -> plt.Figure:
        """
        Create publication-quality landscape visualization.
        
        Args:
            results_file: Path to landscape results JSON
            metric: Metric to plot ("mc_mean" or "gnm_energy")
            show_empirical: Show empirical network fits
            show_best_region: Highlight best parameter region
            interpolation_method: Method for interpolation
            n_interp: Number of interpolation points
            save_path: Path to save figure
            figsize: Figure size
            title: Custom title
            
        Returns:
            Matplotlib figure
        """
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        landscape_data = results["landscape_data"]
        empirical_fits = results.get("empirical_fits", [])
        params = results.get("parameters", {})
        
        # Convert to DataFrame
        df = pd.DataFrame(landscape_data)
        
        # Get unique eta and gamma values
        eta_unique = sorted(df['eta'].unique())
        gamma_unique = sorted(df['gamma'].unique())
        
        # Create dense grid for interpolation
        eta_dense = np.linspace(min(eta_unique), max(eta_unique), n_interp)
        gamma_dense = np.linspace(min(gamma_unique), max(gamma_unique), n_interp)
        eta_grid, gamma_grid = np.meshgrid(eta_dense, gamma_dense)
        
        # Interpolate data
        points = df[['eta', 'gamma']].values
        values = df[metric].values
        
        if interpolation_method == 'nearest':
            grid_values = griddata(points, values, (eta_grid, gamma_grid), method='nearest')
        else:
            # Use cubic interpolation with fallback to linear at boundaries
            grid_values = griddata(points, values, (eta_grid, gamma_grid), method='cubic')
            # Fill NaN values with nearest neighbor
            mask = np.isnan(grid_values)
            if np.any(mask):
                grid_values[mask] = griddata(points, values, 
                                            (eta_grid[mask], gamma_grid[mask]), 
                                            method='nearest')
        
        # Apply smoothing if needed
        from scipy.ndimage import gaussian_filter
        grid_values = gaussian_filter(grid_values, sigma=1.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine colormap and normalization
        if metric == "mc_mean":
            cmap = self.mc_cmap
            label = "Memory Capacity"
            vmin, vmax = np.nanmin(grid_values), np.nanmax(grid_values)
        else:
            cmap = self.energy_cmap
            label = "Energy"
            vmin, vmax = np.nanmin(grid_values), np.nanmax(grid_values)
        
        # Create the main heatmap
        im = ax.imshow(grid_values,
                      extent=[min(eta_unique), max(eta_unique),
                             min(gamma_unique), max(gamma_unique)],
                      origin='lower',
                      aspect='auto',
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax,
                      interpolation='bilinear')
        
        # Add contour lines for better readability
        contour_levels = np.linspace(vmin, vmax, 10)
        CS = ax.contour(eta_grid, gamma_grid, grid_values,
                       levels=contour_levels,
                       colors='black',
                       alpha=0.2,
                       linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=12)
        
        # Show empirical fits if available
        if show_empirical and empirical_fits:
            df_empirical = pd.DataFrame(empirical_fits)
            
            # Plot empirical points
            scatter = ax.scatter(df_empirical['eta'], df_empirical['gamma'],
                               s=150, c='lime', edgecolors='black', linewidths=2,
                               marker='*', label='Empirical connectomes', zorder=10)
            
            # Add circle around empirical region
            if show_best_region and len(df_empirical) > 0:
                center_eta = df_empirical['eta'].mean()
                center_gamma = df_empirical['gamma'].mean()
                radius_eta = df_empirical['eta'].std() + 0.5
                radius_gamma = df_empirical['gamma'].std() + 0.5
                
                circle = patches.Ellipse((center_eta, center_gamma),
                                        2*radius_eta, 2*radius_gamma,
                                        fill=False, edgecolor='blue',
                                        linewidth=2, linestyle='--',
                                        alpha=0.7)
                ax.add_patch(circle)
            
            # Add annotation with arrow
            annotation_text = ("Best fitting gammas and etas\n"
                             "of the empirical\n"
                             "connectomes (green)")
            
            # Find good position for annotation
            if center_eta < (min(eta_unique) + max(eta_unique)) / 2:
                ann_x = center_eta + 2
            else:
                ann_x = center_eta - 2
            
            if center_gamma < (min(gamma_unique) + max(gamma_unique)) / 2:
                ann_y = center_gamma + 2
            else:
                ann_y = center_gamma - 2
            
            ax.annotate(annotation_text,
                       xy=(center_eta, center_gamma),
                       xytext=(ann_x, ann_y),
                       fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="yellow", 
                                alpha=0.7),
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle="arc3,rad=0.3",
                                     color='black', 
                                     lw=1.5))
        
        # Set labels and title
        ax.set_xlabel('η', fontsize=14)
        ax.set_ylabel('γ', fontsize=14)
        
        if title is None:
            if metric == "mc_mean":
                title = "Memory Capacity Landscape - GNM-generated connectomes"
            else:
                title = "Energy landscape - calculated on GNM-generated connectomes"
        
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.2, linestyle=':', color='white')
        
        # Add legend if empirical points shown
        if show_empirical and empirical_fits:
            ax.legend(loc='best', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {save_path}")
        
        return fig
    
    def create_comparison_figure(self,
                                results_file: Path,
                                save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a comparison figure showing both MC and energy landscapes.
        
        Args:
            results_file: Path to results JSON
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        landscape_data = results["landscape_data"]
        empirical_fits = results.get("empirical_fits", [])
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Convert to DataFrame
        df = pd.DataFrame(landscape_data)
        
        # Check if we have both metrics
        has_mc = 'mc_mean' in df.columns and df['mc_mean'].notna().any()
        has_energy = 'gnm_energy' in df.columns and df['gnm_energy'].notna().any()
        
        if has_mc:
            self._plot_single_landscape(df, empirical_fits, 'mc_mean', 
                                       axes[0], "Memory Capacity")
        
        if has_energy:
            self._plot_single_landscape(df, empirical_fits, 'gnm_energy', 
                                       axes[1], "GNM Energy")
        
        # Overall title
        fig.suptitle("GNM-ESN Landscape Analysis", fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison figure to: {save_path}")
        
        return fig
    
    def _plot_single_landscape(self, df: pd.DataFrame, 
                              empirical_fits: List[Dict],
                              metric: str, ax: plt.Axes, 
                              title: str) -> None:
        """Helper method to plot a single landscape."""
        # Get unique values
        eta_unique = sorted(df['eta'].unique())
        gamma_unique = sorted(df['gamma'].unique())
        
        # Create grid
        eta_grid = np.linspace(min(eta_unique), max(eta_unique), 50)
        gamma_grid = np.linspace(min(gamma_unique), max(gamma_unique), 50)
        eta_mesh, gamma_mesh = np.meshgrid(eta_grid, gamma_grid)
        
        # Interpolate
        points = df[['eta', 'gamma']].values
        
        if metric in df.columns and df[metric].notna().any():
            values = df[metric].fillna(df[metric].mean()).values
            
            grid_values = griddata(points, values, 
                                  (eta_mesh, gamma_mesh), 
                                  method='cubic')
            
            # Fill NaNs
            mask = np.isnan(grid_values)
            if np.any(mask):
                grid_values[mask] = griddata(points, values,
                                            (eta_mesh[mask], gamma_mesh[mask]),
                                            method='nearest')
            
            # Plot
            im = ax.imshow(grid_values,
                          extent=[min(eta_unique), max(eta_unique),
                                 min(gamma_unique), max(gamma_unique)],
                          origin='lower',
                          aspect='auto',
                          cmap='hot_r' if metric == 'mc_mean' else 'hot',
                          interpolation='bilinear')
            
            plt.colorbar(im, ax=ax, label=title)
            
            # Add empirical points
            if empirical_fits:
                df_emp = pd.DataFrame(empirical_fits)
                ax.scatter(df_emp['eta'], df_emp['gamma'],
                          s=100, c='lime', edgecolors='black',
                          linewidths=2, marker='*', zorder=10)
        
        ax.set_xlabel('η')
        ax.set_ylabel('γ')
        ax.set_title(title)
        ax.grid(True, alpha=0.2)


def visualize_gnm_esn_landscape(results_path: str,
                               output_dir: Optional[str] = None,
                               show_plots: bool = True) -> Dict[str, Path]:
    """
    Convenience function to create all visualizations for a GNM-ESN landscape.
    
    Args:
        results_path: Path to landscape_results.json
        output_dir: Directory for saving plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with paths to saved figures
    """
    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Set output directory
    if output_dir is None:
        output_dir = results_file.parent / "visualizations"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    viz = EnhancedLandscapeVisualizer()
    
    saved_figures = {}
    
    # Create main landscape plot
    print("Creating main landscape visualization...")
    fig1 = viz.plot_gnm_esn_landscape(
        results_file,
        metric="mc_mean",
        show_empirical=True,
        show_best_region=True,
        save_path=output_dir / "landscape_main.png"
    )
    saved_figures["main"] = output_dir / "landscape_main.png"
    
    if not show_plots:
        plt.close(fig1)
    
    # Create comparison figure if both metrics available
    print("Creating comparison visualization...")
    try:
        fig2 = viz.create_comparison_figure(
            results_file,
            save_path=output_dir / "landscape_comparison.png"
        )
        saved_figures["comparison"] = output_dir / "landscape_comparison.png"
        
        if not show_plots:
            plt.close(fig2)
    except Exception as e:
        print(f"Could not create comparison figure: {e}")
    
    print(f"\nVisualizations saved to: {output_dir}")
    
    if show_plots:
        plt.show()
    
    return saved_figures


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Default test path
        results_path = "output/gnm_esn_landscape_test/landscape_results.json"
    
    try:
        saved = visualize_gnm_esn_landscape(
            results_path,
            show_plots=True
        )
        print(f"Created visualizations: {saved}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()