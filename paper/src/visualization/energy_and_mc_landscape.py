"""
Visualization module for connectome analysis pipeline.
Integrates energy landscape plotting for both GNM and ESN results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import json
from datetime import datetime
from collections import defaultdict
import os
import re


class PipelineVisualizer:
    """Visualization tools for connectome analysis pipeline results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        # self.plot_manager = PlotManager() 
        self.output_dir = Path(output_dir) if output_dir else None

    def plot_average_mc_curve(self, 
                              df_gnms: pd.DataFrame, 
                              df_empirical: Optional[pd.DataFrame] = None,
                              savepath: Optional[Path] = None, 
                              plot_all_individual_mc_curves: bool = False,
                              show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the average memory capacity (MC) for each lag across all runs.
        """
        
        # For df_gnms (simulated GNMs)
        print("Plotting average MC curve for simulated connectomes...")
        mc_cols = sorted([col for col in df_gnms.columns if col.startswith('mc_') and col.split('_')[1].isdigit()], 
                        key=lambda x: int(x.split('_')[1]))
        
        if not mc_cols:
            print("No 'mc_xx' columns found in the simulated DataFrame. Skipping average MC plot.")
            return None, None

        mc_means = df_gnms[mc_cols].mean()
        lags = [int(col.split('_')[1]) for col in mc_cols]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if plot_all_individual_mc_curves: 
            for row in range(df_gnms.shape[0]):
                ax.plot(lags[1:], df_gnms.iloc[row][mc_cols[1:]], color="gray", alpha=0.05) 
        
        ax.plot(lags[1:], mc_means[1:], marker='o', linestyle='-', color='royalblue', label='Average MC (simulated connectomes)')
        print("Finished plotting simulated connectomes...")
        
        
        # For df_empirical (empirical connectomes)
        print("Plotting average MC curve for empirical connectomes...")
        mc_cols = sorted([col for col in df_empirical.columns if col.startswith('mc_') and col.split('_')[1].isdigit()], 
                        key=lambda x: int(x.split('_')[1]))
        
        if not mc_cols:
            print("No 'mc_xx' columns found in the empirical DataFrame. Skipping average MC plot.")
            return None, None

        mc_means = df_empirical[mc_cols].mean()
        lags = [int(col.split('_')[1]) for col in mc_cols]
        
        if plot_all_individual_mc_curves: 
            for row in range(df_empirical.shape[0]):
                ax.plot(lags[1:], df_empirical.iloc[row][mc_cols[1:]], color="green", alpha=0.05) 
        
        ax.plot(lags[1:], mc_means[1:], marker='o', linestyle='-', color='green', label='Average MC (empirical connectomes)')
        print("Finished plotting empirical connectomes...")
                   
        # Format plots 
        self._format_mc_plot(ax, lags)
            
        # Save and display plots 
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
            print(f"Saved average MC curve to: {savepath}")
        if show:
            plt.show()
        return fig, ax
    

    def _format_mc_plot(self, ax: plt.Axes, lags: List[int]) -> None:
        """Format memory capacity plot styling."""
        ax.set_title("Average Memory Capacity vs. Lag", fontsize=16)
        ax.set_xlabel("Lag (k)", fontsize=12)
        ax.set_ylabel("Average Memory Capacity (MC_k)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticks(lags)
        ax.set_xlim(left=0) 
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend()
        plt.tight_layout()


    def _prepare_landscape_data(self, df: pd.DataFrame, 
                                metric_name: str, 
                                normalize_aspect: bool = False, 
                                duplicate_handling: str = "mean", # mean, first, last
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for landscape visualization.
        """
        required_cols = ["eta", "gamma", metric_name]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Duplicate handling. Default: Average metric for each unique (eta, gamma) pair
        if duplicate_handling == "mean":
            # Average metric for each unique (eta, gamma) pair
            print("mean")
            grid_df = df.groupby(["eta", "gamma"], as_index=False)[metric_name].mean()
        elif duplicate_handling == "first":
            # Keep first occurrence of each (eta, gamma) pair
            print("first")
            grid_df = df.drop_duplicates(subset=["eta", "gamma"], keep="first")
        elif duplicate_handling == "last":
            # Keep last occurrence of each (eta, gamma) pair
            grid_df = df.drop_duplicates(subset=["eta", "gamma"], keep="last")
        else:
            raise ValueError(
                f"Invalid duplicate_handling: '{duplicate_handling}'. "
                f"Must be 'mean', 'first', or 'last'."
            )


        # grid_df = df.groupby(["eta", "gamma"], as_index=False)[metric_name].mean()
        points = grid_df[["eta", "gamma"]].values
        metric_values = grid_df[metric_name].values
        
        # Normalize points if requested
        norm_info = None
        if normalize_aspect:
            eta_range = points[:, 0].max() - points[:, 0].min()
            gamma_range = points[:, 1].max() - points[:, 1].min()
            
            # Store normalization info for later denormalization
            norm_info = {
                'eta_min': points[:, 0].min(),
                'eta_max': points[:, 0].max(),
                'gamma_min': points[:, 1].min(),
                'gamma_max': points[:, 1].max(),
                'eta_range': eta_range,
                'gamma_range': gamma_range
            }
            
            # Normalize to [0, 1] range
            points_normalized = points.copy()
            points_normalized[:, 0] = (points[:, 0] - norm_info['eta_min']) / eta_range
            points_normalized[:, 1] = (points[:, 1] - norm_info['gamma_min']) / gamma_range
            
            return points_normalized, metric_values, norm_info
        
        return points, metric_values, None


    def _setup_plot_bounds(self, points: np.ndarray, padding: float = 0.0) -> Tuple[List[float], List[float]]:
        """Calculate plot bounds with optional padding."""
        eta_min, eta_max = points[:, 0].min(), points[:, 0].max()
        gamma_min, gamma_max = points[:, 1].min(), points[:, 1].max()
        eta_range = eta_max - eta_min
        gamma_range = gamma_max - gamma_min
        
        xlim = [eta_min - padding * eta_range, eta_max + padding * eta_range]
        ylim = [gamma_min - padding * gamma_range, gamma_max + padding * gamma_range]
        
        return xlim, ylim


    def _setup_colormap(self, metric_values: np.ndarray, vmin: Optional[float], 
                       vmax: Optional[float], cmap: str):
        """Setup color normalization and colormap."""
        vmin = vmin if vmin is not None else metric_values.min()
        vmax = vmax if vmax is not None else metric_values.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap(cmap)
        
        return norm, colormap, vmin, vmax


    def _add_sample_points(self, ax: plt.Axes, df: pd.DataFrame, points: np.ndarray, 
                          metric_name: str, show_dots: bool, 
                          dot_color: str, point_size: int, 
                          label: bool = False # important for "Rat4" or "Rat" etc labels
                          ):
        """Add sample points and best points per subject to plot."""
        if show_dots:
            ax.scatter(points[:, 0], points[:, 1], 
                      s=point_size, c=dot_color, 
                      edgecolors=dot_color,
                      linewidths=0.5,
                      alpha=0.8, zorder=5)
            
            ax.scatter(points[0, 0], points[0, 1],  # TODO: WHY 0 ??? -> Artefact from earlier? 
                      s=point_size, c=dot_color, 
                      edgecolors=dot_color,
                      linewidths=1,
                      alpha=0.8, zorder=5)
        
        # Show best points per subject if available
        if "subject" in df.columns:
            df_best = df.loc[df.groupby("subject")[metric_name].idxmin()].reset_index(drop=True)
            ax.scatter(df_best["eta"].values, df_best["gamma"].values,
                      s=16, c=dot_color, edgecolors=dot_color, 
                      linewidths=0.5,
                      alpha=0.95, zorder=10, marker='*')
        
            if label: # important for writing "Rat" (i.e.: "name") or "Rat3" (i.e.: "animal") tags! 
                # Annotate each point with its index 
                for idx, row in df_best.iterrows():
                    ax.annotate(str(int(row["subject"])), 
                                (row["eta"], row["gamma"]),
                                textcoords="offset points", 
                                xytext=(0,5), 
                                ha='center', fontsize=8, color="black", zorder=15)


    def _add_estimated_connectomes(self, ax: plt.Axes, estimated_indiv_connectomes: Optional[pd.DataFrame], 
                                  dot_color: str, dot_size_animals_or_humans: int):
        """Add estimated individual connectomes to plot."""
        if estimated_indiv_connectomes is not None:
            ax.scatter(
            #     estimated_indiv_connectomes["eta"],
            #     estimated_indiv_connectomes["gamma"], 
            #     facecolors='none',
            #     # facecolors = estimated_indiv_connectomes[""]
            #     s=20,   # 20                 
            #     edgecolors=dot_color, 
            #     label='Estimated Optima' 
            # )
                estimated_indiv_connectomes["eta"],
                estimated_indiv_connectomes["gamma"], 
                facecolors=dot_color, 
                # facecolors = estimated_indiv_connectomes[""]
                s=dot_size_animals_or_humans,   # 20
                edgecolors=None,
                label='Estimated Optima' 
            )
            

    def _format_landscape_plot(self, ax: plt.Axes, xlim: List[float], ylim: List[float], 
                              title: str):
        """Apply common formatting to landscape plots."""
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title)
        ax.set_aspect('auto')

    def plot_metric_landscape_voronoi(self, df: pd.DataFrame, 
                                      dot_color: str = "steelblue", 
                                      title: str = "", 
                                      cmap: str = "hot", 
                                      savepath: Optional[Path] = None, 
                                      metric_name: Optional[str] = None,
                                      show: bool = True,
                                      show_dots: bool = False, 
                                      point_size: int = 8,
                                      vmin: Optional[float] = None,
                                      eta_span: Optional[list[float]] = None, #  = [-0.125, 1],   # set to None to have no selection 
                                      gamma_span: Optional[list[float]] = None, # = [-7.5, 2.5],  # set to None to have no selection 
                                      vmax: Optional[float] = None,
                                      ax: Optional[plt.Axes] = None, 
                                      annotate_extremes: Optional[bool] = False, 
                                      show_colorbar: bool = True, 
                                      estimated_indiv_connectomes: Optional[pd.DataFrame] = None, 
                                      normalize_aspect: bool = False,
                                      duplicate_handling: str = "mean", # mean, first, last
                                      show_number_samples: bool = False,
                                      ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a metric landscape using Voronoi diagram for randomly sampled points."""
        
        # Prepare data
        if eta_span:
            df = df[df["eta"]>=eta_span[0]]
            df = df[df["eta"]<=eta_span[1]]
        if gamma_span:
            df = df[df["gamma"]>=gamma_span[0]] 
            df = df[df["gamma"]<=gamma_span[1]]
        points, metric_values, norm_info = self._prepare_landscape_data(df, metric_name, duplicate_handling=duplicate_handling)
        
        # Create or use provided axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
        else:
            fig = ax.get_figure()
        
        # Setup plot bounds and colormap
        xlim, ylim = self._setup_plot_bounds(points, padding=0.0)
        norm, colormap, vmin, vmax = self._setup_colormap(metric_values, vmin, vmax, cmap)
        
        # Create boundary points and Voronoi diagram
        boundary_points = self._create_boundary_points(xlim, ylim, n_boundary=20)
        all_points = np.vstack([points, boundary_points])
        vor = Voronoi(all_points)
        
        # Color each Voronoi region (only for actual data points)
        for i in range(len(points)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 not in region and len(region) > 0:
                polygon_vertices = [vor.vertices[v] for v in region]
                polygon = patches.Polygon(polygon_vertices, 
                                         facecolor=colormap(norm(metric_values[i])),
                                         edgecolor='none')
                ax.add_patch(polygon)
        
        # If we normalized, denormalize the axis labels and limits
        if normalize_aspect and norm_info:
            # Set normalized limits for the plot
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            # Create custom tick labels that show the original values
            n_ticks = 5
            xticks_norm = np.linspace(xlim[0], xlim[1], n_ticks)
            yticks_norm = np.linspace(ylim[0], ylim[1], n_ticks)
            
            xticks_orig = xticks_norm * norm_info['eta_range'] + norm_info['eta_min']
            yticks_orig = yticks_norm * norm_info['gamma_range'] + norm_info['gamma_min']
            
            ax.set_xticks(xticks_norm)
            ax.set_yticks(yticks_norm)
            ax.set_xticklabels([f'{x:.2f}' for x in xticks_orig])
            ax.set_yticklabels([f'{y:.2f}' for y in yticks_orig])
            
            # Handle estimated connectomes normalization
            if estimated_indiv_connectomes is not None:
                est_conn_norm = estimated_indiv_connectomes.copy()
                est_conn_norm["eta"] = (estimated_indiv_connectomes["eta"] - norm_info['eta_min']) / norm_info['eta_range']
                est_conn_norm["gamma"] = (estimated_indiv_connectomes["gamma"] - norm_info['gamma_min']) / norm_info['gamma_range']
                estimated_indiv_connectomes = est_conn_norm
                
        # Add colorbar, points, and annotations
        if show_colorbar:
            sm = cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            self._format_colorbar(metric_name, cbar)
            
        self._add_sample_points(ax, df, points, metric_name, show_dots, dot_color, point_size)
        self._add_estimated_connectomes(ax, estimated_indiv_connectomes, dot_color)
        
        if annotate_extremes:
            self._annotate_extreme_points(ax, df, metric_name)
        
        if show_number_samples: 
            title = title + (f"({len(df)} samples)")
        self._format_landscape_plot(ax, xlim, ylim, title)
        


        # Save/show
        self._handle_plot_output(fig, ax, savepath, show)
        
        return fig, ax


    def plot_metric_landscape_interpolated(self, df: pd.DataFrame, 
                                          method: str = 'cubic', 
                                          resolution: int = 100,
                                          dot_color: str = "steelblue", 
                                          title: str = "", 
                                          cmap: str = "hot", 
                                          savepath: Optional[Path] = None, 
                                          metric_name: Optional[str] = None,
                                          show: bool = True,
                                          show_dots: Optional[bool] = True, 
                                          vmin: Optional[float] = None, 
                                          vmax: Optional[float] = None,
                                          ax: Optional[plt.Axes] = None, 
                                          annotate_extremes: Optional[bool] = False, 
                                          ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a metric landscape using interpolation for randomly sampled points."""
        
        # Prepare data
        points, metric_values = self._prepare_landscape_data(df, metric_name)
        
        # Create interpolation grid
        eta_min, eta_max = points[:, 0].min(), points[:, 0].max()
        gamma_min, gamma_max = points[:, 1].min(), points[:, 1].max()
        
        eta_grid = np.linspace(eta_min, eta_max, resolution)
        gamma_grid = np.linspace(gamma_min, gamma_max, resolution)
        eta_mesh, gamma_mesh = np.meshgrid(eta_grid, gamma_grid)
        
        metric_interp = griddata(points, metric_values, (eta_mesh, gamma_mesh), method=method)
        
        # Create or use provided axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
        else:
            fig = ax.get_figure()
        
        # Setup colormap
        vmin = vmin if vmin is not None else metric_values.min()
        vmax = vmax if vmax is not None else metric_values.max()
        
        # Plot interpolated surface
        im = ax.imshow(metric_interp,
                      origin="lower",
                      extent=[eta_min, eta_max, gamma_min, gamma_max],
                      aspect="auto",
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax)
        
        # Add colorbar and points
        cbar = plt.colorbar(im, ax=ax)
        self._format_colorbar(metric_name, cbar)
        
        self._add_sample_points(ax, df, points, metric_name, show_dots, dot_color, 8)
        
        if annotate_extremes:
            self._annotate_extreme_points(ax, df, metric_name)
        
        ax.set_xlabel("η")
        ax.set_ylabel("γ")
        ax.set_title(title)
        
        # Save/show
        self._handle_plot_output(fig, ax, savepath, show)
        
        return fig, ax

    def _format_colorbar(self, metric_name: str, cbar) -> None:
        """Format colorbar with appropriate label."""

        labelpad = 35  
        
        # Use scientific notation if values are very large or very small
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 3))  # Use scientific notation outside this range
        cbar.ax.yaxis.set_major_formatter(formatter)
        
        # Set label
        if metric_name: 
            cbar.set_label(f"Metric: \n{metric_name}", rotation=270, labelpad=labelpad)
        # else: 
        #     cbar.set_label("Metric Value", rotation=270, labelpad=35)


    def _handle_plot_output(self, fig: plt.Figure, ax: plt.Axes, 
                           savepath: Optional[Path], show: bool) -> None:
        """Handle saving and showing plots consistently."""
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
        elif show and ax.get_figure().axes[0] == ax:  # Only show if not part of subplot
            plt.show()

    def compare_visualizations(self, df: pd.DataFrame, 
                          title_prefix: str = "Metric Landscape",
                          metric_name: Optional[str] = None, 
                          show_dots: Optional[bool] = True,
                          savepath: Optional[Path] = None,
                          save_individual: bool = True,
                          save_format: Optional[str] = "png", 
                          annotate_extremes: Optional[bool] = True,
                          ) -> plt.Figure:
        """Create a comparison of different visualization methods for a given metric."""
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Get data range for consistent coloring
        _, metric_values = self._prepare_landscape_data(df, metric_name)
        vmin, vmax = metric_values.min(), metric_values.max()
        
        # Setup save paths
        individual_paths = self._setup_individual_save_paths(savepath, save_individual, save_format)
        
        # Create visualization methods configuration
        viz_methods = [
            ('voronoi', f"{title_prefix} - Voronoi"),
            ('cubic', f"{title_prefix} - Cubic Interpolation"),
            ('linear', f"{title_prefix} - Linear Interpolation")
        ]
        
        # Generate plots
        for i, (method, title) in enumerate(viz_methods):
            if method == 'voronoi':
                self.plot_metric_landscape_voronoi(
                    df, title=title, metric_name=metric_name,
                    show=False, show_dots=show_dots, vmin=vmin, vmax=vmax,
                    ax=axes[i], annotate_extremes=annotate_extremes, savepath=None
                )
            else:  # interpolated methods
                self.plot_metric_landscape_interpolated(
                    df, method=method, title=title, metric_name=metric_name,
                    show=False, show_dots=show_dots, vmin=vmin, vmax=vmax,
                    ax=axes[i], annotate_extremes=annotate_extremes, savepath=None
                )
        
        # Synchronize axis limits
        self._synchronize_axes_limits(axes)
        plt.tight_layout()
        
        # Save individual and comparison plots
        if save_individual and savepath:
            self._save_individual_plots(df, title_prefix, metric_name, show_dots, 
                                      vmin, vmax, individual_paths)
        
        if savepath:
            comparison_path = savepath.parent / f"{savepath.stem}_comparison.{save_format}"
            fig.savefig(comparison_path, bbox_inches="tight", dpi=150)
            print(f"Saved comparison plot to: {comparison_path}")
        
        return fig


    def _setup_individual_save_paths(self, savepath: Optional[Path], save_individual: bool, 
                                   save_format: str) -> Dict[str, Path]:
        """Setup save paths for individual plots."""
        individual_paths = {}
        if savepath and save_individual:
            savepath = Path(savepath)
            save_dir = savepath.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            base_name = savepath.stem
            
            individual_paths = {
                'voronoi': save_dir / f"{base_name}_voronoi.{save_format}",
                'cubic': save_dir / f"{base_name}_cubic.{save_format}",
                'linear': save_dir / f"{base_name}_linear.{save_format}"
            }
        return individual_paths


    def _synchronize_axes_limits(self, axes: List[plt.Axes]) -> None:
        """Synchronize axis limits across multiple plots."""
        xlim = axes[1].get_xlim()
        ylim = axes[1].get_ylim()
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)


    def _save_individual_plots(self, df: pd.DataFrame, title_prefix: str, metric_name: str,
                              show_dots: bool, vmin: float, vmax: float, 
                              individual_paths: Dict[str, Path]) -> None:
        """Save individual plots separately."""
        plot_methods = {
            'voronoi': lambda path: self.plot_metric_landscape_voronoi(
                df, title=f"{title_prefix} - Voronoi", metric_name=metric_name,
                show=False, show_dots=show_dots, vmin=vmin, vmax=vmax, savepath=path
            ),
            'cubic': lambda path: self.plot_metric_landscape_interpolated(
                df, method='cubic', title=f"{title_prefix} - Cubic Interpolation",
                metric_name=metric_name, show=False, show_dots=show_dots, 
                vmin=vmin, vmax=vmax, savepath=path
            ),
            'linear': lambda path: self.plot_metric_landscape_interpolated(
                df, method='linear', title=f"{title_prefix} - Linear Interpolation",
                metric_name=metric_name, show=False, show_dots=show_dots, 
                vmin=vmin, vmax=vmax, savepath=path
            )
        }
        
        for method, path in individual_paths.items():
            fig_ind, _ = plot_methods[method](path)
            plt.close(fig_ind)
        
        print(f"Saved individual plots to: {individual_paths['voronoi'].parent}")


    def plot_mc_landscapes_grid(self, df: pd.DataFrame, 
                               lags: List[int],
                               method: str = 'voronoi',  # 'voronoi' or 'interpolated'
                               interpolation_method: str = 'linear',
                               n_cols: int = 3,
                               savepath: Optional[Path] = None,
                               save_format: str = "png",
                               cmap: str = "hot") -> plt.Figure:
        """
        Creates a grid of landscapes for different memory capacity (MC) lags.
        Unified method that can use either Voronoi or interpolation.
        """
        n_lags = len(lags)
        n_rows = (n_lags + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(n_cols * 5, n_rows * (4.5 if method == 'voronoi' else 4)), 
                                sharex=True, sharey=True, 
                                layout="tight" if method == 'voronoi' else None)
        axes = axes.flatten()

        # Find global min/max for consistent color scaling
        mc_cols = [f'mc_{lag}' for lag in lags if f'mc_{lag}' in df.columns]
        if not mc_cols:
            raise ValueError("None of the specified MC lag columns were found in the DataFrame.")
            
        vmin = df[mc_cols].min().min()
        vmax = df[mc_cols].max().max()

        # Plot each lag
        for i, lag in enumerate(lags):
            ax = axes[i]
            metric_name = f'mc_{lag}'
            
            if metric_name not in df.columns:
                self._handle_missing_metric(ax, lag)
                continue

            if method == 'voronoi':
                self.plot_metric_landscape_voronoi(
                    df, title=f"MC Lag {lag}", metric_name=metric_name,
                    show=False, show_dots=False, vmin=vmin, vmax=vmax,
                    ax=ax, annotate_extremes=True, show_colorbar=False,
                    cmap=cmap
                )
            else:  # interpolated
                self.plot_metric_landscape_interpolated(
                    df, method=interpolation_method, title=f"MC Lag {lag}",
                    metric_name=metric_name, show=False, show_dots=False,
                    vmin=vmin, vmax=vmax, ax=ax, cmap=cmap
                )
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # Add shared colorbar and title
        if method == 'voronoi':
            self._add_shared_colorbar(fig, axes, vmin, vmax, cmap)
        
        method_display = f"Voronoi" if method == 'voronoi' else f"Interpolation: {interpolation_method}"
        fig.suptitle(f"Memory Capacity Landscapes ({method_display})", 
                    fontsize=16 + (4 if method == 'voronoi' else 0), 
                    y=1.02)
        
        # Save and show
        if savepath:
            suffix = f"_{method}" if method == 'voronoi' else f"_{interpolation_method}"
            full_save_path = savepath.parent / f"{savepath.stem}_mc_lags_grid{suffix}.{save_format}"
            fig.savefig(full_save_path, bbox_inches="tight", dpi=150)
            print(f"Saved MC lag grid plot to: {full_save_path}")
            
        # plt.show()
        return fig


    def _handle_missing_metric(self, ax: plt.Axes, lag: int) -> None:
        """Handle case where metric data is missing."""
        ax.set_title(f"MC Lag {lag}\n(Not Found)")
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', fontsize=12)


    def _add_shared_colorbar(self, fig: plt.Figure, axes: List[plt.Axes], 
                           vmin: float, vmax: float, cmap: str) -> None:
        """Add a shared colorbar for the entire figure."""
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), label="Memory Capacity", shrink=0.8)

    # Legacy method aliases for backwards compatibility
    def plot_mc_lag_landscapes(self, *args, **kwargs):
        """Legacy method - use plot_mc_landscapes_grid with method='interpolated'."""
        return self.plot_mc_landscapes_grid(*args, method='interpolated', **kwargs)
    
    
    def plot_mc_lag_landscapes_voronoi(self, *args, **kwargs):
        """Legacy method - use plot_mc_landscapes_grid with method='voronoi'."""
        return self.plot_mc_landscapes_grid(*args, method='voronoi', **kwargs)


    def _create_boundary_points(self, xlim: List[float], ylim: List[float],
                                n_boundary: int = 20) -> np.ndarray:
        """Create boundary points for Voronoi diagram."""
        boundary_points = []
        
        for i in range(n_boundary):
            x_pos = xlim[0] + i * (xlim[1] - xlim[0]) / (n_boundary - 1)
            boundary_points.append([x_pos, ylim[0] - 1])
            boundary_points.append([x_pos, ylim[1] + 1])
            
            y_pos = ylim[0] + i * (ylim[1] - ylim[0]) / (n_boundary - 1)
            boundary_points.append([xlim[0] - 1, y_pos])
            boundary_points.append([xlim[1] + 1, y_pos])
        
        return np.array(boundary_points)


    def _format_plot_title(self, metric_name: str) -> str:
        """Formats a metric name into a pretty plot title with LaTeX."""
        if "MaxCriteria" in metric_name: # "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)"
            title = "Energy"
            match = re.search(r'\((.*?)\)', metric_name)
            if match:
                components_str = match.group(1)
                parts = re.findall(r'([A-Z][a-zA-Z_]*?)KS', components_str)
                
                formatted_parts = []
                for part in parts:
                    sub = "edge" if "EdgeLength" in part else part.lower()
                    formatted_parts.append(f"KS_{{{sub}}}")

                if formatted_parts:
                    title += f" (Max(${', '.join(formatted_parts)}$))"
                    
            return title + "\n" # Parameter Sweep"
        
        return f"{metric_name.replace('_', ' ').title()}\n" # Parameter Sweep"


    def _annotate_extreme_points(self, ax: plt.Axes, df: pd.DataFrame, metric_name: str):
        """Finds and annotates the extreme points on a plot."""
        grid_df = df.groupby(["eta", "gamma"], as_index=False)[metric_name].mean()

        is_energy_plot = 'energy' in metric_name.lower() or 'criteria' in metric_name.lower()

        points_to_annotate = []
        if is_energy_plot:
            min_point = grid_df.loc[grid_df[metric_name].idxmin()]
            points_to_annotate.append(min_point)
        else:
            min_point = grid_df.loc[grid_df[metric_name].idxmin()]
            max_point = grid_df.loc[grid_df[metric_name].idxmax()]
            if not min_point.equals(max_point):
                points_to_annotate.extend([min_point, max_point])
            else:
                points_to_annotate.append(max_point)

        min_gamma = grid_df['gamma'].min()
        max_gamma = grid_df['gamma'].max()

        for point in points_to_annotate:
            x, y, value = point['eta'], point['gamma'], point[metric_name]
            
            ax.scatter(x, y, s=20, # 40
                      facecolors='none', edgecolors='cyan', 
                      linewidths=2.0, zorder=15)

            ax.text(x, y + 0.03 * (max_gamma - min_gamma),
                    f'{value:.4f}',
                    color='white',
                    fontsize=4, # 7,
                    ha='center',
                    zorder=16,
                    bbox=dict(facecolor='black', alpha=0.6, 
                             edgecolor='none', boxstyle='round,pad=0.2'))


    # Specialized plotting methods for different data types
    def plot_gnm_energy_landscape(self, gnm_results_file: Path, 
                                  visualization_type: str = "voronoi",
                                  density: Optional[int] = None,
                                  savepath: Optional[Path] = None,
                                  show: bool = True, 
                                  show_dots: Optional[bool] = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plot GNM energy landscape from pipeline results."""
        df = self._load_gnm_results(gnm_results_file)
        title = f"GNM Energy Landscape (Density {density}%)" if density else "GNM Energy Landscape"
        
        return self._plot_landscape_by_type(df, "energy", title, visualization_type, 
                                          savepath, show, show_dots, "viridis")

    def plot_esn_memory_capacity_landscape(self, esn_results_file: Path,
                                          param1: str = "spectral_radius",
                                          param2: str = "input_length",
                                          visualization_type: str = "voronoi",
                                          density: Optional[int] = None,
                                          savepath: Optional[Path] = None,
                                          show: bool = True, 
                                          show_dots: Optional[bool] = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plot ESN memory capacity landscape from pipeline results."""
        df_landscape = self._load_esn_results(esn_results_file, param1, param2, density)
        
        title_parts = ["ESN Memory Capacity Landscape"]
        if density is not None:
            title_parts[0] += f" (Density {density}%)"
        title_parts.append(f"{param1} vs {param2}")
        title = "\n".join(title_parts)
        
        fig, ax = self._plot_landscape_by_type(df_landscape, "energy", title, visualization_type, 
                                             savepath, show, show_dots, "coolwarm")
        
        # Update colorbar label for memory capacity
        self._update_esn_colorbar_labels(fig)
        
        return fig, ax


    def _load_gnm_results(self, gnm_results_file: Path) -> pd.DataFrame:
        """Load and process GNM results from JSON file."""
        with open(gnm_results_file, 'r') as f:
            results = json.load(f)
        
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
        
        return pd.DataFrame(energy_data)


    def _load_esn_results(self, esn_results_file: Path, param1: str, param2: str, 
                         density: Optional[int] = None) -> pd.DataFrame:
        """Load and process ESN results from CSV file."""
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
        
        return df_landscape.dropna()


    def _plot_landscape_by_type(self, df: pd.DataFrame, metric_name: str, title: str,
                               visualization_type: str, savepath: Optional[Path],
                               show: bool, show_dots: bool, cmap: str) -> Tuple[plt.Figure, plt.Axes]:
        """Plot landscape using specified visualization type."""
        if visualization_type == "voronoi":
            return self.plot_metric_landscape_voronoi(
                df, title=title, savepath=savepath, metric_name=metric_name,
                show=show, show_dots=show_dots, cmap=cmap
            )
        else:
            return self.plot_metric_landscape_interpolated(
                df, method='cubic', title=title, metric_name=metric_name,
                savepath=savepath, show=show, show_dots=show_dots, cmap=cmap
            )


    def _update_esn_colorbar_labels(self, fig: plt.Figure) -> None:
        """Update colorbar labels for ESN plots to show Memory Capacity instead of Energy."""
        if fig.axes:
            for ax_item in fig.axes:
                if hasattr(ax_item, 'get_ylabel') and ax_item.get_ylabel() == "Energy":
                    ax_item.set_ylabel("Memory Capacity", rotation=270, labelpad=18)


    def create_summary_plot(self, esn_results_file: Optional[Path] = None,
                           gnm_results_file: Optional[Path] = None,
                           savepath: Optional[Path] = None, 
                           show_dots: Optional[bool] = True) -> plt.Figure:
        """Create a summary plot combining ESN and GNM results."""
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
            fig_esn, ax_esn = self.plot_esn_memory_capacity_landscape(
                esn_results_file, visualization_type="voronoi",
                show=False, show_dots=show_dots
            )
            # Copy the plot to our summary axes
            self._copy_plot_to_axis(ax_esn, axes[plot_idx])
            axes[plot_idx].set_title("ESN Memory Capacity")
            plt.close(fig_esn)
            plot_idx += 1
        
        # Plot GNM results if available
        if gnm_results_file and gnm_results_file.exists():
            print("Processing GNM results...")
            fig_gnm, ax_gnm = self.plot_gnm_energy_landscape(
                gnm_results_file, visualization_type="voronoi",
                show=False, show_dots=show_dots
            )
            # Copy the plot to our summary axes
            self._copy_plot_to_axis(ax_gnm, axes[plot_idx])
            axes[plot_idx].set_title("GNM Energy Landscape")
            plt.close(fig_gnm)
        
        plt.suptitle("Pipeline Results Summary", fontsize=18, y=1.02)
        plt.tight_layout()
        
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=150)
            print(f"Saved summary plot to: {savepath}")
        
        plt.show()
        return fig


    def _copy_plot_to_axis(self, source_ax: plt.Axes, target_ax: plt.Axes) -> None:
        """Copy plot elements from source axis to target axis."""
        # This is a simplified approach - in practice, you might want to recreate
        # the plots directly on the target axes for better control
        target_ax.set_xlim(source_ax.get_xlim())
        target_ax.set_ylim(source_ax.get_ylim())
        target_ax.set_xlabel(source_ax.get_xlabel())
        target_ax.set_ylabel(source_ax.get_ylabel())


# Integration and utility functions
def generate_entire_df(df_paths: List[str]) -> pd.DataFrame:
    """
    Appends all dataframes from a list of file paths into a single dataframe.
    
    This function groups files by their base filename. Within each group, it combines
    dataframes only if their parameter columns are identical.
    """
    grouped_paths = defaultdict(list)
    for path in df_paths:
        filename = os.path.basename(path)
        grouped_paths[filename].append(path)
        
    all_valid_dfs = []
    
    # Define the columns that must be identical for dataframes to be combined
    consistency_cols = [
        'distance_relationship_type', 
        'preferential_relationship_type', 
        'generative_rule', 
        'num_iterations'
    ]

    print("Starting dataframe processing...")
    
    for filename, paths in grouped_paths.items():
        print(f"\nProcessing group: {filename} ({len(paths)} file(s))")
        
        compatible_dfs_in_group = []
        first_df_signature = None
        
        for path in paths:
            try:
                current_df = pd.read_csv(path)
                
                # Check if required columns exist
                if not all(col in current_df.columns for col in consistency_cols):
                    print(f"  - WARNING: Skipping {path}. Missing consistency columns.")
                    continue
                
                # Create signature from first row of consistency columns
                current_signature = tuple(current_df.loc[0, consistency_cols])
                
                if first_df_signature is None:
                    first_df_signature = current_signature
                    compatible_dfs_in_group.append(current_df)
                    print(f"  - Established baseline parameters from: {path}")
                elif current_signature == first_df_signature:
                    compatible_dfs_in_group.append(current_df)
                    print(f"  - Parameters match. Adding: {path}")
                else:
                    print(f"  - WARNING: Skipping {path}. Parameters don't match baseline.")
                    
            except FileNotFoundError:
                print(f"  - ERROR: File not found at {path}. Skipping.")
            except Exception as e:
                print(f"  - ERROR: Could not process file {path}. Reason: {e}. Skipping.")
        
        if compatible_dfs_in_group:
            group_df = pd.concat(compatible_dfs_in_group, ignore_index=True)
            all_valid_dfs.append(group_df)
            
    if not all_valid_dfs:
        print("\nNo valid dataframes were found to combine.")
        return pd.DataFrame()

    print("\nConcatenating all processed groups into final dataframe...")
    final_df = pd.concat(all_valid_dfs, ignore_index=True)
    print("Processing complete.")
    
    return final_df


def visualize_pipeline_results(experiment_dir: Path, 
                              visualization_types: List[str] = ["voronoi"],
                              save_plots: bool = True, 
                              show_dots: Optional[bool] = True, 
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
    
    # Process ESN results
    esn_files = list(experiment_dir.glob("esn_mc_results_*.csv"))
    if esn_files:
        latest_esn = max(esn_files, key=lambda x: x.stat().st_mtime)
        created_plots.update(
            _create_visualization_plots(visualizer, "esn", latest_esn, 
                                      visualization_types, save_plots, show_dots, save_format)
        )
    
    # Process GNM results
    gnm_files = list(experiment_dir.glob("gnm_comprehensive_results.json"))
    if gnm_files:
        for gnm_file in gnm_files:
            created_plots.update(
                _create_visualization_plots(visualizer, "gnm", gnm_file,
                                          visualization_types, save_plots, show_dots, save_format)
            )
    
    # Create summary plot if both results exist
    if esn_files and gnm_files:
        _create_summary_plot(visualizer, latest_esn, gnm_files[0], 
                           save_plots, show_dots, save_format, created_plots)
    
    return created_plots


def _create_visualization_plots(visualizer: PipelineVisualizer, plot_type: str, 
                               results_file: Path, visualization_types: List[str],
                               save_plots: bool, show_dots: bool, save_format: str) -> Dict[str, Any]:
    """Create visualization plots for a specific result type."""
    plots = {}
    
    for viz_type in visualization_types:
        plot_key = f"{plot_type}_{viz_type}"
        savepath = (visualizer.output_dir / f"{plot_type}_landscape_{viz_type}.{save_format}" 
                   if save_plots else None)
        
        try:
            if plot_type == "esn":
                fig, ax = visualizer.plot_esn_memory_capacity_landscape(
                    results_file, visualization_type=viz_type,
                    savepath=savepath, show_dots=show_dots, show=not save_plots
                )
            else:  # gnm
                fig, ax = visualizer.plot_gnm_energy_landscape(
                    results_file, visualization_type=viz_type,
                    savepath=savepath, show_dots=show_dots, show=not save_plots
                )
            
            plots[plot_key] = savepath
            plt.close(fig)
            print(f"Created {plot_key} visualization")
            
        except Exception as e:
            print(f"Failed to create {plot_type} {viz_type} plot: {e}")
    
    return plots


def _create_summary_plot(visualizer: PipelineVisualizer, esn_file: Path, gnm_file: Path,
                        save_plots: bool, show_dots: bool, save_format: str, 
                        created_plots: Dict[str, Any]) -> None:
    """Create and save summary plot."""
    savepath = (visualizer.output_dir / f"summary_plot.{save_format}" 
               if save_plots else None)
    try:
        fig = visualizer.create_summary_plot(
            esn_results_file=esn_file, gnm_results_file=gnm_file,
            savepath=savepath, show_dots=show_dots
        )
        created_plots["summary"] = savepath
        plt.close(fig)
        print("Created summary plot")
    except Exception as e:
        print(f"Failed to create summary plot: {e}")


def visualize_gnm_results(df_paths: Union[str, Path, List[Union[str, Path]]],
                         metric_to_visualize: str = "MaxCriteria",
                         save_dir: Optional[Path] = None,
                         save_name: Optional[str] = None,
                         save_format: str = "png",
                         show_dots: Optional[bool] = True,
                         save_individual: bool = True) -> str:
    """
    Visualize GNM parameter sweep results for a given metric.
    
    Args:
        df_paths: Path to results CSV file(s). Can be one path or list of paths.
        metric_to_visualize: Name of the metric column to visualize.
        save_dir: Directory to save visualizations.
        save_name: Base name for saved files.
        save_individual: Whether to save individual plots in addition to comparison.
        
    Returns:
        The actual metric column name that was used for visualization.
    """
    
    # Load and prepare data
    gnm_results_df = _load_gnm_data(df_paths)
    gnm_results_df = _add_computed_metrics(gnm_results_df, metric_to_visualize)
    df = _prepare_gnm_dataframe(gnm_results_df, metric_to_visualize)
    
    # Initialize visualizer and setup paths
    visualizer = PipelineVisualizer()
    save_path = _setup_gnm_save_path(save_dir, save_name, metric_to_visualize)
    
    # Find metric column and create visualizations
    metric_col_name = _find_metric_column(gnm_results_df, metric_to_visualize)
    plot_title_prefix = visualizer._format_plot_title(metric_col_name)
    
    print(f"Creating visualizations for '{metric_col_name}' using {len(df)} data points...")
    print(f"Metric range: {df[metric_col_name].min():.4f} to {df[metric_col_name].max():.4f}")
    
    visualizer.compare_visualizations(
        df, title_prefix=plot_title_prefix, metric_name=metric_col_name,
        savepath=save_path, show_dots=show_dots, save_individual=save_individual, 
        save_format=save_format
    )
    
    return metric_col_name


def _load_gnm_data(df_paths: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load GNM data from single file or multiple files."""
    if isinstance(df_paths, (list, tuple)):
        if len(df_paths) > 1:
            return generate_entire_df(df_paths)
        elif len(df_paths) == 1:
            return pd.read_csv(df_paths[0], index_col=False)
        else:
            raise ValueError("Input 'df_paths' is an empty list.")
    else:
        return pd.read_csv(df_paths, index_col=False)


def _add_computed_metrics(df: pd.DataFrame, metric_to_visualize: str) -> pd.DataFrame:
    """Add computed metrics to dataframe if needed."""
    if metric_to_visualize == "mean_mc_divided_by_wiring_cost": 
        df["mean_mc_divided_by_wiring_cost"] = (
            pd.to_numeric(df["mc_mean"], errors='coerce') / 
            pd.to_numeric(df["wiring_cost"], errors='coerce')
        )
    elif metric_to_visualize == "mc_5_divided_by_wiring_cost": 
        df["mc_5_divided_by_wiring_cost"] = (
            pd.to_numeric(df["mc_5"], errors='coerce') / 
            pd.to_numeric(df["wiring_cost"], errors='coerce')
        )
    return df


def _prepare_gnm_dataframe(df: pd.DataFrame, metric_to_visualize: str) -> pd.DataFrame:
    """Prepare dataframe for visualization by converting types and cleaning data."""
    df = df.copy()
    df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
    df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
    
    metric_col_name = _find_metric_column(df, metric_to_visualize)
    df[metric_col_name] = pd.to_numeric(df[metric_col_name], errors='coerce')
    
    # Drop any rows with NaN values in essential columns
    df = df.dropna(subset=['eta', 'gamma', metric_col_name])
    
    if df.empty:
        raise ValueError("No valid data after processing for the specified metric.")
    
    return df


def _find_metric_column(df: pd.DataFrame, metric_to_visualize: str) -> str:
    """Find the actual metric column name in the dataframe."""
    try:
        return next(col for col in df.columns if metric_to_visualize in col)
    except StopIteration:
        raise ValueError(
            f"Metric '{metric_to_visualize}' not found in DataFrame columns: "
            f"{df.columns.tolist()}"
        )


def _setup_gnm_save_path(save_dir: Optional[Path], save_name: Optional[str], 
                        metric_to_visualize: str) -> Optional[Path]:
    """Setup save path for GNM visualizations."""
    if save_dir is None:
        return None
        
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"gnm_landscape_{metric_to_visualize}_{timestamp}"
    
    return save_dir / save_name