"""
Grid-based landscape visualization with subplot mosaic layout.
Combines gridded heatmap with multiple metric subplots.
"""

import os
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib import cm

from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer

class GridMosaicVisualizer(PipelineVisualizer):
    """Extension for gridded data visualization with mosaic subplot layout."""
    
    @staticmethod
    def cm_to_inch(cm_tuple):
        """Convert centimeters to inches for figure sizing."""
        return tuple(x / 2.54 for x in cm_tuple)
    
    def create_heatmap_data(self, df: pd.DataFrame, metric_name: str,
                           duplicate_handling: str = "mean",
                           eta_span: list = None,
                           gamma_span: list = None) -> pd.DataFrame:
        """
        Create a 2D DataFrame suitable for seaborn heatmap from gridded data.
        
        Args:
            df: DataFrame with columns ['eta', 'gamma', metric_name]
            metric_name: Name of the metric column
            duplicate_handling: How to handle duplicates ('mean', 'first', 'last')
            eta_span: Optional [min, max] range for eta
            gamma_span: Optional [min, max] range for gamma
            
        Returns:
            DataFrame with gamma as index, eta as columns, metric as values
        """
        # Filter by spans
        if eta_span:
            df = df[(df["eta"] >= eta_span[0]) & (df["eta"] <= eta_span[1])]
        if gamma_span:
            df = df[(df["gamma"] >= gamma_span[0]) & (df["gamma"] <= gamma_span[1])]
        
        # Handle duplicates
        if duplicate_handling == "mean":
            df_grouped = df.groupby(['eta', 'gamma'])[metric_name].mean().reset_index()
        elif duplicate_handling == "first":
            df_grouped = df.groupby(['eta', 'gamma'])[metric_name].first().reset_index()
        elif duplicate_handling == "last":
            df_grouped = df.groupby(['eta', 'gamma'])[metric_name].last().reset_index()
        else:
            df_grouped = df.copy()
        
        # Pivot to create 2D grid
        heatmap_data = df_grouped.pivot(index='gamma', columns='eta', values=metric_name)
        
        # Sort indices for proper ordering
        heatmap_data = heatmap_data.sort_index(ascending=True)
        heatmap_data = heatmap_data.sort_index(axis=1, ascending=True)
        
        return heatmap_data
    
    def plot_mosaic_landscape(self, df: pd.DataFrame,
                             metric_name: str,
                             additional_metrics: list = None,
                             title: str = "",
                             cmap: str = "hot",
                             savepath: Path = None,
                             show: bool = True,
                             point_color: str = "#9AA582FF",
                             eta_span: list = None,
                             gamma_span: list = None,
                             duplicate_handling: str = "mean",
                             show_colorbar: bool = False,
                             figsize_cm: tuple = (18, 7),
                             dpi: int = 150) -> tuple:
        """
        Create a mosaic plot with main landscape and additional metric subplots.
        
        Args:
            df: DataFrame with eta, gamma, and metric columns
            metric_name: Main metric to plot in landscape
            additional_metrics: List of up to 6 metric names for subplots
            title: Overall figure title
            cmap: Colormap for the main landscape
            savepath: Path to save figure
            show: Whether to display the figure
            point_color: Color for scatter points (if used)
            eta_span: [min, max] for eta filtering
            gamma_span: [min, max] for gamma filtering
            duplicate_handling: 'mean', 'first', or 'last'
            show_colorbar: Whether to show colorbar on landscape
            figsize_cm: Figure size in centimeters
            dpi: Figure DPI
            
        Returns:
            fig, axes: Matplotlib figure and axes dictionary
        """
        # Create heatmap data for main metric
        heatmap_data = self.create_heatmap_data(
            df, metric_name, duplicate_handling, eta_span, gamma_span
        )
        
        # Flip data vertically for proper orientation
        heatmap_data_flipped = heatmap_data.iloc[::-1]
        
        # Create mosaic layout
        if additional_metrics and len(additional_metrics) > 0:
            # Create subplot labels
            subplot_labels = [chr(65 + i) for i in range(min(6, len(additional_metrics)))]
            mosaic_layout = [
                ["landscape", "landscape"] + subplot_labels[:3],
                ["landscape", "landscape"] + subplot_labels[3:6] if len(subplot_labels) > 3 else ["landscape", "landscape"] + ["." for _ in range(3)]
            ]
        else:
            mosaic_layout = [["landscape"]]
        
        fig, axes = plt.subplot_mosaic(
            mosaic_layout,
            figsize=self.cm_to_inch(figsize_cm),
            dpi=dpi
        )
        
        # Plot main landscape heatmap
        sns.heatmap(
            heatmap_data_flipped,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar=show_colorbar,
            cmap=cmap,
            ax=axes["landscape"]
        )
        
        # Configure axes for landscape
        eta_values = heatmap_data_flipped.columns
        gamma_values = heatmap_data_flipped.index
        
        # Set x-axis ticks (η) - first, last, and a few in between
        eta_min, eta_max = eta_values.min(), eta_values.max()
        x_tick_positions = self._get_nice_tick_positions(eta_values, n_ticks=5)
        x_tick_labels = [self._format_tick_label(eta_values[pos]) for pos in x_tick_positions]
        axes["landscape"].set_xticks(x_tick_positions)
        axes["landscape"].set_xticklabels(x_tick_labels)
        
        # Set y-axis ticks (γ) - first, last, and a few in between
        gamma_min, gamma_max = gamma_values.min(), gamma_values.max()
        y_tick_positions = self._get_nice_tick_positions(gamma_values, n_ticks=5)
        # Labels are reversed because data is flipped
        y_tick_labels = [self._format_tick_label(gamma_values[-(pos+1)]) for pos in y_tick_positions]
        axes["landscape"].set_yticks(y_tick_positions)
        axes["landscape"].set_yticklabels(y_tick_labels, rotation=0)
        
        # Set axis labels
        axes["landscape"].set_xlabel(r"$\eta$", fontsize=12)
        axes["landscape"].set_ylabel(r"$\gamma$", fontsize=12)
        
        if title:
            axes["landscape"].set_title(title, fontsize=14, pad=10)
        
        # Plot additional metrics in subplots
        if additional_metrics:
            for idx, metric in enumerate(additional_metrics[:6]):
                subplot_label = chr(65 + idx)
                if subplot_label in axes:
                    try:
                        self._plot_metric_subplot(
                            axes[subplot_label], df, metric,
                            eta_span, gamma_span, duplicate_handling, cmap
                        )
                    except Exception as e:
                        print(f"Warning: Could not plot {metric}: {e}")
                        axes[subplot_label].axis('off')
        
        plt.tight_layout()
        
        # Save and show
        if savepath:
            savepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
            print(f"Saved figure to: {savepath}")
        
        if show:
            plt.show()
        
        return fig, axes
    
    def _get_nice_tick_positions(self, values, n_ticks=5):
        """Get evenly spaced tick positions including first and last."""
        n_values = len(values)
        if n_values <= n_ticks:
            return list(range(n_values))
        
        # Always include first and last
        positions = [0]
        
        # Add intermediate positions
        step = (n_values - 1) / (n_ticks - 1)
        for i in range(1, n_ticks - 1):
            positions.append(int(round(i * step)))
        
        positions.append(n_values - 1)
        
        return positions
    
    def _format_tick_label(self, value):
        """Format tick label to show decimals only when needed."""
        if value % 1 != 0:
            return f"{value:.1f}"
        else:
            return f"{int(value)}"
    
    def _plot_metric_subplot(self, ax, df, metric_name, eta_span, gamma_span,
                            duplicate_handling, cmap):
        """Plot a single metric in a subplot."""
        heatmap_data = self.create_heatmap_data(
            df, metric_name, duplicate_handling, eta_span, gamma_span
        )
        heatmap_data_flipped = heatmap_data.iloc[::-1]
        
        sns.heatmap(
            heatmap_data_flipped,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            cmap=cmap,
            ax=ax
        )
        
        # Format title
        title = self._format_plot_title(metric_name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")


if __name__ == "__main__":
    
    ##### CONFIG STUFF #######################
    experiment_name = "76_90000_samples_animal_206"
    base_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/{experiment_name}")
    all_metrics_file = True 

    ##########################################
    if all_metrics_file: 
        df_paths = [base_path / f"all_metrics_for_{experiment_name}.csv"]
    else: 
        save_path = base_path / f"all_computational_metrics_for_{experiment_name}_updated_combined.csv"
        df_static_path = base_path / f"all_static_metrics_for_{experiment_name}.csv"
        df_static = pd.read_csv(df_static_path)
        df_dynamic_path = base_path / f"all_dynamic_metrics_for_{experiment_name}_updated.csv"
        df_dynamic = pd.read_csv(df_dynamic_path)
        df_computational_path = base_path / f"all_computational_metrics_for_{experiment_name}_updated.csv"
        df_computational = pd.read_csv(df_computational_path)
        df_combined = pd.concat([df_static, df_dynamic, df_computational], axis=1)
        df_combined.to_csv(save_path)
        df_paths = [save_path]

    duplicate_handling = "mean"
    eta_span = [-8, 3]
    gamma_span = [-0.1, 1]
    
    # Output directory
    parent_folder = Path(df_paths[0]).parent
    save_path = parent_folder / f"figures_mosaic_{duplicate_handling}"
    
    # --- Define Metrics ---
    main_metric = "MaxCriteria"
    
    # Additional metrics for subplots (max 6)
    additional_metrics = [
        "global_efficiency",
        "modularity",
        "avg_clustering",
        "transitivity",
        "wiring_cost",
        "mc_mean"
    ]
    
    # Colormap options
    default_cmaps = {
        "hb_bw": "binary",  # Replace with your actual colormap
        "hot": "hot",
        "viridis": "viridis"
    }
    
    print(f"Starting Mosaic visualization process...")
    print(f"Output will be saved to: {save_path}\n")

    # --- Load Data ---
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
                
    except Exception as e:
        print(f"ERROR: Failed to load data. {e}")
        exit()

    # Convert columns to numeric
    gnm_results_df["eta"] = pd.to_numeric(gnm_results_df["eta"], errors='coerce')
    gnm_results_df["gamma"] = pd.to_numeric(gnm_results_df["gamma"], errors='coerce')
    
    # Convert metric columns to numeric
    for metric in [main_metric] + additional_metrics:
        if metric in gnm_results_df.columns:
            gnm_results_df[metric] = pd.to_numeric(gnm_results_df[metric], errors='coerce')
    
    # Drop rows with NaN in essential columns
    gnm_results_df = gnm_results_df.dropna(subset=['eta', 'gamma'])
    
    visualizer = GridMosaicVisualizer()
    
    # --- Generate Mosaic Plot ---
    print(f"--- Generating Mosaic plot for {main_metric} with {len(additional_metrics)} subplots ---")
    try:
        plot_title = visualizer._format_plot_title(main_metric)
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        figure_save_name = f"mosaic_landscape_{main_metric}.pdf"
        full_save_path = save_dir / figure_save_name

        fig, axes = visualizer.plot_mosaic_landscape(
            gnm_results_df,
            metric_name=main_metric,
            additional_metrics=additional_metrics,
            title=plot_title,
            cmap="cubehelix", # default_cmaps["hot"],
            savepath=full_save_path,
            show=False,
            eta_span=eta_span,
            gamma_span=gamma_span,
            duplicate_handling=duplicate_handling,
            show_colorbar=False,
            figsize_cm=(18, 7),
            dpi=150
        )
        
        matplotlib.pyplot.close()
        
        print(f"Successfully generated mosaic plot at {full_save_path}\n")
        
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while plotting.")
        traceback.print_exc()
        print("\n")

    print("--- Visualization process completed. ---")