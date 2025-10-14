"""
Enhanced visualization module for connectome analysis with taxonomic coloring.
Generates Voronoi landscape plots with color-coded points by taxonomy.
"""

import os
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
        print("Warning: 'animal' column not found in results. Merging just by index (is no issue to worry though, if code has not been changed significantly).")
        merged_df = pd.concat([gnm_results_df, metadata_df], axis=1) # gnm_results_df
    
    return merged_df


def plot_voronoi_with_taxonomic_colors(visualizer, df, metric_name, title, 
                                       color_mapper, label_by, 
                                       df_best_estimates,
                                       savepath, 
                                       normalize_aspect=False):
    """
    Create Voronoi plot with taxonomic coloring overlaid.
    
    This function creates the base Voronoi plot and then adds colored points on top.
    """
    # First, create the base Voronoi landscape (without colored dots)
    fig, ax = visualizer.plot_metric_landscape_voronoi(
        df, 
        title=title,
        metric_name=metric_name,
        savepath=None,  # Don't save yet
        dot_color="steelblue", 
        show=False,
        show_dots=False,  # Don't show dots yet
        annotate_extremes=True, 
        estimated_indiv_connectomes=None,  # Add separately
        show_colorbar=True,
        normalize_aspect=normalize_aspect,
    )
    
    # Now add colored scatter points on top
    # Group by eta, gamma to get one point per parameter combination
    # grouped = df.groupby(['eta', 'gamma']).agg({
    #     metric_name: 'mean',
    #     color_mapper.category_column: 'first'  # Take first category for this point
    # }).reset_index()
    
    # # Get colors for each point
    # point_colors = color_mapper.get_colors_for_df(grouped)
    
    # # Add colored scatter points
    # scatter = ax.scatter(
    #     grouped['eta'], 
    #     grouped['gamma'],
    #     c=point_colors,
    #     s=50,
    #     edgecolors='white',
    #     linewidths=1,
    #     alpha=0.9,
    #     zorder=10
    # )
    
    # # Add labels if requested
    # if label_by and label_by in df.columns:
    #     for idx, row in grouped.iterrows():
    #         # Get the label for this eta/gamma combination
    #         point_df = df[(df['eta'] == row['eta']) & (df['gamma'] == row['gamma'])]
    #         if not point_df.empty and label_by in point_df.columns:
    #             label = str(point_df[label_by].iloc[0])
    #             ax.annotate(
    #                 label, 
    #                 (row['eta'], row['gamma']), 
    #                 fontsize=4, 
    #                 alpha=0.8,
    #                 xytext=(5, 5), 
    #                 textcoords='offset points',
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
    #                          alpha=0.7, edgecolor='none')
    #             )
    
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
            s=20,
            edgecolors="black", 
            linewidths=0.5,
            marker='o',
            label='Estimated Optima',
            zorder=15
        )
    
    # # Add labels if requested
    if label_by and label_by in df_best_estimates.columns:
        for index, row in df_best_estimates.iterrows():
            label = str(row[label_by])
            ax.annotate(
                label, 
                (row['eta'], row['gamma']), 
                fontsize=4, 
                alpha=0.8,
                xytext=(5, 5), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.7, edgecolor='none')
            )
                
                
    # Add legend for taxonomic categories
    legend_elements = color_mapper.create_legend_elements()
    # plt.legend()
    legend = ax.legend(
        handles=legend_elements, 
        title=color_mapper.category_column.replace('_', ' ').title(),
        loc='lower left', # center left',
        # bbox_to_anchor=(1.15, 0.5),
        fontsize=5, # 9 
        # framealpha=0.95,
        # edgecolor='gray'
    )
    
    plt.tight_layout()
    
    # Save
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Successfully saved: {savepath}")
    
    return fig, ax


if __name__ == "__main__":
    
    # --- 1. Define Input and Output ---
    
    df_paths = [
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/31_suarez_MaMI_size_100_wider_sweep_57_copy_2_now_run_with_evaluation/all_metrics_for_exp_30_shafiei_size_68.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/33_suarez_MaMI_size_100_extensive_220_300_iter/all_metrics_for_exp_30_shafiei_size_68_interim_copy.csv"
        "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/49_suarez_MaMI_100/all_metrics_for_49_suarez_MaMI_100.csv"
    ]
    
    # Path to the enriched animal metadata
    animal_metadata_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_100.csv" 
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_100.csv"
    
    parent_folder = Path(df_paths[0]).parent
    save_path = parent_folder / "figures_voronoi_taxonomic"
    
    # --- 2. Taxonomic Visualization Settings ---
    
    # Choose which taxonomic level to color by
    # Options: 'family', 'genus', 'species', 'order', 'sub_order', 'super_order', 
    #          'sub_family', 'phylogenetic_group', 'common_name'
    COLOR_BY = "order" # phylogenetic_group' # family' # phylogenetic_group'  # Change this to your preference
    
    # Choose which label to show on points
    LABEL_BY = "name" # phylogenetic_group" # family" # None  # 'common_name' or None
    
    # Whether to plot individual connectomes
    plot_indiv_connectomes = True
    
    # --- 3. Define Metrics to Plot ---
    
    lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 20, 49]
    metrics_to_plot = [
        "MaxCriteria",
        "avg_communicability", 
        "global_efficiency", 
        "modularity", 
        "avg_clustering", 
        "avg_degree", 
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
    
    print(f"Starting Taxonomic Voronoi visualization process...")
    print(f"Coloring by: {COLOR_BY}")
    print(f"Labeling by: {LABEL_BY}")
    print(f"Output will be saved to: {save_path}\n")

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
        path_to_best_gamma_and_eta_estimations = parent_folder / "min_energy_results.csv"
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

    visualizer = PipelineVisualizer()
    
    # --- 5. Generate Individual Plots in a Loop ---
    for metric in metrics_to_plot:
        print(f"\n--- Generating Voronoi plot for metric: {metric} ---")
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

            # --- Setup taxonomic coloring --- is done on df -> but should be done on best fit thingy. 
            if COLOR_BY and COLOR_BY in df_best_gamma_and_eta_estimates.columns: # df.columns:
                # Remove rows with missing taxonomic data
                # df = df.dropna(subset=[COLOR_BY])
                df_best_gamma_and_eta_estimates = df_best_gamma_and_eta_estimates.dropna(subset=[COLOR_BY])
                if df_best_gamma_and_eta_estimates.empty:
                    print(f"SKIPPING: No data after removing missing {COLOR_BY} values.")
                    continue
                    
                color_mapper = TaxonomicColorMapper(COLOR_BY, df_best_gamma_and_eta_estimates)
                print(f"Found {color_mapper.n_categories} unique {COLOR_BY} categories")
            else:
                print("SKIPPING: Cannot color by taxonomy - column not found")
                continue

            # --- Create and save the plot ---
            plot_title = visualizer._format_plot_title(metric_col_name)
            
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            figure_save_name = f"voronoi_landscape_{metric}_by_{COLOR_BY}.pdf"
            full_save_path = save_dir / figure_save_name

            fig, ax = plot_voronoi_with_taxonomic_colors(
                visualizer=visualizer,
                df=df,
                metric_name=metric_col_name,
                title=plot_title,
                color_mapper=color_mapper,
                label_by=LABEL_BY,
                df_best_estimates=df_best_gamma_and_eta_estimates if plot_indiv_connectomes else None,
                savepath=full_save_path, 
                normalize_aspect=True,
            )
            
            matplotlib.pyplot.close(fig)
            
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
            traceback.print_exc()

    print("\n--- Taxonomic visualization process completed. ---")