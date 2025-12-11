"""
Generate separate MaxCrit Voronoi plots for each taxonomic order.
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
# from adjustText import adjust_text

from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer


class TaxonomicColorMapper:
    """Helper class to manage color mapping for taxonomic categories."""
    
    def __init__(self, category_column, df):
        self.category_column = category_column
        self.unique_categories = sorted(df[category_column].dropna().unique())
        self.n_categories = len(self.unique_categories)
        
        if self.n_categories <= 10:
            self.colors = sns.color_palette("yellow", self.n_categories)
        elif self.n_categories <= 20:
            self.colors = sns.color_palette("tab20", self.n_categories)
        else:
            self.colors = sns.color_palette("husl", self.n_categories)
        
        self.color_map = dict(zip(self.unique_categories, self.colors))
    
    def get_color(self, category_value):
        return self.color_map.get(category_value, 'gray')
    
    def get_colors_for_df(self, df):
        return [self.get_color(val) for val in df[self.category_column]]
    
    def create_legend_elements(self):
        return [Patch(facecolor=color, label=cat) 
                for cat, color in self.color_map.items()]


def load_and_merge_taxonomic_data(gnm_results_df, animal_metadata_path):
    metadata_df = pd.read_csv(animal_metadata_path)
    
    if 'animal' in gnm_results_df.columns:
        merged_df = gnm_results_df.merge(metadata_df, on='animal', how='left')
        print(f"Merged {len(gnm_results_df)} rows with metadata. "
              f"Successful matches: {merged_df['family'].notna().sum()}")
    else:
        print("Warning: 'animal' column not found. Merging by index.")
        merged_df = pd.concat([gnm_results_df, metadata_df], axis=1)
    
    return merged_df


def plot_voronoi_for_order(visualizer, df, metric_name, title, 
                           order_name, df_best_estimates,
                           label_by, savepath, normalize_aspect=False):
    """Create Voronoi plot for a specific order."""
    
    # Filter data for this order
    df_order = df[df['order'] == order_name].copy()
    df_best_order = df_best_estimates[df_best_estimates['order'] == order_name].copy()
    
    if df_order.empty or df_best_order.empty:
        print(f"  Skipping {order_name}: No data available")
        return None, None
    
    # Create single-color mapper (all same color for this order)
    color = sns.color_palette("husl", 6)[0]
    
    # Normalize eta and gamma columns each (i.e.: new column) for both df_order_normalized and df_normalized. 
    df_normalized = df.copy()
    min_eta_val = df_normalized['eta'].min()
    eta_range = (df_normalized['eta'].max() - df_normalized['eta'].min())
    
    min_gamma_val = df_normalized['gamma'].min()
    gamma_range = (df_normalized['gamma'].max() - df_normalized['gamma'].min())
    
    df_normalized['eta'] = (df_normalized['eta'] - min_eta_val) / eta_range
    df_normalized['gamma'] = (df_normalized['gamma'] - min_gamma_val) / gamma_range
    
    df_order_normalized = df_order.copy()
    df_order_normalized['eta'] = (df_order_normalized['eta'] - min_eta_val) / eta_range
    df_order_normalized['gamma'] = (df_order_normalized['gamma'] - min_gamma_val) / gamma_range

    df_best_order_normalized = df_best_order.copy()
    df_best_order_normalized['eta'] = (df_best_order_normalized['eta'] - min_eta_val) / eta_range
    df_best_order_normalized['gamma'] = (df_best_order_normalized['gamma'] - min_gamma_val) / gamma_range

    # Create base Voronoi landscape
    fig, ax = visualizer.plot_metric_landscape_voronoi(
        df_normalized, # df, # df_order, 
        title=f"{title} - {order_name}",
        metric_name=metric_name,
        savepath=None,
        dot_color="steelblue", 
        show=False,
        show_dots=False,
        annotate_extremes=True, 
        estimated_indiv_connectomes=None,
        show_colorbar=True,
        normalize_aspect=normalize_aspect,
    )
    
    # Add scatter points for best estimates
    ax.scatter(
        df_best_order_normalized["eta"], # df_best_order["eta"],
        df_best_order_normalized["gamma"], 
        facecolors="blue", # color,
        s=40,
        edgecolors="black", 
        linewidths=0.5,
        marker='o',
        label=order_name,
        zorder=15,
        alpha=0.8
    )
    
    
    # # Rename x and y ticks to original eta and gamma values
    # eta_ticks = ax.get_xticks()
    # gamma_ticks = ax.get_yticks()
    # eta_tick_labels = [f"{df['eta'].min() + tick * (df['eta'].max() - df['eta'].min()):.2f}" for tick in eta_ticks]
    # gamma_tick_labels = [f"{df['gamma'].min() + tick * (df['gamma'].max() - df['gamma'].min()):.2f}" for tick in gamma_ticks]
    # ax.set_xticklabels(eta_tick_labels)
    # ax.set_yticklabels(gamma_tick_labels)
    
    ##################################################################
    # Rename x and y ticks to original eta and gamma values (handle both normalized 0-1 axes and original scale)
    eta_ticks = ax.get_xticks()
    gamma_ticks = ax.get_yticks()

    # original data ranges
    eta_min, eta_max = df['eta'].min(), df['eta'].max()
    gamma_min, gamma_max = df['gamma'].min(), df['gamma'].max()

    # axis limits to detect whether the plot is in normalized (0-1) coordinates
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    def _map_tick(tick, lim_min, lim_max, data_min, data_max):
        # if axis limits are (approximately) 0..1 assume ticks are normalized and map back to data range
        if np.isclose(lim_max - lim_min, 1.0) and np.isclose(lim_min, 0.0):
            return data_min + (tick - lim_min) * (data_max - data_min)
        # otherwise assume ticks are already in original data units
        return tick

    eta_tick_labels = []
    for t in eta_ticks:
        mapped = _map_tick(t, xlim[0], xlim[1], eta_min, eta_max)
        if np.isnan(mapped):
            eta_tick_labels.append("")
        else:
            eta_tick_labels.append(f"{mapped:.2f}")

    gamma_tick_labels = []
    for t in gamma_ticks:
        mapped = _map_tick(t, ylim[0], ylim[1], gamma_min, gamma_max)
        if np.isnan(mapped):
            gamma_tick_labels.append("")
        else:
            gamma_tick_labels.append(f"{mapped:.2f}")

    # set locators explicitly, then apply formatted labels (avoids FixedFormatter/FixedLocator warnings)
    ax.set_xticks(eta_ticks)
    ax.set_yticks(gamma_ticks)
    ax.set_xticklabels(eta_tick_labels)
    ax.set_yticklabels(gamma_tick_labels)
    ##################################################################
    
    
    
    # Add labels with non-overlapping arrangement
    if label_by and label_by in df_best_order.columns:
        texts = []
        for index, row in df_best_order_normalized.iterrows():
            label = str(row[label_by])
            text = ax.annotate(
                label, 
                (row['eta'], row['gamma']), 
                fontsize=6, 
                alpha=0.9,
                xytext=(5, 5), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         alpha=0.8, edgecolor='gray', linewidth=0.5),
                zorder=20
            )
            texts.append(text)
        
        # Try to adjust text positions to avoid overlap
        # try:
        #     from adjustText import adjust_text
        #     adjust_text(texts, 
        #                ax=ax,
        #                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
        #                expand_points=(1.2, 1.2),
        #                force_text=(0.5, 0.5),
        #                force_points=(0.2, 0.2))
        # except ImportError:
        #     print("  Note: Install adjustText for better label positioning: pip install adjustText")
    
    # Add legend
    legend = ax.legend(
        title="Order",
        loc='lower left',
        fontsize=8,
        title_fontsize=9
    )
    
    plt.tight_layout()
    
    # Save
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {savepath}")
    
    return fig, ax


if __name__ == "__main__":
    
    # --- 1. Input Paths ---
    df_paths = [
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/all_metrics_for_60_generally_finer_search_animal_0.csv"
        "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/70_mix_and_match_animal_0/all_metrics_for_70_mix_and_match_animal_0.csv"
    ]
    
    animal_metadata_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50.csv"
    
    parent_folder = Path(df_paths[0]).parent
    save_path = parent_folder / "figures_maxcrit_by_order_uniform_voroni_cells"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # --- 2. Settings ---
    LABEL_BY = "name"
    metric = "MaxCriteria"
    
    print(f"Generating MaxCrit plots by order...")
    print(f"Output folder: {save_path}\n")

    # --- 3. Load Data ---
    try:
        if len(df_paths) > 1:
            gnm_results_raw_df = generate_entire_df(df_paths)
        else:
            gnm_results_raw_df = pd.read_csv(df_paths[0], index_col=False)

        print(f"Original data shape: {gnm_results_raw_df.shape}")
        gnm_results_raw_df = load_and_merge_taxonomic_data(gnm_results_raw_df, animal_metadata_path)
        print(f"After merging: {gnm_results_raw_df.shape}")
        
        # Load best estimates
        path_to_best = parent_folder / "min_energy_results.csv"
        if not os.path.exists(path_to_best):
            print(f"ERROR: Best estimates file not found: {path_to_best}")
            exit()
            
        df_best_estimates = pd.read_csv(path_to_best)
        df_best_estimates = load_and_merge_taxonomic_data(df_best_estimates, animal_metadata_path)
        print(f"Loaded {len(df_best_estimates)} best estimates\n")
                
    except Exception as e:
        print(f"ERROR: Failed to load data. {e}")
        traceback.print_exc()
        exit()

    # --- 4. Prepare Data ---
    visualizer = PipelineVisualizer()
    
    df = gnm_results_raw_df.copy()
    df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
    df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
    
    # Find metric column
    try:
        metric_col_name = next(col for col in df.columns if metric in col)
    except StopIteration:
        print(f"ERROR: Metric '{metric}' not found in DataFrame")
        exit()

    df[metric_col_name] = pd.to_numeric(df[metric_col_name], errors='coerce')
    df = df.dropna(subset=['eta', 'gamma', metric_col_name, 'order'])
    df_best_estimates = df_best_estimates.dropna(subset=['order'])
    
    if df.empty:
        print("ERROR: No valid data after cleaning")
        exit()

    # --- 5. Generate Plot for Each Order ---
    orders = sorted(df_best_estimates['order'].unique())
    print(f"Found {len(orders)} orders: {orders}\n")
    
    plot_title = visualizer._format_plot_title(metric_col_name)
    
    for order_name in orders:
        print(f"Processing: {order_name}")
        try:
            figure_save_name = f"maxcrit_voronoi_{order_name.replace(' ', '_').replace('/', '_')}.pdf"
            full_save_path = save_path / figure_save_name

            fig, ax = plot_voronoi_for_order(
                visualizer=visualizer,
                df=gnm_results_raw_df, 
                metric_name=metric_col_name,
                title=plot_title,
                order_name=order_name,
                df_best_estimates=df_best_estimates,
                label_by=LABEL_BY,
                savepath=full_save_path, 
                normalize_aspect=True,
            )
            
            if fig is not None:
                matplotlib.pyplot.close(fig)
            
        except Exception as e:
            print(f"  ERROR processing {order_name}:")
            traceback.print_exc()
            print()

    print(f"\n✓ Completed! Plots saved to: {save_path}")