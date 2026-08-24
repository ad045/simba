import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.patches as patches
from typing import Literal, List

import os 
# def normalize_color_channels(df: pd.DataFrame) -> pd.DataFrame:
#     """Normalizes data columns into 'r', 'g', 'b' channels."""
#     def _normalize(column):
#         if column.max() == column.min(): return np.zeros_like(column)
#         return (column - column.min()) / (column.max() - column.min())

#     df_norm = df.copy()
#     df_norm['r'] = _normalize(df_norm['mc_mean'])
#     df_norm['g'] = _normalize(df_norm['avg_communicability'])
#     df_norm['b'] = _normalize(df_norm['wiring_cost'])
#     return df_norm

def normalize_color_channels(df: pd.DataFrame, 
                             metrics_to_analyze: List[str], 
                             ) -> pd.DataFrame:
    """Normalizes data columns into 'c', 'm', 'y', 'k' channels."""
    def _normalize(column):
        if column.max() == column.min(): return np.zeros_like(column)
        return (column - column.min()) / (column.max() - column.min())

    df_norm = df.copy()
    # Map data to CMYK channels directly
    df_norm['c'] = _normalize(df_norm[metrics_to_analyze[0]]) # 'mc_mean'])
    df_norm['m'] = _normalize(df_norm[metrics_to_analyze[1]]) # 'avg_communicability'])
    df_norm['y'] = _normalize(df_norm[metrics_to_analyze[2]]) # 'wiring_cost'])
    df_norm['k'] = _normalize(df_norm[metrics_to_analyze[3]]) # 'mc_5']) 
    return df_norm

# def _plot_voronoi(ax: plt.Axes, df: pd.DataFrame, channel: str):
#     """Helper function to draw a Voronoi plot on a given axis."""
#     points = df[['eta', 'gamma']].values
#     if points.shape[0] < 4:
#         ax.text(0.5, 0.5, 'Not enough points for Voronoi', ha='center', va='center', transform=ax.transAxes)
#         return

#     vor = Voronoi(points)
#     for i, region_idx in enumerate(vor.point_region):
#         region = vor.regions[region_idx]
#         if -1 not in region and region:
#             polygon = patches.Polygon([vor.vertices[v] for v in region], edgecolor='none')
            
#             r, g, b = df['r'].iloc[i], df['g'].iloc[i], df['b'].iloc[i]
#             color = (0,0,0) # Default to black
            
#             if channel == 'red': color = (1, 1-r, 1-r) 
#             elif channel == 'yellow': color = (1, 1, 1-g) # g, g, 0)
#             elif channel == 'blue': color = (1-b, 1-b, 1) # (0, 0, b)
#             elif channel == 'rgb': color = (r, g, b)
#             elif channel == 'cyan': color = (0, 1 - r, 1 - r) # max cyan: 0, 255, 255 -> CYMK: 100%, 0%, 0%, 0%
#             elif channel == 'magenta': color = (1 - g, 0, 1 - g) # max magenta: 255, 0, 255
#             elif channel == 'yellow_cmy': color = (1 - b, 1 - b, 0)
#             elif channel == 'cmyk': # NEW: CMYK composite simulation
#                 c, m, y = 1 - r, 1 - g, 1 - b
#                 k = min(c, m, y) # Calculate black component
#                 # Convert CMYK to RGB for screen display
#                 final_r = (1 - c) * (1 - k)
#                 final_g = (1 - m) * (1 - k)
#                 final_b = (1 - y) * (1 - k)
#                 color = (final_r, final_g, final_b)

#             polygon.set_facecolor(color)
#             ax.add_patch(polygon)
    
#     ax.set_xlim(df['eta'].min(), df['eta'].max())
#     ax.set_ylim(df['gamma'].min(), df['gamma'].max())


def _plot_voronoi(ax: plt.Axes, df: pd.DataFrame, channel: str):
    """Helper function to draw a Voronoi plot on a given axis."""
    points = df[['eta', 'gamma']].values
    if points.shape[0] < 4:
        ax.text(0.5, 0.5, 'Not enough points for Voronoi', ha='center', va='center', transform=ax.transAxes)
        return

    vor = Voronoi(points)
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and region:
            polygon = patches.Polygon([vor.vertices[v] for v in region], edgecolor='none')

            # Get CMYK values for the point
            c, m, y, k = df[['c', 'm', 'y', 'k']].iloc[i]

            # Determine color based on the requested channel
            color = (1, 1, 1) # Default to white
            if channel == 'cyan':
                color = (1 - c, 1, 1)      # White minus Red
            elif channel == 'magenta':
                color = (1, 1 - m, 1)      # White minus Green
            elif channel == 'yellow':
                color = (1, 1, 1 - y)      # White minus Blue
            elif channel == 'black':
                color = (1 - k, 1 - k, 1 - k)  # White minus all colors
            elif channel == 'cmyk':
                # Combine CMYK channels for display on an RGB screen
                r = (1 - c) * (1 - k)
                g = (1 - m) * (1 - k)
                b = (1 - y) * (1 - k)
                color = (r, g, b)

            polygon.set_facecolor(color)
            ax.add_patch(polygon)
    
    ax.set_xlim(df['eta'].min(), df['eta'].max())
    ax.set_ylim(df['gamma'].min(), df['gamma'].max())
    
    

def create_cmyk_plots(df: pd.DataFrame, 
                      metrics_to_analyze: List[str], 
                      style: Literal['voronoi'] = 'voronoi', 
                      plot_empirical_connectomes: bool = False, 
                      save_folder = None):
    """Generates and saves the CMYK separation plots in a 2x3 grid."""
    plot_func = _plot_voronoi

    # Add the 'black' channel to the list
    channels = ['cyan', 'magenta', 'yellow', 'black', 'cmyk']
    title_mapping = {
        "mc_mean": "MC Mean", 
        "wiring_cost": "Wiring Cost", 
        "mc_5": "MC_5", 
        "avg_communicability": "Communicability", 
        "richclub_avg_length": "Rich-Club Average Length", 
        "modularity": "Modularity", 
        'Full CMYK Composite': 'Full CMYK Composite'
    }
    
    # titles = [
    #     'Cyan Channel (MC Mean)', 'Magenta Channel (Communicability)',
    #     'Yellow Channel (Wiring Cost)', 'Black Channel (MC 5)',
    #     'Full CMYK Composite'
    # ]
    # titles = [] 
    # for (channel, metric) in zip(channels, metrics_to_analyze + ['Full CMYK Composite']): 
    #     titles.append(channel.capitalize() + f" Channel\n({title_mapping[metric]}))")
    # print(titles)
    
    titles = [] 
    for (channel, metric) in zip(channels, metrics_to_analyze + ['Full CMYK Composite']):
        if channel == 'cmyk':
            titles.append(f"All Channels \n (Full CMYK Composite)") # " + channel.capitalize() + " Channel)")
        else: 
            titles.append(f"{title_mapping[metric]}\n (" + channel.capitalize() + " Channel)")
    print(titles)
    
    
    # Change grid to 2x3 to accommodate the 5 plots
    # fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True)
    # fig.suptitle(f'CMYK Color Separation ({style.capitalize()} Style)', fontsize=16)
    fig.suptitle(f'Relating MC, Wiring Costs, Rich-Club Lengths, and Modularity\n Visualization as CMYK Color Separation', fontsize=16)
    
    axes_flat = axes.flatten()
    for i, (channel, title) in enumerate(zip(channels, titles)):
        plot_func(axes_flat[i], df, channel)
        axes_flat[i].set_title(title)
        
    
    if plot_empirical_connectomes: # TODO: This is not clean yet!! 
        path_to_best_gamma_and_eta_estimations = save_folder.parent.parent / "min_energy_results.csv"    # not entirely clean... TODO: Make this clean. 
        print(path_to_best_gamma_and_eta_estimations)
        # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/26_testing_4_KS_folders_why_so_fast copy/summary_all_metrics_for_exp_26_testing_4_KS_folders_why_so_fast.csv
        if os.path.exists(path_to_best_gamma_and_eta_estimations):
            df_best_gamma_and_eta_estimates = pd.read_csv(path_to_best_gamma_and_eta_estimations)
            print(df_best_gamma_and_eta_estimates.head)
        else: 
            print("Error: Missing 'min_energy_results.csv' file.")
        
        axes_flat[-2].scatter(df_best_gamma_and_eta_estimates["eta"], df_best_gamma_and_eta_estimates["gamma"], color="white", 
                              linewidths=0.5, alpha=0.25, edgecolors="black")
    
    # Turn off the last unused subplot
    axes_flat[-1].axis('off')

    # Add shared axis labels
    fig.text(0.5, 0.04, 'eta', ha='center')
    fig.text(0.04, 0.5, 'gamma', va='center', rotation='vertical')

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    
    if save_folder: 
        filename = f'plot_{style}_cmyk_separation_with_black_switched_colors.pdf'
        plt.savefig(save_folder / filename)
        plt.close()
        print(f"Plot saved as {save_folder / filename}")
    else: 
        plt.show()
    

def main():
    """Main function to run the plotting script."""
    from pathlib import Path
    data_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/26_testing_4_KS_folders_why_so_fast/summary_all_metrics_for_exp_26_testing_4_KS_folders_why_so_fast.csv"
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/2423_24rough_combined/summary_all_metrics_for_exp_2423_24rough_combined.csv"
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/20_sweep_with_individual_connectomes_eta_-7_and_gamma_-0.2/summary_all_metrics_for_exp_20_sweep_with_individual_connectomes_eta_-7_and_gamma_-0.2.csv"
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/18_sweep_with_individual_connectomes_larger_eta_span/summary_all_metrics_for_exp_18_sweep_with_individual_connectomes_larger_eta_span.csv" 
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes/16_big_sweep_with_individual_connectomes_results.csv"
    save_folder = Path(data_path).parent / "figures/rgb_pictures"
    save_folder.mkdir(parents=True, exist_ok=True)
    
    metrics_to_analyze = ["wiring_cost", "mc_mean", "avg_communicability", "richclub_avg_length"]  # modularity"]]
    
    # DRAFT: ONLY 500 SAMPLES FOR SPEED 
    # raw_df = pd.read_csv(data_path)[:500][["eta", "gamma"] + metrics_to_analyze]
    
    # FULL SWEEP 
    raw_df = pd.read_csv(data_path)[["eta", "gamma"] + metrics_to_analyze]
    df = normalize_color_channels(raw_df, metrics_to_analyze)
    
    # Generate the CMYK separation plot
    create_cmyk_plots(df, metrics_to_analyze, style='voronoi',
                      plot_empirical_connectomes=True,  
                      save_folder=save_folder)

if __name__ == "__main__":
    main()
