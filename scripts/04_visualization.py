"""
Using the visualization module for connectome analysis pipeline.
This script generates comparison landscape plots for a predefined list of metrics.
"""

import os
import traceback
from pathlib import Path
from src.visualization.energy_and_mc_landscape import visualize_gnm_results

# This will close all plots after saving, preventing them from displaying in a loop.
import matplotlib
matplotlib.use('Agg')

if __name__ == "__main__":
    
    # --- 1. Define Input and Output ---
    
    # List of CSV files containing the GNM sweep results.
    # The script will combine data from all these files.
    df_paths = [ 
        "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_123348/00_gnm_experiments_results.csv",
        "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_143109/00_gnm_experiments_results.csv", 
    ]
    
    # The folder where the final figures will be saved.
    # It will be created inside the same directory as the first data file.
    save_path = "/Users/adrian/Documents/01_projects/14_4D_lab/output/most_recent_figures"
    # save_path = Path(df_paths[0]).parent / output_dir_name
    
    # --- 2. Define Metrics to Plot ---
    
    # A list of all metrics for which a comparison plot will be generated.
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
        "mc_std"
    ]
    
    print(f"Starting visualization process...")
    print(f"Output will be saved to: {save_path}\n")
    
    # --- 3. Generate Plots in a Loop ---
    
    for metric in metrics_to_plot:
        print(f"--- Generating plot for metric: {metric} ---")
        try:
            # Define a consistent filename for each metric's comparison plot.
            # This ensures that running the script again will overwrite the old file.
            figure_save_name = f"comparison_landscape_{metric}"

            visualize_gnm_results(
                df_paths=df_paths,
                metric_to_visualize=metric,
                save_dir=save_path,
                save_name=figure_save_name,
                save_format="pdf", 
                # Set save_individual to False to only get the comparison plot
                save_individual=False, 
                show_dots=True,
            )
            print(f"Successfully generated plot for {metric}.\n")
            
        except ValueError as ve:
            # Handle cases where a metric might not be in the CSV file
            print(f"SKIPPING: Could not generate plot for '{metric}'. Reason: {ve}\n")
        except Exception as e:
            # Handle other potential errors during plotting
            print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
            traceback.print_exc()
            print("\n")

    print("--- Visualization process completed. ---")
