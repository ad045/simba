# """
# Using the visualization module for connectome analysis pipeline.
# This script generates only the Voronoi landscape plot for a predefined list of metrics.
# """

# import os
# import traceback
# from pathlib import Path
# import pandas as pd
# from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer

# # This will close all plots after saving, preventing them from displaying in a loop.
# import matplotlib
# matplotlib.use('Agg')

# if __name__ == "__main__":
    
#     # --- 1. Define Input and Output ---
    
#     # List of CSV files containing the GNM sweep results.
#     df_paths = [ 
#         # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/09_bigger_sweep_matching_index/combined_results09_bigger_sweep_matching_index_summary.csv"
        
#         # 12: Medium sweep (250) with matching index and communicability
#         "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/12_average_connectome_matching_index_and_communicability/combined_results12_average_connectome_matching_index_and_communicability_summary.csv", 
        
#     ]
    
#     # The folder where the final figures will be saved.
#     # A new subfolder is used to keep these plots separate.
#     save_path = Path(df_paths[0]).parent / "figures_voronoi_only"
    
    
#     # --- 2. Define Metrics to Plot ---
    
#     lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 20, 50]
#     metrics_to_plot = [
#         "MaxCriteria",
#         "avg_communicability", 
#         "global_efficiency", 
#         "modularity", 
#         "avg_clustering", 
#         "avg_degree", 
#         "transitivity", 
#         "avg_edge_distance", 
#         "char_path_length", 
#         "richclub_n_edges", 
#         "richclub_avg_length", 
#         "mc_mean", 
#         "mc_std", 
#         "wiring_cost", 
#         "mean_mc_divided_by_wiring_cost",
#     ] + [f"mc_{lag}" for lag in lags_to_plot]
    
    
#     print(f"Starting Voronoi-only visualization process...")
#     print(f"Output will be saved to: {save_path}\n")

#     # --- 3. Load Data ---
#     print("Loading and combining data...")
#     try:
#         if not df_paths:
#             raise ValueError("Input 'df_paths' is an empty list.")
        
#         if len(df_paths) > 1:
#             gnm_results_df = generate_entire_df(df_paths)
#         else:
#             gnm_results_df = pd.read_csv(df_paths[0], index_col=False)

#         if gnm_results_df.empty:
#             raise ValueError("Dataframe is empty after loading.")

#     except Exception as e:
#         print(f"ERROR: Failed to load data. {e}")
#         exit() # Exit if data cannot be loaded

#     # --- 4. Generate Plots in a Loop ---
    
#     visualizer = PipelineVisualizer() # Initialize the visualizer once

#     for metric in metrics_to_plot:
#         print(f"--- Generating Voronoi plot for metric: {metric} ---")
#         try:
#             # --- Prepare data for the specific metric ---
#             df = gnm_results_df.copy()

#             if metric == "mean_mc_divided_by_wiring_cost": 
#                 df["mean_mc_divided_by_wiring_cost"] = pd.to_numeric(df["mc_mean"], errors='coerce') / pd.to_numeric(df["wiring_cost"], errors='coerce')
            
#             df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
#             df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
            
#             try:
#                 metric_col_name = next(col for col in df.columns if metric in col)
#             except StopIteration:
#                 print(f"SKIPPING: Metric '{metric}' not found in DataFrame columns.\n")
#                 continue

#             df[metric_col_name] = pd.to_numeric(df[metric_col_name], errors='coerce')
#             df = df.dropna(subset=['eta', 'gamma', metric_col_name])
            
#             if df.empty:
#                 print(f"SKIPPING: No valid data for '{metric}' after cleaning.\n")
#                 continue

#             # --- Create and save the plot ---
#             plot_title = visualizer._format_plot_title(metric_col_name)
            
#             save_dir = Path(save_path)
#             save_dir.mkdir(parents=True, exist_ok=True)
#             figure_save_name = f"voronoi_landscape_{metric}.pdf"
#             full_save_path = save_dir / figure_save_name

#             visualizer.plot_metric_landscape_voronoi(
#                 df, 
#                 title=plot_title,
#                 metric_name=metric_col_name,
#                 savepath=full_save_path,
#                 show=False,
#                 show_dots=False,
#                 annotate_extremes=True
#             )
#             print(f"Successfully generated plot for {metric} at {full_save_path}\n")
            
#         except Exception as e:
#             print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
#             traceback.print_exc()
#             print("\n")

#     print("--- Visualization process completed. ---")


"""
Using the visualization module for connectome analysis pipeline.
This script generates only the Voronoi landscape plot for a predefined list of metrics.
"""

import os
import traceback
from pathlib import Path
import pandas as pd
from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer

# This will close all plots after saving, preventing them from displaying in a loop.
import matplotlib
# matplotlib.use('Agg')

if __name__ == "__main__":
    
    # --- 1. Define Input and Output ---
    
    # List of CSV files containing the GNM sweep results.
    df_paths = [ 
        # 12: Medium sweep (250) with matching index and communicability
        "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/12_average_connectome_matching_index_and_communicability/combined_results12_average_connectome_matching_index_and_communicability_summary.csv", 
        
    ]
    
    # The folder where the final figures will be saved.
    # A new subfolder is used to keep these plots separate.
    save_path = Path(df_paths[0]).parent / "figures_voronoi_only"
    
    
    # --- 2. Define Metrics to Plot ---
    
    lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 20, 50]
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
    ] + [f"mc_{lag}" for lag in lags_to_plot]
    
    
    print(f"Starting Voronoi-only visualization process...")
    print(f"Output will be saved to: {save_path}\n")

    # --- 3. Load Data ---
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
        exit() # Exit if data cannot be loaded

    visualizer = PipelineVisualizer() # Initialize the visualizer once
    
    # --- 4. Generate Combined MC Lag Plot ---
    print("--- Generating combined Voronoi plot for all MC lags ---")
    try:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Define a base name for the save file, the function will add the extension
        figure_save_name = "voronoi_landscape_mc_lags_grid"
        full_save_path = save_dir / figure_save_name

        mc_cols_exist = any(f"mc_{lag}" in gnm_results_df.columns for lag in lags_to_plot)

        if mc_cols_exist:
            visualizer.plot_mc_lag_landscapes_voronoi(
                gnm_results_df,
                lags=lags_to_plot,
                n_cols=4, # As requested: 4 columns
                savepath=full_save_path,
                save_format="pdf"
            )
            print("Successfully generated combined MC lag plot.\n")
        else:
            print("SKIPPING: No 'mc_lag' columns found to generate a combined plot.\n")

    except Exception as e:
        print(f"ERROR: An unexpected error occurred while plotting the combined MC lags.")
        traceback.print_exc()
        print("\n")

    # --- 5. Generate Individual Plots in a Loop ---
    for metric in metrics_to_plot:
        print(f"--- Generating Voronoi plot for metric: {metric} ---")
        try:
            # --- Prepare data for the specific metric ---
            df = gnm_results_df.copy()

            if metric == "mean_mc_divided_by_wiring_cost": 
                df["mean_mc_divided_by_wiring_cost"] = pd.to_numeric(df["mc_mean"], errors='coerce') / pd.to_numeric(df["wiring_cost"], errors='coerce')
            
            df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
            df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
            
            try:
                # Exact match for mc_lag columns
                if metric.startswith("mc_") and metric.split("_")[1].isdigit():
                    metric_col_name = metric
                    if metric not in df.columns:
                        raise StopIteration
                else: # Flexible match for other metrics
                    metric_col_name = next(col for col in df.columns if metric in col)
            except StopIteration:
                print(f"SKIPPING: Metric '{metric}' not found in DataFrame columns.\n")
                continue

            df[metric_col_name] = pd.to_numeric(df[metric_col_name], errors='coerce')
            df = df.dropna(subset=['eta', 'gamma', metric_col_name])
            
            if df.empty:
                print(f"SKIPPING: No valid data for '{metric}' after cleaning.\n")
                continue

            # --- Create and save the plot ---
            plot_title = visualizer._format_plot_title(metric_col_name)
            
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            figure_save_name = f"voronoi_landscape_{metric}.pdf"
            full_save_path = save_dir / figure_save_name

            fig, ax = visualizer.plot_metric_landscape_voronoi(
                df, 
                title=plot_title,
                metric_name=metric_col_name,
                savepath=full_save_path,
                show=False,
                show_dots=False,
                annotate_extremes=True
            )
            print(f"Successfully generated plot for {metric} at {full_save_path}\n")
            
            matplotlib.pyplot.close() # Close the plot after saving to free memory. 
            
            
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
            traceback.print_exc()
            print("\n")

    print("--- Visualization process completed. ---")

