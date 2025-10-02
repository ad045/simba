
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
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

if __name__ == "__main__":
    
    # --- 1. Define Input and Output ---
    
    # List of CSV files containing the GNM sweep results.
    df_paths = [ 
        # 12: Medium sweep (250) with matching index and communicability
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/12_average_connectome_matching_index_and_communicability/combined_results12_average_connectome_matching_index_and_communicability_summary.csv", 
        
        # 14: Big sweep, density 10 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/14_average_connectome_matching_index_and_communicability_density10/combined_results14_average_connectome_matching_index_and_communicability_density10_summary.csv", 

        # 15 included evaluation of empirical connectomes 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/15_individual_connectomes/15_individual_connectomes_20250924_142952/15_individual_connectomes_temp/15_individual_connectomes_results.csv", 
        
        # 16: With individual connectomes
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes_results.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes/16_big_sweep_with_individual_connectomes_results.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/17_bigger_connectomes_no_individuals_density_10_8/summary_all_metrics_for_exp_17_bigger_connectomes_no_individuals_density_10_8.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/18_sweep_with_individual_connectomes_larger_eta_span/summary_all_metrics_for_exp_18_sweep_with_individual_connectomes_larger_eta_span.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/20_sweep_with_individual_connectomes_eta_-7_and_gamma_-0.2/summary_all_metrics_for_exp_20_sweep_with_individual_connectomes_eta_-7_and_gamma_-0.2.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/20_sweep_with_individual_connectomes_eta_-7_and_gamma_-0.2/summary_all_metrics_for_exp_20_sweep_with_individual_connectomes_eta_-7_and_gamma_-0.2.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/2423_combined/summary_all_metrics_for_exp_2423_combined.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/2423_24rough_combined/summary_all_metrics_for_exp_2423_24rough_combined.csv"
        
        # NEW CONSENSUS! 
        "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/26_testing_4_KS_folders_why_so_fast/summary_all_metrics_for_exp_26_testing_4_KS_folders_why_so_fast.csv"

    ]       
    
    plot_indiv_connectomes = False # True # False
    
    
    ##############
    
    
    # The folder where the final figures will be saved.
    # A new subfolder is used to keep these plots separate.
    save_path = Path(df_paths[0]).parent / "figures_voronoi_only"
    
    
    # --- 2. Define Metrics to Plot ---
    
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

        path_to_best_gamma_and_eta_estimations = save_path.parent / "min_energy_results.csv"    # not entirely clean... TODO: Make this clean. 
        if os.path.exists(path_to_best_gamma_and_eta_estimations):
            df_best_gamma_and_eta_estimates = pd.read_csv(path_to_best_gamma_and_eta_estimations)
                
    except Exception as e:
        print(f"ERROR: Failed to load data. {e}")
        exit() # Exit if data cannot be loaded

    visualizer = PipelineVisualizer() # Initialize the visualizer once
    
    # --- 4. Generate Individual Plots in a Loop ---
    for metric in metrics_to_plot:
        print(f"--- Generating Voronoi plot for metric: {metric} ---")
        try:
            # --- Prepare data for the specific metric ---
            df = gnm_results_df.copy()

            if metric == "mean_mc_divided_by_wiring_cost": 
                df["mean_mc_divided_by_wiring_cost"] = pd.to_numeric(df["mc_mean"], errors='coerce') / pd.to_numeric(df["wiring_cost"], errors='coerce')
            
                        
            if metric == "mc_5_divided_by_wiring_cost": 
                df["mc_5_divided_by_wiring_cost"] = pd.to_numeric(df["mc_5"], errors='coerce') / pd.to_numeric(df["wiring_cost"], errors='coerce')
            
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
                dot_color="steelblue", 
                show=False,
                show_dots=False,
                annotate_extremes=True, 
                estimated_indiv_connectomes=df_best_gamma_and_eta_estimates if plot_indiv_connectomes else None, 
            )
            
            matplotlib.pyplot.close() # Close the plot after saving to free memory, if returned fig and ax are not used.
            
            
            print(f"Successfully generated plot for {metric} at {full_save_path}\n")
            
            
                # plt.show() 
            
            
            # matplotlib.pyplot.close() # Close the plot after saving to free memory. 
            
            
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
            traceback.print_exc()
            print("\n")


    # --- 5. Generate Combined MC Lag Plot ---
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
            matplotlib.pyplot.close() # Close the plot after saving to free memory, if generated figure is not used. 

            print("Successfully generated combined MC lag plot.\n")
        else:
            print("SKIPPING: No 'mc_lag' columns found to generate a combined plot.\n")

    except Exception as e:
        print(f"ERROR: An unexpected error occurred while plotting the combined MC lags.")
        traceback.print_exc()
        print("\n")


    print("--- Visualization process completed. ---")

