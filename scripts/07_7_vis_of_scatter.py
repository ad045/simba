
"""
Using the visualization module for connectome analysis pipeline.
This script generates only the Voronoi landscape plot for a predefined list of metrics.
"""

import os
import traceback
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer

if __name__ == "__main__":
    
    # --- 1. Define Input and Output ---
    
    animal_id = 0
    # List of CSV files containing the GNM sweep results.
    df_paths = [ 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/53_suarez_MaMI_100_testing_with_100_sample_points_animal_19/all_metrics_for_53_suarez_MaMI_100_testing_with_100_sample_points_animal_19.csv"
        
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/62_testing_conciser_code_animal_0/all_metrics_for_62_testing_conciser_code_animal_0.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/63_fine_grid_animal_0/all_metrics_for_63_fine_grid_animal_0.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/65_grid_higher_eta_animal_1/all_metrics_for_65_grid_higher_eta_animal_1.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/70_mix_and_match_animal_0/all_metrics_for_70_mix_and_match_animal_0.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/64_fine_grid_upper_local_minima_animal_0/all_metrics_for_64_fine_grid_upper_local_minima_animal_0.csv"""
        "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/63_fine_grid_animal_0/all_metrics_for_63_fine_grid_more_metrics_wow.csv"
    ]       
    plot_indiv_connectomes = False # True # False # True # False # True #  False # True # False
    
    
    ##############
    
    
    # The folder where the final figures will be saved.
    # A new subfolder is used to keep these plots separate.
    parent_folder = Path(df_paths[0]).parent
    save_path = parent_folder / "scatter"
    os.makedirs(save_path, exist_ok=True)
    
    
    # --- 2. Define Metrics to Plot ---
    
    lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 20, 49]
    metrics_to_plot = [
        # "scatter_mc_mean_and_communicability",
        # "scatter_mc_mean_and_glob_efficiency"
        "n_connected_components"
    ]

    plot_combined_lag_plot = False # Plot this comparison plot with different MC lags 
    
    print(f"Starting Voronoi-only visualization process...")
    print(f"Output will be saved to: {save_path}\n")

    # --- 3. Load Data ---
    print("Loading and combining data...")

    if not df_paths:
        raise ValueError("Input 'df_paths' is an empty list.")
    
    if len(df_paths) > 1:
        gnm_results_df = generate_entire_df(df_paths)
    else:
        gnm_results_df = pd.read_csv(df_paths[0], index_col=False)

    if gnm_results_df.empty:
        raise ValueError("Dataframe is empty after loading.")

    path_to_best_gamma_and_eta_estimations = parent_folder / "min_energy_results.csv"    # not entirely clean... TODO: Make this clean. 
    if os.path.exists(path_to_best_gamma_and_eta_estimations) and plot_indiv_connectomes:
        df_best_gamma_and_eta_estimates = pd.read_csv(path_to_best_gamma_and_eta_estimations)
            
            
    # --- 4. Generate Individual Plots in a Loop ---
    for metric in metrics_to_plot:
        print(f"--- Generating Voronoi plot for metric: {metric} ---")
        # --- Prepare data for the specific metric ---
        df = gnm_results_df.copy()

        if metric == "scatter_mc_mean_and_communicability": # prop_efficiency
            plt.scatter(
                pd.to_numeric(df["avg_communicability"], errors='coerce'),
                pd.to_numeric(df["mc_mean"], errors='coerce'),
                alpha=0.2,
                s=5
            )

        if metric == "n_connected_components": 
            plt.hist(
                pd.to_numeric(df["n_connected_components"], errors='coerce'), 
                bins=60, 
            )

            plt.xlabel("N connected components")
            # plt.ylabel("MC Mean")
            # plt.title("Scatter plot of Avg Communicability")
            plt.grid()
            plt.savefig(save_path / f"scatter_n_connected_components.pdf")
            plt.close()
            print(f"Path: {save_path / f"scatter_n_connected_components.pdf"}")

        if metric == "scatter_mc_mean_and_glob_efficiency": # prop_efficiency
            plt.scatter(
                pd.to_numeric(df["global_efficiency"], errors='coerce'),
                pd.to_numeric(df["mc_mean"], errors='coerce'),
                alpha=0.2,
                s=5
            )

            plt.xlabel("Global Efficiency")
            plt.ylabel("MC Mean")
            # plt.title("Scatter plot of MC Mean vs Global Efficiency")
            plt.grid()
            plt.savefig(save_path / f"scatter_mc_mean_and_global_efficiency.pdf")
            plt.close()
            print(f"Path: {save_path / f"scatter_mc_mean_and_global_efficiency.pdf"}")



        # if metric == "normalized_mc_mean_times_global_efficiency": # prop_efficiency
        #     normalized_mc_mean = pd.to_numeric(df["mc_mean"], errors='coerce')
        #     normalized_mc_mean = (normalized_mc_mean - normalized_mc_mean.min()) / (normalized_mc_mean.max() - normalized_mc_mean.min())
            
        #     normalized_prop_efficiency = pd.to_numeric(df["avg_communicability"], errors='coerce')
        #     normalized_prop_efficiency = (normalized_prop_efficiency - normalized_prop_efficiency.min()) / (normalized_prop_efficiency.max() - normalized_prop_efficiency.min())
        #     df["normalized_mc_mean_times_global_efficiency"] = normalized_mc_mean * normalized_prop_efficiency



    print("--- Visualization process completed. ---")

