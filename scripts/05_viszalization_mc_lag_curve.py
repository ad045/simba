"""
Using the visualization module for connectome analysis pipeline.
This script generates comparison landscape plots for a predefined list of metrics.
"""

import os
import traceback
from pathlib import Path
import pandas as pd
from src.visualization.energy_and_mc_landscape import visualize_gnm_results, generate_entire_df, PipelineVisualizer

# This will close all plots after saving, preventing them from displaying in a loop.
import matplotlib
# matplotlib.use('Agg')

if __name__ == "__main__":
    
    # --- Configuration Flags ---
    # Set to True to generate the plot of average MC vs. lag
    PLOT_AVERAGE_MC_CURVE = True
    # Set to True to generate the landscape plots for each metric in the list below
    PLOT_METRIC_LANDSCAPES = True
    
    # --- 1. Define Input and Output ---
    
    # List of CSV files containing the GNM sweep results.
    # The script will combine data from all these files.
    df_paths = [ 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_123348/00_gnm_experiments_results.csv",
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_143109/00_gnm_experiments_results.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_172604/00_gnm_experiments_results.csv", # smaller area around estimated energy sink 
        
        # from here on: new multiprocessing
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250913_012446/00_gnm_experiments_results.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/02_gnm_experiments/combined_results02_gnm_experiments_summary.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/02_gnm_experiments/combined_results02_gnm_experiments_summary.csv", 
        
        # from here on: new mc evaluation
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/03_gnm_experiments_0-99_input_scaling_with_new_evaluation_script/combined_results03_gnm_experiments_0-99_input_scaling_with_new_evaluation_script_summary.csv", 
        
        # higher resolution (114) 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/04_bigger_network_114/combined_results04_bigger_network_114_summary.csv", 
        
        # ridge
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/05_ridge/combined_results05_ridge_summary.csv", 
        
        # new eval
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/06_new_mc_eval/combined_results06_resolution1000_summary.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/06_new_mc_eval/combined_results06_new_mc_eval_summary.csv", 
        
        # with wiring cost 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/06_resolution1000/combined_results06_resolution1000_summary.csv", 
        
        # with new mc
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/07_resolution_69_propper_mc_calc/combined_results07_resolution_69_propper_mc_calc_summary.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/08_spectral_radius_1_5/combined_results08_spectral_radius_1_5_summary.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/08_spectral_radius_1_5/combined_results08_spectral_radius_1_5_summary.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/09_spectral_radius_1_5/combined_results09_spectral_radius_1_5_summary.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/10_spectral_radius_2_0/combined_results10_spectral_radius_2_0_summary.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/10_spectral_radius_2_0/combined_results10_spectral_radius_2_0_summary.csv"
        
        # back to no grid experiments
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/08_spectral_radius_1_5/combined_results08_spectral_radius_1_5_summary.csv", 
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/08_spectral_radius_1_5_200_esn_runs/combined_results08_spectral_radius_1_5_200_esn_runs_summary.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/08_spectral_radius_1_5_200_esn_runs_spectral_rad_1/08_spectral_radius_1_5_200_esn_runs_spectral_rad_1_20250917_002929/08_spectral_radius_1_5_200_esn_runs_spectral_rad_1_results.csv"
        
        # now combined with ai cleaned code
        # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/07_run/combined_results07_run_summary.csv"
        
        # run with newly set up conda environment
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/07_run_1p0/07_run_1p0_20250919_091219/07_run_1p0_results.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/07_run_1p0_ridge_old_defaults/combined_results07_run_1p0_ridge_old_defaults_summary.csv"
        
        # new basic parameters
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/08_run_new_parameters_bias_0/combined_results08_run_new_parameters_bias_0_summary.csv"
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/09_bigger_sweep_matching_index/combined_results09_bigger_sweep_matching_index_summary.csv"
        
        # 12: Medium sweep (250) with matching index and communicability
        # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/12_average_connectome_matching_index_and_communicability/combined_results12_average_connectome_matching_index_and_communicability_summary.csv", 
        
        # 14: Big sweep, density 10 
        "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/14_average_connectome_matching_index_and_communicability_density10/combined_results14_average_connectome_matching_index_and_communicability_density10_summary.csv", 

    ]
    
    # The folder where the final figures will be saved.
    # It will be created inside the same directory as the first data file.
    save_path = Path(df_paths[0]).parent / "figures"
    
    # --- 2. Generate Average MC Curve Plot ---
    if PLOT_AVERAGE_MC_CURVE:
        print("--- Generating Average MC Curve Plot ---")
        try:
            # Load the combined dataframe once
            if not df_paths:
                raise ValueError("df_paths list cannot be empty.")
            
            print("Loading and combining data...")
            combined_df = generate_entire_df(df_paths)
            
            if not combined_df.empty:
                visualizer = PipelineVisualizer()
                avg_mc_save_path = Path(save_path) / "average_mc_curve.pdf"
                visualizer.plot_average_mc_curve(combined_df, savepath=avg_mc_save_path, show=False)
            else:
                print("SKIPPING: No data found to generate average MC curve plot.")
        except Exception as e:
            print(f"ERROR: Failed to generate Average MC Curve plot.")
            traceback.print_exc()
        print("\n")

    # --- 3. Define Metrics for Landscape Plots ---
    if PLOT_METRIC_LANDSCAPES:
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
        ] + [f"mc_{lag}" for lag in lags_to_plot]
        
        print(f"Starting landscape visualization process...")
        print(f"Output will be saved to: {save_path}\n")
        
        # --- 4. Generate Landscape Plots in a Loop ---
        for metric in metrics_to_plot:
            print(f"--- Generating plot for metric: {metric} ---")
            try:
                figure_save_name = f"no_dots_comparison_landscape_{metric}"

                visualize_gnm_results(
                    df_paths=df_paths,
                    metric_to_visualize=metric,
                    save_dir=save_path,
                    save_name=figure_save_name,
                    save_format="pdf", 
                    save_individual=False, 
                    show_dots=False,
                )
                print(f"Successfully generated plot for {metric}.\n")
                
            except ValueError as ve:
                print(f"SKIPPING: Could not generate plot for '{metric}'. Reason: {ve}\n")
            except Exception as e:
                print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
                traceback.print_exc()
                print("\n")

    print("--- Visualization process completed. ---")
