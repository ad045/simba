
"""
Using the visualization module for connectome analysis pipeline.
This script generates only the Voronoi landscape plot for a predefined list of metrics.
"""

import os
import traceback
from pathlib import Path
import pandas as pd
from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer

import matplotlib


if __name__ == "__main__":
    
    animal_id = 0
    df_paths = [ 
        "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/summary_indiv_energies_for_exp_60_generally_finer_search_animal_0.csv"
    ]       
    plot_indiv_connectomes = False 
    
    
    ##############
    
    # Folders
    parent_folder = Path(df_paths[0]).parent
    save_path = parent_folder / "figures_voronoi_indiv_fits"
    
    
    # Metrics to plot
    individuals = [i for i in range(255)] # [1, 2, 3, 4, 5, 6, 10, 20, 49]
    metrics_to_plot = [f"MaxCrit_indiv_{lag}" for lag in individuals]
    #     "MaxCriteria",
    #     "avg_communicability", 
    #     "global_efficiency", 
    #     "modularity", 
    #     "avg_clustering", 
    #     "avg_degree", 
    #     "transitivity", 
    #     "avg_edge_distance", 
    #     "char_path_length", 
    #     "richclub_n_edges", 
    #     "richclub_avg_length", 
    #     "mc_mean", 
    #     "mc_std", 
    #     "wiring_cost", 
    #     "mean_mc_divided_by_wiring_cost",
    #     "mc_5_divided_by_wiring_cost", 
        
    #     # "avg_clustering_glob_efficiency_minus_energy", 
    #     # "avg_clustering_divided_by_global_efficiency", 
    # ]
    plot_combined_lag_plot = False # Plot this comparison plot with different MC lags 
    
    print(f"Starting Voronoi-only visualization process...")
    print(f"Output will be saved to: {save_path}\n")

    # Load data 
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

        path_to_best_gamma_and_eta_estimations = parent_folder / "min_energy_results.csv"    # not entirely clean... TODO: Make this clean. 
        if os.path.exists(path_to_best_gamma_and_eta_estimations) and plot_indiv_connectomes:
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
            
            
            if metric == "avg_clustering_divided_by_global_efficiency": 
                # df["avg_clustering_glob_efficiency"] = pd.to_numeric(df["avg_clustering"], errors='coerce') / pd.to_numeric(df["global_efficiency"], errors='coerce') * pd.to_numeric(df["modularity"], errors='coerce')
                upper_factor = pd.to_numeric(df["avg_clustering"], errors='coerce')
                # upper_factor = (upper_factor - upper_factor.min()) / (upper_factor.max() - upper_factor.min())
                
                lower_factor = pd.to_numeric(df["global_efficiency"], errors='coerce')
                # lower_factor = (lower_factor - lower_factor.min()) / (lower_factor.max() - lower_factor.min())
                
                total = upper_factor * lower_factor * (-1)
                total = (total - total.min()) / (total.max() - total.min()) 
                
                df["avg_clustering_divided_by_global_efficiency"] = total #  (total - energy) 
            
            
            # if metric == "avg_clustering_glob_efficiency_minus_energy": 
            #     # df["avg_clustering_glob_efficiency"] = pd.to_numeric(df["avg_clustering"], errors='coerce') / pd.to_numeric(df["global_efficiency"], errors='coerce') * pd.to_numeric(df["modularity"], errors='coerce')
            #     upper_factor = pd.to_numeric(df["avg_clustering"], errors='coerce')
            #     upper_factor = (upper_factor - upper_factor.min()) / (upper_factor.max() - upper_factor.min())
                
            #     lower_factor = pd.to_numeric(df["global_efficiency"], errors='coerce')
            #     lower_factor = (lower_factor - lower_factor.min()) / (lower_factor.max() - lower_factor.min())
                
            #     energy = pd.to_numeric(df["MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)"], errors='coerce')
            #     energy = (energy - energy.min()) / (energy.max() - energy.min())
                
            #     total = upper_factor * lower_factor * (-1)
            #     total = (total - total.min()) / (total.max() - total.min())
            #     total = (total - energy) 
                
            #     df["avg_clustering_glob_efficiency_minus_energy"] = total 
                
                
            
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
                cmap="Blues_r", 
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
    if plot_combined_lag_plot: 
        print("--- Generating combined Voronoi plot for all MC lags ---")
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            # Define a base name for the save file, the function will add the extension
            figure_save_name = "voronoi_landscape_mc_lags_grid"
            full_save_path = save_dir / figure_save_name

            mc_cols_exist = any(f"mc_{lag}" in gnm_results_df.columns for lag in individuals)

            if mc_cols_exist:
                visualizer.plot_mc_lag_landscapes_voronoi(
                    gnm_results_df,
                    lags=individuals,
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

