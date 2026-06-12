
"""
Using the visualization module for connectome analysis pipeline.
This script generates only the Voronoi landscape plot for a predefined list of metrics.
"""

import os
import traceback
from pathlib import Path
import pandas as pd
import matplotlib

from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer

if __name__ == "__main__":
    
    ##### CONFIG STUFF #######################
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/76_90000_samples_animal_206/all_metrics_for_76_90000_samples_animal_206.csv
    experiment_name = "76_90000_samples_animal_206" # 75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206"
    base_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/{experiment_name}")
    save_path = base_path / f"all_computational_metrics_for_{experiment_name}_updated_combined.csv"
    all_metrics_file = True 

    ##########################################
    if all_metrics_file: 
        df_paths = [base_path / f"all_metrics_for_{experiment_name}.csv"]

    else: 
        df_static_path = base_path / f"all_static_metrics_for_{experiment_name}_updated.csv"
        df_static = pd.read_csv(df_static_path)
        df_dynamic_path = base_path / f"all_dynamic_metrics_for_{experiment_name}_updated.csv"
        df_dynamic = pd.read_csv(df_dynamic_path)
        df_computational_path = base_path / f"all_computational_metrics_for_{experiment_name}_updated.csv"
        df_computational = pd.read_csv(df_computational_path)
        df_combined = pd.concat([df_static, df_dynamic, df_computational], axis=1)

        df_combined.to_csv(save_path)
    # --- 1. Define Input and Output ---
    
        # animal_id = 0
        # List of CSV files containing the GNM sweep results.
        df_paths = [save_path]      

    plot_indiv_connectomes = False # True # False # True # False # True #  False # True # False
    duplicate_handling = "mean" #"first" # mean, first, last. 
    show_number_samples = True #  False,
    eta_span = [-8, 3]
    gamma_span = [-0.1, 1]
    
    
    # The folder where the final figures will be saved.
    # A new subfolder is used to keep these plots separate.
    parent_folder = Path(df_paths[0]).parent
    if duplicate_handling == "first" or duplicate_handling == "last": 
        save_path = parent_folder / f"figures_voronoi_only_{duplicate_handling}"
    elif duplicate_handling == "mean": 
        save_path = parent_folder / f"figures_voronoi_only_mean"
    else: 
        print("Attention: Duplicate handling is not really set.")
        exit()

    
    # --- 2. Define Metrics to Plot ---
    
    lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 20, 49]
    metrics_to_plot = [
    ############### ORIG ####################################
        "MaxCriteria",
        "avg_communicability", 
        "global_efficiency", 
        "modularity", 
        "avg_clustering", 
        # "avg_degree", 
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
    ] + [f"mc_{lag}" for lag in lags_to_plot

    ############### NEW #####################################
    # "spectral_radius","spectral_gap","spectral_gap_fatemeh","global_efficiency","diffusion_efficiency","propagation_efficiency","nct_control_avg","nct_control_std","nct_control_n_nodes_90_percent","nct_control_n_nodes_50_percent","nct_energies_energy_total","nct_energies_std_node_energy","nct_energies_n_nodes_90_percent","nct_energies_n_nodes_50_percent","metastability_global","metastability_local_mean","metastability_local_std","metastability_local_kurtosis"

        # "density", "avg_clustering", "avg_degree", "degree_assortativity", "modularity", 
        # "characteristic_path_length", "transitivity", "wiring_cost", "shortest_path_distance", 
        # "structural_complexity", "n_connected_components", "omega",
        # "topological_distance", 
        #     ## "resistance_distance", # DOES NOT WORK!! 
        # "degree_gini", 
    
        # "spectral_radius", # works
        # "spectral_gap", # works
        # "spectral_gap_fatemeh", # works?
        # "global_efficiency", # works
        # "diffusion_efficiency", # -> Returns 0 if unconnected nodes exist: Error calculating diffusion_efficiency for net_eta3.5_gamma1.0_ruleMatchingIndex_id017.npy: Array must not contain infs or NaNs
        # "propagation_efficiency", # works
        # "nct_control",  # works
        # "nct_energies", # works
        # "metastability", # works
        # "synchronizability_eigenratio", # works 
        # "algebraic_connectivity_nx", # works
        # "kuramoto_synchronization", # works. But takes ages (maybe 3 hours for 11,000 networks?)
        # "community_synchronization_vulnerability", # works 
    
        
        # "kernel_rank", # works
        # "kernel_rank_fatemeh", # -> Lots of Runtime warnings in the echoes part of it (generate_esn_open) -> switch to pinv instad of ridge?: /opt/miniconda3/envs/ma_thesis/lib/python3.13/site-packages/sklearn/linear_model/_ridge.py:252: UserWarning: Singular matrix in solving dual problem. Using least-squares solution instead.
        # "effective_dimensionality", # works
        # "multifunctionality", # works?? - or does at least produce values??

    ]

    
        # "metastability_local_mean","metastability_local_std","metastability_local_kurtosis",
        # "kernel_rank_fatemeh"

        # "density",
        # "propagation_efficiency"
        # "spectral_radius","degree_assortativity"
        # "degree_gini", 
        # "spectral_radius",
        # "spectral_gap",
        # "global_efficiency",
        # # "diffusion_efficiency",
        # "propagation_distance",
        # "propagation_efficiency",
        # "average_controllability",

        # "nct_control_avg",
        # "nct_control_std",
        # "nct_control_n_nodes_90_percent",
        # "nct_energies_energy_total",
        # "nct_energies_n_nodes_90_percent"


        # "kernel_rank",
        # "effective_dimensionality",
        # "multifunctionality", # -> Implement how? 

    #     "n_connected_components",
        # "structural_complexity"

        # "avg_clustering_glob_efficiency_minus_energy", 
        # "avg_clustering_divided_by_global_efficiency", 
    # ]
    
    plot_combined_lag_plot = False # Plot this comparison plot with different MC lags 
    
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

            # get eta min add max, turn them to spans + 10% margin
            # eta_min = df["eta"].min()
            # eta_max = df["eta"].max()
            # eta_span = (eta_max - eta_min) * 0.05
            # eta_span = (eta_min - eta_span, eta_max + eta_span)
            # eta_span = (eta_min, eta_max) 

            # gamma_min = df["gamma"].min()
            # gamma_max = df["gamma"].max()
            # gamma_span = (gamma_max - gamma_min) * 0.05
            # gamma_span = (gamma_min - gamma_span, gamma_max + gamma_span)
            # gamma_span = (gamma_min, gamma_max)

            print("Using e.g.:", df_paths[0])
            fig, ax = visualizer.plot_metric_landscape_voronoi(
                df, 
                title=plot_title,
                metric_name=metric_col_name,
                savepath=full_save_path,
                dot_color="steelblue", 
                eta_span=eta_span,
                gamma_span=gamma_span,
                show=True, # False 
                show_dots=False,
                annotate_extremes=True, 
                estimated_indiv_connectomes=df_best_gamma_and_eta_estimates if plot_indiv_connectomes else None, 
                duplicate_handling=duplicate_handling, 
                show_number_samples=show_number_samples, 
            )
            
            matplotlib.pyplot.close() # Close the plot after saving to free memory, if returned fig and ax are not used.
            
            
            print(f"Successfully generated plot for {metric} at {full_save_path}\n")
            
            
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

