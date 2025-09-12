# """
# Using the visualization module for connectome analysis pipeline.
# Integrates energy landscape plotting for (in future both) GNM (and ESN) results.
# """

# from pathlib import Path
# from src.visualization.energy_and_mc_landscape import (visualize_gnm_results)


# # Usage 
# if __name__ == "__main__":
    
#     print("GNM visualization:")
    
#     # name_of_energy_metric = "MaxCriteria(DegreeKS_ClusteringKS)"
    
#     # File path
#     # One old file (not the hyper large one)
#     # df_name = "/Users/adrian/Documents/01_projects/14_4D_lab/OLD_output_3/default_folder/binary_evaluations_resultsdistance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_400.csv"
#     # df_name = "/Users/adrian/Documents/01_projects/14_4D_lab/output/00_gnm_experiments/binary_evaluations_resultsdistance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_227.csv"
#     # df_name = "/Users/adrian/Documents/01_projects/14_4D_lab/output/00_gnm_experiments/binary_evaluations_results_distance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_227.csv"
#     # df_name = "/Users/adrian/Documents/01_projects/14_SAFETY_COPY_2/OLD_output_3/default_folder/binary_evaluations_resultsdistance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_400.csv" 
#     # df_name = "/Users/adrian/Documents/01_projects/14_SAFETY_COPY_1_edited/V2_before_deleting_the_too_big_commit/OLD_output/02_esns_on_observed_weighted_connectomes/esn_grid_resolution68_2025-08-15_11-38-29/gnm_mc_results_2025-08-15_11-38-29.csv"
#     df_names = "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250910_133726/binary_evaluations_results_distance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_455.csv"

#     save_path = Path(df_names).parent / "figures"
#     df_names = [ 
#                 "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_123348/00_gnm_experiments_results.csv", 
#                 "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_134723/00_gnm_experiments_results.csv", 
#                 # /Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250911_190638/00_gnm_experiments_results.csv", 
#                 # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250911_190638/00_gnm_experiments_results.csv", # newest
#                 # "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250910_133726/binary_evaluations_results_distance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_455.csv", 
#                 # "/Users/adrian/Documents/01_projects/14_4D_lab/OLD_output_6/gnm/00_gnm_experiments/00_gnm_experiments_20250910_131743/binary_evaluations_results_distance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_227.csv",
#                 ] 
    
#     save_path = Path(df_names[0]).parent / "figures"
    
    
#     try:
#         name_of_energy_metric = visualize_gnm_results(
#             df_paths=df_names, # can be one file name or array of names. 
#             # name_of_energy_metric=name_of_energy_metric, # if default, it will find the column starting with "MaxCriteria"
#             save_dir=save_path, # defaults to "output/figures"
#             save_format="pdf", 
#             save_individual=True, # Save both individual and comparison plots
#             show_dots=True,
#         )
#         print(f"Used energy metric: {name_of_energy_metric}")
#         print(f"Visualization completed successfully!")
#         print(f"Results saved to: {save_path}")
#     except Exception as e:
#         print(f"Error during visualization: {e}")
#         import traceback
#         traceback.print_exc()


"""
Using the visualization module for connectome analysis pipeline.
Integrates energy landscape plotting for (in future both) GNM (and ESN) results.
"""

from pathlib import Path
import pandas as pd
from src.visualization.energy_and_mc_landscape import (visualize_gnm_results, PipelineVisualizer)


# Usage 
if __name__ == "__main__":
    
    
    # --- Example 1: Visualize a single metric (e.g., energy, avg_edge_distance) ---
    print("--- Visualizing a single metric ---")
    
    # Define the metric you want to visualize from the CSV file.
    # This can be 'MaxCriteria', 'avg_edge_distance', 'mc_mean', etc
        # ,network_index ,avg_communicability, global_efficiency, modularity, avg_clustering, avg_degree, 
        # transitivity, avg_edge_distance, char_path_length, richclub_n_edges, richclub_avg_length, mc_mean, mc_std
    
    metric_to_plot = "char_path_length"  # char_path_length" # modularity" # avg_communicability" # avg_edge_distance" # MaxCriteria" 

    df_paths_single_metric = [ 
        "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_123348/00_gnm_experiments_results.csv",
        "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_143109/00_gnm_experiments_results.csv", 
        "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_123348/00_gnm_experiments_results.csv",
    ] 
    
    save_path_single_metric = Path(df_paths_single_metric[0]).parent / "figures"
    
    try:
        visualized_metric = visualize_gnm_results(
            df_paths=df_paths_single_metric,
            metric_to_visualize=metric_to_plot,
            save_dir=save_path_single_metric,
            save_format="pdf", 
            save_individual=True,
            show_dots=True,
        )
        print(f"Used metric: {visualized_metric}")
        print(f"Single metric visualization completed successfully!")
        print(f"Results saved to: {save_path_single_metric}")
    except Exception as e:
        print(f"Error during single metric visualization: {e}")
        import traceback
        traceback.print_exc()

    # # --- Example 2: Visualize Memory Capacity (MC) across different lags ---
    # print("\n--- Visualizing MC lag landscapes ---")

    # df_paths_mc_lags = [
    #     "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_123348/00_gnm_experiments_results.csv",
    #     "/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_experiments_20250912_143109/00_gnm_experiments_results.csv", 
    # ]
    
    # # Choose interpolation: 'linear', 'cubic', or 'nearest'
    # interpolation = 'linear'

    # save_path_mc_lags = Path(df_paths_mc_lags[0]).parent / "figures_mc_lags"
    # save_path_mc_lags.mkdir(parents=True, exist_ok=True)
    
    # lags_to_plot = [] # [1, 2, 5, 10, 15, 20, 30, 40, 50]    # Lags to visualize, corresponding to 'mc_lag_X' columns in the CSV. Can be [] for "skipping". 
    
    # if lags_to_plot != []:
            
    #     try:
    #         # Load the dataframe
    #         df = pd.read_csv(df_paths_mc_lags[0])
            
    #         # Initialize the visualizer
    #         visualizer = PipelineVisualizer()

    #         # Generate and save the plot
    #         visualizer.plot_mc_lag_landscapes(
    #             df,
    #             lags=lags_to_plot,
    #             interpolation_method=interpolation,
    #             savepath=save_path_mc_lags / f"mc_lag_landscapes_{interpolation}"
    #         )
            
    #         print(f"MC lag visualization completed successfully!")
    #         print(f"Results saved to: {save_path_mc_lags}")
            
    #     except Exception as e:
    #         print(f"Error during MC lag visualization: {e}")
    #         import traceback
    #         traceback.print_exc()