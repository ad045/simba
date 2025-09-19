# Write this completely again... 

# """
# Script for Goal 3: Analyze Preprocessed Human Connectomes.

# This script performs a comprehensive analysis of the 70 preprocessed human
# connectomes by:
# 1. Loading each connectome and the associated distance matrix using the project's DataLoader.
# 2. Calculating its Memory Capacity (MC) using a configurable ESN evaluation pipeline.
# 3. Computing a suite of structural graph theory metrics (e.g., efficiency,
#    modularity, wiring cost).
# 4. Aggregating all functional and structural results into a single CSV file
#    for easy analysis and visualization.

# To run this script, you first need a configuration file.
# Create a file named `configs/analyze_connectomes.yaml` with the following content:

# ```yaml
# defaults:
#   - data: resolution68 # Or another data config
#   - esn: default_esn   # A config for ESN parameters

# results_dir: 'shared_results/connectome_analysis'
# output_filename: 'human_connectome_analysis_results.csv'
# ```

# Then, run from the project's root directory:
# `python scripts/05_analyze_connectomes.py`
# """
# import os
# import sys
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import hydra
# from omegaconf import DictConfig, OmegaConf

# # Add the project root to the Python path to allow for absolute imports
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

# # Import your project's modules
# from src.utils.data_loader import DataLoader
# from src.ESNs.alternative_esn_evaluation import evaluate_memory_capacity_from_connectome
# # esn_evaluation import evaluate_memory_capacity
# from src.structural_analysis.graph_measures import analyze_connectomes



# def analyze_all_connectomes(data_loader: DataLoader, esn_config: DictConfig):
#     """
#     Loads connectomes, calculates their MC and structural metrics, and returns a DataFrame.

#     Args:
#         data_loader (DataLoader): An instance of the DataLoader class.
#         esn_config (DictConfig): Configuration object for the ESN evaluation.

#     Returns:
#         pd.DataFrame: A DataFrame containing the analysis results for all connectomes.
#     """
#     print("Loading data...")
#     connectomes = data_loader.get_all_connectomes()
#     distance_matrix = data_loader.get_distance_matrix()
#     print(f"Loaded {len(connectomes)} connectomes.")

#     # Convert OmegaConf object to a standard python dict for **kwargs expansion
#     esn_params = OmegaConf.to_container(esn_config, resolve=True)
#     results = []

#     # Use tqdm for a progress bar during the loop
#     for subject_id, W in tqdm(connectomes.items(), desc="Analyzing Connectomes"):
#         # --- 1. Functional Analysis: Memory Capacity ---
#         esn_results = evaluate_memory_capacity_from_connectome(W, **esn_params)
#         memory_capacity = esn_results.get('mc', np.nan)

#         # --- 2. Structural Analysis: Graph Metrics ---

#         # global_efficiency = get_global_efficiency(W_binary)
#         # modularity = get_modularity_louvain(W_binary)
#         # wiring_cost = get_wiring_cost(W_binary, distance_matrix)
#         # avg_clustering = get_average_clustering_coefficient(W_binary)
#         # char_path_len = get_characteristic_path_length(W_binary)
        
#         graph_measures = analyze_connectomes(W) # W is transformed to binary in this function before the analysis

#         # array of them: 
#             #     out.append({
#             #     "network_index": idx,
#             #     "avg_communicability": avg_comm,
#             #     "global_efficiency": glob_eff,
#             #     "modularity": modu,
#             #     "avg_clustering": avg_clust,
#             #     "avg_degree": avg_deg,
#             #     "transitivity": trans,
#             #     "avg_edge_distance": avg_dist,
#             #     "char_path_length": cpl,
#             #     "richclub_n_edges": n_rich_edges,
#             #     "richclub_avg_length": avg_rc_length,
#             # })


#         # --- 3. Aggregate Results ---
#         subject_results = {
#             'subject_id': subject_id,
#             'memory_capacity': memory_capacity,
#             # 'global_efficiency': global_efficiency,
#             # 'modularity': modularity,
#             # 'wiring_cost': wiring_cost,
#             # 'average_clustering': avg_clustering,
#             # 'characteristic_path_length': char_path_len,
#         }
#         subject_results.update(graph_measures[0]) # as we are always looking at only one connectome 
#         results.append(subject_results)

#     return pd.DataFrame(results)


# @hydra.main(config_path="../configs", config_name="analyze_connectomes", version_base=None)
# def main(cfg: DictConfig):
#     """
#     Main function to run the connectome analysis pipeline using Hydra configuration.
#     """
#     # --- Configuration & Setup ---
#     print("--- Configuration ---")
#     print(OmegaConf.to_yaml(cfg))
#     print("---------------------")

#     # Instantiate the data loader with the data configuration
#     data_loader = DataLoader(cfg.data)

#     # Define output path
#     results_dir = hydra.utils.to_absolute_path(cfg.results_dir)
#     os.makedirs(results_dir, exist_ok=True)
#     output_csv = os.path.join(results_dir, cfg.output_filename)


#     # --- Run Analysis ---
#     analysis_df = analyze_all_connectomes(data_loader, cfg.esn)

#     # --- Save Results ---
#     analysis_df.to_csv(output_csv, index=False)
#     print("\nAnalysis complete.")
#     print(f"Results saved to: {output_csv}")
#     print("\nFirst 5 rows of the results:")
#     print(analysis_df.head())


# if __name__ == "__main__":
#     main()

