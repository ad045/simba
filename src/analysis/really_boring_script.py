# # # # %%
# # # import numpy as np 
# # # import networkx as nx
# # # import pandas as pd
# # # import os
# # # from pathlib import Path

# # # from abc import ABC, abstractmethod
# # # import networkx as nx


# # # # %%
# # # from src.analysis.kayson_utils import (compute_structural_complexity, 
# # #                                         compute_omega, 
# # #                                         # 
# # #                                         resistance_distance,  # "diffusion_distance" -> needs coords... 
# # #                                         shortest_path_distance,
# # #                                         propagation_distance,
# # #                                         topological_distance,
# # #                                         check_density,
# # #                                         calculate_wiring_cost,
# # #                                         # euclidean_distance,
# # #                                         calculate_endpoint_similarity,
# # #                                         evaluate_adjacency
# # #                                        )                      

# # #         # metrics[6] = nx.average_clustering(G)
# # #         # metrics[7] = nx.degree_assortativity_coefficient(G)
    
# # # from src.analysis.utils import get_eta_and_gamma_from_filename, sorted_listing_by_creation_time

# # # # %%
# # # EXPERIMENT = "60_generally_finer_search_animal_0" # 61_testing_with_kayson_animal_0"
# # # DATASET = "suarez_MaMI_dataset"
# # # BASE_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
# # # OUTPUT_PATH = BASE_PATH / "output" / "gnm" / DATASET / EXPERIMENT

# # # generated_networks_dir = OUTPUT_PATH / "generated_networks"

# # # df_path = OUTPUT_PATH / f"all_metrics_for_{EXPERIMENT}.csv"
# # # df = pd.read_csv(df_path)

# # # df_path_out = OUTPUT_PATH / f"all_metrics_for_{EXPERIMENT}_updated.csv"

# # # distance_matrix_path = BASE_PATH / "data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy"
# # # distance_matrix = np.load(distance_matrix_path)



# # # # %%


# # # class MetricCalculator(ABC):
# # #     def __init__(self, A=None, distance_matrix=None):
# # #         self.A = A
# # #         self.distance_matrix = distance_matrix
    
# # #     @abstractmethod
# # #     def calculate_metric(self, metric_name):
# # #         """Calculate the specified metric. Must be implemented by subclasses."""
# # #         pass
    
# # #     # Any shared helper methods go here
# # #     def _validate_inputs(self):
# # #         """Common validation logic"""
# # #         if self.A is None:
# # #             raise ValueError("Adjacency matrix A is required")


# # # class StaticMetricCalculator(MetricCalculator):
# # #     def calculate_metric(self, metric_name):
# # #         if metric_name == "density":
# # #             return nx.density(nx.from_numpy_array(self.A))
        
# # #         elif metric_name == "compute_structural_complexity":
# # #             return compute_structural_complexity(self.A)
        
# # #         elif metric_name == "n_connected_components": 
# # #             G = nx.from_numpy_array(self.A)
# # #             return nx.number_connected_components(G)
        
# # #         elif metric_name == "omega": 
# # #             return compute_omega(self.A)
        
# # #         # elif metric_name == "diffusion_distance": 
# # #         #     return resistance_distance()

# # # #             [ ] avg clustering
# # # #     [ ] Modularity (Consensus of N Louvain from netneurotools)
# # # #     [x] Small-worldness (omega from mine)
# # # #     [ ] avg wiring cost (Euclidean distance)
# # # #     [ ] Hubness (Gini index, Chini 2023)
# # # #     [ ] Rich club (with k, and looking for maximum → look for library)
# # # #     [ ] Average length


# # # # Number edges
# # # #     [ ] Average degree
# # # #     [ ] Transitivity
# # # #     [ ] Degree assortativity
# # # #     [ ] “Distance-dependent degree assortativity” (Betzel)
# # # #     [ ] Entropy of matrix


        
# # #         else:
# # #             raise ValueError(f"Unknown metric: {metric_name}")


# # # class DynamicMetricCalculator(MetricCalculator):
# # #     def calculate_metric(self, metric_name):
# # #         # TODO 
# # #         # Different implementation for dynamic networks

# # #     #     [ ] PID (priesemann papers)
# # #     # [ ] Spectral radius
# # #     # [ ] normalized Entropy of the eigenspectrum (I have code)
# # #     # [ ] Spectral gap
# # #     # [ ] Maximum metastability and the corresponding coupling
# # #     # [ ] Propagation efficiency (avg communicability)
# # #     # [ ] Global efficiency (avg shortest path length)
# # #     # [ ] Diffusion efficiency (avg effective distance)
# # #     # [ ] avg controllability
# # #         if metric_name == "density":
# # #             return self._calculate_temporal_density()


# # # class ComputationMetricCalculator(MetricCalculator):
# # #     def calculate_metric(self, metric_name):
# # #         # TODO 
# # #         # Different implementation for dynamic networks
# # #     # [ ] Memory capacity
# # #     # [ ] Kernel Rank
# # #     # [ ] Effective dimensionality
# # #     # [ ] Multifunctionality
# # #         if metric_name == "density":
# # #             return self._calculate_temporal_density()


# # # class PortraitMetricCalculator(MetricCalculator):
# # #     def calculate_metric(self, metric_name):
# # #         # TODO 
# # #         # Comparison to empirical connectomes: (or to default networks: random, etc?) 
# # #         # [ ] Portraits
# # #         # [ ] $Rˆ{2}$ (maybe not useful for MaMI, as we have no fixed connectome?)
# # #         if metric_name == "density":
# # #             return self._calculate_temporal_density()



# # # # class MetricCalculator:
# # # #     def __init__(self, A=None, distance_matrix=None):
# # # #         self.A = A
# # # #         self.distance_matrix = distance_matrix
    
# # # #     def calculate_metric(self, metric_name):

# # # #         if metric_name == "density":
# # # #             return nx.density(nx.from_numpy_array(self.A))
        
# # # #         elif metric_name == "compute_structural_complexity":
# # # #             return compute_structural_complexity(self.A)
        
# # # #         elif metric_name == "omega": 
# # # #             return compute_omega(self.A)
        
# # # #         elif metric_name == "diffusion_distance": 
# # # #             return resistance_distance()
        
# # # #         else:
# # # #             print("Unknown metric:", metric_name)
# # # #             return None
        
                        
                    
# # # # %%
# # # # Iterate over files in directory

# # # interesting_metrics = ["density", 
# # #                        "compute_structural_complexity", 
# # #                        "n_connected_components"
# # #                        ]
# # #                     #    "omega"
# # #                     #    ] # omega means small-world-ness

# # # static_metric_calculator = StaticMetricCalculator(distance_matrix=distance_matrix)
# # # # print(sorted_listing_by_creation_time(directory))
# # # for i, name in enumerate(sorted_listing_by_creation_time(generated_networks_dir)): # os.listdir(generated_networks_dir):
# # #     if name.endswith(".npy"):
# # #         full_path = generated_networks_dir / name
# # #         A = np.load(full_path)[0]  # Load the numpy array (mostly: 1, 100, 100)
# # #         eta, gamma = get_eta_and_gamma_from_filename(name)

# # #         static_metric_calculator.A = A  # set only A every time...

# # #         # Find corresponding row in dataframe. If empty, create new row
# # #         matching_rows = df[(df['eta'] == eta) & (df['gamma'] == gamma)]
# # #         if matching_rows.empty:
# # #             new_row = {"eta": eta, "gamma": gamma}
# # #             df.loc[len(df)] = new_row
# # #             matching_rows = df[(df['eta'] == eta) & (df['gamma'] == gamma)]

# # #         # Process that row (again) by iterating over interesting metrics
# # #         for metric in interesting_metrics:
# # #             # print(f"Processing metric: {metric} for eta: {eta}, gamma: {gamma}")
# # #             # Check if metric is missing or NaN -> if so, calculate and update
# # #             if metric not in matching_rows.columns or df.loc[0][metric] != np.nan:
# # #                 row_indexer = (df['eta'] == eta) & (df['gamma'] == gamma)
# # #                 df.loc[row_indexer, metric] = static_metric_calculator.calculate_metric(metric)
# # #                 # print("data: ", df.loc[row_indexer, metric])
# # #         df.loc[row_indexer, "changed_things"] = "yes"

# # #     print(".")

# # #     # save after every new network eval 
# # #     if i % 10 == 9: 
# # #         df.to_csv(df_path_out, index=False)
    
# # #         break

# # # print(df_path_out)



# # # %%
# # import numpy as np 
# # import networkx as nx
# # import pandas as pd
# # import os
# # from pathlib import Path
# # from multiprocessing import Pool, cpu_count
# # from functools import partial
# # import tempfile
# # import shutil

# # from abc import ABC, abstractmethod

# # # %%
# # from src.analysis.kayson_utils import (compute_structural_complexity, 
# #                                         compute_omega, 
# #                                         resistance_distance,
# #                                         shortest_path_distance,
# #                                         propagation_distance,
# #                                         topological_distance,
# #                                         check_density,
# #                                         calculate_wiring_cost,
# #                                         calculate_endpoint_similarity,
# #                                         evaluate_adjacency
# #                                        )                      
    
# # from src.analysis.utils import get_eta_and_gamma_from_filename, sorted_listing_by_creation_time

# # # %%
# # EXPERIMENT = "60_generally_finer_search_animal_0"
# # DATASET = "suarez_MaMI_dataset"
# # BASE_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
# # OUTPUT_PATH = BASE_PATH / "output" / "gnm" / DATASET / EXPERIMENT

# # generated_networks_dir = OUTPUT_PATH / "generated_networks"

# # df_path = OUTPUT_PATH / f"all_metrics_for_{EXPERIMENT}.csv"
# # df = pd.read_csv(df_path)

# # df_path_out = OUTPUT_PATH / f"all_metrics_for_{EXPERIMENT}_updated.csv"

# # distance_matrix_path = BASE_PATH / "data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy"
# # distance_matrix = np.load(distance_matrix_path)

# # # %%
# # class MetricCalculator(ABC):
# #     def __init__(self, A=None, distance_matrix=None):
# #         self.A = A
# #         self.distance_matrix = distance_matrix
    
# #     @abstractmethod
# #     def calculate_metric(self, metric_name):
# #         """Calculate the specified metric. Must be implemented by subclasses."""
# #         pass
    
# #     def _validate_inputs(self):
# #         """Common validation logic"""
# #         if self.A is None:
# #             raise ValueError("Adjacency matrix A is required")


# # class StaticMetricCalculator(MetricCalculator):
# #     def calculate_metric(self, metric_name):
# #         if metric_name == "density":
# #             return nx.density(nx.from_numpy_array(self.A))
        
# #         elif metric_name == "compute_structural_complexity":
# #             return compute_structural_complexity(self.A)
        
# #         elif metric_name == "n_connected_components": 
# #             G = nx.from_numpy_array(self.A)
# #             return nx.number_connected_components(G)
        
# #         elif metric_name == "omega": 
# #             return compute_omega(self.A)
        
# #         else:
# #             raise ValueError(f"Unknown metric: {metric_name}")


# # # %%
# # # Worker function to process a single network file
# # def process_network_file(file_info, distance_matrix, interesting_metrics):
# #     """
# #     Process a single network file and return a dictionary of results.
    
# #     Args:
# #         file_info: tuple of (index, filename)
# #         distance_matrix: numpy array of distances
# #         interesting_metrics: list of metric names to calculate
    
# #     Returns:
# #         dict with eta, gamma, and all calculated metrics
# #     """
# #     idx, name, full_path = file_info
    
# #     try:
# #         # Load the network
# #         A = np.load(full_path)[0]  # Load the numpy array (mostly: 1, 100, 100)
# #         eta, gamma = get_eta_and_gamma_from_filename(name)
        
# #         # Initialize calculator
# #         static_metric_calculator = StaticMetricCalculator(A=A, distance_matrix=distance_matrix)
        
# #         # Calculate all metrics
# #         results = {"eta": eta, "gamma": gamma}
# #         for metric in interesting_metrics:
# #             results[metric] = static_metric_calculator.calculate_metric(metric)
        
# #         print(f"Worker {os.getpid()}: Processed {name} (file {idx})")
# #         return results
    
# #     except Exception as e:
# #         print(f"Error processing {name}: {e}")
# #         return None


# # # %%
# # def multiprocess_networks(n_processes=None):
# #     """
# #     Main function to multiprocess network metric calculations.
    
# #     Args:
# #         n_processes: Number of processes to use. If None, uses cpu_count() - 1
# #     """
# #     # Define metrics to calculate
# #     interesting_metrics = ["density", 
# #                            "compute_structural_complexity", 
# #                            "n_connected_components"
# #                            ]
    
# #     # Get all network files in order
# #     all_files = sorted_listing_by_creation_time(generated_networks_dir)
# #     network_files = [(i, name, generated_networks_dir / name) 
# #                      for i, name in enumerate(all_files) 
# #                      if name.endswith(".npy")]
    
# #     print(f"Found {len(network_files)} network files to process")
    
# #     # Determine number of processes
# #     if n_processes is None:
# #         n_processes = max(1, cpu_count() - 1)
# #     print(f"Using {n_processes} processes")
    
# #     # Create temporary directory for worker outputs
# #     temp_dir = OUTPUT_PATH / "temp_metrics"
# #     temp_dir.mkdir(exist_ok=True)
    
# #     # Create partial function with fixed arguments
# #     worker_func = partial(process_network_file, 
# #                          distance_matrix=distance_matrix,
# #                          interesting_metrics=interesting_metrics)
    
# #     # Process networks in parallel
# #     print("Starting multiprocessing...")
# #     with Pool(processes=n_processes) as pool:
# #         results = pool.map(worker_func, network_files)
    
# #     # Filter out None results (errors)
# #     results = [r for r in results if r is not None]
# #     print(f"Successfully processed {len(results)} networks")
    
# #     # Convert results to DataFrame
# #     results_df = pd.DataFrame(results)
    
# #     # Load original dataframe
# #     df_original = pd.read_csv(df_path)
    
# #     # Merge with original dataframe
# #     # First, ensure eta and gamma are in the original df
# #     merge_cols = ['eta', 'gamma']
    
# #     # Update original dataframe with new metrics
# #     for metric in interesting_metrics:
# #         if metric in results_df.columns:
# #             # Create a mapping from (eta, gamma) to metric value
# #             metric_map = results_df.set_index(['eta', 'gamma'])[metric].to_dict()
            
# #             # Update the original dataframe
# #             for idx, row in df_original.iterrows():
# #                 key = (row['eta'], row['gamma'])
# #                 if key in metric_map:
# #                     df_original.at[idx, metric] = metric_map[key]
    
# #     # Save the updated dataframe
# #     df_original.to_csv(df_path_out, index=False)
# #     print(f"Saved updated metrics to: {df_path_out}")
    
# #     # Clean up temp directory
# #     if temp_dir.exists():
# #         shutil.rmtree(temp_dir)
    
# #     return df_original


# # # %%
# # if __name__ == "__main__":
# #     # Run the multiprocessing
# #     df_updated = multiprocess_networks(n_processes=None)  # Uses cpu_count() - 1
# #     print("\nProcessing complete!")
# #     print(f"Updated dataframe shape: {df_updated.shape}")
# #     print(f"\nFirst few rows:\n{df_updated.head()}")

# import numpy as np 
# import networkx as nx
# import pandas as pd
# import os
# from pathlib import Path
# from multiprocessing import Pool, cpu_count
# from functools import partial
# import tempfile
# import shutil

# from abc import ABC, abstractmethod

# from src.analysis.kayson_utils import (compute_structural_complexity, 
#                                         compute_omega, 
#                                         resistance_distance,
#                                         shortest_path_distance,
#                                         propagation_distance,
#                                         topological_distance,
#                                         check_density,
#                                         calculate_wiring_cost,
#                                         calculate_endpoint_similarity,
#                                         evaluate_adjacency
#                                        )                      
    
# from src.analysis.utils import get_eta_and_gamma_from_filename, sorted_listing_by_creation_time


# class MetricCalculator(ABC):
#     def __init__(self, A=None, distance_matrix=None):
#         self.A = A
#         self.distance_matrix = distance_matrix
    
#     @abstractmethod
#     def calculate_metric(self, metric_name):
#         """Calculate the specified metric. Must be implemented by subclasses."""
#         pass
    
#     def _validate_inputs(self):
#         """Common validation logic"""
#         if self.A is None:
#             raise ValueError("Adjacency matrix A is required")


# class StaticMetricCalculator(MetricCalculator):
#     def calculate_metric(self, metric_name):
#         if metric_name == "density":
#             return nx.density(nx.from_numpy_array(self.A))
        
#         elif metric_name == "compute_structural_complexity":
#             return compute_structural_complexity(self.A)
        
#         elif metric_name == "n_connected_components": 
#             G = nx.from_numpy_array(self.A)
#             return nx.number_connected_components(G)
        
#         elif metric_name == "omega": 
#             return compute_omega(self.A)
        
#         else:
#             raise ValueError(f"Unknown metric: {metric_name}")


# # %%
# # Worker function to process a single network file
# def process_network_file(file_info, distance_matrix, interesting_metrics):
#     """
#     Process a single network file and return a dictionary of results.
    
#     Args:
#         file_info: tuple of (index, filename)
#         distance_matrix: numpy array of distances
#         interesting_metrics: list of metric names to calculate
    
#     Returns:
#         dict with eta, gamma, and all calculated metrics
#     """
#     idx, name, full_path = file_info
    
#     try:
#         # Load the network
#         A = np.load(full_path)[0]  # Load the numpy array (mostly: 1, 100, 100)
#         eta, gamma = get_eta_and_gamma_from_filename(name)
        
#         # Initialize calculator
#         static_metric_calculator = StaticMetricCalculator(A=A, distance_matrix=distance_matrix)
        
#         # Calculate all metrics
#         results = {"eta": eta, "gamma": gamma}
#         for metric in interesting_metrics:
#             results[metric] = static_metric_calculator.calculate_metric(metric)
        
#         print(f"Worker {os.getpid()}: Processed {name} (file {idx})")
#         return results
    
#     except Exception as e:
#         print(f"Error processing {name}: {e}")
#         return None


# # %%
# def multiprocess_networks(n_processes=None, experiment="", dataset=""):
#     """
#     Main function to multiprocess network metric calculations.
    
#     Args:
#         n_processes: Number of processes to use. If None, uses cpu_count() - 1
#     """
    
#     base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
#     output_path = base_path / "output" / "gnm" / dataset / experiment

#     generated_networks_dir = output_path / "generated_networks"

#     df_path = output_path / f"all_metrics_for_{experiment}.csv"
#     df = pd.read_csv(df_path)

#     df_path_out = output_path / f"all_metrics_for_{experiment}_updated.csv"

#     distance_matrix_path = base_path / "data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy"
#     distance_matrix = np.load(distance_matrix_path)
#     # Define metrics to calculate
#     interesting_metrics = ["density", 
#                            "compute_structural_complexity", 
#                            "n_connected_components"
#                            ]
    
#     # Get all network files in order
#     all_files = sorted_listing_by_creation_time(generated_networks_dir)
#     network_files = [(i, name, generated_networks_dir / name) 
#                      for i, name in enumerate(all_files) 
#                      if name.endswith(".npy")]
    
#     print(f"Found {len(network_files)} network files to process")
    
#     # Determine number of processes
#     if n_processes is None:
#         n_processes = max(1, cpu_count() - 1)
#     print(f"Using {n_processes} processes")
    
#     # Create temporary directory for worker outputs
#     temp_dir = output_path / "temp_metrics"
#     temp_dir.mkdir(exist_ok=True)
    
#     # Create partial function with fixed arguments
#     worker_func = partial(process_network_file, 
#                          distance_matrix=distance_matrix,
#                          interesting_metrics=interesting_metrics)
    
#     # Process networks in parallel
#     print("Starting multiprocessing...")
#     with Pool(processes=n_processes) as pool:
#         results = pool.map(worker_func, network_files)
    
#     # Filter out None results (errors)
#     results = [r for r in results if r is not None]
#     print(f"Successfully processed {len(results)} networks")
    
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
    
#     # Load original dataframe
#     df_original = pd.read_csv(df_path)
    
#     # Merge with original dataframe
#     # First, ensure eta and gamma are in the original df
#     merge_cols = ['eta', 'gamma']
#     # merge original and results on eta and gamma
#     df_merged = pd.merge(df_original, results_df, on=merge_cols, how='left', suffixes=('', '_new'))

#     # Update original dataframe with new metrics
#     for metric in interesting_metrics:
#         if metric in results_df.columns:
#             # Create a mapping from (eta, gamma) to metric value
#             metric_map = results_df.set_index(['eta', 'gamma'])[metric].to_dict()
            
#             # Update the original dataframe
#             for idx, row in df_original.iterrows():
#                 key = (row['eta'], row['gamma'])
#                 if key in metric_map:
#                     df_original.at[idx, metric] = metric_map[key]
    
#     # Save the updated dataframe
#     # df_original.to_csv(df_path_out, index=False)
#     results_df.to_csv(df_path_out, index=False)
#     print(f"Saved updated metrics to: {df_path_out}")
#     df_merged.to_csv(output_path / f"all_metrics_for_{experiment}_merged.csv", index=False)
    
#     # Clean up temp directory
#     if temp_dir.exists():
#         shutil.rmtree(temp_dir)
    
#     return results_df 


# # %%
# if __name__ == "__main__":

#     EXPERIMENT = "60_generally_finer_search_animal_0" # "70_mix_and_match_animal_0"
#     DATASET = "suarez_MaMI_dataset"
#     save_base_path = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{DATASET}/{EXPERIMENT}"
    
#     # Run the multiprocessing
#     df_updated = multiprocess_networks(n_processes=None, experiment=EXPERIMENT, dataset=DATASET)  # Uses cpu_count() - 1
#     df_updated.to_csv(save_base_path + f"/all_metrics_for_{EXPERIMENT}_hopefully_unnecessary.csv", index=False)
#     print("\nProcessing complete!")
#     print(f"Updated dataframe shape: {df_updated.shape}")
#     print(f"\nFirst few rows:\n{df_updated.head()}")
    
    
    
#     ############ 07_3 #####################
    
#     # --- 1. Define Input and Output ---
    
#     animal_id = 0
#     # List of CSV files containing the GNM sweep results.
#     df_paths = [ 
#         # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/53_suarez_MaMI_100_testing_with_100_sample_points_animal_19/all_metrics_for_53_suarez_MaMI_100_testing_with_100_sample_points_animal_19.csv"
        
#         # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/62_testing_conciser_code_animal_0/all_metrics_for_62_testing_conciser_code_animal_0.csv"
#         # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/63_fine_grid_animal_0/all_metrics_for_63_fine_grid_animal_0.csv"
#         # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/65_grid_higher_eta_animal_1/all_metrics_for_65_grid_higher_eta_animal_1.csv"
#         # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/70_mix_and_match_animal_0/all_metrics_for_70_mix_and_match_animal_0.csv"
#         # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/64_fine_grid_upper_local_minima_animal_0/all_metrics_for_64_fine_grid_upper_local_minima_animal_0.csv"
#         f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{DATASET}/{EXPERIMENT}/all_metrics_for_{EXPERIMENT}.csv"
#     ]       
#     plot_indiv_connectomes = False # True # False # True # False # True #  False # True # False
    
    
#     ##############
    
    
#     # The folder where the final figures will be saved.
#     # A new subfolder is used to keep these plots separate.
#     parent_folder = Path(df_paths[0]).parent
#     save_path = parent_folder / "figures_voronoi_only"
    
    
#     import os
#     import traceback
#     from pathlib import Path
#     import pandas as pd
#     import matplotlib

#     from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer

#     # --- 2. Define Metrics to Plot ---
    
#     lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 20, 49]
#     metrics_to_plot = [
        
#         # "MaxCriteria",
#         # "avg_communicability", 
#         # "global_efficiency", 
#         # "modularity", 
#         # "avg_clustering", 
#         # "avg_degree", 
#         # "transitivity", 
#         # "avg_edge_distance", 
#         # "char_path_length", 
#         # "richclub_n_edges", 
#         # "richclub_avg_length", 
#         # "mc_mean", 
#         # "mc_std", 
#         # "wiring_cost", 
#         # "mean_mc_divided_by_wiring_cost",
#         # "mc_5_divided_by_wiring_cost", 
#         "n_connected_components",
#     ]
#         # "avg_clustering_glob_efficiency_minus_energy", 
#         # "avg_clustering_divided_by_global_efficiency", 
#     # ] + [f"mc_{lag}" for lag in lags_to_plot]
#     plot_combined_lag_plot = False # Plot this comparison plot with different MC lags 
    
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

#         path_to_best_gamma_and_eta_estimations = parent_folder / "min_energy_results.csv"    # not entirely clean... TODO: Make this clean. 
#         if os.path.exists(path_to_best_gamma_and_eta_estimations) and plot_indiv_connectomes:
#             df_best_gamma_and_eta_estimates = pd.read_csv(path_to_best_gamma_and_eta_estimations)
                
#     except Exception as e:
#         print(f"ERROR: Failed to load data. {e}")
#         exit() # Exit if data cannot be loaded

#     visualizer = PipelineVisualizer() # Initialize the visualizer once
    
#     # --- 4. Generate Individual Plots in a Loop ---
#     for metric in metrics_to_plot:
#         print(f"--- Generating Voronoi plot for metric: {metric} ---")
#         try:
#             # --- Prepare data for the specific metric ---
#             df = gnm_results_df.copy()

#             if metric == "mean_mc_divided_by_wiring_cost": 
#                 df["mean_mc_divided_by_wiring_cost"] = pd.to_numeric(df["mc_mean"], errors='coerce') / pd.to_numeric(df["wiring_cost"], errors='coerce')
            
                        
#             if metric == "mc_5_divided_by_wiring_cost": 
#                 df["mc_5_divided_by_wiring_cost"] = pd.to_numeric(df["mc_5"], errors='coerce') / pd.to_numeric(df["wiring_cost"], errors='coerce')
            
            
#             if metric == "avg_clustering_divided_by_global_efficiency": 
#                 # df["avg_clustering_glob_efficiency"] = pd.to_numeric(df["avg_clustering"], errors='coerce') / pd.to_numeric(df["global_efficiency"], errors='coerce') * pd.to_numeric(df["modularity"], errors='coerce')
#                 upper_factor = pd.to_numeric(df["avg_clustering"], errors='coerce')
#                 # upper_factor = (upper_factor - upper_factor.min()) / (upper_factor.max() - upper_factor.min())
                
#                 lower_factor = pd.to_numeric(df["global_efficiency"], errors='coerce')
#                 # lower_factor = (lower_factor - lower_factor.min()) / (lower_factor.max() - lower_factor.min())
                
#                 total = upper_factor * lower_factor * (-1)
#                 total = (total - total.min()) / (total.max() - total.min()) 
                
#                 df["avg_clustering_divided_by_global_efficiency"] = total #  (total - energy) 
            
            
#             # if metric == "avg_clustering_glob_efficiency_minus_energy": 
#             #     # df["avg_clustering_glob_efficiency"] = pd.to_numeric(df["avg_clustering"], errors='coerce') / pd.to_numeric(df["global_efficiency"], errors='coerce') * pd.to_numeric(df["modularity"], errors='coerce')
#             #     upper_factor = pd.to_numeric(df["avg_clustering"], errors='coerce')
#             #     upper_factor = (upper_factor - upper_factor.min()) / (upper_factor.max() - upper_factor.min())
                
#             #     lower_factor = pd.to_numeric(df["global_efficiency"], errors='coerce')
#             #     lower_factor = (lower_factor - lower_factor.min()) / (lower_factor.max() - lower_factor.min())
                
#             #     energy = pd.to_numeric(df["MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)"], errors='coerce')
#             #     energy = (energy - energy.min()) / (energy.max() - energy.min())
                
#             #     total = upper_factor * lower_factor * (-1)
#             #     total = (total - total.min()) / (total.max() - total.min())
#             #     total = (total - energy) 
                
#             #     df["avg_clustering_glob_efficiency_minus_energy"] = total 
                
                
            
#             df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
#             df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
            
#             try:
#                 # Exact match for mc_lag columns
#                 if metric.startswith("mc_") and metric.split("_")[1].isdigit():
#                     metric_col_name = metric
#                     if metric not in df.columns:
#                         raise StopIteration
#                 else: # Flexible match for other metrics
#                     metric_col_name = next(col for col in df.columns if metric in col)
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

#             fig, ax = visualizer.plot_metric_landscape_voronoi(
#                 df, 
#                 title=plot_title,
#                 metric_name=metric_col_name,
#                 savepath=full_save_path,
#                 dot_color="steelblue", 
#                 eta_span=None, 
#                 gamma_span=None,
#                 show=False,
#                 show_dots=False,
#                 annotate_extremes=True, 
#                 estimated_indiv_connectomes=df_best_gamma_and_eta_estimates if plot_indiv_connectomes else None, 
#             )
            
#             matplotlib.pyplot.close() # Close the plot after saving to free memory, if returned fig and ax are not used.
            
            
#             print(f"Successfully generated plot for {metric} at {full_save_path}\n")
            
            
#                 # plt.show() 
            
            
#             # matplotlib.pyplot.close() # Close the plot after saving to free memory. 
            
            
#         except Exception as e:
#             print(f"ERROR: An unexpected error occurred while plotting '{metric}'.")
#             traceback.print_exc()
#             print("\n")


#     # --- 5. Generate Combined MC Lag Plot ---
#     if plot_combined_lag_plot: 
#         print("--- Generating combined Voronoi plot for all MC lags ---")
#         try:
#             save_dir = Path(save_path)
#             save_dir.mkdir(parents=True, exist_ok=True)
#             # Define a base name for the save file, the function will add the extension
#             figure_save_name = "voronoi_landscape_mc_lags_grid"
#             full_save_path = save_dir / figure_save_name

#             mc_cols_exist = any(f"mc_{lag}" in gnm_results_df.columns for lag in lags_to_plot)

#             if mc_cols_exist:
#                 visualizer.plot_mc_lag_landscapes_voronoi(
#                     gnm_results_df,
#                     lags=lags_to_plot,
#                     n_cols=4, # As requested: 4 columns
#                     savepath=full_save_path,
#                     save_format="pdf"
#                 )
#                 matplotlib.pyplot.close() # Close the plot after saving to free memory, if generated figure is not used. 

#                 print("Successfully generated combined MC lag plot.\n")
#             else:
#                 print("SKIPPING: No 'mc_lag' columns found to generate a combined plot.\n")

#         except Exception as e:
#             print(f"ERROR: An unexpected error occurred while plotting the combined MC lags.")
#             traceback.print_exc()
#             print("\n")


#     print("--- Visualization process completed. ---")
# %%
import numpy as np 
import networkx as nx
import pandas as pd
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import signal
import sys
import gc  # Garbage collector

from src.analysis.metric_calculators import StaticMetricCalculator               
    
from src.analysis.utils import get_eta_and_gamma_from_filename, sorted_listing_by_creation_time

# %%
EXPERIMENT = "60_generally_finer_search_animal_0"
DATASET = "suarez_MaMI_dataset"
BASE_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
OUTPUT_PATH = BASE_PATH / "output" / "gnm" / DATASET / EXPERIMENT

generated_networks_dir = OUTPUT_PATH / "generated_networks"

df_path = OUTPUT_PATH / f"all_metrics_for_{EXPERIMENT}.csv"
df = pd.read_csv(df_path)

df_path_out = OUTPUT_PATH / f"all_metrics_for_{EXPERIMENT}_updated.csv"

distance_matrix_path = BASE_PATH / "data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy"
distance_matrix = np.load(distance_matrix_path)

# Global variable to track results for interrupt handling
accumulated_results = []



# Worker function to process a single network file
def process_network_file(file_info, distance_matrix_path, interesting_metrics):
    """
    Process a single network file and return a dictionary of results.
    
    Args:
        file_info: tuple of (index, filename, full_path)
        distance_matrix_path: path to distance matrix (loaded per worker to save memory)
        interesting_metrics: list of metric names to calculate
    
    Returns:
        dict with eta, gamma, and all calculated metrics
    """
    idx, name, full_path = file_info
    
    try:
        # Load the network
        A = np.load(full_path)[0]  # Load the numpy array (mostly: 1, 100, 100)
        eta, gamma = get_eta_and_gamma_from_filename(name)
        
        # Load distance matrix only if needed (currently not used by metrics)
        # distance_matrix = np.load(distance_matrix_path)
        
        # Initialize calculator
        static_metric_calculator = StaticMetricCalculator(A=A, distance_matrix=None)
        
        # Calculate all metrics
        results = {"eta": eta, "gamma": gamma}
        for metric in interesting_metrics:
            results[metric] = static_metric_calculator.calculate_metric(metric)
        
        # Clean up
        del A
        del static_metric_calculator
        gc.collect()  # Force garbage collection
        
        if (idx + 1) % 50 == 0:  # Print every 50 files
            print(f"Processed file {idx+1}: {name}")
        
        return results
    
    except Exception as e:
        print(f"Error processing {name}: {e}")
        return None


def save_results_to_csv(results_list, df_original, df_path_out, interesting_metrics):
    """Save accumulated results to CSV"""
    if not results_list:
        print("No results to save")
        return df_original
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    
    print(f"\nSaving {len(results_df)} processed networks...")
    
    # Create a copy of the original dataframe
    df_updated = df_original.copy()
    
    # Ensure metric columns exist
    for metric in interesting_metrics:
        if metric not in df_updated.columns:
            df_updated[metric] = np.nan
    
    # Update rows that match eta and gamma
    for _, result_row in results_df.iterrows():
        eta = result_row['eta']
        gamma = result_row['gamma']
        
        # Find matching rows in original dataframe
        mask = (df_updated['eta'] == eta) & (df_updated['gamma'] == gamma)
        
        if mask.any():
            # Update existing row
            for metric in interesting_metrics:
                if metric in result_row:
                    df_updated.loc[mask, metric] = result_row[metric]
        else:
            # Add new row if it doesn't exist
            new_row = {col: np.nan for col in df_updated.columns}
            new_row['eta'] = eta
            new_row['gamma'] = gamma
            for metric in interesting_metrics:
                if metric in result_row:
                    new_row[metric] = result_row[metric]
            df_updated = pd.concat([df_updated, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the updated dataframe
    df_updated.to_csv(df_path_out, index=False)
    print(f"Saved to: {df_path_out}")
    
    return df_updated


def signal_handler(signum, frame, df_original, df_path_out, interesting_metrics):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Interrupt received! Saving progress before exiting...")
    save_results_to_csv(accumulated_results, df_original, df_path_out, interesting_metrics)
    print("✓ Progress saved. Exiting.")
    sys.exit(0)


# %%
def multiprocess_networks(n_processes=None, save_interval=50):
    """
    Main function to multiprocess network metric calculations.
    
    Args:
        n_processes: Number of processes to use. If None, uses cpu_count() - 1
        save_interval: Save progress every N completed files
    """
    global accumulated_results
    accumulated_results = []
    
    # Define metrics to calculate
    interesting_metrics = ["density", 
                           "compute_structural_complexity", 
                           "n_connected_components"
                           ]
    
    # Get all network files in order
    all_files = sorted_listing_by_creation_time(generated_networks_dir)
    network_files = [(i, name, generated_networks_dir / name) 
                     for i, name in enumerate(all_files) 
                     if name.endswith(".npy")]
    
    print(f"Found {len(network_files)} network files to process")
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    print(f"Using {n_processes} processes")
    
    # Load original dataframe
    df_original = pd.read_csv(df_path)
    
    # Set up interrupt handler with partial function
    handler = partial(signal_handler, 
                     df_original=df_original, 
                     df_path_out=df_path_out, 
                     interesting_metrics=interesting_metrics)
    signal.signal(signal.SIGINT, handler)
    
    # Create partial function - pass PATH not the matrix itself to save memory
    worker_func = partial(process_network_file, 
                         distance_matrix_path=distance_matrix_path,
                         interesting_metrics=interesting_metrics)
    
    # Process networks in parallel
    print("Starting multiprocessing... (Press Ctrl+C to stop and save progress)")
    print("-" * 60)
    
    try:
        with Pool(processes=n_processes) as pool:
            # Use imap to get results as they complete (preserves order)
            for i, result in enumerate(pool.imap(worker_func, network_files, chunksize=1)):
                if result is not None:
                    accumulated_results.append(result)
                
                # Save periodically and clear memory
                if (i + 1) % save_interval == 0:
                    df_updated = save_results_to_csv(accumulated_results, df_original, 
                                                     df_path_out, interesting_metrics)
                    print(f"Progress: {i+1}/{len(network_files)} files processed")
                    print(f"Memory checkpoint - forcing garbage collection\n")
                    gc.collect()  # Force garbage collection
        
        # Final save
        print("\n" + "=" * 60)
        print("All processing complete!")
        df_updated = save_results_to_csv(accumulated_results, df_original, 
                                        df_path_out, interesting_metrics)
        
    except KeyboardInterrupt:
        # This shouldn't normally trigger due to signal handler, but just in case
        print("\n\nInterrupted! Saving progress...")
        df_updated = save_results_to_csv(accumulated_results, df_original, 
                                        df_path_out, interesting_metrics)
    
    finally:
        # Clean up
        gc.collect()
    
    print(f"\nFinal results: Processed {len(accumulated_results)} networks")
    print(f"Output saved to: {df_path_out}")
    
    return df_updated


# %%
if __name__ == "__main__":
    # Run the multiprocessing
    df_updated = multiprocess_networks(n_processes=None, save_interval=50)
    print("\n✓ Processing complete!")
    print(f"Updated dataframe shape: {df_updated.shape}")
    
    # Final cleanup
    gc.collect()
