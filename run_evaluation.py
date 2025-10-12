# # """
# # Compare generated and empirical networks with another and creates big csv file. 
# # Optimized with batching and efficient multiprocessing.
# # """

# # import numpy as np
# # import pandas as pd
# # import torch
# # from pathlib import Path
# # from tqdm import tqdm
# # import re
# # from typing import Dict, List, Tuple

# # # Import the evaluation criteria function
# # from src.config.GNM import create_evaluation_criteria
# # from multiprocessing import Pool

# # def extract_params_from_filename(filename: str) -> Dict[str, float]:
# #     """
# #     Extract eta and gamma parameters from network filename.
# #     Expected format: net_eta{value}_gamma{value}_rule{name}.npy
    
# #     Args:
# #         filename: Network filename
        
# #     Returns:
# #         Dictionary with eta and gamma values
# #     """
# #     match = re.search(r'eta([-\d.]+)_gamma([-\d.]+)', filename)
# #     if match:
# #         eta = float(match.group(1))
# #         gamma = float(match.group(2))
# #         return {'eta': eta, 'gamma': gamma}
# #     else:
# #         raise ValueError(f"Could not extract parameters from filename: {filename}")


# # def load_generated_networks(networks_dir: Path) -> List[Tuple[Dict, np.ndarray]]:
# #     """
# #     Load all generated networks from directory.
    
# #     Args:
# #         networks_dir: Directory containing .npy network files
        
# #     Returns:
# #         List of tuples (params_dict, network_array)
# #     """
# #     networks_dir = Path(networks_dir)
# #     network_files = sorted(networks_dir.glob("net_eta*.npy"))
    
# #     if not network_files:
# #         raise FileNotFoundError(f"No network files found in {networks_dir}")
    
# #     print(f"Found {len(network_files)} generated network files")
    
# #     networks = []
# #     network_parameters = []
# #     for net_file in network_files:
# #         try:
# #             params = extract_params_from_filename(net_file.name)
# #             network = np.load(net_file)
# #             networks.append(network)
# #             network_parameters.append(params)
# #         except Exception as e:
# #             print(f"Warning: Could not load {net_file.name}: {e}")
# #             continue
# #     networks = np.concatenate(networks)
    
# #     return networks, network_parameters


# # def load_empirical_networks(empirical_path: Path) -> np.ndarray:
# #     """
# #     Load empirical networks.
    
# #     Args:
# #         empirical_path: Path to .npy file with shape (n_subjects, n_nodes, n_nodes)
        
# #     Returns:
# #         Empirical networks array
# #     """
# #     empirical_path = Path(empirical_path)
    
# #     if not empirical_path.exists():
# #         raise FileNotFoundError(f"Empirical networks file not found: {empirical_path}")
    
# #     empirical_networks = np.load(empirical_path)
    
# #     print(f"Loaded empirical networks with shape: {empirical_networks.shape}")
# #     print(f"Number of subjects: {empirical_networks.shape[0]}")
    
# #     return empirical_networks


# # def create_evaluation_criteria_list(
# #     distance_matrices: torch.Tensor,
# #     config_dict: Dict = None
# # ) -> List:
# #     """
# #     Pre-create evaluation criteria for each distance matrix to avoid recreation overhead.
    
# #     Args:
# #         distance_matrices: Tensor of shape (n_subjects, n_nodes, n_nodes)
# #         config_dict: Optional config dictionary
        
# #     Returns:
# #         List of evaluation criteria objects
# #     """
# #     evaluation_criteria_list = []
    
# #     print("Creating evaluation criteria for each subject...")
# #     for i in tqdm(range(len(distance_matrices)), desc="Building criteria"):
# #         distance_matrix = distance_matrices[i]
        
# #         if config_dict is not None:
# #             criteria = create_evaluation_criteria(
# #                 config=config_dict,
# #                 distance_matrix=distance_matrix
# #             )
# #         else:
# #             from gnm import evaluation
# #             criteria = evaluation.MaxCriteria(
# #                 evaluation.DegreeKS(),
# #                 evaluation.ClusteringKS(),
# #                 evaluation.EdgeLengthKS(distance_matrix),
# #                 evaluation.BetweennessKS()
# #             )
        
# #         evaluation_criteria_list.append(criteria)
    
# #     return evaluation_criteria_list


# # def evaluate_network_against_empirical_optimized(
# #     generated_network_tensor: torch.Tensor,
# #     empirical_networks_tensor: torch.Tensor,
# #     evaluation_criteria_list: List,
# # ) -> Dict[str, float]:
# #     """
# #     Evaluate a single generated network against all empirical networks.
# #     Optimized version using pre-created criteria and torch tensors.
    
# #     Args:
# #         generated_network_tensor: Single generated network tensor (1, n_nodes, n_nodes)
# #         empirical_networks_tensor: Empirical networks tensor (n_subjects, n_nodes, n_nodes)
# #         evaluation_criteria_list: Pre-created evaluation criteria for each subject
        
# #     Returns:
# #         Dictionary with energy values for each empirical network
# #     """
# #     n_subjects = empirical_networks_tensor.shape[0]
# #     results = {}
    
# #     # Evaluate against each empirical network
# #     for subject_idx in range(n_subjects):
# #         empirical_single = empirical_networks_tensor[subject_idx:subject_idx+1, :, :]
        
# #         # Use pre-created evaluation criteria
# #         evaluation_criteria = evaluation_criteria_list[subject_idx]
        
# #         # Calculate energy
# #         energy_dict = evaluation_criteria(generated_network_tensor, empirical_single)
        
# #         # Store results for this subject
# #         for energy_value in energy_dict:
# #             column_name = f"MaxCrit(etc)_indiv_{subject_idx}"
# #             results[column_name] = float(energy_value.mean().item())
    
# #     return results


# # def process_network_optimized(args: Tuple) -> Dict:
# #     """
# #     Worker function to evaluate a single network against empirical data.
# #     Optimized version that receives pre-converted tensors and pre-created criteria.
# #     """
# #     network, generated_parameters, empirical_networks_tensor, evaluation_criteria_list = args
    
# #     # Convert generated network to tensor
# #     gen_network_tensor = torch.tensor(network, dtype=torch.float32).unsqueeze(0)
    
# #     # Evaluate this network against all empirical networks
# #     energy_results = evaluate_network_against_empirical_optimized(
# #         generated_network_tensor=gen_network_tensor,
# #         empirical_networks_tensor=empirical_networks_tensor,
# #         evaluation_criteria_list=evaluation_criteria_list,
# #     )
    
# #     # Combine parameters with results and return the row
# #     return {**generated_parameters, **energy_results}


# # def compare_all_networks(
# #     generated_networks: np.ndarray,
# #     generated_network_parameters: List[Dict], 
# #     empirical_networks: np.ndarray,
# #     distance_matrices_path: Path = None,
# #     config_dict: Dict = None,
# #     output_path: Path = None,
# #     checkpoint_interval: int = 100
# # ) -> pd.DataFrame:
# #     """
# #     Compare all generated networks with empirical networks using optimized multiprocessing.
    
# #     Args:
# #         generated_networks: Array of generated networks
# #         generated_network_parameters: List of parameter dicts
# #         empirical_networks: Empirical networks array
# #         distance_matrices_path: Path to distance matrix file
# #         config_dict: Optional config dictionary
# #         output_path: Path to save results CSV
# #         checkpoint_interval: Save checkpoint every N networks
        
# #     Returns:
# #         DataFrame with comparison results
# #     """
    
# #     # Load distance matrices once
# #     distance_matrices = None
# #     evaluation_criteria_list = None
    
# #     if distance_matrices_path and Path(distance_matrices_path).exists():
# #         print("Loading distance matrices...")
# #         distance_matrices = np.load(distance_matrices_path)
# #         distance_matrices = torch.tensor(distance_matrices, dtype=torch.float32)
# #         print(f"Loaded distance matrix with shape: {distance_matrices.shape}")
        
# #         # Pre-create evaluation criteria for all subjects
# #         evaluation_criteria_list = create_evaluation_criteria_list(
# #             distance_matrices=distance_matrices,
# #             config_dict=config_dict
# #         )
# #     else:
# #         print("No distance matrices provided, creating criteria without edge length...")
# #         # Create evaluation criteria without distance matrix
# #         from gnm import evaluation
# #         n_subjects = empirical_networks.shape[0]
# #         evaluation_criteria_list = []
# #         for _ in range(n_subjects):
# #             criteria = evaluation.MaxCriteria(
# #                 evaluation.DegreeKS(),
# #                 evaluation.ClusteringKS(),
# #                 evaluation.BetweennessKS()
# #             )
# #             evaluation_criteria_list.append(criteria)
    
# #     # Convert empirical networks to torch tensor once
# #     print("Converting empirical networks to torch tensors...")
# #     empirical_networks_tensor = torch.tensor(empirical_networks, dtype=torch.float32)
    
# #     # Prepare tasks (one per generated network, not per combination!)
# #     print("Preparing tasks...")
# #     tasks = []
# #     for i in range(len(generated_networks)):
# #         task_args = (
# #             generated_networks[i],
# #             generated_network_parameters[i],
# #             empirical_networks_tensor,
# #             evaluation_criteria_list
# #         )
# #         tasks.append(task_args)
    
# #     print(f"\nComparing {len(generated_networks)} generated networks with {empirical_networks.shape[0]} empirical networks using multiprocessing...")
# #     print(f"Total tasks: {len(tasks)} (one per generated network)")
    
# #     # Use multiprocessing Pool to process tasks in parallel
# #     all_results = []
    
# #     with Pool() as pool:
# #         # Use tqdm to show progress with imap
# #         for i, result in enumerate(tqdm(
# #             pool.imap(process_network_optimized, tasks), 
# #             total=len(tasks), 
# #             desc="Evaluating networks"
# #         )):
# #             all_results.append(result)
            
# #             # Save checkpoint periodically
# #             if output_path and (i + 1) % checkpoint_interval == 0:
# #                 checkpoint_path = output_path.with_suffix('.checkpoint.csv')
# #                 pd.DataFrame(all_results).to_csv(checkpoint_path, index=False, na_rep="nan")
# #                 print(f"\n💾 Checkpoint saved at {i + 1}/{len(tasks)} networks")
    
# #     # Create DataFrame
# #     df_results = pd.DataFrame(all_results)
    
# #     # Sort by eta and gamma for easier reading
# #     df_results = df_results.sort_values(['eta', 'gamma']).reset_index(drop=True)
    
# #     # Save final results
# #     if output_path:
# #         output_path = Path(output_path)
# #         output_path.parent.mkdir(parents=True, exist_ok=True)
# #         df_results.to_csv(output_path, index=False, na_rep="nan")
# #         print(f"\n✅ Results saved to: {output_path}")
        
# #         # Remove checkpoint file if it exists
# #         checkpoint_path = output_path.with_suffix('.checkpoint.csv')
# #         if checkpoint_path.exists():
# #             checkpoint_path.unlink()
# #             print(f"🗑️  Removed checkpoint file")
    
# #     return df_results


# # def main():
# #     """Main execution function."""
    
# #     # Define paths
# #     dataset_name = "suarez_MaMI_dataset"
# #     experiment_name = "31_suarez_MaMI_size_100_copy" # 31_suarez_MaMI_size_100_wider_sweep_220"
# #     # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/31_suarez_MaMI_size_100_copy/all_metrics_for_exp_30_shafiei_size_68.csv
# #     path_experiment = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm") / dataset_name / experiment_name
# #     generated_networks_dir = path_experiment / "generated_networks"
# #     empirical_networks_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / "01_connectomes/01_consensus_bin_density_10_percent_100.npy"
    
# #     # Distance matrix path
# #     distance_matrix_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / "02_distance_matrices/distance_matrix_100.npy"
    
# #     # Optional: Load a config for custom evaluation metrics
# #     config_dict = {
# #         'gnm': {
# #             'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
# #         }
# #     }
    
# #     # Output path
# #     output_dir = generated_networks_dir.parent
# #     experiment_name = generated_networks_dir.parent.name
# #     output_path = output_dir / f"summary_indiv_energies_for_exp_{experiment_name}.csv"
    
# #     try:
# #         # Load networks
# #         print("\n" + "=" * 60)
# #         print("LOADING NETWORKS")
# #         print("=" * 60)
        
# #         generated_networks, generated_network_parameters = load_generated_networks(generated_networks_dir)
# #         empirical_networks = load_empirical_networks(empirical_networks_path)
        
# #         # Validate dimensions
# #         if generated_networks is not None:
# #             if generated_networks.shape[-1] != empirical_networks.shape[-1]:
# #                 raise ValueError(
# #                     f"Dimension mismatch: Generated networks have {generated_networks.shape[-1]} nodes "
# #                     f"but empirical networks have {empirical_networks.shape[-1]} nodes."
# #                 )
        
# #         # Run comparison
# #         print("\n" + "=" * 60)
# #         print("COMPARING NETWORKS")
# #         print("=" * 60)
        
# #         results_df = compare_all_networks(
# #             generated_networks=generated_networks,
# #             generated_network_parameters=generated_network_parameters, 
# #             empirical_networks=empirical_networks,
# #             distance_matrices_path=distance_matrix_path if distance_matrix_path.exists() else None,
# #             config_dict=config_dict,
# #             output_path=output_path,
# #             checkpoint_interval=100  # Save every 100 networks
# #         )
        
# #         # Display summary
# #         print("\n" + "=" * 60)
# #         print("SUMMARY")
# #         print("=" * 60)
# #         print(f"Total comparisons: {len(results_df)}")
# #         print(f"Parameter ranges:")
# #         print(f"  eta: [{results_df['eta'].min():.3f}, {results_df['eta'].max():.3f}]")
# #         print(f"  gamma: [{results_df['gamma'].min():.3f}, {results_df['gamma'].max():.3f}]")
# #         print(f"\nFirst few rows of results:")
# #         print(results_df.head())
        
# #         print("\n✅ Comparison complete!")
        
# #         return results_df
        
# #     except Exception as e:
# #         print(f"\n❌ Error during comparison: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         return None


# # if __name__ == "__main__":
# #     results = main()

# """
# Compare generated and empirical networks with another and creates big csv file. 
# Optimized version with batching, efficient multiprocessing, and resumption capability.
# """

# import numpy as np
# import pandas as pd
# import torch
# from pathlib import Path
# from tqdm import tqdm
# import re
# from typing import Dict, List, Tuple, Optional
# import csv

# # Import the evaluation criteria function
# from src.config.GNM import create_evaluation_criteria
# from gnm.fitting import RunConfig
# from gnm.model import BinaryGenerativeParameters
# from multiprocessing import Pool

# def extract_params_from_filename(filename: str) -> Dict[str, float]:
#     """
#     Extract eta and gamma parameters from network filename.
#     Expected format: net_eta{value}_gamma{value}_rule{name}.npy
    
#     Args:
#         filename: Network filename
        
#     Returns:
#         Dictionary with eta and gamma values
#     """
#     match = re.search(r'eta([-\d.]+)_gamma([-\d.]+)', filename)
#     if match:
#         eta = float(match.group(1))
#         gamma = float(match.group(2))
#         return {'eta': eta, 'gamma': gamma}
#     else:
#         raise ValueError(f"Could not extract parameters from filename: {filename}")


# def load_generated_networks(networks_dir: Path) -> Tuple[np.ndarray, List[Dict], List[str]]:
#     """
#     Load all generated networks from directory.
    
#     Args:
#         networks_dir: Directory containing .npy network files
        
#     Returns:
#         Tuple of (networks array, parameters list, filenames list)
#     """
#     networks_dir = Path(networks_dir)
#     network_files = sorted(networks_dir.glob("net_eta*.npy"))
    
#     if not network_files:
#         raise FileNotFoundError(f"No network files found in {networks_dir}")
    
#     print(f"Found {len(network_files)} generated network files")
    
#     networks = []
#     network_parameters = []
#     filenames = []
    
#     for net_file in network_files:
#         try:
#             params = extract_params_from_filename(net_file.name)
#             network = np.load(net_file)
#             networks.append(network)
#             network_parameters.append(params)
#             filenames.append(net_file.name)
#         except Exception as e:
#             print(f"Warning: Could not load {net_file.name}: {e}")
#             continue
    
#     networks = np.concatenate(networks)
    
#     return networks, network_parameters, filenames


# def load_empirical_networks(empirical_path: Path) -> np.ndarray:
#     """
#     Load empirical networks.
    
#     Args:
#         empirical_path: Path to .npy file with shape (n_subjects, n_nodes, n_nodes)
        
#     Returns:
#         Empirical networks array
#     """
#     empirical_path = Path(empirical_path)
    
#     if not empirical_path.exists():
#         raise FileNotFoundError(f"Empirical networks file not found: {empirical_path}")
    
#     empirical_networks = np.load(empirical_path)
    
#     print(f"Loaded empirical networks with shape: {empirical_networks.shape}")
#     print(f"Number of subjects: {empirical_networks.shape[0]}")
    
#     return empirical_networks


# def load_existing_results(output_path: Path) -> Tuple[Optional[pd.DataFrame], set]:
#     """
#     Load existing results if they exist and determine which networks have been processed.
    
#     Args:
#         output_path: Path to results CSV file
        
#     Returns:
#         Tuple of (existing DataFrame or None, set of processed (eta, gamma) tuples)
#     """
#     if not output_path.exists():
#         return None, set()
    
#     try:
#         df = pd.read_csv(output_path)
#         if df.empty or 'eta' not in df.columns or 'gamma' not in df.columns:
#             return None, set()
        
#         # Create set of processed parameter combinations
#         processed = set(zip(df['eta'].round(10), df['gamma'].round(10)))
#         print(f"📊 Found existing results with {len(df)} rows")
#         print(f"   Already processed {len(processed)} unique parameter combinations")
        
#         return df, processed
#     except Exception as e:
#         print(f"Warning: Could not load existing results: {e}")
#         return None, set()


# def get_ordered_tasks(
#     generated_networks: np.ndarray,
#     generated_network_parameters: List[Dict],
#     filenames: List[str],
#     existing_results_df: Optional[pd.DataFrame],
#     processed_params: set, 
#     reference_df: Optional[pd.DataFrame] = None, 
# ) -> List[Tuple[int, np.ndarray, Dict, str]]:
#     """
#     Create ordered list of tasks, following the order from existing results if available.
#     Skip already processed networks.
    
#     Args:
#         generated_networks: Array of networks
#         generated_network_parameters: List of parameter dicts
#         filenames: List of network filenames
#         existing_results_df: Existing results DataFrame or None
#         processed_params: Set of already processed (eta, gamma) tuples
        
#     Returns:
#         List of (index, network, params, filename) tuples in proper order
#     """
#     tasks = []
    
#     # Create mapping from (eta, gamma) to network data
#     network_map = {}
#     for i, (params, filename) in enumerate(zip(generated_network_parameters, filenames)):
#         key = (round(params['eta'], 10), round(params['gamma'], 10))
#         network_map[key] = (i, generated_networks[i], params, filename)
    
#     # # If we have existing results, follow their order first
#     # if existing_results_df is not None and not existing_results_df.empty:
#     #     # Get unique parameter combinations in their existing order
#     #     existing_params = list(zip(
#     #         existing_results_df['eta'].round(10),
#     #         existing_results_df['gamma'].round(10)
#     #     ))
        
#     #     # Add networks in the order they appear in existing results (skip processed ones)
#     #     seen_keys = set()
#     #     for key in existing_params:
#     #         if key in network_map and key not in seen_keys:
#     #             if key not in processed_params:
#     #                 tasks.append(network_map[key])
#     #             seen_keys.add(key)
    
#     # # Add any new networks not in existing results
#     # for key, data in network_map.items():
#     #     if key not in processed_params:
#     #         if not any(task[2]['eta'] == data[2]['eta'] and 
#     #                   task[2]['gamma'] == data[2]['gamma'] for task in tasks):
#     #             tasks.append(data)
    
#     # return tasks


#     # Prioritize reference CSV for ordering
#     order_source = reference_df if reference_df is not None else existing_results_df
    
#     # If we have a reference or existing results, follow their order first
#     if order_source is not None and not order_source.empty:
#         # Get unique parameter combinations in their existing order
#         existing_params = list(zip(
#             order_source['eta'].round(10),
#             order_source['gamma'].round(10)
#         ))
        
#         # Add networks in the order they appear (skip processed ones)
#         seen_keys = set()
#         for key in existing_params:
#             if key in network_map and key not in seen_keys:
#                 if key not in processed_params:
#                     tasks.append(network_map[key])
#                 seen_keys.add(key)
    
#     # Add any new networks not in reference/existing results
#     for key, data in network_map.items():
#         if key not in processed_params:
#             if not any(task[2]['eta'] == data[2]['eta'] and 
#                       task[2]['gamma'] == data[2]['gamma'] for task in tasks):
#                 tasks.append(data)
    
#     return tasks


# def create_evaluation_criteria_list(
#     distance_matrices: torch.Tensor,
#     config_dict: Dict = None
# ) -> List:
#     """
#     Pre-create evaluation criteria for each distance matrix to avoid recreation overhead.
    
#     Args:
#         distance_matrices: Tensor of shape (n_subjects, n_nodes, n_nodes)
#         config_dict: Optional config dictionary
        
#     Returns:
#         List of evaluation criteria objects
#     """
#     evaluation_criteria_list = []
    
#     print("Creating evaluation criteria for each subject...")
#     for i in tqdm(range(len(distance_matrices)), desc="Building criteria"):
#         distance_matrix = distance_matrices[i]
        
#         if config_dict is not None:
#             criteria = create_evaluation_criteria(
#                 config=config_dict,
#                 distance_matrix=distance_matrix
#             )
#         else:
#             from gnm import evaluation
#             criteria = evaluation.MaxCriteria(
#                 evaluation.DegreeKS(),
#                 evaluation.ClusteringKS(),
#                 evaluation.EdgeLengthKS(distance_matrix),
#                 evaluation.BetweennessKS()
#             )
        
#         evaluation_criteria_list.append(criteria)
    
#     return evaluation_criteria_list


# def evaluate_network_against_empirical_optimized(
#     generated_network_tensor: torch.Tensor,
#     empirical_networks_tensor: torch.Tensor,
#     evaluation_criteria_list: List,
# ) -> Dict[str, float]:
#     """
#     Evaluate a single generated network against all empirical networks.
#     Optimized version using pre-created criteria and torch tensors.
    
#     Args:
#         generated_network_tensor: Single generated network tensor (1, n_nodes, n_nodes)
#         empirical_networks_tensor: Empirical networks tensor (n_subjects, n_nodes, n_nodes)
#         evaluation_criteria_list: Pre-created evaluation criteria for each subject
        
#     Returns:
#         Dictionary with energy values for each empirical network
#     """
#     n_subjects = empirical_networks_tensor.shape[0]
#     results = {}
    
#     # Evaluate against each empirical network
#     for subject_idx in range(n_subjects):
#         empirical_single = empirical_networks_tensor[subject_idx:subject_idx+1, :, :]
        
#         # Use pre-created evaluation criteria
#         evaluation_criteria = evaluation_criteria_list[subject_idx]
        
#         # Calculate energy
#         energy_dict = evaluation_criteria(generated_network_tensor, empirical_single)
        
#         # Store results for this subject
#         for energy_value in energy_dict:
#             column_name = f"MaxCrit(etc)_indiv_{subject_idx}"
#             results[column_name] = float(energy_value.mean().item())
    
#     return results


# def process_network_optimized(args: Tuple) -> Dict:
#     """
#     Worker function to evaluate a single network against empirical data.
#     Optimized version that receives pre-converted tensors and pre-created criteria.
#     """
#     network_idx, network, generated_parameters, filename, empirical_networks_tensor, evaluation_criteria_list = args
    
#     # Convert generated network to tensor
#     gen_network_tensor = torch.tensor(network, dtype=torch.float32).unsqueeze(0)
    
#     # Evaluate this network against all empirical networks
#     energy_results = evaluate_network_against_empirical_optimized(
#         generated_network_tensor=gen_network_tensor,
#         empirical_networks_tensor=empirical_networks_tensor,
#         evaluation_criteria_list=evaluation_criteria_list,
#     )
    
#     # Combine parameters with results and metadata
#     result = {
#         'network_index': network_idx,
#         'filename': filename,
#         **generated_parameters, 
#         **energy_results
#     }
    
#     return result


# def append_result_to_csv(result: Dict, output_path: Path, write_header: bool = False):
#     """
#     Append a single result to the CSV file.
    
#     Args:
#         result: Dictionary with result data
#         output_path: Path to output CSV
#         write_header: Whether to write header (for first row)
#     """
#     mode = 'w' if write_header else 'a'
    
#     with open(output_path, mode, newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=result.keys())
#         if write_header:
#             writer.writeheader()
#         writer.writerow(result)


# def compare_all_networks(
#     generated_networks: np.ndarray,
#     generated_network_parameters: List[Dict],
#     filenames: List[str],
#     empirical_networks: np.ndarray,
#     distance_matrices_path: Path = None,
#     config_dict: Dict = None,
#     reference_csv_path: Path = None, 
#     output_path: Path = None
# ) -> pd.DataFrame:
#     """
#     Compare all generated networks with empirical networks using optimized multiprocessing.
#     Supports resumption and appends results line-by-line.
    
#     Args:
#         generated_networks: Array of generated networks
#         generated_network_parameters: List of parameter dicts
#         filenames: List of network filenames
#         empirical_networks: Empirical networks array
#         distance_matrices_path: Path to distance matrix file
#         config_dict: Optional config dictionary
#         output_path: Path to save results CSV
        
#     Returns:
#         DataFrame with comparison results
#     """
    
#     # Check for existing results and determine what's already processed
#     existing_df, processed_params = load_existing_results(output_path)
    
#     # Load reference CSV for parameter ordering if provided**
#     reference_df = None
#     if reference_csv_path and reference_csv_path.exists():
#         try:
#             reference_df = pd.read_csv(reference_csv_path)
#             print(f"📖 Using parameter order from: {reference_csv_path.name}")
#         except Exception as e:
#             print(f"Warning: Could not load reference CSV: {e}")
#             reference_df = None
    
#     # Get ordered tasks, skipping already processed ones (reference_df is passed to get ordered tasks) 
#     ordered_tasks_data = get_ordered_tasks(
#         generated_networks=generated_networks,
#         generated_network_parameters=generated_network_parameters,
#         filenames=filenames,
#         existing_results_df=existing_df,
#         processed_params=processed_params,
#         reference_df=reference_df 
#     )

#     if not ordered_tasks_data:
#         print("✅ All networks have already been processed!")
#         if existing_df is not None:
#             return existing_df
#         else:
#             return pd.DataFrame()
    
#     print(f"\n🔄 {len(ordered_tasks_data)} networks to process ({len(processed_params)} already done)")
    
#     # Load distance matrices once
#     distance_matrices = None
#     evaluation_criteria_list = None
    
#     if distance_matrices_path and Path(distance_matrices_path).exists():
#         print("Loading distance matrices...")
#         distance_matrices = np.load(distance_matrices_path)
#         distance_matrices = torch.tensor(distance_matrices, dtype=torch.float32)
#         print(f"Loaded distance matrix with shape: {distance_matrices.shape}")
        
#         # Pre-create evaluation criteria for all subjects
#         evaluation_criteria_list = create_evaluation_criteria_list(
#             distance_matrices=distance_matrices,
#             config_dict=config_dict
#         )
#     else:
#         print("No distance matrices provided, creating criteria without edge length...")
#         # Create evaluation criteria without distance matrix
#         from gnm import evaluation
#         n_subjects = empirical_networks.shape[0]
#         evaluation_criteria_list = []
#         for _ in range(n_subjects):
#             criteria = evaluation.MaxCriteria(
#                 evaluation.DegreeKS(),
#                 evaluation.ClusteringKS(),
#                 evaluation.BetweennessKS()
#             )
#             evaluation_criteria_list.append(criteria)
    
#     # Convert empirical networks to torch tensor once
#     print("Converting empirical networks to torch tensors...")
#     empirical_networks_tensor = torch.tensor(empirical_networks, dtype=torch.float32)
    
#     # Prepare tasks
#     print("Preparing tasks...")
#     tasks = []
#     for net_idx, network, params, filename in ordered_tasks_data:
#         task_args = (
#             net_idx,
#             network,
#             params,
#             filename,
#             empirical_networks_tensor,
#             evaluation_criteria_list
#         )
#         tasks.append(task_args)
    
#     print(f"\nProcessing {len(tasks)} networks...")
    
#     # Determine if we need to write header
#     write_header = existing_df is None or not output_path.exists()
    
#     # Use multiprocessing Pool to process tasks in parallel
#     results_count = 0
    
#     with Pool() as pool:
#         # Use tqdm to show progress with imap
#         for i, result in enumerate(tqdm(
#             pool.imap(process_network_optimized, tasks), 
#             total=len(tasks), 
#             desc="Evaluating networks"
#         )):
#             # Append result to CSV immediately
#             if output_path:
#                 append_result_to_csv(
#                     result=result,
#                     output_path=output_path,
#                     write_header=(write_header and i == 0)
#                 )
            
#             results_count += 1
            
#             # Progress update every 10 networks
#             if results_count % 10 == 0:
#                 print(f"💾 Saved {results_count}/{len(tasks)} results to disk")
    
#     print(f"\n✅ All {results_count} new results saved to: {output_path}")
    
#     # Load and return final results
#     if output_path and output_path.exists():
#         final_df = pd.read_csv(output_path)
#         return final_df
#     else:
#         return pd.DataFrame()


# def main():
#     """Main execution function."""
    
#     # Define paths
#     dataset_name = "suarez_MaMI_dataset"
#     experiment_name = "31_suarez_MaMI_size_100_wider_sweep_57"
    
#     path_experiment = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm") / dataset_name / experiment_name
#     generated_networks_dir = path_experiment / "generated_networks"
#     empirical_networks_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / "01_connectomes/01_consensus_bin_density_10_percent_100.npy"
    
#     # Distance matrix path
#     distance_matrix_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / "02_distance_matrices/distance_matrix_100.npy"
    
#     # Reference CSV with desired parameter order 
#     reference_csv_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm") / dataset_name / "31_suarez_MaMI_size_100_wider_sweep_220_copy/all_metrics_for_exp_30_shafiei_size_68.csv"
    
    
#     # Optional: Load a config for custom evaluation metrics
#     config_dict = {
#         'gnm': {
#             'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
#         }
#     }
    
#     # Output path
#     output_dir = generated_networks_dir.parent
#     experiment_name = generated_networks_dir.parent.name
#     output_path = output_dir / f"summary_indiv_energies_for_exp_{experiment_name}.csv"
    
#     try:
#         # Load networks
#         print("\n" + "=" * 60)
#         print("LOADING NETWORKS")
#         print("=" * 60)
        
#         generated_networks, generated_network_parameters, filenames = load_generated_networks(generated_networks_dir)
#         empirical_networks = load_empirical_networks(empirical_networks_path)
        
#         # Validate dimensions
#         if generated_networks is not None:
#             if generated_networks.shape[-1] != empirical_networks.shape[-1]:
#                 raise ValueError(
#                     f"Dimension mismatch: Generated networks have {generated_networks.shape[-1]} nodes "
#                     f"but empirical networks have {empirical_networks.shape[-1]} nodes."
#                 )
        
#         # Run comparison
#         print("\n" + "=" * 60)
#         print("COMPARING NETWORKS")
#         print("=" * 60)
        
#         results_df = compare_all_networks(
#             generated_networks=generated_networks,
#             generated_network_parameters=generated_network_parameters,
#             filenames=filenames,
#             empirical_networks=empirical_networks,
#             distance_matrices_path=distance_matrix_path if distance_matrix_path.exists() else None,
#             config_dict=config_dict,
#             reference_csv_path=reference_csv_path, 
#             output_path=output_path, 
#         )
        
#         # Display summary
#         print("\n" + "=" * 60)
#         print("SUMMARY")
#         print("=" * 60)
#         print(f"Total comparisons in file: {len(results_df)}")
#         if not results_df.empty:
#             print(f"Parameter ranges:")
#             print(f"  eta: [{results_df['eta'].min():.3f}, {results_df['eta'].max():.3f}]")
#             print(f"  gamma: [{results_df['gamma'].min():.3f}, {results_df['gamma'].max():.3f}]")
#             print(f"\nFirst few rows of results:")
#             print(results_df.head())
        
#         print("\n✅ Comparison complete!")
        
#         return results_df
        
#     except Exception as e:
#         print(f"\n❌ Error during comparison: {e}")
#         import traceback
#         traceback.print_exc()
#         return None


# if __name__ == "__main__":
#     results = main()


"""
Compare generated and empirical networks with another and creates big csv file. 
Optimized version with batching, efficient multiprocessing, and resumption capability.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import re
from typing import Dict, List, Tuple, Optional
import csv

# Import the evaluation criteria function
from src.config.GNM import create_evaluation_criteria
from gnm.fitting import RunConfig
from gnm.model import BinaryGenerativeParameters
from multiprocessing import Pool

def extract_params_from_filename(filename: str) -> Dict[str, float]:
    """
    Extract eta and gamma parameters from network filename.
    Expected format: net_eta{value}_gamma{value}_rule{name}.npy
    
    Args:
        filename: Network filename
        
    Returns:
        Dictionary with eta and gamma values
    """
    match = re.search(r'eta([-\d.]+)_gamma([-\d.]+)', filename)
    if match:
        eta = float(match.group(1))
        gamma = float(match.group(2))
        return {'eta': eta, 'gamma': gamma}
    else:
        raise ValueError(f"Could not extract parameters from filename: {filename}")


def load_generated_networks(networks_dir: Path) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """
    Load all generated networks from directory.
    
    Args:
        networks_dir: Directory containing .npy network files
        
    Returns:
        Tuple of (networks array, parameters list, filenames list)
    """
    networks_dir = Path(networks_dir)
    network_files = sorted(networks_dir.glob("net_eta*.npy"))
    
    if not network_files:
        raise FileNotFoundError(f"No network files found in {networks_dir}")
    
    print(f"Found {len(network_files)} generated network files")
    
    networks = []
    network_parameters = []
    filenames = []
    
    for net_file in network_files:
        try:
            params = extract_params_from_filename(net_file.name)
            network = np.load(net_file)
            networks.append(network)
            network_parameters.append(params)
            filenames.append(net_file.name)
        except Exception as e:
            print(f"Warning: Could not load {net_file.name}: {e}")
            continue
    
    networks = np.concatenate(networks)
    
    return networks, network_parameters, filenames


def load_empirical_networks(empirical_path: Path) -> np.ndarray:
    """
    Load empirical networks.
    
    Args:
        empirical_path: Path to .npy file with shape (n_subjects, n_nodes, n_nodes)
        
    Returns:
        Empirical networks array
    """
    empirical_path = Path(empirical_path)
    
    if not empirical_path.exists():
        raise FileNotFoundError(f"Empirical networks file not found: {empirical_path}")
    
    empirical_networks = np.load(empirical_path)
    
    print(f"Loaded empirical networks with shape: {empirical_networks.shape}")
    print(f"Number of subjects: {empirical_networks.shape[0]}")
    
    return empirical_networks


def load_existing_results(output_path: Path) -> Tuple[Optional[pd.DataFrame], set]:
    """
    Load existing results if they exist and determine which networks have been processed.
    
    Args:
        output_path: Path to results CSV file
        
    Returns:
        Tuple of (existing DataFrame or None, set of processed (eta, gamma) tuples)
    """
    if not output_path.exists():
        return None, set()
    
    try:
        df = pd.read_csv(output_path)
        if df.empty or 'eta' not in df.columns or 'gamma' not in df.columns:
            return None, set()
        
        # Create set of processed parameter combinations
        processed = set(zip(df['eta'].round(10), df['gamma'].round(10)))
        print(f"📊 Found existing results with {len(df)} rows")
        print(f"   Already processed {len(processed)} unique parameter combinations")
        
        return df, processed
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return None, set()


def get_ordered_tasks(
    generated_networks: np.ndarray,
    generated_network_parameters: List[Dict],
    filenames: List[str],
    existing_results_df: Optional[pd.DataFrame],
    processed_params: set,
    reference_df: Optional[pd.DataFrame] = None
) -> List[Tuple[int, np.ndarray, Dict, str]]:
    """
    Create ordered list of tasks, following the order from reference CSV if available,
    otherwise from existing results. Skip already processed networks.
    
    Args:
        generated_networks: Array of networks
        generated_network_parameters: List of parameter dicts
        filenames: List of network filenames
        existing_results_df: Existing results DataFrame or None
        processed_params: Set of already processed (eta, gamma) tuples
        reference_df: Reference DataFrame with desired parameter order
        
    Returns:
        List of (index, network, params, filename) tuples in proper order
    """
    tasks = []
    
    # Create mapping from (eta, gamma) to network data
    network_map = {}
    for i, (params, filename) in enumerate(zip(generated_network_parameters, filenames)):
        key = (round(params['eta'], 10), round(params['gamma'], 10))
        network_map[key] = (i, generated_networks[i], params, filename)
    
    # Prioritize reference CSV for ordering
    order_source = reference_df if reference_df is not None else existing_results_df
    
    # If we have a reference or existing results, follow their order first
    if order_source is not None and not order_source.empty:
        # Get unique parameter combinations in their existing order
        existing_params = list(zip(
            order_source['eta'].round(3),  # TODO: Increase this resolution! 
            order_source['gamma'].round(3), 
        ))
        
        # Add networks in the order they appear (TODO: skip processed ones)
        # seen_keys = set()
        for key in existing_params:
            if key in network_map: #  and key not in seen_keys:
                if key not in processed_params:
                    tasks.append(network_map[key])
                # seen_keys.add(key)
    
    return tasks


def create_evaluation_criteria_list(
    distance_matrices: torch.Tensor,
    config_dict: Dict = None
) -> List:
    """
    Pre-create evaluation criteria for each distance matrix to avoid recreation overhead.
    
    Args:
        distance_matrices: Tensor of shape (n_subjects, n_nodes, n_nodes)
        config_dict: Optional config dictionary
        
    Returns:
        List of evaluation criteria objects
    """
    evaluation_criteria_list = []
    
    print("Creating evaluation criteria for each subject...")
    for i in tqdm(range(len(distance_matrices)), desc="Building criteria"):
        distance_matrix = distance_matrices[i]
        
        if config_dict is not None:
            criteria = create_evaluation_criteria(
                config=config_dict,
                distance_matrix=distance_matrix
            )
        else:
            from gnm import evaluation
            criteria = evaluation.MaxCriteria(
                evaluation.DegreeKS(),
                evaluation.ClusteringKS(),
                evaluation.EdgeLengthKS(distance_matrix),
                evaluation.BetweennessKS()
            )
        
        evaluation_criteria_list.append(criteria)
    
    return evaluation_criteria_list


def evaluate_network_against_empirical_optimized(
    generated_network_tensor: torch.Tensor,
    empirical_networks_tensor: torch.Tensor,
    evaluation_criteria_list: List,
) -> Dict[str, float]:
    """
    Evaluate a single generated network against all empirical networks.
    Optimized version using pre-created criteria and torch tensors.
    
    Args:
        generated_network_tensor: Single generated network tensor (1, n_nodes, n_nodes)
        empirical_networks_tensor: Empirical networks tensor (n_subjects, n_nodes, n_nodes)
        evaluation_criteria_list: Pre-created evaluation criteria for each subject
        
    Returns:
        Dictionary with energy values for each empirical network
    """
    n_subjects = empirical_networks_tensor.shape[0]
    results = {}
    
    # Evaluate against each empirical network
    for subject_idx in range(n_subjects):
        empirical_single = empirical_networks_tensor[subject_idx:subject_idx+1, :, :]
        
        # Use pre-created evaluation criteria
        evaluation_criteria = evaluation_criteria_list[subject_idx]
        
        # Calculate energy
        energy_dict = evaluation_criteria(generated_network_tensor, empirical_single)
        
        # Store results for this subject
        for energy_value in energy_dict:
            column_name = f"MaxCrit(etc)_indiv_{subject_idx}"
            results[column_name] = float(energy_value.mean().item())
    
    return results


def process_network_optimized(args: Tuple) -> Dict:
    """
    Worker function to evaluate a single network against empirical data.
    Optimized version that receives pre-converted tensors and pre-created criteria.
    """
    network_idx, network, generated_parameters, filename, empirical_networks_tensor, evaluation_criteria_list = args
    
    # Convert generated network to tensor
    gen_network_tensor = torch.tensor(network, dtype=torch.float32).unsqueeze(0)
    
    # Evaluate this network against all empirical networks
    energy_results = evaluate_network_against_empirical_optimized(
        generated_network_tensor=gen_network_tensor,
        empirical_networks_tensor=empirical_networks_tensor,
        evaluation_criteria_list=evaluation_criteria_list,
    )
    
    # Combine parameters with results and metadata
    result = {
        'network_index': network_idx,
        'filename': filename,
        **generated_parameters, 
        **energy_results
    }
    
    return result


def append_result_to_csv(result: Dict, output_path: Path, write_header: bool = False):
    """
    Append a single result to the CSV file.
    
    Args:
        result: Dictionary with result data
        output_path: Path to output CSV
        write_header: Whether to write header (for first row)
    """
    mode = 'w' if write_header else 'a'
    
    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def compare_all_networks(
    generated_networks: np.ndarray,
    generated_network_parameters: List[Dict],
    filenames: List[str],
    empirical_networks: np.ndarray,
    distance_matrices_path: Path = None,
    config_dict: Dict = None,
    output_path: Path = None,
    reference_csv_path: Path = None
) -> pd.DataFrame:
    """
    Compare all generated networks with empirical networks using optimized multiprocessing.
    Supports resumption and appends results line-by-line.
    
    Args:
        generated_networks: Array of generated networks
        generated_network_parameters: List of parameter dicts
        filenames: List of network filenames
        empirical_networks: Empirical networks array
        distance_matrices_path: Path to distance matrix file
        config_dict: Optional config dictionary
        output_path: Path to save results CSV
        reference_csv_path: Path to reference CSV for parameter ordering
        
    Returns:
        DataFrame with comparison results
    """
    
    # Check for existing results and determine what's already processed
    existing_df, processed_params = load_existing_results(output_path)
    
    # Load reference CSV for parameter ordering if provided
    reference_df = None
    if reference_csv_path and reference_csv_path.exists():
        try:
            reference_df = pd.read_csv(reference_csv_path)
            print(f"📖 Using parameter order from: {reference_csv_path.name}")
        except Exception as e:
            print(f"Warning: Could not load reference CSV: {e}")
            reference_df = None
    
    # Get ordered tasks, skipping already processed ones
    ordered_tasks_data = get_ordered_tasks(
        generated_networks=generated_networks,
        generated_network_parameters=generated_network_parameters,
        filenames=filenames,
        existing_results_df=existing_df,
        processed_params=processed_params,
        reference_df=reference_df
    )
    
    if not ordered_tasks_data:
        print("✅ All networks have already been processed!")
        if existing_df is not None:
            return existing_df
        else:
            return pd.DataFrame()
    
    print(f"\n🔄 {len(ordered_tasks_data)} networks to process ({len(processed_params)} already done)")
    
    # Load distance matrices once
    distance_matrices = None
    evaluation_criteria_list = None
    
    if distance_matrices_path and Path(distance_matrices_path).exists():
        print("Loading distance matrices...")
        distance_matrices = np.load(distance_matrices_path)
        distance_matrices = torch.tensor(distance_matrices, dtype=torch.float32)
        print(f"Loaded distance matrix with shape: {distance_matrices.shape}")
        
        # Pre-create evaluation criteria for all subjects
        evaluation_criteria_list = create_evaluation_criteria_list(
            distance_matrices=distance_matrices,
            config_dict=config_dict
        )
    else:
        print("No distance matrices provided, creating criteria without edge length...")
        # Create evaluation criteria without distance matrix
        from gnm import evaluation
        n_subjects = empirical_networks.shape[0]
        evaluation_criteria_list = []
        for _ in range(n_subjects):
            criteria = evaluation.MaxCriteria(
                evaluation.DegreeKS(),
                evaluation.ClusteringKS(),
                evaluation.BetweennessKS()
            )
            evaluation_criteria_list.append(criteria)
    
    # Convert empirical networks to torch tensor once
    print("Converting empirical networks to torch tensors...")
    empirical_networks_tensor = torch.tensor(empirical_networks, dtype=torch.float32)
    
    # Prepare tasks
    print("Preparing tasks...")
    tasks = []
    for net_idx, network, params, filename in ordered_tasks_data:
        task_args = (
            net_idx,
            network,
            params,
            filename,
            empirical_networks_tensor,
            evaluation_criteria_list
        )
        tasks.append(task_args)
    
    print(f"\nProcessing {len(tasks)} networks...")
    
    # Determine if we need to write header
    write_header = existing_df is None or not output_path.exists()
    
    # Use multiprocessing Pool to process tasks in parallel
    results_count = 0
    
    with Pool() as pool:
        # Use tqdm to show progress with imap
        for i, result in enumerate(tqdm(
            pool.imap(process_network_optimized, tasks), 
            total=len(tasks), 
            desc="Evaluating networks"
        )):
            # Append result to CSV immediately
            if output_path:
                append_result_to_csv(
                    result=result,
                    output_path=output_path,
                    write_header=(write_header and i == 0)
                )
            
            results_count += 1
            
            # Progress update every 10 networks
            if results_count % 10 == 0:
                print(f"💾 Saved {results_count}/{len(tasks)} results to disk. Path: {output_path}")
    
    print(f"\n✅ All {results_count} new results saved to: {output_path}")
    
    # Load and return final results
    if output_path and output_path.exists():
        final_df = pd.read_csv(output_path)
        return final_df
    else:
        return pd.DataFrame()


def main():
    """Main execution function."""
    
    # Define paths
    dataset_name = "suarez_MaMI_dataset"
    experiment_name = "31_suarez_MaMI_size_100_wider_sweep_57_copy_2_now_run_with_evaluation"
    
    path_experiment = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm") / dataset_name / experiment_name
    generated_networks_dir = path_experiment / "generated_networks"
    empirical_networks_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / "01_connectomes/01_consensus_bin_density_10_percent_100.npy"
    
    # Distance matrix path
    distance_matrix_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / "02_distance_matrices/distance_matrix_100.npy"
    
    # Reference CSV with desired parameter order
    # TODO: Change this later! 
    reference_csv_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm") / dataset_name / experiment_name / "all_metrics_for_exp_30_shafiei_size_68.csv"
    
    # Optional: Load a config for custom evaluation metrics
    config_dict = {
        'gnm': {
            'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
        }
    }
    
    # Output path
    output_dir = generated_networks_dir.parent
    experiment_name = generated_networks_dir.parent.name
    output_path = output_dir / f"summary_indiv_energies_for_exp_{experiment_name}.csv"
    
    try:
        # Load networks
        print("\n" + "=" * 60)
        print("LOADING NETWORKS")
        print("=" * 60)
        
        generated_networks, generated_network_parameters, filenames = load_generated_networks(generated_networks_dir)
        empirical_networks = load_empirical_networks(empirical_networks_path)
        
        # Validate dimensions
        if generated_networks is not None:
            if generated_networks.shape[-1] != empirical_networks.shape[-1]:
                raise ValueError(
                    f"Dimension mismatch: Generated networks have {generated_networks.shape[-1]} nodes "
                    f"but empirical networks have {empirical_networks.shape[-1]} nodes."
                )
        
        # Run comparison
        print("\n" + "=" * 60)
        print("COMPARING NETWORKS")
        print("=" * 60)
        
        results_df = compare_all_networks(
            generated_networks=generated_networks,
            generated_network_parameters=generated_network_parameters,
            filenames=filenames,
            empirical_networks=empirical_networks,
            distance_matrices_path=distance_matrix_path if distance_matrix_path.exists() else None,
            config_dict=config_dict,
            output_path=output_path,
            reference_csv_path=reference_csv_path if reference_csv_path.exists() else None
        )
        
        # Display summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total comparisons in file: {len(results_df)}")
        if not results_df.empty:
            print(f"Parameter ranges:")
            print(f"  eta: [{results_df['eta'].min():.3f}, {results_df['eta'].max():.3f}]")
            print(f"  gamma: [{results_df['gamma'].min():.3f}, {results_df['gamma'].max():.3f}]")
            print(f"\nFirst few rows of results:")
            print(results_df.head())
        
        print("\n✅ Comparison complete!")
        
        return results_df
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()