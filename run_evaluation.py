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

# from src.config.path import PathConfig
# from src.config.GNM import create_evaluation_criteria
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


# def load_generated_networks(network_filenames_set) -> Tuple[np.ndarray, List[Dict], List[str]]:
#     """
#     Load all generated networks from directory.
    
#     Args:
#         network_filenames_set: Set with all PosixPaths of the different networks 
        
#     Returns:
#         Tuple of (networks array, parameters list, filenames list)
#     """
    
#     networks = []
#     network_parameters = []
#     filenames = []
    
#     for net_file in network_filenames_set:
#         networks
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
#         evaluation_criteria = evaluation_criteria_list[0] # [<gnm.evaluation.composite_criteria.MaxCriteria object at 0x32b7963c0>]    ### subject_idx]
        
#         # Calculate energy
#         energy_dict = evaluation_criteria(generated_network_tensor, empirical_single)
        
#         # Store results for this subject
#         for energy_value in energy_dict:
#             column_name = f"MaxCrit_indiv_{subject_idx}" # TODO: Add maxcrit criteria here?
#             results[column_name] = float(energy_value.mean().item())
    
#     return results


# def process_network_optimized(args: Tuple) -> Dict:
#     """
#     Worker function to evaluate a single network against empirical data.
#     Optimized version that receives pre-converted tensors and pre-created criteria.
#     """
#     network_idx, network, generated_parameters, empirical_networks_tensor, evaluation_criteria_list = args
    
#     # Convert generated network to tensor
#     gen_network_tensor = torch.tensor(network, dtype=torch.float32) # .unsqueeze(0)
    
#     # Evaluate this network against all empirical networks
#     energy_results = evaluate_network_against_empirical_optimized(
#         generated_network_tensor=gen_network_tensor,
#         empirical_networks_tensor=empirical_networks_tensor,
#         evaluation_criteria_list=evaluation_criteria_list,
#     )
    
#     # Combine parameters with results and metadata
#     result = {
#         'network_index': network_idx,
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


# def main():
#     """Main execution function."""
    
#     ##########################################################################
#     ########### HARDCODED STUFF ##############################################

#     dataset_name = "suarez_MaMI_dataset"
#     # experiment_name = 
#     experiment_name = "60_generally_finer_search_animal_0" # 49_shafiei" # 49_suarez_MaMI_100" 
    
#     # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/all_metrics_for_60_generally_finer_search_animal_0.csv
#     path_config = PathConfig( 
#         dataset_name=dataset_name,
#         experiment_name=experiment_name, 
#     )
    
#     if dataset_name == "suarez_MaMI_dataset": 
#         empirical_networks_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_50.npy") #
#         # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_100.npy") 
#         # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / "01_connectomes/01_consensus_bin_density_10_percent_100.npy"
        
#         # Distance matrix path
#         distance_matrices_path = path_config.dir_02_distance_matrices /"distance_matrix_50.npy"
    
#     if dataset_name == "shafiei_human_consensus_dataset": 
#         empirical_networks_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/griffa_70_human_connectomes_dataset/01_connectomes/01_indiv_connectomes_bin_density_10_percent_68.npy") 
#         # Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / "01_connectomes/01_indiv_connectomes_bin_density_10_percent_68.npy"
        
#         # Distance matrix path
#         distance_matrices_path = path_config.dir_02_distance_matrices /"distance_matrix_68.npy"
    
#     number_multiprocessing_processes = 8 # "must be at least 1" - so I guess no -1 then? 
    
#     ##########################################################################
       
    
#     generated_networks_dir = path_config.output_experiment_dir / "generated_networks"

#     # Reference CSV with desired parameter order
#     reference_csv_path = path_config.output_experiment_dir / f"all_metrics_for_{experiment_name}.csv" # f"all_metrics_for_{experiment_name}.csv"

#     # Load a config for custom evaluation metrics
#     config_dict = {
#         'gnm': {
#             'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
#         }
#     }
    
#     # Output path
#     output_dir = path_config.output_experiment_dir
#     output_path = output_dir / f"summary_indiv_energies_for_exp_{experiment_name}.csv"
    
#     try:
#         # Load networks
#         print("\n" + "=" * 60)
#         print("LOADING NETWORKS")
#         print("=" * 60)

#         df_prior_results = pd.read_csv(output_path)
#         already_processed_files = set(df_prior_results.filename)


#         networks_dir = Path(generated_networks_dir)
#         network_files = sorted(networks_dir.glob("net_eta*.npy"))
#         if not network_files:
#             raise FileNotFoundError(f"No network files found in {networks_dir}")
#         network_files = set([str(file).split("/")[-1] for file in network_files]) # network_files)

#         to_analyze_files = network_files.difference(already_processed_files)

#         print(f"Found {len(to_analyze_files)} generated network files that are not yet preprocessed.")

#         df_reference_order = pd.read_csv(reference_csv_path)
#         list_eta_gamma_true_order = []
#         for (eta, gamma) in zip(list(df_reference_order["eta"]), list(df_reference_order["gamma"])): 
#             list_eta_gamma_true_order.append((eta, gamma))

#         to_analyze_eta_gamma_pairs = []
#         for file in list(to_analyze_files): 
#             eta = str(file).split("/")[-1].split("_")[1].split("a")[-1]
#             gamma = str(file).split("/")[-1].split("_")[2].split("a")[-1]
#             to_analyze_eta_gamma_pairs.append((eta, gamma))

#         list_eta_gamma_true_order_files = [f'/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/generated_networks/net_eta{pair[0]}_gamma{pair[1]}_ruleMatchingIndex.npy' for pair in list_eta_gamma_true_order]
#             # 50000
#         to_analyze_eta_and_gamma_files = [f'/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/generated_networks/net_eta{pair[0]}_gamma{pair[1]}_ruleMatchingIndex.npy' for pair in to_analyze_eta_gamma_pairs]

#         ordered_and_selected_eta_and_gamma_pairs_files = [path for path in list_eta_gamma_true_order_files if path in to_analyze_eta_and_gamma_files]
#             # 24572
#         # len(list(set(list_eta_gamma_true_order_files)))

#         ordered_and_selected_eta_and_gamma_pairs_files_parameters = []
#         for file in list(ordered_and_selected_eta_and_gamma_pairs_files): 
#             eta = str(file).split("/")[-1].split("_")[1].split("a")[-1]
#             gamma = str(file).split("/")[-1].split("_")[2].split("a")[-1]
#             ordered_and_selected_eta_and_gamma_pairs_files_parameters.append((eta, gamma))

#         generated_network_parameters = ordered_and_selected_eta_and_gamma_pairs_files_parameters

#         generated_networks = []
#         for file in ordered_and_selected_eta_and_gamma_pairs_files: 
#             gen = np.load(file)
#             generated_networks.append(gen)
#         generated_networks = np.stack(generated_networks)
        
#         # filenames = ordered_and_selected_eta_and_gamma_pairs_files
#         # generated_networks, generated_network_parameters, filenames = load_generated_networks(ordered_and_selected_eta_and_gamma_pairs_files)
#         empirical_networks = load_empirical_networks(empirical_networks_path)
        
        
        
#         # #################################################


#         # networks_dir = Path(generated_networks_dir)
#         # network_files = sorted(networks_dir.glob("net_eta*.npy"))
#         # if not network_files:
#         #     raise FileNotFoundError(f"No network files found in {networks_dir}")
#         # network_files = set([str(file).split("/")[-1] for file in network_files]) # network_files)

#         # to_analyze_files = network_files.difference(already_processed_files)

#         # print(f"Found {len(to_analyze_files)} generated network files that are not yet preprocessed.")

#         # df_reference_order = pd.read_csv(reference_csv_path)
#         # list_eta_gamma_true_order = []
#         # for (eta, gamma) in zip(list(df_reference_order["eta"]), list(df_reference_order["gamma"])): 
#         #     list_eta_gamma_true_order.append((eta, gamma))

#         # to_analyze_eta_gamma_pairs = []
#         # for file in list(to_analyze_files): 
#         #     eta = str(file).split("/")[-1].split("_")[1].split("a")[-1]
#         #     gamma = str(file).split("/")[-1].split("_")[2].split("a")[-1]
#         #     to_analyze_eta_gamma_pairs.append((eta, gamma))

#         # list_eta_gamma_true_order_files = [f'/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/generated_networks/net_eta{pair[0]}_gamma{pair[1]}_ruleMatchingIndex.npy' for pair in list_eta_gamma_true_order]
#         #     # 50000
#         # to_analyze_eta_and_gamma_files = [f'/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/generated_networks/net_eta{pair[0]}_gamma{pair[1]}_ruleMatchingIndex.npy' for pair in to_analyze_eta_gamma_pairs]

#         # ordered_and_selected_eta_and_gamma_pairs_files = [path for path in list_eta_gamma_true_order_files if path in to_analyze_eta_and_gamma_files]
#         #     # 24572
#         # # len(list(set(list_eta_gamma_true_order_files)))

#         # ordered_and_selected_eta_and_gamma_pairs_files_parameters = []
#         # for file in list(ordered_and_selected_eta_and_gamma_pairs_files): 
#         #     eta = str(file).split("/")[-1].split("_")[1].split("a")[-1]
#         #     gamma = str(file).split("/")[-1].split("_")[2].split("a")[-1]
#         #     ordered_and_selected_eta_and_gamma_pairs_files_parameters.append((eta, gamma))

#         # generated_network_parameters = ordered_and_selected_eta_and_gamma_pairs_files_parameters
#         # generated_network_files = ordered_and_selected_eta_and_gamma_pairs_files

#         # generated_networks = []
#         # for file in ordered_and_selected_eta_and_gamma_pairs_files: 
#         #     gen = np.load(file)
#         #     generated_networks.append(gen)
#         # generated_networks = np.stack(generated_networks)
        
#         # # filenames = ordered_and_selected_eta_and_gamma_pairs_files
#         # # generated_networks, generated_network_parameters, filenames = load_generated_networks(ordered_and_selected_eta_and_gamma_pairs_files)
#         # empirical_networks = load_empirical_networks(empirical_networks_path)
        
        
#         ###############################
        
        
#         # networks_dir = Path(generated_networks_dir)
#         # network_files = sorted(networks_dir.glob("net_eta*.npy"))
#         # if not network_files:
#         #     raise FileNotFoundError(f"No network files found in {networks_dir}")
#         # network_files = set([str(file).split("/")[-1] for file in network_files]) # network_files)

#         # to_analyze_files = network_files.difference(already_processed_files)

#         # print(f"Found {len(to_analyze_files)} generated network files that are not yet preprocessed.")

#         # df_reference_order = pd.read_csv(reference_csv_path)
#         # list_eta_gamma_true_order = []
#         # for (eta, gamma) in zip(list(df_reference_order["eta"]), list(df_reference_order["gamma"])): 
#         #     list_eta_gamma_true_order.append((eta, gamma))

#         # to_analyze_eta_gamma_pairs = []
#         # for file in list(to_analyze_files): 
#         #     eta = str(file).split("/")[-1].split("_")[1].split("a")[-1]
#         #     gamma = str(file).split("/")[-1].split("_")[2].split("a")[-1]
#         #     to_analyze_eta_gamma_pairs.append((eta, gamma))

#         # list_eta_gamma_true_order_files = [f'/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/generated_networks/net_eta{pair[0]}_gamma{pair[1]}_ruleMatchingIndex.npy' for pair in list_eta_gamma_true_order]
#         #     # 50000
#         # to_analyze_eta_and_gamma_files = [f'/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/generated_networks/net_eta{pair[0]}_gamma{pair[1]}_ruleMatchingIndex.npy' for pair in to_analyze_eta_gamma_pairs]

#         # ordered_and_selected_eta_and_gamma_pairs_files = [path for path in list_eta_gamma_true_order_files if path in to_analyze_eta_and_gamma_files]
#         #     # 24572
#         # # len(list(set(list_eta_gamma_true_order_files)))
#         # # generated_networks, generated_network_parameters, filenames = load_generated_networks(ordered_and_selected_eta_and_gamma_pairs_files)
        
#         # generated_networks = []
#         # for file in ordered_and_selected_eta_and_gamma_pairs_files: 
#         #     generated_networks.append(np.load(file))
#         # generated_networks = np.stack(generated_networks)
            
#         # empirical_networks = load_empirical_networks(empirical_networks_path)
        
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
        
#         # results_df = compare_all_networks(
#         #     generated_networks=generated_networks,
#         #     generated_network_parameters=generated_network_parameters,
#         #     empirical_networks=empirical_networks,
#         #     distance_matrices_path=distance_matrix_path if distance_matrix_path.exists() else None,
#         #     config_dict=config_dict,
#         #     output_path=output_path,
#         #     dataset_name=dataset_name, 
#         #     reference_csv_path=reference_csv_path if reference_csv_path.exists() else None, 
#         #     number_multiprocessing_processes=number_multiprocessing_processes, 
#         # )
        
        
#         # Load distance matrices once
#         distance_matrices = None
#         evaluation_criteria_list = None
        
#         # Distance Matrices
#         print("Loading distance matrices...")
#         distance_matrices = np.load(distance_matrices_path)
#         distance_matrices = torch.tensor(distance_matrices, dtype=torch.float32)
#         if dataset_name == "shafiei_human_consensus_dataset": 
#             distance_matrices = distance_matrices.unsqueeze(0)
#             print("Attention: as assumed that it's the Shafiei dataset, this distance matrix will be unsqueezed.")
#         print(f"Loaded distance matrix with shape: {distance_matrices.shape}")
        
#         # Pre-create evaluation criteria for all subjects
#         evaluation_criteria_list = create_evaluation_criteria_list(
#             distance_matrices=distance_matrices,
#             config_dict=config_dict
#         )
        
#         # Convert empirical networks to torch tensor once
#         print("Converting empirical networks to torch tensors...")
#         empirical_networks_tensor = torch.tensor(empirical_networks, dtype=torch.float32)
        
#         # Prepare tasks
#         print("Preparing tasks...")
#         tasks = []
#         for net_idx, (network, params) in enumerate(zip(generated_networks, generated_network_parameters)): 
#             task_args = (
#                 net_idx,
#                 network,
#                 params,
#                 empirical_networks_tensor,
#                 evaluation_criteria_list
#             )
#             tasks.append(task_args)
        
#         print(f"\nProcessing {len(tasks)} networks...")
        
#         # Determine if we need to write header
#         write_header = df_prior_results is None or not output_path.exists() # df_prior_results if df_prior_results is not None and output_path.exists() else None
        
#         # Use multiprocessing Pool to process tasks in parallel
#         results_count = 0
        
#         with Pool(processes=number_multiprocessing_processes) as pool:
#             # Use tqdm to show progress with imap
#             for i, result in enumerate(tqdm(
#                 pool.imap(process_network_optimized, tasks), 
#                 total=len(tasks), 
#                 desc="Evaluating networks"
#             )):
#                 # Append result to CSV immediately
#                 if output_path:
#                     append_result_to_csv(
#                         result=result,
#                         output_path=output_path,
#                         write_header=(write_header and i == 0)
#                     )
                
#                 results_count += 1
                
#                 # Progress update every 10 networks
#                 if results_count % 10 == 0:
#                     print(f"💾 Saved {results_count}/{len(tasks)} results to disk. Path: {output_path}")
        
#         print(f"\n✅ All {results_count} new results saved to: {output_path}")
        
#         # Load and return final results
#         if output_path and output_path.exists():
#             results_df = pd.read_csv(output_path)
#         else: 
#             results_df = None


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
#         else: 
#             print("⚠️ Attention: results_df is empty.")
#         print("\n✅ Comparison complete!")

        
#         return results_df
        
#     except Exception as e:
#         print(f"\n❌ Error during comparison: {e}")
#         import traceback
#         traceback.print_exc()
#         return None


# if __name__ == "__main__":
#     results = main()