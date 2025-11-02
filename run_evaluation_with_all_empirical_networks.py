"""
Compare generated and empirical networks with another and creates big csv file. 
Optimized version with batching, efficient multiprocessing, and resumption capability.
Now supports network IDs.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import re
from typing import Dict, List, Tuple, Optional
import csv

from src.config.path import PathConfig
from src.config.GNM import create_evaluation_criteria
from multiprocessing import Pool


def extract_params_from_filename(filename: str) -> Dict[str, float]:
    """
    Extract eta, gamma, and id parameters from network filename.
    Expected format: net_eta{value}_gamma{value}_rule{name}_id{value}.npy
    Or without id: net_eta{value}_gamma{value}_rule{name}.npy (defaults to id=0)
    
    Args:
        filename: Network filename
        
    Returns:
        Dictionary with eta, gamma, and id values
    """
    # Extract eta
    eta_match = re.search(r'eta([-+]?\d*\.?\d+)', filename)
    if not eta_match:
        raise ValueError(f"Could not extract eta from filename: {filename}")
    eta = float(eta_match.group(1))
    
    # Extract gamma
    gamma_match = re.search(r'gamma([-+]?\d*\.?\d+)', filename)
    if not gamma_match:
        raise ValueError(f"Could not extract gamma from filename: {filename}")
    gamma = float(gamma_match.group(1))
    
    # Extract id (defaults to 0 if not present)
    id_match = re.search(r'_id(\d+)', filename)
    net_id = int(id_match.group(1)) if id_match else 0
    
    return {'eta': eta, 'gamma': gamma, 'id': net_id}


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
        Tuple of (existing DataFrame or None, set of processed filenames)
    """
    if not output_path.exists():
        return None, set()
    
    try:
        df = pd.read_csv(output_path)
        if df.empty:
            return None, set()
        
        # Extract processed filenames
        if 'filename' in df.columns:
            processed = set(df['filename'].values)
        elif 'eta' in df.columns and 'gamma' in df.columns and 'id' in df.columns:
            # Reconstruct filenames from eta, gamma, and id
            processed = set()
            for _, row in df.iterrows():
                net_id = row['id'] if not pd.isna(row['id']) else 0
                if net_id == 0:
                    filename = f"net_eta{row['eta']}_gamma{row['gamma']}_ruleMatchingIndex.npy"
                else:
                    filename = f"net_eta{row['eta']}_gamma{row['gamma']}_ruleMatchingIndex_id{int(net_id)}.npy"
                processed.add(filename)
        else:
            return None, set()
        
        print(f"📊 Found existing results with {len(df)} rows")
        print(f"   Already processed {len(processed)} networks")
        
        return df, processed
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return None, set()


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
        
        # Use pre-created evaluation criteria (use first one for all subjects)
        evaluation_criteria = evaluation_criteria_list[0]
        
        # Calculate energy
        energy_dict = evaluation_criteria(generated_network_tensor, empirical_single)
        
        # Store results for this subject
        for energy_value in energy_dict:
            column_name = f"MaxCrit_indiv_{subject_idx}"
            results[column_name] = float(energy_value.mean().item())
    
    return results


def process_network_optimized(args: Tuple) -> Dict:
    """
    Worker function to evaluate a single network against empirical data.
    Optimized version that receives pre-converted tensors and pre-created criteria.
    """
    network_idx, network, generated_parameters, filename, empirical_networks_tensor, evaluation_criteria_list = args
    
    # Convert generated network to tensor
    gen_network_tensor = torch.tensor(network, dtype=torch.float32)
    
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


def main(experiment_name: str):
    """Main execution function."""
    
    ##########################################################################
    ########### HARDCODED STUFF ##############################################

    dataset_name = "suarez_MaMI_dataset"
    # experiment_name = IS IN __MAIN__ # "71_testing_animal_0" # 67_testing_animal_0" # 66_filling_in_the_upper_region_animal_0"
    
    path_config = PathConfig( 
        dataset_name=dataset_name,
        experiment_name=experiment_name, 
    )
    
    if dataset_name == "suarez_MaMI_dataset": 
        empirical_networks_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_50.npy")
        distance_matrices_path = path_config.dir_02_distance_matrices / "distance_matrix_50.npy"
    
    if dataset_name == "shafiei_human_consensus_dataset": 
        empirical_networks_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/griffa_70_human_connectomes_dataset/01_connectomes/01_indiv_connectomes_bin_density_10_percent_68.npy")
        distance_matrices_path = path_config.dir_02_distance_matrices / "distance_matrix_68.npy"
    
    number_multiprocessing_processes = 8
    
    ##########################################################################
    
    generated_networks_dir = path_config.output_experiment_dir / "generated_networks"
    
    # Choose which reference CSV to use based on what exists
    static_csv = path_config.output_experiment_dir / f"all_static_metrics_for_{experiment_name}.csv"
    dynamic_csv = path_config.output_experiment_dir / f"all_dynamic_metrics_for_{experiment_name}.csv"
    computational_csv = path_config.output_experiment_dir / f"all_computational_metrics_for_{experiment_name}.csv"
    legacy_csv = path_config.output_experiment_dir / f"all_metrics_for_{experiment_name}.csv"
    
    # Use the first CSV that exists for reference ordering
    if static_csv.exists():
        reference_csv_path = static_csv
    elif dynamic_csv.exists():
        reference_csv_path = dynamic_csv
    elif computational_csv.exists():
        reference_csv_path = computational_csv
    elif legacy_csv.exists():
        reference_csv_path = legacy_csv
    else:
        raise FileNotFoundError(f"No reference CSV found in {path_config.output_experiment_dir}")
    
    print(f"Using reference CSV: {reference_csv_path}")

    # Load a config for custom evaluation metrics
    config_dict = {
        'gnm': {
            'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
        }
    }
    
    # Output path
    output_dir = path_config.output_experiment_dir
    output_path = output_dir / f"summary_indiv_energies_for_exp_{experiment_name}.csv"
    
    try:
        # Load networks
        print("\n" + "=" * 60)
        print("LOADING NETWORKS")
        print("=" * 60)

        # Load existing results to determine what's been processed
        df_prior_results, already_processed_files = load_existing_results(output_path)

        # Get all network files
        networks_dir = Path(generated_networks_dir)
        all_network_files = sorted(networks_dir.glob("net_eta*.npy"))
        if not all_network_files:
            raise FileNotFoundError(f"No network files found in {networks_dir}")
        
        # Determine which files need to be processed
        all_filenames = set([file.name for file in all_network_files])
        to_analyze_files = all_filenames - already_processed_files
        
        print(f"Found {len(all_filenames)} total network files")
        print(f"Already processed: {len(already_processed_files)}")
        print(f"To analyze: {len(to_analyze_files)}")

        if len(to_analyze_files) == 0:
            print("✅ All networks have been processed!")
            return pd.read_csv(output_path) if output_path.exists() else None

        # Load reference order from CSV
        df_reference_order = pd.read_csv(reference_csv_path)
        
        # Build list of (eta, gamma, id) tuples from reference
        if 'id' in df_reference_order.columns:
            list_eta_gamma_id_reference = list(zip(
                df_reference_order["eta"], 
                df_reference_order["gamma"],
                df_reference_order["id"]
            ))
        else:
            # Legacy format without id column - assume all are id=0
            list_eta_gamma_id_reference = [
                (eta, gamma, 0) 
                for eta, gamma in zip(df_reference_order["eta"], df_reference_order["gamma"])
            ]

        # Build ordered list of files to process
        ordered_files_to_process = []
        for eta, gamma, net_id in list_eta_gamma_id_reference:
            # Handle both cases: with and without id in filename
            # if net_id == 0:
            #     filename = f"net_eta{eta}_gamma{gamma}_ruleMatchingIndex.npy"
            # else:
            net_id_str = f"{net_id:03d}"
            filename = f"net_eta{eta}_gamma{gamma}_ruleMatchingIndex_id{net_id_str}.npy"

            if filename in to_analyze_files:
                filepath = networks_dir / filename
                ordered_files_to_process.append(filepath)

        print(f"Processing {len(ordered_files_to_process)} networks in reference order...")

        # Load networks and extract parameters
        generated_networks = []
        generated_network_parameters = []
        filenames = []
        
        for filepath in ordered_files_to_process:
            try:
                network = np.load(filepath)
                params = extract_params_from_filename(filepath.name)
                
                generated_networks.append(network)
                generated_network_parameters.append(params)
                filenames.append(filepath.name)
            except Exception as e:
                print(f"⚠️ Warning: Could not load {filepath.name}: {e}")
                continue
        
        if len(generated_networks) == 0:
            print("❌ No networks to process!")
            return df_prior_results
        
        generated_networks = np.stack(generated_networks)
        empirical_networks = load_empirical_networks(empirical_networks_path)
        
        # Validate dimensions
        if generated_networks.shape[-1] != empirical_networks.shape[-1]:
            raise ValueError(
                f"Dimension mismatch: Generated networks have {generated_networks.shape[-1]} nodes "
                f"but empirical networks have {empirical_networks.shape[-1]} nodes."
            )
        
        # Run comparison
        print("\n" + "=" * 60)
        print("COMPARING NETWORKS")
        print("=" * 60)
        
        # Load distance matrices
        print("Loading distance matrices...")
        distance_matrices = np.load(distance_matrices_path)
        distance_matrices = torch.tensor(distance_matrices, dtype=torch.float32)
        if dataset_name == "shafiei_human_consensus_dataset": 
            distance_matrices = distance_matrices.unsqueeze(0)
            print("Note: Distance matrix unsqueezed for Shafiei dataset")
        print(f"Loaded distance matrix with shape: {distance_matrices.shape}")
        
        # Pre-create evaluation criteria
        evaluation_criteria_list = create_evaluation_criteria_list(
            distance_matrices=distance_matrices,
            config_dict=config_dict
        )
        
        # Convert empirical networks to torch tensor
        print("Converting empirical networks to torch tensors...")
        empirical_networks_tensor = torch.tensor(empirical_networks, dtype=torch.float32)
        
        # Prepare tasks
        print("Preparing tasks...")
        tasks = []
        for net_idx, (network, params, filename) in enumerate(zip(
            generated_networks, 
            generated_network_parameters,
            filenames
        )):
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
        write_header = df_prior_results is None or not output_path.exists()
        
        # Use multiprocessing Pool to process tasks
        results_count = 0
        
        with Pool(processes=number_multiprocessing_processes) as pool:
            for i, result in enumerate(tqdm(
                pool.imap(process_network_optimized, tasks), 
                total=len(tasks), 
                desc="Evaluating networks"
            )):
                # Append result to CSV immediately
                append_result_to_csv(
                    result=result,
                    output_path=output_path,
                    write_header=(write_header and i == 0)
                )
                
                results_count += 1
                
                # Progress update every 10 networks
                if results_count % 10 == 0:
                    print(f"💾 Saved {results_count}/{len(tasks)} results")
        
        print(f"\n✅ All {results_count} new results saved to: {output_path}")
        
        # Load and return final results
        results_df = pd.read_csv(output_path) if output_path.exists() else None

        # Display summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if results_df is not None:
            print(f"Total comparisons in file: {len(results_df)}")
            print(f"Parameter ranges:")
            print(f"  eta: [{results_df['eta'].min():.3f}, {results_df['eta'].max():.3f}]")
            print(f"  gamma: [{results_df['gamma'].min():.3f}, {results_df['gamma'].max():.3f}]")
            if 'id' in results_df.columns:
                print(f"  id range: [{results_df['id'].min():.0f}, {results_df['id'].max():.0f}]")
            print(f"\nFirst few rows of results:")
            print(results_df.head())
        else: 
            print("⚠️ Attention: results_df is empty.")
        
        print("\n✅ Comparison complete!")
        return results_df
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main(experiment_name="76_90000_samples_animal_206") # 75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206") #63_fine_grid_animal_0") # 60_generally_finer_search_animal_0") # 72_clean_10_000") # 71_testing_animal_0")