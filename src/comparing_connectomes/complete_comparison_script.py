"""
Compare generated and empirical networks using modular evaluation strategies.
Supports energy-based and portrait divergence metrics.
Now with incremental subject-by-subject evaluation.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import re
from typing import Dict, List, Tuple, Optional, Set
import csv
from multiprocessing import Pool

from src.config.path import PathConfig
from src.config.GNM import create_evaluation_criteria
from src.comparing_connectomes.base_comparer import NetworkEvaluator
from src.comparing_connectomes.energy_comparer import EnergyEvaluator
from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence
from src.comparing_connectomes.f1_comparer import F1Evaluator
from src.comparing_connectomes.communicability_comparer import CommunicabilityEvaluator


def extract_params_from_filename(filename: str) -> Dict[str, float]:
    """Extract eta, gamma, and id from filename."""
    eta_match = re.search(r'eta([-+]?\d*\.?\d+)', filename)
    if not eta_match:
        raise ValueError(f"Could not extract eta from filename: {filename}")
    eta = float(eta_match.group(1))
    
    gamma_match = re.search(r'gamma([-+]?\d*\.?\d+)', filename)
    if not gamma_match:
        raise ValueError(f"Could not extract gamma from filename: {filename}")
    gamma = float(gamma_match.group(1))
    
    id_match = re.search(r'_id(\d+)', filename)
    net_id = int(id_match.group(1)) if id_match else 0
    
    return {'eta': eta, 'gamma': gamma, 'id': net_id}


def load_empirical_networks(empirical_path: Path) -> np.ndarray:
    """Load empirical networks from .npy file."""
    empirical_path = Path(empirical_path)
    
    if not empirical_path.exists():
        raise FileNotFoundError(f"Empirical networks file not found: {empirical_path}")
    
    empirical_networks = np.load(empirical_path)
    
    print(f"Loaded empirical networks with shape: {empirical_networks.shape}")
    print(f"Number of subjects: {empirical_networks.shape[0]}")
    
    return empirical_networks


def get_processed_subject_ids(output_path: Path, metric_prefix: str, num_subjects: int) -> Set[int]:
    """
    Determine which subject IDs have already been processed by checking column names.
    
    Args:
        output_path: Path to the output CSV file
        metric_prefix: The metric prefix (e.g., 'energy', 'portrait')
        num_subjects: Total number of subjects to expect
    
    Returns:
        Set of subject IDs that have already been processed
    """
    if not output_path.exists():
        return set()
    
    try:
        df = pd.read_csv(output_path, nrows=0)  # Just read the header
        columns = df.columns.tolist()
        
        # Look for columns matching the pattern: {metric_prefix}_subject_{id}
        processed_ids = set()
        pattern = re.compile(rf'{metric_prefix}_subject_(\d+)')
        
        for col in columns:
            match = pattern.match(col)
            if match:
                subject_id = int(match.group(1))
                processed_ids.add(subject_id)
        
        print(f"📊 Found {len(processed_ids)} already processed subjects: {sorted(processed_ids)}")
        return processed_ids
        
    except Exception as e:
        print(f"Warning: Could not check processed subjects: {e}")
        return set()


def load_existing_results(output_path: Path) -> Tuple[Optional[pd.DataFrame], set]:
    """Load existing results and determine which networks have been processed."""
    if not output_path.exists():
        return None, set()
    
    try:
        df = pd.read_csv(output_path)
        if df.empty:
            return None, set()
        
        if 'filename' in df.columns:
            processed = set(df['filename'].values)
        elif 'eta' in df.columns and 'gamma' in df.columns and 'id' in df.columns:
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
    """Pre-create evaluation criteria for each distance matrix."""
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


def evaluate_network_for_subject(
    generated_network_tensor: torch.Tensor,
    empirical_network_tensor: torch.Tensor,
    evaluator: NetworkEvaluator,
    subject_id: int,
) -> Dict[str, float]:
    """Evaluate a single network against a single subject."""
    # The evaluator expects a batch of empirical networks, so we unsqueeze
    empirical_batch = empirical_network_tensor.unsqueeze(0)
    
    metric_dict = evaluator(generated_network_tensor, empirical_batch)
    
    # metric_dict should have one key (index 0) since we only passed one empirical network
    # Return with subject-specific column name
    result = {}
    if 0 in metric_dict:
        column_name = f"{evaluator.metric_prefix}_subject_{subject_id}"
        result[column_name] = metric_dict[0]
    
    return result


def process_network_for_subjects(args: Tuple) -> Dict:
    """Worker function to evaluate a single network against specific subjects."""
    network_idx, network, generated_parameters, filename, \
        empirical_networks_tensor, evaluator, subject_ids_to_process = args
    
    gen_network_tensor = torch.tensor(network, dtype=torch.float32).unsqueeze(0)
    
    results = {
        'network_index': network_idx,
        'filename': filename,
        **generated_parameters,
    }
    
    # Evaluate against each subject that needs processing
    for subject_id in subject_ids_to_process:
        subject_results = evaluate_network_for_subject(
            generated_network_tensor=gen_network_tensor,
            empirical_network_tensor=empirical_networks_tensor[subject_id],
            evaluator=evaluator,
            subject_id=subject_id,
        )
        results.update(subject_results)
    
    return results


def merge_new_columns_to_csv(output_path: Path, new_results: List[Dict]):
    """
    Merge new subject columns into existing CSV file.
    Assumes rows are in the same order (matched by filename).
    """
    if not output_path.exists():
        # No existing file, just write the new results
        if new_results:
            df_new = pd.DataFrame(new_results)
            df_new.to_csv(output_path, index=False)
        return
    
    # Load existing data
    df_existing = pd.read_csv(output_path)
    
    if not new_results:
        return  # Nothing to add
    
    df_new = pd.DataFrame(new_results)
    
    # Merge on filename (or network_index)
    merge_key = 'filename' if 'filename' in df_existing.columns else 'network_index'
    
    # Keep only the new subject columns plus the merge key
    subject_cols = [col for col in df_new.columns if col.startswith(('energy_subject_', 'portrait_subject_'))]
    df_to_merge = df_new[[merge_key] + subject_cols]
    
    # Merge
    df_merged = df_existing.merge(df_to_merge, on=merge_key, how='left')
    
    # Save
    df_merged.to_csv(output_path, index=False)
    print(f"✅ Merged {len(subject_cols)} new subject columns into {output_path}")


def main(dataset_name: str, 
         experiment_name: str,
         evaluation_mode: str = "energy",  # "energy" or "portrait"
         debug_subject_ids: List[int] | None = None, 
         number_multiprocessing_processes: int = 8
         ):
    """Main execution function."""
    
    ##########################################################################
    # CONFIGURATION
    ##########################################################################
    
    path_config = PathConfig( 
        dataset_name=dataset_name,
        experiment_name=experiment_name, 
    )
    
    path_01_connectomes = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/{dataset_name}/01_connectomes") 
    
    if dataset_name == "suarez_MaMI_dataset": 
        empirical_networks_path = path_01_connectomes / "01_consensus_bin_density_10_percent_50.npy"
        distance_matrices_path = path_config.dir_02_distance_matrices / "distance_matrix_50.npy"
    
    if dataset_name == "shafiei_human_consensus_dataset": 
        empirical_networks_path = path_01_connectomes / "01_indiv_connectomes_bin_density_10_percent_68.npy"
        distance_matrices_path = path_config.dir_02_distance_matrices / "distance_matrix_68.npy"
    
    if dataset_name == "hcp_schaefer_100_dataset": 
        empirical_networks_path = path_01_connectomes / "00_connectomes_density10.npy"
        distance_matrices_path = path_config.dir_02_distance_matrices / "distance_matrix_100.npy"
    
    
    ##########################################################################
    
    generated_networks_dir = path_config.output_experiment_dir / "generated_networks"
    
    static_csv = path_config.output_experiment_dir / f"all_static_metrics_for_{experiment_name}.csv"
    dynamic_csv = path_config.output_experiment_dir / f"all_dynamic_metrics_for_{experiment_name}.csv"
    computational_csv = path_config.output_experiment_dir / f"all_computational_metrics_for_{experiment_name}.csv"
    legacy_csv = path_config.output_experiment_dir / f"all_metrics_for_{experiment_name}.csv"
    
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

    config_dict = {
        'gnm': {
            'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
        }
    }
    
    output_dir = path_config.output_experiment_dir
    output_path = output_dir / f"summary_indiv_{evaluation_mode}_for_exp_{experiment_name}.csv"
    
    try:
        # LOAD NETWORKS
        print("\n" + "=" * 60)
        print("LOADING NETWORKS")
        print("=" * 60)

        # Load empirical networks first to know how many subjects we have
        empirical_networks = load_empirical_networks(empirical_networks_path)
        num_subjects = empirical_networks.shape[0]
        
        # Determine metric prefix
        metric_prefix = evaluation_mode  # 'energy' or 'portrait'
        
        # Check which subjects have already been processed
        processed_subject_ids = get_processed_subject_ids(output_path, metric_prefix, num_subjects)
        
        # Determine which subjects still need processing
        all_subject_ids = set(range(num_subjects))
        subjects_to_process = sorted(all_subject_ids - processed_subject_ids)
        
        if debug_subject_ids is not None:
            subjects_to_process = [s for s in subjects_to_process if s in debug_subject_ids]

        print(f"\n📋 Subject processing status:")
        print(f"   Total subjects: {num_subjects}")
        print(f"   Already processed: {len(processed_subject_ids)}")
        print(f"   To process: {len(subjects_to_process)}")
        print(f"   Subject IDs to process: {subjects_to_process}")
        
        if len(subjects_to_process) == 0:
            print("✅ All subjects have been processed!")
            return pd.read_csv(output_path) if output_path.exists() else None

        df_prior_results, already_processed_files = load_existing_results(output_path)

        networks_dir = Path(generated_networks_dir)
        all_network_files = sorted(networks_dir.glob("net_eta*.npy"))
        if not all_network_files:
            raise FileNotFoundError(f"No network files found in {networks_dir}")
        
        # If we have prior results, only process networks that are already in the CSV
        # This ensures we add columns to existing rows
        if df_prior_results is not None and 'filename' in df_prior_results.columns:
            filenames_in_csv = set(df_prior_results['filename'].values)
            network_files_to_process = [f for f in all_network_files if f.name in filenames_in_csv]
            print(f"Found {len(network_files_to_process)} networks already in CSV")
        else:
            # Process all networks if no prior results
            network_files_to_process = all_network_files
            print(f"Found {len(network_files_to_process)} total network files")

        # Load and order networks according to reference CSV
        df_reference_order = pd.read_csv(reference_csv_path)
        
        if 'id' in df_reference_order.columns:
            list_eta_gamma_id_reference = list(zip(
                df_reference_order["eta"], 
                df_reference_order["gamma"],
                df_reference_order["id"]
            ))
        else:
            list_eta_gamma_id_reference = [
                (eta, gamma, 0) 
                for eta, gamma in zip(df_reference_order["eta"], df_reference_order["gamma"])
            ]

        ordered_files_to_process = []
        for eta, gamma, net_id in list_eta_gamma_id_reference:
            net_id_str = f"{net_id:03d}"
            filename = f"net_eta{eta}_gamma{gamma}_ruleMatchingIndex_id{net_id_str}.npy"
            filepath = networks_dir / filename
            
            if filepath.exists():
                ordered_files_to_process.append(filepath)

        print(f"Processing {len(ordered_files_to_process)} networks in reference order...")

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
        
        if generated_networks.shape[-1] != empirical_networks.shape[-1]:
            raise ValueError(
                f"Dimension mismatch: Generated networks have {generated_networks.shape[-1]} nodes "
                f"but empirical networks have {empirical_networks.shape[-1]} nodes."
            )
        
        # INITIALIZE EVALUATOR
        print("\n" + "=" * 60)
        print("INITIALIZING EVALUATOR")
        print("=" * 60)
        
        if evaluation_mode == "portrait":
            evaluator = PortraitDivergence()
            print("Using Portrait Divergence evaluation")
        elif evaluation_mode == "f1":
            evaluator = F1Evaluator()
            print("Using F1 evaluation")
        elif evaluation_mode == "communicability":
            evaluator = CommunicabilityEvaluator()
            print("Using Communicability evaluation")
        elif evaluation_mode == "energy":
            print("Loading distance matrices...")
            distance_matrices = np.load(distance_matrices_path)
            distance_matrices = torch.tensor(distance_matrices, dtype=torch.float32)
            if dataset_name in ["shafiei_human_consensus_dataset", "hcp_schaefer_100_dataset"]: 
                distance_matrices = distance_matrices.unsqueeze(0)
                print("Note: Distance matrix unsqueezed for this dataset")
            print(f"Loaded distance matrix with shape: {distance_matrices.shape}")
            
            evaluation_criteria_list = create_evaluation_criteria_list(
                distance_matrices=distance_matrices,
                config_dict=config_dict
            )
            evaluator = EnergyEvaluator(evaluation_criteria_list)
            print("Using Energy evaluation")    
        else:
            raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
        
        # COMPARE NETWORKS
        print("\n" + "=" * 60)
        print(f"COMPARING NETWORKS AGAINST {len(subjects_to_process)} NEW SUBJECTS")
        print("=" * 60)
        
        print("Converting empirical networks to torch tensors...")
        empirical_networks_tensor = torch.tensor(empirical_networks, dtype=torch.float32)
        
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
                evaluator,
                subjects_to_process  # Only evaluate against subjects that need processing
            )
            tasks.append(task_args)
        
        print(f"\nProcessing {len(tasks)} networks against {len(subjects_to_process)} subjects...")
        
        all_results = []
        
        with Pool(processes=number_multiprocessing_processes) as pool:
            for result in tqdm(
                pool.imap(process_network_for_subjects, tasks), 
                total=len(tasks), 
                desc="Evaluating networks"
            ):
                all_results.append(result)
        
        # Merge new columns into existing CSV
        merge_new_columns_to_csv(output_path, all_results)
        
        print(f"\n✅ All results saved to: {output_path}")
        
        results_df = pd.read_csv(output_path) if output_path.exists() else None

        # SUMMARY
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if results_df is not None:
            print(f"Total networks in file: {len(results_df)}")
            
            # Count how many subjects have been fully processed
            subject_columns = [col for col in results_df.columns if col.startswith(f'{metric_prefix}_subject_')]
            num_subjects_in_file = len(subject_columns)
            
            print(f"Subjects processed: {num_subjects_in_file}/{num_subjects}")
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
    results = main(
        dataset_name="suarez_MaMI_dataset", # hcp_schaefer_100_dataset",
        experiment_name="80_more_animals_animal_169", # 103", # 06_with_seeds", # 05_second_big_overnight_run_10201", 
        evaluation_mode="portrait", # communicability", # f1", # portrait",  # or "energy"
        debug_subject_ids=None, # [0,1,2,3,4,5,6,7,8,9] #None,  # turn this to "None" if not in debug mode. Or array [1,2,]
        number_multiprocessing_processes=8, 
    )
    
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/80_more_animals_animal_103/all_metrics_for_80_more_animals_animal_103.csv