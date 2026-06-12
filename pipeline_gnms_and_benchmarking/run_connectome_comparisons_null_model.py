"""
Analyze how network metrics change as a consensus connectome is progressively
transformed into random chaos through edge rewiring.

Creates 20 independent processes, each showing the degradation from structure to randomness.
"""


# Set multiprocessing method BEFORE any other imports (for MacOS)
import multiprocessing as mp
import os
import sys

if __name__ == "__main__":
    # Force spawn method for macOS compatibility
    mp.set_start_method('spawn', force=True)
    
    # Disable threading in numeric libraries
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Bad fix, but necessary for "ma_thesis_duplicate" env
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
###########################################################################################



# # Set multiprocessing method BEFORE any other imports (for MacOS)
# import multiprocessing as mp
# import os
# import sys

# if __name__ == "__main__":
#     # Force spawn method for macOS compatibility
#     mp.set_start_method('spawn', force=True)
    
#     # Disable threading in numeric libraries
#     os.environ['OMP_NUM_THREADS'] = '1'
#     os.environ['MKL_NUM_THREADS'] = '1'
#     os.environ['OPENBLAS_NUM_THREADS'] = '1'
#     os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
#     os.environ['NUMEXPR_NUM_THREADS'] = '1'
# ###########################################################################################

import numpy as np
import pandas as pd
import torch
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from multiprocessing import Pool
import time

# from src.config.path import PathConfig
# from config.GNM import create_evaluation_criteria
# from src.comparing_connectomes.energy_comparer import EnergyEvaluator
# from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence
# from src.comparing_connectomes.f1_comparer import F1Evaluator
# from src.comparing_connectomes.communicability_comparer import CommunicabilityEvaluator
# from src.comparing_connectomes.delta_con_evaluator import DeltaConEvaluator
# from src.comparing_connectomes.spectral_distance_comparer import SpectralDistanceEvaluator
# from src.comparing_connectomes.edit_distance_comparer import EditDistanceEvaluator
# from src.comparing_connectomes.cosine_embedding_comparer import CosineEmbeddingEvaluator
# from src.comparing_connectomes.wasserstein_gromov_comparer import GromovWassersteinEvaluator
# from src.comparing_connectomes.wasserstein_sinkhorn_comparer import WassersteinSinkhornEvaluator
# from src.comparing_connectomes.hungarian_alignment_comparer import HungarianAlignmentEvaluator
# from src.comparing_connectomes.graph_kernel_comparer import GraphKernelEvaluator
# from src.comparing_connectomes.graph_kernel_networkx_comparer import GraphKernelNetworkxEvaluator
# from src.comparing_connectomes.multiplex_layer_similarity_comparer import MultiplexLayerSimilarityEvaluator
# from src.comparing_connectomes.resistance_distance_comparer import ResistanceDistanceEvaluator
# from src.comparing_connectomes.graph_edit_distance_comparer import GraphEditDistanceEvaluator
# from src.comparing_connectomes.network_mutual_information_comparer import NetworkMutualInformationEvaluator
# from src.comparing_connectomes.communicability_mse_comparer import CommunicabilityMSEEvaluator
# from src.comparing_connectomes.communicability_jsd_comparer import CommunicabilityJSDEvaluator
# from src.comparing_connectomes.frobenius_comparer import FrobeniusEvaluator
# from src.comparing_connectomes.jaccard_comparer import JaccardEvaluator


from src.config.path import PathConfig
from config.GNM import create_evaluation_criteria
from src.comparing_connectomes.base_comparer import NetworkEvaluator
from src.comparing_connectomes.energy_comparer import EnergyEvaluator
from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence
from src.comparing_connectomes.f1_comparer import F1Evaluator
from src.comparing_connectomes.hamming_comparer import HammingEvaluator
from src.comparing_connectomes.communicability_comparer import CommunicabilityCorrEvaluator
# from src.comparing_connectomes.graph_kernel_comparer import GraphKernelEvaluator
from src.comparing_connectomes.spectral_distance_comparer import SpectralDistanceEvaluator
# from src.comparing_connectomes.wasserstein_gromov_comparer import GromovWassersteinEvaluator
# from src.comparing_connectomes.multiplex_layer_similarity_comparer import MultiplexLayerSimilarityEvaluator
from src.comparing_connectomes.cosine_embedding_comparer import CosineEmbeddingEvaluator
from comparing_connectomes.resistance_distance_comparer import ResistanceDistanceEvaluator
from src.comparing_connectomes.delta_con_evaluator import DeltaConEvaluator
from src.comparing_connectomes.delta_con_distance_evaluator import DeltaConDistanceEvaluator
from src.comparing_connectomes.graph_edit_distance_comparer import GraphEditDistanceEvaluator
from src.comparing_connectomes.wasserstein_sinkhorn_comparer import WassersteinSinkhornEvaluator
# from src.comparing_connectomes.hungarian_alignment_comparer import HungarianAlignmentEvaluator
from src.comparing_connectomes.graph_kernel_networkx_comparer import GraphKernelNetworkxEvaluator

from src.comparing_connectomes.network_mutual_information_comparer import NetworkMutualInformationEvaluator, DCNetworkMutualInformationEvaluator

from src.comparing_connectomes.communicability_mse_comparer import CommunicabilityMSEEvaluator
from src.comparing_connectomes.communicability_jsd_comparer import CommunicabilityJSDEvaluator

from src.comparing_connectomes.frobenius_comparer import FrobeniusEvaluator
from src.comparing_connectomes.jaccard_comparer import JaccardEvaluator

from src.comparing_connectomes.netrd_comparer import NetrdEvaluator # resistance, net_smile, net_lsd, quantum_jsd, ...

from src.utils.extract_params_from_filenames import get_eta_gamma_id_from_filename


def load_consensus_connectome(path: Path) -> np.ndarray:
    """Load consensus connectome from .npy file."""
    if not path.exists():
        raise FileNotFoundError(f"Consensus connectome not found: {path}")
    
    connectome = np.load(path)
    print(f"Loaded consensus connectome with shape: {connectome.shape}")
    
    # Handle shape (1, 100, 100) -> (100, 100)
    if len(connectome.shape) == 3 and connectome.shape[0] == 1:
        connectome = connectome[0]
        print(f"Squeezed to shape: {connectome.shape}")
    
    return connectome


def rewire_network_progressively(adj_matrix: np.ndarray, rewire_fraction: float, seed: int = None) -> np.ndarray:
    """
    Rewire a fraction of edges in the network while preserving degree distribution.
    
    Args:
        adj_matrix: Binary adjacency matrix
        rewire_fraction: Fraction of edges to rewire (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Rewired adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to networkx graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Get number of edges to rewire
    num_edges = G.number_of_edges()
    num_rewires = int(num_edges * rewire_fraction)
    
    # Perform double edge swaps (preserves degree distribution)
    # Each swap randomly selects two edges and reconnects them
    try:
        nx.double_edge_swap(G, nswap=num_rewires, max_tries=num_rewires * 100, seed=seed)
    except nx.NetworkXError as e:
        print(f"Warning: Could only perform partial rewiring: {e}")
    
    # Convert back to adjacency matrix
    rewired_adj = nx.to_numpy_array(G)
    
    return rewired_adj.astype(adj_matrix.dtype)


def generate_chaos_sequence(
    original_network: np.ndarray, 
    num_steps: int, 
    process_id: int,
    seed_base: int = 42
) -> List[Dict]:
    """
    Generate a sequence of networks from original to chaos.
    
    Args:
        original_network: Original consensus connectome
        num_steps: Number of steps in the transformation
        process_id: ID for this process (for tracking different runs)
        seed_base: Base seed for reproducibility
        
    Returns:
        List of dicts with step info and network
    """
    sequence = []
    
    # Step 0: Original network
    sequence.append({
        'step': 0,
        'process_id': process_id,
        'rewire_fraction': 0.0,
        'network': original_network.copy()
    })
    
    # Progressive steps toward chaos
    for step in range(1, num_steps + 1):
        # Linearly increase rewiring fraction from 0 to 1
        rewire_fraction = step / num_steps
        
        # Use a unique seed for each (process_id, step) combination
        seed = seed_base + process_id * 10000 + step
        
        rewired_network = rewire_network_progressively(
            original_network, 
            rewire_fraction, 
            seed=seed
        )
        
        sequence.append({
            'step': step,
            'process_id': process_id,
            'rewire_fraction': rewire_fraction,
            'network': rewired_network
        })
    
    return sequence


def evaluate_single_network(args) -> Dict:
    """
    Worker function to evaluate a single network.
    
    Returns dict with: process_id, step, rewire_fraction, metric_value
    """
    step_info, reference_network_tensor, evaluator = args
    
    network = step_info['network']
    network_tensor = torch.tensor(network, dtype=torch.float32)
    
    # Unsqueeze for batch dimension if needed
    if len(network_tensor.shape) == 2:
        network_tensor = network_tensor.unsqueeze(0)
    
    # Reference network should be in batch format (1, N, N)
    ref_batch = reference_network_tensor.unsqueeze(0) if len(reference_network_tensor.shape) == 2 else reference_network_tensor
    
    # Evaluate
    start_time = time.perf_counter()
    metric_dict = evaluator(network_tensor, ref_batch)
    elapsed_time = time.perf_counter() - start_time
    
    # Extract metric value (should be under key 0 since we have one reference)
    metric_value = metric_dict.get(0, np.nan)
    
    result = {
        'process_id': step_info['process_id'],
        'step': step_info['step'],
        'rewire_fraction': step_info['rewire_fraction'],
        'metric_value': metric_value,
        'computation_time': elapsed_time
    }
    
    return result


def initialize_evaluator(evaluation_mode: str, dataset_name: str, path_config: PathConfig):
    """Initialize the appropriate evaluator based on mode."""
    
    if evaluation_mode == "energy":
        # Load distance matrix for energy evaluation
        distance_matrices_path = path_config.dir_02_distance_matrices / "distance_matrix_100.npy"
        print(f"Loading distance matrices from: {distance_matrices_path}")
        
        distance_matrices = np.load(distance_matrices_path)
        distance_matrices = torch.tensor(distance_matrices, dtype=torch.float32)
        
        if dataset_name in ["lexis_data", "hcp_schaefer_100_dataset"]:
            distance_matrices = distance_matrices.unsqueeze(0)
            print("Distance matrix unsqueezed for this dataset")
        
        print(f"Distance matrix shape: {distance_matrices.shape}")
        
        config_dict = {
            'gnm': {
                'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
            }
        }
        
        evaluation_criteria = create_evaluation_criteria(
            config=config_dict,
            distance_matrix=distance_matrices[0]
        )
        
        evaluator = EnergyEvaluator([evaluation_criteria])
        
    elif evaluation_mode == "portrait":
        evaluator = PortraitDivergence()
    elif evaluation_mode == "f1":
        evaluator = F1Evaluator()
    elif evaluation_mode == "hamming":
            evaluator = HammingEvaluator()
            
    elif evaluation_mode == "communicability_corr":
        evaluator = CommunicabilityCorrEvaluator()
    elif evaluation_mode == "delta_con":
        evaluator = DeltaConEvaluator()
    # elif evaluation_mode == "delta_con_distance":
    #     evaluator = DeltaConDistanceEvaluator(distance_matrices[0]) # TODO: This is a cheat rn 
    elif evaluation_mode == "spectral_distance_norm_laplacian":
        evaluator = SpectralDistanceEvaluator(method='normalized_laplacian') # 'adjacency'
    elif evaluation_mode == "spectral_distance_laplacian":
        evaluator = SpectralDistanceEvaluator(method='laplacian')
    elif evaluation_mode == "spectral_distance_adjacency":
        evaluator = SpectralDistanceEvaluator(method='adjacency')
    elif evaluation_mode == "edit_distance":
        evaluator = EditDistanceEvaluator()
    elif evaluation_mode == "cosine_embedding":
        evaluator = CosineEmbeddingEvaluator()
    elif evaluation_mode == "wasserstein_gromov":
        evaluator = GromovWassersteinEvaluator()
    elif evaluation_mode == "wasserstein_sinkhorn":
        evaluator = WassersteinSinkhornEvaluator()
    elif evaluation_mode == "hungarian_alignment":
        evaluator = HungarianAlignmentEvaluator()
    elif evaluation_mode == "graph_kernel":
        evaluator = GraphKernelEvaluator()
    elif evaluation_mode == "graph_kernel_networkx":
        evaluator = GraphKernelNetworkxEvaluator()
    elif evaluation_mode == "multiplex_layer_similarity":
        evaluator = MultiplexLayerSimilarityEvaluator()
    # elif evaluation_mode == "resistance_distance":
    #     evaluator = ResistanceDistanceEvaluator()
    elif evaluation_mode == "graph_edit_distance":
        evaluator = GraphEditDistanceEvaluator()
    elif evaluation_mode == "network_mutual_information":
        evaluator = NetworkMutualInformationEvaluator()
    elif evaluation_mode == "dc_network_mutual_information":
        evaluator = DCNetworkMutualInformationEvaluator()
    elif evaluation_mode == "communicability_mse":
        evaluator = CommunicabilityMSEEvaluator()
    elif evaluation_mode == "communicability_jsd":
        evaluator = CommunicabilityJSDEvaluator()
    elif evaluation_mode == "frobenius":
        evaluator = FrobeniusEvaluator()
    elif evaluation_mode == "jaccard":
        evaluator = JaccardEvaluator()
    elif evaluation_mode == "resistance": # resistance, net_smile, net_lsd, quantum_jsd
        evaluator = NetrdEvaluator(method='resistance')
    elif evaluation_mode == "net_simile":
        evaluator = NetrdEvaluator(method='net_simile')
    elif evaluation_mode == "net_lsd":
        evaluator = NetrdEvaluator(method='net_lsd')
    elif evaluation_mode == "quantum_jsd":
        evaluator = NetrdEvaluator(method='quantum_jsd')
    elif evaluation_mode == "netrd_non_backtracking_spectral": 
        evaluator = NetrdEvaluator(method='netrd_non_backtracking_spectral')
    else:
        raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
    
    print(f"Initialized {evaluation_mode} evaluator")
    return evaluator


def main(
    dataset_name: str = "lexis_data",
    num_processes: int = 20,
    num_steps: int = 50,
    evaluation_mode: str = "energy",
    num_multiprocessing_cores: int = 8,
    seed_base: int = 42
):
    """
    Main function to analyze network degradation to chaos.
    
    Args:
        dataset_name: Name of the dataset
        num_processes: Number of independent chaos processes to run
        num_steps: Number of steps in each process (0=original, num_steps=fully random)
        evaluation_mode: Which metric to use
        num_multiprocessing_cores: Number of CPU cores for parallel processing
        seed_base: Base random seed
    """
    
    print("\n" + "=" * 80)
    print("NETWORK CHAOS TRANSFORMATION ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Evaluation mode: {evaluation_mode}")
    print(f"Number of processes: {num_processes}")
    print(f"Steps per process: {num_steps}")
    print(f"Total networks to evaluate: {num_processes * (num_steps + 1)}")
    
    # Setup paths
    path_config = PathConfig(
        dataset_name=dataset_name,
        experiment_name="chaos_analysis"
    )
    
    # consensus_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/connectome_distances/data/preprocessed/{dataset_name}/01_connectomes/00_connectomes_density10.npy")
    if dataset_name == "lexis_data": 
        consensus_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/connectome_distances/data/preprocessed/{dataset_name}/01_connectomes/00_connectomes_density10.npy")
    elif dataset_name == "hcp_schaefer_100_dataset":
        consensus_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/hcp_schaefer_100_dataset/01_connectomes/01_consensus_bin_density_10_percent_100.npy")
        
    # Load consensus connectome
    print("\n" + "=" * 80)
    print("LOADING CONSENSUS CONNECTOME")
    print("=" * 80)
    original_network = load_consensus_connectome(consensus_path)
    
    # Generate all chaos sequences
    print("\n" + "=" * 80)
    print("GENERATING CHAOS SEQUENCES")
    print("=" * 80)
    
    all_sequences = []
    for process_id in tqdm(range(num_processes), desc="Generating processes"):
        sequence = generate_chaos_sequence(
            original_network, 
            num_steps, 
            process_id,
            seed_base=seed_base
        )
        all_sequences.extend(sequence)
    
    print(f"Generated {len(all_sequences)} network states")
    
    # Initialize evaluator
    print("\n" + "=" * 80)
    print("INITIALIZING EVALUATOR")
    print("=" * 80)
    evaluator = initialize_evaluator(evaluation_mode, dataset_name, path_config)
    
    # Prepare reference network (original consensus)
    reference_network_tensor = torch.tensor(original_network, dtype=torch.float32)
    
    # Prepare evaluation tasks
    print("\n" + "=" * 80)
    print("EVALUATING NETWORKS")
    print("=" * 80)
    
    tasks = [(step_info, reference_network_tensor, evaluator) for step_info in all_sequences]
    
    # Parallel evaluation
    results = []
    with Pool(processes=num_multiprocessing_cores) as pool:
        for result in tqdm(
            pool.imap(evaluate_single_network, tasks),
            total=len(tasks),
            desc="Evaluating"
        ):
            results.append(result)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    output_dir = path_config.output_experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"chaos_analysis_{evaluation_mode}.csv"
    
    df_results.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Saved results to: {output_path}")
    print(f"\nDataFrame shape: {df_results.shape}")
    print(f"\nFirst few rows:")
    print(df_results.head(10))
    print(f"\nLast few rows:")
    print(df_results.tail(10))
    
    # Summary statistics
    print(f"\n" + "=" * 80)
    print("METRIC STATISTICS")
    print("=" * 80)
    print(f"Metric value range: [{df_results['metric_value'].min():.6f}, {df_results['metric_value'].max():.6f}]")
    print(f"Mean metric value: {df_results['metric_value'].mean():.6f}")
    print(f"Std metric value: {df_results['metric_value'].std():.6f}")
    
    # Group by step
    step_summary = df_results.groupby('step')['metric_value'].agg(['mean', 'std', 'min', 'max'])
    print(f"\nMetric by step (first 10 steps):")
    print(step_summary.head(10))
    
    print(f"\nMetric by step (last 10 steps):")
    print(step_summary.tail(10))
    
    # Timing info
    print(f"\n" + "=" * 80)
    print("TIMING STATISTICS")
    print("=" * 80)
    print(f"Average computation time: {df_results['computation_time'].mean():.4f} seconds")
    print(f"Total computation time: {df_results['computation_time'].sum():.2f} seconds")
    
    print("\n✅ Analysis complete!")
    
    return df_results


if __name__ == "__main__":
        
    all_methods = [ # "energy", 
                    # "portrait", 
                    # "spectral_distance_adjacency",
                    #  "communicability_corr",
                    # "net_simile", 
                    # "netrd_non_backtracking_spectral", 
                    # "resistance", 
                    # "delta_con", 
                    
                    # "frobenius", 
                    
        
        # "netrd_non_backtracking_spectral", # add: graph_diffusion!!! 
                    # "energy", 
                    # "portrait", 
                    # "spectral_distance_adjacency",
                    # "spectral_distance_norm_laplacian", 
                    
                    # "communicability_corr",
                    # "communicability_jsd",
                    # "network_mutual_information",
                    # "dc_network_mutual_information", 
                    
                    # "net_simile", 
                    
                    # "resistance", 
                    # "delta_con", 
                    
                    # "f1", 
                    "hamming",
                    # "frobenius", 
                    # "jaccard", 
            
        # Get the name of all measures

    

        # Just copied from the other script thingy - all "runs" do not count... 
        # "f1", # runs!
        # "hamming", # runs!
        # "portrait", # runs!
        # "delta_con", # runs!
        # # "delta_con_distance", # runs! 
        # "spectral_distance_adjacency", # runs! 
        # "spectral_distance_norm_laplacian", # runs!
        # "spectral_distance_laplacian", # runs!
        #                 # "cosine_embedding", # gets stuck: OMP: Error #179: Function pthread_mutex_init failed: OMP: System error #22: Invalid argument for 05. 
        # "wasserstein_sinkhorn", # math errors
        #     # "hungarian_alignment", 
        # "graph_kernel_networkx", # math errors
            # "multiplex_layer_similarity", 
            # "resistance_distance", 
        # "graph_edit_distance", # takes CRAZILY long... stupid error
        # "network_mutual_information", # runs! 
        # "dc_network_mutual_information", # runs!
        # "communicability_mse", # runs, but unsure if it generated any errors? 
        # "communicability_jsd", # runs, but unsure if it generated any errors? 
        # "communicability_corr", # runs, but unsure if it generated any errors? 
        # "frobenius", # runs!
        # "jaccard", # runs!
        
        # "resistance", # runs!
        # "net_simile", # runs!
        # "net_lsd", # runs!
        # "quantum_jsd", # runs, but has overflow complaints 
        
        # "energy", # runs! 
        ]

    successes = [] 
    errors = []
    
    for method in all_methods:
        print(f"\n{'=' * 80}")
        print(f"RUNNING ANALYSIS FOR: {method}")
        print(f"{'=' * 80}")
        
        try: 
            results = main(
                dataset_name="hcp_schaefer_100_dataset", # "lexis_data",
                num_processes=200,  # 20 independent runs
                num_steps=100,  # 50 steps from original to chaos
                evaluation_mode=method,
                num_multiprocessing_cores=10, # 0,
                seed_base=42
            )
            successes.append(method)
        except Exception as e:
            print(f"Error during analysis for {method}: {e}")
            errors.append((method, str(e)))
        
    print(f"\n{'=' * 80}")
    print("ANALYSIS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Successful methods ({len(successes)}): {successes}")
    print(f"Errored methods ({len(errors)}): {errors}")