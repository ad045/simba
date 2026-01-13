"""
Analyze how network metrics change as important edges are progressively removed.

Creates multiple independent processes, each removing edges in order of importance.
Supports multiple importance metrics: betweenness, efficiency drop, rich club.
"""

# Set multiprocessing method BEFORE any other imports (for MacOS)
import multiprocessing as mp
import os
import sys

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from multiprocessing import Pool
import time
from scipy.stats import rankdata

from src.config.path import PathConfig
from config.GNM import create_evaluation_criteria
from src.comparing_connectomes.energy_comparer import EnergyEvaluator
from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence
from src.comparing_connectomes.f1_comparer import F1Evaluator
from src.comparing_connectomes.hamming_comparer import HammingEvaluator
from src.comparing_connectomes.communicability_comparer import CommunicabilityCorrEvaluator
from src.comparing_connectomes.spectral_distance_comparer import SpectralDistanceEvaluator
from src.comparing_connectomes.cosine_embedding_comparer import CosineEmbeddingEvaluator
from comparing_connectomes.resistance_distance_comparer import ResistanceDistanceEvaluator
from src.comparing_connectomes.delta_con_evaluator import DeltaConEvaluator
from src.comparing_connectomes.graph_edit_distance_comparer import GraphEditDistanceEvaluator
from src.comparing_connectomes.wasserstein_sinkhorn_comparer import WassersteinSinkhornEvaluator
from src.comparing_connectomes.graph_kernel_networkx_comparer import GraphKernelNetworkxEvaluator
from src.comparing_connectomes.network_mutual_information_comparer import (
    NetworkMutualInformationEvaluator, DCNetworkMutualInformationEvaluator
)
from src.comparing_connectomes.communicability_mse_comparer import CommunicabilityMSEEvaluator
from src.comparing_connectomes.communicability_jsd_comparer import CommunicabilityJSDEvaluator
from src.comparing_connectomes.frobenius_comparer import FrobeniusEvaluator
from src.comparing_connectomes.jaccard_comparer import JaccardEvaluator
from src.comparing_connectomes.netrd_comparer import NetrdEvaluator


def load_consensus_connectome(path: Path) -> np.ndarray:
    """Load consensus connectome from .npy file."""
    if not path.exists():
        raise FileNotFoundError(f"Consensus connectome not found: {path}")
    
    connectome = np.load(path)
    print(f"Loaded consensus connectome with shape: {connectome.shape}")
    
    if len(connectome.shape) == 3 and connectome.shape[0] == 1:
        connectome = connectome[0]
        print(f"Squeezed to shape: {connectome.shape}")
    
    return connectome


def compute_edge_betweenness(adj_matrix: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Compute edge betweenness centrality for all edges.
    
    Higher values = more important for shortest paths.
    """
    G = nx.from_numpy_array(adj_matrix)
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)
    return edge_betweenness


def compute_efficiency_drop(adj_matrix: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Compute importance as drop in global efficiency when edge is removed.
    
    Higher values = more important for communication efficiency.
    """
    G = nx.from_numpy_array(adj_matrix)
    original_efficiency = nx.global_efficiency(G)
    
    importance = {}
    edges = list(G.edges())
    
    for edge in tqdm(edges, desc="Computing efficiency drops", leave=False):
        G_temp = G.copy()
        G_temp.remove_edge(*edge)
        new_efficiency = nx.global_efficiency(G_temp)
        importance[edge] = original_efficiency - new_efficiency
    
    return importance


def compute_rich_club_importance(adj_matrix: np.ndarray, k_threshold: int = None) -> Dict[Tuple[int, int], float]:
    """
    Compute importance based on rich club membership.
    
    Edges between high-degree nodes get higher importance.
    If k_threshold is None, uses median degree.
    """
    G = nx.from_numpy_array(adj_matrix)
    degrees = dict(G.degree())
    
    if k_threshold is None:
        k_threshold = np.median(list(degrees.values()))
    
    importance = {}
    for edge in G.edges():
        i, j = edge
        # Higher if both nodes are hubs
        importance[edge] = min(degrees[i], degrees[j])
    
    return importance


def compute_communicability_importance(adj_matrix: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Compute importance based on communicability contribution.
    
    Uses the communicability matrix element for each edge.
    """
    G = nx.from_numpy_array(adj_matrix)
    
    try:
        comm_matrix = nx.communicability_exp(G)
    except:
        # Fallback if graph is disconnected
        comm_matrix = {}
        for node in G.nodes():
            comm_matrix[node] = {node: 1.0}
    
    importance = {}
    for edge in G.edges():
        i, j = edge
        # Use communicability between endpoints
        importance[edge] = comm_matrix.get(i, {}).get(j, 0.0)
    
    return importance


def rank_edges_by_importance(
    adj_matrix: np.ndarray, 
    method: str = 'betweenness',
    seed: int = None
) -> List[Tuple[int, int]]:
    """
    Rank all edges by importance, from most to least important.
    
    Args:
        adj_matrix: Binary adjacency matrix
        method: 'betweenness', 'efficiency', 'rich_club', 'communicability', or 'random'
        seed: Random seed for tie-breaking
        
    Returns:
        List of edges sorted by importance (descending)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if method == 'random':
        # Random ordering as baseline
        G = nx.from_numpy_array(adj_matrix)
        edges = list(G.edges())
        np.random.shuffle(edges)
        return edges, {edge: np.random.random() for edge in edges} # Fake thing for importance 
    
    # Compute importance based on method
    if method == 'betweenness':
        importance = compute_edge_betweenness(adj_matrix)
    elif method == 'efficiency':
        importance = compute_efficiency_drop(adj_matrix)
    elif method == 'rich_club':
        importance = compute_rich_club_importance(adj_matrix)
    elif method == 'communicability':
        importance = compute_communicability_importance(adj_matrix)
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    # Sort edges by importance (descending)
    sorted_edges = sorted(importance.items(), key=lambda x: (-x[1], np.random.random()))
    edges_only = [edge for edge, _ in sorted_edges]
    
    return edges_only, importance 


def remove_edges_progressively(
    original_network: np.ndarray,
    num_steps: int,
    importance_method: str = 'betweenness',
    process_id: int = 0,
    seed_base: int = 42
) -> List[Dict]:
    """
    Generate a sequence of networks with progressively more important edges removed.
    
    Args:
        original_network: Original consensus connectome
        num_steps: Number of steps in the degradation
        importance_method: Method to rank edge importance
        process_id: ID for this process
        seed_base: Base seed for reproducibility
        
    Returns:
        List of dicts with step info and network
    """
    seed = seed_base + process_id * 10000
    
    # Rank edges by importance
    print(f"Process {process_id}: Ranking edges by {importance_method}...")
    ranked_edges, importance = rank_edges_by_importance(
        original_network, 
        method=importance_method,
        seed=seed
    )
    
    sequence = []
    current_network = original_network.copy()

    for step, edge in enumerate(ranked_edges): 
        
        # Get fresh network 
        current_network = original_network.copy()
        
        # Remove edge 
        i, j = edge
        current_network[i, j] = 0
        current_network[j, i] = 0  # Ensure symmetry
        
        sequence.append({
            'step': step,
            'process_id': process_id,
            'removal_fraction': 0.0,
            'edges_removed': 0,
            'network': current_network.copy(),
            'importance_method': importance_method
        })
    
    return sequence, importance 

    # total_edges = len(ranked_edges)
    # print(f"Process {process_id}: Total edges = {total_edges}")
    
    # sequence = []
    # current_network = original_network.copy()
    
    # # Step 0: Original network
    
    # # Progressive edge removal
    # for step in range(1, num_steps + 1):
    #     # Determine how many edges to remove at this step
    #     removal_fraction = step / num_steps
    #     target_edges_removed = int(total_edges * removal_fraction)
    #     edges_to_remove_now = target_edges_removed - (step - 1) * (total_edges // num_steps)
        
    #     # Remove the next most important edges
    #     for _ in range(edges_to_remove_now):
            
            
    #         if (step - 1) * (total_edges // num_steps) + _ < len(ranked_edges):
    #             edge = ranked_edges[(step - 1) * (total_edges // num_steps) + _]
    #             i, j = edge
    #             current_network[i, j] = 0
    #             current_network[j, i] = 0  # Ensure symmetry
        
    #     sequence.append({
    #         'step': step,
    #         'process_id': process_id,
    #         'removal_fraction': removal_fraction,
    #         'edges_removed': target_edges_removed,
    #         'network': current_network.copy(),
    #         'importance_method': importance_method
    #     })
    
    # return sequence


def evaluate_single_network(args) -> Dict:
    """Worker function to evaluate a single network."""
    step_info, reference_network_tensor, evaluator = args
    
    network = step_info['network']
    network_tensor = torch.tensor(network, dtype=torch.float32)
    
    if len(network_tensor.shape) == 2:
        network_tensor = network_tensor.unsqueeze(0)
    
    ref_batch = reference_network_tensor.unsqueeze(0) if len(reference_network_tensor.shape) == 2 else reference_network_tensor
    
    start_time = time.perf_counter()
    metric_dict = evaluator(network_tensor, ref_batch)
    elapsed_time = time.perf_counter() - start_time
    
    metric_value = metric_dict.get(0, np.nan)
    
    result = {
        'process_id': step_info['process_id'],
        'step': step_info['step'],
        'removal_fraction': step_info['removal_fraction'],
        'edges_removed': step_info['edges_removed'],
        'importance_method': step_info['importance_method'],
        'metric_value': metric_value,
        'computation_time': elapsed_time
    }
    
    return result


def initialize_evaluator(evaluation_mode: str, dataset_name: str, path_config: PathConfig):
    """Initialize the appropriate evaluator based on mode."""
    
    if evaluation_mode == "energy":
        distance_matrices_path = path_config.dir_02_distance_matrices / "distance_matrix_100.npy"
        print(f"Loading distance matrices from: {distance_matrices_path}")
        
        distance_matrices = np.load(distance_matrices_path)
        distance_matrices = torch.tensor(distance_matrices, dtype=torch.float32)
        
        if dataset_name in ["lexis_data", "hcp_schaefer_100_dataset"]:
            distance_matrices = distance_matrices.unsqueeze(0)
        
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
    elif evaluation_mode == "spectral_distance_norm_laplacian":
        evaluator = SpectralDistanceEvaluator(method='normalized_laplacian')
    elif evaluation_mode == "spectral_distance_laplacian":
        evaluator = SpectralDistanceEvaluator(method='laplacian')
    elif evaluation_mode == "spectral_distance_adjacency":
        evaluator = SpectralDistanceEvaluator(method='adjacency')
    elif evaluation_mode == "cosine_embedding":
        evaluator = CosineEmbeddingEvaluator()
    elif evaluation_mode == "wasserstein_sinkhorn":
        evaluator = WassersteinSinkhornEvaluator()
    elif evaluation_mode == "graph_kernel_networkx":
        evaluator = GraphKernelNetworkxEvaluator()
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
    elif evaluation_mode == "resistance":
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
    importance_method: str = 'betweenness',
    evaluation_mode: str = "energy",
    num_multiprocessing_cores: int = 8,
    seed_base: int = 42
):
    """
    Main function to analyze network degradation by removing important edges.
    
    Args:
        dataset_name: Name of the dataset
        num_processes: Number of independent degradation processes to run
        num_steps: Number of steps in each process
        importance_method: How to rank edges ('betweenness', 'efficiency', 'rich_club', 'communicability', 'random')
        evaluation_mode: Which metric to use
        num_multiprocessing_cores: Number of CPU cores for parallel processing
        seed_base: Base random seed
    """
    
    print("\n" + "=" * 80)
    print("NETWORK DEGRADATION BY EDGE IMPORTANCE")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Importance method: {importance_method}")
    print(f"Evaluation mode: {evaluation_mode}")
    print(f"Number of processes: {num_processes}")
    print(f"Steps per process: {num_steps}")
    print(f"Total networks to evaluate: {num_processes * (num_steps + 1)}")
    
    # Setup paths
    path_config = PathConfig(
        dataset_name=dataset_name,
        experiment_name=f"importance_degradation_{importance_method}"
    )
    
    consensus_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/connectome_distances/data/preprocessed/{dataset_name}/01_connectomes/00_connectomes_density10.npy")
    
    # Load consensus connectome
    print("\n" + "=" * 80)
    print("LOADING CONSENSUS CONNECTOME")
    print("=" * 80)
    original_network = load_consensus_connectome(consensus_path)
    
    # Generate all degradation sequences
    print("\n" + "=" * 80)
    print("GENERATING EDGE REMOVAL SEQUENCES")
    print("=" * 80)
    
    all_sequences = []
    for process_id in tqdm(range(num_processes), desc="Generating processes"):
        sequence, importance = remove_edges_progressively(
            original_network,   
            num_steps,
            importance_method=importance_method,
            process_id=process_id,
            seed_base=seed_base
        )
        all_sequences.extend(sequence)
    
    print(f"Generated {len(all_sequences)} network states")
    
    # Save importance scores for reference
    importance_output_path = path_config.output_experiment_dir / f"edge_importance_{importance_method}.csv"
    importance_df = pd.DataFrame(list(importance.items()), columns=['edge', 'importance'])
    importance_output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_output_path, index=False)
    
    # Initialize evaluator
    print("\n" + "=" * 80)
    print("INITIALIZING EVALUATOR")
    print("=" * 80)
    evaluator = initialize_evaluator(evaluation_mode, dataset_name, path_config)
    
    # Prepare reference network
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
    output_path = output_dir / f"importance_degradation_{importance_method}_{evaluation_mode}.csv"
    
    df_results.to_csv(output_path, index=False)
    print(output_path)
    
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
    
    # Test different importance methods
    importance_methods = [
        # "betweenness",      # Edges on shortest paths
        # "efficiency",       # Edges critical for communication
        # "rich_club",        # Connections between hubs
        # "communicability",  # High communicability edges
        "random"            # Baseline control
    ]
    
    evaluation_methods = [
        "energy", 
        "portrait", 
        "spectral_distance_adjacency",
        
        "net_simile", 
        "netrd_non_backtracking_spectral", 
        "resistance", 
        "delta_con", 
        
        "frobenius", 
    ]
    
    successes = []
    errors = []
    
    for importance_method in importance_methods:
        for eval_method in evaluation_methods:
            print(f"\n{'=' * 80}")
            print(f"RUNNING: {importance_method} importance + {eval_method} evaluation")
            print(f"{'=' * 80}")
            
            try:
                results = main(
                    dataset_name="lexis_data",
                    num_processes=20,
                    num_steps=50,
                    importance_method=importance_method,
                    evaluation_mode=eval_method,
                    num_multiprocessing_cores=8,
                    seed_base=42
                )
                successes.append((importance_method, eval_method))
            except Exception as e:
                print(f"Error: {e}")
                errors.append((importance_method, eval_method, str(e)))
    
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Successful: {len(successes)}")
    print(f"Errors: {len(errors)}")
    if errors:
        print("\nError details:")
        for imp, ev, err in errors:
            print(f"  {imp} + {ev}: {err}")