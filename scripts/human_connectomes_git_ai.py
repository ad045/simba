#!/usr/bin/env python3
"""
Script to evaluate preprocessed empirical connectomes.

This script:
1. Loads each of the 70 preprocessed human connectomes and the associated distance matrix
2. Calculates the Memory Capacity (MC) for each connectome using an Echo State Network (ESN)
3. Computes a suite of structural graph theory metrics (efficiency, modularity, wiring cost)
4. Estimates the best eta and gamma parameters for each network
5. Aggregates all results into a single CSV file matching the "run_gnm_example" pipeline format

Author: AI Assistant
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import argparse
import sys
import json
from datetime import datetime
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add src to path for local imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import sys
import os

# Add the project's root directory to the Python path
# This allows us to import from the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now, your original import will work
from src.structural_analysis.graph_measures import analyze_connectomes 



try:
    from src.structural_analysis.graph_measures import analyze_connectomes
    # from src.preprocessing.preprocess_70_connectomes import get_individual_connectomes
    # from src.preprocessing.preprocess_distance_matrix import get_distance_matrix
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}")
    print("Will use simplified implementations")


class SimpleESN:
    """
    Simplified Echo State Network implementation for Memory Capacity evaluation.
    This is a minimal version that doesn't require external dependencies.
    """
    
    def __init__(self, reservoir_size: int, spectral_radius: float = 0.99, 
                 input_scaling: float = 1.0, leak_rate: float = 1.0, 
                 bias: float = 1.0, random_state: int = 42):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.bias = bias
        self.random_state = random_state
        
        # Initialize random state
        self.rng = np.random.default_rng(random_state)
        
        # Will be set during fit
        self.W_out = None
        self.states = None
        
    def _scale_spectral_radius(self, W: np.ndarray) -> np.ndarray:
        """Scale the reservoir matrix to have the desired spectral radius."""
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W = W * (self.spectral_radius / current_radius)
        return W
    
    def _run_reservoir(self, inputs: np.ndarray, W_reservoir: np.ndarray) -> np.ndarray:
        """Run the reservoir dynamics."""
        n_samples, n_inputs = inputs.shape
        states = np.zeros((n_samples, self.reservoir_size))
        x = np.zeros(self.reservoir_size)
        
        # Input weight matrix (random)
        W_in = self.rng.uniform(-1, 1, (self.reservoir_size, n_inputs)) * self.input_scaling
        
        for t in range(n_samples):
            # Reservoir update equation
            u = np.tanh(W_reservoir @ x + W_in @ inputs[t] + self.bias)
            x = (1 - self.leak_rate) * x + self.leak_rate * u
            states[t] = x
            
        return states
    
    def fit(self, X: np.ndarray, y: np.ndarray, W_reservoir: Optional[np.ndarray] = None):
        """Fit the ESN using ridge regression."""
        if W_reservoir is None:
            # Create random reservoir
            W_reservoir = self.rng.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))
            W_reservoir = self._scale_spectral_radius(W_reservoir)
        else:
            # Use provided reservoir (connectome)
            W_reservoir = self._scale_spectral_radius(W_reservoir.copy())
            self.reservoir_size = W_reservoir.shape[0]
        
        # Run reservoir
        states = self._run_reservoir(X, W_reservoir)
        self.states = states
        
        # Fit output weights using ridge regression
        # Add bias term to states
        states_with_bias = np.column_stack([states, np.ones(states.shape[0])])
        
        try:
            # Ridge regression with small regularization
            lambda_reg = 1e-6
            self.W_out = np.linalg.solve(
                states_with_bias.T @ states_with_bias + lambda_reg * np.eye(states_with_bias.shape[1]),
                states_with_bias.T @ y
            )
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.W_out = np.linalg.pinv(states_with_bias) @ y
    
    def predict(self, X: np.ndarray, W_reservoir: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions."""
        if W_reservoir is None:
            raise ValueError("W_reservoir must be provided for prediction")
        
        W_reservoir = self._scale_spectral_radius(W_reservoir.copy())
        states = self._run_reservoir(X, W_reservoir)
        states_with_bias = np.column_stack([states, np.ones(states.shape[0])])
        
        return states_with_bias @ self.W_out


def generate_memory_capacity_data(train_len: int, test_len: int, n_lags: int, 
                                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate dataset for memory capacity evaluation."""
    rng = np.random.default_rng(random_state)
    total_len = train_len + test_len + n_lags + 100
    
    # Generate random input sequence
    seq = rng.uniform(-0.5, 0.5, size=(total_len,))
    
    # Build target sequences (delayed versions)
    def build_targets(x: np.ndarray, lags: int) -> np.ndarray:
        T = len(x) - lags
        targets = np.zeros((T, lags), dtype=float)
        for i in range(lags):
            targets[:, i] = x[lags - (i + 1): -((i + 1)) if i + 1 > 0 else None]
        return targets
    
    Y_full = build_targets(seq, n_lags)
    
    # Split into training and testing
    start_train = 100
    end_train = start_train + train_len
    
    X_train = seq[start_train:end_train].reshape(-1, 1)
    Y_train = Y_full[start_train:end_train]
    X_test = seq[end_train:end_train + test_len].reshape(-1, 1)
    Y_test = Y_full[end_train:end_train + test_len]
    
    return X_train, Y_train, X_test, Y_test


def evaluate_memory_capacity(connectome: np.ndarray, spectral_radius: float = 0.99,
                           n_lags: int = 50, train_len: int = 4000, test_len: int = 1000,
                           n_runs: int = 10, input_scaling: float = 1.0,
                           random_state: int = 42) -> Dict[str, float]:
    """
    Evaluate memory capacity of a connectome using ESN.
    
    Args:
        connectome: Adjacency matrix of the network
        spectral_radius: Desired spectral radius for the reservoir
        n_lags: Number of memory lags to test
        train_len: Length of training sequence
        test_len: Length of test sequence
        n_runs: Number of independent runs
        input_scaling: Input scaling factor
        random_state: Random seed
        
    Returns:
        Dictionary with memory capacity results
    """
    mc_values = []
    
    for run in range(n_runs):
        # Generate data
        X_train, Y_train, X_test, Y_test = generate_memory_capacity_data(
            train_len, test_len, n_lags, random_state + run
        )
        
        # Create and fit ESN
        esn = SimpleESN(
            reservoir_size=connectome.shape[0],
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            random_state=random_state + run
        )
        
        esn.fit(X_train, Y_train, connectome)
        Y_pred = esn.predict(X_test, connectome)
        
        # Calculate memory capacity
        mc_run = 0.0
        for lag in range(n_lags):
            if lag < Y_test.shape[1] and lag < Y_pred.shape[1]:
                # Pearson correlation coefficient
                corr = np.corrcoef(Y_test[:, lag], Y_pred[:, lag])[0, 1]
                if not np.isnan(corr):
                    mc_run += corr**2
        
        mc_values.append(mc_run)
    
    return {
        'mc_mean': np.mean(mc_values),
        'mc_std': np.std(mc_values),
        'mc_values': mc_values
    }


# def compute_graph_metrics(connectome: np.ndarray, distance_matrix: np.ndarray) -> Dict[str, float]:
#     """
#     Compute structural graph theory metrics for a connectome.
    
#     Args:
#         connectome: Binary adjacency matrix
#         distance_matrix: Distance matrix between nodes
        
#     Returns:
#         Dictionary of graph metrics
#     """
#     # Ensure binary and symmetric
#     A = np.maximum(connectome, connectome.T)
#     A = (A > 0).astype(float)
#     np.fill_diagonal(A, 0)
    
#     # Create NetworkX graph
#     G = nx.from_numpy_array(A)
    
#     if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
#         return {
#             'global_efficiency': 0.0,
#             'modularity': 0.0,
#             'average_clustering': 0.0,
#             'average_path_length': np.inf,
#             'transitivity': 0.0,
#             'density': 0.0,
#             'average_degree': 0.0,
#             'wiring_cost': 0.0
#         }
    
#     # Global efficiency
#     try:
#         glob_eff = nx.global_efficiency(G)
#     except:
#         glob_eff = 0.0
    
#     # Modularity
#     try:
#         communities = list(nx_comm.greedy_modularity_communities(G))
#         modularity = nx_comm.modularity(G, communities) if len(communities) > 1 else 0.0
#     except:
#         modularity = 0.0
    
#     # Clustering
#     try:
#         avg_clustering = nx.average_clustering(G)
#     except:
#         avg_clustering = 0.0
    
#     # Path length (for largest connected component)
#     try:
#         if nx.is_connected(G):
#             avg_path_length = nx.average_shortest_path_length(G)
#         else:
#             # Use largest connected component
#             largest_cc = max(nx.connected_components(G), key=len)
#             subgraph = G.subgraph(largest_cc)
#             avg_path_length = nx.average_shortest_path_length(subgraph)
#     except:
#         avg_path_length = np.inf
    
#     # Transitivity
#     try:
#         transitivity = nx.transitivity(G)
#     except:
#         transitivity = 0.0
    
#     # Basic network properties
#     density = nx.density(G)
#     avg_degree = np.mean([d for n, d in G.degree()])
    
#     # Wiring cost (average edge length)
#     edge_lengths = [distance_matrix[u, v] for u, v in G.edges()]
#     wiring_cost = np.mean(edge_lengths) if edge_lengths else 0.0
    
#     return {
#         'global_efficiency': glob_eff,
#         'modularity': modularity,
#         'average_clustering': avg_clustering,
#         'average_path_length': avg_path_length,
#         'transitivity': transitivity,
#         'density': density,
#         'average_degree': avg_degree,
#         'wiring_cost': wiring_cost
#     }
    
    


def estimate_optimal_parameters(connectome: np.ndarray, distance_matrix: np.ndarray,
                              eta_range: Tuple[float, float] = (-3.0, 0.0),
                              gamma_range: Tuple[float, float] = (0.1, 1.0),
                              n_points: int = 10) -> Dict[str, float]:
    """
    Estimate optimal eta and gamma parameters for a network.
    
    This is a simplified version that doesn't require the full GNM library.
    It uses basic heuristics based on network properties.
    
    Args:
        connectome: Binary adjacency matrix
        distance_matrix: Distance matrix between nodes
        eta_range: Range for eta parameter (distance penalty)
        gamma_range: Range for gamma parameter (topology preference)
        n_points: Number of points to sample in parameter space
        
    Returns:
        Dictionary with estimated optimal parameters
    """
    # Calculate network properties
    metrics = analyze_connectomes(connectome, distance_matrix)
    metrics = metrics[0] # as we only have one connectome to analyze
    
    # Append results for this network
    # out.append({
    #     "network_index": idx,
    #     "avg_communicability": avg_comm,
    #     "global_efficiency": glob_eff,
    #     "modularity": modu,
    #     "avg_clustering": avg_clust,
    #     "avg_degree": avg_deg,
    #     "transitivity": trans,
    #     "avg_edge_distance": avg_dist,
    #     "wiring_cost": wiring_cost,
    #     "char_path_length": cpl,
    #     "richclub_n_edges": n_rich_edges,
    #     "richclub_avg_length": avg_rc_length,
    # })
    
    
    # Simple heuristics for parameter estimation
    # These are approximations based on typical GNM behavior
    
    # Eta (distance penalty): networks with higher wiring cost need stronger distance penalty
    wiring_cost = metrics['wiring_cost']
    if wiring_cost > 0:
        # Normalize wiring cost and map to eta range
        eta_norm = min(1.0, wiring_cost / np.mean(distance_matrix))
        best_eta = eta_range[0] + eta_norm * (eta_range[1] - eta_range[0])
    else:
        best_eta = np.mean(eta_range)
    
    # Gamma (topology): networks with higher clustering prefer higher gamma
    clustering = metrics['avg_clustering']
    modularity = metrics['modularity']
    topology_score = (clustering + modularity) / 2
    best_gamma = gamma_range[0] + topology_score * (gamma_range[1] - gamma_range[0])
    
    # Estimate energy (simplified KS distance approximation)
    # This is a rough approximation - real GNM would use proper KS test
    G = nx.from_numpy_array(connectome)
    density = nx.density(G)
    density_diff = abs(density - 0.15)  # Assuming target density ~15% -> is fixed.... 
    clustering_diff = abs(clustering - 0.3)  # Assuming target clustering ~30%
    best_energy_ai_way = density_diff + clustering_diff # is this the energy calculation we want? 
    
    return {
        'best_eta': best_eta,
        'best_gamma': best_gamma,
        'best_energy_ai_way': best_energy_ai_way,
        'wiring_cost': wiring_cost,
        'topology_score': topology_score, 
        'density': density, 
    }


def load_connectome_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed connectome data and distance matrix.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        Tuple of (connectomes, distance_matrix)
    """
    
    # Load the first available files
    RESOLUTION = 68
    base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/data/preprocessed/01_first_analysises")
    connectomes = np.load(base_path / f"connectomes_weighted_{RESOLUTION}x{RESOLUTION}.npy")
    distance_matrix = np.load(base_path / f"distance_matrix_{RESOLUTION}x{RESOLUTION}.npy")
    
    print(f"Loaded connectomes: {connectomes.shape}")
    print(f"Loaded distance matrix: {distance_matrix.shape}")
    
    # If connectomes is 2D, add subject dimension
    if connectomes.ndim == 2:
        connectomes = connectomes[np.newaxis, :, :]
    
    return connectomes, distance_matrix



def main():
    """Main function to evaluate empirical connectomes."""
    parser = argparse.ArgumentParser(description="Evaluate empirical connectomes")
    parser.add_argument("--data-dir", type=Path, 
                       help="Directory containing preprocessed connectome data")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                       help="Output directory for results")
    parser.add_argument("--n-subjects", type=int, default=70,
                       help="Number of subjects to process")
    parser.add_argument("--spectral-radius", type=float, default=0.99,
                       help="ESN spectral radius")
    parser.add_argument("--n-runs", type=int, default=10,
                       help="Number of ESN runs per subject")
    parser.add_argument("--n-lags", type=int, default=50,
                       help="Number of memory lags to test")
    parser.add_argument("--create-sample-data", action="store_true",
                       help="Create sample data for testing")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    

    h_params = {
        'spectral_radius': args.spectral_radius,
        'n_runs': args.n_runs,
        'n_lags': args.n_lags
    }
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_dir}")
    try:
        connectomes, distance_matrix = load_connectome_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use --create-sample-data to create sample data for testing")
        return 1
    
    n_subjects, n_regions, _ = connectomes.shape
    print(f"Processing {n_subjects} subjects with {n_regions} regions each")
    
    # Initialize results storage
    results = []
    
    # Process each subject
    start_time = time.time()
    for subj_idx in range(n_subjects):
        if args.verbose:
            print(f"\nProcessing subject {subj_idx + 1}/{n_subjects}")
        
        connectome = connectomes[subj_idx]
        
        # Ensure connectome is binary and symmetric
        connectome = np.maximum(connectome, connectome.T)
        connectome = (connectome > 0).astype(float)
        np.fill_diagonal(connectome, 0)
        
        # Skip if no connections
        if np.sum(connectome) == 0:
            if args.verbose:
                print(f"  Skipping subject {subj_idx}: no connections")
            continue
        
        try:
            # 1. Calculate Memory Capacity
            if args.verbose:
                print("  Computing memory capacity...")
            mc_results = evaluate_memory_capacity(
                connectome=connectome,
                spectral_radius=args.spectral_radius,
                n_lags=args.n_lags,
                n_runs=args.n_runs
            )
            
            # 2. Compute structural metrics
            if args.verbose:
                print("  Computing graph metrics...")
            graph_metrics = analyze_connectomes(connectome, distance_matrix)[0]
            
            # 3. Estimate optimal parameters
            if args.verbose:
                print("  Estimating optimal parameters...")
            param_results = estimate_optimal_parameters(connectome, distance_matrix)
            
            # 4. Aggregate results
            result = {
                'subject': subj_idx
            }
            result.update(param_results)
            result.update(graph_metrics)
            
            
            #     'n_regions': n_regions,
            #     'density_percent': graph_metrics['density'] * 100,
            #     'spectral_radius': args.spectral_radius,
            #     'n_runs': args.n_runs,
            #     'n_lags': args.n_lags,
                
            #     # Memory capacity results
            #     'mc_mean': mc_results['mc_mean'],
            #     'mc_std': mc_results['mc_std'],
            #     'mc_values': str(mc_results['mc_values']),
                
            #     # Graph metrics # TODO!!!
            #     'global_efficiency': graph_metrics['global_efficiency'],
            #     'modularity': graph_metrics['modularity'],
            #     'average_clustering': graph_metrics['average_clustering'],
            #     'average_path_length': graph_metrics['average_path_length'],
            #     'transitivity': graph_metrics['transitivity'],
            #     'average_degree': graph_metrics['average_degree'],
            #     'wiring_cost': graph_metrics['wiring_cost'],
                
            #     # Optimal parameters
            #     'best_eta': param_results['best_eta'],
            #     'best_gamma': param_results['best_gamma'],
            #     'best_energy': param_results['best_energy'],
                
            #     # Timestamp
            #     'timestamp': datetime.now().isoformat()
            # }
            
            results.append(result)
            
            if args.verbose:
                print(f"  MC: {mc_results['mc_mean']:.3f}±{mc_results['mc_std']:.3f}")
                print(f"  Efficiency: {graph_metrics['global_efficiency']:.3f}")
                print(f"  Modularity: {graph_metrics['modularity']:.3f}")
                print(f"  Eta: {param_results['best_eta']:.3f}, Gamma: {param_results['best_gamma']:.3f}")
        
        except Exception as e:
            print(f"  Error processing subject {subj_idx}: {e}")
            continue
    
    # Save results
    if results:
        df_results = pd.DataFrame(results)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output_dir / f"empirical_connectome_results_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)
        
        # Save summary statistics
        # summary_file = args.output_dir / f"empirical_connectome_summary_{timestamp}.json"
        # summary = {
        #     'n_subjects_processed': len(results),
        #     'n_subjects_total': n_subjects,
        #     'n_regions': n_regions,
        #     'processing_time_seconds': time.time() - start_time,
        #     'mc_mean_across_subjects': float(df_results['mc_mean'].mean()),
        #     'mc_std_across_subjects': float(df_results['mc_mean'].std()),
        #     'efficiency_mean': float(df_results['global_efficiency'].mean()),
        #     'modularity_mean': float(df_results['modularity'].mean()),
        #     'eta_mean': float(df_results['best_eta'].mean()),
        #     'gamma_mean': float(df_results['best_gamma'].mean()),
        #     'parameters': {
        #         'spectral_radius': args.spectral_radius,
        #         'n_runs': args.n_runs,
        #         'n_lags': args.n_lags
        #     }
        # }
        
        # with open(summary_file, 'w') as f:
        #     json.dump(summary, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  CSV: {output_file}")
        # print(f"  Summary: {summary_file}")
        print(f"\nProcessed {len(results)} subjects in {time.time() - start_time:.1f} seconds")
        # print(f"Average Memory Capacity: {summary['mc_mean_across_subjects']:.3f}±{summary['mc_std_across_subjects']:.3f}")
        
        return 0
    else:
        print("No subjects were successfully processed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())