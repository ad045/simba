"""
Script to compute missing metrics and add them to existing CSV files.
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import re
from tqdm import tqdm

# Import metric computation modules
import structural_metrics as sm
import dynamics_metrics as dm
import computation_metrics as cm


# Metric registry: maps metric names to computation functions
METRIC_REGISTRY = {
    # Structural
    'avg_clustering': lambda A, G, dist: sm.compute_avg_clustering(A, G),
    'consensus_modularity': lambda A, G, dist: sm.compute_consensus_modularity(A, n_iterations=100),
    'small_world_omega': lambda A, G, dist: sm.compute_small_world_omega(G, niter=5, nrand=10),
    'avg_wiring_cost': lambda A, G, dist: sm.compute_avg_wiring_cost(G, dist),
    'total_wiring_cost': lambda A, G, dist: sm.compute_total_wiring_cost(G, dist),
    'hubness_gini': lambda A, G, dist: sm.compute_hubness_gini(G),
    'rich_club_max': lambda A, G, dist: sm.compute_rich_club_coefficient(G, k=None)[0],
    'rich_club_k_max': lambda A, G, dist: sm.compute_rich_club_coefficient(G, k=None)[1],
    'richclub_n_edges': lambda A, G, dist: sm.compute_richclub_stats(G, dist)[0],
    'richclub_avg_length': lambda A, G, dist: sm.compute_richclub_stats(G, dist)[1],
    'avg_degree': lambda A, G, dist: sm.compute_avg_degree(G),
    'transitivity': lambda A, G, dist: sm.compute_transitivity(G),
    'degree_assortativity': lambda A, G, dist: sm.compute_degree_assortativity(G),
    'distance_dependent_assortativity': lambda A, G, dist: sm.compute_distance_dependent_assortativity(G, dist),
    'matrix_entropy': lambda A, G, dist: sm.compute_matrix_entropy(A),
    
    # Dynamics
    'avg_communicability': lambda A, G, dist: dm.compute_communicability(A, mode="estrada_scaled"),
    'spectral_radius': lambda A, G, dist: dm.compute_spectral_radius(A),
    'eigenspectrum_entropy': lambda A, G, dist: dm.compute_eigenspectrum_entropy(A),
    'spectral_gap': lambda A, G, dist: dm.compute_spectral_gap(A),
    'global_efficiency': lambda A, G, dist: dm.compute_global_efficiency(G),
    'diffusion_efficiency': lambda A, G, dist: dm.compute_diffusion_efficiency(A),
    'avg_controllability': lambda A, G, dist: dm.compute_avg_controllability(A),
    
    # Computation
    'structural_complexity': lambda A, G, dist: cm.compute_structural_complexity(A),
    'effective_dimensionality': lambda A, G, dist: cm.compute_effective_dimensionality(A),
    'kernel_rank': lambda A, G, dist: cm.compute_kernel_rank(A),
}


def load_network_from_path(network_path):
    """Load a network from .npy file."""
    A = np.load(network_path)
    
    # Ensure symmetry and zero diagonal
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    
    return A


def parse_network_filename(filename):
    """
    Extract eta and gamma from filename like:
    'net_eta-0.673828125_gamma0.8398066163063049_ruleMatchingIndex.npy'
    """
    eta_match = re.search(r'eta-?([-\d.]+)', filename)
    gamma_match = re.search(r'gamma([-\d.]+)', filename)
    
    eta = float(eta_match.group(1)) if eta_match else None
    gamma = float(gamma_match.group(1)) if gamma_match else None
    
    return eta, gamma


def find_network_path(base_dir, eta, gamma, rule):
    """
    Find the network file path given eta, gamma, and rule.
    """
    generated_networks_dir = Path(base_dir) / "generated_networks"
    
    if not generated_networks_dir.exists():
        return None
    
    # Search for file matching eta and gamma
    for fname in os.listdir(generated_networks_dir):
        if not fname.endswith('.npy'):
            continue
        
        file_eta, file_gamma = parse_network_filename(fname)
        
        # Match with tolerance
        if (file_eta is not None and file_gamma is not None and
            abs(file_eta - eta) < 1e-6 and abs(file_gamma - gamma) < 1e-6):
            return generated_networks_dir / fname
    
    return None


def compute_missing_metrics(csv_path, distance_matrix, metrics_to_compute, 
                           base_dir=None, verbose=True):
    """
    Main function to compute missing metrics and update CSV.
    
    Args:
        csv_path: Path to the CSV file with existing metrics
        distance_matrix: (n, n) array of Euclidean distances
        metrics_to_compute: List of metric names to compute
        base_dir: Base directory containing generated_networks folder
        verbose: Whether to print progress
    """
    if base_dir is None:
        base_dir = Path(csv_path).parent
    
    # Load existing CSV
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    # Identify which metrics are missing or incomplete
    missing_metrics = []
    for metric in metrics_to_compute:
        if metric not in df.columns:
            missing_metrics.append(metric)
            df[metric] = np.nan
        elif df[metric].isna().any():
            missing_metrics.append(metric)
    
    if verbose:
        print(f"Metrics to compute: {missing_metrics}")
    
    if not missing_metrics:
        print("All requested metrics already present!")
        return df
    
    # Process each row
    rows_to_update = df[df[missing_metrics].isna().any(axis=1)].index
    
    if verbose:
        print(f"Processing {len(rows_to_update)} rows with missing data")
    
    for idx in tqdm(rows_to_update, disable=not verbose):
        row = df.loc[idx]
        
        # Get network parameters
        eta = row['eta']
        gamma = row['gamma']
        rule = row.get('generative_rule', 'MatchingIndex')
        
        # Find and load network
        network_path = find_network_path(base_dir, eta, gamma, rule)
        
        if network_path is None:
            if verbose:
                print(f"Warning: Could not find network for eta={eta}, gamma={gamma}")
            continue
        
        try:
            # Load network
            A = load_network_from_path(network_path)
            A_bin = (A != 0).astype(float)
            G = nx.from_numpy_array(A_bin)
            
            # Compute each missing metric for this row
            for metric in missing_metrics:
                # Skip if already computed
                if pd.notna(df.loc[idx, metric]):
                    continue
                
                if metric not in METRIC_REGISTRY:
                    if verbose:
                        print(f"Warning: No computation function for metric '{metric}'")
                    continue
                
                