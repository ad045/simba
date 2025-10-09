"""
Compare generated and empirical networks with another and creates big csv fole. 
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import re
from typing import Dict, List, Tuple

# Import the evaluation criteria function
from src.config.GNM import create_evaluation_criteria
from gnm.fitting import RunConfig
from gnm.model import BinaryGenerativeParameters


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


def load_generated_networks(networks_dir: Path) -> List[Tuple[Dict, np.ndarray]]:
    """
    Load all generated networks from directory.
    
    Args:
        networks_dir: Directory containing .npy network files
        
    Returns:
        List of tuples (params_dict, network_array)
    """
    networks_dir = Path(networks_dir)
    network_files = sorted(networks_dir.glob("net_eta*.npy"))
    
    if not network_files:
        raise FileNotFoundError(f"No network files found in {networks_dir}")
    
    print(f"Found {len(network_files)} generated network files")
    
    networks = []
    network_parameters = []
    for net_file in network_files:
        try:
            params = extract_params_from_filename(net_file.name)
            network = np.load(net_file) # [0,:,:]
            networks.append(network)
            network_parameters.append(params)
        except Exception as e:
            print(f"Warning: Could not load {net_file.name}: {e}")
            continue
    networks = np.concatenate(networks)
    
    return networks, network_parameters


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


def evaluate_network_against_empirical(
    generated_network: np.ndarray,
    empirical_networks: np.ndarray,
    evaluation_criteria,
) -> Dict[str, float]:
    """
    Evaluate a single generated network against all empirical networks.
    
    Args:
        generated_network: Single generated network (n_nodes, n_nodes)
        empirical_networks: Empirical networks (n_subjects, n_nodes, n_nodes)
        evaluation_criteria: Evaluation criteria object
        
    Returns:
        Dictionary with energy values for each empirical network
    """
    n_subjects = empirical_networks.shape[0]
    
    # Convert to torch tensors
    gen_network_tensor = torch.tensor(
        generated_network,
        dtype=torch.float32,
    )
    
    empirical_tensor = torch.tensor(
        empirical_networks,
        dtype=torch.float32,
    )
    
    # Evaluate against each empirical network
    results = {}
    
    # Get the energy metric name from evaluation_criteria
    # The evaluation_criteria returns a dict with energy metric names as keys
    # We need to evaluate for each subject
    for subject_idx in range(n_subjects):
        empirical_single = empirical_tensor[subject_idx:subject_idx+1, :, :]  # Keep 3D shape
        
        # Expand generated network to match batch dimension
        gen_single = gen_network_tensor.unsqueeze(0) # TODO: Add again??
        
        # Calculate energy using the evaluation criteria
        energy_dict = evaluation_criteria(gen_single, empirical_single)
        
        # Store results for this subject
        # for energy_metric_name, energy_value in energy_dict.items():
        for energy_value in energy_dict: # w.items():
            column_name = f"MaxCrit(etc)_indiv_{subject_idx}"
            results[column_name] = float(energy_value.mean().item())
    
    return results


def compare_all_networks(
    generated_networks: List[Tuple[Dict, np.ndarray]],
    generated_network_parameters, 
    empirical_networks: np.ndarray,
    distance_matrices_path: Path = None,
    config_dict: Dict = None,
    output_path: Path = None
) -> pd.DataFrame:
    """
    Compare all generated networks with empirical networks.
    
    Args:
        generated_networks: List of (params, network) tuples
        empirical_networks: Empirical networks array
        distance_matrix_path: Path to distance matrix .npy file
        config_dict: Optional config dictionary for custom evaluation metrics
        output_path: Path to save results CSV
        
    Returns:
        DataFrame with comparison results
    """
    
    # Load distance matrix if provided
    distance_matrices = None
    if distance_matrices_path and Path(distance_matrices_path).exists():
        distance_matrices = np.load(distance_matrices_path) # [0,:,:] # TODO: Remove this hardcoding! 
        distance_matrices = torch.tensor(distance_matrices, dtype=torch.float32)
        print(f"Loaded distance matrix with shape: {distance_matrices.shape}")
    
    # Collect all results
    all_results = []
    
    print(f"\nComparing {len(generated_networks)} generated networks with {empirical_networks.shape[0]} empirical networks...")
    
    for(network, generated_parameters, distance_matrix) in tqdm(zip(generated_networks, generated_network_parameters, distance_matrices), desc="Evaluating networks"):
        
        
        # Create evaluation criteria
        if config_dict is not None:
            # Use config to create evaluation criteria
            evaluation_criteria = create_evaluation_criteria(
                config=config_dict,
                distance_matrix=distance_matrix if distance_matrix is not None else torch.zeros((100, 100))
            )
        else:
            # Use default evaluation criteria
            from gnm import evaluation
            
            if distance_matrix is not None:
                evaluation_criteria = evaluation.MaxCriteria(
                    evaluation.DegreeKS(),
                    evaluation.ClusteringKS(),
                    evaluation.EdgeLengthKS(distance_matrix),
                    evaluation.BetweennessKS()
                )
            else:
                # Without distance matrix, skip edge_length_ks
                evaluation_criteria = evaluation.MaxCriteria(
                    evaluation.DegreeKS(),
                    evaluation.ClusteringKS(),
                    evaluation.BetweennessKS()
                )
                print("Warning: No distance matrix provided, skipping EdgeLengthKS metric")
        
        
        
        # Evaluate this network against all empirical networks
        energy_results = evaluate_network_against_empirical(
            generated_network=network,
            empirical_networks=empirical_networks,
            evaluation_criteria=evaluation_criteria,
        )
        
        # Combine parameters with results
        row = {**generated_parameters, **energy_results}
        all_results.append(row)
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Sort by eta and gamma for easier reading
    df_results = df_results.sort_values(['eta', 'gamma']).reset_index(drop=True)
    
    # Save results if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False, na_rep="nan")
        print(f"\n✅ Results saved to: {output_path}")
    
    return df_results


def main():
    """Main execution function."""
    
    # Define paths
    generated_networks_dir = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/31_suarez_MaMI_size_100_copy/generated_networks")
    # generated_networks_dir = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/31_suarez_MaMI_size_100/generated_networks")
    empirical_networks_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_100.npy")
    
    # Distance matrix path (adjust if you have it)
    distance_matrix_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_100.npy")
    
    # Optional: Load a config for custom evaluation metrics
    # If you want to use specific evaluation metrics from your config
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
        
        generated_networks, generated_network_parameters = load_generated_networks(generated_networks_dir)
        empirical_networks = load_empirical_networks(empirical_networks_path)
        
        # THIS IS ONLY FOR TESTING!! REMOVE BEFORE USE. 
        generated_networks = generated_networks # [:10, :, :]
        empirical_networks = empirical_networks[:8, :, :]
        
        # Validate dimensions
        if generated_networks is not None:
            
            if generated_networks.shape[-1] != empirical_networks.shape[-1]:
                raise ValueError(
                    f"Dimension mismatch: Generated networks and empirical networks have differently many nodes."
                )
            
        # Run comparison
        print("\n" + "=" * 60)
        print("COMPARING NETWORKS")
        print("=" * 60)
        
        results_df = compare_all_networks(
            generated_networks=generated_networks,
            generated_network_parameters=generated_network_parameters, 
            empirical_networks=empirical_networks,
            distance_matrices_path=distance_matrix_path if distance_matrix_path.exists() else None,
            config_dict=config_dict,
            output_path=output_path
        )
        
        # Display summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total comparisons: {len(results_df)}")
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