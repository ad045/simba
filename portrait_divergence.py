"""
Portrait divergence calculation integrated with the existing pipeline.
"""

import numpy as np
import torch
import networkx as nx
from pathlib import Path
from typing import Union, Optional


def calculate_portrait_divergence(
    generated_network: Union[np.ndarray, torch.Tensor],
    empirical_network: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Calculate portrait divergence between a generated and empirical network.
    Uses NetworkX graph conversion for compatibility with portrait_divergence.py
    
    Args:
        generated_network: Generated network adjacency matrix (n_nodes, n_nodes)
        empirical_network: Empirical network adjacency matrix (n_nodes, n_nodes)
        
    Returns:
        Portrait divergence value (JSD)
    """
    # Import here to avoid circular dependencies
    from network_portrait_divergence import portrait_divergence
    
    # Convert to numpy if torch tensor
    if isinstance(generated_network, torch.Tensor):
        generated_network = generated_network.cpu().numpy()
    if isinstance(empirical_network, torch.Tensor):
        empirical_network = empirical_network.cpu().numpy()
    
    # Ensure 2D
    if generated_network.ndim == 3:
        generated_network = generated_network.squeeze()
    if empirical_network.ndim == 3:
        empirical_network = empirical_network.squeeze()
    
    # Ensure binary and symmetric
    generated_network = (generated_network > 0).astype(float)
    empirical_network = (empirical_network > 0).astype(float)
    generated_network = np.maximum(generated_network, generated_network.T)
    empirical_network = np.maximum(empirical_network, empirical_network.T)
    
    # Convert to NetworkX graphs
    G_gen = nx.from_numpy_array(generated_network)
    G_emp = nx.from_numpy_array(empirical_network)
    
    # Calculate portrait divergence
    # The portrait_divergence function handles graph->portrait conversion internally
    divergence = portrait_divergence(G_gen, G_emp)
    
    return float(divergence)


def calculate_portrait_divergence_batch(
    generated_networks: np.ndarray,
    empirical_network: np.ndarray,
) -> np.ndarray:
    """
    Calculate portrait divergence for a batch of generated networks.
    
    Args:
        generated_networks: Array of shape (n_networks, n_nodes, n_nodes) or (n_nodes, n_nodes)
        empirical_network: Single empirical network (n_nodes, n_nodes)
        
    Returns:
        Array of divergence values, shape (n_networks,)
    """
    if generated_networks.ndim == 2:
        generated_networks = generated_networks[np.newaxis, ...]
    
    n_networks = generated_networks.shape[0]
    divergences = np.zeros(n_networks)
    
    for i in range(n_networks):
        try:
            divergences[i] = calculate_portrait_divergence(
                generated_networks[i],
                empirical_network,
            )
        except Exception as e:
            print(f"Error calculating portrait divergence for network {i}: {e}")
            divergences[i] = np.nan
    
    return divergences


def load_networks_for_portrait(
    generated_network_path: Path,
    empirical_networks_path: Path,
    animal_id: int
) -> tuple:
    """
    Load generated and empirical networks for portrait calculation.
    
    Args:
        generated_network_path: Path to generated network .npy file
        empirical_networks_path: Path to empirical networks .npy file (shape: n_animals, n_nodes, n_nodes)
        animal_id: ID of the animal/subject
        
    Returns:
        Tuple of (generated_network, empirical_network)
    """
    # Load generated network
    generated = np.load(generated_network_path)
    if generated.ndim == 3:
        generated = generated[0]  # Take first if multiple simulations
    
    # Load empirical networks and select the correct one
    empirical_all = np.load(empirical_networks_path)
    empirical = empirical_all[animal_id]
    
    return generated, empirical


# Integration with existing evaluation pipeline
def add_portrait_to_evaluation(
    network_path: Path,
    empirical_networks_path: Path,
    animal_id: int,
    result_dict: dict
) -> dict:
    """
    Add portrait divergence to existing result dictionary.
    
    Args:
        network_path: Path to generated network
        empirical_networks_path: Path to empirical networks
        animal_id: Animal/subject ID
        result_dict: Existing result dictionary (will be modified in place)
        
    Returns:
        Updated result dictionary
    """
    try:
        generated, empirical = load_networks_for_portrait(
            network_path,
            empirical_networks_path,
            animal_id
        )
        
        divergence = calculate_portrait_divergence(generated, empirical)
        result_dict['portrait_divergence'] = divergence
        
    except Exception as e:
        print(f"Error calculating portrait divergence for {network_path.name}: {e}")
        result_dict['portrait_divergence'] = np.nan
    
    return result_dict