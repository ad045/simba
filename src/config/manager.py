import torch
import numpy as np

from gnm import fitting, generative_rules

# from config.GNM import GNMConfig, create_weighted_sweep_parameters
# from src.config.ESN import ESNConfig
# from config.data import DataConfig
# from config.compute import ComputeConfig
# from config.path import PathConfig


def create_gnm_sweep_config(config,
                            distance_matrix: torch.Tensor,
                            # num_iterations: int,
                            n_edges: int,
                            mode: str,
                            num_simulations: int,
                            ) -> fitting.SweepConfig:
    """Create GNM sweep configuration with random or grid sampling."""
    
    n_samples = config["experiment"]["search"]["n_samples"]
    eta_range = config['gnm']['eta_range']
    gamma_range = config['gnm']['gamma_range']
    
    # Create parameter value arrays based on mode. TODO: Check if this works correctly?? 
    if mode == "random":
        # For random: create dense array for random.choice to sample from
        pool_size = max(100, n_samples * 2)
        eta_values = torch.linspace(eta_range[0], eta_range[1], pool_size)
        gamma_values = torch.linspace(gamma_range[0], gamma_range[1], pool_size)
    elif mode == "grid":
        # For grid: create grid dimensions
        grid_size = int(np.sqrt(n_samples))
        eta_values = torch.linspace(eta_range[0], eta_range[1], grid_size)
        gamma_values = torch.linspace(gamma_range[0], gamma_range[1], grid_size)
        n_samples = grid_size * grid_size
        print(f"Grid search: {grid_size}x{grid_size} = {n_samples} combinations")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'random' or 'grid'.")
    
    # Get generative rules
    rules = []
    for rule_name in config['gnm']['generative_rules_to_test']:
        if rule_name == "matching_index":
            rules.append(generative_rules.MatchingIndex())
        elif rule_name == "neighbors":
            rules.append(generative_rules.Neighbors())
        elif rule_name == "degree_product":
            rules.append(generative_rules.DegreeProduct())
        elif rule_name == "clustering_coefficient":
            rules.append(generative_rules.ClusteringCoefficient())
        elif rule_name == "spatial":
            rules.append(generative_rules.Spatial())
        else:
            rules.append(generative_rules.MatchingIndex())
    
    density = config["data"]["density"] / 100
    n = distance_matrix.shape[-1]
    
    binary_params = fitting.BinarySweepParameters(
        eta=eta_values,
        gamma=gamma_values,
        lambdah=torch.tensor([config["gnm"]["lambda"]]),
        distance_relationship_type=config["gnm"]["distance_relationship_type"],  # This one is used by GNM library.
        preferential_relationship_type=config["gnm"]["preferential_relationship_type"],
        heterochronicity_relationship_type=config["gnm"]["heterochronicity_relationship_type"],
        generative_rule=rules,
        num_iterations=[n_edges], # int(density * n * n)]
    )
    
    return fitting.SweepConfig(
        binary_sweep_parameters=binary_params,
        num_simulations=num_simulations,
        distance_matrix=[distance_matrix],
        method=mode,
        num_random_samples=n_samples, # number samples for random mode (ignored for grid, defined by GNM library)
    )




    
 