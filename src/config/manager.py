
from typing import List, Optional
import torch

# Import GNM configuration structures
from gnm import fitting, generative_rules

from config.GNM import GNMConfig, create_weighted_sweep_parameters
from src.config.ESN import ESNConfig
from config.data import DataConfig
from config.compute import ComputeConfig
from config.path import PathConfig

def create_gnm_random_sweep_config(config, #  TODO: Combine with function below (and thus make the label "random" and "grid" useable...)
                                    distance_matrix: torch.Tensor,
                                    num_iterations: int, # TODO: Turn this into how many matrices are GENERATED per run... 
                                    num_simulations: int, 
                                    ) -> fitting.SweepConfig:
        """Create GNM sweep configuration with random parameter sampling."""
        
        n_random_samples = config["experiment"]["search"]["n_samples"]
        
        # Generate random parameter values
        eta_values = torch.empty(n_random_samples)
        gamma_values = torch.empty(n_random_samples)
        
        eta_range = config['gnm']['eta_range']
        gamma_range = config['gnm']['gamma_range']

        for i in range(n_random_samples):
            eta_values[i] = torch.rand(1) * (eta_range[1] - eta_range[0]) + eta_range[0]
            gamma_values[i] = torch.rand(1) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]

        # Get generative rules
        rules = []
        for rule_name in config['gnm']['generative_rules_to_test']: # self.gnm.generative_rules_to_test:
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
        
        
        
        density = config["data"]["density"] / 100  # TODO: Turn everything into non-percent but true values?
        n = distance_matrix.shape[-1]
        
        binary_params = fitting.BinarySweepParameters(
            eta=eta_values,
            gamma=gamma_values,
            lambdah=torch.tensor([config["gnm"]["lambda"]]), 
            distance_relationship_type=config["gnm"]["distance_relationship_type"], 
            preferential_relationship_type=config["gnm"]["preferential_relationship_type"], 
            heterochronicity_relationship_type=config["gnm"]["heterochronicity_relationship_type"],
            generative_rule=rules,
            num_iterations=[int(density * n * n)] # TODO: check this! NUMBER EDGES!!!!  If this is set to 10, then "added_edges" will have in the end 9 "elements": torch.Size([9, 1, 2]) # , [num_iterations],
        )
        
        return fitting.SweepConfig(
            binary_sweep_parameters=binary_params,
            num_simulations=num_simulations, # Number of simulations to run in parallel. Each simulation generates a separate network using the same parameters.
            distance_matrix=[distance_matrix], # still the correct distance matrix. 
            method=config["experiment"]["search"]["method"], 
            num_random_samples=n_random_samples, 
        )


class ConfigManager: # is this used? Yup. 
    """Optimized configuration manager using GNM structures."""
    
    
    def __init__(self, 
                 dataset_name: str, 
                 resolution: int, 
                 density: int, 
                 use_weighted: bool,
                 esn_config: Optional[ESNConfig] = None,
                 gnm_config: Optional[GNMConfig] = None,
                 data_config: Optional[DataConfig] = None,
                 compute_config: Optional[ComputeConfig] = None,
                 path_config: Optional[PathConfig] = None):
        
        self.esn = esn_config or ESNConfig()
        self.gnm = gnm_config or GNMConfig()
        self.data = data_config or DataConfig(
                                dataset_name=dataset_name, 
                                resolution=resolution, 
                                densities=density,
                                use_weighted=use_weighted
        )
        self.compute = compute_config or ComputeConfig()
        self.paths = path_config or PathConfig()


    
 