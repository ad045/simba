
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
                                    num_iterations: int,
                                    num_simulations: int, 
                                    include_weights: bool, 
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
        
        binary_params = fitting.BinarySweepParameters(
            eta=eta_values,
            gamma=gamma_values,
            lambdah=torch.tensor([config["gnm"]["lambda"]]), 
            distance_relationship_type=config["gnm"]["distance_relationship_type"], 
            preferential_relationship_type=config["gnm"]["preferential_relationship_type"], 
            heterochronicity_relationship_type=config["gnm"]["heterochronicity_relationship_type"],
            generative_rule=rules,
            num_iterations=[num_iterations],
        )
        
        weighted_params = None
        if include_weights:
            weighted_params = create_weighted_sweep_parameters(config, distance_matrix)

        return fitting.SweepConfig(
            binary_sweep_parameters=binary_params,
            weighted_sweep_parameters=weighted_params,
            num_simulations=num_simulations,
            distance_matrix=[distance_matrix], 
            method=config["experiment"]["search"]["method"], 
            num_random_samples=n_random_samples, 
        )


class ConfigManager:
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


    
 