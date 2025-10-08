
from typing import List, Dict, Any, Optional
import torch

# Import GNM configuration structures
from gnm import fitting, generative_rules

from config.GNM import GNMConfig
from src.config.ESN import ESNConfig
from config.data import DataConfig
from config.compute import ComputeConfig
from config.path import PathConfig

# Import project constants
from .constants import (
    DEFAULT_RESOLUTION, DEFAULT_RANDOM_SEED, DEFAULT_N_ETA, DEFAULT_N_GAMMA, DEFAULT_N_LAMBDA,
    DEFAULT_SPECTRAL_RADIUS, DEFAULT_INPUT_LENGTH, DEFAULT_INPUT_SCALING, DEFAULT_N_RUNS,
    DEFAULT_N_LAGS, DEFAULT_TEST_LENGTH, DEFAULT_N_TRANSIENT, DEFAULT_LEAK_RATE, DEFAULT_BIAS,
    DEFAULT_DENSITIES, DEFAULT_REGULARIZATION_METHOD, DEFAULT_GENERATIVE_RULES,
    DEFAULT_EVALUATION_METRICS, DEFAULT_WEIGHT_CRITERION, DEFAULT_ETA_RANGE, DEFAULT_GAMMA_RANGE,
    DEFAULT_LAMBDA_RANGE, DEFAULT_ALPHA, DEFAULT_NUM_SIMULATIONS, DEFAULT_APPEND_INTERVAL,
    CONNECTOMES_WEIGHTED_PATTERN, CONNECTOMES_BINARY_PATTERN, DISTANCE_MATRIX_PATTERN
)

class ConfigManager:
    """Optimized configuration manager using GNM structures."""
    
    
    def __init__(self, 
                 esn_config: Optional[ESNConfig] = None,
                 gnm_config: Optional[GNMConfig] = None,
                 data_config: Optional[DataConfig] = None,
                 compute_config: Optional[ComputeConfig] = None,
                 path_config: Optional[PathConfig] = None):
        
        self.esn = esn_config or ESNConfig()
        self.gnm = gnm_config or GNMConfig()
        self.data = data_config or DataConfig()
        self.compute = compute_config or ComputeConfig()
        self.paths = path_config or PathConfig()
    
    
    def create_gnm_random_sweep_config(self, # TODO: Combine with function below (and thus make the label "random" and "grid" useable...)
                                    distance_matrix: torch.Tensor,
                                    num_iterations: int,
                                    num_simulations: int, 
                                    method: Optional[str], #  = "random", # "random", # "grid",
                                    n_random_samples: int, #  = 30,
                                    include_weights: bool, 
                                    ) -> fitting.SweepConfig:
        """Create GNM sweep configuration with random parameter sampling."""
        
        # Generate random parameter values
        eta_values = torch.empty(n_random_samples)
        gamma_values = torch.empty(n_random_samples)
        
        for i in range(n_random_samples):
            eta_values[i] = torch.rand(1) * (self.gnm.eta_range[1] - self.gnm.eta_range[0]) + self.gnm.eta_range[0]
            gamma_values[i] = torch.rand(1) * (self.gnm.gamma_range[1] - self.gnm.gamma_range[0]) + self.gnm.gamma_range[0]

        # Get generative rules
        rules = []
        for rule_name in self.gnm.generative_rules_to_test:
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
            lambdah=torch.tensor([0.0]),  # Single lambda value
            distance_relationship_type=["powerlaw"],
            preferential_relationship_type=["powerlaw"],
            heterochronicity_relationship_type=["powerlaw"],
            generative_rule=rules,
            num_iterations=[num_iterations],
        )
        
        weighted_params = None
        if include_weights:
            weighted_params = self.gnm.create_weighted_sweep_parameters(distance_matrix)

        return fitting.SweepConfig(
            binary_sweep_parameters=binary_params,
            weighted_sweep_parameters=weighted_params,
            num_simulations=num_simulations,
            distance_matrix=[distance_matrix], 
            method=method, 
            num_random_samples=n_random_samples, 
        )


    def create_gnm_sweep_config(self, 
                               distance_matrix: torch.Tensor,
                               num_iterations: int,
                               num_simulations: int = 100, # TODO: check if this works 
                               method="grid", # is always this - combine with function above... 
                               include_weights: bool = True) -> fitting.SweepConfig:
        """Create complete GNM sweep configuration."""    
                
        binary_params = self.gnm.create_binary_sweep_parameters(distance_matrix, num_iterations)
        
        weighted_params = None
        if include_weights:
            weighted_params = self.gnm.create_weighted_sweep_parameters(distance_matrix)
        
        return fitting.SweepConfig(
            binary_sweep_parameters=binary_params,
            weighted_sweep_parameters=weighted_params,
            num_simulations=self.gnm.num_simulations,
            distance_matrix=[distance_matrix]
        )
    
    def get_gnm_evaluation_criteria(self, distance_matrix: torch.Tensor) -> Any:
        """Get evaluation criteria for GNM."""
        return self.gnm.create_evaluation_criteria(distance_matrix)
    
    def load_gnm_defaults(self):
        """Load default data from GNM library."""
        from gnm import defaults
        
        device = torch.device(self.compute.device)
        
        # Load default distance matrix and network from GNM
        distance_matrix = defaults.get_distance_matrix(device=device)
        binary_network = defaults.get_binary_network(device=device)
        
        return {
            "distance_matrix": distance_matrix,
            "binary_network": binary_network
        }
    
    def get_data_paths(self):
        """Get paths to data files."""
        
        resolution = self.data.resolution
        
        paths = {
            'weighted_connectome': self.paths.data_dir / CONNECTOMES_WEIGHTED_PATTERN.format(resolution=resolution),
            'distance_matrix': self.paths.data_dir / DISTANCE_MATRIX_PATTERN.format(resolution=resolution),
            'binary_connectomes': {}
        }
        
        for density in self.data.densities:
            paths['binary_connectomes'][density] = (
                self.paths.data_dir / CONNECTOMES_BINARY_PATTERN.format(resolution=resolution, density=density)
            )
        
        return paths
    
    def generate_esn_hparam_grid(self,
                           spectral_radii: Optional[List[float]] = None,
                           input_lengths: Optional[List[int]] = None,
                           input_scalings: Optional[List[float]] = None,
                           regularization_methods: Optional[List[str]] = None,
                           n_runs_list: Optional[List[int]] = None,
                           densities: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Generate hyperparameter grid for ESN evaluation.
        
        Args:
            spectral_radii: List of spectral radius values
            input_lengths: List of input sequence lengths
            input_scalings: List of input scaling factors
            regularization_methods: List of regularization methods
            n_runs_list: List of number of runs per evaluation
            densities: List of connectivity densities
            
        Returns:
            List of hyperparameter dictionaries
        """
        from itertools import product
        
        # Use defaults if not provided
        spectral_radii = spectral_radii or [0.1, 0.5, 0.8, 0.99, 1.2, 1.5, 2.0]
        input_lengths = input_lengths or [500, 1000, 2000, 4000]
        input_scalings = input_scalings or [0.5, 1.0, 1.5, 2.0]
        regularization_methods = regularization_methods or ["pinv", "ridge"]
        n_runs_list = n_runs_list or [self.esn.n_runs]
        densities = densities or self.data.densities
        
        # Generate all combinations
        hparam_grid = []
        for spec_rad, input_len, input_scale, reg_method, n_runs, density in product(
            spectral_radii, input_lengths, input_scalings, 
            regularization_methods, n_runs_list, densities
        ):
            hparam_grid.append({
                "spectral_radius": spec_rad,
                "input_length": input_len,
                "input_scaling": input_scale,
                "regularization_method": reg_method,
                "n_runs": n_runs,
                "density_percent": density
            })
        
        return hparam_grid
        
        
 