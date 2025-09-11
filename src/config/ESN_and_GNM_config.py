"""
Configuration management using GNM library structures.
"""
# -> Used at least for main_pipeline_2_gnm_esn_landscape.py and for main_pipeline_2.py (esn part)

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch

# Import GNM configuration structures
from gnm import fitting, generative_rules, evaluation, weight_criteria


@dataclass
class ESNConfig:
    """ESN hyperparameter configuration."""
    spectral_radius: float = 0.99
    input_length: int = 4000
    input_scaling: float = 1.0
    regularization_method: str = "pinv"
    n_runs: int = 10
    n_lags: int = 50
    test_len: int = 1000
    n_transient: int = 0
    leak_rate: float = 1.0
    bias: float = 1.0


@dataclass
class GNMConfig:
    """GNM configuration using library structures."""
    # Use GNM's fitting structures directly
    binary_sweep_params: Optional[fitting.BinarySweepParameters] = None
    weighted_sweep_params: Optional[fitting.WeightedSweepParameters] = None
    
    # Default parameter ranges
    eta_range: tuple = (-3.0, 0) # (-5.0, 0.0)
    gamma_range: tuple = (0.001, 0.6) # , 1.0)
    lambda_range: tuple = (0.0, 0.0)
    n_eta: int = 200    
    n_gamma: int = 200
    n_lambda: int = 1
    
    # Generative rules to test
    generative_rules_to_test: List[str] = field(default_factory=lambda: ["matching_index"])
    
    # Evaluation metrics
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "degree_ks", "clustering_ks", "edge_length_ks"
    ])
    
    # Weight optimization
    weight_criterion: str = "distance_weighted_communicability"
    alpha: float = 0.01
    
    # Simulation parameters
    num_simulations: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # For dynamic generation process (dynGNM) 
    use_dynamic_generation: bool = False
    dynamic_delta: float = 1.0  # The 'δ' parameter, balancing GNM vs. ESN
    candidate_pool_size: int = 50  # Number of top GNM candidates to evaluate with ESN
    dynamic_fitness_metric: str = "mc_mean"  # Metric to optimize during dynamic generation.  "mc_mean", "branching_ratio", "info_balance", ...
    
    # Configuration for the fast ESN evaluation used during generation od dynGNM
    fast_esn_eval: Dict[str, Any] = field(default_factory=lambda: {
        "input_length": 500,
        "n_runs": 3,
        "spectral_radius": 0.99,
        "input_scaling": 1.0,
        "regularization_method": "pinv"
    })
    
    def create_binary_sweep_parameters(self, 
                                      distance_matrix: torch.Tensor,
                                      num_iterations: int) -> fitting.BinarySweepParameters:
        """Create GNM BinarySweepParameters from config."""
        # Get generative rules
        rules = []
        for rule_name in self.generative_rules_to_test:
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
                # Default to matching index
                rules.append(generative_rules.MatchingIndex())
        
        return fitting.BinarySweepParameters(
            eta=torch.linspace(self.eta_range[0], self.eta_range[1], self.n_eta),
            gamma=torch.linspace(self.gamma_range[0], self.gamma_range[1], self.n_gamma),
            lambdah=torch.linspace(self.lambda_range[0], self.lambda_range[1], self.n_lambda),
            distance_relationship_type=["powerlaw"],
            preferential_relationship_type=["powerlaw"],
            heterochronicity_relationship_type=["powerlaw"],
            generative_rule=rules,
            num_iterations=[num_iterations],
        )
    
    def create_weighted_sweep_parameters(self, 
                                        distance_matrix: torch.Tensor) -> fitting.WeightedSweepParameters:
        """Create GNM WeightedSweepParameters from config."""
        # Get weight criterion
        if self.weight_criterion == "distance_weighted_communicability":
            criterion = weight_criteria.DistanceWeightedCommunicability(distance_matrix)
        elif self.weight_criterion == "communicability":
            criterion = weight_criteria.Communicability()
        elif self.weight_criterion == "flow":
            criterion = weight_criteria.Flow()
        else:
            # Default
            criterion = weight_criteria.DistanceWeightedCommunicability(distance_matrix)
        
        return fitting.WeightedSweepParameters(
            alpha=[self.alpha],
            optimisation_criterion=[criterion]
        )
    
    def create_evaluation_criteria(self, distance_matrix: torch.Tensor) -> Any:
        """Create evaluation criteria from config."""
        criteria = []
        
        for metric in self.evaluation_metrics:
            if metric == "degree_ks":
                criteria.append(evaluation.DegreeKS())
            elif metric == "clustering_ks":
                criteria.append(evaluation.ClusteringKS())
            elif metric == "betweenness_ks":
                criteria.append(evaluation.BetweennessKS())
            elif metric == "edge_length_ks":
                criteria.append(evaluation.EdgeLengthKS(distance_matrix))
            elif metric == "degree_js":
                criteria.append(evaluation.DegreeJS())
            elif metric == "clustering_js":
                criteria.append(evaluation.ClusteringJS())
            elif metric == "betweenness_js":
                criteria.append(evaluation.BetweennessJS())
            elif metric == "edge_length_js":
                criteria.append(evaluation.EdgeLengthJS(distance_matrix))
            elif metric == "frobenius":
                criteria.append(evaluation.Frobenius())
        
        # Combine criteria
        if len(criteria) > 1:
            return evaluation.MaxCriteria(criteria)
        elif len(criteria) == 1:
            return criteria[0]
        else:
            # Default
            return evaluation.MaxCriteria([
                evaluation.DegreeKS(),
                evaluation.ClusteringKS(),
                evaluation.EdgeLengthKS(distance_matrix)
            ])


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    resolution: int = 68
    densities: List[int] = field(default_factory=lambda: [10, 12, 14, 16, 18, 20])
    use_weighted: bool = True
    use_gnm_defaults: bool = False  # Option to use GNM's default data


@dataclass
class ComputeConfig:
    """Computational settings."""
    n_workers: Optional[int] = None  
    timing_flag: bool = True
    append_interval: int = 10
    random_seed: int = 42


@dataclass
class PathConfig:
    """Path configuration."""
    root_dir: Path = Path("/Users/adrian/Documents/01_projects/14_4D_lab")
    
    def __post_init__(self):
        self.data_dir = self.root_dir / "data/preprocessed/01_first_analysises"
        self.output_dir = self.root_dir / "output"
        self.esn_output_dir = self.output_dir / "esn"  # TODO: ESN path. previously: 02_esns_on_observed_weighted_connectomes
        self.gnm_output_dir = self.output_dir / "gnm" # TODO: GNM path. previously: 03_gnm_estimation
        self.dynamic_gnm_output_dir = self.output_dir / "dynamic_gnm" # dynGNM path


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
    
    # Add this method to the ConfigManager class
    def create_gnm_random_sweep_config(self,
                                    distance_matrix: torch.Tensor,
                                    num_iterations: int,
                                    num_simulations: int = 100,
                                    method: Optional[str] = "grid", # "random", # "grid",
                                    n_random_samples: int = 30,
                                    include_weights: bool = True) -> fitting.SweepConfig:
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
        
        device = torch.device(self.gnm.device)
        
        # Load default distance matrix and network from GNM
        distance_matrix = defaults.get_distance_matrix(device=device)
        binary_network = defaults.get_binary_network(device=device)
        
        return {
            "distance_matrix": distance_matrix,
            "binary_network": binary_network
        }
    
    def get_data_paths(self):
        """Get paths to data files."""
        if self.data.use_gnm_defaults:
            # Use GNM's default data
            return {"use_gnm_defaults": True}
        
        resolution = self.data.resolution
        
        paths = {
            'weighted_connectome': self.paths.data_dir / f"connectomes_weighted_{resolution}x{resolution}.npy",
            'distance_matrix': self.paths.data_dir / f"distance_matrix_{resolution}x{resolution}.npy",
            'binary_connectomes': {}
        }
        
        for density in self.data.densities:
            paths['binary_connectomes'][density] = (
                self.paths.data_dir / f"connectomes_binarized_{resolution}x{resolution}_density_{density}_percent.npy"
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
        
        
def get_esn_config():
    """Get configuration specifically for ESN analysis."""
    config = ConfigManager()
    # Ensure we use actual data paths, not GNM defaults
    config.data.use_gnm_defaults = False
    return config


# # Preset configurations optimized for GNM library
# def get_gnm_quick_test_config() -> ConfigManager:
#     """Quick test configuration using GNM defaults."""
#     gnm_config = GNMConfig(
#         n_eta=50, # TODO: I seem to use this one here for the sweep... WHY THO? 
#         n_gamma=50,
#         num_simulations=10,
#         generative_rules_to_test=["matching_index"],
#         evaluation_metrics=["degree_ks", "clustering_ks"]
#     )
    
#     data_config = DataConfig(
#         densities=[10],
#         use_gnm_defaults=True  # Use GNM's default data for testing
#     )
    
#     compute_config = ComputeConfig(
#         timing_flag=True,
#         n_workers=2
#     )
    
#     return ConfigManager(
#         gnm_config=gnm_config,
#         data_config=data_config,
#         compute_config=compute_config
#     )

# # Preset MINIMAL configurations optimized for GNM library
# def get_gnm_minimal_config() -> ConfigManager:
#     """Minimal configuration using GNM defaults."""
#     gnm_config = GNMConfig(
#         n_eta=2,
#         n_gamma=2,
#         num_simulations=1,
#         generative_rules_to_test=["matching_index"],
#         evaluation_metrics=["degree_ks", "clustering_ks"]
#     )
    
#     data_config = DataConfig(
#         densities=[10],
#         use_gnm_defaults=True  # Use GNM's default data for testing
#     )
    
#     compute_config = ComputeConfig(
#         timing_flag=True,
#         n_workers=2
#     )
    
#     return ConfigManager(
#         gnm_config=gnm_config,
#         data_config=data_config,
#         compute_config=compute_config
#     )


# def get_gnm_comprehensive_config() -> ConfigManager:
#     """Comprehensive GNM analysis configuration."""
#     gnm_config = GNMConfig(
#         n_eta=50,
#         n_gamma=50,
#         num_simulations=100,
#         generative_rules_to_test=[
#             "matching_index", 
#             "neighbors",
#             "degree_product",
#             "clustering_coefficient",
#             "spatial"
#         ],
#         evaluation_metrics=[
#             "degree_ks", 
#             "clustering_ks", 
#             "betweenness_ks",
#             "edge_length_ks"
#         ],
#         weight_criterion="distance_weighted_communicability",
#         alpha=0.01
#     )
    
#     data_config = DataConfig(
#         densities=[10, 12, 14, 16, 18, 20],
#         use_weighted=True
#     )
    
#     return ConfigManager(gnm_config=gnm_config, data_config=data_config)


# def get_gnm_rule_comparison_config() -> ConfigManager:
#     """Configuration for comparing different generative rules."""
#     gnm_config = GNMConfig(
#         n_eta=20,
#         n_gamma=20,
#         num_simulations=50,
#         generative_rules_to_test=[
#             "matching_index",
#             "neighbors", 
#             "degree_product",
#             "degree_difference",
#             "clustering_coefficient",
#             "clustering_coefficient_zhang",
#             "clustering_coefficient_bu",
#             "spatial",
#             "communicability",
#             "adamic_adar",
#             "path_index_2",
#             "preferential_attachment",
#             "resource_allocation"
#         ],
#         evaluation_metrics=["degree_ks", "clustering_ks", "edge_length_ks", "frobenius"]
#     )
    
    # return ConfigManager(gnm_config=gnm_config)