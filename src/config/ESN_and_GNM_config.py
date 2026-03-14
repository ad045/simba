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
    spectral_radius: float
    input_length: int
    input_scaling: float
    regularization_method: str
    n_runs: int
    n_lags: int
    test_len: int
    n_transient: int
    leak_rate: float
    bias: float


@dataclass
# class GNMConfig: # I think this is not used... DEFAULT_ALPHA is no-where defined (anymore...) 
#     """GNM configuration using library structures."""
#     # Use GNM's fitting structures directly
#     binary_sweep_params: Optional[fitting.BinarySweepParameters] = None
#     weighted_sweep_params: Optional[fitting.WeightedSweepParameters] = None

#     # Generative rules to test
#     generative_rules_to_test: List[str] = field(default_factory=lambda: DEFAULT_GENERATIVE_RULES.copy())
    
#     # Evaluation metrics
#     evaluation_metrics: List[str] = field(default_factory=lambda: DEFAULT_EVALUATION_METRICS.copy())
    
#     # Weight optimization
#     weight_criterion: str = DEFAULT_WEIGHT_CRITERION
#     alpha: float = DEFAULT_ALPHA
    
#     # Simulation parameters
#     num_simulations: int = DEFAULT_NUM_SIMULATIONS
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # For dynamic generation process (dynGNM) 
#     use_dynamic_generation: bool = False
#     dynamic_delta: float = 1.0  # The 'δ' parameter, balancing GNM vs. ESN
#     candidate_pool_size: int = 50  # Number of top GNM candidates to evaluate with ESN
#     dynamic_fitness_metric: str = "mc_mean"  # Metric to optimize during dynamic generation.  "mc_mean", "branching_ratio", "info_balance", ...
    
#     # Fast ESN evaluation configuration for dynamic generation
#     fast_esn_eval: Dict[str, Any] = field(default_factory=lambda: {
#         "input_length": 500,
#         "n_runs": 3,
#         "spectral_radius": DEFAULT_SPECTRAL_RADIUS,
#         "input_scaling": DEFAULT_INPUT_SCALING,
#         "regularization_method": DEFAULT_REGULARIZATION_METHOD
#     })
    
#     def create_binary_sweep_parameters(self, 
#                                       distance_matrix: torch.Tensor,
#                                       num_iterations: int) -> fitting.BinarySweepParameters:
#         """Create GNM BinarySweepParameters from config."""
#         # Get generative rules
#         rules = []
#         for rule_name in self.generative_rules_to_test:
#             if rule_name == "matching_index":
#                 rules.append(generative_rules.MatchingIndex())
#             elif rule_name == "neighbors":
#                 rules.append(generative_rules.Neighbors())
#             elif rule_name == "degree_product":
#                 rules.append(generative_rules.DegreeProduct())
#             elif rule_name == "clustering_coefficient":
#                 rules.append(generative_rules.ClusteringCoefficient())
#             elif rule_name == "spatial":
#                 rules.append(generative_rules.Spatial())
#             else:
#                 # Default to matching index
#                 rules.append(generative_rules.MatchingIndex())
        
#         return fitting.BinarySweepParameters(
#             eta=torch.linspace(self.eta_range[0], self.eta_range[1], self.n_eta),
#             gamma=torch.linspace(self.gamma_range[0], self.gamma_range[1], self.n_gamma),
#             lambdah=torch.linspace(self.lambda_range[0], self.lambda_range[1], self.n_lambda),
#             distance_relationship_type="tbc", # "powerlaw", # TODO: change all these defaults! I hope I am not using them??
#             preferential_relationship_type="tbc", # powerlaw",
#             heterochronicity_relationship_type="tbc", # powerlaw",
#             generative_rule=rules,
#             num_iterations=[num_iterations],
#         )
    
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
    resolution: int
    densities: List[int]

@dataclass
class ComputeConfig:
    """Computational settings."""
    n_workers: int
    timing_flag: bool
    append_interval: int
    random_seed: int


@dataclass
class PathConfig:
    """Path configuration."""
    root_dir: Path = field(default_factory=lambda: Path.cwd().resolve()) # is a ok default, I think... but better would be to replace it! 
    
    def __post_init__(self):
        # Find project root by looking for markers
        current = self.root_dir
        for parent in (current, *current.parents):
            if any((parent / marker).exists() for marker in ("pyproject.toml", "setup.cfg", ".git")):
                self.root_dir = parent
                break
        
        # self.data_dir = self.root_dir / "14_4D_lab_code" / "data/preprocessed/01_first_analysises" # TODO TODO !!! 
        self.data_dir = self.root_dir / "data/preprocessed/01_first_analysises"
        self.output_dir = self.root_dir / "output"
        self.esn_output_dir = self.output_dir / "esn"
        self.gnm_output_dir = self.output_dir / "gnm"
        self.dynamic_gnm_output_dir = self.output_dir / "dynamic_gnm"


class ConfigManager: # Is it used? Unsure. 
    """Optimized configuration manager using GNM structures."""
    
    def __init__(self):
        
        self.esn = ESNConfig()
        self.gnm = GNMConfig()
        
        self.data = DataConfig(
            resolution=self.data.resolution, 
            densities=self.data.densities, 
        )
        
        self.compute = ComputeConfig(n_workers=self.compute.n_workers, 
                                     timing_flag=self.compute.timing_flag, 
                                     random_seed=self.compute.random_seed,
                                    )
        self.paths = PathConfig()


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
        


