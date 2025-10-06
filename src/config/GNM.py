from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch

# Import GNM configuration structures
from gnm import fitting, generative_rules, evaluation, weight_criteria

# Import project constants
from .constants import (
    DEFAULT_N_ETA, DEFAULT_N_GAMMA, DEFAULT_N_LAMBDA,
    DEFAULT_SPECTRAL_RADIUS, DEFAULT_INPUT_SCALING,
    DEFAULT_REGULARIZATION_METHOD, DEFAULT_GENERATIVE_RULES,
    DEFAULT_EVALUATION_METRICS, DEFAULT_WEIGHT_CRITERION, DEFAULT_ETA_RANGE, DEFAULT_GAMMA_RANGE,
    DEFAULT_LAMBDA_RANGE, DEFAULT_ALPHA, DEFAULT_NUM_SIMULATIONS
)



@dataclass
class GNMConfig:
    """GNM configuration using library structures."""
    # Use GNM's fitting structures directly
    binary_sweep_params: Optional[fitting.BinarySweepParameters] = None
    weighted_sweep_params: Optional[fitting.WeightedSweepParameters] = None
    
    # Default parameter ranges
    eta_range: tuple = DEFAULT_ETA_RANGE
    gamma_range: tuple = DEFAULT_GAMMA_RANGE
    lambda_range: tuple = DEFAULT_LAMBDA_RANGE
    n_eta: int = DEFAULT_N_ETA    
    n_gamma: int = DEFAULT_N_GAMMA
    n_lambda: int = DEFAULT_N_LAMBDA
    
    # Generative rules to test
    generative_rules_to_test: List[str] = field(default_factory=lambda: DEFAULT_GENERATIVE_RULES.copy())
    
    # Evaluation metrics
    evaluation_metrics: List[str] = field(default_factory=lambda: DEFAULT_EVALUATION_METRICS.copy())
    
    # Weight optimization
    weight_criterion: str = DEFAULT_WEIGHT_CRITERION
    alpha: float = DEFAULT_ALPHA
    
    # Simulation parameters
    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # For dynamic generation process (dynGNM) 
    use_dynamic_generation: bool = False
    dynamic_delta: float = 1.0  # The 'δ' parameter, balancing GNM vs. ESN
    candidate_pool_size: int = 50  # Number of top GNM candidates to evaluate with ESN
    dynamic_fitness_metric: str = "mc_mean"  # Metric to optimize during dynamic generation.  "mc_mean", "branching_ratio", "info_balance", ...
    
    # Fast ESN evaluation configuration for dynamic generation
    fast_esn_eval: Dict[str, Any] = field(default_factory=lambda: {
        "input_length": 500,
        "n_runs": 3,
        "spectral_radius": DEFAULT_SPECTRAL_RADIUS,
        "input_scaling": DEFAULT_INPUT_SCALING,
        "regularization_method": DEFAULT_REGULARIZATION_METHOD
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

