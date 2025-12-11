from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch

# Import GNM configuration structures
from gnm import fitting, generative_rules, evaluation, weight_criteria

# Check what of this is used... 

@dataclass
class GNMConfig:
    """GNM configuration using library structures."""

    # Default parameter ranges
    eta_range: tuple 
    gamma_range: tuple 
    lambda_range: tuple 
    n_eta: int 
    n_gamma: int 
    n_lambda: int 

    # Generative rules to test
    generative_rules_to_test: List[str] 

    # Evaluation metrics
    evaluation_metrics: List[str] 

    # Weight optimization
    weight_criterion: str 
    alpha: float 

    # Simulation parameters
    num_simulations: int
    device: str

    # For dynamic generation process (dynGNM)
    use_dynamic_generation: bool
    dynamic_delta: float
    candidate_pool_size: int
    dynamic_fitness_metric: str

    # Fast ESN evaluation configuration for dynamic generation
    fast_esn_eval: Dict[str, Any]
    
    # Use GNM's fitting structures directly # TODO: Remove defaults. ( add 'defalt' option in config file)
    binary_sweep_params: Optional[fitting.BinarySweepParameters] = None
    weighted_sweep_params: Optional[fitting.WeightedSweepParameters] = None
    
    
    
    def create_binary_sweep_parameters(self, 
                                      distance_matrix: torch.Tensor,
                                      num_iterations: int) -> fitting.BinarySweepParameters:
        """Create GNM BinarySweepParameters from config."""
        # Get generative rules
        rules = []
    #     for rule_name in self.generative_rules_to_test:
    #         if rule_name == "matching_index":
    #             rules.append(generative_rules.MatchingIndex())
    #         elif rule_name == "neighbors":
    #             rules.append(generative_rules.Neighbors())
    #         elif rule_name == "degree_product":
    #             rules.append(generative_rules.DegreeProduct())
    #         elif rule_name == "clustering_coefficient":
    #             rules.append(generative_rules.ClusteringCoefficient())
    #         elif rule_name == "spatial":
    #             rules.append(generative_rules.Spatial())
    #         else:
    #             # Default to matching index
    #             rules.append(generative_rules.MatchingIndex())
        
    #     return fitting.BinarySweepParameters(
    #         eta=torch.linspace(self.eta_range[0], self.eta_range[1], self.n_eta),
    #         gamma=torch.linspace(self.gamma_range[0], self.gamma_range[1], self.n_gamma),
    #         lambdah=torch.linspace(self.lambda_range[0], self.lambda_range[1], self.n_lambda),
    #         distance_relationship_type="powerlaw", # TODO: Change all these defaults! 
    #         preferential_relationship_type="powerlaw",
    #         heterochronicity_relationship_type="powerlaw",
    #         generative_rule=rules,
    #         num_iterations=[num_iterations],
    #     )
    
        
def create_weighted_sweep_parameters(config, 
                                     distance_matrix: torch.Tensor) -> fitting.WeightedSweepParameters:
        """Create GNM WeightedSweepParameters from config."""
        # Get weight criterion
        if config['gnm']['weight_criterion'] == "distance_weighted_communicability":
            criterion = weight_criteria.DistanceWeightedCommunicability(distance_matrix)
        elif config['gnm']['weight_criterion'] == "communicability":
            criterion = weight_criteria.Communicability()
        elif config['gnm']['weight_criterion'] == "flow":
            criterion = weight_criteria.Flow()
        else:
            # Default
            criterion = weight_criteria.DistanceWeightedCommunicability(distance_matrix)
        
        return fitting.WeightedSweepParameters(
            alpha=[config['gnm']['alpha']],
            optimisation_criterion=[criterion]
        )


def create_evaluation_criteria(config, 
                               distance_matrix: torch.Tensor) -> Any:
    """Create evaluation criteria from config."""
    criteria = []
    
    for metric in config['gnm']['evaluation_metrics']:
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