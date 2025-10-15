"""
Optimized GNM-based network generation fully leveraging Edward's GNM library.
"""

import numpy as np
import torch
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path 

# Import everything we need from GNM library
from gnm import (
    fitting, 
    generative_rules, 
    evaluation, 
    weight_criteria
)

# For the dynamicGNMGenerator
# from ESNs.esn_evaluation_dynGNM import ESNEvaluator 

@dataclass
class GNMParameters: # TODO: Remove hardcoded stuff. 
    """Parameters for GNM generation - matches GNM library structure."""
    eta: Union[float, torch.Tensor] = -2.0                    # Distance parameter
    gamma: Union[float, torch.Tensor] = 0.3                   # Homophily parameter
    lambdah: Union[float, torch.Tensor] = 0.0                # Time-dependency parameter
    generative_rule: Any = None                               # Will be set to actual rule object
    distance_relationship_type: Union[str, List[str]] = "powerlaw"
    preferential_relationship_type: Union[str, List[str]] = "powerlaw"
    heterochronicity_relationship_type: Union[str, List[str]] = "powerlaw"
    num_simulations: int = 100
    device: str = "cpu"
    random_seed: Optional[int] = None


class GNMGenerator:
    """Optimized GNM generator using all available GNM library features."""
    
    # All available generative rules from GNM library
    AVAILABLE_RULES = {
        "matching_index": generative_rules.MatchingIndex,
        "neighbors": generative_rules.Neighbours,
        "degree_average": generative_rules.DegreeAverage,
        "degree_difference": generative_rules.DegreeDifference,
        "degree_product": generative_rules.DegreeProduct,
        "degree_max": generative_rules.DegreeMax,
        "degree_min": generative_rules.DegreeMin,
        # "clustering_coefficient": generative_rules. ClusteringClusteringCoefficient,
        # "clustering_coefficient_zhang": generative_rules.ClusteringCoefficientZhang,
        # "clustering_coefficient_bu": generative_rules.ClusteringCoefficientBu,
        # "spatial": generative_rules.Spatial,
        # "communicability": generative_rules.Communicability,
        # "adamic_adar": generative_rules.AdamicAdar,
        # "path_index_2": generative_rules.PathIndex2,
        # "preferential_attachment": generative_rules.PreferentialAttachment,
        # "resource_allocation": generative_rules.ResourceAllocation,
        "clustering_average": generative_rules.ClusteringAverage,
        "clustering_difference": generative_rules.ClusteringDifference,
        "clustering_product": generative_rules.ClusteringProduct, 
        "clustering_max": generative_rules.ClusteringMax,
        "clustering_min": generative_rules.ClusteringMin,
        
    }
    
    # All available evaluation metrics from GNM library
    AVAILABLE_METRICS = {
        "degree_ks": evaluation.DegreeKS,
        "edge_length_ks": evaluation.EdgeLengthKS,
        "clustering_ks": evaluation.ClusteringKS,
        "betweenness_ks": evaluation.BetweennessKS,
        "edge_length_ks": evaluation.EdgeLengthKS,
        "weighted_clustering_ks": evaluation.WeightedClusteringKS,
        # "degree_js": evaluation.DegreeJS,
        # "degree_correlatioin": evaluation.DegreeCorrelation,
        # "clustering_js": evaluation.ClusteringJS,
        # "max_criteria": evaluation.MaxCriteria,
        # "mean_criteria": evaluation.MeanCriteria,
        # "weighted_criteria": evaluation.WeightedCriteria,
        # "betweenness_correlation": evaluation.BetweennessCorrelation, 
        # "clustering_correlation": evaluation.ClusteringCorrelation,

        "weighted_betweenness_correlation": evaluation.WeightedBetweennessKS,
    }
    # Available weight optimization criteria - from init (unsure what to do with this yet) 
            #     "EvaluationCriterion",
            # "BinaryEvaluationCriterion",
            # "WeightedEvaluationCriterion",
            # "CorrelationCriterion",
            # "CompositeCriterion",
            # "WeightedSumCriteria",
            # "WeightedNodeStrengthKS",
    
    def __init__(self, device: str):
        """Initialize with device selection."""
        self.device = device # torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    
    def evaluate_network(self, # Is this here even used??? Yup. 
                        generated_network: torch.Tensor,
                        target_network: torch.Tensor,
                        distance_matrix: torch.Tensor,
                        metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate a generated network against a target using GNM metrics.
        
        Args:
            generated_network: Generated network
            target_network: Target network
            distance_matrix: Distance matrix
            metrics: List of metric names to compute
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ["degree_ks", "clustering_ks", "edge_length_ks", "frobenius"]
        
        results = {}
        
        for metric_name in metrics:
            if metric_name not in self.AVAILABLE_METRICS:
                warnings.warn(f"Unknown metric: {metric_name}")
                continue
            
            if metric_name == "edge_length_ks" or metric_name == "edge_length_js":
                metric = self.AVAILABLE_METRICS[metric_name](distance_matrix)
            else:
                metric = self.AVAILABLE_METRICS[metric_name]()
            
            value = metric(generated_network, target_network)
            results[metric_name] = float(value.item() if torch.is_tensor(value) else value)
        
        return results