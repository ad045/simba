"""
Optimized GNM-based network generation fully leveraging Edward's GNM library.
"""

import torch
import warnings
from typing import Optional, Dict, List

# Import everything we need from GNM library
from gnm import (
    generative_rules, 
    evaluation, 
)

class GNMGenerator:
    """Optimized GNM generator using all available GNM library features."""
    
    # SOME (not all) available generative rules from GNM library -> removed all while cleaning up (Dec 11)
    AVAILABLE_RULES = {
        "matching_index": generative_rules.MatchingIndex,
        "neighbors": generative_rules.Neighbours,
        "degree_average": generative_rules.DegreeAverage,
        "degree_difference": generative_rules.DegreeDifference,
        "degree_product": generative_rules.DegreeProduct,
        "degree_max": generative_rules.DegreeMax,
        "degree_min": generative_rules.DegreeMin,
        "clustering_average": generative_rules.ClusteringAverage,
        "clustering_difference": generative_rules.ClusteringDifference,
        "clustering_product": generative_rules.ClusteringProduct, 
        "clustering_max": generative_rules.ClusteringMax,
        "clustering_min": generative_rules.ClusteringMin,
        
    }
    
    # SOME (not all) available evaluation metrics from GNM library -> removed all while cleaning up (Dec 11)
    AVAILABLE_METRICS = {
        "degree_ks": evaluation.DegreeKS,
        "edge_length_ks": evaluation.EdgeLengthKS,
        "clustering_ks": evaluation.ClusteringKS,
        "betweenness_ks": evaluation.BetweennessKS,
        "edge_length_ks": evaluation.EdgeLengthKS,
        "weighted_clustering_ks": evaluation.WeightedClusteringKS,
        "weighted_betweenness_correlation": evaluation.WeightedBetweennessKS,
    }
    
    def __init__(self, device: str):
        """Initialize with device selection."""
        self.device = device 
    
    def evaluate_network(self, 
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