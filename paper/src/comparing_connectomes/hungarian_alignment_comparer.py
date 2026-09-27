import torch
import numpy as np
from typing import Dict, List, Optional
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import networkx as nx
from abc import ABC, abstractmethod

from src.comparing_connectomes.base_comparer import NetworkEvaluator


class HungarianAlignmentEvaluator(NetworkEvaluator):
    """Find optimal node alignment using Hungarian algorithm.
    
    Solves linear assignment problem to find best node correspondence.
    """
    
    def __init__(self, cost_metric: str = 'euclidean'):
        """
        Args:
            cost_metric: Distance metric for computing cost matrix
        """
        self.cost_metric = cost_metric
    
    def _compute_node_features(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Extract node features for matching (e.g., degree, clustering)."""
        # Use degree and local clustering as features
        degrees = adj_matrix.sum(axis=1, keepdims=True)
        
        # Local clustering coefficient
        clustering = np.zeros((adj_matrix.shape[0], 1))
        for i in range(adj_matrix.shape[0]):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 1:
                subgraph = adj_matrix[np.ix_(neighbors, neighbors)]
                clustering[i] = subgraph.sum() / (len(neighbors) * (len(neighbors) - 1))
        
        return np.hstack([degrees, clustering])
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Find optimal alignment and compute aligned distance."""
        results = {}
        n_targets = target.shape[0]
        
        gen_np = generated[0].cpu().numpy()
        gen_features = self._compute_node_features(gen_np)
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            target_features = self._compute_node_features(target_np)
            
            # Compute cost matrix
            cost_matrix = cdist(gen_features, target_features, metric=self.cost_metric)
            
            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Compute total assignment cost
            total_cost = cost_matrix[row_ind, col_ind].sum()
            
            # Also compute post-alignment adjacency distance
            permuted_gen = gen_np[row_ind][:, col_ind]
            alignment_distance = np.linalg.norm(permuted_gen - target_np, ord='fro')
            
            results[target_idx] = float(alignment_distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "Hungarian"
