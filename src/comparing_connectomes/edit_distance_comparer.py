import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class EditDistanceEvaluator(NetworkEvaluator):
    """Evaluate networks using Frobenius/edit distance.
    
    Requires node correspondence (pre-aligned networks).
    """
    
    def __init__(self, normalize: bool = False):
        """
        Args:
            normalize: Whether to normalize by number of edges
        """
        self.normalize = normalize
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute edit distance for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        for target_idx in range(n_targets):
            # Frobenius norm: ||A1 - A2||_F
            diff = generated[0] - target[target_idx]
            distance = torch.norm(diff, p='fro').item()
            
            if self.normalize:
                n_edges = torch.sum(target[target_idx] > 0).item()
                distance = distance / (n_edges + 1e-10)
            
            results[target_idx] = float(distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "EditDistance"

