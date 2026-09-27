import torch
from typing import Dict, List
from src.comparing_connectomes.base_comparer import NetworkEvaluator
import numpy as np 

from src.analysis.kayson_utils import evaluate_adjacency

class F1DistEvaluator(NetworkEvaluator):
    """Evaluate networks using GNM energy-based criteria."""
    
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix.cpu().numpy()
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute energy for single generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        for target_idx in range(n_targets):
            target_single = target[target_idx:target_idx+1, :, :]
            
            # Apply distance weighting. LONG RANGE CONNECTIONS ARE WEIGHTED MORE. 
            target_single = target_single.numpy() * self.distance_matrix
            generated = generated.numpy() * self.distance_matrix

            _, results[target_idx] = evaluate_adjacency(generated, target_single) # accuracy, F1

        return results

    @property
    def metric_prefix(self) -> str:
        return "F1Crit"