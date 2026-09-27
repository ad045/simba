import torch
from typing import Dict, List
from src.comparing_connectomes.base_comparer import NetworkEvaluator
import numpy as np 


class HammingEvaluator(NetworkEvaluator):
    """
    Evaluate networks using Hamming distance. 
    Some notes: 
    - Assumes that the input adjacency matrices are of the same size.
    - Assumes that diagnoals in matrices are zero (no self-loops - they are otherwise counted twice). 
    - Binarizes the adjacency matrices before comparison.
    - No normalization is applied; returns raw Hamming distance (number of differing edges).
    """
    
    def __init__(self):
        pass
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute Hamming distance for single generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        # Convert generated to numpy
        gen_np = generated[0].cpu().numpy()
        
        for target_idx in range(n_targets):
            target_single = target[target_idx:target_idx+1, :, :].cpu().numpy()
            
            # Binarize for structural comparison
            gen_binary = (gen_np > 0).astype(int)
            target_binary = (target_single > 0).astype(int)

            results[target_idx] = np.sum(gen_binary != target_binary) / 2 # Each differing edge counted twice in adjacency matrix

        return results

    @property
    def metric_prefix(self) -> str:
        return "HammingDist"
