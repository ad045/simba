import torch
import numpy as np
from typing import Dict
from scipy.linalg import expm
from scipy.spatial.distance import jensenshannon
from src.comparing_connectomes.base_comparer import NetworkEvaluator



class FrobeniusEvaluator(NetworkEvaluator):
    """Evaluate networks using Frobenius norm distance."""
    
    def __init__(self):
        pass
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute Frobenius norm between adjacency matrices."""
        results = {}
        gen_np = generated.squeeze().numpy()
        
        for target_idx in range(target.shape[0]):
            target_np = target[target_idx].numpy()
            
            # Frobenius norm: ||A - B||_F = sqrt(sum((A - B)^2))
            frob = np.linalg.norm(gen_np - target_np, ord='fro')
            results[target_idx] = frob
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "Frobenius"
