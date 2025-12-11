import torch
import numpy as np
from typing import Dict
from scipy.linalg import expm
from scipy.spatial.distance import jensenshannon
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class CommunicabilityMSEEvaluator(NetworkEvaluator):
    """Evaluate networks using MSE of communicability matrices."""
    
    def __init__(self):
        pass
    
    def _compute_communicability(self, adj: np.ndarray) -> np.ndarray:
        """Compute communicability matrix: expm(A)"""
        return expm(adj)
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute MSE of communicability matrices."""
        results = {}
        gen_np = generated.squeeze().numpy()
        gen_comm = self._compute_communicability(gen_np)
        
        for target_idx in range(target.shape[0]):
            target_np = target[target_idx].numpy()
            target_comm = self._compute_communicability(target_np)
            
            mse = np.mean((gen_comm - target_comm) ** 2)
            results[target_idx] = mse
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "CommMSE"
