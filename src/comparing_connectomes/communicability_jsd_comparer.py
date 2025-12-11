import torch
import numpy as np
from typing import Dict
from scipy.linalg import expm
from scipy.spatial.distance import jensenshannon
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class CommunicabilityJSDEvaluator(NetworkEvaluator):
    """Evaluate networks using Jensen-Shannon Divergence of communicability matrices."""
    
    def __init__(self):
        pass
    
    def _compute_communicability(self, adj: np.ndarray) -> np.ndarray:
        """Compute communicability matrix: expm(A)"""
        return expm(adj)
    
    def _compute_jsd(self, P: np.ndarray, Q: np.ndarray) -> float:
        """Compute JSD between two probability distributions."""
        # Flatten and normalize to probability distributions
        P_flat = P.flatten()
        Q_flat = Q.flatten()
        
        # Normalize to sum to 1
        P_norm = P_flat / (P_flat.sum() + 1e-10)
        Q_norm = Q_flat / (Q_flat.sum() + 1e-10)
        
        return jensenshannon(P_norm, Q_norm, base=2.0) ** 2
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute JSD of communicability matrices."""
        results = {}
        gen_np = generated.squeeze().numpy()
        
        gen_comm = self._compute_communicability(gen_np)
        
        for target_idx in range(target.shape[0]):
            target_np = target[target_idx].numpy()
            target_comm = self._compute_communicability(target_np)
            
            jsd = self._compute_jsd(gen_comm, target_comm)
            results[target_idx] = jsd
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "CommJSD"
