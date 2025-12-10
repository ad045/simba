
import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator



class WassersteinSinkhornEvaluator(NetworkEvaluator):
    """Evaluate networks using Wasserstein distance with Sinkhorn approximation.
    
    Compares degree distributions using optimal transport.
    """
    
    def __init__(self, reg: float = 0.1, max_iter: int = 100):
        """
        Args:
            reg: Entropic regularization parameter
            max_iter: Maximum Sinkhorn iterations
        """
        self.reg = reg
        self.max_iter = max_iter
    
    def _sinkhorn(self, a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
        """Compute Wasserstein distance using Sinkhorn algorithm."""
        K = np.exp(-M / self.reg)
        u = np.ones_like(a) / len(a)
        
        for _ in range(self.max_iter):
            v = b / (K.T @ u + 1e-10)
            u = a / (K @ v + 1e-10)
        
        # Compute transport cost
        transport_plan = u[:, None] * K * v[None, :]
        cost = np.sum(transport_plan * M)
        
        return cost
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute Wasserstein distance between degree distributions."""
        results = {}
        n_targets = target.shape[0]
        
        gen_np = generated[0].cpu().numpy()
        gen_degrees = gen_np.sum(axis=1)
        gen_degrees = gen_degrees / (gen_degrees.sum() + 1e-10)
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            target_degrees = target_np.sum(axis=1)
            target_degrees = target_degrees / (target_degrees.sum() + 1e-10)
            
            # Ensure same size by padding
            max_len = max(len(gen_degrees), len(target_degrees))
            a = np.pad(gen_degrees, (0, max_len - len(gen_degrees)))
            b = np.pad(target_degrees, (0, max_len - len(target_degrees)))
            a = a / (a.sum() + 1e-10)
            b = b / (b.sum() + 1e-10)
            
            # Cost matrix (Euclidean distance between indices)
            indices = np.arange(max_len)
            M = np.abs(indices[:, None] - indices[None, :])
            
            distance = self._sinkhorn(a, b, M)
            results[target_idx] = float(distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "Wasserstein"
