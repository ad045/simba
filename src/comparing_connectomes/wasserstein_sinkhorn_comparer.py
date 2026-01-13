
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
    
    def __init__(self, reg: float = 0.1, max_iter: int = 100, tol: float = 1e-6):
        """
        Args:
            reg: Entropic regularization parameter
            max_iter: Maximum Sinkhorn iterations
        """
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol 
    
    def _sinkhorn(self, a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
        """Compute Wasserstein distance using Sinkhorn algorithm."""
        
        # Ensure inputs are valid
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        M = np.asarray(M, dtype=np.float64)
        
        # Normalize to ensure they're valid probability distributions
        epsilon = 1e-15
        a = np.maximum(a, epsilon)
        b = np.maximum(b, epsilon)
        a = a / a.sum()
        b = b / b.sum()
        
        # Log-domain stabilization
        K = np.exp(-M / self.reg)
        K = np.maximum(K, 1e-300)  # Prevent underflow
        
        # Initialize in log domain for stability
        u = np.ones_like(a)
        v = np.ones_like(b)
        # u = np.ones_like(a) / len(a)
        
        for iteration in range(self.max_iter):
            u_prev = u.copy()
            
            # Update with numerical safeguards
            Kv = K.T @ u
            Kv = np.maximum(Kv, 1e-300)
            v = b / Kv
            v = np.clip(v, 1e-300, 1e300)
            
            Ku = K @ v
            Ku = np.maximum(Ku, 1e-300)
            u = a / Ku
            u = np.clip(u, 1e-300, 1e300)
            
            # Check convergence
            if iteration > 0 and np.allclose(u, u_prev, rtol=self.tol):
                break
            
            # v = b / (K.T @ u + 1e-10)
            # u = a / (K @ v + 1e-10)
        
        # Compute transport plan and cost
        transport_plan = u[:, None] * K * v[None, :]
        transport_plan = np.nan_to_num(transport_plan, nan=0.0, posinf=0.0, neginf=0.0)
        cost = np.sum(transport_plan * M)
        
        # Return 0 if we got NaN or Inf
        if not np.isfinite(cost):
            return 0.0
        
        return float(cost)
    
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute Wasserstein distance between degree distributions."""
        results = {}
        n_targets = target.shape[0]
        
        gen_np = generated[0].cpu().numpy()
        gen_degrees = gen_np.sum(axis=1)
        
        # Check if degrees are all zero
        if gen_degrees.sum() < 1e-10:
            for target_idx in range(n_targets):
                results[target_idx] = 0.0
            return results
        
        gen_degrees = gen_degrees / (gen_degrees.sum() + 1e-10)
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            target_degrees = target_np.sum(axis=1)
            
            # Check if target degrees are all zero
            if target_degrees.sum() < 1e-10:
                results[target_idx] = 0.0
                continue
            
            target_degrees = target_degrees / (target_degrees.sum() + 1e-10)
            
            # Ensure same size by padding
            max_len = max(len(gen_degrees), len(target_degrees))
            a = np.pad(gen_degrees, (0, max_len - len(gen_degrees)))
            b = np.pad(target_degrees, (0, max_len - len(target_degrees)))
            # Renormalize after padding
            a = a / (a.sum() + 1e-10)
            b = b / (b.sum() + 1e-10)
            
            # Cost matrix (Euclidean distance between indices)
            indices = np.arange(max_len, dtype=np.float64)
            M = np.abs(indices[:, None] - indices[None, :])
    
            distance = self._sinkhorn(a, b, M)
            results[target_idx] = float(distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "Wasserstein"
