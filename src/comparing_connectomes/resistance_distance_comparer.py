import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class ResistanceDistanceEvaluator(NetworkEvaluator):
    """Evaluate networks using resistance perturbation distance.
    
    Uses effective graph resistance based on electrical network interpretation.
    """
    
    def __init__(self, approximate: bool = False, sample_ratio: float = 0.1):
        """
        Args:
            approximate: Use approximation for large graphs
            sample_ratio: Ratio of nodes to sample for approximation
        """
        self.approximate = approximate
        self.sample_ratio = sample_ratio
    
    def _compute_resistance_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute effective resistance matrix."""
        n = adj_matrix.shape[0]
        
        # Compute Laplacian
        D = np.diag(adj_matrix.sum(axis=1))
        L = D - adj_matrix
        
        if self.approximate:
            # Sample-based approximation
            n_samples = max(2, int(n * self.sample_ratio))
            samples = np.random.choice(n, n_samples, replace=False)
            
            # Compute resistance for sampled nodes only
            R_sample = np.zeros((n_samples, n_samples))
            L_pinv = np.linalg.pinv(L)
            
            for i, idx_i in enumerate(samples):
                for j, idx_j in enumerate(samples):
                    R_sample[i, j] = L_pinv[idx_i, idx_i] + L_pinv[idx_j, idx_j] - 2 * L_pinv[idx_i, idx_j]
            
            return R_sample
        else:
            # Exact computation via pseudoinverse
            L_pinv = np.linalg.pinv(L)
            R = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    R[i, j] = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
            
            return R
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute resistance distance for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        gen_np = generated[0].cpu().numpy()
        gen_resistance = self._compute_resistance_matrix(gen_np)
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            target_resistance = self._compute_resistance_matrix(target_np)
            
            # Frobenius distance between resistance matrices
            distance = np.linalg.norm(gen_resistance - target_resistance, ord='fro')
            results[target_idx] = float(distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        suffix = "_approx" if self.approximate else "_exact"
        return f"ResistanceDist{suffix}"
