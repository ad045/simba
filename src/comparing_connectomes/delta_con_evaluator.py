
import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class DeltaConEvaluator(NetworkEvaluator):
    """Evaluate networks using DeltaCon metric.
    
    Uses fast belief propagation to capture local and global changes.
    """
    
    def __init__(self, approximate: bool = False, epsilon: float = 0.01, max_iter: int = 10):
        """
        Args:
            approximate: Use approximate variant
            epsilon: Convergence parameter for belief propagation
            max_iter: Maximum iterations for approximate method
        """
        self.approximate = approximate
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def _compute_belief_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute belief propagation matrix S = Σ ε^k A^k."""
        n = adj_matrix.shape[0]
        
        if self.approximate:
            # Iterative approximation
            S = np.eye(n)
            A_k = np.eye(n)
            
            for k in range(1, self.max_iter + 1):
                A_k = A_k @ adj_matrix
                S += (self.epsilon ** k) * A_k
        else:
            # Exact: S = (I - εA)^(-1)
            I = np.eye(n)
            S = np.linalg.inv(I - self.epsilon * adj_matrix)
        
        return S
    
    def _matusita_distance(self, S1: np.ndarray, S2: np.ndarray) -> float:
        """Compute Matusita distance between belief matrices."""
        # Normalize to make probability distributions
        S1_norm = S1 / (S1.sum() + 1e-10)
        S2_norm = S2 / (S2.sum() + 1e-10)
        
        # Matusita distance
        distance = np.sqrt(np.sum((np.sqrt(S1_norm) - np.sqrt(S2_norm)) ** 2))
        return distance
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute DeltaCon distance for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        gen_np = generated[0].cpu().numpy()
        gen_belief = self._compute_belief_matrix(gen_np)
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            target_belief = self._compute_belief_matrix(target_np)
            
            distance = self._matusita_distance(gen_belief, target_belief)
            results[target_idx] = float(distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        suffix = "_approx" if self.approximate else "_exact"
        return f"DeltaCon{suffix}"

