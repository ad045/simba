import numpy as np
import torch
from scipy.stats import entropy
from typing import Dict
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class PortraitDivergence(NetworkEvaluator):
    """Evaluate networks using portrait divergence."""
    
    def __init__(self, max_diameter: int = 500):
        self.max_diameter = max_diameter
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[int, float]:
        """Compute portrait divergence for single generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        gen_np = generated.cpu().numpy()
        if gen_np.ndim == 3:
            if gen_np.shape[0] != 1:
                raise ValueError("Generated tensor has multiple networks; expected a single network. (in portrait_divergence_comparer)")
            gen_np = gen_np[0,:,:]  # Remove batch dimension if present. TODO: BUT WHAT IF WE HAVE MULTIPLE NETWORKS IN THERE? 
        portrait_gen = self._compute_portrait(gen_np)
    
        for target_idx in range(n_targets):
            tgt_np = target[target_idx].cpu().numpy()
            portrait_tgt = self._compute_portrait(tgt_np)
            divergence = self._compute_divergence(portrait_gen, portrait_tgt)
            results[target_idx] = divergence
        
        return results
    
    
    @property
    def metric_prefix(self) -> str:
        return "PortraitDiv"
    
    
    def _compute_portrait(self, adj_matrix: np.ndarray) -> np.ndarray:
        n = len(adj_matrix)
        B = np.zeros((self.max_diameter + 1, n))
        
        # Build adjacency list with explicit int conversion
        adj_list = {}
        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            adj_list[i] = [int(nb) for nb in neighbors]
        
        max_path = 1
        
        for start in range(n):
            distances = {start: 0}
            queue = [start]
            d = 1
            
            while queue:
                next_level = []
                for node in queue:
                    for nb in adj_list[int(node)]:
                        if nb not in distances:
                            distances[nb] = d
                            next_level.append(nb)
                queue = next_level
                d += 1
            
            max_dist = max(distances.values())
            max_path = max(max_path, max_dist)
            
            dist_counts = {}
            for dist in distances.values():
                dist_counts[dist] = dist_counts.get(dist, 0) + 1
            
            for shell, count in dist_counts.items():
                B[shell][count] += 1
            
            for shell in range(max_dist + 1, self.max_diameter + 1):
                B[shell][0] += 1
        
        return B[:max_path + 1, :]
    
    
    def _compute_divergence(self, B1: np.ndarray, B2: np.ndarray) -> float:
        B1, B2 = self._pad_portraits(B1, B2)
        L, K = B1.shape
        V = np.tile(np.arange(K), (L, 1))
        
        X1, X2 = B1 * V, B2 * V
        X1_sum, X2_sum = X1.sum(), X2.sum()
        
        if X1_sum == 0 or X2_sum == 0:
            return 0.0
        
        P = (X1 / X1_sum).ravel()
        Q = (X2 / X2_sum).ravel()
        M = 0.5 * (P + Q)
        
        return 0.5 * (entropy(P, M, base=2) + entropy(Q, M, base=2))
    
    
    def _pad_portraits(self, B1: np.ndarray, B2: np.ndarray) -> tuple:
        last_col1 = max(np.nonzero(B1)[1]) if B1.any() else 0
        last_col2 = max(np.nonzero(B2)[1]) if B2.any() else 0
        last_col = max(last_col1, last_col2)
        
        B1, B2 = B1[:, :last_col + 1], B2[:, :last_col + 1]
        max_rows = max(B1.shape[0], B2.shape[0])
        
        BigB1 = np.zeros((max_rows, last_col + 1))
        BigB2 = np.zeros((max_rows, last_col + 1))
        BigB1[:B1.shape[0], :B1.shape[1]] = B1
        BigB2[:B2.shape[0], :B2.shape[1]] = B2
        
        return BigB1, BigB2