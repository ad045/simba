
import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator



class NetworkMutualInformationEvaluator(NetworkEvaluator):
    """Evaluate networks using Network Mutual Information.
    
    Measures information-theoretic similarity between network structures.
    """
    
    def __init__(self, variant: str = 'standard'):
        """
        Args:
            variant: 'standard', 'degree_corrected', or 'mesoscale'
        """
        self.variant = variant
    
    def _compute_edge_nmi(self, adj1: np.ndarray, adj2: np.ndarray) -> float:
        """Compute standard NMI based on edge overlap."""
        # Flatten adjacency matrices to edge lists
        edges1 = (adj1 > 0).astype(int).flatten()
        edges2 = (adj2 > 0).astype(int).flatten()
        
        # Compute mutual information
        # P(X=1, Y=1), P(X=1, Y=0), etc.
        n = len(edges1)
        p11 = np.sum((edges1 == 1) & (edges2 == 1)) / n
        p10 = np.sum((edges1 == 1) & (edges2 == 0)) / n
        p01 = np.sum((edges1 == 0) & (edges2 == 1)) / n
        p00 = np.sum((edges1 == 0) & (edges2 == 0)) / n
        
        p1_ = p11 + p10
        p0_ = p01 + p00
        p_1 = p11 + p01
        p_0 = p10 + p00
        
        # Mutual information
        mi = 0
        for px, py, pxy in [(p1_, p_1, p11), (p1_, p_0, p10), 
                            (p0_, p_1, p01), (p0_, p_0, p00)]:
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log2(pxy / (px * py))
        
        # Normalized MI
        h1 = -sum([p * np.log2(p) for p in [p1_, p0_] if p > 0])
        h2 = -sum([p * np.log2(p) for p in [p_1, p_0] if p > 0])
        
        if h1 + h2 > 0:
            nmi = 2 * mi / (h1 + h2)
        else:
            nmi = 0.0
        
        return nmi
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute NMI for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        gen_np = generated[0].cpu().numpy()
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            
            # Ensure same size
            min_size = min(gen_np.shape[0], target_np.shape[0])
            gen_crop = gen_np[:min_size, :min_size]
            target_crop = target_np[:min_size, :min_size]
            
            if self.variant == 'standard':
                nmi = self._compute_edge_nmi(gen_crop, target_crop)
            else:
                # Simplified version for other variants
                nmi = self._compute_edge_nmi(gen_crop, target_crop)
            
            # Convert similarity to distance
            distance = 1.0 - nmi
            results[target_idx] = float(distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return f"NMI_{self.variant}"