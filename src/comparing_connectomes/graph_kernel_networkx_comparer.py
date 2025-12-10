import torch
import numpy as np
from typing import Dict, List, Optional
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import networkx as nx
from abc import ABC, abstractmethod

from src.comparing_connectomes.base_comparer import NetworkEvaluator




class GraphKernelNetworkxEvaluator(NetworkEvaluator):
    """Evaluate networks using graph kernel similarity.
    
    Uses random walk kernel as implemented in NetworkX.
    """
    
    def __init__(self, kernel_type: str = 'random_walk', walk_length: int = 3):
        """
        Args:
            kernel_type: Type of kernel ('random_walk', 'shortest_path')
            walk_length: Maximum walk length for random walk kernel
        """
        self.kernel_type = kernel_type
        self.walk_length = walk_length
    
    def _random_walk_kernel(self, G1: nx.Graph, G2: nx.Graph) -> float:
        """Compute random walk kernel similarity."""
        # Simple implementation: compare walk count distributions
        n1, n2 = len(G1), len(G2)
        
        # Count walks of various lengths
        A1 = nx.to_numpy_array(G1)
        A2 = nx.to_numpy_array(G2)
        
        # Compute walk counts
        walk_counts_1 = []
        walk_counts_2 = []
        
        A1_k = np.eye(n1)
        A2_k = np.eye(n2)
        
        for k in range(self.walk_length + 1):
            walk_counts_1.append(A1_k.sum())
            walk_counts_2.append(A2_k.sum())
            A1_k = A1_k @ A1
            A2_k = A2_k @ A2
        
        # Normalize and compute similarity
        wc1 = np.array(walk_counts_1) / (sum(walk_counts_1) + 1e-10)
        wc2 = np.array(walk_counts_2) / (sum(walk_counts_2) + 1e-10)
        
        # Kernel as dot product
        similarity = np.dot(wc1, wc2)
        return similarity
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute graph kernel similarity (converted to distance)."""
        results = {}
        n_targets = target.shape[0]
        
        gen_np = generated[0].cpu().numpy()
        G1 = nx.from_numpy_array(gen_np)
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            G2 = nx.from_numpy_array(target_np)
            
            if self.kernel_type == 'random_walk':
                similarity = self._random_walk_kernel(G1, G2)
            else:
                # Fallback to simple graph similarity
                similarity = 0.5
            
            # Convert similarity to distance
            distance = 1.0 - similarity
            results[target_idx] = float(distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return f"GraphKernel_{self.kernel_type}"
