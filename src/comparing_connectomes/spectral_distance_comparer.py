
import torch
import numpy as np
from typing import Dict, List, Optional
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import networkx as nx
from abc import ABC, abstractmethod

from src.comparing_connectomes.base_comparer import NetworkEvaluator

class SpectralDistanceEvaluator(NetworkEvaluator):
    """Evaluate networks using spectral distance metrics.
    
    Supports adjacency, Laplacian, and normalized Laplacian spectral distances.
    """
    
    def __init__(self, k: int = 50, method: str = 'normalized_laplacian', p: int = 2):
        """
        Args:
            k: Number of eigenvalues to compute
            method: 'adjacency', 'laplacian', or 'normalized_laplacian'
            p: Lp norm to use (default: 2 for L2 distance)
        """
        self.k = k
        self.method = method
        self.p = p
    
        
    # def _compute_spectrum_networkx(self, adj_matrix: np.ndarray) -> np.ndarray:
    #     G = nx.from_numpy_array(adj_matrix)
    #     if self.method == 'normalized_laplacian':
    #     elif self.method == 'laplacian':
            
    
    def _compute_spectrum(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute eigenvalues based on selected method."""
        n = adj_matrix.shape[0]
        k_actual = min(self.k, n - 2)
        G = nx.from_numpy_array(adj_matrix)
        
        if self.method == 'adjacency':
            # Largest k eigenvalues of adjacency matrix
            eigenvalues, _ = eigsh(adj_matrix, k=k_actual, which='LM')
            return np.array(sorted(eigenvalues)[::-1])
        elif self.method == 'laplacian':
            # Smallest k eigenvalues of Laplacian L = D - A
            return np.array(sorted(nx.laplacian_spectrum(G))[:self.k])
            # D = np.diag(adj_matrix.sum(axis=1))
            # L = D - adj_matrix
            # eigenvalues, _ = eigsh(L, k=k_actual, which='SM')
        elif self.method == 'normalized_laplacian':
            # Eigenvalues of normalized Laplacian
            return np.array(sorted(nx.normalized_laplacian_spectrum(G))[:self.k])
        
            # D = np.diag(adj_matrix.sum(axis=1))
            # D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
            # L_norm = np.eye(n) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
            # eigenvalues, _ = eigsh(L_norm, k=k_actual, which='SM')
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # return np.sort(eigenvalues)
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute spectral distance for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        # Convert generated to numpy
        gen_np = generated[0].cpu().numpy()
        gen_spectrum = self._compute_spectrum(gen_np)
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            target_spectrum = self._compute_spectrum(target_np)
            
            # Compute Lp distance between spectra
            min_len = min(len(gen_spectrum), len(target_spectrum))
            distance = np.linalg.norm(
                gen_spectrum[:min_len] - target_spectrum[:min_len], 
                ord=self.p
            )
            results[target_idx] = float(distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return f"SpectralDist_{self.method}"

