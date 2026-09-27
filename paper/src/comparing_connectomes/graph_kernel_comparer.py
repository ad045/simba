
import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator

class GraphKernelEvaluator(NetworkEvaluator):
    """Evaluate networks using advanced graph kernels."""
    
    def __init__(self, kernel_type: str = "weisfeiler_lehman", n_iterations: int = 5):
        self.kernel_type = kernel_type
        self.n_iterations = n_iterations
    
    def _weisfeiler_lehman_kernel(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute Weisfeiler-Lehman graph kernel."""
        n1, n2 = A1.shape[0], A2.shape[0]
        
        # Initialize node labels
        labels1 = torch.zeros(n1, dtype=torch.long)
        labels2 = torch.zeros(n2, dtype=torch.long)
        
        similarity = 0.0
        
        for _ in range(self.n_iterations):
            # Compute label histograms
            hist1 = torch.bincount(labels1, minlength=max(labels1.max(), labels2.max()) + 1)
            hist2 = torch.bincount(labels2, minlength=max(labels1.max(), labels2.max()) + 1)
            
            # Add kernel value (histogram intersection)
            similarity += torch.min(hist1.float(), hist2.float()).sum().item()
            
            # Update labels based on neighborhood
            new_labels1 = labels1.clone()
            new_labels2 = labels2.clone()
            
            for i in range(n1):
                neighbors = A1[i].nonzero(as_tuple=True)[0]
                neighbor_labels = labels1[neighbors].sort()[0]
                new_labels1[i] = hash(tuple(neighbor_labels.tolist())) % 10000
            
            for i in range(n2):
                neighbors = A2[i].nonzero(as_tuple=True)[0]
                neighbor_labels = labels2[neighbors].sort()[0]
                new_labels2[i] = hash(tuple(neighbor_labels.tolist())) % 10000
            
            labels1 = new_labels1
            labels2 = new_labels2
        
        return similarity
    
    def _random_walk_kernel(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute random walk graph kernel."""
        # Normalize adjacency matrices
        D1 = torch.diag(A1.sum(dim=1).pow(-0.5))
        D2 = torch.diag(A2.sum(dim=1).pow(-0.5))
        
        P1 = D1 @ A1 @ D1
        P2 = D2 @ A2 @ D2
        
        # Compute kernel via spectral decomposition
        lambda_param = 0.1
        n1, n2 = A1.shape[0], A2.shape[0]
        
        # Direct product kernel (simplified)
        kernel_val = torch.trace(
            torch.linalg.matrix_power(torch.eye(n1, device=A1.device) - lambda_param * P1, -1)
        ).item()
        
        return kernel_val
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute graph kernel similarity for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        gen_adj = generated[0]
        
        for target_idx in range(n_targets):
            target_adj = target[target_idx]
            
            if self.kernel_type == "weisfeiler_lehman":
                similarity = self._weisfeiler_lehman_kernel(gen_adj, target_adj)
            elif self.kernel_type == "random_walk":
                similarity = self._random_walk_kernel(gen_adj, target_adj)
            else:
                raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
            results[target_idx] = similarity
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return f"GraphKernel_{self.kernel_type}"

