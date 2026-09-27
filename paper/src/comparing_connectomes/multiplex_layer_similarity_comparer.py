import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class MultiplexLayerSimilarityEvaluator(NetworkEvaluator):
    """Evaluate multiplex networks using layer-similarity measures."""
    
    def __init__(self, similarity_type: str = "jaccard"):
        self.similarity_type = similarity_type
    
    def _jaccard_similarity(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute Jaccard similarity between edge sets."""
        A1_binary = (A1 > 0).float()
        A2_binary = (A2 > 0).float()
        
        intersection = (A1_binary * A2_binary).sum()
        union = ((A1_binary + A2_binary) > 0).float().sum()
        
        return (intersection / union).item() if union > 0 else 0.0
    
    def _overlap_coefficient(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute overlap coefficient between edge sets."""
        A1_binary = (A1 > 0).float()
        A2_binary = (A2 > 0).float()
        
        intersection = (A1_binary * A2_binary).sum()
        min_size = min(A1_binary.sum(), A2_binary.sum())
        
        return (intersection / min_size).item() if min_size > 0 else 0.0
    
    def _frobenius_similarity(self, A1: torch.Tensor, A2: torch.Tensor) -> float:
        """Compute normalized Frobenius similarity."""
        # Normalize matrices
        A1_norm = A1 / (torch.norm(A1, p='fro') + 1e-8)
        A2_norm = A2 / (torch.norm(A2, p='fro') + 1e-8)
        
        # Compute inner product
        similarity = torch.sum(A1_norm * A2_norm).item()
        return similarity
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute multiplex layer similarity for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        gen_adj = generated[0]
        
        for target_idx in range(n_targets):
            target_adj = target[target_idx]
            
            if self.similarity_type == "jaccard":
                similarity = self._jaccard_similarity(gen_adj, target_adj)
            elif self.similarity_type == "overlap":
                similarity = self._overlap_coefficient(gen_adj, target_adj)
            elif self.similarity_type == "frobenius":
                similarity = self._frobenius_similarity(gen_adj, target_adj)
            else:
                raise ValueError(f"Unknown similarity type: {self.similarity_type}")
            
            results[target_idx] = similarity
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return f"MultiplexSim_{self.similarity_type}"
