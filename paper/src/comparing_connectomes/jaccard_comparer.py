import torch
import numpy as np
from typing import Dict
from scipy.linalg import expm
from scipy.spatial.distance import jensenshannon
from src.comparing_connectomes.base_comparer import NetworkEvaluator



class JaccardEvaluator(NetworkEvaluator):
    """Evaluate networks using Jaccard similarity of edges."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute Jaccard similarity between binary edge sets."""
        results = {}
        gen_np = generated.squeeze().numpy()
        
        # Binarize generated network
        gen_binary = (gen_np > self.threshold).astype(int)
        
        for target_idx in range(target.shape[0]):
            target_np = target[target_idx].numpy()
            target_binary = (target_np > self.threshold).astype(int)
            
            # Jaccard: |A ∩ B| / |A ∪ B|
            intersection = np.logical_and(gen_binary, target_binary).sum()
            union = np.logical_or(gen_binary, target_binary).sum()
            
            jaccard = intersection / (union + 1e-10)
            results[target_idx] = jaccard
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "Jaccard"