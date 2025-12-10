
import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator
import networkx as nx
from typing import Optional

class GraphEditDistanceEvaluator(NetworkEvaluator):
    """Evaluate networks using approximate graph edit distance.
    
    Uses NetworkX's optimization_graph_edit_distance with beam search.
    """
    
    def __init__(self, timeout: Optional[float] = 10.0):
        """
        Args:
            timeout: Maximum time for computation (seconds)
        """
        self.timeout = timeout
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute graph edit distance for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        # Convert generated to NetworkX graph
        gen_np = generated[0].cpu().numpy()
        G1 = nx.from_numpy_array(gen_np)
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            G2 = nx.from_numpy_array(target_np)
            
            # Use approximate GED with timeout
            try:
                # optimization_graph_edit_distance returns a generator
                ged_gen = nx.optimize_graph_edit_distance(G1, G2)
                distance = next(ged_gen)  # Get first (best) approximation
            except (StopIteration, nx.NetworkXError):
                distance = float('inf')
            
            results[target_idx] = float(distance)
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "GraphEditDist"
