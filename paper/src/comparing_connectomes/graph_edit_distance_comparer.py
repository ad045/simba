import torch
import numpy as np
from typing import Dict, Callable
import networkx as nx
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class GraphEditDistanceEvaluator(NetworkEvaluator):
    """Evaluate networks using exact graph edit distance with node correspondence (!!!). This basically makes it O(m). 
    
    Since nodes correspond between graphs, we enforce that nodes can only 
    match to their corresponding node in the other graph. This makes exact 
    GED computation tractable.
    """
    
    def __init__(self):
        """
        Args:
            node_match_required: If True, nodes must match by index (enforces correspondence)
        """
        # self.node_match_required = node_match_required

    
    def _create_node_match_func(self) -> Callable:
        """Create node matching function that enforces node correspondence."""
        def node_match(n1_attrs, n2_attrs):
            # Nodes must have the same 'id' attribute to match
            return n1_attrs.get('id') == n2_attrs.get('id')
        return node_match
    
    
    def _adjacency_to_graph_with_ids(self, adj_matrix: np.ndarray) -> nx.Graph:
        """Convert adjacency matrix to NetworkX graph with node IDs."""
        n = adj_matrix.shape[0]
        G = nx.Graph()
        
        # Add nodes with ID attribute for matching
        for i in range(n):
            G.add_node(i, id=i)
        
        # Add edges with weights
        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle only for undirected
                if adj_matrix[i, j] != 0:
                    G.add_edge(i, j, weight=float(adj_matrix[i, j]))
        
        return G
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute exact graph edit distance for generated network against target batch.
        
        Note: This uses exact GED which has exponential worst-case complexity.
        With node correspondence enforced, it becomes tractable for moderately-sized graphs.
        """
        results = {}
        n_targets = target.shape[0]
        
        # Convert generated to NetworkX graph
        gen_np = generated[0].cpu().numpy()
        G1 = self._adjacency_to_graph_with_ids(gen_np)
        
        # Prepare matching functions
        node_match = self._create_node_match_func() # if self.node_match_required else None
        
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            
            # Validate same size (required for node correspondence)
            if gen_np.shape != target_np.shape:
                raise ValueError(
                    f"Graphs must have same size for node correspondence. "
                    f"Got {gen_np.shape} and {target_np.shape}"
                )
            
            G2 = self._adjacency_to_graph_with_ids(target_np)
            
            # Compute exact GED with node correspondence
            distance = nx.graph_edit_distance(G1, G2, node_match=node_match)
            results[target_idx] = float(distance)

        return results


    @property
    def metric_prefix(self) -> str:
        # suffix = "_aligned" #if self.node_match_required else "_unaligned"
        return f"ExactGED" # {suffix}"