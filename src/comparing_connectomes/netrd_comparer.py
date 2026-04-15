
import torch
import numpy as np
from typing import Dict, List, Optional
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import networkx as nx
from abc import ABC, abstractmethod

from src.comparing_connectomes.base_comparer import NetworkEvaluator

class NetrdEvaluator(NetworkEvaluator):
    """
        Evaluate networks using different distances from netrd package.
    """
    
    def __init__(self, method: str = 'resistance'):
        """
        Args:
            method: 'resistance', 'net_simile', 'net_lsd', 'quantum_jsd'
        """
        self.method = method
        
        if self.method == 'resistance': 
            from netrd.distance import ResistancePerturbation
            self.distance_metric = ResistancePerturbation()

        elif self.method == "net_simile": 
            from netrd.distance import NetSimile
            self.distance_metric = NetSimile()

        elif self.method == "net_lsd":
            from netrd.distance import NetLSD
            self.distance_metric = NetLSD()

        elif self.method == "quantum_jsd":
            from netrd.distance import QuantumJSD
            self.distance_metric = QuantumJSD()
            
        elif self.method == "graph_diffusion":
            from netrd.distance import GraphDiffusion
            self.distance_metric = GraphDiffusion()
            
        elif self.method == "polynomial_dissimilarity":
            from netrd.distance import PolynomialDissimilarity
            self.distance_metric = PolynomialDissimilarity()

        elif self.method == "degree_divergence":
            from netrd.distance import DegreeDivergence
            self.distance_metric = DegreeDivergence()
            
        elif self.method == "onion_divergence":
            from netrd.distance import OnionDivergence
            self.distance_metric = OnionDivergence()

        elif self.method == "netrd_deltacon":
            from netrd.distance import DeltaCon
            self.distance_metric = DeltaCon()

        elif self.method == "netrd_communicability_jsd":
            from netrd.distance import CommunicabilityJSD
            self.distance_metric = CommunicabilityJSD()
        elif self.method == "distributional_nbd":
            from netrd.distance import DistributionalNBD
            self.distance_metric = DistributionalNBD()
        elif self.method == "dk_series":
            from netrd.distance import dkSeries
            self.distance_metric = dkSeries()
        elif self.method == "d_measure":
            from netrd.distance import DMeasure
            self.distance_metric = DMeasure()
        elif self.method == "netrd_frobenius":
            from netrd.distance import Frobenius
            self.distance_metric = Frobenius()
        elif self.method == "netrd_hamming":
            from netrd.distance import Hamming
            self.distance_metric = Hamming()
            
        elif self.method == "hamming_ipsen_mikhailov":
            from netrd.distance import HammingIpsenMikhailov
            self.distance_metric = HammingIpsenMikhailov()
            
        elif self.method == "ipsen_mikhailov":
            from netrd.distance import IpsenMikhailov
            self.distance_metric = IpsenMikhailov()
        elif self.method == "netrd_jaccard":
            from netrd.distance import JaccardDistance
            self.distance_metric = JaccardDistance()
        elif self.method == "netrd_laplacian_spectral":
            from netrd.distance import LaplacianSpectral
            self.distance_metric = LaplacianSpectral()
        elif self.method == "netrd_non_backtracking_spectral":
            from netrd.distance import NonBacktrackingSpectral
            self.distance_metric = NonBacktrackingSpectral()
        
        elif self.method == "netrd_portrait_divergence":
            from netrd.distance import PortraitDivergence
            self.distance_metric = PortraitDivergence()

        else: 
            raise ValueError(f"Unknown netrd method: {self.method}")
        
        
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute spectral distance for generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        # Convert generated to numpy
        gen_np = generated[0].cpu().numpy()
        G_gen = nx.from_numpy_array(gen_np)
        
        # Get distance for each target
        for target_idx in range(n_targets):
            target_np = target[target_idx].cpu().numpy()
            G_tar = nx.from_numpy_array(target_np)
            try:
                results[target_idx] = float(self.distance_metric.dist(G_gen, G_tar))
            except Exception:
                results[target_idx] = float("nan")
        
        return results
    
    
    @property
    def metric_prefix(self) -> str:
        return f"{self.method.capitalize()}"

