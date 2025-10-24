import numpy as np 
import networkx as nx
import pandas as pd
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import signal
import sys
import gc  # Garbage collector

from abc import ABC, abstractmethod   

from src.analysis.kayson_utils import (compute_structural_complexity, 
                                        compute_omega, 
                                        resistance_distance,
                                        shortest_path_distance,
                                        propagation_distance,
                                        topological_distance,
                                        check_density,
                                        calculate_wiring_cost,
                                        calculate_endpoint_similarity,
                                        evaluate_adjacency
                                       )     

class MetricCalculator(ABC):
    def __init__(self, A=None, distance_matrix=None):
        self.A = A
        self.distance_matrix = distance_matrix
    
    @abstractmethod
    def calculate_metric(self, metric_name):
        """Calculate the specified metric. Must be implemented by subclasses."""
        pass
    
    def _validate_inputs(self):
        """Common validation logic"""
        if self.A is None:
            raise ValueError("Adjacency matrix A is required")


class StaticMetricCalculator(MetricCalculator):
    def calculate_metric(self, metric_name):
        if metric_name == "density":
            return nx.density(nx.from_numpy_array(self.A))
        
        elif metric_name == "compute_structural_complexity":
            return compute_structural_complexity(self.A)
        
        elif metric_name == "n_connected_components": 
            G = nx.from_numpy_array(self.A)
            n_components = nx.number_connected_components(G)
            # Clean up graph object immediately
            del G
            return n_components
        
        elif metric_name == "omega": 
            return compute_omega(self.A)
        
        else:
            raise ValueError(f"Unknown metric: {metric_name}")




# # # class StaticMetricCalculator(MetricCalculator):
# # #     def calculate_metric(self, metric_name):
# # #         if metric_name == "density":
# # #             return nx.density(nx.from_numpy_array(self.A))
        
# # #         elif metric_name == "compute_structural_complexity":
# # #             return compute_structural_complexity(self.A)
        
# # #         elif metric_name == "n_connected_components": 
# # #             G = nx.from_numpy_array(self.A)
# # #             return nx.number_connected_components(G)
        
# # #         elif metric_name == "omega": 
# # #             return compute_omega(self.A)
        
# # #         # elif metric_name == "diffusion_distance": 
# # #         #     return resistance_distance()

# # # #             [ ] avg clustering
# # # #     [ ] Modularity (Consensus of N Louvain from netneurotools)
# # # #     [x] Small-worldness (omega from mine)
# # # #     [ ] avg wiring cost (Euclidean distance)
# # # #     [ ] Hubness (Gini index, Chini 2023)
# # # #     [ ] Rich club (with k, and looking for maximum → look for library)
# # # #     [ ] Average length


# # # # Number edges
# # # #     [ ] Average degree
# # # #     [ ] Transitivity
# # # #     [ ] Degree assortativity
# # # #     [ ] “Distance-dependent degree assortativity” (Betzel)
# # # #     [ ] Entropy of matrix


        
# # #         else:
# # #             raise ValueError(f"Unknown metric: {metric_name}")


# # # class DynamicMetricCalculator(MetricCalculator):
# # #     def calculate_metric(self, metric_name):
# # #         # TODO 
# # #         # Different implementation for dynamic networks

# # #     #     [ ] PID (priesemann papers)
# # #     # [ ] Spectral radius
# # #     # [ ] normalized Entropy of the eigenspectrum (I have code)
# # #     # [ ] Spectral gap
# # #     # [ ] Maximum metastability and the corresponding coupling
# # #     # [ ] Propagation efficiency (avg communicability)
# # #     # [ ] Global efficiency (avg shortest path length)
# # #     # [ ] Diffusion efficiency (avg effective distance)
# # #     # [ ] avg controllability
# # #         if metric_name == "density":
# # #             return self._calculate_temporal_density()


# # # class ComputationMetricCalculator(MetricCalculator):
# # #     def calculate_metric(self, metric_name):
# # #         # TODO 
# # #         # Different implementation for dynamic networks
# # #     # [ ] Memory capacity
# # #     # [ ] Kernel Rank
# # #     # [ ] Effective dimensionality
# # #     # [ ] Multifunctionality
# # #         if metric_name == "density":
# # #             return self._calculate_temporal_density()


# # # class PortraitMetricCalculator(MetricCalculator):
# # #     def calculate_metric(self, metric_name):
# # #         # TODO 
# # #         # Comparison to empirical connectomes: (or to default networks: random, etc?) 
# # #         # [ ] Portraits
# # #         # [ ] $Rˆ{2}$ (maybe not useful for MaMI, as we have no fixed connectome?)
# # #         if metric_name == "density":
# # #             return self._calculate_temporal_density()

