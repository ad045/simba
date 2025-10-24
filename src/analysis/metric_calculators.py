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
                                        # calculate_endpoint_similarity,
                                        # evaluate_adjacency
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
        if metric_name == "density": # Kayson
            return nx.density(nx.from_numpy_array(self.A))
        
        if metric_name == "density":  # Kayson
            return check_density(self.A)
        
        if metric_name == "wiring_cost":  # Kayson
            return calculate_wiring_cost(self.A, self.distance_matrix)
        
        if metric_name == "shortest_path_distance": # Kayson
            return shortest_path_distance(self.A)
        
        elif metric_name == "compute_structural_complexity": # Kayson
            return compute_structural_complexity(self.A)
        
        elif metric_name == "n_connected_components": # nx 
            G = nx.from_numpy_array(self.A)
            n_components = nx.number_connected_components(G)
            # Clean up graph object immediately
            del G
            return n_components
        
        elif metric_name == "omega": # Kayson, basically average clustering
            return compute_omega(self.A)
        
        elif metric_name == "topological_distance": # Kayson
            return topological_distance(self.A)
        
        elif metric_name == "resistance_distance": # Kayson ## "diffusion_distance"
            return resistance_distance(self.A)

        elif metric_name == "propagation_distance": # Kayson
            return propagation_distance(self.A)

        else:
            raise ValueError(f"Unknown metric: {metric_name}")


# # # #     [ ] Modularity (Consensus of N Louvain from netneurotools)
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

