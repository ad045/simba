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

from src.analysis.structural_measures import (
                                              calculate_modularity, 
                                              calculate_avg_degree,
                                              calculate_avg_clustering,
                                              calculate_degree_assortativity,
                                              calculate_transitivity,
                                            #   calculate_wiring_cost,  -> I'm using Kaysons 
                                              calculate_char_path_length
                                             )

              
from src.analysis.dynamic_measures import (
                                           spectral_radius, 
                                           spectral_gap,
                                           calculate_global_efficiency, 
                                           calculate_diffusion_efficiency, 
                                           average_controllability, 
                                           )

     
from src.analysis.computational_measures import (
                                                kernel_rank, 
                                                effective_dimensionality, 
                                                multifunctionality
                                                )   


class MetricCalculator(ABC):
    def __init__(self, A=None, distance_matrix=None):
        self.A = A
        self.G = nx.from_numpy_array(A) # A should be binary! 
        self.distance_matrix = distance_matrix
        self.implemented_metrics = set()  # To be defined in subclasses
    
    @abstractmethod
    def calculate_metric(self, metric_name):
        """Calculate the specified metric. Must be implemented by subclasses."""
        pass
    
    def _validate_inputs(self):
        """Common validation logic"""
        if self.A is None:
            raise ValueError("Adjacency matrix A is required")


class StaticMetricCalculator(MetricCalculator):

    def __init__(self, A=None, distance_matrix=None):
        super().__init__(A=A, distance_matrix=distance_matrix)
        self.implemented_metrics = {
            "density",
            "avg_clustering",
            "avg_degree",
            "degree_assortativity",
            "modularity",
            "average_degree",
            "characteristic_path_length",
            "transitivity",
            "wiring_cost",
            "shortest_path_distance",
            "compute_structural_complexity",
            "n_connected_components",
            "omega",
            "topological_distance",
            "resistance_distance",

        } # TODO: Check if the list is complete or if I already implemented more than that.  
    # Kayson: 
        # "density", "wiring_cost", "shortest_path_distance", "compute_structural_complexity",
        #  "n_connected_components", "omega", "topological_distance", "resistance_distance", 
        #  "propagation_distance", "propagation_efficiency"

    # nx: 
        # global_efficiency, modularity, transitivity, avg_clustering, degree_assortativity

    # Adrian:  
        # avg_degree (with nx) 
        # char_path_length (with nx) (of biggest component -> is this sensible??) 


    def calculate_metric(self, metric_name):
        
        # if metric_name == "density": # Kayson
        #     return nx.density(nx.from_numpy_array(self.A))
        
        if metric_name == "density":  # Kayson
            return check_density(self.A)
        
        if metric_name == "avg_clustering":  # nx
            return calculate_avg_clustering(self.G)
        
        if metric_name == "degree_assortativity":  # nx
            return calculate_degree_assortativity(self.G)

        if metric_name == "modularity":  
            return calculate_modularity(self.G)

        if metric_name == "average_degree":  
            return calculate_avg_degree(self.G)
        
        if metric_name == "characteristic_path_length":
            return calculate_char_path_length(self.G)
        
        if metric_name == "transitivity":
            return calculate_transitivity(self.G)
        
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



class DynamicMetricCalculator(MetricCalculator):
    
    def __init__(self, A=None, distance_matrix=None):
        super().__init__(A=A, distance_matrix=distance_matrix)
        self.implemented_metrics = {
            "spectral_radius",
            "spectral_gap",
            "global_efficiency",
            "diffusion_efficiency",
            "propagation_distance",
            "propagation_efficiency",
            "average_controllability",
        }
        
    def calculate_metric(self, metric_name):
        
        if metric_name == "spectral_radius": 
            return spectral_radius(self.A)
        
        elif metric_name == "spectral_gap":
            return spectral_gap(self.A)

        elif metric_name == "global_efficiency":  
            return calculate_global_efficiency(self.G)

        elif metric_name == "diffusion_efficiency": 
            return calculate_diffusion_efficiency(self.A)

        elif metric_name == "propagation_distance": # Kayson
            return propagation_distance(self.A)
            
        elif metric_name == "propagation_efficiency": # based on Kayson
            return 1/(propagation_distance(self.A).mean())

        elif metric_name == "average_controllability":
            return average_controllability(self.A)
        
        else:
            raise ValueError(f"Unknown metric: {metric_name}")


        # TODO 
        # Different implementation for dynamic networks

    #     [ ] PID (priesemann papers)
    # [ ] Spectral radius
    # [ ] normalized Entropy of the eigenspectrum (I have code)
    # [ ] Spectral gap
    # [ ] Maximum metastability and the corresponding coupling
    # [ ] Propagation efficiency (avg communicability)
    # [ ] Global efficiency (avg shortest path length)
    # [ ] Diffusion efficiency (avg effective distance)
    # [ ] avg controllability
        # if metric_name == "density":
        #     return self._calculate_temporal_density()


class ComputationMetricCalculator(MetricCalculator):
    
    def __init__(self, A=None, distance_matrix=None):
        super().__init__(A=A, distance_matrix=distance_matrix)
        self.implemented_metrics = {
            "kernel_rank",
            "effective_dimensionality",
            "multifunctionality",
        }
        
        
    def calculate_metric(self, metric_name):
        # [ ] Memory capacity

        if metric_name == "kernel_rank":
            return kernel_rank(self.A)
        
        if metric_name == "effective_dimensionality":
            return effective_dimensionality(self.A)
        
        if metric_name == "multifunctionality":
            return multifunctionality(self.A)
        
        else:
            raise ValueError(f"Unknown metric: {metric_name}")


# # # class PortraitMetricCalculator(MetricCalculator):
# # #     def calculate_metric(self, metric_name):
# # #         # TODO 
# # #         # Comparison to empirical connectomes: (or to default networks: random, etc?) 
# # #         # [ ] Portraits
# # #         # [ ] $Rˆ{2}$ (maybe not useful for MaMI, as we have no fixed connectome?)
# # #         if metric_name == "density":
# # #             return self._calculate_temporal_density()

