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
                                        resistance_distance, # from kayson, but changed to use distance matrix instead of coordinates. 
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
                                              calculate_char_path_length, 
                                              calculate_degree_gini
                                             )

              
from src.analysis.dynamic_measures import (
                                            spectral_radius, 
                                            spectral_gap,
                                            calculate_global_efficiency, 
                                            calculate_diffusion_efficiency, 
                                        #    average_controllability, 
                                            compute_spectral_gap_fatemeh, 
                                            calculate_nct_control, 
                                            calculate_nct_energies, 
                                            calculate_metastability, 
                                            # 
                                            compute_synchronizability_eigenratio, 
                                            algebraic_connectivity_nx,
                                            kuramoto_synchronization, 
                                            community_synchronization_vulnerability,
)

     
from src.analysis.computational_measures import (
                                                kernel_rank, 
                                                effective_dimensionality, 
                                                multifunctionality, 
                                                compute_kernel_rank_fatemeh,
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
            "characteristic_path_length",
            "transitivity",
            "wiring_cost",
            "shortest_path_distance",
            "structural_complexity",
            "n_connected_components",
            "omega",
            "topological_distance",
            "resistance_distance", 
            "degree_gini",

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

        if metric_name == "avg_degree":  
            return calculate_avg_degree(self.G)
        
        if metric_name == "characteristic_path_length":
            return calculate_char_path_length(self.G)
        
        if metric_name == "transitivity":
            return calculate_transitivity(self.G)
        
        if metric_name == "wiring_cost":  # Kayson -> similar enough to avg_wiring_cost 
            return calculate_wiring_cost(self.A, self.distance_matrix)
        
        if metric_name == "shortest_path_distance": # Kayson
            # shortest_path_distance_matrix = shortest_path_distance(self.A)
            # return {"mean": np.nanmean(shortest_path_distance_matrix),
            #         "std": np.nanstd(shortest_path_distance_matrix)}
            return shortest_path_distance(self.A)
        
        elif metric_name == "structural_complexity": # Kayson
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
            topological_distance_matrix = topological_distance(self.A)
            return {"mean": np.nanmean(topological_distance_matrix),
                    "std": np.nanstd(topological_distance_matrix)}

        elif metric_name == "resistance_distance": # Kayson ## "diffusion_distance"
            resistance_distance_matrix = resistance_distance(self.A, self.distance_matrix)
            return {"mean": np.nanmean(resistance_distance_matrix),
                    "std": np.nanstd(resistance_distance_matrix)}

        elif metric_name == "degree_gini": 
            return calculate_degree_gini(self.A) 
        
        else:
            raise ValueError(f"Unknown metric: {metric_name}")


# # # #     [ ] Modularity (Consensus of N Louvain from netneurotools)
# # # #     [ ] avg wiring cost (Euclidean distance)
# # # #     [ ] Rich club (with k, and looking for maximum → look for library)
# # # #     [ ] Average length
# # # #     [ ] Entropy of matrix



class DynamicMetricCalculator(MetricCalculator):
    
    def __init__(self, A=None, distance_matrix=None):
        super().__init__(A=A, distance_matrix=distance_matrix)
        self.implemented_metrics = {
            "spectral_radius",
            "spectral_gap",
            "global_efficiency",
            "diffusion_efficiency",
            # "propagation_distance", # Does not make sense, maybe? 
            "propagation_efficiency",
            "spectral_gap_fatemeh", 
            "nct_control",
            "nct_energies",
            "metastability", 
            # "average_controllability",
            "synchronizability_eigenratio",
            "algebraic_connectivity_nx",
            "kuramoto_synchronization",
            "community_synchronization_vulnerability",
        }
        
    def calculate_metric(self, metric_name):
        
        if metric_name == "spectral_radius": 
            return spectral_radius(self.A)
        
        elif metric_name == "spectral_gap":
            return spectral_gap(self.A)
        
        elif metric_name == "spectral_gap_fatemeh":
            return compute_spectral_gap_fatemeh(self.A)

        elif metric_name == "global_efficiency":  
            return calculate_global_efficiency(self.G)

        elif metric_name == "diffusion_efficiency": 
            return calculate_diffusion_efficiency(self.A)

        # elif metric_name == "propagation_distance": # Kayson
        #     return propagation_distance(self.A)
            
        elif metric_name == "propagation_efficiency": # based on Kayson. but there could also be a netneurotools way?
            return 1/(propagation_distance(self.A).mean())

        # elif metric_name == "average_controllability":
        #     return average_controllability(self.A)
        elif metric_name == "nct_control":
            return calculate_nct_control(self.A)

        elif metric_name == "nct_energies":
            return calculate_nct_energies(self.A)

        elif metric_name == "metastability": 
            return calculate_metastability(self.A)
 
        elif metric_name == "synchronizability_eigenratio":
            return compute_synchronizability_eigenratio(self.A)
            # R, lambda_2, lambda_N = compute_synchronizability_eigenratio(self.A)
            # return {"eigenratio": R, "lambda_2": lambda_2, "lambda_N": lambda_N}
        
        elif metric_name == "algebraic_connectivity_nx":
            return algebraic_connectivity_nx(self.A)
        
        elif metric_name == "kuramoto_synchronization":
            return kuramoto_synchronization(self.A, n_steps=1000)
        
        elif metric_name == "community_synchronization_vulnerability":
            return community_synchronization_vulnerability(self.A)
            # vulnerability, communities = community_synchronization_vulnerability(self.A)
            # return {"vulnerability": vulnerability, "n_communities": len(communities)}

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
            "kernel_rank_fatemeh",
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
        
        elif metric_name == "kernel_rank_fatemeh":
            return compute_kernel_rank_fatemeh(np.float64(self.A)) # , dtype=np.float32))
        
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

