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

import numpy as np
from typing import List, Dict, Optional, Any


from src.ESNs.utils_math import _entropy, _calculate_information_dynamics, _calculate_branching_ratio
import echoes
import numpy as np
from scipy.stats import pearsonr


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
                                              calculate_degree_gini, 
                                              calculate_proportion_long_range_connections,
                                              calculate_directed_simplices
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
                                            kuramoto_averaged_synchronization,
                                            community_synchronization_vulnerability,
                                            participation_coefficient, 
                                            departure_from_normality_schur
)

import src.analysis.utils_kayson_damicelli as ut

     
from src.analysis.computational_measures import (
                                                kernel_rank,
                                                effective_dimensionality,
                                                multifunctionality,
                                                compute_kernel_rank_fatemeh,
                                                # kernel_rank_esn, # Fatemeh. 
                                                )

from src.ESNs.memory_capacity_weighted import evaluate_memory_capacity_from_connectome, evaluate_nonlinear_capacity_from_connectome  

from src.analysis.my_attempt_at_metastability import calculate_metastability as calculate_metastability_2

from src.analysis.computational_capacity_measures import computational_capacity

from src.analysis.further_measures import (ollivier_ricci_curvature, 
                                           rich_club_coefficient, 
                                           participation_coefficient, 
                                           persistent_homology, 
                                           targeted_attack_robustness, 
                                           algebraic_connectivity,
                                           basic_measures) 

from src.analysis.from_fatemeh import departure_from_normality # , # kernel_rank_esn, compute_spectral_gap_fatemeh
from src.analysis.from_francisco_newer import repertoire_sweep_weighted_by_distances # repertoire, repertoire_sweep, 

    
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
            "proportion_long_range_connections",
            "directed_simplices"

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
            n_components = nx.number_connected_components(self.G)
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
        
        elif metric_name == "directed_simplices":
            return calculate_directed_simplices(self.A)
        
        elif metric_name == "proportion_long_range_connections":
            return calculate_proportion_long_range_connections(self.A, self.distance_matrix)

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
            "kernel_rank_esn", # Fatemeh's version of kernel rank based on ESNs. War schon ausgeklammert... 
            "departure_from_normality", # also fatemeh
            "departure_from_normality_schur", 
            "nct_control",
            "nct_energies",
            "novel_metastability", # previously: metastability... but this is for testing now 
            # "average_controllability",
            "synchronizability_eigenratio",
            "algebraic_connectivity_nx",
            "kuramoto_synchronization",
            "kuramoto_averaged_synchronization",
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

        elif metric_name == "novel_metastability": # metastability_2": 
            return calculate_metastability_2(self.A, distance_matrix=self.distance_matrix, save_debug_path="/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/metastability_debug_nov_19") # calculate_metastability(self.A)
 
        elif metric_name == "synchronizability_eigenratio":
            return compute_synchronizability_eigenratio(self.A)
            # R, lambda_2, lambda_N = compute_synchronizability_eigenratio(self.A)
            # return {"eigenratio": R, "lambda_2": lambda_2, "lambda_N": lambda_N}
        
        elif metric_name == "algebraic_connectivity_nx":
            return algebraic_connectivity_nx(self.A)
        
        elif metric_name == "kuramoto_synchronization":
            return kuramoto_synchronization(self.A) 
        
        elif metric_name == "kuramoto_averaged_synchronization":
            return kuramoto_averaged_synchronization(self.A)
        
        elif metric_name == "community_synchronization_vulnerability":
            return community_synchronization_vulnerability(self.A)

        elif metric_name == "computational_capacity":
            return computational_capacity(self.A)

        elif metric_name == "kernel_rank_esn":
            return kernel_rank_esn(self.A)
        
        elif metric_name == "departure_from_normality":
            return departure_from_normality(self.A)
        
        elif metric_name == "departure_from_normality_schur": 
            return departure_from_normality_schur(self.A)

        elif metric_name == "participation_coefficient": 
            return participation_coefficient(self.A) 

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
            "computational_capacity",
            "mc_original", "mc_nonlinear_original", 
            "mc_lin_cut40", 
            "mc_nonlin_cut40",
            "repertoire", "repertoire_sweep", "repertoire_sweep_weighted_by_distances", 
        }
        
        # Load the esn data for the memory capacity evaluation. This is currently hardcoded, but it could be made more flexible if needed. 
        # self.X_train, self.X_test, self.y_train_linear, self.y_test_linear, self.y_train_nonlinear, self.y_test_nonlinear = self._load_esn_data()
        self.X_train_cut40, self.X_test_cut40, self.y_train_linear_cut40, self.y_test_linear_cut40, self.y_train_nonlinear_cut40, self.y_test_nonlinear_cut40 = self._load_esn_data()


    def _load_esn_data(self):
        # Load the data for the memory capacity evaluation. This is currently hardcoded, but it could be made more flexible if needed. 
        base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/tasks/esn")
        X_train = np.load(base_path / "X_train.npy")
        X_test = np.load(base_path / "X_test.npy")
        y_train_linear = np.load(base_path / "y_train_linear.npy")
        y_test_linear = np.load(base_path / "y_test_linear.npy")
        y_train_nonlinear = np.load(base_path / "y_train_nonlinear.npy")
        y_test_nonlinear = np.load(base_path / "y_test_nonlinear.npy")
        
        # cut40 version:
        X_train_cut40 = np.load(base_path / "X_train_cut40.npy")
        X_test_cut40 = np.load(base_path / "X_test_cut40.npy")
        y_train_linear_cut40 = np.load(base_path / "y_train_linear_cut40.npy")
        y_test_linear_cut40 = np.load(base_path / "y_test_linear_cut40.npy")
        y_train_nonlinear_cut40 = np.load(base_path / "y_train_nonlinear_cut40.npy")
        y_test_nonlinear_cut40 = np.load(base_path / "y_test_nonlinear_cut40.npy")

        # return X_train_cut40, X_test_cut40, y_train_linear_cut40, y_test_linear_cut40, y_train_nonlinear_cut40, y_test_nonlinear_cut40
        return X_train_cut40, X_test_cut40, y_train_linear_cut40, y_test_linear_cut40, y_train_nonlinear_cut40, y_test_nonlinear_cut40


    def evaluate_esn_memory_capacity(self, h_params, X_train, X_test, y_train, y_test):
        
        from scipy.stats import pearsonr
        
        for run in range(h_params["n_runs"]):
            
            # h_params = {
            #     "spectral_radius": 0.99, # default
            #     "leak_rate": 1, # default. 0.3, #3, # 5, # 1.0,
            #     "n_lags": 40, 
            #     # "train_len": 5000,
            #     # "test_len": 1000,
            #     "n_runs": 10, 
            #     "input_scaling": 1e-5, 
            #     "regression_method": "pinv", # ridge", # pinv",
            #     "n_transient": 100, # 100,
            
            #     "bias": 1, # default. 0.5, # 1.0,
            #     "random_state": None
            # }
            
            # Initialize empty results array for this run
            results = np.zeros((h_params["n_runs"], 1))  # Assuming 1 subject for now, can be expanded to more subjects if needed
            meaned_results = [] 

            esn = echoes.ESNRegressor(W=np.array(self.A, dtype=np.float64), # W=self.A, 
                               spectral_radius=h_params["spectral_radius"],
                               input_scaling=h_params["input_scaling"],
                               leak_rate=h_params["leak_rate"],
                               bias=h_params["bias"],
                               regression_method=h_params["regression_method"],
                               n_transient=h_params["n_transient"],
                               random_state=h_params["random_state"]
                               )
            y_pred = esn.fit(X_train, y_train).predict(X_test)

            # Evaluates the MC score: evaluated[1] is the sum of the R^2 values across all lags, and evaluated[0] is the list of R^2 values for each individual lag.
            evaluated = ut.forgetting(y_test[h_params["n_transient"]:], y_pred[h_params["n_transient"]:])

            return {
                "mc_mean": evaluated[1],
                "mc_std": np.std(evaluated[0]), 
                "mc_values_for_indiv_lags": evaluated[0]
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
        
        elif metric_name == "computational_capacity":
            return computational_capacity(self.A)
        
        elif metric_name == "repertoire": 
            return repertoire(self.A, T=0.1)

        elif metric_name == "repertoire_sweep": 
            return repertoire_sweep(self.A) # np.logspace(-2, 0, 201), for denser thing

        elif metric_name == "repertoire_sweep_weighted_by_distances": 
            return repertoire_sweep_weighted_by_distances(self.A, self.distance_matrix) # np.logspace(-2, 0, 201), for denser thing
        
        elif metric_name == "mc_original" or metric_name == "mc_nonlinear_original":
            
            # TODO: Get hyperparameters for the memory capacity evaluation. These can be adjusted as needed, but for now I'm using the same ones as in the original script.
            # Currently just hardcoded 
            # h_params = {
            #     "spectral_radius": 0.9,
            #     "n_lags": 50,
            #     "train_len": 5000,
            #     "test_len": 1000,
            #     "n_runs": 50,
            #     "input_scaling": 1.0,
            #     "regression_method": "pinv",
            #     "n_transient": 0, # 100,
            #     "leak_rate": 1.0,
            #     "bias": 0.0,
            #     "random_state": 42
            # }
            
            from scipy.stats import pearsonr
            
            h_params = {
                "spectral_radius": 0.99, # 0.9
                "n_lags": 40, 
                # "train_len": 5000,
                # "test_len": 1000,
                "n_runs": 10, 
                "input_scaling": 1e-5, 
                "regression_method": "pinv", # ridge", # pinv",
                "n_transient": 100, # 100,
                "leak_rate": 1, # default. 0.3, #3, # 5, # 1.0,
                "bias": 1, # default. 0.5, # 1.0,
                "random_state": None
            }
            
            if metric_name == "mc_original": 
                X_train = self.X_train
                X_test = self.X_test
                y_train = self.y_train_linear
                y_test = self.y_test_linear
                
                result = self.evaluate_esn_memory_capacity(h_params=h_params, 
                                                           X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
                
            elif metric_name == "mc_nonlinear_original":
                X_train = self.X_train
                X_test = self.X_test
                y_train = self.y_train_nonlinear
                y_test = self.y_test_nonlinear

                result = self.evaluate_esn_memory_capacity(h_params=h_params, 
                                                           X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

            # Flatten per-lag MC values into individual keys (mc_0, mc_1, ..., mc_49)
            flat_result = {
                "mc_mean": result["mc_mean"],
                "mc_std": result["mc_std"],
                # "mc_values_for_indiv_lags": result["mc_values_for_indiv_lags"]
            }
            
            if metric_name == "mc_original": 
                mc_per_lag = result.get("mc_values_for_indiv_lags", [])
                for i, val in enumerate(mc_per_lag):
                    flat_result[f"mc_{i}"] = float(val)

            return flat_result
        


        elif metric_name == "mc_lin_cut40" or metric_name == "mc_nonlin_cut40":

            # TODO: Get hyperparameters for the memory capacity evaluation. These can be adjusted as needed, but for now I'm using the same ones as in the original script.
            # Currently just hardcoded 
            # h_params = {
            #     "spectral_radius": 0.9,
            #     "n_lags": 50,
            #     "train_len": 5000,
            #     "test_len": 1000,
            #     "n_runs": 50,
            #     "input_scaling": 1.0,
            #     "regression_method": "pinv",
            #     "n_transient": 0, # 100,
            #     "leak_rate": 1.0,
            #     "bias": 0.0,
            #     "random_state": 42
            # }
            
            from scipy.stats import pearsonr
            
            h_params = {
                "spectral_radius": 0.95, # 0.9
                "n_lags": 40, 
                # "train_len": 5000,
                # "test_len": 1000,
                "n_runs": 1000, 
                "input_scaling": 0.1, # 1e-5, 
                "regression_method": "pinv", # ridge", # pinv",
                "n_transient": 100, # 100,
                "leak_rate": 1, # default. 0.3, #3, # 5, # 1.0,
                "bias": 1, # default. 
                "random_state": None
            }
            
            if metric_name == "mc_lin_cut40": 
                X_train = self.X_train_cut40
                X_test = self.X_test_cut40
                y_train = self.y_train_linear_cut40
                y_test = self.y_test_linear_cut40

                result = self.evaluate_esn_memory_capacity(h_params=h_params, 
                                                           X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

            elif metric_name == "mc_nonlin_cut40":
                X_train = self.X_train_cut40
                X_test = self.X_test_cut40
                y_train = self.y_train_nonlinear_cut40
                y_test = self.y_test_nonlinear_cut40

                result = self.evaluate_esn_memory_capacity(h_params=h_params, 
                                                           X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

            # Flatten per-lag MC values into individual keys (mc_0, mc_1, ..., mc_49)
            flat_result = {
                "mc_mean": result["mc_mean"],
                "mc_std": result["mc_std"],
                # "mc_values_for_indiv_lags": result["mc_values_for_indiv_lags"]
            }
            
            if metric_name == "mc_lin_cut40": 
                mc_per_lag = result.get("mc_values_for_indiv_lags", [])
                for i, val in enumerate(mc_per_lag):
                    flat_result[f"mc_{i}"] = float(val)

            return flat_result
        
        

        elif metric_name == "mc_input_scaling_0_1" or metric_name == "mc_nonlinear_input_scaling_0_1":
            
            # TODO: Get hyperparameters for the memory capacity evaluation. These can be adjusted as needed, but for now I'm using the same ones as in the original script.
            # Currently just hardcoded 
            # h_params = {
            #     "spectral_radius": 0.9,
            #     "n_lags": 50,
            #     "train_len": 5000,
            #     "test_len": 1000,
            #     "n_runs": 50,
            #     "input_scaling": 1.0,
            #     "regression_method": "pinv",
            #     "n_transient": 0, # 100,
            #     "leak_rate": 1.0,
            #     "bias": 0.0,
            #     "random_state": 42
            # }
            
            from scipy.stats import pearsonr
            
            h_params = {
                "spectral_radius": 0.9, # 0.9
                "n_lags": 40, 
                # "train_len": 5000,
                # "test_len": 1000,
                "n_runs": 500, # 10, 
                "input_scaling": 0.1, #  1e-5, 
                "regression_method": "pinv", # ridge", # pinv",
                "n_transient": 100, # 100,
                "leak_rate": 0.3, # default. 0.3, #3, # 5, # 1.0,
                "bias": 0, # 1, # default. 0.5, # 1.0,
                "random_state": None
            }
            
            if metric_name == "mc_input_scaling_0_1": # "mc_original": 
                X_train = self.X_train
                X_test = self.X_test
                y_train = self.y_train_linear
                y_test = self.y_test_linear
                
                result = self.evaluate_esn_memory_capacity(h_params=h_params, 
                                                           X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

            elif metric_name == "mc_nonlinear_input_scaling_0_1": # "mc_nonlinear_original":
                X_train = self.X_train
                X_test = self.X_test
                y_train = self.y_train_nonlinear
                y_test = self.y_test_nonlinear

                result = self.evaluate_esn_memory_capacity(h_params=h_params, 
                                                           X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

            # Flatten per-lag MC values into individual keys (mc_0, mc_1, ..., mc_49)
            flat_result = {
                "mc_mean": result["mc_mean"],
                "mc_std": result["mc_std"],
                # "mc_values_for_indiv_lags": result["mc_values_for_indiv_lags"]
            }
            
            if metric_name == "mc_input_scaling_0_1": 
                mc_per_lag = result.get("mc_values_for_indiv_lags", [])
                for i, val in enumerate(mc_per_lag):
                    flat_result[f"mc_{i}"] = float(val)

            return flat_result
        
        
        
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




class FurtherMetricCalculator(MetricCalculator):
    
    def __init__(self, A=None, distance_matrix=None):
        super().__init__(A=A, distance_matrix=distance_matrix)
        self.implemented_metrics = {
            "ollivier_ricci_curvature", 
            "rich_club_coefficient", 
            "participation_coefficient", 
            "persistent_homology", 
            "targeted_attack_robustness", 
            "algebraic_connectivity", 
            "basic_measures",
        }
        
        
    def calculate_metric(self, metric_name):
        if metric_name == "basic_measures":
            return basic_measures(self.A, self.distance_matrix)
        
        if metric_name == "ollivier_ricci_curvature":
            return ollivier_ricci_curvature(self.A)

        if metric_name == "rich_club_coefficient":
            return rich_club_coefficient(self.A)

        if metric_name == "participation_coefficient":
            return participation_coefficient(self.A)

        if metric_name == "persistent_homology":
            return persistent_homology(self.A)

        if metric_name == "targeted_attack_robustness":
            return targeted_attack_robustness(self.A)

        if metric_name == "algebraic_connectivity":
            return algebraic_connectivity(self.A)
        
        else:
            raise ValueError(f"Unknown metric: {metric_name}")



