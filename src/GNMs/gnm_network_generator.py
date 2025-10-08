"""
Optimized GNM-based network generation fully leveraging Edward's GNM library.
"""
# -> Used at least for main_pipeline_2_gnm_esn_landscape.py

import numpy as np
import torch
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path 

# Import everything we need from GNM library
from gnm import (
    fitting, 
    generative_rules, 
    evaluation, 
    weight_criteria
)
# from gnm.model import BinaryGenerativeParameters

# For the dynamicGNMGenerator
from src.ESNs.esn_evaluation import ESNEvaluator 

@dataclass
class GNMParameters:
    """Parameters for GNM generation - matches GNM library structure."""
    eta: Union[float, torch.Tensor] = -2.0                    # Distance parameter
    gamma: Union[float, torch.Tensor] = 0.3                   # Homophily parameter
    lambdah: Union[float, torch.Tensor] = 0.0                # Time-dependency parameter
    generative_rule: Any = None                               # Will be set to actual rule object
    distance_relationship_type: Union[str, List[str]] = "powerlaw"
    preferential_relationship_type: Union[str, List[str]] = "powerlaw"
    heterochronicity_relationship_type: Union[str, List[str]] = "powerlaw"
    num_simulations: int = 100
    device: str = "cpu"
    random_seed: Optional[int] = None


class GNMGenerator:
    """Optimized GNM generator using all available GNM library features."""
    
    # All available generative rules from GNM library
    AVAILABLE_RULES = {
        "matching_index": generative_rules.MatchingIndex,
        "neighbors": generative_rules.Neighbours,
        "degree_average": generative_rules.DegreeAverage,
        "degree_difference": generative_rules.DegreeDifference,
        "degree_product": generative_rules.DegreeProduct,
        "degree_max": generative_rules.DegreeMax,
        "degree_min": generative_rules.DegreeMin,
        # "clustering_coefficient": generative_rules. ClusteringClusteringCoefficient,
        # "clustering_coefficient_zhang": generative_rules.ClusteringCoefficientZhang,
        # "clustering_coefficient_bu": generative_rules.ClusteringCoefficientBu,
        # "spatial": generative_rules.Spatial,
        # "communicability": generative_rules.Communicability,
        # "adamic_adar": generative_rules.AdamicAdar,
        # "path_index_2": generative_rules.PathIndex2,
        # "preferential_attachment": generative_rules.PreferentialAttachment,
        # "resource_allocation": generative_rules.ResourceAllocation,
        "clustering_average": generative_rules.ClusteringAverage,
        "clustering_difference": generative_rules.ClusteringDifference,
        "clustering_product": generative_rules.ClusteringProduct, 
        "clustering_max": generative_rules.ClusteringMax,
        "clustering_min": generative_rules.ClusteringMin,
        
    }
    
    # All available evaluation metrics from GNM library
    AVAILABLE_METRICS = {
        "degree_ks": evaluation.DegreeKS,
        "edge_length_ks": evaluation.EdgeLengthKS,
        "clustering_ks": evaluation.ClusteringKS,
        "betweenness_ks": evaluation.BetweennessKS,
        "edge_length_ks": evaluation.EdgeLengthKS,
        "weighted_clustering_ks": evaluation.WeightedClusteringKS,
        # "degree_js": evaluation.DegreeJS,
        # "degree_correlatioin": evaluation.DegreeCorrelation,
        # "clustering_js": evaluation.ClusteringJS,
        # "max_criteria": evaluation.MaxCriteria,
        # "mean_criteria": evaluation.MeanCriteria,
        # "weighted_criteria": evaluation.WeightedCriteria,
        # "betweenness_correlation": evaluation.BetweennessCorrelation, 
        # "clustering_correlation": evaluation.ClusteringCorrelation,

        "weighted_betweenness_correlation": evaluation.WeightedBetweennessKS,
    }
    # Available weight optimization criteria - from init (unsure what to do with this yet) 
            #     "EvaluationCriterion",
            # "BinaryEvaluationCriterion",
            # "WeightedEvaluationCriterion",
            # "CorrelationCriterion",
            # "CompositeCriterion",
            # "WeightedSumCriteria",
            # "WeightedNodeStrengthKS",
    
    def __init__(self, device: str):
        """Initialize with device selection."""
        self.device = device # torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    def generate_network(self,
                        n_nodes: int,
                        n_edges: int,
                        distance_matrix: torch.Tensor,
                        parameters: GNMParameters,
                        seed_network: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate a network using GNM library's binary model.
        
        Args:
            n_nodes: Number of nodes
            n_edges: Target number of edges
            distance_matrix: Distance matrix between nodes
            parameters: GNM parameters
            seed_network: Optional seed network to start from
            
        Returns:
            Generated binary network
        """
        # Ensure inputs are on correct device
        distance_matrix = distance_matrix.to(self.device)
        
        if seed_network is not None:
            seed_network = seed_network.to(self.device)
        else:
            seed_network = torch.zeros((n_nodes, n_nodes), device=self.device)
        
        # Create binary model using GNM library
        model = GNMBinary(
            eta=parameters.eta,
            gamma=parameters.gamma,
            lambdah=parameters.lambdah,
            distance_relationship_type=parameters.distance_relationship_type,
            preferential_relationship_type=parameters.preferential_relationship_type,
            heterochronicity_relationship_type=parameters.heterochronicity_relationship_type,
            generative_rule=parameters.generative_rule,
            distance_matrix=distance_matrix,
            adjacency_matrix=seed_network,
            device=self.device
        )
        
        # Run the model for specified iterations
        model.run(n_edges)
        
        # Get the generated network
        return model.get_adjacency_matrix()
    
    def fit_parameters(self,
                      target_network: torch.Tensor,
                      distance_matrix: torch.Tensor,
                      eta_range: Tuple[float, float] = (-5.0, 0.0),
                      gamma_range: Tuple[float, float] = (0.0, 1.0),
                      n_eta: int = 20,
                      n_gamma: int = 20,
                      generative_rule_name: str = "matching_index",
                      given_output_path: Optional[Path] = None, 
                      num_simulations: int = 100,
                      evaluation_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fit GNM parameters using the library's sweep functionality.
        
        Args:
            target_network: Target network to fit
            distance_matrix: Distance matrix
            eta_range: Range for eta parameter
            gamma_range: Range for gamma parameter
            n_eta: Number of eta values to test
            n_gamma: Number of gamma values to test
            generative_rule_name: Name of generative rule to use
            num_simulations: Number of simulations per parameter set
            evaluation_metrics: List of evaluation metric names
            
        Returns:
            Dictionary with fitting results
        """
        # Prepare inputs
        target_network = target_network.to(self.device)
        distance_matrix = distance_matrix.to(self.device)
        n_edges = int(target_network.sum().item() // 2)
        
        # Get generative rule
        if generative_rule_name not in self.AVAILABLE_RULES:
            raise ValueError(f"Unknown rule: {generative_rule_name}. Available: {list(self.AVAILABLE_RULES.keys())}")
        generative_rule = self.AVAILABLE_RULES[generative_rule_name]()
        
        # Set up evaluation metrics
        if evaluation_metrics is None:
            evaluation_metrics = ["degree_ks", "clustering_ks", "edge_length_ks"]
        
        criteria = []
        for metric_name in evaluation_metrics:
            if metric_name == "edge_length_ks":
                criteria.append(self.AVAILABLE_METRICS[metric_name](distance_matrix))
            elif metric_name in self.AVAILABLE_METRICS:
                criteria.append(self.AVAILABLE_METRICS[metric_name]())
        
        energy_equation = evaluation.MaxCriteria(criteria)
        
        # Create parameter sweep using GNM library
        binary_sweep_parameters = fitting.BinarySweepParameters(
            eta=torch.linspace(eta_range[0], eta_range[1], n_eta),
            gamma=torch.linspace(gamma_range[0], gamma_range[1], n_gamma),
            lambdah=torch.tensor([0.0]),
            distance_relationship_type=["powerlaw"],
            preferential_relationship_type=["powerlaw"],
            heterochronicity_relationship_type=["powerlaw"],
            generative_rule=[generative_rule],
            num_iterations=[n_edges],
        )
        
        sweep_config = fitting.SweepConfig(
            binary_sweep_parameters=binary_sweep_parameters,
            weighted_sweep_parameters=None,
            num_simulations=num_simulations,
            distance_matrix=[distance_matrix]
        )
        
        # Run parameter sweep
        experiments = fitting.perform_sweep(
            sweep_config=sweep_config,
            binary_evaluations=[energy_equation],
            real_binary_matrices=target_network,
            given_output_path=given_output_path, 
            save_model=False,
            save_run_history=False,
            # experiment_name=experiment_name, (added by me)
            verbose=False
        )
        
        # Find optimal parameters
        optimal_experiments, optimal_energies = fitting.optimise_evaluation(
            experiments=experiments,
            criterion=energy_equation,
        )
        
        if optimal_experiments:
            best_exp = optimal_experiments[0]
            return {
                "best_eta": float(best_exp.run_config.binary_parameters.eta),
                "best_gamma": float(best_exp.run_config.binary_parameters.gamma),
                "best_energy": float(optimal_energies[0]),
                "all_experiments": experiments,
                "generative_rule": generative_rule_name,
                "n_edges": n_edges
            }
        else:
            raise RuntimeError("No optimal parameters found")
    
    def batch_generate_with_weights(self,
                                  binary_networks: List[torch.Tensor],
                                  distance_matrix: torch.Tensor,
                                  binary_params: GNMParameters,
                                  weight_criterion: Optional[Any] = None,
                                  alpha: float = 0.01) -> List[torch.Tensor]:
        """
        Generate binary networks and optimize their weights.
        
        Args:
            binary_networks: List of target binary networks
            distance_matrix: Distance matrix
            binary_params: Parameters for binary generation
            weight_criterion: Weight optimization criterion (from weight_criteria module)
            alpha: Learning rate for weight optimization
            
        Returns:
            List of weighted networks
        """
        if weight_criterion is None:
            weight_criterion = weight_criteria.DistanceWeightedCommunicability(distance_matrix)
        
        weighted_networks = []
        
        for binary_net in binary_networks:
            n_edges = int(binary_net.sum().item() // 2)
            
            # Generate binary network
            generated_binary = self.generate_network(
                n_nodes=binary_net.shape[0],
                n_edges=n_edges,
                distance_matrix=distance_matrix,
                parameters=binary_params
            )
            
            # Optimize weights using GNM's weight optimization
            weighted_sweep_params = fitting.WeightedSweepParameters(
                alpha=[alpha],
                optimisation_criterion=[weight_criterion]
            )
            
            sweep_config = fitting.SweepConfig(
                binary_sweep_parameters=None,
                weighted_sweep_parameters=weighted_sweep_params,
                num_simulations=1,
                distance_matrix=[distance_matrix]
            )
            
            # Create weighted model
            from gnm.models import GNMWeighted
            weighted_model = GNMWeighted(
                binary_adjacency_matrix=generated_binary,
                alpha=alpha,
                optimisation_criterion=weight_criterion,
                device=self.device
            )
            
            # Run weight optimization
            weighted_model.run(num_iterations=1000)
            weighted_networks.append(weighted_model.get_adjacency_matrix())
        
        return weighted_networks
    
    def evaluate_network(self,
                        generated_network: torch.Tensor,
                        target_network: torch.Tensor,
                        distance_matrix: torch.Tensor,
                        metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate a generated network against a target using GNM metrics.
        
        Args:
            generated_network: Generated network
            target_network: Target network
            distance_matrix: Distance matrix
            metrics: List of metric names to compute
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ["degree_ks", "clustering_ks", "edge_length_ks", "frobenius"]
        
        results = {}
        
        for metric_name in metrics:
            if metric_name not in self.AVAILABLE_METRICS:
                warnings.warn(f"Unknown metric: {metric_name}")
                continue
            
            if metric_name == "edge_length_ks" or metric_name == "edge_length_js":
                metric = self.AVAILABLE_METRICS[metric_name](distance_matrix)
            else:
                metric = self.AVAILABLE_METRICS[metric_name]()
            
            value = metric(generated_network, target_network)
            results[metric_name] = float(value.item() if torch.is_tensor(value) else value)
        
        return results
    
    def compare_generative_rules(self,
                                target_network: torch.Tensor,
                                distance_matrix: torch.Tensor,
                                rules_to_test: Optional[List[str]] = None,
                                n_simulations: int = 50) -> Dict[str, Any]:
        """
        Compare different generative rules for a target network.
        
        Args:
            target_network: Target network
            distance_matrix: Distance matrix
            rules_to_test: List of rule names to test
            n_simulations: Number of simulations per rule
            
        Returns:
            Comparison results
        """
        if rules_to_test is None:
            # Test the most common rules
            rules_to_test = ["matching_index", "neighbors", "degree_product", 
                           "clustering_coefficient", "spatial"]
        
        results = {}
        
        for rule_name in rules_to_test:
            if rule_name not in self.AVAILABLE_RULES:
                warnings.warn(f"Skipping unknown rule: {rule_name}")
                continue
            
            print(f"Testing {rule_name}...")
            
            try:
                fit_result = self.fit_parameters(
                    target_network=target_network,
                    distance_matrix=distance_matrix,
                    generative_rule_name=rule_name,
                    n_eta=10,  # Reduced for speed
                    n_gamma=10,
                    num_simulations=n_simulations
                )
                
                results[rule_name] = {
                    "best_eta": fit_result["best_eta"],
                    "best_gamma": fit_result["best_gamma"],
                    "best_energy": fit_result["best_energy"]
                }
            except Exception as e:
                results[rule_name] = {"error": str(e)}
        
        # Find best rule
        best_rule = min(
            [k for k, v in results.items() if "best_energy" in v],
            key=lambda x: results[x]["best_energy"]
        )
        
        results["best_rule"] = best_rule
        
        return results


############## 

class DynamicGNMGenerator(GNMGenerator):
    """
    A GNM generator that incorporates ESN-based dynamic fitness into the edge selection process.
    """
    
    def __init__(self, esn_evaluator: ESNEvaluator, device: Optional[str] = None):
        """Initialize with an ESN evaluator."""
        super().__init__(device)
        self.esn_evaluator = esn_evaluator
        # Ensure the evaluator uses a single worker for this internal task
        self.esn_evaluator.config.compute.n_workers = 1

    def generate_dynamically_aware_network(self,
                                           n_nodes: int,
                                           n_edges: int,
                                           distance_matrix: torch.Tensor,
                                           gnm_params: GNMParameters,
                                           dynamic_config: Dict[str, Any], 
                                           fitness_metric: str = "mc_mean") -> torch.Tensor:
        """
        Generates a network using the candidate re-ranking algorithm.
        
        Args:
            n_nodes: Number of nodes.
            n_edges: Target number of edges.
            distance_matrix: Distance matrix.
            gnm_params: Standard GNM parameters (eta, gamma, rule).
            dynamic_config: Dictionary with dynamic parameters (delta, pool_size, etc.).
            fitness_metric: The dynamic metric to optimize (e.g., 'mc_mean', 'branching_ratio').

        Returns:
            The generated dynamically-aware binary network.
        """
        print(f"Starting dynamically-aware GNM generation (optimizing for: {fitness_metric})...")
      
        
        # Initialize network
        adjacency_matrix = torch.zeros((n_nodes, n_nodes), device=self.device)
        distance_matrix = distance_matrix.to(self.device)
        
        # Pre-compute distance term
        dist_term = distance_matrix.pow(gnm_params.eta)
        dist_term.fill_diagonal_(0)
        
        # Add batch dimension for compatibility with GNM library (jax)
        adjacency_matrix = adjacency_matrix.unsqueeze(0)

        for edge_k in range(n_edges):
            print(f"  Adding edge {edge_k + 1}/{n_edges}...", end='\r')
            
            # Candidate selection (standard GNM)
            # Calculate homophily term using the chosen rule
            K = gnm_params.generative_rule(adjacency_matrix)
            K_term = K.pow(gnm_params.gamma)
            
            # Calculate standard GNM probability
            gnm_prob = dist_term * K_term
            
            # Mask out existing edges
            gnm_prob[adjacency_matrix == 1] = 0
            
            # Explicitly prevent self-loops by zeroing the diagonal. (should be unnecessary) + issues with dimensions. 
            # gnm_prob.fill_diagonal_(0)
            
            ####
            # Get top candidates
            flat_probs = gnm_prob.flatten()
            top_k_indices = torch.topk(flat_probs, k=dynamic_config['candidate_pool_size']).indices
            candidate_edges = [(idx // n_nodes, idx % n_nodes) for idx in top_k_indices]
            
            # Dynamic evaluation and adding of edges
            candidate_scores = []
            for u, v in candidate_edges:

                # Skip self-loops 
                if u == v:
                    continue 
                
                # Create a temporary network
                temp_adj = adjacency_matrix.clone()
                temp_adj[0, u, v] = temp_adj[0, v, u] = 1


                # Configure the ESN evaluation based on the chosen metric
                eval_hparams = dynamic_config.get('fast_esn_eval', {}).copy()
                if fitness_metric == 'branching_ratio':
                    eval_hparams['calculate_criticality'] = True
                elif fitness_metric in ['avg_transfer_entropy', 'avg_active_info', 'info_balance']:
                    eval_hparams['calculate_info_dynamics'] = True
                
                
                # Run fast ESN evaluation
                esn_result = self.esn_evaluator.evaluate_single_subject(
                    subject_idx=0,
                    connectome=temp_adj[0,:,:].cpu().numpy().astype(np.float64),
                    hparams=eval_hparams # dynamic_config['fast_esn_eval']
                )
                
                # Get the score for the chosen metric
                dynamic_score = esn_result.get(fitness_metric, 0.0)
                
                
                # mc = esn_result.get("mc_mean", 0.0)
                
                # # Final MC score calculation
                
                # dynamic_score = mc
                
                # For criticality, the goal is a branching ratio of 1.0
                # So we transform the score: higher score means closer to 1.0
                if fitness_metric == 'branching_ratio':
                    dynamic_score = 1.0 - abs(1.0 - dynamic_score)
                
                # Normalize probabilities and MC for stable combination
                gnm_score = gnm_prob[0, u, v].item()
                
                # Combine scores (using delta as a weight here for simplicity): TODO
                delta = dynamic_config['dynamic_delta']
                final_score = (1 - delta) * gnm_score + delta * dynamic_score
                candidate_scores.append(final_score)
            
            # Edge selection
            best_candidate_idx = np.argmax(candidate_scores)
            best_u, best_v = candidate_edges[best_candidate_idx]
            # Print selected edge info, into the same line as previous print
            print(f"    Selected edge ({best_u}, {best_v}) with combined score {candidate_scores[best_candidate_idx]:.4f}")
            
            # Add the best edge permanently
            adjacency_matrix[0, best_u, best_v] = adjacency_matrix[0, best_v, best_u] = 1

        print("Dynamic generation complete.")
        return adjacency_matrix[0,:,:].to(torch.float32)  # Remove batch dimension again




##############


# Convenience functions using GNM library features
def generate_network_gnm(self,
                         target_connectome: np.ndarray,
                         distance_matrix: np.ndarray,
                         rule: str, 
                         eta: float, 
                         gamma: float, 
                         ) -> np.ndarray:
    """
    Quick function to generate a network using GNM library.
    """
    generator = GNMGenerator(self.device)
    
    # Convert to torch tensors
    target_tensor = torch.tensor(target_connectome, dtype=torch.float32)
    dist_tensor = torch.tensor(distance_matrix, dtype=torch.float32)
    
    # Create parameters
    params = GNMParameters(
        eta=eta, 
        gamma=gamma,
        generative_rule=generator.AVAILABLE_RULES[rule]()
    )
    
    # Generate network
    n_edges = int(target_tensor.sum().item() // 2)
    generated = generator.generate_network(
        n_nodes=target_connectome.shape[0],
        n_edges=n_edges,
        distance_matrix=dist_tensor,
        parameters=params
    )
    
    return generated.cpu().numpy()


# def fit_and_generate_batch(
#                           device: str, 
#                           connectomes: np.ndarray,
#                           distance_matrix: np.ndarray,
#                           n_realizations: int = 10,
#                           rule: str = "matching_index") -> Dict[str, Any]:
#     """
#     Fit parameters and generate multiple realizations for each connectome.
#     """
#     generator = GNMGenerator(device)
    
#     results = {
#         "fitted_parameters": [],
#         "generated_networks": [],
#         "evaluations": []
#     }
    
#     dist_tensor = torch.tensor(distance_matrix, dtype=torch.float32)
    
#     for i in range(connectomes.shape[2]):
#         print(f"Processing connectome {i+1}/{connectomes.shape[2]}")
        
#         target = torch.tensor(connectomes[:, :, i], dtype=torch.float32)
        
#         # Fit parameters
#         fit_result = generator.fit_parameters(
#             target_network=target,
#             distance_matrix=dist_tensor,
#             generative_rule_name=rule,
#             n_eta=15,
#             n_gamma=15,
#             num_simulations=50
#         )
        
#         results["fitted_parameters"].append(fit_result)
        
#         # Generate realizations with fitted parameters
#         params = GNMParameters(
#             eta=fit_result["best_eta"],
#             gamma=fit_result["best_gamma"],
#             generative_rule=generator.AVAILABLE_RULES[rule]()
#         )
        
#         realizations = []
#         evaluations = []
        
#         for j in range(n_realizations):
#             generated = generator.generate_network(
#                 n_nodes=target.shape[0],
#                 n_edges=fit_result["n_edges"],
#                 distance_matrix=dist_tensor,
#                 parameters=params
#             )
            
#             realizations.append(generated.cpu().numpy())
            
#             # Evaluate
#             eval_result = generator.evaluate_network(
#                 generated_network=generated,
#                 target_network=target,
#                 distance_matrix=dist_tensor
#             )
#             evaluations.append(eval_result)
        
#         results["generated_networks"].append(realizations)
#         results["evaluations"].append(evaluations)
    
#     return results