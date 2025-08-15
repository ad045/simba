"""
Modern GNM-based network generation for binary connectomes.
Uses Edward's GNM library with flexible wiring rules and clean, modern interfaces.
"""

import numpy as np
import torch
import warnings
from typing import Optional, Union, Dict, Any, List, Tuple, Literal
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

try:
    from gnm import fitting, generative_rules, evaluation, defaults
    GNM_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"GNM library not available: {e}")
    GNM_AVAILABLE = False


class WiringRule(Enum):
    """Available wiring rules for GNM generation."""
    MATCHING_INDEX = "matching_index"
    SPATIAL = "spatial"
    DEGREE_BASED = "degree_based"
    CLUSTERING_BASED = "clustering_based"
    COMMUNICABILITY = "communicability"


class DistanceRelationship(Enum):
    """Distance relationship types."""
    POWERLAW = "powerlaw"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


@dataclass
class GNMParameters:
    """Parameters for GNM generation."""
    eta: float = -2.0                                    # Distance parameter
    gamma: float = 0.3                                   # Homophily parameter
    lambda_param: float = 0.0                           # Additional parameter
    wiring_rule: WiringRule = WiringRule.MATCHING_INDEX
    distance_relationship: DistanceRelationship = DistanceRelationship.POWERLAW
    preferential_relationship: DistanceRelationship = DistanceRelationship.POWERLAW
    heterochronicity_relationship: DistanceRelationship = DistanceRelationship.POWERLAW
    num_simulations: int = 1
    device: str = "cpu"
    random_seed: Optional[int] = None


@dataclass
class NetworkProperties:
    """Properties of generated networks."""
    n_nodes: int
    n_edges: int
    density: float
    spectral_radius: float
    clustering_coefficient: float
    characteristic_path_length: Optional[float] = None


class GNMNetworkGenerator:
    """Modern GNM-based network generator with flexible wiring rules."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the GNM network generator."""
        if not GNM_AVAILABLE:
            raise ImportError("GNM library is required for this generator")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._wiring_rule_map = self._build_wiring_rule_map()
    
    def _build_wiring_rule_map(self) -> Dict[WiringRule, Any]:
        """Build mapping from enum to GNM wiring rule objects."""
        try:
            return {
                WiringRule.MATCHING_INDEX: generative_rules.MatchingIndex(),
                WiringRule.SPATIAL: generative_rules.Spatial() if hasattr(generative_rules, 'Spatial') else generative_rules.MatchingIndex(),
                WiringRule.DEGREE_BASED: generative_rules.DegreeProduct() if hasattr(generative_rules, 'DegreeProduct') else generative_rules.MatchingIndex(),
                WiringRule.CLUSTERING_BASED: generative_rules.ClusteringCoefficient() if hasattr(generative_rules, 'ClusteringCoefficient') else generative_rules.MatchingIndex(),
                WiringRule.COMMUNICABILITY: generative_rules.Communicability() if hasattr(generative_rules, 'Communicability') else generative_rules.MatchingIndex(),
            }
        except Exception as e:
            warnings.warn(f"Could not build all wiring rules: {e}. Using MatchingIndex as fallback.")
            return {rule: generative_rules.MatchingIndex() for rule in WiringRule}
    
    def generate_from_seed(self, 
                          seed_connectome: np.ndarray,
                          distance_matrix: np.ndarray,
                          parameters: GNMParameters) -> np.ndarray:
        """
        Generate a binary connectome from a seed network using GNM.
        
        Args:
            seed_connectome: Seed binary connectome (can be empty)
            distance_matrix: Distance matrix between nodes
            parameters: GNM parameters
            
        Returns:
            Generated binary connectome
        """
        # Prepare inputs
        binary_tensor = self._prepare_binary_input(seed_connectome)
        distance_tensor = self._prepare_distance_input(distance_matrix)
        target_edges = int(binary_tensor.sum().item() // 2) if binary_tensor.sum() > 0 else 100
        
        # Set random seed if provided
        if parameters.random_seed is not None:
            torch.manual_seed(parameters.random_seed)
            np.random.seed(parameters.random_seed)
        
        # Create GNM configuration
        sweep_config = self._create_sweep_config(
            parameters, target_edges, distance_tensor
        )
        
        # Run GNM generation
        try:
            experiments = fitting.perform_sweep(
                sweep_config=sweep_config,
                binary_evaluations=[],  # No evaluation needed for generation
                real_binary_matrices=binary_tensor,
                save_model=True,  # We need the model to extract the network
                save_run_history=False,
                verbose=False
            )
            
            if experiments and len(experiments) > 0:
                return self._extract_network_from_experiment(experiments[0])
            else:
                raise RuntimeError("No networks generated")
                
        except Exception as e:
            warnings.warn(f"GNM generation failed: {e}")
            return self._fallback_generation(seed_connectome, target_edges)
    
    def generate_matched_network(self,
                                target_connectome: np.ndarray,
                                distance_matrix: np.ndarray,
                                parameters: GNMParameters) -> np.ndarray:
        """
        Generate a network that matches the edge count of a target connectome.
        
        Args:
            target_connectome: Target connectome to match
            distance_matrix: Distance matrix
            parameters: GNM parameters
            
        Returns:
            Generated binary connectome with same edge count
        """
        # Calculate target edge count
        binary_target = self._prepare_binary_input(target_connectome).cpu().numpy()
        target_edges = int(np.sum(binary_target) // 2)
        
        # Create empty seed with same number of nodes
        seed = np.zeros_like(binary_target)
        
        # Update parameters for target edge count
        params_copy = GNMParameters(**parameters.__dict__)
        
        return self.generate_from_seed(seed, distance_matrix, params_copy)
    
    def fit_parameters_to_target(self,
                               target_connectome: np.ndarray,
                               distance_matrix: np.ndarray,
                               eta_range: Tuple[float, float] = (-5.0, 0.0),
                               gamma_range: Tuple[float, float] = (0.0, 1.0),
                               n_eta: int = 10,
                               n_gamma: int = 10,
                               wiring_rule: WiringRule = WiringRule.MATCHING_INDEX,
                               num_simulations: int = 50) -> Dict[str, Any]:
        """
        Fit GNM parameters to match a target connectome.
        
        Args:
            target_connectome: Target connectome to fit
            distance_matrix: Distance matrix
            eta_range: Range of eta values to search
            gamma_range: Range of gamma values to search
            n_eta: Number of eta values to test
            n_gamma: Number of gamma values to test
            wiring_rule: Wiring rule to use
            num_simulations: Number of simulations per parameter combination
            
        Returns:
            Dictionary with best parameters and fit results
        """
        # Prepare inputs
        binary_target = self._prepare_binary_input(target_connectome)
        distance_tensor = self._prepare_distance_input(distance_matrix)
        target_edges = int(binary_target.sum().item() // 2)
        
        # Parameter grids
        eta_values = torch.linspace(eta_range[0], eta_range[1], n_eta)
        gamma_values = torch.linspace(gamma_range[0], gamma_range[1], n_gamma)
        
        # Create parameter sweep
        binary_sweep_parameters = fitting.BinarySweepParameters(
            eta=eta_values,
            gamma=gamma_values,
            lambdah=torch.tensor([0.0]),
            distance_relationship_type=[DistanceRelationship.POWERLAW.value],
            preferential_relationship_type=[DistanceRelationship.POWERLAW.value],
            heterochronicity_relationship_type=[DistanceRelationship.POWERLAW.value],
            generative_rule=[self._wiring_rule_map[wiring_rule]],
            num_iterations=[target_edges],
        )
        
        sweep_config = fitting.SweepConfig(
            binary_sweep_parameters=binary_sweep_parameters,
            weighted_sweep_parameters=None,  # Binary only
            num_simulations=num_simulations,
            distance_matrix=[distance_tensor]
        )
        
        # Set up evaluation criteria
        criteria = [
            evaluation.DegreeKS(),
            evaluation.ClusteringKS(),
            evaluation.EdgeLengthKS(distance_tensor)
        ]
        energy_equation = evaluation.MaxCriteria(criteria)
        
        # Run parameter fitting
        try:
            experiments = fitting.perform_sweep(
                sweep_config=sweep_config,
                binary_evaluations=[energy_equation],
                real_binary_matrices=binary_target,
                save_model=False,
                save_run_history=False,
                verbose=True
            )
            
            # Find best parameters
            optimal_experiments, optimal_energies = fitting.optimise_evaluation(
                experiments=experiments,
                criterion=energy_equation,
            )
            
            if optimal_experiments:
                best_experiment = optimal_experiments[0]
                best_energy = optimal_energies[0]
                
                return {
                    "best_eta": float(best_experiment.run_config.binary_parameters.eta),
                    "best_gamma": float(best_experiment.run_config.binary_parameters.gamma),
                    "best_energy": float(best_energy),
                    "wiring_rule": wiring_rule,
                    "target_edges": target_edges,
                    "all_energies": [float(e) for e in optimal_energies],
                    "parameter_grid": {
                        "eta_range": eta_range,
                        "gamma_range": gamma_range,
                        "n_eta": n_eta,
                        "n_gamma": n_gamma
                    }
                }
            else:
                raise RuntimeError("No optimal parameters found")
                
        except Exception as e:
            warnings.warn(f"Parameter fitting failed: {e}")
            return {
                "best_eta": -2.0,
                "best_gamma": 0.3,
                "best_energy": float('inf'),
                "wiring_rule": wiring_rule,
                "target_edges": target_edges,
                "error": str(e)
            }
    
    def batch_generate(self,
                      target_connectomes: List[np.ndarray],
                      distance_matrix: np.ndarray,
                      parameters: GNMParameters,
                      n_realizations: int = 10) -> List[List[np.ndarray]]:
        """
        Generate multiple realizations for multiple target connectomes.
        
        Args:
            target_connectomes: List of target connectomes
            distance_matrix: Distance matrix
            parameters: GNM parameters
            n_realizations: Number of realizations per target
            
        Returns:
            List of lists: [target][realization] -> generated connectome
        """
        results = []
        
        for i, target in enumerate(target_connectomes):
            target_results = []
            
            for j in range(n_realizations):
                # Use different random seed for each realization
                params_copy = GNMParameters(**parameters.__dict__)
                if parameters.random_seed is not None:
                    params_copy.random_seed = parameters.random_seed + i * n_realizations + j
                
                try:
                    generated = self.generate_matched_network(target, distance_matrix, params_copy)
                    target_results.append(generated)
                except Exception as e:
                    warnings.warn(f"Failed to generate realization {j} for target {i}: {e}")
                    # Add empty array as placeholder
                    target_results.append(np.zeros_like(target))
            
            results.append(target_results)
        
        return results
    
    def _prepare_binary_input(self, connectome: np.ndarray) -> torch.Tensor:
        """Prepare binary connectome input for GNM."""
        if isinstance(connectome, torch.Tensor):
            connectome = connectome.cpu().numpy()
        
        # Ensure binary and symmetric
        binary = (connectome > 0).astype(np.float32)
        binary = (binary + binary.T) / 2.0
        np.fill_diagonal(binary, 0.0)
        binary = (binary > 0.5).astype(np.float32)
        
        return torch.tensor(binary, device=self.device)
    
    def _prepare_distance_input(self, distance_matrix: np.ndarray) -> torch.Tensor:
        """Prepare distance matrix input for GNM."""
        if isinstance(distance_matrix, torch.Tensor):
            return distance_matrix.to(self.device)
        
        return torch.tensor(distance_matrix, dtype=torch.float32, device=self.device)
    
    def _create_sweep_config(self, parameters: GNMParameters, target_edges: int, distance_tensor: torch.Tensor):
        """Create GNM sweep configuration."""
        binary_sweep_parameters = fitting.BinarySweepParameters(
            eta=torch.tensor([parameters.eta]),
            gamma=torch.tensor([parameters.gamma]),
            lambdah=torch.tensor([parameters.lambda_param]),
            distance_relationship_type=[parameters.distance_relationship.value],
            preferential_relationship_type=[parameters.preferential_relationship.value],
            heterochronicity_relationship_type=[parameters.heterochronicity_relationship.value],
            generative_rule=[self._wiring_rule_map[parameters.wiring_rule]],
            num_iterations=[target_edges],
        )
        
        return fitting.SweepConfig(
            binary_sweep_parameters=binary_sweep_parameters,
            weighted_sweep_parameters=None,  # Binary only
            num_simulations=parameters.num_simulations,
            distance_matrix=[distance_tensor]
        )
    
    def _extract_network_from_experiment(self, experiment) -> np.ndarray:
        """Extract binary network from GNM experiment."""
        try:
            if hasattr(experiment, 'model') and experiment.model is not None:
                # Try to get adjacency matrix from model
                if hasattr(experiment.model, 'get_adjacency_matrix'):
                    network = experiment.model.get_adjacency_matrix()
                elif hasattr(experiment.model, 'adjacency_matrix'):
                    network = experiment.model.adjacency_matrix
                else:
                    # Fallback: use the model's state
                    network = experiment.model
                
                if isinstance(network, torch.Tensor):
                    network = network.cpu().numpy()
                
                # Ensure binary and symmetric
                network = (network > 0).astype(np.int8)
                network = (network + network.T) // 2
                np.fill_diagonal(network, 0)
                
                return network
            else:
                raise AttributeError("No model found in experiment")
                
        except Exception as e:
            warnings.warn(f"Could not extract network from experiment: {e}")
            # Return empty network as fallback
            return np.zeros((90, 90), dtype=np.int8)  # Default size
    
    def _fallback_generation(self, seed_connectome: np.ndarray, target_edges: int) -> np.ndarray:
        """Fallback network generation when GNM fails."""
        n_nodes = seed_connectome.shape[0]
        
        # Create random network with target edge count
        network = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        
        # Add random edges
        edges_added = 0
        max_attempts = target_edges * 10
        attempts = 0
        
        while edges_added < target_edges and attempts < max_attempts:
            i, j = np.random.randint(0, n_nodes, 2)
            if i != j and network[i, j] == 0:
                network[i, j] = network[j, i] = 1
                edges_added += 1
            attempts += 1
        
        return network


class NetworkAnalyzer:
    """Analyze properties of generated networks."""
    
    @staticmethod
    def compute_properties(network: np.ndarray, distance_matrix: Optional[np.ndarray] = None) -> NetworkProperties:
        """Compute basic network properties."""
        n_nodes = network.shape[0]
        n_edges = int(np.sum(network > 0) // 2)
        density = n_edges / (n_nodes * (n_nodes - 1) / 2)
        
        # Spectral radius
        try:
            eigenvalues = np.linalg.eigvals(network.astype(float))
            spectral_radius = np.max(np.abs(eigenvalues))
        except:
            spectral_radius = 0.0
        
        # Clustering coefficient
        try:
            import networkx as nx
            G = nx.from_numpy_array(network)
            clustering_coefficient = nx.average_clustering(G)
        except:
            clustering_coefficient = 0.0
        
        # Characteristic path length (optional, expensive)
        char_path_length = None
        try:
            if n_edges > 0:
                import networkx as nx
                G = nx.from_numpy_array(network)
                if nx.is_connected(G):
                    char_path_length = nx.average_shortest_path_length(G)
        except:
            pass
        
        return NetworkProperties(
            n_nodes=n_nodes,
            n_edges=n_edges,
            density=density,
            spectral_radius=spectral_radius,
            clustering_coefficient=clustering_coefficient,
            characteristic_path_length=char_path_length
        )
    
    @staticmethod
    def compare_networks(network1: np.ndarray, network2: np.ndarray) -> Dict[str, float]:
        """Compare two networks using various metrics."""
        props1 = NetworkAnalyzer.compute_properties(network1)
        props2 = NetworkAnalyzer.compute_properties(network2)
        
        return {
            "edge_count_diff": abs(props1.n_edges - props2.n_edges),
            "density_diff": abs(props1.density - props2.density),
            "clustering_diff": abs(props1.clustering_coefficient - props2.clustering_coefficient),
            "spectral_radius_diff": abs(props1.spectral_radius - props2.spectral_radius),
            "edge_overlap": np.sum((network1 > 0) & (network2 > 0)) / np.sum((network1 > 0) | (network2 > 0)) if np.any(network1 > 0) or np.any(network2 > 0) else 0.0
        }


# Convenience functions for common use cases
def generate_gnm_network(target_connectome: np.ndarray,
                        distance_matrix: np.ndarray,
                        wiring_rule: WiringRule = WiringRule.MATCHING_INDEX,
                        eta: float = -2.0,
                        gamma: float = 0.3,
                        random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a single GNM network matching a target connectome.
    
    Convenience function for simple use cases.
    """
    generator = GNMNetworkGenerator()
    parameters = GNMParameters(
        eta=eta,
        gamma=gamma,
        wiring_rule=wiring_rule,
        random_seed=random_seed
    )
    
    return generator.generate_matched_network(target_connectome, distance_matrix, parameters)


def fit_gnm_to_connectome(target_connectome: np.ndarray,
                         distance_matrix: np.ndarray,
                         wiring_rule: WiringRule = WiringRule.MATCHING_INDEX) -> Dict[str, Any]:
    """
    Fit GNM parameters to a target connectome.
    
    Convenience function for parameter fitting.
    """
    generator = GNMNetworkGenerator()
    
    return generator.fit_parameters_to_target(
        target_connectome=target_connectome,
        distance_matrix=distance_matrix,
        wiring_rule=wiring_rule
    )


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Testing GNM Network Generator")
    
    if not GNM_AVAILABLE:
        print("❌ GNM library not available - tests skipped")
        exit()
    
    # Create test data
    n_nodes = 90
    
    # Random target connectome
    target = np.random.rand(n_nodes, n_nodes)
    target = (target > 0.9).astype(int)
    target = (target + target.T) // 2
    np.fill_diagonal(target, 0)
    
    # Random distance matrix
    coords = np.random.rand(n_nodes, 3)
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    print(f"Target network: {n_nodes} nodes, {np.sum(target)//2} edges")
    
    # Test different wiring rules
    for wiring_rule in [WiringRule.MATCHING_INDEX, WiringRule.SPATIAL]:
        try:
            print(f"\n🔧 Testing {wiring_rule.value}...")
            
            generated = generate_gnm_network(
                target_connectome=target,
                distance_matrix=dist_matrix,
                wiring_rule=wiring_rule,
                eta=-2.0,
                gamma=0.3,
                random_seed=42
            )
            
            # Analyze results
            props = NetworkAnalyzer.compute_properties(generated)
            comparison = NetworkAnalyzer.compare_networks(target, generated)
            
            print(f"✅ Generated network: {props.n_edges} edges, density={props.density:.3f}")
            print(f"   Edge overlap: {comparison['edge_overlap']:.3f}")
            print(f"   Density difference: {comparison['density_diff']:.3f}")
            
        except Exception as e:
            print(f"❌ {wiring_rule.value} failed: {e}")
    
    # Test parameter fitting
    try:
        print(f"\n🎯 Testing parameter fitting...")
        
        fit_results = fit_gnm_to_connectome(
            target_connectome=target,
            distance_matrix=dist_matrix,
            wiring_rule=WiringRule.MATCHING_INDEX
        )
        
        print(f"✅ Best parameters: eta={fit_results['best_eta']:.3f}, gamma={fit_results['best_gamma']:.3f}")
        print(f"   Best energy: {fit_results['best_energy']:.3f}")
        
    except Exception as e:
        print(f"❌ Parameter fitting failed: {e}")
    
    print("\n🎉 GNM Network Generator testing completed!")