"""
Simulated Annealing Optimization for Generative Network Models

This script:
1. Selects a 4x4 grid of parameter combinations
2. Finds the 10 closest existing networks for each grid point
3. Applies simulated annealing to optimize them
4. Saves optimized networks and creates a comparison CSV
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from tqdm import tqdm
import json
from scipy.spatial.distance import cdist
from copy import deepcopy

# Import your existing modules
from src.config.path import PathConfig
from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence


class SimulatedAnnealingOptimizer:
    """Optimize network parameters using simulated annealing."""
    
    def __init__(
        self,
        empirical_networks: np.ndarray,
        distance_matrix: np.ndarray,
        initial_temp: float = 1.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.95,
        max_iterations: int = 100,
        perturbation_scale: float = 0.1
    ):
        """
        Initialize the optimizer.
        
        Args:
            empirical_networks: Target empirical networks (N, H, W)
            distance_matrix: Euclidean distance matrix for network generation
            initial_temp: Starting temperature for SA
            final_temp: Ending temperature
            cooling_rate: Temperature decay rate (0 < rate < 1)
            max_iterations: Maximum iterations per temperature
            perturbation_scale: Scale of parameter perturbations
        """
        self.empirical_networks = torch.tensor(empirical_networks, dtype=torch.float32)
        self.distance_matrix = distance_matrix
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.perturbation_scale = perturbation_scale
        
        # Initialize evaluator
        self.evaluator = PortraitDivergence()
        
    def evaluate_network(self, network: np.ndarray) -> float:
        """Evaluate a network against empirical targets using portrait divergence."""
        network_tensor = torch.tensor(network, dtype=torch.float32).unsqueeze(0)
        metrics = self.evaluator(network_tensor, self.empirical_networks)
        # Return average divergence across all empirical networks
        return np.mean([v for v in metrics.values()])
    
    def perturb_parameters(
        self, 
        eta: float, 
        gamma: float,
        eta_range: Tuple[float, float],
        gamma_range: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Perturb parameters with bounds checking."""
        # Add Gaussian noise
        eta_new = eta + np.random.normal(0, self.perturbation_scale)
        gamma_new = gamma + np.random.normal(0, self.perturbation_scale)
        
        # Clip to valid ranges
        eta_new = np.clip(eta_new, eta_range[0], eta_range[1])
        gamma_new = np.clip(gamma_new, gamma_range[0], gamma_range[1])
        
        return eta_new, gamma_new
    
    def generate_network(
        self, 
        eta: float, 
        gamma: float, 
        seed_network: np.ndarray
    ) -> np.ndarray:
        """
        Generate a network with given parameters starting from seed.
        This is a simplified version - you'll need to adapt based on your GNM implementation.
        """
        # TODO: Replace with your actual network generation logic
        # For now, this is a placeholder that slightly modifies the seed
        network = seed_network.copy()
        
        # Simple perturbation based on parameters (replace with actual GNM)
        perturbation = np.random.randn(*network.shape) * 0.01
        network = network + perturbation
        network = np.clip(network, 0, 1)
        
        # Ensure same density as seed
        target_edges = int(seed_network.sum())
        network = self._threshold_to_density(network, target_edges)
        
        return network
    
    def _threshold_to_density(self, network: np.ndarray, target_edges: int) -> np.ndarray:
        """Threshold network to maintain target edge density."""
        flat = network.flatten()
        threshold = np.partition(flat, -target_edges)[-target_edges]
        binary = (network >= threshold).astype(float)
        return binary
    
    def optimize(
        self,
        initial_eta: float,
        initial_gamma: float,
        seed_network: np.ndarray,
        eta_range: Tuple[float, float] = (-8.0, 3.0),
        gamma_range: Tuple[float, float] = (-0.1, 1.0)
    ) -> Dict:
        """
        Run simulated annealing optimization.
        
        Returns:
            Dict with best_eta, best_gamma, best_network, best_energy, history
        """
        # Initialize
        current_eta = initial_eta
        current_gamma = initial_gamma
        current_network = seed_network.copy()
        current_energy = self.evaluate_network(current_network)
        
        best_eta = current_eta
        best_gamma = current_gamma
        best_network = current_network.copy()
        best_energy = current_energy
        
        temperature = self.initial_temp
        history = []
        
        print(f"Initial energy: {current_energy:.6f}")
        
        # SA loop
        iteration = 0
        while temperature > self.final_temp:
            for _ in range(self.max_iterations):
                # Perturb parameters
                new_eta, new_gamma = self.perturb_parameters(
                    current_eta, current_gamma, eta_range, gamma_range
                )
                
                # Generate new network
                new_network = self.generate_network(new_eta, new_gamma, seed_network)
                new_energy = self.evaluate_network(new_network)
                
                # Acceptance criterion
                delta_energy = new_energy - current_energy
                if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                    current_eta = new_eta
                    current_gamma = new_gamma
                    current_network = new_network
                    current_energy = new_energy
                    
                    # Update best
                    if current_energy < best_energy:
                        best_eta = current_eta
                        best_gamma = current_gamma
                        best_network = current_network.copy()
                        best_energy = current_energy
                        print(f"  New best: η={best_eta:.4f}, γ={best_gamma:.4f}, E={best_energy:.6f}")
                
                history.append({
                    'iteration': iteration,
                    'temperature': temperature,
                    'eta': current_eta,
                    'gamma': current_gamma,
                    'energy': current_energy,
                    'best_energy': best_energy
                })
                iteration += 1
            
            # Cool down
            temperature *= self.cooling_rate
            print(f"Temperature: {temperature:.6f}, Best energy: {best_energy:.6f}")
        
        return {
            'best_eta': best_eta,
            'best_gamma': best_gamma,
            'best_network': best_network,
            'best_energy': best_energy,
            'history': pd.DataFrame(history)
        }


def create_parameter_grid(
    eta_range: Tuple[float, float],
    gamma_range: Tuple[float, float],
    n_points: int = 4
) -> List[Tuple[float, float]]:
    """Create a grid of parameter combinations."""
    eta_vals = np.linspace(eta_range[0], eta_range[1], n_points)
    gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_points)
    
    grid = []
    for eta in eta_vals:
        for gamma in gamma_vals:
            grid.append((eta, gamma))
    
    return grid


def find_closest_networks(
    target_eta: float,
    target_gamma: float,
    csv_path: Path,
    network_dir: Path,
    n_closest: int = 10
) -> List[Dict]:
    """
    Find the n closest existing networks to target parameters.
    
    Returns:
        List of dicts with 'eta', 'gamma', 'id', 'filepath', 'distance'
    """
    df = pd.read_csv(csv_path)
    
    # Calculate Euclidean distance in parameter space
    params = df[['eta', 'gamma']].values
    target = np.array([[target_eta, target_gamma]])
    distances = cdist(target, params, metric='euclidean').flatten()
    
    # Get indices of closest networks
    closest_indices = np.argsort(distances)[:n_closest]
    
    results = []
    for idx in closest_indices:
        row = df.iloc[idx]
        eta, gamma, net_id = row['eta'], row['gamma'], int(row['id'])
        
        # Construct filename
        filename = f"net_eta{eta}_gamma{gamma}_ruleMatchingIndex_id{net_id:03d}.npy"
        filepath = network_dir / filename
        
        if filepath.exists():
            results.append({
                'eta': eta,
                'gamma': gamma,
                'id': net_id,
                'filepath': filepath,
                'distance': distances[idx]
            })
    
    return results


def main():
    """Main execution function."""
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Paths
    dataset_name = "hcp_schaefer_100_dataset"
    source_experiment = "05_second_big_overnight_run_10201"
    target_experiment = "05_1_simulated_annealing" # 10_simulated_annealing_optimized"
    
    path_config = PathConfig(
        dataset_name=dataset_name,
        experiment_name=source_experiment
    )
    
    # Input paths
    source_csv = path_config.output_experiment_dir / f"all_metrics_for_{source_experiment}.csv"
    source_networks = path_config.output_experiment_dir / "generated_networks"
    
    # Load empirical data
    empirical_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/{dataset_name}/01_connectomes/00_connectomes_density10.npy")
    empirical_networks = np.load(empirical_path)
    
    distance_matrix_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/{dataset_name}/02_distance_matrices/distance_matrix_100.npy")
    distance_matrix = np.load(distance_matrix_path)
    
    # Output paths
    output_dir = path_config.output_gnm_dir / path_config.dataset_name # output_experiment_dir # output_base_dir / target_experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    output_networks = output_dir / "generated_networks"
    output_networks.mkdir(exist_ok=True)
    output_csv = output_dir / f"all_metrics_for_{target_experiment}.csv"
    
    # ========================================================================
    # CREATE PARAMETER GRID
    # ========================================================================
    
    print("Creating 4x4 parameter grid...")
    eta_range = (-8.0, 3.0)
    gamma_range = (-0.1, 1.0)
    param_grid = create_parameter_grid(eta_range, gamma_range, n_points=4)
    print(f"Created {len(param_grid)} parameter combinations")
    
    # ========================================================================
    # INITIALIZE OPTIMIZER
    # ========================================================================
    
    optimizer = SimulatedAnnealingOptimizer(
        empirical_networks=empirical_networks,
        distance_matrix=distance_matrix,
        initial_temp=1.0,
        final_temp=0.01,
        cooling_rate=0.95,
        max_iterations=50,
        perturbation_scale=0.1
    )
    
    # ========================================================================
    # OPTIMIZE NETWORKS
    # ========================================================================
    
    results = []
    network_counter = 0
    
    for target_eta, target_gamma in tqdm(param_grid, desc="Optimizing grid points"):
        print(f"\n{'='*60}")
        print(f"Target: η={target_eta:.4f}, γ={target_gamma:.4f}")
        print(f"{'='*60}")
        
        # Find closest existing networks
        closest = find_closest_networks(
            target_eta, target_gamma, source_csv, source_networks, n_closest=10
        )
        
        print(f"Found {len(closest)} closest networks")
        
        # Optimize each seed network
        for i, seed_info in enumerate(closest[:10]):  # Limit to 10
            print(f"\nOptimizing seed {i+1}/10 (η={seed_info['eta']:.4f}, γ={seed_info['gamma']:.4f})")
            
            # Load seed network
            seed_network = np.load(seed_info['filepath'])
            
            # Run optimization
            result = optimizer.optimize(
                initial_eta=seed_info['eta'],
                initial_gamma=seed_info['gamma'],
                seed_network=seed_network,
                eta_range=eta_range,
                gamma_range=gamma_range
            )
            
            # Save optimized network
            output_filename = f"net_eta{result['best_eta']}_gamma{result['best_gamma']}_ruleMatchingIndex_id{network_counter:03d}.npy"
            output_path = output_networks / output_filename
            np.save(output_path, result['best_network'])
            
            # Record results
            results.append({
                'eta': result['best_eta'],
                'gamma': result['best_gamma'],
                'id': network_counter,
                'original_eta': seed_info['eta'],
                'original_gamma': seed_info['gamma'],
                'distance_relationship_type': 'powerlaw',
                'preferential_relationship_type': 'powerlaw',
                'generative_rule': 'MatchingIndex',
                'num_iterations': 495,
                'energy': result['best_energy'],
                'network_index': network_counter
            })
            
            network_counter += 1
    
    # ========================================================================
    # SAVE RESULTS CSV
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    print(f"✅ Saved {len(results_df)} optimized networks to {output_dir}")
    print(f"✅ CSV saved to {output_csv}")
    
    # Save optimization history
    history_dir = output_dir / "optimization_history"
    history_dir.mkdir(exist_ok=True)
    
    print("\n✅ Optimization complete!")


if __name__ == "__main__":
    main()