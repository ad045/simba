"""
GNM-ESN Energy Landscape Pipeline
Generates a grid of GNMs and evaluates memory capacity at each point.
Creates visualization similar to the reference image.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import warnings

# Import your existing modules
from config import ConfigManager, GNMConfig, get_gnm_quick_test_config
from gnm_network_generator import GNMGenerator, GNMParameters
from data_loader import DataLoader
from esn_evaluation import ESNEvaluator
from visualization import PipelineVisualizer

# GNM library imports
from gnm import generative_rules, evaluation, fitting


class GNMESNLandscapePipeline:
    """Pipeline for generating GNM-ESN energy landscapes."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = torch.device(config.gnm.device)
        
        # Initialize components
        self.data_loader = DataLoader(config) if not config.data.use_gnm_defaults else None
        self.esn_evaluator = ESNEvaluator(config, self.data_loader) if self.data_loader else None
        self.gnm_generator = GNMGenerator(device=config.gnm.device)
        self.visualizer = PipelineVisualizer()
        
        # Ensure output directories exist
        self.config.paths.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_gnm_esn_landscape(self,
                                  eta_range: Tuple[float, float] = (-8, 0),
                                  gamma_range: Tuple[float, float] = (0.1, 8),
                                  n_eta: int = 20,
                                  n_gamma: int = 20,
                                  density: int = 14,
                                  n_subjects: int = 5,
                                  generative_rule: str = "matching_index",
                                  esn_spectral_radius: float = 0.99,
                                  esn_input_length: int = 2000,
                                  experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate GNM-ESN landscape by creating GNMs on a grid and evaluating their memory capacity.
        
        Args:
            eta_range: Range of eta values (log-space parameter)
            gamma_range: Range of gamma values  
            n_eta: Number of eta points
            n_gamma: Number of gamma points
            density: Network density percentage
            n_subjects: Number of subjects to average over
            generative_rule: GNM generative rule to use
            esn_spectral_radius: ESN spectral radius
            esn_input_length: ESN input sequence length
            experiment_name: Custom experiment name
            
        Returns:
            Dictionary with landscape data and results
        """
        print("=" * 60)
        print("GNM-ESN ENERGY LANDSCAPE GENERATION")
        print("=" * 60)
        
        # Set experiment name
        if not experiment_name:
            experiment_name = f"gnm_esn_landscape_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        exp_dir = self.config.paths.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        if self.config.data.use_gnm_defaults:
            print("Using GNM default data...")
            data = self.config.load_gnm_defaults()
            distance_matrix = data["distance_matrix"]
            target_network = data["binary_network"]
            n_nodes = target_network.shape[0]
            n_edges = int(target_network.sum().item() // 2)
        else:
            print(f"Loading connectome data for density {density}%...")
            distance_matrix = torch.tensor(
                self.data_loader.load_distance_matrix(), 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Load binary connectomes for structure
            binary_connectomes = self.data_loader.load_binary_connectomes()
            if density not in binary_connectomes:
                raise ValueError(f"Density {density}% not available")
            
            target_network = torch.tensor(
                binary_connectomes[density][:, :, 0],
                dtype=torch.float32,
                device=self.device
            )
            n_nodes = target_network.shape[0]
            n_edges = int(target_network.sum().item() // 2)
        
        print(f"Network properties: {n_nodes} nodes, {n_edges} edges")
        
        # Create parameter grid
        eta_values = np.linspace(eta_range[0], eta_range[1], n_eta)
        gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
        
        print(f"Parameter grid: {n_eta} x {n_gamma} = {n_eta * n_gamma} points")
        print(f"Eta range: {eta_range}")
        print(f"Gamma range: {gamma_range}")
        
        # Get generative rule
        if generative_rule == "matching_index":
            rule = generative_rules.MatchingIndex()
        elif generative_rule == "neighbors":
            rule = generative_rules.Neighbors()
        elif generative_rule == "degree_product":
            rule = generative_rules.DegreeProduct()
        else:
            rule = generative_rules.MatchingIndex()
        
        # Store results
        landscape_data = []
        empirical_fits = []
        
        # Also evaluate empirical networks if available
        if not self.config.data.use_gnm_defaults:
            print("\nEvaluating empirical networks...")
            weighted_by_density = self.data_loader.load_weighted_by_density()
            if density in weighted_by_density:
                empirical_connectomes = weighted_by_density[density]
                
                for subj_idx in range(min(n_subjects, empirical_connectomes.shape[2])):
                    print(f"  Subject {subj_idx + 1}/{min(n_subjects, empirical_connectomes.shape[2])}")
                    empirical_conn = empirical_connectomes[:, :, subj_idx]
                    
                    # Evaluate memory capacity
                    mc_result = self.esn_evaluator.evaluate_single_subject(
                        subject_idx=subj_idx,
                        connectome=empirical_conn,
                        hparams={
                            "spectral_radius": esn_spectral_radius,
                            "input_length": esn_input_length,
                            "density_percent": density
                        }
                    )
                    
                    # Find best-fitting GNM parameters for this empirical network
                    binary_empirical = (empirical_conn > 0).astype(np.float32)
                    binary_empirical_torch = torch.tensor(binary_empirical, dtype=torch.float32, device=self.device)
                    
                    best_fit = self._find_best_gnm_fit(
                        target_network=binary_empirical_torch,
                        distance_matrix=distance_matrix,
                        rule=rule,
                        eta_range=eta_range,
                        gamma_range=gamma_range
                    )
                    
                    empirical_fits.append({
                        "subject": subj_idx,
                        "eta": best_fit["eta"],
                        "gamma": best_fit["gamma"],
                        "energy": best_fit["energy"],
                        "mc": mc_result.get("mc_mean", 0)
                    })
        
        # Generate and evaluate GNMs on grid
        print(f"\nGenerating GNM grid ({n_eta}x{n_gamma})...")
        total_points = n_eta * n_gamma
        completed = 0
        
        for i, eta in enumerate(eta_values):
            for j, gamma in enumerate(gamma_values):
                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{total_points} ({100*completed/total_points:.1f}%)")
                
                # Create GNM parameters
                params = GNMParameters(
                    eta=eta,
                    gamma=gamma,
                    generative_rule=rule
                )
                
                # Generate multiple network realizations and average MC
                mc_values = []
                energy_values = []
                
                for realization in range(min(3, n_subjects)):  # Generate 3 realizations per point
                    try:
                        # Generate synthetic network
                        synthetic_network = self.gnm_generator.generate_network(
                            n_nodes=n_nodes,
                            n_edges=n_edges,
                            distance_matrix=distance_matrix,
                            parameters=params
                        )
                        
                        # Convert to numpy for ESN evaluation
                        synthetic_np = synthetic_network.cpu().numpy()
                        
                        # Evaluate memory capacity
                        mc_result = self.esn_evaluator.evaluate_single_subject(
                            subject_idx=0,
                            connectome=synthetic_np,
                            hparams={
                                "spectral_radius": esn_spectral_radius,
                                "input_length": esn_input_length,
                                "density_percent": density
                            }
                        )
                        
                        mc_values.append(mc_result.get("mc_mean", 0))
                        
                        # Calculate GNM energy if target network is available
                        if target_network is not None:
                            eval_result = self.gnm_generator.evaluate_network(
                                generated_network=synthetic_network,
                                target_network=target_network,
                                distance_matrix=distance_matrix,
                                metrics=["degree_ks", "clustering_ks"]
                            )
                            energy = max(eval_result.get("degree_ks", 0), 
                                       eval_result.get("clustering_ks", 0))
                            energy_values.append(energy)
                        
                    except Exception as e:
                        print(f"    Warning: Failed at eta={eta:.3f}, gamma={gamma:.3f}: {e}")
                        continue
                
                if mc_values:
                    landscape_data.append({
                        "eta": eta,
                        "gamma": gamma,
                        "mc_mean": np.mean(mc_values),
                        "mc_std": np.std(mc_values) if len(mc_values) > 1 else 0,
                        "gnm_energy": np.mean(energy_values) if energy_values else None,
                        "n_realizations": len(mc_values)
                    })
        
        print(f"\nGenerated {len(landscape_data)} valid points")
        
        # Save results
        results = {
            "experiment_name": experiment_name,
            "parameters": {
                "eta_range": eta_range,
                "gamma_range": gamma_range,
                "n_eta": n_eta,
                "n_gamma": n_gamma,
                "density": density,
                "n_subjects": n_subjects,
                "generative_rule": generative_rule,
                "esn_spectral_radius": esn_spectral_radius,
                "esn_input_length": esn_input_length
            },
            "landscape_data": landscape_data,
            "empirical_fits": empirical_fits
        }
        
        # Save to JSON
        results_file = exp_dir / "landscape_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save to CSV for easier analysis
        df = pd.DataFrame(landscape_data)
        df.to_csv(exp_dir / "landscape_data.csv", index=False)
        
        if empirical_fits:
            df_empirical = pd.DataFrame(empirical_fits)
            df_empirical.to_csv(exp_dir / "empirical_fits.csv", index=False)
        
        print(f"\nResults saved to: {exp_dir}")
        
        return results
    
    def _find_best_gnm_fit(self, 
                          target_network: torch.Tensor,
                          distance_matrix: torch.Tensor,
                          rule: Any,
                          eta_range: Tuple[float, float],
                          gamma_range: Tuple[float, float],
                          n_samples: int = 20) -> Dict[str, float]:
        """
        Find best-fitting GNM parameters for a target network.
        
        Args:
            target_network: Target binary network
            distance_matrix: Distance matrix
            rule: Generative rule
            eta_range: Range for eta parameter
            gamma_range: Range for gamma parameter
            n_samples: Number of samples to test
            
        Returns:
            Dictionary with best eta, gamma, and energy
        """
        best_eta = None
        best_gamma = None
        best_energy = float('inf')
        
        # Sample parameter space
        eta_samples = np.random.uniform(eta_range[0], eta_range[1], n_samples)
        gamma_samples = np.random.uniform(gamma_range[0], gamma_range[1], n_samples)
        
        n_edges = int(target_network.sum().item() // 2)
        
        for eta, gamma in zip(eta_samples, gamma_samples):
            try:
                params = GNMParameters(eta=eta, gamma=gamma, generative_rule=rule)
                
                # Generate synthetic network
                synthetic = self.gnm_generator.generate_network(
                    n_nodes=target_network.shape[0],
                    n_edges=n_edges,
                    distance_matrix=distance_matrix,
                    parameters=params
                )
                
                # Evaluate fit
                eval_result = self.gnm_generator.evaluate_network(
                    generated_network=synthetic,
                    target_network=target_network,
                    distance_matrix=distance_matrix,
                    metrics=["degree_ks", "clustering_ks"]
                )
                
                energy = max(eval_result.get("degree_ks", 1), 
                           eval_result.get("clustering_ks", 1))
                
                if energy < best_energy:
                    best_energy = energy
                    best_eta = eta
                    best_gamma = gamma
                    
            except:
                continue
        
        return {"eta": best_eta, "gamma": best_gamma, "energy": best_energy}
    
    def visualize_landscape(self, 
                           results_file: Path,
                           visualization_type: str = "heatmap",
                           show_empirical: bool = True,
                           save_path: Optional[Path] = None) -> plt.Figure:
        """
        Visualize the GNM-ESN landscape.
        
        Args:
            results_file: Path to landscape results JSON
            visualization_type: Type of visualization ("heatmap", "contour", "3d")
            show_empirical: Whether to show empirical network fits
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        landscape_data = results["landscape_data"]
        empirical_fits = results.get("empirical_fits", [])
        
        # Convert to arrays
        df = pd.DataFrame(landscape_data)
        
        # Create meshgrid for interpolation
        eta_unique = sorted(df['eta'].unique())
        gamma_unique = sorted(df['gamma'].unique())
        eta_grid, gamma_grid = np.meshgrid(eta_unique, gamma_unique)
        
        # Reshape MC values to grid
        mc_grid = np.zeros_like(eta_grid)
        for _, row in df.iterrows():
            i = eta_unique.index(row['eta'])
            j = gamma_unique.index(row['gamma'])
            mc_grid[j, i] = row['mc_mean']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if visualization_type == "heatmap":
            # Create heatmap
            im = ax.imshow(mc_grid, 
                          extent=[min(eta_unique), max(eta_unique), 
                                 min(gamma_unique), max(gamma_unique)],
                          origin='lower',
                          aspect='auto',
                          cmap='hot_r',  # Reversed so high MC is "hot"
                          interpolation='bilinear')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Memory Capacity', rotation=270, labelpad=20)
            
        elif visualization_type == "contour":
            # Create contour plot
            levels = 15
            CS = ax.contourf(eta_grid, gamma_grid, mc_grid, levels=levels, cmap='hot_r')
            ax.contour(eta_grid, gamma_grid, mc_grid, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
            
            cbar = plt.colorbar(CS, ax=ax)
            cbar.set_label('Memory Capacity', rotation=270, labelpad=20)
        
        # Show empirical fits if available
        if show_empirical and empirical_fits:
            df_empirical = pd.DataFrame(empirical_fits)
            ax.scatter(df_empirical['eta'], df_empirical['gamma'],
                      s=100, c='lime', edgecolors='black', linewidths=2,
                      marker='*', label='Empirical Networks', zorder=10)
            
            # Add annotation
            ax.annotate('Best fitting gammas and etas\nof the empirical connectomes',
                       xy=(df_empirical['eta'].mean(), df_empirical['gamma'].mean()),
                       xytext=(-6, 3),
                       fontsize=10,
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            
            ax.legend(loc='upper right')
        
        # Labels and title
        ax.set_xlabel('η (eta)', fontsize=12)
        ax.set_ylabel('γ (gamma)', fontsize=12)
        ax.set_title('GNM-ESN Energy Landscape\nMemory Capacity across Parameter Space', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return fig


def main():
    """Main entry point for GNM-ESN landscape generation."""
    parser = argparse.ArgumentParser(description="Generate GNM-ESN Energy Landscape")
    
    parser.add_argument("--eta-min", type=float, default=-8,
                       help="Minimum eta value")
    parser.add_argument("--eta-max", type=float, default=0,
                       help="Maximum eta value")
    parser.add_argument("--gamma-min", type=float, default=0.1,
                       help="Minimum gamma value")
    parser.add_argument("--gamma-max", type=float, default=8,
                       help="Maximum gamma value")
    parser.add_argument("--n-eta", type=int, default=20,
                       help="Number of eta points")
    parser.add_argument("--n-gamma", type=int, default=20,
                       help="Number of gamma points")
    parser.add_argument("--density", type=int, default=14,
                       help="Network density percentage")
    parser.add_argument("--n-subjects", type=int, default=5,
                       help="Number of subjects to average")
    parser.add_argument("--rule", default="matching_index",
                       help="Generative rule to use")
    parser.add_argument("--esn-spectral-radius", type=float, default=0.99,
                       help="ESN spectral radius")
    parser.add_argument("--esn-input-length", type=int, default=2000,
                       help="ESN input sequence length")
    parser.add_argument("--experiment-name", help="Custom experiment name")
    parser.add_argument("--use-defaults", action="store_true",
                       help="Use GNM default data")
    parser.add_argument("--visualize-only", help="Path to existing results to visualize")
    
    args = parser.parse_args()
    
    # Configure
    config = ConfigManager()
    config.data.use_gnm_defaults = args.use_defaults
    
    # Initialize pipeline
    pipeline = GNMESNLandscapePipeline(config)
    
    if args.visualize_only:
        # Just visualize existing results
        results_file = Path(args.visualize_only)
        if not results_file.exists():
            print(f"Results file not found: {results_file}")
            sys.exit(1)
        
        exp_dir = results_file.parent
        save_path = exp_dir / "landscape_visualization.png"
        
        pipeline.visualize_landscape(
            results_file=results_file,
            visualization_type="heatmap",
            show_empirical=True,
            save_path=save_path
        )
    else:
        # Generate landscape
        results = pipeline.generate_gnm_esn_landscape(
            eta_range=(args.eta_min, args.eta_max),
            gamma_range=(args.gamma_min, args.gamma_max),
            n_eta=args.n_eta,
            n_gamma=args.n_gamma,
            density=args.density,
            n_subjects=args.n_subjects,
            generative_rule=args.rule,
            esn_spectral_radius=args.esn_spectral_radius,
            esn_input_length=args.esn_input_length,
            experiment_name=args.experiment_name
        )
        
        # Visualize results
        exp_dir = Path(config.paths.output_dir) / results["experiment_name"]
        results_file = exp_dir / "landscape_results.json"
        save_path = exp_dir / "landscape_visualization.png"
        
        # pipeline.visualize_landscape(
        #     results_file=results_file,
        #     visualization_type="heatmap",
        #     show_empirical=True,
        #     save_path=save_path
        # )
        
        print("\nPipeline completed successfully!")
        print(f"Results directory: {exp_dir}")


if __name__ == "__main__":
    main()
    
    """
    Example usage:
    
    # Quick test with defaults
    python gnm_esn_landscape.py --use-defaults --n-eta 10 --n-gamma 10
    
    # Full run with custom parameters
    python gnm_esn_landscape.py --eta-min -8 --eta-max 0 --gamma-min 0.1 --gamma-max 8 \
                                --n-eta 30 --n-gamma 30 --density 14 --n-subjects 10
    
    # Visualize existing results
    python gnm_esn_landscape.py --visualize-only output/gnm_esn_landscape_20241231_120000/landscape_results.json
    
    
    
    python src/connectome_analysis/main_pipeline_2_gnm_esn_landscape.py --use-defaults --n-eta 10 --n-gamma 10

    python src/connectome_analysis/enhanced_landscape_viz.py path/to/landscape_results.json

    python src/connectome_analysis/main_pipeline_2_gnm_esn_landscape.py \
    --eta-min -8 --eta-max 0 \
    --gamma-min 0.1 --gamma-max 8 \
    --n-eta 30 --n-gamma 30 \
    --density 14 \
    --n-subjects 10 \
    --esn-spectral-radius 0.99 \
    --esn-input-length 2000
    """