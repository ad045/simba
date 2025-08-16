"""
memory_capacity.py - Integrated memory capacity testing module.
Combines your ESN memory capacity evaluation with GNM-generated networks.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from dataclasses import dataclass
import warnings

# Your existing ESN imports
from echoes.esn import ESNRegressor

# GNM imports
from gnm import defaults
from gnm.models import GNMBinary

# Import centralized paths
from paths_config import get_paths_config, get_experiment_dir

# Import optimized GNM generator
from gnm_network_generator import OptimizedGNMGenerator, GNMParameters


@dataclass
class MemoryCapacityConfig:
    """Configuration for memory capacity experiments."""
    spectral_radius: float = 0.99
    n_lags: int = 50
    train_len: int = 4000
    test_len: int = 1000
    n_runs: int = 10
    input_scaling: float = 1.0
    regression_method: str = "pinv"
    n_transient: int = 0
    leak_rate: float = 1.0
    bias: float = 1.0
    random_state: Optional[int] = 42


class MemoryCapacityEvaluator:
    """
    Integrated memory capacity evaluator for both observed and GNM-generated networks.
    """
    
    def __init__(self, config: Optional[MemoryCapacityConfig] = None):
        """Initialize the memory capacity evaluator."""
        self.config = config or MemoryCapacityConfig()
        self.paths = get_paths_config()
        self.gnm_generator = OptimizedGNMGenerator()
        
    def _generate_mc_dataset(self, 
                           train_len: int, 
                           test_len: int, 
                           n_lags: int, 
                           rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate dataset for memory capacity testing (your existing function)."""
        total_len = int(train_len + test_len + n_lags + 100)
        seq = rng.uniform(-0.5, 0.5, size=(total_len,))
        
        def build_targets(x: np.ndarray, lags: int) -> np.ndarray:
            T = len(x) - lags
            targets = np.zeros((T, lags), dtype=float)
            for i in range(lags):
                targets[:, i] = x[lags - (i + 1) : - (i + 1) if i + 1 > 0 else None]
            return targets
        
        Y_full = build_targets(seq, n_lags)
        start_train = 100
        end_train = start_train + train_len
        
        X_train = seq[start_train : end_train].reshape(-1, 1)
        Y_train = Y_full[start_train : end_train]
        X_test = seq[end_train : end_train + test_len].reshape(-1, 1)
        Y_test = Y_full[end_train : end_train + test_len]
        
        return X_train, Y_train, X_test, Y_test
    
    def evaluate_single_connectome(self, 
                                  connectome: np.ndarray,
                                  config: Optional[MemoryCapacityConfig] = None) -> Dict[str, float]:
        """
        Evaluate memory capacity for a single connectome.
        This is your existing evaluate_memory_capacity_from_connectome function.
        """
        if config is None:
            config = self.config
        
        # Random generator for input sequences
        rng = np.random.default_rng(config.random_state)
        mc_values: List[float] = []
        
        for run in range(config.n_runs):
            # Get data for training and testing
            X_tr, Y_tr, X_te, Y_te = self._generate_mc_dataset(
                config.train_len, config.test_len, config.n_lags, rng
            )
            
            # Create ESN with connectome as weight matrix
            esn = ESNRegressor(
                W=connectome.copy(),
                spectral_radius=config.spectral_radius,
                n_transient=config.n_transient,
                input_scaling=config.input_scaling,
                leak_rate=config.leak_rate,
                bias=config.bias,
                regression_method=config.regression_method,
            )
            
            esn.fit(X_tr, Y_tr)
            Y_pred = esn.predict(X_te)
            
            # Compute Pearson correlation per column
            Yt = Y_te - Y_te.mean(axis=0, keepdims=True)
            Yp = Y_pred - Y_pred.mean(axis=0, keepdims=True)
            denom = (Yt.std(axis=0, ddof=0) * Yp.std(axis=0, ddof=0))
            r = (Yt * Yp).mean(axis=0) / denom
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            
            mc = float(np.sum(r**2))
            mc_values.append(mc)
        
        return {
            "mc_mean": float(np.mean(mc_values)),
            "mc_std": float(np.std(mc_values)),
            "mc_values": mc_values,
            "hparams": {
                "spectral_radius": config.spectral_radius,
                "n_lags": config.n_lags,
                "train_len": config.train_len,
                "test_len": config.test_len,
                "n_runs": config.n_runs,
                "input_scaling": config.input_scaling,
                "regression_method": config.regression_method,
                "n_transient": config.n_transient,
                "leak_rate": config.leak_rate,
                "bias": config.bias,
                "random_state": config.random_state,
            }
        }
    
    def evaluate_observed_connectomes(self,
                                     resolution: int = 68,
                                     densities: Optional[List[int]] = None,
                                     n_subjects: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate memory capacity for observed connectomes.
        
        Args:
            resolution: Connectome resolution
            densities: List of densities to evaluate
            n_subjects: Number of subjects to evaluate (None = all)
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            "resolution": resolution,
            "densities": {},
            "timestamp": datetime.now().isoformat()
        }
        
        if densities is None:
            densities = [10, 12, 14, 16, 18, 20]
        
        for density in densities:
            print(f"Evaluating density {density}%...")
            
            # Load binary connectomes
            conn_path = self.paths.get_connectome_path(resolution, density=density, weighted=False)
            if not conn_path.exists():
                warnings.warn(f"Connectome file not found: {conn_path}")
                continue
            
            connectomes = np.load(conn_path)
            n_eval = min(n_subjects, connectomes.shape[2]) if n_subjects else connectomes.shape[2]
            
            density_results = []
            for i in range(n_eval):
                print(f"  Subject {i+1}/{n_eval}")
                result = self.evaluate_single_connectome(connectomes[:, :, i])
                result["subject_id"] = i
                density_results.append(result)
            
            results["densities"][density] = density_results
            
            # Summary statistics
            mc_means = [r["mc_mean"] for r in density_results]
            results["densities"][f"{density}_summary"] = {
                "mean_mc": np.mean(mc_means),
                "std_mc": np.std(mc_means),
                "min_mc": np.min(mc_means),
                "max_mc": np.max(mc_means),
                "n_subjects": len(density_results)
            }
        
        return results
    
    def evaluate_gnm_generated_networks(self,
                                       target_connectome: np.ndarray,
                                       distance_matrix: np.ndarray,
                                       gnm_params: GNMParameters,
                                       n_realizations: int = 10) -> Dict[str, Any]:
        """
        Generate networks using GNM and evaluate their memory capacity.
        
        Args:
            target_connectome: Target connectome to match
            distance_matrix: Distance matrix
            gnm_params: GNM parameters
            n_realizations: Number of networks to generate
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            "gnm_params": {
                "eta": gnm_params.eta,
                "gamma": gnm_params.gamma,
                "rule": str(gnm_params.generative_rule)
            },
            "n_realizations": n_realizations,
            "evaluations": []
        }
        
        # Generate networks
        print(f"Generating {n_realizations} GNM networks...")
        n_edges = int(np.sum(target_connectome > 0) // 2)
        
        for i in range(n_realizations):
            print(f"  Realization {i+1}/{n_realizations}")
            
            # Generate network using GNM
            generated = self.gnm_generator.generate_network(
                n_nodes=target_connectome.shape[0],
                n_edges=n_edges,
                distance_matrix=torch.tensor(distance_matrix, dtype=torch.float32),
                parameters=gnm_params
            ).cpu().numpy()
            
            # Evaluate memory capacity
            mc_result = self.evaluate_single_connectome(generated)
            mc_result["realization_id"] = i
            results["evaluations"].append(mc_result)
        
        # Summary statistics
        mc_means = [r["mc_mean"] for r in results["evaluations"]]
        results["summary"] = {
            "mean_mc": np.mean(mc_means),
            "std_mc": np.std(mc_means),
            "min_mc": np.min(mc_means),
            "max_mc": np.max(mc_means)
        }
        
        return results
    
    def compare_observed_vs_gnm(self,
                               observed_connectome: np.ndarray,
                               distance_matrix: np.ndarray,
                               gnm_rules: Optional[List[str]] = None,
                               n_realizations: int = 10,
                               fit_parameters: bool = True) -> Dict[str, Any]:
        """
        Compare memory capacity between observed and GNM-generated networks.
        
        Args:
            observed_connectome: Observed connectome to analyze
            distance_matrix: Distance matrix
            gnm_rules: List of GNM rules to test
            n_realizations: Number of realizations per rule
            fit_parameters: Whether to fit GNM parameters first
            
        Returns:
            Comparison results
        """
        if gnm_rules is None:
            gnm_rules = ["matching_index", "spatial", "degree_product"]
        
        results = {
            "observed": {},
            "gnm_generated": {},
            "comparison": {}
        }
        
        # Evaluate observed connectome
        print("Evaluating observed connectome...")
        results["observed"] = self.evaluate_single_connectome(observed_connectome)
        observed_mc = results["observed"]["mc_mean"]
        
        # Test each GNM rule
        for rule_name in gnm_rules:
            print(f"\nTesting GNM rule: {rule_name}")
            
            if fit_parameters:
                # Fit GNM parameters
                print("  Fitting parameters...")
                fit_result = self.gnm_generator.fit_parameters(
                    target_network=torch.tensor(observed_connectome, dtype=torch.float32),
                    distance_matrix=torch.tensor(distance_matrix, dtype=torch.float32),
                    generative_rule_name=rule_name,
                    n_eta=10,
                    n_gamma=10,
                    num_simulations=20
                )
                
                # Create GNM parameters with fitted values
                params = GNMParameters(
                    eta=fit_result["best_eta"],
                    gamma=fit_result["best_gamma"],
                    generative_rule=self.gnm_generator.AVAILABLE_RULES[rule_name]()
                )
            else:
                # Use default parameters
                params = GNMParameters(
                    eta=-2.0,
                    gamma=0.3,
                    generative_rule=self.gnm_generator.AVAILABLE_RULES[rule_name]()
                )
            
            # Generate and evaluate networks
            gnm_results = self.evaluate_gnm_generated_networks(
                target_connectome=observed_connectome,
                distance_matrix=distance_matrix,
                gnm_params=params,
                n_realizations=n_realizations
            )
            
            results["gnm_generated"][rule_name] = gnm_results
            
            # Compare with observed
            gnm_mc = gnm_results["summary"]["mean_mc"]
            results["comparison"][rule_name] = {
                "observed_mc": observed_mc,
                "gnm_mc": gnm_mc,
                "difference": observed_mc - gnm_mc,
                "ratio": observed_mc / gnm_mc if gnm_mc > 0 else float('inf'),
                "gnm_better": gnm_mc > observed_mc
            }
        
        # Find best GNM rule
        best_rule = max(
            results["comparison"].keys(),
            key=lambda x: results["comparison"][x]["gnm_mc"]
        )
        results["best_gnm_rule"] = best_rule
        
        return results
    
    def run_comprehensive_analysis(self,
                                  experiment_name: Optional[str] = None,
                                  resolution: int = 68,
                                  densities: Optional[List[int]] = None,
                                  n_subjects: int = 5,
                                  gnm_rules: Optional[List[str]] = None,
                                  n_gnm_realizations: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive memory capacity analysis.
        
        Args:
            experiment_name: Custom experiment name
            resolution: Connectome resolution
            densities: Densities to analyze
            n_subjects: Number of subjects to analyze
            gnm_rules: GNM rules to test
            n_gnm_realizations: Number of GNM realizations
            
        Returns:
            Comprehensive analysis results
        """
        # Create experiment directory
        exp_dir = self.paths.get_experiment_dir("memory_capacity", experiment_name)
        
        print(f"Running comprehensive memory capacity analysis")
        print(f"Output directory: {exp_dir}")
        
        results = {
            "experiment_name": experiment_name or "comprehensive",
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "analyses": []
        }
        
        if densities is None:
            densities = [10, 15, 20]
        
        # Load distance matrix
        dist_path = self.paths.get_distance_matrix_path(resolution)
        if not dist_path.exists():
            print(f"Warning: Distance matrix not found, using random")
            coords = np.random.rand(resolution, 3) * 100
            distance_matrix = np.zeros((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        else:
            distance_matrix = np.load(dist_path)
        
        # Analyze each density
        for density in densities:
            print(f"\n{'='*60}")
            print(f"Analyzing density {density}%")
            print(f"{'='*60}")
            
            # Load connectomes
            conn_path = self.paths.get_connectome_path(resolution, density=density, weighted=False)
            if not conn_path.exists():
                print(f"Skipping density {density}% - file not found")
                continue
            
            connectomes = np.load(conn_path)
            n_analyze = min(n_subjects, connectomes.shape[2])
            
            density_results = {
                "density": density,
                "subjects": []
            }
            
            # Analyze each subject
            for subj_id in range(n_analyze):
                print(f"\nSubject {subj_id+1}/{n_analyze}")
                observed_conn = connectomes[:, :, subj_id]
                
                # Compare observed vs GNM
                comparison = self.compare_observed_vs_gnm(
                    observed_connectome=observed_conn,
                    distance_matrix=distance_matrix,
                    gnm_rules=gnm_rules,
                    n_realizations=n_gnm_realizations,
                    fit_parameters=True
                )
                
                comparison["subject_id"] = subj_id
                density_results["subjects"].append(comparison)
            
            # Compute density-level statistics
            all_observed_mc = [s["observed"]["mc_mean"] for s in density_results["subjects"]]
            all_gnm_mc = {}
            
            for rule in (gnm_rules or ["matching_index"]):
                rule_mcs = []
                for subj in density_results["subjects"]:
                    if rule in subj["gnm_generated"]:
                        rule_mcs.append(subj["gnm_generated"][rule]["summary"]["mean_mc"])
                if rule_mcs:
                    all_gnm_mc[rule] = rule_mcs
            
            density_results["summary"] = {
                "observed": {
                    "mean": np.mean(all_observed_mc),
                    "std": np.std(all_observed_mc)
                },
                "gnm": {
                    rule: {
                        "mean": np.mean(mcs),
                        "std": np.std(mcs)
                    } for rule, mcs in all_gnm_mc.items()
                }
            }
            
            results["analyses"].append(density_results)
        
        # Save results
        results_file = exp_dir / "memory_capacity_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        # Create summary CSV
        self._create_summary_csv(results, exp_dir / "summary.csv")
        
        return results
    
    def _create_summary_csv(self, results: Dict[str, Any], output_path: Path):
        """Create summary CSV from results."""
        rows = []
        
        for density_analysis in results["analyses"]:
            density = density_analysis["density"]
            
            for subject in density_analysis["subjects"]:
                subject_id = subject["subject_id"]
                observed_mc = subject["observed"]["mc_mean"]
                
                for rule, comparison in subject["comparison"].items():
                    rows.append({
                        "density": density,
                        "subject_id": subject_id,
                        "network_type": "observed",
                        "rule": "observed",
                        "mc_mean": observed_mc,
                        "mc_std": subject["observed"]["mc_std"]
                    })
                    
                    rows.append({
                        "density": density,
                        "subject_id": subject_id,
                        "network_type": "gnm",
                        "rule": rule,
                        "mc_mean": comparison["gnm_mc"],
                        "mc_std": subject["gnm_generated"][rule]["summary"]["std_mc"]
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Summary CSV saved to: {output_path}")


# Convenience functions
def evaluate_memory_capacity_from_connectome(connectome: np.ndarray, **kwargs) -> Dict[str, float]:
    """
    Convenience function matching your original function signature.
    """
    config = MemoryCapacityConfig(**kwargs)
    evaluator = MemoryCapacityEvaluator(config)
    return evaluator.evaluate_single_connectome(connectome)


def test_memory_capacity_with_gnm(use_gnm_defaults: bool = True):
    """
    Quick test function for memory capacity with GNM.
    """
    print("Testing Memory Capacity with GNM Integration")
    print("=" * 60)
    
    evaluator = MemoryCapacityEvaluator()
    
    if use_gnm_defaults:
        # Use GNM's default data
        print("Using GNM default data...")
        device = torch.device("cpu")
        distance_matrix = defaults.get_distance_matrix(device=device).numpy()
        target_network = defaults.get_binary_network(device=device).numpy()
    else:
        # Use random data for testing
        print("Using random test data...")
        n_nodes = 68
        distance_matrix = np.random.rand(n_nodes, n_nodes)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        target_network = np.random.rand(n_nodes, n_nodes) > 0.9
        target_network = ((target_network + target_network.T) > 0).astype(float)
        np.fill_diagonal(target_network, 0)
    
    # Compare observed vs GNM
    results = evaluator.compare_observed_vs_gnm(
        observed_connectome=target_network,
        distance_matrix=distance_matrix,
        gnm_rules=["matching_index", "spatial"],
        n_realizations=3,
        fit_parameters=False  # Use default params for speed
    )
    
    print("\nResults:")
    print(f"Observed MC: {results['observed']['mc_mean']:.3f}")
    
    for rule, comparison in results["comparison"].items():
        print(f"\n{rule}:")
        print(f"  GNM MC: {comparison['gnm_mc']:.3f}")
        print(f"  Difference: {comparison['difference']:.3f}")
        print(f"  GNM Better: {comparison['gnm_better']}")
    
    print(f"\nBest GNM Rule: {results['best_gnm_rule']}")


if __name__ == "__main__":
    # Run test
    test_memory_capacity_with_gnm(use_gnm_defaults=True)