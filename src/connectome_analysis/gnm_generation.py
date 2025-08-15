"""
Generative Network Model (GNM) generation and fitting module.
Handles GNM parameter estimation, model generation, and evaluation.
"""

import os
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from numba import njit
from scipy.stats import ks_2samp

from config import ConfigManager, GNMConfig
from data_loader import DataLoader

# Import your existing functions
try:
    from src.ESNs.generate_weight_matrices_bio_no_rank_weighted import (
        build_weight_matrix_from_bin_conn, 
        build_weight_matrix_from_connectome
    )
    from src.ESNs.test_memory_capacity_weighted import evaluate_memory_capacity
    from src.structural_analysis.graph_measures_optimized_weighted import analyze_connectomes
    from src.utils.saving_and_finding_files import time_stamp_for_saving
except ImportError as e:
    warnings.warn(f"Could not import GNM modules: {e}")
    # Define dummy functions for testing
    def build_weight_matrix_from_bin_conn(A, **kwargs):
        return A.astype(float)
    
    def build_weight_matrix_from_connectome(A, **kwargs):
        return A.astype(float)
    
    def evaluate_memory_capacity(W, **kwargs):
        return np.random.random()
    
    def analyze_connectomes(connectomes, **kwargs):
        return [{"modularity": np.random.random(), "global_efficiency": np.random.random()}]
    
    def time_stamp_for_saving():
        return datetime.now().strftime("%Y%m%d_%H%M%S")


# Numba-compiled helper functions (from your original script)
@njit(cache=True)
def _compute_homophily_bin(A_bin: np.ndarray) -> np.ndarray:
    """Compute the homophily matrix for a binary adjacency matrix."""
    Af = A_bin.astype(np.float64)
    shared = Af @ Af
    deg = Af.sum(axis=1)
    denom = deg[:, None] + deg[None, :] - shared
    H = np.where(denom > 0.0, shared / denom, 0.0)
    np.fill_diagonal(H, 0.0)
    return H


@njit(cache=True)
def _rand_choice_weighted(p_vec):
    """Weighted random choice using numba."""
    tot = 0.0
    x = np.random.random()
    for k in range(p_vec.size):
        tot += p_vec[k]
        if x < tot:
            return k
    return p_vec.size - 1


@njit(cache=True)
def generate_gnm_numba(A_seed, target_m, dist, gamma, eta, eps: float = 1e-12):
    """Generate GNM using numba for speed."""
    n = A_seed.shape[0]
    A = A_seed.copy()
    D = np.maximum(dist, eps)
    cost_term = D ** eta
    np.fill_diagonal(cost_term, 0.0)

    m = int(A.sum() // 2)
    while m < target_m:
        H = _compute_homophily_bin(A)
        prob = cost_term * np.maximum(H, eps) ** gamma

        flat_prob = prob.ravel()
        mask_flat = (A == 0).ravel()
        for i in range(n):
            idx_diag = i * n + i
            mask_flat[idx_diag] = False
            base = i * n
            for j in range(i):
                mask_flat[base + j] = False

        cand_idx = np.nonzero(mask_flat)[0]
        if cand_idx.size == 0:
            break
        p_vec = flat_prob[cand_idx]
        p_sum = p_vec.sum()
        if p_sum == 0.0:
            k = cand_idx[np.random.randint(cand_idx.size)]
        else:
            p_vec /= p_sum
            k = cand_idx[_rand_choice_weighted(p_vec)]
        i, j = divmod(k, n)
        A[i, j] = A[j, i] = 1
        m += 1
    return A


class GNMGenerator:
    """Handles GNM generation, fitting, and evaluation."""
    
    def __init__(self, config: ConfigManager, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
        self.timestamp = time_stamp_for_saving()
        
        # Set multiprocessing method
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    
    def _binarize(self, A: np.ndarray, thr: float = 0.0) -> np.ndarray:
        """Binarize a weighted adjacency matrix."""
        A_sym = (A + A.T) / 2.0
        np.fill_diagonal(A_sym, 0.0)
        return (A_sym > thr).astype(np.int8)
    
    def _precompute_obs_metrics(self, A_bin: np.ndarray, dist: np.ndarray):
        """Precompute observed network metrics for fitting."""
        G = nx.from_numpy_array(A_bin)
        deg = np.array([d for _, d in G.degree()])
        cc = np.array(list(nx.clustering(G).values()))
        bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
        el = np.array([dist[i, j] for i, j in G.edges()])
        return deg, cc, bc, el
    
    def _gnm_energy_fast(self, deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist):
        """Compute energy (fitness) of a simulated network."""
        B_sim = np.triu((A_sim > 0).astype(int), 1)
        B_sim += B_sim.T
        Gs = nx.from_numpy_array(B_sim)

        deg_sim = np.array([d for _, d in Gs.degree()])
        cc_sim = np.array(list(nx.clustering(Gs).values()))
        bc_sim = np.array(list(nx.betweenness_centrality(Gs, normalized=True).values()))
        el_sim = np.array([dist[i, j] for i, j in Gs.edges()])

        return max(
            ks_2samp(deg_obs, deg_sim).statistic,
            ks_2samp(cc_obs, cc_sim).statistic,
            ks_2samp(bc_obs, bc_sim).statistic,
            ks_2samp(el_obs, el_sim).statistic,
        )
    
    def _append_rows_csv(self, rows, columns, path):
        """Utility function to append rows to a CSV file."""
        path = Path(path)
        if not rows:
            return
        pd.DataFrame(rows, columns=columns).to_csv(
            path, mode="a", index=False, header=not path.exists()
        )
    
    def _subject_job(self, 
                    subj_idx: int,
                    A_obs: np.ndarray,
                    dist: np.ndarray,
                    esn_use_observed_real_weights: bool,
                    eta_vals: List[float],
                    gamma_vals: List[float],
                    subset_eta_vals: Optional[List[float]] = None,
                    subset_gamma_vals: Optional[List[float]] = None,
                    timing_flag: bool = True) -> Tuple:
        """
        Process a single subject: fit GNM parameters and evaluate memory capacity.
        
        Returns:
            Tuple of (grid_rows, best_row, gm_labels, timing, subset_rows, subset_labels)
        """
        # Stage 0: observed metrics
        t0 = time.perf_counter() if timing_flag else 0
        A_bin = self._binarize(A_obs)
        m_edges = int(A_bin.sum() // 2)
        deg_obs, cc_obs, bc_obs, el_obs = self._precompute_obs_metrics(A_bin, dist)
        t_metrics = time.perf_counter() - t0 if timing_flag else 0

        # Stage 1: full grid search
        seed = np.zeros_like(A_obs, dtype=np.int8)
        best_energy = np.inf
        best_eta = best_gamma = None
        grid_rows = []
        t_grow = 0.0
        
        grid_combinations = [(eta, gamma) for eta in eta_vals for gamma in gamma_vals]
        
        for eta, gamma in grid_combinations:
            tg0 = time.perf_counter() if timing_flag else 0
            A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
            t_grow += time.perf_counter() - tg0 if timing_flag else 0
            energy = self._gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist)
            grid_rows.append((subj_idx, eta, gamma, energy))
            if energy < best_energy:
                best_energy, best_eta, best_gamma = energy, eta, gamma

        # Stage 2: best model evaluation
        t_esn0 = time.perf_counter() if timing_flag else 0
        A_best = generate_gnm_numba(seed, m_edges, dist, best_gamma, best_eta)
        
        # Graph measures
        gms_raw = analyze_connectomes(
            connectomes=np.expand_dims(A_best, 2),
            distance_matrix=dist,
            comm_mode='estrada_scaled'
        )[0]
        
        if isinstance(gms_raw, dict):
            gm_labels = list(gms_raw.keys())
            gm_vals = list(gms_raw.values())
        else:
            gm_labels = [f"gm_{i}" for i in range(len(gms_raw))]
            gm_vals = list(gms_raw)

        # Memory capacity evaluation
        W = (build_weight_matrix_from_connectome(A_obs, spectral_radius=0.99)
             if esn_use_observed_real_weights else
             build_weight_matrix_from_bin_conn(A_best, spectral_radius=0.99, rank=False))
        
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        if eig_max > 0:
            W *= 0.8 / eig_max
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np.seterr(over="ignore", divide="ignore", invalid="ignore")
            mc = evaluate_memory_capacity(
                W, n_lags=50, train_len=4000, test_len=1000,
                n_runs=10, random_state=subj_idx
            )
        
        t_esn = time.perf_counter() - t_esn0 if timing_flag else 0

        best_row = (subj_idx, best_eta, best_gamma, best_energy, mc, *gm_vals)
        timing = dict(metrics=t_metrics, grow=t_grow, esn=t_esn) if timing_flag else {}

        # Stage 3: subset grid evaluation (if enabled)
        subset_rows = []
        subset_labels = None
        
        if subset_eta_vals and subset_gamma_vals:
            subset_combinations = [(eta, gamma) for eta in subset_eta_vals for gamma in subset_gamma_vals]
            
            for eta, gamma in subset_combinations:
                A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
                gms = analyze_connectomes(
                    connectomes=np.expand_dims(A_sim, 2),
                    distance_matrix=dist,
                    comm_mode='estrada_scaled'
                )[0]
                
                if isinstance(gms, dict):
                    labels = list(gms.keys())
                    vals = list(gms.values())
                else:
                    labels = [f"gm_{i}" for i in range(len(gms))]
                    vals = list(gms)
                
                W = (build_weight_matrix_from_connectome(A_obs, spectral_radius=0.99)
                     if esn_use_observed_real_weights else
                     build_weight_matrix_from_bin_conn(A_sim, spectral_radius=0.99, rank=False))
                
                eig_max = np.max(np.abs(np.linalg.eigvals(W)))
                if eig_max > 0:
                    W *= 0.8 / eig_max
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    np.seterr(over="ignore", divide="ignore", invalid="ignore")
                    mc_s = evaluate_memory_capacity(
                        W, n_lags=50, train_len=4000, test_len=1000,
                        n_runs=10, random_state=subj_idx
                    )
                
                subset_rows.append((subj_idx, eta, gamma, mc_s, *vals))
                subset_labels = labels

        return grid_rows, best_row, gm_labels, timing, subset_rows, subset_labels
    
    def fit_gnm_parameters(self,
                          connectomes: np.ndarray,
                          distance_matrix: np.ndarray,
                          save_dir: Path,
                          esn_use_observed_real_weights: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fit GNM parameters to observed connectomes.
        
        Args:
            connectomes: 3D array (n_nodes, n_nodes, n_subjects)
            distance_matrix: 2D distance matrix (n_nodes, n_nodes)
            save_dir: Directory to save results
            esn_use_observed_real_weights: Whether to use observed weights for ESN evaluation
            
        Returns:
            Dictionary containing result DataFrames
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (save_dir / "results").mkdir(exist_ok=True)
        (save_dir / "preliminary_results").mkdir(exist_ok=True)
        
        # Partial file paths for incremental saving
        grid_partial = save_dir / "preliminary_results" / f"gnm_grid_partial_{self.timestamp}.csv"
        best_partial = save_dir / "preliminary_results" / f"gnm_best_partial_{self.timestamp}.csv"
        subset_partial = save_dir / "preliminary_results" / f"gnm_subset_partial_{self.timestamp}.csv"
        timing_partial = save_dir / "preliminary_results" / f"gnm_timing_partial_{self.timestamp}.csv"
        
        # Prepare parameter grids
        eta_vals = np.linspace(
            self.config.gnm.eta_start, 
            self.config.gnm.eta_end, 
            self.config.gnm.n_eta
        )
        gamma_vals = np.linspace(
            self.config.gnm.gamma_start, 
            self.config.gnm.gamma_end, 
            self.config.gnm.n_gamma
        )
        
        subset_eta_vals = subset_gamma_vals = None
        if self.config.gnm.include_subset:
            subset_eta_vals = np.linspace(
                self.config.gnm.eta_start, 
                self.config.gnm.eta_end, 
                self.config.gnm.subset_size
            )
            subset_gamma_vals = np.linspace(
                self.config.gnm.gamma_start, 
                self.config.gnm.gamma_end, 
                self.config.gnm.subset_size
            )

        n_subj = connectomes.shape[2]
        tasks = [
            (i, connectomes[:, :, i], distance_matrix, esn_use_observed_real_weights,
             eta_vals, gamma_vals, subset_eta_vals, subset_gamma_vals, 
             self.config.compute.timing_flag)
            for i in range(n_subj)
        ]

        # Start parallel processing
        run_t0 = time.perf_counter() if self.config.compute.timing_flag else 0
        
        grid_rows_all, best_rows, timing_list = [], [], []
        subset_rows_all = []
        gm_labels_master = subset_labels_master = None

        n_workers = self.config.compute.n_workers or os.cpu_count()
        
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(self._subject_job, *t) for t in tasks]
            
            for idx, fut in enumerate(as_completed(futures), 1):
                g_rows, b_row, gm_labels, timing, sub_rows, sub_labels = fut.result()
                grid_rows_all.extend(g_rows)
                best_rows.append(b_row)
                timing_list.append(timing)
                subset_rows_all.extend(sub_rows)

                if gm_labels_master is None:
                    gm_labels_master = gm_labels
                if subset_labels_master is None:
                    subset_labels_master = sub_labels

                # Save results incrementally
                self._append_rows_csv(g_rows,
                                     ["subject", "eta", "gamma", "energy"],
                                     grid_partial)

                self._append_rows_csv([b_row],
                                     ["subject", "eta", "gamma", "energy", "memory_capacity", *gm_labels_master],
                                     best_partial)

                if self.config.gnm.include_subset and sub_rows:
                    self._append_rows_csv(sub_rows,
                                         ["subject", "eta", "gamma", "memory_capacity", *subset_labels_master],
                                         subset_partial)

                # Save timing information
                if self.config.compute.timing_flag and timing:
                    self._append_rows_csv([(b_row[0],
                                           timing["metrics"],
                                           timing["grow"],
                                           timing["esn"],
                                           timing["metrics"] + timing["grow"] + timing["esn"])],
                                         ["subject", "metrics_s", "grow_s", "esn_s", "total_s"],
                                         timing_partial)

                # Progress reporting
                if self.config.compute.timing_flag:
                    print(f"Finished {idx}/{n_subj} subjects. "
                          f"Subject {b_row[0]}: metrics={timing.get('metrics', 0):.3f}s, "
                          f"grow={timing.get('grow', 0):.3f}s, esn={timing.get('esn', 0):.3f}s")

        # Final timing summary
        total_sec = time.perf_counter() - run_t0 if self.config.compute.timing_flag else 0
        if self.config.compute.timing_flag:
            print(f"Total run time: {total_sec:.3f}s")
            print(f"Average time per subject: {total_sec / n_subj:.3f}s")
            
            # Save completion marker
            self._append_rows_csv(
                [("COMPLETED", np.nan, np.nan, np.nan, f"{total_sec:.3f}")],
                ["subject", "metrics_s", "grow_s", "esn_s", "total_s"],
                timing_partial
            )

        # Create final DataFrames
        df_grid = pd.DataFrame(grid_rows_all, columns=["subject", "eta", "gamma", "energy"])
        df_best = pd.DataFrame(best_rows, columns=["subject", "eta", "gamma", "energy", "memory_capacity", *gm_labels_master])

        df_subset = None
        if self.config.gnm.include_subset and subset_rows_all:
            cols = ["subject", "eta", "gamma", "memory_capacity", *subset_labels_master]
            df_subset = pd.DataFrame(subset_rows_all, columns=cols)

        # Save final results
        df_grid.to_csv(save_dir / "results" / f"gnm_grid_results_{self.timestamp}.csv", index=False)
        df_best.to_csv(save_dir / "results" / f"gnm_best_results_{self.timestamp}.csv", index=False)
        if df_subset is not None:
            df_subset.to_csv(save_dir / "results" / f"gnm_subset_results_{self.timestamp}.csv", index=False)

        results = {
            'grid': df_grid,
            'best': df_best,
            'subset': df_subset,
            'save_dir': save_dir,
            'timing_summary': {
                'total_time_sec': total_sec,
                'avg_time_per_subject_sec': total_sec / n_subj if n_subj > 0 else 0
            }
        }

        return results
    
    def generate_synthetic_connectomes(self,
                                     reference_connectome: np.ndarray,
                                     distance_matrix: np.ndarray,
                                     eta: float,
                                     gamma: float,
                                     n_realizations: int = 100) -> np.ndarray:
        """
        Generate multiple synthetic connectomes using fitted GNM parameters.
        
        Args:
            reference_connectome: Reference connectome for target edge density
            distance_matrix: Distance matrix for the network
            eta: Fitted eta parameter
            gamma: Fitted gamma parameter
            n_realizations: Number of synthetic networks to generate
            
        Returns:
            3D array of synthetic connectomes (n_nodes, n_nodes, n_realizations)
        """
        A_bin = self._binarize(reference_connectome)
        m_edges = int(A_bin.sum() // 2)
        seed = np.zeros_like(reference_connectome, dtype=np.int8)
        
        n_nodes = reference_connectome.shape[0]
        synthetic_connectomes = np.zeros((n_nodes, n_nodes, n_realizations), dtype=np.int8)
        
        for i in range(n_realizations):
            synthetic_connectomes[:, :, i] = generate_gnm_numba(seed, m_edges, distance_matrix, gamma, eta)
        
        return synthetic_connectomes
    
    def evaluate_gnm_fit_quality(self,
                                observed_connectome: np.ndarray,
                                distance_matrix: np.ndarray,
                                eta: float,
                                gamma: float,
                                n_synthetic: int = 100) -> Dict[str, Any]:
        """
        Evaluate the quality of GNM fit by comparing synthetic networks to observed.
        
        Args:
            observed_connectome: Observed connectome matrix
            distance_matrix: Distance matrix
            eta: GNM eta parameter
            gamma: GNM gamma parameter
            n_synthetic: Number of synthetic networks to generate for comparison
            
        Returns:
            Dictionary with fit quality metrics
        """
        # Generate synthetic networks
        synthetic_connectomes = self.generate_synthetic_connectomes(
            observed_connectome, distance_matrix, eta, gamma, n_synthetic
        )
        
        # Compute metrics for observed network
        A_bin_obs = self._binarize(observed_connectome)
        deg_obs, cc_obs, bc_obs, el_obs = self._precompute_obs_metrics(A_bin_obs, distance_matrix)
        
        # Compute metrics for synthetic networks
        energies = []
        for i in range(n_synthetic):
            A_syn = synthetic_connectomes[:, :, i]
            energy = self._gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_syn, distance_matrix)
            energies.append(energy)
        
        fit_quality = {
            'eta': eta,
            'gamma': gamma,
            'n_synthetic_networks': n_synthetic,
            'energy_stats': {
                'mean': np.mean(energies),
                'std': np.std(energies),
                'min': np.min(energies),
                'max': np.max(energies),
                'median': np.median(energies)
            },
            'best_energy': np.min(energies),
            'energy_distribution': energies
        }
        
        return fit_quality
    
    def load_and_analyze_gnm_results(self, results_dir: Path) -> Dict[str, Any]:
        """
        Load and analyze results from a completed GNM parameter fitting.
        
        Args:
            results_dir: Directory containing GNM results
            
        Returns:
            Dictionary with analysis results
        """
        results_dir = Path(results_dir)
        
        # Find results files
        grid_files = list(results_dir.glob("**/gnm_grid_results_*.csv"))
        best_files = list(results_dir.glob("**/gnm_best_results_*.csv"))
        
        if not grid_files or not best_files:
            return {"error": "GNM results files not found"}
        
        # Load most recent files
        latest_grid = max(grid_files, key=lambda x: x.stat().st_mtime)
        latest_best = max(best_files, key=lambda x: x.stat().st_mtime)
        
        try:
            df_grid = pd.read_csv(latest_grid)
            df_best = pd.read_csv(latest_best)
            
            # Remove completion markers
            df_grid = df_grid[df_grid['subject'] != 'COMPLETED']
            df_best = df_best[df_best['subject'] != 'COMPLETED']
            
            analysis = {
                'grid_file': str(latest_grid),
                'best_file': str(latest_best),
                'n_subjects': df_best['subject'].nunique(),
                'parameter_ranges': {
                    'eta': {
                        'min': df_grid['eta'].min(),
                        'max': df_grid['eta'].max(),
                        'unique_values': len(df_grid['eta'].unique())
                    },
                    'gamma': {
                        'min': df_grid['gamma'].min(),
                        'max': df_grid['gamma'].max(),
                        'unique_values': len(df_grid['gamma'].unique())
                    }
                },
                'energy_statistics': {
                    'min': df_grid['energy'].min(),
                    'max': df_grid['energy'].max(),
                    'mean': df_grid['energy'].mean(),
                    'std': df_grid['energy'].std()
                },
                'best_parameters': {
                    'eta_mean': df_best['eta'].mean(),
                    'eta_std': df_best['eta'].std(),
                    'gamma_mean': df_best['gamma'].mean(),
                    'gamma_std': df_best['gamma'].std()
                },
                'memory_capacity_stats': {
                    'mean': df_best['memory_capacity'].mean(),
                    'std': df_best['memory_capacity'].std(),
                    'min': df_best['memory_capacity'].min(),
                    'max': df_best['memory_capacity'].max()
                }
            }
            
            # Correlation analysis
            analysis['correlations'] = {
                'eta_vs_memory_capacity': df_best['eta'].corr(df_best['memory_capacity']),
                'gamma_vs_memory_capacity': df_best['gamma'].corr(df_best['memory_capacity']),
                'energy_vs_memory_capacity': df_best['energy'].corr(df_best['memory_capacity'])
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Failed to analyze GNM results: {e}"}


def create_gnm_generator(config_path: Optional[Path] = None,
                        config_manager: Optional[ConfigManager] = None,
                        data_loader: Optional[DataLoader] = None) -> GNMGenerator:
    """
    Factory function to create a GNMGenerator instance.
    
    Args:
        config_path: Path to JSON configuration file
        config_manager: Pre-configured ConfigManager instance
        data_loader: Pre-configured DataLoader instance
        
    Returns:
        GNMGenerator instance
    """
    if config_manager is None:
        if config_path is not None:
            config_manager = ConfigManager.load_config(Path(config_path))
        else:
            config_manager = ConfigManager()
    
    if data_loader is None:
        data_loader = DataLoader(config_manager)
    
    return GNMGenerator(config_manager, data_loader)


# Example usage
if __name__ == "__main__":
    from config import get_production_config
    from data_loader import create_data_loader
    
    # Set up configuration
    config = get_production_config()
    data_loader = create_data_loader(config_manager=config)
    gnm_gen = GNMGenerator(config, data_loader)
    
    try:
        # Load data
        summary = data_loader.get_data_summary()
        print("Data summary:", summary)
        
        # Load distance matrix
        dist_matrix = data_loader.load_distance_matrix()
        print(f"Distance matrix loaded: {dist_matrix.shape}")
        
        # Load weighted connectomes
        weighted_by_density = data_loader.load_weighted_by_density()
        
        if weighted_by_density:
            # Use first available density for testing
            test_density = sorted(weighted_by_density.keys())[0]
            test_connectomes = weighted_by_density[test_density]
            
            print(f"Testing with density {test_density}%, connectome shape: {test_connectomes.shape}")
            
            # Fit GNM parameters (on a subset for testing)
            test_save_dir = Path("./test_gnm_results")
            
            # Use only first few subjects for testing
            n_test_subjects = min(3, test_connectomes.shape[2])
            test_conn_subset = test_connectomes[:, :, :n_test_subjects]
            
            results = gnm_gen.fit_gnm_parameters(
                connectomes=test_conn_subset,
                distance_matrix=dist_matrix,
                save_dir=test_save_dir,
                esn_use_observed_real_weights=True
            )
            
            print(f"GNM fitting completed. Results saved to {results['save_dir']}")
            print(f"Best parameters for first subject: eta={results['best'].iloc[0]['eta']:.3f}, "
                  f"gamma={results['best'].iloc[0]['gamma']:.3f}")
            
            # Analyze results
            analysis = gnm_gen.load_and_analyze_gnm_results(test_save_dir)
            print("Analysis results:", analysis)
            
        else:
            print("No connectome data available for testing")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()