"""
ESN evaluation module for the connectome analysis pipeline.
Handles memory capacity evaluation and hyperparameter optimization.
"""

import echoes
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import urllib.request
import zipfile
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import random
import pandas as pd

from src.config.ESN_and_GNM_config import ConfigManager
from src.utils.data_loader import DataLoader
from src.ESNs.alternative_test_memory_capacity_weighted import evaluate_memory_capacity_from_connectome # needs to be here (even if "unused") - otherwise it defaults to row above??
from src.utils.saving_and_finding_files import time_stamp_for_saving
from src.ESNs.utils import _summarize_hparam_space, _write_run_info_txt


class ESNEvaluator:
    """Handles ESN evaluation across multiple subjects and hyperparameters."""
    
    def __init__(self, config: ConfigManager, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
        self.timestamp = time_stamp_for_saving()
        
        # Set multiprocessing method
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    
    def _append_rows_csv(self, rows: List, columns: List[str], path: Path):
        """Utility function to append rows to a CSV file."""
        if not rows:
            return
        
        path = Path(path)
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(path, mode="a", index=False, header=not path.exists())
    
    
    

    def _alternative_evaluate_mc(W, n_lags=50, train_len=4000, test_len=1000):
        """Evaluates the Memory Capacity of a given reservoir matrix W."""
        
        # 1. Generate data for the MC task
        random_sequence = np.random.uniform(-0.5, 0.5, train_len + test_len)
        X = random_sequence.reshape(-1, 1)
        y = np.zeros((len(X), n_lags))
        for i in range(1, n_lags + 1):
            y[i:, i - 1] = X[:-i, 0]
        
        X_train, X_test = X[:train_len], X[train_len:]
        y_train, y_test = y[:train_len], y[train_len:]

        # 2. Create and train the ESN
        esn = echoes.ESNRegressor(
            W=W, 
            spectral_radius=0.99, 
            input_scaling=1e-5, 
            leak_rate=1, 
            bias=1,
            random_state=42
        )
        
        esn.fit(X_train, y_train)
        y_pred = esn.predict(X_test)

        # 3. Calculate the MC score
        mc_score = 0
        for i in range(n_lags):
            corr, _ = pearsonr(y_test[100:, i], y_pred[100:, i]) # Discard initial transient
            mc_score += corr**2
            
        return mc_score
    

    def _subject_job(self, 
                    subj_idx: int,
                    A_obs: np.ndarray, # rename!
                    hparams: Dict[str, Any],
                    timing_flag: bool = True,
                    random_seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate memory capacity for a single subject with given hyperparameters.
        
        Args:
            subj_idx: Subject index
            A_obs: Observed connectome matrix
            hparams: Hyperparameter dictionary
            timing_flag: Whether to track timing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (mc_result_dict, timing_dict)
        """
        if timing_flag:
            t_metrics = 0.0  # Placeholder for potential graph metrics timing: TODO
        
        # Set up hyperparameters with defaults
        esn_hparams = {
            "spectral_radius": 0, #  hparams.get("spectral_radius", self.config.esn.spectral_radius),
            "input_length": 0, # hparams.get("input_length", self.config.esn.input_length),
            "input_scaling": 0, # hparams.get("input_scaling", self.config.esn.input_scaling),
            "regularization_method": 0, # hparams.get("regularization_method", self.config.esn.regularization_method),
            "n_runs": 0, # hparams.get("n_runs", self.config.esn.n_runs),
            "n_lags": 0, # self.config.esn.n_lags,
            "test_len": 0, # self.config.esn.test_len,
            "n_transient": 0, # self.config.esn.n_transient,
            "leak_rate": 0, # self.config.esn.leak_rate,
            "bias": 0, # self.config.esn.bias,
        }
        
        if timing_flag:
            t_esn0 = time.perf_counter()
        
        with warnings.catch_warnings(): # This one fails... 
            warnings.simplefilter("ignore", RuntimeWarning)
            np.seterr(over="ignore", divide="ignore", invalid="ignore")
            
            # mc_result_dict = evaluate_memory_capacity_from_connectome(
            #     connectome=A_obs,  # np float 64
            #     spectral_radius=esn_hparams["spectral_radius"],
            #     n_lags=esn_hparams["n_lags"],
            #     train_len=esn_hparams["input_length"],
            #     test_len=esn_hparams["test_len"],
            #     n_runs=esn_hparams["n_runs"],
            #     input_scaling=esn_hparams["input_scaling"],
            #     regression_method=esn_hparams["regularization_method"],
            #     n_transient=esn_hparams["n_transient"],
            #     leak_rate=esn_hparams["leak_rate"],
            #     bias=esn_hparams["bias"],
            #     random_state=random_seed if random_seed is not None else subj_idx, 
            #     calculate_criticality=hparams.get("calculate_criticality", False),
            #     calculate_info_dynamics=hparams.get("calculate_info_dynamics", False)
            # )
            
            mc_score = self._alternative_evaluate_mc(A_obs, n_lags=50, train_len=4000, test_len=1000)
            
            mc_result_dict = {
            "mc_mean": mc_score, # float(np.mean(mc_values)), 
            "mc_std": 0, # float(np.std(mc_values)),
            "mean_mc_of_individual_runs": 0, # mc_values,  
            "hparams": {
                "spectral_radius": 0, # spectral_radius,
                "n_lags": 0, # n_lags,
                "train_len": 0, # train_len,
                "test_len": 0, # test_len,
                "n_runs": 0, # n_runs,
                "input_scaling": 0, # input_scaling,
                "regression_method": 0, # regression_method,
                "n_transient": 0, #  n_transient,
                "leak_rate": 0, # leak_rate,
                "bias": 0, # bias,
                "random_state": 0, # random_state,
                },
            }
    
            # if (calculate_criticality or calculate_info_dynamics) and all_states_for_metrics:
            #     # Concatenate states from all runs for a more robust estimation
            #     concatenated_states = np.vstack(all_states_for_metrics)
                
            mc_result_dict['branching_ratio'] = 0 # branching_ratio
                    
                # if calculate_info_dynamics:
                #     info_dyn_results = _calculate_information_dynamics(concatenated_states)
                # mc_result_dict.update({"info_dyn_results": 0})

            # return mc_result_dict


            # mc_result_dict = evaluate_memory_capacity_from_connectome(
            #     connectome=A_obs,  # np float 64
            #     spectral_radius=esn_hparams["spectral_radius"],
            #     n_lags=esn_hparams["n_lags"],
            #     train_len=esn_hparams["input_length"],
            #     test_len=esn_hparams["test_len"],
            #     n_runs=esn_hparams["n_runs"],
            #     input_scaling=esn_hparams["input_scaling"],
            #     regression_method=esn_hparams["regularization_method"],
            #     n_transient=esn_hparams["n_transient"],
            #     leak_rate=esn_hparams["leak_rate"],
            #     bias=esn_hparams["bias"],
            #     random_state=random_seed if random_seed is not None else subj_idx, 
            #     calculate_criticality=hparams.get("calculate_criticality", False),
            #     calculate_info_dynamics=hparams.get("calculate_info_dynamics", False)
            # )
            
            # Merge all hyperparameters into the result
            returned_hp = mc_result_dict.get("hparams", {})
            mc_result_dict["hparams"] = {**returned_hp, **hparams, **esn_hparams}
        
        if timing_flag:
            t_esn = time.perf_counter() - t_esn0
            timing_dict = {
                "subject": subj_idx,
                "time_metrics_sec": t_metrics,
                "time_esn_sec": t_esn,
                "time_total_sec": t_metrics + t_esn
            }
        else:
            timing_dict = {"subject": subj_idx}
        
        return mc_result_dict, timing_dict
    

    
    def evaluate_single_subject(self, 
                               subject_idx: int,
                               connectome: np.ndarray,
                               hparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate memory capacity for a single subject.
        
        Args:
            subject_idx: Subject index
            connectome: Connectome matrix
            hparams: Optional hyperparameters (uses config defaults if None)
            
        Returns:
            Dictionary with evaluation results
        """
        if hparams is None:
            hparams = {}
        
        mc_result, timing = self._subject_job(
            subject_idx, connectome, hparams, 
            self.config.compute.timing_flag, self.config.compute.random_seed
        )
        
        return {**mc_result, "timing": timing}
    
    def run_hyperparameter_sweep(self,
                                connectomes: Union[np.ndarray, Dict[int, np.ndarray]],
                                hparam_grid: List[Dict[str, Any]],
                                save_dir: Path,
                                search_mode: str = "grid",
                                random_sample_size: Optional[int] = None) -> str:
        """
        Run hyperparameter sweep across all subjects and hyperparameter combinations.
        
        Args:
            connectomes: Either 3D array (n_nodes, n_nodes, n_subjects) or 
                        dict mapping density -> 3D array
            hparam_grid: List of hyperparameter dictionaries
            save_dir: Directory to save results
            search_mode: "grid" or "random_sample"
            random_sample_size: Number of random samples if using random search
            
        Returns:
            Message indicating completion
        """
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file paths
        mc_values_csv_path = save_dir / f"esn_mc_results_{self.timestamp}.csv"
        duration_csv_path = save_dir / f"esn_durations_{self.timestamp}.csv"
        run_info_path = save_dir / f"run_info_{self.timestamp}.txt"
        
        # Determine number of subjects
        if isinstance(connectomes, dict):
            example_conn = next(iter(connectomes.values()))
            n_subj = example_conn.shape[2]
            densities_available = sorted(connectomes.keys())
        else:
            n_subj = connectomes.shape[2]
            densities_available = None
        
        # Handle random sampling
        if search_mode == "random_sample" and random_sample_size:
            if self.config.compute.random_seed:
                random.seed(self.config.compute.random_seed)
            hparam_grid = random.sample(hparam_grid, min(random_sample_size, len(hparam_grid)))
        
        # Build tasks (subject × hyperparameter combinations)
        tasks = []
        for i in range(n_subj):
            for hp in hparam_grid:
                if isinstance(connectomes, dict):
                    density = hp.get("density_percent")
                    if density not in connectomes:
                        continue
                    A_i = connectomes[density][:, :, i]
                else:
                    A_i = connectomes[:, :, i]
                tasks.append((i, A_i, hp))
        
        total_tasks = len(tasks)
        started_at = datetime.now().isoformat(timespec="seconds")
        
        # Write initial run info
        hparam_space_summary = _summarize_hparam_space(
            [hp for hp in hparam_grid if isinstance(hp, dict)]
        )
        
        _write_run_info_txt(
            run_info_path,
            started_at=started_at,
            updated_at=started_at,
            save_dir=str(save_dir),
            timestamp=str(self.timestamp),
            n_subjects=n_subj,
            total_tasks=total_tasks,
            completed_tasks=0,
            search_mode=search_mode,
            random_sample_size=random_sample_size,
            densities_available=densities_available,
            hparam_space_summary=hparam_space_summary,
        )
        
        # Define output columns
        mc_columns = [
            "subject", "density_percent", "spectral_radius", "input_length",
            "input_scaling", "regularization_method", "n_runs", "mc_mean",
            "mc_std", "mean_mc_of_individual_runs", "hyper_params"
        ]
        
        duration_columns = ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"]
        
        # Run parallel evaluation
        if self.config.compute.timing_flag:
            run_t0 = time.perf_counter()
        
        n_workers = self.config.compute.n_workers or os.cpu_count()
        
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(
                    self._subject_job, 
                    subj_idx, A_i, hp, 
                    self.config.compute.timing_flag, 
                    self.config.compute.random_seed
                )
                for subj_idx, A_i, hp in tasks
            ]
            
            for idx, fut in enumerate(as_completed(futures), 1):
                mc_result_dict, timing_dict = fut.result()
                
                # Extract hyperparameters
                hp_from_job = mc_result_dict.get("hparams", {})
                subject_id = timing_dict["subject"]
                
                # Save MC results
                mc_values = [
                    subject_id,
                    hp_from_job.get("density_percent"),
                    hp_from_job.get("spectral_radius"),
                    hp_from_job.get("input_length"),
                    hp_from_job.get("input_scaling"),
                    hp_from_job.get("regularization_method"),
                    hp_from_job.get("n_runs"),
                    mc_result_dict.get("mc_mean"),
                    mc_result_dict.get("mc_std"),
                    mc_result_dict.get("mean_mc_of_individual_runs"),
                    # mc_result_dict.get("r2_array_from_0_to_n_lags_minus_1"), # TODO: R2 array could be added, but code currently "nearly stops", when added? (TODO_R2_array for searching)
                    hp_from_job,
                ]
                self._append_rows_csv([mc_values], mc_columns, mc_values_csv_path)
                
                # Save timing results
                if self.config.compute.timing_flag:
                    duration_values = [timing_dict[col] for col in duration_columns]
                    self._append_rows_csv([duration_values], duration_columns, duration_csv_path)
                
                # Print progress
                if self.config.compute.timing_flag:
                    print(f"Finished {idx}/{total_tasks} tasks "
                          f"(subject={subject_id}, hp_summary={self._format_hp_for_print(hp_from_job)})")
                
                # Update run info periodically
                if idx % 10 == 0:
                    _write_run_info_txt(
                        run_info_path,
                        started_at=started_at,
                        updated_at=datetime.now().isoformat(timespec="seconds"),
                        save_dir=str(save_dir),
                        timestamp=str(self.timestamp),
                        n_subjects=n_subj,
                        total_tasks=total_tasks,
                        completed_tasks=idx,
                        search_mode=search_mode,
                        random_sample_size=random_sample_size,
                        densities_available=densities_available,
                        hparam_space_summary=hparam_space_summary,
                    )
        
        # Final timing and cleanup
        if self.config.compute.timing_flag:
            total_sec = time.perf_counter() - run_t0
            print(f"Total run time: {total_sec:.3f}s")
            print(f"Average time per task: {total_sec / total_tasks:.3f}s")
            
            # Write final timing summary
            self._append_rows_csv(
                [("COMPLETED", np.nan, np.nan, f"{total_sec:.3f}")],
                duration_columns,
                duration_csv_path
            )
        
        # Mark completion in results file
        self._append_rows_csv(["COMPLETED"], ["subject"], mc_values_csv_path)
        
        # Final run info update
        _write_run_info_txt(
            run_info_path,
            started_at=started_at,
            updated_at=datetime.now().isoformat(timespec="seconds"),
            save_dir=str(save_dir),
            timestamp=str(self.timestamp),
            n_subjects=n_subj,
            total_tasks=total_tasks,
            completed_tasks=total_tasks,
            search_mode=search_mode,
            random_sample_size=random_sample_size,
            densities_available=densities_available,
            hparam_space_summary=hparam_space_summary,
        )
        
        return f"ESN evaluation completed. Results saved to {save_dir}"
    
    def _format_hp_for_print(self, hp_dict: Dict[str, Any]) -> str:
        """Format hyperparameters for concise printing."""
        key_items = ["density_percent", "spectral_radius", "input_length", "regularization_method"]
        formatted_items = []
        
        for key in key_items:
            if key in hp_dict and hp_dict[key] is not None:
                value = hp_dict[key]
                if isinstance(value, float):
                    formatted_items.append(f"{key}={value:.2f}")
                else:
                    formatted_items.append(f"{key}={value}")
        
        return "{" + ", ".join(formatted_items) + "}"
    
    def load_and_analyze_results(self, results_dir: Path) -> Dict[str, Any]:
        """
        Load and analyze results from a completed hyperparameter sweep.
        
        Args:
            results_dir: Directory containing results files
            
        Returns:
            Dictionary with analysis results
        """
        results_dir = Path(results_dir)
        
        # Find the most recent results file
        mc_files = list(results_dir.glob("esn_mc_results_*.csv"))
        if not mc_files:
            raise FileNotFoundError(f"No results files found in {results_dir}")
        
        latest_file = max(mc_files, key=lambda x: x.stat().st_mtime)
        
        try:
            # Load results
            df = pd.read_csv(latest_file)
            # Remove completion marker row if present
            df = df[df['subject'] != 'COMPLETED']
            
            if df.empty:
                return {"error": "No valid results found in file"}
            
            # Convert numeric columns
            numeric_cols = ['subject', 'density_percent', 'spectral_radius', 
                           'input_length', 'input_scaling', 'n_runs', 
                           'mc_mean', 'mc_std']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Basic statistics
            analysis = {
                'file_path': str(latest_file),
                'total_evaluations': len(df),
                'unique_subjects': df['subject'].nunique(),
                'unique_hyperparameter_combinations': len(df.drop('subject', axis=1).drop_duplicates()),
                'best_overall_mc': {
                    'value': df['mc_mean'].max(),
                    'hyperparameters': df.loc[df['mc_mean'].idxmax()].to_dict()
                },
                'mc_statistics': {
                    'mean': df['mc_mean'].mean(),
                    'std': df['mc_mean'].std(),
                    'min': df['mc_mean'].min(),
                    'max': df['mc_mean'].max(),
                    'median': df['mc_mean'].median()
                },
                'hyperparameter_analysis': {}
            }
            
            # Analyze effect of each hyperparameter
            hparam_cols = ['density_percent', 'spectral_radius', 'input_length', 
                          'input_scaling', 'regularization_method']
            
            for col in hparam_cols:
                if col in df.columns and df[col].notna().any():
                    if df[col].dtype in ['object', 'category']:
                        # Categorical analysis
                        grouped = df.groupby(col)['mc_mean'].agg(['mean', 'std', 'count'])
                        analysis['hyperparameter_analysis'][col] = grouped.to_dict('index')
                    else:
                        # Numerical analysis
                        corr = df[col].corr(df['mc_mean'])
                        analysis['hyperparameter_analysis'][col] = {
                            'correlation_with_mc': corr,
                            'unique_values': sorted(df[col].dropna().unique().tolist()),
                            'best_value': df.loc[df['mc_mean'].idxmax(), col]
                        }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Failed to analyze results: {e}"}


def create_esn_evaluator(config_path: Optional[Union[str, Path]] = None,
                        config_manager: Optional[ConfigManager] = None,
                        data_loader: Optional[DataLoader] = None) -> ESNEvaluator:
    """
    Factory function to create an ESNEvaluator instance.
    
    Args:
        config_path: Path to JSON configuration file
        config_manager: Pre-configured ConfigManager instance
        data_loader: Pre-configured DataLoader instance
        
    Returns:
        ESNEvaluator instance
    """
    if config_manager is None:
        if config_path is not None:
            config_manager = ConfigManager.load_config(Path(config_path))
        else:
            config_manager = ConfigManager()
    
    if data_loader is None:
        from src.utils.data_loader import DataLoader
        data_loader = DataLoader(config_manager)
    
    return ESNEvaluator(config_manager, data_loader)