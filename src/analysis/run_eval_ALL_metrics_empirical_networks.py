# #!/usr/bin/env python3
# """
# Robust overnight runner for metric calculations.
# Processes metrics in small batches to ensure failures don't block everything.
# Modified to load connectomes from a numpy array file.
# """

# import sys
# import json
# import time
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import traceback

# # Import your original multiprocessing function
# from analysis.evaluate_further_metrics_utils import multiprocess_networks

# # Configuration
# CONNECTOMES_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_50.npy"  
# EXPERIMENT = "empirical_all_metrics_analysis"  # New experiment name
# DATASET = "suarez_MaMI_dataset"
# BASE_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
# N_PROCESSES = 12
# DEBUG = False
# CREATE_BIG_CSV = True


# # Define all metrics organized by category
# ALL_METRICS = {
#     "static": [
#         "density", # works # done
#         "avg_clustering", # works # done
#         "avg_degree", # works # done
#         "degree_assortativity", # works # done
#         "modularity", # works # done
#         "transitivity", # works # done
        
#         "topological_distance",
#         "degree_gini",
#         "wiring_cost", # no shortest_path_distance - this does not work. 
#         "structural_complexity",
#         "n_connected_components",
        
#         "omega",
#     ],
#     "dynamic": [
#         "spectral_radius",
#         "spectral_gap",
#         "spectral_gap_fatemeh",
        
#         "global_efficiency",
#         "diffusion_efficiency",
#         "propagation_efficiency",
#         "nct_control",
#         "nct_energies",
        
#         # "metastability", # this is now replaced... (soon)
#         ###"novel_metastability", -> not working properly yet
#         "synchronizability_eigenratio",
#         "algebraic_connectivity_nx",
#         "kuramoto_synchronization",
#         "community_synchronization_vulnerability",
#     ],
#     "computational": [
#         "kernel_rank",
#         "kernel_rank_fatemeh",
#         "effective_dimensionality",
#         "multifunctionality",
#     ]
# }

# # Batch size - how many metrics to process together
# BATCH_SIZE = 2


# class RobustMetricRunner:
#     """Manages robust metric calculation with logging and error recovery."""
    
#     def __init__(self, experiment, dataset, base_path, connectomes_array):
#         self.experiment = experiment
#         self.dataset = dataset
#         self.base_path = base_path
#         self.connectomes = connectomes_array
#         self.log_dir = base_path / "output" / "gnm" / dataset / experiment / "processing_logs"
#         self.log_dir.mkdir(parents=True, exist_ok=True)
        
#         self.status_file = self.log_dir / "metric_processing_status.json"
#         self.log_file = self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
#         self.status = self._load_status()
    
#     def _load_status(self):
#         """Load processing status from file."""
#         if self.status_file.exists():
#             with open(self.status_file, 'r') as f:
#                 return json.load(f)
#         return {
#             "completed_metrics": [],
#             "failed_metrics": [],
#             "in_progress": None,
#             "last_updated": None
#         }
    
#     def _save_status(self):
#         """Save current status to file."""
#         self.status["last_updated"] = datetime.now().isoformat()
#         with open(self.status_file, 'w') as f:
#             json.dump(self.status, f, indent=2)
    
#     def log(self, message, level="INFO"):
#         """Log message to both console and file."""
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         log_msg = f"[{timestamp}] [{level}] {message}"
#         print(log_msg)
#         with open(self.log_file, 'a') as f:
#             f.write(log_msg + "\n")
    
#     def _create_metric_batches(self, metrics_dict):
#         """Split metrics into batches."""
#         batches = []
#         for category, metrics in metrics_dict.items():
#             # Filter out already completed metrics
#             remaining_metrics = [
#                 m for m in metrics 
#                 if f"{category}:{m}" not in self.status["completed_metrics"]
#             ]
            
#             # Create batches
#             for i in range(0, len(remaining_metrics), BATCH_SIZE):
#                 batch = remaining_metrics[i:i + BATCH_SIZE]
#                 batches.append({
#                     category: batch
#                 })
        
#         return batches
    
#     def _process_batch(self, batch_metrics):
#         """Process a single batch of metrics."""
#         batch_name = ", ".join([
#             f"{cat}:{','.join(mets)}" 
#             for cat, mets in batch_metrics.items()
#         ])
        
#         self.log(f"Starting batch: {batch_name}")
#         self.status["in_progress"] = batch_name
#         self._save_status()
        
#         try:
#             # Create full metric dict with only this batch's metrics
#             interesting_metrics = {
#                 "static": batch_metrics.get("static", []),
#                 "dynamic": batch_metrics.get("dynamic", []),
#                 "computational": batch_metrics.get("computational", [])
#             }
            
#             # Run the multiprocessing with numpy array
#             start_time = time.time()
#             result = multiprocess_networks(
#                 n_processes=N_PROCESSES,
#                 experiment=self.experiment,
#                 dataset=self.dataset,
#                 base_path=self.base_path,
#                 interesting_metrics=interesting_metrics,
#                 save_interval=2,
#                 debug=DEBUG,
#                 create_big_update_csv=False,  # Only create at the end
#                 connectomes_array=self.connectomes  # Pass the numpy array
#             )
            
#             elapsed = time.time() - start_time
#             self.log(f"Batch completed in {elapsed:.1f}s: {batch_name}", "SUCCESS")
            
#             # Mark metrics as completed
#             for category, metrics in batch_metrics.items():
#                 for metric in metrics:
#                     metric_id = f"{category}:{metric}"
#                     if metric_id not in self.status["completed_metrics"]:
#                         self.status["completed_metrics"].append(metric_id)
            
#             self.status["in_progress"] = None
#             self._save_status()
#             return True
            
#         except Exception as e:
#             self.log(f"Batch FAILED: {batch_name}", "ERROR")
#             self.log(f"Error: {str(e)}", "ERROR")
#             self.log(traceback.format_exc(), "ERROR")
            
#             # Mark metrics as failed
#             for category, metrics in batch_metrics.items():
#                 for metric in metrics:
#                     metric_id = f"{category}:{metric}"
#                     if metric_id not in self.status["failed_metrics"]:
#                         self.status["failed_metrics"].append(metric_id)
            
#             self.status["in_progress"] = None
#             self._save_status()
#             return False
    
#     def run(self):
#         """Main run loop."""
#         self.log("=" * 80)
#         self.log(f"Starting robust metric processing for {self.experiment}")
#         self.log(f"Connectomes shape: {self.connectomes.shape}")
#         self.log(f"Number of connectomes: {self.connectomes.shape[0]}")
#         self.log(f"Batch size: {BATCH_SIZE} metrics per batch")
#         self.log("=" * 80)
        
#         # Create batches
#         batches = self._create_metric_batches(ALL_METRICS)
        
#         if not batches:
#             self.log("No remaining metrics to process!", "SUCCESS")
#             self._create_final_csv()
#             return
        
#         self.log(f"Total batches to process: {len(batches)}")
#         self.log(f"Already completed: {len(self.status['completed_metrics'])} metrics")
#         self.log(f"Previously failed: {len(self.status['failed_metrics'])} metrics")
        
#         # Process each batch
#         successful_batches = 0
#         failed_batches = 0
        
#         for i, batch in enumerate(batches, 1):
#             self.log("-" * 80)
#             self.log(f"Batch {i}/{len(batches)}")
            
#             if self._process_batch(batch):
#                 successful_batches += 1
#             else:
#                 failed_batches += 1
#                 self.log(f"Continuing despite failure...", "WARNING")
            
#             # Small delay between batches
#             time.sleep(2)
        
#         # Final summary
#         self.log("=" * 80)
#         self.log("PROCESSING COMPLETE", "SUCCESS")
#         self.log(f"Successful batches: {successful_batches}/{len(batches)}")
#         self.log(f"Failed batches: {failed_batches}/{len(batches)}")
#         self.log(f"Total completed metrics: {len(self.status['completed_metrics'])}")
#         self.log(f"Total failed metrics: {len(self.status['failed_metrics'])}")
        
#         if self.status['failed_metrics']:
#             self.log("Failed metrics:", "WARNING")
#             for metric in self.status['failed_metrics']:
#                 self.log(f"  - {metric}", "WARNING")
        
#         self.log("=" * 80)
        
#         # Create final combined CSV
#         if CREATE_BIG_CSV:
#             self._create_final_csv()
    
#     def _create_final_csv(self):
#         """Create the final combined CSV file."""
#         self.log("Creating final combined CSV file...")
#         try:
#             from analysis.evaluate_further_metrics_utils import create_combined_csv
#             import pandas as pd
            
#             output_path = self.base_path / "output" / "gnm" / self.dataset / self.experiment
            
#             # Load all category CSVs
#             df_dict = {}
#             for category in ['static', 'dynamic', 'computational']:
#                 csv_path = output_path / f'all_{category}_metrics_for_{self.experiment}_updated.csv'
#                 if csv_path.exists():
#                     df_dict[category] = pd.read_csv(csv_path)
#                 else:
#                     df_dict[category] = pd.DataFrame(columns=['id'])
            
#             # Create combined CSV
#             create_combined_csv(output_path, self.experiment, df_dict)
#             self.log("✓ Combined CSV created successfully", "SUCCESS")
            
#         except Exception as e:
#             self.log(f"Failed to create combined CSV: {e}", "ERROR")


# def load_connectomes(filepath):
#     """Load connectomes from numpy file."""
#     print(f"Loading connectomes from {filepath}...")
#     connectomes = np.load(filepath)
#     print(f"✓ Loaded array with shape: {connectomes.shape}")
    
#     # Validate shape
#     if len(connectomes.shape) != 3:
#         raise ValueError(f"Expected 3D array (n_connectomes, n_nodes, n_nodes), got shape {connectomes.shape}")
    
#     n_connectomes, n_rows, n_cols = connectomes.shape
#     if n_rows != n_cols:
#         raise ValueError(f"Expected square matrices, got {n_rows}x{n_cols}")
    
#     print(f"✓ Validated: {n_connectomes} connectomes of size {n_rows}x{n_rows}")
#     return connectomes


# def main():
#     """Main entry point."""
#     try:
#         # Load connectomes from numpy file
#         connectomes = load_connectomes(CONNECTOMES_FILE)
        
#         # Create runner with the loaded array
#         runner = RobustMetricRunner(EXPERIMENT, DATASET, BASE_PATH, connectomes)
        
#         # Run processing
#         runner.run()
        
#     except KeyboardInterrupt:
#         print("\nProcess interrupted by user")
#         print("Progress saved. Run again to continue.")
#         sys.exit(1)
#     except Exception as e:
#         print(f"ERROR: {e}")
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Robust overnight runner for metric calculations.
Modified to work with numpy array input (225, 100, 100).
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import traceback
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import gc

# Import metric calculators
from src.analysis.metric_calculators import (
    StaticMetricCalculator, 
    DynamicMetricCalculator, 
    ComputationMetricCalculator
)


# Configuration
CONNECTOMES_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_50.npy"  
EXPERIMENT = "empirical_all_metrics_analysis"  
DATASET = "suarez_MaMI_dataset"
BASE_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
N_PROCESSES = 12
DEBUG = False
CREATE_BIG_CSV = True
BATCH_SIZE = 2


# Define all metrics organized by category
ALL_METRICS = {
    "static": [
        "density", # works # done
        "avg_clustering", # works # done
        "avg_degree", # works # done
        "degree_assortativity", # works # done
        "modularity", # works # done
        "transitivity", # works # done
        
        "topological_distance",
        "degree_gini",
        "wiring_cost", # no shortest_path_distance - this does not work. 
        "structural_complexity",
        "n_connected_components",
        
        "omega",
    ],
    "dynamic": [
        "spectral_radius",
        "spectral_gap",
        "spectral_gap_fatemeh",
        
        "global_efficiency",
        "diffusion_efficiency",
        "propagation_efficiency",
        "nct_control",
        "nct_energies",
        
        # "metastability", # this is now replaced... (soon)
        ###"novel_metastability", -> not working properly yet
        "synchronizability_eigenratio",
        "algebraic_connectivity_nx",
        "kuramoto_synchronization",
        "community_synchronization_vulnerability",
    ],
    "computational": [
        "kernel_rank",
        "kernel_rank_fatemeh",
        "effective_dimensionality",
        "multifunctionality",
    ]
}


# Global variable to cache distance matrix per worker process
_worker_distance_matrix = None
_worker_distance_matrix_path = None


def _load_distance_matrix_once(distance_matrix_path):
    """Load distance matrix once per worker process and cache it."""
    global _worker_distance_matrix, _worker_distance_matrix_path
    
    if _worker_distance_matrix is None or _worker_distance_matrix_path != distance_matrix_path:
        if distance_matrix_path and Path(distance_matrix_path).exists():
            _worker_distance_matrix = np.load(distance_matrix_path)
            _worker_distance_matrix_path = distance_matrix_path
        else:
            _worker_distance_matrix = None
    
    return _worker_distance_matrix


def convert_to_python_types(obj):
    """Recursively convert NumPy types to Python native types for pickling."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_to_python_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        if obj.size == 1:
            return convert_to_python_types(obj.item())
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def process_single_connectome(args):
    """
    Process a single connectome from the array.
    
    Args:
        args: tuple of (idx, connectome, distance_matrix_path, metrics_to_calculate)
    
    Returns:
        dict: {'static': {...}, 'dynamic': {...}, 'computational': {...}}
    """
    idx, connectome, distance_matrix_path, metrics_to_calculate = args
    
    try:
        # Load distance matrix once per worker (cached)
        distance_matrix = _load_distance_matrix_once(distance_matrix_path)
        
        # Initialize calculators
        static_calc = StaticMetricCalculator(A=connectome, distance_matrix=distance_matrix)
        dynamic_calc = DynamicMetricCalculator(A=connectome)
        computation_calc = ComputationMetricCalculator(A=connectome)
        
        # Calculate metrics by category
        results = {
            'static': {"id": idx},
            'dynamic': {"id": idx},
            'computational': {"id": idx}
        }
        
        for category, calculator in [
            ('static', static_calc),
            ('dynamic', dynamic_calc),
            ('computational', computation_calc)
        ]:
            for metric in metrics_to_calculate.get(category, []):
                try:
                    result = calculator.calculate_metric(metric)
                    result = convert_to_python_types(result)
                    
                    # Handle dict results by unpacking with prefix
                    if isinstance(result, dict):
                        for key, value in result.items():
                            results[category][f"{metric}_{key}"] = value
                    else:
                        results[category][metric] = result
                        
                except Exception as e:
                    print(f"Error calculating {metric} for connectome {idx}: {e}")
                    results[category][metric] = np.nan
        
        # Clean up
        del static_calc, dynamic_calc, computation_calc
        gc.collect()
        
        if (idx + 1) % 50 == 0:
            print(f"Processed connectome {idx+1}/225")
        
        return results
    
    except Exception as e:
        print(f"Error processing connectome {idx}: {e}")
        return None


def save_checkpoint(results_list, output_path, checkpoint_num):
    """Save accumulated results to checkpoint CSVs by category."""
    if not results_list:
        return
    
    results_by_category = {
        'static': [],
        'dynamic': [],
        'computational': []
    }
    
    for result in results_list:
        if result is None:
            continue
        for category in ['static', 'dynamic', 'computational']:
            if result[category] and len(result[category]) > 1:  # More than just id
                results_by_category[category].append(result[category])
    
    # Save each category separately
    temp_dir = output_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for category in ['static', 'dynamic', 'computational']:
        if not results_by_category[category]:
            continue
        
        results_df = pd.DataFrame(results_by_category[category])
        checkpoint_path = temp_dir / f"result_{category}_{checkpoint_num:04d}.csv"
        results_df.to_csv(checkpoint_path, index=False)
        print(f"Saved {category} checkpoint: {len(results_df)} connectomes")


def merge_checkpoints(output_path, experiment_name):
    """Merge all checkpoint files into final CSV files."""
    temp_dir = output_path / "temp"
    
    if not temp_dir.exists():
        print("No checkpoints to merge")
        return {}
    
    final_dfs = {}
    
    for category in ['static', 'dynamic', 'computational']:
        checkpoint_files = sorted(temp_dir.glob(f'result_{category}_*.csv'))
        
        if not checkpoint_files:
            print(f"No {category} checkpoints found")
            continue
        
        print(f"\nMerging {len(checkpoint_files)} {category} checkpoints...")
        
        dfs = []
        for file in checkpoint_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
        
        if not dfs:
            continue
        
        # Concatenate and remove duplicates
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['id'], keep='last')
        
        # Save final file
        output_file = output_path / f'all_{category}_metrics_{experiment_name}.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"✓ Saved {category} results: {len(merged_df)} connectomes")
        
        final_dfs[category] = merged_df
        
        # Clean up checkpoint files
        for file in checkpoint_files:
            file.unlink()
    
    # Remove temp directory if empty
    try:
        if not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            print("✓ Cleaned up temp directory")
    except:
        pass
    
    return final_dfs


def create_combined_csv(output_path, experiment_name, df_dict):
    """Create a combined CSV with all metrics."""
    print("\nCreating combined CSV...")
    
    df_combined = df_dict['static'].copy()
    
    # Merge dynamic metrics
    if 'dynamic' in df_dict and not df_dict['dynamic'].empty:
        dynamic_cols = [col for col in df_dict['dynamic'].columns if col != 'id']
        if dynamic_cols:
            df_combined = df_combined.merge(
                df_dict['dynamic'][['id'] + dynamic_cols],
                on='id',
                how='outer'
            )
            print(f"✓ Merged {len(dynamic_cols)} dynamic metrics")
    
    # Merge computational metrics
    if 'computational' in df_dict and not df_dict['computational'].empty:
        comp_cols = [col for col in df_dict['computational'].columns if col != 'id']
        if comp_cols:
            df_combined = df_combined.merge(
                df_dict['computational'][['id'] + comp_cols],
                on='id',
                how='outer'
            )
            print(f"✓ Merged {len(comp_cols)} computational metrics")
    
    combined_file = output_path / f'all_metrics_{experiment_name}.csv'
    df_combined.to_csv(combined_file, index=False)
    print(f"✓ Saved combined CSV: {len(df_combined)} connectomes")
    
    return df_combined


class RobustMetricRunner:
    """Manages robust metric calculation with logging and error recovery."""
    
    def __init__(self, connectomes_path, experiment, dataset, base_path):
        self.connectomes_path = connectomes_path
        self.experiment = experiment
        self.dataset = dataset
        self.base_path = base_path
        
        self.output_path = base_path / "output" / "gnm" / dataset / experiment
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.output_path / "processing_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.status_file = self.log_dir / "metric_processing_status.json"
        self.log_file = self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.status = self._load_status()
        self.connectomes = None
        self.distance_matrix_path = base_path / "data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy"
    
    def _load_status(self):
        """Load processing status from file."""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {
            "completed_metrics": [],
            "failed_metrics": [],
            "in_progress": None,
            "last_updated": None
        }
    
    def _save_status(self):
        """Save current status to file."""
        self.status["last_updated"] = datetime.now().isoformat()
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def log(self, message, level="INFO"):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def load_connectomes(self):
        """Load connectomes from numpy file."""
        self.log(f"Loading connectomes from {self.connectomes_path}...")
        self.connectomes = np.load(self.connectomes_path)
        self.log(f"✓ Loaded connectomes with shape: {self.connectomes.shape}")
        
        if self.connectomes.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {self.connectomes.shape}")
        if self.connectomes.shape[1] != self.connectomes.shape[2]:
            raise ValueError(f"Expected square matrices, got {self.connectomes.shape[1]}x{self.connectomes.shape[2]}")
    
    def _create_metric_batches(self, metrics_dict):
        """Split metrics into batches."""
        batches = []
        for category, metrics in metrics_dict.items():
            remaining_metrics = [
                m for m in metrics 
                if f"{category}:{m}" not in self.status["completed_metrics"]
            ]
            
            for i in range(0, len(remaining_metrics), BATCH_SIZE):
                batch = remaining_metrics[i:i + BATCH_SIZE]
                batches.append({category: batch})
        
        return batches
    
    def _process_batch(self, batch_metrics):
        """Process a single batch of metrics."""
        batch_name = ", ".join([
            f"{cat}:{','.join(mets)}" 
            for cat, mets in batch_metrics.items()
        ])
        
        self.log(f"Starting batch: {batch_name}")
        self.status["in_progress"] = batch_name
        self._save_status()
        
        try:
            n_connectomes = len(self.connectomes)
            accumulated_results = []
            checkpoint_counter = 0
            
            # Prepare tasks
            tasks = [
                (idx, self.connectomes[idx], str(self.distance_matrix_path), batch_metrics)
                for idx in range(n_connectomes)
            ]
            
            start_time = time.time()
            
            with Pool(processes=N_PROCESSES, maxtasksperchild=50) as pool:
                for i, result in enumerate(pool.imap(
                    process_single_connectome,
                    tasks,
                    chunksize=10
                )):
                    if result is not None:
                        accumulated_results.append(result)
                    
                    if (i + 1) % 50 == 0:
                        save_checkpoint(
                            accumulated_results,
                            self.output_path,
                            checkpoint_counter
                        )
                        checkpoint_counter += 1
                        accumulated_results = []
                        gc.collect()
            
            # Final save
            if accumulated_results:
                save_checkpoint(
                    accumulated_results,
                    self.output_path,
                    checkpoint_counter
                )
            
            elapsed = time.time() - start_time
            self.log(f"Batch completed in {elapsed:.1f}s: {batch_name}", "SUCCESS")
            
            # Mark metrics as completed
            for category, metrics in batch_metrics.items():
                for metric in metrics:
                    metric_id = f"{category}:{metric}"
                    if metric_id not in self.status["completed_metrics"]:
                        self.status["completed_metrics"].append(metric_id)
            
            self.status["in_progress"] = None
            self._save_status()
            return True
            
        except Exception as e:
            self.log(f"Batch FAILED: {batch_name}", "ERROR")
            self.log(f"Error: {str(e)}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")
            
            for category, metrics in batch_metrics.items():
                for metric in metrics:
                    metric_id = f"{category}:{metric}"
                    if metric_id not in self.status["failed_metrics"]:
                        self.status["failed_metrics"].append(metric_id)
            
            self.status["in_progress"] = None
            self._save_status()
            return False
    
    def run(self):
        """Main run loop."""
        self.log("=" * 80)
        self.log(f"Starting metric processing for {self.experiment}")
        self.log("=" * 80)
        
        # Load connectomes
        self.load_connectomes()
        
        # Create batches
        batches = self._create_metric_batches(ALL_METRICS)
        
        if not batches:
            self.log("No remaining metrics to process!", "SUCCESS")
            self._finalize()
            return
        
        self.log(f"Total batches to process: {len(batches)}")
        
        # Process each batch
        successful_batches = 0
        failed_batches = 0
        
        for i, batch in enumerate(batches, 1):
            self.log("-" * 80)
            self.log(f"Batch {i}/{len(batches)}")
            
            if self._process_batch(batch):
                successful_batches += 1
            else:
                failed_batches += 1
            
            time.sleep(2)
        
        # Final summary
        self.log("=" * 80)
        self.log("PROCESSING COMPLETE", "SUCCESS")
        self.log(f"Successful: {successful_batches}/{len(batches)}")
        self.log(f"Failed: {failed_batches}/{len(batches)}")
        self.log("=" * 80)
        
        if CREATE_BIG_CSV:
            self._finalize()
    
    def _finalize(self):
        """Merge checkpoints and create final CSVs."""
        self.log("Finalizing results...")
        
        # Merge checkpoints
        final_dfs = merge_checkpoints(self.output_path, self.experiment)
        
        # Create combined CSV
        if final_dfs and CREATE_BIG_CSV:
            create_combined_csv(self.output_path, self.experiment, final_dfs)


def main():
    """Main entry point."""
    runner = RobustMetricRunner(
        CONNECTOMES_FILE,
        EXPERIMENT,
        DATASET,
        BASE_PATH
    )
    
    try:
        runner.run()
    except KeyboardInterrupt:
        runner.log("Process interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        runner.log(f"Unexpected error: {e}", "ERROR")
        runner.log(traceback.format_exc(), "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()