# #!/usr/bin/env python3
# """
# Robust overnight runner for metric calculations.
# Processes metrics in small batches to ensure failures don't block everything.
# MODIFIED VERSION: Supports networks stored as 3D numpy arrays (n_networks, n_nodes, n_nodes)
# """

# import sys
# import json
# import time
# from pathlib import Path
# from datetime import datetime
# import traceback
# import numpy as np

# from analysis.evaluate_further_metrics_utils import multiprocess_networks

# # Configuration
# DEBUG = False
# CREATE_BIG_CSV = True

# # Define all metrics organized by category
# ALL_METRICS = {
#     "static": [
        
#         "n_connected_components",

#         "density", # works # done
#         "avg_clustering", # works # done
#         "avg_degree", # works # done
#         "degree_assortativity", # works # done
        
#         "modularity", # works # done
#         "transitivity", # works # done
        
#         "topological_distance",
#         "degree_gini",
        
#         # "wiring_cost", # no shortest_path_distance - this does not work. 
#         "structural_complexity",
#         # "n_connected_components",
        
#         "omega",
        
#         "directed_simplices"
#     ],
#     "dynamic": [
#         "spectral_radius",
#         "spectral_gap",
#             # "spectral_gap_fatemeh",
        
#         "global_efficiency",
#         "diffusion_efficiency",
        
#         "propagation_efficiency",
#         "nct_control",
#         "nct_energies",
        
#             # "metastability", # this is now replaced... (soon - see line below)
#             # "novel_metastability", # THIS COULD WORK, BUT I DID NOT CHECK IT YET... 
        
#         "synchronizability_eigenratio",
#         # "algebraic_connectivity_nx",
#         # "kuramoto_synchronization",
#         # "community_synchronization_vulnerability",
#     ],
#     "computational": [
#         "kernel_rank",
#             # "kernel_rank_fatemeh",
#         "effective_dimensionality",
#             # "multifunctionality",
#     ]
# }


# class RobustMetricRunner:
#     """Manages robust metric calculation with logging and error recovery."""
    
#     def __init__(self, experiment, dataset, base_path, n_processes, number_parallel_metrics, 
#                  network_array_path=None):
#         self.n_processes = n_processes
#         self.number_parallel_metrics = number_parallel_metrics
#         self.experiment = experiment
#         self.dataset = dataset
#         self.base_path = base_path
#         self.network_array_path = network_array_path  # NEW: Path to 3D numpy array
        
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
#             for i in range(0, len(remaining_metrics), self.number_parallel_metrics):
#                 batch = remaining_metrics[i:i + self.number_parallel_metrics]
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
            
#             # Run the multiprocessing
#             start_time = time.time()
#             result = multiprocess_networks(
#                 n_processes=self.n_processes, 
#                 experiment=self.experiment,
#                 dataset=self.dataset,
#                 base_path=self.base_path,
#                 interesting_metrics=interesting_metrics,
#                 save_interval=100, 
#                 debug=DEBUG,
#                 create_big_update_csv=False,  # Only create at the end
#                 network_array_path=self.network_array_path  # NEW: Pass the array path
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
#         self.log(f"Batch size: {self.number_parallel_metrics} metrics per batch")
#         if self.network_array_path:
#             self.log(f"Network array path: {self.network_array_path}")
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
#                     df_dict[category] = pd.DataFrame(columns=['eta', 'gamma', 'id'])
            
#             # Create combined CSV
#             create_combined_csv(output_path, self.experiment, df_dict)
#             self.log("✓ Combined CSV created successfully", "SUCCESS")
            
#         except Exception as e:
#             self.log(f"Failed to create combined CSV: {e}", "ERROR")


# def main(experiment_name: str, 
#          dataset_name: str, 
#          base_path: Path | str, 
#          n_processes: int, 
#          number_parallel_metrics: int,
#          network_array_path: str | Path = None):
#     """
#     Main entry point.
    
#     Args:
#         experiment_name: Name of the experiment
#         dataset_name: Name of the dataset
#         base_path: Base path for the project
#         n_processes: Number of parallel processes
#         number_parallel_metrics: Number of metrics to process in each batch
#         network_array_path: Optional path to 3D numpy array (.npy file) containing networks
#                            Shape should be (n_networks, n_nodes, n_nodes)
#     """
#     runner = RobustMetricRunner(
#         experiment_name, 
#         dataset_name, 
#         base_path,
#         n_processes=n_processes,
#         number_parallel_metrics=number_parallel_metrics,
#         network_array_path=network_array_path
#     )

#     try:
#         runner.run()
#     except KeyboardInterrupt:
#         runner.log("Process interrupted by user", "WARNING")
#         runner.log(f"Progress saved. Run again to continue.", "INFO")
#         sys.exit(1)
#     except Exception as e:
#         runner.log(f"Unexpected error: {e}", "ERROR")
#         runner.log(traceback.format_exc(), "ERROR")
#         sys.exit(1)


# if __name__ == "__main__":
    
#     ########### HUMAN ##############################################################
    
#     dataset_name = "hcp_schaefer_100_dataset"
#     experiment_name = "05_mst_animal_0_compared_with_hcp_schaefer_100" # 105_distance_metrics_mst_animal_0"
#     dataset_experiment_datafile_name = {
#         "hcp_schaefer_100_dataset": ["05_mst_animal_0_compared_with_hcp_schaefer_100", "00_connectomes_density10.npy"],
#         "suarez_MaMI_dataset": ["05_mst_animal_0_compared_with_mami", "00_connectomes_50.npy"],
#         "kaysons_generated_networks_diffusion": ["05_mst_animal_0_compared_with_diffusion", "diffusion_10_percent.npy"],
#         "kaysons_generated_networks_propagation": ["05_mst_animal_0_compared_with_propagation", "propagation_10_percent.npy"],
#         "kaysons_generated_networks_routing": ["05_mst_animal_0_compared_with_routing", "routing_density10.npy"]
#     }

#     # General settings
#     n_processes = 10
#     number_parallel_metrics = 2
    
#     base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")

#     for dataset_name, (experiment_name, datafile_name) in dataset_experiment_datafile_name.items():
#         # create experiment folder if it does not exist
#         # experiment_path = base_path / "output" / "gnm" / dataset_name / experiment_name
#         # experiment_path.mkdir(parents=True, exist_ok=True)
    
#         # NEW: Path to 3D numpy array of networks
#         # Set to None to use the original file-based loading
#         network_array_path = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/{dataset_name}/01_connectomes/{datafile_name}"

#         main(
#             experiment_name=experiment_name, 
#             dataset_name=dataset_name, 
#             base_path=base_path,
#             n_processes=n_processes,
#             number_parallel_metrics=number_parallel_metrics,
#             network_array_path=network_array_path
#         )
    
#     # ############ MAMI ###################################################################
    
#     # dataset_name = "suarez_MaMI_dataset" # hcp_schaefer_100_dataset"
#     # experiment_name = "05_mst_animal_0_compared_with_mami" # 105_distance_metrics_mst_animal_0"
    
#     # # Path to 3D numpy array of networks
#     # network_array_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/kaysons_generated_networks_diffusion/01_connectomes/diffusion_10_percent.npy" 

#     # main(
#     #     experiment_name=experiment_name, 
#     #     dataset_name=dataset_name, 
#     #     base_path=base_path,
#     #     n_processes=n_processes,
#     #     number_parallel_metrics=number_parallel_metrics,
#     #     network_array_path=network_array_path
#     # )
    
    
#     # ############## DIFFUSION ##################################################################
    
#     # dataset_name = "hcp_schaefer_100_dataset"
#     # experiment_name = "05_mst_animal_0_compared_with_hcp_schaefer_100" # 105_distance_metrics_mst_animal_0"
    
#     # # Path to 3D numpy array of networks
#     # network_array_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/kaysons_generated_networks_diffusion/01_connectomes/diffusion_10_percent.npy" 

#     # main(
#     #     experiment_name=experiment_name, 
#     #     dataset_name=dataset_name, 
#     #     base_path=base_path,
#     #     n_processes=n_processes,
#     #     number_parallel_metrics=number_parallel_metrics,
#     #     network_array_path=network_array_path
#     # )
    
    
#     # ############## ROUTING #################################################################
    
#     # dataset_name = "hcp_schaefer_100_dataset"
#     # experiment_name = "05_mst_animal_0_compared_with_hcp_schaefer_100" # 105_distance_metrics_mst_animal_0"
    
#     # # Path to 3D numpy array of networks
#     # network_array_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/kaysons_generated_networks_diffusion/01_connectomes/diffusion_10_percent.npy" 

#     # main(
#     #     experiment_name=experiment_name, 
#     #     dataset_name=dataset_name, 
#     #     base_path=base_path,
#     #     n_processes=n_processes,
#     #     number_parallel_metrics=number_parallel_metrics,
#     #     network_array_path=network_array_path
#     # )
    
    
#     # ################ PROPAGATION ################################################################
    
#     # dataset_name = "hcp_schaefer_100_dataset"
#     # experiment_name = "05_mst_animal_0_compared_with_hcp_schaefer_100" # 105_distance_metrics_mst_animal_0"
    
#     # # Path to 3D numpy array of networks
#     # network_array_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/kaysons_generated_networks_diffusion/01_connectomes/diffusion_10_percent.npy" 

#     # main(
#     #     experiment_name=experiment_name, 
#     #     dataset_name=dataset_name, 
#     #     base_path=base_path,
#     #     n_processes=n_processes,
#     #     number_parallel_metrics=number_parallel_metrics,
#     #     network_array_path=network_array_path
#     # )