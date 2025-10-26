# """
# Batch processor for running all three analysis steps on multiple experiments:
# 1. Compare networks and generate individual energies
# 2. Find minimum energy values for each individual
# 3. Generate visualizations

# This script automatically detects experiments matching a pattern and processes them sequentially.
# """

# import sys
# from pathlib import Path
# import pandas as pd
# import re
# import traceback
# from typing import List, Dict, Optional
# import numpy as np
# import torch
# from tqdm import tqdm
# from multiprocessing import Pool

# # Add your src directory to path if needed
# # sys.path.append('/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code')

# from src.config.path import PathConfig
# from src.config.GNM import create_evaluation_criteria
# from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer


# # =============================================================================
# # STEP 1: Network Comparison Functions (from first script)
# # =============================================================================

# def extract_params_from_filename(filename: str) -> Dict[str, float]:
#     """Extract eta and gamma parameters from network filename."""
#     match = re.search(r'eta([-\d.]+)_gamma([-\d.]+)', filename)
#     if match:
#         eta = float(match.group(1))
#         gamma = float(match.group(2))
#         return {'eta': eta, 'gamma': gamma}
#     else:
#         raise ValueError(f"Could not extract parameters from filename: {filename}")


# def load_generated_networks(networks_dir: Path):
#     """Load all generated networks from directory."""
#     networks_dir = Path(networks_dir)
#     network_files = sorted(networks_dir.glob("net_eta*.npy"))
    
#     if not network_files:
#         raise FileNotFoundError(f"No network files found in {networks_dir}")
    
#     print(f"Found {len(network_files)} generated network files")
    
#     networks = []
#     network_parameters = []
#     filenames = []
    
#     for net_file in network_files:
#         try:
#             params = extract_params_from_filename(net_file.name)
#             network = np.load(net_file)
#             networks.append(network)
#             network_parameters.append(params)
#             filenames.append(net_file.name)
#         except Exception as e:
#             print(f"Warning: Could not load {net_file.name}: {e}")
#             continue
    
#     networks = np.concatenate(networks)
#     return networks, network_parameters, filenames


# def load_empirical_networks(empirical_path: Path) -> np.ndarray:
#     """Load empirical networks."""
#     empirical_path = Path(empirical_path)
#     if not empirical_path.exists():
#         raise FileNotFoundError(f"Empirical networks file not found: {empirical_path}")
    
#     empirical_networks = np.load(empirical_path)
#     print(f"Loaded empirical networks with shape: {empirical_networks.shape}")
#     return empirical_networks


# def run_network_comparison(
#     experiment_path: Path,
#     dataset_name: str,
#     experiment_name: str,
#     empirical_networks_path: Path,
#     distance_matrix_path: Path,
#     num_processes: int = 8
# ) -> bool:
#     """
#     Run the network comparison for a single experiment.
#     Returns True if successful, False otherwise.
#     """
#     try:
#         print(f"\n{'='*60}")
#         print(f"STEP 1: Network Comparison - {experiment_name}")
#         print(f"{'='*60}")
        
#         # Import necessary comparison functions (assuming they're in the same module)
#         from compare_generated_and_empirical_networks import compare_all_networks
        
#         path_config = PathConfig(
#             dataset_name=dataset_name,
#             experiment_name=experiment_name,
#         )
        
#         generated_networks_dir = experiment_path / "generated_networks"
#         output_path = experiment_path / f"summary_indiv_energies_for_exp_{experiment_name}.csv"
#         reference_csv_path = experiment_path / f"all_metrics_for_{experiment_name}.csv"
        
#         # Check if already processed
#         if output_path.exists():
#             print(f"✓ Summary file already exists: {output_path.name}")
#             return True
        
#         # Load networks
#         generated_networks, generated_parameters, filenames = load_generated_networks(generated_networks_dir)
#         empirical_networks = load_empirical_networks(empirical_networks_path)
        
#         # Validate dimensions
#         if generated_networks.shape[-1] != empirical_networks.shape[-1]:
#             raise ValueError(
#                 f"Dimension mismatch: Generated={generated_networks.shape[-1]}, "
#                 f"Empirical={empirical_networks.shape[-1]}"
#             )
        
#         # Config for evaluation
#         config_dict = {
#             'gnm': {
#                 'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
#             }
#         }
        
#         # Run comparison
#         results_df = compare_all_networks(
#             generated_networks=generated_networks,
#             generated_network_parameters=generated_parameters,
#             filenames=filenames,
#             empirical_networks=empirical_networks,
#             distance_matrices_path=distance_matrix_path if distance_matrix_path.exists() else None,
#             config_dict=config_dict,
#             output_path=output_path,
#             dataset_name=dataset_name,
#             reference_csv_path=reference_csv_path if reference_csv_path.exists() else None,
#             number_multiprocessing_processes=num_processes,
#         )
        
#         print(f"✓ Network comparison complete: {len(results_df)} results")
#         return True
        
#     except Exception as e:
#         print(f"✗ Error in network comparison: {e}")
#         traceback.print_exc()
#         return False


# # =============================================================================
# # STEP 2: Find Minimum Energy
# # =============================================================================

# def find_min_energy(input_csv_path: Path, output_csv_path: Path) -> bool:
#     """
#     Find the gamma and eta combination with lowest energy for each individual.
#     Returns True if successful, False otherwise.
#     """
#     try:
#         print(f"\n{'='*60}")
#         print(f"STEP 2: Finding Minimum Energy")
#         print(f"{'='*60}")
        
#         if output_csv_path.exists():
#             print(f"✓ Min energy file already exists: {output_csv_path.name}")
#             return True
        
#         if not input_csv_path.exists():
#             print(f"✗ Input file not found: {input_csv_path}")
#             return False
        
#         df = pd.read_csv(input_csv_path)
        
#         # Identify energy columns
#         energy_columns = [col for col in df.columns if col.startswith('MaxCrit')]
        
#         if not energy_columns:
#             print("✗ No energy columns found")
#             return False
        
#         results = []
        
#         for col_name in energy_columns:
#             min_energy_idx = df[col_name].idxmin()
#             min_energy_row = df.loc[min_energy_idx]
            
#             subj_match = re.search(r'_indiv_(\d+)', col_name)
#             if subj_match:
#                 subj_index = int(subj_match.group(1))
#             else:
#                 subj_index = col_name
            
#             results.append({
#                 'subj_index': subj_index,
#                 'gamma': min_energy_row['gamma'],
#                 'eta': min_energy_row['eta'],
#                 'energy': min_energy_row[col_name]
#             })
        
#         results_df = pd.DataFrame(results)
#         results_df.sort_values(by='subj_index', inplace=True)
#         results_df.to_csv(output_csv_path, index=False)
        
#         print(f"✓ Min energy results saved: {output_csv_path.name}")
#         print(f"  Found {len(results_df)} individuals")
#         return True
        
#     except Exception as e:
#         print(f"✗ Error finding min energy: {e}")
#         traceback.print_exc()
#         return False


# # =============================================================================
# # STEP 3: Generate Visualizations
# # =============================================================================

# def generate_visualizations(
#     experiment_path: Path,
#     experiment_name: str,
#     plot_indiv_connectomes: bool = True,
#     metrics_to_plot: Optional[List[str]] = None
# ) -> bool:
#     """
#     Generate Voronoi landscape visualizations.
#     Returns True if successful, False otherwise.
#     """
#     try:
#         print(f"\n{'='*60}")
#         print(f"STEP 3: Generating Visualizations")
#         print(f"{'='*60}")
        
#         csv_path = experiment_path / f"all_metrics_for_{experiment_name}.csv"
        
#         if not csv_path.exists():
#             print(f"✗ Metrics CSV not found: {csv_path}")
#             return False
        
#         save_path = experiment_path / "figures_voronoi_only"
#         save_path.mkdir(parents=True, exist_ok=True)
        
#         # Load data
#         gnm_results_df = pd.read_csv(csv_path, index_col=False)
        
#         if gnm_results_df.empty:
#             print("✗ Empty dataframe")
#             return False
        
#         # Load min energy results if available
#         min_energy_path = experiment_path / 'min_energy_results.csv'
#         df_best_estimates = None
#         if min_energy_path.exists() and plot_indiv_connectomes:
#             df_best_estimates = pd.read_csv(min_energy_path)
        
#         # Default metrics if none provided
#         if metrics_to_plot is None:
#             lags_to_plot = [1, 2, 3, 4, 5, 6, 10, 20, 49]
#             metrics_to_plot = [
#                 "MaxCriteria",
#                 "avg_communicability",
#                 "global_efficiency",
#                 "modularity",
#                 "avg_clustering",
#                 "avg_degree",
#                 "transitivity",
#                 "avg_edge_distance",
#                 "char_path_length",
#                 "wiring_cost",
#                 "mc_mean",
#                 "mc_std",
#             ] + [f"mc_{lag}" for lag in lags_to_plot]
        
#         visualizer = PipelineVisualizer()
#         successful_plots = 0
        
#         for metric in metrics_to_plot:
#             try:
#                 df = gnm_results_df.copy()
                
#                 # Handle computed metrics
#                 if metric == "mean_mc_divided_by_wiring_cost":
#                     df[metric] = pd.to_numeric(df["mc_mean"], errors='coerce') / \
#                                  pd.to_numeric(df["wiring_cost"], errors='coerce')
                
#                 df["eta"] = pd.to_numeric(df["eta"], errors='coerce')
#                 df["gamma"] = pd.to_numeric(df["gamma"], errors='coerce')
                
#                 # Find metric column
#                 try:
#                     if metric.startswith("mc_") and metric.split("_")[1].isdigit():
#                         metric_col_name = metric
#                         if metric not in df.columns:
#                             continue
#                     else:
#                         metric_col_name = next(col for col in df.columns if metric in col)
#                 except StopIteration:
#                     continue
                
#                 df[metric_col_name] = pd.to_numeric(df[metric_col_name], errors='coerce')
#                 df = df.dropna(subset=['eta', 'gamma', metric_col_name])
                
#                 if df.empty:
#                     continue
                
#                 # Generate plot
#                 plot_title = visualizer._format_plot_title(metric_col_name)
#                 figure_save_name = f"voronoi_landscape_{metric}.pdf"
#                 full_save_path = save_path / figure_save_name
                
#                 fig, ax = visualizer.plot_metric_landscape_voronoi(
#                     df,
#                     title=plot_title,
#                     metric_name=metric_col_name,
#                     savepath=full_save_path,
#                     dot_color="steelblue",
#                     show=False,
#                     show_dots=False,
#                     annotate_extremes=True,
#                     estimated_indiv_connectomes=df_best_estimates,
#                 )
                
#                 import matplotlib.pyplot as plt
#                 plt.close()
#                 successful_plots += 1
                
#             except Exception as e:
#                 print(f"  Warning: Could not plot {metric}: {e}")
#                 continue
        
#         print(f"✓ Generated {successful_plots}/{len(metrics_to_plot)} plots in {save_path}")
#         return successful_plots > 0
        
#     except Exception as e:
#         print(f"✗ Error generating visualizations: {e}")
#         traceback.print_exc()
#         return False


# # =============================================================================
# # MAIN BATCH PROCESSING
# # =============================================================================

# def find_experiments(base_path: Path, pattern: str) -> List[Path]:
#     """
#     Find all experiment directories matching a pattern.
    
#     Args:
#         base_path: Base directory to search in
#         pattern: Glob pattern (e.g., "49_suarez_MaMI_100_animal_*")
    
#     Returns:
#         List of experiment paths sorted by name
#     """
#     experiment_paths = sorted(base_path.glob(pattern))
#     return [p for p in experiment_paths if p.is_dir()]


# def process_single_experiment(
#     experiment_path: Path,
#     dataset_name: str,
#     empirical_networks_path: Path,
#     distance_matrix_path: Path,
#     skip_comparison: bool = False,
#     skip_min_energy: bool = False,
#     skip_visualization: bool = False,
#     num_processes: int = 8
# ) -> Dict[str, bool]:
#     """
#     Process a single experiment through all three steps.
    
#     Returns:
#         Dictionary with success status for each step
#     """
#     experiment_name = experiment_path.name
#     results = {
#         'experiment': experiment_name,
#         'comparison': None,
#         'min_energy': None,
#         'visualization': None
#     }
    
#     print(f"\n{'#'*70}")
#     print(f"# Processing: {experiment_name}")
#     print(f"{'#'*70}")
    
#     # Step 1: Network Comparison
#     if not skip_comparison:
#         results['comparison'] = run_network_comparison(
#             experiment_path=experiment_path,
#             dataset_name=dataset_name,
#             experiment_name=experiment_name,
#             empirical_networks_path=empirical_networks_path,
#             distance_matrix_path=distance_matrix_path,
#             num_processes=num_processes
#         )
#     else:
#         print("Step 1: Skipped (comparison)")
    
#     # Step 2: Find Minimum Energy
#     if not skip_min_energy:
#         input_csv = experiment_path / f"summary_indiv_energies_for_exp_{experiment_name}.csv"
#         output_csv = experiment_path / 'min_energy_results.csv'
#         results['min_energy'] = find_min_energy(input_csv, output_csv)
#     else:
#         print("Step 2: Skipped (min energy)")
    
#     # Step 3: Generate Visualizations
#     if not skip_visualization:
#         results['visualization'] = generate_visualizations(
#             experiment_path=experiment_path,
#             experiment_name=experiment_name,
#             plot_indiv_connectomes=True
#         )
#     else:
#         print("Step 3: Skipped (visualization)")
    
#     return results


# def batch_process_experiments(
#     dataset_name: str,
#     experiment_pattern: str,
#     base_output_path: Optional[Path] = None,
#     skip_comparison: bool = False,
#     skip_min_energy: bool = False,
#     skip_visualization: bool = False,
#     num_processes: int = 8
# ):
#     """
#     Batch process multiple experiments.
    
#     Args:
#         dataset_name: Name of dataset (e.g., "suarez_MaMI_dataset")
#         experiment_pattern: Pattern to match experiments (e.g., "49_suarez_MaMI_100_animal_*")
#         base_output_path: Base path for experiments (defaults to standard location)
#         skip_comparison: Skip network comparison step
#         skip_min_energy: Skip min energy finding step
#         skip_visualization: Skip visualization step
#         num_processes: Number of processes for multiprocessing
#     """
    
#     # Set up paths
#     if base_output_path is None:
#         base_output_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm")
    
#     dataset_path = base_output_path / dataset_name
    
#     # Dataset-specific paths
#     if dataset_name == "suarez_MaMI_dataset":
#         empirical_networks_path = Path(
#             "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/"
#             "suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_100.npy"
#         )
#         distance_matrix_path = Path(
#             "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/"
#             "suarez_MaMI_dataset/02_distance_matrices/distance_matrix_100.npy"
#         )
#     elif dataset_name == "shafiei_human_consensus_dataset":
#         empirical_networks_path = Path(
#             "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/"
#             "griffa_70_human_connectomes_dataset/01_connectomes/01_indiv_connectomes_bin_density_10_percent_68.npy"
#         )
#         distance_matrix_path = Path(
#             "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/"
#             "shafiei_human_consensus_dataset/02_distance_matrices/distance_matrix_68.npy"
#         )
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")
    
#     # Find experiments
#     experiments = find_experiments(dataset_path, experiment_pattern)
    
#     if not experiments:
#         print(f"No experiments found matching pattern: {experiment_pattern}")
#         return
    
#     print(f"\nFound {len(experiments)} experiments to process:")
#     for exp in experiments:
#         print(f"  - {exp.name}")
    
#     # Process each experiment
#     all_results = []
#     for exp_path in experiments:
#         result = process_single_experiment(
#             experiment_path=exp_path,
#             dataset_name=dataset_name,
#             empirical_networks_path=empirical_networks_path,
#             distance_matrix_path=distance_matrix_path,
#             skip_comparison=skip_comparison,
#             skip_min_energy=skip_min_energy,
#             skip_visualization=skip_visualization,
#             num_processes=num_processes
#         )
#         all_results.append(result)
    
#     # Print summary
#     print(f"\n{'='*70}")
#     print("BATCH PROCESSING SUMMARY")
#     print(f"{'='*70}")
    
#     for result in all_results:
#         exp_name = result['experiment']
#         status_str = f"{exp_name}:"
        
#         if result['comparison'] is not None:
#             status_str += f" Comparison={'✓' if result['comparison'] else '✗'}"
#         if result['min_energy'] is not None:
#             status_str += f" MinEnergy={'✓' if result['min_energy'] else '✗'}"
#         if result['visualization'] is not None:
#             status_str += f" Viz={'✓' if result['visualization'] else '✗'}"
        
#         print(status_str)
    
#     print(f"{'='*70}\n")


# if __name__ == "__main__":
#     # Example usage: Process all animal experiments for suarez_MaMI
#     batch_process_experiments(
#         dataset_name="suarez_MaMI_dataset",
#         experiment_pattern="52_suarez_MaMI_100_animal_*",
#         skip_comparison=True, # False,  # Set to True if comparison already done
#         skip_min_energy=True, # False,
#         skip_visualization=False,
#         num_processes=8
#     )
    
#     # Example: Process shafiei experiments
#     # batch_process_experiments(
#     #     dataset_name="shafiei_human_consensus_dataset",
#     #     experiment_pattern="49_shafiei*",
#     #     num_processes=8
#     # )