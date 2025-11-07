import numpy as np 
import pandas as pd
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import signal
import sys
import gc
import re

from src.analysis.metric_calculators import StaticMetricCalculator, DynamicMetricCalculator, ComputationMetricCalculator     
from src.analysis.utils import sorted_listing_by_creation_time

ONLY_SUMMARIZE = False

def get_eta_gamma_id_from_filename(filename):
    """
    Extract eta, gamma, and id from filename.
    Handles formats like:
    - net_eta-6.748_gamma0.819_ruleMatchingIndex_id130.npy
    - net_eta-6.748_gamma0.819_ruleMatchingIndex.npy (no id -> defaults to 0)
    
    Returns:
        tuple: (eta, gamma, id) where id is 0 if not present
    """
    # Extract eta
    eta_match = re.search(r'eta([-+]?\d*\.?\d+)', filename)
    if not eta_match:
        raise ValueError(f"Could not extract eta from filename: {filename}")
    eta = float(eta_match.group(1))
    
    # Extract gamma
    gamma_match = re.search(r'gamma([-+]?\d*\.?\d+)', filename)
    if not gamma_match:
        raise ValueError(f"Could not extract gamma from filename: {filename}")
    gamma = float(gamma_match.group(1))
    
    # Try to extract id (defaults to 0 if not present)
    id_match = re.search(r'_id(\d+)', filename)
    net_id = int(id_match.group(1)) if id_match else 0
    
    return eta, gamma, net_id


def get_metric_category(metric_name, static_calc, dynamic_calc, computation_calc):
    """Determine which category a metric belongs to."""
    if metric_name in static_calc.implemented_metrics:
        return 'static'
    elif metric_name in dynamic_calc.implemented_metrics:
        return 'dynamic'
    elif metric_name in computation_calc.implemented_metrics:
        return 'computational'
    else:
        return None


def merge_checkpoint_files(output_path, experiment_name, df_original_dict, metric_categories):
    """
    Merge all checkpoint CSV files with the original data and save to final output files.
    Saves separate files for static, dynamic, and computational metrics.
    PRESERVES all existing metric values - only updates NaN or missing values.
    
    Args:
        output_path: Path to the experiment output directory
        experiment_name: Name of the experiment
        df_original_dict: Dict of original DataFrames {'static': df, 'dynamic': df, 'computational': df}
        metric_categories: Dict of metrics by category (used for reference)
    """
    temp_dir = output_path / "temp"
    
    if not temp_dir.exists():
        print("No temp directory found - nothing to merge")
        return df_original_dict
    
    # Collect all checkpoint CSV files by category
    checkpoint_files = {
        'static': sorted(temp_dir.glob('result_static_*.csv')),
        'dynamic': sorted(temp_dir.glob('result_dynamic_*.csv')),
        'computational': sorted(temp_dir.glob('result_computational_*.csv'))
    }
    
    results = {}
    
    for category in ['static', 'dynamic', 'computational']:
        files = checkpoint_files[category]
        
        if not files:
            print(f"No {category} checkpoint files found")
            results[category] = df_original_dict[category]
            continue
        
        print(f"\nMerging {len(files)} {category} checkpoint files...")
        
        # Read all checkpoint files
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"  Loaded {file.name}: {len(df)} rows")
            except Exception as e:
                print(f"  Error reading {file.name}: {e}")
        
        if not dfs:
            print(f"No valid {category} checkpoint files to merge")
            results[category] = df_original_dict[category]
            continue
        
        # Concatenate all checkpoint dataframes
        checkpoint_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(dfs)} {category} checkpoint files into {len(checkpoint_df)} rows")
        
        # Remove duplicates - keep last occurrence
        initial_rows = len(checkpoint_df)
        checkpoint_df = checkpoint_df.drop_duplicates(subset=['eta', 'gamma', 'id'], keep='last')
        duplicates_removed = initial_rows - len(checkpoint_df)
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows from {category} checkpoints")
        
        # Automatically detect metrics from checkpoint data (excluding eta, gamma, id)
        metrics_with_results = [col for col in checkpoint_df.columns if col not in ['eta', 'gamma', 'id']]
        print(f"Detected {len(metrics_with_results)} metrics in {category} checkpoints")
        
        # Merge with original data
        print(f"Merging {category} checkpoint data with original data...")
        df_updated = df_original_dict[category].copy()
        
        # Ensure all metric columns exist in the updated dataframe
        for metric in metrics_with_results:
            if metric not in df_updated.columns:
                df_updated[metric] = np.nan
        
        # Update values from checkpoint data - ONLY for missing/NaN values
        updated_count = 0
        added_count = 0
        preserved_count = 0
        
        for _, result_row in checkpoint_df.iterrows():
            eta = result_row['eta']
            gamma = result_row['gamma']
            net_id = result_row['id'] if 'id' in result_row else 0
            
            # Create mask
            mask = (df_updated['eta'] == eta) & (df_updated['gamma'] == gamma) & (df_updated['id'] == net_id)
            
            if mask.any():
                # Network exists - update ONLY missing/NaN metrics
                for metric in metrics_with_results:
                    if metric in result_row.index and not pd.isna(result_row[metric]):
                        # Check if existing value is NaN or missing
                        existing_value = df_updated.loc[mask, metric].iloc[0]
                        if pd.isna(existing_value):
                            df_updated.loc[mask, metric] = result_row[metric]
                            updated_count += 1
                        else:
                            preserved_count += 1
            else:
                # Network doesn't exist - add new row
                new_row = {'eta': eta, 'gamma': gamma, 'id': net_id}
                
                # Add all existing columns as NaN first
                for col in df_updated.columns:
                    if col not in ['eta', 'gamma', 'id']:
                        new_row[col] = np.nan
                
                # Fill in calculated metrics
                for metric in metrics_with_results:
                    if metric in result_row.index and not pd.isna(result_row[metric]):
                        new_row[metric] = result_row[metric]
                
                df_updated = pd.concat([df_updated, pd.DataFrame([new_row])], ignore_index=True)
                added_count += 1
        
        print(f"✓ Updated {updated_count} missing metric values")
        print(f"✓ Added {added_count} new networks")
        print(f"✓ Preserved {preserved_count} existing metric values")
        
        # Save final merged file
        target_file = output_path / f'all_{category}_metrics_for_{experiment_name}_updated.csv'
        df_updated.to_csv(target_file, index=False)
        print(f"✓ Saved {category} results to: {target_file}")
        print(f"  Final dataframe: {len(df_updated)} rows, {len(df_updated.columns)} columns")
        
        results[category] = df_updated
    
    # Cleanup checkpoint files
    print("\nCleaning up checkpoint files...")
    deleted_count = 0
    for category_files in checkpoint_files.values():
        for file in category_files:
            try:
                file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"  Error deleting {file.name}: {e}")
    
    print(f"✓ Deleted {deleted_count} checkpoint files")
    
    # Delete temp directory if empty
    try:
        if not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            print(f"✓ Deleted empty temp directory")
        else:
            remaining = list(temp_dir.iterdir())
            print(f"⚠ Temp directory not empty, contains: {[f.name for f in remaining]}")
    except Exception as e:
        print(f"  Error deleting temp directory: {e}")
    
    print("\n✓ Merge and cleanup complete!")
    return results


def create_combined_csv(output_path, experiment_name, df_dict):
    """
    Create a combined CSV with all metrics from static, dynamic, and computational categories.
    
    Args:
        output_path: Path to the experiment output directory
        experiment_name: Name of the experiment
        df_dict: Dict of DataFrames {'static': df, 'dynamic': df, 'computational': df}
    """
    print("\n" + "=" * 60)
    print("CREATING COMBINED CSV FILE")
    print("=" * 60)
    
    # Start with static metrics as base
    df_combined = df_dict['static'].copy()
    
    # Merge dynamic metrics
    if not df_dict['dynamic'].empty:
        # Get all columns except eta, gamma, id
        dynamic_cols = [col for col in df_dict['dynamic'].columns if col not in ['eta', 'gamma', 'id']]
        
        if dynamic_cols:
            # Merge on eta, gamma, id
            df_combined = df_combined.merge(
                df_dict['dynamic'][['eta', 'gamma', 'id'] + dynamic_cols],
                on=['eta', 'gamma', 'id'],
                how='outer'
            )
            print(f"✓ Merged {len(dynamic_cols)} dynamic metrics")
    
    # Merge computational metrics
    if not df_dict['computational'].empty:
        # Get all columns except eta, gamma, id
        computational_cols = [col for col in df_dict['computational'].columns if col not in ['eta', 'gamma', 'id']]
        
        if computational_cols:
            # Merge on eta, gamma, id
            df_combined = df_combined.merge(
                df_dict['computational'][['eta', 'gamma', 'id'] + computational_cols],
                on=['eta', 'gamma', 'id'],
                how='outer'
            )
            print(f"✓ Merged {len(computational_cols)} computational metrics")
    
    # Save combined file
    combined_file = output_path / f'all_metrics_for_{experiment_name}_updated.csv'
    df_combined.to_csv(combined_file, index=False)
    
    print(f"✓ Saved combined CSV to: {combined_file}")
    print(f"  Combined dataframe: {len(df_combined)} rows, {len(df_combined.columns)} columns")
    
    return df_combined


def get_missing_work(df_dict, network_files, metric_categories):
    """
    Determine which (network, metric) combinations need to be calculated.
    ONLY calculates metrics that are NaN or missing - preserves existing values.
    
    Returns:
        dict: {network_index: {'static': [...], 'dynamic': [...], 'computational': [...]}}
    """
    missing_work = {}
    
    for idx, name, full_path in network_files:
        eta, gamma, net_id = get_eta_gamma_id_from_filename(name)
        
        network_missing = {
            'static': [],
            'dynamic': [],
            'computational': []
        }
        
        for category in ['static', 'dynamic', 'computational']:
            df = df_dict[category]
            
            # Create mask
            mask = (df['eta'] == eta) & (df['gamma'] == gamma) & (df['id'] == net_id)
            
            if not mask.any():
                # Network not in results - calculate all metrics for this category
                network_missing[category] = metric_categories[category].copy()
            else:
                # Network exists - check which metrics are missing or NaN
                existing_row = df[mask].iloc[0]
                
                for metric in metric_categories[category]:
                    # Check if metric column exists
                    if metric not in df.columns:
                        network_missing[category].append(metric)
                    # Check for metrics that return dicts - look for any column starting with metric name
                    elif any(col.startswith(f"{metric}_") for col in df.columns):
                        # This is a dict-returning metric - check if any of its values are NaN
                        metric_cols = [col for col in df.columns if col.startswith(f"{metric}_")]
                        if any(pd.isna(existing_row[col]) for col in metric_cols):
                            network_missing[category].append(metric)
                    # Regular scalar metric - check if NaN
                    elif pd.isna(existing_row[metric]):
                        network_missing[category].append(metric)
        
        # Only add to missing_work if there's actually work to do
        if any(network_missing.values()):
            missing_work[idx] = network_missing
    
    return missing_work


# Global variable to cache distance matrix per worker process
_worker_distance_matrix = None
_worker_distance_matrix_path = None

def _load_distance_matrix_once(distance_matrix_path):
    """
    Load distance matrix once per worker process and cache it.
    This avoids loading the same matrix for every network.
    """
    global _worker_distance_matrix, _worker_distance_matrix_path
    
    # Only load if not already loaded or if path changed
    if _worker_distance_matrix is None or _worker_distance_matrix_path != distance_matrix_path:
        _worker_distance_matrix = np.load(distance_matrix_path)
        _worker_distance_matrix_path = distance_matrix_path
    
    return _worker_distance_matrix

def convert_to_python_types(obj):
    """
    Recursively convert NumPy types to Python native types for pickling.
    """
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

def process_network_file(file_info, distance_matrix_path, metric_categories, metrics_to_calculate=None):
    """
    Process a single network file and return dictionaries of results by category.
    
    Returns:
        dict: {'static': {...}, 'dynamic': {...}, 'computational': {...}}
    """
    idx, name, full_path = file_info
    
    # If no specific metrics specified, calculate all
    if metrics_to_calculate is None:
        metrics_to_calculate = {
            'static': metric_categories['static'],
            'dynamic': metric_categories['dynamic'],
            'computational': metric_categories['computational']
        }
    
    # Skip if no metrics needed for this network
    total_metrics = sum(len(v) for v in metrics_to_calculate.values())
    if total_metrics == 0:
        return None
    
    try:
        # Load the network
        A = np.load(full_path)[0]
        eta, gamma, net_id = get_eta_gamma_id_from_filename(name)
        
        # Load distance matrix once per worker (cached)
        distance_matrix = _load_distance_matrix_once(distance_matrix_path)
        
        # Initialize calculators with distance matrix
        static_calc = StaticMetricCalculator(A=A, distance_matrix=distance_matrix)
        dynamic_calc = DynamicMetricCalculator(A=A)
        computation_calc = ComputationMetricCalculator(A=A)
        
        # Calculate metrics by category
        results = {
            'static': {"eta": eta, "gamma": gamma, "id": net_id},
            'dynamic': {"eta": eta, "gamma": gamma, "id": net_id},
            'computational': {"eta": eta, "gamma": gamma, "id": net_id}
        }
        
        for category, calculator in [
            ('static', static_calc),
            ('dynamic', dynamic_calc),
            ('computational', computation_calc)
        ]:
            for metric in metrics_to_calculate.get(category, []):
                try:
                    result = calculator.calculate_metric(metric)
                    
                      
                    # Convert to Python native types for pickling
                    result = convert_to_python_types(result)
                    
                    # Handle dict results by unpacking with prefix
                    if isinstance(result, dict):
                        for key, value in result.items():
                            results[category][f"{metric}_{key}"] = value
                    else:
                        results[category][metric] = result
                            
                            
                            
                    # # Handle dict results by unpacking with prefix
                    # if isinstance(result, dict):
                    #     for key, value in result.items():
                    #         # Validate that dict values are scalars
                    #         if isinstance(value, (np.ndarray, list)):
                    #             print(f"Warning: {metric}_{key} for {name} returned array/list, storing as NaN")
                    #             results[category][f"{metric}_{key}"] = np.nan
                    #         else:
                    #             results[category][f"{metric}_{key}"] = value
                    #         # results[category][f"{metric}_{key}"] = value
                    # else:
                    #     # Store scalar result directly
                    #     # results[category][metric] = result
                    #     # Validate that result is a scalar, not an array
                    #     if isinstance(result, (np.ndarray, list)):
                    #         print(f"Warning: {metric} for {name} returned array/list instead of scalar, storing as NaN")
                    #         results[category][metric] = np.nan
                    #     else:
                    #         # Store scalar result directly
                    #         results[category][metric] = result
                            
                    
                        
                except Exception as e:
                    print(f"Error calculating {metric} for {name}: {e}")
                    results[category][metric] = np.nan
        
        # Clean up
        del A, static_calc, dynamic_calc, computation_calc
        gc.collect()
        
        if (idx + 1) % 50 == 0:
            print(f"Processed file {idx+1}: {name} (calculated {total_metrics} metrics)")
        
        return results
    
    except Exception as e:
        print(f"Error processing {name}: {e}")
        return None


def save_results_to_csv(results_list, df_path_out, metric_categories, checkpoint_num):
    """Save accumulated results to checkpoint CSVs by category."""
    if not results_list:
        print("No results to save")
        return
    
    # Organize results by category
    results_by_category = {
        'static': [],
        'dynamic': [],
        'computational': []
    }
    
    for result in results_list:
        for category in ['static', 'dynamic', 'computational']:
            if result[category] and len(result[category]) > 3:  # More than just eta, gamma, id
                results_by_category[category].append(result[category])
    
    # Save each category separately
    for category in ['static', 'dynamic', 'computational']:
        if not results_by_category[category]:
            continue
        
        results_df = pd.DataFrame(results_by_category[category])
        
        checkpoint_path = df_path_out.parent / "temp" / f"result_{category}_{checkpoint_num:04d}.csv"
        os.makedirs(checkpoint_path.parent, exist_ok=True)
        results_df.to_csv(checkpoint_path, index=False)
        print(f"Saved {category} checkpoint to: {checkpoint_path} ({len(results_df)} networks)")


def process_network_with_metrics(args):
    """Wrapper function for starmap."""
    file_info, distance_matrix_path, metric_categories, metrics_to_calculate = args
    return process_network_file(file_info, distance_matrix_path, metric_categories, metrics_to_calculate)


def multiprocess_networks(n_processes=None, 
                          experiment="", 
                          dataset="", 
                          base_path="", 
                          interesting_metrics=None, 
                          save_interval=50,
                          debug=False,
                          create_big_update_csv=False):
    """
    Main function to multiprocess network metric calculations.
    Saves results split by metric category (static, dynamic, computational).
    PRESERVES existing metric values - only calculates missing ones.
    
    Args:
        create_big_update_csv: If True, creates a combined CSV with all metrics
    """
    if not ONLY_SUMMARIZE: 
        global accumulated_results
        accumulated_results = []
        checkpoint_counter = 0
        
    # # Define paths
    output_path = base_path / "output" / "gnm" / dataset / experiment
    generated_networks_dir = output_path / "generated_networks"
    distance_matrix_path = base_path / "data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy"
    
    # # Use metric categories directly
    print("Using provided metric categories...")
    metric_categories = interesting_metrics
    print(f"Static metrics: {metric_categories['static']}")
    print(f"Dynamic metrics: {metric_categories['dynamic']}")
    print(f"Computational metrics: {metric_categories['computational']}")
    
    # # Get all network files
    if not ONLY_SUMMARIZE: 
        all_files = sorted_listing_by_creation_time(generated_networks_dir)
        network_files = [(i, name, generated_networks_dir / name) 
                         for i, name in enumerate(all_files) 
                         if name.endswith(".npy")]
        
        print(f"\nFound {len(network_files)} network files")
        
        if debug:
            print("Debug mode: only use first 5000 networks.")
            network_files = network_files[:5000]
    
    # Load existing results for each category
    df_dict = {}
    for category in ['static', 'dynamic', 'computational']:
        df_path = output_path / f"all_{category}_metrics_for_{experiment}_updated.csv"
        
        # Check if updated version exists, otherwise try original
        if df_path.exists():
            df_dict[category] = pd.read_csv(df_path)
            print(f"Loaded existing {category} results with {len(df_dict[category])} rows")
        else:
            # Try loading original file
            df_path_original = output_path / f"all_{category}_metrics_for_{experiment}.csv"
            if df_path_original.exists():
                df_dict[category] = pd.read_csv(df_path_original)
                print(f"Loaded original {category} results with {len(df_dict[category])} rows")
            else:
                # Create empty dataframe with mandatory columns
                df_dict[category] = pd.DataFrame(columns=['eta', 'gamma', 'id'])
                print(f"No existing {category} results - starting fresh")
    
    # # Determine what work needs to be done
    if not ONLY_SUMMARIZE:
        print("\nAnalyzing which metrics need to be calculated...")
        missing_work = get_missing_work(df_dict, network_files, metric_categories)
        
        total_calculations = sum(
            sum(len(metrics) for metrics in work.values())
            for work in missing_work.values()
        )
        print(f"Found {len(missing_work)} networks with missing metrics")
        print(f"Total metric calculations needed: {total_calculations}")
        
        if total_calculations == 0:
            print("✓ All metrics already calculated!")
            
            # Create combined CSV if requested
            if create_big_update_csv:
                create_combined_csv(output_path, experiment, df_dict)
            
            return df_dict
        
        # Filter network files to only those needing work
        network_files_filtered = [(idx, name, path) for idx, name, path in network_files 
                                if idx in missing_work]
        
        # Determine number of processes
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)
        print(f"Using {n_processes} processes")
        
        print("\nStarting multiprocessing...")
        print("-" * 60)
        
        try:
            with Pool(processes=n_processes, maxtasksperchild=50) as pool:
                tasks = []
                for idx, name, path in network_files_filtered:
                    tasks.append((
                        (idx, name, path),
                        distance_matrix_path,
                        metric_categories,
                        missing_work[idx]
                    ))
                
                for i, result in enumerate(pool.imap(
                    process_network_with_metrics,
                    tasks,
                    chunksize=20
                )):
                    if result is not None:
                        accumulated_results.append(result)
                    
                    if (i + 1) % save_interval == 0:
                        save_results_to_csv(accumulated_results, 
                                        output_path / f"all_metrics_for_{experiment}_updated.csv",
                                        metric_categories, 
                                        checkpoint_counter)
                        checkpoint_counter += 1
                        accumulated_results = []
                        print(f"Progress: {i+1}/{len(network_files_filtered)} files processed")
                        print(f"Memory checkpoint - forcing garbage collection\n")
                        gc.collect()
            
            # Final save
            if accumulated_results:
                print("\n" + "=" * 60)
                print("Saving final results...")
                save_results_to_csv(accumulated_results,
                                output_path / f"all_metrics_for_{experiment}_updated.csv",
                                metric_categories,
                                checkpoint_counter)
            
        except KeyboardInterrupt:
            if accumulated_results:
                print("\n\nInterrupted! Saving progress...")
                save_results_to_csv(accumulated_results,
                                output_path / f"all_metrics_for_{experiment}_updated.csv",
                                metric_categories,
                                checkpoint_counter)
            gc.collect()
        
        finally:
            gc.collect()
    
    # Merge all checkpoint files into final outputs
    print("\n" + "=" * 60)
    print("MERGING CHECKPOINT FILES")
    print("=" * 60)
    
    final_dfs = merge_checkpoint_files(output_path, experiment, df_dict, metric_categories)
    
    if final_dfs is not None:
        print(f"\n✓ Processing complete!")
        for category in ['static', 'dynamic', 'computational']:
            print(f"✓ {category.capitalize()} results: {len(final_dfs[category])} networks")
        
        # Create combined CSV if requested
        if create_big_update_csv:
            create_combined_csv(output_path, experiment, final_dfs)
    else:
        print("\n⚠ No checkpoint files were created")
    
    return final_dfs


if __name__ == "__main__":
    
    EXPERIMENT = "76_90000_samples_animal_206"
    # "75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_0" # 75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206" # 71_testing_animal_0"

    DATASET = "suarez_MaMI_dataset"
    base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
    
    CREATE_BIG_UPDATE_CSV = True  # Set to True to create combined CSV file
    
    # Define metrics to calculate
    interesting_metrics = {
        "static": [
            # "density", # works # done
            # "avg_clustering", # works # done
            # "avg_degree", # works # done
            # "degree_assortativity", # works # done
            # "modularity", # works # done
            # "characteristic_path_length", # works # done
            # "transitivity", # works # done
            
            "wiring_cost", # works
            # "shortest_path_distance", # works
            # "structural_complexity", # works
            # "n_connected_components", # works
            
            # "omega", # works
            # "topological_distance", # works
            # # "resistance_distance", # DOES NOT WORK!! 
            # "degree_gini", # works
        ], 
        "dynamic": [
            # "spectral_radius", # works
            # "spectral_gap", # works
            # "spectral_gap_fatemeh", # works?
            # "global_efficiency", # works
            # "diffusion_efficiency", # -> Returns 0 if unconnected nodes exist: Error calculating diffusion_efficiency for net_eta3.5_gamma1.0_ruleMatchingIndex_id017.npy: Array must not contain infs or NaNs
            # "propagation_efficiency", # works
            "nct_control",  # works
            "nct_energies", # works
            # "metastability", # works
            # "synchronizability_eigenratio", # works 
            # "algebraic_connectivity_nx", # works
            # "kuramoto_synchronization", # works. But takes ages (maybe 3 hours for 11,000 networks?)
            # "community_synchronization_vulnerability", # works 
        ],
        "computational": [
            
            # "kernel_rank", # works
            # "kernel_rank_fatemeh", # -> Lots of Runtime warnings in the echoes part of it (generate_esn_open) -> switch to pinv instad of ridge?: /opt/miniconda3/envs/ma_thesis/lib/python3.13/site-packages/sklearn/linear_model/_ridge.py:252: UserWarning: Singular matrix in solving dual problem. Using least-squares solution instead.
            # "effective_dimensionality", # works
            # "multifunctionality", # works?? - or does at least produce values??
        ]
    }

    # Run the multiprocessing
    df_final = multiprocess_networks(n_processes=12, 
                                     experiment=EXPERIMENT, 
                                     dataset=DATASET, 
                                     base_path=base_path,
                                     interesting_metrics=interesting_metrics,
                                     save_interval=50, 
                                     debug=False,
                                     create_big_update_csv=CREATE_BIG_UPDATE_CSV)
    
    if df_final is not None:
        print("\n✓ All done!")
        for category in ['static', 'dynamic', 'computational']:
            if category in df_final:
                print(f"✓ {category.capitalize()} dataframe shape: {df_final[category].shape}")
    else:
        print("\n✓ Processing complete (no new data to merge)")
    
    # Final cleanup
    gc.collect()