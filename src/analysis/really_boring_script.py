import numpy as np 
import networkx as nx
import pandas as pd
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import signal
import sys
import gc  # Garbage collector

# from src.utils.combine_csvs import merge_csv_files 

from src.analysis.metric_calculators import StaticMetricCalculator, DynamicMetricCalculator, ComputationMetricCalculator     
    
from src.analysis.utils import get_eta_and_gamma_from_filename, sorted_listing_by_creation_time


# def merge_checkpoint_files(output_path, experiment_name):
#     """
#     Merge all checkpoint CSV files into the final output file and clean up.
    
#     Args:
#         output_path: Path to the experiment output directory
#         experiment_name: Name of the experiment
#     """
#     # Define paths
#     temp_dir = output_path / "temp"
#     target_file = output_path / f'all_metrics_for_{experiment_name}_updated.csv'
    
#     # Check if temp directory exists
#     if not temp_dir.exists():
#         print("No temp directory found - nothing to merge")
#         return
    
#     # Collect all checkpoint CSV files
#     checkpoint_files = sorted(temp_dir.glob('result_*.csv'))
    
#     if not checkpoint_files:
#         print("No checkpoint files found to merge")
#         return
    
#     print(f"\nMerging {len(checkpoint_files)} checkpoint files...")
    
#     # Read all checkpoint files
#     dfs = []
#     for file in checkpoint_files:
#         try:
#             df = pd.read_csv(file)
#             dfs.append(df)
#             print(f"  Loaded {file.name}: {len(df)} rows")
#         except Exception as e:
#             print(f"  Error reading {file.name}: {e}")
    
#     if not dfs:
#         print("No valid checkpoint files to merge")
#         return
    
#     # Concatenate all dataframes
#     merged_df = pd.concat(dfs, ignore_index=True)
#     print(f"\nCombined {len(dfs)} checkpoint files into {len(merged_df)} rows")
    
#     # Remove duplicates - keep last occurrence (most recent calculation)
#     # Identify duplicates based on eta and gamma
#     initial_rows = len(merged_df)
#     merged_df = merged_df.drop_duplicates(subset=['eta', 'gamma'], keep='last')
#     duplicates_removed = initial_rows - len(merged_df)
    
#     if duplicates_removed > 0:
#         print(f"Removed {duplicates_removed} duplicate rows")
    
#     # Save final merged file
#     merged_df.to_csv(target_file, index=False)
#     print(f"✓ Saved final results to: {target_file}")
#     print(f"  Final dataframe: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
#     # Delete checkpoint files
#     print("\nCleaning up checkpoint files...")
#     deleted_count = 0
#     for file in checkpoint_files:
#         try:
#             file.unlink()
#             deleted_count += 1
#         except Exception as e:
#             print(f"  Error deleting {file.name}: {e}")
    
#     print(f"✓ Deleted {deleted_count} checkpoint files")
    
#     # Delete temp directory if empty
#     try:
#         if not any(temp_dir.iterdir()):
#             temp_dir.rmdir()
#             print(f"✓ Deleted empty temp directory")
#         else:
#             remaining = list(temp_dir.iterdir())
#             print(f"⚠ Temp directory not empty, contains: {[f.name for f in remaining]}")
#     except Exception as e:
#         print(f"  Error deleting temp directory: {e}")
    
#     print("\n✓ Merge and cleanup complete!")
#     return merged_df
def merge_checkpoint_files(output_path, experiment_name, df_original, interesting_metrics):
    """
    Merge all checkpoint CSV files with the original data and save to final output file.
    Only updates metrics that were calculated in this run, preserving other existing values.
    
    Args:
        output_path: Path to the experiment output directory
        experiment_name: Name of the experiment
        df_original: Original DataFrame loaded at the start
        interesting_metrics: List of metrics that were calculated in this run
    """
    # Define paths
    temp_dir = output_path / "temp"
    target_file = output_path / f'all_metrics_for_{experiment_name}_updated.csv'
    
    # Check if temp directory exists
    if not temp_dir.exists():
        print("No temp directory found - nothing to merge")
        return df_original
    
    # Collect all checkpoint CSV files
    checkpoint_files = sorted(temp_dir.glob('result_*.csv'))
    
    if not checkpoint_files:
        print("No checkpoint files found to merge")
        return df_original
    
    print(f"\nMerging {len(checkpoint_files)} checkpoint files...")
    
    # Read all checkpoint files
    dfs = []
    for file in checkpoint_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  Loaded {file.name}: {len(df)} rows")
        except Exception as e:
            print(f"  Error reading {file.name}: {e}")
    
    if not dfs:
        print("No valid checkpoint files to merge")
        return df_original
    
    # Concatenate all checkpoint dataframes
    checkpoint_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined {len(dfs)} checkpoint files into {len(checkpoint_df)} rows")
    
    # Remove duplicates in checkpoint data - keep last occurrence (most recent calculation)
    initial_rows = len(checkpoint_df)
    checkpoint_df = checkpoint_df.drop_duplicates(subset=['eta', 'gamma'], keep='last')
    duplicates_removed = initial_rows - len(checkpoint_df)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows from checkpoints")
    
    # Now merge with original data
    print(f"\nMerging checkpoint data with original data...")
    df_updated = df_original.copy()
    
    # Ensure all metric columns exist in updated df
    for metric in interesting_metrics:
        if metric not in df_updated.columns:
            df_updated[metric] = np.nan
    
    # Update values from checkpoint data
    updated_count = 0
    added_count = 0
    
    for _, result_row in checkpoint_df.iterrows():
        eta = result_row['eta']
        gamma = result_row['gamma']
        
        mask = (df_updated['eta'] == eta) & (df_updated['gamma'] == gamma)
        
        if mask.any():
            # Network exists - update only the calculated metrics
            for metric in interesting_metrics:
                if metric in result_row.index and not pd.isna(result_row[metric]):
                    df_updated.loc[mask, metric] = result_row[metric]
            updated_count += 1
        else:
            # Network doesn't exist - add new row
            new_row = {col: np.nan for col in df_updated.columns}
            new_row['eta'] = eta
            new_row['gamma'] = gamma
            
            for metric in interesting_metrics:
                if metric in result_row.index and not pd.isna(result_row[metric]):
                    new_row[metric] = result_row[metric]
            
            df_updated = pd.concat([df_updated, pd.DataFrame([new_row])], ignore_index=True)
            added_count += 1
    
    print(f"✓ Updated {updated_count} existing networks")
    print(f"✓ Added {added_count} new networks")
    
    # Save final merged file
    df_updated.to_csv(target_file, index=False)
    print(f"✓ Saved final results to: {target_file}")
    print(f"  Final dataframe: {len(df_updated)} rows, {len(df_updated.columns)} columns")
    
    # Delete checkpoint files
    print("\nCleaning up checkpoint files...")
    deleted_count = 0
    for file in checkpoint_files:
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
    return df_updated

def get_missing_work(df_existing, network_files, interesting_metrics):
    """
    Determine which (network, metric) combinations need to be calculated.
    
    Returns:
        dict: {network_index: [list of metrics to calculate]}
    """
    missing_work = {}
    
    # Check which metrics are completely missing (need to calculate for all)
    metrics_to_calc_for_all = [m for m in interesting_metrics if m not in df_existing.columns]
    
    # For each network file, determine what needs calculating
    for idx, name, full_path in network_files:
        eta, gamma = get_eta_and_gamma_from_filename(name)
        
        # Find if this network exists in results
        mask = (df_existing['eta'] == eta) & (df_existing['gamma'] == gamma)
        
        if not mask.any():
            # Network not in results at all - calculate all metrics
            missing_work[idx] = interesting_metrics.copy()
        else:
            # Network exists - check which metrics are missing or NaN
            existing_row = df_existing[mask].iloc[0]
            metrics_needed = metrics_to_calc_for_all.copy()
            
            for metric in interesting_metrics:
                if metric in df_existing.columns:
                    # Check if value is NaN or missing
                    if pd.isna(existing_row[metric]):
                        metrics_needed.append(metric)
                # If not in columns, already in metrics_to_calc_for_all
            
            if metrics_needed:
                missing_work[idx] = metrics_needed
    
    return missing_work


def process_network_file(file_info, distance_matrix_path, interesting_metrics, metrics_to_calculate=None):
    """
    Process a single network file and return a dictionary of results.
    
    Args:
        file_info: tuple of (index, filename, full_path)
        distance_matrix_path: path to distance matrix (loaded per worker to save memory)
        interesting_metrics: list of all possible metric names
        metrics_to_calculate: list of metrics to actually calculate for this network (subset of interesting_metrics)
    
    Returns:
        dict with eta, gamma, and all calculated metrics
    """
    idx, name, full_path = file_info
    
    # If no specific metrics specified, calculate all
    if metrics_to_calculate is None:
        metrics_to_calculate = interesting_metrics
    
    # Skip if no metrics needed for this network
    if not metrics_to_calculate:
        return None
    
    try:
        # Load the network
        A = np.load(full_path)[0]  # Load the numpy array (mostly: 1, 100, 100)
        eta, gamma = get_eta_and_gamma_from_filename(name)
        
        # Initialize calculator
        static_metric_calculator = StaticMetricCalculator(A=A, distance_matrix=None)
        dynamic_metric_calculator = DynamicMetricCalculator(A=A)
        computation_metric_calculator = ComputationMetricCalculator(A=A)
        
        # Calculate only the needed metrics
        results = {"eta": eta, "gamma": gamma}
        for metric in metrics_to_calculate:
            if metric in static_metric_calculator.implemented_metrics: 
                results[metric] = static_metric_calculator.calculate_metric(metric)
                
            elif metric in dynamic_metric_calculator.implemented_metrics:
                results[metric] = dynamic_metric_calculator.calculate_metric(metric)

            elif metric in computation_metric_calculator.implemented_metrics:
                results[metric] = computation_metric_calculator.calculate_metric(metric)

            else:
                results[metric] = np.nan  # Metric not implemented
                print("Metric not implemented:", metric)

        # Clean up
        del A
        del static_metric_calculator
        del dynamic_metric_calculator
        del computation_metric_calculator
        gc.collect()

        if (idx + 1) % 50 == 0:
            print(f"Processed file {idx+1}: {name} (calculated {len(metrics_to_calculate)} metrics)")
        
        return results
    
    except Exception as e:
        print(f"Error processing {name}: {e}")
        return None


def save_results_to_csv(results_list, df_original, df_path_out, interesting_metrics, checkpoint_num):
    """Save accumulated results to a checkpoint CSV WITHOUT merging with existing data."""
    if not results_list:
        print("No results to save")
        return  # Changed: don't return df_original
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save directly as checkpoint file WITHOUT merging
    checkpoint_path = df_path_out.parent / "temp" / f"result_{checkpoint_num:04d}.csv"
    os.makedirs(checkpoint_path.parent, exist_ok=True)
    results_df.to_csv(checkpoint_path, index=False)  # Changed: save results_df directly
    print(f"Saved checkpoint to: {checkpoint_path} ({len(results_df)} networks)")
    

def signal_handler(signum, frame, df_path_out, interesting_metrics, checkpoint_counter):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Interrupt received! Saving progress before exiting...")
    if accumulated_results:
        save_results_to_csv(accumulated_results, None, df_path_out, interesting_metrics, checkpoint_counter)  # Changed: pass None for df_original
    print("✅ Progress saved. Exiting.")
    gc.collect()
    print("Final path: ", df_path_out)
    sys.exit(0)

def merge_results_with_existing(results_df, df_existing, interesting_metrics):
    """
    Merge new results into existing dataframe, only updating missing values.
    
    Args:
        results_df: DataFrame with new calculated results
        df_existing: Existing results DataFrame
        interesting_metrics: List of metric columns
    
    Returns:
        Updated DataFrame
    """
    df_updated = df_existing.copy()
    
    # Ensure all metric columns exist
    for metric in interesting_metrics:
        if metric not in df_updated.columns:
            df_updated[metric] = np.nan
    
    # Update each result
    for _, result_row in results_df.iterrows():
        eta = result_row['eta']
        gamma = result_row['gamma']
        
        mask = (df_updated['eta'] == eta) & (df_updated['gamma'] == gamma)
        
        if mask.any():
            # Update only the metrics that were calculated
            for metric in interesting_metrics:
                if metric not in result_row.index:
                    continue
                    
                value = result_row[metric]
                
                # Convert to scalar if needed
                if isinstance(value, (list, np.ndarray)):
                    if len(value) == 0:
                        continue
                    value = float(value[0]) if len(value) > 0 else np.nan
                elif isinstance(value, pd.Series):
                    if len(value) == 0:
                        continue
                    value = float(value.iloc[0])
                
                # Skip if NaN
                if pd.isna(value):
                    continue
                    
                df_updated.loc[mask, metric] = value
        else:
            # Add new row
            new_row = {col: np.nan for col in df_updated.columns}
            new_row['eta'] = eta
            new_row['gamma'] = gamma
            
            for metric in interesting_metrics:
                if metric not in result_row.index:
                    continue
                    
                value = result_row[metric]
                
                # Convert to scalar if needed
                if isinstance(value, (list, np.ndarray)):
                    if len(value) == 0:
                        continue
                    value = float(value[0]) if len(value) > 0 else np.nan
                elif isinstance(value, pd.Series):
                    if len(value) == 0:
                        continue
                    value = float(value.iloc[0])
                
                # Skip if NaN
                if not pd.isna(value):
                    new_row[metric] = value
                    
            df_updated = pd.concat([df_updated, pd.DataFrame([new_row])], ignore_index=True)
    
    return df_updated

def process_network_with_metrics(args):
    """
    Wrapper function for starmap that unpacks arguments.
    This needs to be a module-level function (not lambda) to be picklable.
    """
    file_info, distance_matrix_path, interesting_metrics, metrics_to_calculate = args
    return process_network_file(file_info, distance_matrix_path, interesting_metrics, metrics_to_calculate)



def multiprocess_networks(n_processes=None, 
                          experiment="", 
                          dataset="", 
                          base_path="", 
                          interesting_metrics=[], 
                          save_interval=50,
                          debug=False):
    """
    Main function to multiprocess network metric calculations.
    Only calculates missing metrics for each network.
    """
    global accumulated_results
    accumulated_results = []
    checkpoint_counter = 0
    
    # Define paths and load data 
    output_path = base_path / "output" / "gnm" / dataset / experiment
    generated_networks_dir = output_path / "generated_networks"

    df_path = output_path / f"all_metrics_for_{experiment}.csv"
    df_path_out = output_path / f"all_metrics_for_{experiment}_updated.csv"

    distance_matrix_path = base_path / "data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy"

    # Get all network files
    all_files = sorted_listing_by_creation_time(generated_networks_dir)
    network_files = [(i, name, generated_networks_dir / name) 
                     for i, name in enumerate(all_files) 
                     if name.endswith(".npy")]
    
    print(f"Found {len(network_files)} network files")
    
    # Load existing results
    df_original = pd.read_csv(df_path)
    print(f"Loaded existing results with {len(df_original)} rows")
    if debug:
        print("Debug mode: only use first 5000 networks.")
        network_files = network_files[:5000]
    
    # Determine what work needs to be done
    print("Analyzing which metrics need to be calculated...")
    missing_work = get_missing_work(df_original, network_files, interesting_metrics)
    
    total_calculations = sum(len(metrics) for metrics in missing_work.values())
    print(f"Found {len(missing_work)} networks with missing metrics")
    print(f"Total metric calculations needed: {total_calculations}")
    
    if total_calculations == 0:
        print("✓ All metrics already calculated!")
        return df_original
    
    # Filter network files to only those needing work
    network_files_filtered = [(idx, name, path) for idx, name, path in network_files 
                              if idx in missing_work]
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    print(f"Using {n_processes} processes")
    
    # Set up interrupt handler - CHANGED: removed df_original parameter
    handler = partial(signal_handler, 
                     df_path_out=df_path_out, 
                     interesting_metrics=interesting_metrics,
                     checkpoint_counter=checkpoint_counter)
    signal.signal(signal.SIGINT, handler)
    
    print("Starting multiprocessing... (Press Ctrl+C to stop and save progress)")
    print("-" * 60)
    
    try:
        with Pool(processes=n_processes) as pool:
            # Create tasks with all necessary arguments
            tasks = []
            for idx, name, path in network_files_filtered:
                tasks.append((
                    (idx, name, path),
                    distance_matrix_path,
                    interesting_metrics,
                    missing_work[idx]
                ))
            
            # Use map with the wrapper function
            for i, result in enumerate(pool.imap(
                process_network_with_metrics,
                tasks,
                chunksize=20
            )):
                if result is not None:
                    accumulated_results.append(result)
                
                if (i + 1) % save_interval == 0:
                    # CHANGED: Don't capture return value, pass None for df_original
                    save_results_to_csv(accumulated_results, None, 
                                       df_path_out, interesting_metrics, checkpoint_counter)
                    checkpoint_counter += 1
                    accumulated_results = []  # Clear results after saving
                    print(f"Progress: {i+1}/{len(network_files_filtered)} files processed")
                    print(f"Memory checkpoint - forcing garbage collection\n")
                    gc.collect()
        
        # Final save - CHANGED: Don't capture return value
        if accumulated_results:
            print("\n" + "=" * 60)
            print("Saving final results...")
            save_results_to_csv(accumulated_results, None, 
                               df_path_out, interesting_metrics, checkpoint_counter)
        
    except KeyboardInterrupt:
        if accumulated_results:
            print("\n\nInterrupted! Saving progress...")
            save_results_to_csv(accumulated_results, None, 
                               df_path_out, interesting_metrics, checkpoint_counter)
        gc.collect() 
    
    finally:
        gc.collect()
    
    # Merge all checkpoint files into final output
    print("\n" + "=" * 60)
    print("MERGING CHECKPOINT FILES")
    print("=" * 60)
    
    final_df = merge_checkpoint_files(output_path, experiment, df_original, interesting_metrics)
    # final_df = merge_checkpoint_files(output_path, experiment)


    if final_df is not None:
        print(f"\n✓ Processing complete!")
        print(f"✓ Final results saved to: {df_path_out}")
        print(f"✓ Total networks in final file: {len(final_df)}")
    else:
        print("\n⚠ No checkpoint files were created (all metrics may have been up to date)")
    
    return final_df


if __name__ == "__main__":
    
    EXPERIMENT = "60_generally_finer_search_animal_0"
    DATASET = "suarez_MaMI_dataset"
    base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")

    # Define metrics to calculate
    interesting_metrics = [
                            # "density", 
                            # "compute_structural_complexity", 
                            # "n_connected_components", 
                            # "propagation_efficiency",
                            # "spectral_radius",
                            # "degree_assortativity",
                            # "global_efficiency",
                            
                            # "kernel_rank",
                            # "effective_dimensionality",
                            # "multifunctionality",
                            
                            # "average_controllability",
                            # "spectral_gap",
                            # "spectral_radius",
                            
                            "spectral_radius",
                            "spectral_gap",
                            "global_efficiency",
                            # "diffusion_efficiency",
                            "propagation_distance",
                            "propagation_efficiency",
                            "average_controllability",
                            
                            # "diffusion_efficiency",  # -> does not work -> throws error.  
                           ]
    
    # Run the multiprocessing
    df_final = multiprocess_networks(n_processes=12,
                                     experiment=EXPERIMENT, 
                                     dataset=DATASET, 
                                     base_path=base_path,
                                     interesting_metrics=interesting_metrics,
                                     save_interval=50, 
                                     debug=True)
    
    if df_final is not None:
        print("\n✓ All done!")
        print(f"✓ Final dataframe shape: {df_final.shape}")
    else:
        print("\n✓ Processing complete (no new data to merge)")
    
    # Final cleanup
    gc.collect()