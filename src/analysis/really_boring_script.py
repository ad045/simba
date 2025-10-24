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

from src.analysis.metric_calculators import StaticMetricCalculator               
    
from src.analysis.utils import get_eta_and_gamma_from_filename, sorted_listing_by_creation_time


# Worker function to process a single network file
def process_network_file(file_info, distance_matrix_path, interesting_metrics):
    """
    Process a single network file and return a dictionary of results.
    
    Args:
        file_info: tuple of (index, filename, full_path)
        distance_matrix_path: path to distance matrix (loaded per worker to save memory)
        interesting_metrics: list of metric names to calculate
    
    Returns:
        dict with eta, gamma, and all calculated metrics
    """
    idx, name, full_path = file_info
    
    try:
        # Load the network
        A = np.load(full_path)[0]  # Load the numpy array (mostly: 1, 100, 100)
        eta, gamma = get_eta_and_gamma_from_filename(name)
        
        # Load distance matrix only if needed (currently not used by metrics)
        # distance_matrix = np.load(distance_matrix_path)
        
        # Initialize calculator
        static_metric_calculator = StaticMetricCalculator(A=A, distance_matrix=None)
        
        # Calculate all metrics
        results = {"eta": eta, "gamma": gamma}
        for metric in interesting_metrics:
            results[metric] = static_metric_calculator.calculate_metric(metric)
        
        # Clean up
        del A
        del static_metric_calculator
        gc.collect()  # Force garbage collection
        
        if (idx + 1) % 50 == 0:  # Print every 50 files
            print(f"Processed file {idx+1}: {name}")
        
        return results
    
    except Exception as e:
        print(f"Error processing {name}: {e}")
        return None


def save_results_to_csv(results_list, df_original, df_path_out, interesting_metrics):
    """Save accumulated results to CSV"""
    if not results_list:
        print("No results to save")
        return df_original
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    
    print(f"\nSaving {len(results_df)} processed networks...")
    
    # Create a copy of the original dataframe
    df_updated = df_original.copy()
    
    # Ensure metric columns exist
    for metric in interesting_metrics:
        if metric not in df_updated.columns:
            df_updated[metric] = np.nan
    
    # Update rows that match eta and gamma
    for _, result_row in results_df.iterrows():
        eta = result_row['eta']
        gamma = result_row['gamma']
        
        # Find matching rows in original dataframe
        mask = (df_updated['eta'] == eta) & (df_updated['gamma'] == gamma)
        
        if mask.any():
            # Update existing row
            for metric in interesting_metrics:
                if metric in result_row:
                    df_updated.loc[mask, metric] = result_row[metric]
        else:
            # Add new row if it doesn't exist
            new_row = {col: np.nan for col in df_updated.columns}
            new_row['eta'] = eta
            new_row['gamma'] = gamma
            for metric in interesting_metrics:
                if metric in result_row:
                    new_row[metric] = result_row[metric]
            df_updated = pd.concat([df_updated, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the updated dataframe
    df_updated.to_csv(df_path_out, index=False)
    print(f"Saved to: {df_path_out}")
    
    return df_updated


def signal_handler(signum, frame, df_original, df_path_out, interesting_metrics):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Interrupt received! Saving progress before exiting...")
    save_results_to_csv(accumulated_results, df_original, df_path_out, interesting_metrics)
    print("✓ Progress saved. Exiting.")
    sys.exit(0)


# %%
def multiprocess_networks(n_processes=None, 
                          experiment="", 
                          dataset="", 
                          base_path="", 
                          interesting_metrics=[], 
                          save_interval=50):
    """
    Main function to multiprocess network metric calculations.
    
    Args:
        n_processes: Number of processes to use. If None, uses cpu_count() - 1
        save_interval: Save progress every N completed files
    """
    # Global list to accumulate results
    global accumulated_results
    accumulated_results = []
    
    # Define paths and load data 
    output_path = base_path / "output" / "gnm" / dataset / experiment
    generated_networks_dir = output_path / "generated_networks"

    df_path = output_path / f"all_metrics_for_{experiment}.csv"
    df = pd.read_csv(df_path)

    df_path_out = output_path / f"all_metrics_for_{experiment}_updated.csv"

    distance_matrix_path = base_path / "data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy"
    distance_matrix = np.load(distance_matrix_path)

    # Get all network files in order
    all_files = sorted_listing_by_creation_time(generated_networks_dir)
    network_files = [(i, name, generated_networks_dir / name) 
                     for i, name in enumerate(all_files) 
                     if name.endswith(".npy")]
    
    print(f"Found {len(network_files)} network files to process")
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    print(f"Using {n_processes} processes")
    
    # Load original dataframe
    df_original = pd.read_csv(df_path)
    
    # Set up interrupt handler with partial function
    handler = partial(signal_handler, 
                     df_original=df_original, 
                     df_path_out=df_path_out, 
                     interesting_metrics=interesting_metrics)
    signal.signal(signal.SIGINT, handler)
    
    # Create partial function - pass PATH not the matrix itself to save memory
    worker_func = partial(process_network_file, 
                         distance_matrix_path=distance_matrix_path,
                         interesting_metrics=interesting_metrics)
    
    # Process networks in parallel
    print("Starting multiprocessing... (Press Ctrl+C to stop and save progress)")
    print("-" * 60)
    
    try:
        with Pool(processes=n_processes) as pool:
            # Use imap to get results as they complete (preserves order)
            for i, result in enumerate(pool.imap(worker_func, network_files, chunksize=20)):
                if result is not None:
                    accumulated_results.append(result)
                
                # Save periodically and clear memory
                if (i + 1) % save_interval == 0:
                    df_updated = save_results_to_csv(accumulated_results, df_original, 
                                                     df_path_out, interesting_metrics)
                    print(f"Progress: {i+1}/{len(network_files)} files processed")
                    print(f"Memory checkpoint - forcing garbage collection\n")
                    gc.collect()  # Force garbage collection
        
        # Final save
        print("\n" + "=" * 60)
        print("All processing complete!")
        df_updated = save_results_to_csv(accumulated_results, df_original, 
                                        df_path_out, interesting_metrics)
        
    except KeyboardInterrupt:
        # This shouldn't normally trigger due to signal handler, but just in case
        print("\n\nInterrupted! Saving progress...")
        df_updated = save_results_to_csv(accumulated_results, df_original, 
                                        df_path_out, interesting_metrics)
        gc.collect() 
    
    finally:
        # Clean up
        gc.collect()
    
    print(f"\nFinal results: Processed {len(accumulated_results)} networks")
    print(f"Output saved to: {df_path_out}")
    
    return df_updated


# %%
if __name__ == "__main__":
    
    EXPERIMENT = "60_generally_finer_search_animal_0"
    DATASET = "suarez_MaMI_dataset"
    base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")

    # Define metrics to calculate
    interesting_metrics = ["density", 
                           "compute_structural_complexity", 
                           "n_connected_components"
                           ]
    
    # Run the multiprocessing
    df_updated = multiprocess_networks(n_processes=12, # None, 
                                       experiment = EXPERIMENT, 
                                       dataset = DATASET, 
                                       base_path=base_path,
                                       interesting_metrics=interesting_metrics,
                                       save_interval=50)
    print("\n✓ Processing complete!")
    print(f"Updated dataframe shape: {df_updated.shape}")
    
    # Final cleanup
    gc.collect()
