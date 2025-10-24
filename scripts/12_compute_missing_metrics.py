#!/usr/bin/env python3
"""
Batch process network files and compute graph metrics.
Writes results incrementally to CSV and is fully resumable.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Import the analysis functions
from src.analysis.kayson_utils import (check_density, 
                                       calculate_wiring_cost, 
                                       compute_omega, 
                                       compute_structural_complexity)

from src.analysis.structural_measures import analyze_connectomes

from scripts.utils_12_file_handling import parse_filename, find_network_files 

def compute_metrics_for_network(
    network_path: str,
    distance_matrix: np.ndarray,
    metrics_to_compute: List[str],
) -> Dict[str, float]:
    """Compute specified metrics for a single network."""
    
    # Load network
    try:
        A = np.load(network_path)
    except Exception as e:
        print(f"Error loading {network_path}: {e}")
        return None
    
    # Initialize results with filename parameters and metadata
    out = parse_filename(network_path)
    
    # Suppress warnings during computation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        
        # First compute analyze_connectomes metrics (many come from here)
        # try: 
            # conn_results = analyze_connectomes(
            #     A,
            #     distance_matrix,
            #     comm_mode="estrada_scaled"
            # )
            # conn_data = conn_results[0]  # First network results
            
            # Map connectome results to output columns
            # out['avg_communicability'] = conn_data.get('avg_communicability', np.nan)
            # out['global_efficiency'] = conn_data.get('global_efficiency', np.nan)
            # out['modularity'] = conn_data.get('modularity', np.nan)
            # out['avg_clustering'] = conn_data.get('avg_clustering', np.nan)
            # out['avg_degree'] = conn_data.get('avg_degree', np.nan)
            # out['density_bct'] = conn_data.get('density_bct', np.nan)
            # out['transitivity'] = conn_data.get('transitivity', np.nan)
            # out['avg_edge_distance'] = conn_data.get('avg_edge_distance', np.nan)
            # out['char_path_length'] = conn_data.get('char_path_length', np.nan)
            # out['richclub_n_edges'] = conn_data.get('richclub_n_edges', np.nan)
            # out['richclub_avg_length'] = conn_data.get('richclub_avg_length', np.nan)
            
        # except Exception as e:
        #     print(f"Error computing connectome metrics for {network_path}: {e}")
        #     for key in ['avg_communicability', 'global_efficiency', 'modularity', 
        #                'avg_clustering', 'avg_degree', 'density_bct', 'transitivity',
        #                'avg_edge_distance', 'char_path_length', 'richclub_n_edges',
        #                'richclub_avg_length']:
        #         out[key] = np.nan

        # Compute additional metrics
        try:
            if 'wiring_cost' in metrics_to_compute:
                out['wiring_cost'] = calculate_wiring_cost(A, distance_matrix)

                
        except Exception as e:
            print(f"Error computing metrics: {e}")
            
            
        out['filepath'] = network_path
    
    return out



def load_or_create_results_df(output_csv: str, metrics: List[str]) -> pd.DataFrame:
    """Load existing results or create new DataFrame."""
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        
        # Ensure all required columns are present
        base_columns = ['eta', 'gamma'] + metrics
        for col in base_columns:
            if col not in df.columns:
                df[col] = np.nan
                
        print(f"Loaded existing results from {output_csv} ({len(df)} rows)")
        
    else:
        # Create empty DataFrame with required columns
        base_columns = ['eta', 'gamma'] + metrics 
        columns = base_columns + ['filepath']
        df = pd.DataFrame(columns=columns)
        print(f"Created new results DataFrame")
    
    return df


def get_missing_networks(all_files: List[str], results_df: pd.DataFrame, 
                        required_columns: List[str]) -> List[str]:
    """Identify networks that need processing."""
    
    if len(results_df) == 0:
        return all_files
    
    # Get already processed files
    if 'filepath' in results_df.columns:
        processed_files = set(results_df['filepath'].values)
    else:
        processed_files = set()
    
    # Find files not yet processed
    missing_files = [f for f in all_files if f not in processed_files]
    
    # Also find files with incomplete metrics (check key columns)
    key_columns = required_columns # ['avg_clustering', 'global_efficiency', 'modularity']
    incomplete_files = []
    
    if 'filepath' in results_df.columns:
        for _, row in results_df.iterrows():
            if any(pd.isna(row.get(col, np.nan)) for col in key_columns):
                incomplete_files.append(row['filepath'])
    
    return sorted(set(missing_files + incomplete_files))


def process_networks(
    network_files: List[str],
    distance_matrix: np.ndarray,
    metrics_to_compute: List[str],
    output_csv: str,
    batch_size: int = 10,
):
    """Process networks and write results incrementally."""
    
    # Load or create results DataFrame
    results_df = load_or_create_results_df(output_csv, metrics_to_compute)
    
    # Find networks that need processing
    files_to_process = get_missing_networks(network_files, results_df, metrics_to_compute)
    
    if len(files_to_process) == 0:
        print("All networks already processed!")
        return results_df
    
    print(f"Processing {len(files_to_process)} networks...")
    
    # Process in batches
    new_results = []
    for i, filepath in enumerate(tqdm(files_to_process, desc="Processing networks")):
        
        # Compute metrics
        result = compute_metrics_for_network(
            filepath,
            distance_matrix,
            metrics_to_compute,
        )
        
        if result is not None:
            new_results.append(result)
            
            
        # Write batch to disk
        if (i + 1) % batch_size == 0 or (i + 1) == len(files_to_process):
            if new_results:
                # Create DataFrame from new results
                new_df = pd.DataFrame(new_results)
                
                # Remove old rows for these files if they exist
                if 'filepath' in results_df.columns:
                    filepaths_to_remove = set(new_df['filepath'].values)
                    results_df = results_df[~results_df['filepath'].isin(filepaths_to_remove)]
                
                # Append new results
                results_df = pd.concat([results_df, new_df], ignore_index=True)
                
                # Reorder columns to match expected format
                expected_order = ['eta', 'gamma']

                # , 'distance_relationship_type', 'preferential_relationship_type',
                #     'generative_rule', 'num_iterations',
                #     'MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)',
                #     'network_index', 'avg_communicability', 'global_efficiency', 'modularity',
                #     'avg_clustering', 'avg_degree', 'density_bct', 'transitivity',
                #     'avg_edge_distance', 'wiring_cost', 'char_path_length',
                #     'richclub_n_edges', 'richclub_avg_length',
                #     'mc_mean', 'mc_std'
                # ] + [f'mc_{i}' for i in range(51)]
                
                # Only reorder columns that exist
                available_cols = [c for c in expected_order if c in results_df.columns]
                other_cols = [c for c in results_df.columns if c not in expected_order]
                results_df = results_df[available_cols + other_cols + ['filepath']]
                
                # Write to CSV
                results_df.to_csv(output_csv, index=False)
                
                # Clear batch
                new_results = []
                
                print(f"  Saved batch to {output_csv} ({len(results_df)} total rows)")
    
    return results_df


def main():
    """Main execution function."""
    
    # =============================================================================
    # CONFIGURATION - MODIFY THESE PATHS AND SETTINGS
    # =============================================================================
    
    
    
#     # Base directory containing generated networks
    DATASET = "suarez_MaMI_dataset"
    EXPERIMENT = "61_testing_with_kayson_animal_0" # 70_mix_and_match_animal_0"
    BASE_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
    OUTPUT_PATH = BASE_PATH / "output" / "gnm" / DATASET / EXPERIMENT
    generated_networks_dir = OUTPUT_PATH / "generated_networks"

    # Output CSV file
    OUTPUT_CSV = OUTPUT_PATH / f"further_metrics_exp_{EXPERIMENT}.csv"

    # Path to distance matrix and coordinates (modify as needed)
    DISTANCE_MATRIX_PATH = BASE_PATH / f"data/preprocessed/{DATASET}/02_distance_matrices/distance_matrix_50.npy"


    METRICS_TO_COMPUTE = [
        # 'density',
        'wiring_cost',
        # 'avg_clustering',
        # 'global_efficiency',
        # 'modularity',
        # 'avg_degree',
        # 'transitivity',
        # 'avg_edge_distance',
        # 'char_path_length',
        # 'small_world_omega',
        # 'structural_complexity',
        # 'avg_communicability',
        # 'richclub_n_edges',
        # 'richclub_avg_length', 
        
        # 'count_components_nx', 
    ]
    
    # # Metrics to compute (these match your CSV columns)
    # METRICS_TO_COMPUTE = [
    #     'avg_communicability',
    #     'global_efficiency',
    #     'modularity',
    #     'avg_clustering',
    #     'avg_degree',
    #     'density_bct',
    #     'transitivity',
    #     'avg_edge_distance',
    #     'wiring_cost',
    #     'char_path_length',
    #     'richclub_n_edges',
    #     'richclub_avg_length',
    #     'mc_mean',
    #     'mc_std'
    # ] + [f'mc_{i}' for i in range(51)]
    
    # Batch size for incremental saves
    BATCH_SIZE = 1  # Write after each network for safety
    
    # =============================================================================
    # LOAD REQUIRED DATA
    # =============================================================================
    
    print("Loading distance matrix and coordinates...")
    try:
        distance_matrix = np.load(DISTANCE_MATRIX_PATH)[0]
        print(f"  Distance matrix shape: {distance_matrix.shape}")
    except FileNotFoundError as e:
        print(f"Error: Could not find required data files: {e}")
        sys.exit(1)
    
    # =============================================================================
    # FIND NETWORK FILES
    # =============================================================================
    
    print(f"\nSearching for network files in {generated_networks_dir}...")
    network_files = find_network_files(generated_networks_dir, pattern="*.npy")
    print(f"Found {len(network_files)} network files")
    
    # =============================================================================
    # PROCESS NETWORKS
    # =============================================================================
    
    results = process_networks(
        network_files,
        distance_matrix,
        METRICS_TO_COMPUTE,
        OUTPUT_CSV,
        batch_size=BATCH_SIZE,
    )
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total networks processed: {len(results)}")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()