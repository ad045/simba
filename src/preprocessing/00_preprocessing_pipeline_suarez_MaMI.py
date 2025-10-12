# CHANGE THE SETUP FUNCTION!! (Is quite unprofessional - it was previously created to allow easy work in notebooks)

"""
Connectome Preprocessing Pipeline.

This script includes the following steps: 
    - Computes distance matrix
    - Loads individual connectomes, thresholds them at specified goal densities (a hyperparameter - can be passed with --goal-densities 1 2 3 ...) 
    - Computes graph measures (if analyzing a single density, use --analyze-density 10) 
    - Loads ROI identifiers
    - Builds structural consensus connectomes
    - Optionally plots/saves illustrative figures


Usage examples: 
    - Run end-to-end with default settings (multiple densities): 
        python connectome_preprocessing_pipeline.py
        
    - Run with a single density (hyperparameter) and disable plotting:
        python connectome_preprocessing_pipeline.py --goal-densities 14 --no-plots
        
    - Change resolution and choose which density to analyze for graph measures:
        python connectome_preprocessing_pipeline.py \
          --resolution 68 \
          --goal-densities 10 12 14 16 18 20 \
          --analyze-density 12


Notes: 
    - (Based on 01_preprocessing.ipynb)
    - Paths are taken from `notebook_setup.setup()`. 
    - Heavy intermediate results are cached to disk to avoid recomputation.  
"""


from __future__ import annotations

import argparse
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import bct
from netneurotools.networks import threshold_network, struct_consensus
from scipy.spatial.distance import cdist


# Local imports from your codebase
# from notebook_setup import setup
from preprocessing.get_distance_matrix import get_distance_matrix_from_coords, get_distance_matrix_from_fiber_lengths
from src.structural_analysis.graph_measures import analyze_connectomes

from src.preprocessing.preprocessing_setup import setup
from preprocessing.threshold_to_density import threshold_to_density


from src.preprocessing.utils import setup_paths, ensure_dir, save_numpy, save_dataframe # TODO: Remove this again, not needed. 
# ----------------------------
# Config & argument parsing
# ----------------------------
@dataclass
class PipelineConfig:
    resolution: int = 68
    goal_densities: Tuple[int, ...] = (10, 12, 14, 16, 18, 20)  # in percent
    analyze_density: int = 10  # used for steps requiring a single binarized set
    do_plots: bool = False # True

    # Communication model and rich-club options for graph measures
    comm_mode: str = "estrada_scaled"
    rich_nodes_global: Optional[np.ndarray] = None
    rich_top_percent: float = 0.20


def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(description="Connectome preprocessing pipeline")
    p.add_argument("--resolution", type=int, default=68, # CHANGE here
                   help="Parcellation resolution for distance matrix & inputs (default: 68)")
    p.add_argument("--goal-densities", type=int, nargs="+",
                   default=[10, 12, 14, 16, 18, 20],
                   help="One or more goal densities in PERCENT to retain during thresholding")
    p.add_argument("--analyze-density", type=int, default=10,
                   help="Which density (in percent) to use for analyses that require a single binarized set")
    p.add_argument("--do_plots", action="store_true",
                   help="Activate plotting (only save data)")
    p.add_argument("--comm-mode", type=str, default="estrada_scaled",
                   help="Communication model used in analyze_connectomes (default: estrada_scaled)")
    p.add_argument("--rich-top-percent", type=float, default=0.20,
                   help="Top fraction for rich club node selection (default: 0.20)")

    args = p.parse_args()

    return PipelineConfig(
        resolution=args.resolution,
        goal_densities=tuple(args.goal_densities),
        analyze_density=args.analyze_density,
        do_plots=args.do_plots,
        comm_mode=args.comm_mode,
        rich_top_percent=args.rich_top_percent,
    )

# def main(resolution = None):
# cfg = parse_args()

cfg = PipelineConfig(
        resolution=50, # args.resolution,
        goal_densities=[10], # tuple(args.goal_densities),
        analyze_density=10, # args.analyze_density,
        do_plots=True, # args.do_plots,
        comm_mode="estrada_scaled", # args.comm_mode,
        rich_top_percent=0.2) 


paths = setup_paths(dataset_name="suarez_MaMI_dataset") 
    
    
    
    

# def enrich_animal_metadata(animal_names_df, info_csv_path):
#     """
#     Enrich animal names dataframe with taxonomic information.
    
#     Args:
#         animal_names_df: DataFrame with 'animal' column
#         info_csv_path: Path to the info.csv file with taxonomic data
    
#     Returns:
#         DataFrame with added taxonomic columns
#     """
#     # Load the info CSV
#     info_df = pd.read_csv(info_csv_path)
    
#     # Standardize column names (remove spaces, lowercase)
#     info_df.columns = [col.strip().replace('-', '_').replace(' ', '_').lower() for col in info_df.columns]
    
#     # Create mapping from filename to taxonomic info
#     info_df['filename_clean'] = info_df['filename'].str.lower()
#     animal_names_df['animal_clean'] = animal_names_df['animal'].str.lower()
    
#     # Merge on filename/animal name
#     enriched_df = animal_names_df.merge(
#         info_df[['filename_clean', 
#                 #  'common_name', 
#                  'name', 'species', 'genus', 
#                  'sub_family', 'family', 'sub_order', 'order', 'super_order']],
#         left_on='animal_clean',
#         right_on='filename_clean',
#         how='left'
#     )
    
#     # Drop temporary columns
#     enriched_df = enriched_df.drop(columns=['animal_clean', 'filename_clean'])
    
#     return enriched_df


def enrich_animal_metadata(animal_names_df, info_csv_path):
    """
    Enrich animal names dataframe with taxonomic information.
    
    Args:
        animal_names_df: DataFrame with 'animal' column
        info_csv_path: Path to the info.csv file with taxonomic data
    
    Returns:
        DataFrame with added taxonomic columns
    """
    # Load the info CSV
    info_df = pd.read_csv(info_csv_path)
    
    # Standardize column names (remove spaces, lowercase)
    info_df.columns = [col.strip().replace('-', '_').replace(' ', '_').lower() for col in info_df.columns]
    
    # Remove double (or multiple) spaces from all string columns in info_df
    for col in info_df.columns:
        if info_df[col].dtype == 'object':  # String columns
            info_df[col] = info_df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Clean animal names in both dataframes
    for col in animal_names_df.columns:
        if animal_names_df[col].dtype == 'object':
            animal_names_df[col] = animal_names_df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Add phylogenetic grouping column based on the tree structure
    def assign_phylogenetic_group(row):
        """Assign phylogenetic group based on order."""
        order = row.get('order', '')
        if pd.isna(order):
            return 'Unknown'
        
        order = str(order).strip()
        
        # Map orders to major phylogenetic groups
        order_to_group = {
            'Xenarthra': 'Xenarthra',
            'Dermoptera': 'Euarchontoglires',
            'Scandentia': 'Euarchontoglires', 
            'Primates': 'Euarchontoglires',
            'Lagomorpha': 'Euarchontoglires',
            'Rodentia': 'Euarchontoglires',
            'Eulipotyphla': 'Laurasiatheria',
            'Carnivora': 'Laurasiatheria',
            'Pholidota': 'Laurasiatheria',
            'Perissodactyla': 'Laurasiatheria',
            'Cetartiodactyla': 'Laurasiatheria',
            'Chiroptera': 'Laurasiatheria'
        }
        
        return order_to_group.get(order, 'Other')
    
    # Create mapping from filename to taxonomic info
    info_df['filename_clean'] = info_df['filename'].str.lower()
    animal_names_df['animal_clean'] = animal_names_df['animal'].str.lower()
    
    # Merge on filename/animal name
    enriched_df = animal_names_df.merge(
        info_df[['filename_clean', 'common_name', 'name', 'species', 'genus', 
                 'sub_family', 'family', 'sub_order', 'order', 'super_order']],
        left_on='animal_clean',
        right_on='filename_clean',
        how='left'
    )
    
    # Add phylogenetic group column
    enriched_df['phylogenetic_group'] = enriched_df.apply(assign_phylogenetic_group, axis=1)
    
    # Drop temporary columns
    enriched_df = enriched_df.drop(columns=['animal_clean', 'filename_clean'])
    
    return enriched_df



def reduce_nodes(connection_matrix, coordinates, resolution=100):
    """
    Reduce nodes in a brain connectivity matrix by merging closest pairs.
    
    Args:
        connection_matrix: (200, 200) connectivity matrix
        coordinates: (200, 3) node coordinates
        n_target: target number of nodes (default 100)
    
    Returns:
        new_matrix: (n_target, n_target) reduced connectivity matrix
        new_coords: (n_target, 3) reduced coordinates
    """
    n_nodes = len(coordinates)
    n_per_half = n_nodes // 2
    n_target_per_half = resolution // 2
    
    # Process each hemisphere
    new_coords_list = []
    merge_maps = []
    
    for half_idx in range(2):
        start_idx = half_idx * n_per_half
        end_idx = start_idx + n_per_half
        
        # Get hemisphere data
        coords_half = coordinates[start_idx:end_idx].copy()
        active = np.ones(n_per_half, dtype=bool)
        merge_map = np.arange(n_per_half)
        
        # Merge nodes until target reached
        n_to_merge = n_per_half - n_target_per_half
        for _ in range(n_to_merge):
            active_indices = np.where(active)[0]
            active_coords = coords_half[active_indices]
            
            # Find closest pair
            distances = cdist(active_coords, active_coords)
            np.fill_diagonal(distances, np.inf)
            i, j = np.unravel_index(distances.argmin(), distances.shape)
            idx_i, idx_j = active_indices[i], active_indices[j]
            
            # Merge: keep idx_i, remove idx_j
            coords_half[idx_i] = (coords_half[idx_i] + coords_half[idx_j]) / 2
            active[idx_j] = False
            merge_map[merge_map == idx_j] = idx_i
        
        new_coords_list.append(coords_half[active])
        merge_maps.append(merge_map + start_idx)
    
    # Combine hemispheres
    new_coords = np.vstack(new_coords_list)
    full_merge_map = np.concatenate(merge_maps)
    
    # Build new connectivity matrix
    new_matrix = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            mask_i = (full_merge_map == full_merge_map[i])
            mask_j = (full_merge_map == full_merge_map[j])
            new_matrix[i, j] = connection_matrix[np.ix_(mask_i, mask_j)].mean()
    
    return new_matrix, new_coords


# Setup paths
base_path = paths["path_raw_data"] / "connectivity/mami"
conn_dir = base_path / "conn_100"
coords_dir = base_path / "coords"

# Intermediate paths 
output_conn_dir = paths["path_00_preprocessed"] / "connectomes_downsampled_to_50"
output_coords_dir = paths["path_00_preprocessed"] / "coords_downsampled_to_50"
output_results_dir = paths["path_output_for_logs_and_plots"] / "results"
output_dist_dir = paths["path_00_preprocessed"] / "dis_matrices_downsampled_to_50" # paths["path_02_distance_matrices"] 

resolution = 100 # final resolution

# Create output directories
output_conn_dir.mkdir(exist_ok=True)
output_coords_dir.mkdir(exist_ok=True)
output_results_dir.mkdir(exist_ok=True)
output_dist_dir.mkdir(exist_ok=True)

# Get all animal names
animal_files = list(conn_dir.glob("*.npy"))
animal_names = [f.stem for f in animal_files]

# Modified section for the MaMI preprocessing script
# Replace the animal processing loop with this version

# Storage for combined arrays and all results
all_conn = []
all_dist = []
all_results = []

# Storage for binarized connectomes at each density
all_bin_by_density = {density: [] for density in cfg.goal_densities}

# Process each animal
for name in animal_names:
    print(f"Processing {name}...")
    
    # Load data
    connection_matrix = np.load(conn_dir / f"{name}.npy")
    coordinates = np.load(coords_dir / f"{name}.npy")
    
    # Reduce resolution
    new_matrix, new_coords = reduce_nodes(connection_matrix, coordinates, resolution=100)
    
    # Calculate distance matrix
    distance_matrix = cdist(new_coords, new_coords)
    
    # Save individual files
    np.save(output_conn_dir / f"{name}.npy", new_matrix)
    np.save(output_coords_dir / f"{name}.npy", new_coords)
    np.save(output_dist_dir / f"{name}.npy", distance_matrix)
    
    # Create binarized versions at different densities
    print(f"  Creating binarized versions:")
    for density in cfg.goal_densities:
        # Create temporary output folder for this density if needed
        temp_output = paths["path_00_preprocessed"] / f"temp_bin_{density}"
        temp_output.mkdir(exist_ok=True)
        
        # Use your existing function
        thres_conn, final_density = threshold_to_density(
            consensus_wei=new_matrix, 
            n_nodes=resolution, 
            density=density, 
            output_folder=temp_output
        )
        
        # Store for later combination
        all_bin_by_density[density].append(thres_conn)
    
    # Analyze weighted connectome
    results = analyze_connectomes(
        new_matrix[np.newaxis, :, :],
        distance_matrix,
        comm_mode="estrada_scaled"
    )
    
    # Add animal name to results
    for result in results:
        result['animal'] = name
        all_results.append(result)
    
    # Store for combined arrays
    all_conn.append(new_matrix)
    all_dist.append(distance_matrix)

# Save combined weighted arrays
all_conn = np.stack(all_conn)  # Shape: (num_animals, 100, 100)
all_dist = np.stack(all_dist)  # Shape: (num_animals, 100, 100)

np.save(paths["path_01_connectomes"] / f"00_connectomes_{resolution}.npy", all_conn)
np.save(paths["path_02_distance_matrices"] / f"distance_matrix_{resolution}.npy", all_dist)

# Save combined binarized arrays for each density
print("\nSaving combined binarized arrays...")
for density in cfg.goal_densities:
    all_bin = np.stack(all_bin_by_density[density])  # Shape: (num_animals, 100, 100)
    
    save_path = paths["path_01_connectomes"] / f"01_consensus_bin_density_{density}_percent_{resolution}.npy"
    np.save(save_path, all_bin)
    print(f"  Density {density}%: {all_bin.shape}")
    
    # Clean up temporary files
    temp_output = paths["path_00_preprocessed"] / f"temp_bin_{density}"
    if temp_output.exists():
        for file in temp_output.glob("*.npy"):
            file.unlink()
        temp_output.rmdir()

# Optionally: Analyze binarized connectomes at the specified density
if cfg.analyze_density in cfg.goal_densities:
    print(f"\nAnalyzing binarized connectomes at {cfg.analyze_density}% density...")
    bin_results = []
    all_bin_analyze = all_bin_by_density[cfg.analyze_density]
    
    for idx, (name, bin_conn) in enumerate(zip(animal_names, all_bin_analyze)):
        results = analyze_connectomes(
            np.array([bin_conn]),  # Shape: (1, 100, 100)
            all_dist[idx],
            comm_mode="estrada_scaled"
        )
        for result in results:
            result['animal'] = name
            bin_results.append(result)
    
    # Save binary analysis results
    bin_results_df = pd.DataFrame(bin_results)
    bin_results_df.to_csv(
        paths["path_03_graph_measures"] / f"df_graph_measures_bin_{resolution}_density_{cfg.analyze_density}_percent.csv", 
        index=False
    )
    print(f"Binary analysis results saved: {bin_results_df.shape}")

# Save weighted results as CSV (existing code)
results_df = pd.DataFrame(all_results)
results_df.to_csv(paths["path_03_graph_measures"] / f"df_graph_measures_{resolution}.csv", index=False)


# Extract animal names
only_names_df = results_df[["animal"]].drop_duplicates().reset_index(drop=True)

# Enrich with taxonomic information
info_csv_path = paths["path_raw_data"] / "info" / "info.csv"
enriched_names_df = enrich_animal_metadata(only_names_df, info_csv_path)

# Save enriched dataframe
enriched_names_df.to_csv(
    paths["path_04_further_info"] / f"names_of_animals_with_preprocessed_connectomes_{resolution}.csv",
    index=True
)

print(f"\nProcessed {len(animal_names)} animals")
print(f"Combined connectivity shape: {all_conn.shape}")
print(f"Combined distances shape: {all_dist.shape}")
print(f"Binarized versions created for densities: {list(cfg.goal_densities)}")
print(f"Results saved to: {paths['path_03_graph_measures']}")