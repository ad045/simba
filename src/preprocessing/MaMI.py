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


# Local imports from your codebase
# from notebook_setup import setup
from src.preprocessing.preprocess_distance_matrix import get_distance_matrix_from_coords, get_distance_matrix_from_fiber_lengths
from src.structural_analysis.graph_measures import analyze_connectomes

from src.preprocessing.preprocessing_setup import setup
from src.preprocessing.process_01_consensus_networks import threshold_to_density


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
    
    
    
    
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

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

# Storage for combined arrays and all results
all_conn = []
all_dist = []
all_results = []

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
        
    # Analyze connectome
    results = analyze_connectomes(
        new_matrix[np.newaxis, :, :],  # Shape: (1, 100, 100)
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

# Save combined arrays
all_conn = np.stack(all_conn)  # Shape: (num_animals, 100, 100)
all_dist = np.stack(all_dist)  # Shape: (num_animals, 100, 100)

np.save(paths["path_01_connectomes"] / f"00_connectomes_{resolution}.npy", all_conn)
np.save( paths["path_02_distance_matrices"] / f"distance_matrix_{resolution}.npy", all_dist)

# Save results as CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv(paths["path_03_graph_measures"] / f"df_graph_measures_{resolution}.csv", index=False)

only_names_df = results_df["animal"]
only_names_df.to_csv(paths["path_04_further_info"] / f"names_of_animals_with_preprocessed_connectomes_{resolution}.csv")

print(f"\nProcessed {len(animal_names)} animals")
print(f"Combined connectivity shape: {all_conn.shape}")
print(f"Combined distances shape: {all_dist.shape}")
print(f"Results saved to: {paths["path_03_graph_measures"] / f"df_graph_measures_{resolution}.csv"}")
print(f"Results shape: {results_df.shape}")