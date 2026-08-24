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
import json
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.get_distance_matrix import get_distance_matrix_from_coords
from analysis.structural_measures import analyze_connectomes

from src.preprocessing.preprocessing_setup import setup
from preprocessing.threshold_to_density import threshold_to_density


from src.preprocessing.utils import setup_paths, save_dataframe # TODO: Remove this again, not needed. 


# Config & argument parsing
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


def step_load_identifiers(paths: dict, resolution: int) -> Tuple[pd.DataFrame, np.ndarray]:
    id_path = paths["path_00_preprocessed"] / f"05_roi_names_rsn_name_hemisphere_{resolution}.csv"
    df_identifiers = pd.read_csv(
        id_path, header=None, names=["roi_name", "roi_name_short", "rsn_name", "hemisphere"]
    )

    hemi_id = np.array([0 if x == "rh" else 1 for x in df_identifiers["hemisphere"]])
    df_identifiers["hemi_id"] = hemi_id

    # Strip whitespace and stray characters
    df_identifiers = df_identifiers.map(lambda x: x.strip() if isinstance(x, str) else x)

    save_dataframe(paths["path_04_further_info"] / f"df_identifiers_{resolution}.csv", df_identifiers)
    return df_identifiers, hemi_id


def main(resolution = None):
    cfg = parse_args()
    if resolution: 
        cfg.resolution = resolution

    paths = setup_paths(dataset_name="shafiei_human_consensus_dataset")

    # Distance matrix
    coords_csv = paths["path_00_preprocessed"] / f"04_coordinates_{resolution}.csv"
    coords = np.loadtxt(coords_csv, delimiter=",")
    dist_mat = get_distance_matrix_from_coords(
        coords=coords, 
        save_dir=paths["path_02_distance_matrices"], 
        resolution=cfg.resolution,
        plot=cfg.do_plots,
    )
    
    # Identifiers & hemisphere id
    df_identifiers, hemi_id = step_load_identifiers(paths, resolution=cfg.resolution)
    
    conns = np.loadtxt(paths["path_00_preprocessed"] / f"01_weighted_adj_mat_{resolution}.csv", 
				delimiter=",", dtype=np.float64) # here only one
    
    np.save(paths["path_01_connectomes"] / f"01_consensus_wei_{resolution}.npy", conns) # here only one
    
    # Analyze the weighted connectome
    df_graph_measures = pd.DataFrame(
        analyze_connectomes(
            connectomes=conns,
            distance_matrix=dist_mat,
            comm_mode=cfg.comm_mode,
            rich_nodes_global=cfg.rich_nodes_global,
            rich_top_percent=cfg.rich_top_percent,
        )
    )
    analysis_path = paths["path_03_graph_measures"] / f"df_graph_measures_wei_{resolution}_percent.csv"
    save_dataframe(
        analysis_path,
        df_graph_measures,
    )
    
    thres_conn, final_density = threshold_to_density(consensus_wei=conns, 
                                                    density=10, 
                                                    output_folder=paths["path_01_connectomes"])

    # Analyze the binarized and thresholded connectome
    df_graph_measures = pd.DataFrame(
        analyze_connectomes(
            connectomes=thres_conn,
            distance_matrix=dist_mat,
            comm_mode=cfg.comm_mode,
            rich_nodes_global=cfg.rich_nodes_global,
            rich_top_percent=cfg.rich_top_percent,
        )
    )
    analysis_path = paths["path_03_graph_measures"] / f"df_graph_measures_bin_{resolution}_density_{cfg.analyze_density}_percent.csv"
    save_dataframe(
        analysis_path,
        df_graph_measures,
    )
    
    # Print summary to console
    print("\nPipeline completed successfully. Summary:")
    summary = {
        "resolution": cfg.resolution,
        "analyze_density": cfg.analyze_density,
        "plots": cfg.do_plots,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    print(json.dumps(summary, indent=2))

    # Save summary to a text file
    output_file = paths["path_output_for_logs_and_plots"] / f"preprocessing_pipeline_summary_{resolution}_density_{cfg.analyze_density}_percent.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_file}")
    print(f"Analysis CSV saved to: {analysis_path}")


if __name__ == "__main__":
    main(resolution=68)
    main(resolution=114)
    main(resolution=219)
    main(resolution=448)
    main(resolution=1000)