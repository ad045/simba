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

import bct
from netneurotools.networks import threshold_network, struct_consensus

# Local imports from your codebase
# from notebook_setup import setup
from src.preprocessing.preprocess_distance_matrix import get_distance_matrix_from_coords, get_distance_matrix_from_fiber_lengths
from src.preprocessing.preprocess_70_connectomes import get_individual_connectomes
from src.structural_analysis.graph_measures import analyze_connectomes

from src.preprocessing.preprocessing_setup import setup
from src.preprocessing.process_01_consensus_networks import threshold_to_density


from utils import ensure_dir, save_numpy, save_dataframe # TODO: Remove this again, not needed. 
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


# ----------------------------
# Pipeline steps
# ----------------------------

def step_setup_paths(dataset_name="shafiei_human_consensus_dataset") -> dict:
    env = setup(dataset_name=dataset_name)
    
    path_raw_data = env["DATA_PATH"] / "raw" / dataset_name
    
    path_output_for_logs_and_plots = env["OUTPUT_PATH"] / "00_preprocessing" / dataset_name # for plots and logs 
    path_preprocessed = env["PREPROCESSED_PATH"] #  / dataset_name 
    
    path_00_preprocessed = path_preprocessed / "00_preprocessed" # / dataset_name
    path_01_connectomes = path_preprocessed / "01_connectomes"  # / dataset_name
    path_02_distance_matrices = path_preprocessed / "02_distance_matrices"  # / dataset_name
    path_03_graph_measures = path_preprocessed / "03_graph_measures"  # / dataset_name
    path_04_further_info = path_preprocessed / "04_further_info" # / dataset_name
    

    # INPUT_DATA_PATH: Path = env["PREPROCESSED_PATH"] / dataset_name / "00_preprocessed"


    for dir_path in [path_00_preprocessed, path_01_connectomes, path_02_distance_matrices, 
                     path_03_graph_measures, path_04_further_info, path_output_for_logs_and_plots,
                     path_raw_data]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return {
        "path_00_preprocessed": path_00_preprocessed,
        "path_01_connectomes": path_01_connectomes,
        "path_02_distance_matrices": path_02_distance_matrices,
        "path_03_graph_measures": path_03_graph_measures,
        "path_04_further_info": path_04_further_info,
        "path_output_for_logs_and_plots": path_output_for_logs_and_plots
    }


# def step_load_individual_connectomes(paths: dict, resolution: int = 68) -> Tuple[np.ndarray, int, int]:
#     sc_mat_path = paths["INPUT_DATA_PATH"] / f"data_700mb_individual_connectomes/SC_{resolution}.mat"
#     all_connectomes, nsub, n = get_individual_connectomes(
#         raw_data_path=sc_mat_path,
#         output_path=paths["PREPROCESSED_PATH"],
#     )
#     return all_connectomes, nsub, n


# def step_threshold_connectomes(
#     all_connectomes: np.ndarray,
#     n: int,
#     goal_densities: Iterable[int],
#     save_dir: Path,
# ) -> None:
#     goal_densities = list(goal_densities)
#     for gd in goal_densities:
#         connectomes_binarized = np.array([
#             threshold_network(all_connectomes[i], retain=gd)
#             for i in range(len(all_connectomes))
#         ])
#         connectome_densities = np.array([
#             bct.density_und(connectomes_binarized[i])[0]
#             for i in range(len(connectomes_binarized))
#         ])
#         print(
#             f"Average density over {len(connectomes_binarized)} subjects: "
#             f"{np.mean(connectome_densities)*100:.3f}% (goal {gd}%)."
#         )
#         save_numpy(
#             save_dir / f"connectomes_binarized_{n}x{n}_density_{gd}_percent.npy",
#             connectomes_binarized,
#         )

#     # Save weighted set once
#     save_numpy(save_dir / f"connectomes_weighted_{n}.npy", all_connectomes)


# def step_graph_measures(
#     paths: dict,
#     dist_mat: np.ndarray,
#     conn: np.array, 
#     # all_connectomes: np.ndarray,
#     n: int,
#     analyze_density: int,
#     comm_mode: str,
#     rich_nodes_global: Optional[np.ndarray],
#     rich_top_percent: float,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
#     bin_path = paths["path_01_connectomes"] / f"01_consensus_wei_{n}.npy"
    
#     # "bin_connectomes_binarized_{n}x{n}_density_{analyze_density}_percent.npy"
#     if not bin_path.exists():
#         raise FileNotFoundError(
#             f"Expected binarized connectomes for density {analyze_density}% at {bin_path}. "
#             "Run thresholding with this density or change --analyze-density."
#         )
#     conn = np.load(bin_path) # connectomes_binarized = np.load(bin_path)

#     df_graph_measures_bin = pd.DataFrame(
#         analyze_connectomes(
#             connectomes=conns,
#             distance_matrix=dist_mat,
#             comm_mode=comm_mode,
#             rich_nodes_global=rich_nodes_global,
#             rich_top_percent=rich_top_percent,
#         )
#     )
#     df_graph_measures_all = pd.DataFrame(
#         analyze_connectomes(
#             connectomes=conns,
#             distance_matrix=dist_mat,
#             comm_mode=comm_mode,
#             rich_nodes_global=rich_nodes_global,
#             rich_top_percent=rich_top_percent,
#         )
#     )
    
    
    
    

#     return df_graph_measures_bin, df_graph_measures_all, connectomes_binarized


def step_load_identifiers(paths: dict, n: int) -> Tuple[pd.DataFrame, np.ndarray]:
    id_path = paths["INPUT_DATA_PATH"] / f"data_10_consensus/05_roi_names_rsn_name_hemisphere_{n}.csv"
    df_identifiers = pd.read_csv(
        id_path, header=None, names=["roi_name", "roi_name_short", "rsn_name", "hemisphere"]
    )

    hemi_id = np.array([0 if x == "rh" else 1 for x in df_identifiers["hemisphere"]])
    df_identifiers["hemi_id"] = hemi_id

    # Strip whitespace and stray characters
    df_identifiers = df_identifiers.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    save_dataframe(paths["PREPROCESSED_PATH"] / f"df_identifiers_{n}x{n}.csv", df_identifiers)
    return df_identifiers, hemi_id


# def step_consensus_connectomes(
#     paths: dict,
#     connectomes_binarized: np.ndarray,
#     all_connectomes: np.ndarray,
#     dist_mat: np.ndarray,
#     hemi_id: np.ndarray,
#     n: int,
#     analyze_density: int,
# ) -> Tuple[np.ndarray, np.ndarray, float, float]:
#     # Shapes expected by struct_consensus: nodes x nodes x subjects
#     # Our arrays are subjects x nodes x nodes, so we transpose with .T
#     consensus_conn_bin = struct_consensus(
#         connectomes_binarized.T, distance=dist_mat, hemiid=hemi_id.reshape(-1, 1)
#     )
#     consensus_density_bin = bct.density_und(consensus_conn_bin)[0]

#     consensus_conn_all = struct_consensus(
#         all_connectomes.T, distance=dist_mat, hemiid=hemi_id.reshape(-1, 1)
#     )
#     consensus_density_all = bct.density_und(consensus_conn_all)[0]

#     print(f"Consensus (binarized @ {analyze_density}%): {consensus_density_bin*100:.2f}%")
#     print(f"Consensus (weighted): {consensus_density_all*100:.2f}%")

#     save_numpy(
#         paths["PREPROCESSED_PATH"]
#         / f"consensus_connectome_bin_{n}x{n}_density_{analyze_density}_percent.npy",
#         consensus_conn_bin,
#     )
#     save_numpy(paths["PREPROCESSED_PATH"] / f"consensus_connectome_all_{n}x{n}.npy", consensus_conn_all)

#     return consensus_conn_bin, consensus_conn_all, consensus_density_bin, consensus_density_all


# def step_plot_consensus(
#     paths: dict,
#     conn: Optional[np.ndarray],
#     dist: Optional[np.ndarray],
#     analyze_density: int,
# ) -> None:
#     fig = plt.figure(figsize=(12, 6))

#     if conn: 
#         ax1 = fig.add_subplot(1, 2, 1)
#         im1 = ax1.imshow(conn, cmap="Blues")
#         fig.colorbar(im1, ax=ax1)
#         ax1.set(xlabel="Region Index", ylabel="Region Index",
#                 title=f"Consensus (Binarized at {analyze_density}%)\nDensity: {density_bin*100:.2f}%")

#     if dist: 
#         ax2 = fig.add_subplot(1, 2, 2)
#         im2 = ax2.imshow(dist, cmap="Blues")
#         fig.colorbar(im2, ax=ax2)
#         ax2.set(xlabel="Region Index", ylabel="Region Index",
#                 title=f"Consensus (Weighted)\nDensity: {density_all*100:.2f}%")

#     fig.suptitle(
#         "Consensus differs if binarization is performed before consensus (density differs, too)",
#         fontsize=12,
#     )

#     outpng = paths["OUTPUT_PATH"] / f"consensus_connectomes_compare_density_{analyze_density}.png"
#     fig.tight_layout()
#     fig.savefig(outpng) 
#     print(f"Saved plot: {outpng}")


# ----------------------------
# Main orchestrator
# ----------------------------

def main(resolution = None):
    cfg = parse_args()
    if resolution: 
        cfg.resolution = resolution

    paths = step_setup_paths()

    # Distance matrix
    dist_mat = get_distance_matrix_from_coords(
        paths=paths, 
        resolution=resolution,
        plot=cfg.do_plots,
    )
    
    # plt.imshow(dist_mat)
    # plt.show()
    
    # dist_mat = get_distance_matrix_from_fiber_lengths(
    #     paths=paths, 
    #     resolution=resolution,
    #     plot=cfg.do_plots,
    # )
        
    # plt.imshow(dist_mat)
    # plt.show()




    # Individual connectomes
    # all_connectomes, nsub, n = step_load_individual_connectomes(paths, resolution=cfg.resolution)

    # Threshold (binarize) at specified goal densities (hyperparameters)
    # step_threshold_connectomes(
    #     all_connectomes=all_connectomes,
    #     n=n,
    #     goal_densities=cfg.goal_densities,
    #     save_dir=paths["PREPROCESSED_PATH"],
    # )

    # Identifiers & hemisphere id
    # df_identifiers, hemi_id = step_load_identifiers(paths, n)

    # Consensus connectomes (binarized @ analyze_density and weighted)
    # consensus_bin, consensus_all, d_bin, d_all = step_consensus_connectomes(
    #     paths=paths,
    #     connectomes_binarized=connectomes_binarized,
    #     all_connectomes=all_connectomes,
    #     dist_mat=dist_mat,
    #     hemi_id=hemi_id,
    #     n=n,
    #     analyze_density=cfg.analyze_density,
    # )
    
   
    
    
    
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
    analysis_path = paths["path_03_graph_measures"] / f"df_graph_measures_wei_{resolution}_density_{cfg.analyze_density}_percent.csv"
    save_dataframe(
        analysis_path,
        df_graph_measures,
    )
    
    
    thres_conn, final_density = threshold_to_density(consensus_wei=conns, 
                                                    n_nodes=resolution, 
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



    # if cfg.do_plots:
    #     step_plot_consensus(
    #         paths=paths,
    #         consensus_conn_bin=None, # consensus_bin,
    #         consensus_conn_wei=consensus_weighted, # consensus_all,
    #         density_bin=None, # d_bin,
    #         density_all=None, # d_all,
    #         analyze_density=cfg.analyze_density,
    #     )

    # Print summary to console
    print("\nPipeline completed successfully. Summary:")
    summary = {
        "resolution": cfg.resolution,
        # "goal_densities": cfg.goal_densities,
        "analyze_density": cfg.analyze_density,
        "plots": cfg.do_plots,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    print(json.dumps(summary, indent=2))

    # Save summary to a text file
    output_file = paths["path_output_for_logs_and_plots"] / "preprocessing_pipeline_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_file}")
    print(f"Analysis CSV saved to: {analysis_path}")




if __name__ == "__main__":
    # try:
    main(resolution=68)
    main(resolution=114)
    main(resolution=219)
    main(resolution=448)
    main(resolution=1000)
        
    # except Exception as e:
    #     print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
    #     sys.exit(1)
