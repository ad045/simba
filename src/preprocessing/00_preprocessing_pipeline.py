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
from src.preprocessing.preprocess_distance_matrix import get_distance_matrix
from src.preprocessing.preprocess_70_connectomes import get_individual_connectomes
from src.structural_analysis.graph_measures import analyze_connectomes

from src.preprocessing.preprocessing_setup import setup

# ----------------------------
# Config & argument parsing
# ----------------------------
@dataclass
class PipelineConfig:
    resolution: int = 68
    goal_densities: Tuple[int, ...] = (10, 12, 14, 16, 18, 20)  # in percent
    analyze_density: int = 10  # used for steps requiring a single binarized set
    do_plots: bool = True

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
    p.add_argument("--no-plots", action="store_true",
                   help="Disable plotting (only save data)")
    p.add_argument("--comm-mode", type=str, default="estrada_scaled",
                   help="Communication model used in analyze_connectomes (default: estrada_scaled)")
    p.add_argument("--rich-top-percent", type=float, default=0.20,
                   help="Top fraction for rich club node selection (default: 0.20)")

    args = p.parse_args()

    return PipelineConfig(
        resolution=args.resolution,
        goal_densities=tuple(args.goal_densities),
        analyze_density=args.analyze_density,
        do_plots=not args.no_plots,
        comm_mode=args.comm_mode,
        rich_top_percent=args.rich_top_percent,
    )


# ----------------------------
# Utility helpers
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_numpy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"Saved: {path}")


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ----------------------------
# Pipeline steps
# ----------------------------

def step_setup_paths() -> dict:
    env = setup()
    DATA_PATH: Path = env["DATA_PATH"]
    OUTPUT_PATH: Path = env["OUTPUT_PATH"] / "preprocessing"
    PREPROCESSED_PATH: Path = env["PREPROCESSED_PATH"] / "01_first_analysises"
    INPUT_DATA_PATH: Path = env["PREPROCESSED_PATH"] / "00_just_converted_for_matlab_and_python"

    ensure_dir(PREPROCESSED_PATH)
    ensure_dir(OUTPUT_PATH)

    return {
        "DATA_PATH": DATA_PATH,
        "OUTPUT_PATH": OUTPUT_PATH,
        "PREPROCESSED_PATH": PREPROCESSED_PATH,
        "INPUT_DATA_PATH": INPUT_DATA_PATH,
    }


def step_distance_matrix(paths: dict, resolution: int, do_plots: bool) -> np.ndarray:
    coords_csv = paths["INPUT_DATA_PATH"] / f"data_10_consensus/04_coordinates_{resolution}.csv"
    dist_mat = get_distance_matrix(
        raw_data_path=coords_csv,
        preprocessed_folder_path=paths["PREPROCESSED_PATH"],
        plot=do_plots,
    )
    # Ensure symmetry and zero diagonal (defensive)
    dist_mat = np.asarray(dist_mat)
    dist_mat = (dist_mat + dist_mat.T) / 2.0
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def step_load_individual_connectomes(paths: dict, resolution: int = 68) -> Tuple[np.ndarray, int, int]:
    sc_mat_path = paths["INPUT_DATA_PATH"] / f"data_700mb_individual_connectomes/SC_{resolution}.mat"
    all_connectomes, nsub, n = get_individual_connectomes(
        raw_data_path=sc_mat_path,
        output_path=paths["PREPROCESSED_PATH"],
    )
    return all_connectomes, nsub, n


def step_threshold_connectomes(
    all_connectomes: np.ndarray,
    n: int,
    goal_densities: Iterable[int],
    save_dir: Path,
) -> None:
    goal_densities = list(goal_densities)
    for gd in goal_densities:
        connectomes_binarized = np.array([
            threshold_network(all_connectomes[i], retain=gd)
            for i in range(len(all_connectomes))
        ])
        connectome_densities = np.array([
            bct.density_und(connectomes_binarized[i])[0]
            for i in range(len(connectomes_binarized))
        ])
        print(
            f"Average density over {len(connectomes_binarized)} subjects: "
            f"{np.mean(connectome_densities)*100:.3f}% (goal {gd}%)."
        )
        save_numpy(
            save_dir / f"connectomes_binarized_{n}x{n}_density_{gd}_percent.npy",
            connectomes_binarized,
        )

    # Save weighted set once
    save_numpy(save_dir / f"connectomes_weighted_{n}x{n}.npy", all_connectomes)


def step_graph_measures(
    paths: dict,
    dist_mat: np.ndarray,
    all_connectomes: np.ndarray,
    n: int,
    analyze_density: int,
    comm_mode: str,
    rich_nodes_global: Optional[np.ndarray],
    rich_top_percent: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    bin_path = paths["PREPROCESSED_PATH"] / f"connectomes_binarized_{n}x{n}_density_{analyze_density}_percent.npy"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"Expected binarized connectomes for density {analyze_density}% at {bin_path}. "
            "Run thresholding with this density or change --analyze-density."
        )
    connectomes_binarized = np.load(bin_path)

    df_graph_measures_bin = pd.DataFrame(
        analyze_connectomes(
            connectomes=connectomes_binarized,
            distance_matrix=dist_mat,
            comm_mode=comm_mode,
            rich_nodes_global=rich_nodes_global,
            rich_top_percent=rich_top_percent,
        )
    )
    df_graph_measures_all = pd.DataFrame(
        analyze_connectomes(
            connectomes=all_connectomes,
            distance_matrix=dist_mat,
            comm_mode=comm_mode,
            rich_nodes_global=rich_nodes_global,
            rich_top_percent=rich_top_percent,
        )
    )

    save_dataframe(
        paths["PREPROCESSED_PATH"]
        / f"df_graph_measures_bin_{n}x{n}_density_{analyze_density}_percent.csv",
        df_graph_measures_bin,
    )
    save_dataframe(
        paths["PREPROCESSED_PATH"]
        / f"df_graph_measures_all_{n}x{n}_density_{analyze_density}_percent.csv",
        df_graph_measures_all,
    )

    return df_graph_measures_bin, df_graph_measures_all, connectomes_binarized


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


def step_consensus_connectomes(
    paths: dict,
    connectomes_binarized: np.ndarray,
    all_connectomes: np.ndarray,
    dist_mat: np.ndarray,
    hemi_id: np.ndarray,
    n: int,
    analyze_density: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # Shapes expected by struct_consensus: nodes x nodes x subjects
    # Our arrays are subjects x nodes x nodes, so we transpose with .T
    consensus_conn_bin = struct_consensus(
        connectomes_binarized.T, distance=dist_mat, hemiid=hemi_id.reshape(-1, 1)
    )
    consensus_density_bin = bct.density_und(consensus_conn_bin)[0]

    consensus_conn_all = struct_consensus(
        all_connectomes.T, distance=dist_mat, hemiid=hemi_id.reshape(-1, 1)
    )
    consensus_density_all = bct.density_und(consensus_conn_all)[0]

    print(f"Consensus (binarized @ {analyze_density}%): {consensus_density_bin*100:.2f}%")
    print(f"Consensus (weighted): {consensus_density_all*100:.2f}%")

    save_numpy(
        paths["PREPROCESSED_PATH"]
        / f"consensus_connectome_bin_{n}x{n}_density_{analyze_density}_percent.npy",
        consensus_conn_bin,
    )
    save_numpy(paths["PREPROCESSED_PATH"] / f"consensus_connectome_all_{n}x{n}.npy", consensus_conn_all)

    return consensus_conn_bin, consensus_conn_all, consensus_density_bin, consensus_density_all


def step_plot_consensus(
    paths: dict,
    consensus_conn_bin: np.ndarray,
    consensus_conn_all: np.ndarray,
    density_bin: float,
    density_all: float,
    analyze_density: int,
) -> None:
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(consensus_conn_bin, cmap="Blues")
    fig.colorbar(im1, ax=ax1)
    ax1.set(xlabel="Region Index", ylabel="Region Index",
            title=f"Consensus (Binarized at {analyze_density}%)\nDensity: {density_bin*100:.2f}%")

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(consensus_conn_all, cmap="Blues")
    fig.colorbar(im2, ax=ax2)
    ax2.set(xlabel="Region Index", ylabel="Region Index",
            title=f"Consensus (Weighted)\nDensity: {density_all*100:.2f}%")

    fig.suptitle(
        "Consensus differs if binarization is performed before consensus (density differs, too)",
        fontsize=12,
    )

    outpng = paths["OUTPUT_PATH"] / f"consensus_connectomes_compare_density_{analyze_density}.png"
    fig.tight_layout()
    fig.savefig(outpng) 
    print(f"Saved plot: {outpng}")


# ----------------------------
# Main orchestrator
# ----------------------------

def main(resolution = None):
    cfg = parse_args()
    if resolution: 
        cfg.resolution = resolution

    paths = step_setup_paths()

    # Distance matrix
    dist_mat = step_distance_matrix(paths, cfg.resolution, cfg.do_plots)

    # Individual connectomes
    all_connectomes, nsub, n = step_load_individual_connectomes(paths, resolution=cfg.resolution)

    # Threshold (binarize) at specified goal densities (hyperparameters)
    step_threshold_connectomes(
        all_connectomes=all_connectomes,
        n=n,
        goal_densities=cfg.goal_densities,
        save_dir=paths["PREPROCESSED_PATH"],
    )

    # Graph measures for selected density & weighted networks
    df_bin, df_all, connectomes_binarized = step_graph_measures(
        paths=paths,
        dist_mat=dist_mat,
        all_connectomes=all_connectomes,
        n=n,
        analyze_density=cfg.analyze_density,
        comm_mode=cfg.comm_mode,
        rich_nodes_global=None,
        rich_top_percent=cfg.rich_top_percent,
    )

    # Identifiers & hemisphere id
    df_identifiers, hemi_id = step_load_identifiers(paths, n)

    # Consensus connectomes (binarized @ analyze_density and weighted)
    consensus_bin, consensus_all, d_bin, d_all = step_consensus_connectomes(
        paths=paths,
        connectomes_binarized=connectomes_binarized,
        all_connectomes=all_connectomes,
        dist_mat=dist_mat,
        hemi_id=hemi_id,
        n=n,
        analyze_density=cfg.analyze_density,
    )

    if cfg.do_plots:
        step_plot_consensus(
            paths=paths,
            consensus_conn_bin=consensus_bin,
            consensus_conn_all=consensus_all,
            density_bin=d_bin,
            density_all=d_all,
            analyze_density=cfg.analyze_density,
        )

    # Print summary to console
    print("\nPipeline completed successfully. Summary:")
    summary = {
        "resolution": cfg.resolution,
        "goal_densities": cfg.goal_densities,
        "analyze_density": cfg.analyze_density,
        "plots": cfg.do_plots,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    print(json.dumps(summary, indent=2))

    # Save summary to a text file
    output_file = paths["OUTPUT_PATH"] / "preprocessing_pipeline_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_file}")



if __name__ == "__main__":
    try:
        main(resolution=68)
        main(resolution=114)
        main(resolution=219)
        main(resolution=448)
        main(resolution=1000)
        
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
