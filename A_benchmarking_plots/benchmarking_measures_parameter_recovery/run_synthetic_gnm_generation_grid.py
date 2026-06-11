"""
Generate GNM networks for the continuous-landscape parameter-recovery experiment.

Two things are produced:

  1. Test networks  — N_TEST (10) independent networks at each of the 5 TRUE
                      parameter combinations, used to probe the landscape.

  2. Grid consensus — N_CONSENSUS_GRID (30) independent networks at every point
                      on a GRID_N_ETA × GRID_N_GAMMA grid, collapsed into one
                      structural consensus per grid point.

Output layout
-------------
  output/gnm/synthetic_parameter_recovery_grid/
    true_test_networks/
      param_<dir>/
        generated_10/
          net_eta<η>_gamma<γ>_ruleMatchingIndex_id000..009.npy
    grid_consensus/
      param_<dir>/
        consensus_30/
          net_eta<η>_gamma<γ>_ruleMatchingIndex_id000..029.npy
        consensus.npy          ← (100, 100) binarised consensus

Dataset: HCP Schaefer-100 (100 nodes, MatchingIndex rule, MST seed, ~495 edges).
All scripts are idempotent — existing files are skipped.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from gnm.model import BinaryGenerativeParameters
from gnm import generative_rules
from gnm.fitting import RunConfig, perform_run
from netneurotools.networks import struct_consensus, threshold_network

from experiments.benchmarking_measures_parameter_recovery.gnm_grid_config import (
    TRUE_PARAM_COMBOS, GRID_COMBOS, GRID_N_ETA, GRID_N_GAMMA,
    N_TEST, N_CONSENSUS_GRID, GENERATIVE_RULE_NAME,
    param_dir_name, net_filename,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR   = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")# Path(__file__).parent.parent.parent
DATA_DIR   = ROOT_DIR / "data" / "preprocessed" / "hcp_schaefer_100_dataset"
SEED_PATH  = ROOT_DIR / "data" / "preprocessed" / "seeds" / "mst_schaeffer.npy"
OUTPUT_DIR = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_grid"

LAMBDAH = 1.0
DEVICE  = "cpu"

HEMIID = np.array([0] * 50 + [1] * 50).reshape(-1, 1)


def generate_one_network(
    eta: float, gamma: float,
    n_edges: int,
    distance_matrix: torch.Tensor,
    seed_adjacency_matrix: torch.Tensor,
) -> np.ndarray:
    bp = BinaryGenerativeParameters(
        eta=eta, gamma=gamma, lambdah=LAMBDAH,
        distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw",
        heterochronicity_relationship_type="powerlaw",
        generative_rule=generative_rules.MatchingIndex(),
        num_iterations=n_edges,
    )
    rc = RunConfig(
        binary_parameters=bp,
        distance_matrix=distance_matrix,
        seed_adjacency_matrix=seed_adjacency_matrix,
        num_simulations=1,
    )
    exp = perform_run(rc, save_model=True, save_run_history=False,
                      device=torch.device(DEVICE))
    return exp.model.adjacency_matrix.cpu().numpy()[0]


def load_or_generate(
    path: Path,
    eta: float, gamma: float, net_id: int,
    n_edges: int,
    distance_matrix: torch.Tensor,
    seed_adjacency_matrix: torch.Tensor,
) -> np.ndarray:
    if path.exists():
        return np.load(path)
    net = generate_one_network(eta, gamma, n_edges, distance_matrix, seed_adjacency_matrix)
    np.save(path, net)
    return net


def build_consensus(nets: List[np.ndarray], dist_np: np.ndarray) -> np.ndarray:
    """Build binarised structural consensus from a list of (n,n) networks."""
    arr = np.array(nets)                              # (n_networks, n_nodes, n_nodes)
    weighted = struct_consensus(
        np.transpose(arr, (1, 2, 0)),                 # (n_nodes, n_nodes, n_networks)
        dist_np,
        hemiid=HEMIID,
        weighted=True,
    )
    return threshold_network(weighted, retain=10)     # (n_nodes, n_nodes) — MST-guaranteed connected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dist_np       = np.load(DATA_DIR / "02_distance_matrices" / "distance_matrix_100.npy")
    consensus_hcp = np.load(DATA_DIR / "01_connectomes" / "01_consensus_bin_density_10_percent_100.npy")
    seed_np       = np.load(SEED_PATH)

    total_edges = int(consensus_hcp[0].sum() / 2)          # 495
    n_edges     = total_edges - (dist_np.shape[0] - 1)     # 396
    print(f"Edges: total={total_edges}, MST seed={dist_np.shape[0]-1}, GNM iters={n_edges}")
    print(f"Grid: {GRID_N_ETA}×{GRID_N_GAMMA} = {len(GRID_COMBOS)} points, "
          f"{N_CONSENSUS_GRID} networks each")
    print(f"Test: {len(TRUE_PARAM_COMBOS)} true combos × {N_TEST} networks\n")

    dist_tensor = torch.tensor(dist_np, dtype=torch.float32)
    seed_tensor = torch.tensor(seed_np, dtype=torch.float32)

    test_dir_root = OUTPUT_DIR / "true_test_networks"
    grid_dir_root = OUTPUT_DIR / "grid_consensus"
    test_dir_root.mkdir(parents=True, exist_ok=True)
    grid_dir_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Test networks at TRUE_PARAM_COMBOS
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Test networks at true parameter combinations")
    print("=" * 60)
    for eta, gamma in TRUE_PARAM_COMBOS:
        pdir     = test_dir_root / param_dir_name(eta, gamma)
        nets_dir = pdir / "generated_10"
        nets_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(N_TEST),
                      desc=f"  η={eta:.2f} γ={gamma:.4f}", leave=False):
            path = nets_dir / net_filename(eta, gamma, i)
            load_or_generate(path, eta, gamma, i, n_edges, dist_tensor, seed_tensor)
    print(f"  Done. ({len(TRUE_PARAM_COMBOS) * N_TEST} test networks)\n")

    # ------------------------------------------------------------------
    # Step 2 — Consensus networks at each grid point
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Step 2: Grid consensus ({len(GRID_COMBOS)} points)")
    print("=" * 60)
    for grid_idx, (eta, gamma) in enumerate(tqdm(GRID_COMBOS, desc="Grid points")):
        pdir           = grid_dir_root / param_dir_name(eta, gamma)
        nets_dir       = pdir / f"consensus_{N_CONSENSUS_GRID}"
        consensus_path = pdir / "consensus.npy"
        nets_dir.mkdir(parents=True, exist_ok=True)

        if consensus_path.exists():
            continue  # already built

        nets = []
        for i in range(N_CONSENSUS_GRID):
            path = nets_dir / net_filename(eta, gamma, i)
            net  = load_or_generate(path, eta, gamma, i, n_edges, dist_tensor, seed_tensor)
            nets.append(net)

        consensus_bin = build_consensus(nets, dist_np)
        np.save(consensus_path, consensus_bin)

    print(f"\nDone. Output → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
