"""
Generate GNM networks at specified eta-gamma combinations and build consensus networks.

Step 1: Generate N_TEST (10) independent networks per parameter combination — saved
        for downstream parameter-recovery comparisons.

Step 2: Generate N_CONSENSUS (100) additional independent networks per combination
        and build a structural consensus from them (netneurotools struct_consensus).

All output lives under:
  output/gnm/synthetic_parameter_recovery/<param_dir>/
    generated_10/   <- N_TEST saved networks
    consensus_100/  <- N_CONSENSUS networks used for consensus
    consensus.npy   <- final binarised consensus (100x100)

Dataset: HCP Schaefer-100 (100 nodes, matching-index rule, ~495 edges).
"""

import sys
import os

# Disable threading in numeric libraries before imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from gnm.model import BinaryGenerativeParameters
from gnm import generative_rules
from gnm.fitting import RunConfig, perform_run
from netneurotools.networks import struct_consensus, threshold_network


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR  = ROOT_DIR / "data" / "preprocessed" / "hcp_schaefer_100_dataset"
SEED_PATH = ROOT_DIR / "data" / "preprocessed" / "seeds" / "mst_schaeffer.npy"
OUTPUT_DIR = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery"

PARAM_COMBOS = [
    (-6.9,    0.89),
    ( 1.9,    0.89),
    (-6.9,    0.01),
    ( 1.9,    0.01),
    (-3.734694004058838, 0.595918357372283),
]

N_TEST      = 10   # networks saved for parameter-recovery test
N_CONSENSUS = 100  # networks used to build the consensus

GENERATIVE_RULE_NAME = "MatchingIndex"
LAMBDAH  = 1.0
DEVICE   = "cpu"

# Schaefer-100: 50 left-hemisphere + 50 right-hemisphere nodes
HEMIID = np.array([0] * 50 + [1] * 50).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def param_dir_name(eta: float, gamma: float) -> str:
    """Filesystem-safe directory name for an eta-gamma combination."""
    eta_str   = f"{eta}".replace("-", "m").replace(".", "p")
    gamma_str = f"{gamma}".replace("-", "m").replace(".", "p")
    return f"param_eta{eta_str}_gamma{gamma_str}"


def net_filename(eta: float, gamma: float, net_id: int) -> str:
    return f"net_eta{eta}_gamma{gamma}_rule{GENERATIVE_RULE_NAME}_id{net_id:03d}.npy"


def generate_one_network(
    eta: float,
    gamma: float,
    n_edges: int,
    distance_matrix: torch.Tensor,
    seed_adjacency_matrix: torch.Tensor,
) -> np.ndarray:
    """Generate a single GNM network; returns (n_nodes, n_nodes) array."""
    bp = BinaryGenerativeParameters(
        eta=eta,
        gamma=gamma,
        lambdah=LAMBDAH,
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
    return exp.model.adjacency_matrix.cpu().numpy()[0]  # (n_nodes, n_nodes)


def load_or_generate(
    path: Path,
    eta: float,
    gamma: float,
    net_id: int,
    n_edges: int,
    distance_matrix: torch.Tensor,
    seed_adjacency_matrix: torch.Tensor,
) -> np.ndarray:
    """Load network from disk if it exists, otherwise generate and save it."""
    if path.exists():
        return np.load(path)
    net = generate_one_network(eta, gamma, n_edges, distance_matrix, seed_adjacency_matrix)
    np.save(path, net)
    return net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load HCP distance matrix and consensus (to infer target edge count)
    dist_np = np.load(DATA_DIR / "02_distance_matrices" / "distance_matrix_100.npy")
    consensus_hcp = np.load(
        DATA_DIR / "01_connectomes" / "01_consensus_bin_density_10_percent_100.npy"
    )
    # MST seed: (1, 100, 100), 99 edges — the GNM grows on top of this
    seed_np = np.load(SEED_PATH)  # (1, 100, 100)
    seed_tensor = torch.tensor(seed_np, dtype=torch.float32)  # (1, 100, 100)

    total_edges = int(consensus_hcp[0].sum() / 2)          # 495
    n_edges     = total_edges - (dist_np.shape[0] - 1)     # 495 - 99 = 396
    print(f"HCP consensus edges: {total_edges}, MST seed edges: {dist_np.shape[0]-1}, "
          f"GNM iterations: {n_edges}")

    dist_tensor = torch.tensor(dist_np, dtype=torch.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for eta, gamma in PARAM_COMBOS:
        print(f"\n{'='*60}")
        print(f"eta={eta:8.4f}  gamma={gamma:7.4f}")
        print(f"{'='*60}")

        pdir = OUTPUT_DIR / param_dir_name(eta, gamma)
        test_dir      = pdir / "generated_10"
        consensus_dir = pdir / "consensus_100"
        test_dir.mkdir(parents=True, exist_ok=True)
        consensus_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Step 1 — N_TEST networks (for parameter recovery)
        # ------------------------------------------------------------------
        print(f"  Step 1: generating {N_TEST} test networks …")
        for i in tqdm(range(N_TEST), desc="  test nets", leave=False):
            path = test_dir / net_filename(eta, gamma, i)
            load_or_generate(path, eta, gamma, i, n_edges, dist_tensor, seed_tensor)

        print(f"  Saved {N_TEST} test networks to {test_dir}")

        # ------------------------------------------------------------------
        # Step 2 — N_CONSENSUS networks → structural consensus
        # ------------------------------------------------------------------
        consensus_path = pdir / "consensus.npy"

        if consensus_path.exists():
            consensus_bin = np.load(consensus_path)
            print(f"  Consensus already exists ({consensus_path.name}), "
                  f"edges: {consensus_bin.sum() / 2:.0f}  — skipping generation")
        else:
            print(f"  Step 2: generating {N_CONSENSUS} consensus-source networks …")
            consensus_nets = []
            for i in tqdm(range(N_CONSENSUS), desc="  consensus nets", leave=False):
                path = consensus_dir / net_filename(eta, gamma, i)
                net = load_or_generate(path, eta, gamma, i, n_edges, dist_tensor, seed_tensor)
                consensus_nets.append(net)

            print("  Building structural consensus …")
            # struct_consensus expects (n_nodes, n_nodes, n_networks)
            consensus_nets_arr = np.array(consensus_nets)        # (n_networks, n_nodes, n_nodes)
            consensus_weighted = struct_consensus(
                np.transpose(consensus_nets_arr, (1, 2, 0)),     # (n_nodes, n_nodes, n_networks)
                dist_np,
                hemiid=HEMIID,
                weighted=True,
            )
            # binarise to same density as HCP consensus (~10 %)
            consensus_bin = threshold_network(consensus_weighted, retain=10)  # (100, 100) — MST-guaranteed connected
            np.save(consensus_path, consensus_bin)
            print(f"  Saved consensus → {consensus_path},  edges: {consensus_bin.sum() / 2:.0f}")

    print("\n" + "=" * 60)
    print("Generation complete.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
