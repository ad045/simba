"""
Compare test GNM networks against all grid-point consensus networks.

For each of the 50 test networks (5 true combos × 10 networks) compute the
distance to every one of the 100 grid-point consensus networks, giving a
full distance landscape per network and per measure.

One CSV per distance measure is written to:
  output/gnm/synthetic_parameter_recovery_grid/comparison_results/
    distances_<measure>.csv

CSV columns
-----------
  true_eta, true_gamma, true_combo_idx, network_id
  dist_to_grid_<i>       for i in 0..99  (flat index: γ-outer, η-inner)
  predicted_grid_idx     argmin of the 100 distances
  predicted_eta          η of the argmin grid point
  predicted_gamma        γ of the argmin grid point
  abs_error              Euclidean distance (predicted − true) in normalised space
  sq_error               Squared Euclidean distance in normalised space

Adding a new distance measure
------------------------------
  Add one line to the `evaluators` dict in main().
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.comparing_connectomes.delta_con_evaluator import DeltaConEvaluator
from src.comparing_connectomes.frobenius_comparer import FrobeniusEvaluator
from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence
from src.comparing_connectomes.hamming_comparer import HammingEvaluator
from src.comparing_connectomes.f1_comparer import F1Evaluator
from src.comparing_connectomes.jaccard_comparer import JaccardEvaluator
from src.comparing_connectomes.spectral_distance_comparer import SpectralDistanceEvaluator
from src.comparing_connectomes.communicability_comparer import CommunicabilityCorrEvaluator
from src.comparing_connectomes.communicability_jsd_comparer import CommunicabilityJSDEvaluator
from src.comparing_connectomes.network_mutual_information_comparer import (
    NetworkMutualInformationEvaluator, DCNetworkMutualInformationEvaluator,
)
from src.comparing_connectomes.netrd_comparer import NetrdEvaluator
from src.comparing_connectomes.energy_comparer import EnergyEvaluator

# Coarse parameter-recovery config (single source of truth, at the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_config import coarse as _cfg
TRUE_PARAM_COMBOS = _cfg.TRUE_PARAM_COMBOS
GRID_COMBOS, GRID_N_ETA, GRID_N_GAMMA = _cfg.GRID_COMBOS, _cfg.GRID_N_ETA, _cfg.GRID_N_GAMMA
ETA_RANGE, GAMMA_RANGE = _cfg.ETA_RANGE, _cfg.GAMMA_RANGE
N_TEST, N_CONSENSUS_GRID = _cfg.N_TEST, _cfg.N_CONSENSUS_GRID
GENERATIVE_RULE_NAME = _cfg.GENERATIVE_RULE_NAME
param_dir_name, net_filename = _cfg.param_dir_name, _cfg.net_filename


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR       = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code") # Path(__file__).parent
DATA_DIR       = ROOT_DIR / "data" / "preprocessed" / "hcp_schaefer_100_dataset"
OUTPUT_DIR     = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_grid"
COMPARISON_DIR = OUTPUT_DIR / "comparison_results"


def load_grid_consensus() -> np.ndarray:
    """Return (n_grid, n_nodes, n_nodes) array of all grid-point consensus networks."""
    grid_dir = OUTPUT_DIR / "grid_consensus"
    consensuses = []
    for eta, gamma in GRID_COMBOS:
        path = grid_dir / param_dir_name(eta, gamma) / "consensus.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing grid consensus: {path}\n"
                "Run run_synthetic_gnm_generation_grid.py first."
            )
        consensuses.append(np.load(path))
    return np.stack(consensuses)  # (n_grid, n_nodes, n_nodes)


def load_test_networks(eta: float, gamma: float) -> List[np.ndarray]:
    test_dir = OUTPUT_DIR / "true_test_networks" / param_dir_name(eta, gamma) / "generated_10"
    nets = []
    for i in range(N_TEST):
        path = test_dir / net_filename(eta, gamma, i)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing test network: {path}\n"
                "Run run_synthetic_gnm_generation_grid.py first."
            )
        nets.append(np.load(path))
    return nets


def normalise(eta: float, gamma: float) -> Tuple[float, float]:
    eta_n   = (eta   - ETA_RANGE[0])   / (ETA_RANGE[1]   - ETA_RANGE[0])
    gamma_n = (gamma - GAMMA_RANGE[0]) / (GAMMA_RANGE[1] - GAMMA_RANGE[0])
    return eta_n, gamma_n


def compute_distances(
    generated: np.ndarray,          # (n_nodes, n_nodes)
    consensus_batch: np.ndarray,    # (n_grid, n_nodes, n_nodes)
    evaluator,
) -> List[float]:
    gen_t = torch.tensor(generated,       dtype=torch.float32).unsqueeze(0)
    tgt_t = torch.tensor(consensus_batch, dtype=torch.float32)
    result = evaluator(gen_t, tgt_t)
    return [float(result[i]) for i in range(len(GRID_COMBOS))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(evaluator, measure_name: str,
                   consensus_batch: np.ndarray) -> pd.DataFrame:
    """Full cross-comparison for one evaluator."""
    records = []
    n_failed = 0
    for true_idx, (true_eta, true_gamma) in enumerate(TRUE_PARAM_COMBOS):
        test_nets = load_test_networks(true_eta, true_gamma)
        true_eta_n, true_gamma_n = normalise(true_eta, true_gamma)

        for net_id, net in enumerate(test_nets):
            try:
                dists    = compute_distances(net, consensus_batch, evaluator)
                pred_idx = int(np.nanargmin(dists))
                pred_eta, pred_gamma = GRID_COMBOS[pred_idx]
                pred_eta_n, pred_gamma_n = normalise(pred_eta, pred_gamma)
                d_eta    = pred_eta_n   - true_eta_n
                d_gamma  = pred_gamma_n - true_gamma_n
                sq_err   = d_eta ** 2 + d_gamma ** 2
                abs_err  = float(np.sqrt(sq_err))
            except Exception as e:
                n_failed += 1
                dists    = [float("nan")] * len(GRID_COMBOS)
                pred_idx = -1
                pred_eta = pred_gamma = float("nan")
                sq_err   = abs_err = float("nan")

            row = {
                "true_eta":           true_eta,
                "true_gamma":         true_gamma,
                "true_combo_idx":     true_idx,
                "network_id":         net_id,
                "predicted_grid_idx": pred_idx,
                "predicted_eta":      pred_eta,
                "predicted_gamma":    pred_gamma,
                "abs_error":          abs_err,
                "sq_error":           float(sq_err),
            }
            for i, d in enumerate(dists):
                row[f"dist_to_grid_{i}"] = d
            records.append(row)

    if n_failed:
        print(f"  Warning: {n_failed} network(s) failed and were recorded as NaN.")
    return pd.DataFrame(records)


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading grid consensus networks …")
    consensus_batch = load_grid_consensus()
    print(f"  Loaded {len(consensus_batch)} consensus networks "
          f"(grid {GRID_N_ETA}×{GRID_N_GAMMA})\n")

    # Energy evaluator needs the HCP distance matrix
    dist_np     = np.load(DATA_DIR / "02_distance_matrices" / "distance_matrix_100.npy")
    dist_tensor = torch.tensor(dist_np, dtype=torch.float32)
    from gnm import evaluation as gnm_eval
    energy_criteria = gnm_eval.MaxCriteria([
        gnm_eval.DegreeKS(),
        gnm_eval.ClusteringKS(),
        gnm_eval.EdgeLengthKS(dist_tensor),
        gnm_eval.BetweennessKS(),
    ])

    # -----------------------------------------------------------------------
    # Evaluator registry
    # -----------------------------------------------------------------------
    evaluators: Dict[str, object] = {
        "delta_con":                       DeltaConEvaluator(),
        "frobenius":                       FrobeniusEvaluator(),
        "portrait":                        PortraitDivergence(),
        "hamming":                         HammingEvaluator(),
        "f1":                              F1Evaluator(),
        "jaccard":                         JaccardEvaluator(),
        "spectral_distance_norm_laplacian": SpectralDistanceEvaluator(method="normalized_laplacian"),
        "spectral_distance_adjacency":     SpectralDistanceEvaluator(method="adjacency"),
        "communicability_corr":            CommunicabilityCorrEvaluator(),
        "communicability_jsd":             CommunicabilityJSDEvaluator(),
        "network_mutual_information":      NetworkMutualInformationEvaluator(),
        "dc_network_mutual_information":   DCNetworkMutualInformationEvaluator(),
        "net_simile":                      NetrdEvaluator(method="net_simile"),
        "resistance":                      NetrdEvaluator(method="resistance"),
        "netrd_non_backtracking_spectral": NetrdEvaluator(method="netrd_non_backtracking_spectral"),
        "energy":                          EnergyEvaluator([energy_criteria]),
    }

    for measure_name, evaluator in evaluators.items():
        out_path = COMPARISON_DIR / f"distances_{measure_name}.csv"
        if out_path.exists():
            print(f"[{measure_name}] already exists — skipping.")
            continue

        print(f"\n{'='*60}\nRunning: {measure_name}\n{'='*60}")
        try:
            df = run_comparison(evaluator, measure_name, consensus_batch)
            df.to_csv(out_path, index=False)
            mae = df["abs_error"].mean()
            print(f"  MAE={mae:.4f}  →  saved to {out_path.name}")
        except Exception as e:
            print(f"  ERROR in {measure_name}: {e}")
            import traceback; traceback.print_exc()

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
