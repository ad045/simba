"""
Compare generated GNM networks against consensus networks at each parameter combination.

For every generated network (50 total: 10 per combo × 5 combos) compute the
distance to each of the 5 consensus networks.  One CSV is written per distance
measure to:

  output/gnm/synthetic_parameter_recovery/comparison_results/
    distances_<measure>.csv

CSV columns
-----------
  true_eta, true_gamma         — true parameters of the generated network
  network_id                   — which of the 10 networks (0–9)
  dist_to_combo_<i>            — distance to the i-th parameter combination's consensus
  predicted_combo_idx          — argmin over the 5 distances
  predicted_eta                — eta of the predicted parameter combination
  predicted_gamma              — gamma of the predicted parameter combination

Adding a new distance measure
------------------------------
  Import its evaluator and add it to the EVALUATORS dict in main().
  Everything else is automatic.
"""

# os.environ must be set before any numeric library is imported
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
    NetworkMutualInformationEvaluator,
    DCNetworkMutualInformationEvaluator,
)
from src.comparing_connectomes.netrd_comparer import NetrdEvaluator
from src.comparing_connectomes.energy_comparer import EnergyEvaluator


# ---------------------------------------------------------------------------
# Configuration — must match run_synthetic_gnm_generation.py
# ---------------------------------------------------------------------------

ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data" / "preprocessed" / "hcp_schaefer_100_dataset"
OUTPUT_DIR = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery"
COMPARISON_DIR = OUTPUT_DIR / "comparison_results"

PARAM_COMBOS: List[Tuple[float, float]] = [
    (-6.9,    1.11),
    ( 4.1,    1.11),
    (-6.9,    0.01),
    ( 4.1,    0.01),
    (-3.734694004058838, 0.595918357372283),
]

GENERATIVE_RULE_NAME = "MatchingIndex"
N_TEST = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def param_dir_name(eta: float, gamma: float) -> str:
    eta_str   = f"{eta}".replace("-", "m").replace(".", "p")
    gamma_str = f"{gamma}".replace("-", "m").replace(".", "p")
    return f"param_eta{eta_str}_gamma{gamma_str}"


def net_filename(eta: float, gamma: float, net_id: int) -> str:
    return f"net_eta{eta}_gamma{gamma}_rule{GENERATIVE_RULE_NAME}_id{net_id:03d}.npy"


def load_consensus(eta: float, gamma: float) -> np.ndarray:
    path = OUTPUT_DIR / param_dir_name(eta, gamma) / "consensus.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Consensus not found at {path}.  "
            "Run run_synthetic_gnm_generation.py first."
        )
    return np.load(path)  # (n_nodes, n_nodes)


def load_test_networks(eta: float, gamma: float) -> List[np.ndarray]:
    test_dir = OUTPUT_DIR / param_dir_name(eta, gamma) / "generated_10"
    nets = []
    for i in range(N_TEST):
        path = test_dir / net_filename(eta, gamma, i)
        if not path.exists():
            raise FileNotFoundError(
                f"Test network not found: {path}.  "
                "Run run_synthetic_gnm_generation.py first."
            )
        nets.append(np.load(path))
    return nets  # list of N_TEST (n_nodes, n_nodes) arrays


def compare_network_to_consensus_batch(
    generated: np.ndarray,        # (n_nodes, n_nodes)
    consensus_batch: np.ndarray,  # (n_combos, n_nodes, n_nodes)
    evaluator,
) -> List[float]:
    """Return distances from generated to each consensus, as a list."""
    gen_tensor = torch.tensor(generated, dtype=torch.float32).unsqueeze(0)  # (1, n, n)
    tgt_tensor = torch.tensor(consensus_batch, dtype=torch.float32)          # (k, n, n)
    result_dict = evaluator(gen_tensor, tgt_tensor)
    return [float(result_dict[i]) for i in range(len(PARAM_COMBOS))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(evaluator, measure_name: str) -> pd.DataFrame:
    """Cross-compare all generated networks against all consensus networks."""
    consensus_batch = np.stack(
        [load_consensus(eta, gamma) for eta, gamma in PARAM_COMBOS]
    )

    records = []
    for true_idx, (true_eta, true_gamma) in enumerate(PARAM_COMBOS):
        print(f"  Combo {true_idx+1}/{len(PARAM_COMBOS)}: eta={true_eta}, gamma={true_gamma}")
        test_nets = load_test_networks(true_eta, true_gamma)

        for net_id, net in enumerate(test_nets):
            dists    = compare_network_to_consensus_batch(net, consensus_batch, evaluator)
            pred_idx = int(np.argmin(dists))
            pred_eta, pred_gamma = PARAM_COMBOS[pred_idx]

            row = {
                "true_eta":            true_eta,
                "true_gamma":          true_gamma,
                "true_combo_idx":      true_idx,
                "network_id":          net_id,
                "predicted_combo_idx": pred_idx,
                "predicted_eta":       pred_eta,
                "predicted_gamma":     pred_gamma,
                "correct":             int(pred_idx == true_idx),
            }
            for i, d in enumerate(dists):
                row[f"dist_to_combo_{i}"] = d

            records.append(row)

    return pd.DataFrame(records)


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    # Energy evaluator needs the HCP distance matrix to build its criteria
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
    # Evaluator registry — add one line here for a new distance measure
    # -----------------------------------------------------------------------
    evaluators: Dict[str, object] = {
        "delta_con":                     DeltaConEvaluator(),
        "frobenius":                     FrobeniusEvaluator(),
        "portrait":                      PortraitDivergence(),
        "hamming":                       HammingEvaluator(),
        "f1":                            F1Evaluator(),
        "jaccard":                       JaccardEvaluator(),
        "spectral_distance_norm_laplacian": SpectralDistanceEvaluator(method="normalized_laplacian"),
        "spectral_distance_adjacency":   SpectralDistanceEvaluator(method="adjacency"),
        "communicability_corr":          CommunicabilityCorrEvaluator(),
        "communicability_jsd":           CommunicabilityJSDEvaluator(),
        "network_mutual_information":    NetworkMutualInformationEvaluator(),
        "dc_network_mutual_information": DCNetworkMutualInformationEvaluator(),
        "net_simile":                    NetrdEvaluator(method="net_simile"),
        "resistance":                    NetrdEvaluator(method="resistance"),
        "netrd_non_backtracking_spectral": NetrdEvaluator(method="netrd_non_backtracking_spectral"),
        "energy":                        EnergyEvaluator([energy_criteria]),
    }

    for measure_name, evaluator in evaluators.items():
        out_path = COMPARISON_DIR / f"distances_{measure_name}.csv"
        if out_path.exists():
            print(f"\n[{measure_name}] already exists — skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {measure_name}")
        print(f"{'='*60}")
        try:
            df = run_comparison(evaluator, measure_name)
            df.to_csv(out_path, index=False)
            acc = df["correct"].mean()
            print(f"  Accuracy: {acc:.2%}  ({df['correct'].sum()}/{len(df)} correct)")
            print(f"  Saved → {out_path}")
        except Exception as e:
            print(f"  ERROR running {measure_name}: {e}")
            import traceback; traceback.print_exc()

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
