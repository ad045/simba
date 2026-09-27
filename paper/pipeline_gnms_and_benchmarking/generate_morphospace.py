"""
Regenerate the morphospace at a true 10% density (495 edges)
============================================================

The paper's morphospace run (`105_distance_metrics_mst_animal_0`) ran 495 wiring
iterations *on top of* the 99-edge MST seed, so every network carries 594 edges
(12%) against the 495-edge (10%) consensus. This script regenerates the same
grid with the seed edges subtracted from the iteration count - 396 iterations,
99 + 396 = 495 edges - so both sides of every comparison sit at 10%.

Everything else is held identical to the original run: the same 50 x 50
`torch.linspace` grid over eta in [-8, 3] and gamma in [-0.1, 1], the same MST
seed, matching-index rule, powerlaw relationships, lambda = 0, 10 replicates per
cell, and the same `net_eta{eta}_gamma{gamma}_ruleMatchingIndex_id{NNN}.npy`
filenames (the eta/gamma in the name come from the same float32 linspace and the
same 14-decimal floor, so filenames match the old run cell for cell).

The 10 replicates of a cell are run as one batch of 10 simulations rather than as
10 separate single-simulation runs. The model samples each batch element
independently (model.py, `run_model`), so this is the same experiment, just
vectorised.

Usage
-----
    conda activate ma_thesis
    python pipeline_gnms_and_benchmarking/generate_morphospace.py --selftest     # checks only, no writes
    python pipeline_gnms_and_benchmarking/generate_morphospace.py                # full 25,000 networks
    python pipeline_gnms_and_benchmarking/generate_morphospace.py --n-cells 4    # short smoke run

Safe to interrupt and restart: cells whose 10 files already exist are skipped.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_config import DIST_MATRIX_PATH, CONSENSUS_PATH

SEED_PATH = ROOT / "data" / "preprocessed" / "seeds" / "mst_schaeffer.npy"
EXP_NAME = "106_distance_metrics_mst_animal_0_density10"
OUT_DIR = ROOT / "output" / "gnm" / "hcp_schaefer_100_dataset" / EXP_NAME

# the original sweep's geometry, from configs/config_gnm_run_hcp_with_seeds.yaml
ETA_RANGE = (-8.0, 3.0)
GAMMA_RANGE = (-0.1, 1.0)
GRID_SIZE = 50               # sqrt(n_samples), n_samples = 2500
N_REPLICATES = 10            # gnm.num_simulations
RULE_NAME = "MatchingIndex"


def floor_to_14_decimal_places(number: float) -> float:
    """Byte-identical to the orchestrator's helper, so filenames match."""
    return np.floor(number * 10e14) / 10e14


def grid_values():
    etas = torch.linspace(*ETA_RANGE, GRID_SIZE)
    gammas = torch.linspace(*GAMMA_RANGE, GRID_SIZE)
    return etas, gammas


def filename(eta: float, gamma: float, rep: int) -> str:
    return f"net_eta{eta}_gamma{gamma}_rule{RULE_NAME}_id{rep:03d}.npy"


def n_iterations(seed: np.ndarray, target: np.ndarray) -> int:
    """Edges the model must add: the consensus' edges minus the seed's."""
    return int(target.sum() // 2) - int(seed.sum() // 2)


def run_cell(eta: float, gamma: float, n_iter: int,
             distance_matrix: torch.Tensor, seed_batch: torch.Tensor) -> np.ndarray:
    from gnm import GenerativeNetworkModel
    from gnm.model import BinaryGenerativeParameters
    from gnm import generative_rules

    params = BinaryGenerativeParameters(
        eta=eta,
        gamma=gamma,
        lambdah=0.0,
        distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw",
        heterochronicity_relationship_type="powerlaw",
        generative_rule=generative_rules.MatchingIndex(),
        num_iterations=n_iter,
    )
    model = GenerativeNetworkModel(
        binary_parameters=params,
        num_simulations=seed_batch.shape[0],
        seed_adjacency_matrix=seed_batch,
        distance_matrix=distance_matrix,
        device=torch.device("cpu"),
        verbose=False,
    )
    model.run_model()
    return model.adjacency_matrix.cpu().numpy()


def load_inputs():
    D = np.load(DIST_MATRIX_PATH)
    seed = np.load(SEED_PATH)
    seed = seed[0] if seed.ndim == 3 else seed
    target = np.load(CONSENSUS_PATH)
    target = target[0] if target.ndim == 3 else target
    return D, seed, target


def _generate_and_save(eta_raw, gamma_raw, names, n_iter, D, seed, nets_dir) -> None:
    """One grid cell: generate the replicate batch, check it, write the files."""
    dist = torch.tensor(D, dtype=torch.float32)
    seed_batch = torch.tensor(seed, dtype=torch.float32).unsqueeze(0).repeat(len(names), 1, 1)
    nets = run_cell(eta_raw, gamma_raw, n_iter, dist, seed_batch)
    for name, A in zip(names, nets):
        assert A.sum() / 2 == 495, f"{name}: {A.sum() / 2} edges"
        np.save(nets_dir / name, A[np.newaxis, :, :])


def selftest() -> None:
    """Two cells at the grid corners, checked for edge count and seed retention."""
    D, seed, target = load_inputs()
    n_iter = n_iterations(seed, target)
    assert n_iter == 396, f"expected 396 iterations, got {n_iter}"

    etas, gammas = grid_values()
    # the filenames must line up with the 12% run, cell for cell
    old = ROOT / "output" / "gnm" / "hcp_schaefer_100_dataset" / \
        "105_distance_metrics_mst_animal_0" / "generated_networks"
    if old.exists():
        have = set(os.listdir(old))
        want = {filename(floor_to_14_decimal_places(float(e)),
                         floor_to_14_decimal_places(float(g)), r)
                for e in etas for g in gammas for r in range(N_REPLICATES)}
        assert want == have, f"grid mismatch: {len(want - have)} missing, {len(have - want)} extra"

    dist = torch.tensor(D, dtype=torch.float32)
    seed_batch = torch.tensor(seed, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    for eta, gamma in [(-3.0, 0.4), (2.0, 0.9)]:
        nets = run_cell(eta, gamma, n_iter, dist, seed_batch)
        assert nets.shape == (2, 100, 100), nets.shape
        for A in nets:
            assert A.sum() / 2 == 495, A.sum() / 2
            assert np.array_equal(A, A.T), "network is not symmetric"
            assert A.diagonal().sum() == 0, "network has self-loops"
            assert (A[seed > 0] > 0).all(), "seed edges were not retained"
    print("selftest ok: 396 iterations, 495 edges, seed retained, grid matches the 12% run")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-cells", type=int, default=None,
                    help="only the first N grid cells (smoke run)")
    ap.add_argument("--n-jobs", type=int, default=10)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    from tqdm import tqdm
    from joblib import Parallel, delayed

    D, seed, target = load_inputs()
    n_iter = n_iterations(seed, target)
    print(f"seed {int(seed.sum() // 2)} edges + {n_iter} iterations "
          f"= {int(target.sum() // 2)} edges (10% of 100 nodes)")

    nets_dir = OUT_DIR / "generated_networks"
    nets_dir.mkdir(parents=True, exist_ok=True)

    etas, gammas = grid_values()
    cells = [(float(e), float(g)) for e in etas for g in gammas]
    if args.n_cells:
        cells = cells[:args.n_cells]

    rows = []
    todo = []
    for eta_raw, gamma_raw in cells:
        eta = floor_to_14_decimal_places(eta_raw)
        gamma = floor_to_14_decimal_places(gamma_raw)
        names = [filename(eta, gamma, r) for r in range(N_REPLICATES)]
        rows += [{"eta": eta, "gamma": gamma, "id": r, "filename": n}
                 for r, n in enumerate(names)]
        if not all((nets_dir / n).exists() for n in names):
            todo.append((eta_raw, gamma_raw, names))

    print(f"{len(cells) - len(todo)} of {len(cells)} cells already on disk, "
          f"{len(todo)} to generate on {args.n_jobs} workers")

    if todo:
        Parallel(n_jobs=args.n_jobs)(
            delayed(_generate_and_save)(eta_raw, gamma_raw, names, n_iter, D, seed, nets_dir)
            for eta_raw, gamma_raw, names in tqdm(todo, desc="grid cells")
        )

    # the comparison runner reads this to fix the row order of every summary file
    ref = OUT_DIR / f"all_metrics_for_{EXP_NAME}.csv"
    pd.DataFrame(rows).to_csv(ref, index=False)
    print(f"{len(rows)} networks in {nets_dir}")
    print(f"reference csv -> {ref}")


if __name__ == "__main__":
    main()
