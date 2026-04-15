"""
Generate Erdős–Rényi G(n, m) random networks as the null-baseline reference.

Each instance draws exactly m = 495 uniformly-random edges from the set of all
C(100, 2) = 4 950 possible pairs, giving a density of exactly 10 %.
100 instances are generated to represent the spread of the null model in the morphospace.

Output:
  data/preprocessed/erdos_renyi_networks/01_connectomes/erdos_renyi_10_percent.npy
      shape (100, 100, 100)  — 100 binarised adjacency matrices
  data/preprocessed/erdos_renyi_networks/02_distance_matrices/distance_matrix_100.npy
      copy of the HCP Schaefer-100 distance matrix (needed for spatial metrics)
"""

import numpy as np
from pathlib import Path
import shutil

BASE = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")

SRC_DIST = (BASE / "data/preprocessed/hcp_schaefer_100_dataset"
            / "02_distance_matrices/distance_matrix_100.npy")

OUT_CONN = BASE / "data/preprocessed/erdos_renyi_networks/01_connectomes"
OUT_DIST = BASE / "data/preprocessed/erdos_renyi_networks/02_distance_matrices"
OUT_CONN.mkdir(parents=True, exist_ok=True)
OUT_DIST.mkdir(parents=True, exist_ok=True)

# ── Parameters ───────────────────────────────────────────────────────────────

N_NODES    = 100
N_NETWORKS = 100
DENSITY    = 0.10
SEED       = 42

n_possible = N_NODES * (N_NODES - 1) // 2          # 4 950
n_edges    = int(round(n_possible * DENSITY))        # 495
print(f"n_nodes={N_NODES}  n_edges={n_edges}  density={n_edges/n_possible:.4f}")

# Pre-compute all (i, j) upper-triangle index pairs as a list
all_pairs = [(i, j) for i in range(N_NODES) for j in range(i + 1, N_NODES)]
all_pairs = np.array(all_pairs)  # shape (4950, 2)

# ── Generate networks ────────────────────────────────────────────────────────

rng  = np.random.default_rng(SEED)
nets = np.zeros((N_NETWORKS, N_NODES, N_NODES), dtype=np.float64)

for k in range(N_NETWORKS):
    chosen = rng.choice(len(all_pairs), size=n_edges, replace=False)
    for idx in chosen:
        i, j = all_pairs[idx]
        nets[k, i, j] = 1.0
        nets[k, j, i] = 1.0

actual_densities = nets.sum(axis=(1, 2)) / 2 / n_possible
print(f"All densities exactly {DENSITY:.2f}: {np.allclose(actual_densities, DENSITY)}")
print(f"Mean density: {actual_densities.mean():.4f}")

# ── Save ────────────────────────────────────────────────────────────────────

out_path = OUT_CONN / "erdos_renyi_10_percent.npy"
np.save(out_path, nets)
print(f"Saved connectomes: {out_path}  shape={nets.shape}")

shutil.copy(SRC_DIST, OUT_DIST / "distance_matrix_100.npy")
print(f"Copied distance matrix to: {OUT_DIST / 'distance_matrix_100.npy'}")
