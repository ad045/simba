"""
Generate a spatially-ordered ring lattice network as the 'economy' vertex anchor.

Strategy: Given the brain distance matrix (Schaefer-100 parcellation, 100 nodes),
connect every node to its k nearest spatial neighbours such that the resulting
network has ≈10 % density.  This is the most wiring-economical graph possible for
these coordinates — an ideal counterpart to the diffusion/routing/propagation
networks that optimise communication.

Output:
  data/preprocessed/ring_lattice_networks/01_connectomes/ring_lattice_10_percent.npy
      shape (1, 100, 100)  — one deterministic binarised adjacency matrix
  data/preprocessed/ring_lattice_networks/02_distance_matrices/distance_matrix_100.npy
      copy of the HCP Schaefer-100 distance matrix
"""

import numpy as np
from pathlib import Path
import shutil
import matplotlib.pyplot as plt 

BASE = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")

SRC_DIST = (BASE / "data/preprocessed/hcp_schaefer_100_dataset"
            / "02_distance_matrices/distance_matrix_100.npy")

OUT_CONN = BASE / "data/preprocessed/ring_lattice_networks/01_connectomes"
OUT_DIST = BASE / "data/preprocessed/ring_lattice_networks/02_distance_matrices"
OUT_CONN.mkdir(parents=True, exist_ok=True)
OUT_DIST.mkdir(parents=True, exist_ok=True)

# ── Load distance matrix ────────────────────────────────────────────────────

dist = np.load(SRC_DIST)
n = dist.shape[0]  # 100
assert dist.shape == (n, n), "Expected square distance matrix"

# ── Build k-nearest-neighbour adjacency ────────────────────────────────────
# Target 10 % density: edges / (n*(n-1)/2) = 0.10  →  edges ≈ 495

target_edges = int(round(n * (n - 1) / 2 * 0.10))
print(f"n={n}  target_edges={target_edges}  target_density={target_edges / (n*(n-1)/2):.4f}")

# Collect all upper-triangle (i, j, distance) triples and sort by distance
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((dist[i, j], i, j))

pairs.sort(key=lambda x: x[0])
print(pairs)

# Keep the closest `target_edges` pairs
adj = np.zeros((n, n), dtype=np.float64)
for d, i, j in pairs[:target_edges]:
    adj[i, j] = 1.0
    adj[j, i] = 1.0

actual_density = adj.sum() / 2 / (n * (n - 1) / 2)
print(f"Actual density: {actual_density:.4f}  (edges={int(adj.sum()/2)})")
print(f"Min distance included: {pairs[target_edges-1][0]:.4f}")
print(f"Max distance included: {pairs[target_edges-1][0]:.4f}")

# ── Save ────────────────────────────────────────────────────────────────────

print("Min degree:", adj.sum(axis=0).min())
print("Max degree:", adj.sum(axis=0).max())
# Show it 
plt.imshow(adj)
plt.show() 

conn_arr = adj[np.newaxis, :, :]          # shape (1, 100, 100)
out_path = OUT_CONN / "ring_lattice_10_percent.npy"
np.save(out_path, conn_arr)
print(f"Saved connectome: {out_path}  shape={conn_arr.shape}")

shutil.copy(SRC_DIST, OUT_DIST / "distance_matrix_100.npy")
print(f"Copied distance matrix to: {OUT_DIST / 'distance_matrix_100.npy'}")
