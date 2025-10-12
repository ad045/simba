# # Basiert noch auf "build_weight_matrix_from_bin_conn" -> also random bio no rank matrix (ganz alte version) 
# # Anscheinend noch nicht getestet


import os, time, warnings, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from numba import njit
from scipy.stats import ks_2samp

from utils.saving_and_finding_files import time_stamp_for_saving

# ESN + graph-measure imports
# from src.ESNs.generate_weight_matrices_bio_no_rank import build_weight_matrix_from_bin_conn
from src.ESNs.X_test_memory_capacity import evaluate_memory_capacity
from src.ESNs.X_test_sequence_recall import evaluate_sequence_recall
from src.structural_analysis.graph_measures import analyze_connectomes

# =============================================================================
# 1) NUMBA HELPERS (binary / integer-friendly)
# =============================================================================

@njit(cache=True, fastmath=True)
def _compute_homophily_bin(A_bin: np.ndarray) -> np.ndarray:
    # Use float32 to cut memory bandwidth; good enough for probabilities
    Af = A_bin.astype(np.float32)
    shared = Af @ Af            # (i,j): # shared neighbors
    deg = Af.sum(axis=1)        # degrees
    denom = deg[:, None] + deg[None, :] - shared
    H = np.where(denom > 0.0, shared / denom, 0.0)
    # zero diagonal
    n = H.shape[0]
    for i in range(n):
        H[i, i] = 0.0
    return H

@njit(cache=True, fastmath=True)
def _rand_choice_weighted(p_vec):
    tot = 0.0
    x = np.random.random()
    for k in range(p_vec.size):
        tot += p_vec[k]
        if x < tot:
            return k
    return p_vec.size - 1

@njit(cache=True, fastmath=True)
def _upper_indices(n: int):
    # Return arrays of i,j and flat indices for the strict upper triangle
    m = n*(n-1)//2
    I = np.empty(m, dtype=np.int32)
    J = np.empty(m, dtype=np.int32)
    k = 0
    for i in range(n-1):
        for j in range(i+1, n):
            I[k] = i
            J[k] = j
            k += 1
    return I, J

@njit(cache=True, fastmath=True)
def generate_gnm_numba(A_seed, target_m, dist, gamma, eta, eps: float = 1e-12):
    """
    Faster: operates on the strict upper triangle only (boolean availability mask),
    avoids rebuilding 'mask_flat' with nested loops, and updates availability in O(1).
    """
    n = A_seed.shape[0]
    A = A_seed.copy()
    # ensure symmetry & binary
    for i in range(n):
        A[i, i] = 0
        for j in range(i+1, n):
            v = 1 if (A[i, j] != 0 or A[j, i] != 0) else 0
            A[i, j] = v
            A[j, i] = v

    # Precompute distance cost term (float32) and zero diagonal
    D = dist.astype(np.float32)
    cost_term = np.power(np.maximum(D, eps), eta)
    for i in range(n):
        cost_term[i, i] = 0.0

    # Upper-tri bookkeeping
    Iu, Ju = _upper_indices(n)
    M = Iu.size

    # availability: True means edge (i<j) can still be added
    avail = np.ones(M, dtype=np.uint8)
    # mark already present edges as unavailable
    for e in range(M):
        if A[Iu[e], Ju[e]] != 0:
            avail[e] = 0

    # current #edges
    m = 0
    for e in range(M):
        m += 1 if avail[e] == 0 else 0

    # main loop
    while m < target_m:
        # homophily (float32)
        H = _compute_homophily_bin(A.astype(np.uint8)).astype(np.float32)

        # compute probability on available upper-tri entries only
        # prob = cost * max(H, eps)**gamma
        # gather values
        p_vec = np.empty(M, dtype=np.float32)
        for e in range(M):
            if avail[e]:
                i = Iu[e]; j = Ju[e]
                h = H[i, j]
                if h < eps:
                    h = eps
                p_vec[e] = cost_term[i, j] * np.power(h, gamma)
            else:
                p_vec[e] = 0.0

        # choose candidate
        p_sum = 0.0
        for e in range(M):
            p_sum += p_vec[e]
        if p_sum == 0.0:
            # uniform among available
            # find count
            count = 0
            for e in range(M):
                if avail[e]:
                    count += 1
            if count == 0:
                break
            # pick k-th available
            # (simple linear selection to keep numba-friendly)
            kth = np.random.randint(count)
            idx = -1
            c = 0
            for e in range(M):
                if avail[e]:
                    if c == kth:
                        idx = e
                        break
                    c += 1
        else:
            # normalize once
            inv_sum = 1.0 / p_sum
            for e in range(M):
                p_vec[e] *= inv_sum
            idx = _rand_choice_weighted(p_vec)

        if idx < 0:
            break

        i = Iu[idx]; j = Ju[idx]
        if avail[idx] == 0:
            # very rare race due to numerical ties; skip
            continue

        # add edge
        A[i, j] = 1
        A[j, i] = 1
        avail[idx] = 0
        m += 1

    return A.astype(np.uint8)

# =============================================================================
# 2) UTILITIES optimized for integer/binary connectomes
# =============================================================================

def _binarize_int(A: np.ndarray) -> np.ndarray:
    """
    Fast binarization for integer/binary adjacency; enforces symmetry and clears diagonal.
    """
    B = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(B, 0)
    return B

def _deg_bin(A_bin: np.ndarray) -> np.ndarray:
    # degree for undirected simple graph (binary)
    return A_bin.sum(axis=1).astype(np.int32)

def _cc_bin(A_bin: np.ndarray) -> np.ndarray:
    """
    Local clustering coefficient (undirected, unweighted) using triangle counts.
    triangles_i = (A^3)_{ii} / 2  for simple undirected graphs
    C_i = 2*triangles_i / (k_i*(k_i-1))  if k_i >= 2 else 0
    """
    A = A_bin.astype(np.float32)
    A2 = A @ A
    A3_diag = np.einsum('ij,ji->i', A2, A)
    tri = A3_diag / 2.0
    k = A.sum(axis=1)
    denom = k * (k - 1.0)
    C = np.zeros_like(k, dtype=np.float32)
    mask = denom > 0
    C[mask] = (2.0 * tri[mask]) / denom[mask]
    return C

def _edge_lengths(A_bin: np.ndarray, dist: np.ndarray) -> np.ndarray:
    # collect lengths on upper triangle where A=1
    iu, ju = np.triu_indices_from(A_bin, k=1)
    mask = (A_bin[iu, ju] != 0)
    return dist[iu[mask], ju[mask]]

def precompute_obs_metrics(A_bin: np.ndarray, dist: np.ndarray,
                           betweenness_backend: str = "approx",
                           bc_k: int = 128,
                           bc_seed: int = 0):
    """
    Compute observed metrics. Degrees & clustering via NumPy; betweenness via:
      - "approx": networkx k-node sample (fast, recommended)
      - "exact": networkx exact (slow)
      - "none": skip (energy will ignore bc)
    """
    deg = _deg_bin(A_bin)
    cc = _cc_bin(A_bin)

    if betweenness_backend == "none":
        bc = None
    else:
        G = nx.from_numpy_array(A_bin, create_using=nx.Graph)
        if betweenness_backend == "approx":
            bc_dict = nx.betweenness_centrality(G, k=min(bc_k, G.number_of_nodes()),
                                                normalized=True, seed=bc_seed)
        else:
            bc_dict = nx.betweenness_centrality(G, normalized=True)
        bc = np.array(list(bc_dict.values()), dtype=np.float64)

    el = _edge_lengths(A_bin, dist)
    return deg, cc, bc, el

# =============================================================================
# 3) ENERGY
# =============================================================================

def gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist,
                    betweenness_backend: str = "approx",
                    bc_k: int = 128,
                    bc_seed: int = 0):
    """
    Compare distributions with KS statistics. Degrees/CC/edge lengths via NumPy.
    Betweenness can be approx/exact/none to trade accuracy vs speed.
    """
    A_sim = (A_sim != 0).astype(np.uint8)

    deg_sim = _deg_bin(A_sim)
    cc_sim  = _cc_bin(A_sim)
    el_sim  = _edge_lengths(A_sim, dist)

    stats = [
        ks_2samp(deg_obs, deg_sim).statistic,
        ks_2samp(cc_obs,  cc_sim).statistic,
        ks_2samp(el_obs,  el_sim).statistic,
    ]

    if bc_obs is not None:
        if betweenness_backend == "approx":
            Gs = nx.from_numpy_array(A_sim, create_using=nx.Graph)
            bc_dict = nx.betweenness_centrality(Gs, k=min(bc_k, Gs.number_of_nodes()),
                                                normalized=True, seed=bc_seed)
            bc_sim = np.array(list(bc_dict.values()), dtype=np.float64)
        elif betweenness_backend == "exact":
            Gs = nx.from_numpy_array(A_sim, create_using=nx.Graph)
            bc_sim = np.array(list(nx.betweenness_centrality(Gs, normalized=True).values()), dtype=np.float64)
        else:
            bc_sim = None

        if bc_sim is not None:
            stats.append(ks_2samp(bc_obs, bc_sim).statistic)

    return max(stats)

# =============================================================================
# 4) SUBJECT-LEVEL WORKER
# =============================================================================

def _subject_job(subj_idx: int,
                 A_obs: np.ndarray,
                 dist: np.ndarray,
                 grid20: List[Tuple[float, float]],
                 grid8: List[Tuple[float, float]],
                 include_subset: bool,
                 subset_size: int,
                 timing_flag: bool):
    # Stage 0: observed metrics (binary/int‑friendly)
    t0 = time.perf_counter()
    A_bin = _binarize_int(A_obs)
    m_edges = int(A_bin.sum() // 2)
    # Use approx betweenness by default for speed
    deg_obs, cc_obs, bc_obs, el_obs = precompute_obs_metrics(
        A_bin, dist, betweenness_backend="approx", bc_k=128, bc_seed=subj_idx
    )
    t_metrics = time.perf_counter() - t0

    # Stage 1: full grid
    seed = np.zeros_like(A_bin, dtype=np.uint8)
    best_energy = np.inf
    best_eta = best_gamma = None
    grid_rows = []
    t_grow = 0.0
    for eta, gamma in grid20:
        tg0 = time.perf_counter()
        A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
        t_grow += time.perf_counter() - tg0
        energy = gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist,
                                 betweenness_backend="approx", bc_k=128, bc_seed=subj_idx)
        grid_rows.append((subj_idx, eta, gamma, energy))
        if energy < best_energy:
            best_energy, best_eta, best_gamma = energy, eta, gamma

    # Stage 2: best model metrics + ESN
    t_esn0 = time.perf_counter()
    A_best = generate_gnm_numba(seed, m_edges, dist, best_gamma, best_eta)

    gms_raw = analyze_connectomes(connectomes=np.expand_dims(A_best, 2),
                                  distance_matrix=dist,
                                  comm_mode='estrada_scaled')[0]
    if isinstance(gms_raw, dict):
        gm_labels = list(gms_raw.keys())
        gm_vals = list(gms_raw.values())
    else:
        gm_labels = [f"gm_{i}" for i in range(len(gms_raw))]
        gm_vals = list(gms_raw)

    # Build ESN weights directly at desired radius (avoid eig)
    W = build_weight_matrix_from_bin_conn(A_best, spectral_radius=0.8, rank=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.seterr(over="ignore", divide="ignore", invalid="ignore")
        mc = evaluate_memory_capacity(W, n_lags=50, train_len=4000, test_len=1000,
                                      n_runs=10, random_state=subj_idx)
        sr = evaluate_sequence_recall(W, pattern_lengths=range(5, 26),
                                      train_trials=800, test_trials=200,
                                      n_runs=5, random_state=subj_idx)
    t_esn = time.perf_counter() - t_esn0

    best_row = (subj_idx, best_eta, best_gamma, best_energy, mc, sr, *gm_vals)
    timing = dict(metrics=t_metrics, grow=t_grow, esn=t_esn)

    # Stage 3: subset grid (optional)
    subset_labels = None
    subset_rows = []
    if include_subset:
        for eta, gamma in grid8:
            A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
            gms = analyze_connectomes(connectomes=np.expand_dims(A_sim, 2),
                                      distance_matrix=dist,
                                      comm_mode='estrada_scaled')[0]
            if isinstance(gms, dict):
                labels = list(gms.keys())
                vals = list(gms.values())
            else:
                labels = [f"gm_{i}" for i in range(len(gms))]
                vals = list(gms)

            W = build_weight_matrix_from_bin_conn(A_sim, spectral_radius=0.8, rank=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                np.seterr(over="ignore", divide="ignore", invalid="ignore")
                mc_s = evaluate_memory_capacity(W, n_lags=50, train_len=4000, test_len=1000,
                                                n_runs=10, random_state=subj_idx)
                sr_s = evaluate_sequence_recall(W, pattern_lengths=range(5, 26),
                                                train_trials=800, test_trials=200,
                                                n_runs=5, random_state=subj_idx)

            subset_rows.append((subj_idx, eta, gamma, mc_s, sr_s, *vals))
            subset_labels = labels

    if timing_flag:
        print(f"Subject {subj_idx}: metrics={t_metrics:.3f}s, grow={t_grow:.3f}s, esn={t_esn:.3f}s")

    return grid_rows, best_row, gm_labels, timing, subset_rows, subset_labels

# =============================================================================
# 5) DRIVER
# =============================================================================

def run_gnm_full(conn: np.ndarray,
                 dist: np.ndarray,
                 n_eta: int = 20,
                 n_gamma: int = 20,
                 subset_size: int = 8,
                 include_subset: bool = True,
                 append_interval: int = 10,
                 timing_flag: bool = False, 
                 timestamp: str = None,
                 save_dir: Path | str = None):
    # ensure dtypes optimal for binary/integer case
    conn = conn.astype(np.uint8, copy=False)
    dist = dist.astype(np.float32, copy=False)

    # prepare grids
    eta_vals = np.linspace(-3.0, 0.0, n_eta, dtype=np.float32)
    gamma_vals = np.linspace(0.1, 0.6, n_gamma, dtype=np.float32)
    grid20 = [(float(e), float(g)) for e in eta_vals for g in gamma_vals]
    eta8 = np.linspace(-3.0, 0.0, subset_size, dtype=np.float32)
    gamma8 = np.linspace(0.1, 0.6, subset_size, dtype=np.float32)
    grid8 = [(float(e), float(g)) for e in eta8 for g in gamma8]

    n_subj = conn.shape[2]
    tasks = [ (i, conn[:,:,i], dist, grid20, grid8, include_subset, subset_size, timing_flag)
              for i in range(n_subj) ]

    # macOS / Jupyter friendly
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    max_workers = min(os.cpu_count() or 1, n_subj)
    save_dir = Path(save_dir) if not isinstance(save_dir, Path) else save_dir

    grid_rows_all, best_rows, timing_list = [], [], []
    subset_rows_all = []
    gm_labels_master = subset_labels_master = None

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_subject_job, *t) for t in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            g_rows, b_row, gm_labels, timing, sub_rows, sub_labels = fut.result()
            grid_rows_all.extend(g_rows)
            best_rows.append(b_row)
            timing_list.append(timing)
            subset_rows_all.extend(sub_rows)

            if gm_labels_master is None:
                gm_labels_master = gm_labels
            if subset_labels_master is None:
                subset_labels_master = sub_labels

            # append intermediate results
            if (append_interval is not None) and (idx % append_interval == 0):
                pd.DataFrame(grid_rows_all, columns=["subject","eta","gamma","energy"])\
                  .to_csv(save_dir / f"gnm_grid_partial_{timestamp}.csv", index=False)
                pd.DataFrame(best_rows, columns=["subject","eta","gamma","energy","memory_capacity","sequence_recall",*gm_labels_master])\
                  .to_csv(save_dir / f"gnm_best_partial_{timestamp}.csv", index=False)
                if include_subset and subset_labels_master is not None:
                    cols = ["subject","eta","gamma","memory_capacity","sequence_recall",*subset_labels_master]
                    pd.DataFrame(subset_rows_all, columns=cols)\
                      .to_csv(save_dir / f"gnm_subset_partial_{timestamp}.csv", index=False)

    # return full dataframes (no final save)
    df_grid = pd.DataFrame(grid_rows_all, columns=["subject","eta","gamma","energy"])
    df_best = pd.DataFrame(best_rows, columns=["subject","eta","gamma","energy","memory_capacity","sequence_recall",*gm_labels_master])
    df_subset = None
    if include_subset and subset_labels_master is not None:
        cols = ["subject","eta","gamma","memory_capacity","sequence_recall",*subset_labels_master]
        df_subset = pd.DataFrame(subset_rows_all, columns=cols)

    return df_grid, df_best, df_subset

# =============================================================================
# 6) MAIN
# =============================================================================
if __name__ == "__main__":
    ROOT = Path("/Users/adrian/Documents/01_projects/14_4D_lab")
    SAVE_DIR = ROOT / "output/02_gnm_estimation"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # smaller grids for quick local tests
    N_ETA = 4
    N_GAMMA = 4
    INCLUDE_SUBSET = True
    SUBSET_SIZE = 4
    APPEND_INTERVAL = 10
    TIMING_FLAG = True

    # load data (integer/binary connectomes, distances)
    conn = np.load(ROOT / "data/preprocessed/01_first_analysises/connectomes_binarized.npy").T
    dist = np.load(ROOT / "data/preprocessed/01_first_analysises/distance_matrix.npy")

    timestamp = time_stamp_for_saving()

    df_grid, df_best, df_subset = run_gnm_full(
        conn, dist,
        n_eta=N_ETA,
        n_gamma=N_GAMMA,
        subset_size=SUBSET_SIZE,
        include_subset=INCLUDE_SUBSET,
        append_interval=APPEND_INTERVAL,
        timing_flag=TIMING_FLAG,
        timestamp=timestamp,
        save_dir=SAVE_DIR
    )

    df_grid.to_csv(SAVE_DIR / f"gnm_grid_results_{timestamp}.csv", index=False)
    df_best.to_csv(SAVE_DIR / f"gnm_best_results_{timestamp}.csv", index=False)
    if df_subset is not None:
        df_subset.to_csv(SAVE_DIR / f"gnm_subset_results_{timestamp}.csv", index=False)
