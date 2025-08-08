# gnm_pipeline_optimized_timed.py 
"""
Fast GNM‑grid + ESN / graph‑measure pipeline **with timing + progress**
-----------------------------------------------------------------------
• 20 × 20 (η, γ) grid shared by all subjects  
• Numba‑JIT GNM growth  
• Subject‑level multiprocessing  
• Live progress + per‑stage timing  
• Stage‑2 evaluation of **best‑energy** model per subject:
    – structural graph measures  
    – ESN memory capacity & sequence recall (robustly scaled)

Outputs two CSVs:
* `gnm_grid_results.csv`  – complete energy surface
* `gnm_best_eval.csv`     – best η/γ + graph + ESN metrics
"""
from __future__ import annotations

import os, time, warnings, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from numba import njit
from scipy.stats import ks_2samp

from src.utils.saving_conventions import time_stamp_for_saving

# ---------------------------------------------------------------------------
# ESN + graph‑measure imports (local project)                                 
# ---------------------------------------------------------------------------
from src.ESNs.generate_weight_matrices_bio_no_rank import build_weight_matrix_from_bin_conn
from src.ESNs.test_memory_capacity import evaluate_memory_capacity
from src.ESNs.test_sequence_recall import evaluate_sequence_recall
from src.structural_analysis.graph_measures import analyze_connectomes

###############################################################################
# 1. NUMBA HELPERS                                                            #
###############################################################################
@njit(cache=True)
def _compute_homophily_bin(A_bin: np.ndarray) -> np.ndarray:
    Af = A_bin.astype(np.float64)
    shared = Af @ Af
    deg = Af.sum(axis=1)
    denom = deg[:, None] + deg[None, :] - shared
    H = np.where(denom > 0.0, shared / denom, 0.0)
    np.fill_diagonal(H, 0.0)
    return H

@njit(cache=True)
def _rand_choice_weighted(p_vec):
    tot = 0.0
    x = np.random.random()
    for k in range(p_vec.size):
        tot += p_vec[k]
        if x < tot:
            return k
    return p_vec.size - 1

@njit(cache=True)
def generate_gnm_numba(A_seed, target_m, dist, gamma, eta, eps: float = 1e-12):
    n = A_seed.shape[0]
    A = A_seed.copy()

    D = np.maximum(dist, eps)
    cost_term = D ** eta
    np.fill_diagonal(cost_term, 0.0)

    m = int(A.sum() // 2)
    while m < target_m:
        H = _compute_homophily_bin(A)
        prob = cost_term * np.maximum(H, eps) ** gamma

        flat_prob = prob.ravel()
        mask_flat = (A == 0).ravel()
        for i in range(n):
            idx_diag = i * n + i
            mask_flat[idx_diag] = False
            base = i * n
            for j in range(i):
                mask_flat[base + j] = False

        cand_idx = np.nonzero(mask_flat)[0]
        if cand_idx.size == 0:
            break
        p_vec = flat_prob[cand_idx]
        p_sum = p_vec.sum()
        if p_sum == 0.0:
            k = cand_idx[np.random.randint(cand_idx.size)]
        else:
            p_vec /= p_sum
            k = cand_idx[_rand_choice_weighted(p_vec)]
        i, j = divmod(k, n)
        A[i, j] = A[j, i] = 1
        m += 1
    return A

###############################################################################
# 2. UTILS                                                                    #
###############################################################################

def _binarize(A: np.ndarray, thr: float = 0.0) -> np.ndarray:
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    return (A > thr).astype(np.int8)


def precompute_obs_metrics(A_bin: np.ndarray, dist: np.ndarray):
    G = nx.from_numpy_array(A_bin)
    deg = np.array([d for _, d in G.degree()])
    cc = np.array(list(nx.clustering(G).values()))
    bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    el = np.array([dist[i, j] for i, j in G.edges()])
    return deg, cc, bc, el

###############################################################################
# 3. ENERGY                                                                   #
###############################################################################

def gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist):
    B_sim = np.triu((A_sim > 0).astype(int), 1)
    B_sim += B_sim.T
    Gs = nx.from_numpy_array(B_sim)

    deg_sim = np.array([d for _, d in Gs.degree()])
    cc_sim = np.array(list(nx.clustering(Gs).values()))
    bc_sim = np.array(list(nx.betweenness_centrality(Gs, normalized=True).values()))
    el_sim = np.array([dist[i, j] for i, j in Gs.edges()])

    return max(
        ks_2samp(deg_obs, deg_sim).statistic,
        ks_2samp(cc_obs, cc_sim).statistic,
        ks_2samp(bc_obs, bc_sim).statistic,
        ks_2samp(el_obs, el_sim).statistic,
    )

###############################################################################
# 4. SUBJECT‑LEVEL WORKER                                                     #
###############################################################################

def _subject_job(subj_idx: int, A_obs: np.ndarray, dist: np.ndarray, grid):
    """Return (grid_rows, best_row, gm_labels, timing_dict) for one subject."""
    # ---- Stage 0: static metrics -----------------------------------------
    t0 = time.perf_counter()
    A_bin = _binarize(A_obs)
    m_edges = int(A_bin.sum() // 2)
    deg_obs, cc_obs, bc_obs, el_obs = precompute_obs_metrics(A_bin, dist)
    t_metrics = time.perf_counter() - t0

    # ---- Stage 1: full grid ----------------------------------------------
    seed = np.zeros_like(A_obs, dtype=np.int8)
    best_energy = np.inf
    best_eta = best_gamma = None
    grid_rows, t_grow = [], 0.0

    for eta, gamma in grid:
        tg0 = time.perf_counter()
        A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
        t_grow += time.perf_counter() - tg0
        energy = gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist)
        grid_rows.append((subj_idx, eta, gamma, energy))
        if energy < best_energy:
            best_energy, best_eta, best_gamma = energy, eta, gamma

    # ---- Stage 2: best model ---------------------------------------------
    t_esn0 = time.perf_counter()
    A_best = generate_gnm_numba(seed, m_edges, dist, best_gamma, best_eta)

    # Graph measures
    gms_raw = analyze_connectomes(connectomes=np.expand_dims(A_best, 2),
                                  distance_matrix=dist, comm_mode='estrada_scaled')[0]
    if isinstance(gms_raw, dict):
        gm_labels = list(gms_raw.keys())
        gm_vals = list(gms_raw.values())
    else:  # assume list‑like
        gm_labels = [f"gm_{i}" for i in range(len(gms_raw))]
        gm_vals = list(gms_raw)

    # Robust Reservoir: rescale to spectral radius 0.8 to avoid overflow
    W = build_weight_matrix_from_bin_conn(A_best, spectral_radius=0.99, rank=False)
    eig_max = np.max(np.abs(np.linalg.eigvals(W)))
    if eig_max > 0:
        W *= 0.8 / eig_max

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # silence numpy matmul overflow
        np.seterr(over="ignore", divide="ignore", invalid="ignore")
        mc = evaluate_memory_capacity(W, n_lags=50, train_len=4000, test_len=1000,
                                      n_runs=10, random_state=subj_idx)
        sr = evaluate_sequence_recall(W, pattern_lengths=range(5, 26),
                                      train_trials=800, test_trials=200,
                                      n_runs=5, random_state=subj_idx)

    t_esn = time.perf_counter() - t_esn0

    best_row = (subj_idx, best_eta, best_gamma, best_energy, mc, sr, *gm_vals)
    timing = dict(metrics=t_metrics, grow=t_grow, esn=t_esn)
    return grid_rows, best_row, gm_labels, timing

# pickle‑able wrapper

def _subject_job_star(args):
    return _subject_job(*args)

###############################################################################
# 5. DRIVER                                                                  #
###############################################################################

def run_gnm_full(connectomes: np.ndarray, dist: np.ndarray, n_eta: int = 20, n_gamma: int = 20):
    eta_vals = np.linspace(-3.0, 0.0, n_eta)
    gamma_vals = np.linspace(0.1, 0.6, n_gamma)
    grid = [(e, g) for e in eta_vals for g in gamma_vals]

    n_subj = connectomes.shape[2]
    tasks = [(i, connectomes[:, :, i], dist, grid) for i in range(n_subj)]

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    grid_rows_all, best_rows, timing_list, gm_labels_master = [], [], [], None
    t_total0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(_subject_job_star, t) for t in tasks]
        for done_idx, fut in enumerate(as_completed(futures), 1):
            grid_r, best_r, gm_labels, timing = fut.result()
            grid_rows_all.extend(grid_r)
            best_rows.append(best_r)
            timing_list.append(timing)
            if gm_labels_master is None:
                gm_labels_master = gm_labels  # first subject defines order

            if done_idx % 5 == 0 or done_idx == n_subj:
                print(f"[{time.strftime('%H:%M:%S')}] processed {done_idx}/{n_subj} subjects")

    total_elapsed = time.perf_counter() - t_total0

    # Timing summary
    timing_df = pd.DataFrame(timing_list)
    print("\nTiming summary (mean per subject)")
    print(timing_df.mean().round(3).to_string())
    print(f"Total wall time : {total_elapsed:7.1f} s\n")

    # DataFrames
    df_grid = pd.DataFrame(grid_rows_all, columns=["subject", "eta", "gamma", "energy"])

    df_best = pd.DataFrame(best_rows, columns=[
        "subject", "eta", "gamma", "energy", "memory_capacity", "sequence_recall", *gm_labels_master])

    return df_grid, df_best

###############################################################################
# 6. MAIN (adjust paths)                                                      #
###############################################################################
if __name__ == "__main__":
    ROOT = "/Users/adrian/Documents/01_projects/14_4D_lab"
    conn = np.load(f"{ROOT}/data/preprocessed/01_first_analysises/connectomes_binarized.npy").T
    dist = np.load(f"{ROOT}/data/preprocessed/01_first_analysises/distance_matrix.npy")

    df_grid, df_best = run_gnm_full(conn, dist)

    out_dir = f"{ROOT}/output/02_gnm_estimation"
    os.makedirs(out_dir, exist_ok=True)
    df_grid.to_csv(f"{out_dir}/gnm_grid_results_{time_stamp_for_saving()}.csv", index=False)
    df_best.to_csv(f"{out_dir}/gnm_best_eval_{time_stamp_for_saving()}.csv", index=False)

    print(df_best.head())



# 20*20 hat 4 min gebraucht. 
# 100x100 hat: (und nur mit 4 kernen!)
    # Timing summary (mean per subject)
    # metrics     0.012
    # grow       17.024
    # esn        24.538
    # Total wall time :   226.0 s


# # # gnm_pipeline_optimized_timed.py
# # """
# # Fast GNM‑grid pipeline **with timing + progress**
# # ------------------------------------------------
# # * Fixed 20 × 20 (η, γ) grid shared by all subjects
# # * Numba‑JIT growth (`generate_gnm_numba`)
# # * Subject‑level multiprocessing (`ProcessPoolExecutor`)
# # * Pre‑computed observed metrics per subject
# # * **Wall‑clock profiling** of each major stage
# # * **Live progress message** every 5 subjects (or on final)

# # Apple‑silicon compatible (uses "spawn").  Requires NumPy ≥ 2, SciPy, Numba 0.59.
# # """
# # from __future__ import annotations

# # import os
# # import time
# # import multiprocessing as mp
# # from concurrent.futures import ProcessPoolExecutor, as_completed
# # from typing import List, Tuple

# # import numpy as np
# # import pandas as pd
# # import networkx as nx  # swap for igraph for >10× speed‑up
# # from numba import njit
# # from scipy.stats import ks_2samp

# # ###############################################################################
# # # 1.  NUMBA HELPERS                                                           #
# # ###############################################################################

# # @njit(cache=True)
# # def _compute_homophily_bin(A_bin: np.ndarray) -> np.ndarray:
# #     Af = A_bin.astype(np.float64)      # BLAS needs float dtype
# #     shared = Af @ Af                   # matrix multiply (deg^2 common neigh.)
# #     deg = Af.sum(axis=1)
# #     denom = deg[:, None] + deg[None, :] - shared
# #     H = np.where(denom > 0.0, shared / denom, 0.0)
# #     np.fill_diagonal(H, 0.0)
# #     return H

# # @njit(cache=True)
# # def _rand_choice_weighted(p_vec):
# #     tot = 0.0
# #     x = np.random.random()
# #     for k in range(p_vec.size):
# #         tot += p_vec[k]
# #         if x < tot:
# #             return k
# #     return p_vec.size - 1

# # @njit(cache=True)
# # def generate_gnm_numba(A_seed, target_m, dist, gamma, eta, eps=1e-12):
# #     n = A_seed.shape[0]
# #     A = A_seed.copy()

# #     D = np.maximum(dist, eps)
# #     cost_term = D ** eta
# #     np.fill_diagonal(cost_term, 0.0)

# #     m = int(A.sum() // 2)
# #     while m < target_m:
# #         H = _compute_homophily_bin(A)
# #         value_term = np.maximum(H, eps) ** gamma
# #         prob = cost_term * value_term

# #         flat_prob = prob.ravel()
# #         mask_flat = (A == 0).ravel()
# #         # zero lower‑triangle + diag
# #         for i in range(n):
# #             idx_diag = i * n + i
# #             mask_flat[idx_diag] = False
# #             base = i * n
# #             for j in range(i):
# #                 mask_flat[base + j] = False

# #         cand_idx = np.nonzero(mask_flat)[0]
# #         if cand_idx.size == 0:
# #             break  # fully connected
# #         p_vec = flat_prob[cand_idx]
# #         p_sum = p_vec.sum()
# #         if p_sum == 0.0:
# #             k = cand_idx[np.random.randint(cand_idx.size)]
# #         else:
# #             p_vec /= p_sum
# #             k_local = _rand_choice_weighted(p_vec)
# #             k = cand_idx[k_local]
# #         i = k // n
# #         j = k % n
# #         A[i, j] = A[j, i] = 1
# #         m += 1
# #     return A

# # ###############################################################################
# # # 2.  OBSERVED‑GRAPH METRICS                                                  #
# # ###############################################################################

# # def _binarize(A: np.ndarray, thr: float = 0.0) -> np.ndarray:
# #     A = (A + A.T) / 2.0
# #     np.fill_diagonal(A, 0.0)
# #     return (A > thr).astype(np.int8)


# # def precompute_obs_metrics(A_bin: np.ndarray, dist: np.ndarray):
# #     G = nx.from_numpy_array(A_bin)
# #     deg = np.array([d for _, d in G.degree()])
# #     cc = np.array(list(nx.clustering(G).values()))
# #     bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
# #     el = np.array([dist[i, j] for i, j in G.edges()])
# #     return deg, cc, bc, el

# # ###############################################################################
# # # 3.  ENERGY                                                                  #
# # ###############################################################################

# # def gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist):
# #     B_sim = np.triu((A_sim > 0).astype(int), 1)
# #     B_sim += B_sim.T
# #     Gs = nx.from_numpy_array(B_sim)

# #     deg_sim = np.array([d for _, d in Gs.degree()])
# #     cc_sim = np.array(list(nx.clustering(Gs).values()))
# #     bc_sim = np.array(list(nx.betweenness_centrality(Gs, normalized=True).values()))
# #     el_sim = np.array([dist[i, j] for i, j in Gs.edges()])

# #     return max(
# #         ks_2samp(deg_obs, deg_sim).statistic,
# #         ks_2samp(cc_obs, cc_sim).statistic,
# #         ks_2samp(bc_obs, bc_sim).statistic,
# #         ks_2samp(el_obs, el_sim).statistic,
# #     )

# # ###############################################################################
# # # 4.  WORKER                                                                  #
# # ###############################################################################

# # def _subject_job(subj_idx: int, A_obs: np.ndarray, dist: np.ndarray, grid):
# #     t0 = time.perf_counter()
# #     A_bin = _binarize(A_obs)
# #     m_edges = int(A_bin.sum() // 2)
# #     deg_obs, cc_obs, bc_obs, el_obs = precompute_obs_metrics(A_bin, dist)
# #     t_metrics = time.perf_counter() - t0

# #     seed = np.zeros_like(A_obs, dtype=np.int8)
# #     rows = []
# #     t_grow = 0.0
# #     for eta, gamma in grid:
# #         tg0 = time.perf_counter()
# #         A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
# #         t_grow += time.perf_counter() - tg0
# #         energy = gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist)
# #         rows.append((subj_idx, eta, gamma, energy))

# #     return rows, t_metrics, t_grow

# # # helper must be pickle‑able
# # def _subject_job_star(args):
# #     return _subject_job(*args)

# # ###############################################################################
# # # 5.  DRIVER WITH TIMERS + PROGRESS                                           #
# # ###############################################################################

# # def run_gnm_grid(connectomes: np.ndarray, dist: np.ndarray, n_eta: int = 20, n_gamma: int = 20):
# #     eta_vals = np.linspace(-3.0, 0.0, n_eta)
# #     gamma_vals = np.linspace(0.1, 0.6, n_gamma)
# #     grid = [(e, g) for e in eta_vals for g in gamma_vals]

# #     n_subj = connectomes.shape[2]
# #     tasks = [(i, connectomes[:, :, i], dist, grid) for i in range(n_subj)]

# #     if mp.get_start_method(allow_none=True) != "spawn":
# #         mp.set_start_method("spawn", force=True)

# #     total_start = time.perf_counter()
# #     subject_timings = []  # list of (metrics_sec, grow_sec)
# #     rows = []

# #     with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
# #         futures = [pool.submit(_subject_job_star, t) for t in tasks]
# #         for done_idx, fut in enumerate(as_completed(futures), 1):
# #             subj_rows, t_metrics, t_grow = fut.result()
# #             rows.extend(subj_rows)
# #             subject_timings.append((t_metrics, t_grow))

# #             if done_idx % 5 == 0 or done_idx == n_subj:
# #                 print(f"[{time.strftime('%H:%M:%S')}] processed {done_idx}/{n_subj} subjects")

# #     total_elapsed = time.perf_counter() - total_start

# #     # --- summary timings
# #     metrics_avg = np.mean([m for m, _ in subject_timings])
# #     grow_avg = np.mean([g for _, g in subject_timings])
# #     print("\nTiming summary (per subject, mean over", n_subj, ")")
# #     print(f"  • observed‑metric precomp : {metrics_avg:6.3f} s")
# #     print(f"  • GNM growth (20×20 grid) : {grow_avg:6.3f} s")
# #     print(f"Total wall time              : {total_elapsed:6.3f} s\n")

# #     return pd.DataFrame(rows, columns=["subject", "eta", "gamma", "energy"])








# # gnm_pipeline_optimized_timed.py
# """
# Fast GNM‑grid + ESN/graph‑measure pipeline **with timing + progress**
# --------------------------------------------------------------------
# • Fixed 20 × 20 (η, γ) grid shared by all subjects  
# • Numba‑JIT growth (`generate_gnm_numba`)  
# • Subject‑level multiprocessing (`ProcessPoolExecutor`)  
# • Pre‑computed observed metrics per subject  
# • **Wall‑clock profiling** of each major stage  
# • **Live progress** every 5 subjects (or on final)  

# Adds stage‑2 evaluation **on the single best‑energy network per subject**:
# 1. Structural graph measures (`analyze_connectomes`)
# 2. Memory capacity  (`evaluate_memory_capacity`)
# 3. Sequence recall (`evaluate_sequence_recall`)

# Outputs two CSVs:
# * `gnm_grid_results.csv`  – raw energy grid (subject × 400 rows)
# * `gnm_best_eval.csv`     – best η/γ with graph + ESN metrics per subject

# Apple‑silicon compatible (uses "spawn"). Requires NumPy ≥ 2, SciPy, Numba 0.59.
# """
# from __future__ import annotations

# import os
# import time
# import multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import List, Tuple

# import numpy as np
# import pandas as pd
# import networkx as nx  # swap for igraph for >10× speed‑up
# from numba import njit
# from scipy.stats import ks_2samp

# # ----- ESN + graph‑measure imports ------------------------------------------
# from src.ESNs.generate_weight_matrices_bio_no_rank import build_weight_matrix_from_bin_conn
# from src.ESNs.test_memory_capacity import evaluate_memory_capacity
# from src.ESNs.test_sequence_recall import evaluate_sequence_recall
# from src.structural_analysis.graph_measures import analyze_connectomes

# ###############################################################################
# # 1.  NUMBA HELPERS                                                           #
# ###############################################################################
# @njit(cache=True)
# def _compute_homophily_bin(A_bin: np.ndarray) -> np.ndarray:
#     Af = A_bin.astype(np.float64)
#     shared = Af @ Af
#     deg = Af.sum(axis=1)
#     denom = deg[:, None] + deg[None, :] - shared
#     H = np.where(denom > 0.0, shared / denom, 0.0)
#     np.fill_diagonal(H, 0.0)
#     return H

# @njit(cache=True)
# def _rand_choice_weighted(p_vec):
#     tot = 0.0
#     x = np.random.random()
#     for k in range(p_vec.size):
#         tot += p_vec[k]
#         if x < tot:
#             return k
#     return p_vec.size - 1

# @njit(cache=True)
# def generate_gnm_numba(A_seed, target_m, dist, gamma, eta, eps=1e-12):
#     n = A_seed.shape[0]
#     A = A_seed.copy()

#     D = np.maximum(dist, eps)
#     cost_term = D ** eta
#     np.fill_diagonal(cost_term, 0.0)

#     m = int(A.sum() // 2)
#     while m < target_m:
#         H = _compute_homophily_bin(A)
#         value_term = np.maximum(H, eps) ** gamma
#         prob = cost_term * value_term

#         flat_prob = prob.ravel()
#         mask_flat = (A == 0).ravel()
#         for i in range(n):
#             idx_diag = i * n + i
#             mask_flat[idx_diag] = False
#             base = i * n
#             for j in range(i):
#                 mask_flat[base + j] = False

#         cand_idx = np.nonzero(mask_flat)[0]
#         if cand_idx.size == 0:
#             break
#         p_vec = flat_prob[cand_idx]
#         p_sum = p_vec.sum()
#         if p_sum == 0.0:
#             k = cand_idx[np.random.randint(cand_idx.size)]
#         else:
#             p_vec /= p_sum
#             k_local = _rand_choice_weighted(p_vec)
#             k = cand_idx[k_local]
#         i = k // n
#         j = k % n
#         A[i, j] = A[j, i] = 1
#         m += 1
#     return A

# ###############################################################################
# # 2.  UTILITIES                                                               #
# ###############################################################################

# def _binarize(A: np.ndarray, thr: float = 0.0) -> np.ndarray:
#     A = (A + A.T) / 2.0
#     np.fill_diagonal(A, 0.0)
#     return (A > thr).astype(np.int8)


# def precompute_obs_metrics(A_bin: np.ndarray, dist: np.ndarray):
#     G = nx.from_numpy_array(A_bin)
#     deg = np.array([d for _, d in G.degree()])
#     cc = np.array(list(nx.clustering(G).values()))
#     bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
#     el = np.array([dist[i, j] for i, j in G.edges()])
#     return deg, cc, bc, el

# ###############################################################################
# # 3.  ENERGY                                                                  #
# ###############################################################################

# def gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist):
#     B_sim = np.triu((A_sim > 0).astype(int), 1)
#     B_sim += B_sim.T
#     Gs = nx.from_numpy_array(B_sim)

#     deg_sim = np.array([d for _, d in Gs.degree()])
#     cc_sim = np.array(list(nx.clustering(Gs).values()))
#     bc_sim = np.array(list(nx.betweenness_centrality(Gs, normalized=True).values()))
#     el_sim = np.array([dist[i, j] for i, j in Gs.edges()])

#     return max(
#         ks_2samp(deg_obs, deg_sim).statistic,
#         ks_2samp(cc_obs, cc_sim).statistic,
#         ks_2samp(bc_obs, bc_sim).statistic,
#         ks_2samp(el_obs, el_sim).statistic,
#     )

# ###############################################################################
# # 4.  SUBJECT‑LEVEL WORKER                                                    #
# ###############################################################################

# def _subject_job(subj_idx: int, A_obs: np.ndarray, dist: np.ndarray, grid):
#     """Returns (grid_rows, best_eval_row, timing_dict)"""
#     t0 = time.perf_counter()
#     A_bin = _binarize(A_obs)
#     m_edges = int(A_bin.sum() // 2)
#     deg_obs, cc_obs, bc_obs, el_obs = precompute_obs_metrics(A_bin, dist)
#     t_metrics = time.perf_counter() - t0

#     seed = np.zeros_like(A_obs, dtype=np.int8)
#     grid_rows = []
#     best_energy = np.inf
#     best_eta = best_gamma = None
#     t_grow = 0.0

#     for eta, gamma in grid:
#         tg0 = time.perf_counter()
#         A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
#         t_grow += time.perf_counter() - tg0
#         energy = gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist)
#         grid_rows.append((subj_idx, eta, gamma, energy))
#         if energy < best_energy:
#             best_energy = energy
#             best_eta, best_gamma = eta, gamma

#     # ---- Stage‑2: evaluate best GNM --------------------------------------
#     t_esn0 = time.perf_counter()
#     A_best = generate_gnm_numba(seed, m_edges, dist, best_gamma, best_eta)

#     # graph measures (returns list-of‑lists – we take first element)
#     gms = analyze_connectomes(connectomes=np.expand_dims(A_best, axis=2),
#                               distance_matrix=dist,
#                               comm_mode='estrada_scaled')[0]

#     # build reservoir and ESN metrics
#     W = build_weight_matrix_from_bin_conn(A_best, spectral_radius=0.99, rank=False)
#     mc = evaluate_memory_capacity(W, n_lags=50, train_len=4000, test_len=1000,
#                                   n_runs=10, random_state=subj_idx)
#     sr = evaluate_sequence_recall(W, pattern_lengths=range(5, 26),
#                                   train_trials=800, test_trials=200,
#                                   n_runs=5, random_state=subj_idx)
#     t_esn = time.perf_counter() - t_esn0

#     best_eval_row = (subj_idx, best_eta, best_gamma, best_energy, mc, sr, *gms)

#     timing = dict(metrics=t_metrics, grow=t_grow, esn=t_esn)
#     return grid_rows, best_eval_row, timing

# # pickle‑able wrapper

# def _subject_job_star(args):
#     return _subject_job(*args)

# ###############################################################################
# # 5.  DRIVER                                                                  #
# ###############################################################################

# def run_gnm_full(connectomes: np.ndarray, dist: np.ndarray,
#                  n_eta: int = 20, n_gamma: int = 20):
#     eta_vals = np.linspace(-3.0, 0.0, n_eta)
#     gamma_vals = np.linspace(0.1, 0.6, n_gamma)
#     grid = [(e, g) for e in eta_vals for g in gamma_vals]

#     n_subj = connectomes.shape[2]
#     tasks = [(i, connectomes[:, :, i], dist, grid) for i in range(n_subj)]

#     if mp.get_start_method(allow_none=True) != "spawn":
#         mp.set_start_method("spawn", force=True)

#     grid_rows_all, best_rows, timing_list = [], [], []
#     total_start = time.perf_counter()

#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
#         futures = [pool.submit(_subject_job_star, t) for t in tasks]
#         for done_idx, fut in enumerate(as_completed(futures), 1):
#             grid_r, best_r, timing = fut.result()
#             grid_rows_all.extend(grid_r)
#             best_rows.append(best_r)
#             timing_list.append(timing)

#             if done_idx % 5 == 0 or done_idx == n_subj:
#                 print(f"[{time.strftime('%H:%M:%S')}] processed {done_idx}/{n_subj} subjects")

#     total_elapsed = time.perf_counter() - total_start

#     # ----- timing summary
#     timing_df = pd.DataFrame(timing_list)
#     print("\nTiming summary (mean per subject)")
#     print(timing_df.mean().round(3).to_string())
#     print(f"Total wall time : {total_elapsed:7.1f} s\n")

#     # ----- assemble DataFrames
#     df_grid = pd.DataFrame(grid_rows_all, columns=["subject", "eta", "gamma", "energy"])

#     gm_labels = ["avg_communicability", "global_efficiency", "modularity", "avg_clustering",
#                  "avg_degree", "transitivity", "avg_edge_distance", "char_path_length",
#                  "richclub_n_edges", "richclub_avg_length"]
#     df_best = pd.DataFrame(best_rows, columns=[
#         "subject", "eta", "gamma", "energy", "memory_capacity", "sequence_recall", *gm_labels])

#     return df_grid, df_best

# ###############################################################################
# # 6.  USAGE                                                                   #
# ###############################################################################
# if __name__ == "__main__":
#     ROOT = "/Users/adrian/Documents/01_projects/14_4D_lab"
#     conn = np.load(f"{ROOT}/data/preprocessed/01_first_analysises/connectomes_binarized.npy").T
#     dist = np.load(f"{ROOT}/data/preprocessed/01_first_analysises/distance_matrix.npy")

#     df_grid, df_best = run_gnm_full(conn, dist)

#     out_dir = f"{ROOT}/output/02_gnm_estimation"
#     os.makedirs(out_dir, exist_ok=True)
#     df_grid.to_csv(f"{out_dir}/gnm_grid_results.csv", index=False)
#     df_best.to_csv(f"{out_dir}/gnm_best_eval.csv", index=False)

#     print(df_best.head())


#     Ok thank you! I want to also compute for a subset of the GNMs (preferably with gammas and etas spread in a orderly gridwise fashion across the entire space. the grid should contain 8*8 GNMs) the graph metrics and the functional metrics. I want to do this such that I can compare them later on. Can you please also add them into the 



# # from src.utils.saving_conventions import time_stamp_for_saving
