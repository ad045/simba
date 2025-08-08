# # gnm_pipeline_optimized_timed.py
# """
# GNM grid + ESN / graph‑measure pipeline **with timing, progress, and 8 × 8 evaluation grid**
# ========================================================================================
# **What it does**
# * Fits a 20 × 20 (η, γ) grid of generative network models (GNMs) for every subject and records the energy landscape.
# * Selects the best‑energy model **and** a systematic 8 × 8 sub‑grid spread over the parameter range.
# * For those 65 models it computes
#   * 10 structural graph measures (`analyze_connectomes`)
#   * ESN memory capacity & sequence recall.
# * Writes three CSVs:
#   1. `gnm_grid_results.csv`   – 20 × 20 energy surface per subject.
#   2. `gnm_best_eval.csv`      – best η/γ with graph + ESN metrics.
#   3. `gnm_subset_eval.csv`    – 8 × 8 sub‑grid models with graph + ESN metrics.

# Runs on Apple‑silicon (``spawn``).  Needs NumPy ≥ 2, SciPy, Numba 0.59.
# """
# from __future__ import annotations

# import os, time, warnings, multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import List, Tuple

# import glob

# import numpy as np
# import pandas as pd
# import networkx as nx
# from numba import njit
# from scipy.stats import ks_2samp

# from src.utils.saving_conventions import time_stamp_for_saving

# # ---- local project imports --------------------------------------------------
# from src.ESNs.generate_weight_matrices_bio_no_rank import build_weight_matrix_from_bin_conn
# from src.ESNs.test_memory_capacity import evaluate_memory_capacity
# from src.ESNs.test_sequence_recall import evaluate_sequence_recall
# from src.structural_analysis.graph_measures import analyze_connectomes

# # ---------- NEW: efficient columnar storage ---------------------------------
# import pyarrow as pa
# import pyarrow.parquet as pq

# from src.utils.saving_conventions import time_stamp_for_saving

# ###############################################################################
# # 1. NUMBA‑accelerated GNM growth                                             #
# ###############################################################################
# @njit(cache=True)
# def _compute_homophily_bin(A_bin: np.ndarray) -> np.ndarray:
#     Af = A_bin.astype(np.float64)
#     shared = Af @ Af
#     deg = Af.sum(axis=1)
#     denom = deg[:, None] + deg[None, :] - shared
#     H = np.where(denom > 0, shared / denom, 0)
#     np.fill_diagonal(H, 0)
#     return H

# @njit(cache=True)
# def _rand_choice_weighted(p_vec):
#     x = np.random.random()
#     tot = 0.0
#     for i in range(p_vec.size):
#         tot += p_vec[i]
#         if x < tot:
#             return i
#     return p_vec.size - 1

# @njit(cache=True)
# def generate_gnm_numba(A_seed, target_m, dist, gamma, eta, eps=1e-12):
#     n = A_seed.shape[0]
#     A = A_seed.copy()
#     D = np.maximum(dist, eps)
#     cost = D ** eta
#     np.fill_diagonal(cost, 0)
#     m = int(A.sum() // 2)
#     while m < target_m:
#         H = _compute_homophily_bin(A)
#         prob = cost * np.maximum(H, eps) ** gamma
#         flat = prob.ravel()
#         mask = (A == 0).ravel()
#         # remove lower‑triangle & diagonal from candidate mask
#         for i in range(n):
#             diag = i * n + i
#             mask[diag] = False
#             base = i * n
#             for j in range(i):
#                 mask[base + j] = False
#         cand = np.nonzero(mask)[0]
#         if cand.size == 0:
#             break
#         p_vec = flat[cand]
#         if p_vec.sum() == 0:
#             k = cand[np.random.randint(cand.size)]
#         else:
#             p_vec /= p_vec.sum()
#             k = cand[_rand_choice_weighted(p_vec)]
#         i, j = divmod(k, n)
#         A[i, j] = A[j, i] = 1
#         m += 1
#     return A

# ###############################################################################
# # 2. Utilities                                                                #
# ###############################################################################

# def _binarize(A, thr=0):
#     A = (A + A.T) / 2
#     np.fill_diagonal(A, 0)
#     return (A > thr).astype(np.int8)


# def _obs_metrics(A_bin, dist):
#     G = nx.from_numpy_array(A_bin)
#     deg = np.array([d for _, d in G.degree()])
#     cc = np.array(list(nx.clustering(G).values()))
#     bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
#     el = np.array([dist[i, j] for i, j in G.edges()])
#     return deg, cc, bc, el


# def _energy(deg_o, cc_o, bc_o, el_o, A_sim, dist):
#     B = np.triu((A_sim > 0).astype(int), 1)
#     B += B.T
#     Gs = nx.from_numpy_array(B)
#     deg_s = np.array([d for _, d in Gs.degree()])
#     cc_s = np.array(list(nx.clustering(Gs).values()))
#     bc_s = np.array(list(nx.betweenness_centrality(Gs, normalized=True).values()))
#     el_s = np.array([dist[i, j] for i, j in Gs.edges()])
#     return max(ks_2samp(deg_o, deg_s).statistic,
#                ks_2samp(cc_o, cc_s).statistic,
#                ks_2samp(bc_o, bc_s).statistic,
#                ks_2samp(el_o, el_s).statistic)

# ###############################################################################
# # 3. Per‑ESN metrics helper                                                   #
# ###############################################################################

# def _write_parquet(row_dict: dict, path: str):
#     """Write a single‑row dict to parquet (overwrites)."""
#     table = pa.Table.from_pydict(row_dict)
#     pq.write_table(table, path, compression="zstd")


# def _read_parquet(path: str) -> dict:
#     table = pq.read_table(path)
#     return {name: table.column(name)[0].as_py() for name in table.column_names}


# def _metrics_single(A_seed, m_edges, dist, eta, gamma, subj_idx, out_dir):
#     """Compute (or read cached) graph + ESN metrics for one (η, γ)."""
#     # ------------------------------------------------------------------
#     # Unique file path                                                
#     # ------------------------------------------------------------------
#     subj_dir = os.path.join(out_dir, f"subj_{subj_idx:03d}")
#     os.makedirs(subj_dir, exist_ok=True)
#     file_stub = f"esn_eta{eta:.3f}_gamma{gamma:.3f}.parquet"
#     file_path = os.path.join(subj_dir, file_stub)

#     # ------------------------------------------------------------------
#     # Try loading cached result first                                  
#     # ------------------------------------------------------------------
#     if os.path.exists(file_path):
#         cached = _read_parquet(file_path)
#         # graph‑metric labels = arbitrary remaining keys in stable order
#         gm_lbls = [k for k in cached.keys() if k not in {"subject", "eta", "gamma", "mc", "sr"}]
#         # Maintain original return order (eta, gamma, mc, sr, *g_vals)
#         met_tuple = (cached["eta"], cached["gamma"], cached["mc"], cached["sr"], *[cached[l] for l in gm_lbls])
#         return gm_lbls, met_tuple

#     # ------------------------------------------------------------------
#     # Simulate fresh ESN & compute metrics                              
#     # ------------------------------------------------------------------
#     A_sim = generate_gnm_numba(A_seed, m_edges, dist, gamma, eta)
#     g_raw = analyze_connectomes(np.expand_dims(A_sim, 2), dist, 'estrada_scaled')[0]
#     if isinstance(g_raw, dict):
#         g_vals, g_lbls = list(g_raw.values()), list(g_raw.keys())
#     else:
#         g_vals, g_lbls = list(g_raw), [f"gm_{i}" for i in range(len(g_raw))]

#     W = build_weight_matrix_from_bin_conn(A_sim, spectral_radius=0.99, rank=False)
#     eig_max = np.max(np.abs(np.linalg.eigvals(W)))
#     if eig_max > 0:
#         W *= 0.8 / eig_max
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore', RuntimeWarning)
#         np.seterr(over='ignore', divide='ignore', invalid='ignore')
#         mc = evaluate_memory_capacity(W, n_lags=50, 
#                                       train_len=4000, test_len=1000,
#                                       n_runs=5, random_state=subj_idx)
#         sr = evaluate_sequence_recall(W, pattern_lengths=range(5, 26), 
#                                       train_trials=800, test_trials=200, 
#                                       n_runs=3, random_state=subj_idx)


#     row_dict: dict[str, float | int] = {
#         'subject': subj_idx,
#         'eta': eta,
#         'gamma': gamma,
#         'mc': mc,
#         'sr': sr,
#         **{lbl: val for lbl, val in zip(g_lbls, g_vals)},
#     }
#     _write_parquet(row_dict, file_path)
#     met_tuple = (eta, gamma, mc, sr, *g_vals)
#     return g_lbls, met_tuple

# ###############################################################################
# # 4. Subject‑level worker                                                     #
# ###############################################################################

# def _subject_job(idx: int, A_obs, dist, grid20, grid8, out_dir):
#     A_bin = _binarize(A_obs)
#     m_edges = int(A_bin.sum() // 2)
#     deg_o, cc_o, bc_o, el_o = _obs_metrics(A_bin, dist)
#     seed = np.zeros_like(A_obs, dtype=np.int8)

#     # ----- full 20×20 energy grid ------------------------------------
#     grid_rows, best_E = [], np.inf
#     best_eta = best_gamma = None
#     for eta, gamma in grid20:
#         A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
#         E = _energy(deg_o, cc_o, bc_o, el_o, A_sim, dist)
#         grid_rows.append((idx, eta, gamma, E))
#         if E < best_E:
#             best_E, best_eta, best_gamma = E, eta, gamma

#     # ----- metrics for best model ------------------------------------
#     g_lbls, best_metrics = _metrics_single(seed, m_edges, dist,
#                                            best_eta, best_gamma, idx, out_dir)
#     best_row = (idx, best_eta, best_gamma, best_E, *best_metrics)

#     # ----- orderly 8×8 sub‑grid metrics ------------------------------
#     sub_rows, g_lbls_sub = [], None
#     for eta, gamma in grid8:
#         g_lbls_sub, met = _metrics_single(seed, m_edges, dist,
#                                           eta, gamma, idx, out_dir)
#         sub_rows.append((idx, *met))
#     return grid_rows, best_row, g_lbls, sub_rows, g_lbls_sub


# def _star(args):
#     return _subject_job(*args)

# ###############################################################################
# # 5. Driver                                                                   #
# ###############################################################################

# def _aggregate_metrics(esn_dir: str) -> pd.DataFrame:
#     """Concatenate all per‑ESN parquet snippets into one DataFrame."""
#     paths = glob.glob(os.path.join(esn_dir, 'subj_*', 'esn_*.parquet'))
#     if not paths:
#         raise RuntimeError("No per‑ESN parquet files found. Did the simulation run?")
#     tables = [pq.read_table(p) for p in paths]
#     return pa.concat_tables(tables).to_pandas()


# def run_gnm_pipeline(conn, dist, out_root: str, n_eta=20, n_gamma=20):
#     out_root = os.path.abspath(out_root)
#     esn_dir = os.path.join(out_root, 'esn_metrics')
#     os.makedirs(esn_dir, exist_ok=True)

#     grid20 = [(e, g) for e in np.linspace(-3, 0, n_eta)
#                       for g in np.linspace(0.1, 0.6, n_gamma)]
#     grid8  = [(e, g) for e in np.linspace(-3, 0, 8)
#                       for g in np.linspace(0.1, 0.6, 8)]

#     n_subj = conn.shape[2]
#     tasks = [(i, conn[:, :, i], dist, grid20, grid8, esn_dir) for i in range(n_subj)]

#     if mp.get_start_method(allow_none=True) != 'spawn':
#         mp.set_start_method('spawn', force=True)

#     grid_rows_all, best_rows = [], []
#     gm_labels_master = gm_labels_subset = None
#     t0 = time.perf_counter()

#     with ProcessPoolExecutor(max_workers=4) as pool:
#         futs = [pool.submit(_star, t) for t in tasks]
#         for done, fut in enumerate(as_completed(futs), 1):
#             print(f"[{time.strftime('%H:%M:%S')}] finished {done}/{n_subj} subjects")
#             g_rows, b_row, g_lbls, _, _ = fut.result()
#             grid_rows_all.extend(g_rows)
#             best_rows.append(b_row)
#             if gm_labels_master is None:
#                 gm_labels_master = g_lbls
#             # if done % 5 == 0 or done == n_subj:
#             #     print(f"[{time.strftime('%H:%M:%S')}] finished {done}/{n_subj} subjects")

#     # ------------------- aggregation ---------------------------------
#     print("Aggregating per‑ESN parquet files … This may take a moment.")
#     df_subset = _aggregate_metrics(esn_dir)

#     print(f"Total wall time: {time.perf_counter()-t0:6.1f}s")

#     df_grid = pd.DataFrame(grid_rows_all, columns=['subject', 'eta', 'gamma', 'energy'])
#     df_best = pd.DataFrame(best_rows, columns=['subject', 'eta', 'gamma', 'energy',
#                                                'eta_m', 'gamma_m', 'mc', 'sr', *gm_labels_master])
#     # df_subset is already complete (subject, eta, gamma, mc, sr, *gm_labels_subset)
#     return df_grid, df_best, df_subset

# ###############################################################################
# # 6. Main                                                                     #
# ###############################################################################
# if __name__ == '__main__':
#     ROOT = '/Users/adrian/Documents/01_projects/14_4D_lab'
#     conn = np.load(f'{ROOT}/data/preprocessed/01_first_analysises/connectomes_binarized.npy').T
#     dist = np.load(f'{ROOT}/data/preprocessed/01_first_analysises/distance_matrix.npy')

#     out_dir = f'{ROOT}/output/02_gnm_estimation_2'
#     os.makedirs(out_dir, exist_ok=True)

#     df_grid, df_best, df_subset = run_gnm_pipeline(conn, dist, out_dir,
#                                                 n_eta=10, n_gamma=10)

#     df_grid.to_parquet(os.path.join(out_dir, f'gnm_grid_results_{time_stamp_for_saving()}.parquet'),
#                        compression='zstd', index=False)
#     df_best.to_parquet(os.path.join(out_dir, f'gnm_best_eval_{time_stamp_for_saving()}.parquet'),
#                        compression='zstd', index=False)
#     df_subset.to_parquet(os.path.join(out_dir, f'gnm_subset_eval_{time_stamp_for_saving()}.parquet'),
#                          compression='zstd', index=False)

#     print(df_best.head())
