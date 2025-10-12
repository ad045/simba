# # Basiert noch auf "build_weight_matrix_from_bin_conn" -> also random bio no rank matrix (ganz alte version) 
# # Anscheinend noch nicht getestet

# import os, time, warnings, multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import List, Tuple

# from pathlib import Path

# import numpy as np
# import pandas as pd
# import networkx as nx
# from numba import njit
# from scipy.stats import ks_2samp

# import sys
# print(sys.path)

# # print current path 
# print(f"Current working directory: {Path.cwd()}")

# from src.utils.saving_and_finding_files import time_stamp_for_saving

# # ESN + graph-measure imports
# from src.ESNs.generate_weight_matrices_bio_no_rank_weighted import build_weight_matrix_from_bin_conn, build_weight_matrix_from_connectome
# from src.ESNs.test_memory_capacity_weighted import evaluate_memory_capacity

# # SEQUENCE RECALL
# # from src.ESNs.test_sequence_recall_weighted import evaluate_sequence_recall
# from src.structural_analysis.graph_measures_optimized_weighted import analyze_connectomes



# # ----------------------------------------------------------------------------
# # 1. NUMBA HELPERS
# # ----------------------------------------------------------------------------

# @njit(cache=True)
# def _compute_homophily_bin(A_bin: np.ndarray) -> np.ndarray:
#     """
#     Compute the homophily matrix for a binary adjacency matrix.
#     """
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
# def generate_gnm_numba(A_seed, target_m, dist, gamma, eta, eps: float = 1e-12):
#     n = A_seed.shape[0]
#     A = A_seed.copy()
#     D = np.maximum(dist, eps)
#     cost_term = D ** eta
#     np.fill_diagonal(cost_term, 0.0)

#     m = int(A.sum() // 2)
#     while m < target_m:
#         H = _compute_homophily_bin(A)
#         prob = cost_term * np.maximum(H, eps) ** gamma

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
#             k = cand_idx[_rand_choice_weighted(p_vec)]
#         i, j = divmod(k, n)
#         A[i, j] = A[j, i] = 1
#         m += 1
#     return A

# # ----------------------------------------------------------------------------
# # 2. UTILITIES
# # ----------------------------------------------------------------------------

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


# def _append_rows_csv(rows, columns, path):
#     path = Path(path)
#     if not rows:
#         return
#     pd.DataFrame(rows, columns=columns).to_csv(
#         path, mode="a", index=False, header=not path.exists()
#     )


# # ----------------------------------------------------------------------------
# # 3. ENERGY
# # ----------------------------------------------------------------------------

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

# # ----------------------------------------------------------------------------
# # 4. SUBJECT-LEVEL WORKER
# # ----------------------------------------------------------------------------
# def _subject_job(subj_idx: int,
#                  A_obs: np.ndarray,
#                  dist: np.ndarray,
#                  esn_use_observed_real_weights: bool,
#                  grid20: List[Tuple[float, float]],
#                  grid8: List[Tuple[float, float]],
#                  include_subset: bool,
#                  subset_size: int,
#                  timing_flag: bool
#                  ):
#     # Stage 0: observed metrics
#     t0 = time.perf_counter()
#     A_bin = _binarize(A_obs)
#     m_edges = int(A_bin.sum() // 2)
#     deg_obs, cc_obs, bc_obs, el_obs = precompute_obs_metrics(A_bin, dist)
#     t_metrics = time.perf_counter() - t0

#     # Stage 1: full grid
#     seed = np.zeros_like(A_obs, dtype=np.int8)
#     best_energy = np.inf
#     best_eta = best_gamma = None
#     grid_rows = []
#     t_grow = 0.0
#     for eta, gamma in grid20:
#         tg0 = time.perf_counter()
#         A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
#         t_grow += time.perf_counter() - tg0
#         energy = gnm_energy_fast(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist)
#         grid_rows.append((subj_idx, eta, gamma, energy))
#         if energy < best_energy:
#             best_energy, best_eta, best_gamma = energy, eta, gamma

#     # Stage 2: best model metrics
#     t_esn0 = time.perf_counter()
#     A_best = generate_gnm_numba(seed, m_edges, dist, best_gamma, best_eta)
#     gms_raw = analyze_connectomes(connectomes=np.expand_dims(A_best, 2),
#                                   distance_matrix=dist,
#                                   comm_mode='estrada_scaled')[0]
    
#     # results = analyze_connectomes(
#     #     connectomes,           # (n,n), (k,n,n) or (n,n,k)
#     #     distance_matrix,       # (n,n)
#     #     comm_mode="estrada_scaled",
#     #     use_weighted=True,     # preserve real weights
#     #     treat_weights_as_lengths=False,  # set True if your entries are already distances
#     #     min_weight=0.0,        # raise to sparsify
#     #     symmetrize="max",      # or "mean"
#     # )
    
#     if isinstance(gms_raw, dict):
#         gm_labels = list(gms_raw.keys())
#         gm_vals = list(gms_raw.values())
#     else:
#         gm_labels = [f"gm_{i}" for i in range(len(gms_raw))]
#         gm_vals = list(gms_raw)

#     W = (build_weight_matrix_from_connectome(A_obs, spectral_radius=0.99)
#          if esn_use_observed_real_weights else
#          build_weight_matrix_from_bin_conn(A_best, spectral_radius=0.99, rank=False))
#     eig_max = np.max(np.abs(np.linalg.eigvals(W)))
#     if eig_max > 0:
#         W *= 0.8 / eig_max
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", RuntimeWarning)
#         np.seterr(over="ignore", divide="ignore", invalid="ignore")
#         mc = evaluate_memory_capacity(W, n_lags=50, train_len=4000, test_len=1000,
#                                       n_runs=10, random_state=subj_idx)
#         # SEQUENCE RECALL
#         # sr = evaluate_sequence_recall(W, pattern_lengths=range(5, 26),
#         #                               train_trials=800, test_trials=200,
#         #                               n_runs=5, random_state=subj_idx)
        
#     t_esn = time.perf_counter() - t_esn0

#     # best_row = (subj_idx, best_eta, best_gamma, best_energy, mc, sr, *gm_vals)
#     best_row = (subj_idx, best_eta, best_gamma, best_energy, mc, *gm_vals)
#     timing = dict(metrics=t_metrics, grow=t_grow, esn=t_esn)

#     # Stage 3: subset grid metrics if enabled
#     subset_labels = None
#     if include_subset:

#         subset_rows = []


#         for eta, gamma in grid8:
#             A_sim = generate_gnm_numba(seed, m_edges, dist, gamma, eta)
#             gms = analyze_connectomes(connectomes=np.expand_dims(A_sim, 2),
#                                      distance_matrix=dist,
#                                      comm_mode='estrada_scaled')[0]
#             # results = analyze_connectomes(
#             #     connectomes,           # (n,n), (k,n,n) or (n,n,k)
#             #     distance_matrix,       # (n,n)
#             #     comm_mode="estrada_scaled",
#             #     use_weighted=True,     # preserve real weights
#             #     treat_weights_as_lengths=False,  # set True if your entries are already distances
#             #     min_weight=0.0,        # raise to sparsify
#             #     symmetrize="max",      # or "mean"
#             # )
#             if isinstance(gms, dict):
#                 labels = list(gms.keys())
#                 vals = list(gms.values())
#             else:
#                 labels = [f"gm_{i}" for i in range(len(gms))]
#                 vals = list(gms)
#             W = (build_weight_matrix_from_connectome(A_obs, spectral_radius=0.99)
#                  if esn_use_observed_real_weights else
#                  build_weight_matrix_from_bin_conn(A_sim, spectral_radius=0.99, rank=False))
#             eig_max = np.max(np.abs(np.linalg.eigvals(W)))
#             if eig_max > 0:
#                 W *= 0.8 / eig_max
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore", RuntimeWarning)
#                 np.seterr(over="ignore", divide="ignore", invalid="ignore")
#                 mc_s = evaluate_memory_capacity(W, n_lags=50, train_len=4000, test_len=1000,
#                                                 n_runs=10, random_state=subj_idx)
#                 # SEQUENCE RECALL
#                 # sr_s = evaluate_sequence_recall(W, pattern_lengths=range(5, 26),
#                 #                                 train_trials=800, test_trials=200,
#                 #                                 n_runs=5, random_state=subj_idx)
                
#             # SEQUENCE RECALL
#             # subset_rows.append((subj_idx, eta, gamma, mc_s, sr_s, *vals))
#             subset_rows.append((subj_idx, eta, gamma, mc_s, *vals))

#             subset_labels = labels

#     # if timing_flag:
#     #     print(f"Subject {subj_idx}: metrics={t_metrics:.3f}s, grow={t_grow:.3f}s, esn={t_esn:.3f}s")

#     return grid_rows, best_row, gm_labels, timing, subset_rows, subset_labels

# # ----------------------------------------------------------------------------
# # 5. DRIVER
# # ----------------------------------------------------------------------------

# def run_gnm_full(conn: np.ndarray,
#                  dist: np.ndarray,
#                  esn_use_observed_real_weights: bool = True,
#                  n_eta: int = 20,
#                  n_gamma: int = 20,
#                  eta_start: float = -3.0,
#                  eta_end: float = 0.0,
#                  gamma_start: float = 0.1,
#                  gamma_end: float = 0.6,
#                  subset_size: int = 8, # how many points to sample in the subset grid (e.g.: 8x8) 
#                  include_subset: bool = True,
#                  append_interval: int = 10,
#                  timing_flag: bool = False, 
#                  timestamp: str = None,
#                  save_dir: str = None):
    
#     # Paths 
#     # partial file paths
#     grid_partial   = save_dir / f"gnm_grid_partial_{timestamp}.csv"
#     best_partial   = save_dir / f"gnm_best_partial_{timestamp}.csv"
#     subset_partial = save_dir / f"gnm_subset_partial_{timestamp}.csv"
#     timing_partial = save_dir / f"gnm_timing_partial_{timestamp}.csv"

#     # Start timer 
#     run_t0 = time.perf_counter()

#     # prepare grids
#     eta_vals = np.linspace(eta_start, eta_end, n_eta)
#     gamma_vals = np.linspace(gamma_start, gamma_end, n_gamma)
#     grid20 = [(e, g) for e in eta_vals for g in gamma_vals]
#     eta8 = np.linspace(eta_start, eta_end, subset_size)
#     gamma8 = np.linspace(gamma_start, gamma_end, subset_size)
#     grid8 = [(e, g) for e in eta8 for g in gamma8]

#     n_subj = conn.shape[2]
#     tasks = [ (i, conn[:,:,i], dist, esn_use_observed_real_weights, grid20, grid8, include_subset, subset_size, timing_flag)
#               for i in range(n_subj) ]

#     if mp.get_start_method(allow_none=True) != "spawn":
#         mp.set_start_method("spawn", force=True)

#     grid_rows_all, best_rows, timing_list = [], [], []
#     subset_rows_all = []
#     gm_labels_master = subset_labels_master = None

#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
#         futures = [pool.submit(_subject_job, *t) for t in tasks]
        
#         for idx, fut in enumerate(as_completed(futures), 1):
#             g_rows, b_row, gm_labels, timing, sub_rows, sub_labels = fut.result()
#             grid_rows_all.extend(g_rows)
#             best_rows.append(b_row)
#             timing_list.append(timing)
#             subset_rows_all.extend(sub_rows)

#             if gm_labels_master is None:
#                 gm_labels_master = gm_labels
#             if subset_labels_master is None:
#                 subset_labels_master = sub_labels

#             # NEW: write freshly finished subject results immediately
#             _append_rows_csv(g_rows,
#                              ["subject","eta","gamma","energy"],
#                              grid_partial)

#             _append_rows_csv([b_row],
#                              ["subject","eta","gamma","energy","memory_capacity", *gm_labels_master],
#                              best_partial)

#             if include_subset and sub_rows:
#                 _append_rows_csv(sub_rows,
#                                  ["subject","eta","gamma","memory_capacity", *subset_labels_master],
#                                  subset_partial)

#             # NEW: save timings for this subject
#             _append_rows_csv([(b_row[0],
#                                timing["metrics"],
#                                timing["grow"],
#                                timing["esn"],
#                                timing["metrics"] + timing["grow"] + timing["esn"])],
#                              ["subject","metrics_s","grow_s","esn_s","total_s"],
#                              timing_partial)

#             # NEW: print exactly when rows are flushed
#             if timing_flag:
#                 print(f"Finished now {idx}/{n_subj} subjects. "
#                       f"Subject {b_row[0]}: metrics={timing['metrics']:.3f}s, "
#                       f"grow={timing['grow']:.3f}s, esn={timing['esn']:.3f}s")

        
#         # for idx, fut in enumerate(as_completed(futures), 1):
#         #     g_rows, b_row, gm_labels, timing, sub_rows, sub_labels = fut.result()
#         #     grid_rows_all.extend(g_rows)
#         #     best_rows.append(b_row)
#         #     timing_list.append(timing)
#         #     subset_rows_all.extend(sub_rows)

#         #     if gm_labels_master is None:
#         #         gm_labels_master = gm_labels
#         #     if subset_labels_master is None:
#         #         subset_labels_master = sub_labels

#         #     # append intermediate results
#         #     if idx % append_interval == 0:
#         #         pd.DataFrame(grid_rows_all, columns=["subject","eta","gamma","energy"])\
#         #           .to_csv(save_dir / f"gnm_grid_partial_{timestamp}.csv", index=False)
#         #         pd.DataFrame(best_rows, columns=["subject","eta","gamma","energy","memory_capacity","sequence_recall",*gm_labels_master])\
#         #           .to_csv(save_dir / f"gnm_best_partial_{timestamp}.csv", index=False)
#         #         if include_subset:
#         #             cols = ["subject","eta","gamma","memory_capacity","sequence_recall",*subset_labels_master]
#         #             pd.DataFrame(subset_rows_all, columns=cols)\
#         #               .to_csv(save_dir / f"gnm_subset_partial_{timestamp}.csv", index=False)
                    
#         #             # save subset of memory capacity and sequence recall  
#         #             # pd.DataFrame({
#         #             #     "subject": [r[0] for r in sub_rows],
#         #             #     "eta": [r[1] for r in sub_rows],
#         #             #     "gamma": [r[2] for r in sub_rows],
#         #             #     "memory_capacity": sub_mc_s,
#         #             #     "sequence_recall": sub_sr_s
#         #             # }).to_csv(save_dir / f"gnm_subset_mc_sr_partial_{timestamp}.csv", index=False)

#     total_sec = time.perf_counter() - run_t0
#     if timing_flag:
#         print(f"Total run time: {total_sec:.3f}s")
#         print(f"Average time per subject: {total_sec / n_subj:.3f}s")
    
#     # timing footer: put the total in 'total_s'
#     _append_rows_csv(
#         [("COMPLETED", np.nan, np.nan, np.nan, f"{total_sec:.3f}")],
#         ["subject","metrics_s","grow_s","esn_s","total_s"],
#         timing_partial
#     )

#     # return full dataframes (no final save)
#     df_grid = pd.DataFrame(grid_rows_all, columns=["subject","eta","gamma","energy"])
#     # SEQUENCE RECALL
#     # df_best = pd.DataFrame(best_rows, columns=["subject","eta","gamma","energy","memory_capacity","sequence_recall",*gm_labels_master])
#     df_best = pd.DataFrame(best_rows, columns=["subject","eta","gamma","energy","memory_capacity",*gm_labels_master])

#     df_subset = None
#     if include_subset:
#         # SEQUENCE RECALL
#         # cols = ["subject","eta","gamma","memory_capacity","sequence_recall",*subset_labels_master]
#         cols = ["subject","eta","gamma","memory_capacity",*subset_labels_master]

#         df_subset = pd.DataFrame(subset_rows_all, columns=cols)

#     return df_grid, df_best, df_subset

# # ----------------------------------------------------------------------------
# # 6. MAIN
# # ----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # user-configurable flags/variables
#     # ROOT = os.path.expanduser("~/Projects/14_4D_lab")
#     ROOT = Path("/Users/adrian/Documents/01_projects/14_4D_lab")
#     SAVE_DIR = ROOT / "output/02_gnm_estimation"
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # flags & parameters
#     N_ETA = 100
#     N_GAMMA = 100
#     RESOLUTION = 68 
#     INCLUDE_SUBSET = True 
#     SUBSET_SIZE = 4
#     APPEND_INTERVAL = 10
#     TIMING_FLAG = True

#     # load data
#     # conn = np.load(os.path.join(ROOT, "data/preprocessed/01_first_analysises/connectomes_binarized.npy")).T
#     conn = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/connectomes_binarized_{RESOLUTION}_{RESOLUTION}.npy")).T
#     # dist = np.load(os.path.join(ROOT, "data/preprocessed/01_first_analysises/distance_matrix.npy"))
#     dist = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/distance_matrix_{RESOLUTION}_{RESOLUTION}.npy"))

#     # timestamp for saving
#     timestamp = time_stamp_for_saving()
    
#     # create save directories
#     folder_name = f"gnm_resolution{RESOLUTION}_etas{N_ETA}_gammas{N_GAMMA}_{timestamp}"
#     (SAVE_DIR / folder_name / "results").mkdir(parents=True, exist_ok=True)
#     (SAVE_DIR / folder_name / "preliminary_results").mkdir(parents=True, exist_ok=True)

#     # run
#     df_grid, df_best, df_subset = run_gnm_full(
#         conn, dist,
#         n_eta=N_ETA,
#         n_gamma=N_GAMMA,
#         subset_size=SUBSET_SIZE,
#         include_subset=INCLUDE_SUBSET,
#         append_interval=APPEND_INTERVAL,
#         timing_flag=TIMING_FLAG,
#         timestamp=timestamp,
#         save_dir=SAVE_DIR / folder_name / "preliminary_results"  # save in a subdirectory for clarity
#     )
#     # save results
#     df_grid.to_csv(os.path.join(SAVE_DIR / folder_name / "results", f"gnm_grid_results_{timestamp}.csv"), index=False)
#     df_best.to_csv(os.path.join(SAVE_DIR / folder_name / "results", f"gnm_best_results_{timestamp}.csv"), index=False)
#     if df_subset is not None:
#         df_subset.to_csv(os.path.join(SAVE_DIR / folder_name / "results", f"gnm_subset_results_{timestamp}.csv"), index=False)

#     print(f"Results saved to {SAVE_DIR / folder_name / 'results'} and {SAVE_DIR / folder_name / 'preliminary_results'}.")
