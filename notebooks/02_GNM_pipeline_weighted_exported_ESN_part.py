import os, time, warnings, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import ks_2samp

import sys
print(sys.path)

# print current path 
print(f"Current working directory: {Path.cwd()}")

from src.utils.saving_and_finding_files import time_stamp_for_saving

# ESN + graph-measure imports
# from src.ESNs.test_memory_capacity_weighted import evaluate_memory_capacity, evaluate_memory_capacity_from_connectome
from src.ESNs.test_memory_capacity_weighted import evaluate_memory_capacity_from_connectome

# SEQUENCE RECALL
# from src.ESNs.test_sequence_recall_weighted import evaluate_sequence_recall
from src.structural_analysis.graph_measures_optimized_weighted import analyze_connectomes


# ----------------------------------------------------------------------------
# 2. UTILITIES
# ----------------------------------------------------------------------------

def precompute_obs_metrics(A_bin: np.ndarray, dist: np.ndarray):
    G = nx.from_numpy_array(A_bin)
    deg = np.array([d for _, d in G.degree()])
    cc = np.array(list(nx.clustering(G).values()))
    bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    el = np.array([dist[i, j] for i, j in G.edges()])
    return deg, cc, bc, el


def _append_rows_csv(rows, columns, path):
    path = Path(path)
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(
        path, mode="a", index=False, header=not path.exists()
    )


# ----------------------------------------------------------------------------
# 3. ENERGY
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# 4. SUBJECT-LEVEL WORKER
# ----------------------------------------------------------------------------
def _subject_job(subj_idx: int,
                 A_obs: np.ndarray,
                 dist: np.ndarray,
                 esn_use_observed_real_weights: bool,
                 timing_flag: bool = True
                 ):
    # Stage 0: metrics directly for the observed (weighted) connectome.
    t0 = time.perf_counter()
    t_metrics = time.perf_counter() - t0 

    # Graph measures on observed weights (i.e.: not simulated) 
    gmns_raw = analyze_connectomes(connectomes=np.expand_dims(A_obs, 2),
                                   distance_matrix=dist,
                                   comm_mode='estrada_scaled',
                                   use_weighted=True,
                                   treat_weights_as_lengths=False,
                                   symmetrize="max", 
                                   min_weight=0.0)

    # if isinstance(gmns_raw, dict):
    #     gmn_labels = list(gmns_raw.keys())
    #     gmn_vals = list(gmns_raw.values())
    # else:
    #     gmn_labels = [f"gm_{i}" for i in range(len(gmns_raw))]
    #     gmn_vals = list(gmns_raw)
        
    # Memory capacity 
    t_esn0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.seterr(over="ignore", divide="ignore", invalid="ignore")
        
        mc_result_dict = evaluate_memory_capacity_from_connectome(A_obs, 
                                                      spectral_radius=0.99, 
                                                      n_lags=50, 
                                                      train_len=4000, 
                                                      test_len=1000,
                                                      n_runs=10, 
                                                      random_state=subj_idx)
            # return {"all_run_outputs": mc_values, 
            # "mc_mean": float(np.mean(mc_values)), 
            # "mc_std": float(np.std(mc_values)), 
            # hparams: dict}
            
    t_esn = time.perf_counter() - t_esn0
    
    timing = {"subject": subj_idx, 
              "time_metrics_sec": t_metrics, 
              "time_esn_sec": t_esn, 
              "time_total_sec": t_metrics + t_esn}

    return mc_result_dict, timing

# ----------------------------------------------------------------------------
# 5. DRIVER
# ----------------------------------------------------------------------------

def run_gnm_full(conn: np.ndarray,
                 dist: np.ndarray,
                 timing_flag: bool = False, 
                 timestamp: str = None,
                 save_preliminary_results: bool = True,
                 save_preliminary_results_dir: str = None):
    
    # Paths 
    # partial file paths
    if save_preliminary_results:
        mc_values_partial = save_preliminary_results_dir / f"gnm_mc_results_partial_{timestamp}.csv"
        timing_partial = save_preliminary_results_dir / f"gnm_timing_partial_{timestamp}.csv"
        
    # Start timer 
    run_t0 = time.perf_counter()

    n_subj = conn.shape[2]
    tasks = [ (i, conn[:,:,i], dist, timing_flag) for i in range(n_subj) ]

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
        
    mc_values = []
    timing_list = []

    rows_to_save_mc = ["subject", "mc_mean", "mc_std", "mean_mc_of_individual_runs", "hparams"]
    rows_to_save_timing = ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(_subject_job, *t) for t in tasks]

        for idx, fut in enumerate(as_completed(futures), 1):
            mc_result_dict, timing_dict = fut.result()
            mc_values.append(mc_result_dict)
            timing_list.append(timing_dict)

            values_to_save_mc = [timing_dict["subject"]] + [mc_result_dict[row_name] for row_name in rows_to_save_mc[1:]]
            _append_rows_csv([values_to_save_mc],
                             rows_to_save_mc,
                             mc_values_partial)

            # Save timings for this subject
            values_to_save_timing = [timing_dict[row_name] for row_name in rows_to_save_timing]
            _append_rows_csv([values_to_save_timing],
                             rows_to_save_timing,
                             timing_partial)

            # Print exactly when rows are flushed
            if timing_flag:
                print(f"Finished now {idx}/{n_subj} subjects.") 

    total_sec = time.perf_counter() - run_t0
    if timing_flag:
        print(f"Total run time: {total_sec:.3f}s")
        print(f"Average time per subject: {total_sec / n_subj:.3f}s")
    
    # timing footer: put the total in 'total_s'
    _append_rows_csv(
        [("COMPLETED", np.nan, np.nan, f"{total_sec:.3f}")],
        ["subject","time_metrics_sec","time_esn_sec","time_total_sec"],
        timing_partial
    )
    _append_rows_csv(
        ["COMPLETED"],
        ["subject"],
        mc_values_partial
    )

    # return full dataframes (no final save)
    df_mc = pd.DataFrame(mc_values, columns=rows_to_save_mc) 
    df_timing = pd.DataFrame(timing_list, columns=rows_to_save_timing)
    
    return df_mc, df_timing


if __name__ == "__main__":
    # user-configurable flags/variables
    ROOT = Path("/Users/adrian/Documents/01_projects/14_4D_lab")
    SAVE_DIR = ROOT / "output/02_gnm_estimation"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Flags and parameters
    RESOLUTION = 68
    DENSITY = 10
    
    TIMING_FLAG = True
    SAVE_INTERMEDIATE_RESULTS = True

    # load data
    conn = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/connectomes_weighted_{RESOLUTION}x{RESOLUTION}.npy")).T
    dist = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/distance_matrix_{RESOLUTION}x{RESOLUTION}.npy"))

    # timestamp for saving
    timestamp = time_stamp_for_saving()
    
    # create save directories
    folder_name = f"esn_resolution{RESOLUTION}_{timestamp}"
    (SAVE_DIR / folder_name / "results").mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / folder_name / "preliminary_results").mkdir(parents=True, exist_ok=True)

    # run
    df_mc, df_timing = run_gnm_full(
        conn, dist,
        timing_flag=TIMING_FLAG,
        timestamp=timestamp,
        save_preliminary_results=SAVE_INTERMEDIATE_RESULTS,
        save_preliminary_results_dir=SAVE_DIR / folder_name / "preliminary_results"  # save in a subdirectory for clarity
    )
    # save results
    df_mc.to_csv(os.path.join(SAVE_DIR / folder_name, f"gnm_mc_results_{timestamp}.csv"), index=False)
    df_timing.to_csv(os.path.join(SAVE_DIR / folder_name, f"gnm_timing_results_{timestamp}.csv"), index=False)

    print(f"Results saved to {SAVE_DIR / folder_name}.")
    if SAVE_INTERMEDIATE_RESULTS: 
        print(f"Intermediate results saved to {SAVE_DIR / folder_name / 'preliminary_results'}.")
