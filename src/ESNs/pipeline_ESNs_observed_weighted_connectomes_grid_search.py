# Next steps: Turn this into a grid search for the best ESN hyperparameters
# This script is the real deal so far. 


import os, time, warnings, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from itertools import product
import random 

import sys
print(sys.path)

# print current path 
print(f"Current working directory: {Path.cwd()}")

from src.utils.saving_and_finding_files import time_stamp_for_saving

# ESN + graph-measure imports
from src.ESNs.test_memory_capacity_weighted import evaluate_memory_capacity_from_connectome
from src.structural_analysis.graph_measures_optimized_weighted import analyze_connectomes


def _append_rows_csv(rows, columns, path):
    """
    Utility function to append rows to a CSV file, creating the file if it does not exist.
    """
    
    path = Path(path)
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(
        path, mode="a", index=False, header=not path.exists()
    )


# def _subject_job(subj_idx: int,
#                  A_obs: np.ndarray,
#                  dist: np.ndarray,
#                  esn_use_observed_real_weights: bool,
#                  timing_flag: bool = True
#                  ):
#     """
#     Subject-level job for processing a single subject's connectome.
#     """

#     # Start timer
#     if timing_flag: 
#         t0 = time.perf_counter()
#         t_metrics = time.perf_counter() - t0 

#     # Graph measures on observed weights (i.e.: not simulated) 
#     #  = analyze_connectomes(connectomes=np.expand_dims(A_obs, 2),
#     #                                distance_matrix=dist,
#     #                                comm_mode='estrada_scaled',
#     #                                use_weighted=True,
#     #                                treat_weights_as_lengths=False,
#     #                                symmetrize="max", 
#     #                                min_weight=0.0)
    
#     #     #     out.append({
#     #     #     "network_index": idx,
#     #     #     "avg_communicability": avg_comm,
#     #     #     "global_efficiency": glob_eff,
#     #     #     "modularity": modu,
#     #     #     "avg_clustering": avg_clust,
#     #     #     "avg_degree_or_strength": avg_deg_or_strength,
#     #     #     "transitivity": trans,
#     #     #     "avg_edge_distance": avg_dist,
#     #     #     "char_path_length": cpl,
#     #     #     "richclub_n_edges": n_rich_edges,
#     #     #     "richclub_avg_length": avg_rc_length,
#     #     #     "used_weighted_graph": bool(use_weighted),
#     #     #     "treat_weights_as_lengths": bool(treat_weights_as_lengths),
#     #     #     "min_weight_threshold": float(min_weight),
#     #     #     "symmetrize": symmetrize,
#         # })


#     # Memory capacity 
#     if timing_flag: 
#         t_esn0 = time.perf_counter()
        
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", RuntimeWarning)
#         np.seterr(over="ignore", divide="ignore", invalid="ignore")
        
#         mc_result_dict = evaluate_memory_capacity_from_connectome(connectome=A_obs, 
#                                                                     spectral_radius=0.99, 
#                                                                     n_lags=50, 
#                                                                     train_len=4000, 
#                                                                     test_len=1000,
#                                                                     n_runs=10, 
#                                                                     input_scaling=1.0,
#                                                                     regression_method="pinv",
#                                                                     n_transient=0,
#                                                                     leak_rate=1.0,
#                                                                     bias=1.0,
#                                                                     random_state=subj_idx)
        
#             # return {"all_run_outputs": mc_values, 
#             # "mc_mean": float(np.mean(mc_values)), 
#             # "mc_std": float(np.std(mc_values)), 
#             # hparams: dict}
          
#     if timing_flag:  
#         t_esn = time.perf_counter() - t_esn0
        
#         timing = {"subject": subj_idx, 
#                 "time_metrics_sec": t_metrics, 
#                 "time_esn_sec": t_esn, 
#                 "time_total_sec": t_metrics + t_esn}
#     else: 
#         timing = np.nan

#     return mc_result_dict, timing


def _subject_job(subj_idx: int,
                 A_obs: np.ndarray,
                 dist: np.ndarray,
                 timing_flag: bool = True,
                 hparams: dict | None = None):
    """
    Subject-level job for a single hparam set.
    """

    if timing_flag:
        t0 = time.perf_counter()
        t_metrics = time.perf_counter() - t0

    # New default hyperparams (backwards compatible)
    if hparams is None:
        hparams = {
            "spectral_radius": 0.99,
            "input_length": 4000,          # was train_len
            "input_scaling": 1.0,
            "regularization_method": "pinv"
        }

    if timing_flag:
        t_esn0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.seterr(over="ignore", divide="ignore", invalid="ignore")

        mc_result_dict = evaluate_memory_capacity_from_connectome(
            connectome=A_obs,
            spectral_radius=hparams["spectral_radius"],
            n_lags=50,
            train_len=hparams["input_length"],               # <- changed
            test_len=1000,
            n_runs=hparams.get("n_runs", 10),                # still supported if provided
            input_scaling=hparams["input_scaling"],          # <- changed
            regression_method=hparams["regularization_method"],  # <- changed
            n_transient=0,
            leak_rate=1.0,
            bias=1.0,
            random_state=subj_idx
        )

    if isinstance(mc_result_dict, dict) and "hparams" not in mc_result_dict:
        mc_result_dict["hparams"] = hparams

    if timing_flag:
        t_esn = time.perf_counter() - t_esn0
        timing = {
            "subject": subj_idx,
            "time_metrics_sec": t_metrics,
            "time_esn_sec": t_esn,
            "time_total_sec": t_metrics + t_esn
        }
    else:
        timing = np.nan

    return mc_result_dict, timing



# def run_gnm_full(conn: np.ndarray,
#                  dist: np.ndarray,
#                  timing_flag: bool = False, 
#                  timestamp: str = None,
#                  save_dir: str = None):
#     """
#     Run the GNM pipeline for all subjects in parallel. 
#     """
    
#     # Paths 
#     mc_values_csv_path = save_dir / f"gnm_mc_results_{timestamp}.csv"
#     duration_csv_path = save_dir / f"gnm_durations_{timestamp}.csv"
        
#     # Start timer 
#     if timing_flag:
#         run_t0 = time.perf_counter()

#     n_subj = conn.shape[2]
#     tasks = [ (i, conn[:,:,i], dist, timing_flag) for i in range(n_subj) ]

#     if mp.get_start_method(allow_none=True) != "spawn":
#         mp.set_start_method("spawn", force=True)
        
#     mc_values = []
#     timing_list = []

#     rows_to_save_mc = ["subject", "mc_mean", "mc_std", "mean_mc_of_individual_runs", "hparams"]
#     rows_to_save_durations = ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"]
    
#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
#         futures = [pool.submit(_subject_job, *t) for t in tasks]

#         for idx, fut in enumerate(as_completed(futures), 1):
#             mc_result_dict, timing_dict = fut.result()
#             mc_values.append(mc_result_dict)
#             if timing_flag:
#                 timing_list.append(timing_dict)

#             values_to_save_mc = [timing_dict["subject"]] + [mc_result_dict[row_name] for row_name in rows_to_save_mc[1:]]
#             _append_rows_csv([values_to_save_mc],
#                              rows_to_save_mc,
#                              mc_values_csv_path)

#             # Save timings for this subject
#             if timing_flag:
#                 values_to_save_durations = [timing_dict[row_name] for row_name in rows_to_save_durations]
#                 _append_rows_csv([values_to_save_durations],
#                                 rows_to_save_durations,
#                                 duration_csv_path)

#             # Print exactly when rows are flushed
#             print(f"Finished now {idx}/{n_subj} subjects.") 

#     if timing_flag:
#         total_sec = time.perf_counter() - run_t0
#         print(f"Total run time: {total_sec:.3f}s")
#         print(f"Average time per subject: {total_sec / n_subj:.3f}s")
    
#         # timing footer: put the total in 'total_s'
#         _append_rows_csv(
#             [("COMPLETED", np.nan, np.nan, f"{total_sec:.3f}")],
#             ["subject","time_metrics_sec","time_esn_sec","time_total_sec"],
#             duration_csv_path
#         )    
           
#     _append_rows_csv(
#         ["COMPLETED"],
#         ["subject"],
#         mc_values_csv_path
#     )

#     return f"Results saved to {save_dir}."


def run_gnm_full(conn,                   # can be np.ndarray or dict[int, np.ndarray]
                 dist: np.ndarray,
                 hparam_grid: list[dict] | None = None,
                 timing_flag: bool = False,
                 timestamp: str = None,
                 save_dir: str = None):
    """
    Run the GNM pipeline for all subjects across a grid of ESN hyperparameters in parallel.
    If hparam_grid is None, a single default config is used (backwards compatible).

    `conn` can be:
      - a single 3D array (n_nodes, n_nodes, n_subj), or
      - a dict mapping density percent -> 3D array as above
    """

    # Paths
    mc_values_csv_path = save_dir / f"gnm_mc_results_{timestamp}.csv"
    duration_csv_path = save_dir / f"gnm_durations_{timestamp}.csv"

    # Start timer
    if timing_flag:
        run_t0 = time.perf_counter()

    # Determine number of subjects
    if isinstance(conn, dict):
        example_conn = next(iter(conn.values()))
        n_subj = example_conn.shape[2]
    else:
        n_subj = conn.shape[2]

    # If no grid given, use a single default (keeps old behavior)
    if not hparam_grid:
        hparam_grid = [None]

    # Build subject × hparam tasks
    tasks = []
    for i in range(n_subj):
        for hp in hparam_grid:
            if isinstance(conn, dict):
                density = hp.get("density_percent")
                A_i = conn[density][:, :, i]
            else:
                A_i = conn[:, :, i]
            tasks.append((i, A_i, dist, timing_flag, hp))

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    mc_values = []
    timing_list = []

    rows_to_save_mc = ["subject", "mc_mean", "mc_std", "mean_mc_of_individual_runs", "hparams"]
    rows_to_save_durations = ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"]

    total_tasks = len(tasks)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(_subject_job, *t) for t in tasks]

        for idx, fut in enumerate(as_completed(futures), 1):
            mc_result_dict, timing_dict = fut.result()
            mc_values.append(mc_result_dict)
            if timing_flag:
                timing_list.append(timing_dict)

            # Save MC row
            subject_id = timing_dict["subject"] if timing_flag else np.nan
            values_to_save_mc = [subject_id] + [mc_result_dict.get(row_name) for row_name in rows_to_save_mc[1:]]
            _append_rows_csv([values_to_save_mc], rows_to_save_mc, mc_values_csv_path)

            # Save timings for this subject/hparam combo
            if timing_flag:
                values_to_save_durations = [timing_dict[row_name] for row_name in rows_to_save_durations]
                _append_rows_csv([values_to_save_durations], rows_to_save_durations, duration_csv_path)

            # Progress
            hp_short = mc_result_dict.get("hparams", {})
            print(f"Finished {idx}/{total_tasks} tasks (subject={subject_id}, hp={hp_short}).")

    if timing_flag:
        total_sec = time.perf_counter() - run_t0
        print(f"Total run time: {total_sec:.3f}s")
        print(f"Average time per task: {total_sec / total_tasks:.3f}s")

        # timing footer
        _append_rows_csv(
            [("COMPLETED", np.nan, np.nan, f"{total_sec:.3f}")],
            ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"],
            duration_csv_path
        )

    _append_rows_csv(
        ["COMPLETED"],
        ["subject"],
        mc_values_csv_path
    )

    return f"Results saved to {save_dir}."



if __name__ == "__main__":
    # user-configurable flags/variables
    ROOT = Path("/Users/adrian/Documents/01_projects/14_4D_lab")
    SAVE_DIR = ROOT / "output/02_gnm_estimation"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Flags and parameters
    RESOLUTION = 68
    # DENSITY = 10
    
    TIMING_FLAG = True
    
    
    
    # --- ADD: define your hyperparameter grid (or random sample of it) ---
    # # Regular grid
    # HP_SPECTRAL_RADII = [0.7, 0.8, 0.9, 0.99]
    # HP_TRAIN_LENS     = [1000, 2000, 4000]
    # HP_N_RUNS_LIST    = [3, 5, 10]

    # input_length
    # Input_scaling
    # regularization_method
    # (spectral radius from 0.1 to 2-3)
    # --- NEW GRID: input_length, input_scaling, regularization_method, spectral_radius ---
    
    # Available densities must match files on disk
    HP_DENSITIES = [10, 12, 14, 16, 18, 20]   # adjust if you have more/less

    # spectral radius extended up to ~3 (adjust resolution as needed)
    N_RUNS_FIXED             = [10] 
    # HP_SPECTRAL_RADII        = [0.99] # np.linspace(0.1, 2.5, 11)   # e.g., 0.1 … 2.5
    HP_SPECTRAL_RADII        = np.linspace(0.1, 2.5, 11)   # e.g., 0.1 … 2.5
    HP_INPUT_LENGTHS         = [500, 1000, 2000, 4000, 8000]          # was train_len
    HP_INPUT_SCALINGS        = [0.1, 0.5, 1.0]
    HP_REGULARIZATION_METHOD = ["pinv", "ridge"]           # keep in sync with your ESN impl
    # N_TRANSIENT              = 0  # keep this fixed for now, can be varied later

# https://fabridamicelli.github.io/echoes/api/ESNGenerator/

    hparam_grid = [
        {
            "spectral_radius": sr,
            "input_length": ilen,
            "input_scaling": iscale,
            "regularization_method": reg,
            # optional: keep compatibility if you want n_runs fixed or varied later
            "n_runs": nr,
            # "n_transient": n_transient, 
            "density_percent": dens, 
        }
        for sr, ilen, iscale, reg, nr, dens in product(
            HP_SPECTRAL_RADII, HP_INPUT_LENGTHS, HP_INPUT_SCALINGS, HP_REGULARIZATION_METHOD, N_RUNS_FIXED, HP_DENSITIES
        )
    ]

    # Optional: randomly subsample the grid for quicker sweeps
    HP_RANDOM_SAMPLE = None  # e.g., 24
    if HP_RANDOM_SAMPLE:
        random.seed(0)
        hparam_grid = random.sample(hparam_grid, min(HP_RANDOM_SAMPLE, len(hparam_grid)))

    print(f"Hyperparameter combos: {len(hparam_grid)}")


    # load data
    # conn = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/connectomes_weighted_{RESOLUTION}x{RESOLUTION}.npy")).T
    
     # Load one connectome cube per density and pass them as a dict
    conn_by_density = {
        d: np.load(os.path.join(
            ROOT,
            f"data/preprocessed/01_first_analysises/connectomes_binarized_{RESOLUTION}x{RESOLUTION}_density_{d}_percent.npy"
        )).T.astype(np.float64, copy=False)

        for d in HP_DENSITIES
    }
    
    dist = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/distance_matrix_{RESOLUTION}x{RESOLUTION}.npy"))

    # timestamp for saving
    timestamp = time_stamp_for_saving()
    
    # create save directories
    folder_name = f"esn_grid_resolution{RESOLUTION}_{timestamp}"
    (SAVE_DIR / folder_name).mkdir(parents=True, exist_ok=True)

    # run 
    # final_message = run_gnm_full(
    #     conn, dist,
    #     timing_flag=TIMING_FLAG,
    #     timestamp=timestamp,
    #     save_dir=SAVE_DIR / folder_name
    # )
    final_message = run_gnm_full(
        conn_by_density, dist,
        hparam_grid=hparam_grid,
        timing_flag=TIMING_FLAG,
        timestamp=timestamp,
        save_dir=SAVE_DIR / folder_name
    )

    print(final_message)
