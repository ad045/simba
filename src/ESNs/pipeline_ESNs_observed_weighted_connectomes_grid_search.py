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

from datetime import datetime
from collections.abc import Iterable

# print current path 
print(f"Current working directory: {Path.cwd()}")

from src.utils.saving_and_finding_files import time_stamp_for_saving

# ESN + graph-measure imports
from src.ESNs.test_memory_capacity_weighted import evaluate_memory_capacity_from_connectome
from src.structural_analysis.graph_measures_optimized_weighted import analyze_connectomes

from src.ESNs.utils import (_summarize_hparam_space, _write_run_info_txt) 

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
                 hparams: dict | None = None, 
                 random_seed: int | None = None):
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
            random_state=random_seed if random_seed is not None else subj_idx  # <- NEW: use random_seed if provided
        )
        
        returned_hp = mc_result_dict.get("hparams", {})
        # Add hparams so that they are not dropped
        mc_result_dict["hparams"] = {**returned_hp, **hparams}
        

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



# def run_gnm_full(conn,                   # can be np.ndarray or dict[int, np.ndarray]
#                  dist: np.ndarray,
#                  hparam_grid: list[dict] | None = None,
#                  timing_flag: bool = False,
#                  timestamp: str = None,
#                  save_dir: str = None):
#     """
#     Run the GNM pipeline for all subjects across a grid of ESN hyperparameters in parallel.
#     If hparam_grid is None, a single default config is used (backwards compatible).

#     `conn` can be:
#       - a single 3D array (n_nodes, n_nodes, n_subj), or
#       - a dict mapping density percent -> 3D array as above
#     """

#     # Paths
#     mc_values_csv_path = save_dir / f"gnm_mc_results_{timestamp}.csv"
#     duration_csv_path = save_dir / f"gnm_durations_{timestamp}.csv"

#     # Start timer
#     if timing_flag:
#         run_t0 = time.perf_counter()

#     # Determine number of subjects
#     if isinstance(conn, dict):
#         example_conn = next(iter(conn.values()))
#         n_subj = example_conn.shape[2]
#     else:
#         n_subj = conn.shape[2]

#     # If no grid given, use a single default (keeps old behavior)
#     if not hparam_grid:
#         hparam_grid = [None]

#     # Build subject × hparam tasks
#     tasks = []
#     for i in range(n_subj):
#         for hp in hparam_grid:
#             if isinstance(conn, dict):
#                 density = hp.get("density_percent")
#                 A_i = conn[density][:, :, i]
#             else:
#                 A_i = conn[:, :, i]
#             tasks.append((i, A_i, dist, timing_flag, hp))

#     if mp.get_start_method(allow_none=True) != "spawn":
#         mp.set_start_method("spawn", force=True)

#     mc_values = []
#     timing_list = []

#     # rows_to_save_mc = ["subject", "mc_mean", "mc_std", "mean_mc_of_individual_runs", "hparams"]
#     rows_to_save_mc = [
#         "subject",
#         "density_percent",
#         "spectral_radius",
#         "input_length",
#         "input_scaling",
#         "regularization_method",
#         "n_runs",
#         "mc_mean",
#         "mc_std",
#         "mean_mc_of_individual_runs",   
#         "hyper_params", # Store the full hyperparameter dict for reference
#     ]
    
#     rows_to_save_durations = ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"]

#     total_tasks = len(tasks)
#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
#         futures = [pool.submit(_subject_job, *t) for t in tasks]

#         for idx, fut in enumerate(as_completed(futures), 1):
#             mc_result_dict, timing_dict = fut.result()
#             mc_values.append(mc_result_dict)
#             if timing_flag:
#                 timing_list.append(timing_dict)
                
#             # Save the hyperparameters and results
#             hp = mc_result_dict.get("hparams", {})
#             subject_id = timing_dict["subject"] if timing_flag else np.nan # Do this in a cleaner way... 

#             values_to_save_mc = [ # Was previously created so nice automatically, maybe implement this in a cleaner way?
#                 subject_id, 
#                 hp.get("density_percent"),
#                 hp.get("spectral_radius"),
#                 hp.get("input_length"),
#                 hp.get("input_scaling"),
#                 hp.get("regularization_method"),
#                 hp.get("n_runs"),
#                 mc_result_dict.get("mc_mean"),
#                 mc_result_dict.get("mc_std"),
#                 mc_result_dict.get("mean_mc_of_individual_runs"),
#                 hp # Store the full hyperparameter dict for reference
#             ]

#             _append_rows_csv([values_to_save_mc], rows_to_save_mc, mc_values_csv_path)
                

#             # # Save MC row
#             # subject_id = timing_dict["subject"] if timing_flag else np.nan
#             # values_to_save_mc = [subject_id] + [mc_result_dict.get(row_name) for row_name in rows_to_save_mc[1:]]
#             # _append_rows_csv([values_to_save_mc], rows_to_save_mc, mc_values_csv_path)

#             # Save timings for this subject/hparam combo
#             if timing_flag:
#                 values_to_save_durations = [timing_dict[row_name] for row_name in rows_to_save_durations]
#                 _append_rows_csv([values_to_save_durations], rows_to_save_durations, duration_csv_path)

#             # Progress
#             hp_short = mc_result_dict.get("hparams", {})
#             print(f"Finished {idx}/{total_tasks} tasks (subject={subject_id}, hp={hp_short}).")

#     if timing_flag:
#         total_sec = time.perf_counter() - run_t0
#         print(f"Total run time: {total_sec:.3f}s")
#         print(f"Average time per task: {total_sec / total_tasks:.3f}s")

#         # timing footer
#         _append_rows_csv(
#             [("COMPLETED", np.nan, np.nan, f"{total_sec:.3f}")],
#             ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"],
#             duration_csv_path
#         )

#     _append_rows_csv(
#         ["COMPLETED"],
#         ["subject"],
#         mc_values_csv_path
#     )

#     return f"Results saved to {save_dir}."


def run_gnm_full(conn,
                 dist: np.ndarray,
                 hparam_grid: list[dict] | None = None,
                 timing_flag: bool = False,
                 timestamp: str = None,
                 save_dir: str = None,
                 search_mode: str = "grid",            # <-- NEW: "grid" or "random_sample"
                 random_sample_size: int | None = None,  # <-- NEW: if used
                 random_seed: int | None = None  # <-- NEW: for reproducibility
                 ):
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
    run_info_path = save_dir / f"run_info_{timestamp}.txt"   # <-- NEW

    # Start timer
    if timing_flag:
        run_t0 = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")  # <-- NEW

    # Determine number of subjects
    if isinstance(conn, dict):
        example_conn = next(iter(conn.values()))
        n_subj = example_conn.shape[2]
        densities_available = sorted(conn.keys())
    else:
        n_subj = conn.shape[2]
        densities_available = None

    # If no grid given, use a single default (keeps old behavior)
    if not hparam_grid:
        hparam_grid = [None]

    # Summarize the hparam space for the info file
    hparam_space_summary = _summarize_hparam_space(
        [hp for hp in hparam_grid if isinstance(hp, dict)]
    )

    # Build subject × hparam tasks
    tasks = []
    for i in range(n_subj):
        for hp in hparam_grid:
            if isinstance(conn, dict):
                density = hp.get("density_percent") if isinstance(hp, dict) else None
                A_i = conn[density][:, :, i]
            else:
                A_i = conn[:, :, i]
            tasks.append((i, A_i, dist, timing_flag, hp))

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    mc_values = []
    timing_list = []

    rows_to_save_mc = [
        "subject",
        "density_percent",
        "spectral_radius",
        "input_length",
        "input_scaling",
        "regularization_method",
        "n_runs",
        "mc_mean",
        "mc_std",
        "mean_mc_of_individual_runs",
        "hyper_params",
    ]
    rows_to_save_durations = ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"]

    total_tasks = len(tasks)

    # --- NEW: write an initial run_info before any task completes ---
    _write_run_info_txt(
        run_info_path,
        started_at=started_at,
        updated_at=started_at,
        save_dir=str(save_dir),
        timestamp=str(timestamp),
        n_subjects=n_subj,
        total_tasks=total_tasks,
        completed_tasks=0,
        search_mode=search_mode,
        random_sample_size=random_sample_size,
        densities_available=densities_available,
        hparam_space_summary=hparam_space_summary,
    )

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(_subject_job, *t) for t in tasks]

        for idx, fut in enumerate(as_completed(futures), 1):
            mc_result_dict, timing_dict = fut.result()
            mc_values.append(mc_result_dict)
            if timing_flag:
                timing_list.append(timing_dict)

            # --- Merge returned + grid hparams so nothing is lost (incl. density) ---
            hp_grid = mc_result_dict.get("hparams", {})
            # If _subject_job didn’t already merge, ensure here:
            if isinstance(hp_grid, dict):
                # hp_grid already includes merge from _subject_job if you added it;
                # this keeps it safe even if not.
                hp_from_job = hp_grid
            else:
                hp_from_job = {}

            # Save the hyperparameters and results
            subject_id = timing_dict["subject"] if timing_flag else np.nan
            values_to_save_mc = [
                subject_id,
                hp_from_job.get("density_percent"),
                hp_from_job.get("spectral_radius"),
                hp_from_job.get("input_length"),
                hp_from_job.get("input_scaling"),
                hp_from_job.get("regularization_method"),
                hp_from_job.get("n_runs"),
                mc_result_dict.get("mc_mean"),
                mc_result_dict.get("mc_std"),
                mc_result_dict.get("mean_mc_of_individual_runs"),
                hp_from_job,
            ]
            _append_rows_csv([values_to_save_mc], rows_to_save_mc, mc_values_csv_path)

            # Save timings
            if timing_flag:
                values_to_save_durations = [timing_dict[row_name] for row_name in rows_to_save_durations]
                _append_rows_csv([values_to_save_durations], rows_to_save_durations, duration_csv_path)

            # Progress
            print(f"Finished {idx}/{total_tasks} tasks (subject={subject_id}, hp={hp_from_job}).")

            # --- NEW: update run_info on every completion ---
            _write_run_info_txt(
                run_info_path,
                started_at=started_at,
                updated_at=datetime.now().isoformat(timespec="seconds"),
                save_dir=str(save_dir),
                timestamp=str(timestamp),
                n_subjects=n_subj,
                total_tasks=total_tasks,
                completed_tasks=idx,
                search_mode=search_mode,
                random_sample_size=random_sample_size,
                densities_available=densities_available,
                hparam_space_summary=hparam_space_summary,
            )

    if timing_flag:
        total_sec = time.perf_counter() - run_t0
        print(f"Total run time: {total_sec:.3f}s")
        print(f"Average time per task: {total_sec / total_tasks:.3f}s")
        _append_rows_csv(
            [("COMPLETED", np.nan, np.nan, f"{total_sec:.3f}")],
            ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"],
            duration_csv_path
        )

    _append_rows_csv(["COMPLETED"], ["subject"], mc_values_csv_path)

    # Final info write
    _write_run_info_txt(
        run_info_path,
        started_at=started_at,
        updated_at=datetime.now().isoformat(timespec="seconds"),
        save_dir=str(save_dir),
        timestamp=str(timestamp),
        n_subjects=n_subj,
        total_tasks=total_tasks,
        completed_tasks=total_tasks,
        search_mode=search_mode,
        random_sample_size=random_sample_size,
        densities_available=densities_available,
        hparam_space_summary=hparam_space_summary,
    )

    return f"Results saved to {save_dir}."


if __name__ == "__main__":
    # user-configurable flags/variables
    ROOT = Path("/Users/adrian/Documents/01_projects/14_4D_lab")
    SAVE_DIR = ROOT / "output/02_esns_on_observed_weighted_connectomes"
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
    
    #################################
    # # Available densities must match files on disk
    # HP_DENSITIES = [10, 12, 14, 16, 18, 20]   # adjust if you have more/less

    # # spectral radius extended up to ~3 (adjust resolution as needed)
    # N_RUNS_FIXED             = [10] 
    # # HP_SPECTRAL_RADII        = [0.99] # np.linspace(0.1, 2.5, 11)   # e.g., 0.1 … 2.5
    # HP_SPECTRAL_RADII        = np.linspace(0.1, 2.5, 11)   # e.g., 0.1 … 2.5
    # HP_INPUT_LENGTHS         = [500, 1000, 2000, 4000, 8000]          # was train_len
    # HP_INPUT_SCALINGS        = [0.1, 0.5, 1.0]
    # HP_REGULARIZATION_METHOD = ["pinv", "ridge"]           # keep in sync with your ESN impl
    # # N_TRANSIENT              = 0  # keep this fixed for now, can be varied later

    # # Took 79 minutes with current grid. 
    #################################
    
    # HP_DENSITIES = [10]   # adjust if you have more/less
    HP_DENSITIES = [10, 12, 14, 16, 18, 20]

    # spectral radius extended up to ~3 (adjust resolution as needed)
    N_RUNS_FIXED             = [100] 
    # HP_SPECTRAL_RADII        = [0.99] # np.linspace(0.1, 2.5, 11)   # e.g., 0.1 … 2.5
    HP_SPECTRAL_RADII        = np.linspace(0.1, 2.5, 10)   # e.g., 0.1 … 2.5
    HP_INPUT_LENGTHS         = [1000]          # was train_len
    HP_INPUT_SCALINGS        = [1.0]
    HP_REGULARIZATION_METHOD = ["ridge"]           # keep in sync with your ESN impl
    
    
    # Group by hyperparameters and calculate the mean of mc_mean
    # density_percent spectral_radius input_length input_scaling regularization_method
    # Best Hyperparameter Combination: (np.float64(10.0), np.float64(2.5), np.float64(1000.0), np.float64(1.0), 'ridge')
    # Best MC Mean: 1.0486020406868306
    #################################
    
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
    search_mode = "grid"
    random_sample_size = None
    
    if HP_RANDOM_SAMPLE:
        search_mode = "random_sample"
        random_sample_size = min(HP_RANDOM_SAMPLE, len(hparam_grid))
        random.seed(0)
        hparam_grid = random.sample(hparam_grid, random_sample_size)

    # load data
    # conn = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/connectomes_weighted_{RESOLUTION}x{RESOLUTION}.npy")).T
    
     # Load one connectome cube per density and pass them as a dict. Is using weighted connectomes.
    weighted_connectome = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/connectomes_weighted_{RESOLUTION}x{RESOLUTION}.npy"))
    conn_by_density = {
        d: (np.load(os.path.join(
            ROOT,
            f"data/preprocessed/01_first_analysises/connectomes_binarized_{RESOLUTION}x{RESOLUTION}_density_{d}_percent.npy"
        ))  * weighted_connectome).T.astype(np.float64, copy=False)

        for d in HP_DENSITIES
    }
    
    dist = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/distance_matrix_{RESOLUTION}x{RESOLUTION}.npy"))

    # timestamp for saving
    timestamp = time_stamp_for_saving()
    
    # create save directories
    folder_name = f"esn_grid_resolution{RESOLUTION}_{timestamp}"
    (SAVE_DIR / folder_name).mkdir(parents=True, exist_ok=True)

    # final_message = run_gnm_full(
    #     conn_by_density, dist,
    #     hparam_grid=hparam_grid,
    #     timing_flag=TIMING_FLAG,
    #     timestamp=timestamp,
    #     save_dir=SAVE_DIR / folder_name
    # )

    final_message = run_gnm_full(
        conn_by_density, dist,
        hparam_grid=hparam_grid,
        timing_flag=TIMING_FLAG,
        timestamp=timestamp,
        save_dir=SAVE_DIR / folder_name,
        search_mode=search_mode,
        random_sample_size=random_sample_size,
        random_seed=42
    )
    
    print(final_message)
