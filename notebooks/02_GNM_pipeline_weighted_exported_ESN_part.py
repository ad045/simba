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



def _subject_job(subj_idx: int,
                 A_obs: np.ndarray,
                 dist: np.ndarray,
                 esn_use_observed_real_weights: bool,
                 timing_flag: bool = True
                 ):
    """
    Subject-level job for processing a single subject's connectome.
    """
    
    # Start timer
    if timing_flag: 
        t0 = time.perf_counter()
        t_metrics = time.perf_counter() - t0 

    # Graph measures on observed weights (i.e.: not simulated) 
    #  = analyze_connectomes(connectomes=np.expand_dims(A_obs, 2),
    #                                distance_matrix=dist,
    #                                comm_mode='estrada_scaled',
    #                                use_weighted=True,
    #                                treat_weights_as_lengths=False,
    #                                symmetrize="max", 
    #                                min_weight=0.0)
    
    #     #     out.append({
    #     #     "network_index": idx,
    #     #     "avg_communicability": avg_comm,
    #     #     "global_efficiency": glob_eff,
    #     #     "modularity": modu,
    #     #     "avg_clustering": avg_clust,
    #     #     "avg_degree_or_strength": avg_deg_or_strength,
    #     #     "transitivity": trans,
    #     #     "avg_edge_distance": avg_dist,
    #     #     "char_path_length": cpl,
    #     #     "richclub_n_edges": n_rich_edges,
    #     #     "richclub_avg_length": avg_rc_length,
    #     #     "used_weighted_graph": bool(use_weighted),
    #     #     "treat_weights_as_lengths": bool(treat_weights_as_lengths),
    #     #     "min_weight_threshold": float(min_weight),
    #     #     "symmetrize": symmetrize,
        # })


    # Memory capacity 
    if timing_flag: 
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
          
    if timing_flag:  
        t_esn = time.perf_counter() - t_esn0
        
        timing = {"subject": subj_idx, 
                "time_metrics_sec": t_metrics, 
                "time_esn_sec": t_esn, 
                "time_total_sec": t_metrics + t_esn}
    else: 
        timing = np.nan

    return mc_result_dict, timing


def run_gnm_full(conn: np.ndarray,
                 dist: np.ndarray,
                 timing_flag: bool = False, 
                 timestamp: str = None,
                 save_dir: str = None):
    """
    Run the GNM pipeline for all subjects in parallel. 
    """
    
    # Paths 
    mc_values_csv_path = save_dir / f"gnm_mc_results_{timestamp}.csv"
    duration_csv_path = save_dir / f"gnm_durations_{timestamp}.csv"
        
    # Start timer 
    if timing_flag:
        run_t0 = time.perf_counter()

    n_subj = conn.shape[2]
    tasks = [ (i, conn[:,:,i], dist, timing_flag) for i in range(n_subj) ]

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
        
    mc_values = []
    timing_list = []

    rows_to_save_mc = ["subject", "mc_mean", "mc_std", "mean_mc_of_individual_runs", "hparams"]
    rows_to_save_durations = ["subject", "time_metrics_sec", "time_esn_sec", "time_total_sec"]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(_subject_job, *t) for t in tasks]

        for idx, fut in enumerate(as_completed(futures), 1):
            mc_result_dict, timing_dict = fut.result()
            mc_values.append(mc_result_dict)
            if timing_flag:
                timing_list.append(timing_dict)

            values_to_save_mc = [timing_dict["subject"]] + [mc_result_dict[row_name] for row_name in rows_to_save_mc[1:]]
            _append_rows_csv([values_to_save_mc],
                             rows_to_save_mc,
                             mc_values_csv_path)

            # Save timings for this subject
            if timing_flag:
                values_to_save_durations = [timing_dict[row_name] for row_name in rows_to_save_durations]
                _append_rows_csv([values_to_save_durations],
                                rows_to_save_durations,
                                duration_csv_path)

            # Print exactly when rows are flushed
            print(f"Finished now {idx}/{n_subj} subjects.") 

    if timing_flag:
        total_sec = time.perf_counter() - run_t0
        print(f"Total run time: {total_sec:.3f}s")
        print(f"Average time per subject: {total_sec / n_subj:.3f}s")
    
        # timing footer: put the total in 'total_s'
        _append_rows_csv(
            [("COMPLETED", np.nan, np.nan, f"{total_sec:.3f}")],
            ["subject","time_metrics_sec","time_esn_sec","time_total_sec"],
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
    DENSITY = 10
    
    TIMING_FLAG = True

    # load data
    conn = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/connectomes_weighted_{RESOLUTION}x{RESOLUTION}.npy")).T
    dist = np.load(os.path.join(ROOT, f"data/preprocessed/01_first_analysises/distance_matrix_{RESOLUTION}x{RESOLUTION}.npy"))

    # timestamp for saving
    timestamp = time_stamp_for_saving()
    
    # create save directories
    folder_name = f"esn_resolution{RESOLUTION}_{timestamp}"
    (SAVE_DIR / folder_name).mkdir(parents=True, exist_ok=True)

    # run 
    final_message = run_gnm_full(
        conn, dist,
        timing_flag=TIMING_FLAG,
        timestamp=timestamp,
        save_dir=SAVE_DIR / folder_name
    )
    print(final_message)
