from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import shutil
import pandas as pd
import json

from preprocessing.threshold_to_density import threshold_to_density
from preprocessing.preprocessing_setup import setup

from netneurotools.networks import struct_consensus
from preprocessing.threshold_to_density import threshold_to_density

from src.structural_analysis.graph_measures import analyze_connectomes

from src.preprocessing.utils import setup_paths, save_dataframe


def _load_mat_connectomes(mat_path: Path) -> Tuple[np.ndarray, int, int, int]:
    """
    Load the Griffa 70-subject connectomes from a MATLAB v7.3+ file using h5py,
    normalize to shape (subjects, nodes, nodes), and ensure clean diagonals.
    Returns: (all_connectomes, num_subjects, num_nodes, num_nodes)
    """
    import h5py

    with h5py.File(mat_path, "r") as f:
        raw = np.array(f["M"])  # underlying layout may be (N, N, S) or (S, N, N)

    # Normalize to (subjects, nodes, nodes)
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {raw.shape}")

    # Heuristic: pick the axis that equals number of subjects (commonly 70)
    # Fallback: pick the axis whose size differs from the other two equal dims
    shape = raw.shape
    if shape[0] != shape[1] and shape[1] == shape[2]:
        # (S, N, N)
        arr = raw
    elif shape[1] != shape[2] and shape[0] == shape[2]:
        # (N, S, N) -> move axis 1 to front
        arr = np.moveaxis(raw, 1, 0)
    elif shape[0] == shape[1] and shape[2] != shape[1]:
        # (N, N, S) -> move axis 2 to front
        arr = np.moveaxis(raw, 2, 0)
    else:
        # If ambiguous, try to interpret as (S, N, N)
        arr = raw

    # Now enforce square last two dims
    if arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Last two dimensions must be equal, got {arr.shape}")

    num_subjects, num_nodes, _ = arr.shape

    # Zero diagonals defensively for each subject
    for s in range(num_subjects):
        np.fill_diagonal(arr[s], 0.0)

    # Sanity checks
    assert np.allclose(arr, arr.transpose(0, 2, 1)), "Connectomes must be symmetric per subject"
    assert np.all(np.diag(arr[0]) == 0.0), "Diagonal is not zero after cleanup"

    return arr, num_subjects, num_nodes, num_nodes


def _save_numpy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"Saved: {path}")


def setup_paths(dataset_name: str) -> dict:
    env = setup(dataset_name=dataset_name)

    path_raw_data = env["DATA_PATH"] / "raw" / dataset_name
    path_output_for_logs_and_plots = env["OUTPUT_PATH"] / "00_preprocessing" / dataset_name
    path_preprocessed = env["PREPROCESSED_PATH"]

    path_00_preprocessed = path_preprocessed / "00_preprocessed"
    path_01_connectomes = path_preprocessed / "01_connectomes"
    path_02_distance_matrices = path_preprocessed / "02_distance_matrices"
    path_03_graph_measures = path_preprocessed / "03_graph_measures"
    path_04_further_info = path_preprocessed / "04_further_info"

    for dir_path in [
        path_00_preprocessed,
        path_01_connectomes,
        path_02_distance_matrices,
        path_03_graph_measures,
        path_04_further_info,
        path_output_for_logs_and_plots,
        path_raw_data,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return {
        "path_raw_data": path_raw_data,
        "path_00_preprocessed": path_00_preprocessed,
        "path_01_connectomes": path_01_connectomes,
        "path_02_distance_matrices": path_02_distance_matrices,
        "path_03_graph_measures": path_03_graph_measures,
        "path_04_further_info": path_04_further_info,
        "path_output_for_logs_and_plots": path_output_for_logs_and_plots,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Griffa 70-subject connectomes")
    # p.add_argument(
    #     "--mat-path",
    #     type=Path,
    #     default=Path(
    #         "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/griffa_70_human_connectomes_dataset/00_preprocessed/SC_68.mat"
    #     ),
    #     help="Absolute path to the MATLAB file containing the connectomes",
    # )
    p.add_argument(
        "--analyze-density",
        type=int,
        default=10,
        help="Density to threshold the consensus connectome to",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=68,
        help="Resolution of the parcellation",
    )
    return p.parse_args()


def main(resolution=None) -> None:
    args = parse_args()
    # Possibility to also enter resolution in __name__== "__main__" part
    if resolution: 
        args.resolution = resolution
    
    dataset_name = "griffa_70_human_connectomes_dataset"
    paths = setup_paths(dataset_name=dataset_name)

    # Load and normalize connectomes to (subjects, nodes, nodes)
    mat_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / f"00_preprocessed/SC_{args.resolution}.mat"
    all_connectomes, num_subjects, resolution, _ = _load_mat_connectomes(mat_path)
    print(f"Loaded {num_subjects} subjects with {resolution} nodes each. Array shape: {all_connectomes.shape}")

    # Save to the same structure as Shafiei pipeline uses
    out_all = paths["path_01_connectomes"] / f"00_connectomes_{args.resolution}.npy"
    _save_numpy(out_all, all_connectomes)

    # Copy distance matrix from one folder to the other (this is the one from Shafiei) 
    distance_matrix_path = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/shafiei_human_consensus_dataset/02_distance_matrices/distance_matrix_{args.resolution}.npy" # 
    # paths["path_02_distance_matrices"] / f"distance_matrix_{n}.npy"
    shutil.copy(distance_matrix_path, paths["path_02_distance_matrices"] / f"distance_matrix_{args.resolution}.npy")

    # Copy identifiers from one folder to the other
    identifiers_path = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/shafiei_human_consensus_dataset/04_further_info/df_identifiers_{args.resolution}.csv"
    shutil.copy(identifiers_path, paths["path_04_further_info"] / f"df_identifiers_{args.resolution}.csv")
    # Extract hemi_id from identifiers (csv)
    identifiers = pd.read_csv(identifiers_path) 
    hemi_id = identifiers["hemi_id"]

    # Calculate consensus with netneurotools (weighted) 
    dist_mat = np.load(paths["path_02_distance_matrices"] / f"distance_matrix_{resolution}.npy")
    # hemi_id = np.load(paths["path_04_further_info"] / f"hemi_id_{resolution}.npy")
    consensus_weighted = struct_consensus(all_connectomes.T, distance=dist_mat, hemiid=hemi_id.to_numpy().reshape(-1, 1), weighted=True)
    out_cons = paths["path_01_connectomes"] / f"01_consensus_wei_{resolution}.npy"
    _save_numpy(out_cons, consensus_weighted)


    thres_conn, final_density = threshold_to_density(consensus_wei=consensus_weighted, 
                                                    n_nodes=resolution, 
                                                    density=args.analyze_density, 
                                                    output_folder=paths["path_01_connectomes"])
                                                    # onsensus_bin_density_{density}_percent_{n}.npy"

    print(f"Thresholded connectome shape: {thres_conn.shape}")
    print(f"Final density: {final_density[0]*100:.2f}%")


    # Analyze the binarized and thresholded connectome
    df_graph_measures = pd.DataFrame(
        analyze_connectomes(
            connectomes=thres_conn,
            distance_matrix=dist_mat,
        )
    )
    analysis_path = paths["path_03_graph_measures"] / f"df_graph_measures_bin_{resolution}_density_{args.analyze_density}_percent.csv"
    save_dataframe(
        analysis_path,
        df_graph_measures,
    )

    # Analyze the weighted connectome
    df_graph_measures = pd.DataFrame(
        analyze_connectomes(
            connectomes=consensus_weighted,
            distance_matrix=dist_mat,
        )
    )
    analysis_path = paths["path_03_graph_measures"] / f"df_graph_measures_wei_{resolution}_percent.csv"
    save_dataframe(
        analysis_path,
        df_graph_measures,
    )

    # Analyze all connectomes
    df_graph_measures = pd.DataFrame(
        analyze_connectomes(
            connectomes=all_connectomes,
            distance_matrix=dist_mat,
        )
    )
    analysis_path = paths["path_03_graph_measures"] / f"df_graph_measures_{resolution}.csv"
    save_dataframe(
        analysis_path,
        df_graph_measures,
    )

    # if args.do_plots:
    #     step_plot_consensus(
    #         paths=paths,
    #         consensus_conn_bin=None, # consensus_bin,
    #         consensus_conn_wei=consensus_weighted, # consensus_all,
    #         density_bin=None, # d_bin,
    #         density_all=None, # d_all,
    #         analyze_density=analyze_density,
    #     )

    # Print summary to console
    print("\nPipeline completed successfully. Summary:")
    summary = {
        "resolution": resolution,
        # "goal_densities": args.goal_densities,
        "analyze_density": args.analyze_density,
        "plots": False,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    print(json.dumps(summary, indent=2))

    # Save summary to a text file
    output_file = paths["path_output_for_logs_and_plots"] / f"preprocessing_pipeline_summary_{resolution}_density_{args.analyze_density}_percent.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_file}")
    print(f"Analysis CSV saved to: {analysis_path}")


    print("\nPreprocessing complete.")


if __name__ == "__main__":
    # preprocessing of shafiei must have been run first, as this script steals the distance matrix from there... 
    main(resolution=68) 




# # Cursor version: 


# from __future__ import annotations

# import argparse
# from pathlib import Path
# from typing import Tuple

# import numpy as np
# import shutil
# import pandas as pd
# import json

# from preprocessing.threshold_to_density import threshold_to_density
# from preprocessing.get_distance_matrix import get_distance_matrix_from_coords, get_distance_matrix_from_fiber_lengths
# from preprocessing.preprocessing_setup import setup

# from netneurotools.networks import struct_consensus
# from preprocessing.threshold_to_density import threshold_to_density

# from src.structural_analysis.graph_measures import analyze_connectomes

# from src.preprocessing.utils import setup_paths, ensure_dir, save_numpy, save_dataframe


# def _load_mat_connectomes(mat_path: Path) -> Tuple[np.ndarray, int, int, int]:
#     """
#     Load the Griffa 70-subject connectomes from a MATLAB v7.3+ file using h5py,
#     normalize to shape (subjects, nodes, nodes), and ensure clean diagonals.
#     Returns: (all_connectomes, num_subjects, num_nodes, num_nodes)
#     """
#     import h5py

#     with h5py.File(mat_path, "r") as f:
#         raw = np.array(f["M"])  # underlying layout may be (N, N, S) or (S, N, N)

#     # Normalize to (subjects, nodes, nodes)
#     if raw.ndim != 3:
#         raise ValueError(f"Expected a 3D tensor, got shape {raw.shape}")

#     # Heuristic: pick the axis that equals number of subjects (commonly 70)
#     # Fallback: pick the axis whose size differs from the other two equal dims
#     shape = raw.shape
#     if shape[0] != shape[1] and shape[1] == shape[2]:
#         # (S, N, N)
#         arr = raw
#     elif shape[1] != shape[2] and shape[0] == shape[2]:
#         # (N, S, N) -> move axis 1 to front
#         arr = np.moveaxis(raw, 1, 0)
#     elif shape[0] == shape[1] and shape[2] != shape[1]:
#         # (N, N, S) -> move axis 2 to front
#         arr = np.moveaxis(raw, 2, 0)
#     else:
#         # If ambiguous, try to interpret as (S, N, N)
#         arr = raw

#     # Now enforce square last two dims
#     if arr.shape[1] != arr.shape[2]:
#         raise ValueError(f"Last two dimensions must be equal, got {arr.shape}")

#     num_subjects, num_nodes, _ = arr.shape

#     # Zero diagonals defensively for each subject
#     for s in range(num_subjects):
#         np.fill_diagonal(arr[s], 0.0)

#     # Sanity checks
#     assert np.allclose(arr, arr.transpose(0, 2, 1)), "Connectomes must be symmetric per subject"
#     assert np.all(np.diag(arr[0]) == 0.0), "Diagonal is not zero after cleanup"

#     return arr, num_subjects, num_nodes, num_nodes


# def _save_numpy(path: Path, arr: np.ndarray) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     np.save(path, arr)
#     print(f"Saved: {path}")


# def setup_paths(dataset_name: str) -> dict:
#     env = setup(dataset_name=dataset_name)

#     path_raw_data = env["DATA_PATH"] / "raw" / dataset_name
#     path_output_for_logs_and_plots = env["OUTPUT_PATH"] / "00_preprocessing" / dataset_name
#     path_preprocessed = env["PREPROCESSED_PATH"]

#     path_00_preprocessed = path_preprocessed / "00_preprocessed"
#     path_01_connectomes = path_preprocessed / "01_connectomes"
#     path_02_distance_matrices = path_preprocessed / "02_distance_matrices"
#     path_03_graph_measures = path_preprocessed / "03_graph_measures"
#     path_04_further_info = path_preprocessed / "04_further_info"

#     for dir_path in [
#         path_00_preprocessed,
#         path_01_connectomes,
#         path_02_distance_matrices,
#         path_03_graph_measures,
#         path_04_further_info,
#         path_output_for_logs_and_plots,
#         path_raw_data,
#     ]:
#         dir_path.mkdir(parents=True, exist_ok=True)

#     return {
#         "path_raw_data": path_raw_data,
#         "path_00_preprocessed": path_00_preprocessed,
#         "path_01_connectomes": path_01_connectomes,
#         "path_02_distance_matrices": path_02_distance_matrices,
#         "path_03_graph_measures": path_03_graph_measures,
#         "path_04_further_info": path_04_further_info,
#         "path_output_for_logs_and_plots": path_output_for_logs_and_plots,
#     }


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Preprocess Griffa 70-subject connectomes")
#     # p.add_argument(
#     #     "--mat-path",
#     #     type=Path,
#     #     default=Path(
#     #         "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/griffa_70_human_connectomes_dataset/00_preprocessed/SC_68.mat"
#     #     ),
#     #     help="Absolute path to the MATLAB file containing the connectomes",
#     # )
#     p.add_argument(
#         "--analyze-density",
#         type=int,
#         default=10,
#         help="Density to threshold the consensus connectome to",
#     )
#     p.add_argument(
#         "--resolution",
#         type=int,
#         default=68,
#         help="Resolution of the parcellation",
#     )
#     return p.parse_args()


# def main(resolution=None) -> None:
#     args = parse_args()
#     # Possibility to also enter resolution in __name__== "__main__" part
#     if resolution: 
#         args.resolution = resolution
    
#     dataset_name = "griffa_70_human_connectomes_dataset"
#     paths = setup_paths(dataset_name=dataset_name)

#     # Load and normalize connectomes to (subjects, nodes, nodes)
#     mat_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed") / dataset_name / f"00_preprocessed/SC_{args.resolution}.mat"
#     all_connectomes, num_subjects, resolution, _ = _load_mat_connectomes(mat_path)
#     print(f"Loaded {num_subjects} subjects with {resolution} nodes each. Array shape: {all_connectomes.shape}")

#     # Save to the same structure as Shafiei pipeline uses
#     out_all = paths["path_01_connectomes"] / f"00_connectomes_{args.resolution}.npy"
#     _save_numpy(out_all, all_connectomes)

#     # Copy distance matrix from one folder to the other (this is the one from Shafiei) 
#     distance_matrix_path = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/shafiei_human_consensus_dataset/02_distance_matrices/distance_matrix_{args.resolution}.npy" # 
#     # paths["path_02_distance_matrices"] / f"distance_matrix_{n}.npy"
#     shutil.copy(distance_matrix_path, paths["path_02_distance_matrices"] / f"distance_matrix_{args.resolution}.npy")

#     # Copy identifiers from one folder to the other
#     identifiers_path = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/shafiei_human_consensus_dataset/04_further_info/df_identifiers_{args.resolution}.csv"
#     shutil.copy(identifiers_path, paths["path_04_further_info"] / f"df_identifiers_{args.resolution}.csv")
#     # Extract hemi_id from identifiers (csv)
#     identifiers = pd.read_csv(identifiers_path) 
#     hemi_id = identifiers["hemi_id"]

#     # Calculate consensus with netneurotools (weighted) 
#     dist_mat = np.load(paths["path_02_distance_matrices"] / f"distance_matrix_{resolution}.npy")
#     # hemi_id = np.load(paths["path_04_further_info"] / f"hemi_id_{resolution}.npy")
#     consensus_weighted = struct_consensus(all_connectomes.T, distance=dist_mat, hemiid=hemi_id.to_numpy().reshape(-1, 1), weighted=True)
#     out_cons = paths["path_01_connectomes"] / f"01_consensus_wei_{resolution}.npy"
#     _save_numpy(out_cons, consensus_weighted)


#     thres_conn, final_density = threshold_to_density(consensus_wei=consensus_weighted, 
#                                                     n_nodes=resolution, 
#                                                     density=args.analyze_density, 
#                                                     output_folder=paths["path_01_connectomes"])
#                                                     # onsensus_bin_density_{density}_percent_{n}.npy"

#     print(f"Thresholded connectome shape: {thres_conn.shape}")
#     print(f"Final density: {final_density[0]*100:.2f}%")


#     # Analyze the binarized and thresholded connectome
#     df_graph_measures = pd.DataFrame(
#         analyze_connectomes(
#             connectomes=thres_conn,
#             distance_matrix=dist_mat,
#         )
#     )
#     analysis_path = paths["path_03_graph_measures"] / f"df_graph_measures_bin_{resolution}_density_{args.analyze_density}_percent.csv"
#     save_dataframe(
#         analysis_path,
#         df_graph_measures,
#     )

#     # Analyze the weighted connectome
#     df_graph_measures = pd.DataFrame(
#         analyze_connectomes(
#             connectomes=consensus_weighted,
#             distance_matrix=dist_mat,
#         )
#     )
#     analysis_path = paths["path_03_graph_measures"] / f"df_graph_measures_wei_{resolution}_percent.csv"
#     save_dataframe(
#         analysis_path,
#         df_graph_measures,
#     )

#     # Analyze all connectomes
#     df_graph_measures = pd.DataFrame(
#         analyze_connectomes(
#             connectomes=all_connectomes,
#             distance_matrix=dist_mat,
#         )
#     )
#     analysis_path = paths["path_03_graph_measures"] / f"df_graph_measures_{resolution}.csv"
#     save_dataframe(
#         analysis_path,
#         df_graph_measures,
#     )

#     # if args.do_plots:
#     #     step_plot_consensus(
#     #         paths=paths,
#     #         consensus_conn_bin=None, # consensus_bin,
#     #         consensus_conn_wei=consensus_weighted, # consensus_all,
#     #         density_bin=None, # d_bin,
#     #         density_all=None, # d_all,
#     #         analyze_density=analyze_density,
#     #     )

#     # Print summary to console
#     print("\nPipeline completed successfully. Summary:")
#     summary = {
#         "resolution": resolution,
#         # "goal_densities": args.goal_densities,
#         "analyze_density": args.analyze_density,
#         "plots": False,
#         "paths": {k: str(v) for k, v in paths.items()},
#     }
#     print(json.dumps(summary, indent=2))

#     # Save summary to a text file
#     output_file = paths["path_output_for_logs_and_plots"] / f"preprocessing_pipeline_summary_{resolution}_density_{args.analyze_density}_percent.json"
#     with open(output_file, 'w') as f:
#         json.dump(summary, f, indent=2)

#     print(f"\nSummary saved to: {output_file}")
#     print(f"Analysis CSV saved to: {analysis_path}")


#     print("\nPreprocessing complete.")


# if __name__ == "__main__":

#     # Possibility to add density here as well
#     main(resolution=68) 
#     # main(analyze_density=12)
#     # main(analyze_density=14)
#     # main(analyze_density=16)
#     # main(analyze_density=18)
#     # main(analyze_density=20)




