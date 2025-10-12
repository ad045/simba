from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.preprocessing_setup import setup


# Utility helpers

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_numpy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"Saved: {path}")


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")




def setup_paths(dataset_name) -> dict:
    env = setup(dataset_name=dataset_name)
    
    path_raw_data = env["DATA_PATH"] / "raw" / dataset_name
    
    path_output_for_logs_and_plots = env["OUTPUT_PATH"] / "00_preprocessing" / dataset_name # for plots and logs 
    path_preprocessed = env["PREPROCESSED_PATH"] #  / dataset_name 
    
    path_00_preprocessed = path_preprocessed / "00_preprocessed" # / dataset_name
    path_01_connectomes = path_preprocessed / "01_connectomes"  # / dataset_name
    path_02_distance_matrices = path_preprocessed / "02_distance_matrices"  # / dataset_name
    path_03_graph_measures = path_preprocessed / "03_graph_measures"  # / dataset_name
    path_04_further_info = path_preprocessed / "04_further_info" # / dataset_name
    

    # INPUT_DATA_PATH: Path = env["PREPROCESSED_PATH"] / dataset_name / "00_preprocessed"


    for dir_path in [path_00_preprocessed, path_01_connectomes, path_02_distance_matrices, 
                     path_03_graph_measures, path_04_further_info, path_output_for_logs_and_plots,
                     path_raw_data]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return {
        "path_raw_data": path_raw_data, 
        "path_00_preprocessed": path_00_preprocessed,
        "path_01_connectomes": path_01_connectomes,
        "path_02_distance_matrices": path_02_distance_matrices,
        "path_03_graph_measures": path_03_graph_measures,
        "path_04_further_info": path_04_further_info,
        "path_output_for_logs_and_plots": path_output_for_logs_and_plots
    }


