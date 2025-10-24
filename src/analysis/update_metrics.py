"""
Main script to compute missing metrics and update CSV files.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from src.analysis.structural_metrics import compute_structural_metrics
# from dynamics_metrics import compute_dynamics_metrics
# from computation_metrics import compute_computation_metrics


def load_network(network_path):
    """Load a network from .npy file."""
    return np.load(network_path)


def get_network_path_from_row(row, base_dir):
    """
    Construct the network path from CSV row information.
    
    Expected format:
    base_dir/generated_networks/net_eta-{eta}_gamma{gamma}_rule{generative_rule}.npy
    """
    eta = row['eta']
    gamma = row['gamma']
    rule = row['generative_rule']
    
    network_filename = f"net_eta{eta}_gamma{gamma}_rule{rule}.npy"
    network_path = Path(base_dir) / network_filename
    
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/61_testing_with_kayson_animal_0/generated_networks/net_eta0.2522252798080444_gamma0.9284507036209106_ruleMatchingIndex.npy
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/61_testing_with_kayson_animal_0/generated_networks/net_eta1.9657100439071655_gamma0.9114527702331543_ruleMatchingIndex.npy
    
    return network_path

### SHOULD NOT BE NECESSARY; BUT HERE WE GO: ####

def get_eta_and_gamma_from_filename(filename):
    """
    Extract eta and gamma values from the network filename.
    
    Expected filename format:
    net_eta-{eta}_gamma{gamma}_rule{generative_rule}.npy
    """
    import re
    
    eta_match = re.search(r'eta([-+]?\d*\.\d+|\d+)', filename)
    gamma_match = re.search(r'gamma([-+]?\d*\.\d+|\d+)', filename)
    
    eta = float(eta_match.group(1)) if eta_match else None
    gamma = float(gamma_match.group(1)) if gamma_match else None
    
    return eta, gamma
#######################################################################


from scripts.utils_12_file_handling import (get_missing_metrics)



def compute_missing_metrics(network, distance_matrix, metrics_to_compute, 
                           metric_category='structural'):
    """
    Compute requested metrics for a single network.
    
    Args:
        network: numpy array (n, n)
        distance_matrix: numpy array (n, n)
        metrics_to_compute: list of metric names
        metric_category: 'structural', 'dynamics', or 'computation'
    
    Returns:
        dict: {metric_name: value}
    """
    if metric_category == 'structural':
        return compute_structural_metrics(network, distance_matrix, metrics_to_compute)
    # elif metric_category == 'dynamics':
    #     return compute_dynamics_metrics(network, distance_matrix, metrics_to_compute)
    # elif metric_category == 'computation':
    #     return compute_computation_metrics(network, distance_matrix, metrics_to_compute)
    else:
        raise ValueError(f"Unknown metric category: {metric_category}")


def update_csv_with_metrics(csv_path, 
                            generated_networks_dir, 
                            distance_matrix, 
                            requested_metrics, 
                            metric_categories
                            ):
    """
    Update CSV file with missing metrics.
    
    Args:
        csv_path: Path to CSV file
        base_dir: Base directory containing generated_networks folder
        distance_matrix: (n, n) Euclidean distance matrix
        requested_metrics: dict {category: [metric_names]}
        metric_categories: dict {metric_name: category}
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    # Flatten requested metrics
    all_requested = []
    for metrics_list in requested_metrics.values():
        all_requested.extend(metrics_list)
    
    # Find missing metrics
    missing_by_row = get_missing_metrics(df, all_requested)
    
    if not missing_by_row:
        print("No missing metrics found. CSV is up to date.")
        return
    
    print(f"Found missing metrics in {len(missing_by_row)} rows")
    
    # Process each row with missing metrics
    for row_idx, missing_metrics in missing_by_row.items():
        print(f"\nProcessing row {row_idx}...")
        print(f"  Missing metrics: {missing_metrics}")
        
        # Load network
        network_path = get_network_path_from_row(df.iloc[row_idx], generated_networks_dir)
        
        if not network_path.exists():
            print(f"  Warning: Network file not found: {network_path}")
            continue
        
        network = load_network(network_path)
        print(f"  Loaded network: {network.shape}")
        
        # Group missing metrics by category
        metrics_by_category = {}
        for metric in missing_metrics:
            category = metric_categories.get(metric, 'structural')
            if category not in metrics_by_category:
                metrics_by_category[category] = []
            metrics_by_category[category].append(metric)
        
        # Compute metrics for each category
        for category, metrics in metrics_by_category.items():
            print(f"  Computing {len(metrics)} {category} metrics...")
            try:
                results = compute_missing_metrics(
                    network, distance_matrix, metrics, category
                )
                
                # Update DataFrame
                for metric, value in results.items():
                    df.at[row_idx, metric] = value
                    
            except Exception as e:
                print(f"  Error computing {category} metrics: {e}")
                import traceback
                traceback.print_exc()
    
    # Save updated CSV
    output_path = csv_path.replace('.csv', '_updated.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved updated CSV to: {output_path}")


# if __name__ == "__main__":
    # SEE example_metric_update_pipeline.py FOR EXECUTION SCRIPT
    # ========== CONFIGURATION ==========
    
    # # Define which metrics to compute (organized by category)
    # REQUESTED_METRICS = {
    #     'structural': [
    #         # 'small_world_omega',
    #         # 'structural_complexity',
    #         # 'hubness_gini',
    #         # 'degree_assortativity',
    #         # 'richclub_coefficient',
    #         # 'richclub_k_max',
    #     ],
    #     'dynamics': [
    #         # 'spectral_radius',
    #         # 'spectral_gap',
    #         # 'eigenspectrum_entropy',
    #         # 'diffusion_efficiency',
    #     ],
    #     'computation': [
    #         # Add computation metrics here if needed
    #     ]
    # }
    
    # # Create mapping of metric -> category
    # METRIC_CATEGORIES = {}
    # for category, metrics in REQUESTED_METRICS.items():
    #     for metric in metrics:
    #         METRIC_CATEGORIES[metric] = category
    
    # # Paths
    # ANIMAL_ID = "70_mix_and_match_animal_0"
    # BASE_DIR = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/{ANIMAL_ID}"
    # CSV_PATH = f"{BASE_DIR}/all_metrics_for_{ANIMAL_ID}.csv"
    
    # # Load distance matrix (you'll need to provide the correct path)
    # DISTANCE_MATRIX_PATH = "/path/to/distance_matrix.npy"  # UPDATE THIS
    
    # # ========== EXECUTION ==========
    
    # print("Starting metric computation pipeline...")
    # print(f"Animal ID: {ANIMAL_ID}")
    # print(f"CSV path: {CSV_PATH}")
    
    # # Load distance matrix
    # if Path(DISTANCE_MATRIX_PATH).exists():
    #     distance_matrix = np.load(DISTANCE_MATRIX_PATH)
    #     print(f"Loaded distance matrix: {distance_matrix.shape}")
    # else:
    #     print(f"Warning: Distance matrix not found at {DISTANCE_MATRIX_PATH}")
    #     print("Creating dummy distance matrix for testing...")
    #     # Create a dummy distance matrix (replace with actual loading)
    #     distance_matrix = np.random.rand(100, 100)  # Adjust size as needed
    
    # # Update CSV
    # update_csv_with_metrics(
    #     CSV_PATH, 
    #     BASE_DIR, 
    #     distance_matrix,
    #     REQUESTED_METRICS,
    #     METRIC_CATEGORIES
    # )
    
    # print("\nDone!")