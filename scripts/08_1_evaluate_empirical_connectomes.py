import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys


try:
    from src.structural_analysis.graph_measures import analyze_connectomes
    from ESNs.esn_evaluation import evaluate_memory_capacity_from_connectome
except ImportError as e:
    print(f"❌ Error importing project modules: {e}", file=sys.stderr)
    print("👉 Please ensure you run this script from your project's root directory.", file=sys.stderr)
    sys.exit(1)

def analyze_empirical_connectomes(connectomes_path: str, distance_matrix_path: str, output_csv_path: str):
    """
    Loads empirical connectomes, calculates graph and MC metrics for each,
    and saves the combined results to a single CSV file.

    Args:
        connectomes_path (str): Path to the .npy file with connectomes (shape: subjects, nodes, nodes).
        distance_matrix_path (str): Path to the .npy file for the distance matrix (shape: nodes, nodes).
        output_csv_path (str): Path to save the resulting CSV file.
    """
    # 1. Load data from specified paths
    print(f"🧠 Loading connectomes from: {connectomes_path}")
    connectomes = np.load(connectomes_path)
    print(f"📏 Loading distance matrix from: {distance_matrix_path}")
    distance_matrix = np.load(distance_matrix_path)
    
    num_subjects, num_nodes, _ = connectomes.shape
    print(f"Found {num_subjects} connectomes with {num_nodes} nodes each.")

    # 2. Define default ESN hyperparameters # TODO: Change this here! 
    # These are based on your original script. Adjust them as needed.
    h_params = {
        "spectral_radius": 0.9,
        "n_lags": 50,
        "train_len": 5000, # ?? Is this input lengths?? 2000,
        "test_len": 1000,
        "n_runs": 50,
        "input_scaling": 1.0, # 0.1,
        "regression_method": "pinv", # "ridge",
        "n_transient": 100, # a????
        "leak_rate": 1.0, # 0.1,
        "bias": 0.0, # True,
        "random_state": 42,
    }
    
    all_subject_results = []

    # 3. Process each connectome individually
    for i in tqdm(range(num_subjects), desc="Analyzing individual connectomes"):
        subject_connectome = connectomes[i, :, :]
        # Start a record for the current subject
        record = {'subject_id': i}

        # --- Calculate graph measures ---
        try:
            # Note: analyze_connectomes expects a list of connectomes, so we wrap the single one
            graph_measures_list = analyze_connectomes(
                connectomes=[subject_connectome], 
                distance_matrix=distance_matrix
            )
            # The function returns a list of dicts; we take the first element
            record.update(graph_measures_list[0])
        except Exception as e:
            print(f"⚠️ Warning: Could not calculate graph measures for subject {i}: {e}")

        # --- Calculate ESN (Memory Capacity) measures ---
        try:
            esn_results = evaluate_memory_capacity_from_connectome(
                connectome=np.array(subject_connectome, dtype=np.float32), 
                h_params=h_params
            )
            

            # Flatten the potentially nested dictionary from ESN results for easy CSV export
            flat_esn_results = pd.json_normalize(esn_results, sep='_').to_dict(orient='records')[0]
        
            # flat_esn_results = {}
            for i, mc_lag_val in enumerate(esn_results["mc_values_for_indiv_lags"]): 
                # print(i, mc_lag_val)
                flat_esn_results.update({"mc_lag_" + str(i): mc_lag_val})
                
            flat_esn_results.pop("mc_values_for_indiv_lags")
                
            
            
            # indiv_energy_values = experiment.evaluation_results.binary_evaluations[energy_metric_name].numpy().flatten()
            # for i in range(individual_networks.shape[0]):
            #     indiv_networks_record.update(
            #         {energy_metric_name + "_indiv_" + str(i): indiv_energy_values[i]}
            #     )
                    
                    
            record.update(flat_esn_results)
        except Exception as e:
            print(f"⚠️ Warning: Could not calculate ESN measures for subject {i}: {e}")
            record.update({"mc_mean": np.nan}) # Add a placeholder on failure

        all_subject_results.append(record)
        
    # 4. Combine all results into a DataFrame and save to CSV
    if not all_subject_results:
        print("❌ No results were generated. Exiting.")
        return
        
    results_df = pd.DataFrame(all_subject_results)
    
    # Ensure the output directory exists before saving
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_path, index=False, na_rep="nan")
    print(f"\n✅ Analysis complete. Results for {len(results_df)} connectomes saved to: {output_path}")


def main():
    """Defines the command-line interface for the analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze empirical connectomes for graph and memory capacity (MC) metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "connectomes_path", 
        type=str,
        help="Path to the input .npy file containing empirical connectomes."
    )
    parser.add_argument(
        "distance_matrix_path", 
        type=str,
        help="Path to the .npy file containing the corresponding distance matrix."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str,
        required=True,
        help="Path for the output CSV file."
    )
    
    args = parser.parse_args()
    
    analyze_empirical_connectomes(
        connectomes_path=args.connectomes_path,
        distance_matrix_path=args.distance_matrix_path,
        output_csv_path=args.output
    )


if __name__ == "__main__":
    # main()
    
    # File paths
    connectomes_file = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/01_first_analysises/connectomes_binarized_68x68_density_10_percent.npy" 
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/01_first_analysises/connectomes_weighted_68x68.npy"
    distance_matrix_file = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/01_first_analysises/distance_matrix_68x68.npy"
    
    output_base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/emprirical_analysis")
    output_base_path.mkdir(parents=True, exist_ok=True)
    output_csv_file = output_base_path / "empirical_analysis.csv"
    
    # Run analysis function
    analyze_empirical_connectomes(
        connectomes_path=connectomes_file,
        distance_matrix_path=distance_matrix_file,
        output_csv_path=output_csv_file
    )