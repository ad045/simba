import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from gnm.fitting import BinaryEvaluationMetric

def compute_energies_for_all_networks(
    generated_networks_dir: str,
    empirical_connectomes_path: str,
    distance_matrix_path: str,
    output_csv_path: str
):
    """
    Compare all generated networks against all empirical connectomes.
    
    Args:
        generated_networks_dir: Directory containing .npy files of generated networks
        empirical_connectomes_path: Path to empirical connectomes (shape: n_subjects, n_nodes, n_nodes)
        distance_matrix_path: Path to distance matrix
        output_csv_path: Where to save the results CSV
    """
    
    # Load empirical data
    print("Loading empirical connectomes and distance matrix...")
    empirical_connectomes = np.load(empirical_connectomes_path)
    distance_matrix = np.load(distance_matrix_path)
    n_empirical = empirical_connectomes.shape[0]
    
    # Convert to tensors
    empirical_tensor = torch.tensor(empirical_connectomes, dtype=torch.float32)
    distance_tensor = torch.tensor(distance_matrix, dtype=torch.float32)
    
    # Setup evaluation metric
    evaluation_metric = BinaryEvaluationMetric.max_criteria(
        binary_criteria=[
            BinaryEvaluationMetric.degree_ks(distance_tensor),
            BinaryEvaluationMetric.clustering_ks(distance_tensor),
            BinaryEvaluationMetric.edge_length_ks(distance_tensor),
            BinaryEvaluationMetric.betweenness_ks(distance_tensor),
        ]
    )
    
    # Get all generated network files
    gen_networks_path = Path(generated_networks_dir)
    network_files = sorted(gen_networks_path.glob("*.npy"))
    print(f"Found {len(network_files)} generated network files")
    
    # Process each generated network
    all_results = []
    
    for network_file in tqdm(network_files, desc="Processing networks"):
        # Load generated network
        generated_net = np.load(network_file)
        
        # Handle different shapes: (n, n) or (1, n, n) or (k, n, n)
        if generated_net.ndim == 2:
            generated_net = generated_net[np.newaxis, :, :]
        
        # Extract parameters from filename
        filename = network_file.stem
        params = {}
        
        # Parse eta and gamma from filename
        if "eta" in filename and "gamma" in filename:
            parts = filename.split("_")
            for part in parts:
                if part.startswith("eta"):
                    params["eta"] = float(part.replace("eta", ""))
                elif part.startswith("gamma"):
                    params["gamma"] = float(part.replace("gamma", ""))
                elif part.startswith("rule"):
                    params["rule"] = part.replace("rule", "").replace(".npy", "")
        
        # For each simulated network in the file
        for sim_idx in range(generated_net.shape[0]):
            row_data = params.copy()
            row_data["filename"] = network_file.name
            row_data["sim_index"] = sim_idx
            
            # Get single network and convert to tensor
            single_net = torch.tensor(generated_net[sim_idx:sim_idx+1], dtype=torch.float32)
            
            # Compute energy against each empirical network
            for emp_idx in range(n_empirical):
                empirical_single = empirical_tensor[emp_idx:emp_idx+1]
                
                # Compute energy
                energy = evaluation_metric(single_net, empirical_single)
                energy_value = energy.item()
                
                # Add to row
                col_name = f"MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)_indiv_{emp_idx}"
                row_data[col_name] = energy_value
            
            all_results.append(row_data)
    
    # Create DataFrame and save
    print("Creating DataFrame...")
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns: params first, then energies
    param_cols = ["eta", "gamma", "rule", "filename", "sim_index"]
    energy_cols = [col for col in results_df.columns if "MaxCriteria" in col]
    results_df = results_df[[col for col in param_cols if col in results_df.columns] + energy_cols]
    
    # Save
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"✅ Saved results to: {output_path}")
    print(f"   Shape: {results_df.shape}")
    print(f"   Columns: {list(results_df.columns[:5])}... + {len(energy_cols)} energy columns")
    
    return results_df


# Example usage:
if __name__ == "__main__":
    
    from src.preprocessing.utils import setup_paths
    paths = setup_paths(dataset_name="shafiei_human_consensus_dataset") # suarez_MaMI_dataset") 
    
    resolution = 68 #  50 
    
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/00_connectomes_100.npy
    results = compute_energies_for_all_networks(
        generated_networks_dir = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/26_testing_4_KS_folders_why_so_fast/all_generated_networks" # paths[""] "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/26_testing_4_KS_folders_why_so_fast/all_generated_networks",
        empirical_connectomes_path=
        distance_matrix_path=paths["path_02_distance_matrices"] / f"distance_matrix_{resolution}.npy", # "/path/to/distance_matrix.npy",  # Shape: (50, 50)
        output_csv_path="/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/26_testing_4_KS_folders_why_so_fast/individual_network_energies.csv"
    )
    
