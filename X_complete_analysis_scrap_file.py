"""
Example of how to integrate portrait divergence into your existing pipeline.
This shows how to modify _run_and_save_single_simulation to include portrait divergence.
"""

from pathlib import Path
import numpy as np
import torch
from portrait_divergence import portrait_divergence
import networkx as nx


def calculate_portrait_in_pipeline(
    generated_network: np.ndarray,
    empirical_network: np.ndarray
) -> float:
    """
    Calculate portrait divergence within the pipeline.
    Designed to be called from _run_and_save_single_simulation.
    
    Args:
        generated_network: Generated adjacency matrix (n_nodes, n_nodes)
        empirical_network: Empirical adjacency matrix (n_nodes, n_nodes)
        
    Returns:
        Portrait divergence value
    """
    try:
        # Ensure 2D binary networks
        if generated_network.ndim == 3:
            generated_network = generated_network[0]
        if empirical_network.ndim == 3:
            empirical_network = empirical_network[0]
        
        # Make binary and symmetric
        gen = (generated_network > 0).astype(float)
        emp = (empirical_network > 0).astype(float)
        gen = np.maximum(gen, gen.T)
        emp = np.maximum(emp, emp.T)
        
        # Convert to NetworkX graphs
        G_gen = nx.from_numpy_array(gen)
        G_emp = nx.from_numpy_array(emp)
        
        # Calculate divergence
        return float(portrait_divergence(G_gen, G_emp))
    except Exception as e:
        print(f"Error calculating portrait divergence: {e}")
        return np.nan


# Example modification to your _run_and_save_single_simulation function
def _run_and_save_single_simulation_with_portrait(
    task_data: dict,
    evaluation_criteria,
    target_network: np.ndarray,
    individual_networks: np.ndarray,
    compare_to_connectome_of_distance_matrix: bool,
    elaborate_analysis: bool,
    compare_to_all_individual_empirical_connectomes: bool,
    device_str: str,
    output_dir: Path,
    temp_dir: str,
    h_params: dict,
    animal_id: int,  # Add this parameter
    calculate_portrait: bool = True  # Flag to enable/disable portrait calculation
):
    """
    Modified version of your worker function that includes portrait divergence.
    
    Add this to your existing function where you calculate elaborate_analysis metrics.
    """
    # ... [Your existing code for performing the run] ...
    
    flat_record = {}
    
    # ... [Your existing code] ...
    
    # Add portrait divergence calculation if elaborate_analysis is True
    if elaborate_analysis and calculate_portrait:
        try:
            # Get the generated network
            networks_np = experiment.model.adjacency_matrix.cpu().numpy()
            
            # Calculate portrait divergence against the empirical network
            if compare_to_connectome_of_distance_matrix:
                # Get the appropriate empirical network
                empirical_network = individual_networks[animal_id]
                
                # Calculate portrait divergence for the first generated network
                # (if multiple simulations, take the mean)
                if networks_np.ndim == 3:
                    portrait_divs = []
                    for i in range(networks_np.shape[0]):
                        pd_val = calculate_portrait_in_pipeline(
                            networks_np[i],
                            empirical_network
                        )
                        portrait_divs.append(pd_val)
                    flat_record['portrait_divergence'] = np.mean(portrait_divs)
                else:
                    flat_record['portrait_divergence'] = calculate_portrait_in_pipeline(
                        networks_np,
                        empirical_network
                    )
                
                print(f"Portrait divergence: {flat_record['portrait_divergence']:.4f}")
        except Exception as e:
            print(f"Error adding portrait divergence: {e}")
            flat_record['portrait_divergence'] = np.nan
    
    # ... [Rest of your existing code] ...
    
    return flat_record


# ==============================================================================
# USAGE EXAMPLE: Running the compute_missing_metrics script
# ==============================================================================

"""
Command-line usage examples:

1. Calculate portrait divergence for a specific experiment:

python compute_missing_metrics.py \
    /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/all_metrics_for_60_generally_finer_search_animal_0.csv \
    --metrics portrait_divergence

2. Calculate multiple metrics:

python compute_missing_metrics.py \
    /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/all_metrics_for_60_generally_finer_search_animal_0.csv \
    --metrics portrait_divergence global_efficiency modularity

3. Use more workers for faster processing:

python compute_missing_metrics.py \
    /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/all_metrics_for_60_generally_finer_search_animal_0.csv \
    --metrics portrait_divergence \
    --workers 16

4. Specify custom data paths:

python compute_missing_metrics.py \
    /path/to/results.csv \
    --metrics portrait_divergence \
    --empirical-path /path/to/empirical_networks.npy \
    --distance-path /path/to/distance_matrix.npy

5. Calculate all available graph measures:

python compute_missing_metrics.py \
    /path/to/results.csv \
    --metrics global_efficiency modularity avg_clustering avg_degree density_bct transitivity avg_edge_distance wiring_cost char_path_length richclub_n_edges richclub_avg_length avg_communicability
"""


# ==============================================================================
# PROGRAMMATIC USAGE EXAMPLE
# ==============================================================================

def example_programmatic_usage():
    """
    Example of calling compute_missing_metrics programmatically 
    (not from command line).
    """
    from compute_missing_metrics import compute_missing_metrics
    
    # Define paths
    csv_path = Path(
        "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/"
        "suarez_MaMI_dataset/60_generally_finer_search_animal_0/"
        "all_metrics_for_60_generally_finer_search_animal_0.csv"
    )
    
    # Metrics you want to calculate
    metrics = ['portrait_divergence', 'global_efficiency', 'modularity']
    
    # Config for MaxCrit (if needed)
    config_dict = {
        'gnm': {
            'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
        }
    }
    
    # Run computation
    df_updated = compute_missing_metrics(
        csv_path=csv_path,
        metrics_to_calculate=metrics,
        empirical_networks_path=None,  # Auto-detect
        distance_matrix_path=None,  # Auto-detect
        config_dict=config_dict,
        n_workers=8,
        save_every=10
    )
    
    print(f"\nUpdated {len(df_updated)} rows")
    print(f"Columns: {df_updated.columns.tolist()}")
    
    return df_updated


# ==============================================================================
# BATCH PROCESSING MULTIPLE EXPERIMENTS
# ==============================================================================

def batch_process_multiple_experiments():
    """
    Process multiple experiments/animals in batch.
    """
    from compute_missing_metrics import compute_missing_metrics
    
    base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset")
    
    # List of experiments to process
    experiments = [
        "60_generally_finer_search_animal_0",
        "60_generally_finer_search_animal_1",
        "60_generally_finer_search_animal_2",
        # ... add more experiments
    ]
    
    metrics = ['portrait_divergence']
    
    config_dict = {
        'gnm': {
            'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
        }
    }
    
    for exp_name in experiments:
        print(f"\n{'='*80}")
        print(f"Processing: {exp_name}")
        print(f"{'='*80}")
        
        csv_path = base_path / exp_name / f"all_metrics_for_{exp_name}.csv"
        
        if not csv_path.exists():
            print(f"⚠️  CSV not found, skipping: {csv_path}")
            continue
        
        try:
            df = compute_missing_metrics(
                csv_path=csv_path,
                metrics_to_calculate=metrics,
                config_dict=config_dict,
                n_workers=8,
                save_every=10
            )
            print(f"✅ Completed {exp_name}")
        except Exception as e:
            print(f"❌ Error processing {exp_name}: {e}")
            continue


# ==============================================================================
# QUICK TEST FUNCTION
# ==============================================================================

def quick_test_portrait_divergence():
    """
    Quick test to verify portrait divergence is working correctly.
    """
    import networkx as nx
    from portrait_divergence import portrait_divergence
    
    # Create two similar random graphs
    np.random.seed(42)
    n = 50
    
    # Generate random adjacency matrices
    A1 = np.random.rand(n, n) > 0.9
    A1 = A1.astype(float)
    A1 = np.maximum(A1, A1.T)  # Make symmetric
    np.fill_diagonal(A1, 0)
    
    A2 = np.random.rand(n, n) > 0.9
    A2 = A2.astype(float)
    A2 = np.maximum(A2, A2.T)
    np.fill_diagonal(A2, 0)
    
    # Calculate portrait divergence
    G1 = nx.from_numpy_array(A1)
    G2 = nx.from_numpy_array(A2)
    
    div = portrait_divergence(G1, G2)
    print(f"Portrait divergence between two random graphs: {div:.6f}")
    
    # Test identical graphs (should be ~0)
    div_same = portrait_divergence(G1, G1)
    print(f"Portrait divergence of graph with itself: {div_same:.6f}")
    
    return div, div_same


if __name__ == "__main__":
    print("Portrait Divergence Integration Examples")
    print("="*80)
    print("\nRun this script to see usage examples.")
    print("\nFor actual computation, use:")
    print("  python compute_missing_metrics.py <csv_path> --metrics portrait_divergence")
    print("\nOr import and use programmatically:")
    print("  from compute_missing_metrics import compute_missing_metrics")
    
    # Uncomment to run quick test
    # quick_test_portrait_divergence()