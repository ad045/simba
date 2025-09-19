import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


import sys
import os

# Get the absolute path of the script's directory
# script_dir = os.path.dirname(os.path.abspath(__file__))
# # Get the path of the parent directory (the project root)
# project_root = os.path.dirname(script_dir)
# # Add the project root to the Python path
# sys.path.insert(0, project_root)
# sys.path.insert(0, project_root + "/GenerativeNetworkModels/src/gnm")
# print(sys.path)

# from src.ESNs.esn_evaluation import ESNEvaluator
from src.ESNs.alternative_test_memory_capacity_weighted import alternative_evaluate_mc
from src.config.ESN_and_GNM_config import ConfigManager
from src.utils.data_loader import DataLoader


def create_test_matrices(n_nodes: int = 50) -> dict:
    """
    Generates a dictionary of test connectome matrices.
    """
    matrices = {
        "zero_matrix": np.array(np.zeros((n_nodes, n_nodes)), dtype=np.float32),
        "identity_matrix": np.array(np.eye(n_nodes), dtype=np.float32),
        "fully_connected": np.array(np.ones((n_nodes, n_nodes)) - np.eye(n_nodes), dtype=np.float32),
        "random_sparse": None,
        "random_dense": None,
        "ring_lattice": None,
        "disconnected_components": None,
    }

    # Random sparse matrix
    sparse = np.random.rand(n_nodes, n_nodes)
    sparse[sparse < 0.8] = 0
    matrices["random_sparse"] = np.array((sparse + sparse.T) / 2, dtype=np.float32)

    # Random dense matrix
    dense = np.random.rand(n_nodes, n_nodes)
    matrices["random_dense"] = np.array((dense + dense.T) / 2, dtype=np.float32)

    # Ring lattice
    ring = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(1, 3):
            ring[i, (i + j) % n_nodes] = 1
            ring[i, (i - j) % n_nodes] = 1
    matrices["ring_lattice"] = np.array(ring, dtype=np.float32)

    # Disconnected components
    disconnected = np.zeros((n_nodes, n_nodes))
    comp_size = n_nodes // 2
    comp1 = np.random.rand(comp_size, comp_size)
    comp1[comp1 < 0.5] = 0
    disconnected[:comp_size, :comp_size] = (comp1 + comp1.T) / 2
    matrices["disconnected_components"] = np.array(disconnected, dtype=np.float32)

    return matrices


def run_mc_tests():
    """
    Runs MC evaluation on the test matrices and prints the results.
    """
    # 1. Setup a minimal configuration for the evaluator
    # We create a temporary config to avoid loading from a file.
    # You may need to adjust these paths if your project structure is different.
    # base_dir = Path(__file__).parent.parent 
    config = ConfigManager(
        # base_dir=str(base_dir),
        # data_dir=str(base_dir / "data"),
        # results_dir=str(base_dir / "results")
    )

    # You can customize ESN hyperparameters here for the test
    hparams = {
        "spectral_radius": 0.9, # 0.9,
        "input_length": 2000,
        "input_scaling": 1.0, #  -0.1, # 0.5, #nA value between 0.1 and 1.0 is a great starting point.
        "leak_rate": 1.0,  # only relies on input data, not older data
        "bias": 0.0, 
        "n_runs": 5,
        "train_len": 5000, 
        "test_len": 1000,
        "n_lags": 50,
        "regression_method": "ridge", # pinv", # "ridge"
        "random_state": 42, 
    }

    # 2. Create the evaluator
    # The DataLoader is needed but we won't use it to load data here.
    data_loader = DataLoader(config)
    # evaluator = ESNEvaluator(config, data_loader)

    # 3. Generate test matrices
    test_matrices = create_test_matrices()

    results_dir = {}
    
    # 4. Run evaluations
    results = {}
    for name, matrix in test_matrices.items():
        print(f"--- Testing: {name} ---")
        try:
            mc_mean, mc_array = alternative_evaluate_mc( # evaluator.evaluate_single_subject(
                W=matrix,
                h_params=hparams,
            )
            # mc_mean = result.get('mc_mean', 'N/A')
            # mc_std = result.get('mc_std', 'N/A')
            # results[name] = mc_mean
            print(f"MC Mean: {mc_mean:.4f}")
            results_dir[name] = mc_mean
            # print(f"  MC Std: {mc_std:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            # results[name] = None
            mc_mean = np.nan
            results_dir[name] = mc_mean
            

    print("\n")
    
    print("-------")
    
    output_path = "/Users/adrian/Documents/01_projects/14_4D_lab/output/testing"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write results to txt file
    with open(output_path / "mc_test_results.txt", "w") as f:
        for entry in results_dir:
            f.write(f"{entry}: {results_dir[entry]}\n")
    # Write results to terminal
    for entry in results_dir: 
        print(f"{entry}: {results_dir[entry]}")
    print("-------")
    
    
    # 5. Visualize results
    plt.figure(figsize=(12, 7))
    # plt.bar(listresults.keys(), results.values()
    D = results_dir
    # replace all nan values in D with zeros 
    D = {k: v if not np.isnan(v) else 0 for k, v in D.items()}
    plt.bar(range(len(D)), list(D.values()), align='center')
    plt.xticks(range(len(D)), list(D.keys()))
    # )
    # import pandas as pd
    # results_keys = list(results.keys())
    # results_values = list(results.values())
    # df_results = pd.DataFrame([results_keys, results_values].T)
    
    # print(df_results)
    # df_results.columns = ["Matrix Type", "Mean Memory Capacity (MC)"]

    # df_results = pd.DataFrame(results) # , index=[0])
    # sns.barplot(x=list(results.keys()), y=list(results.values()))
    # sns.barplot(data=df_results, orient="v")
    plt.title("Memory Capacity (MC) for Different Matrix Types")
    plt.xlabel("Matrix Type")
    plt.ylabel("Mean Memory Capacity (MC)")
    plt.xticks() # rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path / "mc_test_results.png")
    print(f"Results visualization saved to {output_path / "mc_test_results.png"}")
    
    
    plt.show()
    
    
    # print("\n")
    # print("-------")
    
    
    #  # Plotting the results
    # plt.style.use('seaborn-v0_8-whitegrid')
    # plt.figure(figsize=(12, 8))
    
    # n_lags = len(mc_array)
    
    # for name, r2_scores in zip(test_matrices.keys(), mc_array):
    #     print(f"--- Testing: {name} ---")
    #     plt.plot(range(1, n_lags + 1), r2_scores, marker='o', linestyle='-', markersize=4, label=name)

    # plt.title("Memory Capacity (R²) vs. Lag Duration", fontsize=16)
    # plt.xlabel("Lag Duration (k)", fontsize=12)
    # plt.ylabel("Squared Pearson Correlation (R²)", fontsize=12)
    # plt.legend(title="Matrix Type", fontsize=10)
    # # plt.ylim(0, 1) # R^2 is always between 0 and 1
    # # plt.xlim(1, n_lags)
    # plt.tight_layout()
    
    # output_path = Path("./mc_curves_comparison.png")
    # plt.savefig(output_path)
    # print(f"\nResults visualization saved to {output_path}")
    # plt.show()


    # print("-------")
    # print("\n")

if __name__ == "__main__":
    run_mc_tests()