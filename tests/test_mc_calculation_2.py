import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import echoes

# Assuming these imports are available in your project structure
from src.config.ESN_and_GNM_config import ConfigManager
from src.utils.data_loader import DataLoader

def create_test_matrices(n_nodes: int = 50) -> dict:
    """
    Generates a dictionary of test connectome matrices.
    """
    matrices = {
        "zero_matrix": np.zeros((n_nodes, n_nodes)),
        "identity_matrix": np.eye(n_nodes),
        "fully_connected": np.ones((n_nodes, n_nodes)) - np.eye(n_nodes),
        "random_sparse": None,
        "random_dense": None,
        "ring_lattice": None,
        "disconnected_components": None,
    }

    # Seed for reproducibility
    rng = np.random.default_rng(42)

    # Random sparse matrix
    sparse = rng.random((n_nodes, n_nodes))
    sparse[sparse < 0.8] = 0
    matrices["random_sparse"] = (sparse + sparse.T) / 2

    # Random dense matrix
    dense = rng.random((n_nodes, n_nodes))
    matrices["random_dense"] = (dense + dense.T) / 2

    # Ring lattice
    ring = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(1, 3):
            ring[i, (i + j) % n_nodes] = 1
            ring[i, (i - j) % n_nodes] = 1
    matrices["ring_lattice"] = ring

    # Disconnected components
    disconnected = np.zeros((n_nodes, n_nodes))
    comp_size = n_nodes // 2
    comp1 = rng.random((comp_size, comp_size))
    comp1[comp1 < 0.5] = 0
    disconnected[:comp_size, :comp_size] = (comp1 + comp1.T) / 2
    matrices["disconnected_components"] = disconnected

    return matrices

def calculate_mc_over_lags(W, n_lags=50, train_len=2000, test_len=1000, hparams=None):
    """
    Evaluates the Memory Capacity for each lag duration and returns the R^2 scores.
    """
    # 1. Generate data for the MC task
    rng = np.random.default_rng(hparams.get("random_state", 42))
    random_sequence = rng.uniform(-0.5, 0.5, train_len + test_len + n_lags + 1)
    X = random_sequence.reshape(-1, 1)
    
    y = np.zeros((len(X), n_lags))
    
    for i in range(1, n_lags + 1):
        y[i:, i - 1] = X[:-i, 0]
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X_train, X_test = X[:train_len], X[train_len:train_len + test_len]
    y_train, y_test = y[:train_len], y[train_len:train_len + test_len]
    W = np.array(W, dtype=np.float32)
    
    # 2. Create and train the ESN
    esn = echoes.ESNRegressor(
        W=W,
        spectral_radius=hparams["spectral_radius"],
        input_scaling=hparams["input_scaling"],
        leak_rate=hparams["leak_rate"],
        bias=hparams["bias"],
        regression_method=hparams["regression_method"],
        random_state=hparams.get("random_state", 42),
    )
    
    esn.fit(X_train, y_train)
    y_pred = esn.predict(X_test)

    # 3. Calculate the R^2 score for each lag
    r2_scores = []
    for i in range(n_lags):
        # Discard initial transient phase from test data for stable correlation
        corr, _ = pearsonr(y_test[100:, i], y_pred[100:, i])
        r2_scores.append(corr**2)
        
    return np.array(r2_scores)


def plot_mc_curves():
    """
    Runs MC evaluation for different lags and plots the results for all matrices.
    """
    n_lags = 50  # Maximum lag to test

    # hparams = {
    #     "spectral_radius": 0.9, # 0.9,
    #     "input_length": 2000,
    #     "input_scaling": 1.0, #  -0.1, # 0.5, #nA value between 0.1 and 1.0 is a great starting point.
    #     "leak_rate": 1.0,  # only relies on input data, not older data
    #     "bias": 0.0, 
    #     "n_runs": 5,
    #     "train_len": 5000, 
    #     "test_len": 1000,
    #     "n_lags": 50,
    #     "regression_method": "ridge", # pinv", # "ridge"
    #     "random_state": 42, 
    # }
    
    # Optimal hyperparameters based on Damicelli et al.
    hparams = {
        "spectral_radius": 1.2,
        "input_length": 2000,
        "input_scaling": 0.1,
        "leak_rate": 0.3,
        "bias": 1,
        "regression_method": "ridge", # Using ridge as in the paper
        "random_state": 42,
    }

    test_matrices = create_test_matrices()
    results = {}

    for name, matrix in test_matrices.items():
        print(f"--- Testing: {name} ---")
        try:
            # Get the R^2 scores for each lag
            r2_over_lags = calculate_mc_over_lags(
                W=matrix,
                n_lags=n_lags,
                hparams=hparams
            )
            results[name] = r2_over_lags
            # Print the total MC (sum of R^2 scores)
            print(f"  Total MC: {np.sum(r2_over_lags):.4f}")
        except Exception as e:
            print(f"  ERROR: Could not compute MC for {name}. Reason: {e}")
            results[name] = np.zeros(n_lags) # Assign zero array on error

    # Plotting the results
    # plt.style.use('seaborn-v0_8-whitegrid')
    print("-------")
    plt.figure(figsize=(12, 8))
    
    for name, r2_scores in results.items():
        plt.plot(range(1, n_lags + 1), r2_scores, marker='o', linestyle='-', markersize=4, label=name)

    plt.title("Memory Capacity (R²) vs. Lag Duration", fontsize=16)
    plt.xlabel("Lag Duration (k)", fontsize=12)
    plt.ylabel("Squared Pearson Correlation (R²)", fontsize=12)
    plt.legend(title="Matrix Type", fontsize=10)
    # plt.ylim(0, 1) # R^2 is always between 0 and 1
    # plt.xlim(1, n_lags)
    plt.tight_layout()
    
    output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/testing") 
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "mc_curves_comparison.png"
    plt.savefig(output_path)
    print(f"\nResults visualization saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_mc_curves()