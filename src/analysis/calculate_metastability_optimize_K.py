import numpy as np
import scipy.signal as sig
import scipy.stats
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.analysis.from_francisco import evaluate_network_2

def find_optimal_K(
    A: np.ndarray,
    K_range: np.ndarray = None,
    metric: str = 'global',
    sim_time: float = 500,
    dt: float = 1e-2,
    freq: float = 0.05,
    a: float = -0.1,
    noise: float = 0.001,
    estimate_DFA: bool = False,
    verbose: bool = True
) -> Tuple[float, Dict]:
    """
    Find the optimal coupling strength (K) that maximizes metastability.
    
    This function scans through different coupling strengths to find the critical
    point where the network shows maximum metastability - a signature of brain
    dynamics operating at criticality.
    
    Parameters
    ----------
    A : np.ndarray
        Connectivity matrix (adjacency matrix) of shape (N, N)
    K_range : np.ndarray, optional
        Array of K values to test. If None, uses np.linspace(0.01, 0.15, 30)
    metric : str
        Which metastability measure to optimize:
        - 'global': global metastability (default)
        - 'local_mean': mean local metastability
        - 'local_std': std of local metastability
        - 'local_kurtosis': kurtosis of local metastability
    sim_time : float
        Simulation time in arbitrary units
    dt : float
        Time step for integration
    freq : float
        Natural frequency of oscillators
    a : float
        Bifurcation parameter (if using Stuart-Landau)
    noise : float
        Noise amplitude
    estimate_DFA : bool
        Whether to estimate DFA exponents (slower)
    verbose : bool
        Whether to show progress bar and print results
        
    Returns
    -------
    optimal_K : float
        The K value that maximizes the chosen metric
    results : dict
        Dictionary containing:
        - 'K_values': array of tested K values
        - 'global': array of global metastability values
        - 'local_mean': array of mean local metastability
        - 'local_std': array of std local metastability
        - 'local_kurtosis': array of kurtosis local metastability
        - 'optimal_K': the optimal K value
        - 'optimal_metric': name of optimized metric
        - 'optimal_value': value at optimal K
        
    Example
    -------
    >>> # Create random connectivity matrix
    >>> A = np.random.rand(50, 50)
    >>> A = (A + A.T) / 2  # Make symmetric
    >>> np.fill_diagonal(A, 0)  # No self-connections
    >>> 
    >>> # Find optimal K
    >>> optimal_K, results = find_optimal_K(A, metric='global')
    >>> 
    >>> # Plot results
    >>> plot_K_optimization(results)
    """
    # Set default K range if not provided
    if K_range is None:
        K_range = np.linspace(0.01, 0.15, 30)
    
    # Initialize storage
    results = {
        'K_values': K_range,
        'global': [],
        'local_mean': [],
        'local_std': [],
        'local_kurtosis': [],
        'DFA': [] if estimate_DFA else None
    }
    
    # Iterate through K values
    iterator = tqdm(K_range, desc="Optimizing K") if verbose else K_range
    
    for K in iterator:
        try:
            # Simulate network with current K
            meta_global, meta_local, DFA = evaluate_network_2(
                W=A, 
                sim_time=sim_time, 
                dt=dt, 
                K=K, 
                freq=freq, 
                a=a, 
                noise=noise,
                estimate_DFA=estimate_DFA
            )
            
            # Store results
            results['global'].append(meta_global)
            results['local_mean'].append(np.nanmean(meta_local))
            results['local_std'].append(np.nanstd(meta_local))
            results['local_kurtosis'].append(
                scipy.stats.kurtosis(meta_local, nan_policy='omit')
            )
            
            if estimate_DFA:
                results['DFA'].append(np.nanmean(DFA))
                
        except Exception as e:
            if verbose:
                print(f"Warning: Failed at K={K:.4f}: {e}")
            results['global'].append(np.nan)
            results['local_mean'].append(np.nan)
            results['local_std'].append(np.nan)
            results['local_kurtosis'].append(np.nan)
            if estimate_DFA:
                results['DFA'].append(np.nan)
    
    # Convert lists to arrays
    results['global'] = np.array(results['global'])
    results['local_mean'] = np.array(results['local_mean'])
    results['local_std'] = np.array(results['local_std'])
    results['local_kurtosis'] = np.array(results['local_kurtosis'])
    if estimate_DFA:
        results['DFA'] = np.array(results['DFA'])
    
    # Find optimal K
    valid_mask = ~np.isnan(results[metric])
    if np.sum(valid_mask) == 0:
        raise ValueError(f"All simulations failed for metric '{metric}'")
    
    optimal_idx = np.nanargmax(results[metric])
    optimal_K = K_range[optimal_idx]
    optimal_value = results[metric][optimal_idx]
    
    # Add optimization info to results
    results['optimal_K'] = optimal_K
    results['optimal_metric'] = metric
    results['optimal_value'] = optimal_value
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimization Results:")
        print(f"{'='*60}")
        print(f"Metric optimized: {metric}")
        print(f"Optimal K: {optimal_K:.4f}")
        print(f"Value at optimal K: {optimal_value:.4f}")
        print(f"K range tested: [{K_range[0]:.4f}, {K_range[-1]:.4f}]")
        print(f"Number of K values: {len(K_range)}")
        print(f"{'='*60}\n")
    
    return optimal_K, results


def plot_K_optimization(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Plot the results of K optimization showing all metrics.
    
    Parameters
    ----------
    results : dict
        Dictionary returned by find_optimal_K
    save_path : str, optional
        Path to save figure. If None, displays instead
    figsize : tuple
        Figure size (width, height)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Coupling Strength (K) Optimization', fontsize=16, fontweight='bold')
    
    K_values = results['K_values']
    optimal_K = results['optimal_K']
    optimal_metric = results['optimal_metric']
    
    # Plot each metric
    metrics = [
        ('global', 'Global Metastability', axes[0, 0]),
        ('local_mean', 'Mean Local Metastability', axes[0, 1]),
        ('local_std', 'Std Local Metastability', axes[1, 0]),
        ('local_kurtosis', 'Local Metastability Kurtosis', axes[1, 1])
    ]
    
    for metric_key, title, ax in metrics:
        values = results[metric_key]
        
        # Plot curve
        ax.plot(K_values, values, 'o-', linewidth=2, markersize=6, 
                color='steelblue', label=title)
        
        # Highlight optimal point if this is the optimized metric
        if metric_key == optimal_metric:
            ax.axvline(optimal_K, color='red', linestyle='--', linewidth=2, 
                      label=f'Optimal K = {optimal_K:.4f}')
            ax.plot(optimal_K, results['optimal_value'], 'r*', 
                   markersize=20, label=f'Max = {results["optimal_value"]:.4f}')
            ax.set_title(f'{title} (OPTIMIZED)', fontweight='bold', color='red')
        else:
            # Still show where optimal K falls on other metrics
            optimal_idx = np.argmin(np.abs(K_values - optimal_K))
            ax.axvline(optimal_K, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.plot(optimal_K, values[optimal_idx], 'r*', markersize=15, alpha=0.5)
            ax.set_title(title)
        
        ax.set_xlabel('Coupling Strength (K)', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


def calculate_metastability_at_K(A: np.ndarray, K: float = None, **kwargs):
    """
    Convenience function that finds optimal K and returns metastability metrics.
    
    If K is not provided, finds the optimal K first. Otherwise uses the given K.
    
    Parameters
    ----------
    A : np.ndarray
        Connectivity matrix
    K : float, optional
        Coupling strength. If None, finds optimal K first.
    **kwargs : dict
        Additional arguments passed to evaluate_network_2
        
    Returns
    -------
    results : dict
        Dictionary with metastability metrics at the specified (or optimal) K
    """
    
    if K is None:
        print("Finding optimal K...")
        K, opt_results = find_optimal_K(A, verbose=True, **kwargs)
        print(f"Using optimal K = {K:.4f}")
    
    # Calculate metastability at specified K
    meta_global, meta_local, DFA = evaluate_network_2(W=A, K=K, **kwargs)
    
    return {
        "K": K,
        "global": meta_global,
        "local_mean": np.nanmean(meta_local),
        "local_std": np.nanstd(meta_local),
        "local_kurtosis": scipy.stats.kurtosis(meta_local, nan_policy='omit'),
        "local_full": meta_local,
        "DFA": DFA
    }


# Example usage
# if __name__ == "__main__":
#     print("Example: Optimizing coupling strength for a random network\n")
    
#     # Create a random connectivity matrix
#     n_nodes = 50
#     A = np.random.rand(n_nodes, n_nodes)
#     A = (A + A.T) / 2  # Make symmetric
#     np.fill_diagonal(A, 0)  # No self-connections
#     A = A / A.sum(axis=1, keepdims=True)  # Normalize
    
#     # Find optimal K
#     optimal_K, results = find_optimal_K(
#         A, 
#         K_range=np.linspace(0.01, 0.1, 20),
#         metric='global',
#         sim_time=200,  # Shorter for example
#         verbose=True
#     )
    
#     # Plot results
#     plot_K_optimization(results)
    
#     print("\nTip: You can also optimize for different metrics:")
#     print("  - 'global': maximize global synchronization variability")
#     print("  - 'local_mean': maximize average local metastability")
#     print("  - 'local_std': maximize heterogeneity of local dynamics")
#     print("  - 'local_kurtosis': optimize for heavy-tailed local dynamics")