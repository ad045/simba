import numpy as np
import scipy.signal as sig
import scipy.stats
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.analysis.from_francisco import evaluate_network_2

import hashlib

# def find_optimal_K(
#     A: np.ndarray,
#     K_range: np.ndarray = None,
#     metric: str = 'global',
#     # sim_time: float = 5, # in example notebook from git... #  500,
#     # dt: float = 0.1, # in example notebook from git... # 1e-2,
#     sim_time: float = 500, # in example notebook from git... #  500,
#     dt: float = 0.01, # in example notebook from git... # 1e-2,
#     freq: float = 0.05,
#     a: float = -0.1,
#     noise: float = 0.001,
#     estimate_DFA: bool = False,
#     verbose: bool = True
# ) -> Tuple[float, Dict]:
#     """
#     Find the optimal coupling strength (K) that maximizes metastability.
    
#     This function scans through different coupling strengths to find the critical
#     point where the network shows maximum metastability - a signature of brain
#     dynamics operating at criticality.
    
#     Parameters
#     ----------
#     A : np.ndarray
#         Connectivity matrix (adjacency matrix) of shape (N, N)
#     K_range : np.ndarray, optional
#         Array of K values to test. If None, uses np.linspace(0.01, 0.15, 30)
#     metric : str
#         Which metastability measure to optimize:
#         - 'global': global metastability (default)
#         - 'local_mean': mean local metastability
#         - 'local_std': std of local metastability
#         - 'local_kurtosis': kurtosis of local metastability
#     sim_time : float
#         Simulation time in arbitrary units
#     dt : float
#         Time step for integration
#     freq : float
#         Natural frequency of oscillators
#     a : float
#         Bifurcation parameter (if using Stuart-Landau)
#     noise : float
#         Noise amplitude
#     estimate_DFA : bool
#         Whether to estimate DFA exponents (slower)
#     verbose : bool
#         Whether to show progress bar and print results
        
#     Returns
#     -------
#     optimal_K : float
#         The K value that maximizes the chosen metric
#     results : dict
#         Dictionary containing:
#         - 'K_values': array of tested K values
#         - 'global': array of global metastability values
#         - 'local_mean': array of mean local metastability
#         - 'local_std': array of std local metastability
#         - 'local_kurtosis': array of kurtosis local metastability
#         - 'optimal_K': the optimal K value
#         - 'optimal_metric': name of optimized metric
#         - 'optimal_value': value at optimal K
        
#     Example
#     -------
#     >>> # Create random connectivity matrix
#     >>> A = np.random.rand(50, 50)
#     >>> A = (A + A.T) / 2  # Make symmetric
#     >>> np.fill_diagonal(A, 0)  # No self-connections
#     >>> 
#     >>> # Find optimal K
#     >>> optimal_K, results = find_optimal_K(A, metric='global')
#     >>> 
#     >>> # Plot results
#     >>> plot_K_optimization(results)
#     """
#     # Set default K range if not provided
#     if K_range is None:
#         K_range = np.linspace(0.01, 0.15, 30)
    
#     # Initialize storage
#     results = {
#         'K_values': K_range,
#         'global': [],
#         'local_mean': [],
#         'local_std': [],
#         'local_kurtosis': [],
#         'DFA': [] if estimate_DFA else None
#     }
    
#     # Iterate through K values
#     iterator = tqdm(K_range, desc="Optimizing K") if verbose else K_range
    
#     for K in iterator:
#         try:
#             # Simulate network with current K
#             meta_global, meta_local, DFA = evaluate_network_2(
#                 W=A, 
#                 sim_time=sim_time, 
#                 dt=dt, 
#                 K=K, 
#                 freq=freq, 
#                 a=a, 
#                 noise=noise,
#                 estimate_DFA=estimate_DFA
#             )
            
#             # Store results
#             results['global'].append(meta_global)
#             results['local_mean'].append(np.nanmean(meta_local))
#             results['local_std'].append(np.nanstd(meta_local))
#             results['local_kurtosis'].append(
#                 scipy.stats.kurtosis(meta_local, nan_policy='omit')
#             )
            
#             if estimate_DFA:
#                 results['DFA'].append(np.nanmean(DFA))
                
#         except Exception as e:
#             if verbose:
#                 print(f"Warning: Failed at K={K:.4f}: {e}")
#             results['global'].append(np.nan)
#             results['local_mean'].append(np.nan)
#             results['local_std'].append(np.nan)
#             results['local_kurtosis'].append(np.nan)
#             if estimate_DFA:
#                 results['DFA'].append(np.nan)
    
#     # Convert lists to arrays
#     results['global'] = np.array(results['global'])
#     results['local_mean'] = np.array(results['local_mean'])
#     results['local_std'] = np.array(results['local_std'])
#     results['local_kurtosis'] = np.array(results['local_kurtosis'])
#     if estimate_DFA:
#         results['DFA'] = np.array(results['DFA'])
    
#     # Find optimal K
#     valid_mask = ~np.isnan(results[metric])
#     if np.sum(valid_mask) == 0:
#         raise ValueError(f"All simulations failed for metric '{metric}'")
    
#     optimal_idx = np.nanargmax(results[metric])
#     optimal_K = K_range[optimal_idx]
#     optimal_value = results[metric][optimal_idx]
    
#     # Add optimization info to results
#     results['optimal_K'] = optimal_K
#     results['optimal_metric'] = metric
#     results['optimal_value'] = optimal_value
    
#     if verbose:
#         print(f"\n{'='*60}")
#         print(f"Optimization Results:")
#         print(f"{'='*60}")
#         print(f"Metric optimized: {metric}")
#         print(f"Optimal K: {optimal_K:.4f}")
#         print(f"Value at optimal K: {optimal_value:.4f}")
#         print(f"K range tested: [{K_range[0]:.4f}, {K_range[-1]:.4f}]")
#         print(f"Number of K values: {len(K_range)}")
#         print(f"{'='*60}\n")
    
#     return optimal_K, results






# def find_max_valid_K(
#     A: np.ndarray,
#     K_initial: float = 0.5,
#     K_lower_bound: float = 0.0001,
#     K_upper_bound: float = 1.0,
#     n_iterations: int = 20,
#     sim_time: float = 500,
#     dt: float = 0.01,
#     freq: float = 0.05,
#     a: float = -0.1,
#     noise: float = 0.001,
#     verbose: bool = True
# ) -> Tuple[float, Dict]:
#     """
#     Find the maximum K value that still produces valid (non-NaN) metastability using binary search.
    
#     Parameters
#     ----------
#     A : np.ndarray
#         Connectivity matrix (adjacency matrix) of shape (N, N)
#     K_initial : float
#         Initial K value to test (default: 0.5)
#     K_lower_bound : float
#         Lower bound for K search (default: 0.0001)
#     K_upper_bound : float
#         Upper bound for K search (default: 1.0)
#     n_iterations : int
#         Number of binary search iterations (default: 20)
#     sim_time : float
#         Simulation time in arbitrary units
#     dt : float
#         Time step for integration
#     freq : float
#         Natural frequency of oscillators
#     a : float
#         Bifurcation parameter
#     noise : float
#         Noise amplitude
#     verbose : bool
#         Whether to print progress
        
#     Returns
#     -------
#     max_valid_K : float
#         Maximum K value that produces valid results
#     history : dict
#         Dictionary containing search history with keys:
#         - 'K_tested': list of K values tested
#         - 'is_valid': list of booleans indicating if result was valid
#         - 'metastability': list of metastability values (NaN if invalid)
#     """
    
#     # Initialize bounds
#     K_low = K_lower_bound
#     K_high = K_upper_bound
#     K_current = K_initial
    
#     # Storage for search history
#     history = {
#         'K_tested': [],
#         'is_valid': [],
#         'metastability': []
#     }
    
#     # Track best valid K found
#     best_valid_K = None
#     best_metastability = None
    
#     if verbose:
#         print(f"{'='*60}")
#         print(f"Binary Search for Maximum Valid K")
#         print(f"{'='*60}")
#         print(f"Initial K: {K_current:.6f}")
#         print(f"Search bounds: [{K_low:.6f}, {K_high:.6f}]")
#         print(f"Iterations: {n_iterations}")
#         print(f"{'='*60}\n")
    
#     for iteration in range(n_iterations):
#         try:
#             # Test current K value
#             meta_global, meta_local, DFA = evaluate_network_2(
#                 W=A,
#                 sim_time=sim_time,
#                 dt=dt,
#                 K=K_current,
#                 freq=freq,
#                 a=a,
#                 noise=noise,
#                 estimate_DFA=False
#             )
            
#             # Check if result is valid (not NaN)
#             is_valid = not np.isnan(meta_global)
            
#             # Store results
#             history['K_tested'].append(K_current)
#             history['is_valid'].append(is_valid)
#             history['metastability'].append(meta_global if is_valid else np.nan)
            
#             if verbose:
#                 status = "✓ Valid" if is_valid else "✗ NaN"
#                 value_str = f"{meta_global:.6f}" if is_valid else "NaN"
#                 print(f"Iter {iteration+1:2d}: K = {K_current:.6f} → {status} (global metastability = {value_str})")
            
#             # Update bounds based on result
#             if is_valid:
#                 # Valid result - try higher K
#                 best_valid_K = K_current
#                 best_metastability = meta_global
#                 K_low = K_current
#                 K_current = (K_current + K_high) / 2
#             else:
#                 # NaN result - try lower K
#                 K_high = K_current
#                 K_current = (K_low + K_current) / 2
                
#         except Exception as e:
#             # Treat exceptions as invalid
#             if verbose:
#                 print(f"Iter {iteration+1:2d}: K = {K_current:.6f} → ✗ Error: {str(e)[:50]}")
            
#             history['K_tested'].append(K_current)
#             history['is_valid'].append(False)
#             history['metastability'].append(np.nan)
            
#             # Adjust bounds
#             K_high = K_current
#             K_current = (K_low + K_current) / 2
    
#     # Final result
#     if best_valid_K is None:
#         raise ValueError("No valid K value found in the given range. Try adjusting bounds.")
    
#     if verbose:
#         print(f"\n{'='*60}")
#         print(f"Search Results:")
#         print(f"{'='*60}")
#         print(f"Maximum valid K: {best_valid_K:.6f}")
#         print(f"Metastability at max K: {best_metastability:.6f}")
#         print(f"Final search range: [{K_low:.6f}, {K_high:.6f}]")
#         print(f"Valid results found: {sum(history['is_valid'])}/{n_iterations}")
#         print(f"{'='*60}\n")
    
#     return best_valid_K, history


# def calculate_metastability(A, # distance_matrix=None, 
#                             K_initial=0.5,          # Start in middle of range
#                             K_lower_bound=0.0001,   # Lower bound
#                             K_upper_bound=10_000,      # Upper bound (adjust if needed)
#                             n_iterations=20,        # Number of binary search iterations
#                             sim_time=500,
#                             save_debug_path=None): 
#     """
#     Calculate metastability by finding the maximum valid K value using binary search.
    
#     Parameters
#     ----------
#     A : np.ndarray
#         Connectivity matrix
#     distance_matrix : np.ndarray, optional
#         Distance matrix (not used in current implementation)
#     save_debug_path : str or Path, optional
#         Path to save debug information
        
#     Returns
#     -------
#     dict
#         Dictionary with 'optimal_K' and 'optimal_value' (metastability at that K)
#     """
    
#     # Normalize adjacency matrix
#     A = A.astype(float)
#     # A /= np.nanmean(A) # not useful, as binary... 
#     eigenvalues = np.linalg.eigvals(A)
#     spectral_radius = np.max(np.abs(eigenvalues))
#     A = A / spectral_radius * 0.99
    
#     # Find maximum valid K using binary search
#     max_K, history = find_max_valid_K(
#         A,
#         K_initial=K_initial,          # Start in middle of range
#         K_lower_bound=K_lower_bound,   # Lower bound
#         K_upper_bound=K_upper_bound,      # Upper bound (adjust if needed)
#         n_iterations=n_iterations,        # Number of binary search iterations
#         sim_time=sim_time,
#         verbose=True
#     )
    
#     # Save debug information if requested
#     if save_debug_path is not None: 
#         save_debug_path = Path(save_debug_path)
#         save_debug_path.mkdir(parents=True, exist_ok=True)
        
#         # Convert to bytes and hash (to get unique identifier)
#         array_bytes = A.tobytes()
#         unique_id = hashlib.sha256(array_bytes).hexdigest()
        
#         # Save history
#         np.save(
#             save_debug_path / f"metastability_binary_search_hash_id_{unique_id}.npy",
#             history
#         )
    
#     # Get the metastability value at max K
#     valid_indices = [i for i, is_valid in enumerate(history['is_valid']) if is_valid]
#     if valid_indices:
#         best_idx = valid_indices[-1]  # Last valid index should be the maximum K
#         optimal_value = history['metastability'][best_idx]
#     else:
#         optimal_value = np.nan
    
#     return {
#         "optimal_K": max_K, 
#         "optimal_value": optimal_value
#     }
    
    



# finds the max metastability, using adaptive binary search with trend detection (aka: shifting windows. Filters out noise)


def find_max_metastability_K(
    A: np.ndarray,
    K_initial: float = 0.5,
    K_lower_bound: float = 0.0001,
    K_upper_bound: float = 1.0,
    n_iterations: int = 20,
    window_size: int = 3,
    sim_time: float = 500,
    dt: float = 0.01,
    freq: float = 0.05,
    a: float = -0.1,
    noise: float = 0.001,
    verbose: bool = True
) -> Tuple[float, Dict]:
    """
    Find the K value that maximizes metastability using adaptive binary search.
    Uses a moving window to handle noisy signals robustly.
    
    Parameters
    ----------
    A : np.ndarray
        Connectivity matrix (adjacency matrix) of shape (N, N)
    K_initial : float
        Initial K value to test (default: 0.5)
    K_lower_bound : float
        Lower bound for K search (default: 0.0001)
    K_upper_bound : float
        Upper bound for K search (default: 1.0)
    n_iterations : int
        Number of binary search iterations (default: 20)
    window_size : int
        Number of recent points to consider for trend detection (default: 3)
        Higher values = more smoothing but slower adaptation
    sim_time : float
        Simulation time in arbitrary units
    dt : float
        Time step for integration
    freq : float
        Natural frequency of oscillators
    a : float
        Bifurcation parameter
    noise : float
        Noise amplitude
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    optimal_K : float
        K value that maximizes metastability
    history : dict
        Dictionary containing search history with keys:
        - 'K_tested': list of K values tested
        - 'is_valid': list of booleans indicating if result was valid
        - 'metastability': list of metastability values (NaN if invalid)
        - 'trend': list of trend indicators ('up', 'down', 'uncertain')
    """
    
    # Initialize bounds
    K_low = K_lower_bound
    K_high = K_upper_bound
    K_current = K_initial
    
    # Storage for search history
    history = {
        'K_tested': [],
        'is_valid': [],
        'metastability': [],
        'trend': []
    }
    
    # Track best result
    best_K = None
    best_metastability = -np.inf
    
    if verbose:
        print(f"{'='*60}")
        print(f"Binary Search for Maximum Metastability")
        print(f"{'='*60}")
        print(f"Initial K: {K_current:.6f}")
        print(f"Search bounds: [{K_low:.6f}, {K_high:.6f}]")
        print(f"Iterations: {n_iterations}")
        print(f"Window size for trend: {window_size}")
        print(f"{'='*60}\n")
    
    def compute_trend(history_dict, window_size):
        """
        Compute trend from recent valid measurements.
        Returns: 'up', 'down', or 'uncertain'
        """
        # Get recent valid measurements
        recent_K = []
        recent_meta = []
        
        for i in range(len(history_dict['is_valid'])-1, -1, -1):
            if history_dict['is_valid'][i]:
                recent_K.append(history_dict['K_tested'][i])
                recent_meta.append(history_dict['metastability'][i])
                if len(recent_K) >= window_size:
                    break
        
        if len(recent_K) < 2:
            return 'uncertain'
        
        # Reverse to get chronological order
        recent_K = recent_K[::-1]
        recent_meta = recent_meta[::-1]
        
        # Compute average trend (considering we might have noise)
        # If K is increasing and metastability is increasing -> 'up'
        # If K is increasing and metastability is decreasing -> 'down'
        
        # Simple approach: compare mean of first half vs second half
        if len(recent_meta) >= 3:
            first_half = np.mean(recent_meta[:len(recent_meta)//2])
            second_half = np.mean(recent_meta[len(recent_meta)//2:])
            
            if second_half > first_half:
                return 'up'
            elif second_half < first_half:
                return 'down'
            else:
                return 'uncertain'
        else:
            # Just compare last two
            if recent_meta[-1] > recent_meta[-2]:
                return 'up'
            elif recent_meta[-1] < recent_meta[-2]:
                return 'down'
            else:
                return 'uncertain'
    
    for iteration in range(n_iterations):
        try:
            # Test current K value
            meta_global, meta_local, DFA = evaluate_network_2(
                W=A,
                sim_time=sim_time,
                dt=dt,
                K=K_current,
                freq=freq,
                a=a,
                noise=noise,
                estimate_DFA=False
            )
            
            # Check if result is valid (not NaN)
            is_valid = not np.isnan(meta_global)
            
            if is_valid:
                # Update best if current is better
                if meta_global > best_metastability:
                    best_metastability = meta_global
                    best_K = K_current
            
            # Store results
            history['K_tested'].append(K_current)
            history['is_valid'].append(is_valid)
            history['metastability'].append(meta_global if is_valid else np.nan)
            
            # Compute trend
            trend = 'uncertain'
            if len(history['is_valid']) >= 2:
                trend = compute_trend(history, window_size)
            history['trend'].append(trend)
            
            if verbose:
                status = "✓ Valid" if is_valid else "✗ NaN"
                value_str = f"{meta_global:.6f}" if is_valid else "NaN"
                best_str = f" [NEW BEST!]" if (is_valid and meta_global == best_metastability) else ""
                trend_arrow = "↑" if trend == 'up' else "↓" if trend == 'down' else "?"
                print(f"Iter {iteration+1:2d}: K = {K_current:.6f} → {status} (meta = {value_str}) {trend_arrow}{best_str}")
            
            # Update bounds based on trend
            if not is_valid:
                # Hit NaN - definitely went too high
                # K_high = K_current
                # K_current = (K_low + K_current) / 2
                K_buffer = K_high - K_current
                K_high = K_current
                K_current = K_current - K_buffer / 2 # just move K current slightly downwards - not entirely. 
            else:
                # Valid result - use trend to decide direction
                if trend == 'up':
                    # Metastability is increasing - try higher K
                    print("here:", K_current, K_high)
                    K_low = K_current
                    K_current = (K_current + K_high) / 2
                elif trend == 'down':
                    # Metastability is decreasing - try lower K
                    print("here2:", K_current, K_low)
                    K_high = K_current
                    K_current = (K_low + K_current) / 2
                else:
                    # Uncertain - stay in current region but narrow search
                    # Slightly favor going up (since we're looking for maximum)
                    print("here3:", K_current, K_high)
                    K_low = max(K_low, K_current * 0.9)
                    K_high = min(K_high, K_current * 1.1)
                    K_current = (K_low + K_high) / 2
                
        except Exception as e:
            # Treat exceptions as invalid
            if verbose:
                print(f"Iter {iteration+1:2d}: K = {K_current:.6f} → ✗ Error: {str(e)[:50]}")
            
            history['K_tested'].append(K_current)
            history['is_valid'].append(False)
            history['metastability'].append(np.nan)
            history['trend'].append('uncertain')
            
            # Adjust bounds (likely went too high)
            K_high = K_current
            K_current = (K_low + K_current) / 2
    
    # Final result
    if best_K is None:
        raise ValueError("No valid K value found in the given range. Try adjusting bounds.")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Search Results:")
        print(f"{'='*60}")
        print(f"Optimal K (max metastability): {best_K:.6f}")
        print(f"Maximum metastability: {best_metastability:.6f}")
        print(f"Final search range: [{K_low:.6f}, {K_high:.6f}]")
        print(f"Valid results found: {sum(history['is_valid'])}/{n_iterations}")
        print(f"{'='*60}\n")
    
    return best_K, history


def calculate_metastability(A, 
                            K_initial=0.5,
                            K_lower_bound=0.0001,
                            K_upper_bound=10_000,
                            n_iterations=20,
                            window_size=3,
                            sim_time=500,
                            save_debug_path=None): 
    """
    Calculate metastability by finding the K value that maximizes it using binary search.
    
    Parameters
    ----------
    A : np.ndarray
        Connectivity matrix
    K_initial : float
        Initial K to test
    K_lower_bound : float
        Lower search bound
    K_upper_bound : float
        Upper search bound
    n_iterations : int
        Number of search iterations
    window_size : int
        Number of recent points to use for trend detection (default: 3)
    sim_time : float
        Simulation time
    save_debug_path : str or Path, optional
        Path to save debug information
        
    Returns
    -------
    dict
        Dictionary with 'optimal_K' and 'optimal_value' (maximum metastability)
    """
    
    # Normalize adjacency matrix
    A = A.astype(float)
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigenvalues))
    A = A / spectral_radius * 0.99
    
    # Find K that maximizes metastability using binary search
    optimal_K, history = find_max_metastability_K(
        A,
        K_initial=K_initial,
        K_lower_bound=K_lower_bound,
        K_upper_bound=K_upper_bound,
        n_iterations=n_iterations,
        window_size=window_size,
        sim_time=sim_time,
        verbose=True
    )
    
    # Save debug information if requested
    if save_debug_path is not None: 
        save_debug_path = Path(save_debug_path)
        save_debug_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to bytes and hash (to get unique identifier)
        array_bytes = A.tobytes()
        unique_id = hashlib.sha256(array_bytes).hexdigest()
        
        # Save history
        np.save(
            save_debug_path / f"metastability_max_search_hash_id_{unique_id}.npy",
            history
        )
    
    # Get the maximum metastability value found
    valid_indices = [i for i, is_valid in enumerate(history['is_valid']) if is_valid]
    if valid_indices:
        valid_metastabilities = [history['metastability'][i] for i in valid_indices]
        optimal_value = max(valid_metastabilities)
    else:
        optimal_value = np.nan
    
    return {
        "optimal_K": optimal_K, 
        "optimal_value": optimal_value, 
        "history": history
    }
    
    
    
    
    
    
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


def plot_arr_K_optimization(
    # results: List[Dict],
    # K_range, 
    # optimal_K_arr,
    results_arr,
    start_idx_arr,
    optimal_metric, 
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
    
    for i, subject_id in enumerate(start_idx_arr):
        # make this nicer: subject_id in results_arr, ... 
        # K_values = optimal_K_arr[i]
        results_dict = results_arr[i]
        optimal_K = results_dict["optimal_K"] # optimal_K_arr[i]
        results = results_dict["results"] # results_arr[i]
        K_values = results["K_values"]
        values = results[optimal_metric]

        # Plot each metric
        metrics = [
            ('global', 'Global Metastability', axes[0, 0]),
            ('local_mean', 'Mean Local Metastability', axes[0, 1]),
            ('local_std', 'Std Local Metastability', axes[1, 0]),
            ('local_kurtosis', 'Local Metastability Kurtosis', axes[1, 1])
        ]
        
        for metric_key, title, ax in metrics:
            
            # Plot curve
            ax.plot(K_values, values, 'o-', linewidth=2, markersize=6, 
                    color='steelblue', label=title)
            
            # Highlight optimal point if this is the optimized metric
            if metric_key == optimal_metric:
                ax.axvline(optimal_K, color='red', linestyle='--', linewidth=2, 
                        label=f'Optimal K = {optimal_K:.4f}')
                ax.plot(optimal_K, max(values), 'r*', 
                    markersize=20, label=f'Max = {max(values):.4f}')
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
            
            if i == 0: 
                ax.legend(loc='best')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
            
            


import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.cm as cm

from vizman import viz
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

from matplotlib import font_manager
for font in font_manager.findSystemFonts("figures/Atkinson_Typeface/"):
    font_manager.fontManager.addfont(font)

viz.set_visual_style()
default_sizes = viz.load_data_from_json("sizes.json")
default_colors = viz.load_data_from_json("colors.json")
default_cmaps = viz.give_colormaps()

viz.set_visual_style()
default_sizes = viz.load_data_from_json("sizes.json")
default_colors = viz.load_data_from_json("colors.json")
default_cmaps = viz.give_colormaps()


def plot_arr_K_optimization(
    results_arr,
    start_idx_arr,
    optimal_metric, 
    title: Optional[str] = None, 
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = viz.cm_to_inch((18*3,5*3)) # , 12)
): 
    """
    Plot the results of K optimization with improved layout and readability.
    
    Layout: 3 rows × 4 columns
    - Columns 1-3: Main optimized metric plot (spans 3 columns)
    - Column 4: Other three metrics stacked vertically
    
    Parameters
    ----------
    results_arr : list
        List of result dictionaries for each subject
    start_idx_arr : list
        List of subject identifiers
    optimal_metric : str
        The metric being optimized ('global', 'local_mean', 'local_std', 'local_kurtosis')
    save_path : str, optional
        Path to save figure. If None, displays instead
    figsize : tuple
        Figure size (width, height)
    """
    # Create figure with custom grid layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4) # , hspace=0.3, wspace=0.3)
    
    # Main plot spans first 3 columns, all 3 rows
    ax_main = fig.add_subplot(gs[:, :3])
    
    # Three smaller plots in the last column
    ax_top = fig.add_subplot(gs[0, 3])
    ax_mid = fig.add_subplot(gs[1, 3])
    ax_bot = fig.add_subplot(gs[2, 3])
    
    # Color palette for different subjects
    n_subjects = len(start_idx_arr)
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, n_subjects))
    line_styles = ['-', '--', '-.', ':']
    
    # Metric configuration
    metric_config = {
        'global': 'Global Metastability',
        'local_mean': 'Mean Local Metastability',
        'local_std': 'Std Local Metastability',
        'local_kurtosis': 'Local Metastability Kurtosis'
    }
    
    # Determine which axes to use for non-optimized metrics
    other_metrics = [m for m in ['global', 'local_mean', 'local_std', 'local_kurtosis'] 
                     if m != optimal_metric]
    other_axes = [ax_top, ax_mid, ax_bot]
    
    # Plot data for each subject
    for i, subject_id in enumerate(start_idx_arr):
        results_dict = results_arr[i]
        optimal_K = results_dict["optimal_K"]
        results = results_dict["results"]
        K_values = results["K_values"]
        
        # Select color and line style
        color = colors[i]
        line_style = line_styles[i % len(line_styles)]
        
        # Plot main optimized metric
        main_values = results[optimal_metric]
        ax_main.plot(K_values, main_values, 
                    linestyle=line_style, 
                    # linewidth=2.5, 
                    # marker='o',
                    # markersize=4,
                    color=color,
                    alpha=0.8,
                    label=f'Subject {subject_id}')
        
        # Mark optimal point
        max_idx = np.argmax(main_values)
        ax_main.plot(optimal_K, main_values[max_idx], 
                     '*', 
                    # markersize=20, 
                    color=color,
                    markeredgecolor='darkred',
                    # markeredgewidth=1.5,
                    zorder=5)
        
        # Add vertical line for optimal K (only for first subject to avoid clutter)
        if i == 0:
            ax_main.axvline(optimal_K, color='red', linestyle='--', 
                        #   linewidth=1.5, 
                          alpha=0.6, label='Optimal K')
        
        # Plot other metrics on smaller axes
        for metric_key, ax in zip(other_metrics, other_axes):
            values = results[metric_key]
            ax.plot(K_values, values,
                   linestyle=line_style,
                #    linewidth=2,
                #    marker='o',
                #    markersize=3,
                #    color=color,
                   alpha=0.7)
            
            # Mark where optimal K falls on this metric
            optimal_idx = np.argmin(np.abs(K_values - optimal_K))
            ax.plot(optimal_K, values[optimal_idx], # '*',
                #    markersize=12,
                #    color=color,
                #    markeredgecolor='darkred',
                #    markeredgewidth=1,
                   zorder=5)
    
    # Configure main plot
    ax_main.set_xlabel('Coupling Strength (K)') # , fontsize=14, fontweight='bold')
    ax_main.set_ylabel(metric_config[optimal_metric]) # , fontsize=14, fontweight='bold')
    ax_main.set_title(f'{metric_config[optimal_metric]} (OPTIMIZED)') # , 
                    # fontsize=16, fontweight='bold', color='darkred', pad=20)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    # ax_main.legend(False) # loc='upper right') # , fontsize=11, framealpha=0.9)
    ax_main.tick_params() # labelsize=11)
    
    # Configure other metric plots
    for metric_key, ax in zip(other_metrics, other_axes):
        ax.set_xlabel('K') # , fontsize=10)
        ax.set_ylabel(metric_config[metric_key])  # , fontsize=9)
        ax.set_title(metric_config[metric_key]) # , fontsize=11, fontweight='bold')
        # ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params() # labelsize=9)
    
    # Overall title
    if title is not None:
        fig.suptitle(title)
                # fontsize=18, fontweight='bold', y=0.995)
    else:
        fig.suptitle('Coupling Strength (K) Optimization Analysis')
                # fontsize=18, fontweight='bold', y=0.995)
    
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

