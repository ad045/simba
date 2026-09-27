# See calculate_metastability_optimize_K.py -> code is thus mainly stolen from Kayson etc. 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import hashlib

# from src.analysis.calculate_metastability_optimize_K import (find_optimal_K,
#                                                              plot_K_optimization,
#                                                              plot_arr_K_optimization
# )

from src.analysis.calculate_metastability_optimize_K_binary_search import calculate_metastability


# def calculate_metastability(A, distance_matrix=None, save_debug_path=None): 
    
#     # A = A.astype(float) 
#     # A /= np.nanmean(A)
    
#     K_range = np.linspace(0, 1000, 21) # np.logspace(-4, 3, 41)
#     optimal_metric = 'global'

#     # Find optimal K
#     optimal_K, results = find_optimal_K(
#         A, 
#         K_range=K_range,
#         metric=optimal_metric, 
#         sim_time=500,  # Shorter for example
#         verbose=True
#     )
    
#     if save_debug_path is not None: 
#         save_debug_path = Path(save_debug_path)
#         save_debug_path.mkdir(parents=True, exist_ok=True)
        
#         # Convert to bytes and hash (to get unique identifier)
#         array_bytes = A.tobytes()
#         unique_id = hashlib.sha256(array_bytes).hexdigest()

#         np.save(save_debug_path / f"metastability_debug_results_hash_id_{unique_id}.npy", results)

#     return {"optimal_K": results["optimal_K"], 
#             "optimal_value": results["optimal_value"]}
    



# NEW ATTEMPT FOLLOWING FRANCISCOS CODE EXACTLY 
# def calculate_metastability(A, distance_matrix=None, save_debug_path=None): 
    
#     K_range = np.logspace(-4, 0, 101) # np.linspace(0, 1000, 21) # np.logspace(-4, 3, 41)
#     optimal_metric = 'global'
    
#     A = A.astype(float) 
#     A/= np.nanmean(A)
    
#     Z = models.simulate_stuart_landau(W = W, sim_time = sim_time, dt = dt)
    
#     # Find optimal K
#     optimal_K, results = find_optimal_K(
#         A, 
#         K_range=K_range,
#         metric=optimal_metric, 
#         sim_time=500,  # Shorter for example
#         verbose=True
#     )
    
#     if save_debug_path is not None: 
#         save_debug_path = Path(save_debug_path)
#         save_debug_path.mkdir(parents=True, exist_ok=True)
        
#         # Convert to bytes and hash (to get unique identifier)
#         array_bytes = A.tobytes()
#         unique_id = hashlib.sha256(array_bytes).hexdigest()

#         np.save(save_debug_path / f"metastability_debug_results_hash_id_{unique_id}.npy", results)

#     return {"optimal_K": results["optimal_K"], 
#             "optimal_value": results["optimal_value"]}
    
