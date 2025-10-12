# Modelled after damicelli's work. 

import numpy as np
from typing import List, Dict, Optional, Any

import warnings

from src.ESNs.utils_math import _entropy, _calculate_information_dynamics, _calculate_branching_ratio
import echoes
import numpy as np
from scipy.stats import pearsonr


def alternative_evaluate_mc(W, 
                            h_params):
    """Evaluates the Memory Capacity of a given reservoir matrix W."""
    
    train_len = h_params["train_len"]
    test_len = h_params["test_len"]
    n_lags = h_params["n_lags"]
    
    # Generate data for the MC task
    random_sequence = np.random.uniform(-0.5, 0.5, train_len + test_len)
    X = random_sequence.reshape(-1, 1)
    y = np.zeros((len(X), n_lags))
    for i in range(1, n_lags + 1):
        y[i:, i - 1] = X[:-i, 0]
        
    # Turn X and y to correct dtype 
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Train / Test split 
    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]

    # Create and train the ESN
    if h_params: 
        esn = echoes.ESNRegressor(
            W=W, 
            spectral_radius=float(h_params["spectral_radius"]), 
            input_scaling=float(h_params["input_scaling"]),  # tends to be string (like "1e-5")
            leak_rate=h_params["leak_rate"], 
            bias=h_params["bias"],
            regression_method=h_params["regression_method"],
            random_state=h_params["random_state"]
        )
    else: 
        Warning("Oh no, no h_params were given to the ESNRegressor in alternative_test_memory_capacity_weighted.py. ")
    
    # Fit ESN and predict y
    esn.fit(X_train, y_train)
    y_pred = esn.predict(X_test)

    # Calculate the MC score
    mc_score = 0
    r2_scores = []
    for i in range(n_lags):
        # Discard initial transient phase from test data for stable correlation
        n_transient_phase = h_params["n_transient"] # TODO: make this to a hparam
        corr, _ = pearsonr(y_test[n_transient_phase:, i], y_pred[n_transient_phase:, i]) # Discard initial transient. (statistic, p_value)
        r2_scores.append(corr**2)
        mc_score += corr**2
        
    return mc_score, np.array(r2_scores)


def evaluate_memory_capacity_from_connectome(connectome: np.ndarray, 
                                             h_params: Optional[Dict[str, Any]] = None,
                                             calculate_criticality:  Optional[bool] = False,
                                             calculate_info_dynamics: Optional[bool] = False, 
 ) -> Dict[str, float]:
    
    # Random generator to generate the input sequence
    mc_values: List[float] = []
    all_states_for_metrics: List[np.ndarray] = []   
    mc_values_indiv_array = []
    
    n_runs = h_params["n_runs"]
    
    # Loop through the number of runs
    for _ in range(n_runs):
        (mc_mean, individual_mc_values) = alternative_evaluate_mc(connectome, h_params=h_params)  
        mc_values.append(mc_mean) # which is only the mean value
        mc_values_indiv_array.append(np.array(individual_mc_values)) 
        
    mc_values = np.array(mc_values)
    mc_values_for_indiv_lags = np.mean(np.array(mc_values_indiv_array), axis=0) # has shape (50,0) 
    
    mc_result_dict = {
            "mc_mean": float(np.mean(mc_values)),
            "mc_std": float(np.std(mc_values)), 
            "mc_values_for_indiv_lags": mc_values_for_indiv_lags,
            "h_params": h_params
            }
    
    #                                          calculate_criticality:  Optional[bool] = False,
    #                                          calculate_info_dynamics: Optional[bool] = False, 
    #                                          mc_lengths: Optional[List[int]] = None, 
                                             
                                             
    # # TODO: This is not integrated yet - do this later on? -> maybe also in "alternate_evaluate_mc" function?     
    # # What about entropy?                           
    # if (calculate_criticality or calculate_info_dynamics) and all_states_for_metrics:
    #     # Concatenate states from all runs for a more robust estimation
    #     concatenated_states = np.vstack(all_states_for_metrics)
        
    #     if calculate_criticality:
    #         branching_ratio = _calculate_branching_ratio(concatenated_states)
    #         mc_result_dict['branching_ratio'] = branching_ratio
            
    #     if calculate_info_dynamics:
    #         info_dyn_results = _calculate_information_dynamics(concatenated_states)
    #         mc_result_dict.update(info_dyn_results)

    return mc_result_dict
    