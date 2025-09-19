# Modelled after damicelli's work. 
# -> Used at least for main_pipeline_2_gnm_esn_landscape.py and for main_pipeline_2.py (esn part)

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from echoes.esn import ESNRegressor

import warnings

from src.ESNs.utils_math import _entropy, _calculate_information_dynamics, _calculate_branching_ratio
import echoes
import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# import os
# import urllib.request
# import zipfile



# def _generate_mc_dataset(train_len: int, test_len: int, n_lags: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """ Generates a dataset for memory capacity evaluation."""
    
#     total_len = int(train_len + test_len + n_lags + 100)

#     seq = rng.uniform(-0.5, 0.5, size=(total_len,))
#     def build_targets(x: np.ndarray, lags: int) -> np.ndarray:
#         T = len(x) - lags
#         targets = np.zeros((T, lags), dtype=float)
#         for i in range(lags):
#             targets[:, i] = x[lags - (i + 1) : - (i + 1) if i + 1 > 0 else None]
#         return targets
#     Y_full = build_targets(seq, n_lags)
#     start_train = 100
#     end_train = start_train + train_len
#     X_train = seq[start_train : end_train].reshape(-1, 1)
#     Y_train = Y_full[start_train : end_train]
#     X_test = seq[end_train : end_train + test_len].reshape(-1, 1)
#     Y_test = Y_full[end_train : end_train + test_len]
#     return X_train, Y_train, X_test, Y_test


def alternative_evaluate_mc(W, 
                            # n_lags=50, 
                            # train_len=4000, test_len=1000, 
                            h_params):
    """Evaluates the Memory Capacity of a given reservoir matrix W."""
    
    train_len = h_params["train_len"]
    test_len = h_params["test_len"]
    n_lags = h_params["n_lags"]
    
    # 1. Generate data for the MC task
    random_sequence = np.random.uniform(-0.5, 0.5, train_len + test_len)
    X = random_sequence.reshape(-1, 1)
    y = np.zeros((len(X), n_lags))
    for i in range(1, n_lags + 1):
        y[i:, i - 1] = X[:-i, 0]
    
    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]

    # 2. Create and train the ESN
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
        # esn = echoes.ESNRegressor(
        #     W=W, 
        #     spectral_radius=0.99, 
        #     input_scaling=1e-5, 
        #     leak_rate=1, 
        #     bias=1,
        #     random_state=42
        # )
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    esn.fit(X_train, y_train)
    
    # X: None or 2D np.ndarray of shape (n_samples, n_inputs)
            #     Training input, i.e., X, the features.
            #     If None, it is assumed that only the target sequence matters (outputs)
            #     and simply a sequence of zeros will be fed in - matching the
            #     len(outputs).
            #     This is to be used in the case of generative mode.
            # y: 2D np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            #     Training output, i.e., y, the target.
    
    y_pred = esn.predict(X_test)

    # 3. Calculate the MC score
    mc_score = 0
    r2_scores = []
    for i in range(n_lags):
        # Discard initial transient phase (the 100 here) from test data for stable correlation
        corr, _ = pearsonr(y_test[100:, i], y_pred[100:, i]) # Discard initial transient. (statistic, p_value)
        r2_scores.append(corr**2)
        mc_score += corr**2
        
    return mc_score, np.array(r2_scores)


def evaluate_memory_capacity_from_connectome(connectome: np.ndarray, 
                                             h_params: Optional[Dict[str, Any]] = None,
                                             
                                             calculate_criticality:  Optional[bool] = False,
                                             calculate_info_dynamics: Optional[bool] = False, 
                                             
                                            #  spectral_radius: Optional[float] = 0.99, 
                                            # #  n_lags: Optional[int] = 50, 
                                            #  n_lags: Optional[int] = 50,
                                            #  train_len: Optional[int] = 4000, 
                                            #  test_len: Optional[int] = 1000, 
                                            #  n_runs: Optional[int] = 10, 
                                            #  input_scaling: Optional[float] = 1.0,
                                            #  regression_method: Optional[str] = "pinv",
                                            #  n_transient: Optional[int] = 0,
                                            #  leak_rate: Optional[float] = 1.0,
                                            #  bias: Optional[float] = 1.0,
                                            #  random_state: Optional[int] = 42, 
                                            #  calculate_criticality:  Optional[bool] = False,
                                            #  calculate_info_dynamics: Optional[bool] = False, 
                                            # #  mc_lengtxhs: Optional[List[int]] = None, 
                                             
 ) -> Dict[str, float]:
    
    # Random generator to generate the input sequence
    # rng = np.random.default_rng(h_params["random_state"])
    mc_values: List[float] = []
    all_states_for_metrics: List[np.ndarray] = []   
    # mc_values = []
    mc_values_indiv_array = []
    
    
    # h_params = {"spectral_radius": spectral_radius,  #### TODO: HAND A BIG HPARAMS DICT TO THIS FUNCTION!! (instead of doing it all one by one...)
    #             "n_lags": n_lags,
    #             "train_len": train_len,
    #             "test_len": test_len,
    #             "n_runs": n_runs,
    #             "input_scalign": input_scaling, 
    #             "regression_method": regression_method,
    #             "n_transient": n_transient,
    #             "leak_rate": leak_rate, 
    #             "bias": bias,
    #             "random_state": random_state,          
    # }
                
    
    # spectral_radius = h_params["spectral_radius"]
    # n_lags = h_params["n_lags"]
    # train_len = h_params["train_len"]
    # test_len = h_params["test_len"]
    n_runs = h_params["n_runs"]
    # input_scaling = h_params["input_scaling"]
    # regression_method = h_params["regression_method"]
    # n_transient = h_params["n_transient"]
    # leak_rate = h_params["leak_rate"]
    # bias = h_params["bias"]
    # random_state = h_params["random_state"]
    

#  mc_lengtxhs: Optional[List[int]] = None, 
                
    # Loop through the number of runs
    for _ in range(n_runs):
        # Get data for training and testing
    #     X_tr, Y_tr, X_te, Y_te = _generate_mc_dataset(train_len, test_len, n_lags, rng)
        
    #     # Get esn regressor with the connectome as weight matrix
    #     esn = ESNRegressor(
    #         W=connectome.copy(),
    #         spectral_radius=spectral_radius,
    #         n_transient=n_transient,
    #         input_scaling=input_scaling,
    #         leak_rate=leak_rate,
    #         bias=bias,
    #         regression_method=regression_method,
    #         # store_states_train=True,
    #         store_states_pred=True, 
    #     )
        
    #     esn.fit(X_tr, Y_tr)
    #     # check out esn. states_train_
 
    #     Y_pred = esn.predict(X_te)
    #     all_states_for_metrics.append(esn.states_pred_)
        
    #     # Vectorised Pearson r per column
    #     Yt = Y_te - Y_te.mean(axis=0, keepdims=True)
    #     Yp = Y_pred - Y_pred.mean(axis=0, keepdims=True)
    #     denom = (Yt.std(axis=0, ddof=0) * Yp.std(axis=0, ddof=0))
        
    #     r = (Yt * Yp).mean(axis=0) / denom
    #     r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    #     r2 = r**2
        
    #     mc = float(np.sum(r2))
        
        (mc_mean, individual_mc_values) = alternative_evaluate_mc(connectome, 
                                                                #   n_lags=n_lags, 
                                                                #   train_len=train_len, 
                                                                #   test_len=test_len,
                                                                  h_params=h_params, 
                                                                  )  
        mc_values.append(mc_mean) # which is only the mean value
        mc_values_indiv_array.append(np.array(individual_mc_values)) # 
        
    #     mc_result_dict = {} # or is this stupid? 
        
    #     # Calculate MC for specific lags
    #     if mc_lengths:
    #         # loop through the as-args-given-lags
    #         for length in mc_lengths:
    #             key = f"mc_length_{length}"
    #             if length <= len(r2):
    #                 value = float(np.sum(r2[:length]))
    #             else:
    #                 value = mc  # Fallback to total MC if requested length is too long
                
    #             # Store per-run values to be averaged later
    #             if key not in mc_result_dict:
    #                 mc_result_dict[key] = []
    #             mc_result_dict[key].append(value)
    
    # if mc_lengths:
    #     for key in [f"mc_length_{l}" for l in mc_lengths]:
    #         if key in mc_result_dict:
    #             mc_result_dict[key] = float(np.mean(mc_result_dict[key]))
    
    mc_values = np.array(mc_values)
    mc_values_for_indiv_lags = np.mean(np.array(mc_values_indiv_array), axis=0) # has shape (50,0) # TODO: Decide how to continue with this? Is this valuable enough to stay in this script?
    
    mc_result_dict = {
            "mc_mean": float(np.mean(mc_values)),
            "mc_std": float(np.std(mc_values)), 
            # "mean_mc_of_individual_runs": mc_values,   # This is currently not important. 
            "mc_values_for_indiv_lags": mc_values_for_indiv_lags,
            "h_params": h_params
                # {
                # "spectral_radius": spectral_radius, 
                # # "n_lags": n_lags,
                # "n_lag": n_lags,
                # "train_len": train_len,
                # "test_len": test_len,
                # "n_runs": n_runs,
                # "input_scaling": input_scaling,
                # "regression_method": regression_method,
                # "n_transient": n_transient,
                # "leak_rate": leak_rate,
                # "bias": bias,
                # "random_state": random_state,
                # },
            }
    
    #                                          calculate_criticality:  Optional[bool] = False,
    #                                          calculate_info_dynamics: Optional[bool] = False, 
    #                                          mc_lengths: Optional[List[int]] = None, 
                                             
                                             
    # TODO: This is not integrated yet - do this later on? -> maybe also in "alternate_evaluate_mc" function?     
    # What about entropy?                           
    if (calculate_criticality or calculate_info_dynamics) and all_states_for_metrics:
        # Concatenate states from all runs for a more robust estimation
        concatenated_states = np.vstack(all_states_for_metrics)
        
        if calculate_criticality:
            branching_ratio = _calculate_branching_ratio(concatenated_states)
            mc_result_dict['branching_ratio'] = branching_ratio
            
        if calculate_info_dynamics:
            info_dyn_results = _calculate_information_dynamics(concatenated_states)
            mc_result_dict.update(info_dyn_results)

    return mc_result_dict
    