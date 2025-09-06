# Modelled after damicelli's work. 
# -> Used at least for main_pipeline_2_gnm_esn_landscape.py and for main_pipeline_2.py (esn part)

import numpy as np
from typing import List, Tuple, Dict, Optional
from echoes.esn import ESNRegressor

import warnings

######
# Helper functions for criticality and information dynamics calculations
######

def _entropy(p: np.ndarray) -> float:
    """Calculates Shannon entropy H(X) for a probability distribution p."""
    # Filter out probabilities of zero to avoid log(0)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def _calculate_information_dynamics(states: np.ndarray, k: int = 1) -> Dict[str, float]:
    """
    Calculates average AIS and TE from first principles using only NumPy.
    
    Args:
        states: Reservoir state matrix (n_time_steps, n_nodes).
        k: History length. NOTE: This implementation is optimized for k=1.
        
    Returns:
        A dictionary with average AIS, TE, and their balance.
    """
    if k != 1:
        warnings.warn(f"NumPy implementation is optimized for k=1, but k={k}. Results may be slow or incorrect.")

    n_time_steps, n_nodes = states.shape
    
    # 1. Discretize states into integer bins (same as before)
    bins = np.quantile(states, [0, 0.25, 0.5, 0.75, 1.0])
    bins[0] = -np.inf
    bins[-1] = np.inf
    discretized_states = np.digitize(states, bins) - 1

    total_ais = 0.0
    total_te = 0.0
    
    for i in range(n_nodes):
        # --- 2. Calculate Active Information for node i: I(X_t ; X_{t-1}) ---
        # AI = H(X_t) + H(X_{t-1}) - H(X_t, X_{t-1})
        
        present_i = discretized_states[k:, i]
        past_i = discretized_states[:-k, i]

        # Estimate probabilities by counting unique outcomes
        _, p_present_i = np.unique(present_i, return_counts=True)
        _, p_past_i = np.unique(past_i, return_counts=True)
        _, p_joint_ai = np.unique(np.c_[present_i, past_i], axis=0, return_counts=True)

        # Normalize counts to get probabilities
        p_present_i = p_present_i / p_present_i.sum()
        p_past_i = p_past_i / p_past_i.sum()
        p_joint_ai = p_joint_ai / p_joint_ai.sum()
        
        # Calculate entropies
        h_present_i = _entropy(p_present_i)
        h_past_i = _entropy(p_past_i)
        h_joint_ai = _entropy(p_joint_ai)
        
        total_ais += h_present_i + h_past_i - h_joint_ai

        # --- 3. Calculate Transfer Entropy from all other nodes j to node i ---
        # TE(j->i) = H(I_t, J_t) + H(I_{t+1}, I_t) - H(I_t) - H(I_{t+1}, I_t, J_t)
        for j in range(n_nodes):
            if i == j: continue
            
            future_i = discretized_states[k:, i]
            # past_i is the same as present_i from the AI calculation above
            past_j = discretized_states[:-k, j]

            # Estimate joint probability distributions
            _, p_past_ij = np.unique(np.c_[past_i, past_j], axis=0, return_counts=True)
            _, p_future_i_past_i = np.unique(np.c_[future_i, past_i], axis=0, return_counts=True)
            _, p_joint_te = np.unique(np.c_[future_i, past_i, past_j], axis=0, return_counts=True)

            # Normalize
            p_past_ij = p_past_ij / p_past_ij.sum()
            p_future_i_past_i = p_future_i_past_i / p_future_i_past_i.sum()
            p_joint_te = p_joint_te / p_joint_te.sum()
            
            # Calculate entropies for the TE formula
            h_past_ij = _entropy(p_past_ij)
            h_future_i_past_i = _entropy(p_future_i_past_i)
            # h_past_i is the same as h_present_i from the AI calculation
            h_joint_te = _entropy(p_joint_te)
            
            te = h_past_ij + h_future_i_past_i - h_present_i - h_joint_te
            total_te += te
            
    avg_ais = total_ais / n_nodes
    avg_te = total_te / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    info_balance = avg_ais / (avg_te + 1e-9)
    
    return {
        "avg_active_info": avg_ais,
        "avg_transfer_entropy": avg_te,
        "info_balance": info_balance
    }

def _calculate_branching_ratio(states: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculates the branching ratio (sigma) to estimate criticality.
    A value of 1.0 indicates critical dynamics.
    
    Args:
        states: Reservoir state matrix (n_time_steps, n_nodes).
        threshold: Activity threshold to consider a neuron 'active'.
                   0.0 means any non-zero state is active.
                   
    Returns:
        The branching ratio, sigma.
    """
    if threshold == 0.0:
        # Use the median as a more robust threshold if not specified
        threshold = np.median(np.abs(states))
        if threshold == 0: threshold = 1e-6 # Avoid division by zero if states are all zero

    # Binarize activity based on the threshold
    active_states = np.abs(states) > threshold
    
    n_time_steps, n_nodes = active_states.shape
    
    ancestors = np.sum(active_states[:-1], axis=1)
    descendants = np.sum(active_states[1:], axis=1)
    
    # Avoid division by zero for silent steps
    valid_indices = np.where(ancestors > 0)[0]
    if len(valid_indices) == 0:
        return 0.0  # Network was inactive
        
    ratios = descendants[valid_indices] / ancestors[valid_indices]
    
    return np.mean(ratios)


def _generate_mc_dataset(train_len: int, test_len: int, n_lags: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_len = int(train_len + test_len + n_lags + 100)
    # print("REMOVE AGAIN, in test_memory_capacity_weighted.py: total_len", total_len) -> THIS IS USED.
    seq = rng.uniform(-0.5, 0.5, size=(total_len,))
    def build_targets(x: np.ndarray, lags: int) -> np.ndarray:
        T = len(x) - lags
        targets = np.zeros((T, lags), dtype=float)
        for i in range(lags):
            targets[:, i] = x[lags - (i + 1) : - (i + 1) if i + 1 > 0 else None]
        return targets
    Y_full = build_targets(seq, n_lags)
    start_train = 100
    end_train = start_train + train_len
    X_train = seq[start_train : end_train].reshape(-1, 1)
    Y_train = Y_full[start_train : end_train]
    X_test = seq[end_train : end_train + test_len].reshape(-1, 1)
    Y_test = Y_full[end_train : end_train + test_len]
    return X_train, Y_train, X_test, Y_test


def evaluate_memory_capacity_from_connectome(connectome: np.ndarray, 
                                             spectral_radius: Optional[float] = 0.99, 
                                             n_lags: Optional[int] = 50, 
                                             train_len:  Optional[int] = 4000, 
                                             test_len: Optional[int] = 1000, 
                                             n_runs: Optional[int] = 10, 
                                             input_scaling: Optional[float] = 1.0,
                                             regression_method: Optional[str] = "pinv",
                                             n_transient: Optional[int] = 0,
                                             leak_rate: Optional[float] = 1.0,
                                             bias: Optional[float] = 1.0,
                                             random_state: Optional[int] = 42, 
                                             calculate_criticality:  Optional[bool] = False,
                                             calculate_info_dynamics: Optional[bool] = False, 
                                             
 ) -> Dict[str, float]:
    
    # Random generator to generate the input sequence
    rng = np.random.default_rng(random_state)
    mc_values: List[float] = []
    all_states_for_metrics: List[np.ndarray] = []
    
    # Loop through the number of runs
    for _ in range(n_runs):
        # Get data for training and testing
        X_tr, Y_tr, X_te, Y_te = _generate_mc_dataset(train_len, test_len, n_lags, rng)
        
        # Get esn regressor with the connectome as weight matrix
        esn = ESNRegressor(
            W=connectome.copy(),
            spectral_radius=spectral_radius,
            n_transient=n_transient,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            bias=bias,
            regression_method=regression_method,
            # store_states_train=True,
            store_states_pred=True, 
        )
        
        esn.fit(X_tr, Y_tr)
        # check out esn. states_train_
 
        Y_pred = esn.predict(X_te)
        all_states_for_metrics.append(esn.states_pred_)
        
        # Vectorised Pearson r per column
        Yt = Y_te - Y_te.mean(axis=0, keepdims=True)
        Yp = Y_pred - Y_pred.mean(axis=0, keepdims=True)
        denom = (Yt.std(axis=0, ddof=0) * Yp.std(axis=0, ddof=0))
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     r = (Yt * Yp).mean(axis=0) / denom
        #     r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        r = (Yt * Yp).mean(axis=0) / denom
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        r2 = r**2
        
        mc = float(np.sum(r2))
        mc_values.append(mc)
    
    mc_result_dict = {
            "mc_mean": float(np.mean(mc_values)), 
            "mc_std": float(np.std(mc_values)),
            "mean_mc_of_individual_runs": mc_values,  
            # "r2_array_from_0_to_n_lags_minus_1": r2, # TODO: R2 array could be added, but code currently "nearly stops", when added? (TODO_R2_array for searching)
            "hparams": {
                "spectral_radius": spectral_radius,
                "n_lags": n_lags,
                "train_len": train_len,
                "test_len": test_len,
                "n_runs": n_runs,
                "input_scaling": input_scaling,
                "regression_method": regression_method,
                "n_transient": n_transient,
                "leak_rate": leak_rate,
                "bias": bias,
                "random_state": random_state,
                },
            }
    
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
    
    #     result = {
    #     "mc_mean": np.mean(mc_of_runs),
    #     "mc_std": np.std(mc_of_runs),
    #     "mean_mc_of_individual_runs": list(mc_of_runs),
    #     "mc_r2_5_to_25": mc_r2_5_to_25,
    #     "hparams": {
    #         "spectral_radius": spectral_radius,
    #             "n_lags": n_lags,
    #             "train_len": train_len,
    #             "test_len": test_len,
    #             "n_runs": n_runs,
    #             "input_scaling": input_scaling,
    #             "regression_method": regression_method,
    #             "n_transient": n_transient,
    #             "leak_rate": leak_rate,
    #             "bias": bias,
    #             "random_state": random_state,
    #             }
    #     }

    # return result
        