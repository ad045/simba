
# Helper functions for calculating information dynamics and criticality in reservoir states

import numpy as np 
from typing import List, Tuple, Dict, Optional
import warnings


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

