import numpy as np
from netneurotools.metrics import diffusion_efficiency
from scipy.linalg import solve_continuous_lyapunov

import networkx as nx

def spectral_radius(A):
   eigenvalues, _ = np.linalg.eig(A)
   return np.max(np.abs(eigenvalues))
 
def spectral_gap(A):
   eigenvalues, _ = np.linalg.eig(A)
   eigs_sorted = np.sort(np.abs(eigenvalues))[::-1]
   return eigs_sorted[0] - eigs_sorted[1]


def calculate_global_efficiency(G) -> float:
    return nx.global_efficiency(G)
 

def calculate_diffusion_efficiency(A): 
    diff_efficiency_coeff, _ = diffusion_efficiency(A)
    return diff_efficiency_coeff


# def average_controllability(A):
#     return average_controllability(A)
 
 
def average_controllability(A): # Not sure if this works correctly??
    """
    Average controllability from Gu et al. 2015
    Formula: AC_i = Tr(W_i) where W_i = ∫_0^∞ e^(At) B_i B_i^T e^(A^T t) dt
    Solved via Lyapunov equation: A W_i + W_i A^T + B_i B_i^T = 0
    """
    from scipy.linalg import solve_continuous_lyapunov
    
    N = A.shape[0]
    controllability = np.zeros(N)
    
    for i in range(N):
        # B_i is a column vector with 1 at position i, 0 elsewhere
        B_i = np.zeros((N, 1))
        B_i[i, 0] = 1
        
        # Solve Lyapunov equation: A W + W A^T + B B^T = 0
        try:
            W_i = solve_continuous_lyapunov(A, -B_i @ B_i.T)
            controllability[i] = np.trace(W_i)
        except:
            controllability[i] = np.nan
    
    return controllability
 
 
#  "spectral_radius", 
#             # kernel_rank",
#             "spectral_gap",
#             "diffusion_efficiency",
#             "average_controllability",