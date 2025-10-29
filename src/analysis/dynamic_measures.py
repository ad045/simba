import numpy as np
from netneurotools.metrics import diffusion_efficiency
from scipy.linalg import solve_continuous_lyapunov

import networkx as nx

from src.analysis.from_fatemeh import compute_eigenvalue_gap as compute_spectral_gap_fatemeh # needed for next script

import numpy as np
import scipy.signal as sig
import scipy
from numba import njit



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
 
 
# def average_controllability(A): # Not sure if this works correctly??
#     """
#     Average controllability from Gu et al. 2015
#     Formula: AC_i = Tr(W_i) where W_i = ∫_0^∞ e^(At) B_i B_i^T e^(A^T t) dt
#     Solved via Lyapunov equation: A W_i + W_i A^T + B_i B_i^T = 0
#     """
#     from scipy.linalg import solve_continuous_lyapunov
    
#     N = A.shape[0]
#     controllability = np.zeros(N)
    
#     for i in range(N):
#         # B_i is a column vector with 1 at position i, 0 elsewhere
#         B_i = np.zeros((N, 1))
#         B_i[i, 0] = 1
        
#         # Solve Lyapunov equation: A W + W A^T + B B^T = 0
#         try:
#             W_i = solve_continuous_lyapunov(A, -B_i @ B_i.T)
#             controllability[i] = np.trace(W_i)
#         except:
#             controllability[i] = np.nan
    
#     return controllability
 
 
#  "spectral_radius", 
#             # kernel_rank",
#             "spectral_gap",
#             "diffusion_efficiency",
#             "average_controllability",
# "calculate_nct_energies"
# "calculate_nct_control"


from nctpy.utils import matrix_normalization
from nctpy.energies import sim_state_eq
from nctpy.metrics import ave_control
from nctpy.energies import get_control_inputs, integrate_u

def calculate_nct_control(A, 
                      T=20, # time horizon
                      ):
    system = 'discrete'  # 'continuous' or 'discrete'
    A_norm = matrix_normalization(A=A, c=1, system=system)
    # n_nodes = A.shape[0]
    # U = np.zeros((n_nodes, T))  # the input to the system
    # U[:,0] = 1  # impulse, 1 input at the first time point delivered to all nodes. "0" (or 15 or anything) is the point when the stimulus is given
    # B = np.eye(n_nodes)  # uniform full control set
    # x0 = np.ones((n_nodes, 1))  # initial state, all nodes set to 1 unit of neural activity
    # x = sim_state_eq(A_norm=A_norm, B=B, x0=x0, U=U, system=system)

    ac = ave_control(A_norm=A_norm, system=system)
    n_90 = np.sum(np.cumsum(np.sort(ac)[::-1]) <= 0.9 * sum(ac)) # + 1
    # print('number of nodes accounting for 90% of total control =', n_90)   
    n_50 = np.sum(np.cumsum(np.sort(ac)[::-1]) <= 0.5 * sum(ac)) # + 1      
    
    return {"avg": np.mean(ac),
            "std": np.std(ac),
            "n_nodes_90_percent": n_90,
            "n_nodes_50_percent": n_50}


def calculate_nct_energies(A): 
    system = 'continuous'
    A_norm = matrix_normalization(A=A, c=1, system=system) # normalization is different than for discrete
    n_nodes = A.shape[0]
    
    # define initial and target states as random patterns of activity
    np.random.seed(42)  # for reproducibility
    x0 = np.random.rand(n_nodes, 1)  # initial state
    xf = np.random.rand(n_nodes, 1)  # target state

    # set parameters
    T = 1  # time horizon
    rho = 1  # mixing parameter for state trajectory constraint
    B = np.eye(n_nodes)  # uniform full control set
    S = np.eye(n_nodes)  # nodes in state trajectory to be constrained

    # get the state trajectory (x) and the control inputs (u)
    x, u, n_err = get_control_inputs(A_norm=A_norm, T=T, B=B, x0=x0, xf=xf, system=system, rho=rho, S=S)
    
    node_energy = integrate_u(u)
    # summarize nodal energy to get control energy
    energy = np.sum(node_energy)
    std_node_energy = np.std(node_energy)
    
    # get the number of nodes whose sum of control energy accounts for 90% of the total energy
    n_90 = np.sum(np.cumsum(np.sort(node_energy)[::-1]) <= 0.9 * energy) # + 1
    # print('number of nodes accounting for 90% of total energy =', n_90)
    n_50 = np.sum(np.cumsum(np.sort(node_energy)[::-1]) <= 0.5 * energy) # + 1         
    
    return {"energy_total": energy,
            "std_node_energy": std_node_energy,
            "n_nodes_90_percent": n_90,
            "n_nodes_50_percent": n_50,
            }


# from from_francisco import evaluate_network_2
from src.analysis.from_francisco import evaluate_network_2

def calculate_metastability(A): 

    # data_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/00_connectomes_50.npy"
    # c = np.load(data_path)[0]

    meta_global, meta_local, _ = evaluate_network_2(W=A)  # takes like 6.6 seconds... 
    # print("META LOCAL: ", meta_local)
    # print("META LOCAL MEAN: ", np.mean(meta_local))
    return{
        "global": meta_global, 
        "local_mean": np.nanmean(meta_local), 
        "local_std": np.nanstd(meta_local), 
        "local_kurtosis": scipy.stats.kurtosis(meta_local, nan_policy='omit')
    }