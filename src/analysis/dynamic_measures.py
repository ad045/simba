import numpy as np
from netneurotools.metrics import diffusion_efficiency
from scipy.linalg import solve_continuous_lyapunov

import networkx as nx

from src.analysis.from_fatemeh import compute_eigenvalue_gap as compute_spectral_gap_fatemeh # needed for next script

import numpy as np
import scipy.signal as sig
import scipy
from numba import njit

from scipy.sparse.csgraph import connected_components


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
    
    # Check if network has disconnected components
    # For directed networks, you need strong connectivity
    n_components, labels = connected_components(A, directed=True, connection='strong')
    
    if n_components > 1:
        print(f"Warning: Network has {n_components} strongly connected components")
        # Return NaN for disconnected networks
        return 0 #  np.nan
        # Option 2: Only compute on largest component
        # largest_component = np.argmax(np.bincount(labels))
        # mask = labels == largest_component
        # A = A[np.ix_(mask, mask)]
        
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
    n_10 = np.sum(np.cumsum(np.sort(ac)[::-1]) <= 0.1 * sum(ac)) # + 1

    return {"avg": np.mean(ac),
            "std": np.std(ac),
            "max": np.max(ac),
            "n_nodes_90_percent": n_90,
            "n_nodes_50_percent": n_50,
            "n_nodes_10_percent": n_10}


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
    n_10 = np.sum(np.cumsum(np.sort(node_energy)[::-1]) <= 0.1 * energy) # + 1

    return {"total": energy,
            "std": std_node_energy,
            "max": np.max(node_energy),
            "n_nodes_90_percent": n_90,
            "n_nodes_50_percent": n_50,
            "n_nodes_10_percent": n_10,
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
        "local_skewness": scipy.stats.skew(meta_local, nan_policy='omit'),
        "local_kurtosis": scipy.stats.kurtosis(meta_local, nan_policy='omit')
    }
    
    
    
    
def compute_synchronizability_eigenratio(A): # Claude
    """
    Compute eigenratio R = λ_N / λ_2 of the graph Laplacian.
    Lower R = easier to synchronize
    """
    G = nx.from_numpy_array(A)
    
    # Compute Laplacian eigenvalues
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(eigenvalues)
    
    # λ_2 is the algebraic connectivity (Fiedler value)
    # λ_N is the largest eigenvalue
    lambda_2 = eigenvalues[1]  # First non-zero
    lambda_N = eigenvalues[-1]
    
    # Eigenratio
    R = lambda_N / lambda_2 if lambda_2 > 1e-10 else np.inf
    
    return {"eigenratio": R, "lambda_2": lambda_2, "lambda_N": lambda_N}


def algebraic_connectivity_nx(adjacency_matrix): # Claude
    G = nx.from_numpy_array(adjacency_matrix)
    return nx.algebraic_connectivity(G)


def kuramoto_synchronization(adjacency_matrix, n_steps=1000): # Claude
    """
    Simulate Kuramoto oscillators on the network
    Returns: final order parameter (0=desynchronized, 1=synchronized)
    """
    n_nodes = adjacency_matrix.shape[0]
    
    # Initialize random phases
    theta = np.random.uniform(0, 2*np.pi, n_nodes)
    omega = np.random.normal(0, 0.1, n_nodes)  # Natural frequencies
    
    # Coupling strength
    K = 1.0
    dt = 0.01
    
    # Simulate
    for _ in range(n_steps):
        # Kuramoto equation
        coupling = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                coupling[i] += adjacency_matrix[i,j] * np.sin(theta[j] - theta[i])
        
        theta += dt * (omega + K * coupling / n_nodes)
    
    # Order parameter
    r = np.abs(np.mean(np.exp(1j * theta)))
    return r

def community_synchronization_vulnerability(adjacency_matrix): # Claude
    """
    Measures how easily synchronization can spread between communities
    High = cascades easily (epilepsy risk?)
    
    This function:
    1. Detects communities using the Louvain algorithm
    2. Calculates within-community and between-community coupling strengths
    3. Returns vulnerability score (between/within ratio)
    
    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        Network adjacency matrix
    
    Returns:
    --------
    vulnerability : float
        Ratio of between-community to within-community coupling
    communities : list of lists
        Detected communities (node indices)
    """
    import networkx as nx
    # import nx.community as community_louvain
    from networkx.algorithms.community.louvain import louvain_communities # louvain_partitions
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Detect communities using Louvain algorithm
    communities = louvain_communities(G) # louvain_partitions(G)
    
    # Convert partition dict to list of communities
    # num_communities = max(partition.values()) + 1
    # communities = [[] for _ in range(num_communities)]
    # for node, comm_id in partition.items():
    #     communities[comm_id].append(node)
    
    # Within-community coupling strength
    within_strength = []
    # Between-community coupling strength  
    between_strength = []
    
    communities_as_lists = [list(c) for c in communities]
    for c in communities_as_lists: 
        for i, comm_i in enumerate(communities_as_lists):
            for j, comm_j in enumerate(communities_as_lists):
                # comm_i_2 = [c for c in comm_i]
                # comm_j_2 = [c for c in comm_j]
                coupling = adjacency_matrix[np.ix_(comm_i, comm_j)].sum()

            if i == j:
                within_strength.append(coupling)
            else:
                between_strength.append(coupling)
    
    # Vulnerability score
    vulnerability = np.mean(between_strength) / np.mean(within_strength)
    
    # return # vulnerability, communities
    return {"vulnerability": vulnerability, "n_communities": len(communities)}