import numpy as np
from netneurotools.metrics import diffusion_efficiency

import scipy
import scipy.signal as sig
from scipy.linalg import solve_continuous_lyapunov, schur
from scipy.sparse.csgraph import connected_components

import networkx as nx
from networkx.algorithms.community.louvain import louvain_communities

from src.analysis.from_fatemeh import compute_eigenvalue_gap as compute_spectral_gap_fatemeh # needed for next script

from numba import njit

from bct import participation_coef
from kuramoto import Kuramoto # damicelli


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
        return np.nan
        
    diff_efficiency_coeff, _ = diffusion_efficiency(A)
    return diff_efficiency_coeff




from nctpy.utils import matrix_normalization
from nctpy.energies import sim_state_eq
from nctpy.metrics import ave_control
from nctpy.energies import get_control_inputs, integrate_u

def calculate_nct_control(A, 
                      T=20, # time horizon
                      ):
    system = 'discrete'  # 'continuous' or 'discrete'
    A_norm = matrix_normalization(A=A, c=1, system=system)

    ac = ave_control(A_norm=A_norm, system=system)
    n_90 = np.sum(np.cumsum(np.sort(ac)[::-1]) <= 0.9 * sum(ac)) # + 1
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

    meta_global, meta_local, _ = evaluate_network_2(W=A)  
    
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


def kuramoto_synchronization(A, coupling=1.0, dt=0.01, T=10.0, seed=42): # wrapper by claude
    """
    Kuramoto synchronization via the `kuramoto` library.
    Uses scipy.solve_ivp (RK45) internally.
    Returns: final order parameter r ∈ [0, 1]
    """
    np.random.seed(seed)
    n = A.shape[0]
    nat_freqs = np.random.normal(0, 0.1, n)

    model = Kuramoto(
        coupling=coupling,
        dt=dt,
        T=T,
        natfreqs=nat_freqs,
    )
    # model.run() expects an adjacency matrix and returns phases (T_steps x N)
    phases = model.run(adj_mat=A)

    # Order parameter at each timestep
    r = np.abs(np.mean(np.exp(1j * phases), axis=1))

    return {
        "r_final": float(r[-1]),
        "r_mean": float(np.mean(r[len(r)//2:])),  # mean over 2nd half (after transient)
        "r_std": float(np.std(r[len(r)//2:])),     # this IS metastability
    }
    
    

def kuramoto_averaged_synchronization(A, coupling=1.0, dt=0.01, T=20.0,  # claude
                              n_trials=10, T_discard_fraction=0.5, seed=42):
    """
    Kuramoto synchronization averaged over multiple trials
    for stable estimates.
    
    Args:
        A: adjacency matrix
        coupling: coupling strength K
        dt: integration timestep
        T: total simulation time per trial
        n_trials: number of independent trials to average over
        T_discard_fraction: fraction of simulation to discard as transient
        seed: base random seed (each trial uses seed + i)
    
    Returns:
        dict with r_final, r_mean, r_std (metastability), all averaged
    """
    n = A.shape[0]
    
    r_finals = []
    r_means = []
    r_stds = []
    
    for i in range(n_trials):
        np.random.seed(seed + i)
        nat_freqs = np.random.normal(0, 0.1, n)
        
        model = Kuramoto(coupling=coupling, dt=dt, T=T, natfreqs=nat_freqs)
        phases = model.run(adj_mat=A)
        
        # Order parameter timeseries
        r = np.abs(np.mean(np.exp(1j * phases), axis=1))
        
        # Discard transient
        start = int(len(r) * T_discard_fraction)
        r_steady = r[start:]
        
        r_finals.append(r[-1])
        r_means.append(np.mean(r_steady))
        r_stds.append(np.std(r_steady))
    
    return {
        "r_final": float(np.mean(r_finals)),
        "r_mean": float(np.mean(r_means)),
        "r_std": float(np.mean(r_stds)),          # metastability (mean across trials)
        "r_mean_se": float(np.std(r_means) / np.sqrt(n_trials)),  # standard error
        "r_std_se": float(np.std(r_stds) / np.sqrt(n_trials)),    # SE of metastability
    }

def community_synchronization_vulnerability(A): # claude: corrected. 
    """
    Measures cross-community coupling via:
      1. Between/within coupling ratio — global vulnerability score (corrected).
    """
    G = nx.from_numpy_array(A)
    communities = list(louvain_communities(G))

    if len(communities) <= 1:
        return {"vulnerability": 0.0, "n_communities": len(communities)}

    # # Build community label vector for bctpy
    # ci = np.zeros(A.shape[0], dtype=int)
    # for idx, comm in enumerate(communities):
    #     for node in comm:
    #         ci[node] = idx

    # Between/within coupling ratio (corrected bug from original)
    within_total = 0.0
    between_total = 0.0
    for i, comm_i in enumerate(communities):
        nodes_i = list(comm_i)
        for j, comm_j in enumerate(communities):
            nodes_j = list(comm_j)
            coupling = A[np.ix_(nodes_i, nodes_j)].sum()
            if i == j:
                within_total += coupling
            else:
                between_total += coupling

    vulnerability = between_total / within_total if within_total > 0 else np.inf

    return {
        "value": vulnerability,
        "n_communities": len(communities),
    }


def participation_coefficient(A):
    """
    Measures cross-community coupling via:
      1. Participation coefficient (Guimerà & Amaral, 2005) — per-node measure
         of how distributed connections are across communities. P_i = 1 - Σ_s (k_is/k_i)²
    """
    G = nx.from_numpy_array(A)
    communities = list(louvain_communities(G))

    if len(communities) <= 1:
        return {"mean": 0.0, "std": 0.0}

    # Build community label vector for bctpy
    ci = np.zeros(A.shape[0], dtype=int)
    for idx, comm in enumerate(communities):
        for node in comm:
            ci[node] = idx

    # Participation coefficient (standard neuroscience measure)
    pc = participation_coef(A, ci)

    return {
        "mean": float(np.mean(pc)),
        "std": float(np.std(pc)),
    }
    

def departure_from_normality_schur(A): # Not fatemeh's version, but more numerically stable via Schur decomposition.
    """
    Henrici departure from normality via Schur decomposition.
    
    For M = QTQ*, the departure equals ||T_off||_F / ||M||_F,
    where T_off is the strictly upper-triangular part of T.
    
    This avoids the numerically unstable subtraction
    ||M||_F^2 - sum|lambda_i|^2 from the eigenvalue-based formula.
    """
    A = np.array(A, dtype=np.complex128)

    norm_F = np.linalg.norm(A, 'fro')
    if norm_F < 1e-12:
        return 0.0
    
    # Schur decomposition: M = Q T Q^H
    T, _ = schur(A, output='complex')

    # Strictly upper-triangular part = non-normal component
    T_off = np.triu(T, k=1)
    
    return float(np.linalg.norm(T_off, 'fro') / norm_F)
