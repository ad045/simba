import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_comm
from networkx.algorithms import smallworld
from networkx.algorithms.community import modularity, louvain_communities

from bct import density_und

# from netneurotools import modularity # -> clashes with louvain (networkx) implementation

# ["density", "wiring_cost", "shortest_path_distance", "compute_structural_complexity",
#  "n_connected_components", "omega", "topological_distance", "resistance_distance", 
#  "propagation_distance", "propagation_efficiency"
# 
# "modularity",  "avg_degree", "transitivity", "wiring_cost", "char_path_length", 
# avg_clustering, degree_assortativity


def largest_component_char_path_length(G) -> float: # or np.nan
    """
    Characteristic path length on the largest connected component (NaN if <2 nodes).
    Returns a float or np.nan. 
    """
    if G.number_of_nodes() == 0:
        return np.nan
    
    # Pick largest connected component
    components = list(nx.connected_components(G))
    if not components:
        return np.nan
    H = G.subgraph(max(components, key=len)).copy()
    if H.number_of_nodes() < 2:
        return np.nan
    return nx.average_shortest_path_length(H)


def count_components_nx(A) -> int:
    G = nx.from_numpy_array(np.array(A))
    return nx.number_connected_components(G)


def _get_rich_nodes(G, rich_nodes=None, top_percent=0.20):
    """
    Return a set of 'rich' nodes.
        - If rich_nodes provided (iterable of node indices), use that.
        - Else pick top `top_percent` by degree (ties broken arbitrarily).
    """
    if rich_nodes is not None:
        return set(rich_nodes)
    
    # default: top X% by degree
    deg = dict(G.degree())
    if len(deg) == 0:
        return set()
    k = max(1, int(np.ceil(top_percent * len(deg))))

    # sort nodes by degree descending and take top k
    sorted_nodes = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)
    return set(n for n, _ in sorted_nodes[:k])


def old_communicability(A_bin, mode="estrada_scaled", beta=None, t=1.0) -> float: # or np.nan
    """
    Compute average off-diagonal communicability in three modes:
      - estrada:        average of expm(A)
      - estrada_scaled: average of expm(b * A) with b = 1/max(1, λ_max) -> This method is hopefully numerically stable.
      - heat:           average of exp(-t * L_norm) (heat kernel)
    Uses eigen-decomposition for the ‘estrada*’ modes to avoid expm overflows.
    """
    
    try: 
        n = A_bin.shape[0]
        
        # Helper to average off-diagonal entries
        def avg_offdiag(M):
            np.fill_diagonal(M, 0.0)
            return M.sum() / (n*(n-1))

        if mode in ("estrada", "estrada_scaled"):
            # Compute symmetric A
            A = (A_bin + A_bin.T) / 2.0
            # Eigen-decompose once
            eigvals, eigvecs = np.linalg.eigh(A)
            
            lam_max = eigvals[-1]   # get largest eigenvalue - works since eigh returns sorted eigenvalues
            if lam_max <= 0:
                # A has no positive spectrum ⇒ exp(bA) = I ⇒ no off-diag mass
                return 0.0

            # Scaling factor for estrada_scaled
            b = beta if (beta is not None) else 1.0/lam_max

            # Build the exponent safely
            max_log = np.log(np.finfo(float).max)  # ≈ 709.78 for float64
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                args = b * eigvals
                args = np.clip(args, -max_log, +max_log)
                exp_eigs = np.exp(args)

            # Ssanitize e^{-} spectrum
            exp_eigs = np.nan_to_num(
                exp_eigs,
                posinf=np.finfo(float).max,
                neginf=0.0,
                nan=0.0
            )

            # Reconstruct *and* silence any matmul warnings
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                C = (eigvecs * exp_eigs) @ eigvecs.T

            # Final cleanup
            C = np.nan_to_num(
                C,
                posinf=np.finfo(float).max,
                neginf=0.0,
                nan=0.0
            )

            return avg_offdiag(C)

        elif mode == "heat":

            # Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
            deg = A_bin.sum(axis=1)
            inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
            S = np.diag(inv_sqrt)
            L = np.eye(n) - S @ A_bin @ S

            # Hope: It's numerically stable to call expm here because L's spectrum lies in [0,2]
            from scipy.linalg import expm
            C = expm(-t * L)
            return avg_offdiag(C)

        else:
            raise ValueError("mode must be 'estrada', 'estrada_scaled', or 'heat'.")
    
    except Exception as e:
        print(f"Error computing communicability: {e}")
        return np.nan




def calculate_modularity(G) -> float: 
    # comms = list(nx_comm.greedy_modularity_communities(G))
    # return nx_comm.modularity(G, comms) if len(comms) > 1 else 0.0
    louvain_comms = louvain_communities(G)
    # print("louvain_comms:", louvain_comms)
    return modularity(G, louvain_comms) if len(louvain_comms) > 1 else 0.0


def calculate_avg_degree(G) -> float:
    return float(np.mean([d for _, d in G.degree()])) if G.number_of_nodes() > 0 else np.nan 

def calculate_transitivity(G) -> float:
    return nx.transitivity(G) if G.number_of_nodes() > 0 else np.nan # is if statment ok?


def calculate_avg_clustering(G) -> float:
    return nx.average_clustering(G) if G.number_of_nodes() > 0 else np.nan

def calculate_degree_assortativity(G) -> float:
    return nx.degree_assortativity_coefficient(G) if G.number_of_nodes() > 0 else np.nan


# def calculate_avg_edge_distance(G, distance_matrix) -> float: # similar to wiring cost...
#     edge_d = [distance_matrix[i, j] for i, j in G.edges()]
#     avg_dist = float(np.mean(edge_d)) if edge_d else np.nan
#     return avg_dist

def calculate_wiring_cost(G, distance_matrix) -> float: # similar to avg_edge_distance... 
    edge_d = [distance_matrix[i, j] for i, j in G.edges()]
    wiring_cost = float(np.sum(edge_d)) if edge_d else 0.0
    return wiring_cost


def calculate_proportion_long_range_connections(A, distance_matrix, thresholds=[0.1, 0.3,
                                                                                0.3956, # 0.356,
                                                                                0.5]) -> dict:
    """
    Fraction of actual edges whose distance exceeds the given quantile
    of all possible pairwise distances.
    
    Returns:
        - fraction of edges that are "long-range" (float)
    """
    # All possible pairwise distances (upper triangle, excluding diagonal)
    triu = np.triu_indices_from(distance_matrix, k=1)
    all_distances = distance_matrix[triu]
    thresholds_dict = {}
    for quantile in thresholds:
        threshold = np.quantile(all_distances, quantile)

        # Distances of actual edges
        edges = np.array(np.where(np.triu(A) > 0)).T
        if len(edges) == 0:
            thresholds_dict[quantile] = 0.0
            continue
        edge_distances = distance_matrix[edges[:, 0], edges[:, 1]]

        thresholds_dict[f"{quantile}"] = float(np.mean(edge_distances > threshold))

    return thresholds_dict



def calculate_char_path_length(G) -> float: # or np.nan
    """
    Characteristic path length on the largest connected component (NaN if <2 nodes).
    Returns a float or np.nan. 
    """
    if G.number_of_nodes() == 0:
        return np.nan
    
    # Pick largest connected component
    components = list(nx.connected_components(G))
    if not components:
        return np.nan
    H = G.subgraph(max(components, key=len)).copy()
    if H.number_of_nodes() < 2:
        return np.nan
    return nx.average_shortest_path_length(H)


def calculate_degree_gini(A):
    degrees = np.sum(A, axis=1)
    degrees = np.sort(degrees)
    n = len(degrees)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * degrees)) / (n * np.sum(degrees)) - (n + 1) / n



# def _get_rich_nodes(G, rich_nodes=None, top_percent=0.20):
#     """
#     Return a set of 'rich' nodes.
#         - If rich_nodes provided (iterable of node indices), use that.
#         - Else pick top `top_percent` by degree (ties broken arbitrarily).
#     """
#     if rich_nodes is not None:
#         return set(rich_nodes)
    
#     # default: top X% by degree
#     deg = dict(G.degree())
#     if len(deg) == 0:
#         return set()
#     k = max(1, int(np.ceil(top_percent * len(deg))))

#     # sort nodes by degree descending and take top k
#     sorted_nodes = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)
#     return set(n for n, _ in sorted_nodes[:k])

# def calculate_richclub_n_edges(G, rich_nodes) -> int:
#     rich_nodes = _get_rich_nodes(G, rich_nodes_global, top_percent=rich_top_percent)
#     rich_edges = [(u, v) for (u, v) in G.edges() if u in rich_nodes and v in rich_nodes]
#     n_rich_edges = len(rich_edges)

# def calculate_richclub_avg_length(G, rich_neodes, distance_matrix) -> float:
#     rc_lengths = [distance_matrix[u, v] for (u, v) in rich_edges]
#     avg_rc_length = float(np.mean(rc_lengths)) if rc_lengths else np.nan


# This one runs for the GNM creation pipeline. 
def analyze_connectomes(connectomes,  
                        distance_matrix,
                        comm_mode="estrada_scaled", 
                        beta=None, t=1.0,
                        rich_nodes_global=None, 
                        rich_top_percent=0.20):
    """
    Analyze a set of connectomes for various global graph measures.

    Inputs: 
        - connectomes: either list of (n,n) arrays OR array of shape (k,n,n) or (n,n,k)
        - distance_matrix: (n,n) Euclidean distances
        - comm_mode: 'estrada' | 'estrada_scaled' | 'heat'
        - rich_nodes_global: optional iterable of node indices (same for all nets).
                        If None, uses top `rich_top_percent` by degree per-network.
        - rich_top_percent: fraction used when rich_nodes_global is None.
    """
    # Get connectome from different inputs 
    arr = np.asarray(connectomes)  # normalize input shape to (k, n, n)
    if arr.ndim == 2:             
        # Single connectome case
        arr = arr[None, ...]       # (1, n, n)
    elif arr.ndim == 3:
        # Multiple connectomes case 
        # if (n,n,k), move axes to (k,n,n)
        if arr.shape[0] == arr.shape[1]:
            arr = np.transpose(arr, (2, 0, 1))
        # else assume already (k,n,n)
    else:
        raise ValueError("connectomes must be (n,n), (k,n,n), or (n,n,k).")

    # Check dimensions and ensure square shape 
    k, n, _ = arr.shape
    if distance_matrix.shape != (n, n):
        raise ValueError(f"distance_matrix must be {(n,n)}, got {distance_matrix.shape}")

    # Itterate through all available connectomes and compute measures
    out = []
    for idx in range(k):
        A = np.array(arr[idx], dtype=float)

        # Ensure symmetry & zero diagonal
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0.0)

        # Binary graph for global metrics
        A_bin = (A != 0).astype(float)
        G = nx.from_numpy_array(A_bin)

        # Calculate metrics 
        # Communicability (chosen mode)
        # try: 
        avg_comm = old_communicability(A_bin, mode=comm_mode, beta=beta, t=t)
        # except: 
        #     avg_comm = np.nan
        
        # Global efficiency
        glob_eff = nx.global_efficiency(G)

        # Modularity (greedy)
        comms = list(nx_comm.greedy_modularity_communities(G))
        modu = nx_comm.modularity(G, comms) if len(comms) > 1 else 0.0 # is if statement ok? 

        # Other global properties
        avg_clust = nx.average_clustering(G) if G.number_of_nodes() > 0 else np.nan # is if statment ok?
        avg_deg = float(np.mean([d for _, d in G.degree()])) if G.number_of_nodes() > 0 else np.nan # is if statment ok?
        trans = nx.transitivity(G) if G.number_of_nodes() > 0 else np.nan # is if statment ok?
        
        # Density as calculated by BCT 
        density_bct = density_und(A)[0] # 0 is density. Entire output would be for example: (0.28138718173836696, 68, 641)
    
        # Average edge distance
        edge_d = [distance_matrix[i, j] for i, j in G.edges()]
        avg_dist = float(np.mean(edge_d)) if edge_d else np.nan

        # Sum of the lengths of all existing edges.
        wiring_cost = float(np.sum(edge_d)) if edge_d else 0.0
        
        # Characteristic path length (largest CC)
        cpl = largest_component_char_path_length(G)

        # Rich-club measures
        rich_nodes = _get_rich_nodes(G, rich_nodes_global, top_percent=rich_top_percent)
        rich_edges = [(u, v) for (u, v) in G.edges() if u in rich_nodes and v in rich_nodes]
        n_rich_edges = len(rich_edges)
        rc_lengths = [distance_matrix[u, v] for (u, v) in rich_edges]
        avg_rc_length = float(np.mean(rc_lengths)) if rc_lengths else np.nan


        # Append results for this network
        out.append({
            "network_index": idx,
            "avg_communicability": avg_comm,
            "global_efficiency": glob_eff,
            "modularity": modu,
            "avg_clustering": avg_clust,
            "avg_degree": avg_deg,
            "density_bct": density_bct, 
            "transitivity": trans,
            "avg_edge_distance": avg_dist,
            "wiring_cost": wiring_cost,
            "char_path_length": cpl,
            "richclub_n_edges": n_rich_edges,
            "richclub_avg_length": avg_rc_length,
        })
        
        # NEW THINGS!! 
        
        # consensus communities from netneurotools
        # consensus_communities = modularity.consensus_modularity(A, n_iterations=100)

        # out.append({"small_world_omega": _omega(G)})

        # out.append({"structural_complexity": _compute_structural_complexity(A)})


    return out # pd.DataFrame(out)



def calculate_directed_simplices(A) -> int:
    """
    Count the number of directed n-simplices in the graph.
    A directed n-simplex is a motif on n+1 neurons which are all-to-all
    connected in feedforward fashion (i.e., there exists an ordering of nodes
    0,1,...,n such that there is an edge (i,j) whenever i < j).
    
    Returns the count of maximal directed simplices.
    """
    try:
        n = A.shape[0]

        # Find all cliques in the undirected version
        UG = nx.from_numpy_array(np.maximum(A, A.T))
        cliques = list(nx.find_cliques(UG))

        
        simplex_count = 0
        maximum_simplex_size = 0
        
        
        # Check each clique to see if it can form a directed simplex
        for clique in cliques:
            if len(clique) < 2:
                continue
            
            clique_list = list(clique)
            
            # Try all possible orderings of the clique nodes
            # Check if any ordering satisfies the directed simplex property
            from itertools import permutations
            
            is_simplex = False
            for ordering in permutations(clique_list):
                # Check if this ordering gives a directed simplex
                # i.e., edge exists from ordering[i] to ordering[j] for all i < j
                valid_ordering = True
                for i in range(len(ordering)):
                    for j in range(i + 1, len(ordering)):
                        u, v = ordering[i], ordering[j]
                        # Must have edge from u to v (in this ordering)
                        if A[u, v] == 0:
                            valid_ordering = False
                            break
                    if not valid_ordering:
                        break
                
                if valid_ordering:
                    is_simplex = True
                    break
            
            if is_simplex:
                simplex_count += 1
                maximum_simplex_size = max(maximum_simplex_size, len(clique))

        return {"count": simplex_count, 
                "max_size": maximum_simplex_size}

    except Exception as e:
        print(f"Error computing directed simplices: {e}")
        return {"count": 0, 
                "max_size": 0}