"""
Refactored analyze_connectomes function that maintains backward compatibility
while using modular metric functions.
"""
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_comm
from bct import density_und

# # Import from your new modular files
# from structural_metrics import (
#     small_world_omega, 
#     structural_complexity,
#     hubness_gini,
#     degree_assortativity,
#     rich_club_coefficient_max
# )
# from dynamics_metrics import (
#     propagation_efficiency,
#     spectral_radius,
#     spectral_gap,
#     eigenspectrum_entropy
# )


def largest_component_char_path_length(G) -> float:
    """
    Characteristic path length on the largest connected component (NaN if <2 nodes).
    """
    if G.number_of_nodes() == 0:
        return np.nan
    
    components = list(nx.connected_components(G))
    if not components:
        return np.nan
    H = G.subgraph(max(components, key=len)).copy()
    if H.number_of_nodes() < 2:
        return np.nan
    return nx.average_shortest_path_length(H)


def _get_rich_nodes(G, rich_nodes=None, top_percent=0.20):
    """
    Return a set of 'rich' nodes.
        - If rich_nodes provided (iterable of node indices), use that.
        - Else pick top `top_percent` by degree (ties broken arbitrarily).
    """
    if rich_nodes is not None:
        return set(rich_nodes)
    
    deg = dict(G.degree())
    if len(deg) == 0:
        return set()
    k = max(1, int(np.ceil(top_percent * len(deg))))

    sorted_nodes = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)
    return set(n for n, _ in sorted_nodes[:k])


def analyze_connectomes(connectomes, 
                        distance_matrix,
                        comm_mode="estrada_scaled", 
                        beta=None, t=1.0,
                        rich_nodes_global=None, 
                        rich_top_percent=0.20):
    """
    Analyze a set of connectomes for various global graph measures.
    
    BACKWARD COMPATIBLE with original implementation.

    Inputs: 
        - connectomes: either list of (n,n) arrays OR array of shape (k,n,n) or (n,n,k)
        - distance_matrix: (n,n) Euclidean distances
        - comm_mode: 'estrada' | 'estrada_scaled' | 'heat'
        - rich_nodes_global: optional iterable of node indices (same for all nets).
                        If None, uses top `rich_top_percent` by degree per-network.
        - rich_top_percent: fraction used when rich_nodes_global is None.
    
    Returns:
        List of dictionaries, one per network
    """
    # Normalize input shape to (k, n, n)
    arr = np.asarray(connectomes)
    if arr.ndim == 2:             
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] == arr.shape[1]:
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError("connectomes must be (n,n), (k,n,n), or (n,n,k).")

    k, n, _ = arr.shape
    if distance_matrix.shape != (n, n):
        raise ValueError(f"distance_matrix must be {(n,n)}, got {distance_matrix.shape}")

    out = []
    for idx in range(k):
        A = np.array(arr[idx], dtype=float)

        # Ensure symmetry & zero diagonal
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0.0)

        # Binary graph for global metrics
        A_bin = (A != 0).astype(float)
        G = nx.from_numpy_array(A_bin)

        # ========== CALCULATE METRICS ==========
        
        # Communicability (using refactored function)
        # avg_comm = propagation_efficiency(A_bin, mode=comm_mode)
        
        # Global efficiency
        glob_eff = nx.global_efficiency(G)

        # Modularity (greedy)
        comms = list(nx_comm.greedy_modularity_communities(G))
        modu = nx_comm.modularity(G, comms) if len(comms) > 1 else 0.0

        # Other global properties
        avg_clust = nx.average_clustering(G) if G.number_of_nodes() > 0 else np.nan
        avg_deg = float(np.mean([d for _, d in G.degree()])) if G.number_of_nodes() > 0 else np.nan
        trans = nx.transitivity(G) if G.number_of_nodes() > 0 else np.nan
        
        # Density as calculated by BCT 
        density_bct = density_und(A)[0]
    
        # Average edge distance
        edge_d = [distance_matrix[i, j] for i, j in G.edges()]
        avg_dist = float(np.mean(edge_d)) if edge_d else np.nan

        # Sum of the lengths of all existing edges
        wiring_cost = float(np.sum(edge_d)) if edge_d else 0.0
        
        # Characteristic path length (largest CC)
        cpl = largest_component_char_path_length(G)

        # Rich-club measures
        rich_nodes = _get_rich_nodes(G, rich_nodes_global, top_percent=rich_top_percent)
        rich_edges = [(u, v) for (u, v) in G.edges() if u in rich_nodes and v in rich_nodes]
        n_rich_edges = len(rich_edges)
        rc_lengths = [distance_matrix[u, v] for (u, v) in rich_edges]
        avg_rc_length = float(np.mean(rc_lengths)) if rc_lengths else np.nan

        # NEW METRICS (using refactored functions)
        # omega = small_world_omega(G)
        # struct_complexity = structural_complexity(A)

        # Append results for this network
        result_dict = {
            "network_index": idx,
            # "avg_communicability": avg_comm,
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
            # "small_world_omega": omega,
            # "structural_complexity": struct_complexity,
        }
        
        out.append(result_dict)

    return out