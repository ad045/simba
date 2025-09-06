import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_comm

# is used in preprocessing pipeline. 

def _largest_component_char_path_length(G):
    """
    Characteristic path length on the largest connected component (NaN if <2 nodes).
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


def _communicability(A_bin, mode="estrada_scaled", beta=None, t=1.0):
    """
    Compute average off-diagonal communicability in three modes:
      - estrada:        average of expm(A)
      - estrada_scaled: average of expm(b * A) with b = 1/max(1, λ_max) -> This method is hopefully numerically stable.
      - heat:           average of exp(-t * L_norm) (heat kernel)
    Uses eigen-decomposition for the ‘estrada*’ modes to avoid expm overflows.
    """
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
        avg_comm = _communicability(A_bin, mode=comm_mode, beta=beta, t=t)

        # Global efficiency
        glob_eff = nx.global_efficiency(G)

        # Modularity (greedy)
        comms = list(nx_comm.greedy_modularity_communities(G))
        modu = nx_comm.modularity(G, comms) if len(comms) > 1 else 0.0 # is if statement ok? 

        # Other global properties
        avg_clust = nx.average_clustering(G) if G.number_of_nodes() > 0 else np.nan # is if statment ok?
        avg_deg = float(np.mean([d for _, d in G.degree()])) if G.number_of_nodes() > 0 else np.nan # is if statment ok?
        trans = nx.transitivity(G) if G.number_of_nodes() > 0 else np.nan # is if statment ok?

        # Average edge distance
        edge_d = [distance_matrix[i, j] for i, j in G.edges()]
        avg_dist = float(np.mean(edge_d)) if edge_d else np.nan

        # Characteristic path length (largest CC)
        cpl = _largest_component_char_path_length(G)

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
            "transitivity": trans,
            "avg_edge_distance": avg_dist,
            "char_path_length": cpl,
            "richclub_n_edges": n_rich_edges,
            "richclub_avg_length": avg_rc_length,
        })

    return out # pd.DataFrame(out)