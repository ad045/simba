
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_comm
from typing import Iterable, Optional, List, Dict, Any, Tuple


# -----------------------------
# Helper: characteristic path length on LCC (weighted/unweighted)
# -----------------------------
def _largest_component_char_path_length(G: nx.Graph, *, weight: Optional[str] = None) -> float:
    """
    Characteristic path length on the largest connected component (NaN if <2 nodes).
    If ``weight`` is provided, shortest paths are computed using that edge attribute as length.
    """
    
    if G.number_of_nodes() == 0:
        return np.nan

    components = list(nx.connected_components(G))
    if not components:
        return np.nan
    H = G.subgraph(max(components, key=len)).copy()
    if H.number_of_nodes() < 2:
        return np.nan
    try:
        return nx.average_shortest_path_length(H, weight=weight)
    except Exception:
        return np.nan


# -----------------------------
# Helper: choose 'rich' nodes (degree or strength-based)
# -----------------------------
def _get_rich_nodes(
    G: nx.Graph,
    rich_nodes: Optional[Iterable[int]] = None,
    top_percent: float = 0.20,
    use_strength: bool = False,
    weight_attr: str = "weight",
) -> set:
    """Return a set of 'rich' nodes.
    - If ``rich_nodes`` provided (iterable of node indices), use that.
    - Else pick top `top_percent` by degree (or weighted strength if use_strength).
    """
    if rich_nodes is not None:
        return set(rich_nodes)

    if G.number_of_nodes() == 0:
        return set()

    if use_strength:
        # node strength = sum of incident edge weights
        strength = {u: 0.0 for u in G.nodes()}
        for u, v, d in G.edges(data=True):
            w = float(d.get(weight_attr, 0.0))
            strength[u] += w
            strength[v] += w
        scores = strength
    else:
        scores = dict(G.degree())

    k = max(1, int(np.ceil(top_percent * G.number_of_nodes())))
    sorted_nodes = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return set(n for n, _ in sorted_nodes[:k])


# -----------------------------
# Helper: communicability (weighted)
# -----------------------------
def _communicability(A: np.ndarray, mode: str = "estrada_scaled", beta: float = None, t: float = 1.0) -> float:
    """Compute average off-diagonal communicability for a (weighted) adjacency matrix A.
    Modes:
      - 'estrada': average of expm(A)
      - 'estrada_scaled': average of expm(b * A) with b = 1/max(1, λ_max(A))
      - 'heat': average of exp(-t * L_norm) where L_norm is normalized Laplacian of (symmetrized) A
    Uses eigen-decomposition for the ‘estrada*’ modes for stability/perf.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return np.nan

    def avg_offdiag(M: np.ndarray) -> float:
        # For safety, operate on a copy to avoid mutating input matrices
        Mm = M.copy()
        np.fill_diagonal(Mm, 0.0)
        return float(Mm.sum() / (n * (n - 1))) if n > 1 else np.nan

    if mode in ("estrada", "estrada_scaled"):
        # Symmetrize for eigh
        As = 0.5 * (A + A.T)
        # eigh -> sorted eigenvalues ascending
        eigvals, eigvecs = np.linalg.eigh(As)
        lam_max = float(eigvals[-1]) if eigvals.size else 0.0
        if lam_max <= 0:
            # exp(bA) ≈ I => no off-diagonal mass in symmetrical case
            return 0.0
        b = beta if (beta is not None) else 1.0 / lam_max
        max_log = np.log(np.finfo(float).max)  # ~709 for float64
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            args = b * eigvals
            args = np.clip(args, -max_log, +max_log)
            exp_eigs = np.exp(args)
        exp_eigs = np.nan_to_num(exp_eigs, posinf=np.finfo(float).max, neginf=0.0, nan=0.0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            C = (eigvecs * exp_eigs) @ eigvecs.T
        C = np.nan_to_num(C, posinf=np.finfo(float).max, neginf=0.0, nan=0.0)
        return avg_offdiag(C)

    elif mode == "heat":
        # Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
        As = 0.5 * (A + A.T)
        deg = As.sum(axis=1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        S = np.diag(inv_sqrt)
        L = np.eye(n) - S @ As @ S

        from scipy.linalg import expm
        C = expm(-t * L)
        return avg_offdiag(C)

    else:
        raise ValueError("mode must be 'estrada', 'estrada_scaled', or 'heat'.")


# -----------------------------
# Helper: global efficiency (supports weighted graphs via 'length' attribute)
# -----------------------------
def _global_efficiency(G: nx.Graph, *, weight: Optional[str] = None) -> float:
    """Compute global efficiency = average of 1 / d_ij over i != j.
    If 'weight' is provided, it's the edge attribute to use as path length.
    This avoids relying on NetworkX version-specific signatures.
    """
    n = G.number_of_nodes()
    if n < 2:
        return np.nan

    inv_d_sum = 0.0
    cnt = 0
    # Use all_pairs_dijkstra_path_length only when weighted; else use unweighted BFS lengths
    if weight is None:
        lengths_iter = nx.all_pairs_shortest_path_length(G)
    else:
        lengths_iter = nx.all_pairs_dijkstra_path_length(G, weight=weight)

    for u, dists in lengths_iter:
        for v, dist in dists.items():
            if u == v:
                continue
            if dist > 0:
                inv_d_sum += 1.0 / dist
                cnt += 1
    if cnt == 0:
        return np.nan
    return float(inv_d_sum / cnt)


# -----------------------------
# Main analysis
# -----------------------------
def analyze_connectomes(
    connectomes: np.ndarray,
    distance_matrix: np.ndarray,
    comm_mode: str = "estrada_scaled",
    beta: float = None,
    t: float = 1.0,
    rich_nodes_global: Optional[Iterable[int]] = None,
    rich_top_percent: float = 0.20,
    use_weighted: bool = True,
    treat_weights_as_lengths: bool = False,
    # If your connectome weights are *strengths* (larger = stronger connection),
    # leave treat_weights_as_lengths=False (default). We define path length as 1/weight.
    # If your adjacency entries are already *lengths* (larger = longer distance),
    # set treat_weights_as_lengths=True to use them directly in shortest paths.
    min_weight: float = 0.0,  # threshold to keep edges (>=); use >0.0 to sparsify
    symmetrize: str = "max",  # 'max' (default, like original), or 'mean'
) -> List[Dict[str, Any]]:
    """Analyze a set of connectomes with options to preserve and use *real* weights.

    Parameters
    ----------
    connectomes : array-like
        (n,n), (k,n,n) or (n,n,k) adjacency. Weights are preserved if use_weighted=True.
    distance_matrix : (n,n) array
        Euclidean (or other) node distances; used for edge length stats.
    comm_mode : {'estrada','estrada_scaled','heat'}
        Communicability kernel.
    rich_nodes_global : iterable or None
        If provided, use same 'rich' node set for all networks. Otherwise select per-network.
    rich_top_percent : float
        Top-% nodes by degree (or by strength if use_weighted) to define 'rich' set.
    use_weighted : bool
        If True, build weighted graph and compute weighted variants of metrics where meaningful.
    treat_weights_as_lengths : bool
        If True, interpret edge weight as path length directly.
        If False, path length is defined as 1/max(weight, eps).
    min_weight : float
        Threshold: edges with weight < min_weight are dropped.
    symmetrize : {'max', 'mean'}
        Symmetrize rule for A and mask.

    Returns
    -------
    List[dict] with metrics for each connectome.
    """
    arr = np.asarray(connectomes)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] == arr.shape[1]:
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError("connectomes must be (n,n), (k,n,n), or (n,n,k)." )

    k, n, m = arr.shape
    if (n != m) or distance_matrix.shape != (n, n):
        raise ValueError(f"distance_matrix must be {(n,n)}, got {distance_matrix.shape}")

    out: List[Dict[str, Any]] = []
    eps = 1e-12

    for idx in range(k):
        A = np.array(arr[idx], dtype=float)

        # Symmetrize & zero diag
        if symmetrize == "mean":
            A = 0.5 * (A + A.T)
        else:
            A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0.0)

        # Threshold small weights (optional sparsification)
        if min_weight > 0.0:
            A = np.where(A >= min_weight, A, 0.0)

        # Weighted or binary graph for NX
        if use_weighted:
            # Build weighted graph with both 'weight' (strength) and 'length' (for path calculations)
            G = nx.Graph()
            G.add_nodes_from(range(n))
            rows, cols = np.where(A > 0)
            for u, v in zip(rows.tolist(), cols.tolist()):
                if u >= v:
                    continue
                w = float(A[u, v])
                if treat_weights_as_lengths:
                    length = w
                else:
                    length = 1.0 / max(w, eps)
                G.add_edge(u, v, weight=w, length=length)
            # For some metrics we may still want the binary view:
            A_bin = (A > 0).astype(float)
            G_bin = nx.from_numpy_array(A_bin)
        else:
            # Purely binary
            A_bin = (A > 0).astype(float)
            G = nx.from_numpy_array(A_bin)  # unweighted view for everything
            G_bin = G

        # ---- Communicability (weighted adjacency) ----
        avg_comm = _communicability(A if use_weighted else (A > 0).astype(float),
                                    mode=comm_mode, beta=beta, t=t)

        # ---- Global efficiency (weighted if use_weighted) ----
        glob_eff = _global_efficiency(G, weight=('length' if use_weighted else None))

        # ---- Modularity (greedy) ----
        # Use weighted version if available
        comms = list(nx_comm.greedy_modularity_communities(G, weight=('weight' if use_weighted else None)))
        modu = nx_comm.modularity(G, comms, weight=('weight' if use_weighted else None)) if len(comms) > 1 else 0.0

        # ---- Clustering and transitivity ----
        avg_clust = nx.average_clustering(G, weight=('weight' if use_weighted else None)) if G.number_of_nodes() > 0 else np.nan
        # Transitivity in NX is unweighted; use binary graph to keep definition consistent
        trans = nx.transitivity(G_bin) if G_bin.number_of_nodes() > 0 else np.nan

        # ---- Average degree/strength ----
        if use_weighted:
            # average strength = mean over nodes of sum of incident weights
            strengths = np.array([d for _, d in G.degree(weight='weight')], dtype=float)
            avg_deg_or_strength = float(np.mean(strengths)) if strengths.size else np.nan
        else:
            degrees = np.array([d for _, d in G.degree()], dtype=float)
            avg_deg_or_strength = float(np.mean(degrees)) if degrees.size else np.nan

        # ---- Average edge distance (based on presence of edges) ----
        # Use upper triangle mask of edges present (binary presence, regardless of weight magnitude kept after threshold)
        mask = (A > 0)
        iu = np.triu_indices(n, k=1)
        present = mask[iu]
        if present.any():
            avg_dist = float(np.mean(distance_matrix[iu][present]))
        else:
            avg_dist = np.nan

        # ---- Characteristic path length on largest CC ----
        cpl = _largest_component_char_path_length(G if use_weighted else G_bin,
                                                  weight=('length' if use_weighted else None))

        # ---- Rich-club measures ----
        rich_nodes = _get_rich_nodes(
            G if use_weighted else G_bin,
            rich_nodes=rich_nodes_global,
            top_percent=rich_top_percent,
            use_strength=use_weighted,
            weight_attr='weight',
        )
        # Count edges among rich nodes (binary presence)
        if use_weighted:
            # Count edges based on presence; and compute average geometric length
            rich_edges = [(u, v) for (u, v) in G.edges() if u in rich_nodes and v in rich_nodes]
            n_rich_edges = len(rich_edges)
            rc_lengths = [distance_matrix[u, v] for (u, v) in rich_edges]
        else:
            rich_edges = [(u, v) for (u, v) in G_bin.edges() if u in rich_nodes and v in rich_nodes]
            n_rich_edges = len(rich_edges)
            rc_lengths = [distance_matrix[u, v] for (u, v) in rich_edges]
        avg_rc_length = float(np.mean(rc_lengths)) if rc_lengths else np.nan

        out.append({
            "network_index": idx,
            "avg_communicability": avg_comm,
            "global_efficiency": glob_eff,
            "modularity": modu,
            "avg_clustering": avg_clust,
            "avg_degree_or_strength": avg_deg_or_strength,
            "transitivity": trans,
            "avg_edge_distance": avg_dist,
            "char_path_length": cpl,
            "richclub_n_edges": n_rich_edges,
            "richclub_avg_length": avg_rc_length,
            "used_weighted_graph": bool(use_weighted),
            "treat_weights_as_lengths": bool(treat_weights_as_lengths),
            "min_weight_threshold": float(min_weight),
            "symmetrize": symmetrize,
        })

    return out
