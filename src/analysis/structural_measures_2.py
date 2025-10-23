"""
Structural network metrics computation.
"""
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_comm
from bct import density_und
from networkx.algorithms import smallworld
from netneurotools import modularity


def largest_component_char_path_length(G):
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
    """
    if rich_nodes is not None:
        return set(rich_nodes)
    
    deg = dict(G.degree())
    if len(deg) == 0:
        return set()
    k = max(1, int(np.ceil(top_percent * len(deg))))
    sorted_nodes = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)
    return set(n for n, _ in sorted_nodes[:k])


def compute_avg_clustering(A_bin, G=None):
    """Average clustering coefficient."""
    if G is None:
        G = nx.from_numpy_array(A_bin)
    return nx.average_clustering(G) if G.number_of_nodes() > 0 else np.nan


def compute_consensus_modularity(A, n_iterations=100):
    """Consensus modularity using netneurotools."""
    try:
        ci, q = modularity.consensus_modularity(A, n_iterations=n_iterations)
        return q
    except:
        return np.nan


def compute_small_world_omega(G, niter=5, nrand=10):
    """Small-world coefficient omega."""
    try:
        return smallworld.omega(G, niter=niter, nrand=nrand)
    except:
        return np.nan


def compute_avg_wiring_cost(G, distance_matrix):
    """Average edge distance."""
    edge_d = [distance_matrix[i, j] for i, j in G.edges()]
    return float(np.mean(edge_d)) if edge_d else np.nan


def compute_total_wiring_cost(G, distance_matrix):
    """Sum of all edge distances."""
    edge_d = [distance_matrix[i, j] for i, j in G.edges()]
    return float(np.sum(edge_d)) if edge_d else 0.0


def compute_gini_coefficient(degrees):
    """
    Compute Gini coefficient for hubness measure (Chini 2023).
    """
    degrees = np.array(degrees)
    if len(degrees) == 0:
        return np.nan
    
    # Sort degrees
    sorted_degrees = np.sort(degrees)
    n = len(sorted_degrees)
    
    # Gini coefficient formula
    cumsum = np.cumsum(sorted_degrees)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_degrees))) / (n * cumsum[-1]) - (n + 1) / n
    
    return gini


def compute_hubness_gini(G):
    """Hubness as Gini index of degree distribution."""
    degrees = [d for _, d in G.degree()]
    return compute_gini_coefficient(degrees)


def compute_rich_club_coefficient(G, k=None):
    """
    Rich club coefficient for degree threshold k.
    If k is None, finds the k that maximizes the coefficient.
    """
    try:
        if k is None:
            # Find k that maximizes rich club coefficient
            rc_dict = nx.rich_club_coefficient(G, normalized=False)
            if not rc_dict:
                return np.nan, np.nan
            k_max = max(rc_dict.keys(), key=lambda x: rc_dict[x])
            return rc_dict[k_max], k_max
        else:
            rc_dict = nx.rich_club_coefficient(G, normalized=False)
            return rc_dict.get(k, np.nan), k
    except:
        return np.nan, np.nan


def compute_richclub_stats(G, distance_matrix, rich_nodes=None, top_percent=0.20):
    """Rich club average length and number of edges."""
    rich_nodes = _get_rich_nodes(G, rich_nodes, top_percent=top_percent)
    rich_edges = [(u, v) for (u, v) in G.edges() if u in rich_nodes and v in rich_nodes]
    n_rich_edges = len(rich_edges)
    rc_lengths = [distance_matrix[u, v] for (u, v) in rich_edges]
    avg_rc_length = float(np.mean(rc_lengths)) if rc_lengths else np.nan
    return n_rich_edges, avg_rc_length


def compute_avg_degree(G):
    """Average degree."""
    return float(np.mean([d for _, d in G.degree()])) if G.number_of_nodes() > 0 else np.nan


def compute_transitivity(G):
    """Transitivity (global clustering coefficient)."""
    return nx.transitivity(G) if G.number_of_nodes() > 0 else np.nan


def compute_degree_assortativity(G):
    """Degree assortativity coefficient."""
    try:
        return nx.degree_assortativity_coefficient(G)
    except:
        return np.nan


def compute_distance_dependent_assortativity(G, distance_matrix, distance_bins=10):
    """
    Distance-dependent degree assortativity (Betzel).
    Computes assortativity for edges binned by distance.
    """
    edges = list(G.edges())
    if not edges:
        return np.nan
    
    # Get edge distances and degrees
    edge_distances = [distance_matrix[i, j] for i, j in edges]
    degrees = dict(G.degree())
    
    # Bin edges by distance
    bins = np.linspace(np.min(edge_distances), np.max(edge_distances), distance_bins + 1)
    assortativity_by_bin = []
    
    for i in range(len(bins) - 1):
        # Get edges in this distance bin
        bin_edges = [e for e, d in zip(edges, edge_distances) if bins[i] <= d < bins[i+1]]
        if len(bin_edges) > 1:
            # Create subgraph with only these edges
            H = nx.Graph()
            H.add_edges_from(bin_edges)
            try:
                assortativity_by_bin.append(nx.degree_assortativity_coefficient(H))
            except:
                assortativity_by_bin.append(np.nan)
    
    return np.nanmean(assortativity_by_bin) if assortativity_by_bin else np.nan


def compute_matrix_entropy(A):
    """
    Entropy of the weight matrix.
    Treats weights as a probability distribution.
    """
    A_flat = A.flatten()
    A_flat = A_flat[A_flat > 0]  # Only non-zero weights
    
    if len(A_flat) == 0:
        return 0.0
    
    # Normalize to probability distribution
    p = A_flat / A_flat.sum()
    
    # Shannon entropy
    H = -np.sum(p * np.log(p + 1e-12))
    
    return H