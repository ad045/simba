"""
Additional graph properties for connectome / network analysis.

Functions:
    - basic_measures(A, dist)
    - ollivier_ricci_curvature(A)
    - rich_club_coefficient(A)
    - participation_coefficient(A)
    - persistent_homology(A)
    - targeted_attack_robustness(A)
    - algebraic_connectivity(A)

All functions accept a numpy adjacency matrix A (weighted or binary, 
directed or undirected) and return either a scalar summary or a dict 
of summary statistics, consistent with the style:

    result = function_name(A)

Dependencies:
    Core:     numpy, scipy, networkx
    Optional: GraphRicciCurvature  (pip install GraphRicciCurvature)
              giotto-tda            (pip install giotto-tda)
              scikit-learn          (pip install scikit-learn)

Install optional deps:
    pip install GraphRicciCurvature giotto-tda scikit-learn
"""
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh
import warnings

from src.analysis.structural_measures import analyze_connectomes


def basic_measures(A, dist):
    """
    Communicability etc - all the properties already evaluated in my GNM pipeline. 
    """
    res_dict = analyze_connectomes([A], distance_matrix=dist, comm_mode="estrada_scaled")[0]

    # Dont return the entire res_dict, but only the relevant measures:
    relevant_measures = [
        "avg_communicability", "global_efficiency",
        "modularity", "avg_clustering", "avg_degree", "density_bct",
        "transitivity", "avg_edge_distance", "wiring_cost", "char_path_length", 
        "richclub_n_edges", "richclub_avg_length"
    ]

    return {k: res_dict[k] for k in relevant_measures if k in res_dict}
           

# =============================================================================
# 1. Ollivier-Ricci Curvature
# =============================================================================
def ollivier_ricci_curvature(A):
    """
    Compute summary statistics of the Ollivier-Ricci curvature distribution
    over all edges of the graph.

    Ollivier-Ricci curvature measures local geometric properties of each edge:
        - Positive curvature  → redundant / well-clustered region
        - Negative curvature  → bottleneck / bridge
        - Near zero           → tree-like local structure

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (N x N). Can be weighted. Treated as undirected
        (symmetrized if not already symmetric).

    Returns
    -------
    dict
        'orc_mean'     : mean edge curvature
        'orc_std'      : std of edge curvature
        'orc_min'      : minimum edge curvature
        'orc_max'      : maximum edge curvature
        'orc_median'   : median edge curvature
        'orc_skewness' : skewness of edge curvature distribution
        'orc_frac_neg' : fraction of edges with negative curvature

    References
    ----------
    Ni, C.-C., Lin, Y.-Y., Luo, F., & Gao, J. (2019). Community detection
    on networks with Ricci flow. Scientific Reports, 9(1), 9984.

    Repo: https://github.com/saibalmars/GraphRicciCurvature
    """
    try:
        # from GraphRicciCurvature.OllivierRicci import OllivierRicci
        from src.analysis.ollivier_ricci_curvature import OllivierRicci
    except ImportError:
        raise ImportError(
            "GraphRicciCurvature is required. "
            "Install via: pip install GraphRicciCurvature"
        )

    # Symmetrize if needed
    A_sym = (A + A.T) / 2.0
    np.fill_diagonal(A_sym, 0)

    G = nx.from_numpy_array(A_sym)

    # Remove isolated nodes (ORC needs connected edges)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if G.number_of_edges() == 0:
        warnings.warn("Graph has no edges after removing isolates. "
                       "Returning NaN values.")
        return {k: np.nan for k in [
            'orc_mean', 'orc_std', 'orc_min', 'orc_max',
            'orc_median', 'orc_skewness', 'orc_frac_neg'
        ]}

    orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()
    print(orc.G.nodes(data=True))

    curvatures = []
    for u, v, data in orc.G.edges(data=True):
        curvatures.append(data.get("ricciCurvature", 0.0))

    curvatures = np.array(curvatures)

    # Skewness (Fisher-Pearson)
    mu = np.mean(curvatures)
    std = np.std(curvatures, ddof=1)
    n = len(curvatures)
    if std > 0 and n > 2:
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((curvatures - mu) / std) ** 3)
    else:
        skew = 0.0

    return {
        'orc_mean':     float(np.mean(curvatures)),
        'orc_std':      float(np.std(curvatures, ddof=1)) if n > 1 else 0.0,
        'orc_min':      float(np.min(curvatures)),
        'orc_max':      float(np.max(curvatures)),
        'orc_median':   float(np.median(curvatures)),
        'orc_skewness': float(skew),
        'orc_frac_neg': float(np.mean(curvatures < 0)),
    }


# =============================================================================
# 2. Rich-Club Coefficient
# =============================================================================
def rich_club_coefficient(A):
    """
    Compute the normalized rich-club coefficient.

    The rich-club coefficient phi(k) measures the density of connections
    among nodes with degree > k. Normalizing against random graphs reveals
    whether high-degree nodes are *more* interconnected than expected by
    chance (rich-club organization).

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (N x N). Binarized internally (any nonzero edge → 1).
        Treated as undirected (symmetrized if needed).

    Returns
    -------
    dict
        'rc_max_norm'     : maximum normalized rich-club coefficient across k
        'rc_k_at_max'     : degree threshold k at which max occurs
        'rc_mean_norm'    : mean normalized coefficient across all valid k
        'rc_weighted_auc' : area under the normalized phi(k) curve (trapezoidal)
        'rc_regime_frac'  : fraction of k values where phi_norm > 1
                            (i.e., fraction of degree range showing rich-club)

    Notes
    -----
    Normalization is done against 100 random degree-preserving rewirings.
    This can be slow for very large/dense graphs.

    References
    ----------
    Colizza, V., Flammini, A., Serrano, M.A., & Vespignani, A. (2006).
    Detecting rich-club ordering in complex networks. Nature Physics, 2, 110.

    van den Heuvel, M.P. & Sporns, O. (2011). Rich-club organization of
    the human connectome. J. Neuroscience, 31(44), 15775-15786.
    """
    # Binarize and symmetrize
    A_bin = ((A + A.T) > 0).astype(float)
    np.fill_diagonal(A_bin, 0)

    G = nx.from_numpy_array(A_bin)

    # Raw rich-club
    rc_raw = nx.rich_club_coefficient(G, normalized=False)

    if len(rc_raw) == 0:
        return {k: np.nan for k in [
            'rc_max_norm', 'rc_k_at_max', 'rc_mean_norm',
            'rc_weighted_auc', 'rc_regime_frac'
        ]}

    # Generate null distribution from degree-preserving randomizations
    n_rand = 100
    rc_rand_all = {k: [] for k in rc_raw}

    for _ in range(n_rand):
        G_rand = G.copy()
        # Double-edge swap preserving degree sequence
        nx.double_edge_swap(G_rand,
                            nswap=max(1, G.number_of_edges() * 10),
                            max_tries=G.number_of_edges() * 100)
        rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)
        for k in rc_raw:
            if k in rc_rand and rc_rand[k] > 0:
                rc_rand_all[k].append(rc_rand[k])

    # Normalized rich-club: phi_norm(k) = phi(k) / <phi_rand(k)>
    rc_norm = {}
    for k in sorted(rc_raw.keys()):
        if rc_rand_all[k] and np.mean(rc_rand_all[k]) > 0:
            rc_norm[k] = rc_raw[k] / np.mean(rc_rand_all[k])

    if len(rc_norm) == 0:
        return {k: np.nan for k in [
            'rc_max_norm', 'rc_k_at_max', 'rc_mean_norm',
            'rc_weighted_auc', 'rc_regime_frac'
        ]}

    ks = np.array(sorted(rc_norm.keys()))
    vals = np.array([rc_norm[k] for k in ks])

    max_idx = np.argmax(vals)

    return {
        'rc_max_norm':     float(vals[max_idx]),
        'rc_k_at_max':     int(ks[max_idx]),
        'rc_mean_norm':    float(np.mean(vals)),
        'rc_weighted_auc': float(np.trapz(vals, ks)) if len(ks) > 1 else 0.0,
        'rc_regime_frac':  float(np.mean(vals > 1.0)),
    }


# =============================================================================
# 3. Participation Coefficient
# =============================================================================
def participation_coefficient(A):
    """
    Compute the participation coefficient for each node and return summary
    statistics. Optionally also returns the within-module degree z-score.

    The participation coefficient P_i measures how evenly distributed a
    node's connections are across modules:
        P_i = 0  → all connections within its own module (provincial)
        P_i → 1  → connections uniformly distributed across all modules
                    (connector hub)

    Community detection is performed via Louvain modularity maximization.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (N x N). Can be weighted. Treated as undirected
        (symmetrized if needed).

    Returns
    -------
    dict
        'pc_mean'        : mean participation coefficient
        'pc_std'         : std of participation coefficient
        'pc_median'      : median participation coefficient
        'pc_frac_connector' : fraction of nodes with P > 0.62
                              (approximate threshold for "connector hubs")
        'wmd_mean'       : mean within-module degree z-score
        'wmd_std'        : std of within-module degree z-score
        'n_communities'  : number of detected communities

    References
    ----------
    Guimera, R. & Amaral, L.A.N. (2005). Functional cartography of complex
    metabolic networks. Nature, 433, 895-900.

    Repo (bctpy): https://github.com/aestrivex/bctpy
    """
    # Symmetrize
    A_sym = (A + A.T) / 2.0
    np.fill_diagonal(A_sym, 0)

    G = nx.from_numpy_array(A_sym)
    N = len(A_sym)

    # Community detection (Louvain)
    try:
        communities = nx.community.louvain_communities(G, seed=42)
    except AttributeError:
        # Fallback for older networkx versions
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, random_state=42)
            comm_labels = np.array([partition[i] for i in range(N)])
        except ImportError:
            # Last resort: greedy modularity
            communities = nx.community.greedy_modularity_communities(G)
            comm_labels = np.zeros(N, dtype=int)
            for idx, comm in enumerate(communities):
                for node in comm:
                    comm_labels[node] = idx
        else:
            pass  # comm_labels already set
    else:
        comm_labels = np.zeros(N, dtype=int)
        for idx, comm in enumerate(communities):
            for node in comm:
                comm_labels[node] = idx

    n_communities = len(np.unique(comm_labels))

    # --- Participation coefficient ---
    # P_i = 1 - sum_s (k_is / k_i)^2
    # where k_is = connections of node i to module s, k_i = total degree
    strength = np.sum(A_sym, axis=1)  # weighted degree
    pc = np.zeros(N)

    for i in range(N):
        if strength[i] == 0:
            pc[i] = 0.0
            continue
        for s in np.unique(comm_labels):
            # Sum of weights from node i to nodes in module s
            mask_s = (comm_labels == s)
            k_is = np.sum(A_sym[i, mask_s])
            pc[i] += (k_is / strength[i]) ** 2
        pc[i] = 1.0 - pc[i]

    # --- Within-module degree z-score ---
    wmd = np.zeros(N)
    for s in np.unique(comm_labels):
        mask_s = (comm_labels == s)
        indices_s = np.where(mask_s)[0]
        if len(indices_s) < 2:
            continue
        # Strength of connections within module for each member
        k_within = np.array([np.sum(A_sym[i, mask_s]) for i in indices_s])
        mu_s = np.mean(k_within)
        sigma_s = np.std(k_within, ddof=1)
        if sigma_s > 0:
            for j, node in enumerate(indices_s):
                wmd[node] = (k_within[j] - mu_s) / sigma_s

    return {
        'pc_mean':           float(np.mean(pc)),
        'pc_std':            float(np.std(pc, ddof=1)) if N > 1 else 0.0,
        'pc_median':         float(np.median(pc)),
        'pc_frac_connector': float(np.mean(pc > 0.62)),
        'wmd_mean':          float(np.mean(wmd)),
        'wmd_std':           float(np.std(wmd, ddof=1)) if N > 1 else 0.0,
        'n_communities':     int(n_communities),
    }


# =============================================================================
# 4. Persistent Homology
# =============================================================================
def persistent_homology(A):
    """
    Compute persistent homology features from a weighted graph's
    distance/dissimilarity matrix.

    Persistent homology tracks the birth and death of topological features
    (connected components = H0, loops/cycles = H1) as a filtration
    threshold varies. This captures mesoscale topological structure that
    standard graph metrics miss.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (N x N). Weights are interpreted as connection
        strengths (converted to distances internally via 1/w or max-w).
        Treated as undirected (symmetrized if needed).

    Returns
    -------
    dict
        'ph_h0_n_features'      : number of H0 features (connected components
                                   born during filtration)
        'ph_h0_persistence_mean': mean persistence (death - birth) of H0
        'ph_h0_persistence_std' : std of H0 persistence
        'ph_h0_entropy'         : persistence entropy of H0 diagram
        'ph_h1_n_features'      : number of H1 features (loops)
        'ph_h1_persistence_mean': mean persistence of H1 features
        'ph_h1_persistence_std' : std of H1 persistence
        'ph_h1_entropy'         : persistence entropy of H1 diagram
        'ph_total_persistence'  : sum of all persistence values (H0 + H1)

    References
    ----------
    Edelsbrunner, H. & Harer, J. (2010). Computational Topology: An
    Introduction. AMS.

    Petri, G. et al. (2014). Homological scaffolds of brain functional
    networks. J. R. Soc. Interface, 11(101), 20140873.

    Repos:
        https://github.com/giotto-ai/giotto-tda
        https://github.com/scikit-tda
        https://github.com/Ripser/ripser
    """
    try:
        from gtda.homology import VietorisRipsPersistence
    except ImportError:
        # Fallback: use scipy-based Rips via ripser if available
        try:
            from ripser import ripser as ripser_fn
            _USE_RIPSER = True
        except ImportError:
            raise ImportError(
                "Either giotto-tda or ripser is required. "
                "Install via: pip install giotto-tda   OR   pip install ripser"
            )
    else:
        _USE_RIPSER = False

    # Symmetrize
    A_sym = (A + A.T) / 2.0
    np.fill_diagonal(A_sym, 0)

    # Convert weights (strength) → distance: larger weight = shorter distance
    max_val = np.max(A_sym)
    if max_val == 0:
        warnings.warn("All-zero adjacency matrix. Returning NaN values.")
        keys = [
            'ph_h0_n_features', 'ph_h0_persistence_mean',
            'ph_h0_persistence_std', 'ph_h0_entropy',
            'ph_h1_n_features', 'ph_h1_persistence_mean',
            'ph_h1_persistence_std', 'ph_h1_entropy',
            'ph_total_persistence'
        ]
        return {k: np.nan for k in keys}

    # Distance: invert weights; zero-weight edges → large distance
    with np.errstate(divide='ignore'):
        D = np.where(A_sym > 0, 1.0 / A_sym, 0.0)

    # Set diagonal to 0 (self-distance)
    np.fill_diagonal(D, 0)

    # For disconnected pairs (D==0 off-diagonal), set to max distance
    max_dist = np.max(D)
    off_diag = ~np.eye(D.shape[0], dtype=bool)
    D[off_diag & (D == 0)] = max_dist * 1.5

    def _persistence_entropy(persistence_values):
        """Shannon entropy of normalized persistence values."""
        if len(persistence_values) == 0:
            return 0.0
        p = persistence_values / np.sum(persistence_values)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    if _USE_RIPSER:
        result = ripser_fn(D, maxdim=1, distance_matrix=True)
        diagrams = result['dgms']

        # H0
        h0 = diagrams[0]
        # Remove infinite death values
        h0_finite = h0[np.isfinite(h0[:, 1])] if len(h0) > 0 else np.array([])
        h0_pers = (h0_finite[:, 1] - h0_finite[:, 0]) if len(h0_finite) > 0 else np.array([])

        # H1
        h1 = diagrams[1] if len(diagrams) > 1 else np.array([]).reshape(0, 2)
        h1_finite = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else np.array([])
        h1_pers = (h1_finite[:, 1] - h1_finite[:, 0]) if len(h1_finite) > 0 else np.array([])

    else:
        # giotto-tda path
        vr = VietorisRipsPersistence(
            metric="precomputed",
            homology_dimensions=[0, 1],
            n_jobs=-1
        )
        # giotto expects (n_samples, n_points, n_points)
        diagrams = vr.fit_transform(D[np.newaxis, :, :])[0]

        h0_mask = diagrams[:, 2] == 0
        h1_mask = diagrams[:, 2] == 1

        h0_pts = diagrams[h0_mask]
        h1_pts = diagrams[h1_mask]

        # Remove infinite
        h0_finite = h0_pts[np.isfinite(h0_pts[:, 1])]
        h1_finite = h1_pts[np.isfinite(h1_pts[:, 1])]

        h0_pers = (h0_finite[:, 1] - h0_finite[:, 0]) if len(h0_finite) > 0 else np.array([])
        h1_pers = (h1_finite[:, 1] - h1_finite[:, 0]) if len(h1_finite) > 0 else np.array([])

    all_pers = np.concatenate([h0_pers, h1_pers]) if (len(h0_pers) + len(h1_pers)) > 0 else np.array([])

    return {
        'ph_h0_n_features':       int(len(h0_pers)),
        'ph_h0_persistence_mean': float(np.mean(h0_pers)) if len(h0_pers) > 0 else 0.0,
        'ph_h0_persistence_std':  float(np.std(h0_pers, ddof=1)) if len(h0_pers) > 1 else 0.0,
        'ph_h0_entropy':          _persistence_entropy(h0_pers),
        'ph_h1_n_features':       int(len(h1_pers)),
        'ph_h1_persistence_mean': float(np.mean(h1_pers)) if len(h1_pers) > 0 else 0.0,
        'ph_h1_persistence_std':  float(np.std(h1_pers, ddof=1)) if len(h1_pers) > 1 else 0.0,
        'ph_h1_entropy':          _persistence_entropy(h1_pers),
        'ph_total_persistence':   float(np.sum(all_pers)) if len(all_pers) > 0 else 0.0,
    }


# =============================================================================
# 5. Targeted Attack Robustness
# =============================================================================
def targeted_attack_robustness(A):
    """
    Compute robustness of the graph under targeted (high-degree-first)
    and random node removal.

    Nodes are sequentially removed (along with their edges) and the
    fraction of nodes in the largest connected component (LCC) is tracked.
    The area under this curve (R-index) summarizes robustness.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (N x N). Binarized internally. Treated as
        undirected (symmetrized if needed).

    Returns
    -------
    dict
        'rob_targeted_auc'   : R-index under targeted (degree) attack
        'rob_random_auc'     : R-index under random failure (avg over 10 runs)
        'rob_ratio'          : targeted / random ratio (< 1 means vulnerable
                               to targeted attack relative to random)
        'rob_targeted_half'  : fraction of nodes removed to halve the LCC
                               under targeted attack
        'rob_random_half'    : fraction of nodes removed to halve the LCC
                               under random failure

    References
    ----------
    Albert, R., Jeong, H., & Barabási, A.-L. (2000). Error and attack
    tolerance of complex networks. Nature, 406, 378-382.

    Achard, S. et al. (2006). A resilient, low-frequency, small-world
    human brain functional network. J. Neuroscience, 26(1), 63-72.
    """
    # Binarize and symmetrize
    A_bin = ((A + A.T) > 0).astype(float)
    np.fill_diagonal(A_bin, 0)
    N = A_bin.shape[0]

    def _lcc_fraction(adj):
        """Fraction of nodes in the largest connected component."""
        G = nx.from_numpy_array(adj)
        if G.number_of_nodes() == 0:
            return 0.0
        cc = max(nx.connected_components(G), key=len)
        return len(cc) / N  # always relative to original N

    def _attack_curve(adj, order):
        """Remove nodes in given order, track LCC fraction."""
        fracs = [_lcc_fraction(adj)]
        remaining = list(range(adj.shape[0]))
        current_adj = adj.copy()

        for node_to_remove in order:
            if node_to_remove not in remaining:
                continue
            idx = remaining.index(node_to_remove)
            # Remove node by deleting row/col
            current_adj = np.delete(np.delete(current_adj, idx, axis=0), idx, axis=1)
            remaining.pop(idx)
            fracs.append(_lcc_fraction(current_adj) if current_adj.shape[0] > 0 else 0.0)

        return np.array(fracs)

    def _half_threshold(curve):
        """Fraction of nodes removed when LCC drops below 50% of original."""
        if len(curve) == 0 or curve[0] == 0:
            return 0.0
        target = curve[0] / 2.0
        for i, val in enumerate(curve):
            if val <= target:
                return i / N
        return 1.0

    # --- Targeted attack: remove by degree (recalculated statically) ---
    degrees = np.sum(A_bin, axis=1)
    targeted_order = np.argsort(-degrees)  # highest degree first

    targeted_curve = _attack_curve(A_bin, targeted_order)
    targeted_auc = float(np.trapz(targeted_curve, dx=1.0 / N))

    # --- Random failure: average over multiple runs ---
    n_random = 10
    random_aucs = []
    random_halves = []

    rng = np.random.RandomState(42)
    for _ in range(n_random):
        random_order = rng.permutation(N)
        random_curve = _attack_curve(A_bin, random_order)
        random_aucs.append(float(np.trapz(random_curve, dx=1.0 / N)))
        random_halves.append(_half_threshold(random_curve))

    random_auc = float(np.mean(random_aucs))

    return {
        'rob_targeted_auc':  targeted_auc,
        'rob_random_auc':    random_auc,
        'rob_ratio':         targeted_auc / random_auc if random_auc > 0 else np.nan,
        'rob_targeted_half': _half_threshold(targeted_curve),
        'rob_random_half':   float(np.mean(random_halves)),
    }


# =============================================================================
# 6. Algebraic Connectivity
# =============================================================================
def algebraic_connectivity(A):
    """
    Compute the algebraic connectivity (Fiedler value) and related spectral
    properties of the graph Laplacian.

    The algebraic connectivity lambda_2 is the second-smallest eigenvalue
    of the graph Laplacian. It measures:
        - Resistance to disconnection (higher = harder to partition)
        - Speed of consensus/diffusion
        - Lower bound on vertex/edge connectivity

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (N x N). Can be weighted. Treated as undirected
        (symmetrized if needed).

    Returns
    -------
    dict
        'fiedler_value'         : algebraic connectivity (lambda_2 of Laplacian)
        'fiedler_value_norm'    : lambda_2 / lambda_max (normalized)
        'laplacian_spectral_gap': lambda_2 / lambda_3 ratio (if available)
        'fiedler_bipartition_balance' : balance of Fiedler vector bipartition
                                        (0.5 = perfectly balanced cut)

    References
    ----------
    Fiedler, M. (1973). Algebraic connectivity of graphs. Czechoslovak
    Mathematical Journal, 23(2), 298-305.

    de Abreu, N.M.M. (2007). Old and new results on algebraic connectivity
    of graphs. Linear Algebra and its Applications, 423(1), 53-73.
    """
    # Symmetrize
    A_sym = (A + A.T) / 2.0
    np.fill_diagonal(A_sym, 0)
    N = A_sym.shape[0]

    if N < 3:
        return {k: np.nan for k in [
            'fiedler_value', 'fiedler_value_norm',
            'laplacian_spectral_gap', 'fiedler_bipartition_balance'
        ]}

    # Graph Laplacian: L = D - A
    degrees = np.sum(A_sym, axis=1)
    L = np.diag(degrees) - A_sym

    # Compute smallest eigenvalues of Laplacian
    # We need lambda_2 and lambda_3 (skip lambda_1 ≈ 0)
    n_eigs = min(4, N - 1)
    try:
        L_sparse = sparse.csr_matrix(L)
        eigenvalues, eigenvectors = eigsh(L_sparse, k=n_eigs, which='SM')
    except Exception:
        # Fallback to dense
        eigenvalues_full = np.linalg.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues_full)[:n_eigs]
        # Get eigenvectors for Fiedler vector
        eigenvalues_full_all, eigenvectors_all = np.linalg.eigh(L)
        sort_idx = np.argsort(eigenvalues_full_all)
        eigenvectors = eigenvectors_all[:, sort_idx[:n_eigs]]

    # Sort eigenvalues
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # lambda_2 = Fiedler value (second smallest)
    fiedler_val = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    # Ensure non-negative (numerical precision)
    fiedler_val = max(0.0, fiedler_val)

    # lambda_max of Laplacian (for normalization)
    try:
        lambda_max = float(eigsh(L_sparse, k=1, which='LM', return_eigenvectors=False)[0])
    except Exception:
        lambda_max = float(np.max(np.linalg.eigvalsh(L)))

    fiedler_norm = fiedler_val / lambda_max if lambda_max > 0 else 0.0

    # Spectral gap ratio: lambda_2 / lambda_3
    if len(eigenvalues) > 2 and eigenvalues[2] > 1e-10:
        spectral_gap_ratio = float(eigenvalues[1] / eigenvalues[2])
    else:
        spectral_gap_ratio = np.nan

    # Fiedler vector bipartition balance
    if eigenvectors.shape[1] > 1:
        fiedler_vec = eigenvectors[:, 1]
        n_positive = np.sum(fiedler_vec >= 0)
        balance = min(n_positive, N - n_positive) / (N / 2.0)
    else:
        balance = np.nan

    return {
        'fiedler_value':                float(fiedler_val),
        'fiedler_value_norm':           float(fiedler_norm),
        'laplacian_spectral_gap':       float(spectral_gap_ratio),
        'fiedler_bipartition_balance':  float(balance),
    }

