"""The 16 network distance measures benchmarked in the paper.

Every measure is a plain function ``f(A, B) -> float`` on two binary, symmetric
(N, N) adjacency matrices. ``A`` is the generated network and ``B`` the
reference. The implementations are ports of the ones that produced the
published numbers (``paper/src/comparing_connectomes/``) with torch removed;
``tests/test_measures.py`` checks them against the shipped values.

Five of them are similarities (higher = more alike). They are listed in
:data:`SIMILARITIES`, and :func:`simba_networks.evaluate` flips them before
any ranking. A new measure declares its own direction with
``evaluate(..., similarity=True)``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp


# ---------------------------------------------------------------------------
# Matrix-based
# ---------------------------------------------------------------------------

def frobenius(A, B):
    """Frobenius norm of the difference, ``||A - B||_F``."""
    return float(np.linalg.norm(np.asarray(A, float) - np.asarray(B, float)))


def hamming(A, B):
    """Number of edges present in exactly one of the two networks."""
    return float(np.sum((np.asarray(A) > 0) != (np.asarray(B) > 0)) / 2)


def jaccard(A, B):
    """Jaccard index of the two edge sets (similarity)."""
    a, b = np.asarray(A) > 0.5, np.asarray(B) > 0.5
    return float((a & b).sum() / ((a | b).sum() + 1e-10))


def f1(A, B):
    """F1 score of ``A``'s edges against ``B``'s (similarity)."""
    a, b = np.asarray(A).astype(int).ravel(), np.asarray(B).astype(int).ravel()
    tp = np.sum((b == 1) & (a == 1))
    fp = np.sum((b == 0) & (a == 1))
    fn = np.sum((b == 1) & (a == 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


# ---------------------------------------------------------------------------
# Information-theoretic (Felippe et al., github.com/hfelippe/network-MI)
# ---------------------------------------------------------------------------

def _ent(p):
    p = np.asarray(p, float)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _upper_edges(A):
    i, j = np.nonzero(np.triu(np.asarray(A) != 0, 1))
    return set(zip(i.tolist(), j.tolist()))


def network_mutual_information(A, B):
    """Normalised mutual information between the two edge sets (similarity)."""
    n = len(A)
    e1, e2 = _upper_edges(A), _upper_edges(B)
    pairs = n * (n - 1) / 2
    p1, p2, p12 = len(e1) / pairs, len(e2) / pairs, len(e1 & e2) / pairs
    h1, h2 = _ent([p1, 1 - p1]), _ent([p2, 1 - p2])
    mi = h1 + h2 - _ent([p12, p1 - p12, p2 - p12, 1 - p1 - p2 + p12])
    return float((2 * mi + 1e-100) / (h1 + h2 + 1e-100))


def dc_network_mutual_information(A, B):
    """Degree-corrected normalised mutual information (similarity)."""
    n = len(A)
    off_diagonal = ~np.eye(n, dtype=bool)
    a, b = (np.asarray(A) != 0) & off_diagonal, (np.asarray(B) != 0) & off_diagonal
    h1 = h2 = mi = 0.0
    for i in range(n):
        p1, p2, p12 = a[i].sum() / n, b[i].sum() / n, (a[i] & b[i]).sum() / n
        e1, e2 = _ent([p1, 1 - p1]), _ent([p2, 1 - p2])
        h1 += e1
        h2 += e2
        mi += e1 + e2 - _ent([p12, p1 - p12, p2 - p12, 1 - p1 - p2 + p12])
    return float((2 * mi + 1e-100) / (h1 + h2 + 1e-100))


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------

_K_EIGENVALUES = 50


def spectral_distance_adjacency(A, B):
    """L2 distance between the 50 largest-magnitude adjacency eigenvalues."""
    def spectrum(M):
        # float32 as in the published run; ARPACK makes the last digits vary
        vals = eigsh(np.asarray(M, np.float32), k=min(_K_EIGENVALUES, len(M) - 2),
                     which="LM")[0]
        return np.sort(vals)[::-1]
    return float(np.linalg.norm(spectrum(A) - spectrum(B)))


def spectral_distance_norm_laplacian(A, B):
    """L2 distance between the 50 smallest normalised-Laplacian eigenvalues."""
    def spectrum(M):
        return np.sort(nx.normalized_laplacian_spectrum(nx.from_numpy_array(np.asarray(M, float))))[:_K_EIGENVALUES]
    return float(np.linalg.norm(spectrum(A) - spectrum(B)))


# The next three are ports of netrd 0.3.0 (MIT, github.com/netsiphd/netrd), taken
# over so that the package does not inherit netrd's pinned Sphinx and ortools.

def netrd_non_backtracking_spectral(A, B):
    """Non-backtracking spectral distance (Torres et al., 2019; netrd).

    Earth mover's distance between the leading eigenvalues of the two graphs'
    non-backtracking matrices, taken as points in the complex plane. ARPACK
    makes the eigenvalues vary in their last digits between runs; when an
    eigenvalue sits on the real axis that can move a value by up to about 1%
    (netrd behaves the same).
    """
    return float(_earth_mover(_nb_eigenvalues(A), _nb_eigenvalues(B)))


def _nb_eigenvalues(M, batch=100, tol=1e-5):
    from scipy import sparse
    from scipy.sparse.linalg import eigs
    core = nx.from_numpy_array(np.asarray(M, float))
    while True:                                   # 2-core
        leaves = [n for n, nbrs in core.adj.items() if len(nbrs) < 2]
        if not leaves:
            break
        core.remove_nodes_from(leaves)
    if len(core) == 0:
        raise ValueError("graph 2-core is empty")
    ident = sparse.eye(core.order())
    degrees = sparse.diags([float(d) for _, d in core.degree()])
    matrix = sparse.bmat([[None, degrees - ident], [-ident, nx.adjacency_matrix(core)]]).tocsr()
    # netrd's "automatic" mode reassigns topk before testing it, so its refinement
    # loop and magnitude filter never run: one call for the `batch` leading values.
    k = min(batch, 2 * len(M) - 4)
    v0 = np.ones(matrix.shape[0]) / matrix.shape[0]
    vals = eigs(matrix, k=k, v0=v0, return_eigenvectors=False, tol=tol)
    vals = vals[vals.imag >= 0]
    vals = sorted(sorted(sorted(vals, key=lambda z: z.imag), key=lambda z: z.real), key=abs)
    return [(z.real, z.imag) for z in vals]


def _earth_mover(p1, p2):
    """EMD between two point multisets with uniform mass (the LP netrd solves)."""
    from collections import Counter
    from scipy.optimize import linprog
    from scipy.spatial.distance import cdist
    c1, c2 = Counter(p1), Counter(p2)
    x, y = list(c1), list(c2)
    w1 = np.array([c1[v] for v in x], float) / len(p1)
    w2 = np.array([c2[v] for v in y], float) / len(p2)
    n1, n2 = len(x), len(y)
    rows = np.zeros((n1 + n2, n1 * n2))
    for i in range(n1):
        rows[i, i * n2:(i + 1) * n2] = 1
    for j in range(n2):
        rows[n1 + j, j::n2] = 1
    res = linprog(cdist(np.array(x), np.array(y)).ravel(), A_eq=rows,
                  b_eq=np.concatenate([w1, w2]), bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.fun


def net_simile(A, B):
    """NetSimile (Berlingerio et al., 2012; netrd): Canberra distance between
    the graphs' signatures of seven local node features."""
    from scipy.spatial.distance import canberra
    from scipy.stats import kurtosis, skew

    def signature(M):
        G = nx.from_numpy_array(np.asarray(M, float))
        nodes = sorted(G.nodes())
        deg, clust = dict(G.degree()), nx.clustering(G)
        ego = {n: nx.ego_graph(G, n) for n in nodes}
        f = np.zeros((len(nodes), 7))
        for r, n in enumerate(nodes):
            nbrs = [m for m in ego[n].nodes if m != n]
            has = deg[n] > 0
            f[r] = [
                deg[n],
                clust[n],
                np.mean([deg[m] for m in nbrs]) if has else 0,
                np.mean([clust[m] for m in nbrs]) if has else 0,
                ego[n].number_of_edges() if has else 0,
                len([e for e in set.union(*[set(G.edges(j)) for j in ego[n].nodes])
                     if not ego[n].has_edge(*e)]),
                len({p for m in G.neighbors(n) for p in G.neighbors(m)} - set(G.neighbors(n)) - {n})
                if has else 0,
            ]
        f = np.nan_to_num(f)
        return np.concatenate([[c.mean(), np.median(c), c.std(), skew(c), kurtosis(c)] for c in f.T])

    return float(abs(canberra(signature(A), signature(B))))


def resistance(A, B):
    """Resistance-perturbation distance (Monnig and Meyer, 2018; netrd): L2
    norm of the difference of the effective-resistance matrices."""
    def resistance_matrix(M):
        M = np.asarray(M, float)
        if not nx.is_connected(nx.from_numpy_array(M)):
            raise ValueError("resistance is undefined for disconnected graphs")
        n = len(M)
        J = np.ones((n, n)) / n
        Li = np.linalg.solve(np.diag(M.sum(0)) - M + J, np.eye(n)) - J
        d = np.diag(Li)
        return d[:, None] + d[None, :] - 2 * Li
    return float(np.sqrt(np.sum((resistance_matrix(A) - resistance_matrix(B)) ** 2)))


# ---------------------------------------------------------------------------
# Graph kernel / feature-based
# ---------------------------------------------------------------------------

def _portrait(M, max_diameter=500):
    n = len(M)
    nbrs = [np.nonzero(M[i] > 0)[0].tolist() for i in range(n)]
    portrait = np.zeros((max_diameter + 1, n + 1))
    max_path = 1
    for start in range(n):
        dist = {start: 0}
        frontier, d = [start], 1
        while frontier:
            nxt = []
            for u in frontier:
                for v in nbrs[u]:
                    if v not in dist:
                        dist[v] = d
                        nxt.append(v)
            frontier, d = nxt, d + 1
        max_dist = max(dist.values())
        max_path = max(max_path, max_dist)
        shells = np.bincount(list(dist.values()))
        for shell, count in enumerate(shells):
            portrait[shell, count] += 1
        portrait[max_dist + 1:, 0] += 1
    return portrait[:max_path + 1]


def portrait(A, B):
    """Portrait divergence (Bagrow and Bollt, 2019)."""
    b1, b2 = _portrait(np.asarray(A)), _portrait(np.asarray(B))
    last = max(np.nonzero(b1)[1].max(), np.nonzero(b2)[1].max()) + 1
    rows = max(len(b1), len(b2))
    p1, p2 = np.zeros((rows, last)), np.zeros((rows, last))
    p1[:len(b1)], p2[:len(b2)] = b1[:, :last], b2[:, :last]
    k = np.arange(last)
    x1, x2 = p1 * k, p2 * k
    if x1.sum() == 0 or x2.sum() == 0:
        return 0.0
    p, q = (x1 / x1.sum()).ravel(), (x2 / x2.sum()).ravel()
    m = 0.5 * (p + q)
    return float(0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2)))


def delta_con(A, B):
    """DeltaCon: Matusita distance between fast-belief-propagation affinities."""
    def affinity(M):
        M = np.asarray(M, float)
        D = np.diag(M.sum(1))
        eps = 1 / (1 + D.max())
        return np.linalg.inv(np.eye(len(M)) + eps ** 2 * D - eps * M)
    return float(np.sqrt(np.sum((np.sqrt(affinity(A)) - np.sqrt(affinity(B))) ** 2)))


# ---------------------------------------------------------------------------
# Communicability-based
# ---------------------------------------------------------------------------

def communicability_corr(A, B):
    """Pearson correlation of the two communicability matrices (similarity)."""
    return float(np.corrcoef(expm(np.asarray(A, float)).ravel(),
                             expm(np.asarray(B, float)).ravel())[0, 1])


def communicability_jsd(A, B):
    """Jensen-Shannon divergence between the normalised communicability matrices."""
    p, q = expm(np.asarray(A, float)).ravel(), expm(np.asarray(B, float)).ravel()
    return float(jensenshannon(p / (p.sum() + 1e-10), q / (q.sum() + 1e-10), base=2.0) ** 2)


# ---------------------------------------------------------------------------
# Baseline: KS-based energy (Betzel et al., 2016)
# ---------------------------------------------------------------------------

def energy(A, B, distance_matrix=None):
    """KS-based energy: the largest of four KS statistics (degree, clustering,
    betweenness, edge length) between ``A`` and ``B``.

    The edge-length term needs the node-to-node distance matrix. When it is not
    given, the Schaefer-100 matrix of the benchmark is used, so this default
    only makes sense for 100-node networks in that parcellation.
    """
    if distance_matrix is None:
        from .data import distance_matrix as _dm
        distance_matrix = _dm()

    def stats(M):
        M = np.asarray(M, float)
        deg = M.sum(1)
        tri = np.diagonal(M @ M @ M)
        pairs = deg * (deg - 1)
        clustering = np.divide(tri, pairs, out=np.zeros_like(tri), where=pairs > 0)
        betweenness = np.fromiter(nx.betweenness_centrality(nx.from_numpy_array(M)).values(), float)
        iu = np.triu_indices_from(M, 1)
        lengths = distance_matrix[iu][M[iu] > 0]
        return deg, clustering, betweenness, lengths

    return float(max(ks_2samp(x, y).statistic for x, y in zip(stats(A), stats(B))))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: All 16 benchmarked measures, keyed by the name used in every table.
MEASURES = {
    "frobenius": frobenius,
    "hamming": hamming,
    "jaccard": jaccard,
    "f1": f1,
    "network_mutual_information": network_mutual_information,
    "dc_network_mutual_information": dc_network_mutual_information,
    "spectral_distance_adjacency": spectral_distance_adjacency,
    "spectral_distance_norm_laplacian": spectral_distance_norm_laplacian,
    "netrd_non_backtracking_spectral": netrd_non_backtracking_spectral,
    "portrait": portrait,
    "net_simile": net_simile,
    "delta_con": delta_con,
    "communicability_corr": communicability_corr,
    "communicability_jsd": communicability_jsd,
    "resistance": resistance,
    "energy": energy,
}

#: Measures where a higher value means more alike.
SIMILARITIES = frozenset({"jaccard", "f1", "communicability_corr",
                          "network_mutual_information", "dc_network_mutual_information"})

#: The eight measures carried through the main analyses.
SELECTED = ("frobenius", "delta_con", "netrd_non_backtracking_spectral",
            "spectral_distance_adjacency", "communicability_corr", "portrait",
            "net_simile", "energy")

#: Display names used in the paper.
NAMES = {
    "frobenius": "Frobenius", "hamming": "Hamming", "jaccard": "Jaccard", "f1": "F1",
    "network_mutual_information": "NMI", "dc_network_mutual_information": "DC-NMI",
    "spectral_distance_adjacency": "Spectral (adjacency)",
    "spectral_distance_norm_laplacian": "Spectral (normalized Laplacian)",
    "netrd_non_backtracking_spectral": "Spectral (non-backtracking)",
    "portrait": "Portrait divergence", "net_simile": "NetSimile", "delta_con": "DeltaCon",
    "communicability_corr": "Communicability correlation",
    "communicability_jsd": "Communicability JSD", "resistance": "Resistance distance",
    "energy": "KS-based energy",
}
