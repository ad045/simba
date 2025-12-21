
import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator

import networkx as nx


###################
# taken from https://github.com/hfelippe/network-MI/blob/main/functions.py
 
import random
import numpy as np
from scipy.special import loggamma
from collections import defaultdict

def logmultiset(N,K):
    """logarithm of multiset coefficient"""
    return loggamma(N+K-1+1) - loggamma(K+1) - loggamma(N-1+1)

def zero_log(x):
    """log of zero is zero"""
    if x <= 0: return 0
    else: return np.log(x)

def ent(vec):
    """entropy of a distribution"""
    vec  = np.array(vec)/sum(vec)
    return -sum([x*zero_log(x) for x in vec])

def jaccard(A, B):
    """Jaccard index of sets A and B"""
    return len(A & B) / (len(A) + len(B) - len(A & B))
 
def NMI(N,e1,e2):
    """normalized mutual information between N-node graphs with edge sets e1, e2"""
    Nc2 = N*(N-1)/2
    E1,E2,E12,Union = len(e1),len(e2),len(e1.intersection(e2)),len(e1.union(e2))
    p1,p2,p12 = E1/Nc2,E2/Nc2,E12/Nc2
    H1,H2 = ent([p1,1-p1]), ent([p2,1-p2])
    MI = H1 + H2 - ent([p12,p1-p12,p2-p12,1-p1-p2+p12])
    NMI = (2*MI+1e-100)/(H1+H2+1e-100) # negligibly small constants for the empty and complete graphs
    return NMI

def DCNMI(N,e1,e2):
    """degree-corrected normalized mutual information between N-node graphs with edge sets e1, e2"""
    adj1,adj2 = defaultdict(set),defaultdict(set)
    for e in e1:
        i,j = e
        if not(i in adj1): adj1[i] = set([])
        if not(j in adj1): adj1[j] = set([])
        adj1[i].add(j)
        adj1[j].add(i)
    for e in e2:
        i,j = e
        if not(i in adj2): adj2[i] = set([])
        if not(j in adj2): adj2[j] = set([])
        adj2[i].add(j)
        adj2[j].add(i)
    DCH1,DCH2,DCMI = 0,0,0
    for i in range(N):
        p1i,p2i,p12i = len(adj1[i])/N,len(adj2[i])/N,len(adj1[i].intersection(adj2[i]))/N 
        DCH1 += ent([p1i,1-p1i])
        DCH2 += ent([p2i,1-p2i])
        DCMI += ent([p1i,1-p1i]) + ent([p2i,1-p2i]) - ent([p12i,p1i-p12i,p2i-p12i,1-p1i-p2i+p12i])
    DCNMI = (2*DCMI+1e-100)/(DCH1+DCH2+1e-100) # negligibly small constants for the empty and complete graphs
    return DCNMI

def mesoNMI(N,e1,e2,partition):
    """mesoscale normalized mutual information between N-node graphs with edge sets e1, e2 and reference partition"""
    B = len(set(partition))
    Bc2 = B*(B-1)/2
    E1,E2 = len(e1),len(e2)
    
    table1,table2 = defaultdict(int),defaultdict(int)
    for e in e1:
        i,j = e
        r,s = sorted([partition[i],partition[j]])
        if not((r,s) in table1): table1[(r,s)] = 0
        table1[(r,s)] += 1
    for e in e2:
        i,j = e
        r,s = sorted([partition[i],partition[j]])
        if not((r,s) in table2): table2[(r,s)] = 0
        table2[(r,s)] += 1
    
    E12 = 0
    pairs = set(list(table1.keys())+list(table2.keys()))
    for pair in pairs:
        E12 += min(table1[pair],table2[pair])
        
    H1,H2,H12 = logmultiset(Bc2+B,E1),logmultiset(Bc2+B,E2),logmultiset(Bc2+B,E1+E2-E12)
    I = H1 + H2 - H12
    I0 = H1 + H2 - logmultiset(Bc2+B,E1+E2)

    return (I - I0 +1e-100)/((H1+H2)/2 - I0 +1e-100) 

"""
        Attack over graphs
"""

def typeI(Gset, eps):
    """Type I noise over fraction eps of nodes in decreasing order of degree""" # to get the random attack, modify `degree_order`

    def degree_order(dict_node_order):
        """returns node_order sorted by highest-degree"""
        node_degrees={node: len(neighbors) for node, neighbors in dict_node_order.items()}
        sorted_nodes=sorted(node_degrees, key=node_degrees.get, reverse=True)
        return {node: dict_node_order[node] for node in sorted_nodes}

    adjlist = {}
    for e in Gset:
        i,j = e
        if not(i in adjlist):
            adjlist[i] = []
        if not(j in adjlist):
            adjlist[j] = []
        adjlist[i].append(j)
        adjlist[j].append(i)
    N = len(adjlist)

    # create placeholders for both the addition and removal of edges from graph G
    new_edges = set()
    old_edges = set()

    # loop through epsilon*N nodes
    deg_order  = degree_order(adjlist)
    node_order = list(deg_order.keys())
    for i in node_order[:int(eps * N)]:
        for neig in adjlist[i]:
            if neig > i:
                repeated = True
                while repeated == True:
                    to_add = tuple(sorted([i, i]))  # Initialize to (i, i) to enter the while loop
                    while to_add[1] == i:
                        to_add = tuple(sorted([i, random.randint(0, N-1)]))
                    if not(to_add in new_edges) and not(to_add in Gset) and not(to_add in old_edges) and to_add[0]!=to_add[1]:
                        old_edges.add(tuple(sorted([i,neig])))
                        new_edges.add(tuple(sorted(to_add)))
                        repeated = False

    Gset_new = Gset.difference(old_edges)
    Gset_new = Gset_new.union(new_edges)

    return Gset_new

def typeII(Gset, eps):
    """Type II noise over fraction eps of edges"""
    N = 1 + max(max(edge) for edge in Gset) # number of nodes
    edges = Gset.copy()
    new_edges = set()
    rand_ij = eps*len(edges)
    count = 0
    while count < rand_ij:
        to_add = (random.choice(range(N)), random.choice(range(N)))
        if to_add[0]!=to_add[1] and not(to_add in edges) and not((to_add[1], to_add[0]) in edges) and not(to_add in new_edges) and not((to_add[1], to_add[0]) in new_edges):
            to_add = (min(to_add), max(to_add)) # imposes i < j for all edges (i,j)
            edges.pop()
            new_edges.add(to_add)
            count += 1
        else:
            pass
    
    return new_edges.union(edges)

def typeIII(Gset, partition, eps):
    """Type III noise over community-community edges"""   
    import time
    timeout = .1 # set a timeout in seconds

    N = len(partition) # number of nodes
    comms = sorted(list(set(partition)))
    B = len(comms)

    edges = {}
    for edge in Gset:
        i,j = edge
        r,s = sorted([partition[i],partition[j]]) 
        if not((r,s) in edges): 
            edges[(r,s)] = set()
        edges[(r,s)].add((i,j)) 
    
    comm_sets = {l:[] for l in comms}
    for i in range(N):
        comm_sets[partition[i]].append(i)
    
    for rs in edges.keys():
        new_edges=set()
        r,s = rs           
        rand_rs = int(eps*len(edges[(r,s)]))
        count = 0
        start_time = time.time()
        while count < rand_rs and (time.time() - start_time) < timeout: 
            to_add = (random.choice(comm_sets[r]),random.choice(comm_sets[s]))
            if not(to_add in new_edges) and not((to_add[1],to_add[0]) in new_edges) and (to_add[0] != to_add[1]) and not(to_add in edges[(r,s)]) and not((to_add[1],to_add[0]) in edges[(r,s)]):
                edges[(r,s)].pop()
                new_edges.add(to_add)
                count += 1
            else:
                pass            
        edges[(r,s)] = new_edges.union(edges[(r,s)])
    
    return set().union(*list(edges.values()))






###################

###################

# The following wrapper is "my" own


# Helper: adjacency → edge set
def adjacency_to_edge_set(adj: np.ndarray):
    """
    Convert an undirected adjacency matrix to an edge set {(i,j), i<j}.
    
    Parameters
    ----------
    adj : np.ndarray, shape (N, N)
        Binary or weighted adjacency matrix.

    Returns
    -------
    edges : set of tuple
        Edge set representation required by NMI / DCNMI.
    """
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    N = adj.shape[0]
    edges = set()

    # Upper triangle only (undirected, no self-loops)
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] != 0:
                edges.add((i, j))

    return edges



def compute_nmi_from_adjacency(adj1: np.ndarray, adj2: np.ndarray):
    """
    Compute standard NMI between two adjacency matrices
    using the authors' original implementation.
    """
    if adj1.shape != adj2.shape:
        raise ValueError("Adjacency matrices must have the same shape")

    N = adj1.shape[0]
    e1 = adjacency_to_edge_set(adj1)
    e2 = adjacency_to_edge_set(adj2)

    return NMI(N, e1, e2)


def compute_dcnmi_from_adjacency(adj1: np.ndarray, adj2: np.ndarray):
    """
    Compute degree-corrected NMI (DC-NMI) between two adjacency matrices
    using the authors' original implementation.
    """
    if adj1.shape != adj2.shape:
        raise ValueError("Adjacency matrices must have the same shape")

    N = adj1.shape[0]
    e1 = adjacency_to_edge_set(adj1)
    e2 = adjacency_to_edge_set(adj2)

    return DCNMI(N, e1, e2)



class NetworkMutualInformationEvaluator(NetworkEvaluator):
    """Evaluate networks using GNM energy-based criteria."""
    
    def __init__(self):
        pass
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute energy for single generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        target = target.cpu().numpy()
        generated = generated[0].cpu().numpy() # to really get shape 100x100

        # print(target.shape, generated.shape)

        for target_idx in range(n_targets):
            
            target_single = target[target_idx:target_idx+1, :, :][0] # to really get shape 100x100

            nmi = compute_nmi_from_adjacency(generated, target_single)
            # dcnmi = compute_dcnmi_from_adjacency(generated, target_single)
            
            results[target_idx] = nmi # or dcnmi

        return results

    @property
    def metric_prefix(self) -> str:
        return "NMICrit"
    




class DCNetworkMutualInformationEvaluator(NetworkEvaluator):
    """Evaluate networks using GNM energy-based criteria."""
    
    def __init__(self):
        pass
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute energy for single generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        target = target.cpu().numpy()
        generated = generated[0].cpu().numpy()
        
        for target_idx in range(n_targets):
            
            target_single = target[target_idx:target_idx+1, :, :][0]

            # nmi = compute_nmi_from_adjacency(generated, target_single)
            dcnmi = compute_dcnmi_from_adjacency(generated, target_single)

            results[target_idx] = dcnmi

        return results

    @property
    def metric_prefix(self) -> str:
        return "DCNMICrit"
    
    
# if __name__ == "__main__":
#     # Example: two random sparse graphs
#     rng = np.random.default_rng(0)
#     N = 100
#     p = 0.10  # 10% density

#     A = (rng.random((N, N)) < p).astype(int)
#     B = (rng.random((N, N)) < p).astype(int)

#     # Symmetrize & remove self-loops
#     A = np.triu(A, 1)
#     A = A + A.T
#     B = np.triu(B, 1)
#     B = B + B.T

#     nmi = compute_nmi_from_adjacency(A, B)
#     dcnmi = compute_dcnmi_from_adjacency(A, B)

#     print(f"NMI    = {nmi:.4f}")
#     print(f"DC-NMI = {dcnmi:.4f}")




# # Code part taken from network_MI. 


# import numpy as np
# import networkx as nx
# from scipy.linalg import eigh


# # Jensen-Shannon Distance 
# def zero_log(x):
#     """Helper function for entropy calculation"""
#     if x <= 0:
#         return 0
#     return np.log2(x)


# def comb_laplacian(adj_matrix):
#     """Combinatorial laplacian from adjacency matrix"""
#     K = adj_matrix.shape[0]
#     # Create graph from adjacency matrix
#     G = nx.from_numpy_array(adj_matrix)
#     L = nx.laplacian_matrix(G, weight=None).toarray()
#     return L / (2*K)


# def vne_ent(rho):
#     """Von Neumann entropy"""
#     eigs = np.linalg.eigvals(rho)
#     return -sum(x*zero_log(x) for x in eigs) / np.log2(len(eigs) - 1)


# def d_jensen_shannon(adj1, adj2):
#     """Jensen-Shannon distance between two adjacency matrices"""
#     rho = comb_laplacian(adj1)
#     sigma = comb_laplacian(adj2)
#     mu = 0.5 * (rho + sigma)
#     return vne_ent(mu) - 0.5 * (vne_ent(rho) + vne_ent(sigma))


# def adj_to_edges(adj_matrix):
#     """Convert adjacency matrix to set of edges"""
#     edges = set()
#     n = adj_matrix.shape[0]
#     for i in range(n):
#         for j in range(i+1, n):  # Assuming undirected graph
#             if adj_matrix[i, j] != 0:
#                 edges.add((i, j))
#     return edges




# class NetworkMutualInformationEvaluator(NetworkEvaluator):
#     """Evaluate networks using Network Mutual Information.
    
#     Measures information-theoretic similarity between network structures.
#     """
    
#     def __init__(self, variant: str = 'standard'):
#         """
#         Args:
#             variant: 'standard', 'degree_corrected', or 'mesoscale'
#         """
#         self.variant = variant
    
#     # def _compute_edge_nmi(self, adj1: np.ndarray, adj2: np.ndarray) -> float:
#     #     """Compute standard NMI based on edge overlap."""
#     #     # Flatten adjacency matrices to edge lists
#     #     edges1 = (adj1 > 0).astype(int).flatten()
#     #     edges2 = (adj2 > 0).astype(int).flatten()
        
#     #     # Compute mutual information
#     #     # P(X=1, Y=1), P(X=1, Y=0), etc.
#     #     n = len(edges1)
#     #     p11 = np.sum((edges1 == 1) & (edges2 == 1)) / n
#     #     p10 = np.sum((edges1 == 1) & (edges2 == 0)) / n
#     #     p01 = np.sum((edges1 == 0) & (edges2 == 1)) / n
#     #     p00 = np.sum((edges1 == 0) & (edges2 == 0)) / n
        
#     #     p1_ = p11 + p10
#     #     p0_ = p01 + p00
#     #     p_1 = p11 + p01
#     #     p_0 = p10 + p00
        
#     #     # Mutual information
#     #     mi = 0
#     #     for px, py, pxy in [(p1_, p_1, p11), (p1_, p_0, p10), 
#     #                         (p0_, p_1, p01), (p0_, p_0, p00)]:
#     #         if pxy > 0 and px > 0 and py > 0:
#     #             mi += pxy * np.log2(pxy / (px * py))
        
#     #     # Normalized MI
#     #     h1 = -sum([p * np.log2(p) for p in [p1_, p0_] if p > 0])
#     #     h2 = -sum([p * np.log2(p) for p in [p_1, p_0] if p > 0])
        
#     #     if h1 + h2 > 0:
#     #         nmi = 2 * mi / (h1 + h2)
#     #     else:
#     #         nmi = 0.0
        
#     #     return nmi
    
#     def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
#         """Compute NMI for generated network against target batch."""
#         results = {}
#         n_targets = target.shape[0]
        
#         gen_np = generated[0].cpu().numpy()
        
#         G_gen = nx.from_numpy_array(gen_np)
        
#         for target_idx in range(n_targets):
#             target_np = target[target_idx].cpu().numpy()
#             G_target = nx.from_numpy_array(target_np)

#             # Compute Jensen-Shannon distance
#             # jsd = d_jensen_shannon(gen_np, target_np)
#             # nmi_score = jsd
#             # print(f"Jensen-Shannon Distance: {jsd}")

#             # For NMI-based measures (requires your functions.py):
#             # n = gen_np.shape[-1]
#             # edges1 = adj_to_edges(gen_np)
#             # edges2 = adj_to_edges(target_np)
#             # nmi_score = NMI(n, edges1, edges2)
#             # dcnmi_score = dcNMI(n, edges1, edges2)

#             # # Ensure same size
#             # min_size = min(gen_np.shape[0], target_np.shape[0])
#             # gen_crop = gen_np[:min_size, :min_size]
#             # target_crop = target_np[:min_size, :min_size]
            
#             # FROM network-MI FILE 
#             # Turn target and generated into Graphs
#             # G_gen = nx.from_numpy_array(gen_np)
#             # G_target = nx.from_numpy_array(target_np)

#             # nmi: Taken from network_MI
#             n=G_gen.number_of_nodes()
#             N=len(layer_graphs)
#             M=np.zeros((N,N))
#             for i in range(1, N+1):
#                 for j in range(i, N+1):
#                     M[i-1,j-1]=NMI(n, set(layer_graphs[i].edges()),set(layer_graphs[j].edges()))
#                     M[j-1,i-1]=M[i-1,j-1]
#                     M[i-1,i-1]=1
                    
#             # if self.variant == 'standard':
#             #     nmi = self._compute_edge_nmi(gen_np, target_np) # , target_crop)
#             # else:
#             #     # Simplified version for other variants
#             #     nmi = self._compute_edge_nmi(gen_np, target_np)

#             # Convert similarity to distance
#             # distance = 1.0 - nmi_
#             results[target_idx] = float(nmi_score) # distance)
        
#         return results
    
#     @property
#     def metric_prefix(self) -> str:
#         return f"NMI_{self.variant}"