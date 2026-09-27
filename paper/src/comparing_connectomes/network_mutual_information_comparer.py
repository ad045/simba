
import torch
import numpy as np
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator

import networkx as nx


###################
# This code block is taken from https://github.com/hfelippe/network-MI/blob/main/functions.py
 
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
    