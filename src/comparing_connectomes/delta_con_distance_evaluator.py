
import torch
import numpy as np
import networkx as nx
from typing import Dict, List
from scipy.linalg import eigh
from scipy.spatial.distance import cosine
from src.comparing_connectomes.base_comparer import NetworkEvaluator

# This code is adapted to use distance-weighted adjacency matrices.
"""
Compare matrices related to Fast Belief Propagation. 
The following part is a part from https://netrd.readthedocs.io/en/latest/_modules/netrd/distance/deltacon.html

Deltacon measure for graph distance, after:
Koutra, Danai, Joshua T. Vogelstein, and Christos Faloutsos. 2013. “Deltacon: A
Principled Massive-Graph Similarity Function.” In Proceedings of the 2013 SIAM
International Conference on Data Mining, 162–70. Society for Industrial and
Applied Mathematics. https://doi.org/10.1137/1.9781611972832.18.

author: Stefan McCabe
email: stefanmccabe at gmail dot com
Submitted as part of the 2019 NetSI Collabathon.
"""


def matusita_dist(X, Y):
    r"""Return the Matusita distance

    .. math::

        \sqrt{\sum_i \sum_j \left( \sqrt{X_{ij}} - \sqrt{Y_{ij}} \right)^{2}}


    between X and Y.
    """
    return np.sqrt(np.sum(np.square(np.sqrt(X) - np.sqrt(Y))))


def compute_deltacon(G1, G2): # , exact=True, g=None):
        """DeltaCon is based on the Matsusita between matrices created from fast
        belief propagation (FBP) on graphs G1 and G2.

        Because the FBP algorithm requires a costly matrix inversion, there
        is a faster, roughly linear, algorithm that gives approximate
        results.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        # exact (bool) -> not-exact was not implemented. 
        #     if True, use the slower but exact algorithm (DeltaCon_0)

        # g (int) -> not-exact was not implemented. 
        #     the number of groups to use in the efficient algorithm. If
        #     exact is set to False but g is not set, the efficient algorithm
        #     will still behave like the exact algorithm, since each node is
        #     put in its own group.

        Returns
        -------

        dist (float)
            the distance between G1 and G2.

        References
        ----------

        .. [1] Koutra, Danai, Joshua T. Vogelstein, and Christos
               Faloutsos. 2013. "Deltacon: A Principled Massive-Graph
               Similarity Function." In Proceedings of the 2013 SIAM
               International Conference on Data Mining, 162–70. Society for
               Industrial and Applied
               Mathematics. https://doi.org/10.1137/1.9781611972832.18.

        """
        assert G1.number_of_nodes() == G2.number_of_nodes()
        N = G1.number_of_nodes()

        # if not exact and g is None:
        #     g = N

        A1 = nx.to_numpy_array(G1)
        L1 = nx.laplacian_matrix(G1).toarray()
        D1 = L1 + A1

        A2 = nx.to_numpy_array(G2)
        L2 = nx.laplacian_matrix(G2).toarray()
        D2 = L2 + A2

        eps_1 = 1 / (1 + np.max(D1))
        eps_2 = 1 / (1 + np.max(D2))

        S1 = np.linalg.inv(np.eye(N) + (eps_1**2) * D1 - eps_1 * A1)
        S2 = np.linalg.inv(np.eye(N) + (eps_2**2) * D2 - eps_2 * A2)

        dist = matusita_dist(S1, S2)
        
        return dist



class DeltaConDistanceEvaluator(NetworkEvaluator):
    """
    Evaluate networks using DeltaCon metric.
    Uses fast belief propagation to capture local and global changes.
    """
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
        
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
            """Compute energy for single generated network against target batch."""
            results = {}
            n_targets = target.shape[0]
            
            target = target.cpu().numpy()
            generated = generated[0].cpu().numpy() # to really get shape 100x100
            
            # Apply distance weighting. LONG RANGE CONNECTIONS ARE WEIGHTED MORE. 
            target = target * self.distance_matrix
            generated = generated * self.distance_matrix
            
            G_gen = nx.from_numpy_array(generated)
            # print(target.shape, generated.shape)

            for target_idx in range(n_targets):
                
                target_single = target[target_idx:target_idx+1, :, :][0] # to really get shape 100x100
                G_tar = nx.from_numpy_array(target_single)

                results[target_idx] = compute_deltacon(G_gen, G_tar)

            return results

    @property
    def metric_prefix(self) -> str:
        return "DeltaConDist"
    
