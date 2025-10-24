import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_comm

from bct import density_und

from networkx.algorithms import smallworld

import networkx as nx
import numpy as np


from netneurotools import modularity


def count_components_nx(A) -> int:
    G = nx.from_numpy_array(np.array(A))
    return nx.number_connected_components(G)


def compute_structural_metrics(A, metrics_to_analyze, distance_matrix=None):
    """
    Compute requested structural metrics for a single network.
    
    Args:
        A: numpy array (n, n)
        distance_matrix: numpy array (n, n) or None

    Returns:
        dict: {metric_name: value}
    """
    results = {}

    # Compute density
    if 'density' in metrics_to_analyze:
        results['density'] = density_und(A)

    # Compute small-worldness
    if 'small_world' in metrics_to_analyze:
        results['small_world'] = smallworld.small_worldness(A)

    # Compute modularity
    if distance_matrix is not None and 'modularity' in metrics_to_analyze:
        results['modularity'] = modularity.modularity(distance_matrix)

    if "num_components" in metrics_to_analyze:
        results['num_components'] = count_components_nx(A)


    return results
