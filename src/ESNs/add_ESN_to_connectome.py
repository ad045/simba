import numpy as np
import pandas as pd  # noqa: F401  # pandas is imported for completeness but not used directly
from typing import Iterable, List, Tuple, Dict
from echoes.esn import ESNRegressor
import time


def build_reservoir_from_connectome(
    connectome: np.ndarray,
    *,
    spectral_radius: float = 0.99,
    rank: bool = False,
    random_state: int | None = None,
) -> np.ndarray:
    """Construct a reservoir weight matrix from a binary connectome.

    All connectome entries != 0 are treated as potential synapses, weights are set randomly (uniformly distributed in [-1, 1]).
    The diagonal is zeroed out to ensure removal of self-connections. The resulting weight matrix is then rescaled to have the desired spectral radius.

    Parameters
    ----------
    connectome : np.ndarray, shape (N, N)
        Binary or weighted adjacency matrix describing the reservoir
        topology.  Non‑zero entries indicate the presence of a synapse.
        The diagonal will be zeroed.
    spectral_radius : float, default 0.99
        Desired spectral radius of the reservoir weight matrix; was close to (but less than) 1 in Damicelli's work. 
    rank : bool, default False
        If True, weights are assigned to preserve the rank order of the
        original weights (Bio (rank) condition).  If False, weights are
        assigned arbitrarily (Bio (no‑rank)).
    random_state : int or None, default None
        Seed for the pseudo‑random number generator.  Use this to make
        the weight initialisation reproducible.

    Returns
    -------
    W : np.ndarray, shape (N, N)
        Reservoir weight matrix whose non‑zero pattern matches the
        connectome.
    """
    
    # Create random weights 
    rng = np.random.default_rng(random_state) # Random number generator with given seed
    mask = connectome.astype(bool) # Create a mask of existing edges (True where connectome has connections)
    random_weights = rng.uniform(-1.0, 1.0, size=connectome.shape) # Random weights for present edges

    # Optionally: Rank‑order the random weights according to the connectome’s original weights
    if rank:
        # Flatten arrays for sorting
        orig = connectome.flatten()
        rand = random_weights.flatten()

        # Obtain sorting order of non-zero entries in the connectome
        nz_indices = np.where(mask.flatten())[0] 
        rank_order = nz_indices[np.argsort(-orig[nz_indices])] # Sort original weights to get rank order (descending)
        rand_sorted = np.sort(rand[nz_indices])[::-1] # Sort random values (descending) for assignment

        # Fill in new weight matrix according to rank order
        W = np.zeros_like(connectome, dtype=float) 
        for idx, rand_val in zip(rank_order, rand_sorted):
            W.flat[idx] = rand_val

    else:
        W = random_weights * mask

    # Remove self‑connections
    np.fill_diagonal(W, 0.0)

    # Rescale spectral radius if necessary
    def _spectral_radius(matrix: np.ndarray, iterations: int = 100) -> float:
        # Estimate largest eigenvalue magnitude using power iteration
        vec = rng.normal(size=(matrix.shape[0],))
        vec /= np.linalg.norm(vec)
        for _ in range(iterations):
            vec = matrix @ vec
            norm = np.linalg.norm(vec)
            vec /= norm
        # Rayleigh quotient approximation
        return float(np.linalg.norm(matrix @ vec) / np.linalg.norm(vec))
    current_rho = _spectral_radius(W)
    if current_rho > 0:
        W *= spectral_radius / current_rho
    return W

