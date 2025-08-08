import numpy as np
import pandas as pd  # noqa: F401  # pandas is imported for completeness but not used directly
from typing import Iterable, List, Tuple, Dict
import time


def build_weight_matrix_from_bin_conn(
    connectome: np.ndarray,
    *,
    spectral_radius: float = 0.99,
    rank: bool = False,
    random_state: int | None = None,
) -> np.ndarray:
    """Construct a reservoir weight matrix from a binary (or non-binary!) connectome.

    All connectome entries != 0 are treated as potential synapses, weights are set randomly (uniformly distributed in [-1, 1]).
    The diagonal is zeroed out to ensure removal of self-connections. The resulting weight matrix is then rescaled to have the desired spectral radius.

    Parameters
    ----------
    connectome : np.ndarray, shape (N, N)
        Binary or weighted adjacency matrix describing the reservoir
        topology.  Non‑zero entries indicate the presence of a synapse.
        The diagonal will be zeroed.
    spectral_radius : float, default 0.99 (not necessary, I assume, as ESN generator will rescale it?)
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

    # # Rescale spectral radius if necessary
    # def _spectral_radius(matrix: np.ndarray, iterations: int = 100) -> float:
    #     # Estimate largest eigenvalue magnitude using power iteration
    #     vec = rng.normal(size=(matrix.shape[0],))
    #     vec /= np.linalg.norm(vec)
    #     for _ in range(iterations):
    #         vec = matrix @ vec
    #         norm = np.linalg.norm(vec)
    #         vec /= norm
    #     # Rayleigh quotient approximation
    #     return float(np.linalg.norm(matrix @ vec) / np.linalg.norm(vec))
    
    current_rho = _rho = np.max(np.abs(np.linalg.eigvalsh(W))) # spectral_radius(W)
    # current_rho = _spectral_radius_stable(W) 

    if current_rho > 0:
        W *= spectral_radius / current_rho
    return W




# def _spectral_radius_stable(matrix: np.ndarray,
#                         iterations: int = 100,
#                         rng: np.random.Generator | None = None,
#                         *,
#                         tol: float = 1e-12) -> float:
#     """Power-iteration with overflow / div-by-0 protection.

#     Returns 0.0 for (nearly) nilpotent matrices.
#     """
#     rng = np.random.default_rng() if rng is None else rng
#     vec = rng.normal(size=matrix.shape[0]).astype(np.float64, copy=False)

#     # ----- 1.  normalise the start vector safely
#     norm = np.linalg.norm(vec)
#     if norm < tol:
#         raise ValueError("Random start vector had zero norm — try again.")
#     vec /= norm

#     # ----- 2.  iterate
#     eps  = np.finfo(vec.dtype).eps        # machine epsilon
#     for _ in range(iterations):
#         vec = matrix @ vec

#         # 2a.  protect against NaN / Inf early
#         if not np.isfinite(vec).all():
#             raise FloatingPointError("Matrix-vector product produced NaN/Inf.")

#         norm = np.linalg.norm(vec)

#         # 2b.  very small ⇒ nilpotent, very large ⇒ scale down
#         if norm < tol:          # effectively zero → spectral radius is 0
#             return 0.0
#         if norm > 1e200:        # avoid overflow before the *next* multiply
#             vec *= 1e-200
#             norm *= 1e-200

#         vec /= norm             # re-normalise for the next loop

#     # ----- 3.  Rayleigh quotient ≈ last norm because vec is unit length
#     return float(norm)