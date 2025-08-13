# NOT TRIED YET! 
# I also assume it is unnecessary, as I am going to use the weighted matrices, not any randomly generated ones.

import numpy as np

def build_weight_matrix_from_bin_conn(
    connectome: np.ndarray,
    *,
    spectral_radius: float = 0.99,
    rank: bool = False,
    random_state: int | None = None,
    scale_method: str = "power",   # "power" (fast, default) or "eig" (exact, O(N^3))
    power_iters: int = 100,
    power_tol: float = 1e-6,
) -> np.ndarray:
    """
    Construct a reservoir weight matrix from a (possibly non-binary) connectome.

    - Non-zero entries of `connectome` define the synapse pattern.
    - Weights are sampled ~ U[-1, 1] and assigned only at those positions.
    - Diagonal is zeroed (no self-connections).
    - Matrix is rescaled to have the requested spectral radius.

    Parameters
    ----------
    connectome : np.ndarray, shape (N, N)
    spectral_radius : float, default 0.99
    rank : bool, default False
        If True, preserve the rank order of original weights among nonzeros.
    random_state : int or None
    scale_method : {"power","eig"}, default "power"
        "power": fast power-iteration estimate of |lambda_max|.
        "eig":   exact spectral radius via dense eigendecomposition (slow for large N).
    power_iters : int, default 100
    power_tol : float, default 1e-6

    Returns
    -------
    W : np.ndarray, shape (N, N)
    """
    # Ensure float dtype (macOS Accelerate is optimized for float64 BLAS/LAPACK)
    C = np.asarray(connectome, dtype=float)
    N = C.shape[0]

    # Mask of allowed connections (exclude diagonal up front)
    mask = C != 0
    np.fill_diagonal(mask, False)

    # Flattened indices of nonzeros
    nz_idx = np.flatnonzero(mask)
    nnz = nz_idx.size

    rng = np.random.default_rng(random_state)

    # Allocate result
    W = np.zeros_like(C, dtype=float)
    W_flat = W.ravel()

    if nnz == 0:
        return W  # no edges → nothing to do

    if rank:
        # Rank order by original connectome's (nonzero) weights, descending
        C_flat = C.ravel()
        order = np.argsort(C_flat[nz_idx])[::-1]

        # Sample only what's needed, assign by sorted order (vectorized)
        r = rng.uniform(-1.0, 1.0, size=nnz)
        r.sort()               # ascending
        r = r[::-1]            # descending
        W_flat[nz_idx[order]] = r
    else:
        # Sample only for existing edges (vectorized)
        W_flat[nz_idx] = rng.uniform(-1.0, 1.0, size=nnz)

    # (Diagonal already zero and untouched.)

    # --- Spectral radius scaling ---
    def spectral_radius_power(A: np.ndarray, iters: int, tol: float) -> float:
        # Power iteration on possibly non-symmetric A to estimate |lambda_max|
        # Works well for random-like matrices used in ESNs.
        v = rng.standard_normal(A.shape[1])
        v /= np.linalg.norm(v) + 1e-12
        prev = 0.0
        for _ in range(iters):
            v_new = A @ v
            nrm = np.linalg.norm(v_new)
            if nrm == 0.0:
                return 0.0
            v = v_new / nrm
            # Rayleigh quotient magnitude as running estimate
            est = abs(v @ (A @ v))
            if abs(est - prev) <= tol * max(1.0, est):
                return est
            prev = est
        return prev if prev != 0.0 else est

    if spectral_radius is not None and spectral_radius > 0:
        if scale_method == "eig":
            # Exact spectral radius (costly: O(N^3))
            # Use eigvals (not eigvalsh) because W need not be symmetric.
            vals = np.linalg.eigvals(W)
            current_rho = float(np.max(np.abs(vals)))
        else:
            # Fast estimate
            current_rho = float(spectral_radius_power(W, power_iters, power_tol))

        if current_rho > 0:
            W *= (spectral_radius / current_rho)

    return W

