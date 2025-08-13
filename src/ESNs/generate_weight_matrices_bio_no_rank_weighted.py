import numpy as np

def build_weight_matrix_from_connectome(connectome: np.ndarray, *, spectral_radius: float = 0.99, zero_diagonal: bool = True, symmetrize: bool = True) -> np.ndarray:
    """Return W by preserving the (weighted) connectome entries and rescaling to the given spectral radius."""
    A = np.array(connectome, dtype=float, copy=True)
    if symmetrize:
        A = 0.5 * (A + A.T)
    if zero_diagonal:
        np.fill_diagonal(A, 0.0)
    if not np.any(A):
        return A
    try:
        lam_max = float(np.max(np.abs(np.linalg.eigvalsh(A))))
    except Exception:
        lam_max = float(np.max(np.abs(np.linalg.eigvals(A))))
    if lam_max > 0:
        A *= (spectral_radius / lam_max)
    return A

def build_weight_matrix_from_bin_conn(connectome: np.ndarray, *, spectral_radius: float = 0.99, rank: bool = False, random_state: int | None = None, use_connectome_weights: bool = False, symmetrize: bool = True) -> np.ndarray:
    """Legacy helper. If use_connectome_weights=True, preserve connectome weights; else assign random weights on the connectome mask."""
    if use_connectome_weights:
        return build_weight_matrix_from_connectome(connectome, spectral_radius=spectral_radius, zero_diagonal=True, symmetrize=symmetrize)
    rng = np.random.default_rng(random_state)
    A = np.array(connectome, dtype=float, copy=True)
    if symmetrize:
        A = 0.5 * (A + A.T)
    mask = (A != 0)
    random_weights = rng.uniform(-1.0, 1.0, size=A.shape)
    if rank:
        orig = A.flatten()
        rand = random_weights.flatten()
        nz = np.where(mask.flatten())[0]
        order = nz[np.argsort(-orig[nz])]
        rand_sorted = np.sort(rand[nz])[::-1]
        W = np.zeros_like(A, dtype=float)
        for idx, val in zip(order, rand_sorted):
            W.flat[idx] = val
    else:
        W = random_weights * mask
    np.fill_diagonal(W, 0.0)
    try:
        lam_max = float(np.max(np.abs(np.linalg.eigvalsh(W))))
    except Exception:
        lam_max = float(np.max(np.abs(np.linalg.eigvals(W))))
    if lam_max > 0:
        W *= (spectral_radius / lam_max)
    return W
