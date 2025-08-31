# Was previously in 01_notebook. (Until 13.08.2005) 
# Consensus- and absolute-thresholding of structural networks re-implementation of Alexa Mousley’s MATLAB script

consensus_frac = 0.60 # Percentage threshold
min_count = int(np.floor(nsub * consensus_frac)) # Calculate threshold value based on the number of subjects

# mask of edges that appear in ≥ min_count subjects
edge_presence = (all_connectomes != 0).sum(axis=0) # Find nonzero elements in all connectomes, and making a mask out of it, (n, n)
consensus_mask = edge_presence >= min_count # mask of edges that appear in at least min_count subjects

# apply mask to every subject
consensus_thr = all_connectomes.copy()
consensus_thr[:, ~consensus_mask] = 0 # vectorised broadcasting

# quick sanity check — connection counts before / after
before = (all_connectomes != 0).sum(axis=(1, 2)) / 2 # Count the number of nonzero elements in each connectome, divided by 2 (as the matrix is symmetric)
after = (consensus_thr != 0).sum(axis=(1, 2)) / 2
print(f"Mean edges  before consensus: {before.mean():.1f}")
print(f"Mean edges after  consensus: {after.mean():.1f}")




############


import numpy as np
import bct 

def _threshold_absolute(W, thr):
    """Return a copy of W with weights < thr set to 0 (diagonal left untouched)."""
    Wthr = W.copy()
    Wthr[Wthr < thr] = 0
    return Wthr


def binarize_to_fixed_density(connectomes, target=0.10, max_iter=50, tol=1e-4):
    """
    Parameters
    ----------
    connectomes : (S, n, n) ndarray
        Weighted undirected adjacency matrices, one per subject (S = subjects).
    target : float
        Desired undirected density (0–1). 0.10 → 10 % of all possible edges.
    max_iter : int
        Safety cap on binary-search iterations.
    tol : float
        Stop if |density-target| < tol.

    Returns
    -------
    bin_conn : (S, n, n) int8
        Binarised matrices at exactly (or very close to) the target density.
    density  : (S,) float
        Final density per subject (should all be ≈ target).
    thr_used : (S,) float
        Absolute weight threshold chosen for each subject.
    """
    subj, n, _ = connectomes.shape
    bin_conn   = np.zeros_like(connectomes, dtype=np.int8)
    density    = np.zeros(subj)
    thr_used   = np.zeros(subj)

    for s in range(subj):
        W = connectomes[s]

        # extract all positive, upper-triangular weights (no diag) – sorted
        iu = np.triu_indices(n, k=1)
        weights = W[iu]
        weights = weights[weights > 0]

        if len(weights) == 0:
            raise ValueError(f"Subject {s}: matrix is empty (all zeros)")

        lo, hi = weights.min(), weights.max()
        best_thr = lo
        best_density = bct.density_und((W >= best_thr).astype(int))[0] # density, dim_1, dim_2.

        for _ in range(max_iter):
            mid = (lo + hi) / 2
            Wmid = _threshold_absolute(W, mid)
            dmid = bct.density_und(Wmid)[0] # density, dim_1, dim_2. 

            # update best if this is closer
            if abs(dmid - target) < abs(best_density - target):
                best_thr, best_density = mid, dmid

            # perfect (within tol) → done
            if abs(dmid - target) < tol:
                best_thr, best_density = mid, dmid
                break

            # binary-search decision
            if dmid >= target:
                # still too many edges → raise threshold
                lo = mid
            else:
                # too sparse → lower threshold
                hi = mid

        # apply best threshold, binarise and store
        Wsel = _threshold_absolute(W, best_thr)
        B    = (Wsel > 0).astype(np.int8)
        # force symmetry & zero diagonal (just in case)
        B    = np.triu(B, k=1)
        B    = B + B.T
        bin_conn[s]  = B
        density[s]   = best_density
        thr_used[s]  = best_thr

    return bin_conn, density, thr_used


