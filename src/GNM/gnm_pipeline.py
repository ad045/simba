"""
Generative network model (GNM) parameter search pipeline.

This module implements a flexible framework for fitting geometric generative
models to binarised connectomes.  Users can search over ranges of the
exponential distance penalty (η) and homophily/matching exponent (γ) using
either a dense grid or a random sampling strategy.  The pipeline also
supports resuming a previous sweep by loading existing results and
skipping already explored parameter pairs.  Two wiring rules are
available: the traditional homophily rule based on the Jaccard overlap of
neighbourhoods, and the matching index rule based on the normalised
overlap of neighbour sets.

Example
-------
```
from gnm_pipeline import run_gnm_search

# Load binarised connectomes (n_regions × n_regions × n_subjects) and distances
conn = np.load("connectomes_binarized_68x68_density_20_percent.npy")
dist = np.load("distance_matrix_68x68.npy")

# Run a random search over 100 (eta, gamma) pairs between the specified limits
results_df, best_df = run_gnm_search(
    conn,
    dist,
    eta_range=(-3.0, 0.0),
    gamma_range=(0.1, 0.6),
    search_type="random",
    n_samples=100,
    wiring_rule="homophily",
    continue_from=None,
    output_dir="./gnm_results",
    compute_mc=True,
)

# Plot the energy landscape
from plotting_functions import plot_energy_landscape_from_df
plot_energy_landscape_from_df(results_df, title="GNM energy landscape")
```
"""

from __future__ import annotations

import itertools
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from numba import njit
from scipy.stats import ks_2samp

from src.utils.saving_and_finding_files import time_stamp_for_saving, get_latest_file_name_of_data
from src.ESNs.test_memory_capacity_weighted import evaluate_memory_capacity


# -----------------------------------------------------------------------------
# Wiring rules
# -----------------------------------------------------------------------------

@njit(cache=True)
def _compute_homophily_bin(A_bin: np.ndarray) -> np.ndarray:
    """Compute homophily for a binary adjacency matrix.

    The homophily matrix H_{ij} is the Jaccard index of the neighbourhoods of
    nodes i and j: ``shared_ij / (deg_i + deg_j - shared_ij)``, where
    ``shared_ij`` is the number of common neighbours and ``deg_i`` is the degree
    of node i.  Diagonal entries are set to zero.

    Parameters
    ----------
    A_bin : np.ndarray
        Symmetric binary adjacency matrix.

    Returns
    -------
    np.ndarray
        Matrix of homophily values in the range [0, 1].
    """
    Af = A_bin.astype(np.float64)
    shared = Af @ Af
    deg = Af.sum(axis=1)
    denom = deg[:, None] + deg[None, :] - shared
    H = np.where(denom > 0.0, shared / denom, 0.0)
    # Remove self influence
    for i in range(H.shape[0]):
        H[i, i] = 0.0
    return H


@njit(cache=True)
def _compute_matching_index_bin(A_bin: np.ndarray) -> np.ndarray:
    """Compute the matching index for a binary adjacency matrix.

    The matching index M_{ij} quantifies the normalised overlap of the
    neighbourhoods of nodes i and j.  We define it here as

    ``M_{ij} = shared_ij / (deg_i + deg_j - 2*shared_ij)``,

    where ``shared_ij`` is the number of common neighbours of i and j and
    ``deg_i`` is the degree of node i.  This formulation emphasises
    overlapping neighbourhoods relative to the exclusive neighbours of each
    node.  Diagonal entries are zero.

    Parameters
    ----------
    A_bin : np.ndarray
        Symmetric binary adjacency matrix.

    Returns
    -------
    np.ndarray
        Matrix of matching index values in [0, 1].
    """
    Af = A_bin.astype(np.float64)
    shared = Af @ Af
    deg = Af.sum(axis=1)
    denom = deg[:, None] + deg[None, :] - 2.0 * shared
    M = np.where(denom > 0.0, shared / denom, 0.0)
    for i in range(M.shape[0]):
        M[i, i] = 0.0
    return M


@njit(cache=True)
def _rand_choice_weighted(p_vec: np.ndarray) -> int:
    """Randomly sample an index from a probability vector.

    Parameters
    ----------
    p_vec : np.ndarray
        One‑dimensional array of non‑negative weights summing to 1.

    Returns
    -------
    int
        Index sampled according to the weights.
    """
    tot = 0.0
    x = np.random.random()
    for k in range(p_vec.size):
        tot += p_vec[k]
        if x < tot:
            return k
    return p_vec.size - 1


@njit(cache=True)
def generate_gnm_numba(
    A_seed: np.ndarray,
    target_m: int,
    dist: np.ndarray,
    gamma: float,
    eta: float,
    wiring_rule: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Grow a synthetic network according to a geometric generative model.

    New edges are added to ``A_seed`` until the number of edges reaches
    ``target_m``.  The probability of forming an edge between two nodes i and j
    is proportional to ``dist[i, j]**eta * H[i, j]**gamma``, where H is
    either the homophily matrix or the matching index matrix depending on
    ``wiring_rule``.

    Parameters
    ----------
    A_seed : np.ndarray
        Starting binary adjacency matrix (zeros everywhere for an empty graph).
    target_m : int
        Desired number of edges in the final graph.
    dist : np.ndarray
        Matrix of distances between nodes; should be positive and symmetric.
    gamma : float
        Exponent applied to the homophily/matching term.
    eta : float
        Exponent applied to the distance term (negative values penalise long
        distances).
    wiring_rule : int
        Selects the wiring rule: ``0`` for homophily (Jaccard overlap) and
        ``1`` for matching index (normalised overlap).  Any other value
        defaults to homophily.
    eps : float, default ``1e-12``
        Small constant used to avoid division by zero and to prevent zero
        probabilities.

    Returns
    -------
    np.ndarray
        Synthetic binary adjacency matrix with ``target_m`` edges.
    """
    n = A_seed.shape[0]
    A = A_seed.copy()
    # Precompute cost term (distance penalty)
    D = np.maximum(dist, eps)
    cost_term = D ** eta
    for i in range(n):
        cost_term[i, i] = 0.0
    m = int(A.sum() // 2)
    while m < target_m:
        # Select wiring rule
        if wiring_rule == 1:
            H = _compute_matching_index_bin(A)
        else:
            H = _compute_homophily_bin(A)
        # Combine distance and homophily/matching into probabilities
        prob = cost_term * np.maximum(H, eps) ** gamma
        # Flatten into a vector and create a mask for available edges
        flat_prob = prob.ravel()
        mask_flat = (A == 0).ravel()
        # Remove diagonal and lower triangular entries from mask
        for i in range(n):
            idx_diag = i * n + i
            mask_flat[idx_diag] = False
            base = i * n
            for j in range(i):
                mask_flat[base + j] = False
        cand_idx = np.nonzero(mask_flat)[0]
        if cand_idx.size == 0:
            break
        p_vec = flat_prob[cand_idx]
        p_sum = p_vec.sum()
        # Normalise probabilities
        if p_sum > 0.0:
            p_vec /= p_sum
            k = cand_idx[_rand_choice_weighted(p_vec)]
        else:
            k = cand_idx[np.random.randint(cand_idx.size)]
        i, j = divmod(k, n)
        A[i, j] = 1
        A[j, i] = 1
        m += 1
    return A


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _binarize_matrix(A: np.ndarray, thr: float = 0.0) -> np.ndarray:
    """Symmetrise and binarise a weighted adjacency matrix.

    Parameters
    ----------
    A : np.ndarray
        Weighted adjacency matrix (may be asymmetric).
    thr : float, default ``0.0``
        Threshold for binarisation.  Entries greater than ``thr`` are set to 1;
        others to 0.  The matrix is first symmetrised and its diagonal is
        zeroed.

    Returns
    -------
    np.ndarray
        Binarised adjacency matrix with values in {0, 1}.
    """
    A_sym = (A + A.T) / 2.0
    np.fill_diagonal(A_sym, 0.0)
    return (A_sym > thr).astype(np.int8)


def _precompute_obs_metrics(A_bin: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute degree, clustering, betweenness and edge length of a binary graph.

    Parameters
    ----------
    A_bin : np.ndarray
        Binary adjacency matrix.
    dist : np.ndarray
        Distance matrix aligned with ``A_bin``.

    Returns
    -------
    tuple
        ``(degree, clustering, betweenness, edge_lengths)`` arrays used in the
        energy function.
    """
    G = nx.from_numpy_array(A_bin)
    deg = np.array([d for _, d in G.degree()])
    cc = np.array(list(nx.clustering(G).values()))
    bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    el = np.array([dist[i, j] for i, j in G.edges()])
    return deg, cc, bc, el


def _compute_energy(
    deg_obs: np.ndarray,
    cc_obs: np.ndarray,
    bc_obs: np.ndarray,
    el_obs: np.ndarray,
    A_sim: np.ndarray,
    dist: np.ndarray,
) -> float:
    """Compute the maximum KS statistic between observed and simulated metrics.

    Parameters
    ----------
    deg_obs, cc_obs, bc_obs, el_obs : np.ndarray
        Observed distributions of degree, clustering coefficient, betweenness
        centrality and edge lengths.
    A_sim : np.ndarray
        Simulated binary adjacency matrix.
    dist : np.ndarray
        Distance matrix.

    Returns
    -------
    float
        Maximum KS statistic across the four distributions.
    """
    B_sim = np.triu((A_sim > 0).astype(int), 1)
    B_sim += B_sim.T
    Gs = nx.from_numpy_array(B_sim)
    deg_sim = np.array([d for _, d in Gs.degree()])
    cc_sim = np.array(list(nx.clustering(Gs).values()))
    bc_sim = np.array(list(nx.betweenness_centrality(Gs, normalized=True).values()))
    el_sim = np.array([dist[i, j] for i, j in Gs.edges()])
    return max(
        ks_2samp(deg_obs, deg_sim).statistic,
        ks_2samp(cc_obs, cc_sim).statistic,
        ks_2samp(bc_obs, bc_sim).statistic,
        ks_2samp(el_obs, el_sim).statistic,
    )


def _generate_param_pairs(
    eta_range: Tuple[float, float],
    gamma_range: Tuple[float, float],
    *,
    search_type: str,
    n_eta: int,
    n_gamma: int,
    n_samples: int,
) -> List[Tuple[float, float]]:
    """Generate (η, γ) parameter pairs according to a search strategy.

    Parameters
    ----------
    eta_range, gamma_range : tuple of float
        (min, max) bounds for η and γ.
    search_type : {"grid", "random"}
        Type of search strategy.  ``"grid"`` yields ``n_eta * n_gamma`` evenly
        spaced pairs.  ``"random"`` yields ``n_samples`` pairs sampled
        uniformly within the bounds.
    n_eta, n_gamma : int
        Number of grid points along the η and γ axes for grid search.
    n_samples : int
        Number of samples for random search.

    Returns
    -------
    list of tuple
        List of (η, γ) pairs.
    """
    if search_type == "grid":
        eta_vals = np.linspace(eta_range[0], eta_range[1], n_eta)
        gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
        return [(float(e), float(g)) for e in eta_vals for g in gamma_vals]
    elif search_type == "random":
        rng = np.random.default_rng()
        return [
            (float(rng.uniform(eta_range[0], eta_range[1])), float(rng.uniform(gamma_range[0], gamma_range[1])))
            for _ in range(n_samples)
        ]
    else:
        raise ValueError("search_type must be 'grid' or 'random'")

# --- CSV appender (write header only once) ---
from pathlib import Path
def _append_rows_csv(rows, columns, path):
    path = Path(path)
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(
        path, mode="a", index=False, header=not path.exists()
    )

def _load_previous_results(
    previous_path: Optional[str],
    *,
    autodetect_dir: Optional[str],
    pattern: str = "gnm_grid_results_*.csv",
) -> Optional[pd.DataFrame]:
    """Load previous results from a CSV file.

    If ``previous_path`` is provided and points to an existing CSV, that file
    is loaded.  Otherwise, if ``autodetect_dir`` is provided, the most
    recently modified file matching ``pattern`` is located using
    :func:`get_latest_file_name_of_data`.  If no file is found, ``None``
    is returned.

    Parameters
    ----------
    previous_path : str, optional
        Explicit path to a previous results CSV.  If the file does not exist
        or is ``None``, autodetection will be attempted if ``autodetect_dir``
        is provided.
    autodetect_dir : str, optional
        Directory in which to search for previous results.  If provided and
        ``previous_path`` is ``None``, the most recent file matching
        ``pattern`` will be loaded.
    pattern : str, default ``"gnm_grid_results_*.csv"``
        Filename pattern used for autodetection.

    Returns
    -------
    pandas.DataFrame or None
        Previously saved results, or ``None`` if no file was found.
    """
    prev_df = None
    if previous_path is not None:
        path = Path(previous_path)
        if path.is_file():
            prev_df = pd.read_csv(path)
    elif autodetect_dir is not None:
        res = get_latest_file_name_of_data(autodetect_dir, pattern=pattern)
        if res is not None:
            latest_file, _ = res
            prev_df = pd.read_csv(latest_file)
    return prev_df


def run_gnm_search(
    connectomes: np.ndarray,
    dist: np.ndarray,
    *,
    eta_range: Tuple[float, float] = (-3.0, 0.0),
    gamma_range: Tuple[float, float] = (0.1, 0.6),
    search_type: str = "grid",
    n_eta: int = 20,
    n_gamma: int = 20,
    n_samples: int = 100,
    wiring_rule: str = "homophily",
    continue_from: Optional[str] = None,
    autodetect_dir: Optional[str] = None,
    output_dir: str = "./gnm_results",
    compute_mc: bool = True,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a geometric generative model to binarised connectomes.

    This high‑level routine searches parameter space for each subject, computes
    the energy of the resulting synthetic networks relative to the observed
    networks, and selects the best η/γ pair.  Optionally, the memory
    capacity of the best synthetic network can be evaluated by building an
    ESN on the observed connectome.  Results are returned both as a
    detailed grid (one row per parameter pair per subject) and as a summary
    of the best parameters and metrics per subject.

    Parameters
    ----------
    connectomes : np.ndarray
        Binarised or weighted connectomes of shape ``(n_regions, n_regions, n_subjects)``.
        If weighted, they will be binarised prior to model fitting.
    dist : np.ndarray
        Matrix of Euclidean distances between brain regions.  Must be
        symmetric and match the spatial dimensions of ``connectomes``.
    eta_range, gamma_range : tuple of float, optional
        Bounds (min, max) for η and γ.  These define the search space.
    search_type : {"grid", "random"}, default ``"grid"``
        Type of search: evenly spaced grid or random sampling.
    n_eta, n_gamma : int, optional
        Number of grid points along the η and γ axes for grid search.
    n_samples : int, optional
        Number of (η, γ) pairs sampled for random search.
    wiring_rule : {"homophily", "matching"}, default ``"homophily"``
        Wiring rule used to compute the attachment probability.  ``"homophily"``
        uses the Jaccard overlap of neighbourhoods; ``"matching"`` uses the
        matching index defined in :func:`_compute_matching_index_bin`.
    continue_from : str, optional
        Path to a previous results CSV file.  If provided, the function will
        skip parameter pairs already present in the file and append new
        evaluations to a new timestamped CSV in ``output_dir``.  Use
        ``autodetect_dir`` if you prefer automatic discovery of the latest
        results file.
    autodetect_dir : str, optional
        Directory in which to search for previous results via
        :func:`get_latest_file_name_of_data`.  Only used when
        ``continue_from`` is not provided.
    output_dir : str, default ``"./gnm_results"``
        Directory in which to save the generated CSV files.  New files will be
        created with a timestamp to avoid overwriting existing ones.
    compute_mc : bool, default ``True``
        If ``True``, evaluate the memory capacity of an ESN on each subject's
        observed connectome after selecting the best η/γ pair.
    random_state : int, optional
        Random seed used for the energy evaluation and MC computation.  Note
        that the Numba functions draw from a global random state and may not
        respect this seed exactly.
    verbose : bool, default ``True``
        If ``True``, print progress information and timing.

    Returns
    -------
    results_df : pandas.DataFrame
        Detailed results of the parameter search.  Columns include
        ``'subject'``, ``'eta'``, ``'gamma'``, ``'energy'``, and optionally
        ``'mc_mean'`` if ``compute_mc`` is ``True``.
    best_df : pandas.DataFrame
        Summary of the best parameters per subject.  Columns include
        ``'subject'``, ``'eta'``, ``'gamma'``, ``'energy'``, and optionally
        ``'mc_mean'``.  Additional graph measures returned by
        ``analyze_connectomes`` could be incorporated here if desired.
    """
    # Determine wiring rule index
    rule_idx = 0 if wiring_rule.lower().startswith("hom") else 1

    n_regions, _, n_subjects = connectomes.shape
    # Ensure distance matrix matches
    if dist.shape != (n_regions, n_regions):
        raise ValueError(
            f"Distance matrix shape {dist.shape} does not match connectomes ({n_regions},{n_regions})."
        )

    # Load previous results if resuming
    prev_df = _load_previous_results(continue_from, autodetect_dir, pattern="gnm_grid_results_*.csv")
    prev_pairs: set[Tuple[float, float]] = set()
    if prev_df is not None:
        # Round to a reasonable number of decimals to avoid floating point issues
        prev_pairs = set(
            (float(round(row["eta"], 6)), float(round(row["gamma"], 6))) for _, row in prev_df.iterrows()
        )
        if verbose:
            print(f"Loaded {len(prev_pairs)} parameter pairs from previous results.")

    # Generate parameter pairs for this search
    param_pairs = _generate_param_pairs(
        eta_range,
        gamma_range,
        search_type=search_type,
        n_eta=n_eta,
        n_gamma=n_gamma,
        n_samples=n_samples,
    )
    # Filter out previously evaluated pairs
    param_pairs = [
        (e, g)
        for (e, g) in param_pairs
        if (float(round(e, 6)), float(round(g, 6))) not in prev_pairs
    ]
    if verbose:
        print(f"Evaluating {len(param_pairs)} new parameter pairs per subject.")

    # Prepare output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time_stamp_for_saving()
    grid_path = out_dir / f"gnm_grid_results_{timestamp}.csv"
    best_path = out_dir / f"gnm_best_results_{timestamp}.csv"

    # Prepare result containers
    all_rows: List[Dict[str, float]] = []
    best_rows: List[Dict[str, float]] = []

    # Iterate over subjects
    for subj_idx in range(n_subjects):
        if verbose:
            print(f"Subject {subj_idx+1}/{n_subjects}:")
        A_obs = connectomes[:, :, subj_idx]
        A_bin = _binarize_matrix(A_obs)
        m_edges = int(A_bin.sum() // 2)
        deg_obs, cc_obs, bc_obs, el_obs = _precompute_obs_metrics(A_bin, dist)

        best_energy = np.inf
        best_eta: float = np.nan
        best_gamma: float = np.nan
        best_mc: Optional[float] = None

        # Evaluate parameter pairs
        for (eta_val, gamma_val) in param_pairs:
            # Generate synthetic network
            A_sim = generate_gnm_numba(
                np.zeros_like(A_bin, dtype=np.int8),
                m_edges,
                dist,
                gamma_val,
                eta_val,
                rule_idx,
            )
            # Compute energy
            energy = _compute_energy(deg_obs, cc_obs, bc_obs, el_obs, A_sim, dist)
            row = {
                "subject": subj_idx,
                "eta": eta_val,
                "gamma": gamma_val,
                "energy": energy,
            }
            all_rows.append(row)
            # Check if new best
            if energy < best_energy:
                best_energy = energy
                best_eta = eta_val
                best_gamma = gamma_val
        # After scanning all pairs, compute MC for the best if requested
        if compute_mc:
            # Build ESN weight matrix from observed connectome with best scaling
            # We use spectral_radius=0.99 by convention and rescale to 0.8 afterwards as
            # in the original pipeline.  The scaling factor can be tuned separately.
            from src.ESNs.generate_weight_matrices_bio_no_rank_weighted import build_weight_matrix_from_connectome
            W_obs = build_weight_matrix_from_connectome(A_obs, spectral_radius=0.99)
            # Rescale to spectral radius 0.8 for ESN
            eig_max = np.max(np.abs(np.linalg.eigvals(W_obs)))
            if eig_max > 0:
                W_obs *= 0.8 / eig_max
            mc_res = evaluate_memory_capacity(
                W_obs,
                n_lags=50,
                train_len=4000,
                test_len=1000,
                n_runs=10,
                spectral_radius=1.0,
                random_state=(random_state + subj_idx) if random_state is not None else None,
            )
            best_mc = float(mc_res.get("mc_mean", np.nan))
        # Save best row
        best_row = {
            "subject": subj_idx,
            "eta": best_eta,
            "gamma": best_gamma,
            "energy": best_energy,
        }
        if compute_mc:
            best_row["mc_mean"] = best_mc
        best_rows.append(best_row)
        # Optionally print subject summary
        if verbose:
            info = f"Best parameters (η={best_eta:.3f}, γ={best_gamma:.3f}) with energy={best_energy:.3f}"
            if compute_mc:
                info += f", MC={best_mc:.3f}"
            print(info)
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_rows)

    # Include previous results if any
    if prev_df is not None:
        results_df = pd.concat([prev_df, results_df], ignore_index=True)
        # Recompute best per subject including previous data
        best_df = (
            results_df.loc[results_df.groupby("subject")["energy"].idxmin()]
            .reset_index(drop=True)
        )
    # Save CSVs
    results_df.to_csv(grid_path, index=False)
    best_df.to_csv(best_path, index=False)
    if verbose:
        print(f"Saved detailed results to {grid_path}")
        print(f"Saved best parameters to {best_path}")

    return results_df, best_df




if __name__ == "__main__":
    import argparse, json, time, os
    from pathlib import Path
    import numpy as np
    import pandas as pd

    # Optional helper (autodetect latest prior sweep if you use continuation logic inside your pipeline)
    try:
        from src.utils.saving_and_finding_files import time_stamp_for_saving, get_latest_file_name_of_data
    except Exception:
        # Fallback if utils are not on PYTHONPATH
        def time_stamp_for_saving():
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        get_latest_file_name_of_data = None

    parser = argparse.ArgumentParser(
        description="GNM parameter sweep on binarised connectomes with timing and resumable partial outputs."
    )
    # Required IO
    parser.add_argument("--bin_connectomes", required=True,
                        help="Path to binarised connectomes array, shape (n,n,k) or (k,n,n).")
    parser.add_argument("--distance_matrix", required=True,
                        help="Path to distance matrix (n,n).")
    parser.add_argument("--save_dir", required=True,
                        help="Root directory to save outputs.")
    # Search ranges
    parser.add_argument("--eta_start", type=float, default=-3.0)
    parser.add_argument("--eta_end",   type=float, default=0.0)
    parser.add_argument("--n_eta",     type=int,   default=20)
    parser.add_argument("--gamma_start", type=float, default=0.1)
    parser.add_argument("--gamma_end",   type=float, default=0.6)
    parser.add_argument("--n_gamma",     type=int,   default=20)
    # Subset/extra metrics
    parser.add_argument("--include_subset", action="store_true", default=False,
                        help="Also compute graph measures on an additional coarse (eta,gamma) grid.")
    parser.add_argument("--subset_size", type=int, default=8)
    # Quality of life
    parser.add_argument("--timing", action="store_true", default=True,
                        help="Print per-subject timings as they are flushed to CSV.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run label to prefix the timestamped folder.")
    # (Optional) continuation helpers – your pipeline should decide what to do with these
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="If set, look here for latest partial CSVs to continue/append.")
    args = parser.parse_args()

    # ---------- Directories ----------
    t0_total = time.perf_counter()
    timestamp = time_stamp_for_saving() if 'time_stamp_for_saving' in globals() else time.strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(args.save-dir if hasattr(args, 'save-dir') else args.save_dir)  # allow both spellings
    save_root = root / (f"{args.run_name}_" if args.run_name else "") / f"gnm_etas{args.n_eta}_gammas{args.n_gamma}_{timestamp}"
    results_dir = save_root / "results"
    prelim_dir  = save_root / "preliminary_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    prelim_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load data ----------
    t0_load = time.perf_counter()
    conn = np.load(args.bin_connectomes)
    # normalise shape -> (k,n,n)
    conn = np.asarray(conn)
    if conn.ndim != 3:
        raise ValueError(f"bin-connectomes must be 3D, got shape {conn.shape}")
    if conn.shape[0] == conn.shape[1]:
        conn = np.transpose(conn, (2, 0, 1))
    dist = np.load(args.distance_matrix)
    t_load = time.perf_counter() - t0_load

    # ---------- Run sweep ----------
    t0_run = time.perf_counter()
    run_kwargs = dict(
        bin_connectomes=conn,
        distance_matrix=dist,
        n_eta=args.n_eta,
        n_gamma=args.n_gamma,
        eta_start=args.eta_start, eta_end=args.eta_end,
        gamma_start=args.gamma_start, gamma_end=args.gamma_end,
        subset_size=args.subset_size,
        include_subset=args.include_subset,
        timing_flag=args.timing,
        timestamp=timestamp,
        save_dir=prelim_dir,
    )

    # Support either function name in the module
    try:
        df_grid, df_best, df_subset = run_gnm_search(**run_kwargs)          # noqa: F821
    except NameError:
        df_grid, df_best, df_subset = run_gnm_full(**run_kwargs)     # noqa: F821

    t_run = time.perf_counter() - t0_run

    # ---------- Save final CSVs ----------
    t0_save = time.perf_counter()
    grid_path = results_dir / f"gnm_grid_results_{timestamp}.csv"
    best_path = results_dir / f"gnm_best_results_{timestamp}.csv"
    df_grid.to_csv(grid_path, index=False)
    df_best.to_csv(best_path, index=False)
    subset_path = None
    if df_subset is not None:
        subset_path = results_dir / f"gnm_subset_results_{timestamp}.csv"
        df_subset.to_csv(subset_path, index=False)
    t_save = time.perf_counter() - t0_save

    # ---------- Durations summary ----------
    total = time.perf_counter() - t0_total
    timing_summary = pd.DataFrame([{
        "timestamp": timestamp,
        "n_subjects": int(conn.shape[0]),
        "n_eta": int(args.n_eta),
        "n_gamma": int(args.n_gamma),
        "eta_start": float(args.eta_start), "eta_end": float(args.eta_end),
        "gamma_start": float(args.gamma_start), "gamma_end": float(args.gamma_end),
        "include_subset": bool(args.include_subset),
        "subset_size": int(args.subset_size),
        "t_load_s": float(t_load),
        "t_run_s": float(t_run),
        "t_save_s": float(t_save),
        "t_total_s": float(total),
        "grid_csv": str(grid_path),
        "best_csv": str(best_path),
        "subset_csv": (str(subset_path) if subset_path else ""),
    }])
    timing_summary_path = results_dir / f"gnm_timing_summary_{timestamp}.csv"
    timing_summary.to_csv(timing_summary_path, index=False)

    # Also store the exact CLI/config used
    (results_dir / f"gnm_run_config_{timestamp}.json").write_text(
        json.dumps(vars(args), indent=2)
    )

    print("\n=== GNM sweep finished ===")
    print(f"Loaded in  {t_load:8.3f} s")
    print(f"Run time   {t_run:8.3f} s")
    print(f"Saved in   {t_save:8.3f} s")
    print(f"TOTAL      {total:8.3f} s")
    print(f"Results →  {results_dir}")
    print(f"Prelim  →  {prelim_dir}")
