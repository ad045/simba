"""
Robustness analysis for the NCT control-energy property.

The production property `calculate_nct_energies(A)` estimates control energy from
a SINGLE x0->xf transition fixed by the global np.random.seed(42). This module
provides the machinery to test whether the resulting (eta, gamma) landscape is
robust, by averaging over many independent transitions (calculate_nct_energies_multi)
and comparing against transition-independent controllability measures.

All heavy functions are module-level so they pickle cleanly under macOS 'spawn'
multiprocessing. Outputs are written ONLY to a dedicated robustness directory;
no existing property CSV or landscape is overwritten.

Author: robustness check (logs explicit RNG seeds; never touches np.random global).
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.extract_params_from_filenames import get_eta_gamma_id_from_filename
from src.analysis.dynamic_measures import (
    calculate_nct_energies,
    calculate_nct_energies_multi,
    calculate_nct_transition_independent,
)

# ── Paths / constants ────────────────────────────────────────────────────────
BASE = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
DATASET = "hcp_schaefer_100_dataset"
EXPERIMENT = "11_mst_2500_animal_0"
EXP_DIR = BASE / "output" / "gnm" / DATASET / EXPERIMENT
NET_DIR = EXP_DIR / "generated_networks"
OUT_DIR = EXP_DIR / "nct_energy_robustness"
FIG_DIR = OUT_DIR / "figures"
DYNAMIC_CSV = EXP_DIR / f"all_dynamic_metrics_for_{EXPERIMENT}_updated.csv"

# Master seed for the whole robustness experiment. Every per-network RNG is
# derived from this via np.random.default_rng(MASTER_SEED).spawn / jumped seeds,
# so the entire run is reproducible and logged.
MASTER_SEED = 20240611


# ── Network file index ───────────────────────────────────────────────────────
def list_network_files():
    """Return list of (eta, gamma, id, path) for every generated network."""
    out = []
    for name in sorted(os.listdir(NET_DIR)):
        if not name.endswith(".npy"):
            continue
        eta, gamma, net_id = get_eta_gamma_id_from_filename(name)
        if eta is None:
            continue
        out.append((eta, gamma, net_id, NET_DIR / name))
    return out


def load_A(path):
    return np.load(path)[0]


def representative_cells(n_eta=10, n_gamma=10, net_id=0):
    """Subsample the 50x50 grid to a representative (n_eta x n_gamma) set,
    taking the replicate `net_id` from each chosen cell.

    Returns list of (eta, gamma, id, path).
    """
    files = list_network_files()
    df = pd.DataFrame(files, columns=["eta", "gamma", "id", "path"])
    etas = np.sort(df.eta.unique())
    gams = np.sort(df.gamma.unique())
    eta_sel = etas[np.linspace(0, len(etas) - 1, n_eta).round().astype(int)]
    gam_sel = gams[np.linspace(0, len(gams) - 1, n_gamma).round().astype(int)]
    sub = df[df.eta.isin(eta_sel) & df.gamma.isin(gam_sel) & (df.id == net_id)]
    return list(sub.itertuples(index=False, name=None))


# ── Deterministic per-network seeding ────────────────────────────────────────
def network_rng(eta, gamma, net_id, tag=0):
    """A reproducible RNG unique to (eta, gamma, id, tag), derived from MASTER_SEED.

    Using SeedSequence with integer entropy keeps the whole experiment
    reproducible without ever touching the global np.random state.
    """
    key = (int(round(eta * 1e6)) & 0xFFFFFFFF,
           int(round(gamma * 1e6)) & 0xFFFFFFFF,
           int(net_id), int(tag))
    ss = np.random.SeedSequence([MASTER_SEED, *key])
    return np.random.default_rng(ss)


# ── Multiprocessing workers (module-level → picklable) ───────────────────────
def _worker_multi(args):
    """Compute the n_pairs-averaged energy for one network under one config."""
    eta, gamma, net_id, path, cfg = args
    try:
        A = load_A(path)
        rng = network_rng(eta, gamma, net_id, tag=cfg.get("tag", 0))
        r = calculate_nct_energies_multi(
            A,
            n_pairs=cfg["n_pairs"],
            rng=rng,
            T=cfg.get("T", 1.0),
            rho=cfg.get("rho", 1.0),
            state_dist=cfg.get("state_dist", "uniform"),
            B=cfg.get("B", None),
        )
        r.update({"eta": eta, "gamma": gamma, "id": net_id})
        return r
    except Exception as e:  # never let one network kill the run
        return {"eta": eta, "gamma": gamma, "id": net_id, "total": np.nan,
                "error": str(e)}


def _worker_transition_independent(args):
    eta, gamma, net_id, path = args
    try:
        A = load_A(path)
        r = calculate_nct_transition_independent(A, T=1.0)
        r.update({"eta": eta, "gamma": gamma, "id": net_id})
        return r
    except Exception as e:
        return {"eta": eta, "gamma": gamma, "id": net_id, "error": str(e)}


def _worker_single_seed42(args):
    """Recompute the ORIGINAL seed-42 single-pair estimator (for like-for-like
    comparison on the same replicates as the multi estimator)."""
    eta, gamma, net_id, path = args
    try:
        A = load_A(path)
        r = calculate_nct_energies(A)
        return {"eta": eta, "gamma": gamma, "id": net_id,
                "total": r["total"], "std": r["std"], "max": r["max"]}
    except Exception as e:
        return {"eta": eta, "gamma": gamma, "id": net_id, "total": np.nan,
                "error": str(e)}


# ── Grid helpers ─────────────────────────────────────────────────────────────
def run_pool(worker, tasks, n_processes=8, chunksize=8):
    """Run a picklable worker over tasks with a multiprocessing Pool."""
    from multiprocessing import Pool
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    results = []
    with Pool(processes=n_processes) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, tasks, chunksize=chunksize)):
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(tasks)} done", flush=True)
    return results


def cell_pivot(df, value_col, agg="mean"):
    """Realization-average over replicates -> 50x50 (eta x gamma) pivot."""
    p = df.groupby(["eta", "gamma"])[value_col].agg(agg).reset_index()
    return p.pivot(index="eta", columns="gamma", values=value_col)


def spearman_of_landscapes(pivot_a, pivot_b):
    """Spearman rho between two aligned (eta x gamma) pivots, over all cells."""
    from scipy.stats import spearmanr
    a = pivot_a.reindex_like(pivot_b)
    av = a.values.ravel()
    bv = pivot_b.values.ravel()
    m = np.isfinite(av) & np.isfinite(bv)
    return spearmanr(av[m], bv[m]).statistic
