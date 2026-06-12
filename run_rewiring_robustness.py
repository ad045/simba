"""
Parameter robustness under progressive rewiring of the reference (analysis S2)
==============================================================================

Question (Kayson's framing)
---------------------------
The whole benchmark rests on ONE empirical reference (the consensus connectome).
If that reference were slightly different - a few edges moved, as real measurement
noise would do - would each distance measure's recovered best-fit (eta, gamma)
move? We perturb the reference by progressively rewiring N edges, re-recover the
best-fit under every measure at each noise level, and watch how far the recovered
parameters drift. A trustworthy measure tolerates realistic edge noise before its
estimate moves; we report whether each measure degrades *gracefully* (a ramp) or
*falls off a cliff*, and turn it into a per-measure robustness score.

What is reused (spec Section 9 - save compute)
----------------------------------------------
  * The 8 selected measures, their evaluators, orientation handling, colours and
    names are imported verbatim from run_real_vs_artificial.py (analysis S3), which
    in turn uses the SAME evaluator objects as the main morphospace run. So a s=0
    recovery here reproduces the main-landscape best-fit bit-for-bit (verified:
    the Frobenius evaluator reproduces the cached summary_indiv values exactly).
  * Baseline best-fit p0(m) is loaded for free from the cached per-network distance
    CSVs (summary_indiv_<measure>_*.csv, 25,000 rows = 50 eta x 50 gamma x 10 reps)
    - NOT recomputed.
  * The 25,000 GNM adjacency matrices are loaded from the morphospace run's
    generated_networks/ (in the exact CSV row order) and stacked once to a cached
    memmap; we never regenerate them.

What is new
-----------
  For each noise level s and repeat r we build a perturbed reference C_s^(r) via a
  connectivity- and density-preserving rewiring of the consensus, then recover its
  best-fit by argmin over the replicate-averaged distance landscape - exactly the
  main-pipeline recovery convention. Drift = grid-step distance (Euclidean in
  (eta_idx, gamma_idx)) from p0, the SAME unit as the parameter-recovery analysis.

Perturbation models (spec Section 3)
------------------------------------
  (a) connected degree-preserving double-edge swap  [DEFAULT]
  (b) connected single-edge rewiring (remove edge / add non-edge)  [secondary]
  Both preserve edge count (=> density) and connectivity; (a) also preserves degree.

Compute control (spec Section 9)
--------------------------------
  Fast measures (frobenius, communicability_corr, delta_con, spectral_distance_
  adjacency) recover on the FULL 2,500-cell grid. Slow measures (portrait,
  net_simile, netrd_non_backtracking_spectral, energy) recover on a WINDOW of
  +/-W cells around p0; if the windowed argmin lands on the window edge at a
  low/mid noise level (s <= --fallback-cutoff) we widen to the full grid (a real
  early cliff must not be clipped). At high noise a boundary hit just means
  saturation, so we record a right-censored drift = W and a boundary flag instead
  of paying for a full-grid landscape. Parallelism is over the independent (s, r)
  draws within a measure.

Stages (all idempotent)
-----------------------
    stack    : build the cached 25,000-network GNM stack (once)
    compute  : per measure, sweep (s, r), recover, cache drift_curves.csv
    metrics  : aggregate drift curves -> rewiring_robustness_results.csv (+ meta)
    plot     : Panel A drift curves + Panel B robustness ranking -> PDF
    all      : the four above (default)

Usage
-----
    conda activate ma_thesis
    python run_rewiring_robustness.py --stage all
    python run_rewiring_robustness.py --measures energy --jobs 8
    python run_rewiring_robustness.py --noise-model single_edge
    python run_rewiring_robustness.py --smoke           # tiny fast self-test
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import argparse
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Reuse the S3 machinery verbatim (same evaluators / orientation / paths / style).
from run_real_vs_artificial import (
    build_evaluators,
    SELECTED_MEASURES,
    IS_SIMILARITY,
    METHOD_NAMES,
    METRIC_COLORS,
    CONSENSUS_PATH,
    MORPHO_DIR,
    MORPHO_EXP,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUT_DIR = ROOT_DIR / "output" / "rewiring_robustness"
FIG_DIR = ROOT_DIR / "figures" / "appendix"
GNM_DIR = MORPHO_DIR / "generated_networks"

N_NODES = 100
N_ETA = 50
N_GAMMA = 50
N_REPLICATES = 10
E_EDGES = 495                       # 10% density at 100 nodes: 0.10 * 100*99/2

# grid geometry (for axis labelling / param-unit conversion only)
ETA_RANGE = (-8.0, 3.0)
GAMMA_RANGE = (-0.1, 1.0)
ETA_STEP = (ETA_RANGE[1] - ETA_RANGE[0]) / (N_ETA - 1)      # ~0.224
GAMMA_STEP = (GAMMA_RANGE[1] - GAMMA_RANGE[0]) / (N_GAMMA - 1)  # ~0.022

# measures that recover on a window around p0 (everything else uses the full grid)
SLOW_MEASURES = {
    "portrait",
    "net_simile",
    "netrd_non_backtracking_spectral",
    "energy",
}

# default noise ladder (number of rewiring operations). The last entry is the
# near-full-randomisation sanity anchor and is EXCLUDED from N* / AUC.
DEFAULT_NOISE_LEVELS = [1, 2, 5, 10, 20, 50, 100, 200, E_EDGES]
FULL_ANCHOR = E_EDGES

R_DEFAULT = 20
R_SLOW = 10
WINDOW_DEFAULT = 10                 # +/- cells around p0 for slow measures
FALLBACK_CUTOFF_DEFAULT = 50        # widen to full grid on a boundary hit only if s <= this
DRIFT_THRESHOLD = 1.0               # grid steps; N* is the largest s with median_drift <= 1
BASE_SEED = 20260612

# canonical CSV that defines the GNM row order (filenames + eta + gamma + id)
CANON_CSV = MORPHO_DIR / f"summary_indiv_frobenius_for_exp_{MORPHO_EXP}.csv"


# ===========================================================================
# Grid bookkeeping
# ===========================================================================

def load_grid_layout() -> pd.DataFrame:
    """Per-network (network_index, filename, eta, gamma, id) + grid cell indices.

    Cell flat index = i_gamma * N_ETA + i_eta (gamma-outer / eta-inner), with the
    10 replicates (id) of a cell sharing one flat index.
    """
    df = pd.read_csv(CANON_CSV)[["network_index", "filename", "eta", "gamma", "id"]].copy()
    eta_vals = np.sort(df["eta"].unique())
    gamma_vals = np.sort(df["gamma"].unique())
    assert len(eta_vals) == N_ETA and len(gamma_vals) == N_GAMMA, \
        f"grid is {len(eta_vals)}x{len(gamma_vals)}, expected {N_ETA}x{N_GAMMA}"
    eta_to_i = {v: i for i, v in enumerate(eta_vals)}
    gamma_to_i = {v: i for i, v in enumerate(gamma_vals)}
    df["i_eta"] = df["eta"].map(eta_to_i).astype(int)
    df["i_gamma"] = df["gamma"].map(gamma_to_i).astype(int)
    df["cell"] = df["i_gamma"] * N_ETA + df["i_eta"]
    df.attrs["eta_vals"] = eta_vals
    df.attrs["gamma_vals"] = gamma_vals
    return df


def cell_to_idx(cell: int) -> Tuple[int, int]:
    return cell % N_ETA, cell // N_ETA          # (i_eta, i_gamma)


def griddist(cell_a: int, cell_b: int) -> float:
    """Euclidean distance in grid cells - the parameter-recovery unit (S1/S2 share)."""
    ea, ga = cell_to_idx(cell_a)
    eb, gb = cell_to_idx(cell_b)
    return float(np.hypot(ea - eb, ga - gb))


# ===========================================================================
# GNM stack
# ===========================================================================

def build_gnm_stack(layout: pd.DataFrame, stack_path: Path) -> np.ndarray:
    """Stack all 25,000 GNM adjacency matrices (CSV row order) to a cached .npy."""
    if stack_path.exists():
        arr = np.load(stack_path, mmap_mode="r")
        if arr.shape[0] == len(layout):
            print(f"  GNM stack cached -> {stack_path.name} {arr.shape}")
            return arr
    from tqdm import tqdm
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        stack_path, mode="w+", dtype="float32", shape=(len(layout), N_NODES, N_NODES))
    for k, fname in enumerate(tqdm(layout["filename"], desc="stacking GNMs")):
        m = np.load(GNM_DIR / fname)
        arr[k] = m[0] if m.ndim == 3 else m
    arr.flush()
    print(f"  Saved GNM stack -> {stack_path.name} {arr.shape}")
    return np.load(stack_path, mmap_mode="r")


# ===========================================================================
# Baseline best-fit p0 (free, from cached D_art)
# ===========================================================================

def baseline_bestfit(measure: str, layout: pd.DataFrame) -> int:
    """p0(m): argmin over the replicate-averaged cached landscape. Returns cell idx."""
    path = MORPHO_DIR / f"summary_indiv_{measure}_for_exp_{MORPHO_EXP}.csv"
    df = pd.read_csv(path)
    meta = {"network_index", "filename", "eta", "gamma", "id"}
    col = [c for c in df.columns if c not in meta]
    assert len(col) == 1, f"expected one metric column, got {col}"
    df = df[["eta", "gamma", "id"]].assign(raw=df[col[0]].astype(float))
    # attach the same cell indices as the layout via (eta, gamma)
    key = layout.drop_duplicates("cell").set_index(["eta", "gamma"])["cell"]
    df["cell"] = df.set_index(["eta", "gamma"]).index.map(key)
    df["oriented"] = df["raw"] if measure in IS_SIMILARITY else -df["raw"]  # lower=closer
    df["oriented"] = -df["oriented"]
    cell_mean = df.groupby("cell")["oriented"].mean()    # oriented distance per cell
    return int(cell_mean.idxmin())


# ===========================================================================
# Perturbation models (spec Section 3)
# ===========================================================================

def rewire(adj: np.ndarray, n_ops: int, seed: int, model: str) -> np.ndarray:
    """Connectivity- and density-preserving rewiring of a binary undirected graph."""
    import networkx as nx
    if n_ops <= 0:
        return adj.astype("float32").copy()
    G = nx.from_numpy_array((adj > 0).astype(int))
    G.remove_edges_from(nx.selfloop_edges(G))

    if model == "double_edge":
        # connected degree-preserving double-edge swap (default).
        nx.connected_double_edge_swap(G, nswap=n_ops, seed=int(seed))
    elif model == "single_edge":
        rng = np.random.default_rng(seed)
        nodes = list(G.nodes())
        for _ in range(n_ops):
            for _try in range(200):
                edges = list(G.edges())
                u, v = edges[rng.integers(len(edges))]
                x, y = rng.integers(len(nodes), size=2)
                x, y = nodes[x], nodes[y]
                if x == y or G.has_edge(x, y):
                    continue
                G.remove_edge(u, v)
                G.add_edge(x, y)
                if nx.is_connected(G):
                    break
                G.add_edge(u, v)      # revert: reconnect and undo
                G.remove_edge(x, y)
            else:
                continue
    else:
        raise ValueError(f"unknown noise model: {model}")

    B = nx.to_numpy_array(G, nodelist=range(adj.shape[0]))
    B = (B > 0).astype("float32")
    # density (edge count) and connectivity are invariants of the construction
    assert int(B.sum() // 2) == int((adj > 0).sum() // 2), "edge count changed!"
    assert nx.is_connected(nx.from_numpy_array(B)), "perturbed reference disconnected!"
    return B


# ===========================================================================
# Recovery: argmin over the replicate-averaged landscape (optionally windowed)
# ===========================================================================

# module-level worker state (populated by _init_worker)
_W: dict = {}


def _init_worker(measure, stack_path, fast, consensus, all_rows, all_cells, layout_cell):
    import torch  # noqa
    _W["measure"] = measure
    _W["evaluator"] = build_evaluators([measure])[measure]
    _W["stack"] = np.load(stack_path, mmap_mode="r")
    _W["is_sim"] = measure in IS_SIMILARITY
    _W["fast"] = fast
    _W["consensus"] = consensus
    _W["all_rows"] = all_rows
    _W["all_cells"] = all_cells
    _W["layout_cell"] = layout_cell


def _landscape_argmin(ref: np.ndarray, rows: np.ndarray, cells: np.ndarray
                      ) -> Tuple[int, int]:
    """argmin oriented-distance cell over the given GNM rows. Returns (cell, n_nan)."""
    import torch
    ev = _W["evaluator"]
    stack = _W["stack"]
    gen = torch.tensor(ref, dtype=torch.float32).unsqueeze(0)
    tgt = torch.tensor(np.asarray(stack[rows]), dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ev(gen, tgt)
    raw = np.array([out[i] for i in range(len(rows))], dtype=float)
    oriented = raw if _W["is_sim"] else -raw      # higher = closer
    oriented = -oriented                          # -> lower = closer (a distance)
    # per-cell mean over replicates, dropping NaNs
    n_nan = int(np.isnan(oriented).sum())
    dfm = pd.DataFrame({"cell": cells, "d": oriented})
    cell_mean = dfm.groupby("cell")["d"].mean()
    best = int(cell_mean.idxmin())
    return best, n_nan


def _recover(task) -> dict:
    """One (s, r) draw: build perturbed reference, recover best-fit, return drift row."""
    s, r, p0, model, window, fallback_cutoff = task
    consensus = _W["consensus"]
    all_rows = _W["all_rows"]
    all_cells = _W["all_cells"]
    layout_cell = _W["layout_cell"]

    seed = (BASE_SEED * 1_000 + s) * 1_000 + r
    ref = rewire(consensus, n_ops=s, seed=seed, model=model)

    fast = _W["fast"]
    censored = False
    if fast or window is None:
        cell, n_nan = _landscape_argmin(ref, all_rows, all_cells)
    else:
        p0e, p0g = cell_to_idx(p0)
        sel = (np.abs(layout_cell[:, 0] - p0e) <= window) & \
              (np.abs(layout_cell[:, 1] - p0g) <= window)
        rows = all_rows[sel]
        cells = all_cells[sel]
        cell, n_nan = _landscape_argmin(ref, rows, cells)
        ce, cg = cell_to_idx(cell)
        # window box edges, clipped to the global grid
        lo_e, hi_e = max(0, p0e - window), min(N_ETA - 1, p0e + window)
        lo_g, hi_g = max(0, p0g - window), min(N_GAMMA - 1, p0g + window)
        # a boundary hit only counts on a NON-clipped edge (clipped == real grid edge)
        on_edge = ((ce == lo_e and lo_e > 0) or (ce == hi_e and hi_e < N_ETA - 1) or
                   (cg == lo_g and lo_g > 0) or (cg == hi_g and hi_g < N_GAMMA - 1))
        # boundary hit: widen to full grid (low/mid noise) or right-censor (high noise)
        if on_edge:
            if s <= fallback_cutoff:
                cell, n_nan = _landscape_argmin(ref, all_rows, all_cells)
            else:
                censored = True

    drift = griddist(cell, p0)
    if censored:
        drift = float(window)         # right-censored lower bound at saturation
    ce, cg = cell_to_idx(cell)
    p0e, p0g = cell_to_idx(p0)
    return {
        "s": s, "repeat": r, "frac": min(1.0, 2.0 * s / E_EDGES),
        "cell": cell, "i_eta": ce, "i_gamma": cg,
        "d_eta": abs(ce - p0e), "d_gamma": abs(cg - p0g),
        "drift": drift, "n_nan": n_nan, "censored": censored,
        "eta": ETA_RANGE[0] + ce * ETA_STEP,
        "gamma": GAMMA_RANGE[0] + cg * GAMMA_STEP,
    }


def compute_measure(measure: str, layout: pd.DataFrame, consensus: np.ndarray,
                    stack_path: Path, noise_levels: List[int], R: int,
                    model: str, window: Optional[int], fallback_cutoff: int,
                    jobs: int) -> pd.DataFrame:
    """Sweep (s, r) for one measure -> long drift dataframe (one row per draw)."""
    from multiprocessing import Pool

    p0 = baseline_bestfit(measure, layout)
    fast = measure not in SLOW_MEASURES
    win = None if fast else window

    all_rows = layout["network_index"].to_numpy()
    all_cells = layout["cell"].to_numpy()
    layout_cell = np.stack([layout["i_eta"].to_numpy(), layout["i_gamma"].to_numpy()], axis=1)

    tasks = []
    for s in noise_levels:
        for r in range(R):
            tasks.append((s, r, p0, model, win, fallback_cutoff))

    print(f"  [{measure}] p0=cell {p0} {cell_to_idx(p0)}  fast={fast}  "
          f"window={win}  levels={noise_levels}  R={R}  draws={len(tasks)}")

    init_args = (measure, str(stack_path), fast, consensus, all_rows, all_cells, layout_cell)
    rows = []
    from tqdm import tqdm
    if jobs == 1:
        _init_worker(*init_args)
        for t in tqdm(tasks, desc=f"  {measure}"):
            rows.append(_recover(t))
    else:
        with Pool(processes=jobs, initializer=_init_worker, initargs=init_args) as pool:
            for row in tqdm(pool.imap_unordered(_recover, tasks, chunksize=1),
                            total=len(tasks), desc=f"  {measure}"):
                rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "measure", measure)
    df.insert(1, "p0_cell", p0)
    df["baseline_drift_check"] = 0.0      # s=0 has drift 0 by construction
    return df.sort_values(["s", "repeat"]).reset_index(drop=True)


# ===========================================================================
# Aggregation -> robustness metrics (spec Section 5)
# ===========================================================================

def _median_curve(df_m: pd.DataFrame) -> pd.DataFrame:
    """Per-s median / IQR drift, with the s=0 baseline prepended."""
    g = df_m.groupby("s")["drift"]
    cur = pd.DataFrame({
        "s": g.median().index,
        "median": g.median().values,
        "q25": g.quantile(0.25).values,
        "q75": g.quantile(0.75).values,
    })
    base = pd.DataFrame([{"s": 0, "median": 0.0, "q25": 0.0, "q75": 0.0}])
    cur = pd.concat([base, cur], ignore_index=True).sort_values("s").reset_index(drop=True)
    cur["frac"] = np.minimum(1.0, 2.0 * cur["s"] / E_EDGES)
    return cur


def _dispersion(cells: np.ndarray) -> float:
    """Mean pairwise grid distance among recovered cells at a fixed noise level."""
    if len(cells) < 2:
        return 0.0
    d = [griddist(int(a), int(b)) for i, a in enumerate(cells) for b in cells[i + 1:]]
    return float(np.mean(d))


def measure_scalars(df_m: pd.DataFrame, noise_levels: List[int], R: int,
                    model: str) -> dict:
    cur = _median_curve(df_m)
    # exclude the full-randomisation anchor from N* / AUC (saturation, not informative)
    core = cur[cur["s"] != FULL_ANCHOR].reset_index(drop=True)

    # N*: largest s (excl. anchor) whose median drift is still <= threshold
    tol = core[core["median"] <= DRIFT_THRESHOLD]["s"]
    n_star = int(tol.max()) if len(tol) else 0
    n_star_frac = min(1.0, 2.0 * n_star / E_EDGES)

    # drift AUC on the fraction axis, normalised by the tested fraction span
    frac = core["frac"].to_numpy()
    med = core["median"].to_numpy()
    span = frac.max() - frac.min()
    drift_auc = float(np.trapz(med, frac) / span) if span > 0 else float("nan")

    # cliff: max single-step rise vs total rise across the core ladder
    steps = np.diff(med)
    max_step = float(steps.max()) if len(steps) else 0.0
    changepoint_s = int(core["s"].to_numpy()[int(np.argmax(steps))]) if len(steps) else 0
    total_rise = float(med[-1] - med[0])
    cliff_flag = bool(total_rise > DRIFT_THRESHOLD and max_step > 0.5 * total_rise)

    # representative mid level for dispersion + per-axis drift (closest frac to 0.2)
    mid_s = int(min(noise_levels, key=lambda s: abs(min(1.0, 2.0 * s / E_EDGES) - 0.2)))
    at_mid = df_m[df_m["s"] == mid_s]
    dispersion_mid = _dispersion(at_mid["cell"].to_numpy())
    eta_drift_ref = float(at_mid["d_eta"].median()) if len(at_mid) else float("nan")
    gamma_drift_ref = float(at_mid["d_gamma"].median()) if len(at_mid) else float("nan")

    return {
        "noise_model": model,
        "R": R,
        "noise_tolerance_swaps": n_star,
        "noise_tolerance_frac": n_star_frac,
        "drift_auc": drift_auc,
        "max_step_rise": max_step,
        "cliff_flag": cliff_flag,
        "changepoint_s": changepoint_s,
        "dispersion_mid": dispersion_mid,
        "mid_level_s": mid_s,
        "eta_drift_at_ref": eta_drift_ref,
        "gamma_drift_at_ref": gamma_drift_ref,
        "n_nan_total": int(df_m["n_nan"].sum()),
        "n_censored": int(df_m["censored"].sum()),
        "median_drift_at_anchor": float(cur[cur["s"] == FULL_ANCHOR]["median"].iloc[0])
                                  if (cur["s"] == FULL_ANCHOR).any() else float("nan"),
    }


# ===========================================================================
# Figure (spec Section 6.2)
# ===========================================================================

def _viz():
    try:
        from vizman import viz
        viz.set_visual_style(font_family="Arial")
        return viz, True
    except Exception:
        return None, False


def plot(results: pd.DataFrame, curves: pd.DataFrame, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((18, 20)) if have_viz else (18 / 2.54, 20 / 2.54))
    GRAY = (0.5, 0.5, 0.5)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=figsize,
                                   gridspec_kw={"height_ratios": [1.3, 1.0]})

    # ---- Panel A: drift curves (median + IQR) vs fraction of edges rewired ----
    for m, g in curves.groupby("measure"):
        g = g.sort_values("frac")
        c = METRIC_COLORS.get(m, GRAY)
        x = g["frac"].clip(lower=g[g["frac"] > 0]["frac"].min() if (g["frac"] > 0).any() else 1e-3)
        axA.plot(x, g["median"], color=c, lw=1.6, zorder=3,
                 label=METHOD_NAMES.get(m, m))
        axA.fill_between(x, g["q25"], g["q75"], color=c, alpha=0.18, lw=0, zorder=2)
    axA.axhline(DRIFT_THRESHOLD, color=GRAY, ls="--", lw=0.9, zorder=1)
    axA.text(axA.get_xlim()[1], DRIFT_THRESHOLD, "  N* threshold (1 step)",
             va="bottom", ha="right", fontsize=6.5, color=GRAY)
    axA.set_xscale("log")
    axA.set_xlabel("fraction of edges rewired  (2s / E)")
    axA.set_ylabel("recovered-parameter drift  (grid steps)")
    axA.set_title("A   Parameter drift under progressive reference rewiring",
                  loc="left", fontsize=10, fontweight="bold")
    axA.legend(loc="upper left", frameon=False, fontsize=6.5, ncol=2)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # ---- Panel B: robustness ranking (noise tolerance, sorted desc) ----------
    res = results.sort_values("noise_tolerance_frac", ascending=True).reset_index(drop=True)
    y = np.arange(len(res))
    colors = [METRIC_COLORS.get(m, GRAY) for m in res["measure"]]
    bars = axB.barh(y, res["noise_tolerance_frac"], color=colors, edgecolor="black",
                    linewidth=0.4, height=0.7, zorder=2)
    for bar, cliff in zip(bars, res["cliff_flag"]):
        if cliff:
            bar.set_hatch("///")
    axB.set_yticks(y)
    axB.set_yticklabels([METHOD_NAMES.get(m, m) for m in res["measure"]], fontsize=8)
    axB.set_xlabel("noise tolerance  N*  (fraction of edges rewired before drift > 1 step)")
    for yi, v in zip(y, res["noise_tolerance_frac"]):
        axB.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=6.5)
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="///",
                              label="fails via cliff")],
               loc="lower right", frameon=False, fontsize=7)
    axB.set_title("B   Robustness-to-reference-noise ranking",
                  loc="left", fontsize=10, fontweight="bold")
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure -> {out_pdf}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["stack", "compute", "metrics", "plot", "all"],
                    default="all")
    ap.add_argument("--measures", default=None, help="comma-separated subset")
    ap.add_argument("--noise-model", choices=["double_edge", "single_edge"],
                    default="double_edge")
    ap.add_argument("--noise-levels", default=None,
                    help="comma-separated swap counts (default: spec ladder)")
    ap.add_argument("--R", type=int, default=R_DEFAULT)
    ap.add_argument("--R-slow", type=int, default=R_SLOW)
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    ap.add_argument("--fallback-cutoff", type=int, default=FALLBACK_CUTOFF_DEFAULT)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny self-test: 2 fast measures, short ladder, R=3")
    args = ap.parse_args()

    measures = (args.measures.split(",") if args.measures
                else (["frobenius", "delta_con"] if args.smoke else SELECTED_MEASURES))
    noise_levels = ([int(x) for x in args.noise_levels.split(",")] if args.noise_levels
                    else ([1, 5, 20] if args.smoke else DEFAULT_NOISE_LEVELS))

    out_dir = OUT_DIR if not args.smoke else (OUT_DIR.parent / "rewiring_robustness_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    stack_path = OUT_DIR / "gnm_stack.npy"        # shared across smoke + real
    curves_csv = out_dir / "drift_curves.csv"
    results_csv = out_dir / "rewiring_robustness_results.csv"
    meta_json = out_dir / "rewiring_robustness_meta.json"
    fig_pdf = (FIG_DIR / "fig_rewiring_robustness.pdf") if not args.smoke \
        else (out_dir / "fig_rewiring_robustness.pdf")

    print(f"Output dir : {out_dir}")
    print(f"Noise model: {args.noise_model}   levels: {noise_levels}   "
          f"measures: {measures}   jobs: {args.jobs}\n")

    layout = load_grid_layout()
    consensus = np.load(CONSENSUS_PATH)[0].astype("float32")
    assert int(consensus.sum() // 2) == E_EDGES, \
        f"consensus has {int(consensus.sum()//2)} edges, expected {E_EDGES}"

    if args.stage in ("stack", "compute", "all"):
        print("=" * 60, "\nStage: GNM stack\n", "=" * 60, sep="")
        build_gnm_stack(layout, stack_path)

    if args.stage in ("compute", "all"):
        print("\n" + "=" * 60, "\nStage: compute drift curves\n", "=" * 60, sep="")
        # resume: keep already-computed measures in the cache
        existing = pd.read_csv(curves_csv) if curves_csv.exists() else pd.DataFrame()
        done = set(existing["measure"].unique()) if len(existing) else set()
        all_parts = [existing] if len(existing) else []
        for m in measures:
            if m in done:
                print(f"  [{m}] cached -> skip")
                continue
            R = args.R_slow if m in SLOW_MEASURES else args.R
            df_m = compute_measure(
                m, layout, consensus, stack_path, noise_levels, R,
                args.noise_model, args.window, args.fallback_cutoff, args.jobs)
            all_parts.append(df_m)
            pd.concat(all_parts, ignore_index=True).to_csv(curves_csv, index=False)
            print(f"  [{m}] saved drift curves ({len(df_m)} draws)")
        print(f"\n  Drift curves -> {curves_csv.name}")

    if args.stage in ("metrics", "all"):
        print("\n" + "=" * 60, "\nStage: robustness metrics\n", "=" * 60, sep="")
        df = pd.read_csv(curves_csv)
        rows, curve_parts = [], []
        for m in [mm for mm in measures if mm in set(df["measure"].unique())]:
            df_m = df[df["measure"] == m]
            R = args.R_slow if m in SLOW_MEASURES else args.R
            sc = measure_scalars(df_m, noise_levels, R, args.noise_model)
            rows.append({"measure": m, "name": METHOD_NAMES.get(m, m), **sc})
            cur = _median_curve(df_m)
            cur.insert(0, "measure", m)
            curve_parts.append(cur)
        results = (pd.DataFrame(rows)
                   .sort_values("noise_tolerance_frac", ascending=False)
                   .reset_index(drop=True))
        results.to_csv(results_csv, index=False)
        pd.concat(curve_parts, ignore_index=True).to_csv(
            out_dir / "drift_curves_median.csv", index=False)
        meta = {
            "experiment": "S2 parameter robustness under progressive rewiring",
            "noise_model": args.noise_model,
            "noise_model_note": ("connected degree-preserving double-edge swap (spec 3a)"
                                 if args.noise_model == "double_edge"
                                 else "connected single-edge rewiring (spec 3b)"),
            "noise_levels": noise_levels,
            "full_randomisation_anchor": FULL_ANCHOR,
            "R_fast": args.R, "R_slow": args.R_slow,
            "slow_measures": sorted(SLOW_MEASURES),
            "window_cells": args.window,
            "fallback_cutoff_s": args.fallback_cutoff,
            "drift_threshold_steps": DRIFT_THRESHOLD,
            "grid": {"n_eta": N_ETA, "n_gamma": N_GAMMA, "eta_range": ETA_RANGE,
                     "gamma_range": GAMMA_RANGE, "eta_step": ETA_STEP,
                     "gamma_step": GAMMA_STEP},
            "E_edges": E_EDGES, "base_seed": BASE_SEED,
            "morphospace_run": f"{MORPHO_DIR.parent.name}/{MORPHO_EXP}",
            "baseline_source": "cached summary_indiv_<measure> CSVs (p0 not recomputed)",
            "measures": measures,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        meta_json.write_text(json.dumps(meta, indent=2, default=float))
        print(f"  Saved results -> {results_csv.name}")
        print(f"  Saved meta    -> {meta_json.name}\n")
        print(results.to_string(index=False))

    if args.stage in ("plot", "all"):
        print("\n" + "=" * 60, "\nStage: figure\n", "=" * 60, sep="")
        results = pd.read_csv(results_csv)
        curves = pd.read_csv(out_dir / "drift_curves_median.csv")
        plot(results, curves, fig_pdf)

    print("\nDone.")


if __name__ == "__main__":
    main()
