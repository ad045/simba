"""
Topographic biological-plausibility landscape (analysis S6b)
============================================================

A stronger version of the structural-gradient plausibility check
(`experiment_structural_gradient/`), addressing three limitations of it.

Why the first version is not enough
-----------------------------------
1. *No ceiling.* It scored each measure's best-fit networks against a null of
   arbitrary GNMs and reported a percentile. The best measure reached
   r = 0.62 while the null's 95th percentile is r = 0.68 - i.e. GNMs elsewhere
   in the (eta, gamma) plane reproduce the consensus topography *better* than
   any measure's optimum does. A percentile against the null cannot express
   that; a comparison against the achievable maximum can.
2. *One map.* The principal gradient is a single topography. Degree
   distribution fidelity says how many hubs a network has; only a node-wise
   comparison says whether they are in the right *places* - and the same holds
   for clustering, betweenness and connection length.
3. *One cell.* Scoring a single argmin cell says nothing about where the
   topographic optimum actually lies, so it cannot ask the diagnostic question:
   does a measure's distance minimum *point at* the topographically best part
   of the morphospace?

What this does instead
----------------------
For a battery of node-level maps - each one scalar per node - we correlate the
map of every GNM grid point against the same map of the empirical consensus,
across the 100 nodes. Averaging the per-map correlations gives one
*topographic-plausibility landscape* over the full 50 x 50 (eta, gamma) grid.

    maps:  principal structural gradient (diffusion embedding; sign-aligned)
           nodal degree                  (hub placement, not hub count)
           clustering coefficient        (local segregation topography)
           betweenness centrality        (integrative role placement)
           mean connection length        (spatial embedding topography)

The landscape then yields, per measure, three numbers that the single-cell
version could not produce:

    r_topo      topographic agreement at the measure's own best-fit cell
    efficiency  r_topo / r_max, with r_max the best value any GNM in the grid
                attains - i.e. how much of the achievable topography the
                measure captures
    offset      Euclidean distance in grid-index space between the measure's
                distance minimum and the topographic optimum, in grid steps -
                the same unit as the parameter-recovery experiments

Because the whole grid is evaluated, the "any GNM" null is exact (the grid
itself) rather than a 1,500-network sample.

Reused vs. new
--------------
  * D_art (per-network distances -> each measure's best fit) is already cached.
  * The 25,000 GNM adjacency matrices are already on disk.
  * `structural_gradient()` is imported from the S6 script, unchanged.
  * New: the four additional nodal maps and the full-grid sweep.

Stages (all idempotent)
-----------------------
    landscape : consensus maps + per-cell map correlations -> landscape CSV/npz
    metrics   : per-measure scores -> topographic_results.csv (+ meta JSON)
    plot      : landscape + ranking + per-map breakdown -> PDF
    all       : the three above (default)

Usage
-----
    conda activate ma_thesis
    python run_topographic_plausibility.py --stage all
    python run_topographic_plausibility.py --n-reps 10      # all replicates
    python run_topographic_plausibility.py --smoke          # fast self-test
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
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments_config import (
    MORPHO_DATASET, MORPHO_EXP, MORPHO_DIR,
    CONSENSUS_PATH, DIST_MATRIX_PATH,
    SELECTED_MEASURES, IS_SIMILARITY, METHOD_NAMES, METRIC_COLORS,
    ROOT_DIR as CFG_ROOT,
)
# the diffusion-map gradient + the best-fit lookup are reused as-is
from experiment_structural_gradient.run_structural_gradient import (
    structural_gradient, best_fit_rows, load_net,
)

OUT_DIR = CFG_ROOT / "output" / "topographic_plausibility"

# node-level maps making up the battery; `gradient` is the only one whose sign
# is arbitrary (eigenvector), so it is the only one that gets sign-aligned.
MAP_NAMES = ["gradient", "degree", "clustering", "betweenness", "edge_length"]
MAP_LABELS = {
    "gradient": "principal gradient",
    "degree": "degree",
    "clustering": "clustering",
    "betweenness": "betweenness",
    "edge_length": "connection length",
}
SIGN_FREE = {"gradient"}          # arbitrary sign -> align before correlating
TOP_PCT = 99.0                    # percentile defining the achievable ceiling


# ===========================================================================
# Node-level maps
# ===========================================================================

def nodal_maps(A: np.ndarray, D: np.ndarray) -> Dict[str, np.ndarray]:
    """The topographic battery for one binary network (one scalar per node)."""
    import networkx as nx

    A = np.asarray(A, dtype=float)
    if A.ndim == 3:
        A = A[0]
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)

    deg = A.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # mean Euclidean length of a node's connections; undefined (nan) for
        # isolated nodes, which the correlation then drops.
        edge_len = (A * D).sum(axis=1) / np.where(deg > 0, deg, np.nan)

    G = nx.from_numpy_array(A)
    clust = np.array([v for _, v in sorted(nx.clustering(G).items())])
    betw = np.array([v for _, v in sorted(nx.betweenness_centrality(G).items())])

    return {
        "gradient": structural_gradient(A),
        "degree": deg,
        "clustering": clust,
        "betweenness": betw,
        "edge_length": edge_len,
    }


def _corr(a: np.ndarray, b: np.ndarray, sign_free: bool = False) -> float:
    """Pearson r over nodes finite in both maps; sign-aligned if requested.

    A constant map (e.g. clustering all-zero) has no defined correlation and
    returns nan rather than silently scoring 0.
    """
    from scipy.stats import pearsonr
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 5:
        return float("nan")
    x, y = a[finite], b[finite]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    r = float(pearsonr(x, y)[0])
    return abs(r) if sign_free else r


# ===========================================================================
# Stage: landscape
# ===========================================================================

def compute_landscape(measures: List[str], n_reps: int, land_csv: Path,
                      npz_path: Path) -> None:
    from tqdm import tqdm

    D = np.load(DIST_MATRIX_PATH)
    consensus = np.load(CONSENSUS_PATH)
    ref = nodal_maps(consensus, D)
    print(f"  Consensus maps computed ({len(ref)} maps, N={ref['degree'].size} nodes).")

    # filenames/grid are identical across measures -> read one summary as pool
    pool = pd.read_csv(MORPHO_DIR / f"summary_indiv_{measures[0]}_for_exp_{MORPHO_EXP}.csv")
    pool = pool[["filename", "eta", "gamma", "id"]]
    pool = pool[pool["id"] < n_reps]

    rows = []
    for (eta, gamma), grp in tqdm(pool.groupby(["eta", "gamma"]),
                                  desc="topographic landscape"):
        per_rep = {m: [] for m in MAP_NAMES}
        for _, r in grp.iterrows():
            maps = nodal_maps(load_net(r["filename"]), D)
            for m in MAP_NAMES:
                per_rep[m].append(_corr(maps[m], ref[m], sign_free=m in SIGN_FREE))
        row = {"eta": float(eta), "gamma": float(gamma),
               "n_reps": int(len(grp))}
        with np.errstate(invalid="ignore"):
            for m in MAP_NAMES:
                row[m] = float(np.nanmean(per_rep[m]))
            row["combined"] = float(np.nanmean([row[m] for m in MAP_NAMES]))
        rows.append(row)

    land = pd.DataFrame(rows)
    land.to_csv(land_csv, index=False)
    print(f"  Saved landscape ({len(land)} grid points) -> {land_csv.name}")

    # consensus maps kept for the plot (panel of reference topographies)
    np.savez(npz_path, **{f"ref__{m}": ref[m] for m in MAP_NAMES})
    print(f"  Saved consensus maps -> {npz_path.name}")


# ===========================================================================
# Stage: metrics
# ===========================================================================

def cell_means(measure: str) -> pd.Series:
    """Per-(eta, gamma) mean of a measure's cached distance to the consensus."""
    df = pd.read_csv(MORPHO_DIR / f"summary_indiv_{measure}_for_exp_{MORPHO_EXP}.csv")
    col = [c for c in df.columns
           if c not in {"network_index", "filename", "eta", "gamma", "id"}][0]
    return df.groupby(["eta", "gamma"])[col].mean()


def _grid_index(land: pd.DataFrame):
    etas = np.sort(land["eta"].unique())
    gammas = np.sort(land["gamma"].unique())
    return etas, gammas


def compute_metrics(land: pd.DataFrame, measures: List[str]) -> pd.DataFrame:
    etas, gammas = _grid_index(land)
    key = land.set_index(["eta", "gamma"])

    comb = key["combined"].to_numpy(dtype=float)
    r_med = float(np.nanmedian(comb))

    # The single best cell is not a stable target: the top of the landscape is a
    # ridge, and which cell wins it flips with the number of replicates
    # averaged. The optimum is therefore the *region* of cells at or above the
    # 99th percentile of the grid, and the ceiling is that percentile.
    r_max = float(np.nanpercentile(comb, TOP_PCT))
    land_idx = key.reset_index()
    top_cells = land_idx[land_idx["combined"] >= r_max]
    ti = np.array([np.argmin(np.abs(etas - e)) for e in top_cells["eta"]])
    tj = np.array([np.argmin(np.abs(gammas - g)) for g in top_cells["gamma"]])
    eta_top = float(top_cells["eta"].mean())
    gamma_top = float(top_cells["gamma"].mean())
    print(f"  Topographic optimum region: {len(top_cells)} cells, "
          f"eta {top_cells['eta'].mean():+.2f}+-{top_cells['eta'].std():.2f}, "
          f"gamma {top_cells['gamma'].mean():.2f}+-{top_cells['gamma'].std():.2f}; "
          f"ceiling r={r_max:.3f}, grid median r={r_med:.3f}")

    from scipy.stats import spearmanr

    rows = []
    for m in measures:
        eta_star, gamma_star, _ = best_fit_rows(m)
        cell = key.loc[(eta_star, gamma_star)]
        r_topo = float(cell["combined"])

        # Whole-landscape agreement: does the measure call a grid point close to
        # the consensus *because* that grid point reproduces its topography?
        # Spearman over all grid cells, with the measure's landscape oriented so
        # that higher always means "better fit" (distances are negated).
        fit = cell_means(m)
        if m not in IS_SIMILARITY:
            fit = -fit
        joint = pd.concat([fit.rename("fit"), key["combined"]], axis=1).dropna()
        rho = float(spearmanr(joint["fit"], joint["combined"])[0])

        row = {
            "measure": m, "name": METHOD_NAMES.get(m, m),
            "eta": eta_star, "gamma": gamma_star,
            "r_topo": r_topo,
            "efficiency": r_topo / r_max if r_max > 0 else float("nan"),
            "landscape_rho": rho,
            "grid_percentile": float((comb < r_topo).mean() * 100.0),
        }
        for mp in MAP_NAMES:
            row[f"r_{mp}"] = float(cell[mp])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("r_topo", ascending=False)
    # per-map ceiling: the best any GNM in the grid reaches on that map alone
    # (each map's maximum can sit in a different cell - it bounds the criterion,
    #  it is not one network that achieves all five).
    df.attrs["map_max"] = {mp: float(np.nanpercentile(key[mp].to_numpy(dtype=float),
                                                      TOP_PCT))
                           for mp in MAP_NAMES}
    df.attrs["n_top_cells"] = int(len(top_cells))
    df.attrs["r_max"] = r_max
    df.attrs["r_median"] = r_med
    df.attrs["eta_top"] = eta_top
    df.attrs["gamma_top"] = gamma_top
    return df.reset_index(drop=True)


# ===========================================================================
# Stage: figure
# ===========================================================================

def _viz():
    try:
        from vizman import viz
        viz.set_visual_style(font_family="Arial")
        return viz, True
    except Exception:
        return None, False


def plot(res: pd.DataFrame, land: pd.DataFrame, out_pdf: Path,
         summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((18, 21)) if have_viz else (18 / 2.54, 21 / 2.54))
    GRAY = (0.5, 0.5, 0.5)

    etas, gammas = _grid_index(land)
    grid = (land.pivot(index="gamma", columns="eta", values="combined")
                .reindex(index=gammas, columns=etas).to_numpy())

    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(3, 1, height_ratios=[1.5, 1.0, 1.0], hspace=0.06)

    # ---- Panel A: topographic-plausibility landscape ------------------------
    axA = subfigs[0].subplots(1, 1)
    subfigs[0].subplots_adjust(left=0.11, right=0.86, top=0.86, bottom=0.14)
    im = axA.pcolormesh(etas, gammas, grid, cmap="viridis", shading="auto")
    cb = subfigs[0].colorbar(im, ax=axA, pad=0.02)
    cb.set_label("topographic agreement\n(mean r over the 5 maps)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    axA.axvline(0.0, color="white", lw=0.8, ls=":")
    top = land[land["combined"] >= summary["r_max"]]
    axA.scatter(top["eta"], top["gamma"], marker="+", s=26, color="white",
                linewidths=0.9, zorder=5,
                label=f"topographic optimum region (top 1%, r$\\geq${summary['r_max']:.2f})")
    for _, r in res.iterrows():
        axA.scatter([r["eta"]], [r["gamma"]], s=42,
                    color=METRIC_COLORS.get(r["measure"], GRAY),
                    edgecolor="white", linewidths=0.8, zorder=6)
        axA.annotate(r["name"], (r["eta"], r["gamma"]),
                     textcoords="offset points", xytext=(5, 4), fontsize=6,
                     color="white", path_effects=[pe.withStroke(linewidth=1.6,
                                                                foreground="black")])
    axA.set_xlabel(r"$\eta$", fontsize=9)
    axA.set_ylabel(r"$\gamma$", fontsize=9)
    axA.tick_params(labelsize=7)
    axA.legend(loc="upper left", frameon=False, fontsize=6.5, labelcolor="white")
    axA.set_title("A   Topographic plausibility across the morphospace, with "
                  "each measure's best fit", loc="left", fontsize=9.5,
                  fontweight="bold", pad=6)

    # ---- Panel B: ranking against ceiling and grid median -------------------
    axB = subfigs[1].subplots(1, 1)
    subfigs[1].subplots_adjust(left=0.24, right=0.97, top=0.82, bottom=0.24)
    y = np.arange(len(res))[::-1]
    colors = [METRIC_COLORS.get(m, GRAY) for m in res["measure"]]
    axB.axvline(summary["r_max"], color="black", lw=1.0, ls="-",
                label=f"achievable ceiling, top 1% of grid ({summary['r_max']:.2f})")
    axB.axvline(summary["r_median"], color=GRAY, lw=1.0, ls="--",
                label=f"grid median ({summary['r_median']:.2f})")
    axB.barh(y, res["r_topo"], color=colors, edgecolor="black", linewidth=0.4,
             height=0.7, zorder=2)
    axB.set_yticks(y)
    axB.set_yticklabels(res["name"], fontsize=7.5)
    axB.set_xlabel("topographic agreement at the measure's best fit "
                   "(mean r over the 5 maps)", fontsize=7.5)
    axB.tick_params(labelsize=7)
    for yi, (rt, ef, rho) in enumerate(zip(res["r_topo"][::-1],
                                           res["efficiency"][::-1],
                                           res["landscape_rho"][::-1])):
        axB.text(rt + 0.005, yi, f"{rt:.2f}  ({ef*100:.0f}% of ceiling, "
                                 f"landscape $\\rho$={rho:+.2f})", va="center",
                 fontsize=6)
    axB.set_xlim(0, max(summary["r_max"], res["r_topo"].max()) * 2.05)
    axB.legend(loc="lower right", frameon=False, fontsize=6.5)
    axB.set_title("B   Topographic agreement at each measure's best fit, "
                  "against the achievable ceiling",
                  loc="left", fontsize=9.5, fontweight="bold", pad=6)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # ---- Panel C: per-map breakdown ----------------------------------------
    axC = subfigs[2].subplots(1, 1)
    subfigs[2].subplots_adjust(left=0.24, right=0.90, top=0.82, bottom=0.26)
    M = res[[f"r_{m}" for m in MAP_NAMES]].to_numpy(dtype=float)
    ceiling = np.array([[summary["map_max"][m] for m in MAP_NAMES]])
    M = np.vstack([M, ceiling])          # last row: what any GNM can reach
    imC = axC.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axC.axhline(len(res) - 0.5, color="black", lw=1.0)
    axC.set_xticks(range(len(MAP_NAMES)))
    axC.set_xticklabels([MAP_LABELS[m] for m in MAP_NAMES], fontsize=7,
                        rotation=20, ha="right")
    axC.set_yticks(range(len(res) + 1))
    axC.set_yticklabels(list(res["name"]) + ["best GNM anywhere"], fontsize=7.5)
    axC.get_yticklabels()[-1].set_style("italic")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                axC.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                         fontsize=6,
                         color="white" if abs(M[i, j]) > 0.55 else "black")
    cbC = subfigs[2].colorbar(imC, ax=axC, pad=0.02)
    cbC.set_label("node-wise r to consensus", fontsize=7)
    cbC.ax.tick_params(labelsize=7)
    axC.set_title("C   Which topographies each measure's best fit reproduces",
                  loc="left", fontsize=9.5, fontweight="bold", pad=6)

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
    ap.add_argument("--stage", choices=["landscape", "metrics", "plot", "all"],
                    default="all")
    ap.add_argument("--n-reps", type=int, default=3,
                    help="replicates averaged per grid point (max 10)")
    ap.add_argument("--measures", default=None, help="comma-separated subset")
    ap.add_argument("--smoke", action="store_true",
                    help="1 replicate, 3 measures, separate output dir")
    args = ap.parse_args()

    measures = (args.measures.split(",") if args.measures
                else (["frobenius", "delta_con", "portrait"] if args.smoke
                      else SELECTED_MEASURES))
    n_reps = 1 if args.smoke else args.n_reps

    out_dir = OUT_DIR if not args.smoke else (OUT_DIR.parent / "topographic_plausibility_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    land_csv = out_dir / "topographic_landscape.csv"
    npz_path = out_dir / "consensus_maps.npz"
    results_csv = out_dir / "topographic_results.csv"
    meta_json = out_dir / "topographic_meta.json"
    fig_pdf = out_dir / "fig_topographic_plausibility.pdf"

    print(f"Output dir : {out_dir}")
    print(f"Measures   : {measures}   n_reps: {n_reps}\n")

    if args.stage in ("landscape", "all"):
        print("=" * 60, "\nStage: landscape\n", "=" * 60, sep="")
        compute_landscape(measures, n_reps, land_csv, npz_path)

    if args.stage in ("metrics", "all"):
        print("\n" + "=" * 60, "\nStage: metrics\n", "=" * 60, sep="")
        land = pd.read_csv(land_csv)
        res = compute_metrics(land, measures)
        res.to_csv(results_csv, index=False)
        meta = {
            "criterion": "topographic plausibility (node-wise map battery)",
            "maps": MAP_NAMES,
            "sign_free_maps": sorted(SIGN_FREE),
            "combined": "unweighted mean of the per-map Pearson r",
            "reference": "empirical HCP consensus (10% density)",
            "best_fit_rule": "argmin mean distance over replicates "
                             "(argmax for communicability_corr)",
            "null": "the full 50x50 grid itself (exact, not sampled)",
            "n_reps_per_grid_point": n_reps,
            "map_max": res.attrs["map_max"],
            "r_max": res.attrs["r_max"],
            "r_median": res.attrs["r_median"],
            "eta_top": res.attrs["eta_top"],
            "gamma_top": res.attrs["gamma_top"],
            "morphospace_run": f"{MORPHO_DATASET}/{MORPHO_EXP}",
            "measures": measures,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        meta_json.write_text(json.dumps(meta, indent=2))
        print(f"\n  Saved results -> {results_csv.name}")
        print(f"  Saved meta    -> {meta_json.name}\n")
        print(res.to_string(index=False))

    if args.stage in ("plot", "all"):
        print("\n" + "=" * 60, "\nStage: figure\n", "=" * 60, sep="")
        land = pd.read_csv(land_csv)
        res = pd.read_csv(results_csv)
        summary = json.loads(meta_json.read_text())
        plot(res, land, fig_pdf, summary)

    print("\nDone.")


if __name__ == "__main__":
    main()
