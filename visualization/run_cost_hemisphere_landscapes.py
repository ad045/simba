"""
Sanity-check landscapes: total wiring cost and interhemispheric fraction
=======================================================================

Two descriptive landscapes over the full 50 x 50 (eta, gamma) grid, meant as a
"do I read the morphospace correctly?" check - neither is a distance measure.

    total wiring cost      sum of Euclidean lengths of all edges of a network,
                           0.5 * sum(A * D)
    interhemispheric frac  share of edges whose two endpoints sit in opposite
                           hemispheres (hemisphere = sign of the MNI x
                           coordinate in Schaefer_100_MNI_coords.txt)

Both are averaged over the replicates of each grid cell. The empirical
consensus value is drawn as a contour/marker for reference.

Usage
-----
    conda activate ma_thesis
    python visualization/run_cost_hemisphere_landscapes.py
    python visualization/run_cost_hemisphere_landscapes.py --n-reps 3   # faster
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments_config import (
    MORPHO_DIR, MORPHO_EXP, CONSENSUS_PATH, DIST_MATRIX_PATH, RAW_COORDS_PATH,
    SELECTED_MEASURES,
)
from experiment_structural_gradient.run_structural_gradient import load_net

OUT_DIR = ROOT_DIR / "output" / "cost_hemisphere_landscapes"
OUT_CSV = OUT_DIR / "cost_hemisphere_landscape.csv"
OUT_PDF = OUT_DIR / "fig_cost_hemisphere_landscapes.pdf"


def hemisphere_mask(n_nodes: int) -> np.ndarray:
    """True where an (i, j) pair crosses hemispheres. Left = MNI x < 0."""
    x = np.loadtxt(RAW_COORDS_PATH)[:, 0]
    assert x.size == n_nodes, f"coords {x.size} != nodes {n_nodes}"
    left = x < 0
    return left[:, None] != left[None, :]


def net_stats(A: np.ndarray, D: np.ndarray, cross: np.ndarray) -> dict:
    A = np.asarray(A, dtype=float)
    if A.ndim == 3:
        A = A[0]
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    n_edges = 0.5 * A.sum()
    return {
        "total_cost": 0.5 * float((A * D).sum()),
        "mean_edge_length": float((A * D).sum() / A.sum()) if n_edges else np.nan,
        "frac_inter": float(0.5 * (A * cross).sum() / n_edges) if n_edges else np.nan,
        "n_edges": float(n_edges),
    }


def compute(n_reps: int) -> pd.DataFrame:
    from tqdm import tqdm

    D = np.load(DIST_MATRIX_PATH)
    consensus = np.load(CONSENSUS_PATH)
    cross = hemisphere_mask(D.shape[0])

    ref = net_stats(consensus, D, cross)
    print(f"  empirical consensus: cost={ref['total_cost']:.1f}  "
          f"frac_inter={ref['frac_inter']:.3f}  edges={ref['n_edges']:.0f}")

    pool = pd.read_csv(MORPHO_DIR / f"summary_indiv_{SELECTED_MEASURES[0]}_for_exp_{MORPHO_EXP}.csv")
    pool = pool[["filename", "eta", "gamma", "id"]]
    pool = pool[pool["id"] < n_reps]

    rows = []
    for (eta, gamma), grp in tqdm(pool.groupby(["eta", "gamma"]), desc="grid"):
        per = [net_stats(load_net(f), D, cross) for f in grp["filename"]]
        row = {"eta": float(eta), "gamma": float(gamma), "n_reps": len(per)}
        row.update({k: float(np.nanmean([p[k] for p in per])) for k in per[0]})
        rows.append(row)

    land = pd.DataFrame(rows)
    land.attrs["ref"] = ref
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    land.to_csv(OUT_CSV, index=False)
    pd.Series(ref).to_csv(OUT_DIR / "consensus_reference.csv")
    print(f"  saved {len(land)} grid points -> {OUT_CSV}")
    return land


# --- palette, taken from the figure-1 connectome legend ------------------
SHORT_COLOR = "#232324"      # black
LONG_COLOR  = "#FFFCF2" # bonewhite  "#FF961F"      # orange
INTRA_COLOR = "#232324"      # black
INTER_COLOR = "#FFFCF2" # bonewhite  "#3FA5C4"      # blue
INK = "#232323"
REF_COLOR = "#FF961F"     # orange, marks the empirical consensus
ETA0_COLOR = "white"      # marks eta = 0


def plot(land: pd.DataFrame, ref: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import uniform_filter

    try:
        from vizman import viz
        viz.set_visual_style()
        figsize = viz.cm_to_inch((12,7)) # 13, 6.0))
    except Exception:
        figsize = (13 / 2.54, 6.0 / 2.54)

    etas = np.sort(land["eta"].unique())
    gammas = np.sort(land["gamma"].unique())

    def grid_of(col):
        return (land.pivot(index="gamma", columns="eta", values=col)
                    .reindex(index=gammas, columns=etas).to_numpy())

    # show_ref: the empirical consensus is only comparable for the fraction -
    # the generated networks have 594 edges against the consensus' 495, so a
    # summed cost is not on the same scale.
    panels = [
        ("total_cost", "Total wiring cost",
         LinearSegmentedColormap.from_list("cost", [SHORT_COLOR, LONG_COLOR]),
         ("Short", "Long"), False),
        ("frac_inter", "Interhemispheric fraction",
         LinearSegmentedColormap.from_list("hemi", [INTRA_COLOR, INTER_COLOR]),
         ("Intrahemispheric", "Interhemispheric"), True),
    ]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.045], hspace=0.55, wspace=0.10)

    for k, (col, title, cmap, ends, show_ref) in enumerate(panels):
        G = grid_of(col)
        ax = fig.add_subplot(gs[0, k])
        ax.text(-0.06, 1.10, "AB"[k], transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold", color=INK)
        ax.imshow(G, origin="lower", cmap=cmap, aspect="auto",
                  extent=[etas[0], etas[-1], gammas[0], gammas[-1]])
        # iso-line where the grid attains the empirical consensus (drawn on a
        # 3x3-smoothed copy, so replicate noise does not shatter it)
        if show_ref and G.min() <= ref[col] <= G.max():
            ax.contour(etas, gammas, uniform_filter(G, 3), levels=[ref[col]],
                       colors=REF_COLOR, linewidths=0.9, linestyles="--")
        # eta = 0: boundary beyond which long connections are rewarded
        if etas[0] <= 0.0 <= etas[-1]:
            ax.axvline(0.0, color=ETA0_COLOR, lw=0.9)
        ax.set_title(title)
        # one shared axis across the two panels: the range ends carry the ticks
        ax.set_xticks([etas[0], etas[-1]]) # [etas[0]] if k == 0 else [etas[-1]])
        # ax.set_xticklabels(etas) # ["-8"] if k == 0 else ["3"])
        ax.set_yticks([gammas[-1], gammas[0]] if k == 0 else [])
        if k == 0:
            ax.set_yticklabels(["1.0", "-0.1"])
            ax.set_ylabel(r"$\gamma$")
        # else:
            # eta sits at the shared edge of the two panels, as in figure 1
        ax.set_xlabel(r"$\eta$", labelpad=1)
            # ax.xaxis.set_label_coords(-0.06, -0.13)
        for side in ax.spines.values():
            side.set_color(INK)
            side.set_linewidth(1.0)

        # --- colour bar: end labels only, plus the empirical consensus -------
        cax = fig.add_subplot(gs[1, k])
        sm = mpl.cm.ScalarMappable(
            cmap=cmap, norm=mpl.colors.Normalize(vmin=G.min(), vmax=G.max()))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                          ticks=[G.min(), G.max()])
        cb.ax.set_xticklabels(list(ends), fontsize=6.5)
        cb.ax.tick_params(length=0, pad=1.5)
        cb.outline.set_edgecolor(INK)
        cb.outline.set_linewidth(0.8)
        for t, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
            t.set_horizontalalignment(ha)
        if show_ref and G.min() <= ref[col] <= G.max():
            cax.axvline(ref[col], color=REF_COLOR, lw=1.2)

    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"  saved figure -> {OUT_PDF}")


def selftest() -> None:
    """Triangle with one cross-hemisphere edge, unit distances."""
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    D = np.ones((3, 3)) * 2.0
    cross = np.array([[False, False, True],
                      [False, False, True],
                      [True, True, False]])
    s = net_stats(A, D, cross)
    assert s["n_edges"] == 3, s
    assert s["total_cost"] == 6.0, s          # 3 edges x length 2
    assert abs(s["frac_inter"] - 2 / 3) < 1e-12, s
    assert abs(s["mean_edge_length"] - 2.0) < 1e-12, s
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=10)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.plot_only and OUT_CSV.exists():
        land = pd.read_csv(OUT_CSV)
        ref = pd.read_csv(OUT_DIR / "consensus_reference.csv",
                          index_col=0).iloc[:, 0].to_dict()
    else:
        land = compute(args.n_reps)
        ref = land.attrs["ref"]

    plot(land, ref)

    for col in ("total_cost", "frac_inter"):
        print(f"  {col}: grid {land[col].min():.4g} - {land[col].max():.4g}, "
              f"empirical {ref[col]:.4g}")


if __name__ == "__main__":
    main()
