"""
Hub topography: can any GNM place hubs where the brain places them? (S6c)
=========================================================================

Andrea asked for a plausibility criterion beyond degree and connection-length
*distributions* - specifically, agreement with the principal structural gradient.
Kayson narrowed it to the fairest possible version of the same question: skip
the embedding entirely and correlate the *vector of nodal degrees* of the
empirical connectomes against the vector of nodal degrees of the synthetic ones.
Degree is what "hub" means; a node-wise correlation is what "in the right place"
means. Design and motivation: `X_experiment_hub_topography.md` (paper repo).

This supersedes `experiment_topographic_plausibility/` (the five-map battery) as
the headline topographic criterion. That version scored a mean correlation over
five maps, two of which (the diffusion-map gradient and the mean connection
length) are geometry-dominated - which, as this script shows, is exactly the
artefact to control for.

What is computed
----------------
For every one of the 2,500 (eta, gamma) cells, the model's predicted degree map
is the degree vector averaged over that cell's replicates (the expected degree
map; averaging maps, not correlations - a single realisation is noisy). Four
quantities per cell:

    r_deg    corr(model degree map, consensus degree map)
             -> are the hubs in the right places?
    r_cent   corr(model degree map, spatial centrality = -mean row of D)
             -> is hubness dictated by geometry?
    moran    Moran's I of the degree map, inverse-distance weights
             -> is the map spatially clumped?   [Bazinet et al. 2026]
    r_sa     corr(model degree map, Sydnor sensorimotor-association axis)
             -> does the model impose an S-A gradient on degree?

The same four are computed for the 100 individual empirical connectomes, which
supplies the two reference levels the earlier criterion lacked: a noise ceiling
(how well one real brain predicts the consensus degree map) and an empirical
baseline for geometry-dependence and spatial clumping. Without them a low
correlation cannot be told apart from a noisy map.

Per measure: r_deg at its best-fitting cell, its percentile in the grid, and the
Spearman correlation between its whole distance landscape (oriented so higher =
better fit) and the r_deg landscape.

Stages (all idempotent)
-----------------------
    prep      : fetch + parcellate the Sydnor S-A axis to Schaefer-100 (cached)
    landscape : per-cell degree maps and the four quantities -> CSV + npz
    metrics   : empirical references + per-measure scores -> CSV + meta JSON
    plot      : four separate figures -> one PDF each
    all       : the four above (default)

Usage
-----
    conda activate ma_thesis
    python run_hub_topography.py --stage all
    python run_hub_topography.py --smoke        # 1 replicate, 3 measures
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
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments_config import (
    MORPHO_DATASET, MORPHO_EXP, MORPHO_DIR,
    CONSENSUS_PATH, INDIVIDUALS_PATH, DIST_MATRIX_PATH, RAW_COORDS_PATH,
    SELECTED_MEASURES, IS_SIMILARITY, METHOD_NAMES, METRIC_COLORS,
    ROOT_DIR as CFG_ROOT,
)
from experiment_structural_gradient.run_structural_gradient import best_fit_rows, load_net

OUT_DIR   = CFG_ROOT / "output" / "hub_topography"
# the MST seed every network in the morphospace run was grown from
# (config_gnm_run_with_mst_seeds_lexis_dataset.yaml -> seed_adjacency_matrices)
SEED_PATH = Path.home() / ("Documents/01_projects/14_4D_lab/14_4D_lab_code/"
                           "data/preprocessed/seeds/mst_schaeffer.npy")
MAPS_DIR  = CFG_ROOT / "data" / "preprocessed" / MORPHO_DATASET / "03_brain_maps"
SA_CACHE  = MAPS_DIR / "sa_axis_schaefer100.npy"

# Sydnor et al. (2021) archetypal S-A axis, vertex-wise on fsLR 32k, plus the
# medial-wall masks needed to expand 59,412 grayordinates back to 64,984 vertices
_SA_BASE = "https://raw.githubusercontent.com/PennLINC/S-A_ArchetypalAxis/main/FSLRVertex/"
_SA_FILES = {
    "sa": "Sensorimotor_Association_Axis_AverageRanks.csv",
    "mw_l": "medialwall.mask.leftcortex.csv",
    "mw_r": "medialwall.mask.rightcortex.csv",
}
# the seven Yeo networks, in the order the S-A axis must rank them
_SA_SANITY = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Cont", "Limbic", "Default"]


# ===========================================================================
# Stage: prep - the Sydnor S-A axis on our 100 nodes
# ===========================================================================

def build_sa_axis(cache: Path = SA_CACHE, force: bool = False) -> np.ndarray:
    """Sydnor's sensorimotor-association axis, parcellated to Schaefer-100.

    Kayson's point: Margulies' gradient is functional, so it is the wrong
    reference for a structural comparison; the S-A axis is the structural one.
    Vertex ranks are averaged within each Schaefer-100 (7 networks) parcel of the
    fsLR dlabel shipped with netneurotools - the same parcellation and node order
    as our connectomes.
    """
    if cache.exists() and not force:
        return np.load(cache)

    import nibabel as nib
    from netneurotools.datasets import fetch_schaefer2018

    tmp = cache.parent / "_sa_download"
    tmp.mkdir(parents=True, exist_ok=True)
    for key, fname in _SA_FILES.items():
        dest = tmp / fname
        if not dest.exists():
            print(f"  downloading {fname} ...")
            urllib.request.urlretrieve(_SA_BASE + fname, dest)

    # the two masks are written without a header and mark non-medial-wall vertices
    mw = np.concatenate([
        pd.read_csv(tmp / _SA_FILES["mw_l"], header=None).iloc[:, 0].to_numpy().astype(bool),
        pd.read_csv(tmp / _SA_FILES["mw_r"], header=None).iloc[:, 0].to_numpy().astype(bool),
    ])
    sa_grayord = pd.read_csv(tmp / _SA_FILES["sa"])["finalrank.wholebrain"].to_numpy(dtype=float)
    vertex = np.full(mw.size, np.nan)
    vertex[mw] = sa_grayord

    dlabel = fetch_schaefer2018("fslr32k")["100Parcels7Networks"]
    labels = np.asarray(nib.load(str(dlabel)).get_fdata()).squeeze().astype(int)
    sa = np.array([np.nanmean(vertex[labels == i]) for i in range(1, 101)])

    if np.isnan(sa).any():
        raise RuntimeError("S-A axis has empty parcels - parcellation mismatch")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, sa)
    print(f"  Saved S-A axis -> {cache}")
    return sa


def sa_sanity(sa: np.ndarray) -> pd.Series:
    """Mean S-A rank per Yeo network; must increase along _SA_SANITY."""
    names = pd.Series(_parcel_networks())
    means = pd.Series(sa).groupby(names.values).mean().reindex(_SA_SANITY)
    if not means.is_monotonic_increasing:
        raise RuntimeError(f"S-A axis not ordered as expected:\n{means}")
    return means


def _parcel_networks() -> List[str]:
    """Yeo-network name of each of the 100 nodes, from the Schaefer parcel names."""
    from netneurotools.datasets import fetch_schaefer2018
    import nibabel as nib
    dlabel = fetch_schaefer2018("fslr32k")["100Parcels7Networks"]
    axis = nib.load(str(dlabel)).header.get_axis(0)
    table = axis.label[0]
    return [table[i][0].split("_")[2] for i in range(1, 101)]


# ===========================================================================
# Node-level quantities
# ===========================================================================

def binarize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    while A.ndim > 2:
        A = A[0]
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def degree_map(A: np.ndarray) -> np.ndarray:
    return binarize(A).sum(axis=1)


def spatial_centrality(D: np.ndarray) -> np.ndarray:
    """How central a node sits in the point cloud; higher = more central.

    Negated mean Euclidean distance to every other node. This is the only
    node-level information the generative rule's cost term carries, so it is the
    ceiling on how much of a degree map geometry alone can explain.
    """
    return -D.mean(axis=1)


def load_seed() -> np.ndarray:
    """The binarised MST seed shared by every generated network."""
    return binarize(np.load(SEED_PATH))


def moran_weights(D: np.ndarray) -> np.ndarray:
    """Inverse-Euclidean-distance weights, zero diagonal (Bazinet et al. 2026)."""
    with np.errstate(divide="ignore"):
        W = np.where(D > 0, 1.0 / np.where(D > 0, D, 1.0), 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def morans_i(x: np.ndarray, W: np.ndarray) -> float:
    """Moran's I of a node-level map under weights W."""
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    z = (x - np.mean(x)) / sd
    # Accelerate's BLAS raises spurious FP flags on this matmul; the inputs are
    # checked finite above, so the flags are suppressed rather than acted on.
    with np.errstate(all="ignore"):
        return float(z @ W @ z / W.sum())


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import pearsonr
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 5 or np.std(a[finite]) == 0 or np.std(b[finite]) == 0:
        return float("nan")
    return float(pearsonr(a[finite], b[finite])[0])


def score_map(deg: np.ndarray, ref: Dict[str, np.ndarray], W: np.ndarray) -> Dict[str, float]:
    """The four quantities for one degree map."""
    return {
        "r_deg":  _corr(deg, ref["degree"]),
        "r_cent": _corr(deg, ref["centrality"]),
        "moran":  morans_i(deg, W),
        "r_sa":   _corr(deg, ref["sa"]),
    }


# ===========================================================================
# Stage: landscape
# ===========================================================================

def compute_landscape(n_reps: int, land_csv: Path, npz_path: Path) -> None:
    from tqdm import tqdm

    D = np.load(DIST_MATRIX_PATH)
    W = moran_weights(D)
    ref = {
        "degree": degree_map(np.load(CONSENSUS_PATH)),
        "centrality": spatial_centrality(D),
        "sa": build_sa_axis(),
    }
    print(f"  Reference maps ready (N={ref['degree'].size} nodes).")

    # filenames and the grid are identical across measures -> one summary is the pool
    pool = pd.read_csv(MORPHO_DIR / f"summary_indiv_{SELECTED_MEASURES[0]}_for_exp_{MORPHO_EXP}.csv")
    pool = pool.loc[pool["id"] < n_reps, ["filename", "eta", "gamma", "id"]]

    # the seed is identical in every network, so whatever geometry it carries is
    # a constant of the run; scoring the added edges separately isolates the part
    # of the geometry-dependence the wiring rule itself produces
    seed = load_seed()
    keep = 1.0 - seed

    rows, deg_maps = [], []
    for (eta, gamma), grp in tqdm(pool.groupby(["eta", "gamma"]), desc="hub topography"):
        nets = [binarize(load_net(f)) for f in grp["filename"]]
        degs = np.array([A.sum(axis=1) for A in nets])
        mean_deg = degs.mean(axis=0)
        # degree map of the edges the model added on top of the seed
        added_deg = np.mean([(A * keep).sum(axis=1) for A in nets], axis=0)
        row = {"eta": float(eta), "gamma": float(gamma), "n_reps": int(len(grp))}
        row.update(score_map(mean_deg, ref, W))
        row["r_cent_noseed"] = _corr(added_deg, ref["centrality"])
        row["r_deg_noseed"] = _corr(added_deg, ref["degree"])
        row["n_seed_edges_kept"] = float(np.mean([(A * seed).sum() / 2 for A in nets]))
        # the same score computed per replicate and then averaged, so the cost of
        # scoring single realisations instead of the model's expectation is visible
        row["r_deg_per_rep"] = float(np.nanmean([_corr(d, ref["degree"]) for d in degs]))
        rows.append(row)
        deg_maps.append(mean_deg)

    land = pd.DataFrame(rows)
    land.to_csv(land_csv, index=False)
    np.savez(npz_path,
             degree_maps=np.array(deg_maps),
             eta=land["eta"].to_numpy(), gamma=land["gamma"].to_numpy(),
             **{f"ref__{k}": v for k, v in ref.items()})
    print(f"  Saved landscape ({len(land)} cells) -> {land_csv.name}")
    print(f"  Saved degree maps            -> {npz_path.name}")


# ===========================================================================
# Stage: metrics
# ===========================================================================

def empirical_reference(n_reps_note: int) -> Dict[str, object]:
    """The four quantities for the consensus and for each individual connectome."""
    D = np.load(DIST_MATRIX_PATH)
    W = moran_weights(D)
    ref = {
        "degree": degree_map(np.load(CONSENSUS_PATH)),
        "centrality": spatial_centrality(D),
        "sa": build_sa_axis(),
    }
    subjects = np.load(INDIVIDUALS_PATH)
    per_subject = pd.DataFrame([score_map(degree_map(subjects[i]), ref, W)
                                for i in range(subjects.shape[0])])

    # split-half of the subject pool: how reproducible the degree map is at all
    half = subjects.shape[0] // 2
    h1 = np.mean([degree_map(subjects[i]) for i in range(half)], axis=0)
    h2 = np.mean([degree_map(subjects[i]) for i in range(half, subjects.shape[0])], axis=0)

    cons = score_map(ref["degree"], ref, W)
    seed_deg = load_seed().sum(axis=1)
    return {
        "per_subject": per_subject,
        "consensus": cons,
        "seed": score_map(seed_deg, ref, W),
        "split_half_r": _corr(h1, h2),
        "n_subjects": int(subjects.shape[0]),
        "sa_by_network": sa_sanity(ref["sa"]).round(0).to_dict(),
    }


def cell_means(measure: str) -> pd.Series:
    """Per-(eta, gamma) mean of a measure's cached distance to the consensus."""
    df = pd.read_csv(MORPHO_DIR / f"summary_indiv_{measure}_for_exp_{MORPHO_EXP}.csv")
    col = [c for c in df.columns
           if c not in {"network_index", "filename", "eta", "gamma", "id"}][0]
    return df.groupby(["eta", "gamma"])[col].mean()


N_TOP = 100          # best-fitting parameter combinations per measure (as in Fig 2A)
SCORES = ["r_deg", "r_cent", "moran", "r_sa"]


def top_cells(measure: str, land: pd.DataFrame, n: int = N_TOP) -> pd.DataFrame:
    """The landscape rows of a measure's `n` best-fitting grid cells.

    A single argmin is one draw from a noisy landscape; the paper already
    characterises a measure by its 100 best-fitting parameter combinations, and
    the spread across them is what the boxplots show.
    """
    fit = cell_means(measure)
    if measure in IS_SIMILARITY:
        fit = -fit                          # orient so that lower always means closer
    best = fit.nsmallest(n).index           # MultiIndex of (eta, gamma)
    return land.set_index(["eta", "gamma"]).loc[best].reset_index()


def compute_metrics(land: pd.DataFrame, measures: List[str]) -> pd.DataFrame:
    from scipy.stats import spearmanr

    key = land.set_index(["eta", "gamma"])
    r_deg = key["r_deg"].to_numpy(dtype=float)

    rows = []
    for m in measures:
        eta_star, gamma_star, _ = best_fit_rows(m)
        cell = key.loc[(eta_star, gamma_star)]

        fit = cell_means(m)
        if m not in IS_SIMILARITY:
            fit = -fit                      # orient so that higher = better fit
        joint = pd.concat([fit.rename("fit"), key["r_deg"]], axis=1).dropna()

        row = {
            "measure": m, "name": METHOD_NAMES.get(m, m),
            "eta": eta_star, "gamma": gamma_star,
            "r_deg": float(cell["r_deg"]),
            "r_cent": float(cell["r_cent"]),
            "moran": float(cell["moran"]),
            "r_sa": float(cell["r_sa"]),
            "grid_percentile": float((r_deg < float(cell["r_deg"])).mean() * 100.0),
            "landscape_rho": float(spearmanr(joint["fit"], joint["r_deg"])[0]),
        }
        # the same scores over the measure's 100 best cells, for the spread
        top = top_cells(m, land)
        for s in SCORES:
            q1, med, q3 = np.nanpercentile(top[s].to_numpy(dtype=float), [25, 50, 75])
            row[f"{s}_top_median"] = float(med)
            row[f"{s}_top_q1"] = float(q1)
            row[f"{s}_top_q3"] = float(q3)
        rows.append(row)

    return (pd.DataFrame(rows).sort_values("r_deg_top_median", ascending=False)
            .reset_index(drop=True))


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


def _save(fig, out_pdf: Path) -> None:
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"  Saved figure -> {out_pdf}")


def _despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# the manuscript palette
DARK      = "#232324"
ORANGE    = "#FE961F"
LIGHTBLUE = "#3EA5C4"
BONEWHITE = "#FFFCF2"
LECKER_RED = (0.84, 0.19, 0.15) 
FAINT_BLUE = (0.30, 0.45, 0.69)  

GRAY = (0.5, 0.5, 0.5)
# REAL_COLOR = ORANGE       # empirical data
# ART_COLOR = LIGHTBLUE     # generated networks
REAL_COLOR = LECKER_RED      # the red used for empirical data elsewhere
ART_COLOR = FAINT_BLUE     # the blue used for generated networks

def _cmap(name: str, colors: List[str], n: int = 256):
    """Two-colour ramp interpolated in linear light rather than in sRGB.

    Blending sRGB values directly darkens and muddies the midtones, which is
    very visible on the dark-grey-to-orange ramp; undoing the gamma before
    mixing keeps the middle of the ramp on the intended hue.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    ends = np.array([to_rgb(c) for c in colors]) ** 2.2
    t = np.linspace(0, 1, n)[:, None]
    lo, hi = ends[0][None, :], ends[-1][None, :]
    return LinearSegmentedColormap.from_list(name, (lo + t * (hi - lo)) ** (1 / 2.2))


def plot_maps(res: pd.DataFrame, npz_path: Path, out_pdf: Path) -> None:
    """The three node-level maps side by side, on the cortex."""
    import matplotlib.pyplot as plt

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((18, 6.5)) if have_viz else (18 / 2.54, 6.5 / 2.54))

    dat = np.load(npz_path)
    coords = np.loadtxt(RAW_COORDS_PATH)[:, :3]

    # the cell the measures actually select: median of their best fits
    eta_sel, gamma_sel = float(np.median(res["eta"])), float(np.median(res["gamma"]))
    sel = int(np.argmin((dat["eta"] - eta_sel) ** 2 + (dat["gamma"] - gamma_sel) ** 2))

    cent = dat["ref__centrality"]
    panels = [
        (dat["ref__degree"], "empirical consensus\nnodal degree"),
        (dat["degree_maps"][sel], f"generated networks at $\\eta$={dat['eta'][sel]:+.2f}, "
                                  f"$\\gamma$={dat['gamma'][sel]:.2f}\nnodal degree"),
        (cent, "spatial centrality\n(geometry alone)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (vals, title) in zip(axes, panels):
        # axial view: the two hemispheres separate, so all 100 nodes stay visible.
        # Each map is min-max scaled - the panel compares patterns, not values.
        v = (vals - vals.min()) / np.ptp(vals)
        ax.scatter(coords[:, 0], coords[:, 1], c=v, s=10 + 60 * v,
                   cmap=_cmap("degree", [DARK, ORANGE]),
                   edgecolor=BONEWHITE, linewidths=0.25, vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, loc="center")
        # the geometry-dependence of the map shown, including the empirical one -
        # without it the third panel is only a picture and not a comparison
        r = _corr(vals, cent)
        if np.isfinite(r) and not np.allclose(vals, cent):
            ax.text(0.5, -0.02, f"$r$ to spatial centrality = {r:+.3f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7)
    _save(fig, out_pdf)


def plot_landscape(res: pd.DataFrame, land: pd.DataFrame, out_pdf: Path) -> None:
    """Hub-placement agreement across the (eta, gamma) grid."""
    import matplotlib.pyplot as plt

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((12, 12)) if have_viz else (12 / 2.54, 12 / 2.54))

    etas = np.sort(land["eta"].unique())
    gammas = np.sort(land["gamma"].unique())
    grid = (land.pivot(index="gamma", columns="eta", values="r_deg")
                .reindex(index=gammas, columns=etas).to_numpy())

    fig, ax = plt.subplots(figsize=figsize)
    # darker = better agreement, as in the other landscape figures
    im = ax.pcolormesh(etas, gammas, grid, cmap=_cmap("agreement", [BONEWHITE, DARK]),
                       shading="auto")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("r(generated degree map,\nconsensus degree map)")
    ax.axvline(0.0, color=LIGHTBLUE, lw=1.0, ls=":")
    # the best fits sit almost on top of each other, so they are keyed by a
    # legend rather than annotated in place
    for _, r in res.iterrows():
        ax.scatter([r["eta"]], [r["gamma"]], s=38, label=r["name"],
                   color=METRIC_COLORS.get(r["measure"], GRAY),
                   edgecolor=BONEWHITE, linewidths=0.8, zorder=5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, handletextpad=0.3, columnspacing=1.0)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_box_aspect(1)
    _save(fig, out_pdf)


def plot_geometry(res: pd.DataFrame, land: pd.DataFrame, summary: dict,
                  out_pdf: Path) -> None:
    """Geometry-dependence against agreement, for grid cells and for brains."""
    import matplotlib.pyplot as plt

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((13, 9)) if have_viz else (13 / 2.54, 9 / 2.54))
    subj = pd.DataFrame(summary["per_subject"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color=GRAY, lw=0.6, ls=":")
    ax.axvline(0, color=GRAY, lw=0.6, ls=":")
    ax.scatter(land["r_cent"], land["r_deg"], s=5, alpha=0.45, linewidths=0,
               color=ART_COLOR, label=f"generated networks ({len(land):,} grid cells)")
    ax.scatter(subj["r_cent"], subj["r_deg"], s=14, color=REAL_COLOR,
               edgecolor="white", linewidths=0.3, zorder=4,
               label=f"individual connectomes ({summary['n_subjects']})")
    ax.scatter(res["r_cent"], res["r_deg"], s=34, facecolor="none",
               edgecolor="black", linewidths=0.9, zorder=5,
               label="the measures' best fits")
    ax.set_xlabel("r(degree map, spatial centrality)")
    ax.set_ylabel("r(degree map, consensus degree map)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=False, handletextpad=0.3, columnspacing=1.0)
    _despine(ax)
    _save(fig, out_pdf)


def plot_distributions(land: pd.DataFrame, summary: dict, out_pdf: Path) -> None:
    """Generated and empirical distributions of the three quantities."""
    import matplotlib.pyplot as plt

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((18, 6.5)) if have_viz else (18 / 2.54, 6.5 / 2.54))
    subj = pd.DataFrame(summary["per_subject"])

    fields = [
        ("r_deg", "r to the consensus\ndegree map"),
        ("r_cent", "r to spatial\ncentrality"),
        ("moran", "Moran's $I$ of the\ndegree map"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (field, label) in zip(axes, fields):
        parts = [land[field].dropna().to_numpy(), subj[field].to_numpy()]
        vp = ax.violinplot(parts, positions=[0, 1], widths=0.75, showextrema=False)
        for body, col in zip(vp["bodies"], [ART_COLOR, REAL_COLOR]):
            body.set_facecolor(col)
            body.set_alpha(0.55)
            body.set_edgecolor("black")
            body.set_linewidth(0.4)
        for i, part in enumerate(parts):
            ax.hlines(np.median(part), i - 0.26, i + 0.26, color="black", lw=1.1,
                      zorder=4)
        ax.axhline(0, color=GRAY, lw=0.6, ls=":")
        ax.set_xticks([0, 1])
        # one value per grid cell (the replicate-averaged map), not per network
        ax.set_xticklabels([f"{len(land):,}\nparameter\ncombinations",
                            f"{summary['n_subjects']}\nindividual\nconnectomes"])
        ax.set_ylabel(label)
        _despine(ax)
    _save(fig, out_pdf)


def plot_measures(res: pd.DataFrame, land: pd.DataFrame, summary: dict,
                  out_pdf: Path) -> None:
    """The eight measures side by side on every quantity of the criterion.

    Each measure is summarised by its `N_TOP` best-fitting grid cells rather than
    by a single argmin, so every box shows how much the answer moves within the
    region of parameter space the measure calls close to the consensus. The last
    panel is a single number per measure (a whole-landscape statistic) and stays
    a bar.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((18, 8)) if have_viz else (18 / 2.54, 8 / 2.54))
    subj = pd.DataFrame(summary["per_subject"])

    res = res.sort_values("r_deg_top_median", ascending=False).reset_index(drop=True)
    y = np.arange(len(res))[::-1]
    colors = [METRIC_COLORS.get(m, GRAY) for m in res["measure"]]
    tops = {m: top_cells(m, land) for m in res["measure"]}

    # field, axis label, empirical band (or None), grid maximum (or None)
    panels = [
        ("r_deg", "r to the consensus\ndegree map",
         (subj["r_deg"].min(), subj["r_deg"].max()), land["r_deg"].max()),
        ("r_cent", "r to spatial\ncentrality",
         (subj["r_cent"].min(), subj["r_cent"].max()), None),
        ("moran", "Moran's $I$ of the\ndegree map",
         (subj["moran"].min(), subj["moran"].max()), None),
        ("landscape_rho", r"$\rho$(distance landscape," "\n" r"agreement landscape)",
         None, None),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True)
    for ax, (field, label, band, grid_max) in zip(axes, panels):
        if band is not None:
            ax.axvspan(band[0], band[1], color=REAL_COLOR, alpha=0.18, zorder=0)
        if grid_max is not None:
            ax.axvline(grid_max, color="black", lw=0.9, ls="-", zorder=1)
        ax.axvline(0, color=GRAY, lw=0.6, ls=":", zorder=1)

        if field in SCORES:
            data = [tops[m][field].dropna().to_numpy() for m in res["measure"]]
            bp = ax.boxplot(data, positions=y, vert=False, widths=0.62,
                            patch_artist=True, showfliers=False, zorder=2)
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.4)
            for part in ("whiskers", "caps", "medians"):
                for line in bp[part]:
                    line.set_color("black")
                    line.set_linewidth(0.7)
        else:
            ax.barh(y, res[field], color=colors, edgecolor="black", linewidth=0.4,
                    height=0.62, zorder=2)

        ax.set_xlabel(label)
        _despine(ax)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(res["name"])
    axes[0].set_ylim(y.min() - 0.6, y.max() + 0.6)
    fig.legend(handles=[
        Patch(facecolor=GRAY, edgecolor="black", linewidth=0.4,
              label=f"spread over each measure's {N_TOP} best-fitting parameter combinations"),
        Patch(facecolor=REAL_COLOR, alpha=0.18,
              label=f"range across the {summary['n_subjects']} individual connectomes"),
        Line2D([0], [0], color="black", lw=0.9,
               label="best value reached by any grid cell"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    _save(fig, out_pdf)


def plot_seed_control(res: pd.DataFrame, land: pd.DataFrame, npz_path: Path,
                      summary: dict, out_pdf: Path) -> None:
    """Is the geometry-dependence inherited from the shared MST seed?

    Left: nodal degree against spatial centrality, node by node, for the
    empirical consensus and for the generated networks - the raw version of the
    correlation the other panels only summarise. Right: the same correlation
    across the whole grid, computed on the full networks and on the edges the
    model added on top of the seed.
    """
    import matplotlib.pyplot as plt

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((18, 7)) if have_viz else (18 / 2.54, 7 / 2.54))

    dat = np.load(npz_path)
    cent, cons = dat["ref__centrality"], dat["ref__degree"]
    eta_sel, gamma_sel = float(np.median(res["eta"])), float(np.median(res["gamma"]))
    sel = int(np.argmin((dat["eta"] - eta_sel) ** 2 + (dat["gamma"] - gamma_sel) ** 2))
    gen = dat["degree_maps"][sel]
    seed_deg = load_seed().sum(axis=1)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize)

    for vals, col, lab in [(cons, REAL_COLOR, "empirical consensus"),
                           (gen, ART_COLOR, f"generated ($\\eta$={dat['eta'][sel]:+.2f}, "
                                            f"$\\gamma$={dat['gamma'][sel]:.2f})")]:
        ax.scatter(cent, vals, s=14, color=col, edgecolor="white", linewidths=0.3,
                   label=f"{lab}: $r$ = {_corr(vals, cent):+.3f}")
        b, a = np.polyfit(cent, vals, 1)
        xs = np.array([cent.min(), cent.max()])
        ax.plot(xs, a + b * xs, color=col, lw=1.0)
    ax.set_xlabel("spatial centrality (-mean distance to all nodes)")
    ax.set_ylabel("nodal degree")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    _despine(ax)

    subj = pd.DataFrame(summary["per_subject"])
    parts = [(land["r_cent"].dropna().to_numpy(), ART_COLOR,
              f"{len(land):,} parameter\ncombinations\n(full networks)")]
    if "r_cent_noseed" in land:
        parts.append((land["r_cent_noseed"].dropna().to_numpy(), ART_COLOR,
                      "same networks,\nseed edges removed"))
    parts.append((subj["r_cent"].to_numpy(), REAL_COLOR,
                  f"{summary['n_subjects']} individual\nconnectomes"))
    cols = [c for _, c, _ in parts]
    labels = [l for _, _, l in parts]
    parts = [v for v, _, _ in parts]

    vp = bx.violinplot(parts, positions=range(len(parts)), widths=0.75,
                       showextrema=False)
    for body, col in zip(vp["bodies"], cols):
        body.set_facecolor(col)
        body.set_alpha(0.55)
        body.set_edgecolor("black")
        body.set_linewidth(0.4)
    for i, part in enumerate(parts):
        bx.hlines(np.median(part), i - 0.26, i + 0.26, color="black", lw=1.1, zorder=4)
    bx.axhline(0, color=GRAY, lw=0.6, ls=":")
    # the seed on its own: the geometry a network inherits before any edge is added
    bx.axhline(_corr(seed_deg, cent), color=DARK, lw=0.9, ls="--",
               label=f"MST seed alone: $r$ = {_corr(seed_deg, cent):+.3f}")
    bx.set_xticks(range(len(parts)))
    bx.set_xticklabels(labels)
    bx.set_ylabel("r(degree map, spatial centrality)")
    bx.legend(loc="lower right", frameon=False, fontsize=7)
    _despine(bx)
    _save(fig, out_pdf)


def plot_sa(res: pd.DataFrame, land: pd.DataFrame, npz_path: Path,
            summary: dict, out_pdf: Path) -> None:
    """The S-A axis panel: degree against the axis, and the r_sa distributions.

    Panel A shows the two degree maps (empirical consensus, generated network at
    the median best-fitting cell) against Sydnor's sensorimotor-association rank;
    panel B contrasts the 2,500 grid cells with the 100 individual connectomes on
    the same correlation, so the empirical null is visible as a distribution
    rather than as a single consensus number.
    """
    import matplotlib.pyplot as plt

    viz, have_viz = _viz() 
    figsize = (viz.cm_to_inch((12, 6.5)) if have_viz else (16 / 2.54, 6.5 / 2.54)) # The one figure that is by itself in the SI: hub_topography_sa_axis.pdf

    subj = pd.DataFrame(summary["per_subject"])

    z = np.load(npz_path)
    sa = z["ref__sa"]
    cons_deg = z["ref__degree"]
    eta_m, gamma_m = float(res["eta"].median()), float(res["gamma"].median())
    cell = int(np.argmin((z["eta"] - eta_m) ** 2 + (z["gamma"] - gamma_m) ** 2))
    gnm_deg = z["degree_maps"][cell]

    def _z(v):
        return (v - v.mean()) / v.std()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    series = [
        (cons_deg, REAL_COLOR, "empirical consensus"),
        (gnm_deg, ART_COLOR, rf"generated ($\eta={eta_m:.2f}$, $\gamma={gamma_m:.2f}$)"),
    ]
    for deg, col, lab in series:
        r = _corr(deg, sa)
        ax.scatter(sa, _z(deg), s=12, color=col, alpha=0.65, linewidths=0,
                   label=f"{lab}, $r = {r:+.2f}$")
        b, a = np.polyfit(sa, _z(deg), 1)
        xs = np.linspace(sa.min(), sa.max(), 2)
        ax.plot(xs, a + b * xs, color=col, lw=1.2)
    ax.set_xlabel("sensorimotor-association rank")
    ax.set_ylabel("nodal degree ($z$-scored)")
    ax.legend(frameon=False, fontsize=6, loc="lower left",
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)
    _despine(ax)

    ax = axes[1]
    parts = [land["r_sa"].dropna().to_numpy(), subj["r_sa"].to_numpy()]
    vp = ax.violinplot(parts, positions=[0, 1], widths=0.75, showextrema=False)
    for body, col in zip(vp["bodies"], [ART_COLOR, REAL_COLOR]):
        body.set_facecolor(col)
        body.set_alpha(0.55)
        body.set_edgecolor("black")
        body.set_linewidth(0.4)
    for i, part in enumerate(parts):
        ax.hlines(np.median(part), i - 0.26, i + 0.26, color="black", lw=1.1, zorder=4)
    ax.axhline(0, color=GRAY, lw=0.6, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{len(land):,} parameter\ncombinations",
                        f"{summary['n_subjects']} individual\nconnectomes"])
    ax.set_ylabel("$r$ to the S-A axis")
    _despine(ax)

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.18, 1.06, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")

    _save(fig, out_pdf)


def plot_all(res: pd.DataFrame, land: pd.DataFrame, npz_path: Path,
             summary: dict, out_dir: Path, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")

    plot_maps(res, npz_path, out_dir / f"{stem}_maps.pdf")
    plot_landscape(res, land, out_dir / f"{stem}_landscape.pdf")
    plot_geometry(res, land, summary, out_dir / f"{stem}_geometry.pdf")
    plot_distributions(land, summary, out_dir / f"{stem}_distributions.pdf")
    plot_measures(res, land, summary, out_dir / f"{stem}_measures.pdf")
    plot_seed_control(res, land, npz_path, summary, out_dir / f"{stem}_seed_control.pdf")
    plot_sa(res, land, npz_path, summary, out_dir / f"{stem}_sa.pdf")


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["prep", "landscape", "metrics", "plot", "all"],
                    default="all")
    ap.add_argument("--n-reps", type=int, default=10,
                    help="replicates averaged per grid point (max 10)")
    ap.add_argument("--measures", default=None, help="comma-separated subset")
    ap.add_argument("--smoke", action="store_true",
                    help="1 replicate, 3 measures, separate output dir")
    args = ap.parse_args()

    measures = (args.measures.split(",") if args.measures
                else (["frobenius", "delta_con", "portrait"] if args.smoke
                      else SELECTED_MEASURES))
    n_reps = 1 if args.smoke else args.n_reps

    out_dir = OUT_DIR if not args.smoke else (OUT_DIR.parent / "hub_topography_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    land_csv    = out_dir / "hub_topography_landscape.csv"
    npz_path    = out_dir / "degree_maps.npz"
    results_csv = out_dir / "hub_topography_results.csv"
    subject_csv = out_dir / "hub_topography_subjects.csv"
    meta_json   = out_dir / "hub_topography_meta.json"
    fig_stem    = "fig_hub_topography"      # one PDF per figure, suffixed below

    print(f"Output dir : {out_dir}")
    print(f"Measures   : {measures}   n_reps: {n_reps}\n")

    if args.stage in ("prep", "all"):
        print("=" * 60, "\nStage: prep\n", "=" * 60, sep="")
        sa = build_sa_axis()
        print(f"  S-A axis on {sa.size} nodes; mean rank per network:")
        print(sa_sanity(sa).round(0).to_string())

    if args.stage in ("landscape", "all"):
        print("\n" + "=" * 60, "\nStage: landscape\n", "=" * 60, sep="")
        compute_landscape(n_reps, land_csv, npz_path)

    if args.stage in ("metrics", "all"):
        print("\n" + "=" * 60, "\nStage: metrics\n", "=" * 60, sep="")
        land = pd.read_csv(land_csv)
        emp = empirical_reference(n_reps)
        emp["per_subject"].to_csv(subject_csv, index=False)
        res = compute_metrics(land, measures)
        res.to_csv(results_csv, index=False)

        subj = emp["per_subject"]
        meta = {
            "criterion": "hub topography (node-wise degree map)",
            "reference": "empirical HCP consensus (10% density)",
            "model_prediction": "degree map averaged over replicates per grid cell",
            "null": "the full 50x50 grid itself (exact, not sampled)",
            "n_reps_per_grid_point": n_reps,
            "n_subjects": emp["n_subjects"],
            "consensus": emp["consensus"],
            "seed": emp["seed"],
            "split_half_r": emp["split_half_r"],
            "sa_by_network": emp["sa_by_network"],
            "subject_r_deg_mean": float(subj["r_deg"].mean()),
            "subject_r_deg_std": float(subj["r_deg"].std()),
            "subject_r_deg_min": float(subj["r_deg"].min()),
            "subject_r_cent_mean": float(subj["r_cent"].mean()),
            "subject_r_cent_std": float(subj["r_cent"].std()),
            "subject_moran_mean": float(subj["moran"].mean()),
            "subject_moran_std": float(subj["moran"].std()),
            "subject_r_sa_mean": float(subj["r_sa"].mean()),
            "grid_max_r_deg": float(land["r_deg"].max()),
            "grid_max_r_deg_per_rep": float(land["r_deg_per_rep"].max()),
            "grid_median_r_deg": float(land["r_deg"].median()),
            "grid_max_r_cent": float(land["r_cent"].max()),
            "grid_min_r_cent": float(land["r_cent"].min()),
            "grid_max_moran": float(land["moran"].max()),
            "grid_median_r_cent": float(land["r_cent"].median()),
            "grid_median_r_cent_noseed": (float(land["r_cent_noseed"].median())
                                          if "r_cent_noseed" in land else None),
            "grid_max_r_cent_noseed": (float(land["r_cent_noseed"].max())
                                       if "r_cent_noseed" in land else None),
            "grid_max_abs_r_sa": float(land["r_sa"].abs().max()),
            "n_cells_above_worst_subject": int((land["r_deg"] > subj["r_deg"].min()).sum()),
            "morphospace_run": f"{MORPHO_DATASET}/{MORPHO_EXP}",
            "measures": measures,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        meta_json.write_text(json.dumps(meta, indent=2))
        print(f"\n  Saved results  -> {results_csv.name}")
        print(f"  Saved subjects -> {subject_csv.name}")
        print(f"  Saved meta     -> {meta_json.name}\n")
        print(res.to_string(index=False))
        print(f"\n  consensus       : {emp['consensus']}")
        print(f"  split-half r    : {emp['split_half_r']:.3f}")
        print(f"  subject r_deg   : {subj['r_deg'].mean():.3f} +- {subj['r_deg'].std():.3f} "
              f"(min {subj['r_deg'].min():.3f})")
        print(f"  grid max r_deg  : {land['r_deg'].max():.3f}   "
              f"cells beating the worst subject: {meta['n_cells_above_worst_subject']}")
        print(f"  grid r_cent     : {land['r_cent'].min():+.3f} to {land['r_cent'].max():+.3f} "
              f"(subjects {subj['r_cent'].mean():+.3f} +- {subj['r_cent'].std():.3f})")
        if "r_cent_noseed" in land:
            print(f"  seed alone      : r to centrality {emp['seed']['r_cent']:+.3f}, "
                  f"to consensus degree {emp['seed']['r_deg']:+.3f}")
            print(f"  grid r_cent     : median {land['r_cent'].median():+.3f} with seed, "
                  f"{land['r_cent_noseed'].median():+.3f} with the seed edges removed")
        print(f"  grid Moran max  : {land['moran'].max():.3f} "
              f"(subjects {subj['moran'].mean():.3f} +- {subj['moran'].std():.3f})")

    if args.stage in ("plot", "all"):
        print("\n" + "=" * 60, "\nStage: figure\n", "=" * 60, sep="")
        land = pd.read_csv(land_csv)
        res = pd.read_csv(results_csv)
        summary = json.loads(meta_json.read_text())
        summary["per_subject"] = pd.read_csv(subject_csv).to_dict("list")
        plot_all(res, land, npz_path, summary, out_dir, fig_stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
