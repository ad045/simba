"""
Structural-gradient biological-plausibility check (analysis S6)
==============================================================

A *stronger* biological-plausibility criterion for the network-distance
measures, as requested by Andrea's review. The manuscript currently justifies
"biological plausibility" only through degree- and edge-length-distribution
fidelity. Andrea asked for something more fleshed out and suggested the
**principal structural gradient topography** - "somewhat related to the
spectral measures, but not identical".

Idea
----
The principal structural gradient is the dominant axis of a diffusion-map
embedding of a connectome's node-to-node affinity structure (Margulies et al.,
2016). It summarises the macroscale organisational topography of the network as
one scalar per node. A measure is biologically plausible in this richer sense if
the GNM networks it *selects as its best fit to the empirical consensus*
reproduce the consensus's principal-gradient topography - not merely its degree
and distance distributions.

Procedure (per measure)
-----------------------
1. Best fit: from the cached morphospace distances (D_art, one CSV per measure),
   find the (eta, gamma) grid point that a measure judges closest to the
   empirical consensus (argmin distance; argmax for the one similarity measure).
2. Gradient: compute the principal structural gradient of each of that grid
   point's replicate networks, and of the empirical consensus.
3. Score: spatial correlation (Pearson + Spearman) between each best-fit
   network's principal gradient and the consensus's, across the 100 nodes,
   after resolving the arbitrary eigenvector sign. Report the mean over the grid
   point's replicates.

Null / interpretation
---------------------
The eigenvector sign is aligned to the consensus, so a value of ~0 (not -1) is
"no topographic agreement". To turn the correlation into a plausibility
*statement* we also compute the same gradient correlation for a random sample of
GNM networks drawn from across the whole (eta, gamma) plane. Each measure's
best-fit score is reported as a percentile against this "any GNM" null: a high
percentile means the region the measure prefers reproduces the consensus
gradient better than an arbitrary synthetic network does.

What is new vs. reused
----------------------
  * D_art (the per-network distances used to locate each measure's best fit) is
    ALREADY cached by the main morphospace run - we only read it.
  * The 25,000 GNM adjacency matrices are ALREADY on disk (generated_networks/).
  * The only new computation is the diffusion-map gradient (a 100x100 eigen-
    decomposition per network) and its correlation to the consensus gradient -
    cheap, so the whole thing runs in seconds.

Stages (all idempotent)
-----------------------
    gradient : compute consensus gradient + best-fit gradients + null sample,
               cache to disk (npz + CSVs)
    metrics  : aggregate -> structural_gradient_results.csv (+ meta JSON)
    plot     : Panel A ranking bar chart (with null band) + Panel B per-measure
               consensus-vs-best-fit gradient scatter -> PDF
    all      : the three above (default)

Usage
-----
    conda activate ma_thesis
    python run_structural_gradient.py --stage all
    python run_structural_gradient.py --stage gradient
    python run_structural_gradient.py --n-null 2000
    python run_structural_gradient.py --smoke        # tiny fast self-test
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --- repo root on sys.path (so `import src.*` and the shared config resolve) --
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments_config import (
    MORPHO_DATASET, MORPHO_EXP, MORPHO_DIR,
    CONSENSUS_PATH,
    SELECTED_MEASURES, IS_SIMILARITY, METHOD_NAMES, METRIC_COLORS,
    ROOT_DIR as CFG_ROOT,
)

# where the 25k GNM adjacency matrices live (keyed by the `filename` column)
GEN_NETS_DIR = MORPHO_DIR / "generated_networks"
OUT_DIR = CFG_ROOT / "output" / "structural_gradient"

# gradient hyper-parameters (Margulies-style diffusion embedding)
ROW_THRESHOLD_PCT = 90      # keep the strongest 10% of affinities per row
DIFFUSION_ALPHA = 0.5       # anisotropic (Fokker-Planck) normalisation
EPS = 1e-10                 # tiny background affinity -> guarantees connectivity
N_NULL_DEFAULT = 1500       # random GNM networks for the "any GNM" null
NULL_SEED = 20260703


# ===========================================================================
# Principal structural gradient (self-contained diffusion-map embedding)
# ===========================================================================

def structural_gradient(A: np.ndarray,
                        row_threshold_pct: float = ROW_THRESHOLD_PCT,
                        alpha: float = DIFFUSION_ALPHA,
                        eps: float = EPS) -> np.ndarray:
    """Principal structural gradient (one scalar per node) of a connectome.

    Steps (Margulies et al. 2016, Coifman-Lafon diffusion maps):
      1. cosine affinity between node connectivity profiles (rows of A),
      2. row-wise thresholding to the strongest affinities (denoising),
      3. diffusion-map embedding with anisotropic (alpha) normalisation,
      4. return the first non-trivial diffusion coordinate.

    The eigenvector sign is arbitrary and is NOT fixed here (caller aligns it).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim == 3:                       # stored GNM nets are (1, N, N)
        A = A[0]
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, A.T)                # enforce symmetry
    np.fill_diagonal(A, 0.0)
    N = A.shape[0]

    # (1) cosine similarity between connectivity profiles (rows), in [0, 1].
    #     Isolated nodes (zero row) get a unit norm so they map to a zero
    #     profile rather than producing inf/nan in the outer product.
    norm = np.sqrt((A * A).sum(axis=1))
    norm[norm == 0] = 1.0
    An = A / norm[:, None]
    with np.errstate(all="ignore"):   # NumPy over-reports on matmul SIMD lanes
        S = np.nan_to_num(An @ An.T, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(S, 0.0)
    S = np.clip(S, 0.0, None)

    # (2) row-wise threshold, then re-symmetrise
    if row_threshold_pct and row_threshold_pct > 0:
        thr = np.percentile(S, row_threshold_pct, axis=1, keepdims=True)
        S = np.where(S >= thr, S, 0.0)
        S = np.maximum(S, S.T)

    # (3) diffusion-map embedding on W = S + eps (eps keeps the graph connected)
    W = S + eps
    d = W.sum(axis=1)
    d_alpha = d ** alpha
    Wn = W / np.outer(d_alpha, d_alpha)          # anisotropic normalisation
    dd = Wn.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(dd)
    Ms = (Wn * inv_sqrt[:, None]) * inv_sqrt[None, :]   # symmetric-normalised
    Ms = 0.5 * (Ms + Ms.T)                              # numerical symmetry

    w, v = np.linalg.eigh(Ms)                    # ascending eigenvalues
    order = np.argsort(w)[::-1]                   # descending
    v = v[:, order]
    psi = v * inv_sqrt[:, None]                   # diffusion coordinates

    # (4) first non-trivial coordinate = principal gradient
    grad = psi[:, 1].astype(float)
    return grad


def _aligned_corr(g: np.ndarray, g_ref: np.ndarray) -> Tuple[float, float]:
    """Sign-aligned Pearson + Spearman of a gradient against a reference.

    The gradient's sign is flipped, if needed, to make the Pearson correlation
    non-negative (the eigenvector sign carries no meaning). Returns (pearson,
    spearman) computed over nodes finite in both vectors.
    """
    from scipy.stats import pearsonr, spearmanr
    finite = np.isfinite(g) & np.isfinite(g_ref)
    if finite.sum() < 5:
        return float("nan"), float("nan")
    a, b = g[finite], g_ref[finite]
    r = pearsonr(a, b)[0]
    sign = -1.0 if r < 0 else 1.0
    a = sign * a
    r = pearsonr(a, b)[0]
    rho = spearmanr(a, b)[0]
    return float(r), float(rho)


# ===========================================================================
# Best-fit grid point per measure (from cached D_art)
# ===========================================================================

META_COLS = {"network_index", "filename", "eta", "gamma", "id"}


def _summary_path(measure: str) -> Path:
    return MORPHO_DIR / f"summary_indiv_{measure}_for_exp_{MORPHO_EXP}.csv"


def _metric_column(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c not in META_COLS]
    if len(cols) != 1:
        raise ValueError(f"expected exactly one metric column, got {cols}")
    return cols[0]


def best_fit_rows(measure: str) -> Tuple[float, float, pd.DataFrame]:
    """(eta*, gamma*, replicate-rows) for a measure's best fit to the consensus.

    Best fit = the (eta, gamma) grid point minimising the mean distance over its
    replicates (maximising the mean similarity for `communicability_corr`).
    """
    path = _summary_path(measure)
    if not path.exists():
        raise FileNotFoundError(f"cached distances not found for '{measure}': {path}")
    df = pd.read_csv(path)
    col = _metric_column(df)
    grp = df.groupby(["eta", "gamma"])[col].mean()
    best = grp.idxmax() if measure in IS_SIMILARITY else grp.idxmin()
    eta_star, gamma_star = float(best[0]), float(best[1])
    rows = df[(df["eta"] == eta_star) & (df["gamma"] == gamma_star)].copy()
    return eta_star, gamma_star, rows[["filename", "eta", "gamma", "id"]]


# ===========================================================================
# Network loading
# ===========================================================================

def load_net(filename: str) -> np.ndarray:
    arr = np.load(GEN_NETS_DIR / filename)
    return arr[0] if arr.ndim == 3 else arr


# ===========================================================================
# Stage: gradients
# ===========================================================================

def compute_gradients(measures: List[str], n_null: int, out_npz: Path,
                      bestfit_csv: Path, null_csv: Path) -> None:
    from tqdm import tqdm

    # --- consensus principal gradient (the reference topography) -------------
    consensus = np.load(CONSENSUS_PATH)
    g_ref = structural_gradient(consensus)
    print(f"  Consensus gradient computed (N={g_ref.size} nodes).")

    # --- best-fit gradients, per measure -------------------------------------
    bestfit_rows = []
    plot_vectors = {"consensus": g_ref}
    for m in measures:
        eta_star, gamma_star, rows = best_fit_rows(m)
        per_rep = []
        for _, r in rows.iterrows():
            g = structural_gradient(load_net(r["filename"]))
            pear, spear = _aligned_corr(g, g_ref)
            per_rep.append((r["filename"], int(r["id"]), pear, spear, g))
            bestfit_rows.append({
                "measure": m, "name": METHOD_NAMES.get(m, m),
                "eta": eta_star, "gamma": gamma_star,
                "id": int(r["id"]), "filename": r["filename"],
                "pearson": pear, "spearman": spear,
            })
        # medoid replicate (closest Pearson to the group mean) -> plot vector
        pears = np.array([p[2] for p in per_rep], dtype=float)
        medoid = int(np.nanargmin(np.abs(pears - np.nanmean(pears))))
        g_med = per_rep[medoid][4]
        # sign-align the stored vector to the consensus for plotting
        from scipy.stats import pearsonr
        finite = np.isfinite(g_med) & np.isfinite(g_ref)
        if pearsonr(g_med[finite], g_ref[finite])[0] < 0:
            g_med = -g_med
        plot_vectors[f"bestfit__{m}"] = g_med
        print(f"  {m:34s} best fit (eta={eta_star:+.3f}, gamma={gamma_star:.3f}) "
              f"mean r={np.nanmean(pears):+.3f}")

    pd.DataFrame(bestfit_rows).to_csv(bestfit_csv, index=False)
    print(f"  Saved best-fit gradients -> {bestfit_csv.name}")

    # --- null: random GNM networks from across the (eta, gamma) plane --------
    # filenames are identical across measures, so read one summary for the pool.
    pool = pd.read_csv(_summary_path(measures[0]))[["filename", "eta", "gamma"]]
    rng = np.random.default_rng(NULL_SEED)
    n_null = min(n_null, len(pool))
    idx = rng.choice(len(pool), size=n_null, replace=False)
    null_rows = []
    for j in tqdm(idx, desc="null gradients"):
        r = pool.iloc[int(j)]
        g = structural_gradient(load_net(r["filename"]))
        pear, spear = _aligned_corr(g, g_ref)
        null_rows.append({"filename": r["filename"], "eta": float(r["eta"]),
                          "gamma": float(r["gamma"]), "pearson": pear,
                          "spearman": spear})
    pd.DataFrame(null_rows).to_csv(null_csv, index=False)
    print(f"  Saved {n_null} null gradients -> {null_csv.name}")

    np.savez(out_npz, **plot_vectors)
    print(f"  Saved plot vectors -> {out_npz.name}")


# ===========================================================================
# Stage: metrics
# ===========================================================================

def compute_metrics(bestfit_csv: Path, null_csv: Path, measures: List[str]
                    ) -> pd.DataFrame:
    bf = pd.read_csv(bestfit_csv)
    null = pd.read_csv(null_csv)
    null_pear = null["pearson"].to_numpy(dtype=float)
    null_pear = null_pear[np.isfinite(null_pear)]

    rows = []
    for m in measures:
        sub = bf[bf["measure"] == m]
        mean_p = float(np.nanmean(sub["pearson"]))
        std_p = float(np.nanstd(sub["pearson"]))
        mean_s = float(np.nanmean(sub["spearman"]))
        # percentile of this measure's mean_p within the "any GNM" null
        pct = float((null_pear < mean_p).mean() * 100.0)
        rows.append({
            "measure": m, "name": METHOD_NAMES.get(m, m),
            "is_similarity": m in IS_SIMILARITY,
            "eta": float(sub["eta"].iloc[0]), "gamma": float(sub["gamma"].iloc[0]),
            "mean_pearson": mean_p, "std_pearson": std_p,
            "mean_spearman": mean_s,
            "n_replicates": int(len(sub)),
            "null_percentile": pct,
        })
    df = pd.DataFrame(rows).sort_values("mean_pearson", ascending=False)
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


def plot(results: pd.DataFrame, null_csv: Path, out_npz: Path, out_pdf: Path
         ) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((18, 22)) if have_viz else (18 / 2.54, 22 / 2.54))
    GRAY = (0.5, 0.5, 0.5)

    null = pd.read_csv(null_csv)["pearson"].to_numpy(dtype=float)
    null = null[np.isfinite(null)]
    null_med = float(np.median(null))
    null_p95 = float(np.percentile(null, 95))
    vecs = np.load(out_npz)
    g_ref = vecs["consensus"]

    res = results.copy()
    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(2, 1, height_ratios=[1.15, 2.0], hspace=0.05)

    # ---- Panel A: gradient-correlation ranking (with null band) ------------
    axA = subfigs[0].subplots(1, 1)
    subfigs[0].subplots_adjust(left=0.22, right=0.97, top=0.86, bottom=0.22)
    y = np.arange(len(res))[::-1]
    colors = [METRIC_COLORS.get(m, GRAY) for m in res["measure"]]
    axA.axvspan(null.min(), null_p95, color=GRAY, alpha=0.12, zorder=0,
                label="null range (any GNM, to 95th pct)")
    axA.axvline(null_med, color=GRAY, lw=1.0, ls="--", zorder=1,
                label=f"null median ({null_med:.2f})")
    axA.barh(y, res["mean_pearson"], xerr=res["std_pearson"], color=colors,
             edgecolor="black", linewidth=0.4, height=0.7, zorder=2,
             error_kw=dict(ecolor="black", elinewidth=0.7, capsize=2))
    axA.set_yticks(y)
    axA.set_yticklabels(res["name"], fontsize=8)
    axA.set_xlabel("principal-gradient correlation to consensus  (Pearson r, "
                   "sign-aligned; higher = more plausible)", fontsize=8)
    for yi, (rp, pct) in enumerate(zip(res["mean_pearson"][::-1],
                                       res["null_percentile"][::-1])):
        axA.text(rp + 0.01, yi, f"{rp:.2f}  (p{pct:.0f})", va="center",
                 fontsize=6.5)
    axA.legend(loc="lower right", frameon=False, fontsize=6.5)
    axA.set_title("A   Structural-gradient plausibility of each measure's "
                  "best-fit networks", loc="left", fontsize=10,
                  fontweight="bold", pad=8)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # ---- Panel B: per-measure consensus-vs-best-fit gradient scatter -------
    order = list(res["measure"])
    axes = subfigs[1].subplots(2, 4)
    subfigs[1].subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.09,
                               hspace=0.45, wspace=0.30)
    for k, m in enumerate(order):
        ax = axes[k // 4, k % 4]
        g = vecs[f"bestfit__{m}"]
        finite = np.isfinite(g) & np.isfinite(g_ref)
        ax.scatter(g_ref[finite], g[finite], s=8,
                   color=METRIC_COLORS.get(m, GRAY), alpha=0.75, linewidths=0)
        rr = float(res[res["measure"] == m]["mean_pearson"].iloc[0])
        ax.set_title(f"{METHOD_NAMES.get(m, m)}\nr={rr:.2f}", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if k % 4 == 0:
            ax.set_ylabel("best-fit\ngradient", fontsize=7)
        if k // 4 == 1:
            ax.set_xlabel("consensus gradient", fontsize=7)
    subfigs[1].suptitle("B   Node-wise principal gradient: empirical consensus "
                        "vs each measure's best-fit network", x=0.02, ha="left",
                        fontsize=10, fontweight="bold")

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
    ap.add_argument("--stage", choices=["gradient", "metrics", "plot", "all"],
                    default="all")
    ap.add_argument("--n-null", type=int, default=N_NULL_DEFAULT,
                    help="random GNM networks for the 'any GNM' null distribution")
    ap.add_argument("--measures", default=None, help="comma-separated subset")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny self-test: few measures + small null, separate dir")
    args = ap.parse_args()

    measures = (args.measures.split(",") if args.measures
                else (["frobenius", "delta_con", "portrait"] if args.smoke
                      else SELECTED_MEASURES))
    n_null = 60 if args.smoke else args.n_null

    out_dir = OUT_DIR if not args.smoke else (OUT_DIR.parent / "structural_gradient_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "gradient_vectors.npz"
    bestfit_csv = out_dir / "bestfit_gradients.csv"
    null_csv = out_dir / "null_gradients.csv"
    results_csv = out_dir / "structural_gradient_results.csv"
    meta_json = out_dir / "structural_gradient_meta.json"
    fig_pdf = out_dir / "fig_structural_gradient.pdf"

    print(f"Output dir : {out_dir}")
    print(f"Measures   : {measures}   n_null: {n_null}\n")

    if args.stage in ("gradient", "all"):
        print("=" * 60, "\nStage: gradients\n", "=" * 60, sep="")
        compute_gradients(measures, n_null, npz_path, bestfit_csv, null_csv)

    if args.stage in ("metrics", "all"):
        print("\n" + "=" * 60, "\nStage: metrics\n", "=" * 60, sep="")
        results = compute_metrics(bestfit_csv, null_csv, measures)
        results.to_csv(results_csv, index=False)
        meta = {
            "criterion": "principal structural gradient topography (diffusion map)",
            "gradient_params": {
                "row_threshold_pct": ROW_THRESHOLD_PCT,
                "diffusion_alpha": DIFFUSION_ALPHA,
                "affinity": "cosine similarity of connectivity profiles",
            },
            "reference": "empirical HCP consensus (10% density)",
            "best_fit_rule": "argmin mean distance over replicates "
                             "(argmax for communicability_corr)",
            "null": f"{n_null} random GNM networks across the (eta,gamma) plane",
            "morphospace_run": f"{MORPHO_DATASET}/{MORPHO_EXP}",
            "measures": measures,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        meta_json.write_text(json.dumps(meta, indent=2))
        print(f"\n  Saved results -> {results_csv.name}")
        print(f"  Saved meta    -> {meta_json.name}\n")
        print(results.to_string(index=False))

    if args.stage in ("plot", "all"):
        print("\n" + "=" * 60, "\nStage: figure\n", "=" * 60, sep="")
        results = pd.read_csv(results_csv)
        plot(results, null_csv, npz_path, fig_pdf)

    print("\nDone.")


if __name__ == "__main__":
    main()
