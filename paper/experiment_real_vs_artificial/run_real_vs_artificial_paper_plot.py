"""
Real-vs-Artificial ground-truth validity check (analysis S3)
============================================================

A *model-free* validity criterion for network-distance measures, complementing
the synthetic parameter-recovery experiment.

Idea
----
The empirical consensus connectome is built from real subjects, so real
individual connectomes are, by construction, genuinely brain-like; GNM networks
are synthetic approximations. A trustworthy distance measure should therefore
score **real subjects as closer to the consensus than GNM networks are**. We
rank all measures by how cleanly they separate the two, and report it as a new
evaluation axis.

What is new vs. reused
----------------------
  * D_art  (25,000 GNM-to-consensus distances per measure) is ALREADY cached by
    the main morphospace run, one CSV per measure:
        output/gnm/hcp_schaefer_100_dataset/106_distance_metrics_mst_animal_0_density10/
            summary_indiv_<measure>_for_exp_106_distance_metrics_mst_animal_0_density10.csv
    We load it; we do NOT recompute it (spec Section 9).
  * D_real (100 real-subject-to-consensus distances per measure) is the only new
    work. By default we use a leave-one-out (LOO) consensus C_{-i} rebuilt from
    the other 99 subjects (spec Section 3a), using the exact same construction as
    the saved consensus (struct_consensus weighted + binarize_network retain=10).
    This was verified to reproduce 01_consensus_bin_density_10_percent_100.npy
    bit-for-bit when all 100 subjects are used.

Orientation
-----------
All distances come from the SAME evaluator objects for real and artificial, so
orientation is automatically consistent. Among the 8 selected measures only
`communicability_corr` is a similarity (higher = more similar); every other one
is already a distance (higher = less similar). We define a per-network closeness
    s = +raw   for similarity measures
    s = -raw   for distance measures
so that *higher s = closer to the consensus* for every measure.

Stages (all idempotent)
-----------------------
    real    : build LOO consensuses + compute D_real, cache to disk
    metrics : load cached D_art + D_real, compute discrimination metrics ->
              real_vs_artificial_results.csv (+ meta JSON)
    plot    : Panel A ranking bar chart + Panel B distribution overlap -> PDF
    all     : the three above (default)

Usage
-----
    conda activate ma_thesis
    python run_real_vs_artificial.py --stage all
    python run_real_vs_artificial.py --stage real
    python run_real_vs_artificial.py --reference full     # leaky baseline (3c)
    python run_real_vs_artificial.py --smoke              # tiny fast self-test
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

# Paths, the 8 selected measures and their orientation / names / colours now
# live in the single shared config at the repo root (experiments_config.py).
from experiments_config import (
    DATA_DIR, RAW_DIR, MORPHO_DATASET, MORPHO_EXP, MORPHO_DIR,
    CONSENSUS_PATH, INDIVIDUALS_PATH, DIST_MATRIX_PATH, RAW_MAT_PATH, RAW_COORDS_PATH,
    OUT_DIR, HEMIID, DENSITY_RETAIN, N_HARD_DEFAULT, META_COLS,
    SELECTED_MEASURES, IS_SIMILARITY, METHOD_NAMES, METRIC_COLORS,
)


# ===========================================================================
# Evaluators (same objects the morphospace / fine-recovery runs use)
# ===========================================================================

def build_evaluators(measures: List[str]) -> Dict[str, object]:
    import torch
    from src.comparing_connectomes.frobenius_comparer import FrobeniusEvaluator
    from src.comparing_connectomes.delta_con_evaluator import DeltaConEvaluator
    from src.comparing_connectomes.spectral_distance_comparer import SpectralDistanceEvaluator
    from src.comparing_connectomes.communicability_comparer import CommunicabilityCorrEvaluator
    from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence
    from src.comparing_connectomes.netrd_comparer import NetrdEvaluator
    from src.comparing_connectomes.energy_comparer import EnergyEvaluator

    dist_np = np.load(DIST_MATRIX_PATH)
    dist_tensor = torch.tensor(dist_np, dtype=torch.float32)
    from gnm import evaluation as gnm_eval
    energy_criteria = gnm_eval.MaxCriteria([
        gnm_eval.DegreeKS(),
        gnm_eval.ClusteringKS(),
        gnm_eval.EdgeLengthKS(dist_tensor),
        gnm_eval.BetweennessKS(),
    ])

    factories = {
        "frobenius":                       lambda: FrobeniusEvaluator(),
        "delta_con":                       lambda: DeltaConEvaluator(),
        "netrd_non_backtracking_spectral": lambda: NetrdEvaluator(method="netrd_non_backtracking_spectral"),
        "spectral_distance_adjacency":     lambda: SpectralDistanceEvaluator(method="adjacency"),
        "communicability_corr":            lambda: CommunicabilityCorrEvaluator(),
        "portrait":                        lambda: PortraitDivergence(),
        "net_simile":                      lambda: NetrdEvaluator(method="net_simile"),
        "energy":                          lambda: EnergyEvaluator([energy_criteria]),
    }
    return {m: factories[m]() for m in measures}


# ===========================================================================
# Stage: build LOO consensuses
# ===========================================================================

def _load_raw_weighted_connectomes() -> np.ndarray:
    """Raw streamline-count connectomes, shape (subjects, regions, regions).

    Reproduces the loader in the preprocessing notebook exactly so the rebuilt
    consensus matches 01_consensus_bin_density_10_percent_100.npy.
    """
    import h5py
    data = h5py.File(RAW_MAT_PATH, "r")
    n = 100
    conn = np.zeros((n, n, n))
    for s in range(n):
        ref = data["DTI_fibers_VolNorm_HCP"][0][s]
        conn[..., s] = data[ref][()]
    return conn.T  # -> (subjects, regions, regions)


def _consensus_from(weighted_subjects: np.ndarray, dist_np: np.ndarray) -> np.ndarray:
    """struct_consensus (weighted) + binarize_network(retain=10) -> (100,100)."""
    from netneurotools.networks import struct_consensus, binarize_network
    stacked = np.transpose(weighted_subjects, (1, 2, 0))   # (regions, regions, subjects)
    cons = struct_consensus(stacked, dist_np, hemiid=HEMIID, weighted=True)
    return binarize_network(cons, retain=DENSITY_RETAIN).reshape(100, 100)


def build_loo_consensuses(out_path: Path, n_subjects: int) -> np.ndarray:
    """Build (and cache) the n LOO consensuses C_{-i}, shape (n,100,100)."""
    if out_path.exists():
        arr = np.load(out_path)
        if arr.shape[0] >= n_subjects:
            print(f"  LOO consensuses cached -> {out_path.name} {arr.shape}")
            return arr[:n_subjects]

    from tqdm import tqdm
    dist_np = np.load(DIST_MATRIX_PATH)
    weighted = _load_raw_weighted_connectomes()

    # sanity: full rebuild must reproduce the saved consensus
    full = _consensus_from(weighted, dist_np)
    saved = np.load(CONSENSUS_PATH)[0]
    if not np.array_equal(full, saved):
        frac = float((full == saved).mean())
        print(f"  WARNING: full rebuild != saved consensus (match={frac:.4f}). "
              f"LOO references still internally consistent.")
    else:
        print("  Full rebuild reproduces saved consensus exactly (LOO valid).")

    loo = np.zeros((n_subjects, 100, 100), dtype=np.int64)
    for i in tqdm(range(n_subjects), desc="LOO consensus"):
        others = np.delete(weighted, i, axis=0)
        loo[i] = _consensus_from(others, dist_np)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, loo)
    print(f"  Saved {n_subjects} LOO consensuses -> {out_path.name}")
    return loo


# ===========================================================================
# Stage: compute D_real
# ===========================================================================

def compute_d_real(out_csv: Path, measures: List[str], reference: str,
                   n_subjects: int) -> pd.DataFrame:
    """Per-subject raw distance to its reference consensus, one column / measure."""
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        if set(measures).issubset(df.columns) and len(df) >= n_subjects:
            print(f"  D_real cached -> {out_csv.name}")
            return df
    import torch
    from tqdm import tqdm

    individuals = np.load(INDIVIDUALS_PATH).astype(float)[:n_subjects]   # (n,100,100)

    if reference == "loo":
        refs = build_loo_consensuses(OUT_DIR / "loo_consensuses.npy", n_subjects)
    elif reference == "full":
        refs = np.repeat(np.load(CONSENSUS_PATH).astype(float), n_subjects, axis=0)
    else:
        raise ValueError(f"unknown reference: {reference}")

    evaluators = build_evaluators(measures)
    rows = []
    for i in tqdm(range(n_subjects), desc="D_real"):
        gen_t = torch.tensor(individuals[i], dtype=torch.float32).unsqueeze(0)   # (1,100,100)
        tgt_t = torch.tensor(refs[i], dtype=torch.float32).unsqueeze(0)          # (1,100,100)
        row = {"subject": i}
        for m, ev in evaluators.items():
            try:
                row[m] = float(ev(gen_t, tgt_t)[0])
            except Exception as e:
                row[m] = float("nan")
                print(f"    subject {i} / {m}: failed ({e})")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  Saved D_real -> {out_csv.name}")
    return df


# ===========================================================================
# Stage: load cached D_art
# ===========================================================================

def _metric_column(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c not in META_COLS]
    if len(cols) != 1:
        raise ValueError(f"expected exactly one metric column, got {cols}")
    return cols[0]


def load_d_art(measure: str) -> pd.DataFrame:
    """Cached per-network GNM->consensus distances. Columns: eta,gamma,id,raw."""
    path = MORPHO_DIR / f"summary_indiv_{measure}_for_exp_{MORPHO_EXP}.csv"
    if not path.exists():
        raise FileNotFoundError(f"cached D_art not found for '{measure}': {path}")
    df = pd.read_csv(path)
    col = _metric_column(df)
    out = df[["eta", "gamma", "id"]].copy()
    out["raw"] = df[col].astype(float)
    return out


# ===========================================================================
# Stage: discrimination metrics
# ===========================================================================

def _closeness(raw: np.ndarray, measure: str) -> np.ndarray:
    """higher = closer to the reference, for every measure."""
    return raw if measure in IS_SIMILARITY else -raw


def _clean(x: np.ndarray) -> Tuple[np.ndarray, int]:
    finite = np.isfinite(x)
    return x[finite], int((~finite).sum())


def discrimination(s_real: np.ndarray, s_art: np.ndarray,
                   n_hard: int) -> Dict[str, float]:
    from scipy.stats import mannwhitneyu
    n_real, n_art = len(s_real), len(s_art)

    # U1 counts (real_i > art_j) + 0.5*ties  ==  AUC * n_real * n_art
    u1, p = mannwhitneyu(s_real, s_art, alternative="two-sided")
    auc = float(u1) / (n_real * n_art)

    # hard case: only the n_hard GNMs closest to the reference (highest s_art)
    k = min(n_hard, n_art)
    s_art_hard = np.sort(s_art)[::-1][:k]
    u1_h, _ = mannwhitneyu(s_real, s_art_hard, alternative="two-sided")
    auc_hard = float(u1_h) / (n_real * k)

    return {
        "auc": auc,
        "auc_hardcase": auc_hard,
        "frac_gnms_beaten": auc,            # identical to AUC (plain-language gloss)
        "rank_biserial": 2.0 * auc - 1.0,
        "mw_u": float(u1),
        "mw_p": float(p),
        "median_d_real": float(np.median(-s_real)),   # oriented distance: lower = closer
        "median_d_art": float(np.median(-s_art)),
        "n_hard_used": k,
    }


def compute_metrics(d_real: pd.DataFrame, measures: List[str],
                    n_hard: int) -> pd.DataFrame:
    rows = []
    for m in measures:
        s_real_raw = d_real[m].to_numpy(dtype=float)
        d_art = load_d_art(m)
        s_art_raw = d_art["raw"].to_numpy(dtype=float)

        s_real, dropped_real = _clean(_closeness(s_real_raw, m))
        s_art,  dropped_art  = _clean(_closeness(s_art_raw, m))

        res = discrimination(s_real, s_art, n_hard)
        rows.append({
            "measure": m,
            "name": METHOD_NAMES.get(m, m),
            "is_similarity": m in IS_SIMILARITY,
            "n_real_used": len(s_real),
            "n_art_used": len(s_art),
            "n_dropped": dropped_real + dropped_art,
            **res,
        })
    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return df


# ===========================================================================
# Stage: figure
# ===========================================================================

def _viz():
    """(viz, ok) — degrade gracefully if vizman is unavailable."""
    try:
        from vizman import viz
        viz.set_visual_style(font_family="Arial")
        return viz, True
    except Exception:
        return None, False


def _normalise_pooled(s_real: np.ndarray, s_art: np.ndarray):
    lo = min(s_real.min(), s_art.min())
    hi = max(s_real.max(), s_art.max())
    if hi - lo < 1e-12:
        return s_real * 0.0, s_art * 0.0
    return (s_real - lo) / (hi - lo), (s_art - lo) / (hi - lo)


def plot(results: pd.DataFrame, d_real: pd.DataFrame, out_pdf: Path,
         n_hard: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz, have_viz = _viz()
    figsize = (viz.cm_to_inch((12,6)) if have_viz else (18 / 2.54, 20 / 2.54))

    REAL_COLOR = (0.84, 0.19, 0.15)   # red
    ART_COLOR  = (0.30, 0.45, 0.69)   # blue
    GRAY = (0.5, 0.5, 0.5)

    fig, (axB, axA) = plt.subplots(1,2, figsize=figsize,
                                #    gridspec_kw={"height_ratios": [1.0, 1.4]}, 
                                   sharey=True)

    # ---- Panel A: AUC ranking ----------------------------------------------
    res = results.copy()
    y = np.arange(len(res))[::-1]
    colors = [METRIC_COLORS.get(m, GRAY) for m in res["measure"]]
    axA.barh(y, res["auc"], color=colors, edgecolor="black", linewidth=0.4,
             height=0.7, zorder=2, label="AUC (all 25k GNMs)")
    axA.scatter(res["auc_hardcase"], y, marker="D", s=26, color="black",
                zorder=4, label=f"AUC (hard case, {n_hard} closest)")
    axA.axvline(0.5, color=GRAY, lw=0.9, ls="--", zorder=1, label="chance (0.5)")
    axA.set_yticks(y)
    axA.set_yticklabels(res["name"]) # , fontsize=8)
    axA.set_xlim(min(0.45, float(res[["auc", "auc_hardcase"]].min().min()) - 0.03), 1.005)
    axA.set_xlabel("AUC\n(P[real subject closer to consensus than GNM])")
    for yi, (a, h) in zip(y, zip(res["auc"], res["auc_hardcase"])):
        axA.text(a + 0.004, yi + 0.18, f"{a:.3f}", va="center") # , fontsize=6)
        axA.text(h, yi - 0.34, f"{h:.2f}", va="center", ha="center", # fontsize=5.5, 
        color="black")
    # axA.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
    #            frameon=False, fontsize=6.5, handletextpad=0.4, columnspacing=1.2)
    # axA.set_title("A   Discrimination ranking: real subjects vs GNM networks",
    #               loc="left", fontsize=10, fontweight="bold", pad=20)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # ---- Panel B: distribution overlap -------------------------------------
    # per-measure pooled-normalised closeness; artificial as half-violin,
    # real subjects as jittered points. ordered like Panel A (best at top).
    order = list(res["measure"])
    positions = np.arange(len(order))[::-1]
    rng = np.random.default_rng(0)

    for pos, m in zip(positions, order):
        s_real, _ = _clean(_closeness(d_real[m].to_numpy(dtype=float), m))
        s_art, _ = _clean(_closeness(load_d_art(m)["raw"].to_numpy(dtype=float), m))
        nr, na = _normalise_pooled(s_real, s_art)

        parts = axB.violinplot([na], positions=[pos], vert=False,
                               showextrema=False, widths=0.9)
        for body in parts["bodies"]:
            body.set_facecolor(ART_COLOR)
            body.set_edgecolor("none")
            body.set_alpha(0.45)
            # clip to lower half -> half-violin below the row baseline
            verts = body.get_paths()[0].vertices
            verts[:, 1] = np.clip(verts[:, 1], -np.inf, pos)
        jitter = rng.uniform(0.04, 0.30, size=len(nr))
        axB.scatter(nr, pos + jitter, s=12, color=REAL_COLOR, alpha=0.8,
                    linewidths=0, zorder=3)

    axB.set_yticks(positions)
    axB.set_yticklabels(res["name"]) # , fontsize=8)
    axB.set_xlabel("closeness to consensus\n(higher = closer)") # pooled min-max normalised; 
    axB.set_xlim(-0.02, 1.02)
    axB.set_ylim(-0.8, len(order) - 0.2)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axB.legend(handles=[
        Patch(facecolor=ART_COLOR, alpha=0.45, label="GNM networks (25k)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REAL_COLOR,
               markersize=6, label="real subjects (100)"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
    # loc="upper left", frameon=True, framealpha=0.9, edgecolor="none",
       facecolor="white") # , fontsize=7)
    # axB.set_title("B   Closeness distributions: real subjects (points) vs GNM networks (violins)",
    #               loc="left", fontsize=10, fontweight="bold")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure -> {out_pdf}")


# ===========================================================================
# Orientation sanity check
# ===========================================================================

def orientation_sanity(results: pd.DataFrame) -> bool:
    ok = True
    for m in ("frobenius", "energy"):
        row = results[results["measure"] == m]
        if row.empty:
            continue
        dr = float(row["median_d_real"].iloc[0])
        da = float(row["median_d_art"].iloc[0])
        passed = dr < da
        ok &= passed
        flag = "OK" if passed else "*** FAIL ***"
        print(f"  [{flag}] {m}: median_d_real={dr:.4g} < median_d_art={da:.4g} ? {passed}")
    return ok


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["real", "metrics", "plot", "all"], default="all")
    ap.add_argument("--reference", choices=["loo", "full"], default="loo",
                    help="LOO consensus (default, 3a) or full-consensus baseline (leaky, 3c)")
    ap.add_argument("--n-hard", type=int, default=N_HARD_DEFAULT,
                    help="hard-case: number of closest GNMs per measure")
    ap.add_argument("--measures", default=None, help="comma-separated subset")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny self-test: few subjects + fast measures, separate dir")
    args = ap.parse_args()

    measures = (args.measures.split(",") if args.measures
                else (["frobenius", "delta_con"] if args.smoke else SELECTED_MEASURES))
    n_subjects = 8 if args.smoke else 100

    global OUT_DIR
    out_dir = OUT_DIR if not args.smoke else (OUT_DIR.parent / "real_vs_artificial_smoke")
    OUT_DIR = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d_real_csv = out_dir / f"d_real_{args.reference}.csv"
    results_csv = out_dir / "real_vs_artificial_results.csv"
    meta_json = out_dir / "real_vs_artificial_meta.json"
    fig_pdf = out_dir / "fig_real_vs_artificial.pdf"

    print(f"Output dir : {out_dir}")
    print(f"Reference  : {args.reference}   measures: {measures}   n_subjects: {n_subjects}\n")

    if args.stage in ("real", "all"):
        print("=" * 60, "\nStage: D_real\n", "=" * 60, sep="")
        compute_d_real(d_real_csv, measures, args.reference, n_subjects)

    if args.stage in ("metrics", "all"):
        print("\n" + "=" * 60, "\nStage: discrimination metrics\n", "=" * 60, sep="")
        d_real = pd.read_csv(d_real_csv)
        results = compute_metrics(d_real, measures, args.n_hard)
        results.to_csv(results_csv, index=False)
        meta = {
            "reference_method": args.reference,
            "reference_note": ("leave-one-out consensus C_{-i} (spec 3a)"
                               if args.reference == "loo"
                               else "full consensus baseline (leaky, spec 3c)"),
            "n_real": n_subjects,
            "n_art_per_measure": int(load_d_art(measures[0]).shape[0]),
            "n_hard": args.n_hard,
            "morphospace_run": f"{MORPHO_DATASET}/{MORPHO_EXP}",
            "d_art_source": "cached summary_indiv_<measure> CSVs (not recomputed)",
            "similarity_measures": sorted(IS_SIMILARITY & set(SELECTED_MEASURES)),
            "measures": measures,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        meta_json.write_text(json.dumps(meta, indent=2))
        print(f"\n  Saved results -> {results_csv.name}")
        print(f"  Saved meta    -> {meta_json.name}\n")
        print(results.to_string(index=False))
        print("\nOrientation sanity check (Frobenius & energy: real should be closer):")
        ok = orientation_sanity(results)
        print(f"\nOrientation sanity: {'PASSED' if ok else 'FAILED'}")

    if args.stage in ("plot", "all"):
        print("\n" + "=" * 60, "\nStage: figure\n", "=" * 60, sep="")
        results = pd.read_csv(results_csv)
        d_real = pd.read_csv(d_real_csv)
        plot(results, d_real, fig_pdf, args.n_hard)

    print("\nDone.")


if __name__ == "__main__":
    main()
