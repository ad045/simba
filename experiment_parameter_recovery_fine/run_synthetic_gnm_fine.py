"""
Fine-grained parameter recovery within the human-plausible (eta, gamma) range
=============================================================================

Analysis S1. See gnm_fine_config.py for the scientific motivation.

Pipeline (all stages idempotent — existing files are skipped):

  generate : build the fine grid of consensus networks (one consensus per grid
             point, from N_CONSENSUS_FINE networks) + N_GROUND_TRUTH individual
             ground-truth networks at (eta, gamma) sampled from the plausible
             window. Writes ground_truth_params.csv manifest.

  compare  : for every ground-truth network compute its distance to every grid
             consensus network, under each distance measure. Recovered (eta,
             gamma) = argmin grid point. One CSV per measure.

  plot     : per measure, Pearson r between true and recovered eta / gamma
             ("fine-grained accuracy"). Figures:
               - fine_recovery_r_bars.pdf      bar of r per measure
               - fine_recovery_scatter_top.pdf true vs recovered for top measures
               - fine_vs_wide_ranking.pdf      ranking vs the wide-target experiment

Usage
-----
  # from inside this folder, ma_thesis env:
  python run_synthetic_gnm_fine.py --stage all
  python run_synthetic_gnm_fine.py --stage generate
  python run_synthetic_gnm_fine.py --stage compare
  python run_synthetic_gnm_fine.py --stage plot

  # fast end-to-end self-test (tiny grid, few nets, fast measures, separate dir):
  python run_synthetic_gnm_fine.py --stage all --smoke
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --- robust imports: this folder (for the config) + repo root (for src.*) -----
_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR  = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
for _p in (str(_THIS_DIR), str(ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gnm_fine_config as cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = ROOT_DIR / "data" / "preprocessed" / "hcp_schaefer_100_dataset"
SEED_PATH = ROOT_DIR / "data" / "preprocessed" / "seeds" / "mst_schaeffer.npy"

LAMBDAH = 1.0
DEVICE  = "cpu"
HEMIID  = np.array([0] * 50 + [1] * 50).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Runtime config object (so --smoke can shrink everything without touching cfg)
# ---------------------------------------------------------------------------

class RunCfg:
    """Resolved configuration for one run (full or smoke)."""

    def __init__(self, smoke: bool, measures: List[str] | None):
        self.smoke = smoke
        if smoke:
            self.grid_eta   = np.linspace(-4.7, -2.7, 3)
            self.grid_gamma = np.linspace(0.37, 0.63, 3)
            self.n_consensus = 2
            self.n_truth     = 4
            self.out_dir = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_fine_smoke"
            self.default_measures = ["frobenius", "hamming"]
        else:
            self.grid_eta   = cfg.GRID_ETA
            self.grid_gamma = cfg.GRID_GAMMA
            self.n_consensus = cfg.N_CONSENSUS_FINE
            self.n_truth     = cfg.N_GROUND_TRUTH
            self.out_dir = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_fine"
            self.default_measures = None  # = all registered

        self.grid_combos = [
            (float(e), float(g)) for g in self.grid_gamma for e in self.grid_eta
        ]
        self.eta_range   = (float(self.grid_eta[0]),   float(self.grid_eta[-1]))
        self.gamma_range = (float(self.grid_gamma[0]), float(self.grid_gamma[-1]))
        self.measures = measures  # explicit subset, or None

    @property
    def grid_dir(self):        return self.out_dir / "grid_consensus"
    @property
    def gt_dir(self):          return self.out_dir / "ground_truth"
    @property
    def gt_manifest(self):     return self.out_dir / "ground_truth_params.csv"
    @property
    def comparison_dir(self):  return self.out_dir / "comparison_results"
    @property
    def plot_dir(self):        return self.out_dir / "plots"

    def normalise(self, eta: float, gamma: float) -> Tuple[float, float]:
        return (
            (eta   - self.eta_range[0])   / (self.eta_range[1]   - self.eta_range[0]),
            (gamma - self.gamma_range[0]) / (self.gamma_range[1] - self.gamma_range[0]),
        )


# ===========================================================================
# Stage 1 — generation
# ===========================================================================

def _gen_imports():
    import torch
    from gnm.model import BinaryGenerativeParameters
    from gnm import generative_rules
    from gnm.fitting import RunConfig, perform_run
    from netneurotools.networks import struct_consensus, threshold_network
    return (torch, BinaryGenerativeParameters, generative_rules,
            RunConfig, perform_run, struct_consensus, threshold_network)


def generate(rc: RunCfg) -> None:
    (torch, BinaryGenerativeParameters, generative_rules,
     RunConfig, perform_run, struct_consensus, threshold_network) = _gen_imports()
    from tqdm import tqdm

    dist_np       = np.load(DATA_DIR / "02_distance_matrices" / "distance_matrix_100.npy")
    consensus_hcp = np.load(DATA_DIR / "01_connectomes" / "01_consensus_bin_density_10_percent_100.npy")
    seed_np       = np.load(SEED_PATH)

    total_edges = int(consensus_hcp[0].sum() / 2)        # 495
    n_edges     = total_edges - (dist_np.shape[0] - 1)   # 396
    dist_tensor = torch.tensor(dist_np, dtype=torch.float32)
    seed_tensor = torch.tensor(seed_np, dtype=torch.float32)

    def generate_one(eta, gamma):
        bp = BinaryGenerativeParameters(
            eta=eta, gamma=gamma, lambdah=LAMBDAH,
            distance_relationship_type="powerlaw",
            preferential_relationship_type="powerlaw",
            heterochronicity_relationship_type="powerlaw",
            generative_rule=generative_rules.MatchingIndex(),
            num_iterations=n_edges,
        )
        run = RunConfig(binary_parameters=bp, distance_matrix=dist_tensor,
                        seed_adjacency_matrix=seed_tensor, num_simulations=1)
        exp = perform_run(run, save_model=True, save_run_history=False,
                          device=torch.device(DEVICE))
        return exp.model.adjacency_matrix.cpu().numpy()[0]

    def load_or_generate(path: Path, eta, gamma):
        if path.exists():
            return np.load(path)
        net = generate_one(eta, gamma)
        np.save(path, net)
        return net

    def build_consensus(nets: List[np.ndarray]) -> np.ndarray:
        arr      = np.array(nets)                          # (k, n, n)
        weighted = struct_consensus(np.transpose(arr, (1, 2, 0)), dist_np,
                                    hemiid=HEMIID, weighted=True)
        return threshold_network(weighted, retain=10)

    print(f"Edges: total={total_edges}, MST seed={dist_np.shape[0]-1}, GNM iters={n_edges}")
    print(f"Fine grid: {len(rc.grid_eta)}x{len(rc.grid_gamma)} = "
          f"{len(rc.grid_combos)} points x {rc.n_consensus} nets")
    print(f"Ground truth: {rc.n_truth} individual nets "
          f"(eta in {cfg.SAMPLE_ETA_RANGE}, gamma in {cfg.SAMPLE_GAMMA_RANGE})\n")

    rc.grid_dir.mkdir(parents=True, exist_ok=True)
    rc.gt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: ground-truth individual networks + manifest ----------------
    print("=" * 60, "\nStep 1: ground-truth networks\n", "=" * 60, sep="")
    truths = cfg.sample_ground_truth(rc.n_truth, cfg.SAMPLE_SEED)
    manifest = []
    for idx, (eta, gamma) in enumerate(tqdm(truths, desc="Ground truth")):
        gdir = rc.gt_dir / cfg.gt_dir_name(idx)
        gdir.mkdir(parents=True, exist_ok=True)
        for net_id in range(cfg.N_NETS_PER_TRUTH):
            path = gdir / cfg.net_filename(eta, gamma, net_id)
            load_or_generate(path, eta, gamma)
        manifest.append({"gt_idx": idx, "true_eta": eta, "true_gamma": gamma})
    pd.DataFrame(manifest).to_csv(rc.gt_manifest, index=False)
    print(f"  Wrote manifest -> {rc.gt_manifest.name}\n")

    # ---- Step 2: grid consensus --------------------------------------------
    print("=" * 60, f"\nStep 2: grid consensus ({len(rc.grid_combos)} points)\n", "=" * 60, sep="")
    for eta, gamma in tqdm(rc.grid_combos, desc="Grid points"):
        pdir           = rc.grid_dir / cfg.param_dir_name(eta, gamma)
        nets_dir       = pdir / f"consensus_{rc.n_consensus}"
        consensus_path = pdir / "consensus.npy"
        nets_dir.mkdir(parents=True, exist_ok=True)
        if consensus_path.exists():
            continue
        nets = [load_or_generate(nets_dir / cfg.net_filename(eta, gamma, i), eta, gamma)
                for i in range(rc.n_consensus)]
        np.save(consensus_path, build_consensus(nets))

    print(f"\nGeneration done. Output -> {rc.out_dir}")


# ===========================================================================
# Stage 2 — comparison
# ===========================================================================

def build_evaluators(measure_subset: List[str] | None) -> Dict[str, object]:
    import torch
    from src.comparing_connectomes.delta_con_evaluator import DeltaConEvaluator
    from src.comparing_connectomes.frobenius_comparer import FrobeniusEvaluator
    from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence
    from src.comparing_connectomes.hamming_comparer import HammingEvaluator
    from src.comparing_connectomes.f1_comparer import F1Evaluator
    from src.comparing_connectomes.jaccard_comparer import JaccardEvaluator
    from src.comparing_connectomes.spectral_distance_comparer import SpectralDistanceEvaluator
    from src.comparing_connectomes.communicability_comparer import CommunicabilityCorrEvaluator
    from src.comparing_connectomes.communicability_jsd_comparer import CommunicabilityJSDEvaluator
    from src.comparing_connectomes.network_mutual_information_comparer import (
        NetworkMutualInformationEvaluator, DCNetworkMutualInformationEvaluator)
    from src.comparing_connectomes.netrd_comparer import NetrdEvaluator
    from src.comparing_connectomes.energy_comparer import EnergyEvaluator

    dist_np     = np.load(DATA_DIR / "02_distance_matrices" / "distance_matrix_100.npy")
    dist_tensor = torch.tensor(dist_np, dtype=torch.float32)
    from gnm import evaluation as gnm_eval
    energy_criteria = gnm_eval.MaxCriteria([
        gnm_eval.DegreeKS(), gnm_eval.ClusteringKS(),
        gnm_eval.EdgeLengthKS(dist_tensor), gnm_eval.BetweennessKS(),
    ])

    evaluators: Dict[str, object] = {
        "delta_con":                       DeltaConEvaluator(),
        "frobenius":                       FrobeniusEvaluator(),
        "portrait":                        PortraitDivergence(),
        "hamming":                         HammingEvaluator(),
        "f1":                              F1Evaluator(),
        "jaccard":                         JaccardEvaluator(),
        "spectral_distance_norm_laplacian": SpectralDistanceEvaluator(method="normalized_laplacian"),
        "spectral_distance_adjacency":     SpectralDistanceEvaluator(method="adjacency"),
        "communicability_corr":            CommunicabilityCorrEvaluator(),
        "communicability_jsd":             CommunicabilityJSDEvaluator(),
        "network_mutual_information":      NetworkMutualInformationEvaluator(),
        "dc_network_mutual_information":   DCNetworkMutualInformationEvaluator(),
        "net_simile":                      NetrdEvaluator(method="net_simile"),
        "resistance":                      NetrdEvaluator(method="resistance"),
        "netrd_non_backtracking_spectral": NetrdEvaluator(method="netrd_non_backtracking_spectral"),
        "energy":                          EnergyEvaluator([energy_criteria]),
    }
    if measure_subset:
        evaluators = {k: v for k, v in evaluators.items() if k in measure_subset}
    return evaluators


def load_grid_consensus(rc: RunCfg) -> np.ndarray:
    out = []
    for eta, gamma in rc.grid_combos:
        path = rc.grid_dir / cfg.param_dir_name(eta, gamma) / "consensus.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing grid consensus: {path}\nRun --stage generate first.")
        out.append(np.load(path))
    return np.stack(out)


def load_ground_truth(rc: RunCfg) -> pd.DataFrame:
    if not rc.gt_manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {rc.gt_manifest}\nRun --stage generate first.")
    return pd.read_csv(rc.gt_manifest)


def compare(rc: RunCfg) -> None:
    import torch
    from tqdm import tqdm

    rc.comparison_dir.mkdir(parents=True, exist_ok=True)
    print("Loading grid consensus networks ...")
    consensus_batch = load_grid_consensus(rc)
    tgt_t = torch.tensor(consensus_batch, dtype=torch.float32)
    print(f"  {len(consensus_batch)} grid consensus networks loaded.")

    gt = load_ground_truth(rc)
    print(f"  {len(gt)} ground-truth points loaded.\n")

    # Pre-load every ground-truth network once. We glob the directory rather than
    # rebuild the filename from the manifest: pandas' CSV float parser can land
    # 1 ULP off the originally-written float, giving a slightly different string
    # representation in the embedded eta/gamma. Files are named ..._id{NNN}.npy,
    # so sorted() yields net_id order.
    gt_nets: List[Tuple[int, int, float, float, np.ndarray]] = []
    for _, r in gt.iterrows():
        idx, eta, gamma = int(r["gt_idx"]), float(r["true_eta"]), float(r["true_gamma"])
        gdir = rc.gt_dir / cfg.gt_dir_name(idx)
        net_paths = sorted(gdir.glob("net_*.npy"))
        if not net_paths:
            raise FileNotFoundError(
                f"No ground-truth networks in {gdir} — run --stage generate first.")
        for net_id, path in enumerate(net_paths):
            gt_nets.append((idx, net_id, eta, gamma, np.load(path)))

    evaluators = build_evaluators(rc.measures or rc.default_measures)
    n_grid = len(rc.grid_combos)

    for measure_name, evaluator in evaluators.items():
        out_path = rc.comparison_dir / f"distances_{measure_name}.csv"
        if out_path.exists():
            print(f"[{measure_name}] exists — skipping.")
            continue
        print(f"\n{'='*60}\nRunning: {measure_name}\n{'='*60}")

        records, n_failed = [], 0
        for (idx, net_id, true_eta, true_gamma, net) in tqdm(gt_nets, desc=measure_name):
            true_eta_n, true_gamma_n = rc.normalise(true_eta, true_gamma)
            try:
                gen_t  = torch.tensor(net, dtype=torch.float32).unsqueeze(0)
                result = evaluator(gen_t, tgt_t)
                dists  = [float(result[i]) for i in range(n_grid)]
                pred_idx = int(np.nanargmin(dists))
                pred_eta, pred_gamma = rc.grid_combos[pred_idx]
                pred_eta_n, pred_gamma_n = rc.normalise(pred_eta, pred_gamma)
                abs_err = float(np.sqrt((pred_eta_n - true_eta_n) ** 2 +
                                        (pred_gamma_n - true_gamma_n) ** 2))
            except Exception:
                n_failed += 1
                dists = [float("nan")] * n_grid
                pred_idx = -1
                pred_eta = pred_gamma = abs_err = float("nan")

            row = {
                "gt_idx": idx, "network_id": net_id,
                "true_eta": true_eta, "true_gamma": true_gamma,
                "recovered_grid_idx": pred_idx,
                "recovered_eta": pred_eta, "recovered_gamma": pred_gamma,
                "abs_error": abs_err,
            }
            for i, d in enumerate(dists):
                row[f"dist_to_grid_{i}"] = d
            records.append(row)

        if n_failed:
            print(f"  Warning: {n_failed} network(s) failed (recorded as NaN).")
        df = pd.DataFrame(records)
        df.to_csv(out_path, index=False)
        r_eta   = df[["true_eta", "recovered_eta"]].corr().iloc[0, 1]
        r_gamma = df[["true_gamma", "recovered_gamma"]].corr().iloc[0, 1]
        print(f"  r(eta)={r_eta:.3f}  r(gamma)={r_gamma:.3f}  -> {out_path.name}")

    print("\nComparison complete.")


# ===========================================================================
# Stage 3 — plots
# ===========================================================================

# Display names (mirrors the wide-target plotting script).
METRIC_NAMES: Dict[str, str] = {
    "delta_con": "DeltaCon", "frobenius": "Frobenius", "portrait": "Portrait",
    "hamming": "Hamming", "f1": "F1", "jaccard": "Jaccard",
    "spectral_distance_norm_laplacian": "Spectral (Norm. Laplacian)",
    "spectral_distance_adjacency": "Spectral (Adjacency)",
    "communicability_corr": "Communicability Corr.",
    "communicability_jsd": "Communicability JSD",
    "network_mutual_information": "Network MI",
    "dc_network_mutual_information": "DC Network MI",
    "net_simile": "NetSimile", "resistance": "Resistance",
    "netrd_non_backtracking_spectral": "NBS Spectral", "energy": "Energy",
}

# Wide-target experiment (5 widely-spread combos) for the ranking comparison.
WIDE_COMPARISON_DIR = (ROOT_DIR / "output" / "gnm" /
                       "synthetic_parameter_recovery_grid" / "comparison_results")
WIDE_GRID_N_ETA = 10  # from gnm_grid_config


def _pearson(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return float("nan")
    return float(a[m].rank().corr(b[m].rank()))


def load_fine_results(rc: RunCfg) -> Dict[str, pd.DataFrame]:
    results = {}
    for p in sorted(rc.comparison_dir.glob("distances_*.csv")):
        results[p.stem.replace("distances_", "")] = pd.read_csv(p)
    return results


def fine_scores(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-measure Pearson/Spearman r for eta and gamma + combined score."""
    rows = []
    for measure, df in results.items():
        r_eta   = _pearson(df["true_eta"],   df["recovered_eta"])
        r_gamma = _pearson(df["true_gamma"], df["recovered_gamma"])
        rows.append({
            "measure": measure,
            "name": METRIC_NAMES.get(measure, measure),
            "r_eta": r_eta, "r_gamma": r_gamma,
            "r_mean": np.nanmean([r_eta, r_gamma]),
            "rho_eta":   _spearman(df["true_eta"],   df["recovered_eta"]),
            "rho_gamma": _spearman(df["true_gamma"], df["recovered_gamma"]),
            "mae_norm":  float(df["abs_error"].mean()),
        })
    return pd.DataFrame(rows).sort_values("r_mean", ascending=False).reset_index(drop=True)


def wide_ranking() -> pd.DataFrame | None:
    """Ranking from the wide-target experiment, by mean grid-step error (lower better)."""
    if not WIDE_COMPARISON_DIR.exists():
        return None
    rows = []
    for p in sorted(WIDE_COMPARISON_DIR.glob("distances_*.csv")):
        measure = p.stem.replace("distances_", "")
        df = pd.read_csv(p)
        if "predicted_grid_idx" not in df:
            continue
        valid = df[df["predicted_grid_idx"] >= 0].copy()
        # Grid-step error in index space (γ-outer, η-inner; 10×10).
        pg = valid["predicted_grid_idx"] // WIDE_GRID_N_ETA
        pe = valid["predicted_grid_idx"] %  WIDE_GRID_N_ETA
        # Wide grid axes (from gnm_grid_config).
        wide_eta   = np.linspace(-8.0, 3.0, WIDE_GRID_N_ETA)
        wide_gamma = np.linspace(-0.1, 1.0, WIDE_GRID_N_ETA)
        te = valid["true_eta"].apply(lambda e: int(np.argmin(np.abs(wide_eta   - e))))
        tg = valid["true_gamma"].apply(lambda g: int(np.argmin(np.abs(wide_gamma - g))))
        step = np.sqrt((pe.values - te.values) ** 2 + (pg.values - tg.values) ** 2)
        rows.append({"measure": measure, "wide_grid_steps": float(np.nanmean(step))})
    if not rows:
        return None
    w = pd.DataFrame(rows).sort_values("wide_grid_steps").reset_index(drop=True)
    w["wide_rank"] = np.arange(1, len(w) + 1)  # 1 = best (lowest error)
    return w


def _viz():
    """Return (viz, colors, ok) — gracefully degrade if vizman is unavailable."""
    try:
        import json
        from vizman import viz
        viz.set_visual_style(font_family="Arial")
        pkg = Path(viz.__file__).parent
        colors = {n: v for cat in json.loads((pkg / "colors.json").read_text()).values()
                  for n, v in cat.items()}
        return viz, colors, True
    except Exception:
        return None, {}, False


def plot(rc: RunCfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz, C, have_viz = _viz()
    def col(name, fallback): return C.get(name, fallback)
    GREEN = col("JUST_GREEN", "#2ca02c")
    BLUE  = col("LAKE_BLUE", "#1f77b4")
    RED   = col("LECKER_RED", "#d62728")
    GRAY  = col("GRAY", "#888888")
    DARK  = col("HALF_BLACK", "#333333")

    results = load_fine_results(rc)
    if not results:
        print(f"No CSVs in {rc.comparison_dir}. Run --stage compare first.")
        return
    rc.plot_dir.mkdir(parents=True, exist_ok=True)

    scores = fine_scores(results)
    scores.to_csv(rc.plot_dir / "fine_recovery_scores.csv", index=False)
    print("\n--- fine-grained accuracy (Pearson r) ---")
    for _, r in scores.iterrows():
        print(f"  {r['name']:30s}  r(eta)={r['r_eta']:.3f}  "
              f"r(gamma)={r['r_gamma']:.3f}  mean={r['r_mean']:.3f}")

    figsize = (lambda wh: viz.cm_to_inch(wh)) if have_viz else (lambda wh: (wh[0]/2.54, wh[1]/2.54))

    # ---- Figure 1: grouped bar of r per measure ----------------------------
    n = len(scores)
    fig, ax = plt.subplots(figsize=figsize((18, max(6, n * 0.55 + 2))))
    y = np.arange(n)[::-1]  # best at top
    h = 0.38
    ax.barh(y + h/2, scores["r_eta"],   height=h, color=BLUE,  label=r"$\eta$",
            edgecolor=DARK, linewidth=0.3)
    ax.barh(y - h/2, scores["r_gamma"], height=h, color=GREEN, label=r"$\gamma$",
            edgecolor=DARK, linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(scores["name"], fontsize=7)
    ax.set_xlabel("Pearson r  (true vs recovered)")
    ax.set_xlim(min(0, np.nanmin(scores[["r_eta", "r_gamma"]].values)) - 0.05, 1.0)
    ax.axvline(0, color=DARK, linewidth=0.6)
    for yi, (re, rg) in zip(y, zip(scores["r_eta"], scores["r_gamma"])):
        if not np.isnan(re): ax.text(re + 0.01, yi + h/2, f"{re:.2f}", va="center", fontsize=5.5, color=DARK)
        if not np.isnan(rg): ax.text(rg + 0.01, yi - h/2, f"{rg:.2f}", va="center", fontsize=5.5, color=DARK)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Fine-grained parameter recovery in the plausible window\n"
                 rf"($\eta\in{list(cfg.SAMPLE_ETA_RANGE)}$, $\gamma\in{list(cfg.SAMPLE_GAMMA_RANGE)}$, "
                 f"{rc.n_truth} ground-truth nets)", fontsize=9)
    if have_viz:
        try:
            import seaborn as sns; sns.despine(ax=ax, left=True)
        except Exception:
            pass
    fig.tight_layout()
    p1 = rc.plot_dir / "fine_recovery_r_bars.pdf"
    fig.savefig(p1, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved {p1.name}")

    # ---- Figure 2: true vs recovered scatter for top measures --------------
    top = scores.head(min(4, n))
    fig, axes = plt.subplots(2, len(top), figsize=figsize((4.5 * len(top), 9)),
                             squeeze=False)
    for j, (_, srow) in enumerate(top.iterrows()):
        df = results[srow["measure"]]
        for i, (param, lo_hi, color) in enumerate([
            ("eta",   rc.eta_range,   BLUE),
            ("gamma", rc.gamma_range, GREEN),
        ]):
            ax = axes[i][j]
            rng = np.random.default_rng(0)
            jit = rng.normal(0, (lo_hi[1] - lo_hi[0]) * 0.004, size=len(df))
            ax.scatter(df[f"true_{param}"], df[f"recovered_{param}"] + jit,
                       s=14, alpha=0.55, color=color, linewidths=0)
            ax.plot(lo_hi, lo_hi, color=GRAY, lw=0.8, ls="--", zorder=0)
            rr = _pearson(df[f"true_{param}"], df[f"recovered_{param}"])
            if i == 0:
                ax.set_title(f"{srow['name']}", fontsize=8)
            ax.text(0.04, 0.92, f"r={rr:.2f}", transform=ax.transAxes,
                    fontsize=7, va="top", color=DARK)
            if j == 0:
                ax.set_ylabel(rf"recovered $\{param}$")
            if i == 1:
                ax.set_xlabel(rf"true $\{param}$")
    fig.suptitle("True vs recovered (top measures by mean r)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p2 = rc.plot_dir / "fine_recovery_scatter_top.pdf"
    fig.savefig(p2, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p2.name}")

    # ---- Figure 3: fine vs wide ranking ------------------------------------
    wide = wide_ranking()
    if wide is None:
        print("Wide-target results not found — skipping ranking comparison.")
    else:
        fine = scores.copy()
        fine["fine_rank"] = np.arange(1, len(fine) + 1)  # 1 = best (highest r)
        merged = fine.merge(wide, on="measure", how="inner")
        if len(merged) < 2:
            print("Too few overlapping measures for ranking comparison — skipping.")
        else:
            merged.to_csv(rc.plot_dir / "fine_vs_wide_ranking.csv", index=False)
            fig, ax = plt.subplots(figsize=figsize((12, max(6, len(merged) * 0.5 + 1))))
            # Bump chart: wide rank (left) -> fine rank (right).
            for _, r in merged.iterrows():
                moved = r["fine_rank"] - r["wide_rank"]
                lc = GREEN if moved < 0 else (RED if moved > 0 else GRAY)
                ax.plot([0, 1], [r["wide_rank"], r["fine_rank"]], "-o",
                        color=lc, lw=1.4, markersize=4)
                ax.text(-0.03, r["wide_rank"], r["name"], ha="right", va="center", fontsize=6.5)
                ax.text(1.03, r["fine_rank"], r["name"], ha="left", va="center", fontsize=6.5)
            ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1])
            ax.set_xticklabels(["Wide-target\n(5 spread, grid-steps)",
                                "Fine window\n(mean Pearson r)"], fontsize=8)
            ax.invert_yaxis()  # rank 1 on top
            ax.set_ylabel("rank (1 = best)")
            ax.set_title("Measure ranking: wide-target vs human-plausible window", fontsize=9)
            for s in ("top", "right", "bottom"):
                ax.spines[s].set_visible(False)
            ax.tick_params(bottom=False)
            fig.tight_layout()
            p3 = rc.plot_dir / "fine_vs_wide_ranking.pdf"
            fig.savefig(p3, dpi=300, bbox_inches="tight"); plt.close(fig)
            print(f"Saved {p3.name}")

            tau = merged[["fine_rank", "wide_rank"]].corr(method="kendall").iloc[0, 1]
            print(f"\nKendall tau (fine vs wide ranking) = {tau:.3f}  "
                  f"(1=identical order, low/neg = ranking shifts)")

    print(f"\nAll plots -> {rc.plot_dir}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["generate", "compare", "plot", "all"],
                    default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end self-test in a separate output dir")
    ap.add_argument("--measures", default=None,
                    help="comma-separated subset of measure names")
    args = ap.parse_args()

    measures = args.measures.split(",") if args.measures else None
    rc = RunCfg(smoke=args.smoke, measures=measures)
    print(f"Output dir: {rc.out_dir}\n")

    if args.stage in ("generate", "all"):
        generate(rc)
    if args.stage in ("compare", "all"):
        compare(rc)
    if args.stage in ("plot", "all"):
        plot(rc)


if __name__ == "__main__":
    main()
