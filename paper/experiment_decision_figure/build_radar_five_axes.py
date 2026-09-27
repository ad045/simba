"""
Radar plots for the overview figure, one per selected measure, on the five
evaluation axes the Results are structured around.

Axis -> scalar (all oriented so that a larger radius is the more desirable value):

  agreement      mean Pearson correlation of the measure's landscape with the
                 landscapes of the other seven selected measures
  plausibility   total variation of the locations of its 20 best-fitting
                 networks (inverted); set PLAUSIBILITY = "ks" for the
                 distributional variant instead
  efficiency     runtime per comparison (inverted)
  sensitivity    mean rank over the three perturbation read-outs - the MAE of
                 the rewiring response against the diagonal, the coefficient of
                 variation across the 200 rewiring trajectories, and the noise
                 tolerance N* of the recovered parameters; set
                 SENSITIVITY = "isnr" for the landscape signal-to-noise ratio
  accuracy       grid-step error inside the plausible window. Scored apart from
                 the other axes: a measure at or beyond the chance level of the
                 window scores 0, and the measures below chance are spread
                 linearly between 0 at chance and 8 at the best error.

Values are turned into ordinal scores by rank across the eight measures:
best = 8, worst = 1, one step per rank, ties averaged. The scores are relative
to this set of eight; they preserve order, not the size of the differences.

Run:  conda activate ma_thesis && python build_radar_five_axes.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, rankdata
from vizman import viz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_config import MANUSCRIPT_FIGURES  # noqa: E402
from experiments_config import (ROOT_DIR, MORPHO_DIR, MORPHO_EXP, TIMING_DIR, TIMING_EXP, CONSENSUS_PATH,
                                DIST_MATRIX_PATH, fine, to_distance)

viz.set_visual_style()

OUT = MANUSCRIPT_FIGURES / "radar"
NET_DIR = MORPHO_DIR / "generated_networks"
FINE_DIR = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_fine" / "comparison_results"

MEASURES = ["frobenius", "delta_con", "netrd_non_backtracking_spectral",
            "spectral_distance_adjacency", "communicability_corr", "portrait",
            "net_simile", "energy"]

NAMES = {
    "frobenius": "Frobenius", "delta_con": "DeltaCon",
    "netrd_non_backtracking_spectral": "Spectral\n(non-backtracking)",
    "spectral_distance_adjacency": "Spectral\n(adjacency)",
    "communicability_corr": "Communicability\ncorrelation",
    "portrait": "Portrait\ndivergence", "net_simile": "NetSimile",
    "energy": "KS-based energy",
}

# iSNR of the landscape, from the notebook that also draws
# Figure 4H: visualization/9_between_and_withhin_variance.ipynb
COLORS = {
    "frobenius":                       (0.902, 0.671, 0.008),
    "delta_con":                       (0.400, 0.400, 0.400),
    "netrd_non_backtracking_spectral": (0.216, 0.494, 0.722),
    "spectral_distance_adjacency":     (0.906, 0.161, 0.541),
    "communicability_corr":            (0.600, 0.600, 0.600),
    "portrait":                        (0.106, 0.620, 0.467),
    "net_simile":                      (0.400, 0.651, 0.118),
    "energy":                          (0.851, 0.373, 0.008),
}

# iSNR of the landscape, scale-invariant definition, from
# visualization/run_isnr_consistent_panel.py (which also draws Figure 4H)
ISNR = {"energy": 15.30, "frobenius": 15.08, "portrait": 9.88,
        "netrd_non_backtracking_spectral": 8.02, "spectral_distance_adjacency": 7.72,
        "net_simile": 3.26, "delta_con": 1.34, "communicability_corr": -2.34}

N_BEST = 20
ETA_RANGE, GAMMA_RANGE = (-8.0, 3.0), (-0.1, 1.0)      # morphospace extent

# Grid-step error of a measure recovering a uniformly random cell of the
# window's 10x10 grid (see Methods, "Chance level").
WINDOW_CHANCE = 4.53

# Read-out used on the plausibility spoke: "total_variation" or "ks".
PLAUSIBILITY = "total_variation"

# Read-out on the sensitivity-and-robustness spoke: "perturbation" or "isnr".
SENSITIVITY = "perturbation"

# The three perturbation read-outs, all reported in the Results.
# MAE of the rewiring response against the diagonal (lower better), from
# visualization/run_degeneration_effective_panels.py:
# MAE and CV below come from output/gnm/<dataset>/chaos_analysis/, which rewires the
# CONSENSUS and compares it to itself. The consensus is unchanged by the 10%-density
# regeneration, so both are unchanged (verified against the published values).
REWIRING_MAE = {"communicability_corr": 0.076, "frobenius": 0.123, "delta_con": 0.126,
                "netrd_non_backtracking_spectral": 0.129, "net_simile": 0.139,
                "spectral_distance_adjacency": 0.161, "portrait": 0.164, "energy": 0.174}
# CV across the 200 rewiring trajectories (lower better), same notebook as the iSNR:
REWIRING_CV = {"frobenius": 0.0130, "delta_con": 0.0161, "energy": 0.0422,
               "portrait": 0.0549, "netrd_non_backtracking_spectral": 0.0674,
               "spectral_distance_adjacency": 0.0849, "net_simile": 0.0996,
               "communicability_corr": 0.1536}
# Noise tolerance N*: largest effective degeneration of the reference (fraction
# of its 495 edges that actually differ) at which the median drift of the
# best-fitting combination is still within one grid step (higher better). Read
# from the rewiring-robustness run rather than pinned - the recovery step is an
# argmin over the morphospace landscape, so this moves whenever the landscapes
# are re-scored, and a stale copy would silently mis-rank the robustness spoke.
def _noise_tolerance() -> dict:
    path = ROOT_DIR / "output" / "rewiring_robustness" / "rewiring_robustness_results.csv"
    assert path.exists(), f"missing {path} - run experiment_rewiring_robustness first"
    r = pd.read_csv(path).set_index("measure")["noise_tolerance_frac"]
    return {m: float(r[m]) for m in MEASURES}

# Bar heights of Figure 4A, reproduced to the reading precision of that panel.
# Note the selection: the 20 best-fitting *parameter combinations* (cell means
# over the 10 replicates), not the 20 best-fitting individual networks - the
# latter gives a different answer (e.g. portrait 0.026 against 0.037).
PUBLISHED_TOTAL_VARIATION = {"delta_con": 0.0033, "energy": 0.0043,
                             "net_simile": 0.0096, "portrait": 0.0139,
                             "spectral_distance_adjacency": 0.0230,
                             "netrd_non_backtracking_spectral": 0.0493,
                             "frobenius": 0.0778, "communicability_corr": 0.1091}
# Order sets the layout: first spoke points up, the rest follow counter-clockwise.
# accuracy top, plausibility upper left, agreement lower left,
# sensitivity and robustness lower right, computational efficiency upper right.
AXES = ["Accuracy", "Biological\nplausibility", "Agreement",
        "Sensitivity and\nrobustness", "Computational\nefficiency"]

# What each spoke actually carries, for the annotated key.
LONG_LABELS = {
    "Accuracy":
        "Accuracy\nmean grid-step error when recovering\n100 known $(\\eta, \\gamma)$ combinations drawn from\n"
        "the plausible window, on a 10\u00d710 grid\n(0 at or beyond chance, else linear to the best)",
    "Biological\nplausibility":
        "Biological plausibility\ntotal variation of the 20 best-fitting\nparameter combinations,\n"
        "$\\mathrm{var}(\\eta) + \\mathrm{var}(\\gamma)$ over the morphospace extent\n(lower is better)",
    "Agreement":
        "Agreement\nmean Pearson correlation of the measure's\ndistance landscape with the landscapes\n"
        "of the other seven selected measures\n(higher is better)",
    "Sensitivity and\nrobustness":
        "Sensitivity and robustness\nmean rank over three perturbation read-outs:\nlinearity of the rewiring response (MAE),\n"
        "stability across the 200 trajectories (CV),\nand noise tolerance of the recovered\nparameters ($N^*$)\n(higher is better)",
    "Computational\nefficiency":
        "Computational efficiency\nmean runtime of one network comparison,\nover the 25,000 morphospace networks\n"
        "(lower is better)",
}


def landscape_distance(measure):
    """Per-network distance, oriented so that lower is closer."""
    df = pd.read_csv(MORPHO_DIR / f"summary_indiv_{measure}_for_exp_{MORPHO_EXP}.csv")
    value_col = [c for c in df.columns
                 if c not in ("network_index", "filename", "eta", "gamma", "id")][0]
    df["distance"] = to_distance(df[value_col].to_numpy(), measure)
    return df


def agreement():
    """Mean Pearson correlation of a measure's landscape with the other seven.

    The correlation matrix is built here rather than read from
    method_evaluation/correlation_matrix.csv: that file is written by a figure
    script whose loader flips the similarity-valued measures twice (to_distance
    negates them, then a 1 - minmax step flips them back), so communicability
    correlation enters it as a similarity and its correlations carry the wrong
    sign. Building it from to_distance-oriented values keeps this spoke
    consistent with Figure 2B and with the orientation stated in the Methods.
    """
    cols = {m: landscape_distance(m)
            .sort_values("network_index")["distance"].to_numpy()
            for m in MEASURES}
    corr = pd.DataFrame(cols).corr()
    return {m: float((corr.loc[m].sum() - 1.0) / (len(MEASURES) - 1)) for m in MEASURES}


def plausibility_ks():
    consensus = np.squeeze(np.load(CONSENSUS_PATH))
    dist = np.load(DIST_MATRIX_PATH)
    iu = np.triu_indices_from(consensus, k=1)
    emp_degree = consensus.sum(0)
    emp_length = dist[iu][consensus[iu] > 0]

    out = {}
    for m in MEASURES:
        df = landscape_distance(m).nsmallest(N_BEST, "distance")
        deg, length = [], []
        for fn in df.filename:
            a = (np.squeeze(np.load(NET_DIR / fn)) > 0).astype(int)
            deg.append(a.sum(0))
            length.append(dist[iu][a[iu] > 0])
        ks_deg = ks_2samp(np.concatenate(deg), emp_degree).statistic
        ks_len = ks_2samp(np.concatenate(length), emp_length).statistic
        out[m] = 1.0 - 0.5 * (ks_deg + ks_len)
    return out


def total_variation():
    """Var(eta) + var(gamma) on the normalised parameter ranges, over the 20
    best-fitting parameter combinations. Reproduces the bars of Figure 4A."""
    out = {}
    for m in MEASURES:
        df = landscape_distance(m)
        cell = df.groupby(["eta", "gamma"])["distance"].mean().nsmallest(N_BEST).reset_index()
        e = (cell.eta.to_numpy() - ETA_RANGE[0]) / (ETA_RANGE[1] - ETA_RANGE[0])
        g = (cell.gamma.to_numpy() - GAMMA_RANGE[0]) / (GAMMA_RANGE[1] - GAMMA_RANGE[0])
        out[m] = float(e.var() + g.var())
    for m, published in PUBLISHED_TOTAL_VARIATION.items():
        assert abs(out[m] - published) < 0.006, (
            f"{m}: total variation {out[m]:.4f} no longer matches Figure 4A ({published})")
    return out


def efficiency():
    out = {}
    for m in MEASURES:
        # runtime comes from the reference timing run, see experiments_config.TIMING_EXP
        paths = sorted(TIMING_DIR.glob(f"timing_{m}_for_exp_{TIMING_EXP}.csv"))
        td = pd.read_csv(paths[0])
        cols = [c for c in td.columns if c.startswith("time_")]
        out[m] = float(np.mean(td[cols].to_numpy())) * 1000        # ms
    return out


def accuracy():
    out = {}
    for m in MEASURES:
        df = pd.read_csv(FINE_DIR / f"distances_{m}.csv")
        idx = df.recovered_grid_idx.to_numpy()
        v = idx >= 0
        pg, pe = idx[v] // fine.GRID_N_ETA, idx[v] % fine.GRID_N_ETA
        te = np.array([int(np.argmin(np.abs(fine.GRID_ETA - e))) for e in df.true_eta[v]])
        tg = np.array([int(np.argmin(np.abs(fine.GRID_GAMMA - g))) for g in df.true_gamma[v]])
        out[m] = float(np.mean(np.hypot(pe - te, pg - tg)))
    return out


def perturbation_composite():
    """Mean of the three perturbation ranks, higher = more robust."""
    parts = [to_scores(REWIRING_MAE, False), to_scores(REWIRING_CV, False),
             to_scores(_noise_tolerance(), True)]
    return {m: float(np.mean([p[m] for p in parts])) for m in MEASURES}


def accuracy_scores(errors):
    """0 at or beyond chance, otherwise linear from 0 at chance to 8 at the best."""
    best = min(errors.values())
    assert best < WINDOW_CHANCE, "no measure beats chance - the scale would collapse"
    return {m: (0.0 if e >= WINDOW_CHANCE
                else 8.0 * (WINDOW_CHANCE - e) / (WINDOW_CHANCE - best))
            for m, e in errors.items()}


def to_scores(values, higher_is_better):
    """Rank across the eight measures -> 8 (best) ... 1 (worst), one step per rank."""
    v = np.array([values[m] for m in MEASURES], dtype=float)
    r = rankdata(-v if higher_is_better else v)          # 1 = best
    return dict(zip(MEASURES, len(MEASURES) + 1.0 - r))


def radar(ax, scores, title=None, labels=None, color="#3F6FA8"):
    ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False) + np.pi / 2
    ax.set_theta_zero_location("E")
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    fine_grid = np.linspace(0, 2 * np.pi, 200)
    for r in (2, 4, 6, 8):
        ax.plot(fine_grid, np.full_like(fine_grid, r), c="0.55", lw=0.5, zorder=1)
    for a in ang:
        ax.plot([a, a], [0, 8.9], c="0.55", lw=0.5, zorder=1)
    ax.scatter([0], [0], c="0.55", s=4, zorder=2)
    v = np.array([scores[a] for a in AXES])
    closed_a, closed_v = np.r_[ang, ang[:1]], np.r_[v, v[:1]]
    ax.plot(closed_a, closed_v, c=color, lw=2.2, zorder=3)
    ax.fill(closed_a, closed_v, color=color, alpha=0.25, zorder=3)
    ax.scatter(ang, v, c=[color], s=16, zorder=4)
    if title:
        ax.set_title(title, pad=14)
    if labels == "short":
        for a, lab in zip(ang, AXES):
            ax.text(a, 10.4, lab, ha="center", va="center", fontsize=6, color="0.25")
    elif labels == "long":
        for a, key in zip(ang, AXES):
            x = np.cos(a)
            ha = "center" if abs(x) < 0.2 else ("left" if x > 0 else "right")
            ax.text(a, 9.6, LONG_LABELS[key], ha=ha, va="center",
                    fontsize=4.6, color="0.25", linespacing=1.45)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {
        "Agreement": (agreement(), True),
        "Biological\nplausibility": ((total_variation(), False) if PLAUSIBILITY == "total_variation"
                                     else (plausibility_ks(), True)),
        "Computational\nefficiency": (efficiency(), False),
        "Sensitivity and\nrobustness": ((perturbation_composite(), True)
                                      if SENSITIVITY == "perturbation" else (ISNR, True)),
        "Accuracy": (accuracy(), False),
    }
    scores = {ax: (accuracy_scores(vals) if ax == "Accuracy" else to_scores(vals, hib))
              for ax, (vals, hib) in raw.items()}

    print(f"{'measure':34s} " + " ".join(f"{a.replace(chr(10),' '):>28s}" for a in AXES))
    for m in MEASURES:
        cells = [f"{raw[a][0][m]:10.3f} -> {scores[a][m]:4.2f}" for a in AXES]
        print(f"{m:34s} " + " ".join(f"{c:>28s}" for c in cells))

    for m in MEASURES:
        fig = plt.figure(figsize=viz.cm_to_inch((3.6, 3.6)))
        ax = fig.add_subplot(111, projection="polar")
        radar(ax, {a: scores[a][m] for a in AXES}, title=NAMES[m], color=COLORS[m])
        fig.savefig(OUT / f"radar_{m}.pdf", bbox_inches="tight", transparent=True)
        plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=viz.cm_to_inch((16, 11.5)),
                            subplot_kw={"projection": "polar"})
    for ax, m in zip(axs.ravel(), MEASURES):
        radar(ax, {a: scores[a][m] for a in AXES}, title=NAMES[m], color=COLORS[m])
    fig.tight_layout(h_pad=4.0, w_pad=1.5)
    fig.savefig(OUT / "radar_all.pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)

    for tag, mode, size in (("radar_axis_key", "short", 6.0),
                            ("radar_axis_key_long_descriptions", "long", 13.0)):
        fig = plt.figure(figsize=viz.cm_to_inch((size, size)))
        ax = fig.add_subplot(111, projection="polar")
        radar(ax, {a: 8 for a in AXES}, labels=mode, color="0.55")
        if mode == "long":
            ax.text(0.5, -0.16, "Each spoke is the measure's rank among the eight, 8 for the most\n"
                    "desirable value down to 1 for the least. On accuracy instead, a measure\n"
                    "at or beyond chance scores 0 and the rest run linearly up to 8.",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=5.5, color="0.25", linespacing=1.45)
        fig.savefig(OUT / f"{tag}.pdf", bbox_inches="tight", transparent=True)
        plt.close(fig)
    print("written to", OUT)


if __name__ == "__main__":
    main()
