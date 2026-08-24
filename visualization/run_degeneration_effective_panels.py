"""
Panels D and F of the precision/robustness figure, plotted against the
EFFECTIVE degeneration instead of the attempted rewiring fraction.

The published panels used `rewire_fraction` (the fraction of edges we *tried*
to swap) on the horizontal axis, while caption and Methods describe the
effective degeneration - the Hamming distance |A - A'|, i.e. the edges that
actually differ. `double_edge_swap` rejects swaps and later swaps can move an
already-moved edge, so the two are not the same: 25% attempted swaps leave
only ~48% of the maximal edge difference.

Both panels are built from one construction, so the MAE in F is exactly the
deviation from the diagonal drawn in D:

    per trajectory  -> min-max normalise the measure value and the Hamming
                       distance to [0, 1]
    per step        -> average over the 200 trajectories
    MAE             -> mean |y(k) - x(k)| over the 101 steps

Writes fig_panel_D_effective.pdf and fig_panel_F_effective.pdf (6 x 6 cm each)
next to the other chaos-analysis outputs, for assembly in the vector editor.

Run:  conda activate ma_thesis && python run_degeneration_effective_panels.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vizman import viz

viz.set_visual_style()

DATA = (Path(__file__).resolve().parents[1]
        / "output/gnm/hcp_schaefer_100_dataset/chaos_analysis")
OUT = DATA

MEASURES = ["energy", "portrait", "spectral_distance_adjacency", "communicability_corr",
            "net_simile", "netrd_non_backtracking_spectral", "delta_con", "frobenius"]

# Measures stored as a similarity rather than a distance.
INVERT = {"f1", "jaccard", "communicability_jsd", "communicability_corr"}

COLORS = {
    "communicability_corr":            (0.600, 0.600, 0.600),
    "netrd_non_backtracking_spectral": (0.216, 0.494, 0.722),
    "delta_con":                       (0.400, 0.400, 0.400),
    "frobenius":                       (0.902, 0.671, 0.008),
    "net_simile":                      (0.400, 0.651, 0.118),
    "spectral_distance_adjacency":     (0.906, 0.161, 0.541),
    "energy":                          (0.851, 0.373, 0.008),
    "portrait":                        (0.106, 0.620, 0.467),
}

# Nominal-axis MAEs of the published panel F, used as a regression check that
# this pipeline still reproduces the figure it replaces.
PUBLISHED_NOMINAL_MAE = {
    "frobenius": 0.285, "delta_con": 0.288, "netrd_non_backtracking_spectral": 0.279,
    "spectral_distance_adjacency": 0.316, "communicability_corr": 0.125,
    "portrait": 0.316, "net_simile": 0.272, "energy": 0.335,
}


def _norm_per_trajectory(df, col):
    return df.groupby("process_id")[col].transform(lambda s: (s - s.min()) / (s.max() - s.min()))


def load_curves():
    """Return (x_effective, {measure: y}, {measure: (mae_eff, mae_nom, time_s)})."""
    ham = pd.read_csv(DATA / "chaos_analysis_hamming.csv")
    ham["x"] = _norm_per_trajectory(ham, "metric_value")
    x = ham.groupby("step")["x"].mean()

    curves, scores = {}, {}
    for m in MEASURES:
        df = pd.read_csv(DATA / f"chaos_analysis_{m}.csv")
        v = (df.metric_value - df.metric_value.min()) / (df.metric_value.max() - df.metric_value.min())
        df["v"] = 1 - v if m in INVERT else v
        df["y"] = _norm_per_trajectory(df, "v")
        y = df.groupby("step")["y"].mean()
        nom = df.groupby("step")["rewire_fraction"].mean()
        nom = nom / nom.max()
        curves[m] = y
        scores[m] = (float(np.abs(y - x).mean()),
                     float(np.abs(y - nom).mean()),
                     float(df.computation_time.median()))
    return x, curves, scores


def check(scores):
    """Fails if the loading/normalisation drifts away from the published panel."""
    for m, (_, mae_nom, _) in scores.items():
        assert abs(mae_nom - PUBLISHED_NOMINAL_MAE[m]) < 0.002, (
            f"{m}: nominal MAE {mae_nom:.4f} no longer reproduces the published "
            f"{PUBLISHED_NOMINAL_MAE[m]:.3f}")
    print("check passed: nominal-axis MAEs still reproduce the published panel F")


def panel_d(x, curves, out_pdf):
    # Axes box sized to the panel it replaces in the assembled figure
    # (201.6 x 121.6 pt = 7.11 x 4.29 cm), so it drops in at scale 1.
    fig = plt.figure(figsize=viz.cm_to_inch((9.2, 6.0)))
    ax = fig.add_axes([0.19, 0.20, 7.112 / 9.2, 4.288 / 6.0])
    ax.plot([0, 100], [0, 1], ls="--", c="0.75", lw=1, zorder=1)
    for m in MEASURES:
        ax.plot(x.values * 100, curves[m].values, c=COLORS[m], lw=1.2, zorder=2)
    ax.set_xlabel("Effective Degeneration (%)")
    ax.set_ylabel("Normalized Distance")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, ls=":", c="0.9", lw=0.5)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(out_pdf)


def panel_f(scores, out_pdf):
    # Axes box sized to the panel it replaces (116 x 116 pt = 4.09 cm square).
    fig = plt.figure(figsize=viz.cm_to_inch((6.0, 6.0)))
    ax = fig.add_axes([0.24, 0.20, 4.093 / 6.0, 4.093 / 6.0])
    for m in MEASURES:
        mae_eff, _, t = scores[m]
        ax.scatter(t, mae_eff, c=[COLORS[m]], s=30, edgecolors="black", linewidths=0.8, zorder=2)
    ax.set_xlabel("Computation Time")
    ax.set_ylabel("MAE\n(lower is better)")
    ax.set_xlim(-0.004, 0.08)
    ax.set_ylim(0, 0.2)
    ax.set_xticks([0.0, 0.04, 0.08])
    ax.set_yticks([0.0, 0.1, 0.2])
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(out_pdf)


def main():
    x, curves, scores = load_curves()
    check(scores)
    print(f"{'measure':34s} {'MAE_eff':>8} {'MAE_nom':>8} {'time_ms':>8}")
    for m in sorted(MEASURES, key=lambda k: scores[k][0]):
        e, n, t = scores[m]
        print(f"{m:34s} {e:8.4f} {n:8.4f} {t * 1000:8.3f}")
    panel_d(x, curves, OUT / "fig_panel_D_effective.pdf")
    panel_f(scores, OUT / "fig_panel_F_effective.pdf")


if __name__ == "__main__":
    main()
