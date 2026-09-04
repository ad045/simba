"""
Panel H of the precision/robustness figure: intrinsic signal-to-noise ratio.

The published panel compared a MIN-MAX NORMALISED landscape against
UN-normalised replicate standard deviations, i.e.

    10 log10( var(norm01(M)) / mean(sigma_raw^2) ),

so the numerator was scale-free while the denominator carried the measure's own
units. The ranking it produced therefore tracked how large a measure's raw
numbers happen to be, not its signal-to-noise behaviour: every bounded measure
(energy, portrait, communicability correlation, non-backtracking) came out
positive and every unbounded one (Frobenius, DeltaCon, spectral (adjacency),
NetSimile) came out negative.

This recomputes the definition given in the Methods, with numerator and
denominator in the same units,

    iSNR = 10 log10( var(M) / mean_cells(sigma^2) ),

which is scale-invariant: multiplying a measure by a constant leaves it
unchanged. That invariance is the check at the bottom of this file.

Run:  conda activate ma_thesis && python run_isnr_consistent_panel.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vizman import viz

viz.set_visual_style()

MORPHO = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/"
              "hcp_schaefer_100_dataset/106_distance_metrics_mst_animal_0_density10")
EXP = "106_distance_metrics_mst_animal_0_density10"
OUT = MORPHO

MEASURES = ["energy", "portrait", "spectral_distance_adjacency", "communicability_corr",
            "net_simile", "netrd_non_backtracking_spectral", "delta_con", "frobenius"]
INVERT = {"communicability_corr"}

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


def cell_stats(measure):
    df = pd.read_csv(MORPHO / f"summary_indiv_{measure}_for_exp_{EXP}.csv")
    col = [c for c in df.columns
           if c not in ("eta", "gamma", "network_index", "filename", "id")][0]
    v = df[col].astype(float)
    if measure in INVERT:                       # similarity -> distance
        v = 1 - (v - v.min()) / (v.max() - v.min())
    g = df.assign(val=v).groupby(["eta", "gamma"])["val"].agg(["mean", "std"])
    return g["mean"].to_numpy(), g["std"].to_numpy()


def isnr(mean, std):
    return float(10 * np.log10(np.var(mean) / np.mean(std ** 2)))


def timing(measure):
    td = pd.read_csv(MORPHO / f"timing_{measure}_for_exp_{EXP}.csv")
    cols = [c for c in td.columns if c.startswith("time_")]
    return float(np.mean(td[cols].to_numpy()))


def check():
    """iSNR must not move when a measure is rescaled - that was the bug."""
    mean, std = cell_stats("frobenius")
    base = isnr(mean, std)
    for factor in (1e-3, 7.0, 250.0):
        assert abs(isnr(mean * factor, std * factor) - base) < 1e-9, "iSNR is not scale-invariant"
    print("check passed: iSNR invariant under rescaling of the measure")


def main():
    check()
    vals = {m: isnr(*cell_stats(m)) for m in MEASURES}
    times = {m: timing(m) for m in MEASURES}
    print(f"{'measure':34s} {'iSNR':>8} {'time_s':>9}")
    for m in sorted(MEASURES, key=lambda k: -vals[k]):
        print(f"{m:34s} {vals[m]:8.2f} {times[m]:9.5f}")

    # Axes box sized to the panel it replaces (116 x 116 pt = 4.09 cm square).
    fig = plt.figure(figsize=viz.cm_to_inch((6.0, 6.0)))
    ax = fig.add_axes([0.24, 0.20, 4.093 / 6.0, 4.093 / 6.0])
    ax.axhline(0, ls="--", c="0.75", lw=1, zorder=1)
    for m in MEASURES:
        ax.scatter(times[m], vals[m], c=[COLORS[m]], s=30,
                   edgecolors="black", linewidths=0.8, zorder=2)
    ax.set_xlabel("Computation Time")
    ax.set_ylabel("Intrinsic SNR\n(higher is better)")
    ax.set_xlim(-0.004, 0.08)
    ax.set_ylim(-5, 20)
    ax.set_xticks([0.0, 0.04, 0.08])
    ax.set_yticks([0, 10, 20])
    out = OUT / "fig_panel_H_isnr_consistent.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
