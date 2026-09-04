"""
Appendix figure for the wide-target recovery with uniform ground truth
======================================================================

One subpanel per measure: the 100 uniformly drawn ground-truth combinations
(x) joined to the grid cell each measure recovered (dot), the line coloured by
that network's recovery error in grid steps. Replaces the old six-target
scatter + per-combination variability figure, whose panel B needed the 10
repeat networks per combination that uniform sampling no longer has.

    conda activate ma_thesis
    python experiment_parameter_recovery_fine/build_wide_uniform_scatter.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "experiment_parameter_recovery_fine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments_config import METHOD_NAMES, coarse as wide_cfg  # noqa: E402
import run_recovery_main_figures as rmf  # noqa: E402

CD = ROOT / "output" / "gnm" / "synthetic_parameter_recovery_wide_uniform" / "comparison_results"
OUT = Path("/Users/adrian/Desktop/Benchmarking_/figures/appendix/fig_wide_uniform_recovery.pdf")

LOW, HIGH = "#3EA5C4", "#FE961F"


def main(out: Path = OUT) -> None:
    rmf.WIDE_COMPARISON_DIR = CD
    scores = rmf.wide_scores()                     # ordered by grid steps
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("err", [LOW, HIGH])
    norm = matplotlib.colors.Normalize(0, 12.73)

    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.6), sharex=True, sharey=True)
    for ax, (_, row) in zip(axes.ravel(), scores.iterrows()):
        df = pd.read_csv(CD / f"distances_{row['measure']}.csv")
        df = df[df["recovered_grid_idx"] >= 0]
        te = df["true_eta"].to_numpy(); tg = df["true_gamma"].to_numpy()
        re = df["recovered_eta"].to_numpy(); rg = df["recovered_gamma"].to_numpy()
        def _idx(axis, vals):
            return np.array([int(np.argmin(np.abs(axis - v))) for v in vals])
        ie = np.abs(_idx(wide_cfg.GRID_ETA, re) - _idx(wide_cfg.GRID_ETA, te))
        ig = np.abs(_idx(wide_cfg.GRID_GAMMA, rg) - _idx(wide_cfg.GRID_GAMMA, tg))
        err = np.sqrt(ie ** 2 + ig ** 2)

        segs = [[(a, b), (c, d)] for a, b, c, d in zip(te, tg, re, rg)]
        ax.add_collection(LineCollection(segs, colors=cmap(norm(err)), lw=0.4,
                                         alpha=0.75, zorder=1))
        ax.scatter(te, tg, marker="x", s=5, c="#232324", lw=0.5, zorder=2)
        ax.scatter(re, rg, s=4, c="#3EA5C4", lw=0, alpha=0.8, zorder=3)
        ax.set_title(f"{row['name']}  ({row['grid_steps']:.2f})", fontsize=6.5, pad=2)
        ax.tick_params(labelsize=5.5, length=2, width=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\eta$", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\gamma$", fontsize=7)

    fig.legend(handles=[
        Line2D([0], [0], marker="x", color="#232324", lw=0, markersize=4,
               label="true combination"),
        Line2D([0], [0], marker="o", color="#3EA5C4", lw=0, markersize=3,
               label="recovered cell"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=6.5,
        bbox_to_anchor=(0.5, -0.005))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("recovery error (grid steps)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
