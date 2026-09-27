"""
One-figure version of the topographic-plausibility result (18 x 9 cm)
====================================================================

`run_topographic_plausibility.py` writes three stacked panels on a portrait
page. The manuscript needs one landscape-format figure, so this script re-plots
the same cached results side by side, in the paper's palette (vizman colours -
bone-white to half-black for the landscape, deep-blue/bone-white/lecker-red for
the signed map correlations; no default matplotlib ramps).

Reads only what the experiment already wrote to
`output/topographic_plausibility/`; it generates nothing and computes nothing
beyond the pivot into grid shape.

    python plot_topographic_compact.py [--out FILE.pdf] [--selfcheck]

    A  topographic-plausibility landscape over the (eta, gamma) grid, with each
       measure's best fit and the topographic optimum
    B  topographic agreement at each measure's best fit against the ceiling any
       GNM in the grid reaches
    C  which of the five node-level maps each best fit reproduces
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments_config import METRIC_COLORS, ROOT_DIR as CFG_ROOT  # noqa: E402

OUT_DIR = CFG_ROOT / "output" / "topographic_plausibility"

# the five node-level maps, in the order they are plotted
MAPS = ["gradient", "degree", "clustering", "betweenness", "edge_length"]
MAP_LABELS = ["principal\ngradient", "degree", "clustering", "betweenness",
              "connection\nlength"]

FIG_CM = (20, 6)


def _style():
    """vizman house style, tightened for three panels on 18 cm."""
    from vizman import viz

    viz.set_visual_style(font_family="Arial")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.8,
    })
    return viz, viz.give_colormaps(), viz.load_data_from_json("colors.json")


def load(out_dir: Path):
    land = pd.read_csv(out_dir / "topographic_landscape.csv")
    res = pd.read_csv(out_dir / "topographic_results.csv")
    meta = json.loads((out_dir / "topographic_meta.json").read_text())
    return land, res.sort_values("r_topo", ascending=False).reset_index(drop=True), meta


def plot(land: pd.DataFrame, res: pd.DataFrame, meta: dict, cmaps: dict,
         colors: dict, out_pdf: Path) -> None:
    import matplotlib.pyplot as plt
    from vizman import viz

    bone = colors["neutrals"]["BONE_WHITE"]
    black = colors["neutrals"]["HALF_BLACK"]
    gray = colors["neutrals"]["GRAY"]
    red = colors["warms"]["LECKER_RED"]

    fig = plt.figure(figsize=viz.cm_to_inch(FIG_CM), constrained_layout=True)
    # the narrow second column is the measure colour key: B and C carry no
    # measure names, so the strip is what identifies a row (and the dots in A)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.30, 0.05, 0.95, 1.05], wspace=0.04)
    ax_a, ax_key, ax_b, ax_c = (fig.add_subplot(g) for g in gs)

    # B and C share one row grid: rows 0..n-1 are the measures, row n is the
    # "best GNM anywhere" ceiling, which exists in C only. Keeping the empty
    # slot in B is what makes the colour key line up with both panels.
    rows = np.arange(len(res))
    n_slots = len(res) + 1
    row_colors = [METRIC_COLORS.get(m, gray) for m in res["measure"]]

    # -- A: the landscape -------------------------------------------------
    etas = np.sort(land["eta"].unique())
    gammas = np.sort(land["gamma"].unique())
    grid = (land.pivot(index="gamma", columns="eta", values="combined")
                .reindex(index=gammas, columns=etas).to_numpy())

    im = ax_a.pcolormesh(etas, gammas, grid, cmap=cmaps["bw_hb"], shading="auto",
                         rasterized=True)
    cb = fig.colorbar(im, ax=ax_a, pad=0.02, fraction=0.05, aspect=13)
    cb.set_label("topographic agreement\n(mean $r$ over the 5 maps)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, width=0.6, length=1.5)
    cb.outline.set_linewidth(0.6)

    ax_a.axvline(0.0, color=red, lw=0.8, ls=":")
    ax_a.scatter([meta["eta_top"]], [meta["gamma_top"]], marker="*", s=70,
                 color=bone, edgecolor=black, linewidths=0.6, zorder=6,
                 label="topographic optimum")
    for _, r in res.iterrows():
        # several measure colours are greys, so the marker gets a dark ring
        # rather than a light one to stay visible on the grey landscape
        ax_a.scatter([r["eta"]], [r["gamma"]], s=26,
                     color=METRIC_COLORS.get(r["measure"], gray),
                     edgecolor=black, linewidths=0.6, zorder=5)
    ax_a.set_xlabel(r"$\eta$")
    ax_a.set_ylabel(r"$\gamma$")
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.13, 1.0), frameon=False,
                handletextpad=0.1, borderpad=0.1, labelcolor=black, fontsize=5.5)

    # -- the measure colour key --------------------------------------------
    from matplotlib.patches import Rectangle

    for row, col in zip(rows, row_colors):
        ax_key.add_patch(Rectangle((0, row - 0.5), 1, 1, facecolor=col,
                                   edgecolor=bone, linewidth=0.4))
    # one frame around the measures only, so the empty ceiling slot stays blank
    ax_key.add_patch(Rectangle((0, -0.5), 1, len(res), facecolor="none",
                               edgecolor=black, linewidth=0.6))
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(n_slots - 0.5, -0.5)
    ax_key.set_xticks([])
    ax_key.set_yticks([])
    for s in ax_key.spines.values():
        s.set_visible(False)

    # -- B: agreement at the best fit, against the ceiling -----------------
    ax_b.barh(rows, res["r_topo"], color=row_colors, edgecolor=black,
              linewidth=0.4, height=0.72, zorder=2)
    ax_b.axvline(meta["r_max"], color=black, lw=0.9, zorder=3)
    ax_b.axvline(meta["r_median"], color=gray, lw=0.9, ls="--", zorder=3)
    for row, eff in zip(rows, res["efficiency"]):
        ax_b.text(0.004, row, f"{eff:.0%}", va="center", ha="left", fontsize=5.5,
                  color=black, zorder=4)
    ax_b.set_yticks([])
    ax_b.set_ylim(n_slots - 0.5, -0.5)
    ax_b.set_xlim(0, max(meta["r_max"], res["r_topo"].max()) * 1.18)
    ax_b.set_xlabel("topographic agreement\nat the measure's best fit")
    ax_b.text(meta["r_max"], -0.62, "ceiling", fontsize=5.5, ha="right",
              va="bottom", color=black)
    ax_b.text(meta["r_median"], -0.62, "grid median", fontsize=5.5,
              ha="right", va="bottom", color=gray)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)

    # -- C: which topographies the best fits reproduce ---------------------
    heat = res[[f"r_{m}" for m in MAPS]].to_numpy()
    heat = np.vstack([heat, [meta["map_max"][m] for m in MAPS]])
    vmax = 0.75
    im_c = ax_c.imshow(heat, cmap=cmaps["db_bw_lr"], vmin=-vmax, vmax=vmax,
                       aspect="auto")
    cb_c = fig.colorbar(im_c, ax=ax_c, pad=0.10, fraction=0.05, aspect=13)
    cb_c.set_label("node-wise $r$ to the consensus", fontsize=6)
    cb_c.ax.tick_params(labelsize=5.5, width=0.6, length=1.5)
    cb_c.outline.set_linewidth(0.6)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            ax_c.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                      color=bone if abs(v) > 0.45 * vmax else black)
    # the ceiling row is not a measure; separate it and label it on the right
    ax_c.axhline(len(res) - 0.5, color=black, lw=0.8)
    ax_c.set_xticks(range(len(MAPS)))
    ax_c.set_xticklabels(MAP_LABELS, rotation=40, ha="right", fontsize=5.5)
    ax_c.yaxis.tick_right()
    ax_c.set_yticks([len(res)])
    ax_c.set_yticklabels(["best GNM\nanywhere"], fontsize=5.5, style="italic")
    ax_c.tick_params(axis="y", length=0)

    # B's letter goes on the colour key, which is the left edge of that panel
    for ax, letter in zip((ax_a, ax_key, ax_c), "ABC"):
        ax.text(0.0, 1.04, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="left", color=black)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # no bbox_inches="tight": constrained_layout already fits the content and
    # trimming would make the canvas something other than the requested 18 x 9 cm
    fig.savefig(out_pdf, dpi=300)
    print(f"  Saved figure -> {out_pdf}")


def selfcheck(land: pd.DataFrame, res: pd.DataFrame, meta: dict) -> None:
    """The two things that silently produce a wrong figure: a mis-shaped grid
    pivot, and a heatmap whose rows no longer line up with the bars."""
    etas = np.sort(land["eta"].unique())
    gammas = np.sort(land["gamma"].unique())
    grid = (land.pivot(index="gamma", columns="eta", values="combined")
                .reindex(index=gammas, columns=etas).to_numpy())
    assert grid.shape == (gammas.size, etas.size) == (50, 50), grid.shape
    assert not np.isnan(grid).any(), "grid has holes"
    # pcolormesh(etas, gammas, grid) means grid[i, j] must be the cell at
    # (gammas[i], etas[j]) - check one cell against the long form
    i, j = 7, 13
    cell = land[(land["gamma"] == gammas[i]) & (land["eta"] == etas[j])]
    assert np.isclose(grid[i, j], cell["combined"].iloc[0]), "pivot is transposed"
    assert list(meta["map_max"]) == MAPS, meta["map_max"].keys()
    assert [f"r_{m}" for m in MAPS] == [c for c in res.columns if c.startswith("r_")
                                        and c != "r_topo"]
    assert res["r_topo"].is_monotonic_decreasing, "bars are not sorted"
    print("  selfcheck ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "fig_topographic_plausibility_compact.pdf")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    land, res, meta = load(OUT_DIR)
    if args.selfcheck:
        selfcheck(land, res, meta)

    import matplotlib
    matplotlib.use("Agg")
    _, cmaps, colors = _style()
    plot(land, res, meta, cmaps, colors, args.out)


if __name__ == "__main__":
    main()
