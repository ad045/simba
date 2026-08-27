"""
PCA of the distance landscapes over the EIGHT selected measures.

The published run (PC1 51.4%, PC2 22.9%, PC3 10.3%) also contained resistance
distance, which the plausibility filter had excluded - visible in the appendix
figure as a scree plot running to PC9 and nine loading bars against eight
legend entries. This script repeats it on the eight selected measures only.

Input : one distance per generated network per measure, min-max normalised to
        [0, 1] over the morphospace, similarity-valued measures inverted,
        columns standardised -> 25,000 x 8 matrix.
Output: fig_pca_scree.pdf, fig_pca_landscapes.pdf, fig_pca_loadings_pc1.pdf,
        fig_pca_loadings_pc2.pdf and a combined fig_pca_landscapes_loadings.pdf,
        written to figures/appendix/pca_eight/ of the manuscript repo.

Run:  conda activate ma_thesis && python run_pca_eight_measures.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from vizman import viz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_config import MORPHO_DIR, MORPHO_EXP
from build_radar_five_axes import MEASURES, COLORS, NAMES

viz.set_visual_style()

OUT = Path("/Users/adrian/Desktop/Benchmarking_/figures/appendix/pca_eight")

# Column names that hold a similarity rather than a distance.
SIMILARITY_COLS = {"Resistance_subject_0", "Communicability_subject_0"}

POS, NEG = "#10b981", "#ee4444"      # loading bar colours, sampled from the published panel
INK, GRID = "#232323", "#f0f0f0"


def load_matrix():
    cols, coords = {}, None
    for m in MEASURES:
        df = pd.read_csv(MORPHO_DIR / f"summary_indiv_{m}_for_exp_{MORPHO_EXP}.csv")
        name = [c for c in df.columns
                if c not in ("eta", "gamma", "network_index", "filename", "id")][0]
        v = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
        cols[m] = (1 - v) if name in SIMILARITY_COLS else v
        if coords is None:
            coords = df[["eta", "gamma"]].copy()
    return pd.DataFrame(cols), coords


def orient(pca, scores):
    """Fix the arbitrary component signs: PC1 so its loadings sum positive, the
    rest so their largest-magnitude loading is positive."""
    comp = pca.components_.copy()
    flip = np.ones(len(comp))
    flip[0] = np.sign(comp[0].sum()) or 1.0
    for k in range(1, len(comp)):
        flip[k] = np.sign(comp[k][np.argmax(np.abs(comp[k]))]) or 1.0
    return comp * flip[:, None], scores * flip[None, :]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, coords = load_matrix()
    keep = X.dropna().index
    X, coords = X.loc[keep], coords.loc[keep]

    pca = PCA()
    scores = pca.fit_transform(StandardScaler().fit_transform(X))
    comp, scores = orient(pca, scores)
    var = pca.explained_variance_ratio_

    print(f"n = {len(X)} networks x {X.shape[1]} measures")
    for k in range(3):
        print(f"  PC{k+1}: {var[k]*100:.1f}%")
    loadings = pd.DataFrame(comp[:3].T, index=MEASURES, columns=["PC1", "PC2", "PC3"])
    print("\n" + loadings.round(4).to_string())
    loadings.to_csv(OUT / "pca_loadings_eight.csv")

    # --- A upper: scree -----------------------------------------------------
    fig = plt.figure(figsize=viz.cm_to_inch((6, 3))) # 4.6, 3.2)))
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.8)
    ax.plot(range(1, len(var) + 1), var, "-o", c=INK, ms=5, lw=1.4,
            markeredgecolor=INK, clip_on=False, zorder=3)
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained\nVariance")
    ax.set_xticks([1, len(var)])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_ylim(-0.02, 0.62)
    ax.set_xlim(0.6, len(var) + 0.4)
    # for side in ("top", "right"):
    #     ax.spines[side].set_visible(False)
    fig.savefig(OUT / "fig_pca_scree.pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)

    # --- A lower: the first three PC landscapes -----------------------------
    cells = coords.copy()
    for k in range(3):
        cells[f"PC{k+1}"] = scores[:, k]
    grid = cells.groupby(["eta", "gamma"]).mean()
    etas = np.sort(cells.eta.unique())
    gammas = np.sort(cells.gamma.unique())
    fig, axs = plt.subplots(1, 3, figsize=viz.cm_to_inch((6, 2))) # 7.4, 2.8)))
    fig.subplots_adjust(wspace=0.12)
    for k, ax in enumerate(axs):
        img = grid[f"PC{k+1}"].unstack().values.T      # rows gamma, cols eta
        ax.imshow(img, origin="lower", cmap="gray", aspect="auto",
                  extent=[etas[0], etas[-1], gammas[0], gammas[-1]])
        ax.set_title(f"PC{k+1}")
        # one shared axis across the three panels: the range ends carry the ticks
        ax.set_xticks({0: [etas[0]], 2: [etas[-1]]}.get(k, []))
        ax.set_yticks([gammas[-1], gammas[0]] if k == 0 else [])
        if k == 0:
            ax.set_ylabel(r"$\gamma$")
            ax.set_yticklabels(["1.0", "-0.1"])
        if k == 1:
            ax.set_xlabel(r"$\eta$")
        for side in ax.spines.values():
            side.set_color(INK)
            side.set_linewidth(1.0)
    fig.savefig(OUT / "fig_pca_landscapes.pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)

    # --- B and C: loadings --------------------------------------------------
    # Both loading panels share one row order - ordered by the PC1 loading - so a
    # measure sits on the same row in B and C and the figure needs only one legend.
    order = loadings["PC1"].sort_values().index
    for pc in ("PC1", "PC2"):
        fig = plt.figure(figsize=viz.cm_to_inch((6, 6))) # 4.6, 4.0)))
        ax = fig.add_subplot(111)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, lw=0.8)
        vals = loadings.loc[order, pc].values
        ax.barh(range(len(order)), vals,
                color=[POS if v >= 0 else NEG for v in vals], height=0.78)
        ax.axvline(0, c=INK, lw=1.0)
        # measures are identified by the figure legend, not by tick labels
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.invert_yaxis()          # first row of the shared order at the top
        ax.set_xlabel(f"{pc} Loading")
        for side in ax.spines.values():
            side.set_color(INK)
            side.set_linewidth(1.0)
        fig.savefig(OUT / f"fig_pca_loadings_{pc.lower()}.pdf",
                    bbox_inches="tight", transparent=True)
        plt.close(fig)

    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
