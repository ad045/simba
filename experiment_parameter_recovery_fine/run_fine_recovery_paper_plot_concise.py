"""
Paper figure for the fine-grained (human-plausible window) parameter recovery (S1)
==================================================================================

Renders the supplementary figure that accompanies the S1 analysis. Reads only the
cached per-measure distance CSVs produced by ``run_synthetic_gnm_fine.py --stage
compare``; nothing is recomputed here.

Two panels:

  A  Recovery error per measure: bars give the mean absolute index error along
     each axis separately, the gutter on the right gives the joint error in grid
     steps. The chance level of a random cell is drawn as a reference line.

  B  Rank comparison: wide-target recovery vs plausible-window recovery, both in
     grid steps, so the two rankings share a unit and every measure appears in
     both.

Why both scores. Grid steps are the same currency as the wide-target experiment
(both grids are 10 x 10, maximum error ~12.73), so the two experiments can be
ranked against each other, a chance level can be stated, and the degenerate
measures still get a score - but inside this window the joint error is dominated
by gamma, which no measure recovers, so it ranks measures largely on a noise
axis. Pearson r on eta discriminates on the only axis carrying signal, but is
blind to bias and scale compression and is undefined when recovery is constant.
The figure shows grid steps only; r(eta) stays in the CSV and the printout, and
is reported in the text where the eta-only view matters. Note also that
one fine grid step is 0.16 in eta against 1.22 on the wide grid: equal grid-step
errors mean very different parameter errors in the two experiments.

The window itself is derived from the morphospace (central 50% of the pooled
best-fit cells of the 8 selected measures) - see derive_window.py.

Usage
-----
  python run_fine_recovery_paper_plot.py
  python run_fine_recovery_paper_plot.py --out /path/to/fig.pdf
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from vizman import viz

_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = _THIS_DIR.parent
for _p in (str(ROOT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_config import fine as cfg, METHOD_NAMES, METRIC_COLORS  # noqa: E402
from run_synthetic_gnm_fine import METRIC_NAMES, wide_ranking, _pearson  # noqa: E402

COMPARISON_DIR = (ROOT_DIR / "output" / "gnm" /
                  "synthetic_parameter_recovery_fine" / "comparison_results")
DEFAULT_OUT = (ROOT_DIR / "output" / "gnm" /
               "synthetic_parameter_recovery_fine" / "plots" /
               "fig_fine_recovery.pdf")

UNSELECTED_COLOR = (0.78, 0.78, 0.78)


def _grid_indices(df: pd.DataFrame):
    """Recovered and true (eta, gamma) grid indices; true = nearest grid cell.

    Same convention as the wide-target experiment (gamma-outer, eta-inner flat
    index; true location snapped to its closest cell), so the resulting errors
    are directly comparable between the two.
    """
    valid = df[df["recovered_grid_idx"] >= 0]
    rec_gamma = (valid["recovered_grid_idx"] // cfg.GRID_N_ETA).to_numpy()
    rec_eta = (valid["recovered_grid_idx"] % cfg.GRID_N_ETA).to_numpy()
    true_eta = valid["true_eta"].apply(
        lambda e: int(np.argmin(np.abs(cfg.GRID_ETA - e)))).to_numpy()
    true_gamma = valid["true_gamma"].apply(
        lambda g: int(np.argmin(np.abs(cfg.GRID_GAMMA - g)))).to_numpy()
    return rec_eta, rec_gamma, true_eta, true_gamma


def chance_grid_steps(df: pd.DataFrame, n_draws: int = 2000,
                      seed: int = 0) -> Tuple[float, float]:
    """Error of a measure that recovers a uniformly random cell.

    Returns (joint grid-step error, mean absolute error along one axis).
    """
    _, _, true_eta, true_gamma = _grid_indices(df)
    rng = np.random.default_rng(seed)
    n = len(true_eta)
    joint, per_axis = [], []
    for _ in range(n_draws):
        d_eta = rng.integers(0, cfg.GRID_N_ETA, n) - true_eta
        d_gamma = rng.integers(0, cfg.GRID_N_GAMMA, n) - true_gamma
        joint.append(np.sqrt(d_eta ** 2 + d_gamma ** 2).mean())
        per_axis.append(np.abs(np.r_[d_eta, d_gamma]).mean())
    return float(np.mean(joint)), float(np.mean(per_axis))


def load_scores() -> pd.DataFrame:
    """Per-measure grid-step error, its per-axis parts, and r(eta), r(gamma)."""
    rows = []
    for path in sorted(COMPARISON_DIR.glob("distances_*.csv")):
        measure = path.stem.replace("distances_", "")
        df = pd.read_csv(path)
        rec_eta, rec_gamma, true_eta, true_gamma = _grid_indices(df)
        d_eta = np.abs(rec_eta - true_eta)
        d_gamma = np.abs(rec_gamma - true_gamma)
        step = np.sqrt(d_eta ** 2 + d_gamma ** 2)
        constant_eta = df["recovered_eta"].nunique(dropna=True) <= 1
        rows.append({
            "measure": measure,
            "name": METHOD_NAMES.get(measure, METRIC_NAMES.get(measure, measure)),
            "grid_steps": float(np.nanmean(step)),
            "grid_steps_sd": float(np.nanstd(step)),
            "d_eta_steps": float(np.nanmean(d_eta)),
            "d_gamma_steps": float(np.nanmean(d_gamma)),
            "r_eta": _pearson(df["true_eta"], df["recovered_eta"]),
            "r_gamma": _pearson(df["true_gamma"], df["recovered_gamma"]),
            "degenerate": constant_eta,
            "n_networks": len(df),
        })
    return (pd.DataFrame(rows).sort_values("grid_steps")
                              .reset_index(drop=True))


def make_figure(scores: pd.DataFrame, chance_axis: float, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figsize = viz.cm_to_inch((18,12))
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=figsize) # , gridspec_kw={"width_ratios": [1.0, 1.35]})

    DARK = "#333333"
    GRAY = "#888888"
    FS = 7          # one size for every label; only the panel titles differ

    # ---------------- Panel A: mean grid-step error per measure -------------
    n = len(scores)
    y = np.arange(n)[::-1]
    h = 0.38
    colors = [METRIC_COLORS.get(m, UNSELECTED_COLOR) for m in scores["measure"]]
    # gamma bars: same hue, lightened towards white. Hatching marks the
    # measures outside the eight selected ones (those without a colour).
    light = [tuple(1 - 0.45 * (1 - np.asarray(c))) for c in colors]
    # measures outside the selected eight: empty bars rather than hatched ones
    selected = [m in METRIC_COLORS for m in scores["measure"]]
    colors = [c if s else "white" for c, s in zip(colors, selected)]
    light = [c if s else "white" for c, s in zip(light, selected)]

    for yi, ce, cg, de, dg in zip(y, colors, light,
                                  scores["d_eta_steps"],
                                  scores["d_gamma_steps"]):
        axA.barh(yi + h / 2, de, height=h, color=ce, edgecolor=GRAY,
                 linewidth=0.3)
        axA.barh(yi - h / 2, dg, height=h, color=cg, edgecolor=GRAY,
                 linewidth=0.3)
        # Add small eta and gamma signs
        # label_pos_e = 0.16 # 06 # de - 0.06
        # label_pos_g = 0.16 # 06 # dg - 0.06
        # axA.text(label_pos_e, yi + h / 2 - 0.06, r"$\eta$", ha="right", va="center",
        #          fontsize=5.5, color=DARK)
        # axA.text(label_pos_g, yi - h / 2 - 0.06, r"$\gamma$", ha="right", va="center",
        #          fontsize=5.5, color=DARK)

    # Right-hand gutter: the joint grid-step error the rows are ordered by.
    bar_max = float(max(scores[["d_eta_steps", "d_gamma_steps"]].max()))
    x_max = bar_max + 1.0
    x_steps = x_max - 0.05

    # Add back the grid-step values directly into the plot.
    # axA.text(x_steps, n - 0.15, "grid\nsteps", ha="right", va="bottom",
    #         #  fontsize=5.5, 
    #          color=DARK, linespacing=0.9)
    # for yi, (_, row) in zip(y, scores.iterrows()):
    #     axA.text(x_steps, yi, f"{row['grid_steps']:.2f}", va="center",
    #              ha="right", # fontsize=6, 
    #              color=DARK)

        # if row["degenerate"]:
        #     axA.text(x_steps - 0.95, yi, "const.", va="center", ha="right",
        #              fontsize=5.5, style="italic", color=GRAY)

    axA.axvline(chance_axis, color=GRAY, ls="--", lw=0.9)
    axA.text(chance_axis + 0.08, n - 0.1, "chance (per axis)", fontsize=FS,
             color=GRAY, va="bottom", ha="left")
    axA.set_yticks(y)
    axA.set_yticklabels(scores["name"], fontsize=FS)
    for lab, sel in zip(axA.get_yticklabels(), selected):
        lab.set_color(DARK if sel else GRAY)
    axA.set_xlabel("Mean absolute index error, per axis (grid steps)", fontsize=FS)
    axA.set_xlim(0, x_max)
    axA.tick_params(axis="x", labelsize=FS)
    axA.legend(handles=[
        Patch(facecolor=GRAY, edgecolor=DARK, linewidth=0.3, label=r"$\eta$ axis"),
        Patch(facecolor="#c8c8c8", edgecolor=DARK, linewidth=0.3,
              label=r"$\gamma$ axis"),
        # Patch(facecolor="white", edgecolor=DARK, linewidth=0.1, hatch="///",
        #       label="not selected"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
       frameon=False, fontsize=FS)
    axA.set_title("A", loc="left", fontsize=10, fontweight="bold")
    for s in ("top", "right", "left"):
        axA.spines[s].set_visible(False)

    # ---------------- Panel B: wide vs plausible-window ranking -------------
    # Two columns, both in grid steps, so every measure appears in both (a
    # degenerate recovery still has a well-defined error).
    wide = wide_ranking()
    fine = scores.copy()
    fine["fine_rank"] = np.arange(1, len(fine) + 1)
    merged = fine.merge(wide, on="measure", how="inner")
    # "first" rather than "min": Frobenius and Hamming score identically on the
    # wide targets, and a shared rank would draw their labels on top of each other.
    merged = merged.sort_values("wide_grid_steps").reset_index(drop=True)
    merged["wide_rank"] = np.arange(1, len(merged) + 1)
    defined = merged[merged["r_eta"].notna()].sort_values(
        "r_eta", ascending=False).reset_index(drop=True)
    merged["r_rank"] = merged["measure"].map(
        {m: i + 1 for i, m in enumerate(defined["measure"])})   # CSV only

    LBL_GAP = 0.13   # gap between the marker and its label
    # Columns 1.5 apart (not 1) so the two axis headers do not run into each other.
    X0, X1 = 0.0, 2.1
    for _, r in merged.iterrows():
        c = METRIC_COLORS.get(r["measure"], UNSELECTED_COLOR)
        sel = r["measure"] in METRIC_COLORS
        axB.plot([X0, X1], [r["wide_rank"], r["fine_rank"]], "-o",
                 color=c, lw=0.9 if sel else 0.55,
                 markersize=3 if sel else 2.2)
        for x, rank, ha in ((X0 - LBL_GAP, r["wide_rank"], "right"),
                            (X1 + LBL_GAP, r["fine_rank"], "left")):
            axB.text(x, rank, r["name"], ha=ha, va="center",
                     fontsize=FS, color=DARK if sel else GRAY)

    axB.set_xlim(X0 - 1.6, X1 + 1.6)
    axB.set_xticks([X0, X1])
    axB.set_xticklabels(["Wide targets\n(grid steps)",
                         "Plausible window\n(grid steps)"], fontsize=FS)
    axB.spines["left"].set_visible(False)
    axB.set_yticks([])
    # rank 1 (best) at the top, matching panel A; axis itself stays hidden.
    axB.set_ylim(len(merged) + 0.5, 0.5)
    # axB.set_yticks(range(1, len(merged) + 1))
    # axB.tick_params(axis="y", labelsize=7, pad=1)
    # axB.set_ylabel("Rank (1 = most accurate)", fontsize=8, labelpad=2)
    axB.set_title("B", loc="left", fontsize=10, fontweight="bold")
    for s in ("left", "top", "right", "bottom"):
        axB.spines[s].set_visible(False)
    axB.tick_params(bottom=False)
    axB.tick_params(left=False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    tau = merged[["fine_rank", "wide_rank"]].corr(method="kendall").iloc[0, 1]
    print(f"Kendall tau, wide vs window (both grid steps)  = {tau:.3f}")
    sub = merged.dropna(subset=["r_rank"])
    tau_r = sub[["r_rank", "wide_rank"]].corr(method="kendall").iloc[0, 1]
    tau_rs = sub[["r_rank", "fine_rank"]].corr(method="kendall").iloc[0, 1]
    print(f"Kendall tau, wide vs window r(eta)             = {tau_r:.3f} "
          f"(n={len(sub)})")
    print(f"Kendall tau, window grid steps vs r(eta)       = {tau_rs:.3f}")
    print(f"Saved -> {out_pdf}")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    scores = load_scores()
    chance_joint, chance_axis = chance_grid_steps(
        pd.read_csv(sorted(COMPARISON_DIR.glob("distances_*.csv"))[0]))
    print(f"\nGround-truth networks per measure: {scores['n_networks'].iloc[0]}")
    print(f"Window: eta in {cfg.SAMPLE_ETA_RANGE}, gamma in {cfg.SAMPLE_GAMMA_RANGE}")
    print(f"Chance: {chance_joint:.2f} grid steps (joint), "
          f"{chance_axis:.2f} per axis\n")
    for _, r in scores.iterrows():
        re = "  n/a" if pd.isna(r["r_eta"]) else f"{r['r_eta']:5.3f}"
        print(f"  {r['name']:28s} {r['grid_steps']:5.2f} grid steps "
              f"(eta {r['d_eta_steps']:4.2f}, gamma {r['d_gamma_steps']:4.2f})  "
              f"r(eta)={re}{'   [constant recovery]' if r['degenerate'] else ''}")

    merged = make_figure(scores, chance_axis, Path(args.out))
    scores.to_csv(Path(args.out).parent / "fine_recovery_scores.csv", index=False)
    merged[["measure", "name", "grid_steps", "d_eta_steps", "d_gamma_steps",
            "r_eta", "wide_grid_steps", "wide_rank", "fine_rank"]] \
        .to_csv(Path(args.out).parent / "fine_vs_wide_rank.csv", index=False)


if __name__ == "__main__":
    main()
