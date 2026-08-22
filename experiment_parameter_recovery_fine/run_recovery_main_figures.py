"""
Main-text recovery figures (wide bars, window bars, value-spaced slopegraph)
===========================================================================

Three standalone panels, sized for assembly in a vector editor:

  fig_recovery_bars_wide.pdf     6 x 6 cm   ranking on the 5 widely-spread targets
  fig_recovery_bars_window.pdf   6 x 6 cm   ranking inside the plausible window
  fig_recovery_slopegraph.pdf    6 x 12 cm  wide -> window, positioned by value

Both bar panels follow panel A of the supplementary recovery figure: two bars
per measure, eta on top in the measure's colour and gamma below in a lightened
version of it, rows ordered by the joint grid-step error. Measures outside the
selected eight are drawn empty with grey labels, and the dashed line marks the
per-axis chance level.

The slopegraph places every measure at its actual error value on both sides
rather than at its rank, so vertical distance carries information: measures that
sit close together score alike, and the crossing lines are the measures whose
standing changes between the two regimes. Labels are nudged apart only as far as
needed to stay legible, with a leader line drawn whenever a label had to move.

Scores come from the same loaders as run_fine_recovery_paper_plot_concise.py, so
the numbers here and in the supplementary figure cannot drift apart.

Usage
-----
    conda activate ma_thesis
    python run_recovery_main_figures.py
    python run_recovery_main_figures.py --outdir /some/where
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vizman import viz

_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = _THIS_DIR.parent
for _p in (str(ROOT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_config import fine as cfg, METHOD_NAMES, METRIC_COLORS  # noqa: E402
from run_synthetic_gnm_fine import (  # noqa: E402
    METRIC_NAMES, WIDE_COMPARISON_DIR, WIDE_GRID_N_ETA,
)
from run_fine_recovery_paper_plot_concise import (  # noqa: E402
    COMPARISON_DIR, UNSELECTED_COLOR, _grid_indices, chance_grid_steps,
)

DEFAULT_OUTDIR = (ROOT_DIR / "output" / "gnm" /
                  "synthetic_parameter_recovery_fine" / "plots")

DARK, GRAY, BLACK = "#333333", "#888888", "#000000"
FS = 5 # 6
HATCH = "//////"

# wide recovery grid axes (gnm_grid_config), for snapping true values to cells
WIDE_ETA = np.linspace(-8.0, 3.0, WIDE_GRID_N_ETA)
WIDE_GAMMA = np.linspace(-0.1, 1.0, WIDE_GRID_N_ETA)


# ---------------------------------------------------------------------------
# Per-measure errors, keeping the per-network arrays so spans can be computed
# ---------------------------------------------------------------------------

def _rows_from(d_eta: np.ndarray, d_gamma: np.ndarray, measure: str,
               n: int) -> Dict:
    step = np.sqrt(d_eta ** 2 + d_gamma ** 2)
    return {
        "measure": measure,
        "name": METHOD_NAMES.get(measure, METRIC_NAMES.get(measure, measure)),
        "grid_steps": float(np.nanmean(step)),
        "eta_mean": float(np.nanmean(d_eta)), "eta_sd": float(np.nanstd(d_eta)),
        "gamma_mean": float(np.nanmean(d_gamma)), "gamma_sd": float(np.nanstd(d_gamma)),
        "n_networks": n,
    }


def window_scores() -> pd.DataFrame:
    rows = []
    for path in sorted(COMPARISON_DIR.glob("distances_*.csv")):
        measure = path.stem.replace("distances_", "")
        df = pd.read_csv(path)
        rec_eta, rec_gamma, true_eta, true_gamma = _grid_indices(df)
        rows.append(_rows_from(np.abs(rec_eta - true_eta),
                               np.abs(rec_gamma - true_gamma), measure, len(df)))
    return pd.DataFrame(rows).sort_values("grid_steps").reset_index(drop=True)


def wide_scores() -> pd.DataFrame:
    """Same index convention as run_synthetic_gnm_fine.wide_ranking()."""
    rows = []
    for path in sorted(WIDE_COMPARISON_DIR.glob("distances_*.csv")):
        measure = path.stem.replace("distances_", "")
        df = pd.read_csv(path)
        if "predicted_grid_idx" not in df:
            continue
        valid = df[df["predicted_grid_idx"] >= 0].copy()
        pg = (valid["predicted_grid_idx"] // WIDE_GRID_N_ETA).to_numpy()
        pe = (valid["predicted_grid_idx"] % WIDE_GRID_N_ETA).to_numpy()
        te = valid["true_eta"].apply(
            lambda e: int(np.argmin(np.abs(WIDE_ETA - e)))).to_numpy()
        tg = valid["true_gamma"].apply(
            lambda g: int(np.argmin(np.abs(WIDE_GAMMA - g)))).to_numpy()
        rows.append(_rows_from(np.abs(pe - te), np.abs(pg - tg), measure, len(valid)))
    return pd.DataFrame(rows).sort_values("grid_steps").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Panel 1 + 2: paired eta / gamma bars, styled like the supplementary panel A
# ---------------------------------------------------------------------------

def bars_figure(scores: pd.DataFrame, chance_axis: float, title: str,
                out_pdf: Path) -> None:
    """Two bars per measure - eta on top in the measure colour, gamma below in
    a lightened version of it - matching panel A of the supplementary recovery
    figure. Measures outside the selected eight are drawn empty with grey
    labels. The dashed line is the PER-AXIS chance level, since these bars are
    per-axis errors rather than the joint grid-step error.
    """
    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6, 9)))
    n = len(scores)
    y = np.arange(n)[::-1]
    h = 0.38

    selected = [m in METRIC_COLORS for m in scores["measure"]]
    base = [METRIC_COLORS.get(m, UNSELECTED_COLOR) for m in scores["measure"]]
    light = [tuple(1 - 0.45 * (1 - np.asarray(c))) for c in base]
    colors = [c if s else "white" for c, s in zip(base, selected)]
    light = [c if s else "white" for c, s in zip(light, selected)]

    x_max = float(max(scores[["eta_mean", "gamma_mean"]].max())) * 1.30

    for yi, ce, cg, sel, (_, row) in zip(y, colors, light, selected,
                                         scores.iterrows()):
        ax.barh(yi + h / 2, row["eta_mean"], height=h, color=ce,
                edgecolor=GRAY, linewidth=0.3, zorder=2)
        ax.barh(yi - h / 2, row["gamma_mean"], height=h, color=cg,
                edgecolor=GRAY, linewidth=0.3, zorder=2)
        # variation across the ground-truth networks: the bar end is the mean,
        # so only the outward half is drawn, from the mean out to mean + 1 SD.
        # It takes the colour of its own bar, or grey where the bar is empty.
        for yy, mean, sd, cc in ((yi + h / 2, row["eta_mean"], row["eta_sd"], ce),
                                 (yi - h / 2, row["gamma_mean"], row["gamma_sd"], cg)):
            mark = cc if sel else GRAY
            hi = min(x_max, mean + sd)
            ax.plot([mean, hi], [yy, yy], color=mark, lw=0.35, zorder=5,
                    solid_capstyle="butt")
            ax.plot([hi, hi], [yy - 0.09, yy + 0.09], color=mark, lw=0.35,
                    zorder=5)

    ax.axvline(chance_axis, color=GRAY, ls="-", lw=0.4, zorder=1)
    ax.text(chance_axis + 0.05, n - 0.55, "chance\n(per axis)", fontsize=FS - 1,
            color=GRAY, va="top", ha="left", linespacing=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(scores["name"], fontsize=FS)
    for lab, sel in zip(ax.get_yticklabels(), selected):
        lab.set_color(DARK if sel else GRAY)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Mean absolute index error,\nper axis (grid steps)", fontsize=FS,
                  linespacing=1.2)
    ax.tick_params(axis="x", labelsize=FS)
    ax.set_title(title, fontsize=FS + 1, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GRAY, edgecolor=DARK, linewidth=0.3,
                             label=r"$\eta$ axis"),
                       Patch(facecolor="#c8c8c8", edgecolor=DARK, linewidth=0.3,
                             label=r"$\gamma$ axis")],
              loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              frameon=False, fontsize=FS)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_pdf}")


# ---------------------------------------------------------------------------
# Panel 3: value-spaced slopegraph
# ---------------------------------------------------------------------------

def _declutter(values: np.ndarray, min_gap: float, n_iter: int = 400) -> np.ndarray:
    """Nudge label positions apart while keeping order and staying near values."""
    pos = values.astype(float).copy()
    order = np.argsort(pos)
    for _ in range(n_iter):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            gap = pos[b] - pos[a]
            if gap < min_gap:
                shift = (min_gap - gap) / 2
                pos[a] -= shift
                pos[b] += shift
                moved = True
        if not moved:
            break
    return pos


def slopegraph(wide: pd.DataFrame, window: pd.DataFrame, out_pdf: Path) -> None:
    m = wide.merge(window, on="measure", suffixes=("_wide", "_win"))
    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6, 9)))

    xl, xr = 0.0, 1.0
    lo = min(m["grid_steps_wide"].min(), m["grid_steps_win"].min())
    hi = max(m["grid_steps_wide"].max(), m["grid_steps_win"].max())
    span = hi - lo
    # labels need ~2.6% of the axis height each to stay readable at 12 cm
    lab_w = _declutter(m["grid_steps_wide"].to_numpy(), 0.026 * span)
    lab_n = _declutter(m["grid_steps_win"].to_numpy(), 0.026 * span)

    for (_, row), lw_, ln_ in zip(m.iterrows(), lab_w, lab_n):
        c = METRIC_COLORS.get(row["measure"], UNSELECTED_COLOR)
        sel = row["measure"] in METRIC_COLORS
        ax.plot([xl, xr], [row["grid_steps_wide"], row["grid_steps_win"]],
                color=c, lw=0.9 if sel else 0.55, alpha=0.95 if sel else 0.5,
                zorder=3 if sel else 2)
        for x, v in ((xl, row["grid_steps_wide"]), (xr, row["grid_steps_win"])):
            ax.plot([x], [v], marker="o", ms=2.2 if sel else 1.5, color=c,
                    zorder=4)
        # leader lines only where the label had to leave its value
        for x, v, lab, ha, sgn in ((xl, row["grid_steps_wide"], lw_, "right", -1),
                                   (xr, row["grid_steps_win"], ln_, "left", 1)):
            if abs(lab - v) > 1e-9:
                ax.plot([x + sgn * 0.03, x + sgn * 0.08], [v, lab],
                        color=GRAY, lw=0.25, zorder=1)
            ax.text(x + sgn * 0.10, lab,
                    f"{row['name_wide']}  {v:.2f}" if ha == "right"
                    else f"{v:.2f}  {row['name_wide']}",
                    fontsize=FS - 0.5, color=DARK if sel else GRAY,
                    va="center", ha=ha, zorder=5)

    ax.set_xlim(-1.15, 2.15)
    top = lo - 0.10 * span
    ax.set_ylim(hi + 0.05 * span, top)                # inverted: better is up
    # every point carries its value, so the axis itself would only add clutter
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    # headers point outwards from their own column, so they cannot collide
    for x, lab, ha in ((xl, "Widely-spread\ntargets", "right"),
                       (xr, "Plausible\nwindow", "left")):
        ax.text(x, top, lab, fontsize=FS, color=DARK, va="top", ha=ha,
                linespacing=1.15)
    ax.set_xlabel("vertical position = recovery error (grid steps)",
                  fontsize=FS - 1, color=GRAY)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()
    outdir = Path(args.outdir)

    viz.set_visual_style(font_family="Arial")

    win = window_scores()
    wide = wide_scores()

    any_df = pd.read_csv(sorted(COMPARISON_DIR.glob("distances_*.csv"))[0])
    chance_joint, chance_axis = chance_grid_steps(any_df)
    # the wide grid has the same 10x10 shape, so the same chance level applies
    print(f"\nMeasures: {len(win)} window / {len(wide)} wide; "
          f"chance = {chance_joint:.2f} grid steps joint, "
          f"{chance_axis:.2f} per axis\n")

    bars_figure(wide, chance_axis, "Widely-spread targets",
                outdir / "fig_recovery_bars_wide.pdf")
    bars_figure(win, chance_axis, "Plausible window",
                outdir / "fig_recovery_bars_window.pdf")
    slopegraph(wide, win, outdir / "fig_recovery_slopegraph.pdf")

    merged = wide.merge(win, on="measure", suffixes=("_wide", "_win"))
    merged["rank_wide"] = merged["grid_steps_wide"].rank().astype(int)
    merged["rank_win"] = merged["grid_steps_win"].rank().astype(int)
    merged.to_csv(outdir / "recovery_main_figure_scores.csv", index=False)
    print(merged[["name_wide", "grid_steps_wide", "rank_wide",
                  "grid_steps_win", "rank_win"]]
          .sort_values("grid_steps_win").to_string(index=False))


if __name__ == "__main__":
    main()
