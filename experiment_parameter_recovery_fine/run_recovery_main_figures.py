# SLOPEGRAPH ! 

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

from experiments_config import (  # noqa: E402
    fine as cfg, coarse as wide_cfg, METHOD_NAMES, METRIC_COLORS,
)
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


def _wide_chance(df: pd.DataFrame, n_draws: int = 2000,
                 seed: int = 0) -> Tuple[float, float]:
    """Chance level on the WIDE grid: what a measure returning a uniformly
    random cell would score against the five widely-spread true combinations.

    The window version lives in run_fine_recovery_paper_plot_concise; it is the
    same calculation against a different ground truth, and the two levels differ
    because the true points sit differently, not because the grids differ.
    """
    valid = df[df["predicted_grid_idx"] >= 0]
    true_eta = valid["true_eta"].apply(
        lambda e: int(np.argmin(np.abs(WIDE_ETA - e)))).to_numpy()
    true_gamma = valid["true_gamma"].apply(
        lambda g: int(np.argmin(np.abs(WIDE_GAMMA - g)))).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(true_eta)
    joint, per_axis = [], []
    for _ in range(n_draws):
        d_eta = rng.integers(0, wide_cfg.GRID_N_ETA, n) - true_eta
        d_gamma = rng.integers(0, wide_cfg.GRID_N_GAMMA, n) - true_gamma
        joint.append(np.sqrt(d_eta ** 2 + d_gamma ** 2).mean())
        per_axis.append(np.abs(np.r_[d_eta, d_gamma]).mean())
    return float(np.mean(joint)), float(np.mean(per_axis))


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
                out_pdf: Path, panel_letter: str | None = None) -> None:
    """Two bars per measure - eta on top in the measure colour, gamma below in
    a lightened version of it - matching panel A of the supplementary recovery
    figure. Measures outside the selected eight are drawn empty with grey
    labels. The red line is the PER-AXIS chance level, since these bars are
    per-axis errors rather than the joint grid-step error, and it is labelled
    under the x axis where its value can be read off the same scale as the bars.

    The two axes are told apart by small eta / gamma marks on the top row rather
    than by a legend, so nothing sits between the panel and its caption.
    """
    from matplotlib.transforms import blended_transform_factory

    F_LAB, F_TITLE, F_PANEL = 8.0, 9.0, 14.0
    CHANCE_RED = "#e8000b"

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((9, 9))) # 10.5, 9.5)))
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
                edgecolor=GRAY, linewidth=0.4, zorder=2)
        ax.barh(yi - h / 2, row["gamma_mean"], height=h, color=cg,
                edgecolor=GRAY, linewidth=0.4, zorder=2)
        # variation across the ground-truth networks: the bar end is the mean,
        # so only the outward half is drawn, from the mean out to mean + 1 SD.
        # It takes the colour of its own bar, or grey where the bar is empty.
        for yy, mean, sd, cc in ((yi + h / 2, row["eta_mean"], row["eta_sd"], ce),
                                 (yi - h / 2, row["gamma_mean"], row["gamma_sd"], cg)):
            mark = cc if sel else GRAY
            hi = min(x_max, mean + sd)
            ax.plot([mean, hi], [yy, yy], color=mark, lw=0.45, zorder=5,
                    solid_capstyle="butt")
            ax.plot([hi, hi], [yy - 0.09, yy + 0.09], color=mark, lw=0.45,
                    zorder=5)

    # Which bar is which axis, marked once on the top row instead of a legend.
    for yy, sym in ((y[0] + h / 2, r"$\eta$"), (y[0] - h / 2, r"$\gamma$")):
        ax.text(x_max * 0.022, yy, sym, fontsize=F_LAB - 2.5, # 1.5, 
                color=DARK, va="center", ha="left", zorder=6)

    ax.axvline(chance_axis, color=CHANCE_RED, ls="-", lw=1.0, zorder=0)
    # The label sits under the axis, so it reads against the same x scale as
    # the bars instead of floating inside the plot.
    ax.text(chance_axis, -0.075, "chance\n(per axis)",
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            fontsize=F_LAB - 1.5, color=CHANCE_RED, va="top", ha="center",
            linespacing=1.05, clip_on=False, zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(scores["name"], fontsize=F_LAB)
    for lab, sel in zip(ax.get_yticklabels(), selected):
        lab.set_color(DARK if sel else GRAY)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Mean absolute index error, per axis (grid steps)",
                  fontsize=F_LAB, labelpad=16)
    ax.tick_params(axis="x", labelsize=F_LAB)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=F_TITLE, fontweight="bold", color=BLACK,
                 loc="center", pad=8)
    for sp in ("top", "right"): # , "left"):
        ax.spines[sp].set_visible(False)
    if panel_letter:
        fig.text(0.0, 1.0, panel_letter, fontsize=F_PANEL, fontweight="bold",
                 color=BLACK, va="top", ha="left")

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


def slopegraph(wide: pd.DataFrame, window: pd.DataFrame,
               chance_wide: float, chance_window: float,
               out_pdf: Path) -> None:
    """Wide -> window, every measure placed at its actual error value.

    Vertical distance carries meaning: measures level with each other score
    alike, and a crossing line is a measure whose standing changes between the
    two regimes. Each column carries its OWN chance level (the two experiments
    place their ground truth differently, so the level a random cell would
    score is not the same on both), drawn in red so a reader can see at a
    glance which measures fall past it.
    """
    m = wide.merge(window, on="measure", suffixes=("_wide", "_win"))
    fig, ax = plt.subplots(figsize=viz.cm_to_inch((9, 16))) # 9.5, 15)))

    F_HEAD, F_LAB, F_CAP = 8.0, 7.0, 8.0
    CHANCE_RED = "#d62728"

    xl, xr = 0.0, 1.0
    vals_w = m["grid_steps_wide"].to_numpy()
    vals_n = m["grid_steps_win"].to_numpy()
    lo = min(vals_w.min(), vals_n.min(), chance_wide, chance_window)
    hi = max(vals_w.max(), vals_n.max(), chance_wide, chance_window)
    span = hi - lo

    # Declutter each column WITH its chance label included, so the red label
    # cannot land on top of a measure label.
    min_gap = 0.030 * span
    lab_w = _declutter(np.r_[vals_w, chance_wide], min_gap)
    lab_n = _declutter(np.r_[vals_n, chance_window], min_gap)
    chance_lab_w, chance_lab_n = lab_w[-1], lab_n[-1]
    lab_w, lab_n = lab_w[:-1], lab_n[:-1]

    for (_, row), lw_, ln_ in zip(m.iterrows(), lab_w, lab_n):
        c = METRIC_COLORS.get(row["measure"], UNSELECTED_COLOR)
        sel = row["measure"] in METRIC_COLORS
        ax.plot([xl, xr], [row["grid_steps_wide"], row["grid_steps_win"]],
                color=c if sel else "#d8d8d8", lw=1.8 if sel else 0.9,
                alpha=1.0 if sel else 0.9, zorder=3 if sel else 2,
                solid_capstyle="round")
        for x, v in ((xl, row["grid_steps_wide"]), (xr, row["grid_steps_win"])):
            ax.plot([x], [v], marker="o", ms=5.0 if sel else 3.0,
                    color=c if sel else "#d8d8d8",
                    markeredgecolor="white" if sel else "none",
                    markeredgewidth=0.6, zorder=4)

        # Left column: the value only. Right column: value + measure name.
        for x, v, lab, ha, sgn, text in (
            (xl, row["grid_steps_wide"], lw_, "right", -1, f"{row['grid_steps_wide']:.2f}"),
            (xr, row["grid_steps_win"], ln_, "left", 1,
             f"{row['grid_steps_win']:.2f}  {row['name_wide']}"),
        ):
            if abs(lab - v) > 1e-9:      # leader line only where a label moved
                ax.plot([x + sgn * 0.03, x + sgn * 0.09], [v, lab],
                        color="#bbbbbb", lw=0.4, zorder=1)
            ax.text(x + sgn * 0.11, lab, text, fontsize=F_LAB,
                    color=DARK if sel else "#b0b0b0",
                    va="center", ha=ha, zorder=5)

    ax.set_xlim(-0.62, 2.55)
    top = lo - 0.13 * span
    bottom = hi + 0.07 * span
    ax.set_ylim(bottom, top)                          # inverted: better is up

    # The two columns as labelled arrows, so the reading direction is explicit.
    for x, head in ((xl, "Widely-spread\nrange"), (xr, "Plausible\nwindow")):
        ax.annotate("", xy=(x, bottom), xytext=(x, top),
                    arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.3,
                                    shrinkA=0, shrinkB=0), zorder=1)
        ax.text(x + 0.06, top, head, fontsize=F_HEAD, fontweight="bold",
                color=BLACK, va="top", ha="left", linespacing=1.15, zorder=6)

    # Per-column chance level.
    for x, chance, lab_y, sgn, ha in ((xl, chance_wide, chance_lab_w, -1, "right"),
                                      (xr, chance_window, chance_lab_n, 1, "left")):
        ax.plot([x - 0.055, x + 0.055], [chance, chance],
                color=CHANCE_RED, lw=1.4, zorder=6)
        ax.annotate(f"{chance:.2f}  Chance level",
                    xy=(x + sgn * 0.06, chance),
                    xytext=(x + sgn * 0.11, lab_y),
                    fontsize=F_LAB - 0.5, color=CHANCE_RED,
                    va="center", ha=ha, zorder=7,
                    arrowprops=dict(arrowstyle="->", color=CHANCE_RED, lw=0.7,
                                    shrinkA=0, shrinkB=0),
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=CHANCE_RED, lw=0.6))

    # every point carries its value, so the axis itself would only add clutter
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.text(0.5, bottom + 0.055 * span,
            "Recovery error\n(grid steps - lower is better)",
            fontsize=F_CAP, fontweight="bold", color=BLACK,
            va="top", ha="center", linespacing=1.2)

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

    # Chance is experiment-specific: both grids are 10x10, but the two place
    # their ground truth differently, so the error a random cell would score is
    # not the same on both. (This previously reused the window level for the
    # wide bars, understating the wide chance line.)
    win_df = pd.read_csv(sorted(COMPARISON_DIR.glob("distances_*.csv"))[0])
    wide_df = pd.read_csv(sorted(WIDE_COMPARISON_DIR.glob("distances_*.csv"))[0])
    win_chance_joint, win_chance_axis = chance_grid_steps(win_df)
    wide_chance_joint, wide_chance_axis = _wide_chance(wide_df)
    print(f"\nMeasures: {len(win)} window / {len(wide)} wide")
    print(f"  window chance = {win_chance_joint:.2f} joint, "
          f"{win_chance_axis:.2f} per axis")
    print(f"  wide   chance = {wide_chance_joint:.2f} joint, "
          f"{wide_chance_axis:.2f} per axis\n")

    bars_figure(wide, wide_chance_axis, "Widely-spread targets",
                outdir / "fig_recovery_bars_wide.pdf", panel_letter="A")
    bars_figure(win, win_chance_axis, "Plausible window",
                outdir / "fig_recovery_bars_window.pdf", panel_letter="B")
    slopegraph(wide, win, wide_chance_joint, win_chance_joint,
               outdir / "fig_recovery_slopegraph.pdf")

    merged = wide.merge(win, on="measure", suffixes=("_wide", "_win"))
    merged["rank_wide"] = merged["grid_steps_wide"].rank().astype(int)
    merged["rank_win"] = merged["grid_steps_win"].rank().astype(int)
    merged.to_csv(outdir / "recovery_main_figure_scores.csv", index=False)
    print(merged[["name_wide", "grid_steps_wide", "rank_wide",
                  "grid_steps_win", "rank_win"]]
          .sort_values("grid_steps_win").to_string(index=False))


if __name__ == "__main__":
    main()
