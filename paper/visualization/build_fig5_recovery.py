"""
Figure 5 (parameter recovery + real-vs-artificial) as ONE figure
================================================================

Replaces the old route for this figure - three standalone panels from
`experiment_parameter_recovery_fine/run_recovery_main_figures.py`, two clipped
halves of the real-vs-artificial figure, and a hand-measured re-assembly in
`visualization/build_composites.py`. Panels drawn by different scripts at
different figure sizes ended up with mismatched font sizes and rows that did
not line up once scaled into their slots.

Here every panel is an axes of the same figure, so one font block governs all
text and the gridspec does the alignment:

    A  recovery error, widely-spread targets   (16 measures, eta/gamma bars)
    B  recovery error, plausible window        (16 measures, eta/gamma bars)
    C  closeness to the consensus: real subjects vs GNM networks (8 measures)
    D  discrimination AUC, full and hard case                    (8 measures)
    E  wide -> window slopegraph, measures placed at their error value

The numbers come from the same loaders the standalone scripts use, so nothing
can drift apart from the supplementary figures.

    conda activate ma_thesis
    python visualization/build_fig5_recovery.py
    python visualization/build_fig5_recovery.py --out /some/where/fig.pdf
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "experiment_parameter_recovery_fine",
           ROOT / "experiment_real_vs_artificial"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments_config import MANUSCRIPT_FIGURES  # noqa: E402
from experiments_config import (  # noqa: E402
    METHOD_NAMES, METRIC_COLORS, OUT_DIR as RVA_DIR,
)
from run_recovery_main_figures import (  # noqa: E402
    window_scores, wide_scores, _wide_chance, _declutter,
    COMPARISON_DIR, WIDE_COMPARISON_DIR, chance_grid_steps, UNSELECTED_COLOR,
)
from run_real_vs_artificial import load_d_art, _closeness, _clean  # noqa: E402

PAPER = MANUSCRIPT_FIGURES
DEFAULT_OUT = PAPER / "fig_5_recovery_error_2.pdf"

# ---------------------------------------------------------------------------
# One style block for the whole figure
# ---------------------------------------------------------------------------
FIG_W_CM, FIG_H_CM = 18, 11 # 19.0, 11.5

FS_TICK = 5.5      # axis ticks and measure names
FS_LABEL = 6.0     # axis labels
FS_TITLE = 7.0     # panel titles
FS_LETTER = 9.0    # panel letters
FS_ANN = 4.8       # in-panel value annotations and legends

DARK, GRAY, BLACK = "#333333", "#888888", "#000000"
CHANCE_RED = "#e8000b"
REAL_COLOR = (0.84, 0.19, 0.15)
ART_COLOR = (0.30, 0.45, 0.69)

# Panel E: when True, only the eight selected measures are labelled; the other
# eight keep their grey lines but lose their name/value text.
E_ONLY_SELECTED_LABELS = True # False

LW_AXIS = 0.6
LW_BAR_EDGE = 0.35


def cm(w, h):
    return (w / 2.54, h / 2.54)


def despine(ax, keep=("left", "bottom")):
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(sp in keep)
        ax.spines[sp].set_linewidth(LW_AXIS)


# ---------------------------------------------------------------------------
# A / B: paired eta / gamma bars
# ---------------------------------------------------------------------------

def bars_panel(ax, scores: pd.DataFrame, chance_axis: float, title: str,
               show_axis_marks: bool) -> None:
    """Two bars per measure - eta on top in the measure colour, gamma below in
    a lightened version of it. Measures outside the selected eight are drawn
    empty with grey labels; the red line is the per-axis chance level.
    """
    n = len(scores)
    y = np.arange(n)[::-1]
    h = 0.38

    selected = [m in METRIC_COLORS for m in scores["measure"]]
    base = [METRIC_COLORS.get(m, UNSELECTED_COLOR) for m in scores["measure"]]
    light = [tuple(1 - 0.45 * (1 - np.asarray(c))) for c in base]
    colors = [c if s else "white" for c, s in zip(base, selected)]
    light = [c if s else "white" for c, s in zip(light, selected)]

    x_max = float(max(scores[["eta_mean", "gamma_mean"]].max())) * 1.28

    for yi, ce, cg, sel, (_, row) in zip(y, colors, light, selected,
                                         scores.iterrows()):
        ax.barh(yi + h / 2, row["eta_mean"], height=h, color=ce,
                edgecolor=GRAY, linewidth=LW_BAR_EDGE, zorder=2)
        ax.barh(yi - h / 2, row["gamma_mean"], height=h, color=cg,
                edgecolor=GRAY, linewidth=LW_BAR_EDGE, zorder=2)
        # variation across the ground-truth networks: the bar end is the mean,
        # so only the outward half is drawn, from the mean out to mean + 1 SD
        for yy, mean, sd, cc in ((yi + h / 2, row["eta_mean"], row["eta_sd"], ce),
                                 (yi - h / 2, row["gamma_mean"], row["gamma_sd"], cg)):
            mark = cc if sel else GRAY
            hi = min(x_max, mean + sd)
            ax.plot([mean, hi], [yy, yy], color=mark, lw=0.4, zorder=5,
                    solid_capstyle="butt")
            ax.plot([hi, hi], [yy - 0.09, yy + 0.09], color=mark, lw=0.4, zorder=5)

    # chance line, labelled at its foot: on the same x scale as the bars, and
    # clear of both the bars and the x label
    ax.axvline(chance_axis, color=CHANCE_RED, ls="-", lw=0.9, zorder=1)
    ax.text(chance_axis, -1.30, "chance", fontsize=FS_ANN, color=CHANCE_RED,
            va="bottom", ha="center", zorder=6)

    # which bar is which axis; the empty band above the top row keeps the
    # legend off the data
    if show_axis_marks:
        ax.legend(handles=[
            Patch(facecolor="#6f6f6f", edgecolor=GRAY, linewidth=LW_BAR_EDGE,
                  label=r"$\eta$ (upper bar)"),
            Patch(facecolor="#cfcfcf", edgecolor=GRAY, linewidth=LW_BAR_EDGE,
                  label=r"$\gamma$ (lower bar)"),
        ], loc="upper right", frameon=False, fontsize=FS_ANN, handlelength=1.1,
            handleheight=0.7, handletextpad=0.5, borderpad=0.0, labelspacing=0.25)

    ax.set_yticks(y)
    ax.set_yticklabels(scores["name"], fontsize=FS_TICK)
    for lab, sel in zip(ax.get_yticklabels(), selected):
        lab.set_color(DARK if sel else GRAY)
    ax.set_ylim(-1.5, n + 1.8)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Mean absolute index error, per axis\n(grid steps)",
                  fontsize=FS_LABEL, linespacing=1.2)
    ax.tick_params(axis="x", labelsize=FS_TICK, length=2, width=LW_AXIS, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", color=BLACK, pad=4)
    despine(ax, keep=("left", "bottom"))


# ---------------------------------------------------------------------------
# C / D: real subjects vs GNM networks
# ---------------------------------------------------------------------------

def _normalise_pooled(s_real, s_art):
    lo = min(s_real.min(), s_art.min())
    hi = max(s_real.max(), s_art.max())
    if hi - lo < 1e-12:
        return s_real * 0.0, s_art * 0.0
    return (s_real - lo) / (hi - lo), (s_art - lo) / (hi - lo)


def violins_panel(ax, res: pd.DataFrame, d_real: pd.DataFrame) -> None:
    """Per measure: GNM networks as a half-violin, real subjects as jittered
    points, both on a pooled min-max normalised closeness scale."""
    order = list(res["measure"])
    positions = np.arange(len(order))[::-1]
    rng = np.random.default_rng(0)

    for pos, m in zip(positions, order):
        s_real, _ = _clean(_closeness(d_real[m].to_numpy(dtype=float), m))
        s_art, _ = _clean(_closeness(load_d_art(m)["raw"].to_numpy(dtype=float), m))
        nr, na = _normalise_pooled(s_real, s_art)

        parts = ax.violinplot([na], positions=[pos], vert=False,
                              showextrema=False, widths=0.9)
        for body in parts["bodies"]:
            body.set_facecolor(ART_COLOR)
            body.set_edgecolor("none")
            body.set_alpha(0.45)
            # clip to the lower half -> half-violin below the row baseline
            verts = body.get_paths()[0].vertices
            verts[:, 1] = np.clip(verts[:, 1], -np.inf, pos)
        jitter = rng.uniform(0.08, 0.55, size=len(nr))
        ax.scatter(nr, pos + jitter, s=1.6, color=REAL_COLOR, alpha=0.8,
                   linewidths=0, zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels(res["name"], fontsize=FS_TICK, color=DARK)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.8, len(order) + 1.0)
    ax.set_xlabel("Closeness to consensus\n(pooled min-max normalised)",
                  fontsize=FS_LABEL, linespacing=1.2)
    ax.tick_params(axis="x", labelsize=FS_TICK, length=2, width=LW_AXIS, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_title("Real subjects vs GNM networks", fontsize=FS_TITLE,
                 fontweight="bold", color=BLACK, pad=4)
    ax.legend(handles=[
        Patch(facecolor=ART_COLOR, alpha=0.45, label="GNM networks (25,000)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REAL_COLOR,
               markersize=2.5, label="real subjects (100)"),
    ], loc="upper left", frameon=False, fontsize=FS_ANN, handlelength=1.2,
        handletextpad=0.5, borderpad=0.0, labelspacing=0.3)
    despine(ax, keep=("left", "bottom"))


def auc_panel(ax, res: pd.DataFrame, n_hard: int) -> None:
    """AUC over all GNM networks as bars, the hard case as a diamond."""
    y = np.arange(len(res))[::-1]
    colors = [METRIC_COLORS.get(m, GRAY) for m in res["measure"]]
    ax.barh(y, res["auc"], color=colors, edgecolor="black",
            linewidth=LW_BAR_EDGE, height=0.7, zorder=2)
    ax.scatter(res["auc_hardcase"], y, marker="D", s=5, color="black", zorder=4)
    ax.axvline(0.5, color=GRAY, lw=0.7, ls="--", zorder=1)

    x_lo = min(0.45, float(res[["auc", "auc_hardcase"]].min().min()) - 0.05)
    ax.set_xlim(x_lo, 1.12)
    for yi, (a, hard) in zip(y, zip(res["auc"], res["auc_hardcase"])):
        ax.text(a + 0.022, yi, f"{a:.3f}", fontsize=FS_ANN, va="center",
                ha="left", color=DARK)
        # the hard-case number sits beside its own diamond, haloed so it stays
        # legible on any bar colour; where the hard case equals the full AUC the
        # diamond sits on the bar end and the number would duplicate it
        if abs(hard - a) > 0.02:
            left = hard - x_lo < 0.12          # flip inward near the axis start
            ax.text(hard + (0.016 if left else -0.016), yi, f"{hard:.2f}",
                    fontsize=FS_ANN, va="center", ha="left" if left else "right",
                    color=BLACK, zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.1, foreground="white")])

    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.8, len(res) + 1.0)
    ax.set_xlabel("AUC\n(P[real subject closer to consensus than GNM])",
                  fontsize=FS_LABEL, linespacing=1.2)
    ax.tick_params(axis="x", labelsize=FS_TICK, length=2, width=LW_AXIS, pad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Discrimination", fontsize=FS_TITLE, fontweight="bold",
                 color=BLACK, pad=4)
    ax.legend(handles=[
        Patch(facecolor="white", edgecolor="black", linewidth=LW_BAR_EDGE,
              label="all GNM networks"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="black",
               markersize=2.5, label=f"hard case ({n_hard} closest)"),
    ], loc="upper left", frameon=False, fontsize=FS_ANN, handlelength=1.2,
        handletextpad=0.5, borderpad=0.0, labelspacing=0.3)
    despine(ax, keep=("left", "bottom"))


# ---------------------------------------------------------------------------
# E: value-spaced slopegraph
# ---------------------------------------------------------------------------

def slopegraph_panel(ax, wide: pd.DataFrame, window: pd.DataFrame,
                     chance_wide: float, chance_window: float) -> None:
    """Wide -> window, every measure placed at its actual error value, so
    vertical distance carries meaning and a crossing line is a measure whose
    standing changes between the two regimes. Each column carries its own
    chance level."""
    m = wide.merge(window, on="measure", suffixes=("_wide", "_win"))
    xl, xr = 0.0, 1.0
    vals_w = m["grid_steps_wide"].to_numpy()
    vals_n = m["grid_steps_win"].to_numpy()
    lo = min(vals_w.min(), vals_n.min(), chance_wide, chance_window)
    hi = max(vals_w.max(), vals_n.max(), chance_wide, chance_window)
    span = hi - lo

    # declutter each column WITH its chance label, so the red label cannot land
    # on top of a measure label
    min_gap = 0.030 * span
    labelled = np.array([mm in METRIC_COLORS or not E_ONLY_SELECTED_LABELS
                         for mm in m["measure"]])

    def _lab_positions(vals, chance):
        """Declutter only the labels that get drawn, plus the chance label."""
        out = vals.copy()
        moved = _declutter(np.r_[vals[labelled], chance], min_gap)
        out[labelled] = moved[:-1]
        return out, moved[-1]

    lab_w, chance_lab_w = _lab_positions(vals_w, chance_wide)
    lab_n, chance_lab_n = _lab_positions(vals_n, chance_window)

    for (_, row), lw_, ln_ in zip(m.iterrows(), lab_w, lab_n):
        c = METRIC_COLORS.get(row["measure"], UNSELECTED_COLOR)
        sel = row["measure"] in METRIC_COLORS
        ax.plot([xl, xr], [row["grid_steps_wide"], row["grid_steps_win"]],
                color=c if sel else "#d8d8d8", lw=1.4 if sel else 0.7,
                zorder=3 if sel else 2, solid_capstyle="round")
        for x, v in ((xl, row["grid_steps_wide"]), (xr, row["grid_steps_win"])):
            ax.plot([x], [v], marker="o", ms=3.4 if sel else 2.0,
                    color=c if sel else "#d8d8d8",
                    markeredgecolor="white" if sel else "none",
                    markeredgewidth=0.5, zorder=4)

        if not sel and E_ONLY_SELECTED_LABELS:
            continue

        # both columns carry the measure name, so each side reads on its own
        for x, v, lab, ha, sgn, text in (
            (xl, row["grid_steps_wide"], lw_, "right", -1,
             f"{row['name_wide']}  {row['grid_steps_wide']:.2f}"),
            (xr, row["grid_steps_win"], ln_, "left", 1,
             f"{row['grid_steps_win']:.2f}  {row['name_wide']}"),
        ):
            if abs(lab - v) > 1e-9:      # leader line only where a label moved
                ax.plot([x + sgn * 0.03, x + sgn * 0.09], [v, lab],
                        color="#bbbbbb", lw=0.35, zorder=1)
            ax.text(x + sgn * 0.11, lab, text, fontsize=FS_TICK,
                    color=DARK if sel else "#b0b0b0", va="center", ha=ha, zorder=5)

    ax.set_xlim(-2.1, 3.1)
    top = lo - 0.13 * span
    bottom = hi + 0.07 * span
    ax.set_ylim(bottom, top)                          # inverted: better is up

    for x, head in ((xl, "Widely-spread\nrange"), (xr, "Plausible\nwindow")):
        ax.annotate("", xy=(x, bottom), xytext=(x, top),
                    arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.0,
                                    shrinkA=0, shrinkB=0), zorder=1)
        sgn = -1 if x == xl else 1
        ax.text(x + sgn * 0.06, top, head, fontsize=FS_LABEL, fontweight="bold",
                color=BLACK, va="top", ha="right" if sgn < 0 else "left",
                linespacing=1.15, zorder=6)

    for x, chance, lab_y, sgn, ha in ((xl, chance_wide, chance_lab_w, -1, "right"),
                                      (xr, chance_window, chance_lab_n, 1, "left")):
        ax.plot([x - 0.055, x + 0.055], [chance, chance],
                color=CHANCE_RED, lw=1.1, zorder=6)
        ax.annotate(f"{chance:.2f}  Chance level",
                    xy=(x + sgn * 0.06, chance),
                    xytext=(x + sgn * 0.11, lab_y),
                    fontsize=FS_ANN, color=CHANCE_RED, va="center", ha=ha, zorder=7,
                    arrowprops=dict(arrowstyle="->", color=CHANCE_RED, lw=0.6,
                                    shrinkA=0, shrinkB=0),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=CHANCE_RED, lw=0.5))

    # every point carries its value, so the axis itself would only add clutter
    ax.set_yticks([]); ax.set_xticks([])
    despine(ax, keep=())
    ax.text(0.5, bottom + 0.05 * span,
            "Recovery error\n(grid steps - lower is better)",
            fontsize=FS_LABEL, fontweight="bold", color=BLACK,
            va="top", ha="center", linespacing=1.2)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def add_letters(fig, pairs) -> None:
    """Panel letters at the top-left of each panel's full extent (labels
    included), so they sit at a consistent offset instead of a measured one."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in pairs:
        bb = ax.get_tightbbox(r).transformed(inv)
        fig.text(bb.x0 - 0.015, bb.y1 + 0.012, letter, fontsize=FS_LETTER,
                 fontweight="bold", color=BLACK, va="bottom", ha="left")


def build(out_pdf: Path) -> None:
    try:
        from vizman import viz
        viz.set_visual_style(font_family="Arial")
    except Exception:
        pass
    # set_visual_style may carry its own sizes; the panels set every size they
    # use explicitly, and this fixes the few that are inherited
    plt.rcParams.update({"font.size": FS_TICK, "axes.linewidth": LW_AXIS,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    # ---- recovery scores + the two chance levels ---------------------------
    win = window_scores()
    wide = wide_scores()
    win_df = pd.read_csv(sorted(COMPARISON_DIR.glob("distances_*.csv"))[0])
    wide_df = pd.read_csv(sorted(WIDE_COMPARISON_DIR.glob("distances_*.csv"))[0])
    win_chance_joint, win_chance_axis = chance_grid_steps(win_df)
    wide_chance_joint, wide_chance_axis = _wide_chance(wide_df)

    # ---- real vs artificial ------------------------------------------------
    res = pd.read_csv(RVA_DIR / "real_vs_artificial_results.csv")
    res = res.sort_values(["auc", "auc_hardcase"], ascending=False).reset_index(drop=True)
    res["name"] = [METHOD_NAMES.get(m, n) for m, n in zip(res["measure"], res["name"])]
    d_real = pd.read_csv(RVA_DIR / "d_real_loo.csv")
    n_hard = int(res["n_hard_used"].max())

    fig = plt.figure(figsize=cm(FIG_W_CM, FIG_H_CM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.05],
                          height_ratios=[1.45, 1.0],
                          left=0.10, right=0.995, top=0.93, bottom=0.075,
                          wspace=0.75, hspace=0.62)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[:, 2])

    bars_panel(axA, wide, wide_chance_axis, "Widely-spread targets", True)
    bars_panel(axB, win, win_chance_axis, "Plausible window", True)
    violins_panel(axC, res, d_real)
    auc_panel(axD, res, n_hard)
    slopegraph_panel(axE, wide, win, wide_chance_joint, win_chance_joint)

    add_letters(fig, [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"), (axE, "E")])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved -> {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only-selected-labels", action="store_true",
                    help="panel E: label only the eight selected measures")
    args = ap.parse_args()
    global E_ONLY_SELECTED_LABELS
    E_ONLY_SELECTED_LABELS = args.only_selected_labels
    build(Path(args.out))


if __name__ == "__main__":
    main()
