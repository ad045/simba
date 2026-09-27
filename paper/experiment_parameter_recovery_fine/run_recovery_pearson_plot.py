"""
Pearson recovery accuracy, wide targets vs plausible window
===========================================================

Companion to run_fine_recovery_paper_plot_concise.py. That figure scores recovery
in grid steps; this one scores it as the Pearson correlation between the true and
the recovered parameter, for BOTH experiments, on the same axes.

  A  Wide targets   (5 widely-spread ground-truth combos, coarse 10 x 10 grid)
  B  Plausible window (100 draws from the morphospace-derived window, fine grid)

Both panels: one bar pair per measure, r(eta) and r(gamma), ordered by the mean
of the two within the panel. Nothing is recomputed - both read the cached
per-measure distance CSVs.

Read the wide panel with care. Its ground truth is only 5 distinct (eta, gamma)
combinations, four of which sit at grid corners, so r there largely reports
whether a measure tells the corners apart; it saturates easily and its spread is
not comparable in resolution to panel B, which has 100 distinct targets. The
grid-step figure remains the score the two experiments are ranked against each
other on.

Usage
-----
  python run_recovery_pearson_plot.py
  python run_recovery_pearson_plot.py --out /path/to/fig.pdf
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vizman import viz

_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = _THIS_DIR.parent
for _p in (str(ROOT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_config import fine as cfg, METHOD_NAMES, METRIC_COLORS  # noqa: E402
from run_synthetic_gnm_fine import (  # noqa: E402
    METRIC_NAMES, WIDE_COMPARISON_DIR, _pearson,
)
from run_fine_recovery_paper_plot_concise import COMPARISON_DIR, UNSELECTED_COLOR  # noqa: E402

DEFAULT_OUT = (ROOT_DIR / "output" / "gnm" /
               "synthetic_parameter_recovery_fine" / "plots" /
               "fig_recovery_pearson.pdf")

DARK, GRAY = "#333333", "#888888"
FS = 7
HATCH = "//////"

N_BOOT = 2000      # bootstrap resamples for the CI on r
CI = (2.5, 97.5)   # percentile interval reported as the error bar
BOOT_SEED = 0


def boot_ci(true: np.ndarray, rec: np.ndarray,
            n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """r plus a percentile bootstrap CI, resampling ground-truth networks.

    The CI is the sampling spread of r over the ground-truth draws - it says how
    much r would move had different networks been drawn, which is the relevant
    uncertainty here. Resamples in which either side is constant give an
    undefined r and are dropped; if too few survive, the CI is undefined.
    """
    m = np.isfinite(true) & np.isfinite(rec)
    true, rec = true[m], rec[m]
    if len(true) < 3 or true.std() == 0 or rec.std() == 0:
        return float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(true, rec)[0, 1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(true), size=(n_boot, len(true)))
    t, p = true[idx], rec[idx]
    # vectorised Pearson per resample; zero-variance rows -> nan via 0/0
    with np.errstate(invalid="ignore", divide="ignore"):
        tc = t - t.mean(1, keepdims=True)
        pc = p - p.mean(1, keepdims=True)
        boot = (tc * pc).sum(1) / np.sqrt((tc ** 2).sum(1) * (pc ** 2).sum(1))
    boot = boot[np.isfinite(boot)]
    if len(boot) < n_boot // 10:
        return r, float("nan"), float("nan")
    lo, hi = np.percentile(boot, CI)
    return r, float(lo), float(hi)


def _scores_from(comparison_dir: Path, rec_prefix: str, valid_col: str,
                 tag: str) -> pd.DataFrame:
    """r + CI per measure from one experiment's cached distance CSVs."""
    rows = []
    for path in sorted(comparison_dir.glob("distances_*.csv")):
        measure = path.stem.replace("distances_", "")
        df = pd.read_csv(path)
        if valid_col not in df:
            continue
        valid = df[df[valid_col] >= 0]
        row = {"measure": measure,
               "name": METHOD_NAMES.get(measure, METRIC_NAMES.get(measure, measure)),
               f"{tag}_n": len(valid),
               f"{tag}_n_true_combos": int(
                   valid[["true_eta", "true_gamma"]].drop_duplicates().shape[0])}
        for axis in ("eta", "gamma"):
            r, lo, hi = boot_ci(valid[f"true_{axis}"].to_numpy(float),
                                valid[f"{rec_prefix}_{axis}"].to_numpy(float))
            row[f"{tag}_r_{axis}"] = r
            row[f"{tag}_lo_{axis}"] = lo
            row[f"{tag}_hi_{axis}"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def collect() -> pd.DataFrame:
    """One row per measure with r(eta), r(gamma) and CIs for both experiments."""
    fine = _scores_from(COMPARISON_DIR, "recovered", "recovered_grid_idx", "fine")
    wide = _scores_from(WIDE_COMPARISON_DIR, "predicted", "predicted_grid_idx", "wide")
    if wide.empty:
        raise FileNotFoundError(
            "No wide-target results found - run experiment_parameter_recovery first.")
    df = fine.merge(wide.drop(columns=["name"]), on="measure", how="outer")
    df["name"] = df["name"].fillna(
        df["measure"].map(lambda m: METHOD_NAMES.get(m, METRIC_NAMES.get(m, m))))
    return df


def _panel(ax, df, tag, title, letter):
    """Horizontal r(eta)/r(gamma) bar pairs, best (highest mean r) at the top.

    Error bars are the percentile bootstrap CI over ground-truth networks.
    """
    r_eta_col, r_gamma_col = f"{tag}_r_eta", f"{tag}_r_gamma"
    d = df.copy()
    d["_mean"] = d[[r_eta_col, r_gamma_col]].mean(axis=1)
    # NaN (constant recovery) sorts last, and is drawn as a marker, not a bar.
    d = d.sort_values("_mean", ascending=False, na_position="last").reset_index(drop=True)

    y = np.arange(len(d))[::-1]
    h = 0.38
    colors = [METRIC_COLORS.get(m, UNSELECTED_COLOR) for m in d["measure"]]
    light = [tuple(1 - 0.45 * (1 - np.asarray(c))) for c in colors]
    selected = [m in METRIC_COLORS for m in d["measure"]]
    colors = [c if s else "white" for c, s in zip(colors, selected)]
    hatches = [None] * len(colors)

    for i, yi in enumerate(y):
        row = d.iloc[i]
        ce, cg, hh = colors[i], light[i], hatches[i]
        for offset, axis, col in ((+h / 2, "eta", ce), (-h / 2, "gamma", cg)):
            val = row[f"{tag}_r_{axis}"]
            if pd.isna(val):
                ax.plot(0, yi + offset, marker="x", ms=3, color=GRAY, mew=0.8)
                continue
            ax.barh(yi + offset, val, height=h, color=col,
                    edgecolor=GRAY, linewidth=0.3, hatch=hh)
            lo, hi = row[f"{tag}_lo_{axis}"], row[f"{tag}_hi_{axis}"]
            if not (pd.isna(lo) or pd.isna(hi)):
                ax.errorbar(val, yi + offset,
                            xerr=[[max(val - lo, 0)], [max(hi - val, 0)]],
                            fmt="none", ecolor=DARK, elinewidth=0.35,
                            capsize=1.0, capthick=0.35)

    ax.axvline(0, color=DARK, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["name"], fontsize=FS)
    for lab, sel in zip(ax.get_yticklabels(), selected):
        lab.set_color(DARK if sel else GRAY)
    ax.set_xlabel("Pearson $r$ (true vs recovered)", fontsize=FS)
    lo_cols = [f"{tag}_lo_eta", f"{tag}_lo_gamma"]
    hi_cols = [f"{tag}_hi_eta", f"{tag}_hi_gamma"]
    x_lo = float(np.nanmin(d[lo_cols + [r_eta_col, r_gamma_col]].values))
    x_hi = float(np.nanmax(d[hi_cols + [r_eta_col, r_gamma_col]].values))
    ax.set_xlim(min(-0.55, x_lo - 0.08), max(1.0, x_hi + 0.08))
    ax.tick_params(axis="x", labelsize=FS)
    ax.set_title(f"{letter}", loc="left", fontsize=10, fontweight="bold", pad=12)
    # right-aligned, so it never collides with the bold panel letter on the left
    ax.text(1.0, 1.02, title, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS, color=DARK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return d


def make_figure(df: pd.DataFrame, out_pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    matplotlib.rcParams["hatch.linewidth"] = 0.25
    fig, (axA, axB) = plt.subplots(1, 2, figsize=viz.cm_to_inch((18, 12)))

    n_wide = int(df["wide_n_true_combos"].dropna().iloc[0])
    n_fine = int(df["fine_n"].dropna().iloc[0])
    _panel(axA, df, "wide", f"Wide targets ({n_wide} combos)", "A")
    ordered = _panel(axB, df, "fine", f"Plausible window ({n_fine} draws)", "B")

    axA.legend(handles=[
        Patch(facecolor=GRAY, edgecolor=DARK, linewidth=0.3, label=r"$\eta$ axis"),
        Patch(facecolor="#c8c8c8", edgecolor=DARK, linewidth=0.3, label=r"$\gamma$ axis"),
    ], loc="upper center", bbox_to_anchor=(1.05, -0.09), ncol=2,
       frameon=False, fontsize=FS)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_pdf}")
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    df = collect()
    print(f"\nWindow: eta in {cfg.SAMPLE_ETA_RANGE}, gamma in {cfg.SAMPLE_GAMMA_RANGE}")
    print(f"Error bars: {int((CI[1]-CI[0]))}% percentile bootstrap "
          f"({N_BOOT} resamples over ground-truth networks)\n")
    print(f"{'measure':26s} {'wide r(eta)':>20s} {'wide r(gam)':>20s} "
          f"{'win r(eta)':>20s} {'win r(gam)':>20s}")
    for _, r in df.sort_values("fine_r_eta", ascending=False).iterrows():
        def f(tag, axis):
            v, lo, hi = (r[f"{tag}_r_{axis}"], r[f"{tag}_lo_{axis}"], r[f"{tag}_hi_{axis}"])
            if pd.isna(v):
                return "n/a"
            ci = "" if pd.isna(lo) else f" [{lo:5.2f},{hi:5.2f}]"
            return f"{v:6.3f}{ci}"
        print(f"  {r['name']:24s} {f('wide','eta'):>20s} {f('wide','gamma'):>20s} "
              f"{f('fine','eta'):>20s} {f('fine','gamma'):>20s}")

    for a, b, label in (("wide_r_eta", "fine_r_eta", "r(eta)"),
                        ("wide_r_gamma", "fine_r_gamma", "r(gamma)")):
        sub = df[[a, b]].dropna()
        tau = sub.corr(method="kendall").iloc[0, 1] if len(sub) > 2 else float("nan")
        print(f"Kendall tau, wide vs window on {label:9s} = {tau:.3f} (n={len(sub)})")

    make_figure(df, Path(args.out))
    df.to_csv(Path(args.out).parent / "recovery_pearson_scores.csv", index=False)


if __name__ == "__main__":
    main()
