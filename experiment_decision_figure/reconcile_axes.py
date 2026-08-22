"""
Reconciliation probe for the S4 decision-figure axis scalars.

Before building the decision figure we must know that each scalar we load or
recompute reproduces the numbers already published in the manuscript (pinned in
CLAUDE.md). This script computes candidate values for iSNR, total variation, CV
and timing from the cached landscape/timing files and prints them next to the
pinned references, so we can lock the correct definition per axis before wiring
them into the aggregation. It writes nothing.

Run: python reconcile_axes.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_benchmarking")
EXP = "105_distance_metrics_mst_animal_0"
MORPHO = ROOT / "output" / "gnm" / "hcp_schaefer_100_dataset" / EXP

# The 8 selected measures.
MEASURES = ["frobenius", "delta_con", "netrd_non_backtracking_spectral",
            "spectral_distance_adjacency", "communicability_corr", "portrait",
            "net_simile", "energy"]

# Measures stored as similarities -> inverted to distances (verbatim from the
# manuscript notebook's load_method_data).
INVERT = {"f1", "jaccard", "communicability_corr",
          "network_mutual_information", "dc_network_mutual_information"}

# Pinned reference values from CLAUDE.md (what we must reproduce).
PIN_ISNR = {"energy": 17.72, "portrait": 14.11, "communicability_corr": 7.75,
            "netrd_non_backtracking_spectral": 6.24}
PIN_CV = {"frobenius": 0.0130, "delta_con": 0.0161, "energy": 0.0422,
          "portrait": 0.0549}
PIN_TIMING = {"energy": 71.72, "netrd_non_backtracking_spectral": 67.48,
              "frobenius": 0.019}


# --- VisualQualityAnalyzer methods, verbatim from the manuscript notebook ----
def tv(matrix):
    m = np.array(matrix, dtype=float)
    gy = np.diff(m, axis=0)
    gx = np.diff(m, axis=1)
    return (np.sum(np.abs(gy)) + np.sum(np.abs(gx[:, :-1]))) / m.size


def snr(matrix, sigma=2):
    m = np.array(matrix, dtype=float)
    sm = gaussian_filter(m, sigma=sigma)
    noise = m - sm
    sp, npow = np.var(sm), np.var(noise)
    if npow < 1e-10:
        return 100.0
    return 10 * np.log10(sp / npow)


def load_landscape(measure, normalise_all=True):
    """Return (pivot_matrix, long_df) for a measure, mirroring load_method_data."""
    path = MORPHO / f"summary_indiv_{measure}_for_exp_{EXP}.csv"
    df = pd.read_csv(path)
    metric_cols = [c for c in df.columns
                   if c not in ("eta", "gamma", "network_index", "filename", "id")]
    v = df[metric_cols[0]].astype(float)
    if measure in INVERT:
        v = 1 - (v - v.min()) / (v.max() - v.min())
    df = df.assign(metric_value=v)
    # Per-cell mean across the 10 replicates.
    cell = df.groupby(["eta", "gamma"])["metric_value"].agg(["mean", "std", "count"])
    pivot = cell["mean"].reset_index().pivot(index="eta", columns="gamma",
                                             values="mean")
    return pivot, cell, df


def norm01(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def main():
    print(f"{'measure':32s} {'iSNR':>8} {'iSNR_n01':>9} {'TV':>8} "
          f"{'cv_land':>8} {'cv_repl':>8} {'timing_ms':>10}")
    rows = []
    for m in MEASURES:
        pivot, cell, longdf = load_landscape(m)
        mat = pivot.values

        isnr = snr(mat)
        isnr_n01 = snr(norm01(mat))       # if landscape was min-max normalised first
        totalvar = tv(norm01(mat))

        # CV candidate A: over the min-max-normalised landscape values.
        vals = norm01(cell["mean"].values)
        cv_land = np.nanstd(vals) / np.nanmean(vals)
        # CV candidate B: mean across cells of (replicate std / replicate mean).
        with np.errstate(divide="ignore", invalid="ignore"):
            per_cell_cv = cell["std"].values / cell["mean"].values
        cv_repl = np.nanmean(per_cell_cv[np.isfinite(per_cell_cv)])

        # Timing.
        tpath = MORPHO / f"timing_{m}_for_exp_{EXP}.csv"
        if not tpath.exists():
            # a couple of measures use a suffixed timing filename
            cand = list(MORPHO.glob(f"timing_{m}*_for_exp_{EXP}.csv"))
            tpath = cand[0] if cand else None
        timing_ms = np.nan
        if tpath and tpath.exists():
            td = pd.read_csv(tpath)
            tcols = [c for c in td.columns if c.startswith("time_")]
            timing_ms = float(np.mean(td[tcols].values)) * 1000

        print(f"{m:32s} {isnr:8.2f} {isnr_n01:9.2f} {totalvar:8.4f} "
              f"{cv_land:8.4f} {cv_repl:8.4f} {timing_ms:10.4f}")
        rows.append(dict(measure=m, isnr=isnr, isnr_n01=isnr_n01, tv=totalvar,
                         cv_land=cv_land, cv_repl=cv_repl, timing_ms=timing_ms))

    df = pd.DataFrame(rows).set_index("measure")

    def check(name, pinned, col):
        print(f"\n-- {name}: computed vs pinned (CLAUDE.md) --")
        for meas, ref in pinned.items():
            got = df.loc[meas, col] if meas in df.index else np.nan
            flag = "OK" if (not np.isnan(got) and abs(got - ref) <= 0.05 * abs(ref) + 1e-6) else "XX"
            print(f"   {meas:32s} got={got:10.4f}  pinned={ref:10.4f}  [{flag}]")

    check("iSNR (raw snr)", PIN_ISNR, "isnr")
    check("iSNR (norm01 snr)", PIN_ISNR, "isnr_n01")
    check("timing_ms", PIN_TIMING, "timing_ms")
    check("CV (landscape)", PIN_CV, "cv_land")
    check("CV (replicate)", PIN_CV, "cv_repl")


if __name__ == "__main__":
    main()
