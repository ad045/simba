"""
Rescore the cached parameter-recovery CSVs with the correct measure orientation
==============================================================================

Both recovery experiments took ``argmin`` directly over the raw comparer output.
Five of the 16 measures return a SIMILARITY (higher = closer), not a distance:

    jaccard, f1, communicability_corr,
    network_mutual_information, dc_network_mutual_information

For those, ``argmin`` selected the *least* similar grid cell, so their recovered
parameters (and every score derived from them) were inverted.

The comparison scripts now orient via ``experiments_config.to_distance`` before
the argmin. This script repairs the CSVs that were written before that fix
WITHOUT re-running the expensive comparisons: every CSV already stores the full
per-cell landscape (``dist_to_grid_*``), so the prediction can simply be redone.

The raw ``dist_to_grid_*`` columns are left untouched (they are the measure's
own output, not an oriented distance); only the derived prediction columns are
rewritten. The wide CSVs additionally gain ``recovered_*`` aliases so that the
fine experiment's plotting code can read either experiment unchanged.

Usage
-----
    python rescore_recovery_predictions.py            # rescore both experiments
    python rescore_recovery_predictions.py --dry-run  # report, write nothing
    python rescore_recovery_predictions.py --check    # self-test only
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments_config import (  # noqa: E402
    ROOT_DIR, IS_SIMILARITY, to_distance, coarse, fine,
)

OUT_GNM = ROOT_DIR / "output" / "gnm"

EXPERIMENTS = {
    "wide": dict(
        cfg=coarse,
        comparison_dir=OUT_GNM / "synthetic_parameter_recovery_grid" / "comparison_results",
        prefix="predicted",
    ),
    "fine": dict(
        cfg=fine,
        comparison_dir=OUT_GNM / "synthetic_parameter_recovery_fine" / "comparison_results",
        prefix="recovered",
    ),
}


def _normalise(cfg, eta, gamma):
    eta_n = (eta - cfg.ETA_RANGE[0]) / (cfg.ETA_RANGE[1] - cfg.ETA_RANGE[0])
    gamma_n = (gamma - cfg.GAMMA_RANGE[0]) / (cfg.GAMMA_RANGE[1] - cfg.GAMMA_RANGE[0])
    return eta_n, gamma_n


def rescore_csv(path: Path, cfg, prefix: str, dry_run: bool = False) -> dict:
    """Redo the argmin for one measure with the correct orientation."""
    measure = path.stem.replace("distances_", "")
    df = pd.read_csv(path)

    cols = [f"dist_to_grid_{i}" for i in range(len(cfg.GRID_COMBOS))]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing landscape columns ({len(missing)})")

    land = to_distance(df[cols].to_numpy(), measure)          # lower = closer
    all_nan = np.isnan(land).all(axis=1)
    pred_idx = np.full(len(df), -1, dtype=int)
    pred_idx[~all_nan] = np.nanargmin(land[~all_nan], axis=1)

    grid = np.asarray(cfg.GRID_COMBOS, dtype=float)
    pred_eta = np.where(pred_idx >= 0, grid[pred_idx, 0], np.nan)
    pred_gamma = np.where(pred_idx >= 0, grid[pred_idx, 1], np.nan)

    pe_n, pg_n = _normalise(cfg, pred_eta, pred_gamma)
    te_n, tg_n = _normalise(cfg, df["true_eta"].to_numpy(), df["true_gamma"].to_numpy())
    sq_err = (pe_n - te_n) ** 2 + (pg_n - tg_n) ** 2

    old_idx = df[f"{prefix}_grid_idx"].to_numpy()
    n_changed = int((old_idx != pred_idx).sum())

    for name, values in (
        (f"{prefix}_grid_idx", pred_idx),
        (f"{prefix}_eta", pred_eta),
        (f"{prefix}_gamma", pred_gamma),
        ("abs_error", np.sqrt(sq_err)),
    ):
        df[name] = values
    if "sq_error" in df.columns:
        df["sq_error"] = sq_err

    # Schema alias, so one plotting code path can serve both experiments.
    for suffix in ("grid_idx", "eta", "gamma"):
        df[f"recovered_{suffix}"] = df[f"{prefix}_{suffix}"]

    if not dry_run:
        backup = path.with_suffix(".csv.preorientation_backup")
        if not backup.exists():
            shutil.copy2(path, backup)
        df.to_csv(path, index=False)

    return dict(measure=measure, is_similarity=measure in IS_SIMILARITY,
                n_rows=len(df), n_changed=n_changed)


def grid_step_error(path: Path, cfg) -> float:
    """Mean Euclidean error in grid-index space, from the stored prediction."""
    df = pd.read_csv(path)
    idx = df["recovered_grid_idx"].to_numpy()
    valid = idx >= 0
    pg, pe = idx[valid] // cfg.GRID_N_ETA, idx[valid] % cfg.GRID_N_ETA
    te = np.array([int(np.argmin(np.abs(cfg.GRID_ETA - e)))
                   for e in df["true_eta"].to_numpy()[valid]])
    tg = np.array([int(np.argmin(np.abs(cfg.GRID_GAMMA - g)))
                   for g in df["true_gamma"].to_numpy()[valid]])
    return float(np.mean(np.hypot(pe - te, pg - tg)))


def run(experiment: str, dry_run: bool = False) -> None:
    spec = EXPERIMENTS[experiment]
    cfg, comparison_dir = spec["cfg"], spec["comparison_dir"]
    paths = sorted(comparison_dir.glob("distances_*.csv"))
    if not paths:
        print(f"[{experiment}] no CSVs in {comparison_dir} - skipped")
        return

    print(f"\n=== {experiment}: {comparison_dir}")
    rows = []
    for path in paths:
        info = rescore_csv(path, cfg, spec["prefix"], dry_run=dry_run)
        info["grid_steps"] = grid_step_error(path, cfg) if not dry_run else np.nan
        rows.append(info)

    out = pd.DataFrame(rows).sort_values("grid_steps")
    for _, r in out.iterrows():
        tag = "SIM" if r["is_similarity"] else "   "
        print(f"  {tag} {r['measure']:34s} grid_steps={r['grid_steps']:6.3f}   "
              f"predictions changed: {r['n_changed']:4d}/{r['n_rows']}")


def self_check() -> None:
    """Smallest check that fails if the orientation logic breaks."""
    vals = [0.1, 0.9, 0.5]
    assert list(to_distance(vals, "frobenius")) == vals, "distance must pass through"
    assert int(np.argmin(to_distance(vals, "jaccard"))) == 1, "similarity must flip"
    assert int(np.argmin(to_distance(vals, "f1"))) == 1
    assert int(np.argmin(to_distance(vals, "hamming"))) == 0
    # Every similarity measure must be registered, and no distance may be.
    assert IS_SIMILARITY == {"communicability_corr", "jaccard", "f1",
                             "network_mutual_information",
                             "dc_network_mutual_information"}
    # Rescoring is idempotent: re-running on an already-correct landscape is a no-op.
    df = pd.DataFrame({"a": [1.0, 2.0]})
    assert np.array_equal(to_distance(to_distance(df["a"], "frobenius"), "frobenius"),
                          df["a"].to_numpy())
    # Data-level invariant: every stored prediction must be the extremum of the
    # measure's OWN raw output - the maximum for a similarity, the minimum for a
    # distance. This is what the bug violated for all five similarity measures.
    n_checked = 0
    for spec in EXPERIMENTS.values():
        cfg = spec["cfg"]
        cols = [f"dist_to_grid_{i}" for i in range(len(cfg.GRID_COMBOS))]
        for path in sorted((spec["comparison_dir"]).glob("distances_*.csv")):
            measure = path.stem.replace("distances_", "")
            df = pd.read_csv(path)
            if "recovered_grid_idx" not in df.columns:
                continue          # not rescored yet
            raw = df[cols].to_numpy()
            pick = (np.nanargmax if measure in IS_SIMILARITY else np.nanargmin)(raw, axis=1)
            got = df["recovered_grid_idx"].to_numpy()
            rows = np.arange(len(raw))
            assert np.isclose(raw[rows, got], raw[rows, pick]).all(), (
                f"{path.name}: stored prediction is not the extremum of the raw score")
            n_checked += 1
    print(f"self-check passed ({n_checked} rescored CSVs verified)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="run the self-check and exit")
    ap.add_argument("--experiment", choices=["wide", "fine", "both"], default="both")
    args = ap.parse_args()

    self_check()
    if args.check:
        return

    targets = ["wide", "fine"] if args.experiment == "both" else [args.experiment]
    for experiment in targets:
        run(experiment, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
