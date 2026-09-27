"""
Score all 16 benchmarked measures on the 10%-density morphospace
================================================================

Drives `run_connectome_comparisons.main` once per measure over the networks
written by `generate_morphospace.py`, comparing each against the 495-edge HCP
consensus - so both sides of every comparison now sit at 10% density.

Writes `summary_indiv_<measure>_for_exp_<EXP_NAME>.csv` and the matching
`timing_<measure>_...csv` into the run's output directory, the same files the
figure and analysis scripts read.

`PathConfig` anchors on the working directory, so this changes into the paper/
folder and puts it and its `src/` on sys.path before importing the runner.

Usage
-----
    conda activate ma_thesis
    python pipeline_gnms_and_benchmarking/score_morphospace.py --list        # what would run
    python pipeline_gnms_and_benchmarking/score_morphospace.py --only frobenius energy
    python pipeline_gnms_and_benchmarking/score_morphospace.py               # all 16

Each measure is skipped if its summary file already holds every network, so the
run is safe to interrupt and restart.
"""

import multiprocessing as mp
import os
import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_NAME = "106_distance_metrics_mst_animal_0_density10"
DATASET = "hcp_schaefer_100_dataset"
N_NETWORKS = 25_000

# The 16 measures of the manuscript, as `evaluation_mode` tokens.
# Ordered cheapest-looking first, so a failure surfaces early.
MEASURES = [
    "frobenius",
    "hamming",
    "jaccard",
    "f1",
    "delta_con",
    "communicability_corr",
    "communicability_jsd",
    "spectral_distance_adjacency",
    "spectral_distance_norm_laplacian",
    "network_mutual_information",
    "dc_network_mutual_information",
    "portrait",
    "net_simile",
    "resistance",
    "netrd_non_backtracking_spectral",
    "energy",
]

# The four KS criteria that the energy's MaxCriteria aggregates. Not part of the
# benchmarked sixteen; the appendix figure on energy's contributors needs them.
CONTRIBUTORS = [
    "test_energy_degree",
    "test_energy_clustering",
    "test_energy_edge_length",
    "test_energy_betweenness",
]


def summary_path(measure: str) -> Path:
    return (ROOT / "output" / "gnm" / DATASET / EXP_NAME /
            f"summary_indiv_{measure}_for_exp_{EXP_NAME}.csv")


def already_done(measure: str) -> bool:
    """True if the summary file holds a score for every network."""
    p = summary_path(measure)
    if not p.exists():
        return False
    import pandas as pd
    df = pd.read_csv(p)
    score_cols = [c for c in df.columns
                  if c not in {"network_index", "filename", "eta", "gamma", "id"}]
    return len(df) == N_NETWORKS and bool(score_cols) and not df[score_cols].isna().any().any()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of measures")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--processes", type=int, default=12)
    ap.add_argument("--force", action="store_true", help="rescore even if complete")
    ap.add_argument("--with-contributors", action="store_true",
                    help="also score the four KS criteria behind the energy")
    args = ap.parse_args()

    measures = args.only if args.only else list(MEASURES)
    if args.with_contributors and not args.only:
        measures += CONTRIBUTORS
    unknown = set(measures) - set(MEASURES) - set(CONTRIBUTORS)
    if unknown:
        raise SystemExit(f"not a known measure: {sorted(unknown)}")

    if args.list:
        for m in measures:
            print(f"  {'done ' if already_done(m) else 'todo '} {m}")
        return

    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "pipeline_gnms_and_benchmarking")]
    from run_connectome_comparisons import main as compare

    for i, measure in enumerate(measures, 1):
        if already_done(measure) and not args.force:
            print(f"[{i}/{len(measures)}] {measure}: already complete, skipping")
            continue
        print(f"\n{'=' * 70}\n[{i}/{len(measures)}] {measure}\n{'=' * 70}", flush=True)
        t0 = time.time()
        compare(
            dataset_name=DATASET,
            experiment_name=EXP_NAME,
            evaluation_mode=measure,
            debug_subject_ids=None,
            number_multiprocessing_processes=args.processes,
        )
        print(f"[{i}/{len(measures)}] {measure}: {(time.time() - t0) / 60:.1f} min", flush=True)

    print("\nStatus:")
    for m in measures:
        print(f"  {'ok  ' if already_done(m) else 'FAIL'} {m}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
