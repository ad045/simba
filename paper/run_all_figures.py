"""
Rebuild every figure of the benchmarking paper, in dependency order
===================================================================

One pass over all the figure producers for the current MORPHO_EXP, ending with
`collect_figure_parts.py`. Running them in a defined order matters: several
scripts write the same filename into the same directory, so an ad-hoc order
leaves a set that is internally inconsistent (see SNAPSHOTS below).

    conda activate ma_thesis
    python run_all_figures.py --list        # the plan, nothing runs
    python run_all_figures.py               # everything except the slow recompute
    python run_all_figures.py --all         # including it
    python run_all_figures.py --only radar collect

The rewiring-robustness *compute* stage is skipped by default. It takes about
three hours and is seeded (BASE_SEED), so it reproduces its existing
rewiring_robustness_results.csv exactly; its plot stage is re-run either way.
Pass --all to recompute it from scratch.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments_config import MORPHO_DIR, MORPHO_EXP

PY = sys.executable

# (name, argv, slow?) - order is the dependency order, do not sort.
STEPS = [
    ("fig2_16_measures",      ["visualization/run_8_7_manuscript_16_measures.py"], False),
    ("fig2_appendix_variant", ["visualization/run_8_7_6_manuscript_16_measures_hcp_apdx.py"], False),
    ("fig1_connectomes",      ["visualization/run_21_2_fig_1_six_example_gen_connectomes_hcp_data.py"], False),
    ("eight_measures",        ["visualization/run_8_9_manuscript_selected_8_measures.py"], False),
    ("minima_landscape",      ["visualization/run_8_9_2_minima_from_average_landscape.py"], False),
    ("ks_energy_contributors", ["visualization/run_8_7_2_ks_energy_contributors.py"], False),
    ("qc_and_variance",       ["visualization/run_9_between_and_withhin_variance.py"], False),
    ("pca_eight",             ["experiment_decision_figure/run_pca_eight_measures.py"], False),
    ("isnr_panel",            ["visualization/run_isnr_consistent_panel.py"], False),
    ("degeneration_panels",   ["visualization/run_degeneration_effective_panels.py"], False),
    ("cost_hemisphere",       ["visualization/run_cost_hemisphere_landscapes.py"], False),
    ("topographic",           ["experiment_topographic_plausibility/run_topographic_plausibility.py",
                               "--stage", "all"], False),
    ("structural_gradient",   ["experiment_structural_gradient/run_structural_gradient.py",
                               "--stage", "all"], False),
    ("hub_topography",        ["experiment_hub_topography/run_hub_topography.py", "--stage", "all"], False),
    ("real_vs_artificial",    ["experiment_real_vs_artificial/run_real_vs_artificial.py",
                               "--stage", "all"], False),
    ("effective_degeneration", ["experiment_rewiring_robustness/compute_effective_degeneration.py"], False),
    ("rewiring_compute",      ["experiment_rewiring_robustness/run_rewiring_robustness.py",
                               "--stage", "compute"], True),
    ("rewiring_figure",       ["experiment_rewiring_robustness/run_rewiring_robustness.py",
                               "--stage", "plot"], False),
    ("radar_five_axes",       ["experiment_decision_figure/build_radar_five_axes.py"], False),
    ("patch_figure3",         ["visualization/patch_figure4_panels.py"], False),
    ("legend_8_measures",     ["make_legend_8_measures.py"], False),
    ("grid_and_variance",     ["experiment_parameter_recovery/run_grid_and_variance_paper_plot.py"], False),
    ("fig5_recovery",         ["visualization/build_fig5_recovery.py"], False),
    ("composites",            ["visualization/build_composites.py"], False),
    ("collect",               ["collect_figure_parts.py", "--clean"], False),
]

# Files that two steps both write, where the earlier one is the keeper.
# ks_energy_contributors writes its own correlation_matrix.csv over the
# eight-measure one, and the radar reads the eight-measure version.
SNAPSHOTS = [
    ("eight_measures",
     MORPHO_DIR / "ground_truth_analysis" / "correlation_matrix.csv",
     MORPHO_DIR / "method_evaluation" / "correlation_matrix.csv"),
]


def run(step, argv, log_dir: Path) -> tuple[bool, float]:
    log = log_dir / f"{step}.log"
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.call([PY, *argv], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    return rc == 0, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include the slow recompute")
    ap.add_argument("--only", nargs="*", default=None, help="run only these steps")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    steps = [s for s in STEPS if not s[2] or args.all]
    if args.only:
        unknown = set(args.only) - {s[0] for s in STEPS}
        if unknown:
            raise SystemExit(f"unknown step(s): {sorted(unknown)}")
        steps = [s for s in STEPS if s[0] in args.only]

    print(f"run: {MORPHO_EXP}\n")
    if args.list:
        for name, argv, slow in STEPS:
            mark = "skip" if (slow and not args.all) else ("run " if (name, argv, slow) in steps else "  - ")
            print(f"  {mark}  {name:<24} {' '.join(argv)}")
        return

    log_dir = ROOT / "output" / "benchmarking_paper" / "_logs" / MORPHO_EXP
    log_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, (name, argv, _slow) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {name:<24} ", end="", flush=True)
        ok, secs = run(name, argv, log_dir)
        print(f"{'ok  ' if ok else 'FAIL'} {secs / 60:6.1f} min")
        if not ok:
            failed.append(name)
            print(f"         see {log_dir / (name + '.log')}")
        for owner, src, dst in SNAPSHOTS:
            if owner == name and ok and src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"         kept {src.name} -> {dst.parent.name}/")

    print()
    if failed:
        print(f"{len(failed)} step(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print(f"all {len(steps)} steps ok; logs in {log_dir}")


if __name__ == "__main__":
    main()
