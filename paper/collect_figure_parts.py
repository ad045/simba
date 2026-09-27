"""
Gather every figure part of the benchmarking paper into one folder
==================================================================

The figure-producing scripts each write into their own output directory, and the
manuscript's figures are then assembled from those parts by hand in a vector
editor. This copies all the parts for one morphospace run into

    output/benchmarking_paper/<MORPHO_EXP>/

grouped by the manuscript figure they feed, with a MANIFEST.md recording where
each part came from and whether the 10%-density regeneration changed it.

Nothing is moved and no source is modified - this only copies, so it is safe to
re-run after regenerating any single figure.

    conda activate ma_thesis
    python collect_figure_parts.py            # collect for the current MORPHO_EXP
    python collect_figure_parts.py --clean    # wipe the target folder first
    python collect_figure_parts.py --list     # show what would be collected
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments_config import MANUSCRIPT_FIGURES  # noqa: E402
from experiments_config import MORPHO_DATASET, MORPHO_EXP, ROOT_DIR

OUT_ROOT = ROOT_DIR / "output" / "benchmarking_paper"
MANUSCRIPT = MANUSCRIPT_FIGURES

_GNM = ROOT_DIR / "output" / "gnm" / MORPHO_DATASET
_MORPHO = _GNM / MORPHO_EXP

# group -> (source dir, glob, manuscript figure it feeds, changed by the fix?)
GROUPS = [
    ("fig_1_landscape_and_connectomes",
     _MORPHO / "landscape_and_six_example_gen_connectomes", "*.pdf",
     "Figure 1 (overview composite)", "yes"),
    ("fig_2_sixteen_measures",
     _MORPHO / "figures_manuscript", "*.pdf",
     "Figure 2 - fig_huge_02_landscapes_corr_network_timing_plausible.pdf", "yes"),
    ("fig_2_appendix_variant",
     _MORPHO / "figures_manuscript_apdx", "*.pdf",
     "Appendix variant of Figure 2", "yes"),
    ("fig_3_qc_and_panels",
     _MORPHO, "*.pdf",
     "Figure 3 panels + QC appendix (qc*, cv_vs_time*, iSNR*, mae*, fig_panel_H)", "mixed"),
    ("fig_3_degeneration_panels",
     _GNM / "chaos_analysis", "fig_panel_*.pdf",
     "Figure 3 panels D and F", "no - consensus-only, reproduces published values"),
    ("eight_measures_ground_truth",
     _MORPHO / "ground_truth_analysis", "*.pdf",
     "Eight-measure figures and KS-energy contributors", "yes"),
    ("eight_measures_method_evaluation",
     _MORPHO / "method_evaluation", "*.pdf",
     "Method-evaluation panels", "yes"),
    ("appendix_pca_eight",
     MANUSCRIPT / "appendix" / "pca_eight", "*.*",
     "fig_pca_landscapes_loadings_2.pdf (composite)", "yes - PC1 55.1 / PC2 16.1 / PC3 15.0"),
    ("appendix_cost_hemisphere",
     ROOT_DIR / "output" / "cost_hemisphere_landscapes", "*.pdf",
     "fig_cost_hemisphere_landscapes.pdf (direct copy)", "yes"),
    ("appendix_topographic_plausibility",
     ROOT_DIR / "output" / "topographic_plausibility", "*.pdf",
     "Topographic plausibility appendix", "yes"),
    ("appendix_structural_gradient",
     ROOT_DIR / "output" / "structural_gradient", "*.pdf",
     "Structural gradient appendix", "yes"),
    ("appendix_hub_topography",
     ROOT_DIR / "output" / "hub_topography", "*.pdf",
     "hub_topography_sa_axis.pdf, hubs_landscapes_violins_2.pdf", "yes"),
    ("appendix_real_vs_artificial",
     ROOT_DIR / "output" / "real_vs_artificial", "*.pdf",
     "Real-vs-artificial appendix", "yes - hard-case AUC drops sharply"),
    ("appendix_rewiring_robustness",
     ROOT / "figures" / "appendix", "fig_rewiring_robustness*.pdf",
     "drift_of_recovered_parameters.pdf", "yes - noise tolerance N* reorders"),
    ("appendix_radar_five_axes",
     MANUSCRIPT / "radar", "*.pdf",
     "method_legend.pdf and the radar figures", "yes - aggregates all five axes"),
    ("unchanged_parameter_recovery",
     ROOT / "figures" / "appendix", "grid_and_variance*.pdf",
     "grid_and_variance.pdf, fig_5_recovery_error_2.pdf", "no - generated-vs-generated at 495 edges"),
]


def rescore_time() -> float:
    """When this run came into existence - the cutoff for "belongs to this run".

    The reference CSV is written once the 25,000 networks are on disk, so anything
    older than it cannot have been produced from them. Deliberately not the newest
    summary_indiv_*.csv: measures get added later (the KS contributors were), which
    would wrongly flag figures made in between.
    """
    ref = _MORPHO / f"all_metrics_for_{MORPHO_EXP}.csv"
    return ref.stat().st_mtime if ref.exists() else 0.0


def collect(dry_run: bool) -> None:
    target = OUT_ROOT / MORPHO_EXP
    cutoff = rescore_time()
    print(f"run    : {MORPHO_EXP}")
    print(f"target : {target}\n")

    rows, n_files, missing, stale = [], 0, [], []
    for name, src, pattern, feeds, changed in GROUPS:
        files = sorted(f for f in src.glob(pattern) if " copy" not in f.name) \
            if src.is_dir() else []
        if not files:
            missing.append((name, src))
            print(f"  --   {name:<36} (nothing at {src})")
            continue
        if not dry_run:
            dest = target / name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
        n_files += len(files)
        old = [f.name for f in files if f.stat().st_mtime < cutoff]
        stale.extend(f"{name}/{n}" for n in old)
        rows.append((name, src, files, feeds, changed))
        print(f"  {len(files):>3}  {name}" + (f"   ({len(old)} pre-date the rescore)" if old else ""))

    print(f"\n{n_files} files in {len(rows)} groups"
          + (f", {len(missing)} group(s) with no source yet" if missing else ""))

    if dry_run:
        return

    lines = [
        f"# Figure parts - {MORPHO_EXP}",
        "",
        "Every part the manuscript's figures are assembled from, for this run. Collected by",
        "`collect_figure_parts.py`; copies only, the sources are untouched.",
        "",
        "The manuscript's `figures/` holds vector-editor composites built from these parts,",
        "under different names. Only `fig_3_total_variation_degree_distance_3.pdf` (via",
        "`patch_figure4_panels.py`) and `fig_cost_hemisphere_landscapes.pdf` are written directly.",
        "",
        "| Group | Files | Feeds | Changed by the 10% regeneration? |",
        "|---|---|---|---|",
    ]
    for name, _src, files, feeds, changed in rows:
        lines.append(f"| `{name}/` | {len(files)} | {feeds} | {changed} |")
    lines += ["", "## Sources", ""]
    for name, src, files, _f, _c in rows:
        try:
            shown = src.relative_to(ROOT_DIR)
        except ValueError:
            shown = src
        lines.append(f"- `{name}/` <- `{shown}`")
    if stale:
        lines += ["", "## Carried along, NOT regenerated", "",
                  "These pre-date this run's rescore. They are hand-assembled composites (or their",
                  "leftovers) and still show the 594-edge content until the vector-editor pass is done:",
                  ""]
        for f in stale:
            lines.append(f"- `{f}`")
    if missing:
        lines += ["", "## Not collected (no source present)", ""]
        for name, src in missing:
            lines.append(f"- `{name}` - nothing at `{src}`")
    (target / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"manifest -> {target / 'MANIFEST.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="wipe the target folder first")
    ap.add_argument("--list", action="store_true", help="show what would be collected")
    args = ap.parse_args()

    if args.clean and not args.list:
        target = OUT_ROOT / MORPHO_EXP
        if target.exists():
            shutil.rmtree(target)
            print(f"removed {target}\n")
    collect(dry_run=args.list)


if __name__ == "__main__":
    main()
