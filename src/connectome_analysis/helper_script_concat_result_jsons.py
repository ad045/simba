# #!/usr/bin/env python3
# """
# Batch runner: iterate over folders that contain W&B runs and generate visualizations
# using the visualize_pipeline_results(...) function from your visualization module.

# Usage:
#   python batch_visualize.py \
#     --root /path/to/experiments \
#     --module pipeline_visualizer \
#     --viz voronoi interpolated \
#     --no-save   # (optional) display plots instead of saving
    
#     python batch_visualize.py \
#     --root /Users/adrian/Documents/01_projects/14_4D_lab/src/connectome_analysis/wandb
#     --module pipeline_visualizer \
#     --viz voronoi interpolated \
#     --no-save   # (optional) display plots instead of saving
    
#     /Users/adrian/Documents/01_projects/14_4D_lab/src/connectome_analysis/wandb
    
    
# """

# import visualization_2
# from visualization_2 import PipelineVisualizer

# from __future__ import annotations
# import argparse
# import logging
# from pathlib import Path
# from typing import Iterable, List, Set, Dict, Any
# import importlib
# import fnmatch
# import sys

# WandbSummaryName = "wandb-summary.json"
# ESN_PATTERN = "esn_mc_results_*.csv"
# GNM_FILENAME = "gnm_comprehensive_results.json"

# def find_files(root: Path, name: str) -> Iterable[Path]:
#     """Recursively yield files named exactly `name`."""
#     for p in root.rglob(name):
#         if p.is_file():
#             yield p

# def contains_results(dirpath: Path) -> bool:
#     """Return True if directory looks like an experiment dir."""
#     esn_any = any(dirpath.glob(ESN_PATTERN))
#     gnm_any = (dirpath / GNM_FILENAME).exists()
#     return esn_any or gnm_any

# def nearest_experiment_dir(start: Path, max_hops: int = 6) -> Path | None:
#     """
#     Walk up from `start` at most `max_hops` levels until a directory that contains
#     known result files is found.
#     """
#     cur = start
#     for _ in range(max_hops):
#         if contains_results(cur):
#             return cur
#         if cur.parent == cur:
#             break
#         cur = cur.parent
#     return None

# def collect_experiment_dirs(root: Path) -> List[Path]:
#     """Find all wandb summaries, map them to nearest experiment directories, de-duplicate."""
#     candidates: Set[Path] = set()
#     for summary_file in find_files(root, WandbSummaryName):
#         exp_dir = nearest_experiment_dir(summary_file.parent)
#         if exp_dir:
#             candidates.add(exp_dir)
#         else:
#             logging.debug(f"No results found upward from: {summary_file}")
#     return sorted(candidates)

# def main():
#     ap = argparse.ArgumentParser(description="Batch visualize pipeline results near wandb runs.")
#     ap.add_argument("--root", type=Path, # required=True,
#                     default=Path("/Users/adrian/Documents/01_projects/14_4D_lab/src/connectome_analysis/wandb"),
#                     help="Root directory to search (will be scanned recursively).")
#     # ap.add_argument("--module", type=str, default="pipeline_visualizer",
#     #                 help="Python module that exports visualize_pipeline_results.")
#     # ap.add_argument("--viz", nargs="+", default=["voronoi"],
#     #                 help='Visualization types to generate (e.g. "voronoi", "interpolated").')
#     # ap.add_argument("--save", dest="save", action="store_true", default=True,
#     #                 help="Save plots to disk (default).")
#     # ap.add_argument("--no-save", dest="save", action="store_false",
#     #                 help="Show plots interactively instead of saving.")
#     ap.add_argument("-v", "--verbose", action="count", default=0,
#                     help="Increase logging verbosity (-v, -vv).")
#     args = ap.parse_args()

#     # Logging
#     level = logging.WARNING
#     if args.verbose == 1:
#         level = logging.INFO
#     elif args.verbose >= 2:
#         level = logging.DEBUG
#     logging.basicConfig(format="%(levelname)s: %(message)s", level=level)

#     root: Path = args.root.resolve()
#     if not root.exists():
#         logging.error(f"Root path does not exist: {root}")
#         # root = Path("/Users/adrian/Documents/01_projects/14_4D_lab/src/connectome_analysis/wandb").resolve()
#         # logging.warning("Replaced with DEFAULT ABOLUTE path: /Users/adrian/Documents/01_projects/14_4D_lab/src/connectome_analysis/wandb")
#         sys.exit(1)

#     # Import the user’s visualization module
#     # try:
#     #     viz_mod = importlib.import_module(args.module)
#     # except Exception as e:
#     #     logging.error(f"Failed to import module '{args.module}': {e}")
#     #     sys.exit(1)

#     # if not hasattr(viz_mod, "visualize_pipeline_results"):
#     #     logging.error(
#     #         f"Module '{args.module}' does not export visualize_pipeline_results(experiment_dir, ...)."
#     #     )
#     #     sys.exit(1)

#     # visualize_pipeline_results = getattr(viz_mod, "visualize_pipeline_results")

#     # Collect experiment directories
#     logging.info(f"Scanning for '{WandbSummaryName}' under: {root}")
#     exp_dirs = collect_experiment_dirs(root)
#     if not exp_dirs:
#         logging.warning("No experiment directories with results were found.")
#         sys.exit(0)

#     logging.info(f"Found {len(exp_dirs)} experiment directories with results.")
#     summary: Dict[str, Any] = {}

#     # Process each experiment directory once
#     for i, exp_dir in enumerate(exp_dirs, 1):
#         logging.info(f"[{i}/{len(exp_dirs)}] Processing: {exp_dir}")
#         try:
#             created = visualize_pipeline_results(
#                 experiment_dir=exp_dir,
#                 visualization_types=args.viz,
#                 save_plots=args.save,
#             )
#             # Coerce Paths to strings for nice printing
#             created_str = {k: str(v) if v is not None else None for k, v in (created or {}).items()}
#             logging.debug(f"Created: {created_str}")
#             summary[str(exp_dir)] = created_str
#         except Exception as e:
#             logging.exception(f"Failed on {exp_dir}: {e}")

#     # Pretty print a compact summary
#     print("\n=== Batch summary ===")
#     for k, v in summary.items():
#         print(f"- {k}")
#         if not v:
#             print("  (no plots created)")
#         else:
#             for name, path in v.items():
#                 print(f"  {name}: {path}")

# if __name__ == "__main__":
#     main()

# # python helper_script_concat_result_jsons.py --root /Users/adrian/Documents/01_projects/wandb --module connecotome.visualizer --viz voronoi interpolated --no-save

# # class PipelineVisualizer:
# #     """Visualization tools for connectome analysis pipeline results."""
    
# #     def __init__(self, output_dir: Optional[Path] = None):
# #         """
# #         Initialize visualizer.
        
# #         Args:
# #             output_dir: Directory to save visualizations
# #         """
# #         self.output_dir = Path(output_dir) if output_dir else Path("./visualizations")
# #         self.output_dir.mkdir(parents=True, exist_ok=True)
    
# #     def plot_energy_landscape_voronoi(self, df, dot_color="white", title="", 
# #                                       cmap="hot", savepath=None, show=True,
# #                                       show_points=True, point_size=8,
# #                                       vmin=None, vmax=None):
# #         """
