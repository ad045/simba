from datetime import datetime
from collections.abc import Iterable
from pathlib import Path


# -> Used at least for main_pipeline_2.py (esn part)

def _summarize_hparam_space(hparam_grid: list[dict]) -> dict:
    """
    Build a summary of unique values for each hyperparameter across the grid.
    Returns a dict: {param_name: sorted_unique_values}
    """
    if not hparam_grid:
        return {}

    uniq = {}
    for hp in hparam_grid:
        for k, v in hp.items():
            uniq.setdefault(k, set()).add(v)

    def _safe_sorted(vals: set):
        try:
            # Try numeric sort first, then fallback to string
            return sorted(vals)  # ok for homogenous numeric/str
        except Exception:
            return sorted(vals, key=lambda x: str(x))

    return {k: _safe_sorted(vs) for k, vs in uniq.items()}

def _write_run_info_txt(
    path,
    *,
    started_at: str,
    updated_at: str,
    save_dir: str,
    timestamp: str,
    n_subjects: int,
    total_tasks: int,
    completed_tasks: int,
    search_mode: str,                # "grid" or "random_sample"
    random_sample_size: int | None,  # if used, else None
    densities_available: list[int] | None,
    hparam_space_summary: dict,
):
    """
    Overwrites a human-readable run_info.txt with the latest status.
    Call this repeatedly (e.g., every time a task completes).
    """
    lines = []
    lines.append("=== ESN Grid/Random Search Run Info ===")
    lines.append(f"Started at:      {started_at}")
    lines.append(f"Last updated:    {updated_at}")
    lines.append(f"Save dir:        {save_dir}")
    lines.append(f"Timestamp:       {timestamp}")
    lines.append("")
    lines.append(f"Subjects:        {n_subjects}")
    lines.append(f"Total tasks:     {total_tasks}")
    lines.append(f"Completed:       {completed_tasks}  ({(completed_tasks/total_tasks*100):.2f}%)")
    lines.append("")
    lines.append(f"Search mode:     {search_mode}")
    if search_mode == "random_sample":
        lines.append(f"Sample size:     {random_sample_size}")
    if densities_available:
        lines.append(f"Densities:       {', '.join(map(str, densities_available))}")
    lines.append("")
    lines.append("Hyperparameter space (unique values):")
    for k, vals in sorted(hparam_space_summary.items()):
        # Keep lines compact but readable; cap very long lists
        shown = vals if len(vals) <= 20 else (vals[:20] + ["..."])
        lines.append(f"  - {k}: {shown}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - This file is overwritten as the run progresses.")
    lines.append("  - The CSVs contain per-task results and timings.")
    path = Path(path)
    path.write_text("\n".join(lines))

