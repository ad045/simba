"""
Wide-target recovery with UNIFORM ground truth, and the figure that pairs it
===========================================================================

The published wide-target experiment places its ground truth at six hand-picked
combinations (10 networks each). The plausible-window experiment instead draws
its ground truth uniformly from the window. That difference is a confound when
the two are read side by side: part of the gap between them is the sampling
scheme, not the size of the region.

This script removes it. It repeats the wide experiment with the SAME sampling
the window experiment uses - 100 combinations drawn uniformly, one network each,
same seed - over the full morphospace, and then rebuilds the recovery figure
from the two uniformly-sampled regimes:

    fig_5_recovery_error_uniform_sampling.pdf

Nothing about the window experiment changes; its CSVs are read as they are.
The wide recovery grid is reused as well: the 10x10 grid consensuses of
`synthetic_parameter_recovery_grid` (30 networks per point) are read in place,
so only the 100 ground-truth networks are generated here.

Stages (idempotent - existing files are skipped):

    generate : 100 uniform (eta, gamma) over eta in [-8, 3], gamma in [-0.1, 1];
               one network each. Writes ground_truth_params.csv.
    compare  : distance of every ground-truth network to all 100 grid
               consensuses, per measure; recovered cell = argmin. One CSV per
               measure, with `predicted_*` aliases so the wide-target readers
               work unchanged.
    figure   : the three panels (wide-uniform bars, window bars, slopegraph) and
               the assembled composite.

Usage
-----
    conda activate ma_thesis
    python experiment_parameter_recovery_fine/run_wide_uniform_recovery.py --stage all
    python experiment_parameter_recovery_fine/run_wide_uniform_recovery.py --stage figure
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = _THIS_DIR.parent
for _p in (str(ROOT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_config import (  # noqa: E402
    coarse as wide_cfg, fine as fine_cfg,
    gt_dir_name, param_dir_name, net_filename,
)
import run_synthetic_gnm_fine as rsgf  # noqa: E402

OUT_DIR = ROOT_DIR / "output" / "gnm" / "synthetic_parameter_recovery_wide_uniform"
# the wide recovery grid is reused in place; its consensuses were built from 30
# networks per point, which is why n_consensus below is 30 and not the window's 20
WIDE_GRID_DIR = (ROOT_DIR / "output" / "gnm" /
                 "synthetic_parameter_recovery_grid" / "grid_consensus")
N_CONSENSUS_WIDE = 30

# Same count, one network per point, and the same seed as the window experiment,
# so the two regimes differ only in the region the ground truth is drawn from.
N_GROUND_TRUTH = fine_cfg.N_GROUND_TRUTH        # 100
N_NETS_PER_TRUTH = fine_cfg.N_NETS_PER_TRUTH    # 1
SAMPLE_SEED = fine_cfg.SAMPLE_SEED              # 12345
SAMPLE_ETA_RANGE: Tuple[float, float] = wide_cfg.ETA_RANGE      # (-8, 3)
SAMPLE_GAMMA_RANGE: Tuple[float, float] = wide_cfg.GAMMA_RANGE  # (-0.1, 1)


def sample_ground_truth(n: int = N_GROUND_TRUTH,
                        seed: int = SAMPLE_SEED) -> List[Tuple[float, float]]:
    """Uniform draw over the full morphospace - the window experiment's
    `_fine_sample_ground_truth` with the wide ranges."""
    rng = np.random.default_rng(seed)
    etas = rng.uniform(SAMPLE_ETA_RANGE[0], SAMPLE_ETA_RANGE[1], n)
    gammas = rng.uniform(SAMPLE_GAMMA_RANGE[0], SAMPLE_GAMMA_RANGE[1], n)
    return [(float(e), float(g)) for e, g in zip(etas, gammas)]


# The runner reads its sampling settings off a module-level `cfg`; this is that
# object for the uniform wide run (same attribute names as `experiments_config.fine`).
CFG = SimpleNamespace(
    SAMPLE_ETA_RANGE=SAMPLE_ETA_RANGE,
    SAMPLE_GAMMA_RANGE=SAMPLE_GAMMA_RANGE,
    SAMPLE_SEED=SAMPLE_SEED,
    N_GROUND_TRUTH=N_GROUND_TRUTH,
    N_NETS_PER_TRUTH=N_NETS_PER_TRUTH,
    GRID_N_ETA=wide_cfg.GRID_N_ETA,
    GRID_N_GAMMA=wide_cfg.GRID_N_GAMMA,
    GRID_ETA=wide_cfg.GRID_ETA,
    GRID_GAMMA=wide_cfg.GRID_GAMMA,
    sample_ground_truth=sample_ground_truth,
    gt_dir_name=gt_dir_name,
    param_dir_name=param_dir_name,
    net_filename=net_filename,
)


class WideUniformCfg(rsgf.RunCfg):
    """The window runner's config, pointed at the wide grid and ranges."""

    def __init__(self, measures=None):
        super().__init__(smoke=False, measures=measures)
        self.grid_eta = wide_cfg.GRID_ETA
        self.grid_gamma = wide_cfg.GRID_GAMMA
        self.grid_combos = [(float(e), float(g))
                            for g in self.grid_gamma for e in self.grid_eta]
        self.eta_range = (float(self.grid_eta[0]), float(self.grid_eta[-1]))
        self.gamma_range = (float(self.grid_gamma[0]), float(self.grid_gamma[-1]))
        self.n_consensus = N_CONSENSUS_WIDE
        self.n_truth = N_GROUND_TRUTH
        self.out_dir = OUT_DIR

    @property
    def grid_dir(self):
        return WIDE_GRID_DIR      # read in place; nothing is written there


def _add_predicted_aliases(comparison_dir: Path) -> None:
    """The wide-target readers expect `predicted_*`; the window runner writes
    `recovered_*`. Same numbers, two names."""
    for path in sorted(comparison_dir.glob("distances_*.csv")):
        df = pd.read_csv(path)
        if "predicted_grid_idx" in df.columns or "recovered_grid_idx" not in df:
            continue
        for src, dst in (("recovered_grid_idx", "predicted_grid_idx"),
                         ("recovered_eta", "predicted_eta"),
                         ("recovered_gamma", "predicted_gamma")):
            df[dst] = df[src]
        df.to_csv(path, index=False)


def figure(outdir: Path | None = None) -> None:
    """The single-figure recovery panel set, wide half read from the uniform run."""
    import importlib.util
    import run_recovery_main_figures as rmf

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)      # visualization/ is not a package
        return mod

    comparison_dir = OUT_DIR / "comparison_results"
    if not any(comparison_dir.glob("distances_*.csv")):
        raise FileNotFoundError(f"No comparisons in {comparison_dir} - run --stage compare.")
    _add_predicted_aliases(comparison_dir)

    outdir = outdir or (OUT_DIR / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    rmf.WIDE_COMPARISON_DIR = comparison_dir       # wide half = the uniform run
    rsgf.WIDE_COMPARISON_DIR = comparison_dir
    sys.argv = ["run_recovery_main_figures.py", "--outdir", str(outdir)]
    rmf.main()

    # figure 5 is one figure now, not a hand-measured composite; it reads the
    # same loaders, so patching rmf's wide dir above is what switches its A panel
    f5 = _load("build_fig5_recovery", ROOT_DIR / "visualization" / "build_fig5_recovery.py")
    f5.WIDE_COMPARISON_DIR = comparison_dir
    out_pdf = f5.PAPER / "fig_5_recovery_error_uniform_sampling.pdf"
    f5.build(out_pdf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "compare", "figure", "all"],
                    default="all")
    ap.add_argument("--measures", nargs="*", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    rc = WideUniformCfg(measures=args.measures)
    rsgf.cfg = CFG          # the runner's sampling/naming config for this run

    if args.stage in ("generate", "all"):
        rsgf.generate(rc)
    if args.stage in ("compare", "all"):
        rsgf.compare(rc)
        _add_predicted_aliases(rc.comparison_dir)
    if args.stage in ("figure", "all"):
        figure(Path(args.outdir) if args.outdir else None)


if __name__ == "__main__":
    main()
