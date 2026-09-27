"""
Derive the human-plausible (eta, gamma) window from the morphospace itself.

Rule
----
Pool the 100 best-fitting *networks* of each of the 8 selected measures (800
points) and take the central 50% (IQR) along each axis. This is the region the
candidate measures agree the empirical consensus lives in; its width along each
axis is their disagreement, not a hand-set half-width.

Note on the unit: the selection below ranks the 25,000 individual generated
networks, not the 2,500 replicate-averaged (eta, gamma) cells. The two give
different windows - networks yield gamma in [0.08, 0.44], cells would yield
[0.10, 0.55] - and it is the networks variant that produced the published
window. Do not "fix" this to cells without also regenerating the recovery grid,
the ground truth, and every number in the fine-recovery section of the paper.

The recovery grid is the window padded by one grid step on each side, so that a
ground-truth point near the window edge can still be recovered without the
argmin being truncated:  step = span / 7,  grid = [lo - step, hi + step] with
10 points.

Run this to reproduce the numbers hard-coded in experiments_config.py (`fine`):

    python derive_window.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments_config import (  # noqa: E402
    MORPHO_DIR, MORPHO_EXP, SELECTED_MEASURES, IS_SIMILARITY, META_COLS,
)

N_BEST = 100
GRID_N = 10


def best_fit_points(measure: str, n_best: int = N_BEST) -> np.ndarray:
    """The (eta, gamma) locations of the n_best individual networks with the
    lowest distance. Ranks networks, not replicate-averaged cells - see the
    module docstring. Several of the returned points may share a cell."""
    df  = pd.read_csv(MORPHO_DIR / f"summary_indiv_{measure}_for_exp_{MORPHO_EXP}.csv")
    col = [c for c in df.columns if c not in META_COLS][0]
    val = df[col].to_numpy(dtype=float)
    if measure in IS_SIMILARITY:      # higher = more similar -> flip to a distance
        val = -val
    df = df.assign(_d=val).dropna(subset=["_d"])
    return df.nsmallest(n_best, "_d")[["eta", "gamma"]].to_numpy()


def derive_window(n_best: int = N_BEST):
    pts = np.vstack([best_fit_points(m, n_best) for m in SELECTED_MEASURES])
    eta_win   = tuple(np.percentile(pts[:, 0], [25, 75]))
    gamma_win = tuple(np.percentile(pts[:, 1], [25, 75]))
    return eta_win, gamma_win, pts


def padded_grid(lo: float, hi: float, n: int = GRID_N) -> np.ndarray:
    """n points over [lo, hi] padded by one step on each side."""
    step = (hi - lo) / (n - 3)
    return np.linspace(lo - step, hi + step, n)


def main() -> None:
    eta_win, gamma_win, pts = derive_window()
    print(f"Pooled best-fit points: {len(pts)} "
          f"({len(SELECTED_MEASURES)} measures x {N_BEST} best-fitting networks)\n")
    for name, (lo, hi) in (("eta", eta_win), ("gamma", gamma_win)):
        grid = padded_grid(lo, hi)
        print(f"{name:6s} window (IQR) [{lo:.4f}, {hi:.4f}]  span {hi-lo:.4f}")
        print(f"{name:6s} grid  [{grid[0]:.4f}, {grid[-1]:.4f}]  step {grid[1]-grid[0]:.4f}\n")

    # sanity: the window must sit inside the grid, with room on both sides
    for (lo, hi) in (eta_win, gamma_win):
        grid = padded_grid(lo, hi)
        assert grid[0] < lo and hi < grid[-1], "grid does not enclose the window"
    print("OK: both grids strictly enclose their window.")


if __name__ == "__main__":
    main()
