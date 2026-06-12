"""
Configuration for the FINE-GRAINED, human-plausible-range parameter-recovery
experiment (analysis S1).

Motivation
----------
The original recovery experiment (gnm_grid_config.py) places 5 widely-spread
true parameter combinations across the whole (eta, gamma) plane. A measure can
look good there simply because the targets are so different that almost any
measure tells them apart. Francesco's hypothesis: the *accuracy ranking* of
measures may shift in the small, realistic regime that researchers actually
operate in (differences between individuals / species), which sits in the
well-recovered region around eta ~= -3.7, gamma ~= 0.4-0.6.

Design
------
  * Ground truth : N_GROUND_TRUTH networks with (eta, gamma) sampled *uniformly
                   at random* from a tight plausible window (SAMPLE_*_RANGE).
                   These are individual networks (no consensus) -> they stand in
                   for individual brains.
  * Recovery     : each ground-truth network is matched (argmin distance) to a
                   FINE grid of consensus networks. The grid spans slightly WIDER
                   than the sampling window so that argmin recovery near the
                   window edges is not truncated (truncation would artificially
                   attenuate the true-vs-recovered correlation).
  * Score        : per measure, Pearson r between true and recovered eta, and
                   between true and recovered gamma ("fine-grained accuracy").

This file is the single source of truth; the runner (run_synthetic_gnm_fine.py)
imports everything from here.
"""

from typing import List, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Human-plausible region
# ---------------------------------------------------------------------------

# Centre of the well-recovered region (for reference / plotting only).
CENTER_ETA   = -3.7
CENTER_GAMMA = 0.5

# Ground-truth sampling window: true (eta, gamma) are drawn uniformly from here.
SAMPLE_ETA_RANGE:   Tuple[float, float] = (-4.5, -2.9)
SAMPLE_GAMMA_RANGE: Tuple[float, float] = (0.40, 0.60)

# ---------------------------------------------------------------------------
# Fine recovery grid (consensus networks live at these points)
# ---------------------------------------------------------------------------
# Deliberately a touch WIDER than the sampling window on every side so the
# argmin can land outside the sampling range -> no edge truncation bias.

GRID_N_ETA   = 10
GRID_N_GAMMA = 10
GRID_ETA     = np.linspace(-4.7, -2.7, GRID_N_ETA)   # step ~0.222
GRID_GAMMA   = np.linspace(0.37,  0.63, GRID_N_GAMMA) # step ~0.029

# gamma-outer / eta-inner so reshape(GRID_N_GAMMA, GRID_N_ETA) gives a matrix
GRID_COMBOS: List[Tuple[float, float]] = [
    (float(eta), float(gamma))
    for gamma in GRID_GAMMA
    for eta   in GRID_ETA
]

ETA_RANGE   = (float(GRID_ETA[0]),   float(GRID_ETA[-1]))
GAMMA_RANGE = (float(GRID_GAMMA[0]), float(GRID_GAMMA[-1]))

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------

GENERATIVE_RULE_NAME = "MatchingIndex"

N_GROUND_TRUTH   = 100   # number of true (eta, gamma) samples (individual nets)
N_NETS_PER_TRUTH = 1     # individual networks per ground-truth point
N_CONSENSUS_FINE = 20    # networks per grid point used to build the consensus

# Reproducible sampling of the ground-truth (eta, gamma) points.
SAMPLE_SEED = 12345

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_ground_truth(n: int = N_GROUND_TRUTH,
                        seed: int = SAMPLE_SEED) -> List[Tuple[float, float]]:
    """Uniformly sample n (eta, gamma) points from the plausible window."""
    rng    = np.random.default_rng(seed)
    etas   = rng.uniform(SAMPLE_ETA_RANGE[0],   SAMPLE_ETA_RANGE[1],   n)
    gammas = rng.uniform(SAMPLE_GAMMA_RANGE[0], SAMPLE_GAMMA_RANGE[1], n)
    return [(float(e), float(g)) for e, g in zip(etas, gammas)]


def param_dir_name(eta: float, gamma: float) -> str:
    """Directory name for a grid point (matches the existing convention)."""
    return (f"param_eta{eta}".replace("-", "m").replace(".", "p")
            + f"_gamma{gamma}".replace("-", "m").replace(".", "p"))


def gt_dir_name(idx: int) -> str:
    """Directory name for a ground-truth point (indexed; params live in manifest)."""
    return f"gt_{idx:03d}"


def net_filename(eta: float, gamma: float, net_id: int) -> str:
    return f"net_eta{eta}_gamma{gamma}_rule{GENERATIVE_RULE_NAME}_id{net_id:03d}.npy"


def normalise(eta: float, gamma: float) -> Tuple[float, float]:
    """Map (eta, gamma) to [0,1]^2 using the grid extent."""
    eta_n   = (eta   - ETA_RANGE[0])   / (ETA_RANGE[1]   - ETA_RANGE[0])
    gamma_n = (gamma - GAMMA_RANGE[0]) / (GAMMA_RANGE[1] - GAMMA_RANGE[0])
    return eta_n, gamma_n
