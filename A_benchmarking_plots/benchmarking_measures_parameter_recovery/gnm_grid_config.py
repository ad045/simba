"""
Shared configuration for the continuous-landscape GNM parameter-recovery pipeline.

To add a true parameter combination:
  1. Append (eta, gamma) to TRUE_PARAM_COMBOS
  2. Append the matching LaTeX string to COMBO_LABELS
  — that's it. All three scripts pick it up automatically.
"""

from typing import List, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# True parameter combinations (test networks generated at these points)
# ---------------------------------------------------------------------------

TRUE_PARAM_COMBOS: List[Tuple[float, float]] = [
    (-6.9,                  0.89),
    ( 1.9,                  0.89),
    (-6.9,                  0.01),
    ( 1.9,                  0.01),
    (-3.734694004058838,    0.595918357372283),
    # (-2.836734771728516,    0.595918357372283)
]

COMBO_LABELS: List[str] = [
    r"$\eta{=}{-}6.9,\ \gamma{=}0.89$",
    r"$\eta{=}1.9,\ \gamma{=}0.89$",
    r"$\eta{=}{-}6.9,\ \gamma{=}0.01$",
    r"$\eta{=}1.9,\ \gamma{=}0.01$",
    r"$\eta{=}{-}3.73,\ \gamma{=}0.60$",
    # r"$\eta{=}{-}2.84,\ \gamma{=}0.60$",
]

# ---------------------------------------------------------------------------
# η-γ grid for consensus networks
# ---------------------------------------------------------------------------

GRID_N_ETA   = 10
GRID_N_GAMMA = 10
GRID_ETA     = np.linspace(-8.0, 3.0, GRID_N_ETA)
GRID_GAMMA   = np.linspace(-0.1, 1.0, GRID_N_GAMMA)

# gamma-outer / eta-inner so reshape(GRID_N_GAMMA, GRID_N_ETA) gives a proper matrix
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
N_TEST               = 10   # test networks per true parameter combination
N_CONSENSUS_GRID     = 30   # networks per grid point used to build consensus

# ---------------------------------------------------------------------------
# Helpers shared by generation and comparison scripts
# ---------------------------------------------------------------------------

def param_dir_name(eta: float, gamma: float) -> str:
    return (f"param_eta{eta}".replace("-", "m").replace(".", "p")
            + f"_gamma{gamma}".replace("-", "m").replace(".", "p"))


def net_filename(eta: float, gamma: float, net_id: int) -> str:
    return f"net_eta{eta}_gamma{gamma}_rule{GENERATIVE_RULE_NAME}_id{net_id:03d}.npy"
