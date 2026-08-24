"""
Single source of truth for the benchmarking experiments
========================================================

This file consolidates the configuration that the four top-level experiment
folders used to keep scattered (two stand-alone ``gnm_*_config.py`` modules plus
constants embedded directly in the runner scripts):

    experiment_parameter_recovery/        -> `coarse`   (wide 5-combo recovery)
    experiment_parameter_recovery_fine/   -> `fine`     (human-plausible window)
    experiment_real_vs_artificial/        -> shared paths + measures/styling
    experiment_rewiring_robustness/       -> `rewiring`

Design
------
* Shared things (paths, the 8 selected measures, their orientation, names and
  colours, and the GNM filename helpers) live at module top level.
* The two parameter-recovery experiments collide on grid-geometry names
  (``GRID_ETA``, ``ETA_RANGE``, ...), so each is exposed as its own namespace
  (`coarse`, `fine`); the rewiring experiment likewise gets `rewiring`.

Paths are anchored at the **repo root** (this file's directory), where the
``data`` / ``output`` symlinks live - NOT at each experiment sub-folder, which
is what the old embedded copies got subtly wrong.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

# ===========================================================================
# Shared paths (anchored at the repo root, where data/ and output/ symlinks live)
# ===========================================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "hcp_schaefer_100_dataset"
RAW_DIR  = ROOT_DIR / "data" / "raw" / "hcp_schaefer_100"

# main morphospace run that produced the cached per-network distances (D_art)
MORPHO_DATASET = "hcp_schaefer_100_dataset"
MORPHO_EXP     = "105_distance_metrics_mst_animal_0"
MORPHO_DIR     = ROOT_DIR / "output" / "gnm" / MORPHO_DATASET / MORPHO_EXP

CONSENSUS_PATH    = DATA_DIR / "01_connectomes" / "01_consensus_bin_density_10_percent_100.npy"
INDIVIDUALS_PATH  = DATA_DIR / "01_connectomes" / "00_connectomes_density10.npy"
DIST_MATRIX_PATH  = DATA_DIR / "02_distance_matrices" / "distance_matrix_100.npy"
RAW_MAT_PATH      = RAW_DIR / "DTI_fibers_VolNorm_HCP.mat"
RAW_COORDS_PATH   = RAW_DIR / "Schaefer_100_MNI_coords.txt"

# default output dir for the real-vs-artificial experiment
OUT_DIR = ROOT_DIR / "output" / "real_vs_artificial"

HEMIID = np.array([0] * 50 + [1] * 50).reshape(-1, 1)
DENSITY_RETAIN = 10          # percent, as in the preprocessing pipeline
N_HARD_DEFAULT = 500         # hard-case: the N closest GNMs per measure

# metadata columns present in every summary_indiv_*.csv
META_COLS = {"network_index", "filename", "eta", "gamma", "id"}


# ===========================================================================
# The 8 selected measures (manuscript). key == summary-file token.
#   is_similarity : raw value higher => MORE similar (must be flipped to a dist)
#   colour/name   : taken from 8_9_manuscript_selected_8_measures.ipynb
# ===========================================================================

SELECTED_MEASURES: List[str] = [
    "frobenius",
    "delta_con",
    "netrd_non_backtracking_spectral",
    "spectral_distance_adjacency",
    "communicability_corr",
    "portrait",
    "net_simile",
    "energy",
]

# Raw comparer output higher => MORE similar, so it must be flipped before any
# argmin / "closest cell" logic. Covers all 16 benchmarked measures, not just the
# 8 selected ones (communicability_corr is the only similarity among those 8).
IS_SIMILARITY = {
    "communicability_corr",
    "jaccard",
    "f1",
    "network_mutual_information",
    "dc_network_mutual_information",
}


def to_distance(values, measure: str):
    """Orient raw comparer output so that LOWER always means more similar.

    Similarities are negated rather than mapped through 1-x: only the ordering
    matters for argmin / ranking, and negation is defined for unbounded scores
    (Pearson r) as well as for the bounded ones (Jaccard, F1, NMI).
    """
    v = np.asarray(values, dtype=float)
    return -v if measure in IS_SIMILARITY else v

METHOD_NAMES = {
    "portrait": "Portrait",
    "energy": "Energy",
    "spectral_distance_adjacency": "Spectral (Adjacency)",
    "delta_con": "DeltaCon",
    "frobenius": "Frobenius",
    "net_simile": "NetSimile",
    "netrd_non_backtracking_spectral": "Spectral Non-BT",
    "communicability_corr": "Communicability Corr.",
}

METRIC_COLORS = {
    "communicability_corr": (0.6, 0.6, 0.6),
    "netrd_non_backtracking_spectral": (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
    "delta_con": (0.4, 0.4, 0.4),
    "frobenius": (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
    "net_simile": (0.4, 0.6509803921568628, 0.11764705882352941),
    "spectral_distance_adjacency": (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
    "energy": (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
    "portrait": (0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
}


# ===========================================================================
# Shared GNM helpers (filename / directory conventions). The generative rule is
# the same ("MatchingIndex") across every experiment.
# ===========================================================================

GENERATIVE_RULE_NAME = "MatchingIndex"


def param_dir_name(eta: float, gamma: float) -> str:
    """Directory name for a grid point (matches the existing convention)."""
    return (f"param_eta{eta}".replace("-", "m").replace(".", "p")
            + f"_gamma{gamma}".replace("-", "m").replace(".", "p"))


def net_filename(eta: float, gamma: float, net_id: int) -> str:
    return f"net_eta{eta}_gamma{gamma}_rule{GENERATIVE_RULE_NAME}_id{net_id:03d}.npy"


def gt_dir_name(idx: int) -> str:
    """Directory name for a ground-truth point (indexed; params live in manifest)."""
    return f"gt_{idx:03d}"


# ===========================================================================
# Parameter recovery - COARSE wide grid  (experiment_parameter_recovery/)
# ===========================================================================
# True parameter combinations (test networks generated at these points).
# To add one: append to TRUE_PARAM_COMBOS *and* COMBO_LABELS - that's it.

_COARSE_TRUE_PARAM_COMBOS: List[Tuple[float, float]] = [
    (-6.9,                  0.89),
    ( 1.9,                  0.89),
    (-6.9,                  0.01),
    ( 1.9,                  0.01),
    (-3.734694004058838,    0.595918357372283),
    (-2.836734771728516,    0.595918357372283),
]

_COARSE_COMBO_LABELS: List[str] = [
    r"$\eta{=}{-}6.9,\ \gamma{=}0.89$",
    r"$\eta{=}1.9,\ \gamma{=}0.89$",
    r"$\eta{=}{-}6.9,\ \gamma{=}0.01$",
    r"$\eta{=}1.9,\ \gamma{=}0.01$",
    r"$\eta{=}{-}3.73,\ \gamma{=}0.60$",
    r"$\eta{=}{-}2.84,\ \gamma{=}0.60$",
]

_COARSE_N_ETA   = 10
_COARSE_N_GAMMA = 10
_COARSE_ETA     = np.linspace(-8.0, 3.0, _COARSE_N_ETA)
_COARSE_GAMMA   = np.linspace(-0.1, 1.0, _COARSE_N_GAMMA)

# gamma-outer / eta-inner so reshape(N_GAMMA, N_ETA) gives a proper matrix
_COARSE_COMBOS: List[Tuple[float, float]] = [
    (float(eta), float(gamma))
    for gamma in _COARSE_GAMMA
    for eta   in _COARSE_ETA
]

coarse = SimpleNamespace(
    TRUE_PARAM_COMBOS=_COARSE_TRUE_PARAM_COMBOS,
    COMBO_LABELS=_COARSE_COMBO_LABELS,
    GRID_N_ETA=_COARSE_N_ETA,
    GRID_N_GAMMA=_COARSE_N_GAMMA,
    GRID_ETA=_COARSE_ETA,
    GRID_GAMMA=_COARSE_GAMMA,
    GRID_COMBOS=_COARSE_COMBOS,
    ETA_RANGE=(float(_COARSE_ETA[0]),   float(_COARSE_ETA[-1])),
    GAMMA_RANGE=(float(_COARSE_GAMMA[0]), float(_COARSE_GAMMA[-1])),
    # generation settings
    GENERATIVE_RULE_NAME=GENERATIVE_RULE_NAME,
    N_TEST=10,            # test networks per true parameter combination
    N_CONSENSUS_GRID=30,  # networks per grid point used to build consensus
    # helpers
    param_dir_name=param_dir_name,
    net_filename=net_filename,
)


# ===========================================================================
# Parameter recovery - FINE human-plausible window  (experiment_parameter_recovery_fine/)
# ===========================================================================
# Ground-truth (eta, gamma) are drawn uniformly from a plausible window; the
# recovery grid is one step WIDER than the window on each side so argmin
# recovery near the edges is not truncated.
#
# The window is DERIVED FROM THE MORPHOSPACE, not hand-set: pool the 100
# best-fitting combinations of each of the 8 selected measures (800 points) and
# take the central 50% (IQR) per axis. Its width is therefore the measures'
# disagreement about where the consensus lives. Reproduce with
# experiment_parameter_recovery_fine/derive_window.py - the numbers below are
# that script's output, rounded to 4 decimals.
#
# (Superseded: a hand-set box of eta = -3.7 +- 0.8, gamma = 0.5 +- 0.10. Its
#  gamma range sat almost entirely ABOVE where the 8 measures actually place
#  their best fits - pooled gamma IQR is 0.08-0.44 - which is the main reason
#  gamma looked unrecoverable there.)

_FINE_SAMPLE_ETA_RANGE:   Tuple[float, float] = (-3.9592, -2.8367)
_FINE_SAMPLE_GAMMA_RANGE: Tuple[float, float] = ( 0.0796,  0.4388)

_FINE_CENTER_ETA   = float(np.mean(_FINE_SAMPLE_ETA_RANGE))
_FINE_CENTER_GAMMA = float(np.mean(_FINE_SAMPLE_GAMMA_RANGE))

_FINE_N_ETA   = 10
_FINE_N_GAMMA = 10
_FINE_ETA     = np.linspace(-4.1195, -2.6764, _FINE_N_ETA)  # step ~0.160
_FINE_GAMMA   = np.linspace( 0.0283,  0.4901, _FINE_N_GAMMA)  # step ~0.051

_FINE_COMBOS: List[Tuple[float, float]] = [
    (float(eta), float(gamma))
    for gamma in _FINE_GAMMA
    for eta   in _FINE_ETA
]

_FINE_ETA_RANGE   = (float(_FINE_ETA[0]),   float(_FINE_ETA[-1]))
_FINE_GAMMA_RANGE = (float(_FINE_GAMMA[0]), float(_FINE_GAMMA[-1]))

_FINE_N_GROUND_TRUTH   = 100   # number of true (eta, gamma) samples (individual nets)
_FINE_N_NETS_PER_TRUTH = 1     # individual networks per ground-truth point
_FINE_N_CONSENSUS_FINE = 20    # networks per grid point used to build the consensus
_FINE_SAMPLE_SEED      = 12345 # reproducible sampling of the ground-truth points


def _fine_sample_ground_truth(n: int = _FINE_N_GROUND_TRUTH,
                              seed: int = _FINE_SAMPLE_SEED) -> List[Tuple[float, float]]:
    """Uniformly sample n (eta, gamma) points from the plausible window."""
    rng    = np.random.default_rng(seed)
    etas   = rng.uniform(_FINE_SAMPLE_ETA_RANGE[0],   _FINE_SAMPLE_ETA_RANGE[1],   n)
    gammas = rng.uniform(_FINE_SAMPLE_GAMMA_RANGE[0], _FINE_SAMPLE_GAMMA_RANGE[1], n)
    return [(float(e), float(g)) for e, g in zip(etas, gammas)]


def _fine_normalise(eta: float, gamma: float) -> Tuple[float, float]:
    """Map (eta, gamma) to [0,1]^2 using the fine grid extent."""
    eta_n   = (eta   - _FINE_ETA_RANGE[0])   / (_FINE_ETA_RANGE[1]   - _FINE_ETA_RANGE[0])
    gamma_n = (gamma - _FINE_GAMMA_RANGE[0]) / (_FINE_GAMMA_RANGE[1] - _FINE_GAMMA_RANGE[0])
    return eta_n, gamma_n


fine = SimpleNamespace(
    CENTER_ETA=_FINE_CENTER_ETA,
    CENTER_GAMMA=_FINE_CENTER_GAMMA,
    SAMPLE_ETA_RANGE=_FINE_SAMPLE_ETA_RANGE,
    SAMPLE_GAMMA_RANGE=_FINE_SAMPLE_GAMMA_RANGE,
    GRID_N_ETA=_FINE_N_ETA,
    GRID_N_GAMMA=_FINE_N_GAMMA,
    GRID_ETA=_FINE_ETA,
    GRID_GAMMA=_FINE_GAMMA,
    GRID_COMBOS=_FINE_COMBOS,
    ETA_RANGE=_FINE_ETA_RANGE,
    GAMMA_RANGE=_FINE_GAMMA_RANGE,
    # generation settings
    GENERATIVE_RULE_NAME=GENERATIVE_RULE_NAME,
    N_GROUND_TRUTH=_FINE_N_GROUND_TRUTH,
    N_NETS_PER_TRUTH=_FINE_N_NETS_PER_TRUTH,
    N_CONSENSUS_FINE=_FINE_N_CONSENSUS_FINE,
    SAMPLE_SEED=_FINE_SAMPLE_SEED,
    # helpers
    sample_ground_truth=_fine_sample_ground_truth,
    param_dir_name=param_dir_name,
    gt_dir_name=gt_dir_name,
    net_filename=net_filename,
    normalise=_fine_normalise,
)


# ===========================================================================
# Rewiring robustness  (experiment_rewiring_robustness/)
# ===========================================================================
# Perturb the empirical reference by progressively rewiring N edges and watch how
# far each measure's recovered best-fit (eta, gamma) drifts.

_REWIRE_E_EDGES = 495    # 10% density at 100 nodes: 0.10 * 100*99/2

rewiring = SimpleNamespace(
    N_NODES=100,
    N_ETA=50,
    N_GAMMA=50,
    N_REPLICATES=10,
    E_EDGES=_REWIRE_E_EDGES,
    # grid geometry (for axis labelling / param-unit conversion only)
    ETA_RANGE=(-8.0, 3.0),
    GAMMA_RANGE=(-0.1, 1.0),
    # measures that recover on a window around p0 (rest use the full grid)
    SLOW_MEASURES={
        "portrait",
        "net_simile",
        "netrd_non_backtracking_spectral",
        "energy",
    },
    # noise ladder (number of rewiring ops). Last entry is the near-full-
    # randomisation anchor and is EXCLUDED from N* / AUC.
    DEFAULT_NOISE_LEVELS=[1, 2, 5, 10, 20, 50, 100, 200, _REWIRE_E_EDGES],
    FULL_ANCHOR=_REWIRE_E_EDGES,
    R_DEFAULT=20,
    R_SLOW=10,
    WINDOW_DEFAULT=10,            # +/- cells around p0 for slow measures
    FALLBACK_CUTOFF_DEFAULT=50,   # widen to full grid on a boundary hit only if s <= this
    DRIFT_THRESHOLD=1.0,          # grid steps; N* is the largest s with median_drift <= 1
    BASE_SEED=20260612,
)
