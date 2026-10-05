"""simba_networks - score a network distance measure against the benchmark of
*How Similar Are Two Brains?* (Dendorfer et al.)."""

from .data import (distance_matrix, empirical_summary, load_networks, load_reference,
                   load_table, reference_dir)
from .benchmark import CRITERIA, READOUTS, Report, evaluate, published_readouts
from .measures import MEASURES, NAMES, SELECTED, SIMILARITIES

__version__ = "1.0.0"
