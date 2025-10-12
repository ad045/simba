from dataclasses import dataclass
from typing import Optional

# Import project constants
from .constants import (
    DEFAULT_RANDOM_SEED, 
    DEFAULT_APPEND_INTERVAL
)

@dataclass
class ComputeConfig:
    """Computational settings."""
    n_workers: Optional[int] = None  
    timing_flag: bool = True
    append_interval: int = DEFAULT_APPEND_INTERVAL
    random_seed: int = DEFAULT_RANDOM_SEED

