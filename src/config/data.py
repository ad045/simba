from dataclasses import dataclass, field
from typing import List

# Import project constants
from .constants import (
    DEFAULT_RESOLUTION,
    DEFAULT_DENSITIES,
    DEFAULT_DATASET_NAME
)


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset_name: str = DEFAULT_DATASET_NAME 
    resolution: int = DEFAULT_RESOLUTION
    densities: List[int] = field(default_factory=lambda: DEFAULT_DENSITIES.copy())
    use_weighted: bool = True
