from dataclasses import dataclass, field
from typing import List

# Import project constants
from .constants import (
    DEFAULT_RESOLUTION,
    DEFAULT_DENSITIES,
)


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    resolution: int = DEFAULT_RESOLUTION
    densities: List[int] = field(default_factory=lambda: DEFAULT_DENSITIES.copy())
    use_weighted: bool = True
    use_gnm_defaults: bool = False  # Option to use GNM's default data
