from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset_name: str 
    resolution: int
    densities: List[int]
    use_weighted: bool
