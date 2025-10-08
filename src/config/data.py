from dataclasses import dataclass
from typing import List





@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    def __init__(self,         
                 dataset_name: str, 
                 resolution: int, 
                 densities: List[int], 
                 use_weighted: bool,
                ):
    
        self.dataset_name = dataset_name
        self.resolution = resolution
        self.densities = densities
        self.use_weighted = use_weighted 
                    
                




