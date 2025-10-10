from dataclasses import dataclass

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    def __init__(self,         
                 dataset_name: str, 
                 resolution: int, 
                 density: int, 
                 use_weighted: bool,
                ):
    
        self.dataset_name = dataset_name
        self.resolution = resolution
        self.density = density 
        self.use_weighted = use_weighted 
                    
                