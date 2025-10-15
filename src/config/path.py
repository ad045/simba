from pathlib import Path
from dataclasses import dataclass

@dataclass
class PathConfig:
    """Path configuration."""
    dataset_name: str
    experiment_name: str 
    animal: int = None
    root_dir: Path = None
    connectome_dir: Path = None
    output_dir: Path = None
    output_specific_dataset_dir: Path = None
    output_gnm_dir: Path = None
    dynamic_gnm_output_dir: Path = None
    
    
    def __post_init__(self): # instead of init, due to dataclass
        if self.root_dir is None:
            self.root_dir = Path.cwd().resolve()
            
        self.connectome_dir = self.root_dir / "data" / "preprocessed" / self.dataset_name 
        self.dir_01_connectomes = self.connectome_dir / "01_connectomes"
        self.dir_02_distance_matrices = self.connectome_dir / "02_distance_matrices"
        
        # if self.animal is not None: 
        #     self.output_dir = self.root_dir / f"output_{self.animal}" 
        # else: 
        self.output_dir = self.root_dir / "output" 
        self.output_specific_dataset_dir = self.output_dir / "empirical_data" / self.dataset_name
        
        self.output_gnm_dir = self.output_dir / "gnm" # make this smoother
        if self.animal is not None: # TODO: add "self.experiment_name" as folder in between... it will destroy something else, though. 
            self.output_experiment_dir = self.output_gnm_dir / self.dataset_name / f"{self.experiment_name}" # _{self.animal}" #+ f"_{time.strftime("%Y%m%d_%H%M%S")}")
        else: 
            self.output_experiment_dir = self.output_gnm_dir / self.dataset_name / self.experiment_name #+ f"_{time.strftime("%Y%m%d_%H%M%S")}")
            
        self.dynamic_gnm_output_dir = self.output_dir / "dynamic_gnm" / self.experiment_name
        

        # Create directories
        for dir_path in [self.connectome_dir, self.output_specific_dataset_dir, 
                         self.output_gnm_dir, self.dynamic_gnm_output_dir,
                         self.output_experiment_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    
    def to_dict(self) -> dict:
        """Return paths as a dictionary with string keys and Path values."""
        return {'paths': 
                        {
                            'root_dir': self.root_dir,
                            'connectome_dir': self.connectome_dir,
                            'output_dir': self.output_dir,
                            'output_specific_dataset_dir': self.output_specific_dataset_dir,
                            'output_gnm_dir': self.output_gnm_dir,
                            'output_experiment_dir': self.output_experiment_dir,
                            'dynamic_gnm_output_dir': self.dynamic_gnm_output_dir,
                            
                            'output_experiment_dir': self.output_experiment_dir,
                             
                            '01_connectomes': self.dir_01_connectomes,
                            '02_distance_matrices': self.dir_02_distance_matrices,
                        
                        }
                }