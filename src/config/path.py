# from pathlib import Path
# from dataclasses import dataclass, field


# @dataclass
# class PathConfig:
#     """Path configuration."""
#     # root_dir: Path # = field(default_factory=lambda: Path.cwd().resolve())
#     # dataset_name: str 
    
#     def __init__(self, dataset_name: str):
#         # Find project root by looking for markers
#         # current = self.root_dir
#         # for parent in (current, *current.parents):
#         #     if any((parent / marker).exists() for marker in ("pyproject.toml", "setup.cfg", ".git")):
#         #         self.root_dir = parent
#         #         break
        
#         self.dataset_name = dataset_name
        
#         self.root_dir = Path.cwd().resolve()
        
#         # self.data_dir = self.root_dir / "14_4D_lab_code" / "data/preprocessed/01_first_analysises" # TODO TODO !!! 
#         # self.data_dir = self.root_dir / "data/preprocessed/01_first_analysises"
#         self.connectome_dir = self.root_dir / "data" / "preprocessed" / self.dataset_name 
#         self.output_dir = self.root_dir / "output" 
#         self.output_specific_dataset_dir = self.output_dir / "empirical_data" / self.dataset_name
#         self.output_gnm_dir = self.output_dir / "gnm"
#         self.dynamic_gnm_output_dir = self.output_dir / "dynamic_gnm"
        
#         self.connectome_dir.mkdir(parents=True, exist_ok=True)
#         self.output_specific_dataset_dir.mkdir(parents=True, exist_ok=True)
#         self.output_gnm_dir.mkdir(parents=True, exist_ok=True)
#         self.dynamic_gnm_output_dir.mkdir(parents=True, exist_ok=True)
    



# # # Create an instance
# # path_config = PathConfig(dataset_name="mami")

# # # Access the paths
# # print(path_config.connectome_dir)
# # print(path_config.output_specific_dataset_dir) 
# # print(path_config.output_gnm_dir)
# # print(path_config.root_dir)

# # # Use them in your code
# # data_file = path_config.connectome_dir / "connectivity_matrix.npy"
# # output_file = path_config.output_specific_dataset_dir / "results.csv"



#     # def get_connectome_file(self, filename: str) -> Path:
#     #     """Helper to get a file in the connectome directory."""
#     #     return self.connectome_dir / filename
    
#     # def get_output_file(self, filename: str) -> Path:
#     #     """Helper to get a file in the output directory."""
#     #     return self.output_specific_dataset_dir / filename




from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class PathConfig:
    """Path configuration."""
    dataset_name: str
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
        self.output_dir = self.root_dir / "output" 
        self.output_specific_dataset_dir = self.output_dir / "empirical_data" / self.dataset_name
        self.output_gnm_dir = self.output_dir / "gnm"
        self.dynamic_gnm_output_dir = self.output_dir / "dynamic_gnm"
        
        # Create directories
        for dir_path in [self.connectome_dir, self.output_specific_dataset_dir, 
                         self.output_gnm_dir, self.dynamic_gnm_output_dir]:
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
                            'dynamic_gnm_output_dir': self.dynamic_gnm_output_dir
                        }
                }