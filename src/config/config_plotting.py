# TODO: Turn this into a yaml config file! 
# TODO: Go to src/visualization/init and un-comment the relevant lines again!

from dataclasses import dataclass
from pathlib import Path

@dataclass
class PlotConfig:
    """Global plotting configuration."""
    # Style settings
    style: str = "default" # "seaborn-v0_8-darkgrid"
    figure_format: str = "pdf" # 'png', 'pdf', 'svg'
    dpi: int = 300
    
    # Default sizes
    figsize_single: tuple = (8, 6)
    figsize_double: tuple = (12, 6)
    figsize_landscape: tuple = (10, 8)
    figsize_huge: tuple = (18, 5) # is for example used in the comparison between energy fields 
    
    # Color schemes
    colormap_continuous: str = "viridis"
    colormap_diverging: str = "RdBu_r"
    color_empirical: str = "#2E86AB"
    color_synthetic: str = "#A23B72"
    
    # Font settings                                 # TODO: Add these settings again...
    # font_family: str = "sans-serif"
    # font_size_base: int = 8 # 12
    # font_size_title: int = 14
    # font_size_label: int = 12
    # font_size_tick: int = 10

@dataclass
class PathConfig:
    """Path configuration with plot output paths."""
    root_dir: Path = Path("/Users/adrian/Documents/01_projects/14_4D_lab")
    
    def __post_init__(self):
        # Data paths
        self.data_dir = self.root_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.results_dir = self.data_dir / "results"
        
        # Output paths
        self.output_dir = self.root_dir / "output"
        self.figures_dir = self.output_dir / "figures"
        self.esn_output_dir = self.output_dir / "esn"
        self.gnm_output_dir = self.output_dir / "gnm"
        
        # Experiment paths
        self.experiments_dir = self.root_dir / "experiments"
        
        # Ensure key directories exist
        for dir_path in [self.output_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)