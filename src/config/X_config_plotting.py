# TODO: Turn this into a yaml config file! 
# TODO: Go to src/visualization/init and un-comment the relevant lines again!

from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    DEFAULT_COLORMAP_CONTINUOUS, DEFAULT_COLORMAP_DIVERGING, DEFAULT_FIGURE_FORMAT, DEFAULT_DPI,
    DEFAULT_FIGSIZE_SINGLE, DEFAULT_FIGSIZE_DOUBLE, DEFAULT_FIGSIZE_LANDSCAPE, DEFAULT_FIGSIZE_HUGE
)

@dataclass
class PlotConfig:
    """Global plotting configuration."""
    # Style settings
    style: str = "default" # "seaborn-v0_8-darkgrid"
    figure_format: str = DEFAULT_FIGURE_FORMAT
    dpi: int = DEFAULT_DPI
    
    # Default sizes
    figsize_single: tuple = DEFAULT_FIGSIZE_SINGLE
    figsize_double: tuple = DEFAULT_FIGSIZE_DOUBLE
    figsize_landscape: tuple = DEFAULT_FIGSIZE_LANDSCAPE
    figsize_huge: tuple = DEFAULT_FIGSIZE_HUGE
    
    # Color schemes
    colormap_continuous: str = DEFAULT_COLORMAP_CONTINUOUS
    colormap_diverging: str = DEFAULT_COLORMAP_DIVERGING
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
    root_dir: Path = field(default_factory=lambda: Path.cwd().resolve())
    
    def __post_init__(self):
        # Find project root by looking for markers
        current = self.root_dir
        for parent in (current, *current.parents):
            if any((parent / marker).exists() for marker in ("pyproject.toml", "setup.cfg", ".git")):
                self.root_dir = parent
                break
        
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