# Did check over it, but did not try it yet. 
# TODO: Is this too hidden, or do people find it sufficiently easy to find? 

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from src.utils.config import PlotConfig, PathConfig


class PlotManager:
    """Centralized plot configuration and management."""
    
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.plot_config = config or PlotConfig()
        self.path_config = PathConfig()
        self._setup_matplotlib()
    
    
    def _setup_matplotlib(self):
        """Apply global matplotlib settings."""
        # plt.style.use(self.plot_config.style)
        plt.rcParams.update({                                           # TODO: Add these settings again... 
            # 'figure.dpi': self.plot_config.dpi,
            # 'savefig.dpi': self.plot_config.dpi,
            # 'font.family': self.plot_config.font_family,
            # 'font.size': self.plot_config.font_size_base,
            # 'axes.titlesize': self.plot_config.font_size_title,
            # 'axes.labelsize': self.plot_config.font_size_label,
            # 'xtick.labelsize': self.plot_config.font_size_tick,
            # 'ytick.labelsize': self.plot_config.font_size_tick,
        })
        
        # Set seaborn context
        # sns.set_context("paper", font_scale=1.2)


    def create_figure(self, nrows=1, ncols=1, plot_type='single'):
        """Create figure with predefined size."""
        sizes = {
            'single': self.plot_config.figsize_single,
            'double': self.plot_config.figsize_double,
            'landscape': self.plot_config.figsize_landscape, 
            'huge': self.plot_config.figsize_huge
        }
        return plt.subplots(nrows=nrows, ncols=ncols, figsize=sizes.get(plot_type, self.plot_config.figsize_single))

    
    def get_figure_path(self, name: str, experiment: Optional[str] = None) -> Path:
        """Get standardized figure path."""
        if experiment:
            fig_dir = self.path_config.experiments_dir / experiment / "figures"
        else:
            fig_dir = self.path_config.figures_dir

        fig_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir / f"{name}.{self.plot_config.figure_format}"