"""
Configuration management for the connectome analysis pipeline.
Centralizes all hyperparameters, paths, and experimental settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from itertools import product
import json


@dataclass
class ESNConfig:
    """ESN hyperparameter configuration."""
    spectral_radius: float = 0.99
    input_length: int = 4000
    input_scaling: float = 1.0
    regularization_method: str = "pinv"
    n_runs: int = 10
    n_lags: int = 50
    test_len: int = 1000
    n_transient: int = 0
    leak_rate: float = 1.0
    bias: float = 1.0


@dataclass
class GNMConfig:
    """GNM (Generative Network Model) configuration."""
    n_eta: int = 20
    n_gamma: int = 20
    eta_start: float = -3.0
    eta_end: float = 0.0
    gamma_start: float = 0.1
    gamma_end: float = 0.6
    subset_size: int = 8
    include_subset: bool = True


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    resolution: int = 68
    densities: List[int] = field(default_factory=lambda: [10, 12, 14, 16, 18, 20])
    use_weighted: bool = True


@dataclass
class ComputeConfig:
    """Computational settings."""
    n_workers: Optional[int] = None  # None means use all available CPUs
    timing_flag: bool = True
    append_interval: int = 10
    random_seed: int = 42


@dataclass
class PathConfig:
    """Path configuration."""
    root_dir: Path = Path("/Users/adrian/Documents/01_projects/14_4D_lab")
    
    def __post_init__(self):
        self.data_dir = self.root_dir / "data/preprocessed/01_first_analysises"
        self.output_dir = self.root_dir / "output"
        self.esn_output_dir = self.output_dir / "02_esns_on_observed_weighted_connectomes"
        self.gnm_output_dir = self.output_dir / "02_gnm_estimation"


class ConfigManager:
    """Manages all configuration aspects of the pipeline."""
    
    def __init__(self, 
                 esn_config: Optional[ESNConfig] = None,
                 gnm_config: Optional[GNMConfig] = None,
                 data_config: Optional[DataConfig] = None,
                 compute_config: Optional[ComputeConfig] = None,
                 path_config: Optional[PathConfig] = None):
        
        self.esn = esn_config or ESNConfig()
        self.gnm = gnm_config or GNMConfig()
        self.data = data_config or DataConfig()
        self.compute = compute_config or ComputeConfig()
        self.paths = path_config or PathConfig()
    
    def generate_esn_hparam_grid(self, 
                                spectral_radii: Optional[List[float]] = None,
                                input_lengths: Optional[List[int]] = None,
                                input_scalings: Optional[List[float]] = None,
                                regularization_methods: Optional[List[str]] = None,
                                n_runs_list: Optional[List[int]] = None,
                                densities: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Generate hyperparameter grid for ESN experiments."""
        
        # Use defaults if not provided
        spectral_radii = spectral_radii or np.linspace(0.1, 2.5, 10).tolist()
        input_lengths = input_lengths or [1000, 2000, 4000]
        input_scalings = input_scalings or [0.5, 1.0, 2.0]
        regularization_methods = regularization_methods or ["pinv", "ridge"]
        n_runs_list = n_runs_list or [self.esn.n_runs]
        densities = densities or self.data.densities
        
        hparam_grid = [
            {
                "spectral_radius": sr,
                "input_length": ilen,
                "input_scaling": iscale,
                "regularization_method": reg,
                "n_runs": nr,
                "density_percent": dens,
            }
            for sr, ilen, iscale, reg, nr, dens in product(
                spectral_radii, input_lengths, input_scalings, 
                regularization_methods, n_runs_list, densities
            )
        ]
        
        return hparam_grid
    
    def get_data_paths(self):
        """Get paths to data files based on current configuration."""
        resolution = self.data.resolution
        
        paths = {
            'weighted_connectome': self.paths.data_dir / f"connectomes_weighted_{resolution}x{resolution}.npy",
            'distance_matrix': self.paths.data_dir / f"distance_matrix_{resolution}x{resolution}.npy",
            'binary_connectomes': {}
        }
        
        for density in self.data.densities:
            paths['binary_connectomes'][density] = (
                self.paths.data_dir / f"connectomes_binarized_{resolution}x{resolution}_density_{density}_percent.npy"
            )
        
        return paths
    
    def save_config(self, save_path: Path):
        """Save current configuration to JSON file."""
        config_dict = {
            'esn': self.esn.__dict__,
            'gnm': self.gnm.__dict__,
            'data': self.data.__dict__,
            'compute': self.compute.__dict__,
            'paths': {
                'root_dir': str(self.paths.root_dir),
                'data_dir': str(self.paths.data_dir),
                'output_dir': str(self.paths.output_dir),
                'esn_output_dir': str(self.paths.esn_output_dir),
                'gnm_output_dir': str(self.paths.gnm_output_dir),
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, config_path: Path):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct the configuration objects
        esn_config = ESNConfig(**config_dict['esn'])
        gnm_config = GNMConfig(**config_dict['gnm'])
        data_config = DataConfig(**config_dict['data'])
        compute_config = ComputeConfig(**config_dict['compute'])
        path_config = PathConfig(root_dir=Path(config_dict['paths']['root_dir']))
        
        return cls(esn_config, gnm_config, data_config, compute_config, path_config)


# # Example usage and default configurations
# def get_quick_test_config() -> ConfigManager:
#     """Get a configuration suitable for quick testing."""
#     esn_config = ESNConfig(
#         spectral_radius=0.99,
#         input_length=1000,
#         n_runs=3
#     )
    
#     data_config = DataConfig(
#         resolution=68,
#         densities=[10, 20]  # Just two densities for quick test
#     )
    
    
#     compute_config = ComputeConfig(
#         timing_flag=True,
#         n_workers=2  # Limit workers for testing
#     )
    
#     return ConfigManager(esn_config, data_config, compute_config)

def get_quick_test_config() -> ConfigManager:
    """Get a configuration suitable for quick testing."""
    esn_config = ESNConfig(
        spectral_radius=0.99,
        input_length=1000,
        n_runs=3
    )
    
    data_config = DataConfig(
        densities=[10, 20]  # Just two densities for quick test
    )
    
    compute_config = ComputeConfig(
        timing_flag=True,
        n_workers=2  # Limit workers for testing
    )
    
    # Fixed: proper parameter passing
    return ConfigManager(
        esn_config=esn_config, 
        gnm_config=None,  # Will use default
        data_config=data_config, 
        compute_config=compute_config,
        path_config=None  # Will use default
    )
    

def get_production_config() -> ConfigManager:
    """Get a configuration suitable for production runs."""
    esn_config = ESNConfig(
        n_runs=100,  # More runs for better statistics
        input_length=4000
    )
    
    gnm_config = GNMConfig(
        n_eta=100,
        n_gamma=100
    )
    
    return ConfigManager(esn_config, gnm_config)


def get_hyperparameter_sweep_config() -> ConfigManager:
    """Get a configuration for extensive hyperparameter sweeping."""
    data_config = DataConfig(
        densities=[10, 12, 14, 16, 18, 20]
    )
    
    compute_config = ComputeConfig(
        timing_flag=True,
        random_seed=42
    )
    
    return ConfigManager(data_config=data_config, compute_config=compute_config)