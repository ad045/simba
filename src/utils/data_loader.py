"""
Data loading and preprocessing utilities for the connectome analysis pipeline.
Handles loading of connectomes, distance matrices, and data validation.
"""
# -> Used at least for main_pipeline_2_gnm_esn_landscape.py and for main_pipeline_2.py (esn part)

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import warnings
from config.manager import ConfigManager
from src.config.constants import NUMERICAL_TOLERANCE


class DataLoader:
    """Handles loading and preprocessing of connectome data."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.resolution = self.config['data']['connectome_resolution']
       
       
        # PosixPath('/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/shafiei_human_consensus_dataset')
    #    self.config['paths']['connectome_dir']
    # PosixPath('/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/shafiei_human_consensus_dataset')
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/shafiei_human_consensus_dataset/02_distance_matrices/distance_matrix_68x68.npy
       
    def load_distance_matrix(self) -> np.ndarray:
        """Load the distance matrix."""
        try:
            dist_matrix = np.load(self.config["paths"]['02_distance_matrices'] / f"distance_matrix_{self.resolution}.npy")
            
            if self.config["data"]["dataset_name"] == "suarez_MaMI_dataset": 
                dist_matrix = dist_matrix[0,:,:] # TODO: THIS ONLY USES THE FIRST CONNECTOME TO GET ENERGY. 
            # ATTENTION: REALLY CHANGE THIS! 
            self._validate_distance_matrix(dist_matrix)
            
            
            return dist_matrix
        except Exception as e:
            raise RuntimeError(f"Failed to load distance matrix: {e}")
    
    def load_weighted_connectome(self) -> np.ndarray:
        """Load the weighted connectome data."""
        try:
            weighted_conn = np.load(self.config['paths']['01_connectomes'] / f"01_consensus_wei_{self.resolution}.npy") # subj, resolution, resolution
            self._validate_connectome(weighted_conn, "weighted")
            return weighted_conn
        except Exception as e:
            raise RuntimeError(f"Failed to load weighted connectome: {e}")
    
    def load_binary_connectomes(self) -> Dict[int, np.ndarray]:
        """Load all binary connectomes for specified densities."""
        binary_connectomes = {}
        
        for density in self.config["data"]["densities"]: # .{}.densities:
            try:
                resolution = self.config['data']['connectome_resolution']
                
                # TODO: Fix this HACK or put it somewhere else? Or good where it is?
                if self.config["data"]["dataset_name"] == "shafiei_human_consensus_dataset":  
                    binary_conn_path = self.config['paths']['connectome_dir'] / f"01_connectomes/01_consensus_bin_density_{density}_percent_{resolution}.npy"
                    binary_conn = np.load(binary_conn_path)
            
                elif self.config["data"]["dataset_name"] == "suarez_MaMI_dataset": 
                    binary_conn_path = self.config['paths']['connectome_dir'] / f"01_connectomes/00_connectomes_{resolution}.npy" # 01_consensus_bin_density_{density}_percent_{resolution}.npy"
                    print("BINARY_CONN: ", binary_conn_path)
                    binary_conn = np.load(binary_conn_path)[0,:,:] # TODO: THIS ONLY USES THE FIRST CONNECTOME TO GET ENERGY. 

                else: 
                    print("Experiment name was", self.config["data"]["dataset_name"], "but this is not defined. ")
                
                
                binary_conn = np.load(binary_conn_path) 
                self._validate_connectome(binary_conn, f"binary (density {density}%)")
                binary_connectomes[density] = binary_conn
                
            except Exception as e:
                warnings.warn(f"Failed to load binary connectome for density {density}%: {e}")
                continue
        
        if not binary_connectomes:
            raise RuntimeError("No binary connectomes could be loaded")
        
        return binary_connectomes
    
    def load_weighted_by_density(self) -> Dict[int, np.ndarray]:
        """
        Load connectomes that combine binary structure with weighted values.
        Returns dict mapping density -> (binary_mask * weighted_values).T
        """
        weighted_connectome = self.load_weighted_connectome()
        binary_connectomes = self.load_binary_connectomes()
        
        weighted_by_density = {}
        
        for density, binary_conn in binary_connectomes.items():
            try:
                # Element-wise multiplication: binary structure × weighted values
                combined = (binary_conn * weighted_connectome).T.astype(np.float64, copy=False)
                weighted_by_density[density] = combined
            except Exception as e:
                warnings.warn(f"Failed to create weighted connectome for density {density}%: {e}")
                continue
        
        return weighted_by_density
    
    def _validate_distance_matrix(self, dist_matrix: np.ndarray):
        """Validate distance matrix properties."""
        if dist_matrix.ndim != 2:
            raise ValueError(f"Distance matrix must be 2D, got {dist_matrix.ndim}D")
        
        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise ValueError(f"Distance matrix must be square, got shape {dist_matrix.shape}")
        
        if dist_matrix.shape[0] != self.config['data']['connectome_resolution']: # self.config.data.resolution:
            raise ValueError(f"Distance matrix size {dist_matrix.shape[0]} doesn't match "
                           f"expected resolution {self.config.data.resolution}")
        
        # Check symmetry (within tolerance)
        if not np.allclose(dist_matrix, dist_matrix.T, rtol=NUMERICAL_TOLERANCE):
            warnings.warn("Distance matrix is not symmetric")
        
        # Check non-negativity
        if np.any(dist_matrix < 0):
            warnings.warn("Distance matrix contains negative values")
        
        # Check diagonal is zero
        if not np.allclose(np.diag(dist_matrix), 0, atol=NUMERICAL_TOLERANCE):
            warnings.warn("Distance matrix diagonal is not zero")
    
    def _validate_connectome(self, connectome: np.ndarray, conn_type: str):
        """Validate connectome properties."""
        expected_shape = (self.config['data']['connectome_resolution'], self.config['data']['connectome_resolution']) # , self.config.data.resolution)
        
        if connectome.ndim == 2:
            if connectome.shape != expected_shape:
                raise ValueError(f"Single connectome ({conn_type}) has shape {connectome.shape}, "
                               f"expected {expected_shape}")
        elif connectome.ndim == 3:
            if connectome.shape[1:] != expected_shape:
                raise ValueError(f"Connectome stack ({conn_type}) has shape {connectome.shape}, "
                               f"expected {expected_shape} for first two dimensions")
        else:
            raise ValueError(f"Connectome ({conn_type}) must be 2D or 3D, got {connectome.ndim}D")
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary information about loaded data."""
        summary = {
            'resolution': self.config.data.resolution,
            'available_densities': [],
            'data_shapes': {},
            'file_paths': self.paths
        }
        
        try:
            # Distance matrix info
            dist_matrix = self.load_distance_matrix()
            summary['data_shapes']['distance_matrix'] = dist_matrix.shape
            
            # Weighted connectome info
            weighted_conn = self.load_weighted_connectome()
            summary['data_shapes']['weighted_connectome'] = weighted_conn.shape
            
            # Binary connectomes info
            binary_connectomes = self.load_binary_connectomes()
            summary['available_densities'] = sorted(binary_connectomes.keys())
            
            for density, conn in binary_connectomes.items():
                summary['data_shapes'][f'binary_density_{density}'] = conn.shape
                
        except Exception as e:
            summary['error'] = str(e)
        
        return summary


def create_data_loader(config_path: Optional[Union[str, Path]] = None, 
                      config_manager: Optional[ConfigManager] = None) -> DataLoader:
    """
    Factory function to create a DataLoader instance.
    
    Args:
        config_path: Path to JSON configuration file
        config_manager: Pre-configured ConfigManager instance
        
    Returns:
        DataLoader instance
    """
    if config_manager is not None:
        return DataLoader(config_manager)
    elif config_path is not None:
        config_manager = ConfigManager.load_config(Path(config_path))
        return DataLoader(config_manager)
    else:
        # Use default configuration
        config_manager = ConfigManager()
        return DataLoader(config_manager)


# Utility functions for common data operations
def binarize_connectome(connectome: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Binarize a weighted connectome.
    
    Args:
        connectome: Input connectome matrix
        threshold: Threshold for binarization
        
    Returns:
        Binary connectome
    """
    # Symmetrize
    conn_sym = (connectome + connectome.T) / 2.0
    # Remove diagonal
    np.fill_diagonal(conn_sym, 0.0)
    # Binarize
    return (conn_sym > threshold).astype(np.int8)


def symmetrize_connectome(connectome: np.ndarray, method: str = "max") -> np.ndarray:
    """
    Symmetrize a connectome matrix.
    
    Args:
        connectome: Input connectome matrix
        method: Symmetrization method ("max", "mean", "min")
        
    Returns:
        Symmetrized connectome
    """
    if method == "max":
        return np.maximum(connectome, connectome.T)
    elif method == "mean":
        return (connectome + connectome.T) / 2.0
    elif method == "min":
        return np.minimum(connectome, connectome.T)
    else:
        raise ValueError(f"Unknown symmetrization method: {method}")


def get_connectome_stats(connectome: np.ndarray) -> Dict[str, float]:
    """Get basic statistics for a connectome."""
    # Remove diagonal for stats calculation
    conn_nodiag = connectome.copy()
    np.fill_diagonal(conn_nodiag, 0.0)
    
    stats = {
        'density': np.sum(conn_nodiag > 0) / (connectome.size - connectome.shape[0]),
        'mean_weight': np.mean(conn_nodiag[conn_nodiag > 0]) if np.any(conn_nodiag > 0) else 0.0,
        'std_weight': np.std(conn_nodiag[conn_nodiag > 0]) if np.any(conn_nodiag > 0) else 0.0,
        'max_weight': np.max(conn_nodiag),
        'min_weight': np.min(conn_nodiag[conn_nodiag > 0]) if np.any(conn_nodiag > 0) else 0.0,
        'total_edges': np.sum(conn_nodiag > 0),
        'is_symmetric': np.allclose(connectome, connectome.T, rtol=NUMERICAL_TOLERANCE)
    }
    
    return stats


# Example usage
if __name__ == "__main__":
    # Create a data loader with default configuration
    config = ConfigManager()
    loader = DataLoader(config)
    
    # Print data summary
    summary = loader.get_data_summary()
    print("Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Load and inspect distance matrix
    try:
        dist_matrix = loader.load_distance_matrix()
        print(f"\nDistance matrix loaded: shape {dist_matrix.shape}")
        print(f"Distance range: {dist_matrix.min():.3f} to {dist_matrix.max():.3f}")
    except Exception as e:
        print(f"Could not load distance matrix: {e}")
    
    # Load weighted connectomes by density
    try:
        weighted_by_density = loader.load_weighted_by_density()
        print(f"\nLoaded weighted connectomes for densities: {sorted(weighted_by_density.keys())}")
        
        for density, conn in weighted_by_density.items():
            stats = get_connectome_stats(conn[:, :, 0] if conn.ndim == 3 else conn)
            print(f"  Density {density}%: {conn.shape}, density={stats['density']:.3f}")
            
    except Exception as e:
        print(f"Could not load weighted connectomes: {e}")