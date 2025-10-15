"""
YAML-based configuration system integrated with existing ConfigManager.
This bridges YAML files with the existing config.py structure.
"""

# TODO: 
    # - Save in run_info.txt the actual config used (including lists for grid search)
    
import yaml
from pathlib import Path

from config.manager import ConfigManager, PathConfig

class YAMLConfigLoader:
    """Load and merge YAML configurations with existing ConfigManager."""
    
    def __init__(self, yaml_path: Path):
        """Initialize with path to YAML config file."""
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(self.yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate required sections
        self._validate_config()
    
    def _validate_config(self):
        """Validate that config has required structure."""
        required_sections = ['experiment', 'data', 'gnm', 'compute', 'output']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section '{section}' in config")
        
        # Validate experiment type
        if 'experiment_type' not in self.config['experiment']:
            raise ValueError("Missing 'experiment_type' in experiment section")
        
        valid_types = ['gnm_sweep', 'dynamic_gnm']
        if self.config['experiment']['experiment_type'] not in valid_types:
            raise ValueError(f"Invalid experiment type. Must be one of: {valid_types}")
    
    
    def create_config_manager(self) -> ConfigManager:
        """Create a ConfigManager instance from YAML configuration."""
        
        config = self.config
        path_config = PathConfig(
            dataset_name=self.config['data']['dataset_name'], 
            experiment_name=self.config['experiment']['name'], 
            animal=self.config['experiment']['animal']
        )
        
        # self._create_path_config()
        config.update(path_config.to_dict())
        
        return config
    
