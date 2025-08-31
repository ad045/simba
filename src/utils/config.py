import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = None):
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to configs/default.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "default.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """Get config value using dot notation (e.g., 'gnm.eta_range')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def __getitem__(self, key: str):
        """Allow dictionary-style access."""
        return self.get(key)
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})
    
    @property
    def gnm(self) -> Dict[str, Any]:
        """Get GNM configuration."""
        return self._config.get('gnm', {})
    
    @property
    def esn(self) -> Dict[str, Any]:
        """Get ESN configuration."""
        return self._config.get('esn', {})