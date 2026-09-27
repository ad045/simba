from abc import ABC, abstractmethod
import torch
from typing import Dict

class NetworkEvaluator(ABC):
    """Base class for network evaluation metrics."""
    
    @abstractmethod
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate generated network against target.
        
        Args:
            generated: (batch, n_nodes, n_nodes)
            target: (batch, n_nodes, n_nodes)
            
        Returns:
            Dictionary of metric_name: value
        """
        pass
    
    @property
    @abstractmethod
    def metric_prefix(self) -> str:
        """Prefix for column names (e.g., 'MaxCrit', 'PortraitDiv')."""
        pass