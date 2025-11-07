import torch
from typing import Dict, List
from src.comparing_connectomes.base_comparer import NetworkEvaluator

class EnergyEvaluator(NetworkEvaluator):
    """Evaluate networks using GNM energy-based criteria."""
    
    def __init__(self, evaluation_criteria_list: List):
        self.evaluation_criteria_list = evaluation_criteria_list
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute energy for single generated network against target batch."""
        results = {}
        n_targets = target.shape[0]
        
        for target_idx in range(n_targets):
            target_single = target[target_idx:target_idx+1, :, :]
            criteria = self.evaluation_criteria_list[0]
            energy_dict = criteria(generated, target_single)
            
            for energy_value in energy_dict:
                results[target_idx] = float(energy_value.mean().item())
        
        return results
    
    @property
    def metric_prefix(self) -> str:
        return "MaxCrit"