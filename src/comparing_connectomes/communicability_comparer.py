import torch
import numpy as np
from scipy.linalg import expm
from typing import Dict
from src.comparing_connectomes.base_comparer import NetworkEvaluator


class CommunicabilityCorrEvaluator(NetworkEvaluator):
    """Evaluate networks via communicability correlation."""

    def __init__(self):
        pass

    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        results = {}
        gen_exp = expm(generated.squeeze(0).cpu().numpy())
        n_targets = target.shape[0]

        for i in range(n_targets):
            tar_exp = expm(target[i].cpu().numpy())
            # Pearson correlation of flattened matrices
            corr = np.corrcoef(gen_exp.flatten(), tar_exp.flatten())[0, 1]
            results[i] = corr

        return results

    @property
    def metric_prefix(self) -> str:
        return "CommCorr"
