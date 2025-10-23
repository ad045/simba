
import re
from pathlib import Path
from typing import List, Dict


def parse_filename(filepath: str) -> Dict[str, float]:
    """Extract parameters from filename."""
    filename = Path(filepath).stem
    params = {}
    
    # Extract eta
    eta_match = re.search(r'eta-([\d.]+)', filename)
    if eta_match:
        params['eta'] = float(eta_match.group(1))
    
    # Extract gamma
    gamma_match = re.search(r'gamma([\d.]+)', filename)
    if gamma_match:
        params['gamma'] = float(gamma_match.group(1))
    
    # Extract rule
    # rule_match = re.search(r'rule(\w+)', filename)
    # if rule_match:
    #     params['generative_rule'] = rule_match.group(1)
    
    return params


def find_network_files(base_dir: str, pattern: str = "*.npy") -> List[str]:
    """Recursively find all network files."""
    base_path = Path(base_dir)
    return sorted([str(f) for f in base_path.rglob(pattern)])
