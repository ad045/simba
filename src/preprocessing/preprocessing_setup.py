from pathlib import Path
import sys
import os 
from IPython import get_ipython


def find_project_root(start_path: Path = None) -> Path:
    """Find project root by looking for common markers."""
    if start_path is None:
        start_path = Path.cwd().resolve()
    
    current = start_path
    for parent in (current, *current.parents):
        if any((parent / marker).exists() for marker in ("pyproject.toml", "setup.cfg", ".git")):
            return parent
    return current


# Find project root dynamically
PROJECT_ROOT = find_project_root()
DATA_PATH = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "output"
PREPROCESSED_PATH = DATA_PATH / "preprocessed"

# Create directories if they don't exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PREPROCESSED_PATH.mkdir(parents=True, exist_ok=True)


def setup():
    """
    Find project root and add src to paths 
    """
    # Find project root
    root = find_project_root()

    # Add source to path 
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
        print("Added src path.")

    # Enable autoreload if in IPython (for Jupyter notebooks)
    ip = get_ipython()
    if ip:
        try:
            ip.run_line_magic("load_ext", "autoreload")
        except Exception:
            pass
        ip.run_line_magic("autoreload", "2")
        print("Enabled autoreload.")

    print("Current path is:", Path.cwd())
    return dict(
        PROJECT_ROOT=root,
        SRC_PATH=src,
        DATA_PATH=DATA_PATH,
        OUTPUT_PATH=OUTPUT_PATH,
        PREPROCESSED_PATH=PREPROCESSED_PATH,
    )


# Copy this into notebooks (not relevant for scripts):
"""

## Setup notebook, and get paths 

from notebook_setup import setup
env = setup()

DATA_PATH = env["DATA_PATH"]
OUTPUT_PATH = env["OUTPUT_PATH"] / "05_plotting_energy_grids"
PREPROCESSED_PATH = env["PREPROCESSED_PATH"] / "01_first_analysises"

"""
