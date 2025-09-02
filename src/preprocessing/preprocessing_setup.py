from pathlib import Path
import sys
import os 
from IPython import get_ipython


# Define paths here 
DATA_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/data")
OUTPUT_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/output")

PREPROCESSED_PATH = DATA_PATH / "preprocessed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PREPROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# create output path if it does not exist
if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True)

if not PREPROCESSED_PATH.exists():
    PREPROCESSED_PATH.mkdir(parents=True)


def setup():
    """
    Find project root and add src to paths 
    """
    # Find project root
    p = Path.cwd().resolve()
    for parent in (p, *p.parents):
        if any((parent / m).exists() for m in ("pyproject.toml", "setup.cfg", ".git")):
            root = parent
            break
    else:
        root = p

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

    print("Current path is:", p)
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
