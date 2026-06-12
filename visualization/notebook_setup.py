from pathlib import Path
import sys
import os 
from IPython import get_ipython

from pathlib import Path 
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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

    # 
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


# Copy this into notebooks:
"""

## Setup notebook, and get paths 

from notebook_setup import setup
env = setup()

DATA_PATH = env["DATA_PATH"]
OUTPUT_PATH = env["OUTPUT_PATH"] / "05_plotting_energy_grids"
PREPROCESSED_PATH = env["PREPROCESSED_PATH"] / "01_first_analysises"

"""


# def setup_kaysons_design(): 
        
#     from vizman import viz
#     # import numpy as np
#     # import matplotlib.pyplot as plt
#     # import seaborn as sns
    
#     # import warnings
#     # import scipy
#     # import networkx as nx
#     # import seaborn as sns
#     # import utils as ut
#     # import pandas as pd
#     # import numpy as np

#     # import matplotlib.pyplot as plt
#     # import matplotlib.animation as animation
#     # import matplotlib.colors as mcolors
#     # import matplotlib.ticker as ticker

#     # from tqdm import tqdm
#     # from msapy import msa

#     # from IPython.display import HTML

#     # from scipy.stats import pearsonr, spearmanr
#     # from scipy.spatial.distance import pdist, squareform, cosine
#     # from scipy.special import factorial

#     # from sklearn.preprocessing import StandardScaler
#     # from sklearn.model_selection import train_test_split, ParameterGrid
#     # from sklearn.linear_model import LinearRegression, LassoCV

#     # from netneurotools.metrics import (
#     #     communicability_wei,
#     #     communicability_bin,
#     #     distance_wei_floyd,
#     # )
    
#     # from matplotlib import font_manager


#     for font in font_manager.findSystemFonts("figures/Atkinson_Typeface/"):
#         font_manager.fontManager.addfont(font)

#     viz.set_visual_style()
#     default_sizes = viz.load_data_from_json("sizes.json")
#     default_colors = viz.load_data_from_json("colors.json")
#     default_cmaps = viz.give_colormaps()
    
#     return default_sizes, default_colors, default_cmaps