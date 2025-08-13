import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_mystyle():
    """
    Set context and a couple of defaults for nicer plots.
    """
    
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.4,
        rc={"grid.linestyle": "--", "grid.linewidth": 0.8},
    )
