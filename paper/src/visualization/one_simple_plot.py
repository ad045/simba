from vizman import viz
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

from matplotlib import font_manager
for font in font_manager.findSystemFonts("figures/Atkinson_Typeface/"):
    font_manager.fontManager.addfont(font)

viz.set_visual_style()
default_sizes = viz.load_data_from_json("sizes.json")
default_colors = viz.load_data_from_json("colors.json")
default_cmaps = viz.give_colormaps()

# from kaysons_visual_config import * # Kayson's file. 


def plot_one_aesthetic_plot(x_series, 
                            y_series, 
                            c_series, 
                            save_path=None, 
                            dot_color=default_colors["warms"]["LECKER_RED"], 
                            cmap=default_cmaps["metric_purple_beige"],
                            point_size=2,
                            mark_minimum_point=True, 
                            title="", 
                            ):

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((8,8)), dpi=150)

    ax.scatter(x_series, y_series, 
            c=c_series, 
            cmap=cmap, 
            s=point_size)
    
    ax.set_xlim(-8,3)
    ax.set_ylim(-0.1,1)

    # add the minimum point: get the index of the minimum value
    if mark_minimum_point:
        min_index = np.argmin(c_series)
        ax.scatter(x_series[min_index], 
                    y_series[min_index],
                    c=dot_color, 
                    s=20,
                    marker="o",
                edgecolors='black',
                linewidths=1.5)

    ax.set_yticks([np.ceil(y_series.min()*10)/10, np.floor(y_series.max()*10)/10])
    ax.set_xticks([np.ceil(x_series.min()*10)/10, np.floor(x_series.max()*10)/10])

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\gamma$")

    if title != "":
        ax.set_title(title)
    
    if save_path is not None:
        plt.savefig(save_path) 
        
    plt.show()
