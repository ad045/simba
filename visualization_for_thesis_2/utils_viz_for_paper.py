import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# External configs (assumes these are in your config.py and vizman)
from vizman import viz
from config import COLOR_SCHEME, LABEL_MAP # PROPERTY_NAMES, CATEGORY_COLOURS, CATEGORY_ORDER, remaining_categories, SELECTED_PROPERTIES_NAMES 
from analysis_08_cluster_contributions import CLUSTERS 
from analysis_08_cluster_contributions import CLUSTER_COLORS as CATEGORY_COLOURS
from config import MERGED_PROPERTIES_NAMES as SELECTED_PROPERTIES_NAMES
from utils import get_combined_colors
from config_pca_parameter_selection import remaining_categories



def plot_scree(pca, output_path):
    """Bar + cumulative line scree plot."""
    evr = pca.explained_variance_ratio_ * 100
    n   = len(evr)

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,4)), dpi=150)

    # Horizontal line at 90%
    ax.axhline(90, 0.05, 10, color="grey", linestyle="--", lw=0.5) # lw=0.5, alpha=0.4)
    # Ticks on y for 0, 45, 90 
    ax.set_yticks([0, 45, 90])


    ax.bar(range(1, n + 1), evr, color="#394D73", label="Individual")
    ax.plot(range(1, n + 1), np.cumsum(evr), color="#E84653",
            marker="o", markersize=4, linewidth=1.2, label="Cumulative")
    # ax.set_xlabel("Principal Component")
    ax.set_xlabel("PC")
    # ax.set_ylabel("Explained Variance (%)")
    ax.set_ylabel("Explained Var.")
    ax.set_xticks(range(1, n + 1))
    # ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)



    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # plt.close()
    plt.show()
    print(f"{output_path}")






#### QUIVERS 


def compute_biplot_loadings(pca, used_features):
    """
    Returns a DataFrame of biplot-scaled loadings (n_features × 2).

    Each loading is scaled by sqrt(eigenvalue) so that arrow length in a biplot
    reflects the variance explained for that variable on that PC.
    Columns: PC1, PC2.  Index: used_features.
    """
    biplot = pca.components_.T * np.sqrt(pca.explained_variance_)   # (n_features, n_components)
    return pd.DataFrame(
        {"PC1": biplot[:, 0], "PC2": biplot[:, 1]},
        index=used_features,
    )



def plot_category_quivers(loading_and_cat_df, meta, output_path, with_labels=False, equal_axes=True):

    """
    Quiver plot of category-aggregate loading vectors in PC1–PC2 space.

    Each category arrow = sum of biplot-scaled loadings for all variables in that category.
    """
    cat_vectors = {}
    for cat in CATEGORY_COLOURS:
        cat_vars = [
            c for c in loading_and_cat_df.index
            if c in meta.index and meta.loc[c, "Category"] == cat
        ]
        if not cat_vars:
            continue
        cat_loads = loading_and_cat_df.loc[cat_vars]
        vx = cat_loads["PC1"].sum()
        vy = cat_loads["PC2"].sum()
        cat_vectors[cat] = (vx, vy)
        
    print(cat_vectors)

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,6)), 
                           dpi=150)

    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
    ax.scatter([0], [0], color="black", s=20, zorder=6)

    for cat, (vx, vy) in cat_vectors.items():
        colour = CATEGORY_COLOURS[cat]
        ax.quiver(
            0, 0, vx, vy,
            angles="xy", scale_units="xy", scale=1,
            color=colour, 
            width=0.018, # 012,
            headwidth=4,
            headlength=5, headaxislength=5,
            zorder=5,
        )
        if with_labels:
            nudge = 1.3 # 1.15
            ax.text(vx * nudge, vy * nudge, cat,
                    ha="center", va="center", color=colour)

    # max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) # * 1.4
    # maximum value in any direction (PC1 or PC2)
    max_val = max([max(abs(vx), abs(vy)) for vx, vy in cat_vectors.values()]) # * 1.4
    # # ax.set_xlim(-max_val, max_val)
    # # ax.set_ylim(-max_val, max_val)
    # ax.set_xlim(-max_val, max_val)
    # ax.set_ylim(-max_val, max_val)
    # if equal_axes:
    #     ax.set_aspect("equal")
    # else:
    #     plt.tight_layout()  # only apply when axes are free to scale

    # if equal_axes:
    #     ax.set_xlim(-max_val, max_val)
    #     ax.set_ylim(-max_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    # else:
    #     ax.set_xlim(-max_val, max_val)
    #     ax.set_ylim(-max_val, max_val)

    max_val = 5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ticks = [-4, -2, 0, 2, 4]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # ax.set_aspect("equal")
    # ax.set_xlabel("PC1", fontsize=9)
    # ax.set_ylabel("PC2", fontsize=9)
    # ax.set_title("Category vectors in PCA space\n(weighted by loading magnitude)", fontsize=9)
    print("Category vectors in PCA space\n(weighted by loading magnitude)")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # plt.tight_layout()
    # output_path = OUTPUT_FOLDER / output_filename
    plt.savefig(output_path, bbox_inches="tight")
    # plt.close()
    plt.show()
    print(f"Saved category quivers to {output_path}")




def plot_category_quivers_l2(loading_and_cat_df, meta, output_path, with_labels=False, equal_axes=True):

    """
    Quiver plot of category-aggregate loading vectors in PC1–PC2 space.

    Each category arrow = sum of biplot-scaled loadings for all variables in that category.
    """
    cat_vectors = {}
    for cat in CATEGORY_COLOURS:
        cat_vars = [
            c for c in loading_and_cat_df.index
            if c in meta.index and meta.loc[c, "Category"] == cat
        ]
        if not cat_vars:
            continue
        cat_loads = loading_and_cat_df.loc[cat_vars]
        # vx = cat_loads["PC1"].sum()
        # vy = cat_loads["PC2"].sum()
        # cat_vectors[cat] = (vx, vy)
        # THE L2 PART:  
        cat_loads = loading_and_cat_df.loc[cat_vars]
        loads_pc1 = cat_loads["PC1"].values                            
        loads_pc2 = cat_loads["PC2"].values
        magnitude = np.sqrt((loads_pc1 ** 2).sum() + (loads_pc2 ** 2).sum())                                                      
        mean_dir = np.array([loads_pc1.mean(), loads_pc2.mean()])
        norm = np.linalg.norm(mean_dir)                                
        unit = mean_dir / norm if norm > 1e-10 else np.array([1.0, 0.0])                                                          
        cat_vectors[cat] = (float(unit[0] * magnitude), float(unit[1] * magnitude))   
        
    print(cat_vectors)

    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,6)), 
                           dpi=150)

    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
    ax.scatter([0], [0], color="black", s=20, zorder=6)

    for cat, (vx, vy) in cat_vectors.items():
        colour = CATEGORY_COLOURS[cat]
        ax.quiver(
            0, 0, vx, vy,
            angles="xy", scale_units="xy", scale=1,
            color=colour, 
            width=0.018, # 012,
            headwidth=4,
            headlength=5, headaxislength=5,
            zorder=5,
        )
        if with_labels:
            nudge = 1.3 # 1.15
            ax.text(vx * nudge, vy * nudge, cat,
                    ha="center", va="center", color=colour)

    # max_val = max(np.sqrt(vx**2 + vy**2) for vx, vy in cat_vectors.values()) # * 1.4
    # maximum value in any direction (PC1 or PC2)
    max_val = max([max(abs(vx), abs(vy)) for vx, vy in cat_vectors.values()]) # * 1.4
    # # ax.set_xlim(-max_val, max_val)
    # # ax.set_ylim(-max_val, max_val)
    # ax.set_xlim(-max_val, max_val)
    # ax.set_ylim(-max_val, max_val)
    # if equal_axes:
    #     ax.set_aspect("equal")
    # else:
    #     plt.tight_layout()  # only apply when axes are free to scale

    # if equal_axes:
    #     ax.set_xlim(-max_val, max_val)
    #     ax.set_ylim(-max_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    # else:
    #     ax.set_xlim(-max_val, max_val)
    #     ax.set_ylim(-max_val, max_val)

    max_val = 5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ticks = [-4, -2, 0, 2, 4]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # ax.set_aspect("equal")
    # ax.set_xlabel("PC1", fontsize=9)
    # ax.set_ylabel("PC2", fontsize=9)
    # ax.set_title("Category vectors in PCA space\n(weighted by loading magnitude)", fontsize=9)
    print("Category vectors in PCA space\n(weighted by loading magnitude)")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    # plt.close()
    plt.show()
    print(f"Saved category quivers to {output_path}")
