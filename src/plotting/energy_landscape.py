import numpy as np
import scipy.io
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

# from utils.saving_and_finding_files import time_stamp_for_saving

def plot_energy_landscape(etas, gammas, energy_grid, df_best=None,
                          dot_color="blue", 
                          title="", # 10,000 simulations",
                          interpolation="nearest", # "bilinear"
                          vmin=None, vmax=None, savepath=None, show=True):
    """
    Plot a Figure-4-style heatmap with optional white dots for best-fit models.
    
    Args:
        - etas (np.ndarray): 1D array of eta values (x-axis).
        - gammas (np.ndarray): 1D array of gamma values (y-axis).
        - energy_grid (np.ndarray): 2D array of energy values, shape (len(gammas), len(etas)).
        - df_best (pd.DataFrame): DataFrame with columns 'eta' and 'gamma' for best-fit points (optional).
        - dot_color (str): Color for the best-fit points.
        - title (str): Title for the plot.
        - interpolation (str): Interpolation method for the heatmap.
        - vmin (float): Minimum value for the color scale.
        - vmax (float): Maximum value for the color scale.
        - savepath (str): Path to save the figure.
        - show (bool): Whether to display the plot.
    Returns:
        - fig, ax: Matplotlib figure and axis objects.
    """
    
    fig, ax = plt.subplots(figsize=(6.2, 5.2)) # , dpi=150)

    # Heatmap: black=low (good), yellow=high (bad) like Mousley; 'hot' does that.
    im = ax.imshow(energy_grid,
                   origin="lower",
                   extent=[etas.min(), etas.max(), gammas.min(), gammas.max()],
                   aspect="auto",
                   cmap="hot",
                   vmin=vmin, vmax=vmax,
                   interpolation=interpolation)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Energy", rotation=270, labelpad=12)

    # Axis labels/range
    ax.set_xlabel("η")
    ax.set_ylabel("γ")
    ax.set_title(title)

    # Overlay best-fit points (one per subject)
    if df_best is not None and {"eta","gamma"}.issubset(df_best.columns):
        ax.scatter(df_best["eta"].values, df_best["gamma"].values,
                #    s=8, c="white", edgecolors="none", alpha=0.95, zorder=3)
                   s=16, c=dot_color, edgecolors="none", alpha=0.95, zorder=3)
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_energy_landscape_from_df(df, dot_color="blue", title="", interpolation="nearest", savepath=None, show=True):
    """
    Convenience function to plot energy landscape from a DataFrame with columns 'eta', 'gamma', and 'energy'.
    
    Args: 
        - df (pd.DataFrame): DataFrame containing 'eta', 'gamma', and 'energy' columns.
        - dot_color (str): Color for the best-fit points.
        - title (str): Title for the plot.
        - interpolation (str): Interpolation method for the heatmap.
        - savepath (str): Path to save the figure.
        - show (bool): Whether to display the plot.
        
    Returns:
        - fig, ax: Matplotlib figure and axis objects.  
    """
    
    # Ensure df has the required columns
    if not all(col in df.columns for col in ["eta", "gamma", "energy"]):
        raise ValueError("DataFrame must contain 'eta', 'gamma', and 'energy' columns.")

    # Turn it into a 2D array for plotting
    etas = np.sort(df["eta"].unique()) # x-axis values (η)
    gammas = np.sort(df["gamma"].unique()) # y-axis values (γ)
    # Collapse duplicates:  average energy for each (γ, η) cell  ↓
    grid_df = df.groupby(["gamma", "eta"], as_index=False)["energy"].mean()

    # 2D energy matrix:
    #     rows  → γ (ascending)
    #     cols  → η (ascending)

    energy_grid = (
        grid_df.pivot(index="gamma", columns="eta", values="energy")   # γ as rows, η as columns
        .reindex(index=gammas, columns=etas)                    # enforce full, sorted grid
        .to_numpy()
    )

    # (alternative way to get the best GNM per subject)
    df_best = (
        df.loc[df.groupby("subject")["energy"].idxmin()]
        .reset_index(drop=True)
    )

    # Plot the energy landscape
    # Note: df_best is optional, so we can pass None if not needed.
    fig, ax = plot_energy_landscape(
                etas, gammas, energy_grid,
                df_best=df_best,                 # comment out if not needed
                dot_color=dot_color,               # or any matplotlib-valid colour
                title=title,
                vmin=energy_grid.min(),         
                vmax=energy_grid.max(), 
                interpolation=interpolation,  # linear
                savepath=savepath,
                show=show 
            )
    
    return fig, ax
    