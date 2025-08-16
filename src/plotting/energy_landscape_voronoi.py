import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches

def plot_energy_landscape_voronoi(df, dot_color="white", title="", 
                                  cmap="hot", savepath=None, show=True,
                                  show_points=True, point_size=8,
                                  vmin=None, vmax=None):
    """
    Plot energy landscape using Voronoi diagram for randomly sampled points.
    
    Args:
        - df (pd.DataFrame): DataFrame containing 'eta', 'gamma', and 'energy' columns.
        - dot_color (str): Color for the sample points.
        - title (str): Title for the plot.
        - cmap (str): Colormap for the energy values.
        - savepath (str): Path to save the figure.
        - show (bool): Whether to display the plot.
        - show_points (bool): Whether to show the actual sample points.
        - point_size (int): Size of the sample points if shown.
        - vmin (float): Minimum value for color scale.
        - vmax (float): Maximum value for color scale.
        
    Returns:
        - fig, ax: Matplotlib figure and axis objects.
    """
    
    # Ensure df has the required columns
    if not all(col in df.columns for col in ["eta", "gamma", "energy"]):
        raise ValueError("DataFrame must contain 'eta', 'gamma', and 'energy' columns.")
    
    # For each unique (eta, gamma) pair, average the energy
    grid_df = df.groupby(["eta", "gamma"], as_index=False)["energy"].mean()
    
    # Extract points and values
    points = grid_df[["eta", "gamma"]].values
    energies = grid_df["energy"].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    
    # Determine plot bounds with some padding
    eta_min, eta_max = points[:, 0].min(), points[:, 0].max()
    gamma_min, gamma_max = points[:, 1].min(), points[:, 1].max()
    eta_range = eta_max - eta_min
    gamma_range = gamma_max - gamma_min
    padding = 0.1
    
    xlim = [eta_min - padding * eta_range, eta_max + padding * eta_range]
    ylim = [gamma_min - padding * gamma_range, gamma_max + padding * gamma_range]
    
    # Add boundary points to ensure complete Voronoi cells
    boundary_points = []
    n_boundary = 20
    
    # Add points along the boundaries
    for i in range(n_boundary):
        # Bottom and top
        boundary_points.append([xlim[0] + i * (xlim[1] - xlim[0]) / (n_boundary - 1), ylim[0] - 1])
        boundary_points.append([xlim[0] + i * (xlim[1] - xlim[0]) / (n_boundary - 1), ylim[1] + 1])
        # Left and right
        boundary_points.append([xlim[0] - 1, ylim[0] + i * (ylim[1] - ylim[0]) / (n_boundary - 1)])
        boundary_points.append([xlim[1] + 1, ylim[0] + i * (ylim[1] - ylim[0]) / (n_boundary - 1)])
    
    # Combine actual points with boundary points
    all_points = np.vstack([points, boundary_points])
    
    # Create Voronoi diagram
    vor = Voronoi(all_points)
    
    # Set up color normalization
    if vmin is None:
        vmin = energies.min()
    if vmax is None:
        vmax = energies.max()
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)
    
    # Color each Voronoi region
    for i, point_idx in enumerate(range(len(points))):  # Only color regions for actual data points
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]
        
        if -1 not in region and len(region) > 0:
            polygon_vertices = [vor.vertices[v] for v in region]
            polygon = patches.Polygon(polygon_vertices, 
                                     facecolor=colormap(norm(energies[i])),
                                     edgecolor='none')
            ax.add_patch(polygon)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Energy", rotation=270, labelpad=12)
    
    # Optionally show the actual sample points
    if show_points:
        ax.scatter(points[:, 0], points[:, 1], 
                  s=point_size, c=dot_color, 
                  edgecolors='black', linewidths=0.5,
                  alpha=0.8, zorder=5)
    
    # If best points per subject are needed
    if "subject" in df.columns:
        df_best = df.loc[df.groupby("subject")["energy"].idxmin()].reset_index(drop=True)
        ax.scatter(df_best["eta"].values, df_best["gamma"].values,
                  s=16, c="cyan", edgecolors="black", linewidths=0.5,
                  alpha=0.95, zorder=10, marker='*')
    
    # Set limits and labels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("η")
    ax.set_ylabel("γ")
    ax.set_title(title)
    ax.set_aspect('auto')
    
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    
    if show:
        plt.show()
    
    return fig, ax


def plot_energy_landscape_interpolated(df, method='cubic', resolution=100,
                                       dot_color="white", title="", 
                                       cmap="hot", savepath=None, show=True,
                                       vmin=None, vmax=None):
    """
    Plot energy landscape using interpolation for randomly sampled points.
    
    Args:
        - df (pd.DataFrame): DataFrame containing 'eta', 'gamma', and 'energy' columns.
        - method (str): Interpolation method ('linear', 'cubic', 'nearest').
        - resolution (int): Grid resolution for interpolation.
        - dot_color (str): Color for the best-fit points.
        - title (str): Title for the plot.
        - cmap (str): Colormap for the energy values.
        - savepath (str): Path to save the figure.
        - show (bool): Whether to display the plot.
        - vmin (float): Minimum value for color scale.
        - vmax (float): Maximum value for color scale.
        
    Returns:
        - fig, ax: Matplotlib figure and axis objects.
    """
    from scipy.interpolate import griddata
    
    # Ensure df has the required columns
    if not all(col in df.columns for col in ["eta", "gamma", "energy"]):
        raise ValueError("DataFrame must contain 'eta', 'gamma', and 'energy' columns.")
    
    # For each unique (eta, gamma) pair, average the energy
    grid_df = df.groupby(["eta", "gamma"], as_index=False)["energy"].mean()
    
    # Create regular grid for interpolation
    eta_min, eta_max = grid_df["eta"].min(), grid_df["eta"].max()
    gamma_min, gamma_max = grid_df["gamma"].min(), grid_df["gamma"].max()
    
    eta_grid = np.linspace(eta_min, eta_max, resolution)
    gamma_grid = np.linspace(gamma_min, gamma_max, resolution)
    eta_mesh, gamma_mesh = np.meshgrid(eta_grid, gamma_grid)
    
    # Interpolate
    points = grid_df[["eta", "gamma"]].values
    values = grid_df["energy"].values
    
    energy_interp = griddata(points, values, (eta_mesh, gamma_mesh), method=method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    
    # Plot interpolated surface
    im = ax.imshow(energy_interp,
                   origin="lower",
                   extent=[eta_min, eta_max, gamma_min, gamma_max],
                   aspect="auto",
                   cmap=cmap,
                   vmin=vmin if vmin is not None else values.min(),
                   vmax=vmax if vmax is not None else values.max())
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Energy", rotation=270, labelpad=12)
    
    # Show sample points
    ax.scatter(points[:, 0], points[:, 1], 
              s=8, c=dot_color, edgecolors='black', 
              linewidths=0.5, alpha=0.7, zorder=5)
    
    # If best points per subject are needed
    if "subject" in df.columns:
        df_best = df.loc[df.groupby("subject")["energy"].idxmin()].reset_index(drop=True)
        ax.scatter(df_best["eta"].values, df_best["gamma"].values,
                  s=16, c="cyan", edgecolors="black", linewidths=0.5,
                  alpha=0.95, zorder=10, marker='*')
    
    ax.set_xlabel("η")
    ax.set_ylabel("γ")
    ax.set_title(title)
    
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    
    if show:
        plt.show()
    
    return fig, ax


# Keep your original functions as well
def plot_energy_landscape(etas, gammas, energy_grid, df_best=None,
                          dot_color="blue", 
                          title="",
                          interpolation="nearest",
                          vmin=None, vmax=None, savepath=None, show=True):
    """
    Original function - kept for backward compatibility
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    im = ax.imshow(energy_grid,
                   origin="lower",
                   extent=[etas.min(), etas.max(), gammas.min(), gammas.max()],
                   aspect="auto",
                   cmap="hot",
                   vmin=vmin, vmax=vmax,
                   interpolation=interpolation)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Energy", rotation=270, labelpad=12)

    ax.set_xlabel("η")
    ax.set_ylabel("γ")
    ax.set_title(title)

    if df_best is not None and {"eta","gamma"}.issubset(df_best.columns):
        ax.scatter(df_best["eta"].values, df_best["gamma"].values,
                   s=16, c=dot_color, edgecolors="none", alpha=0.95, zorder=3)
    
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_energy_landscape_from_df(df, dot_color="blue", title="", interpolation="nearest", 
                                  savepath=None, show=True):
    """
    Original function - kept for backward compatibility
    """
    if not all(col in df.columns for col in ["eta", "gamma", "energy"]):
        raise ValueError("DataFrame must contain 'eta', 'gamma', and 'energy' columns.")

    etas = np.sort(df["eta"].unique())
    gammas = np.sort(df["gamma"].unique())
    grid_df = df.groupby(["gamma", "eta"], as_index=False)["energy"].mean()

    energy_grid = (
        grid_df.pivot(index="gamma", columns="eta", values="energy")
        .reindex(index=gammas, columns=etas)
        .to_numpy()
    )

    df_best = (
        df.loc[df.groupby("subject")["energy"].idxmin()]
        .reset_index(drop=True)
    )

    fig, ax = plot_energy_landscape(
                etas, gammas, energy_grid,
                df_best=df_best,
                dot_color=dot_color,
                title=title,
                vmin=energy_grid.min(),         
                vmax=energy_grid.max(), 
                interpolation=interpolation,
                savepath=savepath,
                show=show 
            )
    
    return fig, ax


# Example usage function
def compare_visualizations(df, title_prefix="Energy Landscape"):
    """
    Create a comparison of different visualization methods.
    
    Args:
        - df: DataFrame with 'eta', 'gamma', 'energy' columns
        - title_prefix: Prefix for titles
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get data range for consistent coloring
    grid_df = df.groupby(["eta", "gamma"], as_index=False)["energy"].mean()
    vmin, vmax = grid_df["energy"].min(), grid_df["energy"].max()
    
    # 1. Voronoi diagram
    plt.sca(axes[0])
    plot_energy_landscape_voronoi(df, title=f"{title_prefix} - Voronoi", 
                                  show=False, vmin=vmin, vmax=vmax)
    
    # 2. Interpolated (cubic)
    plt.sca(axes[1])
    plot_energy_landscape_interpolated(df, method='cubic', 
                                       title=f"{title_prefix} - Cubic Interpolation",
                                       show=False, vmin=vmin, vmax=vmax)
    
    # 3. Interpolated (linear)
    plt.sca(axes[2])
    plot_energy_landscape_interpolated(df, method='linear',
                                       title=f"{title_prefix} - Linear Interpolation",
                                       show=False, vmin=vmin, vmax=vmax)
    
    plt.tight_layout()
    plt.show()
    
    return fig