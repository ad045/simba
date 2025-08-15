"""
General plotting utilities for the GNM and ESN pipelines.

This module collects common plotting functions in one place.  Functions
leverage the style parameters defined in :mod:`initialize_plots`.  See the
docstrings below for usage examples.

Note: these plotting functions do not call :func:`initialize_plots.apply_plot_style`
automatically.  Users should call that function once at the beginning of
their scripts or notebooks to apply the global matplotlib configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .initialize_plots import (
    PLOT_COLORMAP_DISTANCE,
    PLOT_COLORMAP_ENERGY,
    PLOT_FIGSIZE,
    PLOT_SAVE_FORMAT,
)

def plot_energy_landscape(
    etas: np.ndarray,
    gammas: np.ndarray,
    energy_grid: np.ndarray,
    df_best: pd.DataFrame | None = None,
    *,
    dot_color: str = "white",
    title: str = "",
    interpolation: str = "nearest",
    vmin: float | None = None,
    vmax: float | None = None,
    savepath: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a two–dimensional energy landscape.

    Parameters
    ----------
    etas : np.ndarray
        Sorted 1‑D array of η values for the horizontal axis.
    gammas : np.ndarray
        Sorted 1‑D array of γ values for the vertical axis.
    energy_grid : np.ndarray
        2‑D array of energy values with shape ``(len(gammas), len(etas))``.
    df_best : pandas.DataFrame, optional
        DataFrame containing ``'eta'`` and ``'gamma'`` columns specifying points
        to overlay on the heatmap (e.g. the best parameter pairs per subject).
    dot_color : str, default ``"white"``
        Colour used for the overlay points.
    title : str, optional
        Title of the plot.
    interpolation : str, default ``"nearest"``
        Interpolation method passed to :func:`imshow`.
    vmin, vmax : float, optional
        Minimum and maximum values for the colour scale.  If ``None``, the
        minimum/maximum of ``energy_grid`` will be used.
    savepath : str, optional
        If given, save the figure to this file path.  The file format is
        inferred from the extension or defaults to the global setting in
        :mod:`initialize_plots` when absent.
    show : bool, default ``True``
        If ``True``, display the figure on screen.  Disable when running in
        non-interactive contexts.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axis objects.

    Notes
    -----
    This function uses the global energy colormap defined in
    :mod:`initialize_plots`.  To change the default colour scheme, modify
    ``PLOT_COLORMAP_ENERGY`` in that module.
    """
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    if vmin is None:
        vmin = float(np.nanmin(energy_grid))
    if vmax is None:
        vmax = float(np.nanmax(energy_grid))

    # Display the heatmap: origin='lower' so that small gamma appears at the bottom
    im = ax.imshow(
        energy_grid,
        origin="lower",
        extent=[etas.min(), etas.max(), gammas.min(), gammas.max()],
        aspect="auto",
        cmap=PLOT_COLORMAP_ENERGY,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Energy", rotation=270, labelpad=12)

    ax.set_xlabel("η")
    ax.set_ylabel("γ")
    ax.set_title(title)

    # Overlay points if provided
    if df_best is not None and {"eta", "gamma"}.issubset(df_best.columns):
        ax.scatter(
            df_best["eta"].to_numpy(),
            df_best["gamma"].to_numpy(),
            s=16,
            c=dot_color,
            edgecolors="none",
            alpha=0.95,
            zorder=3,
        )

    # Save figure if requested
    if savepath is not None:
        # Determine file format: use extension if provided, else fall back to global
        fmt = savepath.split(".")[-1] if "." in savepath else PLOT_SAVE_FORMAT
        fig.savefig(savepath, format=fmt, bbox_inches="tight")

    if show:
        plt.show()
    return fig, ax


def plot_energy_landscape_from_df(
    df: pd.DataFrame,
    *,
    dot_color: str = "white",
    title: str = "",
    interpolation: str = "nearest",
    savepath: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Generate an energy landscape plot directly from a results DataFrame.

    This is a convenience wrapper around :func:`plot_energy_landscape`.  The
    DataFrame must contain the columns ``'eta'``, ``'gamma'``, and ``'energy'``.
    It will be pivoted onto a regular grid, missing values will be filled with
    NaN, and the average energy across duplicates will be computed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ``'eta'``, ``'gamma'``, and ``'energy'``.
    dot_color, title, interpolation, savepath, show : see
        :func:`plot_energy_landscape`.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axis.
    """
    required_cols = {"eta", "gamma", "energy"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain columns {sorted(required_cols)}. Got {df.columns.tolist()}"
        )

    etas = np.sort(df["eta"].unique())
    gammas = np.sort(df["gamma"].unique())

    # Average duplicate measurements (if any) and pivot
    grid_df = df.groupby(["gamma", "eta"], as_index=False)["energy"].mean()
    energy_grid = (
        grid_df.pivot(index="gamma", columns="eta", values="energy")
        .reindex(index=gammas, columns=etas)
        .to_numpy()
    )

    # Derive best eta/gamma per subject if subject column exists
    df_best = None
    if "subject" in df.columns:
        df_best = df.loc[df.groupby("subject")["energy"].idxmin()].reset_index(drop=True)

    return plot_energy_landscape(
        etas,
        gammas,
        energy_grid,
        df_best=df_best,
        dot_color=dot_color,
        title=title,
        interpolation=interpolation,
        savepath=savepath,
        show=show,
    )


def plot_metric_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    *,
    s: float = 20.0,
    alpha: float = 0.8,
    title: str = "",
    savepath: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a scatter plot of a metric versus two parameters.

    This utility can be used to visualise the results of hyperparameter searches.
    For example, ``x='spectral_radius'``, ``y='input_scaling'``,
    ``color='mc_mean'`` will scatter the memory capacity values across the
    parameter grid.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns specified by ``x``, ``y``, and ``color``.
    x, y, color : str
        Names of the DataFrame columns to use for the x‑axis, y‑axis and
        colour encoding, respectively.
    s : float, default 20.0
        Marker size.
    alpha : float, default 0.8
        Transparency of the markers.
    title : str, optional
        Title of the plot.
    savepath : str, optional
        If provided, save the figure to this location.
    show : bool, default ``True``
        Display the plot on screen.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axis.
    """
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    sc = ax.scatter(
        df[x],
        df[y],
        c=df[color],
        cmap=PLOT_COLORMAP_ENERGY,
        s=s,
        alpha=alpha,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color, rotation=270, labelpad=12)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    if savepath is not None:
        fmt = savepath.split(".")[-1] if "." in savepath else PLOT_SAVE_FORMAT
        fig.savefig(savepath, format=fmt, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
