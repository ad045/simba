"""
Implementation: Add equal axis scaling to plot_metric_landscape_voronoi method.

Replace the relevant section in your PipelineVisualizer class.
"""

def plot_metric_landscape_voronoi(self, df: pd.DataFrame, 
                                  dot_color: str = "steelblue", 
                                  title: str = "", 
                                  cmap: str = "hot", 
                                  savepath: Optional[Path] = None, 
                                  metric_name: Optional[str] = None,
                                  show: bool = True,
                                  show_dots: bool = False, 
                                  point_size: int = 8,
                                  vmin: Optional[float] = None, 
                                  vmax: Optional[float] = None,
                                  ax: Optional[plt.Axes] = None, 
                                  annotate_extremes: Optional[bool] = False, 
                                  show_colorbar: bool = True, 
                                  estimated_indiv_connectomes: Optional[pd.DataFrame] = None, 
                                  ) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a metric landscape using Voronoi diagram for randomly sampled points."""
    
    # Prepare data
    points, metric_values = self._prepare_landscape_data(df, metric_name)
    
    # Create or use provided axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
    else:
        fig = ax.get_figure()
    
    # Setup plot bounds and colormap
    xlim, ylim = self._setup_plot_bounds(points, padding=0.0)
    norm, colormap, vmin, vmax = self._setup_colormap(metric_values, vmin, vmax, cmap)
    
    # ============================================================================
    # NEW: Calculate uniform axis ranges for equal scaling
    # ============================================================================
    eta_min, eta_max = points[:, 0].min(), points[:, 0].max()
    gamma_min, gamma_max = points[:, 1].min(), points[:, 1].max()
    
    eta_range = eta_max - eta_min
    gamma_range = gamma_max - gamma_min
    
    # Use the larger range for both axes to ensure square aspect
    max_range = max(eta_range, gamma_range)
    
    # Calculate centers
    eta_center = (eta_max + eta_min) / 2
    gamma_center = (gamma_max + gamma_min) / 2
    
    # Set uniform limits with small padding (5% of max_range)
    padding_factor = 0.05
    padding = padding_factor * max_range
    
    xlim_uniform = [eta_center - max_range/2 - padding, eta_center + max_range/2 + padding]
    ylim_uniform = [gamma_center - max_range/2 - padding, gamma_center + max_range/2 + padding]
    
    # Update xlim and ylim to use uniform scaling
    xlim = xlim_uniform
    ylim = ylim_uniform
    # ============================================================================
    
    # Create boundary points and Voronoi diagram
    boundary_points = self._create_boundary_points(xlim, ylim, n_boundary=20)
    all_points = np.vstack([points, boundary_points])
    vor = Voronoi(all_points)
    
    # Color each Voronoi region (only for actual data points)
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 not in region and len(region) > 0:
            polygon_vertices = [vor.vertices[v] for v in region]
            polygon = patches.Polygon(polygon_vertices, 
                                     facecolor=colormap(norm(metric_values[i])),
                                     edgecolor='none')
            ax.add_patch(polygon)
    
    # Add colorbar, points, and annotations
    if show_colorbar:
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        self._format_colorbar(metric_name, cbar)
        
    self._add_sample_points(ax, df, points, metric_name, show_dots, dot_color, point_size)
    self._add_estimated_connectomes(ax, estimated_indiv_connectomes, dot_color)
    
    if annotate_extremes:
        self._annotate_extreme_points(ax, df, metric_name)
    
    # ============================================================================
    # NEW: Set equal aspect ratio and apply uniform limits
    # ============================================================================
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')  # Force equal scaling
    # ============================================================================
    
    ax.set_xlabel("η")
    ax.set_ylabel("γ")
    ax.set_title(title)
    
    # Save/show
    self._handle_plot_output(fig, ax, savepath, show)
    
    return fig, ax


# ============================================================================
# Also update plot_mc_landscapes_grid for consistent multi-plot scaling
# ============================================================================

def plot_mc_landscapes_grid(self, df: pd.DataFrame, 
                           lags: List[int],
                           method: str = 'voronoi',
                           interpolation_method: str = 'linear',
                           n_cols: int = 3,
                           savepath: Optional[Path] = None,
                           save_format: str = "png",
                           cmap: str = "hot") -> plt.Figure:
    """
    Creates a grid of landscapes for different memory capacity (MC) lags.
    Unified method that can use either Voronoi or interpolation.
    """
    n_lags = len(lags)
    n_rows = (n_lags + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * 5, n_rows * (4.5 if method == 'voronoi' else 4)), 
                            sharex=True, sharey=True, 
                            layout="tight" if method == 'voronoi' else None)
    axes = axes.flatten()

    # Find global min/max for consistent color scaling
    mc_cols = [f'mc_{lag}' for lag in lags if f'mc_{lag}' in df.columns]
    if not mc_cols:
        raise ValueError("None of the specified MC lag columns were found in the DataFrame.")
        
    vmin = df[mc_cols].min().min()
    vmax = df[mc_cols].max().max()
    
    # ============================================================================
    # NEW: Calculate global uniform axis ranges for all subplots
    # ============================================================================
    eta_min = df['eta'].min()
    eta_max = df['eta'].max()
    gamma_min = df['gamma'].min()
    gamma_max = df['gamma'].max()
    
    eta_range = eta_max - eta_min
    gamma_range = gamma_max - gamma_min
    max_range = max(eta_range, gamma_range)
    
    eta_center = (eta_max + eta_min) / 2
    gamma_center = (gamma_max + gamma_min) / 2
    
    padding_factor = 0.05
    padding = padding_factor * max_range
    
    xlim_uniform = [eta_center - max_range/2 - padding, eta_center + max_range/2 + padding]
    ylim_uniform = [gamma_center - max_range/2 - padding, gamma_center + max_range/2 + padding]
    # ============================================================================

    # Plot each lag
    for i, lag in enumerate(lags):
        ax = axes[i]
        metric_name = f'mc_{lag}'
        
        if metric_name not in df.columns:
            self._handle_missing_metric(ax, lag)
            continue

        if method == 'voronoi':
            self.plot_metric_landscape_voronoi(
                df, title=f"MC Lag {lag}", metric_name=metric_name,
                show=False, show_dots=False, vmin=vmin, vmax=vmax,
                ax=ax, annotate_extremes=True, show_colorbar=False,
                cmap=cmap
            )
        else:  # interpolated
            self.plot_metric_landscape_interpolated(
                df, method=interpolation_method, title=f"MC Lag {lag}",
                metric_name=metric_name, show=False, show_dots=False,
                vmin=vmin, vmax=vmax, ax=ax, cmap=cmap
            )
        
        # ========================================================================
        # NEW: Apply uniform limits and equal aspect to each subplot
        # ========================================================================
        ax.set_xlim(xlim_uniform)
        ax.set_ylim(ylim_uniform)
        ax.set_aspect('equal', adjustable='box')
        # ========================================================================
        
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add shared colorbar and title
    if method == 'voronoi':
        self._add_shared_colorbar(fig, axes, vmin, vmax, cmap)
    
    method_display = f"Voronoi" if method == 'voronoi' else f"Interpolation: {interpolation_method}"
    fig.suptitle(f"Memory Capacity Landscapes ({method_display})", 
                fontsize=16 + (4 if method == 'voronoi' else 0), 
                y=1.02)
    
    # Save and show
    if savepath:
        suffix = f"_{method}" if method == 'voronoi' else f"_{interpolation_method}"
        full_save_path = savepath.parent / f"{savepath.stem}_mc_lags_grid{suffix}.{save_format}"
        fig.savefig(full_save_path, bbox_inches="tight", dpi=150)
        print(f"Saved MC lag grid plot to: {full_save_path}")
        
    return fig