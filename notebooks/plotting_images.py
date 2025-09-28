# # import pandas as pd
# # import io
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from pathlib import Path 

# # # --- 1. Load Data ---
# # data_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes/16_big_sweep_with_individual_connectomes_results.csv"
# # folder_path = Path(data_path).parent
# # # Since the provided data has only one row, we'll create a dummy DataFrame
# # # for a more illustrative plot. Remove or comment out this block when using your data.
# # num_points = 100
# # df = pd.read_csv(data_path)


# # # --- 2. Normalize Data for Color Channels ---
# # # This function scales a column to the 0-1 range for RGB colors.
# # def normalize(column):
# #     # Avoid division by zero if all values are the same
# #     if column.max() == column.min():
# #         return np.zeros_like(column)
# #     return (column - column.min()) / (column.max() - column.min())

# # # Assign normalized values to R, G, B channels
# # # Yellow is made by combining Red and Green
# # df['r'] = normalize(df['mc_mean'])
# # df['g'] = normalize(df['avg_communicability']) # Green for the "yellow" channel
# # df['b'] = normalize(df['wiring_cost'])


# # # --- 3. Create and Save the Plot ---
# # plt.figure(figsize=(10, 8))

# # # Create a scatter plot with 'eta' and 'gamma' as axes
# # # The 'c' argument takes the RGB values for each point
# # plt.scatter(df['eta'], df['gamma'], c=df[['r', 'g', 'b']].values)

# # plt.xlabel('eta')
# # plt.ylabel('gamma')
# # plt.title('Plot of eta vs. gamma with RGB Color Mapping')
# # plt.grid(True)

# # # Save the plot to a file
# # plt.savefig('eta_gamma_plot.png')
# # plt.close()

# # print("Plot saved as eta_gamma_plot.png")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi
# import matplotlib.patches as patches
# from typing import Literal
# from pathlib import Path

# def load_data(num_points: int = 100) -> pd.DataFrame:
#     """
#     Loads data for plotting.
    
#     For demonstration, this function generates a random DataFrame if no file is provided.
#     In a real scenario, you would replace the dummy data generation with:
#     # return pd.read_csv('your_file_name.csv')

#     Args:
#         num_points (int): The number of random data points to generate.

#     Returns:
#         pd.DataFrame: A DataFrame with eta, gamma, and color channel data.
#     """
#     print(f"Generating {num_points} random data points for demonstration.")
#     data = {
#         'eta': np.random.uniform(-3, 3, num_points),
#         'gamma': np.random.uniform(0, 1, num_points),
#         'mc_mean': np.random.uniform(0, 10, num_points),
#         'wiring_cost': np.random.uniform(10000, 20000, num_points),
#         'avg_communicability': np.random.uniform(0, 0.1, num_points)
#     }
#     return pd.DataFrame(data)

# def normalize_color_channels(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Normalizes the data columns to be used as RGB color channels.

#     Args:
#         df (pd.DataFrame): The input DataFrame.

#     Returns:
#         pd.DataFrame: The DataFrame with added 'r', 'g', 'b' columns (0-1 range).
#     """
#     def _normalize(column):
#         if column.max() == column.min():
#             return np.zeros_like(column)
#         return (column - column.min()) / (column.max() - column.min())

#     df_norm = df.copy()
#     df_norm['r'] = _normalize(df_norm['mc_mean'])
#     df_norm['g'] = _normalize(df_norm['avg_communicability']) # Green for the "yellow" channel
#     df_norm['b'] = _normalize(df_norm['wiring_cost'])
#     return df_norm

# def _plot_voronoi(ax: plt.Axes, 
#                   df: pd.DataFrame, 
#                   channel: Literal['rgb', 'red', 'yellow', 'blue']
#                   ):
#     """Helper function to draw a Voronoi plot on a given axis."""
#     points = df[['eta', 'gamma']].values
#     if points.shape[0] < 4:
#         # For Voronoi to work, you need at least 4 points.
#         ax.scatter(df['eta'], df['gamma'], c='gray', marker='x')
#         ax.text(0.5, 0.5, 'Not enough points for Voronoi', ha='center', va='center', transform=ax.transAxes)
#         return

#     vor = Voronoi(points)

#     # Color each Voronoi region
#     for i, region_idx in enumerate(vor.point_region):
#         region = vor.regions[region_idx]
#         if -1 not in region and region:
#             polygon = patches.Polygon([vor.vertices[v] for v in region], edgecolor='none')
            
#             # --- CORRECTED LOGIC ---
#             # Directly create the correct RGB tuple for each channel
#             if channel == 'red':
#                 color = (df['r'].iloc[i], 0, 0)
#             elif channel == 'yellow':
#                 # Use the 'g' value for both red and green channels to make yellow
#                 color = (df['g'].iloc[i], df['g'].iloc[i], 0)
#             elif channel == 'blue':
#                 color = (0, 0, df['b'].iloc[i])
#             else:  # 'rgb'
#                 color = (df['r'].iloc[i], df['g'].iloc[i], df['b'].iloc[i])
            
#             polygon.set_facecolor(color)
#             ax.add_patch(polygon)
    
#     ax.set_xlim(df['eta'].min(), df['eta'].max())
#     ax.set_ylim(df['gamma'].min(), df['gamma'].max())
    
# # def _plot_voronoi(ax: plt.Axes, df: pd.DataFrame, channel: Literal['rgb', 'red', 'yellow', 'blue']):
# #     """Helper function to draw a Voronoi plot on a given axis."""
# #     points = df[['eta', 'gamma']].values
# #     if points.shape[0] < 4:
# #         raise ValueError("Voronoi plotting requires at least 4 points.")

# #     vor = Voronoi(points)
    
# #     # Define color mappings
# #     color_map = {
# #         'rgb': df[['r', 'g', 'b']].values,
# #         'red': df[['r', 'r', 'r']].clip(0, 0.2).values * np.array([1, 0, 0]), # Show as pure red
# #         'yellow': df[['g', 'g', 'g']].clip(0, 0.2).values * np.array([1, 1, 0]), # Show as pure yellow
# #         'blue': df[['b', 'b', 'b']].clip(0, 0.2).values * np.array([0, 0, 1]) # Show as pure blue
# #     }
# #     colors = color_map[channel]

# #     # Color each Voronoi region
# #     for i, region_idx in enumerate(vor.point_region):
# #         region = vor.regions[region_idx]
# #         if -1 not in region and region:
# #             polygon = patches.Polygon([vor.vertices[v] for v in region], edgecolor='none')
# #             # For single channels, we create a colormap-like effect
# #             if channel != 'rgb':
# #                  polygon.set_facecolor(plt.cm.viridis(colors[i][0]))
# #             else:
# #                  polygon.set_facecolor(colors[i])
# #             ax.add_patch(polygon)
    
# #     ax.set_xlim(df['eta'].min(), df['eta'].max())
# #     ax.set_ylim(df['gamma'].min(), df['gamma'].max())

# def _plot_scatter(ax: plt.Axes, df: pd.DataFrame, channel: Literal['rgb', 'red', 'yellow', 'blue']):
#     """Helper function to draw a scatter plot on a given axis."""
#     color_map = {
#         'rgb': df[['r', 'g', 'b']].values,
#         'red': df['r'],
#         'yellow': df['g'],
#         'blue': df['b'],
#     }
#     cmap_map = {
#         'red': 'Reds',
#         'yellow': 'YlOrBr',
#         'blue': 'Blues'
#     }

#     colors = color_map[channel]
#     cmap = cmap_map.get(channel)
    
#     ax.scatter(df['eta'], df['gamma'], c=colors, cmap=cmap, s=50)


# def create_plots(df: pd.DataFrame, 
#                  layout: Literal['single', 'subplots'] = 'subplots', 
#                 #  color_map: Literal['red', 'yellow', 'blue', 'rgb'], 
#                  style: Literal['voronoi', 'scatter'] = 'voronoi', 
#                  save_folder: Path = None):
#     """
#     Generates and saves the requested plot(s).

#     Args:
#         df (pd.DataFrame): The DataFrame containing normalized data.
#         layout (str): 'single' for one RGB plot, 'subplots' for a 4-panel figure.
#         style (str): 'voronoi' or 'scatter' for the plot style.
#     """
#     plot_func = _plot_voronoi if style == 'voronoi' else _plot_scatter
    
#     if layout == 'single':
#         fig, ax = plt.subplots(figsize=(8, 6))
#         plot_func(ax, df, 'rgb')
#         ax.set_title('RGB Composite')
#         ax.set_xlabel('eta')
#         ax.set_ylabel('gamma')
    
#     elif layout == 'subplots':
#         fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
#         channels = ['blue', 'yellow', 'red', 'rgb']
#         titles = ['Wiring Cost (Blue)', 'Communicability (Yellow)', 'MC Mean (Red)', 'RGB Composite']
        
#         for ax, channel, title in zip(axes, channels, titles):
#             plot_func(ax, df, channel)
#             ax.set_title(title)
#             ax.set_xlabel('eta')
        
#         axes[0].set_ylabel('gamma')

#     else:
#         raise ValueError("Invalid layout. Choose 'single' or 'subplots'.")

#     plt.suptitle(f'Eta vs. Gamma - {style.capitalize()} Style', fontsize=16, y=1.02)
#     plt.tight_layout()
    
#     if save_folder: 
#         filename = save_folder / f'plot_{layout}_{style}.png'
#         plt.savefig(filename)
#         plt.close()
#         print(f"Plot saved as {filename}")
#     else: 
#         print("Provide save_folder to save images.")
#         plt.show()

# # --- Main Execution Block ---
# def main():
#     """Main function to run the plotting script."""
#     # 1. Load data
#     # raw_df = load_data(num_points=200)
    
#     data_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes/16_big_sweep_with_individual_connectomes_results.csv"
#     save_folder = Path(data_path).parent / "figures/rgb_pictures"
#     save_folder.mkdir(parents=True, exist_ok=True)
    
#     raw_df = pd.read_csv(data_path)[["eta", "gamma", "mc_mean", "wiring_cost", "avg_communicability"]]
    
#     # 'eta': np.random.uniform(-3, 3, num_points),
#     #     'gamma': np.random.uniform(0, 1, num_points),
#     #     'mc_mean': np.random.uniform(0, 10, num_points),
#     #     'wiring_cost': np.random.uniform(10000, 20000, num_points),
#     #     'avg_communicability': np.random.uniform(0, 0.1, num_points)
    
#     # 2. Normalize for color channels
#     df = normalize_color_channels(raw_df)
    
#     # 3. Create and save plots
#     # Example 1: 4-panel subplot with Voronoi style
#     create_plots(df, layout='subplots', style='voronoi', save_folder=save_folder)
    
#     # Example 2: Single RGB plot with scatter style
#     create_plots(df, layout='single', style='scatter', save_folder=save_folder)
    
#     # Example 3: Single RGB plot with Voronoi style
#     create_plots(df, layout='single', style='voronoi', save_folder=save_folder)

# if __name__ == "__main__":
#     main()
    
    
    


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.patches as patches
from typing import Literal

# def load_data(num_points: int = 100) -> pd.DataFrame:
#     """Loads or generates data for plotting."""
#     print(f"Generating {num_points} random data points for demonstration.")
#     data = {
#         'eta': np.random.uniform(-3, 3, num_points),
#         'gamma': np.random.uniform(0, 1, num_points),
#         'mc_mean': np.random.uniform(0, 10, num_points),
#         'wiring_cost': np.random.uniform(10000, 20000, num_points),
#         'avg_communicability': np.random.uniform(0, 0.1, num_points)
#     }
#     return pd.DataFrame(data)


def normalize_color_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes data columns into 'r', 'g', 'b' channels."""
    def _normalize(column):
        if column.max() == column.min(): return np.zeros_like(column)
        return (column - column.min()) / (column.max() - column.min())

    df_norm = df.copy()
    df_norm['r'] = _normalize(df_norm['mc_mean'])
    df_norm['g'] = _normalize(df_norm['avg_communicability'])
    df_norm['b'] = _normalize(df_norm['wiring_cost'])
    return df_norm

def _plot_voronoi(ax: plt.Axes, df: pd.DataFrame, channel: str):
    """Helper function to draw a Voronoi plot on a given axis."""
    points = df[['eta', 'gamma']].values
    if points.shape[0] < 4:
        ax.text(0.5, 0.5, 'Not enough points for Voronoi', ha='center', va='center', transform=ax.transAxes)
        return

    vor = Voronoi(points)
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and region:
            polygon = patches.Polygon([vor.vertices[v] for v in region], edgecolor='none')
            
            r, g, b = df['r'].iloc[i], df['g'].iloc[i], df['b'].iloc[i]
            color = (0,0,0) # Default to black
            
            if channel == 'red': color = (r, 0, 0)
            elif channel == 'yellow': color = (g, g, 0)
            elif channel == 'blue': color = (0, 0, b)
            elif channel == 'rgb': color = (r, g, b)
            elif channel == 'cyan': color = (0, 1 - r, 1 - r)
            elif channel == 'magenta': color = (1 - g, 0, 1 - g)
            elif channel == 'yellow_cmy': color = (1 - b, 1 - b, 0)
            elif channel == 'cmyk': # NEW: CMYK composite simulation
                c, m, y = 1 - r, 1 - g, 1 - b
                k = min(c, m, y) # Calculate black component
                # Convert CMYK to RGB for screen display
                final_r = (1 - c) * (1 - k)
                final_g = (1 - m) * (1 - k)
                final_b = (1 - y) * (1 - k)
                color = (final_r, final_g, final_b)

            polygon.set_facecolor(color)
            ax.add_patch(polygon)
    
    ax.set_xlim(df['eta'].min(), df['eta'].max())
    ax.set_ylim(df['gamma'].min(), df['gamma'].max())


def create_plots(df: pd.DataFrame, 
                 layout: Literal['single', 'subplots'] = 'subplots', 
                 style: Literal['voronoi', 'scatter'] = 'voronoi',
                 color_model: Literal['additive', 'subtractive'] = 'additive'):
    """Generates and saves the requested plot(s)."""
    plot_func = _plot_voronoi # Scatter plot logic would need similar updates
    
    if color_model == 'additive':
        channels = ['blue', 'yellow', 'red', 'rgb']
        titles = ['Wiring Cost (Blue)', 'Communicability (Yellow)', 'MC Mean (Red)', 'RGB Composite']
        suptitle = f'Eta vs. Gamma ({style.capitalize()} Style, Additive Model)'
    else: # Subtractive
        # MODIFIED: Use 'cmyk' for the final composite plot
        channels = ['yellow_cmy', 'magenta', 'cyan', 'cmyk']
        titles = ['Wiring Cost (Yellow)', 'Communicability (Magenta)', 'MC Mean (Cyan)', 'CMYK Composite']
        suptitle = f'Eta vs. Gamma ({style.capitalize()} Style, Subtractive Model)'
        
    if layout == 'single':
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use the correct composite channel based on the model
        composite_channel = 'rgb' if color_model == 'additive' else 'cmyk'
        plot_func(ax, df, composite_channel)
        ax.set_title(titles[-1])
        ax.set_xlabel('eta')
        ax.set_ylabel('gamma')
    
    elif layout == 'subplots':
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
        for ax, channel, title in zip(axes, channels, titles):
            plot_func(ax, df, channel)
            ax.set_title(title)
            ax.set_xlabel('eta')
        axes[0].set_ylabel('gamma')

    plt.suptitle(suptitle, fontsize=16, y=1.02)
    plt.tight_layout()
    
    filename = f'plot_{layout}_{style}_{color_model}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")
    
    
    
# def normalize_color_channels(df: pd.DataFrame) -> pd.DataFrame:
#     """Normalizes data columns into 'r', 'g', 'b' channels."""
#     def _normalize(column):
#         if column.max() == column.min(): return np.zeros_like(column)
#         return (column - column.min()) / (column.max() - column.min())

#     df_norm = df.copy()
#     df_norm['r'] = _normalize(df_norm['mc_mean'])
#     df_norm['g'] = _normalize(df_norm['avg_communicability'])
#     df_norm['b'] = _normalize(df_norm['wiring_cost'])
#     return df_norm

# def _plot_voronoi(ax: plt.Axes, df: pd.DataFrame, channel: str):
#     """Helper function to draw a Voronoi plot on a given axis."""
#     points = df[['eta', 'gamma']].values
#     if points.shape[0] < 4:
#         ax.text(0.5, 0.5, 'Not enough points for Voronoi', ha='center', va='center', transform=ax.transAxes)
#         return

#     vor = Voronoi(points)
#     for i, region_idx in enumerate(vor.point_region):
#         region = vor.regions[region_idx]
#         if -1 not in region and region:
#             polygon = patches.Polygon([vor.vertices[v] for v in region], edgecolor='none')
            
#             # MODIFIED: Logic to handle all color channels
#             r, g, b = df['r'].iloc[i], df['g'].iloc[i], df['b'].iloc[i]
#             base_val = 0 # was previously 0 (background is black)
#             color_map = {
#                 'red': (r, base_val, base_val),
#                 'yellow': (g, g, base_val),
#                 'blue': (base_val, base_val, b),
#                 'rgb': (r, g, b),
#                 'cyan': (base_val, 1 - r, 1 - r),    # Cyan is inverse of Red
#                 'magenta': (1 - g, base_val, 1 - g), # Magenta is inverse of Green
#                 'yellow_cmy': (1 - b, 1 - b, base_val) # Yellow is inverse of Blue
                
#             }
#             polygon.set_facecolor(color_map.get(channel, (r, g, b)))
#             ax.add_patch(polygon)
    
#     ax.set_xlim(df['eta'].min(), df['eta'].max())
#     ax.set_ylim(df['gamma'].min(), df['gamma'].max())

# def _plot_scatter(ax: plt.Axes, df: pd.DataFrame, channel: str):
#     """Helper function to draw a scatter plot on a given axis."""
#     # This function would need similar logic to _plot_voronoi for CMY colors
#     # For brevity, this example focuses on the Voronoi implementation
#     r, g, b = df['r'], df['g'], df['b']
#     color_map = {'red': (r, 0, 0), 'blue': (0, 0, b)} # Simplified for example
#     ax.scatter(df['eta'], df['gamma'], c=df[['r','g','b']].values, s=50)


# def create_plots(df: pd.DataFrame, 
#                  layout: Literal['single', 'subplots'] = 'subplots', 
#                  style: Literal['voronoi', 'scatter'] = 'voronoi',
#                  color_model: Literal['additive', 'subtractive'] = 'additive'): # NEW PARAMETER
#     """
#     Generates and saves the requested plot(s).
#     """
#     plot_func = _plot_voronoi if style == 'voronoi' else _plot_scatter
    
#     # MODIFIED: Titles and channels now depend on the color model
#     if color_model == 'additive':
#         channels = ['blue', 'yellow', 'red', 'rgb']
#         titles = ['Wiring Cost (Blue)', 'Communicability (Yellow)', 'MC Mean (Red)', 'RGB Composite']
#         suptitle = f'Eta vs. Gamma ({style.capitalize()} Style, Additive Model)'
#     else: # Subtractive
#         channels = ['yellow_cmy', 'magenta', 'cyan', 'rgb']
#         titles = ['Wiring Cost (Yellow)', 'Communicability (Magenta)', 'MC Mean (Cyan)', 'CMY Composite']
#         suptitle = f'Eta vs. Gamma ({style.capitalize()} Style, Subtractive Model)'
        
#     if layout == 'single':
#         fig, ax = plt.subplots(figsize=(8, 6))
#         plot_func(ax, df, 'rgb')
#         ax.set_title(titles[-1]) # Use the composite title
#         ax.set_xlabel('eta')
#         ax.set_ylabel('gamma')
    
#     elif layout == 'subplots':
#         fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
#         for ax, channel, title in zip(axes, channels, titles):
#             plot_func(ax, df, channel)
#             ax.set_title(title)
#             ax.set_xlabel('eta')
#         axes[0].set_ylabel('gamma')

#     plt.suptitle(suptitle, fontsize=16, y=1.02)
#     plt.tight_layout()
    
#     filename = f'plot_{layout}_{style}_{color_model}.png'
#     plt.savefig(filename)
#     plt.close()
#     print(f"Plot saved as {filename}")

# --- Main Execution Block ---
def main():
    """Main function to run the plotting script."""
    # raw_df = load_data(num_points=500)
    # df = normalize_color_channels(raw_df)
    
    from pathlib import Path
    data_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes/16_big_sweep_with_individual_connectomes_results.csv"
    save_folder = Path(data_path).parent / "figures/rgb_pictures"
    save_folder.mkdir(parents=True, exist_ok=True)
    
    raw_df = pd.read_csv(data_path)[["eta", "gamma", "mc_mean", "wiring_cost", "avg_communicability"]]
    df = normalize_color_channels(raw_df)
    
    # Example 1: Additive (RGB) model - this is the default
    create_plots(df, layout='subplots', style='voronoi', color_model='additive')
    
    # NEW Example 2: Subtractive (CMY) model
    create_plots(df, layout='subplots', style='voronoi', color_model='subtractive')

if __name__ == "__main__":
    main()