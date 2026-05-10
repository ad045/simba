import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from config import CORNERS_OF_2D_SPACE


# Turn eta and gamma into two different colorchannels 
# def get_colorchannels_from_eta_gamma(df, eta_col="eta", gamma_col="gamma"):
#     from matplotlib.colors import Normalize

#     # Normalize the eta and gamma values
#     eta_norm = Normalize()(df[eta_col])
#     gamma_norm = Normalize()(df[gamma_col])

#     # cmap_eta = plt.cm.Reds # 
#     cmap_eta = plt.cm.YlOrRd
#     cmap_gamma = plt.cm.Blues # Greys # Reds #] Blues
#     # Map normalized eta and gamma to colors
#     colors_eta = cmap_eta(eta_norm)
#     colors_gamma = cmap_gamma(gamma_norm)
#     # Combine the two color channels (for simplicity, we'll just average them here)
#     # combined_colors = (colors_eta[:, :3] * colors_gamma[:, :3]) 
#     # combined_colors = combined_colors / np.max(combined_colors, axis=0)
#     combined_colors = (colors_eta[:, :3] + colors_gamma[:, :3]) / 2
#     combined_colors.shape
#     return combined_colors


def hex_to_rgb(hex_str):
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=float)


def get_combined_colors(huge_df):
    
    # Hex codes
    HEX_BOTTOM_LEFT  = CORNERS_OF_2D_SPACE["HEX_BOTTOM_LEFT"]
    HEX_BOTTOM_RIGHT = CORNERS_OF_2D_SPACE["HEX_BOTTOM_RIGHT"]
    HEX_TOP_LEFT     = CORNERS_OF_2D_SPACE["HEX_TOP_LEFT"]
    HEX_TOP_RIGHT    = CORNERS_OF_2D_SPACE["HEX_TOP_RIGHT"]

    bottom_left  = hex_to_rgb(HEX_BOTTOM_LEFT)
    bottom_right = hex_to_rgb(HEX_BOTTOM_RIGHT)
    top_left     = hex_to_rgb(HEX_TOP_LEFT)
    top_right    = hex_to_rgb(HEX_TOP_RIGHT)


    # Normalise eta → tx  (x-axis, left=0 → right=1)
    # Normalise gamma → ty (y-axis, bottom=0 → top=1)
    eta_norm   = Normalize()(huge_df["eta"]).data    # shape (N,)
    gamma_norm = Normalize()(huge_df["gamma"]).data  # shape (N,)

    tx = eta_norm          # (N,)
    ty = gamma_norm        # (N,)

    # Bilinear interpolation — same formula as the image, applied per-point
    # colour(tx, ty) = (1-ty)*[(1-tx)*BL + tx*BR] + ty*[(1-tx)*TL + tx*TR]
    combined_colors = (
        (1 - ty)[:, None] * ((1 - tx)[:, None] * bottom_left  + tx[:, None] * bottom_right) +
            ty [:, None] * ((1 - tx)[:, None] * top_left     + tx[:, None] * top_right   )
    ) / 255.0   # keep in [0,1] for matplotlib

    # huge_df["combined_color"] = 
    return [tuple(c) for c in combined_colors]