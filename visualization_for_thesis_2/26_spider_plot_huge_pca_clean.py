I have the categories that I manually defined, but I am worrying
that the categories are too random and are not following true
clusters or so. Can you please figure out a way how I can best
find good clusters that make intuitive sense, such that I can use
those? Or how I can justify using mine? Especially also the        
flipping part we have been talking about in a different chat?     

# """
# spider_plot_clean.py
# ────────────────────
# Standalone spider / radar plots where each axis = one property category.

# Category score per network
# ──────────────────────────
# Rather than summing or naively averaging z-scored features, each category
# score is computed as the projection onto the FIRST principal component of
# that category's features (across all networks).  This:
#   • is invariant to the number of features in the category
#   • finds the direction of maximum shared variance within the category
#   • gives a single number that represents "how much of this category" a
#     network has, regardless of whether there are 2 or 10 features

# Standardisation
# ───────────────
# Only ONE round of z-scoring:
#   1. Per-feature StandardScaler across all networks  →  `scaled_data`
#   2. PC1 projection within each category             →  `cat_scores`
#   No second z-score is applied; the PC1 score is already interpretable as
#   a number of standard deviations in the direction of maximum variance.

# Spider plot contents
# ────────────────────
# • One thick line  : the focal network
# • Thin faint lines: its k nearest neighbours in the full PCA space
# • Filled polygon  : the focal network (low alpha)
# • All rings always shown (no dynamic hiding)
# • Spoke labels on every axis, coloured by category
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.metrics import pairwise_distances

# from analysis_config import OUTPUT_DIR
# # ──────────────────────────────────────────────────────────────────────────────
# # 1.  BUILD CATEGORY SCORES
# # ──────────────────────────────────────────────────────────────────────────────

# def build_category_scores(
#     df_properties,          # pd.DataFrame: rows = networks, cols = features
#     feature_to_category,    # dict: feature_name -> category_name
#     category_order,         # list of category names in desired axis order
# ):
#     """
#     Returns
#     -------
#     cat_scores : np.ndarray  shape (n_networks, n_categories)
#         Each column is the PC1 score of that category's features.
#         Sign is chosen so that higher = "more" of the category
#         (PC1 sign is fixed by making the mean loading positive).
#     cat_labels : list[str]   category names, in category_order
#     scaler     : fitted StandardScaler  (keep for later use)
#     per_cat_pca: dict  category -> fitted PCA(n_components=1)
#     """

#     # ── 1a. z-score every feature across all networks (once) ─────────────────
#     scaler = StandardScaler()
#     scaled = scaler.fit_transform(df_properties.values)   # (n, p)

#     cols = list(df_properties.columns)

#     cat_scores   = []
#     cat_labels   = []
#     per_cat_pca  = {}

#     for cat in category_order:
#         # Features belonging to this category that are present in the data
#         feat_idx = [i for i, c in enumerate(cols)
#                     if feature_to_category.get(c) == cat]
#         if not feat_idx:
#             continue

#         X_cat = scaled[:, feat_idx]          # (n_networks, n_cat_features)

#         if X_cat.shape[1] == 1:
#             # Only one feature: just use it directly
#             scores = X_cat[:, 0]
#             per_cat_pca[cat] = None
#         else:
#             pca_cat = PCA(n_components=1)
#             scores  = pca_cat.fit_transform(X_cat)[:, 0]   # (n_networks,)

#             # Fix sign: majority of loadings should be positive
#             if pca_cat.components_[0].mean() < 0:
#                 scores = -scores
#                 pca_cat.components_ *= -1

#             per_cat_pca[cat] = pca_cat

#         cat_scores.append(scores)
#         cat_labels.append(cat)

#     cat_scores = np.stack(cat_scores, axis=1)   # (n_networks, n_categories)
#     return cat_scores, cat_labels, scaler, per_cat_pca


# # ──────────────────────────────────────────────────────────────────────────────
# # 2.  SPIDER PLOT
# # ──────────────────────────────────────────────────────────────────────────────

# def spider_plot(
#     focal_values,           # 1-D array, length = n_categories
#     cat_labels,             # list of category name strings
#     color,                  # color for the focal network
#     title       = "",
#     neighbour_values = None,  # list of 1-D arrays (one per neighbour)
#     ylim        = None,     # (ymin, ymax); auto if None
#     ring_vals   = None,     # e.g. [-2, 0, 2]; auto if None
#     cat_colors  = None,     # list of colors for spoke labels; gray if None
#     ax          = None,     # pass an existing polar Axes, or None to create
#     figsize     = (5, 5),
#     dpi         = 150,
# ):
#     """
#     Draw a single spider plot.

#     Returns the Figure and Axes so callers can save / further customise.
#     """
#     n       = len(focal_values)
#     angles  = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
#     angles_c = angles + angles[:1]
#     vals_c   = list(focal_values) + [focal_values[0]]

#     # ── Axis limits ───────────────────────────────────────────────────────────
#     if ylim is None:
#         all_vals = list(focal_values)
#         if neighbour_values:
#             for nv in neighbour_values:
#                 all_vals.extend(nv)
#         pad  = 0.5
#         ymin = np.floor(np.nanmin(all_vals)) - pad
#         ymax = np.ceil (np.nanmax(all_vals)) + pad
#         ylim = (ymin, ymax)

#     if ring_vals is None:
#         step      = 1 if (ylim[1] - ylim[0]) <= 6 else 2
#         ring_vals = list(range(int(np.ceil(ylim[0])),
#                                int(np.floor(ylim[1])) + 1, step))

#     # ── Figure / axes ─────────────────────────────────────────────────────────
#     own_fig = ax is None
#     if own_fig:
#         fig, ax = plt.subplots(figsize=figsize,
#                                subplot_kw=dict(polar=True), dpi=dpi)
#     else:
#         fig = ax.figure

#     ax.set_ylim(*ylim)
#     ax.spines["polar"].set_visible(False)
#     ax.yaxis.grid(False)
#     ax.xaxis.grid(False)
#     ax.set_thetagrids([])     # we'll draw spoke labels manually

#     # ── Gridlines ─────────────────────────────────────────────────────────────
#     theta_ring = np.linspace(0, 2 * np.pi, 300)
#     for rv in ring_vals:
#         lw  = 1.4 if rv == 0 else 0.5
#         ls  = "-" if rv == 0 else "--"
#         col = "#888888" if rv == 0 else "#cccccc"
#         ax.plot(theta_ring, np.full(300, rv),
#                 color=col, linewidth=lw, linestyle=ls, zorder=0)
#         # small tick label on the right side
#         ax.text(np.pi / 2, rv, f"{rv:g}",
#                 ha="center", va="bottom", fontsize=6, color="#aaaaaa")

#     # ── Spokes ────────────────────────────────────────────────────────────────
#     for angle in angles:
#         ax.plot([angle, angle], [ylim[0], ylim[1]],
#                 color="#dddddd", linewidth=0.5, linestyle="--", zorder=0)

#     # ── Neighbour lines ───────────────────────────────────────────────────────
#     if neighbour_values:
#         for nv in neighbour_values:
#             nv_c = list(nv) + [nv[0]]
#             ax.plot(angles_c, nv_c,
#                     color=color, linewidth=0.8, alpha=0.25, zorder=2)
#             ax.fill(angles_c, nv_c,
#                     color=color, alpha=0.03, zorder=2)

#     # ── Focal network ─────────────────────────────────────────────────────────
#     ax.fill(angles_c, vals_c, color=color, alpha=0.15, zorder=3)
#     ax.plot(angles_c, vals_c, color=color, linewidth=2.2, zorder=4)

#     # ── Spoke labels ──────────────────────────────────────────────────────────
#     label_r = ylim[1] + (ylim[1] - ylim[0]) * 0.08
#     for i, (angle, lbl) in enumerate(zip(angles, cat_labels)):
#         ha = "center"
#         if angle < np.pi * 0.1 or angle > np.pi * 1.9:
#             ha = "center"
#         elif angle < np.pi:
#             ha = "left"
#         else:
#             ha = "right"

#         va = "center"
#         if np.pi * 0.4 < angle < np.pi * 0.6:
#             va = "bottom"
#         elif np.pi * 1.4 < angle < np.pi * 1.6:
#             va = "top"

#         col = cat_colors[i] if cat_colors else "#444444"
#         # wrap long labels
#         wrapped = lbl.replace(" ", "\n") if len(lbl) > 14 else lbl
#         ax.text(angle, label_r, wrapped,
#                 ha=ha, va=va, fontsize=7, color=col,
#                 fontweight="bold", clip_on=False)

#     if title:
#         ax.set_title(title, pad=18, fontsize=9)

#     if own_fig:
#         fig.tight_layout()

#     return fig, ax


# # ──────────────────────────────────────────────────────────────────────────────
# # 3.  CONVENIENCE WRAPPER: build + save one plot per focal network
# # ──────────────────────────────────────────────────────────────────────────────

# def make_spider_for_network(
#     focal_idx,
#     cat_scores,
#     cat_labels,
#     color,
#     title,
#     pca_embed,          # (n_networks, n_pcs) used to find neighbours
#     n_neighbours  = 5,
#     exclude_mask  = None,   # boolean array; True = exclude from neighbour search
#     cat_colors    = None,
#     ylim          = None,
#     ring_vals     = None,
#     filepath      = None,
#     figsize       = (5, 5),
#     dpi           = 150,
# ):
#     """
#     Build a spider plot for one focal network, with its nearest neighbours
#     shown as faint background lines.

#     Parameters
#     ----------
#     focal_idx    : int, row index into cat_scores / pca_embed
#     exclude_mask : boolean array length n_networks; those rows are excluded
#                    from neighbour search (e.g. to exclude GNM networks)
#     filepath     : if given, save to this path
#     """
#     # ── Find neighbours ───────────────────────────────────────────────────────
#     dists = pairwise_distances(pca_embed[focal_idx:focal_idx+1], pca_embed)[0]
#     dists[focal_idx] = np.inf
#     if exclude_mask is not None:
#         dists[exclude_mask] = np.inf
#     nb_idx = np.argsort(dists)[:n_neighbours]
#     nb_vals = [cat_scores[i] for i in nb_idx]

#     # ── Plot ──────────────────────────────────────────────────────────────────
#     fig, ax = spider_plot(
#         focal_values     = cat_scores[focal_idx],
#         cat_labels       = cat_labels,
#         color            = color,
#         title            = title,
#         neighbour_values = nb_vals,
#         ylim             = ylim,
#         ring_vals        = ring_vals,
#         cat_colors       = cat_colors,
#         figsize          = figsize,
#         dpi              = dpi,
#     )

#     if filepath:
#         fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
#         print(f"  Saved: {filepath}")

#     return fig, ax


# # ──────────────────────────────────────────────────────────────────────────────
# # 4.  DEMO  (runs when you execute this file directly with synthetic data)
# # ──────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import pandas as pd

#     rng = np.random.default_rng(0)

#     # ── Synthetic data ────────────────────────────────────────────────────────
#     n_networks = 200
#     features = {
#         "Topology":      ["mod", "omega", "simplices"],
#         "Efficiency":    ["glob_eff", "diff_eff"],
#         "Wiring":        ["wiring_cost", "long_range"],
#         "Spectral":      ["spec_radius", "spec_gap", "sync_ratio", "alg_conn"],
#         "Dynamics":      ["T_crit", "size_crit", "div_crit"],
#         "Robustness":    ["rob_auc", "rob_ratio"],
#         "Control":       ["nct_std"],
#         "Memory":        ["mc_lin", "mc_nonlin", "ipc_deg1", "ipc_deg2"],
#     }

#     all_features = [f for flist in features.values() for f in flist]
#     df = pd.DataFrame(rng.standard_normal((n_networks, len(all_features))),
#                       columns=all_features)

#     feat_to_cat = {f: cat for cat, flist in features.items() for f in flist}
#     cat_order   = list(features.keys())

#     cat_colors  = [
#         "#E07B39", "#5B8DB8", "#6AAB6A", "#9B6BAD",
#         "#D4A017", "#C75D5D", "#4DADA6", "#888888",
#     ]

#     # ── Build scores ──────────────────────────────────────────────────────────
#     cat_scores, cat_labels, scaler, per_cat_pca = build_category_scores(
#         df, feat_to_cat, cat_order
#     )

#     print("Category scores shape:", cat_scores.shape)
#     print("Categories:", cat_labels)

#     # ── Full PCA for neighbour lookup ─────────────────────────────────────────
#     pca_full = PCA(n_components=5)
#     pca_embed = pca_full.fit_transform(scaler.transform(df.values))

#     # ── Shared y-limits across all plots ─────────────────────────────────────
#     ymin = np.floor(cat_scores.min()) - 0.3
#     ymax = np.ceil (cat_scores.max()) + 0.3
#     shared_ylim  = (ymin, ymax)
#     shared_rings = [r for r in range(int(np.ceil(ymin)),
#                                      int(np.floor(ymax)) + 1)]

#     # ── Plot a few example networks ───────────────────────────────────────────
#     example_indices = [0, 42, 99]
#     example_colors  = ["steelblue", "tomato", "seagreen"]
#     example_titles  = [f"Network {i}" for i in example_indices]

#     fig, axs = plt.subplots(
#         1, len(example_indices),
#         figsize=(5 * len(example_indices), 5),
#         subplot_kw=dict(polar=True),
#         dpi=150,
#     )

#     for ax, idx, col, ttl in zip(axs, example_indices,
#                                   example_colors, example_titles):
#         dists = pairwise_distances(pca_embed[idx:idx+1], pca_embed)[0]
#         dists[idx] = np.inf
#         nb_idx  = np.argsort(dists)[:5]
#         nb_vals = [cat_scores[i] for i in nb_idx]

#         spider_plot(
#             focal_values     = cat_scores[idx],
#             cat_labels       = cat_labels,
#             color            = col,
#             title            = ttl,
#             neighbour_values = nb_vals,
#             ylim             = shared_ylim,
#             ring_vals        = shared_rings,
#             cat_colors       = cat_colors,
#             ax               = ax,
#         )

#     plt.tight_layout()
#     plt.savefig(OUTPUT_DIR / "spider_demo.pdf", dpi=150, bbox_inches="tight")
#     print(f"Demo saved to {OUTPUT_DIR / 'spider_demo.pdf'}")
#     plt.show()


"""
spider_mami_subgroups.py
────────────────────────
Spider / radar plots for MaMI animal subgroups (orders with > 10 animals).

One axis per property category.  Category score = PC1 of that category's
z-scored features, fitted on the MaMI subset only.

Each spider shows:
  • Thin lines per individual animal (low alpha)
  • Shaded band  = mean ± 1 SD across animals in the group
  • Thick line   = group mean

PCA for neighbour structure (not used for the group plots themselves, but
kept so you can also call make_spider_for_network() if needed) is also
MaMI-only.

Drop-in usage
─────────────
Adjust the five lines under "── USER CONFIG ──" to point at your data,
then run:
    python spider_mami_subgroups.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


# ══════════════════════════════════════════════════════════════════════════════
# ── USER CONFIG  ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# 1. Your combined dataframe (rows = networks, columns = features + "dataset")
#    and its property-only slice (no "dataset" / "color_dataset" columns)
#    Replace these two with your actual variables:
#       huge_df              → contains "dataset" column
#       huge_df_properties   → feature columns only
# from your_notebook import huge_df, huge_df_properties

# 2. MaMI taxonomy CSV  (must have an "order" column aligned with the MaMI rows)
MAMI_TAXONOMY_CSV = (
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "data/preprocessed/suarez_MaMI_dataset/04_further_info/"
    "names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv"
)

# 3. Output folder
OUTPUT_FOLDER = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "output/00_trade_off_analysis/spider_mami"
)

# 4. Mapping: feature column name  →  category name
#    Build this from your `meta` DataFrame, e.g.:
#       FEAT_TO_CAT = meta["Category"].to_dict()
#    or hard-code it.  Here we use `meta` (loaded below).

# 5. Category order & colours — paste your config values:
#    from config import CATEGORY_ORDER, CATEGORY_COLOURS
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def build_category_scores(df_properties, feat_to_cat, cat_order):
    """
    One-round z-score, then PC1 projection per category.

    Returns
    -------
    cat_scores   : np.ndarray (n_networks, n_categories)
    cat_labels   : list[str]
    scaler       : fitted StandardScaler
    per_cat_pca  : dict  category -> fitted PCA(1) or None (single-feature cat)
    feat_idx_map : dict  category -> list of column indices
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_properties.values)
    cols   = list(df_properties.columns)

    cat_scores, cat_labels = [], []
    per_cat_pca, feat_idx_map = {}, {}

    for cat in cat_order:
        idx = [i for i, c in enumerate(cols) if feat_to_cat.get(c) == cat]
        if not idx:
            continue
        feat_idx_map[cat] = idx
        X = scaled[:, idx]

        if X.shape[1] == 1:
            scores = X[:, 0]
            per_cat_pca[cat] = None
        else:
            pca1 = PCA(n_components=1)
            scores = pca1.fit_transform(X)[:, 0]
            # Fix sign: mean loading positive → higher score = "more" of category
            if pca1.components_[0].mean() < 0:
                scores *= -1
                pca1.components_ *= -1
            per_cat_pca[cat] = pca1

        cat_scores.append(scores)
        cat_labels.append(cat)

    return (np.stack(cat_scores, axis=1), cat_labels,
            scaler, per_cat_pca, feat_idx_map)


def spider_group(
    ax,
    group_scores,       # (n_animals, n_cats)
    cat_labels,
    color,
    group_name  = "",
    ylim        = (-3, 3),
    ring_vals   = (-2, 0, 2),
    cat_colors  = None,
    n_cats      = None,
    show_individuals = True,
):
    """
    Draw a group spider plot onto an existing polar Axes.

    Shows individual lines (faint), mean ± 1 SD band, and mean line.
    """
    n = len(cat_labels)
    angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_c = angles + angles[:1]

    ax.set_ylim(*ylim)
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_thetagrids([])

    theta = np.linspace(0, 2 * np.pi, 300)

    # ── Gridlines ─────────────────────────────────────────────────────────────
    for rv in ring_vals:
        lw  = 1.2 if rv == 0 else 0.5
        ls  = "-"  if rv == 0 else "--"
        col = "#888888" if rv == 0 else "#cccccc"
        ax.plot(theta, np.full(300, rv),
                color=col, linewidth=lw, linestyle=ls, zorder=0)
        ax.text(np.pi / 2, rv, f"{rv:g}",
                ha="center", va="bottom", fontsize=5.5, color="#aaaaaa")

    # ── Spokes ────────────────────────────────────────────────────────────────
    for angle in angles:
        ax.plot([angle, angle], [ylim[0], ylim[1]],
                color="#e0e0e0", linewidth=0.5, linestyle="--", zorder=0)

    # ── Individual animal lines ───────────────────────────────────────────────
    if show_individuals:
        for row in group_scores:
            row_c = list(row) + [row[0]]
            ax.plot(angles_c, row_c,
                    color=color, linewidth=0.6, alpha=0.18, zorder=2)

    # ── Mean ± SD band ────────────────────────────────────────────────────────
    mean_vals = np.nanmean(group_scores, axis=0)
    std_vals  = np.nanstd (group_scores, axis=0)
    upper     = mean_vals + std_vals
    lower     = mean_vals - std_vals

    # Filled band: fill_between on polar axes needs the closed arrays
    mean_c  = list(mean_vals)  + [mean_vals[0]]
    upper_c = list(upper)      + [upper[0]]
    lower_c = list(lower)      + [lower[0]]

    # Draw the band as a filled polygon between upper and lower
    # We go upper → reverse(lower) to close the polygon
    band_angles = angles_c + angles_c[::-1]
    band_vals   = upper_c  + lower_c[::-1]
    ax.fill(band_angles, band_vals,
            color=color, alpha=0.20, zorder=3)

    # Mean line
    ax.plot(angles_c, mean_c,
            color=color, linewidth=2.2, zorder=4)
    ax.fill(angles_c, mean_c,
            color=color, alpha=0.10, zorder=3)

    # ── Spoke labels ──────────────────────────────────────────────────────────
    label_r = ylim[1] + (ylim[1] - ylim[0]) * 0.10
    for i, (angle, lbl) in enumerate(zip(angles, cat_labels)):
        ha = "center"
        if   angle < np.pi * 0.15 or angle > np.pi * 1.85: ha = "center"
        elif angle < np.pi:                                  ha = "left"
        else:                                                ha = "right"

        va = "center"
        if   np.pi * 0.35 < angle < np.pi * 0.65: va = "bottom"
        elif np.pi * 1.35 < angle < np.pi * 1.65: va = "top"

        col = cat_colors[i] if cat_colors else "#444444"
        wrapped = lbl.replace(" ", "\n") if len(lbl) > 14 else lbl
        ax.text(angle, label_r, wrapped,
                ha=ha, va=va, fontsize=6.5, color=col,
                fontweight="bold", clip_on=False)

    ax.set_title(f"{group_name}\n(n={len(group_scores)})",
                 pad=20, fontsize=9, color=color, fontweight="bold")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(
    huge_df,
    huge_df_properties,
    feat_to_cat,
    cat_order,
    cat_colours,        # dict  category -> hex color
    mami_taxonomy_csv   = MAMI_TAXONOMY_CSV,
    output_folder       = OUTPUT_FOLDER,
    taxonomy_col        = "order",
    min_group_size      = 10,
    n_components_full   = 10,
    show_individuals    = True,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # ── 1. Isolate MaMI rows ──────────────────────────────────────────────────
    mami_mask  = (huge_df["dataset"] == "suarez_MaMI_dataset").values
    mami_props = huge_df_properties[mami_mask].reset_index(drop=True)

    info       = pd.read_csv(mami_taxonomy_csv)
    assert len(info) == mami_mask.sum(), (
        f"Taxonomy CSV has {len(info)} rows but MaMI subset has "
        f"{mami_mask.sum()} rows — they must be aligned."
    )

    # ── 2. Build category scores on MaMI only ─────────────────────────────────
    (cat_scores, cat_labels,
     scaler, per_cat_pca, _) = build_category_scores(
        mami_props, feat_to_cat, cat_order
    )

    cat_colors_list = [cat_colours.get(c, "#444444") for c in cat_labels]

    print(f"Categories ({len(cat_labels)}): {cat_labels}")
    print(f"cat_scores shape: {cat_scores.shape}")

    # ── 3. MaMI-only full PCA (for reference / neighbour lookup if needed) ────
    scaled_mami = scaler.transform(mami_props.values)
    pca_full    = PCA(n_components=min(n_components_full, scaled_mami.shape[1]))
    pca_embed   = pca_full.fit_transform(scaled_mami)
    print(f"MaMI PCA explained variance (first 3 PCs): "
          f"{pca_full.explained_variance_ratio_[:3].round(3)}")

    # ── 4. Identify groups with > min_group_size animals ─────────────────────
    group_counts = info[taxonomy_col].value_counts()
    groups       = group_counts[group_counts > min_group_size].index.tolist()
    print(f"\nGroups with > {min_group_size} animals: {groups}")

    # ── 5. Shared y-limits across all groups ──────────────────────────────────
    ymin = np.floor(cat_scores.min()) - 0.3
    ymax = np.ceil (cat_scores.max()) + 0.3
    ylim = (ymin, ymax)

    step      = 1 if (ymax - ymin) <= 8 else 2
    ring_vals = list(range(int(np.ceil(ymin)),
                            int(np.floor(ymax)) + 1, step))

    # Colour per group — use a qualitative palette
    group_palette = plt.cm.tab10.colors
    group_colors  = {g: group_palette[i % 10] for i, g in enumerate(groups)}

    # ── 6. Combined figure — all groups side by side in one row ──────────────
    n_grps   = len(groups)
    fig_w    = 4.2 * n_grps   # 4.2 inches per panel
    fig_h    = 5.5             # fixed height; labels have room above

    fig_all, axs_all = plt.subplots(
        1, n_grps,
        figsize=(fig_w, fig_h),
        subplot_kw=dict(polar=True),
        dpi=150,
    )
    # ensure iterable even for a single group
    if n_grps == 1:
        axs_all = [axs_all]

    for ax_i, grp in enumerate(groups):
        mask_grp   = (info[taxonomy_col] == grp).values
        grp_scores = cat_scores[mask_grp]
        grp_color  = group_colors[grp]

        spider_group(
            ax               = axs_all[ax_i],
            group_scores     = grp_scores,
            cat_labels       = cat_labels,
            color            = grp_color,
            group_name       = grp,
            ylim             = ylim,
            ring_vals        = ring_vals,
            cat_colors       = cat_colors_list,
            show_individuals = show_individuals,
        )

    fig_all.tight_layout(pad=1.5, w_pad=0.5)
    out_combined = output_folder / "spider_mami_all_groups.pdf"
    fig_all.savefig(out_combined, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_combined}")
    plt.show()

    return cat_scores, cat_labels, pca_embed, group_colors


# ──────────────────────────────────────────────────────────────────────────────
# CALL FROM YOUR NOTEBOOK
# ──────────────────────────────────────────────────────────────────────────────
# In your notebook, after loading everything, just do:
#
#   from spider_mami_subgroups import run
#   from config import CATEGORY_ORDER, CATEGORY_COLOURS
#
#   feat_to_cat = meta["Category"].to_dict()   # meta is your Excel-derived df
#
#   cat_scores, cat_labels, pca_embed, group_colors = run(
#       huge_df            = huge_df,
#       huge_df_properties = huge_df_properties,
#       feat_to_cat        = feat_to_cat,
#       cat_order          = CATEGORY_ORDER,
#       cat_colours        = CATEGORY_COLOURS,
#   )
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # ── Synthetic demo ────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    features = {
        "Topology":   ["mod", "omega", "simplices"],
        "Efficiency": ["glob_eff", "diff_eff"],
        "Wiring":     ["wiring_cost", "long_range"],
        "Spectral":   ["spec_radius", "spec_gap", "sync_ratio", "alg_conn"],
        "Dynamics":   ["T_crit", "size_crit", "div_crit"],
        "Robustness": ["rob_auc", "rob_ratio"],
        "Memory":     ["mc_lin", "mc_nonlin", "ipc_deg1", "ipc_deg2"],
    }
    all_feats   = [f for flist in features.values() for f in flist]
    feat_to_cat = {f: cat for cat, flist in features.items() for f in flist}
    cat_order   = list(features.keys())
    cat_colours = {
        "Topology":   "#E07B39",
        "Efficiency": "#5B8DB8",
        "Wiring":     "#6AAB6A",
        "Spectral":   "#9B6BAD",
        "Dynamics":   "#D4A017",
        "Robustness": "#C75D5D",
        "Memory":     "#4DADA6",
    }

    # Synthetic animal orders with different profiles
    ORDER_MEANS = {
        "Primates":       [ 1.0,  0.5,  0.0, -0.5, 0.2,  0.8,  1.2],
        "Rodentia":       [-0.5, -0.3,  1.0,  0.5, 0.0, -0.5, -0.3],
        "Carnivora":      [ 0.2,  1.0, -0.5,  0.0, 0.8,  0.3,  0.0],
        "Cetartiodactyla":[ 0.0, -0.5,  0.5,  1.0,-0.3,  0.2, -0.8],
        "Chiroptera":     [-1.0,  0.0,  0.2, -0.8, 1.0, -0.2,  0.5],
        "Lagomorpha":     [ 0.5, -1.0, -0.3,  0.3,-0.5,  1.0,  0.2],  # <10 → excluded
    }
    ORDER_N = {
        "Primates": 34, "Rodentia": 28, "Carnivora": 18,
        "Cetartiodactyla": 16, "Chiroptera": 12, "Lagomorpha": 7,
    }

    rows_feats, rows_order = [], []
    for order, n in ORDER_N.items():
        means = ORDER_MEANS[order]
        # Generate correlated features within each category from the order mean
        for _ in range(n):
            row = []
            for cat_i, cat in enumerate(cat_order):
                cat_feats = features[cat]
                base  = means[cat_i] + rng.normal(0, 0.4)
                noise = rng.normal(0, 0.3, len(cat_feats))
                row.extend([base + noise[j] for j in range(len(cat_feats))])
            rows_feats.append(row)
            rows_order.append(order)

    df_feats = pd.DataFrame(rows_feats, columns=all_feats)
    df_meta  = pd.DataFrame({"dataset": "suarez_MaMI_dataset",
                              "color_dataset": "gray"}, index=df_feats.index)
    huge_df_demo  = pd.concat([df_meta, df_feats], axis=1)
    huge_df_props = df_feats.copy()
    info_demo     = pd.DataFrame({"order": rows_order})
    info_demo.to_csv("/tmp/mami_taxonomy_demo.csv", index=False)

    run(
        huge_df            = huge_df_demo,
        huge_df_properties = huge_df_props,
        feat_to_cat        = feat_to_cat,
        cat_order          = cat_order,
        cat_colours        = cat_colours,
        mami_taxonomy_csv  = "/tmp/mami_taxonomy_demo.csv",
        output_folder      = "/tmp/spider_mami_demo",
        taxonomy_col       = "order",
        min_group_size     = 10,
    )