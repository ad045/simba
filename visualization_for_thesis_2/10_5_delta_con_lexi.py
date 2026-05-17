#!/usr/bin/env python
# coding: utf-8

# # Two plots:
# - Total variation bar chart — deltacon, energy, frobenius
# - Minima scatter — top-100 per individual subject, subjects shaded by intensity of dataset color

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from vizman import viz
viz.set_visual_style()

# CONFIG

# distance_measure = "energy" # 
distance_measure = "delta_con" # energy" # "hamming" # delta_con

base_dir = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm")
output_path = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/trade_off_analysis/00_{distance_measure}_results")
output_path.mkdir(parents=True, exist_ok=True)

from config import COLOR_SCHEME

# η / γ normalisation bounds (same as original script)
ETA_MIN, ETA_MAX     = -3, 8
GAMMA_MIN, GAMMA_MAX = -0.1, 1


# # Distance measures for both plots
# DISTANCE_MEASURES_DICT = {"delta_con": "DeltaCon", 
#                      "energy": "Energy", 
#                      "frobenius": "Frobenius"} 

# DISTANCE_MEASURES = DISTANCE_MEASURES_DICT.keys()
# DISTANCE_NAME_IN_CSV = {
#     "delta_con":  "DeltaCon",
#     "energy":    "MaxCrit",
#     "frobenius": "Frobenius",
#     "hamming": "HammingDist",
# }

# # Bar colors for the total-variation plot
# MEASURE_BAR_COLORS = {
#     "energy":    "#E6B213",
#     "delta_con":  "#4BAE6A",
#     "frobenius": "#006685", 
#     "hamming":  "#394D73",
# }

# # CSV paths per distance measure
# # Structure mirrors the new script's csv_files dict, but we need all 3 measures.
# # We use the MEAN across subjects for the total-variation bar (consistent with original).
# # For the minima scatter we only use "energy" (or whichever you prefer — see SCATTER_MEASURE).
# SCATTER_MEASURE = "energy"   # which distance measure drives the scatter plot

# def get_csv_path(dataset_key: str, experiment_suffix: str, measure: str) -> Path:
#     """Build the CSV path following the project convention."""
#     return (
#         base_dir
#         / dataset_key
#         / experiment_suffix
#         / f"summary_indiv_{measure}_for_exp_{experiment_suffix}.csv"
#     )

# # ── datasets to include in the scatter plot ───────────────────────────────────
# SCATTER_DATASETS = {
#     # key → (dataset_folder, experiment_suffix, dataset_color_key)
#     "HCP":      ("hcp_schaefer_100_dataset",
#                     "05_mst_animal_0_compared_with_hcp_schaefer_100",
#                     "hcp_schaefer_100_dataset"),
#     "MaMI":        ("suarez_MaMI_dataset",
#                     "05_mst_animal_0_compared_with_mami",
#                     "suarez_MaMI_dataset"),
#     "Diffusion":   ("kaysons_generated_networks_diffusion",
#                     "05_mst_animal_0_compared_with_diffusion",
#                     "kaysons_generated_networks_diffusion"),
#     "Propagation": ("kaysons_generated_networks_propagation",
#                     "05_mst_animal_0_compared_with_propagation",
#                     "kaysons_generated_networks_propagation"),
#     "Routing":     ("kaysons_generated_networks_routing",
#                     "05_mst_animal_0_compared_with_routing",
#                     "kaysons_generated_networks_routing"),
# }

# # ── datasets to include in the total-variation bar chart ─────────────────────
# VARIATION_DATASETS = SCATTER_DATASETS   # same set

# # ─────────────────────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def lighten_color(hex_color: str, amount: float) -> tuple:
#     """
#     Blend hex_color toward white.
#     amount=0 → original color, amount=1 → white.
#     """
#     rgb = mcolors.to_rgb(hex_color)
#     return tuple(c + (1 - c) * amount for c in rgb)


# def subject_colors(base_hex: str, n: int) -> list:
#     """
#     Return n colors ranging from the base color (darkest / most saturated)
#     to a lighter tint, so each subject is distinguishable.
#     """
#     if n == 1:
#         return [mcolors.to_rgb(base_hex)]
#     # lighten from 0 (full color) to 0.65 (fairly light)
#     return [lighten_color(base_hex, i * 0.65 / (n - 1)) for i in range(n)]


# def load_and_clean(csv_path: Path) -> pd.DataFrame:
#     """Load CSV and deduplicate _x/_y/_z suffixed columns."""
#     df = pd.read_csv(csv_path)
#     meta = {"eta", "filename", "id", "gamma", "network_index"}
#     keys = set(
#         k.replace("_x", "").replace("_y", "").replace("_z", "")
#         for k in df.columns
#     ) - meta
#     all_keys = set(df.columns)
#     first_versions = []
#     for key in keys:
#         for suffix in ("", "_x", "_y", "_z"):
#             if key + suffix in all_keys:
#                 first_versions.append(key + suffix)
#                 break
#     keep = first_versions + [c for c in ("eta", "id", "gamma", "network_index") if c in all_keys]
#     return df[keep]


# def get_df_mean(csv_p: Path, measure: str) -> pd.DataFrame:
#     """Return df_mean with per-subject and mean distance columns."""
#     df = load_and_clean(csv_p)
#     df_mean = df.groupby(["eta", "gamma"]).mean().reset_index()
#     col_prefix = DISTANCE_NAME_IN_CSV[measure]
#     subj_cols  = [c for c in df_mean.columns if f"{col_prefix}_subject_" in c]
#     df_mean[f"{col_prefix}_mean"] = df_mean[subj_cols].mean(axis=1)
#     return df_mean, subj_cols, col_prefix


# In[2]:


# # TOTAL VARIATION BAR CHART
# def compute_combined_variability(dataset_info: dict, measure: str, top_n: int = 100) -> float:
#     """
#     For a given dataset + distance measure, pick the top_n (η, γ) pairs by
#     mean distance score, normalise η and γ, return combined variance.
#     """
#     folder, exp, _ = dataset_info
#     csv_p = get_csv_path(folder, exp, measure)
#     if not csv_p.exists():
#         return np.nan

#     df_mean, subj_cols, col_prefix = get_df_mean(csv_p, measure)
#     mean_col = f"{col_prefix}_mean"

#     top = df_mean.nsmallest(top_n, mean_col)
#     norm_eta   = (top["eta"].values   - ETA_MIN)   / (ETA_MAX   - ETA_MIN)
#     norm_gamma = (top["gamma"].values - GAMMA_MIN) / (GAMMA_MAX - GAMMA_MIN)

#     return float(norm_eta.std() ** 2 + norm_gamma.std() ** 2)



# fig = plt.figure(figsize=viz.cm_to_inch((6, 6)), dpi=150)
# ax  = fig.add_axes([0, 0, 1, 1])

# # Average variability across all datasets for each measure
# bars = {}

# for measure in DISTANCE_MEASURES:
#     vals = []
#     for ds_key, ds_info in VARIATION_DATASETS.items():
#         v = compute_combined_variability(ds_info, measure)
#         if not np.isnan(v):
#             vals.append(v)
#     bars[measure] = np.mean(vals) if vals else np.nan


# # Sort: energy first, then descending
# energy_val   = bars.pop("energy")
# other_sorted = sorted(bars.items(), key=lambda x: x[1], reverse=True)
# ordered      = [("energy", energy_val)] + other_sorted

# xlabel_measures = []
# for x_pos, (measure, val) in enumerate(ordered):
#     xlabel_measures.append(DISTANCE_MEASURES_DICT[measure])
#     ax.bar(x_pos, val, color=MEASURE_BAR_COLORS.get(measure, "gray"), label=measure)

# # Vertical dashed line after "energy" bar
# ax.axvline(x=0.5, linestyle="--", linewidth=0.5, color="k")

# ax.set_xticks([0,1,2], xlabel_measures)
# ax.set_yticks([0, 0.1])
# ax.set_ylabel("Total variation\n(lower is better)")

# ax_w, ax_h = viz.cm_to_inch((6, 6))
# ax.set_position([0.15, 0.15, ax_w / (ax_w + 1), ax_h / (ax_h + 1)])

# # fig.savefig(output_path / "total_variation_deltacon_energy_frobenius.pdf",
# #             bbox_inches="tight")
# # print("Saved:", output_path / "total_variation_deltacon_energy_frobenius.pdf")
# # plt.close(fig)


# In[3]:


# # MINIMA SCATTER  (top-100 per individual subject)
# from scipy.spatial import ConvexHull
# from matplotlib.patches import PathPatch
# from matplotlib.path import Path
# from shapely.geometry import MultiPolygon, Polygon


# measure=SCATTER_MEASURE
# N_DOTS = 10
    
# import alphashape

# OCTAGON_RADIUS = 0.025 # 5  # padding per point
# ALPHA = 4 # 4 ## 2# 5.0            # higher = tighter fit; tune to taste
# N_CORNERS = 50 + 1
# N_SUBJECTS = 3

# ETA_RANGE   = 3 - (-8)   # 11
# GAMMA_RANGE = 1 - (-0.1) # 1.1
# RADIUS_GAMMA = 0.05       # tune this one value in data units of gamma
# RADIUS_ETA   = RADIUS_GAMMA * (ETA_RANGE / GAMMA_RANGE)  # scaled to match


# def achteck_vertices(cx, cy, r):

#     angles = np.linspace(0, 2 * np.pi, N_CORNERS)[:-1]
#     return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


# def achteck_vertices(cx, cy, rx, ry):
#     """Elliptical 'octagon' that appears circular given the axis aspect ratio."""
#     angles = np.linspace(0, 2 * np.pi, N_CORNERS)[:-1]
#     return np.column_stack([cx + rx * np.cos(angles),
#                             cy + ry * np.sin(angles)])
    
# fig = plt.figure(figsize=viz.cm_to_inch((6, 6)), dpi=150)
# ax  = fig.add_axes([0, 0, 1, 1])

# col_prefix = DISTANCE_NAME_IN_CSV[measure]

# for ds_key, (folder, exp, color_key) in SCATTER_DATASETS.items():
#     csv_p = get_csv_path(folder, exp, measure)
#     if not csv_p.exists():
#         print(f"  [SKIP] {csv_p} not found")
#         continue

#     df_mean, subj_cols, _ = get_df_mean(csv_p, measure)
#     base_color = COLOR_SCHEME.get(color_key, "#888888")

#     # For diffusion / propagation / routing there is only one subject
#     # For humans / mami there may be many — autodetect all
#     subj_cols = subj_cols[:N_SUBJECTS] if len(subj_cols) > N_SUBJECTS else subj_cols
#     n_subjects = len(subj_cols)
#     n_subjects = min(n_subjects, N_SUBJECTS)  # avoid zero division if no subj cols
#     colors     = subject_colors(base_color, n_subjects)


#     for i, (col, color) in enumerate(zip(subj_cols, colors)):
#         top100 = df_mean.nsmallest(N_DOTS, col)
#         x = top100["eta"].values
#         y = top100["gamma"].values



#         if len(x) >= 1:
#             all_verts = np.vstack([achteck_vertices(cx, cy, RADIUS_ETA, RADIUS_GAMMA)
#                                    for cx, cy in zip(x, y)])

#             shape = alphashape.alphashape(all_verts, ALPHA)
#             polys = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
#             for poly in polys:
#                 exterior = np.array(poly.exterior.coords)
#                 patch = plt.Polygon(exterior, color=color, alpha=0.5, zorder=1)
#                 ax.add_patch(patch)
#                 # ax.plot(exterior[:, 0], exterior[:, 1], color=color, linewidth=1)
                
#         ax.scatter(x, y, color=color, s=5, # 10,
#                 edgecolor="black", linewidth=0.25, zorder=100)

        

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_ylabel(r"$\gamma$")
# ax.set_xlabel(r"$\eta$")





# from matplotlib.patches import Patch


# legend_elements = []
# for ds_key, (folder, exp, color_key) in SCATTER_DATASETS.items():
#     csv_p = get_csv_path(folder, exp, measure)
#     if not csv_p.exists():
#         continue

#     df_mean, subj_cols, _ = get_df_mean(csv_p, measure)
#     subj_cols = subj_cols[:3] if len(subj_cols) > 3 else subj_cols
#     n_subjects = len(subj_cols)
#     base_color = COLOR_SCHEME.get(color_key, "#888888")
#     colors = subject_colors(base_color, n_subjects)

#     if n_subjects == 1:
#         legend_elements.append(Patch(facecolor=colors[0], label=ds_key))
#     else:
#         for i, color in enumerate(colors):
#             legend_elements.append(Patch(facecolor=color, label=f"{ds_key} - Index {i+1}"))
# ax.legend(
#     handles=legend_elements,
#     # loc="upper right",
#     bbox_to_anchor=(1.15, 1),
#     frameon=False,
#     fontsize=7,
# )



# ax_w, ax_h = viz.cm_to_inch((6, 6))
# ax.set_position([0.15, 0.15, ax_w / (ax_w + 1), ax_h / (ax_h + 1)])

# out = output_path / f"minima_scatter_top100_per_subject_{measure}.pdf"
# # fig.savefig(out, bbox_inches="tight")
# # print("Saved:", out)
# plt.show()


# # Create this plot for every measure, to compare visually how distributed the dots are for each... 

# In[4]:


# N_DOTS = 100

# for measure in DISTANCE_MEASURES:
#     fig = plt.figure(figsize=viz.cm_to_inch((6, 6)), dpi=150)
#     ax  = fig.add_axes([0, 0, 1, 1])
    
        
#     col_prefix = DISTANCE_NAME_IN_CSV[measure]

#     for ds_key, (folder, exp, color_key) in SCATTER_DATASETS.items():
#         csv_p = get_csv_path(folder, exp, measure)
#         if not csv_p.exists():
#             print(f"  [SKIP] {csv_p} not found")
#             continue

#         df_mean, subj_cols, _ = get_df_mean(csv_p, measure)
#         base_color = COLOR_SCHEME.get(color_key, "#888888")

#         # For diffusion / propagation / routing there is only one subject
#         # For humans / mami there may be many — autodetect all
#         subj_cols = subj_cols[:N_SUBJECTS] if len(subj_cols) > N_SUBJECTS else subj_cols
#         n_subjects = len(subj_cols)
#         n_subjects = min(n_subjects, N_SUBJECTS)  # avoid zero division if no subj cols
#         colors     = subject_colors(base_color, n_subjects)


#         for i, (col, color) in enumerate(zip(subj_cols, colors)):
#             top100 = df_mean.nsmallest(N_DOTS, col)
#             x = top100["eta"].values
#             y = top100["gamma"].values



#             if len(x) >= 1:
#                 all_verts = np.vstack([achteck_vertices(cx, cy, RADIUS_ETA, RADIUS_GAMMA)
#                                     for cx, cy in zip(x, y)])

#                 shape = alphashape.alphashape(all_verts, ALPHA)
#                 polys = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
#                 for poly in polys:
#                     exterior = np.array(poly.exterior.coords)
#                     patch = plt.Polygon(exterior, color=color, alpha=0.5, zorder=1)
#                     ax.add_patch(patch)
#                     # ax.plot(exterior[:, 0], exterior[:, 1], color=color, linewidth=1)
                    
#             ax.scatter(x, y, color=color, s=5, # 10,
#                     edgecolor="black", linewidth=0.25, zorder=100)

            

#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_ylabel(r"$\gamma$")
#     ax.set_xlabel(r"$\eta$")

#     legend_elements = []
#     for ds_key, (folder, exp, color_key) in SCATTER_DATASETS.items():
#         csv_p = get_csv_path(folder, exp, measure)
#         if not csv_p.exists():
#             print("cont")
#             continue

#         df_mean, subj_cols, _ = get_df_mean(csv_p, measure)
#         subj_cols = subj_cols[:3] if len(subj_cols) > 3 else subj_cols
#         n_subjects = len(subj_cols)
#         base_color = COLOR_SCHEME.get(color_key, "#888888")
#         colors = subject_colors(base_color, n_subjects)

#         if n_subjects == 1:
#             legend_elements.append(Patch(facecolor=colors[0], label=ds_key))
#         else:
#             for i, color in enumerate(colors):
#                 legend_elements.append(Patch(facecolor=color, label=f"{ds_key} - Index {i+1}"))

#     ax.legend(
#         handles=legend_elements,
#         # loc="upper right",
#         bbox_to_anchor=(1.15, 1),
#         frameon=False,
#         fontsize=7,
#     )
    
#     plt.title(f"{DISTANCE_MEASURES_DICT[measure]}") # .replace(" ", "")}")



#     ax_w, ax_h = viz.cm_to_inch((6, 6))
#     ax.set_position([0.15, 0.15, ax_w / (ax_w + 1), ax_h / (ax_h + 1)])

#     out = output_path / f"minima_scatter_top100_per_subject_{measure}.pdf"
#     # fig.savefig(out, bbox_inches="tight")
#     # print("Saved:", out)
#     plt.show()


# In[5]:


# N_DOTS = 100

# for measure in DISTANCE_MEASURES:
#     fig = plt.figure(figsize=viz.cm_to_inch((6, 6)), dpi=150)
#     ax  = fig.add_axes([0, 0, 1, 1])
    
        
#     col_prefix = DISTANCE_NAME_IN_CSV[measure]

#     for ds_key, (folder, exp, color_key) in SCATTER_DATASETS.items():
#         csv_p = get_csv_path(folder, exp, measure)
#         if not csv_p.exists():
#             print(f"  [SKIP] {csv_p} not found")
#             continue

#         df_mean, subj_cols, _ = get_df_mean(csv_p, measure)
#         base_color = COLOR_SCHEME.get(color_key, "#888888")

#         # For diffusion / propagation / routing there is only one subject
#         # For humans / mami there may be many — autodetect all
#         subj_cols = subj_cols[:N_SUBJECTS] if len(subj_cols) > N_SUBJECTS else subj_cols
#         n_subjects = len(subj_cols)
#         n_subjects = min(n_subjects, N_SUBJECTS)  # avoid zero division if no subj cols
#         colors     = subject_colors(base_color, n_subjects)


#         for i, (col, color) in enumerate(zip(subj_cols, colors)):
#             top100 = df_mean.nsmallest(N_DOTS, col)
#             x = top100["eta"].values
#             y = top100["gamma"].values



#             if len(x) >= 1:
#                 all_verts = np.vstack([achteck_vertices(cx, cy, RADIUS_ETA, RADIUS_GAMMA)
#                                     for cx, cy in zip(x, y)])

#                 shape = alphashape.alphashape(all_verts, ALPHA)
#                 polys = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
#                 for poly in polys:
#                     exterior = np.array(poly.exterior.coords)
#                     patch = plt.Polygon(exterior, color=color, alpha=0.5, zorder=1)
#                     ax.add_patch(patch)
#                     # ax.plot(exterior[:, 0], exterior[:, 1], color=color, linewidth=1)
                    
#             # ax.scatter(x, y, color=color, s=5, # 10,
#             #         edgecolor="black", linewidth=0.25, zorder=100)

            

#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_ylabel(r"$\gamma$")
#     ax.set_xlabel(r"$\eta$")

#     legend_elements = []
#     for ds_key, (folder, exp, color_key) in SCATTER_DATASETS.items():
#         csv_p = get_csv_path(folder, exp, measure)
#         if not csv_p.exists():
#             print("cont")
#             continue

#         df_mean, subj_cols, _ = get_df_mean(csv_p, measure)
#         subj_cols = subj_cols[:3] if len(subj_cols) > 3 else subj_cols
#         n_subjects = len(subj_cols)
#         base_color = COLOR_SCHEME.get(color_key, "#888888")
#         colors = subject_colors(base_color, n_subjects)

#         if n_subjects == 1:
#             legend_elements.append(Patch(facecolor="none", edgecolor=colors[0], label=ds_key))
#         else:
#             for i, color in enumerate(colors):
#                 legend_elements.append(Patch(facecolor="none", edgecolor=color, label=f"{ds_key} - Index {i+1}"))

#     ax.legend(
#         handles=legend_elements,
#         # loc="upper right",
#         bbox_to_anchor=(1.15, 1),
#         frameon=False,
#         fontsize=7,
#     )
    
#     plt.title(f"{DISTANCE_MEASURES_DICT[measure]}") # .replace(" ", "")}")



#     ax_w, ax_h = viz.cm_to_inch((6, 6))
#     ax.set_position([0.15, 0.15, ax_w / (ax_w + 1), ax_h / (ax_h + 1)])

#     out = output_path / f"minima_scatter_top100_per_subject_{measure}.pdf"
#     # fig.savefig(out, bbox_inches="tight")
#     # print("Saved:", out)
#     plt.show()


# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from vizman import viz

from config import emp_dataset_and_experiment_pairs, LABEL_MAP, PROPERTY_NAMES, COLOR_SCHEME

base_dir = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm")



distance_name_in_csv_dict = {
    "hamming": "HammingDist", 
    "delta_con": "DeltaCon", 
    "energy": "MaxCrit"
}
distance_name_in_csv = distance_name_in_csv_dict[distance_measure]

csv_files = {}

datasets_to_look_at = { # emp_dataset_and_experiment_pairs
                    "suarez_MaMI_dataset": "05_mst_animal_0_compared_with_mami", 
                    
                    "lexis_data_developing": "05_mst_animal_0_compared_with_lexis_data_developing", 
                    "lexis_data_young": "05_mst_animal_0_compared_with_lexis_data_young",
                    "lexis_data_aging": "05_mst_animal_0_compared_with_lexis_data_aging",
                    
                    # "hcp_schaefer_100_dataset_gnm": "11_mst_2500_animal_0", 
                    # "hcp_schaefer_100_dataset": "05_mst_animal_0_compared_with_hcp_schaefer_100",
                     
                    "kaysons_generated_networks_diffusion": "05_mst_animal_0_compared_with_diffusion", 
                    "kaysons_generated_networks_propagation": "05_mst_animal_0_compared_with_propagation", 
                    #  "kaysons_generated_networks_resistance": "05_mst_animal_0_compared_with_resistance", 
                    "kaysons_generated_networks_routing": "05_mst_animal_0_compared_with_routing", 
                    # "kaysons_generated_networks_topology": "05_mst_animal_0_compared_with_topology", 
                    
                    # "lexis_data_developing_consensus_per_age_1_year": "00_pca",
                    # "lexis_data_young_consensus_per_age_1_year": "00_pca",
                    # "lexis_data_aging_consensus_per_age_1_year": "00_pca",
                    # "lexis_data_all_consensus_per_age_1_year": "00_pca",
                    # "lexis_data_all_consensus_per_age_2_year": "00_pca",
                }


if distance_measure == "energy": 
    datasets_to_look_at = { # emp_dataset_and_experiment_pairs
                        "suarez_MaMI_dataset": "05_mst_animal_0_compared_with_mami", 
                        
                        # "lexis_data_developing": "05_mst_animal_0_compared_with_lexis_data_developing", 
                        # "lexis_data_young": "05_mst_animal_0_compared_with_lexis_data_young",
                        # "lexis_data_aging": "05_mst_animal_0_compared_with_lexis_data_aging",
                        
                        # "hcp_schaefer_100_dataset_gnm": "11_mst_2500_animal_0", 
                        # "hcp_schaefer_100_dataset": "05_mst_animal_0_compared_with_hcp_schaefer_100",
                        
                        "kaysons_generated_networks_diffusion": "05_mst_animal_0_compared_with_diffusion", 
                        "kaysons_generated_networks_propagation": "05_mst_animal_0_compared_with_propagation", 
                        #  "kaysons_generated_networks_resistance": "05_mst_animal_0_compared_with_resistance", 
                        "kaysons_generated_networks_routing": "05_mst_animal_0_compared_with_routing", 
                        # "kaysons_generated_networks_topology": "05_mst_animal_0_compared_with_topology", 
                        
                        # "lexis_data_developing_consensus_per_age_1_year": "00_pca",
                        # "lexis_data_young_consensus_per_age_1_year": "00_pca",
                        # "lexis_data_aging_consensus_per_age_1_year": "00_pca",
                        # "lexis_data_all_consensus_per_age_1_year": "00_pca",
                        # "lexis_data_all_consensus_per_age_2_year": "00_pca",
                    }


for dataset_name in datasets_to_look_at: # emp_dataset_and_experiment_pairs
    dataset_folder = dataset_name
    experiment_name = emp_dataset_and_experiment_pairs[dataset_name]
    if dataset_name == "hcp_schaefer_100_dataset_gnm": 
        # dataset_folder = "hcp_schaefer_100_dataset"
        continue
    csv_files[dataset_name] = base_dir / f"{dataset_folder}/{experiment_name}/summary_indiv_{distance_measure}_for_exp_{experiment_name}.csv"
    # if dataset_name == "hcp_schaefer_100_dataset_gnm":
    #     csv_files[dataset_name] = base_dir / f"{dataset_folder}/{experiment_name}/summary_indiv_{distance_measure}_for_exp_{experiment_name}.csv"

# csv_files = {
#     'humans': base_dir / f"hcp_schaefer_100_dataset/05_mst_animal_0_compared_with_hcp_schaefer_100/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_hcp_schaefer_100.csv",
#     'diffusion': base_dir / f"kaysons_generated_networks_diffusion/05_mst_animal_0_compared_with_diffusion/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_diffusion.csv",
#     'propagation': base_dir / f"kaysons_generated_networks_propagation/05_mst_animal_0_compared_with_propagation/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_propagation.csv",
#     'routing': base_dir / f"kaysons_generated_networks_routing/05_mst_animal_0_compared_with_routing/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_routing.csv",
#     'mami': base_dir / f"suarez_MaMI_dataset/05_mst_animal_0_compared_with_mami/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_mami.csv"
# }


# In[7]:


# distance_measure = "energy" # "hamming" # delta_con

# Store all minima for each network type
all_minima = {}

# Do this in subplots
fig, axs = plt.subplots(nrows=3, ncols=3, 
                        figsize=viz.cm_to_inch((18, 18)), 
                        sharex=True, sharey=True, dpi=100)
axs = axs.flatten()

# Process each CSV file
for i, (network_name, get_csv_path) in enumerate(csv_files.items()):
    print(f"Processing {network_name}...")
    
    # Load the data
    df = pd.read_csv(get_csv_path)
    
    # Remove all "_x" and "_y" etc from the keys, and remove then all columns that have already appeared (exactly do this for "_x", "_y", "_z")
    # unique_keys = list(set([k.replace("_x", "").replace("_y", "").replace("_z", "")+"_x" for k in df.keys()]) - set(['eta_x', 'filename_x', 'id_x', 'gamma_x', 'network_index_x']))
    # unique_keys = unique_keys + ['eta', 'filename', 'id', 'gamma', 'network_index']
    # df = df[unique_keys]
    keys = set([k.replace("_x", "").replace("_y", "").replace("_z", "") for k in df.keys()])
    keys = keys - set(['eta', 'filename', 'id', 'gamma', 'network_index'])
    unique_keys = list(keys)
    unique_keys

    all_keys = set(df.keys())
    # Get the first column-name version for each unique key (e.g. "eta_x" for "eta", "gamma_x" for "gamma", etc.)
    first_versions = []
    for key in unique_keys:
        for suffix in ["", "_x", "_y", "_z"]:
            candidate = key + suffix
            if candidate in all_keys:
                first_versions.append(candidate)
                break

    # print(len(first_versions), len(df.columns))

    df = df[first_versions + ['eta', 'id', 'gamma', 'network_index']] # 'filename', 
    
    # # Drop filename column if it exists
    # if 'filename' in df.columns:
    #     df.drop(columns=["filename"], inplace=True)
    
    print(len(df.keys()))
    
    # Group by eta and gamma and calculate mean
    df_mean = df.groupby(["eta", "gamma"]).mean().reset_index()
    
    # Find minima for each subject (DeltaCon columns)
    distance_measure_columns = [col for col in df_mean.columns if f"{distance_name_in_csv}_subject_" in col]

    df_mean[f"{distance_name_in_csv}_mean"] = df_mean[distance_measure_columns].mean(axis=1)

    minima_eta = []
    minima_gamma = []
    number_minima = []

    for col in distance_measure_columns:
        # Find the row with minimum DeltaCon for this subject
        min_idx = df_mean[col].idxmin()
        minima_eta.append(df_mean.loc[min_idx, "eta"])
        minima_gamma.append(df_mean.loc[min_idx, "gamma"])
        # Add the number of minimum values to an array 
        # number_minima.append(df_mean.loc[min_idx, "gamma"].to_numpy())
        # print(np.array(df_mean.loc[min_idx, "gamma"]).shape)

    all_minima[network_name] = {
        'eta': minima_eta,
        'gamma': minima_gamma,
        'df_mean': df_mean, 
        'number_minima': number_minima
    }
    print(f"  Found {len(minima_eta)} minima")
    
    cmap = "Grays"

    # CHANGE SCALING HERE: EITHER FROM 300 to 2300, or for every plot individually. 
    background = axs[i].imshow(df_mean.pivot(index="gamma", columns="eta", values=f"{distance_name_in_csv}_mean"), 
                               extent=(df_mean["eta"].min(), df_mean["eta"].max(), df_mean["gamma"].min(), df_mean["gamma"].max()), 
                               origin='lower', aspect='auto', cmap=cmap) # , vmin=300, vmax=2300) # , alpha=0.5)
    
    
    axs[i].set_title(LABEL_MAP[network_name]) # .capitalize())
    if i in [0, 3, 6]:  # left column
        axs[i].set_ylabel(PROPERTY_NAMES["gamma"])
        axs[i].set_yticks([-0.1, 0, 1])
    if i in [6, 7]: 
        axs[i].set_xlabel(PROPERTY_NAMES["eta"])
        axs[i].set_xticks([-8, 0, 3])
        
    # axs[i].set_ylabel(PROPERTY_NAMES["gamma"])
    if i in [2,5]: 
        plt.colorbar(background, ax=axs[i], label=f"Mean {distance_name_in_csv.capitalize()}")
    else: 
        plt.colorbar(background, ax=axs[i])

    # Add minima points
    axs[i].scatter(minima_eta, minima_gamma, 
                   color=COLOR_SCHEME[network_name], # 'red', 
                   s=20, 
                   label='Minima', 
                   edgecolor='black')

# Remove empty subplot
if len(csv_files) < len(axs):
    for j in range(len(csv_files), len(axs)):
        fig.delaxes(axs[j])

plt.suptitle(f"{distance_measure.capitalize()}") # .replace(" ", "")}")
plt.tight_layout()
plt.savefig(output_path / f"{distance_measure}_all_datasets.pdf", bbox_inches="tight")
print(output_path / f"{distance_measure}_all_datasets.pdf")


# In[8]:


# SHARED COLORBAR

# distance_measure = "energy" # "hamming" # delta_con

# Store all minima for each network type
all_minima = {}

# Do this in subplots
fig, axs = plt.subplots(nrows=3, ncols=3, 
                        figsize=viz.cm_to_inch((18, 18)), 
                        sharex=True, sharey=True, dpi=100)
axs = axs.flatten()

# Process each CSV file
for i, (network_name, get_csv_path) in enumerate(csv_files.items()):
    print(f"Processing {network_name}...")
    
    # Load the data
    df = pd.read_csv(get_csv_path)
    
    # Remove all "_x" and "_y" etc from the keys, and remove then all columns that have already appeared (exactly do this for "_x", "_y", "_z")
    # unique_keys = list(set([k.replace("_x", "").replace("_y", "").replace("_z", "")+"_x" for k in df.keys()]) - set(['eta_x', 'filename_x', 'id_x', 'gamma_x', 'network_index_x']))
    # unique_keys = unique_keys + ['eta', 'filename', 'id', 'gamma', 'network_index']
    # df = df[unique_keys]
    keys = set([k.replace("_x", "").replace("_y", "").replace("_z", "") for k in df.keys()])
    keys = keys - set(['eta', 'filename', 'id', 'gamma', 'network_index'])
    unique_keys = list(keys)
    unique_keys

    all_keys = set(df.keys())
    # Get the first column-name version for each unique key (e.g. "eta_x" for "eta", "gamma_x" for "gamma", etc.)
    first_versions = []
    for key in unique_keys:
        for suffix in ["", "_x", "_y", "_z"]:
            candidate = key + suffix
            if candidate in all_keys:
                first_versions.append(candidate)
                break

    # print(len(first_versions), len(df.columns))

    df = df[first_versions + ['eta', 'id', 'gamma', 'network_index']] # 'filename', 
    
    # # Drop filename column if it exists
    # if 'filename' in df.columns:
    #     df.drop(columns=["filename"], inplace=True)
    
    print(len(df.keys()))
    
    # Group by eta and gamma and calculate mean
    df_mean = df.groupby(["eta", "gamma"]).mean().reset_index()
    
    # Find minima for each subject (DeltaCon columns)
    distance_measure_columns = [col for col in df_mean.columns if f"{distance_name_in_csv}_subject_" in col]

    df_mean[f"{distance_name_in_csv}_mean"] = df_mean[distance_measure_columns].mean(axis=1)

    minima_eta = []
    minima_gamma = []
    number_minima = []

    for col in distance_measure_columns:
        # Find the row with minimum DeltaCon for this subject
        min_idx = df_mean[col].idxmin()
        minima_eta.append(df_mean.loc[min_idx, "eta"])
        minima_gamma.append(df_mean.loc[min_idx, "gamma"])
        # Add the number of minimum values to an array 
        # number_minima.append(df_mean.loc[min_idx, "gamma"].to_numpy())
        # print(np.array(df_mean.loc[min_idx, "gamma"]).shape)

    all_minima[network_name] = {
        'eta': minima_eta,
        'gamma': minima_gamma,
        'df_mean': df_mean, 
        'number_minima': number_minima
    }
    print(f"  Found {len(minima_eta)} minima")
    
    cmap = "Grays"

    # CHANGE SCALING HERE: EITHER FROM 300 to 2300, or for every plot individually. 
    background = axs[i].imshow(df_mean.pivot(index="gamma", columns="eta", values=f"{distance_name_in_csv}_mean"), 
                               extent=(df_mean["eta"].min(), df_mean["eta"].max(), df_mean["gamma"].min(), df_mean["gamma"].max()), 
                               origin='lower', aspect='auto', cmap=cmap, 
                               vmin=6, vmax=11
                               ) # , vmin=300, vmax=2300) # , alpha=0.5)
    
    
    axs[i].set_title(LABEL_MAP[network_name]) # .capitalize())
    if i in [0, 3, 6]:  # left column
        axs[i].set_ylabel(PROPERTY_NAMES["gamma"])
        axs[i].set_yticks([-0.1, 0, 1])
    if i in [6, 7]: 
        axs[i].set_xlabel(PROPERTY_NAMES["eta"])
        axs[i].set_xticks([-8, 0, 3])
        
    # axs[i].set_ylabel(PROPERTY_NAMES["gamma"])
    if i in [2,5]: 
        plt.colorbar(background, ax=axs[i], label=f"Mean {distance_name_in_csv.capitalize()}")
    # else: 
        # plt.colorbar(background, ax=axs[i])

    # Add minima points
    axs[i].scatter(minima_eta, minima_gamma, 
                   color=COLOR_SCHEME[network_name], # 'red', 
                   s=20, 
                   label='Minima', 
                   edgecolor='black')

# Remove empty subplot
if len(csv_files) < len(axs):
    for j in range(len(csv_files), len(axs)):
        fig.delaxes(axs[j])

plt.suptitle(f"{distance_measure.capitalize()}") # .replace(" ", "")}")
plt.tight_layout()
# plt.savefig(output_path / f"energy_all_datasets.pdf", bbox_inches="tight")
# print(output_path / f"energy_all_datasets.pdf")


# In[9]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Collect the 100 lowest DeltaCon values for each network type
boxplot_data = []
labels = []

# colors_list = ['red', 'blue', 'green', 'orange', 'purple']
colors_list = []
# network_types = ['diffusion', 'mami', 'routing', 'propagation', 'humans']

for network_name in all_minima: # network_types:
    df_mean = all_minima[network_name]['df_mean']
    
    # Ensure DeltaCon_mean exists
    if f'{distance_name_in_csv}_mean' not in df_mean.columns:
        distance_measure_columns = [col for col in df_mean.columns if f"{distance_name_in_csv}_subject_" in col]
        df_mean[f'{distance_name_in_csv}_mean'] = df_mean[distance_measure_columns].mean(axis=1)

    # Get the 100 lowest values from the ENTIRE dataframe
    lowest_100 = df_mean[f'{distance_name_in_csv}_mean'].nsmallest(100).values
    boxplot_data.append(lowest_100)
    labels.append(LABEL_MAP[network_name]) # .capitalize())
    colors_list.append(COLOR_SCHEME[network_name])

# Create the boxplot
fig, ax = plt.subplots(figsize=viz.cm_to_inch((9,6)), dpi=100)

# Add scatter points for the 100 lowest values
for i, data in enumerate(boxplot_data):
    x = np.random.normal(i + 1, 0.1, # 04, 
                         size=len(data))  # Jitter the x-values
    ax.scatter(x, data, color=colors_list[i], 
               s=5, 
               alpha=0.6, 
               edgecolor='black', 
               linewidth=0, # 0.5
               )
   
# Box plot
bp = ax.boxplot(boxplot_data, labels=labels, 
                patch_artist=True, showfliers=False, # to hide outliers
                # showmeans=True, meanline=False,
                medianprops=dict(color='black', 
                                 linewidth=1
                                 ),
                # meanprops=dict(marker='.',  # D
                #                markerfacecolor='white', 
                #                markeredgecolor='black', 
                #                markersize=6
                #                )
                )

# Color the boxes
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel(f'{distance_name_in_csv.capitalize()} Mean')
# Set every second tick a bit lower to avoid overlap
for tick in ax.get_xticklabels()[::2]:
    tick.set_y(tick.get_position()[1] - 0.05)
ax.set_xticklabels(labels, # rotation=45, 
                   ha='center') # right')

ax.set_title(f'100 Lowest {distance_name_in_csv.capitalize()} Values per Network Type')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_path / f"{distance_measure}_boxplot_lowest_100_values.pdf", bbox_inches="tight")
print(output_path / f"{distance_measure}_boxplot_lowest_100_values.pdf")
plt.show()

# Print summary statistics for the 100 lowest values
print("\n" + "="*60)
print("SUMMARY STATISTICS (100 LOWEST VALUES)")
print("="*60)
for i, network_name in enumerate(all_minima.keys()):
    data = boxplot_data[i]
    print(f"\n{network_name.upper()}:")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Q1 (25th percentile): {np.percentile(data, 25):.6f}")
    print(f"  Median: {np.median(data):.6f}")
    print(f"  Q3 (75th percentile): {np.percentile(data, 75):.6f}")
    print(f"  Max: {np.max(data):.6f}")
    print(f"  Mean ± std: {np.mean(data):.6f} ± {np.std(data):.6f}")
    


# In[10]:


from mpl_toolkits.axes_grid1 import make_axes_locatable


# ── Combined figure using subplot_mosaic ──────────────────────────────────────

network_names = list(csv_files.keys())   # 7 entries expected

# Build mosaic layout: 7 heatmap panels + 1 wide boxplot (bottom-right 2 cells)
if distance_measure == "energy": 
   mosaic = [
        [network_names[0], network_names[1], network_names[2]],
        [network_names[3], 'boxplot',        'boxplot'       ],
    ]
   figsize = viz.cm_to_inch((18, 12))

else: 
    mosaic = [
        [network_names[0], network_names[1], network_names[2]],
        [network_names[3], network_names[4], network_names[5]],
        [network_names[6], 'boxplot',        'boxplot'       ],
    ]
    figsize = viz.cm_to_inch((18, 18))

fig, axs = plt.subplot_mosaic(
    mosaic,
    figsize=figsize,
    dpi=100,
)

# ── shared axis limits (applied manually since mosaic doesn't do sharex/sharey) ──
X_MIN, X_MAX = None, None   # will be set from first dataset
Y_MIN, Y_MAX = None, None

# ── 1. Heatmap panels ─────────────────────────────────────────────────────────
all_minima = {}

for pos_idx, (network_name, get_csv_path) in enumerate(csv_files.items()):
    print(f"Processing {network_name}...")
    ax = axs[network_name]

    # ── Load & deduplicate columns ────────────────────────────────────────────
    df = pd.read_csv(get_csv_path)
    all_keys = set(df.keys())
    keys = set([k.replace("_x","").replace("_y","").replace("_z","") for k in all_keys])
    keys -= {'eta', 'filename', 'id', 'gamma', 'network_index'}
    first_versions = []
    for key in keys:
        for suffix in ["", "_x", "_y", "_z"]:
            candidate = key + suffix
            if candidate in all_keys:
                first_versions.append(candidate)
                break
    df = df[first_versions + ['eta', 'id', 'gamma', 'network_index']]
    print(f"  {len(df.keys())} columns retained")

    df_mean = df.groupby(["eta", "gamma"]).mean().reset_index()

    # Update shared axis limits
    if X_MIN is None:
        X_MIN, X_MAX = df_mean["eta"].min(),   df_mean["eta"].max()
        Y_MIN, Y_MAX = df_mean["gamma"].min(), df_mean["gamma"].max()

    # ── Per-subject minima ────────────────────────────────────────────────────
    distance_measure_columns = [c for c in df_mean.columns if f"{distance_name_in_csv}_subject_" in c]
    df_mean[f"{distance_name_in_csv}_mean"] = df_mean[distance_measure_columns].mean(axis=1)

    minima_eta, minima_gamma = [], []
    for col in distance_measure_columns:
        min_idx = df_mean[col].idxmin()
        minima_eta.append(df_mean.loc[min_idx, "eta"])
        minima_gamma.append(df_mean.loc[min_idx, "gamma"])

    all_minima[network_name] = {
        'eta': minima_eta, 'gamma': minima_gamma,
        'df_mean': df_mean,
    }
    print(f"  Found {len(minima_eta)} minima")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    pivot = df_mean.pivot(index="gamma", columns="eta", values=f"{distance_name_in_csv}_mean")
    im = ax.imshow(
        pivot,
        extent=(df_mean["eta"].min(), df_mean["eta"].max(),
                df_mean["gamma"].min(), df_mean["gamma"].max()),
        origin='lower', aspect='auto', cmap="Grays",
    )
    ax.set_title(LABEL_MAP[network_name])

    # Determine grid position for axis-label / tick decisions
    col_idx = mosaic[pos_idx // 3][pos_idx % 3]   # same as network_name
    row = pos_idx // 3
    col = pos_idx % 3

    if col == 0:
        ax.set_ylabel(PROPERTY_NAMES["gamma"])
        ax.set_yticks([-0.1, 0, 1])
    else:
        ax.set_yticks([])

    if row == 2:                         # bottom row (only Routing here)
        ax.set_xlabel(PROPERTY_NAMES["eta"])
        ax.set_xticks([-8, 0, 3])
    else:
        ax.set_xticks([])

    # # Colorbar on right column panels only
    # if col == 2:
    #     plt.colorbar(im, ax=ax, label=f"Mean {distance_name_in_csv.capitalize()}")
    # else:
    #     plt.colorbar(im, ax=ax)


    # Replace every colorbar call inside the heatmap loop with this pattern:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if col == 2:
        plt.colorbar(im, cax=cax, label=f"Mean {distance_name_in_csv.capitalize()}")
    else:
        plt.colorbar(im, cax=cax)
        
    # ── Scatter minima ────────────────────────────────────────────────────────
    ax.scatter(minima_eta, minima_gamma,
               color=COLOR_SCHEME[network_name],
               s=20, label='Minima', edgecolor='black')

    # Enforce shared limits
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)





# ── 2. Boxplot panel ──────────────────────────────────────────────────────────
ax_box = axs['boxplot']

boxplot_data, labels, colors_list = [], [], []

for network_name in all_minima:
    df_mean = all_minima[network_name]['df_mean']
    if f'{distance_name_in_csv}_mean' not in df_mean.columns:
        dcols = [c for c in df_mean.columns if f"{distance_name_in_csv}_subject_" in c]
        df_mean[f'{distance_name_in_csv}_mean'] = df_mean[dcols].mean(axis=1)

    lowest_100 = df_mean[f'{distance_name_in_csv}_mean'].nsmallest(100).values
    boxplot_data.append(lowest_100)
    labels.append(LABEL_MAP[network_name])
    colors_list.append(COLOR_SCHEME[network_name])

# Scatter jitter
for i, data in enumerate(boxplot_data):
    x = np.random.normal(i + 1, 0.1, size=len(data))
    ax_box.scatter(x, data, color=colors_list[i],
                   s=5, alpha=0.6, edgecolor='black', linewidth=0)

bp = ax_box.boxplot(
    boxplot_data, labels=labels,
    patch_artist=True, showfliers=False,
    medianprops=dict(color='black', linewidth=1),
)
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax_box.set_ylabel(f'{distance_name_in_csv.capitalize()} Mean')
ax_box.set_title(f'100 Lowest {distance_name_in_csv.capitalize()} Values per Network Type')
ax_box.grid(axis='y', alpha=0.3, linestyle='--')

# Stagger every other x-tick label slightly lower to avoid overlap
for j, tick in enumerate(ax_box.get_xticklabels()):
    if j % 2 == 0:
        tick.set_y(tick.get_position()[1] - 0.05)

# After building the boxplot, balance its width to match heatmap columns:
# divider = make_axes_locatable(ax_box)
# cax_dummy = divider.append_axes("right", size="5%", pad=0.05)
# cax_dummy.set_visible(False) 

# ── Final touches ─────────────────────────────────────────────────────────────
plt.suptitle(distance_name_in_csv.capitalize())
plt.tight_layout()
plt.savefig(output_path / f"{distance_name_in_csv}_combined_mosaic.pdf", bbox_inches="tight")
print(output_path / f"{distance_name_in_csv}_combined_mosaic.pdf")
plt.show()


# In[11]:


# {name: path for name, path in csv_files.items() if name in network_names}
{name: csv_files[name]for name in list(datasets_to_look_at.keys()) if name in network_names}


# In[12]:


from mpl_toolkits.axes_grid1 import make_axes_locatable


datasets_to_look_at = { # emp_dataset_and_experiment_pairs
                    "suarez_MaMI_dataset": "05_mst_animal_0_compared_with_mami", 
                    
                    "lexis_data_developing": "05_mst_animal_0_compared_with_lexis_data_developing", 
                    # "lexis_data_young": "05_mst_animal_0_compared_with_lexis_data_young",
                    # "lexis_data_aging": "05_mst_animal_0_compared_with_lexis_data_aging",
                    
                    # "hcp_schaefer_100_dataset_gnm": "11_mst_2500_animal_0", 
                    # "hcp_schaefer_100_dataset": "05_mst_animal_0_compared_with_hcp_schaefer_100",
                     
                    "kaysons_generated_networks_diffusion": "05_mst_animal_0_compared_with_diffusion", 
                    "kaysons_generated_networks_propagation": "05_mst_animal_0_compared_with_propagation", 
                    #  "kaysons_generated_networks_resistance": "05_mst_animal_0_compared_with_resistance", 
                    "kaysons_generated_networks_routing": "05_mst_animal_0_compared_with_routing", 
                    # "kaysons_generated_networks_topology": "05_mst_animal_0_compared_with_topology", 
                    
                    # "lexis_data_developing_consensus_per_age_1_year": "00_pca",
                    # "lexis_data_young_consensus_per_age_1_year": "00_pca",
                    # "lexis_data_aging_consensus_per_age_1_year": "00_pca",
                    # "lexis_data_all_consensus_per_age_1_year": "00_pca",
                    # "lexis_data_all_consensus_per_age_2_year": "00_pca",
                }

# ── Combined figure using subplot_mosaic ──────────────────────────────────────

network_names = list(datasets_to_look_at.keys()) # list(csv_files.keys())   # 7 entries expected

# csv_files: only have the ones that have the network_names names
# csv_files = {name: path for name, path in csv_files.items() if name in network_names}
csv_files = {name: csv_files[name]for name in list(datasets_to_look_at.keys()) if name in network_names}
# Build mosaic layout: 7 heatmap panels + 1 wide boxplot (bottom-right 2 cells)
if distance_measure == "energy": 
   mosaic = [
        [network_names[0], network_names[1], network_names[2]],
        [network_names[3], 'boxplot',        'boxplot'       ],
    ]
   figsize = viz.cm_to_inch((18, 12))

else: 
    mosaic = [
        [network_names[0], network_names[1], 'boxplot'],
        [network_names[2], network_names[3], network_names[4]],
    ]
    figsize = viz.cm_to_inch((18, 10))

fig, axs = plt.subplot_mosaic(
    mosaic,
    figsize=figsize,
    dpi=100,
)

# ── shared axis limits (applied manually since mosaic doesn't do sharex/sharey) ──
X_MIN, X_MAX = None, None   # will be set from first dataset
Y_MIN, Y_MAX = None, None

# ── 1. Heatmap panels ─────────────────────────────────────────────────────────
all_minima = {}

# for pos_idx, network_name in enumerate(network_names): # 
for pos_idx, (network_name, get_csv_path) in enumerate(csv_files.items()):
    print(f"Processing {network_name}...")
    ax = axs[network_name]

    # ── Load & deduplicate columns ────────────────────────────────────────────
    df = pd.read_csv(get_csv_path)
    all_keys = set(df.keys())
    keys = set([k.replace("_x","").replace("_y","").replace("_z","") for k in all_keys])
    keys -= {'eta', 'filename', 'id', 'gamma', 'network_index'}
    first_versions = []
    for key in keys:
        for suffix in ["", "_x", "_y", "_z"]:
            candidate = key + suffix
            if candidate in all_keys:
                first_versions.append(candidate)
                break
    df = df[first_versions + ['eta', 'id', 'gamma', 'network_index']]
    print(f"  {len(df.keys())} columns retained")

    df_mean = df.groupby(["eta", "gamma"]).mean().reset_index()

    # Update shared axis limits
    if X_MIN is None:
        X_MIN, X_MAX = df_mean["eta"].min(),   df_mean["eta"].max()
        Y_MIN, Y_MAX = df_mean["gamma"].min(), df_mean["gamma"].max()

    # ── Per-subject minima ────────────────────────────────────────────────────
    distance_measure_columns = [c for c in df_mean.columns if f"{distance_name_in_csv}_subject_" in c]
    df_mean[f"{distance_name_in_csv}_mean"] = df_mean[distance_measure_columns].mean(axis=1)

    minima_eta, minima_gamma = [], []
    for col in distance_measure_columns:
        min_idx = df_mean[col].idxmin()
        minima_eta.append(df_mean.loc[min_idx, "eta"])
        minima_gamma.append(df_mean.loc[min_idx, "gamma"])

    all_minima[network_name] = {
        'eta': minima_eta, 'gamma': minima_gamma,
        'df_mean': df_mean,
    }
    print(f"  Found {len(minima_eta)} minima")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    pivot = df_mean.pivot(index="gamma", columns="eta", values=f"{distance_name_in_csv}_mean")
    im = ax.imshow(
        pivot,
        extent=(df_mean["eta"].min(), df_mean["eta"].max(),
                df_mean["gamma"].min(), df_mean["gamma"].max()),
        origin='lower', aspect='auto', cmap="Grays",
    )
    ax.set_title(LABEL_MAP[network_name])

    # Determine grid position for axis-label / tick decisions
    col_idx = mosaic[pos_idx // 3][pos_idx % 3]   # same as network_name
    row = pos_idx // 3
    col = pos_idx % 3

    if col == 0:
        ax.set_ylabel(PROPERTY_NAMES["gamma"])
        ax.set_yticks([-0.1, 0, 1])
    else:
        ax.set_yticks([])

    if row == 2:                         # bottom row (only Routing here)
        ax.set_xlabel(PROPERTY_NAMES["eta"])
        ax.set_xticks([-8, 0, 3])
    else:
        ax.set_xticks([])

    # # Colorbar on right column panels only
    # if col == 2:
    #     plt.colorbar(im, ax=ax, label=f"Mean {distance_name_in_csv.capitalize()}")
    # else:
    #     plt.colorbar(im, ax=ax)


    # Replace every colorbar call inside the heatmap loop with this pattern:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if col == 2:
        plt.colorbar(im, cax=cax, label=f"Mean {distance_name_in_csv.capitalize()}")
    else:
        plt.colorbar(im, cax=cax)
        
    # ── Scatter minima ────────────────────────────────────────────────────────
    ax.scatter(minima_eta, minima_gamma,
               color=COLOR_SCHEME[network_name],
               s=20, label='Minima', edgecolor='black')

    # Enforce shared limits
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)





# ── 2. Boxplot panel ──────────────────────────────────────────────────────────
ax_box = axs['boxplot']

boxplot_data, labels, colors_list = [], [], []

for network_name in all_minima:
    df_mean = all_minima[network_name]['df_mean']
    if f'{distance_name_in_csv}_mean' not in df_mean.columns:
        dcols = [c for c in df_mean.columns if f"{distance_name_in_csv}_subject_" in c]
        df_mean[f'{distance_name_in_csv}_mean'] = df_mean[dcols].mean(axis=1)

    lowest_100 = df_mean[f'{distance_name_in_csv}_mean'].nsmallest(100).values
    boxplot_data.append(lowest_100)
    labels.append(LABEL_MAP[network_name])
    colors_list.append(COLOR_SCHEME[network_name])

# Scatter jitter
for i, data in enumerate(boxplot_data):
    x = np.random.normal(i + 1, 0.1, size=len(data))
    ax_box.scatter(x, data, color=colors_list[i],
                   s=5, alpha=0.6, edgecolor='black', linewidth=0)

bp = ax_box.boxplot(
    boxplot_data, labels=labels,
    patch_artist=True, showfliers=False,
    medianprops=dict(color='black', linewidth=1),
)
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax_box.set_ylabel(f'{distance_name_in_csv.capitalize()} Mean')
# ax_box.set_title(f'100 Lowest {distance_name_in_csv.capitalize()} Values') #  per Network Type')
ax_box.set_title(f'100 Lowest Values') #  per Network Type')
ax_box.grid(axis='y', alpha=0.3, linestyle='--')

new_labels = ["MaMI", "Dev.", "Diff.", "Prop.", "Rout."]
ax_box.set_xticklabels(new_labels, ha='center') # rotation=45,
# Stagger every other x-tick label slightly lower to avoid overlap
# for j, tick in enumerate(ax_box.get_xticklabels()):
#     if j % 2 == 0:
#         tick.set_y(tick.get_position()[1] - 0.05)
ax_box.vlines(2.5, *ax_box.get_ylim(), colors='gray', linestyles='dashed', alpha=0.5)

# After building the boxplot, balance its width to match heatmap columns:
# divider = make_axes_locatable(ax_box)
# cax_dummy = divider.append_axes("right", size="5%", pad=0.05)
# cax_dummy.set_visible(False) 

# ── Final touches ─────────────────────────────────────────────────────────────
plt.suptitle(distance_name_in_csv.capitalize())
plt.tight_layout()
plt.savefig(output_path / f"{distance_name_in_csv}_combined_mosaic_only_developing.pdf", bbox_inches="tight")
print(output_path / f"{distance_name_in_csv}_combined_mosaic_only_developing.pdf")
plt.show()


# In[13]:


# ── Combined figure using subplot_mosaic ──────────────────────────────────────

network_names = list(csv_files.keys())   # 7 entries expected

# Build mosaic layout: 7 heatmap panels + 1 wide boxplot (bottom-right 2 cells)
mosaic = [
    [network_names[0], network_names[1], network_names[2]],
    [network_names[3], network_names[4], network_names[5]],
    [network_names[6], 'boxplot',        'boxplot'       ],
]

fig, axs = plt.subplot_mosaic(
    mosaic,
    figsize=viz.cm_to_inch((18, 18)),
    dpi=100,
)

# ── shared axis limits (applied manually since mosaic doesn't do sharex/sharey) ──
X_MIN, X_MAX = None, None   # will be set from first dataset
Y_MIN, Y_MAX = None, None

# ── 1. Heatmap panels ─────────────────────────────────────────────────────────
all_minima = {}

for pos_idx, (network_name, get_csv_path) in enumerate(csv_files.items()):
    print(f"Processing {network_name}...")
    ax = axs[network_name]

    # ── Load & deduplicate columns ────────────────────────────────────────────
    df = pd.read_csv(get_csv_path)
    all_keys = set(df.keys())
    keys = set([k.replace("_x","").replace("_y","").replace("_z","") for k in all_keys])
    keys -= {'eta', 'filename', 'id', 'gamma', 'network_index'}
    first_versions = []
    for key in keys:
        for suffix in ["", "_x", "_y", "_z"]:
            candidate = key + suffix
            if candidate in all_keys:
                first_versions.append(candidate)
                break
    df = df[first_versions + ['eta', 'id', 'gamma', 'network_index']]
    print(f"  {len(df.keys())} columns retained")

    # df_mean = df # 
    df_mean_2 = df.groupby(["eta", "gamma"]).mean().reset_index()

    # Update shared axis limits
    if X_MIN is None:
        X_MIN, X_MAX = df_mean_2["eta"].min(),   df_mean_2["eta"].max()
        Y_MIN, Y_MAX = df_mean_2["gamma"].min(), df_mean_2["gamma"].max()

    # ── Per-subject minima ────────────────────────────────────────────────────
    distance_measure_columns = [c for c in df_mean_2.columns if f"{distance_name_in_csv}_subject_" in c]
    df_mean_2[f"{distance_name_in_csv}_mean"] = df_mean_2[distance_measure_columns].mean(axis=1)

    minima_eta, minima_gamma, minima_distances = [], [], []
    for col in distance_measure_columns:
        min_idx = df[col].idxmin()
        minima_eta.append(df.loc[min_idx, "eta"])
        minima_gamma.append(df.loc[min_idx, "gamma"])
        minima_distances.append(df.loc[min_idx, col])

    all_minima[network_name] = {
        'eta': minima_eta, 'gamma': minima_gamma,
        'df_mean': df_mean_2, # .loc[min_idx, "eta"],
        'minima_distances': minima_distances
    }
    print(f"  Found {len(minima_eta)} minima")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    pivot = df_mean_2.pivot(index="gamma", columns="eta", values=f"{distance_name_in_csv}_mean")
    im = ax.imshow(
        pivot,
        extent=(df_mean_2["eta"].min(), df_mean_2["eta"].max(),
                df_mean_2["gamma"].min(), df_mean_2["gamma"].max()),
        origin='lower', aspect='auto', cmap="Grays",
    )
    ax.set_title(LABEL_MAP[network_name])

    # Determine grid position for axis-label / tick decisions
    col_idx = mosaic[pos_idx // 3][pos_idx % 3]   # same as network_name
    row = pos_idx // 3
    col = pos_idx % 3

    if col == 0:
        ax.set_ylabel(PROPERTY_NAMES["gamma"])
        ax.set_yticks([-0.1, 0, 1])
    else:
        ax.set_yticks([])

    if row == 2:                         # bottom row (only Routing here)
        ax.set_xlabel(PROPERTY_NAMES["eta"])
        ax.set_xticks([-8, 0, 3])
    else:
        ax.set_xticks([])

    # Colorbar on right column panels only
    if col == 2:
        plt.colorbar(im, ax=ax, label=f"Mean {distance_name_in_csv.capitalize()}")
    else:
        plt.colorbar(im, ax=ax)

    # ── Scatter minima ────────────────────────────────────────────────────────
    ax.scatter(minima_eta, minima_gamma,
               color=COLOR_SCHEME[network_name],
               s=20, label='Minima', edgecolor='black')

    # Enforce shared limits
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)





# ── 2. Boxplot panel ──────────────────────────────────────────────────────────
ax_box = axs['boxplot']

boxplot_data, labels, colors_list = [], [], []

for network_name in all_minima:
    df_mean = all_minima[network_name]['df_mean']
    if f'{distance_name_in_csv}_mean' not in df_mean.columns:
        dcols = [c for c in df_mean.columns if f"{distance_name_in_csv}_subject_" in c]
        df_mean[f'{distance_name_in_csv}_mean'] = df_mean[dcols].mean(axis=1)

    lowest_100 = df_mean[f'{distance_name_in_csv}_mean'].nsmallest(100).values
    boxplot_data.append(lowest_100)
    labels.append(LABEL_MAP[network_name])
    colors_list.append(COLOR_SCHEME[network_name])

# Scatter jitter
for i, data in enumerate(boxplot_data):
    x = np.random.normal(i + 1, 0.1, size=len(data))
    ax_box.scatter(x, data, color=colors_list[i],
                   s=5, alpha=0.6, edgecolor='black', linewidth=0)

bp = ax_box.boxplot(
    boxplot_data, labels=labels,
    patch_artist=True, showfliers=False,
    medianprops=dict(color='black', linewidth=1),
)
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax_box.set_ylabel(f'{distance_name_in_csv.capitalize()} Mean')
ax_box.set_title(f'100 Lowest {distance_name_in_csv.capitalize()} Values per Network Type')
ax_box.grid(axis='y', alpha=0.3, linestyle='--')

# Stagger every other x-tick label slightly lower to avoid overlap
for j, tick in enumerate(ax_box.get_xticklabels()):
    if j % 2 == 0:
        tick.set_y(tick.get_position()[1] - 0.05)

# ── Final touches ─────────────────────────────────────────────────────────────
plt.suptitle(distance_name_in_csv.capitalize())
plt.tight_layout()
plt.savefig(output_path / f"{distance_name_in_csv}_combined_mosaic_not_meaned.pdf", bbox_inches="tight")
print(output_path / f"{distance_name_in_csv}_combined_mosaic_not_meaned.pdf")
plt.show()


# In[ ]:


# all_minima = {}

# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=viz.cm_to_inch((18,18)), sharex=True, sharey=True, dpi=100)
# axs = axs.flatten()

# for i, (network_name, get_csv_path) in enumerate(csv_files.items()):
#     print(f"Processing {network_name}...")
#     df = pd.read_csv(get_csv_path)

#     keys = set([k.replace("_x", "").replace("_y", "").replace("_z", "") for k in df.keys()])
#     keys = keys - set(['eta', 'filename', 'id', 'gamma', 'network_index'])
#     unique_keys = list(keys)

#     all_keys = set(df.keys())
#     first_versions = []
#     for key in unique_keys:
#         for suffix in ["", "_x", "_y", "_z"]:
#             candidate = key + suffix
#             if candidate in all_keys:
#                 first_versions.append(candidate)
#                 break

#     df = df[first_versions + ['eta', 'id', 'gamma', 'network_index']]
#     print(len(df.keys()))

#     df_mean = df.groupby(["eta", "gamma"]).mean().reset_index()

#     distance_measure_columns = [col for col in df_mean.columns if f"{distance_name_in_csv}_subject_" in col]
#     df_mean[f"{distance_name_in_csv}_mean"] = df_mean[distance_measure_columns].mean(axis=1)

#     # --- CHANGED: Select top 100 eta/gamma combinations by mean distance ---
#     top100 = df_mean.nsmallest(100, f"{distance_name_in_csv}_mean")
#     best_eta   = top100["eta"].tolist()
#     best_gamma = top100["gamma"].tolist()
#     best_scores = top100[f"{distance_name_in_csv}_mean"].tolist()
#     # -----------------------------------------------------------------------

#     all_minima[network_name] = {
#         'eta': best_eta,
#         'gamma': best_gamma,
#         'df_mean': df_mean,
#         'best_scores': best_scores
#     }
#     print(f"  Found top {len(best_eta)} combinations")

#     cmap = "Grays"
#     background = axs[i].imshow(
#         df_mean.pivot(index="gamma", columns="eta", values=f"{distance_name_in_csv}_mean"),
#         extent=(df_mean["eta"].min(), df_mean["eta"].max(),
#                 df_mean["gamma"].min(), df_mean["gamma"].max()),
#         origin='lower', aspect='auto', cmap=cmap
#     )
#     axs[i].set_title(network_name.capitalize())
#     axs[i].set_xlabel("Eta")
#     axs[i].set_ylabel("Gamma")
#     plt.colorbar(background, ax=axs[i], label=f"Mean {distance_name_in_csv.capitalize()}")

#     # --- CHANGED: Plot top 100, coloured by their mean distance score ---
#     sc = axs[i].scatter(
#         best_eta, best_gamma,
#         c=best_scores,          # colour encodes how good each point is
#         cmap="RdYlGn_r",        # green = best (lowest), red = worst of the top-100
#         s=20, edgecolor='black',
#         label='Top 100', zorder=5
#     )
#     # plt.colorbar(sc, ax=axs[i], label="Score (top 100)")
#     # -------------------------------------------------------------------

# if len(csv_files) < len(axs):
#     for j in range(len(csv_files), len(axs)):
#         fig.delaxes(axs[j])

# plt.suptitle(f"{distance_measure.capitalize()}")
# plt.tight_layout()


# In[ ]:


# # ── Combined figure using subplot_mosaic ──────────────────────────────────────

# network_names = list(csv_files.keys())   # 7 entries expected

# mosaic = [
#     [network_names[0], network_names[1], network_names[2]],
#     [network_names[3], network_names[4], network_names[5]],
#     [network_names[6], 'boxplot',        'boxplot'       ],
# ]

# fig, axs = plt.subplot_mosaic(
#     mosaic,
#     figsize=viz.cm_to_inch((18, 18)),
#     dpi=100,
# )

# X_MIN, X_MAX = None, None
# Y_MIN, Y_MAX = None, None

# TOP_NUMBER = 10 

# # ── 1. Heatmap panels ─────────────────────────────────────────────────────────
# all_minima = {}

# for pos_idx, (network_name, get_csv_path) in enumerate(csv_files.items()):
#     print(f"Processing {network_name}...")
#     ax = axs[network_name]
#     row, col = pos_idx // 3, pos_idx % 3

#     # ── Load & deduplicate columns ────────────────────────────────────────────
#     df = pd.read_csv(get_csv_path)
#     all_keys = set(df.keys())
#     keys = set([k.replace("_x","").replace("_y","").replace("_z","") for k in all_keys])
#     keys -= {'eta', 'filename', 'id', 'gamma', 'network_index'}
#     first_versions = []
#     for key in keys:
#         for suffix in ["", "_x", "_y", "_z"]:
#             candidate = key + suffix
#             if candidate in all_keys:
#                 first_versions.append(candidate)
#                 break
#     df = df[first_versions + ['eta', 'id', 'gamma', 'network_index']]
#     print(f"  {len(df.keys())} columns retained")

#     df_mean = df.groupby(["eta", "gamma"]).mean().reset_index()

#     if X_MIN is None:
#         X_MIN, X_MAX = df_mean["eta"].min(),   df_mean["eta"].max()
#         Y_MIN, Y_MAX = df_mean["gamma"].min(), df_mean["gamma"].max()

#     # ── Mean distance column ──────────────────────────────────────────────────
#     distance_measure_columns = [c for c in df_mean.columns if f"{distance_name_in_csv}_subject_" in c]
#     df_mean[f"{distance_name_in_csv}_mean"] = df_mean[distance_measure_columns].mean(axis=1)

#     # ── CHANGED: top-100 eta/gamma combinations by mean distance ─────────────
#     top100       = df_mean.nsmallest(TOP_NUMBER, f"{distance_name_in_csv}_mean")
#     best_eta     = top100["eta"].tolist()
#     best_gamma   = top100["gamma"].tolist()
#     best_scores  = top100[f"{distance_name_in_csv}_mean"].tolist()
#     # ─────────────────────────────────────────────────────────────────────────

#     all_minima[network_name] = {
#         'eta': best_eta,
#         'gamma': best_gamma,
#         'df_mean': df_mean,
#         'best_scores': best_scores,
#     }
#     print(f"  Found top {len(best_eta)} combinations")

#     # ── Heatmap ───────────────────────────────────────────────────────────────
#     pivot = df_mean.pivot(index="gamma", columns="eta", values=f"{distance_name_in_csv}_mean")
#     im = ax.imshow(
#         pivot,
#         extent=(df_mean["eta"].min(), df_mean["eta"].max(),
#                 df_mean["gamma"].min(), df_mean["gamma"].max()),
#         origin='lower', aspect='auto', cmap="Grays",
#     )
#     ax.set_title(LABEL_MAP[network_name])

#     if col == 0:
#         ax.set_ylabel(PROPERTY_NAMES["gamma"])
#         ax.set_yticks([-0.1, 0, 1])
#     else:
#         ax.set_yticks([])

#     if row == 2:
#         ax.set_xlabel(PROPERTY_NAMES["eta"])
#         ax.set_xticks([-8, 0, 3])
#     else:
#         ax.set_xticks([])

#     if col == 2:
#         plt.colorbar(im, ax=ax, label=f"Mean {distance_name_in_csv.capitalize()}")
#     else:
#         plt.colorbar(im, ax=ax)

#     # ── CHANGED: scatter top-100, coloured by score ───────────────────────────
#     sc = ax.scatter(
#         best_eta, best_gamma,
#         c=best_scores,          # colour encodes distance value
#         cmap="RdYlGn_r",        # green = lowest (best), red = worst of top-100
#         s=20, edgecolor='black',
#         label=f'Top {TOP_NUMBER}', zorder=5,
#     )
#     # ─────────────────────────────────────────────────────────────────────────

#     ax.set_xlim(X_MIN, X_MAX)
#     ax.set_ylim(Y_MIN, Y_MAX)


# # ── 2. Boxplot panel ──────────────────────────────────────────────────────────
# ax_box = axs['boxplot']

# boxplot_data, labels, colors_list = [], [], []

# for network_name in all_minima:
#     # best_scores already contains the 100 lowest mean-distance values
#     lowest_100 = np.array(all_minima[network_name]['best_scores'])
#     boxplot_data.append(lowest_100)
#     labels.append(LABEL_MAP[network_name])
#     colors_list.append(COLOR_SCHEME[network_name])

# for i, data in enumerate(boxplot_data):
#     x = np.random.normal(i + 1, 0.1, size=len(data))
#     ax_box.scatter(x, data, color=colors_list[i],
#                    s=5, alpha=0.6, edgecolor='black', linewidth=0)

# bp = ax_box.boxplot(
#     boxplot_data, labels=labels,
#     patch_artist=True, showfliers=False,
#     medianprops=dict(color='black', linewidth=1),
# )
# for patch, color in zip(bp['boxes'], colors_list):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.6)

# ax_box.set_ylabel(f'{distance_name_in_csv.capitalize()} Mean')
# ax_box.set_title(f'{TOP_NUMBER} Lowest {distance_name_in_csv.capitalize()} Values per Network Type')
# ax_box.grid(axis='y', alpha=0.3, linestyle='--')

# for j, tick in enumerate(ax_box.get_xticklabels()):
#     if j % 2 == 0:
#         tick.set_y(tick.get_position()[1] - 0.05)

# # ── Final touches ─────────────────────────────────────────────────────────────
# plt.suptitle(distance_name_in_csv.capitalize())
# plt.tight_layout()
# plt.savefig(output_path / f"{distance_name_in_csv}_combined_mosaic.pdf", bbox_inches="tight")
# print(output_path / f"{distance_name_in_csv}_combined_mosaic.pdf")
# plt.show()

