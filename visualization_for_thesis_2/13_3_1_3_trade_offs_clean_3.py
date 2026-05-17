#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from vizman import viz
import os
import re 

import pickle

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from config import COLORS, COLOR_SCHEME, LABEL_MAP, gray_cmap, bone_white, half_black, emp_dataset_and_experiment_pairs, PROPERTY_NAMES


# In[103]:


# Generate output folder
output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/trade_off_analysis")
output_folder.mkdir(exist_ok=True)

# Load data 
with open("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis/all_datasets_precise_categories.pkl", "rb") as f:  # output_folder / "all_datasets_precise_categories.pkl", "rb") as f: # all_datasets_filtered.pkl", "rb") as f:
    dict_with_all_datasets = pickle.load(f)
    
# Folder now changed for saving 
output_folder = output_folder / "03_trade_offs" # scatter_taxonomy"
output_folder.mkdir(exist_ok=True)


# In[104]:


# Eta and gamma pairs 

# save df_gnm_eta_and_gamma as pkl
with open("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis/hcp_schaefer_100_dataset_gnm_eta_and_gamma.pkl", "rb") as f:
    df_eta_and_gamma = pickle.load(f)

df_eta_and_gamma


# # Trade-offs etc 

# In[105]:


dict_with_all_datasets


# In[106]:


# df_gnm = df_eta_and_gamma.merge(dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"], left_index=True, right_index=True) # , on=["eta", "gamma", "id"], how="left", suffixes=("", f"_fundamental"))
df_gnm = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]


# In[107]:


df_gnm_g =df_gnm # .groupby(["eta", "gamma"]).mean().reset_index()
cols_of_interest = [col for col in df_gnm_g.columns if col not in ["eta", "gamma"]]
df_gnm_g


# In[108]:


datasets_to_look_at = [
                'hcp_schaefer_100_dataset_gnm', 
                # 'hcp_schaefer_100_dataset',
                # 'kaysons_generated_networks_topology', 
                'suarez_MaMI_dataset', 
                # 'lexis_data_young', 
                # 'lexis_data_aging', 
                'lexis_data_developing', 
                
                'kaysons_generated_networks_diffusion', 
                'kaysons_generated_networks_propagation', 
                'kaysons_generated_networks_routing', 
                
                
                "ring_lattice_networks",
                "erdos_renyi_networks",
]


# In[109]:


x_name = "proportion_long_range_connections_0.3956"

for idx, col in enumerate(cols_of_interest):
    fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

    plt.scatter(df_gnm_g[x_name], df_gnm_g[col], 
               color="gray", linewidth=0, s=5, alpha=0.2) 

    if len(col) > 20:
        col_for_title = re.split(r'[,,_]+', col)
        len_col = len(col_for_title)
        col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

    if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
        col = "min_value"
        col_for_title = "Distance Measure"

    for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

        # Skip the gnm dataset, as it will now have two eta and gamma columns. 
        if dataset == "hcp_schaefer_100_dataset_gnm":
            continue

        df_merged = dict_with_all_datasets[dataset]

        if col in df_merged.columns:
            plt.scatter(df_merged[x_name], # df_merged["eta"], df_merged['gamma'],
                       df_merged[col],
                    #    c = df_merged[col],
                       edgecolor="black",
                       color=COLOR_SCHEME[dataset],
                       linewidth=0.25, s=5, label=LABEL_MAP[dataset]
            ) #  alpha=0.7)

    plt.xlabel(PROPERTY_NAMES[x_name]) # Percentage of\nLR connections (> 50 % length)")
    # plt.ylabel(col.replace("_", " ").capitalize())
    # if the title is too long (define this), then split it into two lines at the last underscore
    plt.title(col) # , fontsize=6)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(output_folder / f"fig_{idx}_{col}_scatter.png", dpi=200)
    if idx == 0:
        plt.show()
    else: 
        plt.close()
        
    break # just do one for now

print("Save path: ", output_folder)
print("remove break to save all of the scatter plots")


# In[110]:


x_name = "proportion_long_range_connections_0.3956"
col = "wiring_cost"
fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])


for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_datasets[dataset]

    if col in df_merged.columns:
        plt.scatter(df_merged[x_name], # df_merged["eta"], df_merged['gamma'],
                    df_merged[col],
                #    c = df_merged[col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    linewidth=0.25, s=5, label=LABEL_MAP[dataset]
        ) #  alpha=0.7)

plt.xlabel(PROPERTY_NAMES[x_name]) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.title(col) # , fontsize=6)
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig(output_folder / f"fig_{idx}_{col}_scatter_w_o_gnm.png", dpi=200)
plt.show()

print("Save path: ", output_folder)


# In[111]:


for k in dict_with_all_datasets[dataset].keys():
    print(k)


# In[112]:


from utils import get_combined_colors
with open(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis/all_datasets_precise_categories.pkl",
    "rb",
) as f:
    dict_with_all_datasets = pickle.load(f)

output_folder = Path(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/trade_off_analysis/03_trade_offs"
)
output_folder.mkdir(parents=True, exist_ok=True)

INCLUDE_LEXIS = True

datasets_to_plot = [
    "hcp_schaefer_100_dataset_gnm",
    "suarez_MaMI_dataset",
    "lexis_data_developing",
    "kaysons_generated_networks_diffusion",
    "kaysons_generated_networks_propagation",
    "kaysons_generated_networks_routing",
    "ring_lattice_networks",
    "erdos_renyi_networks",
]

# all_datasets_precise_categories.pkl dropped several columns (mc_input_scaling,
# proportion_long_range_connections_0.3956, repertoire, eta, gamma) from the GNM entry.
# Replace it with the full GNM from all_datasets.pkl which has everything.
with open(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis/all_datasets.pkl",
    "rb",
) as _f:
    _dict_full = pickle.load(_f)
dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"] = _dict_full["hcp_schaefer_100_dataset_gnm"]

df_gnm = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]
combined_gnm_colors = get_combined_colors(df_gnm)


# In[113]:


# from utils import get_combined_colors

# combined_gnm_colors = get_combined_colors(df_gnm_g)
# len(combined_gnm_colors)


# In[114]:


def plot_simple_trade_off(x_col, y_col): 
    fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

   
    plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
                color=combined_gnm_colors, # "gray", 
                linewidth=0, s=5, alpha=0.2) # 0.2 


    if y_col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
        y_col = "min_value"
        col_for_title = "Distance Measure"

    for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

        # Skip the gnm dataset, as it will now have two eta and gamma columns. 
        if dataset == "hcp_schaefer_100_dataset_gnm":
            continue

        df_merged = dict_with_all_datasets[dataset]

        if y_col in df_merged.columns:
            plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                        df_merged[y_col],
                    #    c = df_merged[col],
                        edgecolor="black",
                        color=COLOR_SCHEME[dataset],
                        label=dataset,
                        linewidth=0.25, 
                        s=10
                        )

    # plt.xscale("log")
    plt.xlabel(PROPERTY_NAMES[x_col]) # Percentage of\nLR connections (> 50 % length)")
    # plt.ylabel(col.replace("_", " ").capitalize())
    # if the title is too long (define this), then split it into two lines at the last underscore
    plt.ylabel(PROPERTY_NAMES[y_col]) # , fontsize=6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

    plt.tight_layout()
    plt.show() 



# In[115]:


x_col = "global_efficiency" # computational_capacity_memory_capacity_total"
y_col = "modularity" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[ ]:





# In[116]:


y_col = "global_efficiency" # computational_capacity_memory_capacity_total"
x_col = "wiring_cost" # modularity" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[117]:


y_col = "global_efficiency" # computational_capacity_memory_capacity_total"
x_col = "proportion_long_range_connections_0.3956" # modularity" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[118]:


x_col = "diffusion_efficiency"
y_col = "global_efficiency" # propagation_efficiency" # mc_mean"
plot_simple_trade_off(x_col, y_col)

x_col = "diffusion_efficiency"
y_col = "propagation_efficiency" # mc_mean"
plot_simple_trade_off(x_col, y_col)

x_col = "global_efficiency"
y_col = "propagation_efficiency" # mc_mean"
plot_simple_trade_off(x_col, y_col)


# In[119]:


x_col = "ipc_ipc_deg1_mean"
y_col = "ipc_ipc_deg2_mean"
plot_simple_trade_off(x_col, y_col)


# In[120]:


x_col = "ipc_ipc_deg1_mean"
y_col = "ipc_ipc_deg2_mean"

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color=combined_gnm_colors, # "gray", 
            linewidth=0, s=5, alpha=0.5) # 0.2 


if y_col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    y_col = "min_value"
    col_for_title = "Distance Measure"

for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_datasets[dataset]

    if y_col in df_merged.columns:
        plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                    df_merged[y_col],
                #    c = df_merged[col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    label=dataset,
                    linewidth=0.25, 
                    s=10
                    )

# plt.xscale("log")
plt.xlabel(PROPERTY_NAMES[x_col]) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.ylabel(PROPERTY_NAMES[y_col]) # , fontsize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
plt.xlim(left=8)
plt.ylim(bottom=15)
plt.tight_layout()
plt.show() 



# In[121]:


x_col = "mc_input_scaling_0_1_mc_mean"
y_col = "mc_nonlinear_input_scaling_0_1_mc_mean"
plot_simple_trade_off(x_col, y_col)


# In[122]:


y_col = "ollivier_ricci_curvature_orc_std" # synchronizability_eigenratio_eigenratio" # computational_capacity_nonlinear_capacity_total" # 
x_col = "repertoire_sweep_weighted_by_distances_size_critical" # "repertoire_sweep_weighted_by_distances_diversity_critical" # "nct_control_std" # computational_capacity_nonlinear_capacity_total"
x_col = "repertoire_sweep_weighted_by_distances_diversity_critical"
# y_col = "ipc_ipc_deg1_mean"
y_col = "mc_input_scaling_0_1_mc_mean"
# y_col = "mc_nonlinear_input_scaling_0_1_mc_mean"
x_col = "kuramoto_averaged_synchronization_r_std"
x_col = "repertoire_sweep_weighted_by_distances_T_critical"
# x_col = "repertoire_sweep_weighted_by_distances_size_critical"
fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    


plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color=combined_gnm_colors, # "gray", 
            linewidth=0, s=5, alpha=0.5) # 0.2 


if y_col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    y_col = "min_value"
    col_for_title = "Distance Measure"

for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_datasets[dataset]

    if y_col in df_merged.columns:
        plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                    df_merged[y_col],
                #    c = df_merged[col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    label=dataset,
                    linewidth=0.25, 
                    s=10, 
                    # alpha=0.2
                    )
        
    # for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

    #     # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    #     if dataset == "hcp_schaefer_100_dataset_gnm":
    #         continue

    #     df_merged = dict_with_all_datasets[dataset]

    #     if y_col in df_merged.columns:
    #         plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
    #                     df_merged[y_col],
    #                 #    c = df_merged[col],
    #                     edgecolor="black",
    #                     color="none", # COLOR_SCHEME[dataset],
    #                     # label=dataset,
    #                     linewidth=0.1, # 5, 
    #                     s=10, 
    #                     # alpha=0.2
    #                     )

# plt.xscale("log")
plt.xlabel(PROPERTY_NAMES[x_col]) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.ylabel(PROPERTY_NAMES[y_col]) # , fontsize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
# plt.xlim(left=8)
# plt.ylim(0,100)
# plt.ylim(bottom=8)
plt.tight_layout()

plt.xscale("log")

plt.show() 



# In[123]:


x_col = "mc_input_scaling_0_1_mc_mean"
y_col = "mc_nonlinear_input_scaling_0_1_mc_mean"

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    


plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color=combined_gnm_colors, # "gray", 
            linewidth=0, s=5, alpha=0.5) # 0.2 


if y_col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    y_col = "min_value"
    col_for_title = "Distance Measure"

for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_datasets[dataset]

    if y_col in df_merged.columns:
        plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                    df_merged[y_col],
                #    c = df_merged[col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    label=dataset,
                    linewidth=0.25, 
                    s=10
                    )

# plt.xscale("log")
plt.xlabel(PROPERTY_NAMES[x_col]) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.ylabel(PROPERTY_NAMES[y_col]) # , fontsize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
plt.xlim(left=8)
plt.ylim(bottom=8)
plt.tight_layout()
plt.show() 



# In[124]:


# x_col = "wiring_cost"
x_col = "richclub_n_edges" # "proportion_long_range_connections_0.3956"
y_col = "targeted_attack_robustness_rob_targeted_auc"
# plot_simple_trade_off(x_col, y_col)

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    


plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color=combined_gnm_colors, # "gray", 
            linewidth=0, s=5, alpha=0.5) # 0.2 


if y_col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    y_col = "min_value"
    col_for_title = "Distance Measure"

for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_datasets[dataset]

    if y_col in df_merged.columns:
        plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                    df_merged[y_col],
                #    c = df_merged[col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    label=dataset,
                    linewidth=0.25, 
                    s=1
                    )
    

# plt.xscale("log")
plt.xlabel(PROPERTY_NAMES[x_col]) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.ylabel(PROPERTY_NAMES[y_col]) # , fontsize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
plt.tight_layout()
plt.show() 



# In[125]:


x_col = "mc_input_scaling_0_1_mc_mean"
y_col = "community_synchronization_vulnerability_value"
plot_simple_trade_off(x_col, y_col)


# In[126]:


x_col = "mc_input_scaling_0_1_mc_mean"
y_col = "targeted_attack_robustness_rob_targeted_auc" # algebraic_connectivity_fiedler_value"
plot_simple_trade_off(x_col, y_col)


# In[127]:


x_col = "ollivier_ricci_curvature_orc_median" # ollivier_ricci_curvature_orc_frac_neg" # mc_input_scaling_0_1_mc_mean"
y_col = "omega" # algebraic_connectivity_fiedler_value"
plot_simple_trade_off(x_col, y_col)


# In[128]:


x_col = "kuramoto_averaged_synchronization_r_std" # ollivier_ricci_curvature_orc_frac_neg" # mc_input_scaling_0_1_mc_mean"
y_col = "repertoire_sweep_weighted_by_distances_diversity_critical" # algebraic_connectivity_fiedler_value"
# plot_simple_trade_off(x_col, y_col)

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    


plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color=combined_gnm_colors, # "gray", 
            linewidth=0, s=5, alpha=0.5) # 0.2 


if y_col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    y_col = "min_value"
    col_for_title = "Distance Measure"

df_total = pd.DataFrame()

for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_datasets[dataset]
    df_total = pd.concat([df_total, df_merged], ignore_index=True)
    

    if y_col in df_merged.columns:
        plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                    df_merged[y_col],
                #    c = df_merged[col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    label=dataset,
                    linewidth=0.25, 
                    s=1
                    )
    
# how much do x and y correlate across the gnm space?
from scipy.stats import spearmanr
corr, _ = spearmanr(df_total[x_col], df_total[y_col])
print(f"Spearman corr: {corr} between {x_col} and {y_col}")


# plt.xscale("log")
plt.xlabel(PROPERTY_NAMES[x_col]) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.ylabel(PROPERTY_NAMES[y_col]) # , fontsize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
plt.tight_layout()
plt.show() 



# In[129]:


x_col = "ollivier_ricci_curvature_orc_median" # ollivier_ricci_curvature_orc_frac_neg" # mc_input_scaling_0_1_mc_mean"
y_col = "repertoire_sweep_weighted_by_distances_diversity_critical" # algebraic_connectivity_fiedler_value"
plot_simple_trade_off(x_col, y_col)


# In[130]:


x_col = "ollivier_ricci_curvature_orc_min" # ollivier_ricci_curvature_orc_frac_neg" # mc_input_scaling_0_1_mc_mean"
y_col = "omega" # algebraic_connectivity_fiedler_value"
plot_simple_trade_off(x_col, y_col)


# In[131]:


for k in dict_with_all_datasets[dataset].keys():
    print(k)


# In[132]:


# THE GOOD STUFF 

x_col = "wiring_cost"
y_col = "global_efficiency"
plot_simple_trade_off(x_col, y_col)


# THE GOOD STUFF 

x_col = "proportion_long_range_connections_0.3956"
y_col = "global_efficiency"
plot_simple_trade_off(x_col, y_col)


# In[133]:


x_col = "modularity"
y_col = "global_efficiency"
plot_simple_trade_off(x_col, y_col)


# In[134]:


x_col = "richclub_n_edges"
y_col = "targeted_attack_robustness_rob_targeted_auc"
plot_simple_trade_off(x_col, y_col)


# In[135]:


x_col = "spectral_radius" # structural_complexity"
y_col = "algebraic_connectivity_fiedler_value" # global_efficiency" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[136]:


x_col = "spectral_radius" # structural_complexity"
y_col = "algebraic_connectivity_fiedler_value" # global_efficiency" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[137]:


x_col = "wiring_cost"
y_col = "structural_complexity" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[138]:


x_col = "avg_clustering"
y_col = "global_efficiency" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[139]:


x_col = "richclub_n_edges"
y_col = "targeted_attack_robustness_rob_targeted_auc" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[140]:


x_col = "char_path_length"
y_col = "avg_clustering" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[141]:


x_col = "diffusion_efficiency"
y_col = "global_efficiency" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[142]:


x_col = "char_path_length"
y_col = "avg_clustering" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[143]:


# patankar_2020_pathdependent
x_col = "modularity" # computational_capacity_memory_capacity_total"
y_col = "nct_control_std" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[144]:


# patankar_2020_pathdependent
x_col = "targeted_attack_robustness_rob_targeted_auc" # computational_capacity_memory_capacity_total"
x_col = "algebraic_connectivity_fiedler_value"
y_col = "global_efficiency" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[145]:


# patankar_2020_pathdependent
# x_col = "targeted_attack_robustness_rob_targeted_auc" # computational_capacity_memory_capacity_total"
# x_col = "algebraic_connectivity_fiedler_value"
x_col = "targeted_attack_robustness_rob_random_auc"
y_col = "global_efficiency" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[ ]:





# In[146]:


# patankar_2020_pathdependent
# x_col = "targeted_attack_robustness_rob_targeted_auc" # computational_capacity_memory_capacity_total"
# x_col = "algebraic_connectivity_fiedler_value"
x_col = "propagation_efficiency" # ipc_ipc_deg2_mean" # mc_input_scaling_0_1_mc_mean" # targeted_attack_robustness_rob_random_auc"
y_col = "mc_input_scaling_0_1_mc_mean" # ipc_ipc_deg1_mean" # global_efficiency" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[147]:


x_col = "propagation_efficiency" # mc_input_scaling_0_1_mc_mean"
x_col = "diffusion_efficiency" # global_efficiency"
y_col = "mc_input_scaling_0_1_mc_mean" # repertoire_sweep_weighted_by_distances_diversity_critical" # mc_nonlinear_input_scaling_0_1_mc_mean"
# x_col = "proportion_long_range_connections_0.3956" #
# x_col = "wiring_cost"
# y_col = "targeted_attack_robustness_rob_random_auc"
# y_col = "algebraic_connectivity_fiedler_value"
# y_col = "targeted_attack_robustness_rob_targeted_auc"
# y_col = "global_efficiency" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[148]:


# x_col = "propagation_efficiency" # mc_input_scaling_0_1_mc_mean"
y_col = "nct_control_avg" # nct_control_std" # global_efficiency"
# y_col = "mc_input_scaling_0_1_mc_mean" # repertoire_sweep_weighted_by_distances_diversity_critical" # mc_nonlinear_input_scaling_0_1_mc_mean"
x_col = "proportion_long_range_connections_0.3956" #
# x_col = "wiring_cost"
# y_col = "targeted_attack_robustness_rob_random_auc"
# y_col = "algebraic_connectivity_fiedler_value"
# y_col = "targeted_attack_robustness_rob_targeted_auc"
# y_col = "global_efficiency" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[149]:


x_col = "nct_control_std" # nct_control_avg" # global_efficiency" # modularity"
y_col = "algebraic_connectivity_fiedler_value"
y_col = "targeted_attack_robustness_rob_targeted_auc"
y_col = "targeted_attack_robustness_rob_random_auc"
# y_col = "global_efficiency" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[150]:


x_col = "mc_input_scaling_0_1_mc_mean"
y_col = "mc_nonlinear_input_scaling_0_1_mc_mean"
# x_col = "proportion_long_range_connections_0.3956" #
# x_col = "wiring_cost"
y_col = "targeted_attack_robustness_rob_random_auc"
# y_col = "algebraic_connectivity_fiedler_value"
# y_col = "targeted_attack_robustness_rob_targeted_auc"
# y_col = "global_efficiency" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[151]:


# y_col = "targeted_attack_robustness_rob_targeted_auc" # computational_capacity_memory_capacity_total"
y_col = "algebraic_connectivity_fiedler_value"
x_col = "proportion_long_range_connections_0.3956" #
x_col = "wiring_cost"
# y_col = "targeted_attack_robustness_rob_random_auc"
# y_col = "global_efficiency" # computational_capacity_nonlinear_capacity_total"
plot_simple_trade_off(x_col, y_col)


# In[152]:


x_col = "richclub_n_edges"
y_col = "targeted_attack_robustness_rob_random_auc"
y_col = "algebraic_connectivity_fiedler_value_norm"
# y_col = "targeted_attack_robustness_rob_targeted_auc" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[153]:


x_col = "synchronizability_eigenratio_eigenratio"
y_col = "targeted_attack_robustness_rob_random_auc"
y_col = "algebraic_connectivity_fiedler_value_norm"
y_col = "repertoire_sweep_weighted_by_distances_diversity_critical"
# y_col = "targeted_attack_robustness_rob_targeted_auc" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[ ]:





# In[154]:


x_col = "richclub_n_edges"
y_col = "targeted_attack_robustness_rob_random_auc"
y_col = "algebraic_connectivity_fiedler_value"
# y_col = "targeted_attack_robustness_rob_targeted_auc" # spectral_radius" # repertoire_sweep_weighted_by_distances_diversity_critical"
plot_simple_trade_off(x_col, y_col)


# In[155]:


# fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

# x_col = "modularity" # computational_capacity_memory_capacity_total"
# y_col = "algebraic_connectivity_fiedler_value" # computational_capacity_nonlinear_capacity_total"
# plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
#             color="gray", linewidth=0, s=5, alpha=0.2) 

# if len(col) > 20:
#     col_for_title = re.split(r'[,,_]+', y_col)
#     len_col = len(col_for_title)
#     col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

# if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
#     col = "min_value"
#     col_for_title = "Distance Measure"

# for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

#     # Skip the gnm dataset, as it will now have two eta and gamma columns. 
#     if dataset == "hcp_schaefer_100_dataset_gnm":
#         continue

#     df_merged = dict_with_all_datasets[dataset]

#     if y_col in df_merged.columns:
#         plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
#                     df_merged[y_col],
#                 #    c = df_merged[col],
#                     edgecolor="black",
#                     color=COLOR_SCHEME[dataset],
#                     label=dataset,
#                     linewidth=0.25, 
#                     s=10
#                     )

# # plt.xscale("log")
# plt.xlabel(x_col) # Percentage of\nLR connections (> 50 % length)")
# # plt.ylabel(col.replace("_", " ").capitalize())
# # if the title is too long (define this), then split it into two lines at the last underscore
# plt.ylabel(col_for_title) # , fontsize=6)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

# plt.tight_layout()


# In[156]:


# fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

# x_col = "char_path_length" # modularity" # computational_capacity_memory_capacity_total"
# y_col = "avg_clustering" # algebraic_connectivity_fiedler_value" # computational_capacity_nonlinear_capacity_total"
# plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
#             color="gray", linewidth=0, s=5, alpha=0.2) 

# if len(col) > 20:
#     col_for_title = re.split(r'[,,_]+', y_col)
#     len_col = len(col_for_title)
#     col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

# if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
#     col = "min_value"
#     col_for_title = "Distance Measure"

# for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

#     # Skip the gnm dataset, as it will now have two eta and gamma columns. 
#     if dataset == "hcp_schaefer_100_dataset_gnm":
#         continue

#     df_merged = dict_with_all_datasets[dataset]

#     if y_col in df_merged.columns:
#         plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
#                     df_merged[y_col],
#                 #    c = df_merged[col],
#                     edgecolor="black", 
#                     color=COLOR_SCHEME[dataset],
#                     label=dataset,
#                     linewidth=0.25, 
#                     s=10
#                     )

# # plt.xscale("log")
# plt.xlabel(x_col) # Percentage of\nLR connections (> 50 % length)")
# # plt.ylabel(col.replace("_", " ").capitalize())
# # if the title is too long (define this), then split it into two lines at the last underscore
# plt.ylabel(col_for_title) # , fontsize=6)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

# plt.tight_layout() # TODO: INCLUDE IN DOCS


# In[ ]:





# In[157]:


# # patankar_2020_pathdependent
 
# fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

# x_col = "modularity" # computational_capacity_memory_capacity_total"
# y_col = "nct_control_std" # computational_capacity_nonlinear_capacity_total"
# plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
#             color="gray", linewidth=0, s=5, alpha=0.2) 

# if len(col) > 20:
#     col_for_title = re.split(r'[,,_]+', y_col)
#     len_col = len(col_for_title)
#     col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

# if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
#     col = "min_value"
#     col_for_title = "Distance Measure"

# for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

#     # Skip the gnm dataset, as it will now have two eta and gamma columns. 
#     if dataset == "hcp_schaefer_100_dataset_gnm":
#         continue

#     df_merged = dict_with_all_datasets[dataset]

#     if y_col in df_merged.columns:
#         plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
#                     df_merged[y_col],
#                 #    c = df_merged[col],
#                     edgecolor="black",
#                     color=COLOR_SCHEME[dataset],
#                     label=dataset,
#                     linewidth=0.25, 
#                     s=10
#                     )

# # plt.xscale("log")
# plt.xlabel(x_col) # Percentage of\nLR connections (> 50 % length)")
# # plt.ylabel(col.replace("_", " ").capitalize())
# # if the title is too long (define this), then split it into two lines at the last underscore
# plt.ylabel(col_for_title) # , fontsize=6)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
# plt.title("patankar_2020_pathdependent")
# plt.tight_layout()


# In[158]:


# # patankar_2020_pathdependent
 
# fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

# x_col = "richclub_n_edges" # computational_capacity_memory_capacity_total"
# y_col = "nct_control_std" # computational_capacity_nonlinear_capacity_total"
# plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
#             color="gray", linewidth=0, s=5, alpha=0.2) 

# if len(col) > 20:
#     col_for_title = re.split(r'[,,_]+', y_col)
#     len_col = len(col_for_title)
#     col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

# if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
#     col = "min_value"
#     col_for_title = "Distance Measure"

# for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

#     # Skip the gnm dataset, as it will now have two eta and gamma columns. 
#     if dataset == "hcp_schaefer_100_dataset_gnm":
#         continue

#     df_merged = dict_with_all_datasets[dataset]

#     if y_col in df_merged.columns:
#         plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
#                     df_merged[y_col],
#                 #    c = df_merged[col],
#                     edgecolor="black",
#                     color=COLOR_SCHEME[dataset],
#                     label=dataset,
#                     linewidth=0.25, 
#                     s=10
#                     )

# # plt.xscale("log")
# plt.xlabel(x_col) # Percentage of\nLR connections (> 50 % length)")
# # plt.ylabel(col.replace("_", " ").capitalize())
# # if the title is too long (define this), then split it into two lines at the last underscore
# plt.ylabel(col_for_title) # , fontsize=6)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
# plt.title("patankar_2020_pathdependent")
# plt.tight_layout()


# In[159]:


# patankar_2020_pathdependent
 
fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

y_col = "synchronizability_eigenratio_eigenratio" # computational_capacity_nonlinear_capacity_total" # 
# y_col = "computational_capacity_memory_capacity_total" # computational_capacity_memory_capacity_total"
# x_col = "repertoire_sweep_weighted_by_distances_T_critical" # 
x_col = "repertoire_sweep_weighted_by_distances_size_critical" # "repertoire_sweep_weighted_by_distances_diversity_critical" # "nct_control_std" # computational_capacity_nonlinear_capacity_total"

plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color="gray", linewidth=0, s=5, alpha=0.2) 

col_for_title = PROPERTY_NAMES[y_col] if y_col in PROPERTY_NAMES else y_col
if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_datasets[dataset]

    if y_col in df_merged.columns:
        plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                    df_merged[y_col],
                #    c = df_merged[col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    label=dataset,
                    linewidth=0.25, 
                    s=10
                    )

plt.ylim([-10, 100])
# plt.xscale("log")
plt.xlabel(PROPERTY_NAMES[x_col]) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.ylabel(col_for_title) # , fontsize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
# plt.title("patankar_2020_pathdependent")
plt.tight_layout()


# In[160]:


# kaiser_2006_nonoptimal
 
fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

y_col = "char_path_length"
x_col = "proportion_long_range_connections_0.3956" 

plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color="gray", linewidth=0, s=5, alpha=0.2) 

col_for_title = PROPERTY_NAMES[y_col] if y_col in PROPERTY_NAMES else y_col
if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in datasets_to_look_at: # dict_with_all_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_datasets[dataset]

    if y_col in df_merged.columns:
        plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                    df_merged[y_col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    label=dataset,
                    linewidth=0.25, 
                    s=10
                    )

plt.xlabel(PROPERTY_NAMES[x_col])
plt.ylabel(col_for_title)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

plt.tight_layout()


# In[161]:


# output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/connectome_distances/output") 
# output_folder = output_folder / "trade_off_scatters"
# output_folder.mkdir(exist_ok=True)

x_col = "proportion_long_range_connections_0.3956"

panels = {
    "A": ("computational_capacity_memory_capacity_total",    "Memory capacity"),
    "B": ("computational_capacity_total_capacity",           "Total capacity"),
    "C": ("computational_capacity_nonlinear_capacity_total", "Nonlinear capacity"),
    "D": ("computational_capacity_memory_nonlinear_ratio",   "Memory / nonlinear ratio"),
}

x_label = f"$f_{{\\rm LR\\,(39.56\%)}}$" # "Proportion long-range\nconnections (≥ 0.40)"

# ── mosaic layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplot_mosaic(
    """
    AAAB
    AAAC
    AAAD
    """,
    figsize=viz.cm_to_inch((18, 12)),
    layout="constrained",
    # share x across the small panels so zoom/pan stays in sync
    per_subplot_kw={
        "B": {"sharex": None},   # placeholder; real sharing done below
    },
)

# Share x-axis of B, C, D with each other (not with A — limits differ)
axes["C"].sharex(axes["B"])
axes["D"].sharex(axes["B"])

# ── shared plotting function ──────────────────────────────────────────────────
def plot_panel(ax, y_col, y_label, is_main=False):
    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        # print(dataset, len(df_merged.columns))
        if dataset == "hcp_schaefer_100_dataset_gnm":
            # ax.scatter(
            #     df_merged[x_col],
            #     df_merged[y_col],
            #     color=COLOR_SCHEME[dataset],
            #     edgecolor="black",
            #     linewidth=0.25,
            #     s=10 if is_main else 5,
            #     alpha=0.02 if "gnm" in dataset else 0.3,
            #     label=LABEL_MAP[dataset],
            # )
            ax.scatter(
                df_merged[x_col],
                df_merged[y_col],
                color="gray",
                linewidth=0,
                s=5,
                alpha=0.2, 
                label=LABEL_MAP[dataset]
            )   
        #     print("Skipped")
        #     continue
        else: 
            if x_col not in df_merged.columns or y_col not in df_merged.columns:
                print(f"Skipping {dataset} for panel {y_label} because required columns are missing.")
                continue

            ax.scatter(
                df_merged[x_col],
                df_merged[y_col],
                color=COLOR_SCHEME[dataset],
                edgecolor="black",
                linewidth=0.25,
                s=10 if is_main else 5,
                alpha=0.02 if "gnm" in dataset else 0.3,
                label=LABEL_MAP[dataset],
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if is_main:
        ax.legend()

# ── draw panels ───────────────────────────────────────────────────────────────
for panel_id, (y_col, y_label) in panels.items():
    ax = axes[panel_id]
    is_main = (panel_id == "A")
    plot_panel(ax, y_col, y_label, is_main=is_main)

    # ax.text(
    #     -0.15 if is_main else -0.35, 1.02,
    #     panel_id,
    #     transform=ax.transAxes,
    #     fontsize=8 if is_main else 6,
    #     fontweight="bold",
    #     va="bottom",
    # )

# ── align small-panel x-ticks to A's outermost ticks ─────────────────────────
fig.canvas.draw()                          # force matplotlib to compute auto-ticks

a_ticks = axes["A"].get_xticks()
# keep only ticks that fall within A's current view limits
a_xlim  = axes["A"].get_xlim()
a_ticks_visible = [t for t in a_ticks if a_xlim[0] <= t <= a_xlim[1]]

first_tick, last_tick = a_ticks_visible[0], a_ticks_visible[-1]

for panel_id in ("B", "C", "D"):
    ax = axes[panel_id]
    ax.set_xlim(a_xlim)                    # same data range as A
    ax.set_xticks([first_tick, last_tick])  # only the two boundary ticks
    # hide x-label on B and C to avoid clutter (only D gets the label)
    if panel_id != "D":
        ax.set_xlabel("")

plt.savefig(output_folder / "pareto_lr_connections.pdf", bbox_inches="tight", dpi=300)
plt.show()


# In[162]:


# output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/connectome_distances/output") 
# output_folder = output_folder / "trade_off_scatters"
# output_folder.mkdir(exist_ok=True)

# x_col = "proportion_long_range_connections_0.3956"

# panels = {
#     "A": ("targeted_attack_robustness_rob_targeted_auc",    "Robn"),
#     "B": ("targeted_attack_robustness_rob_total_capacity",           "Total capacity"),
#     "C": ("targeted_attack_robustness_rob_nonlinear_capacity", "Nonlinear capacity"),
#     # "D": ("targeted_attack_robustness_rob_memory_nonlinear_ratio",   "Memory / nonlinear ratio"),
# }

# x_label = f"$f_{{\\rm LR\\,(39.56\%)}}$" # "Proportion long-range\nconnections (≥ 0.40)"

# # ── mosaic layout ─────────────────────────────────────────────────────────────
# fig, axes = plt.subplot_mosaic(
#     """
#     ABC 
#     """,
#     figsize=viz.cm_to_inch((18, 12)),
#     layout="constrained",
#     # share x across the small panels so zoom/pan stays in sync
#     per_subplot_kw={
#         "B": {"sharex": None},   # placeholder; real sharing done below
#     },
# )

# # Share x-axis of B, C, D with each other (not with A — limits differ)
# axes["C"].sharex(axes["B"])
# # axes["D"].sharex(axes["B"])

# # ── shared plotting function ──────────────────────────────────────────────────
# def plot_panel(ax, y_col, y_label, is_main=False):
#     for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
#         df_merged = dict_with_all_datasets[dataset] # for dataset, df_merged in dict_with_all_datasets.items():
#         print(dataset, len(df_merged.columns))
#         if dataset == "hcp_schaefer_100_dataset_gnm":
#             # ax.scatter(
#             #     df_merged[x_col],
#             #     df_merged[y_col],
#             #     color=COLOR_SCHEME[dataset],
#             #     edgecolor="black",
#             #     linewidth=0.25,
#             #     s=10 if is_main else 5,
#             #     alpha=0.02 if "gnm" in dataset else 0.3,
#             #     label=LABEL_MAP[dataset],
#             # )
#             ax.scatter(
#                 df_merged[x_col],
#                 df_merged[y_col],
#                 color="gray",
#                 linewidth=0,
#                 s=5,
#                 alpha=0.2, 
#                 label=LABEL_MAP[dataset]
#             )   
#         #     print("Skipped")
#         #     continue
#         else: 
#             if x_col not in df_merged.columns or y_col not in df_merged.columns:
#                 print(f"Skipping {dataset} for panel {y_label} because required columns are missing.")
#                 continue

#             ax.scatter(
#                 df_merged[x_col],
#                 df_merged[y_col],
#                 color=COLOR_SCHEME[dataset],
#                 edgecolor="black",
#                 linewidth=0.25,
#                 s=10 if is_main else 5,
#                 alpha=0.02 if "gnm" in dataset else 0.3,
#                 label=LABEL_MAP[dataset],
#             )

#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)

#     if is_main:
#         ax.legend()

# # ── draw panels ───────────────────────────────────────────────────────────────
# for panel_id, (y_col, y_label) in panels.items():
#     ax = axes[panel_id]
#     is_main = (panel_id == "A")
#     plot_panel(ax, y_col, y_label, is_main=is_main)

#     # ax.text(
#     #     -0.15 if is_main else -0.35, 1.02,
#     #     panel_id,
#     #     transform=ax.transAxes,
#     #     fontsize=8 if is_main else 6,
#     #     fontweight="bold",
#     #     va="bottom",
#     # )

# # ── align small-panel x-ticks to A's outermost ticks ─────────────────────────
# fig.canvas.draw()                          # force matplotlib to compute auto-ticks

# a_ticks = axes["A"].get_xticks()
# # keep only ticks that fall within A's current view limits
# a_xlim  = axes["A"].get_xlim()
# a_ticks_visible = [t for t in a_ticks if a_xlim[0] <= t <= a_xlim[1]]

# first_tick, last_tick = a_ticks_visible[0], a_ticks_visible[-1]

# for panel_id in ("B", "C", "D"):
#     ax = axes[panel_id]
#     ax.set_xlim(a_xlim)                    # same data range as A
#     ax.set_xticks([first_tick, last_tick])  # only the two boundary ticks
#     # hide x-label on B and C to avoid clutter (only D gets the label)
#     if panel_id != "D":
#         ax.set_xlabel("")

# plt.savefig(output_folder / "pareto_lr_connections.pdf", bbox_inches="tight", dpi=300)
# plt.show()


# In[163]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]


# In[164]:


from utils import get_combined_colors

combined_colors = get_combined_colors(dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]) 


# In[165]:


# output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/connectome_distances/output") 
# output_folder = output_folder / "trade_off_scatters"
# output_folder.mkdir(exist_ok=True)

x_col = "proportion_long_range_connections_0.3956"

panels = {
    "A": ("computational_capacity_memory_capacity_total",    "Memory capacity"),
    "B": ("computational_capacity_total_capacity",           "Total capacity"),
    "C": ("computational_capacity_nonlinear_capacity_total", "Nonlinear capacity"),
    "D": ("computational_capacity_memory_nonlinear_ratio",   "Memory / nonlinear ratio"),
}

x_label = f"$f_{{\\rm LR\\,(39.56\\%)}}$" # "Proportion long-range\nconnections (≥ 0.40)"

# ── mosaic layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplot_mosaic(
    """
    AAAB
    AAAC
    AAAD
    """,
    figsize=viz.cm_to_inch((18, 12)),
    layout="constrained",
    # share x across the small panels so zoom/pan stays in sync
    per_subplot_kw={
        "B": {"sharex": None},   # placeholder; real sharing done below
    },
)

# Share x-axis of B, C, D with each other (not with A — limits differ)
axes["C"].sharex(axes["B"])
axes["D"].sharex(axes["B"])

# ── shared plotting function ──────────────────────────────────────────────────
def plot_panel(ax, y_col, y_label, is_main=False):
    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset] # for dataset, df_merged in dict_with_all_datasets.items():
        
        print(dataset, len(df_merged.columns), len(df_merged))
        
        # Skip Lexis datasets
        # if "lexi" in dataset:
        #     print("Found lexi dataset, skipping for now.")
        #     continue
        
        if dataset == "hcp_schaefer_100_dataset_gnm":
            # ax.scatter(
            #     df_merged[x_col],
            #     df_merged[y_col],
            #     color=COLOR_SCHEME[dataset],
            #     edgecolor="black",
            #     linewidth=0.25,
            #     s=10 if is_main else 5,
            #     alpha=0.02 if "gnm" in dataset else 0.3,
            #     label=LABEL_MAP[dataset],
            # )
            len(df_merged)
            ax.scatter(
                df_merged[x_col],
                df_merged[y_col],
                color=combined_colors, # "gray",
                linewidth=0,
                s=5 if is_main else 1,
                alpha=0.2, 
                label=LABEL_MAP[dataset]
            )   
        #     print("Skipped")
        #     continue
        else: 
            if x_col not in df_merged.columns or y_col not in df_merged.columns:
                print(f"Skipping {dataset} for panel {y_label} because required columns are missing.")
                continue

            ax.scatter(
                df_merged[x_col],
                df_merged[y_col],
                color=COLOR_SCHEME[dataset],
                edgecolor="black",
                linewidth=0.25,
                s=10 if is_main else 3,
                alpha=1, # 0.02 if "gnm" in dataset else 0.3,
                label=LABEL_MAP[dataset],
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if is_main:
        ax.legend()

# ── draw panels ───────────────────────────────────────────────────────────────
for panel_id, (y_col, y_label) in panels.items():
    ax = axes[panel_id]
    is_main = (panel_id == "A")
    plot_panel(ax, y_col, y_label, is_main=is_main)

    # ax.text(
    #     -0.15 if is_main else -0.35, 1.02,
    #     panel_id,
    #     transform=ax.transAxes,
    #     fontsize=8 if is_main else 6,
    #     fontweight="bold",
    #     va="bottom",
    # )

# ── align small-panel x-ticks to A's outermost ticks ─────────────────────────
fig.canvas.draw()                          # force matplotlib to compute auto-ticks

a_ticks = axes["A"].get_xticks()
# keep only ticks that fall within A's current view limits
a_xlim  = axes["A"].get_xlim()
a_ticks_visible = [t for t in a_ticks if a_xlim[0] <= t <= a_xlim[1]]

first_tick, last_tick = a_ticks_visible[0], a_ticks_visible[-1]

for panel_id in ("B", "C", "D"):
    ax = axes[panel_id]
    ax.set_xlim(a_xlim)                    # same data range as A
    ax.set_xticks([first_tick, last_tick])  # only the two boundary ticks
    # hide x-label on B and C to avoid clutter (only D gets the label)
    if panel_id != "D":
        ax.set_xlabel("")

plt.savefig(output_folder / "pareto_lr_connections_mami_and_hcp_colorful.pdf", bbox_inches="tight", dpi=200)
print(output_folder / "pareto_lr_connections_mami_and_hcp_colorful.pdf")
plt.show()


# In[166]:


# output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/connectome_distances/output") 
# output_folder = output_folder / "trade_off_scatters"
# output_folder.mkdir(exist_ok=True)

x_col = "proportion_long_range_connections_0.3956"

panels = {
    "A": ("computational_capacity_memory_capacity_total",    "Memory capacity"),
    "B": ("computational_capacity_total_capacity",           "Total capacity"),
    "C": ("computational_capacity_nonlinear_capacity_total", "Nonlinear capacity"),
    "D": ("computational_capacity_memory_nonlinear_ratio",   "Memory / nonlinear ratio"),
}

x_label = f"$f_{{\\rm LR\\,(39.56\\%)}}$" # "Proportion long-range\nconnections (≥ 0.40)"

# ── mosaic layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplot_mosaic(
    """
    AAAB
    AAAC
    AAAD
    """,
    figsize=viz.cm_to_inch((18, 12)),
    layout="constrained",
    # share x across the small panels so zoom/pan stays in sync
    per_subplot_kw={
        "B": {"sharex": None},   # placeholder; real sharing done below
    },
)

# Share x-axis of B, C, D with each other (not with A — limits differ)
axes["C"].sharex(axes["B"])
axes["D"].sharex(axes["B"])

# ── shared plotting function ──────────────────────────────────────────────────
def plot_panel(ax, y_col, y_label, is_main=False):
    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset] # for dataset, df_merged in dict_with_all_datasets.items():
        
        print(dataset, len(df_merged.columns), len(df_merged))
        
        # Skip Lexis datasets
        # if "lexi" in dataset:
        #     print("Found lexi dataset, skipping for now.")
        #     continue
        
        if dataset == "hcp_schaefer_100_dataset_gnm":
            # ax.scatter(
            #     df_merged[x_col],
            #     df_merged[y_col],
            #     color=COLOR_SCHEME[dataset],
            #     edgecolor="black",
            #     linewidth=0.25,
            #     s=10 if is_main else 5,
            #     alpha=0.02 if "gnm" in dataset else 0.3,
            #     label=LABEL_MAP[dataset],
            # )
            len(df_merged)
            ax.scatter(
                df_merged[x_col],
                df_merged[y_col],
                color=combined_colors, # "gray",
                linewidth=0,
                s=5 if is_main else 1,
                alpha=0.2, 
                label=LABEL_MAP[dataset]
            )   
        #     print("Skipped")
        #     continue
        else: 
            if x_col not in df_merged.columns or y_col not in df_merged.columns:
                print(f"Skipping {dataset} for panel {y_label} because required columns are missing.")
                continue

            ax.scatter(
                df_merged[x_col],
                df_merged[y_col],
                color=COLOR_SCHEME[dataset],
                edgecolor="black",
                linewidth=0.25,
                s=10 if is_main else 3,
                alpha=1, # 0.02 if "gnm" in dataset else 0.3,
                label=LABEL_MAP[dataset],
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if is_main:
        ax.legend()

# ── draw panels ───────────────────────────────────────────────────────────────
for panel_id, (y_col, y_label) in panels.items():
    ax = axes[panel_id]
    is_main = (panel_id == "A")
    plot_panel(ax, y_col, y_label, is_main=is_main)

    # ax.text(
    #     -0.15 if is_main else -0.35, 1.02,
    #     panel_id,
    #     transform=ax.transAxes,
    #     fontsize=8 if is_main else 6,
    #     fontweight="bold",
    #     va="bottom",
    # )

# ── align small-panel x-ticks to A's outermost ticks ─────────────────────────
fig.canvas.draw()                          # force matplotlib to compute auto-ticks

a_ticks = axes["A"].get_xticks()
# keep only ticks that fall within A's current view limits
a_xlim  = axes["A"].get_xlim()
a_ticks_visible = [t for t in a_ticks if a_xlim[0] <= t <= a_xlim[1]]

first_tick, last_tick = a_ticks_visible[0], a_ticks_visible[-1]

for panel_id in ("B", "C", "D"):
    ax = axes[panel_id]
    ax.set_xlim(a_xlim)                    # same data range as A
    ax.set_xticks([first_tick, last_tick])  # only the two boundary ticks
    # hide x-label on B and C to avoid clutter (only D gets the label)
    if panel_id != "D":
        ax.set_xlabel("")

plt.savefig(output_folder / "pareto_lr_connections_mami_and_hcp_colorful.pdf", bbox_inches="tight", dpi=200)
print(output_folder / "pareto_lr_connections_mami_and_hcp_colorful.pdf")
plt.show()


# In[167]:


# x_col = "wiring_cost" # 
x_col = "proportion_long_range_connections_0.3956"

panels = {
    "A": ("computational_capacity_memory_capacity_total",    "Memory capacity"),
    "B": ("computational_capacity_total_capacity",           "Total capacity"),
    "C": ("computational_capacity_nonlinear_capacity_total", "Nonlinear capacity"),
    "D": ("computational_capacity_memory_nonlinear_ratio",   "Memory / nonlinear ratio"),
}

x_label = f"$f_{{\\rm LR\\,(39.56\\%)}}$" # "Proportion long-range\nconnections (≥ 0.40)"

# ── mosaic layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplot_mosaic(
    """
    AAAB
    AAAC
    AAAD
    """,
    figsize=viz.cm_to_inch((18, 12)),
    layout="constrained",
    # share x across the small panels so zoom/pan stays in sync
    per_subplot_kw={
        "B": {"sharex": None},   # placeholder; real sharing done below
    },
)

# Share x-axis of B, C, D with each other (not with A — limits differ)
axes["C"].sharex(axes["B"])
axes["D"].sharex(axes["B"])

# ── shared plotting function ──────────────────────────────────────────────────
def plot_panel(ax, y_col, y_label, is_main=False):
    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset] 
        print(dataset, len(df_merged.columns), len(df_merged))
        
        # Skip Lexis datasets
        # if "lexi" in dataset:
        #     print("Found lexi dataset, skipping for now.")
        #     continue
        
        if dataset == "hcp_schaefer_100_dataset_gnm":
            # pass
            # ax.scatter(
            #     df_merged[x_col],
            #     df_merged[y_col],
            #     color=COLOR_SCHEME[dataset],
            #     edgecolor="black",
            #     linewidth=0.25,
            #     s=10 if is_main else 5,
            #     alpha=0.02 if "gnm" in dataset else 0.3,
            #     label=LABEL_MAP[dataset],
            # )
            len(df_merged)
            ax.scatter(
                df_merged[x_col],
                df_merged[y_col],
                color=combined_colors, # "gray",
                linewidth=0,
                s=5,
                alpha=0.2, 
                label=LABEL_MAP[dataset]
            )   
        #     print("Skipped")
        # #     continue
        else: 
            pass
            # if x_col not in df_merged.columns or y_col not in df_merged.columns:
            #     print(f"Skipping {dataset} for panel {y_label} because required columns are missing.")
            #     continue

            # ax.scatter(
            #     df_merged[x_col],
            #     df_merged[y_col],
            #     color=COLOR_SCHEME[dataset],
            #     edgecolor="black",
            #     linewidth=0.25,
            #     s=10 if is_main else 5,
            #     alpha=1, # 0.02 if "gnm" in dataset else 0.3,
            #     label=LABEL_MAP[dataset],
            # )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if is_main:
        ax.legend()

# ── draw panels ───────────────────────────────────────────────────────────────
for panel_id, (y_col, y_label) in panels.items():
    ax = axes[panel_id]
    is_main = (panel_id == "A")
    plot_panel(ax, y_col, y_label, is_main=is_main)

    # ax.text(
    #     -0.15 if is_main else -0.35, 1.02,
    #     panel_id,
    #     transform=ax.transAxes,
    #     fontsize=8 if is_main else 6,
    #     fontweight="bold",
    #     va="bottom",
    # )

# ── align small-panel x-ticks to A's outermost ticks ─────────────────────────
fig.canvas.draw()                          # force matplotlib to compute auto-ticks

a_ticks = axes["A"].get_xticks()
# keep only ticks that fall within A's current view limits
a_xlim  = axes["A"].get_xlim()
a_ticks_visible = [t for t in a_ticks if a_xlim[0] <= t <= a_xlim[1]]

first_tick, last_tick = a_ticks_visible[0], a_ticks_visible[-1]

for panel_id in ("B", "C", "D"):
    ax = axes[panel_id]
    ax.set_xlim(a_xlim)                    # same data range as A
    ax.set_xticks([first_tick, last_tick])  # only the two boundary ticks
    # hide x-label on B and C to avoid clutter (only D gets the label)
    if panel_id != "D":
        ax.set_xlabel("")

# plt.savefig(output_folder / "pareto_lr_connections_mami_and_hcp.pdf", bbox_inches="tight", dpi=200)
# plt.show()


# ![image.png](attachment:image.png)

# # Further trade-off experiments

# In[168]:


# ── Trade-off grid: 2×3 established pairs ─────────────────────────────────────

trade_off_panels = {
    "A": {
        "x": "global_efficiency",
        "y": "modularity",
        "title": "Integration vs. Segregation",
        "ref": "Sporns & Betzel, 2016",
    },
    "B": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "global_efficiency",
        "title": "Economy vs. Efficiency",
        "ref": "Bullmore & Sporns, 2012",
    },
    "C": {
        "x": "algebraic_connectivity_nx",
        "y": "modularity",
        "title": "Robustness vs. Segregation",
        "ref": "see e.g. Sporns, 2013",
    },
    "D": {
        "x": "computational_capacity_memory_capacity_total",
        "y": "computational_capacity_nonlinear_capacity_total",
        "title": "Memory vs. Nonlinear Computation",
        "ref": "Dambre et al., 2012",
    },
    "E": {
        "x": "char_path_length",
        "y": "avg_clustering",
        "title": "Small-World Plane",
        "ref": "Watts & Strogatz, 1998",
    },
    "F": {
        "x": "global_efficiency",
        "y": "targeted_attack_robustness_rob_targeted_auc",
        "title": "Efficiency vs. Robustness",
        "ref": "Albert et al., 2000",
    },
}

INCLUDE_LEXIS = True # False # True

fig, axes_flat = plt.subplots(2, 3, figsize=viz.cm_to_inch((18, 12)), layout="constrained")
axes_grid = {k: ax for k, ax in zip(trade_off_panels.keys(), axes_flat.ravel())}

for panel_id, cfg in trade_off_panels.items():
    ax = axes_grid[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col = cfg["x"]
        y_col = cfg["y"]

        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray"
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,  # rasterize the 90k cloud for fast PDF
        )

    # Axis labels from PROPERTY_NAMES (fall back to raw name)
    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=7)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=7)
    ax.tick_params(labelsize=6)

    # Panel letter + title
    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7.5, fontweight="bold", loc="left")

    # Reference annotation (bottom-right, small)
    ax.annotate(
        cfg["ref"],
        xy=(1, 0), xycoords="axes fraction",
        fontsize=4.5, color="gray",
        ha="right", va="bottom",
    )

# Single shared legend from panel A
handles, labels = axes_grid["A"].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="outside lower center",
    ncol=min(len(labels), 6),
    fontsize=6,
    markerscale=1.5,
    frameon=False,
)

plt.savefig(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf", bbox_inches="tight", dpi=300)
print(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf")
plt.show()


# In[169]:


# ── Trade-off grid: 2×3 established pairs ─────────────────────────────────────
from config import PROPERTY_NAMES


trade_off_panels = {
    "A": {
        "x": "global_efficiency",
        "y": "modularity",
        "title": "Integration vs. Segregation",
        "ref": "Sporns & Betzel, 2016",
    },
    "B": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "global_efficiency",
        "title": "Economy vs. Efficiency",
        "ref": "Bullmore & Sporns, 2012",
    },
    "C": {
        "x": "algebraic_connectivity_fiedler_value", # "algebraic_connectivity_nx",
        "y": "modularity",
        "title": "Robustness vs. Segregation",
        "ref": "see e.g. Sporns, 2013",
    },
    "D": {
        "x": "computational_capacity_memory_capacity_total",
        "y": "computational_capacity_nonlinear_capacity_total",
        "title": "Memory vs. Nonlinear Computation",
        "ref": "Dambre et al., 2012",
    },
    "E": {
        "x": "char_path_length",
        "y": "avg_clustering",
        "title": "Small-World Plane",
        "ref": "Watts & Strogatz, 1998",
    },
    "F": {
        "x": "global_efficiency",
        "y": "algebraic_connectivity_fiedler_value", # "targeted_attack_robustness_rob_targeted_auc",
        "title": "Efficiency vs. Robustness",
        "ref": "Albert et al., 2000",
    },
}

INCLUDE_LEXIS = True # False # True

fig, axes_flat = plt.subplots(2, 3, figsize=viz.cm_to_inch((18, 12))) # , layout="constrained")
axes_grid = {k: ax for k, ax in zip(trade_off_panels.keys(), axes_flat.ravel())}

for panel_id, cfg in trade_off_panels.items():
    ax = axes_grid[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col = cfg["x"]
        y_col = cfg["y"]

        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray"
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,  # rasterize the 90k cloud for fast PDF
        )

    # Axis labels from PROPERTY_NAMES (fall back to raw name)
    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"])) # , fontsize=7)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"])) # , fontsize=7)
    ax.tick_params() # labelsize=6)

    # Panel letter + title
    # ax.set_title(f"{panel_id}  {cfg['title']}", # fontsize=7.5, fontweight="bold", 
    #             #  loc="left"
    #              )
    ax.set_title(f"{cfg['title']}", # fontsize=7.5, fontweight="bold", 
                #  loc="left"
                )

    # Reference annotation (bottom-right, small)
    # ax.annotate(
    #     cfg["ref"],
    #     xy=(1, 0), xycoords="axes fraction",
    #     # fontsize=4.5, 
    #     color="gray",
    #     ha="right", va="bottom",
    # )

# # Single shared legend from panel A
# handles, labels = axes_grid["A"].get_legend_handles_labels()
# fig.legend(
#     handles, labels,
#     # loc="outside lower", #  center",
#     ncol=min(len(labels), 3),
#     # fontsize=6,
#     # markerscale=1.5,
#     # frameon=False,
#     bbox_to_anchor=(0.5, 0), 
#     loc='upper center',
# )

plt.tight_layout()

plt.savefig(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf", bbox_inches="tight", dpi=300)
print(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf")
plt.show()


# In[170]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from config import COLORS, COLOR_SCHEME, LABEL_MAP, gray_cmap, bone_white, half_black, emp_dataset_and_experiment_pairs
from config import CATEGORY_ORDER, CATEGORY_COLOURS # Integration etc.
import re
from config import PROPERTY_NAMES, remaining_categories, REPRESENTATIVES_FOR_GOALS, selected_properties


# In[ ]:


# # ── Trade-off grid: 2×3 established pairs ─────────────────────────────────────
# from config import PROPERTY_NAMES

# trade_off_panels = {
#     "A": {
#         "x": "global_efficiency",
#         "y": "modularity",
#         "title": "Integration vs. Segregation",
#         "ref": "Sporns & Betzel, 2016",
#     },
#     "B": {
#         "x": "proportion_long_range_connections_0.3956",
#         "y": "global_efficiency",
#         "title": "Economy vs. Efficiency",
#         "ref": "Bullmore & Sporns, 2012",
#     },
#     "C": {
#         "x": "algebraic_connectivity_nx",
#         "y": "modularity",
#         "title": "Robustness vs. Segregation",
#         "ref": "see e.g. Sporns, 2013",
#     },
#     "D": {
#         "x": "computational_capacity_memory_capacity_total",
#         "y": "computational_capacity_nonlinear_capacity_total",
#         "title": "Memory vs. Nonlinear Computation",
#         "ref": "Dambre et al., 2012",
#     },
#     "E": {
#         "x": "char_path_length",
#         "y": "avg_clustering",
#         "title": "Small-World Plane",
#         "ref": "Watts & Strogatz, 1998",
#     },
#     "F": {
#         "x": "global_efficiency",
#         "y": "targeted_attack_robustness_rob_targeted_auc",
#         "title": "Efficiency vs. Robustness",
#         "ref": "Albert et al., 2000",
#     },
# }

# INCLUDE_LEXIS = True # False # True

# fig, axes_flat = plt.subplots(2, 3, figsize=viz.cm_to_inch((18, 12))) # , layout="constrained")
# axes_grid = {k: ax for k, ax in zip(trade_off_panels.keys(), axes_flat.ravel())}

# for panel_id, cfg in trade_off_panels.items():
#     ax = axes_grid[panel_id]

#     for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
#         df_merged = dict_with_all_datasets[dataset]
        
#         if (not INCLUDE_LEXIS) and ("lexi" in dataset):
#             print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
#             continue
        
#         x_col = cfg["x"]
#         y_col = cfg["y"]

#         if x_col not in df_merged.columns or y_col not in df_merged.columns:
#             continue

#         is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

#         ax.scatter(
#             df_merged[x_col],
#             df_merged[y_col],
#             color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray"
#             edgecolor="black" if not is_gnm else "none",
#             linewidth=0.25 if not is_gnm else 0,
#             s=5,
#             alpha=0.15 if is_gnm else 0.4,
#             label=LABEL_MAP[dataset] if panel_id == "A" else None,
#             zorder=1 if is_gnm else 2,
#             rasterized=is_gnm,  # rasterize the 90k cloud for fast PDF
#         )

#     # Axis labels from PROPERTY_NAMES (fall back to raw name)
#     ax.set_xlabel(PROPERTY_NAMES[cfg["x"]]) # .replace(" ", "\n")) # .get(cfg["x"], cfg["x"])) # , fontsize=7)
#     ax.set_ylabel(PROPERTY_NAMES[cfg["y"]]) # .replace(" ", "\n")) # .get(cfg["y"], cfg["y"])) # , fontsize=7)
#     ax.tick_params() # labelsize=6)

#     # Panel letter + title
#     # ax.set_title(f"{panel_id}  {cfg['title']}", # fontsize=7.5, fontweight="bold", 
#     #             #  loc="left"
#     #              )
#     ax.set_title(f"{cfg['title']}", # fontsize=7.5, fontweight="bold", 
#                 #  loc="left"
#                 )

#     # Reference annotation (bottom-right, small)
#     # ax.annotate(
#     #     cfg["ref"],
#     #     xy=(1, 0), xycoords="axes fraction",
#     #     # fontsize=4.5, 
#     #     color="gray",
#     #     ha="right", va="bottom",
#     # )

# # # Single shared legend from panel A
# # handles, labels = axes_grid["A"].get_legend_handles_labels()
# # fig.legend(
# #     handles, labels,
# #     # loc="outside lower", #  center",
# #     ncol=min(len(labels), 3),
# #     # fontsize=6,
# #     # markerscale=1.5,
# #     # frameon=False,
# #     bbox_to_anchor=(0.5, 0), 
# #     loc='upper center',
# # )

# plt.tight_layout()

# plt.savefig(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf", bbox_inches="tight", dpi=300)
# print(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf")
# plt.show()


# In[ ]:


# ── Trade-off grid: 2×3 established pairs ─────────────────────────────────────
from config import PROPERTY_NAMES

trade_off_panels = {
    "A": {
        "x": "global_efficiency",
        "y": "modularity",
        "title": "Integration vs. Segregation",
        "ref": "Sporns & Betzel, 2016",
    },
    "B": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "global_efficiency",
        "title": "Economy vs. Efficiency",
        "ref": "Bullmore & Sporns, 2012",
    },
    "C": {
        "x": "algebraic_connectivity_fiedler_value", # "algebraic_connectivity_nx",
        "y": "modularity",
        "title": "Robustness vs. Segregation",
        "ref": "see e.g. Sporns, 2013",
    },
    "D": {
        "x": "computational_capacity_memory_capacity_total",
        "y": "computational_capacity_nonlinear_capacity_total",
        "title": "Memory vs. Nonlinear Computation",
        "ref": "Dambre et al., 2012",
    },
    "E": {
        "x": "char_path_length",
        "y": "avg_clustering",
        "title": "Small-World Plane",
        "ref": "Watts & Strogatz, 1998",
    },
    "F": {
        "x": "global_efficiency",
        "y":  "algebraic_connectivity_fiedler_value", # "targeted_attack_robustness_rob_targeted_auc",
        "title": "Efficiency vs. Robustness",
        "ref": "Albert et al., 2000",
    },
}

INCLUDE_LEXIS = True # False # True

fig, axes_flat = plt.subplots(2, 3, figsize=viz.cm_to_inch((18, 12))) # , layout="constrained")
axes_grid = {k: ax for k, ax in zip(trade_off_panels.keys(), axes_flat.ravel())}

for panel_id, cfg in trade_off_panels.items():
    ax = axes_grid[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col = cfg["x"]
        y_col = cfg["y"]

        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray"
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,  # rasterize the 90k cloud for fast PDF
        )

    # Axis labels from PROPERTY_NAMES (fall back to raw name)
    ax.set_xlabel(PROPERTY_NAMES[cfg["x"]]) # .replace(" ", "\n")) # .get(cfg["x"], cfg["x"])) # , fontsize=7)
    ax.set_ylabel(PROPERTY_NAMES[cfg["y"]]) # .replace(" ", "\n")) # .get(cfg["y"], cfg["y"])) # , fontsize=7)
    ax.tick_params() # labelsize=6)

    # Panel letter + title
    # ax.set_title(f"{panel_id}  {cfg['title']}", # fontsize=7.5, fontweight="bold", 
    #             #  loc="left"
    #              )
    ax.set_title(f"{cfg['title']}", # fontsize=7.5, fontweight="bold", 
                #  loc="left"
                )

    # Reference annotation (bottom-right, small)
    # ax.annotate(
    #     cfg["ref"],
    #     xy=(1, 0), xycoords="axes fraction",
    #     # fontsize=4.5, 
    #     color="gray",
    #     ha="right", va="bottom",
    # )

# # Single shared legend from panel A
# handles, labels = axes_grid["A"].get_legend_handles_labels()
# fig.legend(
#     handles, labels,
#     # loc="outside lower", #  center",
#     ncol=min(len(labels), 3),
#     # fontsize=6,
#     # markerscale=1.5,
#     # frameon=False,
#     bbox_to_anchor=(0.5, 0), 
#     loc='upper center',
# )

plt.tight_layout()

# plt.savefig(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf", bbox_inches="tight", dpi=300)
# print(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf")
plt.show()


# In[ ]:


# ── Trade-off grid: 2×3 established pairs ─────────────────────────────────────
from config import PROPERTY_NAMES

trade_off_panels = {
    "A": {
        "x": "global_efficiency",
        "y": "modularity",
        "title": "Integration vs. Segregation",
        "ref": "Sporns & Betzel, 2016",
    },
    "B": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "global_efficiency",
        "title": "Economy vs. Efficiency",
        "ref": "Bullmore & Sporns, 2012",
    },
    "C": {
        "x": "char_path_length",
        "y": "avg_clustering",
        "title": "Small-World Plane",
        "ref": "Watts & Strogatz, 1998",
    },
    # "C": {
    #     "x": "algebraic_connectivity_fiedler_value", # "algebraic_connectivity_nx",
    #     "y": "modularity",
    #     "title": "Robustness vs. Segregation",
    #     "ref": "see e.g. Sporns, 2013",
    # },
    "D": {
        "x": "diffusion_efficiency",
        "y": "global_efficiency", 
        "title": "",
        "ref": "",
    },
    "E": {
        "x": "diffusion_efficiency",
        "y": "propagation_efficiency", 
        "title": "",
        "ref": "",
    },
    "F": {
        "x": "global_efficiency",
        "y": "propagation_efficiency", 
        "title": "",
        "ref": "",
    },
    # "D": {
    #     "x": "computational_capacity_memory_capacity_total",
    #     "y": "computational_capacity_nonlinear_capacity_total",
    #     "title": "Memory vs. Nonlinear Computation",
    #     "ref": "Dambre et al., 2012",
    # },
    # "E": {
    #     "x": "char_path_length",
    #     "y": "avg_clustering",
    #     "title": "Small-World Plane",
    #     "ref": "Watts & Strogatz, 1998",
    # },
    # "F": {
    #     "x": "global_efficiency",
    #     "y":  "algebraic_connectivity_fiedler_value", # "targeted_attack_robustness_rob_targeted_auc",
    #     "title": "Efficiency vs. Robustness",
    #     "ref": "Albert et al., 2000",
    # },
}

INCLUDE_LEXIS = True # False # True

fig, axes_flat = plt.subplots(2, 3, figsize=viz.cm_to_inch((18, 12))) # , layout="constrained")
axes_grid = {k: ax for k, ax in zip(trade_off_panels.keys(), axes_flat.ravel())}

for panel_id, cfg in trade_off_panels.items():
    ax = axes_grid[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col = cfg["x"]
        y_col = cfg["y"]

        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray"
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,  # rasterize the 90k cloud for fast PDF
        )

    # Axis labels from PROPERTY_NAMES (fall back to raw name)
    ax.set_xlabel(PROPERTY_NAMES[cfg["x"]]) # .replace(" ", "\n")) # .get(cfg["x"], cfg["x"])) # , fontsize=7)
    ax.set_ylabel(PROPERTY_NAMES[cfg["y"]]) # .replace(" ", "\n")) # .get(cfg["y"], cfg["y"])) # , fontsize=7)
    ax.tick_params() # labelsize=6)

    # Panel letter + title
    # ax.set_title(f"{panel_id}  {cfg['title']}", # fontsize=7.5, fontweight="bold", 
    #             #  loc="left"
    #              )
    ax.set_title(f"{cfg['title']}", # fontsize=7.5, fontweight="bold", 
                #  loc="left"
                )

    # Reference annotation (bottom-right, small)
    # ax.annotate(
    #     cfg["ref"],
    #     xy=(1, 0), xycoords="axes fraction",
    #     # fontsize=4.5, 
    #     color="gray",
    #     ha="right", va="bottom",
    # )

# # Single shared legend from panel A
# handles, labels = axes_grid["A"].get_legend_handles_labels()
# fig.legend(
#     handles, labels,
#     # loc="outside lower", #  center",
#     ncol=min(len(labels), 3),
#     # fontsize=6,
#     # markerscale=1.5,
#     # frameon=False,
#     bbox_to_anchor=(0.5, 0), 
#     loc='upper center',
# )

plt.tight_layout()

plt.savefig(output_folder / f"trade_off_grid_established_6_squares.pdf", bbox_inches="tight", dpi=300)
print(output_folder / f"trade_off_grid_established_6_squares.pdf")
plt.show()


# In[ ]:


"""
Novel trade-off figure: 4 scatter plots + 1 ternary triangle.

Layout (mosaic):
    A B E E
    C D E E 

A: Memory–nonlinear ratio vs. mean Ricci curvature
B: Total capacity vs. Lyapunov exponent
C: Targeted robustness vs. proportion LR connections
D: State dimensionality vs. modularity
E: Ternary triangle (wiring economy – robustness – computation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from pathlib import Path

INCLUDE_LEXIS = True

# ── Assumes these are already defined in your notebook: ──────────────────────
# dict_with_all_datasets, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES, viz
# If running standalone, import/define them above this point.

# ── Output ────────────────────────────────────────────────────────────────────
# output_folder = Path("...your path.../output/trade_off_scatters")
# output_folder.mkdir(exist_ok=True)

# ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        "x": "nct_control_std", 
        "y": "targeted_attack_robustness_rob_random_auc", 
        "title": "Controllability vs. Robustness",
        "note": "",
    },
    "B": {
        "x": "repertoire_sweep_weighted_by_distances_T_critical",
        "y": "mc_input_scaling_0_1_mc_mean", 
        "title": "Repertoire vs. Computation",
        "note": "",
    },
    "C": {
        "x": "mc_input_scaling_0_1_mc_mean",
        "y": "targeted_attack_robustness_rob_random_auc",
        "title": "Memory vs. Random Failure Robustness",
        "note": "",
    },
    "D": {
        # "x": "targeted_attack_robustness_rob_random_half", # modularity",
        # "y": "targeted_attack_robustness_rob_targeted_half", # computational_capacity_state_dimensionality",
        "x": "mc_input_scaling_0_1_mc_mean", # modularity",
        "y": "targeted_attack_robustness_rob_targeted_auc", # computational_capacity_state_dimensionality",
        "title": "Memory vs. Targeted Attack Robustness", # Segregation vs. State Richness",
        "note": "",
    },
}

# ── Ternary axes (panel E) ────────────────────────────────────────────────────
ternary_cols = {
    "economy":     "proportion_long_range_connections_0.3956",
    "robustness":  "targeted_attack_robustness_rob_targeted_auc", # algebraic_connectivity_fiedler_value", # targeted_attack_robustness_rob_targeted_auc",
    "computation": "mc_input_scaling_0_1_mc_mean", # computational_capacity_total_capacity",
}
ternary_labels = {
    "economy":     "Wiring economy\n(1 - f_LR)",
    "robustness":  "Robustness\n(targeted AUC)",
    "computation": "Computation\n(total capacity)",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Ternary helpers (no mpltern needed)
# ═════════════════════════════════════════════════════════════════════════════

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0,  0.0],                        # vertex 0: economy     (bottom-left)
    [1.0,  0.0],                        # vertex 1: robustness  (bottom-right)
    [0.5,  np.sqrt(3) / 2],             # vertex 2: computation (top)
])

def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — already normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])    # (N, 3)
    xy  = pts @ _TRI                    # (N, 2)
    return xy[:, 0], xy[:, 1]


def _draw_ternary_frame(ax):
    """Draw the triangle boundary, gridlines, and vertex labels."""
    # Triangle boundary
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge (20 % steps)
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        label = f"{f:.0%}" if f in (0, 1) else f"{f:.0%}"
        # Bottom edge (economy → robustness): economy fraction = 1-f
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        # Left edge (economy → computation): computation fraction = f
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        # Right edge (robustness → computation): robustness fraction = 1-f
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"],
            ha="center", va="top", fontsize=6.5) # , fontweight="bold")
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"],
            ha="center", va="top", fontsize=6.5) # , fontweight="bold")
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"],
            ha="center", va="bottom", fontsize=6.5) # , fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
#  Build figure
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ────────────────────────────────────────────────────

for panel_id, cfg in scatter_panels.items():
    ax = axes[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,
        )

        if "kaysons" in dataset: 
            print(df_merged[x_col], df_merged[y_col])
            ax.scatter(
                df_merged[x_col], df_merged[y_col],
                color=COLOR_SCHEME[dataset],
                edgecolor="black",
                linewidth=0.25,
                s=10,
                alpha=1, 
                zorder=4,
            )
            
    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    ax.set_title(f"{panel_id}  {cfg['title']}", 
                 fontsize=7, # fontweight="bold", 
                 loc="left")
    ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
                fontsize=4.5, color="gray", ha="right", va="bottom")


# ── 2. Ternary panel E ───────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E  Economy - Robustness - Computation", 
               fontsize=7,
            #    fontweight="bold", 
               loc="left", 
               pad=8
               )


for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    # For economy: invert LR fraction so that LOW LR = HIGH economy
    v_economy     = 1.0 - raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    # Min-max normalize each axis across ALL datasets jointly
    # (on first pass, collect global ranges; here we normalize per-panel
    #  using the data present — you may want to precompute global ranges)
    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Shift to non-negative & normalize rows to sum=1
    for col_idx in range(3):
        col_min = stack[:, col_idx].min()
        col_max = stack[:, col_idx].max()
        rng = col_max - col_min
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - col_min) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"
    print(dataset)
    ax_t.scatter(
        x_cart, y_cart,
        color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # color="gray" if is_gnm else COLOR_SCHEME[dataset],
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.2 if not is_gnm else 0,
        s=5,
        alpha=0.1 if is_gnm else 1, # 0.45,
        zorder=1 if is_gnm else 3 if "kayson" in dataset else 2,
        rasterized=is_gnm,
    )
    
    if "kaysons" in dataset: 
        print(x_cart, y_cart)
        ax_t.scatter(
            x_cart, y_cart,
            color=COLOR_SCHEME[dataset],
            edgecolor="black",
            linewidth=0.25,
            s=10,
            alpha=1, 
            zorder=4,
        )

# ── 4. Save ──────────────────────────────────────────────────────────────────

plt.savefig(output_folder / f"trade_off_grid_4_with_triangle.pdf", bbox_inches="tight", dpi=200)
print(output_folder / f"trade_off_grid_4_with_triangle.pdf")
plt.show()


# In[ ]:


total_len = 0 
for dataset in datasets_to_look_at:
    df_merged = dict_with_all_datasets[dataset]
    total_len += len(df_merged)
    print(len(df_merged), dataset)
print(f"Total number of points plotted across all datasets: {total_len}")


# In[ ]:


# ── Trade-off grid: 2×3 established pairs ─────────────────────────────────────
from config import PROPERTY_NAMES

trade_off_panels = {
    "a": {
        "x": "global_efficiency",
        "y": "modularity",
        "title": "Integration vs. Segregation",
        "ref": "Sporns & Betzel, 2016",
    },
    "b": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "global_efficiency",
        "title": "Economy vs. Efficiency",
        "ref": "Bullmore & Sporns, 2012",
    },
    "c": {
        "x": "algebraic_connectivity_fiedler_value", # "algebraic_connectivity_nx",
        "y": "modularity",
        "title": "Robustness vs. Segregation",
        "ref": "see e.g. Sporns, 2013",
    },
    "d": {
        "x": "computational_capacity_memory_capacity_total",
        "y": "computational_capacity_nonlinear_capacity_total",
        "title": "Memory vs. Nonlinear Computation",
        "ref": "Dambre et al., 2012",
    },
    "e": {
        "x": "char_path_length",
        "y": "avg_clustering",
        "title": "Small-World Plane",
        "ref": "Watts & Strogatz, 1998",
    },
    "f": {
        "x": "global_efficiency",
        "y":  "algebraic_connectivity_fiedler_value", # "targeted_attack_robustness_rob_targeted_auc",
        "title": "Efficiency vs. Robustness",
        "ref": "Albert et al., 2000",
    },
}

INCLUDE_LEXIS = True # False # True

fig, axes_flat = plt.subplots(2, 3, figsize=viz.cm_to_inch((18, 12))) # , layout="constrained")
axes_grid = {k: ax for k, ax in zip(trade_off_panels.keys(), axes_flat.ravel())}

for panel_id, cfg in trade_off_panels.items():
    ax = axes_grid[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col = cfg["x"]
        y_col = cfg["y"]

        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray"
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,  # rasterize the 90k cloud for fast PDF
        )

    # Axis labels from PROPERTY_NAMES (fall back to raw name)
    ax.set_xlabel(PROPERTY_NAMES[cfg["x"]]) 
    ax.set_ylabel(PROPERTY_NAMES[cfg["y"]]) 
    ax.tick_params()

    ax.set_title(f"{panel_id} {cfg['title']}", fontsize=7.5, 
                 fontweight="bold", loc="left")

    # Reference annotation (bottom-right, small)
    # ax.annotate(
    #     cfg["ref"],
    #     xy=(1, 0), xycoords="axes fraction",
    #     # fontsize=4.5, 
    #     color="gray",
    #     ha="right", va="bottom",
    # )

# # Single shared legend from panel A
# handles, labels = axes_grid["A"].get_legend_handles_labels()
# fig.legend(
#     handles, labels,
#     # loc="outside lower", #  center",
#     ncol=min(len(labels), 3),
#     # fontsize=6,
#     # markerscale=1.5,
#     # frameon=False,
#     bbox_to_anchor=(0.5, 0), 
#     loc='upper center',
# )

plt.tight_layout()

plt.savefig(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf", bbox_inches="tight", dpi=300)
print(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf")
plt.show()


# In[ ]:


# ── Trade-off grid: 2×3 established pairs ─────────────────────────────────────
from config import PROPERTY_NAMES

trade_off_panels = {
    "A": {
        "x": "algebraic_connectivity_fiedler_value", 
        "y": "algebraic_connectivity_nx", 
        "title": "Robustnesses",
        "ref": "", # Sporns & Betzel, 2016",
    },
}

INCLUDE_LEXIS = True # False # True

fig, axes_flat = plt.subplots(2, 3, figsize=viz.cm_to_inch((18, 12))) # , layout="constrained")
axes_grid = {k: ax for k, ax in zip(trade_off_panels.keys(), axes_flat.ravel())}

for panel_id, cfg in trade_off_panels.items():
    ax = axes_grid[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col = cfg["x"]
        y_col = cfg["y"]

        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"
        if is_gnm: 
            continue

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray"
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,  # rasterize the 90k cloud for fast PDF
        )

    # Axis labels from PROPERTY_NAMES (fall back to raw name)
    ax.set_xlabel(PROPERTY_NAMES[cfg["x"]]) # .replace(" ", "\n")) # .get(cfg["x"], cfg["x"])) # , fontsize=7)
    ax.set_ylabel(PROPERTY_NAMES[cfg["y"]]) # .replace(" ", "\n")) # .get(cfg["y"], cfg["y"])) # , fontsize=7)
    ax.tick_params() # labelsize=6)

    # Panel letter + title
    # ax.set_title(f"{panel_id}  {cfg['title']}", # fontsize=7.5, fontweight="bold", 
    #             #  loc="left"
    #              )
    ax.set_title(f"{cfg['title']}", # fontsize=7.5, fontweight="bold", 
                #  loc="left"
                )
    break


# plt.tight_layout()

# plt.savefig(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf", bbox_inches="tight", dpi=300)
# print(output_folder / f"trade_off_grid_established{'_w_o_lexi' if INCLUDE_LEXIS else ''}.pdf")
# plt.show()


# In[ ]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].columns


# In[ ]:


"""
Novel trade-off figure: 4 scatter plots + 1 ternary triangle.

Layout (mosaic):
    A B E E
    C D E E 

A: Memory–nonlinear ratio vs. mean Ricci curvature
B: Total capacity vs. Lyapunov exponent
C: Targeted robustness vs. proportion LR connections
D: State dimensionality vs. modularity
E: Ternary triangle (wiring economy – robustness – computation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from pathlib import Path

INCLUDE_LEXIS = True

# ── Assumes these are already defined in your notebook: ──────────────────────
# dict_with_all_datasets, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES, viz
# If running standalone, import/define them above this point.

# ── Output ────────────────────────────────────────────────────────────────────
# output_folder = Path("...your path.../output/trade_off_scatters")
# output_folder.mkdir(exist_ok=True)

# ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        # "x": "ollivier_ricci_curvature_orc_mean",
        # "y": "computational_capacity_memory_nonlinear_ratio",
        "x": "proportion_long_range_connections_0.3956", 
        "y": "char_path_length", 
        "title": "LR to char_path_length",
        "note": "kaiser_2006_nonoptimal",
    },
    "B": {
        "x": "computational_capacity_lyapunov_exponent",
        "y": "mc_mean", # computational_capacity_total_capacity",
        "title": "Edge-of-Chaos",
        "note": "Büsing et al., 2010",
    },
    "C": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "targeted_attack_robustness_rob_targeted_auc",
        "title": "Economy vs. Robustness",
        "note": "Albert et al., 2000",
    },
    "D": {
        # "x": "targeted_attack_robustness_rob_random_half", # modularity",
        # "y": "targeted_attack_robustness_rob_targeted_half", # computational_capacity_state_dimensionality",
        "x": "targeted_attack_robustness_rob_random_auc", # modularity",
        "y": "targeted_attack_robustness_rob_targeted_auc", # computational_capacity_state_dimensionality",
        "title": "Robustness - Attacks", # Segregation vs. State Richness",
        "note": "novel",
    },
}

# ── Ternary axes (panel E) ────────────────────────────────────────────────────
ternary_cols = {
    "economy":     "proportion_long_range_connections_0.3956",
    "robustness":  "algebraic_connectivity_fiedler_value", # targeted_attack_robustness_rob_targeted_auc",
    "computation": "mc_mean", # computational_capacity_total_capacity",
}
ternary_labels = {
    "economy":     "Wiring economy\n(1 − f_LR)",
    "robustness":  "Robustness\n(targeted AUC)",
    "computation": "Computation\n(total capacity)",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Ternary helpers (no mpltern needed)
# ═════════════════════════════════════════════════════════════════════════════

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0,  0.0],                        # vertex 0: economy     (bottom-left)
    [1.0,  0.0],                        # vertex 1: robustness  (bottom-right)
    [0.5,  np.sqrt(3) / 2],             # vertex 2: computation (top)
])

def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — already normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])    # (N, 3)
    xy  = pts @ _TRI                    # (N, 2)
    return xy[:, 0], xy[:, 1]


def _draw_ternary_frame(ax):
    """Draw the triangle boundary, gridlines, and vertex labels."""
    # Triangle boundary
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge (20 % steps)
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        label = f"{f:.0%}" if f in (0, 1) else f"{f:.0%}"
        # Bottom edge (economy → robustness): economy fraction = 1-f
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        # Left edge (economy → computation): computation fraction = f
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        # Right edge (robustness → computation): robustness fraction = 1-f
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"],
            ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
#  Build figure
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ────────────────────────────────────────────────────

for panel_id, cfg in scatter_panels.items():
    ax = axes[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,
        )

        if "kaysons" in dataset: 
            print(df_merged[x_col], df_merged[y_col])
            ax.scatter(
                df_merged[x_col], df_merged[y_col],
                color=COLOR_SCHEME[dataset],
                edgecolor="black",
                linewidth=0.25,
                s=10,
                alpha=1, 
                zorder=4,
            )
            
    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, fontweight="bold", loc="left")
    ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
                fontsize=4.5, color="gray", ha="right", va="bottom")


# ── 2. Ternary panel E ───────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E  Economy – Robustness – Computation", fontsize=7,
               fontweight="bold", loc="left", pad=8)


for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    # For economy: invert LR fraction so that LOW LR = HIGH economy
    v_economy     = 1.0 - raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    # Min-max normalize each axis across ALL datasets jointly
    # (on first pass, collect global ranges; here we normalize per-panel
    #  using the data present — you may want to precompute global ranges)
    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Shift to non-negative & normalize rows to sum=1
    for col_idx in range(3):
        col_min = stack[:, col_idx].min()
        col_max = stack[:, col_idx].max()
        rng = col_max - col_min
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - col_min) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"
    print(dataset)
    ax_t.scatter(
        x_cart, y_cart,
        color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # color="gray" if is_gnm else COLOR_SCHEME[dataset],
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.2 if not is_gnm else 0,
        s=5,
        alpha=0.1 if is_gnm else 1, # 0.45,
        zorder=1 if is_gnm else 3 if "kayson" in dataset else 2,
        rasterized=is_gnm,
    )
    
    if "kaysons" in dataset: 
        print(x_cart, y_cart)
        ax_t.scatter(
            x_cart, y_cart,
            color=COLOR_SCHEME[dataset],
            edgecolor="black",
            linewidth=0.25,
            s=10,
            alpha=1, 
            zorder=4,
        )

# ── 4. Save ──────────────────────────────────────────────────────────────────

plt.savefig(output_folder / f"trade_off_grid_novel{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf", bbox_inches="tight", dpi=200)
print(output_folder / f"trade_off_grid_novel{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf")
plt.show()


# In[ ]:


"""
Novel trade-off figure: 4 scatter plots + 1 ternary triangle.

Layout (mosaic):
    A B E E
    C D E E 

A: Memory–nonlinear ratio vs. mean Ricci curvature
B: Total capacity vs. Lyapunov exponent
C: Targeted robustness vs. proportion LR connections
D: State dimensionality vs. modularity
E: Ternary triangle (wiring economy – robustness – computation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from pathlib import Path

INCLUDE_LEXIS = True

# ── Assumes these are already defined in your notebook: ──────────────────────
# dict_with_all_datasets, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES, viz
# If running standalone, import/define them above this point.

# ── Output ────────────────────────────────────────────────────────────────────
# output_folder = Path("...your path.../output/trade_off_scatters")
# output_folder.mkdir(exist_ok=True)

# ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        # "x": "ollivier_ricci_curvature_orc_mean",
        # "y": "computational_capacity_memory_nonlinear_ratio",
        "x": "proportion_long_range_connections_0.3956", 
        "y": "char_path_length", 
        "title": "LR to char_path_length",
        "note": "kaiser_2006_nonoptimal",
    },
    "B": {
        "x": "computational_capacity_lyapunov_exponent",
        "y": "mc_mean", # computational_capacity_total_capacity",
        "title": "Edge-of-Chaos",
        "note": "Büsing et al., 2010",
    },
    "C": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "targeted_attack_robustness_rob_targeted_auc",
        "title": "Economy vs. Robustness",
        "note": "Albert et al., 2000",
    },
    "D": {
        # "x": "targeted_attack_robustness_rob_random_half", # modularity",
        # "y": "targeted_attack_robustness_rob_targeted_half", # computational_capacity_state_dimensionality",
        "x": "targeted_attack_robustness_rob_random_auc", # modularity",
        "y": "targeted_attack_robustness_rob_targeted_auc", # computational_capacity_state_dimensionality",
        "title": "Robustness - Attacks", # Segregation vs. State Richness",
        "note": "novel",
    },
}

# ── Ternary axes (panel E) ────────────────────────────────────────────────────
ternary_cols = {
    "economy":     "proportion_long_range_connections_0.3956",
    "robustness":  "algebraic_connectivity_fiedler_value", # targeted_attack_robustness_rob_targeted_auc",
    "computation": "mc_mean", # computational_capacity_total_capacity",
}
ternary_labels = {
    "economy":     "Wiring economy\n(1 − f_LR)",
    "robustness":  "Robustness\n(targeted AUC)",
    "computation": "Computation\n(total capacity)",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Ternary helpers (no mpltern needed)
# ═════════════════════════════════════════════════════════════════════════════

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0,  0.0],                        # vertex 0: economy     (bottom-left)
    [1.0,  0.0],                        # vertex 1: robustness  (bottom-right)
    [0.5,  np.sqrt(3) / 2],             # vertex 2: computation (top)
])

def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — already normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])    # (N, 3)
    xy  = pts @ _TRI                    # (N, 2)
    return xy[:, 0], xy[:, 1]


def _draw_ternary_frame(ax):
    """Draw the triangle boundary, gridlines, and vertex labels."""
    # Triangle boundary
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge (20 % steps)
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        label = f"{f:.0%}" if f in (0, 1) else f"{f:.0%}"
        # Bottom edge (economy → robustness): economy fraction = 1-f
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        # Left edge (economy → computation): computation fraction = f
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        # Right edge (robustness → computation): robustness fraction = 1-f
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"],
            ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
#  Build figure
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ────────────────────────────────────────────────────

for panel_id, cfg in scatter_panels.items():
    ax = axes[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,
        )

        if "kaysons" in dataset: 
            print(df_merged[x_col], df_merged[y_col])
            ax.scatter(
                df_merged[x_col], df_merged[y_col],
                color=COLOR_SCHEME[dataset],
                edgecolor="black",
                linewidth=0.25,
                s=10,
                alpha=1, 
                zorder=4,
            )
            
    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, fontweight="bold", loc="left")
    ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
                fontsize=4.5, color="gray", ha="right", va="bottom")


# ── 2. Ternary panel E ───────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E  Economy – Robustness – Computation", fontsize=7,
               fontweight="bold", loc="left", pad=8)


for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    # For economy: invert LR fraction so that LOW LR = HIGH economy
    v_economy     = 1.0 - raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    # Min-max normalize each axis across ALL datasets jointly
    # (on first pass, collect global ranges; here we normalize per-panel
    #  using the data present — you may want to precompute global ranges)
    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Shift to non-negative & normalize rows to sum=1
    for col_idx in range(3):
        col_min = stack[:, col_idx].min()
        col_max = stack[:, col_idx].max()
        rng = col_max - col_min
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - col_min) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"
    print(dataset)
    ax_t.scatter(
        x_cart, y_cart,
        color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # color="gray" if is_gnm else COLOR_SCHEME[dataset],
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.2 if not is_gnm else 0,
        s=5,
        alpha=0.1 if is_gnm else 1, # 0.45,
        zorder=1 if is_gnm else 3 if "kayson" in dataset else 2,
        rasterized=is_gnm,
    )
    
    if "kaysons" in dataset: 
        print(x_cart, y_cart)
        ax_t.scatter(
            x_cart, y_cart,
            color=COLOR_SCHEME[dataset],
            edgecolor="black",
            linewidth=0.25,
            s=10,
            alpha=1, 
            zorder=4,
        )

# ── 4. Save ──────────────────────────────────────────────────────────────────

plt.savefig(output_folder / f"trade_off_grid_established_4_triangle.pdf", bbox_inches="tight", dpi=200)
print(output_folder / f"trade_off_grid_established_4_triangle.pdf")
plt.show()


# In[ ]:


"""
Novel trade-off figure: 4 scatter plots + 1 ternary triangle.

Layout (mosaic):
    A B E E
    C D E E 

A: Memory–nonlinear ratio vs. mean Ricci curvature
B: Total capacity vs. Lyapunov exponent
C: Targeted robustness vs. proportion LR connections
D: State dimensionality vs. modularity
E: Ternary triangle (wiring economy – robustness – computation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from pathlib import Path

INCLUDE_LEXIS = True

# ── Assumes these are already defined in your notebook: ──────────────────────
# dict_with_all_datasets, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES, viz
# If running standalone, import/define them above this point.

# ── Output ────────────────────────────────────────────────────────────────────
# output_folder = Path("...your path.../output/trade_off_scatters")
# output_folder.mkdir(exist_ok=True)

# ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        # "x": "ollivier_ricci_curvature_orc_mean",
        # "y": "computational_capacity_memory_nonlinear_ratio",
        "x": "proportion_long_range_connections_0.3956", 
        "y": "char_path_length", 
        "title": "LR to char_path_length",
        "note": "kaiser_2006_nonoptimal",
    },
    "B": {
        "x": "computational_capacity_lyapunov_exponent",
        "y": "mc_mean", # computational_capacity_total_capacity",
        "title": "Edge-of-Chaos",
        "note": "Büsing et al., 2010",
    },
    "C": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "targeted_attack_robustness_rob_targeted_auc",
        "title": "Economy vs. Robustness",
        "note": "Albert et al., 2000",
    },
    "D": {
        "x": "targeted_attack_robustness_rob_random_half", # modularity",
        "y": "targeted_attack_robustness_rob_targeted_half", # computational_capacity_state_dimensionality",
        "title": "Robustness - Attacks", # Segregation vs. State Richness",
        "note": "novel",
    },
}

# ── Ternary axes (panel E) ────────────────────────────────────────────────────
ternary_cols = {
    "economy":     "proportion_long_range_connections_0.3956",
    "robustness":  "algebraic_connectivity_fiedler_value", # targeted_attack_robustness_rob_targeted_auc",
    "computation": "mc_mean", # computational_capacity_total_capacity",
}
ternary_labels = {
    "economy":     "Wiring economy\n(1 − f_LR)",
    "robustness":  "Robustness\n(targeted AUC)",
    "computation": "Computation\n(total capacity)",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Ternary helpers (no mpltern needed)
# ═════════════════════════════════════════════════════════════════════════════

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0,  0.0],                        # vertex 0: economy     (bottom-left)
    [1.0,  0.0],                        # vertex 1: robustness  (bottom-right)
    [0.5,  np.sqrt(3) / 2],             # vertex 2: computation (top)
])

def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — already normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])    # (N, 3)
    xy  = pts @ _TRI                    # (N, 2)
    return xy[:, 0], xy[:, 1]


def _draw_ternary_frame(ax):
    """Draw the triangle boundary, gridlines, and vertex labels."""
    # Triangle boundary
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge (20 % steps)
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        label = f"{f:.0%}" if f in (0, 1) else f"{f:.0%}"
        # Bottom edge (economy → robustness): economy fraction = 1-f
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        # Left edge (economy → computation): computation fraction = f
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        # Right edge (robustness → computation): robustness fraction = 1-f
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"],
            ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
#  Build figure
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ────────────────────────────────────────────────────

for panel_id, cfg in scatter_panels.items():
    ax = axes[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,
        )

    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, fontweight="bold", loc="left")
    ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
                fontsize=4.5, color="gray", ha="right", va="bottom")


# ── 2. Ternary panel E ───────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E  Economy – Robustness – Computation", fontsize=7,
               fontweight="bold", loc="left", pad=8)


for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    # For economy: invert LR fraction so that LOW LR = HIGH economy
    v_economy     = 1.0 - raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    # Min-max normalize each axis across ALL datasets jointly
    # (on first pass, collect global ranges; here we normalize per-panel
    #  using the data present — you may want to precompute global ranges)
    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Shift to non-negative & normalize rows to sum=1
    for col_idx in range(3):
        col_min = stack[:, col_idx].min()
        col_max = stack[:, col_idx].max()
        rng = col_max - col_min
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - col_min) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"
    print(dataset)
    ax_t.scatter(
        x_cart, y_cart,
        color="white" if not is_gnm else combined_colors, #  if is_gnm else COLOR_SCHEME[dataset], # color="gray" if is_gnm else COLOR_SCHEME[dataset],
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.2 if not is_gnm else 0,
        s=5,
        alpha=1, # 0.1 if is_gnm else 1, # 0.45,
        zorder=1 if is_gnm else 3 if "kayson" in dataset else 2,
        rasterized=is_gnm,
    )
    
    ax_t.scatter(
        x_cart, y_cart,
        color=COLOR_SCHEME[dataset] if not is_gnm else "none", # color="gray" if is_gnm else COLOR_SCHEME[dataset],
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.05 if not is_gnm else 0,
        s=5,
        alpha=0.3, # if is_gnm else 1, # 0.45,
        zorder=1 if is_gnm else 3 if "kayson" in dataset else 2,
        rasterized=is_gnm,
    )
        
    # ax_t.scatter(
    #     x_cart, y_cart,
    #     color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # color="gray" if is_gnm else COLOR_SCHEME[dataset],
    #     edgecolor="black" if not is_gnm else "none",
    #     linewidth=0.2 if not is_gnm else 0,
    #     s=5,
    #     alpha=0.1 if is_gnm else 1, # 0.45,
    #     zorder=1 if is_gnm else 3 if "kayson" in dataset else 2,
    #     rasterized=is_gnm,
    # )
    

# ── 4. Save ──────────────────────────────────────────────────────────────────

# plt.savefig(output_folder / f"trade_off_grid_novel{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf", bbox_inches="tight", dpi=200)
# print(output_folder / f"trade_off_grid_novel{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf")
# plt.show()


# In[ ]:


"""
Novel trade-off figure: 4 scatter plots + 1 ternary triangle.

Layout (mosaic):
    A B E E
    C D E E 

A: Memory–nonlinear ratio vs. mean Ricci curvature
B: Total capacity vs. Lyapunov exponent
C: Targeted robustness vs. proportion LR connections
D: State dimensionality vs. modularity
E: Ternary triangle (wiring economy – robustness – computation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from pathlib import Path

INCLUDE_LEXIS = True

# ── Assumes these are already defined in your notebook: ──────────────────────
# dict_with_all_datasets, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES, viz
# If running standalone, import/define them above this point.

# ── Output ────────────────────────────────────────────────────────────────────
# output_folder = Path("...your path.../output/trade_off_scatters")
# output_folder.mkdir(exist_ok=True)

# ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        "x": "ollivier_ricci_curvature_orc_mean",
        "y": "computational_capacity_memory_nonlinear_ratio",
        "title": "Geometry → Computation",
        "note": "novel",
    },
    "B": {
        "x": "computational_capacity_lyapunov_exponent",
        "y": "computational_capacity_total_capacity",
        "title": "Edge-of-Chaos",
        "note": "Büsing et al., 2010",
    },
    "C": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "targeted_attack_robustness_rob_targeted_auc",
        "title": "Economy vs. Robustness",
        "note": "Albert et al., 2000",
    },
    "D": {
        "x": "modularity",
        "y": "computational_capacity_state_dimensionality",
        "title": "Segregation vs. State Richness",
        "note": "novel",
    },
}

# ── Ternary axes (panel E) ────────────────────────────────────────────────────
ternary_cols = {
    "economy":     "proportion_long_range_connections_0.3956",
    "robustness":  "targeted_attack_robustness_rob_targeted_auc",
    "computation": "computational_capacity_total_capacity",
}
ternary_labels = {
    "economy":     "Wiring economy\n(1 − f_LR)",
    "robustness":  "Robustness\n(targeted AUC)",
    "computation": "Computation\n(total capacity)",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Ternary helpers (no mpltern needed)
# ═════════════════════════════════════════════════════════════════════════════

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0,  0.0],                        # vertex 0: economy     (bottom-left)
    [1.0,  0.0],                        # vertex 1: robustness  (bottom-right)
    [0.5,  np.sqrt(3) / 2],             # vertex 2: computation (top)
])

def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — already normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])    # (N, 3)
    xy  = pts @ _TRI                    # (N, 2)
    return xy[:, 0], xy[:, 1]


def _draw_ternary_frame(ax):
    """Draw the triangle boundary, gridlines, and vertex labels."""
    # Triangle boundary
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge (20 % steps)
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        label = f"{f:.0%}" if f in (0, 1) else f"{f:.0%}"
        # Bottom edge (economy → robustness): economy fraction = 1-f
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        # Left edge (economy → computation): computation fraction = f
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        # Right edge (robustness → computation): robustness fraction = 1-f
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"],
            ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
#  Build figure
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ────────────────────────────────────────────────────

for panel_id, cfg in scatter_panels.items():
    ax = axes[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,
        )

    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, fontweight="bold", loc="left")
    ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
                fontsize=4.5, color="gray", ha="right", va="bottom")


# ── 2. Ternary panel E ───────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E  Economy – Robustness – Computation", fontsize=7,
               fontweight="bold", loc="left", pad=8)

for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    # For economy: invert LR fraction so that LOW LR = HIGH economy
    v_economy     = 1.0 - raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    # Min-max normalize each axis across ALL datasets jointly
    # (on first pass, collect global ranges; here we normalize per-panel
    #  using the data present — you may want to precompute global ranges)
    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Shift to non-negative & normalize rows to sum=1
    for col_idx in range(3):
        col_min = stack[:, col_idx].min()
        col_max = stack[:, col_idx].max()
        rng = col_max - col_min
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - col_min) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

    ax_t.scatter(
        x_cart, y_cart,
        color="gray" if is_gnm else COLOR_SCHEME[dataset],
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.2 if not is_gnm else 0,
        s=4,
        alpha=0.1 if is_gnm else 0.45,
        zorder=1 if is_gnm else 2,
        rasterized=is_gnm,
    )


# ── 3. Shared legend ─────────────────────────────────────────────────────────

handles, labels = axes["A"].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="outside lower center",
    ncol=min(len(labels), 6),
    fontsize=5.5,
    markerscale=1.8,
    frameon=False,
    handletextpad=0.3,
    columnspacing=1.0,
)

# ── 4. Save ──────────────────────────────────────────────────────────────────

plt.savefig(output_folder / f"trade_off_grid_novel{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms_new_style.pdf", bbox_inches="tight", dpi=200)
print(output_folder / f"trade_off_grid_novel{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms_new_style.pdf")
plt.show()


# In[ ]:


"""
Novel trade-off figure: 4 scatter plots + 1 ternary triangle.

Layout (mosaic):
    A B E E
    C D E E 

A: Memory–nonlinear ratio vs. mean Ricci curvature
B: Total capacity vs. Lyapunov exponent
C: Targeted robustness vs. proportion LR connections
D: State dimensionality vs. modularity
E: Ternary triangle (wiring economy – robustness – computation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from pathlib import Path

INCLUDE_LEXIS = True

# ── Assumes these are already defined in your notebook: ──────────────────────
# dict_with_all_datasets, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES, viz
# If running standalone, import/define them above this point.

# ── Output ────────────────────────────────────────────────────────────────────
# output_folder = Path("...your path.../output/trade_off_scatters")
# output_folder.mkdir(exist_ok=True)

# ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        "x": "modularity", # "ollivier_ricci_curvature_orc_mean",
        "y": "nct_control_std", # "computational_capacity_memory_nonlinear_ratio",
        "title": "Segregation vs. Influence",
        "note": "novel",
    },
    "B": {
        "x": "computational_capacity_lyapunov_exponent",
        "y": "computational_capacity_total_capacity",
        "title": "Edge-of-Chaos",
        "note": "Büsing et al., 2010",
    },
    "C": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "targeted_attack_robustness_rob_targeted_auc",
        "title": "Economy vs. Robustness",
        "note": "Albert et al., 2000",
    },
    "D": {
        "x": "modularity",
        "y": "computational_capacity_state_dimensionality",
        "title": "Segregation vs. State Richness",
        "note": "novel",
    },
}

# ── Ternary axes (panel E) ────────────────────────────────────────────────────
ternary_cols = {
    "economy":     "proportion_long_range_connections_0.3956",
    "robustness":  "targeted_attack_robustness_rob_targeted_auc",
    "computation": "computational_capacity_total_capacity",
}
ternary_labels = {
    "economy":     "Wiring economy\n(1 − f_LR)",
    "robustness":  "Robustness\n(targeted AUC)",
    "computation": "Computation\n(total capacity)",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Ternary helpers (no mpltern needed)
# ═════════════════════════════════════════════════════════════════════════════

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0,  0.0],                        # vertex 0: economy     (bottom-left)
    [1.0,  0.0],                        # vertex 1: robustness  (bottom-right)
    [0.5,  np.sqrt(3) / 2],             # vertex 2: computation (top)
])

def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — already normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])    # (N, 3)
    xy  = pts @ _TRI                    # (N, 2)
    return xy[:, 0], xy[:, 1]


def _draw_ternary_frame(ax):
    """Draw the triangle boundary, gridlines, and vertex labels."""
    # Triangle boundary
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge (20 % steps)
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        label = f"{f:.0%}" if f in (0, 1) else f"{f:.0%}"
        # Bottom edge (economy → robustness): economy fraction = 1-f
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        # Left edge (economy → computation): computation fraction = f
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        # Right edge (robustness → computation): robustness fraction = 1-f
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"],
            ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
#  Build figure
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ────────────────────────────────────────────────────

for panel_id, cfg in scatter_panels.items():
    ax = axes[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,
        )

    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, fontweight="bold", loc="left")
    ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
                fontsize=4.5, color="gray", ha="right", va="bottom")


# ── 2. Ternary panel E ───────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E  Economy – Robustness – Computation", fontsize=7,
               fontweight="bold", loc="left", pad=8)

for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    # For economy: invert LR fraction so that LOW LR = HIGH economy
    v_economy     = 1.0 - raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    # Min-max normalize each axis across ALL datasets jointly
    # (on first pass, collect global ranges; here we normalize per-panel
    #  using the data present — you may want to precompute global ranges)
    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Shift to non-negative & normalize rows to sum=1
    for col_idx in range(3):
        col_min = stack[:, col_idx].min()
        col_max = stack[:, col_idx].max()
        rng = col_max - col_min
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - col_min) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

    ax_t.scatter(
        x_cart, y_cart,
        color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" color="gray" if is_gnm else COLOR_SCHEME[dataset],
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.2 if not is_gnm else 0,
        s=4,
        alpha=0.1 if is_gnm else 0.45,
        zorder=1 if is_gnm else 2,
        rasterized=is_gnm,
    )


# ── 3. Shared legend ─────────────────────────────────────────────────────────

handles, labels = axes["A"].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="outside lower center",
    ncol=min(len(labels), 6),
    fontsize=5.5,
    markerscale=1.8,
    frameon=False,
    handletextpad=0.3,
    columnspacing=1.0,
)

# ── 4. Save ──────────────────────────────────────────────────────────────────

plt.savefig(output_folder / f"trade_off_grid_novel_2{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf", bbox_inches="tight", dpi=200)
print(output_folder / f"trade_off_grid_novel_2{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf")
plt.show()


# In[ ]:


"""
Novel trade-off figure: 4 scatter plots + 1 ternary triangle.

Layout (mosaic):
    A B E E
    C D E E 

A: Memory–nonlinear ratio vs. mean Ricci curvature
B: Total capacity vs. Lyapunov exponent
C: Targeted robustness vs. proportion LR connections
D: State dimensionality vs. modularity
E: Ternary triangle (wiring economy – robustness – computation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from pathlib import Path

INCLUDE_LEXIS = True

# ── Assumes these are already defined in your notebook: ──────────────────────
# dict_with_all_datasets, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES, viz
# If running standalone, import/define them above this point.

# ── Output ────────────────────────────────────────────────────────────────────
# output_folder = Path("...your path.../output/trade_off_scatters")
# output_folder.mkdir(exist_ok=True)

# ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        "x": "modularity", # "ollivier_ricci_curvature_orc_mean",
        "y": "nct_control_std", # "computational_capacity_memory_nonlinear_ratio",
        "title": "Segregation vs. Influence",
        "note": "novel",
    },
    "B": {
        "x": "computational_capacity_lyapunov_exponent",
        "y": "computational_capacity_total_capacity",
        "title": "Edge-of-Chaos",
        "note": "Büsing et al., 2010",
    },
    "C": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "targeted_attack_robustness_rob_targeted_auc",
        "title": "Economy vs. Robustness",
        "note": "Albert et al., 2000",
    },
    "D": {
        "x": "modularity",
        "y": "computational_capacity_state_dimensionality",
        "title": "Segregation vs. State Richness",
        "note": "novel",
    },
}

# ── Ternary axes (panel E) ────────────────────────────────────────────────────
ternary_cols = {
    "economy":  "global_efficiency", # corr too highly with omega
    "robustness": "diffusion_efficiency",
    "computation": "propagation_efficiency",

    # "economy":     "proportion_long_range_connections_0.3956",
    # "robustness":  "targeted_attack_robustness_rob_targeted_auc",
    # "computation": "computational_capacity_total_capacity",
}
ternary_labels = {
    "economy":     "Global efficiency\n(corr with omega)",
    "robustness":  "Diffusion efficiency",
    "computation": "Propagation efficiency",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Ternary helpers (no mpltern needed)
# ═════════════════════════════════════════════════════════════════════════════

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0,  0.0],                        # vertex 0: economy     (bottom-left)
    [1.0,  0.0],                        # vertex 1: robustness  (bottom-right)
    [0.5,  np.sqrt(3) / 2],             # vertex 2: computation (top)
])

def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — already normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])    # (N, 3)
    xy  = pts @ _TRI                    # (N, 2)
    return xy[:, 0], xy[:, 1]


def _draw_ternary_frame(ax):
    """Draw the triangle boundary, gridlines, and vertex labels."""
    # Triangle boundary
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge (20 % steps)
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        label = f"{f:.0%}" if f in (0, 1) else f"{f:.0%}"
        # Bottom edge (economy → robustness): economy fraction = 1-f
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        # Left edge (economy → computation): computation fraction = f
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        # Right edge (robustness → computation): robustness fraction = 1-f
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"],
            ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
#  Build figure
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ────────────────────────────────────────────────────

# for panel_id, cfg in scatter_panels.items():
#     ax = axes[panel_id]

#     for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
#         df_merged = dict_with_all_datasets[dataset]
        
#         if (not INCLUDE_LEXIS) and ("lexi" in dataset):
#             print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
#             continue
        
#         x_col, y_col = cfg["x"], cfg["y"]
#         if x_col not in df_merged.columns or y_col not in df_merged.columns:
#             continue

#         is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

#         ax.scatter(
#             df_merged[x_col],
#             df_merged[y_col],
#             color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
#             edgecolor="black" if not is_gnm else "none",
#             linewidth=0.25 if not is_gnm else 0,
#             s=5,
#             alpha=0.15 if is_gnm else 0.4,
#             label=LABEL_MAP[dataset] if panel_id == "A" else None,
#             zorder=1 if is_gnm else 2,
#             rasterized=is_gnm,
#         )

#     ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
#     ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
#     ax.tick_params(labelsize=5.5)

#     ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, fontweight="bold", loc="left")
#     ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
#                 fontsize=4.5, color="gray", ha="right", va="bottom")


# ── 2. Ternary panel E ───────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E Efficiencies", # , fontsize=7,
               fontweight="bold", loc="left", pad=8)

for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    v_economy     = raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    # Min-max normalize each axis across ALL datasets jointly
    # (on first pass, collect global ranges; here we normalize per-panel
    #  using the data present — you may want to precompute global ranges)
    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Shift to non-negative & normalize rows to sum=1
    for col_idx in range(3):
        col_min = stack[:, col_idx].min()
        col_max = stack[:, col_idx].max()
        rng = col_max - col_min
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - col_min) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums
    # stack = stack + 1e-12

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

    ax_t.scatter(
        x_cart, y_cart,
        # color="gray" if is_gnm else COLOR_SCHEME[dataset],
        color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.2 if not is_gnm else 0,
        s=4,
        alpha=0.1 if is_gnm else 0.45,
        zorder=1 if is_gnm else 2,
        rasterized=is_gnm,
    )


# ── 3. Shared legend ─────────────────────────────────────────────────────────

# handles, labels = axes["A"].get_legend_handles_labels()
# fig.legend(
#     handles, labels,
#     loc="outside lower center",
#     ncol=min(len(labels), 6),
#     fontsize=5.5,
#     markerscale=1.8,
#     frameon=False,
#     handletextpad=0.3,
#     columnspacing=1.0,
# )

# # ── 4. Save ──────────────────────────────────────────────────────────────────

plt.savefig(output_folder / f"trade_off_grid_novel_2_efficiencies{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf", bbox_inches="tight", dpi=200)
print(output_folder / f"trade_off_grid_novel_2_efficiencies{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf")
# plt.show()


# In[ ]:





# In[ ]:


# # ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        "x": "propagation_efficiency", # "ollivier_ricci_curvature_orc_mean",
        "y": "mc_mean", # computational_capacity_memory_capacity_total", # "computational_capacity_memory_nonlinear_ratio",
        "title": "Kayson - mc_mean",
        "note": "Kayson - mc_mean",
    },
    "B": {
        "x": "propagation_efficiency",
        "y": "computational_capacity_nonlinear_capacity_total", # computational_capacity_total_capacity",
        "title": "Kayson - nonlinear capacity",
        "note": "Kayson - nonlinear capacity",
    },
    "C": {
        "x": "propagation_efficiency",
        "y": "mc_nonlinear_original_mc_mean",
        "title": "Kayson - mc_nonlinear",
        "note": "Kayson - mc_nonlinear",
    },
    "D": {
        "x": "propagation_efficiency",
        "y": "computational_capacity_memory_capacity_total",
        "title": "Kayson - memory capacity",
        "note": "Kayson - memory capacity",
    },
    
    "E": {
        "x": "propagation_efficiency",
        "y": "computational_capacity_memory_capacity_total",
        "title": "Kayson - memory capacity",
        "note": "Kayson - memory capacity",
    },
}



# In[ ]:


for a,val in axes.items(): 
    print(a,val)
    val.plot(df_merged[x_col])


# In[ ]:


fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

for (label_ax, ax), (label_cfg, cfg) in zip(axes.items(), scatter_panels.items()): 
    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset].copy()
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        if cfg["x"] == "propagation_efficiency": 
            df_merged[x_col] = 1 / df_merged[x_col]
            print(dataset, is_gnm)
            
        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,
        )

    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, fontweight="bold", loc="left")
    ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
                fontsize=4.5, color="gray", ha="right", va="bottom")


# In[ ]:


"""
Novel trade-off figure: 4 scatter plots + 1 ternary triangle.

Layout (mosaic):
    A B E E
    C D E E 

A: Memory–nonlinear ratio vs. mean Ricci curvature
B: Total capacity vs. Lyapunov exponent
C: Targeted robustness vs. proportion LR connections
D: State dimensionality vs. modularity
E: Ternary triangle (wiring economy – robustness – computation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from pathlib import Path

INCLUDE_LEXIS = True

# ── Assumes these are already defined in your notebook: ──────────────────────
# dict_with_all_datasets, COLOR_SCHEME, LABEL_MAP, PROPERTY_NAMES, viz
# If running standalone, import/define them above this point.

# ── Output ────────────────────────────────────────────────────────────────────
# output_folder = Path("...your path.../output/trade_off_scatters")
# output_folder.mkdir(exist_ok=True)

# ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        "x": "propagation_efficiency", # "ollivier_ricci_curvature_orc_mean",
        "y": "ipc_ipc_deg1_mean", # mc_mean", # computational_capacity_memory_capacity_total", # "computational_capacity_memory_nonlinear_ratio",
        "title": "Kayson - mc_mean",
        "note": "Kayson - mc_mean",
    },
    "B": {
        "x": "propagation_efficiency",
        "y": "computational_capacity_nonlinear_capacity_total", # computational_capacity_total_capacity",
        "title": "Kayson - nonlinear capacity",
        "note": "Kayson - nonlinear capacity",
    },
    "C": {
        "x": "propagation_efficiency",
        "y": "ipc_ipc_deg2_mean", # mc_nonlinear_original_mc_mean",
        "title": "Kayson - mc_nonlinear",
        "note": "Kayson - mc_nonlinear",
    },
    "D": {
        "x": "propagation_efficiency",
        "y": "computational_capacity_memory_capacity_total",
        "title": "Kayson - memory capacity",
        "note": "Kayson - memory capacity",
    },
}

# ── Ternary axes (panel E) ────────────────────────────────────────────────────
ternary_cols = {
    "economy":  "global_efficiency", # corr too highly with omega
    "robustness": "diffusion_efficiency",
    "computation": "propagation_efficiency",

    # "economy":     "proportion_long_range_connections_0.3956",
    # "robustness":  "targeted_attack_robustness_rob_targeted_auc",
    # "computation": "computational_capacity_total_capacity",
}
ternary_labels = {
    "economy":     "Global efficiency\n(corr with omega)",
    "robustness":  "Diffusion efficiency",
    "computation": "Propagation efficiency",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Ternary helpers (no mpltern needed)
# ═════════════════════════════════════════════════════════════════════════════

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0,  0.0],                        # vertex 0: economy     (bottom-left)
    [1.0,  0.0],                        # vertex 1: robustness  (bottom-right)
    [0.5,  np.sqrt(3) / 2],             # vertex 2: computation (top)
])

def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — already normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])    # (N, 3)
    xy  = pts @ _TRI                    # (N, 2)
    return xy[:, 0], xy[:, 1]


def _draw_ternary_frame(ax):
    """Draw the triangle boundary, gridlines, and vertex labels."""
    # Triangle boundary
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge (20 % steps)
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        label = f"{f:.0%}" if f in (0, 1) else f"{f:.0%}"
        # Bottom edge (economy → robustness): economy fraction = 1-f
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        # Left edge (economy → computation): computation fraction = f
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        # Right edge (robustness → computation): robustness fraction = 1-f
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"],
            ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
#  Build figure
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ────────────────────────────────────────────────────

for panel_id, cfg in scatter_panels.items():
    ax = axes[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,
        )

    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, fontweight="bold", loc="left")
    ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
                fontsize=4.5, color="gray", ha="right", va="bottom")


# ── 2. Ternary panel E ───────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E Efficiencies", # , fontsize=7,
               fontweight="bold", loc="left", pad=8)

for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    v_economy     = raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    # Min-max normalize each axis across ALL datasets jointly
    # (on first pass, collect global ranges; here we normalize per-panel
    #  using the data present — you may want to precompute global ranges)
    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Shift to non-negative & normalize rows to sum=1
    for col_idx in range(3):
        col_min = stack[:, col_idx].min()
        col_max = stack[:, col_idx].max()
        rng = col_max - col_min
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - col_min) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums
    # stack = stack + 1e-12

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

    ax_t.scatter(
        x_cart, y_cart,
        # color="gray" if is_gnm else COLOR_SCHEME[dataset],
        color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.2 if not is_gnm else 0,
        s=4,
        alpha=0.1 if is_gnm else 0.45,
        zorder=1 if is_gnm else 2,
        rasterized=is_gnm,
    )


# ── 3. Shared legend ─────────────────────────────────────────────────────────

# handles, labels = axes["A"].get_legend_handles_labels()
# fig.legend(
#     handles, labels,
#     loc="outside lower center",
#     ncol=min(len(labels), 6),
#     fontsize=5.5,
#     markerscale=1.8,
#     frameon=False,
#     handletextpad=0.3,
#     columnspacing=1.0,
# )

# # ── 4. Save ──────────────────────────────────────────────────────────────────

plt.savefig(output_folder / f"trade_off_grid_novel_2_efficiencies{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf", bbox_inches="tight", dpi=200)
print(output_folder / f"trade_off_grid_novel_2_efficiencies{'_w_o_lexi' if INCLUDE_LEXIS else ''}_with_gnms.pdf")
# # plt.show()


# In[ ]:


"""
Novel trade-off figure: 4 scatter plots + 1 ternary triangle.

Layout (mosaic):
    A B E E
    C D E E 

A: Memory–nonlinear ratio vs. mean Ricci curvature
B: Total capacity vs. Lyapunov exponent
C: Targeted robustness vs. proportion LR connections
D: State dimensionality vs. modularity
E: Ternary triangle (wiring economy – robustness – computation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from pathlib import Path

INCLUDE_LEXIS = True

# ── Panel definitions (4 scatter plots) ───────────────────────────────────────
scatter_panels = {
    "A": {
        "x": "ollivier_ricci_curvature_orc_mean",
        "y": "computational_capacity_memory_nonlinear_ratio",
        "title": "Geometry → Computation",
        "note": "novel",
    },
    "B": {
        "x": "computational_capacity_lyapunov_exponent",
        "y": "computational_capacity_total_capacity",
        "title": "Edge-of-Chaos",
        "note": "Büsing et al., 2010",
    },
    "C": {
        "x": "proportion_long_range_connections_0.3956",
        "y": "targeted_attack_robustness_rob_targeted_auc",
        "title": "Economy vs. Robustness",
        "note": "Albert et al., 2000",
    },
    "D": {
        "x": "modularity",
        "y": "computational_capacity_state_dimensionality",
        "title": "Segregation vs. State Richness",
        "note": "novel",
    },
}

# ── Ternary axes (panel E) ────────────────────────────────────────────────────
ternary_cols = {
    "economy":     "proportion_long_range_connections_0.3956",
    "robustness":  "targeted_attack_robustness_rob_targeted_auc",
    "computation": "computational_capacity_total_capacity",
}
ternary_labels = {
    "economy":     "Wiring economy\n(1 − f_LR)",
    "robustness":  "Robustness\n(targeted AUC)",
    "computation": "Computation\n(total capacity)",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Ternary helpers (no mpltern needed)
# ═════════════════════════════════════════════════════════════════════════════

# Vertices of equilateral triangle (bottom-left, bottom-right, top)
_TRI = np.array([
    [0.0,  0.0],                        # vertex 0: economy     (bottom-left)
    [1.0,  0.0],                        # vertex 1: robustness  (bottom-right)
    [0.5,  np.sqrt(3) / 2],             # vertex 2: computation (top)
])

def _bary_to_cart(a, b, c):
    """Convert barycentric (a, b, c) — already normalized to sum=1 — to 2-D Cartesian."""
    pts = np.column_stack([a, b, c])    # (N, 3)
    xy  = pts @ _TRI                    # (N, 2)
    return xy[:, 0], xy[:, 1]


def _draw_ternary_frame(ax):
    """Draw the triangle boundary, gridlines, and vertex labels."""
    # Triangle boundary
    tri_closed = np.vstack([_TRI, _TRI[0]])
    ax.plot(tri_closed[:, 0], tri_closed[:, 1], color="black", linewidth=0.8, zorder=5)

    # Gridlines at 20 % intervals
    for f in [0.2, 0.4, 0.6, 0.8]:
        for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
            p0 = (1 - f) * _TRI[i] + f * _TRI[j]
            p1 = (1 - f) * _TRI[i] + f * _TRI[k]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="#d0d0d0", linewidth=0.4, zorder=1)

    # Tick labels along each edge (20 % steps)
    fontsize = 5
    offset = 0.04
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        label = f"{f:.0%}" if f in (0, 1) else f"{f:.0%}"
        # Bottom edge (economy → robustness): economy fraction = 1-f
        px = (1 - f) * _TRI[0] + f * _TRI[1]
        ax.text(px[0], px[1] - offset, f"{1-f:.0%}",
                ha="center", va="top", fontsize=fontsize, color="#888888")
        # Left edge (economy → computation): computation fraction = f
        pl = (1 - f) * _TRI[0] + f * _TRI[2]
        ax.text(pl[0] - offset * 0.8, pl[1], f"{f:.0%}",
                ha="right", va="center", fontsize=fontsize, color="#888888", rotation=60)
        # Right edge (robustness → computation): robustness fraction = 1-f
        pr = (1 - f) * _TRI[1] + f * _TRI[2]
        ax.text(pr[0] + offset * 0.8, pr[1], f"{1-f:.0%}",
                ha="left", va="center", fontsize=fontsize, color="#888888", rotation=-60)

    # Vertex labels
    label_offset = 0.09
    ax.text(_TRI[0][0], _TRI[0][1] - label_offset * 1.8,
            ternary_labels["economy"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[1][0], _TRI[1][1] - label_offset * 1.8,
            ternary_labels["robustness"],
            ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(_TRI[2][0], _TRI[2][1] + label_offset * 0.8,
            ternary_labels["computation"],
            ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.22, _TRI[2][1] + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
#  Build figure
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplot_mosaic(
    """
    ABEE
    CDEE
    """,
    figsize=viz.cm_to_inch((18, 9)),
    layout="constrained",
)

# ── 1. Scatter panels A–D ────────────────────────────────────────────────────

for panel_id, cfg in scatter_panels.items():
    ax = axes[panel_id]

    for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
        
        if "gnm" in dataset:
            continue 
        
        df_merged = dict_with_all_datasets[dataset]
        
        if (not INCLUDE_LEXIS) and ("lexi" in dataset):
            print(f"Skipping {dataset} because it's a Lexis dataset and INCLUDE_LEXIS is False.")
            continue
        
        x_col, y_col = cfg["x"], cfg["y"]
        if x_col not in df_merged.columns or y_col not in df_merged.columns:
            continue

        is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

        ax.scatter(
            df_merged[x_col],
            df_merged[y_col],
            color=combined_colors if is_gnm else COLOR_SCHEME[dataset], # "gray" 
            edgecolor="black" if not is_gnm else "none",
            linewidth=0.25 if not is_gnm else 0,
            s=5,
            alpha=0.15 if is_gnm else 0.4,
            label=LABEL_MAP[dataset] if panel_id == "A" else None,
            zorder=1 if is_gnm else 2,
            rasterized=is_gnm,
        )

    ax.set_xlabel(PROPERTY_NAMES.get(cfg["x"], cfg["x"]), fontsize=6.5)
    ax.set_ylabel(PROPERTY_NAMES.get(cfg["y"], cfg["y"]), fontsize=6.5)
    ax.tick_params(labelsize=5.5)

    ax.set_title(f"{panel_id}  {cfg['title']}", fontsize=7, fontweight="bold", loc="left")
    ax.annotate(cfg["note"], xy=(1, 0), xycoords="axes fraction",
                fontsize=4.5, color="gray", ha="right", va="bottom")


# ── 2. Ternary panel E ───────────────────────────────────────────────────────

ax_t = axes["E"]
_draw_ternary_frame(ax_t)
ax_t.set_title("E  Economy – Robustness – Computation", fontsize=7,
               fontweight="bold", loc="left", pad=8)

for dataset in datasets_to_look_at: # , df_merged in dict_with_all_datasets.items():
    
    if "gnm" in dataset:
            continue 
        
    df_merged = dict_with_all_datasets[dataset]
    cols = list(ternary_cols.values())
    if not all(c in df_merged.columns for c in cols):
        continue

    raw = df_merged[cols].dropna()
    if len(raw) == 0:
        continue

    # For economy: invert LR fraction so that LOW LR = HIGH economy
    v_economy     = 1.0 - raw[ternary_cols["economy"]].values
    v_robustness  = raw[ternary_cols["robustness"]].values
    v_computation = raw[ternary_cols["computation"]].values

    # Min-max normalize each axis across ALL datasets jointly
    # (on first pass, collect global ranges; here we normalize per-panel
    #  using the data present — you may want to precompute global ranges)
    stack = np.column_stack([v_economy, v_robustness, v_computation])

    # Shift to non-negative & normalize rows to sum=1
    for col_idx in range(3):
        col_min = stack[:, col_idx].min()
        col_max = stack[:, col_idx].max()
        rng = col_max - col_min
        if rng > 1e-12:
            stack[:, col_idx] = (stack[:, col_idx] - col_min) / rng
        else:
            stack[:, col_idx] = 1.0 / 3.0

    row_sums = stack.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    stack /= row_sums

    x_cart, y_cart = _bary_to_cart(stack[:, 0], stack[:, 1], stack[:, 2])

    is_gnm = dataset == "hcp_schaefer_100_dataset_gnm"

    ax_t.scatter(
        x_cart, y_cart,
        color="gray" if is_gnm else COLOR_SCHEME[dataset],
        edgecolor="black" if not is_gnm else "none",
        linewidth=0.2 if not is_gnm else 0,
        s=4,
        alpha=0.1 if is_gnm else 0.45,
        zorder=1 if is_gnm else 2,
        rasterized=is_gnm,
    )


# ── 3. Shared legend ─────────────────────────────────────────────────────────

handles, labels = axes["A"].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="outside lower center",
    ncol=min(len(labels), 6),
    fontsize=5.5,
    markerscale=1.8,
    frameon=False,
    handletextpad=0.3,
    columnspacing=1.0,
)

# ── 4. Save ──────────────────────────────────────────────────────────────────

plt.savefig(output_folder / f"trade_off_grid_novel{'_w_o_lexi' if INCLUDE_LEXIS else ''}_w_o_gnms.pdf", bbox_inches="tight", dpi=300)
print(output_folder / f"trade_off_grid_novel{'_w_o_lexi' if INCLUDE_LEXIS else ''}_w_o_gnms.pdf")
plt.show()

