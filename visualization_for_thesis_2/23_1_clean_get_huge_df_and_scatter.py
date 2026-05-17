#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from config import COLORS, COLOR_SCHEME, LABEL_MAP, gray_cmap, bone_white, half_black, emp_dataset_and_experiment_pairs, PROPERTY_NAMES

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Generate output folder
output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/00_trade_off_analysis")
output_folder.mkdir(exist_ok=True)


# In[3]:


# Those are now also the only datasets that get put into the data_dict! Because otherwise it deletes rows and columns... 
# Change that obv for the age part....  
datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
                    'suarez_MaMI_dataset', 
                    'lexis_data_developing', 
                    'kaysons_generated_networks_diffusion', 
                    'kaysons_generated_networks_propagation', 
                    'kaysons_generated_networks_routing', 
                    "ring_lattice_networks",
                    "erdos_renyi_networks"
                    ]
# datasets_to_plot = emp_dataset_and_experiment_pairs.keys()


# In[4]:


columns_to_remove = ["distance_relationship_type", "generative_rule", "num_iterations", "preferential_relationship_type", 
                         "id", "network_idx", "network_index", 
                         "mc_values_for_indiv_lags"]

for dataset_name in ["hcp_schaefer_100_dataset"]:
    
    experiment_name_gnm = "11_mst_2500_animal_0"
    base_path_gnm = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/{experiment_name_gnm}")
    precise_gnm_files = {
                        "fundamental": f"all_metrics_for_{experiment_name_gnm}.csv", 
                        "static": f"all_static_metrics_for_{experiment_name_gnm}_updated.csv", 
                        "dynamic": f"all_dynamic_metrics_for_{experiment_name_gnm}_updated.csv", 
                        "computational": f"all_computational_metrics_for_{experiment_name_gnm}_updated.csv", 
                        "further": f"all_further_metrics_for_{experiment_name_gnm}_updated.csv", 
                        }

    # Combine them all to one big dataframe. The indeces correspond to same entries across the different metric types, so we can easily merge them. Check first if the dfs are equally long. 
    dfs_gnm = {key: pd.read_csv(base_path_gnm / file) for key, file in precise_gnm_files.items()}
    lengths_gnm = {key: len(df) for key, df in dfs_gnm.items()}
    print(lengths_gnm)
    # They are all equally long, so we can merge them together (eta, gamma, id) 
    df_gnm = dfs_gnm["fundamental"]  # Start with the fundamental df as the base
    for key in precise_gnm_files.keys():
        if key == "fundamental":
            continue  # Skip the fundamental df since it's already our base
        print(key)
        df_gnm = df_gnm.merge(dfs_gnm[key], on=["eta", "gamma", "id"], how="left", suffixes=("", f"_{key}"))
        
    # df_gnm = pd.concat(dfs_gnm.values(), axis=1)
    # Check if there are duplicate columns after merging. If so, we can drop them - but print them first to see which ones they are.
    duplicate_columns_gnm = df_gnm.columns[df_gnm.columns.duplicated()]
    print("Attention: Duplicate entries (got removed from the 'previous' df):", duplicate_columns_gnm)
    # Drop the duplicate columns. We want to delete the ones from the "previous" df
    df_gnm = df_gnm.loc[:, ~df_gnm.columns.duplicated()]
    
    for row in df_gnm.isna().sum().sort_values(ascending=False).items(): 
        print(row)


# In[5]:


df_gnm


# In[6]:


df_gnm.isna().sum().sort_values(ascending=False).items()


# In[7]:


dict_with_all_datasets = {}


columns_to_remove = ["distance_relationship_type", "generative_rule", "num_iterations", "preferential_relationship_type", 
                         "id", "network_idx", "network_index", 
                         "mc_values_for_indiv_lags"]

for dataset_name in ["hcp_schaefer_100_dataset"]:
    
    experiment_name_gnm = "11_mst_2500_animal_0"
    base_path_gnm = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/{experiment_name_gnm}")
    precise_gnm_files = {
                        "fundamental": f"all_metrics_for_{experiment_name_gnm}.csv", 
                        "static": f"all_static_metrics_for_{experiment_name_gnm}_updated.csv", 
                        "dynamic": f"all_dynamic_metrics_for_{experiment_name_gnm}_updated.csv", 
                        "computational": f"all_computational_metrics_for_{experiment_name_gnm}_updated.csv", 
                        "further": f"all_further_metrics_for_{experiment_name_gnm}_updated.csv", 
                        }

    # Combine them all to one big dataframe. The indeces correspond to same entries across the different metric types, so we can easily merge them. Check first if the dfs are equally long. 
    dfs_gnm = {key: pd.read_csv(base_path_gnm / file) for key, file in precise_gnm_files.items()}
    lengths_gnm = {key: len(df) for key, df in dfs_gnm.items()}
    print(lengths_gnm)
    # They are all equally long, so we can merge them together (eta, gamma, id) 
    df_gnm = dfs_gnm["fundamental"]  # Start with the fundamental df as the base
    for key in precise_gnm_files.keys():
        if key == "fundamental":
            continue  # Skip the fundamental df since it's already our base
        print(key)
        df_gnm = df_gnm.merge(dfs_gnm[key], on=["eta", "gamma", "id"], how="left", suffixes=("", f"_{key}"))
        
    # df_gnm = pd.concat(dfs_gnm.values(), axis=1)
    # Check if there are duplicate columns after merging. If so, we can drop them - but print them first to see which ones they are.
    duplicate_columns_gnm = df_gnm.columns[df_gnm.columns.duplicated()]
    print("Attention: Duplicate entries (got removed from the 'previous' df):", duplicate_columns_gnm)
    # Drop the duplicate columns. We want to delete the ones from the "previous" df
    df_gnm = df_gnm.loc[:, ~df_gnm.columns.duplicated()]
        
    # Remove columns if all their entries are empty 

    df_gnm = df_gnm.dropna(axis=1, how="all")
    
    # dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].drop(columns=["repertoire_sweep_T_vec", "repertoire_sweep_sizes", "repertoire_sweep_diversities", 
    #                                                                  "repertoire_sweep_weighted_by_distances_T_vec", "repertoire_sweep_weighted_by_distances_sizes", "repertoire_sweep_weighted_by_distances_diversities", 
    #                                                                  "repertoire_sweep_T_critical", "repertoire_sweep_size_critical", "repertoire_sweep_diversity_critical" # "repertoire_sweep_weighted_by_distances_T_critical", "repertoire_sweep_weighted_by_distances_size_critical"
    #                                                                  ], inplace=True, errors="ignore")
    # drop "repertoire_sweep_T_vec", "repertoire_sweep_sizes", "repertoire_sweep_diversities", if those columns are present in the df, since they are not relevant for the analysis and they have many missing values.
    for col in ["repertoire_sweep_T_vec", "repertoire_sweep_sizes", "repertoire_sweep_diversities", 
                "repertoire_sweep_weighted_by_distances_T_vec", "repertoire_sweep_weighted_by_distances_sizes", "repertoire_sweep_weighted_by_distances_diversities",
                "repertoire_sweep_T_critical", "repertoire_sweep_size_critical", "repertoire_sweep_diversity_critical", 
                "departure_from_normality", # As this one has nones now??? TODO. 
                "proportion_long_range_connections_0.356"
                ]:
        if col in df_gnm.columns:
            df_gnm = df_gnm.drop(col, axis=1)
            print(f"Dropped column '{col}' from '{dataset_name}' because it had many missing values and is not relevant for the analysis.")
        else:
            print(f"Column '{col}' not found in '{dataset_name}', so it was not dropped.")

    # Remove unimportant columns
    columns_to_remove += [col for col in df_gnm.columns if col.startswith("h_params")]
    df_gnm = df_gnm.drop(columns=[col for col in columns_to_remove if col in df_gnm.columns])

    # Check the shape of the final df
    print(df_gnm.shape)

    dict_with_all_datasets[dataset_name+"_gnm"] = df_gnm
    
    

for dataset_name in datasets_to_plot:
    
    if dataset_name == "hcp_schaefer_100_dataset_gnm":
        continue  # We already processed the gnm data for this dataset, so we skip it here.
    
    # Do the same thing for the empirical data.
    experiment_name_emp = emp_dataset_and_experiment_pairs[dataset_name]
    base_path_emp = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/{experiment_name_emp}")
    precise_emp_files = {"static": f"metrics_2_static.csv", 
                         "dynamic": f"metrics_2_dynamic.csv",
                         "computational": f"metrics_2_computational.csv",
                         "further": f"metrics_2_further.csv"}
    
    # Combine them all to one big dataframe. The indeces correspond to same entries across the different metric types, so we can easily merge them. Check first if the dfs are equally long. 
    dfs = {key: pd.read_csv(base_path_emp / file) for key, file in precise_emp_files.items()}
    lengths = {key: len(df) for key, df in dfs.items()}
    print(lengths)
    # They are all equally long, so we can merge them on the index.
    # df = pd.concat(dfs.values(), axis=1)
    df = dfs["static"]  # Start with the static df as the base
    for key in precise_emp_files.keys():
        if key == "static":
            continue  # Skip the static df since it's already our base
        print(key)
        df = df.merge(dfs[key], left_index=True, right_index=True, how="left", suffixes=("", f"_{key}"))
    
        
    # Rename columns: strip "basic_measures_" prefix where present
    df.columns = [col.replace("basic_measures_", "") if col.startswith("basic_measures_") else col for col in df.columns]
    
    # Rename columns: strip "mc_original_" prefix where present
    df.columns = [col.replace("mc_original_", "") if col.startswith("mc_original_") else col for col in df.columns]

    # Check if there are duplicate columns after merging. If so, we can drop them - but print them first to see which ones they are.
    duplicate_columns_emp = df.columns[df.columns.duplicated()]
    print("Attention: Duplicate entries (did not get removed so far):", duplicate_columns_emp)
    # Remove exact-duplicate columns (content-identical), keep first occurrence
    cols_to_keep = []
    seen = {}
    for i, col in enumerate(df.columns):
        if col not in seen:
            seen[col] = i
            cols_to_keep.append(i)
        else:
            # Check if the duplicate column is identical in content
            if df.iloc[:, seen[col]].equals(df.iloc[:, i]):
                print(f"  -> Dropping identical duplicate column: '{col}'")
            else:
                # Keep both but rename the duplicate
                new_name = f"{col}_dup"
                df.columns.values[i] = new_name
                cols_to_keep.append(i)
                print(f"  -> Kept non-identical duplicate column '{col}' as '{new_name}'")
    df = df.iloc[:, cols_to_keep]
    
    # drop "repertoire_sweep_T_vec", "repertoire_sweep_sizes", "repertoire_sweep_diversities", if those columns are present in the df, since they are not relevant for the analysis and they have many missing values.
    for col in ["repertoire_sweep_T_vec", "repertoire_sweep_sizes", "repertoire_sweep_diversities", 
                "repertoire_sweep_weighted_by_distances_T_vec", "repertoire_sweep_weighted_by_distances_sizes", "repertoire_sweep_weighted_by_distances_diversities", 
                "repertoire_sweep_T_critical", "repertoire_sweep_size_critical", "repertoire_sweep_diversity_critical", 
                "proportion_long_range_connections_0.356", 
                "repertoire_diversity", "repertoire_size", 
                "departure_from_normality" # As this one has nones now??? TODO. 
                ]:
        if col in df.columns:
            df = df.drop(col, axis=1)
            print(f"Dropped column '{col}' from '{dataset_name}' because it had many missing values and is not relevant for the analysis.")
        else:
            print(f"Column '{col}' not found in '{dataset_name}', so it was not dropped.")

            
    # Remove columns if all their entries are empty 
    df = df.dropna(axis=1, how="all")

    # if there is n_components in the df, then exclude the rows where n_components != 1 and print the row ids 
    if "n_connected_components" in df.keys():          
        # Get all rows where n_connected_components != 1; print them and exclude them
        non_singleton_components = df[df['n_connected_components'] != 1]
        indices_to_remove = non_singleton_components.index.tolist()
        if len(indices_to_remove) > 0: 
            print("⚠️ Non-singleton components. IDs that are going to be removed: ", indices_to_remove) #  "their 'network_id' or similar values:", non_singleton_components[["id", "network_index", "network_idx"]]))
            # df = df[df['n_connected_components'] == 1]
            df = df[~df.index.isin(indices_to_remove)]
            print(" ---> Done. New df length:", len(df))
        
    # Remove unimportant columns
    columns_to_remove += [col for col in df.columns if col.startswith("h_params")]
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])


    # Check the shape of the final df   
    print(df.shape)
    dict_with_all_datasets[dataset_name] = df
    

# SAVE THE DICT 
with open(output_folder / "all_datasets.pkl", "wb") as f:
    pickle.dump(dict_with_all_datasets, f)


# In[8]:


# for the other datasets, drop "wiring_cost" if it is present. Then, rename "wiring_cost_dup" to "wiring_cost"
for dataset in dict_with_all_datasets:
    
    print(f"\nDataset: {dataset}")
    for k in dict_with_all_datasets[dataset].keys():
        if "wiring" in k: 
            print(k, " -  First value:", dict_with_all_datasets[dataset][k].iloc[0])
            
    if "gnm" in dataset: 
            dict_with_all_datasets[dataset].drop(columns=["wiring_cost_static"], inplace=True, errors='ignore') # and keep "wiring_cost", which has values around 37000 mm
            print("Dropped 'wiring_cost_static' from", dataset, "and kept 'wiring_cost' with values around 37000 mm.")
    else: 
        
        if "wiring_cost" in dict_with_all_datasets[dataset].columns and "wiring_cost_dup" in dict_with_all_datasets[dataset].columns:
            dict_with_all_datasets[dataset].drop(columns=["wiring_cost"], inplace=True)
            dict_with_all_datasets[dataset].rename(columns={"wiring_cost_dup": "wiring_cost"}, inplace=True)
            print("Dropped 'wiring_cost' and renamed 'wiring_cost_dup' to 'wiring_cost' in", dataset)


# In[9]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]
# dict_with_all_datasets["lexis_data_developing"]["wiring_cost"]


# In[10]:


COLOR_SCHEME


# In[11]:


mc_per_dataset = {}

lin ="mc_input_scaling_0_1_mc_values_for_indiv_lags"
non_lin = "mc_nonlinear_input_scaling_0_1_mc_values_for_indiv_lags"
    

for dataset in datasets_to_plot:
    
    if "gnm" in dataset: 
        continue 
    
    plt.figure(figsize=viz.cm_to_inch((3,3)))
    
    print(dataset)  
    mc_lin_arr = []
    mc_nonlin_arr = []
    colors = []
    
    # iterate trough all rows 
    for idx, row in dict_with_all_datasets[dataset].iterrows():

        lin_profile = row[lin]
        nonlin_profile = row[non_lin]

        lin_profile = lin_profile.strip("[]").split(",")
        lin_profile = np.array(lin_profile).astype(np.float32)

        nonlin_profile = nonlin_profile.strip("[]").split(",")
        nonlin_profile = np.array(nonlin_profile).astype(np.float32)

        plt.plot(lin_profile, # label="linear", 
                 color=COLOR_SCHEME[dataset], linestyle="-") # "blue")
        plt.plot(nonlin_profile, # label="nonlinear", 
                 color=COLOR_SCHEME[dataset], linestyle=":") # "orange")

        mc_lin_arr.append(lin_profile.sum())
        mc_nonlin_arr.append(nonlin_profile.sum())
        colors.append(COLOR_SCHEME[dataset])
        
        if idx % 10 == 5:
            print("done")
            break

    mc_per_dataset[dataset] = {
        "lin": mc_lin_arr,
        "nonlin": mc_nonlin_arr
    }
    
    if "MaMI" in dataset: 
        # Only add legend one for lin and once for nonlin, to avoid duplicate legend entries.
        plt.plot([], [], label="linear", color=COLOR_SCHEME[dataset], linestyle="-") # "blue")
        plt.plot([], [], label="nonlinear", color=COLOR_SCHEME[dataset], linestyle=":") # "orange")
        plt.legend()


    plt.show() 


# plt.boxplot([mc_per_dataset[dataset]["lin"] for dataset in datasets_to_plot if "gnm" not in dataset], 
#             positions=np.array(range(len(datasets_to_plot)-1))*2.0-0.4, 
#             widths=0.6, 
#             patch_artist=True, 
#             boxprops=dict(facecolor="blue", color=colors),
#             medianprops=dict(color="white"))
# plt.title("Memory capacity (linear vs nonlinear)")

# plt.boxplot([mc_per_dataset[dataset]["nonlin"] for dataset in datasets_to_plot if "gnm" not in dataset], 
#             positions=np.array(range(len(datasets_to_plot)-1))*2.0+0.4, 
#             widths=0.6, 
#             patch_artist=True, 
#             boxprops=dict(facecolor="orange", color=colors),
#             medianprops=dict(color="white"))

# plt.xticks(np.array(range(len(datasets_to_plot)-1))*2.0, [LABEL_MAP[dataset] for dataset in datasets_to_plot if "gnm" not in dataset], 
#         #    rotation=45, 
#         #    ha="right"
#         )
# plt.show() 
    
# plt.tight_layout()

datasets_no_gnm = [d for d in datasets_to_plot if "gnm" not in d]
n = len(datasets_no_gnm)

fig, ax = plt.subplots(figsize=viz.cm_to_inch((9,9))) # 6, 6)))

for i, dataset in enumerate(datasets_no_gnm):
    color = COLOR_SCHEME[dataset]

    for values, hatch, alpha, zorder in [
        (mc_per_dataset[dataset]["lin"],    "",    1.0, 3),  # solid, on top
        (mc_per_dataset[dataset]["nonlin"], "///", 0.5, 2),  # hatched, behind
    ]:
        bp = ax.boxplot(
            values,
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=color, edgecolor="black", linewidth=1.2,
                          hatch=hatch, alpha=alpha),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="black", linewidth=1.2, alpha=alpha),
            capprops=dict(color="black", linewidth=1.2, alpha=alpha),
            flierprops=dict(marker="o", markeredgecolor="black",
                            markerfacecolor=color, markersize=3, alpha=alpha),
            zorder=zorder,
        )

from matplotlib.patches import Patch
ax.legend(
    handles=[
        Patch(facecolor="gray", edgecolor="black", label="linear"),
        Patch(facecolor="gray", edgecolor="black", hatch="///", alpha=0.5, label="nonlinear"),
    ],
    loc="lower right",
    framealpha=0.9,
    fontsize=8,
)

ax.set_xticks(range(n))
ax.set_xticklabels([LABEL_MAP[d] for d in datasets_no_gnm], # rotation=30, ha="right")
                   ha="center")
# shift every second xtick label a bit to the right to avoid overlap
for i, label in enumerate(ax.get_xticklabels()):
    if i % 2 == 1:
        label.set_y(label.get_position()[1] - 0.1)
ax.set_title("Memory capacity (linear vs nonlinear)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()


# In[12]:


from matplotlib.patches import Patch

datasets_no_gnm = [d for d in datasets_to_plot if "gnm" not in d]
n = len(datasets_no_gnm)  # 5 datasets → A–E

mosaic = [
    ['box', 'box', 'box', 'A', 'B', "leg"],
    ['box', 'box', 'box', 'C', 'D', 'E'],
]

fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=viz.cm_to_inch((18, 9)),
    gridspec_kw={'wspace': 0.35, 'hspace': 0.4},
)

# ── Boxplot (left, spanning both rows) ───────────────────────────────────────
ax_box = axd['box']

for i, dataset in enumerate(datasets_no_gnm):
    color = COLOR_SCHEME[dataset]
    for values, hatch, alpha, zorder in [
        (mc_per_dataset[dataset]["lin"],    "",    1.0, 3),
        (mc_per_dataset[dataset]["nonlin"], "///", 0.5, 2),
    ]:
        ax_box.boxplot(
            values,
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=color, edgecolor="black", #  linewidth=1.2,
                          hatch=hatch, alpha=alpha),
            medianprops=dict(color="black"), # , linewidth=2),
            whiskerprops=dict(color="black", # linewidth=1.2,
                              alpha=alpha),
            capprops=dict(color="black", # linewidth=1.2,
                          alpha=alpha),
            flierprops=dict(marker="o", markeredgecolor="black",
                            markerfacecolor=color, markersize=3, alpha=alpha),
            zorder=zorder,
        )

ax_box.set_xticks(range(n))
ax_box.set_xticklabels([LABEL_MAP[d] for d in datasets_no_gnm], rotation=30, ha="right") # , fontsize=8)
ax_box.set_ylabel("Memory capacity") #, fontsize=8)
# ax_box.set_title("Memory capacity (linear vs nonlinear)") # , fontsize=9)
ax_box.spines[["top", "right"]].set_visible(False)

# ── MC lag profiles (A–E, one per dataset) ───────────────────────────────────
panel_keys = ['A', 'B', 'C', 'D', 'E']

# Set up shared axes before plotting
# ref_ax = axd['A']
# for key in panel_keys[1:]:
#     axd[key].sharex(ref_ax)
#     axd[key].sharey(ref_ax)

for key, dataset in zip(panel_keys, datasets_no_gnm):
    ax_p = axd[key]
    color = COLOR_SCHEME[dataset]

    for idx, row in dict_with_all_datasets[dataset].iterrows():
        lin_profile    = np.array(row[lin].strip("[]").split(",")).astype(np.float32)
        nonlin_profile = np.array(row[non_lin].strip("[]").split(",")).astype(np.float32)

        ax_p.plot(lin_profile,    color=color, linestyle="-")
                #   linewidth=0.6,
                #   alpha=0.8)
        ax_p.plot(nonlin_profile, color=color, linestyle="--") 
                #   linewidth=0.6, 
                #   alpha=0.4)
        ax_p.plot(lin_profile-nonlin_profile, 
                  color=color, linestyle=":") 
                  # linewidth=0.6, alpha=0.4)
        
        
        if idx % 10 == 5:
            break

    # ax_p.set_title(LABEL_MAP[dataset], fontsize=8, 
    #                color=color)
    ax_p.spines[["top", "right"]].set_visible(False)
    # ax_p.tick_params(labelsize=7)

    # Only show y-tick labels on left column (A, D)
    if key in ['B', "D", 'E']:
        ax_p.set_yticklabels([])
    if key in ["A", "C"]: 
        ax_p.set_ylabel("R^2 per lag") # , fontsize=8)
    # Only show x-tick labels on bottom row (D, E)
    if key in ['A', 'B']:
        ax_p.set_xticklabels([])
    if key in ["C", "D", "E"]:
        ax_p.set_xlabel("Lag") # , fontsize=8)

# ── Legend panel ─────────────────────────────────────────────────────────────
ax_leg = axd['leg']
ax_leg.axis('off')
ax_leg.legend(
    handles=[
        Patch(facecolor="gray", edgecolor="black", label="linear"),
        Patch(facecolor="gray", edgecolor="black", hatch="///", alpha=0.5, label="nonlinear"),
    ],
    loc="center",
    # fontsize=9,
    framealpha=0.9,
    title="MC type",
    # title_fontsize=9,
)

plt.tight_layout()
plt.show()


# In[13]:


import pandas as pd                                                                                               
df = pd.read_csv('/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/05_mst_animal_0_compared_with_mami/metrics_2_computational.csv', nrows=2)
ipc_cols = [c for c in df.columns if 'ipc' in c.lower()]
mc_cols = [c for c in df.columns if 'mc_' in c.lower()][:5]
print('IPC cols:', ipc_cols)
print('MC cols (first 5):', mc_cols)
print('All cols:', list(df.columns))


# In[14]:


import pandas as pd
from pathlib import Path                                                                                          
                                                                                                                    
base = Path('/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm')                            
datasets = {                                                                                                      
    'MaMI':        base / 'suarez_MaMI_dataset/05_mst_animal_0_compared_with_mami/metrics_2_computational.csv',   
    'diffusion':   base /                                                                                         
'kaysons_generated_networks_diffusion/05_mst_animal_0_compared_with_diffusion/metrics_2_computational.csv',       
    'propagation': base /                                                                                         
'kaysons_generated_networks_propagation/05_mst_animal_0_compared_with_propagation/metrics_2_computational.csv',   
    'routing':     base /                                                                                         
'kaysons_generated_networks_routing/05_mst_animal_0_compared_with_routing/metrics_2_computational.csv',           
    'topology':    base /
'kaysons_generated_networks_topology/05_mst_animal_0_compared_with_topology/metrics_2_computational.csv',
    'lexis_dev':   base /
'lexis_data_developing/05_mst_animal_0_compared_with_lexis_data_developing/metrics_2_computational.csv',
    'lexis_consensus': base /
'lexis_data_developing_consensus_per_age_1_year/00_pca/metrics_2_computational.csv',
}
for name, path in datasets.items():
    if path.exists():
        df = pd.read_csv(path, nrows=2)
        has_ipc = 'ipc_ipc_total_mean' in df.columns
        print(f'{name}: {len(pd.read_csv(path))} rows, ipc={has_ipc}')
    else:
        print(f'{name}: MISSING')


# In[ ]:





# In[15]:


from matplotlib.patches import Patch

datasets_no_gnm = [d for d in datasets_to_plot if "gnm" not in d]
n = len(datasets_no_gnm)

mosaic = [
    ['box', 'box', 'box', 'A', 'B', "leg"],
    ['box', 'box', 'box', 'C', 'D', 'E'],
]

fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=viz.cm_to_inch((18, 9)),
    gridspec_kw={'wspace': 0.6, 'hspace': 0.4},  # more wspace to avoid y-axis overlap
)

# ── Boxplot ───────────────────────────────────────────────────────────────────
ax_box = axd['box']

for i, dataset in enumerate(datasets_no_gnm):
    color = COLOR_SCHEME[dataset]
    for values, hatch, alpha, zorder in [
        (mc_per_dataset[dataset]["lin"],    "",    1.0, 3),
        (mc_per_dataset[dataset]["nonlin"], "///", 0.5, 2),
    ]:
        ax_box.boxplot(
            values,
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=color, edgecolor="black", hatch=hatch, alpha=alpha),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black", alpha=alpha),
            capprops=dict(color="black", alpha=alpha),
            flierprops=dict(marker="o", markeredgecolor="black",
                            markerfacecolor=color, markersize=3, alpha=alpha),
            zorder=zorder,
        )

ax_box.set_xticks(range(n))
ax_box.set_xticklabels([LABEL_MAP[d] for d in datasets_no_gnm], rotation=30, ha="right")
ax_box.set_ylabel("Memory capacity")
ax_box.set_xlim(-0.6, n - 0.4)  # prevent first/last box from touching spines
ax_box.spines[["top", "right"]].set_visible(False)

# ── Cumulative R² panels ──────────────────────────────────────────────────────
panel_keys = ['A', 'B', 'C', 'D', 'E']

ref_ax = axd['A']
for key in panel_keys[1:]:
    axd[key].sharex(ref_ax)
    axd[key].sharey(ref_ax)

for key, dataset in zip(panel_keys, datasets_no_gnm):
    ax_p = axd[key]
    color = COLOR_SCHEME[dataset]

    for idx, row in dict_with_all_datasets[dataset].iterrows():
        lin_profile    = np.array(row[lin].strip("[]").split(",")).astype(np.float32)
        nonlin_profile = np.array(row[non_lin].strip("[]").split(",")).astype(np.float32)

        cum_lin    = np.cumsum(lin_profile)
        cum_nonlin = np.cumsum(nonlin_profile)
        lags = np.arange(len(cum_lin))

        # Linear: solid filled area
        ax_p.fill_between(lags, cum_lin,    color=color, alpha=0.2, linewidth=0)
        # Nonlinear: hatched, no solid fill — so both are visible when overlapping
        ax_p.fill_between(lags, cum_nonlin, facecolor="none",
                          hatch="///", edgecolor=color, linewidth=0.0, alpha=0.3)
        # Thin outline so the nonlinear boundary is readable
        ax_p.plot(lags, cum_nonlin, color=color, linewidth=0.7, alpha=0.7)

        if idx % 10 == 5:
            break

    ax_p.spines[["top", "right"]].set_visible(False)

    if key in ['B', 'D', 'E']:
        ax_p.set_yticklabels([])
    if key in ['A', 'C']:
        ax_p.set_ylabel("Cumulative R²")
    if key in ['A', 'B']:
        ax_p.set_xticklabels([])
    if key in ['C', 'D', 'E']:
        ax_p.set_xlabel("Lag")

# ── Shared legend ─────────────────────────────────────────────────────────────
ax_leg = axd['leg']
ax_leg.axis('off')
ax_leg.legend(
    handles=[
        Patch(facecolor="gray", edgecolor="none",  alpha=0.5, label="linear"),
        Patch(facecolor="none", edgecolor="gray",  hatch="///", label="nonlinear"),
    ],
    loc="center",
    framealpha=0.9,
    title="MC type",
)

plt.tight_layout()
plt.show()


# In[16]:


# For the distance matrices and connectomes of MaMI: create copy that has the 95 removed. 
mami_data_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset")
conns = np.load(mami_data_path / "01_connectomes/00_connectomes_50_bin.npy")
dists_scaled = np.load(mami_data_path / "02_distance_matrices/distance_matrix_50_scaled_to_schaeffer.npy")
dists_unscaled = np.load(mami_data_path / "02_distance_matrices/distance_matrix_50.npy")

# Now, delete that one entry and save them as "..._wo_idx_95
# Index to remove
idx_to_remove = 95

# Remove subject 95 (axis=0)
conns_wo = np.delete(conns, idx_to_remove, axis=0)
dists_scaled_wo = np.delete(dists_scaled, idx_to_remove, axis=0)
dists_unscaled_wo = np.delete(dists_unscaled, idx_to_remove, axis=0)

# Save
np.save(mami_data_path / "01_connectomes/00_connectomes_50_bin_wo_idx_95.npy", conns_wo)
np.save(mami_data_path / "02_distance_matrices/distance_matrix_50_scaled_to_schaeffer_wo_idx_95.npy", dists_scaled_wo)
np.save(mami_data_path / "02_distance_matrices/distance_matrix_50_wo_idx_95.npy", dists_unscaled_wo)

print("Done. New shapes:")
print("Connectomes:", conns_wo.shape)
print("Scaled distances:", dists_scaled_wo.shape)
print("Unscaled distances:", dists_unscaled_wo.shape)


# In[17]:


# Quick overview of what's in each dataset 
print("\n" + "=" * 60)
print("Datasets loaded:")
for name in dict_with_all_datasets:
    print(f"  {name}: {dict_with_all_datasets[name].shape}")


# In[18]:


# Make a nice comparison table: which properties each dataset has
# Collect all unique column names across datasets
all_columns = set()
for df in dict_with_all_datasets.values():
    all_columns.update(df.columns)
all_columns = sorted(all_columns)

# Build a presence table
presence_table = pd.DataFrame(index=all_columns, columns=list(dict_with_all_datasets.keys()))
for dataset_name, df in dict_with_all_datasets.items():
    for col in all_columns:
        presence_table.loc[col, dataset_name] = "y" if col in df.columns else "missing"

presence_table.rename(columns=LABEL_MAP, inplace=True)

print("\n" + "=" * 60)
print("Property presence table:")
print(presence_table.to_string())

# Save the table to CSV
presence_table.to_csv(output_folder / "property_presence_table.csv")
print(output_folder / "property_presence_table.csv")
print("\nSaved property presence table.")


# In[19]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]["wiring_cost"] 


# In[ ]:





# In[ ]:





# In[20]:


for dataset in dict_with_all_datasets:
    # if "wiring_cost" in dict_with_all_datasets[dataset].columns:
    #     plt.hist(dict_with_all_datasets[dataset]["wiring_cost"], alpha=0.5) # density=True, label=dataset)
    if "wiring_cost_dup" in dict_with_all_datasets[dataset].columns:
        plt.hist(dict_with_all_datasets[dataset]["wiring_cost_dup"], alpha=0.5, 
                #  density=True, 
                 label=dataset)

plt.legend()


# In[21]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]


# In[22]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"] # ["wiring_cost_dup"]


# In[23]:


# datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
#                     'suarez_MaMI_dataset', 'lexis_data_developing', 
#                     'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 
#                     ]

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))

pairs_to_plot = [
    ("computational_capacity_notebook_memory_capacity_total_notebook", "computational_capacity_notebook_nonlinear_capacity_total_notebook"),
    ("mc_lin_cut40_mc_mean", "mc_nonlin_cut40_mc_mean"),
    ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"), 
    ("computational_capacity_memory_capacity_total", "computational_capacity_nonlinear_capacity_total"),
    ("computational_capacity_memory_capacity_total", "computational_capacity_cubic_capacity_total"), 
    ("mc_lin_cut40_mc_mean", "mc_nonlin_cut40_mc_mean"), 
    ("computational_capacity_notebook_damicelli_memory_capacity_total_notebook",
     "computational_capacity_notebook_damicelli_nonlinear_capacity_total_notebook"), 
    ("computational_capacity_notebook_memory_capacity_total_notebook", "computational_capacity_notebook_nonlinear_capacity_total_notebook"),
    ("computational_capacity_supplementary_kayson_mc_mean", "computational_capacity_supplementary_kayson_mc_nonlinear_mean"
     ), 
    ("computational_capacity_supplementary_kayson_mc_5", "computational_capacity_supplementary_kayson_mc_nonlin_5"
     ), 
    # ("computational_capacity_total_capacity"), 
    # ("mc_5"), 
    ("mc_mean", "mc_nonlinear_original_mc_mean"), # ?? 
    # ("mc_input_scaling_0_1_mc_5", )
    ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"),
    # ("mc_lin_cut40_mc_5", )
    ("mc_lin_cut40_mc_mean", "mc_nonlin_cut40_mc_mean"),
    ("ipc_ipc_deg1_mean", "ipc_ipc_deg2_mean")

]


for x_col, y_col in pairs_to_plot:
        for dataset in datasets_to_plot:
            print(f"Plotting '{dataset}' for pair ('{x_col}', '{y_col}')...")
            len(dict_with_all_datasets[dataset])
            if not x_col in dict_with_all_datasets[dataset].columns or not y_col in dict_with_all_datasets[dataset].columns:
                print(f"⚠️ Dataset '{dataset}' does not have the required columns '{x_col}' and/or '{y_col}' for plotting, so it was skipped for this pair.")
                continue
            else: 
                print(f"Dataset '{dataset}' has the required columns. Proceeding with plotting...")
                print(f"Number of data points in '{dataset}': {len(dict_with_all_datasets[dataset])}")

            plt.scatter(
                    x = dict_with_all_datasets[dataset][x_col], 
                    y = dict_with_all_datasets[dataset][y_col], 
                    color=COLOR_SCHEME[dataset],
                    s=30 if "kayson" in dataset else 5, 
                    alpha=0.2 if "gnm" in dataset else 1)
            
        # plt.ylim(0,0.00002)
        plt.xlabel(x_col, fontsize=6)
        plt.ylabel(y_col, fontsize=6)   
        plt.show()


# In[24]:


from utils import get_combined_colors

datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
                    'suarez_MaMI_dataset', 'lexis_data_developing', 
                    'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 
                    ]

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))

pairs_to_plot = [
    ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"),
    
]


for x_col, y_col in pairs_to_plot:
        for dataset in datasets_to_plot:
            print(f"Plotting '{dataset}' for pair ('{x_col}', '{y_col}')...")
            len(dict_with_all_datasets[dataset])
            if not x_col in dict_with_all_datasets[dataset].columns or not y_col in dict_with_all_datasets[dataset].columns:
                print(f"⚠️ Dataset '{dataset}' does not have the required columns '{x_col}' and/or '{y_col}' for plotting, so it was skipped for this pair.")
                continue
            else: 
                print(f"Dataset '{dataset}' has the required columns. Proceeding with plotting...")
                print(f"Number of data points in '{dataset}': {len(dict_with_all_datasets[dataset])}")

            if "gnm" in dataset:
                combined_colors = get_combined_colors(dict_with_all_datasets[dataset])

            # Activate this for only gnm 
            # else: 
            #     continue 
            
            plt.scatter(
                    x = dict_with_all_datasets[dataset][x_col], 
                    y = dict_with_all_datasets[dataset][y_col], 
                    color=combined_colors if "gnm" in dataset else COLOR_SCHEME[dataset],
                    s=20 if "kayson" in dataset else 2, 
                    alpha=0.2 if "gnm" in dataset else 1)
            
        plt.xlabel(x_col, fontsize=6)
        plt.ylabel(y_col, fontsize=6)   
        plt.show()


# In[25]:


from utils import get_combined_colors

datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
                    'suarez_MaMI_dataset', 'lexis_data_developing', 
                    'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 
                    ]

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))

pairs_to_plot = [
    ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"),
    
]


for x_col, y_col in pairs_to_plot:
        for dataset in datasets_to_plot:
            print(f"Plotting '{dataset}' for pair ('{x_col}', '{y_col}')...")
            len(dict_with_all_datasets[dataset])
            if not x_col in dict_with_all_datasets[dataset].columns or not y_col in dict_with_all_datasets[dataset].columns:
                print(f"⚠️ Dataset '{dataset}' does not have the required columns '{x_col}' and/or '{y_col}' for plotting, so it was skipped for this pair.")
                continue
            else: 
                print(f"Dataset '{dataset}' has the required columns. Proceeding with plotting...")
                print(f"Number of data points in '{dataset}': {len(dict_with_all_datasets[dataset])}")

            if "gnm" in dataset:
                combined_colors = get_combined_colors(dict_with_all_datasets[dataset])

            # Activate this for only gnm 
            else: 
                continue 
            
            plt.scatter(
                    x = dict_with_all_datasets[dataset][x_col], 
                    y = dict_with_all_datasets[dataset][y_col], 
                    c = dict_with_all_datasets[dataset]["modularity"],  # Use x_col values for coloring
                    # color=combined_colors if "gnm" in dataset else COLOR_SCHEME[dataset],
                    s=20 if "kayson" in dataset else 2, 
                    alpha=0.2 if "gnm" in dataset else 1)
            
        plt.xlabel(x_col, fontsize=6)
        plt.ylabel(y_col, fontsize=6)  
        plt.colorbar(label="Modularity")  # Add colorbar with label
        plt.show()


# In[26]:


from utils import get_combined_colors

datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
                    'suarez_MaMI_dataset', 'lexis_data_developing', 
                    'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 
                    ]

from config import remaining_categories

categories_to_plot = [] 
for cat in remaining_categories: 
    categories_to_plot.extend(remaining_categories[cat])
print(categories_to_plot)


pairs_to_plot = [
    ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"),
    
]

for category in categories_to_plot:
    fig = plt.figure(figsize=viz.cm_to_inch((6,6)))
    for x_col, y_col in pairs_to_plot:
        for dataset in datasets_to_plot:

            if "gnm" in dataset:
                combined_colors = get_combined_colors(dict_with_all_datasets[dataset])

            # Activate this for only gnm 
            else: 
                continue 
            
            plt.scatter(
                    x = dict_with_all_datasets[dataset][x_col], 
                    y = dict_with_all_datasets[dataset][y_col], 
                    c = dict_with_all_datasets[dataset][category],  # Use x_col values for coloring
                    # color=combined_colors if "gnm" in dataset else COLOR_SCHEME[dataset],
                    s=20 if "kayson" in dataset else 2, 
                    alpha=0.2 if "gnm" in dataset else 1)
            
        plt.xlabel(x_col, fontsize=6)
        plt.ylabel(y_col, fontsize=6)  
        plt.colorbar(label=PROPERTY_NAMES[category])  # Add colorbar with label
        plt.show()


# In[27]:


from utils import get_combined_colors

datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
                    'suarez_MaMI_dataset', 'lexis_data_developing', 
                    'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 
                    ]

from config import remaining_categories

categories_to_plot = [] 
for cat in remaining_categories: 
    categories_to_plot.extend(remaining_categories[cat])
print(categories_to_plot)


pairs_to_plot = [
    ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"),
    
]

for category in categories_to_plot:
    fig = plt.figure(figsize=viz.cm_to_inch((6,6)))
    for x_col, y_col in pairs_to_plot:
        for dataset in datasets_to_plot:

            if "gnm" in dataset:
                combined_colors = get_combined_colors(dict_with_all_datasets[dataset])

            # Activate this for only gnm 
            else: 
                continue 
            
            plt.scatter(
                    x = dict_with_all_datasets[dataset][x_col], 
                    y = dict_with_all_datasets[dataset][y_col], 
                    c = combined_colors, # dict_with_all_datasets[dataset][category],  # Use x_col values for coloring
                    # color=combined_colors if "gnm" in dataset else COLOR_SCHEME[dataset],
                    s=20 if "kayson" in dataset else 2, 
                    alpha=0.2 if "gnm" in dataset else 1)
            
        plt.xlabel(x_col, fontsize=6)
        plt.ylabel(y_col, fontsize=6)  
        plt.colorbar(label=PROPERTY_NAMES[category])  # Add colorbar with label
        plt.show()
    break 


# In[28]:


df_gnm


# In[29]:


from utils import get_combined_colors

datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
                    'suarez_MaMI_dataset', 'lexis_data_developing', 
                    'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 
                    ]

from config import remaining_categories

# categories_to_plot = [] 
# for cat in remaining_categories: 
#     categories_to_plot.extend(remaining_categories[cat])
# print(categories_to_plot)
categories_to_plot = list(PROPERTY_NAMES.keys()) + ["MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)"] # Plot all categories

pairs_to_plot = [
    ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"),
    
]

for category in categories_to_plot:
    fig = plt.figure(figsize=viz.cm_to_inch((6,6)))
    for x_col, y_col in pairs_to_plot:
        for dataset in datasets_to_plot:
            
            if not category in dict_with_all_datasets[dataset].columns:
                print(f"⚠️ Dataset '{dataset}' does not have the required column '{category}' for coloring, so it was skipped for this category.")
                continue
            
            if "gnm" in dataset:
                combined_colors = get_combined_colors(dict_with_all_datasets[dataset])

            # Activate this for only gnm 
            else: 
                continue 
            
            plt.scatter(
                    x = dict_with_all_datasets[dataset][x_col], 
                    y = dict_with_all_datasets[dataset][y_col], 
                    c = dict_with_all_datasets[dataset][category],  # Use x_col values for coloring
                    # color=combined_colors if "gnm" in dataset else COLOR_SCHEME[dataset],
                    s=20 if "kayson" in dataset else 2, 
                    alpha=0.2 if "gnm" in dataset else 1)
            
        plt.xlabel(x_col, fontsize=6)
        plt.ylabel(y_col, fontsize=6)  
        plt.title(category)
        # if category in PROPERTY_NAMES:
        #     plt.colorbar(label=PROPERTY_NAMES[category])  # Add colorbar with label
        # else: 
        #     plt.colorbar(label=category)  # Add colorbar with label
        plt.show()


# In[30]:


from utils import get_combined_colors

datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
                    'suarez_MaMI_dataset', 'lexis_data_developing', 
                    'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 
                    ]

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))

pairs_to_plot = [
    ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"),
    
]


for x_col, y_col in pairs_to_plot:
        for dataset in datasets_to_plot:
            print(f"Plotting '{dataset}' for pair ('{x_col}', '{y_col}')...")
            len(dict_with_all_datasets[dataset])
            if not x_col in dict_with_all_datasets[dataset].columns or not y_col in dict_with_all_datasets[dataset].columns:
                print(f"⚠️ Dataset '{dataset}' does not have the required columns '{x_col}' and/or '{y_col}' for plotting, so it was skipped for this pair.")
                continue
            else: 
                print(f"Dataset '{dataset}' has the required columns. Proceeding with plotting...")
                print(f"Number of data points in '{dataset}': {len(dict_with_all_datasets[dataset])}")

            if "gnm" in dataset:
                combined_colors = get_combined_colors(dict_with_all_datasets[dataset])

            # Activate this for only gnm 
            else: 
                continue 
            
            plt.scatter(
                    x = dict_with_all_datasets[dataset][x_col], 
                    y = dict_with_all_datasets[dataset][y_col], 
                    c = dict_with_all_datasets[dataset]["modularity"],  # Use x_col values for coloring
                    # color=combined_colors if "gnm" in dataset else COLOR_SCHEME[dataset],
                    s=20 if "kayson" in dataset else 2, 
                    alpha=0.2 if "gnm" in dataset else 1)
            
        plt.xlabel(x_col, fontsize=6)
        plt.ylabel(y_col, fontsize=6)  
        plt.colorbar(label="Modularity")  # Add colorbar with label
        plt.show()


# In[31]:


import json

# Generate a template dict mapping each column name to an empty display name
property_names_template = {col: "" for col in all_columns}

# Print as a Python dict literal ready to copy-paste and fill in
print("\nPROPERTY_NAMES = {")
for col in all_columns:
    print(f'    "{col}": "",')
print("}")


# In[32]:


x_value = "wiring_cost"
thresholds = ["0.1", "0.3", "0.3956", "0.5"]
threshold_labels = ["10%", "30%", "39.65%", "50%"]

for dataset in dict_with_all_datasets.keys():
    df = dict_with_all_datasets[dataset]
    print(dataset, df.shape)
    
    fig, axs = plt.subplots(1, 4, figsize=viz.cm_to_inch((18, 5)), sharex=True, # sharey=True, 
                            dpi=200)

    for i, (thresh, label) in enumerate(zip(thresholds, threshold_labels)):
        col = f"proportion_long_range_connections_{thresh}"
        
        axs[i].scatter(
            df[x_value], df[col],
            alpha=0.03 if dataset == "hcp_schaefer_100_dataset_gnm" else 0.4,  # Adjust alpha based on dataset size
            color=COLOR_SCHEME[dataset],
            edgecolor="none", s=2, rasterized=True,
        )
        
        sns.regplot(
            x=x_value, y=col, data=df,
            scatter=False, ax=axs[i],
            line_kws={"color": COLORS["colds"]["LAKE_BLUE"], "linewidth": 1.2},
        )

        r = df[[x_value, col]].corr().iloc[0, 1]
        axs[i].text( # lower right corner of the plot
            x=0.95, y=0.05,  # Adjust as needed
            s=f"$r = {r:.3f}$",
            transform=axs[i].transAxes, fontsize=7,
            color=COLORS["colds"]["LAKE_BLUE"],
            va="bottom", # top", 
            ha="right", # left",
        )
        
        # axs[i].set_title(f"Threshold: {label}", fontsize=8)
        axs[i].set_xlabel("") # "Wiring cost" if i == 1 else "")  # single shared label
        axs[i].set_xlabel(PROPERTY_NAMES[x_value]) # f"Proportion LR ({label})" if i == 0 else "")  # single shared label
        # axs[i].set_ylabel("$f_{\\mathrm{LR (10%)}}$")
        # axs[i].set_ylabel("$f_{\\mathregular{LR}(10\\%)}$")
        # axs[i].set_ylabel(f"Long-range fraction\n$f_{{\\rm LR\\,({label.replace('%', '\\%')})}}$")
        axs[i].set_yticks([])
        axs[i].set_ylabel(f"$f_{{\\rm LR\\,({label.replace('%', '\\%')})}}$")

    # axs[0].set_ylabel("Long-range fraction\n$f_{\\mathrm{LR (10%)}}$")
    axs[0].set_ylabel("Long-range fraction\n$f_{\\mathregular{LR}(10\\%)}$")
    # axs[0].set_yticks([0,0.5,1])

    plt.suptitle(dataset)

    # Single shared x-label
    # fig.supxlabel("Wiring cost (total Euclidean distance)") # , # fontsize=8, 
                #   y=0.02)

    plt.tight_layout()
    plt.savefig(
        output_folder / f"scatter_{dataset}_wiring_cost_vs_lr_fraction.pdf",
        dpi=200, bbox_inches="tight" 
    )
    print(output_folder / f"scatter_{dataset}_wiring_cost_vs_lr_fraction.pdf")
    plt.show()


# In[33]:


plt.plot([0.9999938232966319, 0.9998456554936185, 0.9957645230208472, 0.9718742773025172, 0.8166287939283895, 0.3851583923947024, 0.2235158366272425, 0.13503354906178788, 0.07198306826461887, 0.08555081226199002, 0.09624020328898686, 0.08539137546084863, 0.07142682522061006, 0.060798890902574065, 0.043773900187310644, 0.025934191126813833, 0.027995040530486293, 0.03951216374857447, 0.04492666062976558, 0.02214018815166141, 0.012670096009738563, 0.015844205316635773, 0.012552883750104327, 0.011887250233061808, 0.0013663048431581393, 0, 0, 0, 0, 0])
plt.plot([0.9929453441538056, 0.889576561588745, 0.24245526919105875, 0.01813752399499058, 0.000984117900906134, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# In[34]:


for c in dict_with_all_datasets[dataset].columns:
    print(c)


# In[35]:


datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
                    'suarez_MaMI_dataset', 'lexis_data_developing', 
                    'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 
                    ]

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))

pairs_to_plot = [
    ("computational_capacity_notebook_memory_capacity_total_notebook", "computational_capacity_notebook_nonlinear_capacity_total_notebook"),
    # ("mc_lin_cut40_mc_mean", "mc_nonlin_cut40_mc_mean"),
    # ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"), 
    # ("computational_capacity_memory_capacity_total", "computational_capacity_nonlinear_capacity_total"),
    # ("computational_capacity_memory_capacity_total", "computational_capacity_cubic_capacity_total"), 
    # ("mc_lin_cut40_mc_mean", "mc_nonlin_cut40_mc_mean"), 
    ("computational_capacity_notebook_damicelli_memory_capacity_total_notebook",
     "computational_capacity_notebook_damicelli_nonlinear_capacity_total_notebook")
]


for x_col, y_col in pairs_to_plot:
        for dataset in datasets_to_plot:

            plt.scatter(
                    x = dict_with_all_datasets[dataset][x_col], 
                    y = dict_with_all_datasets[dataset][y_col], 
                    color=COLOR_SCHEME[dataset],
                    s=30 if "kayson" in dataset else 5, 
                    alpha=0.2 if "gnm" in dataset else 1)
        plt.ylim(0,0.00002)
        plt.xlabel(x_col, fontsize=6)
        plt.ylabel(y_col, fontsize=6)   
        plt.show()


# In[36]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]


# In[ ]:





# In[37]:


mc_per_dataset = {}


for dataset in datasets_to_plot:
    
    mc_lin_arr = []
    mc_nonlin_arr = []
    
    # iterate trough all rows 
    for idx, row in dict_with_all_datasets[dataset].iterrows():
        
        lin_profile = row["computational_capacity_notebook_damicelli_memory_capacity_profile_notebook"]
        nonlin_profile = row["computational_capacity_notebook_damicelli_nonlinear_capacity_profile_notebook"]

        lin_profile = lin_profile.strip("[]").split(",")
        lin_profile = np.array(lin_profile).astype(np.float32)

        nonlin_profile = nonlin_profile.strip("[]").split(",")
        nonlin_profile = np.array(nonlin_profile).astype(np.float32)

        plt.plot(lin_profile, label="lin", color="blue")
        plt.plot(nonlin_profile, label="nonlin", color="orange")
        
        mc_lin_arr.append(lin_profile.sum())
        mc_nonlin_arr.append(nonlin_profile.sum())

        if idx % 10 == 5:
            print("done")
            break

    mc_per_dataset[dataset] = {
        "lin": mc_lin_arr,
        "nonlin": mc_nonlin_arr
    }

plt.show() 

plt.boxplot([mc_per_dataset[dataset]["lin"] for dataset in datasets_to_plot], 
            positions=np.array(range(len(datasets_to_plot)))*2.0-0.4, 
            widths=0.6, 
            patch_artist=True, 
            boxprops=dict(facecolor="blue", color="blue"),
            medianprops=dict(color="white"))
plt.title("Memory capacity (linear vs nonlinear)")

plt.boxplot([mc_per_dataset[dataset]["nonlin"] for dataset in datasets_to_plot], 
            positions=np.array(range(len(datasets_to_plot)))*2.0+0.4, 
            widths=0.6, 
            patch_artist=True, 
            boxprops=dict(facecolor="orange", color="orange"),
            medianprops=dict(color="white"))

plt.xticks(np.array(range(len(datasets_to_plot)))*2.0, [LABEL_MAP[dataset] for dataset in datasets_to_plot], 
        #    rotation=45, 
        #    ha="right"
           )
plt.tight_layout()


# In[38]:


dataset


# In[39]:


for dataset in dict_with_all_datasets.keys():
    
    column_names = dict_with_all_datasets[dataset].columns
    # drom the following two columns from the notebook, if they are present: computational_capacity_notebook_memory_capacity_profile_notebook, computational_capacity_notebook_nonlinear_capacity_profile_notebook
    column_names = [col for col in column_names if col not in [
        "computational_capacity_notebook_memory_capacity_profile_notebook",
        "computational_capacity_notebook_nonlinear_capacity_profile_notebook", 
        "computational_capacity_notebook_damicelli_memory_capacity_profile_notebook",
        "computational_capacity_notebook_damicelli_nonlinear_capacity_profile_notebook"
    ]]
    dict_with_all_datasets[dataset] = dict_with_all_datasets[dataset][column_names]
    
    
    # Drop all rows that have nans in them (and print how many rows got dropped)
    before_rows = len(dict_with_all_datasets[dataset])
    # dict_with_all_datasets[dataset] = dict_with_all_datasets[dataset].dropna()
    dict_with_all_datasets[dataset] = dict_with_all_datasets[dataset].dropna(axis=1, how="all")
    after_rows = len(dict_with_all_datasets[dataset])
    print(f"{dataset}: Dropped {before_rows - after_rows} rows with NaN values.") #  Remaining rows: {after_rows}.")


# In[40]:


for k in dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].keys():
    print(k)


# In[ ]:





# In[41]:


for dataset in datasets_to_plot:

    plt.scatter(
            # x = dict_with_all_datasets[dataset]["computational_capacity_notebook_memory_capacity_total_notebook"], 
            x = dict_with_all_datasets[dataset]["proportion_long_range_connections_0.3956"], 
            y = dict_with_all_datasets[dataset]["computational_capacity_notebook_nonlinear_capacity_total_notebook"], 
            color=COLOR_SCHEME[dataset],
            s=30 if "kayson" in dataset else 5, 
            alpha=0.2 if "gnm" in dataset else 1)


# In[42]:


for dataset in datasets_to_plot:

    plt.scatter(
            # x = dict_with_all_datasets[dataset]["computational_capacity_notebook_memory_capacity_total_notebook"], 
            x = dict_with_all_datasets[dataset]["computational_capacity_notebook_damicelli_memory_capacity_total_notebook"], 
            y = dict_with_all_datasets[dataset]["computational_capacity_notebook_damicelli_nonlinear_capacity_total_notebook"], 
            color=COLOR_SCHEME[dataset],
            s=30 if "kayson" in dataset else 5, 
            alpha=0.2 if "gnm" in dataset else 1)
    
plt.ylim(0, 0.00002)


# In[43]:


for dataset in datasets_to_plot:

    plt.scatter(
            # x = dict_with_all_datasets[dataset]["computational_capacity_notebook_memory_capacity_total_notebook"], 
            x = dict_with_all_datasets[dataset]["computational_capacity_notebook_memory_capacity_total_notebook"], 
            y = dict_with_all_datasets[dataset]["computational_capacity_notebook_nonlinear_capacity_total_notebook"], 
            color=COLOR_SCHEME[dataset],
            s=30 if "kayson" in dataset else 5, 
            alpha=0.2 if "gnm" in dataset else 1)
    
# plt.ylim(0, 0.00002)


# In[44]:


x_col = "mc_nonlinear_input_scaling_0_1_mc_mean"
y_col = "mc_input_scaling_0_1_mc_mean"

for dataset in datasets_to_plot:

    plt.scatter(
            # x = dict_with_all_datasets[dataset]["computational_capacity_notebook_memory_capacity_total_notebook"], 
            y = dict_with_all_datasets[dataset][y_col], 
            x = dict_with_all_datasets[dataset][x_col], 
            color=COLOR_SCHEME[dataset],
            s=30 if "kayson" in dataset else 5, 
            alpha=0.2 if "gnm" in dataset else 1)

plt.ylabel(PROPERTY_NAMES[y_col])
plt.xlabel(PROPERTY_NAMES[x_col])

    # break
# plt.ylim(0, 0.00002)


# In[45]:


for dataset in datasets_to_plot:

    plt.scatter(
            # x = dict_with_all_datasets[dataset]["computational_capacity_notebook_memory_capacity_total_notebook"], 
            x = dict_with_all_datasets[dataset]["mc_mean"], 
            y = dict_with_all_datasets[dataset]["mc_nonlinear_original_mc_mean"], 
            color=COLOR_SCHEME[dataset],
            s=30 if "kayson" in dataset else 5, 
            alpha=0.2 if "gnm" in dataset else 1)
    break
# plt.ylim(0, 0.00002)


# In[46]:


for dataset in datasets_to_plot:
    plt.scatter(
            x = dict_with_all_datasets[dataset]["proportion_long_range_connections_0.3956"], 
            y = dict_with_all_datasets[dataset]["computational_capacity_notebook_memory_capacity_total_notebook"], 
            color=COLOR_SCHEME[dataset],
            s=30 if "kayson" in dataset else 5, 
            alpha=0.2 if "gnm" in dataset else 1)


# In[47]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]


# In[48]:


# dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].drop(columns=["computational_capacity_notebook_damicelli_memory_capacity_profile_notebook", "computational_capacity_notebook_damicelli_nonlinear_capacity_profile_notebook"], inplace=True, errors="ignore")


# In[49]:


# Print dtypes of columns of dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]
print(dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].dtypes)    

# Drop all columns that are not of type float or int
for dataset in dict_with_all_datasets.keys():
    columns_to_drop = [col for col in dict_with_all_datasets[dataset].columns if dict_with_all_datasets[dataset][col].dtype not in [np.float64, np.float32, np.int64, np.int32]]
    dict_with_all_datasets[dataset].drop(columns=columns_to_drop, inplace=True)
    print(f"{dataset}: Dropped columns: {columns_to_drop}")


# In[50]:


# df_gnm_g = df_gnm.groupby(["eta", "gamma"]).mean().reset_index()
df_gnm_g = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].groupby(["eta", "gamma"]).mean().reset_index() #

# EXCLUDE COLUMNS: 
constant_columns = ["avg_degree", "n_connected_components", "density", "density_bct", "kernel_rank_phase_of_lambda_max"]
df_gnm_g = df_gnm_g.drop(columns=[col for col in constant_columns if col in df_gnm_g.columns])

cols_of_interest = [col for col in df_gnm_g.columns if col not in ["eta", "gamma"]]
len(cols_of_interest)


n_cols = 8 
n_rows = (len(cols_of_interest) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((n_cols * 3, n_rows * 3)), sharex=True, sharey=True, dpi=200)
for idx, col in enumerate(cols_of_interest):
    row = idx // n_cols
    col_idx = idx % n_cols
    ax = axes[row, col_idx]
    im = ax.imshow(df_gnm_g.pivot(index="gamma", columns="eta", values=col),
                   aspect="equal", origin="lower", 
                   cmap="RdBu" # viridis" "berlin"
                )

    # if the title is too long (define this), then split it into two lines at the last underscore
    if len(col) > 20:  
        col = re.split(r'[,,_]+', col) # THIS IS OBV UGGLY: SEE ENERGY!! TODO Add comata again
        # col = col.split("_")
        len_col = len(col)
        col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
    ax.set_title(col, fontsize=6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=4) 

# remove empty subplots
for idx in range(len(cols_of_interest), n_rows * n_cols):
    row = idx // n_cols
    col_idx = idx % n_cols
    fig.delaxes(axes[row, col_idx])
    
plt.tight_layout(pad=0.4)

plt.savefig(output_folder / "gnm_property_heatmaps.pdf", dpi=200)
print(output_folder / "gnm_property_heatmaps.pdf")


# In[51]:


import re
from config import PROPERTY_NAMES

# ── Category definitions ──────────────────────────────────────────────────────
categories = {
    "Fundamental Topology": [ 
        "transitivity", "avg_clustering", "modularity", "degree_gini", # Basic
        "degree_assortativity", "omega", "structural_complexity", 
        
        "directed_simplices_count", # Higher order
        "directed_simplices_max_size",
        
        "energy" # i.e.: Fit to networks - maybe remove again? 
    ],
    "Paths, Efficiency & Communication": [
        "char_path_length", "global_efficiency", "diffusion_efficiency",
        "propagation_efficiency", "avg_communicability",
        "topological_distance_mean", "topological_distance_std",
    ],
    "Spatial Embedding & Wiring Cost": [
        "avg_edge_distance", "wiring_cost",
        "proportion_long_range_connections_0.1",
        "proportion_long_range_connections_0.3",
        "proportion_long_range_connections_0.3956",
        "proportion_long_range_connections_0.5",
    ],
    "Rich-Club Organization": [
        "richclub_n_edges", "richclub_avg_length",
        "rich_club_coefficient_rc_max_norm",
        "rich_club_coefficient_rc_mean_norm",
        "rich_club_coefficient_rc_k_at_max",
        "rich_club_coefficient_rc_regime_frac",
        "rich_club_coefficient_rc_weighted_auc",
    ],
    "Spectral Properties, Algebraic Connectivity & Fiedler Analysis": [
        "spectral_radius", 
        # "synchronizability_eigenratio_lambda_N", # largest eigenvalue. -> spectral radius
        "spectral_gap", # Has exactly the same output as Fatemehs version, bc it is identical (lambda_n - lambda_(n-1))
        # "spectral_gap_fatemeh", 
    # ],
    # "Synchronization & Eigenratio": [
        "synchronizability_eigenratio_eigenratio", # lambda_n / lambda_2
    # ],
        "kernel_rank_thresholded_and_summed_0.01",          # np.sum(np.abs(eigs) > threshold * np.abs(eigs).max())
        # "kernel_rank_max",                                  # biggest eigenvalue
        "kernel_rank_phase_diff_of_lambda_max_and_2nd",     # np.angle(eigs[np.argsort(np.abs(eigs))[-2]]) - np.angle(eigs.max())
                # eigs = np.linalg.eigvals(A)
                #     "thresholded_and_summed_0.01": np.sum(np.abs(eigs) > threshold * np.abs(eigs).max()), 
                #     "max": np.abs(eigs).max(), 
                #     "phase_of_lambda_max": np.angle(eigs.max()), 
                #     "phase_diff_of_lambda_max_and_2nd": np.angle(eigs[np.argsort(np.abs(eigs))[-2]]) - np.angle(eigs.max()), 
                
    # "Algebraic Connectivity & Fiedler Analysis": [
        # "algebraic_connectivity_nx", # Identical to fiedler value....
        "algebraic_connectivity_fiedler_value", # Relies on a different library, I hope? 
        "algebraic_connectivity_fiedler_value_norm",
        # "synchronizability_eigenratio_lambda_2", # Should be fiedler value...  (first non-zero)
        "algebraic_connectivity_laplacian_spectral_gap",
        "algebraic_connectivity_fiedler_bipartition_balance",
        
        "departure_from_normality_schur",
                # """
                # from scipy.linalg import solve_continuous_lyapunov, schur
                # Henrici departure from normality via Schur decomposition.
                
                # For M = QTQ*, the departure equals ||T_off||_F / ||M||_F,
                # where T_off is the strictly upper-triangular part of T.
                
                # This avoids the numerically unstable subtraction
                # ||M||_F^2 - sum|lambda_i|^2 from the eigenvalue-based formula.
                # """
                # A = np.array(A, dtype=np.complex128)

                # norm_F = np.linalg.norm(A, 'fro')
                # if norm_F < 1e-12:
                #     return 0.0
                
                # # Schur decomposition: M = Q T Q^H
                # T, _ = schur(A, output='complex')

                # # Strictly upper-triangular part = non-normal component
                # T_off = np.triu(T, k=1)
                
                # return float(np.linalg.norm(T_off, 'fro') / norm_F)
            
                # ____________________________________________
                # M = np.array(M)
                # # Calculate the Frobenius norm of M
                # norm_F = linalg.norm(M, 'fro')
                # # Compute the eigenvalues of M
                # eigenvalues = np.linalg.eigvals(M)
                # # Compute the sum of the squares of the eigenvalues
                # sum_squares_eigenvalues = np.sum(np.abs(eigenvalues)**2)
                # # Compute the departure from normality
                # d_F = np.sqrt(norm_F**2 - sum_squares_eigenvalues)
                # return d_F/norm_F

        "effective_dimensionality",
                # eigs = np.linalg.eigvals(A)
                # eigs_abs = np.abs(eigs)
                # return np.sum(eigs_abs)**2 / np.sum(eigs_abs**2)
    ],
    # "Kuramoto Synchronization (Single Trial)": [
    #     "kuramoto_synchronization_r_final",
    #     "kuramoto_synchronization_r_mean",
    #     "kuramoto_synchronization_r_std",
    # ],
    "Metastability & Kuramoto Synchronization": [ # (Multi-Trial Averaged)": [
        "repertoire_sweep_weighted_by_distances_T_critical",
        "repertoire_sweep_weighted_by_distances_size_critical",
        "repertoire_sweep_weighted_by_distances_diversity_critical", 
        
        "kuramoto_averaged_synchronization_r_final",
        "kuramoto_averaged_synchronization_r_mean",
        "kuramoto_averaged_synchronization_r_mean_se",
        "kuramoto_averaged_synchronization_r_std",
        "kuramoto_averaged_synchronization_r_std_se",
    # ],
    # "Metastability": [
        # "repertoire_size", -> sweeping and weighting are very important
        # "repertoire_diversity",  -> sweeping and weighting are very important
    ],

    "Participation Coefficient (Louvain)": [
        "participation_coefficient_n_communities",
        "participation_coefficient_pc_mean",
        "participation_coefficient_pc_median",
        "participation_coefficient_pc_std",
        "participation_coefficient_pc_frac_connector",
        # "participation_coefficient_wmd_mean", 
                # # The within-module degree z-score (WMD) mean is theoretically exactly zero by construction — 
                # it's a z-score, so across all nodes within each module, the mean is always 0. 
                # What you're seeing (1e-8 range) is pure floating-point numerical noise, not a real signal.
                # This means participation_coefficient_wmd_mean is essentially uninformative and 
                # you can safely drop it. The noisy salt-and-pepper pattern in that panel is the giveaway — 
                # it looks like random rounding error, not a structured response to η/γ.
        "participation_coefficient_wmd_std",
            # The std (wmd_std) is still informative because it captures how spread the hub structure 
            # is across modules, even though the mean is anchored at zero. You can see it has a clean 
            # structured pattern in your plot.
    ],
    # "Participation Coefficient (Further Partition)": [ # exactly identical as the code above... 
    #     "participation_coefficient_n_communities_further",
    #     "participation_coefficient_pc_mean_further",
    #     "participation_coefficient_pc_median_further",
    #     "participation_coefficient_pc_std_further",
    #     "participation_coefficient_pc_frac_connector_further",
    #     "participation_coefficient_wmd_mean_further",
    #     "participation_coefficient_wmd_std_further",
    # ],
    "Ollivier-Ricci Curvature": [
        "ollivier_ricci_curvature_orc_mean",
        "ollivier_ricci_curvature_orc_median",
        "ollivier_ricci_curvature_orc_std",
        "ollivier_ricci_curvature_orc_min",
        "ollivier_ricci_curvature_orc_max",
        "ollivier_ricci_curvature_orc_skewness",
        "ollivier_ricci_curvature_orc_frac_neg",
    ],
    "Persistent Homology (TDA)": [
        # "persistent_homology_ph_h0_n_features",
        # "persistent_homology_ph_h0_persistence_mean",
        # "persistent_homology_ph_h0_persistence_std",
        # "persistent_homology_ph_h0_entropy",
        "persistent_homology_ph_h1_n_features",
        "persistent_homology_ph_h1_persistence_mean",
        # "persistent_homology_ph_h1_persistence_std",
        "persistent_homology_ph_h1_entropy",
        "persistent_homology_ph_total_persistence",
    ],

    "Targeted Attack Robustness & Community Structure & Vulnerability": [
        "targeted_attack_robustness_rob_targeted_auc",
        "targeted_attack_robustness_rob_targeted_half",
        "targeted_attack_robustness_rob_random_auc",
        "targeted_attack_robustness_rob_random_half",
        "targeted_attack_robustness_rob_ratio",
        
        "community_synchronization_vulnerability_value",
        "community_synchronization_vulnerability_n_communities",
            # Measures cross-community coupling via:
            #   1. Between/within coupling ratio — global vulnerability score (corrected).
    ],
    "Network Control Theory - Average Controllability": [
        "nct_control_avg", "nct_control_std", "nct_control_max",
        "nct_control_n_nodes_90_percent",
        "nct_control_n_nodes_50_percent",
        "nct_control_n_nodes_10_percent",
    ],
    "Network Control Theory - Control Energy": [
        "nct_energies_total", "nct_energies_std", "nct_energies_max",
        "nct_energies_n_nodes_90_percent",
        "nct_energies_n_nodes_50_percent",
        "nct_energies_n_nodes_10_percent",
    ],
    "Computational Capacity": [
        "computational_capacity_memory_capacity_total",
        "computational_capacity_memory_timescale",
        "computational_capacity_nonlinear_capacity_total",
        "computational_capacity_cubic_capacity_total",
        "computational_capacity_cross_capacity_total",
        "computational_capacity_memory_nonlinear_ratio",
        "computational_capacity_total_capacity",
        "computational_capacity_state_dimensionality",
        "computational_capacity_state_entropy",
        "computational_capacity_state_rank",
        "computational_capacity_separation_ratio",
        "computational_capacity_lyapunov_exponent",
    ],
    # "Reservoir Computing - Basic Measures": 
    "Memory Capacity - Full Lag Profile": 
        [
            f"mc_{i}" for i in range(1, 50)
        ] + [
            "mc_mean", "mc_std", #  "mc_nonlinear_original_mc_mean", "mc_nonlinear_original_mc_std"
        ] + [
            # "mc_input_scaling_0_1_mc_mean", "mc_input_scaling_0_1_mc_std", 
            # "mc_nonlinear_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_std", 
            # "mc_lin_cut40",
            # "mc_nonlin_cut40",
            
            
            # "mc_lin_cut40_mc_mean", 
            # "mc_lin_cut40_mc_std",
            # "mc_lin_cut40_mc_0",
            # "mc_lin_cut40_mc_1",
            # "mc_lin_cut40_mc_2",
            # "mc_lin_cut40_mc_3",
            # "mc_lin_cut40_mc_4",
            # "mc_lin_cut40_mc_5",
            # "mc_lin_cut40_mc_6",
            # "mc_lin_cut40_mc_7",
            # "mc_lin_cut40_mc_8",
            # "mc_lin_cut40_mc_9",
            # "mc_lin_cut40_mc_10",
            # "mc_lin_cut40_mc_11",
            # "mc_lin_cut40_mc_12",
            # "mc_lin_cut40_mc_13",
            # "mc_lin_cut40_mc_14",
            # "mc_lin_cut40_mc_15",
            # "mc_lin_cut40_mc_16",
            # "mc_lin_cut40_mc_17",
            # "mc_lin_cut40_mc_18",
            # "mc_lin_cut40_mc_19",
            # "mc_lin_cut40_mc_20",
            # "mc_lin_cut40_mc_21",
            # "mc_lin_cut40_mc_22",
            # "mc_lin_cut40_mc_23",
            # "mc_lin_cut40_mc_24",
            # "mc_lin_cut40_mc_25",
            # "mc_lin_cut40_mc_26",
            # "mc_lin_cut40_mc_27",
            # "mc_lin_cut40_mc_28",
            # "mc_lin_cut40_mc_29",
            # "mc_lin_cut40_mc_30",
            # "mc_lin_cut40_mc_31",
            # "mc_lin_cut40_mc_32",
            # "mc_lin_cut40_mc_33",
            # "mc_lin_cut40_mc_34",
            # "mc_lin_cut40_mc_35",
            # "mc_lin_cut40_mc_36",
            # "mc_lin_cut40_mc_37",
            # "mc_lin_cut40_mc_38",
            # "mc_lin_cut40_mc_39",
            # "mc_nonlin_cut40_mc_mean",
            # "mc_nonlin_cut40_mc_std", 
            
            # "computational_capacity_notebook_memory_capacity_total_notebook", 
            # "computational_capacity_notebook_nonlinear_capacity_total_notebook", 
            
            # "computational_capacity_notebook_damicelli", 
            # "computational_capacity_notebook_damicelli_memory_capacity_total_notebook",
            # "computational_capacity_notebook_damicelli_nonlinear_capacity_total_notebook",
            
            "mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean",
            
            "ipc_ipc_deg1_mean", "ipc_ipc_deg1_std", 
            "ipc_ipc_deg2_mean", "ipc_ipc_deg2_std",
            # "ipc_ipc_deg3_mean", "ipc_ipc_deg3_std",
            # "ipc_ipc_linear_mean", "ipc_ipc_nonlinear_mean",
            # "ipc_ipc_total_mean", "ipc_ipc_total_std"
        ],
        "IPC": [
            # Remove again? 
            "ipc_ipc_deg1_mean", "ipc_ipc_deg1_std",
            "ipc_ipc_deg2_mean", "ipc_ipc_deg2_std",
            "ipc_ipc_deg3_mean", "ipc_ipc_deg3_std",
            "ipc_ipc_linear_mean", "ipc_ipc_nonlinear_mean",
            "ipc_ipc_total_mean", "ipc_ipc_total_std"
        ]

}

# ── Coverage check ────────────────────────────────────────────────────────────
all_categorised = set(col for cols in categories.values() for col in cols)
uncategorised = [c for c in cols_of_interest if c not in all_categorised]
if uncategorised:
    print(f"⚠️  {len(uncategorised)} column(s) not assigned to any category:")
    for c in uncategorised:
        print(f"   • {c}")
else:
    print("✅  All columns covered.")

# ── Helper: pretty-wrap long titles ──────────────────────────────────────────
def wrap_title(col: str, max_len: int = 20) -> str:
    if len(col) <= max_len:
        return col
    parts = re.split(r'[_,]+', col)
    mid = len(parts) // 2
    return "_".join(parts[:mid]) + "\n" + "_".join(parts[mid:])

# ── Plot one figure per category ──────────────────────────────────────────────
n_cols = 4

for cat_name, cat_cols in categories.items():
    # keep only columns that actually exist in the dataframe
    present = [c for c in cat_cols if c in df_gnm_g.columns]
    if not present:
        print(f"⚠️  Skipping '{cat_name}' - no columns found in dataframe.")
        continue

    n_rows = (len(present) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=viz.cm_to_inch((n_cols * 3.5, n_rows * 3.5)),
        sharex=True, sharey=True,
        dpi=200,
    )
    # normalise axes to always be 2-D
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle(cat_name, fontsize=8, fontweight="bold", y=1.01)

    for idx, col in enumerate(present):
        ax = axes[idx // n_cols, idx % n_cols]
        pivoted = df_gnm_g.pivot(index="gamma", columns="eta", values=col)
        vabs = np.nanmax(np.abs(pivoted.values))  # symmetric around 0
        im = ax.imshow(
            pivoted,
            aspect="equal",
            origin="lower",
            cmap="RdBu_r",
            # vmin=-vabs, vmax=vabs, 
        )
        # ax.set_title(wrap_title(PROPERTY_NAMES[col]), fontsize=5)
        ax.set_title(col, fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=4)
        # The next line forces the e-notation to be at the ticks, such that the labels don't collide with the title. 
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1e}")
        )

    # remove unused axes
    for idx in range(len(present), n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols, idx % n_cols])

    plt.tight_layout(pad=0.5)

    safe_name = re.sub(r'[^\w]+', '_', cat_name).strip('_').lower()
    out_path = output_folder / f"gnm_heatmaps_{safe_name}.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


# In[52]:


import re
from config import PROPERTY_NAMES

# ── Category definitions ──────────────────────────────────────────────────────
categories = {
    "Fundamental Topology": [ 
        "transitivity", "avg_clustering", "modularity", "degree_gini", # Basic
        "degree_assortativity", "omega", "structural_complexity", 
        
        "directed_simplices_count", # Higher order
        "directed_simplices_max_size",
        
        "energy", # i.e.: Fit to networks - maybe remove again? 
        "eta", 
        "gamma", 
    ],
    "Paths, Efficiency & Communication": [
        "char_path_length", "global_efficiency", "diffusion_efficiency",
        "propagation_efficiency", "avg_communicability",
        "topological_distance_mean", "topological_distance_std",
    ],
    "Spatial Embedding & Wiring Cost": [
        "avg_edge_distance", "wiring_cost",
        "proportion_long_range_connections_0.1",
        "proportion_long_range_connections_0.3",
        "proportion_long_range_connections_0.3956",
        "proportion_long_range_connections_0.5",
    ],
    "Rich-Club Organization": [
        "richclub_n_edges", "richclub_avg_length",
        "rich_club_coefficient_rc_max_norm",
        "rich_club_coefficient_rc_mean_norm",
        "rich_club_coefficient_rc_k_at_max",
        "rich_club_coefficient_rc_regime_frac",
        "rich_club_coefficient_rc_weighted_auc",
    ],
    "Spectral Properties, Algebraic Connectivity & Fiedler Analysis": [
        "spectral_radius", 
        # "synchronizability_eigenratio_lambda_N", # largest eigenvalue. -> spectral radius
        "spectral_gap", # Has exactly the same output as Fatemehs version, bc it is identical (lambda_n - lambda_(n-1))
        # "spectral_gap_fatemeh", 
    # ],
    # "Synchronization & Eigenratio": [
        "synchronizability_eigenratio_eigenratio", # lambda_n / lambda_2
    # ],
        "kernel_rank_thresholded_and_summed_0.01",          # np.sum(np.abs(eigs) > threshold * np.abs(eigs).max())
        # "kernel_rank_max",                                  # biggest eigenvalue
        "kernel_rank_phase_diff_of_lambda_max_and_2nd",     # np.angle(eigs[np.argsort(np.abs(eigs))[-2]]) - np.angle(eigs.max())
                # eigs = np.linalg.eigvals(A)
                #     "thresholded_and_summed_0.01": np.sum(np.abs(eigs) > threshold * np.abs(eigs).max()), 
                #     "max": np.abs(eigs).max(), 
                #     "phase_of_lambda_max": np.angle(eigs.max()), 
                #     "phase_diff_of_lambda_max_and_2nd": np.angle(eigs[np.argsort(np.abs(eigs))[-2]]) - np.angle(eigs.max()), 
                
    # "Algebraic Connectivity & Fiedler Analysis": [
        # "algebraic_connectivity_nx", # Identical to fiedler value....
        "algebraic_connectivity_fiedler_value", # Relies on a different library, I hope? 
        "algebraic_connectivity_fiedler_value_norm",
        # "synchronizability_eigenratio_lambda_2", # Should be fiedler value...  (first non-zero)
        "algebraic_connectivity_laplacian_spectral_gap",
        "algebraic_connectivity_fiedler_bipartition_balance",
        
        "departure_from_normality_schur",
                # """
                # from scipy.linalg import solve_continuous_lyapunov, schur
                # Henrici departure from normality via Schur decomposition.
                
                # For M = QTQ*, the departure equals ||T_off||_F / ||M||_F,
                # where T_off is the strictly upper-triangular part of T.
                
                # This avoids the numerically unstable subtraction
                # ||M||_F^2 - sum|lambda_i|^2 from the eigenvalue-based formula.
                # """
                # A = np.array(A, dtype=np.complex128)

                # norm_F = np.linalg.norm(A, 'fro')
                # if norm_F < 1e-12:
                #     return 0.0
                
                # # Schur decomposition: M = Q T Q^H
                # T, _ = schur(A, output='complex')

                # # Strictly upper-triangular part = non-normal component
                # T_off = np.triu(T, k=1)
                
                # return float(np.linalg.norm(T_off, 'fro') / norm_F)
            
                # ____________________________________________
                # M = np.array(M)
                # # Calculate the Frobenius norm of M
                # norm_F = linalg.norm(M, 'fro')
                # # Compute the eigenvalues of M
                # eigenvalues = np.linalg.eigvals(M)
                # # Compute the sum of the squares of the eigenvalues
                # sum_squares_eigenvalues = np.sum(np.abs(eigenvalues)**2)
                # # Compute the departure from normality
                # d_F = np.sqrt(norm_F**2 - sum_squares_eigenvalues)
                # return d_F/norm_F

        "effective_dimensionality",
                # eigs = np.linalg.eigvals(A)
                # eigs_abs = np.abs(eigs)
                # return np.sum(eigs_abs)**2 / np.sum(eigs_abs**2)
    ],
    # "Kuramoto Synchronization (Single Trial)": [
    #     "kuramoto_synchronization_r_final",
    #     "kuramoto_synchronization_r_mean",
    #     "kuramoto_synchronization_r_std",
    # ],
    "Metastability & Kuramoto Synchronization": [ # (Multi-Trial Averaged)": [
        "repertoire_sweep_weighted_by_distances_T_critical",
        "repertoire_sweep_weighted_by_distances_size_critical",
        "repertoire_sweep_weighted_by_distances_diversity_critical", 
        
        "kuramoto_averaged_synchronization_r_final",
        "kuramoto_averaged_synchronization_r_mean",
        "kuramoto_averaged_synchronization_r_mean_se",
        "kuramoto_averaged_synchronization_r_std",
        "kuramoto_averaged_synchronization_r_std_se",
    # ],
    # "Metastability": [
        # "repertoire_size", -> sweeping and weighting are very important
        # "repertoire_diversity",  -> sweeping and weighting are very important
    ],

    "Participation Coefficient (Louvain)": [
        "participation_coefficient_n_communities",
        "participation_coefficient_pc_mean",
        "participation_coefficient_pc_median",
        "participation_coefficient_pc_std",
        "participation_coefficient_pc_frac_connector",
        # "participation_coefficient_wmd_mean", 
                # # The within-module degree z-score (WMD) mean is theoretically exactly zero by construction — 
                # it's a z-score, so across all nodes within each module, the mean is always 0. 
                # What you're seeing (1e-8 range) is pure floating-point numerical noise, not a real signal.
                # This means participation_coefficient_wmd_mean is essentially uninformative and 
                # you can safely drop it. The noisy salt-and-pepper pattern in that panel is the giveaway — 
                # it looks like random rounding error, not a structured response to η/γ.
        "participation_coefficient_wmd_std",
            # The std (wmd_std) is still informative because it captures how spread the hub structure 
            # is across modules, even though the mean is anchored at zero. You can see it has a clean 
            # structured pattern in your plot.
    ],
    # "Participation Coefficient (Further Partition)": [ # exactly identical as the code above... 
    #     "participation_coefficient_n_communities_further",
    #     "participation_coefficient_pc_mean_further",
    #     "participation_coefficient_pc_median_further",
    #     "participation_coefficient_pc_std_further",
    #     "participation_coefficient_pc_frac_connector_further",
    #     "participation_coefficient_wmd_mean_further",
    #     "participation_coefficient_wmd_std_further",
    # ],
    "Ollivier-Ricci Curvature": [
        "ollivier_ricci_curvature_orc_mean",
        "ollivier_ricci_curvature_orc_median",
        "ollivier_ricci_curvature_orc_std",
        "ollivier_ricci_curvature_orc_min",
        "ollivier_ricci_curvature_orc_max",
        "ollivier_ricci_curvature_orc_skewness",
        "ollivier_ricci_curvature_orc_frac_neg",
    ],
    "Persistent Homology (TDA)": [
        # "persistent_homology_ph_h0_n_features",
        # "persistent_homology_ph_h0_persistence_mean",
        # "persistent_homology_ph_h0_persistence_std",
        # "persistent_homology_ph_h0_entropy",
        "persistent_homology_ph_h1_n_features",
        "persistent_homology_ph_h1_persistence_mean",
        # "persistent_homology_ph_h1_persistence_std",
        "persistent_homology_ph_h1_entropy",
        "persistent_homology_ph_total_persistence",
    ],

    "Targeted Attack Robustness & Community Structure & Vulnerability": [
        "targeted_attack_robustness_rob_targeted_auc",
        "targeted_attack_robustness_rob_targeted_half",
        "targeted_attack_robustness_rob_random_auc",
        "targeted_attack_robustness_rob_random_half",
        "targeted_attack_robustness_rob_ratio",
        
        "community_synchronization_vulnerability_value",
        "community_synchronization_vulnerability_n_communities",
            # Measures cross-community coupling via:
            #   1. Between/within coupling ratio — global vulnerability score (corrected).
    ],
    "Network Control Theory - Average Controllability": [
        "nct_control_avg", "nct_control_std", "nct_control_max",
        "nct_control_n_nodes_90_percent",
        "nct_control_n_nodes_50_percent",
        "nct_control_n_nodes_10_percent",
    ],
    "Network Control Theory - Control Energy": [
        "nct_energies_total", "nct_energies_std", "nct_energies_max",
        "nct_energies_n_nodes_90_percent",
        "nct_energies_n_nodes_50_percent",
        "nct_energies_n_nodes_10_percent",
    ],
    "Computational Capacity": [
        # "computational_capacity_memory_capacity_total",
        # "computational_capacity_memory_timescale",
        # "computational_capacity_nonlinear_capacity_total",
        # "computational_capacity_cubic_capacity_total",
        # "computational_capacity_cross_capacity_total",
        # "computational_capacity_memory_nonlinear_ratio",
        # "computational_capacity_total_capacity",
        # "computational_capacity_state_dimensionality",
        # "computational_capacity_state_entropy",
        # "computational_capacity_state_rank",
        # "computational_capacity_separation_ratio",
        # "computational_capacity_lyapunov_exponent",
    ],
    # "Reservoir Computing - Basic Measures": 
    "Memory Capacity - Full Lag Profile": (
        # [f"mc_{i}" for i in range(1, 50)] + [  # + ["mc_mean", "mc_std"]
        # "mc_input_scaling_0_1_mc_mean", # "mc_input_scaling_0_1_mc_std", 
        # "mc_nonlinear_input_scaling_0_1_mc_mean", # "mc_nonlinear_input_scaling_0_1_mc_std"
        # ]+ [
            ["mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean", 
             
            "ipc_ipc_deg1_mean", "ipc_ipc_deg1_std", 
            "ipc_ipc_deg2_mean", "ipc_ipc_deg2_std"]
            # "mc_input_scaling_0_1_mc_mean", "mc_input_scaling_0_1_mc_std", 
            # "mc_nonlinear_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_std", 
            # "mc_lin_cut40",
            # "mc_nonlin_cut40",
            
            
            # "mc_lin_cut40_mc_mean", 
            # "mc_lin_cut40_mc_std",
            # "mc_lin_cut40_mc_0",
            # "mc_lin_cut40_mc_1",
            # "mc_lin_cut40_mc_2",
            # "mc_lin_cut40_mc_3",
            # "mc_lin_cut40_mc_4",
            # "mc_lin_cut40_mc_5",
            # "mc_lin_cut40_mc_6",
            # "mc_lin_cut40_mc_7",
            # "mc_lin_cut40_mc_8",
            # "mc_lin_cut40_mc_9",
            # "mc_lin_cut40_mc_10",
            # "mc_lin_cut40_mc_11",
            # "mc_lin_cut40_mc_12",
            # "mc_lin_cut40_mc_13",
            # "mc_lin_cut40_mc_14",
            # "mc_lin_cut40_mc_15",
            # "mc_lin_cut40_mc_16",
            # "mc_lin_cut40_mc_17",
            # "mc_lin_cut40_mc_18",
            # "mc_lin_cut40_mc_19",
            # "mc_lin_cut40_mc_20",
            # "mc_lin_cut40_mc_21",
            # "mc_lin_cut40_mc_22",
            # "mc_lin_cut40_mc_23",
            # "mc_lin_cut40_mc_24",
            # "mc_lin_cut40_mc_25",
            # "mc_lin_cut40_mc_26",
            # "mc_lin_cut40_mc_27",
            # "mc_lin_cut40_mc_28",
            # "mc_lin_cut40_mc_29",
            # "mc_lin_cut40_mc_30",
            # "mc_lin_cut40_mc_31",
            # "mc_lin_cut40_mc_32",
            # "mc_lin_cut40_mc_33",
            # "mc_lin_cut40_mc_34",
            # "mc_lin_cut40_mc_35",
            # "mc_lin_cut40_mc_36",
            # "mc_lin_cut40_mc_37",
            # "mc_lin_cut40_mc_38",
            # "mc_lin_cut40_mc_39",
            # "mc_nonlin_cut40_mc_mean",
            # "mc_nonlin_cut40_mc_std"
        # ]
    ),
}


# In[53]:


precise_categories = []

for cat_name, cat_cols in categories.items(): 
    
    # if cat_name == "Memory Capacity - Full Lag Profile": 
        # precise_categories.append("mc_mean")
        # precise_categories.append("mc_lin_cut40_mc_mean")
        # precise_categories.append("mc_nonlin_cut40_mc_mean")
        
        # precise_categories = precise_categories + ["computational_capacity_notebook_memory_capacity_total_notebook", 
        #                                             "computational_capacity_notebook_nonlinear_capacity_total_notebook", "computational_capacity_notebook_damicelli_memory_capacity_profile_notebook", 
        #                                             "computational_capacity_notebook_damicelli_nonlinear_capacity_profile_notebook"
        #                                         ]
        # continue

    for col in cat_cols:

        if col == "energy":
            continue

        precise_categories.append(col)


# Save this as precise_categories (pkl) 
with open(output_folder / "selected_categories.pkl", "wb") as f:
    pickle.dump(precise_categories, f)


print(output_folder / "selected_categories.pkl")
precise_categories


# In[54]:


dict_with_all_datasets_precise_categories = {}
for dataset_name, df in dict_with_all_datasets.items():
    cols_to_keep = [col for col in precise_categories if col in df.columns]
    dict_with_all_datasets_precise_categories[dataset_name] = df[cols_to_keep]
    print(f"{dataset_name}: kept {len(cols_to_keep)} columns out of {len(df.columns)}")


# In[55]:


# save as pkl
with open(output_folder / "all_datasets_precise_categories.pkl", "wb") as f:
    pickle.dump(dict_with_all_datasets_precise_categories, f)
    
print(output_folder / "all_datasets_precise_categories.pkl")


# # Some trade-offs (see 13_3_1_2 for more)

# In[56]:


dict_with_all_datasets_precise_categories.keys()

datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 'suarez_MaMI_dataset', 'lexis_data_developing']


# In[57]:


datasets_to_plot = ['hcp_schaefer_100_dataset_gnm', 
                    'suarez_MaMI_dataset', 'lexis_data_developing', 
                    'kaysons_generated_networks_diffusion', 'kaysons_generated_networks_propagation', 'kaysons_generated_networks_routing', 
                    ]

for dataset in datasets_to_plot:

    plt.scatter(
            x = dict_with_all_datasets[dataset]["computational_capacity_notebook_memory_capacity_total_notebook"], 
            # x = dict_with_all_datasets[dataset]["proportion_long_range_connections_0.3956"], 
            y = dict_with_all_datasets[dataset]["computational_capacity_notebook_nonlinear_capacity_total_notebook"], 
            color=COLOR_SCHEME[dataset],
            s=30 if "kayson" in dataset else 5, 
            alpha=0.2 if "gnm" in dataset else 1)


# In[58]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

x_col = "computational_capacity_memory_capacity_total"
y_col = "mc_nonlin_cut40_mc_mean" # 
x_col = "mc_lin_cut40_mc_mean"
# x_col = 'proportion_long_range_connections_0.3956'
# plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
#             color="gray", linewidth=0, s=5, alpha=0.2) 

if len(col) > 5:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in datasets_to_plot: # dict_with_all_datasets_precise_categories.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    # if dataset == "hcp_schaefer_100_dataset_gnm":
    #     continue
    df_merged = dict_with_all_datasets_precise_categories[dataset]
    print(df_merged.keys())

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
plt.xlabel(x_col) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.ylabel(col_for_title) # , fontsize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

plt.tight_layout()


# In[59]:


for c in dict_with_all_datasets_precise_categories[dataset].columns:
    print(c)


# In[60]:


#  "mc_nonlinear_original_mc_mean", "mc_nonlinear_original_mc_std"
#         ] + [
#             "mc_input_scaling_0_1_mc_mean", "mc_input_scaling_0_1_mc_std", 
#             "mc_nonlinear_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_std", 


# In[61]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))
x_col = "mc_nonlinear_original_mc_mean"
y_col = "mc_original_mc_mean"    
# y_col = "mc_lin_cut40_mc_mean" >> doof, wird richtig weird abgeschnitten... 
# x_col = "mc_nonlin_cut40_mc_mean"
# y_col = "computational_capacity_memory_capacity_total" # computational_capacity_memory_timescale"
# x_col = "computational_capacity_nonlinear_capacity_total"
# y_col = "computational_capacity_notebook_memory_capacity_total_notebook" # "computational_capacity_memory_capacity_total"
# y_col = "computational_capacity_notebook_nonlinear_capacity_total_notebook" # "mc_nonlin_cut40_mc_mean" # 
# x_col = "mc_lin_cut40_mc_mean"
# x_col = 'proportion_long_range_connections_0.3956'
# plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
#             color="gray", linewidth=0, s=5, alpha=0.2) 

if len(col) > 5:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in datasets_to_plot: # dict_with_all_datasets_precise_categories.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    # if dataset == "hcp_schaefer_100_dataset_gnm":
    #     continue
    df_merged = dict_with_all_datasets_precise_categories[dataset]
    print(df_merged.keys())

    if y_col in df_merged.columns:
        plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                    df_merged[y_col],
                #    c = df_merged[col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    label=dataset,
                    linewidth=0.25, 
                    alpha=0.2 if "gnm" in dataset else 1,
                    s=10
                    )

# plt.xscale("log")
plt.xlabel(x_col) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.ylabel(col_for_title) # , fontsize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

plt.tight_layout()


# In[62]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))

pairs_to_plot = [
    # ("computational_capacity_notebook_memory_capacity_total_notebook", "computational_capacity_notebook_nonlinear_capacity_total_notebook"),
    # ("mc_lin_cut40_mc_mean", "mc_nonlin_cut40_mc_mean"),
    # ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"), 
    # ("computational_capacity_memory_capacity_total", "computational_capacity_nonlinear_capacity_total"),
    # ("computational_capacity_memory_capacity_total", "computational_capacity_cubic_capacity_total"), 
    # ("mc_lin_cut40_mc_mean", "mc_nonlin_cut40_mc_mean"), 
    ("computational_capacity_notebook_damicelli_memory_capacity_profile_notebook", "computational_capacity_notebook_damicelli_nonlinear_capacity_profile_notebook")
]


for x_col, y_col in pairs_to_plot:

    if len(col) > 5:
        col_for_title = re.split(r'[,,_]+', y_col)
        len_col = len(col_for_title)
        col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

    for dataset in datasets_to_plot: 
        
        df_merged = dict_with_all_datasets_precise_categories[dataset]
        # print(df_merged.keys())

        if y_col in df_merged.columns:
            plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                        df_merged[y_col],
                    #    c = df_merged[col],
                        edgecolor="black",
                        color=COLOR_SCHEME[dataset],
                        label=dataset,
                        linewidth=0.25, 
                        alpha=0.2 if "gnm" in dataset else 1,
                        s=10
                        )

    # plt.xscale("log")
    plt.xlabel(x_col) # Percentage of\nLR connections (> 50 % length)")
    plt.ylabel(col_for_title) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

    plt.tight_layout()
    plt.show()


# In[63]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))

pairs_to_plot = [
    ("computational_capacity_notebook_memory_capacity_total_notebook", "computational_capacity_notebook_nonlinear_capacity_total_notebook"),
    ("mc_lin_cut40_mc_mean", "mc_nonlin_cut40_mc_mean"),
    ("mc_input_scaling_0_1_mc_mean", "mc_nonlinear_input_scaling_0_1_mc_mean"), 
    ("computational_capacity_memory_capacity_total", "computational_capacity_nonlinear_capacity_total"),
    ("computational_capacity_memory_capacity_total", "computational_capacity_cubic_capacity_total"), 
    ("mc_lin_cut40_mc_mean", "mc_nonlin_cut40_mc_mean"), 
    ("computational_capacity_notebook_damicelli_memory_capacity_profile_notebook", "computational_capacity_notebook_damicelli_nonlinear_capacity_profile_notebook")
]


for x_col, y_col in pairs_to_plot:

    if len(col) > 5:
        col_for_title = re.split(r'[,,_]+', y_col)
        len_col = len(col_for_title)
        col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

    for dataset in datasets_to_plot: 
        
        df_merged = dict_with_all_datasets_precise_categories[dataset]
        # print(df_merged.keys())

        if y_col in df_merged.columns:
            plt.scatter(df_merged[x_col], # df_merged["eta"], df_merged['gamma'],
                        df_merged[y_col],
                    #    c = df_merged[col],
                        edgecolor="black",
                        color=COLOR_SCHEME[dataset],
                        label=dataset,
                        linewidth=0.25, 
                        alpha=0.2 if "gnm" in dataset else 1,
                        s=10
                        )

    # plt.xscale("log")
    plt.xlabel(x_col) # Percentage of\nLR connections (> 50 % length)")
    plt.ylabel(col_for_title) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

    plt.tight_layout()
    plt.show()


# # Save eta and gamma

# In[64]:


# columns_to_remove = ["distance_relationship_type", "generative_rule", "num_iterations", "preferential_relationship_type", 
#                          "id", "network_idx", "network_index", 
#                          "mc_values_for_indiv_lags"]

dataset_name = "hcp_schaefer_100_dataset"

experiment_name_gnm = "11_mst_2500_animal_0"
base_path_gnm = Path(f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/{experiment_name_gnm}")
precise_gnm_files = {
                    "fundamental": f"all_metrics_for_{experiment_name_gnm}.csv", 
                    # "static": f"all_static_metrics_for_{experiment_name_gnm}_updated.csv", 
                    # "dynamic": f"all_dynamic_metrics_for_{experiment_name_gnm}_updated.csv", 
                    # "computational": f"all_computational_metrics_for_{experiment_name_gnm}_updated.csv", 
                    # "further": f"all_further_metrics_for_{experiment_name_gnm}_updated.csv", 
                    }

# Combine them all to one big dataframe. The indeces correspond to same entries across the different metric types, so we can easily merge them. Check first if the dfs are equally long. 
dfs_gnm = {key: pd.read_csv(base_path_gnm / file) for key, file in precise_gnm_files.items()}
lengths_gnm = {key: len(df) for key, df in dfs_gnm.items()}
print(lengths_gnm)
# They are all equally long, so we can merge them together (eta, gamma, id) 
df_gnm = dfs_gnm["fundamental"]  # Start with the fundamental df as the base

df_gnm_eta_and_gamma = df_gnm[["eta", "gamma", "id"]]

# save df_gnm_eta_and_gamma as pkl
with open(output_folder / f"{dataset_name}_gnm_eta_and_gamma.pkl", "wb") as f:
    pickle.dump(df_gnm_eta_and_gamma, f)
print(output_folder / f"{dataset_name}_gnm_eta_and_gamma.pkl")

