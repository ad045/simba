#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from vizman import viz

from config import COLORS, COLOR_SCHEME, LABEL_MAP, gray_cmap, bone_white, half_black, emp_dataset_and_experiment_pairs

base_dir = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm")
distance_measure = "delta_con"
distance_name_in_csv_dict = {
    "hamming": "HammingDist", 
    "delta_con": "DeltaCon", 
    "energy": "MaxCrit"
}
distance_name_in_csv = distance_name_in_csv_dict[distance_measure]

csv_files = {
    'humans': base_dir / f"hcp_schaefer_100_dataset/05_mst_animal_0_compared_with_hcp_schaefer_100/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_hcp_schaefer_100.csv",
    'diffusion': base_dir / f"kaysons_generated_networks_diffusion/05_mst_animal_0_compared_with_diffusion/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_diffusion.csv",
    'propagation': base_dir / f"kaysons_generated_networks_propagation/05_mst_animal_0_compared_with_propagation/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_propagation.csv",
    'routing': base_dir / f"kaysons_generated_networks_routing/05_mst_animal_0_compared_with_routing/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_routing.csv",
    'mami': base_dir / f"suarez_MaMI_dataset/05_mst_animal_0_compared_with_mami/summary_indiv_{distance_measure}_for_exp_05_mst_animal_0_compared_with_mami.csv"
}


# In[2]:


# Taxonomies 
info_mami = pd.read_csv("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv")
taxonomy = "order" # phylogenetic_group" # "order" 

GROUPS_TO_INCLUDE = [
                    'Primates',
                    'Rodentia',
                    # 'Hyracoidea',
                    'Carnivora',
                    # 'Perissodactyla',
                    # 'Chiroptera',
                    'Cetartiodactyla',
                    # 'Eulipotyphla',
                    # 'Scandentia',
                    # 'Xenarthra',
                    # 'Lagomorpha',
                    # 'Marsupialia'
                ]
# info_mami = info_mami[info_mami[taxonomy].isin(GROUPS_TO_INCLUDE)]

# info_mami["tax_color"] = info_mami[taxonomy].astype("category").cat.codes
info_mami


# In[3]:


info_mami = info_mami[info_mami[taxonomy].isin(GROUPS_TO_INCLUDE)]


# In[4]:


df_mami = pd.read_csv(csv_files["mami"])
# drop column "DeltaCon_subject_95"
df_mami = df_mami.drop(columns=["DeltaCon_subject_95"])
df_mami


# In[5]:


df_mami = df_mami.drop(columns=["network_index", "filename"]).groupby(["eta", "gamma"]).mean().reset_index()


# In[6]:


# add a column to df_mami that has the mean delta_con value in it
all_columns = [i for i in df_mami.columns if "DeltaCon_subject_" in i]
df_mami["mean_delta_con"] = df_mami[all_columns].mean(axis=1)


# In[7]:


minimum_df = [] 

for col in df_mami.columns:
    # if col in ["network_index", "id", "filename", "eta", "gamma", 
    #            "Unnamed: 0", "order", "dataset"]:
    #     continue
    
    if "DeltaCon_subject_" in col:

        # Get minimum value of that column
        min_value = df_mami[col].min()
        # Save minimum value and corresponding column name
        minimum_df.append({"column": col, 
                        "min_value_individual_networks": min_value, 
                        "eta_individual_networks": df_mami.loc[df_mami[col] == min_value, "eta"].values[0],
                        "gamma_individual_networks": df_mami.loc[df_mami[col] == min_value, "gamma"].values[0],
                        # "order": df_mami.loc[df_mami[col] == min_value, "order"].values[0],
                        # "dataset": "mami"
                        })
minimum_df = pd.DataFrame(minimum_df)


# MEAN DeltaCon per network 

# # Get minimum value of that column
# min_value = df_mami["mean_delta_con"].min()
# minimum_df_mean = []
# # Save minimum value and corresponding column name
# minimum_df_mean.append({"column": col, 
#                 "min_value_mean_networks": min_value, 
#                 "eta_mean_networks": df_mami.loc[df_mami["mean_delta_con"] == min_value, "eta"].values[0],
#                 "gamma_mean_networks": df_mami.loc[df_mami["mean_delta_con"] == min_value, "gamma"].values[0],
#                 # "order": df_mami.loc[df_mami[col] == min_value, "order"].values[0],
#                 # "dataset": "mami"
#                 })
# # df_mami["mean_delta_con"] = df_mami[all_columns].mean(axis=1)

# minimum_df_mean = pd.DataFrame(minimum_df_mean)

# minimum_df = pd.concat([minimum_df, minimum_df_mean, info_mami], axis=1)
minimum_df = pd.concat([minimum_df, info_mami], axis=1)
minimum_df


# In[8]:


minimum_df


# In[ ]:


minimum_df = minimum_df[minimum_df[taxonomy].isin(GROUPS_TO_INCLUDE)]
print(minimum_df[taxonomy].unique())

names = minimum_df[taxonomy].unique() # ["tax_color"]
color_map = plt.cm.get_cmap("RdYlGn_r", len(names))
color_dict = {name: color_map(i) for i, name in enumerate(names)}


# In[26]:


plt.figure(figsize=viz.cm_to_inch((10,6)), dpi=200)

# Create the heatmap
pivot_data = df_mami.pivot(index="gamma", columns="eta", values="mean_delta_con")

# Get the actual data ranges (STEP 1 TO COMBINE SCATTER AND IMSHOW)
x_min, x_max = pivot_data.columns.min(), pivot_data.columns.max()
y_min, y_max = pivot_data.index.min(), pivot_data.index.max()

plt.imshow(pivot_data, 
                extent=[x_min, x_max, y_min, y_max], # STEP 2 TO COMBINE SCATTER AND IMSHOW
                aspect="auto", origin="lower", cmap=gray_cmap) # "viridis")

# Dont plot nan rows
# minimum_df = minimum_df[minimum_df["eta_individual_networks"].notna() & minimum_df["gamma_individual_networks"].notna()]

# Add slight randomness jitter in both x and y (normal dist)
jitter_strength = 0.005
plt.scatter(minimum_df["eta_individual_networks"] + np.random.normal(0, 
                                                                     jitter_strength * (x_max - x_min), 
                                                                     size=len(minimum_df)
                                                                     ),
            minimum_df["gamma_individual_networks"] + np.random.normal(0, 
                                                                       jitter_strength * (y_max - y_min), 
                                                                       size=len(minimum_df)
                                                                       ),
            color=[color_dict[i] for i in minimum_df[taxonomy]], # minimum_df["tax_color"],
            # label=df_merged_mami["order"],
            edgecolor="black",
            linewidth=0.25, 
            s=20, # 20, 
            alpha=0.5)

# if the title is too long (define this), then split it into two lines at the last underscore
# if len(col) > 20:
#         col = re.split(r'[,,_]+', col)
#         len_col = len(col)
#         col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
#     ax.set_title(col, fontsize=6)
    
#     ax.set_xticks([])
#     ax.set_yticks([])

    # Here we create a legend: # TODO: FIX???
    # we'll plot empty lists with the desired size and label
for name in color_dict:
    plt.scatter([], [], # c=color, # alpha=0.3 # , s=unique_scatters[0],
                c=color_dict[name],
                # cmap="RdYlGn_r",
                label=str(name))
    
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # , fontsize=3, frameon=False, labelspacing=1, title='City Area') phylogenetic_group
plt.title(f"DeltaCon, {taxonomy.capitalize()}")
plt.tight_layout()


# In[31]:


# Turn this into four subplots: 

fig, axs = plt.subplots(2, 2, figsize=viz.cm_to_inch((10,6)), dpi=200)
axs = axs.flatten()
# Create the heatmap
pivot_data = df_mami.pivot(index="gamma", columns="eta", values="mean_delta_con")

# Get the actual data ranges (STEP 1 TO COMBINE SCATTER AND IMSHOW)
x_min, x_max = pivot_data.columns.min(), pivot_data.columns.max()
y_min, y_max = pivot_data.index.min(), pivot_data.index.max()

for i, (ax, name) in enumerate(zip(axs, names)):
    ax.imshow(pivot_data, 
                extent=[x_min, x_max, y_min, y_max], # STEP 2 TO COMBINE SCATTER AND IMSHOW
                aspect="auto", origin="lower", cmap=gray_cmap) # "viridis")

    # Add slight randomness jitter in both x and y (normal dist)
    jitter_strength = 0.005
    ax.scatter(minimum_df[minimum_df[taxonomy] == name]["eta_individual_networks"] + np.random.normal(0, 
                                                                        jitter_strength * (x_max - x_min), 
                                                                        size=len(minimum_df[minimum_df[taxonomy] == name])
                                                                        ),
                minimum_df[minimum_df[taxonomy] == name]["gamma_individual_networks"] + np.random.normal(0, 
                                                                        jitter_strength * (y_max - y_min), 
                                                                        size=len(minimum_df[minimum_df[taxonomy] == name])
                                                                        ),
                color=[color_dict[i] for i in minimum_df[minimum_df[taxonomy] == name][taxonomy]], # minimum_df["tax_color"],
                # label=df_merged_mami["order"],
                edgecolor="black",
                linewidth=0.25, 
                s=20, # 20, 
                alpha=0.5)

    ax.set_title(name)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)   
# if the title is too long (define this), then split it into two lines at the last underscore
# if len(col) > 20:
#         col = re.split(r'[,,_]+', col)
#         len_col = len(col)
#         col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
#     ax.set_title(col, fontsize=6)
    
#     ax.set_xticks([])
#     ax.set_yticks([])

    # Here we create a legend: # TODO: FIX???
    # we'll plot empty lists with the desired size and label
for name in color_dict:
    # plt.scatter([], [], # c=color, # alpha=0.3 # , s=unique_scatters[0],
    #             c=color_dict[name],
    #             # cmap="RdYlGn_r",
    #             label=str(name))
    
    axs[1].scatter([], [], c=color_dict[name], label=str(name))


axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left') # , fontsize=3, frameon=False, labelspacing=1, title='City Area') phylogenetic_group
# plt.title(f"DeltaCon, {taxonomy.capitalize()}")
plt.tight_layout()


# In[ ]:


color_dict


# In[ ]:


# Store all minima for each network type
all_minima = {}

# Do this in subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=viz.cm_to_inch((18,12)), sharex=True, sharey=True, dpi=100)
axs = axs.flatten()

# Process each CSV file
for i, (network_name, csv_path) in enumerate(csv_files.items()):
    print(f"Processing {network_name}...")
    
    # Load the data
    df = pd.read_csv(csv_path)
    
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
    minima_for_that_point = []
    # dataset_of_each_minimum = []
    
    for col in distance_measure_columns:
        # Find the row with minimum DeltaCon for this subject
        min_idx = df_mean[col].idxmin()
        minima_eta.append(df_mean.loc[min_idx, "eta"])
        minima_gamma.append(df_mean.loc[min_idx, "gamma"])
        minima_for_that_point.append(df_mean.loc[min_idx, col])
        # dataset_of_each_minimum.append(
        # Add the number of minimum values to an array 
        # number_minima.append(df_mean.loc[min_idx, "gamma"].to_numpy())
        # print(np.array(df_mean.loc[min_idx, "gamma"]).shape)

    # if mami, drop index 95 
    if network_name == "mami":
        minima_eta.pop(95)
        minima_gamma.pop(95)
        # dataset_of_each_minimum.pop(95)
        # number_minima.pop(95)
        
    all_minima[network_name] = {
        'eta': minima_eta,
        'gamma': minima_gamma,
        'minima_for_that_point': minima_for_that_point,
        'df_mean': df_mean, 
        # 'number_minima': number_minima, 
    }
    print(f"  Found {len(minima_eta)} minima")
    
    cmap = "Grays"

    # CHANGE SCALING HERE: EITHER FROM 300 to 2300, or for every plot individually.
    background = axs[i].imshow(df_mean.pivot(index="gamma", columns="eta", values=f"{distance_name_in_csv}_mean"),
                               extent=(df_mean["eta"].min(), df_mean["eta"].max(), df_mean["gamma"].min(), df_mean["gamma"].max()),
                               origin='lower', aspect='auto', cmap=cmap)  # , vmin=300, vmax=2300) # , alpha=0.5)


    axs[i].set_title(network_name.capitalize())
    axs[i].set_xlabel("Eta")
    axs[i].set_ylabel("Gamma") 
    plt.colorbar(background, ax=axs[i], label=f"Mean {distance_name_in_csv.capitalize()}")

    # Add minima points
    axs[i].scatter(minima_eta, minima_gamma, color='red', s=20, label='Minima', edgecolor='black')
    # ax_min = axs[i].scatter(minima_eta, minima_gamma, c=number_minima, s=20, label='Minima', edgecolor='black', cmap="Reds")
    # axs[i].legend()
    
    
    # Add colorbar
    # norm = plt.Normalize(df_mean[f"{distance_name_in_csv}_mean"].min(), df_mean[f"{distance_name_in_csv}_mean"].max())
    # norm = plt.Normalize(300, 2300) # df_mean[f"{distance_name_in_csv}_mean"].min(), df_mean[f"{distance_name_in_csv}_mean"].max())
    # sm = plt.cm.ScalarMappable(cmap=cmap) # , norm=norm)
    # sm.set_array([])
    # fig.colorbar(sm, ax=axs[i], label=f"Mean {distance_name_in_csv.capitalize()}")
    # plt.colorbar(ax_min, ax=axs[i], label=f"Mean {distance_name_in_csv.capitalize()}")

# Remove empty subplot
if len(csv_files) < len(axs):
    for j in range(len(csv_files), len(axs)):
        fig.delaxes(axs[j])

plt.suptitle(f"{distance_measure.capitalize()}") # .replace(" ", "")}")
plt.tight_layout()


# In[ ]:


for i in all_minima["mami"]: 
    print(len(all_minima["mami"][i]))


# In[ ]:


df_mami = pd.DataFrame(all_minima["mami"])
df_mami


# In[ ]:


# Taxonomies 
info_mami = pd.read_csv("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv")
info_mami


# In[ ]:


plt.scatter(all_minima["mami"]["eta"], all_minima["mami"]["gamma"])


# In[ ]:


df_mean


# In[ ]:


df_mean.loc[min_idx]


# In[ ]:


# Create the combined plot
plt.figure(figsize=(6,6), dpi=100)

# Define colors for each network type
colors = {
    'diffusion': 'red',
    'mami': 'blue',
    'routing': 'green', 
    'propagation': 'orange',
    'humans': 'purple'
}

cmap = "Grays"

# Plot the landscape from one network (they should be similar)
# Using diffusion as the base
base_network = 'diffusion'
df_mean = all_minima[base_network]['df_mean']
if f"{distance_name_in_csv}_mean" not in df_mean.columns:
    # Calculate mean across all subjects
    distance_measure_columns = [col for col in df_mean.columns if f"{distance_name_in_csv}_subject_" in col]
    df_mean[f"{distance_name_in_csv}_mean"] = df_mean[distance_measure_columns].mean(axis=1)

plt.scatter(df_mean["eta"], df_mean["gamma"], c=df_mean[f"{distance_name_in_csv}_mean"], 
            cmap=cmap) # , # alpha=0.3, s=30, label='Landscape (mean DeltaCon)')

# Add colorbar for the landscape
plt.colorbar(label=f'Mean {distance_name_in_csv.capitalize()}')

# Plot minima for each network type
for network_name, color in colors.items():
    minima = all_minima[network_name]
    plt.scatter(minima['eta'], minima['gamma'], 
                c=color, marker='o', s=50, alpha=0.6, 
                label=f'{network_name.capitalize()} minima (n={len(minima["eta"])})',
                edgecolors='black', linewidths=0.5)

plt.xlabel('Eta')
plt.ylabel('Gamma') 
plt.title('Combined Minima Across Network Types') 
plt.legend(loc='best')
plt.tight_layout()

# Save the figure
output_path = "/home/claude/combined_minima_plot.png"
# plt.savefig(output_path, dpi=150, bbox_inches='tight')
# print(f"\nPlot saved to: {output_path}")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for network_name in colors.keys():
    minima = all_minima[network_name]
    print(f"\n{network_name.upper()}:")
    print(f"  Eta range: [{np.min(minima['eta']):.3f}, {np.max(minima['eta']):.3f}]")
    print(f"  Gamma range: [{np.min(minima['gamma']):.3f}, {np.max(minima['gamma']):.3f}]")
    print(f"  Eta mean ± std: {np.mean(minima['eta']):.3f} ± {np.std(minima['eta']):.3f}")
    print(f"  Gamma mean ± std: {np.mean(minima['gamma']):.3f} ± {np.std(minima['gamma']):.3f}")


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Collect the 100 lowest DeltaCon values for each network type
boxplot_data = []
labels = []

colors_list = ['red', 'blue', 'green', 'orange', 'purple']
network_types = ['diffusion', 'mami', 'routing', 'propagation', 'humans']

for network_name in network_types:
    df_mean = all_minima[network_name]['df_mean']
    
    # Ensure DeltaCon_mean exists
    if f'{distance_name_in_csv}_mean' not in df_mean.columns:
        distance_measure_columns = [col for col in df_mean.columns if f"{distance_name_in_csv}_subject_" in col]
        df_mean[f'{distance_name_in_csv}_mean'] = df_mean[distance_measure_columns].mean(axis=1)

    # Get the 100 lowest values from the ENTIRE dataframe
    lowest_100 = df_mean[f'{distance_name_in_csv}_mean'].nsmallest(100).values
    boxplot_data.append(lowest_100)
    labels.append(network_name.capitalize())

# Create the boxplot
fig, ax = plt.subplots(figsize=viz.cm_to_inch((12,12)), dpi=100)

bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True,
                showmeans=True, meanline=False,
                medianprops=dict(color='black', linewidth=2),
                meanprops=dict(marker='D', markerfacecolor='white', 
                              markeredgecolor='black', markersize=6))

# Add scatter points for the 100 lowest values
for i, data in enumerate(boxplot_data):
    x = np.random.normal(i + 1, 0.04, size=len(data))  # Jitter the x-values
    ax.scatter(x, data, color=colors_list[i], alpha=0.6, edgecolor='black', linewidth=0.5)

# Color the boxes
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel(f'{distance_name_in_csv.capitalize()} Mean')
ax.set_xlabel('Network Type')
ax.set_title(f'100 Lowest {distance_name_in_csv.capitalize()} Values per Network Type')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Print summary statistics for the 100 lowest values
print("\n" + "="*60)
print("SUMMARY STATISTICS (100 LOWEST VALUES)")
print("="*60)
for i, network_name in enumerate(network_types):
    data = boxplot_data[i]
    print(f"\n{network_name.upper()}:")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Q1 (25th percentile): {np.percentile(data, 25):.6f}")
    print(f"  Median: {np.median(data):.6f}")
    print(f"  Q3 (75th percentile): {np.percentile(data, 75):.6f}")
    print(f"  Max: {np.max(data):.6f}")
    print(f"  Mean ± std: {np.mean(data):.6f} ± {np.std(data):.6f}")

