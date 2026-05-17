#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from vizman import viz
import os
import re 

import pickle

from config import COLORS, COLOR_SCHEME, LABEL_MAP, gray_cmap, bone_white, half_black, emp_dataset_and_experiment_pairs


# In[4]:


# Generate output folder
output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/trade_off_analysis")
output_folder.mkdir(exist_ok=True)

# Load data 
with open(output_folder / "all_datasets_filtered.pkl", "rb") as f:
    dict_with_all_datasets = pickle.load(f)
    
# Folder now changed for saving 
output_folder = output_folder / "spiders"
output_folder.mkdir(exist_ok=True)


# In[5]:


# Taxonomies 
info_mami = pd.read_csv("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv")
print(info_mami.columns)


# In[ ]:


# selected_properties = {
#     "Integration": ["char_path_length"], # global_efficiency"], # , "char_path_length"],
#     "Segregation": ["modularity"], # , "avg_communicability"],
#     "Wiring\neconomy": ["proportion_long_range_connections_0.5"], # "wiring_cost", "algebraic_connectivity_fiedler_value"],
#     "Robustness": ["targeted_attack_robustness_rob_targeted_auc"], # synchronizability_eigenratio_lambda_2"], # "targeted_attack_robustness_rob_ratio", "persistent_homology_ph_total_persistence"],
#     "Robustness lambda_2": ["algebraic_connectivity_fiedler_value"], 
#     "Synchronizability": ["synchronizability_eigenratio_eigenratio"], # "synchronisability", "synchronisability_normalised"],
#     "Computational\ncapacity": ["computational_capacity_total_capacity"], # "computational_capacity", "computational_capacity_normalised"],
#     "Metastability": ["kuramoto_averaged_synchronization_r_std"], # "metastability", "metastability_normalised"],
#     "Metastability Reservoir Size": ["repertoire_sweep_weighted_by_distances_size_critical"], 
#     "Metastability Reservoir Diversity": ["repertoire_sweep_weighted_by_distances_size_critical"], 
    
# }

# selected_properties = {
#     "Integration": ["global_efficiency"], # global_efficiency"], # , "char_path_length"],
#     "Segregation": ["modularity"], # , "avg_communicability"],
#     "Wiring\neconomy": ["proportion_long_range_connections_0.5"], # "wiring_cost", "algebraic_connectivity_fiedler_value"],
#     "Robustness": ["targeted_attack_robustness_rob_targeted_auc"], # synchronizability_eigenratio_lambda_2"], # "targeted_attack_robustness_rob_ratio", "persistent_homology_ph_total_persistence"],
#     "Robustness lambda_2": ["algebraic_connectivity_fiedler_value"], 
#     # "Synchronizability": ["synchronizability_eigenratio_eigenratio"], # "synchronisability", "synchronisability_normalised"],
#     "Dynamics": ["spectral_radius"], 
#     # "Computational\ncapacity": ["computational_capacity_total_capacity"], # "computational_capacity", "computational_capacity_normalised"],
#     "Memory": ["mc_mean"],
#     "Computational\ncapacity": ["mc_nonlin_mean"],
#     # "Metastability": ["kuramoto_averaged_synchronization_r_std"], # "metastability", "metastability_normalised"],
#     # "Metastability Reservoir Size": ["repertoire_sweep_weighted_by_distances_size_critical"], 
#     "Metastability Reservoir Diversity": ["repertoire_sweep_weighted_by_distances_diversity_critical"], 
    
# }
    
selected_properties = {
    "Integration": "global_efficiency",
    "Segregation": "modularity", 
    "Wiring\neconomy": "proportion_long_range_connections_0.5",
    "Robustness": "targeted_attack_robustness_rob_targeted_auc", 
    # "Synchronizability": ["synchronizability_eigenratio_eigenratio"], # "synchronisability", "synchronisability_normalised"],
    "Dynamics": "spectral_radius", 
    # "Computational\ncapacity": ["computational_capacity_total_capacity"], # "computational_capacity", "computational_capacity_normalised"],
    "Memory": "mc_mean",
    "Computational\ncapacity": "mc_nonlin_mean",
    # "Metastability": ["kuramoto_averaged_synchronization_r_std"], # "metastability", "metastability_normalised"],
    # "Metastability Reservoir Size": ["repertoire_sweep_weighted_by_distances_size_critical"], 
    "Metastability Reservoir Diversity": "repertoire_sweep_weighted_by_distances_diversity_critical", 

}
#     # SPIDERS 
# "global_efficiency"       
# "algebraic_connectivity_fiedler_value"  
# "targeted_attack_robustness_rob_targeted_auc" 
# "proportion_long_range_connections_0.3956" 
# "mc_mean"
# "mc_nonlin_mean"
# "repertoire_sweep_weighted_by_distances_diversity_critical"
# "spectral_radius"        
# "modularity"        
     


SELECTED_COLS = []
for prop, metrics in selected_properties.items():
    if metrics:
        SELECTED_COLS.extend(metrics)
    else:
        print(f"Warning: No metrics selected for property '{prop}'")
        

N_SAMPLE = 100
SEED     = 42

# ── 2. Sample & aggregate ─────────────────────────────────────────────────────
def get_stats(df, cols, n_sample, seed):
    """Return (mean_series, std_series) for the selected cols."""
    sub = df[cols].dropna()
    if len(sub) > n_sample:
        sub = sub.sample(n_sample, random_state=seed)
    return sub.mean(), sub.std(ddof=0).fillna(0)


datasets_to_look_at = ['hcp_schaefer_100_dataset_gnm', 
                       'hcp_schaefer_100_dataset', 
                       'suarez_MaMI_dataset',
                       'kaysons_generated_networks_diffusion', 
                       'kaysons_generated_networks_propagation', 
                       'kaysons_generated_networks_routing', 
                    #    'kaysons_generated_networks_topology', 
                       ]
stats = {}
for name in datasets_to_look_at: # df in dict_with_all_datasets.items():
    df = dict_with_all_datasets[name]
    stats[name] = get_stats(df, SELECTED_COLS, N_SAMPLE, SEED)



# ── 3. Normalise across ALL datasets so axes are [0, 1] ──────────────────────
all_means = pd.DataFrame({k: v[0] for k, v in stats.items()}).T   # datasets × cols
col_min   = all_means.min()
col_max   = all_means.max()
col_range = (col_max - col_min).replace(0, 1)   # avoid /0
COL_RANGE = col_range
COL_MIN = col_min


def normalise(series):
    return (series - COL_MIN) / COL_RANGE


# ── 4. Spider-plot helper ─────────────────────────────────────────────────────
def make_spider(ax, values, errors, labels, color, title, alpha_fill=0.20):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]                        # close the loop

    vals = list(values) + [values[0]]
    errs = list(errors) + [errors[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, vals, color=color, linewidth=2)
    # ax.fill(angles, vals, color=color, alpha=alpha_fill)

    # ±1 std shaded band
    upper = np.clip(np.array(vals) + np.array(errs), 0, 1)
    lower = np.clip(np.array(vals) - np.array(errs), 0, 1)
    ax.fill_between(angles, lower, upper, color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [l.replace('_', '\n') for l in labels],
        fontsize=7, color='#333333'
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=6, color='grey')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=14,
                 color=color, wrap=True)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['polar'].set_visible(False)



# ── 5. Draw ───────────────────────────────────────────────────────────────────
n_datasets = len(stats)

ncols = 3
nrows = int(np.ceil(n_datasets / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 3.6, nrows * 3.6),
    subplot_kw=dict(polar=True)
)
axes_flat = np.array(axes).flatten()

for idx, (name, (mean_raw, std_raw)) in enumerate(stats.items()):
    mean_n = normalise(mean_raw).values
    std_n  = (std_raw / COL_RANGE).values          # scale std the same way
    color  = COLOR_SCHEME[name] 

    make_spider(
        ax     = axes_flat[idx],
        values = mean_n,
        errors = std_n,
        labels = selected_properties.keys(),
        color  = color,
        title  = LABEL_MAP[name],
    )

# hide unused axes
for ax in axes_flat[n_datasets:]:
    ax.set_visible(False)

fig.suptitle('Network metric profiles', #  (normalised) - mean ± std',
             fontsize=13, fontweight='bold', y=0.9) #1.01)
plt.tight_layout(pad=4)
# plt.savefig('spider_plots.png', dpi=150, bbox_inches='tight')
plt.show()


# In[7]:


# Remove row 95 
dict_with_all_datasets['suarez_MaMI_dataset'] = dict_with_all_datasets['suarez_MaMI_dataset'][dict_with_all_datasets['suarez_MaMI_dataset'].index != 95]


# In[8]:


plt.figure(figsize=viz.cm_to_inch((6,6)))
plt.hist(info_mami["order"]) 
plt.xticks(rotation=90)
list(set(info_mami["order"]))


# In[9]:


# ── 0. Configuration ──────────────────────────────────────────────────────────
what_to_look_at   = "order"
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

# None to include ALL groups # list(set(info_mami["order"])) #

# ── 1. Merge taxonomy with metric data ────────────────────────────────────────
mami_metrics = dict_with_all_datasets['suarez_MaMI_dataset'][SELECTED_COLS].reset_index(drop=True)
info_mami    = info_mami.reset_index(drop=True)

assert len(mami_metrics) == len(info_mami), \
    f"Row mismatch: metrics={len(mami_metrics)}, taxonomy={len(info_mami)}"

mami_full = pd.concat(
    [info_mami[['animal', 'common_name', what_to_look_at]], mami_metrics], axis=1
)

# ── 1b. Optional group filter ─────────────────────────────────────────────────
if GROUPS_TO_INCLUDE is not None:
    # Case-insensitive matching so "rodentia" == "Rodentia"
    available = mami_full[what_to_look_at].unique()
    matched   = [g for g in available
                 if g.lower() in [x.lower() for x in GROUPS_TO_INCLUDE]]
    not_found = [x for x in GROUPS_TO_INCLUDE
                 if x.lower() not in [g.lower() for g in available]]

    if not_found:
        print(f"⚠ The following groups were not found in '{what_to_look_at}' "
              f"and will be ignored: {not_found}")
    print(f"✓ Keeping {len(matched)} group(s): {sorted(matched)}")

    mami_full = mami_full[mami_full[what_to_look_at].isin(matched)].reset_index(drop=True)

# ── 2. Find minimum group size → sample that many per group ──────────────────
group_counts = mami_full.groupby(what_to_look_at).size()
print(f"\nAnimals per '{what_to_look_at}':")
print(group_counts.sort_values().to_string())
print(f"\n→ Minimum group size: {group_counts.min()} ({group_counts.idxmin()})")

N_PER_GROUP = group_counts.min()

sampled = (
    mami_full
    .groupby(what_to_look_at, group_keys=False)
    .apply(lambda g: g.sample(n=N_PER_GROUP, random_state=SEED))
    .reset_index(drop=True)
)

# ── 3. Compute per-group mean & std ───────────────────────────────────────────
group_stats = {}
for group, gdf in sampled.groupby(what_to_look_at):
    vals = gdf[SELECTED_COLS]
    group_stats[group] = (vals.mean(), vals.std(ddof=0).fillna(0))

# ── 4. Normalise across selected groups only ──────────────────────────────────
all_group_means = pd.DataFrame({k: v[0] for k, v in group_stats.items()}).T
g_min   = all_group_means.min()
g_max   = all_group_means.max()
# g_range = (g_max - g_min).replace(0, 1)

g_range = COL_RANGE
# def normalise_mami(series):
#     return (series - g_min) / COL_RANGE #  g_range

# ── 5. Plot ───────────────────────────────────────────────────────────────────
n_groups  = len(group_stats)
PALETTE   = plt.cm.Set2.colors

ncols = min(4, n_groups)
nrows = int(np.ceil(n_groups / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 3.8, nrows * 3.8),
    subplot_kw=dict(polar=True)
)
axes_flat = np.array(axes).flatten()

for idx, (group, (mean_raw, std_raw)) in enumerate(sorted(group_stats.items())):
    # mean_n = normalise_mami(mean_raw).values
    mean_n = normalise(mean_raw).values
    std_n  = (std_raw / COL_RANGE).values
    # std_n  = (std_raw / g_range).values
    color  = PALETTE[idx % len(PALETTE)]

    make_spider(
        ax     = axes_flat[idx],
        values = mean_n,
        errors = std_n,
        labels = selected_properties.keys(), # SELECTED_COLS,
        color  = color,
        title  = f"{group}\n(n={N_PER_GROUP} of {group_counts[group]})",
    )

for ax in axes_flat[n_groups:]:
    ax.set_visible(False)

group_label = "all groups" if GROUPS_TO_INCLUDE is None else ", ".join(sorted(matched))
fig.suptitle(
    f'MaMI · {what_to_look_at} — {group_label}\n'
    f'(n={N_PER_GROUP} per group, balanced · mean ± std)',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()

fname = f"spider_mami_{what_to_look_at}.png"
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved → {fname}")


# In[ ]:


# ── 0. Configuration ──────────────────────────────────────────────────────────
what_to_look_at   = "order"
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

# None to include ALL groups # list(set(info_mami["order"])) #

# ── 1. Merge taxonomy with metric data ────────────────────────────────────────
mami_metrics = dict_with_all_datasets['suarez_MaMI_dataset'][SELECTED_COLS].reset_index(drop=True)
info_mami    = info_mami.reset_index(drop=True)

assert len(mami_metrics) == len(info_mami), \
    f"Row mismatch: metrics={len(mami_metrics)}, taxonomy={len(info_mami)}"

mami_full = pd.concat(
    [info_mami[['animal', 'common_name', what_to_look_at]], mami_metrics], axis=1
)

# ── 1b. Optional group filter ─────────────────────────────────────────────────
if GROUPS_TO_INCLUDE is not None:
    # Case-insensitive matching so "rodentia" == "Rodentia"
    available = mami_full[what_to_look_at].unique()
    matched   = [g for g in available
                 if g.lower() in [x.lower() for x in GROUPS_TO_INCLUDE]]
    not_found = [x for x in GROUPS_TO_INCLUDE
                 if x.lower() not in [g.lower() for g in available]]

    if not_found:
        print(f"⚠ The following groups were not found in '{what_to_look_at}' "
              f"and will be ignored: {not_found}")
    print(f"✓ Keeping {len(matched)} group(s): {sorted(matched)}")

    mami_full = mami_full[mami_full[what_to_look_at].isin(matched)].reset_index(drop=True)

# ── 2. Find minimum group size → sample that many per group ──────────────────
group_counts = mami_full.groupby(what_to_look_at).size()
print(f"\nAnimals per '{what_to_look_at}':")
print(group_counts.sort_values().to_string())
print(f"\n→ Minimum group size: {group_counts.min()} ({group_counts.idxmin()})")

N_PER_GROUP = group_counts.min()

sampled = (
    mami_full
    .groupby(what_to_look_at, group_keys=False)
    .apply(lambda g: g.sample(n=N_PER_GROUP, random_state=SEED))
    .reset_index(drop=True)
)

# ── 3. Compute per-group mean & std ───────────────────────────────────────────
group_stats = {}
for group, gdf in sampled.groupby(what_to_look_at):
    vals = gdf[SELECTED_COLS]
    group_stats[group] = (vals.mean(), vals.std(ddof=0).fillna(0))

# ── 4. Normalise across selected groups only ──────────────────────────────────
all_group_means = pd.DataFrame({k: v[0] for k, v in group_stats.items()}).T
g_min   = all_group_means.min()
g_max   = all_group_means.max()
g_range = (g_max - g_min).replace(0, 1)

# g_range = COL_RANGE
def normalise_mami(series):
    return (series - g_min) / g_range

# ── 5. Plot ───────────────────────────────────────────────────────────────────
n_groups  = len(group_stats)
PALETTE   = plt.cm.Set2.colors

ncols = min(4, n_groups)
nrows = int(np.ceil(n_groups / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 3.8, nrows * 3.8),
    subplot_kw=dict(polar=True)
)
axes_flat = np.array(axes).flatten()

for idx, (group, (mean_raw, std_raw)) in enumerate(sorted(group_stats.items())):
    # mean_n = normalise_mami(mean_raw).values
    mean_n = normalise_mami(mean_raw).values
    # std_n  = (std_raw / COL_RANGE).values
    std_n  = (std_raw / g_range).values
    color  = PALETTE[idx % len(PALETTE)]
    ax = axes_flat[idx]
    
    # ── Individual animal contours (low alpha) ────────────────────────────
    group_df = sampled[sampled[what_to_look_at] == group][SELECTED_COLS]
    n_cols   = len(SELECTED_COLS)
    angles   = np.linspace(0, 2 * np.pi, n_cols, endpoint=False).tolist()
    angles  += angles[:1]  # close the polygon

    for _, row in group_df.iterrows():
        ind_n  = normalise_mami(row).values.tolist()
        ind_n += ind_n[:1]  # close the polygon
        ax.plot(angles, ind_n, color=color, alpha=0.2, # 12, 
                linewidth=0.8)
        # ax.fill(angles, ind_n, color=color, alpha=0.03)
        

    make_spider(
        ax     = ax,
        values = mean_n,
        errors = std_n,
        labels = selected_properties.keys(), # SELECTED_COLS,
        color  = color,
        title  = f"{group}\n(n={N_PER_GROUP} of {group_counts[group]})",
    )

for ax in axes_flat[n_groups:]:
    ax.set_visible(False)

group_label = "all groups" if GROUPS_TO_INCLUDE is None else ", ".join(sorted(matched))
fig.suptitle(
    f'MaMI NORMALIZED BY MaMI {what_to_look_at} — {group_label}\n'
    f'(n={N_PER_GROUP} per group, balanced · mean ± std)',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()

fname = f"spider_mami_{what_to_look_at}.png"
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved → {fname}")


# In[ ]:


# Sort by alphabet and print
print(sorted(set(info_mami["name"])))

info_mami["name_cleaned"] = info_mami["name"].str.replace(" ", "_").str.lower()


# In[ ]:


# Replace the following: 
info_mami["name_cleaned"] = info_mami["name_cleaned"].replace({
    "stripped_dolphin": "dolphin", # stripped_
    "macaque_black": "macaque", # _black",
    "macaque_lion_tail": "macaque",
    "macaque_p_t": "macaque",
    "marmoset": "marmoset",
    "mouse": "mouse",
    "rabbit": "rabbit",
    "rat": "rat"
})


# In[ ]:


# "Chimpanzee", "Dog", "Cat", "Dolphin", "Stripped Dolphin", "Macaque", "Macaque Black", "Macaque Lion Tail", "Macaque P T", "Marmoset", "Mouse", "Rabbit", "Rat"


# In[ ]:


# Count occurrences of each name
name_counts = info_mami["name_cleaned"].value_counts()
for i,n in name_counts.items():
    print(f"{i}: {n}")


# In[ ]:


name_counts.keys()[:9]


# In[ ]:


# ── 0. Configuration ──────────────────────────────────────────────────────────
what_to_look_at   = "name_cleaned"
GROUPS_TO_INCLUDE = name_counts.keys()[:9] # [
                #     'Primates',
                #     'Rodentia',
                #     # 'Hyracoidea',
                #     'Carnivora',
                #     # 'Perissodactyla',
                #     # 'Chiroptera',
                #     'Cetartiodactyla',
                #     # 'Eulipotyphla',
                #     # 'Scandentia',
                #     # 'Xenarthra',
                #     # 'Lagomorpha',
                #     # 'Marsupialia'
                # ]

# None to include ALL groups # list(set(info_mami["order"])) #

# ── 1. Merge taxonomy with metric data ────────────────────────────────────────
mami_metrics = dict_with_all_datasets['suarez_MaMI_dataset'][SELECTED_COLS].reset_index(drop=True)
info_mami    = info_mami.reset_index(drop=True)

assert len(mami_metrics) == len(info_mami), \
    f"Row mismatch: metrics={len(mami_metrics)}, taxonomy={len(info_mami)}"

mami_full = pd.concat(
    [info_mami[['animal', 'common_name', what_to_look_at]], mami_metrics], axis=1
)

# ── 1b. Optional group filter ─────────────────────────────────────────────────
if GROUPS_TO_INCLUDE is not None:
    # Case-insensitive matching so "rodentia" == "Rodentia"
    available = mami_full[what_to_look_at].unique()
    matched   = [g for g in available
                 if g.lower() in [x.lower() for x in GROUPS_TO_INCLUDE]]
    not_found = [x for x in GROUPS_TO_INCLUDE
                 if x.lower() not in [g.lower() for g in available]]

    if not_found:
        print(f"⚠ The following groups were not found in '{what_to_look_at}' "
              f"and will be ignored: {not_found}")
    print(f"✓ Keeping {len(matched)} group(s): {sorted(matched)}")

    mami_full = mami_full[mami_full[what_to_look_at].isin(matched)].reset_index(drop=True)

# ── 2. Find minimum group size → sample that many per group ──────────────────
group_counts = mami_full.groupby(what_to_look_at).size()
print(f"\nAnimals per '{what_to_look_at}':")
print(group_counts.sort_values().to_string())
print(f"\n→ Minimum group size: {group_counts.min()} ({group_counts.idxmin()})")

N_PER_GROUP = group_counts.min()

sampled = (
    mami_full
    .groupby(what_to_look_at, group_keys=False)
    .apply(lambda g: g.sample(n=N_PER_GROUP, random_state=SEED))
    .reset_index(drop=True)
)

# ── 3. Compute per-group mean & std ───────────────────────────────────────────
group_stats = {}
for group, gdf in sampled.groupby(what_to_look_at):
    vals = gdf[SELECTED_COLS]
    group_stats[group] = (vals.mean(), vals.std(ddof=0).fillna(0))

# ── 4. Normalise across selected groups only ──────────────────────────────────
all_group_means = pd.DataFrame({k: v[0] for k, v in group_stats.items()}).T
g_min   = all_group_means.min()
g_max   = all_group_means.max()
g_range = (g_max - g_min).replace(0, 1)

# g_range = COL_RANGE
def normalise_mami(series):
    return (series - g_min) / g_range

# ── 5. Plot ───────────────────────────────────────────────────────────────────
n_groups  = len(group_stats)
PALETTE   = plt.cm.Set2.colors

ncols = min(4, n_groups)
nrows = int(np.ceil(n_groups / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 3.8, nrows * 3.8),
    subplot_kw=dict(polar=True)
)
axes_flat = np.array(axes).flatten()

for idx, (group, (mean_raw, std_raw)) in enumerate(sorted(group_stats.items())):
    # mean_n = normalise_mami(mean_raw).values
    mean_n = normalise_mami(mean_raw).values
    # std_n  = (std_raw / COL_RANGE).values
    std_n  = (std_raw / g_range).values
    color  = PALETTE[idx % len(PALETTE)]
    ax = axes_flat[idx]
    
    # ── Individual animal contours (low alpha) ────────────────────────────
    group_df = sampled[sampled[what_to_look_at] == group][SELECTED_COLS]
    n_cols   = len(SELECTED_COLS)
    angles   = np.linspace(0, 2 * np.pi, n_cols, endpoint=False).tolist()
    angles  += angles[:1]  # close the polygon

    for _, row in group_df.iterrows():
        ind_n  = normalise_mami(row).values.tolist()
        ind_n += ind_n[:1]  # close the polygon
        ax.plot(angles, ind_n, color=color, alpha=0.2, # 12, 
                linewidth=0.8)
        # ax.fill(angles, ind_n, color=color, alpha=0.03)
        

    make_spider(
        ax     = ax,
        values = mean_n,
        errors = std_n,
        labels = selected_properties.keys(), # SELECTED_COLS,
        color  = color,
        title  = f"{group}\n(n={N_PER_GROUP} of {group_counts[group]})",
    )

for ax in axes_flat[n_groups:]:
    ax.set_visible(False)

group_label = "all groups" if GROUPS_TO_INCLUDE is None else ", ".join(sorted(matched))
fig.suptitle(
    f'MaMI NORMALIZED BY MaMI {what_to_look_at} — {group_label}\n'
    f'(n={N_PER_GROUP} per group, balanced · mean ± std)',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()

fname = f"spider_mami_{what_to_look_at}.png"
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved → {fname}")

