#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

from config import COLORS, COLOR_SCHEME, LABEL_MAP, gray_cmap, bone_white, half_black, emp_dataset_and_experiment_pairs


# In[ ]:


# Generate output folder
output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/trade_off_analysis")
output_folder.mkdir(exist_ok=True)

# Load data 
with open(output_folder / "all_datasets_filtered.pkl", "rb") as f:
    dict_with_all_datasets = pickle.load(f)
    
# Folder now changed for saving 
output_folder = output_folder / "spiders"
output_folder.mkdir(exist_ok=True)


# In[ ]:


# Taxonomies 
info_mami = pd.read_csv("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed_removed_95.csv")
print(info_mami.columns)


# In[ ]:


dict_with_all_datasets.keys()


# In[ ]:


plt.hist(info_mami["order"])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# dict_with_all_datasets['suarez_MaMI_dataset'].


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import pandas as pd

# ── 1. Select your six metrics ────────────────────────────────────────────────
# SELECTED_COLS = [
#     "global_efficiency",
#     "modularity", 
#     "" 
    
#     'char_path_length',
#     'avg_communicability',
#     'wiring_cost',
#     'algebraic_connectivity_fiedler_value',
#     'targeted_attack_robustness_rob_ratio',
#     'persistent_homology_ph_total_persistence',
# ]

selected_properties = {
    "Integration": ["global_efficiency"], # , "char_path_length"],
    "Segregation": ["modularity"], # , "avg_communicability"],
    "Wiring\neconomy": ["proportion_long_range_connections_0.5"], # "wiring_cost", "algebraic_connectivity_fiedler_value"],
    "Robustness": ["synchronizability_eigenratio_lambda_2"], # "targeted_attack_robustness_rob_ratio", "persistent_homology_ph_total_persistence"],
    "Synchronisability": ["synchronizability_eigenratio_eigenratio"], # "synchronisability", "synchronisability_normalised"],
    "Computational\ncapacity": ["computational_capacity_total_capacity"], # "computational_capacity", "computational_capacity_normalised"],
    "Metastability": ["kuramoto_averaged_synchronization_r_std"], # "metastability", "metastability_normalised"],
}
    
SELECTED_COLS = []
for prop, metrics in selected_properties.items():
    if metrics:
        SELECTED_COLS.extend(metrics)
    else:
        print(f"Warning: No metrics selected for property '{prop}'")
        
# create mapping from metric to property for later labelling
# METRIC_TO_PROPERTY = {}
# for prop, metrics in selected_properties.items():
#     for metric in metrics:
#         METRIC_TO_PROPERTY[metric] = prop

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
# PALETTE    = plt.cm.tab10.colors

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
    std_n  = (std_raw / col_range).values          # scale std the same way
    color  = COLOR_SCHEME[name] # PALETTE[idx % len(PALETTE)]

    make_spider(
        ax     = axes_flat[idx],
        values = mean_n,
        errors = std_n,
        labels = selected_properties.keys(), # SELECTED_COLS,
        color  = color,
        title  = name.replace('_', ' '),
    )

# hide unused axes
for ax in axes_flat[n_datasets:]:
    ax.set_visible(False)

fig.suptitle('Network metric profiles', #  (normalised) - mean ± std',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
# plt.savefig('spider_plots.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import pandas as pd

# ── 1. Select your six metrics ────────────────────────────────────────────────
SELECTED_COLS = [
    'char_path_length',
    'avg_communicability',
    'wiring_cost',
    'algebraic_connectivity_fiedler_value',
    'targeted_attack_robustness_rob_ratio',
    'persistent_homology_ph_total_persistence',
]

N_SAMPLE = 100
SEED     = 42

# ── 2. Sample & aggregate ─────────────────────────────────────────────────────
def get_stats(df, cols, n_sample, seed):
    """Return (mean_series, std_series) for the selected cols."""
    sub = df[cols].dropna()
    if len(sub) > n_sample:
        sub = sub.sample(n_sample, random_state=seed)
    return sub.mean(), sub.std(ddof=0).fillna(0)



stats = {}
for name, df in dict_with_all_datasets.items():
    if name in ['hcp_schaefer_100_dataset_gnm', 'kaysons_generated_networks_topology']: 
        continue  
    stats[name] = get_stats(df, SELECTED_COLS, N_SAMPLE, SEED)



# ── 3. Normalise across ALL datasets so axes are [0, 1] ──────────────────────
all_means = pd.DataFrame({k: v[0] for k, v in stats.items()}).T   # datasets × cols
col_min   = all_means.min()
col_max   = all_means.max()
col_range = (col_max - col_min).replace(0, 1)   # avoid /0


col_range = COL_RANGE
# def normalise(series):
#     return (series - col_min) / col_range


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
    ax.fill(angles, vals, color=color, alpha=alpha_fill)

    # ±1 std shaded band
    upper = np.clip(np.array(vals) + np.array(errs), 0, 1)
    lower = np.clip(np.array(vals) - np.array(errs), 0, 1)
    ax.fill_between(angles, lower, upper, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [l.replace('_', '\n') for l in labels] # ,
        # fontsize=7, color='#333333'
    )
    ax.set_ylim(0, 1)
    # ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    # ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=6, color='grey')
    ax.set_title(title) # , fontsize=9, fontweight='bold', pad=14,
                #  color=color, wrap=True)
        # ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    # ax.spines['polar'].set_visible(False)



# ── 5. Draw ───────────────────────────────────────────────────────────────────
n_datasets = len(stats)
# PALETTE    = plt.cm.tab10.colors

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
    std_n  = (std_raw / col_range).values          # scale std the same way
    color  = COLOR_SCHEME[name] # PALETTE[idx % len(PALETTE)]

    make_spider(
        ax     = axes_flat[idx],
        values = mean_n,
        errors = std_n,
        labels = SELECTED_COLS,
        color  = color,
        title  = name.replace('_', ' '),
    )

# hide unused axes
for ax in axes_flat[n_datasets:]:
    ax.set_visible(False)

fig.suptitle('Network metric profiles (normalised) — mean ± std',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
# plt.savefig('spider_plots.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


dict_with_all_datasets['suarez_MaMI_dataset']


# In[ ]:


# Remove row 95 
dict_with_all_datasets['suarez_MaMI_dataset'] = dict_with_all_datasets['suarez_MaMI_dataset'][dict_with_all_datasets['suarez_MaMI_dataset'].index != 95]


# In[ ]:


# ── 1. Merge taxonomy with metric data ────────────────────────────────────────
mami_metrics = dict_with_all_datasets['suarez_MaMI_dataset'][SELECTED_COLS].reset_index(drop=True)
info_mami    = info_mami.reset_index(drop=True)
what_to_look_at = "phylogenetic_group"


# Safety check
assert len(mami_metrics) == len(info_mami), \
    f"Row mismatch: metrics={len(mami_metrics)}, taxonomy={len(info_mami)}"

mami_full = pd.concat([info_mami[['animal', 'common_name', what_to_look_at]], 
                        mami_metrics], axis=1)

# ── 2. Find minimum group size → sample that many per group ──────────────────
group_counts = mami_full.groupby(what_to_look_at).size()
print(f"Animals per {what_to_look_at} group:")
print(group_counts.sort_values())
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

# ── 4. Normalise — using the global col_min/col_range from before,
#       OR recompute fresh from MaMI only (pick one) ─────────────────────────
# Fresh normalisation across MaMI groups only:
all_group_means = pd.DataFrame({k: v[0] for k, v in group_stats.items()}).T
g_min   = all_group_means.min()
g_max   = all_group_means.max()
g_range = (g_max - g_min).replace(0, 1)


col_range = COL_RANGE
# def normalise(series):
#     return (series - col_min) / col_range

# ── 5. Plot ───────────────────────────────────────────────────────────────────
n_groups = len(group_stats)
PALETTE  = plt.cm.Set2.colors

ncols = min(4, n_groups)
nrows = int(np.ceil(n_groups / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 3.8, nrows * 3.8),
    subplot_kw=dict(polar=True)
)
axes_flat = np.array(axes).flatten()

for idx, (group, (mean_raw, std_raw)) in enumerate(sorted(group_stats.items())):
    # mean_n  = normalise_mami(mean_raw).values
    mean_n = normalise(mean_raw).values
    std_n   = (std_raw / g_range).values
    color   = PALETTE[idx % len(PALETTE)]
    n_shown = group_counts[group]   # original count for subtitle

    make_spider(
        ax     = axes_flat[idx],
        values = mean_n,
        errors = std_n,
        labels = SELECTED_COLS,
        color  = color,
        title  = f"{group}\n(n={N_PER_GROUP} of {n_shown})",
    )

for ax in axes_flat[n_groups:]:
    ax.set_visible(False)

fig.suptitle(
    f'MaMI — metric profiles by {what_to_look_at} group\n'
    f'(n={N_PER_GROUP} per group, balanced sampling · mean ± std)',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('spider_mami_order.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# ── 1. Merge taxonomy with metric data ────────────────────────────────────────
mami_metrics = dict_with_all_datasets['suarez_MaMI_dataset'][SELECTED_COLS].reset_index(drop=True)
info_mami    = info_mami.reset_index(drop=True)
what_to_look_at = "order"



# Safety check
assert len(mami_metrics) == len(info_mami), \
    f"Row mismatch: metrics={len(mami_metrics)}, taxonomy={len(info_mami)}"

mami_full = pd.concat([info_mami[['animal', 'common_name', what_to_look_at]], 
                        mami_metrics], axis=1)

# ── 2. Find minimum group size → sample that many per group ──────────────────
group_counts = mami_full.groupby(what_to_look_at).size()
print(f"Animals per {what_to_look_at} group:")
print(group_counts.sort_values())
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

# ── 4. Normalise — using the global col_min/col_range from before,
#       OR recompute fresh from MaMI only (pick one) ─────────────────────────
# Fresh normalisation across MaMI groups only:
all_group_means = pd.DataFrame({k: v[0] for k, v in group_stats.items()}).T
g_min   = all_group_means.min()
g_max   = all_group_means.max()
g_range = (g_max - g_min).replace(0, 1)

def normalise_mami(series):
    return (series - g_min) / g_range

# ── 5. Plot ───────────────────────────────────────────────────────────────────
n_groups = len(group_stats)
PALETTE  = plt.cm.Set2.colors

ncols = min(4, n_groups)
nrows = int(np.ceil(n_groups / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 3.8, nrows * 3.8),
    subplot_kw=dict(polar=True)
)
axes_flat = np.array(axes).flatten()

for idx, (group, (mean_raw, std_raw)) in enumerate(sorted(group_stats.items())):
    mean_n  = normalise_mami(mean_raw).values
    std_n   = (std_raw / g_range).values
    color   = PALETTE[idx % len(PALETTE)]
    n_shown = group_counts[group]   # original count for subtitle

    make_spider(
        ax     = axes_flat[idx],
        values = mean_n,
        errors = std_n,
        labels = SELECTED_COLS,
        color  = color,
        title  = f"{group}\n(n={N_PER_GROUP} of {n_shown})",
    )

for ax in axes_flat[n_groups:]:
    ax.set_visible(False)

fig.suptitle(
    f'MaMI — metric profiles by {what_to_look_at} group\n'
    f'(n={N_PER_GROUP} per group, balanced sampling · mean ± std)',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('spider_mami_order.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


plt.figure(figsize=viz.cm_to_inch((6,6)))
plt.hist(info_mami["order"]) 
plt.xticks(rotation=90)
list(set(info_mami["order"]))


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
    std_n  = (std_raw / g_range).values
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
    mean_n = normalise_mami(mean_raw).values
    std_n  = (std_raw / g_range).values
    color  = PALETTE[idx % len(PALETTE)]
    ax     = axes_flat[idx]

    # ── Individual animal contours (low alpha) ────────────────────────────
    group_df = sampled[sampled[what_to_look_at] == group][SELECTED_COLS]
    n_cols   = len(SELECTED_COLS)
    angles   = np.linspace(0, 2 * np.pi, n_cols, endpoint=False).tolist()
    angles  += angles[:1]  # close the polygon

    for _, row in group_df.iterrows():
        ind_n  = normalise_mami(row).values.tolist()
        ind_n += ind_n[:1]  # close the polygon
        ax.plot(angles, ind_n, color=color, alpha=0.12, linewidth=0.8)
        ax.fill(angles, ind_n, color=color, alpha=0.03)

    # ── Group mean contour (drawn on top) ─────────────────────────────────
    make_spider(
        ax     = ax,
        values = mean_n,
        errors = std_n,
        labels = SELECTED_COLS,
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


df_gnm_g = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].groupby(["eta", "gamma"]).mean().reset_index()

# EXCLUDE COLUMNS: 
constant_columns = ["avg_degree", "n_connected_components", "density", "density_bct", "kernel_rank_phase_of_lambda_max"]
df_gnm_g = df_gnm_g.drop(columns=[col for col in constant_columns if col in df_gnm_g.columns])

cols_of_interest = [col for col in df_gnm_g.columns if col not in ["eta", "gamma"]]
len(cols_of_interest)


# In[ ]:


dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"]


# ## Now adding DeltaCon (or similar)

# In[ ]:


# Get the metric values at these minimum locations for coloring
# We'll use the same metric as shown in each subplot
# df_gnm_with_coords = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].copy()

n_cols = 8
n_rows = (len(cols_of_interest) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((n_cols * 3, n_rows * 2.5)), sharex=True, sharey=True, dpi=200)

for idx, col in enumerate(cols_of_interest):
    row = idx // n_cols
    col_idx = idx % n_cols
    ax = axes[row, col_idx]
    
    # Create the heatmap
    pivot_data = df_gnm_g.pivot(index="gamma", columns="eta", values=col)

    # Get the actual data ranges (STEP 1 TO COMBINE SCATTER AND IMSHOW)
    x_min, x_max = pivot_data.columns.min(), pivot_data.columns.max()
    y_min, y_max = pivot_data.index.min(), pivot_data.index.max()

    im = ax.imshow(pivot_data, 
                   extent=[x_min, x_max, y_min, y_max], # STEP 2 TO COMBINE SCATTER AND IMSHOW
                   aspect="auto", origin="lower", cmap=gray_cmap.reversed()) # "viridis")

    # Plot individual points
    for dataset in dict_with_all_datasets.keys():

        # Skip the gnm dataset, as it will now have two eta and gamma columns. 
        if dataset == "hcp_schaefer_100_dataset_gnm":
            continue

        df_merged = dict_with_all_datasets[dataset]
        if col in df_merged.columns:
            ax.scatter(df_merged["eta"], df_merged['gamma'],
                #    c=df_merged[col],
                       c=COLOR_SCHEME[dataset],
                       edgecolor="black",
                       label=dataset,
                       linewidth=0.25, s=5)

    # if the title is too long (define this), then split it into two lines at the last underscore
    if len(col) > 20:
        col = re.split(r'[,,_]+', col)
        len_col = len(col)
        col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
    ax.set_title(col, fontsize=6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # ax.legend(fontsize=3, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=4) 
    
    if idx >= 2: 
        break

# remove empty subplots
for idx in range(idx+1, n_rows * n_cols):
    row = idx // n_cols
    col_idx = idx % n_cols
    fig.delaxes(axes[row, col_idx])
    
plt.tight_layout(pad=0.4)

# plt.savefig(output_folder / "gnm_property_heatmaps_with_deltacon_minimums.pdf", dpi=200)


# In[ ]:


# Get the metric values at these minimum locations for coloring
# We'll use the same metric as shown in each subplot
# df_gnm_with_coords = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].copy()

# n_cols = 8
# n_rows = (len(cols_of_interest) + n_cols - 1) // n_cols
# fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((n_cols * 3, n_rows * 2.5)), sharex=True, sharey=True, dpi=200)
col = cols_of_interest[0] # energy


plt.figure(figsize=viz.cm_to_inch((8,6)))
# for idx, col in enumerate(cols_of_interest):

# Create the heatmap
pivot_data = df_gnm_g.pivot(index="gamma", columns="eta", values=col)

# Get the actual data ranges (STEP 1 TO COMBINE SCATTER AND IMSHOW)
x_min, x_max = pivot_data.columns.min(), pivot_data.columns.max()
y_min, y_max = pivot_data.index.min(), pivot_data.index.max()

plt.imshow(pivot_data, 
                extent=[x_min, x_max, y_min, y_max], # STEP 2 TO COMBINE SCATTER AND IMSHOW
                aspect="auto", origin="lower", cmap=gray_cmap.reversed()) # "viridis")

# Plot individual points
for dataset in dict_with_all_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_datasets[dataset]
    if col in df_merged.columns:
        plt.scatter(df_merged["eta"], df_merged['gamma'],
            #    c=df_merged[col],
                    c=COLOR_SCHEME[dataset],
                    edgecolor="black",
                    label=dataset,
                    linewidth=0.25, s=20)

# if the title is too long (define this), then split it into two lines at the last underscore
if len(col) > 20:
    col = re.split(r'[,,_]+', col)
    len_col = len(col)
    col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
plt.title(col) #  fontsize=6)

plt.xticks([])
plt.yticks([])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=4) 
    
plt.tight_layout(pad=0.4)

# plt.savefig(output_folder / "gnm_property_heatmaps_with_deltacon_minimums.pdf", dpi=200)


# In[ ]:


# drop row 95
dict_with_all_datasets["suarez_MaMI_dataset"] = dict_with_all_datasets["suarez_MaMI_dataset"].drop(index=95)

dict_with_all_datasets["suarez_MaMI_dataset"]


# In[ ]:





# In[ ]:


# join dict_with_all_datasets["suarez_MaMI_dataset"] and info_mami by index 
df_merged_mami = pd.merge(dict_with_all_datasets["suarez_MaMI_dataset"], info_mami, left_index=True, right_index=True)
df_merged_mami


# In[ ]:


# df_merged_mami["order"] to color the points in the scatter plot. But: Order entries are strings rn...
df_merged_mami["order_color"] = df_merged_mami["order"].astype("category").cat.codes
df_merged_mami["phylogenetic_group_color"] = df_merged_mami["phylogenetic_group"].astype("category").cat.codes
df_merged_mami[["order_color", "phylogenetic_group_color"]]


# In[ ]:


# Get the metric values at these minimum locations for coloring
# We'll use the same metric as shown in each subplot
# df_gnm_with_coords = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].copy()

n_cols = 8
n_rows = (len(cols_of_interest) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((n_cols * 3, n_rows * 2.5)), sharex=True, sharey=True, dpi=200)

for idx, col in enumerate(cols_of_interest):
    row = idx // n_cols
    col_idx = idx % n_cols
    ax = axes[row, col_idx]
    
    # Create the heatmap
    pivot_data = df_gnm_g.pivot(index="gamma", columns="eta", values=col)

    # Get the actual data ranges (STEP 1 TO COMBINE SCATTER AND IMSHOW)
    x_min, x_max = pivot_data.columns.min(), pivot_data.columns.max()
    y_min, y_max = pivot_data.index.min(), pivot_data.index.max()

    im = ax.imshow(pivot_data, 
                   extent=[x_min, x_max, y_min, y_max], # STEP 2 TO COMBINE SCATTER AND IMSHOW
                   aspect="auto", origin="lower", cmap="viridis")
    
    # Plot individual points
    for dataset in dict_with_all_datasets.keys():

        # Skip the gnm dataset, as it will now have two eta and gamma columns. 
        if dataset == "hcp_schaefer_100_dataset_gnm":
            continue

        # df_merged = dict_with_all_datasets[dataset]
        # if col in df_merged.columns:
        #     ax.scatter(df_merged["eta"], df_merged['gamma'],
        #                c=df_merged[col],
        #                edgecolor="black",
        #                linewidth=0.25, s=5)
        
        ax.scatter(df_merged_mami["eta"], df_merged_mami['gamma'],
            c=df_merged_mami["order_color"],
            # label=df_merged_mami["order"],
            edgecolor="black",
            linewidth=0.25, s=5, cmap="RdYlGn_r")

    # if the title is too long (define this), then split it into two lines at the last underscore
    if len(col) > 20:
        col = re.split(r'[,,_]+', col)
        len_col = len(col)
        col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
    ax.set_title(col, fontsize=6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Make legend with "order_color" colors and "order" labels
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(list(set(labels)), list(set(handles))))
    # ax.legend(by_label.values(), by_label.keys(), title="Order", fontsize=2, title_fontsize=2, loc="upper right", markerscale=2) # , bbox_to_anchor=(1.2, 1))
    
        
    # Here we create a legend: # TODO: FIX???
    # we'll plot empty lists with the desired size and label
    for color, name in zip(list(set(df_merged_mami["order_color"])), list(set(df_merged_mami["order"]))):
        ax.scatter([], [], # c=color, # alpha=0.3 # , s=unique_scatters[0],
                    label=str(name))
    ax.legend(fontsize=2, bbox_to_anchor=(1.05, 1), loc='upper left') # , frameon=False, labelspacing=1, title='City Area') phylogenetic_group

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=4) 
    
    break
    
    # if idx >= 2: 
    #     break
    


# remove empty subplots
for idx in range(idx+1, n_rows * n_cols):
    row = idx // n_cols
    col_idx = idx % n_cols
    fig.delaxes(axes[row, col_idx])
    
# title="Order") # , fontsize=6, title_fontsize=8, loc="upper right", markerscale=2) # , bbox_to_anchor=(1.2, 1))

plt.tight_layout(pad=0.4)

# plt.savefig(output_folder / "gnm_property_heatmaps_with_deltacon_minimums.pdf", dpi=200)


# In[ ]:


# Get the metric values at these minimum locations for coloring
# We'll use the same metric as shown in each subplot
# df_gnm_with_coords = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].copy()

n_cols = 2
n_rows = (len(cols_of_interest) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((n_cols * 5, n_rows * 4)), sharex=True, sharey=True, dpi=200)

for idx, col in enumerate(cols_of_interest):
    row = idx // n_cols
    col_idx = idx % n_cols
    ax = axes[row, col_idx]
    
    # Create the heatmap
    pivot_data = df_gnm_g.pivot(index="gamma", columns="eta", values=col)

    # Get the actual data ranges (STEP 1 TO COMBINE SCATTER AND IMSHOW)
    x_min, x_max = pivot_data.columns.min(), pivot_data.columns.max()
    y_min, y_max = pivot_data.index.min(), pivot_data.index.max()

    im = ax.imshow(pivot_data, 
                   extent=[x_min, x_max, y_min, y_max], # STEP 2 TO COMBINE SCATTER AND IMSHOW
                   aspect="auto", origin="lower", cmap=gray_cmap) # "viridis")
    
    # Plot individual points
    for dataset in dict_with_all_datasets.keys():

        # Skip the gnm dataset, as it will now have two eta and gamma columns. 
        if dataset == "hcp_schaefer_100_dataset_gnm":
            continue

        # df_merged = dict_with_all_datasets[dataset]
        # if col in df_merged.columns:
        #     ax.scatter(df_merged["eta"], df_merged['gamma'],
        #                c=df_merged[col],
        #                edgecolor="black",
        #                linewidth=0.25, s=5)
        
        ax.scatter(df_merged_mami["eta"], df_merged_mami['gamma'],
            c=df_merged_mami["order_color"],
            # label=df_merged_mami["order"],
            edgecolor="black",
            linewidth=0.25, s=5, cmap="RdYlGn_r")

    # if the title is too long (define this), then split it into two lines at the last underscore
    if len(col) > 20:
        col = re.split(r'[,,_]+', col)
        len_col = len(col)
        col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
    ax.set_title(col, fontsize=6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Make legend with "order_color" colors and "order" labels
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(list(set(labels)), list(set(handles))))
    # ax.legend(by_label.values(), by_label.keys(), title="Order", fontsize=2, title_fontsize=2, loc="upper right", markerscale=2) # , bbox_to_anchor=(1.2, 1))
    
        
    # Here we create a legend: # TODO: FIX???
    # we'll plot empty lists with the desired size and label
    for color, name in zip(list(set(df_merged_mami["order_color"])), list(set(df_merged_mami["order"]))):
        ax.scatter([], [], # c=color, # alpha=0.3 # , s=unique_scatters[0],
                    label=str(name))
    ax.legend(fontsize=3, bbox_to_anchor=(1.05, 1), loc='upper left') # , frameon=False, labelspacing=1, title='City Area') phylogenetic_group

    # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=4) 
    
    break
    
    # if idx >= 2: 
    #     break
    


# remove empty subplots
for idx in range(idx+1, n_rows * n_cols):
    row = idx // n_cols
    col_idx = idx % n_cols
    fig.delaxes(axes[row, col_idx])
    
# title="Order") # , fontsize=6, title_fontsize=8, loc="upper right", markerscale=2) # , bbox_to_anchor=(1.2, 1))

plt.tight_layout() # pad=0.4)

# plt.savefig(output_folder / "gnm_property_heatmaps_with_deltacon_minimums.pdf", dpi=200)


# In[ ]:


df_merged_mami


# In[ ]:


# Get the metric values at these minimum locations for coloring
# We'll use the same metric as shown in each subplot
# df_gnm_with_coords = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].copy()

n_cols = 2
n_rows = (len(cols_of_interest) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((n_cols * 5, n_rows * 4)), sharex=True, sharey=True, dpi=200)

for idx, col in enumerate(cols_of_interest):
    row = idx // n_cols
    col_idx = idx % n_cols
    ax = axes[row, col_idx]
    
    # Create the heatmap
    pivot_data = df_gnm_g.pivot(index="gamma", columns="eta", values=col)

    # Get the actual data ranges (STEP 1 TO COMBINE SCATTER AND IMSHOW)
    x_min, x_max = pivot_data.columns.min(), pivot_data.columns.max()
    y_min, y_max = pivot_data.index.min(), pivot_data.index.max()

    im = ax.imshow(pivot_data, 
                   extent=[x_min, x_max, y_min, y_max], # STEP 2 TO COMBINE SCATTER AND IMSHOW
                   aspect="auto", origin="lower", cmap=gray_cmap) # "viridis")
    
    # Plot individual points
    for dataset in dict_with_all_datasets.keys():

        # Skip the gnm dataset, as it will now have two eta and gamma columns. 
        if dataset == "hcp_schaefer_100_dataset_gnm":
            continue

        # df_merged = dict_with_all_datasets[dataset]
        # if col in df_merged.columns:
        #     ax.scatter(df_merged["eta"], df_merged['gamma'],
        #                c=df_merged[col],
        #                edgecolor="black",
        #                linewidth=0.25, s=5)
        
        ax.scatter(df_merged_mami["eta"], df_merged_mami['gamma'],
            c=df_merged_mami["order_color"],
            # label=df_merged_mami["order"],
            edgecolor="black",
            linewidth=0.25, s=5, cmap="RdYlGn_r")

    # if the title is too long (define this), then split it into two lines at the last underscore
    if len(col) > 20:
        col = re.split(r'[,,_]+', col)
        len_col = len(col)
        col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
    ax.set_title(col, fontsize=6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Make legend with "order_color" colors and "order" labels
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(list(set(labels)), list(set(handles))))
    # ax.legend(by_label.values(), by_label.keys(), title="Order", fontsize=2, title_fontsize=2, loc="upper right", markerscale=2) # , bbox_to_anchor=(1.2, 1))
    
        
    # Here we create a legend: # TODO: FIX???
    # we'll plot empty lists with the desired size and label
    for color, name in zip(list(set(df_merged_mami["phylogenetic_group_color"])), list(set(df_merged_mami["phylogenetic_group"]))):
        ax.scatter([], [], # c=color, # alpha=0.3 # , s=unique_scatters[0],
                    label=str(name))
    ax.legend(fontsize=3, bbox_to_anchor=(1.05, 1), loc='upper left') # , frameon=False, labelspacing=1, title='City Area') phylogenetic_group

    # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=4) 
    
    break
    
    # if idx >= 2: 
    #     break
    


# remove empty subplots
for idx in range(idx+1, n_rows * n_cols):
    row = idx // n_cols
    col_idx = idx % n_cols
    fig.delaxes(axes[row, col_idx])
    
# title="Order") # , fontsize=6, title_fontsize=8, loc="upper right", markerscale=2) # , bbox_to_anchor=(1.2, 1))

plt.tight_layout() # pad=0.4)

# plt.savefig(output_folder / "gnm_property_heatmaps_with_deltacon_minimums.pdf", dpi=200)


# In[ ]:


# Get the metric values at these minimum locations for coloring
# We'll use the same metric as shown in each subplot
# df_gnm_with_coords = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].copy()

n_cols = 2
n_rows = (len(cols_of_interest) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((n_cols * 5, n_rows * 2.5)), sharex=True, sharey=True, dpi=200)

for idx, col in enumerate(cols_of_interest):
    row = idx // n_cols
    col_idx = idx % n_cols
    ax = axes[row, col_idx]
    
    # Create the heatmap
    pivot_data = df_gnm_g.pivot(index="gamma", columns="eta", values=col)

    # Get the actual data ranges (STEP 1 TO COMBINE SCATTER AND IMSHOW)
    x_min, x_max = pivot_data.columns.min(), pivot_data.columns.max()
    y_min, y_max = pivot_data.index.min(), pivot_data.index.max()

    im = ax.imshow(pivot_data, 
                   extent=[x_min, x_max, y_min, y_max], # STEP 2 TO COMBINE SCATTER AND IMSHOW
                   aspect="auto", origin="lower", cmap="viridis")
    
    # Plot individual points
    for dataset in dict_with_all_datasets.keys():

        # Skip the gnm dataset, as it will now have two eta and gamma columns. 
        if dataset == "hcp_schaefer_100_dataset_gnm":
            continue

        # df_merged = dict_with_all_datasets[dataset]
        # if col in df_merged.columns:
        #     ax.scatter(df_merged["eta"], df_merged['gamma'],
        #                c=df_merged[col],
        #                edgecolor="black",
        #                linewidth=0.25, s=5)
        
        ax.scatter(df_merged_mami["eta"], df_merged_mami['gamma'],
            c=df_merged_mami["order_color"],
            # label=df_merged_mami["order"],
            edgecolor="black",
            linewidth=0.25, s=5, cmap="RdYlGn_r")

    # if the title is too long (define this), then split it into two lines at the last underscore
    if len(col) > 20:
        col = re.split(r'[,,_]+', col)
        len_col = len(col)
        col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
    ax.set_title(col, fontsize=6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Make legend with "order_color" colors and "order" labels
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(list(set(labels)), list(set(handles))))
    # ax.legend(by_label.values(), by_label.keys(), title="Order", fontsize=2, title_fontsize=2, loc="upper right", markerscale=2) # , bbox_to_anchor=(1.2, 1))
    
        
    # Here we create a legend: # TODO: FIX???
    # we'll plot empty lists with the desired size and label
    for color, name in zip(list(set(df_merged_mami["order_color"])), list(set(df_merged_mami["order"]))):
        ax.scatter([], [], # c=color, # alpha=0.3 # , s=unique_scatters[0],
                    label=str(name))
    ax.legend(fontsize=3, bbox_to_anchor=(1.05, 1), loc='upper left') # , frameon=False, labelspacing=1, title='City Area') phylogenetic_group

    # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=4) 
    
    break
    
    # if idx >= 2: 
    #     break
    


# remove empty subplots
for idx in range(idx+1, n_rows * n_cols):
    row = idx // n_cols
    col_idx = idx % n_cols
    fig.delaxes(axes[row, col_idx])
    
# title="Order") # , fontsize=6, title_fontsize=8, loc="upper right", markerscale=2) # , bbox_to_anchor=(1.2, 1))

plt.tight_layout() # pad=0.4)

# plt.savefig(output_folder / "gnm_property_heatmaps_with_deltacon_minimums.pdf", dpi=200)


# In[ ]:


# # # Get the metric values at these minimum locations for coloring
# # # We'll use the same metric as shown in each subplot
# # # df_gnm_with_coords = dict_with_all_datasets["hcp_schaefer_100_dataset_gnm"].copy()

# n_cols = 8
# n_rows = (len(cols_of_interest) + n_cols - 1) // n_cols
# fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((n_cols * 3, n_rows * 2.5)), dpi=200) #  sharex=True, sharey=True, 

# for idx, col in enumerate(cols_of_interest):
#     row = idx // n_cols
#     col_idx = idx % n_cols
#     ax = axes[row, col_idx]
    
# #     # Create the heatmap
# #     pivot_data = df_gnm_g.pivot(index="gamma", columns="eta", values=col)
#     ax.scatter(df_gnm_g["proportion_long_range_connections_0.5"], df_gnm_g[col], 
#                color="gray", linewidth=0, s=5, alpha=0.2) # edgecolor="black")


# #     # Plot individual points
#     for dataset in dict_with_all_merged_datasets.keys():

#         # Skip the gnm dataset, as it will now have two eta and gamma columns. 
#         if dataset == "hcp_schaefer_100_dataset_gnm":
#             continue

#         df_merged = dict_with_all_merged_datasets[dataset]

#         if col in df_merged.columns:
#             ax.scatter(df_merged["proportion_long_range_connections_0.5"], # df_merged["eta"], df_merged['gamma'],
#                        df_merged[col],
#                     #    c=df_merged[col],
#                        edgecolor="black",
#                        color=COLOR_SCHEME[dataset],
#                        linewidth=0.25, s=5)

#     # if the title is too long (define this), then split it into two lines at the last underscore
#     if len(col) > 20:
#         col = re.split(r'[,,_]+', col)
#         len_col = len(col)
#         col = "_".join(col[:len_col//2]) + "\n" + "_".join(col[len_col//2:])
#     ax.set_title(col, fontsize=6)
    
#     # ax.set_xticks([])
#     # ax.set_yticks([])
    
#     # Change the fontsize of the ax ticks to 6
#     ax.tick_params(axis='both', labelsize=6)
#     # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     # cbar.ax.tick_params(labelsize=4) 
    
# #     # if idx >= 2: 
# #     #     break

# # remove empty subplots
# for idx in range(len(cols_of_interest), n_rows * n_cols):
#     row = idx // n_cols
#     col_idx = idx % n_cols
#     fig.delaxes(axes[row, col_idx])
    
# plt.tight_layout(pad=0.4)


# In[ ]:


# save dict_with_all_merged_datasets: 
with open(output_folder / "all_datasets_merged_with_minima_locations_deltacon.pkl", "wb") as f:
    pickle.dump(dict_with_all_merged_datasets, f)


# In[ ]:


dict_with_all_merged_datasets["suarez_MaMI_dataset"]


# In[ ]:


for idx, col in enumerate(cols_of_interest):
    fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

    plt.scatter(df_gnm_g["proportion_long_range_connections_0.5"], df_gnm_g[col], 
               color="gray", linewidth=0, s=5, alpha=0.2) 

    if len(col) > 20:
        col_for_title = re.split(r'[,,_]+', col)
        len_col = len(col_for_title)
        col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

    if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
        col = "min_value"
        col_for_title = "Distance Measure"
    
    for dataset in dict_with_all_merged_datasets.keys():

        # Skip the gnm dataset, as it will now have two eta and gamma columns. 
        if dataset == "hcp_schaefer_100_dataset_gnm":
            continue

        df_merged = dict_with_all_merged_datasets[dataset]

        if col in df_merged.columns:
            plt.scatter(df_merged["proportion_long_range_connections_0.5"], # df_merged["eta"], df_merged['gamma'],
                       df_merged[col],
                    #    c = df_merged[col],
                       edgecolor="black",
                       color=COLOR_SCHEME[dataset],
                       linewidth=0.25, s=5)
    
    plt.xlabel("% LR (threshold: 50%)") # Percentage of\nLR connections (> 50 % length)")
    # plt.ylabel(col.replace("_", " ").capitalize())
    # if the title is too long (define this), then split it into two lines at the last underscore
    plt.title(col) # , fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_folder / f"fig_{idx}_{col}_scatter.png", dpi=200)
    if idx == 0:
        plt.show()
    else: 
        plt.close()

print("Save path: ", output_folder)


# In[ ]:


df_gnm_g


# In[ ]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

plt.scatter(df_gnm_g["proportion_long_range_connections_0.5"], df_gnm_g[col], 
            color="gray", linewidth=0, s=5, alpha=0.2) 

if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in dict_with_all_merged_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_merged_datasets[dataset]

    if col in df_merged.columns:
        plt.scatter(df_merged["proportion_long_range_connections_0.5"], # df_merged["eta"], df_merged['gamma'],
                    df_merged[col],
                #    c = df_merged[col],
                    edgecolor="black",
                    color=COLOR_SCHEME[dataset],
                    linewidth=0.25, s=5)

plt.xlabel("% LR (threshold: 50%)") # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.title(col) # , fontsize=6)

plt.tight_layout()


# In[ ]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

x_col = "computational_capacity_memory_capacity_total"
y_col = "computational_capacity_nonlinear_capacity_total"
plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color="gray", linewidth=0, s=5, alpha=0.2) 

if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in dict_with_all_merged_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_merged_datasets[dataset]

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


# In[ ]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

x_col = "global_efficiency" # computational_capacity_memory_capacity_total"
y_col = "modularity" # computational_capacity_nonlinear_capacity_total"
plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color="gray", linewidth=0, s=5, alpha=0.2) 

if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in dict_with_all_merged_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_merged_datasets[dataset]

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


# In[ ]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

x_col = "modularity" # computational_capacity_memory_capacity_total"
y_col = "algebraic_connectivity_fiedler_value" # computational_capacity_nonlinear_capacity_total"
plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color="gray", linewidth=0, s=5, alpha=0.2) 

if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in dict_with_all_merged_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_merged_datasets[dataset]

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


# In[ ]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

x_col = "char_path_length" # modularity" # computational_capacity_memory_capacity_total"
y_col = "avg_clustering" # algebraic_connectivity_fiedler_value" # computational_capacity_nonlinear_capacity_total"
plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color="gray", linewidth=0, s=5, alpha=0.2) 

if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in dict_with_all_merged_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_merged_datasets[dataset]

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

plt.tight_layout() # TODO: INCLUDE IN DOCS


# In[ ]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

x_col = "wiring_cost"
y_col = "computational_capacity_nonlinear_capacity_total"
# plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
#             color="gray", linewidth=0, s=5, alpha=0.2) 

# if len(col) > 20:
#     col_for_title = re.split(r'[,,_]+', y_col)
#     len_col = len(col_for_title)
#     col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

# if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
#     col = "min_value"
#     col_for_title = "Distance Measure"

for dataset in dict_with_all_merged_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_merged_datasets[dataset]

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

plt.xscale("log")
plt.xlabel(x_col) # Percentage of\nLR connections (> 50 % length)")
# plt.ylabel(col.replace("_", " ").capitalize())
# if the title is too long (define this), then split it into two lines at the last underscore
plt.ylabel(col_for_title) # , fontsize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

plt.tight_layout()


# In[ ]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

x_col = "proportion_long_range_connections_0.3" # wiring_cost"
y_col = "computational_capacity_nonlinear_capacity_total"
plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color="gray", linewidth=0, s=5, alpha=0.2) 

if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in dict_with_all_merged_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_merged_datasets[dataset]

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


# In[ ]:


fig = plt.figure(figsize=viz.cm_to_inch((6,6)))    

x_col = "computational_capacity_memory_capacity_total"
y_col = "computational_capacity_nonlinear_capacity_total"
plt.scatter(df_gnm_g[x_col], df_gnm_g[y_col], 
            color="gray", linewidth=0, s=5, alpha=0.2) 

if len(col) > 20:
    col_for_title = re.split(r'[,,_]+', y_col)
    len_col = len(col_for_title)
    col_for_title = "_".join(col_for_title[:len_col//2]) + "\n" + "_".join(col_for_title[len_col//2:])

if col == "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)": 
    col = "min_value"
    col_for_title = "Distance Measure"

for dataset in dict_with_all_merged_datasets.keys():

    # Skip the gnm dataset, as it will now have two eta and gamma columns. 
    if dataset == "hcp_schaefer_100_dataset_gnm":
        continue

    df_merged = dict_with_all_merged_datasets[dataset]

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


# # OLD
# 

# In[ ]:


plt.figure(figsize=viz.cm_to_inch((12,12)))

for dataset_name in dict_with_all_datasets:
    print(dataset_name)
    df = dict_with_all_datasets[dataset_name]
    print(df["wiring_cost"].shape)
    if dataset_name == "hcp_schaefer_100_dataset_gnm": 
        plt.scatter(df["wiring_cost"], df["computational_capacity_nonlinear_capacity_total"], c=df["MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)"], cmap=gray_cmap, s=5, alpha=0.2, linewidths=0, label=dataset_name)
    else: 
        print(COLOR_SCHEME[dataset_name])
        plt.scatter(df["wiring_cost"], df["computational_capacity_nonlinear_capacity_total"], s=10, label=dataset_name, marker="o", color=COLOR_SCHEME[dataset_name], linewidths=0.25, edgecolors="black")


# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


# In[ ]:





# In[ ]:


plt.figure(figsize=viz.cm_to_inch((12,12)))

for dataset_name in dict_with_all_datasets:
    print(dataset_name)
    df = dict_with_all_datasets[dataset_name]
    print(df["wiring_cost"].shape)
    if dataset_name == "hcp_schaefer_100_dataset_gnm": 
        pass
        # plt.scatter(df["wiring_cost"], df["wiring_cost_dupa"], c=df["MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)"], cmap=gray_cmap, s=5, alpha=0.2, linewidths=0, label=dataset_name)
    else: 
        print(COLOR_SCHEME[dataset_name])
        plt.scatter(df["wiring_cost"], df["avg_communicability"], s=10, label=dataset_name, marker="o", color=COLOR_SCHEME[dataset_name], linewidths=0.25, edgecolors="black")


# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


# In[ ]:


# --- Single scatter: wiring_cost vs computational capacity ---
fig, ax = plt.subplots(figsize=(10, 7))
for dataset_name, df in dict_with_all_datasets.items():
    if "wiring_cost" in df.columns and "computational_capacity_nonlinear_capacity_total" in df.columns:
        
        if dataset_name == "hcp_schaefer_100_dataset_gnm":
            ax.scatter(
                df["wiring_cost"],
                df["computational_capacity_nonlinear_capacity_total"],
                label=LABEL_MAP.get(dataset_name, dataset_name),
                color=COLOR_SCHEME.get(dataset_name, "gray"),
                # marker=MARKER_MAP.get(dataset_name, "o"),
                alpha=0.6,
                edgecolors="k",
                linewidths=0.3,
                s=30,
            )
        else:
            ax.scatter(
                df["wiring_cost"],
                df["computational_capacity_nonlinear_capacity_total"],
                label=LABEL_MAP.get(dataset_name, dataset_name),
                color=COLOR_SCHEME.get(dataset_name, "gray"),
                # marker=MARKER_MAP.get(dataset_name, "o"),
                alpha=0.6,
                edgecolors="k",
                linewidths=0.3,
                s=30,
            )
            
ax.set_xlabel("Wiring Cost", fontsize=12)
ax.set_ylabel("Computational Capacity (Nonlinear Total)", fontsize=12)
ax.set_title("Wiring Cost vs. Computational Capacity", fontsize=14)
ax.legend(fontsize=9, framealpha=0.9)
sns.despine()
plt.tight_layout()

plt.savefig(output_folder / "scatter_wiring_vs_comp_capacity.png", dpi=200)
print(output_folder / "scatter_wiring_vs_comp_capacity.png")
plt.show()


# In[ ]:


shared_numeric_cols = None
for df in dict_with_all_datasets.values():
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    if shared_numeric_cols is None:
        shared_numeric_cols = numeric_cols
    else:
        shared_numeric_cols = shared_numeric_cols & numeric_cols

for col in shared_numeric_cols: 
    print(col)

