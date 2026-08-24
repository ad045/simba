import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Repo root, so this file works from any clone. The `data/` and `output/`
# symlinks at the root point at the run tree that used to be hardcoded here.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])

def quality_check_regarding_means_and_avgs(path_to_csv_file, bin_into_100=True):
    path_to_csv_file = Path(path_to_csv_file)
    df = pd.read_csv(path_to_csv_file)

    # Define the columns to analyze (excluding eta, gamma, and other metadata)
    metric_columns = [
        'MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)',
        'avg_communicability',
        'global_efficiency',
        'modularity',
        'avg_clustering',
        'transitivity',
        'avg_edge_distance',
        'wiring_cost',
        'char_path_length',
        'richclub_n_edges',
        'richclub_avg_length', 
        
        # "avg_degree",
        # "degree_assortativity",
        # "shortest_path_distance_mean",
        # "shortest_path_distance_std",
        # "structural_complexity",
        # "n_connected_components",
        # "topological_distance_mean",
        # "topological_distance_std",
        # "degree_gini",
        # "spectral_radius",
        # "spectral_gap",
        # "spectral_gap_fatemeh",
        # "diffusion_efficiency",
        # "propagation_efficiency",
        # "nct_control_avg",
        # "nct_control_std",
        # "nct_control_n_nodes_90_percent",
        # "nct_control_n_nodes_50_percent",
        # "nct_energies_energy_total",
        # "nct_energies_std_node_energy",
        # "nct_energies_n_nodes_90_percent",
        # "nct_energies_n_nodes_50_percent",
        # "metastability_global",
        # "metastability_local_mean",
        # "metastability_local_std",
        # "metastability_local_kurtosis",
        # "kernel_rank",
        # "kernel_rank_fatemeh",
        # "effective_dimensionality",
        # "multifunctionality"
    ]

    # Group by eta and gamma, then calculate mean and std for each metric. Group into 100 bins for eta and gamma first!
    if bin_into_100: 
        df['eta'] = pd.cut(df['eta'], bins=100, labels=False)
        df['gamma'] = pd.cut(df['gamma'], bins=100, labels=False)
    # Group by eta and gamma, then calculate mean and std for each metric
    grouped = df.groupby(['eta', 'gamma'])[metric_columns].agg(['mean', 'std']).reset_index()

    # Display the grouped statistics
    print("Grouped Statistics by Eta and Gamma:")
    print(grouped)
    print(f"\nNumber of unique eta-gamma combinations: {len(grouped)}")

    # Prepare data for heatmaps
    output_folder = path_to_csv_file.parent / 'heatmaps_mean_and_std/'
    os.makedirs(output_folder, exist_ok=True)

    # plot it
    things = ["mean", "std", "std_dev_normalized_by_mean"]
    for metric in metric_columns:
        for thing in things:
            plt.figure(figsize=(10, 8))
            pivot_table = grouped.pivot(index='eta', columns='gamma', values=(metric, thing)) # , thing if thing != "std_dev_normalized_by_mean" else 'std'))
            if thing == "std_dev_normalized_by_mean":
                pivot_table = grouped.pivot(index='eta', columns='gamma', values=(metric, "std")) / grouped.pivot(index='eta', columns='gamma', values=(metric, "mean")) # , 'mean'))
            sns.heatmap(pivot_table, # annot=True, 
                        fmt=".2f", cmap="Blues") # YlGnBu")
            
            # Invert y-axis
            plt.gca().invert_yaxis()

            plt.title(f'[{metric.replace("_", " ").title()}] - {thing.replace("_", " ").title()}') #  by Eta and Gamma')
            plt.xlabel('Gamma (rounded)')
            plt.ylabel('Eta (rounded)')
            
            # Make ticks readable
            distance = 10
            plt.xticks(ticks=np.arange(len(pivot_table.columns)//distance)*distance, labels=[f"{x:.1f}" for x in pivot_table.columns[::distance]])  # Show not every label
            plt.yticks(ticks=np.arange(len(pivot_table.index)//distance)*distance, labels=[f"{y:.1f}" for y in pivot_table.index[::distance]])  # Show not every label

            plt.savefig(output_folder / f'{metric}_{thing}.pdf')

        
        
    # Similar to the style above: Plot how many samples went into each point in the heatmap
    plt.figure(figsize=(10, 8))
    count_pivot_table = df.groupby(['eta', 'gamma']).size().unstack(fill_value=0)
    sns.heatmap(count_pivot_table, # annot=True, 
                fmt="d", cmap="Greens")
    plt.title('Number of Samples by Eta and Gamma')
    plt.xlabel('Gamma (rounded)')
    plt.ylabel('Eta (rounded)')   
    
    # Change ticks to not have the bin numbers but the actual ranges
    plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=[f"{x:.1f}" for x in pivot_table.columns])
    plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=[f"{y:.1f}" for y in pivot_table.index])
    
    # Make ticks readable
    distance = 10
    plt.xticks(ticks=np.arange(len(count_pivot_table.columns)//distance)*distance, labels=[f"{x:.1f}" for x in count_pivot_table.columns[::distance]])  # Show not every label
    plt.yticks(ticks=np.arange(len(count_pivot_table.index)//distance)*distance, labels=[f"{y:.1f}" for y in count_pivot_table.index[::distance]])  # Show not every label
    plt.savefig(output_folder / f'count_samples.pdf')


# Read the CSV file
path_to_csv_file = f"{_REPO_ROOT}/output/gnm/suarez_MaMI_dataset/75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_0/all_metrics_for_75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_0.csv" 
# /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/63_fine_grid_animal_0/all_metrics_for_63_fine_grid_animal_0.csv'
quality_check_regarding_means_and_avgs(path_to_csv_file, bin_into_100=False) 



path_to_csv_file = Path(path_to_csv_file)
df = pd.read_csv(path_to_csv_file)

# Define the columns to analyze (excluding eta, gamma, and other metadata)
metric_columns = [
    'MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)',
    'avg_communicability',
    'global_efficiency',
    'modularity',
    'avg_clustering',
    'transitivity',
    'avg_edge_distance',
    'wiring_cost',
    'char_path_length',
    'richclub_n_edges',
    'richclub_avg_length'
]

# Group by eta and gamma, then calculate mean and std for each metric. Group into 100 bins for eta and gamma first!

df['eta'] = pd.cut(df['eta'], bins=100, labels=False)
df['gamma'] = pd.cut(df['gamma'], bins=100, labels=False)
# Group by eta and gamma, then calculate mean and std for each metric
grouped = df.groupby(['eta', 'gamma'])[metric_columns].agg(['mean', 'std']).reset_index()

# Display the grouped statistics
print("Grouped Statistics by Eta and Gamma:")
print(grouped)
print(f"\nNumber of unique eta-gamma combinations: {len(grouped)}")

# Prepare data for heatmaps
output_folder = path_to_csv_file.parent / 'heatmaps_mean_and_std/'
os.makedirs(output_folder, exist_ok=True)

# plot it
things = ["mean", "std", "std_dev_normalized_by_mean"]
for metric in metric_columns:
    for thing in things:
        plt.figure(figsize=(10, 8))
        # pivot_table = grouped.pivot(index='eta', columns='gamma', values=(metric, thing if thing != "std_dev_normalized_by_mean" else 'std'))        if thing == "std_dev_normalized_by_mean":
        pivot_table = grouped.pivot(index='gamma', columns='eta', values=(metric, thing if thing != "std_dev_normalized_by_mean" else 'std'))
        # pivot_table = pivot_table / grouped.pivot(index='eta', columns='gamma', values=(metric, 'mean'))
        pivot_table = pivot_table / grouped.pivot(index='gamma', columns='eta', values=(metric, 'mean'))
        sns.heatmap(pivot_table, # annot=True, 
                    fmt=".2f", cmap="Blues") # YlGnBu")
        
        # Invert y-axis
        plt.gca().invert_yaxis()

        plt.title(f'[{metric.replace("_", " ").title()}] - {thing.replace("_", " ").title()}') #  by Eta and Gamma')
        plt.ylabel('Gamma (rounded)')
        plt.xlabel('Eta (rounded)')
        
        # Change ticks to not have the bin numbers but the actual ranges
        plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=[f"{x:.1f}" for x in pivot_table.columns])
        plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=[f"{y:.1f}" for y in pivot_table.index])
        
        # Make ticks readable
        distance = 10
        plt.xticks(ticks=np.arange(len(pivot_table.columns)//distance)*distance, labels=[f"{x:.1f}" for x in pivot_table.columns[::distance]])  # Show not every label
        plt.yticks(ticks=np.arange(len(pivot_table.index)//distance)*distance, labels=[f"{y:.1f}" for y in pivot_table.index[::distance]])  # Show not every label

        plt.savefig(output_folder / f'{metric}_{thing}.pdf')
    
    
# Similar to the style above: Plot how many samples went into each point in the heatmap
plt.figure(figsize=(10, 8))
count_pivot_table = df.groupby(['eta', 'gamma']).size().unstack(fill_value=0)
sns.heatmap(count_pivot_table, # annot=True, 
            fmt="d", cmap="Greens")
plt.title('Number of Samples by Eta and Gamma')
plt.ylabel('Gamma (rounded)')
plt.xlabel('Eta (rounded)')     
# Make ticks readable
distance = 10
plt.yticks(ticks=np.arange(len(count_pivot_table.columns)//distance)*distance, labels=[f"{x:.1f}" for x in count_pivot_table.columns[::distance]])  # Show not every label
plt.xticks(ticks=np.arange(len(count_pivot_table.index)//distance)*distance, labels=[f"{y:.1f}" for y in count_pivot_table.index[::distance]])  # Show not every label
plt.savefig(output_folder / f'count_samples.pdf')

print(output_folder / f'count_samples.pdf')
