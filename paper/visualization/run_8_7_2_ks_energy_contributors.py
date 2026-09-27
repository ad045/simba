"""
8_7_2_ks_energy_contributors

Extracted from 8_7_2_ks_energy_contributors.ipynb by extract_notebook.py - do not edit by hand.
The run comes from experiments_config (MORPHO_EXP); paths that pointed into the
connectome_distances repo now use this repo's own data.
"""
import matplotlib
matplotlib.use("Agg")

# ====================================================================
# cell 1
# ====================================================================
# Imports 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
import seaborn as sns

import matplotlib as mpl


import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from vizman import viz
viz.set_visual_style()
default_sizes = viz.load_data_from_json("sizes.json")
default_colors = viz.load_data_from_json("colors.json")
default_cmaps = viz.give_colormaps()

from src.visualization.one_simple_plot import plot_one_aesthetic_plot

from scipy.stats import entropy
# from skimage.measure import shannon_entropy

# ====================================================================
# cell 2
# ====================================================================
# Colors 

lecker_red = default_colors["warms"]["LECKER_RED"]
bone_white = default_colors["neutrals"]["BONE_WHITE"]
olive_gray = default_colors["neutrals"]["OLIVE_GRAY"]
halfblack = default_colors["neutrals"]["HALF_BLACK"]

# get the colors for -1 and 1
cmap = default_cmaps["db_bw_lr"].resampled(2)
color_neg1 = cmap(0)
color_pos1 = cmap(1)

black_white_map = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_black_white', 
        [
            bone_white,
            "black",
         ]
    )

# ====================================================================
# cell 3
# ====================================================================
# dataset_name = "lexis_data" # "suarez_MaMI_dataset"
# experiment_name = "05_mst_animal_0" # 
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from experiments_config import (MORPHO_DATASET, MORPHO_EXP, ROOT_DIR as _RD,
                                DIST_MATRIX_PATH, INDIVIDUALS_PATH, to_distance)

dataset_name = MORPHO_DATASET
experiment_name = MORPHO_EXP

# experiment_name = "02_ring_sweeps_animal_0" # 05_mst_animal_0" # 
# experiment_name = "02_ring_sweeps_animal_0" # "01_ring_sweep_animal_0" # "95_ring_seed_100_sweep_animal_206"
# experiment_name = "03_no_ring_sweeps_animal_0"
base_path = (_RD / "output" / "gnm" / dataset_name / experiment_name)
save_path = base_path / f"all_metrics_for_{experiment_name}.csv"


filtering_df_path = base_path / f"all_metrics_for_{experiment_name}.csv" # _updated.csv"
filtering_df = pd.read_csv(filtering_df_path)

output_path = Path(base_path) / "ground_truth_analysis"
output_path.mkdir(exist_ok=True)

# ====================================================================
# cell 4
# ====================================================================
# Get the name of all measures
all_dist_measures = [
    "energy", 
    # "portrait", 
    # "spectral_distance_adjacency",
    # "spectral_distance_norm_laplacian", 
    
    # "communicability_corr",
    # "communicability_jsd",
    # "network_mutual_information",
    # "dc_network_mutual_information", 
    
    # "net_simile", 
    # "netrd_non_backtracking_spectral", 
    # "resistance", 
    # "delta_con", 
    
    # "f1", 
    # "hamming",
    # "frobenius", 
    # "jaccard", 
    
    "test_energy_degree", 
    "test_energy_clustering", 
    "test_energy_edge_length", 
    "test_energy_betweenness"
]




# ====================================================================
# cell 5
# ====================================================================
# Method names for plotting

method_names = {
    'portrait': 'Portrait', #  Divergence',
    'energy': 'Energy', #  Distance',
        'spectral_distance': 'Spectral Distance', # Different versions?? 
    'spectral_distance_laplacian': 'Spectral (Laplacian)',
    'spectral_distance_norm_laplacian': 'Spectral (Normalized Laplacian)',
    'spectral_distance_adjacency': 'Spectral (Adjacency)',
    'f1': 'F1', #  Score',
    'hamming': 'Hamming', # Distance',
        'communicability': 'Communicability Correlation', #  Distance',
        'graph_kernel': 'Graph Kernel Distance old',
        'multiplex_layer_similarity': 'Multiplex Layer Similarity',
    'cosine_embedding': 'Cosine Embedding', #  Distance',
    'graph_edit_distance': 'Graph Edit', #  Distance',
    'network_mutual_information': 'Network Mutual Information',
    'dc_network_mutual_information': 'DC Network Mutual Information',
        'hungarian_alignment': 'Hungarian Alignment',
    'graph_kernel_networkx': 'Graph Kernel', #  Distance',
    'delta_con': 'DeltaCon',
    'delta_con_distance': 'DeltaCon (distance)', # Distance',
        'resistance_distance': 'Resistance old', #  Distance',
        'wasserstein_gromov': 'Wasserstein-Gromov Distance',
    'wasserstein_sinkhorn': 'Wasserstein-Sinkhorn', #  Distance',

        'edit_distance': 'Edit Distance old',
    'communicability_corr': 'Communicability Correlation',
    'communicability_mse': 'Communicability MSE',
    'communicability_jsd': 'Communicability JSD',
    'frobenius': 'Frobenius', #  Distance',
    'jaccard': 'Jaccard', #  Distance',

    'resistance': 'Resistance', #  Distance',
    'net_simile': 'NetSimile', #  Distance',
    # 'net_smile': 'NetSimile', #  Distance', # !!!! Typo, remove this later (after 05 it should be fixed)
    'net_lsd': 'NetLSD', #  Distance',
    'quantum_jsd': 'Quantum JSD', #  Distance',
    'graph_diffusion': 'Graph Diffusion', #  Distance',
    'polynomial_dissimilarity': 'Polynomial Dissimilarity', #  Distance',
    'degree_divergence': 'Degree Divergence', #  Distance',
    'onion_divergence': 'Onion Divergence', #  Distance',

    "network_mutual_information_old": "Network Mutual Information old",
    "netrd_deltacon": "NetRD DeltaCon",
    "netrd_communicability_jsd": "NetRD Communicability JSD",
    "distributional_nbd": "Distributional NBD",
    "dk_series": "DK Series",
    "d_measure": "D Measure",
    "netrd_frobenius": "NetRD Frobenius",
    "netrd_hamming": "NetRD Hamming",
    "hamming_ipsen_mikhailov": "Hamming Ipsen-Mikhailov",
    "ipsen_mikhailov": "Ipsen-Mikhailov",
    "netrd_jaccard": "NetRD Jaccard",
    "netrd_laplacian_spectral": "NetRD Laplacian Spectral",
    "netrd_non_backtracking_spectral": "Spectral (Non-Backtracking)", # NetRD Non-Backtracking Spectral",
    "netrd_portrait_divergence": "NetRD Portrait Divergence",
    
    "test_energy_degree": "KS Degree", 
    "test_energy_clustering": "KS Clustering",
    "test_energy_edge_length": "KS Edge Length",
    "test_energy_betweenness": "KS Betweenness",
}

# Individual heatmaps (works like a charm)
# for m in all_methods:
#     plot_comparison_landscape(m)

# metric_colors = {}
# colors = sns.color_palette("tab20", len(method_names.values()))
# for i, (metric_name, metric_label) in enumerate(zip(method_names.keys(), method_names.values())):
#     metric_colors[metric_name] = colors[i]


# ====================================================================
# cell 6
# ====================================================================
# Get values for every measure

matrices_of_metrics = {}

for idx, mode in enumerate(all_dist_measures):

    path = f"output/gnm/{dataset_name}/{experiment_name}/summary_indiv_{mode}_for_exp_{experiment_name}.csv"
    df = pd.read_csv(path)
    
    # Get the metric column
    metric_col = [col for col in df.columns if col not in ["eta", "gamma", "network_index", "filename", "id"]]
    if not metric_col:
        raise ValueError(f"No metric column found in {df.columns}")
    
    # Round eta and gamma to a reasonable number of decimal places (e.g., 2 or 3)
    df['eta'] = df['eta'].round(2)
    df['gamma'] = df['gamma'].round(2)
    filtering_df['eta'] = filtering_df['eta'].round(2)
    filtering_df['gamma'] = filtering_df['gamma'].round(2)

    # Now merge
    if "no" in experiment_name: # FILTER BY NUMBER COMPONENTS. Alternative: not "ring" in experiment_name:
        df = df.merge(filtering_df[['eta', 'gamma', 'n_connected_components']], 
                    on=['eta', 'gamma'], 
                    how='left')
        # Filter
        df = df[df['n_connected_components'] <= 1]
    
    else: 
        df = df.merge(filtering_df[['eta', 'gamma']], 
                    on=['eta', 'gamma'], 
                    how='left')



    # Turn df into a pivot table. If there are multiple values for one eta-gamma pair, take the mean
    df_pivot = df.pivot_table(index='gamma', columns='eta', values=metric_col[0], aggfunc='mean')

    # Min-max normalize the values to [0, 1]
    df_pivot = (df_pivot - df_pivot.min().min()) / (df_pivot.max().max() - df_pivot.min().min())
    
    # Store the matrix as a numpy array
    matrix = df_pivot.values[::-1]
    if mode in ["f1", 
                "multiplex_layer_similarity", 
                "jaccard", 
                # "communicability_jsd", 
                "communicability_corr",
                "network_mutual_information", 
                "dc_network_mutual_information", 
            ]: # TODO: WHICH OTHER SIMILARITY METRICS? 
        matrix = 1 - matrix  # Invert for similarity measures

    # if mode == "resistance":
    #     matrix = np.log1p(matrix)  # Apply log1p transformation
        
    matrices_of_metrics[mode] = matrix


# ====================================================================
# cell 7
# ====================================================================
measures = all_dist_measures

for m in measures: 
    plt.imshow(np.log(matrices_of_metrics[m]), cmap='viridis')
    plt.colorbar()
    plt.title(m)
    plt.show()

# ====================================================================
# cell 8
# ====================================================================
measures = all_dist_measures

for m in measures: 
    plt.imshow(matrices_of_metrics[m], cmap='viridis')
    plt.colorbar()
    plt.title(m)
    plt.show()

# ====================================================================
# cell 9
# ====================================================================
n_methods = len(all_dist_measures)
n_cols = 5
n_rows = int(np.ceil(n_methods / n_cols))

# Calculate height to maintain roughly square subplots
height = 18 * (n_rows / n_cols) * 1.2 # 0.9 is random factor for colorbar

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((18, height)), 
                            dpi=150,
                            sharex=True, 
                            sharey=True)

axes = axes.flatten()

for idx, mode in enumerate(all_dist_measures):
    ax = axes[idx]
    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            # metric_colors[mode], 
            # "white", 
            bone_white
        #  
        #  color_neg1,
        #  halfblack, 
        #  bone_white, 
         ]
    )
    # cmap = default_cmaps["metric_purple_beige"]

    scatter = ax.imshow(matrices_of_metrics[mode], 
                        cmap=cmap, 
                       aspect='auto',
                    #    origin='lower',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # ax.set_xlim(-8, 3)
    # ax.set_ylim(-0.1, 1)
    
    # Set ticks
    ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
    ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
    
    # Add title for each subplot
    print(mode)
    title = method_names[mode] # f"{metric_col[0].split('subject')[0].replace('_', ' ').capitalize()}"
    ax.set_title(title, fontsize=8)
    
    # # Add colorbar for each subplot
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.tick_params(labelsize=6)
    
    # Only add labels to edge subplots
    if idx % n_cols == 0:  # Left column
        ax.set_ylabel(r"$\gamma$")
    if idx >= n_methods - n_cols:  # Bottom row
        ax.set_xlabel(r"$\eta$")

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Add one shared colorbar below all subplots
# cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.01])  # [left, bottom, width, height]
# cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
# cbar.set_label("Darker: More similar. Lighter: Less similar")
# cbar.ax.tick_params(labelsize=8)
plt.tight_layout()

plt.savefig(output_path / "components_of_ks_energy_grayscale.pdf")
print(output_path / "components_of_ks_energy_grayscale.pdf")

# ====================================================================
# cell 11
# ====================================================================
# Functions: Correlation matrix, clustering, reordering
def load_method_data(method):
    """Load data for a specific method."""
    path = f"{base_path}/summary_indiv_{method}_for_exp_{experiment_name}.csv"
    try:
        df = pd.read_csv(path)
        metric_cols = [col for col in df.columns 
                      if col not in ["eta", "gamma", "network_index", "filename", "id"]]
        if metric_cols:
            df['metric_value'] = to_distance(df[metric_cols[0]].values, method)
            df['method'] = method
            return df[['eta', 'gamma', 'id', 'metric_value', 'method']]
        return None
    except FileNotFoundError:
        print(f"Warning: File not found for method {method}")
        return None

def calculate_ground_truth(all_data):
    """Calculate approximated ground truth as max across methods for each eta-gamma-id."""
    ground_truth = all_data.groupby(['eta', 'gamma', 'id'])['metric_value'].max().reset_index()
    ground_truth.rename(columns={'metric_value': 'approximated_ground_truth'}, inplace=True)
    return ground_truth

def create_full_dataset(all_data, ground_truth):
    """Create dataset with all methods and ground truth."""
    pivot_data = all_data.pivot_table(
        index=['eta', 'gamma', 'id'],
        columns='method',
        values='metric_value'
    ).reset_index()
    
    if ground_truth is not None:
        full_data = pivot_data.merge(ground_truth, on=['eta', 'gamma', 'id'])
    else: 
        full_data = pivot_data
    return full_data

def calculate_correlation_matrix(full_data, methods):
    """Calculate Pearson correlation matrix between methods and ground truth."""
    cols_to_correlate = methods # + ['approximated_ground_truth']
    corr_matrix = full_data[cols_to_correlate].corr(method='pearson')
    return corr_matrix

def cluster_correlation_matrix(corr_matrix):
    """
    Cluster the correlation matrix using hierarchical clustering.
    Returns the reordered correlation matrix and the order of items.
    """
    # Convert correlation to distance
    # Use 1 - correlation for positive correlations (closer to 1 = more similar)
    distance_matrix = 1 - corr_matrix.values
    
    # Ensure all values are non-negative and clip to avoid numerical issues
    distance_matrix = np.clip(distance_matrix, 0, 2)
    
    # Make sure the matrix is symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Set diagonal to 0
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed distance matrix
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Replace any NaN or inf values
    condensed_dist = np.nan_to_num(condensed_dist, nan=1.0, posinf=2.0, neginf=0.0)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')  # Changed to 'average' which is more robust
    
    # Get the order from the dendrogram
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']

    # Reorder the correlation matrix
    ordered_corr = corr_matrix.iloc[order, order]
    
    return ordered_corr, order, linkage_matrix


# ====================================================================
# cell 12
# ====================================================================
print("Loading method data...")
all_data_list = []
loaded_methods = []

for method in all_dist_measures:
    df = load_method_data(method)
    if df is not None:
        all_data_list.append(df)
        loaded_methods.append(method)
        print(f"Loaded {method}: {len(df)} rows")
    
all_data = pd.concat(all_data_list, ignore_index=True)

full_data = create_full_dataset(all_data, ground_truth=None)
corr_matrix = calculate_correlation_matrix(full_data, loaded_methods)

corr_path = output_path / 'correlation_matrix.csv'
corr_matrix.to_csv(corr_path)


# Cluster the correlation matrix
clustered_corr, order, linkage_matrix = cluster_correlation_matrix(corr_matrix)

# Reorder all_methods based on clustering
all_dist_measures = [all_dist_measures[i] for i in order]  

# ====================================================================
# cell 14
# ====================================================================
# Functions: Visual Quality Analyzation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import pandas as pd

class VisualQualityAnalyzer:
    """Analyze visual quality and structure of similarity matrices"""
    
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)
        
    def calculate_gradient_smoothness(self):
        """Lower = smoother gradients, less noisy"""
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)
        
        # Variation in gradients indicates noise
        smoothness_score = np.std(grad_y) + np.std(grad_x)
        return smoothness_score
    
    def calculate_total_variation(self):
        """Total variation - lower = smoother"""
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)
        
        tv = np.sum(np.abs(grad_y)) + np.sum(np.abs(grad_x[:, :-1]))
        return tv / self.matrix.size
    
    def calculate_local_coherence(self, window_size=3):
        """Higher = more locally coherent"""
        local_mean = uniform_filter(self.matrix, size=window_size)
        local_sq_mean = uniform_filter(self.matrix**2, size=window_size)
        local_var = local_sq_mean - local_mean**2
        
        # Avoid division by zero
        coherence = 1 / (np.mean(local_var) + 1e-10)
        return coherence
    
    def calculate_snr(self, sigma=2):
        """Signal-to-noise ratio - higher = clearer structure"""
        smoothed = gaussian_filter(self.matrix, sigma=sigma)
        noise = self.matrix - smoothed
        
        signal_power = np.var(smoothed)
        noise_power = np.var(noise)
        
        if noise_power < 1e-10:
            return 100
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def calculate_structure_similarity(self, sigma=5):
        """Similarity to ideal smooth version - higher = better"""
        ideal = gaussian_filter(self.matrix, sigma=sigma)
        
        data_range = self.matrix.max() - self.matrix.min()
        if data_range < 1e-10:
            return 0
            
        score = ssim(self.matrix, ideal, data_range=data_range)
        return score
    
    def calculate_entropy(self):
        """Shannon entropy - for reference"""
        return shannon_entropy(self.matrix)
    
    def calculate_contrast(self):
        """Dynamic range"""
        return self.matrix.max() - self.matrix.min()
    
    def calculate_edge_strength(self):
        """Measure edge clarity using Sobel-like gradients"""
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)
        
        # Pad to same size
        grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
        grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        
        # Gradient magnitude
        grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        return np.mean(grad_magnitude)
    
    def get_all_metrics(self):
        """Calculate all quality metrics"""
        return {
            'entropy': self.calculate_entropy(),
            'gradient_smoothness': self.calculate_gradient_smoothness(),
            'total_variation': self.calculate_total_variation(),
            'local_coherence': self.calculate_local_coherence(),
            'snr': self.calculate_snr(),
            'structure_similarity': self.calculate_structure_similarity(),
            'contrast': self.calculate_contrast(),
            'edge_strength': self.calculate_edge_strength()
        }
    
    def calculate_composite_quality(self):
        """
        Composite quality score emphasizing visual structure
        Higher = better visual quality
        """
        metrics = self.get_all_metrics()
        
        # Normalize and combine (weights tuned for visual quality)
        score = (
            metrics['snr'] * 0.25 +  # High SNR = clear structure
            metrics['structure_similarity'] * 50 * 0.25 +  # Similar to smooth ideal
            (1 / (metrics['gradient_smoothness'] + 0.1)) * 0.20 +  # Smooth gradients
            metrics['local_coherence'] * 0.15 +  # Locally coherent
            (1 / (metrics['total_variation'] + 0.01)) * 0.15  # Low total variation
        )
        
        return score


def analyze_all_methods(matrices_dict, nan_strategy='mean'):
    """Analyze all similarity methods"""
    results = {}
    
    for name, matrix in matrices_dict.items():
        matrix = matrix.copy()
        
        # Handle NaNs
        if np.isnan(matrix).any():
            if nan_strategy == 'mean':
                matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix))
            elif nan_strategy == 'zero':
                matrix = np.nan_to_num(matrix, nan=0)
            elif nan_strategy == 'interpolate':
                # Simple interpolation
                mask = np.isnan(matrix)
                matrix[mask] = np.interp(np.flatnonzero(mask), 
                                        np.flatnonzero(~mask), 
                                        matrix[~mask])
        
        # min max normalize the matrix
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        analyzer = VisualQualityAnalyzer(matrix)
        results[name] = analyzer.get_all_metrics()
        results[name]['composite_quality'] = analyzer.calculate_composite_quality()
    
    return pd.DataFrame(results).T

def plot_comprehensive_comparison(results_df, metric_colors=None):
    """Plot all metrics for comparison"""
    
    # Metrics to plot
    metrics = [
        'entropy', 'gradient_smoothness', 'total_variation', 'local_coherence',
        'snr', 'structure_similarity', 'contrast', 'composite_quality'
    ]
    
    # Better interpretation labels
    labels = {
        'entropy': 'Entropy\n(lower = more structure)',
        'gradient_smoothness': 'Gradient Smoothness\n(lower = less noisy)',
        'total_variation': 'Total Variation\n(lower = smoother)',
        'local_coherence': 'Local Coherence\n(higher = better)',
        'snr': 'Signal-to-Noise Ratio\n(higher = better)',
        'structure_similarity': 'Structure Similarity\n(higher = better)',
        'contrast': 'Contrast\n(higher = better)',
        'composite_quality': 'Composite Quality Score\n(higher = BETTER)'
    }
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Sort by metric value for better visualization
        sorted_data = results_df[metric].sort_values()
        
        # Get colors if provided
        if metric_colors:
            colors = [metric_colors.get(name, 'gray') for name in sorted_data.index]
        else:
            colors = 'steelblue'
        
        ax.barh(range(len(sorted_data)), sorted_data.values, color=colors)
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data.index, fontsize=8)
        ax.set_xlabel(labels.get(metric, metric))
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(axis='x', alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    # plt.show()
    
    return fig


def rank_methods(results_df, metric='composite_quality'):
    """Rank methods by a specific metric"""
    ranked = results_df.sort_values(metric, ascending=False)
    
    print(f"\n{'='*70}")
    print(f"RANKING BY: {metric.upper()}")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Method':<30} {'Score':>10}")
    print(f"{'-'*70}")
    
    for idx, (method, row) in enumerate(ranked.iterrows(), 1):
        print(f"{idx:<6} {method:<30} {row[metric]:>10.4f}")
    
    print(f"{'='*70}\n")
    
    return ranked


# ====================================================================
# cell 15
# ====================================================================
# With your actual data
results_df = analyze_all_methods(matrices_of_metrics)

# View full results
print(results_df.round(4))

# Rank by composite quality (should match visual inspection better)
# ranked = rank_methods(results_df, 'composite_quality')

# # Or rank by specific metrics
# rank_methods(results_df, 'snr')  # Signal-to-noise
# rank_methods(results_df, 'structure_similarity')  # Smoothness

# ====================================================================
# cell 16
# ====================================================================
# get indices of "ranked" in its ascending order of "snr"
snr_ranked_indices = results_df['snr'].sort_values(ascending=False).index.tolist()

metric_colors = {}
# colors = sns.color_palette("tab20", len(method_names.values()))
colors = sns.color_palette("tab20", len(method_names))
# for i, (metric_name, metric_label) in enumerate(zip(snr_ranked_indices.keys(), method_names.values())):
#     metric_colors[metric_name] = colors[i]

for i, metric_name in enumerate(snr_ranked_indices):
        metric_colors[metric_name] = colors[i]

# ====================================================================
# cell 17
# ====================================================================
len(snr_ranked_indices)

# ====================================================================
# cell 18
# ====================================================================
metric = "snr"
labels = {'snr': 'Signal-to-Noise Ratio'} # \n(higher = better)'}

fig = plt.figure(figsize=viz.cm_to_inch((9, 6)), dpi=150)
ax = fig.add_axes([0.15, 0.25, 0.7, 0.65])

sorted_data = results_df[metric].sort_values()
colors = [metric_colors.get(name) for name in sorted_data.index]
colors_new = colors.copy()

bars = ax.barh(range(len(sorted_data)), sorted_data.values, color=colors)
ax.set_yticks([])  # Remove y-axis ticks
ax.set_xlabel(labels.get(metric, metric))
# ax.set_title("SNR")
ax.grid(axis='x', alpha=0.3)

# Add colorbar on left side. It should start at the same height as the bars and end at the same height: So get the starting height of the first bar and the ending height of the last bar
bar_height = bars[0].get_height()
start_height = bars[0].get_y() + bar_height
end_height = bars[-1].get_y() + bar_height #    bars[-1].get_height()
# Scale this accordingly to the figure height
start_height = 0.25 + (start_height / len(sorted_data)) * 0.65
end_height = 0.25 + ((end_height + bars[-1].get_height()) / len(sorted_data)) * 0.65
bar_height = (bar_height / len(sorted_data)) * 0.65
# Use this to set the position of the colorbar
# cax = fig.add_axes([0.1, 0.25, 0.03, 0.65])
cax = fig.add_axes([0.1, start_height, 0.03, end_height - start_height - bar_height])
norm = plt.Normalize(vmin=0, vmax=len(sorted_data)-1)
sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(colors), norm=norm)
cb = plt.colorbar(sm, cax=cax)
cb.set_ticks([])

# Add legend below - ordered by appearance in barplot (reversed since bars go bottom to top)
handles = [patches.Patch(color=metric_colors[name], label=method_names[name]) 
            for name in reversed(sorted_data.index) if metric_colors]
legend = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.25), # 0.5, -0.15), 
                    ncol=3, fontsize=7, frameon=False)

# Add horizontal line below the first 5 entries
ax.axhline(y=15.5, color='black', linestyle='--')

# Add a box around the top 5 entries in the legend
fig.canvas.draw()  # Need to draw to get accurate positions

# Get legend box position in figure coordinates
legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())

# Calculate width of one column (ncol=3, so 5 entries = 2 full columns)
col_width = legend_bbox.width / 3
box_width = col_width * 1 # 2  # First 5 entries span ~2 columns


height_legend_box = legend_bbox.height + 0.01

rect = patches.Rectangle(
    (legend_bbox.x0 - 0.005, legend_bbox.y0 + height_legend_box * 2/7), #  - 0.005),  # Small padding
    box_width - 0.03, # + 0.01, 
    height_legend_box*5/7 - 0.005,
    # linewidth=1, 
    edgecolor='black', 
    facecolor='none',
    linestyle='--',
    transform=fig.transFigure, zorder=10
)


fig.patches.append(rect)

plt.show()

# HERE: GOOD PLOT. 

# ====================================================================
# cell 20
# ====================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import pandas as pd

class VisualQualityAnalyzer:
    """Analyze visual quality and structure of similarity matrices"""
    
    def __init__(self, matrix, mask=None):
        self.matrix = np.array(matrix, dtype=float)
        self.mask = mask
        
    def calculate_gradient_smoothness(self):
        """Lower = smoother gradients, less noisy"""
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)

        # # Apply mask if provided - but how? 
        # if self.mask is not None:
        #     grad_y = grad_y[self.mask[:-1, :]]
        #     grad_x = grad_x[:, self.mask[:-1]]

        # Variation in gradients indicates noise
        smoothness_score = np.std(grad_y) + np.std(grad_x)
        return smoothness_score
    
    def calculate_total_variation(self):
        """Total variation - lower = smoother"""
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)

        # Apply mask if provided - but how? 
        # grad_y = grad_y[self.mask[:-1, :]]
        # grad_x = grad_x[:, self.mask[:-1]]

        tv = np.sum(np.abs(grad_y)) + np.sum(np.abs(grad_x[:, :-1]))
        return tv / self.matrix.size
    
    def calculate_local_coherence(self, window_size=3):
        """Higher = more locally coherent"""
        local_mean = uniform_filter(self.matrix, size=window_size)
        local_sq_mean = uniform_filter(self.matrix**2, size=window_size)
        local_var = local_sq_mean - local_mean**2
        
        local_var = local_var[self.mask]
        
        # Avoid division by zero
        coherence = 1 / (np.mean(local_var) + 1e-10)
        return coherence
    
    def calculate_snr(self, sigma=2):
        """Signal-to-noise ratio - higher = clearer structure"""
        smoothed = gaussian_filter(self.matrix, sigma=sigma)
        noise = self.matrix - smoothed
        
        # Remove all masked values (NaNs) from calculation
        smoothed = smoothed[self.mask]
        noise = noise[self.mask]

        signal_power = np.var(smoothed - np.mean(smoothed))
        noise_power = np.var(noise)
        
        if noise_power < 1e-10:
            return 100
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def calculate_structure_similarity(self, sigma=5):
        """Similarity to ideal smooth version - higher = better"""
        ideal = gaussian_filter(self.matrix, sigma=sigma)
        
        data_range = self.matrix.max() - self.matrix.min()
        if data_range < 1e-10:
            return 0
            
        score = ssim(self.matrix, ideal, data_range=data_range)
        return score
    
    def calculate_entropy(self):
        """Shannon entropy - for reference"""
        return shannon_entropy(self.matrix)
    
    # def calculate_contrast(self):
    #     """Dynamic range"""
    #     return self.matrix.max() - self.matrix.min()
    
    def calculate_edge_strength(self):
        """Measure edge clarity using Sobel-like gradients"""
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)
        
        # Pad to same size
        grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
        grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        
        # Gradient magnitude
        grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        return np.mean(grad_magnitude)
    
    def get_all_metrics(self):
        """Calculate all quality metrics"""
        return {
            'entropy': self.calculate_entropy(),
            'gradient_smoothness': self.calculate_gradient_smoothness(),
            'total_variation': self.calculate_total_variation(),
            'local_coherence': self.calculate_local_coherence(),
            'snr': self.calculate_snr(),
            'structure_similarity': self.calculate_structure_similarity(),
            # 'contrast': self.calculate_contrast(),
            'edge_strength': self.calculate_edge_strength()
        }
    
    # def calculate_composite_quality(self):
    #     """
    #     Composite quality score emphasizing visual structure
    #     Higher = better visual quality
    #     """
    #     metrics = self.get_all_metrics()
        
    #     # Normalize and combine (weights tuned for visual quality)
    #     score = (
    #         metrics['snr'] * 0.25 +  # High SNR = clear structure
    #         metrics['structure_similarity'] * 50 * 0.25 +  # Similar to smooth ideal
    #         (1 / (metrics['gradient_smoothness'] + 0.1)) * 0.20 +  # Smooth gradients
    #         metrics['local_coherence'] * 0.15 +  # Locally coherent
    #         (1 / (metrics['total_variation'] + 0.01)) * 0.15  # Low total variation
    #     )
        
    #     return score


# def analyze_all_methods(matrices_dict):
#     """Analyze all similarity methods"""
#     results = {}
    
#     for name, matrix in matrices_dict.items():
        
#         # min max normalize the matrix
#         matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
#         analyzer = VisualQualityAnalyzer(matrix)
#         results[name] = analyzer.get_all_metrics()
#         # results[name]['composite_quality'] = analyzer.calculate_composite_quality()
    
#     return pd.DataFrame(results).T

def analyze_all_methods(matrices_dict, nan_strategy='mean'):
    """Analyze all similarity methods"""
    results = {}
    
    for name, matrix in matrices_dict.items():
        matrix = matrix.copy()
        
        # Turn mask into a 0, 1 mask (from the current nan positions)
        mask = np.ones_like(matrix, dtype=bool)
        mask[np.isnan(matrix)] = False
        
        # Handle NaNs
        if np.isnan(matrix).any():
            if nan_strategy == 'mean':
                matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix))
            elif nan_strategy == 'zero':
                matrix = np.nan_to_num(matrix, nan=0)
            elif nan_strategy == 'interpolate':
                # Simple interpolation
                mask = np.isnan(matrix)
                matrix[mask] = np.interp(np.flatnonzero(mask), 
                                        np.flatnonzero(~mask), 
                                        matrix[~mask])
        
        # min max normalize the matrix
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        analyzer = VisualQualityAnalyzer(matrix)
        results[name] = analyzer.get_all_metrics()
        # results[name]['composite_quality'] = analyzer.calculate_composite_quality()
    
    return pd.DataFrame(results).T


def plot_comprehensive_comparison(results_df, metric_colors=None):
    """Plot all metrics for comparison"""
    
    # Metrics to plot
    metrics = [
        'entropy', 'gradient_smoothness', 'total_variation', 'local_coherence',
        'snr', 'structure_similarity', # 'contrast', 'composite_quality'
    ]
    
    # Better interpretation labels
    labels = {
        'entropy': 'Entropy\n(lower = more structure)',
        'gradient_smoothness': 'Gradient Smoothness\n(lower = less noisy)',
        'total_variation': 'Total Variation\n(lower = smoother)',
        'local_coherence': 'Local Coherence\n(higher = better)',
        'snr': 'Signal-to-Noise Ratio\n(higher = better)',
        'structure_similarity': 'Structure Similarity\n(higher = better)',
        # 'contrast': 'Contrast\n(higher = better)',
        # 'composite_quality': 'Composite Quality Score\n(higher = BETTER)'
    }
    
    # Get order of snr ranking
    # snr_ranked_indices = results_df['snr'].sort_values(ascending=False).index.tolist()[::-1]
    
    snr_ranked_indices = all_dist_measures # [
                            # # 'graph_kernel_networkx',
                            # # 'communicability_jsd',
                            # 'delta_con',
                            # 'delta_con_distance',
                            # 'hamming',
                            # # 'graph_kernel',
                            # # 'wasserstein_sinkhorn',
                            # # 'wasserstein_gromov',
                            # 'portrait',
                            # # 'spectral_distance',
                            # # 'jaccard',
                            # # 'multiplex_layer_similarity',
                            # # 'edit_distance',
                            # # 'frobenius',
                            # 'f1',
                            # 'graph_edit_distance',
                            # 'network_mutual_information',
                            # 'energy',
                            # 'communicability',
                            # 'communicability_mse',
                            # 'cosine_embedding',
                            # 'hungarian_alignment',
                            # 'resistance_distance'
                            # ]
                                
    results_df = results_df.loc[snr_ranked_indices]
    
    # Create subplots
    fig, axes = plt.subplots(6, 1, figsize=(6, 18))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        sorted_data = results_df[metric]
        
        # Get colors if provided
        if metric_colors:
            colors = [metric_colors.get(name, 'gray') for name in sorted_data.index]
        else:
            colors = 'steelblue'
        
        ax.barh(range(len(sorted_data)), sorted_data.values, color=colors)
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data.index, fontsize=8)
        ax.set_xlabel(labels.get(metric, metric))
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(axis='x', alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    # plt.show()
    
    return fig


def rank_methods(results_df, metric='composite_quality'):
    """Rank methods by a specific metric"""
    ranked = results_df.sort_values(metric, ascending=False)
    
    print(f"\n{'='*70}")
    print(f"RANKING BY: {metric.upper()}")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Method':<30} {'Score':>10}")
    print(f"{'-'*70}")
    
    for idx, (method, row) in enumerate(ranked.iterrows(), 1):
        print(f"{idx:<6} {method:<30} {row[metric]:>10.4f}")
    
    print(f"{'='*70}\n")
    
    return ranked



results_df = analyze_all_methods(matrices_of_metrics)

# Rank 
ranked = rank_methods(results_df, "snr") # , 'composite_quality')

# Plot everything
plot_comprehensive_comparison(results_df, metric_colors)

# ====================================================================
# cell 22
# ====================================================================
n_methods = len(all_dist_measures)
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

# Calculate height to maintain roughly square subplots
height = 18 * (n_rows / n_cols) # * 0.9 # 0.9 is random factor for colorbar

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((18, height)), 
                            dpi=150,
                            sharex=True, 
                            sharey=True)

axes = axes.flatten()

for idx, mode in enumerate(all_dist_measures):
    ax = axes[idx]
    
    # matrices_of_metrics[mode]
    # Create scatter plot
    # scatter = ax.scatter(df["eta"], df["gamma"], 
    #                     c=df[metric_col[0]], 
    #                     cmap=default_cmaps["metric_purple_beige"], 
    #                     s=2.4)
    scatter = ax.imshow(matrices_of_metrics[mode], 
                        cmap=default_cmaps["metric_purple_beige"], 
                       aspect='auto',
                    #    origin='lower',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # ax.set_xlim(-8, 3)
    # ax.set_ylim(-0.1, 1)
    
    # Set ticks
    ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
    ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
    
    # Add title for each subplot
    print(mode)
    title = method_names[mode] # f"{metric_col[0].split('subject')[0].replace('_', ' ').capitalize()}"
    ax.set_title(title, fontsize=8)
    
    # # Add colorbar for each subplot
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.tick_params(labelsize=6)
    
    # Only add labels to edge subplots
    if idx % n_cols == 0:  # Left column
        ax.set_ylabel(r"$\gamma$")
    if idx >= n_methods - n_cols:  # Bottom row
        ax.set_xlabel(r"$\eta$")

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Add one shared colorbar below all subplots
cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.01])  # [left, bottom, width, height]
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Darker: More similar. Lighter: Less similar")
cbar.ax.tick_params(labelsize=8)
plt.tight_layout()


# ====================================================================
# cell 23
# ====================================================================
n_methods = len(all_dist_measures)
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

# Calculate height to maintain roughly square subplots
height = 18 * (n_rows / n_cols) # * 0.9 # 0.9 is random factor for colorbar

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((18, height)), 
                            dpi=150,
                            sharex=True, 
                            sharey=True)

axes = axes.flatten()

for idx, mode in enumerate(all_dist_measures):
    ax = axes[idx]
    
    # matrices_of_metrics[mode]
    # Create scatter plot
    # scatter = ax.scatter(df["eta"], df["gamma"], 
    #                     c=df[metric_col[0]], 
    #                     cmap=default_cmaps["metric_purple_beige"], 
    #                     s=2.4)
    
    # create cmap from metric_colors: going from bone_white to metric_colors[mode]
    # cmap = "cividis" 
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [metric_colors[mode], "black"
         # bone_white, 
        ]
    )

    scatter = ax.imshow(np.log(matrices_of_metrics[mode]), 
                        cmap=cmap, 
                       aspect='auto',
                    #    origin='lower',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # ax.set_xlim(-8, 3)
    # ax.set_ylim(-0.1, 1)
    
    # Set ticks
    ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
    ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
    
    # Add title for each subplot
    print(mode)
    title = method_names[mode] # f"{metric_col[0].split('subject')[0].replace('_', ' ').capitalize()}"
    ax.set_title(title, fontsize=8)
    
    # # Add colorbar for each subplot
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.tick_params(labelsize=6)
    
    # Only add labels to edge subplots
    if idx % n_cols == 0:  # Left column
        ax.set_ylabel(r"$\gamma$")
    if idx >= n_methods - n_cols:  # Bottom row
        ax.set_xlabel(r"$\eta$")

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Add one shared colorbar below all subplots
cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.01])  # [left, bottom, width, height]
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Darker: More similar. Lighter: Less similar")
cbar.ax.tick_params(labelsize=8)
plt.tight_layout()

# ====================================================================
# cell 24
# ====================================================================
selected_methods = [
    "energy",
    "portrait",
    "delta_con",
    "hamming",
    "frobenius",
    "spectral_distance_adjacency",
    # "spectral_distance_laplacian",
    "spectral_distance_norm_laplacian",
    "communicability_mse",
    "communicability_corr", 
    "communicability_jsd",
    "network_mutual_information",
    "dc_network_mutual_information",
    "net_lsd",
    "net_smile",
    "resistance",
    "quantum_jsd",
    # "delta_con_distance",
    # "f1",
    # "jaccard",
    # "graph_kernel_networkx",
    # "wasserstein_sinkhorn"
]

# ====================================================================
# cell 25
# ====================================================================
methods_for_gridplot = all_dist_measures # [
#         "energy",
#         "f1", # "one_minus_f1", 
#         "portrait", 
#         "delta_con", 
#         "spectral_distance",
#         "edit_distance", 
#             "graph_edit_distance", # braucht ewig 
#             "network_mutual_information", 
#         "wasserstein_gromov",
#         "graph_kernel_networkx", 
#         "communicability_jsd", 
#         "frobenius", 
#         "jaccard", # one_minus_jaccard", 
         
        
#         # "f1", 
#         "multiplex_layer_similarity", # one_minus_multiplex_layer_similarity",
#         # "jaccard", 
#         # "communicability", 
#         "cosine_embedding", 
#         # "wasserstein_sinkhorn", 
#         "hungarian_alignment", 
#         # "graph_kernel", 
        
#         # "resistance_distance", 
        
#         # "graph_edit_distance", 
#         # "network_mutual_information", 
#         # "communicability_mse", 
        
        
# ]
        
n_methods = len(methods_for_gridplot)
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

# Calculate height to maintain roughly square subplots
height = 18 * (n_rows / n_cols) # * 0.9 # 0.9 is random factor for colorbar

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((18, height)), 
                            dpi=150,
                            sharex=True, 
                            sharey=True)

axes = axes.flatten()

for idx, mode in enumerate(methods_for_gridplot):
    ax = axes[idx]
    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            # metric_colors[mode], 
            # "white", 
            bone_white
        #  
        #  color_neg1,
        #  halfblack, 
        #  bone_white, 
         ]
    )
    # cmap = default_cmaps["metric_purple_beige"]

    scatter = ax.imshow(np.log(matrices_of_metrics[mode]), # matrices_of_metrics[mode], 
                        cmap=cmap, 
                       aspect='auto',
                    #    origin='lower',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # ax.set_xlim(-8, 3)
    # ax.set_ylim(-0.1, 1)
    
    # Set ticks
    ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
    ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
    
    # Add title for each subplot
    print(mode)
    title = method_names[mode] # f"{metric_col[0].split('subject')[0].replace('_', ' ').capitalize()}"
    ax.set_title(title, fontsize=8)
    
    # # Add colorbar for each subplot
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.tick_params(labelsize=6)
    
    # Only add labels to edge subplots
    if idx % n_cols == 0:  # Left column
        ax.set_ylabel(r"$\gamma$")
    if idx >= n_methods - n_cols:  # Bottom row
        ax.set_xlabel(r"$\eta$")

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Add one shared colorbar below all subplots
cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.01])  # [left, bottom, width, height]
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Darker: More similar. Lighter: Less similar")
cbar.ax.tick_params(labelsize=8)
plt.tight_layout()

plt.savefig(output_path / "all_similarity_plots_grayscale.pdf")
print(output_path / "all_similarity_plots_grayscale.pdf")

# ====================================================================
# cell 26
# ====================================================================
methods_for_gridplot = all_dist_measures # [
#         "energy",
#         "f1", # "one_minus_f1", 
#         "portrait", 
#         "delta_con", 
#         "spectral_distance",
#         "edit_distance", 
#             "graph_edit_distance", # braucht ewig 
#             "network_mutual_information", 
#         "wasserstein_gromov",
#         "graph_kernel_networkx", 
#         "communicability_jsd", 
#         "frobenius", 
#         "jaccard", # one_minus_jaccard", 
         
        
#         # "f1", 
#         "multiplex_layer_similarity", # one_minus_multiplex_layer_similarity",
#         # "jaccard", 
#         # "communicability", 
#         "cosine_embedding", 
#         # "wasserstein_sinkhorn", 
#         "hungarian_alignment", 
#         # "graph_kernel", 
        
#         # "resistance_distance", 
        
#         # "graph_edit_distance", 
#         # "network_mutual_information", 
#         # "communicability_mse", 
        
        
# ]
        
n_methods = len(methods_for_gridplot)
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

# Calculate height to maintain roughly square subplots
height = 18 * (n_rows / n_cols) # * 0.9 # 0.9 is random factor for colorbar

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((18, height)), 
                            dpi=150,
                            sharex=True, 
                            sharey=True)

axes = axes.flatten()

for idx, mode in enumerate(methods_for_gridplot):
    ax = axes[idx]
    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            # metric_colors[mode], 
            # "white", 
            bone_white
        #  
        #  color_neg1,
        #  halfblack, 
        #  bone_white, 
         ]
    )
    # cmap = default_cmaps["metric_purple_beige"]

    scatter = ax.imshow(np.log(matrices_of_metrics[mode]), # matrices_of_metrics[mode], 
                        cmap=cmap, 
                       aspect='auto',
                    #    origin='lower',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # ax.set_xlim(-8, 3)
    # ax.set_ylim(-0.1, 1)
    
    # Set ticks
    ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
    ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])

    ax.vlines(x=0, 
              ymin=df["gamma"].min(), ymax=df["gamma"].max(), 
              color='black', linestyle='--', linewidth=0.4)
    
    # Add title for each subplot
    print(mode)
    title = method_names[mode] # f"{metric_col[0].split('subject')[0].replace('_', ' ').capitalize()}"
    ax.set_title(title, fontsize=8)
    
    # # Add colorbar for each subplot
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.tick_params(labelsize=6)
    
    # Only add labels to edge subplots
    if idx % n_cols == 0:  # Left column
        ax.set_ylabel(r"$\gamma$")
    if idx >= n_methods - n_cols:  # Bottom row
        ax.set_xlabel(r"$\eta$")

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Add one shared colorbar below all subplots
cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.01])  # [left, bottom, width, height]
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Darker: More similar. Lighter: Less similar")
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([])  # Remove ticks
plt.tight_layout()

plt.savefig(output_path / "sixteen_similarity_plots_grayscale_log_with_x_eq_0.pdf")
print(output_path / "sixteen_similarity_plots_grayscale_log_with_x_eq_0.pdf")

# ====================================================================
# cell 27
# ====================================================================
all_dist_measures = methods_for_gridplot

# ====================================================================
# cell 28
# ====================================================================
selected_methods = all_dist_measures # ["energy", "spectral_distance", "portrait", "graph_kernel_networkx", "communicability_jsd", "delta_con"]


# ====================================================================
# cell 30
# ====================================================================
timing_data = []
method_labels = []

for method in selected_methods:
    # Construct the timing file path
    timing_file = Path(base_path) / f"timing_{method}_for_exp_{experiment_name}.csv"
    
    # Check if file exists
    if not timing_file.exists():
        print(f"Warning: File not found - {timing_file}")
        continue
    
    # Read timing data
    df = pd.read_csv(timing_file) # ALL dots! 
    # df = pd.read_csv(timing_file)[:2500]
    
    # Find the timing column
    timing_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['time']):
            timing_col = col
            break
    
    # Extract timing values
    timings = df[timing_col].dropna().values
    
    if len(timings) > 0:
        timing_data.append(timings)
        method_labels.append(method_names[method]) # .replace("_", " ").title())

# Order methods by median timing
# if order_by_median:
median_timings = [np.median(data) for data in timing_data]
sorted_indices = np.argsort(median_timings)[::-1]
timing_data = [timing_data[i] for i in sorted_indices]
method_labels = [method_labels[i] for i in sorted_indices]

print("number dots:", len(timing_data[0]))
# Create boxplot
plt.figure(figsize=viz.cm_to_inch((6,6))) # (8, 8))

bp = plt.boxplot(timing_data, 
                labels=method_labels,
                patch_artist=True,
                showmeans=False, # True,
                showfliers=False, # True, # False, 
                ) 

# Customize appearance
for patch, color in zip(bp['boxes'], [metric_colors[method] for method in selected_methods]):
    patch.set_facecolor(color)


# Change median lines to black
for median in bp['medians']:
    median.set_color('black')


plt.ylabel('Time (seconds)') 
plt.xticks([]) 
# # Add grid for better readability
plt.gca().yaxis.grid(True, alpha=0.3, linestyle='--')
plt.gca().set_axisbelow(True)

plt.tight_layout()


# ====================================================================
# cell 33
# ====================================================================
timing_data = []
method_labels = []
method_key_names_ordered = []

for method in selected_methods:
    # Construct the timing file path
    timing_file = Path(base_path) / f"timing_{method}_for_exp_{experiment_name}.csv"
    
    # Check if file exists
    if not timing_file.exists():
        print(f"Warning: File not found - {timing_file}")
        continue
    
    # Read timing data
    df = pd.read_csv(timing_file) # ALL dots! 
    # df = pd.read_csv(timing_file)[:2500]
    
    # Find the timing column
    timing_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['time']):
            timing_col = col
            break
    
    # Extract timing values
    timings = df[timing_col].dropna().values
    
    if len(timings) > 0:
        timing_data.append(timings)
        method_labels.append(method_names[method]) # .replace("_", " ").title())

# Order methods by median timing
# if order_by_median:
median_timings = [np.median(data) for data in timing_data]
sorted_indices = np.argsort(median_timings)[::-1]
timing_data = [timing_data[i] for i in sorted_indices]
method_labels = [method_labels[i] for i in sorted_indices]
method_key_names_ordered = [selected_methods[i] for i in sorted_indices]

print("number dots:", len(timing_data[0]))
# Create boxplot
plt.figure(figsize=viz.cm_to_inch((12,9))) # (8, 8))

bp = plt.boxplot(timing_data, 
                labels=method_labels,
                patch_artist=True,
                showmeans=False, # True,
                showfliers=False, # True, # False, 
                ) 

# Customize appearance
for patch, color in zip(bp['boxes'], [metric_colors[method] for method in method_key_names_ordered]):
    patch.set_facecolor(color)


# Change median lines to black
for median in bp['medians']:
    median.set_color('black')


plt.ylabel('Time (seconds)') 
    # plt.xticks([]) 
plt.xticks(rotation=90) 
plt.yscale('log')
# # Add grid for better readability
plt.gca().yaxis.grid(True, alpha=0.3, linestyle='--')
plt.gca().set_axisbelow(True)


plt.tight_layout()

plt.savefig(output_path / "similarity_methods_timing_comparison.pdf")
print(output_path / "similarity_methods_timing_comparison.pdf")


# ====================================================================
# cell 34
# ====================================================================
# metric = "snr"
# labels = {'snr': 'Signal-to-Noise Ratio'} # \n(higher = better)'}

fig = plt.figure(figsize=viz.cm_to_inch((18, 4)))  # Adjust size as needed
ax = fig.add_subplot(111)

# sorted_data = results_df[metric].sort_values()

# Create legend handles
handles = [patches.Patch(color=metric_colors[name], label=method_names[name]) 
           for name in method_key_names_ordered]

# Add legend
legend = ax.legend(handles=handles, loc='center', 
                   ncol=3, # fontsize=12, 
                   frameon=False)

# Hide the axes
ax.axis('off')

# plt.tight_layout()
plt.savefig(output_path / "method_legend.pdf")
print(output_path / "method_legend.pdf")
plt.show()

# ====================================================================
# cell 36
# ====================================================================
from scipy.cluster.hierarchy import fcluster

def plot_correlation_network(corr_matrix, save_path, threshold=0.5):
    """
    Create a network graph where methods are nodes and correlations are edges.
    Uses hierarchical clustering to position nodes.
    """
    # Get clustering information
    clustered_corr, order, linkage_matrix = cluster_correlation_matrix(corr_matrix)
    
    # Extract communities (adjust t parameter to control number of clusters)
    communities = fcluster(linkage_matrix, t=1.5, criterion='distance')
    
    # Create graph
    G = nx.Graph()
    methods = corr_matrix.columns.tolist()
    G.add_nodes_from(methods)
    
    # Add edges for correlations above threshold
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                corr_value = corr_matrix.loc[method1, method2]
                if not np.isnan(corr_value) and abs(corr_value) >= threshold:
                    G.add_edge(method1, method2, weight=abs(corr_value), 
                              correlation=corr_value)
    
    # Create community-informed layout
    # First, get positions for each community center
    unique_communities = np.unique(communities)
    n_communities = len(unique_communities)
    
    # Place community centers in a circle
    community_centers = {}
    for i, comm in enumerate(unique_communities):
        angle = 2 * np.pi * i / n_communities
        community_centers[comm] = np.array([np.cos(angle), np.sin(angle)]) * 3


    np.random.seed(0)  # For reproducibility
    # Initialize positions based on community membership
    initial_pos = {}
    for method, comm in zip(methods, communities):
        # Add small random offset within community
        offset = np.random.randn(2) * 0.3
        initial_pos[method] = community_centers[comm] + offset
    
    # Refine with spring layout using community-informed initial positions
    pos = nx.spring_layout(G, pos=initial_pos, k=2, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=viz.cm_to_inch((18,18)))
    
    # Draw nodes (keep original colors)
    node_colors = [metric_colors[node] for node in G.nodes()]
    node_sizes = [2000 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=1, ax=ax)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    # normalize weights
    weights = (weights - min(weights)) / (max(weights) - min(weights))
    
    correlations = [G[u][v]['correlation'] for u, v in edges]
    edge_widths = [w * 5 for w in weights]
    
    # use the extreme values of this: default_cmaps["db_bw_lr"]
    edge_colors = [color_neg1 if c < 0 else color_pos1 for c in correlations]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                          edge_color=edge_colors, ax=ax)
    
    # Draw labels
    labels = {node: method_names[node].replace(' ', '\n').replace('-', '-\n') 
              for node in G.nodes() if node in selected_methods}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved correlation network to {save_path}")
    
    return G


plot_correlation_network(corr_matrix, save_path, threshold=0.5)

# ====================================================================
# cell 37
# ====================================================================
from scipy.cluster.hierarchy import fcluster

def plot_correlation_network(corr_matrix, save_folder, selected_methods, threshold=0.5):
    """
    Create a network graph where methods are nodes and correlations are edges.
    Uses hierarchical clustering to position nodes.
    """
    # Get clustering information
    clustered_corr, order, linkage_matrix = cluster_correlation_matrix(corr_matrix)
    
    # Extract communities (adjust t parameter to control number of clusters)
    communities = fcluster(linkage_matrix, t=5, # 2, # 1.5, 
                           criterion='distance')
    
    # Create graph
    G = nx.Graph()
    methods = corr_matrix.columns.tolist()
    G.add_nodes_from(methods)
    
    # Add edges for correlations above threshold
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                corr_value = corr_matrix.loc[method1, method2]
                if not np.isnan(corr_value) and abs(corr_value) >= threshold:
                    G.add_edge(method1, method2, weight=abs(corr_value), 
                              correlation=corr_value)
    
    # Create community-informed layout
    # First, get positions for each community center
    unique_communities = np.unique(communities)
    n_communities = len(unique_communities)
    
    # Place community centers in a circle
    community_centers = {}
    for i, comm in enumerate(unique_communities):
        angle = 2 * np.pi * i / n_communities
        community_centers[comm] = np.array([np.cos(angle), np.sin(angle)]) * 3


    np.random.seed(0)  # For reproducibility
    # Initialize positions based on community membership
    initial_pos = {}
    for method, comm in zip(methods, communities):
        # Add small random offset within community
        offset = np.random.randn(2) * 0.3
        initial_pos[method] = community_centers[comm] + offset
    
    # Refine with spring layout using community-informed initial positions
    pos = nx.spring_layout(G, pos=initial_pos, k=3, # 1.5, 
                           iterations=55,  # 50 
                           seed=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=viz.cm_to_inch((6,6)), dpi=150)
    
    # Draw nodes (keep original colors)
    node_colors = [metric_colors[node] for node in G.nodes()]
    node_sizes = [300 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=1, ax=ax, linewidths=0.25, edgecolors='white')
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    # normalize weights
    weights = (weights - min(weights)) / (max(weights) - min(weights))
    
    correlations = [G[u][v]['correlation'] for u, v in edges]
    edge_widths = [w * 4 for w in weights]
    
    # use the extreme values of this: default_cmaps["db_bw_lr"]
    edge_colors = [color_neg1 if c < 0 else color_pos1 for c in correlations]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                          edge_color=edge_colors, ax=ax)
    label_shorthands = {
        "energy": "Energy", 
        "portrait": "Portrait",
        "spectral_distance_adjacency": "Spectral\nAdjacency",
        # "spectral_distance_norm_laplacian", 
        
        # "communicability_corr",
        "communicability_jsd": "Comm.\nJSD",
        # "network_mutual_information",
        # "dc_network_mutual_information", 
        
        "net_simile": "NetSimile", 
        "netrd_non_backtracking_spectral": "Spectral\nNon-BT", 
        "resistance": "Resistance", 
        "delta_con": "DeltaCon", 
        
        # "f1", 
        # "hamming",
        # "frobenius", 
        # "jaccard", 
    }
    # Draw labels
    # labels = {node: method_names[node].replace(' ', '\n').replace('-', '-\n') 
    #           for node in G.nodes() if node in selected_methods}
    labels = {node: label_shorthands.get(node, node) for node in G.nodes()}
    
    
    nx.draw_networkx_labels(G, pos, labels, font_size=5, ax=ax) # 3
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path / "correlation_graph.pdf")
    plt.show()
    print(f"Saved correlation network to {save_path}")
    print(output_path / "correlation_graph.pdf")
    
    return G




selected_methods_2 = [
    "energy", 
    "portrait",
    "spectral_distance_adjacency",
    # "spectral_distance_norm_laplacian", 
    
    # "communicability_corr",
    "communicability_jsd",
    # "network_mutual_information",
    # "dc_network_mutual_information", 
    
    "net_simile", 
    "netrd_non_backtracking_spectral", 
    "resistance", 
    "delta_con", 
    
    # "f1", 
    # "hamming",
    # "frobenius", 
    # "jaccard", 
]


plot_correlation_network(corr_matrix, save_folder=save_path, selected_methods=selected_methods_2, threshold=0.5)

# ====================================================================
# cell 38
# ====================================================================
# BUILT FOR 16 networks! (mit plt und ohne subplot) 

def plot_comprehensive_comparison(results_df, metric_colors=None):

    """Plot all metrics for comparison"""
    metric = "snr"
    labels = {'snr': 'Signal-to-Noise Ratio'} # \n(higher = better)'}

    fig = plt.figure(figsize=viz.cm_to_inch((6,6)), dpi=150)
    # ax = fig.add_axes([0.15, 0.25, 0.7, 0.65])
    
    sorted_data = results_df[metric].sort_values()[::-1]
    colors = [metric_colors.get(name) for name in sorted_data.index]

    plt.bar(range(len(sorted_data)), sorted_data.values, color=colors)
    plt.xticks([])  # Remove y-axis ticks

    plt.ylabel(labels.get(metric, metric))
    # plt.title("SNR")
    plt.grid(axis='x', alpha=0.3)

    # Add vertical line below the first 5 entries
    plt.axvline(x=4.5, color='black', linestyle='--')

plot_comprehensive_comparison(results_df, metric_colors)

# ====================================================================
# cell 39
# ====================================================================
all_methods_sorted_for_legend = [m for m in snr_ranked_indices if m in all_dist_measures]
all_methods_sorted_for_legend = all_methods_sorted_for_legend[::-1]

# ====================================================================
# cell 41
# ====================================================================
methods_for_gridplot = all_dist_measures 
# #[
#         "energy",
#         "f1", # "one_minus_f1", 
#         "portrait", 
#         "delta_con", 
#         "spectral_distance",
#         "edit_distance", 
#             "graph_edit_distance", # braucht ewig 
#             "network_mutual_information", 
#         "wasserstein_gromov",
#         "graph_kernel_networkx", 
#         "communicability_jsd", 
#         "frobenius", 
#         "jaccard", # one_minus_jaccard", 
         
        
#         # "f1", 
#         "multiplex_layer_similarity", # one_minus_multiplex_layer_similarity",
#         # "jaccard", 
#         # "communicability", 
#         "cosine_embedding", 
#         # "wasserstein_sinkhorn", 
#         "hungarian_alignment", 
#         # "graph_kernel", 
        
#         # "resistance_distance", 
        
#         # "graph_edit_distance", 
#         # "network_mutual_information", 
#         # "communicability_mse", 
        
        
# ]
        
n_methods = len(all_methods_sorted_for_legend)
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

# Calculate height to maintain roughly square subplots
height = 18 * (n_rows / n_cols) # * 0.9 # 0.9 is random factor for colorbar

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((18, height)), 
                            dpi=150,
                            sharex=True, 
                            sharey=True)

axes = axes.flatten()

for idx, mode in enumerate(all_methods_sorted_for_legend[::-1]):
    ax = axes[idx]
    
    if mode not in selected_methods:
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
            f'custom_cmap_{mode}', 
            [
                "black",
                bone_white
            ]
        )
    else: 
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
            f'custom_cmap_{mode}', 
            [
                "black",
                metric_colors[mode], 
                # "white", 
                bone_white
            #  
            #  color_neg1,
            #  halfblack, 
            #  bone_white, 
            ]
        )
    # cmap = default_cmaps["metric_purple_beige"]

    scatter = ax.imshow(matrices_of_metrics[mode], 
                        cmap=cmap, 
                       aspect='auto',
                    #    origin='lower',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # ax.set_xlim(-8, 3)
    # ax.set_ylim(-0.1, 1)
    
    # Set ticks
    ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
    ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
    
    # Add title for each subplot
    print(mode)
    title = method_names[mode] # f"{metric_col[0].split('subject')[0].replace('_', ' ').capitalize()}"
    ax.set_title(title, fontsize=8)
    
    # # Add colorbar for each subplot
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.tick_params(labelsize=6)
    
    # Only add labels to edge subplots
    if idx % n_cols == 0:  # Left column
        ax.set_ylabel(r"$\gamma$")
    if idx >= n_methods - n_cols:  # Bottom row
        ax.set_xlabel(r"$\eta$")

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Add one shared colorbar below all subplots
cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.01])  # [left, bottom, width, height]
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Darker: More similar. Lighter: Less similar")
cbar.ax.tick_params(labelsize=8)
plt.tight_layout()

# ====================================================================
# cell 42
# ====================================================================
# methods_for_gridplot = [
#         "energy",
#         "f1", # "one_minus_f1", 
#         "portrait", 
#         "delta_con", 
#         "spectral_distance",
#         "edit_distance", 
#             "graph_edit_distance", # braucht ewig 
#             "network_mutual_information", 
#         "wasserstein_gromov",
#         "graph_kernel_networkx", 
#         "communicability_jsd", 
#         "frobenius", 
#         "jaccard", # one_minus_jaccard", 
         
        
#         # "f1", 
#         "multiplex_layer_similarity", # one_minus_multiplex_layer_similarity",
#         # "jaccard", 
#         # "communicability", 
#         "cosine_embedding", 
#         # "wasserstein_sinkhorn", 
#         "hungarian_alignment", 
#         # "graph_kernel", 
        
#         # "resistance_distance", 
        
#         # "graph_edit_distance", 
#         # "network_mutual_information", 
#         # "communicability_mse", 
        
        
# ]

all_methods_sorted_for_legend = all_dist_measures
        
n_methods = len(all_methods_sorted_for_legend[::-1])
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

image_size_in_cm = 6
fontsize = 8
# Calculate height to maintain roughly square subplots
height = image_size_in_cm * (n_rows / n_cols) 

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((image_size_in_cm, height)), # 18, height)), 
                            dpi=150,
                            squeeze=True,
                            # sharex=True, 
                            # sharey=True
                            )

axes = axes.flatten()

for idx, mode in enumerate(all_methods_sorted_for_legend[::-1]):
    ax = axes[idx]

    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            metric_colors[mode], 
            "white", 
            # bone_white

        #  
        #  color_neg1,
        #  halfblack, 
        #  bone_white, 
         ]
    )
    # cmap = default_cmaps["metric_purple_beige"]

    scatter = ax.imshow(matrices_of_metrics[mode], 
                        cmap=cmap, 
                       aspect='auto',
                    #    origin='lower',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # ax.set_xlim(-8, 3)
    # ax.set_ylim(-0.1, 1)
    
    # Set ticks
    if (idx % n_cols == 0 and idx // n_cols == n_rows -1): 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == 0: 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([]) 
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == len(methods_for_gridplot) -1: 
        ax.set_yticks([])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
    else: 
        ax.set_yticks([])
        ax.set_xticks([])
    # Add title for each subplot
    print(mode)
    title = method_names[mode] # f"{metric_col[0].split('subject')[0].replace('_', ' ').capitalize()}"
    # ax.set_title(title, fontsize=8)
    
    # # Add colorbar for each subplot
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.tick_params(labelsize=6)
    
    # # Only add labels to edge subplots
    # if idx % n_cols == 0:  # Left column
    #     ax.set_ylabel(r"$\gamma$")
    # if idx >= n_methods - n_cols:  # Bottom row
    #     ax.set_xlabel(r"$\eta$")

# fig.supxlabel(r"$\eta$")
# fig.supylabel(r"$\gamma$")

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')


# Get min and max location of the corners of the subplots
x_min = np.min([ax.get_position().x0 for ax in axes]) + 0.025
x_max = np.max([ax.get_position().x1 for ax in axes]) + 2 * 0.025
y_min = np.min([ax.get_position().y0 for ax in axes])
y_max = np.max([ax.get_position().y1 for ax in axes])

# Add one shared colorbar below all subplots
cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            # metric_colors[mode], 
            "white", 
            # bone_white

        #  
        #  color_neg1,
        #  halfblack, 
        #  bone_white, 
         ])
values_for_cbar = np.linspace(0, 1, 256)
norm = plt.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.01])  # [left, bottom, width, height]
cbar_ax = fig.add_axes([x_min, 0, x_max - x_min, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Similarity") #  (Higher for darker colors)") # Darker: More similar. Lighter: Less similar")
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xticks([0, 10], labels=["High", "Low"])
plt.tight_layout()

plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce these values for tighter spacing

plt.savefig(output_path / "sixteen_subplots_colored.pdf")
print(output_path / "sixteen_subplots_colored.pdf")

plt.show()

# ====================================================================
# cell 43
# ====================================================================
all_dist_measures = all_dist_measures
        
n_methods = len(all_dist_measures[::-1])
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

image_size_in_cm = 6
fontsize = 8
height = image_size_in_cm * (n_rows / n_cols) 

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((image_size_in_cm, height)),
                            dpi=150,
                            squeeze=True,
                            constrained_layout=False
                            )

axes = axes.flatten()

for idx, mode in enumerate(all_dist_measures):  # same order as panel A
    ax = axes[idx]
    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            metric_colors[mode], 
            "white", 
         ]
    )

    scatter = ax.imshow(matrices_of_metrics[mode], 
                        cmap=cmap, 
                       aspect='auto',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # Minimum points: Scatter plot around the 100 lowest values
    flat_indices = np.argsort(matrices_of_metrics[mode].flatten())[:100]
    row_indices, col_indices = np.unravel_index(flat_indices, matrices_of_metrics[mode].shape)

    # Convert matrix indices to coordinate space
    eta_coords = np.linspace(df["eta"].min(), df["eta"].max(), matrices_of_metrics[mode].shape[1])
    gamma_coords = np.linspace(df["gamma"].min(), df["gamma"].max(), matrices_of_metrics[mode].shape[0])

    # Flip row indices to match imshow's top-to-bottom orientation
    row_indices = matrices_of_metrics[mode].shape[0] - 1 - row_indices

    ax.scatter(eta_coords[col_indices], gamma_coords[row_indices], 
            color=metric_colors[mode], 
            edgecolor='white', 
            linewidth=0.5,
            s=3,
            label='Lowest 100 values')
    
    # Set ticks
    if (idx % n_cols == 0 and idx // n_cols == n_rows -1): 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == 0: 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([]) 
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == len(all_dist_measures[::-1]) -1: 
        ax.set_yticks([])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
    else: 
        ax.set_yticks([])
        ax.set_xticks([])

    ax.vlines(x=0, ymin=df["gamma"].min(), ymax=df["gamma"].max(), 
              colors='black', linestyles='dashed', linewidth=0.4)

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Adjust spacing manually to leave room for colorbar at bottom
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.98, wspace=0.1, hspace=0.1)

# Get positions AFTER subplots_adjust
x_min = np.min([ax.get_position().x0 for ax in axes[:n_methods]])
x_max = np.max([ax.get_position().x1 for ax in axes[:n_methods]])

# Add one shared colorbar below all subplots
cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_colorbar', 
        ["black", "white"])
norm = plt.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Position colorbar with space for labels
cbar_ax = fig.add_axes([x_min, 
                        -0.05, # 0.03, 
                        x_max - x_min, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Similarity")
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xticks([0, 10], labels=["High", "Low"])

# Save FIRST, then show
plt.savefig(output_path / "sixteen_subplots_colored_with_100_mins.pdf", bbox_inches='tight', pad_inches=0.1)
print(output_path / "sixteen_subplots_colored_with_100_mins.pdf")
plt.show()

# ====================================================================
# cell 44
# ====================================================================
all_methods_sorted_for_legend = all_dist_measures
        
n_methods = len(all_methods_sorted_for_legend[::-1])
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

image_size_in_cm = 6
fontsize = 8
height = image_size_in_cm * (n_rows / n_cols) 

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((image_size_in_cm, height)), # 18, height)), 
                            dpi=150,
                            squeeze=True,
                            # sharex=True, 
                            # sharey=True
                            )


axes = axes.flatten()

for idx, mode in enumerate(all_methods_sorted_for_legend[::-1]):
    ax = axes[idx]

    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            metric_colors[mode], 
            "white", 
         ]
    )
    # cmap = default_cmaps["metric_purple_beige"]

    scatter = ax.imshow(matrices_of_metrics[mode], 
                        cmap=cmap, 
                       aspect='auto',
                    #    origin='lower',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # ax.set_xlim(-8, 3)
    # ax.set_ylim(-0.1, 1)
    
    # Minimum points: Scatter plot around the 100 lowest values
    flat_indices = np.argsort(matrices_of_metrics[mode].flatten())[:100]
    row_indices, col_indices = np.unravel_index(flat_indices, matrices_of_metrics[mode].shape)

    # Convert matrix indices to coordinate space
    eta_coords = np.linspace(df["eta"].min(), df["eta"].max(), matrices_of_metrics[mode].shape[1])
    gamma_coords = np.linspace(df["gamma"].min(), df["gamma"].max(), matrices_of_metrics[mode].shape[0])

    # Flip row indices to match imshow's top-to-bottom orientation
    row_indices = matrices_of_metrics[mode].shape[0] - 1 - row_indices

    ax.scatter(eta_coords[col_indices], gamma_coords[row_indices], 
            color=metric_colors[mode], 
            edgecolor='white', 
            linewidth=0.5,
            s=3,
            label='Lowest 100 values')
    
    
    # Set ticks
    if (idx % n_cols == 0 and idx // n_cols == n_rows -1): 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == 0: 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([]) 
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == len(methods_for_gridplot) -1: 
        ax.set_yticks([])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
    else: 
        ax.set_yticks([])
        ax.set_xticks([])
    # Add title for each subplot
    print(mode)
    title = method_names[mode] # f"{metric_col[0].split('subject')[0].replace('_', ' ').capitalize()}"
    # ax.set_title(title, fontsize=8)
    
    # # Add colorbar for each subplot
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.ax.tick_params(labelsize=6)
    
    # # Only add labels to edge subplots
    # if idx % n_cols == 0:  # Left column
    #     ax.set_ylabel(r"$\gamma$")
    # if idx >= n_methods - n_cols:  # Bottom row
    #     ax.set_xlabel(r"$\eta$")

    ax.vlines(x=0, ymin=df["gamma"].min(), ymax=df["gamma"].max(), colors='black', linestyles='dashed', linewidth=0.4)

# fig.supxlabel(r"$\eta$")
# fig.supylabel(r"$\gamma$")

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')


# Get min and max location of the corners of the subplots
x_min = np.min([ax.get_position().x0 for ax in axes]) + 0.025
x_max = np.max([ax.get_position().x1 for ax in axes]) + 2 * 0.025
y_min = np.min([ax.get_position().y0 for ax in axes])
y_max = np.max([ax.get_position().y1 for ax in axes])

# Add one shared colorbar below all subplots
cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            # metric_colors[mode], 
            "white", 
            # bone_white

        #  
        #  color_neg1,
        #  halfblack, 
        #  bone_white, 
         ])
values_for_cbar = np.linspace(0, 1, 256)
norm = plt.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.01])  # [left, bottom, width, height]
cbar_ax = fig.add_axes([x_min, 0, x_max - x_min, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Similarity") #  (Higher for darker colors)") # Darker: More similar. Lighter: Less similar")
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xticks([0, 10], labels=["High", "Low"])
plt.tight_layout()

plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce these values for tighter spacing

plt.savefig(output_path / "sixteen_subplots_colored.pdf")
print(output_path / "sixteen_subplots_colored.pdf")

plt.show()




# ====================================================================
# cell 45
# ====================================================================
all_methods_sorted_for_legend = all_dist_measures
        
n_methods = len(all_methods_sorted_for_legend[::-1])
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

image_size_in_cm = 6
fontsize = 8
height = image_size_in_cm * (n_rows / n_cols) 

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((image_size_in_cm, height)),
                            dpi=150,
                            squeeze=True,
                            constrained_layout=False
                            )

axes = axes.flatten()

for idx, mode in enumerate(all_methods_sorted_for_legend[::-1]):
    ax = axes[idx]
    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            metric_colors[mode], 
            "white", 
         ]
    )

    scatter = ax.imshow(matrices_of_metrics[mode], 
                        cmap=cmap, 
                       aspect='auto',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # Minimum points: Scatter plot around the 100 lowest values
    flat_indices = np.argsort(matrices_of_metrics[mode].flatten())[:100]
    row_indices, col_indices = np.unravel_index(flat_indices, matrices_of_metrics[mode].shape)

    # Convert matrix indices to coordinate space
    eta_coords = np.linspace(df["eta"].min(), df["eta"].max(), matrices_of_metrics[mode].shape[1])
    gamma_coords = np.linspace(df["gamma"].min(), df["gamma"].max(), matrices_of_metrics[mode].shape[0])

    # Flip row indices to match imshow's top-to-bottom orientation
    row_indices = matrices_of_metrics[mode].shape[0] - 1 - row_indices

    ax.scatter(eta_coords[col_indices], gamma_coords[row_indices], 
            color=metric_colors[mode], 
            edgecolor='white', 
            linewidth=0.5,
            s=3,
            label='Lowest 100 values')
    
    # Set ticks
    if (idx % n_cols == 0 and idx // n_cols == n_rows -1): 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == 0: 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([]) 
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == len(all_methods_sorted_for_legend[::-1]) -1: 
        ax.set_yticks([])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
    else: 
        ax.set_yticks([])
        ax.set_xticks([])

    ax.vlines(x=0, ymin=df["gamma"].min(), ymax=df["gamma"].max(), 
              colors='black', linestyles='dashed', linewidth=0.4)

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Adjust spacing manually to leave room for colorbar at bottom
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.98, wspace=0.1, hspace=0.1)

# Get positions AFTER subplots_adjust
x_min = np.min([ax.get_position().x0 for ax in axes[:n_methods]])
x_max = np.max([ax.get_position().x1 for ax in axes[:n_methods]])

# Add one shared colorbar below all subplots
cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_colorbar', 
        ["black", "white"])
norm = plt.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Position colorbar with space for labels
cbar_ax = fig.add_axes([x_min, 
                        -0.05, # 0.03, 
                        x_max - x_min, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Similarity")
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xticks([0, 10], labels=["High", "Low"])

# Save FIRST, then show
plt.savefig(output_path / "sixteen_subplots_colored_with_100_mins.pdf", bbox_inches='tight', pad_inches=0.1)
print(output_path / "sixteen_subplots_colored_with_100_mins.pdf")
plt.show()

# ====================================================================
# cell 46
# ====================================================================
all_methods_sorted_for_legend = all_dist_measures
        
n_methods = len(all_methods_sorted_for_legend[::-1])
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

image_size_in_cm = 6
fontsize = 8
# Calculate height to maintain roughly square subplots
height = image_size_in_cm * (n_rows / n_cols) 

# Use constrained_layout instead of tight_layout
fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((image_size_in_cm, height)),
                            dpi=150,
                            squeeze=True,
                            constrained_layout=False  # We'll handle layout manually
                            )

axes = axes.flatten()

for idx, mode in enumerate(all_methods_sorted_for_legend[::-1]):
    ax = axes[idx]
    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            metric_colors[mode], 
            "white", 
         ]
    )

    scatter = ax.imshow(matrices_of_metrics[mode], # np.log(matrices_of_metrics[mode]), 
                        cmap=cmap, 
                       aspect='auto',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # Set ticks
    if (idx % n_cols == 0 and idx // n_cols == n_rows -1): 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == 0: 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([]) 
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == len(all_methods_sorted_for_legend[::-1]) -1: 
        ax.set_yticks([])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
    else: 
        ax.set_yticks([])
        ax.set_xticks([])

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Adjust spacing manually to leave room for colorbar at bottom
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.98, wspace=0.1, hspace=0.1)

# Get positions AFTER subplots_adjust
x_min = np.min([ax.get_position().x0 for ax in axes[:n_methods]])
x_max = np.max([ax.get_position().x1 for ax in axes[:n_methods]])
y_min = np.min([ax.get_position().y0 for ax in axes[:n_methods]])

# Add colorbar in the reserved space at bottom
cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_colorbar', 
        ["black", "white"])
norm = plt.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Position colorbar below the plots
cbar_ax = fig.add_axes([x_min, -0.1, x_max - x_min, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Similarity")
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xticks([0, 10], labels=["High", "Low"])

# Save FIRST, then show
plt.savefig(output_path / "sixteen_subplots_colored.pdf", bbox_inches='tight')
print(output_path / "sixteen_subplots_colored.pdf")
plt.show()

# ====================================================================
# cell 47
# ====================================================================
fig = plt.figure(figsize=(9, 1))  # Adjust size as needed
ax = fig.add_subplot(111)

# sorted_data = results_df[metric].sort_values()

# Create legend handles
handles = [patches.Patch(color=metric_colors[name], label=method_names[name]) 
           for name in all_methods_sorted_for_legend[::-1]]

# Add legend
legend = ax.legend(handles=handles, loc='center', 
                   ncol=4, 
                #    fontsize=12, 
                   frameon=False)

# Hide the axes
ax.axis('off')

# plt.tight_layout()

plt.savefig(output_path / "method_legend_colored.pdf")
plt.show()
print(output_path / "method_legend_colored.pdf")

# ====================================================================
# cell 48
# ====================================================================
# ONLY COLORBAR + LABELS 

metric = "snr"
labels = {'snr': 'Signal-to-Noise Ratio\n(higher = better)'}

fig = plt.figure(figsize=viz.cm_to_inch((9,9)), dpi=150)

colors_for_legend = [metric_colors[m] for m in all_methods_sorted_for_legend]
names_for_legend = [method_names[m] for m in all_methods_sorted_for_legend]

# Add colorbar on right side (changed from 0.02 to 0.92)
cax = fig.add_axes([0.92, 0.25, 0.03, 0.65])
norm = plt.Normalize(vmin=0, vmax=len(all_methods_sorted_for_legend))
sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(colors_for_legend), norm=norm)
cb = plt.colorbar(sm, cax=cax)
cb.set_ticks(
    ticks=[i + 0.5 for i in range(len(names_for_legend))], 
    labels=names_for_legend)

cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_label_position('left')

# ====================================================================
# cell 49
# ====================================================================
# Create figure with dendrogram
fig = plt.figure(figsize=viz.cm_to_inch((6,6)))

cmap_for_corr = default_cmaps["db_bw_lr"]
thickness_bars = 0.03
distance_between_bar_and_corr_heatmap = 0.03

# Add heatmap - leave more room at bottom for colorbar
ax_heatmap = fig.add_axes([0.22, 0.15, 0.7, 0.75])
sns.heatmap(clustered_corr, annot=False, 
            cmap=cmap_for_corr, 
            square=True,
            cbar=False, 
            ax=ax_heatmap, 
            )

# Add spines
for spine in ax_heatmap.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)

# Get xticklabels, and replace them with their corresponding method names
method_names_arr = [method_names[method] for method in all_dist_measures]
method_colors_arr = [metric_colors[method] for method in all_dist_measures]
ax_heatmap.set_yticklabels([])
ax_heatmap.set_xticklabels([])

ax_heatmap.set_xlabel(None)
ax_heatmap.set_ylabel(None)

# Remove the small "ticks"
ax_heatmap.tick_params(top=False, right=False, bottom=False, left=False)

ax_heatmap.invert_xaxis()
ax_heatmap.invert_yaxis()

# Get positions AFTER setting up heatmap
heatmap_pos = ax_heatmap.get_position()
highest_pos = heatmap_pos.y1
lowest_pos = heatmap_pos.y0
leftest_pos = heatmap_pos.x0
rightest_pos = heatmap_pos.x1

# Add left colorbar (method colors)
colors_for_legend = [metric_colors[m] for m in all_dist_measures] 
cax = fig.add_axes([leftest_pos - thickness_bars - distance_between_bar_and_corr_heatmap, 
                    lowest_pos, 
                    thickness_bars, 
                    highest_pos - lowest_pos])
norm = plt.Normalize(vmin=0, vmax=len(all_dist_measures))
sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(colors_for_legend), norm=norm)
cb = plt.colorbar(sm, cax=cax)
cb.set_ticks(ticks=[], labels=[])
cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_label_position('left')

# Add bottom colorbar (correlation values) - position with enough space for labels
cbar_ax = fig.add_axes([leftest_pos, 
                        0.05,  # Position with space below for labels
                        rightest_pos - leftest_pos, 
                        thickness_bars])
norm = plt.Normalize(vmin=clustered_corr.values.min(), vmax=clustered_corr.values.max())
sm = plt.cm.ScalarMappable(cmap=cmap_for_corr, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Pearson Correlation') 
cbar.set_ticks([-1, 0, 1])
cbar.ax.tick_params(labelsize=8)

# Save with proper bounding box
plt.savefig(output_path / "correlation_heatmap_clustered.pdf", bbox_inches='tight', pad_inches=0.1)
print(output_path / "correlation_heatmap_clustered.pdf")
plt.show()

# The right one! See the flipped axes...

# ====================================================================
# cell 50
# ====================================================================
# Create figure with dendrogram
fig = plt.figure(figsize=viz.cm_to_inch((6,6))) # (16, 14)


cmap_for_corr = default_cmaps["db_bw_lr"] # black_white_map, # default_cmaps["db_bw_lr"], # 'RdBu_r'
thickness_bars = 0.03
distance_between_bar_and_corr_heatmap = 0.03

# Add heatmap
ax_heatmap = fig.add_axes([0.22, 0.1, 0.7, 0.8])
sns.heatmap(clustered_corr, annot=False, # True, fmt='.2f', annot_kws={'size': 8},
            cmap=cmap_for_corr, 
            # center=0, vmin=-1, vmax=1, 
            square=True,
            cbar=False, 
            # cbar_kws={'label': 'Pearson Correlation'},
            ax=ax_heatmap, 
            # linewidths=0.5
            )

# Add spines
for spine in ax_heatmap.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)

# Get xticklabels, and replace them with their corresponding method names
method_names_arr = [method_names[method] for method in all_dist_measures] # loaded_methods] 
method_colors_arr = [metric_colors[method] for method in all_dist_measures] # loaded_methods]
# .get(name, name) for name in clustered_corr.index]
ax_heatmap.set_yticklabels([]) # method_names_arr) # , rotation=45, ha='right')
ax_heatmap.set_xticklabels([])# False) # []) # method_names_arr) # , rotation=0)

# new_colors = [metric_colors.get(name) for name in methods_for_gridplot]
# print(corr_matrix.shape, new_colors)
ax_heatmap.set_xlabel(None)
ax_heatmap.set_ylabel(None)

# Get upper corner of heatmap for positioning dendrogram
heatmap_pos = ax_heatmap.get_position()


###############

# Get positions
highest_pos = ax_heatmap.get_position().y1
lowest_pos = ax_heatmap.get_position().y0

leftest_pos = ax_heatmap.get_position().x0
rightest_pos = ax_heatmap.get_position().x1

# Add colorbar on left side
#
# # cax = fig.add_axes([0.02, 0.25, 0.03, 0.65])
# cax = fig.add_axes([heatmap_pos.x0 - 0.05, heatmap_pos.y0, 0.03, heatmap_pos.height])

# norm = plt.Normalize(vmin=0, vmax=len(method_colors_arr))
# sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(method_colors_arr), norm=norm)
# cb = plt.colorbar(sm, cax=cax)
# cb.set_ticks([])
colors_for_legend = [metric_colors[m] for m in all_dist_measures] # loaded_methods] # method_colors_arrall_methods_sorted_for_legend]
#
cax = fig.add_axes([leftest_pos - thickness_bars - distance_between_bar_and_corr_heatmap, lowest_pos, thickness_bars, highest_pos - lowest_pos])
norm = plt.Normalize(vmin=0, vmax=len(all_methods_sorted_for_legend))
sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(colors_for_legend), norm=norm)
cb = plt.colorbar(sm, cax=cax)
cb.set_ticks(
    ticks=[], 
    # ticks=[i + 0.5 for i in range(len(names_for_legend))], 
    labels=[])# names_for_legend)

cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_label_position('left')


################


# remove the small "ticks"
ax_heatmap.tick_params(top=False, right=False, bottom=False, left=False)

# cbar
# Add cbar that is as high as the heatmap. So get the lowest and highest position of the heatmap
highest_pos = ax_heatmap.get_position().y1
lowest_pos = ax_heatmap.get_position().y0

leftest_pos = ax_heatmap.get_position().x0
rightest_pos = ax_heatmap.get_position().x1
buffer = 0.01
# cbar_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
# cbar_ax = fig.add_axes([rightest_pos + buffer, lowest_pos, 0.02, highest_pos - lowest_pos])
cbar_ax = fig.add_axes([leftest_pos, lowest_pos - thickness_bars - distance_between_bar_and_corr_heatmap, rightest_pos - leftest_pos, thickness_bars]) # highest_pos - lowest_pos])
norm = plt.Normalize(vmin=clustered_corr.values.min(), vmax=clustered_corr.values.max())
sm = plt.cm.ScalarMappable(cmap=cmap_for_corr, norm=norm) # default_cmaps["db_bw_lr"], norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Pearson Correlation') 

# only add ticks for -1, 0, 1
cbar.set_ticks([-1, 0, 1])
cbar.ax.tick_params(labelsize=8)

plt.savefig(output_path / "correlation_heatmap_clustered_with_dendrogram_and_colorbar.pdf")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
# print(f"Saved clustered correlation heatmap to {save_path}")              



# # Clustered correlation heatmap
# heatmap_path = output_path / 'correlation_heatmap_clustered.png'
# clustered_corr = plot_clustered_correlation_heatmap(corr_matrix, heatmap_path)

# # GOOD: DONE HERE


# ====================================================================
# cell 51
# ====================================================================
# Create figure with dendrogram
fig = plt.figure(figsize=viz.cm_to_inch((6,6)))

cmap_for_corr = default_cmaps["db_bw_lr"]
thickness_bars = 0.03
distance_between_bar_and_corr_heatmap = 0.03

# Add heatmap - leave more room at bottom for colorbar
ax_heatmap = fig.add_axes([0.22, 0.15, 0.7, 0.75])
sns.heatmap(clustered_corr, annot=False, 
            cmap=cmap_for_corr, 
            square=True,
            cbar=False, 
            ax=ax_heatmap, 
            )

# Add spines
for spine in ax_heatmap.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)

# Get xticklabels, and replace them with their corresponding method names
method_names_arr = [method_names[method] for method in all_dist_measures]
method_colors_arr = [metric_colors[method] for method in all_dist_measures]

ax_heatmap.set_xticklabels([])

ax_heatmap.set_xlabel(None)
ax_heatmap.set_ylabel(None)

# Remove the small "ticks"
ax_heatmap.tick_params(top=False, right=False, bottom=False, left=False)

# Get positions AFTER setting up heatmap
heatmap_pos = ax_heatmap.get_position()
highest_pos = heatmap_pos.y1
lowest_pos = heatmap_pos.y0
leftest_pos = heatmap_pos.x0
rightest_pos = heatmap_pos.x1

# Add left colorbar (method colors)
colors_for_legend = [metric_colors[m] for m in all_dist_measures] 
cax = fig.add_axes([leftest_pos - thickness_bars - distance_between_bar_and_corr_heatmap, 
                    lowest_pos, 
                    thickness_bars, 
                    highest_pos - lowest_pos])
norm = plt.Normalize(vmin=0, vmax=len(all_methods_sorted_for_legend))
sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(colors_for_legend), norm=norm)
cb = plt.colorbar(sm, cax=cax)
cb.set_ticks(ticks=[], labels=[])
cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_label_position('left')

# Add bottom colorbar (correlation values) - position with enough space for labels
cbar_ax = fig.add_axes([leftest_pos, 
                        -0.1, # 0.05,  # Position with space below for labels
                        rightest_pos - leftest_pos, 
                        thickness_bars])
norm = plt.Normalize(vmin=clustered_corr.values.min(), vmax=clustered_corr.values.max())
sm = plt.cm.ScalarMappable(cmap=cmap_for_corr, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Pearson Correlation') 
cbar.set_ticks([-1, 0, 1])
cbar.ax.tick_params(labelsize=8)

# Save with proper bounding box
plt.savefig(output_path / "correlation_heatmap_clustered.pdf", bbox_inches='tight', pad_inches=0.1)
print(output_path / "correlation_heatmap_clustered.pdf")
plt.show()

# ====================================================================
# cell 52
# ====================================================================
def get_top_networks_distribution(dataset_name, experiment_name, mode='portrait', top_n=20):

    # Paths
    base_path = str(_RD / "output" / "gnm" / dataset_name / experiment_name)
    summary_path = f"{base_path}/summary_indiv_{mode}_for_exp_{experiment_name}.csv"
    networks_path = f"{base_path}/generated_networks"
    
    # Load summary data
    df = pd.read_csv(summary_path)
    # print(f"Loaded summary with columns: {list(df.columns)}")
    
    # Identify metric column (excluding metadata columns)
    metadata_cols = ["eta", "gamma", "network_index", "filename", "id"]
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    
    if not metric_cols:
        raise ValueError(f"No metric column found in {df.columns}")
    
    metric_col = metric_cols[0]
    print(f"Using metric column: '{metric_col}'")
    
    # Select top N networks (lowest values = best); similarities flipped first
    df = df.assign(_oriented=to_distance(df[metric_col].values, mode))
    df_sorted = df.nsmallest(top_n, "_oriented")
    # print(f"\nSelected top {top_n} networks with {metric_col} range: "
    #       f"[{df_sorted[metric_col].min():.6f}, {df_sorted[metric_col].max():.6f}]")
        
    # Load distance matrix (schaefer) 
    distance_matrix_file = DIST_MATRIX_PATH
    distance_matrix = np.load(distance_matrix_file)
    
    
    # Load networks and calculate degree distributions
    all_degrees = []
    all_distances = []
    
    for idx, row in df_sorted.iterrows():
        # Construct filename
        network_file = Path(networks_path) / row['filename']
        
        # Load network
        network = np.load(network_file)
        
        # Extract adjacency matrix (assuming shape is (1, 100, 100))
        if network.shape[0] == 1:
            adj_matrix = network[0]
        else:
            adj_matrix = network
        
        # Calculate degree for each node
        degrees = np.sum(adj_matrix, axis=1)
        all_degrees.extend(degrees)
        
        # get all distances, remove zeros 
        distances_for_this_network = (distance_matrix*network).flatten()
        all_distances.extend(distances_for_this_network[distances_for_this_network > 0])
        
    # Convert to numpy array
    all_degrees = np.array(all_degrees)
    all_distances = np.array(all_distances)
    # print(f"\nCollected {len(all_degrees)} node degrees from {len(df_sorted)} networks")
    # print(f"Degree statistics: mean={all_degrees.mean():.2f}, "
    #       f"std={all_degrees.std():.2f}, "
    #       f"min={all_degrees.min()}, max={all_degrees.max()}")   

    # Get distance matrix values (excluding zeros on diagonal)
    # all_distances = distance_matrix[distance_matrix > 0]
    
    return all_degrees, all_distances

distributions_degree = {}
distributions_distances = {}
for mode in all_dist_measures: 
    distributions_degree[mode], distributions_distances[mode] = get_top_networks_distribution(dataset_name, experiment_name, mode=mode, top_n=20)


# ====================================================================
# cell 53
# ====================================================================
# Human distributions 
human_connectomes = np.load(INDIVIDUALS_PATH)
distance_matrix = np.load(DIST_MATRIX_PATH)

all_degrees = []
all_distances = []

for A in human_connectomes: 
    # Calculate degree for each node
    degrees = np.sum(A, axis=1)
    all_degrees.extend(degrees)
    
    # get all distances, remove zeros 
    distances_for_this_network = (distance_matrix*A).flatten()
    all_distances.extend(distances_for_this_network[distances_for_this_network > 0])

# Convert to numpy array
human_degrees = np.array(all_degrees)
human_distances = np.array(all_distances)

distributions_distances['human'] = human_distances
distributions_degree['human'] = human_degrees

human_distances

# ====================================================================
# cell 54
# ====================================================================
import scipy.ndimage as ndimage

# plot degree distributions (smoothed) 
fig, ax = plt.subplots(figsize=viz.cm_to_inch((6, 6)), dpi=200)

max_value_for_line = 0 

for mode, distances in distributions_distances.items():
    
    if mode not in selected_methods and not mode == 'human': 
        continue
    
    # Create histogram to get counts
    number_bins = 50
    counts, bins = np.histogram(distances, bins=number_bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Add padding to bin_centers (linear expansion)
    bin_centers = np.array([bins[0] - (bins[1]-bins[0])*i for i in range(20,0,-1)] + list(bin_centers) + 
                           [bins[-1] + (bins[1]-bins[0])*i for i in range(1,21)])

    # padded_counts = []
    # for count in counts: 
    count = np.array([0]*20 + list(counts) + [0]*20)
    count = ndimage.gaussian_filter1d(count, sigma=10) 
    #     padded_counts.append(count)

    if count.max() > max_value_for_line:
        max_value_for_line = count.max()
        ax.vlines(x=90, 
                  ymin=0, ymax=max_value_for_line, 
                  color="gray", 
                  linestyle='--', alpha=0.5)

    # # Smooth the histogram with Gaussian filter
    # # smoothed_counts = ndimage.gaussian_filter1d(padded_counts, sigma=10) # 2
    
        # Plot smoothed curve
    ax.plot(bin_centers[20:-20], 
            count[20:-20],
            label=method_names[mode] if mode != 'human' else 'Human Connectomes',
            color=metric_colors[mode] if mode != 'human' else 'black',
            linewidth=2 if mode == "human" else 1
            )

plt.xlabel('Distance [mm]')
plt.ylabel("")
# plt.legend(fontsize=8, bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout()
plt.show()

# ====================================================================
# cell 55
# ====================================================================
import scipy.ndimage as ndimage

# plot degree distributions (smoothed) 
fig, ax = plt.subplots(figsize=viz.cm_to_inch((6, 6)), dpi=200)

for mode, degrees in distributions_degree.items():
    
    if mode not in selected_methods and not mode == 'human': 
        continue
    
    # Create bins that always have steps of one and start quantity 0
    max_degree = 100
    bins = np.arange(0, max_degree + 2, step=1) - 0.5  # bins of width 1 centered on integers
    
    # Create histogram to get counts
    counts, bins = np.histogram(degrees, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    count = ndimage.gaussian_filter1d(counts, sigma=2) 
    
    # Plot smoothed curve
    ax.plot(bin_centers, 
            count,
            label=method_names[mode] if mode != 'human' else 'Human Connectomes',
            color=metric_colors[mode] if mode != 'human' else 'black',
            linewidth=1 if mode != 'human' else 2
            )

plt.xlim(0, 40)
    

plt.xlabel('Degree')
plt.ylabel("")
# plt.legend()
plt.tight_layout()
plt.show()

# ====================================================================
# cell 56
# ====================================================================
from scipy.stats import wasserstein_distance, ks_2samp

# Calculate fit metrics for each method
fit_metrics = []

for method in selected_methods:
    # Get method distributions
    method_degrees = distributions_degree[method]
    method_distances = distributions_distances[method]
    
    # Get human distributions
    human_deg = distributions_degree['human']
    human_dist = distributions_distances['human']
    
    # Calculate Wasserstein distance (or you can use KS statistic)
    degree_fit = wasserstein_distance(method_degrees, human_deg)
    distance_fit = wasserstein_distance(method_distances, human_dist)
    
    fit_metrics.append({
        'method': method,
        'degree_fit': degree_fit,
        'distance_fit': distance_fit,
        'color': metric_colors[method]
    })

fit_df = pd.DataFrame(fit_metrics)

# Create scatter plot
fig, ax = plt.subplots(figsize=viz.cm_to_inch((8, 8)), dpi=150)

# Normalize the values of both axes for better visualization
max_degree_fit = fit_df['degree_fit'].max()
max_distance_fit = fit_df['distance_fit'].max()

for idx, row in fit_df.iterrows():
    ax.scatter(row['degree_fit'] / max_degree_fit, 
               row['distance_fit'] / max_distance_fit, 
               color=row['color'], s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
    
ax.set_xlabel('Degree Distribution Fit\n(Wasserstein Distance from Human)')
ax.set_ylabel('Distance Distribution Fit\n(Wasserstein Distance from Human)')
ax.set_title('Distribution Fit Quality: Degree vs Distance')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add diagonal line for reference
max_val = max(fit_df['degree_fit'].max(), fit_df['distance_fit'].max())
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

ax.set_ylim(0, 1.2)
ax.set_xlim(0, 1.2)

plt.tight_layout()
plt.savefig(output_path / "distribution_fit_comparison.pdf")
print(output_path / "distribution_fit_comparison.pdf")
plt.show()

# ====================================================================
# cell 57
# ====================================================================
from scipy.stats import ks_2samp

# Calculate fit metrics for each method using KS statistic
fit_metrics = []

for method in selected_methods:
    # Get method distributions
    method_degrees = distributions_degree[method]
    method_distances = distributions_distances[method]
    
    # Get human distributions
    human_deg = distributions_degree['human']
    human_dist = distributions_distances['human']
    
    # Calculate KS statistic (lower = better fit)
    degree_fit = ks_2samp(method_degrees, human_deg).statistic
    distance_fit = ks_2samp(method_distances, human_dist).statistic
    
    fit_metrics.append({
        'method': method,
        'degree_fit': degree_fit,
        'distance_fit': distance_fit,
        'color': metric_colors[method]
    })

fit_df = pd.DataFrame(fit_metrics)

# Create scatter plot
fig, ax = plt.subplots(figsize=viz.cm_to_inch((8, 8)), dpi=150)

for idx, row in fit_df.iterrows():
    ax.scatter(row['degree_fit'], row['distance_fit'], 
               color=row['color'], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    
ax.set_xlabel('Degree Distribution Fit\n(KS Statistic from Human)')
ax.set_ylabel('Distance Distribution Fit\n(KS Statistic from Human)')
ax.set_title('Distribution Fit Quality: Degree vs Distance')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add diagonal line for reference
max_val = max(fit_df['degree_fit'].max(), fit_df['distance_fit'].max())
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig(output_path / "distribution_fit_comparison.pdf")
print(output_path / "distribution_fit_comparison.pdf")
plt.show()

# ====================================================================
# cell 58
# ====================================================================
from scipy.stats import ks_2samp, entropy

# Calculate fit metrics for each method
fit_metrics = []

for method in selected_methods:
    # Get method distributions
    method_degrees = distributions_degree[method]
    method_distances = distributions_distances[method]
    
    # Get human distributions
    human_deg = distributions_degree['human']
    human_dist = distributions_distances['human']
    
    # Option 1: KS Statistic (recommended)
    degree_fit_ks = ks_2samp(method_degrees, human_deg).statistic
    distance_fit_ks = ks_2samp(method_distances, human_dist).statistic
    
    # Option 2: KL Divergence (requires binning)
    # For degrees (0-100)
    bins_deg = np.arange(0, 101, 1)
    hist_method_deg, _ = np.histogram(method_degrees, bins=bins_deg, density=True)
    hist_human_deg, _ = np.histogram(human_deg, bins=bins_deg, density=True)
    # Add small epsilon to avoid log(0)
    hist_method_deg = hist_method_deg + 1e-10
    hist_human_deg = hist_human_deg + 1e-10
    # Normalize to probabilities
    hist_method_deg = hist_method_deg / hist_method_deg.sum()
    hist_human_deg = hist_human_deg / hist_human_deg.sum()
    degree_fit_kl = entropy(hist_human_deg, hist_method_deg)
    
    # For distances
    bins_dist = np.linspace(0, human_dist.max(), 50)
    hist_method_dist, _ = np.histogram(method_distances, bins=bins_dist, density=True)
    hist_human_dist, _ = np.histogram(human_dist, bins=bins_dist, density=True)
    hist_method_dist = hist_method_dist + 1e-10
    hist_human_dist = hist_human_dist + 1e-10
    hist_method_dist = hist_method_dist / hist_method_dist.sum()
    hist_human_dist = hist_human_dist / hist_human_dist.sum()
    distance_fit_kl = entropy(hist_human_dist, hist_method_dist)
    
    fit_metrics.append({
        'method': method,
        'degree_fit_ks': degree_fit_ks,
        'distance_fit_ks': distance_fit_ks,
        'degree_fit_kl': degree_fit_kl,
        'distance_fit_kl': distance_fit_kl,
        'color': metric_colors[method]
    })

fit_df = pd.DataFrame(fit_metrics)

# Create scatter plots for both metrics
fig, axes = plt.subplots(1, 2, figsize=viz.cm_to_inch((16, 8)), dpi=150)

# KS Statistic plot
for idx, row in fit_df.iterrows():
    axes[0].scatter(row['degree_fit_ks'], row['distance_fit_ks'], 
                   color=row['color'], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    
axes[0].set_xlabel('Degree Distribution Fit\n(KS Statistic)')
axes[0].set_ylabel('Distance Distribution Fit\n(KS Statistic)')
axes[0].set_title('KS Statistic (Lower = Better)')
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
max_val_ks = max(fit_df['degree_fit_ks'].max(), fit_df['distance_fit_ks'].max())
axes[0].plot([0, max_val_ks], [0, max_val_ks], 'k--', alpha=0.3, linewidth=1)

# KL Divergence plot
for idx, row in fit_df.iterrows():
    axes[1].scatter(row['degree_fit_kl'], row['distance_fit_kl'], 
                   color=row['color'], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    
axes[1].set_xlabel('Degree Distribution Fit\n(KL Divergence)')
axes[1].set_ylabel('Distance Distribution Fit\n(KL Divergence)')
axes[1].set_title('KL Divergence (Lower = Better)')
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
max_val_kl = max(fit_df['degree_fit_kl'].max(), fit_df['distance_fit_kl'].max())
axes[1].plot([0, max_val_kl], [0, max_val_kl], 'k--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig(output_path / "distribution_fit_comparison_ks_vs_kl.pdf")
print(output_path / "distribution_fit_comparison_ks_vs_kl.pdf")
plt.show()

# ====================================================================
# cell 59
# ====================================================================
from scipy.stats import wasserstein_distance

# Calculate fit metrics
fit_metrics = []

for method in selected_methods:
    method_degrees = distributions_degree[method]
    method_distances = distributions_distances[method]
    human_deg = distributions_degree['human']
    human_dist = distributions_distances['human']
    
    degree_fit_wd = wasserstein_distance(method_degrees, human_deg)
    distance_fit_wd = wasserstein_distance(method_distances, human_dist)
    
    fit_metrics.append({
        'method': method,
        'degree_fit': degree_fit_wd,
        'distance_fit': distance_fit_wd,
        'color': metric_colors[method]
    })

fit_df = pd.DataFrame(fit_metrics)

# Create improved visualization
fig, axes = plt.subplots(1, 3, figsize=viz.cm_to_inch((24, 8)), dpi=150)

# Plot 1: Original with better zoom on x-axis
ax = axes[0]
for idx, row in fit_df.iterrows():
    ax.scatter(row['degree_fit'], row['distance_fit'], 
               color=row['color'], s=150, alpha=0.8, 
               edgecolors='black', linewidth=0.5)
    
ax.set_xlabel('Degree Distribution Fit\n(Wasserstein Distance)')
ax.set_ylabel('Distance Distribution Fit\n(Wasserstein Distance)')
ax.set_title('Distribution Fit Quality')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Zoom in on x-axis to show degree differences better
x_min, x_max = fit_df['degree_fit'].min(), fit_df['degree_fit'].max()
x_margin = (x_max - x_min) * 0.1
ax.set_xlim(x_min - x_margin, x_max + x_margin)

# Plot 2: Normalized to [0,1] for each axis
ax = axes[1]
fit_df['degree_fit_norm'] = (fit_df['degree_fit'] - fit_df['degree_fit'].min()) / \
                             (fit_df['degree_fit'].max() - fit_df['degree_fit'].min())
fit_df['distance_fit_norm'] = (fit_df['distance_fit'] - fit_df['distance_fit'].min()) / \
                               (fit_df['distance_fit'].max() - fit_df['distance_fit'].min())

for idx, row in fit_df.iterrows():
    ax.scatter(row['degree_fit_norm'], row['distance_fit_norm'], 
               color=row['color'], s=150, alpha=0.8, 
               edgecolors='black', linewidth=0.5)
    
ax.set_xlabel('Degree Distribution Fit\n(Normalized, 0=Best, 1=Worst)')
ax.set_ylabel('Distance Distribution Fit\n(Normalized, 0=Best, 1=Worst)')
ax.set_title('Normalized Fit Quality')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Plot 3: Combined score (Euclidean distance from origin)
ax = axes[2]
fit_df['combined_score'] = np.sqrt(fit_df['degree_fit_norm']**2 + 
                                    fit_df['distance_fit_norm']**2)
fit_df_sorted = fit_df.sort_values('combined_score')

colors_sorted = [row['color'] for _, row in fit_df_sorted.iterrows()]
methods_sorted = [method_names[m] for m in fit_df_sorted['method']]

ax.barh(range(len(fit_df_sorted)), fit_df_sorted['combined_score'], 
        color=colors_sorted, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(fit_df_sorted)))
ax.set_yticklabels(methods_sorted, fontsize=8)
ax.set_xlabel('Combined Fit Score\n(Lower = Better)')
ax.set_title('Overall Distribution Fit Quality')
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(output_path / "distribution_fit_comparison_improved.pdf")
print(output_path / "distribution_fit_comparison_improved.pdf")
plt.show()

# Print ranking
print("\nRanking by combined fit quality (lower = better):")
for idx, (_, row) in enumerate(fit_df_sorted.iterrows(), 1):
    print(f"{idx:2d}. {method_names[row['method']]:30s} "
          f"(Degree: {row['degree_fit']:.2f}, Distance: {row['distance_fit']:.2f}, "
          f"Combined: {row['combined_score']:.3f})")

# ====================================================================
# cell 60
# ====================================================================
metric = "snr"
labels = {'snr': 'Signal-to-Noise Ratio'} # \n(higher = better)'}

fig = plt.figure(figsize=viz.cm_to_inch((9, 8)), dpi=150)
ax = fig.add_axes([0.15, 0.25, 0.7, 0.65])

sorted_data = results_df[metric].sort_values()
colors = [metric_colors.get(name) for name in sorted_data.index]

bars = ax.barh(range(len(sorted_data)), sorted_data.values, color=colors)
ax.set_yticks([])  # Remove y-axis ticks
ax.set_xlabel(labels.get(metric, metric))
ax.set_title("SNR")
ax.grid(axis='x', alpha=0.3)

# Add colorbar on left side
cax = fig.add_axes([0.02, 0.25, 0.03, 0.65])
norm = plt.Normalize(vmin=0, vmax=len(sorted_data)-1)
sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(colors), norm=norm)
cb = plt.colorbar(sm, cax=cax)
cb.set_ticks([])

# Add legend below - ordered by appearance in barplot (reversed since bars go bottom to top)
handles = [patches.Patch(color=metric_colors[name], label=name) 
            for name in reversed(sorted_data.index) if metric_colors]
legend = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                    ncol=5, fontsize=12, 
                    frameon=False)

# ====================================================================
# cell 61
# ====================================================================
metric = "snr"
labels = {'snr': 'Signal-to-Noise Ratio'} # \n(higher = better)'}

fig = plt.figure(figsize=(10, 2))  # Adjust size as needed
ax = fig.add_subplot(111)

sorted_data = results_df[metric].sort_values()

# Create legend handles
handles = [patches.Patch(color=metric_colors[name], label=method_names[name]) 
           for name in reversed(sorted_data.index) if metric_colors]

# Add legend
legend = ax.legend(handles=handles, loc='center', 
                   ncol=5, fontsize=12, 
                   frameon=False)

# Hide the axes
ax.axis('off')

plt.tight_layout()
plt.show()

# ====================================================================
# cell 63
# ====================================================================
from typing import Dict, List

# Base directory for your chaos analysis results
base_dir = "output/gnm/lexis_data/chaos_analysis"
output_dir = f"{base_dir}/plots"
Path(output_dir).mkdir(parents=True, exist_ok=True)


# degeneration_x_label_descriptor = "Degeneration Level\n(Consensus to Chaos)" # 'Step (0=Consensus -> Max=Chaos)'
# degeneration_x_label_descriptor = "Degeneration: Consensus to Chaos" # 'Step (0=Consensus -> Max=Chaos)'
degeneration_x_label_descriptor = "Increasing Degeneration" # : Consensus to Chaos" # 'Step (0=Consensus -> Max=Chaos)'


# ====================================================================
# cell 64
# ====================================================================
def plot_multiple_metrics_comparison(
    csv_paths: Dict[str, str],
    output_path: str = None,
    figsize_cm: tuple = (20, 12),
    normalize: bool = True
):
    """
    Compare multiple metrics on the same plot.
    """
    fig, ax = plt.subplots(figsize=viz.cm_to_inch(figsize_cm))
    
    for idx, (metric_name, csv_path) in enumerate(csv_paths.items()):
        # Load data
        df = pd.read_csv(csv_path)
        
        # Calculate mean trajectory
        df_summary = df.groupby('step')['metric_value'].agg(['mean', 'std']).reset_index()
        
        # Normalize if requested
        if normalize:
            min_val = df_summary['mean'].min()
            max_val = df_summary['mean'].max()
            if max_val > min_val:  # Avoid division by zero
                df_summary['mean_norm'] = (df_summary['mean'] - min_val) / (max_val - min_val)
                df_summary['std_norm'] = df_summary['std'] / (max_val - min_val)
                y_col = 'mean_norm'
                std_col = 'std_norm'
            else:
                y_col = 'mean'
                std_col = 'std'
        else:
            y_col = 'mean'
            std_col = 'std'
        
        # Plot
        color = metric_colors[metric_name]
        ax.plot(
            df_summary['step'],
            df_summary[y_col],
            color=color,
            label=method_names[metric_name],
            alpha=0.9
        )

        print(method_names[metric_name])
        # Optional: add std band
        ax.fill_between(
            df_summary['step'],
            df_summary[y_col] - df_summary[std_col],
            df_summary[y_col] + df_summary[std_col],
            alpha=0.15,
            color=color
        )
    
    # Formatting
    ax.set_xlabel(degeneration_x_label_descriptor)
    if normalize:
        ax.set_ylabel('Normalized Metric Distance') # , fontsize=11)
    else:
        ax.set_ylabel('Metric Distance')

    # ax.set_title('Comparison of Metrics During Network Degradation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add diagonal line for reference
    max_x = df_summary['step'].max()
    ax.plot([0, max_x], [0, 1], 'k--', alpha=0.3, linewidth=1)

    # plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")
    else:
        plt.show()
    
    
    # plt.close()
    plt.show()
    
    return fig, ax



# Plot comparison of all metrics
print("\nPlotting comparison...")
csv_paths = {}
for metric in selected_methods: 
    csv_path = f"{base_dir}/chaos_analysis_{metric}.csv"
    if Path(csv_path).exists():
        csv_paths[metric] = csv_path

if csv_paths:
    plot_multiple_metrics_comparison(
        csv_paths=csv_paths,
        output_path=f"{output_dir}/chaos_comparison_all_metrics.pdf",
        figsize_cm=(10,6), # 6),
        normalize=True
    )
    



# ====================================================================
# cell 65
# ====================================================================
# Plotting comparison of top five metrics only

selected_methods_top_five = [] 
csv_paths = {}
for metric in selected_methods_top_five: 
    csv_path = f"{base_dir}/chaos_analysis_{metric}.csv"
    if Path(csv_path).exists():
        csv_paths[metric] = csv_path

if csv_paths:
    plot_multiple_metrics_comparison(
        csv_paths=csv_paths,
        output_path=f"{output_dir}/chaos_comparison_all_metrics_only_top_five.pdf",
        figsize_cm=(10,6), # 6),
        normalize=True
    )
    



# ====================================================================
# cell 68
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration"
output_path = Path(base_path) / "method_evaluation"
output_path.mkdir(exist_ok=True)

# all_methods = all_methods # [
#     "portrait", "energy", "spectral_distance", "f1", "communicability",
#     "graph_kernel", "multiplex_layer_similarity", "cosine_embedding",
#     "graph_edit_distance", "network_mutual_information", "hungarian_alignment",
#     "graph_kernel_networkx", "delta_con", "resistance_distance",
#     "wasserstein_gromov", "wasserstein_sinkhorn"
# ]

def load_all_methods():
    """Load and combine all method data."""
    all_data_list = []
    loaded_methods = []
    
    for method in all_dist_measures:
        path = f"{base_path}/summary_indiv_{method}_for_exp_{experiment_name}.csv"
        try:
            df = pd.read_csv(path)
            metric_cols = [col for col in df.columns 
                          if col not in ["eta", "gamma", "network_index", "filename", "id"]]
            if metric_cols:
                df['metric_value'] = to_distance(df[metric_cols[0]].values, method)
                df['method'] = method
                all_data_list.append(df[['eta', 'gamma', 'id', 'metric_value', 'method']])
                loaded_methods.append(method)
                print(f"  ✓ Loaded {method}")
        except FileNotFoundError:
            print(f"  ✗ File not found for {method}")
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    # Create wide format
    wide_data = all_data.pivot_table(
        index=['eta', 'gamma', 'id'],
        columns='method',
        values='metric_value'
    ).reset_index()
    
    return wide_data, loaded_methods


def normalize_methods(data, methods):
    """Normalize each method to [0, 1] range."""
    normalized = data.copy()
    for method in methods:
        min_val = data[method].min()
        max_val = data[method].max()
        if max_val > min_val:
            normalized[method] = (data[method] - min_val) / (max_val - min_val)
    return normalized


# ============================================================================
# STRATEGY 1: CONSENSUS-BASED APPROACHES
# ============================================================================

def consensus_approaches(data, methods):
    """Calculate different consensus measures."""
    print("\n" + "="*70)
    print("STRATEGY 1: CONSENSUS-BASED GROUND TRUTH")
    print("="*70)
    
    results = {}
    
    # Normalize first
    norm_data = normalize_methods(data, methods)
    
    # 1. Mean consensus (ensemble average)
    norm_data['consensus_mean'] = norm_data[methods].mean(axis=1)
    
    # 2. Median consensus (robust to outliers)
    norm_data['consensus_median'] = norm_data[methods].median(axis=1)
    
    # 3. Trimmed mean (remove top/bottom 10%)
    norm_data['consensus_trimmed'] = norm_data[methods].apply(
        lambda x: stats.trim_mean(x.dropna(), 0.1), axis=1
    )
    
    # Calculate correlations with each consensus
    for consensus_type in ['mean', 'median', 'trimmed']:
        col = f'consensus_{consensus_type}'
        print(f"\nCorrelations with {consensus_type.upper()} consensus:")
        corrs = {}
        for method in methods:
            corr = norm_data[method].corr(norm_data[col])
            corrs[method] = corr
            print(f"  {method:30s}: {corr:.4f}")
        results[consensus_type] = corrs
    
    # Save
    consensus_df = pd.DataFrame(results).T
    consensus_df.to_csv(output_path / 'consensus_correlations.csv')
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, consensus_type in enumerate(['mean', 'median', 'trimmed']):
        corrs = results[consensus_type]
        sorted_methods = sorted(corrs.keys(), key=lambda x: corrs[x], reverse=True)
        values = [corrs[m] for m in sorted_methods]
        
        axes[idx].barh(range(len(sorted_methods)), values, color='#6366f1')
        axes[idx].set_yticks(range(len(sorted_methods)))
        axes[idx].set_yticklabels([m.replace('_', ' ').title() for m in sorted_methods], fontsize=8)
        axes[idx].set_xlabel('Correlation', fontsize=10)
        axes[idx].set_title(f'{consensus_type.title()} Consensus') # , fontsize=12, fontweight='bold')
        axes[idx].axvline(x=np.mean(values), color='red', linestyle='--', alpha=0.5, label='Mean')
        axes[idx].grid(axis='x', alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'consensus_correlations.png', dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()
    
    return norm_data, results


# ============================================================================
# STRATEGY 2: INTER-METHOD AGREEMENT
# ============================================================================

def inter_method_agreement(data, methods):
    """Calculate how well each method agrees with all others."""
    print("\n" + "="*70)
    print("STRATEGY 2: INTER-METHOD AGREEMENT")
    print("="*70)
    
    norm_data = normalize_methods(data, methods)
    
    # Calculate average correlation with all other methods
    agreement_scores = {}
    for method1 in methods:
        corrs = []
        for method2 in methods:
            if method1 != method2:
                corr = norm_data[method1].corr(norm_data[method2])
                if not np.isnan(corr):
                    corrs.append(corr)
        agreement_scores[method1] = np.mean(corrs) if corrs else 0
    
    print("\nAverage agreement with other methods:")
    for method, score in sorted(agreement_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:30s}: {score:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_methods = sorted(agreement_scores.keys(), key=lambda x: agreement_scores[x], reverse=True)
    values = [agreement_scores[m] for m in sorted_methods]
    
    colors = ['#10b981' if v > np.mean(values) else '#6366f1' for v in values]
    ax.barh(range(len(sorted_methods)), values, color=colors)
    ax.set_yticks(range(len(sorted_methods)))
    ax.set_yticklabels([m.replace('_', ' ').title() for m in sorted_methods]) # , fontsize=9)
    ax.set_xlabel('Average Correlation with Other Methods') # , fontsize=11)
    ax.set_title('Inter-Method Agreement Scores\n(Higher = More consensus with other methods)') 
                #  fontsize=13, fontweight='bold', pad=15)
    ax.axvline(x=np.mean(values), color='red', linestyle='--', alpha=0.5, label=f'Mean = {np.mean(values):.3f}')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'inter_method_agreement.png', dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()
    
    pd.DataFrame({'method': list(agreement_scores.keys()), 
                  'agreement_score': list(agreement_scores.values())}).to_csv(
        output_path / 'inter_method_agreement.csv', index=False)
    
    return agreement_scores


# ============================================================================
# STRATEGY 3: PCA-BASED APPROACH
# ============================================================================

def pca_approach(data, methods):
    """Use PCA to find the main component across methods."""
    print("\n" + "="*70)
    print("STRATEGY 3: PCA-BASED APPROACH")
    print("="*70)
    
    norm_data = normalize_methods(data, methods)
    
    # Prepare data (remove NaNs)
    X = norm_data[methods].dropna()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nExplained variance by component:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    print(f"  Cumulative (first 3): {pca.explained_variance_ratio_[:3].sum():.4f}")
    
    # Loadings (how much each method contributes to PC1)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(methods))],
        index=methods
    )
    
    print(f"\nPC1 Loadings (main component):")
    for method, loading in loadings['PC1'].abs().sort_values(ascending=False).items():
        print(f"  {method:30s}: {loadings.loc[method, 'PC1']:7.4f} (|{loading:.4f}|)")
    
    # Save
    loadings.to_csv(output_path / 'pca_loadings.csv')
    
    # Plot loadings
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PC1 loadings
    sorted_methods = loadings['PC1'].abs().sort_values(ascending=False).index
    values = [loadings.loc[m, 'PC1'] for m in sorted_methods]
    colors = ['#10b981' if v > 0 else '#ef4444' for v in values]
    
    axes[0].barh(range(len(sorted_methods)), values, color=colors)
    axes[0].set_yticks(range(len(sorted_methods)))
    axes[0].set_yticklabels([m.replace('_', ' ').title() for m in sorted_methods]) # , fontsize=9)
    axes[0].set_xlabel('PC1 Loading') # , fontsize=11)
    axes[0].set_title(f'PC1 Loadings (Explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance)') # , 
                        # fontsize=12, fontweight='bold')
    axes[0].axvline(x=0, color='black', linewidth=0.8)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Scree plot
    axes[1].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                 pca.explained_variance_ratio_, 'o-', linewidth=2, markersize=8, color='#6366f1')
    axes[1].set_xlabel('Principal Component') # , fontsize=11)
    axes[1].set_ylabel('Explained Variance Ratio') # , fontsize=11)
    axes[1].set_title('Scree Plot') # , fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()
    
    # Calculate correlation of each method with PC1
    pc1_scores = X_pca[:, 0]
    pc1_correlations = {}
    for i, method in enumerate(methods):
        pc1_correlations[method] = np.corrcoef(X[method], pc1_scores)[0, 1]
    
    return loadings, pca, pc1_correlations


# ============================================================================
# STRATEGY 4: COEFFICIENT OF VARIATION (STABILITY)
# ============================================================================

def stability_analysis(data, methods):
    """Analyze which methods have more stable/consistent values."""
    print("\n" + "="*70)
    print("STRATEGY 4: STABILITY ANALYSIS")
    print("="*70)
    
    norm_data = normalize_methods(data, methods)
    
    stability_metrics = {}
    
    for method in methods:
        values = norm_data[method].dropna()
        stability_metrics[method] = {
            'cv': values.std() / values.mean() if values.mean() != 0 else np.inf,  # Coefficient of variation
            'iqr': values.quantile(0.75) - values.quantile(0.25),  # Interquartile range
            'range': values.max() - values.min(),
            'mad': np.median(np.abs(values - values.median()))  # Median absolute deviation
        }
    
    print("\nStability metrics (lower = more stable/consistent):")
    print("\nCoefficient of Variation (CV):")
    for method, metrics in sorted(stability_metrics.items(), key=lambda x: x[1]['cv']):
        print(f"  {method:30s}: {metrics['cv']:.4f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (metric_name, ylabel) in enumerate([
        ('cv', 'Coefficient of Variation'),
        ('iqr', 'Interquartile Range'),
        ('range', 'Range'),
        ('mad', 'Median Absolute Deviation')
    ]):
        ax = axes[idx // 2, idx % 2]
        
        sorted_methods = sorted(stability_metrics.keys(), 
                               key=lambda x: stability_metrics[x][metric_name])
        values = [stability_metrics[m][metric_name] for m in sorted_methods]
        
        ax.barh(range(len(sorted_methods)), values, color='#6366f1')
        ax.set_yticks(range(len(sorted_methods)))
        ax.set_yticklabels([m.replace('_', ' ').title() for m in sorted_methods], fontsize=8)
        ax.set_xlabel(ylabel) # , fontsize=10)
        ax.set_title(f'{ylabel}\n(Lower = More Stable)') # , fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'stability_analysis.png', dpi=300, bbox_inches='tight')
    # plt.close() 
    plt.show()
    
    pd.DataFrame(stability_metrics).T.to_csv(output_path / 'stability_metrics.csv')
    
    return stability_metrics


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def create_summary_report(consensus_results, agreement_scores, pc1_correlations, stability_metrics):
    """Create a comprehensive summary ranking methods."""
    print("\n" + "="*70)
    print("COMPREHENSIVE SUMMARY")
    print("="*70)
    
    methods = list(agreement_scores.keys())
    
    summary = pd.DataFrame(index=methods)
    
    # Add consensus correlations
    summary['consensus_mean'] = [consensus_results['mean'][m] for m in methods]
    summary['consensus_median'] = [consensus_results['median'][m] for m in methods]
    
    # Add inter-method agreement
    summary['inter_method_agreement'] = [agreement_scores[m] for m in methods]
    
    # Add PC1 correlation (absolute value)
    summary['pc1_correlation'] = [abs(pc1_correlations[m]) for m in methods]
    
    # Add stability (inverse CV, so higher = better)
    summary['stability'] = [1 / (stability_metrics[m]['cv'] + 0.01) for m in methods]
    
    # Normalize all metrics to [0, 1]
    for col in summary.columns:
        min_val = summary[col].min()
        max_val = summary[col].max()
        if max_val > min_val:
            summary[col] = (summary[col] - min_val) / (max_val - min_val)
    
    # Calculate composite score (equal weighting)
    summary['composite_score'] = summary.mean(axis=1)
    
    # Rank
    summary['rank'] = summary['composite_score'].rank(ascending=False)
    
    # Sort by composite score
    summary = summary.sort_values('composite_score', ascending=False)
    
    print("\nFINAL RANKING (by composite score):")
    print(summary.to_string())
    
    summary.to_csv(output_path / 'method_ranking_summary.csv')
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_methods = summary.index
    y_pos = range(len(sorted_methods))
    
    ax.barh(y_pos, summary['composite_score'], color='#6366f1', alpha=0.7, label='Composite Score')
    
    # Add individual metric markers
    for col, marker, color in [
        ('consensus_mean', 'o', '#10b981'),
        ('inter_method_agreement', 's', '#f59e0b'),
        ('pc1_correlation', '^', '#ef4444'),
        ('stability', 'D', '#8b5cf6')
    ]:
        ax.scatter(summary[col], y_pos, marker=marker, s=80, 
                  color=color, label=col.replace('_', ' ').title(), 
                  edgecolors='white', linewidth=1, zorder=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ').title() for m in sorted_methods]) # , fontsize=9)
    ax.set_xlabel('Score (0-1, normalized)') # , fontsize=11)
    ax.set_title('Method Ranking: Composite Score and Individual Metrics') # , 
                # fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right') # , fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'method_ranking_summary.png', dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()
    
    return summary


def main():
    print("="*70)
    print("COMPREHENSIVE METHOD EVALUATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data, methods = load_all_methods()
    print(f"\nLoaded {len(methods)} methods with {len(data)} networks")
    
    # Run all strategies
    norm_data, consensus_results = consensus_approaches(data, methods)
    agreement_scores = inter_method_agreement(data, methods)
    loadings, pca_model, pc1_correlations = pca_approach(data, methods)
    stability_metrics = stability_analysis(data, methods)
    
    # Create summary
    summary = create_summary_report(consensus_results, agreement_scores, 
                                   pc1_correlations, stability_metrics)
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! All outputs saved to: {output_path}")
    print(f"{'='*70}")
    print("\nKey insights:")
    print(f"1. Best overall method: {summary.index[0]}")
    print(f"2. Top 3 methods: {', '.join(summary.index[:3])}")
    print(f"3. Check '{output_path}/method_ranking_summary.csv' for details")

if __name__ == "__main__":
    main()

# ====================================================================
# cell 69
# ====================================================================

# STRATEGY 3: PCA-BASED APPROACH
"""Use PCA to find the main component across methods."""

# Load data
data, methods = load_all_methods()


norm_data = normalize_methods(data, methods)

# Prepare data (remove NaNs)
X = norm_data[methods].dropna()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nExplained variance by component:")
for i, var in enumerate(pca.explained_variance_ratio_[:5]):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
print(f"  Cumulative (first 3): {pca.explained_variance_ratio_[:3].sum():.4f}")

# Loadings (how much each method contributes to PC1)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(methods))],
    index=methods
)

print(f"\nPC1 Loadings (main component):")
for method, loading in loadings['PC1'].abs().sort_values(ascending=False).items():
    print(f"  {method:30s}: {loadings.loc[method, 'PC1']:7.4f} (|{loading:.4f}|)")

# Save
loadings.to_csv(output_path / 'pca_loadings.csv')

# Plot loadings
fig, axes = plt.subplots(1, 2, figsize=viz.cm_to_inch((18,7)))# (16, 6))

# PC1 loadings
sorted_methods = loadings['PC1'].abs().sort_values(ascending=False).index
values = [loadings.loc[m, 'PC1'] for m in sorted_methods]
colors = ['#10b981' if v > 0 else '#ef4444' for v in values]

axes[0].barh(range(len(sorted_methods)), values, color=colors)
axes[0].set_yticks(range(len(sorted_methods)))
axes[0].set_yticklabels([m.replace('_', ' ').title() for m in sorted_methods]) # , fontsize=9)
axes[0].set_xlabel('PC1 Loading') # , fontsize=11)
axes[0].set_title(f'PC1 Loadings (Explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance)') # , 
                    # fontsize=12, fontweight='bold')
axes[0].axvline(x=0, color='black', linewidth=0.8)
axes[0].grid(axis='x', alpha=0.3)

# Scree plot
axes[1].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, 'o-', linewidth=2, markersize=8, color='#6366f1')
axes[1].set_xlabel('Principal Component') # , fontsize=11)
axes[1].set_ylabel('Explained Variance Ratio') # , fontsize=11)
axes[1].set_title('Scree Plot') # , fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'pca_analysis.png', dpi=300, bbox_inches='tight')
# plt.close()
plt.show()

# Calculate correlation of each method with PC1
pc1_scores = X_pca[:, 0]
pc1_correlations = {}
for i, method in enumerate(methods):
    pc1_correlations[method] = np.corrcoef(X[method], pc1_scores)[0, 1]

# ====================================================================
# cell 70
# ====================================================================
# Calculate variance for each metric across networks at each eta-gamma point
variance_matrices_of_metrics = {}

for idx, mode in enumerate(all_dist_measures):
    path = f"output/gnm/{dataset_name}/{experiment_name}/summary_indiv_{mode}_for_exp_{experiment_name}.csv"
    df = pd.read_csv(path)
    
    # Get the metric column
    metric_col = [col for col in df.columns if col not in ["eta", "gamma", "network_index", "filename", "id"]]
    if not metric_col:
        raise ValueError(f"No metric column found in {df.columns}")
    
    # Round eta and gamma
    df['eta'] = df['eta'].round(2)
    df['gamma'] = df['gamma'].round(2)
    filtering_df['eta'] = filtering_df['eta'].round(2)
    filtering_df['gamma'] = filtering_df['gamma'].round(2)

    # Merge and filter if needed
    if "no" in experiment_name:
        df = df.merge(filtering_df[['eta', 'gamma', 'n_connected_components']], 
                    on=['eta', 'gamma'], 
                    how='left')
        df = df[df['n_connected_components'] <= 1]
    else: 
        df = df.merge(filtering_df[['eta', 'gamma']], 
                    on=['eta', 'gamma'], 
                    how='left')

    # Turn df into a pivot table - compute VARIANCE instead of mean
    df_pivot = df.pivot_table(index='gamma', columns='eta', values=metric_col[0], aggfunc='var')

    # Min-max normalize the variance values to [0, 1]
    df_pivot = (df_pivot - df_pivot.min().min()) / (df_pivot.max().max() - df_pivot.min().min())
    
    # Store the matrix as a numpy array
    matrix = df_pivot.values[::-1]
    
    variance_matrices_of_metrics[mode] = matrix

# Create the grid plot for variance
n_methods = len(all_methods_sorted_for_legend[::-1])
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

image_size_in_cm = 6
fontsize = 8
height = image_size_in_cm * (n_rows / n_cols) 

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((image_size_in_cm, height)),
                            dpi=150,
                            squeeze=True)

axes = axes.flatten()

for idx, mode in enumerate(all_methods_sorted_for_legend[::-1]):
    ax = axes[idx]
    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            metric_colors[mode], 
            "white", 
        ]
    )

    scatter = ax.imshow(variance_matrices_of_metrics[mode], 
                        cmap=cmap, 
                       aspect='auto',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # Set ticks
    if (idx % n_cols == 0 and idx // n_cols == n_rows -1): 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == 0: 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([]) 
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == len(all_methods_sorted_for_legend[::-1]) -1: 
        ax.set_yticks([])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
    else: 
        ax.set_yticks([])
        ax.set_xticks([])

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Get min and max location of the corners of the subplots
x_min = np.min([ax.get_position().x0 for ax in axes]) + 0.025
x_max = np.max([ax.get_position().x1 for ax in axes]) + 2 * 0.025
y_min = np.min([ax.get_position().y0 for ax in axes])
y_max = np.max([ax.get_position().y1 for ax in axes])

# Add one shared colorbar below all subplots
cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_colorbar', 
        [
            "black",
            "white", 
         ])
norm = plt.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([x_min, 0, x_max - x_min, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Variance (Normalized)")
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xticks([0, 10], labels=['Low', 'High'])

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.savefig(output_path / "sixteen_subplots_variance.pdf")
print(output_path / "sixteen_subplots_variance.pdf")
plt.show()


# ====================================================================
# cell 71
# ====================================================================
# Calculate variance for each metric across networks at each eta-gamma point
variance_matrices_of_metrics = {}

for idx, mode in enumerate(all_dist_measures):
    path = f"output/gnm/{dataset_name}/{experiment_name}/summary_indiv_{mode}_for_exp_{experiment_name}.csv"
    df = pd.read_csv(path)
    
    # Get the metric column
    metric_col = [col for col in df.columns if col not in ["eta", "gamma", "network_index", "filename", "id"]]
    if not metric_col:
        raise ValueError(f"No metric column found in {df.columns}")
    
    # Round eta and gamma
    df['eta'] = df['eta'].round(2)
    df['gamma'] = df['gamma'].round(2)
    filtering_df['eta'] = filtering_df['eta'].round(2)
    filtering_df['gamma'] = filtering_df['gamma'].round(2)

    # Merge and filter if needed
    if "no" in experiment_name:
        df = df.merge(filtering_df[['eta', 'gamma', 'n_connected_components']], 
                    on=['eta', 'gamma'], 
                    how='left')
        df = df[df['n_connected_components'] <= 1]
    else: 
        df = df.merge(filtering_df[['eta', 'gamma']], 
                    on=['eta', 'gamma'], 
                    how='left')

    # Turn df into a pivot table - compute VARIANCE instead of mean
    df_pivot = df.pivot_table(index='gamma', columns='eta', values=metric_col[0], aggfunc='var')

    # Store unnormalized for finding min points
    variance_matrices_of_metrics[mode] = {
        'pivot': df_pivot,
        'matrix_raw': df_pivot.values[::-1]
    }
    
    # Min-max normalize the variance values to [0, 1]
    df_pivot_norm = (df_pivot - df_pivot.min().min()) / (df_pivot.max().max() - df_pivot.min().min())
    variance_matrices_of_metrics[mode]['matrix_norm'] = df_pivot_norm.values[::-1]

# Create the grid plot for variance
n_methods = len(all_methods_sorted_for_legend[::-1])
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

image_size_in_cm = 6
fontsize = 8
height = image_size_in_cm * (n_rows / n_cols) 

fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=viz.cm_to_inch((image_size_in_cm, height)),
                            dpi=150,
                            squeeze=True,
                            constrained_layout=False)

axes = axes.flatten()

for idx, mode in enumerate(all_methods_sorted_for_legend[::-1]):
    ax = axes[idx]
    
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_{mode}', 
        [
            "black",
            metric_colors[mode], 
            "white", 
        ]
    )

    matrix_norm = variance_matrices_of_metrics[mode]['matrix_norm']
    scatter = ax.imshow(matrix_norm, 
                        cmap=cmap, 
                       aspect='auto',
                       extent=[
                           df["eta"].min(), df["eta"].max(),
                           df["gamma"].min(), df["gamma"].max()
                       ])
    
    # Find 20 points with minimum variance
    pivot_df = variance_matrices_of_metrics[mode]['pivot']
    
    # Flatten and find indices of 20 smallest values
    flat_values = pivot_df.values.flatten()
    flat_indices = np.argsort(flat_values)[:100]
    
    # Convert flat indices back to 2D coordinates
    rows, cols = np.unravel_index(flat_indices, pivot_df.shape)
    
    # Get eta and gamma values for these points
    eta_values = pivot_df.columns[cols]
    gamma_values = pivot_df.index[rows]
    
    # Plot markers at these points
    ax.scatter(eta_values, gamma_values, 
               color=metric_colors[mode], 
               s=3, 
               marker='o', 
               linewidths=0.25, 
               edgecolors='white',
               alpha=1, zorder=10)
    
    # Set ticks
    if (idx % n_cols == 0 and idx // n_cols == n_rows -1): 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == 0: 
        ax.set_yticks([np.ceil(df["gamma"].min()*10)/10, np.floor(df["gamma"].max()*10)/10])
        ax.set_xticks([]) 
        ax.set_ylabel(r"$\gamma$", fontsize=fontsize)
    elif idx == len(all_methods_sorted_for_legend[::-1]) -1: 
        ax.set_yticks([])
        ax.set_xticks([np.ceil(df["eta"].min()*10)/10, np.floor(df["eta"].max()*10)/10])
        ax.set_xlabel(r"$\eta$", fontsize=fontsize)
    else: 
        ax.set_yticks([])
        ax.set_xticks([])

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis('off')

# Adjust spacing manually - increase bottom margin to leave more room for colorbar
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.98, wspace=0.1, hspace=0.1)

# Get positions AFTER subplots_adjust
x_min = np.min([ax.get_position().x0 for ax in axes[:n_methods]])
x_max = np.max([ax.get_position().x1 for ax in axes[:n_methods]])
y_min = np.min([ax.get_position().y0 for ax in axes[:n_methods]])

# Add colorbar in the reserved space at bottom - position it higher
cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        f'custom_cmap_colorbar', 
        ["black", "white"])
norm = plt.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Position colorbar with more space below it (y=0.03 instead of 0.01)
cbar_ax = fig.add_axes([x_min, -0.1, x_max - x_min, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Variance (Normalized)")
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xticks([0, 10], labels=['Low', 'High'])

# Save with bbox_inches to include everything
plt.savefig(output_path / "sixteen_subplots_variance_marked.pdf", bbox_inches='tight', pad_inches=0.1)
print(output_path / "sixteen_subplots_variance_marked.pdf")
plt.show()

# ====================================================================
# cell 72
# ====================================================================
fig, ax = plt.subplots(figsize=viz.cm_to_inch((12, 6)))

bp = ax.boxplot(
    timing_data,
    patch_artist=True,
    showfliers=False,
)

# Color boxes
for patch, method in zip(bp['boxes'], method_key_names_ordered):
    patch.set_facecolor(metric_colors[method])

# Median lines
for median in bp['medians']:
    median.set_color('black')

ax.set_ylabel('Time (seconds)')
ax.set_yscale('log')
ax.set_xticks([])  # remove x ticks entirely


pos = ax.get_position()

bar_height = 0.03
gap = 0.1

cax = fig.add_axes([
    pos.x0 - 0.005, # - bar_width - gap,  # left of boxplot
    pos.y0 - bar_height - gap, # below boxplot
    pos.width + 0.075, # + gap, # + 0.015,  # same width as boxplot
    bar_height
])

colors_for_legend = [metric_colors[m] for m in method_key_names_ordered]

cmap = mpl.colors.ListedColormap(colors_for_legend)
norm = mpl.colors.BoundaryNorm(
    boundaries=np.arange(len(colors_for_legend) + 1),
    ncolors=len(colors_for_legend)
)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cb = fig.colorbar(sm, cax=cax, orientation='horizontal')

cb.set_ticklabels([]) 
cb.ax.yaxis.set_visible(False)
cb.ax.xaxis.set_visible(False)

plt.tight_layout()

plt.savefig(output_path / "timing_16.pdf")
print(output_path / "timing_16.pdf")

# ====================================================================
# cell 73
# ====================================================================
# --- Figure and axis ---------------------------------------------------------
fig, ax = plt.subplots(figsize=viz.cm_to_inch((12, 6)))


# --- Boxplot -----------------------------------------------------------------
boxplot = ax.boxplot(
    timing_data,
    patch_artist=True,
    showfliers=False,
)

# Color boxes by method
for box, method in zip(boxplot["boxes"], method_key_names_ordered):
    box.set_facecolor(metric_colors[method])

# Style median lines
for median in boxplot["medians"]:
    median.set_color("black")

# Axis styling
ax.set_ylabel("Time (seconds)")
# ax.set_yscale("log")
ax.set_xticks([])

# --- Colorbar-style legend ---------------------------------------------------
ax_pos = ax.get_position()

bar_height = 0.03
gap = 0.025 # 1

cax = fig.add_axes([
    ax_pos.x0 - 0.005,
    ax_pos.y0 - bar_height - gap,
    ax_pos.width + 0.075,
    bar_height,
])

legend_colors = [metric_colors[m] for m in method_key_names_ordered]

cmap = mpl.colors.ListedColormap(legend_colors)
norm = mpl.colors.BoundaryNorm(
    boundaries=np.arange(len(legend_colors) + 1),
    ncolors=len(legend_colors),
)

scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
scalar_mappable.set_array([])

colorbar = fig.colorbar(
    scalar_mappable,
    cax=cax,
    orientation="horizontal",
)

# Hide all colorbar axes/ticks
colorbar.set_ticklabels([])
colorbar.ax.xaxis.set_visible(False)
colorbar.ax.yaxis.set_visible(False)

# --- Save --------------------------------------------------------------------
# plt.tight_layout()
plt.tight_layout(rect=[0, 0.06, 1, 1])

output_file = output_path / "timing_16.pdf"
plt.savefig(output_file)
print(output_file)

