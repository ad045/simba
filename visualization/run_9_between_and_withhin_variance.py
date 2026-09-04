"""
9_between_and_withhin_variance

Extracted from 9_between_and_withhin_variance.ipynb by extract_notebook.py - do not edit by hand.
The run comes from experiments_config (MORPHO_EXP); paths that pointed into the
connectome_distances repo now use this repo's own data.
"""
import matplotlib
matplotlib.use("Agg")

# ====================================================================
# cell 0
# ====================================================================
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
import seaborn as sns

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
# cell 1
# ====================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Base directory
dataset = "hcp_schaefer_100_dataset"
base_dir = Path(f'/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset}/first_chaos_analysis/')

# Get all CSV files
csv_files = list(base_dir.glob('chaos_analysis_*.csv'))

results = []

csv_files

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



# Get colormaps
dark2 = plt.get_cmap("Dark2")
set1 = plt.get_cmap("Set1")

# Extract colors
dark2_colors = list(dark2.colors)
set1_colors = [c for i, c in enumerate(set1.colors)] # if i != 4]

# Combine into one list
all_colors = dark2_colors + set1_colors
# remove index 13
all_colors.pop(13)

# reverse, just out of aesthetic preference
all_colors = all_colors # [::-1]

# ====================================================================
# cell 3
# ====================================================================

metric_colors = {'communicability_corr': (0.6, 0.6, 0.6),
                'dc_network_mutual_information': (0.9686274509803922,
                0.5058823529411764,
                0.7490196078431373),
                'network_mutual_information': (0.6509803921568628,
                0.33725490196078434,
                0.1568627450980392),
                'f1': (1.0, 0.4980392156862745, 0.0),
                'jaccard': (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
                'communicability_jsd': (0.30196078431372547,
                0.6862745098039216,
                0.2901960784313726),
                'netrd_non_backtracking_spectral': (0.21568627450980393,
                0.49411764705882355,
                0.7215686274509804),
                'resistance': (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
                'delta_con': (0.4, 0.4, 0.4),
                'hamming': (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                'frobenius': (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                'net_simile': (0.4, 0.6509803921568628, 0.11764705882352941),
                'spectral_distance_adjacency': (0.9058823529411765,
                0.1607843137254902,
                0.5411764705882353),
                'spectral_distance_norm_laplacian': (0.4588235294117647,
                0.4392156862745098,
                0.7019607843137254),
                'energy': (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                'portrait': (0.10588235294117647, 0.6196078431372549, 0.4666666666666667)}



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
}



# Get the name of all measures
all_dist_measures = [
    "energy", 
    "portrait", 
    "spectral_distance_adjacency",
    # "spectral_distance_norm_laplacian", 
    
    "communicability_corr",
    # "communicability_jsd",
    # "network_mutual_information",
    # "dc_network_mutual_information", 
    
    "net_simile", 
    "netrd_non_backtracking_spectral", 
    # "resistance", 
    "delta_con", 
    
    # "f1", 
    # "hamming",
    "frobenius", 
    # "jaccard", 
]

# ====================================================================
# cell 6
# ====================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import pandas as pd

class VisualQualityAnalyzerWithVariation:
    """Analyze visual quality incorporating within-pixel variation"""
    
    def __init__(self, matrix, std_matrix=None, mask=None):
        """
        Args:
            matrix: Mean values at each pixel
            std_matrix: Standard deviation at each pixel (from the 10 samples)
            mask: Boolean mask for valid pixels
        """
        self.matrix = np.array(matrix, dtype=float)
        self.std_matrix = std_matrix if std_matrix is not None else None
        self.mask = mask if mask is not None else np.ones_like(matrix, dtype=bool)
        
    # ============= ORIGINAL METRICS (for comparison) =============
    
    def calculate_gradient_smoothness(self):
        """Lower = smoother gradients, less noisy"""
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)
        smoothness_score = np.std(grad_y) + np.std(grad_x)
        return smoothness_score
    
    def calculate_total_variation(self):
        """Total variation - lower = smoother"""
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)
        tv = np.sum(np.abs(grad_y)) + np.sum(np.abs(grad_x[:, :-1]))
        return tv / self.matrix.size
    
    def calculate_snr(self, sigma=2):
        """Signal-to-noise ratio - higher = clearer structure"""
        smoothed = gaussian_filter(self.matrix, sigma=sigma)
        noise = self.matrix - smoothed
        
        smoothed = smoothed[self.mask]
        noise = noise[self.mask]

        signal_power = np.var(smoothed - np.mean(smoothed))
        noise_power = np.var(noise)
        
        if noise_power < 1e-10:
            return 100
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    # ============= NEW METRICS USING WITHIN-PIXEL VARIATION =============
    
    def calculate_mean_pixel_uncertainty(self):
        """
        Average coefficient of variation across all pixels.
        Lower = more reliable measurements
        """
        if self.std_matrix is None:
            return np.nan
        
        # Avoid division by zero
        cv = np.divide(self.std_matrix, np.abs(self.matrix) + 1e-10, 
                       where=self.mask)
        cv = cv[self.mask]
        
        return np.mean(cv)
    
    def calculate_intrinsic_snr(self):
        """
        SNR using within-pixel variation as the true noise.
        Higher = signal dominates intrinsic measurement noise
        """
        if self.std_matrix is None:
            return np.nan
        
        matrix_masked = self.matrix[self.mask]
        std_masked = self.std_matrix[self.mask]
        
        signal_power = np.var(matrix_masked)
        noise_power = np.mean(std_masked**2)  # Mean variance across pixels
        
        if noise_power < 1e-10:
            return 100
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def calculate_uncertainty_smoothness(self):
        """
        How smooth is the uncertainty landscape?
        Lower = uncertainty changes smoothly (good)
        """
        if self.std_matrix is None:
            return np.nan
        
        grad_y = np.diff(self.std_matrix, axis=0)
        grad_x = np.diff(self.std_matrix, axis=1)
        
        return np.std(grad_y) + np.std(grad_x)
    
    def calculate_high_uncertainty_fraction(self, cv_threshold=0.3):
        """
        Fraction of pixels with high uncertainty (CV > threshold).
        Lower = fewer unreliable regions
        """
        if self.std_matrix is None:
            return np.nan
        
        cv = np.divide(self.std_matrix, np.abs(self.matrix) + 1e-10, 
                       where=self.mask)
        cv = cv[self.mask]
        
        return np.mean(cv > cv_threshold)
    
    def calculate_confidence_weighted_smoothness(self):
        """
        Gradient smoothness weighted by confidence (inverse uncertainty).
        Focuses on regions where we're confident about the values.
        Lower = smoother in high-confidence regions
        """
        if self.std_matrix is None:
            return self.calculate_gradient_smoothness()
        
        # Calculate confidence weights (inverse of CV)
        cv = np.divide(self.std_matrix, np.abs(self.matrix) + 1e-10)
        confidence = 1 / (cv + 0.1)  # Add small constant to avoid division by zero
        
        # Calculate gradients
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)
        
        # Weight gradients by average confidence of adjacent pixels
        conf_y = (confidence[:-1, :] + confidence[1:, :]) / 2
        conf_x = (confidence[:, :-1] + confidence[:, 1:]) / 2
        
        # Weighted variance of gradients
        weighted_var_y = np.average(grad_y**2, weights=conf_y)
        weighted_var_x = np.average(grad_x**2, weights=conf_x)
        
        return np.sqrt(weighted_var_y + weighted_var_x)
    
    def calculate_signal_vs_gradient_noise(self):
        """
        Compare the magnitude of signal gradients to measurement noise.
        Higher = gradients are real signal, not just noise
        """
        if self.std_matrix is None:
            return np.nan
        
        # Calculate signal gradients
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)
        
        # Propagate uncertainty for gradients (assuming independent measurements)
        grad_noise_y = np.sqrt(self.std_matrix[:-1, :]**2 + self.std_matrix[1:, :]**2)
        grad_noise_x = np.sqrt(self.std_matrix[:, :-1]**2 + self.std_matrix[:, 1:]**2)
        
        # Calculate ratio: gradient magnitude / gradient uncertainty
        ratio_y = np.abs(grad_y) / (grad_noise_y + 1e-10)
        ratio_x = np.abs(grad_x) / (grad_noise_x + 1e-10)
        
        # Average ratio across all gradients
        return np.mean(np.concatenate([ratio_y.flatten(), ratio_x.flatten()]))
    
    def calculate_effective_resolution(self):
        """
        How many 'effective' independent measurements do we have?
        Accounts for both spatial correlation and measurement uncertainty.
        Higher = more reliable information
        """
        if self.std_matrix is None:
            return np.nan
        
        # Calculate local correlation length (simplified)
        smoothed = gaussian_filter(self.matrix, sigma=2)
        correlation = np.corrcoef(self.matrix.flatten(), smoothed.flatten())[0, 1]
        
        # Calculate effective sample size considering uncertainty
        cv = np.divide(self.std_matrix, np.abs(self.matrix) + 1e-10)
        mean_cv = np.mean(cv[self.mask])
        
        # Effective resolution decreases with correlation and uncertainty
        base_resolution = self.matrix.size
        uncertainty_penalty = 1 / (1 + mean_cv)
        correlation_penalty = 1 - correlation
        
        return base_resolution * uncertainty_penalty * correlation_penalty
    
    def calculate_uncertainty_vs_structure_ratio(self):
        """
        Ratio of measurement uncertainty to actual structure variation.
        Lower = structure dominates over measurement noise
        """
        if self.std_matrix is None:
            return np.nan
        
        # Structure variation (how much the mean values vary)
        structure_var = np.var(self.matrix[self.mask])
        
        # Average measurement uncertainty
        avg_measurement_var = np.mean(self.std_matrix[self.mask]**2)
        
        return avg_measurement_var / (structure_var + 1e-10)
        # return np.log(avg_measurement_var / (structure_var + 1e-10))
    
    # ============= EXISTING METRICS =============
    
    def calculate_local_coherence(self, window_size=3):
        """Higher = more locally coherent"""
        local_mean = uniform_filter(self.matrix, size=window_size)
        local_sq_mean = uniform_filter(self.matrix**2, size=window_size)
        local_var = local_sq_mean - local_mean**2
        
        local_var = local_var[self.mask]
        coherence = 1 / (np.mean(local_var) + 1e-10)
        return coherence
    
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
    
    def calculate_edge_strength(self):
        """Measure edge clarity using Sobel-like gradients"""
        grad_y = np.diff(self.matrix, axis=0)
        grad_x = np.diff(self.matrix, axis=1)
        
        grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
        grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        
        grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        return np.mean(grad_magnitude)
    
    def get_all_metrics(self):
        """Calculate all quality metrics"""
        metrics = {
            # Original metrics
            'entropy': self.calculate_entropy(),
            'gradient_smoothness': self.calculate_gradient_smoothness(),
            'total_variation': self.calculate_total_variation(),
            'local_coherence': self.calculate_local_coherence(),
            'snr': self.calculate_snr(),
            'structure_similarity': self.calculate_structure_similarity(),
            'edge_strength': self.calculate_edge_strength(),
        }
        
        # Add new variation-based metrics if std_matrix is provided
        if self.std_matrix is not None:
            metrics.update({
                'mean_pixel_uncertainty': self.calculate_mean_pixel_uncertainty(),
                'intrinsic_snr': self.calculate_intrinsic_snr(),
                'uncertainty_smoothness': self.calculate_uncertainty_smoothness(),
                'high_uncertainty_fraction': self.calculate_high_uncertainty_fraction(),
                'confidence_weighted_smoothness': self.calculate_confidence_weighted_smoothness(),
                'signal_vs_gradient_noise': self.calculate_signal_vs_gradient_noise(),
                'effective_resolution': self.calculate_effective_resolution(),
                'uncertainty_structure_ratio': self.calculate_uncertainty_vs_structure_ratio(),
            })
        
        return metrics


def analyze_all_methods_with_variation(matrices_dict, std_matrices_dict=None, nan_strategy='mean'):
    """
    Analyze all similarity methods with optional variation information
    
    Args:
        matrices_dict: Dict of mean matrices
        std_matrices_dict: Dict of std matrices (same keys as matrices_dict)
        nan_strategy: How to handle NaNs
    """
    results = {}
    
    for name, matrix in matrices_dict.items():
        matrix = matrix.copy()
        
        # Get std matrix if available
        std_matrix = None
        if std_matrices_dict is not None and name in std_matrices_dict:
            std_matrix = std_matrices_dict[name].copy()
        
        # Create mask from NaN positions
        mask = np.ones_like(matrix, dtype=bool)
        mask[np.isnan(matrix)] = False
        
        # Handle NaNs in both matrices
        if np.isnan(matrix).any():
            if nan_strategy == 'mean':
                matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix))
                if std_matrix is not None:
                    std_matrix = np.nan_to_num(std_matrix, nan=np.nanmean(std_matrix))
            elif nan_strategy == 'zero':
                matrix = np.nan_to_num(matrix, nan=0)
                if std_matrix is not None:
                    std_matrix = np.nan_to_num(std_matrix, nan=0)
        
        # Min-max normalize the mean matrix
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        
        # Normalize std matrix proportionally
        if std_matrix is not None:
            original_range = matrices_dict[name].max() - matrices_dict[name].min()
            if original_range > 1e-10:
                std_matrix = std_matrix / original_range
        
        analyzer = VisualQualityAnalyzerWithVariation(matrix, std_matrix, mask)
        results[name] = analyzer.get_all_metrics()
    
    return pd.DataFrame(results).T


# ====================================================================
# cell 8
# ====================================================================

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from experiments_config import (MORPHO_DATASET, MORPHO_EXP, ROOT_DIR as _RD,
                                DIST_MATRIX_PATH, INDIVIDUALS_PATH, to_distance)

dataset_name = MORPHO_DATASET
experiment_name = MORPHO_EXP

# experiment_name = "02_ring_sweeps_animal_0" # "01_ring_sweep_animal_0" # "95_ring_seed_100_sweep_animal_206"
# experiment_name = "03_no_ring_sweeps_animal_0"
base_path = (_RD / "output" / "gnm" / dataset_name / experiment_name)
save_path = base_path / f"all_metrics_for_{experiment_name}.csv"


filtering_df_path = base_path / f"all_metrics_for_{experiment_name}.csv" # _updated.csv"
filtering_df = pd.read_csv(filtering_df_path)
# Modify your data loading code to also compute std:


# ====================================================================
# cell 9
# ====================================================================
matrices_of_metrics = {}
std_matrices_of_metrics = {}

base_path = (_RD / "output" / "gnm" / dataset_name / experiment_name)

for idx, mode in enumerate(all_dist_measures):
    
    if mode in ["communicability_mse"]:
        continue

    path = base_path / f"summary_indiv_{mode}_for_exp_{experiment_name}.csv"
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

    # Create pivot tables for BOTH mean and std
    df_pivot_mean = df.pivot_table(index='gamma', columns='eta', 
                                    values=metric_col[0], aggfunc='mean')
    df_pivot_std = df.pivot_table(index='gamma', columns='eta', 
                                   values=metric_col[0], aggfunc='std')
    
    # Normalize mean
    matrix_mean = df_pivot_mean.values[::-1]
    matrix_mean = (matrix_mean - np.nanmin(matrix_mean)) / (np.nanmax(matrix_mean) - np.nanmin(matrix_mean))
    
    # Keep std in original scale (will be normalized later)
    matrix_std = df_pivot_std.values[::-1]


    if mode in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance: 
        matrix_mean = 1 - matrix_mean
    
    matrices_of_metrics[mode] = matrix_mean
    std_matrices_of_metrics[mode] = matrix_std

# Analyze with variation information
results_df = analyze_all_methods_with_variation(
    matrices_of_metrics, 
    std_matrices_of_metrics
)


# ====================================================================
# cell 10
# ====================================================================
"""Plot metrics comparing original vs variation-aware metrics"""

# Separate original and new metrics
original_metrics = ['entropy', 'gradient_smoothness', 'total_variation', 
                    'local_coherence', 'snr', 'structure_similarity', 'edge_strength']
variation_metrics = ['mean_pixel_uncertainty', 'intrinsic_snr', 
                    'uncertainty_smoothness', 'high_uncertainty_fraction',
                    'confidence_weighted_smoothness', 'signal_vs_gradient_noise',
                    'effective_resolution', 'uncertainty_structure_ratio']

# Check which metrics are available
available_variation = [m for m in variation_metrics if m in results_df.columns]

# n_plots = len(original_metrics) + len(available_variation)
# n_cols = 2
# n_rows = (n_plots) // n_cols + 1

# fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((18, 24)))
# axes = axes.flatten()

# # Plot original metrics
# for idx, metric in enumerate(original_metrics):
#     ax = axes[idx]
#     sorted_data = results_df[metric].sort_values(ascending=False)
#     ax.barh(range(len(sorted_data)), sorted_data.values, color='steelblue', alpha=0.7)
#     ax.set_yticks(range(len(sorted_data)))
#     ax.set_yticklabels(sorted_data.index, fontsize=7)
#     ax.set_title(metric.replace('_', ' ').title(), fontsize=9)
#     ax.grid(axis='x', alpha=0.3)

# # Plot variation metrics
# for idx, metric in enumerate(available_variation, start=len(original_metrics)):
#     ax = axes[idx]
#     sorted_data = results_df[metric].sort_values(ascending=False)
#     ax.barh(range(len(sorted_data)), sorted_data.values, color='coral', alpha=0.7)
#     ax.set_yticks(range(len(sorted_data)))
#     ax.set_yticklabels(sorted_data.index, fontsize=7)
#     ax.set_title(f"NEW: {metric.replace('_', ' ').title()}", fontsize=9)
#     ax.grid(axis='x', alpha=0.3)
    
#     if metric == "uncertainty_structure_ratio": 
#         # ax.set_xlim([0,25])
#         ax.set_xscale('log')
        

# # Remove empty subplots
# for idx in range(n_plots, len(axes)):
#     fig.delaxes(axes[idx])

# plt.tight_layout()

# plt.show()

# Print summary
print(results_df[['snr', 'intrinsic_snr', 'mean_pixel_uncertainty']].round(3))



# ====================================================================
# cell 13
# ====================================================================
"""Plot metrics comparing original vs variation-aware metrics"""

variation_metrics = [
                    'mean_pixel_uncertainty', 
                    'intrinsic_snr', 
                    'uncertainty_smoothness', 
                    'high_uncertainty_fraction',
                    # 'confidence_weighted_smoothness', 
                    'signal_vs_gradient_noise',
                    'effective_resolution', 
                    'uncertainty_structure_ratio'
                    ]



variation_metrics_names = {
    'mean_pixel_uncertainty': "Log Mean Pixel Uncertainty\n(lower = better)", 
    'intrinsic_snr': "Intrinsic SNR\n(higher = better)", 
    'uncertainty_smoothness': "Log Uncertainty Smoothness\n(lower = better)", 
    'high_uncertainty_fraction': "High Uncertainty Fraction\n(lower = better)",
    'confidence_weighted_smoothness': "Confidence Weighted Smoothness", 
    'signal_vs_gradient_noise': "Signal vs Gradient Noise\n(higher = better)",
    'effective_resolution': "Effective Resolution\n(higher = better)", 
    'uncertainty_structure_ratio': "Uncertainty Structure Ratio\n(lower = better)"
}


# Check which metrics are available
available_variation = [m for m in variation_metrics if m in results_df.columns]

n_plots = len(available_variation) 
n_cols = 3
n_rows = (n_plots) // n_cols + 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((6*n_cols, 6*n_rows)))
axes = axes.flatten()

# Plot variation metrics
for idx, metric in enumerate(available_variation):
    ax = axes[idx]
    sorted_data = results_df[metric]
    ax.barh(range(len(sorted_data)), sorted_data.values, color=[metric_colors[i] for i in sorted_data.index]) 
    ax.set_title(f"{variation_metrics_names[metric]}") # metric.replace('_', ' ').title()}") 
    ax.grid(axis='x', alpha=0.3)
    
    if metric == "uncertainty_structure_ratio" or metric == "mean_pixel_uncertainty" or metric == "uncertainty_smoothness": 
        # ax.set_xlim([0,25])
        ax.set_xscale('log')
        

# Remove empty subplots
for idx in range(n_plots, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()

plt.savefig(base_path / "qc_appendix.pdf", bbox_inches="tight")
plt.show()

# Print summary
# print(results_df[['snr', 'intrinsic_snr', 'mean_pixel_uncertainty']].round(3))
print(base_path / "qc_appendix.pdf")


# ====================================================================
# cell 14
# ====================================================================
"""Plot metrics comparing original vs variation-aware metrics"""

variation_metrics = [
                    'mean_pixel_uncertainty', 
                    'intrinsic_snr', 
                    # 'uncertainty_smoothness', 
                    'high_uncertainty_fraction',
                    # 'confidence_weighted_smoothness', 
                    # 'signal_vs_gradient_noise',
                    # 'effective_resolution', 
                    # 'uncertainty_structure_ratio'
                    ]



variation_metrics_names = {
    'mean_pixel_uncertainty': "Log Mean Pixel Uncertainty\n(lower = better)", 
    'intrinsic_snr': "Intrinsic SNR\n(higher = better)", 
    'uncertainty_smoothness': "Log Uncertainty Smoothness\n(lower = better)", 
    'high_uncertainty_fraction': "High Uncertainty Fraction\n(lower = better)",
    'confidence_weighted_smoothness': "Confidence Weighted Smoothness", 
    'signal_vs_gradient_noise': "Signal vs Gradient Noise\n(higher = better)",
    'effective_resolution': "Effective Resolution\n(higher = better)", 
    'uncertainty_structure_ratio': "Uncertainty Structure Ratio\n(lower = better)"
}


# Check which metrics are available
available_variation = [m for m in variation_metrics if m in results_df.columns]

n_plots = len(available_variation) 
n_cols = 3
n_rows = (n_plots) // n_cols + 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((6*n_cols, 6*n_rows)))
axes = axes.flatten()

# Plot variation metrics
for idx, metric in enumerate(available_variation):
    ax = axes[idx]
    sorted_data = results_df[metric]
    ax.barh(range(len(sorted_data)), sorted_data.values, color=[metric_colors[i] for i in sorted_data.index]) 
    ax.set_title(f"{variation_metrics_names[metric]}") # metric.replace('_', ' ').title()}") 
    ax.grid(axis='x', alpha=0.3)
    
    ax.set_yticks([])
    
    if metric == "uncertainty_structure_ratio" or metric == "mean_pixel_uncertainty" or metric == "uncertainty_smoothness": 
        # ax.set_xlim([0,25])
        ax.set_xscale('log')
        

# Remove empty subplots
for idx in range(n_plots, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()

plt.savefig(base_path / "qc.pdf", bbox_inches="tight")
plt.show()

# Print summary
# print(results_df[['snr', 'intrinsic_snr', 'mean_pixel_uncertainty']].round(3))
print(base_path / "qc.pdf")


# ====================================================================
# cell 15
# ====================================================================
# Only intrinsic SNR (iSNR)

metric = 'intrinsic_snr'

variation_metrics_names = {
    'intrinsic_snr': "Intrinsic SNR", 
}

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))
ax = fig.add_subplot(111)

sorted_data = results_df[metric]
ax.barh(range(len(sorted_data)), sorted_data.values, color=[metric_colors[i] for i in sorted_data.index]) 
ax.set_xlabel(f"{variation_metrics_names[metric]}") # metric.replace('_', ' ').title()}") 
# ax.grid(axis='x', alpha=0.3)

ax.set_yticks([])
ax.set_xticks([-40, 20])

ax.vlines(0, -1, len(sorted_data), 
          colors='black', 
        #   linestyles='dashed', 
        #   alpha=0.5
          )

ax_width_inch, ax_height_inch = viz.cm_to_inch((6, 6))
ax.set_position([0.15, 0.15, ax_width_inch/(ax_width_inch + 1), 
                ax_height_inch/(ax_height_inch + 1)])

plt.savefig(base_path / "qc_only_iSNR.pdf", bbox_inches="tight")
print(base_path / "qc_only_iSNR.pdf")


# ====================================================================
# cell 16
# ====================================================================
fig = plt.figure(figsize=(8, 1))  # Adjust size as needed
ax = fig.add_subplot(111)

# sorted_data = results_df[metric].sort_values()

# Create legend handles
handles = [patches.Patch(color=metric_colors[name], label=method_names[name]) 
           for name in all_dist_measures[::-1]]

# Add legend
legend = ax.legend(handles=handles, loc='center', 
                   ncol=4, 
                #    fontsize=12, 
                   frameon=False)

# Hide the axes
ax.axis('off')

# plt.tight_layout()

plt.savefig(base_path / "legend.pdf", bbox_inches="tight")
plt.show()
print(base_path / "legend.pdf")

# ====================================================================
# cell 18
# ====================================================================
"""Plot metrics comparing original vs variation-aware metrics"""

variation_metrics = [
                    'mean_pixel_uncertainty', 
                    'intrinsic_snr', 
                    # 'uncertainty_smoothness', 
                    'high_uncertainty_fraction',
                    # 'confidence_weighted_smoothness', 
                    # 'signal_vs_gradient_noise',
                    # 'effective_resolution', 
                    # 'uncertainty_structure_ratio'
                    ]



variation_metrics_names = {
    'mean_pixel_uncertainty': "Log Mean Pixel Uncertainty\n(lower = better)", 
    'intrinsic_snr': "Intrinsic SNR\n(higher = better)", 
    'uncertainty_smoothness': "Log Uncertainty Smoothness\n(lower = better)", 
    'high_uncertainty_fraction': "High Uncertainty Fraction\n(lower = better)",
    'confidence_weighted_smoothness': "Confidence Weighted Smoothness", 
    'signal_vs_gradient_noise': "Signal vs Gradient Noise\n(higher = better)",
    'effective_resolution': "Effective Resolution\n(higher = better)", 
    'uncertainty_structure_ratio': "Uncertainty Structure Ratio\n(lower = better)"
}


# Check which metrics are available
available_variation = [m for m in variation_metrics if m in results_df.columns]

n_plots = len(available_variation) 
n_cols = 3
n_rows = (n_plots) // n_cols + 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=viz.cm_to_inch((6*n_cols, 6*n_rows)))
axes = axes.flatten()

# Plot variation metrics
for idx, metric in enumerate(available_variation):
    ax = axes[idx]
    sorted_data = results_df[metric]
    ax.barh(range(len(sorted_data)), sorted_data.values, color=[metric_colors[i] for i in sorted_data.index]) 
    ax.set_title(f"{variation_metrics_names[metric]}") # metric.replace('_', ' ').title()}") 
    ax.grid(axis='x', alpha=0.3)
    
    ax.set_yticks([])
    
    if metric == "uncertainty_structure_ratio" or metric == "mean_pixel_uncertainty" or metric == "uncertainty_smoothness": 
        # ax.set_xlim([0,25])
        ax.set_xscale('log')
        

# # Remove empty subplots
for idx in range(n_plots, len(axes)):
    fig.delaxes(axes[idx])

# plt.tight_layout()

# plt.savefig(base_path / "qc.pdf", bbox_inches="tight")
# plt.show()

# # Print summary
# # print(results_df[['snr', 'intrinsic_snr', 'mean_pixel_uncertainty']].round(3))
# print(base_path / "qc.pdf")


# ====================================================================
# cell 19
# ====================================================================
timing_data = []
method_labels = []
method_key_names_ordered = []

for method in all_dist_measures:
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
method_key_names_ordered = [all_dist_measures[i] for i in sorted_indices]

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


# plt.tight_layout()

# plt.savefig(output_path / "similarity_methods_timing_comparison.pdf")
# print(output_path / "similarity_methods_timing_comparison.pdf")


# ====================================================================
# cell 20
# ====================================================================
matrices_of_metrics = {}
std_matrices_of_metrics = {}

base_path = (_RD / "output" / "gnm" / dataset_name / experiment_name)

for idx, mode in enumerate(all_dist_measures):

    path = base_path / f"summary_indiv_{mode}_for_exp_{experiment_name}.csv"
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

    # Create pivot tables for BOTH mean and std
    df_pivot_mean = df.pivot_table(index='gamma', columns='eta', 
                                    values=metric_col[0], aggfunc='mean')
    df_pivot_std = df.pivot_table(index='gamma', columns='eta', 
                                   values=metric_col[0], aggfunc='std')
    
    # Normalize mean
    matrix_mean = df_pivot_mean.values[::-1]
    matrix_mean = (matrix_mean - np.nanmin(matrix_mean)) / (np.nanmax(matrix_mean) - np.nanmin(matrix_mean))
    
    # Keep std in original scale (will be normalized later)
    matrix_std = df_pivot_std.values[::-1]
    
    if mode in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:    
        matrix_mean = 1 - matrix_mean
    
    matrices_of_metrics[mode] = matrix_mean
    std_matrices_of_metrics[mode] = matrix_std

# Analyze with variation information
results_df = analyze_all_methods_with_variation(
    matrices_of_metrics, 
    std_matrices_of_metrics
)


# ====================================================================
# cell 21
# ====================================================================
sorted_data

# ====================================================================
# cell 22
# ====================================================================
timings.shape, sorted_data.shape

# ====================================================================
# cell 23
# ====================================================================
for method, label, timings in zip(all_dist_measures, method_labels, timing_data):
    plt.scatter([method]*len(timings), timings, alpha=0.05, color=metric_colors[method], s=1, rasterized=True)

# ====================================================================
# cell 24
# ====================================================================
results = {}
name = "energy"
matrix = matrices_of_metrics[name].copy()

# Get std matrix if available
std_matrix = None
if std_matrices_of_metrics is not None and name in std_matrices_of_metrics:
    std_matrix = std_matrices_of_metrics[name].copy()

# Create mask from NaN positions
mask = np.ones_like(matrix, dtype=bool)
mask[np.isnan(matrix)] = False

# Handle NaNs in both matrices
    # if np.isnan(matrix).any():
    #     if nan_strategy == 'mean':
    #         matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix))
    #         if std_matrix is not None:
    #             std_matrix = np.nan_to_num(std_matrix, nan=np.nanmean(std_matrix))
    #     elif nan_strategy == 'zero':
    #         matrix = np.nan_to_num(matrix, nan=0)
    #         if std_matrix is not None:
    #             std_matrix = np.nan_to_num(std_matrix, nan=0)

# Min-max normalize the mean matrix
matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

# Normalize std matrix proportionally
if std_matrix is not None:
    original_range = matrices_of_metrics[name].max() - matrices_of_metrics[name].min()
    if original_range > 1e-10:
        std_matrix = std_matrix / original_range

analyzer = VisualQualityAnalyzerWithVariation(matrix, std_matrix, mask)
res = analyzer.calculate_snr() # get_all_metrics()
res


# ====================================================================
# cell 28
# ====================================================================

fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

for distance_measure in all_dist_measures:
    # distance_measure = "energy"
    base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    # if mode in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
    ############
    
    # Group by rewire_fraction and calculate mean and std
    grouped = df.groupby('rewire_fraction').agg({
        'computation_time': 'median', # 'mean',
        'metric_value': ['mean', 'std']
    }).reset_index()
    
    
    # get sigma_M for iSNR 
    df2 = df.groupby("rewire_fraction")["metric_value"].mean()
    sigma_m = df2.values.std()

    # Flatten the column names
    grouped.columns = ['rewire_fraction', 'computation_time_median', 'metric_value_mean', 'metric_value_std']

    # Calculate std^2 / mean^2 (coefficient of variation squared)
    grouped['std2_over_mean2'] = (grouped['metric_value_std']**2) / (grouped['metric_value_mean']**2)
    grouped['cv'] = (grouped['metric_value_std']) / (grouped['metric_value_mean'])

    plt.scatter(grouped['computation_time_median'], 
                grouped['cv'], # std2_over_mean2'], 
                c=np.ones(shape=(grouped['cv'].shape[0], 3)) * metric_colors[distance_measure],
                s=10, 
                alpha=0.2, 
                # edgecolors='black', 
                # linewidth=1.5
                )

    plt.scatter(grouped['computation_time_median'].median(), 
                grouped['cv'].mean(), # std2_over_mean2'], 
                c=metric_colors[distance_measure],
                s=30, 
                alpha=1, # 0.2, 
                edgecolors='black', 
                # linewidth=1.5
                )
    
    
    plt.ylim([0, 0.2])
    plt.xticks([0, 0.04, 0.08])
    plt.yticks([0, 0.1, 0.2])
    
    ax_width_inch, ax_height_inch = viz.cm_to_inch((6, 6))
    # fig.set_size_inches(ax_width_inch + 1, ax_height_inch + 1)  # Add margin for labels
    ax.set_position([0.15, 0.15, ax_width_inch/(ax_width_inch + 1), 
                    ax_height_inch/(ax_height_inch + 1)])

    plt.xlabel("Computation Time") # Average Computation Time (seconds)")
    plt.ylabel("CV\n(lower is better)") # Coefficient of Variation (std/mean)")
    
plt.savefig(base_path / "chaos_analysis_cv_vs_time.pdf", bbox_inches="tight")
print(base_path / "chaos_analysis_cv_vs_time.pdf")

# ====================================================================
# cell 32
# ====================================================================
base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"
csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
df = pd.read_csv(csv_path)

plt.figure(dpi=50) 
# # Group by rewire_fraction to get the groups, but don't aggregate yet
# grouped = df.groupby('rewire_fraction')


for i in df["process_id"].unique():

    x_vals = df[df["process_id"] == i]["rewire_fraction"].values
    y_vals = df[df["process_id"] == i]["metric_value"].values
    # normalize y_vals
    y_vals = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())

    # subtract the diagonal
    y_vals = y_vals - x_vals / np.max(x_vals)


    plt.plot( # x_vals, 
                y_vals, 
                c=metric_colors[distance_measure])

# ====================================================================
# cell 33
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict = {}

for distance_measure in all_dist_measures:
    
    base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    print(df.keys())
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
    ############


    # Group by rewire_fraction to get the groups, but don't aggregate yet
    grouped = df.groupby('rewire_fraction')

    x_vals_arr = []
    y_vals_arr = []
    print("Number processes:", len(df["process_id"].unique()))
    for i in df["process_id"].unique():

        x_vals = df[df["process_id"] == i]["rewire_fraction"].values
        y_vals = df[df["process_id"] == i]["metric_value"].values
        # normalize y_vals
        y_vals = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min()) 

        
        # if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        #     y_vals = 1 - y_vals

        # subtract the diagonal
        y_vals = y_vals - x_vals / np.max(x_vals)

        # Take the absolute 
        # y_vals = np.abs(y_vals)
        
        # sensitivity = 1 / y_vals.mean()

        # print(len(df[df["process_id"] == i]["computation_time"]), len(y_vals))
        plt.scatter(df[df["process_id"] == i]["computation_time"].median(), # mean(), #  # x_vals,
                    y_vals.mean(),
                    c=metric_colors[distance_measure], 
                    s=10, 
                    alpha=0.2, 
                    )
        print("number scatter points:", len(y_vals))
        # plt.plot(y_vals)
        x_vals_arr.append(df[df["process_id"] == i]["computation_time"].median()) # mean())
        y_vals_arr.append(y_vals.mean())

    mean_points_dict[distance_measure] = (np.median(np.array(x_vals_arr)), # .median(), # mean(), 
                                          np.array(y_vals_arr).mean())
    
    # plt.title(distance_measure)
    # plt.show()
        
for distance_measure in all_dist_measures:
    x_vals_arr, y_vals_arr = mean_points_dict[distance_measure]
    plt.scatter(mean_points_dict[distance_measure][0],
                mean_points_dict[distance_measure][1],
                c=metric_colors[distance_measure],
                s=30, 
                alpha=1, # 0.2, 
                edgecolors='black', 
                # linewidth=1.5
                )
    print(distance_measure, ": ", mean_points_dict[distance_measure][1])
# plt.xlim([0, 0.15])
# plt.ylim([0.2, 0.4])

# plt.xticks([0, 0.15])
plt.yticks([0.0, 0.2, 0.4])
# plt.yticks([0.25, 0.35, 0.45])
# plt.xticks([0, 0.1, 0.2])
# plt.yticks([0.15, 0.25, 0.35])

# plt.xscale("log")
# plt.yscale("log")

ax_width_inch, ax_height_inch = viz.cm_to_inch((6, 6))
# fig.set_size_inches(ax_width_inch + 1, ax_height_inch + 1)  # Add margin for labels
ax.set_position([0.15, 0.15, ax_width_inch/(ax_width_inch + 1), 
                ax_height_inch/(ax_height_inch + 1)])

plt.xlabel("Computation Time") # Average Computation Time (seconds)")
plt.ylabel("MAE\n(lower is better)") # Mean Absolute Error

plt.savefig(base_path / "mae_vs_time_scatter.pdf", bbox_inches="tight", dpi=300) # rasterize=True)
print(base_path / "mae_vs_time_scatter.pdf")



# ====================================================================
# cell 34
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict = {}

for distance_measure in all_dist_measures:
    
    base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    print(df.keys())
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
    ############


    # Group by rewire_fraction to get the groups, but don't aggregate yet
    grouped = df.groupby('rewire_fraction')

    x_vals_arr = []
    y_vals_arr = []
    for i in df["process_id"].unique():

        x_vals = df[df["process_id"] == i]["rewire_fraction"].values
        y_vals = df[df["process_id"] == i]["metric_value"].values
        # normalize y_vals
        y_vals = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min()) 

        
        # if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        #     y_vals = 1 - y_vals

        # subtract the diagonal
        y_vals = y_vals - x_vals / np.max(x_vals)

        # Take the absolute 
        # y_vals = np.abs(y_vals)
        
        # sensitivity = 1 / y_vals.mean()

        # print(len(df[df["process_id"] == i]["computation_time"]), len(y_vals))
        # plt.scatter(df[df["process_id"] == i]["computation_time"].median(), # mean(), #  # x_vals,
        #             y_vals.mean(),
        #             c=metric_colors[distance_measure], 
        #             s=10, 
        #             alpha=0.2, 
        #             )
        # plt.plot(y_vals)
        x_vals_arr.append(df[df["process_id"] == i]["computation_time"].median()) # mean())
        y_vals_arr.append(y_vals) # .mean())

    y_vals_mean = np.array(y_vals_arr).mean(axis=0)
    mae = np.mean(np.abs(y_vals_mean))
    
    # print(y_vals_mean.shape)
    # plt.plot(y_vals_mean, c=metric_colors[distance_measure])
    # .median(), # mean(),
    # mean_points_dict[distance_measure] = (np.median(np.array(x_vals_arr)), # .median(), # mean(), 
    #                                       np.array(y_vals_arr).mean())
    
    # plt.title(distance_measure)
    # plt.show()
        
# for distance_measure in all_dist_measures:
    # x_vals_arr, y_vals_arr = mean_points_dict[distance_measure]
    # plt.scatter(mean_points_dict[distance_measure][0],
    #             mean_points_dict[distance_measure][1],
    #             c=metric_colors[distance_measure],
    #             s=30, 
    #             alpha=1, # 0.2, 
    #             edgecolors='black', 
    #             # linewidth=1.5
    #             )

    plt.scatter(np.median(np.array(x_vals_arr)), # ^mean_points_dict[distance_measure][0],
                mae, # mean_points_dict[distance_measure][1],
                c=metric_colors[distance_measure],
                s=30, 
                alpha=1, # 0.2, 
                edgecolors='black', 
                # linewidth=1.5
                )
    

# plt.xlim([0, 0.15])
# plt.ylim([0.2, 0.4])

# plt.xticks([0, 0.15])
plt.xticks([0, 0.04, 0.08])
plt.yticks([0.0, 0.2, 0.4])
# plt.yticks([0.25, 0.35, 0.45])
# plt.xticks([0, 0.1, 0.2])
# plt.yticks([0.15, 0.25, 0.35])

# plt.xscale("log")
# plt.yscale("log")

ax_width_inch, ax_height_inch = viz.cm_to_inch((6, 6))
# fig.set_size_inches(ax_width_inch + 1, ax_height_inch + 1)  # Add margin for labels
ax.set_position([0.15, 0.15, ax_width_inch/(ax_width_inch + 1), 
                ax_height_inch/(ax_height_inch + 1)])

plt.xlabel("Computation Time") # Average Computation Time (seconds)")
plt.ylabel("MAE\n(lower is better)") # Mean Absolute Error

plt.savefig(base_path / "mae_vs_time.pdf", bbox_inches="tight", dpi=300) # rasterize=True)
print(base_path / "mae_vs_time.pdf")



# ====================================================================
# cell 36
# ====================================================================

base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"

hamming_df = pd.read_csv(f"{base_dir_dataset}/chaos_analysis_hamming.csv")
hamming_df['metric_value'] = (hamming_df['metric_value'] - hamming_df['metric_value'].min()) / (hamming_df['metric_value'].max() - hamming_df['metric_value'].min())
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})


# join hamming_df and df on the index 
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})
merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)

# sort by hamming_metric_value 
merged_df = merged_df.sort_values("hamming_metric_value")
plt.scatter(merged_df["hamming_metric_value"], merged_df["metric_value"]-merged_df["hamming_metric_value"]) 
# plt.scatter(merged_df["hamming_metric_value"], np.abs(merged_df["metric_value"]-merged_df["hamming_metric_value"]))
merged_df



# ====================================================================
# cell 39
# ====================================================================
# join hamming_df and df on the index 
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})
merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)
merged_df

# ====================================================================
# cell 40
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict_2 = {}

base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"

hamming_df = pd.read_csv(f"{base_dir_dataset}/chaos_analysis_hamming.csv")
hamming_df['metric_value'] = (hamming_df['metric_value'] - hamming_df['metric_value'].min()) / (hamming_df['metric_value'].max() - hamming_df['metric_value'].min())
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})

# Group by rewire_fraction to get the groups, but don't aggregate yet
grouped_hamming = hamming_df.groupby('rewire_fraction').mean()
print(grouped_hamming.keys())

for distance_measure in all_dist_measures:
    
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    # print(df.keys())
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
        
    # Group by rewire_fraction to get the groups, but don't aggregate yet
    # grouped = df.groupby('rewire_fraction')
    merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)
    merged_df["mae"] = np.abs(merged_df['metric_value'] - merged_df["hamming_metric_value"])
    # print(merged_df.keys())
    grouped_merged_df = merged_df.groupby(["process_id_x"]).median() # step_x
    print(grouped_merged_df.keys())
    print(grouped_merged_df)
    ############

    # plt.scatter(merged_df['hamming_metric_value'], merged_df['metric_value'], label=distance_measure)
    plt.scatter(grouped_merged_df.index, # ['computation_time'], 
                grouped_merged_df['mae'], 
                c=metric_colors[distance_measure], 
                s=10, 
                alpha=0.2,
                label=distance_measure)

    # plt.scatter(grouped_merged_df['computation_time'], # rewire_fraction_y'], # computation_time'], step_x
    #             grouped_merged_df['metric_value'], # mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=10, 
    #             alpha=0.2,
    #             label=distance_measure)
    
    mean_points_dict_2[distance_measure] = (np.median(np.array(grouped_merged_df['computation_time'])), # .median(), # mean(), 
                                          np.median(np.array(grouped_merged_df['mae'])))
    # plt.show()
    # plt.scatter(hamming_df['metric_value'], df['metric_value']) #  label="hamming")

    # x_vals_arr = []
    # y_vals_arr = []
    # print("Number processes:", len(df["process_id"].unique()))
    # for i in df["process_id"].unique():

    #     x_vals = df[df["process_id"] == i]["rewire_fraction"].values
    #     y_vals = df[df["process_id"] == i]["metric_value"].values
    #     # normalize y_vals
    #     y_vals = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min()) 

        
    #     # if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
    #     #     y_vals = 1 - y_vals

    #     # subtract the diagonal
    #     y_vals = y_vals - x_vals / np.max(x_vals)

    #     # Take the absolute 
    #     # y_vals = np.abs(y_vals)
        
    #     # sensitivity = 1 / y_vals.mean()

    #     # print(len(df[df["process_id"] == i]["computation_time"]), len(y_vals))
    #     plt.scatter(df[df["process_id"] == i]["computation_time"].median(), # mean(), #  # x_vals,
    #                 y_vals.mean(),
    #                 c=metric_colors[distance_measure], 
    #                 s=10, 
    #                 alpha=0.2, 
    #                 )
    #     print("number scatter points:", len(y_vals))
    #     # plt.plot(y_vals)
    #     x_vals_arr.append(df[df["process_id"] == i]["computation_time"].median()) # mean())
    #     y_vals_arr.append(y_vals.mean())

    # mean_points_dict[distance_measure] = (np.median(np.array(x_vals_arr)), # .median(), # mean(), 
    #                                       np.array(y_vals_arr).mean())
    
    # # plt.title(distance_measure)
    # # plt.show()
    
    # break
        
for distance_measure in all_dist_measures:
    x_vals_arr, y_vals_arr = mean_points_dict_2[distance_measure]
    plt.scatter(mean_points_dict_2[distance_measure][0],
                mean_points_dict_2[distance_measure][1],
                c=metric_colors[distance_measure],
                s=30, 
                alpha=1, # 0.2, 
                edgecolors='black', 
                # linewidth=1.5
                )
    print(distance_measure, ": ", mean_points_dict_2[distance_measure][1])
    # break

# 
# plt.yticks([0.0, 0.2, 0.4])

# ax_width_inch, ax_height_inch = viz.cm_to_inch((6, 6))
# # fig.set_size_inches(ax_width_inch + 1, ax_height_inch + 1)  # Add margin for labels
# ax.set_position([0.15, 0.15, ax_width_inch/(ax_width_inch + 1), 
#                 ax_height_inch/(ax_height_inch + 1)])

plt.xlabel("Computation Time") # Average Computation Time (seconds)")
plt.ylabel("MAE\n(lower is better)") # Mean Absolute Error

# plt.savefig(base_path / "mae_nonlin_vs_time_scatter.pdf", bbox_inches="tight", dpi=300) # rasterize=True)
# print(base_path / "mae_nonlin_vs_time_scatter.pdf")
print(len(merged_df["process_id_x"].unique()))
print(len(merged_df["step_x"].unique()))


# ====================================================================
# cell 41
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict_2 = {}

base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"

hamming_df = pd.read_csv(f"{base_dir_dataset}/chaos_analysis_hamming.csv")
hamming_df['metric_value'] = (hamming_df['metric_value'] - hamming_df['metric_value'].min()) / (hamming_df['metric_value'].max() - hamming_df['metric_value'].min())
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})

# Group by rewire_fraction to get the groups, but don't aggregate yet
# grouped_hamming = hamming_df.groupby('rewire_fraction').mean()
# print(grouped_hamming.keys())

for distance_measure in all_dist_measures:
    
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    # print(df.keys())
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
        
    # Group by rewire_fraction to get the groups, but don't aggregate yet
    # grouped = df.groupby('rewire_fraction')
    merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)
    merged_df["mae"] = np.abs(merged_df['metric_value'] - merged_df["hamming_metric_value"])
    # print(merged_df.keys())
    grouped_merged_df = merged_df.groupby(["process_id_x"]).median() # step_x
    print(grouped_merged_df.keys())
    print(grouped_merged_df)
    # ############

    # plt.scatter(merged_df['hamming_metric_value'], merged_df['metric_value'], label=distance_measure)
    plt.scatter(grouped_merged_df.index, # ['computation_time'], 
                grouped_merged_df['mae'], 
                c=metric_colors[distance_measure], 
                s=10, 
                alpha=0.2,
                label=distance_measure)

    mean_points_dict_2[distance_measure] = (np.median(np.array(grouped_merged_df['computation_time'])), # .median(), # mean(), 
                                          np.median(np.array(grouped_merged_df['mae'])))
    

plt.xlabel("Computation Time") # Average Computation Time (seconds)")
plt.ylabel("MAE\n(lower is better)") # Mean Absolute Error

# plt.savefig(base_path / "mae_nonlin_vs_time_scatter.pdf", bbox_inches="tight", dpi=300) # rasterize=True)
# print(base_path / "mae_nonlin_vs_time_scatter.pdf")
print(len(merged_df["process_id_x"].unique()))
print(len(merged_df["step_x"].unique()))


# ====================================================================
# cell 42
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict_2 = {}

base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"

hamming_df = pd.read_csv(f"{base_dir_dataset}/chaos_analysis_hamming.csv")
hamming_df['metric_value'] = (hamming_df['metric_value'] - hamming_df['metric_value'].min()) / (hamming_df['metric_value'].max() - hamming_df['metric_value'].min())
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})

# Group by rewire_fraction to get the groups, but don't aggregate yet
grouped_hamming = hamming_df.groupby('rewire_fraction').mean()
print(grouped_hamming.keys())

for distance_measure in all_dist_measures:
    
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    # print(df.keys())
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
        
    # Group by rewire_fraction to get the groups, but don't aggregate yet
    # # grouped = df.groupby('rewire_fraction')
    merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)
    merged_df["mae"] = np.abs(merged_df['metric_value'] - merged_df["hamming_metric_value"])
    # # print(merged_df.keys())
    # grouped_merged_df = merged_df.groupby(["hamming_metric_value"]).median() # process_id_x"]).median() # step_x
    # # print(grouped_merged_df.keys())
    # print(grouped_merged_df)
    
    # join hamming_df and df on the index 
    hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})
    merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)

    # sort by hamming_metric_value 
    merged_df = merged_df.sort_values("hamming_metric_value")
    plt.scatter(merged_df["hamming_metric_value"], 
                merged_df["metric_value"], # -merged_df["hamming_metric_value"], 
                s=1) 
    # plt.scatter(merged_df["hamming_metric_value"], np.abs(merged_df["metric_value"]-merged_df["hamming_metric_value"]))
    merged_df




    ############

    # plt.scatter(merged_df['hamming_metric_value'], merged_df['metric_value'], label=distance_measure)
    # plt.scatter(grouped_merged_df.index, # ['computation_time'], 
    #             grouped_merged_df['mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=10, 
    #             alpha=0.2,
    #             label=distance_measure)

    # plt.scatter(grouped_merged_df['hamming_metric_value'], # .index, # ['computation_time'], 
    #             grouped_merged_df['metric_value'], # mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=10, 
    #             alpha=0.2,
    #             label=distance_measure)
    
    # plt.scatter(grouped_merged_df.index, # ['hamming_metric_value'], # .index, # ['computation_time'], 
    #             grouped_merged_df['mae'], # metric_value'], # mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=2, 
    #             alpha=0.2,
    #             label=distance_measure)




# ====================================================================
# cell 43
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict_2 = {}

base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"

hamming_df = pd.read_csv(f"{base_dir_dataset}/chaos_analysis_hamming.csv")
hamming_df['metric_value'] = (hamming_df['metric_value'] - hamming_df['metric_value'].min()) / (hamming_df['metric_value'].max() - hamming_df['metric_value'].min())
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})

# Group by rewire_fraction to get the groups, but don't aggregate yet
grouped_hamming = hamming_df.groupby('rewire_fraction').mean()
print(grouped_hamming.keys())

for distance_measure in all_dist_measures:
    
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    # print(df.keys())
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
        
    # Group by rewire_fraction to get the groups, but don't aggregate yet
    # # grouped = df.groupby('rewire_fraction')
    merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)
    merged_df["mae"] = np.abs(merged_df['metric_value'] - merged_df["hamming_metric_value"])
    # # print(merged_df.keys())
    # grouped_merged_df = merged_df.groupby(["hamming_metric_value"]).median() # process_id_x"]).median() # step_x
    # # print(grouped_merged_df.keys())
    # print(grouped_merged_df)
    
    # join hamming_df and df on the index 
    hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})
    merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)

    # sort by hamming_metric_value 
    merged_df = merged_df.sort_values("hamming_metric_value")
    merged_df_gr = merged_df.groupby("hamming_metric_value").mean()
    
    # plt.scatter(merged_df_gr.index, # merged_df["hamming_metric_value"], 
    #             merged_df_gr["metric_value"], # -merged_df_gr.index,  # -merged_df["hamming_metric_value"], 
    #             s=1) 
    # merged_df
    
    
    plt.scatter(merged_df_gr["computation_time"], # .index, # merged_df["hamming_metric_value"], 
                np.abs(merged_df_gr["metric_value"]-merged_df_gr.index),  # -merged_df["hamming_metric_value"], 
                s=1) 
    print("Number dots: ", len(merged_df_gr))




    ############

    # plt.scatter(merged_df['hamming_metric_value'], merged_df['metric_value'], label=distance_measure)
    # plt.scatter(grouped_merged_df.index, # ['computation_time'], 
    #             grouped_merged_df['mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=10, 
    #             alpha=0.2,
    #             label=distance_measure)

    # plt.scatter(grouped_merged_df['hamming_metric_value'], # .index, # ['computation_time'], 
    #             grouped_merged_df['metric_value'], # mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=10, 
    #             alpha=0.2,
    #             label=distance_measure)
    
    # plt.scatter(grouped_merged_df.index, # ['hamming_metric_value'], # .index, # ['computation_time'], 
    #             grouped_merged_df['mae'], # metric_value'], # mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=2, 
    #             alpha=0.2,
    #             label=distance_measure)




# ====================================================================
# cell 44
# ====================================================================
len(merged_df["hamming_metric_value"].unique())

# ====================================================================
# cell 45
# ====================================================================
plt.scatter(hamming_df["step"], hamming_df["hamming_metric_value"]) # rewire_fraction"]) # process_id: gerade (20,000), step + rewire_fraction_x: schnelles zigzack. 
# 101. 

# ====================================================================
# cell 46
# ====================================================================
hamming_df_g = hamming_df.groupby("step").median()
plt.scatter(hamming_df_g.index, hamming_df_g["hamming_metric_value"])
# hamming_df_g.keys() # ['process_id', 'rewire_fraction', 'hamming_metric_value', 'hamming_computation_time'],

# ====================================================================
# cell 47
# ====================================================================
print("df: ")
print(df)
print("hamming_df: ")
print(hamming_df)

# ====================================================================
# cell 48
# ====================================================================
df_g = df.groupby("step").median()
plt.scatter(df_g.index, df_g["metric_value"])
plt.scatter(df_g.index, df_g["metric_value"]-hamming_df_g["hamming_metric_value"])

# ====================================================================
# cell 49
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict_2 = {}

base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"

hamming_df = pd.read_csv(f"{base_dir_dataset}/chaos_analysis_hamming.csv")
hamming_df['metric_value'] = (hamming_df['metric_value'] - hamming_df['metric_value'].min()) / (hamming_df['metric_value'].max() - hamming_df['metric_value'].min())
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})

# Group by rewire_fraction to get the groups, but don't aggregate yet
grouped_hamming = hamming_df.groupby('rewire_fraction').mean()
print(grouped_hamming.keys())

for distance_measure in all_dist_measures:
    
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    # print(df.keys())
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
        
    df_g = df.groupby("step").median()
    # plt.scatter(df_g.index, df_g["metric_value"])
    plt.scatter(df_g.index, df_g["metric_value"]-hamming_df_g["hamming_metric_value"])
    # # Group by rewire_fraction to get the groups, but don't aggregate yet
    # # grouped = df.groupby('rewire_fraction')
    # merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)
    # merged_df["mae"] = np.abs(merged_df['metric_value'] - merged_df["hamming_metric_value"])
    # # print(merged_df.keys())
    # grouped_merged_df = merged_df.groupby(["hamming_metric_value"]).median() # process_id_x"]).median() # step_x
    # # print(grouped_merged_df.keys())
    # print(grouped_merged_df)
    # ############

    # plt.scatter(merged_df['hamming_metric_value'], merged_df['metric_value'], label=distance_measure)
    # plt.scatter(grouped_merged_df.index, # ['computation_time'], 
    #             grouped_merged_df['mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=10, 
    #             alpha=0.2,
    #             label=distance_measure)

    # plt.scatter(grouped_merged_df['hamming_metric_value'], # .index, # ['computation_time'], 
    #             grouped_merged_df['metric_value'], # mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=10, 
    #             alpha=0.2,
    #             label=distance_measure)
    
    plt.scatter(grouped_merged_df["computation_time"], # ['hamming_metric_value'], # .index, # ['computation_time'], 
                grouped_merged_df['mae'], # metric_value'], # mae'], 
                c=metric_colors[distance_measure], 
                s=2, 
                alpha=0.2,
                label=distance_measure)


# ====================================================================
# cell 50
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict_2 = {}

base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"

hamming_df = pd.read_csv(f"{base_dir_dataset}/chaos_analysis_hamming.csv")
hamming_df['metric_value'] = (hamming_df['metric_value'] - hamming_df['metric_value'].min()) / (hamming_df['metric_value'].max() - hamming_df['metric_value'].min())
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})

# Group by rewire_fraction to get the groups, but don't aggregate yet
grouped_hamming = hamming_df.groupby('rewire_fraction').mean()
print(grouped_hamming.keys())

for distance_measure in all_dist_measures:
    
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    # print(df.keys())
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
        
    # Group by rewire_fraction to get the groups, but don't aggregate yet
    # grouped = df.groupby('rewire_fraction')
    merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)
    merged_df["mae"] = np.abs(merged_df['metric_value'] - merged_df["hamming_metric_value"])
    # print(merged_df.keys())
    grouped_merged_df = merged_df.groupby(["hamming_metric_value"]).median() # process_id_x"]).median() # step_x
    # print(grouped_merged_df.keys())
    print(grouped_merged_df)
    ############

    # plt.scatter(merged_df['hamming_metric_value'], merged_df['metric_value'], label=distance_measure)
    # plt.scatter(grouped_merged_df.index, # ['computation_time'], 
    #             grouped_merged_df['mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=10, 
    #             alpha=0.2,
    #             label=distance_measure)

    # plt.scatter(grouped_merged_df['hamming_metric_value'], # .index, # ['computation_time'], 
    #             grouped_merged_df['metric_value'], # mae'], 
    #             c=metric_colors[distance_measure], 
    #             s=10, 
    #             alpha=0.2,
    #             label=distance_measure)
    
    plt.scatter(grouped_merged_df["computation_time"], # ['hamming_metric_value'], # .index, # ['computation_time'], 
                grouped_merged_df['mae'], # metric_value'], # mae'], 
                c=metric_colors[distance_measure], 
                s=2, 
                alpha=0.2,
                label=distance_measure)




# ====================================================================
# cell 51
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict_2 = {}

base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"

# Get hamming data (effective distnace) 
hamming_df = pd.read_csv(f"{base_dir_dataset}/chaos_analysis_hamming.csv")
hamming_df['metric_value'] = (hamming_df['metric_value'] - hamming_df['metric_value'].min()) / (hamming_df['metric_value'].max() - hamming_df['metric_value'].min())
hamming_df = hamming_df.rename(columns={"metric_value": "hamming_metric_value", "computation_time": "hamming_computation_time"})


for distance_measure in all_dist_measures:
    
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)
    
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
        
    merged_df = pd.merge(df, hamming_df, left_index=True, right_index=True)
    merged_df["mae"] = np.abs(merged_df['metric_value'] - merged_df["hamming_metric_value"])

    # Group by effective distance (hamming distance)
    grouped_merged_df = merged_df.groupby(["hamming_metric_value"]).median() 
    print(len(grouped_merged_df))
    plt.scatter(grouped_merged_df["computation_time"], # ['hamming_metric_value'], # .index, # ['computation_time'], 
                grouped_merged_df['mae'], # metric_value'], # mae'], 
                c=metric_colors[distance_measure], 
                s=2, 
                alpha=0.2,
                label=distance_measure)
    
    mean_points_dict[distance_measure] = (np.median(np.array(grouped_merged_df["computation_time"])), # .median(), # mean(), 
                                          np.median(np.array(grouped_merged_df["mae"])))
    
        
for distance_measure in all_dist_measures:
    x_vals_arr, y_vals_arr = mean_points_dict[distance_measure]
    plt.scatter(mean_points_dict[distance_measure][0],
                mean_points_dict[distance_measure][1],
                c=metric_colors[distance_measure],
                s=30, 
                alpha=1, # 0.2, 
                edgecolors='black', 
                )
    print(distance_measure, ": ", mean_points_dict[distance_measure][1])


plt.yticks([0.0, 0.2, 0.4])
plt.xticks([0, 0.04, 0.08])

ax_width_inch, ax_height_inch = viz.cm_to_inch((6, 6))
# fig.set_size_inches(ax_width_inch + 1, ax_height_inch + 1)  # Add margin for labels
ax.set_position([0.15, 0.15, ax_width_inch/(ax_width_inch + 1), 
                ax_height_inch/(ax_height_inch + 1)])

plt.xlabel("Computation Time") # Average Computation Time (seconds)")
plt.ylabel("MAE\n(lower is better)") # Mean Absolute Error

plt.savefig(base_path / "mae_vs_time_scatter.pdf", bbox_inches="tight", dpi=300) # rasterize=True)
print(base_path / "mae_vs_time_scatter.pdf")



# ====================================================================
# cell 54
# ====================================================================
base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"
csv_path = f"{base_dir_dataset}/chaos_analysis_{"energy"}.csv"
df = pd.read_csv(csv_path)

grouped = df.groupby('rewire_fraction').agg({
    'computation_time': 'median',
    'metric_value': ['mean', 'std']
}).reset_index()

grouped

# ====================================================================
# cell 55
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict = {}

for distance_measure in all_dist_measures:
    
    base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)

    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    # if mode in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
    ############

    # Group by rewire_fraction to get the groups, but don't aggregate yet
    grouped = df.groupby('rewire_fraction').agg({
        'computation_time': 'median', # mean',
        'metric_value': ['mean', 'std']
    }).reset_index()


    grouped["cv"] = grouped["metric_value"]["std"] / grouped["metric_value"]["mean"]

    x_vals_arr = []
    y_vals_arr = []


    plt.scatter(grouped["computation_time"]["median"], 
                grouped["cv"], 
                s=10, 
                alpha=0.2, 
                c=metric_colors[distance_measure], 
                )

    mean_points_dict[distance_measure] = (grouped["computation_time"]["median"].mean(), grouped["cv"].mean())


    if distance_measure == "resistance" or distance_measure == "spectral_distance_adjacency":
        print(f"{distance_measure}:")
        
        valid = ~np.isnan(grouped["cv"].values)
        valid_indices = np.where(valid)[0]

        top3_indices = valid_indices[np.argsort(grouped["cv"].values[valid])[-3:]]

        for i in top3_indices:
            print(f"    cv: {grouped.iloc[i]['cv'].values[0]}, computation time: {grouped.iloc[i]['computation_time'].values[0]}")


        
for distance_measure in all_dist_measures:
    ax.scatter(mean_points_dict[distance_measure][0],
                mean_points_dict[distance_measure][1],
                c=metric_colors[distance_measure],
                s=30, 
                alpha=1, # 0.2, 
                edgecolors='black', 
                # linewidth=1.5
                )

# set x to log
# ax.set_xscale("log")
# ax.set_yscale("log")

# plt.xlim([0, 0.2])
# ax.set_ylim([0, 0.3])

ax.set_xticks([0, 0.04, 0.08])
ax.set_yticks([0, 0.15, 0.3])



ax_width_inch, ax_height_inch = viz.cm_to_inch((6, 6))
# fig.set_size_inches(ax_width_inch + 1, ax_height_inch + 1)  # Add margin for labels
ax.set_position([0.15, 0.15, ax_width_inch/(ax_width_inch + 1), 
                ax_height_inch/(ax_height_inch + 1)])

plt.xlabel("Computation Time") # Average Computation Time (seconds)")
plt.ylabel("CV\n(lower is better)") # Coefficient of Variation (std/mean)")

plt.savefig(base_path / "cv_vs_time.pdf", bbox_inches="tight", dpi=300) # rasterize=True)
print(base_path / "cv_vs_time.pdf")



# ====================================================================
# cell 56
# ====================================================================
fig = plt.figure(figsize=viz.cm_to_inch((6, 6)))
ax = fig.add_subplot(111)

mean_points_dict = {}

for distance_measure in all_dist_measures:
    
    base_dir_dataset = f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/{dataset_name}/chaos_analysis"
    csv_path = f"{base_dir_dataset}/chaos_analysis_{distance_measure}.csv"
    df = pd.read_csv(csv_path)

    ############
    # normalize metric_value per distance measure
    df['metric_value'] = (df['metric_value'] - df['metric_value'].min()) / (df['metric_value'].max() - df['metric_value'].min())
    
    # Turn similarities into distances 
    # if mode in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
    if distance_measure in ["f1", "multiplex_layer_similarity", "jaccard", "communicability_corr"]  # communicability JSD is a divergence, i.e. already a distance:
        df['metric_value'] = 1 - df['metric_value']
    ############
    
    # Group by rewire_fraction to get the groups, but don't aggregate yet
    grouped = df.groupby('rewire_fraction').agg({
        'computation_time': 'median', # mean',
        'metric_value': ['mean', 'std']
    }).reset_index()


    grouped["cv"] = grouped["metric_value"]["std"] / grouped["metric_value"]["mean"]

    x_vals_arr = []
    y_vals_arr = []


    plt.scatter(grouped["computation_time"]["median"], 
                grouped["cv"], 
                s=2, 
                alpha=0.2, 
                c=[metric_colors[distance_measure]] * len(grouped["cv"])
                )

    mean_points_dict[distance_measure] = (grouped["computation_time"]["median"].mean(), grouped["cv"].mean())

    print(f"With that, I excluded the high CV values above 0.3, which are (for {distance_measure}, it is also possible that ALL dots are under 0.3...): ")
    dropped_indices = np.where(grouped['cv'].values > 0.3)[0]
    for i in dropped_indices:
        print(f"     cv: {grouped['cv'].values[i]}, computation time: {grouped['computation_time'].values[i]}")
        

    if distance_measure == "resistance" or distance_measure == "spectral_distance_adjacency":
        # print(f"{distance_measure}:")
        
        valid = ~np.isnan(grouped["cv"].values)
        valid_indices = np.where(valid)[0]

        top3_indices = valid_indices[np.argsort(grouped["cv"].values[valid])[-3:]]

        # for i in top3_indices:
        #     print(f"    cv: {grouped.iloc[i]['cv'].values[0]}, computation time: {grouped.iloc[i]['computation_time'].values[0]}")


        
for distance_measure in all_dist_measures:
    ax.scatter(mean_points_dict[distance_measure][0],
                mean_points_dict[distance_measure][1],
                color=metric_colors[distance_measure], #  * np.ones_like(mean_points_dict[distance_measure][1]),
                s=30, 
                alpha=1, # 0.2, 
                edgecolors='black', 
                # linewidth=1.5
                )
    print(distance_measure, ": ", mean_points_dict[distance_measure][1])

# set x to log
# ax.set_xscale("log")
# ax.set_yscale("log")

# plt.xlim([0, 0.2])



ax.set_ylim([0, 0.3])
ax.set_xticks([0, 0.04, 0.08])
ax.set_yticks([0, 0.15, 0.3])



ax_width_inch, ax_height_inch = viz.cm_to_inch((6, 6))
# fig.set_size_inches(ax_width_inch + 1, ax_height_inch + 1)  # Add margin for labels
ax.set_position([0.15, 0.15, ax_width_inch/(ax_width_inch + 1), 
                ax_height_inch/(ax_height_inch + 1)])

plt.xlabel("Computation Time") # Average Computation Time (seconds)")
plt.ylabel("CV\n(lower is better)") # Coefficient of Variation (std/mean)")

plt.savefig(base_path / "cv_vs_time_ylimited.pdf", bbox_inches="tight", dpi=300) # rasterize=True)
print(base_path / "cv_vs_time_ylimited.pdf")



# ====================================================================
# cell 57
# ====================================================================

# Only intrinsic SNR (iSNR)

metric = 'intrinsic_snr'

variation_metrics_names = {
    'intrinsic_snr': "Intrinsic SNR", 
}

fig = plt.figure(figsize=viz.cm_to_inch((6,6)))
ax = fig.add_subplot(111)

# Horizontal line at y=0
ax.hlines(0, 0, 0.08, 
          colors='black', 
          linestyles='dashed', 
          alpha=0.5
          )


sorted_data = results_df[metric]
# ax.barh(range(len(sorted_data)), sorted_data.values, color=[metric_colors[i] for i in sorted_data.index]) 

for distance_measure in all_dist_measures:
    x_vals_arr, y_vals_arr = mean_points_dict[distance_measure]
    ax.scatter(mean_points_dict[distance_measure][0],
                sorted_data[distance_measure], # mean_points_dict[distance_measure][1],
                c=metric_colors[distance_measure],
                s=30, 
                alpha=1, # 0.2, 
                edgecolors='black', 
                # linewidth=1.5
    )
    print(distance_measure, ": ", sorted_data[distance_measure])
    
    
ax.set_ylabel(f"{variation_metrics_names[metric]}\n(higher is better)") # metric.replace('_', ' ').title()}") 
ax.set_xlabel("Computation Time")
# ax.grid(axis='x', alpha=0.3)

ax.set_yticks([-20, 0, 20])
ax.set_xticks([0, 0.04, 0.08])
# ax.set_xticks([-40, 20])


ax_width_inch, ax_height_inch = viz.cm_to_inch((6, 6))
ax.set_position([0.15, 0.15, ax_width_inch/(ax_width_inch + 1), 
                ax_height_inch/(ax_height_inch + 1)])

plt.savefig(base_path / "iSNR_vs_time.pdf", bbox_inches="tight")
print(base_path / "iSNR_vs_time.pdf")


# ====================================================================
# cell 59
# ====================================================================
# HCP 
A = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/hcp_schaefer_100_dataset/01_connectomes/01_consensus_bin_density_10_percent_100.npy")
# Lexi 
# A = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/lexis_data/01_connectomes/00_connectomes_density10.npy")
A = A[0]
A.shape

# >> Both fully connected. 

# ====================================================================
# cell 60
# ====================================================================
import numpy as np

def is_fully_connected(A):
    n = A.shape[0]
    visited = set()
    stack = [0]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            neighbors = np.where(A[node] != 0)[0]
            stack.extend(neighbors)

    return len(visited) == n


is_fully_connected(A)

