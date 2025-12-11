# """
# Calculate and visualize connectome similarity across mammalian species.

# This script computes pairwise similarity metrics between animal connectomes and
# creates a hierarchical clustering dendrogram colored by taxonomic information.
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from scipy.spatial.distance import pdist, squareform
# from scipy.cluster.hierarchy import dendrogram, linkage
# from typing import Dict, Tuple
# import warnings
# warnings.filterwarnings('ignore')


# def load_data(
#     connectomes_path: Path,
#     animal_names_path: Path,
#     taxonomy_info_path: Path
# ) -> Tuple[np.ndarray, pd.DataFrame]:
#     """
#     Load connectomes and match with taxonomic information.
    
#     Args:
#         connectomes_path: Path to connectome array (num_animals, num_nodes, num_nodes)
#         animal_names_path: Path to CSV with animal names
#         taxonomy_info_path: Path to CSV with taxonomic classification
        
#     Returns:
#         connectomes: Array of connectomes
#         animal_info: DataFrame with matched animal names and taxonomy
#     """
#     # Load connectomes
#     connectomes = np.load(connectomes_path)
#     print(f"Loaded {connectomes.shape[0]} connectomes with {connectomes.shape[1]} nodes")
    
#     # Load animal names
#     animal_names = pd.read_csv(animal_names_path)
#     print(f"Loaded {len(animal_names)} animal names")
    
#     # Load taxonomy info
#     taxonomy = pd.read_csv(taxonomy_info_path)
    
#     # Match animal names with taxonomy using the Filename column
#     animal_info = animal_names.merge(
#         taxonomy,
#         left_on='animal',
#         right_on='Filename',
#         how='left'
#     )
    
#     # Check for unmatched animals
#     unmatched = animal_info[animal_info['Order'].isna()]
#     if len(unmatched) > 0:
#         print(f"Warning: {len(unmatched)} animals without taxonomy match:")
#         print(unmatched['animal'].values)
    
#     print(f"\nTaxonomic distribution:")
#     print(animal_info['Order'].value_counts())
    
#     return connectomes, animal_info


# def calculate_connectome_similarity(connectomes: np.ndarray) -> np.ndarray:
#     """
#     Calculate pairwise similarity between connectomes.
    
#     Uses multiple metrics:
#     - Frobenius norm (normalized matrix difference)
#     - Jaccard similarity (edge overlap)
#     - Correlation of connectivity patterns
    
#     Args:
#         connectomes: Array (num_animals, num_nodes, num_nodes)
        
#     Returns:
#         distance_matrix: Pairwise distance matrix (lower = more similar)
#     """
#     num_animals = connectomes.shape[0]
#     distances = np.zeros((num_animals, num_animals))
    
#     print("\nCalculating pairwise connectome similarities...")
    
#     for i in range(num_animals):
#         for j in range(i + 1, num_animals):
#             conn_i = connectomes[i]
#             conn_j = connectomes[j]
            
#             # Get upper triangular indices (excluding diagonal)
#             triu_idx = np.triu_indices_from(conn_i, k=1)
#             vec_i = conn_i[triu_idx]
#             vec_j = conn_j[triu_idx]
            
#             # Metric 1: Normalized Frobenius distance
#             frobenius_dist = np.linalg.norm(conn_i - conn_j) / np.sqrt(conn_i.size)
            
#             # Metric 2: Jaccard distance (for binary networks)
#             intersection = np.sum((vec_i > 0) & (vec_j > 0))
#             union = np.sum((vec_i > 0) | (vec_j > 0))
#             jaccard_sim = intersection / union if union > 0 else 0
#             jaccard_dist = 1 - jaccard_sim
            
#             # Metric 3: Correlation distance
#             if np.std(vec_i) > 0 and np.std(vec_j) > 0:
#                 corr = np.corrcoef(vec_i, vec_j)[0, 1]
#                 corr_dist = 1 - corr
#             else:
#                 corr_dist = 1.0
            
#             # Combined distance (weighted average)
#             combined_dist = 0.4 * frobenius_dist + 0.3 * jaccard_dist + 0.3 * corr_dist
            
#             distances[i, j] = combined_dist
#             distances[j, i] = combined_dist
    
#     return distances


# def create_taxonomy_colors(animal_info: pd.DataFrame) -> Dict[str, str]:
#     """
#     Create color mapping for taxonomic groups.
    
#     Args:
#         animal_info: DataFrame with taxonomic information
        
#     Returns:
#         color_dict: Mapping from animal name to color
#     """
#     # Get unique orders and assign colors
#     orders = animal_info['Order'].dropna().unique()
#     palette = sns.color_palette("husl", len(orders))
#     order_colors = dict(zip(orders, [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
#                                      for r, g, b in palette]))
    
#     # Map each animal to its order color
#     color_dict = {}
#     for idx, row in animal_info.iterrows():
#         animal_name = row['animal']
#         order = row['Order']
#         if pd.notna(order):
#             color_dict[animal_name] = order_colors[order]
#         else:
#             color_dict[animal_name] = '#808080'  # Gray for unknown
    
#     return color_dict, order_colors


# def plot_similarity_dendrogram(
#     distances: np.ndarray,
#     animal_info: pd.DataFrame,
#     output_path: Path = None
# ):
#     """
#     Create hierarchical clustering dendrogram of connectome similarities.
    
#     Args:
#         distances: Pairwise distance matrix
#         animal_info: DataFrame with animal names and taxonomy
#         output_path: Optional path to save figure
#     """
#     # Convert to condensed distance matrix for linkage
#     condensed_dist = squareform(distances, checks=False)
    
#     # Perform hierarchical clustering
#     linkage_matrix = linkage(condensed_dist, method='average')
    
#     # Create color mapping
#     color_dict, order_colors = create_taxonomy_colors(animal_info)
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(14, 8))
    
#     # Create dendrogram
#     dendro = dendrogram(
#         linkage_matrix,
#         labels=animal_info['animal'].values,
#         ax=ax,
#         color_threshold=0,
#         above_threshold_color='gray',
#         orientation='right'
#     )
    
#     # Color labels by taxonomy
#     ax = plt.gca()
#     labels = ax.get_ymajorticklabels()
#     for label in labels:
#         animal_name = label.get_text()
#         if animal_name in color_dict:
#             label.set_color(color_dict[animal_name])
#             label.set_fontweight('bold')
    
#     ax.set_xlabel('Connectome Distance', fontsize=12)
#     ax.set_title('Hierarchical Clustering of Mammalian Connectomes', 
#                  fontsize=14, fontweight='bold')
    
#     # Add legend for orders
#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor=color, label=order) 
#                       for order, color in order_colors.items()]
#     ax.legend(handles=legend_elements, loc='lower right', 
#              title='Order', framealpha=0.9)
    
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"\n✅ Dendrogram saved to: {output_path}")
    
#     plt.show()


# def plot_similarity_heatmap(
#     distances: np.ndarray,
#     animal_info: pd.DataFrame,
#     output_path: Path = None
# ):
#     """
#     Create heatmap of pairwise connectome similarities.
    
#     Args:
#         distances: Pairwise distance matrix
#         animal_info: DataFrame with animal names and taxonomy
#         output_path: Optional path to save figure
#     """
#     # Convert distances to similarities
#     similarities = 1 - (distances / distances.max())
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(12, 10))
    
#     # Create heatmap
#     sns.heatmap(
#         similarities,
#         xticklabels=animal_info['animal'].values,
#         yticklabels=animal_info['animal'].values,
#         cmap='viridis',
#         cbar_kws={'label': 'Similarity'},
#         square=True,
#         ax=ax
#     )
    
#     ax.set_title('Pairwise Connectome Similarity Matrix', 
#                 fontsize=14, fontweight='bold')
    
#     plt.xticks(rotation=45, ha='right', fontsize=4)
#     plt.yticks(rotation=0, fontsize=4)
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"✅ Heatmap saved to: {output_path}")
    
#     plt.show()


# def main():
#     """Main execution function."""
    
#     # Define paths
#     base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
    
#     connectomes_path = (base_path / "data/preprocessed/suarez_MaMI_dataset/" 
#                        "01_connectomes/00_connectomes_100.npy")
    
#     animal_names_path = (base_path / "data/preprocessed/suarez_MaMI_dataset/"
#                         "04_further_info/names_of_animals_with_preprocessed_connectomes_100.csv")
    
#     taxonomy_info_path = (base_path / "data/raw/suarez_MaMI_dataset/"
#                          "info/info.csv")
    
#     output_dir = base_path / "output/connectome_similarity_analysis"
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     print("=" * 70)
#     print("CONNECTOME SIMILARITY ANALYSIS")
#     print("=" * 70)
    
#     # Load data
#     print("\n1. Loading data...")
#     connectomes, animal_info = load_data(
#         connectomes_path,
#         animal_names_path,
#         taxonomy_info_path
#     )
    
#     # Calculate similarities
#     print("\n2. Computing pairwise similarities...")
#     distances = calculate_connectome_similarity(connectomes)
    
#     # Save distance matrix
#     distance_output = output_dir / "connectome_distance_matrix.npy"
#     np.save(distance_output, distances)
#     print(f"✅ Distance matrix saved to: {distance_output}")
    
#     # Create visualizations
#     print("\n3. Creating visualizations...")
    
#     plot_similarity_dendrogram(
#         distances,
#         animal_info,
#         output_path=output_dir / "connectome_dendrogram.png"
#     )
    
#     plot_similarity_heatmap(
#         distances,
#         animal_info,
#         output_path=output_dir / "connectome_similarity_heatmap.png"
#     )
    
#     # Save detailed results
#     print("\n4. Saving detailed results...")
#     results_path = output_dir / "similarity_summary.csv"
    
#     # Create summary statistics
#     similarity_matrix = 1 - (distances / distances.max())
#     summary_data = []
    
#     for i, animal_i in enumerate(animal_info['animal']):
#         for j, animal_j in enumerate(animal_info['animal']):
#             if i < j:
#                 summary_data.append({
#                     'Animal_1': animal_i,
#                     'Animal_2': animal_j,
#                     'Order_1': animal_info.iloc[i]['Order'],
#                     'Order_2': animal_info.iloc[j]['Order'],
#                     'Distance': distances[i, j],
#                     'Similarity': similarity_matrix[i, j],
#                     'Same_Order': animal_info.iloc[i]['Order'] == animal_info.iloc[j]['Order']
#                 })
    
#     summary_df = pd.DataFrame(summary_data)
#     summary_df.to_csv(results_path, index=False)
#     print(f"✅ Summary saved to: {results_path}")
    
#     print("\n" + "=" * 70)
#     print("ANALYSIS COMPLETE")
#     print("=" * 70)
#     print(f"\nResults saved to: {output_dir}")
    
#     return distances, animal_info, summary_df


# if __name__ == "__main__":
#     distances, animal_info, summary = main()


"""
Calculate and visualize connectome similarity across mammalian species.

This script computes pairwise similarity metrics between animal connectomes and
creates a hierarchical clustering dendrogram colored by taxonomic information.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import re
from typing import Dict, List, Tuple, Callable

# Import the evaluation criteria function
from config.GNM import create_evaluation_criteria
from gnm.fitting import RunConfig
from gnm.model import BinaryGenerativeParameters
from multiprocessing import Pool


def load_data(
    connectomes_path: Path,
    animal_names_path: Path,
    taxonomy_info_path: Path
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load connectomes and match with taxonomic information.
    
    Args:
        connectomes_path: Path to connectome array (num_animals, num_nodes, num_nodes)
        animal_names_path: Path to CSV with animal names
        taxonomy_info_path: Path to CSV with taxonomic classification
        
    Returns:
        connectomes: Array of connectomes
        animal_info: DataFrame with matched animal names and taxonomy
    """
    # Load connectomes
    connectomes = np.load(connectomes_path)
    print(f"Loaded {connectomes.shape[0]} connectomes with {connectomes.shape[1]} nodes")
    
    # Load animal names
    animal_names = pd.read_csv(animal_names_path)
    print(f"Loaded {len(animal_names)} animal names")
    
    # Load taxonomy info
    taxonomy = pd.read_csv(taxonomy_info_path)
    
    # Match animal names with taxonomy using the Filename column
    animal_info = animal_names.merge(
        taxonomy,
        left_on='animal',
        right_on='Filename',
        how='left'
    )
    
    # Check for unmatched animals
    unmatched = animal_info[animal_info['Order'].isna()]
    if len(unmatched) > 0:
        print(f"Warning: {len(unmatched)} animals without taxonomy match:")
        print(unmatched['animal'].values)
    
    print(f"\nTaxonomic distribution:")
    print(animal_info['Order'].value_counts())
    
    return connectomes, animal_info


def create_evaluation_criteria_list(
    distance_matrices: torch.Tensor,
    config_dict: Dict = None
) -> List:
    """
    Pre-create evaluation criteria for each distance matrix to avoid recreation overhead.
    
    Args:
        distance_matrices: Tensor of shape (n_subjects, n_nodes, n_nodes)
        config_dict: Optional config dictionary
        
    Returns:
        List of evaluation criteria objects
    """
    evaluation_criteria_list = []
    
    print("Creating evaluation criteria for each subject...")
    for i in tqdm(range(len(distance_matrices)), desc="Building criteria"):
        distance_matrix = distance_matrices[i]
        
        if config_dict is not None:
            criteria = create_evaluation_criteria(
                config=config_dict,
                distance_matrix=distance_matrix
            )
        else:
            from gnm import evaluation
            criteria = evaluation.MaxCriteria(
                evaluation.DegreeKS(),
                evaluation.ClusteringKS(),
                evaluation.EdgeLengthKS(distance_matrix),
                evaluation.BetweennessKS()
            )
        
        evaluation_criteria_list.append(criteria)
    
    return evaluation_criteria_list


def calculate_connectome_similarity(
    config_dict, 
    connectomes: np.ndarray,
    distance_matrix: np.ndarray = None
) -> np.ndarray:
    """
    Calculate pairwise similarity between connectomes using MaxCrit evaluation.
    
    Uses the same MaxCrit evaluation criteria from the GNM fitting script:
    - DegreeKS
    - ClusteringKS
    - EdgeLengthKS (if distance matrix provided)
    - BetweennessKS
    
    Args:
        connectomes: Array (num_animals, num_nodes, num_nodes)
        distance_matrix: Optional distance matrix for EdgeLengthKS
        
    Returns:
        distance_matrix: Pairwise distance matrix (lower = more similar)
    """
    import torch
    from gnm import evaluation
    
    num_animals = connectomes.shape[0]
    distances = np.zeros((num_animals, num_animals))
    
    print("\nCalculating pairwise connectome similarities using MaxCrit...")
    
    # Create evaluation criteria
    if distance_matrix is not None:
        distance_tensor = torch.tensor(distance_matrix, dtype=torch.float32)
        # eval_criteria = evaluation.MaxCriteria(
        #     evaluation.DegreeKS(),
        #     evaluation.ClusteringKS(),
        #     evaluation.EdgeLengthKS(distance_tensor),
        #     evaluation.BetweennessKS()
        # )
        eval_criteria = create_evaluation_criteria_list(
            distance_matrices=distance_tensor.unsqueeze(0),
            config_dict=config_dict
        )[0]
        print("Using MaxCrit with: DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS")
    else:
        eval_criteria = evaluation.MaxCriteria(
            evaluation.DegreeKS(),
            evaluation.ClusteringKS(),
            evalation.BetweennessKS()
        )
        print("Using MaxCrit with: DegreeKS, ClusteringKS, BetweennessKS")
        print("Warning: No distance matrix provided, skipping EdgeLengthKS metric")
    
    # Calculate pairwise distances using MaxCrit
    for i in range(num_animals):
        for j in range(i + 1, num_animals):
            # Convert to torch tensors and add batch dimension
            conn_i = torch.tensor(connectomes[i:i+1], dtype=torch.float32)
            conn_j = torch.tensor(connectomes[j:j+1], dtype=torch.float32)
            
            # Calculate energy (distance) using evaluation criteria
            # MaxCrit returns the maximum of all metric distances
            energy = eval_criteria(conn_i, conn_j)
            
            # Convert to numpy scalar
            distance = float(energy.mean().item())
            
            distances[i, j] = distance
            distances[j, i] = distance
    
    return distances


def create_taxonomy_colors(animal_info: pd.DataFrame) -> Dict[str, str]:
    """
    Create color mapping for taxonomic groups.
    
    Args:
        animal_info: DataFrame with taxonomic information
        
    Returns:
        color_dict: Mapping from animal name to color
    """
    # Get unique orders and assign colors
    orders = animal_info['Order'].dropna().unique()
    palette = sns.color_palette("husl", len(orders))
    order_colors = dict(zip(orders, [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                                     for r, g, b in palette]))
    
    # Map each animal to its order color
    color_dict = {}
    for idx, row in animal_info.iterrows():
        animal_name = row['animal']
        order = row['Order']
        if pd.notna(order):
            color_dict[animal_name] = order_colors[order]
        else:
            color_dict[animal_name] = '#808080'  # Gray for unknown
    
    return color_dict, order_colors


def plot_similarity_dendrogram(
    distances: np.ndarray,
    animal_info: pd.DataFrame,
    output_path: Path = None
):
    """
    Create hierarchical clustering dendrogram of connectome similarities.
    
    Args:
        distances: Pairwise distance matrix
        animal_info: DataFrame with animal names and taxonomy
        output_path: Optional path to save figure
    """
    # Convert to condensed distance matrix for linkage
    condensed_dist = squareform(distances, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Create color mapping
    color_dict, order_colors = create_taxonomy_colors(animal_info)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create dendrogram
    dendro = dendrogram(
        linkage_matrix,
        labels=animal_info['animal'].values,
        ax=ax,
        color_threshold=0,
        above_threshold_color='gray',
        orientation='right'
    )
    
    # Color labels by taxonomy
    ax = plt.gca()
    labels = ax.get_ymajorticklabels()
    for label in labels:
        animal_name = label.get_text()
        if animal_name in color_dict:
            label.set_color(color_dict[animal_name])
            label.set_fontweight('bold')
    
    ax.set_xlabel('Connectome Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering of Mammalian Connectomes', 
                 fontsize=14, fontweight='bold')
    
    # Add legend for orders
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=order) 
                      for order, color in order_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', 
             title='Order', framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Dendrogram saved to: {output_path}")
    
    plt.show()


def plot_similarity_heatmap(
    distances: np.ndarray,
    animal_info: pd.DataFrame,
    output_path: Path = None
):
    """
    Create heatmap of pairwise connectome similarities, ordered by taxonomy.
    
    Args:
        distances: Pairwise distance matrix
        animal_info: DataFrame with animal names and taxonomy
        output_path: Optional path to save figure
    """
    # Convert distances to similarities
    similarities = 1 - (distances / distances.max())
    
    # Sort animals by Order, then by name
    animal_info_sorted = animal_info.copy()
    animal_info_sorted['Order'] = animal_info_sorted['Order'].fillna('Unknown')
    animal_info_sorted = animal_info_sorted.sort_values(['Order', 'animal'])
    
    # Get sorting indices
    sorted_indices = animal_info_sorted.index.values
    
    # Reorder similarity matrix
    similarities_sorted = similarities[sorted_indices][:, sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        similarities_sorted,
        xticklabels=animal_info_sorted['animal'].values,
        yticklabels=animal_info_sorted['animal'].values,
        cmap='viridis',
        cbar_kws={'label': 'Similarity'},
        square=True,
        ax=ax
    )
    
    # Add visual separators between orders
    order_boundaries = [0]
    current_order = animal_info_sorted.iloc[0]['Order']
    for i, order in enumerate(animal_info_sorted['Order']):
        if order != current_order:
            order_boundaries.append(i)
            current_order = order
    order_boundaries.append(len(animal_info_sorted))
    
    # Draw lines to separate orders
    for boundary in order_boundaries[1:-1]:
        ax.axhline(boundary, color='white', linewidth=2)
        ax.axvline(boundary, color='white', linewidth=2)
    
    ax.set_title('Pairwise Connectome Similarity Matrix (Grouped by Order)', 
                fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Heatmap saved to: {output_path}")
    
    plt.show()


def main():
    """Main execution function."""
    
    # Define paths
    base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code")
    
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_100.npy
    connectomes_path = (base_path / "data/preprocessed/suarez_MaMI_dataset/" 
                       "01_consensus_bin_density_10_percent_100.npy") # 01_connectomes/00_connectomes_100.npy")
    connectomes_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_100.npy")
    
    animal_names_path = (base_path / "data/preprocessed/suarez_MaMI_dataset/"
                        "04_further_info/names_of_animals_with_preprocessed_connectomes_100.csv")
    
    taxonomy_info_path = (base_path / "data/raw/suarez_MaMI_dataset/"
                         "info/info.csv")
    
    distance_matrix_path = (base_path / "data/preprocessed/suarez_MaMI_dataset/"
                           "02_distance_matrices/distance_matrix_100.npy")
    
    output_dir = base_path / "output/connectome_similarity_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_dict = {
        'gnm': {
            'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
        }
    }
    
    print("=" * 70)
    print("CONNECTOME SIMILARITY ANALYSIS")
    print("=" * 70)
    
    
    
    # Load data
    print("\n1. Loading data...")
    connectomes, animal_info = load_data(
        connectomes_path,
        animal_names_path,
        taxonomy_info_path
    )
    
    # Load distance matrix if available
    distance_matrix = None
    if distance_matrix_path.exists():
        distance_matrix = np.load(distance_matrix_path)
        print(f"Loaded distance matrix with shape: {distance_matrix.shape}")
        # Use the first distance matrix (assuming they're similar across animals)
        distance_matrix = distance_matrix[0, :, :] # !!!!
    else:
        print("Warning: Distance matrix not found, EdgeLengthKS will be skipped")
    
    # Calculate similarities
    print("\n2. Computing pairwise similarities using MaxCrit...")
    distances = calculate_connectome_similarity(config_dict=config_dict, 
                                                connectomes=connectomes, 
                                                distance_matrix=distance_matrix)
    
    # Save distance matrix
    distance_output = output_dir / "connectome_distance_matrix_maxcrit.npy"
    np.save(distance_output, distances)
    print(f"✅ Distance matrix saved to: {distance_output}")
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    
    plot_similarity_dendrogram(
        distances,
        animal_info,
        output_path=output_dir / "connectome_dendrogram_maxcrit.png"
    )
    
    plot_similarity_heatmap(
        distances,
        animal_info,
        output_path=output_dir / "connectome_similarity_heatmap_maxcrit.png"
    )
    
    # Save detailed results
    print("\n4. Saving detailed results...")
    results_path = output_dir / "similarity_summary_maxcrit.csv"
    
    # Create summary statistics
    similarity_matrix = 1 - (distances / distances.max())
    summary_data = []
    
    for i, animal_i in enumerate(animal_info['animal']):
        for j, animal_j in enumerate(animal_info['animal']):
            if i < j:
                summary_data.append({
                    'Animal_1': animal_i,
                    'Animal_2': animal_j,
                    'Order_1': animal_info.iloc[i]['Order'],
                    'Order_2': animal_info.iloc[j]['Order'],
                    'Distance': distances[i, j],
                    'Similarity': similarity_matrix[i, j],
                    'Same_Order': animal_info.iloc[i]['Order'] == animal_info.iloc[j]['Order']
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_path, index=False)
    print(f"✅ Summary saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    
    return distances, animal_info, summary_df


if __name__ == "__main__":
    distances, animal_info, summary = main()