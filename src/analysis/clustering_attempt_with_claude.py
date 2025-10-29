import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (silhouette_score, adjusted_rand_score, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPECIES = 225
N_NODES = 100

# ============================================================================
# NETWORK PORTRAIT FUNCTIONS
# ============================================================================
def compute_network_portrait(adj_matrix, B_max=10):
    """Compute network portrait from adjacency matrix."""
    G = nx.from_numpy_array(adj_matrix)
    portrait = defaultdict(int)
    
    for node in G.nodes():
        shells = {node: 0}
        current_shell = [node]
        
        for b in range(1, B_max + 1):
            next_shell = []
            for n in current_shell:
                for neighbor in G.neighbors(n):
                    if neighbor not in shells:
                        shells[neighbor] = b
                        next_shell.append(neighbor)
            current_shell = next_shell
            if not current_shell:
                break
        
        for n, shell in shells.items():
            degree = G.degree(n)
            portrait[(shell, degree)] += 1
    
    total = sum(portrait.values())
    if total > 0:
        portrait = {k: v/total for k, v in portrait.items()}
    
    return portrait

def portrait_divergence(portrait1, portrait2):
    """Compute Jensen-Shannon divergence between two network portraits."""
    all_keys = set(portrait1.keys()) | set(portrait2.keys())
    
    if not all_keys:
        return 0.0
    
    p1 = np.array([portrait1.get(k, 0) for k in all_keys])
    p2 = np.array([portrait2.get(k, 0) for k in all_keys])
    
    p1 = p1 / (p1.sum() + 1e-10)
    p2 = p2 / (p2.sum() + 1e-10)
    
    m = (p1 + p2) / 2
    
    def kl_div(p, q):
        mask = (p > 0) & (q > 0)
        if not np.any(mask):
            return 0.0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    js_div = 0.5 * kl_div(p1, m) + 0.5 * kl_div(p2, m)
    
    return np.sqrt(max(0, js_div))  # Ensure non-negative

def compute_portrait_distance_matrix(connectomes):
    """Compute pairwise portrait divergence for all connectomes."""
    n = len(connectomes)
    dist_matrix = np.zeros((n, n))
    
    print("Computing network portraits...")
    portraits = [compute_network_portrait(conn) for conn in connectomes]
    
    print("Computing pairwise portrait divergences...")
    for i in range(n):
        if i % 50 == 0:
            print(f"  Processing species {i}/{n}")
        for j in range(i+1, n):
            dist = portrait_divergence(portraits[i], portraits[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix

# ============================================================================
# TOPOLOGICAL DISTANCE FUNCTIONS
# ============================================================================
def compute_topological_features(adj_matrix):
    """Compute various topological features from adjacency matrix."""
    G = nx.from_numpy_array(adj_matrix)
    
    features = {}
    
    degrees = [d for n, d in G.degree()]
    features['mean_degree'] = np.mean(degrees) if degrees else 0.0
    features['std_degree'] = np.std(degrees) if degrees else 0.0
    features['max_degree'] = np.max(degrees) if degrees else 0.0
    
    features['clustering'] = nx.average_clustering(G)
    
    if nx.is_connected(G):
        features['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        components = list(nx.connected_components(G))
        if components:
            gcc = max(components, key=len)
            if len(gcc) > 1:
                features['avg_path_length'] = nx.average_shortest_path_length(G.subgraph(gcc))
            else:
                features['avg_path_length'] = 0.0
        else:
            features['avg_path_length'] = 0.0
    
    features['assortativity'] = nx.degree_assortativity_coefficient(G)
    features['density'] = nx.density(G)
    
    return features

def compute_topological_distance_matrix(connectomes):
    """Compute distance matrix based on topological features."""
    print("Computing topological features...")
    features_list = []
    
    for i, conn in enumerate(connectomes):
        if i % 50 == 0:
            print(f"  Processing species {i}/{len(connectomes)}")
        features_list.append(compute_topological_features(conn))
    
    feature_names = list(features_list[0].keys())
    feature_matrix = np.array([[f[name] for name in feature_names] 
                               for f in features_list])
    
    # Check for NaN/Inf and replace
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    
    print("Computing pairwise topological distances...")
    dist_matrix = squareform(pdist(feature_matrix, metric='euclidean'))
    
    return dist_matrix

# ============================================================================
# CLUSTERING FUNCTIONS
# ============================================================================
def perform_clustering(dist_matrix, n_clusters, method='average'):
    """Perform hierarchical clustering on distance matrix."""
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage=method
    )
    labels = clustering.fit_predict(dist_matrix)
    return labels

def evaluate_clustering(dist_matrix, labels):
    """Compute clustering quality metrics."""
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(labels):
        return {'silhouette': 0.0, 'n_clusters': n_unique}
    
    sil_score = silhouette_score(dist_matrix, labels, metric='precomputed')
    
    return {
        'silhouette': sil_score,
        'n_clusters': n_unique
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_dendrogram(dist_matrix, labels, method='average', title='Dendrogram'):
    """Plot dendrogram."""
    condensed_dist = squareform(dist_matrix)
    linkage_matrix = hierarchy.linkage(condensed_dist, method=method)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    hierarchy.dendrogram(linkage_matrix, no_labels=True, ax=ax)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Species Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    
    # Add text showing label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_text = '\n'.join([f'{label}: {count}' for label, count in zip(unique, counts)])
    ax.text(1.02, 0.5, label_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return linkage_matrix

def plot_2d_projection(dist_matrix, labels, method='MDS', title='2D Projection'):
    """Visualize in 2D using MDS, PCA, or t-SNE."""
    
    if method == 'MDS':
        reducer = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords_2d = reducer.fit_transform(dist_matrix)
    elif method == 'PCA':
        # For PCA, we need features not distances - convert back approximately
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(-dist_matrix)
    elif method == 'tSNE':
        tsne = TSNE(n_components=2,
                    init="random", 
                    metric='precomputed',
                    random_state=42, 
                    perplexity=min(30, len(dist_matrix) - 1))
        coords_2d = tsne.fit_transform(dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Use distinct colors
    if n_labels <= 10:
        colors = plt.cm.tab10(np.arange(n_labels))
    elif n_labels <= 20:
        colors = plt.cm.tab20(np.arange(n_labels))
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_labels))
    
    label_to_color = dict(zip(unique_labels, colors))
    
    for label in unique_labels:
        mask = (labels == label)
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                  c=[label_to_color[label]], label=str(label),
                  s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'{method} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method} Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Place legend outside
    if n_labels <= 20:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    else:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=2)
    
    plt.tight_layout()

def plot_confusion_matrix(labels1, labels2, name1, name2):
    """Create confusion matrix between two label sets."""
    # Convert all labels to strings
    labels1 = np.array([str(l) for l in labels1])
    labels2 = np.array([str(l) for l in labels2])
    
    unique1 = sorted(np.unique(labels1))
    unique2 = sorted(np.unique(labels2))
    
    # FIXED: Provide both label sets explicitly
    cm = confusion_matrix(labels1, labels2, labels=unique1)
    
    # Get actual dimensions after creating confusion matrix
    n_rows, n_cols = cm.shape
    
    fig, ax = plt.subplots(figsize=(max(12, n_cols*0.6), 
                                    max(10, n_rows*0.5)))
    
    im = ax.imshow(cm, cmap='YlOrRd', aspect='auto')
    
    # Use actual unique values from labels2 for x-axis
    unique2_actual = sorted(np.unique(labels2))
    
    ax.set_xticks(range(len(unique2_actual)))
    ax.set_yticks(range(len(unique1)))
    ax.set_xticklabels(unique2_actual, rotation=90, ha='right', fontsize=9)
    ax.set_yticklabels(unique1, fontsize=9)
    
    ax.set_xlabel(name2, fontsize=12)
    ax.set_ylabel(name1, fontsize=12)
    ax.set_title(f'{name1} vs {name2}', fontsize=14)
    
    # Add counts as text with bounds checking
    for i in range(n_rows):
        for j in range(n_cols):
            if cm[i, j] > 0:
                text = ax.text(j, i, cm[i, j],
                              ha="center", va="center",
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================
def run_clustering_analysis(distance_matrices, connectomes, 
                           species_df, label_columns,
                           n_clusters=5, save_folder=None):
    """
    Run complete clustering analysis pipeline with phylogenetic labels.
    """
    
    results = {}
    
    # ========================================================================
    # METHOD 1: Distance matrices
    # ========================================================================
    print("\n" + "="*70)
    print("METHOD 1: Clustering on provided distance matrices")
    print("="*70)
    
    species_dist_matrix = np.zeros((N_SPECIES, N_SPECIES))
    for i in range(N_SPECIES):
        for j in range(i+1, N_SPECIES):
            dist = np.linalg.norm(distance_matrices[i] - distance_matrices[j])
            species_dist_matrix[i, j] = dist
            species_dist_matrix[j, i] = dist
    
    labels_dist = perform_clustering(species_dist_matrix, n_clusters)
    results['Distance Matrix'] = labels_dist
    eval_dist = evaluate_clustering(species_dist_matrix, labels_dist)
    print(f"  Silhouette Score: {eval_dist['silhouette']:.3f}")
    
    # ========================================================================
    # METHOD 2: Topological features
    # ========================================================================
    print("\n" + "="*70)
    print("METHOD 2: Clustering on topological features")
    print("="*70)
    
    topo_dist_matrix = compute_topological_distance_matrix(connectomes)
    labels_topo = perform_clustering(topo_dist_matrix, n_clusters)
    results['Topological'] = labels_topo
    eval_topo = evaluate_clustering(topo_dist_matrix, labels_topo)
    print(f"  Silhouette Score: {eval_topo['silhouette']:.3f}")
    
    # ========================================================================
    # METHOD 3: Network portraits
    # ========================================================================
    print("\n" + "="*70)
    print("METHOD 3: Clustering on network portraits")
    print("="*70)
    
    portrait_dist_matrix = compute_portrait_distance_matrix(connectomes)
    labels_portrait = perform_clustering(portrait_dist_matrix, n_clusters)
    results['Network Portrait'] = labels_portrait
    eval_portrait = evaluate_clustering(portrait_dist_matrix, labels_portrait)
    print(f"  Silhouette Score: {eval_portrait['silhouette']:.3f}")
    
    # ========================================================================
    # Compare clustering solutions
    # ========================================================================
    print("\n" + "="*70)
    print("CLUSTERING AGREEMENT")
    print("="*70)
    
    ari_dist_topo = adjusted_rand_score(labels_dist, labels_topo)
    ari_dist_portrait = adjusted_rand_score(labels_dist, labels_portrait)
    ari_topo_portrait = adjusted_rand_score(labels_topo, labels_portrait)
    
    print(f"  Distance vs Topological: ARI = {ari_dist_topo:.3f}")
    print(f"  Distance vs Portrait: ARI = {ari_dist_portrait:.3f}")
    print(f"  Topological vs Portrait: ARI = {ari_topo_portrait:.3f}")
    
    # ========================================================================
    # Visualizations
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    distance_matrices_dict = {
        'distance': species_dist_matrix,
        'topological': topo_dist_matrix,
        'portrait': portrait_dist_matrix
    }
    
    clustering_labels_dict = {
        'distance': labels_dist,
        'topological': labels_topo,
        'portrait': labels_portrait
    }
    
    # For each phylogenetic label column
    for label_col in label_columns:
        print(f"\n  Creating plots for label: {label_col}")
        
        if label_col not in species_df.columns:
            print(f"    Warning: Column '{label_col}' not found. Skipping.")
            continue
        
        phylo_labels = species_df[label_col].values
        
        # Check for missing values
        n_missing = pd.isna(phylo_labels).sum()
        if n_missing > 0:
            print(f"    Warning: {n_missing} missing values in {label_col}, filling with 'Unknown'")
            phylo_labels = np.where(pd.isna(phylo_labels), 'Unknown', phylo_labels)
        
        label_folder = save_folder / label_col
        label_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"    Label distribution in {label_col}:")
        unique, counts = np.unique(phylo_labels, return_counts=True)
        for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
            print(f"      {u}: {c}")
        
        # 1. Dendrograms
        print(f"    Creating dendrograms...")
        for method_name, dist_matrix in distance_matrices_dict.items():
            plot_dendrogram(dist_matrix, phylo_labels,
                          title=f'{method_name.capitalize()} Dendrogram (colored by {label_col})')
            plt.savefig(label_folder / f'dendrogram_{method_name}.pdf', 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. 2D projections - MDS, PCA, t-SNE
        print(f"    Creating 2D projections...")
        for method_name, dist_matrix in distance_matrices_dict.items():
            # By cluster
            for proj_method in ['MDS', 'PCA', 'tSNE']:
                try:
                    plot_2d_projection(
                        dist_matrix, 
                        clustering_labels_dict[method_name],
                        method=proj_method,
                        title=f'{method_name.capitalize()} - By Cluster ({proj_method})'
                    )
                    plt.savefig(label_folder / f'{proj_method.lower()}_{method_name}_by_cluster.pdf', 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"      Warning: Could not create {proj_method} plot for {method_name}: {e}")
                    plt.close()
                
                # By phylogeny
                try:
                    plot_2d_projection(
                        dist_matrix, phylo_labels,
                        method=proj_method,
                        title=f'{method_name.capitalize()} - By {label_col} ({proj_method})'
                    )
                    plt.savefig(label_folder / f'{proj_method.lower()}_{method_name}_by_{label_col}.pdf', 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"      Warning: Could not create {proj_method} plot by phylogeny for {method_name}: {e}")
                    plt.close()
        
        # 3. Confusion matrices
        print(f"    Creating confusion matrices...")
        for method_name, cluster_labels in clustering_labels_dict.items():
            try:
                plot_confusion_matrix(
                    cluster_labels, phylo_labels,
                    f'{method_name.capitalize()}', label_col
                )
                plt.savefig(label_folder / f'confusion_{method_name}_vs_{label_col}.pdf', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                ari = adjusted_rand_score(cluster_labels, phylo_labels)
                print(f"      {method_name.capitalize()} vs {label_col}: ARI = {ari:.3f}")
            except Exception as e:
                print(f"      Warning: Could not create confusion matrix for {method_name}: {e}")
                plt.close()
    
    # Method agreement
    print("\n  Creating method agreement heatmap...")
    agreement_matrix = np.array([
        [1.0, ari_dist_topo, ari_dist_portrait],
        [ari_dist_topo, 1.0, ari_topo_portrait],
        [ari_dist_portrait, ari_topo_portrait, 1.0]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=['Distance', 'Topological', 'Portrait'],
                yticklabels=['Distance', 'Topological', 'Portrait'],
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'ARI'}, ax=ax)
    ax.set_title('Method Agreement (ARI)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_folder / 'method_agreement.pdf', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)

    return results, {
        'distance_matrix': species_dist_matrix,
        'topological_matrix': topo_dist_matrix,
        'portrait_matrix': portrait_dist_matrix
    }


if __name__ == "__main__":
    # # Load data
    # distance_matrices = np.load('/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_orig.npy') 
    # connectomes = np.load('/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/01_connectomes/00_connectomes_orig.npy') 
    
    # # Load species labels
    # LABEL_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50.csv"
    
    # species_df = pd.read_csv(LABEL_FILE, index_col=0)
    
    # # Define which phylogenetic levels to analyze
    # label_columns = ['order', 'family', 'phylogenetic_group', 'super_order']
    
    # # Output folder
    # save_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/empirical_data/suarez_MaMI_dataset/clustering_2")
    # save_folder.mkdir(parents=True, exist_ok=True)
    
    # # Run analysis
    # results, dist_mats = run_clustering_analysis(
    #     distance_matrices, connectomes, 
    #     species_df=species_df,
    #     label_columns=label_columns,
    #     n_clusters=5,
    #     save_folder=save_folder
    # )
    
    # # Save results
    # np.save(save_folder / 'cluster_labels.npy', results)
    # np.save(save_folder / 'distance_matrix.npy', dist_mats["distance_matrix"])
    # np.save(save_folder / 'topological_matrix.npy', dist_mats["topological_matrix"])
    # np.save(save_folder / 'portrait_matrix.npy', dist_mats["portrait_matrix"])
    
    # print(f"\nAll results saved to: {save_folder}")
    
    
    
    ##### LOOOPING 
    
    data_names = [# "orig", "50", 
                  "bin_density_10_percent_50"]
    # Prepare base path 
    data_base_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/")
    save_base_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/empirical_data/suarez_MaMI_dataset/clustering_looped")
    # Load species labels
    LABEL_FILE = data_base_path / "04_further_info/names_of_animals_with_preprocessed_connectomes_50.csv"
    
    species_df = pd.read_csv(LABEL_FILE, index_col=0)
    
    # Define which phylogenetic levels to analyze
    label_columns = ['order', 'family', 'phylogenetic_group', 'super_order']
    
    # Start looping 
    for d_name in data_names:
        print(f"\n\n\n########## RUNNING ANALYSIS FOR DATASET: {d_name} ##########\n\n")
        
        # Load data
        if d_name == "orig":
            distance_matrices = np.load(data_base_path / f'02_distance_matrices/distance_matrix_{d_name}.npy') 
            connectomes = np.load(data_base_path / f'01_connectomes/00_connectomes_{d_name}.npy') 
        elif d_name == "50" or d_name == "bin_density_10_percent_50":
            distance_matrices = np.load(data_base_path / f'02_distance_matrices/distance_matrix_50.npy') 
            connectomes = np.load(data_base_path / f'01_connectomes/01_consensus_{d_name}.npy') # ATTENTION: I'LL NEED TO CHANGE THIS NAMING SOON. 
        else: 
            print(f"Unknown dataset name: {d_name}, skipping...")
            continue

        # Output folder
        save_folder = save_base_folder / f'clustering_{d_name}'
        save_folder.mkdir(parents=True, exist_ok=True)

        # Run analysis
        results, dist_mats = run_clustering_analysis(
            distance_matrices, 
            connectomes, 
            species_df=species_df,
            label_columns=label_columns,
            n_clusters=5,
            save_folder=save_folder
        )
        
        # Save results
        np.save(save_folder / 'cluster_labels.npy', results)
        np.save(save_folder / 'distance_matrix.npy', dist_mats["distance_matrix"])
        np.save(save_folder / 'topological_matrix.npy', dist_mats["topological_matrix"])
        np.save(save_folder / 'portrait_matrix.npy', dist_mats["portrait_matrix"])
        
        print(f"\nAll results saved to: {save_folder}")