"""
Direct Connectome Clustering with CEBRA
Treats each connectome as a sequence of node connectivity patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

from cebra import CEBRA


# ============================================================================
# CONFIGURATION
# ============================================================================

N_SPECIES = 225
N_NODES = 100
N_EMBEDDING_DIM = 16
N_CLUSTERS = 5


DATA_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset") 
CONNECTOME_FILE = DATA_PATH / "01_connectomes/00_connectomes_orig.npy"
LABEL_FILE = DATA_PATH / "04_further_info/names_of_animals_with_preprocessed_connectomes_50.csv"

OUTPUT_PATH = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/empirical_data/suarez_MaMI_dataset/cebra") 
OUTPUT_PATH = OUTPUT_PATH / "cebra_connectome_clustering" / f"embedding_dim_{N_EMBEDDING_DIM}_clusters_{N_CLUSTERS}"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_connectome_sequences(connectomes):
    """
    Convert connectomes to sequences for CEBRA.
    Each connectome becomes a sequence where each 'timestep' is a node,
    and features are the connectivity pattern of that node.
    
    Args:
        connectomes: (n_species, n_nodes, n_nodes) array
        
    Returns:
        sequences: (n_species, n_nodes, n_features) array
    """
    n_species, n_nodes, _ = connectomes.shape
    
    # Strategy 1: Each row of adjacency matrix as features
    sequences = connectomes.copy()
    
    # Strategy 2: Add graph-theoretic node features
    # Compute degree, clustering coefficient, etc. for each node
    from scipy.sparse import csr_matrix
    import networkx as nx
    
    enhanced_sequences = []
    
    for i in range(n_species):
        G = nx.from_numpy_array(connectomes[i])
        
        # Node features
        degree = np.array([d for n, d in G.degree()])
        clustering = np.array(list(nx.clustering(G).values()))
        betweenness = np.array(list(nx.betweenness_centrality(G).values()))
        
        # Combine connectivity with features
        node_features = np.column_stack([
            connectomes[i],  # Full connectivity row
            degree.reshape(-1, 1),
            clustering.reshape(-1, 1),
            betweenness.reshape(-1, 1)
        ])
        
        enhanced_sequences.append(node_features)
    
    return np.array(enhanced_sequences)


def normalize_sequences(sequences):
    """Normalize sequences for CEBRA training."""
    # Z-score normalization per feature
    mean = sequences.mean(axis=(0, 1), keepdims=True)
    std = sequences.std(axis=(0, 1), keepdims=True) + 1e-8
    return (sequences - mean) / std


# ============================================================================
# CEBRA EMBEDDING
# ============================================================================
def fit_cebra_single_session(sequences, output_dim=16, max_iterations=5000):
    """
    Fit separate CEBRA model for each species (discovery-driven).
    
    Args:
        sequences: (n_species, n_nodes, n_features)
        output_dim: Embedding dimension
        
    Returns:
        embeddings: List of embeddings, one per species
    """
    embeddings = []
    
    for i, seq in enumerate(sequences):
        if i % 25 == 0:
            print(f"  Fitting CEBRA for species {i}/{len(sequences)}")
        
        # Time-contrastive learning (discovery-driven)
        cebra_model = CEBRA(
            model_architecture='offset10-model',
            batch_size=512,
            learning_rate=3e-4,
            temperature=1.0,
            output_dimension=output_dim,
            max_iterations=max_iterations,
            distance='cosine',
            conditional='time_delta',
            time_offsets=10,
            device='cuda_if_available',
            verbose=False
        )
        
        cebra_model.fit(seq)
        embedding = cebra_model.transform(seq)
        embeddings.append(embedding)
    
    return embeddings


def fit_cebra_multi_session(sequences, labels=None, output_dim=16, max_iterations=10000):
    """
    Fit joint CEBRA model across all species.
    
    Args:
        sequences: (n_species, n_nodes, n_features)
        labels: Optional phylogenetic labels for hypothesis-driven embedding
        output_dim: Embedding dimension
        
    Returns:
        embeddings: List of embeddings, one per species
        model: Fitted CEBRA model
    """
    print("  Fitting multi-session CEBRA model...")
    
    # Multi-session training
    cebra_model = CEBRA(
        model_architecture='offset10-model',
        batch_size=512,
        learning_rate=3e-4,
        temperature=1.0,
        output_dimension=output_dim,
        max_iterations=max_iterations,
        distance='cosine',
        conditional='time_delta',
        time_offsets=10,
        device='cuda_if_available',
        verbose=True
    )
    
    # Prepare data as list of sessions
    session_data = [seq for seq in sequences]
    
    if labels is not None:
        # Hypothesis-driven: use phylogenetic labels
        # Repeat labels for each node in sequence
        session_labels = [np.full(len(seq), label) for seq, label in zip(sequences, labels)]
        cebra_model.fit(session_data, session_labels)
    else:
        # Discovery-driven: time only
        cebra_model.fit(session_data)
    
    # Transform each session
    embeddings = [cebra_model.transform(seq) for seq in session_data]
    
    return embeddings, cebra_model


# ============================================================================
# EMBEDDING COMPARISON
# ============================================================================
def compute_embedding_distances(embeddings):
    """
    Compute pairwise distances between species embeddings.
    
    Args:
        embeddings: List of (n_nodes, n_features) arrays
        
    Returns:
        distance_matrix: (n_species, n_species) array
    """
    n_species = len(embeddings)
    dist_matrix = np.zeros((n_species, n_species))
    
    for i in range(n_species):
        for j in range(i+1, n_species):
            # Option 1: Mean pairwise distance between embeddings
            emb_i = embeddings[i]
            emb_j = embeddings[j]
            
            # Compute distance between distributions
            # Use Wasserstein distance (Earth Mover's Distance)
            from scipy.stats import wasserstein_distance
            
            # Average over all dimensions
            dist = np.mean([
                wasserstein_distance(emb_i[:, k], emb_j[:, k])
                for k in range(emb_i.shape[1])
            ])
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def compute_embedding_summary(embeddings):
    """
    Compute summary statistics of each embedding for clustering.
    
    Returns:
        features: (n_species, n_summary_features) array
    """
    summaries = []
    
    for emb in embeddings:
        # Statistical summaries across nodes
        summary = np.concatenate([
            emb.mean(axis=0),  # Mean
            emb.std(axis=0),   # Std
            np.percentile(emb, 25, axis=0),  # Q1
            np.percentile(emb, 75, axis=0),  # Q3
        ])
        summaries.append(summary)
    
    return np.array(summaries)


# ============================================================================
# CLUSTERING
# ============================================================================
def cluster_from_distances(dist_matrix, n_clusters=5):
    """Perform hierarchical clustering on distance matrix."""
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(dist_matrix)
    return labels


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_embeddings_mds(embeddings, labels, title, save_path):
    """Visualize embedding summaries with MDS."""
    summaries = compute_embedding_summary(embeddings)
    
    # Compute distance matrix
    dist_matrix = squareform(pdist(summaries, metric='euclidean'))
    
    # MDS projection
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_matrix)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[color], label=str(label), s=50, alpha=0.7,
                  edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('MDS Dimension 1', fontsize=12)
    ax.set_ylabel('MDS Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cluster_labels, phylo_labels, save_path):
    """Plot confusion matrix between clusters and phylogeny."""
    from sklearn.metrics import confusion_matrix
    
    # Convert to strings
    cluster_labels = np.array([str(l) for l in cluster_labels])
    phylo_labels = np.array([str(l) for l in phylo_labels])
    
    unique_clusters = sorted(np.unique(cluster_labels))
    unique_phylo = sorted(np.unique(phylo_labels))
    
    cm = confusion_matrix(cluster_labels, phylo_labels, labels=unique_clusters)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(cm, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(len(unique_phylo)))
    ax.set_yticks(range(len(unique_clusters)))
    ax.set_xticklabels(unique_phylo, rotation=90, ha='right', fontsize=8)
    ax.set_yticklabels(unique_clusters, fontsize=9)
    
    ax.set_xlabel('Phylogenetic Label', fontsize=12)
    ax.set_ylabel('CEBRA Cluster', fontsize=12)
    ax.set_title('CEBRA Clusters vs Phylogeny', fontsize=14)
    
    # Add text annotations
    for i in range(len(unique_clusters)):
        for j in range(len(unique_phylo)):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j],
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black",
                       fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main analysis pipeline."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DIRECT CONNECTOME CLUSTERING WITH CEBRA")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    connectomes = np.load(CONNECTOME_FILE)
    df_labels = pd.read_csv(LABEL_FILE, index_col=0)
    phylo_labels = df_labels['order'].values
    
    print(f"Loaded {len(connectomes)} connectomes of shape {connectomes[0].shape}")
    
    # Prepare sequences
    print("\nPreparing connectome sequences...")
    sequences = prepare_connectome_sequences(connectomes)
    sequences = normalize_sequences(sequences)
    print(f"Sequence shape: {sequences.shape}")
    
    # ========================================================================
    # Method 1: Single-session CEBRA (discovery-driven)
    # ========================================================================
    print("\n" + "="*70)
    print("METHOD 1: Single-Session CEBRA (Discovery-Driven)")
    print("="*70)

    embeddings_single = fit_cebra_single_session(
        sequences, 
        output_dim=N_EMBEDDING_DIM,
        max_iterations=3000
    )
    
    # Compute distances and cluster
    dist_matrix_single = compute_embedding_distances(embeddings_single)
    labels_single = cluster_from_distances(dist_matrix_single, N_CLUSTERS)
    
    # Evaluate
    ari_single = adjusted_rand_score(labels_single, phylo_labels)
    sil_single = silhouette_score(dist_matrix_single, labels_single, metric='precomputed')
    
    print(f"\nSingle-Session Results:")
    print(f"  ARI vs Phylogeny: {ari_single:.3f}")
    print(f"  Silhouette Score: {sil_single:.3f}")
    
    # Visualize
    plot_embeddings_mds(
        embeddings_single, phylo_labels,
        'Single-Session CEBRA - Colored by Phylogeny',
        OUTPUT_PATH / 'single_session_mds_phylogeny.pdf'
    )
    
    plot_embeddings_mds(
        embeddings_single, labels_single,
        'Single-Session CEBRA - Colored by Cluster',
        OUTPUT_PATH / 'single_session_mds_cluster.pdf'
    )
    
    plot_confusion_matrix(
        labels_single, phylo_labels,
        OUTPUT_PATH / 'single_session_confusion.pdf'
    )

    # ========================================================================
    # Method 2: Multi-session CEBRA (discovery-driven)
    # ========================================================================
    print("\n" + "="*70)
    print("METHOD 2: Multi-Session CEBRA (Discovery-Driven)")
    print("="*70)

    embeddings_multi, model_multi = fit_cebra_multi_session(
        sequences,
        labels=None,  # Discovery-driven
        output_dim=N_EMBEDDING_DIM,
        max_iterations=5000
    )
    
    dist_matrix_multi = compute_embedding_distances(embeddings_multi)
    labels_multi = cluster_from_distances(dist_matrix_multi, N_CLUSTERS)
    
    ari_multi = adjusted_rand_score(labels_multi, phylo_labels)
    sil_multi = silhouette_score(dist_matrix_multi, labels_multi, metric='precomputed')
    
    print(f"\nMulti-Session Results:")
    print(f"  ARI vs Phylogeny: {ari_multi:.3f}")
    print(f"  Silhouette Score: {sil_multi:.3f}")
    
    # Visualize
    plot_embeddings_mds(
        embeddings_multi, phylo_labels,
        'Multi-Session CEBRA - Colored by Phylogeny',
        OUTPUT_PATH / 'multi_session_mds_phylogeny.pdf'
    )
    
    plot_embeddings_mds(
        embeddings_multi, labels_multi,
        'Multi-Session CEBRA - Colored by Cluster',
        OUTPUT_PATH / 'multi_session_mds_cluster.pdf'
    )
    
    plot_confusion_matrix(
        labels_multi, phylo_labels,
        OUTPUT_PATH / 'multi_session_confusion.pdf'
    )
    
    # ========================================================================
    # Method 3: Multi-session CEBRA (hypothesis-driven with phylogeny)
    # ========================================================================
    print("\n" + "="*70)
    print("METHOD 3: Multi-Session CEBRA (Hypothesis-Driven)")
    print("="*70)

    embeddings_hyp, model_hyp = fit_cebra_multi_session(
        sequences,
        labels=phylo_labels,  # Use phylogeny as labels
        output_dim=N_EMBEDDING_DIM,
        max_iterations=5000
    )
    
    dist_matrix_hyp = compute_embedding_distances(embeddings_hyp)
    labels_hyp = cluster_from_distances(dist_matrix_hyp, N_CLUSTERS)
    
    ari_hyp = adjusted_rand_score(labels_hyp, phylo_labels)
    sil_hyp = silhouette_score(dist_matrix_hyp, labels_hyp, metric='precomputed')
    
    print(f"\nHypothesis-Driven Results:")
    print(f"  ARI vs Phylogeny: {ari_hyp:.3f}")
    print(f"  Silhouette Score: {sil_hyp:.3f}")
    
    # Visualize
    plot_embeddings_mds(
        embeddings_hyp, phylo_labels,
        'Hypothesis-Driven CEBRA - Colored by Phylogeny',
        OUTPUT_PATH / 'hypothesis_mds_phylogeny.pdf'
    )
    
    plot_embeddings_mds(
        embeddings_hyp, labels_hyp,
        'Hypothesis-Driven CEBRA - Colored by Cluster',
        OUTPUT_PATH / 'hypothesis_mds_cluster.pdf'
    )
    
    plot_confusion_matrix(
        labels_hyp, phylo_labels,
        OUTPUT_PATH / 'hypothesis_confusion.pdf'
    )
    
    # Save results
    np.save(OUTPUT_PATH / 'embeddings_hypothesis.npy', embeddings_hyp)
    np.save(OUTPUT_PATH / 'distance_matrix_hypothesis.npy', dist_matrix_hyp)
    np.save(OUTPUT_PATH / 'cluster_labels_hypothesis.npy', labels_hyp)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()