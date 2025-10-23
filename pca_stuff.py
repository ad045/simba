# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.decomposition import PCA
# # from sklearn.manifold import TSNE
# # from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# # from sklearn.metrics import silhouette_score
# # import warnings
# # warnings.filterwarnings('ignore')

# # # File paths
# # names_file = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50.csv'
# # metrics_file = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm_old/suarez_MaMI_dataset/emprirical_analysis/empirical_analysis_binarized.csv'

# # # Load data
# # names_df = pd.read_csv(names_file)
# # metrics_df = pd.read_csv(metrics_file)

# # print(f"Names shape: {names_df.shape}")
# # print(f"Metrics shape: {metrics_df.shape}")

# # # Features to use for clustering
# # feature_cols = [
# #     'avg_communicability', 'global_efficiency', 'modularity', 'avg_clustering',
# #     'avg_degree', 'density_bct', 'transitivity', 'avg_edge_distance',
# #     'wiring_cost', 'char_path_length', 'richclub_n_edges', 'richclub_avg_length',
# #     'mc_mean', 'mc_std'
# # ] + [f'mc_{i}' for i in range(50)]

# # # Extract features
# # X = metrics_df[feature_cols].values
# # print(f"\nFeature matrix shape: {X.shape}")

# # # Standardize features
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # # === DIMENSIONALITY REDUCTION ===
# # print("\n" + "="*50)
# # print("DIMENSIONALITY REDUCTION")
# # print("="*50)

# # # PCA
# # pca = PCA(n_components=2)
# # X_pca = pca.fit_transform(X_scaled)
# # print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# # # t-SNE
# # tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# # X_tsne = tsne.fit_transform(X_scaled)

# # # === CLUSTERING ===
# # print("\n" + "="*50)
# # print("CLUSTERING")
# # print("="*50)

# # # K-Means with different k values
# # silhouette_scores = {}
# # for k in range(2, 11):
# #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
# #     labels = kmeans.fit_predict(X_scaled)
# #     score = silhouette_score(X_scaled, labels)
# #     silhouette_scores[k] = score
# #     print(f"K-Means (k={k}): Silhouette Score = {score:.3f}")

# # # Best k based on silhouette score
# # best_k = max(silhouette_scores, key=silhouette_scores.get)
# # print(f"\nBest k: {best_k} (Silhouette Score: {silhouette_scores[best_k]:.3f})")

# # # Final clustering with best k
# # kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
# # kmeans_labels = kmeans_best.fit_predict(X_scaled)

# # # Hierarchical clustering
# # hierarchical = AgglomerativeClustering(n_clusters=best_k)
# # hier_labels = hierarchical.fit_predict(X_scaled)

# # # DBSCAN
# # dbscan = DBSCAN(eps=3, min_samples=5)
# # dbscan_labels = dbscan.fit_predict(X_scaled)
# # n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
# # print(f"\nDBSCAN found {n_dbscan_clusters} clusters")

# # # === VISUALIZATION ===
# # print("\n" + "="*50)
# # print("CREATING VISUALIZATIONS")
# # print("="*50)

# # fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# # fig.suptitle('Clustering and Dimensionality Reduction Analysis', fontsize=16, y=1.00)

# # # Get order values (row indices as order)
# # order = np.arange(len(metrics_df))

# # # 1. PCA colored by order
# # scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=order, cmap='viridis', s=50, alpha=0.7)
# # axes[0, 0].set_title('PCA (colored by order)')
# # axes[0, 0].set_xlabel('PC1')
# # axes[0, 0].set_ylabel('PC2')
# # plt.colorbar(scatter, ax=axes[0, 0], label='Order')

# # # 2. PCA colored by K-Means clusters
# # scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', s=50, alpha=0.7)
# # axes[0, 1].set_title(f'PCA (K-Means, k={best_k})')
# # axes[0, 1].set_xlabel('PC1')
# # axes[0, 1].set_ylabel('PC2')
# # plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')

# # # 3. t-SNE colored by order
# # scatter = axes[0, 2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=order, cmap='viridis', s=50, alpha=0.7)
# # axes[0, 2].set_title('t-SNE (colored by order)')
# # axes[0, 2].set_xlabel('t-SNE 1')
# # axes[0, 2].set_ylabel('t-SNE 2')
# # plt.colorbar(scatter, ax=axes[0, 2], label='Order')

# # # 4. t-SNE colored by K-Means clusters
# # scatter = axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap='tab10', s=50, alpha=0.7)
# # axes[1, 0].set_title(f't-SNE (K-Means, k={best_k})')
# # axes[1, 0].set_xlabel('t-SNE 1')
# # axes[1, 0].set_ylabel('t-SNE 2')
# # plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')

# # # 5. Hierarchical clustering
# # scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap='tab10', s=50, alpha=0.7)
# # axes[1, 1].set_title(f'PCA (Hierarchical, k={best_k})')
# # axes[1, 1].set_xlabel('PC1')
# # axes[1, 1].set_ylabel('PC2')
# # plt.colorbar(scatter, ax=axes[1, 1], label='Cluster')

# # # 6. DBSCAN clustering
# # scatter = axes[1, 2].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='tab10', s=50, alpha=0.7)
# # axes[1, 2].set_title(f'PCA (DBSCAN, {n_dbscan_clusters} clusters)')
# # axes[1, 2].set_xlabel('PC1')
# # axes[1, 2].set_ylabel('PC2')
# # plt.colorbar(scatter, ax=axes[1, 2], label='Cluster')

# # plt.tight_layout()
# # plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
# # print("\nFigure saved as 'clustering_analysis.png'")
# # plt.show()

# # # === SILHOUETTE SCORE PLOT ===
# # fig, ax = plt.subplots(figsize=(10, 6))
# # k_values = list(silhouette_scores.keys())
# # scores = list(silhouette_scores.values())
# # ax.plot(k_values, scores, 'o-', linewidth=2, markersize=8)
# # ax.set_xlabel('Number of Clusters (k)', fontsize=12)
# # ax.set_ylabel('Silhouette Score', fontsize=12)
# # ax.set_title('K-Means Clustering: Silhouette Score vs Number of Clusters', fontsize=14)
# # ax.grid(True, alpha=0.3)
# # ax.axvline(best_k, color='r', linestyle='--', label=f'Best k={best_k}')
# # ax.legend()
# # plt.tight_layout()
# # plt.savefig('silhouette_scores.png', dpi=300, bbox_inches='tight')
# # print("Silhouette scores plot saved as 'silhouette_scores.png'")
# # plt.show()

# # # === ORDER vs CLUSTER ANALYSIS ===
# # print("\n" + "="*50)
# # print("ORDER vs CLUSTER CORRESPONDENCE")
# # print("="*50)

# # # Create DataFrame with results
# # results_df = pd.DataFrame({
# #     'order': order,
# #     'kmeans_cluster': kmeans_labels,
# #     'hierarchical_cluster': hier_labels,
# #     'dbscan_cluster': dbscan_labels,
# #     'common_name': names_df['common_name'].values if 'common_name' in names_df.columns else ['Unknown'] * len(order)
# # })

# # print("\nCluster distribution:")
# # print(results_df['kmeans_cluster'].value_counts().sort_index())

# # print("\nFirst 20 rows (order vs cluster):")
# # print(results_df.head(20))

# # # Save results
# # results_df.to_csv('clustering_results.csv', index=False)
# # print("\nResults saved to 'clustering_results.csv'")

# # print("\n" + "="*50)
# # print("ANALYSIS COMPLETE")
# # print("="*50)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# import warnings
# warnings.filterwarnings('ignore')

# # File paths
# names_file = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50.csv'
# metrics_file = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm_old/suarez_MaMI_dataset/emprirical_analysis/empirical_analysis_binarized.csv'

# # Load data
# names_df = pd.read_csv(names_file)
# metrics_df = pd.read_csv(metrics_file)

# print("="*70)
# print("PHYLOGENETIC CLUSTERING ANALYSIS")
# print("="*70)
# print(f"\nData loaded: {len(names_df)} animals")

# # Features to use
# feature_cols = [
#     'avg_communicability', 'global_efficiency', 'modularity', 'avg_clustering',
#     'avg_degree', 'density_bct', 'transitivity', 'avg_edge_distance',
#     'wiring_cost', 'char_path_length', 'richclub_n_edges', 'richclub_avg_length',
#     'mc_mean', 'mc_std'
# ] + [f'mc_{i}' for i in range(50)]

# X = metrics_df[feature_cols].values
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Taxonomic levels to analyze
# taxonomic_levels = ['order', 'family', 'genus', 'phylogenetic_group']

# # Print distribution of taxonomic groups
# print("\n" + "-"*70)
# print("TAXONOMIC DISTRIBUTION")
# print("-"*70)
# for level in taxonomic_levels:
#     if level in names_df.columns:
#         counts = names_df[level].value_counts()
#         print(f"\n{level.upper()} ({len(counts)} unique groups):")
#         print(counts.head(10))

# # === DIMENSIONALITY REDUCTION ===
# print("\n" + "-"*70)
# print("DIMENSIONALITY REDUCTION")
# print("-"*70)

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# X_tsne = tsne.fit_transform(X_scaled)
# print("t-SNE computed")

# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
# X_umap = reducer.fit_transform(X_scaled)
# print("UMAP computed")

# # === VISUALIZATION: PCA BY TAXONOMY ===
# print("\n" + "-"*70)
# print("CREATING VISUALIZATIONS")
# print("-"*70)

# # Create color maps for each taxonomic level
# color_maps = {}
# label_encoders = {}

# for level in taxonomic_levels:
#     if level in names_df.columns:
#         le = LabelEncoder()
#         encoded = le.fit_transform(names_df[level].fillna('Unknown'))
#         color_maps[level] = encoded
#         label_encoders[level] = le

# # Figure 1: PCA colored by different taxonomic levels
# fig, axes = plt.subplots(2, 2, figsize=(16, 14))
# fig.suptitle('PCA: Colored by Taxonomic Levels', fontsize=16, y=0.995)

# for idx, level in enumerate(taxonomic_levels):
#     if level in color_maps:
#         ax = axes[idx // 2, idx % 2]
        
#         n_groups = len(label_encoders[level].classes_)
#         scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
#                            c=color_maps[level], 
#                            cmap='tab20' if n_groups <= 20 else 'hsv',
#                            s=60, alpha=0.7, edgecolors='black', linewidths=0.3)
        
#         ax.set_title(f'{level.upper()} ({n_groups} groups)', fontsize=12, fontweight='bold')
#         ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
#         ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
        
#         # Add legend if not too many groups
#         if n_groups <= 10:
#             handles = []
#             labels = []
#             for i, class_name in enumerate(label_encoders[level].classes_):
#                 mask = color_maps[level] == i
#                 if mask.sum() > 0:
#                     handles.append(plt.Line2D([0], [0], marker='o', color='w', 
#                                              markerfacecolor=plt.cm.tab20(i/n_groups), markersize=8))
#                     labels.append(f'{class_name} (n={mask.sum()})')
#             ax.legend(handles, labels, loc='best', fontsize=8, framealpha=0.9)

# plt.tight_layout()
# plt.savefig('pca_by_taxonomy.png', dpi=300, bbox_inches='tight')
# print("Saved: pca_by_taxonomy.png")
# plt.show()

# # Figure 2: t-SNE colored by different taxonomic levels
# fig, axes = plt.subplots(2, 2, figsize=(16, 14))
# fig.suptitle('t-SNE: Colored by Taxonomic Levels', fontsize=16, y=0.995)

# for idx, level in enumerate(taxonomic_levels):
#     if level in color_maps:
#         ax = axes[idx // 2, idx % 2]
        
#         n_groups = len(label_encoders[level].classes_)
#         scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
#                            c=color_maps[level], 
#                            cmap='tab20' if n_groups <= 20 else 'hsv',
#                            s=60, alpha=0.7, edgecolors='black', linewidths=0.3)
        
#         ax.set_title(f'{level.upper()} ({n_groups} groups)', fontsize=12, fontweight='bold')
#         ax.set_xlabel('t-SNE 1', fontsize=10)
#         ax.set_ylabel('t-SNE 2', fontsize=10)
        
#         if n_groups <= 10:
#             handles = []
#             labels = []
#             for i, class_name in enumerate(label_encoders[level].classes_):
#                 mask = color_maps[level] == i
#                 if mask.sum() > 0:
#                     handles.append(plt.Line2D([0], [0], marker='o', color='w', 
#                                              markerfacecolor=plt.cm.tab20(i/n_groups), markersize=8))
#                     labels.append(f'{class_name} (n={mask.sum()})')
#             ax.legend(handles, labels, loc='best', fontsize=8, framealpha=0.9)

# plt.tight_layout()
# plt.savefig('tsne_by_taxonomy.png', dpi=300, bbox_inches='tight')
# print("Saved: tsne_by_taxonomy.png")
# plt.show()

# # Figure 3: UMAP colored by different taxonomic levels
# fig, axes = plt.subplots(2, 2, figsize=(16, 14))
# fig.suptitle('UMAP: Colored by Taxonomic Levels', fontsize=16, y=0.995)

# for idx, level in enumerate(taxonomic_levels):
#     if level in color_maps:
#         ax = axes[idx // 2, idx % 2]
        
#         n_groups = len(label_encoders[level].classes_)
#         scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], 
#                            c=color_maps[level], 
#                            cmap='tab20' if n_groups <= 20 else 'hsv',
#                            s=60, alpha=0.7, edgecolors='black', linewidths=0.3)
        
#         ax.set_title(f'{level.upper()} ({n_groups} groups)', fontsize=12, fontweight='bold')
#         ax.set_xlabel('UMAP 1', fontsize=10)
#         ax.set_ylabel('UMAP 2', fontsize=10)
        
#         if n_groups <= 10:
#             handles = []
#             labels = []
#             for i, class_name in enumerate(label_encoders[level].classes_):
#                 mask = color_maps[level] == i
#                 if mask.sum() > 0:
#                     handles.append(plt.Line2D([0], [0], marker='o', color='w', 
#                                              markerfacecolor=plt.cm.tab20(i/n_groups), markersize=8))
#                     labels.append(f'{class_name} (n={mask.sum()})')
#             ax.legend(handles, labels, loc='best', fontsize=8, framealpha=0.9)

# plt.tight_layout()
# plt.savefig('umap_by_taxonomy.png', dpi=300, bbox_inches='tight')
# print("Saved: umap_by_taxonomy.png")
# plt.show()

# # === CLUSTERING QUALITY ASSESSMENT ===
# print("\n" + "-"*70)
# print("CLUSTERING QUALITY ASSESSMENT")
# print("-"*70)

# # Test K-means with different k values and compare to taxonomy
# results = []

# for level in taxonomic_levels:
#     if level in color_maps:
#         n_groups = len(label_encoders[level].classes_)
#         true_labels = color_maps[level]
        
#         # K-means with number of clusters = number of taxonomic groups
#         kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
#         pred_labels = kmeans.fit_predict(X_scaled)
        
#         ari = adjusted_rand_score(true_labels, pred_labels)
#         nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
#         results.append({
#             'Taxonomic Level': level,
#             'N Groups': n_groups,
#             'ARI': ari,
#             'NMI': nmi
#         })
        
#         print(f"\n{level.upper()}:")
#         print(f"  Number of groups: {n_groups}")
#         print(f"  Adjusted Rand Index: {ari:.3f}")
#         print(f"  Normalized Mutual Info: {nmi:.3f}")

# results_df = pd.DataFrame(results)

# # Figure 4: Clustering quality metrics
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# axes[0].bar(results_df['Taxonomic Level'], results_df['ARI'], color='steelblue', alpha=0.7)
# axes[0].set_ylabel('Adjusted Rand Index', fontsize=12)
# axes[0].set_title('K-Means Agreement with Taxonomy (ARI)', fontsize=12, fontweight='bold')
# axes[0].tick_params(axis='x', rotation=45)
# axes[0].grid(True, alpha=0.3, axis='y')
# axes[0].set_ylim([-0.1, 1.0])

# axes[1].bar(results_df['Taxonomic Level'], results_df['NMI'], color='coral', alpha=0.7)
# axes[1].set_ylabel('Normalized Mutual Information', fontsize=12)
# axes[1].set_title('K-Means Agreement with Taxonomy (NMI)', fontsize=12, fontweight='bold')
# axes[1].tick_params(axis='x', rotation=45)
# axes[1].grid(True, alpha=0.3, axis='y')
# axes[1].set_ylim([-0.1, 1.0])

# plt.tight_layout()
# plt.savefig('clustering_quality_metrics.png', dpi=300, bbox_inches='tight')
# print("\nSaved: clustering_quality_metrics.png")
# plt.show()

# # === SAVE RESULTS ===
# results_output = names_df.copy()
# results_output['pca_1'] = X_pca[:, 0]
# results_output['pca_2'] = X_pca[:, 1]
# results_output['tsne_1'] = X_tsne[:, 0]
# results_output['tsne_2'] = X_tsne[:, 1]
# results_output['umap_1'] = X_umap[:, 0]
# results_output['umap_2'] = X_umap[:, 1]

# results_output.to_csv('taxonomic_clustering_results.csv', index=False)
# print("\nSaved: taxonomic_clustering_results.csv")

# print("\n" + "="*70)
# print("ANALYSIS COMPLETE")
# print("="*70)
# print("\nInterpretation Guide:")
# print("- ARI/NMI close to 0: Network features don't cluster by taxonomy")
# print("- ARI/NMI > 0.3: Moderate correspondence")
# print("- ARI/NMI > 0.6: Strong correspondence")
# print("\nLook for:")
# print("- Do animals from same taxonomic group cluster together?")
# print("- Which taxonomic level shows the strongest clustering?")
# print("- Are there outliers (animals far from their taxonomic group)?")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, skip if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: UMAP not installed. Install with: pip install umap-learn")
    print("Continuing without UMAP...\n")

# File paths
names_file = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50.csv'
metrics_file = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm_old/suarez_MaMI_dataset/emprirical_analysis/empirical_analysis_binarized.csv'

# Load data
names_df = pd.read_csv(names_file)
metrics_df = pd.read_csv(metrics_file)

print("="*70)
print("PHYLOGENETIC CLUSTERING ANALYSIS")
print("="*70)
print(f"\nData loaded: {len(names_df)} animals")

# Features to use
feature_cols = [
    'avg_communicability', 'global_efficiency', 'modularity', 'avg_clustering',
    'avg_degree', 'density_bct', 'transitivity', 'avg_edge_distance',
    'wiring_cost', 'char_path_length', 'richclub_n_edges', 'richclub_avg_length',
    'mc_mean', 'mc_std'
] + [f'mc_{i}' for i in range(50)]

X = metrics_df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Taxonomic levels to analyze
taxonomic_levels = ['order', 'family', 'genus', 'phylogenetic_group']

# Print distribution of taxonomic groups
print("\n" + "-"*70)
print("TAXONOMIC DISTRIBUTION")
print("-"*70)
for level in taxonomic_levels:
    if level in names_df.columns:
        counts = names_df[level].value_counts()
        print(f"\n{level.upper()} ({len(counts)} unique groups):")
        print(counts.head(10))

# === DIMENSIONALITY REDUCTION ===
print("\n" + "-"*70)
print("DIMENSIONALITY REDUCTION")
print("-"*70)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
print("t-SNE computed")

if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    print("UMAP computed")
else:
    X_umap = None
    print("UMAP skipped (not installed)")

# === VISUALIZATION: PCA BY TAXONOMY ===
print("\n" + "-"*70)
print("CREATING VISUALIZATIONS")
print("-"*70)

# Create color maps for each taxonomic level
color_maps = {}
label_encoders = {}

for level in taxonomic_levels:
    if level in names_df.columns:
        le = LabelEncoder()
        encoded = le.fit_transform(names_df[level].fillna('Unknown'))
        color_maps[level] = encoded
        label_encoders[level] = le

# Figure 1: PCA colored by different taxonomic levels
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('PCA: Colored by Taxonomic Levels', fontsize=16, y=0.995)

for idx, level in enumerate(taxonomic_levels):
    if level in color_maps:
        ax = axes[idx // 2, idx % 2]
        
        n_groups = len(label_encoders[level].classes_)
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                           c=color_maps[level], 
                           cmap='tab20' if n_groups <= 20 else 'hsv',
                           s=60, alpha=0.7, edgecolors='black', linewidths=0.3)
        
        ax.set_title(f'{level.upper()} ({n_groups} groups)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
        
        # Add legend if not too many groups
        if n_groups <= 10:
            handles = []
            labels = []
            for i, class_name in enumerate(label_encoders[level].classes_):
                mask = color_maps[level] == i
                if mask.sum() > 0:
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=plt.cm.tab20(i/n_groups), markersize=8))
                    labels.append(f'{class_name} (n={mask.sum()})')
            ax.legend(handles, labels, loc='best', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig('pca_by_taxonomy.png', dpi=300, bbox_inches='tight')
print("Saved: pca_by_taxonomy.png")
plt.show()

# Figure 2: t-SNE colored by different taxonomic levels
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('t-SNE: Colored by Taxonomic Levels', fontsize=16, y=0.995)

for idx, level in enumerate(taxonomic_levels):
    if level in color_maps:
        ax = axes[idx // 2, idx % 2]
        
        n_groups = len(label_encoders[level].classes_)
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                           c=color_maps[level], 
                           cmap='tab20' if n_groups <= 20 else 'hsv',
                           s=60, alpha=0.7, edgecolors='black', linewidths=0.3)
        
        ax.set_title(f'{level.upper()} ({n_groups} groups)', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        
        if n_groups <= 10:
            handles = []
            labels = []
            for i, class_name in enumerate(label_encoders[level].classes_):
                mask = color_maps[level] == i
                if mask.sum() > 0:
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=plt.cm.tab20(i/n_groups), markersize=8))
                    labels.append(f'{class_name} (n={mask.sum()})')
            ax.legend(handles, labels, loc='best', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig('tsne_by_taxonomy.png', dpi=300, bbox_inches='tight')
print("Saved: tsne_by_taxonomy.png")
plt.show()

# Figure 3: UMAP colored by different taxonomic levels
if UMAP_AVAILABLE and X_umap is not None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('UMAP: Colored by Taxonomic Levels', fontsize=16, y=0.995)

    for idx, level in enumerate(taxonomic_levels):
        if level in color_maps:
            ax = axes[idx // 2, idx % 2]
            
            n_groups = len(label_encoders[level].classes_)
            scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], 
                               c=color_maps[level], 
                               cmap='tab20' if n_groups <= 20 else 'hsv',
                               s=60, alpha=0.7, edgecolors='black', linewidths=0.3)
            
            ax.set_title(f'{level.upper()} ({n_groups} groups)', fontsize=12, fontweight='bold')
            ax.set_xlabel('UMAP 1', fontsize=10)
            ax.set_ylabel('UMAP 2', fontsize=10)
            
            if n_groups <= 10:
                handles = []
                labels = []
                for i, class_name in enumerate(label_encoders[level].classes_):
                    mask = color_maps[level] == i
                    if mask.sum() > 0:
                        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=plt.cm.tab20(i/n_groups), markersize=8))
                        labels.append(f'{class_name} (n={mask.sum()})')
                ax.legend(handles, labels, loc='best', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('umap_by_taxonomy.png', dpi=300, bbox_inches='tight')
    print("Saved: umap_by_taxonomy.png")
    plt.show()
else:
    print("Skipping UMAP visualization (not available)")

# === CLUSTERING QUALITY ASSESSMENT ===
print("\n" + "-"*70)
print("CLUSTERING QUALITY ASSESSMENT")
print("-"*70)

# Test K-means with different k values and compare to taxonomy
results = []

for level in taxonomic_levels:
    if level in color_maps:
        n_groups = len(label_encoders[level].classes_)
        true_labels = color_maps[level]
        
        # K-means with number of clusters = number of taxonomic groups
        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(X_scaled)
        
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
        results.append({
            'Taxonomic Level': level,
            'N Groups': n_groups,
            'ARI': ari,
            'NMI': nmi
        })
        
        print(f"\n{level.upper()}:")
        print(f"  Number of groups: {n_groups}")
        print(f"  Adjusted Rand Index: {ari:.3f}")
        print(f"  Normalized Mutual Info: {nmi:.3f}")

results_df = pd.DataFrame(results)

# Figure 4: Clustering quality metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(results_df['Taxonomic Level'], results_df['ARI'], color='steelblue', alpha=0.7)
axes[0].set_ylabel('Adjusted Rand Index', fontsize=12)
axes[0].set_title('K-Means Agreement with Taxonomy (ARI)', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([-0.1, 1.0])

axes[1].bar(results_df['Taxonomic Level'], results_df['NMI'], color='coral', alpha=0.7)
axes[1].set_ylabel('Normalized Mutual Information', fontsize=12)
axes[1].set_title('K-Means Agreement with Taxonomy (NMI)', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim([-0.1, 1.0])

plt.tight_layout()
plt.savefig('clustering_quality_metrics.png', dpi=300, bbox_inches='tight')
print("\nSaved: clustering_quality_metrics.png")
plt.show()

# === SAVE RESULTS ===
results_output = names_df.copy()
results_output['pca_1'] = X_pca[:, 0]
results_output['pca_2'] = X_pca[:, 1]
results_output['tsne_1'] = X_tsne[:, 0]
results_output['tsne_2'] = X_tsne[:, 1]
if UMAP_AVAILABLE and X_umap is not None:
    results_output['umap_1'] = X_umap[:, 0]
    results_output['umap_2'] = X_umap[:, 1]

results_output.to_csv('taxonomic_clustering_results.csv', index=False)
print("\nSaved: taxonomic_clustering_results.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nInterpretation Guide:")
print("- ARI/NMI close to 0: Network features don't cluster by taxonomy")
print("- ARI/NMI > 0.3: Moderate correspondence")
print("- ARI/NMI > 0.6: Strong correspondence")
print("\nLook for:")
print("- Do animals from same taxonomic group cluster together?")
print("- Which taxonomic level shows the strongest clustering?")
print("- Are there outliers (animals far from their taxonomic group)?")