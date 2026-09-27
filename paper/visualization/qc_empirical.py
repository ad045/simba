import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import from your existing modules
from src.analysis.structural_measures import (
    calculate_modularity,
    calculate_avg_degree,
    calculate_avg_clustering,
    calculate_transitivity,
    calculate_char_path_length,
    calculate_degree_gini
)

from src.analysis.dynamic_measures import (
    calculate_global_efficiency,
)

from src.analysis.kayson_utils import (
    check_density,
    compute_omega,
)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ConnectomeQualityChecker:
    """
    Comprehensive quality checker for empirical connectomes.
    Uses existing metric functions from structural_measures, dynamic_measures, and kayson_utils.
    """
    
    def __init__(self, connectomes, threshold=None):
        """
        Initialize with connectome data.
        
        Parameters:
        -----------
        connectomes : np.ndarray
            Shape (n_connectomes, n_nodes, n_nodes)
        threshold : float, optional
            Threshold for binarizing weighted connectomes
        """
        self.connectomes = connectomes
        self.n_connectomes = connectomes.shape[0]
        self.n_nodes = connectomes.shape[1]
        self.threshold = threshold
        
        # Results storage
        self.metrics = defaultdict(list)
        self.quality_scores = {}
        
    def compute_all_metrics(self):
        """Compute all quality metrics for all connectomes."""
        print(f"Analyzing {self.n_connectomes} connectomes with {self.n_nodes} nodes each...")
        
        for i in range(self.n_connectomes):
            if (i + 1) % 25 == 0:
                print(f"  Processing connectome {i+1}/{self.n_connectomes}")
            
            A = self.connectomes[i]
            self._compute_single_connectome_metrics(A, i)
        
        print("✓ Metric computation complete!")
        self._compute_quality_scores()
        
    def _compute_single_connectome_metrics(self, A, idx):
        """Compute metrics for a single connectome."""
        # Binarize if needed
        if self.threshold is not None:
            A_bin = (A > self.threshold).astype(float)
        else:
            A_bin = (A > 0).astype(float)
        
        # Create NetworkX graph
        G = nx.from_numpy_array(A_bin)
        
        # ===== CONNECTIVITY METRICS =====
        # Number of disconnected nodes
        degrees = dict(G.degree())
        n_disconnected = sum(1 for d in degrees.values() if d == 0)
        self.metrics['n_disconnected'].append(n_disconnected)
        
        # Number of connected components
        n_components = nx.number_connected_components(G)
        self.metrics['n_components'].append(n_components)
        
        # Size of largest component
        if n_components > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            largest_cc_size = len(largest_cc)
        else:
            largest_cc_size = 0
        self.metrics['largest_component_size'].append(largest_cc_size)
        
        # ===== DEGREE DISTRIBUTION =====
        degrees_list = [d for n, d in G.degree()]
        self.metrics['degrees'].append(degrees_list)
        
        # Use calculate_avg_degree from structural_measures
        mean_degree = calculate_avg_degree(G)
        self.metrics['mean_degree'].append(mean_degree)
        self.metrics['std_degree'].append(np.std(degrees_list))
        
        # Degree distribution shape (power law vs uniform)
        # Use coefficient of variation as simple metric
        if mean_degree > 0:
            cv_degree = np.std(degrees_list) / mean_degree
        else:
            cv_degree = 0
        self.metrics['cv_degree'].append(cv_degree)
        
        # Degree Gini coefficient from structural_measures
        gini = calculate_degree_gini(A_bin)
        self.metrics['degree_gini'].append(gini)
        
        # ===== SMALL-WORLD METRICS =====
        if nx.is_connected(G):
            # Clustering coefficient - use calculate_avg_clustering
            clustering = calculate_avg_clustering(G)
            self.metrics['clustering'].append(clustering)
            
            # Characteristic path length - use calculate_char_path_length
            path_length = calculate_char_path_length(G)
            self.metrics['path_length'].append(path_length)
            
            # Global efficiency from dynamic_measures
            global_eff = calculate_global_efficiency(G)
            self.metrics['global_efficiency'].append(global_eff)
            
            # Small-world coefficient (sigma)
            # Compare to random graph
            try:
                # Use check_density from kayson_utils
                density = check_density(A_bin)
                G_rand = nx.erdos_renyi_graph(self.n_nodes, density)
                C_rand = nx.average_clustering(G_rand)
                L_rand = nx.average_shortest_path_length(G_rand)
                
                if C_rand > 0 and L_rand > 0:
                    sigma = (clustering / C_rand) / (path_length / L_rand)
                else:
                    sigma = np.nan
            except:
                sigma = np.nan
            
            self.metrics['small_world_sigma'].append(sigma)
        else:
            self.metrics['clustering'].append(np.nan)
            self.metrics['path_length'].append(np.nan)
            self.metrics['small_world_sigma'].append(np.nan)
            self.metrics['global_efficiency'].append(np.nan)
        
        # ===== WEIGHT DISTRIBUTION =====
        weights = A[A > 0]  # Non-zero weights
        if len(weights) > 0:
            self.metrics['weight_mean'].append(np.mean(weights))
            self.metrics['weight_std'].append(np.std(weights))
            self.metrics['weight_skew'].append(stats.skew(weights))
            self.metrics['weight_kurtosis'].append(stats.kurtosis(weights))
            
            # Check for suspicious clustering of weight values
            # Use histogram to detect modes
            hist, _ = np.histogram(weights, bins=50)
            n_modes = np.sum(hist > np.percentile(hist, 90))
            self.metrics['n_weight_modes'].append(n_modes)
        else:
            for key in ['weight_mean', 'weight_std', 'weight_skew', 
                       'weight_kurtosis', 'n_weight_modes']:
                self.metrics[key].append(np.nan)
        
        # ===== MODULARITY =====
        if nx.is_connected(G) and G.number_of_edges() > 0:
            try:
                # Use calculate_modularity from structural_measures
                modularity = calculate_modularity(G)
                self.metrics['modularity'].append(modularity)
                
                # Get communities for module analysis
                communities = nx.community.greedy_modularity_communities(G)
                n_modules = len(communities)
                self.metrics['n_modules'].append(n_modules)
                
                # Module size balance (should not be too skewed)
                module_sizes = [len(c) for c in communities]
                if len(module_sizes) > 1:
                    module_balance = np.std(module_sizes) / np.mean(module_sizes)
                else:
                    module_balance = 0
                self.metrics['module_balance'].append(module_balance)
            except:
                self.metrics['modularity'].append(np.nan)
                self.metrics['n_modules'].append(np.nan)
                self.metrics['module_balance'].append(np.nan)
        else:
            self.metrics['modularity'].append(np.nan)
            self.metrics['n_modules'].append(np.nan)
            self.metrics['module_balance'].append(np.nan)
        
        # ===== DENSITY =====
        # Use check_density from kayson_utils
        density = check_density(A_bin)
        self.metrics['density'].append(density)
        
        # ===== TRANSITIVITY =====
        # Use calculate_transitivity from structural_measures
        transitivity = calculate_transitivity(G)
        self.metrics['transitivity'].append(transitivity)
        
        # ===== OMEGA (Alternative clustering measure) =====
        # Use compute_omega from kayson_utils
        omega = compute_omega(A_bin)
        self.metrics['omega'].append(omega)
        
    def _compute_quality_scores(self):
        """
        Compute quality scores for each connectome based on multiple criteria.
        Higher score = better quality.
        """
        scores = np.zeros(self.n_connectomes)
        
        # 1. Connectivity (30 points)
        # Fewer disconnected nodes is better
        disconnected_score = 1 - np.array(self.metrics['n_disconnected']) / self.n_nodes
        scores += disconnected_score * 15
        
        # Single connected component is best
        component_score = (np.array(self.metrics['n_components']) == 1).astype(float)
        scores += component_score * 15
        
        # 2. Small-world properties (25 points)
        # Good clustering (should be > 0.3)
        clustering = np.array(self.metrics['clustering'])
        clustering_score = np.clip(clustering / 0.5, 0, 1)
        clustering_score = np.nan_to_num(clustering_score, 0)
        scores += clustering_score * 10
        
        # Small-world sigma > 1
        sigma = np.array(self.metrics['small_world_sigma'])
        sigma_score = np.clip((sigma - 1) / 2, 0, 1)
        sigma_score = np.nan_to_num(sigma_score, 0)
        scores += sigma_score * 15
        
        # 3. Degree distribution (20 points)
        # Heavy-tailed is better (higher CV or Gini)
        cv_degree = np.array(self.metrics['cv_degree'])
        cv_score = np.clip(cv_degree / 1.0, 0, 1)
        scores += cv_score * 20
        
        # 4. Weight distribution (15 points)
        # Moderate skew is okay, extreme values are suspicious
        weight_skew = np.array(self.metrics['weight_skew'])
        skew_score = 1 - np.clip(np.abs(weight_skew - 2) / 5, 0, 1)
        skew_score = np.nan_to_num(skew_score, 0)
        scores += skew_score * 15
        
        # 5. Modularity (10 points)
        # Modularity should be > 0.3
        modularity = np.array(self.metrics['modularity'])
        mod_score = np.clip(modularity / 0.5, 0, 1)
        mod_score = np.nan_to_num(mod_score, 0)
        scores += mod_score * 10
        
        # Normalize to 0-100
        self.quality_scores['overall'] = scores
        self.quality_scores['connectivity'] = disconnected_score * 15 + component_score * 15
        self.quality_scores['small_world'] = clustering_score * 10 + sigma_score * 15
        self.quality_scores['degree_dist'] = cv_score * 20
        self.quality_scores['weight_dist'] = skew_score * 15
        self.quality_scores['modularity'] = mod_score * 10
        
    def plot_degree_distributions(self, n_cols=5, figsize=(20, 12)):
        """Plot degree distributions for all connectomes."""
        n_rows = int(np.ceil(self.n_connectomes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(self.n_connectomes):
            degrees = self.metrics['degrees'][i]
            axes[i].hist(degrees, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Connectome {i}\nCV={self.metrics["cv_degree"][i]:.2f}',
                            fontsize=8)
            axes[i].set_xlabel('Degree', fontsize=7)
            axes[i].set_ylabel('Count', fontsize=7)
            axes[i].tick_params(labelsize=6)
        
        # Hide unused subplots
        for i in range(self.n_connectomes, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Degree Distributions for All Connectomes', 
                    fontsize=16, y=1.001)
        return fig
    
    def plot_topology_overview(self, figsize=(18, 12)):
        """Create overview plots of topological properties."""
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        connectome_ids = np.arange(self.n_connectomes)
        
        # Row 1: Connectivity
        # Disconnected nodes
        axes[0, 0].bar(connectome_ids, self.metrics['n_disconnected'])
        axes[0, 0].set_title('Number of Disconnected Nodes')
        axes[0, 0].set_xlabel('Connectome ID')
        axes[0, 0].axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Threshold')
        axes[0, 0].legend()
        
        # Components
        axes[0, 1].bar(connectome_ids, self.metrics['n_components'])
        axes[0, 1].set_title('Number of Connected Components')
        axes[0, 1].set_xlabel('Connectome ID')
        axes[0, 1].axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Ideal')
        axes[0, 1].legend()
        
        # Density
        axes[0, 2].bar(connectome_ids, self.metrics['density'])
        axes[0, 2].set_title('Network Density')
        axes[0, 2].set_xlabel('Connectome ID')
        
        # Row 2: Small-world properties
        # Clustering
        axes[1, 0].bar(connectome_ids, self.metrics['clustering'])
        axes[1, 0].set_title('Clustering Coefficient')
        axes[1, 0].set_xlabel('Connectome ID')
        axes[1, 0].axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Good')
        axes[1, 0].legend()
        
        # Path length
        axes[1, 1].bar(connectome_ids, self.metrics['path_length'])
        axes[1, 1].set_title('Characteristic Path Length')
        axes[1, 1].set_xlabel('Connectome ID')
        
        # Small-world sigma
        axes[1, 2].bar(connectome_ids, self.metrics['small_world_sigma'])
        axes[1, 2].set_title('Small-World Sigma')
        axes[1, 2].set_xlabel('Connectome ID')
        axes[1, 2].axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Small-world')
        axes[1, 2].legend()
        
        # Row 3: Degree and weight distributions
        # Degree CV
        axes[2, 0].bar(connectome_ids, self.metrics['cv_degree'])
        axes[2, 0].set_title('Degree Distribution CV\n(higher = more heterogeneous)')
        axes[2, 0].set_xlabel('Connectome ID')
        
        # Weight skewness
        axes[2, 1].bar(connectome_ids, self.metrics['weight_skew'])
        axes[2, 1].set_title('Weight Distribution Skewness')
        axes[2, 1].set_xlabel('Connectome ID')
        axes[2, 1].axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Typical')
        axes[2, 1].legend()
        
        # Modularity
        axes[2, 2].bar(connectome_ids, self.metrics['modularity'])
        axes[2, 2].set_title('Modularity')
        axes[2, 2].set_xlabel('Connectome ID')
        axes[2, 2].axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Good')
        axes[2, 2].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_weight_distributions(self, n_cols=5, figsize=(20, 12)):
        """Plot weight distributions for all connectomes."""
        n_rows = int(np.ceil(self.n_connectomes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(self.n_connectomes):
            A = self.connectomes[i]
            weights = A[A > 0]
            
            if len(weights) > 0:
                axes[i].hist(weights, bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Connectome {i}\nSkew={self.metrics["weight_skew"][i]:.2f}',
                                fontsize=8)
                axes[i].set_xlabel('Weight', fontsize=7)
                axes[i].set_ylabel('Count', fontsize=7)
                axes[i].tick_params(labelsize=6)
                axes[i].set_yscale('log')
        
        # Hide unused subplots
        for i in range(self.n_connectomes, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Weight Distributions for All Connectomes (log scale)', 
                    fontsize=16, y=1.001)
        return fig
    
    def plot_modularity_analysis(self, figsize=(18, 6)):
        """Plot modularity-related metrics."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        connectome_ids = np.arange(self.n_connectomes)
        
        # Modularity values
        axes[0].bar(connectome_ids, self.metrics['modularity'])
        axes[0].set_title('Modularity')
        axes[0].set_xlabel('Connectome ID')
        axes[0].axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Good modularity')
        axes[0].legend()
        
        # Number of modules
        axes[1].bar(connectome_ids, self.metrics['n_modules'])
        axes[1].set_title('Number of Modules Detected')
        axes[1].set_xlabel('Connectome ID')
        
        # Module balance
        axes[2].bar(connectome_ids, self.metrics['module_balance'])
        axes[2].set_title('Module Size Balance\n(lower = more balanced)')
        axes[2].set_xlabel('Connectome ID')
        
        plt.tight_layout()
        return fig
    
    def plot_quality_heatmap(self, figsize=(16, 8)):
        """Create heatmap showing quality scores for each connectome."""
        # Prepare data for heatmap
        score_data = np.array([
            self.quality_scores['connectivity'],
            self.quality_scores['small_world'],
            self.quality_scores['degree_dist'],
            self.quality_scores['weight_dist'],
            self.quality_scores['modularity'],
            self.quality_scores['overall']
        ])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(score_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=30)
        
        # Set ticks
        ax.set_yticks(np.arange(6))
        ax.set_yticklabels(['Connectivity\n(30 pts)', 'Small-World\n(25 pts)', 
                           'Degree Dist\n(20 pts)', 'Weight Dist\n(15 pts)',
                           'Modularity\n(10 pts)', 'OVERALL\n(100 pts)'])
        ax.set_xticks(np.arange(0, self.n_connectomes, 10))
        ax.set_xticklabels(np.arange(0, self.n_connectomes, 10))
        ax.set_xlabel('Connectome ID')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Quality Score', rotation=270, labelpad=20)
        
        # Add text annotations for overall score
        for i in range(self.n_connectomes):
            text = ax.text(i, 5, f'{self.quality_scores["overall"][i]:.0f}',
                         ha="center", va="center", color="black", fontsize=6)
        
        plt.title('Connectome Quality Scores', fontsize=14, pad=20)
        plt.tight_layout()
        return fig
    
    def get_summary_dataframe(self):
        """Return a summary DataFrame with all metrics."""
        df = pd.DataFrame({
            'connectome_id': range(self.n_connectomes),
            'n_disconnected': self.metrics['n_disconnected'],
            'n_components': self.metrics['n_components'],
            'largest_component_pct': np.array(self.metrics['largest_component_size']) / self.n_nodes * 100,
            'density': self.metrics['density'],
            'mean_degree': self.metrics['mean_degree'],
            'cv_degree': self.metrics['cv_degree'],
            'degree_gini': self.metrics['degree_gini'],
            'clustering': self.metrics['clustering'],
            'path_length': self.metrics['path_length'],
            'small_world_sigma': self.metrics['small_world_sigma'],
            'global_efficiency': self.metrics['global_efficiency'],
            'transitivity': self.metrics['transitivity'],
            'omega': self.metrics['omega'],
            'weight_skew': self.metrics['weight_skew'],
            'modularity': self.metrics['modularity'],
            'n_modules': self.metrics['n_modules'],
            'quality_score': self.quality_scores['overall']
        })
        return df
    
    def print_summary(self):
        """Print summary statistics."""
        print("=" * 70)
        print("CONNECTOME QUALITY SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 Dataset: {self.n_connectomes} connectomes with {self.n_nodes} nodes each")
        
        print("\n🔌 CONNECTIVITY:")
        print(f"  Connectomes with disconnected nodes: {sum(np.array(self.metrics['n_disconnected']) > 0)}")
        print(f"  Connectomes with >5 disconnected nodes: {sum(np.array(self.metrics['n_disconnected']) > 5)}")
        print(f"  Connectomes with single component: {sum(np.array(self.metrics['n_components']) == 1)}")
        
        print("\n🌐 SMALL-WORLD PROPERTIES:")
        sigma = np.array(self.metrics['small_world_sigma'])
        sigma_valid = sigma[~np.isnan(sigma)]
        if len(sigma_valid) > 0:
            print(f"  Connectomes with small-world property (σ>1): {sum(sigma_valid > 1)}")
            print(f"  Mean small-world sigma: {np.mean(sigma_valid):.2f}")
        
        print("\n📈 DEGREE DISTRIBUTION:")
        cv_mean = np.mean(self.metrics['cv_degree'])
        gini_mean = np.nanmean(self.metrics['degree_gini'])
        print(f"  Mean CV of degree: {cv_mean:.2f}")
        print(f"  Mean Gini coefficient: {gini_mean:.2f}")
        print(f"  {'✓ Heterogeneous (good)' if cv_mean > 0.5 else '⚠ Relatively uniform'}")
        
        print("\n⚖️ WEIGHT DISTRIBUTION:")
        skew = np.array(self.metrics['weight_skew'])
        skew_valid = skew[~np.isnan(skew)]
        if len(skew_valid) > 0:
            print(f"  Mean weight skewness: {np.mean(skew_valid):.2f}")
            print(f"  Connectomes with suspicious skewness (|skew|>5): {sum(np.abs(skew_valid) > 5)}")
        
        print("\n🧩 MODULARITY:")
        mod = np.array(self.metrics['modularity'])
        mod_valid = mod[~np.isnan(mod)]
        if len(mod_valid) > 0:
            print(f"  Mean modularity: {np.mean(mod_valid):.2f}")
            print(f"  Connectomes with good modularity (>0.3): {sum(mod_valid > 0.3)}")
        
        print("\n⭐ OVERALL QUALITY:")
        overall = self.quality_scores['overall']
        print(f"  Mean quality score: {np.mean(overall):.1f}/100")
        print(f"  High quality (>70): {sum(overall > 70)} connectomes")
        print(f"  Medium quality (50-70): {sum((overall >= 50) & (overall <= 70))} connectomes")
        print(f"  Low quality (<50): {sum(overall < 50)} connectomes")
        
        print("\n🚨 POTENTIAL ISSUES:")
        issues = []
        if sum(np.array(self.metrics['n_disconnected']) > 5) > 0:
            issues.append(f"  • {sum(np.array(self.metrics['n_disconnected']) > 5)} connectomes have >5 disconnected nodes")
        if sum(np.array(self.metrics['n_components']) > 1) > 10:
            issues.append(f"  • {sum(np.array(self.metrics['n_components']) > 1)} connectomes are fragmented")
        if len(sigma_valid) > 0 and sum(sigma_valid < 1) > self.n_connectomes * 0.3:
            issues.append(f"  • {sum(sigma_valid < 1)} connectomes lack small-world properties")
        
        if issues:
            print("\n".join(issues))
        else:
            print("  ✓ No major issues detected!")
        
        print("\n" + "=" * 70)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

# Load your connectomes (shape: 225, 100, 100)
# connectomes = np.load('your_connectomes.npy')

# Initialize checker
# checker = ConnectomeQualityChecker(connectomes, threshold=None)

# Compute all metrics
# checker.compute_all_metrics()

# Print summary
# checker.print_summary()

# Generate plots
# fig1 = checker.plot_topology_overview()
# fig2 = checker.plot_quality_heatmap()
# fig3 = checker.plot_degree_distributions()
# fig4 = checker.plot_weight_distributions()
# fig5 = checker.plot_modularity_analysis()

# Get detailed dataframe
# df_summary = checker.get_summary_dataframe()
# df_summary.to_csv('connectome_quality_summary.csv', index=False)

# Identify problematic connectomes
# problematic = df_summary[df_summary['quality_score'] < 50]
# print(f"\nProblematic connectomes: {list(problematic['connectome_id'].values)}")