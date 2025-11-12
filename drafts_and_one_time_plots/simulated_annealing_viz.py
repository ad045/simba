"""
Visualization Notebook: Simulated Annealing Optimization Results

This notebook:
1. Loads original and optimized networks
2. Evaluates portrait divergence for both
3. Visualizes parameter space with arrows showing optimization trajectory
4. Creates comparison plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple

from src.config.path import PathConfig
from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


class SAResultsAnalyzer:
    """Analyze and visualize simulated annealing optimization results."""
    
    def __init__(
        self,
        dataset_name: str,
        original_experiment: str,
        optimized_experiment: str
    ):
        self.dataset_name = dataset_name
        self.original_experiment = original_experiment
        self.optimized_experiment = optimized_experiment
        
        # Setup paths
        self.original_config = PathConfig(
            dataset_name=dataset_name,
            experiment_name=original_experiment
        )
        self.optimized_config = PathConfig(
            dataset_name=dataset_name,
            experiment_name=optimized_experiment
        )
        
        # Load empirical networks
        empirical_path = Path(
            f"/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/"
            f"{dataset_name}/01_connectomes/00_connectomes_density10.npy"
        )
        self.empirical_networks = torch.tensor(
            np.load(empirical_path), dtype=torch.float32
        )
        
        # Initialize evaluator
        self.evaluator = PortraitDivergence()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load original and optimized CSVs."""
        original_csv = self.original_config.output_experiment_dir / \
            f"all_metrics_for_{self.original_experiment}.csv"
        optimized_csv = self.optimized_config.output_experiment_dir / \
            f"all_metrics_for_{self.optimized_experiment}.csv"
        
        df_original = pd.read_csv(original_csv)
        df_optimized = pd.read_csv(optimized_csv)
        
        print(f"Loaded {len(df_original)} original networks")
        print(f"Loaded {len(df_optimized)} optimized networks")
        
        return df_original, df_optimized
    
    def evaluate_networks(
        self,
        df: pd.DataFrame,
        network_dir: Path,
        label: str
    ) -> pd.DataFrame:
        """Evaluate portrait divergence for networks in dataframe."""
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {label}"):
            # Construct filename
            filename = f"net_eta{row['eta']}_gamma{row['gamma']}_ruleMatchingIndex_id{int(row['id']):03d}.npy"
            filepath = network_dir / filename
            
            if not filepath.exists():
                print(f"⚠️ File not found: {filepath}")
                continue
            
            # Load network
            network = np.load(filepath)
            network_tensor = torch.tensor(network, dtype=torch.float32).unsqueeze(0)
            
            # Evaluate
            metrics = self.evaluator(network_tensor, self.empirical_networks)
            avg_divergence = np.mean([v for v in metrics.values()])
            
            results.append({
                'eta': row['eta'],
                'gamma': row['gamma'],
                'id': row['id'],
                'portrait_divergence': avg_divergence,
                'original_eta': row.get('original_eta', row['eta']),
                'original_gamma': row.get('original_gamma', row['gamma'])
            })
        
        return pd.DataFrame(results)
    
    def create_parameter_space_plot(
        self,
        df_original_eval: pd.DataFrame,
        df_optimized_eval: pd.DataFrame,
        save_path: Path = None
    ):
        """Create parameter space visualization with optimization arrows."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # ====================================================================
        # Left plot: Original networks
        # ====================================================================
        ax = axes[0]
        
        scatter = ax.scatter(
            df_original_eval['eta'],
            df_original_eval['gamma'],
            c=df_original_eval['portrait_divergence'],
            cmap='viridis_r',
            s=50,
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5
        )
        
        ax.set_xlabel('η (eta)', fontsize=14)
        ax.set_ylabel('γ (gamma)', fontsize=14)
        ax.set_title('Original Networks\nPortrait Divergence', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Portrait Divergence', fontsize=12)
        
        # ====================================================================
        # Right plot: Optimization trajectories
        # ====================================================================
        ax = axes[1]
        
        # Plot original networks (lighter)
        ax.scatter(
            df_original_eval['eta'],
            df_original_eval['gamma'],
            c='lightgray',
            s=30,
            alpha=0.3,
            label='Original networks'
        )
        
        # Plot optimized networks
        scatter_opt = ax.scatter(
            df_optimized_eval['eta'],
            df_optimized_eval['gamma'],
            c=df_optimized_eval['portrait_divergence'],
            cmap='plasma_r',
            s=100,
            alpha=0.8,
            edgecolors='k',
            linewidth=1,
            marker='*',
            label='Optimized networks'
        )
        
        # Draw arrows from original to optimized
        for idx, row in df_optimized_eval.iterrows():
            ax.annotate(
                '',
                xy=(row['eta'], row['gamma']),
                xytext=(row['original_eta'], row['original_gamma']),
                arrowprops=dict(
                    arrowstyle='->',
                    color='cyan',
                    lw=1.5,
                    alpha=0.6,
                    shrinkA=0,
                    shrinkB=0
                )
            )
        
        ax.set_xlabel('η (eta)', fontsize=14)
        ax.set_ylabel('γ (gamma)', fontsize=14)
        ax.set_title('Optimization Trajectories\nOriginal → Optimized', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        
        cbar = plt.colorbar(scatter_opt, ax=ax)
        cbar.set_label('Portrait Divergence (Optimized)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved parameter space plot to {save_path}")
        
        plt.show()
    
    def create_improvement_histogram(
        self,
        df_optimized_eval: pd.DataFrame,
        save_path: Path = None
    ):
        """Plot histogram of improvements."""
        # Calculate improvement (negative means better)
        # This requires matching original and optimized networks
        # For now, we'll use a simplified approach
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distribution of portrait divergence
        ax.hist(
            df_optimized_eval['portrait_divergence'],
            bins=30,
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )
        
        ax.set_xlabel('Portrait Divergence', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('Distribution of Optimized Network Performance', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_div = df_optimized_eval['portrait_divergence'].mean()
        median_div = df_optimized_eval['portrait_divergence'].median()
        
        ax.axvline(mean_div, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_div:.4f}')
        ax.axvline(median_div, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_div:.4f}')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved histogram to {save_path}")
        
        plt.show()
    
    def create_grid_plot(
        self,
        df_optimized_eval: pd.DataFrame,
        save_path: Path = None
    ):
        """Create 4x4 grid visualization of optimized parameter combinations."""
        # Extract unique grid points (target combinations)
        unique_etas = sorted(df_optimized_eval['eta'].unique())
        unique_gammas = sorted(df_optimized_eval['gamma'].unique())
        
        # Create approximate 4x4 grid
        n_eta = min(4, len(unique_etas))
        n_gamma = min(4, len(unique_gammas))
        
        fig, axes = plt.subplots(n_gamma, n_eta, figsize=(16, 16))
        
        if n_eta == 1 or n_gamma == 1:
            axes = axes.reshape(n_gamma, n_eta)
        
        # Plot each grid cell
        for i, gamma in enumerate(sorted(unique_gammas)[:n_gamma]):
            for j, eta in enumerate(sorted(unique_etas)[:n_eta]):
                ax = axes[i, j]
                
                # Get networks for this grid point
                mask = (np.abs(df_optimized_eval['eta'] - eta) < 1e-6) & \
                       (np.abs(df_optimized_eval['gamma'] - gamma) < 1e-6)
                subset = df_optimized_eval[mask]
                
                if len(subset) > 0:
                    # Plot starting points
                    ax.scatter(
                        subset['original_eta'],
                        subset['original_gamma'],
                        c='lightblue',
                        s=100,
                        alpha=0.5,
                        edgecolors='k',
                        label='Start'
                    )
                    
                    # Plot end point
                    ax.scatter(
                        [eta], [gamma],
                        c='red',
                        s=200,
                        marker='*',
                        edgecolors='k',
                        linewidth=2,
                        label='Target'
                    )
                    
                    # Draw arrows
                    for _, row in subset.iterrows():
                        ax.annotate(
                            '',
                            xy=(eta, gamma),
                            xytext=(row['original_eta'], row['original_gamma']),
                            arrowprops=dict(
                                arrowstyle='->',
                                color='green',
                                lw=1.5,
                                alpha=0.6
                            )
                        )
                    
                    # Statistics
                    mean_div = subset['portrait_divergence'].mean()
                    ax.text(
                        0.05, 0.95,
                        f'Avg Div: {mean_div:.4f}',
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=8
                    )
                
                ax.set_xlabel('η', fontsize=9)
                ax.set_ylabel('γ', fontsize=9)
                ax.set_title(f'η={eta:.2f}, γ={gamma:.2f}', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                
                if i == 0 and j == 0:
                    ax.legend(fontsize=7, loc='upper right')
        
        plt.suptitle('4×4 Parameter Grid: Optimization Trajectories', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved grid plot to {save_path}")
        
        plt.show()
    
    def run_full_analysis(self, output_dir: Path = None):
        """Run complete analysis pipeline."""
        if output_dir is None:
            output_dir = self.optimized_config.output_experiment_dir / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("SIMULATED ANNEALING RESULTS ANALYSIS")
        print("="*60 + "\n")
        
        # Load data
        print("Step 1: Loading data...")
        df_original, df_optimized = self.load_data()
        
        # Evaluate networks
        print("\nStep 2: Evaluating networks...")
        df_original_eval = self.evaluate_networks(
            df_original,
            self.original_config.output_experiment_dir / "generated_networks",
            "original"
        )
        df_optimized_eval = self.evaluate_networks(
            df_optimized,
            self.optimized_config.output_experiment_dir / "generated_networks",
            "optimized"
        )
        
        # Save evaluation results
        df_original_eval.to_csv(output_dir / "original_portrait_divergence.csv", index=False)
        df_optimized_eval.to_csv(output_dir / "optimized_portrait_divergence.csv", index=False)
        
        # Create visualizations
        print("\nStep 3: Creating visualizations...")
        
        self.create_parameter_space_plot(
            df_original_eval,
            df_optimized_eval,
            save_path=output_dir / "parameter_space_comparison.pdf"
        )
        
        self.create_improvement_histogram(
            df_optimized_eval,
            save_path=output_dir / "performance_distribution.pdf"
        )
        
        self.create_grid_plot(
            df_optimized_eval,
            save_path=output_dir / "grid_trajectories.pdf"
        )
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"\nOriginal networks:")
        print(f"  Mean divergence: {df_original_eval['portrait_divergence'].mean():.6f}")
        print(f"  Std divergence:  {df_original_eval['portrait_divergence'].std():.6f}")
        print(f"\nOptimized networks:")
        print(f"  Mean divergence: {df_optimized_eval['portrait_divergence'].mean():.6f}")
        print(f"  Std divergence:  {df_optimized_eval['portrait_divergence'].std():.6f}")
        
        improvement = (df_original_eval['portrait_divergence'].mean() - 
                      df_optimized_eval['portrait_divergence'].mean())
        print(f"\nImprovement: {improvement:.6f}")
        print(f"Improvement %: {(improvement / df_original_eval['portrait_divergence'].mean() * 100):.2f}%")
        
        print(f"\n✅ Analysis complete! Results saved to {output_dir}")
        
        return df_original_eval, df_optimized_eval


# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    # Configuration
    dataset_name = "hcp_schaefer_100_dataset"
    original_experiment = "05_second_big_overnight_run_10201"
    optimized_experiment = "05_1_simulated_annealing"
    
    # Create analyzer
    analyzer = SAResultsAnalyzer(
        dataset_name=dataset_name,
        original_experiment=original_experiment,
        optimized_experiment=optimized_experiment
    )
    
    # Run analysis
    df_original_eval, df_optimized_eval = analyzer.run_full_analysis()