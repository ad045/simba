"""
Utility for analyzing logged pipeline runs.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np


class LogAnalyzer:
    """Analyze logged pipeline runs."""
    
    def __init__(self, log_dir: Path = Path("./output")):
        """Initialize analyzer with log directory."""
        self.log_dir = Path(log_dir)
        self.sessions = self._find_sessions()
    
    def _find_sessions(self) -> List[Dict[str, Any]]:
        """Find all session summaries."""
        sessions = []
        for summary_file in self.log_dir.glob("session_summary_*.json"):
            with open(summary_file, 'r') as f:
                session = json.load(f)
                session['summary_file'] = summary_file
                session['log_file'] = self.log_dir / f"all_runs_{session['session_id']}.jsonl"
                sessions.append(session)
        return sorted(sessions, key=lambda x: x['session_id'], reverse=True)
    
    def list_sessions(self) -> pd.DataFrame:
        """List all available sessions."""
        if not self.sessions:
            print("No sessions found")
            return pd.DataFrame()
        
        session_data = []
        for session in self.sessions:
            session_data.append({
                'session_id': session['session_id'],
                'start_time': session['start_time'],
                'end_time': session.get('end_time', 'In progress'),
                'total_runs': session['total_runs'],
                'experiments': ', '.join(session['experiments'].keys()),
                'duration_seconds': session.get('duration_seconds', 0)
            })
        
        return pd.DataFrame(session_data)
    
    def load_session_runs(self, session_id: str) -> pd.DataFrame:
        """Load all runs from a specific session."""
        session = next((s for s in self.sessions if s['session_id'] == session_id), None)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        runs = []
        log_file = session['log_file']
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                for line in f:
                    runs.append(json.loads(line))
        
        return pd.json_normalize(runs)
    
    def load_global_runs(self) -> pd.DataFrame:
        """Load all runs from the global log."""
        global_log = self.log_dir / "global_runs.jsonl"
        
        if not global_log.exists():
            print("No global log found")
            return pd.DataFrame()
        
        runs = []
        with open(global_log, 'r') as f:
            for line in f:
                runs.append(json.loads(line))
        
        return pd.json_normalize(runs)
    
    def analyze_gnm_sweep(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze GNM parameter sweep results."""
        if session_id:
            df = self.load_session_runs(session_id)
        else:
            df = self.load_global_runs()
        
        # Filter for GNM sweep runs
        gnm_df = df[df['run_type'] == 'gnm_sweep'].copy()
        
        if gnm_df.empty:
            return {"error": "No GNM sweep runs found"}
        
        # Extract parameters and results
        gnm_df['eta'] = gnm_df['parameters.eta']
        gnm_df['gamma'] = gnm_df['parameters.gamma']
        gnm_df['energy'] = gnm_df['results.energy']
        gnm_df['rule'] = gnm_df['parameters.generative_rule']
        
        # Find best parameters
        best_idx = gnm_df['energy'].idxmin()
        best_run = gnm_df.loc[best_idx]
        
        analysis = {
            'total_runs': len(gnm_df),
            'best_parameters': {
                'eta': best_run['eta'],
                'gamma': best_run['gamma'],
                'energy': best_run['energy'],
                'generative_rule': best_run['rule']
            },
            'energy_stats': {
                'mean': gnm_df['energy'].mean(),
                'std': gnm_df['energy'].std(),
                'min': gnm_df['energy'].min(),
                'max': gnm_df['energy'].max()
            },
            'parameter_ranges': {
                'eta': [gnm_df['eta'].min(), gnm_df['eta'].max()],
                'gamma': [gnm_df['gamma'].min(), gnm_df['gamma'].max()]
            },
            'dataframe': gnm_df[['eta', 'gamma', 'energy', 'rule', 'experiment_name', 'timestamp']]
        }
        
        return analysis
    
    def analyze_esn_evaluation(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze ESN evaluation results."""
        if session_id:
            df = self.load_session_runs(session_id)
        else:
            df = self.load_global_runs()
        
        # Filter for ESN evaluation runs
        esn_df = df[df['run_type'] == 'esn_evaluation'].copy()
        
        if esn_df.empty:
            return {"error": "No ESN evaluation runs found"}
        
        # Extract key metrics
        esn_df['mc_mean'] = esn_df['results.mc_mean']
        esn_df['mc_std'] = esn_df['results.mc_std']
        esn_df['subject'] = esn_df['parameters.subject']
        
        # Extract hyperparameters if available
        param_cols = [col for col in esn_df.columns if col.startswith('parameters.') and col != 'parameters.subject']
        
        # Find best configuration
        best_idx = esn_df['mc_mean'].idxmax()
        best_run = esn_df.loc[best_idx]
        
        analysis = {
            'total_runs': len(esn_df),
            'n_subjects': esn_df['subject'].nunique(),
            'best_configuration': {
                'mc_mean': best_run['mc_mean'],
                'mc_std': best_run['mc_std'],
                'parameters': {col.replace('parameters.', ''): best_run[col] 
                              for col in param_cols if pd.notna(best_run[col])}
            },
            'mc_stats': {
                'mean': esn_df['mc_mean'].mean(),
                'std': esn_df['mc_mean'].std(),
                'min': esn_df['mc_mean'].min(),
                'max': esn_df['mc_mean'].max()
            },
            'dataframe': esn_df[['subject', 'mc_mean', 'mc_std', 'experiment_name', 'timestamp'] + param_cols]
        }
        
        return analysis
    
    def plot_gnm_landscape(self, session_id: Optional[str] = None, save_path: Optional[Path] = None):
        """Plot GNM parameter landscape."""
        analysis = self.analyze_gnm_sweep(session_id)
        
        if "error" in analysis:
            print(analysis["error"])
            return
        
        df = analysis['dataframe']
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot of all points
        scatter = axes[0].scatter(df['eta'], df['gamma'], c=df['energy'], 
                                 cmap='viridis', s=50, alpha=0.6)
        axes[0].set_xlabel('η (eta)')
        axes[0].set_ylabel('γ (gamma)')
        axes[0].set_title('GNM Parameter Sweep - All Runs')
        plt.colorbar(scatter, ax=axes[0], label='Energy')
        
        # Mark best point
        best = analysis['best_parameters']
        axes[0].scatter(best['eta'], best['gamma'], color='red', s=200, 
                       marker='*', edgecolors='black', linewidth=2, 
                       label=f'Best: E={best["energy"]:.3f}')
        axes[0].legend()
        
        # Histogram of energies
        axes[1].hist(df['energy'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(best['energy'], color='red', linestyle='--', 
                       linewidth=2, label=f'Best: {best["energy"]:.3f}')
        axes[1].set_xlabel('Energy')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Energy Distribution')
        axes[1].legend()
        
        plt.suptitle(f'GNM Analysis - {len(df)} runs', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_esn_results(self, session_id: Optional[str] = None, save_path: Optional[Path] = None):
        """Plot ESN evaluation results."""
        analysis = self.analyze_esn_evaluation(session_id)
        
        if "error" in analysis:
            print(analysis["error"])
            return
        
        df = analysis['dataframe']
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Memory capacity by subject
        subject_means = df.groupby('subject')['mc_mean'].mean().sort_values()
        axes[0].bar(range(len(subject_means)), subject_means.values)
        axes[0].set_xlabel('Subject Index')
        axes[0].set_ylabel('Mean Memory Capacity')
        axes[0].set_title('Memory Capacity by Subject')
        
        # Overall distribution
        axes[1].hist(df['mc_mean'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(df['mc_mean'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["mc_mean"].mean():.3f}')
        axes[1].axvline(analysis['best_configuration']['mc_mean'], 
                       color='green', linestyle='--', linewidth=2, 
                       label=f'Best: {analysis["best_configuration"]["mc_mean"]:.3f}')
        axes[1].set_xlabel('Memory Capacity')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Memory Capacity Distribution')
        axes[1].legend()
        
        plt.suptitle(f'ESN Analysis - {len(df)} runs, {analysis["n_subjects"]} subjects', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, session_id: str, output_file: Optional[Path] = None):
        """Generate a comprehensive report for a session."""
        session = next((s for s in self.sessions if s['session_id'] == session_id), None)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        report = []
        report.append("=" * 60)
        report.append(f"PIPELINE RUN REPORT - Session {session_id}")
        report.append("=" * 60)
        report.append("")
        
        # Session info
        report.append("SESSION INFORMATION:")
        report.append(f"  Start Time: {session['start_time']}")
        report.append(f"  End Time: {session.get('end_time', 'In progress')}")
        report.append(f"  Duration: {session.get('duration_seconds', 0):.1f} seconds")
        report.append(f"  Total Runs: {session['total_runs']}")
        report.append(f"  Experiments: {', '.join(session['experiments'].keys())}")
        report.append("")
        
        # GNM Analysis
        gnm_analysis = self.analyze_gnm_sweep(session_id)
        if "error" not in gnm_analysis:
            report.append("GNM PARAMETER SWEEP:")
            report.append(f"  Total Runs: {gnm_analysis['total_runs']}")
            report.append(f"  Best Parameters:")
            best = gnm_analysis['best_parameters']
            report.append(f"    - eta: {best['eta']:.4f}")
            report.append(f"    - gamma: {best['gamma']:.4f}")
            report.append(f"    - energy: {best['energy']:.4f}")
            report.append(f"    - rule: {best['generative_rule']}")
            report.append(f"  Energy Statistics:")
            stats = gnm_analysis['energy_stats']
            report.append(f"    - Mean: {stats['mean']:.4f}")
            report.append(f"    - Std: {stats['std']:.4f}")
            report.append(f"    - Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            report.append("")
        
        # ESN Analysis
        esn_analysis = self.analyze_esn_evaluation(session_id)
        if "error" not in esn_analysis:
            report.append("ESN EVALUATION:")
            report.append(f"  Total Runs: {esn_analysis['total_runs']}")
            report.append(f"  Subjects: {esn_analysis['n_subjects']}")
            report.append(f"  Best Configuration:")
            best = esn_analysis['best_configuration']
            report.append(f"    - MC Mean: {best['mc_mean']:.4f}")
            report.append(f"    - MC Std: {best['mc_std']:.4f}")
            if best['parameters']:
                report.append("    - Parameters:")
                for k, v in best['parameters'].items():
                    report.append(f"      {k}: {v}")
            report.append(f"  MC Statistics:")
            stats = esn_analysis['mc_stats']
            report.append(f"    - Mean: {stats['mean']:.4f}")
            report.append(f"    - Std: {stats['std']:.4f}")
            report.append(f"    - Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        else:
            print(report_text)
        
        return report_text


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze pipeline run logs")
    parser.add_argument("command", choices=["list", "analyze", "plot", "report"],
                       help="Command to execute")
    parser.add_argument("--session", help="Session ID to analyze")
    parser.add_argument("--log-dir", default="./output", help="Log directory")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(Path(args.log_dir))
    
    if args.command == "list":
        sessions = analyzer.list_sessions()
        if not sessions.empty:
            print(sessions.to_string())
        else:
            print("No sessions found")
    
    elif args.command == "analyze":
        if args.session:
            print("\nGNM Analysis:")
            gnm = analyzer.analyze_gnm_sweep(args.session)
            if "error" not in gnm:
                print(f"  Best: eta={gnm['best_parameters']['eta']:.3f}, "
                      f"gamma={gnm['best_parameters']['gamma']:.3f}, "
                      f"energy={gnm['best_parameters']['energy']:.3f}")
            
            print("\nESN Analysis:")
            esn = analyzer.analyze_esn_evaluation(args.session)
            if "error" not in esn:
                print(f"  Best MC: {esn['best_configuration']['mc_mean']:.3f}")
        else:
            print("Please specify --session")
    
    elif args.command == "plot":
        if args.session:
            analyzer.plot_gnm_landscape(args.session)
            analyzer.plot_esn_results(args.session)
        else:
            print("Please specify --session")
    
    elif args.command == "report":
        if args.session:
            output_path = Path(args.output) if args.output else None
            analyzer.generate_report(args.session, output_path)
        else:
            print("Please specify --session")