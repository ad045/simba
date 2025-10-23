#!/usr/bin/env python
"""
Quick compute script for calculating missing metrics.
Simplified interface for common use cases.

Usage:
    python quick_compute.py --help
    python quick_compute.py --experiment 60_generally_finer_search_animal_0 --metrics portrait_divergence
    python quick_compute.py --csv /path/to/csv --metrics portrait_divergence global_efficiency
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from compute_missing_metrics import compute_missing_metrics, AVAILABLE_METRICS


def main():
    parser = argparse.ArgumentParser(
        description="Quick compute missing metrics for GNM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available metrics:
  {', '.join(AVAILABLE_METRICS.keys())}

Examples:
  # By experiment name (auto-detects paths):
  python quick_compute.py --experiment 60_generally_finer_search_animal_0 --metrics portrait_divergence
  
  # By direct CSV path:
  python quick_compute.py --csv /path/to/results.csv --metrics portrait_divergence global_efficiency
  
  # Multiple metrics with more workers:
  python quick_compute.py --experiment 60_generally_finer_search_animal_0 \\
      --metrics portrait_divergence global_efficiency modularity \\
      --workers 16
  
  # Calculate all graph measures:
  python quick_compute.py --experiment 60_generally_finer_search_animal_0 \\
      --metrics global_efficiency modularity avg_clustering avg_degree density_bct \\
                transitivity avg_edge_distance wiring_cost char_path_length
        """
    )
    
    # Input specification (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--experiment', '-e', type=str,
                            help='Experiment name (e.g., 60_generally_finer_search_animal_0)')
    input_group.add_argument('--csv', '-c', type=Path,
                            help='Direct path to CSV file')
    
    # Metrics specification
    parser.add_argument('--metrics', '-m', nargs='+', required=True,
                       choices=list(AVAILABLE_METRICS.keys()),
                       help='Metrics to calculate')
    
    # Optional arguments
    parser.add_argument('--dataset', '-d', type=str, default='suarez_MaMI_dataset',
                       help='Dataset name (default: suarez_MaMI_dataset)')
    parser.add_argument('--base-path', type=Path,
                       default=Path('/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code'),
                       help='Base project path')
    parser.add_argument('--workers', '-w', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save progress every N rows (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually computing')
    
    args = parser.parse_args()
    
    # Construct CSV path
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = (
            args.base_path / 'output' / 'gnm' / args.dataset / args.experiment /
            f'all_metrics_for_{args.experiment}.csv'
        )
    
    # Check if CSV exists
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print("="*80)
    print("QUICK COMPUTE - MISSING METRICS")
    print("="*80)
    print(f"\n📄 CSV: {csv_path}")
    print(f"📊 Metrics to calculate: {', '.join(args.metrics)}")
    print(f"⚙️  Workers: {args.workers}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No calculations will be performed")
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\nDataFrame info:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        for metric in args.metrics:
            if metric in df.columns:
                n_missing = df[metric].isna().sum()
                print(f"\n  {metric}:")
                print(f"    Exists: Yes")
                print(f"    Missing values: {n_missing}/{len(df)}")
            else:
                print(f"\n  {metric}:")
                print(f"    Exists: No (will be calculated for all rows)")
        
        sys.exit(0)
    
    # Config for evaluation criteria
    config_dict = {
        'gnm': {
            'evaluation_metrics': ['degree_ks', 'clustering_ks', 'edge_length_ks', 'betweenness_ks']
        }
    }
    
    # Run computation
    try:
        df_updated = compute_missing_metrics(
            csv_path=csv_path,
            metrics_to_calculate=args.metrics,
            empirical_networks_path=None,  # Auto-detect
            distance_matrix_path=None,  # Auto-detect
            config_dict=config_dict,
            n_workers=args.workers,
            save_every=args.save_every
        )
        
        print("\n" + "="*80)
        print("✅ SUCCESS")
        print("="*80)
        print(f"Updated CSV saved to: {csv_path}")
        print(f"Total rows: {len(df_updated)}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Note: Progress has been saved up to the last checkpoint")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())