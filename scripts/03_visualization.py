"""
Using the visualization module for connectome analysis pipeline.
Integrates energy landscape plotting for GNM and ESN results.

This script demonstrates how to visualize analysis results from CSV files.
Replace the example paths with your actual data file paths.
"""

import argparse
from pathlib import Path
import pandas as pd
from src.visualization.energy_and_mc_landscape import visualize_gnm_results, PipelineVisualizer


def main():
    """Main visualization function with configurable parameters."""
    parser = argparse.ArgumentParser(description='Visualize GNM/ESN analysis results')
    parser.add_argument('--input-files', nargs='+', required=False,
                        help='CSV files containing analysis results',
                        default=['output/gnm/results.csv'])
    parser.add_argument('--metric', type=str, default='char_path_length',
                        help='Metric to visualize (e.g., char_path_length, modularity, mc_mean)')
    parser.add_argument('--output-dir', type=str, default='output/figures',
                        help='Directory to save visualization results')
    parser.add_argument('--format', type=str, default='pdf',
                        help='Output format (pdf, png, svg)')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    df_paths = [Path(p) for p in args.input_files]
    output_dir = Path(args.output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Visualizing metric: {args.metric} ---")
    print(f"Input files: {[str(p) for p in df_paths]}")
    print(f"Output directory: {output_dir}")
    
    try:
        visualized_metric = visualize_gnm_results(
            df_paths=df_paths,
            metric_to_visualize=args.metric,
            save_dir=output_dir,
            save_format=args.format,
            save_individual=True,
            show_dots=True,
        )
        print(f"Used metric: {visualized_metric}")
        print(f"Visualization completed successfully!")
        print(f"Results saved to: {output_dir}")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    main()