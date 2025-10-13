"""
Main orchestrator for running the complete GNM analysis pipeline.

This script reads a YAML configuration file to determine which steps of the 
pipeline to execute. It uses a PathManager to dynamically handle all file paths,
ensuring that all scripts use a centralized configuration.
"""
import argparse
import sys
import subprocess
from pathlib import Path
from src.config.yaml_loader import YAMLConfigLoader

class PathManager:
    """Manages all file and directory paths for the pipeline from a single project root."""
    def __init__(self, config: dict):
        self.exp_name = config['experiment']['name']
        self.dataset_name = config['data']['dataset_name']
        
        # Base Path
        self.project_root = Path(config['paths']['project_root'])
        
        # Derived Base Dirs
        base_output_dir = self.project_root / "output"
        data_dir = self.project_root / "data"
        
        # Project-specific output directory
        self.project_path = base_output_dir / "gnm" / self.dataset_name / self.exp_name

        # Input data paths
        preprocessed_dir = data_dir / "preprocessed" / self.dataset_name
        self.empirical_connectomes = preprocessed_dir / config['paths']['empirical_connectomes_file']
        self.distance_matrix = preprocessed_dir / config['paths']['distance_matrix_file']
        self.animal_metadata = preprocessed_dir / config['paths']['animal_metadata_file']
        self.empirical_analysis = base_output_dir / config['paths']['empirical_analysis_file']

        # Output file paths
        self.individual_energies = self.project_path / f"summary_indiv_energies_for_exp_{self.exp_name}.csv"
        self.combined_metrics = self.project_path / f"all_metrics_for_exp_{self.exp_name}.csv"
        self.best_params = self.project_path / "min_energy_results.csv"
        self.matched_empirical = self.project_path / "comparison_empirical_connectomes_with_estiamted_eta_and_gamma_and_graph_analysis.csv"
        self.matched_generated = self.project_path / "eval_most_similar_gen_conns_with_graph_analysis.csv"
        self.visualization_output_dir = self.project_path / "figures_voronoi_taxonomic"

    def ensure_dirs(self):
        """Create necessary output directories."""
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.visualization_output_dir.mkdir(parents=True, exist_ok=True)


def run_script(script_path: str, args: list):
    """Helper function to run a python script as a subprocess."""
    command = [sys.executable, script_path] + [str(arg) for arg in args]
    print(f"\n--- Running: {' '.join(command)} ---")
    try:
        subprocess.run(command, check=True, text=True)
        print(f"--- Successfully finished {Path(script_path).name} ---")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Error running {script_path}: {e}", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    
    # Parse 
    parser = argparse.ArgumentParser(description="Run the full GNM analysis pipeline.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load 
    print("--- Loading configuration and setting up paths ---")
    loader = YAMLConfigLoader(args.config)
    config = loader.config
    paths = PathManager(config)
    paths.ensure_dirs()
    
    # Get steps
    steps = config.get('pipeline_steps', {})
    print(f"--- Starting pipeline for experiment: {paths.exp_name} ---")

    # Run experiment 
    if steps.get('run_experiment'):
        num_runs = config.get('experiment', {}).get('num_runs', 1)
        run_script('run_experiment.py', [args.config, '-n', num_runs])

    # Run evaluation
    if steps.get('run_evaluation'):
        run_script('run_evaluation.py', ['--config', args.config])

    # Extract best eta and gamma parameters 
    if steps.get('extract_best_params'):
        run_script('07_2_extract_best_eta_and_gamma_parameter.py', [
            '--input', paths.individual_energies,
            '--output', paths.best_params
        ])
        
    # Save matches 
    if steps.get('save_matches'):
        run_script('08_2_save_matches.py', [
            '--estimated-params', paths.best_params,
            '--empirical-metrics', paths.empirical_analysis,
            '--generated-metrics', paths.combined_metrics,
            '--output-empirical', paths.matched_empirical,
            '--output-generated', paths.matched_generated,
        ])

    # Visualize 
    if steps.get('run_visualization'):
        run_script('07_4_visualization_with_colored_dots.py', [
            '--metrics-file', paths.combined_metrics,
            '--metadata-file', paths.animal_metadata,
            '--best-params-file', paths.best_params,
            '--output-dir', paths.visualization_output_dir,
        ])

    print("\n✅ Full pipeline execution completed for all enabled steps.")

