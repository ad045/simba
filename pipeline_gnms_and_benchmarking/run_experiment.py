"""
This script orchestrates running an experiment multiple times, manages configuration
files, and records the total execution time.
"""

# Set multiprocessing method BEFORE any other imports (for MacOS)
import multiprocessing as mp
import os
import sys

if __name__ == "__main__":
    # Force spawn method for macOS compatibility
    mp.set_start_method('spawn', force=True)
    
    # Disable threading in numeric libraries
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
###########################################################################################

import argparse
import filecmp
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from src.config.yaml_loader import YAMLConfigLoader
from src.pipeline.orchestrator import run_from_yaml

from config.path import PathConfig

# Repo root, so this file works from any clone. The `data/` and `output/`
# symlinks at the root point at the run tree that used to be hardcoded here.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])


class ExperimentRunner:
    """Manages experiment execution, configuration, and output."""
    
    def __init__(self, config_path: str, num_runs: int):
        self.config_path = config_path
        self.num_runs = num_runs
        self.output_dir: Optional[str] = None
        self.start_time: Optional[float] = None
        
    def load_config(self) -> str:
        """Load YAML config and extract experiment name."""
        print(f"Loading configuration from '{self.config_path}'...")
        loader = YAMLConfigLoader(self.config_path)
        experiment_name = loader.config["experiment"]["name"]
        dataset_name = loader.config["data"]["dataset_name"]
        
        
        self.path_config = PathConfig( # is this even necessary? 
            dataset_name=dataset_name,
            experiment_name=experiment_name, 
        )
        
        if not experiment_name:
            raise ValueError("'name' not found under 'experiment' in the config file.")
        
        self.output_dir = self.path_config.output_experiment_dir 
        print(f"✅ Experiment name: '{experiment_name}'")
        print(f"✅ Output directory set to: '{self.output_dir}'")
        
        return experiment_name
    
    def manage_config_file(self):
        """Copy config file to output directory with conflict detection."""
        os.makedirs(self.output_dir, exist_ok=True)
        dest_path = os.path.join(self.output_dir, os.path.basename(self.config_path))
        
        if os.path.exists(dest_path):
            if not filecmp.cmp(self.config_path, dest_path, shallow=False):
                raise ValueError(
                    f"Configuration file mismatch in '{self.output_dir}'.\n"
                    f"The provided config '{self.config_path}' differs from the existing one."
                )
            print("✅ Existing config file matches the provided one. Continuing.")
        else:
            shutil.copy(self.config_path, dest_path)
            print(f"✅ Config file copied to '{dest_path}'")
    
    def run_experiments(self):
        """Execute the experiment multiple times."""
        print(f"\n   Starting {self.num_runs} experiment run(s)...")
        
        for i in range(1, self.num_runs + 1):
            print(f"\n   Starting run #{i}/{self.num_runs} ⏱️")
            run_from_yaml(self.config_path)
            print(f"✅ Finished run #{i}/{self.num_runs} ✅")
        
        print(f"\n✅ All {self.num_runs} runs completed.")
    
    def combine_csv_files(self):
        """Combine CSV files if the combine script exists."""
        combine_script = Path("src/utils/combine_csvs.py")
        
        if not combine_script.exists():
            print(f"\n    Skipping CSV combination: '{combine_script}' not found.")
            return
        
        print("\n   Combining 'results' and 'indiv_connectome' CSVs...")
        subprocess.run(
            [sys.executable, str(combine_script), self.output_dir],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ CSVs combined successfully.")
    
    def save_timing_info(self, duration: float):
        """Save execution timing information to JSON file."""
        if not self.output_dir:
            return
        
        timing_info = {
            "total_duration_seconds": round(duration, 2),
            "total_duration_minutes": round(duration / 60, 2),
            "start_time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(self.start_time)),
            "end_time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "number_of_runs": self.num_runs
        }
        
        timing_file = os.path.join(self.output_dir, "run_duration.json")
        try:
            with open(timing_file, 'a') as f:
                json.dump(timing_info, f, indent=4)
                f.write("\n")
            print(f"✅ Timing information saved to '{timing_file}'")
        except Exception as e:
            print(f"⚠️  Could not save timing information: {e}", file=sys.stderr)
    
    def run(self):
        """Execute the complete experiment workflow."""
        self.start_time = time.time()
        
        try:
            self.load_config()
            self.manage_config_file()
            self.run_experiments()
            self.combine_csv_files()
            
        except FileNotFoundError as e:
            print(f"❌ Error: File not found - {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"❌ Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during script execution: {e.stderr}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            duration = time.time() - self.start_time
            print(f"\n   Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes).")
            self.save_timing_info(duration)


def parse_arguments(entry_points_hardcoded_path=None, entry_points_hardcoded_runs=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an experiment series from a YAML configuration file."
    )
    
    if entry_points_hardcoded_path:
        parser.add_argument(
            "config",
            type=str,
            nargs='?',
            default=entry_points_hardcoded_path, # None,
            help="Path to the YAML configuration file (overridden by HARDCODED_CONFIG_PATH if set)."
        )
        
    if entry_points_hardcoded_runs:
        parser.add_argument(
            "-n", "--num_runs",
            type=int,
            default=entry_points_hardcoded_runs, # 1,
            help="Number of times to run the experiment (overridden by HARDCODED_NUM_RUNS if set)."
        )
        
    return parser.parse_args()


def main(entry_points_hardcoded_path=None, 
         entry_points_hardcoded_runs=None): 
    """Main entry point for the experiment runner. Has the option to override things, useful for __main__."""

    args = parse_arguments(entry_points_hardcoded_path, entry_points_hardcoded_runs)
    
    # Use hardcoded values if set, otherwise fall back to command-line arguments
    config_path = args.config
    num_runs = args.num_runs
    
    if not config_path:
        print(
            "❌ Error: No configuration file specified. Provide one via command-line.",
            file=sys.stderr
        )
        sys.exit(1)
    
    runner = ExperimentRunner(config_path, num_runs)
    runner.run()


if __name__ == "__main__": 
    entry_points_hardcoded_path = f"{_REPO_ROOT}/configs/config_gnm_run_suarez_MaMI_dataset.yaml"
    entry_points_hardcoded_runs = 2
    main(entry_points_hardcoded_path=entry_points_hardcoded_path, 
         entry_points_hardcoded_runs=entry_points_hardcoded_runs)
