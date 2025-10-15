# # """
# # YAML-based configuration system integrated with existing ConfigManager.
# # This script orchestrates running an experiment multiple times, manages configuration
# # files, and records the total execution time.
# # """



# # ###########################################################################################
# # # ============================================================================
# # # CRITICAL: Set multiprocessing method BEFORE any other imports
# # # This must be the very first code that runs (after docstring)
# # # ============================================================================
# # import multiprocessing as mp
# # import os
# # import sys

# # if __name__ == "__main__":
# #     # Force spawn method for macOS compatibility
# #     mp.set_start_method('spawn', force=True)
    
# #     # Disable threading in numeric libraries
# #     os.environ['OMP_NUM_THREADS'] = '1'
# #     os.environ['MKL_NUM_THREADS'] = '1'
# #     os.environ['OPENBLAS_NUM_THREADS'] = '1'
# #     os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# #     os.environ['NUMEXPR_NUM_THREADS'] = '1'
# # ###########################################################################################


# # ###########################################################################################
# # # HARDCODED CONFIGURATION
# # # Set the path to your configuration file here.
# # # All command-line arguments will be ignored when these are set.
# # ###########################################################################################
# # HARDCODED_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_shafiei_human_consensus_dataset.yaml" 
# # # HARDCODED_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_suarez_MaMI_dataset.yaml"
# # HARDCODED_NUM_RUNS = 2
# # ###########################################################################################



# # import argparse
# # import filecmp
# # import json
# # import os
# # import shutil
# # import subprocess
# # import sys
# # import time
# # from pathlib import Path
# # from typing import Optional

# # from src.config.yaml_loader import YAMLConfigLoader
# # from src.pipeline.orchestrator import run_from_yaml


# # from config.manager import ConfigManager, PathConfig





# # class ExperimentRunner:
# #     """Manages experiment execution, configuration, and output."""
    
# #     def __init__(self, config_path: str, num_runs: int):
# #         self.config_path = config_path
# #         self.num_runs = num_runs
# #         self.output_dir: Optional[str] = None
# #         self.start_time: Optional[float] = None
        
# #     def load_config(self, animal) -> str:
# #         """Load YAML config and extract experiment name."""
# #         print(f"Loading configuration from '{self.config_path}'...")
# #         loader = YAMLConfigLoader(self.config_path)
# #         loader.config["experiment"]["animal"] = animal
# #         experiment_name = loader.config["experiment"]["name"] + f"_{animal}"
# #         loader.config["experiment"]["name"] = experiment_name
# #         dataset_name = loader.config["data"]["dataset_name"]
        
        
# #         self.path_config = PathConfig( # is this even necessary? 
# #             dataset_name=dataset_name,
# #             experiment_name=experiment_name, 
# #         )
        
# #         if not experiment_name:
# #             raise ValueError("'name' not found under 'experiment' in the config file.")
        
# #         self.output_dir = self.path_config.output_experiment_dir # os.path.join("output", "gnm", experiment_name)
# #         print(f"✅ Experiment name: '{experiment_name}'")
# #         print(f"✅ Output directory set to: '{self.output_dir}'")
        
# #         return experiment_name
    
# #     def manage_config_file(self):
# #         """Copy config file to output directory with conflict detection."""
# #         os.makedirs(self.output_dir, exist_ok=True)
# #         dest_path = os.path.join(self.output_dir, os.path.basename(self.config_path))
        
# #         if os.path.exists(dest_path):
# #             if not filecmp.cmp(self.config_path, dest_path, shallow=False):
# #                 raise ValueError(
# #                     f"Configuration file mismatch in '{self.output_dir}'.\n"
# #                     f"The provided config '{self.config_path}' differs from the existing one."
# #                 )
# #             print("✅ Existing config file matches the provided one. Continuing.")
# #         else:
# #             shutil.copy(self.config_path, dest_path)
# #             print(f"✅ Config file copied to '{dest_path}'")
    
# #     def run_experiments(self):
# #         """Execute the experiment multiple times."""
# #         print(f"\n🚀 Starting {self.num_runs} experiment run(s)...")
        
# #         for i in range(1, self.num_runs + 1):
# #             print(f"\n⏱️  Starting run #{i}/{self.num_runs} ⏱️")
# #             run_from_yaml(self.config_path)
# #             print(f"✅ Finished run #{i}/{self.num_runs} ✅")
        
# #         print(f"\n✅ All {self.num_runs} runs completed.")
    
# #     def combine_csv_files(self):
# #         """Combine CSV files if the combine script exists."""
# #         combine_script = Path("src/utils/combine_csvs.py")
        
# #         if not combine_script.exists():
# #             print(f"\nℹ️  Skipping CSV combination: '{combine_script}' not found.")
# #             return
        
# #         print("\n📊 Combining 'results' and 'indiv_connectome' CSVs...")
# #         subprocess.run(
# #             [sys.executable, str(combine_script), self.output_dir],
# #             check=True,
# #             capture_output=True,
# #             text=True
# #         )
# #         print("✅ CSVs combined successfully.")
    
# #     def save_timing_info(self, duration: float):
# #         """Save execution timing information to JSON file."""
# #         if not self.output_dir:
# #             return
        
# #         timing_info = {
# #             "total_duration_seconds": round(duration, 2),
# #             "total_duration_minutes": round(duration / 60, 2),
# #             "start_time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(self.start_time)),
# #             "end_time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
# #             "number_of_runs": self.num_runs
# #         }
        
# #         timing_file = os.path.join(self.output_dir, "run_duration.json")
# #         try:
# #             with open(timing_file, 'a') as f:
# #                 json.dump(timing_info, f, indent=4)
# #                 f.write("\n")
# #             print(f"✅ Timing information saved to '{timing_file}'")
# #         except Exception as e:
# #             print(f"⚠️  Could not save timing information: {e}", file=sys.stderr)
    
# #     def run(self):
# #         """Execute the complete experiment workflow."""
# #         self.start_time = time.time()
        
# #         try:
# #             self.load_config()
# #             self.manage_config_file()
# #             self.run_experiments()
# #             self.combine_csv_files()
            
# #         except FileNotFoundError as e:
# #             print(f"❌ Error: File not found - {e}", file=sys.stderr)
# #             sys.exit(1)
# #         except ValueError as e:
# #             print(f"❌ Configuration error: {e}", file=sys.stderr)
# #             sys.exit(1)
# #         except KeyboardInterrupt:
# #             print("\n❌ Operation cancelled by user.")
# #             sys.exit(1)
# #         except subprocess.CalledProcessError as e:
# #             print(f"❌ Error during script execution: {e.stderr}", file=sys.stderr)
# #             sys.exit(1)
# #         except Exception as e:
# #             print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
# #             import traceback
# #             traceback.print_exc()
# #             sys.exit(1)
# #         finally:
# #             duration = time.time() - self.start_time
# #             print(f"\n⏱️  Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes).")
# #             self.save_timing_info(duration)


# # def parse_arguments() -> argparse.Namespace:
# #     """Parse command-line arguments."""
# #     parser = argparse.ArgumentParser(
# #         description="Run an experiment series from a YAML configuration file."
# #     )
# #     parser.add_argument(
# #         "config",
# #         type=str,
# #         nargs='?',
# #         default=None,
# #         help="Path to the YAML configuration file (overridden by HARDCODED_CONFIG_PATH if set)."
# #     )
# #     parser.add_argument(
# #         "-n", "--num_runs",
# #         type=int,
# #         default=1,
# #         help="Number of times to run the experiment (overridden by HARDCODED_NUM_RUNS if set)."
# #     )
# #     return parser.parse_args()


# # def main():
# #     """Main entry point for the experiment runner."""
# #     args = parse_arguments()
    
# #     # Use hardcoded values if set, otherwise fall back to command-line arguments
# #     config_path = HARDCODED_CONFIG_PATH or args.config
# #     num_runs = HARDCODED_NUM_RUNS if HARDCODED_NUM_RUNS is not None else args.num_runs
    
# #     if not config_path:
# #         print(
# #             "❌ Error: No configuration file specified. "
# #             "Provide one via command-line or set HARDCODED_CONFIG_PATH.",
# #             file=sys.stderr
# #         )
# #         sys.exit(1)
    
# #     runner = ExperimentRunner(config_path, num_runs)
# #     runner.run()


# # if __name__ == "__main__": 
# #     main()



# """
# Run experiments across multiple animals by modifying the config file.
# """

# import yaml
# import subprocess
# import sys
# import time
# from pathlib import Path
# from typing import Optional

# # Configuration
# BASE_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_suarez_MaMI_dataset.yaml"
# TEMP_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/temp_config_animal.yaml"
# ANIMAL_START = 0
# ANIMAL_END = 255
# RUN_SCRIPT = "run_experiment.py"

# NUM_RUNS = 2


# def modify_config_for_animal(base_config_path: str, temp_config_path: str, animal_id: int):
#     """
#     Load the base config, modify the animal number, and save to temp file.
    
#     Args:
#         base_config_path: Path to the original config file
#         temp_config_path: Path where modified config will be saved
#         animal_id: Animal ID to set in the config
#     """
#     with open(base_config_path, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Modify the animal number
#     config['experiment']['animal'] = animal_id
    
#     # Optionally modify the experiment name to include animal ID
#     original_name = config['experiment']['name']
#     config['experiment']['name'] = f"{original_name}_animal_{animal_id}"
    
#     # Save modified config
#     with open(temp_config_path, 'w') as f:
#         yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
#     print(f"✅ Created temp config for animal {animal_id}")


# def run_experiment_for_animal(animal_id: int, temp_config_path: str, run_script: str, num_runs: int):
#     """
#     Run the experiment script with the temporary config.
    
#     Args:
#         animal_id: Animal ID being processed
#         temp_config_path: Path to the temporary config file
#         run_script: Path to the run_experiment.py script
#     """
#     print(f"\n{'='*80}")
#     print(f"🐾 Starting experiment for Animal {animal_id}")
#     print(f"{'='*80}\n")
    
#     try:
#         # Run the experiment script
#         # Note: The script uses HARDCODED_CONFIG_PATH, so we need to modify that
#         # or pass the config path as an argument
#         result = subprocess.run(
#             [sys.executable, run_script, temp_config_path, "-n", str(num_runs)], # [sys.executable, run_script, temp_config_path, num_runs],
#             check=True,
#             capture_output=False,  # Show output in real-time
#             text=True
#         )
#         print(f"\n✅ Completed experiment for Animal {animal_id}")
#         return True
        
#     except subprocess.CalledProcessError as e:
#         print(f"\n❌ Error running experiment for Animal {animal_id}: {e}")
#         return False
#     except KeyboardInterrupt:
#         print(f"\n⚠️ Interrupted during Animal {animal_id}")
#         raise


# def cleanup_temp_config(temp_config_path: str):
#     """Remove the temporary config file."""
#     try:
#         Path(temp_config_path).unlink(missing_ok=True)
#         print(f"🧹 Cleaned up temporary config file")
#     except Exception as e:
#         print(f"⚠️ Could not remove temp config: {e}")


# def main():
#     """Main execution loop."""
#     start_time = time.time()
#     total_animals = ANIMAL_END - ANIMAL_START + 1
#     successful = 0
#     failed = 0
    
#     print(f"\n{'='*80}")
#     print(f"🚀 Starting batch experiment run")
#     print(f"   Animals: {ANIMAL_START} to {ANIMAL_END} (total: {total_animals})")
#     print(f"   Base config: {BASE_CONFIG_PATH}")
#     print(f"{'='*80}\n")
    
#     try:
#         for animal_id in range(ANIMAL_START, ANIMAL_END + 1):
#             try:
#                 # Create modified config
#                 modify_config_for_animal(BASE_CONFIG_PATH, TEMP_CONFIG_PATH, animal_id)
                
#                 # Run experiment
#                 success = run_experiment_for_animal(animal_id, TEMP_CONFIG_PATH, RUN_SCRIPT, NUM_RUNS)
                
#                 if success:
#                     successful += 1
#                 else:
#                     failed += 1
                
#                 # Clean up after each run
#                 cleanup_temp_config(TEMP_CONFIG_PATH)
                
#             except KeyboardInterrupt:
#                 print("\n\n⚠️ User interrupted the batch run")
#                 cleanup_temp_config(TEMP_CONFIG_PATH)
#                 break
#             except Exception as e:
#                 print(f"\n❌ Unexpected error for Animal {animal_id}: {e}")
#                 failed += 1
#                 cleanup_temp_config(TEMP_CONFIG_PATH)
#                 continue
    
#     finally:
#         duration = time.time() - start_time
#         print(f"\n{'='*80}")
#         print(f"📊 Batch Run Summary")
#         print(f"{'='*80}")
#         print(f"   Total animals processed: {successful + failed}/{total_animals}")
#         print(f"   ✅ Successful: {successful}")
#         print(f"   ❌ Failed: {failed}")
#         print(f"   ⏱️ Total time: {duration:.2f}s ({duration/60:.2f} min)")
#         print(f"{'='*80}\n")


# if __name__ == "__main__":
#     main()



"""
Run experiments across multiple animals by modifying the config file.
"""

import yaml
import subprocess
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Optional

# Configuration
BASE_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_suarez_MaMI_dataset.yaml"
TEMP_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/temp_config_animal.yaml"
ANIMAL_START = 0
ANIMAL_END = 255
RUN_SCRIPT = "run_experiment.py"


def modify_config_for_animal(base_config_path: str, temp_config_path: str, animal_id: int):
    """
    Load the base config, modify the animal number, and save to temp file.
    
    Args:
        base_config_path: Path to the original config file
        temp_config_path: Path where modified config will be saved
        animal_id: Animal ID to set in the config
    """
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify the animal number
    config['experiment']['animal'] = animal_id
    
    # Optionally modify the experiment name to include animal ID
    original_name = config['experiment']['name']
    # config['experiment']['name'] = f"{original_name}/{original_name}_animal_{animal_id}"
    config['experiment']['name'] = f"{original_name}_animal_{animal_id}"
    
    # Save modified config
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created temp config for animal {animal_id}")


def run_experiment_for_animal(animal_id: int, temp_config_path: str, run_script: str):
    """
    Run the experiment script with the temporary config.
    
    Args:
        animal_id: Animal ID being processed
        temp_config_path: Path to the temporary config file
        run_script: Path to the run_experiment.py script
    """
    print(f"\n{'='*80}")
    print(f"🐾 Starting experiment for Animal {animal_id}")
    print(f"{'='*80}\n")
    
    try:
        # Run the experiment script
        # Note: The script uses HARDCODED_CONFIG_PATH, so we need to modify that
        # or pass the config path as an argument
        result = subprocess.run(
            [sys.executable, run_script, temp_config_path],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✅ Completed experiment for Animal {animal_id}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running experiment for Animal {animal_id}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted during Animal {animal_id}")
        raise


def cleanup_temp_config(temp_config_path: str):
    """Remove the temporary config file."""
    try:
        Path(temp_config_path).unlink(missing_ok=True)
        print(f"🧹 Cleaned up temporary config file")
    except Exception as e:
        print(f"⚠️ Could not remove temp config: {e}")


def main():
    """Main execution loop."""
    start_time = time.time()
    total_animals = ANIMAL_END - ANIMAL_START + 1
    successful = 0
    failed = 0
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting batch experiment run")
    print(f"   Animals: {ANIMAL_START} to {ANIMAL_END} (total: {total_animals})")
    print(f"   Base config: {BASE_CONFIG_PATH}")
    print(f"{'='*80}\n")
    
    try:
        for animal_id in range(ANIMAL_START, ANIMAL_END + 1):
            try:
                # Create modified config
                modify_config_for_animal(BASE_CONFIG_PATH, TEMP_CONFIG_PATH, animal_id)
                
                # Run experiment
                success = run_experiment_for_animal(animal_id, TEMP_CONFIG_PATH, RUN_SCRIPT)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                # Clean up after each run
                cleanup_temp_config(TEMP_CONFIG_PATH)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ User interrupted the batch run")
                cleanup_temp_config(TEMP_CONFIG_PATH)
                break
            except Exception as e:
                print(f"\n❌ Unexpected error for Animal {animal_id}: {e}")
                failed += 1
                cleanup_temp_config(TEMP_CONFIG_PATH)
                continue
    
    finally:
        duration = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"📊 Batch Run Summary")
        print(f"{'='*80}")
        print(f"   Total animals processed: {successful + failed}/{total_animals}")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️ Total time: {duration:.2f}s ({duration/60:.2f} min)")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()