# # """
# # YAML-based configuration system integrated with existing ConfigManager.
# # This bridges YAML files with the existing config.py structure.
# # """

# # import argparse
# # import sys

# # from src.config.yaml_loader import YAMLConfigLoader
# # from src.pipeline.orchestrator import run_from_yaml 


# # def main():
# #     """Command-line interface for YAML-based configuration."""
# #     parser = argparse.ArgumentParser(
# #         description="Run GNM-ESN pipeline from YAML configuration"
# #     )
# #     parser.add_argument(
# #         "config",
# #         type=str,
# #         default="configs/example_gnm_random.yaml",
# #         help="Path to YAML configuration file"
# #     )
    
# #     parser.add_argument(
# #         "--validate-only",
# #         action="store_true",
# #         help="Only validate the configuration without running",
# #     )

# #     args = parser.parse_args()
    
# #     try:
# #         # Load and validate config
# #         loader = YAMLConfigLoader(args.config)
# #         print(f"✅ Configuration loaded from: {args.config}")
# #         print(f"✅ Experiment type: {loader.config['experiment']['experiment_type']}")
        
# #         if args.validate_only:
# #             # Create config to validate it can be built
# #             config = loader.create_config_manager()
# #             print("✅ Configuration is valid")
# #             print(f"  - Data resolution: {config.data.resolution}")
# #             print(f"  - Densities: {config.data.density}")
# #             if loader.config['experiment']['type'] in ['gnm_sweep', 'gnm_comprehensive']:
# #                 print(f"  - GNM eta range: {config.gnm.eta_range}")
# #                 print(f"  - GNM gamma range: {config.gnm.gamma_range}")
# #                 print(f"  - Wiring rules: {config.gnm.generative_rules_to_test}")
# #             sys.exit(0)
        
# #         # Run the experiment
# #         run_from_yaml(args.config)
        
# #     except FileNotFoundError as e:
# #         print(f"❌ Error: {e}")
# #         sys.exit(1)
# #     except ValueError as e:
# #         print(f"❌ Configuration error: {e}")
# #         sys.exit(1)
# #     except KeyboardInterrupt:
# #         print("\n✗ Operation cancelled by user")
# #         sys.exit(1)
# #     except Exception as e:
# #         print(f"❌ Error during execution: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         sys.exit(1)


# # if __name__ == "__main__":
# #     main()


# """
# YAML-based configuration system integrated with existing ConfigManager.
# This script orchestrates running an experiment multiple times, manages configuration
# files, and records the total execution time.
# """
# import argparse
# import sys
# import os
# import shutil
# import filecmp
# import time
# import json
# import subprocess

# from src.config.yaml_loader import YAMLConfigLoader
# from src.pipeline.orchestrator import run_from_yaml


# def manage_config_file(config_path: str, output_dir: str):
#     """
#     Copies the config file to the output directory.

#     If a config file already exists in the destination, it compares the two.
#     If they differ, the script exits with an error. If they are the same,
#     the run continues.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     destination_config_path = os.path.join(output_dir, os.path.basename(config_path))

#     if os.path.exists(destination_config_path):
#         if not filecmp.cmp(config_path, destination_config_path, shallow=False):
#             print(f"❌ Error: Configuration file mismatch in '{output_dir}'.")
#             print(f"  The provided config '{config_path}' is different from the existing one.")
#             sys.exit(1)
#         print("✅ Existing config file matches the provided one. Continuing.")
#     else:
#         shutil.copy(config_path, destination_config_path)
#         print(f"✅ Config file copied to '{destination_config_path}'")




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run an experiment series from a YAML configuration file."
#     )
#     parser.add_argument(
#         "config",
#         type=str,
#         help="Path to the YAML configuration file."
#     )
#     parser.add_argument(
#         "-n", "--num_runs",
#         type=int,
#         default=1,
#         help="Number of times to run the experiment."
#     )
#     args = parser.parse_args()

#     start_time = time.time()
#     output_dir = None  # Initialize to handle potential errors before assignment

#     try:
#         #  1. Load Config and Define Paths 
#         print(f"Loading configuration from '{args.config}'...")
#         loader = YAMLConfigLoader(args.config)
#         experiment_name = loader.config.get("experiment", {}).get("name")
#         if not experiment_name:
#             print("❌ Error: 'name' not found under 'experiment' in the config file.")
#             sys.exit(1)
        
#         output_dir = os.path.join("output", "gnm", experiment_name)
#         print(f"✅ Experiment name: '{experiment_name}'")
#         print(f"✅ Output directory set to: '{output_dir}'")

#         #  2. Manage Config File 
#         manage_config_file(args.config, output_dir)

#         #  3. Run Experiment Loop 
#         print(f"\n🚀 Starting {args.num_runs} experiment run(s)...")
#         for i in range(1, args.num_runs + 1):
#             print(f"\n Starting run #{i}/{args.num_runs} ")
#             run_from_yaml(args.config)
#             print(f" Finished run #{i}/{args.num_runs} ")
        
#         print(f"\n✅ All {args.num_runs} runs completed.")

#         #  4. Combine CSVs (if applicable) 
#         combine_script_path = os.path.join("src", "utils", "combine_csvs.py")
#         if os.path.exists(combine_script_path):
#             print("\nCombining 'results' and 'indiv_connectome' CSVs...")
#             subprocess.run(
#                 [sys.executable, combine_script_path, output_dir], 
#                 check=True, 
#                 capture_output=True, 
#                 text=True
#             )
#             print("✅ CSVs combined successfully.")
#         else:
#             print(f"\n- Skipping CSV combination: '{combine_script_path}' not found.")

#     except FileNotFoundError as e:
#         print(f"❌ Error: File not found - {e}", file=sys.stderr)
#         sys.exit(1)
#     except ValueError as e:
#         print(f"❌ Configuration error: {e}", file=sys.stderr)
#         sys.exit(1)
#     except KeyboardInterrupt:
#         print("\n✗ Operation cancelled by user.")
#         sys.exit(1)
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Error during script execution: {e.stderr}", file=sys.stderr)
#         sys.exit(1)
#     except Exception as e:
#         print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
#     finally:
#         #  5. Record Total Time 
#         end_time = time.time()
#         duration = end_time - start_time
        
#         print(f"\nTotal execution time: {duration:.2f} seconds ({duration/60:.2f} minutes).")

#         if output_dir:
#             timing_info = {
#                 "total_duration_seconds": round(duration, 2),
#                 "total_duration_minutes": round(duration / 60, 2),
#                 "start_time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(start_time)),
#                 "end_time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(end_time)),
#                 "config_file": args.config,
#                 "number_of_runs": args.num_runs
#             }
#             timing_file_path = os.path.join(output_dir, "run_duration.json")
#             try:
#                 with open(timing_file_path, 'w') as f:
#                     json.dump(timing_info, f, indent=4)
#                 print(f"✅ Timing information saved to '{timing_file_path}'")
#             except Exception as e:
#                 print(f"❌ Could not save timing information: {e}", file=sys.stderr)


"""
YAML-based configuration system integrated with existing ConfigManager.
This script orchestrates running an experiment multiple times, manages configuration
files, and records the total execution time.
"""
import argparse
import sys
import os
import shutil
import filecmp
import time
import json
import subprocess

from src.config.yaml_loader import YAMLConfigLoader
from src.pipeline.orchestrator import run_from_yaml


def manage_config_file(config_path: str, output_dir: str):
    """
    Copies the config file to the output directory.

    If a config file already exists in the destination, it compares the two.
    If they differ, the script exits with an error. If they are the same,
    the run continues.
    """
    os.makedirs(output_dir, exist_ok=True)
    destination_config_path = os.path.join(output_dir, os.path.basename(config_path))

    if os.path.exists(destination_config_path):
        if not filecmp.cmp(config_path, destination_config_path, shallow=False):
            print(f"❌ Error: Configuration file mismatch in '{output_dir}'.")
            print(f"  The provided config '{config_path}' is different from the existing one.")
            sys.exit(1)
        print("✅ Existing config file matches the provided one. Continuing.")
    else:
        shutil.copy(config_path, destination_config_path)
        print(f"✅ Config file copied to '{destination_config_path}'")


if __name__ == "__main__":
    
    ###########################################################################################
    
    #  Hardcoded Configuration 
    # Set the path to your configuration file here. All command-line arguments will be ignored, and this file will be used.
    HARDCODED_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_suarez_MaMI_dataset.yaml"
    HARDCODED_NUM_RUNS = 2
    
    ###########################################################################################

    #  Argument Parsing (optional, can be removed if always hardcoding) 
    parser = argparse.ArgumentParser(
        description="Run an experiment series from a YAML configuration file."
    )
    parser.add_argument(
        "config",
        type=str,
        nargs='?',  # Make the config argument optional
        default=None,
        help="Path to the YAML configuration file (overridden by HARDCODED_CONFIG_PATH if set)."
    )
    parser.add_argument(
        "-n", "--num_runs",
        type=int,
        default=1,
        help="Number of times to run the experiment (overridden by HARDCODED_NUM_RUNS if set)."
    )
    args = parser.parse_args()

    #  Determine Config and Run Count 
    config_file_to_use = HARDCODED_CONFIG_PATH if HARDCODED_CONFIG_PATH else args.config
    num_runs_to_execute = HARDCODED_NUM_RUNS if HARDCODED_NUM_RUNS is not None else args.num_runs

    if not config_file_to_use:
        print("❌ Error: No configuration file specified. Provide one via command-line or set HARDCODED_CONFIG_PATH.", file=sys.stderr)
        sys.exit(1)

    ###########################################################################################
    
    
    start_time = time.time()
    output_dir = None  # Initialize to handle potential errors before assignment

    try:
        #  1. Load Config and Define Paths 
        print(f"Loading configuration from '{config_file_to_use}'...")
        loader = YAMLConfigLoader(config_file_to_use)
        experiment_name = loader.config.get("experiment", {}).get("name")
        if not experiment_name:
            print("❌ Error: 'name' not found under 'experiment' in the config file.")
            sys.exit(1)
        
        output_dir = os.path.join("output", "gnm", experiment_name)
        print(f"✅ Experiment name: '{experiment_name}'")
        print(f"✅ Output directory set to: '{output_dir}'")

        #  2. Manage Config File 
        manage_config_file(config_file_to_use, output_dir)

        #  3. Run Experiment Loop 
        print(f"\n🚀 Starting {num_runs_to_execute} experiment run(s)...")
        for i in range(1, num_runs_to_execute + 1):
            print(f"\n Starting run #{i}/{num_runs_to_execute} ")
            run_from_yaml(config_file_to_use)
            print(f" Finished run #{i}/{num_runs_to_execute} ")
        
        print(f"\n✅ All {num_runs_to_execute} runs completed.")

        #  4. Combine CSVs (if applicable) 
        combine_script_path = os.path.join("src", "utils", "combine_csvs.py")
        if os.path.exists(combine_script_path):
            print("\nCombining 'results' and 'indiv_connectome' CSVs...")
            subprocess.run(
                [sys.executable, combine_script_path, output_dir], 
                check=True, 
                capture_output=True, 
                text=True
            )
            print("✅ CSVs combined successfully.")
        else:
            print(f"\n- Skipping CSV combination: '{combine_script_path}' not found.")

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
        #  5. Record Total Time 
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nTotal execution time: {duration:.2f} seconds ({duration/60:.2f} minutes).")

        if output_dir:
            timing_info = {
                "total_duration_seconds": round(duration, 2),
                "total_duration_minutes": round(duration / 60, 2),
                "start_time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(start_time)),
                "end_time_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(end_time)),
                # "config_file": config_file_to_use,
                "number_of_runs": num_runs_to_execute
            }
            timing_file_path = os.path.join(output_dir, "run_duration.json")
            try:
                with open(timing_file_path, 'w') as f:
                    json.dump(timing_info, f, indent=4)
                print(f"✅ Timing information saved to '{timing_file_path}'")
            except Exception as e:
                print(f"❌ Could not save timing information: {e}", file=sys.stderr)

