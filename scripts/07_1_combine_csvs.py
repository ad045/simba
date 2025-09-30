import pandas as pd
from pathlib import Path
import argparse

# Combines CVSs in the "TEMP" folders (but in a wrong way, I believe? It returns entries for #runs, not #runs**trials_of_run!! But better is to run the script/run file while commenting out 2. 
def _combine_csvs_by_pattern(search_dir: Path, file_pattern: str, output_name: str):
    """
    Finds, combines, and saves CSVs based on a recursive glob pattern.

    Args:
        search_dir (Path): The directory to search within.
        file_pattern (str): The glob pattern for the files to combine (e.g., "result_*.csv").
        output_name (str): The name of the output CSV file.
    """
    print(f"Searching for files matching '{file_pattern}' in '{search_dir}'...")
    
    # Use rglob to find files in the directory and all subdirectories
    all_files = list(search_dir.rglob(file_pattern))

    if not all_files:
        print(f"-> No files found matching pattern: '{file_pattern}'")
        return

    try:
        df_list = [pd.read_csv(f) for f in all_files]
        full_df = pd.concat(df_list, ignore_index=True)

        # The parent directory of the temp folder will be the main experiment output folder.
        # CORRECTED: .parent is a property, not a method, so we remove the parentheses.
        output_path = search_dir.parent / output_name
        full_df.to_csv(output_path, index=False)
        print(f"✅ Combined {len(all_files)} files into: {output_path}")
        
    except Exception as e:
        print(f"❌ Error while processing pattern '{file_pattern}': {e}")


def main(args=None):
    """
    Main function to parse command-line arguments and run the CSV combination.
    """
    if args: 
        experiment_name = args["experiment"]
        temp_results_dir = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/24_testing_4_KS_folders_rougher_grid/24_testing_4_KS_folders_rougher_grid_20250930_050118/24_testing_4_KS_folders_rougher_grid_temp") # args["temp_results_dir"]
        
    else: 
        parser = argparse.ArgumentParser(
            description="""
            A script to manually combine temporary CSV results from a GNM sweep.
            This is useful if the main pipeline was interrupted after the parallel
            simulations finished but before the results could be combined.
            """,
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument(
            "temp_dir",
            type=str,
            help="The full path to the temporary directory containing the partial CSV files.\n(e.g., 'output/gnm/16_big_sweep_with_individual_connectomes_temp')"
        )
        # parser.add_argument(
        #     "experiment_name",
        #     type=str,
        #     help="The base name of the experiment.\n(e.g., '16_big_sweep_with_individual_connectomes')"
        # )
        
        args = parser.parse_args()
        experiment_name = args.temp_dir.split("/")[-1]
        temp_results_dir = Path(args.temp_dir)


    if not temp_results_dir.is_dir():
        print(f"❌ Error: The specified temporary directory does not exist: {temp_results_dir}")
        return

    print("-" * 50)
    print(f"Starting CSV combination for experiment: '{experiment_name}'")
    print(f"Using temporary directory: {temp_results_dir}")
    print("-" * 50)

    # 1. Combine the main result files
    _combine_csvs_by_pattern(
        search_dir=temp_results_dir,
        file_pattern="*result_*.csv",
        output_name=temp_results_dir.parent.parent / f"{experiment_name}_results.csv" #  Path(experiment_name) / f"{experiment_name}_results.csv"
    )

    # 2. Combine the individual connectome energy files
    _combine_csvs_by_pattern(
        search_dir=temp_results_dir,
        file_pattern="*indiv_connectome_energies_*.csv",
        output_name=temp_results_dir.parent.parent / f"{experiment_name}_indiv_connectome_energies_results.csv" # / Path(experiment_name)
    )

    print("\nCombination process complete.")


if __name__ == '__main__':
    folder_to_output = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm")
    experiment = "24_testing_4_KS_folders_rougher_grid_copy_2"
    main(args={"experiment": experiment, 
              "temp_results_dir": folder_to_output / experiment})
# run with 
# scripts/07_1_combine_csvs.py <path> (previously: <experiment_name>)
# i.e.: 
# python scripts/07_1_combine_csvs.py /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/17_bigger_connectomes_no_individuals_density_10/17_bigger_connectomes_no_individuals_density_10_20250925_175325/17_bigger_connectomes_no_individuals_density_10_temp       # 17_bigger_connectomes_no_individuals_density_10_temp
# /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes      # 16_big_sweep_with_individual_connectomes