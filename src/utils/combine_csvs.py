import pandas as pd
from pathlib import Path
import sys
import os

def combine_and_cleanup(root_directory, filename_to_find):
    """
    Finds all CSVs with a specific name in subdirectories, combines them,
    and deletes the original files.
    """
    root_path = Path(root_directory)
    output_filename = "combined_results" + filename_to_find + "_summary" + ".csv"

    # 1. Find all matching CSV files recursively
    csv_files = list(root_path.rglob(filename_to_find + "_results.csv"))

    if not csv_files:
        print(f"⚠️ No files matching '{filename_to_find}' found in '{root_path}'.")
        return

    print(f"Found {len(csv_files)} files to combine...")

    # 2. Read all found CSVs into a list of DataFrames
    df_list = [pd.read_csv(file) for file in csv_files]

    # 3. Concatenate them into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # 4. Save the new combined CSV in the root directory
    output_path = root_path / output_filename
    combined_df.to_csv(output_path, index=False)
    print(f"✅ Combined data saved to '{output_path}'.")

    # 5. Delete the original files
    for file_path in csv_files:
        os.remove(file_path)
    print(f"🗑️  Original {len(csv_files)} files have been deleted.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python combine_csv.py <path_to_experiments_folder>")
        sys.exit(1)
    
    # The parent folder containing all your experiment runs
    target_folder = sys.argv[1] 
    
    experiment_name = target_folder.split("/")[-1]
    
    # The name of the csv file inside each experiment folder
    csv_name = experiment_name # "*.csv"

    combine_and_cleanup(target_folder, csv_name)