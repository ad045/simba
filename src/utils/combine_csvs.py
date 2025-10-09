# import pandas as pd
# from pathlib import Path
# import sys
# import os

# def combine_and_cleanup(root_directory, filename_to_find):
#     """
#     Finds all CSVs with a specific name in subdirectories, combines them,
#     and deletes the original files.
#     """
#     root_path = Path(root_directory)
#     output_filename = "summary_" + filename_to_find # .csv"

#     # 1. Find all matching CSV files recursively
#     csv_files = list(root_path.rglob("*" + filename_to_find))

#     if not csv_files:
#         print(f"⚠️ No files matching '{filename_to_find}' found in '{root_path}'.")
#         return

#     print(f"Found {len(csv_files)} files to combine...")

#     # 2. Read all found CSVs into a list of DataFrames
#     df_list = [pd.read_csv(file) for file in csv_files]

#     # 3. Concatenate them into a single DataFrame
#     combined_df = pd.concat(df_list, ignore_index=True)

#     # 4. Save the new combined CSV in the root directory
#     output_path = root_path / output_filename
#     combined_df.to_csv(output_path, index=False)
#     print(f"✅ Combined data saved to '{output_path}'.")

#     # 5. Delete the original files
#     # for file_path in csv_files:
#     #     os.remove(file_path)
#     # print(f"🗑️  Original {len(csv_files)} files have been deleted.")


# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python combine_csv.py <path_to_experiments_folder>")
#         sys.exit(1)
    
#     # The parent folder containing all your experiment runs
#     target_folder = sys.argv[1] 
#     # target_folder = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/17_bigger_connectomes_no_individuals_density_10_2"
#     experiment_name = target_folder.split("/")[-1]
    
#     # The name of the csv file inside each experiment folder
#     csv_name = experiment_name # "*.csv"

#     combine_and_cleanup(target_folder, "all_metrics_for_exp_" + experiment_name + ".csv")
#     combine_and_cleanup(target_folder, "indiv_energies_for_exp_" + experiment_name + ".csv")


import os
import pandas as pd
import shutil
from pathlib import Path

def merge_csv_files(base_dir):
    # Define paths
    target_file = base_dir / 'all_metrics_for_exp_30_shafiei_size_68.csv'
    
    # Collect all result CSV files from base directory (recursively)
    result_files = list(base_dir.glob('**/result_*.csv'))
    
    if not result_files:
        print("No result_*.csv files found in temp folders")
        return
    
    print(f"Found {len(result_files)} result CSV files")
    
    # Read existing target file if it exists
    existing_df = None
    if target_file.exists():
        existing_df = pd.read_csv(target_file)
        print(f"Found existing target file with {len(existing_df)} rows")
    
    # Read all result files and check headers
    dfs = []
    headers = None
    
    for file in result_files:
        try:
            df = pd.read_csv(file)
            
            # Check if headers match
            if headers is None:
                headers = list(df.columns)
            else:
                if list(df.columns) != headers:
                    print(f"WARNING: Header mismatch in {file.name}")
                    print(f"Expected: {headers}")
                    print(f"Got: {list(df.columns)}")
                    return
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(dfs)} files with {len(merged_df)} total rows")
    
    # Append to existing data if it exists
    if existing_df is not None:
        # Check if headers match
        if list(existing_df.columns) != list(merged_df.columns):
            print("ERROR: Headers don't match between existing file and new data")
            print(f"Existing: {list(existing_df.columns)}")
            print(f"New: {list(merged_df.columns)}")
            return
        
        final_df = pd.concat([existing_df, merged_df], ignore_index=True)
        print(f"Appended {len(merged_df)} new rows to {len(existing_df)} existing rows")
    else:
        final_df = merged_df
    
    # Save merged file
    final_df.to_csv(target_file, index=False)
    print(f"Saved merged data to {target_file} ({len(final_df)} total rows)")
    
    # Delete the CSV files that were combined
    for file in result_files:
        try:
            file.unlink()
            print(f"Deleted file: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    # Delete all folders with "_temp_" in the name if they are empty
    for folder in base_dir.glob('*_temp_*'):
        if folder.is_dir():
            try:
                # Check if folder is empty
                if not any(folder.iterdir()):
                    folder.rmdir()
                    print(f"Deleted empty folder: {folder}")
                else:
                    print(f"Skipped non-empty folder: {folder}")
            except Exception as e:
                print(f"Error deleting {folder}: {e}")
    
    print("\nDone!")

# if __name__ == "__main__":
#     base_dir = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/shafiei_human_consensus_dataset/30_shafiei_size_68_copy")
#     merge_csv_files(base_dir)