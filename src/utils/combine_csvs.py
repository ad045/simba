import pandas as pd

def merge_csv_files(base_dir, experiment_name):
    # Define paths
    target_file = base_dir / f'all_metrics_for_{experiment_name}.csv'
    # Ensure the target directory exists
    target_file.parent.mkdir(parents=True, exist_ok=True)

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
            # print(f"Deleted file: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    # Delete all folders with "_temp_" in the name if they are empty
    for folder in base_dir.glob('*_temp_*'):
        if folder.is_dir():
            try:
                # Check if folder is empty
                if not any(folder.iterdir()):
                    folder.rmdir()
                    # print(f"Deleted empty folder") # : {folder}")
                else:
                    print(f"Skipped non-empty folder: {folder}")
            except Exception as e:
                print(f"Error deleting {folder}: {e}")
    print("\nDone, deleted all temporary folders.")

# if __name__ == "__main__":
#     base_dir = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/shafiei_human_consensus_dataset/30_shafiei_size_68_copy")
#     merge_csv_files(base_dir)