from pathlib import Path
import pandas as pd
import re

def find_min_energy(input_csv_path: str, output_csv_path: str):
    """
    Finds the gamma and eta combination with the lowest energy value for each individual.

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path to save the output CSV file.
    """
    try:
        # Load the dataset from the specified CSV file
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return

    # Identify the columns that contain the individual energy values
    energy_columns = [col for col in df.columns if col.startswith('MaxCrit')]

    if not energy_columns:
        print("Error: No energy columns found. Please check the column names in your CSV file.")
        return

    # A list to store the results for each individual
    results = []

    # Iterate over each energy column (each individual)
    for col_name in energy_columns:
        # Find the index of the row with the minimum energy value for the current individual
        min_energy_idx = df[col_name].idxmin()

        # Get the entire row that contains the minimum value
        min_energy_row = df.loc[min_energy_idx]

        # Extract the subject index from the column name using a regular expression
        subj_match = re.search(r'_indiv_(\d+)', col_name)
        if subj_match:
            subj_index = int(subj_match.group(1))
        else:
            # If the pattern doesn't match, use the column name as a fallback
            subj_index = col_name

        # Append the results to our list
        results.append({
            'subj_index': subj_index,
            'gamma': min_energy_row['gamma'],
            'eta': min_energy_row['eta'],
            'energy': min_energy_row[col_name]
        })

    # Create a new DataFrame from our list of results
    results_df = pd.DataFrame(results)

    # Sort the results by subject index for a clean output file
    results_df.sort_values(by='subj_index', inplace=True)

    # Save the final DataFrame to the specified output CSV file
    results_df.to_csv(output_csv_path, index=False)
    print(f"Processing complete. Results have been saved to '{output_csv_path}'.")


if __name__ == '__main__':
    # dataset_name = "shafiei_human_consensus_dataset" # 
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/summary_indiv_energies_for_exp_60_generally_finer_search_animal_0.csv
    dataset_name = "suarez_MaMI_dataset"
    experiment_name = "70_mix_and_match_animal_0" # 64_fine_grid_upper_local_minima_animal_0" # "60_generally_finer_search_animal_0" # 49_shafiei" # 49_suarez_MaMI_100" # 33_suarez_MaMI_size_100_extensive_220_300_iter" # 31_suarez_MaMI_size_100_wider_sweep_57_copy_2_now_run_with_evaluation"
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/shafiei_human_consensus_dataset/46_shafiei/all_metrics_for_46_shafiei.csv
    # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/49_suarez_MaMI_100/summary_indiv_energies_for_exp_49_suarez_MaMI_100.csv"
    base_output_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/") / dataset_name / experiment_name
    input_file = base_output_path / ("summary_indiv_energies_for_exp_" + experiment_name + ".csv")
    # input_file = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/33_suarez_MaMI_size_100_extensive_220_300_iter/summary_indiv_energies_for_exp_33_suarez_MaMI_size_100_extensive_220_300_iter_intermediate_copy.csv")
    save_file_path = input_file.parent / 'min_energy_results.csv'
    
    find_min_energy(input_csv_path=input_file, 
                    output_csv_path=save_file_path)
