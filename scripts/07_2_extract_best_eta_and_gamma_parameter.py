from pathlib import Path
import pandas as pd
import re

def find_min_energy(input_csv_path: str, output_csv_path: str, metric_type: str = "energy"):
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
    if metric_type == "energy":
        energy_columns = [col for col in df.columns if col.startswith('MaxCrit')]
    elif metric_type == "portraits":
        energy_columns = [col for col in df.columns if col.startswith('PortraitDiv')]
    elif metric_type == "f1":
        energy_columns = [col for col in df.columns if col.startswith('F1Crit')]
    elif metric_type == "communicability":
        energy_columns = [col for col in df.columns if col.startswith('CommCorr')]
    else:
        print(f"Error: Unknown metric_type '{metric_type}'. Please use 'energy', 'portraits', or 'f1'.")
        return
    

    if not energy_columns:
        print("Error: No energy/portraits columns found. Please check the column names in your CSV file.")
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
    dataset_name = "suarez_MaMI_dataset"
    # dataset_name = "hcp_schaefer_100_dataset"
        # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/hcp_schaefer_100_dataset/76_90000_samples_animal_206/summary_indiv_energies_for_exp_76_90000_samples_animal_206.csv
    # dataset_name = "hcp_schaefer_100_dataset"
# PROJECT_NAME="76_90000_samples_animal_206" "suarez_MaMI_dataset" # hcp_schaefer_100_dataset" # suarez_MaMI_dataset"
    
    # experiment_name ="05_second_big_overnight_run_10201" # 76_90000_samples_animal_206" # 04_big_overnight_run" # 1_first_bigger_run_animal_0" # 76_90000_samples_animal_206" # 60_generally_finer_search_animal_0" # 26_testing_4_KS_folders_why_so_fast" # #  "2423_24rough_combined" # 24_testing_4_KS_folders_rougher_grid" # 20_sweep_with_individual_connectomes_eta_-7_and_gamma_-0.2"
    # experiment_name = "76_90000_samples_animal_206" # 75_10000_samples_hopefully_no_lost_entries_gamma_minus0p1_to_1_animal_206_identical_version_just_without_minus_etc" # 75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206"
    # experiment_name = "06_with_seeds"
    experiment_name = "75_10000_samples_hopefully_no_lost_entries_gamma_-0p1_to_1_animal_206"
    metric_type ="energy" # "portraits"   #  "energy" # "portraits"  # "energy"  #
    # metric_type ="portraits"   #  "energy" # "portraits"  # "energy"  #
    # metric_type = "f1"
    # metric_type = "portraits" # communicability"
    # base_output_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output")
    # project_path = base_output_path / "gnm" / DATASET_NAME / PROJECT_NAME
    
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_0/summary_indiv_energies_for_exp_60_generally_finer_search_animal_0.csv
    # dataset_name = "suarez_MaMI_dataset"
    # experiment_name = "70_mix_and_match_animal_0" # 64_fine_grid_upper_local_minima_animal_0" # "60_generally_finer_search_animal_0" # 49_shafiei" # 49_suarez_MaMI_100" # 33_suarez_MaMI_size_100_extensive_220_300_iter" # 31_suarez_MaMI_size_100_wider_sweep_57_copy_2_now_run_with_evaluation"
    # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/shafiei_human_consensus_dataset/46_shafiei/all_metrics_for_46_shafiei.csv
    # "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/49_suarez_MaMI_100/summary_indiv_energies_for_exp_49_suarez_MaMI_100.csv"
    base_output_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/") / dataset_name / experiment_name
    
    if metric_type == "energy":     
        # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/hcp_schaefer_100_dataset/04_big_overnight_run/summary_indiv_portrait_for_exp_04_big_overnight_run.csv
        input_file = base_output_path / f"summary_indiv_energies_for_exp_{experiment_name}.csv"
        save_file_path = input_file.parent / 'min_energy_results.csv'
    elif metric_type == "portraits": 
        input_file = base_output_path / f"summary_indiv_portrait_for_exp_{experiment_name}.csv"
        save_file_path = input_file.parent / 'min_portrait_results.csv' # TODO: is min or max best?? 
    elif metric_type == "f1":
        input_file = base_output_path / f"summary_indiv_f1_for_exp_{experiment_name}.csv"
        save_file_path = input_file.parent / 'max_f1_results.csv'
    elif metric_type == "communicability":
        input_file = base_output_path / f"summary_indiv_communicability_for_exp_{experiment_name}.csv"
        save_file_path = input_file.parent / 'max_communicability_results.csv'
    else:
        raise ValueError(f"Unknown metric_type '{metric_type}'. Please use 'energy', 'portraits', or 'f1'.")
    # input_file = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/33_suarez_MaMI_size_100_extensive_220_300_iter/summary_indiv_energies_for_exp_33_suarez_MaMI_size_100_extensive_220_300_iter_intermediate_copy.csv")
    
    
    find_min_energy(input_csv_path=input_file, 
                    output_csv_path=save_file_path, 
                    metric_type=metric_type)
