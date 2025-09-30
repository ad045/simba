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
    energy_columns = [col for col in df.columns if col.startswith('MaxCriteria')]

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
    # --- Configuration ---
    # Set the path to your input CSV file
    from pathlib import Path
    
    PROJECT_NAME = "2423_combined" # 20_sweep_with_individual_connectomes_eta_-7_and_gamma_-0.2"
    base_output_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output"
    project_path = base_output_path + "/gnm/" + PROJECT_NAME
    
    # file_emp_connectome_estimated_eta_and_gamma = project_path + "/min_energy_results.csv"
    # file_emp_connectome_calculated_graph_metrics = base_output_path + "/emprirical_analysis/empirical_analysis.csv"
    # file_generated_connectome_calculated_graph_metrics = project_path + "/" + PROJECT_NAME + ".csv" # summary_all_metrics_for_exp_18_sweep_with_individual_connectomes_larger_eta_span.csv"
    
    # output_file = Path(project_path)

    
    INPUT_FILE = project_path + "/summary_indiv_energies_for_exp_" + PROJECT_NAME + ".csv"
    # INPUT_FILE = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes_indiv_connectome_energies_results.csv'
    # INPUT_FILE = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes_results.csv'
    input_file = Path(INPUT_FILE)
    OUTPUT_FILE = input_file.parent / 'min_energy_results.csv'
    # -------------------

    find_min_energy(INPUT_FILE, OUTPUT_FILE)
