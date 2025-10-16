import pandas as pd

def find_and_save_matches(mode, # "empirical_conns" or "select_generated_conns"
                          file_estimated_eta_and_gamma, 
                          file_calculated_graph_metrics, 
                          output_file):
    """
    Puts together csv that includes both the estimated eta and gamma of each empirical connectome and their true evaluated graph analysis values.

    Args:
        file_a (str): The path to the first CSV file (e.g., 'a.csv').
        file_b (str): The path to the second CSV file (e.g., 'b.csv').
        output_file (str): The path to save the matched rows CSV file.
    """
    try:
        df_estimated_eta_and_gamma = pd.read_csv(file_estimated_eta_and_gamma)
        df_calculated_graph_metrics = pd.read_csv(file_calculated_graph_metrics)

        # Read the two csv files into pandas DataFrames
        if mode == "empirical_conns": # for the empirical connectomes: Adding the estimated eta and gamma to the calculated metrics 
            matched_df = pd.merge(df_estimated_eta_and_gamma, 
                                df_calculated_graph_metrics, 
                                left_on='subj_index', 
                                right_on='subject_id')
        elif mode == "select_generated_conns": # for the generated connectomes: Selecting the rows of the file with the estimated eta and gamma values
            keys_df = df_estimated_eta_and_gamma[['gamma', 'eta']]
            matched_df = pd.merge(df_calculated_graph_metrics, 
                                  keys_df, 
                                  on=['gamma', 'eta'], 
                                  how='inner')

        else: 
            print("Attention: This mode is not an option.")
        # Save the resulting matched rows to a new csv file
        matched_df.to_csv(output_file, index=False)

        print(f"Successfully found {len(matched_df)} matching rows.")
        print(f"Results saved to '{output_file}'")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure both CSV files are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define the file names
    from pathlib import Path
    
    DATASET_NAME = "suarez_MaMI_dataset"
    PROJECT_NAME ="60_generally_finer_search_animal_0" # 26_testing_4_KS_folders_why_so_fast" # #  "2423_24rough_combined" # 24_testing_4_KS_folders_rougher_grid" # 20_sweep_with_individual_connectomes_eta_-7_and_gamma_-0.2"
    base_output_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output")
    project_path = base_output_path / "gnm" / DATASET_NAME / PROJECT_NAME
    

    file_emp_connectome_estimated_eta_and_gamma = project_path / "min_energy_results.csv"
    # file_emp_connectome_calculated_graph_metrics = base_output_path + "/emprirical_analysis/empirical_analysis.csv"
    file_emp_connectome_calculated_graph_metrics = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm_old/suarez_MaMI_dataset/emprirical_analysis/empirical_analysis_weighted.csv"
    file_generated_connectome_calculated_graph_metrics = project_path / f"summary_all_metrics_for_exp_{PROJECT_NAME}.csv" # summary_all_metrics_for_exp_18_sweep_with_individual_connectomes_larger_eta_span.csv"
    
    output_file = Path(project_path)

    # Run the function
    print("Starting with empirical connectomes.")
    find_and_save_matches(mode="empirical_conns",
                          file_estimated_eta_and_gamma=file_emp_connectome_estimated_eta_and_gamma, 
                          file_calculated_graph_metrics=file_emp_connectome_calculated_graph_metrics, 
                          output_file=output_file / "comparison_empirical_connectomes_with_estiamted_eta_and_gamma_and_graph_analysis.csv")
    
    print("Starting with generated connectomes.")
    find_and_save_matches(mode="select_generated_conns",
                          file_estimated_eta_and_gamma=file_emp_connectome_estimated_eta_and_gamma, 
                          file_calculated_graph_metrics=file_generated_connectome_calculated_graph_metrics, 
                          output_file=output_file / "eval_most_similar_gen_conns_with_graph_analysis.csv") # comparison_generated_connectomes_with_eta_and_gamma_and_graph_analysis.csv")
#     /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes/16_big_sweep_with_individual_connectomes_results.csv
# /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/18_sweep_with_individual_connectomes_larger_eta_span/summary_all_metrics_for_exp_18_sweep_with_individual_connectomes_larger_eta_span.csv
# '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes/summary_all_metrics_for_exp_18_sweep_with_individual_connectomes_larger_eta_span.csv'