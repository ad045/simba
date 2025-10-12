"""
Using the visualization module for connectome analysis pipeline.
This script generates comparison landscape plots for a predefined list of metrics.
"""

# Now add: Look only at region around best location or such. 

import os
import traceback
from pathlib import Path
import pandas as pd
from src.visualization.energy_and_mc_landscape import visualize_gnm_results, generate_entire_df, PipelineVisualizer
import matplotlib

if __name__ == "__main__":
    
    # Config 
    
    # Set to True to generate the plot of average MC vs. lag
    PLOT_AVERAGE_MC_CURVE = True
    
    # Set to True to generate the landscape plots for each metric in the list below
    PLOT_METRIC_LANDSCAPES = True
    
    # List of CSV files containing the GNM sweep results. The script will combine data from all these files.
    df_paths_simulated = [               
        # 26: Data for report 
        "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/26_testing_4_KS_folders_why_so_fast/summary_indiv_energies_for_exp_26_testing_4_KS_folders_why_so_fast.csv"
    ]
    
    df_path_empirical = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/emprirical_analysis/empirical_analysis.csv" # or None
    
    # The folder where the final figures will be saved. It will be created inside the same directory as the first data file.
    save_path = Path(df_paths_simulated[0]).parent / "figures"
    
    # Generate Average MC Curve Plot 
    if PLOT_AVERAGE_MC_CURVE:
        print("Generating Average MC Curve Plot.")
        try:
            # Load the combined dataframe once
            if not df_paths_simulated:
                raise ValueError("df_paths list cannot be empty.")
            
            print("Loading and combining data...")
            combined_df = generate_entire_df(df_paths_simulated)
            
            # Get empirical mc results, if they are provided 
            if df_path_empirical: 
                print("Adding empirical data...")
                df_empirical = pd.read_csv(df_path_empirical)
            else: 
                df_empirical = None    
            
            # Visualize
            if not combined_df.empty:
                visualizer = PipelineVisualizer()
                avg_mc_save_path = Path(save_path) / "average_mc_curve.pdf"
                visualizer.plot_average_mc_curve(combined_df,   
                                                 df_empirical=df_empirical, # as this can also be None. 
                                                 plot_all_individual_mc_curves=False, 
                                                 savepath=avg_mc_save_path, 
                                                 show=False)
            else:
                print("SKIPPING: No data found to generate average MC curve plot.")
        except Exception as e:
            print(f"ERROR: Failed to generate Average MC Curve plot.")
            traceback.print_exc()
        print("\n")

    print("Visualization process completed.")
