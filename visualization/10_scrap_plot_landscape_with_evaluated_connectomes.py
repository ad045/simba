
"""
Using the visualization module for connectome analysis pipeline.
This script generates only the Voronoi landscape plot for a predefined list of metrics.
"""

import os
import traceback
from pathlib import Path
import pandas as pd
from src.visualization.energy_and_mc_landscape import generate_entire_df, PipelineVisualizer

# This will close all plots after saving, preventing them from displaying in a loop.
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')


def plot_landscape_with_points_of_visualized_connectomes(path_to_experiment): 
    
    metric = "MaxCriteria" 
   
    
    # Get df of evaluated comninations
    df_aimed_for_and_true_combinations = pd.read_csv(evaluated_combinations_path)
    df_visualized_connectomes = pd.DataFrame()
    df_visualized_connectomes["eta"] = pd.to_numeric(df_aimed_for_and_true_combinations["found_eta"], errors='coerce')
    df_visualized_connectomes["gamma"] = pd.to_numeric(df_aimed_for_and_true_combinations["found_gamma"], errors='coerce')
            
    
    # get df of big csv file for metrics 
    experiment_name = str(path_to_experiment).split("/")[-1]
    df_metrics_path = path_to_experiment / f"summary_all_metrics_for_exp_{experiment_name}.csv"
    df = pd.read_csv(df_metrics_path)
    
    # Column name
    metric_col_name = next(col for col in df.columns if metric in col)

    
    visualizer = PipelineVisualizer() # Initialize the visualizer once
    
    # --- Create and save the plot ---
    plot_title = visualizer._format_plot_title(metric_col_name)
    
    figure_save_name = f"voronoi_landscape_{metric}.pdf"
    full_save_path = result_folder / figure_save_name

    fig, ax = visualizer.plot_metric_landscape_voronoi(
        df, 
        title=plot_title,
        metric_name=metric_col_name,
        savepath=full_save_path,
        dot_color="steelblue", 
        show=False,
        show_dots=False,
        annotate_extremes=True, 
        estimated_indiv_connectomes=df_visualized_connectomes
    )
    
    matplotlib.pyplot.close() # Close the plot after saving to free memory, if returned fig and ax are not used.
    
    
    print(f"Successfully generated plot for {metric} at {full_save_path}\n")
            
            
    
    
    
if __name__ == "__main__":
    
        
    path_to_experiment = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/26_testing_4_KS_folders_why_so_fast") # /all_generated_networks")
    result_folder = path_to_experiment / "connectome_plots"
    evaluated_combinations_path = result_folder / "evaluated_eta_and_gamma_combinations.csv"


    plot_landscape_with_points_of_visualized_connectomes(path_to_experiment=path_to_experiment)
