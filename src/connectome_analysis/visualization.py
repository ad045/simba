"""
Using the visualization module for connectome analysis pipeline.
Integrates energy landscape plotting for (in future both) GNM (and ESN) results.
"""

from pathlib import Path
from src.connectome_analysis.visualization.energy_and_mc_landscape import (visualize_gnm_results)


# Usage 
if __name__ == "__main__":
    
    print("GNM visualization:")
    
    # name_of_energy_metric = "MaxCriteria(DegreeKS_ClusteringKS)"
    
    # File path
    # One old file (not the hyper large one)
    # df_name = "/Users/adrian/Documents/01_projects/14_4D_lab/OLD_output_3/default_folder/binary_evaluations_resultsdistance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_400.csv"
    df_name = "/Users/adrian/Documents/01_projects/14_4D_lab/output/00_gnm_experiments/binary_evaluations_resultsdistance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_227.csv"
    # Random search (new one, currently still being created)
    # df_name = "/Users/adrian/Documents/01_projects/14_4D_lab/output/default_folder/binary_evaluations_resultsdistance_rel_powerlaw_pref_rel_powerlaw_gen_rule_MatchingIndex_num_iterations_400.csv"
    
    # Extract folder name for organization
    folder_name = Path(df_name).stem
    save_path = Path("output/figures") / folder_name
    
    try:
        name_of_energy_metric = visualize_gnm_results(
            df_path=df_name,
            # name_of_energy_metric=name_of_energy_metric, # if default, it will find the column starting with "MaxCriteria"
            save_dir=save_path, # defaults to "output/figures"
            save_format="pdf", 
            save_individual=True  # Save both individual and comparison plots
        )
        print(f"Used energy metric: {name_of_energy_metric}")
        print(f"Visualization completed successfully!")
        print(f"Results saved to: {save_path}")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()