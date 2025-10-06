""" 
Those box plots + those latex comparison files between the metrics. 

"""

import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from pathlib import Path 

def format_p_value(p):
    """Formats p-values for the LaTeX table with significance stars."""
    if p < 0.001:
        return "\\textbf{$<$ 0.001}***"
    elif p < 0.01:
        return f"\\textbf{{{p:.3f}}}**"
    elif p < 0.05:
        return f"\\textbf{{{p:.3f}}}*"
    else:
        return f"{p:.3f} (n.s.)"


def _filter_columns(df: pd.DataFrame, 
                    cols_to_remove: list = [],
                    cols_to_remove_starting_with: str = "hparams_", 
                    ) -> pd.DataFrame:
        """
        Selects all columns except for those starting with 'hparams_'
        or named 'not_important', and prints the removed column names.
        """
        # Find columns that meet the drop criteria
        cols_to_drop = [
            col for col in df.columns
            if col.startswith(cols_to_remove_starting_with) or col in cols_to_remove
        ]

        # Print the columns that will be dropped
        print(f"Dropped columns: {cols_to_drop}")

        # Return the DataFrame with the identified columns removed
        return df.drop(columns=cols_to_drop)
    
    
def compare_and_visualize(file_gen_connectomes, 
                          file_empirical_connectomes, 
                          output_dir="."):
    """
    Compares metrics from two CSV files, generates a LaTeX summary table,
    and creates a boxplot figure.

    Args:
        file_gen (str): Path to the first CSV file (e.g., 'matched_rows.csv').
        file_c (str): Path to the second CSV file (e.g., 'c.csv').
        output_dir (str): Directory to save the output files.
    """
    try:
        df_gen_connectomes = pd.read_csv(file_gen_connectomes)
        df_empirical_connectomes = pd.read_csv(file_empirical_connectomes)

        cols_to_remove = ["eta", "gamma", "num_iterations", "network_index", "avg_degree", "subj_index", "energy", "subject_id"]
        cols_to_remove_starting_with="h_params_"
        
        df_gen_connectomes = _filter_columns(df_gen_connectomes, 
                                             cols_to_remove=cols_to_remove, 
                                             cols_to_remove_starting_with=cols_to_remove_starting_with)
        
        df_empirical_connectomes = _filter_columns(df_empirical_connectomes, 
                                             cols_to_remove=cols_to_remove, 
                                             cols_to_remove_starting_with=cols_to_remove_starting_with)

    except FileNotFoundError as e:
        print(f"Error: Could not find a file. {e}")
        return

    # Identify numeric columns to compare, excluding 'mc_' columns
    metrics = [
        col for col in df_gen_connectomes.columns
        if pd.api.types.is_numeric_dtype(df_gen_connectomes[col]) and not col.startswith('mc_')
    ]

    # --- Statistical Comparison ---
    results = []
    for metric in metrics:
        if metric not in df_empirical_connectomes.columns:
            print(f"Warning: Metric '{metric}' not found in {file_empirical_connectomes}. Skipping.")
            continue

        data_gen = df_gen_connectomes[metric].dropna()
        data_emp = df_empirical_connectomes[metric].dropna()

        if len(data_gen) < 2 or len(data_emp) < 2:
            print(f"Warning: Not enough data for metric '{metric}'. Skipping.")
            continue
        
        # Perform Welch's t-test (doesn't assume equal variance)
        _, p_value = ttest_ind(data_gen, data_emp, equal_var=False, nan_policy='omit')
        
        results.append({
            "metric": metric,
            "mean_gen": data_gen.mean(),
            "std_gen": data_gen.std(),
            "mean_emp": data_emp.mean(),
            "std_emp": data_emp.std(),
            "p_value": p_value
        })

    # --- Generate LaTeX Table ---
    latex_parts = [
        # "\\documentclass{article}",
        # "\\usepackage{booktabs}",
        # "\\usepackage{graphicx}",
        # "\\usepackage{caption}",
        # "\\begin{document}",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Comparison of Metrics}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Generated Connectomes} & \\textbf{Empirical Connectomes} & \\textbf{P-Values} \\\\",
        "\\midrule"
    ]

    for res in results:
        metric_escaped = res['metric'].replace('_', r'\_')
        new_vals = f"{res['mean_gen']:.2f} $\\pm$ {res['std_gen']:.2f}"
        c_vals = f"{res['mean_emp']:.2f} $\\pm$ {res['std_emp']:.2f}"
        p_str = format_p_value(res['p_value'])
        latex_parts.append(f"{metric_escaped} & {new_vals} & {c_vals} & {p_str} \\\\")

    latex_parts.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption*{Significance levels: n.s. (not significant) p $>$ 0.05, * p $<$ 0.05, ** p $<$ 0.01, *** p $<$ 0.001}",
        "\\end{table}",
        # "\\end{document}"
    ])
    
    latex_string = "\n".join(latex_parts)
    latex_output_dir = os.path.join(output_dir, "latex")
    os.makedirs(latex_output_dir, exist_ok=True)
    latex_filepath = os.path.join(latex_output_dir, "comparison_table.tex")
    with open(latex_filepath, "w") as f:
        f.write(latex_string)
    print(f"LaTeX table saved to '{latex_filepath}'")

    # --- Generate Boxplots Figure ---
    if not metrics:
        print("No common numeric metrics to plot.")
        return

    ncols = 4
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        plot_data = [df_gen_connectomes[metric].dropna(), df_empirical_connectomes[metric].dropna()]
        ax.boxplot(plot_data, tick_labels=['Generated Connectomes', 'Empirical Connectomes'])
        ax.set_title(metric, fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Hide any unused subplots
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig_filepath = os.path.join(output_dir, "figures", "comparison_boxplots.png")
    plt.suptitle("Metric Comparison between Generated and Empirical Connectomes") # , fontsize=16)
    plt.savefig(fig_filepath, dpi=300)
    plt.close()
    print(f"Boxplot figure saved to '{fig_filepath}'")


if __name__ == "__main__":

    # Define file paths
    output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes")
    
    file_gen_connectomes = output_folder / 'comparison_empirical_connectomes_with_estiamted_eta_and_gamma_and_graph_analysis.csv'
    file_empirical_connectomes = output_folder / 'comparison_generated_connectomes_with_eta_and_gamma_and_graph_analysis.csv'
    
    
    if not os.path.exists(file_gen_connectomes):
        print(f"Error: The file '{file_gen_connectomes}' does not exist.")
        print("Please run 'find_matches.py' first to generate it.")
    elif not os.path.exists(file_empirical_connectomes):
        print(f"Error: The file '{file_empirical_connectomes}' does not exist.")
    else:
        # Run the comparison
        compare_and_visualize(file_gen_connectomes, 
                              file_empirical_connectomes, 
                              output_dir=output_folder)
