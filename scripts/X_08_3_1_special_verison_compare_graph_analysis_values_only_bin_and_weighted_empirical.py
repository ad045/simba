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

def get_significance_symbol(p):
    """Returns significance symbol for a given p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


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
    and creates a violin plot figure with scatter points.

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
    ])
    
    latex_string = "\n".join(latex_parts)
    latex_output_dir = os.path.join(output_dir, "latex")
    os.makedirs(latex_output_dir, exist_ok=True)
    latex_filepath = os.path.join(latex_output_dir, "comparison_table_weighted_vs_binary.tex")
    with open(latex_filepath, "w") as f:
        f.write(latex_string)
    print(f"LaTeX table saved to '{latex_filepath}'")


    # --- Generate Violin Plots with Scatter Points ---
    if not metrics:
        print("No common numeric metrics to plot.")
        return

    # Create a dictionary for quick p-value lookup
    results_dict = {res['metric']: res for res in results}

    ncols = 5
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Prepare data
        data_gen = df_gen_connectomes[metric].dropna()
        data_emp = df_empirical_connectomes[metric].dropna()
        plot_data = [data_gen, data_emp]
        
        # Create violin plot
        parts = ax.violinplot(plot_data, positions=[1, 2], showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        for pc in parts['bodies']:
            pc.set_facecolor('#8dd3c7')
            pc.set_alpha(0.6)
        
        # Overlay scatter points with jitter
        np.random.seed(42)  # For reproducibility
        jitter_strength = 0.04
        
        # Scatter for weighted (position 1)
        x_jitter_gen = np.random.normal(1, jitter_strength, size=len(data_gen))
        ax.scatter(x_jitter_gen, data_gen, alpha=0.5, s=20, color='#1f78b4', edgecolors='black', linewidth=0.5)
        
        # Scatter for binarized (position 2)
        x_jitter_emp = np.random.normal(2, jitter_strength, size=len(data_emp))
        ax.scatter(x_jitter_emp, data_emp, alpha=0.5, s=20, color='#e31a1c', edgecolors='black', linewidth=0.5)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Weighted', 'Binarized'])
        ax.set_title(metric, fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        # Add significance annotation
        p_value = results_dict[metric]['p_value']
        symbol = get_significance_symbol(p_value)

        if symbol != 'n.s.':
            # Determine position for the annotation
            y_max = max(data_gen.max(), data_emp.max())
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            bar_y = y_max + y_range * 0.07
            text_y = bar_y + y_range * 0.01

            # Draw the bar and the symbol
            ax.plot([1, 2], [bar_y, bar_y], color='black', lw=0.8)
            ax.text(1.5, text_y, symbol, ha='center', va='bottom', color='black', fontsize=10)
            
            ax.set_ylim(top=text_y + y_range * 0.1)

    # Hide any unused subplots
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig_filepath = os.path.join(output_dir, "figures", "comparison_violinplots_weighted_vs_binarized.pdf")
    os.makedirs(os.path.dirname(fig_filepath), exist_ok=True)
    plt.suptitle("Metric Comparison between Weighted and Binarized Empirical Connectomes")
    # plt.tight_layout()
    plt.savefig(fig_filepath)
    plt.close()
    print(f"Violin plot figure saved to '{fig_filepath}'")


if __name__ == "__main__":

    # Define file paths
    output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/emprirical_analysis")
    
    file_weighted_conns = output_folder / 'empirical_analysis_weighted.csv'
    file_bin_conns = output_folder / 'empirical_analysis_binarized.csv'
    
    if not os.path.exists(file_weighted_conns):
        print(f"Error: The file '{file_weighted_conns}' does not exist.")
    elif not os.path.exists(file_bin_conns):
        print(f"Error: The file '{file_bin_conns}' does not exist.")
    else:
        # Run the comparison
        compare_and_visualize(file_weighted_conns, 
                              file_bin_conns, 
                              output_dir=output_folder)

