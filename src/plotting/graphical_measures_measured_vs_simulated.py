
import numpy as np
import scipy.io
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from utils.saving_and_finding_files import time_stamp_for_saving


def plot_graph_measure_vs_simulated(graph_measures_empirical_connectomes, 
                                    best_gnm_per_subject, 
                                    variable_name, 
                                    color_metric=None, color_metric_name=None, 
                                    save_path=None):    
    """ 
    Plot a scatter plot of the empirical graph measure vs. the simulated one.
    """

    printable_name_dict = {
        "richclub_avg_length": "rich-club average length",
        "modularity": "modularity",
    }

    emp_values = graph_measures_empirical_connectomes[variable_name].values
    simulated_values = best_gnm_per_subject[variable_name].values

    if color_metric:
        plt.scatter(emp_values, simulated_values, c=color_metric, alpha=0.5, s=12)
        plt.colorbar(label=color_metric_name)
    else:
        plt.scatter(emp_values, simulated_values, alpha=0.5, s=12)

    plt.xlabel(f"Empirical {printable_name_dict[variable_name]}")
    plt.ylabel(f"Simulated {printable_name_dict[variable_name]}")
    # but here capitcalize the first letter
    plt.title(f"{printable_name_dict[variable_name].capitalize()}: empirical vs. simulated")

    # Add a note with pearsons correclation coefficient
    from scipy.stats import pearsonr
    corr, _ = pearsonr(emp_values, simulated_values)
    plt.annotate(f"Pearson's r = {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, ha='left', va='top',
                #  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
                    bbox=dict(boxstyle="round, pad=0.3", edgecolor="black", facecolor="white"),
                    color="black")

    # draw linear regression line
    from sklearn.linear_model import LinearRegression
    X = emp_values.reshape(-1, 1)
    y = simulated_values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.plot(emp_values, y_pred, color='red', linewidth=1, label='Linear fit')
    plt.legend()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()