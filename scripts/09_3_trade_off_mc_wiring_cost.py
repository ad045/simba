import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def identify_pareto_front(df, cost_col, benefit_col):
    """
    Identifies the Pareto front from a DataFrame.

    In this context, we want to minimize wiring cost and maximize memory capacity.
    A point is on the Pareto front if no other point has a lower cost AND a higher benefit.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cost_col (str): The name of the column representing cost (to be minimized).
        benefit_col (str): The name of the column representing benefit (to be maximized).

    Returns:
        pd.DataFrame: A DataFrame containing only the points on the Pareto front.
    """
    # Sort the dataframe by cost in ascending order, and then by benefit in descending order
    # This helps in efficiently finding the front
    df_sorted = df.sort_values(by=[cost_col, benefit_col], ascending=[True, False])

    pareto_front = []
    max_benefit_so_far = -np.inf

    # Iterate through the sorted points
    for index, row in df_sorted.iterrows():
        # A point is on the Pareto front if its benefit is greater than the max benefit
        # of all points with a lower or equal cost.
        if row[benefit_col] > max_benefit_so_far:
            pareto_front.append(row)
            max_benefit_so_far = row[benefit_col]

    return pd.DataFrame(pareto_front)

def plot_tradeoff(df, pareto_df, required_col_1, required_col_2, save_path):
    """
    Generates and saves a scatter plot of the memory vs. cost trade-off.

    Args:
        df (pd.DataFrame): The full DataFrame of all networks.
        pareto_df (pd.DataFrame): The DataFrame of Pareto-optimal networks.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot all the generated networks
    if "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)" in df.columns:
        scatter_ax = ax.scatter(
            df[required_col_1],
            df[required_col_2],
            alpha=0.6,
            c=df["MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS, BetweennessKS)"], 
            cmap='hot',
            edgecolor='k',
            s=20,
            label='Generated Networks (color == Energy. The lower, the darker.)'
        )
        # ax.set_colorbar(label='Energy')
        # Add the colorbar using the figure object
        cbar = fig.colorbar(scatter_ax, ax=ax) # Add again? 

        # Optional: Add a label to the colorbar
        cbar.set_label('Energy') 
        
    else: 
        ax.scatter(
            df[required_col_1],
            df[required_col_2],
            alpha=0.6,
            c='lightblue',
            edgecolor='k',
            s=20,
            label='Generated Networks'
        )
        # plt.colorbar(ax=ax, label='Energy')

    # Highlight the optimal networks (Pareto front)
    # ax.plot(
    #     pareto_df['wiring_cost'],
    #     pareto_df['mc_mean'],
    #     color='red',
    #     marker='o',
    #     linestyle='--',
    #     linewidth=2,
    #     markersize=8,
    #     label='Optimal Trade-off (Pareto Front)'
    # )

    # --- Plot Formatting ---
    word_1 = "Wiring Cost" if required_col_1 == "wiring_cost" else ""
    if required_col_2 == "mc_mean": 
        word_2 = "Memory Capacity (mean)" 
    if required_col_2.startswith("mc_"):
        number = required_col_2.split("_")[1]
        word_2 = f"Memory Capacity (lag {number})"
        
    ax.set_title('Trade-off between ' + word_1 + ' and ' + word_2, fontsize=16, pad=20)
    ax.set_xlabel(word_1, fontsize=12)
    ax.set_ylabel(word_2, fontsize=12)
    # ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add text to explain the optimal front
    # ax.text(
    #     0.05, 0.95,
    #     'Optimal networks offer the highest memory\nfor a given wiring cost.',
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     verticalalignment='top',
    #     bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8)
    # )

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(save_path) # , dpi=300)
    print("Plot saved as " + save_path + ".") 
    # plt.show()

def main():
    """
    Main function to run the analysis.
    """
    try:
        # Load the dataset
        # Make sure your data file is named 'network_data.csv' and is in the same directory
        file_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/26_testing_4_KS_folders_why_so_fast/summary_all_metrics_for_exp_26_testing_4_KS_folders_why_so_fast.csv"
        # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes/26_testing_4_KS_folders_why_so_fast.csv"
        data = pd.read_csv(file_path)
        
        # filter entries to only include entries that have energies lower than the average energy
        # appendix_save_path = "filtered_by_energy_below_its_median" # nly_below_average_energy_conns"
        # energy_name = "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS)"
        # data = data[data[energy_name] < data[energy_name].median()]
        
        # filter entries to only include entries that have memory capacities lower than the average mc_mean
        appendix_save_path = "filtered_by_mc_mean_above_its_median"
        energy_name = "mc_mean" # "MaxCriteria(DegreeKS, ClusteringKS, EdgeLengthKS)"
        data = data[data[energy_name] > data[energy_name].median()]
        
        # dont filter entries
        # appendix_save_path = "not_filtered" 
        
        
        def short_pipeline_part(required_cols, appendix_save_path): 
            # Ensure the necessary columns exist
            if not all(col in data.columns for col in required_cols):
                print(f"Error: CSV must contain the columns: {', '.join(required_cols)}")
                return
            
            folder_path = ("/").join(file_path.split("/")[:-1]) + "/figures_trade_off_wiring_cost_vs_mc"
            os.makedirs(folder_path, exist_ok=True)
            save_path = folder_path + "/" + required_cols[0] + "_vs_" + required_cols[1] + "_" + appendix_save_path +".pdf"

            # Identify the Pareto-optimal networks
            pareto_optimal_networks = identify_pareto_front(data, required_cols[0], required_cols[1]) # 'wiring_cost', 'mc_mean')

            print("Analysis Complete.")
            print(f"Found {len(pareto_optimal_networks)} optimal networks out of {len(data)} total.")

            # Plot the results
            plot_tradeoff(data, pareto_optimal_networks, required_cols[0], required_cols[1], save_path)
        
        
        short_pipeline_part(['wiring_cost', 'mc_mean'], appendix_save_path)
        short_pipeline_part(['wiring_cost', 'mc_5'], appendix_save_path)
        short_pipeline_part(['wiring_cost', 'mc_20'], appendix_save_path)
        short_pipeline_part(['wiring_cost', 'mc_10'], appendix_save_path)
        
        
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure your data file is named correctly and is in the same folder as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
