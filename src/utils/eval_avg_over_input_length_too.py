import pandas as pd
import numpy as np


def analyze_hyperparameters(csv_file_path, top_n=5):
    """
    Analyze hyperparameter configurations to find the best performing ones
    based on mc_mean values, averaged across subjects.
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    top_n (int): Number of top configurations to return
    
    Returns:
    pandas.DataFrame: Top configurations with their average performance
    """
    
    # Load the data
    df = pd.read_csv(csv_file_path)
    
    # Remove "COMPLETED" - i.e. the last row if it exists
    if df.iloc[-1].get('subject') == "COMPLETED":
        df = df.iloc[:-1]
    
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    print(f"Unique subjects: {df['subject'].nunique()}")
    
    # Define the hyperparameter columns for grouping
    hyperparam_cols = [
        'regularization_method',
        'density_percent', 
        'spectral_radius', 
        # 'input_length', 
        'input_scaling', 
    ]
    
    # Group by hyperparameter configuration and calculate statistics
    grouped = df.groupby(hyperparam_cols).agg({
        'mc_mean': ['mean', 'std', 'count'],
        'mc_std': 'mean',
        'subject': 'count'  # This gives us the number of subjects per configuration
    }).round(6)
    
    # Flatten column names
    grouped.columns = ['avg_mc_mean', 'std_mc_mean', 'count_mc_mean', 'avg_mc_std', 'number_runs_in_group']
    
    # Reset index to make hyperparameter columns accessible
    grouped = grouped.reset_index()
    
    # Sort by average mc_mean in descending order
    grouped_sorted = grouped.sort_values('avg_mc_mean', ascending=False)
    
    # Get top configurations
    top_configs = grouped_sorted.head(top_n)
    
    print(f"\n=== TOP {top_n} HYPERPARAMETER CONFIGURATIONS ===")
    print("Ranked by average mc_mean across all subjects\n")
    
    for idx, row in top_configs.iterrows():
        rank = top_configs.index.get_loc(idx) + 1
        print(f"{rank}. Average mc_mean: {row['avg_mc_mean']:.6f}")
        print(f"   Parameters:")
        print(f"   - Density: {row['density_percent']}%")
        print(f"   - Spectral Radius: {row['spectral_radius']}")
        # print(f"   - Input Length: {row['input_length']}")
        print(f"   - Input Scaling: {row['input_scaling']}")
        print(f"   - Regularization: {row['regularization_method']}")
        print(f"   Performance Metrics:") 
        print(f"   - Std of mc_mean across subjects: {row['std_mc_mean']:.6f}")
        print(f"   - Average mc_std within runs: {row['avg_mc_std']:.6f}")
        print(f"   - Number of subjects: {row['number_runs_in_group']}")
        print()
    
    return top_configs

def detailed_analysis(csv_file_path):
    """
    Perform a more detailed analysis including parameter impact assessment
    """
    df = pd.read_csv(csv_file_path)
    
    print("=== DETAILED PARAMETER ANALYSIS ===\n")
    
    # Analyze impact of each parameter individually
    params_to_analyze = ['regularization_method', 'density_percent', 'spectral_radius', 'input_scaling']
    
    for param in params_to_analyze:
        print(f"Impact of {param}:")
        param_analysis = df.groupby(param)['mc_mean'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        print(param_analysis.round(6))
        print()
    
    # Correlation analysis for numerical parameters
    print("=== CORRELATION ANALYSIS ===")
    numerical_params = ['density_percent', 'spectral_radius', 'input_scaling', 'mc_mean', 'mc_std']
    correlation_matrix = df[numerical_params].corr()
    
    # Focus on correlations with mc_mean
    mc_mean_correlations = correlation_matrix['mc_mean'].sort_values(key=abs, ascending=False)
    print("Correlations with mc_mean:")
    for param, corr in mc_mean_correlations.items():
        if param != 'mc_mean':
            print(f"{param}: {corr:.4f}")
    print()


def create_config_summary(csv_file_path):
    """
    Create a summary of all unique configurations and their performance
    """
    df = pd.read_csv(csv_file_path)
    
    hyperparam_cols = [
        'regularization_method', 
        'density_percent', 
        'spectral_radius', 
        #  'input_length', 
        'input_scaling', 
    ]
    
    # Create a configuration summary
    config_summary = df.groupby(hyperparam_cols).agg({
        'mc_mean': ['mean', 'std', 'min', 'max'],
        'subject': 'count'
    }).round(6)

    config_summary.columns = ['mean_mc_mean', 'std_mc_mean', 'min_mc_mean', 'max_mc_mean', 'number_runs_in_group']
    config_summary = config_summary.reset_index().sort_values('mean_mc_mean', ascending=False)
    
    return config_summary




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def create_matplotlib_parallel_coordinates(df_top, top_n, save_folder=None):
    """
    Create an improved matplotlib version of the parallel coordinates plot
    with independent y-axes for each parameter.
    """
    df_plot = df_top.copy()

    # 1. Prepare data for plotting
    # Convert regularization method to numeric for plotting positions
    if 'regularization_method' in df_plot.columns:
        reg_methods = sorted(df_plot['regularization_method'].unique())
        reg_mapping = {method: i for i, method in enumerate(reg_methods)}
        df_plot['reg_method_numeric'] = df_plot['regularization_method'].map(reg_mapping)

    plot_columns = ['reg_method_numeric', 'density_percent', 'spectral_radius', 'input_scaling', 'mean_mc_mean']
    column_labels = ['Regularization', 'Density %', 'Spectral Radius', 'Input Scaling', 'Mean mc_mean']

    df_subset = df_plot[plot_columns]

    # 2. Normalize data to the range [0, 1] for plotting on a common canvas
    df_normalized = pd.DataFrame()
    bounds = {}
    for col in df_subset.columns:
        min_val = df_subset[col].min()
        max_val = df_subset[col].max()
        bounds[col] = (min_val, max_val)
        # Avoid division by zero if a column has all the same values
        if max_val == min_val:
            df_normalized[col] = 0.5
        else:
            df_normalized[col] = (df_subset[col] - min_val) / (max_val - min_val)
        

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(16, 9))

    # Setup continuous color mapping based on performance (mean_mc_mean)
    norm = Normalize(vmin=df_plot['mean_mc_mean'].min(), vmax=df_plot['mean_mc_mean'].max())
    cmap = plt.cm.viridis_r # Using reversed viridis so high values are bright yellow

    # Plot each configuration as a colored line on the normalized scale
    for i in range(len(df_normalized)):
        y_values = df_normalized.iloc[i].values
        color_val = df_plot['mean_mc_mean'].iloc[i]
        ax.plot(range(len(plot_columns)), y_values, color=cmap(norm(color_val)), alpha=0.6, linewidth=1.5)

    # 4. Customize axes to show original values (the key improvement)
    ax.set_xticks(range(len(plot_columns)))
    ax.set_xticklabels(column_labels, rotation=30, ha='right', fontsize=12)
    ax.set_ylim(-0.05, 1.05) # Add a little padding
    ax.set_yticks([]) # Hide the default 0-1 y-axis

    # Draw custom y-axis ticks and labels for each parameter
    for i, col in enumerate(plot_columns):
        min_val, max_val = bounds[col]
        
        # Draw a vertical line for each axis
        ax.axvline(i, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # Handle the categorical 'regularization' axis
        if col == 'reg_method_numeric':
            tick_positions_norm = np.linspace(0, 1, num=len(reg_methods))
            for tick_pos, label in zip(tick_positions_norm, reg_methods):
                ax.text(i - 0.03, tick_pos, f' {label}', ha='right', va='center', fontsize=10)
                
        # Handle numerical axes
        else:
            # Get the original min and max values for this column
            min_val, max_val = bounds[col]
            
            # Generate 5 evenly spaced tick values in the ABSOLUTE range
            tick_values = np.linspace(min_val, max_val, num=5)
            
            # For each absolute tick value, calculate its normalized position and draw it
            for val in tick_values:
                # Calculate the normalized position (0.0 to 1.0) for placement
                norm_pos = (val - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0.5
                
                # Format the ABSOLUTE value as the label string
                label_str = f'{val:.2f}'
                
                # Place the text on the plot
                ax.text(i + 0.03, norm_pos, label_str, ha='left', va='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))


        # Handle numerical axes
        # else:
        #     # Generate 5 nice tick labels for each numerical axis
        #     tick_values = np.linspace(min_val, max_val, num=5)
        #     for val in tick_values:
        #         # Find the normalized position for the tick label
        #         norm_pos = (val - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0.5
        #         # Format label nicely
        #         label_str = f'{val:.2f}'
        #         ax.text(i + 0.03, norm_pos, label_str, ha='left', va='center', fontsize=9,
        #                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))

    # 5. Finalize plot aesthetics
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.5)
    ax.set_title(f'Parallel Coordinates Plot: Top {top_n} Hyperparameter Configurations', fontsize=16, pad=20)
    ax.set_xlabel('Parameters', fontsize=12, labelpad=20)
    ax.set_ylabel('Normalized Parameter Value', fontsize=12)

    # Add a color bar to show the performance scale
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Mean MC Mean (Performance)', fontsize=12, rotation=270, labelpad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for title

    # Save the plot
    if save_folder:
        save_path = save_folder / 'hyperparameter_parallel_coordinates_matplotlib_improved.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Improved matplotlib plot saved as '{save_path.name}'")

    plt.show()
    
    
    
    
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def create_parallel_coordinates_plot(summary_csv_path='hyperparameter_summary.csv', save_folder=None, top_n=20):
    """
    Create a parallel coordinates plot from the hyperparameter summary
    
    Parameters:
    summary_csv_path (str): Path to the hyperparameter_summary.csv file
    top_n (int): Number of top configurations to include in the plot
    """
    
    # Load the summary data
    df = pd.read_csv(summary_csv_path)
    
    # Take top N configurations for better visualization
    df_top = df.head(top_n).copy()
    
    # Convert categorical variable to numeric for plotting
    if 'regularization_method' in df_top.columns:
        reg_methods = sorted(df_top['regularization_method'].unique())
        reg_mapping = {method: i for i, method in enumerate(reg_methods)}
        df_top['reg_method_numeric'] = df_top['regularization_method'].map(reg_mapping)
    
    # Normalize numerical columns for better visualization
    # for col in ['density_percent', 'spectral_radius', 'input_scaling', 'mean_mc_mean']:
    #     # normalize 
    #     df_top[col] = (df_top[col] - df_top[col].min()) / (df_top[col].max() - df_top[col].min())
    # df_top['input_length'] = df_top['input_length'] / 1000  
        
        
    # Create parallel coordinates plot using Plotly with individual scaling
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=df_top['mean_mc_mean'],
                     colorscale='Viridis',
                     showscale=True,
                     colorbar=dict(title="Mean mc_mean"),
                     cmin=df_top['mean_mc_mean'].min(),
                     cmax=df_top['mean_mc_mean'].max()),
            dimensions=list([
                dict(tickvals=list(range(len(reg_methods))),
                     ticktext=reg_methods,
                     label="Regularization", values=df_top['reg_method_numeric']),
                dict(label="Density %", values=df_top['density_percent']),
                dict(label="Spectral Radius", values=df_top['spectral_radius']),
                # dict(label="Input Length", values=df_top['input_length']),
                dict(label="Input Scaling", values=df_top['input_scaling']),
                dict(label="Mean mc_mean", values=df_top['mean_mc_mean'])
            ])
        )
    )
    
    fig.update_layout(
        title=f'Parallel Coordinates Plot: Top {top_n} Hyperparameter Configurations',
        font=dict(size=12),
        height=600
    )
    
    # Save the plot
    fig.write_html(save_folder / 'hyperparameter_parallel_coordinates.html')
    fig.show()
    
    print(f"Parallel coordinates plot saved as 'hyperparameter_parallel_coordinates.html'")
    
    # Also create a matplotlib version for those who prefer it
    create_matplotlib_parallel_coordinates(df_top, top_n, save_folder=save_folder)
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file path - example shows expected structure
    from pathlib import Path
    
    # Look for CSV files in typical output structure
    project_root = Path.cwd()
    while not any((project_root / marker).exists() for marker in ("pyproject.toml", ".git")):
        project_root = project_root.parent
    
    # Example path - replace with actual file
    csv_file = project_root / "output/esn/results.csv"
    
    if not csv_file.exists():
        print(f"CSV file not found at {csv_file}")
        print("Please provide the correct path to your ESN results CSV file")
        exit(1)
    
    base_path = csv_file.parent

    # Run the main analysis
    top_configs = analyze_hyperparameters(csv_file, top_n=5)
    
    # Run detailed analysis
    detailed_analysis(csv_file)
    
    # Create and save a comprehensive summary
    
    summary_file_name = 'hyperparameter_summary_avg_over_input_length_too.csv'
    summary = create_config_summary(csv_file)
    summary.to_csv(base_path / summary_file_name, index=False)
    print(f"Comprehensive summary saved to '{summary_file_name}'")

    # Create parallel coordinates plot
    create_parallel_coordinates_plot(base_path / summary_file_name, save_folder=base_path, top_n=400) # 20)

    # Or run the complete analysis with plots:
    # run_complete_analysis_with_plots('your_data.csv')
    
    print("Added visualization complete.")