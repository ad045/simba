import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Define paths
INPUT_FILE = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/hcp_schaefer_100_dataset/05_second_big_overnight_run_10201/all_metrics_for_05_second_big_overnight_run_10201 copy 2.csv'
OUTPUT_DIR = '/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/hcp_schaefer_100_dataset/05_second_big_overnight_run_10201/pca_results/'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Columns to exclude from PCA but keep for analysis
    meta_cols = ['eta', 'gamma']

    # Columns to generally exclude (because e.g. they are strings or hparams)
    remove_cols = ['id', 'distance_relationship_type', 'preferential_relationship_type', 'generative_rule', 'num_iterations', 'network_index', 'h_params_spectral_radius','h_params_input_scaling','h_params_train_len','h_params_n_runs','h_params_n_lags','h_params_test_len','h_params_leak_rate','h_params_bias','h_params_n_transient','h_params_random_state']    
    
    # Separate features and metadata
    df_meta = df[meta_cols]
    df_features = df.drop(columns=meta_cols + remove_cols)

    # Handle infinite values and missing values
    print("Checking for infinite and missing values...")
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if df_features.isnull().values.any():
        print("Warning: Missing or infinite values found. Filling with mean.")
        df_features = df_features.fillna(df_features.mean())
        
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    # Perform PCA
    print("Performing PCA...")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame for PCA results
    pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    
    # Combine with metadata
    df_final = pd.concat([df_meta, df_pca], axis=1)
    
    # Save scores
    scores_path = os.path.join(OUTPUT_DIR, 'pca_scores.csv')
    df_final.to_csv(scores_path, index=False)
    print(f"PCA scores saved to {scores_path}")
    
    # Save loadings
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=pca_cols, 
        index=df_features.columns
    )
    loadings_path = os.path.join(OUTPUT_DIR, 'pca_loadings.csv')
    loadings.to_csv(loadings_path)
    print(f"PCA loadings saved to {loadings_path}")
    
    # --- Visualizations ---
    
    # 1. Scree Plot
    plt.figure(figsize=(10, 6))
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title('Scree Plot')
    plt.legend(loc='best')
    plt.grid(True)
    scree_path = os.path.join(OUTPUT_DIR, 'pca_scree_plot.png')
    plt.savefig(scree_path)
    plt.close()
    print(f"Scree plot saved to {scree_path}")
    
    # 2. PC1 vs PC2 Scatter Plot (colored by eta)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_final, x='PC1', y='PC2', hue='eta', palette='viridis', style='gamma')
    plt.title('PCA: PC1 vs PC2 (colored by eta, style by gamma)')
    plt.grid(True)
    scatter_path = os.path.join(OUTPUT_DIR, 'pca_scatter_plot.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")

    # 3. PC1 vs PC2 Scatter Plot (colored by gamma)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_final, x='PC1', y='PC2', hue='gamma', palette='magma', style='eta')
    plt.title('PCA: PC1 vs PC2 (colored by gamma, style by eta)')
    plt.grid(True)
    scatter_gamma_path = os.path.join(OUTPUT_DIR, 'pca_scatter_plot_gamma.png')
    plt.savefig(scatter_gamma_path)
    plt.close()
    print(f"Scatter plot (gamma) saved to {scatter_gamma_path}")

if __name__ == "__main__":
    main()
