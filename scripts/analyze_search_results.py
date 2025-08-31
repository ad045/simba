# scripts/analyze_search_results.py
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def analyze_results(results_path):
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    successful = df[df['status'] == 'success']
    
    # Extract parameters into columns
    param_df = pd.json_normalize(successful['params'])
    results_df = pd.concat([param_df, successful['memory_capacity']], axis=1)
    
    # Find best parameters
    best_idx = results_df['memory_capacity'].idxmax()
    best_params = results_df.loc[best_idx]
    
    print("Best parameters:")
    print(best_params)
    
    # Create visualization
    # Example: 2D heatmap for two parameters
    if 'gnm.eta' in results_df.columns and 'gnm.gamma' in results_df.columns:
        pivot = results_df.pivot_table(
            values='memory_capacity',
            index='gnm.gamma',
            columns='gnm.eta',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        plt.imshow(pivot, aspect='auto', origin='lower')
        plt.colorbar(label='Memory Capacity')
        plt.xlabel('Eta')
        plt.ylabel('Gamma')
        plt.title('Memory Capacity Landscape')
        plt.show()
    
    return results_df