import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform

import networkx as nx 


def find_closest_connectome(target_eta: float, 
                            target_gamma: float, 
                            data_dir: Path | str,  # Directory containing .npy files
                            number_connectomes: int = 1): # number of connectomes to load
    """
    Finds the .npy file(s) with eta/gamma values closest to the target,
    then loads the connectome(s) and returns them.

    Returns:
    --------
    If number_connectomes == 1:
        found_eta : float
        found_gamma : float
        connectome : numpy.ndarray of shape (n_nodes, n_nodes)
    
    If number_connectomes > 1:
        found_eta : numpy.ndarray of shape (number_connectomes,)
        found_gamma : numpy.ndarray of shape (number_connectomes,)
        connectomes : numpy.ndarray of shape (number_connectomes, n_nodes, n_nodes)
    """
    
    # TODO: Replace this with other function! (I think in utils...)
    # Regex to extract eta and gamma values from filenames
    pattern = re.compile(r"eta(-?\d+\.\d+)_gamma(-?\d+\.\d+)")

    # Search for files and calculate distances
    print(f"Searching in: {data_dir}")
    file_distances = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            match = pattern.search(filename)
            if match:
                eta_val = float(match.group(1))
                gamma_val = float(match.group(2))

                # Calculate squared Euclidean distance
                dist = (eta_val - target_eta)**2 + (gamma_val - target_gamma)**2
                
                file_distances.append((dist, filename, eta_val, gamma_val))

    if not file_distances:
        print("Error: No matching .npy files found in the specified directory.")
        return

    # Sort by distance and take the closest number_connectomes files
    file_distances.sort(key=lambda x: x[0])
    closest_files = file_distances[:number_connectomes]
    
    print(f"Found {len(closest_files)} closest file(s)")
    print(f"Target (η, γ): ({target_eta}, {target_gamma})")
    
    if number_connectomes == 1:
        # Original behavior: return single connectome
        dist, best_file, found_eta, found_gamma = closest_files[0]
        print(f"Closest file found: {best_file}")
        print(f"Found  (η, γ): ({found_eta:.3f}, {found_gamma:.3f})")
        
        file_path = os.path.join(data_dir, best_file)
        data = np.load(file_path)
        connectome = data[0, :, :]
        return found_eta, found_gamma, connectome
    else:
        # New behavior: load multiple connectomes from multiple files
        connectomes_list = []
        found_eta_list = []
        found_gamma_list = []
        
        for i, (dist, filename, eta_val, gamma_val) in enumerate(closest_files):
            print(f"  {i+1}. {filename} - (η, γ): ({eta_val:.3f}, {gamma_val:.3f})")
            
            file_path = os.path.join(data_dir, filename)
            data = np.load(file_path)
            connectomes_list.append(data[0, :, :])
            found_eta_list.append(eta_val)
            found_gamma_list.append(gamma_val)
        
        # Stack into arrays
        connectomes = np.stack(connectomes_list, axis=0)
        found_eta_array = np.array(found_eta_list)
        found_gamma_array = np.array(found_gamma_list)
        
        return found_eta_array, found_gamma_array, connectomes
    
    

def _get_plot_save_path(data_dir, save_subdir="connectome_plots"): 
    # Create the save directory inside the parent folder of the data_dir
    parent_dir = os.path.dirname(data_dir)
    save_path = os.path.join(parent_dir, save_subdir)
    os.makedirs(save_path, exist_ok=True)
    
    return Path(save_path)


def reorder_connectome_blockwise(connectome, method='spectral', block_sizes=[50, 50]):
    """
    Reorders nodes within separate blocks to reveal structure in the connectome.
    
    Parameters:
    -----------
    connectome : np.ndarray
        The connectivity matrix (N x N)
    method : str
        Reordering method. Options:
        - 'spectral': Uses spectral clustering (Fiedler vector)
        - 'hierarchical': Uses hierarchical clustering
        - 'modularity': Uses community detection (Louvain)
        - 'degree': Orders by node degree
    block_sizes : list
        List of block sizes. Default [50, 50] for two 50-node blocks.
        
    Returns:
    --------
    reordered_connectome : np.ndarray
        Reordered connectivity matrix
    order : np.ndarray
        The permutation indices used for reordering
    """
    n = connectome.shape[0]
    
    # Validate block sizes
    if sum(block_sizes) != n:
        raise ValueError(f"Sum of block_sizes {sum(block_sizes)} must equal matrix size {n}")
    
    # Initialize the full ordering array
    order = np.zeros(n, dtype=int)
    
    # Process each block separately
    block_start = 0
    for block_size in block_sizes:
        block_end = block_start + block_size
        
        # Extract the submatrix for this block (only within-block connections)
        block_connectome = connectome[block_start:block_end, block_start:block_end]
        
        print(f"Processing block [{block_start}:{block_end}] with method: {method}")
        
        # Apply reordering to this block
        if method == 'spectral':
            # Use the Fiedler vector (2nd smallest eigenvector of Laplacian)
            G = nx.from_numpy_array(block_connectome)
            if nx.is_connected(G):
                # Use Laplacian eigenvector
                L = nx.laplacian_matrix(G).toarray()
                eigvals, eigvecs = np.linalg.eigh(L)
                fiedler = eigvecs[:, 1]  # Second smallest eigenvector
                block_order = np.argsort(fiedler)
            else:
                # Fallback to degree ordering if graph is disconnected
                print(f"  Block [{block_start}:{block_end}] is disconnected, using degree ordering")
                degrees = block_connectome.sum(axis=0) + block_connectome.sum(axis=1)
                block_order = np.argsort(degrees)[::-1]
        
        elif method == 'hierarchical':
            # Hierarchical clustering based on connectivity similarity
            conn_normed = block_connectome + block_connectome.T  # Symmetrize
            
            # Compute correlation distance
            corr_matrix = np.corrcoef(conn_normed)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            distance_matrix = 1 - np.abs(corr_matrix)
            
            # Convert to condensed distance matrix
            condensed_dist = squareform(distance_matrix, checks=False)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_dist, method='average')
            block_order = leaves_list(linkage_matrix)
        
        elif method == 'modularity':
            # Use Louvain community detection
            G = nx.from_numpy_array(block_connectome)
            try:
                communities = nx.community.louvain_communities(G)
                # Create ordering: group nodes by community
                block_order = []
                for community in communities:
                    community_list = sorted(list(community))
                    block_order.extend(community_list)
                block_order = np.array(block_order)
            except:
                print(f"  Modularity ordering failed for block [{block_start}:{block_end}], using degree ordering")
                degrees = block_connectome.sum(axis=0) + block_connectome.sum(axis=1)
                block_order = np.argsort(degrees)[::-1]
        
        elif method == 'degree':
            # Simple degree-based ordering
            degrees = block_connectome.sum(axis=0) + block_connectome.sum(axis=1)
            block_order = np.argsort(degrees)[::-1]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Map block-local indices to global indices
        order[block_start:block_end] = block_order + block_start
        
        block_start = block_end
    
    # Reorder the full matrix using the block-wise ordering
    reordered_connectome = connectome[order, :][:, order]
    
    return reordered_connectome, order


# Update the plot_connectome function to use block-wise reordering
def plot_connectome_blockwise(connectome, 
                              found_eta, found_gamma, 
                              data_dir, 
                              save_subdir="connectome_plots",
                              cluster_method=None, 
                              block_sizes=[50, 50],
                              show=True):
    """
    Plots and saves the connectome with block-wise reordering.
    """
    
    # Reorder if clustering method is specified
    if cluster_method is not None:
        print(f"Reordering connectome using block-wise method: {cluster_method}")
        connectome_to_plot, order = reorder_connectome_blockwise(
            connectome, method=cluster_method, block_sizes=block_sizes
        )
        title_suffix = f" ({cluster_method} blockwise ordering)"
        filename_suffix = f"_{cluster_method}_blockwise"
    else:
        connectome_to_plot = connectome
        order = np.arange(connectome.shape[0])
        title_suffix = ""
        filename_suffix = ""
    
    # Plot the connectome
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(connectome_to_plot, cmap='Blues', origin='lower')
    
    # Add visual separators between blocks
    if cluster_method is not None:
        block_boundary = block_sizes[0]
        ax.axhline(y=block_boundary - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=block_boundary - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_title(f"Connectome (η={found_eta:.3f}, γ={found_gamma:.3f}){title_suffix}", 
                 fontsize=14)
    ax.set_xlabel("Neuron Index", fontsize=12)
    ax.set_ylabel("Neuron Index", fontsize=12)
    fig.colorbar(im, ax=ax, label="Connection Strength")
    
    plt.tight_layout()

    # Save the plot
    save_path = Path(_get_plot_save_path(data_dir, save_subdir=save_subdir))
    
    output_filename = f"connectome_eta{found_eta:.3f}_gamma{found_gamma:.3f}{filename_suffix}.png"
    full_save_path = os.path.join(save_path, "adjacency_matrix_" + output_filename)
    
    plt.savefig(full_save_path, dpi=300)
    print(f"Plot saved to: {full_save_path}")
    
    if show: 
        plt.show()
    
    plot_kamada_kawai(connectome, save_path, output_filename, show=show)
    
    return Path(save_path)



def plot_kamada_kawai(connectome, save_path, output_file_name, show=True): 
    G = nx.from_numpy_array(connectome)
    
    # Calculate node degrees
    degrees = [val for (node, val) in G.degree()]
    
    # Draw the graph with updated parameters
    plt.figure(figsize=(10, 10))
    nx.draw(G, 
            pos=nx.kamada_kawai_layout(G), 
            node_color=degrees,
            node_size=50,
            cmap=plt.cm.autumn,
            with_labels=False,
            edge_color='gray',
            alpha=0.6)
    
    plt.savefig(save_path / f"kamada_kawai_{output_file_name}", dpi=300)
    
    if show: 
        plt.show()


def compare_orderings(connectome, found_eta, found_gamma, data_dir, show=True):
    """
    Create a comparison plot showing different ordering methods side by side.
    """
    methods = ['spectral', 'hierarchical', 'modularity', 'degree']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Find global min/max for consistent colorbar
    vmin, vmax = connectome.min(), connectome.max()
    
    for idx, method in enumerate(methods):
        reordered, _ = reorder_connectome_blockwise(connectome, method=method)
        
        im = axes[idx].imshow(reordered, cmap='Blues', origin='lower', 
                             vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"{method.capitalize()} Ordering", fontsize=12)
        
        # Only show labels on left and bottom edges
        if idx in [2, 3]:  # Bottom row
            axes[idx].set_xlabel("Neuron Index")
        if idx in [0, 2]:  # Left column
            axes[idx].set_ylabel("Neuron Index")
    
    # Add single colorbar on the right side
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Connection Strength")
    
    fig.suptitle(f"Connectome Orderings (η={found_eta:.3f}, γ={found_gamma:.3f})", 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    # Save comparison plot
    save_path = Path(_get_plot_save_path(data_dir))
    output_filename = f"ordering_comparison_eta{found_eta:.3f}_gamma{found_gamma:.3f}.png"
    plt.savefig(save_path / output_filename, dpi=300)
    print(f"Comparison plot saved to: {save_path / output_filename}")
    
    if show: 
        plt.show()
    
    
    
def main(): 
    # Data paths 
    data_dir = "output/gnm/26_testing_4_KS_folders_why_so_fast/all_generated_networks"
    save_subdir = "output/gnm/26_testing_4_KS_folders_why_so_fast/all_generated_networks/"

    interesting_eta_and_gamma_combinations = [
        [-3, 0.1],
        [-1, 0.5],
        [-6, 0.5],
        [-6, -0.1],
        [-3.75, 0.5],
        [-1, -0.15],
    ]

    # Choose clustering method: 'spectral', 'hierarchical', 'modularity', 'degree', or None
    clustering_method = 'spectral'  # Change this to experiment with different orderings

    for parameter_combination in interesting_eta_and_gamma_combinations:
        target_eta = parameter_combination[0]
        target_gamma = parameter_combination[1]
        
        found_eta, found_gamma, connectome = find_closest_connectome(target_eta, target_gamma, data_dir)
        
        # Plot original (unordered) connectome
        save_path = plot_connectome(connectome, found_eta, found_gamma, data_dir, 
                                save_subdir="connectome_plots",
                                cluster_method=None, 
                                show=False)
        
        # Plot clustered connectome
        plot_connectome(connectome, found_eta, found_gamma, data_dir, 
                    save_subdir="connectome_plots",
                    cluster_method=clustering_method, 
                    show=False)
        
        # Optional: Create comparison plot of all ordering methods
        compare_orderings(connectome, found_eta, found_gamma, data_dir, show=False)
        
        parameter_combination.append(found_eta)
        parameter_combination.append(found_gamma)

    import pandas as pd
    df_interesting_eta_and_gamma_combinations = pd.DataFrame(
        interesting_eta_and_gamma_combinations, 
        columns=["eta", "gamma", "found_eta", "found_gamma"], 
        index=None
    )
    df_interesting_eta_and_gamma_combinations.to_csv(save_path / "evaluated_eta_and_gamma_combinations.csv")
    
    
if __name__ == "__main__":
    main()
