import numpy as np 
import matplotlib.pyplot as plt

def get_distance_matrix(raw_data_path, preprocessed_folder_path, plot=True): 
    """
    Returns the distance matrix. 
    """
    
    coords = np.loadtxt(raw_data_path, delimiter=",")
    dist_mat = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1) # pairwise Euclidean distances
    np.fill_diagonal(dist_mat, 0.0)

    # Plot it
    if plot: 
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(dist_mat, cmap='Blues') 
        plt.colorbar()
        plt.title("Distance Matrix")
        plt.xlabel("Region Index")
        plt.ylabel("Region Index")
        plt.tight_layout()
        plt.show()

    # Sanity checks: 
    # - it is symmetric
    # - diagonal is 0
    assert np.allclose(dist_mat, dist_mat.T), "Distance matrix is not symmetric!"
    assert np.all(np.diag(dist_mat) == 0), "Diagonal of distance matrix is not zero!"

    # save distance matrix
    np.save(preprocessed_folder_path / f"distance_matrix_{dist_mat.shape[0]}x{dist_mat.shape[1]}.npy", dist_mat)
    
    return dist_mat