import numpy as np 


def get_individual_connectomes(raw_data_path, output_path): 
    """
    Get individual connectomes from 70 subjects
    """

    # Open the MAT-file as an HDF5 container. Loading with "sio.loadmat" does not work here, as the matlab version is too new (v7.3+).
    import h5py

    with h5py.File(raw_data_path, "r") as f:
        all_connectomes = np.array(f["M"]).T
        print("Data shape:", all_connectomes.shape) # (68, 68, 70) -> 68 regions, 68 regions, 70 subjects


    # Check the minimal value of all connectomes (to ensure no negative distances)
    print("Statistics of all connectomes:\n",
        "Min:", np.min(all_connectomes), "\n",
        "Max:", np.max(all_connectomes), "\n",
        "Mean:", np.mean(all_connectomes), "\n",
        "Std:", np.std(all_connectomes))


    # The connectomes sometimes have diagonal values that are not zero, so they are now set to zero
    for i in range(1, all_connectomes.shape[0]):
        all_connectomes[i,:,:][np.diag_indices_from(all_connectomes[i,:,:])] = 0.0
        # print(f"Checking connectome {i+1}...")
        # print(all_connectomes[i,:,:].shape)
        assert np.all(np.diag(all_connectomes[i,:,:]) == 0), "Diagonal of distance matrix is not zero!"

    # Sanity check:
    # - check if diagonal is zero for all connectomes
    for i in range(1, all_connectomes.shape[0]):
        assert np.all(np.diag(all_connectomes[i,:,:]) == 0), "Diagonal of distance matrix is not zero!"

    # Print information
    nsub, n, _ = all_connectomes.shape
    print(f"{nsub} subjects | {n} × {n} matrices")

    # Save the connectomes as numpy arrays
    np.save(output_path / f"all_connectomes_{n}_{n}.npy", all_connectomes)

    return all_connectomes, nsub, n
