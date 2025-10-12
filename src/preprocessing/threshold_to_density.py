import numpy as np
from netneurotools.networks import threshold_network
from bct import density_und


def threshold_to_density(consensus_wei, n_nodes, density, output_folder): 
    # Threshold at multiple densities
    print("\nThresholding at multiple densities:")
    # for density in densities:
    print(f"  Density {density}%...", end=" ")
    
    thres_conn = threshold_network(consensus_wei, density)
    final_density = density_und(thres_conn)
    
    print(f"Final density: {final_density[0]:.6f} (edges: {final_density[2]})")
    
    # Save thresholded network
    np.save(
        output_folder / f"01_consensus_bin_density_{density}_percent_{n_nodes}.npy",
        thres_conn
    )
    
    return thres_conn, final_density
        
        