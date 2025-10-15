import numpy as np
from netneurotools.networks import threshold_network
from bct import density_und


def threshold_to_density(conn_wei, n_nodes, density, output_folder, conn_type="consensus"): 
    # Threshold at multiple densities
    print("\nThresholding at multiple densities:")
    # for density in densities:
    print(f"  Density {density}%...", end=" ")
    
    if len(conn_wei.shape) == 2: 
        thres_conn = threshold_network(conn_wei, density)
        final_density = density_und(thres_conn)
        
    elif len(conn_wei.shape) == 3: 
        thres_conn_array = []
        densities_array = []
        number_of_edges_array = []
        
        for network in conn_wei: 
            thres_conn = threshold_network(network, density)
            thres_conn_array.append(thres_conn)
            dens, n, number_of_edges = density_und(thres_conn)
            densities_array.append(dens)
            number_of_edges_array.append(number_of_edges)
            
        thres_conn = np.array(thres_conn_array)
        number_of_edges_array = np.array(number_of_edges_array).mean()
        fin_density = np.array(densities_array).mean()
        final_density = [fin_density, n, number_of_edges_array]
            
        
    else: 
        print("ERROR: connectome arrays are neither 2D nor 3D")
    
    print(f"Final density: {final_density[0]:.6f} (edges: {final_density[2]})")
    
    # Save thresholded network
    np.save(
        output_folder / f"01_{conn_type}_bin_density_{density}_percent_{n_nodes}.npy",
        thres_conn
    )
    
    return thres_conn, final_density
        
        