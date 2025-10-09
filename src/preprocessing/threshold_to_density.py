from pathlib import Path
import pandas as pd
import numpy as np
import os
from netneurotools.networks import struct_consensus, threshold_network
from bct import density_und



# HELPER FUNCTIONS - can be probably be cleaned up. 




# def load_data(base_path, n_nodes):
#     """Load connectomes, distance matrix, and hemisphere identifiers."""
#     data_path = Path(base_path) / "01_first_analysises"
    
#     all_conns = np.load(data_path / f"all_connectomes_{n_nodes}_{n_nodes}.npy")
#     dist = np.load(data_path / f"distance_matrix_{n_nodes}x{n_nodes}.npy")
#     hemi_id = pd.read_csv(data_path / f"df_identifiers_{n_nodes}x{n_nodes}.csv")
#     hemi_id = hemi_id["hemi_id"].to_numpy().reshape(-1, 1)
    
#     return all_conns, dist, hemi_id


# def create_output_folder(base_path):
#     """Create output directory for consensus networks."""
#     output_folder = Path(base_path) / "02_calculated_consensus"
#     os.makedirs(output_folder, exist_ok=True)
#     return output_folder


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
        
        
# # def calculate_consensus(all_conns, dist, hemi_id, n_nodes, densities, output_folder):
# #     """Calculate weighted consensus and threshold at multiple densities."""
# #     print(f"\n{'='*60}")
# #     print(f"Processing {n_nodes} nodes")
# #     print(f"{'='*60}")
# #     print(f"Input shape: {all_conns.shape}")
    
# #     # Calculate weighted consensus
# #     print("Calculating weighted consensus...")
# #     weighted_consensus = struct_consensus(all_conns.T, dist, hemi_id, weighted=True)
    
# #     # Report initial density
# #     initial_density = density_und(weighted_consensus)
# #     print(f"Initial density: {initial_density[0]:.6f} (edges: {initial_density[2]})")
    
# #     # Save unthresholded consensus
# #     np.save(
# #         output_folder / f"consensus_{n_nodes}_weighted_unthresholded.npy",
# #         weighted_consensus
# #     )
    
# #     # Threshold at multiple densities
# #     print("\nThresholding at multiple densities:")
# #     for density in densities:
# #         threshold_to_density(weighted_consensus, n_nodes, density, output_folder)
        
# #         # print(f"  Density {density}%...", end=" ")
        
# #         # thres_conn = threshold_network(weighted_consensus, density)
# #         # final_density = density_und(thres_conn)
        
# #         # print(f"Final density: {final_density[0]:.6f} (edges: {final_density[2]})")
        
# #         # # Save thresholded network
# #         # np.save(
# #         #     output_folder / f"consensus_{n_nodes}_binarized_density_{density}_percent.npy",
# #         #     thres_conn
# #         # )

# def process_consensus_networks(): # -> I will not do this, as we already have the precalculated consensus networks... And these algos do not match... 
#     """Main execution function."""
#     # Configuration
#     # 00_preprocessed
#     base_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed"
#     n_nodes_list = [68, 114]
#     densities = [10, 12, 14, 16, 18, 20]
    
#     # Create output folder
#     output_folder = create_output_folder(base_path)
#     print(f"Output folder: {output_folder}")
    
#     # Process each node configuration
#     for n_nodes in n_nodes_list:
#         try:
#             # Load data
#             all_conns, dist, hemi_id = load_data(base_path, n_nodes)
            
#             # Calculate consensus networks
#             calculate_consensus(all_conns, dist, hemi_id, n_nodes, densities, output_folder)
            
#         except FileNotFoundError as e:
#             print(f"\nWarning: Could not process {n_nodes} nodes - {e}")
#             continue
#         except Exception as e:
#             print(f"\nError processing {n_nodes} nodes: {e}")
#             continue
    
#     print(f"\n{'='*60}")
#     print("Processing complete!")
#     print(f"{'='*60}")

# if __name__ == "__main__":
#     process_consensus_networks()






# # from pathlib  import Path
# # import pandas as pd 
# # import matplotlib.pyplot as plt 
# # import torch

# # from netneurotools.networks import struct_consensus, threshold_network
# # from bct import density_und
# # import numpy as np 

# # import os 

# # # do this for n in n_nodes = [68, 114] and for d in densities = [10,12,14,16,18,20]

# # all_conns = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/01_first_analysises/all_connectomes_68_68.npy")

# # output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/01_first_analysises/all_connectomes_68_68.npy").parent.parent / "02_calculated_consensus" 
# # os.makedirs(output_folder)


# # dist = np.load("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/01_first_analysises/distance_matrix_68x68.npy")

# # hemi_id = pd.read_csv("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/01_first_analysises/df_identifiers_68x68.csv")
# # hemi_id = hemi_id["hemi_id"].to_numpy().reshape(-1, 1)

# # print(all_conns.shape)
# # weighted_non_thresholded_consensus = struct_consensus(all_conns.T, dist, hemi_id, weighted=True)
# # print(density_und(weighted_non_thresholded_consensus)) # for my previously calculated connectomes... -> density (0.09833187006145742, 68, 224)
# # print("Thresholding starts")
# # thres_conn = threshold_network(weighted_non_thresholded_consensus, 10)
# # print(thres_conn.shape)
# # print(density_und(thres_conn)) 
# # print(thres_conn)
# # np.save(output_folder / "consensus_68_binarized_density_10_percent.npy", thres_conn)