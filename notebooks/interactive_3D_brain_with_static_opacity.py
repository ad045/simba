
# import numpy as np
# import pandas as pd
# import networkx as nx
# import plotly.graph_objects as go
# import plotly.express as px # for color palettes
# import os
# from pathlib import Path
# from nilearn import datasets, surface
# from scipy.spatial import Delaunay
# from nilearn.surface import load_surf_mesh

# # --- User Configuration ---
# # Update these file paths to match your files
# COORDINATES_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/00_just_converted_for_matlab_and_python/data_10_consensus/04_coordinates_68.csv"
# CONNECTION_MATRIX_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed_old/01_first_analysises/connectomes_weighted_68x68.npy" 
# # /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/24_testing_4_KS_folders_rougher_grid/all_generated_networks/24_testing_4_KS_folders_rougher_grid_20250930_041936_net_eta-1.170_gamma0.485_ruleMatchingIndex.npy"
# ROI_NAMES_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/00_just_converted_for_matlab_and_python/data_10_consensus/05_roi_names_rsn_name_hemisphere_68.csv"

# save_folder = Path(CONNECTION_MATRIX_FILE).parent / "3D_visualizations"
# save_folder.mkdir(parents=True, exist_ok=True)


# # --- Data Loading with Better Error Handling ---
# def load_data():
#     """Load coordinates and connection matrix with proper error handling."""
#     try:
#         if not os.path.exists(COORDINATES_FILE):
#             raise FileNotFoundError(f"Coordinates file not found: {COORDINATES_FILE}")
#         if not os.path.exists(CONNECTION_MATRIX_FILE):
#             raise FileNotFoundError(f"Connection matrix file not found: {CONNECTION_MATRIX_FILE}")
        
#         print("Loading coordinates...")
#         coords_df = pd.read_csv(COORDINATES_FILE, header=None)
#         coordinates = coords_df.values
#         print(f"Loaded coordinates for {len(coordinates)} nodes")
        
#         if coordinates.shape[1] != 3:
#             raise ValueError(f"Expected 3D coordinates (3 columns), got {coordinates.shape[1]} columns")
        
#         print("Loading connection matrix...")
#         connection_data = np.load(CONNECTION_MATRIX_FILE)
        
#         if connection_data.ndim == 3:
#             connection_matrix = connection_data[0, :, :]
#         elif connection_data.ndim == 2:
#             connection_matrix = connection_data
#         else:
#             raise ValueError(f"Unexpected connection matrix dimensions: {connection_data.shape}")
        
#         print(f"Loaded connection matrix of shape: {connection_matrix.shape}")
        
#         if len(coordinates) != connection_matrix.shape[0]:
#             raise ValueError(f"Mismatch: {len(coordinates)} coordinate points but {connection_matrix.shape[0]} nodes in connection matrix")

#         return coordinates, connection_matrix
        
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         print("Please update the file paths in the configuration section.")
#         return None, None
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return None, None

# def load_brain_surface_mni():
#     """Load brain surface in MNI space using nilearn."""
#     try:
#         print("Loading brain surface in MNI space...")
        
#         # Fetch fsaverage surface
#         fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
        
#         # Load both hemispheres with their mesh data
#         lh_coords, lh_faces = load_surf_mesh(fsaverage['pial_left'])
#         rh_coords, rh_faces = load_surf_mesh(fsaverage['pial_right'])
        
#         # FreeSurfer to MNI transformation
#         # FreeSurfer surface coordinates need to be transformed to MNI space
#         # Standard transformation parameters
#         def freesurfer_to_mni(coords):
#             # Apply typical FreeSurfer to MNI152 transformation
#             # Scale and translate to align with MNI space
#             mni_coords = coords.copy()
#             # FreeSurfer is in mm, roughly centered, but needs adjustment
#             # Typical scaling factor and translation
#             mni_coords = mni_coords * np.array([1.0, 1.0, 1.0])  # May need adjustment
#             return mni_coords
        
#         lh_coords_mni = freesurfer_to_mni(lh_coords)
#         rh_coords_mni = freesurfer_to_mni(rh_coords)
        
#         # Combine hemispheres
#         rh_faces_offset = rh_faces + len(lh_coords_mni)
#         all_coords = np.vstack([lh_coords_mni, rh_coords_mni])
#         all_faces = np.vstack([lh_faces, rh_faces_offset])
        
#         print(f"Loaded brain surface with {len(all_coords)} vertices and {len(all_faces)} faces")
#         return all_coords, all_faces
        
#     except Exception as e:
#         print(f"Error loading brain surface: {e}")
#         print("Make sure nilearn is installed: pip install nilearn")
#         return None, None



# def align_brain_to_nodes(brain_coords, node_coords):
#     """
#     Align brain surface to nodes by rotating, flipping, and then centering.
#     This corrects for common orientation mismatches between fsaverage and MNI space.
#     """
#     print("Aligning brain surface to network coordinates...")
    
#     # --- Step 1: Correct the rotation and orientation ---
#     # A common transformation from fsaverage surface space to MNI space involves
#     # swapping the Y and Z axes and inverting the new Y axis.
#     # We transform the brain from its original (x, y, z) to a new (x, -z, y).
#     corrected_coords = brain_coords.copy()
#     corrected_coords[:, [0, 1, 2]] = corrected_coords[:, [1, 0, 2]]  # Swap stuff
#     # corrected_coords[[0,1,2]] = corrected_coords[[1,2,0]]  # Swap stuff
#     # corrected_coords[:, 1] *= -1                               # Invert the new Y axis
#     corrected_coords[:, 0] *= -1          
#     # corrected_coords = np.transpose(corrected_coords, (1, 0, 2))
    
#     # --- Step 2: Correct the position by matching centers ---
#     # Now, center the correctly rotated brain on the nodes
#     node_center = node_coords.mean(axis=0)
#     brain_center = corrected_coords.mean(axis=0)
    
#     aligned_coords = corrected_coords - brain_center + node_center
    
#     print("Brain mesh rotated and repositioned.")
#     return aligned_coords

# def create_brain_mesh_trace(coords, faces, opacity=0.15, color='lightgray'):
#     """Create a Plotly mesh3d trace for the brain surface."""
#     mesh_trace = go.Mesh3d(
#         x=coords[:, 0],
#         y=coords[:, 1],
#         z=coords[:, 2],
#         i=faces[:, 0],
#         j=faces[:, 1],
#         k=faces[:, 2],
#         opacity=opacity,
#         color=color,
#         name='Brain Surface',
#         hoverinfo='skip',
#         flatshading=False,
#         lighting=dict(
#             ambient=0.6,
#             diffuse=0.8,
#             specular=0.1,
#             roughness=0.5,
#             fresnel=0.2
#         ),
#         lightposition=dict(
#             x=100,
#             y=200,
#             z=0
#         )
#     )
#     return mesh_trace

# # Load data
# coordinates, connection_matrix = load_data()
# if coordinates is None or connection_matrix is None:
#     print("Failed to load data. Exiting.")
#     exit()
    
# print("Loading ROI information...")
# roi_info = pd.read_csv(ROI_NAMES_FILE, header=None, names=['full_name', 'abbr', 'network', 'hemisphere'])


# # Load brain surface and align it to the node coordinates
# brain_coords, brain_faces = load_brain_surface_mni()
# if brain_coords is not None and brain_faces is not None:
#     brain_coords = align_brain_to_nodes(brain_coords, coordinates)

# # --- Graph Creation ---
# print("Creating network graph...")

# G = nx.from_numpy_array(connection_matrix)
# pos = {i: coordinates[i] for i in range(len(coordinates))}

# print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# # --- Prepare Data for Plotting ---
# print("Preparing visualization data...")

# node_x = [pos[i][0] for i in G.nodes()]
# node_y = [pos[i][1] for i in G.nodes()]
# node_z = [pos[i][2] for i in G.nodes()]

# edge_x = []
# edge_y = []
# edge_z = []
# for edge in G.edges():
#     x0, y0, z0 = pos[edge[0]]
#     x1, y1, z1 = pos[edge[1]]
#     edge_x.extend([x0, x1, None])
#     edge_y.extend([y0, y1, None])
#     edge_z.extend([z0, z1, None])

# # --- Create Plotly Traces ---
# traces = []

# # Add brain surface if loaded successfully
# if brain_coords is not None and brain_faces is not None:
#     brain_trace = create_brain_mesh_trace(brain_coords,
#                                           brain_faces, 
#                                           opacity=0.1, 
#                                           color="#A4D3F7") # '#F5E6D3')
#     traces.append(brain_trace)




# # --- Create Edge Traces with Strength and Depth ---
# print("Creating edge traces with variable properties...")

# # Get all z-coordinates and weights for normalization
# all_z = np.array([pos[i][2] for i in G.nodes()])
# z_min, z_max = all_z.min(), all_z.max()
# weights = [d['weight'] for u, v, d in G.edges(data=True)]
# w_min, w_max = (min(weights), max(weights)) if weights else (0, 1)

# # Create a trace for each edge to control individual properties
# for u, v, data in G.edges(data=True):
#     x0, y0, z0 = pos[u]
#     x1, y1, z1 = pos[v]
#     weight = data.get('weight', 0)

#     # Normalize weight to a visible line width (e.g., 0.5 to 5 pixels)
#     own_factor = 4
#     line_width = 0.5 + ((weight - w_min) / (w_max - w_min)) * 4.5 * own_factor if w_max > w_min else 0.5

#     # Normalize the edge's average depth to an opacity value (0.1 to 0.8)
#     # Edges farther back (smaller z) will be more transparent, creating a fog effect
#     avg_z = (z0 + z1) / 2
#     opacity = 0.1 + ((avg_z - z_min) / (z_max - z_min)) * 0.7 if z_max > z_min else 0.1
    
#     gray_value = 0 # 40 
#     edge_color = f'rgba({gray_value}, {gray_value}, {gray_value}, {opacity})' # Dark grey with variable alpha

#     traces.append(go.Scatter3d(
#         x=[x0, x1],
#         y=[y0, y1],
#         z=[z0, z1],
#         mode='lines',
#         line=dict(width=line_width, color=edge_color),
#         hoverinfo='none',
#         showlegend=False
#     ))


# # Node properties
# node_adjacencies = []
# node_text = []
# for i, adjacencies in enumerate(G.adjacency()):
#     degree = len(adjacencies[1])
#     node_adjacencies.append(degree)
#     node_text.append(f'Node {i}<br># of connections: {degree}')




# # Node traces, colored by network
# networks = roi_info['network'].unique()
# colors = px.colors.qualitative.Plotly  # Get a color palette


# for i, network_name in enumerate(networks):
#     # Find the indices of nodes belonging to the current network
#     network_nodes = roi_info[roi_info['network'] == network_name]
#     node_indices = network_nodes.index

#     # Create hover text with ROI and network name
#     hover_texts = [f"ROI: {row.full_name}<br>Network: {row.network}" 
#                    for _, row in network_nodes.iterrows()]

#     node_trace = go.Scatter3d(
#         x=coordinates[node_indices, 0],
#         y=coordinates[node_indices, 1],
#         z=coordinates[node_indices, 2],
#         mode='markers',
#         name=network_name,  # This name will appear in the legend
#         hovertext=hover_texts,
#         hoverinfo='text',
#         marker=dict(
#             color=colors[i % len(colors)], # Assign a color for the network
#             size=8,
#             line=dict(width=1, color='white')
#         )
#     )
#     traces.append(node_trace)
    

# # --- Create Figure ---
# print("Creating visualization...")

# fig = go.Figure(data=traces,
#              layout=go.Layout(
#                 title='3D Brain Network with Connection Strength and Depth Cueing',
#                 showlegend=True, # Show legend for node networks
#                 hovermode='closest',
#                 margin=dict(b=20, l=5, r=5, t=40),
#                 scene=dict(
#                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
#                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
#                     zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
#                     # Set the background to white
#                     bgcolor='white', 
#                     camera=dict(
#                         eye=dict(x=1.5, y=1.5, z=1.5)
#                     ),
#                     aspectmode='data'
#                 ),
#                 # Remove or update annotations
#                 annotations=[] 
#             ))


# # Save to HTML file
# save_path = save_folder / "network_visualization_aligned_brain.html"
# fig.write_html(save_path)
# print(f"Visualization saved to {save_path}")
# print(f"Open this file in your browser to view the interactive plot")


import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px  # for color palettes
import os
from pathlib import Path
from nilearn import datasets
from nilearn.surface import load_surf_mesh
from matplotlib.colors import to_rgb

# --- User Configuration ---
# Update these file paths to match your files
COORDINATES_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/00_just_converted_for_matlab_and_python/data_10_consensus/04_coordinates_68.csv"
CONNECTION_MATRIX_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed_old/01_first_analysises/connectomes_weighted_68x68.npy"
ROI_NAMES_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/00_just_converted_for_matlab_and_python/data_10_consensus/05_roi_names_rsn_name_hemisphere_68.csv"

save_folder = Path(CONNECTION_MATRIX_FILE).parent / "3D_visualizations"
save_folder.mkdir(parents=True, exist_ok=True)

# --- Visualization Configuration ---
# Set node coloring mode: 'network' for coloring by network, 'uniform' for a single color.
NODE_COLORING_MODE = 'uniform'  # Options: 'network' or 'uniform'
UNIFORM_NODE_COLOR = 'blue'    # Color to use if NODE_COLORING_MODE is 'uniform'

# Set base opacity for the brain surface.
# Note: Plotly Mesh3d doesn't support per-vertex opacity for a fog effect. This sets a global transparency.
BRAIN_OPACITY = 0.1
BRAIN_COLOR = '#A4D3F7'
CONNECTION_COLOR = '#028cf5'


# --- Data Loading with Better Error Handling ---
def load_data():
    """Load coordinates and connection matrix with proper error handling."""
    try:
        if not os.path.exists(COORDINATES_FILE):
            raise FileNotFoundError(f"Coordinates file not found: {COORDINATES_FILE}")
        if not os.path.exists(CONNECTION_MATRIX_FILE):
            raise FileNotFoundError(f"Connection matrix file not found: {CONNECTION_MATRIX_FILE}")
        if not os.path.exists(ROI_NAMES_FILE):
            raise FileNotFoundError(f"ROI names file not found: {ROI_NAMES_FILE}")

        print("Loading coordinates...")
        coords_df = pd.read_csv(COORDINATES_FILE, header=None)
        coordinates = coords_df.values
        print(f"Loaded coordinates for {len(coordinates)} nodes")

        if coordinates.shape[1] != 3:
            raise ValueError(f"Expected 3D coordinates (3 columns), got {coordinates.shape[1]} columns")

        print("Loading connection matrix...")
        connection_data = np.load(CONNECTION_MATRIX_FILE)

        if connection_data.ndim == 3:
            connection_matrix = connection_data[0, :, :]
        elif connection_data.ndim == 2:
            connection_matrix = connection_data
        else:
            raise ValueError(f"Unexpected connection matrix dimensions: {connection_data.shape}")

        print(f"Loaded connection matrix of shape: {connection_matrix.shape}")

        if len(coordinates) != connection_matrix.shape[0]:
            raise ValueError(f"Mismatch: {len(coordinates)} coordinate points but {connection_matrix.shape[0]} nodes in connection matrix")
            
        print("Loading ROI information...")
        roi_info = pd.read_csv(ROI_NAMES_FILE, header=None, names=['full_name', 'abbr', 'network', 'hemisphere'])

        return coordinates, connection_matrix, roi_info

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the file paths in the configuration section.")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def load_brain_surface_mni():
    """Load fsaverage5 brain surface using nilearn."""
    try:
        print("Loading brain surface...")
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
        lh_coords, lh_faces = load_surf_mesh(fsaverage['pial_left'])
        rh_coords, rh_faces = load_surf_mesh(fsaverage['pial_right'])

        # Combine hemispheres
        rh_faces_offset = rh_faces + len(lh_coords)
        all_coords = np.vstack([lh_coords, rh_coords])
        all_faces = np.vstack([lh_faces, rh_faces_offset])

        print(f"Loaded brain surface with {len(all_coords)} vertices and {len(all_faces)} faces")
        return all_coords, all_faces

    except Exception as e:
        print(f"Error loading brain surface: {e}")
        print("Make sure nilearn is installed: pip install nilearn")
        return None, None

def align_brain_to_nodes(brain_coords, node_coords):
    """Align brain surface to nodes by rotating, flipping, and then centering."""
    print("Aligning brain surface to network coordinates...")
    # Correct rotation and orientation from fsaverage to MNI-like space
    corrected_coords = brain_coords.copy()
    corrected_coords[:, [0, 1, 2]] = corrected_coords[:, [1, 0, 2]]
    corrected_coords[:, 0] *= -1

    # Center the correctly rotated brain on the nodes
    node_center = node_coords.mean(axis=0)
    brain_center = corrected_coords.mean(axis=0)
    aligned_coords = corrected_coords - brain_center + node_center

    print("Brain mesh rotated and repositioned.")
    return aligned_coords

def create_brain_mesh_trace(coords, faces, opacity=0.1, color='lightgray'):
    """Create a Plotly mesh3d trace for the brain surface."""
    return go.Mesh3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=opacity,
        color=color,
        name='Brain Surface',
        hoverinfo='skip',
        flatshading=False,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1, roughness=0.5, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=0)
    )

# --- Main Script Execution ---
coordinates, connection_matrix, roi_info = load_data()
if coordinates is None:
    print("Failed to load data. Exiting.")
    exit()

# Load brain surface and align it to the node coordinates
brain_coords, brain_faces = load_brain_surface_mni()
if brain_coords is not None:
    aligned_brain_coords = align_brain_to_nodes(brain_coords, coordinates)

# --- Graph Creation ---
print("Creating network graph...")
G = nx.from_numpy_array(connection_matrix)
pos = {i: coordinates[i] for i in range(len(coordinates))}
print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# --- Prepare Data for Plotting ---
print("Preparing visualization data...")
traces = []

# Add brain surface trace if loaded successfully
if 'aligned_brain_coords' in locals() and brain_faces is not None:
    brain_trace = create_brain_mesh_trace(aligned_brain_coords, brain_faces, opacity=BRAIN_OPACITY, color=BRAIN_COLOR)
    traces.append(brain_trace)

# --- Create Edge Traces with Strength and Depth Cueing ---
print("Creating edge traces...")
all_z = np.array([pos[i][2] for i in G.nodes()])
z_min, z_max = (all_z.min(), all_z.max()) if all_z.size > 0 else (0, 0)
weights = [d['weight'] for _, _, d in G.edges(data=True)]
w_min, w_max = (min(weights), max(weights)) if weights else (0, 1)

for u, v, data in G.edges(data=True):
    x0, y0, z0 = pos[u]
    x1, y1, z1 = pos[v]
    weight = data.get('weight', 0)

    own_factor = 12
    line_width = 0.5 + ((weight - w_min) / (w_max - w_min)) * own_factor if w_max > w_min else 0.5
    avg_z = (z0 + z1) / 2
    opacity = line_width # 0.1 + ((avg_z - z_min) / (z_max - z_min)) * 0.7 if z_max > z_min else 0.1
    # edge_color = f'rgba(40, 40, 40, {opacity})'
    # edge_color = f'rgba(0, 0, 255, {opacity})'
    r, g, b = to_rgb(CONNECTION_COLOR)
    edge_color = f'rgba({r},{g},{b},{opacity})'
    
    traces.append(go.Scatter3d(
        x=[x0, x1], y=[y0, y1], z=[z0, z1],
        mode='lines',
        line=dict(width=line_width, color=edge_color),
        hoverinfo='none',
        showlegend=False
    ))

# --- Create Node Traces with Depth Cueing and Color Options ---
if NODE_COLORING_MODE == 'uniform':
    print("Creating node traces with uniform color and depth cueing...")
    node_opacities = [0.2 + ((pos[i][2] - z_min) / (z_max - z_min)) * 0.8 if z_max > z_min else 0.2 for i in G.nodes()]
    
    r, g, b = to_rgb(UNIFORM_NODE_COLOR)
    node_colors = [f'rgba({r},{g},{b},{opacity})' for opacity in node_opacities]
    
    hover_texts = [f"ROI: {roi_info.loc[i, 'full_name']}<br>Network: {roi_info.loc[i, 'network']}" for i in G.nodes()]

    node_trace = go.Scatter3d(
        x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
        mode='markers',
        name='Nodes',
        hovertext=hover_texts,
        hoverinfo='text',
        marker=dict(color=node_colors, size=8, line=dict(width=1, color='white'))
    )
    traces.append(node_trace)

elif NODE_COLORING_MODE == 'network':
    print("Creating node traces with network-based color and depth cueing...")
    networks = roi_info['network'].unique()
    color_palette = px.colors.qualitative.Plotly

    for i, network_name in enumerate(networks):
        network_nodes_df = roi_info[roi_info['network'] == network_name]
        node_indices = network_nodes_df.index
        if len(node_indices) == 0:
            continue

        network_coords = coordinates[node_indices]
        node_opacities = [0.2 + ((coord[2] - z_min) / (z_max - z_min)) * 0.8 if z_max > z_min else 0.2 for coord in network_coords]
        
        base_color_hex = color_palette[i % len(color_palette)]
        
        from matplotlib.colors import to_rgb
        r, g, b = to_rgb(base_color_hex)
        # r, g, b = to_rgb_tuple(base_color_hex)
        node_colors = [f'rgba({r},{g},{b},{opacity})' for opacity in node_opacities]

        hover_texts = [f"ROI: {row.full_name}<br>Network: {row.network}" for _, row in network_nodes_df.iterrows()]

        traces.append(go.Scatter3d(
            x=network_coords[:, 0], y=network_coords[:, 1], z=network_coords[:, 2],
            mode='markers',
            name=network_name,
            hovertext=hover_texts,
            hoverinfo='text',
            marker=dict(color=node_colors, size=8, line=dict(width=1, color='white'))
        ))

# --- Create and Save Figure ---
print("Creating final visualization...")
fig = go.Figure(data=traces,
             layout=go.Layout(
                title='3D Brain Network with Depth Cueing',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                scene=dict(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    bgcolor='white',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='data'
                ),
                annotations=[]
            ))

save_path = save_folder / "network_visualization_with_depth_cueing.html"
fig.write_html(save_path)
print(f"Visualization saved to {save_path}")
print(f"Open this file in your browser to view the interactive plot.")
