import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import os
from pathlib import Path 

# --- User Configuration ---
# Update these file paths to match your files
COORDINATES_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/00_just_converted_for_matlab_and_python/data_10_consensus/04_coordinates_68.csv"
save_folder = Path(CONNECTION_MATRIX_FILE).parent / "3D_visualizations"
save_folder.mkdir(parents=True, exist_ok=True)

# --- Data Loading with Better Error Handling ---
def load_data():
    """Load coordinates and connection matrix with proper error handling."""
    try:
        # Check if files exist
        if not os.path.exists(COORDINATES_FILE):
            raise FileNotFoundError(f"Coordinates file not found: {COORDINATES_FILE}")
        if not os.path.exists(CONNECTION_MATRIX_FILE):
            raise FileNotFoundError(f"Connection matrix file not found: {CONNECTION_MATRIX_FILE}")
        
        # Load the 3D coordinates for each node from the CSV file
        print("Loading coordinates...")
        coords_df = pd.read_csv(COORDINATES_FILE, header=None)
        coordinates = coords_df.values
        print(f"Loaded coordinates for {len(coordinates)} nodes")
        
        # Validate coordinates shape
        if coordinates.shape[1] != 3:
            raise ValueError(f"Expected 3D coordinates (3 columns), got {coordinates.shape[1]} columns")
        
        # Load the connection matrix (adjacency matrix) from the .npy file
        print("Loading connection matrix...")
        connection_data = np.load(CONNECTION_MATRIX_FILE)
        
        # Handle different possible shapes of the loaded data
        if connection_data.ndim == 3:
            connection_matrix = connection_data[0, :, :]
        elif connection_data.ndim == 2:
            connection_matrix = connection_data
        else:
            raise ValueError(f"Unexpected connection matrix dimensions: {connection_data.shape}")
        
        print(f"Loaded connection matrix of shape: {connection_matrix.shape}")
        
        # Validate that coordinates and connection matrix match
        if len(coordinates) != connection_matrix.shape[0]:
            raise ValueError(f"Mismatch: {len(coordinates)} coordinate points but {connection_matrix.shape[0]} nodes in connection matrix")
        
        return coordinates, connection_matrix
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the file paths in the configuration section.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Load data
coordinates, connection_matrix = load_data()
if coordinates is None or connection_matrix is None:
    print("Failed to load data. Exiting.")
    exit()

# --- Graph Creation ---
print("Creating network graph...")

# Create a graph object from the connection matrix
G = nx.from_numpy_array(connection_matrix)

# Create a dictionary to hold the 3D position of each node
pos = {i: coordinates[i] for i in range(len(coordinates))}

print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# --- Prepare Data for Plotting ---
print("Preparing visualization data...")

# Extract node positions into separate lists for Plotly
node_x = [pos[i][0] for i in G.nodes()]
node_y = [pos[i][1] for i in G.nodes()]
node_z = [pos[i][2] for i in G.nodes()]

# Extract edge positions
# Edges are drawn as lines between nodes. We create lists of coordinates,
# using 'None' to create breaks in the line between separate edges.
edge_x = []
edge_y = []
edge_z = []
for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])

# --- Create Plotly Traces ---
# Create the 3D trace for the edges (lines)
edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(width=0.8, color='#888'),
    hoverinfo='none',
    mode='lines',
    name='Connections'
)

# Calculate node properties
node_adjacencies = []
node_text = []
for i, adjacencies in enumerate(G.adjacency()):
    degree = len(adjacencies[1])
    node_adjacencies.append(degree)
    node_text.append(f'Node {i}<br># of connections: {degree}')

# Create the 3D trace for the nodes (markers)
node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    text=node_text,
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        reversescale=True,
        color=node_adjacencies,
        size=6,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            # titleside='right'
        ),
        line=dict(width=2)
    ),
    name='Nodes'
)

# --- Create and Display Figure ---
print("Creating visualization...")

# Combine the edge and node traces into a single figure
fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Interactive 3D Network Visualization',
                # titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                scene=dict(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    bgcolor='rgb(240, 240, 240)'
                ),
                annotations=[
                    dict(
                        text="Node color indicates degree centrality (number of connections)",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ]
            ))

# Save to HTML file instead of showing in browser
save_path = Path(save_folder) / "network_visualization.html"
fig.write_html(save_path)
print("Visualization saved to", save_path)
print("Open this file in your browser to view the interactive plot:", save_path)

