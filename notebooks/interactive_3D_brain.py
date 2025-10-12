import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px  # for color palettes
import os
from pathlib import Path
from nilearn import datasets
from nilearn.surface import load_surf_mesh


# --- User Configuration ---
# Update these file paths to match your files
COORDINATES_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/00_preprocessed/data_10_consensus/04_coordinates_68.csv"
CONNECTION_MATRIX_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed_old/01_first_analysises/connectomes_weighted_68x68.npy"
ROI_NAMES_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/00_preprocessed/data_10_consensus/05_roi_names_rsn_name_hemisphere_68.csv"

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

    own_factor = 9
    line_width = 0.5 + ((weight - w_min) / (w_max - w_min)) * own_factor if w_max > w_min else 0.5
    avg_z = (z0 + z1) / 2
    opacity = 0.1 + ((avg_z - z_min) / (z_max - z_min)) * 0.7 if z_max > z_min else 0.1
    edge_color = f'rgba(40, 40, 40, {opacity})'

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
    
    from matplotlib.colors import to_rgb
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



# --- Create Figure and Add JavaScript for Dynamic Updates ---
print("Creating final visualization with dynamic update script...")

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
                    # Set a default starting camera view
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='data'
                ),
                annotations=[]
            ))

# --- JavaScript Injection for the "Refresh Depth" Button ---
# We will pass the node and edge coordinates to the browser as a JSON object.
js_data = {
    'nodes': {i: pos[i].tolist() for i in G.nodes()},
    'edges': [{'u': u, 'v': v, 'w': data.get('weight', 0)} for u, v, data in G.edges(data=True)]
}

# This JavaScript code will be embedded in the final HTML file.
# It finds the plot, adds a button, and attaches the update logic to it.
javascript_code = f"""
<div style="position: absolute; top: 10px; left: 10px; z-index: 1000;">
    <button id="refresh-depth-btn" style="padding: 8px 12px; font-size: 14px; cursor: pointer;">Refresh Depth View</button>
</div>

<script>
    // Store the raw coordinate data passed from Python
    const graphData = {js_data};
    
    // Wait until the Plotly plot is fully rendered
    document.addEventListener('DOMContentLoaded', (event) => {{
        const plotDiv = document.querySelector('.js-plotly-plot');
        const refreshBtn = document.getElementById('refresh-depth-btn');

        if (plotDiv && refreshBtn) {{
            refreshBtn.addEventListener('click', () => updatePlotOpacities(plotDiv));
        }}
    }});

    function distance_to_opacity(dist, min_d, max_d) {{
        if (max_d <= min_d) return 0.9;
        const normalized_dist = (dist - min_d) / (max_d - min_d);
        return Math.max(0.15, 0.95 - (normalized_dist * 0.8));
    }}

    function updatePlotOpacities(gd) {{
        console.log("Updating opacities...");
        const cameraPos = gd.layout.scene.camera.eye;
        const cam = [cameraPos.x, cameraPos.y, cameraPos.z];

        // Calculate all node distances first to find min/max for normalization
        const node_positions = Object.values(graphData.nodes);
        const all_distances = node_positions.map(p => Math.hypot(p[0]-cam[0], p[1]-cam[1], p[2]-cam[2]));
        const dist_min = Math.min(...all_distances);
        const dist_max = Math.max(...all_distances);

        const updates = {{}};
        const new_node_colors = [];
        const new_edge_colors = [];
        let node_trace_count = 0;
        let edge_trace_count = 0;

        gd.data.forEach((trace, i) => {{
            // Identify traces by checking for marker vs line properties
            if (trace.mode.includes('markers')) {{ // This is a node trace
                const trace_colors = [];
                // Re-calculate color for each node in this trace based on its original color
                const base_rgb = trace.marker.color[0].match(/(\\d+,\\d+,\\d+)/)[0];
                for (let j = 0; j < trace.x.length; j++) {{
                    const node_pos = [trace.x[j], trace.y[j], trace.z[j]];
                    const dist = Math.hypot(node_pos[0]-cam[0], node_pos[1]-cam[1], node_pos[2]-cam[2]);
                    const opacity = distance_to_opacity(dist, dist_min, dist_max);
                    trace_colors.push(`rgba(${'{base_rgb}'},${'{opacity}'})`);
                }}
                if (!updates['marker.color']) updates['marker.color'] = [];
                updates['marker.color'][i] = trace_colors;

            }} else if (trace.mode === 'lines') {{ // This is an edge trace
                const p1 = [trace.x[0], trace.y[0], trace.z[0]];
                const p2 = [trace.x[1], trace.y[1], trace.z[1]];
                const mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2];
                const dist = Math.hypot(mid[0]-cam[0], mid[1]-cam[1], mid[2]-cam[2]);
                const opacity = distance_to_opacity(dist, dist_min, dist_max);

                if (!updates['line.color']) updates['line.color'] = [];
                updates['line.color'][i] = `rgba(40,40,40,${'{opacity}'})`;
            }}
        }});
        
        Plotly.restyle(gd, updates);
        console.log("Update complete.");
    }}
</script>
"""

# Save to HTML file, injecting the button and the JavaScript logic
save_path = save_folder / "network_visualization_dynamic_depth.html"
fig.write_html(save_path, post_script=javascript_code, include_plotlyjs='cdn')

print(f"Visualization saved to {save_path}")
print(f"Open this file in your browser, rotate the view, and press 'Refresh Depth View'.")



# # --- Create and Save Figure ---
# print("Creating final visualization...")
# fig = go.Figure(data=traces,
#              layout=go.Layout(
#                 title='3D Brain Network with Depth Cueing',
#                 showlegend=True,
#                 hovermode='closest',
#                 margin=dict(b=20, l=5, r=5, t=40),
#                 scene=dict(
#                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
#                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
#                     zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
#                     bgcolor='white',
#                     camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
#                     aspectmode='data'
#                 ),
#                 annotations=[]
#             ))

# save_path = save_folder / "network_visualization_with_depth_cueing.html"
# fig.write_html(save_path)
# print(f"Visualization saved to {save_path}")
# print(f"Open this file in your browser to view the interactive plot.")
