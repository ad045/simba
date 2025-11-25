import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Your CSV data as a string (or load from file with pd.read_csv('your_file.csv'))
csv_data = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_100.csv"
# pd.read_csv("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_100.csv")

# Parse the CSV
from io import StringIO
df = pd.read_csv(StringIO(csv_data))

# Create a directed graph
G = nx.DiGraph()

# Define the taxonomic hierarchy (from broad to specific)
hierarchy = ['phylogenetic_group', 'super_order', 'order', 'family', 'genus', 'animal']

# Add root node
G.add_node('Mammalia', level=-1, label='Mammalia')

# Build the tree
for _, row in df.iterrows():
    parent = 'Mammalia'
    
    for level_idx, level in enumerate(hierarchy):
        if pd.notna(row[level]) and row[level] != '':
            node_name = row[level]
            
            # Add node if it doesn't exist
            if node_name not in G:
                label = row['common_name'] if level == 'animal' else node_name
                G.add_node(node_name, level=level_idx, taxonomy=level, label=label)
            
            # Add edge from parent to current node
            if not G.has_edge(parent, node_name):
                G.add_edge(parent, node_name)
            
            parent = node_name

# Use springs
pos = {}
levels = defaultdict(list)
for node in G.nodes():
    level = G.nodes[node]['level']
    levels[level].append(node)

for level, nodes in levels.items():
    y = -level * 2
    for i, node in enumerate(nodes):
        x = (i - len(nodes)/2) * 2
        pos[node] = (x, y)

# Create the plot
plt.figure(figsize=(10, 8)) # 20, 16))

# Define colors for each level
level_colors = {
    -1: '#8b4513',  # Root - brown
    0: '#e74c3c',   # phylogenetic_group - red
    1: '#e67e22',   # super_order - orange
    2: '#f39c12',   # order - yellow
    3: '#2ecc71',   # family - green
    4: '#3498db',   # genus - blue
    5: '#9b59b6'    # animal - purple
}

# Get node colors based on level
node_colors = [level_colors[G.nodes[node]['level']] for node in G.nodes()]

# Get node sizes based on level (smaller for species)
scalar = 100 # 1000 
node_sizes = [3*scalar if G.nodes[node]['level'] == -1 
              else 15*scalar if G.nodes[node]['level'] < 3
              else 8*scalar if G.nodes[node]['level'] < 5
              else 4*scalar for node in G.nodes()]

# Draw the network
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', 
                       arrows=True, arrowsize=10, width=0.5)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                       node_size=node_sizes, alpha=0.9)

# Draw labels (only for higher levels to avoid clutter)
labels = {}
for node in G.nodes():
    if G.nodes[node]['level'] <= 3:  # Only label up to family level
        labels[node] = G.nodes[node].get('label', node)

nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

plt.title('Mammal Phylogenetic Tree\n(Taxonomic Hierarchy)', 
          fontsize=20, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()

# # Add legend
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='#8b4513', label='Root (Mammalia)'),
#     Patch(facecolor='#e74c3c', label='Phylogenetic Group'),
#     Patch(facecolor='#e67e22', label='Super Order'),
#     Patch(facecolor='#f39c12', label='Order'),
#     Patch(facecolor='#2ecc71', label='Family'),
#     Patch(facecolor='#3498db', label='Genus'),
#     Patch(facecolor='#9b59b6', label='Species')
# ]
# plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

# plt.savefig('mammal_phylogenetic_tree.png', dpi=300, bbox_inches='tight')
# plt.show()

print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"\nTaxonomic levels:")
for level in range(-1, 6):
    nodes_at_level = [n for n in G.nodes() if G.nodes[n]['level'] == level]
    level_name = ['Root', 'Phylo Group', 'Super Order', 'Order', 'Family', 'Genus', 'Species'][level+1]
    print(f"  {level_name}: {len(nodes_at_level)} nodes")