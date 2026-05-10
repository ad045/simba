import json
import numpy as np

nb_path = "/Users/adrian/Documents/01_projects/14_4D_lab/connectome_distances/B_pca_and_pareto_trade-off/24_1_4_only_lexi_ages_clean_by_consensus_age.ipynb"
with open(nb_path, "r") as f:
    nb = json.load(f)

new_source = """# ── Robustness components vector addition ─────────────────────────────────────
# Find the exact category name for robustness
target_cat = next((cat for cat in CATEGORY_COLOURS.keys() if "robustness" in cat.lower()), None)

if target_cat:
    colour = CATEGORY_COLOURS[target_cat]
    
    # Variables in this category that also have loadings
    cat_vars = [c for c in loadings.index
                if c in meta.index and meta.loc[c, "Category"] == target_cat]
    
    if cat_vars:
        cat_loads = loadings.loc[cat_vars]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
        ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
        ax.scatter([0], [0], color="black", s=20, zorder=6)
        
        current_x, current_y = 0, 0
        
        for var_name, row in cat_loads.iterrows():
            vx = row["PC1"]
            vy = row["PC2"]
            
            # Draw the individual variable vector
            ax.quiver(
                current_x, current_y, vx, vy,
                angles="xy", scale_units="xy", scale=1,
                color=colour, width=0.008, alpha=0.6,
                headwidth=4, headlength=5, headaxislength=4,
                zorder=4,
            )
            
            # Label with friendly property name if available, else var_name
            nice_name = PROPERTY_NAMES.get(var_name, var_name) if "PROPERTY_NAMES" in globals() else var_name
            
            # Place label slightly offset from the midpoint of the arrow
            mid_x = current_x + vx / 2
            mid_y = current_y + vy / 2
            
            # Small heuristic for label offset to avoid overlapping the line
            offset_x = 0.05 * np.sign(vx) if vx != 0 else 0.05
            offset_y = 0.05 * np.sign(vy) if vy != 0 else 0.05
            
            ax.text(mid_x + offset_x, mid_y + offset_y, nice_name,
                    fontsize=8, ha="center", va="center",
                    color=colour, alpha=0.9,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
            
            current_x += vx
            current_y += vy
            
        # Draw the final sum vector
        ax.quiver(
            0, 0, current_x, current_y,
            angles="xy", scale_units="xy", scale=1,
            color="black", width=0.012,
            headwidth=4, headlength=5, headaxislength=4,
            zorder=5,
        )
        
        nudge_x = 0.05 * np.sign(current_x) if current_x != 0 else 0.05
        nudge_y = 0.05 * np.sign(current_y) if current_y != 0 else 0.05
        ax.text(current_x + nudge_x, current_y + nudge_y, f"Total {target_cat}",
                fontsize=10, ha="center", va="center",
                color="black", fontweight="bold")
                
        # Calculate axis limits
        max_lim = 0.1
        cx, cy = 0, 0
        all_x, all_y = [0], [0]
        for _, row in cat_loads.iterrows():
            cx += row["PC1"]
            cy += row["PC2"]
            all_x.append(cx)
            all_y.append(cy)
            
        max_x = max(max(all_x), abs(min(all_x))) * 1.3
        max_y = max(max(all_y), abs(min(all_y))) * 1.3
        max_lim = max(max_x, max_y, 0.1)

        ax.set_xlim(-max_lim, max_lim)
        ax.set_ylim(-max_lim, max_lim)
        ax.set_aspect("equal")
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.set_title(f"Composition of '{target_cat}' vector in PCA space", fontsize=11)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        plt.show()
else:
    print("Could not find a category matching 'robustness'")
"""

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in new_source.split("\n")]
}
if new_cell["source"]:
    new_cell["source"][-1] = new_cell["source"][-1].rstrip("\n")

insert_idx = len(nb["cells"])
# Find the cell containing the user's cursor
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "category_vectors_in_PCA_space" in src or "Category vectors in PCA space\\n" in src or "for spine in ax.spines.values():" in src and "ax.quiver" in src and "cat_vectors.items()" in src:
            insert_idx = i + 1

print(f"Inserting new cell at index {insert_idx}")
nb["cells"].insert(insert_idx, new_cell)

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)
    
print("Successfully wrote notebook.")
