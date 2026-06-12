import numpy as np
import matplotlib.pyplot as plt
from neurolib.models.aln import ALNModel
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import seaborn as sns

# Load your connectome (replace with your actual data)
connectome = np.load('/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/01_first_analysises/connectomes_weighted_68x68.npy') 
connectome = connectome[0,:,:]

# For demonstration, create a random 68x68 connectome
# np.random.seed(42)
# connectome = np.random.rand(68, 68)
# connectome = (connectome + connectome.T) / 2  # Make symmetric
# np.fill_diagonal(connectome, 0)  # No self-connections

# Normalize connectome
connectome = connectome / np.max(connectome)

print(f"Connectome shape: {connectome.shape}")
print(f"Connectome range: [{np.min(connectome):.3f}, {np.max(connectome):.3f}]")

# Create fiber length matrix (required by neurolib, can be uniform if unknown)
Dmat = np.ones_like(connectome) * 20.0  # Average 20mm length
np.fill_diagonal(Dmat, 0)

# Initialize ALN model with your connectome
model = ALNModel(Cmat=connectome, Dmat=Dmat)

# Set simulation parameters
model.params['duration'] = 10000  # 10 seconds in ms
model.params['dt'] = 0.1  # Integration time step

# Run the simulation
print("\nRunning whole-brain simulation...")
model.run()
print("Simulation complete!")

# Extract activity data
# rates_exc contains excitatory firing rates for each node over time
rates_exc = model.rates_exc  # Shape: (n_nodes, n_timepoints)
rates_inh = model.rates_inh  # Inhibitory rates

print(f"\nActivity shape: {rates_exc.shape}")
print(f"Time points: {rates_exc.shape[1]}")

# ============= ANALYSIS METRICS =============

print("\n" + "="*50)
print("COMPUTING DYNAMICS METRICS")
print("="*50)

# 1. MUTUAL INFORMATION between regions
print("\n1. Computing pairwise mutual information...")
n_nodes = rates_exc.shape[0]
MI_matrix = np.zeros((n_nodes, n_nodes))

# Compute MI between all pairs of regions
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        # Use last 50% of simulation (after transient)
        start_idx = rates_exc.shape[1] // 2
        x = rates_exc[i, start_idx:].flatten()
        y = rates_exc[j, start_idx:].flatten()
        
        # Mutual information calculation
        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]
        MI_matrix[i, j] = mi
        MI_matrix[j, i] = mi

print(f"Mean MI: {np.mean(MI_matrix[MI_matrix > 0]):.4f}")
print(f"Max MI: {np.max(MI_matrix):.4f}")

# 2. FUNCTIONAL CONNECTIVITY (Correlation)
print("\n2. Computing functional connectivity (Pearson correlation)...")
start_idx = rates_exc.shape[1] // 2
FC_matrix = np.corrcoef(rates_exc[:, start_idx:])
print(f"Mean FC: {np.mean(FC_matrix[np.triu_indices_from(FC_matrix, k=1)]):.4f}")

# 3. METASTABILITY (std of Kuramoto order parameter)
print("\n3. Computing metastability...")
# Calculate Kuramoto order parameter over time
phases = np.angle(signal.hilbert(rates_exc, axis=1))
order_param = np.abs(np.mean(np.exp(1j * phases), axis=0))
metastability = np.std(order_param[start_idx:])
print(f"Metastability: {metastability:.4f}")

# 4. GLOBAL SYNCHRONIZATION
print("\n4. Computing global synchronization...")
global_sync = np.mean(order_param[start_idx:])
print(f"Global synchronization: {global_sync:.4f}")

# 5. NODE-WISE STATISTICS
print("\n5. Computing node-wise statistics...")
mean_rates = np.mean(rates_exc[:, start_idx:], axis=1)
std_rates = np.std(rates_exc[:, start_idx:], axis=1)
cv_rates = std_rates / (mean_rates + 1e-10)  # Coefficient of variation

print(f"Mean firing rate across nodes: {np.mean(mean_rates):.3f} Hz")
print(f"Mean CV: {np.mean(cv_rates):.3f}")

# 6. INTEGRATION vs SEGREGATION
print("\n6. Computing integration and segregation metrics...")
# Integration: global efficiency of FC network
FC_pos = FC_matrix.copy()
FC_pos[FC_pos < 0] = 0
integration = np.mean(FC_pos)

# Segregation: modularity (simplified version using clustering coefficient)
segregation = np.mean(np.sum(FC_pos > 0.5, axis=1)) / n_nodes

print(f"Integration (mean FC): {integration:.4f}")
print(f"Segregation proxy: {segregation:.4f}")

# ============= VISUALIZATION =============
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

from scipy import signal

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Structural connectivity
ax = axes[0, 0]
im = ax.imshow(connectome, cmap='viridis', aspect='auto')
ax.set_title('Structural Connectivity (Input)')
ax.set_xlabel('Region')
ax.set_ylabel('Region')
plt.colorbar(im, ax=ax)

# Plot 2: Functional connectivity
ax = axes[0, 1]
im = ax.imshow(FC_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_title('Functional Connectivity (Pearson)')
ax.set_xlabel('Region')
ax.set_ylabel('Region')
plt.colorbar(im, ax=ax)

# Plot 3: Mutual Information matrix
ax = axes[0, 2]
im = ax.imshow(MI_matrix, cmap='hot', aspect='auto')
ax.set_title('Pairwise Mutual Information')
ax.set_xlabel('Region')
ax.set_ylabel('Region')
plt.colorbar(im, ax=ax)

# Plot 4: Example time series
ax = axes[1, 0]
time = np.arange(rates_exc.shape[1]) * model.params['dt'] / 1000  # Convert to seconds
for i in range(min(5, n_nodes)):
    ax.plot(time, rates_exc[i, :], alpha=0.7, label=f'Node {i}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Firing Rate (Hz)')
ax.set_title('Example Time Series (first 5 nodes)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Global synchronization over time
ax = axes[1, 1]
ax.plot(time[start_idx:], order_param[start_idx:], 'b-', alpha=0.7)
ax.axhline(global_sync, color='r', linestyle='--', label=f'Mean: {global_sync:.3f}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Kuramoto Order Parameter')
ax.set_title(f'Global Synchronization (Metastability: {metastability:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Node firing rate distribution
ax = axes[1, 2]
ax.hist(mean_rates, bins=20, alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('Mean Firing Rate (Hz)')
ax.set_ylabel('Number of Nodes')
ax.set_title('Distribution of Node Firing Rates')
ax.axvline(np.mean(mean_rates), color='r', linestyle='--', 
           label=f'Mean: {np.mean(mean_rates):.2f} Hz')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

from pathlib import Path
output_folder = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/experiments_neurolib")
plt.savefig(output_folder / 'connectome_dynamics_analysis.png', dpi=300, bbox_inches='tight')
print("\nFigure saved as 'connectome_dynamics_analysis.png'")
plt.show()

# ============= SUMMARY REPORT =============
print("\n" + "="*50)
print("SUMMARY REPORT")
print("="*50)

print(f"""
Network Properties:
- Number of nodes: {n_nodes}
- Structural density: {np.sum(connectome > 0) / (n_nodes * (n_nodes - 1)):.3f}

Dynamics Metrics:
- Mean Mutual Information: {np.mean(MI_matrix[MI_matrix > 0]):.4f}
- Mean Functional Connectivity: {np.mean(FC_matrix[np.triu_indices_from(FC_matrix, k=1)]):.4f}
- Global Synchronization: {global_sync:.4f}
- Metastability: {metastability:.4f}
- Mean Firing Rate: {np.mean(mean_rates):.3f} Hz
- Coefficient of Variation: {np.mean(cv_rates):.3f}
- Integration (mean FC): {integration:.4f}
- Segregation proxy: {segregation:.4f}

Interpretation:
- Higher MI indicates stronger information sharing between regions
- FC close to structural connectivity suggests structure drives function
- Metastability > 0.1 indicates rich dynamic repertoire
- Global sync close to 1 indicates high synchronization
""")

print("="*50)
print("Analysis complete!")