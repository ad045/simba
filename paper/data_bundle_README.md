# simba_networks data bundle

Data behind *How Similar Are Two Brains? A Comprehensive Benchmark of Brain
Network Similarity Measures* (Dendorfer, Luppi, Poli, Mousley, Astle, Fakhar).
Built by `paper/make_publication_data.py`; read by the `simba_networks`
package, which downloads and unpacks it on first use.

Networks are binary, 100 x 100 (Schaefer-100), bit-packed along the last axis.
Unpack with `np.unpackbits(x, axis=-1, count=100)`, or use
`simba_networks.load_networks()`.

| File | Content |
|---|---|
| `networks/morphospace.npz` | `networks` (25,000): the GNM sweep, 50 eta x 50 gamma x 10 replicates, matching-index rule, eta in [-8, 3], gamma in [-0.1, 1], 495 edges each. `eta`, `gamma`, `replicate` per network. Row order = `tables/landscapes.csv.gz`. |
| `networks/recovery_wide.npz` | Parameter recovery over the whole morphospace: `grid` (100 grid-point consensuses, 10 x 10, gamma-outer / eta-inner), `grid_members` (the 30 networks behind each), `targets` (100 test networks at uniformly drawn parameters), and their `*_eta` / `*_gamma`. |
| `networks/recovery_window.npz` | Same inside the plausible window (20 members per grid point). |
| `tables/landscapes.csv.gz` | Raw value of each of the 16 measures for each of the 25,000 networks against the empirical consensus, plus the four KS statistics inside the energy. |
| `tables/timing_ms.csv.gz` | Runtime per comparison (ms), the single reference measurement quoted in the paper. |
| `tables/degeneration.csv.gz` | Measure value along 200 progressive-rewiring trajectories of the consensus (101 steps each); `hamming` is the effective degeneration. |
| `tables/noise_tolerance_*.csv*`, `tables/effective_degeneration.csv` | Drift of the best-fitting combination when the reference is rewired (per draw, and summarised). |
| `tables/recovery_wide.csv.gz`, `tables/recovery_window.csv.gz` | Per measure and test network: true parameters, recovered grid cell, and the distance to every grid point (`dist_to_grid_*`). |
| `tables/real_vs_artificial.csv` | AUCs separating real subjects from GNMs. |
| `tables/experiments/` | Result tables of the topographic analyses (hub topography, five-map battery, structural gradient). |
| `reference/distance_matrix.npy` | Euclidean distance (mm) between Schaefer-100 parcel centroids. |
| `reference/empirical_summary.json` | Group-level statistics of the empirical connectomes (connection length, long-range fraction, degree SD). |

Not included: the empirical connectomes (HCP data use terms). The package
explains where to drop in your own copy of the consensus.
