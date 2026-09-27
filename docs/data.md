# Data

The package downloads one bundle (about 30 MB zipped) on first use, into
`~/netdistancebench_data/netdistancebench-data/`. Set `NETDISTANCEBENCH_HOME`
to download elsewhere, or `NETDISTANCEBENCH_DATA` to use an unpacked copy.
The bundle is also attached to the GitHub release `data-v1` for direct download.

## Networks

```python
import netdistancebench as ndb

nets, meta = ndb.load_networks()              # (25000, 100, 100) bool, and eta/gamma/replicate
wide = ndb.load_networks("recovery_wide")     # dict: grid, grid_members, targets, *_eta, *_gamma
window = ndb.load_networks("recovery_window")
```

| Set | Content |
|---|---|
| `morphospace` | The main sweep: 50 x 50 grid over $\eta \in [-8, 3]$, $\gamma \in [-0.1, 1]$, 10 replicates, matching-index rule, seeded with the minimum spanning tree of the Schaefer-100 distance matrix; 99 seed edges plus 396 added = 495 edges (10% density, matched to the consensus). |
| `recovery_wide` | 100 test networks at uniformly drawn parameters over the whole morphospace; a 10 x 10 recovery grid, each point a consensus of 30 networks (`grid_members`). |
| `recovery_window` | 100 test networks inside the plausible window; a 10 x 10 grid of 20-network consensuses. |

Grid points are ordered gamma-outer, eta-inner: index `i` is
`(eta[i % 10], gamma[i // 10])`.

## Tables

```python
ndb.load_table("landscapes")    # 25,000 rows: eta, gamma, replicate, one column per measure
```

| Name | Content |
|---|---|
| `landscapes` | Raw output of all 16 measures for every morphospace network against the consensus, plus the four KS statistics inside the energy. Same row order as `load_networks()`. |
| `timing_ms` | Runtime of each published measure per comparison, the reference measurement quoted in the paper. |
| `degeneration` | The 200 x 101 rewiring trajectories (`hamming` = effective degeneration). |
| `noise_tolerance_draws`, `noise_tolerance_results`, `effective_degeneration` | Drift of the best fit under rewiring of the reference, per draw and summarised. |
| `recovery_wide`, `recovery_window` | Per measure and test network: true parameters, recovered cell and the distance to all 100 grid points. |
| `real_vs_artificial` | AUCs of the eight selected measures. |
| `experiments/<name>/<table>` | Topographic analyses: hub topography, the five-map battery, the structural gradient. |

Values are the measures' raw output; similarities are not flipped.

## Geometry

```python
ndb.distance_matrix()       # (100, 100) Euclidean distances between Schaefer-100 centroids, mm
ndb.empirical_summary()     # group-level statistics of the empirical connectomes
```
