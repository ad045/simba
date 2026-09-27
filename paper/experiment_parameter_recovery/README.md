# Synthetic GNM Parameter-Recovery Experiment

Tests whether 16 network-distance measures can correctly identify which
generative parameter combination (η, γ) produced a given network, by
comparing generated networks against structural consensus networks.

---

## Overview

Five η-γ parameter combinations were chosen from the HCP Schaefer-100
energy/DeltaCon landscape:

| # | η | γ | Notes |
|---|---|---|-------|
| 0 | −6.9 | 1.11 | high distance-penalty, high topology-preference |
| 1 | 4.1 | 1.11 | low distance-penalty, high topology-preference |
| 2 | −6.9 | 0.01 | high distance-penalty, near-zero topology-preference |
| 3 | 4.1 | 0.01 | low distance-penalty, near-zero topology-preference |
| 4 | −3.734… | 0.5959 | DeltaCon optimum for HCP subject 105 |

For each combination:
- 10 **test networks** are generated (for parameter recovery)
- 100 **independent networks** are generated and collapsed into a single
  **structural consensus** (via `netneurotools.struct_consensus`)

The 50 test networks (10 × 5) are compared against all 5 consensus networks
using 16 distance measures. For each test network the measure with the
minimum distance to a consensus predicts the parameter combination.
Accuracy = fraction of correct predictions; precision = variability of
predictions in normalised parameter space.

---

## Pipeline

```
run_synthetic_gnm_generation.py   →   run_synthetic_gnm_comparison.py   →   run_synthetic_gnm_plots.py
```

Run in order:

```bash
python run_synthetic_gnm_generation.py
python run_synthetic_gnm_comparison.py
python run_synthetic_gnm_plots.py
```

All scripts are **idempotent** — they skip files that already exist on disk.

---

## Scripts

### `run_synthetic_gnm_generation.py`

Generates networks and consensus networks.

**Key configuration** (top of file):

| Variable | Value | Meaning |
|---|---|---|
| `PARAM_COMBOS` | 5 tuples | η-γ combinations to probe |
| `N_TEST` | 10 | test networks per combo |
| `N_CONSENSUS` | 100 | networks used to build consensus |
| `GENERATIVE_RULE_NAME` | `"MatchingIndex"` | GNM wiring rule |
| `LAMBDAH` | 1.0 | λ parameter |
| `DEVICE` | `"cpu"` | torch device |
| `SEED_PATH` | `data/preprocessed/seeds/mst_schaeffer.npy` | MST seed (99 edges, shape 1×100×100) |

**Edge count logic**: HCP consensus has 495 edges; MST seed contributes 99
(= n\_nodes − 1); GNM iterates for the remaining 396.

**Consensus construction**: `netneurotools.struct_consensus` with
hemisphere IDs [0×50, 1×50] and the HCP Schaefer-100 distance matrix,
then `binarize_network(retain=10)` to ~10 % density.

**Output layout**:
```
output/gnm/synthetic_parameter_recovery/
└── param_eta<η>_gamma<γ>/
    ├── generated_10/
    │   └── net_eta<η>_gamma<γ>_ruleMatchingIndex_id000..009.npy
    ├── consensus_100/
    │   └── net_eta<η>_gamma<γ>_ruleMatchingIndex_id000..099.npy
    └── consensus.npy    ← (100, 100) binary consensus
```

---

### `run_synthetic_gnm_comparison.py`

Cross-compares all 50 test networks against all 5 consensus networks using
each registered distance measure. Writes one CSV per measure.

**Adding a new measure**: import its evaluator class and add one line to
the `evaluators` dict in `main()`.

**Implemented measures** (16):

| Key | Class | Notes |
|---|---|---|
| `delta_con` | `DeltaConEvaluator` | Fast belief propagation (Matusita distance) |
| `frobenius` | `FrobeniusEvaluator` | Frobenius norm of adjacency matrix difference |
| `portrait` | `PortraitDivergence` | Graph portrait divergence |
| `hamming` | `HammingEvaluator` | Hamming distance on adjacency matrices |
| `f1` | `F1Evaluator` | F1 score on edge sets |
| `jaccard` | `JaccardEvaluator` | Jaccard similarity of edge sets |
| `spectral_distance_norm_laplacian` | `SpectralDistanceEvaluator` | Normalised Laplacian eigenvalue distance |
| `spectral_distance_adjacency` | `SpectralDistanceEvaluator` | Adjacency eigenvalue distance |
| `communicability_corr` | `CommunicabilityCorrEvaluator` | Communicability matrix correlation |
| `communicability_jsd` | `CommunicabilityJSDEvaluator` | Jensen-Shannon divergence of communicability |
| `network_mutual_information` | `NetworkMutualInformationEvaluator` | NMI between adjacency structures |
| `dc_network_mutual_information` | `DCNetworkMutualInformationEvaluator` | Degree-corrected NMI |
| `net_simile` | `NetrdEvaluator(method=…)` | NetRD Net-SIMILE |
| `resistance` | `NetrdEvaluator(method=…)` | NetRD resistance distance |
| `netrd_non_backtracking_spectral` | `NetrdEvaluator(method=…)` | Non-backtracking spectral distance |
| `energy` | `EnergyEvaluator` | GNM energy (MaxCriteria: degree, clustering, edge-length, betweenness KS) |

**Output**:
```
output/gnm/synthetic_parameter_recovery/comparison_results/
└── distances_<measure>.csv
```

**CSV columns**: `true_eta`, `true_gamma`, `true_combo_idx`, `network_id`,
`predicted_combo_idx`, `predicted_eta`, `predicted_gamma`, `correct`,
`dist_to_combo_0` … `dist_to_combo_4`

---

### `run_synthetic_gnm_plots.py`

Reads all `distances_*.csv` files and produces two PDF figures.

**Key configuration**:

| Variable | Default | Meaning |
|---|---|---|
| `ERROR_METRIC` | `"mae"` | `"mse"` or `"mae"` for the accuracy panel |
| `ETA_RANGE` | `(−8, 3)` | normalisation bounds for η |
| `GAMMA_RANGE` | `(−0.1, 1)` | normalisation bounds for γ |
| `SCATTER_NCOLS` | 4 | columns in the scatter grid |

**Accuracy** (MAE/MSE): mean Euclidean error (or squared) between true and
predicted (η, γ) in normalised [0,1]×[0,1] space.

**Precision**: per-combo mean of std(pred\_η\_norm) and std(pred\_γ\_norm)
across the 10 test networks; lower = more consistent predictions.

**Output**:
```
output/gnm/synthetic_parameter_recovery/plots/
├── parameter_recovery_scatter.pdf   ← 4×4 grid, one panel per measure
└── precision_accuracy.pdf           ← horizontal bar chart (accuracy) + heatmap (precision)
```

---

## Data dependencies

| File | Role |
|---|---|
| `data/preprocessed/hcp_schaefer_100_dataset/01_connectomes/01_consensus_bin_density_10_percent_100.npy` | Determines target edge count (495) |
| `data/preprocessed/hcp_schaefer_100_dataset/02_distance_matrices/distance_matrix_100.npy` | Spatial distance for GNM generation and energy criteria |
| `data/preprocessed/seeds/mst_schaeffer.npy` | MST seed graph (1×100×100, 99 edges) |

---

## TODOs / Ideas for Further Improvement

### High priority

- [ ] **Parallelise generation** — `run_synthetic_gnm_generation.py` runs
  networks sequentially. With 5×110 = 550 networks this is the main
  bottleneck. Use `joblib.Parallel` with the same pattern as
  `gnm_and_esn_orchestrator.py`.

- [ ] **Increase N_TEST** — 10 networks per combo gives noisy precision
  estimates. 50–100 would make variability statistics more reliable.

- [ ] **Continuous landscape** — instead of 5 discrete consensus targets,
  sweep a coarse η-γ grid (e.g. 10×10) and compare each test network
  against all grid-point consensus networks. This gives a proper
  "landscape" and allows gradient-based localisation.

- [ ] **Distance measure calibration** — distances are currently compared
  on their raw scale. Normalise each measure to [0, 1] (or z-score across
  the 5 distances per network) before taking the argmin, so that
  high-magnitude measures don't dominate.

### Medium priority

- [ ] **Weighted consensus alternative** — `struct_consensus` applies
  anatomical hemisphere constraints designed for real brains. For purely
  synthetic GNMs a simple mean or majority-vote consensus might be more
  appropriate. Compare results.

- [ ] **More generative rules** — currently only `MatchingIndex`. Run the
  same experiment with `Neighbours`, `DegreeProduct`, `ClusteringAverage`,
  etc. to test whether the distance measures are also rule-agnostic.

- [ ] **Confidence / soft predictions** — instead of argmin (hard
  assignment), compute softmax over negative distances and report entropy
  as an additional uncertainty measure.

- [ ] **Cross-dataset validation** — repeat with Suarez MaMI (50 nodes) or
  Shafiei (68 nodes) to test whether measure rankings are consistent across
  brain parcellations.

- [ ] **Bootstrap confidence intervals** — resample the 10 test networks
  with replacement to get error bars on MSE/MAE and precision estimates.

### Nice-to-haves

- [ ] **Interactive HTML figure** — use Plotly to make the scatter grid
  zoomable and hoverable (show network ID on hover).

- [ ] **Confusion matrix panel** — add an optional third panel to
  `precision_accuracy.pdf` showing the full confusion matrix per measure
  as a grid of heatmaps.

- [ ] **Distance landscape surface plot** — for a test network at combo i,
  plot the 5 distances as a surface/contour in η-γ space (interpolated)
  to visually show how "peaked" the landscape is.

- [ ] **Shared config file** — `PARAM_COMBOS`, `ETA_RANGE`, `GAMMA_RANGE`,
  `N_TEST`, `N_CONSENSUS`, etc. are duplicated across all three scripts.
  Extract to a single `config_synthetic_experiment.py`.

- [ ] **Logging** — replace `print()` statements with Python `logging` so
  output can be redirected to a file without modifying code.
