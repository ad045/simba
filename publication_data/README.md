# publication_data

The shareable output of the analysis. Built by `make_publication_data.py` at the
repository root, which reads the local `output/` run tree and copies out the part
that backs the manuscript.

```bash
python make_publication_data.py            # rebuild
python make_publication_data.py --check    # verify the GNM pack round-trips
```

---

## `gnms/` - the generated networks

| File | Contents |
|---|---|
| `gnms_hcp_25000.npz` | The **25,000 networks of the main sweep**. 50 x 50 grid over eta in [-8, 3] and gamma in [-0.1, 1], matching-index rule, MST seed, 10 replicates per cell. |
| `gnms_recovery_wide.npz` | 3,160 networks of the wide parameter-recovery experiment (grid consensus + ground-truth test networks). |
| `gnms_recovery_window.npz` | 2,200 networks of the plausible-window recovery experiment. |

The networks are binary, so they are stored **bit-packed**: 977 MB of individual
`.npy` files become 17 MB, losslessly. `make_publication_data.py --check`
verifies this by re-reading random networks from the original tree and comparing
byte for byte.

```python
from make_publication_data import load_networks

stack, meta = load_networks("publication_data/gnms/gnms_hcp_25000.npz")
stack.shape          # (25000, 100, 100), float32, values in {0, 1}
meta["eta"], meta["gamma"], meta["id"]   # parameters of each network
meta["filenames"]    # original filename, so results tables can be joined on it
```

Or without importing anything from this repo:

```python
import numpy as np
z = np.load("publication_data/gnms/gnms_hcp_25000.npz")
n = int(np.prod(z["shape"]))
stack = np.unpackbits(z["packed"])[:n].reshape(tuple(z["shape"]))
```

## `benchmark_results/` - the numbers

Every result table from the main sweep, gzipped (`pandas.read_csv` reads `.gz`
directly). The two that matter most, one file per measure:

- `summary_indiv_<measure>_for_exp_105_distance_metrics_mst_animal_0.csv.gz` -
  25,000 rows: `network_index, filename, eta, gamma, id, <measure value>`.
  **This is the distance landscape** of that measure.
- `timing_<measure>_for_exp_105_distance_metrics_mst_animal_0.csv.gz` -
  the wall-clock cost of each of those 25,000 comparisons.

Also here: `all_metrics_..._updated.csv.gz` (per-network topology metrics, the
biological-plausibility axis), `chaos_analysis/` (degeneration curves behind
figure 3 panels D and F), and the per-analysis tables written by
`method_evaluation/` (`correlation_matrix.csv`, `pca_loadings.csv`,
`stability_metrics.csv`, `inter_method_agreement.csv`).

Values are **raw comparer output**. Five of the 16 measures are similarities, not
distances, and must be flipped before any `argmin`: see `IS_SIMILARITY` and
`to_distance()` in `experiments_config.py`.

## `parameter_recovery/`, `experiment_results/`

Result tables of the recovery experiments and of analyses S2, S3, S6, S6b, S6c.

## `figures/`

The figure PDFs as the scripts wrote them, grouped by the output folder each
script writes to. The manuscript renames them on copy; the mapping is in the
root `README.md`.

## `empirical_derived/` - **not tracked in git**

Arrays computed from the empirical connectomes: leave-one-out consensus networks
and nodal reference maps (degree, clustering, betweenness, connection length,
principal gradient, S-A axis). They are staged here so nothing is silently lost,
but they are gitignored, because redistributing them is a question about the HCP
data use terms rather than a question about this repository. See that folder's
README.
