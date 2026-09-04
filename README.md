# How Similar Are Two Brains? A Comprehensive Benchmark of Brain Network Similarity Measures

Code and generated data for the manuscript by Adrian Dendorfer, Andrea Luppi,
Francesco Poli, Alexa Mousley, Duncan Astle and Kayson Fakhar.

The paper benchmarks **16 network distance measures** for the task of comparing
generative network model (GNM) output against an empirical connectome. It scores
them on five axes - agreement, biological plausibility, computational efficiency,
sensitivity and robustness, and accuracy (synthetic parameter recovery) - and
carries **8 measures** through to the main analyses.

---

## Data availability

| | |
|---|---|
| **Shared here** | All 25,000 generated networks, every benchmark result table, every figure. See [`publication_data/`](publication_data/). |
| **Not shared** | The empirical connectomes (HCP S900), which are governed by the HCP data use terms. |
| **Shared instead** | The complete preprocessing code that turns the raw HCP release into the consensus connectome used as the reference - see [Stage 0](#stage-0-empirical-preprocessing-code-only). |

Anything computed *from* the empirical connectomes (leave-one-out consensus
networks, nodal reference maps) is staged into `publication_data/empirical_derived/`,
which is **gitignored**. Read that folder's README before sharing any of it.

---

## Setup

```bash
conda env create -f environment.yml    # creates the `ma_thesis` env
conda activate ma_thesis
```

Two dependencies are local forks and are **not** on PyPI in the form used here.
Clone them next to this repository and install both editable:

```bash
# patched generative network model library (adds seed-adjacency-matrix support)
git clone https://github.com/ad045/GenerativeNetworkModels.git
pip install -e GenerativeNetworkModels          # pinned at commit e3bc425

# plotting helpers used by a few figure scripts
git clone https://github.com/kuffmode/Pyvizman.git
pip install -e Pyvizman
```

`environment.yml` is an exact export of the macOS/arm64 environment the results
were produced in, build strings included. On another platform, install from the
`pip:` section and let conda resolve the rest. The versions that matter:
Python 3.13.7, numpy 2.3.5, scipy 1.17.0, networkx 3.6.1, netrd 0.3.0,
netneurotools 0.2.5, torch 2.8.0.

**Run every script from the repository root.** Paths are anchored there
(`experiments_config.ROOT_DIR`), and the `data/` and `output/` symlinks live there.

---

## The 16 measures

| Family | Measures |
|---|---|
| Matrix-based | Frobenius, Hamming, Jaccard, F1 |
| Information-theoretic | NMI, DC-NMI |
| Spectral | adjacency, normalized Laplacian, non-backtracking |
| Kernel / feature-based | portrait divergence, NetSimile, DeltaCon |
| Communicability-based | communicability correlation, communicability JSD, resistance distance |
| Baseline | KS-based energy (Betzel et al., 2016) |

The **8 selected** after the plausibility and redundancy filter: Frobenius,
DeltaCon, spectral (non-backtracking), spectral (adjacency), communicability
correlation, portrait divergence, NetSimile, KS-based energy. They are defined
once, in [`experiments_config.py`](experiments_config.py), together with their
orientation (which are similarities and must be flipped before any `argmin`),
display names and colours. That file is the single source of truth - start there.

---

## The pipeline, end to end

### Stage 0: empirical preprocessing (code only)

Turns the raw HCP release into the binarized 10 %-density consensus connectome
that every later stage compares against.

| File | Does |
|---|---|
| `src/preprocessing/02_preprocessing_pipeline_hcp_schaefer_100.ipynb` | HCP S900, 100 unrelated subjects, Schaefer-100 parcellation, QSDR + deterministic tractography, binarized at 10 % density; consensus via `netneurotools.struct_consensus`. **This is the reference connectome of the paper.** |
| `src/preprocessing/08_preprocessing_lexis_data.ipynb` (+ `09_`, `10_`) | Second dataset (HCP Young Adult, 1065 subjects), used in the appendix only, to check that the landscapes survive a different seed and sample. |
| `src/preprocessing/get_distance_matrix.py` | Euclidean distance matrix between parcel centroids (the GNM's distance term). |
| `src/preprocessing/threshold_to_density.py`, `utils.py`, `getting_seeds/` | Thresholding, binarizing, MST seed construction. |

Running these requires the raw HCP data, which we cannot redistribute. Their
output - and only their output - is what the rest of the pipeline consumes.

### Stage 1: generate the GNMs

50 x 50 grid over eta in [-8, 3] and gamma in [-0.1, 1], matching-index rule,
MST seed from the Schaefer-100 distance matrix, 10 replicates per cell =
**25,000 networks**.

```bash
python pipeline_gnms_and_benchmarking/run_experiment.py configs/config_gnm_run_hcp.yaml
# or the sweep wrapper, which also merges the per-run CSVs:
bash pipeline_gnms_and_benchmarking/run_gnm.sh
```

Output lands in `output/gnm/hcp_schaefer_100_dataset/<experiment name>/`. The run
behind the manuscript is `105_distance_metrics_mst_animal_0`; its networks are
the ones shipped as `publication_data/gnms/gnms_hcp_25000.npz`.

### Stage 2: compare every generated network to the empirical reference

```bash
python pipeline_gnms_and_benchmarking/run_connectome_comparisons.py
```

Each measure is a `NetworkEvaluator` subclass in `src/comparing_connectomes/`
(`base_comparer.py` is the interface; ~20 implementations sit beside it). The
script writes, per measure, `summary_indiv_<measure>_for_exp_<run>.csv`
(25,000 distances) and `timing_<measure>_for_exp_<run>.csv` (25,000 wall-clock
timings) - these two families of file are the raw material of the whole paper.

> The list of measures to run is a comment-toggled Python list near the bottom
> of that script rather than a CLI flag. Uncomment the ones you want.

### Stage 3: per-network topology metrics

Static, dynamic and computational graph metrics for all 25,000 networks
(`src/analysis/`), producing
`all_metrics_for_<run>_updated.csv`. These feed the biological-plausibility axis.

### Stage 4: the five evaluation axes

| Analysis | Script | Question |
|---|---|---|
| Landscapes, agreement, timing | `visualization/8_7_manuscript_16_measures.ipynb` | Do the 16 measures agree on where the best fit lies, and what do they cost? |
| Selected 8, total variation, PCA | `visualization/8_9_manuscript_selected_8_measures.ipynb` | How do the 8 differ in the networks they select? |
| Sensitivity (iSNR, MAE, CV) | `visualization/9_between_and_withhin_variance.ipynb` | Does a measure's landscape carry signal above its own replicate noise? |
| Degeneration panels (D, F) | `visualization/run_degeneration_effective_panels.py` | Linearity of each measure against *effective* degeneration. |
| **S2** robustness | `experiment_rewiring_robustness/run_rewiring_robustness.py` | If the reference had a few edges moved, does the recovered (eta, gamma) drift gracefully or fall off a cliff? |
| **S3** real vs artificial | `experiment_real_vs_artificial/run_real_vs_artificial.py` | Model-free check: does the measure rank real subjects closer to the consensus than GNMs? |
| **S6** structural gradient | `experiment_structural_gradient/run_structural_gradient.py` | Do best-fit networks reproduce the principal gradient topography? |
| **S6b** topographic battery | `experiment_topographic_plausibility/run_topographic_plausibility.py` | Same, over five nodal maps, against an achievable ceiling. |
| **S6c** hub topography | `experiment_hub_topography/run_hub_topography.py` | Nodal-degree correlation - are the hubs in the right *places*? |
| **S4** axis reconciliation | `experiment_decision_figure/reconcile_axes.py` | Sanity probe: recompute each axis scalar and print it next to the published number. Writes nothing. |

S6 -> S6b -> S6c is a deliberate progression: each supersedes the previous one,
and the earlier two are kept because the paper argues *why* the simpler versions
are not enough. S6c is the headline topographic criterion.

### Stage 5: synthetic parameter recovery (accuracy)

Generate networks at known (eta, gamma), then ask each measure to recover them.

```bash
# wide regime: 6 widely-spread ground-truth combinations, 10 networks each
python experiment_parameter_recovery/run_synthetic_gnm_generation.py
python experiment_parameter_recovery/run_synthetic_gnm_comparison.py
python experiment_parameter_recovery/run_grid_and_variance_paper_plot.py

# window regime: 100 ground truths drawn from the biologically plausible window
python experiment_parameter_recovery_fine/derive_window.py     # derives the window from the morphospace
python experiment_parameter_recovery_fine/run_synthetic_gnm_fine.py
python experiment_parameter_recovery_fine/run_recovery_main_figures.py
```

The window is **derived, not hand-set**: pool the 100 best-fitting cells of each
of the 8 measures (800 points) and take the central 50 % per axis. See
`derive_window.py` and the comment block in `experiments_config.py`.

`rescore_recovery_predictions.py` re-scores both experiments after the August 2026
measure-orientation fix (Jaccard, F1, communicability correlation, NMI and DC-NMI
are similarities and had been `argmin`-ed as distances). Both sets of numbers in
the manuscript are post-fix.

`experiment_parameter_recovery/README.md` documents the recovery design in detail.

---

## Which script made which figure

| Manuscript figure | Made by |
|---|---|
| `fig_0_overview.pdf` | Hand-drawn schematic, not generated from data. |
| `fig_huge_02_landscapes_corr_network_timing_plausible.pdf` | `visualization/8_7_manuscript_16_measures.ipynb` |
| `fig_3_total_variation_degree_distance_2.pdf` | `visualization/8_9_manuscript_selected_8_measures.ipynb`; panels D and F from `visualization/run_degeneration_effective_panels.py` |
| `fig_5_recovery_error_2.pdf` | `visualization/build_fig5_recovery.py` (whole figure, all five panels) |
| `legend_8_measures.pdf` | `make_legend_8_measures.py` |
| appendix `grid_and_variance.pdf` | `experiment_parameter_recovery/run_grid_and_variance_paper_plot.py` |
| appendix `drift_of_recovered_parameters.pdf` | `experiment_rewiring_robustness/run_rewiring_robustness.py` |
| appendix `hub_topography_sa_axis.pdf`, `hubs_landscapes_violins_2.pdf` | `experiment_hub_topography/run_hub_topography.py` |
| appendix `fig_pca_landscapes_loadings.pdf` | `visualization/8_9_manuscript_selected_8_measures.ipynb` |
| appendix `components_of_ks_energy_grayscale.pdf` | `visualization/8_7_2_ks_energy_contributors.ipynb` |
| appendix `similarity_methods_timing_comparison.pdf`, `method_legend.pdf` | `visualization/8_7_manuscript_16_measures.ipynb` |
| appendix `lexis_data_figure_1_landscape_and_connectomes_*.pdf` | `visualization/21_2_fig_1_six_example_gen_connectomes_lexis_data.ipynb` |
| appendix `explaining_measures/*.pdf` (8 files) | Hand-drawn schematics, not generated from data. |
| Figure 1 (landscape + six example connectomes) | `visualization/21_2_fig_1_six_example_gen_connectomes_hcp_data.ipynb` |

Figure files are renamed when they are copied into the manuscript repository, so
the names above are the manuscript's, not the script's. The script's own output
is in `publication_data/figures/` under the folder each script writes to.

---

## Repository layout

```
experiments_config.py         single source of truth: paths, the 8 measures,
                              orientation, grid geometry, filename conventions
make_publication_data.py      builds publication_data/ from the local run tree
make_legend_8_measures.py     shared figure legend
rescore_recovery_predictions.py

configs/                      YAML configs for the GNM sweeps
src/
  comparing_connectomes/      NetworkEvaluator + ~20 distance measures  <- the core
  GNMs/, pipeline/            GNM generation and orchestration
  analysis/                   per-network topology metrics
  preprocessing/              empirical connectome preprocessing (Stage 0)
  config/, utils/             paths, YAML loading, filename parsing
  ESNs/                       reservoir-computing metrics; see "rough edges"
pipeline_gnms_and_benchmarking/   Stage 1 and Stage 2 runners
experiment_*/                 the individual analyses (S2, S3, S4, S6, S6b, S6c,
                              and the two parameter-recovery regimes)
visualization/                the notebooks and scripts that draw the figures
publication_data/             the shareable data (see its README)
figures/                      figures kept alongside the code
archive/                      everything not part of this manuscript (see below)

data/   -> symlink into the local run tree (empirical data; not in git)
output/ -> symlink into the local run tree (~225 GB of run history; not in git)
```

`archive/` holds work that is not part of this paper but was part of getting
here: the cross-species (MaMI) and other-dataset preprocessing, the exploratory
notebooks that preceded the manuscript figures, superseded drafts, and an
unrelated autoresearch side project. Nothing there is needed to reproduce the
paper; it is kept so the path taken stays visible. Deleting it is a one-liner
(`git rm -r archive/`).

---

## Rough edges, stated plainly

These are real and known; none of them affect the published numbers.

- **`src/ESNs/` is not used by this paper.** It implements reservoir-computing
  memory capacity, which belongs to a different project. It stays because
  `src/pipeline/gnm_and_esn_orchestrator.py` imports it at module level, so the
  Stage 1 generator will not import without it. Untangling that import is the
  clean fix.
- **Measure selection in `run_connectome_comparisons.py` is a comment-toggled
  list**, not a CLI argument.
- **`configs/config_gnm_run_hcp.yaml` is not the exact config of the published
  sweep.** Its `name` and eta/gamma ranges were edited after that run. The grid
  actually used is the one recorded in `experiments_config.py`
  (eta in [-8, 3], gamma in [-0.1, 1], 50 x 50, 10 replicates).
- **Two dataset branches still carry absolute paths** into a sibling project
  (`run_connectome_comparisons_null_model.py`, `run_connectome_targeted_removal.py`).
  They are only reached for datasets not shipped here; the HCP branch is
  repo-relative.
- The two 41 GB intermediate CSVs in the run tree are not shipped. They are
  regenerable by joining the `summary_indiv_*` files with `all_metrics_*`.
