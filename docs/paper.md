# Reproducing the paper

The `paper/` folder of the repository holds the full research pipeline: the code
that generated the networks, scored the 16 measures, ran every analysis and drew
every figure. The package covers scoring a new measure; this page is for
re-running the paper itself, and for finding the code behind any section,
figure or number of it.

## From the paper to the code

### Measure names

The paper uses display names; the code uses keys (`ndb.NAMES` maps one to the
other). The eight selected measures are marked.

| Paper | Code key | |
|---|---|---|
| Frobenius | `frobenius` | selected |
| Hamming | `hamming` | |
| Jaccard | `jaccard` | similarity |
| F1 | `f1` | similarity |
| NMI | `network_mutual_information` | similarity |
| DC-NMI | `dc_network_mutual_information` | similarity |
| Spectral (adjacency) | `spectral_distance_adjacency` | selected |
| Spectral (normalized Laplacian) | `spectral_distance_norm_laplacian` | |
| Spectral (non-backtracking) | `netrd_non_backtracking_spectral` | selected |
| Portrait divergence | `portrait` | selected |
| NetSimile | `net_simile` | selected |
| DeltaCon | `delta_con` | selected |
| Communicability correlation | `communicability_corr` | selected, similarity |
| Communicability JSD | `communicability_jsd` | |
| Resistance distance | `resistance` | |
| KS-based energy | `energy` | selected |

The five similarities (higher = more alike) are negated before any ranking:
`experiments_config.to_distance` in the pipeline, `SIMILARITIES` in the package.
The pipeline computed the measures with the classes in
`paper/src/comparing_connectomes/`; `netdistancebench/measures.py` holds
torch-free ports that reproduce their output (checked by `tests/`).

### Materials and Methods

| Methods section | Code (under `paper/`) |
|---|---|
| Empirical Consensus Connectome | `src/preprocessing/02_preprocessing_pipeline_hcp_schaefer_100.ipynb`, `src/preprocessing/get_distance_matrix.py` |
| Generative Network Model | `pipeline_gnms_and_benchmarking/generate_morphospace.py`; grid and seed in `experiments_config.py` |
| Distance measures, runtime analysis | `pipeline_gnms_and_benchmarking/score_morphospace.py`, `src/comparing_connectomes/` |
| Principal Component Analysis of the Landscapes | `experiment_decision_figure/run_pca_eight_measures.py` |
| Comparison with Empirical Connectome Properties | `visualization/run_8_9_2_minima_from_average_landscape.py` (degree, connection length), `visualization/run_cost_hemisphere_landscapes.py` (long-range, interhemispheric) |
| Hub Topography | `experiment_hub_topography/run_hub_topography.py` |
| Synthetic Parameter Recovery | `experiment_parameter_recovery_fine/run_wide_uniform_recovery.py` (whole morphospace), `derive_window.py` + `run_synthetic_gnm_fine.py` (plausible window) |
| Real Versus Synthetic Connectomes | `experiment_real_vs_artificial/run_real_vs_artificial.py` |
| Uncertainty-Measurements (iSNR) | `visualization/run_isnr_consistent_panel.py` |
| Degeneration Analysis (MAE, CV) | `pipeline_gnms_and_benchmarking/run_connectome_comparisons_null_model.py`, `visualization/run_degeneration_effective_panels.py`, `visualization/run_9_between_and_withhin_variance.py` |
| Edge Importance | `pipeline_gnms_and_benchmarking/run_connectome_targeted_removal.py` |
| Robustness of the Recovered Parameters to Reference Noise ($N^*$) | `experiment_rewiring_robustness/run_rewiring_robustness.py`, `compute_effective_degeneration.py` |
| Summary comparison of the selected measures (radar) | `experiment_decision_figure/build_radar_five_axes.py` |

Every per-measure number of the Results can also be recomputed without the
pipeline, from the shipped tables: `netdistancebench.published_readouts()`.

### Figures

The scripts write each figure's panels; most figures were then laid out in a
vector editor. `visualization/build_composites.py`, `visualization/build_fig5_recovery.py`
and `visualization/patch_figure4_panels.py` rebuild the layouts from the panels,
so a re-run reproduces the content of every data figure. File names below are
those in `figures/manuscript/` (see Setup).

| Figure | What | Produced by (under `paper/`) |
|---|---|---|
| 1 | Study overview | Schematic composite. B: `visualization/run_21_2_fig_1_six_example_gen_connectomes_hcp_data.py`; D: `experiment_decision_figure/build_radar_five_axes.py` |
| 2 | Mechanics of DeltaCon | Schematic on toy graphs, not produced by this repository |
| 3 | The 16 measures: landscapes, correlations, runtime, plausibility | `visualization/run_8_7_manuscript_16_measures.py` (panels; `run_fig2_panels.py` is a faster subset), assembled by `build_composites.py --only fig2` |
| 4 | Precision and reaction to perturbations | A-C: `visualization/run_8_9_2_minima_from_average_landscape.py`; D, F: `run_degeneration_effective_panels.py`; E: `run_8_9_manuscript_selected_8_measures.py`; G: `run_9_between_and_withhin_variance.py`; H: `run_isnr_consistent_panel.py`; layout: `patch_figure4_panels.py` |
| 5 | Parameter recovery and real vs synthetic | `visualization/build_fig5_recovery.py` (one script, all panels) |
| S1-S7 | Mechanics of the other measures | Schematics, not produced by this repository |
| S8-S10 | Landscapes on the second dataset | `visualization/21_2_fig_1_six_example_gen_connectomes_lexis_data.ipynb` |
| S11 | Components of the KS-based energy | `visualization/run_8_7_2_ks_energy_contributors.py` |
| S12 | PCA of the eight landscapes | `experiment_decision_figure/run_pca_eight_measures.py`, assembled by `build_composites.py --only pca` |
| S13 | Long-range and interhemispheric wiring | `visualization/run_cost_hemisphere_landscapes.py` |
| S14 | Hub topography | `experiment_hub_topography/run_hub_topography.py`, assembled by `build_composites.py --only hubs` |
| S15, Table S2 | Runtime of all 16 measures | `visualization/run_8_7_manuscript_16_measures.py` (prints Table S2), on the reference timing run (see Notes) |
| S16 | Recovered locations, whole morphospace | `experiment_parameter_recovery_fine/build_wide_uniform_scatter.py` |
| S17 | Drift of the recovered parameters under reference noise | `experiment_rewiring_robustness/run_rewiring_robustness.py --stage plot` |
| Table S1 | The 16 measures | `netdistancebench/measures.py` |

`python run_all_figures.py` runs all of the above in dependency order.
Some figure scripts are named after an earlier figure numbering
(`run_fig2_panels.py` draws Figure 3); the table above is the current mapping.

## Setup

```bash
git clone https://github.com/ad045/14_4D_benchmarking
cd 14_4D_benchmarking/paper
conda env create -f environment.yml      # creates the `ma_thesis` env
conda activate ma_thesis

# patched generative network model library (adds seed-adjacency-matrix support)
git clone https://github.com/ad045/GenerativeNetworkModels.git ../../GenerativeNetworkModels
pip install -e ../../GenerativeNetworkModels     # pinned at commit e3bc425
# plotting helpers used by some figure scripts
git clone https://github.com/kuffmode/Pyvizman.git ../../Pyvizman
pip install -e ../../Pyvizman
```

`environment.yml` is an exact export of the macOS/arm64 environment the results
were produced in. On another platform, install the `pip:` section and let conda
resolve the rest. The versions that matter: numpy, scipy, networkx 3.6 (its
random number stream fixes the rewiring trajectories), netrd 0.3.0,
netneurotools 0.2.5, torch 2.8.

**Run every script from `paper/`.** Paths are relative to it. The scripts read
and write two folders there, `data/` (empirical inputs) and `output/` (every run
and result); create them, or link them to wherever your copy lives. Stage 0
writes the empirical inputs to the places the rest of the pipeline reads them:

```
data/preprocessed/
    seeds/mst_schaeffer.npy                         MST seed (99 edges)
    hcp_schaefer_100_dataset/
        01_connectomes/
            01_consensus_bin_density_10_percent_100.npy   the consensus (fitting target)
            00_connectomes_density10.npy                   the 100 individuals
        02_distance_matrices/distance_matrix_100.npy     centroid distances, mm
        03_brain_maps/                                   S-A axis cache (hub topography)
```

`experiments_config.py` is the single source of truth for the eight selected
measures, their orientation, names, colours, the grids and the file layout.

## Pipeline

| Stage | Run | Produces |
|---|---|---|
| 0. Empirical preprocessing (needs HCP access) | `src/preprocessing/02_preprocessing_pipeline_hcp_schaefer_100.ipynb`, `get_distance_matrix.py`, `getting_seeds/06_get_minimum_spanning_tree.ipynb` | Consensus connectome, the 100 individual connectomes, distance matrix, MST seed |
| 1. Generate the 25,000 networks | `python pipeline_gnms_and_benchmarking/generate_morphospace.py` | `output/gnm/hcp_schaefer_100_dataset/106_distance_metrics_mst_animal_0_density10/generated_networks/` |
| 2. Score the 16 measures against the consensus | `python pipeline_gnms_and_benchmarking/score_morphospace.py --with-contributors` | `summary_indiv_<measure>_*.csv`, `timing_<measure>_*.csv` |
| 3a. Degeneration trajectories (MAE, CV) | `python pipeline_gnms_and_benchmarking/run_connectome_comparisons_null_model.py` | `output/gnm/hcp_schaefer_100_dataset/chaos_analysis/` |
| 3b. Noise tolerance $N^*$ | `python experiment_rewiring_robustness/run_rewiring_robustness.py --stage all` (then `compute_effective_degeneration.py`) | `output/rewiring_robustness/` |
| 3c. Recovery, whole morphospace | `python experiment_parameter_recovery_fine/run_wide_uniform_recovery.py --stage all` | `output/gnm/synthetic_parameter_recovery_wide_uniform/` |
| 3d. Recovery, plausible window | `python experiment_parameter_recovery_fine/derive_window.py`, then `run_synthetic_gnm_fine.py --stage all` | `output/gnm/synthetic_parameter_recovery_fine/` |
| 3e. Real vs artificial (needs the individuals) | `python experiment_real_vs_artificial/run_real_vs_artificial.py --stage all` | `output/real_vs_artificial/` |
| 3f. Hub topography | `python experiment_hub_topography/run_hub_topography.py --stage all` | `output/hub_topography/` |
| 3g. Edge importance | `python pipeline_gnms_and_benchmarking/run_connectome_targeted_removal.py` | `output/gnm/hcp_schaefer_100_dataset/importance_degradation_*/` |
| 4. Every figure | `python run_all_figures.py` (`--list` shows the plan) | figure parts, collected by `collect_figure_parts.py` |
| 5. The data bundle | `python make_publication_data.py` | `publication_data/netdistancebench-data-v1.zip` |

Stage 3c reuses the 10 x 10 grid of `experiment_parameter_recovery/`
(`run_synthetic_gnm_generation_grid.py`). The recovery experiments were
rescored in August 2026 after a fix to the orientation of the five similarity
measures (`rescore_recovery_predictions.py`); every number in the paper is
post-fix.

Figure scripts write the manuscript's figure files to `figures/manuscript/`.
Set `MANUSCRIPT_FIGURES` to write them into a LaTeX project instead.

## Notes

- **Runtimes** are wall-clock and move with the machine, so they are not
  re-measured with a rescore. The paper quotes one reference measurement, from
  the earlier run `105_distance_metrics_mst_animal_0` (`experiments_config.TIMING_EXP`).
- `105_distance_metrics_mst_animal_0` is superseded otherwise: it ran 495 model
  iterations on top of the 99-edge seed, so its networks carry 594 edges (12%)
  against the 495-edge consensus. `generate_morphospace.py` subtracts the seed.
- `configs/*.yaml` and `pipeline_gnms_and_benchmarking/run_experiment.py` are
  the generic GNM runner used for the earlier runs and the second dataset; the
  published grid is the one in `generate_morphospace.py` and `experiments_config.py`.
- The notebooks in `visualization/` are the source of the figure scripts;
  `visualization/extract_notebook.py` turns them into the `run_*.py` files that
  `run_all_figures.py` executes.
- `experiment_structural_gradient/` and `experiment_topographic_plausibility/`
  are follow-up analyses that are not in the current manuscript;
  `run_all_figures.py` still runs them because the hub-topography analysis
  imports from the first.
- `src/ESNs/` is not used by this paper; `src/pipeline/gnm_and_esn_orchestrator.py`
  imports it at module level.
- Two dataset branches of `run_connectome_comparisons_null_model.py` and
  `run_connectome_targeted_removal.py` still point into a sibling project; they
  are only reached for datasets not part of this paper.
