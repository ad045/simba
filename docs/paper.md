# Reproducing the paper

The `paper/` folder of the repository holds the full research pipeline: the code
that generated the networks, scored the 16 measures, ran every analysis and drew
every figure. The package covers scoring a new measure; this page is for
re-running the paper itself.

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
and result); create them, or link them to wherever your copy lives.

`experiments_config.py` is the single source of truth for the eight selected
measures, their orientation, names, colours, the grids and the file layout.

## Pipeline

| Stage | Run | Produces |
|---|---|---|
| 0. Empirical preprocessing (needs HCP access) | `src/preprocessing/02_preprocessing_pipeline_hcp_schaefer_100.ipynb`, `get_distance_matrix.py` | Consensus connectome, the 100 individual connectomes, distance matrix, MST seed |
| 1. Generate the 25,000 networks | `python pipeline_gnms_and_benchmarking/generate_morphospace.py` | `output/gnm/hcp_schaefer_100_dataset/106_distance_metrics_mst_animal_0_density10/generated_networks/` |
| 2. Score the 16 measures against the consensus | `python pipeline_gnms_and_benchmarking/score_morphospace.py --with-contributors` | `summary_indiv_<measure>_*.csv`, `timing_<measure>_*.csv` |
| 3a. Degeneration trajectories (MAE, CV) | `python pipeline_gnms_and_benchmarking/run_connectome_comparisons_null_model.py` | `output/gnm/hcp_schaefer_100_dataset/chaos_analysis/` |
| 3b. Noise tolerance $N^*$ | `python experiment_rewiring_robustness/run_rewiring_robustness.py --stage all` (then `compute_effective_degeneration.py`) | `output/rewiring_robustness/` |
| 3c. Recovery, whole morphospace | `python experiment_parameter_recovery_fine/run_wide_uniform_recovery.py --stage all` | `output/gnm/synthetic_parameter_recovery_wide_uniform/` |
| 3d. Recovery, plausible window | `python experiment_parameter_recovery_fine/derive_window.py`, then `run_synthetic_gnm_fine.py --stage all` | `output/gnm/synthetic_parameter_recovery_fine/` |
| 3e. Real vs artificial (needs the individuals) | `python experiment_real_vs_artificial/run_real_vs_artificial.py --stage all` | `output/real_vs_artificial/` |
| 3f. Topographic analyses | `experiment_structural_gradient/`, `experiment_topographic_plausibility/`, `experiment_hub_topography/` (each `--stage all`) | `output/<analysis>/` |
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
- `src/ESNs/` is not used by this paper; `src/pipeline/gnm_and_esn_orchestrator.py`
  imports it at module level.
- Two dataset branches of `run_connectome_comparisons_null_model.py` and
  `run_connectome_targeted_removal.py` still point into a sibling project; they
  are only reached for datasets not part of this paper.
