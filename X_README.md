# Memory Capacity as a Driver of Brain Network Topology: A Reservoir Computing Approach to Understanding Long-Range Connections

## Overview

This project investigates why Generative Network Models (GNMs) fail to capture long-range connections in brain networks. We hypothesize that these connections exist primarily to support function rather than minimize wiring costs. Using Echo State Networks (ESNs) - a form of reservoir computing - we evaluate the functional properties of both empirical brain connectomes and synthetically generated networks.

## Key Features

- Generate synthetic brain networks using GNMs with varying parameters
- Evaluate network memory capacity using Echo State Networks
- Compare empirical human connectomes with synthetic networks
- Create 2D parameter landscapes showing functional properties
- Parallel processing support for large-scale parameter searches

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/14_4D_lab.git
cd 14_4D_lab

# Install dependencies
pip install -r requirements.txt

# Run basic pipeline
python scripts/run_full_pipeline.py

# Run with custom parameters
python scripts/run_full_pipeline.py --config configs/default.yaml --density 0.15
Installation
See docs/setup.md for detailed installation instructions.
Requirements

Python 3.8+
8GB+ RAM
CUDA GPU (optional, for acceleration)

Project Structure
14_4D_lab/
├── src/                 # Source code
│   ├── preprocessing/   # Data preprocessing
│   ├── models/         # GNM and ESN models
│   ├── analysis/       # Graph metrics and analysis
│   └── visualization/  # Plotting functions
├── configs/            # Configuration files
├── notebooks/          # Jupyter notebooks
├── scripts/           # Standalone scripts
└── docs/              # Documentation
Usage
Basic Pipeline
pythonfrom src.preprocessing import preprocess_connectomes
from src.models.gnm import generate_gnm
from src.models.esn import evaluate_memory_capacity

# Process connectomes
connectomes = preprocess_connectomes('data/raw', 
                                    consensus_threshold=0.6,
                                    density_threshold=0.2)

# Generate synthetic network
gnm = generate_gnm(connectomes['consensus'],
                  eta=1.0, gamma=-2.0)

# Evaluate memory capacity
mc_score = evaluate_memory_capacity(gnm, spectral_radius=1.0)
Parameter Search
bash# Grid search
python scripts/run_parameter_search.py --config configs/gridsearch.yaml

# Random search
python scripts/run_parameter_search.py --config configs/randomsearch.yaml
Data
The project uses 70 human connectomes from:

Resolution options: 68, 100, 200, 1000 nodes
Processing: Consensus thresholding (60%), density thresholding (10-20%), binarization

Methods
Generative Network Models (GNMs)

Parameters: η (distance penalty), γ (topology)
Wiring rule: Matching index
Energy: Kolmogorov-Smirnov distance

Echo State Networks (ESNs)

Memory capacity evaluation
Sequence recall tasks
Parameters: spectral radius, input scaling, regularization

Graph Metrics

Modularity
Global efficiency
Rich club coefficient
Characteristic path length
Average clustering

Results
Our approach demonstrates that:

Long-range connections provide functional benefits beyond wiring cost considerations
Memory capacity landscapes reveal optimal parameter regions for brain-like networks
Empirical connectomes cluster in functionally advantageous parameter spaces

Documentation

Setup Instructions
Pipeline Documentation
API Reference

Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open a Pull Request

Citation
If you use this code in your research, please cite:
bibtex@mastersthesis{dendorfer2025memory,
  title={Memory Capacity as a Driver of Brain Network Topology: 
         A Reservoir Computing Approach to Understanding Long-Range Connections},
  author={Dendorfer, Adrian},
  year={2025},
  school={Technical University of Munich},
  type={Master's Thesis}
}
References
Key papers this work builds upon:

Akarca et al. (2021) - Generative network model of neurodevelopmental diversity
Damicelli et al. (2022) - Brain connectivity meets reservoir computing
Oldham et al. (2024) - "Coming up short": GNMs and long-range connectivity
Mousley et al. (2025) - Premature birth changes wiring constraints

License
MIT License - see LICENSE file for details
Contact
Adrian Dendorfer
Email: adrian.dendorfer@tum.de
Supervisor: Kayson Fakhar (kaysonfakhar.com)
Principal Investigator: Prof. Duncan Astle (astlelab.com)
MRC Cognition and Brain Sciences Unit, University of Cambridge
Acknowledgments

MRC Cognition and Brain Sciences Unit, Cambridge
Technical University of Munich (TUM)
Authors of the Echoes and GenerativeNetworkModels libraries


tree -I '__pycache__|cache|test_*'
tree -I '__pycache__|X_additional_analyses'

Current tree: tree -I '__pycache__'
.
├── __init__.py
├── configs
│   ├── esn_sweep.yaml
│   ├── example_dynamic_gnm copy.yaml
│   ├── example_dynamic_gnm.yaml
│   ├── example_esn.yaml
│   ├── example_gnm.yaml
│   ├── gnm_sweep_small_for_testing.yaml
│   ├── gnm_sweep.yaml
├── data
│   ├── notes.md
│   ├── preprocessed
│   │   ├── 00_just_converted_for_matlab_and_python
│   │   │   ├── data_10_consensus
│   │   │   │   ├── 01_weighted_adj_mat_1000.csv
│   │   │   │   ├── 01_weighted_adj_mat_114.csv
│   │   │   │   ├── 01_weighted_adj_mat_219.csv
│   │   │   │   ├── 01_weighted_adj_mat_448.csv
│   │   │   │   ├── 01_weighted_adj_mat_68.csv
│   │   │   │   ├── 02_fiber_length_mat_1000.csv
│   │   │   │   ├── 02_fiber_length_mat_114.csv
│   │   │   │   ├── 02_fiber_length_mat_219.csv
│   │   │   │   ├── 02_fiber_length_mat_448.csv
│   │   │   │   ├── 02_fiber_length_mat_68.csv
│   │   │   │   ├── 03_fc_mat_1000.csv
│   │   │   │   ├── 03_fc_mat_114.csv
│   │   │   │   ├── 03_fc_mat_219.csv
│   │   │   │   ├── 03_fc_mat_448.csv
│   │   │   │   ├── 03_fc_mat_68.csv
│   │   │   │   ├── 04_coordinates_1000.csv
│   │   │   │   ├── 04_coordinates_114.csv
│   │   │   │   ├── 04_coordinates_219.csv
│   │   │   │   ├── 04_coordinates_448.csv
│   │   │   │   ├── 04_coordinates_68.csv
│   │   │   │   ├── 05_roi_names_rsn_name_hemisphere_1000.csv
│   │   │   │   ├── 05_roi_names_rsn_name_hemisphere_114.csv
│   │   │   │   ├── 05_roi_names_rsn_name_hemisphere_219.csv
│   │   │   │   ├── 05_roi_names_rsn_name_hemisphere_448.csv
│   │   │   │   └── 05_roi_names_rsn_name_hemisphere_68.csv
│   │   │   └── data_700mb_individual_connectomes
│   │   │       ├── SC_1000.mat
│   │   │       ├── SC_114.mat
│   │   │       ├── SC_219.mat
│   │   │       ├── SC_448.mat
│   │   │       └── SC_68.mat
│   │   └── 01_first_analysises
│   │       ├── all_connectomes_68_68.npy
│   │       ├── connectomes_binarized_68x68_density_10_percent.npy
│   │       ├── connectomes_binarized_68x68_density_12_percent.npy
│   │       ├── connectomes_binarized_68x68_density_14_percent.npy
│   │       ├── connectomes_binarized_68x68_density_16_percent.npy
│   │       ├── connectomes_binarized_68x68_density_18_percent.npy
│   │       ├── connectomes_binarized_68x68_density_20_percent.npy
│   │       ├── connectomes_binarized_68x68_density_4_percent.npy
│   │       ├── connectomes_weighted_68x68.npy
│   │       ├── consensus_connectome_all_68x68.npy
│   │       ├── consensus_connectome_bin_68x68_density_10_percent.npy
│   │       ├── df_graph_measures_all_68x68_density_10_percent.csv
│   │       ├── df_graph_measures_bin_68x68_density_10_percent.csv
│   │       ├── df_identifiers_68x68.csv
│   │       └── distance_matrix_68x68.npy
│   └── raw
│       ├── consensus_connectomes_from_70_young_adults_10mb.mat
│       ├── consensus_connectomes_from_70_young_adults_700mb.mat
│       └── downloaded_data
│           ├── atlas
│           │   ├── regions.mat
│           │   └── X_euclidean_distance.mat
├── notebooks
│   ├── __init__.py
│   ├── 00_experiments.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_experiments.ipynb
│   ├── 03_combining_GMNs_with_ESNs.ipynb
│   ├── 04_energy_grids.ipynb
│   ├── 05_plotting_energy_grids.ipynb
│   ├── 06_plotting_energy_grids_voronoi.ipynb
│   ├── 07_esn_results.ipynb
│   ├── experimenting_with_GNM_from_edward.ipynb
│   ├── notebook_setup.py
├── output
│   ├── dynamic_gnm
│   │   └── 01_dynamic_gnm_test
│   │       ├── 01_dynamic_gnm_test_20250906_112519
│   │       │   ├── config.yaml
│   │       │   ├── generated_network.npy
│   │       │   └── session_summary_2.json
│   │       ├── 01_dynamic_gnm_test_20250906_113556
│   │       │   ├── config.yaml
│   │       │   ├── dynamic_gnm_results.csv
│   │       │   ├── generated_network.npy
│   │       │   └── session_summary_2.json
│   ├── esn
│   │   └── 00_esn_experiments
│   │       └── 00_esn_experiments_20250906_161304
│   │           ├── config.yaml
│   │           ├── esn_durations_2025-09-06_16-13-05.csv
│   │           ├── esn_mc_results_2025-09-06_16-13-05.csv
│   │           ├── hyperparameter_parallel_coordinates_matplotlib_improved.png
│   │           ├── hyperparameter_parallel_coordinates.html
│   │           ├── hyperparameter_summary_avg_over_input_length_too.csv
│   │           └── run_info_2025-09-06_16-13-05.txt
│   └── gnm
├── pyproject.toml
├── README.md
├── run_experiment.py
├── scripts
│   ├── __init__.py
│   ├── 03_visualization.py
│   └── run_esn_example.sh
├── src
│   ├── __init__.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── config_plotting.py
│   │   ├── ESN_and_GNM_config.py
│   │   └── yaml_loader.py
│   ├── ESNs
│   │   ├── __init__.py
│   │   ├── esn_evaluation.py
│   │   ├── test_memory_capacity_weighted.py
│   │   ├── utils_math.py
│   │   └── utils.py
│   ├── eval_avg_over_input_length_too.py
│   ├── gnm_and_esn_orchestrator.py
│   ├── GNMs
│   │   └── gnm_network_generator.py
│   ├── imported_libraries
│   │   ├── __init__.py
│   │   ├── GenerativeNetworkModels_2
│   │   │   ├── __init__.py
│   │   │   ├── build
│   │   │   │   ├── bdist.macosx-11.0-arm64
│   │   │   │   ├── bdist.macosx-11.1-arm64
│   │   │   │   └── lib
│   │   │   │       └── gnm
│   │   │   │           ├── __init__.py
│   │   │   │           ├── defaults
│   │   │   │           │   ├── __init__.py
│   │   │   │           │   ├── generate_heterochronous_matrix.py
│   │   │   │           │   └── get_defaults.py
│   │   │   │           ├── evaluation
│   │   │   │           │   ├── __init__.py
│   │   │   │           │   ├── binary_corr_criteria.py
│   │   │   │           │   ├── binary_ks_criteria.py
│   │   │   │           │   ├── composite_criteria.py
│   │   │   │           │   ├── evaluation_base.py
│   │   │   │           │   ├── population_evaluation.py
│   │   │   │           │   └── weighted_ks_criteria.py
│   │   │   │           ├── fitting
│   │   │   │           │   ├── __init__.py
│   │   │   │           │   ├── analysis.py
│   │   │   │           │   ├── experiment_dataclasses.py
│   │   │   │           │   ├── experiment_saving.py
│   │   │   │           │   └── sweep.py
│   │   │   │           ├── generative_rules
│   │   │   │           │   ├── __init__.py
│   │   │   │           │   └── generative_rules.py
│   │   │   │           ├── model.py
│   │   │   │           ├── utils
│   │   │   │           │   ├── __init__.py
│   │   │   │           │   ├── checks.py
│   │   │   │           │   ├── control.py
│   │   │   │           │   ├── convert_datatypes.py
│   │   │   │           │   ├── graph_properties.py
│   │   │   │           │   └── statistics.py
│   │   │   │           └── weight_criteria
│   │   │   │               ├── __init__.py
│   │   │   │               └── optimisation_criteria.py
│   │   │   ├── CITATION.cff
│   │   │   ├── docs
│   │   │   │   ├── api-reference
│   │   │   │   │   ├── defaults.md
│   │   │   │   │   ├── evaluation.md
│   │   │   │   │   ├── fitting.md
│   │   │   │   │   ├── generative-rules.md
│   │   │   │   │   ├── index.md
│   │   │   │   │   ├── model.md
│   │   │   │   │   ├── utils.md
│   │   │   │   │   └── weight-criteria.md
│   │   │   │   ├── examples
│   │   │   │   │   ├── eval_bct_gnm_performance.py
│   │   │   │   │   ├── example_wandb_run.ipynb
│   │   │   │   │   ├── experiment_saving_example.ipynb
│   │   │   │   │   ├── get_graph_metrics_example_2 copy.ipynb
│   │   │   │   │   ├── get_graph_metrics_example_2 copy.txt
│   │   │   │   │   ├── get_graph_metrics_example_2.ipynb
│   │   │   │   │   ├── get_graph_metrics_example.ipynb
│   │   │   │   │   ├── graph_model_performance.ipynb
│   │   │   │   │   ├── index.md
│   │   │   │   │   ├── sweep_example.ipynb
│   │   │   │   │   ├── weighted_increments.ipynb
│   │   │   │   │   └── weighted_sweep.ipynb
│   │   │   │   ├── getting-started.md
│   │   │   │   ├── images
│   │   │   │   │   └── binary_consensus.png
│   │   │   │   ├── index.md
│   │   │   │   ├── javascripts
│   │   │   │   │   └── mathjax.js
│   │   │   │   ├── requirements.txt
│   │   │   │   ├── stylesheets
│   │   │   │   │   └── extra.css
│   │   │   │   ├── understanding-gnms
│   │   │   │   │   ├── binary-gnms.md
│   │   │   │   │   ├── figures
│   │   │   │   │   │   ├── all_figures.pdf
│   │   │   │   │   │   ├── fig1.pdf
│   │   │   │   │   │   ├── fig2.pdf
│   │   │   │   │   │   ├── fig3.pdf
│   │   │   │   │   │   ├── fig4.pdf
│   │   │   │   │   │   └── fig5.pdf
│   │   │   │   │   ├── figures-png
│   │   │   │   │   │   ├── fig1.png
│   │   │   │   │   │   ├── fig2.png
│   │   │   │   │   │   ├── fig3.png
│   │   │   │   │   │   ├── fig4.png
│   │   │   │   │   │   └── fig5.png
│   │   │   │   │   ├── fitting-gnms.md
│   │   │   │   │   ├── glossary.md
│   │   │   │   │   ├── heterochronous-gnms.md
│   │   │   │   │   ├── index.md
│   │   │   │   │   ├── networks-and-graphs.md
│   │   │   │   │   └── weighted-gnms.md
│   │   │   │   └── user-guide
│   │   │   │       ├── design-philosophy.md
│   │   │   │       └── index.md
│   │   │   ├── environment
│   │   │   │   └── development_environment.yml
│   │   │   ├── LICENSE.txt
│   │   │   ├── mkdocs.yml
│   │   │   ├── pyproject.toml
│   │   │   ├── README.md
│   │   │   ├── requirements.txt
│   │   │   ├── setup.py
│   │   │   ├── src
│   │   │   │   ├── __init__.py
│   │   │   │   ├── GenerativeNetworkModels.egg-info
│   │   │   │   │   ├── dependency_links.txt
│   │   │   │   │   ├── PKG-INFO
│   │   │   │   │   ├── requires.txt
│   │   │   │   │   ├── SOURCES.txt
│   │   │   │   │   └── top_level.txt
│   │   │   │   └── gnm
│   │   │   │       ├── __init__.py
│   │   │   │       ├── _device.py
│   │   │   │       ├── defaults
│   │   │   │       │   ├── __init__.py
│   │   │   │       │   ├── binary_networks
│   │   │   │       │   │   └── CALM_BINARY_CONSENSUS.pt
│   │   │   │       │   ├── coordinates
│   │   │   │       │   │   └── AAL_COORDINATES.pt
│   │   │   │       │   ├── distance_matrices
│   │   │   │       │   │   └── AAL_DISTANCES.pt
│   │   │   │       │   ├── generate_heterochronous_matrix.py
│   │   │   │       │   ├── get_defaults.py
│   │   │   │       │   └── weighted_networks
│   │   │   │       │       └── CALM_WEIGHTED_CONSENSUS.pt
│   │   │   │       ├── evaluation
│   │   │   │       │   ├── __init__.py
│   │   │   │       │   ├── binary_corr_criteria.py
│   │   │   │       │   ├── binary_ks_criteria.py
│   │   │   │       │   ├── composite_criteria.py
│   │   │   │       │   ├── evaluation_base.py
│   │   │   │       │   ├── population_evaluation.py
│   │   │   │       │   └── weighted_ks_criteria.py
│   │   │   │       ├── fitting
│   │   │   │       │   ├── __init__.py
│   │   │   │       │   ├── analysis.py
│   │   │   │       │   ├── experiment_dataclasses.py
│   │   │   │       │   ├── experiment_saving.py
│   │   │   │       │   └── sweep.py
│   │   │   │       ├── generative_rules
│   │   │   │       │   ├── __init__.py
│   │   │   │       │   └── generative_rules.py
│   │   │   │       ├── model.py
│   │   │   │       ├── tools
│   │   │   │       │   └── patch_macos_gnm.py
│   │   │   │       ├── utils
│   │   │   │       │   ├── __init__.py
│   │   │   │       │   ├── checks.py
│   │   │   │       │   ├── control.py
│   │   │   │       │   ├── convert_datatypes.py
│   │   │   │       │   ├── graph_properties.py
│   │   │   │       │   └── statistics.py
│   │   │   │       └── weight_criteria
│   │   │   │           ├── __init__.py
│   │   │   │           └── optimisation_criteria.py
│   │   │   └── tests
│   │   │       ├── conftest.py
│   │   │       ├── mean_connectome.npy
│   │   │       ├── test_experiment_saving.py
│   │   │       ├── test_graph_metrics.py
│   │   │       ├── test_ks.py
│   │   │       ├── wandb_bayesian_optimisation_test.py
│   │   │       ├── wandb_logging_test.py
│   │   │       └── wandb_test.py
│   │   └── what_to_include_here.md
│   ├── pipeline
│   │   ├── __init__.py
│   │   └── orchestrator.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── get_70_connectomes_700mb.m
│   │   ├── get_consensus_data_10_2.m
│   │   ├── preprocess_70_connectomes.py
│   │   ├── preprocess_distance_matrix.py
│   │   ├── preprocessing_pipeline.py
│   │   └── preprocessing_setup.py
│   ├── structural_analysis
│   │   ├── __init__.py
│   │   └── graph_measures.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── run_logger.py
│   │   └── saving_and_finding_files.py
│   └── visualization
│       ├── __init__.py
│       └── energy_and_mc_landscape.py