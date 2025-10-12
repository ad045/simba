TBD!! 
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
tbd

Results
(tbd) 

Documentation

Setup Instructions
Pipeline Documentation
API Reference

References
Key papers this work builds upon:
tbd (Akarca, Damicelli, Oldham, Mousley)
