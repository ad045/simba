#!/bin/bash

# This script runs the GNM experiment defined in the example config.

echo "Running GNM, with the 'example_gnm.yaml' config file."
python run_experiment.py configs/example_gnm.yaml # --validate-only