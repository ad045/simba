#!/bin/bash

# This script runs the ESN experiment defined in the example config. 

echo "Running ESN, with the 'example_esn.yaml' config file."
python run_experiment.py configs/example_esn.yaml # --validate-only