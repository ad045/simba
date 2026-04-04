#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 1: Generate networks ==="
python run_synthetic_gnm_generation_grid.py

echo "=== Step 2: Compare against grid ==="
python run_synthetic_gnm_comparison_grid.py

echo "=== Step 3: Plot ==="
python run_synthetic_gnm_plots_grid.py

echo "=== Done ==="
