# Connectome Analysis Pipeline

A modular, scalable pipeline for analyzing connectomes using Echo State Networks (ESNs) and Generative Network Models (GNMs).

## Overview

This pipeline replaces and improves upon your original two scripts by providing:

1. **Modular Architecture**: Separate modules for configuration, data loading, ESN evaluation, and GNM generation
2. **Flexible Configuration**: Easy-to-modify configuration system with presets
3. **Parallel Processing**: Efficient multiprocessing for large-scale experiments
4. **Incremental Saving**: Results saved progressively to prevent data loss
5. **Command-Line Interface**: Easy-to-use CLI for different experiment types
6. **Comprehensive Analysis**: Built-in result analysis and visualization

## Module Structure

```
├── config.py              # Configuration management
├── data_loader.py          # Data loading and validation
├── esn_evaluation.py       # ESN hyperparameter optimization
├── gnm_generation.py       # GNM parameter fitting
├── main_pipeline.py        # Main orchestrator with CLI
└── run_examples.py         # Example usage scripts
```

## Quick Start

### 1. Basic Usage (Command Line)

```bash
# Quick test run
python main_pipeline.py full --quick-test

# Data summary
python main_pipeline.py data-summary

# ESN hyperparameter sweep
python main_pipeline.py esn --config production

# GNM parameter fitting
python main_pipeline.py gnm --gnm-density 15 --gnm-n-eta 50 --gnm-n-gamma 50

# Full pipeline
python main_pipeline.py full --config production
```

### 2. Programmatic Usage

```python
from main_pipeline import create_pipeline_orchestrator

# Create orchestrator with production config
orchestrator = create_pipeline_orchestrator("production")

# Run ESN hyperparameter sweep
esn_results = orchestrator.run_esn_hyperparameter_sweep(
    densities=[10, 15, 20],
    spectral_radii=[0.8, 0.99, 1.2],
    input_lengths=[2000, 4000],
    regularization_methods=["ridge"]
)

# Run GNM parameter fitting
gnm_results = orchestrator.run_gnm_parameter_fitting(
    density=15,
    n_eta=100,
    n_gamma=100
)
```

## Configuration Presets

### Quick Test
- Small hyperparameter grids
- Few subjects
- 2 CPU workers
- Good for development and testing

### Production
- Comprehensive hyperparameter grids
- All subjects
- All available CPU cores
- High number of runs for statistical significance

### Hyperparameter Sweep
- Extended parameter ranges
- All density levels
- Optimized for thorough exploration

## Key Improvements Over Original Scripts

### Compared to Script No 1:
1. **Modular Structure**: GNM generation separated from evaluation
2. **Better Error Handling**: Robust error handling and recovery
3. **Incremental Saving**: Results saved continuously, not just at the end
4. **Flexible Parameters**: Easy to modify GNM parameter ranges and grid sizes
5. **Progress Tracking**: Real-time progress updates and timing information

### Compared to Script No 2:
1. **Fixed Parameter Handling**: Consistent hyperparameter management
2. **Better Random Seed Control**: Reproducible results
3. **Cleaner Grid Generation**: More intuitive hyperparameter specification
4. **Enhanced Analysis**: Built-in result analysis and best parameter identification
5. **Memory Efficiency**: Better handling of large parameter grids

## Advanced Features

### Custom Configuration

```python
from config import ConfigManager, ESNConfig, GNMConfig, DataConfig

# Create custom configuration
esn_config = ESNConfig(
    spectral_radius=0.99,
    input_length=4000,
    n_runs=100,
    regularization_method="ridge"
)

gnm_config = GNMConfig(
    n_eta=200,
    n_gamma=200,
    eta_start=-4.0,
    eta_end=1.0
)

data_config = DataConfig(
    resolution=68,
    densities=[10, 12, 14, 16, 18, 20]
)

config = ConfigManager(esn_config, gnm_config, data_config)
```

### Result Analysis

```python
from esn_evaluation import create_esn_evaluator
from gnm_generation import create_gnm_generator

# Analyze ESN results
evaluator = create_esn_evaluator()
analysis = evaluator.load_and_analyze_results("path/to/results")

print(f"Best memory capacity: {analysis['best_overall_mc']['value']}")
print(f"Best hyperparameters: {analysis['best_overall_mc']['hyperparameters']}")

# Analyze GNM results
generator = create_gnm_generator()
gnm_analysis = generator.load_and_analyze_gnm_results("path/to/gnm/results")

print(f"Average best eta: {gnm_analysis['best_parameters']['eta_mean']}")
print(f"Average best gamma: {gnm_analysis['best_parameters']['gamma_mean']}")
```

## Command-Line Options

### ESN Commands
```bash
# Basic ESN sweep
python main_pipeline.py esn

# Custom experiment name
python main_pipeline.py esn --esn-experiment-name "my_esn_experiment"

# Random sampling instead of full grid
python main_pipeline.py esn --esn-search-mode random_sample --esn-random-sample-size 100
```

### GNM Commands
```bash
# Basic GNM fitting
python main_pipeline.py gnm

# Specific density and custom grid size
python main_pipeline.py gnm --gnm-density 15 --gnm-n-eta 50 --gnm-n-gamma 50

# Custom parameter ranges
python main_pipeline.py gnm --gnm-eta-range -4.0 1.0 --gnm-gamma-range 0.05 0.8
```

### Full Pipeline
```bash
# Production run
python main_pipeline.py full --config production

# Quick test
python main_pipeline.py full --quick-test

# With custom experiment names
python main_pipeline.py full --esn-experiment-name "comprehensive_esn" --gnm-experiment-name "comprehensive_gnm"
```

## Output Structure

Each experiment creates a structured output directory:

```
experiment_directory/
├── experiment_config.json      # Configuration used
├── run_info_TIMESTAMP.txt      # Runtime information
├── result_analysis.json        # Analysis results
├── esn_mc_results_TIMESTAMP.csv        # Main results
├── esn_durations_TIMESTAMP.csv         # Timing information
└── preliminary_results/         # Incremental saves (for GNM)
    ├── gnm_grid_partial_TIMESTAMP.csv
    ├── gnm_best_partial_TIMESTAMP.csv
    └── gnm_timing_partial_TIMESTAMP.csv
```

## Performance Considerations

- **Memory Usage**: Large hyperparameter grids can use significant memory
- **CPU Usage**: Uses all available CPU cores by default (configurable)
- **Disk Space**: Results files can be large for comprehensive sweeps
- **Time Estimates**: 
  - Quick test: ~5-10 minutes
  - Small ESN sweep: ~30-60 minutes  
  - Full production pipeline: ~2-8 hours (depending on hardware)

## Error Handling

The pipeline includes robust error handling:
- **Data validation**: Checks for missing files and data consistency
- **Graceful degradation**: Continues with available data if some files are missing
- **Progress preservation**: Incremental saving prevents loss of partial results
- **Detailed logging**: Comprehensive error messages and timing information

## Migration from Original Scripts

To migrate from your original scripts:

1. **Replace Script No 2** with: `python main_pipeline.py esn --config hyperparameter_sweep`
2. **Replace Script No 1** with: `python main_pipeline.py gnm --config production`
3. **For combined analysis**: `python main_pipeline.py full --config production`

The new pipeline provides all the functionality of your original scripts plus significant improvements in modularity, error handling, and usability.