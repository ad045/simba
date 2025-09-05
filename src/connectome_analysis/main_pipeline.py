"""
YAML-based configuration system integrated with existing ConfigManager.
This bridges YAML files with the existing config.py structure.
"""

# TODO: 
    # - Save in run_info.txt the actual config used (including lists for grid search)
    # - Print different things based on experiment type (ESN vs GNM)
    # - Test ESN and GNM sweep experiments -> ESN works really well now - what about GNM, though? 
    
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
import time

import json
import shutil

import torch
import numpy as np

# Import your existing configuration
from config import (
    ConfigManager, ESNConfig, GNMConfig, DataConfig, 
    ComputeConfig, PathConfig
)

class YAMLConfigLoader:
    """Load and merge YAML configurations with existing ConfigManager."""
    
    def __init__(self, yaml_path: Path):
        """Initialize with path to YAML config file."""
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(self.yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate required sections
        self._validate_config()
    
    def _validate_config(self):
        """Validate that config has required structure."""
        required_sections = ['experiment', 'data', 'output']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section '{section}' in config")
        
        # Validate experiment type
        if 'type' not in self.config['experiment']:
            raise ValueError("Missing 'type' in experiment section")
        
        valid_types = ['esn', 'gnm_sweep', 'gnm_comprehensive', 'full_pipeline', 'gnm_esn_grid']
        if self.config['experiment']['type'] not in valid_types:
            raise ValueError(f"Invalid experiment type. Must be one of: {valid_types}")
    
    def create_config_manager(self) -> ConfigManager:
        """Create a ConfigManager instance from YAML configuration."""
        
        # Create component configs
        esn_cfg = self._create_esn_config()
        gnm_cfg = self._create_gnm_config()
        data_cfg = self._create_data_config()
        compute_cfg = self._create_compute_config()
        path_cfg = self._create_path_config()
        
        # Create and return ConfigManager
        return ConfigManager(
            esn_config=esn_cfg,
            gnm_config=gnm_cfg,
            data_config=data_cfg,
            compute_config=compute_cfg,
            path_config=path_cfg
        )
    
    def _create_esn_config(self) -> ESNConfig:
        """Create ESNConfig from YAML."""
        esn_cfg = ESNConfig()
        
        if 'esn' not in self.config:
            return esn_cfg
        
        yaml_esn = self.config['esn']
        
        # Map YAML fields to ESNConfig fields
        if 'spectral_radius' in yaml_esn:
            # Handle both single value and list
            if isinstance(yaml_esn['spectral_radius'], list):
                # Store first value as default, keep list for grid search
                esn_cfg.spectral_radius = yaml_esn['spectral_radius'][0]
                # Store the full list in a custom attribute for grid generation
                esn_cfg.spectral_radii_grid = yaml_esn['spectral_radius']
            else:
                esn_cfg.spectral_radius = yaml_esn['spectral_radius']
        
        if 'input_scaling' in yaml_esn:
            if isinstance(yaml_esn['input_scaling'], list):
                esn_cfg.input_scaling = yaml_esn['input_scaling'][0]
                esn_cfg.input_scalings_grid = yaml_esn['input_scaling']
            else:
                esn_cfg.input_scaling = yaml_esn['input_scaling']
        
        if 'input_lengths' in yaml_esn:
            if isinstance(yaml_esn['input_lengths'], list):
                esn_cfg.input_length = yaml_esn['input_lengths'][0]
                esn_cfg.input_lengths_grid = yaml_esn['input_lengths']
            else:
                esn_cfg.input_length = yaml_esn['input_lengths']
        elif 'input_length' in yaml_esn:
            esn_cfg.input_length = yaml_esn['input_length']
        
        if 'regularization' in yaml_esn:
            esn_cfg.regularization_method = yaml_esn['regularization']
        elif 'regularization_methods' in yaml_esn:
            if isinstance(yaml_esn['regularization_methods'], list):
                esn_cfg.regularization_method = yaml_esn['regularization_methods'][0]
                esn_cfg.regularization_methods_grid = yaml_esn['regularization_methods']
            else:
                esn_cfg.regularization_method = yaml_esn['regularization_methods']
        
        if 'n_runs' in yaml_esn:
            esn_cfg.n_runs = yaml_esn['n_runs']
        if 'n_lags' in yaml_esn:
            esn_cfg.n_lags = yaml_esn['n_lags']
        if 'test_len' in yaml_esn:
            esn_cfg.test_len = yaml_esn['test_len']
        if 'n_transient' in yaml_esn:
            esn_cfg.n_transient = yaml_esn['n_transient']
        if 'leak_rate' in yaml_esn:
            esn_cfg.leak_rate = yaml_esn['leak_rate']
        if 'bias' in yaml_esn:
            esn_cfg.bias = yaml_esn['bias']
        
        return esn_cfg
    
    def _create_gnm_config(self) -> GNMConfig:
        """Create GNMConfig from YAML."""
        gnm_cfg = GNMConfig()
        
        if 'gnm' not in self.config:
            return gnm_cfg
        
        yaml_gnm = self.config['gnm']
        
        # Parameter ranges
        if 'eta_range' in yaml_gnm:
            gnm_cfg.eta_range = tuple(yaml_gnm['eta_range'])
        if 'gamma_range' in yaml_gnm:
            gnm_cfg.gamma_range = tuple(yaml_gnm['gamma_range'])
        if 'lambda_range' in yaml_gnm:
            gnm_cfg.lambda_range = tuple(yaml_gnm['lambda_range'])
        
        # Grid points
        if 'n_eta' in yaml_gnm:
            gnm_cfg.n_eta = yaml_gnm['n_eta']
        if 'n_gamma' in yaml_gnm:
            gnm_cfg.n_gamma = yaml_gnm['n_gamma']
        if 'n_lambda' in yaml_gnm:
            gnm_cfg.n_lambda = yaml_gnm['n_lambda']
        
        # For backward compatibility with n_grid_points
        if 'n_grid_points' in yaml_gnm:
            gnm_cfg.n_eta = yaml_gnm['n_grid_points']
            gnm_cfg.n_gamma = yaml_gnm['n_grid_points']
        
        # Wiring rules
        if 'wiring_rules' in yaml_gnm:
            gnm_cfg.generative_rules_to_test = yaml_gnm['wiring_rules']
        elif 'wiring_rule' in yaml_gnm:
            gnm_cfg.generative_rules_to_test = [yaml_gnm['wiring_rule']]
        
        # Evaluation metrics
        if 'evaluation_metrics' in yaml_gnm:
            gnm_cfg.evaluation_metrics = yaml_gnm['evaluation_metrics']
        
        # Weight optimization
        if 'weight_criterion' in yaml_gnm:
            gnm_cfg.weight_criterion = yaml_gnm['weight_criterion']
        if 'alpha' in yaml_gnm:
            gnm_cfg.alpha = yaml_gnm['alpha']
        
        # Simulation parameters
        if 'num_simulations' in yaml_gnm:
            gnm_cfg.num_simulations = yaml_gnm['num_simulations']
        if 'device' in yaml_gnm:
            gnm_cfg.device = yaml_gnm['device']
        
        return gnm_cfg
    
    def _create_data_config(self) -> DataConfig:
        """Create DataConfig from YAML."""
        data_cfg = DataConfig()
        
        if 'data' not in self.config:
            return data_cfg
        
        yaml_data = self.config['data']
        
        if 'connectome_resolution' in yaml_data:
            data_cfg.resolution = yaml_data['connectome_resolution']
        if 'resolution' in yaml_data:  # Alternative name
            data_cfg.resolution = yaml_data['resolution']
        
        if 'densities' in yaml_data:
            data_cfg.densities = yaml_data['densities']
        elif 'density_threshold' in yaml_data:
            # Convert single threshold to list of densities
            # This is a simplified conversion - adjust as needed
            density = int(yaml_data['density_threshold'] * 100)
            data_cfg.densities = [density]
        
        if 'use_weighted' in yaml_data:
            data_cfg.use_weighted = yaml_data['use_weighted']
        if 'use_gnm_defaults' in yaml_data:
            data_cfg.use_gnm_defaults = yaml_data['use_gnm_defaults']
        
        return data_cfg
    
    def _create_compute_config(self) -> ComputeConfig:
        """Create ComputeConfig from YAML."""
        compute_cfg = ComputeConfig()
        
        if 'compute' in self.config:
            yaml_compute = self.config['compute']
            
            if 'n_workers' in yaml_compute:
                compute_cfg.n_workers = yaml_compute['n_workers']
            if 'timing_flag' in yaml_compute:
                compute_cfg.timing_flag = yaml_compute['timing_flag']
            if 'append_interval' in yaml_compute:
                compute_cfg.append_interval = yaml_compute['append_interval']
            if 'random_seed' in yaml_compute:
                compute_cfg.random_seed = yaml_compute['random_seed']
        
        # Also check output section for some compute-related settings
        if 'output' in self.config:
            yaml_output = self.config['output']
            if 'random_seed' in yaml_output:
                compute_cfg.random_seed = yaml_output['random_seed']
        
        return compute_cfg
    
    def _create_path_config(self) -> PathConfig:
        """Create PathConfig from YAML."""
        path_cfg = PathConfig()
        
        if 'output' in self.config:
            yaml_output = self.config['output']
            
            if 'base_dir' in yaml_output:
                # Override the output directory
                path_cfg.output_dir = Path(yaml_output['base_dir'])
                path_cfg.esn_output_dir = path_cfg.output_dir / "esn"
                path_cfg.gnm_output_dir = path_cfg.output_dir / "gnm"
            
            if 'root_dir' in yaml_output:
                path_cfg.root_dir = Path(yaml_output['root_dir'])
                # Recompute derived paths
                path_cfg.data_dir = path_cfg.root_dir / "data/preprocessed/01_first_analysises"
                if 'base_dir' not in yaml_output:
                    path_cfg.output_dir = path_cfg.root_dir / "output"
                    path_cfg.esn_output_dir = path_cfg.output_dir / "esn"
                    path_cfg.gnm_output_dir = path_cfg.output_dir / "gnm"
        
        return path_cfg
    
    def get_experiment_args(self) -> Dict[str, Any]:
        """Extract experiment-specific arguments from config."""
        exp_cfg = self.config['experiment']
        args = {}
        
        # Experiment name
        if 'name' in exp_cfg:
            args['experiment_name'] = exp_cfg['name']
        elif 'output' in self.config and 'experiment_name' in self.config['output']:
            args['experiment_name'] = self.config['output']['experiment_name']
        
        # Search strategy
        if 'search' in exp_cfg:
            search_cfg = exp_cfg['search']
            if 'method' in search_cfg:
                if search_cfg['method'] == 'random':
                    args['random_sample'] = True
                    if 'n_samples' in search_cfg:
                        args['n_random_samples'] = search_cfg['n_samples']
                    if 'random_sample_size' in search_cfg:
                        args['random_sample_size'] = search_cfg['random_sample_size']
                elif search_cfg['method'] == 'grid':
                    args['random_sample'] = False
                    args['search_mode'] = 'grid'
                elif search_cfg['method'] == 'bayesian':
                    args['no_wandb'] = False
        
        # Wandb settings
        if 'wandb' in exp_cfg:
            wandb_cfg = exp_cfg['wandb']
            if 'enabled' in wandb_cfg:
                args['no_wandb'] = not wandb_cfg['enabled']
            if 'project' in wandb_cfg:
                args['wandb_project'] = wandb_cfg['project']
        else:
            # Default to no wandb unless explicitly enabled
            args['no_wandb'] = True
        
        # GNM-specific arguments
        if exp_cfg['type'] in ['gnm_sweep', 'gnm_comprehensive']:
            if 'compare_rules' in exp_cfg:
                args['compare_rules'] = exp_cfg['compare_rules']
            if 'fit_weights' in exp_cfg:
                args['fit_weights'] = exp_cfg['fit_weights']
        
        # ESN-specific arguments  
        if exp_cfg['type'] == 'esn':
            if 'search' in exp_cfg:
                search_cfg = exp_cfg['search']
                if 'method' in search_cfg:
                    args['search_mode'] = 'random_sample' if search_cfg['method'] == 'random' else 'grid'
                if 'random_sample_size' in search_cfg:
                    args['random_sample_size'] = search_cfg['random_sample_size']
        
        return args
    
    def generate_esn_hparam_grid(self, config: ConfigManager) -> List[Dict[str, Any]]:
        """Generate ESN hyperparameter grid from YAML configuration."""
        
        # Extract grid values from ESN config (stored during creation)
        esn_cfg = config.esn
        
        # Get grid values or use defaults
        spectral_radii = getattr(esn_cfg, 'spectral_radii_grid', [esn_cfg.spectral_radius])
        input_scalings = getattr(esn_cfg, 'input_scalings_grid', [esn_cfg.input_scaling])
        input_lengths = getattr(esn_cfg, 'input_lengths_grid', [esn_cfg.input_length])
        reg_methods = getattr(esn_cfg, 'regularization_methods_grid', [esn_cfg.regularization_method])
        
        # Use data config for densities
        densities = config.data.densities
        
        # Generate grid using ConfigManager's method
        return config.generate_esn_hparam_grid(
            spectral_radii=spectral_radii,
            input_lengths=input_lengths,
            input_scalings=input_scalings,
            regularization_methods=reg_methods,
            densities=densities
        )


def run_from_yaml(yaml_path: str):
    """Main function to run pipeline from YAML configuration."""
    
    # Load YAML config
    loader = YAMLConfigLoader(yaml_path)
    
    # Create ConfigManager from YAML
    config = loader.create_config_manager()
    
    # Create run_info.txt in the experiment output directory
    def save_run_info(config, yaml_path, exp_args, exp_type):
        # Determine the output directory based on experiment type
        current_time = time.strftime('%Y%m%d_%H%M%S') 
        precise_folder_name = exp_args.get('experiment_name', 'default_no_exp_name_set') + f"_{current_time}"
        if exp_type == 'esn':
            output_dir = config.paths.esn_output_dir / exp_args.get('experiment_name', f"esn_{current_time}") / precise_folder_name
        else:
            output_dir = config.paths.gnm_output_dir / exp_args.get('experiment_name', f"gnm_{current_time}") / precise_folder_name
        
        # # Ensure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # # Save run_info.txt
        # run_info_path = output_dir / "run_info.txt"
        # with open(run_info_path, 'w') as f:
        #     # Write experiment details
        #     f.write("Experiment Configuration\n")
        #     f.write("=" * 40 + "\n")
        #     f.write(f"Experiment Type: {exp_type}\n")
        #     f.write(f"Experiment Name: {exp_args.get('experiment_name', 'Unnamed')}\n\n")
            
        #     # Save a copy of the original YAML config
        #     f.write("Original YAML Configuration File\n")
        #     f.write("=" * 40 + "\n")
        #     with open(yaml_path, 'r') as yaml_file:
        #         f.write(yaml_file.read() + "\n\n")
            
        #     # Serialize and write parsed configuration details
        #     f.write("Parsed Configuration Summary\n")
        #     f.write("=" * 40 + "\n")
        #     config_summary = {
        #         "data": {
        #             "resolution": config.data.resolution,
        #             "densities": config.data.densities,
        #             "use_weighted": config.data.use_weighted,
        #         },
        #         "esn": {
        #             "spectral_radius": config.esn.spectral_radius,
        #             "input_scaling": config.esn.input_scaling,
        #             "input_length": config.esn.input_length,
        #             "regularization_method": config.esn.regularization_method,
        #             "n_runs": config.esn.n_runs,
        #             "n_lags": config.esn.n_lags,
        #         },
        #         "gnm": {
        #             "eta_range": config.gnm.eta_range,
        #             "gamma_range": config.gnm.gamma_range,
        #             "generative_rules_to_test": config.gnm.generative_rules_to_test,
        #         },
        #         "compute": {
        #             "n_workers": config.compute.n_workers,
        #             "random_seed": config.compute.random_seed,
        #         },
        #         "experiment_args": exp_args
        #     }
            
        #     # Use json.dump for cleaner formatting
        #     json.dump(config_summary, f, indent=2)
        
        # Copy the original YAML config to the output directory
        shutil.copy(yaml_path, output_dir / "config.yaml")
        
        return output_dir

    # Get experiment type and arguments
    exp_type = loader.config['experiment']['type']
    exp_args = loader.get_experiment_args()
    # Save run info
    output_dir_for_saving = save_run_info(config, yaml_path, exp_args, exp_type)

    
    # Print configuration summary
    print("=" * 60)
    print(f"RUNNING EXPERIMENT FROM YAML: {yaml_path}")
    print("=" * 60)
    print(f"Experiment Type: {exp_type}")
    if 'experiment_name' in exp_args:
        print(f"Experiment Name: {exp_args['experiment_name']}")
    if 'search' in loader.config['experiment']:
        print(f"Search Method: {loader.config['experiment']['search'].get('method', 'default')}")
    print("-" * 60)
    
    # Import pipeline orchestrator
    from src.connectome_analysis.gnm_and_esn_orchestrator import GNMandESNPipelineOrchestrator
    
    # Run appropriate experiment type
    if exp_type == 'esn':
        # Special handling for ESN
        from esn_evaluation import create_esn_evaluator
        from data_loader import DataLoader
        
        esn_evaluator = create_esn_evaluator(config_manager=config)
        
        # Generate hyperparameter grid from YAML config
        hparam_grid = loader.generate_esn_hparam_grid(config)
        
        # Load data
        data_loader = DataLoader(config)
        weighted_by_density = data_loader.load_weighted_by_density()
        
        # Run ESN sweep
        exp_name = exp_args.get('experiment_name', f"esn_{time.strftime('%Y%m%d_%H%M%S')}")
        exp_dir = output_dir_for_saving # is already taken from config; and now adapted. config.paths.esn_output_dir / exp_name
        
        
        search_mode = exp_args.get('search_mode', 'random_sample')
        random_sample_size = exp_args.get('random_sample_size', None)
        
        esn_message = esn_evaluator.run_hyperparameter_sweep(
            connectomes=weighted_by_density,
            hparam_grid=hparam_grid,
            save_dir=exp_dir,
            search_mode=search_mode,
            random_sample_size=random_sample_size
        )
        print(esn_message)
        
    else:
        # Use orchestrator for GNM experiments
        orchestrator = GNMandESNPipelineOrchestrator(config)
        
        if exp_type == 'gnm_sweep':
            results = orchestrator.run_gnm_parameter_sweep(
                experiment_name=exp_args.get('experiment_name'),
                no_wandb=exp_args.get('no_wandb', True),
                random_sample=exp_args.get('random_sample', False),
                n_random_samples=exp_args.get('n_random_samples', 30)
            )
            
        elif exp_type == 'gnm_comprehensive':
            results = orchestrator.run_gnm_comprehensive_analysis(
                experiment_name=exp_args.get('experiment_name'),
                compare_rules=exp_args.get('compare_rules', True),
                fit_weights=exp_args.get('fit_weights', True)
            )
            
        elif exp_type == 'full_pipeline':
            results = orchestrator.run_full_pipeline(
                esn_experiment_name=exp_args.get('experiment_name'),
                gnm_experiment_name=exp_args.get('experiment_name')
            )
            
        elif exp_type == 'gnm_esn_grid':
            # Extract additional parameters if provided
            gnm_cfg = loader.config.get('gnm', {})
            results = orchestrator.run_gnm_esn_grid_evaluation(
                experiment_name=exp_args.get('experiment_name'),
                eta_range=gnm_cfg.get('eta_range', (-8, 0)),
                gamma_range=gnm_cfg.get('gamma_range', (0.2, 8)),
                n_eta=gnm_cfg.get('n_eta', 20),
                n_gamma=gnm_cfg.get('n_gamma', 20)
            )
        
        
        print("\nExperiment completed!")
        
        # Finalize logger
        orchestrator.logger.finalize()


def main():
    """Command-line interface for YAML-based configuration."""
    parser = argparse.ArgumentParser(
        description="Run GNM-ESN pipeline from YAML configuration"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the configuration without running"
    )
    
    args = parser.parse_args()
    
    try:
        # Load and validate config
        loader = YAMLConfigLoader(args.config)
        print(f"✓ Configuration loaded from: {args.config}")
        print(f"✓ Experiment type: {loader.config['experiment']['type']}")
        
        if args.validate_only:
            # Create config to validate it can be built
            config = loader.create_config_manager()
            print("✓ Configuration is valid")
            print(f"  - Data resolution: {config.data.resolution}")
            print(f"  - Densities: {config.data.densities}")
            if loader.config['experiment']['type'] in ['gnm_sweep', 'gnm_comprehensive']:
                print(f"  - GNM eta range: {config.gnm.eta_range}")
                print(f"  - GNM gamma range: {config.gnm.gamma_range}")
                print(f"  - Wiring rules: {config.gnm.generative_rules_to_test}")
            sys.exit(0)
        
        # Run the experiment
        run_from_yaml(args.config)
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

    # TODO: Test ESN. Then test GNM sweep.
    # TASK 1: ESN
    # python src/connectome_analysis/main_pipeline.py configs/example_esn.yaml
    # -> Testing ESN worked. Did this in "/Users/adrian/Documents/01_projects/14_4D_lab/output/esn/my_gnm_experiment/my_gnm_experiment_20250902_175257"
    
    # TASK 2: GNM sweep
    # python src/connectome_analysis/main_pipeline.py configs/example_gnm.yaml
    
# TODO: Try out the next few lines. 
# # Validate config without running
# python yaml_config.py example_config.yaml --validate-only

# # Or still use the original command-line interface
# python main_pipeline_2.py gnm_sweep --experiment-name test


# THIS IS THE COMMAND TO RUN THE GNM SWEEP EXPERIMENT THAT I AM CURRENTLY USING

# python src/connectome_analysis/main_pipeline.py configs/gnm_sweep.yaml