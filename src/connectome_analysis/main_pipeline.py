"""
Main pipeline orchestrator for the connectome analysis project.
Provides high-level interfaces for running ESN evaluation and GNM fitting experiments.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import time

# Import our modules
from config import (
    ConfigManager, ESNConfig, GNMConfig, DataConfig, ComputeConfig, PathConfig,
    get_quick_test_config, get_production_config, get_hyperparameter_sweep_config
)
from data_loader import DataLoader, create_data_loader
from esn_evaluation import ESNEvaluator, create_esn_evaluator
from gnm_generation import GNMGenerator, create_gnm_generator


class PipelineOrchestrator:
    """Main pipeline orchestrator that coordinates all components."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.data_loader = DataLoader(config)
        self.esn_evaluator = ESNEvaluator(config, self.data_loader)
        self.gnm_generator = GNMGenerator(config, self.data_loader)
        
        # Ensure output directories exist
        self.config.paths.esn_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.paths.gnm_output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_esn_hyperparameter_sweep(self,
                                   densities: Optional[List[int]] = None,
                                   spectral_radii: Optional[List[float]] = None,
                                   input_lengths: Optional[List[int]] = None,
                                   input_scalings: Optional[List[float]] = None,
                                   regularization_methods: Optional[List[str]] = None,
                                   n_runs_list: Optional[List[int]] = None,
                                   search_mode: str = "grid",
                                   random_sample_size: Optional[int] = None,
                                   experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run ESN hyperparameter sweep experiment.
        
        Args:
            densities: List of connectivity densities to test
            spectral_radii: List of spectral radius values
            input_lengths: List of input sequence lengths
            input_scalings: List of input scaling factors
            regularization_methods: List of regularization methods
            n_runs_list: List of number of runs per evaluation
            search_mode: "grid" or "random_sample"
            random_sample_size: Number of random samples if using random search
            experiment_name: Custom name for the experiment
            
        Returns:
            Dictionary with experiment results and metadata
        """
        print("=" * 60)
        print("STARTING ESN HYPERPARAMETER SWEEP")
        print("=" * 60)
        
        # Generate hyperparameter grid
        hparam_grid = self.config.generate_esn_hparam_grid(
            spectral_radii=spectral_radii,
            input_lengths=input_lengths,
            input_scalings=input_scalings,
            regularization_methods=regularization_methods,
            n_runs_list=n_runs_list,
            densities=densities or self.config.data.densities
        )
        
        print(f"Generated hyperparameter grid with {len(hparam_grid)} combinations")
        if len(hparam_grid) > 0:
            print(f"Sample combination: {hparam_grid[0]}")
        
        # Load connectome data
        print("Loading connectome data...")
        try:
            weighted_by_density = self.data_loader.load_weighted_by_density()
            print(f"Loaded connectomes for densities: {sorted(weighted_by_density.keys())}")
            
            example_conn = next(iter(weighted_by_density.values()))
            print(f"Connectome dimensions: {example_conn.shape}")
            
        except Exception as e:
            print(f"Failed to load connectome data: {e}")
            return {"status": f"Error. Data loading failed: {e}"}
        
        # Create experiment directory
        if experiment_name:
            exp_dir = self.config.paths.esn_output_dir / experiment_name
        else:
            resolution = self.config.data.resolution
            timestamp = self.esn_evaluator.timestamp
            exp_dir = self.config.paths.esn_output_dir / f"esn_sweep_resolution{resolution}_{timestamp}"
        
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / "experiment_config.json"
        self.config.save_config(config_path)
        print(f"Saved experiment configuration to {config_path}")
        
        # Run the sweep
        print(f"Starting hyperparameter sweep with {len(hparam_grid)} combinations...")
        start_time = time.time()
        
        try:
            result_message = self.esn_evaluator.run_hyperparameter_sweep(
                connectomes=weighted_by_density,
                hparam_grid=hparam_grid,
                save_dir=exp_dir,
                search_mode=search_mode,
                random_sample_size=random_sample_size
            )
            
            elapsed_time = time.time() - start_time
            print(f"Experiment completed in {elapsed_time:.2f} seconds")
            print(result_message)
            
            # Analyze results
            print("Analyzing results...")
            analysis = self.esn_evaluator.load_and_analyze_results(exp_dir)
            
            # Save analysis
            analysis_path = exp_dir / "result_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            return {
                "status": "success",
                "experiment_dir": str(exp_dir),
                "elapsed_time_sec": elapsed_time,
                "n_combinations": len(hparam_grid),
                "analysis": analysis,
                "result_message": result_message
            }
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "experiment_dir": str(exp_dir),
                "elapsed_time_sec": time.time() - start_time
            }
    
    def run_gnm_parameter_fitting(self,
                                 density: Optional[int] = None,
                                 n_eta: Optional[int] = None,
                                 n_gamma: Optional[int] = None,
                                 eta_range: Optional[tuple] = None,
                                 gamma_range: Optional[tuple] = None,
                                 include_subset: Optional[bool] = None,
                                 subset_size: Optional[int] = None,
                                 esn_use_observed_weights: bool = True,
                                 experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run GNM parameter fitting experiment.
        
        Args:
            density: Single density percentage to use for fitting
            n_eta: Number of eta values in grid
            n_gamma: Number of gamma values in grid
            eta_range: Tuple of (start, end) for eta values
            gamma_range: Tuple of (start, end) for gamma values
            include_subset: Whether to include subset evaluation
            subset_size: Size of subset grid
            esn_use_observed_weights: Whether to use observed weights for ESN evaluation
            experiment_name: Custom name for the experiment
            
        Returns:
            Dictionary with experiment results and metadata
        """
        print("=" * 60)
        print("STARTING GNM PARAMETER FITTING")
        print("=" * 60)
        
        # Override config values if provided
        if n_eta is not None:
            self.config.gnm.n_eta = n_eta
        if n_gamma is not None:
            self.config.gnm.n_gamma = n_gamma
        if eta_range is not None:
            self.config.gnm.eta_start, self.config.gnm.eta_end = eta_range
        if gamma_range is not None:
            self.config.gnm.gamma_start, self.config.gnm.gamma_end = gamma_range
        if include_subset is not None:
            self.config.gnm.include_subset = include_subset
        if subset_size is not None:
            self.config.gnm.subset_size = subset_size
        
        print(f"Parameter grid: {self.config.gnm.n_eta} eta × {self.config.gnm.n_gamma} gamma values")
        print(f"Eta range: {self.config.gnm.eta_start} to {self.config.gnm.eta_end}")
        print(f"Gamma range: {self.config.gnm.gamma_start} to {self.config.gnm.gamma_end}")
        
        # Load data
        print("Loading data...")
        try:
            distance_matrix = self.data_loader.load_distance_matrix()
            print(f"Distance matrix loaded: {distance_matrix.shape}")
            
            if density is not None:
                # Load specific density
                weighted_by_density = self.data_loader.load_weighted_by_density()
                if density not in weighted_by_density:
                    available = sorted(weighted_by_density.keys())
                    return {"error": f"Density {density}% not available. Available: {available}"}
                
                connectomes = weighted_by_density[density]
                print(f"Using density {density}%, connectome shape: {connectomes.shape}")
            else:
                # Use default density (first available)
                weighted_by_density = self.data_loader.load_weighted_by_density()
                density = sorted(weighted_by_density.keys())[0]
                connectomes = weighted_by_density[density]
                print(f"Using default density {density}%, connectome shape: {connectomes.shape}")
                
        except Exception as e:
            print(f"Failed to load data: {e}")
            return {"error": f"Data loading failed: {e}"}
        
        # Create experiment directory
        if experiment_name:
            exp_dir = self.config.paths.gnm_output_dir / experiment_name
        else:
            resolution = self.config.data.resolution
            timestamp = self.gnm_generator.timestamp
            exp_dir = self.config.paths.gnm_output_dir / f"gnm_fitting_resolution{resolution}_density{density}_{timestamp}"
        
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / "experiment_config.json"
        self.config.save_config(config_path)
        print(f"Saved experiment configuration to {config_path}")
        
        # Run GNM fitting
        print(f"Starting GNM parameter fitting for {connectomes.shape[2]} subjects...")
        start_time = time.time()
        
        try:
            results = self.gnm_generator.fit_gnm_parameters(
                connectomes=connectomes,
                distance_matrix=distance_matrix,
                save_dir=exp_dir,
                esn_use_observed_real_weights=esn_use_observed_weights
            )
            
            elapsed_time = time.time() - start_time
            print(f"GNM fitting completed in {elapsed_time:.2f} seconds")
            print(f"Results saved to {results['save_dir']}")
            
            # Analyze results
            print("Analyzing results...")
            analysis = self.gnm_generator.load_and_analyze_gnm_results(exp_dir)
            
            # Save analysis
            analysis_path = exp_dir / "result_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            return {
                "status": "success",
                "experiment_dir": str(exp_dir),
                "elapsed_time_sec": elapsed_time,
                "density_used": density,
                "n_subjects": connectomes.shape[2],
                "analysis": analysis,
                "results": {
                    "best_parameters_summary": {
                        "eta_mean": results['best']['eta'].mean(),
                        "gamma_mean": results['best']['gamma'].mean(),
                        "memory_capacity_mean": results['best']['memory_capacity'].mean()
                    }
                }
            }
            
        except Exception as e:
            print(f"GNM fitting failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "experiment_dir": str(exp_dir),
                "elapsed_time_sec": time.time() - start_time
            }
    
    def run_full_pipeline(self,
                         esn_experiment_name: Optional[str] = None,
                         gnm_experiment_name: Optional[str] = None,
                         quick_test: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline: ESN evaluation followed by GNM fitting.
        
        Args:
            esn_experiment_name: Custom name for ESN experiment
            gnm_experiment_name: Custom name for GNM experiment
            quick_test: If True, run with reduced parameters for quick testing
            
        Returns:
            Dictionary with results from both experiments
        """
        print("=" * 60)
        print("STARTING FULL PIPELINE")
        print("=" * 60)
        
        pipeline_start = time.time()
        results = {"esn": None, "gnm": None, "status": "started"}
        
        # Adjust parameters for quick test
        if quick_test:
            print("Running in quick test mode with reduced parameters")
            # Override with smaller grids
            self.config.gnm.n_eta = 10
            self.config.gnm.n_gamma = 10
            self.config.esn.n_runs = 3
        
        # Run ESN hyperparameter sweep
        print("\n" + "=" * 40)
        print("PHASE 1: ESN HYPERPARAMETER SWEEP")
        print("=" * 40)
        
        if quick_test:
            # Smaller grid for testing
            esn_results = self.run_esn_hyperparameter_sweep(
                densities=[10, 20],
                spectral_radii=[0.8, 0.99],
                input_lengths=[1000, 2000],
                input_scalings=[1.0],
                regularization_methods=["pinv"],
                n_runs_list=[3],
                experiment_name=esn_experiment_name
            )
        else:
            # Full grid
            esn_results = self.run_esn_hyperparameter_sweep(
                experiment_name=esn_experiment_name
            )
        
        results["esn"] = esn_results
        
        if esn_results["status"] != "success":
            results["status"] = "failed_at_esn"
            return results
        
        # Run GNM parameter fitting
        print("\n" + "=" * 40)
        print("PHASE 2: GNM PARAMETER FITTING")
        print("=" * 40)
        
        # Use a representative density for GNM fitting
        gnm_density = 15 if 15 in self.config.data.densities else self.config.data.densities[0]
        
        gnm_results = self.run_gnm_parameter_fitting(
            density=gnm_density,
            experiment_name=gnm_experiment_name
        )
        
        results["gnm"] = gnm_results
        
        # Final status
        if gnm_results["status"] == "success":
            results["status"] = "success"
        else:
            results["status"] = "failed_at_gnm"
        
        total_time = time.time() - pipeline_start
        results["total_elapsed_time_sec"] = total_time
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Total elapsed time: {total_time:.2f} seconds")
        print(f"ESN experiment: {esn_results['status']}")
        print(f"GNM experiment: {gnm_results['status']}")
        
        return results
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of available data."""
        return self.data_loader.get_data_summary()


def create_pipeline_orchestrator(config_type: str = "default") -> PipelineOrchestrator:
    """
    Factory function to create a PipelineOrchestrator with predefined configurations.
    
    Args:
        config_type: Type of configuration ("quick_test", "production", "hyperparameter_sweep", "default")
        
    Returns:
        PipelineOrchestrator instance
    """
    if config_type == "quick_test":
        config = get_quick_test_config()
    elif config_type == "production":
        config = get_production_config()
    elif config_type == "hyperparameter_sweep":
        config = get_hyperparameter_sweep_config()
    else:
        config = ConfigManager()
    
    return PipelineOrchestrator(config)


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description="Connectome Analysis Pipeline")
    
    # Main command
    parser.add_argument("command", choices=["esn", "gnm", "full", "data-summary"], 
                       help="Command to run")
    
    # Configuration options
    parser.add_argument("--config", choices=["default", "quick_test", "production", "hyperparameter_sweep"],
                       default="default", help="Configuration preset to use")
    parser.add_argument("--config-file", type=Path, help="Path to custom configuration JSON file")
    
    # ESN-specific options
    parser.add_argument("--esn-experiment-name", help="Custom name for ESN experiment")
    parser.add_argument("--esn-search-mode", choices=["grid", "random_sample"], 
                       default="grid", help="Search mode for ESN hyperparameters")
    parser.add_argument("--esn-random-sample-size", type=int, 
                       help="Number of random samples for ESN (if using random search)")
    
    # GNM-specific options
    parser.add_argument("--gnm-experiment-name", help="Custom name for GNM experiment")
    parser.add_argument("--gnm-density", type=int, help="Specific density percentage for GNM fitting")
    parser.add_argument("--gnm-n-eta", type=int, help="Number of eta values in GNM grid")
    parser.add_argument("--gnm-n-gamma", type=int, help="Number of gamma values in GNM grid")
    parser.add_argument("--gnm-eta-range", nargs=2, type=float, 
                       help="Eta range as two values: start end")
    parser.add_argument("--gnm-gamma-range", nargs=2, type=float,
                       help="Gamma range as two values: start end")
    
    # General options
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run with reduced parameters for quick testing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create pipeline orchestrator
    if args.config_file:
        config = ConfigManager.load_config(args.config_file)
        orchestrator = PipelineOrchestrator(config)
    else:
        orchestrator = create_pipeline_orchestrator(args.config)
    
    # Enable verbose timing if requested
    if args.verbose:
        orchestrator.config.compute.timing_flag = True
    
    try:
        if args.command == "data-summary":
            # Data summary command
            summary = orchestrator.get_data_summary()
            print("Data Summary:")
            print(json.dumps(summary, indent=2, default=str))
            
        elif args.command == "esn":
            # ESN hyperparameter sweep
            result = orchestrator.run_esn_hyperparameter_sweep(
                search_mode=args.esn_search_mode,
                random_sample_size=args.esn_random_sample_size,
                experiment_name=args.esn_experiment_name
            )
            
            if result["status"] == "success":
                print(f"ESN experiment completed successfully!")
                print(f"Results saved to: {result['experiment_dir']}")
                if "analysis" in result and "best_overall_mc" in result["analysis"]:
                    best_mc = result["analysis"]["best_overall_mc"]["value"]
                    print(f"Best memory capacity achieved: {best_mc:.4f}")
            else:
                print(f"ESN experiment failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
                
        elif args.command == "gnm":
            # GNM parameter fitting
            eta_range = tuple(args.gnm_eta_range) if args.gnm_eta_range else None
            gamma_range = tuple(args.gnm_gamma_range) if args.gnm_gamma_range else None
            
            result = orchestrator.run_gnm_parameter_fitting(
                density=args.gnm_density,
                n_eta=args.gnm_n_eta,
                n_gamma=args.gnm_n_gamma,
                eta_range=eta_range,
                gamma_range=gamma_range,
                experiment_name=args.gnm_experiment_name
            )
            
            if result["status"] == "success":
                print(f"GNM experiment completed successfully!")
                print(f"Results saved to: {result['experiment_dir']}")
                if "results" in result:
                    best_params = result["results"]["best_parameters_summary"]
                    print(f"Average best parameters: eta={best_params['eta_mean']:.3f}, "
                          f"gamma={best_params['gamma_mean']:.3f}")
                    print(f"Average memory capacity: {best_params['memory_capacity_mean']:.4f}")
            else:
                print(f"GNM experiment failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
                
        elif args.command == "full":
            # Full pipeline
            result = orchestrator.run_full_pipeline(
                esn_experiment_name=args.esn_experiment_name,
                gnm_experiment_name=args.gnm_experiment_name,
                quick_test=args.quick_test
            )
            
            if result["status"] == "success":
                print(f"Full pipeline completed successfully!")
                print(f"Total time: {result['total_elapsed_time_sec']:.2f} seconds")
                if result["esn"]["status"] == "success":
                    print(f"ESN results: {result['esn']['experiment_dir']}")
                if result["gnm"]["status"] == "success":
                    print(f"GNM results: {result['gnm']['experiment_dir']}")
            else:
                print(f"Pipeline failed at stage: {result['status']}")
                if result["esn"] and result["esn"]["status"] != "success":
                    print(f"ESN error: {result['esn'].get('error', 'Unknown')}")
                if result["gnm"] and result["gnm"]["status"] != "success":
                    print(f"GNM error: {result['gnm'].get('error', 'Unknown')}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()