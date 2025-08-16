"""
Example run scripts demonstrating how to use the modular connectome analysis pipeline.
These scripts replace your original two scripts with cleaner, more modular versions.
"""

# ==============================================================================
# SCRIPT 1: ESN Hyperparameter Sweep (replaces your Script No 2)
# ==============================================================================

def run_esn_hyperparameter_sweep_example():
    """
    Example script for running ESN hyperparameter sweeps.
    This replaces and improves upon your Script No 2.
    """
    from pathlib import Path
    import numpy as np
    from itertools import product
    
    from config import ConfigManager, ESNConfig, DataConfig, ComputeConfig
    from main_pipeline import GNMPipelineOrchestrator
    
    # Create custom configuration
    esn_config = ESNConfig(
        n_runs=10, # 100,  # High number of runs for better statistics
        input_length=4000,
        regularization_method="ridge"
    )
    
    data_config = DataConfig(
        resolution=68,
        # densities=1[0, 12, 14, 16, 18, 20],  # All available densities
        densities=[10, 16, 20],  # All available densities
        use_weighted=True
    )
    
    compute_config = ComputeConfig(
        timing_flag=True,
        random_seed=42,
        n_workers=None  # Use all available CPUs
    )
    
    config = ConfigManager(esn_config, data_config=data_config, compute_config=compute_config)
    
    # Create pipeline orchestrator
    orchestrator = GNMPipelineOrchestrator(config)

    # Define hyperparameter grid (similar to your original script)
    HP_SPECTRAL_RADII = np.linspace(0.1, 2.5, 4) # 10)  # Extended range
    HP_INPUT_LENGTHS = [1000] # , 2000, 4000, 8000]
    HP_INPUT_SCALINGS = [0.5, 1.0, 2.0]
    HP_REGULARIZATION_METHODS = ["pinv", "ridge"]
    
    print("Starting ESN hyperparameter sweep...")
    print(f"Grid size: {len(HP_SPECTRAL_RADII)} × {len(HP_INPUT_LENGTHS)} × {len(HP_INPUT_SCALINGS)} × {len(HP_REGULARIZATION_METHODS)} × {len(data_config.densities)}")
    
    # Run the sweep
    results = orchestrator.run_gnm_parameter_sweep( # OR full run? -> Previously run_esn_hyperparameter_sweep(
        densities=data_config.densities,
        spectral_radii=HP_SPECTRAL_RADII,
        input_lengths=HP_INPUT_LENGTHS,
        input_scalings=HP_INPUT_SCALINGS,
        regularization_methods=HP_REGULARIZATION_METHODS,
        n_runs_list=[esn_config.n_runs],
        experiment_name="comprehensive_esn_sweep"
    )
    
    if results["status"] == "success":
        print(f"Experiment completed successfully!")
        print(f"Results saved to: {results['experiment_dir']}")
        
        # Print best results
        analysis = results["analysis"]
        if "best_overall_mc" in analysis:
            best = analysis["best_overall_mc"]
            print(f"Best memory capacity: {best['value']:.4f}")
            best_hp = best["hyperparameters"]
            print(f"Best hyperparameters:")
            print(f"  - Density: {best_hp.get('density_percent')}%")
            print(f"  - Spectral radius: {best_hp.get('spectral_radius'):.3f}")
            print(f"  - Input length: {best_hp.get('input_length')}")
            print(f"  - Input scaling: {best_hp.get('input_scaling')}")
            print(f"  - Regularization: {best_hp.get('regularization_method')}")
    else:
        print(f"Experiment failed: {results.get('error', 'Unknown error')}")


# ==============================================================================
# SCRIPT 2: GNM Parameter Fitting (replaces part of your Script No 1)
# ==============================================================================

def run_gnm_parameter_fitting_example():
    """
    Example script for running GNM parameter fitting.
    This replaces and improves upon the GNM part of your Script No 1.
    """
    from config import ConfigManager, GNMConfig, DataConfig, ComputeConfig
    from main_pipeline import GNMPipelineOrchestrator
    
    # Create custom configuration for GNM fitting
    gnm_config = GNMConfig(
        n_eta=10, # 20,      # Reduced for quick test
        n_gamma=10, # 20,
        eta_start=-3.0,
        eta_end=0.0,
        gamma_start=0.1,
        gamma_end=0.6
    )
    
    data_config = DataConfig(
        resolution=68,
        densities=[10, 16, 20],  # Reduced for quick test
        use_weighted=True
    )
    
    compute_config = ComputeConfig(
        timing_flag=True,
        random_seed=42
    )
    
    config = ConfigManager(gnm_config=gnm_config, data_config=data_config, compute_config=compute_config)
    
    # Create pipeline orchestrator
    orchestrator = GNMPipelineOrchestrator(config)

    print("Starting GNM parameter fitting...")
    print(f"Parameter grid: {gnm_config.n_eta} eta × {gnm_config.n_gamma} gamma values")
    
    # Run GNM fitting for a representative density
    density = 16
    print(f"\nFitting GNM for density {density}%...")
    
    results = orchestrator.run_gnm_parameter_fitting(
        density=density,
        n_eta=gnm_config.n_eta,
        n_gamma=gnm_config.n_gamma,
        eta_range=(gnm_config.eta_start, gnm_config.eta_end),
        gamma_range=(gnm_config.gamma_start, gnm_config.gamma_end),
        experiment_name=f"gnm_fitting_density_{density}pct"
    )
    
    if results["status"] == "success":
        print(f"GNM fitting completed for density {density}%")
        print(f"Results saved to: {results['experiment_dir']}")
        
        # Print summary statistics
        print(f"Tested {results['n_subjects']} subjects")
        print(f"Wiring rules: {results['wiring_rules_tested']}")
    else:
        print(f"GNM fitting failed for density {density}%: {results.get('error', 'Unknown error')}")


# ==============================================================================
# SCRIPT 3: Complete Pipeline (combines both analyses)
# ==============================================================================

def run_complete_pipeline_example():
    """
    Example script for running the complete analysis pipeline.
    This provides a comprehensive analysis combining ESN evaluation and GNM fitting.
    """
    from config import get_production_config
    from main_pipeline import PipelineOrchestrator
    
    # Use production configuration
    config = get_production_config()
    orchestrator = PipelineOrchestrator(config)
    
    print("Starting complete connectome analysis pipeline...")
    
    # Run the full pipeline
    results = orchestrator.run_full_pipeline(
        esn_experiment_name="production_esn_sweep",
        gnm_experiment_name="production_gnm_fitting",
        quick_test=False  # Set to True for faster testing
    )
    
    print(f"\nPipeline completed with status: {results['status']}")
    print(f"Total elapsed time: {results['total_elapsed_time_sec']:.2f} seconds")
    
    # Print results summary
    if results["esn"] and results["esn"]["status"] == "success":
        esn_analysis = results["esn"]["analysis"]
        print(f"\nESN Results:")
        print(f"  - Best memory capacity: {esn_analysis['best_overall_mc']['value']:.4f}")
        print(f"  - Total evaluations: {esn_analysis['total_evaluations']}")
        print(f"  - Results directory: {results['esn']['experiment_dir']}")
    
    if results["gnm"] and results["gnm"]["status"] == "success":
        print(f"\nGNM Results:")
        print(f"  - Results directory: {results['gnm']['experiment_dir']}")


# ==============================================================================
# SCRIPT 4: Quick Test Script
# ==============================================================================

def run_quick_test_example():
    """
    Quick test script with reduced parameters for development and testing.
    """
    from config import get_gnm_quick_test_config
    from main_pipeline import GNMPipelineOrchestrator
    
    # Use quick test configuration
    config = get_gnm_quick_test_config()
    print("Got config successfully")
    
    try:
        orchestrator = GNMPipelineOrchestrator(config)
        print("Created GNMPipelineOrchestrator successfully")
    except Exception as e:
        print(f"Failed to create GNMPipelineOrchestrator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Running quick test with reduced parameters...")
    
    # Quick ESN test
    print("\n--- Quick ESN Test ---")
    try:
        esn_results = orchestrator.run_esn_hyperparameter_sweep(
            densities=[10],
            spectral_radii=[0.8, 0.99],
            input_lengths=[1000, 2000],
            input_scalings=[1.0],
            regularization_methods=["pinv"],
            n_runs_list=[3],
            experiment_name="quick_esn_test"
        )
        
        if esn_results["status"] == "success":
            print(f"ESN test completed: {esn_results['experiment_dir']}")
        else:
            print(f"ESN test failed: {esn_results.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"ESN test failed with exception: {e}")
    
    # Quick GNM test
    print("\n--- Quick GNM Test ---")
    try:
        gnm_results = orchestrator.run_gnm_parameter_fitting(
            density=10,
            n_eta=5,
            n_gamma=5,
            experiment_name="quick_gnm_test"
        )
        
        if gnm_results["status"] == "success":
            print(f"GNM test completed: {gnm_results['experiment_dir']}")
        else:
            print(f"GNM test failed: {gnm_results.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"GNM test failed with exception: {e}")


# ==============================================================================
# SCRIPT 5: Analysis Script for Results
# ==============================================================================

def analyze_existing_results_example():
    """
    Example script for analyzing existing results.
    """
    from pathlib import Path
    from esn_evaluation import create_esn_evaluator
    from config import ConfigManager
    
    config = ConfigManager()
    
    # Analyze ESN results
    esn_evaluator = create_esn_evaluator(config_manager=config)
    
    # Specify the directory containing your ESN results
    esn_results_dir = Path("path/to/your/esn/results")
    
    if esn_results_dir.exists():
        print("Analyzing ESN results...")
        esn_analysis = esn_evaluator.load_and_analyze_results(esn_results_dir)
        
        if "error" not in esn_analysis:
            print(f"ESN Analysis Summary:")
            print(f"  - Total evaluations: {esn_analysis['total_evaluations']}")
            print(f"  - Best memory capacity: {esn_analysis['best_overall_mc']['value']:.4f}")
            print(f"  - Memory capacity statistics:")
            mc_stats = esn_analysis['mc_statistics']
            print(f"    Mean: {mc_stats['mean']:.4f}, Std: {mc_stats['std']:.4f}")
            print(f"    Range: {mc_stats['min']:.4f} - {mc_stats['max']:.4f}")
        else:
            print(f"ESN analysis failed: {esn_analysis['error']}")
    else:
        print(f"ESN results directory not found: {esn_results_dir}")


# ==============================================================================
# SCRIPT 6: Data Summary Test
# ==============================================================================

def run_data_summary_test():
    """
    Test script to check data availability and summary.
    """
    from config import get_gnm_quick_test_config
    from main_pipeline import GNMPipelineOrchestrator

    print("Testing data summary...")
    
    try:
        config = get_gnm_quick_test_config()
        orchestrator = GNMPipelineOrchestrator(config)
        
        summary = orchestrator.get_data_summary()
        
        print("Data Summary:")
        import json
        print(json.dumps(summary, indent=2, default=str))
        
        if 'error' in summary:
            print(f"\n⚠️  Data loading issues detected: {summary['error']}")
            print("This is expected if data files don't exist yet.")
        else:
            print("\n✅ Data summary generated successfully!")
            
    except Exception as e:
        print(f"❌ Data summary test failed: {e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# Main execution examples
# ==============================================================================

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Data summary test (recommended first step)
    print("=" * 60)
    print("RUNNING DATA SUMMARY TEST")
    print("=" * 60)
    run_data_summary_test()
    
    print("\n" + "=" * 60)
    print("RUNNING QUICK TEST")
    print("=" * 60)
    # Quick test (recommended for first run)
    run_quick_test_example()
    
    # Individual components
    run_esn_hyperparameter_sweep_example()
    run_gnm_parameter_fitting_example()
    
    # Complete pipeline
    # run_complete_pipeline_example()
    
    # Analysis of existing results
    # analyze_existing_results_example()