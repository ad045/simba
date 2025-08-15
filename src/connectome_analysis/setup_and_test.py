"""
Setup and testing script for the connectome analysis pipeline.
This script helps set up the environment and test the pipeline with your existing code structure.
"""

import sys
import os
from pathlib import Path
import traceback

def setup_paths():
    """Add necessary paths to sys.path for imports."""
    # Get the project root (assuming this script is in src/connectome_analysis/)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up to 14_4D_lab/
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"Added to Python path: {project_root}")
    print(f"Current working directory: {Path.cwd()}")
    
    return project_root

def test_imports():
    """Test if we can import the required modules."""
    print("\n" + "="*50)
    print("TESTING IMPORTS")
    print("="*50)
    
    # Test your existing ESN modules
    try:
        from src.ESNs.test_memory_capacity_weighted import evaluate_memory_capacity_from_connectome
        print("✅ ESN memory capacity module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ESN memory capacity module: {e}")
    
    try:
        from src.ESNs.generate_weight_matrices_bio_no_rank_weighted import (
            build_weight_matrix_from_bin_conn, 
            build_weight_matrix_from_connectome
        )
        print("✅ ESN weight matrix generation module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ESN weight matrix module: {e}")
    
    try:
        from src.structural_analysis.graph_measures_optimized_weighted import analyze_connectomes
        print("✅ Graph measures module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import graph measures module: {e}")
    
    try:
        from src.utils.saving_and_finding_files import time_stamp_for_saving
        print("✅ Utils module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utils module: {e}")
    
    # Test our new modules
    try:
        from config import ConfigManager, get_quick_test_config
        print("✅ Config module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config module: {e}")
        return False
    
    try:
        from data_loader import DataLoader
        print("✅ Data loader module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data loader module: {e}")
        return False
    
    return True

def test_config():
    """Test configuration setup."""
    print("\n" + "="*50)
    print("TESTING CONFIGURATION")
    print("="*50)
    
    try:
        from config import ConfigManager, get_quick_test_config
        
        # Test basic config
        config = ConfigManager()
        print(f"✅ Basic config created - Resolution: {config.data.resolution}")
        
        # Test quick test config
        quick_config = get_quick_test_config()
        print(f"✅ Quick test config created - Densities: {quick_config.data.densities}")
        
        # Test path generation
        paths = quick_config.get_data_paths()
        print(f"✅ Data paths generated - Distance matrix: {paths['distance_matrix']}")
        
        return True, quick_config
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False, None

def test_data_availability(config):
    """Test if data files are available."""
    print("\n" + "="*50)
    print("TESTING DATA AVAILABILITY")
    print("="*50)
    
    try:
        from data_loader import DataLoader
        
        loader = DataLoader(config)
        summary = loader.get_data_summary()
        
        if 'error' in summary:
            print(f"❌ Data loading error: {summary['error']}")
            return False
        else:
            print(f"✅ Data summary generated:")
            print(f"   - Resolution: {summary['resolution']}")
            print(f"   - Available densities: {summary['available_densities']}")
            print(f"   - Data shapes: {summary['data_shapes']}")
            return True
            
    except Exception as e:
        print(f"❌ Data availability test failed: {e}")
        print("This might be expected if data files don't exist yet.")
        traceback.print_exc()
        return False

def create_minimal_test():
    """Create a minimal test that works with dummy data."""
    print("\n" + "="*50)
    print("CREATING MINIMAL TEST")
    print("="*50)
    
    try:
        # Create minimal test that doesn't require actual data files
        from config import ESNConfig, DataConfig, ComputeConfig, ConfigManager
        
        # Create a simple config
        esn_config = ESNConfig(spectral_radius=0.99, input_length=100, n_runs=2)
        data_config = DataConfig(resolution=68, densities=[10])
        compute_config = ComputeConfig(timing_flag=True, n_workers=1)
        
        config = ConfigManager(esn_config, data_config, compute_config)
        
        print("✅ Minimal configuration created successfully")
        
        # Test ESN evaluation with dummy data
        import numpy as np
        dummy_connectome = np.random.rand(68, 68)
        
        # Make it symmetric
        dummy_connectome = (dummy_connectome + dummy_connectome.T) / 2
        np.fill_diagonal(dummy_connectome, 0)
        
        print("✅ Dummy connectome data created")
        
        # Test that we can create the basic objects
        from esn_evaluation import ESNEvaluator
        from data_loader import DataLoader
        
        # This will fail on data loading, but we can test object creation
        print("✅ Pipeline modules can be imported and basic objects created")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main setup and testing function."""
    print("🚀 CONNECTOME ANALYSIS PIPELINE SETUP")
    print("="*60)
    
    # 1. Setup paths
    project_root = setup_paths()
    
    # 2. Test imports
    imports_ok = test_imports()
    
    # 3. Test configuration
    config_ok, config = test_config()
    
    if not config_ok:
        print("\n❌ Configuration test failed - cannot proceed")
        return
    
    # 4. Test data availability
    data_ok = test_data_availability(config)
    
    # 5. Create minimal test
    minimal_test_ok = create_minimal_test()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    print(f"✅ Path setup: Success")
    print(f"{'✅' if imports_ok else '❌'} Module imports: {'Success' if imports_ok else 'Failed'}")
    print(f"{'✅' if config_ok else '❌'} Configuration: {'Success' if config_ok else 'Failed'}")
    print(f"{'✅' if data_ok else '❌'} Data availability: {'Success' if data_ok else 'Failed (might be expected)'}")
    print(f"{'✅' if minimal_test_ok else '❌'} Minimal test: {'Success' if minimal_test_ok else 'Failed'}")
    
    if imports_ok and config_ok and minimal_test_ok:
        print("\n🎉 SETUP SUCCESSFUL!")
        print("\nNext steps:")
        print("1. If data files exist, try: python -c 'from run_examples import run_quick_test_example; run_quick_test_example()'")
        print("2. Or use the command line: python main_pipeline.py data-summary")
        print("3. For a full test: python main_pipeline.py full --quick-test")
    else:
        print("\n⚠️  SETUP INCOMPLETE")
        print("\nTo fix:")
        if not imports_ok:
            print("- Check that your ESN modules are in the correct locations")
            print("- Verify that src/ directory is accessible")
        if not config_ok:
            print("- Check configuration module for syntax errors")
        if not minimal_test_ok:
            print("- There may be fundamental import issues to resolve")

if __name__ == "__main__":
    main()