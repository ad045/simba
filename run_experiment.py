"""
YAML-based configuration system integrated with existing ConfigManager.
This bridges YAML files with the existing config.py structure.
"""

import argparse
import sys

from src.config.yaml_loader import YAMLConfigLoader
from src.pipeline.orchestrator import run_from_yaml #


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