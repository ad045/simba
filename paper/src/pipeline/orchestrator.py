from src.config.yaml_loader import YAMLConfigLoader
from src.pipeline.gnm_and_esn_orchestrator import GNMandESNPipelineOrchestrator
import time 
import os 

def run_from_yaml(yaml_path: str):
    """Main function to run pipeline from YAML configuration."""
    
    # Load YAML config, create ConfigManager from that 
    loader = YAMLConfigLoader(yaml_path)
    config = loader.create_config_manager()

    # Get experiment type and arguments
    exp_type = loader.config['experiment']['experiment_type']

    # Print configuration summary
    print("=" * 60)
    print(f"RUNNING EXPERIMENT FROM YAML: {yaml_path}")
    print("=" * 60)
    print(f"Experiment Type: {exp_type}")
    if 'experiment_name' in config:
        print(f"Experiment Name: {config['experiment_name']}")
    if 'search' in loader.config['experiment']:
        print(f"Search Method: {loader.config['experiment']['search'].get('method', 'default')}")
    print("-" * 60)

    # Use orchestrator for GNM experiments
    orchestrator = GNMandESNPipelineOrchestrator(config)
    
    # Update the ending of the experiment folder to make it unique
    # Get the current path
    current_path = config['paths']['output_experiment_dir']
    
    # Get the parent directory and the leaf folder name
    parent_dir = os.path.dirname(current_path)
    leaf_folder = os.path.basename(current_path)
    
    # Append timestamp to the leaf folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    new_leaf_folder = f"{leaf_folder}_{timestamp}"
    
    # Combine back together
    config['paths']['output_experiment_dir'] = os.path.join(parent_dir, new_leaf_folder)


    # config.update(
    if exp_type == 'gnm_sweep':
        results = orchestrator.run_gnm_parameter_sweep(
            # **config,
            config, 
        ) 
    elif exp_type == "dynamic_gnm":
        results = orchestrator.run_dynamic_gnm_generation(
            experiment_name=config.get('experiment_name')
        )
    
    print("\nExperiment completed!")
    
    # TODO: Add logger again
    # Finalize logger
    from pathlib import Path 
    orchestrator.logger.finalize(Path(parent_dir) / leaf_folder) 

