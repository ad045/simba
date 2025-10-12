from src.config.yaml_loader import YAMLConfigLoader
from src.pipeline.gnm_and_esn_orchestrator import GNMandESNPipelineOrchestrator
import time 
import os 

def run_from_yaml(yaml_path: str):
    """Main function to run pipeline from YAML configuration."""
    
    # Load YAML config, create ConfigManager from that 
    loader = YAMLConfigLoader(yaml_path)
    config = loader.create_config_manager()
<<<<<<< HEAD
    
    # Create run_info.txt in the experiment output directory
    def save_run_info(config, yaml_path, exp_args, exp_type):
        # Determine the output directory based on experiment type
        current_time = time.strftime('%Y%m%d_%H%M%S') 
        precise_folder_name = exp_args.get('experiment_name', 'default_no_exp_name_set') + f"_{current_time}"
        if exp_type == 'gnm_sweep':
            output_dir = config.paths.gnm_output_dir / exp_args.get('experiment_name', f"gnm_{current_time}") / precise_folder_name
        elif exp_type == 'dynamic_gnm': # TODO: Change this here.
            output_dir = config.paths.dynamic_gnm_output_dir / exp_args.get('experiment_name', f"dynamic_gnm_{current_time}") / precise_folder_name
        else: 
            print(f"Unknown experiment type: '{exp_type}'.")
        # # Ensure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        config.paths.current_projects_output_dir = output_dir
        
        # Copy the original YAML config to the output directory
        # TODO: Check that there is no other run in this experiment with the same name. 
        shutil.copy(yaml_path, output_dir / "config.yaml")
        
        return output_dir
=======
>>>>>>> through_back_to_write_report_quickly

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
<<<<<<< HEAD
    
    # Import pipeline orchestrator
    from src.pipeline.gnm_and_esn_orchestrator import GNMandESNPipelineOrchestrator
    
    
    # Use orchestrator for GNM experiments
    orchestrator = GNMandESNPipelineOrchestrator(config)
    
    if exp_type == 'gnm_sweep':
        results = orchestrator.run_gnm_parameter_sweep(
            **exp_args,
        ) 
        
    elif exp_type == "dynamic_gnm":
        results = orchestrator.run_dynamic_gnm_generation(
            experiment_name=exp_args.get('experiment_name')
=======

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
>>>>>>> through_back_to_write_report_quickly
        )
    
    print("\nExperiment completed!")
    
<<<<<<< HEAD
    # Finalize logger
    orchestrator.logger.finalize(config.paths.current_projects_output_dir)
=======
    # TODO: Add logger again
    # Finalize logger
    # orchestrator.logger.finalize(config['paths']['output_experiment_dir'])
    #                                              # output_gnm_dir'] / config['experiment']['name'])
>>>>>>> through_back_to_write_report_quickly

