from src.config.yaml_loader import YAMLConfigLoader
from src.pipeline.gnm_and_esn_orchestrator import GNMandESNPipelineOrchestrator

def run_from_yaml(yaml_path: str):
    """Main function to run pipeline from YAML configuration."""
    
    # Load YAML config, create ConfigManager from that 
    loader = YAMLConfigLoader(yaml_path)
    config = loader.create_config_manager()

    # Get experiment type and arguments
    exp_type = loader.config['experiment']['experiment_type']
    # exp_args = loader.get_experiment_args()

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
    
    if exp_type == 'gnm_sweep':
        results = orchestrator.run_gnm_parameter_sweep(
            **config,
        ) 
    elif exp_type == "dynamic_gnm":
        results = orchestrator.run_dynamic_gnm_generation(
            experiment_name=config.get('experiment_name')
        )
    
    print("\nExperiment completed!")
    
    # Finalize logger
    orchestrator.logger.finalize(config.paths.current_projects_output_dir)

