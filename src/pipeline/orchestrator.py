
import time
import shutil

from src.config.yaml_loader import YAMLConfigLoader


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
        elif exp_type == 'gnm' or "gnm_sweep":
            output_dir = config.paths.gnm_output_dir / exp_args.get('experiment_name', f"gnm_{current_time}") / precise_folder_name
        elif exp_type == 'dynamic_gnm': # TODO: Change this here.
            output_dir = config.paths.dynamic_gnm_output_dir / exp_args.get('experiment_name', f"dynamic_gnm_{current_time}") / precise_folder_name
        else: 
            print(f"Unknown experiment type: '{exp_type}'.")
        # # Ensure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        config.paths.current_projects_output_dir = output_dir
        
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
    from src.pipeline.gnm_and_esn_orchestrator import GNMandESNPipelineOrchestrator
    
    # Run appropriate experiment type
    if exp_type == 'esn':
        # Special handling for ESN
        from src.ESNs.esn_evaluation import create_esn_evaluator
        from src.utils.data_loader import DataLoader
        
        esn_evaluator = create_esn_evaluator(config_manager=config)
        
        # Generate hyperparameter grid from YAML config
        hparam_grid = loader.generate_esn_hparam_grid(config)
        
        # Load data
        data_loader = DataLoader(config)
        weighted_by_density = data_loader.load_weighted_by_density()
        
        # Run ESN sweep
        # exp_name = exp_args.get('experiment_name', f"esn_{time.strftime('%Y%m%d_%H%M%S')}")
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
                **exp_args,
            ) 
            
        elif exp_type == "dynamic_gnm":
            results = orchestrator.run_dynamic_gnm_generation(
                experiment_name=exp_args.get('experiment_name')
            )
        
        print("\nExperiment completed!")
        
        # Finalize logger
        orchestrator.logger.finalize(config.paths.current_projects_output_dir)

