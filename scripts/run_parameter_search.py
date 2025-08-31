import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config

from src.models.esn.memory_capacity import evaluate_memory_capacity
from src.models.gnm.generator import generate_gnm # generate_network_gnm generate_gnm

def evaluate_params(params_dict, base_config):
    """Evaluate a single parameter combination."""
    # Create a config with these specific parameters
    config = Config(base_config)
    
    # Override with search parameters
    for key, value in params_dict.items():
        config.set(key, value)  # You'd need to implement set() method
    
    # Run your evaluation
    try:
        # Generate GNM
        gnm = generate_gnm(
            eta=config.get('gnm.eta'),
            gamma=config.get('gnm.gamma'),
            density=config.get('data.density_threshold')
        )
        
        # Evaluate with ESN
        memory_capacity = evaluate_memory_capacity(
            network=gnm,
            spectral_radius=config.get('esn.spectral_radius'),
            input_scaling=config.get('esn.input_scaling'),
            regularization_alpha=config.get('esn.regularization_alpha')
        )
        
        return {
            'params': params_dict,
            'memory_capacity': memory_capacity,
            'status': 'success'
        }
    except Exception as e:
        return {
            'params': params_dict,
            'error': str(e),
            'status': 'failed'
        }

def main(config_path: str):
    config = Config(config_path)
    search_config = config._config.get('search', {})
    
    # Get base config path
    base_config = config._config.get('base_config', 'configs/default.yaml')
    
    # Get all parameter combinations
    param_generator = config.get_search_params()
    param_list = list(param_generator)
    
    print(f"Running {search_config['type']} search with {len(param_list)} combinations")
    
    # Run search
    results = []
    n_jobs = search_config.get('n_jobs', 1)
    
    if n_jobs == 1:
        # Sequential execution
        for params in tqdm(param_list):
            result = evaluate_params(params, base_config)
            results.append(result)
    else:
        # Parallel execution
        if n_jobs == -1:
            n_jobs = None  # Use all available cores
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(evaluate_params, params, base_config)
                for params in param_list
            ]
            
            for future in tqdm(futures):
                results.append(future.result())
    
    # Save results
    save_path = Path(search_config.get('save_results', 'search_results.pkl'))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results)
    successful = df_results[df_results['status'] == 'success']
    
    if not successful.empty:
        best_idx = successful['memory_capacity'].idxmax()
        best_result = successful.loc[best_idx]
        
        print(f"\nBest parameters found:")
        for key, value in best_result['params'].items():
            print(f"  {key}: {value}")
        print(f"Best memory capacity: {best_result['memory_capacity']:.4f}")
    
    # Save summary
    df_results.to_csv(save_path.with_suffix('.csv'), index=False)
    print(f"\nResults saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/gridsearch.yaml',
        help='Path to search config file'
    )
    args = parser.parse_args()
    
    main(args.config)