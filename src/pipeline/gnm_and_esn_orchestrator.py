"""
Main pipeline fully integrated with GNM library and centralized logging.
This version is updated for robust, interrupt-safe multiprocessing on macOS.
"""

import argparse
import sys
from typing import Optional, Dict, Any
import json
import time
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import uuid
import os
from pathlib import Path

import pickle 

# For parallel processing
from joblib import Parallel, delayed
from tqdm import tqdm

# GNM library imports
from gnm import fitting
# from src.imported_libraries.GenerativeNetworkModels_2.src.gnm.model import GenerativeNetworkModel, BinaryGenerativeParameters

# Import our optimized modules
from src.config.ESN_and_GNM_config import ConfigManager
from src.GNMs.gnm_network_generator import GNMGenerator
from src.utils.data_loader import DataLoader
# from src.ESNs.alternative_esn_evaluation import ESNEvaluator
# from src.ESNs.esn_evaluation import ESNEvaluator
from src.utils.run_logger import get_logger

from src.structural_analysis.graph_measures import analyze_connectomes
# from src.ESNs.esn_evaluation import evaluate_memory_capacity_from_connectome
from src.ESNs.alternative_esn_evaluation import evaluate_memory_capacity_from_connectome

# --- Helper function for interrupt-safe parallel execution ---

def _run_and_save_single_simulation(task_data: dict, 
                                    evaluation_criteria, 
                                    target_network, 
                                    elaborate_analysis,
                                    device_str: str, 
                                    output_dir: Path, temp_dir: str, 
                                    h_params: dict): # esn_params: dict):
    """
    Worker function that reconstructs objects from simple data before running the simulation.
    """
    from gnm.fitting import perform_run, RunConfig # , BinaryGenerativeParameters
    from gnm import generative_rules # Important import for reconstruction
    # from gnm.fitting import BinarySweepParameters
    from gnm.model import BinaryGenerativeParameters
    # --- Reconstruct the RunConfig object from the dictionary ---
    bp_data = task_data['binary_parameters']
    
    # Get the rule class from the gnm library using its name
    RuleClass = getattr(generative_rules, bp_data['generative_rule_name'])
    
    binary_params = BinaryGenerativeParameters( # BinaryGenerativeParameters(
        eta=bp_data['eta'],
        gamma=bp_data['gamma'],
        lambdah=bp_data['lambdah'], 
        heterochronicity_relationship_type=bp_data['heterochronicity_relationship_type'],
        generative_rule=RuleClass(),
        num_iterations=bp_data['num_iterations'],
        distance_relationship_type=bp_data['distance_relationship_type'],
        preferential_relationship_type=bp_data['preferential_relationship_type']
    )
    
    run_config = RunConfig(
        binary_parameters=binary_params,
        distance_matrix=task_data['distance_matrix']
    )
    # --- End Reconstruction ---

    flat_record = {}
    try:
        # 1. Run the simulation with the newly reconstructed run_config
        experiment = perform_run(
            run_config=run_config,
            binary_evaluations=[evaluation_criteria],
            real_binary_matrices=target_network.unsqueeze(0),
            save_model=True,
            save_run_history=False,
            device=torch.device(device_str),
        )

        # 2. Prepare the data record for this iteration
        params = experiment.run_config.binary_parameters
        flat_record.update({
            "eta": float(params.eta), "gamma": float(params.gamma),
            "distance_relationship_type": str(params.distance_relationship_type),
            "preferential_relationship_type": str(params.preferential_relationship_type),
            "generative_rule": str(params.generative_rule.__class__.__name__),
            "num_iterations": int(params.num_iterations),
        })

        energy_metric_name = list(experiment.evaluation_results.binary_evaluations.keys())[0]
        energy_value_mean = experiment.evaluation_results.binary_evaluations[energy_metric_name].mean().item() # check if these values make sense 
        flat_record.update({energy_metric_name: energy_value_mean})

        # 3. If elaborate_analysis is true, run detailed analysis
        if elaborate_analysis and experiment.model:
            networks_np = experiment.model.adjacency_matrix.cpu().numpy()

            rule_name = params.generative_rule.__class__.__name__
            filename = f"net_eta{params.eta.item():.3f}_gamma{params.gamma.item():.3f}_rule{rule_name}.npy"
            save_path = output_dir / "generated_networks" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, networks_np)

            graph_measures_list = analyze_connectomes(
                connectomes=networks_np, distance_matrix=run_config.distance_matrix.cpu().numpy()
            )
            flat_record.update(pd.DataFrame(graph_measures_list).mean().to_dict())

            # try:
            #     mc_lags_to_calc = esn_params.get('mc_lags_to_calc', [1, 5, 10, 20, 50])
            #     esn_results = [evaluate_memory_capacity_from_connectome(
            #         connectome=net, mc_lengths=mc_lags_to_calc, **esn_params.get('esn_eval_params', {})
            #     ) for net in networks_np] # THIS ONE...
            #     df_esn = pd.DataFrame(esn_results)
            #     numeric_cols = [c for c in df_esn.columns if 'mc' in c and 'indiv' not in c]
            #     # flat_record.update(df_esn[numeric_cols].mean().to_dict())
            #     flat_record = flat_record | df_esn[numeric_cols].mean().to_dict()
            # except Exception as e:
            #     mc_keys = ["mc_mean", "mc_std"] + [f"mc_lag_{l}" for l in mc_lags_to_calc]
            #     flat_record.update({key: np.nan for key in mc_keys})


            try:
                # mc_lags_to_calc = esn_params.get('mc_lags_to_calc', [1, 5, 10, 20, 50])
                # esn_eval_params = h_params # esn_params.get('esn_eval_params', {})
                esn_results = [evaluate_memory_capacity_from_connectome(
                    connectome=net, 
                    # mc_lengths=mc_lags_to_calc, 
                    h_params=h_params
                ) for net in networks_np]
                # df_esn = pd.DataFrame(esn_results)
                # numeric_cols = [c for c in df_esn.columns if 'mc' in c]
                # flat_record.update(df_esn[numeric_cols].mean().to_dict())
                
                # stuff s.t. hparams are not in lower level anymore 
                df_esn = pd.json_normalize(esn_results[0], sep='_')
                
                # # It works by removing all columns which are not numbers. Numeric columns will be averaged, and then included to the hparams
                numeric_cols = df_esn.select_dtypes(include=np.number).columns
                flat_record.update(df_esn[numeric_cols].mean().to_dict())
                
                # Add mc values for lags of 1, 5, 10, 20, and 50
                if "mc_values_for_indiv_lags" in df_esn.columns:
                    for i in range(50): #  [1, 2, 3, 4, 5, 6, 10, 20, 50]: # TODO: Remove hardcoding... 
                        flat_record.update({f"mc_{i}": df_esn["mc_values_for_indiv_lags"][0][i-1]}) # TODO: Check: Or always +1? 

            except Exception as e:
                mc_keys = ["mc_mean"] # + [f"mc_lag_{l}" for l in mc_lags_to_calc]
                flat_record.update({key: np.nan for key in mc_keys})
                
                
    except Exception as e:
        print(f"Error in worker process: {e}")
        flat_record.update({"error": str(e)})

    # 4. Save the result to a unique file in the temporary directory
    if flat_record:
        result_filename = f"result_{uuid.uuid4()}.csv"
        result_path = os.path.join(temp_dir, result_filename)
        pd.DataFrame([flat_record]).to_csv(result_path, index=False, na_rep="nan") # Added that missing values appear as "nan" for more clarity


class GNMandESNPipelineOrchestrator:
    """Pipeline orchestrator with integrated logging."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = torch.device(config.gnm.device)
        self.data_loader = DataLoader(config) if not config.data.use_gnm_defaults else None
        # self.esn_evaluator = ESNEvaluator(config, self.data_loader) if self.data_loader else None
        self.gnm_generator = GNMGenerator(device=self.config.gnm.device)
        self.logger = get_logger(config.paths.output_dir)
        self.config.paths.esn_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.paths.gnm_output_dir.mkdir(parents=True, exist_ok=True)
    
    
    def run_gnm_parameter_sweep(self,
                           target_network: Optional[torch.Tensor] = None,
                           experiment_name: Optional[str] = None, 
                           no_wandb: Optional[bool] = False,
                           random_sample: bool = False,
                           n_random_samples: int = 30, 
                           elaborate_analysis: Optional[bool] = False, 
                           **kwargs) -> Dict[str, Any]:
        """
        Run a parameter sweep in parallel with robust, interrupt-safe saving.
        """
        
        print("=" * 60)
        print("GNM PARAMETER SWEEP")
        print("=" * 60)
        
        if not experiment_name:
            experiment_name = f"gnm_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # --- Data Loading ---
        if target_network is None:
            if self.config.data.use_gnm_defaults:
                data = self.config.load_gnm_defaults()
                target_network = data["binary_network"]
                distance_matrix = data["distance_matrix"]
            else: # Should always use this case, actually 
                distance_matrix = torch.tensor(
                    self.data_loader.load_distance_matrix(), 
                    dtype=torch.float32,
                    device=self.device
                )
                binary_connectomes = self.data_loader.load_binary_connectomes()
                first_density = sorted(binary_connectomes.keys())[0]
                target_network = torch.tensor(
                    binary_connectomes[first_density][0, :, :], # TODO: So far, we only look at the first network.... 
                    dtype=torch.float32,
                    device=self.device
                )
        else:
            if self.config.data.use_gnm_defaults:
                distance_matrix = self.config.load_gnm_defaults()["distance_matrix"]
            else: # Or alternatively this case 
                 distance_matrix = torch.tensor(
                    self.data_loader.load_distance_matrix(),
                    dtype=torch.float32,
                    device=self.device
                )

        num_iterations = int(target_network.sum().item() // 2)
        num_simulations = 100
        
        # Create sweep config 
        if no_wandb and random_sample:
            sweep_config = self.config.create_gnm_random_sweep_config(
                distance_matrix=distance_matrix, num_iterations=num_iterations,
                num_simulations=num_simulations, method="random",
                n_random_samples=n_random_samples, include_weights=True
            )
        else:
            sweep_config = self.config.create_gnm_sweep_config(
                distance_matrix=distance_matrix, num_iterations=num_iterations,
                num_simulations=num_simulations, method="grid", 
                include_weights=True
            )
        
        evaluation_criteria = self.config.get_gnm_evaluation_criteria(distance_matrix)
        
            
        # IS THIS UNNECESSARY?? H_PARAMS. check which parameters get through until here... 
        h_params = {"spectral_radius": self.config.esn.spectral_radius,  #### TODO: HAND A BIG HPARAMS DICT TO THIS FUNCTION!! (instead of doing it all one by one...)
                    "n_lags": self.config.esn.n_lags,
                    "train_len": self.config.esn.input_length, # seems to have gotten two names... train_len,
                    "test_len": self.config.esn.test_len,
                    "n_runs": self.config.esn.n_runs,
                    "input_scaling": self.config.esn.input_scaling, 
                    "regression_method": self.config.esn.regularization_method, 
                    "n_transient": self.config.esn.n_transient,
                    "leak_rate": self.config.esn.leak_rate, 
                    "bias": self.config.esn.bias,  
                    "random_state": self.config.compute.random_seed,     
        }
                    
        # Pack the necessary ESN parameters into a picklable dictionary
        # esn_params = {
        #     # 'mc_lags_to_calc': [], # self.config.esn.mc_lags_to_calc,
        #     'esn_eval_params': 
        #         # {
        #         # 'train_len': self.config.esn.input_length,
        #         # 'n_runs': 5, # prob. self.config.esn.n_runs ? 
        #         # 'spectral_radius': self.config.esn.spectral_radius, 
        #         ### add more: 
        #     }
        # }
          
        # [FIX] Convert the generator to a list *before* the parallel call.
        # This resolves the serialization error by ensuring a simple, picklable list
        # is passed to the workers, not a complex generator object.
        print("Generating sweep configurations...")
        sweep_config_list = list(sweep_config)
        print(f"{len(sweep_config_list)} configurations generated.")
          
        temp_results_dir = self.config.paths.current_projects_output_dir / f"{experiment_name}_temp"
        os.makedirs(temp_results_dir, exist_ok=True)
        
        print("Generating and deconstructing sweep configurations for parallel processing...")
        
        # Create a list of simple, picklable dictionaries instead of complex objects -> joblib otherwise gets issues regarding "unable to serialize" 
        deconstructed_tasks = []
        for run_config in sweep_config_list:
            task = {
                "binary_parameters": {
                    "eta": run_config.binary_parameters.eta,
                    "gamma": run_config.binary_parameters.gamma,
                    "lambdah": run_config.binary_parameters.lambdah,
                    "heterochronicity_relationship_type": str(run_config.binary_parameters.heterochronicity_relationship_type),
                    "generative_rule_name": run_config.binary_parameters.generative_rule.__class__.__name__,
                    "num_iterations": run_config.binary_parameters.num_iterations,
                    "distance_relationship_type": str(run_config.binary_parameters.distance_relationship_type),
                    "preferential_relationship_type": str(run_config.binary_parameters.preferential_relationship_type),
                },
                "distance_matrix": run_config.distance_matrix
            }
            deconstructed_tasks.append(task)

        try:
            # Set n_jobs back to -1 for parallel execution
            Parallel(n_jobs=-1)(
                delayed(_run_and_save_single_simulation)(
                    task_data, # Pass the deconstructed dictionary
                    evaluation_criteria,
                    target_network,
                    elaborate_analysis,
                    self.config.gnm.device,
                    self.config.paths.current_projects_output_dir,
                    temp_results_dir,
                    h_params
                    # esn_params
                )
                for task_data in tqdm(deconstructed_tasks, desc="Configuration Iterations")
            )

        except (KeyboardInterrupt, Exception) as e:
            print(f"\n--- Process interrupted or failed: {e} ---")

        finally:
            print("\nCombining results...")
            all_result_files = [os.path.join(temp_results_dir, f) for f in os.listdir(temp_results_dir) if f.endswith('.csv')]
            
            if all_result_files:
                df_list = [pd.read_csv(f) for f in all_result_files]
                full_results_df = pd.concat(df_list, ignore_index=True)
                csv_path = self.config.paths.current_projects_output_dir / f"{experiment_name}_results.csv"
                full_results_df.to_csv(csv_path, index=False)
                
                 # --- 4. (Optional) Clean up temporary files ---
                for f in all_result_files:
                    os.remove(f)
                os.rmdir(temp_results_dir)
                
                print(f"Sweep finished. {len(full_results_df)} results saved to: {csv_path}")
            else:
                print("No results were generated.")
                

        return {"status": "completed"}


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description="GNM-Optimized Connectome Analysis Pipeline")
    parser.add_argument("command", choices=["sweep", "esn", "gnm_esn_grid"],
                   help="Command to run: 'sweep' for GNM parameter sweep, 'esn' for ESN analysis, 'gnm_esn_grid' for grid evaluation.")
    
    parser.add_argument("--experiment-name", help="Custom experiment name for the run.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable weights and biases logging.")
    
    # GNM-specific arguments
    parser.add_argument("--random-sample", action="store_true",
                    help="Use random sampling for GNM sweep instead of grid search (only if wandb is disabled).")
    parser.add_argument("--n-random-samples", type=int, default=30,
                    help="Number of random samples for GNM parameter sweep.")

    # ESN-specific arguments
    parser.add_argument("--esn-search-mode", choices=["grid", "random_sample"], default="random_sample",
                    help="Search mode for ESN hyperparameter sweep.")
    
    args = parser.parse_args()
    
    config = ConfigManager()
        
    orchestrator = GNMandESNPipelineOrchestrator(config)
    
    try:
        if args.command == "sweep":
            orchestrator.run_gnm_parameter_sweep(
                experiment_name=args.experiment_name, 
                no_wandb=args.no_wandb,
                random_sample=args.random_sample,
                n_random_samples=args.n_random_samples
            )
        elif args.command == "esn":
            orchestrator.run_esn_only_analysis(
                experiment_name=args.experiment_name,
                search_mode=args.esn_search_mode
            )
        elif args.command == "gnm_esn_grid": 
            orchestrator.run_gnm_esn_grid_evaluation(
                experiment_name=args.experiment_name
            )
        
        orchestrator.logger.finalize()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        orchestrator.logger.finalize()
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()
        orchestrator.logger.finalize()
        sys.exit(1)

if __name__ == "__main__":
    main()