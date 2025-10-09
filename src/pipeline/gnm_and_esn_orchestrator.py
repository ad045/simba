"""
Main pipeline fully integrated with GNM library and centralized logging.
This version is updated for robust, interrupt-safe multiprocessing on macOS.
"""

import argparse
import sys
from typing import Optional, Dict, Any
import time
import numpy as np
import torch
import pandas as pd
import uuid
import os
from pathlib import Path

# For parallel processing
from joblib import Parallel, delayed
from tqdm import tqdm

# Import our optimized modules
from config.manager import ConfigManager, create_gnm_random_sweep_config
from src.GNMs.gnm_network_generator import GNMGenerator
from src.utils.data_loader import DataLoader
from src.utils.run_logger import get_logger

from src.config.GNM import create_evaluation_criteria

from src.structural_analysis.graph_measures import analyze_connectomes
from src.ESNs.alternative_esn_evaluation import evaluate_memory_capacity_from_connectome

from src.utils.combine_csvs import merge_csv_files

# --- Helper function for interrupt-safe parallel execution ---

def _run_and_save_single_simulation(
                                    task_data: dict, 
                                    evaluation_criteria, 
                                    # weighted_evaluation_criteria, 
                                    target_network, # can be one or more?  
                                    individual_networks, # needs to be set: a float32 torch Tensor of shape (num_subjects, n_nodes, n_nodes)
                                    directly_compare_with_empirical_networks: bool, 
                                    elaborate_analysis: bool, 
                                    evaluate_individual_connectomes: bool, 
                                    device_str: str, 
                                    output_dir: Path, temp_dir: str, 
                                    h_params: dict): # esn_params: dict):
    """
    Worker function that reconstructs objects from simple data before running the simulation.
    """
    from gnm.fitting import perform_run, RunConfig
    from gnm import generative_rules 
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
    indiv_networks_record = {}
    
    try:
        
        # 1. Run the simulation with the newly reconstructed run_config
        if directly_compare_with_empirical_networks: 
            # if individual networks are given+
            
            individual_networks = torch.tensor(
                    individual_networks, 
                    dtype=torch.float32,
            )
                        
            experiment = perform_run(
                run_config=run_config,
                binary_evaluations=[evaluation_criteria],
                real_binary_matrices=individual_networks, 
                save_model=True,
                save_run_history=False,
                device=torch.device(device_str),
            )
        else: 
            experiment = perform_run(
                run_config=run_config,
                save_model=True,
                save_run_history=False,
                device=torch.device(device_str),
            )

        # 2. Prepare the data record for this iteration
        params = experiment.run_config.binary_parameters
        flat_record.update({
            "eta": float(params.eta), 
            "gamma": float(params.gamma),
            "distance_relationship_type": str(params.distance_relationship_type),
            "preferential_relationship_type": str(params.preferential_relationship_type),
            "generative_rule": str(params.generative_rule.__class__.__name__),
            "num_iterations": int(params.num_iterations),
        })

        if evaluate_individual_connectomes: 
            indiv_networks_record.update({
                "eta": float(params.eta), 
                "gamma": float(params.gamma),
            })

        if directly_compare_with_empirical_networks: 
            for energy_metric_name in list(experiment.evaluation_results.binary_evaluations.keys()):
                 
                energy_value_mean = experiment.evaluation_results.binary_evaluations[energy_metric_name].mean().item()  # TODO: Include all names # check if these values make sense 
                flat_record.update({energy_metric_name: energy_value_mean})

                # HERE: ADD evaluation results of the individual networks
                if evaluate_individual_connectomes: # TODO: Figure out if this is efficient
                    indiv_energy_values = experiment.evaluation_results.binary_evaluations[energy_metric_name].numpy().flatten()
                    for i in range(individual_networks.shape[0]):
                        indiv_networks_record.update(
                            {energy_metric_name + "_indiv_" + str(i): indiv_energy_values[i]}
                        )
            

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

            try:
                esn_results = [evaluate_memory_capacity_from_connectome(
                    connectome=net, 
                    h_params=h_params
                ) for net in networks_np]
                
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
                print(f"Error in worker process: {e}")
                mc_keys = ["mc_mean"] # + [f"mc_lag_{l}" for l in mc_lags_to_calc]
                flat_record.update({key: np.nan for key in mc_keys})
                
                
    except Exception as e:
        print(f"Error in worker process: {e}")
        flat_record.update({"error": str(e)})

    # 4. Save the results to a unique file in the temporary directory
    if flat_record:
        result_filename = f"result_{uuid.uuid4()}.csv"
        result_path = os.path.join(temp_dir, result_filename)
        pd.DataFrame([flat_record]).to_csv(result_path, index=False, na_rep="nan") # Added that missing values appear as "nan" for more clarity

    if indiv_networks_record: 
        result_filename = f"indiv_connectome_energies_{uuid.uuid4()}.csv"
        result_path = os.path.join(temp_dir, result_filename)
        pd.DataFrame([indiv_networks_record]).to_csv(result_path, index=False, na_rep="nan") # Added that missing values appear as "nan" for more clarity


class GNMandESNPipelineOrchestrator:
    """Pipeline orchestrator with integrated logging."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = torch.device(config["compute"]["device"])
        self.data_loader = DataLoader(config) 
        # self.esn_evaluator = ESNEvaluator(config, self.data_loader) if self.data_loader else None
        self.gnm_generator = GNMGenerator(device=config["compute"]["device"])
        self.output_dir = config["paths"]["output_experiment_dir"]
        self.logger = get_logger(self.output_dir) # output_dir"])
    
    
    def run_gnm_parameter_sweep(self, config, 
                           target_network: Optional[torch.Tensor] = None, # TODO: removre all defaults here. 
                           n_random_samples: int = 30, 
                           ) -> Dict[str, Any]:
        """
        Run a parameter sweep in parallel with robust, interrupt-safe saving.
        """
        
        print("=" * 60)
        print("GNM PARAMETER SWEEP")
        print("=" * 60)
        
        animal_id = 220 # CHANGE
        
        experiment_name = config['experiment']['name'] 
        average_connectomes = config['experiment']['average_connectomes'] # set to false to only evaluate one connectome

        binary_connectomes = self.data_loader.load_binary_connectomes(connectome_id=animal_id)
        distance_matrix = torch.tensor(
                    self.data_loader.load_distance_matrix(connectome_id=animal_id),
                    dtype=torch.float32,
                    device=self.device
                )
        first_density = sorted(binary_connectomes.keys())[0] # TODO: Make this itterable, such that one can do a grid search - or remove this. 

        directly_compare_with_empirical_networks =config['experiment']['directly_compare_with_empirical_networks']
        if directly_compare_with_empirical_networks: 
            # Connectome selection: TODO: Figure out if this is the right approach (i.e. if energies are adding distributive)
            if average_connectomes:
                
                consensus_network = binary_connectomes[first_density]
                target_network = consensus_network
                
            
            else:
                print("Using the first connectome as the target network.")
                target_network = torch.tensor(
                    binary_connectomes[first_density][animal_id, :, :], 
                    dtype=torch.float32,
                    device=self.device
                )

        else: 
            print("No comparison with empirical networks is performed.")
            
        resolution = self.config['data']['connectome_resolution']
        
        # Get number itterations where one edge is added
        if target_network is not None: 
            num_iterations = int(target_network.sum().item() // 2)
        else: 
            num_iterations = resolution**2 * (first_density/100)
            target_network = np.zeros(shape=(resolution, resolution))
            
        # Load the empirical networks, of course! 
        empirical_binary_connectomes = np.zeros(shape=(1, resolution, resolution)) # TODO: THIS IS OBV WRONG! 
        
        # Get number of simulations
        num_simulations = self.config['gnm']['num_simulations'] # 100 
        
        # Create sweep config 
        sweep_config = create_gnm_random_sweep_config( # add a grid version again? 
            config=self.config,
            distance_matrix=torch.Tensor(distance_matrix), 
            num_iterations=num_iterations,
            num_simulations=num_simulations,
            method="random", # TODO: add grid search again.
            n_random_samples=n_random_samples, 
            include_weights=True
        )
        
        evaluation_criteria = create_evaluation_criteria(config=self.config, distance_matrix=distance_matrix)
            
        # IS THIS UNNECESSARY?? H_PARAMS. check which parameters get through until here... 
        h_params = {"spectral_radius": config['esn']['spectral_radius'], # self.config.esn.spectral_radius,  #### TODO: HAND A BIG HPARAMS DICT TO THIS FUNCTION!! (instead of doing it all one by one...)
                    "n_lags": config['esn']['n_lags'], # self.config.esn.n_lags,
                    "train_len": config['esn']['input_lengths'], # self.config.esn.input_length, # seems to have gotten two names... train_len,
                    "test_len": config['esn']['test_len'], # self.config.esn.test_len,
                    "n_runs": config['esn']['n_runs'], # self.config.esn.n_runs,
                    "input_scaling": config['esn']['input_scaling'], # self.config.esn.input_scaling, 
                    "regression_method": config['esn']['regression_method'], # self.config.esn.regularization_method, 
                    "n_transient": config['esn']['n_transient'], # self.config.esn.n_transient,
                    "leak_rate": config['esn']['leak_rate'], # self.config.esn.leak_rate, 
                    "bias": config['esn']['bias'], # self.config.esn.bias,  
                    "random_state": config['esn']['random_state'],  # self.config.compute.random_seed,     
        }

          
        # [FIX] Convert the generator to a list *before* the parallel call.
        # This resolves the serialization error by ensuring a simple, picklable list
        # is passed to the workers, not a complex generator object.
        print("Generating sweep configurations...")
        sweep_config_list = list(sweep_config)
        print(f"{len(sweep_config_list)} configurations generated.")
        
        # experiment_dir = self.config['paths']['output_gnm_dir'] / self.config['experiment']['name']
        temp_results_dir = self.output_dir / f"{self.config['experiment']['name']}_temp_{time.strftime("%Y%m%d_%H%M%S")}" # Ugly HACK
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
            # Set number of workers (default: -1 for maximum parallel execution)
            n_jobs = self.config['compute']['n_workers'] 
            Parallel(n_jobs=n_jobs)( 
                delayed(_run_and_save_single_simulation)(
                    task_data=task_data, # Pass the deconstructed dictionary
                    evaluation_criteria=evaluation_criteria,
                    # weighted_evaluation_criteria=weighted_criteria, # TODO: STOP HARDCODING THIS (SEE ABOVE)
                    target_network=target_network,
                    directly_compare_with_empirical_networks=config['experiment']['directly_compare_with_empirical_networks'], 
                    individual_networks=empirical_binary_connectomes, 
                    elaborate_analysis=config['experiment']['elaborate_analysis'],
                    evaluate_individual_connectomes=self.config['experiment']['evaluate_individual_connectomes'], 
                    device_str=self.config['compute']['device'], #self.config.compute.device,
                    output_dir=self.output_dir, 
                    temp_dir=temp_results_dir,
                    h_params=h_params
                )
                for task_data in tqdm(deconstructed_tasks, desc="Configuration Iterations")
            )

        except (KeyboardInterrupt, Exception) as e:
            print(f"\n--- Process interrupted or failed: {e} ---")

        # finally:
        #     print("\nCombining results...")
            
        #     print(f"\nCombining CSVs for experiment: '{experiment_name}'...")

        #     # 1. Combine the 'result_....csv' files
        #     try: 
        #         _combine_csvs_by_pattern(
        #             search_dir=temp_results_dir,
        #             file_pattern="result_*.csv",
        #             output_name=f"all_metrics_for_exp_{experiment_name}.csv"
        #         )
        #     except: 
        #         print("No results_*.csv were previously generated.")

        #     # 2. Combine the 'indiv_connectome...' files
        #     try: 
        #         _combine_csvs_by_pattern(
        #             search_dir=temp_results_dir,
        #             file_pattern="indiv_connectome_energies_*.csv",
        #             output_name=f"indiv_energies_for_exp_{experiment_name}.csv"
        #         )
        #     except: 
        #         print("No indiv_connectome_energies_*.csv were previously generated.")

        #     print("\nCombination complete. Temporary results:", temp_results_dir)
            
        merge_csv_files(self.output_dir)

            