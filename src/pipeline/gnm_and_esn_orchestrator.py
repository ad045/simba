"""
Main pipeline fully integrated with GNM library and centralized logging.
This version is updated for robust, interrupt-safe multiprocessing on macOS.
"""

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
from ESNs.alternative_esn_evaluation import evaluate_memory_capacity_from_connectome

from src.utils.combine_csvs import merge_csv_files

# Helper functions

def _run_and_save_single_simulation( # is USED1. 
                                    task_data: dict, 
                                    evaluation_criteria, 
                                    # weighted_evaluation_criteria, 
                                    target_network, # can be one or more?  # TODO: What do I do with this instead? 
                                    individual_networks, # needs to be set: a float32 torch Tensor of shape (num_subjects, n_nodes, n_nodes)
                                    compare_to_connectome_of_distance_matrix: bool, 
                                    elaborate_analysis: bool, 
                                    compare_to_all_individual_empirical_connectomes: bool, 
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
    
    #     n_edges = int(target_tensor.sum().item() // 2) / 2 # not sure why this / 2 would be needed, but let's try. (TODO)
#                                                         #    Density according to GNM library 
#                                                                 # CIJ = c[0,:,:]
#                                                                 # n = len(CIJ)
#                                                                 # k = np.size(np.where(np.triu(CIJ).flatten()))
#                                                                 # kden = k / ((n * n - n) / 2)
#                                         # kden * ((n * n - n) / 2) = k = n_edges * 2? 
    n_edges = int(target_network.sum().item() // 2)
    binary_params = BinaryGenerativeParameters( # BinaryGenerativeParameters
        eta=bp_data['eta'],
        gamma=bp_data['gamma'],
        lambdah=bp_data['lambdah'], 
        heterochronicity_relationship_type=bp_data['heterochronicity_relationship_type'],
        generative_rule=RuleClass(),
        # num_iterations=# bp_data['num_iterations'], -> !!! No, num_iterations is the number of edges put into the network... 
        num_iterations=n_edges,
        distance_relationship_type=bp_data['distance_relationship_type'],
        preferential_relationship_type=bp_data['preferential_relationship_type']
    )
    
    run_config = RunConfig(
        num_simulations=None, # TODO: is currently hardcoded - is supposed to run in parallel? Check if it actually does that... 
        binary_parameters=binary_params,
        distance_matrix=task_data['distance_matrix'] #  still correct
    )

    flat_record = {}
    indiv_networks_record = {}
    
    # Perform a run - either while evaluating individual connectomes, or not. 
    try:
        if compare_to_connectome_of_distance_matrix: 
            # if target network is given
            target_network = torch.tensor(
                    target_network,
                    dtype=torch.float32,
            )
                        
            experiment = perform_run(
                run_config=run_config,
                binary_evaluations=[evaluation_criteria],
                real_binary_matrices=target_network.unsqueeze(0), 
                save_model=True,
                save_run_history=False,
                device=torch.device(device_str),
            )
        else: 
            # if no target network is given
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
            "generative_rule": str(params.generative_rule.__class__.__name__), # TODO: change this? 
            "num_iterations": int(params.num_iterations),
        })

        if compare_to_all_individual_empirical_connectomes: 
            indiv_networks_record.update({
                "eta": float(params.eta), 
                "gamma": float(params.gamma),
            })

        if compare_to_connectome_of_distance_matrix: 
            for energy_metric_name in list(experiment.evaluation_results.binary_evaluations.keys()):
                 
                energy_value_mean = experiment.evaluation_results.binary_evaluations[energy_metric_name].mean().item()  # TODO: Include all names # check if these values make sense 
                flat_record.update({energy_metric_name: energy_value_mean})

                # HERE: ADD evaluation results of the individual networks
                if compare_to_all_individual_empirical_connectomes: # TODO: Figure out if this is efficient
                    indiv_energy_values = experiment.evaluation_results.binary_evaluations[energy_metric_name].numpy().flatten()
                    for i in range(individual_networks.shape[0]):
                        indiv_networks_record.update(
                            {energy_metric_name + "_indiv_" + str(i): indiv_energy_values[i]}
                        )
            

        # 3. If elaborate_analysis is true, run detailed analysis
        if elaborate_analysis and experiment.model:
            networks_np = experiment.model.adjacency_matrix.cpu().numpy() # TODO (prob somewhere slightly else): Save ALL generated conns

            rule_name = params.generative_rule.__class__.__name__
            # filename = f"net_eta{params.eta.item():.3f}_gamma{params.gamma.item():.3f}_rule{rule_name}.npy"
            filename = f"net_eta{params.eta.item()}_gamma{params.gamma.item()}_rule{rule_name}.npy"
            save_path = output_dir / "generated_networks" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, networks_np)

            graph_measures_list = analyze_connectomes(
                connectomes=networks_np, 
                distance_matrix=run_config.distance_matrix.cpu().numpy() # WORKS. 
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
        Path(temp_dir).mkdir(exist_ok=True)
        result_path = os.path.join(temp_dir, result_filename)
        pd.DataFrame([flat_record]).to_csv(result_path, index=False, na_rep="nan") # Added that missing values appear as "nan" for more clarity

    if indiv_networks_record: 
        result_filename = f"indiv_connectome_energies_{uuid.uuid4()}.csv"
        result_path = os.path.join(temp_dir, result_filename)
        pd.DataFrame([indiv_networks_record]).to_csv(result_path, index=False, na_rep="nan") # Added that missing values appear as "nan" for more clarity


class GNMandESNPipelineOrchestrator: # IS USED1 
    """Pipeline orchestrator with integrated logging."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = torch.device(config["compute"]["device"])
        self.data_loader = DataLoader(config) 
        # self.esn_evaluator = ESNEvaluator(config, self.data_loader) if self.data_loader else None
        self.gnm_generator = GNMGenerator(device=config["compute"]["device"])
        self.output_dir = config["paths"]["output_experiment_dir"]
        self.logger = get_logger(self.output_dir) # output_dir"])
    
    
    def run_gnm_parameter_sweep(self, # IS USED1 # config, 
                           target_network: Optional[torch.Tensor] = None, # TODO: removre all defaults here. 
                           ) -> Dict[str, Any]:
        """
        Run a parameter sweep in parallel with robust, interrupt-safe saving.
        """
        
        print("=" * 60)
        print("GNM PARAMETER SWEEP")
        print("=" * 60)
        
        animal_id = self.config['experiment']['animal'] 
        
        average_connectomes = self.config['experiment']['average_connectomes'] # set to false to only evaluate one connectome

        # CAN THIS BE REMOVED? 
        binary_connectomes = self.data_loader.load_binary_connectomes(connectome_id=animal_id)
        
        distance_matrix = torch.tensor( # Typing deviation (from float64 to float32) 
                    self.data_loader.load_distance_matrix(connectome_id=animal_id),
                    dtype=torch.float32, # here it gets 10e-6 differences... Normal difference between e.g. loaded 42 and orig 0: max. 40
                    device=self.device
                )
        
        first_density = sorted(binary_connectomes.keys())[0] # TODO: Make this itterable, such that one can do a grid search - or remove this. 

        compare_to_connectome_of_distance_matrix = self.config['experiment']['compare_to_connectome_of_distance_matrix']
        if compare_to_connectome_of_distance_matrix: 
            # Connectome selection: TODO: Figure out if this is the right approach (i.e. if energies are adding distributive)
            if average_connectomes:
                # print(binary_connectomes)
                # print(first_density)
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
            num_iterations = int(target_network[first_density].sum().item() // 2) # TODO: do not make it to a dict anymore, just take the target_network directly
        else: 
            num_iterations = resolution**2 * (first_density/100)
            target_network = np.zeros(shape=(resolution, resolution))
            
        # Load the empirical networks, of course! 
        empirical_binary_connectomes = np.zeros(shape=(1, resolution, resolution)) # TODO: THIS IS OBV WRONG! 
        
        # Get number of simulations
        num_simulations = self.config['gnm']['num_simulations'] # 100 
        
        # Create sweep config 
        sweep_config = create_gnm_random_sweep_config( # this is used # add a grid version again? 
            config=self.config,
            distance_matrix=torch.Tensor(distance_matrix),  # still working
            num_iterations=num_iterations,
            num_simulations=num_simulations,
            include_weights=False # True
        )
        # print("SWEEP CONFIG", sweep_config)
        evaluation_criteria = create_evaluation_criteria(config=self.config, distance_matrix=distance_matrix)
          
        # Convert the generator to a list *before* the parallel call
        print("Generating sweep configurations...")
        sweep_config_list = list(sweep_config) # already the issue. 
        print(f"{len(sweep_config_list)} configurations generated.")
        
        # experiment_dir = self.config['paths']['output_gnm_dir'] / self.config['experiment']['name']
        temp_results_dir = self.output_dir / "_temp_results" / f"{self.config['experiment']['name']}_batch_{time.strftime("%Y%m%d_%H%M%S")}" # Ugly HACK
        os.makedirs(temp_results_dir, exist_ok=True)
        
        print("Generating and deconstructing sweep configurations for parallel processing...")
        
        # Create a list of simple, picklable dictionaries instead of complex objects -> joblib otherwise gets issues regarding "unable to serialize" 
        deconstructed_tasks = []
        for run_config in sweep_config_list:
            task = {
                "binary_parameters": {
                    "eta": run_config.binary_parameters.eta,
                    "gamma": run_config.binary_parameters.gamma,
                    "lambdah": run_config.binary_parameters.lambdah, # TODO: imporve writing...
                    "heterochronicity_relationship_type": str(run_config.binary_parameters.heterochronicity_relationship_type),
                    "generative_rule_name": run_config.binary_parameters.generative_rule.__class__.__name__, # TODO: Get those from the config file.... 
                    "num_iterations": run_config.binary_parameters.num_iterations,
                    "distance_relationship_type": str(run_config.binary_parameters.distance_relationship_type),
                    "preferential_relationship_type": str(run_config.binary_parameters.preferential_relationship_type),
                },
                "distance_matrix": run_config.distance_matrix # already the issue. Works.
            }
            deconstructed_tasks.append(task)

        try: 
            # Set number of workers (default: -1 for maximum parallel execution)
            Parallel(n_jobs=self.config['compute']['n_workers'])( 
                delayed(_run_and_save_single_simulation)(
                    task_data=task_data, # Pass the deconstructed dictionary
                    evaluation_criteria=evaluation_criteria,
                    target_network=target_network,
                    compare_to_connectome_of_distance_matrix=self.config['experiment']['compare_to_connectome_of_distance_matrix'],  # THIS IS WRONG. 
                    individual_networks=empirical_binary_connectomes, 
                    elaborate_analysis=self.config['experiment']['elaborate_analysis'],
                    compare_to_all_individual_empirical_connectomes=self.config['experiment']['compare_to_all_individual_empirical_connectomes'], 
                    device_str=self.config['compute']['device'], #self.config.compute.device,
                    output_dir=self.output_dir, 
                    temp_dir=temp_results_dir,
                    h_params=self.config["esn"]
                )
                for task_data in tqdm(deconstructed_tasks, desc="Configuration Iterations")
            )
            
            # Pot improvement (did not try yet)
            #  with parallel_backend('loky', n_jobs=n_workers):
            #     Parallel()(
            #         delayed(_run_and_save_single_simulation)(
            #             task_data=task_data,
            #             evaluation_criteria=evaluation_criteria,
            #             target_network=target_network_np,
            #             compare_to_connectome_of_distance_matrix=self.config['experiment']['compare_to_connectome_of_distance_matrix'], 
            #             individual_networks=empirical_binary_connectomes, 
            #             elaborate_analysis=self.config['experiment']['elaborate_analysis'],
            #             compare_to_all_individual_empirical_connectomes=self.config['experiment']['compare_to_all_individual_empirical_connectomes'], 
            #             device_str=self.config['compute']['device'],
            #             output_dir=self.output_dir, 
            #             temp_dir=temp_results_dir,
            #             h_params=self.config["esn"]
            #         )
            #         for task_data in tqdm(deconstructed_tasks, desc="Configuration Iterations")
            #     )

        except (KeyboardInterrupt, Exception) as e:
            print(f"\n--- Process interrupted or failed: {e} ---")

        merge_csv_files(self.output_dir, self.config["experiment"]["name"])

            
