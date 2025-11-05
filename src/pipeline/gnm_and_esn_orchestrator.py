"""
Main pipeline fully integrated with GNM library and centralized logging.
This version includes automatic ID generation for duplicate parameter combinations.
"""

from typing import Optional, Dict, Any
import time
import numpy as np
import torch
import pandas as pd
import uuid
import os
from pathlib import Path
from collections import defaultdict

# For parallel processing
from joblib import Parallel, delayed
from tqdm import tqdm

# Import our optimized modules
from config.manager import create_gnm_sweep_config
from src.GNMs.gnm_network_generator import GNMGenerator
from src.utils.data_loader import DataLoader
from src.utils.run_logger import get_logger

from src.config.GNM import create_evaluation_criteria

from analysis.structural_measures import analyze_connectomes
from ESNs.esn_evaluation import evaluate_memory_capacity_from_connectome

from src.utils.combine_csvs import merge_csv_files



# small helper
def floor_to_14_decimal_places(number: float) -> float:
    return np.floor(number * 10e14) / 10e14


class NetworkIDManager:
    """
    Manages ID assignment for generated networks.
    Ensures that networks with duplicate (eta, gamma) parameters get sequential IDs.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize ID manager.
        
        Args:
            output_dir: Directory where networks are/will be saved
        """
        self.output_dir = output_dir
        self.generated_networks_dir = output_dir / "generated_networks"
        self.generated_networks_dir.mkdir(parents=True, exist_ok=True)
        
        # Track parameter combinations and their current max IDs
        self.param_id_counter = defaultdict(int)
        
        # Scan existing networks to initialize counters
        self._initialize_from_existing_networks()
    
    def _initialize_from_existing_networks(self):
        """Scan existing network files to determine current ID counters."""
        import re
        
        if not self.generated_networks_dir.exists():
            return
        
        for filepath in self.generated_networks_dir.glob("net_eta*.npy"):
            filename = filepath.name
            
            # Extract eta and gamma
            eta_match = re.search(r'eta([-+]?\d*\.?\d+)', filename)
            gamma_match = re.search(r'gamma([-+]?\d*\.?\d+)', filename)
            
            if not eta_match or not gamma_match:
                continue
            
            eta = float(eta_match.group(1))
            gamma = float(gamma_match.group(1))
            
            # Extract id (defaults to 0)
            id_match = re.search(r'_id(\d+)', filename)
            net_id = int(id_match.group(1)) if id_match else 0
            
            # Update counter for this parameter combination
            param_key = (eta, gamma)
            self.param_id_counter[param_key] = max(
                self.param_id_counter[param_key], 
                net_id + 1
            )
        
        if self.param_id_counter:
            print(f"📊 Initialized ID manager with {len(self.param_id_counter)} existing parameter combinations")
            max_id = max(self.param_id_counter.values()) - 1
            print(f"   Highest existing ID: {max_id}")
    
    def get_next_id(self, eta: float, gamma: float) -> int:
        """
        Get the next available ID for a given (eta, gamma) combination.
        
        Args:
            eta: Eta parameter value
            gamma: Gamma parameter value
            
        Returns:
            Next available ID (0 for first occurrence, then 1, 2, 3, ...)
        """
        param_key = (eta, gamma)
        current_id = self.param_id_counter[param_key]
        self.param_id_counter[param_key] += 1
        return current_id
    
    def generate_filename(self, eta: float, gamma: float, rule_name: str) -> str:
        """
        Generate filename with appropriate ID.
        
        Args:
            eta: Eta parameter value
            gamma: Gamma parameter value
            rule_name: Name of the generative rule
            
        Returns:
            Filename string (without path)
        """
        net_id = self.get_next_id(eta, gamma)
        
        if net_id == 0:
            # First network with these parameters - no ID suffix
            filename = f"net_eta{eta}_gamma{gamma}_rule{rule_name}.npy"
        else:
            # Duplicate parameters - add ID suffix
            filename = f"net_eta{eta}_gamma{gamma}_rule{rule_name}_id{net_id}.npy"
        
        return filename


def _run_and_save_single_simulation(
    task_data: dict, 
    evaluation_criteria, 
    target_network,
    
    n_edges,
    calculate_energy: bool, 
    
    elaborate_analysis: bool, 
    # calculate_energies_of_all_individual_connectomes: bool, 
    # individual_networks: Optional[np.ndarray], # only needs to be set if compare_to_all_individual_empirical_connectomes is True
    
    device_str: str, 
    output_dir: Path, 
    temp_dir: str, 
    h_params: dict,
    id_manager: NetworkIDManager
):
    """
    Worker function that reconstructs objects from simple data before running the simulation.
    Now includes ID management.
    """
    from gnm.fitting import perform_run, RunConfig
    from gnm import generative_rules 
    from gnm.model import BinaryGenerativeParameters
    
    # --- Reconstruct the RunConfig object from the dictionary ---
    bp_data = task_data['binary_parameters']
    
    # Get the rule class from the gnm library using its name
    RuleClass = getattr(generative_rules, bp_data['generative_rule_name'])
    
    # n_edges = int(target_network.sum().item() // 2)
    binary_params = BinaryGenerativeParameters(
        eta=bp_data['eta'],
        gamma=bp_data['gamma'],
        lambdah=bp_data['lambdah'], 
        heterochronicity_relationship_type=bp_data['heterochronicity_relationship_type'],
        generative_rule=RuleClass(),
        num_iterations=n_edges,
        distance_relationship_type=bp_data['distance_relationship_type'],
        preferential_relationship_type=bp_data['preferential_relationship_type']
    )
    
    run_config = RunConfig(
        num_simulations=None,
        binary_parameters=binary_params,
        distance_matrix=task_data['distance_matrix']
    )

    flat_record = {}
    indiv_networks_record = {}
    
    # Get next ID for this parameter combination
    eta_val = floor_to_14_decimal_places(float(bp_data['eta']))
    gamma_val = floor_to_14_decimal_places(float(bp_data['gamma']))
    net_id = id_manager.get_next_id(eta_val, gamma_val)
    
    # Perform a run
    try:
        if calculate_energy: 
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
            experiment = perform_run(
                run_config=run_config,
                save_model=True,
                save_run_history=False,
                device=torch.device(device_str),
            )

        # 2. Prepare the data record for this iteration
        params = experiment.run_config.binary_parameters
        params.eta = floor_to_14_decimal_places(float(params.eta))
        params.gamma = floor_to_14_decimal_places(float(params.gamma))
        flat_record.update({
            "eta": params.eta,
            "gamma": params.gamma,
            "id": net_id,
            "distance_relationship_type": str(params.distance_relationship_type),
            "preferential_relationship_type": str(params.preferential_relationship_type),
            "generative_rule": str(params.generative_rule.__class__.__name__),
            "num_iterations": int(params.num_iterations),
        })

        # if calculate_energies_of_all_individual_connectomes: 
        #     indiv_networks_record.update({
        #         "eta": params.eta,
        #         "gamma": params.gamma,
        #         "id_of_generated_network": net_id,
        #         # "id_of_animal": id_manager.get_animal_id()
        #     })

        if calculate_energy: 
            for energy_metric_name in list(experiment.evaluation_results.binary_evaluations.keys()):
                 
                energy_value_mean = experiment.evaluation_results.binary_evaluations[energy_metric_name].mean().item()
                flat_record.update({energy_metric_name: energy_value_mean})

                # if calculate_energies_of_all_individual_connectomes:
                #     indiv_energy_values = experiment.evaluation_results.binary_evaluations[energy_metric_name].numpy().flatten()
                #     for i in range(individual_networks.shape[0]):
                #         indiv_networks_record.update(
                #             {energy_metric_name + "_indiv_" + str(i): indiv_energy_values[i]}
                #         )
            

        # 3. If elaborate_analysis is true, run detailed analysis and save network
        if elaborate_analysis and experiment.model:
            networks_np = experiment.model.adjacency_matrix.cpu().numpy()

            # Generate filename with ID using the manager
            rule_name = params.generative_rule.__class__.__name__
            # turn id into something like "000", "001", etc. for consistent sorting
            net_id_str = f"{net_id:03d}"
            
            # if net_id == 0:
            #     filename = f"net_eta{np.floor(params.eta.item(),10)}_gamma{np.floor(params.gamma.item(),10)}_rule{rule_name}.npy"
            # else:
            filename = f"net_eta{params.eta}_gamma{params.gamma}_rule{rule_name}_id{net_id_str}.npy"

            save_path = output_dir / "generated_networks" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, networks_np)

            graph_measures_list = analyze_connectomes(
                connectomes=networks_np, 
                distance_matrix=run_config.distance_matrix.cpu().numpy()
            )
    
            flat_record.update(pd.DataFrame(graph_measures_list).mean().to_dict())

            try:
                esn_results = [evaluate_memory_capacity_from_connectome(
                    connectome=net, 
                    h_params=h_params
                ) for net in networks_np]
                
                df_esn = pd.json_normalize(esn_results[0], sep='_')
                
                numeric_cols = df_esn.select_dtypes(include=np.number).columns
                flat_record.update(df_esn[numeric_cols].mean().to_dict())
                
                # Add mc values for individual lags
                if "mc_values_for_indiv_lags" in df_esn.columns:
                    for i in range(50):
                        flat_record.update({f"mc_{i}": df_esn["mc_values_for_indiv_lags"][0][i-1]})

            except Exception as e:
                print(f"Error in ESN evaluation: {e}")
                mc_keys = ["mc_mean"]
                flat_record.update({key: np.nan for key in mc_keys})
                
                
    except Exception as e:
        print(f"Error in worker process: {e}")
        flat_record.update({
            "eta": eta_val,
            "gamma": gamma_val,
            "id_of_generated_network": net_id,
            "error": str(e)
        })

    # 4. Save the results to a unique file in the temporary directory
    if flat_record:
        result_filename = f"result_{uuid.uuid4()}.csv"
        Path(temp_dir).mkdir(exist_ok=True)
        result_path = os.path.join(temp_dir, result_filename)
        pd.DataFrame([flat_record]).to_csv(result_path, index=False, na_rep="nan")

    if indiv_networks_record: 
        result_filename = f"indiv_connectome_energies_{uuid.uuid4()}.csv"
        result_path = os.path.join(temp_dir, result_filename)
        pd.DataFrame([indiv_networks_record]).to_csv(result_path, index=False, na_rep="nan")


class GNMandESNPipelineOrchestrator:
    """Pipeline orchestrator with integrated logging and ID management."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["compute"]["device"])
        self.data_loader = DataLoader(config) 
        self.gnm_generator = GNMGenerator(device=config["compute"]["device"])
        self.output_dir = config["paths"]["output_experiment_dir"]
        self.logger = get_logger(self.output_dir)
        
        # Initialize ID manager
        self.id_manager = NetworkIDManager(self.output_dir)
    
    
    def run_gnm_parameter_sweep(self, 
                                target_network: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Run a parameter sweep in parallel with robust, interrupt-safe saving and ID management.
        """
        
        print("=" * 60)
        print("GNM PARAMETER SWEEP")
        print("=" * 60)
        
        animal_id = self.config['experiment']['animal'] 
        
        average_connectomes = self.config['experiment']['average_connectomes'] # could also be called: Use connectome as it is 

        binary_connectomes = self.data_loader.load_binary_connectomes(connectome_id=animal_id)
        
        distance_matrix = torch.tensor(
            self.data_loader.load_distance_matrix(connectome_id=animal_id),
            dtype=torch.float32,
            device=self.device
        )
        
        first_density = sorted(binary_connectomes.keys())[0]

        calculate_energy = self.config['experiment']['calculate_energy']
        if calculate_energy: 
            if average_connectomes:
                consensus_network = binary_connectomes[first_density]
                target_network = consensus_network # HPC is here, gets 990 connections
            else:
                print("Using the first connectome as the target network.") # TODO: Not sure if this works... 
                target_network = torch.tensor( # for HPC, this should not be chosen (and it is not) 
                    binary_connectomes[first_density][animal_id, :, :], 
                    dtype=torch.float32,
                    device=self.device
                )
        else: 
            print("No comparison with empirical networks is performed.")
            
        resolution = self.config['data']['connectome_resolution']
        
        # Get number iterations
        if target_network is not None and calculate_energy: 
            # num_iterations = int(target_network[first_density].sum().item() // 2) # no density needed, as this is already part of the target network?? (at least for HPC)) 
            # TODO: Check how it is for the other datasets.
            n_edges = int(target_network.sum().item() // 2) # divided by 2 because undirected -> should be 495 approximately for 100*100 and 10 percent
        else: 
            # num_iterations = (distance_matrix.shape[0]*(distance_matrix.shape[0]-1))*(first_density/100)
            n_edges = int((distance_matrix.shape[0]*(distance_matrix.shape[0]-1))*(first_density/100) // 2)
            target_network = np.zeros(shape=(resolution, resolution))
            
        # Load the empirical networks TODO: What was this doing??
        # empirical_binary_connectomes = np.zeros(shape=(1, resolution, resolution))
        
        # Get number of simulations
        num_simulations = self.config['gnm']['num_simulations']
        
        # Create sweep config 
        # print("DEBUG: ", self.config) # DEBUG
        sweep_config = create_gnm_sweep_config(
            config=self.config, # CHECK HERE!! is it already rounded? Does it contain eta and gamma? 
            distance_matrix=torch.Tensor(distance_matrix),
            mode=self.config['experiment']['search']['method'],
            # num_iterations=num_iterations,
            n_edges=n_edges,
            num_simulations=num_simulations,
        )
        evaluation_criteria = create_evaluation_criteria(
            config=self.config, 
            distance_matrix=distance_matrix
        )
          
        # Convert the generator to a list
        print("Generating sweep configurations...")
        sweep_config_list = list(sweep_config)
        print(f"{len(sweep_config_list)} configurations generated.")
        print(f"Sampling method: {self.config['experiment']['search']['method']}")
        print(f"Total configurations to evaluate: {len(sweep_config_list)}")

        temp_results_dir = self.output_dir / "_temp_results" / f"{self.config['experiment']['name']}_batch_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(temp_results_dir, exist_ok=True)
        
        print("Generating and deconstructing sweep configurations for parallel processing...")
        
        # Create a list of simple, picklable dictionaries
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
            # Run parallel processing with ID manager
            Parallel(n_jobs=self.config['compute']['n_workers'])( 
                delayed(_run_and_save_single_simulation)(
                    task_data=task_data,
                    evaluation_criteria=evaluation_criteria,
                    target_network=target_network,
                    calculate_energy=self.config['experiment']['calculate_energy'],
                    n_edges=n_edges,
                    elaborate_analysis=self.config['experiment']['elaborate_analysis'],
                    # TODO: MAKE SURE THE TWO LINES BELOW WOULD WORK, in case the first line is set to true. 
                    # calculate_energies_of_all_individual_connectomes=self.config['experiment']['calculate_energies_of_all_individual_connectomes'], # self.config['experiment']['calculate_energies_of_all_individual_connectomes'], 
                    # individual_networks=None, # does not need to be set - does only need to be set if calculate_energies_of_all_individual_connectomes is true.  
            
                    device_str=self.config['compute']['device'],
                    output_dir=self.output_dir, 
                    temp_dir=temp_results_dir,
                    h_params=self.config["esn"],
                    id_manager=self.id_manager
                )
                for task_data in tqdm(deconstructed_tasks, desc="Configuration Iterations")
            )

        except (KeyboardInterrupt, Exception) as e:
            print(f"\n--- Process interrupted or failed: {e} ---")

        merge_csv_files(self.output_dir, self.config["experiment"]["name"]) 