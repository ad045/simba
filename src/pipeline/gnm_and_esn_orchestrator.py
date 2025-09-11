"""
Main pipeline fully integrated with GNM library and centralized logging.
Clean version with proper logging integration.
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

# GNM library imports
from gnm import fitting 
# from src.imported_libraries.GenerativeNetworkModels_2.src.gnm.model import GenerativeNetworkModel, BinaryGenerativeParameters

# Import our optimized modules
from src.config.ESN_and_GNM_config import ConfigManager # , get_gnm_quick_test_config, get_gnm_comprehensive_config, get_gnm_minimal_config
from src.GNMs.gnm_network_generator import GNMGenerator
from src.utils.data_loader import DataLoader
from src.ESNs.esn_evaluation import ESNEvaluator
from src.utils.run_logger import get_logger


class GNMandESNPipelineOrchestrator:
    """Pipeline orchestrator with integrated logging."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = torch.device(config.gnm.device)
        
        # Initialize components
        self.data_loader = DataLoader(config) if not config.data.use_gnm_defaults else None
        self.esn_evaluator = ESNEvaluator(config, self.data_loader) if self.data_loader else None
        self.gnm_generator = GNMGenerator(device=config.gnm.device)
        
        # Initialize logger
        self.logger = get_logger(config.paths.output_dir)
        
        # Ensure output directories exist
        self.config.paths.esn_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.paths.gnm_output_dir.mkdir(parents=True, exist_ok=True)
    
    # def run_gnm_comprehensive_analysis(self,
    #                                   experiment_name: Optional[str] = None,
    #                                   compare_rules: bool = True,
    #                                   fit_weights: bool = True) -> Dict[str, Any]:
    #     """
    #     Run comprehensive GNM analysis with logging.
        
    #     Args:
    #         experiment_name: Custom experiment name
    #         compare_rules: Whether to compare different generative rules
    #         fit_weights: Whether to optimize edge weights
            
    #     Returns:
    #         Dictionary with comprehensive results
    #     """
    #     print("=" * 60)
    #     print("COMPREHENSIVE GNM ANALYSIS")
    #     print("=" * 60)
        
    #     # Set experiment name
    #     if not experiment_name:
    #         experiment_name = f"gnm_comprehensive_{time.strftime('%Y%m%d_%H%M%S')}"
        
    #     # Load data
    #     if self.config.data.use_gnm_defaults:
    #         print("Using GNM default data...")
    #         data = self.config.load_gnm_defaults()
    #         distance_matrix = data["distance_matrix"]
    #         target_networks = [data["binary_network"]]
    #     else:
    #         print("Loading custom connectome data...")
    #         distance_matrix = torch.tensor(
    #             self.data_loader.load_distance_matrix(), 
    #             dtype=torch.float32, 
    #             device=self.device
    #         )
            
    #         # Load binary connectomes
    #         binary_connectomes = self.data_loader.load_binary_connectomes()
            
    #         # Convert to list of torch tensors
    #         target_networks = []
    #         for density, conn_array in binary_connectomes.items():
    #             for i in range(min(5, conn_array.shape[2])):  # Limit subjects
    #                 target_networks.append(
    #                     torch.tensor(conn_array[:, :, i], dtype=torch.float32, device=self.device)
    #                 )
        
    #     # Create experiment directory
    #     exp_dir = self.config.paths.gnm_output_dir / experiment_name
    #     exp_dir.mkdir(parents=True, exist_ok=True)
        
    #     results = {
    #         "experiment_dir": str(exp_dir),
    #         "experiment_name": experiment_name,
    #         "n_target_networks": len(target_networks),
    #         "generative_rules_tested": self.config.gnm.generative_rules_to_test,
    #         "evaluation_metrics": self.config.gnm.evaluation_metrics,
    #         "network_results": []
    #     }
        
    #     # Process each target network
    #     for idx, target_network in enumerate(target_networks):
    #         print(f"\nProcessing network {idx + 1}/{len(target_networks)}...")
            
    #         network_result = {
    #             "network_index": idx,
    #             "n_nodes": target_network.shape[0],
    #             "n_edges": int(target_network.sum().item() // 2)
    #         }
            
    #         # 1. Compare generative rules if requested
    #         if compare_rules:
    #             print("  Comparing generative rules...")
    #             rule_comparison = self.gnm_generator.compare_generative_rules(
    #                 target_network=target_network,
    #                 distance_matrix=distance_matrix,
    #                 rules_to_test=self.config.gnm.generative_rules_to_test,
    #                 n_simulations=min(20, self.config.gnm.num_simulations)
    #             )
    #             network_result["rule_comparison"] = rule_comparison
    #             best_rule = rule_comparison["best_rule"]
    #             print(f"    Best rule: {best_rule}")
                
    #             # Log rule comparison results
    #             for rule, rule_results in rule_comparison.items():
    #                 if isinstance(rule_results, dict) and "best_energy" in rule_results:
    #                     self.logger.log_run(
    #                         run_type="gnm_rule_comparison",
    #                         experiment_name=experiment_name,
    #                         parameters={
    #                             "network_index": idx,
    #                             "generative_rule": rule,
    #                             "eta": rule_results.get("best_eta"),
    #                             "gamma": rule_results.get("best_gamma")
    #                         },
    #                         results={
    #                             "energy": rule_results.get("best_energy")
    #                         },
    #                         metadata={
    #                             "n_edges": network_result["n_edges"],
    #                             "n_nodes": network_result["n_nodes"]
    #                         }
    #                     )
    #         else:
    #             best_rule = self.config.gnm.generative_rules_to_test[0]
            
    #         # 2. Fit parameters for best rule
    #         print(f"  Fitting parameters for {best_rule}...")
    #         fit_result = self.gnm_generator.fit_parameters(
    #             target_network=target_network,
    #             distance_matrix=distance_matrix,
    #             generative_rule_name=best_rule,
    #             n_eta=self.config.gnm.n_eta,
    #             n_gamma=self.config.gnm.n_gamma,
    #             num_simulations=self.config.gnm.num_simulations,
    #             evaluation_metrics=self.config.gnm.evaluation_metrics
    #         )
            
    #         network_result["parameter_fitting"] = {
    #             "best_eta": fit_result["best_eta"],
    #             "best_gamma": fit_result["best_gamma"],
    #             "best_energy": fit_result["best_energy"],
    #             "generative_rule": best_rule
    #         }
            
    #         # Log parameter fitting result
    #         self.logger.log_run(
    #             run_type="gnm_parameter_fitting",
    #             experiment_name=experiment_name,
    #             parameters={
    #                 "network_index": idx,
    #                 "generative_rule": best_rule,
    #                 "eta": fit_result["best_eta"],
    #                 "gamma": fit_result["best_gamma"]
    #             },
    #             results={
    #                 "energy": fit_result["best_energy"]
    #             },
    #             metadata={
    #                 "n_eta_tested": self.config.gnm.n_eta,
    #                 "n_gamma_tested": self.config.gnm.n_gamma,
    #                 "num_simulations": self.config.gnm.num_simulations
    #             }
    #         )
            
    #         # 3. Generate and evaluate synthetic networks
    #         print("  Generating synthetic networks...")
    #         from gnm_network_generator import GNMParameters
            
    #         params = GNMParameters(
    #             eta=fit_result["best_eta"],
    #             gamma=fit_result["best_gamma"],
    #             generative_rule=self.gnm_generator.AVAILABLE_RULES[best_rule]()
    #         )
            
    #         synthetic_networks = []
    #         for i in range(10):  # Generate 10 realizations
    #             synthetic = self.gnm_generator.generate_network(
    #                 n_nodes=target_network.shape[0],
    #                 n_edges=network_result["n_edges"],
    #                 distance_matrix=distance_matrix,
    #                 parameters=params
    #             )
    #             synthetic_networks.append(synthetic)
            
    #         # 4. Evaluate synthetic networks
    #         print("  Evaluating synthetic networks...")
    #         evaluations = []
    #         for synthetic in synthetic_networks:
    #             eval_result = self.gnm_generator.evaluate_network(
    #                 generated_network=synthetic,
    #                 target_network=target_network,
    #                 distance_matrix=distance_matrix,
    #                 metrics=self.config.gnm.evaluation_metrics
    #             )
    #             evaluations.append(eval_result)
            
    #         # Compute average metrics
    #         avg_metrics = {}
    #         for metric in self.config.gnm.evaluation_metrics:
    #             values = [e[metric] for e in evaluations if metric in e]
    #             if values:
    #                 avg_metrics[metric] = {
    #                     "mean": np.mean(values),
    #                     "std": np.std(values)
    #                 }
            
    #         network_result["evaluation"] = avg_metrics
            
    #         # 5. Optimize weights if requested
    #         if fit_weights:
    #             print("  Optimizing edge weights...")
    #             try:
    #                 weighted_networks = self.gnm_generator.batch_generate_with_weights(
    #                     binary_networks=[synthetic_networks[0]],
    #                     distance_matrix=distance_matrix,
    #                     binary_params=params,
    #                     alpha=self.config.gnm.alpha
    #                 )
                    
    #                 if weighted_networks:
    #                     network_result["weight_optimization"] = {
    #                         "alpha": self.config.gnm.alpha,
    #                         "criterion": self.config.gnm.weight_criterion,
    #                         "success": True
    #                     }
    #             except Exception as e:
    #                 print(f"    Weight optimization failed: {e}")
    #                 network_result["weight_optimization"] = {
    #                     "success": False,
    #                     "error": str(e)
    #                 }
            
    #         results["network_results"].append(network_result)
        
    #     # Save results
    #     results_file = exp_dir / "gnm_comprehensive_results.json"
    #     with open(results_file, 'w') as f:
    #         json.dump(results, f, indent=2, default=str)
        
    #     print(f"\nResults saved to: {results_file}")
        
    #     # Generate summary statistics
    #     summary = self._generate_summary(results)
    #     summary_file = exp_dir / "summary.json"
    #     with open(summary_file, 'w') as f:
    #         json.dump(summary, f, indent=2)
        
    #     return results
    

    def run_gnm_parameter_sweep(self,
                           target_network: Optional[torch.Tensor] = None, # ATTENTION: If this is none, then the first connectome of the binary connectomes is used....
                           experiment_name: Optional[str] = None, 
                           no_wandb: Optional[bool] = False,
                           random_sample: bool = False,
                           n_random_samples: int = 30, 
                           elaborate_analysis: Optional[bool] = False, 
                           **kwargs) -> Dict[str, Any]:
        """
        Run parameter sweep with integrated logging.
        
        Args:
            target_network: Optional specific target network
            experiment_name: Custom experiment name
            no_wandb: If True, disable wandb logging
            random_sample: If True, use random sampling instead of grid search
            n_random_samples: Number of random samples to use
            elaborate_analysis: If True, run detailed analysis and save all networks
            
        Returns:
            Sweep results
        """
        print("=" * 60)
        print("GNM PARAMETER SWEEP")
        print("=" * 60)
        
        # Set experiment name
        if not experiment_name:
            experiment_name = f"gnm_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Load data
        if target_network is None:
            if self.config.data.use_gnm_defaults:
                data = self.config.load_gnm_defaults()
                target_network = data["binary_network"]
                distance_matrix = data["distance_matrix"]
            else:
                distance_matrix = torch.tensor(
                    self.data_loader.load_distance_matrix(), 
                    dtype=torch.float32,
                    device=self.device
                )
                # Use first available connectome
                # Added fix previously - use first slice! 
                binary_connectomes = self.data_loader.load_binary_connectomes()
                first_density = sorted(binary_connectomes.keys())[0]
                target_network = torch.tensor(
                    binary_connectomes[first_density][0, :, :],
                    dtype=torch.float32,
                    device=self.device
                )
        else: # I do not understand what this here does: 
            # USES DEFAULTS: 
            if self.config.data.use_gnm_defaults:
                distance_matrix = self.config.load_gnm_defaults()["distance_matrix"] 
                binary_connectomes = self.config.load_gnm_defaults()["binary_connectomes"]
            else:
                print("Please either hand over binary connectomes or use 'use_gnm_defaults'.")
            #     distance_matrix = torch.tensor(
            #         self.data_loader.load_distance_matrix(),
            #         dtype=torch.float32,
            #         device=self.device
            #     )
        
        # Number of edges to generate (number of iterations per config)
        num_iterations = int(target_network.sum().item() // 2)
        num_simulations = 100
        
        # Create sweep configuration based on sampling method
        if no_wandb and random_sample:
            sweep_config = self.config.create_gnm_random_sweep_config(
                distance_matrix=distance_matrix,
                num_iterations=num_iterations,
                num_simulations=num_simulations,
                method="random", # method, # should be "random"
                n_random_samples=n_random_samples,
                include_weights=True
            )
        else:
            sweep_config = self.config.create_gnm_sweep_config(
                distance_matrix=distance_matrix,
                num_iterations=num_iterations,
                method="grid", # method, # should be "grid" 
                num_simulations=num_simulations,
                include_weights=True
            )
            
        # Get evaluation criteria
        evaluation_criteria = self.config.get_gnm_evaluation_criteria(distance_matrix)
          
        #############
        
        # --- NEW CORE LOGIC WITH INCREMENTAL SAVING ---
        from tqdm import tqdm
        from gnm.fitting import perform_run
        import pandas as pd
        import numpy as np
        from src.structural_analysis.graph_measures import analyze_connectomes
        from src.ESNs.test_memory_capacity_weighted import evaluate_memory_capacity_from_connectome

        # Define the path for the output CSV before the loop starts
        csv_path = self.config.paths.current_projects_output_dir / f"{experiment_name}_results.csv"
        print(f"Results will be incrementally saved to: {csv_path}")

        for run_config in tqdm(sweep_config, desc="Configuration Iterations"):
            # 1. Run the basic GNM simulation
            experiment = perform_run(
                run_config=run_config,
                binary_evaluations=[evaluation_criteria],
                real_binary_matrices=target_network.unsqueeze(0),
                save_model=True, # Temporarily keep model for elaborate analysis
                save_run_history=False,
                device=self.device,
            )

            # 2. Prepare the data record for this iteration
            flat_record = {}
            # Get parameters
            params = experiment.run_config.binary_parameters
            flat_record.update({
                "eta": float(params.eta), "gamma": float(params.gamma),
                "distance_relationship_type": str(params.distance_relationship_type),
                "preferential_relationship_type": str(params.preferential_relationship_type),
                "generative_rule": str(params.generative_rule.__class__.__name__),
                "num_iterations": int(params.num_iterations),
            })
            # Get standard GNM evaluation results
            # flat_record.update(experiment.evaluation_results.binary_evaluations)
            
            # Decision: Option 1: Tracking ALL individual energy scores 
            # energy_metric_name = list(experiment.evaluation_results.binary_evaluations.keys())[0] # as it's only one key until now. 
            # energy_values_arr = experiment.evaluation_results.binary_evaluations[energy_metric_name].reshape(-1).numpy()
            # energy_values_arr = str(energy_values_arr.tolist())
            # flat_record.update({energy_metric_name: energy_values_arr})
            # energy_values_arr_as_string = str(...)
            
            # Decision Option 2: Only tracking mean energy score 
            energy_metric_name = list(experiment.evaluation_results.binary_evaluations.keys())[0] # + "_mean" # as it's only one key until now. 
            energy_value_mean = experiment.evaluation_results.binary_evaluations[energy_metric_name].mean().item() # reshape(-1).numpy()
            # energy_values_arr = str(energy_values_arr.tolist())
            flat_record.update({energy_metric_name: energy_value_mean})
            
            
            # 3. If elaborate_analysis is true, run your custom code
            if elaborate_analysis and experiment.model:
                networks_np = experiment.model.adjacency_matrix.cpu().numpy()
                output_dir = self.config.paths.current_projects_output_dir

                # (The logic for saving networks remains the same)
                rule_name = params.generative_rule.__class__.__name__
                filename = f"net_eta{params.eta.item():.3f}_gamma{params.gamma.item():.3f}_rule{rule_name}.npy"
                save_path = output_dir / "generated_networks" / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, networks_np)

                # Calculate graph and ESN measures and add them to the record
                graph_measures_list = analyze_connectomes(
                    connectomes=networks_np, distance_matrix=run_config.distance_matrix.cpu().numpy()
                )
                flat_record.update(pd.DataFrame(graph_measures_list).mean().to_dict())

                try:
                    mc_lags_to_calc = [1, 2, 5, 10, 30, 50]
                    esn_results = [evaluate_memory_capacity_from_connectome(
                        connectome=net, mc_lengths=mc_lags_to_calc, train_len=1000, n_runs=5, spectral_radius=0.99
                    ) for net in networks_np]
                    df_esn = pd.DataFrame(esn_results)
                    numeric_cols = [c for c in df_esn.columns if 'mc' in c and 'indiv' not in c]
                    flat_record.update(df_esn[numeric_cols].mean().to_dict())
                except Exception as e:
                    print(f"    [Warning] ESN eval failed: {e}. Logging NaN.")
                    mc_keys = ["mc_mean", "mc_std"] + [f"mc_length_{l}" for l in mc_lags_to_calc]
                    flat_record.update({key: np.nan for key in mc_keys})

            # 4. Append the flattened record to the CSV file
            df_to_append = pd.DataFrame([flat_record])
            df_to_append.to_csv(
                csv_path,
                mode='a',
                header=not csv_path.exists(), # Write header only if file doesn't exist
                index=False
            )
            # Clear model from memory to prevent overuse
            experiment.model = None
            
        print("\nSweep completed! All iterations have been saved.")
        return {"status": "success", "output_file": str(csv_path)}
    
        #############

        print(f"Running parameter sweep...")
        print(f"  - Generative rules: {self.config.gnm.generative_rules_to_test}")
        print(f"  - Simulations per parameter set: {self.config.gnm.num_simulations}")
        print(f"  - Wandb logging: {'DISABLED' if no_wandb else 'ENABLED'}")
        
        # Better print statements based on sampling method
        if no_wandb and random_sample:
            print(f"  - Random sampling: {n_random_samples} parameter combinations")
            print(f"  - Eta range: {self.config.gnm.eta_range}")
            print(f"  - Gamma range: {self.config.gnm.gamma_range}")
        else:
            print(f"  - Parameter grid: {self.config.gnm.n_eta} × {self.config.gnm.n_gamma}")
            print(f"  - Eta range: {self.config.gnm.eta_range}")
            print(f"  - Gamma range: {self.config.gnm.gamma_range}")
            
        # Initialize wandb if requested
        if not no_wandb: 
            try:
                import wandb
                if not wandb.run:
                    
                    # FIXED: Use dynamic wandb directory based on config
                    wandb_dir = self.config.paths.output_dir / "wandb"
                    wandb_dir.mkdir(parents=True, exist_ok=True)

                    wandb.init(
                        project="GNM_Pipeline",
                        name=experiment_name,
                        dir=str(wandb_dir),  # Use config-based directory
                        config={
                            "n_eta": self.config.gnm.n_eta,
                            "n_gamma": self.config.gnm.n_gamma,
                            "num_simulations": self.config.gnm.num_simulations,
                            "generative_rules": self.config.gnm.generative_rules_to_test, 
                        }
                    )
            except:
                pass  # Wandb not available or not needed
    
        # Determine method based on flags
        if no_wandb:
            if random_sample:
                method = "random"
            else:
                method = "grid"
        else:
            method = "bayesian"

        # from pathlib import Path 
        # given_output_path = Path("/Users/adrian/Documents/01_projects/14_4D_lab/output/gnm/00_gnm_experiments/00_gnm_XXXX")
        experiments, path_for_the_binary_csv_file = fitting.perform_sweep(
            sweep_config=sweep_config,
            binary_evaluations=[evaluation_criteria],
            # real_binary_matrices=target_network.unsqueeze(0), # TODO: SHOULD ACTUALLY BE INCLUDED. should be of size (n_subj, n_nodes, n_nodes)
            real_binary_matrices=torch.Tensor(binary_connectomes[first_density]), # should be of size (n_subj, n_nodes, n_nodes)
            # real_binary_matrices=Float[target_network, 'num_real_binary_networks num_nodes num_nodes'], # needs to be jax
            method=method,
            num_bayesian_runs=200 if not no_wandb else n_random_samples if random_sample else None,
            weighted_evaluations=None,
            # given_output_path=given_output_path,
            save_model=False, # True,
            # experiment_name=experiment_name,
            save_run_history=True,
            verbose=True,
            wandb_logging=not no_wandb,
            # no_different_project_name=True,
            device=self.device, 
            elaborate_analysis=elaborate_analysis,
            output_dir=self.config.paths.current_projects_output_dir,
        )

        # Log all experiments using centralized logger
        self.logger.log_gnm_sweep(
            experiments=experiments,
            evaluation_criteria=evaluation_criteria,
            config=self.config,
            experiment_name=experiment_name
        )
        
        # Find optimal parameters
        optimal_experiments, optimal_energies = fitting.optimise_evaluation(
            experiments=experiments,
            criterion=evaluation_criteria
        )
        
        # Create experiment directory
        exp_dir = self.config.paths.gnm_output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sweep results summary
        results = {
            "experiment_name": experiment_name,
            "n_experiments": len(experiments),
            "optimal_parameters": [],
            "parameter_grid": {
                "eta_range": self.config.gnm.eta_range,
                "gamma_range": self.config.gnm.gamma_range,
                "n_eta": self.config.gnm.n_eta,
                "n_gamma": self.config.gnm.n_gamma,
                "sampling_method": "random" if (no_wandb and random_sample) else "grid" if no_wandb else "bayesian",
                "n_random_samples": n_random_samples if (no_wandb and random_sample) else None
            }
        }
        
        # Add top 10 optimal parameters
        for exp, energy in zip(optimal_experiments[:10], optimal_energies[:10]):
            results["optimal_parameters"].append({
                "eta": float(exp.run_config.binary_parameters.eta),
                "gamma": float(exp.run_config.binary_parameters.gamma),
                "energy": float(energy),
                "generative_rule": str(exp.run_config.binary_parameters.generative_rule)
            })
        
        path_for_the_binary_csv_folder = path_for_the_binary_csv_file.parent
        results_file = path_for_the_binary_csv_folder / "sweep_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSweep completed!")
        print(f"Best parameters:")
        if results["optimal_parameters"]:
            best = results["optimal_parameters"][0]
            print(f"  - eta: {best['eta']:.3f}")
            print(f"  - gamma: {best['gamma']:.3f}")
            print(f"  - energy: {best['energy']:.3f}")
        
        print(f"Results saved to: {results_file}")
        
        return results
    
    
    
    
    # def run_full_pipeline(self,
    #                      esn_experiment_name: Optional[str] = None,
    #                      gnm_experiment_name: Optional[str] = None) -> Dict[str, Any]:
    #     """
    #     Run complete pipeline with integrated logging.
        
    #     Args:
    #         esn_experiment_name: Custom ESN experiment name
    #         gnm_experiment_name: Custom GNM experiment name
            
    #     Returns:
    #         Combined results
    #     """
    #     print("=" * 60)
    #     print("FULL PIPELINE: ESN + GNM ANALYSIS")
    #     print("=" * 60)
        
    #     results = {"esn": None, "gnm": None}
        
    #     # Run ESN evaluation if data loader is available
    #     if self.esn_evaluator:
    #         print("\nPhase 1: ESN Memory Capacity Evaluation")
    #         print("-" * 40)
            
    #         # Set ESN experiment name
    #         if not esn_experiment_name:
    #             esn_experiment_name = f"esn_{time.strftime('%Y%m%d_%H%M%S')}"
            
    #         try:
    #             # Generate ESN hyperparameter grid
    #             hparam_grid = self.config.generate_esn_hparam_grid(
    #                 spectral_radii=np.linspace(0.1, 2.5, 25),
    #                 input_lengths=[500, 1000, 2000, 3000, 4000],
    #                 input_scalings=[1.0, 1.5],
    #                 regularization_methods=["pinv", "ridge"],
    #                 densities=[10, 12, 14, 16, 18, 20]
    #             )
                
    #             # Load data
    #             weighted_by_density = self.data_loader.load_weighted_by_density()
                
    #             # Create experiment directory
    #             exp_dir = self.config.paths.esn_output_dir / esn_experiment_name
    #             exp_dir.mkdir(parents=True, exist_ok=True)
                
    #             # Run ESN sweep with enhanced logging
    #             class ESNEvaluatorWithLogging(ESNEvaluator):
    #                 """Extended ESN evaluator that logs to our centralized logger."""
                    
    #                 def __init__(self, evaluator, logger, experiment_name):
    #                     self.__dict__.update(evaluator.__dict__)
    #                     self.central_logger = logger
    #                     self.exp_name = experiment_name
                    
    #                 def _subject_job(self, subj_idx, A_obs, hparams, timing_flag=True, random_seed=None):
    #                     """Override to add logging."""
    #                     mc_result_dict, timing_dict = super()._subject_job(
    #                         subj_idx, A_obs, hparams, timing_flag, random_seed
    #                     )
                        
    #                     # Log to centralized logger
    #                     self.central_logger.log_esn_evaluation(
    #                         subject_id=subj_idx,
    #                         hyperparameters=hparams,
    #                         mc_result=mc_result_dict,
    #                         experiment_name=self.exp_name
    #                     )
                        
    #                     return mc_result_dict, timing_dict
                
    #             # Create wrapped evaluator with logging
    #             evaluator_with_logging = ESNEvaluatorWithLogging(
    #                 self.esn_evaluator, self.logger, esn_experiment_name
    #             )
                
    #             # Run ESN sweep
    #             esn_message = evaluator_with_logging.run_hyperparameter_sweep(
    #                 connectomes=weighted_by_density,
    #                 hparam_grid=hparam_grid,
    #                 save_dir=exp_dir,
    #                 search_mode="random_sample"
    #             )
                
    #             # Load and analyze results
    #             esn_analysis = self.esn_evaluator.load_and_analyze_results(exp_dir)
    #             print("Loaded and analyzed ESN results.")
                
    #             print(esn_analysis)  # TODO: Remove again
                
    #             results["esn"] = {
    #                 "status": "success",
    #                 "message": esn_message,
    #                 "analysis": esn_analysis,
    #                 "experiment_dir": str(exp_dir)
    #             }
                
                
    #         except Exception as e:
    #             results["esn"] = {
    #                 "status": "failed",
    #                 "error": str(e)
    #             }
    #             print(f"ESN evaluation failed: {e}")
        
    #     # Run GNM analysis
    #     print("\nPhase 2: GNM Comprehensive Analysis")
    #     print("-" * 40)
        
    #     try:
    #         gnm_results = self.run_gnm_comprehensive_analysis(
    #             experiment_name=gnm_experiment_name,
    #             compare_rules=True,
    #             fit_weights=True
    #         )
            
    #         results["gnm"] = {
    #             "status": "success",
    #             "results": gnm_results
    #         }
            
    #     except Exception as e:
    #         results["gnm"] = {
    #             "status": "failed",
    #             "error": str(e)
    #         }
    #         print(f"GNM analysis failed: {e}")
        
    #     print("\n" + "=" * 60)
    #     print("PIPELINE COMPLETED")
    #     print("=" * 60)
        
    #     # Finalize logging session
    #     self.logger.finalize()
        
    #     return results
    
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        summary = {
            "n_networks_analyzed": len(results.get("network_results", [])),
            "generative_rules_tested": results.get("generative_rules_tested", []),
            "best_parameters": {},
            "average_metrics": {}
        }
        
        # Collect all parameters and metrics
        all_etas = []
        all_gammas = []
        all_energies = []
        
        for network_result in results.get("network_results", []):
            if "parameter_fitting" in network_result:
                all_etas.append(network_result["parameter_fitting"]["best_eta"])
                all_gammas.append(network_result["parameter_fitting"]["best_gamma"])
                all_energies.append(network_result["parameter_fitting"]["best_energy"])
        
        if all_etas:
            summary["best_parameters"] = {
                "eta_mean": np.mean(all_etas),
                "eta_std": np.std(all_etas),
                "gamma_mean": np.mean(all_gammas),
                "gamma_std": np.std(all_gammas),
                "energy_mean": np.mean(all_energies),
                "energy_std": np.std(all_energies)
            }
        
        return summary


    def run_esn_only_analysis(self,
                            experiment_name: Optional[str] = None,
                            search_mode: str = "random_sample") -> Dict[str, Any]:
        """
        Run ESN analysis only.
        
        Args:
            experiment_name: Custom ESN experiment name
            search_mode: "grid" or "random_sample"
            
        Returns:
            ESN results
        """
        if not self.esn_evaluator:
            raise ValueError("ESN evaluator not available. Check data configuration.")
        
        print("=" * 60)
        print("ESN MEMORY CAPACITY ANALYSIS")
        print("=" * 60)
        
        # Set ESN experiment name
        if not experiment_name:
            experiment_name = f"esn_only_{time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Generate ESN hyperparameter grid
            hparam_grid = self.config.generate_esn_hparam_grid(
                spectral_radii=np.linspace(0.1, 2.5, 25),
                input_lengths=[500, 1000, 2000, 3000, 4000],
                input_scalings=[1.0, 1.5],
                regularization_methods=["pinv", "ridge"],
                densities=[10, 12, 14, 16, 18, 20]
            )
            
            # Load data
            weighted_by_density = self.data_loader.load_weighted_by_density()
            
            # Create experiment directory
            exp_dir = self.config.paths.esn_output_dir / experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Running ESN hyperparameter sweep with {len(hparam_grid)} combinations...")
            print(f"Search mode: {search_mode}")
            
            # Run ESN sweep
            esn_message = self.esn_evaluator.run_hyperparameter_sweep(
                connectomes=weighted_by_density,
                hparam_grid=hparam_grid,
                save_dir=exp_dir,
                search_mode=search_mode
            )
            
            # Load and analyze results
            esn_analysis = self.esn_evaluator.load_and_analyze_results(exp_dir)
            
            results = {
                "status": "success",
                "message": esn_message,
                "analysis": esn_analysis,
                "experiment_dir": str(exp_dir),
                "n_combinations_tested": len(hparam_grid),
                "search_mode": search_mode
            }
            
            print(f"\nESN analysis completed!")
            if "best_overall_mc" in esn_analysis:
                best = esn_analysis["best_overall_mc"]
                print(f"Best memory capacity: {best['value']:.4f}")
                best_hp = best["hyperparameters"]
                print(f"Best hyperparameters:")
                print(f"  - Density: {best_hp.get('density_percent')}%")
                print(f"  - Spectral radius: {best_hp.get('spectral_radius'):.3f}")
                print(f"  - Input length: {best_hp.get('input_length')}")
            
            print(f"Results saved to: {exp_dir}")
            
            return results
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "experiment_name": experiment_name
            }


    def run_gnm_esn_grid_evaluation(self,
                                    target_network: Optional[torch.Tensor] = None,
                                    eta_range: tuple = (-8, 0),
                                    gamma_range: tuple = (0.2, 8),
                                    n_eta: int = 20,
                                    n_gamma: int = 20,
                                    experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate GNM networks on a parameter grid and evaluate ESN memory capacity.
        
        Args:
            target_network: Target connectome for determining n_edges
            eta_range: Range for eta parameter
            gamma_range: Range for gamma parameter
            n_eta: Number of eta values
            n_gamma: Number of gamma values
            experiment_name: Name for the experiment
            
        Returns:
            Dictionary with grid evaluation results
        """
        print("=" * 60)
        print("GNM-ESN GRID EVALUATION")
        print("=" * 60)
        
        if not experiment_name:
            experiment_name = f"gnm_esn_grid_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Load data
        if target_network is None:
            if self.config.data.use_gnm_defaults:
                data = self.config.load_gnm_defaults()
                target_network = data["binary_network"]
                distance_matrix = data["distance_matrix"]
            else:
                distance_matrix = torch.tensor(
                    self.data_loader.load_distance_matrix(),
                    dtype=torch.float32,
                    device=self.device
                )
                binary_connectomes = self.data_loader.load_binary_connectomes()
                first_density = sorted(binary_connectomes.keys())[0]
                target_network = torch.tensor( # Calculated thus only from first subject (i.e. representative connectome)
                    binary_connectomes[first_density][:, :, 0],
                    dtype=torch.float32,
                    device=self.device
                )
        else:
            distance_matrix = torch.tensor(
                self.data_loader.load_distance_matrix(),
                dtype=torch.float32,
                device=self.device
            )
        
        n_nodes = target_network.shape[0] # Calculated thus only from first subject (i.e. representative connectome)
        n_edges = int(target_network.sum().item() // 2) # Calculated thus only from first subject (i.e. representative connectome)
        
        # Create parameter grid
        eta_values = np.linspace(eta_range[0], eta_range[1], n_eta)
        gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
        
        # Results storage
        grid_results = []
        
        # Use matching index as default rule
        from src.GNMs.gnm_network_generator import GNMParameters
        from gnm import generative_rules
        
        print(f"Evaluating {n_eta} x {n_gamma} = {n_eta * n_gamma} parameter combinations...")
        
        # Generate and evaluate networks for each parameter combination
        for i, eta in enumerate(eta_values):
            for j, gamma in enumerate(gamma_values):
                print(f"  Processing η={eta:.3f}, γ={gamma:.3f} ({i*n_gamma + j + 1}/{n_eta*n_gamma})")
                
                # Generate GNM network
                params = GNMParameters(
                    eta=eta,
                    gamma=gamma,
                    generative_rule=generative_rules.MatchingIndex()
                )
                
                try:
                    # Generate network
                    gnm_network = self.gnm_generator.generate_network(
                        n_nodes=n_nodes,
                        n_edges=n_edges,
                        distance_matrix=distance_matrix,
                        parameters=params
                    )
                    
                    # Convert to numpy for ESN evaluation
                    gnm_network_np = gnm_network.cpu().numpy()
                    
                    # Evaluate ESN memory capacity
                    esn_result = self.esn_evaluator.evaluate_single_subject(
                        subject_idx=0,
                        connectome=gnm_network_np,
                        hparams={
                            "spectral_radius": self.config.esn.spectral_radius,
                            "input_length": self.config.esn.input_length,
                            "input_scaling": self.config.esn.input_scaling,
                            "regularization_method": self.config.esn.regularization_method,
                            "n_runs": 5  # Reduce for speed
                        }
                    )
                    
                    mc_mean = esn_result.get("mc_mean", 0.0)
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    mc_mean = 0.0
                
                grid_results.append({
                    "eta": eta,
                    "gamma": gamma,
                    "mc_mean": mc_mean
                })
        
        # Also fit best parameters to empirical data
        print("\nFitting to empirical connectomes...")
        empirical_fits = []
        
        # Load empirical weighted connectomes
        weighted_by_density = self.data_loader.load_weighted_by_density()
        
        for density, connectomes in weighted_by_density.items():
            print(f"  Fitting density {density}%...")
            # Use first subject as representative. # IMPORTANT! 
            empirical_conn = torch.tensor(
                connectomes[:, :, 0],
                dtype=torch.float32,
                device=self.device
            )
            
            # Binarize for GNM fitting
            empirical_binary = (empirical_conn > 0).float()
            
            # Fit GNM parameters
            fit_result = self.gnm_generator.fit_parameters(
                target_network=empirical_binary, 
                distance_matrix=distance_matrix,
                generative_rule_name="matching_index",
                n_eta=20,
                n_gamma=20,
                num_simulations=10,
                evaluation_metrics=["degree_ks", "clustering_ks"]
            )
            
            empirical_fits.append({
                "density": density,
                "best_eta": fit_result["best_eta"],
                "best_gamma": fit_result["best_gamma"],
                "best_energy": fit_result["best_energy"]
            })
        
        # Save results
        exp_dir = self.config.paths.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "experiment_name": experiment_name,
            "eta_range": eta_range,
            "gamma_range": gamma_range,
            "n_eta": n_eta,
            "n_gamma": n_gamma,
            "grid_results": grid_results,
            "empirical_fits": empirical_fits
        }
        
        results_file = exp_dir / "gnm_esn_grid_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Create visualization
        from visualization import PipelineVisualizer
        visualizer = PipelineVisualizer(output_dir=exp_dir)
        
        # Convert to DataFrame for visualization
        df = pd.DataFrame(grid_results)
        
        # Plot memory capacity landscape
        fig, ax = visualizer.plot_gnm_esn_landscape(
            df=df,
            empirical_fits=empirical_fits,
            title="Memory Capacity Landscape",
            savepath=exp_dir / "mc_landscape.png"
        )
        
        return results


    def run_dynamic_gnm_generation(self, experiment_name: Optional[str] = None):
        """
        Runs the full dynamically-aware GNM generation and evaluation pipeline.
        """
        print("=" * 60)
        print("DYNAMICALLY-AWARE GNM GENERATION")
        print("=" * 60)
        
        if not experiment_name:
            experiment_name = f"dynamic_gnm_{time.strftime('%Y%m%d_%H%M%S')}"

        # --- Data Loading ---
        # (Assuming non-default data for this advanced task)
        distance_matrix = torch.tensor(
            self.data_loader.load_distance_matrix(), dtype=torch.float32
        )
        binary_connectomes = self.data_loader.load_binary_connectomes()
        
        # Use first subject of first density as target
        density = sorted(binary_connectomes.keys())[0]
        target_network = binary_connectomes[density][0, :, :]
        n_edges = int(target_network.sum() // 2)
        n_nodes = target_network.shape[-1]

        # --- Initialize Dynamic Generator ---
        from src.GNMs.gnm_network_generator import DynamicGNMGenerator, GNMParameters
        
        dynamic_generator = DynamicGNMGenerator(self.esn_evaluator, self.config.gnm.device)
        
        # --- Set up Parameters ---
        rule_name = self.config.gnm.generative_rules_to_test[0] # Use first rule from config. TODO
        gnm_params = GNMParameters(
            eta=self.config.gnm.eta_range[0], # Example: use start of range. TODO
            gamma=self.config.gnm.gamma_range[0],
            generative_rule=dynamic_generator.AVAILABLE_RULES[rule_name]()
        )
        dynamic_config = {
            "dynamic_delta": self.config.gnm.dynamic_delta,
            "candidate_pool_size": self.config.gnm.candidate_pool_size,
            "fast_esn_eval": self.config.gnm.fast_esn_eval
        }

        # --- Run Generation ---
        
        fitness_metric = self.config.gnm.dynamic_fitness_metric
        
        generated_network = dynamic_generator.generate_dynamically_aware_network(
            n_nodes=n_nodes,
            n_edges=n_edges,
            distance_matrix=distance_matrix,
            gnm_params=gnm_params,
            dynamic_config=dynamic_config,
            fitness_metric=fitness_metric
        )

        # --- Evaluation and Logging ---
        print("Evaluating final generated network...")
        
        final_eval_hparams = {
            "calculate_criticality": True,
            "calculate_info_dynamics": True
        }
        
        final_esn_eval = self.esn_evaluator.evaluate_single_subject(
            subject_idx=0,
            connectome=generated_network.cpu().numpy(), # is float 32??, 68x68
            hparams=final_eval_hparams
        )
        
        self.logger.log_run(
            run_type="dynamic_gnm_generation",
            experiment_name=experiment_name,
            parameters={
                "eta": gnm_params.eta,
                "gamma": gnm_params.gamma,
                "delta": dynamic_config['dynamic_delta'],
                "rule": rule_name,
                "candidate_pool_size": dynamic_config['candidate_pool_size'], 
                "fitness_metric_used": fitness_metric 
            },
            results={
                "final_mc_mean": final_esn_eval.get("mc_mean"),
                "final_mc_std": final_esn_eval.get("mc_std"), 
                "final_branching_ratio": final_esn_eval.get("branching_ratio"),
                "final_avg_transfer_entropy": final_esn_eval.get("avg_transfer_entropy"),
                "final_avg_active_info": final_esn_eval.get("avg_active_info"),
                "final_info_balance": final_esn_eval.get("info_balance")
            },
            metadata={
                "target_density": density,
                "n_nodes": n_nodes,
                "n_edges": n_edges
            }
        )
        
        fitness_metric = "mc_mean" # TODO: REMOVE HARDCODED-NESS!!! 
        
        try:
            results_dir = self.config.paths.current_projects_output_dir
            csv_path = results_dir / "dynamic_gnm_results.csv"

            # Flatten the nested fast_esn_eval dictionary for clear CSV columns
            fast_esn_params = {f"fast_{k}": v for k, v in dynamic_config.get('fast_esn_eval', {}).items()}

            # Consolidate all parameters and results into a single dictionary for the CSV row
            result_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "experiment_name": experiment_name,
                "fitness_metric_used": fitness_metric, # e.g. "mc_mean" 
                "target_density": density,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "eta": gnm_params.eta,
                "gamma": gnm_params.gamma,
                "generative_rule": rule_name,
                "dynamic_delta": dynamic_config['dynamic_delta'],
                "candidate_pool_size": dynamic_config['candidate_pool_size'],
                **fast_esn_params,  # Add the flattened fast ESN params
                "final_mc_mean": final_esn_eval.get("mc_mean"),
                "final_mc_std": final_esn_eval.get("mc_std"),
                "final_branching_ratio": final_esn_eval.get("branching_ratio"),
                "final_avg_transfer_entropy": final_esn_eval.get("avg_transfer_entropy"),
                "final_avg_active_info": final_esn_eval.get("avg_active_info"),
                "final_info_balance": final_esn_eval.get("info_balance"),
                "random_seed": self.config.compute.random_seed
            }

            # Create a DataFrame and save/append to the CSV file
            # This mode handles creating the file with a header if it doesn't exist,
            # and appending a new row without a header if it does.
            df = pd.DataFrame([result_row])
            df.to_csv(csv_path, mode='a', index=False, header=not csv_path.exists())
            
            print(f"Dynamic GNM results appended to CSV: {csv_path}")

        except Exception as e:
            print(f"[Warning] Could not save dynamic GNM results to CSV: {e}")
        
        self.logger.finalize(self.config.paths.current_projects_output_dir)
        print("Run complete and logged.")
        np.save(self.config.paths.current_projects_output_dir / "generated_network.npy", generated_network.cpu().numpy())
        
        return {"status": "success", "network": generated_network}




def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description="GNM-Optimized Connectome Analysis Pipeline")
    # Choice between GNM (different options), and ESN
    parser.add_argument("command", choices=["sweep", "esn", "test", "full", "gnm_esn_grid"], # "full", "comprehensive", "test", "minimal_test",
                   help="Command to run: Either GNM (different versions) or ESN")
    
    parser.add_argument("--experiment-name", help="Custom experiment name")
    
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    
    parser.add_argument("--wandb-project", default="GNM_Pipeline",
                       help="Wandb project name") # TODO: Not sure how well wandb works for ESN. 
    
    parser.add_argument("--default_wandb_project_name", default="GNMs", # TODO: Deprecated! Remove... 
                       help="Default wandb project name")
    
    
    # GNM-specific arguments: Configuration
    parser.add_argument("--config", choices=["quick_test", "comprehensive", "rule_comparison", "minimal_test"],
                       default="quick_test", help="Choice of configuration")
    
    parser.add_argument("--compare-rules", action="store_true",
                       help="Compare different generative rules. Not perfectly implemented yet.")
    
    parser.add_argument("--fit-weights", action="store_true",
                       help="Optimize edge weights. Not perfectly implemented yet.")

    
    # GNM-specific arguments: Random sampling option
    parser.add_argument("--random-sample", action="store_true",
                    help="Use random sampling instead of grid search when wandb is disabled")

    parser.add_argument("--n-random-samples", type=int, default=30,
                    help="Number of random samples for parameter sweep (default: 30)")


    # ESN-specific arguments
    parser.add_argument("--esn-search-mode", choices=["grid", "random_sample"], 
                    default="random_sample", help="ESN search mode")
    
    parser.add_argument("--esn-random-sample-size", type=int, 
                    help="Number of random samples for ESN random search. Will be multiplied by number of subjects - i.e. setting this parameter to 50 will lead to 3500 tasks for 70 subjects.")
    
    
    args = parser.parse_args()

    
    # Load configuration
    if args.config == "quick_test":
        config = get_gnm_quick_test_config()
    elif args.config == "comprehensive":
        config = get_gnm_comprehensive_config()
    elif args.config == "rule_comparison":
        from config import get_gnm_rule_comparison_config
        config = get_gnm_rule_comparison_config()
    else:
        config = ConfigManager()
        
    # Initialize wandb if not disabled
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name or f"{args.command}_{time.strftime('%Y%m%d_%H%M%S')}",
                config={
                    "command": args.command,
                    "config_preset": args.config,
                    "compare_rules": args.compare_rules,
                    "fit_weights": args.fit_weights
                }
            )
        except Exception as e:
            print(f"Wandb initialization failed: {e}")
    
    # Create orchestrator
    orchestrator = GNMandESNPipelineOrchestrator(config)
    
    try:
        if args.command == "sweep":
            results = orchestrator.run_gnm_parameter_sweep(
                experiment_name=args.experiment_name, 
                no_wandb=args.no_wandb,
                random_sample=args.random_sample,
                n_random_samples=args.n_random_samples
            )
            
        elif args.command == "test":
            print("Running quick test with GNM defaults...")
            config = get_gnm_quick_test_config()
            orchestrator = GNMandESNPipelineOrchestrator(config)
            results = orchestrator.run_gnm_parameter_sweep(
                experiment_name=args.experiment_name or "quick_test",
                no_wandb=args.no_wandb,
                random_sample=args.random_sample,
                n_random_samples=args.n_random_samples
            )
            print(f"Test completed successfully!")
    
        elif args.command == "minimal_test":
            print("Running minimal test with GNM defaults...")
            config = get_gnm_minimal_config()
            orchestrator = GNMandESNPipelineOrchestrator(config)
            results = orchestrator.run_gnm_parameter_sweep(
                experiment_name=args.experiment_name or "minimal_test",
                no_wandb=args.no_wandb,
                random_sample=args.random_sample,
                n_random_samples=args.n_random_samples
            )
            print(f"Test completed successfully!")
            
        elif args.command == "comprehensive":
            results = orchestrator.run_gnm_comprehensive_analysis(
                experiment_name=args.experiment_name,
                compare_rules=args.compare_rules,
                fit_weights=args.fit_weights
            )
  
        elif args.command == "esn":
            
            # Use ESN-specific configuration
            from config import get_esn_config
            esn_config = get_esn_config()
          
            # Create ESN evaluator directly
            from src.ESNs.esn_evaluation import create_esn_evaluator
            esn_evaluator = create_esn_evaluator(config_manager=esn_config)
            
            # Generate hyperparameter grid
            hparam_grid = esn_config.generate_esn_hparam_grid()

            # Load data
            from src.utils.data_loader import DataLoader
            data_loader = DataLoader(esn_config)
            weighted_by_density = data_loader.load_weighted_by_density()
            
            # Run ESN sweep
            exp_dir = esn_config.paths.esn_output_dir / (args.experiment_name or f"esn_{time.strftime('%Y%m%d_%H%M%S')}")
            esn_message = esn_evaluator.run_hyperparameter_sweep(
                connectomes=weighted_by_density,
                hparam_grid=hparam_grid,
                save_dir=exp_dir,
                search_mode=args.esn_search_mode,
                random_sample_size=args.esn_random_sample_size
            )
            print(esn_message)
            
        elif args.command == "full":
            results = orchestrator.run_full_pipeline(
                esn_experiment_name=args.experiment_name,
                gnm_experiment_name=args.experiment_name
            )
            print(f"Full pipeline completed successfully!")
            
        elif args.command == "gnm_esn_grid": 
            results = orchestrator.run_gnm_esn_grid_evaluation(
                experiment_name=args.experiment_name
            )
            print(f"GNM-ESN grid evaluation completed successfully!")
        
        print("Final Results:")
        print(results)
        print("----")
        # Always finalize logger at the end
        orchestrator.logger.finalize()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        orchestrator.logger.finalize()
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        orchestrator.logger.finalize()
        
        
if __name__ == "__main__":
    # main(argparse)
    main()
    
    
        
    """
        ### ESN ONLY -> this works. TODO: Test with more args. 
        python src/connectome_analysis/main_pipeline_2.py esn --experiment-name "my_esn_analysis"
        python src/connectome_analysis/main_pipeline_2.py esn --esn-search-mode grid --no-wandb --experiment-name "grid_esn_analysis"
        python src/connectome_analysis/main_pipeline_2.py esn --esn-search-mode grid --no-wandb --experiment-name "esn"
        
        ### RANDOM SEARCH
        python src/connectome_analysis/main_pipeline_2.py sweep --no-wandb --random-sample --experiment-name "gnm_random_local"
        
        ### GRID SEARCH 
        python src/connectome_analysis/main_pipeline_2.py sweep --no-wandb --compare-rules --config minimal_test --experiment-name "gnm"
        python src/connectome_analysis/main_pipeline_2.py sweep --no-wandb --experiment-name "gnm"
        
        
        # -> This works. 
        python src/connectome_analysis/main_pipeline_2.py esn --esn-search-mode random_sample --esn-random-sample-size 50 --no-wandb --experiment-name "esn_main_pipeline_2"

        ### GNM-ESN GRID EVALUATION (Not sure yet if it works)
        python src/connectome_analysis/main_pipeline_2.py gnm_esn_grid --experiment-name "gnm_esn_grid" --no-wandb
        
        # TODO: "full" and "gnm_esn_grid" commands not tested yet.
    

        ########## GNM-ESN COMPARATIVE ANALYSIS ##########

        python src/connectome_analysis/gnm_and_esn_orchestrator.py sweep --no-wandb --experiment-name "gnm"

    """


   