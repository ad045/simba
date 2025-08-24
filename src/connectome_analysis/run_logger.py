"""
Centralized logging module for pipeline runs.
Handles both local JSON logging and wandb integration.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading


class RunLogger:
    """Centralized logger for all pipeline runs."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, output_dir: Optional[Path] = None):
        """Singleton pattern to ensure one logger instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the logger with output directory."""
        if not hasattr(self, 'initialized'):
            self.output_dir = output_dir # or Path("./output") TODO: ADD DEFAULT PATH HERE
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a session ID for this execution
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_start = datetime.now().isoformat()
            
            # Paths for different log files
            self.all_runs_path = self.output_dir / f"all_runs_{self.session_id}.jsonl"
            self.session_summary_path = self.output_dir / f"session_summary_{self.session_id}.json"
            self.global_log_path = self.output_dir / "global_runs.jsonl"  # Persistent across sessions
            
            # Initialize session summary
            self.session_summary = {
                "session_id": self.session_id,
                "start_time": self.session_start,
                "runs": [],
                "total_runs": 0,
                "experiments": {}
            }
            
            # Check wandb availability
            self._wandb_available = self._check_wandb()
            
            self.initialized = True
    
    def _check_wandb(self) -> bool:
        """Check if wandb is available and initialized."""
        try:
            import wandb
            return wandb.run is not None
        except:
            return False
    
    def log_run(self, 
                run_type: str,
                experiment_name: str,
                parameters: Dict[str, Any],
                results: Dict[str, Any],
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a single run to all destinations.
        
        Args:
            run_type: Type of run (e.g., "gnm_sweep", "esn_eval", "gnm_comprehensive")
            experiment_name: Name of the experiment
            parameters: Run parameters (eta, gamma, spectral_radius, etc.)
            results: Run results (energy, memory_capacity, etc.)
            metadata: Additional metadata
        """
        # Create run record
        run_record = {
            "session_id": self.session_id,
            "run_id": f"{self.session_id}_{self.session_summary['total_runs']}",
            "timestamp": datetime.now().isoformat(),
            "run_type": run_type,
            "experiment_name": experiment_name,
            "parameters": parameters,
            "results": results,
            "metadata": metadata or {}
        }
        
        # Log to session file (JSONL format - one line per run)
        self._append_jsonl(self.all_runs_path, run_record)
        
        # Log to global file
        self._append_jsonl(self.global_log_path, run_record)
        
        # Update session summary
        self.session_summary["total_runs"] += 1
        self.session_summary["runs"].append({
            "run_id": run_record["run_id"],
            "run_type": run_type,
            "timestamp": run_record["timestamp"]
        })
        
        # Track experiments
        if experiment_name not in self.session_summary["experiments"]:
            self.session_summary["experiments"][experiment_name] = {
                "run_count": 0,
                "run_types": set(),
                "start_time": run_record["timestamp"]
            }
        
        exp_summary = self.session_summary["experiments"][experiment_name]
        exp_summary["run_count"] += 1
        exp_summary["run_types"].add(run_type)
        exp_summary["last_update"] = run_record["timestamp"]
        
        # Convert set to list for JSON serialization
        for exp in self.session_summary["experiments"].values():
            if isinstance(exp.get("run_types"), set):
                exp["run_types"] = list(exp["run_types"])
        
        # Save session summary
        self._save_session_summary()
        
        # Log to wandb if available
        if self._wandb_available:
            self._log_to_wandb(run_type, parameters, results)
    
    def log_gnm_sweep(self,
                     experiments: List[Any],
                     evaluation_criteria: Any,
                     config: Any,
                     experiment_name: str) -> None:
        """
        Specialized method for logging GNM parameter sweep results.
        
        Args:
            experiments: List of GNM experiment objects
            evaluation_criteria: Evaluation criteria used
            config: Configuration object
            experiment_name: Name of the experiment
        """
        for i, exp in enumerate(experiments):
            try:
                print("DEBUG: Logging GNM experiment", i)
                # Extract parameters
                bp = exp.run_config.binary_parameters
                
                parameters = {
                    "eta": float(bp.eta),
                    "gamma": float(bp.gamma),
                    "generative_rule": str(bp.generative_rule),
                    "num_iterations": getattr(bp, 'num_iterations', None),
                    "distance_relationship": getattr(bp, 'distance_relationship_type', 'powerlaw')
                }
                print("DEBUG: Getting parameters worked") 
                
                # Extract results
                try:
                    energy = float(exp.evaluation_dict[evaluation_criteria])
                except:
                    energy = float(exp.evaluation_dict.get(str(evaluation_criteria), float("nan")))
                
                # Get individual energies if available
                individual_energies = {}
                if hasattr(exp, 'evaluation_results') and hasattr(exp.evaluation_results, 'binary_evaluations'):
                    for j, eval_result in enumerate(exp.evaluation_results.binary_evaluations):
                        individual_energies[f"sim_{j}"] = float(eval_result) if eval_result is not None else None
                
                print("DEBUG: Getting results worked")
                
                results = {
                    "energy": energy,
                    "individual_energies": individual_energies,
                    "rank": i
                }
                
                metadata = {
                    "num_simulations": config.gnm.num_simulations,
                    "device": config.gnm.device,
                    "sweep_method": "bayesian"
                }
                
                print("DEBUG: Getting metadata worked")
                
                self.log_run(
                    run_type="gnm_sweep",
                    experiment_name=experiment_name,
                    parameters=parameters,
                    results=results,
                    metadata=metadata
                )
                print(f"[{i+1}/{len(experiments)}] GNM experiment logged successfully.")
                
            except Exception as e:
                print(f"[Warning] Could not log experiment {i}: {e}")
    
    def log_esn_evaluation(self,
                          subject_id: int,
                          hyperparameters: Dict[str, Any],
                          mc_result: Dict[str, Any],
                          experiment_name: str) -> None:
        """
        Log ESN evaluation results.
        
        Args:
            subject_id: Subject identifier
            hyperparameters: ESN hyperparameters
            mc_result: Memory capacity results
            experiment_name: Name of the experiment
        """
        parameters = {
            "subject": subject_id,
            **hyperparameters
        }
        
        results = {
            "mc_mean": mc_result.get("mc_mean"),
            "mc_std": mc_result.get("mc_std"),
            "mean_mc_of_individual_runs": mc_result.get("mean_mc_of_individual_runs")
        }
        
        self.log_run(
            run_type="esn_evaluation",
            experiment_name=experiment_name,
            parameters=parameters,
            results=results
        )
    
    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        """Append a record to a JSONL file."""
        with open(path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    
    def _save_session_summary(self) -> None:
        """Save the session summary to JSON."""
        with open(self.session_summary_path, 'w') as f:
            json.dump(self.session_summary, f, indent=2, default=str)
    
    def _log_to_wandb(self, run_type: str, parameters: Dict, results: Dict) -> None:
        """Log to wandb if available."""
        try:
            import wandb
            
            # Flatten nested dictionaries for wandb
            log_dict = {
                f"{run_type}/param_{k}": v for k, v in parameters.items()
                if not isinstance(v, (dict, list))
            }
            log_dict.update({
                f"{run_type}/result_{k}": v for k, v in results.items()
                if not isinstance(v, (dict, list))
            })
            
            wandb.log(log_dict)
        except Exception as e:
            pass  # Silently fail wandb logging
    
    def finalize(self) -> None:
        """Finalize the logging session."""
        self.session_summary["end_time"] = datetime.now().isoformat()
        self.session_summary["duration_seconds"] = (
            datetime.now() - datetime.fromisoformat(self.session_start)
        ).total_seconds()
        
        self._save_session_summary()
        
        print(f"\nLogging session completed:")
        print(f"  Session ID: {self.session_id}")
        print(f"  Total runs logged: {self.session_summary['total_runs']}")
        print(f"  Session log: {self.all_runs_path}")
        print(f"  Session summary: {self.session_summary_path}")
        print(f"  Global log: {self.global_log_path}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        return {
            "session_id": self.session_id,
            "total_runs": self.session_summary["total_runs"],
            "experiments": list(self.session_summary["experiments"].keys()),
            "run_types": list(set(
                run["run_type"] for run in self.session_summary["runs"]
            ))
        }


# Convenience function for backward compatibility
def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a record to a JSONL file (backward compatibility)."""
    with open(path, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


# Global logger instance getter
def get_logger(output_dir: Optional[Path] = None) -> RunLogger:
    """Get or create the global logger instance."""
    return RunLogger(output_dir)