"""
Centralized logging module for pipeline runs.
Handles both local JSON logging and wandb integration.
"""
# -> Used at least for main_pipeline_2.py


import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import numpy as np

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
            if output_dir is None:
                # Find project root and set default output directory
                current = Path.cwd()
                for parent in (current, *current.parents):
                    if any((parent / marker).exists() for marker in ("pyproject.toml", "setup.cfg", ".git")):
                        output_dir = parent / "output"
                        break
                else:
                    output_dir = current / "output"
                    
            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a session ID for this execution
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_start = datetime.now().isoformat()
            
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
        if not exp_summary["run_types"]:  # Is this sensible? Think if there's a smarter way to do so 
            exp_summary["run_types"] = [] # .add(run_type)
        exp_summary["run_types"].append(run_type)
        exp_summary["last_update"] = run_record["timestamp"]
        
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
        
        This version extracts standard GNM evaluation metrics and, if available,
        the extended results from the 'elaborate_analysis' feature.
        
        Args:
            experiments: List of GNM experiment objects.
            evaluation_criteria: Evaluation criteria used.
            config: The configuration object for the run.
            experiment_name: Name of the experiment.
        """

        for i, exp in enumerate(experiments):
            try:
                # Extract GNM parameters from the run configuration for safety
                params_obj = exp.run_config.binary_parameters
                parameters = {
                    "eta": float(params_obj.eta),
                    "gamma": float(params_obj.gamma),
                    "distance_relationship_type": str(params_obj.distance_relationship_type),
                    "preferential_relationship_type": str(params_obj.preferential_relationship_type),
                    "generative_rule": str(params_obj.generative_rule.__class__.__name__),
                    "num_iterations": int(params_obj.num_iterations),
                }

                # Extract standard GNM evaluation results (e.g., DegreeKS)
                results = {k: v for k, v in exp.evaluation_results.binary_evaluations.items()}

                # Check for, validate, and add elaborate analysis results
                if hasattr(exp.evaluation_results, 'elaborate_results'):
                    elaborate_data = exp.evaluation_results.elaborate_results
                    # Clean data: ensure all values are JSON-serializable Python natives
                    cleaned_elaborate_data = {
                        key: float(val) if isinstance(val, (np.floating, np.integer)) else val
                        for key, val in elaborate_data.items()
                    }
                    results.update(cleaned_elaborate_data)

                # Define metadata for the run
                metadata = {
                    "num_simulations": exp.run_config.num_simulations,
                    "device": config.gnm.device,
                    "rank_in_sweep": i
                }
                
                # Log the complete, flattened record
                self.log_run(
                    run_type="gnm_sweep_elaborate" if 'mc_mean' in results else "gnm_sweep",
                    experiment_name=experiment_name,
                    parameters=parameters,
                    results=results,
                    metadata=metadata
                )
                
            except Exception as e:
                import traceback
                print(f"[ERROR] Could not log experiment {i}. Reason: {e}\n{traceback.format_exc()}")

    
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
    
    def _save_session_summary(self, current_projects_output_dir) -> None:
        """Save the session summary to JSON."""
        summary_session_path = Path(current_projects_output_dir) / "session_summary.json"
        with open(summary_session_path, 'a') as f: 
            json.dump(self.session_summary, f, indent=2, default=str)
            f.write("\n")
        print("Saved session summary to", summary_session_path)


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
    
    def finalize(self, current_projects_output_dir) -> None:
        """Finalize the logging session."""
        self.session_summary["end_time"] = datetime.now().isoformat()
        self.session_summary["duration_seconds"] = (
            datetime.now() - datetime.fromisoformat(self.session_start)
        ).total_seconds()
        
        try:    
            self._save_session_summary(current_projects_output_dir)
        except Exception as e:
            print(f"Error saving session summary: {e}")
        
        print(f"\nLogging session completed:")
        print(f"  Session ID: {self.session_id}")
        print(f"  Total runs logged: {self.session_summary['total_runs']}")
        
    
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