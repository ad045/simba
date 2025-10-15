#!/usr/bin/env python3
"""
Script to run experiments for all animals (0-225).
Usage: python run_all_animals.py [--start 0] [--end 225] [--config path/to/config.yaml]
"""

import argparse
import subprocess
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime


class AnimalExperimentRunner:
    """Runs experiments across multiple animals with automatic config updating."""
    
    def __init__(self, config_path: str, start_animal: int, end_animal: int, num_runs: int = None):
        self.config_path = Path(config_path)
        self.start_animal = start_animal
        self.end_animal = end_animal
        self.num_runs = num_runs  # If None, will read from config (???)
        self.success_count = 0
        self.fail_count = 0
        self.failed_animals = []
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging
        self.log_dir = Path("logs") / self.config["experiment"]["name"]
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"all_animals_{timestamp}.log"
        
    def update_config(self, animal: int):
        """Update the animal number in the YAML config."""
        # with open(self.config_path, 'r') as f:
        #     config = yaml.safe_load(f)
        
        self.config['experiment']['animal'] = int(animal)
        self.config["experiment"]["name"] = str(self.config["experiment"]["name"]) + "_" + str(animal)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def run_animal_experiment(self, animal: int) -> bool:
        """Run experiment for a single animal. Returns True if successful."""
        print(f"\n{'='*60}")
        print(f"Processing animal: {animal} ({animal - self.start_animal + 1}/"
              f"{self.end_animal - self.start_animal + 1})")
        print(f"{'='*60}")
        
        try:
            # Update config
            self.update_config(animal)
            
            # Run experiment
            start_time = time.time()
            with open(self.log_file, 'a') as log:
                log.write(f"\n\n{'='*60}\n")
                log.write(f"Animal {animal} - {datetime.now()}\n")
                log.write(f"{'='*60}\n\n")
                
                result = subprocess.run(
                    [sys.executable, "run_experiment.py"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Animal {animal} completed successfully in {duration:.1f}s")
                self.success_count += 1
                return True
            else:
                print(f"❌ Animal {animal} FAILED after {duration:.1f}s")
                self.fail_count += 1
                self.failed_animals.append(animal)
                return False
                
        except Exception as e:
            print(f"❌ Animal {animal} FAILED with exception: {e}")
            self.fail_count += 1
            self.failed_animals.append(animal)
            with open(self.log_file, 'a') as log:
                log.write(f"\nEXCEPTION: {e}\n")
            return False
    
    def run_all(self):
        """Run experiments for all animals in range."""
        print("\n" + "="*60)
        print(f"Running experiments for animals {self.start_animal} to {self.end_animal}")
        print("="*60)
        print(f"Log file: {self.log_file}")
        
        script_start = time.time()
        
        for animal in range(self.start_animal, self.end_animal + 1):
            for n_run in range(self.num_runs): 
                self.run_animal_experiment(animal)
        
        # Print summary
        total_duration = time.time() - script_start
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Total animals processed: {self.end_animal - self.start_animal + 1}")
        print(f"✅ Successful: {self.success_count}")
        print(f"❌ Failed: {self.fail_count}")
        print(f"⏱️  Total time: {hours}h {minutes}m {seconds}s")
        
        if self.failed_animals:
            print(f"\n❌ Failed animals: {self.failed_animals}")
        
        print(f"\nFull log available at: {self.log_file}")
        print("="*60)
        
        return self.fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for multiple animals"
    )
    parser.add_argument(
        "--config",
        default="/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_suarez_MaMI_dataset.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting animal number (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=225,
        help="Ending animal number (default: 225)"
    )
    
    args = parser.parse_args()
    
    runner = AnimalExperimentRunner(args.config, args.start, args.end, num_runs=3)
    success = runner.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()