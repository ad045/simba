"""
Run experiments across multiple animals by modifying the config file.
"""

import yaml
import subprocess
import sys
import time
from pathlib import Path

# Configuration
BASE_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_suarez_MaMI_dataset.yaml"
# BASE_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_debug.yaml"
TEMP_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/temp_config_animal.yaml"
# ANIMAL_START = 55
# ANIMAL_END = 5 # 2 # 255
RUN_SCRIPT = "run_experiment.py"

NUMBER_RUNS_PER_ANIMAL = 10 # 11 # 50 # number of LOOPS - total number is: this times n_samples multiplied. 



def modify_config_for_animal(base_config_path: str, temp_config_path: str, animal_id: int): # , loop_number: int):
    """
    Load the base config, modify the animal number, and save to temp file.
    
    Args:
        base_config_path: Path to the original config file
        temp_config_path: Path where modified config will be saved
        animal_id: Animal ID to set in the config
    """
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify the animal number
    config['experiment']['animal'] = animal_id
    
    # Optionally modify the experiment name to include animal ID
    original_name = config['experiment']['name']
    # config['experiment']['name'] = f"{original_name}/{original_name}_animal_{animal_id}"
    config['experiment']['name'] = f"{original_name}_animal_{animal_id}" # _loop_{loop_number}"
    
    # Save modified config
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created temp config for animal {animal_id}")


def run_experiment_for_animal(animal_id: int, temp_config_path: str, run_script: str):
    """
    Run the experiment script with the temporary config.
    
    Args:
        animal_id: Animal ID being processed
        temp_config_path: Path to the temporary config file
        run_script: Path to the run_experiment.py script
    """
    print(f"\n{'='*80}")
    print(f"🐾 Starting experiment for Animal {animal_id}")
    print(f"{'='*80}\n")
    
    try:
        # Run the experiment script
        # Note: The script uses HARDCODED_CONFIG_PATH, so we need to modify that
        # or pass the config path as an argument
        result = subprocess.run(
            [sys.executable, run_script, temp_config_path, "--num_runs", str(NUMBER_RUNS_PER_ANIMAL)],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✅ Completed experiment for Animal {animal_id}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running experiment for Animal {animal_id}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted during Animal {animal_id}")
        raise





def cleanup_temp_config(temp_config_path: str):
    """Remove the temporary config file."""
    try:
        Path(temp_config_path).unlink(missing_ok=True)
        print(f"🧹 Cleaned up temporary config file")
    except Exception as e:
        print(f"⚠️ Could not remove temp config: {e}")


def main():
    """Main execution loop."""
    start_time = time.time()
    animals_to_analyze = [0, 206, 188, 169, 103] # range(ANIMAL_START, ANIMAL_END + 1)
    total_animals = len(animals_to_analyze)
    successful = 0
    failed = 0
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting batch experiment run")
    # print(f"   Animals: {ANIMAL_START} to {ANIMAL_END} (total: {total_animals})")
    print(f"   Base config: {BASE_CONFIG_PATH}")
    print(f"{'='*80}\n")
    
    try:
        
        # 0,Rat4,
        # 103,Orangutan2,
        # 169,RedKangaroo3,
        # 188,FruitBat5,
        # 206,Chimpanzee,


        for animal_id in animals_to_analyze:   #  range(ANIMAL_START, ANIMAL_END + 1):
            try:
                # Create modified config
                modify_config_for_animal(BASE_CONFIG_PATH, TEMP_CONFIG_PATH, animal_id)
                
                # Run experiment
                success = run_experiment_for_animal(animal_id, TEMP_CONFIG_PATH, RUN_SCRIPT)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                # Clean up after each run
                cleanup_temp_config(TEMP_CONFIG_PATH)
                
                
                # ############
                
                # # Create modified config
                # TEMP_CONFIG_PATH_2 = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_suarez_MaMI_dataset_focused.yaml"
                # modify_config_for_animal(BASE_CONFIG_PATH, TEMP_CONFIG_PATH_2, animal_id)
                
                # # Run experiment
                # success = run_experiment_for_animal(animal_id, TEMP_CONFIG_PATH_2, RUN_SCRIPT)
                
                # if success:
                #     successful += 1
                # else:
                #     failed += 1
                
                # # Clean up after each run
                # cleanup_temp_config(TEMP_CONFIG_PATH_2)
                
                
                #############
                
            except KeyboardInterrupt:
                print("\n\n⚠️ User interrupted the batch run")
                cleanup_temp_config(TEMP_CONFIG_PATH)
                break
            except Exception as e:
                print(f"\n❌ Unexpected error for Animal {animal_id}: {e}")
                failed += 1
                cleanup_temp_config(TEMP_CONFIG_PATH)
                continue
            
            #########################################################################
            # try:
            #     # Create modified config
            #     TEMP_CONFIG_PATH_2 = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_suarez_MaMI_dataset_focused.yaml"
            #     modify_config_for_animal(BASE_CONFIG_PATH, TEMP_CONFIG_PATH_2, animal_id)
                
            #     # Run experiment
            #     success = run_experiment_for_animal(animal_id, TEMP_CONFIG_PATH_2, RUN_SCRIPT)
                
            #     if success:
            #         successful += 1
            #     else:
            #         failed += 1
                
            #     # Clean up after each run
            #     cleanup_temp_config(TEMP_CONFIG_PATH_2)
                
            # except KeyboardInterrupt:
            #     print("\n\n⚠️ User interrupted the batch run")
            #     cleanup_temp_config(TEMP_CONFIG_PATH_2)
            #     break
            # except Exception as e:
            #     print(f"\n❌ Unexpected error for Animal {animal_id}: {e}")
            #     failed += 1
            #     cleanup_temp_config(TEMP_CONFIG_PATH_2)
            #     continue
            
            
            
    
    finally:
        duration = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"📊 Batch Run Summary")
        print(f"{'='*80}")
        print(f"   Total animals processed: {successful + failed}/{total_animals}")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️ Total time: {duration:.2f}s ({duration/60:.2f} min)")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()