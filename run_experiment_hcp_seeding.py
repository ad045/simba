"""
Run experiments across multiple animals by modifying the config file.
"""

import yaml
import subprocess
import sys
import time
from pathlib import Path


RUN_SCRIPT = "run_experiment.py"

TEMP_CONFIG_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/temp_config_hcp_seed.yaml"

def modify_config_for_animal(base_config_path: str, temp_config_path: str, id: int): # , loop_number: int):
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
    # gnm seed_params seed_id seed_adjacency_matrices
    config['gnm']['seed_params']['seed_id'] = int(id) # otherwise numpy...
    
    # Modify the experiment name to include idx
    # original_name = config['experiment']['name']
    # config['experiment']['name'] = f"{original_name}/{original_name}_idx_{id}"
    original_name = config['experiment']['name']
    config['experiment']['name'] = f"{original_name}_idx_{id}" 
    # config['experiment']['name'] = f"{original_name}_animal_{id}" # _loop_{loop_number}"
    
    # # Save modified config
    # with open(temp_config_path, 'w') as f:
    #     yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    with open(temp_config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False) 
    
    
    print(f"✅ Created temp config for animal {id}")
    
    return config['experiment']['name']


def run_experiment_for_animal(animal_id: int, temp_config_path: str, run_script: str, number_of_runs_per_animal: int):
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
            [sys.executable, run_script, temp_config_path, "--num_runs", str(number_of_runs_per_animal)],
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


def main(
    path_to_config_file: str,
    number_of_runs_per_animal: int
):
    """Main execution loop."""
    start_time = time.time() 
    
    # TODO THIS IS REALLY ROUGH SO FAR....
    # seed_ids_to_analyze = [0,1]
    import numpy as np 
    rows = 10
    cols = 16
    arr = np.arange(rows * cols).reshape(rows, cols)
    seed_ids_to_analyze = arr.T.flatten()

    # seed_ids_to_analyze = [  0  16  32  48  64  80  96 112 128 144   1  17  33  49  65  81  97 113
    #     129 145   2  18  34  50  66  82  98 114 130 146   3  19  35  51  67  83
    #     99 115 131 147   4  20  36  52  68  84 100 116 132 148   5  21  37  53
    #     69  85 101 117 133 149   6  22  38  54  70  86 102 118 134 150   7  23
    #     39  55  71  87 103 119 135 151   8  24  40  56  72  88 104 120 136 152
    #     9  25  41  57  73  89 105 121 137 153  10  26  42  58  74  90 106 122
    #     138 154  11  27  43  59  75  91 107 123 139 155  12  28  44  60  76  92
    #     108 124 140 156  13  29  45  61  77  93 109 125 141 157  14  30  46  62
    #     78  94 110 126 142 158  15  31  47  63  79  95 111 127 143 159]
        #   seed_params:
            # seed_adjacency_matrices: /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/hcp_schaefer_100_dataset/05_seeds_of_humans/seeds_density2_16_regions_conns.npy # seeds_density2_16_regions_params.npy
            # seed_id: 0
            # Sweep from 0 to 159!!

    total_animals = len(seed_ids_to_analyze)
    successful = 0
    failed = 0

    temp_config_path = path_to_config_file.parent / "temp_config_hcp_seeds.yaml"
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting batch experiment run")
    # print(f"   Animals: {ANIMAL_START} to {ANIMAL_END} (total: {total_animals})")
    print(f"   Base config: {path_to_config_file}")
    print(f"{'='*80}\n")
    
    try:
        
        with open(path_to_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        
        # config['gnm']['seed_params']['use_seed_adjacency_matrices'])
        # 0,Rat4,
        # 103,Orangutan2,
        # 169,RedKangaroo3,
        # 188,FruitBat5,
        # 206,Chimpanzee,


        for seed_id in seed_ids_to_analyze:   #  range(ANIMAL_START, ANIMAL_END + 1):
            try:
                
                # Create output folder (THIS IS A HACK - BETTER TO DO IT INSIDE THE EXPERIMENT SCRIPT?)
                # output_dir = /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/hcp_schaefer_100_dataset/07_with_seeds/07_with_seeds_idx_200/all_metrics_for_07_with_seeds
                # Create modified config
                experiment_name = modify_config_for_animal(path_to_config_file, temp_config_path, seed_id)

                # Run experiment
                success = run_experiment_for_animal(seed_id, temp_config_path, RUN_SCRIPT, number_of_runs_per_animal)

                if success:
                    successful += 1
                else:
                    failed += 1
                
                # Clean up after each run
                cleanup_temp_config(temp_config_path)


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
                cleanup_temp_config(temp_config_path)
                break
            except Exception as e:
                print(f"\n❌ Unexpected error for Animal {seed_id}: {e}")
                failed += 1
                cleanup_temp_config(temp_config_path)
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

    return experiment_name


if __name__ == "__main__":

    path_to_config_file = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_hcp_with_seeds.yaml")  # BASE_CONFIG_PATH
    number_of_runs_per_animal = 1 # 11 # 50 # number of LOOPS - total number is: this times n_samples multiplied.

    experiment_name = main(
        path_to_config_file=path_to_config_file,
        number_of_runs_per_animal=number_of_runs_per_animal
    )