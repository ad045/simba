#!/bin/bash

###############################################################
# Configuration

# CONFIG_FILE="/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_shafiei_human_consensus_dataset.yaml"
CONFIG_FILE="/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/config_gnm_run_suarez_MaMI_dataset.yaml"
NUMBER_RUNS=300 # 300 # 300 # 250 # 500 # 1 #3 # 250


###############################################################
# Extract experiment name from YAML file

echo "Extracting experiment name from '$CONFIG_FILE'..."

# Using grep and awk (works on most systems)
EXPERIMENT_NAME=$(grep -E "^\s*name:" "$CONFIG_FILE" | awk -F':' '{gsub(/^[ \t]*/, "", $2); gsub(/[ \t]*#.*$/, "", $2); gsub(/["'"'"']/, "", $2); print $2}')

# Check if experiment name was extracted successfully
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "❌ Error: Could not extract experiment name from '$CONFIG_FILE'"
    echo "Please check that the YAML file contains a 'name:' field under 'experiment:'"
    exit 1
fi

echo "✅ Extracted experiment name: '$EXPERIMENT_NAME'"


###############################################################
# Define the main output directory using extracted name 

OUTPUT_DIR="output/gnm/$EXPERIMENT_NAME"

echo "Using output directory: '$OUTPUT_DIR'"
echo


###############################################################
# Run the Python experiment

echo "Running run_experiment with the '$CONFIG_FILE' config file."

for (( i=1; i<=$NUMBER_RUNS; i++ ))
do
   echo "--- Starting run #$i ---"
   python run_experiment.py "$CONFIG_FILE"
   echo "--- Finished run #$i ---"
done

echo "All '$NUMBER_RUNS' runs completed."
echo # Adding a blank line for readability



###############################################################

echo "Combining 'results' and 'indiv_connectome' CSVs each..."
python src/utils/combine_csvs.py "$OUTPUT_DIR" 
echo "Combined individual connectome energy CSVs."
echo
