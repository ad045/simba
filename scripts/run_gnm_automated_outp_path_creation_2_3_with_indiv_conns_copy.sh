#!/bin/bash

# --- Configuration ---
CONFIG_FILE="/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/example_gnm_random_5_four_factors_in_energy.yaml" # example_gnm_random_5_four_factors_in_energy.yaml"
# CONFIG_FILE="/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/configs/example_gnm_random_3_indiv_connectomes.yaml"
NUMBER_RUNS=300 # 250 # 500 # 1 #3 # 250

# --- New Flags for Cleanup Control ---
CLEANUP_GENERATED_NETWORKS=false # Set to false to keep 'generated_networks' folders
DELETE_SUBFOLDERS=true           # Set to false to keep individual (i.e. the second run of a yaml file) subfolders (e.g., "..._20250913_111422")

# --- Extract experiment name from YAML file ---
echo "Extracting experiment name from '$CONFIG_FILE'..."

# Method 1: Using grep and awk (works on most systems)
EXPERIMENT_NAME=$(grep -E "^\s*name:" "$CONFIG_FILE" | awk -F':' '{gsub(/^[ \t]*/, "", $2); gsub(/[ \t]*#.*$/, "", $2); gsub(/["'"'"']/, "", $2); print $2}')

# Method 2: Alternative using sed (uncomment if Method 1 doesn't work)
# EXPERIMENT_NAME=$(grep -E "^\s*name:" "$CONFIG_FILE" | sed -E 's/^\s*name:\s*//; s/\s*#.*$//; s/["\047]//g')

# Check if experiment name was extracted successfully
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "❌ Error: Could not extract experiment name from '$CONFIG_FILE'"
    echo "Please check that the YAML file contains a 'name:' field under 'experiment:'"
    exit 1
fi

echo "✅ Extracted experiment name: '$EXPERIMENT_NAME'"

# --- Define the main output directory using extracted name ---
OUTPUT_DIR="output/gnm/$EXPERIMENT_NAME"

echo "Using output directory: '$OUTPUT_DIR'"
echo



# # --- 2. Run the Python experiment ---
# echo "Running run_experiment with the '$CONFIG_FILE' config file."

# for (( i=1; i<=$NUMBER_RUNS; i++ ))
# do
#    echo "--- Starting run #$i ---"
#    python run_experiment.py "$CONFIG_FILE"
#    echo "--- Finished run #$i ---"
# done

# echo "All '$NUMBER_RUNS' runs completed."
# echo # Adding a blank line for readability


# echo "Combining 'results' and 'indiv_connectome' CSVs each..."
# python src/utils/combine_csvs.py "$OUTPUT_DIR" 
# echo "Combined individual connectome energy CSVs."
# echo



# --- 4. Clean up unnecessary files ---
echo "Starting cleanup..."

if [ "$CLEANUP_GENERATED_NETWORKS" = true ] ; then
    # Delete 'generated_networks' folders
    find "$OUTPUT_DIR" -type d -name "generated_networks" -exec rm -rf {} +
    echo "Deleted 'generated_networks' directories."
else
    echo "Skipping deletion of 'generated_networks' directories."
fi

# # Delete 'session_summary.json' files
# find "$OUTPUT_DIR" -type f -name "session_summary.json" -exec rm -f {} +
# echo "Deleted 'session_summary.json' files."
# echo


# Find all session summary files, handling spaces/special characters
readarray -t files < <(find "$OUTPUT_DIR" -type f -name "session_summary.json")

if [ ${#files[@]} -gt 0 ]; then
  # Aggregate data from all found files into a new total summary file
  jq -s '
    {
      "earliest_start_time": (map(.start_time) | min),
      "latest_end_time": (map(.end_time) | max),
      "sum_of_durations_seconds": (map(.duration_seconds) | add),
      "durations_seconds": map(.duration_seconds)
    }
  ' "${files[@]}" > "$OUTPUT_DIR/session_summary_total.json"

  echo "Created 'session_summary_total.json'."
  
  # Delete the original files
  rm -f "${files[@]}"
  echo "Deleted individual 'session_summary.json' files."
else
  echo "No 'session_summary.json' files found to process."
fi

echo


# --- 5. Process Configs, Calculate Duration, and Save Summary ---
echo "Processing config files, calculating duration, and saving summary..."

# Define the summary file path
SUMMARY_FILE="$OUTPUT_DIR/experiment_summary.txt"

# This block's output will be printed to the console AND saved to the summary file
{
    # Find all config files
    CONFIG_FILES=($(find "$OUTPUT_DIR" -type f -name "config.yaml"))

    if [ ${#CONFIG_FILES[@]} -gt 0 ]; then
        # Check if all configs are identical by comparing checksums
        UNIQUE_CHECKSUMS=$(find "$OUTPUT_DIR" -type f -name "config.yaml" -exec md5 -q {} + | sort -u | wc -l)

        if [ "$UNIQUE_CHECKSUMS" -eq 1 ]; then
            echo "All config.yaml files are identical."
            # Move the first one to the output directory
            mv "${CONFIG_FILES[0]}" "$OUTPUT_DIR/experiment_config_file.yaml"
            echo "Moved one config.yaml to the root and deleting others."
            # Now, delete all original config files that are still in subdirectories
            find "$OUTPUT_DIR" -path "*/config.yaml" -delete
        else
            echo "Warning: Not all config.yaml files are identical. No action taken."
        fi

        # Get a sorted list of all experiment subdirectories before they are deleted
        SUBDIRS=($(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort))
        
        echo "------------------------------------------------------------------"
        echo "Experiment name: $EXPERIMENT_NAME"
        echo "Number of loops: $NUMBER_RUNS."

        # Check if there are any subdirectories
        if [ ${#SUBDIRS[@]} -gt 0 ]; then
            # Extract timestamp from the first and last directory names
            FIRST_DIR_NAME=$(basename "${SUBDIRS[0]}")
            LATEST_DIR_NAME=$(basename "${SUBDIRS[${#SUBDIRS[@]}-1]}")

            START_TIMESTAMP=$(echo "$FIRST_DIR_NAME" | grep -o '[0-9]\{8\}_[0-9]\{6\}')
            END_TIMESTAMP=$(echo "$LATEST_DIR_NAME" | grep -o '[0-9]\{8\}_[0-9]\{6\}')
            
            # Convert timestamps to epoch seconds (syntax for macOS date)
            START_EPOCH=$(date -j -f "%Y%m%d_%H%M%S" "$START_TIMESTAMP" "+%s")
            END_EPOCH=$(date -j -f "%Y%m%d_%H%M%S" "$END_TIMESTAMP" "+%s")
            
            # Calculate duration
            DURATION=$((END_EPOCH - START_EPOCH))
            
            echo "Earliest run started at: $START_TIMESTAMP"
            echo "Latest run started at:   $END_TIMESTAMP"

            # Format and display the duration
            echo "Total duration was: $(date -u -r $DURATION +'%H hours, %M minutes and %S seconds')"
        fi
    else
        echo "No config.yaml files found to process."
    fi
    echo "------------------------------------------------------------------"

} | tee "$SUMMARY_FILE"


# --- 6. Final cleanup of experiment subfolders ---
if [ "$DELETE_SUBFOLDERS" = true ] ; then
    
    # --- 6a. Consolidate 'generated_networks' before deleting ---
    DEST_DIR="$OUTPUT_DIR/all_generated_networks"
    echo "Consolidating all 'generated_networks' into '$DEST_DIR'..."
    mkdir -p "$DEST_DIR"

    # Find all 'generated_networks' directories within the subfolders
    for network_dir in $(find "$OUTPUT_DIR" -mindepth 2 -type d -name "generated_networks"); do
        # Get the name of the parent directory (the unique run folder) to use as a prefix
        parent_dir_name=$(basename "$(dirname "$network_dir")")
        echo "  -> Processing files from '$parent_dir_name'"

        # Loop through each file and move it with a unique prefix to avoid overwrites
        for file_path in "$network_dir"/*; do
            if [ -f "$file_path" ]; then
                file_name=$(basename "$file_path")
                # Move the file, prefixing it with its original parent folder's name
                mv "$file_path" "$DEST_DIR/${parent_dir_name}_${file_name}"
            fi
        done
    done
    echo "Consolidation complete."
    echo

    # --- 6b. Delete the original subfolders ---
    # Find all original subdirectories, making sure not to target the new consolidated one
    SUBDIRS_TO_DELETE=($(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -not -name "$(basename "$DEST_DIR")"))
    
    if [ ${#SUBDIRS_TO_DELETE[@]} -gt 0 ]; then
        echo "Deleting individual experiment subfolders..."
        rm -rf "${SUBDIRS_TO_DELETE[@]}"
        echo "Deleted ${#SUBDIRS_TO_DELETE[@]} subfolders."
    else
        echo "No subfolders to delete."
    fi

else
    echo "Skipping deletion of experiment subfolders."
fi
echo

echo "✅ Experiment and cleanup finished."
echo "Final output directory: '$OUTPUT_DIR'"