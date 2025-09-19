#!/bin/bash

# --- 1. Define the main output directory ---
# Using a variable makes the script easier to maintain.
OUTPUT_DIR="output/gnm/07_run_ridge"
CONFIG_FILE="configs/example_gnm_random.yaml"
NUMBER_RUNS=1 #3 # 250

# --- New Flags for Cleanup Control ---
CLEANUP_GENERATED_NETWORKS=true # false # Set to false to keep 'generated_networks' folders
DELETE_SUBFOLDERS=true # false         # Set to false to keep individual run subfolders (e.g., "..._20250913_111422")


# --- 2. Run the Python experiment 3 times ---
echo "Running run_experiment with the '$CONFIG_FILE' config file."

for (( i=1; i<=$NUMBER_RUNS; i++ ))
do
   echo "--- Starting run #$i ---"
   python run_experiment.py "$CONFIG_FILE"
   echo "--- Finished run #$i ---"
done

echo "All '$NUMBER_RUNS' runs completed."
echo # Adding a blank line for readability

# --- 3. Combine the resulting CSV files ---
echo "Combining CSVs from '$OUTPUT_DIR'..."
python src/utils/combine_csvs.py "$OUTPUT_DIR"
echo "Combined all CSVs into one big one."
echo

# --- 4. Clean up unnecessary files ---
echo "Starting cleanup..."

if [ "$CLEANUP_GENERATED_NETWORKS" = true ] ; then
    # Delete 'generated_networks' folders
    find "$OUTPUT_DIR" -type d -name "generated_networks" -exec rm -rf {} +
    echo "Deleted 'generated_networks' directories."
else
    echo "Skipping deletion of 'generated_networks' directories."
fi

# Delete 'session_summary.json' files
find "$OUTPUT_DIR" -type f -name "session_summary.json" -exec rm -f {} +
echo "Deleted 'session_summary.json' files."
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
    # Get the list of subdirectories again in case it's needed
    SUBDIRS_TO_DELETE=($(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d))
    if [ ${#SUBDIRS_TO_DELETE[@]} -gt 0 ]; then
        echo "Deleting individual experiment subfolders..."
        find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
        echo "Deleted ${#SUBDIRS_TO_DELETE[@]} subfolders."
    fi
else
    echo "Skipping deletion of experiment subfolders."
fi
echo

echo "✅ Experiment and cleanup finished."