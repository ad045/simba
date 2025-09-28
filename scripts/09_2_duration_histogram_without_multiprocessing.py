import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Define the root directory to search
# Replace '.' with the path to your main folder if not running from there
root_directory = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes") # Path('.') 

all_durations = []

# Recursively find all 'session_summary.json' files
for json_path in root_directory.rglob('session_summary.json'):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Check if 'duration_seconds' exists and is a number
            if 'duration_seconds' in data and isinstance(data['duration_seconds'], (int, float)):
                all_durations.append(data['duration_seconds'])
    except (json.JSONDecodeError, IOError) as e:
        print(f"Could not read or parse {json_path}: {e}")

# --- Calculate and Print Total Duration ---
if all_durations:
    total_duration_seconds = sum(all_durations)
    print(f"Found {len(all_durations)} session files.")
    print(f"Total duration: {total_duration_seconds:.2f} seconds")
    # Convert to a more readable format (Hours:Minutes:Seconds)
    h = int(total_duration_seconds // 3600)
    m = int((total_duration_seconds % 3600) // 60)
    s = int(total_duration_seconds % 60)
    print(f"Which is {h} hours, {m} minutes, and {s} seconds.")

    # --- Plot Histogram ---
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    # Use numpy for better bin calculation if you have many data points
    bins = min(30, len(all_durations)) # Cap the number of bins at 30
    plt.hist(all_durations, bins=bins, edgecolor='black', alpha=0.75)
    
    plt.title('Histogram of Session Durations', fontsize=16)
    plt.xlabel('Duration (seconds)', fontsize=12)
    plt.ylabel('Number of Sessions', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No 'session_summary.json' files with valid durations found.")