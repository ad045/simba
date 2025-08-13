import re
from pathlib import Path
from datetime import datetime


def time_stamp_for_saving():
    """
    Get current date and time.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_latest_file_name_of_data(dir_path, pattern="gnm_grid_results_*.csv"):
    """
    Get the latest file matching the pattern in the specified directory.
    Returns:
        (str,str) or None:
            - The latest file name or None if no matching files are found.
            - The timestamp of the latest file or None if no matching files are found.
    """

    # Glob all matching files
    dir_path = Path(dir_path)
    files = list(dir_path.glob(pattern))

    # Parse the timestamp in the filename
    pattern_parts = pattern.split("*")
    new_pattern = pattern_parts[0]+"(\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2})"+pattern_parts[1] # gnm_grid_results_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}).csv
    pattern = re.compile(new_pattern)

    # Small helper function to extract all timestamps of files available in this folder
    def extract_ts(path: Path) -> datetime:
        m = pattern.search(path.name)
        if not m:
            return datetime.min
        return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")

    # Get latest file if available. Otherwise: Return case, no files in folder available
    if files == []:
        return None
    latest_file = max(files, key=extract_ts)

    # Extract the timestamp from the file name
    file_name_str = str(latest_file)
    file_timestamp = file_name_str.split(".")[-2].split("_")[-2:]  # this gives us the time stamp

    # Combine it into a single string
    file_timestamp = "_".join(file_timestamp)
  
    return latest_file, file_timestamp