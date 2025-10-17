import json
import argparse
import os

def process_json_stream(input_file_path):
    """
    Reads a file containing a stream of JSON objects, calculates the total duration,
    and collects a list of all individual durations.

    Args:
        input_file_path (str): The path to the input file.
    """
    total_duration = 0.0
    individual_durations = []

    earliest_starting_time = 0
    latest_ending_time = 0 
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            # Read the entire file content. This approach is robust for files
            # that aren't excessively large.
            content = f.read()
            
            # The file consists of multiple JSON objects concatenated together.
            # We can split them by finding the boundaries. A simple way for this
            # structure is to replace the delimiter between objects with a unique
            # separator and then split the string.
            # This handles both `}{` and `}\n{` cases.
            json_strings = content.strip().replace('}\n{', '}|{').split('|')

            for json_str in json_strings:
                if not json_str:
                    continue
                try:
                    # Ensure the JSON string is properly formed if it was split
                    if not json_str.endswith('}'):
                        json_str += '}'
                    if not json_str.startswith('{'):
                       json_str = '{' + json_str

                    data = json.loads(json_str)
                    
                    # Safely get the duration, checking if it exists and is a number
                    duration = data.get('duration_seconds')
                    if isinstance(duration, (int, float)):
                        individual_durations.append(duration)
                        total_duration += duration
                    else:
                        print(f"Warning: 'duration_seconds' key missing or not a number in object starting with: {json_str[:50]}...")

                except json.JSONDecodeError:
                    print(f"Warning: Skipping segment that could not be parsed as JSON: {json_str[:50]}...")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Determine the output file path in the same folder as the input file
    base_name = os.path.splitext(input_file_path)[0]
    output_file_path = f"{base_name}_durations.txt"

    # Write the results to the output file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            
            f_out.write("=" * 60 + "\n")
            f_out.write("SUMMARY DURATIONS \n")
            f_out.write("=" * 60 + "\n\n")
            
            minutes, seconds = divmod(total_duration, 60)
            hours, minutes = divmod(minutes, 60)
            f_out.write(f"  Total Duration (seconds): {total_duration:.4f}\n")
            f_out.write(f"  Total Duration: {int(hours)}h {int(minutes)}m {seconds:.4f}s\n")
            f_out.write(f"  Remember: This is before checking the effects of multiprocessing\n\n")
            # f_out.write(f"  Time difference form first run ever to finishing of last run: {}\n\n")
                        
            f_out.write("=" * 60 + "\n\n")
            
            f_out.write("   Individual Durations (for histogram):\n")
            for dur in individual_durations:
                f_out.write(f"      {dur}\n")
        
        print(f"Processing complete. Results saved to: {output_file_path}")

    except IOError as e:
        print(f"Error: Could not write to output file '{output_file_path}'. Reason: {e}")


if __name__ == "__main__":
    # # Set up an argument parser to accept the input file path from the command line
    # parser = argparse.ArgumentParser(
    #     description="Calculate total and individual durations from a stream of JSON objects in a file."
    # )
    # parser.add_argument(
    #     "input_file", 
    #     help="Path to the input JSON file."
    # )
    
    # args = parser.parse_args()
    
    process_json_stream(
        input_file_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/suarez_MaMI_dataset/60_generally_finer_search_animal_1/session_summary.json"
    ) # args.input_file)
