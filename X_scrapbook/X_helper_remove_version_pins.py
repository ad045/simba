import re

# --- Configuration ---
input_filename = "/Users/adrian/Documents/01_projects/14_4D_lab/environment_ma_thesis.yml" # environment.yml"  # The file conda created
output_filename = "/Users/adrian/Documents/01_projects/14_4D_lab/environment_ma_thesis_automatically_cleaned.yml" # environment.clean.yml"  # The cleaned file to use
python_version_to_keep = "3.13"
# --- End Configuration ---

# List of high-level packages you know you need.
# Add your key packages here. The script will keep these and python/pip.
# Leave empty to just rely on the pip list.
packages_to_keep = [
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "nilearn",
    "h5py",
    "hydra-core",
]

print(f"Reading from {input_filename}...")

with open(input_filename, "r") as f:
    lines = f.readlines()

cleaned_lines = []
in_pip_section = False
dependencies_started = False
essential_conda_packages = {"python", "pip"} | set(packages_to_keep)

for line in lines:
    stripped_line = line.strip()

    # Handle conda dependencies
    if stripped_line == "dependencies:":
        dependencies_started = True
        cleaned_lines.append(line)
        # Add the essential packages
        cleaned_lines.append(f"  - python={python_version_to_keep}\n")
        for pkg in sorted(list(set(packages_to_keep) | {"pip"})):
             cleaned_lines.append(f"  - {pkg}\n")
        continue

    # Start of pip section
    if stripped_line == "- pip:":
        in_pip_section = True
        cleaned_lines.append(line)
        continue

    # Skip all conda packages once the section has started
    if dependencies_started and not in_pip_section:
        continue

    # Handle pip dependencies
    if in_pip_section and stripped_line.startswith("-"):
        # Remove version specifiers like ==, >=, <, etc.
        package_name = re.split(r"[=<>~]", stripped_line)[0].strip("- ").strip()
        if package_name: # Avoid adding empty lines
            cleaned_lines.append(f"      - {package_name}\n")
    # Keep other lines (name, channels) as is, if not in dependencies
    elif not dependencies_started:
         cleaned_lines.append(line)


# Write the cleaned output
with open(output_filename, "w") as f:
    f.writelines(cleaned_lines)

print(f"✅ Successfully created cleaned file: {output_filename}")