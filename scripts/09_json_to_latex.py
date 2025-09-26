import yaml
import os
from pathlib import Path

def escape_latex(value):
    """Escapes special LaTeX characters in a string."""
    if value is None:
        return ""
    # Convert value to string, handling lists and other types
    if isinstance(value, list):
        s = ', '.join(map(str, value))
    else:
        s = str(value)

    # Dictionary of special characters and their LaTeX escaped equivalents
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    # Use a loop for replacements to handle multiple characters
    for char, escaped_char in replacements.items():
        s = s.replace(char, escaped_char)
    return s

def generate_latex_table(data, file_path):
    """Generates a LaTeX table from a nested dictionary."""
    latex_string = []

    # Keep document wrappers commented to produce a table snippet
    # latex_string.append(r"\documentclass{article}")
    # latex_string.append(r"\usepackage[a4paper, margin=1in]{geometry}")
    # latex_string.append(r"\usepackage{booktabs}")
    # latex_string.append(r"\usepackage{longtable}")
    # latex_string.append(r"\begin{document}")
    latex_string.append(r"\begin{longtable}{ll}")
    latex_string.append(r"\toprule")
    latex_string.append(r"\textbf{Parameter Group} & \textbf{Configuration} \\")
    latex_string.append(r"\midrule")
    latex_string.append(r"\endhead")
    latex_string.append(r"\bottomrule")
    latex_string.append(r"\endfoot")

    # Recursive function to process dictionary items
    def process_items(d, indent_level=0):
        for key, value in d.items():
            if isinstance(value, dict):
                # Add a section header for nested dictionaries - no indentation in the header text itself
                latex_string.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{escape_latex(key)}}}}} \\\\")
                # Process nested items with increased indentation
                process_items(value, indent_level + 1)
            else:
                # Regular key-value pair with proper indentation
                indent = r"\quad " * indent_level
                current_key_display = f"{indent}{escape_latex(key)}"
                latex_string.append(f"{current_key_display} & {escape_latex(value)} \\\\")

    # Start processing the main dictionary
    process_items(data)

    # LaTeX document footer
    latex_string.append(r"\end{longtable}")
    # latex_string.append(r"\end{document}")

    # Write to file
    with open(file_path, 'w') as f:
        f.write("\n".join(latex_string))
    print(f"LaTeX table successfully generated at: {file_path}")


# --- Main execution ---
config_filename = Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/output/gnm/16_big_sweep_with_individual_connectomes/16_big_sweep_with_individual_connectomes_20250925_071351/config.yaml")

# Read the YAML file and generate the LaTeX table
if os.path.exists(config_filename):
    with open(config_filename, 'r') as f:
        config_data = yaml.safe_load(f)

    output_tex_filename = config_filename.parent / "config_table.tex"
    generate_latex_table(config_data, output_tex_filename)
else:
    print(f"Error: {config_filename} not found.")