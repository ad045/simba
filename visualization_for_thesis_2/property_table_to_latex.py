"""
excel_to_latex.py
-----------------
Converts the 'for_latex' sheet of an Excel file into a set of LaTeX tables,
one per cluster.  Edit the two configuration blocks below to control which
Excel file to use, which columns to include, and how they are labelled.
"""

import pandas as pd
import textwrap, re, sys
from pathlib import Path

from config import MERGED_PROPERTIES_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these two blocks
# ─────────────────────────────────────────────────────────────────────────────

EXCEL_FILE  = "/Users/adrian/Desktop/overview_metrics_all.xlsx"
SHEET_NAME  = "for_latex"        # change if the sheet name differs
OUTPUT_FILE = "/Users/adrian/Documents/01_projects/14_4D_lab/latex_thesis/tables/latex_tables.tex"  # set to None to print to stdout

# # Select the columns you want in the output tables.
# # Each entry is  "excel_column_name": "LaTeX header label"
# # Comment out any line to exclude that column.
# SELECTED_COLUMNS = {
#     "metric_name":                    r"\textbf{Metric}",
#     "reason":                         r"\textbf{Reason}",
#     # "category":                     r"\textbf{Category}",
#     # "code_location":                r"\textbf{Code location}",
#     "how_computed_in_code":           r"\textbf{How computed}",
#     # "statistical_features_extracted": r"\textbf{Features extracted}",
#     "scientific_source":              r"\textbf{Scientific source}",
#     # "recommended features":         r"\textbf{Recommended features}",
#     "library_used":                   r"\textbf{Library}",
#     # "library_call_correct":         r"\textbf{Call correct?}",
#     "why_useful":                     r"\textbf{Why useful}",
#     "latex_formula":                  r"\textbf{Formula}",
#     # "implementation_correct":       r"\textbf{Implementation correct?}",
#     # "redundant_features":           r"\textbf{Redundant features}",
#     # "duplicates_with":              r"\textbf{Duplicates with}",
#     # "notes":                        r"\textbf{Notes}",
#     # "pca_include":                  r"\textbf{PCA include?}",
#     # "pca_include_reason":           r"\textbf{PCA include reason}",
#     # "recommended_features":         r"\textbf{Recommended features (2)}",
# }

# # ─────────────────────────────────────────────────────────────────────────────
# # CLUSTER DEFINITIONS  ← these come from your research code
# # ─────────────────────────────────────────────────────────────────────────────

# CLUSTERS = {
#     "Wiring Economy": [
#         "wiring_cost",
#         "proportion_long_range_connections_0.3956",
#     ],
#     "Global Integration": [
#         "global_efficiency",
#         "diffusion_efficiency",
#         "propagation_efficiency",
#     ],
#     "Local Segregation": [
#         "modularity",
#         "local_efficiency_stats_mean",
#         "local_efficiency_stats_std",
#         "omega",
#         "directed_simplices_count",
#         "directed_simplices_max_size",
#         "ollivier_ricci_curvature_orc_mean",
#         "ollivier_ricci_curvature_orc_min",
#         "ollivier_ricci_curvature_orc_skewness",
#     ],
#     "Communication Efficiency": [
#         # Hub routing
#         "degree_gini",
#         "betweenness_centrality_stats_gini",
#         "betweenness_centrality_stats_mean",
#         "degree_assortativity",
#         "rich_club_coefficient_rc_k_at_max",
#         "participation_coefficient_pc_std",
#         "nct_control_std",
#         # Synchronization speed
#         "spectral_gap",
#         "synchronizability_eigenratio_eigenratio",
#         "synchronizability_eigenratio_eigenratio_lambda_2", 
#         #  "algebraic_connectivity_fiedler_value",
#         "algebraic_connectivity_laplacian_spectral_gap",
#         "community_synchronization_vulnerability_value",
#     ],
#     "Computational Capacity": [
#         # Reservoir properties
#         "ipc_ipc_deg1_mean",
#         "ipc_ipc_deg2_mean",
#         "kernel_rank_max",
#         "effective_dimensionality",
#         "departure_from_normality_schur",
#         # Spectral substrate
#         "spectral_radius",
#         "structural_complexity",
#         # Critical state repertoire
#         "repertoire_sweep_weighted_by_distances_T_critical",
#         "repertoire_sweep_weighted_by_distances_size_critical",
#         "repertoire_sweep_weighted_by_distances_diversity_critical",
#     ],
#     "Robustness": [
#         "targeted_attack_robustness_rob_targeted_auc",
#         "targeted_attack_robustness_rob_random_auc",
#     ],
# }

# # ─────────────────────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# _SPECIAL = re.compile(r'(?<!\\)([&%$#_{}])')


# def _escape(text: str) -> str:
#     """Escape LaTeX special characters (leave already-escaped sequences alone)."""
#     if not isinstance(text, str):
#         text = "" if pd.isna(text) else str(text)
#     text = _SPECIAL.sub(r'\\\1', text)
#     return text


# def _cell(value) -> str:
#     if pd.isna(value):
#         return "---"
#     return str(value).strip()


# def _col_spec(n: int) -> str:
#     """Simple column spec: first column left-aligned, rest paragraph-width."""
#     specs = ["l"] + ["p{3.5cm}"] * (n - 1)
#     return " ".join(specs)


# def build_table(cluster_name: str, rows: pd.DataFrame, columns: dict) -> str:
#     """Return a full LaTeX table environment for one cluster."""
#     col_keys = list(columns.keys())
#     headers  = list(columns.values())
#     n        = len(col_keys)

#     lines = []
#     lines.append(r"\begin{table}[htbp]")
#     lines.append(r"  \centering")
#     lines.append(r"  \footnotesize")
#     lines.append(f"  \\caption{{\\textbf{{{_escape(cluster_name)}}}}}")
#     lines.append(f"  \\label{{tab:{cluster_name.lower().replace(' ', '_')}}}")
#     lines.append(f"  \\begin{{tabular}}{{{_col_spec(n)}}}")
#     lines.append(r"    \toprule")

#     # Header row
#     lines.append("    " + " & ".join(headers) + r" \\")
#     lines.append(r"    \midrule")

#     # Data rows
#     for _, row in rows.iterrows():
#         cells = []
#         for k in col_keys:
#             val = _cell(row[k]) if k in row.index else "---"
#             cells.append(_escape(val))
#         lines.append("    " + " & ".join(cells) + r" \\")
#         lines.append(r"    \midrule")

#     # Remove the last \midrule and replace with \bottomrule
#     lines[-1] = r"    \bottomrule"

#     lines.append(r"  \end{tabular}")
#     lines.append(r"\end{table}")
#     lines.append("")
#     return "\n".join(lines)


# def preamble() -> str:
#     return textwrap.dedent(r"""
#         % Auto-generated by excel_to_latex.py
#         % Required packages (add to your document preamble):
#         %   \usepackage{booktabs}
#         %   \usepackage{array}
#         %   \usepackage{longtable}  % optional, for very long tables
#     """).lstrip()


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     excel_path = Path(EXCEL_FILE)
#     if not excel_path.exists():
#         sys.exit(f"ERROR: '{EXCEL_FILE}' not found. "
#                  "Set EXCEL_FILE at the top of this script.")

#     df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)

#     # Fix mojibake: repair UTF-8 text that was mis-decoded as latin-1/cp1252
#     def fix_encoding(val):
#         if not isinstance(val, str):
#             return val
#         try:
#             return val.encode("cp1252").decode("utf-8")
#         except (UnicodeEncodeError, UnicodeDecodeError):
#             return val

#     df = df.apply(lambda col: col.map(fix_encoding))

#     # Validate selected columns
#     missing_cols = [c for c in SELECTED_COLUMNS if c not in df.columns]
#     if missing_cols:
#         print(f"WARNING: These selected columns are not in the sheet and will "
#               f"be skipped: {missing_cols}", file=sys.stderr)
#     valid_cols = {k: v for k, v in SELECTED_COLUMNS.items() if k in df.columns}

#     if "metric_name" not in df.columns:
#         sys.exit("ERROR: 'metric_name' column not found in sheet.")

#     # Prefix-based lookup: find the sheet row whose metric_name is a prefix of
#     # the cluster metric name, pick the longest match (most specific), and
#     # display the full cluster metric name in the output.
#     def find_row(metric: str) -> pd.Series | None:
#         candidates = [name for name in df["metric_name"] if metric.startswith(name)]
#         if not candidates:
#             return None
#         best = max(candidates, key=len)
#         row = df[df["metric_name"] == best].iloc[0].copy()
#         row["metric_name"] = metric  # display the full name in the table
#         return row

#     output_parts = [preamble()]
#     total_found   = 0
#     total_missing = 0

#     for cluster_name, metric_list in CLUSTERS.items():
#         found_rows      = []
#         missing_metrics = []

#         for m in metric_list:
#             row = find_row(m)
#             if row is not None:
#                 found_rows.append(row)
#             else:
#                 missing_metrics.append(m)

#         if missing_metrics:
#             print(f"[{cluster_name}] Metrics not found in sheet (skipped): "
#                   f"{missing_metrics}", file=sys.stderr)

#         total_found   += len(found_rows)
#         total_missing += len(missing_metrics)

#         if not found_rows:
#             print(f"[{cluster_name}] No metrics found – table skipped.",
#                   file=sys.stderr)
#             continue

#         subset = pd.DataFrame(found_rows).reset_index(drop=True)
#         output_parts.append(build_table(cluster_name, subset, valid_cols))

#     latex = "\n".join(output_parts)

#     if OUTPUT_FILE:
#         Path(OUTPUT_FILE).write_text(latex, encoding="utf-8")
#         print(f"Written {len(CLUSTERS)} table(s) to '{OUTPUT_FILE}'")
#         print(f"  Metrics matched: {total_found} | Not found: {total_missing}")
#     else:
#         print(latex)


# if __name__ == "__main__":
#     main()
    
    
"""
excel_to_latex.py
-----------------
Converts the 'for_latex' sheet of an Excel file into a set of LaTeX tables,
one per cluster.  Edit the two configuration blocks below to control which
Excel file to use, which columns to include, and how they are labelled.
"""

import pandas as pd
import textwrap, re, sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these two blocks
# ─────────────────────────────────────────────────────────────────────────────

# EXCEL_FILE  = "overview_metrics_all.xlsx"
# SHEET_NAME  = "for_latex"        # change if the sheet name differs
# OUTPUT_FILE = "latex_tables.tex" # set to None to print to stdout

# Select the columns you want in the output tables.
# Each entry is  "excel_column_name": "LaTeX header label"
# Comment out any line to exclude that column.
SELECTED_COLUMNS = {
    "metric_name":                    r"\textbf{Metric}",
    # "reason":                       r"\textbf{Reason}", # my column
    # "category":                     r"\textbf{Category}",
    # "code_location":                r"\textbf{Code location}",
    # "how_computed_in_code":         r"\textbf{How computed}",
    # "statistical_features_extracted": r"\textbf{Features extracted}",
    "scientific_source":              r"\textbf{Scientific source}",
    # "recommended features":           r"\textbf{Recommended features}",
    # "library_used":                 r"\textbf{Library}",
    # "library_call_correct":         r"\textbf{Call correct?}",
    "why_useful":                     r"\textbf{Why useful}",
    "latex_formula":                  r"\textbf{Formula}",
    # "implementation_correct":       r"\textbf{Implementation correct?}",
    # "redundant_features":           r"\textbf{Redundant features}",
    # "duplicates_with":              r"\textbf{Duplicates with}",
    # "notes":                        r"\textbf{Notes}",
    # "pca_include":                    r"\textbf{PCA include?}",
    # "pca_include_reason":           r"\textbf{PCA include reason}",
    # "recommended_features":         r"\textbf{Recommended features (2)}",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER DEFINITIONS  ← these come from your research code
# ─────────────────────────────────────────────────────────────────────────────

CLUSTERS = {
    "Wiring Economy": [
        "wiring_cost",
        "proportion_long_range_connections_0.3956",
    ],
    "Global Integration": [
        "global_efficiency",
        "diffusion_efficiency",
        "propagation_efficiency",
    ],
    "Local Segregation": [
        "modularity",
        "local_efficiency_stats_mean",
        "local_efficiency_stats_std",
        "omega",
        "directed_simplices_count",
        "directed_simplices_max_size",
        "ollivier_ricci_curvature_orc_mean",
        "ollivier_ricci_curvature_orc_min",
        "ollivier_ricci_curvature_orc_skewness",
    ],
    "Communication Efficiency": [
        # Hub routing
        "degree_gini",
        "betweenness_centrality_stats_gini",
        "betweenness_centrality_stats_mean",
        "degree_assortativity",
        "rich_club_coefficient_rc_k_at_max",
        "participation_coefficient_pc_std",
        "nct_control_std",
        # Synchronization speed
        "spectral_gap",
        "synchronizability_eigenratio_eigenratio",
        "algebraic_connectivity_fiedler_value",
        "algebraic_connectivity_laplacian_spectral_gap",
        "community_synchronization_vulnerability_value",
    ],
    "Computational Capacity": [
        # Reservoir properties
        "ipc_ipc_deg1_mean",
        "ipc_ipc_deg2_mean",
        "kernel_rank_max",
        "effective_dimensionality",
        "departure_from_normality_schur",
        # Spectral substrate
        "spectral_radius",
        "structural_complexity",
        # Critical state repertoire
        "repertoire_sweep_weighted_by_distances_T_critical",
        "repertoire_sweep_weighted_by_distances_size_critical",
        "repertoire_sweep_weighted_by_distances_diversity_critical",
    ],
    "Robustness": [
        "targeted_attack_robustness_rob_targeted_auc",
        "targeted_attack_robustness_rob_random_auc",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ESCAPING STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
# RAW_LATEX_COLUMNS   → passed through with _pass_through (encoding + unicode fix only)
#                       Use for columns whose content is 100% valid LaTeX already.
# MIXED_LATEX_COLUMNS → escaped with _escape_mixed (protects $...$ spans, escapes prose)
#                       Use for columns that contain a mix of plain text and $...$ math.
# Everything else     → _escape_plain (full plain-text escaping, no math awareness)

RAW_LATEX_COLUMNS = {
}

MIXED_LATEX_COLUMNS = {
    "latex_formula",          # seemingly pure LaTeX formulas
    "why_useful",             # prose + occasional $...$ inline math
    "how_computed_in_code",   # code descriptions + some $...$
    "scientific_source",      # author & year — & must be escaped, but has $...$
    "library_used",
    "statistical_features_extracted",
    "redundant_features",
    "duplicates_with",
    "notes",
    "pca_include_reason",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Unicode → LaTeX replacements applied everywhere (including inside math spans)
_UNICODE_MAP = {
    "≈": r"\approx{}",
    "≠": r"\neq{}",
    "≤": r"\leq{}",
    "≥": r"\geq{}",
    "→": r"\to{}",
    "←": r"\leftarrow{}",
    "∞": r"\infty{}",
    "∑": r"\sum{}",
    "∏": r"\prod{}",
    "∈": r"\in{}",
    "∉": r"\notin{}",
    "⊂": r"\subset{}",
    "⊃": r"\supset{}",
    "∩": r"\cap{}",
    "∪": r"\cup{}",
    "∀": r"\forall{}",
    "∃": r"\exists{}",
    "·": r"\cdot{}",
    "×": r"\times{}",
    "÷": r"\div{}",
    "±": r"\pm{}",
    "√": r"\sqrt{}",
    "∂": r"\partial{}",
    "∇": r"\nabla{}",
    "α": r"\alpha{}",
    "β": r"\beta{}",
    "γ": r"\gamma{}",
    "δ": r"\delta{}",
    "λ": r"\lambda{}",
    "μ": r"\mu{}",
    "σ": r"\sigma{}",
    "τ": r"\tau{}",
    "φ": r"\phi{}",
    "ψ": r"\psi{}",
    "ω": r"\omega{}",
    "—": r"---",
    "–": r"--",
}


def _fix_encoding(val: str) -> str:
    """Repair mojibake produced when UTF-8 bytes were decoded as Mac Roman."""
    if not isinstance(val, str):
        return val
    try:
        return val.encode("mac_roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return val


def _fix_unicode(text: str) -> str:
    """Replace unicode math/punctuation with safe LaTeX equivalents."""
    for char, replacement in _UNICODE_MAP.items():
        text = text.replace(char, replacement)
    return text


# Splits a string into alternating (outside-math, inside-math) segments.
# Handles $...$ but not $$...$$. Escaped \$ is ignored.
_MATH_SPLIT = re.compile(r'(?<!\\)\$.*?(?<!\\)\$', re.DOTALL)


def _escape_mixed(text: str) -> str:
    """
    Smart escaper for columns that contain a MIX of plain prose and $...$
    LaTeX math (e.g. why_useful, how_computed_in_code, scientific_source).

    Rules applied to text OUTSIDE math spans:
      - Escape bare & (author 'A & B' must become 'A \\& B')
      - Escape bare % → \\%
      - Escape bare # → \\#
      - Escape bare _ → \\_  (plain text underscores like global_efficiency)
      - Replace unicode characters

    Text INSIDE $...$ is left entirely untouched (it is already valid LaTeX).
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = _fix_encoding(text)
    text = _fix_unicode(text)

    result = []
    last = 0
    for m in _MATH_SPLIT.finditer(text):
        # Process the plain-text segment before this math span
        plain = text[last:m.start()]
        plain = re.sub(r'(?<!\\)&',  r'\\&',  plain)
        plain = re.sub(r'(?<!\\)%',  r'\\%',  plain)
        plain = re.sub(r'(?<!\\)#',  r'\\#',  plain)
        plain = re.sub(r'(?<!\\)_',  r'\\_',  plain)
        result.append(plain)
        # Keep the math span verbatim
        result.append(m.group())
        last = m.end()
    # Tail after last math span
    tail = text[last:]
    tail = re.sub(r'(?<!\\)&',  r'\\&',  tail)
    tail = re.sub(r'(?<!\\)%',  r'\\%',  tail)
    tail = re.sub(r'(?<!\\)#',  r'\\#',  tail)
    tail = re.sub(r'(?<!\\)_',  r'\\_',  tail)
    result.append(tail)
    return "".join(result)


def _escape_plain(text: str) -> str:
    """For fully plain-text columns (metric_name, reason, category, …)."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = _fix_encoding(text)
    text = _fix_unicode(text)
    text = re.sub(r'(?<!\\)&',  r'\\&',  text)
    text = re.sub(r'(?<!\\)%',  r'\\%',  text)
    text = re.sub(r'(?<!\\)#',  r'\\#',  text)
    text = re.sub(r'(?<!\\)_',  r'\\_',  text)
    return text


def _pass_through(text: str) -> str:
    """For latex_formula — already valid LaTeX, only fix encoding + unicode."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return _fix_unicode(_fix_encoding(text))


def _cell(value) -> str:
    if pd.isna(value):
        return "---"
    return str(value).strip()


# def _col_spec(n: int) -> str:
#     """Simple column spec: first column left-aligned, rest paragraph-width."""
#     specs = ["l"] + ["p{3.5cm}"] * (n - 1)
#     return " ".join(specs)
def _col_spec(n: int) -> str:
    specs = ["p{2.5cm}"] + ["p{3.5cm}"] * (n - 1)
    return " ".join(specs)

def build_table(cluster_name: str, rows: pd.DataFrame, columns: dict) -> str:
    """Return a full LaTeX table environment for one cluster."""
    col_keys = list(columns.keys())
    headers  = list(columns.values())
    n        = len(col_keys)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \footnotesize")
    lines.append(f"  \\caption{{\\textbf{{{_escape_plain(cluster_name)}}}}}")
    lines.append(f"  \\label{{tab:{cluster_name.lower().replace(' ', '_')}}}")
    lines.append(f"  \\begin{{tabular}}{{{_col_spec(n)}}}")
    lines.append(r"    \toprule")

    # Header row
    lines.append("    " + " & ".join(headers) + r" \\")
    lines.append(r"    \midrule")

    # Data rows
    for _, row in rows.iterrows():
        cells = []
        for k in col_keys:
            val = _cell(row[k]) if k in row.index else "---"
            if k in RAW_LATEX_COLUMNS:
                cells.append(_pass_through(val))
            elif k in MIXED_LATEX_COLUMNS:
                cells.append(_escape_mixed(val))
            else:
                cells.append(_escape_plain(val))
        lines.append("    " + " & ".join(cells) + r" \\")
        lines.append(r"    \midrule")

    # Remove the last \midrule and replace with \bottomrule
    lines[-1] = r"    \bottomrule"

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def preamble() -> str:
    return textwrap.dedent(r"""
        % Auto-generated by excel_to_latex.py
        % Required packages (add to your document preamble):
        %   \usepackage{booktabs}
        %   \usepackage{array}
        %   \usepackage{longtable}  % optional, for very long tables
    """).lstrip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    excel_path = Path(EXCEL_FILE)
    if not excel_path.exists():
        sys.exit(f"ERROR: '{EXCEL_FILE}' not found. "
                 "Set EXCEL_FILE at the top of this script.")

    df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)

    # Validate selected columns
    missing = [c for c in SELECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"WARNING: These selected columns are not in the sheet and will "
              f"be skipped: {missing}", file=sys.stderr)
    valid_cols = {k: v for k, v in SELECTED_COLUMNS.items() if k in df.columns}

    if "metric_name" not in df.columns:
        sys.exit("ERROR: 'metric_name' column not found in sheet.")

    sheet_names = list(df["metric_name"])

    def common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    # MIN_MATCH only applies to fuzzy (non-prefix) matches.
    # If the sheet name is a true prefix of the cluster metric, always match.
    MIN_MATCH = 6

    def find_row(metric: str):
        """Return the sheet row whose name shares the longest common prefix
        with `metric`. A sheet name that is a full prefix of `metric` is
        always accepted; otherwise MIN_MATCH characters are required."""
        best_name, best_len = None, 0
        for name in sheet_names:
            n = common_prefix_len(metric, name)
            if n > best_len:
                best_len, best_name = n, name
        if best_name is None:
            return None
        # Accept if the sheet name is a full prefix of the cluster metric,
        # or if the common prefix is long enough.
        is_full_prefix = metric.startswith(best_name)
        if not is_full_prefix and best_len < MIN_MATCH:
            return None
        row = df[df["metric_name"] == best_name].iloc[0].copy()
        # row["metric_name"] = metric
        row["metric_name"] = MERGED_PROPERTIES_NAMES.get(metric, metric)
        return row

    output_parts = [preamble()]
    total_found   = 0
    total_missing = 0

    for cluster_name, metric_list in CLUSTERS.items():
        found_rows, missing_metrics = [], []
        for m in metric_list:
            row = find_row(m)
            if row is not None:
                found_rows.append(row)
            else:
                missing_metrics.append(m)

        if missing_metrics:
            print(f"[{cluster_name}] Metrics not found (skipped): "
                  f"{missing_metrics}", file=sys.stderr)

        total_found   += len(found_rows)
        total_missing += len(missing_metrics)

        if not found_rows:
            print(f"[{cluster_name}] No metrics found - table skipped.",
                  file=sys.stderr)
            continue

        subset = pd.DataFrame(found_rows).reset_index(drop=True)
        output_parts.append(build_table(cluster_name, subset, valid_cols))

    latex = "\n".join(output_parts)

    if OUTPUT_FILE:
        Path(OUTPUT_FILE).write_text(latex, encoding="utf-8")
        print(f"Written {len(CLUSTERS)} table(s) to '{OUTPUT_FILE}'")
        print(f"  Metrics matched: {total_found} | Not found: {total_missing}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
    
