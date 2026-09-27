"""
Turn a figure notebook into a runnable script
=============================================

The manuscript figures were built in notebooks that hardcode the run name and a
few absolute paths into a sibling repo. This lifts the code cells out and
rewrites exactly those hardcoded bits so the script follows `experiments_config`
(and therefore `MORPHO_EXP`) instead. Nothing else about the notebook code is
changed - comment-only cells are dropped, everything else is copied verbatim in
cell order.

    python visualization/extract_notebook.py 9_between_and_withhin_variance.ipynb
    python visualization/extract_notebook.py --all

Writes `visualization/run_<stem>.py`. Re-running overwrites it, so edit the
notebook (or the rewrite rules here), never the generated file.
"""

import argparse
import json
import re
from pathlib import Path

VIZ = Path(__file__).resolve().parent

HEADER = '''"""
{title}

Extracted from {src} by extract_notebook.py - do not edit by hand.
The run comes from experiments_config (MORPHO_EXP); paths that pointed into the
connectome_distances repo now use this repo's own data.
"""
import matplotlib
matplotlib.use("Agg")
'''

CONFIG_BLOCK = '''import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from experiments_config import (MORPHO_DATASET, MORPHO_EXP, ROOT_DIR as _RD,
                                DIST_MATRIX_PATH, INDIVIDUALS_PATH, to_distance)

dataset_name = MORPHO_DATASET
experiment_name = MORPHO_EXP'''

LAB = r"/Users/adrian/Documents/01_projects/14_4D_lab"
CD = f"{LAB}/connectome_distances/data/preprocessed/lexis_data"

# (pattern, replacement) applied to the whole assembled source, in order.
RULES = [
    # absolute paths into the old run tree -> relative to the repository root
    # (scripts run with the paper/ folder as working directory)
    (re.compile(re.escape(f"{LAB}/14_4D_lab_code/")), ""),
    # the run: dataset_name + experiment_name assignment pair
    (re.compile(r'^dataset_name\s*=\s*"hcp_schaefer_100_dataset".*?\nexperiment_name\s*=\s*"105_distance_metrics_mst_animal_0".*?$',
                re.M | re.S),
     CONFIG_BLOCK),
    # base_path, in its three spellings
    (re.compile(re.escape(f'Path(f"{LAB}/14_4D_lab_code/output/gnm/{{dataset_name}}/{{experiment_name}}")')),
     '(_RD / "output" / "gnm" / dataset_name / experiment_name)'),
    (re.compile(re.escape(f'Path(f"{LAB}/14_4D_lab_code/output/gnm") / dataset_name / experiment_name')),
     '(_RD / "output" / "gnm" / dataset_name / experiment_name)'),
    (re.compile(re.escape(f'f"{LAB}/14_4D_lab_code/output/gnm/{{dataset_name}}/{{experiment_name}}"')),
     'str(_RD / "output" / "gnm" / dataset_name / experiment_name)'),
    # the stale sibling-repo data
    (re.compile(re.escape(f'"{CD}/02_distance_matrices/distance_matrix_100.npy"')),
     'DIST_MATRIX_PATH'),
    (re.compile(re.escape(f'"{CD}/01_connectomes/00_individual_connectomes_bin.npy"')),
     'INDIVIDUALS_PATH'),
    # Similarity measures must be flipped before a "lowest = best" selection.
    # Two forms occur: some notebooks already branch to `nlargest` for the
    # similarity columns and are left alone; this one sorts the raw column and
    # so picked the LEAST similar networks. Anchored on the preceding comment so
    # only the unguarded form matches, and the indent is carried over.
    (re.compile(r'( *)# Select top N networks \(lowest values = best\)\n'
                r'( *)df_sorted = df\.nsmallest\(top_n, metric_col\)'),
     r'\1# Select top N networks (lowest values = best); similarities flipped first\n'
     r'\2df = df.assign(_oriented=to_distance(df[metric_col].values, mode))\n'
     r'\2df_sorted = df.nsmallest(top_n, "_oriented")'),
    # The correlation-network cell leaves the axes auto-scaled, and the community
    # layout can push nodes past those limits - silently clipping most of the
    # network (it rendered 5 of 16 nodes after the rescore). Fit the view to pos.
    (re.compile(r"ax\.axis\('off'\)\nplt\.tight_layout\(\)\n"
                r"plt\.savefig\(save_folder\.parent / \"correlation_graph\.pdf\"\)"),
     "ax.axis('off')\n"
     "_xs = [q[0] for q in pos.values()]; _ys = [q[1] for q in pos.values()]\n"
     "_mx = max((max(_xs) - min(_xs)) * 0.15, 0.1); _my = max((max(_ys) - min(_ys)) * 0.15, 0.1)\n"
     "ax.set_xlim(min(_xs) - _mx, max(_xs) + _mx); ax.set_ylim(min(_ys) - _my, max(_ys) + _my)\n"
     "plt.tight_layout()\n"
     'plt.savefig(save_folder.parent / "correlation_graph.pdf")'),
    # Colours were assigned by position in the clustering order, which makes the
    # palette data-dependent: the density rescore reshuffled the clustering and
    # with it every measure's colour, so Figure 2 disagreed with the radar, the
    # eight-measure legend and the appendix figures - the same colour meaning a
    # different measure in different figures. Pin the eight selected measures to
    # the project palette and hand the rest the leftover colours.
    (re.compile(r"metric_colors = \{\}\n\nfor i, metric_name in enumerate\(all_dist_measures\):\n"
                r" +metric_colors\[metric_name\] = all_colors\[i\]"),
     "from experiments_config import METRIC_COLORS as _PINNED\n"
     "_used = {tuple(round(float(c), 3) for c in v[:3]) for v in _PINNED.values()}\n"
     "_spare = [c for c in all_colors\n"
     "          if tuple(round(float(x), 3) for x in c[:3]) not in _used]\n"
     "metric_colors = {}\n"
     "_n = 0\n"
     "for metric_name in all_dist_measures:\n"
     "    if metric_name in _PINNED:\n"
     "        metric_colors[metric_name] = tuple(_PINNED[metric_name])\n"
     "    else:\n"
     "        metric_colors[metric_name] = _spare[_n % len(_spare)]\n"
     "        _n += 1"),
    # Panel E of Figure 2 iterated the measure list reversed while panel A
    # iterated it forward, so the two 4x4 grids showed the same sixteen
    # landscapes in opposite orders and could not be read against each other.
    (re.compile(r"for idx, mode in enumerate\(all_dist_measures\[::-1\]\):"),
     "for idx, mode in enumerate(all_dist_measures):  # same order as panel A"),
    # The correlation network labelled only a hardcoded subset of nodes, and not
    # even the eight selected measures. Label every node. The edge threshold
    # (strongest 30% by magnitude) is deliberately kept as published.
    (re.compile(re.escape('label_shorthands = {\n    "energy": "Energy", \n    "portrait": "Portrait",\n    "spectral_distance_adjacency": "Spectral\\nAdj.",\n    "communicability_jsd": "Comm.\\nJSD",\n    "net_simile": "NetSimile", \n    "netrd_non_backtracking_spectral": "Spectral\\nNon-BT", \n    "resistance": "Res.", \n    "delta_con": "DeltaCon", \n}')), (lambda _m, _r='label_shorthands = {\n    "energy": "Energy",\n    "portrait": "Portrait",\n    "spectral_distance_adjacency": "Spectral\\nAdj.",\n    "spectral_distance_norm_laplacian": "Spectral\\nNormLap",\n    "communicability_jsd": "Comm.\\nJSD",\n    "communicability_corr": "Comm.\\nCorr.",\n    "net_simile": "NetSimile",\n    "netrd_non_backtracking_spectral": "Spectral\\nNon-BT",\n    "resistance": "Res.",\n    "delta_con": "DeltaCon",\n    "frobenius": "Frobenius",\n    "hamming": "Hamming",\n    "jaccard": "Jaccard",\n    "f1": "F1",\n    "network_mutual_information": "NMI",\n    "dc_network_mutual_information": "DC-NMI",\n}': _r)),
    (re.compile(r"labels = \{node: label_shorthands\[node\]\n *for node in G\.nodes\(\) if node in selected_methods\}"),
     "labels = {node: label_shorthands.get(node, node) for node in G.nodes()}"),
    # Panel E's similarity colourbar labelled the dark end "Low" while panel A
    # labelled the same colormap's dark end "High". Dark means closer to the
    # consensus in both, so E's labels were the reversed pair.
    (re.compile(r'(cbar\.set_label\("Similarity"\)[^\n]*\n[^\n]*\n'
                r'cbar\.ax\.set_xticks\(\[0, 10\], labels=\[)'
                r"'Low', 'High'" r'(\])'),
     r'\1"High", "Low"\2'),
    # Lay the correlation network out from the thresholded edges themselves, so
    # the clusters are spatial. The community-centre layout placed nodes by
    # cluster membership and an aggressive separation pass then pushed them into
    # a ring, which hid the very structure the panel is meant to show. Instead:
    # Kamada-Kawai on the connected core, the measures with no edge above the
    # threshold on a ring outside it, and a gentle separation pass that removes
    # label overlap without collapsing the layout.
    (re.compile(r"pos = nx\.kamada_kawai_layout\(G,\n[^\n]*\n[^\n]*\n"
                r"(?:[^\n]*\n)?pos = separate_overlapping_nodes\(pos, node_radius=0\.5\)[^\n]*"),
     "_core = [n for n in G.nodes() if G.degree(n) > 0]\n"
     "_iso = [n for n in G.nodes() if G.degree(n) == 0]\n"
     "if _core:\n"
     "    pos = nx.kamada_kawai_layout(G.subgraph(_core), scale=2.6)\n"
     "    _c = np.mean(list(pos.values()), axis=0)\n"
     "    _rad = max(np.linalg.norm(np.array(_p) - _c) for _p in pos.values())\n"
     "else:\n"
     "    pos, _c, _rad = {}, np.zeros(2), 1.0\n"
     "for _n, _a in zip(_iso, np.linspace(0, 2 * np.pi, max(len(_iso), 1), endpoint=False) + 0.5):\n"
     "    pos[_n] = _c + np.array([np.cos(_a), np.sin(_a)]) * (_rad + 1.35)\n"
     "pos = separate_overlapping_nodes(pos, node_radius=0.78, max_iter=600)"),
    # Orient at load: the correlation matrix, its clustering and the correlation
    # network were all computed on raw output, so the five similarity-valued
    # measures entered with the opposite sign to the distance-valued ones and a
    # strong negative entry meant close agreement. Every measure is a distance
    # from here on, so a positive correlation always means agreement.
    (re.compile(r"df\['metric_value'\] = df\[metric_cols\[0\]\]"),
     "df['metric_value'] = to_distance(df[metric_cols[0]].values, method)"),
    # Resistance is a distance, not a similarity: taking its largest values
    # selected the LEAST similar networks. Only communicability correlation is
    # similarity-valued among the measures these branches handle.
    (re.compile(r'if metric_col in \["CommCorr_subject_0", "Resistance_subject_0"\]:'),
     'if metric_col in ["CommCorr_subject_0"]:'),
    # correlation_graph.pdf was written to the run directory, which every one of
    # these notebooks shares - so the KS-energy-contributor version overwrote the
    # 16-measure one and panel C of Figure 2 silently became the wrong graph.
    # Write it beside the other panels of its own figure instead, which makes the
    # collision impossible rather than something the driver has to work around.
    (re.compile(r'savefig\(save_folder\.parent / "correlation_graph\.pdf"\)'),
     'savefig(output_path / "correlation_graph.pdf")'),
    (re.compile(r'print\(save_folder\.parent / "correlation_graph\.pdf"\)'),
     'print(output_path / "correlation_graph.pdf")'),
]


# Cells of 8_7_manuscript_16_measures.ipynb that only produce outputs Figure 2
# does not use (grayscale landscapes, colourbar variants, the dendrogram and
# heatmap-only variants, the variance grid, the violin timing panel, ...).
# Skipping them yields `run_fig2_panels.py`, which reproduces the six panels and
# the legend of Figure 2 byte-for-byte in about half the runtime. Cell 18 is NOT
# in here: it saves a figure of its own but also builds `timing_data`, which
# panel D needs.
FIG2_SKIP = {14, 29, 30, 35, 41, 42, 43, 53, 54, 60, 64}

FIG2_TITLE = """Figure 2 panels only - the sixteen landscapes, the correlation
matrix, the correlation network, the timing boxplots and their colour strip, the
100-best-fitting landscapes, and the measure legend.

Generated by `extract_notebook.py --figure2`; see FIG2_SKIP there for what is
left out and why. Verified to reproduce the panels of the full extraction
exactly. The composite is assembled from them by visualization/build_composites.py."""


def cell_source(cell: dict) -> str:
    """Join a cell's source, repairing the one-character-per-line corruption.

    Some cells in 8_7_6_manuscript_16_measures_hcp_apdx.ipynb are stored as
    ['d\\n', 'e\\n', 'f\\n', ...] - every character on its own line. Joining that
    directly yields one character per line and does not parse. Taking the first
    character of each entry recovers the text (a real newline is stored as
    '\\n\\n', so it survives).
    """
    src = cell["source"]
    if isinstance(src, list) and src and all(len(x) <= 2 for x in src):
        return "".join(x[0] for x in src)
    return "".join(src)


def extract(nb_path: Path, skip: set | None = None, out_name: str | None = None,
            title: str | None = None) -> Path:
    nb = json.loads(nb_path.read_text())
    skip = skip or set()
    blocks = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code" or i in skip:
            continue
        src = cell_source(cell)
        if not [l for l in src.split("\n") if l.strip() and not l.strip().startswith("#")]:
            continue
        blocks.append(f"# {'=' * 68}\n# cell {i}\n# {'=' * 68}\n{src}\n")

    body = "\n".join(blocks)
    for pattern, repl in RULES:
        body = pattern.sub(repl, body)

    leftover = re.findall(r'"[^"]*connectome_distances[^"]*"', body)
    if leftover:
        print(f"  ! {nb_path.name}: unrewritten sibling-repo path(s): {leftover}")
    if "105_distance_metrics_mst_animal_0" in body:
        print(f"  ! {nb_path.name}: still contains the old run name")

    out = VIZ / (out_name or f"run_{nb_path.stem}.py")
    out.write_text(HEADER.format(title=title or nb_path.stem, src=nb_path.name) + "\n" + body)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--figure2", action="store_true",
                    help="emit run_fig2_panels.py: only the panels Figure 2 uses")
    args = ap.parse_args()

    if args.figure2:
        nb = VIZ / "8_7_manuscript_16_measures.ipynb"
        out = extract(nb, skip=FIG2_SKIP, out_name="run_fig2_panels.py", title=FIG2_TITLE)
        print(f"{nb.name} -> {out.name}  (skipped cells {sorted(FIG2_SKIP)})")
        return

    if args.all:
        targets = [VIZ / n for n in [
            "8_7_manuscript_16_measures.ipynb",
            "9_between_and_withhin_variance.ipynb",
            "8_9_manuscript_selected_8_measures.ipynb",
            "8_9_2_minima_from_average_landscape.ipynb",
            "8_7_2_ks_energy_contributors.ipynb",
            "8_7_6_manuscript_16_measures_hcp_apdx.ipynb",
            "21_2_fig_1_six_example_gen_connectomes_hcp_data.ipynb",
        ]]
    else:
        targets = [VIZ / args.notebook]

    for nb in targets:
        print(f"{nb.name} -> {extract(nb).name}")


if __name__ == "__main__":
    main()
