#!/usr/bin/env python3
"""Convert .ipynb → .py with # %% cell markers for easy editing."""
import sys
import json
import pathlib


def nb_to_py(nb_path, py_path):
    nb = json.loads(pathlib.Path(nb_path).read_text())
    parts = []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        cell_type = cell["cell_type"]

        if cell_type == "markdown":
            header = "# %% [markdown]"
            body = "\n".join("# " + line if line else "#" for line in src.split("\n"))
        elif cell_type == "raw":
            header = "# %% [raw]"
            body = "\n".join("# " + line if line else "#" for line in src.split("\n"))
        else:
            header = "# %%"
            body = src

        parts.append(header + "\n" + body)

    pathlib.Path(py_path).write_text("\n\n".join(parts) + "\n")
    print(f"Converted {nb_path} -> {py_path}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: nb_to_py.py <in.ipynb> <out.py>")
    nb_to_py(sys.argv[1], sys.argv[2])
