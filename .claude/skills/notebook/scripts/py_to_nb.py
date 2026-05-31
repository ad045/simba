#!/usr/bin/env python3
"""Convert .py with # %% markers → .ipynb, preserving original notebook metadata."""
import sys
import json
import re
import pathlib

CELL_RE = re.compile(r"^# %%([^\n]*)$", re.MULTILINE)


def strip_comment_prefix(text):
    lines = []
    for line in text.split("\n"):
        if line.startswith("# "):
            lines.append(line[2:])
        elif line == "#":
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines)


def py_to_nb(py_path, nb_path):
    text = pathlib.Path(py_path).read_text()

    # Preserve metadata from existing notebook if present
    metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.8.0"},
    }
    if pathlib.Path(nb_path).exists():
        try:
            orig = json.loads(pathlib.Path(nb_path).read_text())
            metadata = orig.get("metadata", metadata)
        except Exception:
            pass

    matches = list(CELL_RE.finditer(text))
    cells = []
    for i, m in enumerate(matches):
        tag = m.group(1).strip()
        start = m.end() + 1  # skip newline after the marker line
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].rstrip("\n")

        if "[markdown]" in tag:
            src = strip_comment_prefix(body)
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": src.splitlines(keepends=True),
                }
            )
        elif "[raw]" in tag:
            src = strip_comment_prefix(body)
            cells.append(
                {
                    "cell_type": "raw",
                    "metadata": {},
                    "source": src.splitlines(keepends=True),
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": body.splitlines(keepends=True),
                }
            )

    nb = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
    pathlib.Path(nb_path).write_text(json.dumps(nb, indent=1))
    print(f"Converted {py_path} -> {nb_path}  ({len(cells)} cells)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: py_to_nb.py <in.py> <out.ipynb>")
    py_to_nb(sys.argv[1], sys.argv[2])
