---
name: notebook
description: Convert Jupyter notebooks to Python scripts for easy editing with Read/Edit/Write tools, then convert back to .ipynb. Use when asked to edit, fix, update, or work on a .ipynb notebook, or when user says "edit notebook", "work on notebook", "fix notebook", or invokes /notebook.
---

# Notebook

Round-trip a Jupyter notebook through a clean Python script so you can use Read/Edit/Write instead of NotebookEdit.

## Workflow

**1. Convert notebook → Python**
```bash
python .claude/skills/notebook/scripts/nb_to_py.py <notebook.ipynb> <notebook.py>
```
Each cell becomes a `# %%` block. Markdown cells use `# %% [markdown]` with lines prefixed `# `.

**2. Edit the Python file**
Use Read / Edit / Write on the `.py` file as normal. Cell boundaries are the `# %%` lines.

**3. Convert Python → notebook**
```bash
python .claude/skills/notebook/scripts/py_to_nb.py <notebook.py> <notebook.ipynb>
```
If the original `.ipynb` exists, its kernelspec metadata is preserved. Cell outputs are cleared — user must re-run cells.

**4. Clean up**
```bash
rm <notebook.py>
```

## Naming convention

Use the same stem: `00_analysis.ipynb` ↔ `00_analysis.py`.

## Notes

- Scripts use only Python stdlib — no extra dependencies
- Always confirm the round-trip succeeded before deleting the `.py`
- If the task is only reading/understanding a notebook, skip the round-trip and just run `nb_to_py.py` to get a readable `.py` without converting back
