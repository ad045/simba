# CLAUDE.md — 14_4D_benchmarking

## What this repo is

Code and generated data for the manuscript *How Similar Are Two Brains? A
Comprehensive Benchmark of Brain Network Similarity Measures* (Dendorfer, Luppi,
Poli, Mousley, Astle, Fakhar). It benchmarks 16 network distance measures for
comparing GNM output against an empirical connectome, and carries 8 through to
the main analyses.

**Read `README.md` first** — it documents the full pipeline, the figure→script
map, and the known rough edges. This file only adds what an agent needs on top.

Manuscript repo: `~/Desktop/Benchmarking_` (LaTeX; has its own `CLAUDE.md` with
the pinned numerical results — check any number you touch against it).
Sibling repo: `../14_4D_property_space` (morphospace half).
Read-only backup of the pre-split state: `../14_4D_lab_code`.

## Environment

```bash
conda activate ma_thesis
```

Two local forks, both editable installs, both required:
`../GenerativeNetworkModels` (patched GNM library, commit `e3bc425`) and
`../Pyvizman` (plotting helpers).

**Run everything from the repository root** — paths anchor there.

## Where things live

- `experiments_config.py` — single source of truth: the 8 measures, their
  orientation (`IS_SIMILARITY` / `to_distance`), colours, names, grid geometry,
  filename conventions, paths. Change measure metadata here and nowhere else.
- `src/comparing_connectomes/` — `NetworkEvaluator` + ~20 metric subclasses. The
  spine of the benchmarking side.
- `pipeline_gnms_and_benchmarking/` — GNM generation (Stage 1) and the
  generated-vs-empirical comparison (Stage 2).
- `experiment_*/` — one folder per analysis; each runner has a docstring
  explaining the question it answers. Labels S2/S3/S4/S6/S6b/S6c match the
  manuscript's supplementary analyses.
- `visualization/` — the notebooks and scripts that draw the manuscript figures.
- `publication_data/` — the shareable data, built by `make_publication_data.py`.
- `archive/` — everything not part of this manuscript, kept for provenance.

## Hard-won facts

- **Five of the 16 measures are similarities, not distances** (`communicability_corr`,
  `jaccard`, `f1`, `network_mutual_information`, `dc_network_mutual_information`).
  They must be flipped before any `argmin` / "closest cell" logic. Both recovery
  experiments were rescored in Aug 2026 after this was found
  (`rescore_recovery_predictions.py`). Always route raw comparer output through
  `experiments_config.to_distance()`.
- **`data/` and `output/` are symlinks** into `../14_4D_lab_code` (~225 GB of run
  history). They are gitignored. Do not delete `../14_4D_lab_code`.
- The manuscript run is `output/gnm/hcp_schaefer_100_dataset/105_distance_metrics_mst_animal_0`.
- **The empirical connectomes cannot be shared.** Anything derived from them
  belongs in `publication_data/empirical_derived/`, which is gitignored.
  Preprocessing *code* is shared; preprocessing *output* is not.
- `src/ESNs/` is unused by this paper but cannot be removed without untangling a
  module-level import in `src/pipeline/gnm_and_esn_orchestrator.py`.
- `configs/config_gnm_run_hcp.yaml` is **not** the exact config of the published
  sweep — its ranges were edited afterwards. The real grid is in `experiments_config.py`.

## Style

- Dashes: never `--` or `---`; always `-`. Same rule as the manuscript.
- `graphify-out/` is gitignored and regenerable; run `graphify update .` after
  code changes if you use it.
