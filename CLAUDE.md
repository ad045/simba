# CLAUDE.md — 14_4D_benchmarking

## What this repo is

The **benchmarking** half of the old `14_4D_lab_code` connectome pipeline (split out
2026-06-11). Focus: **generate brain networks (GNMs) and benchmark generated vs.
empirical connectomes** — including the reservoir-computing / memory-capacity angle
(*"Memory Capacity as a potential driver of brain-network topology"*).

Sibling repo: `../14_4D_property_space` (the morphospace / feature-property-space half).
Read-only backup of the pre-split state: `../14_4D_lab_code`.

## Environment

```bash
conda activate ma_thesis          # primary env for the pipeline
```

The `A_benchmarking_plots/` notebooks came from the `connectome_distances` project and
expect `conda activate connectome_distances` instead.

The pipeline depends on a **custom-patched GNM library** (Adrian's fork with local
changes) — a stock `gnm` install may not match. Scientific stack: `nilearn`, `nibabel`,
`networkx`, `torch`, `scipy`, `numpy`.

## Layout

- `src/comparing_connectomes/` — `NetworkEvaluator` base class + ~20 metric evaluators
  (DeltaCon, portrait divergence, communicability, graph kernels, mutual information, …).
  This is the core of the benchmarking side.
- `src/GNMs/` — generative network model generation.
- `src/pipeline/` — GNM + ESN orchestration (`gnm_network_generator`, `gnm_and_esn_orchestrator`).
- `src/ESNs/` — echo-state networks / reservoir computing / memory capacity.
- `src/preprocessing/` — connectome preprocessing pipelines (Shafiei, Suárez/MaMI, …).
- `src/analysis/`, `src/utils/`, `src/config/` — **shared, duplicated** code (see below).
- `A_benchmarking_plots/` — analysis notebooks copied from `connectome_distances`.
- `autoresearch/` — GNM-alternatives autoresearch + `autoresearch_karpathy` (nanoGPT).

### Recommended entry points
- `run_experiment.py` / `run_experiment_lexis_data.py` — generate GNMs.
- `run_connectome_comparisons.py` (+ `_null_model`, `_routing_etc`) — compare generated vs empirical.
- `run_connectome_targeted_removal.py`, `find_best_energy_row.py`.

## Split caveats (important)

- **`src/analysis`, `src/utils`, `src/config` are duplicated** with the sibling repo.
  Edits here do **not** propagate to `../14_4D_property_space`; apply fixes in both.
- **`src/ipc` is present only as a transitive dependency** of `src/analysis` (its IPC
  metric lazy-imports `src.ipc.utils.polynomials` / `degdelaysets`). It is otherwise a
  property-space concern. The clean long-term fix is to trim that import out of this
  repo's `analysis` copy.
- **`scripts/` and `notebooks/` are mixed scratch** carried into both repos for pruning.
  Some still reference property-space code — delete what doesn't belong here.
- **`data/` and `output/` are symlinks** into `../14_4D_lab_code`. The real data still
  lives there — do **not** delete `../14_4D_lab_code` until those dirs are relocated.
- `graphify-out/` is gitignored (regenerable knowledge-graph output).

## Agent skills

### Issue tracker
Issues live as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels
Default canonical label strings (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs
Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
