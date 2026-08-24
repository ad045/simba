# Codebase context

## What this repo is

`14_4D_lab_code/` is Adrian's MSc thesis pipeline. The headline question:

> Memory capacity (or alternatively something else like transfer entropy?) as a
> potential driver of brain network topology: A reservoir-computing approach
> to understanding long-range connections.

The pipeline grows binary connectomes with **generative network models (GNMs)**
in the Betzel et al. 2016 family and compares them to empirical connectomes on
a battery of metrics (degree, clustering, betweenness, edge length, plus a long
tail of optional ones: ORC, hyperbolicity, persistent H1, memory capacity, …).

## Key data-handling pattern

Every preprocessed dataset under `data/preprocessed/<name>/` follows the same
shape:

```
01_connectomes/             — adjacency matrices, .npy (N_subjects, N_nodes, N_nodes)
02_distance_matrices/       — Euclidean distance, .npy
03_graph_measures/          — precomputed metrics (where applicable)
04_further_info/            — metadata (ages, species, …) per-subject
05_seeds_of_humans/         — empirical seed adjacency matrices for GNM growth
```

`05_seeds_of_humans/` is specific to HCP — the GNM starts from a known
sub-skeleton (usually the minimum spanning tree, MST) rather than a single
random edge.

## Things to remember about the GNM growth API

- Binary, undirected, no self-loops.
- Sequential edge addition: each step samples one new edge with probability
  ∝ `f_dist(D_uv ; θ) · K(u,v ; G_t)^γ · extras(u,v)`.
- Total edge count `n_edges` is set by matching empirical **density** (10 % in
  every config seen so far → 495 edges on N=100).
- Multi-seed averaging is mandatory above ~0.20 energy because the growth
  trajectory is highly noisy.

## Glossary

| Term | What it means here |
|------|---|
| **GNM** | Generative network model — Betzel-style sequential edge addition. *Not* a graph neural network. |
| **GNN** | Graph neural network — a learned model that operates on graphs. Different family. The user wrote "GNN" in one place and "GNM alternatives" in the folder name; my reading is they mean *alternative generative models*, possibly including GNN-style decoders. Worth clarifying. See `05_goal_options.md`. |
| **Energy** | `max KS` across (degree, clustering, betweenness, edge length). Lower is better. Standard Betzel selection metric. |
| **DeltaCon** | Koutra et al. 2013 whole-graph similarity. Used as a reporting metric in the prior multica run. Disagreed with energy. |
| **Wiring rule (K)** | The topological term in the per-pair probability. Variants: `spatial`, `deg_{avg/min/max/prod}`, `clu_{avg/min/max}`, `matching`, `anti_matching`, `four_cycle`, `homophily_yeo`. |
| **Distance term (f_dist)** | The geometric term. Variants: `powerlaw`, `edr`, `piecewise`, `two_exp_mixture`, `gaussian_resonance`. |
| **Modulator** | Multiplicative gate — `hemisphere`, `yeo_match`, `hub_leniency`. |
| **Energy bottleneck** | On HCP Schaefer-100, edge-length KS ≈ 0.17 even after exhaustive search — the dominant residual. |
| **Lexi / Lexis data** | A dataset originating from "Lexi"; preprocessed in `08_preprocessing_lexis_data*.ipynb`. `lexis_data_aging/` has ages attached. |
| **Suarez MaMI** | Suárez et al. mammalian-MRI dataset. 225 individuals, 12 orders, 107 species. Per-animal distance matrix (animal brains aren't aligned to a common atlas). |
| **Animal IDs** | In `run_experiment_lexis_data.py`, the integer `animal` actually indexes a row of the *human* lexis_data participants — naming is legacy. The Suárez side uses real animal names (Rat4, Orangutan2, …). |

## Where to look in the code

- `run_experiment_lexis_data.py` — top-level driver, takes a YAML config and
  loops `animal` IDs.
- `configs/config_gnm_run_*.yaml` — example configs per dataset, including
  `lexis_dataset` and `suarez_MaMI_dataset`.
- `src/GNMs/gnm_network_generator.py` — the GNM growth implementation
  (depends on Adrian's patched GNM library — see top-level README).
- `src/comparing_connectomes/` — every metric / comparer used for evaluation,
  including `delta_con_distance_evaluator.py`.
- `src/preprocessing/` — how each dataset was built. `08_*.ipynb`,
  `09_*.ipynb`, `10_*.ipynb` for Lexi's data; `00_preprocessing_pipeline_suarez_MaMI*`
  for Suárez.
- `metrics_reference.md`, `metrics_overview.csv` — the long catalogue of
  candidate metrics with notes on which are usable in an inner loop vs. only
  as a final re-rank.

## Style preferences that show up in the code

- `.npy` for matrices, `.csv` for metadata, `.yaml` for configs.
- Datasets are stored under `data/preprocessed/<name>/` with a fixed
  sub-directory naming scheme — keep new outputs consistent with it.
- The previous multica project kept everything one level deep in a `workdir/`
  — flat layout, easy to scan. The Karpathy reference also uses a flat layout.
  No reason to nest.
