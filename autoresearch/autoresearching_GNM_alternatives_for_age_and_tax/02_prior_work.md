# Prior work — `multica_workspaces/7207a23d-…` summary

Adrian's earlier autoresearch effort in the Multica platform. Documented as
`KNOWLEDGE_SUMMARY.md` in workspace `63c5a83e`, with the actual code in
workspace `f965ae43`. The numbered "ADR-…" issues are an internal tracker.

This summary is for context only — that work targeted a **single empirical
connectome** (HCP Schaefer-100, group consensus). The new goal differs (see
`05_goal_options.md`), so prior conclusions transfer as priors, not as facts.

## What was being fit

Empirical target: **HCP Schaefer-100 consensus**, 10 % density, 495 edges,
single 100×100 binary adjacency.

Selection metric: **energy** = `max KS` across (degree, clustering, betweenness,
edge length). Reporting metric: **DeltaCon₀** (distance-weighted).

## The Karpathy-style file layout that was used

| File | Status | Role |
|---|---|---|
| `prepare_gnm.py` | fixed | loads data; defines energy + DeltaCon |
| `gnm.py` | agent-modifiable | wiring rules, distance terms, modulators, growth loop |
| `program_gnm.md` | human-editable | research instructions |
| `loop.py` (+ `loop_extension.py`, `loop_phase2/3.py`) | orchestrator | multi-cycle propose → evaluate → JSONL log |

Cycle structure: broad sweep (1–2 seeds) → fine refinement around top combos
(3 seeds) → modulator sweep (3 seeds) → final validation (7–50 seeds).

**Important departure from Karpathy:** Karpathy lets the agent git-reset after
a regression. The prior run used cycle-style JSONL streaming and did *not*
roll back the file between hypotheses — the "hypothesis" is a row of params,
not a code diff.

## Key findings to carry forward (~5500 hypotheses, ~94 min compute)

### What worked
1. **Clustering-based wiring rules** (`clu_max` > `clu_avg`) dominate on real
   cortex.
2. **Strong negative η** (~ −4 for power-law distance) is essential.
3. **`hub_leniency` modulator** (κ ≈ 1.5) — single biggest single-knob win.
   Confirms rich-club / hub-exemption from distance penalty.
4. **`hemisphere` modulator with α > 1** *boosts* inter-hemispheric edges.
   Counter-intuitive — power-law `D^η` under-produces callosal edges, so the
   gate corrects upward.
5. **Multi-seed averaging mandatory** above the ~0.20 energy band. A 3-seed
   minimum of 0.159 regressed to 0.190 on 50 fresh seeds.

### What didn't work / surprised
1. **Plain `D^η` beats every elaborated distance term** (EDR, piecewise,
   two-exp mixture, Gaussian resonance). The edge-length bottleneck is **not**
   solved by a smarter `f_dist`.
2. **`yeo_match` modulator with β < 1** — i.e. *discourage* within-Yeo-network
   edges — gave a slight gain once `clu_max` already captured local clustering.
   Cytoarchitectonic homophily is not an additional positive predictor.
3. **Novel rules** (`anti_matching`, `four_cycle`, `homophily_yeo`) did **not**
   beat the clustering family. `anti_matching` was the best of them (5th
   overall).
4. **DeltaCon disagrees with energy.** Energy picks
   `clu_max + powerlaw + hub_leniency`; DeltaCon picks `deg_max + piecewise`.
   The four marginal KS distributions are under-determining "good fit".

### Ceilings hit
- Energy plateaus around **0.18** on the HCP consensus.
- Edge-length KS plateaus at **~0.17** (down from 0.26 baseline, but still the
  dominant KS term).

### Falsified / out-of-scope-for-now structural extensions
The plateau suggests **structurally different generators** are needed to break
through:
- Two-phase **pioneer-follower** growth
- **Heterochronous** wiring (Akarca 2021)
- **Grow-then-prune** with activity-dependent pruning
- **Chemoaffinity / gradient-matching** rules (needs gene-expression coords)

All of these require changing the per-step growth API, not just the per-pair
probability.

## What carries directly into the new project

- **Code layout** (`prepare.py` fixed + agent-editable model + `program.md`).
- **Multi-seed-throughout** philosophy (single-seed energies are noise).
- **DeltaCon as a reporting metric** that exposes energy-function blind spots.
- **Existing rule / distance / modulator catalogue** in `gnm.py` —
  re-usable as a baseline.

## What does *not* directly carry over

- The empirical target was *one* consensus connectome. The new datasets give
  hundreds of connectomes with covariates (age, taxonomic order). The energy
  function needs to accommodate that — either by per-subject scoring, per-bin
  consensus scoring, or by promoting the covariate into a regression
  objective.
- The selection metric was `max KS` of four marginals. For covariate-aware
  evaluation that may not be enough — see `05_goal_options.md`.

## Pointers

- Code & artefacts: `/Users/adrian/multica_workspaces/7207a23d-d1d5-4b6d-9639-688195d523dc/f965ae43/workdir/`
- The consolidated readout: `…/63c5a83e/workdir/KNOWLEDGE_SUMMARY.md`
