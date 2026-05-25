# Autoresearch: GNM alternatives for age and taxonomic order

A new autoresearch experiment in this repo. The idea: borrow the Karpathy "let
an agent edit one file and iterate overnight" pattern, but apply it to **generative
network models (GNMs) of brain connectomes** — and crucially, evaluate them against
**two preprocessed datasets that carry meaningful covariate structure**:

1. `lexis_data_aging` — 718 human connectomes spanning ages 28–100 yrs.
2. `suarez_MaMI_dataset` — 225 mammal connectomes across 12 orders (Rodentia,
   Carnivora, Primates, Chiroptera, Cetartiodactyla, …) / 107 species.

The research question is *not* "match a single consensus connectome" (which is
what the prior multica project already addressed). It is whether a single
generator can be parameterised by **age** and **taxonomic order** such that, as
those covariates vary, the generated connectome's properties track the
empirical changes.

## Where things live

| File | Role |
|---|---|
| `00_README.md` (this file) | Entry point |
| `01_context.md` | What the codebase is about, key glossary, terminology |
| `02_prior_work.md` | What the previous multica project tried, found, and ruled out |
| `03_datasets.md` | The two evaluation datasets in detail (shapes, metadata, gotchas) |
| `04_karpathy_pattern.md` | How the original autoresearch pattern works, and how we'll adapt it |
| `05_goal_options.md` | Multiple framings of the research goal (background reading) |
| `06_setup_decisions.md` | **← locked decisions + git-safety plan — read this, approve, then we write code** |

Code (`prepare.py`, the agent-editable `model.py` / `gnm.py`, the orchestrator
`loop.py`, `program.md` instructions for the agent) will be added once
`06_setup_decisions.md` is approved.
