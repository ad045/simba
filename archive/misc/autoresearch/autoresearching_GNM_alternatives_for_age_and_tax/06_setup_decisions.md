# Setup decisions (locked) + git-safety plan

The decisions below answer the open questions in `05_goal_options.md`. They're
*recommendations as defaults* — easy to revisit by editing this file before we
start writing code.

## Locked decisions

### Goal framing
- **C + A hybrid** from `05_goal_options.md`:
  1. Inner loop scores **per-bin consensus connectomes**, not per-subject fits
     (cycle is fast: ~seconds per bin).
  2. The generator is **covariate-conditional** — the same `model.py` must
     handle every bin; it's not allowed to re-fit independently per bin.
  3. A **parameter-monotonicity** secondary metric flags generators whose
     fitted η / γ / κ track the covariate.

### GNM, not GNN
- We work in the **classical Betzel-family GNM** regime. No PyTorch, no learned
  decoders. The agent extends `gnm.py` with new wiring rules, distance terms,
  modulators, growth modifications — same surface as the prior multica run,
  now parameterised by age and order.
- If at some point a generator wants to *learn* a parameter from the data, that
  is fine — but the search space stays GNM-shaped (probabilistic edge
  addition), not GNN-shaped (message-passing on a graph).

### Selection metric: **Energy** (max-KS)
Why energy and not DeltaCon:
1. Energy is the established Betzel selection metric → directly comparable to
   the prior multica run, Betzel 2016, Liu 2024, Akarca 2023, etc.
2. Energy is **cheap** (four KS stats on N=100 vectors) → fits the inner loop
   when we have to evaluate 18+ bins × 3 seeds per hypothesis.
3. DeltaCon was the prior project's *reporting* metric and it preferred
   `deg_max + piecewise` — which had **worse marginals**. That tells us
   DeltaCon is missing the topology signals KS captures, not the other way
   round.
4. DeltaCon stays as a **parallel reporting-only metric** (computed for every
   hypothesis, logged, never used to rank) — same role it played before. If
   the two metrics disagree we want to *see* the disagreement, not hide it.

### Composite scalar the agent minimises
The agent must minimise **one number**, otherwise the loop has nothing to
greedy-search. Composite:

```
score = mean_age_energy + mean_taxon_energy
        + 0.5 * worst_bin_energy_age
        + 0.5 * worst_bin_energy_taxon
        + 0.1 * (1 − abs(spearman(bin_index, fitted_param)))
```

- `mean_*_energy` is the average energy across all bins on that axis (drives
  global fit quality).
- `worst_bin_energy_*` penalises the agent for sacrificing one bin to gain on
  another (drives consistent fit across the covariate range).
- The monotonicity term is small (λ = 0.1) — a tiebreaker that flags
  generators whose parameters move *coherently* with the covariate. Spearman
  rather than Pearson because we only care about ordering.

`mean + worst` is intentionally not a Pareto front — we want one number, and
"average plus worst-case" is a reasonable scalarisation that the literature
uses for fairness in similar settings.

### Datasets and bins
- **Primary age axis**: `lexis_data_developing` (ages 6–22), 2-year
  consensuses → **9 bins**.
- **Secondary age axis**: `lexis_data_aging` (ages 28–100), 2-year consensuses
  → ~37 bins (we'll cap to 9 by re-binning into ~8-yr buckets to match
  developing's resolution and keep eval cost bounded).
- **Taxonomy axis**: `suarez_MaMI_dataset`, restricted to the **top 5 orders**
  by sample count → Carnivora (51), Primates (46), Cetartiodactyla (40),
  Chiroptera (30), Rodentia (29). For each order, a per-order consensus is
  built once in `prepare.py`. The 1–9 long-tail orders are excluded (too few
  samples for a stable consensus).
- Aging + developing are evaluated **separately** (different distance matrices
  is fine — same one actually — but separate score components for clarity);
  the score adds them.

### Time budget per hypothesis
Each hypothesis evaluates against **~24 bins** (9 developing + ~10 aging + 5
taxonomy), each with **3 seeds**. Per growth ≈ 1 s on N=100, plus KS + DeltaCon
≈ negligible. Plus optional per-bin parameter fit (~5–10 inner iterations of
analytic-fitting or coarse grid search).

Realistic budget: **3 minutes wall-clock per hypothesis**, ~20/hour, ~150 over
an 8-hour overnight run. That's the same order as Karpathy's 12/hour.

## Git safety plan

The risk is that the agent runs `git reset --hard` or commits inside the main
`14_4D_lab_code` repo and damages your thesis state. Fix: make the autoresearch
project its **own git repository**, nested inside a folder the parent repo
ignores.

This is exactly how `autoresearch/autoresearch_karpathy/` is set up — it has
its own `.git/` (you can see it: `ls autoresearch/autoresearch_karpathy/.git`).
The parent repo never tracks the inner one.

### Concrete steps (to be approved + executed)

```bash
# 0. We're at the repo root: /Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code
# 1. Tell the parent repo to ignore the inner autoresearch project.
#    Append one line to the parent .gitignore (manual edit, not a destructive op):
#    /autoresearch/autoresearching_GNM_alternatives_for_age_and_tax/

# 2. Initialise a fresh git repo inside the project folder
cd autoresearch/autoresearching_GNM_alternatives_for_age_and_tax
git init -b main
git add 00_README.md 01_context.md 02_prior_work.md 03_datasets.md 04_karpathy_pattern.md 05_goal_options.md 06_setup_decisions.md
git commit -m "initial context: brief, prior work, datasets, goal options"

# 3. Each autoresearch session runs on its own branch
git checkout -b autoresearch/<tag>   # tag e.g. "may25" — the date works
```

That isolates everything the agent does to:
`autoresearch/autoresearching_GNM_alternatives_for_age_and_tax/.git`.

A `git reset --hard` inside that folder cannot touch:
- the parent repo's working tree (different .git, different index)
- any other folder
- your thesis notebooks, the prior multica artefacts, etc.

### Inner-repo `.gitignore`

Will be created with the first commit to keep run noise out of git:

```
# heavy / disposable / per-run artefacts
results/
run.log
*.tmp.*
__pycache__/
.DS_Store

# the experimental log we maintain by hand, not by git
# (per Karpathy's results.tsv convention — agent records to it, doesn't commit it)
results.tsv
```

### Parent `.gitignore` addition

Just one line, appended to `14_4D_lab_code/.gitignore`:

```
/autoresearch/autoresearching_GNM_alternatives_for_age_and_tax/
```

(The leading `/` anchors it to the repo root, so it only matches that exact
folder.)

### Hard rules in `program.md` (to keep the agent in the safe zone)

When we write the `program.md` skill for the agent, it will include:

> **Git scope.** You operate inside this folder's git repo only. Every
> `git`, `cd`, or filesystem write you do is relative to the current working
> directory. Never run `git` from anywhere outside this folder. Never run
> `git reset --hard`, `git checkout .`, `git clean -fd`, or any other
> destructive op unless you are *in this folder* and *on the
> `autoresearch/<tag>` branch*. If `git rev-parse --show-toplevel` does not
> point at this folder, stop and ask the human.

That's belt-and-braces — the nested-repo setup already prevents cross-repo
contamination, but the explicit instruction reminds the agent.

### What happens if a session goes catastrophically wrong

- **Worst case**: the agent corrupts the inner repo. You delete the whole
  `autoresearching_GNM_alternatives_for_age_and_tax/` folder and `git clone`
  a backup, or re-run the setup steps. **No thesis code touched.**
- **Backup**: push the inner repo to a private GitHub repo periodically — the
  `results.tsv` and selected commits are what's expensive to recreate, and
  GitHub gives you a recoverable mirror.

## What I'm asking you for

1. **Approve / tweak the goal-framing locks** above. The headline ones:
   - Energy as selection, DeltaCon as reporting ✅
   - 5 most-populated mammalian orders as taxonomy bins ✅
   - 2-year developing consensus + 8-year aging consensus as age bins ✅
   - 3-min per-hypothesis budget ✅
   - Composite scalar = mean + worst-case + small monotonicity term ✅
2. **Approve the git-safety setup** so I can run the three commands in
   "Concrete steps".
3. Once both are approved, I write `prepare.py`, an initial baseline
   `model.py`, `loop.py`, and `program.md` — and we hand off to the agent.
