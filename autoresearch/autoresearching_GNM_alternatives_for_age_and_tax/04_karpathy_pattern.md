# The Karpathy autoresearch pattern, adapted

`autoresearch/autoresearch_karpathy/` is a faithful copy of Karpathy's original
single-GPU LLM-pretraining autoresearch demo
(<https://x.com/karpathy/status/2029701092347630069>).

## The original three files

| File | Status | Lines | Role |
|---|---|---|---|
| `prepare.py` | **read-only** | ~390 | Download data + train BPE tokenizer + `evaluate_bpb`. |
| `train.py`   | **agent-editable** | ~630 | Single file containing the full GPT model, optimiser, training loop. Everything fair game: architecture, hyperparameters, optimiser, batch size. |
| `program.md` | **human-editable** | ~115 | Instructions to the agent: branch convention, what's allowed, what's off-limits, the experiment loop. |

The loop the agent runs (from `program.md`):

```
LOOP FOREVER:
  1. Read current branch / commit
  2. Tune train.py with one experimental idea
  3. git commit
  4. uv run train.py > run.log 2>&1
  5. grep val_bpb / peak_vram from run.log
  6. record row in results.tsv (commit, val_bpb, memory, status, description)
  7. if better → advance branch; else → git reset
```

## Why this pattern fits us

1. **One file the agent owns** keeps diffs auditable.
2. **A frozen benchmark** (`prepare.py` + `evaluate_bpb`) guarantees apples-to-apples
   comparisons across radical architectural changes.
3. **A single scalar metric** (`val_bpb`) — lower is better, comparable
   regardless of model size, vocab, etc.
4. **A fixed time budget** (5 minutes wall-clock) means ~12 hypotheses/hour,
   ~100 overnight.
5. **The agent edits real code**, not a config grid — so it can introduce *new*
   wiring rules, *new* distance terms, *new* growth procedures, not just sweep
   parameters.

## How the prior multica project deviated, and what we should pull back

The prior run mostly stayed in **grid mode**: the "hypothesis" was a row of
parameters, not a code diff. Cycles 1–8 streamed JSONL rows of `(rule, dist,
modulators, params)`. That's powerful for parameter exploration but it's *not*
the Karpathy pattern — the agent never edited `gnm.py` between hypotheses.

For the new project we want both:
- Karpathy-style **code-edit cycles** when the change is structural (a new
  growth procedure, a new distance term, a new evaluation hook).
- Grid-mode sweeps **inside** a cycle when the change is "find good
  parameters for this new rule I just added".

## Concrete file layout we'll mirror

```
autoresearching_GNM_alternatives_for_age_and_tax/
  00_README.md           ← entry point (already written)
  01_context.md          ← codebase context (already)
  02_prior_work.md       ← prior multica findings (already)
  03_datasets.md         ← data shapes + load snippets (already)
  04_karpathy_pattern.md ← this file
  05_goal_options.md     ← decision point ← **read & pick**

  program.md     ← human-editable: research goals + constraints + loop
  prepare.py     ← read-only: data loading + the frozen evaluator
  model.py       ← agent-editable: the generator (current candidate)
  loop.py        ← orchestrator: propose / run / log / advance-or-reset
  results.tsv    ← append-only log (commit, score, memory, status, description)
  run.log        ← latest experiment's stdout
```

## What `prepare.py` will need to define (frozen)

- Functions to load both datasets exactly as in `03_datasets.md`.
- A frozen evaluation function — call it `evaluate(model_fn)` — that returns a
  **single scalar** the agent tries to minimise. Candidates for what that
  scalar is are the whole point of `05_goal_options.md`.

## What `model.py` will look like (agent-editable)

A single function (or class) `generate(D, n_edges, covariate, seed) → A` that
takes the covariate (age in years, or one-hot of taxonomic order, or both) and
returns a 100×100 binary adjacency matrix. The agent is allowed to change
*everything* about how that function works — the growth procedure, the
parameterisation by covariate, the optimisation scheme — provided
`prepare.py`'s evaluator can still call it.

## What `program.md` will tell the agent

Mirroring Karpathy's structure:

- **Setup**: a branch convention (`autoresearch/<tag>` per session), how to
  initialise `results.tsv`, where the data and tokenizer-analog live.
- **What you can do**: edit `model.py` freely.
- **What you cannot do**: touch `prepare.py`, change the evaluator, install
  new packages, change which datasets are used.
- **The loop**: identical to Karpathy's — propose → run → log → advance or
  reset — with one extra rule: *never stop*. The user might be asleep.

We can finalise this once the goal is locked in `05_goal_options.md`.
