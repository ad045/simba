# Goal options for discussion

The brief was:

> Auto-research if there is a GNN model which performs better at capturing
> changes with progressing age and changing taxonomic order.

That sentence has at least three genuine ambiguities, and they multiply with
each other into very different research projects. Below I lay out the
ambiguities, then five concrete goal framings to choose between, then a
recommendation.

## What I think you mean (sanity check)

- **GNN vs GNM.** You wrote "GNN" (graph neural network) in the brief but the
  folder name says "GNM alternatives". Both are defensible interpretations.
  My working assumption is: *alternatives to the Betzel-family GNM*, which
  could include — but isn't restricted to — GNN-based generators. If you
  actually mean **a learned, neural network-based** generator specifically,
  that narrows the search a lot. Worth confirming.
- **"Captures changes with progressing age."** Could mean: (a) the model's
  output distribution shifts the way the empirical distribution shifts, or
  (b) the model's *parameters* track age in an interpretable way, or
  (c) the model is *conditioned on* age and reproduces age-specific
  connectomes. These are different objectives.
- **"Changing taxonomic order."** Twelve mammalian orders, very unbalanced
  sample sizes. Could mean: cross-order generalisation, order-conditioned
  generation, or recovering order from generated networks.
- **"Performs better."** Better than what baseline? Better-by-what-metric?
  The prior project used `max KS` energy + DeltaCon. For
  covariate-conditioned generation we probably need additional metrics like
  *covariate recovery accuracy* or *delta-energy across bins*.

## Five goal framings (pick one or hybridise)

### A. Covariate-conditioned generator, single global model

> **One generator** `g(D, age) → A` (or `g(D, taxon) → A`) is fit globally
> across all subjects/animals. The agent tries to find architectures /
> parameterisations such that varying the covariate causes the generator's
> output to track empirical age- or taxon-specific connectome shifts.

- Scoring: average per-subject energy (or DeltaCon) against the matching
  empirical connectome, with the covariate as an input. Plus a
  *covariate-derivative* term that rewards how much output topology actually
  changes when the covariate changes.
- Two evaluations, two scalars; pick one (age) as the primary or combine.
- Pros: closest fit to the literal brief; immediate win-condition is clean.
- Cons: deciding "the model captures the change" requires *more* than per-bin
  energy — needs a regression-style score (does the predicted-vs-empirical
  delta correlate across bins?).

### B. Recoverability framing — generator-as-feature-extractor

> Fit a generator per subject (or per age-bin / per order). The generator
> outputs a small parameter vector (η, γ, modulator weights, …). Train a
> classifier on those parameters to predict age (regression) or order
> (classification). **"Better" = higher recoverability.**

- Scoring: cross-validated R² for age, balanced accuracy / macro-F1 for
  order. Both reported.
- Pros: very crisp metric. Generators that don't encode age/order info won't
  score, regardless of how well they match marginals.
- Pros: tests an underexplored claim — the GNM literature largely shows that
  fitted GNM parameters *do* vary across age and species but rarely tests
  *how recoverable* the covariate is from the fitted parameters.
- Cons: requires per-subject fits (~700 + ~225 = ~950 fits per architecture).
  The agent needs an efficient inner fitter, not a 1-h grid search per
  subject.

### C. Bin-consensus benchmark, with a covariate-derivative term

> Use the **per-age-bin** consensuses already preprocessed
> (`lexis_data_aging_consensus_per_age_*`) as multiple empirical targets, and
> the **per-order consensuses** built from the Suárez dataset as another
> multi-target evaluation. A "good" model is one that fits all bin
> consensuses *and* whose fitted parameters trend monotonically across the
> covariate.

- Scoring: per-bin energy averaged across bins + a monotonicity penalty on
  parameters.
- Pros: avoids the per-subject fit cost; bins are already prepared.
- Pros: directly addresses "captures changes" by reading off how the model's
  parameters move as the covariate moves.
- Cons: bin consensuses are themselves opinionated — they bake in the
  binning choice (1-yr vs 2-yr) and the consensus-construction algorithm.

### D. Two separate small-budget runs, one per dataset

> Drop the "joint" framing. Run one autoresearch loop on `lexis_data_aging`
> (objective: predict age from a generator) and one on `suarez_MaMI_dataset`
> (objective: predict order from a generator). Compare what each loop ends up
> selecting.

- Scoring: per-loop, the metric for that dataset.
- Pros: smaller, more tractable. Lets you compare what age-driving features
  and taxonomy-driving features look like.
- Pros: matches the prior multica project's "one connectome, one target"
  shape — the new ingredient is just *which* connectome each loop targets.
- Cons: punts on "are these the same kind of changes?" — which is the
  scientific question lurking in the original brief.

### E. Comparative shoot-out, fixed generators

> Don't have the agent *invent* generators — instead, **enumerate** a fixed
> set (classic Betzel rule × distance term × modulator combos, plus a few
> GNN-based candidates like a VGAE conditioned on age and a Graph Isomorphism
> Network decoder) and let the agent autonomously fit each one to both
> datasets, then write up which performs best on which criterion.

- Scoring: a 2D table — rows = generators, columns = (age-recoverability,
  taxon-recoverability, mean per-subject energy, edge-length KS, …).
- Pros: shorter; less ambitious; perfectly answers the literal brief.
- Cons: doesn't really "autoresearch" — closer to "auto-benchmark". The
  agent's role shrinks to writing the loop and producing the table.

## Quick comparison table

|     | Conditioning | What's "better" | Code complexity | Inner-loop speed | Closest to brief |
|---|---|---|---|---|---|
| A | continuous (age) / categorical (taxon) | per-subject energy + covariate-delta | medium | medium | strong |
| B | per-subject fit | recoverability R² / accuracy | high | low (many fits) | medium |
| C | bin consensus | per-bin energy + parameter monotonicity | low–medium | high | strong |
| D | two separate runs | metric per dataset | low | high | medium |
| E | none (enumerated) | benchmark table | low | high | weak (no "research") |

> **Note (2026-05-25):** The recommendation below was accepted with one
> tweak: the primary age dataset is now `lexis_data_developing` (6–22 yr),
> with `lexis_data_aging` (28–100 yr) as secondary. Final, executable
> decisions live in `06_setup_decisions.md`. The remainder of this file is
> kept as background.

## Recommendation

A **hybrid of C + A**, in that order:

1. **First, with `prepare.py` frozen**, define the evaluator as:
   - per-bin energy on age (using `lexis_data_all_consensus_per_age_2_year/`),
   - per-order energy on taxonomy (built once from Suárez metadata),
   - both reported, and a *single composite scalar* the agent optimises.
2. Let the agent edit `model.py` to introduce **age-/order-conditional**
   wiring rules, distance terms, and modulators (framing A) — but evaluate
   them against the bin consensuses (framing C's cheaper objective).
3. Add a *parameter-monotonicity* secondary metric to flag generators whose
   fitted η or γ track age/order monotonically — that's the
   "captures the change" criterion in concrete form.

This way:
- The inner loop stays fast (bin consensuses, not per-subject fits).
- The agent has a real architectural search space, not a parameter grid.
- "Captures changes" is operationalised as both *per-bin fit quality* and
  *parameter trajectory*, not just energy minimisation.
- We sidestep B's expensive per-subject fits unless we want to graduate to
  them later.

## Open questions to settle before locking the goal

1. **GNN, GNM, or both?** If you want learned (PyTorch-style) generators in
   scope, that doubles the dependency footprint and requires defining a
   training budget. If "GNM alternatives" means *non-neural* alternatives
   only, we stay in the previous project's regime.
2. **Single scalar or multi-objective?** The Karpathy template assumes one
   scalar. If you want the agent autonomous overnight, we need exactly one
   number it tries to push down — but that number can be a composite (e.g.
   weighted sum of age-energy + taxon-energy + monotonicity penalty).
3. **Which Lexi sub-cohort?** `lexis_data_aging` is 28–100 yrs; you also
   have `lexis_data_young`, `lexis_data_developing`, and `lexis_data_all_*`.
   The fuller the age range, the more "aging" signal there is — but the
   harder it gets to fit one generator across it all. *(Recommended:
   start with the 2-year bins from `lexis_data_all_consensus_per_age_2_year`
   — fewer bins, more samples per bin.)*
4. **Order vs super-order?** Taxonomic order has 12 levels but only 5 are
   well-populated (>20 samples). Super-order has 5 levels but only 2 are
   well-populated. *(Recommended: order, restricted to the top 5 by sample
   count: Carnivora, Primates, Cetartiodactyla, Chiroptera, Rodentia.)*
5. **Time budget per hypothesis?** Karpathy's default is 5 minutes. The
   prior multica project's individual evaluations were on the order of
   seconds. We'll want a budget aligned to "the agent can run ~100
   hypotheses overnight" — so something in the 1–5 minute range for a
   single end-to-end evaluation including any inner fit.

Once you've picked a framing (or proposed your own), I'll write `program.md`,
`prepare.py`, an initial `model.py` baseline, and `loop.py` accordingly.
