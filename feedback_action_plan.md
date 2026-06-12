# Feedback Action Plan

Synthesis of all reviewer feedback into actionable tasks, each with a proposed solution.

**Sources:** `feedback_andrea.md` + 14 inline PDF comments (`feedback_andrea_pdf_comments.md`), `feedback_duncan.md`, `feedback_francesco.md`, `feedback_lexi.md`.

**Reviewers:** A = Andrea, D = Duncan, F = Francesco, L = Lexi.

---

## Cross-cutting themes (read first)

Four points were raised by more than one reviewer. These are the high-leverage decisions; most other tasks hang off them.

- **Theme 1 - GNM framing (D, F).** Both ask: is the paper *about GNMs*, or *about benchmarking network-similarity metrics in general*, with GNMs merely a convenient network generator? Both lean toward the latter. This decision propagates to title, abstract, intro structure, and how much GNM/eta-gamma explanation is needed. → Decision **D1**.
- **Theme 2 - a single "which metric should I pick" deliverable (D, F, A).** Duncan wants one figure that names the winners across all tests; Francesco proposes a radar chart per measure; Andrea wants a short recommendations paragraph. Converges on: one consolidated decision figure + a bolder, explicit recommendation (incl. in the abstract). → Software **S4**, Text **T5/T6**, Decision **D3**.
- **Theme 3 - accuracy / recovery within the plausible range (F, A).** Francesco notes KS "wins" only because the 5 recovery targets are very different from each other; within human-plausible ranges the ranking may flip. Accuracy is, to him, *the* deciding axis. Andrea wants a second ground-truth handle (real vs artificial). → Software **S1, S3**.
- **Theme 4 - consensus-network robustness (A).** So much rests on one consensus network; how do results move under a different consensus method / dataset / single subjects? → Software **S2**.

---

## LIST 1 - Software & Plots (Claude instance + Adrian; requires touching code/figures)

Pipeline lives in `~/Documents/01_projects/14_4D_lab/14_4D_lab_code/` (`run_connectome_comparisons.py`, `run_evaluate_further_metrics.py`, configs in `configs/`).

### S1 - Parameter recovery *within the human-plausible range* [HIGH] (F, also A)
- **Ask:** Current recovery uses 5 widely-spread targets; KS wins partly because targets are very different. Does the ranking hold for small, realistic differences?
- **Solution:** Generate ~100 ground-truth networks with known (η, γ) sampled *only* from the plausible window (around η≈-3.7, γ≈0.4-0.6, the well-recovered region). Recover each on a fine grid; per measure, correlate true vs recovered η and γ (Pearson r). Report r as the "fine-grained accuracy" score. New panel/figure (bar of r per measure, or scatter of true vs recovered for top measures). Compare ranking against the existing wide-target recovery.
- **Why:** Directly tests Francesco's hypothesis that the accuracy ranking shifts in the regime researchers actually use (individuals/species).

### S2 - Consensus-network robustness analysis [HIGH] (A)
- **Ask:** How stable are the landscapes/rankings if the consensus network is replaced?
- **Solution:** Re-run the core landscape + best-fit identification for the 8 selected measures under: (a) a different consensus method (e.g. distance-dependent vs `struct_consensus`); (b) the second dataset already in SI A.2 (extend beyond KS-only to all 8 measures); (c) a handful of individual subjects instead of a consensus. Quantify rank stability across measures (e.g. correlate per-measure best-fit locations or landscape correlations across conditions). New SI figure + a summary sentence in main. Note: SI A.2 already covers seed variation for KS - reuse that scaffolding.
- **Why:** Andrea's single biggest concern; pre-empts the obvious "it all rests on one network" review.

### S3 - "Real vs artificial" ground-truth check [MED-HIGH] (A)
- **Ask:** Complement synthetic recovery with a second ground truth: do measures correctly rank real connectomes as more similar to the consensus than GNM networks are?
- **Solution:** For each measure, compute distance(consensus, each real subject) and distance(consensus, GNM networks across the grid). Test whether real subjects are scored as closer than artificial networks (e.g. AUC / separation, or % of GNMs the real subjects beat). Rank measures by this discrimination. New figure + paragraph.
- **Why:** Adds a model-free validity criterion; strengthens recommendations.

### S4 - Consolidated "which metric to choose" decision figure [HIGH] (D, F)
- **Ask:** One figure a hurried reader can use to pick a metric across all axes.
- **Solution:** Reduce each axis to one scalar per measure (most already computed): accuracy = recovery grid-steps (+ S1 r); precision = total variation (Fig 4A); robustness/stability = CV + iSNR; cost = timing; plausibility = fraction of best-fits with η>0. Min-max normalise each axis to 0-1 (higher = better). Plot a radar chart (one polygon per selected measure) **or** a compact score-card heatmap. Place in main results/discussion. Requires a small aggregation script that pulls the per-axis numbers into one table.
- **Why:** Serves the "just tell me the answer" readership (Duncan) via Francesco's radar idea. Weighting/emphasis is Decision **D3**.

### S5 - Fix Fig 1C to show all five axes [MED] (A PDF #5, F)
- **Ask:** Caption says five axes; panel shows four. Francesco also wants all axes (Plausibility, Cost, Precision, Accuracy, Robustness) to appear in Fig 1C.
- **Solution:** Update Fig 1C to depict all five evaluation axes (align names with the short result titles from **T4**). Either fix the figure to five or the caption to four - prefer five.

### S6 - Biological-plausibility: add a stronger criterion [MED, optional] (A)
- **Ask:** Plausibility is currently just degree + distance distributions; Andrea expected more, e.g. correlation with the principal structural gradient topography.
- **Solution:** Compute the principal structural gradient (diffusion-map embedding) of the empirical consensus and of each measure's best-fit networks; correlate gradient topographies as an added plausibility metric. New panel. Related to but distinct from the spectral measures. Mark optional - include if time allows; otherwise address the *definition* gap in text (**T14**).

### S7 - Promote landscape plots (+ optional per-measure schematic) to main [MED/LOW] (A)
- **Ask:** A version of the landscape plots in the main text, ideally beside an intuitive schematic of how each measure works.
- **Solution:** Move a cleaned subset of the Fig 2A landscapes (or the per-measure landscapes) into the main. Schematic-per-measure is explicitly optional (Andrea acknowledges it's hard) - scope only if cheap.

### S8 - Clarify Fig 5B (edge-importance brain plots) [MED] (A PDF #10)
- **Ask:** Unclear what the brain plots show / how to read them.
- **Solution:** Add in-figure annotation (what node positions, what edge colour/width encode), a colour-bar, and a one-line "how to read" in the legend. May need a small re-render with clearer styling. Legend wording is **T13**.

### S9 - Reconcile recovery measures vs Fig 4 [LOW] (A PDF #6)
- **Ask:** Text at p10 cites DC-NMI / NMI grid-steps, but those measures aren't in Fig 4.
- **Solution:** Mostly a cross-reference fix (point to SI Fig 13, which has all 16) - see **T20**. Only a figure task if you decide to add those measures to a main panel; otherwise text-only.

---

## LIST 2 - Text only (Claude + Adrian; no code/figures)

### T1 - Execute the reframing in prose [HIGH, depends on D1] (D, F)
- After Decision **D1**, rewrite abstract opening + intro so the broad premise is "how should we compare brain networks?" and GNMs enter later as the convenient network generator (if that's the chosen direction). Keep eta/gamma explanation proportional to the chosen framing.

### T2 - Drop "GNMs simulate network growth" [HIGH] (D)
- Intro currently: "GNMs simulate network growth through simple wiring rules." Replace "simulate network growth" with framing as *compression* of the connectome into a few wiring rules; avoid implying biological growth. Sweep the whole manuscript for similar "growth/simulate" phrasing.

### T3 - Explain GNMs / η / γ at first use [HIGH] (D, A PDF #1)
- η and γ are used in Results before being clearly defined for a general reader. Add a compact, self-contained explanation (cost penalty η, homophily γ) at first Results mention, or ensure the intro definition is unmistakable. Depth depends on **D1**.

### T4 - Short result-section titles [MED] (F)
- Add short, measurement-based subheadings: Plausibility; Computational Cost; Precision; Accuracy; Robustness. Lay them out up front; mirror them in Fig 1C (**S5**).

### T5 - Bolder recommendation in the abstract [HIGH] (F, D)
- Abstract currently hedges ("no silver-bullet"). Add an explicit line naming the advised metric(s) (e.g. KS for accuracy-critical work; DeltaCon as all-rounder; Frobenius for large-scale screening) or at minimum state that concrete recommendations are provided. Final wording depends on **D3**.

### T6 - Recommendations paragraph at end of Results (or pointer) [MED] (A PDF #8/#9)
- Add a brief paragraph at the end of Results summarising what each measure is best for, **or** a single sentence: "we provide overall recommendations in the Discussion." (Andrea is fine with either.)

### T7 - Reduce bold [LOW] (F)
- Remove full-sentence bold; bold only one or two key terms (e.g. *accuracy*, *robustness*). Sweep all chapters.

### T8 - Emphasise accuracy; note the axes aren't equivalent [MED] (F)
- In Discussion (and intro framing), state that the axes are not on the same level: a fast-but-wrong measure is useless, a slow-but-accurate one is worth it; precision without accuracy is insufficient. Foreground accuracy as the primary deciding axis.

### T9 - Explain "Canberra distance" [LOW] (A PDF #2)
- One parenthetical clause at first use (NetSimile) explaining what the Canberra distance is.

### T10 - Explain "Matusita distance" [LOW] (A PDF #3)
- Same for DeltaCon's Matusita distance at first use (brief gloss; full form already in SI A.6).

### T11 - Define iSNR signal vs noise in Results [LOW] (A PDF #7)
- One sentence in Results defining what counts as signal and noise for the iSNR (formula is in Methods 4.3 - surface it briefly where iSNR first appears).

### T12 - Clarify consensus construction in Results [LOW] (A PDF #4)
- Brief note where the consensus is introduced: deterministic vs probabilistic, how subjects were aggregated (point to `struct_consensus`). Pull a sentence forward from Methods 4.1.

### T13 - Fig 5B legend wording [LOW] (A PDF #10)
- Legend text part of **S8**: state explicitly what the brain plots depict and how to interpret edge colour/width.

### T14 - Define biological plausibility earlier [MED] (A)
- Define "biological plausibility" the first time it's used (currently only fleshed out later). If **S6** is done, reference the added criterion; if not, at least scope clearly that here it means metabolic constraint (η<0) + degree/distance distribution fidelity, and acknowledge richer criteria as future work.

### T15 - Discussion first sentence rephrase [LOW] (L)
- "Generative network modelling reduces the similarity..." → "The generative network modelling framework/methodology reduces..." (it's the procedure, not the GNMs).

### T16 - Remove duplicated seed sentence in Discussion [LOW] (L)
- The seeding point appears twice (once at the start of the paragraph, once near the end with `\cite{akarca_2021_generativeb}`). Merge into one, keep the citation.

### T17 - Add Fiber Data Hub provenance + citation (SI) [MED] (L)
- In SI A.2, make clear the second dataset's connectomes were *accessed* from DSI Studio's Fiber Data Hub (Yeh et al., 2025) following their GQI reconstruction - we did not reconstruct them. Add the Fiber Hub citation to `references.bib` (new bib entry) and cite it.

### T18 - Revisit the "landscape visualization" recommendation [LOW, depends on D4] (F)
- Francesco isn't convinced about recommending a specific metric just for landscape visualization (he'd show the same metric used in the analysis). If **D4** says drop/soften it, edit the Discussion accordingly.

### T19 - Portrait in the spatial-realism Discussion paragraph [LOW] (A PDF #11)
- Andrea flags that the spatial-realism recommendation paragraph doesn't mention portrait divergence. Check whether it should be included/excluded and state why.

### T20 - Fix recovery cross-reference [LOW] (A PDF #6)
- Point the p10 recovery text (which lists all-16 measures incl. NMI/DC-NMI) to SI Fig 13 rather than implying Fig 4, so the cited numbers map to a figure that actually shows them.

---

## LIST 3 - Decisions & everything else (Adrian + co-authors; strategic, not executable by Claude alone)

### D1 - GNM as *topic* vs GNM as *tool* [BLOCKER] (D, F)
- The framing decision both Duncan and Francesco push. Recommendation: lean to "benchmarking similarity metrics; GNMs are the generator," which de-risks the GNM-skeptic reviewer and broadens appeal - but confirm with co-authors. Unblocks **T1, T2, T3, T5**, and possibly the title.

### D2 - Scope of new experiments [HIGH] (A, F)
- Decide which of **S1, S2, S3, S6** actually get run given time. Suggested priority: S1 (plausible-range recovery) and S2 (consensus robustness) first; S3 next; S6 optional. Some can be SI-only.

### D3 - Headline metric recommendation + radar weighting [HIGH] (D, F)
- Agree the single take-home recommendation for the abstract/decision figure, and how the axes are weighted in **S4** (Duncan: "depends what you think most important"). Francesco's view: accuracy dominates. May depend on S1 results.

### D4 - Keep the landscape-visualization recommendation? [MED] (F)
- Decide whether to keep, soften, or drop the "use portrait divergence for landscape visualization" advice. Drives **T18**.

### D5 - Title [MED, depends on D1] (D implied)
- If reframed, retitle away from "...for Generative Connectome Models" toward network-similarity-metric benchmarking.

### D6 - Author-summary / journal-fit pass [LOW] (general)
- Once framing settles, revisit the PLoS Author Summary (`10_author_summary.tex`) and abstract to match the new emphasis.

---

## Suggested order of attack

1. **D1** (framing) - unblocks the most text.
2. **S1 + S2** (the two experiments reviewers will expect) in parallel with text edits.
3. **S4 + D3** (decision figure + headline rec), then **T5/T6** (abstract + recommendations prose).
4. Mop-up text: T2, T3, T4, T7-T20.
5. Optional: S3, S6, S7.
