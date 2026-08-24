# archive

Work that is **not** part of *How Similar Are Two Brains?* but was part of getting
there. Nothing here is needed to reproduce the paper. It is kept so the path
taken stays visible rather than being quietly deleted.

| Folder | What it is |
|---|---|
| `visualization/` | The exploratory notebooks and plotting scripts that preceded the manuscript figures - earlier measure selections, landscape plotting experiments, QC passes, cross-species comparisons, superseded drafts of the figure notebooks. |
| `preprocessing/` | Preprocessing for datasets this paper does not use: Suárez/MaMI cross-species, Griffa 70 young adults, Shafiei consensus, Kayson's generated networks, ring-lattice and optimal-network constructions, ESN task data. |
| `configs/` | GNM sweep configs for those same datasets. |
| `misc/` | An unrelated autoresearch side project (nanoGPT), agent tooling, environment exports, an empty radar-plot template, a feedback action plan, and `X_`-prefixed superseded drafts. |

Many scripts here still carry absolute paths into the author's machine and refer
to data trees that are not shipped. They are preserved as historical record, not
as runnable code.

To drop the lot: `git rm -r archive/`
