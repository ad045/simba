# The five criteria

Every read-out below is computed by `simba_networks/benchmark.py`, for your
measure and for the published ones alike. Where the paper's Methods give more
detail, they are the authority; this page states what the code does.

Two conventions hold throughout:

- **Orientation.** All rankings use the measure oriented so that lower means
  more alike. Similarities are negated first.
- **Two selections.** The *best-fitting parameter combinations* are the cells of
  the 50 x 50 grid with the lowest mean distance over their 10 replicates. The
  *best-fitting networks* are the individual networks with the lowest distance.
  The paper keeps the two apart, and so does the table.

The morphospace is $\eta \in [-8, 3]$, $\gamma \in [-0.1, 1]$, matching-index
rule, 10 replicates per cell: 25,000 networks of 495 edges.

## Agreement

| Read-out | Definition |
|---|---|
| Mean r with the selected landscapes | Pearson correlation of the measure's 25,000 values with each of the eight selected published measures (excluding itself), averaged. |
| Largest \|r\| with a published landscape | Maximum over all 16 published measures. The paper treated \|r\| of 0.973 or more as redundant. |

## Biological plausibility

| Read-out | Definition |
|---|---|
| Best-fitting eta, gamma | The best-fitting combination. |
| Combinations at eta > 0, of the 100 / 20 best | Positive $\eta$ means wiring that prefers long connections, which brains do not. The paper excluded measures with any among their 20 best. |
| Total variation of the 20 best combinations | $\mathrm{var}(\eta) + \mathrm{var}(\gamma)$ of the 20 best combinations, each axis scaled to the morphospace extent. Lower = the measure points to one region. |
| Mean connection length / connections beyond 90 mm | Pooled over the edges of the 20 best-fitting networks, using the Euclidean distance between parcel centroids. Compare with `sb.empirical_summary()`: the 100 individuals span 39.5-44.8 mm and 3.4-9.3%. |
| Degree SD | Standard deviation of node degree, averaged over the 20 best-fitting networks (individuals: 3.49). |

## Computational efficiency

| Read-out | Definition |
|---|---|
| Runtime per comparison | Mean wall-clock time of one call, over the 25,000 comparisons. |

Runtime depends on the machine. The published values are one reference
measurement; a new measure is timed on yours. For a like-for-like comparison,
time a published measure on the same machine, e.g.
`sb.evaluate(sb.MEASURES["frobenius"], criteria=["efficiency"])`.

## Sensitivity and robustness

| Read-out | Definition |
|---|---|
| MAE against the linear response | The reference is progressively rewired (degree-preserving double-edge swaps, 0-100% of its edges in 101 steps, 200 trajectories). Along each trajectory, the measure's distance to the unperturbed reference and the effective degeneration (edges that actually differ) are min-max normalised; the curves are averaged over trajectories, and the MAE is the mean absolute deviation of the measure's curve from the degeneration's. |
| CV across the 200 trajectories | The measure's values are min-max normalised over all trajectories (similarities as $1 - x$); at each step, standard deviation over mean across trajectories; averaged over steps. |
| Noise tolerance $N^*$ | The reference is rewired by $s$ connected double-edge swaps ($s$ = 1, 2, 5, 10, 20, 50, 100, 200; 20 draws each, 10 for measures slower than 10 ms, which recover on a window of plus or minus 10 cells around the unperturbed best fit). The best-fitting combination is recovered against each perturbed reference. $N^*$ is the effective degeneration at the largest $s$ whose median drift stays within one grid step. |
| Intrinsic signal-to-noise ratio | $10 \log_{10}\big(\mathrm{var}(\text{cell means}) / \text{mean}(\text{cell variances})\big)$ over the 2,500 cells; invariant to rescaling the measure. |

## Accuracy

| Read-out | Definition |
|---|---|
| Recovery error, whole morphospace | 100 test networks generated at parameters drawn uniformly from the whole morphospace are compared with 100 grid-point consensuses on a 10 x 10 grid (30 networks each). The recovered cell is the closest one; the error is the Euclidean distance, in grid steps, to the cell nearest the true parameters. Chance: 4.96. |
| Recovery error, plausible window | Same inside $\eta \in [-3.96, -1.49]$, $\gamma \in [0.08, 0.30]$ (grid of 20-network consensuses). Chance: 4.53. |
| r(true eta, recovered eta) | Pearson correlation, per experiment. |
| AUC real vs GNMs | Only with the optional subject files (see [Reference connectome](reference.md)): probability that a real subject is closer to its leave-one-out consensus than a generated network is to the consensus; against all 25,000 networks, and against the measure's own 500 best. |
