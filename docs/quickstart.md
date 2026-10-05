# Quickstart

## 1. Write the measure

A measure is any function `f(A, B) -> float` on two binary, symmetric
100 x 100 adjacency matrices (NumPy float arrays, Schaefer-100 order, no
self-loops). `A` is the generated network and `B` the reference. Return `nan`
or raise if the comparison fails; the benchmark records it as missing.

```python
import numpy as np

def degree_l1(A, B):
    """L1 distance between the two degree sequences."""
    return float(np.abs(A.sum(0) - B.sum(0)).sum())
```

Lower must mean more alike. If your measure is a similarity (higher = more
alike), say so with `similarity=True`; the benchmark then flips it before any
ranking, as it does for the five similarities among the published measures.

## 2. Drop in the reference

Put the empirical consensus at `~/simba_networks_data/reference/consensus.npy`,
or pass it as `reference=`. See [Reference connectome](reference.md).

## 3. Evaluate

```python
import simba_networks as sb

report = sb.evaluate(degree_l1, name="degree L1")
print(report)
```

`report.table` has one row per read-out: the criterion it belongs to, the
desirable direction, your measure's value, the value of each of the eight
selected published measures, and your measure's rank among the nine (1 = best).
[The five criteria](criteria.md) defines every row.

`report.landscape` holds your measure's value and runtime for each of the
25,000 networks, with their `eta`, `gamma` and replicate, so you can plot its
landscape or inspect the best-fitting networks yourself.

## Choosing what to run

```python
sb.evaluate(degree_l1, criteria=["agreement", "plausibility", "efficiency"])
sb.evaluate(degree_l1, criteria=["accuracy"])    # needs no reference
```

| Criterion | Calls of your measure |
|---|---|
| agreement, plausibility, efficiency | 25,000 (one pass over the morphospace, shared) |
| sensitivity and robustness | 20,200 for the rewiring trajectories, plus one landscape per noise-tolerance draw: 160 x 25,000 for a measure faster than 10 ms, 80 x 4,410 for a slower one |
| accuracy | 20,000 (two recovery experiments of 100 x 100) |

The noise tolerance $N^*$ dominates the cost; it follows the paper's protocol
exactly, so that the number is comparable. For a first look, leave out
`"sensitivity"`.

## Parallel runs

```python
report = sb.evaluate(degree_l1, n_jobs=-1)     # all cores
```

With `n_jobs` other than 1, the measure must be importable by the worker
processes: define it in a module (not a lambda, not only in a notebook cell).
Runtimes are measured per call in either case, but on a busy machine the
parallel ones are inflated; measure efficiency with `n_jobs=1`.

## Comparing against the published measures directly

```python
sb.published_readouts()          # every read-out, all 16 measures
sb.MEASURES["delta_con"](A, B)   # any published measure as a function
```

`published_readouts()` recomputes the published side from the shipped tables
with the same code that scores your measure, so the two columns of a report
cannot disagree about definitions.
