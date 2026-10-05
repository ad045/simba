![SimBa](assets/simba.png)

# SimBa - Similarity Benchmark for Brain Networks

Companion package to *How Similar Are Two Brains? A Comprehensive Benchmark of
Brain Network Similarity Measures* (Dendorfer, Luppi, Poli, Mousley, Astle and
Fakhar).

The paper benchmarks 16 network distance measures on a single task: comparing
networks produced by a generative network model (GNM) with an empirical human
connectome. It scores each measure on five criteria:

- **agreement** with the other measures,
- **biological plausibility** of the networks it selects,
- **computational efficiency**,
- **sensitivity and robustness**,
- **accuracy**, meaning how well it recovers known model parameters.

This package ships everything needed to put a new measure through the same
benchmark:

- all **25,000 generated networks** of the main sweep, and the networks of both
  parameter-recovery experiments,
- **every benchmark table** behind the paper's numbers,
- the **16 published measures** as plain Python functions,
- **`evaluate()`**, which takes your measure and returns every read-out of the
  paper next to the published measures.

```python
import simba_networks as sb

def my_measure(A, B):
    """Lower = more alike. A: generated network, B: reference."""
    return abs(A.sum(0) - B.sum(0)).sum()

report = sb.evaluate(my_measure)
report.table
```

## Install

```bash
pip install git+https://github.com/ad045/simba
```

Python 3.10 or newer. In a clone of the repository the data bundle is already
there, as `data/`. A pip install has no clone, so the first call that needs data
downloads the same bundle (about 30 MB) into `~/simba_networks_data`.

## Before you run it

Four of the five criteria compare generated networks against the **empirical
consensus connectome** from the Human Connectome Project. The HCP data use terms
do not allow us to redistribute it, so you drop in your own copy. See
[Reference connectome](reference.md). Without it, only the accuracy criterion
runs.

## Citing

If you use the benchmark, please cite the paper; `CITATION.cff` in the
repository holds the machine-readable metadata. The generated networks were
produced with the `generativenetworkmodels` library, and several measures come
from `netrd`.
