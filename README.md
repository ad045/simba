# netdistancebench

Code and data for *How Similar Are Two Brains? A Comprehensive Benchmark of
Brain Network Similarity Measures* by Adrian Dendorfer, Andrea Luppi, Francesco
Poli, Alexa Mousley, Duncan Astle and Kayson Fakhar.

The paper benchmarks 16 network distance measures for comparing generative
network model (GNM) output with an empirical human connectome, on five criteria:
agreement, biological plausibility, computational efficiency, sensitivity and
robustness, and accuracy. This repository releases all 25,000 generated networks,
every benchmark table and the full pipeline, as a package that scores a new
measure against the same five criteria.

**Documentation: https://ad045.github.io/14_4D_benchmarking/**

## Score your own measure

```bash
pip install git+https://github.com/ad045/14_4D_benchmarking
```

```python
import numpy as np
import netdistancebench as ndb

def degree_l1(A, B):
    """Lower = more alike. A: generated network, B: reference (100 x 100, binary)."""
    return float(np.abs(A.sum(0) - B.sum(0)).sum())

report = ndb.evaluate(degree_l1)
print(report.table)     # every read-out of the paper, next to the eight selected measures
```

The data (about 30 MB) are downloaded on first use. Four of the five criteria
score against the empirical HCP consensus connectome, which we cannot
redistribute: put your copy at `~/netdistancebench_data/reference/consensus.npy`
([details](https://ad045.github.io/14_4D_benchmarking/reference/)). Without it,
only accuracy (parameter recovery) runs.

## Repository layout

```
netdistancebench/    the package: the 16 measures, data access, evaluate()
tests/               checks that the package reproduces the published numbers
docs/                documentation site (mkdocs)
paper/               the research pipeline behind the paper: GNM generation,
                     scoring, every analysis and figure script
```

`paper/` is documented in [Reproducing the paper](https://ad045.github.io/14_4D_benchmarking/paper/).

## Development

```bash
pip install -e ".[test,docs]"
pytest                       # needs the data bundle; no empirical data
mkdocs serve                 # documentation at http://127.0.0.1:8000
```

## License

MIT. Several measures come from [netrd](https://github.com/netsiphd/netrd) and the
network mutual information from [network-MI](https://github.com/hfelippe/network-MI).
