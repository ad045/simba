# Verification of the measures

Verified on 2026-10-29 with `scripts/check_against_original_code.py`, against
the evaluator classes in `14_4D_property_space/src/comparing_connectomes`
(byte-identical to `14_4D_benchmarking`, `14_4D_lab_code` and the `main`
branch's `paper/src/comparing_connectomes`) and against the published
landscape values in `data/tables/landscapes.csv.gz`. Environment `ma_thesis`:
Python 3.13.7, numpy 2.2.6, scipy 1.17.0, torch 2.8.0, netrd 0.3.0, and the
local patched `GenerativeNetworkModels` checkout for the gnm library.

The script runs two checks:

1. **Original code.** Each measure is computed with the research pipeline's
   evaluator classes, set up exactly as `run_connectome_comparisons.py`
   dispatches them, and with this package, on the same 30 pairs of generated
   networks. Resistance, NetSimile and the non-backtracking distance go
   through the `netrd` library there, and the KS-based energy through the
   `gnm` library.
2. **Published values.** This package against the shipped published value
   for 30 random networks of the 25,000, compared with the empirical HCP
   consensus (which does not ship).

Largest relative difference over all pairs or networks:

| Measure | vs original code | pairs | vs published values | networks |
|---|---|---|---|---|
| Frobenius | 4.9e-08 | 30 | 8.9e-08 | 30 |
| Hamming | 0.0e+00 | 30 | 0.0e+00 | 30 |
| Jaccard | 0.0e+00 | 30 | 8.0e-16 | 30 |
| F1 | 0.0e+00 | 30 | 4.4e-16 | 30 |
| NMI | 2.0e-14 | 30 | 1.9e-14 | 30 |
| DC-NMI | 3.7e-16 | 30 | 2.5e-15 | 30 |
| Spectral (adjacency) | 7.0e-06 | 30 | 1.5e-06 | 30 |
| Spectral (normalized Laplacian) | 0.0e+00 | 30 | 2.3e-16 | 30 |
| Spectral (non-backtracking) | 1.8e-15 | 30 | 1.3e-02 | 30 |
| Portrait divergence | 0.0e+00 | 30 | 2.3e-16 | 30 |
| NetSimile | 0.0e+00 | 30 | 1.7e-16 | 30 |
| DeltaCon | 0.0e+00 | 30 | 0.0e+00 | 30 |
| Communicability correlation | 4.3e-06 | 30 | 1.5e-05 | 30 |
| Communicability JSD | 1.4e-06 | 30 | 7.9e-07 | 30 |
| Resistance distance | 0.0e+00 | 30 | 1.5e-16 | 30 |
| KS-based energy | 2.4e-07 | 30 | 9.2e-08 | 30 |

Differences up to about 1e-5 come from the original pipeline working in
float32 (torch tensors) where this package works in float64.

## The exception: spectral (non-backtracking)

Against the published values this measure differs by up to 1.3%. The cause is
the eigenvalue solver, not the code. ARPACK (`scipy.sparse.linalg.eigs`)
returns eigenvalues that vary in their last digits from run to run, and when an
eigenvalue sits on the real axis that can move it in or out of the kept half of
the spectrum (imaginary part >= 0). `netrd` itself, run five times on the same
30 pairs, spreads by the same 1.3% and differs from the published values by as
much as this package does. The tests therefore allow this measure a 2%
tolerance (`tests/test_benchmark.py`); every other measure is held to 1e-4.

## Rerunning it

```bash
python scripts/check_against_original_code.py PIPELINE_ROOT CONSENSUS.npy
```

`PIPELINE_ROOT` holds `src/comparing_connectomes`; it needs torch, netrd and
the gnm library. The script prints the table above.
