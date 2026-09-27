# Reference connectome

Every generated network in the benchmark is scored against one empirical
reference: the group consensus connectome of 100 unrelated subjects from the
Human Connectome Project (HCP S900), Schaefer-100 parcellation, deterministic
tractography, binarized at 10% density (495 edges). The HCP data use terms do
not allow us to redistribute it, so the package does not contain it.

## Where to put it

```
~/netdistancebench_data/
    reference/
        consensus.npy          required for agreement, plausibility,
                               efficiency and sensitivity
        individuals.npy        optional: the 100 binarized subjects
        loo_consensuses.npy    optional: the leave-one-out consensuses
```

`consensus.npy` must be a (100, 100) binary, symmetric array without self-loops,
in Schaefer-100 order. Set `NETDISTANCEBENCH_HOME` to use a different folder,
or skip the file and pass the array:

```python
ndb.evaluate(my_measure, reference=my_consensus)
ndb.reference_dir()          # prints where the package looks
```

The two optional files enable the real-vs-artificial read-outs (AUC against the
full GNM population and against the measure's own 500 best networks): subject
`i` of `individuals.npy` is compared with consensus `i` of `loo_consensuses.npy`,
which was built from the other 99 subjects. Both are (100, 100, 100).

## Is it the right one?

`load_reference` recomputes the Frobenius distance of the first 25 generated
networks to your array and compares it with the published values. If they
differ, it warns that the scores will not be comparable to the paper's. A
different reference is legitimate (another cohort, another parcellation with
the same 100 nodes); the published columns of the report just no longer
describe the same experiment.

## Building it from the HCP release

With HCP access, the preprocessing code in the repository rebuilds the
consensus exactly:

| Step | File |
|---|---|
| Connectivity matrices of the 100 subjects, binarized at 10% density, consensus via `netneurotools.networks.struct_consensus` | `paper/src/preprocessing/02_preprocessing_pipeline_hcp_schaefer_100.ipynb` |
| Distance matrix between parcel centroids | `paper/src/preprocessing/get_distance_matrix.py` |
| Thresholding and binarizing | `paper/src/preprocessing/threshold_to_density.py` |

The leave-one-out consensuses come from `build_loo_consensuses` in
`paper/experiment_real_vs_artificial/run_real_vs_artificial.py`.

## What is shipped from the empirical side

Only group-level statistics, in `reference/empirical_summary.json` of the data
bundle (see `ndb.empirical_summary()`): mean connection length, the fraction of
connections beyond 90 mm and the degree standard deviation, as mean, SD and
range over the 100 subjects and for the consensus. They are the reference
values for the plausibility read-outs.
