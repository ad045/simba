# The evaluation datasets

All three are already preprocessed and live under
`/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed/`.

**Primary age axis: `lexis_data_developing` (ages 6–22).**
**Secondary age axis: `lexis_data_aging` (ages 28–100).**
**Taxonomy axis: `suarez_MaMI_dataset` (12 mammalian orders).**

Together the two human datasets give a near-lifespan coverage (6–100 yr) with
a deliberate gap around 22–28 yr where neither cohort has subjects.

## 1 · `lexis_data_developing` — human development *(primary age axis)*

Childhood through young adulthood. Same atlas, same shared distance matrix as
the aging set. **This is the dataset to focus the age side of the loop on.**

| File | Shape | Notes |
|---|---|---|
| `lexis_data_developing/01_connectomes/00_individual_connectomes_bin.npy` | (635, 100, 100) int64 0/1 | 635 binary, undirected connectomes |
| `lexis_data_developing/01_connectomes/00_connectomes_density10.npy` | (1, 100, 100) | group consensus at 10 % density |
| `lexis_data_developing/02_distance_matrices/distance_matrix_100.npy` | (100, 100) | shared Euclidean distance (same as aging set: 0 – 162.7 mm) |
| `lexis_data_developing/04_further_info/00_ages.npy` | (635,) float64 | age in years (6–22) |

Age distribution (years):

```
  N = 635, min = 6, max = 22, 17 unique integer ages

   6– 7.6: ██                            10
   7.6– 9.2: ██████████████              71
   9.2–10.8: ██████████                  50
  10.8–12.4: █████████████████           87
  12.4–14.0: █████████                   45
  14.0–15.6: █████████████████████████   128
  15.6–17.2: ████████████████            81
  17.2–18.8: ██████                      30
  18.8–20.4: █████████████               65
  20.4–22.0: █████████████               68
```

Pre-built per-age consensuses (these are the right targets for the inner loop):

| Folder | A.shape | Ages |
|---|---|---|
| `lexis_data_developing_consensus_per_age_1_year/01_connectomes/00_individual_connectomes_bin.npy` | (17, 100, 100) | 6, 7, 8, …, 22 |
| `lexis_data_developing_consensus_per_age_2_year/01_connectomes/00_individual_connectomes_bin.npy` | (9, 100, 100) | 6, 8, 10, …, 22 |

The 2-year binning keeps every bin at ≥ 75 subjects (median ≈ 100), which is
what we want for a stable consensus per bin.

## 2 · `lexis_data_aging` — human aging *(secondary age axis)*

The "age" half of the question. Single-atlas, single common distance matrix,
binary connectomes per subject, one age per subject.

| File | Shape | Notes |
|---|---|---|
| `lexis_data_aging/01_connectomes/00_individual_connectomes_bin.npy` | (718, 100, 100) int64 0/1 | 718 binary, undirected connectomes |
| `lexis_data_aging/01_connectomes/00_connectomes_density10.npy` | (1, 100, 100) | group-consensus at 10 % density |
| `lexis_data_aging/02_distance_matrices/distance_matrix_100.npy` | (100, 100) | one shared Euclidean distance, 0 – 162.7 mm |
| `lexis_data_aging/04_further_info/00_ages.npy` | (718,) float64 | age in years |

Age distribution (years):

```
  N = 718, min=28, max=100, mean=60.4, median=58.5 (57 unique ages)

  20–30:                                              1
  30–40: ████████████                                 60
  40–50: ███████████████████████████████              157
  50–60: ███████████████████████████████              159
  60–70: ███████████████████████                      119
  70–80: ███████████████████████                      117
  80–90: █████████████████                            89
  90–100: ███                                         16
```

Also pre-binned consensuses available for direct use:
`lexis_data_aging_consensus_per_age_1_year/` and `…_per_age_2_year/` contain
per-age consensus connectomes (and their distance matrices) — these are the
natural targets for an age-conditional generator.

There are sibling folders for finer cuts:
- `lexis_data_developing` / `…_consensus_per_age_{1,2}_year` — under-thirties
- `lexis_data_young` / `…_consensus_per_age_{1,2}_year`
- `lexis_data_all_consensus_per_age_{1,2}_year` — combined

These sub-cohorts were prepared in
`src/preprocessing/10_preprocessing_lexis_data_aging_etc_age_stratified.ipynb`.

## 3 · `suarez_MaMI_dataset` — cross-species mammalian *(taxonomy axis)*

The "taxonomy" half of the question. Per-animal distance matrices (each brain
is in its own native space — there is no shared atlas).

| File | Shape | Notes |
|---|---|---|
| `suarez_MaMI_dataset/01_connectomes/00_connectomes_50_bin.npy` | (225, 100, 100) int64 0/1 | 225 mammals, 50 nodes per hemisphere = 100 nodes |
| `suarez_MaMI_dataset/01_connectomes/00_connectomes_orig.npy` | (225, 200, 200) uint16 | original 200-node parcellation, weighted |
| `suarez_MaMI_dataset/01_connectomes/01_consensus_bin_density_10_percent_50.npy` | (225, 100, 100) | per-animal binarised at 10 % density (named "consensus" but is per-animal) |
| `suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy` | (225, 100, 100) | **per-animal** distance matrix |
| `suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50_scaled_to_schaeffer.npy` | (225, 100, 100) | rescaled so absolute distances are comparable across species |
| `suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed.csv` | 225 rows | taxonomy + brain weight/volume |

Metadata schema (`names_of_animals_with_preprocessed_connectomes_50_processed.csv`):

```
animal, common_name, name, species, genus, sub_family, family,
sub_order, order, super_order, phylogenetic_group,
brain_weight_g, brain_volume_cm3, brain_data_source
```

Taxonomic distribution:

```
Order              N        Super-order         N
Carnivora          51       Laurasiatheria      132
Primates           46       Euarchontoglires    81
Cetartiodactyla    40       No placenta         9
Chiroptera         30       Afrotheria          2
Rodentia           29       Xenarthra           1
Marsupialia        9
Perissodactyla     7        Phylogenetic group  N
Lagomorpha         5        Laurasiatheria      132
Eulipotyphla       4        Euarchontoglires    81
Hyracoidea         2        Other               11
Scandentia         1        Xenarthra           1
Xenarthra          1
```

107 unique species, 50 unique families — *not* every species has multiple
samples. The natural taxonomic-order axis has ~5 well-populated bins
(Carnivora, Primates, Cetartiodactyla, Chiroptera, Rodentia) plus a long
sparse tail.

## Gotchas to know up front

1. **Different N for the same atlas name.** Lexi: 100. Suárez bin: 100 (but
   was 200 originally; the "50" in filenames means 50/hemisphere). HCP
   Schaefer: 100. They're all `N=100` for the binarised matrices, which is
   convenient — but pay attention to the original dimensionality if you ever
   touch raw data.

2. **Per-animal distance matrices on Suárez.** The growth API used by the
   prior multica project assumed a single `D` shared across the evaluation.
   Per-animal `D` means the per-pair distance term itself varies across the
   evaluation set. That's a feature, not a bug — different brain sizes /
   shapes are the *whole point* of the taxonomic comparison — but the code
   needs to thread `D[i]` per evaluated subject.

3. **Binarisation density is 10 %** by convention. Both datasets honour that
   in their `*_bin.npy` or `01_consensus_bin_density_10_percent_*.npy` files.

4. **Sample sizes are unbalanced.** 16 nonagenarians vs. 159 fifty-somethings;
   1 Xenarthran vs. 51 Carnivores. Anything that averages within bins needs
   to weight or stratify carefully.

5. **"Animal" in `run_experiment_lexis_data.py` is misleading naming.** The
   integer `animal_id` actually indexes a row of the *human* lexis dataset.
   For the new code, prefer `subject_id` / `participant_id` to avoid the trap.

6. **The Suárez taxonomy table has missing rows.** `brain_weight_g` and
   `brain_volume_cm3` are blank for some entries (e.g. Capra Nubiana,
   Colobus2). Order-level metadata is complete.

## Quick load snippet

```python
import numpy as np
import pandas as pd

ROOT = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/data/preprocessed"

# --- developing (primary age axis) ---
A_dev = np.load(f"{ROOT}/lexis_data_developing/01_connectomes/00_individual_connectomes_bin.npy")
D_dev = np.load(f"{ROOT}/lexis_data_developing/02_distance_matrices/distance_matrix_100.npy")
ages_dev = np.load(f"{ROOT}/lexis_data_developing/04_further_info/00_ages.npy")
# A_dev: (635, 100, 100), D_dev: (100, 100), ages_dev: (635,) in [6, 22]

# --- developing per-age-bin consensus (inner-loop target) ---
A_dev_bin2y = np.load(f"{ROOT}/lexis_data_developing_consensus_per_age_2_year/01_connectomes/00_individual_connectomes_bin.npy")
# (9, 100, 100), ages 6, 8, 10, ..., 22

# --- aging (secondary age axis) ---
A_age = np.load(f"{ROOT}/lexis_data_aging/01_connectomes/00_individual_connectomes_bin.npy")
D_age = np.load(f"{ROOT}/lexis_data_aging/02_distance_matrices/distance_matrix_100.npy")
ages  = np.load(f"{ROOT}/lexis_data_aging/04_further_info/00_ages.npy")
# A_age: (718, 100, 100), D_age: (100, 100), ages: (718,) in [28, 100]

# --- cross-species (taxonomy axis) ---
A_sp  = np.load(f"{ROOT}/suarez_MaMI_dataset/01_connectomes/00_connectomes_50_bin.npy")
D_sp  = np.load(f"{ROOT}/suarez_MaMI_dataset/02_distance_matrices/distance_matrix_50.npy")
meta  = pd.read_csv(f"{ROOT}/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50_processed.csv")
# A_sp: (225, 100, 100), D_sp: (225, 100, 100), meta: 225 rows with taxonomy
```
