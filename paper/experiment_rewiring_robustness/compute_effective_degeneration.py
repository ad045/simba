r"""Effective degeneration of each perturbed reference used in analysis S2.

The rewiring ladder is specified in attempted swaps s; the fraction of edges that
actually MOVED is smaller, because connected_double_edge_swap rejects swaps and
later swaps can relocate an already-relocated edge. The main text plots the
degeneration analyses against this effective degeneration, so the drift figure
uses it too.

Effective degeneration = |{edges in C} \ {edges in C_s^(r)}| / E, i.e. half the
Hamming distance between the two adjacency matrices (edge counts are equal by
construction), reproduced with the same seeds as the drift run.

Writes output/rewiring_robustness/effective_degeneration.csv (s, repeat, eff).

Run:  conda activate ma_thesis && python compute_effective_degeneration.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiment_rewiring_robustness.run_rewiring_robustness import (
    rewire, CONSENSUS_PATH, OUT_DIR, BASE_SEED, E_EDGES,
    DEFAULT_NOISE_LEVELS, R_DEFAULT,
)

OUT_CSV = OUT_DIR / "effective_degeneration.csv"


def effective_frac(consensus, s, r, model="double_edge"):
    seed = (BASE_SEED * 1_000 + s) * 1_000 + r          # same seed as the drift run
    ref = rewire(consensus, n_ops=s, seed=seed, model=model)
    a = (consensus > 0)
    b = (ref > 0)
    moved = int(np.triu(a & ~b, 1).sum())
    return moved / E_EDGES


def main():
    consensus = np.load(CONSENSUS_PATH)[0].astype("float32")
    rows = [{"s": 0, "repeat": r, "eff": 0.0} for r in range(R_DEFAULT)]
    for s in DEFAULT_NOISE_LEVELS:
        for r in range(R_DEFAULT):
            rows.append({"s": s, "repeat": r, "eff": effective_frac(consensus, s, r)})
        print(f"  s={s:4d}  mean effective degeneration "
              f"{np.mean([x['eff'] for x in rows if x['s'] == s]):.4f}")
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")

    # monotone in s, and never above the nominal 2s/E it is derived from
    med = df.groupby("s")["eff"].mean()
    assert med.is_monotonic_increasing, med
    nominal = np.minimum(1.0, 2.0 * med.index.to_numpy() / E_EDGES)
    assert np.all(med.to_numpy() <= nominal + 1e-9), (med.to_numpy(), nominal)


if __name__ == "__main__":
    main()
