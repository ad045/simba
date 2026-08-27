"""
Rebuild the HCP consensus at 12% density (594 edges), to match the edge count of
the generated networks in the morphospace (which carry the 99-edge MST seed on
top of 495 model-added edges).

Same pipeline as src/preprocessing/02_preprocessing_pipeline_hcp_schaefer_100.ipynb,
only `retain` changes. With retain=10 it reproduces the stored consensus exactly.

    conda activate ma_thesis
    python density_fix/build_consensus_12.py            # writes + checks
    python density_fix/build_consensus_12.py --selftest  # checks only
"""
import sys
from pathlib import Path

import numpy as np
import h5py
from netneurotools.networks import struct_consensus, binarize_network

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments_config import RAW_COORDS_PATH, DIST_MATRIX_PATH, CONSENSUS_PATH

RAW_MAT = ROOT / "data" / "raw" / "hcp_schaefer_100" / "DTI_fibers_VolNorm_HCP.mat"
OUT = CONSENSUS_PATH.parent / "01_consensus_bin_density_12_percent_100.npy"


def weighted_consensus() -> np.ndarray:
    with h5py.File(RAW_MAT) as f:
        refs = f["DTI_fibers_VolNorm_HCP"][0]
        C = np.stack([f[refs[s]][()] for s in range(len(refs))], axis=-1)
    C = C.T                                     # (subjects, 100, 100)
    D = np.load(DIST_MATRIX_PATH)
    hemiid = (np.loadtxt(RAW_COORDS_PATH)[:, 0] > 0).astype(int).reshape(-1, 1)
    return struct_consensus(C.T, D, hemiid=hemiid, weighted=True)


def binarised(cons_w: np.ndarray, retain: int) -> np.ndarray:
    b = binarize_network(cons_w, retain=retain).reshape(100, 100)
    b = np.maximum(b, b.T)
    np.fill_diagonal(b, 0.0)
    return b


def main() -> None:
    cons_w = weighted_consensus()
    c10, c12 = binarised(cons_w, 10), binarised(cons_w, 12)
    stored = np.load(CONSENSUS_PATH)[0]

    assert np.array_equal(c10, stored), "retain=10 no longer reproduces the stored consensus"
    assert c12.sum() / 2 == 594, c12.sum() / 2
    assert ((c12 - c10) >= 0).all(), "12% is not a superset of 10%"
    print(f"ok: 10% -> {c10.sum()/2:.0f} edges (matches stored), "
          f"12% -> {c12.sum()/2:.0f} edges, superset")

    if "--selftest" not in sys.argv:
        np.save(OUT, c12.reshape(1, 100, 100))
        print("wrote", OUT)


if __name__ == "__main__":
    main()
