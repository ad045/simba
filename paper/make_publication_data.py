"""
Build the data bundle that ``simba_networks`` downloads
=========================================================

Copies the part of the local run tree that backs the manuscript into one
directory and zips it for a GitHub release:

    publication_data/simba-networks-data/        the bundle (layout in its README)
    publication_data/simba-networks-data-<v>.zip what gets uploaded

Nothing computed from subject-level empirical data goes in. The empirical
consensus connectome is not shipped either; users drop in their own copy (see
the package docs, "Reference connectome"). What is shipped from the empirical
side is summary statistics only (``reference/empirical_summary.json``).

Usage
-----
    python make_publication_data.py            # build + zip (run from paper/)
    python make_publication_data.py --check    # verify the bundle against the run tree
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments_config import (CONSENSUS_PATH, DIST_MATRIX_PATH, INDIVIDUALS_PATH,  # noqa: E402
                                MORPHO_DIR, MORPHO_EXP, TIMING_DIR, TIMING_EXP,
                                coarse, fine, gt_dir_name, param_dir_name)

VERSION = "v1"
OUT = ROOT / "publication_data" / "simba-networks-data"
# The same bundle is committed to the repository, where the package reads it
# from a clone; the zip is only for pip installs, as a GitHub release asset.
REPO_DATA = ROOT.parent / "data"
GNM = ROOT / "output" / "gnm"

MEASURES = ["frobenius", "hamming", "jaccard", "f1", "network_mutual_information",
            "dc_network_mutual_information", "spectral_distance_adjacency",
            "spectral_distance_norm_laplacian", "netrd_non_backtracking_spectral",
            "portrait", "net_simile", "delta_con", "communicability_corr",
            "communicability_jsd", "resistance", "energy"]
# the four KS statistics whose maximum is the energy (SI figure)
ENERGY_PARTS = {"test_energy_degree": "energy_degree_ks",
                "test_energy_clustering": "energy_clustering_ks",
                "test_energy_betweenness": "energy_betweenness_ks",
                "test_energy_edge_length": "energy_edge_length_ks"}
META = ["network_index", "filename", "eta", "gamma", "id"]


def pack(stack: np.ndarray) -> np.ndarray:
    """Binary (..., N, N) -> uint8 bit-packed along the last axis."""
    assert np.isin(stack, (0, 1)).all(), "not binary - bit-packing would lose information"
    return np.packbits(stack.astype(bool), axis=-1)


def load_net(path: Path) -> np.ndarray:
    a = np.load(path)
    return a.reshape(a.shape[-2], a.shape[-1])


def value_column(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c not in META]
    assert len(cols) == 1, cols
    return cols[0]


# ---------------------------------------------------------------------------

def landscapes() -> pd.DataFrame:
    """One row per morphospace network, one column per measure (raw values)."""
    out = None
    for m in MEASURES + list(ENERGY_PARTS):
        df = pd.read_csv(MORPHO_DIR / f"summary_indiv_{m}_for_exp_{MORPHO_EXP}.csv")
        df = df.sort_values("network_index").reset_index(drop=True)
        if out is None:
            out = df[["network_index", "eta", "gamma", "id", "filename"]].rename(columns={"id": "replicate"})
        assert (df["filename"].to_numpy() == out["filename"].to_numpy()).all(), m
        out[ENERGY_PARTS.get(m, m)] = df[value_column(df)].astype(float)
    return out


def timings() -> pd.DataFrame:
    """Per-comparison runtime in ms from the reference timing run (TIMING_EXP).

    Some timing files carry several time_* columns from merged re-runs; the
    manuscript (Table S2, Figure 3D) reads the first one, so that is shipped.
    """
    cols = {}
    for m in MEASURES:
        df = pd.read_csv(TIMING_DIR / f"timing_{m}_for_exp_{TIMING_EXP}.csv")
        first = next(c for c in df.columns if c.startswith("time_"))
        cols[m] = df[first].to_numpy() * 1000.0
    return pd.DataFrame(cols)


def degeneration() -> pd.DataFrame:
    """The 200 x 101 progressive-rewiring trajectories (Figure 4 D/F, CV)."""
    src = GNM / "hcp_schaefer_100_dataset" / "chaos_analysis"
    frames = []
    for f in sorted(src.glob("chaos_analysis_*.csv")):
        df = pd.read_csv(f).rename(columns={"metric_value": "value", "computation_time": "time_s"})
        df.insert(0, "measure", f.stem.replace("chaos_analysis_", ""))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def recovery(run: str) -> pd.DataFrame:
    frames = []
    for m in MEASURES:
        df = pd.read_csv(GNM / run / "comparison_results" / f"distances_{m}.csv")
        df = df.drop(columns=[c for c in df.columns if c.startswith("predicted_")])
        df.insert(0, "measure", m)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def recovery_networks(grid_dir: Path, combos, n_members: int, gt_dir: Path,
                      gt_manifest: Path) -> dict:
    grid, members = [], []
    for eta, gamma in combos:
        d = grid_dir / param_dir_name(eta, gamma)
        grid.append(load_net(d / "consensus.npy"))
        member_paths = sorted((d / f"consensus_{n_members}").glob("net_*.npy"))
        assert len(member_paths) == n_members, (d, len(member_paths))
        members.append([load_net(p) for p in member_paths])
    gt = pd.read_csv(gt_manifest).sort_values("gt_idx")
    targets = []
    for i in gt["gt_idx"]:
        paths = sorted((gt_dir / gt_dir_name(int(i))).glob("net_*.npy"))
        assert len(paths) == 1, paths
        targets.append(load_net(paths[0]))
    combos = np.asarray(combos)
    return dict(grid=pack(np.array(grid)), grid_members=pack(np.array(members)),
                grid_eta=combos[:, 0], grid_gamma=combos[:, 1],
                targets=pack(np.array(targets)),
                target_eta=gt["true_eta"].to_numpy(), target_gamma=gt["true_gamma"].to_numpy())


def empirical_summary() -> dict:
    """Group-level statistics of the empirical data - the only empirical content shipped."""
    dist = np.load(DIST_MATRIX_PATH)
    iu = np.triu_indices_from(dist, 1)

    def stats(a):
        e = dist[iu][a[iu] > 0]
        return e.mean(), 100 * (e > 90).mean(), a.sum(0).std()

    ind = np.array([stats(a) for a in np.load(INDIVIDUALS_PATH) > 0])
    cons = stats(np.squeeze(np.load(CONSENSUS_PATH)) > 0)
    summary = lambda x: {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                         "min": float(x.min()), "max": float(x.max())}
    return {
        "note": "100 unrelated HCP S900 subjects, Schaefer-100, binarized at 10% density. "
                "Connection length in mm (Euclidean between parcel centroids).",
        "individuals": {"mean_connection_length_mm": summary(ind[:, 0]),
                        "percent_beyond_90mm": summary(ind[:, 1]),
                        "degree_sd": summary(ind[:, 2])},
        "consensus": {"mean_connection_length_mm": float(cons[0]),
                      "percent_beyond_90mm": float(cons[1]), "degree_sd": float(cons[2]),
                      "n_edges": int((np.squeeze(np.load(CONSENSUS_PATH)) > 0)[iu].sum())},
    }


def copy_csvs(src: Path, dest: Path, skip=("subjects", "d_real")) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.csv")):
        if any(s in f.name for s in skip):      # per-subject tables stay local
            continue
        shutil.copy2(f, dest / f.name)


# ---------------------------------------------------------------------------

def build() -> None:
    if not MORPHO_DIR.exists():
        sys.exit(f"run tree not found at {MORPHO_DIR} (the output/ symlink must point at it)")
    for sub in ("networks", "tables", "reference"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    print("[1/6] landscapes + the 25,000 networks")
    land = landscapes()
    morpho = OUT / "networks" / "morphospace.npz"
    if not morpho.exists():   # reading 25,000 files is the slow step; delete to rebuild
        nets = np.array([load_net(MORPHO_DIR / "generated_networks" / f) for f in land["filename"]])
        np.savez_compressed(morpho, networks=pack(nets), eta=land["eta"].to_numpy(),
                            gamma=land["gamma"].to_numpy(), replicate=land["replicate"].to_numpy())
    land.drop(columns="filename").to_csv(OUT / "tables" / "landscapes.csv.gz", index=False)

    print("[2/6] timings")
    timings().to_csv(OUT / "tables" / "timing_ms.csv.gz", index=False)

    print("[3/6] degeneration trajectories + noise tolerance")
    degeneration().to_csv(OUT / "tables" / "degeneration.csv.gz", index=False)
    rw = ROOT / "output" / "rewiring_robustness"
    pd.read_csv(rw / "drift_curves.csv").to_csv(OUT / "tables" / "noise_tolerance_draws.csv.gz", index=False)
    shutil.copy2(rw / "effective_degeneration.csv", OUT / "tables" / "effective_degeneration.csv")
    shutil.copy2(rw / "rewiring_robustness_results.csv", OUT / "tables" / "noise_tolerance_results.csv")

    print("[4/6] parameter recovery (tables + networks)")
    wide, window = "synthetic_parameter_recovery_wide_uniform", "synthetic_parameter_recovery_fine"
    recovery(wide).to_csv(OUT / "tables" / "recovery_wide.csv.gz", index=False)
    recovery(window).to_csv(OUT / "tables" / "recovery_window.csv.gz", index=False)
    np.savez_compressed(OUT / "networks" / "recovery_wide.npz", **recovery_networks(
        GNM / "synthetic_parameter_recovery_grid" / "grid_consensus", coarse.GRID_COMBOS, 30,
        GNM / wide / "ground_truth", GNM / wide / "ground_truth_params.csv"))
    np.savez_compressed(OUT / "networks" / "recovery_window.npz", **recovery_networks(
        GNM / window / "grid_consensus", fine.GRID_COMBOS, fine.N_CONSENSUS_FINE,
        GNM / window / "ground_truth", GNM / window / "ground_truth_params.csv"))

    print("[5/6] downstream experiments")
    shutil.copy2(ROOT / "output" / "real_vs_artificial" / "real_vs_artificial_results.csv",
                 OUT / "tables" / "real_vs_artificial.csv")
    for name in ("hub_topography", "topographic_plausibility", "structural_gradient"):
        copy_csvs(ROOT / "output" / name, OUT / "tables" / "experiments" / name)

    print("[6/6] reference geometry + empirical summary")
    np.save(OUT / "reference" / "distance_matrix.npy", np.load(DIST_MATRIX_PATH))
    (OUT / "reference" / "empirical_summary.json").write_text(json.dumps(empirical_summary(), indent=2))
    shutil.copy2(ROOT / "data_bundle_README.md", OUT / "README.md")

    if REPO_DATA.exists():
        shutil.rmtree(REPO_DATA)
    shutil.copytree(OUT, REPO_DATA)

    archive = shutil.make_archive(str(OUT.parent / f"simba-networks-data-{VERSION}"), "zip",
                                  root_dir=OUT.parent, base_dir=OUT.name)
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    digest = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    print(f"\n{OUT.relative_to(ROOT)} = {total / 1e6:.0f} MB, copied to data/, zipped -> {Path(archive).name}")
    print(f"sha256 {digest}  <- set DATA_SHA256 in simba_networks/data.py to this, then upload the zip")


def check() -> None:
    """The bundle's networks round-trip against the run tree."""
    z = np.load(OUT / "networks" / "morphospace.npz")
    nets = np.unpackbits(z["networks"], axis=-1, count=z["networks"].shape[1])
    land = pd.read_csv(MORPHO_DIR / f"summary_indiv_frobenius_for_exp_{MORPHO_EXP}.csv").sort_values("network_index")
    for i in np.random.default_rng(0).choice(len(land), 50, replace=False):
        assert np.array_equal(nets[i], load_net(MORPHO_DIR / "generated_networks" / land["filename"].iloc[i]))
    print(f"OK: 50 random networks of {len(nets)} round-trip exactly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    check() if ap.parse_args().check else build()
