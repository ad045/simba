"""
Build ``publication_data/`` - the shareable subset of the local run tree.
=========================================================================

The local ``output/`` symlink points into ``../14_4D_lab_code/output`` and holds
~225 GB of run history, most of it scratch. This script copies out the part that
backs the manuscript, in a form small enough to live in the repository:

* the 25,000 GNMs of the main sweep, bit-packed into a single ``.npz``
  (977 MB of ``.npy`` files -> ~25 MB, losslessly: the matrices are binary)
* the same for the two synthetic parameter-recovery sweeps
* every result table (``.csv`` / ``.json``), gzipped
* the figure PDFs the manuscript draws from

Anything derived from the empirical connectomes is written to
``publication_data/empirical_derived/`` instead, which is gitignored - see that
folder's README before publishing any of it.

Usage
-----
    python make_publication_data.py            # build everything
    python make_publication_data.py --check    # verify an existing pack round-trips
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "publication_data"

RUN = ROOT / "output" / "gnm" / "hcp_schaefer_100_dataset" / "105_distance_metrics_mst_animal_0"

# Node-level vectors computed from the empirical consensus connectome. Not
# redistributable without checking the HCP data use terms, so they are staged
# out of the tracked tree. Keys are matched by prefix inside the .npz files.
EMPIRICAL_KEY_PREFIXES = ("ref__", "consensus")
EMPIRICAL_FILES = {
    "real_vs_artificial/loo_consensuses.npy",   # leave-one-out consensus of 100 subjects
    "real_vs_artificial/d_real_loo.csv",        # per-subject distances to the consensus
}


# ---------------------------------------------------------------------------
# GNM packing
# ---------------------------------------------------------------------------

def _params_from_param_dir(name: str) -> tuple[float, float] | tuple[None, None]:
    """Decode ``experiments_config.param_dir_name`` output back to (eta, gamma).

    ``param_etam3p959_gamma0p079`` -> (-3.959, 0.079); 'm' encodes the minus
    sign and 'p' the decimal point.
    """
    if not name.startswith("param_eta") or "_gamma" not in name:
        return None, None
    eta_s, gamma_s = name[len("param_eta"):].split("_gamma", 1)
    try:
        return (float(eta_s.replace("m", "-").replace("p", ".")),
                float(gamma_s.replace("m", "-").replace("p", ".")))
    except ValueError:
        return None, None


def pack_networks(paths: list[Path], dest: Path, root: Path | None = None,
                  extra: dict | None = None) -> None:
    """Bit-pack a list of binary (1, N, N) network .npy files into one .npz.

    Stores ``np.packbits`` of the flattened stack plus the shape needed to undo
    it, and the (eta, gamma, id) triple for each network. Parameters come from
    the filename where the generator wrote them there, and otherwise from the
    enclosing ``param_eta..._gamma...`` directory (the per-grid-point consensus
    networks are simply named ``consensus.npy``).
    """
    from src.utils.extract_params_from_filenames import get_eta_gamma_id_from_filename

    if dest.exists():
        print(f"  -> {dest.relative_to(ROOT)} already built, skipping")
        return

    root = root or dest.parent
    n_nodes = np.load(paths[0]).shape[-1]
    stack = np.empty((len(paths), n_nodes, n_nodes), dtype=bool)

    eta, gamma, net_id = np.full(len(paths), np.nan), np.full(len(paths), np.nan), np.full(len(paths), -1, dtype=np.int32)
    for i, p in enumerate(paths):
        a = np.load(p).reshape(n_nodes, n_nodes)
        if not np.isin(a, (0, 1)).all():
            raise ValueError(f"{p} is not binary - bit-packing would lose information")
        stack[i] = a.astype(bool)
        e, g, k = get_eta_gamma_id_from_filename(p.name)
        if e is None:
            e, g = _params_from_param_dir(p.parent.name)
        if e is not None:
            eta[i], gamma[i] = e, g
        if k is not None:
            net_id[i] = k
        if i % 2500 == 0:
            print(f"    {i:>6}/{len(paths)}", flush=True)

    n_unparsed = int(np.isnan(eta).sum())
    if n_unparsed:
        print(f"  note: {n_unparsed} networks have no (eta, gamma) in their path - see 'paths'")

    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        packed=np.packbits(stack.reshape(-1)),
        shape=np.array(stack.shape),
        eta=eta,
        gamma=gamma,
        id=net_id,
        filenames=np.array([p.name for p in paths]),
        paths=np.array([str(p.relative_to(root)) for p in paths]),
        **(extra or {}),
    )
    print(f"  -> {dest.relative_to(ROOT)}  ({dest.stat().st_size / 1e6:.1f} MB, {len(paths)} networks)")


def load_networks(npz_path: Path) -> tuple[np.ndarray, dict]:
    """Inverse of :func:`pack_networks`. Returns (stack, metadata)."""
    z = np.load(npz_path, allow_pickle=False)
    shape = tuple(z["shape"])
    n = int(np.prod(shape))
    stack = np.unpackbits(z["packed"])[:n].reshape(shape).astype(np.float32)
    meta = {k: z[k] for k in z.files if k not in ("packed", "shape")}
    return stack, meta


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------

def copy_tables(src_dir: Path, dest_dir: Path, patterns=("*.csv", "*.json")) -> int:
    """Gzip every table under src_dir into dest_dir (flat). Returns file count."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pattern in patterns:
        for f in sorted(src_dir.glob(pattern)):
            # the two 41 GB intermediates are regenerable from the summary files
            if f.stat().st_size > 500_000_000:
                print(f"  skip (too large, regenerable): {f.name}")
                continue
            with open(f, "rb") as fin, gzip.open(dest_dir / (f.name + ".gz"), "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout)
            n += 1
    return n


def copy_figures(src_dir: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob("*.pdf")):
        shutil.copy2(f, dest_dir / f.name)
        n += 1
    return n


def split_npz(src: Path, public_dest: Path, private_dest: Path) -> None:
    """Copy an .npz, routing empirical-derived arrays to the private tree."""
    z = np.load(src)
    public = {k: z[k] for k in z.files if not k.startswith(EMPIRICAL_KEY_PREFIXES)}
    private = {k: z[k] for k in z.files if k.startswith(EMPIRICAL_KEY_PREFIXES)}
    if public:
        public_dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(public_dest, **public)
    if private:
        private_dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(private_dest, **private)
        print(f"  empirical-derived arrays -> {private_dest.relative_to(ROOT)}: {sorted(private)}")


# ---------------------------------------------------------------------------

def build() -> None:
    if not RUN.exists():
        sys.exit(f"main sweep not found at {RUN}\n(the output/ symlink must point at the run tree)")

    print("[1/5] main sweep: 25,000 GNMs")
    nets = sorted((RUN / "generated_networks").glob("*.npy"))
    pack_networks(nets, OUT / "gnms" / "gnms_hcp_25000.npz", root=RUN / "generated_networks")

    print("[2/5] main sweep: result tables")
    n = copy_tables(RUN, OUT / "benchmark_results")
    print(f"  -> {n} tables")

    print("[3/5] main sweep: figures and the chaos/degeneration tables")
    # chaos_analysis backs figure 3 panels D and F (run_degeneration_effective_panels.py)
    chaos = RUN.parent / "chaos_analysis"
    if chaos.exists():
        copy_tables(chaos, OUT / "benchmark_results" / "chaos_analysis")
        copy_figures(chaos, OUT / "figures" / "chaos_analysis")
        print(f"  chaos_analysis: copied")
    for sub in ("figures_manuscript", "figures_manuscript_apdx", "method_evaluation",
                "00_combined_figures_for_manuscript", "landscape_and_six_example_gen_connectomes"):
        if (RUN / sub).exists():
            k = copy_figures(RUN / sub, OUT / "figures" / sub)
            copy_tables(RUN / sub, OUT / "benchmark_results" / sub)
            print(f"  {sub}: {k} figures")

    print("[4/5] synthetic parameter recovery")
    for name, dest in [("synthetic_parameter_recovery_grid", "gnms_recovery_wide.npz"),
                       ("synthetic_parameter_recovery_fine", "gnms_recovery_window.npz")]:
        src = ROOT / "output" / "gnm" / name
        if not src.exists():
            print(f"  skip (missing): {name}")
            continue
        nets = sorted(src.rglob("*.npy"))
        pack_networks(nets, OUT / "gnms" / dest, root=src)
        for sub in ("comparison_results", "plots", "."):
            if (src / sub).exists():
                copy_tables(src / sub, OUT / "parameter_recovery" / name)
        copy_figures(src / "plots", OUT / "figures" / name) if (src / "plots").exists() else None

    print("[5/5] downstream experiments")
    for name in ("rewiring_robustness", "hub_topography", "topographic_plausibility",
                 "structural_gradient", "real_vs_artificial"):
        src = ROOT / "output" / name
        if not src.exists():
            print(f"  skip (missing): {name}")
            continue
        dest = OUT / "experiment_results" / name
        for f in sorted(src.glob("*.csv")) + sorted(src.glob("*.json")):
            if f"{name}/{f.name}" in EMPIRICAL_FILES:
                d = OUT / "empirical_derived" / name / (f.name + ".gz")
                d.parent.mkdir(parents=True, exist_ok=True)
                with open(f, "rb") as fin, gzip.open(d, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                print(f"  empirical-derived -> {d.relative_to(ROOT)}")
                continue
            dest.mkdir(parents=True, exist_ok=True)
            with open(f, "rb") as fin, gzip.open(dest / (f.name + ".gz"), "wb") as fout:
                shutil.copyfileobj(fin, fout)
        copy_figures(src, OUT / "figures" / name)
        for f in sorted(src.glob("*.npz")):
            split_npz(f, dest / f.name, OUT / "empirical_derived" / name / f.name)
        # rewiring's gnm_stack.npy is the main sweep restacked - already packed above
        for f in sorted(src.glob("*.npy")):
            if f"{name}/{f.name}" in EMPIRICAL_FILES:
                d = OUT / "empirical_derived" / name / f.name
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d)
                print(f"  empirical-derived -> {d.relative_to(ROOT)}")
        print(f"  {name}: done")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\npublication_data/ = {total / 1e6:.0f} MB")


def check() -> None:
    """Round-trip the main GNM pack against the original .npy files."""
    pack = OUT / "gnms" / "gnms_hcp_25000.npz"
    stack, meta = load_networks(pack)
    nets = sorted((RUN / "generated_networks").glob("*.npy"))
    assert stack.shape[0] == len(nets) == 25000, (stack.shape, len(nets))
    rng = np.random.default_rng(0)
    for i in rng.choice(len(nets), 50, replace=False):
        original = np.load(nets[i]).reshape(stack.shape[1], stack.shape[2])
        assert np.array_equal(stack[i], original), f"mismatch at {nets[i].name}"
        assert meta["filenames"][i] == nets[i].name
    print(f"OK: {pack.name} round-trips exactly (50 random networks checked, 25,000 packed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify an existing pack instead of building")
    args = ap.parse_args()
    check() if args.check else build()
