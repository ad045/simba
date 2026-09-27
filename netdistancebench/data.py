"""Access to the benchmark data: the 25,000 generated networks, the recovery
networks, every result table, and the empirical reference you drop in.

Everything lives under one folder, ``~/netdistancebench_data`` by default (set
``NETDISTANCEBENCH_HOME`` to move it)::

    ~/netdistancebench_data/
        netdistancebench-data/     downloaded on first use (about 30 MB)
        reference/
            consensus.npy          <- you put the empirical consensus here
            individuals.npy        <- optional, for the real-vs-artificial AUCs
            loo_consensuses.npy    <- optional, ditto

The empirical connectomes are not distributed (HCP data use terms). See
:func:`load_reference` and the documentation page "Reference connectome".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_VERSION = "v1"
DATA_URL = ("https://github.com/ad045/14_4D_benchmarking/releases/download/"
            f"data-{DATA_VERSION}/netdistancebench-data-{DATA_VERSION}.zip")
DATA_SHA256 = "0024c11f6de6b997bc17ca6893d41cd514c0ad51f9c44c5fc185d25548a47e95"

N_NODES = 100
N_EDGES = 495      # 10% density


def home() -> Path:
    """Root folder for the downloaded bundle and the dropped-in reference."""
    return Path(os.environ.get("NETDISTANCEBENCH_HOME", Path.home() / "netdistancebench_data")).expanduser()


def data_dir() -> Path:
    """Path of the unpacked data bundle; downloads it on first call.

    Point ``NETDISTANCEBENCH_DATA`` at an unpacked bundle to skip the download.
    """
    if "NETDISTANCEBENCH_DATA" in os.environ:
        return Path(os.environ["NETDISTANCEBENCH_DATA"]).expanduser()
    target = home() / "netdistancebench-data"
    if not (target / "networks" / "morphospace.npz").exists():
        _download(target)
    return target


def _download(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / Path(DATA_URL).name
    print(f"Downloading the benchmark data (about 30 MB) to {target.parent} ...")
    with urllib.request.urlopen(DATA_URL) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out)
    if DATA_SHA256:
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != DATA_SHA256:
            archive.unlink()
            raise OSError(f"checksum mismatch for {archive.name}: got {digest}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(target.parent)
    archive.unlink()


def _unpack(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, count=N_NODES).astype(bool)


# ---------------------------------------------------------------------------
# Networks and tables
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_networks(which: str = "morphospace") -> tuple[np.ndarray, pd.DataFrame] | dict:
    """Load a set of generated networks as boolean arrays.

    Args:
        which: ``"morphospace"`` (the 25,000 networks of the main sweep),
            ``"recovery_wide"`` or ``"recovery_window"``.

    Returns:
        For ``"morphospace"``, ``(networks, meta)``: ``networks`` of shape (25000, 100, 100) and ``meta`` a DataFrame of ``eta``, ``gamma``, ``replicate`` in the row order of ``load_table("landscapes")``. For the recovery sets, a dict with ``grid`` (100 grid-point consensuses, gamma-outer / eta-inner), ``grid_members``, ``targets`` (100 test networks) and ``grid_eta``, ``grid_gamma``, ``target_eta``, ``target_gamma``.
    """
    z = np.load(data_dir() / "networks" / f"{which}.npz")
    if which == "morphospace":
        meta = pd.DataFrame({k: z[k] for k in ("eta", "gamma", "replicate")})
        return _unpack(z["networks"]), meta
    return {k: (_unpack(z[k]) if k in ("grid", "grid_members", "targets") else z[k]) for k in z.files}


@lru_cache(maxsize=None)
def load_table(name: str) -> pd.DataFrame:
    """Load one of the shipped result tables by name, e.g. ``"landscapes"``,
    ``"timing_ms"``, ``"degeneration"``, ``"recovery_wide"``,
    ``"noise_tolerance_draws"``, ``"real_vs_artificial"`` or
    ``"experiments/hub_topography/hub_topography_results"``."""
    base = data_dir() / "tables"
    for suffix in (".csv.gz", ".csv"):
        if (base / f"{name}{suffix}").exists():
            return pd.read_csv(base / f"{name}{suffix}")
    raise FileNotFoundError(f"no table '{name}' in {base}")


@lru_cache(maxsize=None)
def distance_matrix() -> np.ndarray:
    """Euclidean distance (mm) between the Schaefer-100 parcel centroids."""
    return np.load(data_dir() / "reference" / "distance_matrix.npy")


@lru_cache(maxsize=None)
def empirical_summary() -> dict:
    """Group-level statistics of the empirical connectomes (no subject data)."""
    return json.loads((data_dir() / "reference" / "empirical_summary.json").read_text())


# ---------------------------------------------------------------------------
# The empirical reference (drop-in)
# ---------------------------------------------------------------------------

def reference_dir() -> Path:
    """Folder where :func:`load_reference` looks for ``consensus.npy``."""
    return home() / "reference"


_MISSING = """No reference connectome found.

The benchmark scores a measure against the empirical HCP consensus connectome,
which we are not allowed to redistribute. Put your copy here:

    {path}

as a (100, 100) binary, symmetric .npy array (Schaefer-100 order, 10% density,
495 edges), or pass it directly: evaluate(measure, reference=my_array).
How to build it from the HCP release: see "Reference connectome" in the docs.

Without a reference only the accuracy criterion (parameter recovery) can run:
evaluate(measure, criteria=["accuracy"])."""


def load_reference(reference: np.ndarray | str | Path | None = None) -> np.ndarray:
    """Return the empirical reference connectome as a float (100, 100) array.

    Args:
        reference: an array, a path to a ``.npy`` file, or ``None`` to read
            ``reference_dir() / "consensus.npy"``.

    Warns if the array is not the consensus the published numbers were computed
    against, so that a mismatch cannot pass silently.
    """
    if reference is None:
        path = reference_dir() / "consensus.npy"
        if not path.exists():
            raise FileNotFoundError(_MISSING.format(path=path))
        reference = path
    ref = np.squeeze(np.load(reference) if isinstance(reference, (str, Path)) else np.asarray(reference))
    ref = ref.astype(float)
    if ref.shape != (N_NODES, N_NODES):
        raise ValueError(f"reference must be ({N_NODES}, {N_NODES}), got {ref.shape}")
    if not (np.isin(ref, (0, 1)).all() and np.array_equal(ref, ref.T) and not ref.diagonal().any()):
        raise ValueError("reference must be binary, symmetric and without self-loops")
    if not matches_published_reference(ref):
        warnings.warn("This reference is not the consensus the paper used: scores will not be "
                      "comparable to the published ones.", stacklevel=2)
    return ref


def matches_published_reference(ref: np.ndarray, n: int = 25) -> bool:
    """True if ``ref`` reproduces the published Frobenius distances exactly."""
    nets, _ = load_networks("morphospace")
    published = load_table("landscapes")["frobenius"].to_numpy()[:n]
    ours = np.array([np.linalg.norm(nets[i].astype(float) - ref) for i in range(n)])
    return bool(np.allclose(ours, published, rtol=1e-5))


def load_optional_reference(name: str):
    """``individuals.npy`` / ``loo_consensuses.npy`` from :func:`reference_dir`, or None."""
    path = reference_dir() / f"{name}.npy"
    return np.load(path).astype(float) if path.exists() else None
