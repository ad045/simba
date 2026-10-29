"""Check the 16 measures of this package against the code that produced the paper.

Two checks, one table:

1. original code: each measure is computed with the evaluator classes of the
   research pipeline, set up exactly as ``run_connectome_comparisons.py``
   dispatches them, and with this package, on the same pairs of generated
   networks (25 morphospace pairs, 5 morphospace-vs-recovery-target pairs).
2. published values: this package against the shipped published landscape
   (``load_table("landscapes")``) for 30 random morphospace networks against
   the empirical HCP consensus.

Needs the research pipeline's ``src`` (with torch, netrd and the gnm library
installed) and the consensus, neither of which ship here::

    python scripts/check_against_original_code.py PIPELINE_ROOT CONSENSUS.npy

PIPELINE_ROOT is the folder that holds ``src/comparing_connectomes`` (e.g. a
checkout of the ``main`` branch's ``paper/`` folder).
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
root, consensus = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path[:0] = [str(root), str(root / "src"), str(Path(__file__).resolve().parent.parent)]

import torch  # noqa: E402
from gnm import evaluation  # noqa: E402
from src.comparing_connectomes.communicability_comparer import CommunicabilityCorrEvaluator  # noqa: E402
from src.comparing_connectomes.communicability_jsd_comparer import CommunicabilityJSDEvaluator  # noqa: E402
from src.comparing_connectomes.delta_con_evaluator import DeltaConEvaluator  # noqa: E402
from src.comparing_connectomes.energy_comparer import EnergyEvaluator  # noqa: E402
from src.comparing_connectomes.f1_comparer import F1Evaluator  # noqa: E402
from src.comparing_connectomes.frobenius_comparer import FrobeniusEvaluator  # noqa: E402
from src.comparing_connectomes.hamming_comparer import HammingEvaluator  # noqa: E402
from src.comparing_connectomes.jaccard_comparer import JaccardEvaluator  # noqa: E402
from src.comparing_connectomes.netrd_comparer import NetrdEvaluator  # noqa: E402
from src.comparing_connectomes.network_mutual_information_comparer import (  # noqa: E402
    DCNetworkMutualInformationEvaluator, NetworkMutualInformationEvaluator)
from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence  # noqa: E402
from src.comparing_connectomes.spectral_distance_comparer import SpectralDistanceEvaluator  # noqa: E402

import simba_networks as sb  # noqa: E402

dist = torch.tensor(sb.distance_matrix(), dtype=torch.float32)
ORIGINAL = {
    "frobenius": FrobeniusEvaluator(), "hamming": HammingEvaluator(), "jaccard": JaccardEvaluator(),
    "f1": F1Evaluator(),
    "network_mutual_information": NetworkMutualInformationEvaluator(),
    "dc_network_mutual_information": DCNetworkMutualInformationEvaluator(),
    "spectral_distance_adjacency": SpectralDistanceEvaluator(method="adjacency"),
    "spectral_distance_norm_laplacian": SpectralDistanceEvaluator(method="normalized_laplacian"),
    "netrd_non_backtracking_spectral": NetrdEvaluator(method="netrd_non_backtracking_spectral"),
    "portrait": PortraitDivergence(), "net_simile": NetrdEvaluator(method="net_simile"),
    "delta_con": DeltaConEvaluator(),
    "communicability_corr": CommunicabilityCorrEvaluator(),
    "communicability_jsd": CommunicabilityJSDEvaluator(),
    "resistance": NetrdEvaluator(method="resistance"),
    # criteria in the order of config_dict in run_connectome_comparisons.py
    "energy": EnergyEvaluator([evaluation.MaxCriteria([evaluation.DegreeKS(), evaluation.ClusteringKS(),
                                                       evaluation.EdgeLengthKS(dist), evaluation.BetweennessKS()])]),
}
assert set(ORIGINAL) == set(sb.MEASURES)


def original(name, a, b):
    out = ORIGINAL[name](torch.tensor(a[None], dtype=torch.float32), torch.tensor(b[None], dtype=torch.float32))
    v = out[0] if 0 in out else next(iter(out.values()))
    return float(v.item() if hasattr(v, "item") else v)


def max_rel(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    assert np.array_equal(np.isnan(xs), np.isnan(ys)), "NaN in one but not the other"
    ok = ~np.isnan(xs)
    return float((np.abs(xs[ok] - ys[ok]) / np.maximum(np.abs(xs[ok]), 1e-12)).max()), int(ok.sum())


def safe(f, *args):
    try:
        return f(*args)
    except Exception:
        return np.nan


nets, _ = sb.load_networks()
targets = sb.load_networks("recovery_wide")["targets"]
rng = np.random.default_rng(0)
pairs = [(nets[i], nets[j]) for i, j in rng.integers(0, len(nets), (25, 2))]
pairs += [(nets[i], targets[j]) for i, j in zip(rng.integers(0, len(nets), 5), rng.integers(0, len(targets), 5))]
pairs = [(a.astype(float), b.astype(float)) for a, b in pairs]

ref = sb.load_reference(consensus)
assert sb.data.matches_published_reference(ref)
published = sb.load_table("landscapes")
idx = np.random.default_rng(1).choice(len(nets), 30, replace=False)

print("| Measure | vs original code | pairs | vs published values | networks |")
print("|---|---|---|---|---|")
for name, f in sb.MEASURES.items():
    r1, n1 = max_rel([safe(original, name, a, b) for a, b in pairs], [safe(f, a, b) for a, b in pairs])
    r2, n2 = max_rel(published[name].to_numpy()[idx], [safe(f, nets[i].astype(float), ref) for i in idx])
    print(f"| {sb.NAMES[name]} | {r1:.1e} | {n1} | {r2:.1e} | {n2} |")
