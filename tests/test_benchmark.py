"""Checks that the package reproduces the published benchmark.

Needs the data bundle (downloaded on first use, or SIMBA_DATA), but
not the empirical reference: the measures are checked on generated-vs-generated
comparisons, which the bundle ships in full.
"""

import numpy as np
import pytest

import simba_networks as sb

# The per-measure read-outs of the paper (eight selected measures), at the precision printed there.
PAPER = {
    # The paper prints portrait 0.38: 0.3748 rounded twice (via 0.375)
    "agreement": dict(frobenius=0.46, delta_con=0.43, netrd_non_backtracking_spectral=0.61,
                      spectral_distance_adjacency=0.57, communicability_corr=0.27, portrait=0.37,
                      net_simile=0.55, energy=0.45),
    "eta_positive_of_100": dict(frobenius=0, delta_con=0, netrd_non_backtracking_spectral=0,
                                spectral_distance_adjacency=1, communicability_corr=0, portrait=0,
                                net_simile=4, energy=18),
    "total_variation": dict(frobenius=0.078, delta_con=0.003, netrd_non_backtracking_spectral=0.049,
                            spectral_distance_adjacency=0.023, communicability_corr=0.109,
                            portrait=0.014, net_simile=0.010, energy=0.004),
    "connection_length_mm": dict(frobenius=35.4, delta_con=39.7, netrd_non_backtracking_spectral=41.3,
                                 spectral_distance_adjacency=40.9, communicability_corr=60.8,
                                 portrait=38.7, net_simile=41.8, energy=47.6),
    "degree_sd": dict(frobenius=3.90, delta_con=3.54, netrd_non_backtracking_spectral=3.65,
                      spectral_distance_adjacency=3.81, communicability_corr=3.79, portrait=3.87,
                      net_simile=4.02, energy=4.55),
    "runtime_ms": dict(frobenius=0.019, delta_con=3.44, netrd_non_backtracking_spectral=67.48,
                       spectral_distance_adjacency=4.97, communicability_corr=0.90, portrait=23.46,
                       net_simile=57.23, energy=71.72),
    "mae": dict(frobenius=0.123, delta_con=0.126, netrd_non_backtracking_spectral=0.129,
                spectral_distance_adjacency=0.161, communicability_corr=0.076, portrait=0.164,
                net_simile=0.139, energy=0.174),
    "cv": dict(frobenius=0.013, delta_con=0.016, netrd_non_backtracking_spectral=0.067,
               spectral_distance_adjacency=0.085, communicability_corr=0.154, portrait=0.055,
               net_simile=0.100, energy=0.042),
    "noise_tolerance": dict(frobenius=0.029, delta_con=0.063, netrd_non_backtracking_spectral=0.015,
                            spectral_distance_adjacency=0.015, communicability_corr=0.264,
                            portrait=0.007, net_simile=0.029, energy=0.007),
    "isnr_db": dict(frobenius=14.49, delta_con=1.79, netrd_non_backtracking_spectral=8.57,
                    spectral_distance_adjacency=8.36, communicability_corr=-1.51, portrait=7.59,
                    net_simile=3.94, energy=15.62),
    "recovery_wide": dict(frobenius=3.71, delta_con=2.74, netrd_non_backtracking_spectral=4.09,
                          spectral_distance_adjacency=3.67, communicability_corr=3.26, portrait=3.77,
                          net_simile=3.44, energy=3.23),
    "recovery_window": dict(frobenius=4.01, delta_con=2.36, netrd_non_backtracking_spectral=6.68,
                            spectral_distance_adjacency=5.26, communicability_corr=3.63, portrait=3.37,
                            net_simile=3.72, energy=6.29),
    # The paper prints 0.47 and 0.27 here: 0.4646 and 0.2645 rounded twice (via 0.465 / 0.265)
    "auc_hard": dict(frobenius=1.00, delta_con=1.00, netrd_non_backtracking_spectral=0.59,
                     spectral_distance_adjacency=0.46, communicability_corr=1.00, portrait=0.26,
                     net_simile=0.10, energy=0.27),
}


@pytest.fixture(scope="module")
def published():
    return sb.published_readouts(sb.SELECTED)


@pytest.mark.parametrize("readout", sorted(PAPER))
def test_published_readouts_match_the_paper(published, readout):
    for measure, value in PAPER[readout].items():
        digits = len(str(value).split(".")[1]) if "." in str(value) else 0
        assert round(published.loc[readout, measure], digits) == pytest.approx(value, abs=1e-9), measure


# ARPACK run-to-run noise (see the measure's docstring)
TOLERANCE = {"netrd_non_backtracking_spectral": 2e-2}


@pytest.mark.parametrize("name", sorted(sb.MEASURES))
def test_measure_reproduces_published_values(name):
    """Recompute a few target-vs-grid distances of the window recovery."""
    rec = sb.load_networks("recovery_window")
    table = sb.load_table("recovery_window")
    table = table[table.measure == name].sort_values("gt_idx")
    grid = rec["grid"].astype(float)
    for t in (0, 50):
        for g in (0, 37, 99):
            ours = sb.MEASURES[name](rec["targets"][t].astype(float), grid[g])
            assert ours == pytest.approx(table[f"dist_to_grid_{g}"].iloc[t], rel=TOLERANCE.get(name, 1e-4),
                                         nan_ok=True)


def test_evaluate_accuracy_reproduces_frobenius(published):
    report = sb.evaluate(sb.measures.frobenius, criteria=["accuracy"], progress=False)
    for key in ("recovery_wide", "recovery_window", "r_eta_wide", "r_eta_window"):
        assert report.table.loc[key, "frobenius"] == pytest.approx(published.loc[key, "frobenius"])
    assert np.isfinite(report.table.loc["recovery_wide", "rank"])
