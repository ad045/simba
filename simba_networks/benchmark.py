"""Score a network distance measure on the benchmark's five criteria.

:func:`evaluate` runs a new measure through the same experiments as the paper
and returns a :class:`Report` that places it next to the published measures.
:func:`published_readouts` computes the published side from the shipped tables
with the same code, so the two columns cannot drift apart.
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

from . import data
from .measures import MEASURES, NAMES, SELECTED, SIMILARITIES

# ---------------------------------------------------------------------------
# Protocol constants (Methods of the paper)
# ---------------------------------------------------------------------------

ETA_RANGE, GAMMA_RANGE = (-8.0, 3.0), (-0.1, 1.0)
N_GRID = 50                           # morphospace grid, per axis
N_BEST_COMBINATIONS = 20              # "20 best-fitting parameter combinations"
N_BEST_NETWORKS = 20                  # "20 best-fitting networks"
LONG_RANGE_MM = 90.0
N_HARD = 500                          # real vs artificial: own 500 best GNMs

# degeneration: 200 trajectories x 101 steps of degree-preserving rewiring
DEGENERATION_TRAJECTORIES, DEGENERATION_STEPS, DEGENERATION_SEED = 200, 100, 42

# noise tolerance: rewire the reference by s swaps, re-find the best fit
NOISE_LEVELS = (1, 2, 5, 10, 20, 50, 100, 200)
NOISE_SEED = 20260612
NOISE_REPEATS_FAST, NOISE_REPEATS_SLOW = 20, 10
SLOW_MS = 10.0                        # above this, recover on a window around the best fit
NOISE_WINDOW, NOISE_FALLBACK_CUTOFF, DRIFT_THRESHOLD = 10, 50, 1.0

# chance level of the grid-step recovery error (Methods, "Chance level")
CHANCE = {"wide": 4.96, "window": 4.53}

CRITERIA = ("agreement", "plausibility", "efficiency", "sensitivity", "accuracy")

#: Every read-out: (key, criterion, label, desirable direction: +1 up, -1 down, 0 none)
READOUTS = [
    ("agreement", "agreement", "Mean r with the selected landscapes", +1),
    ("max_abs_r", "agreement", "Largest |r| with a published landscape", 0),
    ("best_eta", "plausibility", "Best-fitting eta", 0),
    ("best_gamma", "plausibility", "Best-fitting gamma", 0),
    ("eta_positive_of_100", "plausibility", "Combinations at eta > 0, of the 100 best", -1),
    ("eta_positive_of_20", "plausibility", "Combinations at eta > 0, of the 20 best", -1),
    ("total_variation", "plausibility", "Total variation of the 20 best combinations", -1),
    ("connection_length_mm", "plausibility", "Mean connection length of the 20 best networks, mm", 0),
    ("percent_beyond_90mm", "plausibility", "Connections beyond 90 mm, %", 0),
    ("degree_sd", "plausibility", "Degree SD of the 20 best networks", 0),
    ("runtime_ms", "efficiency", "Runtime per comparison, ms", -1),
    ("mae", "sensitivity", "MAE against the linear response", -1),
    ("cv", "sensitivity", "CV across the 200 rewiring trajectories", -1),
    ("noise_tolerance", "sensitivity", "Noise tolerance N*", +1),
    ("isnr_db", "sensitivity", "Intrinsic signal-to-noise ratio, dB", +1),
    ("recovery_wide", "accuracy", "Recovery error, whole morphospace, grid steps", -1),
    ("recovery_window", "accuracy", "Recovery error, plausible window, grid steps", -1),
    ("r_eta_wide", "accuracy", "r(true eta, recovered eta), whole morphospace", +1),
    ("r_eta_window", "accuracy", "r(true eta, recovered eta), plausible window", +1),
    ("auc", "accuracy", "AUC real vs the full GNM population", +1),
    ("auc_hard", "accuracy", "AUC real vs the own 500 best networks", +1),
]


def _orient(values, similarity: bool) -> np.ndarray:
    """Lower = more alike, for every measure (similarities are negated)."""
    v = np.asarray(values, dtype=float)
    return -v if similarity else v


# ---------------------------------------------------------------------------
# Read-outs - each takes raw measure output and is shared by both sides
# ---------------------------------------------------------------------------

def _grid_index(meta: pd.DataFrame):
    i_eta = pd.Index(np.sort(meta["eta"].unique())).get_indexer(meta["eta"])
    i_gamma = pd.Index(np.sort(meta["gamma"].unique())).get_indexer(meta["gamma"])
    return i_eta, i_gamma


def landscape_readouts(raw, similarity: bool, meta: pd.DataFrame, networks: np.ndarray,
                       published: pd.DataFrame, exclude: str | None = None) -> dict:
    """Plausibility, agreement and iSNR from one value per morphospace network.

    ``published``: the oriented landscapes of the published measures (one
    column each); ``exclude`` names the measure itself when it is one of them.
    """
    d = _orient(raw, similarity)
    cell = pd.Series(d).groupby([meta["eta"], meta["gamma"]]).mean()
    best = cell.idxmin()
    top100, top20 = cell.nsmallest(100), cell.nsmallest(N_BEST_COMBINATIONS)
    e = (top20.index.get_level_values(0) - ETA_RANGE[0]) / (ETA_RANGE[1] - ETA_RANGE[0])
    g = (top20.index.get_level_values(1) - GAMMA_RANGE[0]) / (GAMMA_RANGE[1] - GAMMA_RANGE[0])

    dist = data.distance_matrix()
    iu = np.triu_indices_from(dist, 1)
    best_nets = networks[np.argsort(d, kind="stable")[:N_BEST_NETWORKS]]
    lengths = np.concatenate([dist[iu][a[iu]] for a in best_nets])

    raw_cells = pd.Series(np.asarray(raw, float)).groupby([meta["eta"], meta["gamma"]])
    isnr = 10 * np.log10(np.var(raw_cells.mean().to_numpy()) / np.nanmean(raw_cells.std().to_numpy() ** 2))

    others = published.drop(columns=exclude, errors="ignore").corrwith(pd.Series(d))
    return {
        "agreement": float(others[others.index.isin(SELECTED)].mean()),
        "max_abs_r": float(others.abs().max()),
        "best_eta": float(best[0]), "best_gamma": float(best[1]),
        "eta_positive_of_100": int((top100.index.get_level_values(0) > 0).sum()),
        "eta_positive_of_20": int((top20.index.get_level_values(0) > 0).sum()),
        "total_variation": float(np.var(e) + np.var(g)),
        "connection_length_mm": float(lengths.mean()),
        "percent_beyond_90mm": float(100 * (lengths > LONG_RANGE_MM).mean()),
        "degree_sd": float(np.mean([a.sum(0).std() for a in best_nets])),
        "isnr_db": float(isnr),
    }


def degeneration_readouts(values: pd.DataFrame, hamming: pd.DataFrame, similarity: bool) -> dict:
    """MAE and CV from the rewiring trajectories.

    ``values`` / ``hamming``: columns process_id, step, rewire_fraction, value.
    """
    def per_trajectory(df, col):
        return df.groupby("process_id")[col].transform(lambda s: (s - s.min()) / (s.max() - s.min()))

    h = hamming.assign(x=per_trajectory(hamming, "value"))
    x = h.groupby("step")["x"].mean()
    v = values["value"]
    v = (v - v.min()) / (v.max() - v.min())
    df = values.assign(v=1 - v if similarity else v)
    y = df.assign(y=per_trajectory(df, "v")).groupby("step")["y"].mean()
    g = df.groupby("rewire_fraction")["v"]
    return {"mae": float(np.abs(y - x).mean()), "cv": float((g.std() / g.mean()).mean())}


def noise_tolerance(draws: pd.DataFrame, effective: dict) -> float:
    """N*: effective degeneration at the largest swap count whose median drift
    of the best-fitting combination stays within one grid step."""
    med = draws.groupby("s")["drift"].median()
    ok = [0] + [int(s) for s, m in med.items() if m <= DRIFT_THRESHOLD]
    s_star = max(ok)
    return float(effective[s_star]) if s_star else 0.0


def recovery_readouts(dists: np.ndarray, similarity: bool, rec: dict) -> tuple[float, float]:
    """Grid-step error and r(eta) of argmin recovery. ``dists``: (targets, grid)."""
    eta_axis, gamma_axis = np.unique(rec["grid_eta"]), np.unique(rec["grid_gamma"])
    d = _orient(dists, similarity)
    valid = ~np.all(np.isnan(d), axis=1)
    pred = np.array([np.nanargmin(row) for row in d[valid]])
    pe, pg = pred % len(eta_axis), pred // len(eta_axis)
    te = np.abs(rec["target_eta"][valid, None] - eta_axis).argmin(1)
    tg = np.abs(rec["target_gamma"][valid, None] - gamma_axis).argmin(1)
    error = float(np.mean(np.hypot(pe - te, pg - tg)))
    recovered = rec["grid_eta"][pred]
    r = float(np.corrcoef(rec["target_eta"][valid], recovered)[0, 1]) if recovered.std() > 0 else float("nan")
    return error, r


def discrimination(d_real, d_art, similarity: bool) -> dict:
    """AUC of real subjects against GNMs, full population and the hard case."""
    s_real, s_art = -_orient(d_real, similarity), -_orient(d_art, similarity)
    s_real, s_art = s_real[np.isfinite(s_real)], s_art[np.isfinite(s_art)]
    auc = mannwhitneyu(s_real, s_art).statistic / (len(s_real) * len(s_art))
    hard = np.sort(s_art)[::-1][:N_HARD]
    auc_hard = mannwhitneyu(s_real, hard).statistic / (len(s_real) * len(hard))
    return {"auc": float(auc), "auc_hard": float(auc_hard)}


# ---------------------------------------------------------------------------
# The published side
# ---------------------------------------------------------------------------

def _published_landscapes() -> pd.DataFrame:
    land = data.load_table("landscapes")
    return pd.DataFrame({m: _orient(land[m], m in SIMILARITIES) for m in MEASURES})


def published_readouts(measures=MEASURES) -> pd.DataFrame:
    """Every read-out of the published measures, recomputed from the shipped tables.

    Returns a DataFrame indexed by read-out key with one column per measure.
    Read-outs a measure was not run through (e.g. the degeneration experiments
    for the eight measures not selected) are NaN.
    """
    nets, meta = data.load_networks("morphospace")
    land, timing = data.load_table("landscapes"), data.load_table("timing_ms")
    landscapes = _published_landscapes()
    degen = data.load_table("degeneration")
    draws = data.load_table("noise_tolerance_draws")
    effective = data.load_table("effective_degeneration").groupby("s")["eff"].mean().to_dict()
    rva = data.load_table("real_vs_artificial").set_index("measure")
    rec = {k: (data.load_networks(f"recovery_{k}"), data.load_table(f"recovery_{k}")) for k in ("wide", "window")}

    out = {}
    for m in measures:
        sim = m in SIMILARITIES
        row = landscape_readouts(land[m], sim, meta, nets, landscapes, exclude=m)
        row["runtime_ms"] = float(timing[m].mean())
        if m in set(degen["measure"]):
            row.update(degeneration_readouts(degen[degen.measure == m], degen[degen.measure == "hamming"], sim))
        if m in set(draws["measure"]):
            row["noise_tolerance"] = noise_tolerance(draws[draws.measure == m], effective)
        for k, (networks, table) in rec.items():
            t = table[table.measure == m].sort_values("gt_idx")
            dists = t[[f"dist_to_grid_{i}" for i in range(len(networks["grid"]))]].to_numpy()
            row[f"recovery_{k}"], row[f"r_eta_{k}"] = recovery_readouts(dists, sim, networks)
        if m in rva.index:
            row["auc"], row["auc_hard"] = rva.loc[m, "auc"], rva.loc[m, "auc_hardcase"]
        out[m] = row
    keys = [k for k, *_ in READOUTS]
    return pd.DataFrame(out).reindex(keys)


# ---------------------------------------------------------------------------
# Running a new measure
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(measure, stack):
    _WORKER["measure"], _WORKER["stack"] = measure, stack


def _score(pairs):
    """[(A or stack index, B), ...] -> (values, ms per call). Networks travel as
    bool arrays (8x smaller to pickle) and reach the measure as float."""
    f, stack = _WORKER["measure"], _WORKER["stack"]
    vals, secs = np.empty(len(pairs)), np.empty(len(pairs))
    for k, (a, b) in enumerate(pairs):
        a = (stack[a] if isinstance(a, (int, np.integer)) else a).astype(float)
        b = np.asarray(b, dtype=float)
        t0 = time.perf_counter()
        try:
            vals[k] = f(a, b)
        except Exception:
            vals[k] = np.nan
        secs[k] = time.perf_counter() - t0
    return vals, secs


class _Runner:
    """Evaluates the measure on lists of pairs, in-process or on a pool."""

    def __init__(self, measure, stack, n_jobs, progress):
        self.progress = progress
        if n_jobs == 1:
            _init_worker(measure, stack)
            self.pool = None
        else:
            self.pool = ProcessPoolExecutor(n_jobs if n_jobs > 0 else None,
                                            initializer=_init_worker, initargs=(measure, stack))

    def __call__(self, pairs, label=None, chunk=250):
        chunks = [pairs[i:i + chunk] for i in range(0, len(pairs), chunk)]
        results = map(_score, chunks) if self.pool is None else self.pool.map(_score, chunks)
        if self.progress and label:
            from tqdm.auto import tqdm
            results = tqdm(results, total=len(chunks), desc=label)
        vals, secs = zip(*results) if chunks else ((), ())
        return np.concatenate(vals), np.concatenate(secs) * 1000.0

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def _degeneration_trajectories(reference):
    """The 200 x 101 rewiring trajectories, seeded as in the paper."""
    G0 = nx.from_numpy_array(reference)
    n_edges = G0.number_of_edges()
    rows, nets = [], []
    for pid in range(DEGENERATION_TRAJECTORIES):
        for step in range(DEGENERATION_STEPS + 1):
            frac = step / DEGENERATION_STEPS
            G = G0.copy()
            if step:
                n_swaps = int(n_edges * frac)
                try:
                    nx.double_edge_swap(G, nswap=n_swaps, max_tries=n_swaps * 100,
                                        seed=DEGENERATION_SEED + pid * 10000 + step)
                except nx.NetworkXError:
                    pass
            a = nx.to_numpy_array(G, nodelist=range(len(reference))) > 0
            nets.append(a)
            rows.append((pid, step, frac, np.sum(a != (reference > 0)) / 2))
    meta = pd.DataFrame(rows, columns=["process_id", "step", "rewire_fraction", "value"])
    return nets, meta


def _rewire(reference, s, r):
    G = nx.from_numpy_array(reference)
    nx.connected_double_edge_swap(G, nswap=s, seed=int((NOISE_SEED * 1000 + s) * 1000 + r))
    return nx.to_numpy_array(G, nodelist=range(len(reference)))


def _noise_draws(run, reference, raw, similarity, meta, slow):
    i_eta, i_gamma = _grid_index(meta)
    cell = i_gamma * N_GRID + i_eta

    def best_cell(values, rows):
        return int(pd.Series(_orient(values, similarity)).groupby(cell[rows]).mean().idxmin())

    everything = np.arange(len(meta))
    p0 = best_cell(raw, everything)
    p0e, p0g = p0 % N_GRID, p0 // N_GRID
    window = everything[(np.abs(i_eta - p0e) <= NOISE_WINDOW) & (np.abs(i_gamma - p0g) <= NOISE_WINDOW)]
    lo_e, hi_e = max(0, p0e - NOISE_WINDOW), min(N_GRID - 1, p0e + NOISE_WINDOW)
    lo_g, hi_g = max(0, p0g - NOISE_WINDOW), min(N_GRID - 1, p0g + NOISE_WINDOW)
    repeats = NOISE_REPEATS_SLOW if slow else NOISE_REPEATS_FAST

    effective, draws = {}, []
    for s in NOISE_LEVELS:
        moved = []
        for r in range(NOISE_REPEATS_FAST):          # effective degeneration always over 20
            ref = _rewire(reference, s, r)
            moved.append(np.triu((reference > 0) & (ref == 0), 1).sum() / data.N_EDGES)
            if r >= repeats:
                continue
            rows = window if slow else everything
            vals, _ = run([(int(i), ref) for i in rows], label=f"noise tolerance, s={s}, draw {r + 1}/{repeats}")
            c = best_cell(vals, rows)
            censored = False
            ce, cg = c % N_GRID, c // N_GRID
            on_edge = slow and ((ce == lo_e and lo_e > 0) or (ce == hi_e and hi_e < N_GRID - 1) or
                                (cg == lo_g and lo_g > 0) or (cg == hi_g and hi_g < N_GRID - 1))
            if on_edge and s <= NOISE_FALLBACK_CUTOFF:
                vals, _ = run([(int(i), ref) for i in everything])
                c = best_cell(vals, everything)
                ce, cg = c % N_GRID, c // N_GRID
            elif on_edge:
                censored = True
            drift = float(NOISE_WINDOW) if censored else float(np.hypot(ce - p0e, cg - p0g))
            draws.append({"s": s, "repeat": r, "drift": drift})
        effective[s] = float(np.mean(moved))
    return pd.DataFrame(draws), effective


@dataclass
class Report:
    """Result of :func:`evaluate`.

    Attributes:
        name: the name given to the measure.
        table: one row per read-out, the new measure next to the eight selected
            published measures, with each read-out's criterion, direction and
            the new measure's rank among the nine (1 = best).
        landscape: the new measure's value and runtime for every morphospace network.
    """

    name: str
    table: pd.DataFrame
    landscape: pd.DataFrame
    extras: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        with pd.option_context("display.width", 200, "display.max_columns", 20,
                               "display.float_format", "{:.4g}".format):
            return f"Report for '{self.name}'\n{self.table}"


def evaluate(measure: Callable[[np.ndarray, np.ndarray], float], name: str | None = None,
             similarity: bool = False, reference: np.ndarray | str | Path | None = None,
             criteria: Iterable[str] = CRITERIA, n_jobs: int = 1, progress: bool = True) -> Report:
    """Score a network distance measure on the benchmark.

    Args:
        measure: ``f(A, B) -> float`` on two binary, symmetric (100, 100) float
            arrays; ``A`` is the generated network, ``B`` the reference.
        name: label for the report (defaults to the function name).
        similarity: set True if higher values mean more alike.
        reference: the empirical consensus connectome (array or ``.npy`` path).
            Defaults to ``data.reference_dir() / "consensus.npy"``. Only the
            accuracy criterion runs without it.
        criteria: subset of ``("agreement", "plausibility", "efficiency",
            "sensitivity", "accuracy")``. Agreement, plausibility and efficiency
            share one pass over the 25,000 networks.
        n_jobs: worker processes (1 = in-process, -1 = all cores). With more
            than one, ``measure`` must be importable (defined in a module, not
            a lambda) and the per-comparison runtimes include pool overhead.
        progress: show progress bars.

    Returns:
        A :class:`Report`.

    Cost, in calls of ``measure``: 25,000 for the landscape; 20,200 for the
    degeneration read-outs (MAE, CV); 20,000 for the two recovery experiments;
    and for the noise tolerance N* one landscape per draw - 160 full ones for a
    fast measure, 80 windowed ones (4,410 calls each) for one slower than
    10 ms. N* dominates; leave out ``"sensitivity"`` to skip it.
    """
    criteria = set(criteria)
    unknown = criteria - set(CRITERIA)
    if unknown:
        raise ValueError(f"unknown criteria {sorted(unknown)}; choose from {CRITERIA}")
    name = name or getattr(measure, "__name__", "new measure")
    needs_reference = criteria - {"accuracy"}
    ref = data.load_reference(reference) if needs_reference else None

    nets, meta = data.load_networks("morphospace")
    run = _Runner(measure, nets if needs_reference else None, n_jobs, progress)
    row, extras = {}, {}
    landscape = pd.DataFrame(meta)
    try:
        if needs_reference:
            raw, ms = run([(i, ref) for i in range(len(nets))], label="landscape (25,000 networks)")
            landscape["value"], landscape["time_ms"] = raw, ms
            if np.isnan(raw).any():
                warnings.warn(f"{int(np.isnan(raw).sum())} comparisons returned NaN or raised", stacklevel=2)
            row.update(landscape_readouts(raw, similarity, meta, nets, _published_landscapes(), exclude=name))
            row["runtime_ms"] = float(np.mean(ms))
            extras["runtime_sd_ms"] = float(np.std(ms, ddof=1))

        if "sensitivity" in criteria:
            traj, tmeta = _degeneration_trajectories(ref)
            vals, _ = run([(a, ref) for a in traj], label="degeneration (20,200 networks)")
            row.update(degeneration_readouts(tmeta.assign(value=vals), tmeta, similarity))
            draws, effective = _noise_draws(run, ref, raw, similarity, meta, row["runtime_ms"] > SLOW_MS)
            row["noise_tolerance"] = noise_tolerance(draws, effective)
            extras["noise_draws"] = draws

        if "accuracy" in criteria:
            for k in ("wide", "window"):
                rec = data.load_networks(f"recovery_{k}")
                pairs = [(t, g) for t in rec["targets"] for g in rec["grid"]]
                vals, _ = run(pairs, label=f"parameter recovery, {k}")
                row[f"recovery_{k}"], row[f"r_eta_{k}"] = recovery_readouts(
                    vals.reshape(len(rec["targets"]), -1), similarity, rec)
            individuals, loo = data.load_optional_reference("individuals"), data.load_optional_reference("loo_consensuses")
            if needs_reference and individuals is not None and loo is not None:
                d_real, _ = run(list(zip(individuals, loo)))
                row.update(discrimination(d_real, landscape["value"].to_numpy(), similarity))
    finally:
        run.close()

    return Report(name, _compare(name, row, criteria), landscape, extras)


def _compare(name: str, row: dict, criteria) -> pd.DataFrame:
    published = published_readouts(SELECTED).rename(columns=NAMES)
    keep = [(k, c, label, direction) for k, c, label, direction in READOUTS if c in criteria]
    table = pd.DataFrame(
        [{"criterion": c, "readout": label, "direction": {1: "higher", -1: "lower", 0: ""}[direction],
          name: row.get(k, np.nan), **published.loc[k].to_dict()} for k, c, label, direction in keep],
        index=[k for k, *_ in keep])
    ranks = []
    for k, c, label, direction in keep:
        vals = table.loc[k, [name] + list(published.columns)].to_numpy(float)
        ranks.append(rankdata(-direction * vals, nan_policy="omit")[0] if direction and np.isfinite(vals[0]) else np.nan)
    table["rank"] = ranks
    return table
