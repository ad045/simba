import numpy as np
import pandas as pd  # noqa: F401  # pandas is imported for completeness but not used directly
from typing import Iterable, List, Tuple, Dict
from echoes.esn import ESNRegressor
import time


def _generate_sequence_recall_dataset(
    L: int,
    n_trials: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Generate data for the sequence recall task.

    Each trial consists of a fixation phase of length ``L`` during
    which a random pattern of ``L`` numbers (uniformly drawn from
    [0, 1]) is presented via the first input channel while the recall
    channel is zero.  In the subsequent recall phase of length ``L``
    the recall channel is set to one and the network must reproduce
    the memorised pattern on its single output.  The second input
    channel is otherwise zero.  The target is zero during fixation
    (no recall) and equal to the pattern during recall.  All trials
    are concatenated into a single sequence for training or testing.

    Parameters
    ----------
    L : int
        Pattern length; also controls trial duration (2*L steps).
    n_trials : int
        Number of independent trials to generate.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    X : np.ndarray, shape (n_trials * 2*L, 2)
        Input matrix.  Column 0 carries the pattern, column 1 the
        recall cue.
    Y : np.ndarray, shape (n_trials * 2*L, 1)
        Target output.
    recall_indices : list of int
        Indices of the time steps corresponding to the recall phase.
        These indices are used to compute the performance metric.
    """
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    recall_indices: List[int] = []
    idx_offset = 0
    for _ in range(n_trials):
        pattern = rng.uniform(0.0, 1.0, size=L)
        # Fixation phase
        X_fix = np.zeros((L, 2), dtype=float)
        X_fix[:, 0] = pattern  # pattern presented on first input
        Y_fix = np.zeros((L, 1), dtype=float)
        # Recall phase
        X_rec = np.zeros((L, 2), dtype=float)
        X_rec[:, 1] = 1.0  # recall cue on second input
        Y_rec = pattern.reshape(-1, 1)
        X_trial = np.vstack([X_fix, X_rec])
        Y_trial = np.vstack([Y_fix, Y_rec])
        X_list.append(X_trial)
        Y_list.append(Y_trial)
        # Record recall indices (offset from start of sequence)
        recall_indices.extend(list(range(idx_offset + L, idx_offset + 2 * L)))
        idx_offset += 2 * L
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    return X, Y, recall_indices

def evaluate_sequence_recall(
    W: np.ndarray,
    *,
    pattern_lengths: Iterable[int] = range(5, 26),
    train_trials: int = 800,
    test_trials: int = 200,
    n_runs: int = 5,
    spectral_radius: float | None = None,
    random_state: int | None = None,
) -> Dict[int, Dict[str, float]]:
    """Evaluate sequence recall performance for multiple pattern lengths.

    For each pattern length in ``pattern_lengths`` and each run, an
    independent dataset is generated and the ESN is reinitialised.
    Performance is measured by the coefficient of determination (R²)
    between the predicted and true outputs, computed over the recall
    period only.  Results are aggregated by pattern length.

    Parameters
    ----------
    W : np.ndarray, shape (N, N)
        Reservoir weight matrix.
    pattern_lengths : iterable of int, default range(5, 26)
        Pattern lengths (task difficulties) to evaluate.
    train_trials : int, default 800
        Number of trials for training per run.
    test_trials : int, default 200
        Number of trials for testing per run.
    n_runs : int, default 5
        Number of independent runs per pattern length.
    spectral_radius : float or None, default None
        Optional spectral radius to override the scaling of ``W``.
    random_state : int or None, default None
        Seed controlling the data generation and input weight initialisation.

    Returns
    -------
    results : dict
        Mapping from pattern length to a dictionary with keys
        ``"r2_mean"`` and ``"r2_std"`` representing the average R²
        across runs and its standard deviation.
    """
    rng = np.random.default_rng(random_state)
    results: Dict[int, Dict[str, float]] = {}
    for L in pattern_lengths:
        r2_values: List[float] = []
        for _ in range(n_runs):
            # Generate dataset
            X_tr, Y_tr, recall_idx_tr = _generate_sequence_recall_dataset(L, train_trials, rng)
            X_te, Y_te, recall_idx_te = _generate_sequence_recall_dataset(L, test_trials, rng)
            # Instantiate ESN
            esn = ESNRegressor(
                W=W.copy(),
                spectral_radius=spectral_radius if spectral_radius is not None else 1.0,
                n_transient=0,
                input_scaling=1.0,
                leak_rate=1.0,
                bias=1.0,
                regression_method="pinv",
            )
            # Fit and predict
            esn.fit(X_tr, Y_tr)
            Y_pred = esn.predict(X_te)
            # Evaluate R² on recall period only
            y_true = Y_te[recall_idx_te, 0]
            y_hat = Y_pred[recall_idx_te, 0]
            # Compute R² manually
            ss_res = np.sum((y_true - y_hat) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_values.append(r2)
        results[L] = {
            "r2_mean": float(np.mean(r2_values)),
            "r2_std": float(np.std(r2_values)),
        }
    return results


