import numpy as np
from typing import Iterable, List, Tuple, Dict, Optional
from echoes.esn import ESNRegressor
from .generate_weight_matrices_bio_no_rank_weighted import build_weight_matrix_from_connectome

def _generate_sequence_recall_dataset(L: int, n_trials: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    recall_indices: List[int] = []
    idx_offset = 0
    for _ in range(n_trials):
        pattern = rng.uniform(0.0, 1.0, size=L)
        X_fix = np.zeros((L, 2), dtype=float)
        X_fix[:, 0] = pattern
        Y_fix = np.zeros((L, 1), dtype=float)
        X_rec = np.zeros((L, 2), dtype=float)
        X_rec[:, 1] = 1.0
        Y_rec = pattern.reshape(-1, 1)
        X_trial = np.vstack([X_fix, X_rec])
        Y_trial = np.vstack([Y_fix, Y_rec])
        X_list.append(X_trial)
        Y_list.append(Y_trial)
        recall_indices.extend(list(range(idx_offset + L, idx_offset + 2 * L)))
        idx_offset += 2 * L
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    return X, Y, recall_indices

def evaluate_sequence_recall(W: np.ndarray, *, pattern_lengths: Iterable[int] = range(5, 26), train_trials: int = 800, test_trials: int = 200, n_runs: int = 5, spectral_radius: float | None = None, random_state: int | None = None) -> Dict[int, Dict[str, float]]:
    rng = np.random.default_rng(random_state)
    results: Dict[int, Dict[str, float]] = {}
    for L in pattern_lengths:
        r2_values: List[float] = []
        for _ in range(n_runs):
            X_tr, Y_tr, _ = _generate_sequence_recall_dataset(L, train_trials, rng)
            X_te, Y_te, recall_idx_te = _generate_sequence_recall_dataset(L, test_trials, rng)
            esn = ESNRegressor(
                W=W.copy(),
                spectral_radius=spectral_radius if spectral_radius is not None else 1.0,
                n_transient=0,
                input_scaling=1.0,
                leak_rate=1.0,
                bias=1.0,
                regression_method="pinv",
            )
            esn.fit(X_tr, Y_tr)
            Y_pred = esn.predict(X_te)
            y_true = Y_te[recall_idx_te, 0]
            y_hat = Y_pred[recall_idx_te, 0]
            ss_res = np.sum((y_true - y_hat) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_values.append(r2)
        results[L] = { "r2_mean": float(np.mean(r2_values)), "r2_std": float(np.std(r2_values)) }
    return results

def evaluate_sequence_recall_from_connectome(connectome: np.ndarray, *, spectral_radius: float = 0.99, pattern_lengths = range(5, 26), train_trials: int = 800, test_trials: int = 200, n_runs: int = 5, random_state: Optional[int] = None, symmetrize: bool = True) -> Dict[int, Dict[str, float]]:
    """Convenience: preserve weights from connectome when building W."""
    W = build_weight_matrix_from_connectome(connectome, spectral_radius=spectral_radius, symmetrize=symmetrize, zero_diagonal=True)
    return evaluate_sequence_recall(W, pattern_lengths=pattern_lengths, train_trials=train_trials, test_trials=test_trials, n_runs=n_runs, spectral_radius=1.0, random_state=random_state)
