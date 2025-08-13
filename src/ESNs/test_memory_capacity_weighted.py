# Modelled after damicelli's work. 

import numpy as np
from typing import List, Tuple, Dict, Optional
from echoes.esn import ESNRegressor
from .generate_weight_matrices_bio_no_rank_weighted import build_weight_matrix_from_connectome

def _generate_mc_dataset(train_len: int, test_len: int, n_lags: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_len = train_len + test_len + n_lags + 100
    seq = rng.uniform(-0.5, 0.5, size=(total_len,))
    def build_targets(x: np.ndarray, lags: int) -> np.ndarray:
        T = len(x) - lags
        targets = np.zeros((T, lags), dtype=float)
        for i in range(lags):
            targets[:, i] = x[lags - (i + 1) : - (i + 1) if i + 1 > 0 else None]
        return targets
    Y_full = build_targets(seq, n_lags)
    start_train = 100
    end_train = start_train + train_len
    X_train = seq[start_train : end_train].reshape(-1, 1)
    Y_train = Y_full[start_train : end_train]
    X_test = seq[end_train : end_train + test_len].reshape(-1, 1)
    Y_test = Y_full[end_train : end_train + test_len]
    return X_train, Y_train, X_test, Y_test

def evaluate_memory_capacity(W: np.ndarray, *, n_lags: int = 50, train_len: int = 4000, test_len: int = 1000, n_runs: int = 10, spectral_radius: float | None = None, random_state: int | None = None) -> Dict[str, float]:
    rng = np.random.default_rng(random_state)
    mc_values: List[float] = []
    for _ in range(n_runs):
        X_tr, Y_tr, X_te, Y_te = _generate_mc_dataset(train_len, test_len, n_lags, rng)
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
        # Vectorised Pearson r per column
        Yt = Y_te - Y_te.mean(axis=0, keepdims=True)
        Yp = Y_pred - Y_pred.mean(axis=0, keepdims=True)
        denom = (Yt.std(axis=0, ddof=0) * Yp.std(axis=0, ddof=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            r = (Yt * Yp).mean(axis=0) / denom
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        mc = float(np.sum(r**2))
        mc_values.append(mc)
    return { "mc_mean": float(np.mean(mc_values)), "mc_std": float(np.std(mc_values)) }

def evaluate_memory_capacity_from_connectome(connectome: np.ndarray, *, spectral_radius: float = 0.99, n_lags: int = 50, train_len: int = 4000, test_len: int = 1000, n_runs: int = 10, random_state: Optional[int] = None, symmetrize: bool = True) -> Dict[str, float]:
    """Convenience: preserve weights from connectome when building W."""
    W = build_weight_matrix_from_connectome(connectome, spectral_radius=spectral_radius, symmetrize=symmetrize, zero_diagonal=True)
    return evaluate_memory_capacity(W, n_lags=n_lags, train_len=train_len, test_len=test_len, n_runs=n_runs, spectral_radius=1.0, random_state=random_state)
