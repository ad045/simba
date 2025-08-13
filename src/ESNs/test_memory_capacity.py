# Modelled after damicelli's work. 

import numpy as np
from typing import List, Tuple, Dict
from echoes.esn import ESNRegressor


def _generate_mc_dataset(train_len: int, #  = 4000,
                        test_len: int, #  = 1000,
                        n_lags: int, # = 50,
                        rng: np.random.Generator, 
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for the memory capacity task.

    Generated data is a random univariate input sequence X. Target consists of delayed versions of X for each lag in `1..n_lags`.
    Training and testing sequences are drawn sequentially from a single long sequence to avoid boundary effects.
    An initial transient segment is left in front of the training data.

    Args: 
        - train_len (int): Length of the training sequence.
        - test_len (int): Length of the testing sequence.
        - n_lags (int): Number of delayed outputs to generate.
        - rng (np.random.Generator): Random number generator. Random number generator instance.

    Returns: 
        - X_train (np.ndarray, shape (train_len, 1)): Input training sequence.
        - Y_train (np.ndarray, shape (train_len, n_lags)): Training targets; column 'i' is the input delayed by 'i+1' time steps.
        - X_test (np.ndarray, shape (test_len, 1)): Input testing sequence.
        - Y_test (np.ndarray, shape (test_len, n_lags)): Testing targets.
    """

    # Get total length of to-be-generated sequence, and generate sequence 
    total_len = train_len + test_len + n_lags + 100  # extra for lagging and transient
    seq = rng.uniform(-0.5, 0.5, size=(total_len,))

    # Helper function to create delayed targets
    def build_targets(x: np.ndarray, lags: int) -> np.ndarray:
        T = len(x) - lags
        targets = np.zeros((T, lags), dtype=float)
        for i in range(lags):
            targets[:, i] = x[lags - (i + 1) : - (i + 1) if i + 1 > 0 else None]
        return targets
    
    Y_full = build_targets(seq, n_lags)

    # Discard initial transient and split
    start_train = 100  # discard first 100 samples (transient)
    end_train = start_train + train_len
    X_train = seq[start_train : end_train].reshape(-1, 1)
    Y_train = Y_full[start_train : end_train]
    X_test = seq[end_train : end_train + test_len].reshape(-1, 1)
    Y_test = Y_full[end_train : end_train + test_len]

    return X_train, Y_train, X_test, Y_test


def evaluate_memory_capacity(W: np.ndarray,
                             *, # for better overview: From here on only keyword arguments 
                             n_lags: int = 50,
                             train_len: int = 4000,
                             test_len: int = 1000,
                             n_runs: int = 10,
                             spectral_radius: float | None = None,
                             random_state: int | None = None,
                             ) -> Dict[str, float]:
    """
    Evaluate the memory capacity of an ESN with a given reservoir.

    Generates a new input/output sequence for each run and reinitialize the ESN. 
    MC (memory capacity) is computed as the sum of squared Pearson correaltion coefficients between each delayed output and it's target across the test set. 
    The final result aggregates the mean and the standard deviation over runs.

    Args: 
        - W (np.ndarray, shape (N, N)): Reservoir weight matrix.
        - n_lags (int, default 50): Number of delays to include in the target.  
            The original article sums correlations up to lag 200; adjust as needed.
        - train_len (int, default 4000): Number of time steps for training.
        - test_len (int, default 1000): Number of time steps for testing.
        - n_runs (int, default 10): Number of independent runs to average over.
            Each run uses a freshly generated input sequence and randomly initialised input weights.
        - spectral_radius (float or None, default None): Optional spectral radius to override the rescaling performed in 'build_reservoir_from_connectome'.  
            Pass 'None' to use the existing scaling of 'W'.
        - random_state (int or None, default None): Seed controlling both the data generation and the ESN’s input weights.  
            Different runs will still be independent because the seed is advanced internally.

    Returns: 
        - results (dict): Dictionary containing the mean and standard deviation of the memory capacity across runs.  Keys are 'mc_mean' and 'mc_std'.
    
    """

    rng = np.random.default_rng(random_state)
    mc_values: List[float] = []
    for _ in range(n_runs):

        # Generate dataset
        X_tr, Y_tr, X_te, Y_te = _generate_mc_dataset(train_len, test_len, n_lags, rng)

        # Instantiate ESN !!! 
        esn = ESNRegressor(
            W=W.copy(),
            spectral_radius=spectral_radius if spectral_radius is not None else 1.0,
            n_transient=0,
            input_scaling=1.0,
            leak_rate=1.0,
            bias=1.0,
            regression_method="pinv",
        )

        # Fit on training data
        esn.fit(X_tr, Y_tr)
        # Predict on test data
        Y_pred = esn.predict(X_te)

        # Compute squared Pearson correlations for each lag
        mc = 0.0
        for col in range(n_lags):
            y_true = Y_te[:, col]
            y_hat = Y_pred[:, col]
            # Compute Pearson correlation
            if np.std(y_true) == 0 or np.std(y_hat) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(y_true, y_hat)[0, 1]
            mc += corr ** 2
        mc_values.append(mc)

    return {
        "mc_mean": float(np.mean(mc_values)),
        "mc_std": float(np.std(mc_values)),
    }

