# Modelled after damicelli's work. 
# -> Used at least for main_pipeline_2_gnm_esn_landscape.py and for main_pipeline_2.py (esn part)

import numpy as np
from typing import List, Tuple, Dict, Optional
from echoes.esn import ESNRegressor


def _generate_mc_dataset(train_len: int, test_len: int, n_lags: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_len = int(train_len + test_len + n_lags + 100)
    # print("REMOVE AGAIN, in test_memory_capacity_weighted.py: total_len", total_len) -> THIS IS USED.
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


def evaluate_memory_capacity_from_connectome(connectome: np.ndarray, 
                                             *, 
                                             spectral_radius: float = 0.99, 
                                             n_lags: int = 50, 
                                             train_len: int = 4000, 
                                             test_len: int = 1000, 
                                             n_runs: int = 10, 
                                             input_scaling: float = 1.0,
                                             regression_method: str = "pinv",
                                             n_transient: int = 0,
                                             leak_rate: float = 1.0,
                                             bias: float = 1.0,
                                             random_state: Optional[int] = 42, 
 ) -> Dict[str, float]:
    
    # Random generator to generate the input sequence
    rng = np.random.default_rng(random_state)
    mc_values: List[float] = []
    
    # Loop through the number of runs
    for _ in range(n_runs):
        # Get data for training and testing
        X_tr, Y_tr, X_te, Y_te = _generate_mc_dataset(train_len, test_len, n_lags, rng)
        
        # Get esn regressor with the connectome as weight matrix
        esn = ESNRegressor(
            W=connectome.copy(),
            spectral_radius=spectral_radius,
            n_transient=n_transient,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            bias=bias,
            regression_method=regression_method,
        )
        
        esn.fit(X_tr, Y_tr)
        Y_pred = esn.predict(X_te)
        
        # Vectorised Pearson r per column
        Yt = Y_te - Y_te.mean(axis=0, keepdims=True)
        Yp = Y_pred - Y_pred.mean(axis=0, keepdims=True)
        denom = (Yt.std(axis=0, ddof=0) * Yp.std(axis=0, ddof=0))
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     r = (Yt * Yp).mean(axis=0) / denom
        #     r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        r = (Yt * Yp).mean(axis=0) / denom
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        r2 = r**2
        
        mc = float(np.sum(r2))
        mc_values.append(mc)
        
    return {"mc_mean": float(np.mean(mc_values)), 
            "mc_std": float(np.std(mc_values)),
            "mean_mc_of_individual_runs": mc_values,  
            # "r2_array_from_0_to_n_lags_minus_1": r2, # TODO: R2 array could be added, but code currently "nearly stops", when added? (TODO_R2_array for searching)
            "hparams": {
                "spectral_radius": spectral_radius,
                "n_lags": n_lags,
                "train_len": train_len,
                "test_len": test_len,
                "n_runs": n_runs,
                "input_scaling": input_scaling,
                "regression_method": regression_method,
                "n_transient": n_transient,
                "leak_rate": leak_rate,
                "bias": bias,
                "random_state": random_state,
                },
            }
    
    #     result = {
    #     "mc_mean": np.mean(mc_of_runs),
    #     "mc_std": np.std(mc_of_runs),
    #     "mean_mc_of_individual_runs": list(mc_of_runs),
    #     "mc_r2_5_to_25": mc_r2_5_to_25,
    #     "hparams": {
    #         "spectral_radius": spectral_radius,
    #             "n_lags": n_lags,
    #             "train_len": train_len,
    #             "test_len": test_len,
    #             "n_runs": n_runs,
    #             "input_scaling": input_scaling,
    #             "regression_method": regression_method,
    #             "n_transient": n_transient,
    #             "leak_rate": leak_rate,
    #             "bias": bias,
    #             "random_state": random_state,
    #             }
    #     }

    # return result
        