"""
Comprehensive computational capacity assessment of network architectures
via Echo State Network (ESN) reservoir computing.

Measures multiple facets of computation:
  - Linear memory capacity (faithful recall of past inputs)
  - Nonlinear capacity (quadratic/cubic transformations of past inputs)
  - Cross-term capacity (interaction between inputs at different lags)
  - Input separation (discrimination between distinct input streams)
  - State richness (dimensionality and entropy of reservoir dynamics)
  - Edge-of-chaos proximity (Lyapunov exponent estimate)

Compatible with any (N, N) adjacency matrix. Designed for batch evaluation
across tens of thousands of networks.

References:
  - Jaeger (2001) - Echo state networks
  - Büsing et al. (2010) - Connectivity, dynamics, and memory in RC
  - Dambre et al. (2012) - Information processing capacity of dynamical systems
  - Suárez et al. (2024) - Connectome-based reservoir computing (conn2res)
"""

import numpy as np
from scipy import linalg
from typing import Optional


def _prepare_reservoir_weights(
    A: np.ndarray,
    spectral_radius: float = 0.95,
    exc_inh_ratio: float = 0.8,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """
    Normalize adjacency matrix for ESN use.

    For binary matrices: assigns random excitatory/inhibitory signs (Dale's law)
    to create heterogeneous dynamics. For weighted matrices: preserves relative
    weight structure.

    Always rescales to target spectral_radius.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    W = A.copy().astype(np.float64)
    np.fill_diagonal(W, 0)

    if np.allclose(W, 0):
        return W

    # Check if binary (only 0s and 1s)
    unique_vals = np.unique(W[W > 0])
    is_binary = len(unique_vals) <= 1

    if is_binary:
        # Assign excitatory/inhibitory signs per NODE (Dale's law):
        # each node is either excitatory or inhibitory
        n = W.shape[0]
        n_exc = int(n * exc_inh_ratio)
        signs = np.ones(n)
        inh_nodes = rng.choice(n, size=n - n_exc, replace=False)
        signs[inh_nodes] = -1
        # All outgoing connections from a node have the same sign
        W = W * signs[:, np.newaxis]
    else:
        # Weighted matrix: apply random sign to fraction of edges
        # to introduce inhibitory connections if all weights are positive
        if np.all(W >= 0):
            mask = W > 0
            sign_mask = rng.rand(*W.shape) > exc_inh_ratio
            W[mask & sign_mask] *= -1

    # Normalize spectral radius
    eigenvalues = linalg.eigvals(W)
    rho = np.max(np.abs(eigenvalues))

    if rho < 1e-10:
        return W

    W = W * (spectral_radius / rho)
    return W


def _run_reservoir(
    W: np.ndarray,
    inputs: np.ndarray,
    input_weights: np.ndarray,
    leak_rate: float = 1.0,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Drive reservoir with input signal and collect states.

    Parameters
    ----------
    W : (N, N) reservoir weight matrix
    inputs : (T,) or (T, n_inputs) input signal
    input_weights : (N, n_inputs) input-to-reservoir weights
    leak_rate : leaky integration rate (1.0 = no leak = standard ESN)
    bias : (N,) optional bias vector

    Returns
    -------
    states : (T, N) reservoir state matrix
    """
    N = W.shape[0]
    inputs = np.atleast_2d(inputs)
    if inputs.shape[0] == 1:
        inputs = inputs.T  # ensure (T, n_inputs)
    T = inputs.shape[0]

    states = np.zeros((T, N))
    x = np.zeros(N)

    if bias is None:
        bias = np.zeros(N)

    for t in range(T):
        u = input_weights @ inputs[t]
        x_new = np.tanh(W @ x + u + bias)
        x = (1 - leak_rate) * x + leak_rate * x_new
        states[t] = x

    return states


def _ridge_regression(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """
    Solve Y = X @ W via ridge regression.
    Returns W of shape (n_features, n_targets).
    """
    # X: (T, n_features), Y: (T, n_targets)
    n = X.shape[1]
    W = linalg.solve(X.T @ X + alpha * np.eye(n), X.T @ Y, assume_a='pos')
    return W


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, clipped to [0, 1]."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(np.clip(1 - ss_res / ss_tot, 0.0, 1.0))


def _effective_dimensionality(S: np.ndarray) -> float:
    """
    Effective dimensionality from singular values S.
    Uses the participation ratio: (sum s_i^2)^2 / sum s_i^4
    This equals 1 if one component dominates, N if all equal.
    """
    s2 = S ** 2
    s2 = s2[s2 > 1e-12]
    if len(s2) == 0:
        return 0.0
    return float((np.sum(s2)) ** 2 / np.sum(s2 ** 2))


def _spectral_entropy(S: np.ndarray) -> float:
    """
    Entropy of the normalized singular value spectrum.
    High entropy = rich, distributed dynamics. Low = low-dimensional.
    """
    s2 = S ** 2
    s2 = s2[s2 > 1e-12]
    if len(s2) == 0:
        return 0.0
    p = s2 / np.sum(s2)
    return float(-np.sum(p * np.log(p)))


def computational_capacity(
    A: np.ndarray,
    n_trials: int = 3,
    T_washout: int = 200,
    T_train: int = 2000,
    T_test: int = 500,
    spectral_radius: float = 0.95,
    input_scaling: float = 0.5,
    leak_rate: float = 1.0,
    max_delay: int = 30,
    max_nonlinear_delay: int = 15,
    ridge_alpha: float = 1e-4,
    seed: int = 42,
) -> dict:
    """
    Comprehensive computational capacity assessment of a network.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Adjacency or connectivity matrix. Can be weighted or binary,
        directed or undirected. Will be normalized to desired spectral radius.

    n_trials : int
        Number of independent random input trials to average over.
        More trials = more stable estimates, but slower.

    T_washout : int
        Initial timesteps to discard (let transients decay).

    T_train : int
        Timesteps for training the readout.

    T_test : int
        Timesteps for evaluating capacity (held-out).

    spectral_radius : float
        Target spectral radius for reservoir. 0.9 is standard;
        values near 1.0 probe edge-of-chaos dynamics.

    input_scaling : float
        Scaling of input weights. Small values keep dynamics in
        linear regime of tanh; larger values probe nonlinear regime.

    leak_rate : float
        Leaky integrator rate. 1.0 = standard ESN (no leak).
        Values < 1 slow down dynamics (longer timescales).

    max_delay : int
        Maximum lag for linear memory capacity evaluation.

    max_nonlinear_delay : int
        Maximum lag for nonlinear (quadratic) targets. Kept shorter
        than linear because nonlinear capacity decays faster.

    ridge_alpha : float
        Regularization for ridge regression readout.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Comprehensive capacity profile with the following keys:

        ## Linear memory capacity
        'memory_capacity_total' : float
            Total linear MC = sum of R² across all lags.
            Theoretical maximum = N (number of nodes).
        'memory_capacity_profile' : np.ndarray, shape (max_delay,)
            R² at each lag k = 1, ..., max_delay.
        'memory_timescale' : float
            Characteristic timescale: lag at which MC drops to 1/e of peak.

        ## Nonlinear computation capacity
        'nonlinear_capacity_total' : float
            Total quadratic MC = sum of R² for x(t-k)² targets.
        'nonlinear_capacity_profile' : np.ndarray, shape (max_nonlinear_delay,)
            R² for each quadratic target.
        'cubic_capacity_total' : float
            Total cubic MC = sum of R² for x(t-k)³ targets.

        ## Cross-term capacity
        'cross_capacity_total' : float
            Total R² for x(t-i)*x(t-j) interaction targets.
            Measures ability to compute conjunctions of past inputs.

        ## Capacity balance
        'memory_nonlinear_ratio' : float
            MC_linear / (MC_linear + MC_nonlinear). Near 1 = memory-dominated
            (routing-like). Near 0 = transformation-dominated (diffusion-like).
        'total_capacity' : float
            Sum of linear + quadratic + cubic + cross capacities.

        ## State richness
        'state_dimensionality' : float
            Participation ratio of reservoir state singular values.
            Higher = richer dynamics exploiting more of the network.
        'state_entropy' : float
            Entropy of normalized singular value spectrum.
        'state_rank' : int
            Numerical rank of the state matrix (singular values > 1e-6).

        ## Input separation
        'separation_ratio' : float
            How well the reservoir separates distinct input streams.
            Measured as ratio of between-class to within-class state variance.

        ## Edge-of-chaos / stability
        'lyapunov_exponent' : float
            Estimated largest Lyapunov exponent from state perturbation.
            Negative = stable (ordered), near 0 = edge of chaos, positive = chaotic.

        ## Summary scores (for quick use in PCA/embedding)
        'capacity_vector' : np.ndarray, shape (8,)
            [memory_total, nonlinear_total, cubic_total, cross_total,
             memory_nonlinear_ratio, state_dimensionality, separation_ratio,
             lyapunov_exponent]
    """
    rng = np.random.RandomState(seed)
    N = A.shape[0]
    T_total = T_washout + T_train + T_test + max_delay

    # Prepare reservoir (signs assigned with rng for reproducibility)
    W = _prepare_reservoir_weights(A, spectral_radius=spectral_radius, rng=rng)

    # Accumulators across trials
    mc_profiles = []
    nl_profiles = []
    cb_profiles = []
    cross_totals = []
    state_dims = []
    state_ents = []
    state_ranks = []
    sep_ratios = []
    lyap_exps = []

    for trial in range(n_trials):
        trial_seed = seed + trial * 1000
        rng_trial = np.random.RandomState(trial_seed)

        # ---- Generate input and drive reservoir ----
        # Uniform input in [-1, 1] (standard for capacity measurement)
        u = rng_trial.uniform(-1, 1, size=T_total)

        # Input weights: all nodes receive input (standard for capacity measurement)
        # Drawn from uniform [-1, 1] and scaled by input_scaling
        w_in = rng_trial.uniform(-1, 1, size=N) * input_scaling
        w_in = w_in.reshape(-1, 1)

        # Small random bias to break symmetry
        bias = rng_trial.uniform(-0.1, 0.1, size=N)

        # Run reservoir
        states = _run_reservoir(W, u.reshape(-1, 1), w_in, leak_rate=leak_rate, bias=bias)

        # Discard washout
        states = states[T_washout:]
        u_valid = u[T_washout:]

        # For predicting u(t-k) from state(t), we need t >= max_delay
        # so all lags 1..max_delay are available as past inputs.
        # States usable for train+test start at index max_delay.
        S_usable = states[max_delay:]  # length = T_train + T_test
        S_train = S_usable[:T_train]
        S_test = S_usable[T_train:T_train + T_test]

        # ---- 1. LINEAR MEMORY CAPACITY ----
        mc_profile = np.zeros(max_delay)
        for k in range(1, max_delay + 1):
            # Target: input k steps before each state
            # S_train[j] = state at absolute time (max_delay + j)
            # target[j] = u_valid[max_delay + j - k]
            y_all = u_valid[max_delay - k : max_delay - k + T_train + T_test]
            y_train = y_all[:T_train]
            y_test = y_all[T_train:T_train + T_test]

            if len(y_train) < 10 or len(y_test) < 10:
                continue

            W_out = _ridge_regression(S_train, y_train.reshape(-1, 1), alpha=ridge_alpha)
            y_pred = S_test @ W_out
            mc_profile[k - 1] = _r_squared(y_test, y_pred.ravel())

        mc_profiles.append(mc_profile)

        # ---- 2. NONLINEAR (QUADRATIC) CAPACITY ----
        nl_profile = np.zeros(max_nonlinear_delay)
        for k in range(1, max_nonlinear_delay + 1):
            y_all = u_valid[max_delay - k : max_delay - k + T_train + T_test]
            y_nl = y_all ** 2

            y_train_nl = y_nl[:T_train]
            y_test_nl = y_nl[T_train:T_train + T_test]

            if len(y_train_nl) < 10 or len(y_test_nl) < 10:
                continue

            W_out = _ridge_regression(S_train, y_train_nl.reshape(-1, 1), alpha=ridge_alpha)
            y_pred = S_test @ W_out
            nl_profile[k - 1] = _r_squared(y_test_nl, y_pred.ravel())

        nl_profiles.append(nl_profile)

        # ---- 3. CUBIC CAPACITY ----
        cb_profile = np.zeros(max_nonlinear_delay)
        for k in range(1, max_nonlinear_delay + 1):
            y_all = u_valid[max_delay - k : max_delay - k + T_train + T_test]
            y_cb = y_all ** 3

            y_train_cb = y_cb[:T_train]
            y_test_cb = y_cb[T_train:T_train + T_test]

            if len(y_train_cb) < 10 or len(y_test_cb) < 10:
                continue

            W_out = _ridge_regression(S_train, y_train_cb.reshape(-1, 1), alpha=ridge_alpha)
            y_pred = S_test @ W_out
            cb_profile[k - 1] = _r_squared(y_test_cb, y_pred.ravel())

        cb_profiles.append(cb_profile)

        # ---- 4. CROSS-TERM CAPACITY ----
        cross_total = 0.0
        n_cross = 0
        for i in range(1, min(8, max_nonlinear_delay)):
            for j in range(i + 1, min(8, max_nonlinear_delay)):
                y_i = u_valid[max_delay - i : max_delay - i + T_train + T_test]
                y_j = u_valid[max_delay - j : max_delay - j + T_train + T_test]
                y_cross = y_i * y_j

                y_train_c = y_cross[:T_train]
                y_test_c = y_cross[T_train:T_train + T_test]

                if len(y_train_c) < 10 or len(y_test_c) < 10:
                    continue

                W_out = _ridge_regression(S_train, y_train_c.reshape(-1, 1), alpha=ridge_alpha)
                y_pred = S_test @ W_out
                cross_total += _r_squared(y_test_c, y_pred.ravel())
                n_cross += 1

        cross_totals.append(cross_total)

        # ---- 5. STATE RICHNESS ----
        # SVD of the full usable states for robust dimensionality estimate
        S_centered = S_usable - S_usable.mean(axis=0)
        try:
            _, S_vals, _ = linalg.svd(S_centered, full_matrices=False)
            state_dims.append(_effective_dimensionality(S_vals))
            state_ents.append(_spectral_entropy(S_vals))
            state_ranks.append(int(np.sum(S_vals > 1e-6 * S_vals[0])))
        except linalg.LinAlgError:
            state_dims.append(0.0)
            state_ents.append(0.0)
            state_ranks.append(0)

        # ---- 6. INPUT SEPARATION ----
        # Drive with two distinct input streams, measure state divergence
        u2 = rng_trial.uniform(-1, 1, size=T_washout + T_test)
        states_alt = _run_reservoir(W, u2.reshape(-1, 1), w_in, leak_rate=leak_rate, bias=bias)
        states_alt = states_alt[T_washout:]

        # Also re-run original input for fair comparison
        u1_seg = u[:T_washout + T_test]
        states_orig = _run_reservoir(W, u1_seg.reshape(-1, 1), w_in, leak_rate=leak_rate, bias=bias)
        states_orig = states_orig[T_washout:]

        min_t = min(len(states_orig), len(states_alt), T_test)
        states_orig = states_orig[:min_t]
        states_alt = states_alt[:min_t]

        # Between-stream distance vs within-stream variance
        between_var = np.mean(np.sum((states_orig - states_alt) ** 2, axis=1))
        within_var = 0.5 * (np.var(states_orig, axis=0).sum() +
                            np.var(states_alt, axis=0).sum())
        if within_var > 1e-12:
            sep_ratios.append(float(between_var / within_var))
        else:
            sep_ratios.append(0.0)

        # ---- 7. LYAPUNOV EXPONENT ESTIMATE ----
        # Perturb initial state slightly, track divergence
        u_lyap = rng_trial.uniform(-1, 1, size=T_washout + 200)

        states_base = _run_reservoir(W, u_lyap.reshape(-1, 1), w_in,
                                     leak_rate=leak_rate, bias=bias)

        # Perturbed run: nudge state at t=T_washout
        eps = 1e-8
        x_base = states_base[T_washout - 1].copy()
        x_pert = x_base + rng_trial.randn(N) * eps

        # Continue both from T_washout with same input
        n_lyap = 200
        divergences = np.zeros(n_lyap)
        x_b = x_base.copy()
        x_p = x_pert.copy()

        for t in range(n_lyap):
            idx = T_washout + t
            if idx >= len(u_lyap):
                break
            inp = w_in.ravel() * u_lyap[idx] + bias
            x_b = (1 - leak_rate) * x_b + leak_rate * np.tanh(W @ x_b + inp)
            x_p = (1 - leak_rate) * x_p + leak_rate * np.tanh(W @ x_p + inp)
            divergences[t] = np.log(np.linalg.norm(x_p - x_b) + 1e-20)

        # Linear fit to log-divergence gives Lyapunov exponent
        valid = np.isfinite(divergences) & (divergences > -40)
        if np.sum(valid) > 10:
            t_vals = np.arange(n_lyap)[valid]
            d_vals = divergences[valid]
            # Simple linear regression
            slope = np.polyfit(t_vals, d_vals, 1)[0]
            lyap_exps.append(float(slope))
        else:
            lyap_exps.append(0.0)

    # ---- AGGREGATE ACROSS TRIALS ----
    mc_profile = np.mean(mc_profiles, axis=0)
    nl_profile = np.mean(nl_profiles, axis=0)
    cb_profile = np.mean(cb_profiles, axis=0)

    mc_total = float(np.sum(mc_profile))
    nl_total = float(np.sum(nl_profile))
    cb_total = float(np.sum(cb_profile))
    cross_total = float(np.mean(cross_totals))

    total_cap = mc_total + nl_total + cb_total + cross_total

    # Memory timescale: find lag where MC drops to 1/e of peak
    if mc_profile.max() > 0:
        threshold = mc_profile.max() / np.e
        below = np.where(mc_profile < threshold)[0]
        memory_timescale = float(below[0] + 1) if len(below) > 0 else float(max_delay)
    else:
        memory_timescale = 0.0

    # Memory vs nonlinear balance
    denom = mc_total + nl_total
    mn_ratio = float(mc_total / denom) if denom > 1e-12 else 0.5

    state_dim = float(np.mean(state_dims))
    state_ent = float(np.mean(state_ents))
    state_rk = int(np.round(np.mean(state_ranks)))
    sep_ratio = float(np.mean(sep_ratios))
    lyap_exp = float(np.mean(lyap_exps))

    # ---- BUILD RESULTS ----
    results = {
        # Linear memory
        'memory_capacity_total': mc_total,
        # 'memory_capacity_profile': mc_profile,
        'memory_timescale': memory_timescale,
        # Nonlinear
        'nonlinear_capacity_total': nl_total,
        # 'nonlinear_capacity_profile': nl_profile,
        'cubic_capacity_total': cb_total,
        # 'cubic_capacity_profile': cb_profile,
        # Cross-terms
        'cross_capacity_total': cross_total,
        # Balance
        'memory_nonlinear_ratio': mn_ratio,
        'total_capacity': total_cap,
        # State richness
        'state_dimensionality': state_dim,
        'state_entropy': state_ent,
        'state_rank': state_rk,
        # Separation
        'separation_ratio': sep_ratio,
        # Stability
        'lyapunov_exponent': lyap_exp,
        # Summary vector for embedding / PCA (8 features)
        # 'capacity_vector': np.array([
        #     mc_total,
        #     nl_total,
        #     cb_total,
        #     cross_total,
        #     mn_ratio,
        #     state_dim,
        #     sep_ratio,
        #     lyap_exp,
        # ]),
    }

    return results


# comp -> computational_capacity

# --------------------------------------------------------------------------------
# Notebook simplified implementation
# --------------------------------------------------------------------------------

from sklearn.linear_model import Ridge

def prepare_reservoir_notebook(W_raw, spectral_radius):
    eigenvalues = linalg.eigvals(W_raw)
    rho = np.max(np.abs(eigenvalues))
    if rho > 0:
        W = W_raw * (spectral_radius / rho)
    else:
        W = W_raw
    return W

def run_reservoir_notebook(W, inputs, input_scaling, leak_rate=1.0, bias_scale=0.1, seed=42):
    rng = np.random.RandomState(seed)
    N = W.shape[0]
    T = len(inputs)
    
    # Input weights: uniform [-1, 1] scaled by input_scaling
    W_in = rng.uniform(-1, 1, size=(N, 1)) * input_scaling
    bias = rng.uniform(-bias_scale, bias_scale, size=N)
    
    states = np.zeros((T, N))
    x = np.zeros(N)
    
    for t in range(T):
        u = inputs[t]
        x_new = np.tanh(W @ x + W_in[:, 0] * u + bias)
        x = (1 - leak_rate) * x + leak_rate * x_new
        states[t] = x
        
    return states, W_in

def evaluate_memory_capacities_notebook(W_raw, spectral_radius=0.95, input_scaling=0.5, leak_rate=1.0, max_delay=40, n_trials=10):
    """
    A straightforward baseline implementation of linear and non-linear memory capacity
    evaluation natively matched to what is in the hyperparameter tuning notebook.
    """
    # Setup
    T_train = 2000
    T_test = 1000
    T_washout = 200
    T_total = T_train + T_test + T_washout + max_delay
    
    all_linear_mc = []
    all_nonlinear_mc = []
    
    for trial in range(n_trials):
        # Use a different seed for each trial to ensure different input sequences and weights
        trial_seed = 42 + trial * 1000
        np.random.seed(trial_seed)
        inputs = np.random.uniform(-1, 1, T_total)
        
        W = prepare_reservoir_notebook(W_raw, spectral_radius)
        states, _ = run_reservoir_notebook(W, inputs, input_scaling, leak_rate=leak_rate, seed=trial_seed)
        
        states = states[T_washout:]
        trial_inputs = inputs[T_washout:]
        
        # States available for training/testing (we start from max_delay so we have history)
        S_valid = states[max_delay:]
        S_train = S_valid[:T_train]
        S_test = S_valid[T_train : T_train + T_test]
        
        # Evaluate Linear MC
        linear_mc = []
        nonlinear_mc = []
        
        for k in range(1, max_delay + 1):
            # The target is the input k steps in the past
            target_seq = trial_inputs[max_delay - k : max_delay - k + T_train + T_test]
            
            # 1. Linear target
            y_train = target_seq[:T_train]
            y_test = target_seq[T_train : T_train + T_test]
            
            model_lin = Ridge(alpha=1e-4)
            model_lin.fit(S_train, y_train)
            score_lin = model_lin.score(S_test, y_test)
            linear_mc.append(max(0, score_lin)) # R^2
            
            # 2. Non-linear target (quadratic)
            y_train_nl = y_train**2
            y_test_nl = y_test**2
            
            model_nl = Ridge(alpha=1e-4)
            model_nl.fit(S_train, y_train_nl)
            score_nl = model_nl.score(S_test, y_test_nl)
            nonlinear_mc.append(max(0, score_nl))
            
        all_linear_mc.append(linear_mc)
        all_nonlinear_mc.append(nonlinear_mc)
        
    # Average across trials
    avg_linear_mc = np.mean(all_linear_mc, axis=0)
    avg_nonlinear_mc = np.mean(all_nonlinear_mc, axis=0)
        
    return {
        'memory_capacity_total_notebook': float(np.sum(avg_linear_mc)),
        'nonlinear_capacity_total_notebook': float(np.sum(avg_nonlinear_mc)),
        'memory_capacity_profile_notebook': avg_linear_mc,
        'nonlinear_capacity_profile_notebook': avg_nonlinear_mc
    }


def evaluate_memory_capacities_notebook_damicelli(W_raw, max_delay=40, n_trials=10):
    """
    Wrapper for validate evaluating the notebook implementation with parameters 
    identified by Damicelli et al.
    """
    return evaluate_memory_capacities_notebook(
        W_raw, 
        spectral_radius=0.99, 
        input_scaling=1e-5, 
        leak_rate=1.0, 
        max_delay=max_delay, 
        n_trials=n_trials
    )

# --------------------------------------------------------------------------------
# Supplementary Kayson implementation (from supplementary_stuff.ipynb)
# --------------------------------------------------------------------------------

def evaluate_supplementary_kayson(W_raw, X_train, X_test, y_train_lin, y_test_lin, y_train_nl, y_test_nl,
                                  spectral_radius=0.9, input_scaling=1e-5, leak_rate=1.0, 
                                  bias=0, n_transient=100, n_trials=100):
    """
    Extracts the exact plot_forgetting_curve inner-loop evaluation used in supplementary_stuff.ipynb.
    Averages over multiple trials.
    """
    import echoes
    import src.analysis.utils_kayson_damicelli as ut
    
    results_lin = []
    results_nonlin = []
    mc_lin = []
    mc_nonlin = []
    
    for trial in range(n_trials):
        esn = echoes.ESNRegressor(
            W=np.array(W_raw, dtype=np.float64),
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            bias=bias,
            n_transient=n_transient,
            random_state=trial,
            regression_method="pinv" # default in Echoes
        )

        # Linear MC
        y_pred = esn.fit(X_train, y_train_lin).predict(X_test)
        r_lin  = ut.forgetting(y_test_lin[n_transient:], y_pred[n_transient:])
        results_lin.append(r_lin[0])  # per-lag array
        mc_lin.append(r_lin[1])       # total score

        # Nonlinear MC
        y_pred_nl = esn.fit(X_train, y_train_nl).predict(X_test)
        r_nonlin  = ut.forgetting(y_test_nl[n_transient:], y_pred_nl[n_transient:])
        results_nonlin.append(r_nonlin[0])
        mc_nonlin.append(r_nonlin[1])

    return {
        'mc_mean': float(np.mean(mc_lin)),
        'mc_std': float(np.std(mc_lin)),
        'mc_values_for_indiv_lags': np.mean(results_lin, axis=0),
        'mc_nonlinear_mean': float(np.mean(mc_nonlin)),
        'mc_nonlinear_std': float(np.std(mc_nonlin)),
        'mc_nonlinear_values_for_indiv_lags': np.mean(results_nonlin, axis=0)
    }