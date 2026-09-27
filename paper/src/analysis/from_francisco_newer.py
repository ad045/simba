# STOLEN FROM FRANCISCO: https://gitlab.com/francpsantos/dynamics_on_graphs/-/blob/master/models.py?ref_type=heads

import numpy as np
import scipy.signal as sig
from numba import njit
from sklearn.linear_model import LinearRegression
import pandas as pd


######################
## Evaluate network ##
######################
# Simulates and extracts relevant information

def evaluate_network(W, sim_time = 500, dt = 1e-2, K = 0.03, freq = 0.05, a = -0.1, noise = 0.001, estimate_DFA = False):

    # Spectral normalization. 
    eval, _ = np.linalg.eig(W)  
    W /= np.nanmax(np.abs(eval))

    # Simulates signal
    Z = simulate_stuart_landau(W = K * W, sim_time = sim_time + 10, dt = dt, freq = freq, a = a, noise = noise)

    # Removes initial transient
    Z = Z[:, int(10/dt):]
    signal = np.real(Z)

    if np.isnan(np.sum(signal)):
        return np.nan, np.nan * np.ones(W.shape[0]), np.nan * np.ones(W.shape[0])

    else:
        # Downsamples signal to sampling rates characteristic of BOLD -> makes further calculations much faster
        dt_ds = 0.5 # Sampling frequency of 2 Hz
        signal_ds = signal[:, ::int(dt_ds/dt)]

        # Filters signal in band of interest
        b, a = sig.butter(N = 2, Wn = [0.01, 0.1], btype = 'bandpass', fs = 1/dt_ds)
        signal_filt = sig.filtfilt(b, a, signal_ds, axis = -1)
        # Removes some signal from beginning and end to avoid edge effects from filter
        signal_filt = signal_filt[:, int(5/dt_ds):-int(5/dt_ds)]

        # Computes local and global metastability
        KOP_global, KOP_local = compute_KOP(signal = signal_ds, fs = 1/dt, W = W)

        meta_global = np.nanstd(KOP_global)
        meta_local = np.nanstd(KOP_local, axis = -1)

        if estimate_DFA:
            # Detrended fluctuation analysis
            ws = np.logspace(np.log10(50), np.log10(80), 5)
            DFA, _ = DFA_analysis(signal_ds, window_sizes = ws, fs = 1/dt_ds)
        else:
            DFA = np.nan * np.zeros(W.shape[0])

        return meta_global, meta_local, DFA










##########################
## Model Implementation ##
##########################

@njit(nogil=True)
def simulate_stuart_landau(W, sim_time, dt, freq = 0.05, a = -0.1, noise = 0.001):
        
    # Number of neuronal populations
    n = W.shape[0]
    
    # Defines vectors to store activity. Storage starts in the points defined in store and store_time timesteps of activity are recorded
    Z = np.zeros((W.shape[0], int(np.floor(sim_time / dt))), dtype = np.complex128)

    # Random initial conditions
    Z[:, 0] = 1e-5 * np.random.randn(n) + 1J*1e-5 * np.random.randn(n)

    # Runs simulations
    for st in range(1, int(np.floor(sim_time / dt))):

        # Gets input to each node
        input = np.sum(W * (Z[:, st-1][None, :] - Z[:, st-1][:, None]), axis = -1)

        # Calculates state changes  
        dz = Z[:, st-1] * (a + (2 * np.pi * freq * 1J) - np.abs(Z[:, st-1]**2)) + \
             input + noise * (np.random.randn(n) + 1J*np.random.randn(n))
        
        # Updates firing rate and state variable arrays using Euler method     
        Z[:, st] = Z[:, st-1] + dt * dz       
    
    return Z


@njit(nogil=True)
def simulate_kuramoto(W, sim_time, dt, freq = 0.05, noise = 0.001):
        
    # Number of neuronal populations
    n = W.shape[0]
    
    # Defines vectors to store activity. Storage starts in the points defined in store and store_time timesteps of activity are recorded
    phase = np.zeros((W.shape[0], int(np.floor(sim_time / dt))))

    # Random initial conditions
    phase[:, 0] = np.random.randn(n)

    # Runs simulations
    for st in range(1, int(np.floor(sim_time / dt))):

        # Gets "input" to each node
        input = np.sum(W * np.sin(phase[:, st-1][None, :] - phase[:, st-1][:, None]), axis = -1)

        # Calculates state changes  
        dphase = 2 * np.pi * freq + input + noise * np.random.randn(n)
        
        # Updates firing rate and state variable arrays using Euler method     
        phase[:, st] = phase[:, st-1] + dt * dphase       
    
    return phase




@njit(nogil=True)
def simulate_cellular_automaton(W, sim_time, T = 0.1, r1 = 0.001, r2 = 0.1):
        
    # From: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.178101
    
    # Number of neuronal populations
    n = int(W.shape[0])
    
    # Defines vectors to store activity. Storage starts in the points defined in store and store_time timesteps of activity are recorded
    state = np.zeros((W.shape[0], int(sim_time)))

    # Random initial conditions
    state[:, 0] = np.random.choice(np.array([-1, 0, 1]), n)

    # Runs simulations
    for st in range(1, int(sim_time)):

        # Gets "input" to each node
        state_aux = state[:, st-1].copy()
        state_aux[state_aux == -1] = 0
        # input = (W @ state_aux[:, None])[:, 0]
        # input = (W @ state_aux.astype(np.float64)[:, None])[:, 0]
        # input = (W @ np.ascontiguousarray(state_aux.astype(np.float64))[:, None])[:, 0] # makes it faster: "NumbaPerformanceWarning: '@' is faster on contiguous arrays, called on (Array(float64, 2, 'C', False, aligned=True), Array(float64, 2, 'A', False, aligned=True))"
        # input = (W @ np.ascontiguousarray(state_aux.astype(np.float64).reshape(-1, 1)))[:, 0]
        input = (W @ np.ascontiguousarray(state_aux.astype(W.dtype).reshape(-1, 1)))[:, 0] # no hardcoding of type, now allows both float32 and float64 adjacency matrices
        ## Transition from quiescent to excited ##
        # If input is larger than threshold T
        # Small probability = noise
        Q_to_E = np.logical_and(state[:, st-1] == 0, np.logical_or((input > T), np.random.uniform(low = 0, high = 1, size = n) < r1))
        state[:, st][Q_to_E] = 1

        ## Transition from excited to refractory ##
        E_to_R = state[:, st-1] == 1

        ## Transition from refractory to quiescent ##
        R_to_Q = np.logical_and(state[:, st-1] == -1, np.random.uniform(low = 0, high = 1, size = n) < r2)
    
        state[:, st] = state[:, st-1]
        state[:, st][E_to_R] = -1    
        state[:, st][R_to_Q] = 0
        state[:, st][Q_to_E] = 1

    return state



##############
## Analysis ##
##############
def compute_KOP(signal, fs, W = None):

    # Computes global KOP
    KOP_global = np.abs(np.mean(sig.hilbert(signal, axis = -1)/np.abs(sig.hilbert(signal, axis = -1)), axis = 0))

    # Computes local KOP if connectivity matrix is given
    if W is None:
        KOP_local = np.nan * np.ones(signal.shape)
    else:
        KOP_local = np.abs((W @ (sig.hilbert(signal, axis = -1)/np.abs(sig.hilbert(signal, axis = -1))))/W.sum(0)[:, None])

    return KOP_global, KOP_local




def DFA_analysis(signal, window_sizes,
                 fs = 100, overlap = 0):
    
    # De-means signal
    signal -= np.nanmean(signal, axis = -1)[:, None]
    
    # Computes cumulative signal
    signal_prof = np.cumsum(signal, axis = -1)

    Ft_windows = np.zeros((signal_prof.shape[0], len(window_sizes)))
    window_counts = np.zeros(len(window_sizes))

    for w, window_size in enumerate(window_sizes):

        if signal_prof.shape[-1] > (window_size*fs):
                
            window_size = int(window_size*fs) # Gets window sizes in samples
            w_step = int((1 - overlap) * window_size)

            # Iterates over time-windows and calculates nFt
            for t in range(int(np.floor(signal_prof.shape[-1]/w_step))):
                # Defines signal window
                sig_window = signal_prof[:, int(t*w_step):(int(t*w_step)+window_size)]
                
                # Detrends signals in current window
                sig_window = sig.detrend(sig_window, axis = -1)

                # Computes fluctuation within current window
                Ft_windows[:, w] += np.std(sig_window, axis = -1)
                window_counts[w] += 1

    Ft_windows = Ft_windows / window_counts[None, :]

    # Iterates over time-series and estimates DFA exponents
    DFA = np.zeros(signal.shape[0])
    for n in range(Ft_windows.shape[0]):

        mask = (window_counts > 0)

        reg = LinearRegression().fit(np.log10(window_sizes)[mask][:, None], 
                                     np.log10(Ft_windows[n, :][mask])[:, None])
        
        DFA[n] = reg.coef_[0][0]
    
    return DFA, Ft_windows







def extract_avalanches(state):

    state = (state == 1).astype(int)
    av_ts = np.sum(state, axis = 0)
    avalanches = []

    starts = np.where(np.diff((av_ts > 0).astype(int)) == 1)[0]
    ends = np.where(np.diff((av_ts > 0).astype(int)) == -1)[0]

    if not (len(starts) == 0 and len(ends) == 0):
        if len(starts) > len(ends):
            ends = np.insert(ends, len(ends), len(av_ts))
        elif len(ends) > len(starts):
            starts = np.insert(starts, 0, 0)
        elif starts[0] > ends[0]:
            starts = np.insert(starts, 0, 0)
            ends = np.insert(ends, len(ends), len(av_ts))

        for s, e in zip(starts, ends):
            
            avalanches.append(state[:, int(s):int(e)])

    return avalanches


def extract_unique_patterns(state):

    avalanches = extract_avalanches(state)

    if len(avalanches) > 0:
        patterns = np.array([(np.sum(a, axis = -1)>0).astype(int) for a in avalanches]).T
        data = patterns.astype(np.byte)
        s = pd.Series(data.T.tolist())
        indices = np.where(~s.duplicated(keep='first'))
        unique_patterns = data[:, indices][:, 0, :]
    else:
        unique_patterns = np.array([])

    return unique_patterns


def get_repertoire_metrics(state):

    unique_patterns = extract_unique_patterns(state)
    repertoire_size = unique_patterns.shape[-1]

    distances = []

    for i in range(repertoire_size):
        p1 = unique_patterns[:, i]
        for j in range(i+1, repertoire_size):
            p2 = unique_patterns[:, j]
            distances.append(np.mean(1 - (p1 == p2).astype(int))) # computes Hamming distance

    # nanmedian([]) returns NaN; return 0.0 instead (no diversity when <= 1 pattern)
    repertoire_diversity = float(np.nanmedian(distances)) if distances else 0.0

    return repertoire_size, repertoire_diversity


def compute_branching_ratio(state):

    branching_aux = (np.sum(state[:, 1:]==1, axis = 0)/np.sum(state[:, :-1]==1, axis = 0))
    branching_aux[np.isinf(branching_aux)] = np.nan

    return np.nanmean(branching_aux)



def bin_state(state, bin_size):
    
    if bin_size == 1:
        state_binned = state.copy()
    else:
        state = (state == 1).astype(int)
        state = state[:, :int(bin_size * (state.shape[-1]//bin_size))]
        state_binned = (np.nanmean(state.reshape(state.shape[0], -1, bin_size), axis = -1) > 0).astype(int)
    return state_binned





### NEW (ADRIAN + CLAUDE) 

# FORGOT WEIGHTS! 
# def repertoire(A, T: float) -> tuple[float, float]:
#     """Returns (repertoire_size, repertoire_diversity) for a given adjacency matrix and threshold T."""

#     # Spectral normalization. I guess this is NOT needed here, but for evaluation - so I dediced to add it here, too, just in case? 
#     eval, _ = np.linalg.eig(A)  
#     A /= np.nanmax(np.abs(eval))

#     state = simulate_cellular_automaton(W=A, sim_time=1000, T=T)
#     state_binned = bin_state(state, bin_size=1)
#     rs, rd = get_repertoire_metrics(state_binned)
#     return {"size": rs, "diversity": rd}


# FORGOT WEIGHTS!
# def repertoire_sweep(A, T_vec=np.logspace(-2, 0, 50)) -> dict:
#     """
#     Sweeps over T values and returns T, repertoire size, and repertoire diversity.
#     Returns the results for all T values, plus the T and metrics at peak repertoire size.
#     """
#     # Spectral normalisation
#     eval, _ = np.linalg.eig(A)
#     A = A / np.nanmax(np.abs(eval))

#     sizes, diversities = [], []

#     for T in T_vec:
#         state = simulate_cellular_automaton(W=A, sim_time=1000, T=T)
#         state_binned = bin_state(state, bin_size=1)
#         rs, rd = get_repertoire_metrics(state_binned)
#         sizes.append(rs)
#         diversities.append(rd)

#     sizes = np.array(sizes)
#     diversities = np.array(diversities)

#     # Find critical T as the peak repertoire size
#     peak_idx = np.argmax(sizes)

#     return {
#         "T_vec":          T_vec,
#         "sizes":          sizes,
#         "diversities":    diversities,
#         "T_critical":     T_vec[peak_idx],
#         "size_critical":  sizes[peak_idx],
#         "diversity_critical": diversities[peak_idx],
#     }
    
    


def repertoire_sweep_weighted_by_distances(A, D, T_vec=np.logspace(-2, 0, 50)) -> dict:
    """
    Sweeps over T values and returns T, repertoire size, and repertoire diversity.
    Returns the results for all T values, plus the T and metrics at peak repertoire size.
    """

    # Distance weighting
    A = A / (D + 1e-8)  # Avoid division by zero

    # Spectral normalisation — use eigh (symmetric solver) since A/(D+eps) is symmetric.
    # eigh is guaranteed to converge and returns real eigenvalues; eig (general solver) can
    # fail to converge on some matrices even when they are symmetric.
    try:
        evals = np.linalg.eigh(A)[0]  # ascending real eigenvalues
        rho = np.max(np.abs(evals))
    except np.linalg.LinAlgError:
        rho = 0.0

    if rho < 1e-10:
        # Zero or near-zero matrix — no meaningful dynamics possible
        return {
            "T_vec":              T_vec.tolist(),
            "sizes":              [0] * len(T_vec),
            "diversities":        [0.0] * len(T_vec),
            "T_critical":         np.nan,
            "size_critical":      0,
            "diversity_critical": 0.0,
        }

    A = A / rho

    sizes, diversities = [], []

    for T in T_vec:
        state = simulate_cellular_automaton(W=A, sim_time=1000, T=T)
        state_binned = bin_state(state, bin_size=1)
        rs, rd = get_repertoire_metrics(state_binned)
        sizes.append(rs)
        diversities.append(rd)

    sizes = np.array(sizes)
    diversities = np.array(diversities)

    # Find critical T as the peak repertoire size
    peak_idx = np.argmax(sizes)

    return {
        "T_vec":          T_vec,
        "sizes":          sizes,
        "diversities":    diversities,
        "T_critical":     T_vec[peak_idx],
        "size_critical":  sizes[peak_idx],
        "diversity_critical": diversities[peak_idx],
    }