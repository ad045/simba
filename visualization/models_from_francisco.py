
import numpy as np
import scipy.signal as sig
from numba import njit
from sklearn.linear_model import LinearRegression


######################
## Evaluate network ##
######################
# Simulates and extracts relevant information

def evaluate_network(W, sim_time = 500, dt = 1e-2, K = 0.03, freq = 0.05, a = -0.1, noise = 0.001, estimate_DFA = False):

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