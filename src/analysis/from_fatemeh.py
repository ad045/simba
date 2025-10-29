import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from echoes import ESNRegressor
from echoes.plotting import set_mystyle
from echoes.reservoir._leaky_numba import harvest_states
from numba import njit
from scipy.spatial import distance_matrix
from scipy import linalg
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from numpy import linalg


def compute_eigenvalue_gap(matrix):
    """
    Compute the gap between the largest and second largest eigenvalues (in magnitude)
    """
    # Compute eigenvalues
    eigenvalues = linalg.eigvals(matrix)
    # Sort eigenvalues by magnitude in descending order
    sorted_eigenvalues = sorted(abs(eigenvalues), reverse=True)

    # If matrix has at least 2 eigenvalues, compute gap
    if len(sorted_eigenvalues) >= 2:
        gap = sorted_eigenvalues[0] - sorted_eigenvalues[1]
    else:
        gap = sorted_eigenvalues[0]  # If only one eigenvalue exists

    return gap

def departure_from_normality(M):
    M = np.array(M)
    # Calculate the Frobenius norm of M
    norm_F = linalg.norm(M, 'fro')
    # Compute the eigenvalues of M
    eigenvalues = np.linalg.eigvals(M)
    # Compute the sum of the squares of the eigenvalues
    sum_squares_eigenvalues = np.sum(np.abs(eigenvalues)**2)
    # Compute the departure from normality
    d_F = np.sqrt(norm_F**2 - sum_squares_eigenvalues)
    return d_F/norm_F


def generate_iid_signal(n, distribution='gaussian', **params):

    if distribution.lower() == 'gaussian':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        signal = np.random.normal(mu, sigma, n)

    elif distribution.lower() == 'uniform':
        low = params.get('low', -1)
        high = params.get('high', 1)
        signal = np.random.uniform(low, high, n)

    elif distribution.lower() == 'bernoulli':
        p = params.get('p', 0.5)
        signal = np.random.binomial(1, p, n)

    elif distribution.lower() == 'exponential':
        scale = params.get('scale', 1.0)
        signal = np.random.exponential(scale, n)

    else:
        raise ValueError(f"Distribution '{distribution}' not supported")

    return signal

def generate_esn_open(W, alpha=0.96):
    ESN = ESNRegressor(
        n_reservoir=W.shape[0],
        spectral_radius =alpha, #0.08 * utils.find_spectral_radius(adj_mat)
        leak_rate=1.0,
        noise=0.00001,
        n_transient=100,
        W_in=np.random.uniform(low=-0.05, high=0.05, size=(W.shape[0], 1)),
        bias=0.00,
        regression_method='ridge',
        ridge_alpha=1e-10,
        random_state=283,
        fit_only_states=True,
        W=W.astype(float),
        store_states_train=True
        )
    return ESN


# I tend to use 'relative' method. you can simply go for 'absolute' and Set threshold_value by once visulaizing singular_values
#num_inputs equals number of nodes in the reservoir

def compute_KR(network, num_inputs=150, threshold_method='relative', threshold_value=1e-2, alpha=0.96):
    def signal_generator():
        return generate_iid_signal(150, 'gaussian', mu=0, sigma=1) # n is the length of the signal

    esn = generate_esn_open(network, alpha)

    # Create num_inputs input streams
    input_signals = [signal_generator() for _ in range(num_inputs)]

    # Collect final reservoir states for each input stream
    final_states = []
    for signal in input_signals:
        # Get reservoir states for this input
        reservoir_states = esn.fit(signal.reshape(-1,1), signal.reshape(-1,1)).states_train_

        # Extract the state at the final time step
        final_state = reservoir_states[-1, :]  # Assuming reservoir_states has shape (time_steps, num_nodes)

        # Store this final state
        final_states.append(final_state)

    # Create the state matrix (M×N where M=num_nodes, N=num_inputs)
    state_matrix = np.column_stack(final_states)

    # Calculate singular values
    singular_values = linalg.svd(state_matrix, compute_uv=False)

    # Determine threshold based on method
    if threshold_method == 'relative':
        threshold = singular_values[0] * threshold_value
    elif threshold_method == 'variance':
        explained_variance = np.cumsum(singular_values ** 2) / np.sum(singular_values ** 2)
        kernel_rank = np.sum(explained_variance <= threshold_value) + 1

        return kernel_rank
    else:  # absolute
        threshold = threshold_value

    # Count singular values above threshold or compute sum of the values above threshold
    # kernel_rank = np.sum(singular_values > threshold)
    kernel_rank = np.count_nonzero(singular_values > threshold)

    return kernel_rank