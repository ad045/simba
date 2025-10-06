
from dataclasses import dataclass

# Import project constants
from .constants import (
    DEFAULT_SPECTRAL_RADIUS, DEFAULT_INPUT_LENGTH, DEFAULT_INPUT_SCALING, DEFAULT_N_RUNS,
    DEFAULT_N_LAGS, DEFAULT_TEST_LENGTH, DEFAULT_N_TRANSIENT, DEFAULT_LEAK_RATE, DEFAULT_BIAS,
    DEFAULT_REGULARIZATION_METHOD
)


@dataclass
class ESNConfig:
    """ESN hyperparameter configuration."""
    spectral_radius: float = DEFAULT_SPECTRAL_RADIUS
    input_length: int = DEFAULT_INPUT_LENGTH
    input_scaling: float = DEFAULT_INPUT_SCALING
    regularization_method: str = DEFAULT_REGULARIZATION_METHOD
    n_runs: int = DEFAULT_N_RUNS
    n_lags: int = DEFAULT_N_LAGS
    test_len: int = DEFAULT_TEST_LENGTH
    n_transient: int = DEFAULT_N_TRANSIENT
    leak_rate: float = DEFAULT_LEAK_RATE
    bias: float = DEFAULT_BIAS
